"""Calibrated wafer classifier with uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from wafer_inspect import CLASSES
from wafer_inspect.features import extract_features


def normalized_entropy(proba: np.ndarray) -> np.ndarray:
    """Predictive entropy scaled to [0, 1] (1 = uniform over classes)."""
    p = np.clip(proba, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(proba.shape[1])


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def fit_rf(X: np.ndarray, y: np.ndarray, seed: int = 0, n_estimators: int = 100) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1).fit(X, y)


def fit_calibrated(X: np.ndarray, y: np.ndarray, seed: int = 0) -> CalibratedClassifierCV:
    """Random forest with sigmoid (Platt) calibration fitted by 3-fold cross-validation."""
    base = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X, y)


@dataclass
class Prediction:
    label: str
    confidence: float
    uncertainty: float
    probabilities: Dict[str, float]


class WaferModel:
    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    @classmethod
    def load(cls, path: str) -> "WaferModel":
        return cls(joblib.load(path))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(features)

    def predict_image(self, gray: np.ndarray) -> Prediction:
        proba = self.predict_proba(extract_features(gray)[None, :])
        return self.to_predictions(proba)[0]

    @staticmethod
    def to_predictions(proba: np.ndarray) -> List[Prediction]:
        unc = normalized_entropy(proba)
        return [
            Prediction(
                label=CLASSES[int(p.argmax())],
                confidence=float(p.max()),
                uncertainty=float(u),
                probabilities={c: float(v) for c, v in zip(CLASSES, p)},
            )
            for p, u in zip(proba, unc)
        ]
