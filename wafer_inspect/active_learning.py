"""How many labels does the model need? Uncertainty sampling vs random sampling.

Simulates the review loop on a synthetic pool: start from a few labelled wafers, repeatedly
"send" the batch the model is least sure about (or a random batch) to the labeller, retrain,
and track held-out accuracy. Reports labels needed to reach a target accuracy.

    python -m wafer_inspect.active_learning --target 0.9 --out reports/active_learning.json

Runs on CPU in about a minute with the defaults. Synthetic data only.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np

from wafer_inspect.features import extract_batch
from wafer_inspect.model import fit_rf
from wafer_inspect.synth import make_dataset

STRATEGIES = ("random", "uncertainty")


def initial_labelled(y: np.ndarray, rng: np.random.Generator, per_class: int = 2) -> List[int]:
    chosen: List[int] = []
    for c in np.unique(y):
        chosen += rng.choice(np.flatnonzero(y == c), per_class, replace=False).tolist()
    return chosen


def run_loop(
    X: np.ndarray,
    y: np.ndarray,
    Xt: np.ndarray,
    yt: np.ndarray,
    strategy: str,
    seed: int,
    batch: int,
    max_labels: int,
) -> Dict[int, float]:
    """Return {n_labels: held-out accuracy} for one run."""
    rng = np.random.default_rng(seed)
    labelled = initial_labelled(y, rng)
    curve: Dict[int, float] = {}
    while True:
        model = fit_rf(X[labelled], y[labelled], seed=seed)
        curve[len(labelled)] = float((model.predict(Xt) == yt).mean())
        if len(labelled) >= max_labels:
            return curve
        rest = np.setdiff1d(np.arange(len(y)), labelled)
        if strategy == "uncertainty":
            proba = np.zeros((len(rest), 3))
            proba[:, model.classes_.astype(int)] = model.predict_proba(X[rest])
            picks = rest[np.argsort(proba.max(axis=1))[:batch]]  # least confident first
        else:
            picks = rng.choice(rest, batch, replace=False)
        labelled += picks.tolist()


def labels_needed(curve: Dict[int, float], target: float) -> Optional[int]:
    return next((n for n, acc in sorted(curve.items()) if acc >= target), None)


def run(pool: int, test: int, seeds: int, batch: int, max_labels: int, target: float) -> dict:
    Xi, y, _ = make_dataset(pool, seed=11)
    Xti, yt, _ = make_dataset(test, seed=12)
    X, Xt = extract_batch(Xi), extract_batch(Xti)
    ceiling = float((fit_rf(X, y).predict(Xt) == yt).mean())

    result: dict = {
        "data": "synthetic (wafer_inspect.synth), not real fab data",
        "pool": pool,
        "test": test,
        "target_accuracy": target,
        "ceiling_accuracy_all_pool_labelled": ceiling,
        "seeds": seeds,
        "batch": batch,
        "strategies": {},
    }
    for strategy in STRATEGIES:
        curves = [run_loop(X, y, Xt, yt, strategy, s, batch, max_labels) for s in range(seeds)]
        needed = [labels_needed(c, target) for c in curves]
        reached = [n for n in needed if n is not None]
        sizes = sorted(curves[0])
        result["strategies"][strategy] = {
            "labels_needed_per_seed": needed,
            "median_labels_needed": float(np.median(reached)) if reached else None,
            "seeds_reaching_target": len(reached),
            "mean_accuracy_at": {
                str(n): round(float(np.mean([c[n] for c in curves])), 4)
                for n in sizes[:: max(1, len(sizes) // 8)]
            },
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=float, default=0.90)
    p.add_argument("--pool", type=int, default=800)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--max-labels", type=int, default=200)
    p.add_argument("--out", help="write JSON here")
    a = p.parse_args()
    res = run(a.pool, a.test, a.seeds, a.batch, a.max_labels, a.target)
    print(json.dumps(res, indent=2))
    if a.out:
        from pathlib import Path

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
