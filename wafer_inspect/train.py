"""Train + calibrate the wafer model on synthetic data and write metrics.

python -m wafer_inspect.train --out models
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from wafer_inspect.features import extract_batch
from wafer_inspect.model import (
    WaferModel,
    expected_calibration_error,
    fit_calibrated,
    fit_rf,
)
from wafer_inspect.synth import make_dataset


def run(out_dir: str, n_train: int = 600, n_test: int = 400, seed: int = 0) -> dict:
    Xi, yi, _ = make_dataset(n_train, seed=seed)
    Xt_img, yt, _ = make_dataset(n_test, seed=seed + 1000)
    X, Xt = extract_batch(Xi), extract_batch(Xt_img)

    raw = fit_rf(X, yi, seed)
    cal = fit_calibrated(X, yi, seed)
    p_raw, p_cal = raw.predict_proba(Xt), cal.predict_proba(Xt)
    metrics = {
        "data": "synthetic (wafer_inspect.synth), not real fab data",
        "n_train": n_train,
        "n_test": n_test,
        "seed": seed,
        "accuracy_raw_rf": float((p_raw.argmax(1) == yt).mean()),
        "accuracy_calibrated": float((p_cal.argmax(1) == yt).mean()),
        "ece_raw_rf": expected_calibration_error(p_raw, yt),
        "ece_calibrated": expected_calibration_error(p_cal, yt),
        "machine": f"{platform.machine()} {platform.platform()}",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    WaferModel(cal).save(str(out / "wafer_model.joblib"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models")
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.n_train, args.n_test, args.seed), indent=2))


if __name__ == "__main__":
    main()
