"""Regenerate the demo wafer maps in samples/ (synthetic, deterministic).

With ``--model`` it also saves two faint wafers the model is unsure about (confidence < 0.7)
so the demo shows the review queue. Which wafers qualify depends on the trained model.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from wafer_inspect import CLASSES
from wafer_inspect.synth import make_wafer

LEVELS = {"clear": 0.9, "medium": 0.3, "faint": 0.07}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples")
    parser.add_argument("--model", help="trained model; adds low-confidence 'ambiguous_*' samples")
    args = parser.parse_args()

    Path(args.out).mkdir(exist_ok=True)
    rng = np.random.default_rng(2024)
    for label in CLASSES:
        for name, contrast in LEVELS.items():
            cv2.imwrite(f"{args.out}/{label}_{name}.png", make_wafer(rng, label, contrast))

    if args.model:
        from wafer_inspect.model import WaferModel

        model = WaferModel.load(args.model)
        rng = np.random.default_rng(77)
        found = 0
        while found < 2:
            label = CLASSES[int(rng.integers(0, 3))]
            img = make_wafer(rng, label, float(rng.uniform(0.04, 0.12)))
            if model.predict_image(img).confidence < 0.7:
                found += 1
                cv2.imwrite(f"{args.out}/ambiguous_{found}.png", img)


if __name__ == "__main__":
    main()
