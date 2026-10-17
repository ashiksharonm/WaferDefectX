"""Synthetic wafer-map generator with realistic noise and variable defect contrast.

This is a harder variant of ``python/data_generator.py``: Gaussian sensor noise, a slow
illumination gradient, soft-edged defects and a per-wafer ``contrast`` knob, so faint
defects overlap with background texture. It is still synthetic; accuracy on it says
nothing about real fab data.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from wafer_inspect import CLASSES

DEFAULT_SIZE = 512
MIN_CONTRAST = 0.03  # fraction of full defect contrast; defects below ~0.1 hide in the noise


def make_wafer(
    rng: np.random.Generator,
    label: str,
    contrast: float = 1.0,
    size: int = DEFAULT_SIZE,
) -> np.ndarray:
    """Return a uint8 grayscale wafer image with at most one defect of type ``label``."""
    if label not in CLASSES:
        raise ValueError(f"unknown label {label!r}")
    scale = size / 800.0
    center = (size // 2, size // 2)
    radius = int(size * 0.45)

    img = np.zeros((size, size), np.float32)
    cv2.circle(img, center, radius, 100.0, -1)

    # slow illumination variation + Gaussian sensor noise
    gradient = rng.normal(0, 1, (8, 8)).astype(np.float32)
    img += cv2.resize(gradient, (size, size), interpolation=cv2.INTER_CUBIC) * 4.0
    img += rng.normal(0, 7.0, img.shape).astype(np.float32)

    layer = np.zeros_like(img)
    amplitude = 100.0 * contrast * rng.uniform(0.8, 1.2)
    if label != "good":
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, radius * 0.8)
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        if label == "particle":
            pr = max(1.0, rng.integers(2, 7) * scale * 1.5)
            cv2.circle(layer, (int(x), int(y)), int(round(pr)), amplitude, -1)
        else:
            length = rng.uniform(20, 100) * scale * 1.5
            a = rng.uniform(0, 2 * np.pi)
            end = (int(x + length * np.cos(a)), int(y + length * np.sin(a)))
            cv2.line(layer, (int(x), int(y)), end, amplitude, thickness=int(rng.integers(1, 4)))
        layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    img += layer

    disk = np.zeros((size, size), np.uint8)
    cv2.circle(disk, center, radius, 1, -1)
    img *= disk
    return np.clip(img, 0, 255).astype(np.uint8)


def make_dataset(
    n: int,
    seed: int,
    size: int = DEFAULT_SIZE,
    min_contrast: float = MIN_CONTRAST,
    labels: Optional[Tuple[str, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balanced-ish random dataset. Returns (images[N,H,W], label_idx[N], contrast[N])."""
    rng = np.random.default_rng(seed)
    classes = labels or CLASSES
    idx = rng.integers(0, len(classes), n)
    contrast = rng.uniform(min_contrast, 1.0, n)
    images = np.stack([make_wafer(rng, classes[i], c, size) for i, c in zip(idx, contrast)])
    return images, np.array([CLASSES.index(classes[i]) for i in idx]), contrast
