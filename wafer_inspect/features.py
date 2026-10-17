"""Hand-crafted features: noise-normalised difference-of-Gaussians blob statistics."""

from __future__ import annotations

import cv2
import numpy as np

FEATURE_NAMES = (
    "z_max",
    "z_top20_mean",
    "frac_z2",
    "frac_z3",
    "frac_z4",
    "n_comp",
    "largest_area",
    "elongation",
    "extent",
    "long_axis",
    "skew",
)


def extract_features(gray: np.ndarray) -> np.ndarray:
    """Feature vector for one grayscale wafer image (any size, uint8)."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    f = gray.astype(np.float32)
    dog = cv2.GaussianBlur(f, (0, 0), 1.2) - cv2.GaussianBlur(f, (0, 0), 10.0)

    disk = np.zeros((h, w), np.uint8)
    cv2.circle(disk, (w // 2, h // 2), int(min(h, w) * 0.45 * 0.92), 1, -1)
    inside = dog[disk > 0]
    med = float(np.median(inside))
    sigma = float(1.4826 * np.median(np.abs(inside - med))) + 1e-6
    z = np.where(disk > 0, (dog - med) / sigma, 0.0)
    zin = z[disk > 0]

    top = np.sort(zin)[-20:]
    feats = [
        float(zin.max()),
        float(top.mean()),
        float((zin > 2).mean()),
        float((zin > 3).mean()),
        float((zin > 4).mean()),
    ]

    mask = (z > 3.0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([], dtype=int)
    keep = np.flatnonzero(areas >= 3)
    if len(keep) == 0:
        feats += [0.0, 0.0, 1.0, 0.0, 0.0]
    else:
        k = int(keep[np.argmax(areas[keep])]) + 1
        ys, xs = np.nonzero(labels == k)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1.0)
        feats += [
            float(len(keep)),
            float(areas[k - 1]),
            float(long_side / short_side),
            float(areas[k - 1] / max(rw * rh, 1.0)),
            float(long_side),
        ]
    centred = zin - zin.mean()
    feats.append(float((centred**3).mean() / (centred.std() ** 3 + 1e-9)))
    return np.asarray(feats, dtype=np.float64)


def extract_batch(images: np.ndarray) -> np.ndarray:
    return np.stack([extract_features(im) for im in images])
