from __future__ import annotations

import numpy as np


def robust_normalize(frame: np.ndarray, q_low: float = 1.0, q_high: float = 99.5) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    lo, hi = np.percentile(arr, [q_low, q_high])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def detect_local_maxima_3d(
    frame: np.ndarray,
    *,
    threshold_abs: float = 0.5,
    max_peaks: int = 2000,
) -> list[tuple[int, int, int, float]]:
    norm = robust_normalize(frame)
    if norm.ndim != 3:
        raise ValueError(f"Expected 3D frame (Z,Y,X), got shape {norm.shape}")
    padded = np.pad(norm, 1, mode="edge")
    center = padded[1:-1, 1:-1, 1:-1]
    is_peak = center >= threshold_abs
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                is_peak &= center >= padded[1 + dz : 1 + dz + norm.shape[0], 1 + dy : 1 + dy + norm.shape[1], 1 + dx : 1 + dx + norm.shape[2]]
    coords = np.argwhere(is_peak)
    if coords.size == 0:
        return []
    scores = norm[tuple(coords.T)]
    order = np.argsort(scores)[::-1][:max_peaks]
    return [
        (int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]), float(scores[i]))
        for i in order
    ]

