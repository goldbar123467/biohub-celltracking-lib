from __future__ import annotations

import numpy as np


def anisotropic_nms(
    coords: list[tuple[int, int, int]],
    scores: list[float],
    *,
    scale: tuple[float, float, float],
    radius_um: float,
) -> list[int]:
    order = sorted(range(len(coords)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    kept_phys: list[np.ndarray] = []
    scale_arr = np.asarray(scale, dtype=float)
    for idx in order:
        point = np.asarray(coords[idx], dtype=float) * scale_arr
        if all(float(np.linalg.norm(point - other)) > radius_um for other in kept_phys):
            kept.append(idx)
            kept_phys.append(point)
    return kept

