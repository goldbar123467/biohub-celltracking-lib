from __future__ import annotations

import numpy as np

from biohub_ct.spatial import cell, neighbors, validate_geometry


def anisotropic_nms(
    coords: list[tuple[int, int, int]],
    scores: list[float],
    *,
    scale: tuple[float, float, float],
    radius_um: float,
) -> list[int]:
    validate_geometry(scale, radius_um)
    if len(coords) != len(scores):
        raise ValueError("Coordinates and scores must have equal length")
    order = sorted(range(len(coords)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    bins = {}
    scale_arr = np.asarray(scale, dtype=float)
    for idx in order:
        point = np.asarray(coords[idx], dtype=float) * scale_arr
        key = cell(point, radius_um)
        if all(
            float(np.linalg.norm(point - other)) > radius_um
            for neighbor in neighbors(key)
            for other in bins.get(neighbor, ())
        ):
            kept.append(idx)
            bins.setdefault(key, []).append(point)
    return kept
