from __future__ import annotations

import numpy as np


def refine_centroid_moment(
    frame: np.ndarray,
    coord: tuple[int, int, int],
    *,
    radius: int = 2,
) -> tuple[int, int, int]:
    z, y, x = coord
    z0, z1 = max(0, z - radius), min(frame.shape[0], z + radius + 1)
    y0, y1 = max(0, y - radius), min(frame.shape[1], y + radius + 1)
    x0, x1 = max(0, x - radius), min(frame.shape[2], x + radius + 1)
    crop = np.asarray(frame[z0:z1, y0:y1, x0:x1], dtype=np.float64)
    weights = crop - crop.min()
    if weights.sum() <= 0:
        return coord
    zz, yy, xx = np.indices(crop.shape)
    rz = int(round(float((zz * weights).sum() / weights.sum()) + z0))
    ry = int(round(float((yy * weights).sum() / weights.sum()) + y0))
    rx = int(round(float((xx * weights).sum() / weights.sum()) + x0))
    return (rz, ry, rx)

