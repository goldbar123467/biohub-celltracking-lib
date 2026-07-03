from __future__ import annotations

import numpy as np


def crop_around(frame: np.ndarray, center: tuple[int, int, int], radius: int) -> np.ndarray:
    z, y, x = center
    return frame[
        max(0, z - radius) : min(frame.shape[0], z + radius + 1),
        max(0, y - radius) : min(frame.shape[1], y + radius + 1),
        max(0, x - radius) : min(frame.shape[2], x + radius + 1),
    ]

