from __future__ import annotations

import numpy as np


def intensity_summary(crop: np.ndarray) -> dict[str, float]:
    arr = np.asarray(crop, dtype=float)
    return {"mean": float(arr.mean()), "max": float(arr.max()), "std": float(arr.std())}

