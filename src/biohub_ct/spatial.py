"""Exact bounded-neighborhood lookup, with no optional dependencies."""

import math
from itertools import product

OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def cell(point, radius):
    return tuple(math.floor(float(x) / radius) for x in point)


def neighbors(key):
    return (tuple(k + d for k, d in zip(key, offset)) for offset in OFFSETS)


def validate_geometry(scale, radius):
    if len(scale) != 3 or not all(math.isfinite(s) and s > 0 for s in scale):
        raise ValueError("Scale must contain three positive finite values")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Radius must be positive and finite")
