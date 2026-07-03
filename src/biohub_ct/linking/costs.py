from __future__ import annotations

from biohub_ct.data.schema import Node
from biohub_ct.metrics.edge import scaled_distance


def physical_distance(
    source: Node,
    target: Node,
    scale: tuple[float, float, float],
) -> float:
    return scaled_distance(source, target, scale)

