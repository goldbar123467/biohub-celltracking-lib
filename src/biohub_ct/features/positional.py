from __future__ import annotations

from biohub_ct.data.schema import Node


def normalized_position(node: Node, shape: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    t, z, y, x = shape
    return (
        node.t / max(1, t - 1),
        node.z / max(1, z - 1),
        node.y / max(1, y - 1),
        node.x / max(1, x - 1),
    )

