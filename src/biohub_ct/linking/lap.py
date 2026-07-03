from __future__ import annotations

from biohub_ct.data.schema import Edge, Node
from biohub_ct.linking.greedy import greedy_link


def lap_link(
    nodes: list[Node],
    *,
    scale: tuple[float, float, float],
    max_distance: float,
) -> list[Edge]:
    """Dependency-light placeholder using greedy assignment.

    SciPy's Hungarian implementation can be swapped in without changing callers.
    """
    return greedy_link(nodes, scale=scale, max_distance=max_distance)

