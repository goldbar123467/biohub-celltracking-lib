from __future__ import annotations

from biohub_ct.data.schema import Node
from biohub_ct.linking.costs import physical_distance


def adjacent_frame_candidates(
    sources: list[Node],
    targets: list[Node],
    *,
    scale: tuple[float, float, float],
    max_distance: float,
) -> list[tuple[int, int, float]]:
    out = []
    for source in sources:
        for target in targets:
            dist = physical_distance(source, target, scale)
            if dist <= max_distance:
                out.append((source.node_id, target.node_id, dist))
    return sorted(out, key=lambda row: (row[2], row[0], row[1]))

