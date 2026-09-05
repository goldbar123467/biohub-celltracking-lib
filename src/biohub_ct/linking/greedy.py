from __future__ import annotations

import time

from biohub_ct.data.schema import Edge, Node
from biohub_ct.linking.costs import physical_distance
from biohub_ct.spatial import cell, neighbors, validate_geometry


def greedy_link(
    nodes: list[Node],
    *,
    scale: tuple[float, float, float],
    max_distance: float,
    allow_divisions: bool = False,
    deadline_at: float | None = None,
) -> list[Edge]:
    validate_geometry(scale, max_distance)
    by_t: dict[int, list[Node]] = {}
    for node in nodes:
        by_t.setdefault(node.t, []).append(node)

    edges: list[Edge] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    source_counts: dict[int, int] = {}
    for t in sorted(by_t):
        if deadline_at is not None and time.monotonic() >= deadline_at:
            raise TimeoutError("Inference budget exhausted during linking")
        if t + 1 not in by_t:
            continue
        candidates = []
        bins = {}
        for target in by_t[t + 1]:
            bins.setdefault(cell(target.physical_coord(scale), max_distance), []).append(target)
        for source in by_t[t]:
            nearby = (
                target
                for key in neighbors(cell(source.physical_coord(scale), max_distance))
                for target in bins.get(key, ())
            )
            for target in nearby:
                dist = physical_distance(source, target, scale)
                if dist <= max_distance:
                    candidates.append((dist, source.node_id, target.node_id))
        for _, source_id, target_id in sorted(candidates):
            if target_id in used_targets:
                continue
            if allow_divisions:
                if source_counts.get(source_id, 0) >= 2:
                    continue
            elif source_id in used_sources:
                continue
            edges.append(Edge(source_id, target_id))
            used_targets.add(target_id)
            used_sources.add(source_id)
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
    return edges
