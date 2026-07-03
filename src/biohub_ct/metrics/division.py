from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from biohub_ct.config import DEFAULT_SCALE, MAX_MATCH_DISTANCE_MICRONS
from biohub_ct.data.schema import Graph
from biohub_ct.metrics.edge import match_nodes


@dataclass(frozen=True)
class DivisionEvaluation:
    tp: int
    fn: int
    fp: int
    division_jaccard: float


def evaluate_divisions(
    pred: Graph,
    gt: Graph,
    *,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_MATCH_DISTANCE_MICRONS,
) -> DivisionEvaluation:
    scores = score_divisions(pred, gt, scale=scale, max_distance=max_distance)
    tp = sum(scores.values())
    fn = len(scores) - tp
    matched_pred_divs = count_matched_pred_divisions(
        pred, gt, scale=scale, max_distance=max_distance
    )
    fp = max(0, matched_pred_divs - tp)
    denom = tp + fp + fn
    return DivisionEvaluation(tp=tp, fn=fn, fp=fp, division_jaccard=tp / denom if denom else float("nan"))


def score_divisions(
    pred: Graph,
    gt: Graph,
    *,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_MATCH_DISTANCE_MICRONS,
) -> dict[int, int]:
    gt_divisions = _extract_divisions(gt)
    pred_div_nodes = set(pred.dividing_nodes())
    candidates: dict[int, set[int]] = {}
    for div_node, gt_div in gt_divisions.items():
        matches = match_nodes(pred, gt_div, scale=scale, max_distance=max_distance)
        matched_pred_ids = list(matches.pred_to_gt)
        components = _weakly_connected_components(pred, matched_pred_ids)
        div_candidates: set[int] = set()
        for matched_subset, visited in components:
            if _has_stage_coverage(matched_subset, matches.pred_to_gt, pred, gt_div, div_node):
                div_candidates |= visited & pred_div_nodes
        candidates[div_node] = div_candidates
    pairing = _bipartite_max_matching(list(candidates), candidates)
    return {div_node: int(div_node in pairing) for div_node in candidates}


def count_matched_pred_divisions(
    pred: Graph,
    gt: Graph,
    *,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_MATCH_DISTANCE_MICRONS,
) -> int:
    matches = match_nodes(pred, gt, scale=scale, max_distance=max_distance)
    count = 0
    for pred_id, gt_id in matches.pred_to_gt.items():
        if pred.out_degree(pred_id) >= 2 and gt.out_degree(gt_id) >= 1:
            count += 1
    return count


def _extract_divisions(graph: Graph) -> dict[int, Graph]:
    divisions: dict[int, Graph] = {}
    for div_node in graph.dividing_nodes():
        parents = graph.predecessors(div_node)
        children = graph.successors(div_node)
        grandchildren = [gc for child in children for gc in graph.successors(child)]
        divisions[div_node] = graph.subgraph([*parents, div_node, *children, *grandchildren])
    return divisions


def _has_stage_coverage(
    matched_subset: set[int],
    pred_to_gt: dict[int, int],
    pred: Graph,
    gt_div: Graph,
    divider_id: int,
) -> bool:
    if not matched_subset:
        return False
    counts_by_t: dict[int, int] = {}
    for node in gt_div.nodes_list:
        counts_by_t[node.t] = counts_by_t.get(node.t, 0) + 1
    one_node_times = {t for t, count in counts_by_t.items() if count == 1}
    has_one_node_stage = any(pred.node(pred_id).t in one_node_times for pred_id in matched_subset)
    if not has_one_node_stage:
        return False

    children = gt_div.successors(divider_id)
    if len(children) < 2:
        return False
    lineages = [_descendants(gt_div, child) for child in children]
    matched_gt = {pred_to_gt[pred_id] for pred_id in matched_subset if pred_id in pred_to_gt}
    return sum(1 for lineage in lineages if lineage & matched_gt) >= 2


def _descendants(graph: Graph, seed: int) -> set[int]:
    out = {seed}
    stack = [seed]
    while stack:
        current = stack.pop()
        for nxt in graph.successors(current):
            if nxt not in out:
                out.add(nxt)
                stack.append(nxt)
    return out


def _weakly_connected_components(
    graph: Graph,
    node_ids: list[int],
) -> list[tuple[set[int], set[int]]]:
    remaining = set(node_ids)
    components: list[tuple[set[int], set[int]]] = []
    while remaining:
        seed = next(iter(remaining))
        visited = {seed}
        matched_subset = {seed}
        queue: deque[int] = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor in [*graph.successors(current), *graph.predecessors(current)]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
                if neighbor in remaining:
                    matched_subset.add(neighbor)
        components.append((matched_subset, visited))
        remaining -= matched_subset
    return components


def _bipartite_max_matching(left: list[int], edges: dict[int, set[int]]) -> dict[int, int]:
    match_r: dict[int, int] = {}
    match_l: dict[int, int] = {}

    def augment(u: int, seen: set[int]) -> bool:
        for v in sorted(edges.get(u, ())):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or augment(match_r[v], seen):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    for u in left:
        augment(u, set())
    return match_l

