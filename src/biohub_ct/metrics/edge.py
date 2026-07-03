from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isnan

import numpy as np

from biohub_ct.config import ADJUSTMENT_ALPHA, DEFAULT_SCALE, MAX_MATCH_DISTANCE_MICRONS
from biohub_ct.data.schema import Graph, Node


@dataclass(frozen=True)
class NodeMatches:
    pred_to_gt: dict[int, int]
    gt_to_pred: dict[int, int]


@dataclass(frozen=True)
class EdgeEvaluation:
    edge_tp: int
    edge_fp: int
    edge_fn: int
    num_pred_nodes: int
    edge_jaccard: float
    adjusted_edge_jaccard: float
    total_node_ratio: float


def scaled_distance(a: Node, b: Node, scale: tuple[float, float, float] = DEFAULT_SCALE) -> float:
    return float(np.linalg.norm(a.physical_coord(scale) - b.physical_coord(scale)))


def match_nodes(
    pred: Graph,
    gt: Graph,
    *,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_MATCH_DISTANCE_MICRONS,
) -> NodeMatches:
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    pred_by_t = pred.nodes_by_time()
    gt_by_t = gt.nodes_by_time()
    for t in sorted(set(pred_by_t) & set(gt_by_t)):
        pairs = _match_nodes_one_time(pred_by_t[t], gt_by_t[t], scale, max_distance)
        for pred_id, gt_id in pairs.items():
            pred_to_gt[pred_id] = gt_id
            gt_to_pred[gt_id] = pred_id
    return NodeMatches(pred_to_gt=pred_to_gt, gt_to_pred=gt_to_pred)


def _match_nodes_one_time(
    pred_nodes: list[Node],
    gt_nodes: list[Node],
    scale: tuple[float, float, float],
    max_distance: float,
) -> dict[int, int]:
    if not pred_nodes or not gt_nodes:
        return {}
    distances = [
        [scaled_distance(p, g, scale) for g in gt_nodes]
        for p in pred_nodes
    ]
    if len(pred_nodes) <= 20 and len(gt_nodes) <= 20:
        return _dp_assignment(pred_nodes, gt_nodes, distances, max_distance)
    return _greedy_assignment(pred_nodes, gt_nodes, distances, max_distance)


def _dp_assignment(
    pred_nodes: list[Node],
    gt_nodes: list[Node],
    distances: list[list[float]],
    max_distance: float,
) -> dict[int, int]:
    @lru_cache(maxsize=None)
    def best(i: int, used_mask: int) -> tuple[int, float]:
        if i == len(pred_nodes):
            return (0, 0.0)
        out = best(i + 1, used_mask)
        for j, dist in enumerate(distances[i]):
            if used_mask & (1 << j) or dist > max_distance:
                continue
            sub_count, sub_neg_dist = best(i + 1, used_mask | (1 << j))
            candidate = (sub_count + 1, sub_neg_dist - dist)
            if candidate > out:
                out = candidate
        return out

    assignment: dict[int, int] = {}
    i = 0
    used_mask = 0
    while i < len(pred_nodes):
        current = best(i, used_mask)
        if best(i + 1, used_mask) == current:
            i += 1
            continue
        for j, dist in enumerate(distances[i]):
            if used_mask & (1 << j) or dist > max_distance:
                continue
            sub_count, sub_neg_dist = best(i + 1, used_mask | (1 << j))
            if (sub_count + 1, sub_neg_dist - dist) == current:
                assignment[pred_nodes[i].node_id] = gt_nodes[j].node_id
                used_mask |= 1 << j
                break
        i += 1
    return assignment


def _greedy_assignment(
    pred_nodes: list[Node],
    gt_nodes: list[Node],
    distances: list[list[float]],
    max_distance: float,
) -> dict[int, int]:
    candidates = sorted(
        (
            (dist, pred_nodes[i].node_id, gt_nodes[j].node_id)
            for i, row in enumerate(distances)
            for j, dist in enumerate(row)
            if dist <= max_distance
        ),
        key=lambda x: (x[0], x[1], x[2]),
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    assignment: dict[int, int] = {}
    for _, pred_id, gt_id in candidates:
        if pred_id in used_pred or gt_id in used_gt:
            continue
        assignment[pred_id] = gt_id
        used_pred.add(pred_id)
        used_gt.add(gt_id)
    return assignment


def evaluate_edges(
    pred: Graph,
    gt: Graph,
    *,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_MATCH_DISTANCE_MICRONS,
    total_true_nodes: int | float | None = None,
) -> EdgeEvaluation:
    matches = match_nodes(pred, gt, scale=scale, max_distance=max_distance)
    gt_edges = gt.edges_set()
    pred_edges = sorted(set((e.source_id, e.target_id) for e in pred.edges_list))
    gt_out_valid = {node_id: gt.out_degree(node_id) > 0 for node_id in gt.node_ids()}
    gt_in_valid = {node_id: gt.in_degree(node_id) > 0 for node_id in gt.node_ids()}

    edge_tp = 0
    valid_pred_edges = 0
    for source_id, target_id in pred_edges:
        gt_source = matches.pred_to_gt.get(source_id)
        gt_target = matches.pred_to_gt.get(target_id)
        matched_edge = (
            gt_source is not None
            and gt_target is not None
            and (gt_source, gt_target) in gt_edges
        )
        source_valid = gt_source is not None and gt_out_valid.get(gt_source, False)
        target_valid = gt_target is not None and gt_in_valid.get(gt_target, False)
        if matched_edge:
            edge_tp += 1
            valid_pred_edges += 1
        elif source_valid or target_valid:
            valid_pred_edges += 1

    edge_fp = valid_pred_edges - edge_tp
    edge_fn = gt.num_edges - edge_tp
    denom = edge_tp + edge_fp + edge_fn
    edge_jaccard = edge_tp / denom if denom else float("nan")
    total_node_ratio = _total_node_ratio(pred.num_nodes, total_true_nodes)
    adjusted = _adjusted_edge_jaccard(edge_jaccard, total_node_ratio)
    return EdgeEvaluation(
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        num_pred_nodes=pred.num_nodes,
        edge_jaccard=edge_jaccard,
        adjusted_edge_jaccard=adjusted,
        total_node_ratio=total_node_ratio,
    )


def _total_node_ratio(num_pred_nodes: int, total_true_nodes: int | float | None) -> float:
    if total_true_nodes is None or total_true_nodes <= 0:
        return float("nan")
    return (num_pred_nodes - float(total_true_nodes)) / float(total_true_nodes)


def _adjusted_edge_jaccard(edge_jaccard: float, total_node_ratio: float) -> float:
    if isnan(edge_jaccard) or isnan(total_node_ratio):
        return float("nan")
    return max(0.0, edge_jaccard * (1.0 - ADJUSTMENT_ALPHA * total_node_ratio))

