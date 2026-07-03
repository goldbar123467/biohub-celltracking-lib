from __future__ import annotations

from dataclasses import dataclass
from math import isnan

from biohub_ct.config import DEFAULT_SCALE, MAX_MATCH_DISTANCE_MICRONS
from biohub_ct.data.schema import Graph
from biohub_ct.metrics.division import evaluate_divisions
from biohub_ct.metrics.edge import evaluate_edges


@dataclass(frozen=True)
class OfficialEvaluation:
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int
    node_recall: float
    total_node_ratio: float
    edge_jaccard: float
    adjusted_edge_jaccard: float
    division_jaccard: float
    score: float


def official_metric_available() -> bool:
    try:
        import tracking_cellmot.metrics  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def to_tracksdata_graph(graph: Graph):
    """Convert the repo's lightweight graph to a tracksdata graph for official scoring."""
    try:
        import polars as pl
        import tracksdata as td
    except Exception as exc:
        raise ImportError(
            "Official metric parity requires polars and tracksdata. Install with "
            "`python -m pip install polars scipy 'tracksdata @ "
            "git+https://github.com/royerlab/tracksdata@main'`."
        ) from exc

    out = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        out.add_node_attr_key(key, pl.Float64, 0.0)
    id_map: dict[int, int] = {}
    for node in graph.nodes_list:
        id_map[node.node_id] = out.add_node(
            {"t": int(node.t), "z": float(node.z), "y": float(node.y), "x": float(node.x)}
        )
    for edge in graph.edges_list:
        if edge.source_id in id_map and edge.target_id in id_map:
            out.add_edge(id_map[edge.source_id], id_map[edge.target_id], {})
    return out


def evaluate_official(
    pred: Graph,
    gt: Graph,
    *,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_MATCH_DISTANCE_MICRONS,
    total_true_nodes: int | float | None = None,
) -> OfficialEvaluation:
    """Run the organizer metric implementation on internal graphs."""
    try:
        from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics
    except Exception as exc:
        raise ImportError(
            "Official metric package is not importable. Install it with "
            "`python -m pip install --no-deps 'tracking-cellmot @ "
            "git+https://github.com/royerlab/kaggle-cell-tracking-competition@main'`."
        ) from exc

    pred_td = to_tracksdata_graph(pred)
    gt_td = to_tracksdata_graph(gt)
    er = evaluate(pred_td, gt_td, scale=scale, max_distance=max_distance)
    recall = node_recall(pred_td, gt_td) if gt.num_nodes > 0 else float("nan")
    n_total = float(total_true_nodes) if total_true_nodes is not None else float("nan")
    metrics = per_sample_metrics(er, n_total=n_total, node_recall=recall)
    division_jaccard = _jaccard(er.division_tp, er.division_fp, er.division_fn)
    has_divisions = er.division_tp + er.division_fp + er.division_fn > 0
    if has_divisions and not isnan(metrics["adj_edge_jaccard"]):
        score = metrics["adj_edge_jaccard"] + 0.1 * division_jaccard
    else:
        score = metrics["adj_edge_jaccard"]
    return OfficialEvaluation(
        edge_tp=er.edge_tp,
        edge_fp=er.edge_fp,
        edge_fn=er.edge_fn,
        division_tp=er.division_tp,
        division_fp=er.division_fp,
        division_fn=er.division_fn,
        num_pred_nodes=er.num_pred_nodes,
        node_recall=metrics["node_recall"],
        total_node_ratio=metrics["total_node_ratio"],
        edge_jaccard=metrics["edge_jaccard"],
        adjusted_edge_jaccard=metrics["adj_edge_jaccard"],
        division_jaccard=division_jaccard,
        score=score,
    )


def evaluate_with_best_available(
    pred: Graph,
    gt: Graph,
    *,
    total_true_nodes: int | float | None = None,
):
    edge = evaluate_edges(pred, gt, total_true_nodes=total_true_nodes)
    div = evaluate_divisions(pred, gt)
    return {"edge": edge, "division": div, "official_available": official_metric_available()}


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom else float("nan")
