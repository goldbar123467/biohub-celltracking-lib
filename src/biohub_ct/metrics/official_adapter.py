from __future__ import annotations

from biohub_ct.data.schema import Graph
from biohub_ct.metrics.division import evaluate_divisions
from biohub_ct.metrics.edge import evaluate_edges


def official_metric_available() -> bool:
    try:
        import tracking_cellmot.metrics  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def evaluate_with_best_available(
    pred: Graph,
    gt: Graph,
    *,
    total_true_nodes: int | float | None = None,
):
    """Use the local metric unless the caller has converted graphs for official code.

    The organizer implementation depends on tracksdata/polars graph objects. This adapter
    deliberately keeps the repo's no-dependency graph path separate and documents when the
    official package is importable.
    """
    edge = evaluate_edges(pred, gt, total_true_nodes=total_true_nodes)
    div = evaluate_divisions(pred, gt)
    return {"edge": edge, "division": div, "official_available": official_metric_available()}

