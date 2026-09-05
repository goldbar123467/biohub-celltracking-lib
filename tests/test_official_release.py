import pytest

pytest.importorskip("tracking_cellmot.metrics")
from biohub_ct.data.schema import Edge, Graph, Node
from biohub_ct.metrics.official_adapter import evaluate_official
from biohub_ct.pipelines.evaluate import official_module


def test_pinned_official_scorer_preserves_sparse_estimate_adjustment():
    official_module()
    graph = Graph([Node(0, 0, 1, 1, 1), Node(1, 1, 1, 1, 1)], [Edge(0, 1)])
    result = evaluate_official(graph, graph, total_true_nodes=20)
    assert result.edge_tp == 1 and result.edge_fp == 0 and result.edge_fn == 0
    assert result.node_recall == 1
    assert result.edge_jaccard == 1
    assert result.adjusted_edge_jaccard == pytest.approx(1.09)


def test_official_summary_uses_edge_weights_and_global_divisions():
    metric = official_module()
    rows = [
        {
            "edge_tp": 1,
            "edge_fp": 0,
            "edge_fn": 0,
            "division_tp": 1,
            "division_fp": 0,
            "division_fn": 0,
            "num_pred_nodes": 2,
            "node_recall": 1,
            "adj_edge_jaccard": 1.09,
        },
        {
            "edge_tp": 2,
            "edge_fp": 1,
            "edge_fn": 1,
            "division_tp": 0,
            "division_fp": 1,
            "division_fn": 1,
            "num_pred_nodes": 8,
            "node_recall": 0.5,
            "adj_edge_jaccard": 0.45,
        },
    ]
    result = metric.summarise(rows)
    expected_adj = (1.09 + 4 * 0.45) / 5
    assert result["adj_edge_jaccard"] == pytest.approx(expected_adj)
    assert result["edge_jaccard"] == pytest.approx(3 / 5)
    assert result["score"] == pytest.approx(expected_adj + 0.1 / 3)
