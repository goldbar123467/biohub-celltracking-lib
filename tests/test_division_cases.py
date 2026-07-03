from __future__ import annotations

from biohub_ct.data.schema import Edge, Graph, Node
from biohub_ct.metrics.division import evaluate_divisions


def gt_division_graph() -> Graph:
    return Graph(
        nodes=[
            Node(1, 0, 0, 0, 0),
            Node(2, 1, 0, 0, 1),
            Node(3, 2, 0, -1, 2),
            Node(4, 2, 0, 1, 2),
            Node(5, 3, 0, -2, 3),
            Node(6, 3, 0, 2, 3),
        ],
        edges=[Edge(1, 2), Edge(2, 3), Edge(2, 4), Edge(3, 5), Edge(4, 6)],
    )


def test_division_component_coverage_scores_tp():
    gt = gt_division_graph()
    pred = gt.copy()

    result = evaluate_divisions(pred, gt)

    assert result.tp == 1
    assert result.fn == 0
    assert result.fp == 0
    assert result.division_jaccard == 1.0


def test_missing_predicted_fork_scores_fn_even_if_daughters_are_touched():
    gt = gt_division_graph()
    pred = Graph(
        nodes=gt.nodes_list,
        edges=[Edge(1, 2), Edge(2, 3), Edge(3, 5), Edge(4, 6)],
    )

    result = evaluate_divisions(pred, gt)

    assert result.tp == 0
    assert result.fn == 1


def test_predicted_division_in_annotated_region_counts_fp_when_not_paired():
    gt = Graph(
        nodes=[Node(1, 0, 0, 0, 0), Node(2, 1, 0, 0, 1), Node(3, 1, 0, 3, 1)],
        edges=[Edge(1, 2)],
    )
    pred = Graph(
        nodes=gt.nodes_list,
        edges=[Edge(1, 2), Edge(1, 3)],
    )

    result = evaluate_divisions(pred, gt)

    assert result.tp == 0
    assert result.fn == 0
    assert result.fp == 1

