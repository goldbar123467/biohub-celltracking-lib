from __future__ import annotations

from biohub_ct.data.schema import Edge, Graph, Node
from biohub_ct.metrics.edge import evaluate_edges, match_nodes


SCALE = (1.625, 0.40625, 0.40625)


def line_graph(offset_x: int = 0) -> Graph:
    return Graph(
        nodes=[
            Node(1, t=0, z=0, y=0, x=0 + offset_x),
            Node(2, t=1, z=0, y=0, x=1 + offset_x),
        ],
        edges=[Edge(1, 2)],
    )


def test_perfect_edge_graph_scores_one():
    result = evaluate_edges(line_graph(), line_graph(), scale=SCALE, total_true_nodes=2)
    assert result.edge_tp == 1
    assert result.edge_fp == 0
    assert result.edge_fn == 0
    assert result.edge_jaccard == 1.0
    assert result.adjusted_edge_jaccard == 1.0


def test_shifted_nodes_within_7_microns_match():
    matches = match_nodes(line_graph(offset_x=10), line_graph(), scale=SCALE, max_distance=7.0)
    assert matches.pred_to_gt == {1: 1, 2: 2}


def test_shifted_nodes_outside_7_microns_do_not_match():
    matches = match_nodes(line_graph(offset_x=20), line_graph(), scale=SCALE, max_distance=7.0)
    assert matches.pred_to_gt == {}


def test_duplicate_edges_do_not_inflate_true_positives():
    pred = Graph(
        nodes=line_graph().nodes_list,
        edges=[Edge(1, 2), Edge(1, 2)],
    )

    result = evaluate_edges(pred, line_graph(), scale=SCALE, total_true_nodes=2)

    assert result.edge_tp == 1
    assert result.edge_fp == 0
    assert result.edge_jaccard == 1.0


def test_spurious_edge_touching_annotated_node_counts_as_fp():
    gt = line_graph()
    pred = Graph(
        nodes=[
            Node(1, 0, 0, 0, 0),
            Node(2, 1, 0, 0, 1),
            Node(3, 1, 0, 10, 10),
        ],
        edges=[Edge(1, 3)],
    )

    result = evaluate_edges(pred, gt, scale=SCALE, total_true_nodes=3)

    assert result.edge_tp == 0
    assert result.edge_fp == 1
    assert result.edge_fn == 1


def test_unannotated_region_edges_are_ignored_but_nodes_affect_penalty():
    gt = line_graph()
    pred = Graph(
        nodes=[
            Node(1, 0, 0, 0, 0),
            Node(2, 1, 0, 0, 1),
            Node(100, 0, 20, 20, 20),
            Node(101, 1, 20, 20, 21),
        ],
        edges=[Edge(1, 2), Edge(100, 101)],
    )

    result = evaluate_edges(pred, gt, scale=SCALE, total_true_nodes=2)

    assert result.edge_tp == 1
    assert result.edge_fp == 0
    assert result.edge_fn == 0
    assert result.edge_jaccard == 1.0
    assert result.adjusted_edge_jaccard == 0.9


def test_node_count_penalty_can_reward_underprediction_as_official_formula_does():
    result = evaluate_edges(line_graph(), line_graph(), scale=SCALE, total_true_nodes=4)
    assert result.total_node_ratio == -0.5
    assert result.adjusted_edge_jaccard == 1.05

