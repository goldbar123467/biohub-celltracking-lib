from __future__ import annotations

from biohub_ct.data.schema import Edge, Graph, Node
from biohub_ct.metrics.division import evaluate_divisions
from biohub_ct.metrics.edge import evaluate_edges


def simple_line_graph() -> Graph:
    return Graph(nodes=[Node(1, 0, 0, 0, 0), Node(2, 1, 0, 0, 1)], edges=[Edge(1, 2)])


def run_probe_report() -> str:
    gt = simple_line_graph()
    pred = simple_line_graph()
    edge = evaluate_edges(pred, gt, total_true_nodes=2)
    div = evaluate_divisions(pred, gt)
    return (
        "metric_probe\n"
        f"edge_tp={edge.edge_tp} edge_fp={edge.edge_fp} edge_fn={edge.edge_fn} "
        f"edge_jaccard={edge.edge_jaccard:.6f} adjusted={edge.adjusted_edge_jaccard:.6f}\n"
        f"division_tp={div.tp} division_fp={div.fp} division_fn={div.fn}\n"
    )


if __name__ == "__main__":
    print(run_probe_report())

