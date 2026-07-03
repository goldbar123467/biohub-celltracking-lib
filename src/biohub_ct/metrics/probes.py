from __future__ import annotations

import argparse

from biohub_ct.data.schema import Edge, Graph, Node
from biohub_ct.metrics.division import evaluate_divisions
from biohub_ct.metrics.edge import evaluate_edges
from biohub_ct.metrics.official_adapter import evaluate_official


def simple_line_graph() -> Graph:
    return Graph(nodes=[Node(1, 0, 0, 0, 0), Node(2, 1, 0, 0, 1)], edges=[Edge(1, 2)])


def run_probe_report(*, official: bool = False) -> str:
    gt = simple_line_graph()
    pred = simple_line_graph()
    edge = evaluate_edges(pred, gt, total_true_nodes=2)
    div = evaluate_divisions(pred, gt)
    report = (
        "metric_probe\n"
        f"edge_tp={edge.edge_tp} edge_fp={edge.edge_fp} edge_fn={edge.edge_fn} "
        f"edge_jaccard={edge.edge_jaccard:.6f} adjusted={edge.adjusted_edge_jaccard:.6f}\n"
        f"division_tp={div.tp} division_fp={div.fp} division_fn={div.fn}\n"
    )
    if official:
        off = evaluate_official(pred, gt, total_true_nodes=2)
        report += (
            "official_metric_probe\n"
            f"edge_tp={off.edge_tp} edge_fp={off.edge_fp} edge_fn={off.edge_fn} "
            f"edge_jaccard={off.edge_jaccard:.6f} adjusted={off.adjusted_edge_jaccard:.6f}\n"
            f"division_tp={off.division_tp} division_fp={off.division_fp} "
            f"division_fn={off.division_fn}\n"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", action="store_true")
    args = parser.parse_args(argv)
    print(run_probe_report(official=args.official))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
