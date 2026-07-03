from __future__ import annotations

from biohub_ct.data.schema import Graph


def graph_summary(graph: Graph) -> str:
    return f"nodes={graph.num_nodes} edges={graph.num_edges}"

