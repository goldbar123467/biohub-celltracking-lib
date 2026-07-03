from __future__ import annotations

from biohub_ct.data.schema import Graph, Node


def ensure_nonempty(graph: Graph) -> Graph:
    if graph.num_nodes:
        return graph
    return Graph(nodes=[Node(0, 0, 0, 0, 0)])

