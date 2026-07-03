from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, order=True)
class Node:
    node_id: int
    t: int
    z: int
    y: int
    x: int

    @property
    def coord(self) -> tuple[int, int, int]:
        return (self.z, self.y, self.x)

    def physical_coord(self, scale: tuple[float, float, float]) -> np.ndarray:
        return np.asarray(self.coord, dtype=float) * np.asarray(scale, dtype=float)


@dataclass(frozen=True, order=True)
class Edge:
    source_id: int
    target_id: int


class Graph:
    """Small directed tracking graph used by the dependency-light baseline."""

    def __init__(
        self,
        nodes: Iterable[Node] | None = None,
        edges: Iterable[Edge] | None = None,
    ) -> None:
        self._nodes: dict[int, Node] = {}
        self._edges: list[Edge] = []
        for node in nodes or ():
            self.add_node(node)
        for edge in edges or ():
            self.add_edge(edge)

    def copy(self) -> "Graph":
        return Graph(self.nodes_list, self.edges_list)

    def add_node(self, node: Node) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node_id {node.node_id}")
        self._nodes[int(node.node_id)] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.source_id not in self._nodes:
            raise ValueError(f"Edge source {edge.source_id} is not a graph node")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Edge target {edge.target_id} is not a graph node")
        self._edges.append(Edge(int(edge.source_id), int(edge.target_id)))

    @property
    def nodes(self) -> dict[int, Node]:
        return dict(self._nodes)

    @property
    def nodes_list(self) -> list[Node]:
        return sorted(self._nodes.values(), key=lambda n: (n.t, n.node_id))

    @property
    def edges_list(self) -> list[Edge]:
        return list(self._edges)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def node(self, node_id: int) -> Node:
        return self._nodes[node_id]

    def node_ids(self) -> list[int]:
        return sorted(self._nodes)

    def edges_set(self) -> set[tuple[int, int]]:
        return {(e.source_id, e.target_id) for e in self._edges}

    def successors(self, node_id: int) -> list[int]:
        return [e.target_id for e in self._edges if e.source_id == node_id]

    def predecessors(self, node_id: int) -> list[int]:
        return [e.source_id for e in self._edges if e.target_id == node_id]

    def out_degree(self, node_id: int) -> int:
        return len(self.successors(node_id))

    def in_degree(self, node_id: int) -> int:
        return len(self.predecessors(node_id))

    def dividing_nodes(self) -> list[int]:
        return [node_id for node_id in self.node_ids() if self.out_degree(node_id) >= 2]

    def subgraph(self, node_ids: Iterable[int]) -> "Graph":
        keep = set(node_ids)
        return Graph(
            nodes=[n for n in self.nodes_list if n.node_id in keep],
            edges=[
                e
                for e in self._edges
                if e.source_id in keep and e.target_id in keep
            ],
        )

    def nodes_by_time(self) -> dict[int, list[Node]]:
        by_t: dict[int, list[Node]] = {}
        for node in self.nodes_list:
            by_t.setdefault(node.t, []).append(node)
        return by_t


def graph_from_points(points: Iterable[tuple[int, int, int, int]]) -> Graph:
    nodes = [
        Node(node_id=i, t=int(t), z=int(z), y=int(y), x=int(x))
        for i, (t, z, y, x) in enumerate(points)
    ]
    return Graph(nodes=nodes)

