from __future__ import annotations

from biohub_ct.data.schema import Node
from biohub_ct.linking.greedy import greedy_link


def test_greedy_link_enforces_one_parent_per_child():
    nodes = [
        Node(1, 0, 0, 0, 0),
        Node(2, 0, 0, 0, 1),
        Node(3, 1, 0, 0, 0),
    ]

    edges = greedy_link(nodes, scale=(1.0, 1.0, 1.0), max_distance=5.0)

    assert len(edges) == 1
    assert edges[0].target_id == 3


def test_greedy_link_uses_physical_distance_gate():
    nodes = [Node(1, 0, 0, 0, 0), Node(2, 1, 0, 0, 100)]

    edges = greedy_link(nodes, scale=(1.0, 1.0, 1.0), max_distance=5.0)

    assert edges == []

