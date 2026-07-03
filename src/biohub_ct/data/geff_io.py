from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biohub_ct.data.schema import Edge, Graph, Node


@dataclass(frozen=True)
class GeffMetadata:
    estimated_number_of_nodes: int | None = None
    source: str = "unknown"


def read_geff_graph(path: Path | str) -> tuple[Graph, GeffMetadata]:
    geff_path = Path(path)
    json_graph = geff_path / "graph.json"
    if json_graph.exists():
        return _read_json_graph(json_graph)

    try:
        import geff  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Reading real GEFF files requires the optional 'geff' package. "
            "Synthetic tests can use graph.json inside a .geff directory."
        ) from exc

    graph_obj = geff.read(str(geff_path), backend="networkx")
    nodes = []
    for node_id, attrs in graph_obj.nodes(data=True):
        nodes.append(
            Node(
                node_id=int(node_id),
                t=int(attrs["t"]),
                z=int(attrs["z"]),
                y=int(attrs["y"]),
                x=int(attrs["x"]),
            )
        )
    edges = [Edge(int(s), int(t)) for s, t in graph_obj.edges()]
    return Graph(nodes=nodes, edges=edges), GeffMetadata(source="geff")


def _read_json_graph(path: Path) -> tuple[Graph, GeffMetadata]:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    nodes = [
        Node(
            node_id=int(row["node_id"]),
            t=int(row["t"]),
            z=int(row["z"]),
            y=int(row["y"]),
            x=int(row["x"]),
        )
        for row in raw.get("nodes", [])
    ]
    edges = [
        Edge(source_id=int(row["source_id"]), target_id=int(row["target_id"]))
        for row in raw.get("edges", [])
    ]
    return Graph(nodes=nodes, edges=edges), GeffMetadata(
        estimated_number_of_nodes=raw.get("estimated_number_of_nodes"),
        source="json-fallback",
    )

