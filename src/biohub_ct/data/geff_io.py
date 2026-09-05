from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biohub_ct.data.schema import Edge, Graph, Node


@dataclass(frozen=True)
class GeffMetadata:
    estimated_number_of_nodes: float | None = None
    source: str = "unknown"


def read_geff_graph(path: Path | str) -> tuple[Graph, GeffMetadata]:
    geff_path = Path(path)
    json_graph = geff_path / "graph.json"
    if json_graph.exists():
        return _read_json_graph(json_graph)

    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "Reading real GEFF files requires the optional 'zarr' package. "
            "Synthetic tests can use graph.json inside a .geff directory."
        ) from exc

    group = zarr.open_group(str(geff_path), mode="r")
    metadata = dict(group.attrs)["geff"]
    if metadata.get("directed") is not True:
        raise ValueError("Tracking GEFF must be directed")
    ids = np.asarray(group["nodes/ids"][:])
    props = [np.asarray(group[f"nodes/props/{key}/values"][:]) for key in ("t", "z", "y", "x")]
    links = np.asarray(group["edges/ids"][:])
    for array in [ids, *props, links]:
        if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
            raise ValueError("GEFF IDs and voxel coordinates must be nonnegative integers")
    if (
        ids.ndim != 1
        or any(a.shape != ids.shape for a in props)
        or links.ndim != 2
        or links.shape[1] != 2
    ):
        raise ValueError("Invalid GEFF array shapes")
    nodes = [Node(int(i), *(int(a[j]) for a in props)) for j, i in enumerate(ids)]
    edges = [Edge(int(s), int(t)) for s, t in links]
    estimate = metadata.get("extra", {}).get("estimated_number_of_nodes")
    if estimate is not None and (not np.isfinite(estimate) or estimate <= 0):
        raise ValueError("Invalid estimated_number_of_nodes")
    return Graph(nodes, edges), GeffMetadata(estimated_number_of_nodes=estimate, source="geff-zarr")


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
