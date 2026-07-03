from __future__ import annotations

import csv
from pathlib import Path

from biohub_ct.config import SUBMISSION_COLUMNS
from biohub_ct.data.schema import Graph, Node


def write_submission(graphs: dict[str, Graph], output_path: Path | str) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row_id = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for dataset, graph in sorted(graphs.items()):
            graph = _with_fallback_node(graph)
            for node in graph.nodes_list:
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": dataset,
                        "row_type": "node",
                        "node_id": node.node_id,
                        "t": node.t,
                        "z": node.z,
                        "y": node.y,
                        "x": node.x,
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                row_id += 1
            node_ids = {n.node_id for n in graph.nodes_list}
            for edge in sorted(set((e.source_id, e.target_id) for e in graph.edges_list)):
                if edge[0] not in node_ids or edge[1] not in node_ids:
                    continue
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": dataset,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": edge[0],
                        "target_id": edge[1],
                    }
                )
                row_id += 1
    return out


def _with_fallback_node(graph: Graph) -> Graph:
    if graph.num_nodes:
        return graph
    return Graph(nodes=[Node(0, t=0, z=0, y=0, x=0)])

