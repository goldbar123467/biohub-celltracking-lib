from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

from biohub_ct.config import SUBMISSION_COLUMNS
from biohub_ct.submission.validator import validate_graph, validate_submission


def write_submission(graphs, output_path, *, expected_datasets=None, shapes=None):
    """Stream dataset graphs; atomically publish only a completely validated CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(graphs.items()) if isinstance(graphs, dict) else graphs
    handle, name = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent)
    temp = Path(name)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(SUBMISSION_COLUMNS)
            row_id = 0
            for dataset, graph in entries:
                validate_graph(graph, shape=shapes[dataset] if shapes else None)
                for node in graph.nodes_list:
                    writer.writerow(
                        [
                            row_id,
                            dataset,
                            "node",
                            node.node_id,
                            node.t,
                            node.z,
                            node.y,
                            node.x,
                            -1,
                            -1,
                        ]
                    )
                    row_id += 1
                for edge in sorted(graph.edges_list):
                    writer.writerow(
                        [
                            row_id,
                            dataset,
                            "edge",
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            edge.source_id,
                            edge.target_id,
                        ]
                    )
                    row_id += 1
            stream.flush()
            os.fsync(stream.fileno())
        validate_submission(temp, expected_datasets=expected_datasets, shapes=shapes)
        os.replace(temp, out)
    finally:
        temp.unlink(missing_ok=True)
    return out
