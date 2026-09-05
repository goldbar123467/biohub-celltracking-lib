from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from biohub_ct.config import SUBMISSION_COLUMNS
from biohub_ct.data.schema import Edge, Graph, Node


class SubmissionError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    row_count: int
    datasets: tuple[str, ...]


def validate_graph(graph: Graph, *, shape=None) -> None:
    if not graph.num_nodes:
        raise SubmissionError("Dataset has no nodes")
    for node in graph.nodes_list:
        values = (node.node_id, node.t, node.z, node.y, node.x)
        if any(not isinstance(v, int) or v < 0 for v in values):
            raise SubmissionError("Nodes require nonnegative integer IDs and coordinates")
        if shape and any(v >= n for v, n in zip(values[1:], shape)):
            raise SubmissionError(f"Node {node.node_id} outside image bounds {shape}")
    seen, parents, children = set(), {}, {}
    for edge in graph.edges_list:
        key = (edge.source_id, edge.target_id)
        if key in seen:
            raise SubmissionError("Duplicate edge")
        seen.add(key)
        if graph.node(edge.target_id).t != graph.node(edge.source_id).t + 1:
            raise SubmissionError("Edges must connect consecutive forward frames")
        parents[edge.target_id] = parents.get(edge.target_id, 0) + 1
        children[edge.source_id] = children.get(edge.source_id, 0) + 1
        if parents[edge.target_id] > 1 or children[edge.source_id] > 2:
            raise SubmissionError("Graph permits at most one parent and two children")


def _integer(value, column):
    try:
        parsed = int(value)
    except (ValueError, TypeError) as exc:
        raise SubmissionError(f"{column} must be an integer") from exc
    if str(parsed) != value:
        raise SubmissionError(f"{column} must be a canonical integer")
    return parsed


def _make_graph(nodes, edges):
    try:
        graph = Graph(nodes, edges)
        validate_graph(graph)
        return graph
    except ValueError as exc:
        raise SubmissionError(str(exc)) from exc


def iter_submission_graphs(csv_path):
    """Parse one contiguous dataset at a time, checking schema and graph invariants."""
    current, seen, nodes, edges = None, set(), [], []
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SUBMISSION_COLUMNS:
            raise SubmissionError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}")
        for index, row in enumerate(reader):
            if None in row or any(v is None or v == "" for v in row.values()):
                raise SubmissionError("Unexpected, blank or missing CSV fields")
            values = {
                k: _integer(row[k], k)
                for k in SUBMISSION_COLUMNS
                if k not in ("dataset", "row_type")
            }
            if values["id"] != index:
                raise SubmissionError("Submission IDs must be consecutive starting at 0")
            dataset = row["dataset"]
            if dataset != current:
                if current is not None:
                    yield current, _make_graph(nodes, edges)
                if dataset in seen:
                    raise SubmissionError("Dataset rows must be contiguous")
                seen.add(dataset)
                current, nodes, edges = dataset, [], []
            if row["row_type"] == "node":
                if values["source_id"] != -1 or values["target_id"] != -1:
                    raise SubmissionError("Node endpoints must be -1")
                nodes.append(Node(*(values[k] for k in ("node_id", "t", "z", "y", "x"))))
            elif row["row_type"] == "edge":
                if any(values[k] != -1 for k in ("node_id", "t", "z", "y", "x")):
                    raise SubmissionError("Edge node fields must be -1")
                if values["source_id"] < 0 or values["target_id"] < 0:
                    raise SubmissionError("Edge endpoints must be nonnegative")
                edges.append(Edge(values["source_id"], values["target_id"]))
            else:
                raise SubmissionError("Invalid row_type")
    if current is not None:
        yield current, _make_graph(nodes, edges)


def validate_submission(path, *, expected_datasets=None, shapes=None):
    datasets, row_count = [], 0
    for dataset, graph in iter_submission_graphs(path):
        if shapes is not None and dataset not in shapes:
            raise SubmissionError(f"Unexpected dataset {dataset}")
        validate_graph(graph, shape=shapes[dataset] if shapes else None)
        datasets.append(dataset)
        row_count += graph.num_nodes + graph.num_edges
    if not datasets:
        raise SubmissionError("Submission contains no datasets")
    if expected_datasets is not None and set(expected_datasets) != set(datasets):
        raise SubmissionError(
            f"Dataset coverage mismatch: expected {sorted(expected_datasets)}, got {sorted(datasets)}"
        )
    return ValidationResult(Path(path), row_count, tuple(sorted(datasets)))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("submission_csv")
    args = parser.parse_args(argv)
    print(validate_submission(args.submission_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
