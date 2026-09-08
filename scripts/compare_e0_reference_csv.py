#!/usr/bin/env python3
"""Compare one E0 rehearsal CSV with the frozen upstream reference exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.data.schema import Graph
from biohub_ct.submission.validator import (
    SubmissionError,
    iter_submission_graphs,
    validate_submission,
)


class ComparisonError(ValueError):
    """A CSV or its reference binding cannot be compared safely."""


REFERENCE_CSV = ROOT / "reports" / "e0-reference" / "upstream-v1" / "submission.csv"
REFERENCE_SHA256 = "a852d1d07ff8c9307d9b10db7f9b4b12e8b1882f14c5dbeb1316d099f0795b3e"
SCHEMA_VERSION = 1
MAX_MISMATCH_EXAMPLES = 20


@dataclass(frozen=True)
class _GraphRecords:
    nodes: dict[int, tuple[int, int, int, int]]
    edges: frozenset[tuple[int, int]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ComparisonError(f"cannot read CSV: {path}") from exc
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ComparisonError(f"{label} must be an existing regular file")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ComparisonError(f"cannot resolve {label}") from exc


def _records_digest(records: Iterable[tuple[int, ...]]) -> str:
    canonical = json.dumps(
        sorted(records), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_graphs(path: Path, label: str) -> tuple[dict[str, _GraphRecords], int]:
    try:
        validation = validate_submission(path)
        graphs = dict(iter_submission_graphs(path))
    except (OSError, UnicodeError, SubmissionError) as exc:
        raise ComparisonError(f"{label} failed structural validation: {exc}") from exc
    if set(graphs) != set(validation.datasets):
        raise ComparisonError(f"{label} graph inventory disagrees with validation")
    return {
        dataset: _graph_records(graph) for dataset, graph in graphs.items()
    }, validation.row_count


def _graph_records(graph: Graph) -> _GraphRecords:
    return _GraphRecords(
        nodes={node.node_id: (node.t, node.z, node.y, node.x) for node in graph.nodes_list},
        edges=frozenset((edge.source_id, edge.target_id) for edge in graph.edges_list),
    )


def _sample(values: Iterable[tuple[int, ...]]) -> list[list[int]]:
    return [list(value) for value in sorted(values)[:MAX_MISMATCH_EXAMPLES]]


def _dataset_comparison(
    reference: _GraphRecords | None, candidate: _GraphRecords | None
) -> dict[str, object]:
    reference = reference or _GraphRecords({}, frozenset())
    candidate = candidate or _GraphRecords({}, frozenset())
    reference_ids = set(reference.nodes)
    candidate_ids = set(candidate.nodes)
    missing_ids = reference_ids - candidate_ids
    extra_ids = candidate_ids - reference_ids
    common_ids = reference_ids & candidate_ids
    changed_values = [
        node_id
        for node_id in sorted(common_ids)
        if reference.nodes[node_id] != candidate.nodes[node_id]
    ]
    missing_edges = reference.edges - candidate.edges
    extra_edges = candidate.edges - reference.edges
    node_ids_exact = not missing_ids and not extra_ids
    node_values_exact = node_ids_exact and not changed_values
    edges_exact = not missing_edges and not extra_edges
    if node_ids_exact:
        isomorphism_status = "NOT_PERFORMED_NODE_IDENTITIES_ALIGNED"
    else:
        isomorphism_status = "UNRESOLVED_NODE_ID_CHANGE_NO_ISOMORPHISM_CLAIM"
    changed_examples = [
        {
            "node_id": node_id,
            "reference_tzyx": list(reference.nodes[node_id]),
            "candidate_tzyx": list(candidate.nodes[node_id]),
        }
        for node_id in changed_values[:MAX_MISMATCH_EXAMPLES]
    ]
    return {
        "reference": {
            "node_count": len(reference.nodes),
            "edge_count": len(reference.edges),
            "node_records_sha256": _records_digest(
                (node_id, *values) for node_id, values in reference.nodes.items()
            ),
            "edge_identities_sha256": _records_digest(reference.edges),
        },
        "candidate": {
            "node_count": len(candidate.nodes),
            "edge_count": len(candidate.edges),
            "node_records_sha256": _records_digest(
                (node_id, *values) for node_id, values in candidate.nodes.items()
            ),
            "edge_identities_sha256": _records_digest(candidate.edges),
        },
        "node_ids_exact": node_ids_exact,
        "node_values_exact": node_values_exact,
        "edge_identities_exact": edges_exact,
        "graph_isomorphism_status": isomorphism_status,
        "differences": {
            "missing_node_ids_count": len(missing_ids),
            "extra_node_ids_count": len(extra_ids),
            "changed_node_values_count": len(changed_values),
            "missing_edges_count": len(missing_edges),
            "extra_edges_count": len(extra_edges),
            "missing_node_ids_sample": sorted(missing_ids)[:MAX_MISMATCH_EXAMPLES],
            "extra_node_ids_sample": sorted(extra_ids)[:MAX_MISMATCH_EXAMPLES],
            "changed_node_values_sample": changed_examples,
            "missing_edges_sample": _sample(missing_edges),
            "extra_edges_sample": _sample(extra_edges),
            "sample_limit": MAX_MISMATCH_EXAMPLES,
        },
    }


def compare_reference_csv(
    candidate_csv: Path,
    *,
    reference_csv: Path = REFERENCE_CSV,
    expected_reference_sha256: str = REFERENCE_SHA256,
) -> dict[str, object]:
    """Return an exact, row-order-independent comparison of two valid CSVs.

    Node equality uses ``(node_id, t, z, y, x)`` integer records.  Edge
    equality uses ``(source_id, target_id)`` identities.  The function does not
    attempt to infer a node renaming or graph isomorphism.
    """

    if not isinstance(expected_reference_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_reference_sha256
    ):
        raise ComparisonError("expected reference SHA-256 must be lowercase hexadecimal")
    reference_path = _regular_file(Path(reference_csv), "reference CSV")
    candidate_path = _regular_file(Path(candidate_csv), "candidate CSV")
    reference_sha256 = _sha256_file(reference_path)
    if reference_sha256 != expected_reference_sha256:
        raise ComparisonError("reference CSV bytes do not match the frozen expected SHA-256")
    candidate_sha256 = _sha256_file(candidate_path)
    reference_graphs, reference_rows = _load_graphs(reference_path, "reference CSV")
    candidate_graphs, candidate_rows = _load_graphs(candidate_path, "candidate CSV")
    reference_datasets = set(reference_graphs)
    candidate_datasets = set(candidate_graphs)
    datasets = {
        dataset: _dataset_comparison(reference_graphs.get(dataset), candidate_graphs.get(dataset))
        for dataset in sorted(reference_datasets | candidate_datasets)
    }
    dataset_sets_exact = reference_datasets == candidate_datasets
    node_ids_exact = dataset_sets_exact and all(
        bool(result["node_ids_exact"]) for result in datasets.values()
    )
    node_values_exact = dataset_sets_exact and all(
        bool(result["node_values_exact"]) for result in datasets.values()
    )
    edges_exact = dataset_sets_exact and all(
        bool(result["edge_identities_exact"]) for result in datasets.values()
    )
    semantic_exact = node_values_exact and edges_exact
    any_node_id_change = not node_ids_exact
    # Bind the result to the bytes that were parsed.  The candidate may still
    # be written or replaced while a download process is winding down.
    if (
        _sha256_file(reference_path) != reference_sha256
        or _sha256_file(candidate_path) != candidate_sha256
    ):
        raise ComparisonError("an input CSV changed during comparison")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "EXACT_SEMANTIC_MATCH" if semantic_exact else "DIFFERENT",
        "comparison_contract": {
            "row_order_ignored": True,
            "numeric_tolerance": None,
            "node_record": ["node_id", "t", "z", "y", "x"],
            "edge_identity": ["source_id", "target_id"],
            "graph_isomorphism_attempted": False,
        },
        "reference": {
            "path": str(reference_path),
            "sha256": reference_sha256,
            "expected_sha256": expected_reference_sha256,
            "row_count": reference_rows,
            "datasets": sorted(reference_datasets),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": candidate_sha256,
            "row_count": candidate_rows,
            "datasets": sorted(candidate_datasets),
        },
        "byte_identical": reference_sha256 == candidate_sha256,
        "dataset_sets_exact": dataset_sets_exact,
        "node_ids_exact": node_ids_exact,
        "node_values_exact": node_values_exact,
        "edge_identities_exact": edges_exact,
        "graph_isomorphism_status": (
            "UNRESOLVED_NODE_ID_CHANGE_NO_ISOMORPHISM_CLAIM"
            if any_node_id_change
            else "NOT_PERFORMED_EXACT_IDENTITY_COMPARISON_ONLY"
        ),
        "missing_datasets": sorted(reference_datasets - candidate_datasets),
        "extra_datasets": sorted(candidate_datasets - reference_datasets),
        "datasets": datasets,
    }


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise ComparisonError("report path already exists")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ComparisonError("report parent must be an existing regular directory")
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise ComparisonError("deterministic temporary report path already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a valid rehearsal CSV with the frozen E0 upstream reference."
    )
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = compare_reference_csv(args.candidate_csv)
        if args.report_json is not None:
            _write_report(args.report_json, report)
    except ComparisonError as exc:
        print(f"comparison rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "EXACT_SEMANTIC_MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
