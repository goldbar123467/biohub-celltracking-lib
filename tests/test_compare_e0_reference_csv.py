from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
from scripts import compare_e0_reference_csv as comparator
from scripts.compare_e0_reference_csv import ComparisonError, compare_reference_csv

HEADER = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def _node(dataset: str, node_id: int, t: int, z: int, y: int, x: int) -> list[object]:
    return [0, dataset, "node", node_id, t, z, y, x, -1, -1]


def _edge(dataset: str, source_id: int, target_id: int) -> list[object]:
    return [0, dataset, "edge", -1, -1, -1, -1, -1, source_id, target_id]


def _write_csv(path: Path, rows: list[list[object]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HEADER)
        for row_id, row in enumerate(rows):
            writer.writerow([row_id, *row[1:]])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_rows() -> list[list[object]]:
    return [
        _node("alpha", 10, 0, 1, 2, 3),
        _node("alpha", 11, 1, 2, 3, 4),
        _node("alpha", 12, 1, 5, 6, 7),
        _edge("alpha", 10, 11),
        _node("beta", 20, 0, 8, 9, 10),
        _node("beta", 21, 1, 9, 10, 11),
        _edge("beta", 20, 21),
    ]


def _compare(tmp_path: Path, candidate_rows: list[list[object]]) -> dict[str, object]:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    expected_hash = _write_csv(reference, _reference_rows())
    _write_csv(candidate, candidate_rows)
    return compare_reference_csv(
        candidate,
        reference_csv=reference,
        expected_reference_sha256=expected_hash,
    )


def test_exact_records_and_edges_match_independent_of_csv_row_order(tmp_path: Path) -> None:
    reference = _reference_rows()
    candidate = [
        reference[4],
        reference[5],
        reference[6],
        reference[2],
        reference[0],
        reference[3],
        reference[1],
    ]

    report = _compare(tmp_path, candidate)

    assert report["status"] == "EXACT_SEMANTIC_MATCH"
    assert report["byte_identical"] is False
    assert report["dataset_sets_exact"] is True
    assert report["node_ids_exact"] is True
    assert report["node_values_exact"] is True
    assert report["edge_identities_exact"] is True
    assert report["reference"]["row_count"] == 7
    assert report["candidate"]["row_count"] == 7
    assert report["datasets"]["alpha"]["reference"]["node_count"] == 3
    assert report["datasets"]["alpha"]["candidate"]["edge_count"] == 1
    assert (
        report["datasets"]["alpha"]["reference"]["node_records_sha256"]
        == report["datasets"]["alpha"]["candidate"]["node_records_sha256"]
    )


def test_exact_byte_match_is_reported_separately(tmp_path: Path) -> None:
    report = _compare(tmp_path, _reference_rows())
    assert report["status"] == "EXACT_SEMANTIC_MATCH"
    assert report["byte_identical"] is True
    assert report["reference"]["sha256"] == report["candidate"]["sha256"]


def test_changed_node_value_is_exact_difference_without_tolerance(tmp_path: Path) -> None:
    candidate = _reference_rows()
    candidate[1] = _node("alpha", 11, 1, 2, 3, 5)

    report = _compare(tmp_path, candidate)

    assert report["status"] == "DIFFERENT"
    assert report["node_ids_exact"] is True
    assert report["node_values_exact"] is False
    assert report["datasets"]["alpha"]["differences"]["changed_node_values_count"] == 1
    assert report["datasets"]["alpha"]["differences"]["changed_node_values_sample"] == [
        {"node_id": 11, "reference_tzyx": [1, 2, 3, 4], "candidate_tzyx": [1, 2, 3, 5]}
    ]
    assert report["comparison_contract"]["numeric_tolerance"] is None


def test_changed_edge_identity_is_reported_exactly(tmp_path: Path) -> None:
    candidate = _reference_rows()
    candidate[3] = _edge("alpha", 10, 12)

    report = _compare(tmp_path, candidate)

    alpha = report["datasets"]["alpha"]
    assert report["status"] == "DIFFERENT"
    assert report["node_values_exact"] is True
    assert report["edge_identities_exact"] is False
    assert alpha["differences"]["missing_edges_sample"] == [[10, 11]]
    assert alpha["differences"]["extra_edges_sample"] == [[10, 12]]


def test_node_id_change_remains_explicitly_unresolved(tmp_path: Path) -> None:
    reference = _reference_rows()
    candidate = [
        _node("alpha", 110, 0, 1, 2, 3),
        _node("alpha", 111, 1, 2, 3, 4),
        _node("alpha", 112, 1, 5, 6, 7),
        _edge("alpha", 110, 111),
        *reference[4:],
    ]

    report = _compare(tmp_path, candidate)

    alpha = report["datasets"]["alpha"]
    assert report["status"] == "DIFFERENT"
    assert report["node_ids_exact"] is False
    assert report["graph_isomorphism_status"] == "UNRESOLVED_NODE_ID_CHANGE_NO_ISOMORPHISM_CLAIM"
    assert alpha["graph_isomorphism_status"] == "UNRESOLVED_NODE_ID_CHANGE_NO_ISOMORPHISM_CLAIM"
    assert alpha["differences"]["missing_node_ids_count"] == 3
    assert alpha["differences"]["extra_node_ids_count"] == 3
    assert report["comparison_contract"]["graph_isomorphism_attempted"] is False


def test_missing_dataset_reports_exact_counts_and_coverage_difference(tmp_path: Path) -> None:
    report = _compare(tmp_path, _reference_rows()[:4])

    assert report["status"] == "DIFFERENT"
    assert report["missing_datasets"] == ["beta"]
    assert report["extra_datasets"] == []
    assert report["datasets"]["beta"]["candidate"]["node_count"] == 0
    assert report["datasets"]["beta"]["reference"]["edge_count"] == 1


def test_reference_hash_must_match_before_comparison(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(reference, _reference_rows())
    _write_csv(candidate, _reference_rows())

    with pytest.raises(ComparisonError, match="frozen expected SHA-256"):
        compare_reference_csv(
            candidate,
            reference_csv=reference,
            expected_reference_sha256="0" * 64,
        )


def test_both_csvs_require_structural_validation(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    expected_hash = _write_csv(reference, _reference_rows())
    candidate.write_text("wrong,header\n", encoding="utf-8")

    with pytest.raises(ComparisonError, match="candidate CSV failed structural validation"):
        compare_reference_csv(
            candidate,
            reference_csv=reference,
            expected_reference_sha256=expected_hash,
        )


def test_candidate_mutation_after_parse_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    expected_hash = _write_csv(reference, _reference_rows())
    _write_csv(candidate, _reference_rows())
    original_load = comparator._load_graphs

    def mutating_load(path: Path, label: str):
        result = original_load(path, label)
        if label == "candidate CSV":
            path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(comparator, "_load_graphs", mutating_load)

    with pytest.raises(ComparisonError, match="changed during comparison"):
        compare_reference_csv(
            candidate,
            reference_csv=reference,
            expected_reference_sha256=expected_hash,
        )


def test_report_writer_refuses_symlink_without_resolving_it(tmp_path: Path) -> None:
    target = tmp_path / "existing.json"
    target.write_text("preserve\n", encoding="utf-8")
    link = tmp_path / "report-link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ComparisonError, match="already exists"):
        comparator._write_report(link, {"status": "EXACT_SEMANTIC_MATCH"})

    assert target.read_text(encoding="utf-8") == "preserve\n"
