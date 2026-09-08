import copy
import hashlib
import json
import time
import zipfile
from pathlib import Path

import pytest
from scripts.package_public_reference import (
    PINNED_REFERENCE,
    _release_validation_source,
    build_package,
    sha256_file,
    verify_reference_inputs,
)

REF = "owner/reference-data"


def run_release_validator(
    tmp_path: Path,
    rows: list[list[object]],
    *,
    datasets: tuple[str, ...] = ("a",),
) -> None:
    competition = "example-competition"
    kaggle_root = tmp_path / "kaggle"
    working = kaggle_root / "working"
    test_root = kaggle_root / "input" / "competitions" / competition / "test"
    working.mkdir(parents=True)
    for dataset in datasets:
        metadata = test_root / f"{dataset}.zarr" / "0" / "zarr.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps({"shape": [2, 3, 4, 5]}), encoding="utf-8")

    submission = working / "submission.csv"
    csv_lines = [",".join(PINNED_REFERENCE["submission_columns"])]
    csv_lines.extend(",".join(str(value) for value in row) for row in rows)
    submission.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    def mapped_path(raw: object) -> Path:
        text = str(raw).replace("\\", "/")
        if text == "/kaggle":
            return kaggle_root
        if text.startswith("/kaggle/"):
            return kaggle_root / text.removeprefix("/kaggle/")
        return Path(raw)

    lock = {
        "competition": competition,
        "reference": {"notebook": {"enable_internet": False}},
        "release_digest": "test-release",
        "submission_columns": PINNED_REFERENCE["submission_columns"],
    }
    namespace = {
        "_E0Path": mapped_path,
        "_E0_LOCK": lock,
        "_E0_INPUT_INTEGRITY": {},
        "_E0_STARTED_MONOTONIC": time.monotonic(),
        "_e0_json": json,
        "_e0_sha256": lambda path: sha256_file(Path(path)),
    }
    source = _release_validation_source(lock)
    exec(compile(source, "test-e0-release-validation", "exec"), namespace)  # noqa: S102


def write_notebook(path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {"tag": "algorithm"},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": ["old\n"]}],
                "source": ["answer = 40 + 2\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["algorithm notes\n"],
            },
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def write_archive(path: Path) -> dict[str, str]:
    payloads = {
        "ARTIFACT_MANIFEST.json": b'{"schema_version":1}',
        "weights/model.bin": b"exact-weights",
        "wheels/dependency.whl": b"offline-wheel",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    return {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}


def make_inputs(tmp_path: Path) -> tuple[Path, Path, dict]:
    notebook = tmp_path / "source.ipynb"
    archive = tmp_path / "reference.zip"
    write_notebook(notebook)
    member_hashes = write_archive(archive)
    pin = {
        "schema_version": 1,
        "evidence_class": "overlap_unknown",
        "notebook": {
            "ref": "owner/reference-notebook",
            "version": 1,
            "script_version_id": 123,
            "source_sha256": sha256_file(notebook),
            "displayed_public_score": 0.946,
            "displayed_runtime": "1h",
            "machine_shape": "NvidiaTeslaT4",
            "displayed_accelerators": 2,
            "docker_image": "example@sha256:" + "a" * 64,
            "enable_internet": False,
        },
        "datasets": {
            REF: {
                "dataset_id": 456,
                "version": 3,
                "last_updated_utc": "2026-01-01T00:00:00Z",
                "license": "CC0-1.0",
                "archive_sha256": sha256_file(archive),
                "required_members": {
                    "ARTIFACT_MANIFEST.json": member_hashes["ARTIFACT_MANIFEST.json"],
                    "weights/model.bin": member_hashes["weights/model.bin"],
                },
                "minimum_wheel_count": 1,
            }
        },
        "competition": "example-competition",
        "submission_columns": [
            "id",
            "dataset",
            "row_type",
            "node_id",
            "t",
            "z",
            "y",
            "x",
            "source_id",
            "target_id",
        ],
    }
    return notebook, archive, pin


def test_build_preserves_algorithm_cells_and_is_deterministic(tmp_path: Path) -> None:
    notebook, archive, pin = make_inputs(tmp_path)
    source = json.loads(notebook.read_text(encoding="utf-8"))
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = build_package(notebook, {REF: archive}, first, "owner/e0-reference", pin=pin)
    second_manifest = build_package(notebook, {REF: archive}, second, "owner/e0-reference", pin=pin)

    packaged = json.loads((first / "submission.ipynb").read_text(encoding="utf-8"))
    assert packaged["cells"][1:-1] == [
        {**source["cells"][0], "execution_count": None, "outputs": []},
        source["cells"][1],
    ]
    assert packaged["cells"][0]["metadata"]["biohub_e0_instrumentation"] == ("e0-input-integrity")
    assert packaged["cells"][-1]["metadata"]["biohub_e0_instrumentation"] == (
        "e0-release-validation"
    )
    for cell in packaged["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "test-packaged-cell", "exec")

    metadata = json.loads((first / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == [f"{REF}/3"]
    assert first_manifest["algorithm_changes"] == []
    assert first_manifest["release_digest"] == second_manifest["release_digest"]
    assert sha256_file(first / "submission.ipynb") == sha256_file(second / "submission.ipynb")


def test_malformed_pin_fails_closed(tmp_path: Path) -> None:
    notebook, archive, pin = make_inputs(tmp_path)
    pin["notebook"]["source_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="Invalid SHA-256"):
        verify_reference_inputs(notebook, {REF: archive}, pin=pin)


def test_dataset_reference_set_must_match_exactly(tmp_path: Path) -> None:
    notebook, archive, pin = make_inputs(tmp_path)
    with pytest.raises(ValueError, match="Dataset reference mismatch"):
        verify_reference_inputs(notebook, {"owner/other": archive}, pin=pin)


def test_archive_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    notebook, archive, pin = make_inputs(tmp_path)
    pin["datasets"][REF]["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Archive checksum mismatch"):
        verify_reference_inputs(notebook, {REF: archive}, pin=pin)


def test_required_member_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    notebook, archive, pin = make_inputs(tmp_path)
    pin = copy.deepcopy(pin)
    pin["datasets"][REF]["required_members"]["weights/model.bin"] = "0" * 64
    with pytest.raises(ValueError, match="Member checksum mismatch"):
        verify_reference_inputs(notebook, {REF: archive}, pin=pin)


def test_release_validator_rejects_negative_node_id(tmp_path: Path) -> None:
    rows = [[0, "a", "node", -2, 0, 0, 0, 0, -1, -1]]
    with pytest.raises(RuntimeError, match="negative_node_id"):
        run_release_validator(tmp_path, rows)


def test_release_validator_rejects_negative_edge_endpoint(tmp_path: Path) -> None:
    rows = [
        [0, "a", "node", 0, 0, 0, 0, 0, -1, -1],
        [1, "a", "node", 1, 1, 0, 0, 0, -1, -1],
        [2, "a", "edge", -1, -1, -1, -1, -1, -2, 1],
    ]
    with pytest.raises(RuntimeError, match="negative_edge_endpoint"):
        run_release_validator(tmp_path, rows)


def test_release_validator_rejects_noncontiguous_dataset_rows(tmp_path: Path) -> None:
    rows = [
        [0, "a", "node", 0, 0, 0, 0, 0, -1, -1],
        [1, "b", "node", 0, 0, 0, 0, 0, -1, -1],
        [2, "a", "node", 1, 1, 0, 0, 0, -1, -1],
    ]
    with pytest.raises(RuntimeError, match="noncontiguous_dataset"):
        run_release_validator(tmp_path, rows, datasets=("a", "b"))


def test_release_validator_rejects_surplus_csv_fields(tmp_path: Path) -> None:
    rows = [[0, "a", "node", 0, 0, 0, 0, 0, -1, -1, "surplus"]]
    with pytest.raises(RuntimeError, match="unexpected_blank_missing_or_extra_field"):
        run_release_validator(tmp_path, rows)
