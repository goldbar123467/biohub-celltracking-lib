from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from biohub_ct.campaign import release_validation
from biohub_ct.campaign.kaggle_rehearsal import PackageIdentity
from biohub_ct.campaign.release_validation import (
    ExactVersionOutputProof,
    ReleaseValidationError,
    validate_downloaded_e0_release,
)
from biohub_ct.config import SUBMISSION_COLUMNS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def make_package(tmp_path: Path) -> tuple[Path, PackageIdentity, dict[str, object]]:
    package = tmp_path / "package"
    package.mkdir()
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": "print('synthetic fixture')\n",
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    write_json(package / "submission.ipynb", notebook)
    notebook_hash = sha256(package / "submission.ipynb")
    normalized_notebook = json.loads((package / "submission.ipynb").read_text(encoding="utf-8"))
    normalized_hash = hashlib.sha256(json.dumps(normalized_notebook).encode()).hexdigest()

    reference_notebook = {
        "ref": "upstream/reference",
        "version": 1,
        "script_version_id": 101,
        "source_sha256": "1" * 64,
        "displayed_public_score": 0.5,
        "displayed_runtime": "fixture",
        "machine_shape": "NvidiaTeslaT4",
        "displayed_accelerators": 1,
        "docker_image": "fixture@sha256:" + "2" * 64,
        "enable_internet": False,
    }
    dataset_ref = "owner/assets"
    dataset_record = {
        "ref": dataset_ref,
        "dataset_id": 7,
        "version": 3,
        "last_updated_utc": "2026-09-08T00:00:00Z",
        "license": "CC0-1.0",
        "archive_file": "assets.zip",
        "archive_bytes": 3,
        "archive_sha256": "3" * 64,
        "artifact_manifest": {"schema_version": 1},
        "files": {"weights.bin": {"bytes": 3, "sha256": "4" * 64}},
        "file_count": 1,
        "wheel_count": 0,
    }
    reference = {
        "schema_version": 1,
        "evidence_class": "overlap_unknown",
        "notebook": reference_notebook,
        "datasets": {
            dataset_ref: {
                "dataset_id": 7,
                "version": 3,
                "last_updated_utc": "2026-09-08T00:00:00Z",
                "license": "CC0-1.0",
                "archive_sha256": "3" * 64,
                "required_members": {"weights.bin": "4" * 64},
                "minimum_wheel_count": 0,
            }
        },
        "competition": "biohub-cell-tracking-during-development",
        "submission_columns": SUBMISSION_COLUMNS,
    }
    artifact_lock: dict[str, object] = {
        "schema_version": 1,
        "reference": reference,
        "competition": "biohub-cell-tracking-during-development",
        "submission_columns": SUBMISSION_COLUMNS,
        "datasets": {dataset_ref: dataset_record},
    }
    artifact_lock["release_digest"] = canonical_digest(artifact_lock)
    write_json(package / "artifact-lock.json", artifact_lock)
    artifact_hash = sha256(package / "artifact-lock.json")

    metadata = {
        "id": "owner/e0-release",
        "title": "Synthetic E0 release fixture",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["owner/assets/3"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    write_json(package / "kernel-metadata.json", metadata)
    metadata_hash = sha256(package / "kernel-metadata.json")

    package_manifest = {
        "schema_version": 1,
        "status": "ready_for_root_review_not_launched",
        "release_digest": artifact_lock["release_digest"],
        "source_notebook": {**reference_notebook, "cell_count": 1, "cell_source_sha256": []},
        "packaged_notebook": {"path": "submission.ipynb", "sha256": notebook_hash},
        "kernel_metadata_sha256": metadata_hash,
        "artifact_lock_sha256": artifact_hash,
        "dataset_archives": {
            dataset_ref: {
                key: dataset_record[key]
                for key in (
                    "dataset_id",
                    "version",
                    "last_updated_utc",
                    "license",
                    "archive_file",
                    "archive_bytes",
                    "archive_sha256",
                    "file_count",
                    "wheel_count",
                )
            }
        },
        "algorithm_changes": [],
        "evidence_class": "overlap_unknown",
    }
    write_json(package / "package-manifest.json", package_manifest)
    identity = PackageIdentity(
        notebook_slug="owner/e0-release",
        notebook_title="Synthetic E0 release fixture",
        competition="biohub-cell-tracking-during-development",
        machine_shape="NvidiaTeslaT4",
        release_digest=str(artifact_lock["release_digest"]),
        package_manifest_sha256=sha256(package / "package-manifest.json"),
        kernel_metadata_sha256=metadata_hash,
        artifact_lock_sha256=artifact_hash,
        notebook_sha256=notebook_hash,
        cli_normalized_notebook_sha256=normalized_hash,
        dataset_versions=((dataset_ref, 3),),
    )
    return package, identity, artifact_lock


def submission_text() -> str:
    return (
        ",".join(SUBMISSION_COLUMNS)
        + "\n"
        + "0,a,node,10,0,1,2,3,-1,-1\n"
        + "1,a,node,11,1,1,2,3,-1,-1\n"
        + "2,a,node,12,1,1,3,3,-1,-1\n"
        + "3,a,edge,-1,-1,-1,-1,-1,10,11\n"
        + "4,a,edge,-1,-1,-1,-1,-1,10,12\n"
        + "5,b,node,10,0,0,1,1,-1,-1\n"
    )


def make_run_evidence(
    tmp_path: Path, artifact_lock: dict[str, object]
) -> tuple[Path, Path, ExactVersionOutputProof, dict[str, object]]:
    submission = tmp_path / "submission.csv"
    submission.write_text(submission_text(), encoding="utf-8", newline="")
    shapes = {"a": [2, 4, 5, 6], "b": [1, 2, 3, 4]}
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "reference": artifact_lock["reference"],
        "release_digest": artifact_lock["release_digest"],
        "input_integrity": {
            "owner/assets": {
                "status": "verified",
                "root": "/kaggle/input/assets",
                "version": 3,
                "files": 1,
                "bytes": 3,
            }
        },
        "actual_input_shapes_tzyx": shapes,
        "dataset_shapes": shapes,
        "expected_dataset_ids": ["a", "b"],
        "completed_dataset_ids": ["a", "b"],
        "csv_sha256": sha256(submission),
        "coordinate_bounds_valid": True,
        "graph_valid": True,
        "hidden_discovery_supported": True,
        "internet_enabled_configured": False,
        "submission": {
            "path": "/kaggle/working/submission.csv",
            "sha256": sha256(submission),
            "rows": 6,
            "datasets": {"a": {"nodes": 3, "edges": 2}, "b": {"nodes": 1, "edges": 0}},
            "validation": "PASS",
        },
        "effective_biohub_environment": {},
        "environment": {"python": "fixture"},
        "elapsed_seconds": 12.5,
        "full_runtime_seconds": 12.5,
    }
    manifest_path = tmp_path / "public_reference_run_manifest.json"
    write_json(manifest_path, manifest)
    proof = ExactVersionOutputProof(
        schema_version=1,
        kind="KAGGLE_EXACT_VERSION_OUTPUT_PROOF",
        status="PASS",
        transport_status="VERIFIED",
        notebook_slug="owner/e0-release",
        notebook_version=9,
        version_label="v9",
        release_digest=str(artifact_lock["release_digest"]),
        run_manifest_sha256=sha256(manifest_path),
        submission_sha256=sha256(submission),
        embedded_release_identity_verified=True,
        embedded_reference_verified=True,
        embedded_submission_hash_verified=True,
        output_files=[
            {
                "path": str(manifest_path),
                "basename": manifest_path.name,
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256(manifest_path),
            },
            {
                "path": str(submission),
                "basename": submission.name,
                "bytes": submission.stat().st_size,
                "sha256": sha256(submission),
            },
        ],
        expected_input_shapes_tzyx=shapes,
        shape_identity_status="VERIFIED",
        release_validation_status="UNRESOLVED_PENDING_INDEPENDENT_VALIDATOR",
    )
    return submission, manifest_path, proof, manifest


def rebind_manifest(proof: ExactVersionOutputProof, manifest_path: Path) -> ExactVersionOutputProof:
    files = [dict(record) for record in proof.output_files]
    manifest_record = next(
        record for record in files if record["basename"] == "public_reference_run_manifest.json"
    )
    manifest_record["bytes"] = manifest_path.stat().st_size
    manifest_record["sha256"] = sha256(manifest_path)
    return ExactVersionOutputProof(
        **{
            **proof.__dict__,
            "run_manifest_sha256": sha256(manifest_path),
            "output_files": files,
        }
    )


def test_valid_release_passes_identity_and_structure_but_blocks_missing_evidence(
    tmp_path: Path,
) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest, proof, _ = make_run_evidence(tmp_path, artifact_lock)

    result = validate_downloaded_e0_release(
        submission,
        manifest,
        package,
        exact_version_proof=proof,
        package_identity=identity,
    )

    assert result.status == "STRUCTURAL_AND_IDENTITY_PASS_ADMISSION_BLOCKED"
    assert result.identity_status == result.official_format_status == "PASS"
    assert result.scorer_compatibility_status == result.e0_lineage_contract_status == "PASS"
    assert result.row_count == 6
    assert [(item.dataset, item.forks) for item in result.datasets] == [("a", 1), ("b", 0)]
    assert result.quality_admission == "BLOCKED_MISSING_UPSTREAM_EVIDENCE"
    assert result.resource_admission == "BLOCKED_MISSING_UPSTREAM_EVIDENCE"
    assert "peak_extraction.capped_frame_count" in result.missing_quality_evidence
    assert "peak_vram_bytes" in result.missing_resource_evidence
    json.dumps(result.to_dict(), allow_nan=False)


def test_raw_csv_bytes_must_match_exact_version_proof(tmp_path: Path) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest, proof, _ = make_run_evidence(tmp_path, artifact_lock)
    submission.write_bytes(submission.read_bytes() + b"\n")

    with pytest.raises(ReleaseValidationError, match="proof submission hash mismatch"):
        validate_downloaded_e0_release(
            submission,
            manifest,
            package,
            exact_version_proof=proof,
            package_identity=identity,
        )


def test_distinct_zarr_paths_may_share_a_basename(tmp_path: Path) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest, proof, _ = make_run_evidence(tmp_path, artifact_lock)
    records = [
        *proof.output_files,
        *[
            {"path": path, "basename": "zarr.json", "bytes": 2, "sha256": "a" * 64}
            for path in ("graph-a/nodes/zarr.json", "graph-b/nodes/zarr.json")
        ],
    ]
    proof = ExactVersionOutputProof(**{**proof.__dict__, "output_files": records})
    result = validate_downloaded_e0_release(
        submission, manifest, package, exact_version_proof=proof, package_identity=identity
    )
    assert result.identity_status == "PASS"


@pytest.mark.parametrize("second_path", ["graph/zarr.json", "graph//zarr.json", "graph\\zarr.json"])
def test_duplicate_full_output_path_is_rejected(tmp_path: Path, second_path: str) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest, proof, _ = make_run_evidence(tmp_path, artifact_lock)
    records = [
        *proof.output_files,
        *[
            {"path": path, "basename": "zarr.json", "bytes": 2, "sha256": "a" * 64}
            for path in ("graph/zarr.json", second_path)
        ],
    ]
    proof = ExactVersionOutputProof(**{**proof.__dict__, "output_files": records})
    with pytest.raises(ReleaseValidationError, match="duplicate output path"):
        validate_downloaded_e0_release(
            submission, manifest, package, exact_version_proof=proof, package_identity=identity
        )


@pytest.mark.parametrize("nested_path", ["nested/submission.csv", "nested\\submission.csv"])
def test_nested_duplicate_submission_remains_ambiguous(tmp_path: Path, nested_path: str) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest, proof, _ = make_run_evidence(tmp_path, artifact_lock)
    duplicate = dict(next(row for row in proof.output_files if row["basename"] == "submission.csv"))
    duplicate["path"] = nested_path
    proof = ExactVersionOutputProof(
        **{**proof.__dict__, "output_files": [*proof.output_files, duplicate]}
    )
    with pytest.raises(ReleaseValidationError, match="inventory does not bind submission.csv"):
        validate_downloaded_e0_release(
            submission, manifest, package, exact_version_proof=proof, package_identity=identity
        )


def test_backslash_path_cannot_hide_required_basename(tmp_path: Path) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest, proof, _ = make_run_evidence(tmp_path, artifact_lock)
    duplicate = dict(next(row for row in proof.output_files if row["basename"] == "submission.csv"))
    duplicate["path"] = duplicate["basename"] = "nested\\submission.csv"
    proof = ExactVersionOutputProof(
        **{**proof.__dict__, "output_files": [*proof.output_files, duplicate]}
    )
    with pytest.raises(ReleaseValidationError, match="inventory is invalid"):
        validate_downloaded_e0_release(
            submission, manifest, package, exact_version_proof=proof, package_identity=identity
        )


def test_canonicalized_provider_identity_requires_verified_mapping_and_exact_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from biohub_ct.campaign import kaggle_canonicalization

    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest, proof, _ = make_run_evidence(tmp_path, artifact_lock)
    alias = identity.notebook_slug + "-canonical"
    proof = ExactVersionOutputProof(**{**proof.__dict__, "notebook_slug": alias})
    with pytest.raises(ReleaseValidationError, match="notebook slug"):
        validate_downloaded_e0_release(
            submission, manifest, package, exact_version_proof=proof, package_identity=identity
        )
    mapping_path = tmp_path / "canonicalization.json"
    write_json(mapping_path, {"test_boundary": True})
    seen = []

    def verified_mapping(path, *, identity):
        seen.append((path, identity))
        return SimpleNamespace(actual_notebook_slug=alias, notebook_version=proof.notebook_version)

    monkeypatch.setattr(
        kaggle_canonicalization, "verify_canonicalization_receipt", verified_mapping
    )
    result = validate_downloaded_e0_release(
        submission,
        manifest,
        package,
        exact_version_proof=proof,
        package_identity=identity,
        canonicalization_receipt=mapping_path,
    )
    assert result.notebook_slug == alias
    assert result.quality_admission == "BLOCKED_MISSING_UPSTREAM_EVIDENCE"
    assert seen == [(mapping_path, identity)]
    changed = ExactVersionOutputProof(
        **{**proof.__dict__, "notebook_version": proof.notebook_version + 1}
    )
    with pytest.raises(ReleaseValidationError, match="version differs from canonicalization"):
        validate_downloaded_e0_release(
            submission,
            manifest,
            package,
            exact_version_proof=changed,
            package_identity=identity,
            canonicalization_receipt=mapping_path,
        )


def test_bounds_require_verified_exact_version_shape_identity(tmp_path: Path) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest, proof, _ = make_run_evidence(tmp_path, artifact_lock)
    proof = ExactVersionOutputProof(
        **{
            **proof.__dict__,
            "expected_input_shapes_tzyx": None,
            "shape_identity_status": "UNAVAILABLE",
        }
    )

    with pytest.raises(ReleaseValidationError, match="no verified input shape identity"):
        validate_downloaded_e0_release(
            submission,
            manifest,
            package,
            exact_version_proof=proof,
            package_identity=identity,
        )


def test_manifest_source_identity_cannot_drift_even_with_rehashed_transport(tmp_path: Path) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest_path, proof, manifest = make_run_evidence(tmp_path, artifact_lock)
    manifest["reference"] = {**artifact_lock["reference"], "competition": "other"}  # type: ignore[dict-item]
    write_json(manifest_path, manifest)
    proof = rebind_manifest(proof, manifest_path)

    with pytest.raises(ReleaseValidationError, match="exact reference contract"):
        validate_downloaded_e0_release(
            submission,
            manifest_path,
            package,
            exact_version_proof=proof,
            package_identity=identity,
        )


def test_mounted_file_inventory_must_match_artifact_lock(tmp_path: Path) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest_path, proof, manifest = make_run_evidence(tmp_path, artifact_lock)
    integrity = manifest["input_integrity"]
    assert isinstance(integrity, dict)
    record = integrity["owner/assets"]
    assert isinstance(record, dict)
    record["bytes"] = 2
    write_json(manifest_path, manifest)
    proof = rebind_manifest(proof, manifest_path)

    with pytest.raises(ReleaseValidationError, match="mounted input identity mismatch"):
        validate_downloaded_e0_release(
            submission,
            manifest_path,
            package,
            exact_version_proof=proof,
            package_identity=identity,
        )


@pytest.mark.parametrize(
    ("placeholder", "expected_status", "expect_missing"),
    [
        (None, "BLOCKED_MISSING_UPSTREAM_EVIDENCE", True),
        (0, "NOT_EVALUATED", False),
    ],
)
def test_telemetry_presence_never_claims_admission(
    tmp_path: Path,
    placeholder: object,
    expected_status: str,
    expect_missing: bool,
) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest_path, proof, manifest = make_run_evidence(tmp_path, artifact_lock)
    manifest["peak_extraction"] = {
        "pre_cap_candidates": placeholder,
        "capped_frame_count": placeholder,
        "capped_frame_denominator": placeholder,
        "capped_clip_count": placeholder,
        "capped_clip_denominator": placeholder,
    }
    manifest["execution"] = {
        "output_fallback_count": placeholder,
        "retry_count": placeholder,
    }
    for key in (
        "stage_times_seconds",
        "peak_ram_bytes",
        "peak_vram_bytes",
        "verified_runtime_limit_seconds",
        "runtime_headroom_seconds",
        "measured_quota_debit",
    ):
        manifest[key] = placeholder
    write_json(manifest_path, manifest)
    proof = rebind_manifest(proof, manifest_path)

    result = validate_downloaded_e0_release(
        submission,
        manifest_path,
        package,
        exact_version_proof=proof,
        package_identity=identity,
    )

    assert result.quality_admission == expected_status
    assert result.resource_admission == expected_status
    assert bool(result.missing_quality_evidence) is expect_missing
    assert bool(result.missing_resource_evidence) is expect_missing


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "0,a,node,1,0,0,0,0,-1,-1\n1,a,node,1,1,0,0,0,-1,-1\n",
            "duplicate node id",
        ),
        ("0,a,node,1,0,0,0,0,-1,-1\n1,a,edge,-1,-1,-1,-1,-1,1,9\n", "endpoint"),
        (
            "0,a,node,1,0,0,0,0,-1,-1\n1,a,node,2,2,0,0,0,-1,-1\n2,a,edge,-1,-1,-1,-1,-1,1,2\n",
            "frame gap",
        ),
        (
            (
                "0,a,node,1,0,0,0,0,-1,-1\n1,a,node,2,0,0,0,0,-1,-1\n"
                "2,a,node,3,1,0,0,0,-1,-1\n3,a,edge,-1,-1,-1,-1,-1,1,3\n"
                "4,a,edge,-1,-1,-1,-1,-1,2,3\n"
            ),
            "multiple parents",
        ),
        (
            (
                "0,a,node,1,0,0,0,0,-1,-1\n1,a,node,2,1,0,0,0,-1,-1\n"
                "2,a,node,3,1,0,0,0,-1,-1\n3,a,node,4,1,0,0,0,-1,-1\n"
                "4,a,edge,-1,-1,-1,-1,-1,1,2\n5,a,edge,-1,-1,-1,-1,-1,1,3\n"
                "6,a,edge,-1,-1,-1,-1,-1,1,4\n"
            ),
            "more than two children",
        ),
        (
            (
                "0,a,node,1,0,0,0,0,-1,-1\n1,a,node,2,1,0,0,0,-1,-1\n"
                "2,a,edge,-1,-1,-1,-1,-1,1,2\n3,a,edge,-1,-1,-1,-1,-1,1,2\n"
            ),
            "duplicates edge",
        ),
        ("0,a,node,1,0,0,0,4,-1,-1\n", "out of bounds"),
        ("00,a,node,1,0,0,0,0,-1,-1\n", "canonical integer"),
    ],
)
def test_csv_corner_cases_fail_closed(tmp_path: Path, body: str, message: str) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text(",".join(SUBMISSION_COLUMNS) + "\n" + body, encoding="utf-8")

    with pytest.raises(ReleaseValidationError, match=message):
        release_validation._parse_submission(submission, {"a": (3, 2, 3, 4)})


def test_dataset_coverage_and_scope_are_exact(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text(
        ",".join(SUBMISSION_COLUMNS) + "\n0,a,node,1,0,0,0,0,-1,-1\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseValidationError, match="coverage mismatch"):
        release_validation._parse_submission(submission, {"a": (1, 1, 1, 1), "b": (1, 1, 1, 1)})


def test_package_bytes_are_rechecked(tmp_path: Path) -> None:
    package, identity, artifact_lock = make_package(tmp_path)
    submission, manifest, proof, _ = make_run_evidence(tmp_path, artifact_lock)
    with (package / "submission.ipynb").open("ab") as stream:
        stream.write(b" ")

    with pytest.raises(ReleaseValidationError, match="package preflight failed"):
        validate_downloaded_e0_release(
            submission,
            manifest,
            package,
            exact_version_proof=proof,
            package_identity=identity,
        )
