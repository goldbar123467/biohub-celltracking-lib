from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from scripts import verify_e0_telemetry_outputs as verifier

from biohub_ct.campaign.e0_notebook_telemetry import (
    _COUNTER_SEMANTICS,
    NotebookTelemetryConfig,
    harvest_e0_telemetry,
)
from e0_package_fixtures import SyntheticE0Package, build_e0_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path("synthetic-package-is-installed-by-fixture")
PRODUCTION_ROOT = "/kaggle/input/biohub-cell-tracking-during-development/test"
STAGES = (
    "data_read",
    "encode_tta",
    "detector_extraction",
    "pair_score",
    "threshold",
    "graph_build",
    "ilp",
    "geff",
)


@pytest.fixture(autouse=True)
def synthetic_r4_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticE0Package:
    package = build_e0_package(tmp_path / "package-fixture", generation="r4")
    monkeypatch.setattr(verifier, "E0_R4_PACKAGE_IDENTITY", package.identity)
    monkeypatch.setitem(globals(), "PACKAGE", package.path)
    return package


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
        newline="\n",
    )


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _package_contract() -> tuple[dict[str, object], dict[str, object]]:
    return (
        json.loads((PACKAGE / "package-manifest.json").read_text(encoding="utf-8")),
        json.loads((PACKAGE / "artifact-lock.json").read_text(encoding="utf-8")),
    )


def _portable_harvest(output: Path) -> dict[str, object]:
    support_root = output / "e0_support_telemetry"
    launch_path = support_root / "launch-events.jsonl"
    event_path = support_root / "invocation-100-1" / "events.jsonl"
    coordinate_path = support_root / "invocation-100-1" / "coordinates" / "movie-a.i2"
    durations = {
        stage: {"status": "available", "value": 0.25, "unit": "seconds", "duration_ns": 250_000_000}
        for stage in STAGES
    }
    process_resources = {
        "cuda_peak_allocated": {"status": "available", "value": 1024, "unit": "bytes"},
        "cuda_peak_reserved": {"status": "available", "value": 2048, "unit": "bytes"},
        "host_ru_maxrss": {
            "status": "available",
            "value": 4,
            "unit": "kibibytes",
            "normalized_bytes": 4096,
        },
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "shapes_tzyx": {"movie-a": [1, 2, 3, 4]},
        "support": {
            "status": "PASS",
            "production_data_root": PRODUCTION_ROOT,
            "production_test_names": ["movie-a"],
            "event_files": [
                {
                    "path": "/kaggle/working/e0_support_telemetry/invocation-100-1/events.jsonl",
                    "bytes": event_path.stat().st_size,
                    "sha256": sha256(event_path),
                    "process_id": "100-1",
                }
            ],
            "launch_evidence": {
                "path": "/kaggle/working/e0_support_telemetry/launch-events.jsonl",
                "bytes": launch_path.stat().st_size,
                "sha256": sha256(launch_path),
                "source_chain": {},
                "launch_count": 1,
            },
            "selected_invocations": [
                {
                    "invocation_id": "100-1-1",
                    "process_id": "100-1",
                    "gpu_shard": "single",
                    "diagnostic_arm": "production",
                    "identity": {
                        "data_root": PRODUCTION_ROOT,
                        "output_dir": "/kaggle/working/predictions/user/public-reference/split_0",
                        "method": "public-reference",
                        "fold": 0,
                        "test_names": ["movie-a"],
                        "invocation_id": "100-1-1",
                    },
                    "outcome": "returned",
                    "resource_scope": "production_shard_process",
                    "resources": process_resources,
                    "process_summary": {},
                }
            ],
            "datasets": {
                "movie-a": {
                    "invocation_id": "100-1-1",
                    "process_id": "100-1",
                    "gpu_shard": "single",
                    "diagnostic_arm": "production",
                    "dataset_path": f"{PRODUCTION_ROOT}/movie-a",
                    "durations": durations,
                    "counts": {},
                    "ilp": {"status": "returned"},
                }
            },
            "ignored_nonproduction_invocation_count": 0,
        },
        "retention": {"status": "PASS"},
        "coordinates": {
            "status": "PASS",
            "datasets": {
                "movie-a": {
                    "rows": 1,
                    "artifact": {
                        "status": "VERIFIED",
                        "path": (
                            "/kaggle/working/e0_support_telemetry/"
                            "invocation-100-1/coordinates/movie-a.i2"
                        ),
                        "bytes": coordinate_path.stat().st_size,
                        "sha256": sha256(coordinate_path),
                        "dense_frame_counts": [1],
                    },
                }
            },
        },
        "run_stats": {
            "status": "PASS",
            "path": "/kaggle/working/run_stats.csv",
            "sha256": sha256(output / "run_stats.csv"),
            "counter_semantics": {},
            "datasets": {},
        },
        "reconciliations": {},
        "submission": {
            "status": "VERIFIED",
            "path": "/kaggle/working/submission.csv",
            "sha256": sha256(output / "submission.csv"),
            "counts": {},
        },
    }


def make_output(tmp_path: Path) -> Path:
    package_manifest, artifact_lock = _package_contract()
    output = tmp_path / "downloaded"
    support = output / "e0_support_telemetry"
    process = support / "invocation-100-1"
    coordinates = process / "coordinates"
    coordinates.mkdir(parents=True)
    (output / "submission.csv").write_text("id,dataset\n0,movie-a\n", encoding="utf-8")
    (output / "run_stats.csv").write_text("dataset,nodes\nmovie-a,1\n", encoding="utf-8")
    (coordinates / "movie-a.i2").write_bytes(b"\x00" * 8)
    launch_records = [
        {"event": "SUPPORT_PATCHED"},
        {
            "event": "SHARD_LAUNCHED",
            "pid": 100,
            "argv": [
                "python",
                "scripts/predict_unet_transformer.py",
                "--data-dir",
                PRODUCTION_ROOT,
                "--split",
                "0",
            ],
        },
        {
            "event": "LAUNCH_OBSERVATION_FINISHED",
            "status": "PASS",
            "launch_count": 1,
            "expected_launch_count": 1,
        },
    ]
    write_jsonl(support / "launch-events.jsonl", launch_records)
    write_jsonl(
        process / "events.jsonl",
        [
            {
                "schema_version": 1,
                "record_type": "coordinate_artifact",
                "process_id": "100-1",
                "pid": 100,
                "gpu_shard": "single",
                "diagnostic_arm": "production",
                "invocation_id": "100-1-1",
                "dataset": "movie-a",
                "dataset_path": f"{PRODUCTION_ROOT}/movie-a",
                "artifact": {
                    "path": "e0_support_telemetry/invocation-100-1/coordinates/movie-a.i2"
                },
            }
        ],
    )
    callback_hash = artifact_lock["instrumentation"]["config"][  # type: ignore[index]
        "callback_after_public_cell_sha256"
    ]
    write_json(
        support / "launcher-callback-status.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "callback_after_public_cell_sha256": callback_hash,
            "launch_count": 1,
            "expected_launch_count": 1,
            "cleanup_status": "PASS",
            "cleanup_errors": [],
        },
    )

    generated = package_manifest["instrumentation"]["generated_cell_sha256"]  # type: ignore[index]
    public = package_manifest["source_notebook"]["cell_source_sha256"]  # type: ignore[index]
    identities = [
        ("instrumentation", generated["interceptor_install"], None),
        ("instrumentation", generated["input_integrity"], None),
        *(("public", digest, index) for index, digest in enumerate(public)),
        ("instrumentation", generated["base_validation"], None),
        ("instrumentation", generated["interceptor_close"], None),
    ]
    cells: list[dict[str, object]] = []
    for role, digest, public_index in identities:
        row: dict[str, object] = {
            "role": role,
            "sha256": digest,
            "started_at_utc": "2026-09-08T08:00:00+00:00",
            "ended_at_utc": "2026-09-08T08:00:01+00:00",
            "elapsed_seconds": 1.0,
            "post_sha256": digest,
            "post_hash_matches_pre": True,
            "status": "PASS",
            "error_type": None,
        }
        if public_index is not None:
            row["public_index"] = public_index
        cells.append(row)
    write_jsonl(output / "e0_cell_events.jsonl", cells)

    sample = {
        "schema_version": 1,
        "captured_at_utc": "2026-09-08T08:00:00+00:00",
        "monotonic_seconds": 10.0,
        "host": {"status": "MEASURED", "rss_bytes": 4096},
        "gpus": {
            "status": "MEASURED",
            "devices": [{"index": 0, "uuid": "GPU-0", "used_bytes": 1024, "total_bytes": 8192}],
        },
    }
    write_jsonl(output / "e0_resource_samples.jsonl", [sample])
    harvest = _portable_harvest(output)
    telemetry = {
        "schema_version": 1,
        "status": "PASS",
        "started_at_utc": "2026-09-08T08:00:00+00:00",
        "finalized_at_utc": "2026-09-08T08:01:40+00:00",
        "runtime_scopes": {
            "wrapper": {"status": "MEASURED", "elapsed_seconds": 100.0},
            "cells": {"status": "MEASURED", "records": cells, "violations": []},
            "submission_csv_serialization": {
                "status": "MEASURED",
                "reason": None,
                "records": [{"status": "PASS", "elapsed_seconds": 0.5}],
            },
            "provider": {
                "status": "UNAVAILABLE",
                "reason": "provider-complete elapsed is external to notebook execution",
                "elapsed_seconds": None,
            },
        },
        "resources": {
            "scope": "whole_notebook_wrapper",
            "status": "PASS",
            "coverage_status": "COMPLETE",
            "cleanup_complete": True,
            "sample_count": 1,
            "dropped_samples": 0,
            "write_errors": 0,
            "sample_interval_seconds": 5.0,
            "sampled_until_monotonic_seconds": 10.0,
            "samples_path": "/kaggle/working/e0_resource_samples.jsonl",
            "samples_sha256": sha256(output / "e0_resource_samples.jsonl"),
            "host_peak": {"status": "MEASURED", "rss_bytes": 4096},
            "gpu_peaks": {
                "status": "MEASURED",
                "used_bytes_by_uuid": {"GPU-0": 1024},
                "total_bytes_by_uuid": {"GPU-0": 8192},
            },
        },
        "harvest": harvest,
    }
    write_json(output / "e0_notebook_telemetry.json", telemetry)
    write_json(
        output / "public_reference_run_manifest.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "actual_input_shapes_tzyx": {"movie-a": [1, 2, 3, 4]},
            "dataset_shapes": {"movie-a": [1, 2, 3, 4]},
            "expected_dataset_ids": ["movie-a"],
            "completed_dataset_ids": ["movie-a"],
            "e0_telemetry": {
                "schema_version": 1,
                "status": "PASS",
                "path": "/kaggle/working/e0_notebook_telemetry.json",
                "sha256": sha256(output / "e0_notebook_telemetry.json"),
                "wrapper_elapsed_seconds": 100.0,
                "provider_elapsed_seconds": None,
                "provider_elapsed_status": "UNAVAILABLE_INSIDE_NOTEBOOK",
            },
        },
    )
    return output


def _install_real_harvester_evidence(output: Path, shadow_root: Path) -> None:
    support = output / "e0_support_telemetry"
    process = support / "invocation-100-1"
    coordinate_path = process / "coordinates" / "movie-a.i2"
    coordinate_hash = sha256(coordinate_path)
    identity = {
        "invocation_id": "100-1-1",
        "data_root": PRODUCTION_ROOT,
        "output_dir": "/kaggle/working/predictions/user/public-reference/split_0",
        "method": "public-reference",
        "fold": 0,
        "test_names": ["movie-a"],
    }
    envelope = {
        "schema_version": 1,
        "process_id": "100-1",
        "pid": 100,
        "gpu_shard": "single",
        "diagnostic_arm": "production",
        "invocation_id": identity["invocation_id"],
    }
    artifact = {
        "path": "e0_support_telemetry/invocation-100-1/coordinates/movie-a.i2",
        "bytes": coordinate_path.stat().st_size,
        "dtype": "<i2",
        "shape": [1, 4],
        "sha256": coordinate_hash,
    }
    coordinate = {
        "dataset": "movie-a",
        "dataset_path": f"{PRODUCTION_ROOT}/movie-a",
        "columns": ["t", "z", "y", "x"],
        "dtype": "<i2",
        "stage": "post_detection_pre_graph_pre_ilp",
        "rows": 1,
        "coordinate_sha256": coordinate_hash,
        "frame_counts": [[0, 1]],
        "artifact": artifact,
    }
    durations = {
        stage: {
            "status": "available",
            "value": 0.25,
            "unit": "seconds",
            "duration_ns": 250_000_000,
        }
        for stage in STAGES
    }
    counts = {
        "pair_universe": {
            "status": "available",
            "value": 0,
            "unit": "ordered_source_target_pairs",
        },
        "threshold_passing_edge_candidates": {
            "status": "available",
            "value": 0,
            "unit": "edges",
        },
        "detected_nodes": {"status": "available", "value": 1, "unit": "nodes"},
        "pre_ilp_edges": {"status": "available", "value": 0, "unit": "edges"},
        "output_edges": {"status": "available", "value": 0, "unit": "edges"},
    }
    resources = {
        "cuda_peak_allocated": {"status": "available", "value": 1024, "unit": "bytes"},
        "cuda_peak_reserved": {"status": "available", "value": 2048, "unit": "bytes"},
        "host_ru_maxrss": {
            "status": "available",
            "value": 4,
            "unit": "kibibytes",
            "normalized_bytes": 4096,
        },
    }
    write_jsonl(
        process / "events.jsonl",
        [
            {**envelope, "record_type": "invocation_start", "identity": identity},
            {
                **envelope,
                "record_type": "retention",
                "dataset": "movie-a",
                "dataset_path": f"{PRODUCTION_ROOT}/movie-a",
                "frame": 0,
                "primary_candidates": 1,
                "blended_candidates": 1,
                "retention": 1.0,
                "minimum_retention": 0.9,
                "use_primary": False,
            },
            {**envelope, "record_type": "coordinate_artifact", **coordinate},
            {
                **envelope,
                "record_type": "dataset_summary",
                "dataset": "movie-a",
                "dataset_path": f"{PRODUCTION_ROOT}/movie-a",
                "outcome": "returned",
                "durations": durations,
                "counts": counts,
                "ilp": {"status": "returned"},
                "coordinate_artifact": {"status": "available", **coordinate},
            },
            {
                **envelope,
                "record_type": "invocation_finish",
                "identity": identity,
                "outcome": "returned",
            },
            {
                **{key: value for key, value in envelope.items() if key != "invocation_id"},
                "invocation_id": None,
                "record_type": "process_summary",
                "invocation_ids": [identity["invocation_id"]],
                **resources,
            },
        ],
    )
    source_hash = "c" * 64
    launch = [
        {
            "event": "SUPPORT_PATCHED",
            "source_chain": {
                "pre_public_support_sha256": "a" * 64,
                "public_patched_support_sha256": "b" * 64,
                "telemetry_output_support_sha256": source_hash,
                "runtime_module_sha256": "d" * 64,
            },
        },
        {
            "event": "SHARD_LAUNCHED",
            "pid": 100,
            "argv": [
                "python",
                "scripts/predict_unet_transformer.py",
                "--data-dir",
                PRODUCTION_ROOT,
                "--output-dir",
                identity["output_dir"],
                "--method",
                "public-reference",
                "--split",
                "0",
            ],
            "source_sha256": source_hash,
        },
        {
            "event": "LAUNCH_OBSERVATION_FINISHED",
            "status": "PASS",
            "launch_count": 1,
            "expected_launch_count": 1,
        },
    ]
    write_jsonl(support / "launch-events.jsonl", launch)
    fields = ["dataset", "raw_nodes", "raw_edges", "nodes", "edges", *_COUNTER_SEMANTICS]
    stats = {name: 0 for name in fields}
    stats.update({"dataset": "movie-a", "raw_nodes": 1, "nodes": 1})
    (output / "run_stats.csv").write_text(
        ",".join(fields) + "\n" + ",".join(str(stats[name]) for name in fields) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "submission.csv").write_text(
        "id,dataset,row_type\n0,movie-a,node\n", encoding="utf-8", newline="\n"
    )

    package_manifest, _artifact_lock = _package_contract()
    launch_records = verifier._strict_jsonl(support / "launch-events.jsonl", "launch")
    portable = shadow_root / "recorded-harvest"
    portable.mkdir()
    shadow_output, provider_root, _mapped, originals = verifier._build_harvest_shadow(
        output, portable, launch_records
    )
    mapped_root = str(provider_root.joinpath(*Path(PRODUCTION_ROOT).parts[1:]))
    source = package_manifest["source_notebook"]
    instrumentation = package_manifest["instrumentation"]
    config = NotebookTelemetryConfig(
        output_dir=str(shadow_output.resolve()),
        expected_public_cell_sha256=tuple(source["cell_source_sha256"]),
        allowed_auxiliary_cell_sha256=tuple(instrumentation["allowed_auxiliary_cell_sha256"]),
        expected_shapes_tzyx={"movie-a": [1, 2, 3, 4]},
        production_data_root=mapped_root,
        production_test_names=("movie-a",),
        sample_interval_seconds=5.0,
        nvidia_smi_timeout_seconds=2.0,
        max_samples=20_000,
        require_coordinate_artifacts=True,
        require_submission_crosscheck=True,
    )
    recorded_harvest = harvest_e0_telemetry(
        config, production_data_root=mapped_root, production_test_names=("movie-a",)
    )
    verifier._restore_original_event_evidence(recorded_harvest, originals)
    recorded_harvest = verifier._normalize_paths(recorded_harvest, shadow_output, provider_root)
    telemetry_path = output / "e0_notebook_telemetry.json"
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["harvest"] = recorded_harvest
    write_json(telemetry_path, telemetry)
    manifest_path = output / "public_reference_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["e0_telemetry"]["sha256"] = sha256(telemetry_path)
    write_json(manifest_path, manifest)


def fake_harvester(output: Path):
    def run(config, *, production_data_root, production_test_names):
        assert Path(config.output_dir).is_absolute()
        assert Path(production_data_root).is_absolute()
        assert production_test_names == ("movie-a",)
        raw = json.loads((output / "e0_notebook_telemetry.json").read_text())["harvest"]

        def adapt(value):
            if isinstance(value, dict):
                return {key: adapt(child) for key, child in value.items()}
            if isinstance(value, list):
                return [adapt(child) for child in value]
            if isinstance(value, str) and value == PRODUCTION_ROOT:
                return production_data_root
            if isinstance(value, str) and value.startswith(PRODUCTION_ROOT + "/"):
                return str(Path(production_data_root) / value[len(PRODUCTION_ROOT) + 1 :])
            if isinstance(value, str) and value == "/kaggle/working":
                return config.output_dir
            if isinstance(value, str) and value.startswith("/kaggle/working/"):
                return str(Path(config.output_dir) / value[len("/kaggle/working/") :])
            return value

        return adapt(raw)

    return run


def test_complete_r4_telemetry_writes_exclusive_report_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    output = make_output(tmp_path)
    before = tree_hashes(output)
    report_path = tmp_path / "verification.json"

    report = verifier.verify_downloaded_e0_telemetry(
        output_dir=output,
        package_dir=PACKAGE,
        report_json=report_path,
        harvester=fake_harvester(output),
        now_factory=lambda: datetime(2026, 9, 8, 9, 0, tzinfo=UTC),
    )

    assert report["status"] == "PASS"
    assert report["reviewed_package_generation"] == "E0_R4"
    assert report["production_data_root"] == PRODUCTION_ROOT
    assert report["cell_coverage"] == {
        "status": "PASS",
        "record_count": 16,
        "public_cell_count": 12,
    }
    assert report["runtime_scopes"]["provider_elapsed_seconds"] is None
    assert report["support_stage_coverage"]["stages"] == list(STAGES)
    assert report_path.is_file()
    assert json.loads(report_path.read_text()) == report
    assert tree_hashes(output) == before


def test_real_harvester_shadow_preserves_original_paths_hashes_and_inputs(
    tmp_path: Path,
) -> None:
    output = make_output(tmp_path)
    _install_real_harvester_evidence(output, tmp_path)
    before = tree_hashes(output)

    report = verifier.verify_downloaded_e0_telemetry(
        output_dir=output,
        package_dir=PACKAGE,
        report_json=tmp_path / "real-harvester-verification.json",
        now_factory=lambda: datetime(2026, 9, 8, 9, 0, tzinfo=UTC),
    )

    assert report["status"] == "PASS"
    assert report["offline_reharvest_status"] == "PASS"
    assert tree_hashes(output) == before
    reported_files = {row["path"]: row["sha256"] for row in report["input_files"]}
    assert reported_files == before
    recorded = json.loads((output / "e0_notebook_telemetry.json").read_text())
    event = recorded["harvest"]["support"]["event_files"][0]
    assert event["path"] == ("/kaggle/working/e0_support_telemetry/invocation-100-1/events.jsonl")
    assert event["sha256"] == before["e0_support_telemetry/invocation-100-1/events.jsonl"]


def test_reports_partial_sensor_availability_without_rejecting_measured_maxima(
    tmp_path: Path,
) -> None:
    output = make_output(tmp_path)
    samples_path = output / "e0_resource_samples.jsonl"
    samples = [
        {
            "schema_version": 1,
            "captured_at_utc": "2026-09-08T08:00:00+00:00",
            "monotonic_seconds": 10.0,
            "host": {"status": "MEASURED", "rss_bytes": 4096},
            "gpus": {
                "status": "MEASURED",
                "devices": [{"index": 0, "uuid": "GPU-0", "used_bytes": 1024, "total_bytes": 8192}],
            },
        },
        {
            "schema_version": 1,
            "captured_at_utc": "2026-09-08T08:00:05+00:00",
            "monotonic_seconds": 15.0,
            "host": {"status": "UNAVAILABLE", "reason": "transient host probe failure"},
            "gpus": {"status": "UNAVAILABLE", "reason": "transient GPU probe failure"},
        },
    ]
    write_jsonl(samples_path, samples)
    telemetry_path = output / "e0_notebook_telemetry.json"
    telemetry = json.loads(telemetry_path.read_text())
    telemetry["resources"].update(
        {
            "sample_count": 2,
            "sampled_until_monotonic_seconds": 15.0,
            "samples_sha256": sha256(samples_path),
        }
    )
    write_json(telemetry_path, telemetry)
    manifest_path = output / "public_reference_run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["e0_telemetry"]["sha256"] = sha256(telemetry_path)
    write_json(manifest_path, manifest)

    report = verifier.verify_downloaded_e0_telemetry(
        output_dir=output,
        package_dir=PACKAGE,
        report_json=tmp_path / "partial-sensor-verification.json",
        harvester=fake_harvester(output),
    )

    resources = report["whole_wrapper_resources"]
    assert resources["host_sensor_coverage"] == {
        "measured_sample_count": 1,
        "unavailable_sample_count": 1,
    }
    assert resources["gpu_sensor_coverage"] == {
        "measured_sample_count": 1,
        "unavailable_sample_count": 1,
    }


def test_rejects_evidence_that_changes_during_verification(tmp_path: Path) -> None:
    output = make_output(tmp_path)
    delegated = fake_harvester(output)

    def mutating_harvester(*args, **kwargs):
        result = delegated(*args, **kwargs)
        with (output / "submission.csv").open("a", encoding="utf-8", newline="") as stream:
            stream.write("1,movie-a\n")
        return result

    report_path = tmp_path / "must-not-exist.json"
    with pytest.raises(
        verifier.TelemetryVerificationError, match="evidence changed during verification"
    ):
        verifier.verify_downloaded_e0_telemetry(
            output_dir=output,
            package_dir=PACKAGE,
            report_json=report_path,
            harvester=mutating_harvester,
        )
    assert not report_path.exists()


@pytest.mark.parametrize(
    "failure", ["phase", "callback", "host_peak", "cuda_peak", "wrapper", "sampler_phase"]
)
def test_partial_or_cleanup_evidence_fails_without_writing_report(
    tmp_path: Path, failure: str
) -> None:
    output = make_output(tmp_path)
    if failure == "phase":
        path = output / "e0_notebook_telemetry.json"
        telemetry = json.loads(path.read_text())
        telemetry["harvest"]["support"]["datasets"]["movie-a"]["durations"]["ilp"] = {
            "status": "unavailable",
            "reason": "not_reached",
        }
        write_json(path, telemetry)
        manifest = json.loads((output / "public_reference_run_manifest.json").read_text())
        manifest["e0_telemetry"]["sha256"] = sha256(path)
        write_json(output / "public_reference_run_manifest.json", manifest)
    elif failure == "callback":
        path = output / "e0_support_telemetry/launcher-callback-status.json"
        callback = json.loads(path.read_text())
        callback["cleanup_status"] = "ERROR"
        callback["cleanup_errors"] = ["interceptor close failed"]
        write_json(path, callback)
    elif failure == "host_peak":
        path = output / "e0_notebook_telemetry.json"
        telemetry = json.loads(path.read_text())
        telemetry["resources"]["host_peak"] = {
            "status": "UNAVAILABLE",
            "reason": "no successful samples",
        }
        write_json(path, telemetry)
        manifest = json.loads((output / "public_reference_run_manifest.json").read_text())
        manifest["e0_telemetry"]["sha256"] = sha256(path)
        write_json(output / "public_reference_run_manifest.json", manifest)
    elif failure == "cuda_peak":
        path = output / "e0_notebook_telemetry.json"
        telemetry = json.loads(path.read_text())
        telemetry["harvest"]["support"]["selected_invocations"][0]["resources"][
            "cuda_peak_allocated"
        ] = {"status": "unavailable", "reason": "not_recorded"}
        write_json(path, telemetry)
        manifest = json.loads((output / "public_reference_run_manifest.json").read_text())
        manifest["e0_telemetry"]["sha256"] = sha256(path)
        write_json(output / "public_reference_run_manifest.json", manifest)
    elif failure == "wrapper":
        path = output / "e0_notebook_telemetry.json"
        telemetry = json.loads(path.read_text())
        telemetry["runtime_scopes"]["wrapper"] = {
            "status": "UNAVAILABLE",
            "reason": "not_recorded",
        }
        write_json(path, telemetry)
        manifest = json.loads((output / "public_reference_run_manifest.json").read_text())
        manifest["e0_telemetry"]["sha256"] = sha256(path)
        manifest["e0_telemetry"]["wrapper_elapsed_seconds"] = None
        write_json(output / "public_reference_run_manifest.json", manifest)
    else:
        samples_path = output / "e0_resource_samples.jsonl"
        samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
        samples[0]["captured_at_utc"] = "2026-09-08T07:59:59+00:00"
        write_jsonl(samples_path, samples)
        path = output / "e0_notebook_telemetry.json"
        telemetry = json.loads(path.read_text())
        telemetry["resources"]["samples_sha256"] = sha256(samples_path)
        write_json(path, telemetry)
        manifest = json.loads((output / "public_reference_run_manifest.json").read_text())
        manifest["e0_telemetry"]["sha256"] = sha256(path)
        write_json(output / "public_reference_run_manifest.json", manifest)

    report_path = tmp_path / "must-not-exist.json"
    with pytest.raises(verifier.TelemetryVerificationError):
        verifier.verify_downloaded_e0_telemetry(
            output_dir=output,
            package_dir=PACKAGE,
            report_json=report_path,
            harvester=fake_harvester(output),
        )
    assert not report_path.exists()


def test_report_path_must_be_new_and_outside_evidence_tree(tmp_path: Path) -> None:
    output = make_output(tmp_path)
    existing = tmp_path / "existing.json"
    existing.write_text("occupied", encoding="utf-8")
    with pytest.raises(verifier.TelemetryVerificationError, match="new path"):
        verifier.verify_downloaded_e0_telemetry(
            output_dir=output,
            package_dir=PACKAGE,
            report_json=existing,
            harvester=fake_harvester(output),
        )
    with pytest.raises(verifier.TelemetryVerificationError, match="cannot modify evidence"):
        verifier.verify_downloaded_e0_telemetry(
            output_dir=output,
            package_dir=PACKAGE,
            report_json=output / "report.json",
            harvester=fake_harvester(output),
        )


def test_tampered_package_source_is_rejected_before_evidence_validation(
    tmp_path: Path, synthetic_r4_package: SyntheticE0Package
) -> None:
    output = make_output(tmp_path)
    with (synthetic_r4_package.path / "submission.ipynb").open("a", encoding="utf-8") as stream:
        stream.write(" ")

    with pytest.raises(
        verifier.TelemetryVerificationError,
        match="does not match compiled E0 R4 identity",
    ):
        verifier.verify_downloaded_e0_telemetry(
            output_dir=output,
            package_dir=synthetic_r4_package.path,
            report_json=tmp_path / "must-not-exist.json",
            harvester=fake_harvester(output),
        )


def test_cli_has_no_generation_or_identity_override() -> None:
    parsed = verifier.parse_args(["--output-dir", "downloaded", "--report-json", "report.json"])
    assert parsed.package_dir == ROOT / "work/e0-reference/package-r4-title-fixed-a"
    with pytest.raises(SystemExit):
        verifier.parse_args(
            [
                "--output-dir",
                "downloaded",
                "--report-json",
                "report.json",
                "--generation",
                "r3",
            ]
        )
