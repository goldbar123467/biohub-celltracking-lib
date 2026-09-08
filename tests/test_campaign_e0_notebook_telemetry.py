from __future__ import annotations

import csv
import hashlib
import json
import struct
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from biohub_ct.campaign.e0_notebook_telemetry import (
    _COUNTER_SEMANTICS,
    E0TelemetryError,
    NotebookTelemetryConfig,
    NotebookTelemetryTracker,
    ResourceSampler,
    _read_nvidia_smi_snapshot,
    build_notebook_telemetry_sources,
    harvest_e0_telemetry,
)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_evidence(
    output: Path,
    *,
    missing_retention_frame: bool = False,
    corrupt_coordinate_hash: bool = False,
    raw_nodes: int = 2,
    final_nodes: int = 2,
    submission_nodes: int = 2,
    include_coordinate_artifact: bool = True,
    include_nonproduction_invocation: bool = False,
    malformed_process_resource: bool = False,
) -> None:
    output.mkdir(parents=True)
    shapes = {"movie-a": [2, 3, 4, 5]}
    write_json(
        output / "public_reference_run_manifest.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "actual_input_shapes_tzyx": shapes,
            "elapsed_seconds": 123.0,
            "full_runtime_seconds": 123.0,
        },
    )
    retention = [
        {
            "dataset": "movie-a",
            "frame": 0,
            "primary_candidates": 0,
            "blended_candidates": 0,
            "retention": 1.0,
            "minimum_retention": 0.9,
            "use_primary": False,
        },
        {
            "dataset": "movie-a",
            "frame": 1,
            "primary_candidates": 2,
            "blended_candidates": 1,
            "retention": 0.5,
            "minimum_retention": 0.9,
            "use_primary": True,
        },
    ]
    if missing_retention_frame:
        retention.pop()
    coordinates = [(0, 1, 2, 3), (1, 2, 3, 4)]
    coordinate_bytes = b"".join(struct.pack("<hhhh", *row) for row in coordinates)
    coordinate_hash = sha_bytes(coordinate_bytes)
    coordinate_record: dict[str, object] = {
        "columns": ["t", "z", "y", "x"],
        "coordinate_sha256": "f" * 64 if corrupt_coordinate_hash else coordinate_hash,
        "dataset": "movie-a",
        "dtype": "<i2",
        "frame_counts": [[0, 1], [1, 1]],
        "rows": 2,
        "stage": "post_detection_pre_graph_pre_ilp",
    }
    support_root = output / "e0_support_telemetry"
    process_dir = support_root / "invocation-production"
    process_dir.mkdir(parents=True)
    if include_coordinate_artifact:
        artifact_path = process_dir / "coordinates" / "movie-a.i2"
        artifact_path.parent.mkdir()
        artifact_path.write_bytes(coordinate_bytes)
        coordinate_record["artifact"] = {
            "path": "e0_support_telemetry/invocation-production/coordinates/movie-a.i2",
            "bytes": len(coordinate_bytes),
            "dtype": "<i2",
            "shape": [2, 4],
            "sha256": coordinate_hash,
        }
    data_root = (output / "test").resolve()
    data_root.mkdir()
    output_dir = (output / "predictions" / "public-reference" / "split_0").resolve()
    identity = {
        "invocation_id": "production-invocation",
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "method": "public-reference",
        "fold": 0,
        "test_names": ["movie-a"],
    }
    envelope = {
        "schema_version": 1,
        "process_id": "production-process",
        "pid": 7001,
        "gpu_shard": "unavailable",
        "diagnostic_arm": "production",
        "invocation_id": identity["invocation_id"],
    }
    available_durations = {
        stage: {
            "status": "available",
            "value": 0.000001,
            "unit": "seconds",
            "duration_ns": 1000,
        }
        for stage in (
            "data_read",
            "encode_tta",
            "detector_extraction",
            "pair_score",
            "threshold",
            "graph_build",
            "ilp",
            "geff",
        )
    }
    coordinate_event = {
        **envelope,
        "record_type": "coordinate_artifact",
        "dataset_path": str((data_root / "movie-a").resolve()),
        **coordinate_record,
    }
    support_events = [
        {**envelope, "record_type": "invocation_start", "identity": identity},
        *(
            {
                **envelope,
                "record_type": "retention",
                "dataset_path": str((data_root / "movie-a").resolve()),
                **row,
            }
            for row in retention
        ),
        coordinate_event,
        {
            **envelope,
            "record_type": "dataset_summary",
            "dataset": "movie-a",
            "dataset_path": str((data_root / "movie-a").resolve()),
            "outcome": "returned",
            "durations": available_durations,
            "counts": {
                "pair_universe": {
                    "status": "available",
                    "value": 3,
                    "unit": "ordered_source_target_pairs",
                },
                "threshold_passing_edge_candidates": {
                    "status": "available",
                    "value": 1,
                    "unit": "edges",
                },
                "detected_nodes": {
                    "status": "available",
                    "value": 2,
                    "unit": "nodes",
                },
                "pre_ilp_edges": {
                    "status": "available",
                    "value": 1,
                    "unit": "edges",
                },
                "output_edges": {
                    "status": "available",
                    "value": 1,
                    "unit": "edges",
                },
            },
            "ilp": {"status": "returned"},
            "coordinate_artifact": {"status": "available", **coordinate_record},
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
            "cuda_peak_allocated": {
                "status": "available",
                "value": 1024,
                "unit": "bytes",
            },
            "cuda_peak_reserved": {
                "status": "available",
                "value": 2048,
                "unit": "bytes",
            },
            "host_ru_maxrss": (
                {"status": "available", "value": 3, "unit": "bytes"}
                if malformed_process_resource
                else {
                    "status": "available",
                    "value": 3,
                    "unit": "kibibytes",
                    "normalized_bytes": 3072,
                }
            ),
        },
    ]
    (process_dir / "events.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in support_events),
        encoding="utf-8",
    )

    if include_nonproduction_invocation:
        diagnostic_dir = support_root / "invocation-diagnostic"
        diagnostic_dir.mkdir()
        diagnostic_identity = {
            **identity,
            "invocation_id": "diagnostic-invocation",
            "data_root": str((output / "validation").resolve()),
        }
        diagnostic_envelope = {
            **envelope,
            "process_id": "diagnostic-process",
            "pid": 7999,
            "invocation_id": "diagnostic-invocation",
        }
        diagnostic_events = [
            {
                **diagnostic_envelope,
                "record_type": "invocation_start",
                "identity": diagnostic_identity,
            },
            {
                **diagnostic_envelope,
                "record_type": "invocation_finish",
                "identity": diagnostic_identity,
                "outcome": "returned",
            },
            {
                **diagnostic_envelope,
                "invocation_id": None,
                "record_type": "process_summary",
                "invocation_ids": ["diagnostic-invocation"],
                "cuda_peak_allocated": {"status": "unavailable", "reason": "fixture"},
                "cuda_peak_reserved": {"status": "unavailable", "reason": "fixture"},
                "host_ru_maxrss": {"status": "unavailable", "reason": "fixture"},
            },
        ]
        (diagnostic_dir / "events.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostic_events),
            encoding="utf-8",
        )

    source_hash = "a" * 64
    source_chain = {
        "pre_public_support_sha256": "b" * 64,
        "public_patched_support_sha256": "c" * 64,
        "telemetry_output_support_sha256": source_hash,
        "runtime_module_sha256": "d" * 64,
    }
    launch_events = [
        {"event": "SUPPORT_PATCHED", "source_chain": source_chain},
        {
            "event": "SHARD_LAUNCHED",
            "pid": 7001,
            "argv": [
                sys.executable,
                "scripts/predict_unet_transformer.py",
                "--data-dir",
                str(data_root),
                "--output-dir",
                str(output_dir),
                "--method",
                "public-reference",
                "--split",
                "0",
            ],
            "source_sha256": source_hash,
        },
        {
            "event": "LAUNCH_OBSERVATION_FINISHED",
            "launch_count": 1,
            "expected_launch_count": 1,
            "status": "PASS",
        },
    ]
    (support_root / "launch-events.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in launch_events),
        encoding="utf-8",
    )

    stats = {
        "dataset": "movie-a",
        "raw_nodes": raw_nodes,
        "raw_edges": 1,
        "nodes": final_nodes,
        "edges": 1,
        **{name: 0 for name in _COUNTER_SEMANTICS},
    }
    write_csv(output / "run_stats.csv", list(stats), [stats])

    submission_rows: list[dict[str, object]] = []
    for index in range(submission_nodes):
        submission_rows.append(
            {"id": index, "dataset": "movie-a", "row_type": "node"}
        )
    submission_rows.append({"id": 100, "dataset": "movie-a", "row_type": "edge"})
    write_csv(output / "submission.csv", ["id", "dataset", "row_type"], submission_rows)


def config(output: Path, public_sources: tuple[str, ...] = ("public-a\n", "public-b\n")):
    return NotebookTelemetryConfig(
        output_dir=str(output.resolve()),
        expected_public_cell_sha256=tuple(sha_text(source) for source in public_sources),
        production_data_root=str((output / "test").resolve()),
        production_test_names=("movie-a",),
        sample_interval_seconds=0.01,
        nvidia_smi_timeout_seconds=0.1,
        max_samples=100,
    )


def test_source_factory_preserves_absolute_kaggle_posix_paths_on_windows() -> None:
    cfg = NotebookTelemetryConfig(
        output_dir="/kaggle/working",
        expected_public_cell_sha256=(sha_text("public\n"),),
        production_data_root="/kaggle/input/biohub-cell-tracking-test-data/test",
        production_test_names=("movie-a",),
    ).validated()

    assert cfg.output_dir == "/kaggle/working"
    assert cfg.production_data_root == (
        "/kaggle/input/biohub-cell-tracking-test-data/test"
    )
    sources = build_notebook_telemetry_sources(cfg)
    assert '"output_dir":"/kaggle/working"' in sources.prepended_source
    assert (
        '"production_data_root":"/kaggle/input/biohub-cell-tracking-test-data/test"'
        in sources.prepended_source
    )


class FakeEvents:
    def __init__(self) -> None:
        self.callbacks: dict[str, list[object]] = {"pre_run_cell": [], "post_run_cell": []}

    def register(self, event: str, callback: object) -> None:
        self.callbacks[event].append(callback)

    def unregister(self, event: str, callback: object) -> None:
        self.callbacks[event].remove(callback)

    def emit(self, event: str, value: object) -> None:
        for callback in list(self.callbacks[event]):
            callback(value)  # type: ignore[operator]


class FakeIPython:
    def __init__(self) -> None:
        self.events = FakeEvents()


@dataclass
class CellInfo:
    raw_cell: str


@dataclass
class CellResult:
    info: CellInfo | None = None
    error_before_exec: BaseException | None = None
    error_in_exec: BaseException | None = None


def measured_host() -> dict[str, object]:
    return {
        "status": "MEASURED",
        "backend": "test",
        "root_pid": 1,
        "process_count": 2,
        "rss_bytes": 4096,
    }


def measured_gpu() -> dict[str, object]:
    return {
        "status": "MEASURED",
        "backend": "test",
        "devices": [
            {
                "index": 0,
                "uuid": "GPU-test",
                "used_bytes": 1024,
                "total_bytes": 8192,
            }
        ],
    }


def test_embedded_callbacks_sampler_and_finalizer_execute(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    make_evidence(output)
    public_sources = ("public-a\n", "public-b\n")
    instrumentation_source = "existing release validation\n"
    cfg = NotebookTelemetryConfig(
        **{
            **config(output, public_sources).__dict__,
            "allowed_auxiliary_cell_sha256": (sha_text(instrumentation_source),),
        }
    )
    sources = build_notebook_telemetry_sources(cfg)
    assert sources.runtime_source_sha256 == sha_text(
        Path(sys.modules[NotebookTelemetryConfig.__module__].__file__).read_text(encoding="utf-8")
    )

    shell = FakeIPython()
    namespace = {"get_ipython": lambda: shell}
    exec(  # noqa: S102 - executing generated notebook source is the behavior under test
        compile(sources.prepended_source, "prepended-cell", "exec"), namespace
    )
    tracker = namespace["_E0_NOTEBOOK_TELEMETRY"]
    tracker.host_probe = measured_host
    tracker.gpu_probe = measured_gpu

    # The bootstrap registers during its own execution and may receive a post
    # callback without a matching pre callback.
    shell.events.emit("post_run_cell", CellResult())
    for source in (instrumentation_source, *public_sources):
        shell.events.emit("pre_run_cell", CellInfo(source))
        shell.events.emit("post_run_cell", CellResult(info=CellInfo(source)))
    shell.events.emit("pre_run_cell", CellInfo(sources.final_source))
    exec(  # noqa: S102 - executing generated notebook source is the behavior under test
        compile(sources.final_source, "final-cell", "exec"), namespace
    )

    result = namespace["_E0_NOTEBOOK_TELEMETRY_RESULT"]
    assert result["status"] == "PASS"
    assert result["provider_elapsed_seconds"] is None
    telemetry = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    cell_records = telemetry["runtime_scopes"]["cells"]["records"]
    assert [row["role"] for row in cell_records] == [
        "instrumentation",
        "public",
        "public",
    ]
    assert [row["sha256"] for row in cell_records[1:]] == list(
        cfg.expected_public_cell_sha256
    )
    assert telemetry["runtime_scopes"]["provider"]["status"] == "UNAVAILABLE"
    assert telemetry["resources"]["cleanup_complete"] is True
    assert shell.events.callbacks == {"pre_run_cell": [], "post_run_cell": []}
    manifest = json.loads((output / cfg.run_manifest_filename).read_text(encoding="utf-8"))
    assert manifest["elapsed_seconds"] == 123.0
    assert manifest["full_runtime_seconds"] == 123.0
    assert manifest["e0_telemetry"]["provider_elapsed_seconds"] is None


def test_unexpected_or_duplicate_public_cell_fails_immediately(tmp_path: Path) -> None:
    cfg = config(tmp_path / "evidence")
    shell = FakeIPython()
    tracker = NotebookTelemetryTracker(cfg, measured_host, measured_gpu)
    tracker.start(shell)
    with pytest.raises(E0TelemetryError, match="unexpected notebook cell"):
        tracker.pre_run_cell(CellInfo("changed source\n"))

    tracker.pre_run_cell(CellInfo("public-a\n"))
    tracker.post_run_cell(CellResult(info=CellInfo("public-a\n")))
    with pytest.raises(E0TelemetryError, match="unexpected notebook cell"):
        tracker.pre_run_cell(CellInfo("public-a\n"))
    assert tracker._sampler is not None
    assert tracker._sampler.stop()["cleanup_complete"] is True


def test_event_manager_swallowing_callback_error_still_poisons_final_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "swallowed-callback"
    make_evidence(output)

    class CatchingEvents(FakeEvents):
        def emit(self, event: str, value: object) -> list[BaseException]:
            errors: list[BaseException] = []
            for callback in list(self.callbacks[event]):
                try:
                    callback(value)  # type: ignore[operator]
                except BaseException as exc:  # noqa: BLE001 - emulate IPython EventManager
                    errors.append(exc)
            return errors

    shell = FakeIPython()
    shell.events = CatchingEvents()
    tracker = NotebookTelemetryTracker(config(output), measured_host, measured_gpu)
    tracker.start(shell)
    for source in ("public-a\n", "public-b\n"):
        shell.events.emit("pre_run_cell", CellInfo(source))
        shell.events.emit("post_run_cell", CellResult(info=CellInfo(source)))
        if source == "public-a\n":
            errors = shell.events.emit("pre_run_cell", CellInfo("unexpected extra cell\n"))
            assert len(errors) == 1
            # EventManager-style handling logs the callback error and still runs the cell.
            shell.events.emit(
                "post_run_cell", CellResult(info=CellInfo("unexpected extra cell\n"))
            )
    with pytest.raises(E0TelemetryError, match="callback evidence has 1 violation"):
        tracker.finalize()
    assert tracker._sampler is not None
    assert tracker._sampler.stop()["cleanup_complete"] is True
    assert shell.events.callbacks == {"pre_run_cell": [], "post_run_cell": []}


def test_post_callback_hash_mismatch_poison_is_persisted(tmp_path: Path) -> None:
    output = tmp_path / "post-mismatch"
    make_evidence(output)
    shell = FakeIPython()
    tracker = NotebookTelemetryTracker(config(output), measured_host, measured_gpu)
    tracker.start(shell)
    tracker.pre_run_cell(CellInfo("public-a\n"))
    with pytest.raises(E0TelemetryError, match="callback evidence"):
        tracker.post_run_cell(CellResult(info=CellInfo("mutated\n")))
        tracker.finalize()
    # post_run_cell records rather than relying on callback exceptions for enforcement.
    assert tracker._violations[0]["kind"] == "POST_SOURCE_HASH_MISMATCH"
    assert tracker._sampler is not None
    assert tracker._sampler.stop()["cleanup_complete"] is True


def test_harvest_failure_guarantees_sampler_and_callback_cleanup(tmp_path: Path) -> None:
    output = tmp_path / "cleanup-on-failure"
    make_evidence(output, missing_retention_frame=True)
    shell = FakeIPython()
    original_dict_writer = csv.DictWriter
    tracker = NotebookTelemetryTracker(config(output), measured_host, measured_gpu)
    tracker.start(shell)
    for source in ("public-a\n", "public-b\n"):
        tracker.pre_run_cell(CellInfo(source))
        tracker.post_run_cell(CellResult(info=CellInfo(source)))
    with pytest.raises(E0TelemetryError, match="retention frame coverage mismatch"):
        tracker.finalize()
    assert tracker._sampler is not None
    summary = tracker._sampler.stop()
    assert summary["cleanup_complete"] is True
    assert shell.events.callbacks == {"pre_run_cell": [], "post_run_cell": []}
    assert csv.DictWriter is original_dict_writer


def test_sampler_is_bounded_and_cleans_up(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    sampler = ResourceSampler(
        path=samples,
        interval_seconds=0.005,
        nvidia_smi_timeout_seconds=0.1,
        max_samples=2,
        host_probe=measured_host,
        gpu_probe=measured_gpu,
    )
    sampler.start()
    time.sleep(0.04)
    summary = sampler.stop()
    assert summary["status"] == "ERROR"
    assert summary["coverage_status"] == "PARTIAL_CAP_EXHAUSTED"
    assert summary["cleanup_complete"] is True
    assert summary["sample_count"] == 2
    assert summary["dropped_samples"] > 0
    assert summary["host_peak"] == {"status": "LOWER_BOUND", "rss_bytes": 4096}
    assert summary["gpu_peaks"]["status"] == "LOWER_BOUND"
    assert summary["gpu_peaks"]["used_bytes_by_uuid"] == {"GPU-test": 1024}
    assert len(samples.read_text(encoding="utf-8").splitlines()) == 2


def test_sampler_cap_does_not_include_an_unpersisted_peak(tmp_path: Path) -> None:
    host_values = iter((10, 999))
    gpu_values = iter((20, 888))
    sampler = ResourceSampler(
        path=tmp_path / "bounded.jsonl",
        interval_seconds=10.0,
        nvidia_smi_timeout_seconds=0.1,
        max_samples=1,
        host_probe=lambda: {"status": "MEASURED", "rss_bytes": next(host_values)},
        gpu_probe=lambda: {
            "status": "MEASURED",
            "devices": [
                {
                    "uuid": "GPU-test",
                    "used_bytes": next(gpu_values),
                    "total_bytes": 1000,
                }
            ],
        },
    )

    sampler._sample_once()
    sampler._sample_once()
    summary = sampler.stop()

    assert summary["status"] == "ERROR"
    assert summary["sample_count"] == 1
    assert summary["dropped_samples"] == 1
    assert summary["host_peak"] == {"status": "LOWER_BOUND", "rss_bytes": 10}
    assert summary["gpu_peaks"]["used_bytes_by_uuid"] == {"GPU-test": 20}


def test_finalizer_persists_error_manifest_when_sampler_cap_is_exhausted(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cap-finalizer"
    make_evidence(output)
    capped = NotebookTelemetryConfig(
        **{
            **config(output).__dict__,
            "sample_interval_seconds": 0.005,
            "max_samples": 1,
        }
    )
    shell = FakeIPython()
    tracker = NotebookTelemetryTracker(capped, measured_host, measured_gpu)
    tracker.start(shell)
    time.sleep(0.03)
    for source in ("public-a\n", "public-b\n"):
        tracker.pre_run_cell(CellInfo(source))
        tracker.post_run_cell(CellResult(info=CellInfo(source)))

    with pytest.raises(E0TelemetryError, match="resource telemetry failed"):
        tracker.finalize()

    telemetry = json.loads((output / capped.telemetry_filename).read_text(encoding="utf-8"))
    manifest = json.loads((output / capped.run_manifest_filename).read_text(encoding="utf-8"))
    assert telemetry["status"] == "ERROR"
    assert telemetry["resources"]["coverage_status"] == "PARTIAL_CAP_EXHAUSTED"
    assert manifest["e0_telemetry"]["status"] == "ERROR"
    assert shell.events.callbacks == {"pre_run_cell": [], "post_run_cell": []}


def test_unavailable_resource_backends_are_explicit_without_false_zero(tmp_path: Path) -> None:
    unavailable_host = lambda: {
        "status": "UNAVAILABLE",
        "backend": "test",
        "reason": "no procfs",
    }
    unavailable_gpu = lambda: {
        "status": "UNAVAILABLE",
        "backend": "test",
        "reason": "no nvidia-smi",
    }
    sampler = ResourceSampler(
        path=tmp_path / "unavailable.jsonl",
        interval_seconds=0.005,
        nvidia_smi_timeout_seconds=0.1,
        max_samples=10,
        host_probe=unavailable_host,
        gpu_probe=unavailable_gpu,
    )
    sampler.start()
    time.sleep(0.02)
    result = sampler.stop()
    assert result["status"] == "PASS"
    assert result["host_peak"] == {"status": "UNAVAILABLE", "reason": "no procfs"}
    assert result["gpu_peaks"] == {
        "status": "UNAVAILABLE",
        "reason": "no nvidia-smi",
    }


def test_harvest_requires_every_retention_frame(tmp_path: Path) -> None:
    output = tmp_path / "missing-frame"
    make_evidence(output, missing_retention_frame=True)
    with pytest.raises(E0TelemetryError, match="retention frame coverage mismatch"):
        harvest_e0_telemetry(config(output))


def test_harvest_rehashes_raw_coordinates_and_checks_bounds(tmp_path: Path) -> None:
    output = tmp_path / "bad-coordinate"
    make_evidence(output, corrupt_coordinate_hash=True)
    with pytest.raises(E0TelemetryError, match="coordinate artifact hash mismatch"):
        harvest_e0_telemetry(config(output))


def test_missing_raw_coordinates_are_unavailable_or_required(tmp_path: Path) -> None:
    output = tmp_path / "summary-only"
    make_evidence(output, include_coordinate_artifact=False)
    with pytest.raises(E0TelemetryError, match="raw coordinate artifact is required"):
        harvest_e0_telemetry(config(output))

    permissive = NotebookTelemetryConfig(
        **{**config(output).__dict__, "require_coordinate_artifacts": False}
    )
    result = harvest_e0_telemetry(permissive)
    artifact = result["coordinates"]["datasets"]["movie-a"]["artifact"]
    assert artifact["status"] == "UNAVAILABLE"
    assert "no bound raw coordinate artifact" in artifact["reason"]


def test_harvest_records_pre_to_post_ilp_delta_without_false_equality(tmp_path: Path) -> None:
    output = tmp_path / "ilp-delta"
    make_evidence(output, raw_nodes=1)
    result = harvest_e0_telemetry(config(output))
    reconciliation = result["reconciliations"]["movie-a"]
    assert reconciliation["detector_coordinate_rows"] == 2
    assert reconciliation["post_ilp_raw_nodes"] == 1
    assert reconciliation["pre_ilp_to_post_ilp_node_delta"] == -1


def test_harvest_ignores_unlaunched_validation_invocation(tmp_path: Path) -> None:
    output = tmp_path / "mixed-invocations"
    make_evidence(output, include_nonproduction_invocation=True)

    result = harvest_e0_telemetry(config(output))

    support = result["support"]
    assert support["ignored_nonproduction_invocation_count"] == 1
    assert [row["invocation_id"] for row in support["selected_invocations"]] == [
        "production-invocation"
    ]


def test_single_launch_accepts_reviewed_support_default_method(tmp_path: Path) -> None:
    output = tmp_path / "default-method"
    make_evidence(output)
    launch_path = output / "e0_support_telemetry" / "launch-events.jsonl"
    launches = [json.loads(line) for line in launch_path.read_text().splitlines()]
    argv = launches[1]["argv"]
    method_index = argv.index("--method")
    del argv[method_index : method_index + 2]
    launch_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in launches),
        encoding="utf-8",
    )

    result = harvest_e0_telemetry(config(output))

    assert result["support"]["selected_invocations"][0]["identity"]["method"] == (
        "public-reference"
    )


def test_harvest_rejects_swapped_launched_shard_cohort(tmp_path: Path) -> None:
    output = tmp_path / "swapped-shard"
    make_evidence(output)
    manifest_path = output / "public_reference_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["actual_input_shapes_tzyx"]["movie-b"] = [2, 3, 4, 5]
    write_json(manifest_path, manifest)

    support_root = output / "e0_support_telemetry"
    launch_path = support_root / "launch-events.jsonl"
    launches = [json.loads(line) for line in launch_path.read_text().splitlines()]
    first_launch = launches[1]
    first_launch["argv"].extend(["--slice", "0::2"])
    second_launch = {
        **first_launch,
        "pid": 7002,
        "argv": [
            value if value != "0::2" else "1::2" for value in first_launch["argv"]
        ],
    }
    launches.insert(2, second_launch)
    launches[-1]["launch_count"] = 2
    launches[-1]["expected_launch_count"] = 2
    launch_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in launches),
        encoding="utf-8",
    )
    event_path = support_root / "invocation-production" / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    for event in events:
        event["gpu_shard"] = "0/2"
        if isinstance(event.get("identity"), dict):
            event["identity"]["test_names"] = ["movie-b"]
    event_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
        encoding="utf-8",
    )
    swapped = NotebookTelemetryConfig(
        **{**config(output).__dict__, "production_test_names": ("movie-a", "movie-b")}
    )

    with pytest.raises(E0TelemetryError, match="wrong ordered shard cohort"):
        harvest_e0_telemetry(swapped)


def test_harvest_rejects_malformed_process_resource_schema(tmp_path: Path) -> None:
    output = tmp_path / "malformed-resource"
    make_evidence(output, malformed_process_resource=True)

    with pytest.raises(E0TelemetryError, match="host RSS normalized_bytes"):
        harvest_e0_telemetry(config(output))


def test_submission_count_mismatch_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "bad-submission"
    make_evidence(output, final_nodes=2, submission_nodes=1)
    with pytest.raises(E0TelemetryError, match="submission/run_stats count mismatch"):
        harvest_e0_telemetry(config(output))


def test_nvidia_smi_units_are_converted_from_mib_to_bytes(monkeypatch) -> None:
    def run(*args, **kwargs):
        return types.SimpleNamespace(
            returncode=0,
            stdout="0, GPU-a, 2, 8\n1, GPU-b, 3, 16\n",
        )

    monkeypatch.setattr("subprocess.run", run)
    result = _read_nvidia_smi_snapshot(0.5)
    assert result["status"] == "MEASURED"
    assert result["devices"] == [
        {
            "index": 0,
            "uuid": "GPU-a",
            "used_bytes": 2 * 1024 * 1024,
            "total_bytes": 8 * 1024 * 1024,
        },
        {
            "index": 1,
            "uuid": "GPU-b",
            "used_bytes": 3 * 1024 * 1024,
            "total_bytes": 16 * 1024 * 1024,
        },
    ]


def test_targeted_to_csv_wrapper_records_and_restores(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "serialization"
    output.mkdir()

    class FakeDataFrame:
        def to_csv(self, destination, **kwargs):
            Path(destination).write_text("data", encoding="utf-8")
            return kwargs.get("return_value")

    original = FakeDataFrame.to_csv
    monkeypatch.setitem(sys.modules, "pandas", types.SimpleNamespace(DataFrame=FakeDataFrame))
    tracker = NotebookTelemetryTracker(config(output), measured_host, measured_gpu)
    tracker._install_to_csv_wrapper_if_available()
    frame = FakeDataFrame()
    assert frame.to_csv(output / "other.csv", return_value="other") == "other"
    assert frame.to_csv(output / "submission.csv", return_value="target") == "target"
    assert len(tracker._serialization_records) == 1
    assert tracker._serialization_records[0]["status"] == "PASS"
    tracker._restore_to_csv_wrapper()
    assert FakeDataFrame.to_csv is original


def test_targeted_dict_writer_times_only_writes_and_preserves_csv_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "dict-writer"
    output.mkdir()
    target = output / "submission.csv"
    control = output / "control.csv"
    other = output / "other.csv"
    fieldnames = ["id", "dataset", "value"]
    rows = [
        {"id": 0, "dataset": "movie-a", "value": "a,b"},
        {"id": 1, "dataset": "movie-a", "value": 'quoted "value"'},
        {"id": 2, "dataset": "movie-b", "value": "last"},
        {"id": 3, "dataset": "movie-b", "value": "bulk"},
    ]
    original = csv.DictWriter
    with control.open("w", encoding="utf-8", newline="") as stream:
        writer = original(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(rows[0])
        writer.writerow(rows[1])
        writer.writerows(rows[2:])

    class Clock:
        value = 0.0

        @classmethod
        def monotonic(cls) -> float:
            return cls.value

    class ClockedStream:
        def __init__(self, stream, path: Path) -> None:
            self._stream = stream
            self.name = str(path)

        def write(self, value: str) -> int:
            Clock.value += 0.25
            return self._stream.write(value)

    monkeypatch.setattr(
        "biohub_ct.campaign.e0_notebook_telemetry.time.monotonic",
        Clock.monotonic,
    )
    tracker = NotebookTelemetryTracker(config(output), measured_host, measured_gpu)
    tracker._install_to_csv_wrapper_if_available()

    with other.open("w", encoding="utf-8", newline="") as raw:
        writer = csv.DictWriter(ClockedStream(raw, other), fieldnames=fieldnames)
        writer.writeheader()
    assert tracker._serialization_records == []

    with target.open("w", encoding="utf-8", newline="") as raw:
        # The stdlib API accepts ``f`` by keyword; the wrapper must retain that API.
        writer = csv.DictWriter(f=ClockedStream(raw, target), fieldnames=fieldnames)
        writer.writeheader()
        Clock.value += 100.0  # Representative graph postprocessing between writes.
        first_result = writer.writerow(rows[0])
        Clock.value += 200.0
        second_result = writer.writerow(rows[1])
        Clock.value += 300.0

        def lazy_rows():
            for row in rows[2:]:
                Clock.value += 10.0  # Production inside writerows is in scope.
                yield row

        bulk_result = writer.writerows(lazy_rows())

    assert first_result > 0
    assert second_result > 0
    assert bulk_result is None
    assert target.read_bytes() == control.read_bytes()
    assert len(tracker._serialization_records) == 1
    record = tracker._serialization_records[0]
    assert record["implementation"] == "csv.DictWriter"
    assert record["status"] == "PASS"
    # The 600 seconds between calls are excluded. The 20 seconds spent producing
    # lazy writerows rows are necessarily inside that one timed call.
    assert record["elapsed_seconds"] == pytest.approx(21.25)
    assert record["call_count"] == 4
    assert record["row_count"] == 5
    assert record["calls_by_method"] == {
        "writeheader": 1,
        "writerow": 2,
        "writerows": 1,
    }
    assert "excludes argument construction, graph computation" in str(record["scope"])
    assert "final stream close/fsync" in str(record["scope"])

    tracker._restore_to_csv_wrapper()
    assert csv.DictWriter is original


def test_targeted_dict_writer_preserves_exception_and_atexit_restores(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dict-writer-error"
    output.mkdir()
    target = output / "submission.csv"
    original = csv.DictWriter

    class FailingStream:
        name = str(target)

        @staticmethod
        def write(_value: str) -> int:
            raise LookupError("controlled write failure")

    tracker = NotebookTelemetryTracker(config(output), measured_host, measured_gpu)
    tracker._install_to_csv_wrapper_if_available()
    writer = csv.DictWriter(FailingStream(), fieldnames=["id"])
    with pytest.raises(LookupError, match="controlled write failure"):
        writer.writeheader()

    assert tracker._serialization_records[0]["status"] == "ERROR"
    assert tracker._serialization_records[0]["error_type"] == "LookupError"
    assert tracker._serialization_records[0]["call_count"] == 1
    assert tracker._serialization_records[0]["row_count"] == 1
    tracker._atexit_cleanup()
    assert csv.DictWriter is original
