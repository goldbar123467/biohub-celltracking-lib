#!/usr/bin/env python3
"""Validate downloaded E0 R4 telemetry without modifying its evidence tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.campaign.e0_notebook_telemetry import (
    NotebookTelemetryConfig,
    harvest_e0_telemetry,
)
from biohub_ct.campaign.kaggle_rehearsal import preflight_package
from biohub_ct.campaign.rehearsal_packages import (
    DEFAULT_PACKAGE_DIRS,
    E0_R4_PACKAGE_IDENTITY,
)


class TelemetryVerificationError(RuntimeError):
    """Downloaded telemetry is incomplete, inconsistent, or not safely readable."""


_STAGES = (
    "data_read",
    "encode_tta",
    "detector_extraction",
    "pair_score",
    "threshold",
    "graph_build",
    "ilp",
    "geff",
)
_SHA256 = "sha256"
_RECORDED_WORKING_ROOT = PurePosixPath("/kaggle/working")
_TOP_LEVEL_FILES = (
    "public_reference_run_manifest.json",
    "submission.csv",
    "run_stats.csv",
    "e0_notebook_telemetry.json",
    "e0_resource_samples.jsonl",
    "e0_cell_events.jsonl",
)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, _SHA256).hexdigest()


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise TelemetryVerificationError(f"{label} must be an existing regular directory")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise TelemetryVerificationError(f"cannot resolve {label}") from exc


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TelemetryVerificationError(f"{label} must be an existing regular file")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise TelemetryVerificationError(f"cannot resolve {label}") from exc


def _inside_file(root: Path, relative: PurePosixPath, label: str) -> Path:
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise TelemetryVerificationError(f"{label} is not a confined relative path")
    candidate = root.joinpath(*relative.parts)
    resolved = _regular_file(candidate, label)
    if not resolved.is_relative_to(root):
        raise TelemetryVerificationError(f"{label} escaped the downloaded output directory")
    return resolved


def _strict_json_bytes(raw: bytes, label: str) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TelemetryVerificationError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise TelemetryVerificationError(f"nonfinite JSON value in {label}: {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TelemetryVerificationError(f"cannot parse strict JSON from {label}") from exc


def _strict_json_object(path: Path, label: str) -> dict[str, object]:
    value = _strict_json_bytes(_regular_file(path, label).read_bytes(), label)
    if not isinstance(value, dict):
        raise TelemetryVerificationError(f"{label} must contain a JSON object")
    return value


def _strict_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    raw = _regular_file(path, label).read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise TelemetryVerificationError(f"{label} must be nonempty newline-terminated JSONL")
    records: list[dict[str, object]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        value = _strict_json_bytes(line, f"{label} line {index}")
        if not isinstance(value, dict):
            raise TelemetryVerificationError(f"{label} line {index} must be a JSON object")
        records.append(value)
    return records


def _file_evidence(path: Path, root: Path) -> dict[str, object]:
    resolved = _regular_file(path, str(path))
    if not resolved.is_relative_to(root):
        raise TelemetryVerificationError("evidence file escaped the downloaded output directory")
    relative = resolved.relative_to(root).as_posix()
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _relevant_evidence_files(output: Path) -> list[dict[str, object]]:
    paths = [output / name for name in _TOP_LEVEL_FILES]
    violations = output / "e0_cell_violations.jsonl"
    if violations.exists() or violations.is_symlink():
        paths.append(violations)
    support = _regular_directory(output / "e0_support_telemetry", "support telemetry")
    paths.extend(sorted(path for path in support.rglob("*") if path.is_file() or path.is_symlink()))
    files = sorted(
        (_file_evidence(path, output) for path in paths), key=lambda row: str(row["path"])
    )
    if len({row["path"] for row in files}) != len(files):
        raise TelemetryVerificationError("duplicate relevant telemetry file identity")
    return files


def _canonical_posix_absolute(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise TelemetryVerificationError(f"{label} must be a canonical absolute POSIX path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise TelemetryVerificationError(f"{label} must be a canonical absolute POSIX path")
    return path


def _argv_option(argv: object, name: str) -> str:
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise TelemetryVerificationError("support launch argv is malformed")
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise TelemetryVerificationError(f"support launch argv needs exactly one {name}")
    return argv[positions[0] + 1]


def _production_data_root(
    launch_records: Sequence[Mapping[str, object]], competition: str
) -> PurePosixPath:
    launched = [record for record in launch_records if record.get("event") == "SHARD_LAUNCHED"]
    if len(launched) not in (1, 2):
        raise TelemetryVerificationError(
            "support launch evidence needs one or two production shards"
        )
    roots = {
        _canonical_posix_absolute(_argv_option(record.get("argv"), "--data-dir"), "--data-dir")
        for record in launched
    }
    if len(roots) != 1:
        raise TelemetryVerificationError("production shard data roots disagree")
    root = next(iter(roots))
    accepted = {
        PurePosixPath(f"/kaggle/input/{competition}/test"),
        PurePosixPath(f"/kaggle/input/competitions/{competition}/test"),
    }
    if root not in accepted:
        raise TelemetryVerificationError("production launch used an unexpected Kaggle input root")
    return root


def _map_provider_paths(value: object, provider_root: Path) -> object:
    """Map POSIX evidence paths into a private shadow for the existing harvester."""

    if isinstance(value, dict):
        return {key: _map_provider_paths(child, provider_root) for key, child in value.items()}
    if isinstance(value, list):
        return [_map_provider_paths(child, provider_root) for child in value]
    if isinstance(value, str) and value.startswith("/kaggle/"):
        pure = _canonical_posix_absolute(value, "recorded Kaggle path")
        return str(provider_root.joinpath(*pure.parts[1:]).resolve(strict=False))
    return value


def _normalize_paths(value: object, shadow_output: Path, provider_root: Path) -> object:
    if isinstance(value, dict):
        return {
            key: _normalize_paths(child, shadow_output, provider_root)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_normalize_paths(child, shadow_output, provider_root) for child in value]
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    shadow = str(shadow_output).replace("\\", "/").rstrip("/")
    provider = str(provider_root).replace("\\", "/").rstrip("/")
    if normalized == shadow:
        return _RECORDED_WORKING_ROOT.as_posix()
    if normalized.startswith(shadow + "/"):
        return (_RECORDED_WORKING_ROOT / normalized[len(shadow) + 1 :]).as_posix()
    if normalized == provider:
        return "/"
    if normalized.startswith(provider + "/"):
        return "/" + normalized[len(provider) + 1 :]
    return value


def _copy_regular(source: Path, destination: Path, root: Path, label: str) -> None:
    resolved = _regular_file(source, label)
    if not resolved.is_relative_to(root):
        raise TelemetryVerificationError(f"{label} escaped the downloaded output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise TelemetryVerificationError(f"duplicate shadow destination for {label}")
    shutil.copyfile(resolved, destination)


def _write_shadow_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
            stream.write("\n")


def _build_harvest_shadow(
    output: Path,
    shadow: Path,
    launch_records: Sequence[Mapping[str, object]],
) -> tuple[Path, Path, list[dict[str, object]], dict[str, dict[str, object]]]:
    shadow_output = shadow / "output"
    provider_root = shadow / "provider"
    shadow_output.mkdir()
    provider_root.mkdir()
    for name in ("public_reference_run_manifest.json", "run_stats.csv", "submission.csv"):
        _copy_regular(output / name, shadow_output / name, output, name)

    original_files: dict[str, dict[str, object]] = {}
    launch_path = output / "e0_support_telemetry" / "launch-events.jsonl"
    original_files["launch"] = _file_evidence(launch_path, output)
    mapped_launch = _map_provider_paths(list(launch_records), provider_root)
    assert isinstance(mapped_launch, list)
    _write_shadow_jsonl(
        shadow_output / "e0_support_telemetry" / "launch-events.jsonl", mapped_launch
    )

    event_paths = sorted((output / "e0_support_telemetry").glob("invocation-*/events.jsonl"))
    if not event_paths:
        raise TelemetryVerificationError("support telemetry has no invocation event files")
    for event_path in event_paths:
        if event_path.parent.parent != output / "e0_support_telemetry":
            raise TelemetryVerificationError("unsafe support invocation event path")
        records = _strict_jsonl(event_path, str(event_path.relative_to(output)))
        process_ids = {record.get("process_id") for record in records}
        if len(process_ids) != 1 or not all(isinstance(item, str) and item for item in process_ids):
            raise TelemetryVerificationError("support event file has inconsistent process identity")
        process_id = str(next(iter(process_ids)))
        if process_id in original_files:
            raise TelemetryVerificationError("duplicate support process identity")
        original_files[process_id] = _file_evidence(event_path, output)
        relative = event_path.relative_to(output)
        mapped_records = _map_provider_paths(records, provider_root)
        assert isinstance(mapped_records, list)
        _write_shadow_jsonl(shadow_output / relative, mapped_records)
        for record in records:
            if record.get("record_type") != "coordinate_artifact":
                continue
            artifact = record.get("artifact")
            if not isinstance(artifact, Mapping):
                continue
            raw_path = artifact.get("path")
            if not isinstance(raw_path, str):
                raise TelemetryVerificationError("coordinate artifact path is missing")
            relative_artifact = PurePosixPath(raw_path)
            source = _inside_file(output, relative_artifact, "coordinate artifact")
            destination = shadow_output.joinpath(*relative_artifact.parts)
            if not destination.exists():
                _copy_regular(source, destination, output, "coordinate artifact")
    return shadow_output, provider_root, list(mapped_launch), original_files


def _restore_original_event_evidence(
    harvested: dict[str, object], original_files: Mapping[str, Mapping[str, object]]
) -> None:
    support = harvested.get("support")
    if not isinstance(support, dict):
        raise TelemetryVerificationError("offline harvester returned no support object")
    launch = support.get("launch_evidence")
    if not isinstance(launch, dict):
        raise TelemetryVerificationError("offline harvester returned no launch evidence")
    original_launch = original_files["launch"]
    launch["bytes"] = original_launch["bytes"]
    launch["sha256"] = original_launch["sha256"]
    event_files = support.get("event_files")
    if not isinstance(event_files, list):
        raise TelemetryVerificationError("offline harvester returned no event-file evidence")
    for event in event_files:
        if not isinstance(event, dict) or not isinstance(event.get("process_id"), str):
            raise TelemetryVerificationError("offline harvester returned malformed event evidence")
        original = original_files.get(str(event["process_id"]))
        if original is None:
            raise TelemetryVerificationError("offline harvester returned an unknown process")
        event["bytes"] = original["bytes"]
        event["sha256"] = original["sha256"]


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryVerificationError(f"{label} must be finite numeric evidence")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        raise TelemetryVerificationError(f"{label} must be finite numeric evidence")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TelemetryVerificationError(f"{label} must be a positive integer")
    return value


def _utc_datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise TelemetryVerificationError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TelemetryVerificationError(f"{label} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise TelemetryVerificationError(f"{label} must be an ISO-8601 UTC timestamp")
    return parsed


def _verify_cells(
    telemetry: Mapping[str, object], output: Path, package_manifest: Mapping[str, object]
) -> dict[str, object]:
    runtime = telemetry.get("runtime_scopes")
    cells_scope = runtime.get("cells") if isinstance(runtime, Mapping) else None
    if not isinstance(cells_scope, Mapping) or cells_scope.get("status") != "MEASURED":
        raise TelemetryVerificationError("notebook cell timing scope is unavailable")
    records = cells_scope.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise TelemetryVerificationError("notebook cell records are malformed")
    if cells_scope.get("violations") != []:
        raise TelemetryVerificationError("notebook telemetry contains callback violations")
    event_records = _strict_jsonl(output / "e0_cell_events.jsonl", "e0_cell_events.jsonl")
    if event_records != records:
        raise TelemetryVerificationError("cell-event JSONL differs from final telemetry")

    source = package_manifest.get("source_notebook")
    instrumentation = package_manifest.get("instrumentation")
    generated = (
        instrumentation.get("generated_cell_sha256")
        if isinstance(instrumentation, Mapping)
        else None
    )
    public = source.get("cell_source_sha256") if isinstance(source, Mapping) else None
    if not isinstance(public, list) or len(public) != 12 or not isinstance(generated, Mapping):
        raise TelemetryVerificationError(
            "reviewed R4 package lacks the exact cell identity contract"
        )
    expected = [
        ("instrumentation", generated.get("interceptor_install"), None),
        ("instrumentation", generated.get("input_integrity"), None),
        *(("public", digest, index) for index, digest in enumerate(public)),
        ("instrumentation", generated.get("base_validation"), None),
        ("instrumentation", generated.get("interceptor_close"), None),
    ]
    if len(records) != len(expected):
        raise TelemetryVerificationError("notebook cell coverage is incomplete")
    for index, (record, expected_record) in enumerate(zip(records, expected, strict=True)):
        role, digest, public_index = expected_record
        if (
            record.get("role") != role
            or record.get("sha256") != digest
            or record.get("post_sha256") != digest
            or record.get("post_hash_matches_pre") is not True
            or record.get("status") != "PASS"
            or record.get("error_type") is not None
        ):
            raise TelemetryVerificationError(
                f"cell record {index} violates exact R4 execution identity"
            )
        if public_index is None:
            if "public_index" in record:
                raise TelemetryVerificationError("instrumentation cell claimed a public index")
        elif record.get("public_index") != public_index:
            raise TelemetryVerificationError("public cell index coverage is incomplete")
        _number(record.get("elapsed_seconds"), f"cell {index} elapsed_seconds")
    violations = output / "e0_cell_violations.jsonl"
    if violations.exists() or violations.is_symlink():
        raise TelemetryVerificationError("a PASS run must not contain a cell-violation log")
    return {"status": "PASS", "record_count": len(records), "public_cell_count": 12}


def _verify_callback(
    output: Path,
    artifact_lock: Mapping[str, object],
    *,
    observed_launch_count: int,
) -> dict[str, object]:
    callback = _strict_json_object(
        output / "e0_support_telemetry" / "launcher-callback-status.json",
        "launcher callback status",
    )
    instrumentation = artifact_lock.get("instrumentation")
    config = instrumentation.get("config") if isinstance(instrumentation, Mapping) else None
    expected_hash = (
        config.get("callback_after_public_cell_sha256") if isinstance(config, Mapping) else None
    )
    expected_keys = {
        "schema_version",
        "status",
        "callback_after_public_cell_sha256",
        "launch_count",
        "expected_launch_count",
        "cleanup_status",
        "cleanup_errors",
    }
    launch_count = callback.get("launch_count")
    if (
        set(callback) != expected_keys
        or callback.get("schema_version") != 1
        or callback.get("status") != "PASS"
        or callback.get("callback_after_public_cell_sha256") != expected_hash
        or isinstance(launch_count, bool)
        or launch_count not in (1, 2)
        or launch_count != observed_launch_count
        or callback.get("expected_launch_count") != launch_count
        or callback.get("cleanup_status") != "PASS"
        or callback.get("cleanup_errors") != []
    ):
        raise TelemetryVerificationError("launcher callback or cleanup evidence did not pass")
    return dict(callback)


def _verify_samples(telemetry: Mapping[str, object], output: Path) -> dict[str, object]:
    resources = telemetry.get("resources")
    if not isinstance(resources, Mapping):
        raise TelemetryVerificationError("whole-wrapper resource summary is missing")
    samples_path = output / "e0_resource_samples.jsonl"
    samples = _strict_jsonl(samples_path, "e0_resource_samples.jsonl")
    sample_hash = _sha256_file(samples_path)
    if (
        resources.get("scope") != "whole_notebook_wrapper"
        or resources.get("status") != "PASS"
        or resources.get("coverage_status") != "COMPLETE"
        or resources.get("cleanup_complete") is not True
        or resources.get("sample_count") != len(samples)
        or resources.get("dropped_samples") != 0
        or resources.get("write_errors") != 0
        or resources.get("sample_interval_seconds") != 5.0
        or resources.get("samples_sha256") != sample_hash
    ):
        raise TelemetryVerificationError(
            "whole-wrapper resource sampling is partial or inconsistent"
        )
    recorded_samples_path = _canonical_posix_absolute(resources.get("samples_path"), "samples_path")
    if recorded_samples_path != _RECORDED_WORKING_ROOT / "e0_resource_samples.jsonl":
        raise TelemetryVerificationError(
            "resource sample path is outside the recorded working root"
        )

    host_values: list[int] = []
    gpu_values: dict[str, int] = {}
    gpu_totals: dict[str, int] = {}
    monotonic_values: list[float] = []
    host_measured_count = 0
    host_unavailable_count = 0
    gpu_measured_count = 0
    gpu_unavailable_count = 0
    wrapper_started = _utc_datetime(telemetry.get("started_at_utc"), "telemetry started_at_utc")
    wrapper_finalized = _utc_datetime(
        telemetry.get("finalized_at_utc"), "telemetry finalized_at_utc"
    )
    if wrapper_finalized < wrapper_started:
        raise TelemetryVerificationError("telemetry wrapper timestamps are reversed")
    for index, sample in enumerate(samples):
        if sample.get("schema_version") != 1:
            raise TelemetryVerificationError("resource sample schema version is invalid")
        captured = _utc_datetime(sample.get("captured_at_utc"), f"resource sample {index}")
        if not wrapper_started <= captured <= wrapper_finalized:
            raise TelemetryVerificationError(
                f"resource sample {index} falls outside the notebook wrapper"
            )
        monotonic_values.append(_number(sample.get("monotonic_seconds"), "sample monotonic time"))
        host = sample.get("host")
        if not isinstance(host, Mapping) or host.get("status") not in {
            "MEASURED",
            "UNAVAILABLE",
        }:
            raise TelemetryVerificationError(f"resource sample {index} has invalid host status")
        if host.get("status") == "MEASURED":
            host_measured_count += 1
            rss = host.get("rss_bytes")
            if isinstance(rss, bool) or not isinstance(rss, int) or rss < 0:
                raise TelemetryVerificationError(f"resource sample {index} has invalid host RSS")
            host_values.append(rss)
        else:
            host_unavailable_count += 1
            if not isinstance(host.get("reason"), str) or not host.get("reason"):
                raise TelemetryVerificationError(
                    f"resource sample {index} lacks a host-unavailable reason"
                )
        gpu = sample.get("gpus")
        if not isinstance(gpu, Mapping) or gpu.get("status") not in {
            "MEASURED",
            "UNAVAILABLE",
        }:
            raise TelemetryVerificationError(f"resource sample {index} has invalid GPU status")
        if gpu.get("status") == "MEASURED":
            gpu_measured_count += 1
            devices = gpu.get("devices")
            if not isinstance(devices, list):
                raise TelemetryVerificationError("measured GPU sample lacks devices")
            for device in devices:
                if not isinstance(device, Mapping):
                    raise TelemetryVerificationError("GPU sample device is malformed")
                uuid = device.get("uuid")
                used = device.get("used_bytes")
                total = device.get("total_bytes")
                if (
                    not isinstance(uuid, str)
                    or not uuid
                    or isinstance(used, bool)
                    or not isinstance(used, int)
                    or used < 0
                    or isinstance(total, bool)
                    or not isinstance(total, int)
                    or total <= 0
                    or used > total
                ):
                    raise TelemetryVerificationError("GPU sample device values are invalid")
                if uuid in gpu_totals and gpu_totals[uuid] != total:
                    raise TelemetryVerificationError("GPU total memory changed across samples")
                gpu_totals[uuid] = total
                gpu_values[uuid] = max(gpu_values.get(uuid, 0), used)
        else:
            gpu_unavailable_count += 1
            if not isinstance(gpu.get("reason"), str) or not gpu.get("reason"):
                raise TelemetryVerificationError(
                    f"resource sample {index} lacks a GPU-unavailable reason"
                )
    if any(right <= left for left, right in pairwise(monotonic_values)):
        raise TelemetryVerificationError("resource sample monotonic timestamps are not increasing")
    host_peak = resources.get("host_peak")
    gpu_peaks = resources.get("gpu_peaks")
    if (
        not host_values
        or not isinstance(host_peak, Mapping)
        or host_peak.get("status") != "MEASURED"
        or host_peak.get("rss_bytes") != max(host_values)
    ):
        raise TelemetryVerificationError("whole-wrapper host peak is unavailable or inconsistent")
    if (
        not gpu_values
        or not isinstance(gpu_peaks, Mapping)
        or gpu_peaks.get("status") != "MEASURED"
        or gpu_peaks.get("used_bytes_by_uuid") != dict(sorted(gpu_values.items()))
        or gpu_peaks.get("total_bytes_by_uuid") != dict(sorted(gpu_totals.items()))
    ):
        raise TelemetryVerificationError("whole-wrapper GPU peaks are unavailable or inconsistent")
    if resources.get("sampled_until_monotonic_seconds") != monotonic_values[-1]:
        raise TelemetryVerificationError("resource sample endpoint differs from the raw sample log")
    return {
        "status": "PASS",
        "sample_count": len(samples),
        "host_sensor_coverage": {
            "measured_sample_count": host_measured_count,
            "unavailable_sample_count": host_unavailable_count,
        },
        "gpu_sensor_coverage": {
            "measured_sample_count": gpu_measured_count,
            "unavailable_sample_count": gpu_unavailable_count,
        },
        "sample_probe_status": (
            "ALL_PROBES_MEASURED"
            if host_unavailable_count == 0 and gpu_unavailable_count == 0
            else "PARTIAL_SENSOR_AVAILABILITY"
        ),
        "samples_sha256": sample_hash,
        "host_peak": dict(host_peak),
        "gpu_peaks": dict(gpu_peaks),
        "measurement_limit": "sampled maxima at the frozen five-second interval",
    }


def _verify_runtime_scopes(telemetry: Mapping[str, object]) -> dict[str, object]:
    scopes = telemetry.get("runtime_scopes")
    if not isinstance(scopes, Mapping):
        raise TelemetryVerificationError("runtime scopes are missing")
    wrapper = scopes.get("wrapper")
    provider = scopes.get("provider")
    serialization = scopes.get("submission_csv_serialization")
    if not isinstance(wrapper, Mapping) or wrapper.get("status") != "MEASURED":
        raise TelemetryVerificationError("notebook-wrapper elapsed time is unavailable")
    wrapper_seconds = _number(
        wrapper.get("elapsed_seconds"), "wrapper elapsed_seconds", positive=True
    )
    if (
        not isinstance(provider, Mapping)
        or provider.get("status") != "UNAVAILABLE"
        or provider.get("elapsed_seconds") is not None
    ):
        raise TelemetryVerificationError("provider elapsed scope is overstated or malformed")
    if not isinstance(serialization, Mapping) or serialization.get("status") != "MEASURED":
        raise TelemetryVerificationError("submission serialization timing was not observed")
    serialization_records = serialization.get("records")
    if (
        not isinstance(serialization_records, list)
        or not serialization_records
        or any(
            not isinstance(row, Mapping) or row.get("status") != "PASS"
            for row in serialization_records
        )
    ):
        raise TelemetryVerificationError("submission serialization timing is incomplete")
    for index, record in enumerate(serialization_records):
        assert isinstance(record, Mapping)
        _number(
            record.get("elapsed_seconds"),
            f"submission serialization {index} elapsed_seconds",
        )
    return {
        "status": "PASS",
        "wrapper_elapsed_seconds": wrapper_seconds,
        "provider_elapsed_status": "UNAVAILABLE_INSIDE_NOTEBOOK",
        "provider_elapsed_seconds": None,
        "scope_limit": (
            "wrapper excludes kernel startup and final telemetry/manifest writes; "
            "provider-complete elapsed requires external evidence"
        ),
    }


def _verify_stage_and_process_coverage(harvested: Mapping[str, object]) -> dict[str, object]:
    support = harvested.get("support")
    if not isinstance(support, Mapping) or support.get("status") != "PASS":
        raise TelemetryVerificationError("support harvest did not pass")
    datasets = support.get("datasets")
    invocations = support.get("selected_invocations")
    if not isinstance(datasets, Mapping) or not datasets:
        raise TelemetryVerificationError("support harvest selected no production datasets")
    if not isinstance(invocations, list) or not invocations:
        raise TelemetryVerificationError("support harvest selected no production processes")
    phase_rows: dict[str, dict[str, float]] = {}
    unavailable: list[str] = []
    for dataset, raw in datasets.items():
        durations = raw.get("durations") if isinstance(raw, Mapping) else None
        if (
            not isinstance(dataset, str)
            or not isinstance(durations, Mapping)
            or set(durations) != set(_STAGES)
        ):
            raise TelemetryVerificationError("support duration phase coverage is malformed")
        normalized: dict[str, float] = {}
        for stage in _STAGES:
            value = durations[stage]
            if not isinstance(value, Mapping) or value.get("status") != "available":
                unavailable.append(f"{dataset}.{stage}")
                continue
            normalized[stage] = _number(value.get("value"), f"{dataset}.{stage}")
        phase_rows[dataset] = normalized
    process_rows: list[dict[str, object]] = []
    for invocation in invocations:
        resources = invocation.get("resources") if isinstance(invocation, Mapping) else None
        if not isinstance(resources, Mapping):
            raise TelemetryVerificationError("support process resource evidence is missing")
        for name in ("cuda_peak_allocated", "cuda_peak_reserved", "host_ru_maxrss"):
            value = resources.get(name)
            if not isinstance(value, Mapping) or value.get("status") != "available":
                unavailable.append(f"{invocation.get('invocation_id')}.{name}")
        process_rows.append(
            {
                "invocation_id": invocation.get("invocation_id"),
                "process_id": invocation.get("process_id"),
                "gpu_shard": invocation.get("gpu_shard"),
                "resources": dict(resources),
            }
        )
    if unavailable:
        raise TelemetryVerificationError(
            "telemetry has unavailable production phases or resource peaks: "
            + ", ".join(unavailable)
        )
    return {
        "status": "PASS",
        "stages": list(_STAGES),
        "dataset_durations_seconds": phase_rows,
        "production_processes": process_rows,
        "scope_limit": "stage durations exclude model loading, gaps between stages, and notebook overhead",
    }


def _verify_manifest_binding(
    telemetry: Mapping[str, object], telemetry_path: Path, run_manifest: Mapping[str, object]
) -> None:
    binding = run_manifest.get("e0_telemetry")
    if not isinstance(binding, Mapping) or set(binding) != {
        "schema_version",
        "status",
        "path",
        "sha256",
        "wrapper_elapsed_seconds",
        "provider_elapsed_seconds",
        "provider_elapsed_status",
    }:
        raise TelemetryVerificationError("run manifest has no exact e0_telemetry binding")
    recorded = _canonical_posix_absolute(binding.get("path"), "manifest telemetry path")
    runtime = telemetry.get("runtime_scopes")
    wrapper = runtime.get("wrapper") if isinstance(runtime, Mapping) else None
    if (
        binding.get("schema_version") != 1
        or binding.get("status") != "PASS"
        or recorded != _RECORDED_WORKING_ROOT / telemetry_path.name
        or binding.get("sha256") != _sha256_file(telemetry_path)
        or not isinstance(wrapper, Mapping)
        or binding.get("wrapper_elapsed_seconds") != wrapper.get("elapsed_seconds")
        or binding.get("provider_elapsed_seconds") is not None
        or binding.get("provider_elapsed_status") != "UNAVAILABLE_INSIDE_NOTEBOOK"
    ):
        raise TelemetryVerificationError("run manifest telemetry binding is inconsistent")


def _report_target(path: Path, output: Path, package: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise TelemetryVerificationError("--report-json must be a new path")
    parent = _regular_directory(path.parent, "--report-json parent")
    target = (parent / path.name).resolve(strict=False)
    if target.is_relative_to(output) or target.is_relative_to(package):
        raise TelemetryVerificationError(
            "--report-json cannot modify evidence or reviewed package bytes"
        )
    return target


def _write_report(path: Path, value: Mapping[str, object]) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise TelemetryVerificationError("cannot exclusively persist --report-json") from exc


def verify_downloaded_e0_telemetry(
    *,
    output_dir: Path,
    package_dir: Path,
    report_json: Path,
    harvester: Callable[..., dict[str, object]] = harvest_e0_telemetry,
    now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Validate one downloaded R4 tree and exclusively persist a read-only report."""

    output = _regular_directory(output_dir, "--output-dir")
    package = _regular_directory(package_dir, "--package-dir")
    target = _report_target(report_json, output, package)
    try:
        preflight_package(package, E0_R4_PACKAGE_IDENTITY)
    except Exception as exc:
        raise TelemetryVerificationError(
            "reviewed package does not match compiled E0 R4 identity"
        ) from exc
    package_manifest = _strict_json_object(package / "package-manifest.json", "R4 package manifest")
    artifact_lock = _strict_json_object(package / "artifact-lock.json", "R4 artifact lock")
    for name in _TOP_LEVEL_FILES:
        _regular_file(output / name, name)
    callback_path = output / "e0_support_telemetry" / "launcher-callback-status.json"
    launch_path = output / "e0_support_telemetry" / "launch-events.jsonl"
    _regular_file(callback_path, "launcher callback status")
    initial_files = _relevant_evidence_files(output)
    launch_records = _strict_jsonl(launch_path, "support launch events")
    production_root = _production_data_root(launch_records, E0_R4_PACKAGE_IDENTITY.competition)

    run_manifest = _strict_json_object(
        output / "public_reference_run_manifest.json", "public reference run manifest"
    )
    raw_shapes = run_manifest.get("actual_input_shapes_tzyx")
    if not isinstance(raw_shapes, Mapping) or not raw_shapes:
        raise TelemetryVerificationError("run manifest lacks actual input shapes")
    production_names = tuple(sorted(raw_shapes))

    with tempfile.TemporaryDirectory(prefix="e0-telemetry-verify-") as temporary:
        shadow_root = Path(temporary)
        shadow_output, provider_root, _mapped_launch, original_events = _build_harvest_shadow(
            output, shadow_root, launch_records
        )
        mapped_production_root = str(provider_root.joinpath(*production_root.parts[1:]))
        source = package_manifest.get("source_notebook")
        instrumentation = package_manifest.get("instrumentation")
        public_hashes = source.get("cell_source_sha256") if isinstance(source, Mapping) else None
        auxiliary_hashes = (
            instrumentation.get("allowed_auxiliary_cell_sha256")
            if isinstance(instrumentation, Mapping)
            else None
        )
        if not isinstance(public_hashes, list) or not isinstance(auxiliary_hashes, list):
            raise TelemetryVerificationError("R4 package telemetry hashes are unavailable")
        config = NotebookTelemetryConfig(
            output_dir=str(shadow_output.resolve()),
            expected_public_cell_sha256=tuple(public_hashes),
            allowed_auxiliary_cell_sha256=tuple(auxiliary_hashes),
            expected_shapes_tzyx=raw_shapes,
            production_data_root=mapped_production_root,
            production_test_names=production_names,
            sample_interval_seconds=5.0,
            nvidia_smi_timeout_seconds=2.0,
            max_samples=20_000,
            require_coordinate_artifacts=True,
            require_submission_crosscheck=True,
        )
        try:
            offline_harvest = harvester(
                config,
                production_data_root=mapped_production_root,
                production_test_names=production_names,
            )
        except Exception as exc:
            raise TelemetryVerificationError("portable offline telemetry harvest failed") from exc
        if not isinstance(offline_harvest, dict):
            raise TelemetryVerificationError("offline harvester returned a non-object")
        _restore_original_event_evidence(offline_harvest, original_events)
        normalized_harvest = _normalize_paths(offline_harvest, shadow_output, provider_root)

    telemetry_path = output / "e0_notebook_telemetry.json"
    telemetry = _strict_json_object(telemetry_path, "E0 notebook telemetry")
    if (
        set(telemetry)
        != {
            "schema_version",
            "status",
            "started_at_utc",
            "finalized_at_utc",
            "runtime_scopes",
            "resources",
            "harvest",
        }
        or telemetry.get("schema_version") != 1
        or telemetry.get("status") != "PASS"
    ):
        raise TelemetryVerificationError("final notebook telemetry did not pass its exact schema")
    recorded_harvest = _normalize_paths(telemetry.get("harvest"), output, output / "unused")
    if normalized_harvest != recorded_harvest:
        raise TelemetryVerificationError("offline re-harvest differs from final notebook telemetry")

    _verify_manifest_binding(telemetry, telemetry_path, run_manifest)
    cell_summary = _verify_cells(telemetry, output, package_manifest)
    callback_summary = _verify_callback(
        output,
        artifact_lock,
        observed_launch_count=sum(
            record.get("event") == "SHARD_LAUNCHED" for record in launch_records
        ),
    )
    sample_summary = _verify_samples(telemetry, output)
    runtime_summary = _verify_runtime_scopes(telemetry)
    stage_summary = _verify_stage_and_process_coverage(normalized_harvest)

    files = _relevant_evidence_files(output)
    if files != initial_files:
        raise TelemetryVerificationError("downloaded evidence changed during verification")
    try:
        preflight_package(package, E0_R4_PACKAGE_IDENTITY)
    except Exception as exc:
        raise TelemetryVerificationError("reviewed package changed during verification") from exc

    generated_at = now_factory()
    if generated_at.tzinfo is None or generated_at.utcoffset() != UTC.utcoffset(generated_at):
        raise TelemetryVerificationError("report time must be UTC-aware")
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "E0_R4_DOWNLOADED_TELEMETRY_VERIFICATION",
        "status": "PASS",
        "generated_at": generated_at.isoformat(),
        "reviewed_package_generation": "E0_R4",
        "package_dir": str(package),
        "package_identity": asdict(E0_R4_PACKAGE_IDENTITY),
        "output_dir": str(output),
        "read_only_evidence_operation": True,
        "provider_operation_performed": False,
        "input_files": files,
        "production_data_root": production_root.as_posix(),
        "production_test_names": list(production_names),
        "cell_coverage": cell_summary,
        "callback_cleanup": callback_summary,
        "whole_wrapper_resources": sample_summary,
        "runtime_scopes": runtime_summary,
        "support_stage_coverage": stage_summary,
        "offline_reharvest_status": normalized_harvest.get("status"),
        "limitations": [
            "provider-complete elapsed time is unavailable inside notebook telemetry",
            "whole-wrapper host/GPU peaks are five-second sampled maxima, not continuous high-water marks",
            "host/GPU sensor availability is reported per sample; unavailable probes do not invalidate maxima from measured samples",
            "device-wide nvidia-smi samples may include unrelated GPU processes",
            "support process CUDA and ru_maxrss peaks are validated emitted evidence, not independently recomputed",
            "retention, coordinate, run-stat, and submission semantics reuse harvest_e0_telemetry and are not separately reimplemented",
        ],
    }
    # Return the same JSON data model that was persisted (for example, dataclass
    # tuples become JSON arrays) so callers can compare the in-memory result to
    # the report bytes without Python-only container differences.
    persisted_report = json.loads(json.dumps(report, allow_nan=False))
    _write_report(target, persisted_report)
    return persisted_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=ROOT / DEFAULT_PACKAGE_DIRS["r4"],
        help="Exact reviewed R4 package; bytes must match the compiled identity",
    )
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = verify_downloaded_e0_telemetry(
        output_dir=args.output_dir,
        package_dir=args.package_dir,
        report_json=args.report_json,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
