"""Additive, self-contained telemetry for the frozen E0 public notebook.

The source factory embeds this module in a prepended notebook cell.  The
embedded runtime uses only the Python standard library, does not import torch,
and keeps provider elapsed time distinct from notebook-local clocks.
"""

from __future__ import annotations

import atexit
import csv
import hashlib
import json
import math
import os
import re
import secrets
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


class E0TelemetryError(RuntimeError):
    """Notebook telemetry is incomplete, inconsistent, or unsafe to trust."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RUNTIME_MODULE_NAME = "_e0_notebook_telemetry_runtime"
_TRACKER_GLOBAL = "_E0_NOTEBOOK_TELEMETRY"


@dataclass(frozen=True)
class NotebookTelemetryConfig:
    """Validated configuration shared by the source factory and runtime."""

    output_dir: str
    expected_public_cell_sha256: tuple[str, ...]
    allowed_auxiliary_cell_sha256: tuple[str, ...] = ()
    expected_shapes_tzyx: Mapping[str, Sequence[int]] | None = None
    production_data_root: str | None = None
    production_test_names: tuple[str, ...] | None = None
    sample_interval_seconds: float = 5.0
    nvidia_smi_timeout_seconds: float = 2.0
    max_samples: int = 20_000
    retention_glob: str = "retention_guard_*.jsonl"
    coordinate_glob: str = "detector_coordinates_*.jsonl"
    support_telemetry_dirname: str = "e0_support_telemetry"
    run_stats_filename: str = "run_stats.csv"
    submission_filename: str = "submission.csv"
    run_manifest_filename: str = "public_reference_run_manifest.json"
    telemetry_filename: str = "e0_notebook_telemetry.json"
    samples_filename: str = "e0_resource_samples.jsonl"
    cell_events_filename: str = "e0_cell_events.jsonl"
    violations_filename: str = "e0_cell_violations.jsonl"
    require_coordinate_artifacts: bool = True
    require_submission_crosscheck: bool = True

    def validated(self) -> NotebookTelemetryConfig:
        output = Path(self.output_dir)
        output_is_posix_absolute = (
            isinstance(self.output_dir, str) and PurePosixPath(self.output_dir).is_absolute()
        )
        if not self.output_dir or not (output.is_absolute() or output_is_posix_absolute):
            raise E0TelemetryError("output_dir must be an absolute path")
        normalized_output = self.output_dir if output_is_posix_absolute else str(output)
        if not self.expected_public_cell_sha256:
            raise E0TelemetryError("expected public-cell hashes must be nonempty")
        expected = _validated_hashes(
            self.expected_public_cell_sha256, "expected public-cell hashes"
        )
        allowed = _validated_hashes(
            self.allowed_auxiliary_cell_sha256, "allowed auxiliary-cell hashes"
        )
        if len(set(expected)) != len(expected):
            raise E0TelemetryError("expected public-cell hashes must be unique")
        if set(expected) & set(allowed):
            raise E0TelemetryError("public and auxiliary cell hashes must be disjoint")
        if not math.isfinite(self.sample_interval_seconds) or self.sample_interval_seconds <= 0:
            raise E0TelemetryError("sample interval must be finite and positive")
        if (
            not math.isfinite(self.nvidia_smi_timeout_seconds)
            or self.nvidia_smi_timeout_seconds <= 0
        ):
            raise E0TelemetryError("nvidia-smi timeout must be finite and positive")
        if isinstance(self.max_samples, bool) or not isinstance(self.max_samples, int):
            raise E0TelemetryError("max_samples must be a positive integer")
        if self.max_samples <= 0:
            raise E0TelemetryError("max_samples must be a positive integer")
        for value, name in (
            (self.run_stats_filename, "run_stats_filename"),
            (self.submission_filename, "submission_filename"),
            (self.run_manifest_filename, "run_manifest_filename"),
            (self.telemetry_filename, "telemetry_filename"),
            (self.samples_filename, "samples_filename"),
            (self.cell_events_filename, "cell_events_filename"),
            (self.violations_filename, "violations_filename"),
            (self.support_telemetry_dirname, "support_telemetry_dirname"),
        ):
            _validate_basename(value, name)
        for value, name in (
            (self.retention_glob, "retention_glob"),
            (self.coordinate_glob, "coordinate_glob"),
        ):
            if not value or PurePosixPath(value).name != value or ".." in value:
                raise E0TelemetryError(f"{name} must be a single-directory glob")
        shapes = _validate_shapes(self.expected_shapes_tzyx)
        production_data_root = self.production_data_root
        if production_data_root is not None:
            if not isinstance(production_data_root, str):
                raise E0TelemetryError("production_data_root must be an absolute path")
            native_root = Path(production_data_root)
            posix_root = PurePosixPath(production_data_root)
            if not (native_root.is_absolute() or posix_root.is_absolute()):
                raise E0TelemetryError("production_data_root must be an absolute path")
            production_data_root = (
                production_data_root
                if posix_root.is_absolute()
                else str(native_root.resolve(strict=False))
            )
        production_test_names = _validate_test_names(self.production_test_names)
        return NotebookTelemetryConfig(
            output_dir=normalized_output,
            expected_public_cell_sha256=expected,
            allowed_auxiliary_cell_sha256=allowed,
            expected_shapes_tzyx=shapes,
            production_data_root=production_data_root,
            production_test_names=production_test_names,
            sample_interval_seconds=float(self.sample_interval_seconds),
            nvidia_smi_timeout_seconds=float(self.nvidia_smi_timeout_seconds),
            max_samples=self.max_samples,
            retention_glob=self.retention_glob,
            coordinate_glob=self.coordinate_glob,
            support_telemetry_dirname=self.support_telemetry_dirname,
            run_stats_filename=self.run_stats_filename,
            submission_filename=self.submission_filename,
            run_manifest_filename=self.run_manifest_filename,
            telemetry_filename=self.telemetry_filename,
            samples_filename=self.samples_filename,
            cell_events_filename=self.cell_events_filename,
            violations_filename=self.violations_filename,
            require_coordinate_artifacts=bool(self.require_coordinate_artifacts),
            require_submission_crosscheck=bool(self.require_submission_crosscheck),
        )

    def to_json_value(self) -> dict[str, object]:
        value = asdict(self.validated())
        value["expected_public_cell_sha256"] = list(self.expected_public_cell_sha256)
        value["allowed_auxiliary_cell_sha256"] = list(
            self.allowed_auxiliary_cell_sha256
        )
        if self.expected_shapes_tzyx is not None:
            value["expected_shapes_tzyx"] = {
                key: list(shape) for key, shape in self.expected_shapes_tzyx.items()
            }
        if self.production_test_names is not None:
            value["production_test_names"] = list(self.production_test_names)
        return value

    @classmethod
    def from_json_value(cls, value: Mapping[str, object]) -> NotebookTelemetryConfig:
        if not isinstance(value, Mapping):
            raise E0TelemetryError("telemetry configuration must be an object")
        try:
            return cls(**dict(value)).validated()
        except TypeError as exc:
            raise E0TelemetryError("telemetry configuration has invalid fields") from exc


@dataclass(frozen=True)
class NotebookTelemetrySources:
    """Self-contained sources and their exact identities."""

    prepended_source: str
    final_source: str
    runtime_source_sha256: str
    prepended_source_sha256: str
    final_source_sha256: str


def _validated_hashes(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise E0TelemetryError(f"{field_name} must be a sequence")
    result = tuple(values)
    if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in result):
        raise E0TelemetryError(f"{field_name} contain an invalid SHA-256")
    return result


def _validate_basename(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or PurePosixPath(value).name != value:
        raise E0TelemetryError(f"{field_name} must be a basename")


def _validate_shapes(
    value: Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[int, int, int, int]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise E0TelemetryError("expected_shapes_tzyx must be a nonempty mapping")
    result: dict[str, tuple[int, int, int, int]] = {}
    for dataset, raw_shape in value.items():
        if not isinstance(dataset, str) or not dataset:
            raise E0TelemetryError("expected shape has an invalid dataset ID")
        if (
            isinstance(raw_shape, (str, bytes))
            or not isinstance(raw_shape, Sequence)
            or len(raw_shape) != 4
        ):
            raise E0TelemetryError(f"shape for {dataset} must be [t,z,y,x]")
        shape = tuple(raw_shape)
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        ):
            raise E0TelemetryError(f"shape for {dataset} has invalid dimensions")
        result[dataset] = shape  # type: ignore[assignment]
    return dict(sorted(result.items()))


def _validate_test_names(value: Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise E0TelemetryError("production_test_names must be a nonempty sequence")
    result = tuple(value)
    if any(
        not isinstance(name, str)
        or not name
        or PurePosixPath(name).name != name
        or "\\" in name
        for name in result
    ):
        raise E0TelemetryError("production_test_names contain an invalid name")
    if len(result) != len(set(result)):
        raise E0TelemetryError("production_test_names must be unique")
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _strict_json_text(text: str, label: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise E0TelemetryError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise E0TelemetryError(f"nonfinite JSON value in {label}: {value}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise E0TelemetryError(f"invalid JSON in {label}") from exc


def _strict_json_object(path: Path) -> dict[str, object]:
    try:
        value = _strict_json_text(path.read_text(encoding="utf-8"), str(path))
    except (OSError, UnicodeError) as exc:
        raise E0TelemetryError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise E0TelemetryError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    encoded = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise E0TelemetryError(f"{field_name} must be an integer >= {minimum}")
    return value


def _float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise E0TelemetryError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise E0TelemetryError(f"{field_name} must be finite")
    return result


def _read_proc_snapshot(root_pid: int | None = None) -> dict[str, object]:
    """Return aggregate RSS for a Linux process tree without third-party imports."""

    root = os.getpid() if root_pid is None else root_pid
    proc = Path("/proc")
    if os.name != "posix" or not proc.is_dir():
        return {
            "status": "UNAVAILABLE",
            "reason": "procfs is unavailable",
            "backend": "procfs",
        }
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError) as exc:
        return {
            "status": "UNAVAILABLE",
            "reason": f"cannot determine page size: {type(exc).__name__}",
            "backend": "procfs",
        }
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for child in proc.iterdir():
        if not child.name.isdigit():
            continue
        pid = int(child.name)
        try:
            stat_text = (child / "stat").read_text(encoding="ascii")
            tail = stat_text[stat_text.rfind(")") + 2 :].split()
            parents[pid] = int(tail[1])
            statm = (child / "statm").read_text(encoding="ascii").split()
            rss[pid] = int(statm[1]) * int(page_size)
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    included = {root}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in included and pid not in included:
                included.add(pid)
                changed = True
    observed = sorted(pid for pid in included if pid in rss)
    if root not in rss:
        return {
            "status": "UNAVAILABLE",
            "reason": "root process disappeared or procfs was unreadable",
            "backend": "procfs",
        }
    return {
        "status": "MEASURED",
        "backend": "procfs",
        "root_pid": root,
        "process_count": len(observed),
        "rss_bytes": sum(rss[pid] for pid in observed),
    }


def _read_nvidia_smi_snapshot(timeout_seconds: float) -> dict[str, object]:
    """Read per-GPU memory without importing torch or creating a CUDA context."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="strict",
            shell=False,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, UnicodeError) as exc:
        return {
            "status": "UNAVAILABLE",
            "reason": f"nvidia-smi failed: {type(exc).__name__}",
            "backend": "nvidia-smi",
        }
    if completed.returncode != 0:
        return {
            "status": "UNAVAILABLE",
            "reason": f"nvidia-smi exit code {completed.returncode}",
            "backend": "nvidia-smi",
        }
    devices: list[dict[str, object]] = []
    try:
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 4:
                raise ValueError("unexpected column count")
            index = int(fields[0])
            used_mib = int(fields[2])
            total_mib = int(fields[3])
            if index < 0 or used_mib < 0 or total_mib <= 0 or used_mib > total_mib:
                raise ValueError("invalid GPU memory value")
            devices.append(
                {
                    "index": index,
                    "uuid": fields[1],
                    "used_bytes": used_mib * 1024 * 1024,
                    "total_bytes": total_mib * 1024 * 1024,
                }
            )
    except ValueError as exc:
        return {
            "status": "UNAVAILABLE",
            "reason": f"cannot parse nvidia-smi output: {exc}",
            "backend": "nvidia-smi",
        }
    if not devices:
        return {
            "status": "UNAVAILABLE",
            "reason": "nvidia-smi returned no devices",
            "backend": "nvidia-smi",
        }
    return {"status": "MEASURED", "backend": "nvidia-smi", "devices": devices}


HostProbe = Callable[[], Mapping[str, object]]
GpuProbe = Callable[[], Mapping[str, object]]


class ResourceSampler:
    """Bounded low-rate process-tree RSS and per-GPU memory sampler."""

    def __init__(
        self,
        *,
        path: Path,
        interval_seconds: float,
        nvidia_smi_timeout_seconds: float,
        max_samples: int,
        host_probe: HostProbe | None = None,
        gpu_probe: GpuProbe | None = None,
    ) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self.max_samples = max_samples
        self._host_probe = host_probe or _read_proc_snapshot
        self._gpu_probe = gpu_probe or (
            lambda: _read_nvidia_smi_snapshot(nvidia_smi_timeout_seconds)
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sample_count = 0
        self._dropped_samples = 0
        self._write_errors = 0
        self._max_host_rss_bytes: int | None = None
        self._max_gpu_used_bytes: dict[str, int] = {}
        self._gpu_total_bytes: dict[str, int] = {}
        self._last_host_unavailable_reason: str | None = None
        self._last_gpu_unavailable_reason: str | None = None
        self._last_persisted_monotonic_seconds: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise E0TelemetryError("resource sampler was already started")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="e0-resource-sampler", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def _sample_once(self) -> None:
        with self._lock:
            if self._sample_count >= self.max_samples:
                self._dropped_samples += 1
                self._stop.set()
                return
        sample: dict[str, object] = {
            "schema_version": 1,
            "captured_at_utc": _utc_now(),
            "monotonic_seconds": time.monotonic(),
        }
        try:
            host = dict(self._host_probe())
        except Exception as exc:  # noqa: BLE001 - contain an injected probe failure
            host = {
                "status": "UNAVAILABLE",
                "reason": f"host probe raised {type(exc).__name__}",
                "backend": "callback",
            }
        try:
            gpu = dict(self._gpu_probe())
        except Exception as exc:  # noqa: BLE001 - contain an injected probe failure
            gpu = {
                "status": "UNAVAILABLE",
                "reason": f"GPU probe raised {type(exc).__name__}",
                "backend": "callback",
            }
        sample["host"] = host
        sample["gpus"] = gpu
        with self._lock:
            try:
                _append_jsonl(self.path, sample)
                self._sample_count += 1
                self._last_persisted_monotonic_seconds = float(sample["monotonic_seconds"])
            except (OSError, TypeError, ValueError):
                self._write_errors += 1
                return
            if host.get("status") == "MEASURED":
                rss = host.get("rss_bytes")
                if isinstance(rss, int) and not isinstance(rss, bool) and rss >= 0:
                    self._max_host_rss_bytes = max(self._max_host_rss_bytes or 0, rss)
            else:
                self._last_host_unavailable_reason = str(host.get("reason", "unknown"))
            if gpu.get("status") == "MEASURED" and isinstance(gpu.get("devices"), list):
                for device in gpu["devices"]:
                    if not isinstance(device, Mapping):
                        continue
                    uuid = device.get("uuid")
                    used = device.get("used_bytes")
                    if isinstance(uuid, str) and isinstance(used, int) and not isinstance(used, bool):
                        self._max_gpu_used_bytes[uuid] = max(
                            self._max_gpu_used_bytes.get(uuid, 0), used
                        )
                        total = device.get("total_bytes")
                        if isinstance(total, int) and not isinstance(total, bool) and total > 0:
                            prior = self._gpu_total_bytes.get(uuid)
                            if prior is not None and prior != total:
                                self._last_gpu_unavailable_reason = (
                                    f"GPU total memory changed for {uuid}: {prior} to {total}"
                                )
                            self._gpu_total_bytes[uuid] = total
            else:
                self._last_gpu_unavailable_reason = str(gpu.get("reason", "unknown"))

    def stop(self, timeout_seconds: float | None = None) -> dict[str, object]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            timeout = max(1.0, self.interval_seconds * 2) if timeout_seconds is None else timeout_seconds
            thread.join(timeout)
        alive = bool(thread and thread.is_alive())
        with self._lock:
            partial = self._dropped_samples > 0
            failed = alive or bool(self._write_errors) or partial
            measured_status = "LOWER_BOUND" if partial else "MEASURED"
            return {
                "status": "ERROR" if failed else "PASS",
                "coverage_status": "PARTIAL_CAP_EXHAUSTED" if partial else "COMPLETE",
                "cleanup_complete": not alive,
                "sample_count": self._sample_count,
                "dropped_samples": self._dropped_samples,
                "write_errors": self._write_errors,
                "sample_interval_seconds": self.interval_seconds,
                "sampled_until_monotonic_seconds": self._last_persisted_monotonic_seconds,
                "samples_path": str(self.path),
                "samples_sha256": _sha256_file(self.path) if self.path.is_file() else None,
                "host_peak": (
                    {"status": measured_status, "rss_bytes": self._max_host_rss_bytes}
                    if self._max_host_rss_bytes is not None
                    else {
                        "status": "UNAVAILABLE",
                        "reason": self._last_host_unavailable_reason or "no successful samples",
                    }
                ),
                "gpu_peaks": (
                    {
                        "status": measured_status,
                        "used_bytes_by_uuid": dict(sorted(self._max_gpu_used_bytes.items())),
                        "total_bytes_by_uuid": dict(sorted(self._gpu_total_bytes.items())),
                    }
                    if self._max_gpu_used_bytes
                    else {
                        "status": "UNAVAILABLE",
                        "reason": self._last_gpu_unavailable_reason or "no successful samples",
                    }
                ),
            }


@dataclass
class _ActiveCell:
    digest: str
    role: str
    expected_public_index: int | None
    started_monotonic: float
    started_at_utc: str


@dataclass
class NotebookTelemetryTracker:
    """Tracks exact public cells, resources, strict harvest, and final manifest update."""

    config: NotebookTelemetryConfig
    host_probe: HostProbe | None = None
    gpu_probe: GpuProbe | None = None
    _started_monotonic: float = field(init=False, default=0.0)
    _started_at_utc: str = field(init=False, default="")
    _public_cursor: int = field(init=False, default=0)
    _active: _ActiveCell | None = field(init=False, default=None)
    _cell_records: list[dict[str, object]] = field(init=False, default_factory=list)
    _violations: list[dict[str, object]] = field(init=False, default_factory=list)
    _violation_write_errors: int = field(init=False, default=0)
    _ipython: object | None = field(init=False, default=None)
    _sampler: ResourceSampler | None = field(init=False, default=None)
    _serialization_records: list[dict[str, object]] = field(
        init=False, default_factory=list
    )
    _dataframe_type: type | None = field(init=False, default=None)
    _original_to_csv: Callable[..., object] | None = field(init=False, default=None)
    _started: bool = field(init=False, default=False)
    _finalized: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.config = self.config.validated()

    def start(self, ipython: object) -> None:
        if self._started:
            raise E0TelemetryError("notebook telemetry was already started")
        events = getattr(ipython, "events", None)
        if events is None or not callable(getattr(events, "register", None)):
            raise E0TelemetryError("IPython event callbacks are unavailable")
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        samples = output / self.config.samples_filename
        for event_path in (
            samples,
            output / self.config.cell_events_filename,
            output / self.config.violations_filename,
        ):
            if event_path.exists():
                raise E0TelemetryError(
                    f"refusing to append to existing telemetry file: {event_path}"
                )
        self._started_monotonic = time.monotonic()
        self._started_at_utc = _utc_now()
        events.register("pre_run_cell", self.pre_run_cell)
        events.register("post_run_cell", self.post_run_cell)
        self._ipython = ipython
        self._sampler = ResourceSampler(
            path=samples,
            interval_seconds=self.config.sample_interval_seconds,
            nvidia_smi_timeout_seconds=self.config.nvidia_smi_timeout_seconds,
            max_samples=self.config.max_samples,
            host_probe=self.host_probe,
            gpu_probe=self.gpu_probe,
        )
        self._sampler.start()
        atexit.register(self._atexit_cleanup)
        self._started = True

    def _atexit_cleanup(self) -> None:
        sampler = self._sampler
        if sampler is not None and not self._finalized:
            sampler.stop()

    def pre_run_cell(self, info: object) -> None:
        if self._active is not None:
            self._record_violation(
                "OVERLAPPING_CELL_CALLBACKS",
                "a cell started before the prior cell completed",
            )
            raise E0TelemetryError("a cell started before the prior cell completed")
        raw_cell = getattr(info, "raw_cell", None)
        if not isinstance(raw_cell, str):
            self._record_violation(
                "PRE_SOURCE_UNAVAILABLE",
                "IPython pre_run_cell did not provide raw cell source",
            )
            raise E0TelemetryError("IPython pre_run_cell did not provide raw cell source")
        digest = _sha256_bytes(raw_cell.encode("utf-8"))
        expected = self.config.expected_public_cell_sha256
        expected_index: int | None = None
        if self._public_cursor < len(expected) and digest == expected[self._public_cursor]:
            role = "public"
            expected_index = self._public_cursor
        elif digest in self.config.allowed_auxiliary_cell_sha256:
            role = "instrumentation"
        else:
            self._record_violation(
                "UNEXPECTED_CELL_SOURCE",
                f"unexpected source hash at public cursor {self._public_cursor}: {digest}",
            )
            raise E0TelemetryError(
                "unexpected notebook cell source hash at public cursor "
                f"{self._public_cursor}: {digest}"
            )
        self._active = _ActiveCell(
            digest=digest,
            role=role,
            expected_public_index=expected_index,
            started_monotonic=time.monotonic(),
            started_at_utc=_utc_now(),
        )
        self._install_to_csv_wrapper_if_available()

    def post_run_cell(self, result: object) -> None:
        active = self._active
        if active is None:
            # Registration occurs during the prepended cell, so that cell can
            # produce a post event without a corresponding observed pre event.
            return
        ended = time.monotonic()
        result_info = getattr(result, "info", None)
        post_raw_cell = getattr(result_info, "raw_cell", None)
        post_digest = (
            _sha256_bytes(post_raw_cell.encode("utf-8"))
            if isinstance(post_raw_cell, str)
            else None
        )
        post_hash_matches = post_digest == active.digest
        if not post_hash_matches:
            self._record_violation(
                "POST_SOURCE_HASH_MISMATCH",
                f"pre={active.digest}, post={post_digest}",
            )
        error = getattr(result, "error_before_exec", None) or getattr(
            result, "error_in_exec", None
        )
        record: dict[str, object] = {
            "role": active.role,
            "sha256": active.digest,
            "started_at_utc": active.started_at_utc,
            "ended_at_utc": _utc_now(),
            "elapsed_seconds": ended - active.started_monotonic,
            "post_sha256": post_digest,
            "post_hash_matches_pre": post_hash_matches,
            "status": "PASS" if error is None and post_hash_matches else "ERROR",
            "error_type": (
                None
                if error is None and post_hash_matches
                else (
                    type(error).__name__
                    if error is not None
                    else "PostCellSourceHashMismatch"
                )
            ),
        }
        if active.expected_public_index is not None:
            record["public_index"] = active.expected_public_index
            self._public_cursor += 1
        self._cell_records.append(record)
        try:
            _append_jsonl(
                Path(self.config.output_dir) / self.config.cell_events_filename,
                record,
            )
        except (OSError, TypeError, ValueError) as exc:
            self._record_violation(
                "CELL_EVENT_WRITE_FAILED",
                f"cannot persist cell event: {type(exc).__name__}",
            )
        self._active = None

    def _record_violation(self, kind: str, detail: str) -> None:
        record = {
            "kind": kind,
            "detail": detail,
            "captured_at_utc": _utc_now(),
            "monotonic_seconds": time.monotonic(),
        }
        self._violations.append(record)
        try:
            _append_jsonl(
                Path(self.config.output_dir) / self.config.violations_filename,
                record,
            )
        except (OSError, TypeError, ValueError):
            self._violation_write_errors += 1

    def _unregister_callbacks(self) -> None:
        ipython = self._ipython
        events = getattr(ipython, "events", None) if ipython is not None else None
        if events is not None and callable(getattr(events, "unregister", None)):
            for event, callback in (
                ("pre_run_cell", self.pre_run_cell),
                ("post_run_cell", self.post_run_cell),
            ):
                try:
                    events.unregister(event, callback)
                except (KeyError, ValueError):
                    pass
        # The final auxiliary cell is active when it calls finalize. It is not
        # a completed public cell and is deliberately excluded from cell time.
        self._active = None
        self._restore_to_csv_wrapper()

    def _install_to_csv_wrapper_if_available(self) -> None:
        """Time only writes to the final submission path, preserving pandas behavior."""

        if self._original_to_csv is not None:
            return
        pandas = sys.modules.get("pandas")
        dataframe_type = getattr(pandas, "DataFrame", None) if pandas is not None else None
        original = getattr(dataframe_type, "to_csv", None)
        if not isinstance(dataframe_type, type) or not callable(original):
            return
        target = (Path(self.config.output_dir) / self.config.submission_filename).resolve(
            strict=False
        )
        tracker = self

        def timed_to_csv(frame: object, *args: object, **kwargs: object) -> object:
            destination = args[0] if args else kwargs.get("path_or_buf")
            timed = False
            if isinstance(destination, (str, os.PathLike)):
                try:
                    timed = Path(destination).resolve(strict=False) == target
                except (OSError, TypeError, ValueError):
                    timed = False
            started = time.monotonic()
            started_at = _utc_now()
            error_type: str | None = None
            try:
                return original(frame, *args, **kwargs)
            except BaseException as exc:
                error_type = type(exc).__name__
                raise
            finally:
                if timed:
                    tracker._serialization_records.append(
                        {
                            "stage": "submission_csv_serialization",
                            "status": "PASS" if error_type is None else "ERROR",
                            "error_type": error_type,
                            "started_at_utc": started_at,
                            "ended_at_utc": _utc_now(),
                            "elapsed_seconds": time.monotonic() - started,
                            "path": str(target),
                        }
                    )

        type.__setattr__(dataframe_type, "to_csv", timed_to_csv)
        self._dataframe_type = dataframe_type
        self._original_to_csv = original

    def _restore_to_csv_wrapper(self) -> None:
        if self._dataframe_type is not None and self._original_to_csv is not None:
            type.__setattr__(self._dataframe_type, "to_csv", self._original_to_csv)
        self._dataframe_type = None
        self._original_to_csv = None

    def finalize(
        self,
        *,
        production_data_root: str | os.PathLike[str] | None = None,
        production_test_names: Sequence[str] | None = None,
    ) -> dict[str, object]:
        if not self._started:
            raise E0TelemetryError("notebook telemetry was never started")
        if self._finalized:
            raise E0TelemetryError("notebook telemetry was already finalized")
        sampler = self._sampler
        if sampler is None:
            raise E0TelemetryError("resource sampler was not initialized")
        try:
            self._unregister_callbacks()
            if self._violations:
                raise E0TelemetryError(
                    f"notebook callback evidence has {len(self._violations)} violation(s)"
                )
            if self._violation_write_errors:
                raise E0TelemetryError("notebook callback violations could not be persisted")
            expected = self.config.expected_public_cell_sha256
            public = [record for record in self._cell_records if record["role"] == "public"]
            observed_hashes = [record["sha256"] for record in public]
            if self._public_cursor != len(expected) or observed_hashes != list(expected):
                raise E0TelemetryError(
                    f"public-cell execution incomplete: observed {len(public)} of {len(expected)}"
                )
            if any(record["status"] != "PASS" for record in public):
                raise E0TelemetryError("one or more exact public cells failed")
            harvested = harvest_e0_telemetry(
                self.config,
                production_data_root=production_data_root,
                production_test_names=production_test_names,
            )
        except BaseException:
            sampler.stop()
            atexit.unregister(self._atexit_cleanup)
            raise
        resources = sampler.stop()
        atexit.unregister(self._atexit_cleanup)
        wrapper_elapsed = time.monotonic() - self._started_monotonic
        result: dict[str, object] = {
            "schema_version": 1,
            "status": "PASS" if resources["status"] == "PASS" else "ERROR",
            "started_at_utc": self._started_at_utc,
            "finalized_at_utc": _utc_now(),
            "runtime_scopes": {
                "wrapper": {
                    "status": "MEASURED",
                    "elapsed_seconds": wrapper_elapsed,
                    "start_boundary": "prepended telemetry cell after kernel startup",
                    "end_boundary": (
                        "final telemetry cell after strict harvest and sampler cleanup, "
                        "before telemetry/manifest writes"
                    ),
                },
                "cells": {
                    "status": "MEASURED",
                    "records": self._cell_records,
                    "violations": self._violations,
                },
                "submission_csv_serialization": {
                    "status": "MEASURED" if self._serialization_records else "UNAVAILABLE",
                    "reason": (
                        None
                        if self._serialization_records
                        else "pandas DataFrame.to_csv target call was not observed"
                    ),
                    "records": self._serialization_records,
                },
                "provider": {
                    "status": "UNAVAILABLE",
                    "reason": "provider-complete elapsed is external to notebook execution",
                    "elapsed_seconds": None,
                },
            },
            "resources": {"scope": "whole_notebook_wrapper", **resources},
            "harvest": harvested,
        }
        telemetry_path = Path(self.config.output_dir) / self.config.telemetry_filename
        _write_json_atomic(telemetry_path, result)
        telemetry_sha256 = _sha256_file(telemetry_path)
        self._update_run_manifest(
            telemetry_path,
            telemetry_sha256,
            wrapper_elapsed,
            telemetry_status=str(result["status"]),
        )
        self._finalized = True
        if result["status"] != "PASS":
            raise E0TelemetryError("resource telemetry failed; failure evidence was persisted")
        return {
            "status": result["status"],
            "path": str(telemetry_path),
            "sha256": telemetry_sha256,
            "wrapper_elapsed_seconds": wrapper_elapsed,
            "provider_elapsed_seconds": None,
        }

    def _update_run_manifest(
        self,
        telemetry_path: Path,
        telemetry_sha256: str,
        wrapper_elapsed: float,
        *,
        telemetry_status: str,
    ) -> None:
        manifest_path = Path(self.config.output_dir) / self.config.run_manifest_filename
        manifest = _strict_json_object(manifest_path)
        if "e0_telemetry" in manifest:
            raise E0TelemetryError("run manifest already contains e0_telemetry")
        manifest["e0_telemetry"] = {
            "schema_version": 1,
            "status": telemetry_status,
            "path": str(telemetry_path),
            "sha256": telemetry_sha256,
            "wrapper_elapsed_seconds": wrapper_elapsed,
            "provider_elapsed_seconds": None,
            "provider_elapsed_status": "UNAVAILABLE_INSIDE_NOTEBOOK",
        }
        _write_json_atomic(manifest_path, manifest)


_COUNTER_SEMANTICS = {
    "gap_skipped_node_cap": "gap repair global node-cap branch entries",
    "gap2_skipped_cap": "gap-2 repair global cap branch entries",
    "safe_division_skipped_cap": "safe-division global cap branch entries",
    "motion_relink_fallback_raw": "motion relink retained raw edges after relink fallback",
    "motion_relink_skipped_large_frame": "motion relink skipped a large frame",
    "deepcenter_gap_missing": "DeepCenter gap checks unavailable due to missing prediction",
    "deepcenter_safe_div_missing": "DeepCenter division checks unavailable due to missing prediction",
    "short_track_filter_skipped_all": "short-track filter skipped removal to avoid emptying graph",
    "short_track_rescue_triggered": "short-track component rescue path triggered",
}


def _resolve_shapes(
    config: NotebookTelemetryConfig, manifest: Mapping[str, object]
) -> dict[str, tuple[int, int, int, int]]:
    configured = _validate_shapes(config.expected_shapes_tzyx)
    if configured is not None:
        return configured
    for key in ("actual_input_shapes_tzyx", "dataset_shapes"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            return _validate_shapes(value) or {}
    raise E0TelemetryError("expected TZYX shapes are unavailable")


def _read_jsonl(paths: Sequence[Path], label: str) -> list[dict[str, object]]:
    if not paths:
        raise E0TelemetryError(f"no {label} files matched")
    records: list[dict[str, object]] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise E0TelemetryError(f"cannot read {path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise E0TelemetryError(f"blank line in {path}:{line_number}")
            value = _strict_json_text(line, f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise E0TelemetryError(f"non-object record in {path}:{line_number}")
            records.append(value)
    return records


def _validate_retention(
    records: Sequence[Mapping[str, object]],
    shapes: Mapping[str, tuple[int, int, int, int]],
) -> dict[str, object]:
    observed: dict[tuple[str, int], dict[str, object]] = {}
    for record in records:
        dataset = record.get("dataset")
        if not isinstance(dataset, str) or dataset not in shapes:
            raise E0TelemetryError(f"retention record has unexpected dataset: {dataset!r}")
        frame_index = _integer(record.get("frame"), f"retention frame for {dataset}")
        if frame_index >= shapes[dataset][0]:
            raise E0TelemetryError(f"retention frame out of range for {dataset}: {frame_index}")
        key = (dataset, frame_index)
        if key in observed:
            raise E0TelemetryError(f"duplicate retention record for {dataset} frame {frame_index}")
        primary = _integer(record.get("primary_candidates"), "primary_candidates")
        blended = _integer(record.get("blended_candidates"), "blended_candidates")
        retention = _float(record.get("retention"), "retention")
        expected_retention = blended / primary if primary else 1.0
        if not math.isclose(retention, expected_retention, rel_tol=0.0, abs_tol=1e-12):
            raise E0TelemetryError(f"retention ratio mismatch for {dataset} frame {frame_index}")
        minimum = _float(record.get("minimum_retention"), "minimum_retention")
        use_primary = record.get("use_primary")
        if not isinstance(use_primary, bool):
            raise E0TelemetryError("use_primary must be boolean")
        if use_primary != (primary > 0 and retention < minimum):
            raise E0TelemetryError(f"use_primary mismatch for {dataset} frame {frame_index}")
        observed[key] = {
            "primary_candidates": primary,
            "blended_candidates": blended,
            "retention": retention,
            "minimum_retention": minimum,
            "use_primary": use_primary,
        }
    expected_keys = {
        (dataset, frame_index)
        for dataset, shape in shapes.items()
        for frame_index in range(shape[0])
    }
    if set(observed) != expected_keys:
        missing = sorted(expected_keys - set(observed))
        extra = sorted(set(observed) - expected_keys)
        raise E0TelemetryError(
            f"retention frame coverage mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    per_dataset: dict[str, object] = {}
    for dataset, shape in shapes.items():
        ordered = [observed[(dataset, frame)] for frame in range(shape[0])]
        per_dataset[dataset] = {
            "frames": shape[0],
            "fallback_frames": sum(bool(row["use_primary"]) for row in ordered),
            "zero_primary_frames": sum(row["primary_candidates"] == 0 for row in ordered),
            "zero_blended_frames": sum(row["blended_candidates"] == 0 for row in ordered),
            "primary_candidates": [row["primary_candidates"] for row in ordered],
            "blended_candidates": [row["blended_candidates"] for row in ordered],
            "use_primary": [row["use_primary"] for row in ordered],
        }
    return {
        "status": "PASS",
        "record_count": len(records),
        "semantics": "primary detector selected instead of blend for that frame",
        "datasets": per_dataset,
    }


def _artifact_path(output: Path, raw_path: object, dataset: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise E0TelemetryError(f"coordinate artifact path missing for {dataset}")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise E0TelemetryError(f"unsafe coordinate artifact path for {dataset}")
    path = output.joinpath(*pure.parts)
    try:
        path.resolve(strict=True).relative_to(output.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise E0TelemetryError(f"coordinate artifact escapes output directory for {dataset}") from exc
    if path.is_symlink() or not path.is_file():
        raise E0TelemetryError(f"coordinate artifact is not a regular file for {dataset}")
    return path


def _validate_coordinate_artifact(
    *,
    output: Path,
    dataset: str,
    record: Mapping[str, object],
    rows: int,
    shape: tuple[int, int, int, int],
) -> dict[str, object]:
    artifact = record.get("artifact")
    if not isinstance(artifact, Mapping):
        return {
            "status": "UNAVAILABLE",
            "reason": "coordinate manifest has no bound raw coordinate artifact",
        }
    if artifact.get("dtype") != "<i2" or artifact.get("shape") != [rows, 4]:
        raise E0TelemetryError(f"coordinate artifact contract mismatch for {dataset}")
    path = _artifact_path(output, artifact.get("path"), dataset)
    expected_bytes = rows * 4 * 2
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes or artifact.get("bytes") != actual_bytes:
        raise E0TelemetryError(f"coordinate artifact byte count mismatch for {dataset}")
    expected_hash = artifact.get("sha256")
    if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
        raise E0TelemetryError(f"invalid coordinate artifact hash for {dataset}")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash or actual_hash != record.get("coordinate_sha256"):
        raise E0TelemetryError(f"coordinate artifact hash mismatch for {dataset}")
    frame_counts = [0] * shape[0]
    with path.open("rb") as stream:
        row_index = 0
        for chunk in iter(lambda: stream.read(8 * 8192), b""):
            if len(chunk) % 8:
                raise E0TelemetryError(f"partial coordinate row for {dataset}")
            for t, z, y, x in struct.iter_unpack("<hhhh", chunk):
                if not (0 <= t < shape[0]):
                    raise E0TelemetryError(f"coordinate t out of bounds for {dataset} row {row_index}")
                if not (0 <= z < shape[1] and 0 <= y < shape[2] and 0 <= x < shape[3]):
                    raise E0TelemetryError(
                        f"coordinate spatial value out of bounds for {dataset} row {row_index}"
                    )
                frame_counts[t] += 1
                row_index += 1
    if row_index != rows:
        raise E0TelemetryError(f"coordinate row count mismatch for {dataset}")
    return {
        "status": "VERIFIED",
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "dense_frame_counts": frame_counts,
    }


def _validate_coordinates(
    records: Sequence[Mapping[str, object]],
    shapes: Mapping[str, tuple[int, int, int, int]],
    output: Path,
    *,
    require_artifacts: bool,
) -> dict[str, object]:
    by_dataset: dict[str, dict[str, object]] = {}
    for record in records:
        dataset = record.get("dataset")
        if not isinstance(dataset, str) or dataset not in shapes:
            raise E0TelemetryError(f"coordinate record has unexpected dataset: {dataset!r}")
        if dataset in by_dataset:
            raise E0TelemetryError(f"duplicate coordinate record for {dataset}")
        if record.get("columns") != ["t", "z", "y", "x"]:
            raise E0TelemetryError(f"coordinate columns mismatch for {dataset}")
        if record.get("dtype") != "<i2":
            raise E0TelemetryError(f"coordinate dtype mismatch for {dataset}")
        if record.get("stage") != "post_detection_pre_graph_pre_ilp":
            raise E0TelemetryError(f"coordinate stage mismatch for {dataset}")
        rows = _integer(record.get("rows"), f"coordinate rows for {dataset}")
        coordinate_hash = record.get("coordinate_sha256")
        if not isinstance(coordinate_hash, str) or not _SHA256_RE.fullmatch(coordinate_hash):
            raise E0TelemetryError(f"invalid coordinate hash for {dataset}")
        raw_counts = record.get("frame_counts")
        if not isinstance(raw_counts, list):
            raise E0TelemetryError(f"frame_counts must be a list for {dataset}")
        dense = [0] * shapes[dataset][0]
        seen: set[int] = set()
        for pair in raw_counts:
            if not isinstance(pair, list) or len(pair) != 2:
                raise E0TelemetryError(f"invalid frame-count pair for {dataset}")
            frame_index = _integer(pair[0], f"coordinate frame for {dataset}")
            count = _integer(pair[1], f"coordinate count for {dataset}")
            if frame_index >= len(dense) or frame_index in seen:
                raise E0TelemetryError(f"invalid or duplicate coordinate frame for {dataset}")
            dense[frame_index] = count
            seen.add(frame_index)
        if sum(dense) != rows:
            raise E0TelemetryError(f"coordinate frame counts do not sum to rows for {dataset}")
        artifact = _validate_coordinate_artifact(
            output=output,
            dataset=dataset,
            record=record,
            rows=rows,
            shape=shapes[dataset],
        )
        if require_artifacts and artifact["status"] != "VERIFIED":
            raise E0TelemetryError(f"raw coordinate artifact is required for {dataset}")
        if artifact["status"] == "VERIFIED" and artifact["dense_frame_counts"] != dense:
            raise E0TelemetryError(f"coordinate artifact frame counts mismatch for {dataset}")
        by_dataset[dataset] = {
            "rows": rows,
            "coordinate_sha256": coordinate_hash,
            "dense_frame_counts": dense,
            "artifact": artifact,
        }
    if set(by_dataset) != set(shapes):
        raise E0TelemetryError(
            "coordinate dataset coverage mismatch: "
            f"expected={sorted(shapes)}, observed={sorted(by_dataset)}"
        )
    return {"status": "PASS", "datasets": dict(sorted(by_dataset.items()))}


def _read_run_stats(path: Path, datasets: set[str]) -> dict[str, dict[str, int]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if fields is None or len(fields) != len(set(fields)):
                raise E0TelemetryError("run_stats has missing or duplicate columns")
            required = {"dataset", "raw_nodes", "raw_edges", "nodes", "edges"} | set(
                _COUNTER_SEMANTICS
            )
            missing = sorted(required - set(fields))
            if missing:
                raise E0TelemetryError(f"run_stats is missing required columns: {missing}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise E0TelemetryError(f"cannot read run_stats: {path}") from exc
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        dataset = row.get("dataset")
        if not isinstance(dataset, str) or dataset not in datasets or dataset in result:
            raise E0TelemetryError(f"invalid or duplicate run_stats dataset: {dataset!r}")
        numeric: dict[str, int] = {}
        for field_name in ("raw_nodes", "raw_edges", "nodes", "edges", *_COUNTER_SEMANTICS):
            try:
                value = int(row[field_name])
            except (KeyError, TypeError, ValueError) as exc:
                raise E0TelemetryError(
                    f"run_stats {field_name} is not an integer for {dataset}"
                ) from exc
            if value < 0 or str(value) != row[field_name].strip():
                raise E0TelemetryError(
                    f"run_stats {field_name} is not a canonical nonnegative integer for {dataset}"
                )
            numeric[field_name] = value
        result[dataset] = numeric
    if set(result) != datasets:
        raise E0TelemetryError("run_stats dataset coverage mismatch")
    return dict(sorted(result.items()))


def _submission_counts(path: Path, datasets: set[str]) -> dict[str, dict[str, int]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if fields is None or len(fields) != len(set(fields)):
                raise E0TelemetryError("submission has missing or duplicate columns")
            if not {"dataset", "row_type"}.issubset(fields):
                raise E0TelemetryError("submission lacks dataset or row_type")
            counts = {dataset: {"nodes": 0, "edges": 0} for dataset in datasets}
            for row in reader:
                dataset = row.get("dataset")
                row_type = row.get("row_type")
                if dataset not in counts or row_type not in ("node", "edge"):
                    raise E0TelemetryError("submission has unexpected dataset or row_type")
                key = "nodes" if row_type == "node" else "edges"
                counts[dataset][key] += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise E0TelemetryError(f"cannot read submission: {path}") from exc
    return counts


_SUPPORT_STAGES = (
    "data_read",
    "encode_tta",
    "detector_extraction",
    "pair_score",
    "threshold",
    "graph_build",
    "ilp",
    "geff",
)


def _resolved_absolute_path(value: object, field_name: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise E0TelemetryError(f"{field_name} must be a path")
    path = Path(value)
    if not path.is_absolute():
        raise E0TelemetryError(f"{field_name} must be absolute")
    return str(path.resolve(strict=False))


def _support_available_count(value: object, field_name: str) -> int:
    if not isinstance(value, Mapping) or value.get("status") != "available":
        raise E0TelemetryError(f"support {field_name} is unavailable")
    return _integer(value.get("value"), f"support {field_name}")


def _validate_support_duration(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise E0TelemetryError(f"support duration {field_name} must be an object")
    status = value.get("status")
    if status == "unavailable":
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason:
            raise E0TelemetryError(f"support duration {field_name} lacks unavailable reason")
        return dict(value)
    if status != "available" or value.get("unit") != "seconds":
        raise E0TelemetryError(f"support duration {field_name} has invalid status or unit")
    duration_ns = _integer(value.get("duration_ns"), f"support duration_ns {field_name}")
    seconds = _float(value.get("value"), f"support seconds {field_name}")
    if not math.isclose(seconds, duration_ns / 1_000_000_000, rel_tol=0.0, abs_tol=1e-12):
        raise E0TelemetryError(f"support duration unit conversion mismatch for {field_name}")
    return dict(value)


def _production_identity(
    config: NotebookTelemetryConfig,
    shapes: Mapping[str, tuple[int, int, int, int]],
    production_data_root: str | os.PathLike[str] | None,
    production_test_names: Sequence[str] | None,
) -> tuple[str, tuple[str, ...]]:
    root_value = production_data_root or config.production_data_root
    names_value = production_test_names or config.production_test_names
    if root_value is None or names_value is None:
        raise E0TelemetryError(
            "production data root and exact test names are required for support-event join"
        )
    root = _resolved_absolute_path(root_value, "production_data_root")
    names = _validate_test_names(names_value)
    if names is None:
        raise E0TelemetryError("production_test_names are unavailable")
    if set(names) != set(shapes):
        raise E0TelemetryError(
            "production test-name/shape coverage mismatch: "
            f"names={sorted(names)}, shapes={sorted(shapes)}"
        )
    return root, names


def _read_support_events(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if root.is_symlink() or not root.is_dir():
        raise E0TelemetryError(f"support telemetry directory is unavailable: {root}")
    event_paths = sorted(root.glob("invocation-*/events.jsonl"))
    if not event_paths:
        raise E0TelemetryError("support telemetry contains no invocation event files")
    records: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    seen_process_ids: set[str] = set()
    for path in event_paths:
        if path.is_symlink() or not path.is_file() or path.parent.parent != root:
            raise E0TelemetryError(f"unsafe support telemetry event path: {path}")
        file_records = _read_jsonl([path], "support telemetry event")
        process_ids = {record.get("process_id") for record in file_records}
        if len(process_ids) != 1 or not all(isinstance(value, str) and value for value in process_ids):
            raise E0TelemetryError(f"support event file has inconsistent process identity: {path}")
        process_id = next(iter(process_ids))
        if process_id in seen_process_ids:
            raise E0TelemetryError(f"duplicate support process identity: {process_id}")
        seen_process_ids.add(process_id)
        for record in file_records:
            if record.get("schema_version") != 1:
                raise E0TelemetryError("support event has an unsupported schema version")
            record_type = record.get("record_type")
            if record_type not in {
                "invocation_start",
                "invocation_finish",
                "retention",
                "coordinate_artifact",
                "dataset_summary",
                "process_summary",
            }:
                raise E0TelemetryError(f"support event has unknown record type: {record_type!r}")
            copied = dict(record)
            copied["_event_path"] = str(path)
            records.append(copied)
        files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "process_id": process_id,
            }
        )
    return records, files


def _argv_option(argv: Sequence[str], name: str) -> str | None:
    positions = [index for index, value in enumerate(argv) if value == name]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise E0TelemetryError(f"launch argv has invalid {name} option")
    return argv[positions[0] + 1]


def _read_launch_bindings(
    support_root: Path,
    *,
    data_root: str,
    exact_names: tuple[str, ...],
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    launch_path = support_root / "launch-events.jsonl"
    events = _read_jsonl([launch_path], "support launch event")
    event_types = [event.get("event") for event in events]
    if not events or event_types[0] != "SUPPORT_PATCHED":
        raise E0TelemetryError("support launch evidence does not start with SUPPORT_PATCHED")
    if event_types[-1] != "LAUNCH_OBSERVATION_FINISHED":
        raise E0TelemetryError("support launch evidence lacks final observation result")
    patched = [event for event in events if event.get("event") == "SUPPORT_PATCHED"]
    launched = [event for event in events if event.get("event") == "SHARD_LAUNCHED"]
    finished = [
        event for event in events if event.get("event") == "LAUNCH_OBSERVATION_FINISHED"
    ]
    if len(patched) != 1 or len(finished) != 1 or len(launched) not in {1, 2}:
        raise E0TelemetryError("support launch evidence has invalid event cardinality")
    if len(events) != 2 + len(launched):
        raise E0TelemetryError("support launch evidence has unexpected event types")
    finish = finished[0]
    if (
        finish.get("status") != "PASS"
        or finish.get("launch_count") != len(launched)
        or finish.get("expected_launch_count") != len(launched)
    ):
        raise E0TelemetryError("support launch observation did not pass")
    source_chain = patched[0].get("source_chain")
    if not isinstance(source_chain, Mapping):
        raise E0TelemetryError("support patch source chain is unavailable")
    output_source_hash = source_chain.get("telemetry_output_support_sha256")
    for name in (
        "pre_public_support_sha256",
        "public_patched_support_sha256",
        "telemetry_output_support_sha256",
        "runtime_module_sha256",
    ):
        value = source_chain.get(name)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise E0TelemetryError(f"support patch source chain has invalid {name}")

    bindings: dict[int, dict[str, object]] = {}
    worker_count = len(launched)
    seen_slices: set[int] = set()
    for event in launched:
        pid = _integer(event.get("pid"), "launched support PID", minimum=1)
        if pid in bindings:
            raise E0TelemetryError(f"duplicate launched support PID: {pid}")
        if event.get("source_sha256") != output_source_hash:
            raise E0TelemetryError("launched support source hash differs from patch chain")
        argv = event.get("argv")
        if (
            not isinstance(argv, list)
            or len(argv) < 2
            or not all(isinstance(value, str) for value in argv)
        ):
            raise E0TelemetryError("support launch argv is invalid")
        if argv[1].replace("\\", "/") != "scripts/predict_unet_transformer.py":
            raise E0TelemetryError("support launch used an unexpected entry point")
        launch_root = _argv_option(argv, "--data-dir")
        if _resolved_absolute_path(launch_root, "launch data root") != data_root:
            raise E0TelemetryError("support launch data root differs from production root")
        if _argv_option(argv, "--split") != "0":
            raise E0TelemetryError("support launch did not use fold 0")
        method = _argv_option(argv, "--method")
        if method is not None and not method:
            raise E0TelemetryError("support launch method is empty")
        raw_slice = _argv_option(argv, "--slice")
        if worker_count == 1:
            if raw_slice is not None:
                raise E0TelemetryError("single production launch unexpectedly used a slice")
            slice_index = 0
            expected_subset = exact_names
            expected_shard_values = {"unavailable", "single"}
        else:
            expected_slices = {f"{index}::{worker_count}": index for index in range(worker_count)}
            if raw_slice not in expected_slices:
                raise E0TelemetryError(f"support launch has invalid shard slice: {raw_slice!r}")
            slice_index = expected_slices[raw_slice]
            expected_subset = exact_names[slice_index::worker_count]
            expected_shard_values = {f"{slice_index}/{worker_count}"}
        if slice_index in seen_slices:
            raise E0TelemetryError(f"duplicate production shard slice: {slice_index}")
        seen_slices.add(slice_index)
        bindings[pid] = {
            "pid": pid,
            "argv": argv,
            "method": method,
            "slice_index": slice_index,
            "worker_count": worker_count,
            "expected_test_names": expected_subset,
            "expected_gpu_shard_values": expected_shard_values,
            "source_sha256": output_source_hash,
        }
    if seen_slices != set(range(worker_count)):
        raise E0TelemetryError("support launch slices are incomplete")
    return (
        bindings,
        {
            "path": str(launch_path),
            "bytes": launch_path.stat().st_size,
            "sha256": _sha256_file(launch_path),
            "source_chain": dict(source_chain),
            "launch_count": worker_count,
        },
    )


def _require_same_support_process(
    record: Mapping[str, object],
    start: Mapping[str, object],
    *,
    require_invocation_id: str | None,
) -> None:
    for key in ("process_id", "pid", "gpu_shard", "diagnostic_arm"):
        if record.get(key) != start.get(key):
            raise E0TelemetryError(f"support event process identity mismatch: {key}")
    if require_invocation_id is not None and record.get("invocation_id") != require_invocation_id:
        raise E0TelemetryError("support event invocation identity mismatch")


def _validate_support_resource(
    value: object,
    field_name: str,
    *,
    allow_native_rss_unit: bool = False,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise E0TelemetryError(f"support resource {field_name} must be an object")
    status = value.get("status")
    if status == "unavailable":
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason:
            raise E0TelemetryError(f"support resource {field_name} lacks unavailable reason")
        return dict(value)
    if status != "available":
        raise E0TelemetryError(f"support resource {field_name} has invalid status")
    raw_value = _integer(value.get("value"), f"support resource {field_name}")
    if allow_native_rss_unit:
        if value.get("unit") not in {"bytes", "kibibytes"}:
            raise E0TelemetryError("support host RSS has invalid native unit")
        normalized = _integer(
            value.get("normalized_bytes"), "support host RSS normalized_bytes"
        )
        expected = raw_value if value.get("unit") == "bytes" else raw_value * 1024
        if normalized != expected:
            raise E0TelemetryError("support host RSS byte normalization mismatch")
    elif value.get("unit") != "bytes":
        raise E0TelemetryError(f"support resource {field_name} must use bytes")
    return dict(value)


def _harvest_support_events(
    *,
    config: NotebookTelemetryConfig,
    output: Path,
    shapes: Mapping[str, tuple[int, int, int, int]],
    production_data_root: str | os.PathLike[str] | None,
    production_test_names: Sequence[str] | None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    data_root, exact_names = _production_identity(
        config, shapes, production_data_root, production_test_names
    )
    expected_names = set(exact_names)
    support_root = output / config.support_telemetry_dirname
    records, event_files = _read_support_events(support_root)
    launch_bindings, launch_evidence = _read_launch_bindings(
        support_root, data_root=data_root, exact_names=exact_names
    )
    starts = [record for record in records if record["record_type"] == "invocation_start"]
    selected: list[tuple[dict[str, object], dict[str, object], tuple[str, ...]]] = []
    for pid, binding in sorted(
        launch_bindings.items(), key=lambda item: int(item[1]["slice_index"])
    ):
        matched_starts = [record for record in starts if record.get("pid") == pid]
        if len(matched_starts) != 1:
            raise E0TelemetryError(f"launched PID has invalid invocation-start count: {pid}")
        record = matched_starts[0]
        identity = record.get("identity")
        if not isinstance(identity, Mapping):
            raise E0TelemetryError("support invocation_start lacks identity")
        if _resolved_absolute_path(identity.get("data_root"), "support data_root") != data_root:
            raise E0TelemetryError("launched support invocation used the wrong data root")
        names = _validate_test_names(identity.get("test_names"))
        if names != binding["expected_test_names"]:
            raise E0TelemetryError(
                "production support invocation has the wrong ordered shard cohort: "
                f"expected={binding['expected_test_names']}, observed={names}"
            )
        invocation_id = identity.get("invocation_id")
        if not isinstance(invocation_id, str) or not invocation_id:
            raise E0TelemetryError("support invocation has invalid invocation_id")
        if record.get("invocation_id") != invocation_id:
            raise E0TelemetryError("support invocation top-level identity mismatch")
        if identity.get("fold") != 0:
            raise E0TelemetryError("production support invocation must use fold 0")
        method = identity.get("method")
        if not isinstance(method, str) or not method or PurePosixPath(method).name != method:
            raise E0TelemetryError("support invocation method is invalid")
        if binding["method"] is not None and method != binding["method"]:
            raise E0TelemetryError("support invocation method differs from launch argv")
        output_dir = Path(
            _resolved_absolute_path(identity.get("output_dir"), "support output_dir")
        )
        if output_dir.name != "split_0" or output_dir.parent.name != method:
            raise E0TelemetryError("support invocation output path does not bind method/fold")
        if record.get("gpu_shard") not in binding["expected_gpu_shard_values"]:
            raise E0TelemetryError("support invocation GPU shard differs from launch slice")
        selected.append((record, dict(identity), names))
    if not selected:
        raise E0TelemetryError("no support invocation matches the exact production data root")

    covered: set[str] = set()
    selected_invocation_ids: set[str] = set()
    selected_process_ids: set[str] = set()
    selected_shards: set[str] = set()
    for record, identity, names in selected:
        overlap = covered & set(names)
        if overlap:
            raise E0TelemetryError(f"production support invocations overlap: {sorted(overlap)}")
        covered.update(names)
        invocation_id = str(identity["invocation_id"])
        process_id = record.get("process_id")
        shard = record.get("gpu_shard")
        if invocation_id in selected_invocation_ids:
            raise E0TelemetryError(f"duplicate support invocation ID: {invocation_id}")
        if not isinstance(process_id, str) or process_id in selected_process_ids:
            raise E0TelemetryError("duplicate or invalid selected support process")
        if not isinstance(shard, str) or not shard or shard in selected_shards:
            raise E0TelemetryError("duplicate or invalid selected support GPU shard")
        selected_invocation_ids.add(invocation_id)
        selected_process_ids.add(process_id)
        selected_shards.add(shard)
    if covered != expected_names:
        raise E0TelemetryError(
            "support invocation cohort coverage mismatch: "
            f"missing={sorted(expected_names - covered)}, extra={sorted(covered - expected_names)}"
        )

    coordinate_records: list[dict[str, object]] = []
    retention_records: list[dict[str, object]] = []
    dataset_evidence: dict[str, object] = {}
    invocation_evidence: list[dict[str, object]] = []
    for start_record, identity, names in selected:
        invocation_id = str(identity["invocation_id"])
        process_id = str(start_record["process_id"])
        finish = [
            record
            for record in records
            if record["record_type"] == "invocation_finish"
            and record.get("invocation_id") == invocation_id
        ]
        if len(finish) != 1 or finish[0].get("outcome") != "returned":
            raise E0TelemetryError(f"support invocation did not return exactly once: {invocation_id}")
        _require_same_support_process(
            finish[0], start_record, require_invocation_id=invocation_id
        )
        if finish[0].get("identity") != identity:
            raise E0TelemetryError(f"support invocation finish identity mismatch: {invocation_id}")
        process = [
            record
            for record in records
            if record["record_type"] == "process_summary"
            and record.get("process_id") == process_id
        ]
        if len(process) != 1:
            raise E0TelemetryError(f"support process summary coverage mismatch: {process_id}")
        _require_same_support_process(process[0], start_record, require_invocation_id=None)
        invocation_ids = process[0].get("invocation_ids")
        if invocation_ids != [invocation_id]:
            raise E0TelemetryError(
                f"support process summary is not exclusive to invocation: {invocation_id}"
            )
        process_resources = {
            "cuda_peak_allocated": _validate_support_resource(
                process[0].get("cuda_peak_allocated"), "cuda_peak_allocated"
            ),
            "cuda_peak_reserved": _validate_support_resource(
                process[0].get("cuda_peak_reserved"), "cuda_peak_reserved"
            ),
            "host_ru_maxrss": _validate_support_resource(
                process[0].get("host_ru_maxrss"),
                "host_ru_maxrss",
                allow_native_rss_unit=True,
            ),
        }
        summaries = [
            record
            for record in records
            if record["record_type"] == "dataset_summary"
            and record.get("invocation_id") == invocation_id
        ]
        coordinates = [
            record
            for record in records
            if record["record_type"] == "coordinate_artifact"
            and record.get("invocation_id") == invocation_id
        ]
        retention_for_invocation = [
            record
            for record in records
            if record["record_type"] == "retention"
            and record.get("invocation_id") == invocation_id
        ]
        if {record.get("dataset") for record in summaries} != set(names):
            raise E0TelemetryError(f"support dataset-summary coverage mismatch: {invocation_id}")
        if {record.get("dataset") for record in coordinates} != set(names):
            raise E0TelemetryError(f"support coordinate coverage mismatch: {invocation_id}")
        if len(summaries) != len(names) or len(coordinates) != len(names):
            raise E0TelemetryError(f"duplicate support dataset evidence: {invocation_id}")
        by_coordinate = {str(record["dataset"]): record for record in coordinates}
        for bound_record in (*summaries, *coordinates, *retention_for_invocation):
            _require_same_support_process(
                bound_record, start_record, require_invocation_id=invocation_id
            )
        if any(record.get("dataset") not in set(names) for record in retention_for_invocation):
            raise E0TelemetryError(f"support retention cohort mismatch: {invocation_id}")
        for summary in summaries:
            dataset = str(summary["dataset"])
            if dataset in dataset_evidence or summary.get("outcome") != "returned":
                raise E0TelemetryError(f"invalid support dataset outcome for {dataset}")
            expected_dataset_path = str((Path(data_root) / dataset).resolve(strict=False))
            if _resolved_absolute_path(
                summary.get("dataset_path"), "support dataset_path"
            ) != expected_dataset_path:
                raise E0TelemetryError(f"support dataset path mismatch for {dataset}")
            coordinate = by_coordinate[dataset]
            if coordinate.get("dataset_path") != summary.get("dataset_path"):
                raise E0TelemetryError(f"support coordinate path identity mismatch for {dataset}")
            summary_coordinate = summary.get("coordinate_artifact")
            if not isinstance(summary_coordinate, Mapping) or summary_coordinate.get(
                "status"
            ) != "available":
                raise E0TelemetryError(f"support dataset coordinate is unavailable for {dataset}")
            for key in (
                "stage",
                "columns",
                "dtype",
                "rows",
                "coordinate_sha256",
                "frame_counts",
                "artifact",
            ):
                if summary_coordinate.get(key) != coordinate.get(key):
                    raise E0TelemetryError(
                        f"support coordinate event/summary mismatch for {dataset}: {key}"
                    )
            dataset_retention = [
                record
                for record in retention_for_invocation
                if record.get("dataset") == dataset
            ]
            for retention_record in dataset_retention:
                if retention_record.get("dataset_path") != summary.get("dataset_path"):
                    raise E0TelemetryError(
                        f"support retention path identity mismatch for {dataset}"
                    )
                retention_records.append(
                    {
                        key: value
                        for key, value in retention_record.items()
                        if not key.startswith("_")
                    }
                )
            counts = summary.get("counts")
            if not isinstance(counts, Mapping):
                raise E0TelemetryError(f"support counts are unavailable for {dataset}")
            normalized_counts = {
                name: _support_available_count(counts.get(name), f"{dataset}.{name}")
                for name in (
                    "pair_universe",
                    "threshold_passing_edge_candidates",
                    "detected_nodes",
                    "pre_ilp_edges",
                    "output_edges",
                )
            }
            if normalized_counts["threshold_passing_edge_candidates"] > normalized_counts[
                "pair_universe"
            ]:
                raise E0TelemetryError(f"support pair counts are inconsistent for {dataset}")
            if normalized_counts["detected_nodes"] != coordinate.get("rows"):
                raise E0TelemetryError(f"support coordinate/node count mismatch for {dataset}")
            durations = summary.get("durations")
            if not isinstance(durations, Mapping) or set(durations) != set(_SUPPORT_STAGES):
                raise E0TelemetryError(f"support duration coverage mismatch for {dataset}")
            normalized_durations = {
                stage: _validate_support_duration(durations[stage], f"{dataset}.{stage}")
                for stage in _SUPPORT_STAGES
            }
            ilp = summary.get("ilp")
            if not isinstance(ilp, Mapping) or ilp.get("status") not in {
                "returned",
                "unavailable",
            }:
                raise E0TelemetryError(f"support ILP outcome is invalid for {dataset}")
            if ilp.get("status") == "unavailable" and not isinstance(ilp.get("reason"), str):
                raise E0TelemetryError(f"support ILP unavailable reason is missing for {dataset}")
            coordinate_records.append(
                {key: value for key, value in coordinate.items() if not key.startswith("_")}
            )
            dataset_evidence[dataset] = {
                "invocation_id": invocation_id,
                "process_id": process_id,
                "gpu_shard": start_record.get("gpu_shard"),
                "diagnostic_arm": start_record.get("diagnostic_arm"),
                "dataset_path": summary.get("dataset_path"),
                "durations": normalized_durations,
                "counts": normalized_counts,
                "ilp": dict(ilp),
            }
        invocation_evidence.append(
            {
                "invocation_id": invocation_id,
                "process_id": process_id,
                "gpu_shard": start_record.get("gpu_shard"),
                "diagnostic_arm": start_record.get("diagnostic_arm"),
                "identity": identity,
                "outcome": "returned",
                "resource_scope": "production_shard_process",
                "resources": process_resources,
                "process_summary": {
                    key: value
                    for key, value in process[0].items()
                    if not key.startswith("_")
                },
            }
        )
    ignored_invocation_count = len(starts) - len(selected)
    return (
        {
            "status": "PASS",
            "production_data_root": data_root,
            "production_test_names": list(exact_names),
            "event_files": event_files,
            "launch_evidence": launch_evidence,
            "selected_invocations": invocation_evidence,
            "datasets": dict(sorted(dataset_evidence.items())),
            "ignored_nonproduction_invocation_count": ignored_invocation_count,
        },
        coordinate_records,
        retention_records,
    )


def harvest_e0_telemetry(
    config: NotebookTelemetryConfig,
    *,
    production_data_root: str | os.PathLike[str] | None = None,
    production_test_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Strictly validate E0 frame, coordinate, graph, and submission evidence."""

    config = config.validated()
    output = Path(config.output_dir)
    manifest_path = output / config.run_manifest_filename
    manifest = _strict_json_object(manifest_path)
    shapes = _resolve_shapes(config, manifest)
    support, support_coordinates, support_retention = _harvest_support_events(
        config=config,
        output=output,
        shapes=shapes,
        production_data_root=production_data_root,
        production_test_names=production_test_names,
    )
    retention = _validate_retention(support_retention, shapes)
    retention["source"] = "selected production support invocation events"
    coordinates = _validate_coordinates(
        support_coordinates,
        shapes,
        output,
        require_artifacts=config.require_coordinate_artifacts,
    )
    run_stats = _read_run_stats(output / config.run_stats_filename, set(shapes))
    reconciliations: dict[str, object] = {}
    for dataset in sorted(shapes):
        coordinate_rows = coordinates["datasets"][dataset]["rows"]  # type: ignore[index]
        stats = run_stats[dataset]
        reconciliations[dataset] = {
            "detector_coordinate_rows": coordinate_rows,
            "post_ilp_raw_nodes": stats["raw_nodes"],
            "pre_ilp_to_post_ilp_node_delta": stats["raw_nodes"] - coordinate_rows,
            "post_ilp_raw_edges": stats["raw_edges"],
            "final_nodes": stats["nodes"],
            "final_edges": stats["edges"],
            "post_ilp_to_final_node_delta": stats["nodes"] - stats["raw_nodes"],
            "post_ilp_to_final_edge_delta": stats["edges"] - stats["raw_edges"],
            "counter_values": {
                name: stats[name] for name in _COUNTER_SEMANTICS
            },
        }
    submission_path = output / config.submission_filename
    if submission_path.is_file():
        submission_counts = _submission_counts(submission_path, set(shapes))
        for dataset, counts in submission_counts.items():
            stats = run_stats[dataset]
            if counts["nodes"] != stats["nodes"] or counts["edges"] != stats["edges"]:
                raise E0TelemetryError(f"submission/run_stats count mismatch for {dataset}")
        submission: dict[str, object] = {
            "status": "VERIFIED",
            "path": str(submission_path),
            "sha256": _sha256_file(submission_path),
            "counts": submission_counts,
        }
    elif config.require_submission_crosscheck:
        raise E0TelemetryError("submission crosscheck is required but submission is missing")
    else:
        submission = {"status": "UNAVAILABLE", "reason": "submission file is missing"}
    return {
        "schema_version": 1,
        "status": "PASS",
        "shapes_tzyx": {dataset: list(shape) for dataset, shape in shapes.items()},
        "support": support,
        "retention": retention,
        "coordinates": coordinates,
        "run_stats": {
            "status": "PASS",
            "path": str(output / config.run_stats_filename),
            "sha256": _sha256_file(output / config.run_stats_filename),
            "counter_semantics": _COUNTER_SEMANTICS,
            "datasets": run_stats,
        },
        "reconciliations": reconciliations,
        "submission": submission,
    }


def standalone_runtime_source() -> str:
    """Return this dependency-free module for execution inside a notebook cell."""

    try:
        return Path(__file__).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise E0TelemetryError("cannot read telemetry runtime source") from exc


def build_notebook_telemetry_sources(
    config: NotebookTelemetryConfig,
) -> NotebookTelemetrySources:
    """Build exact prepended/final cell sources without changing public cells."""

    config = config.validated()
    final_source = (
        "# E0 additive telemetry finalizer.\n"
        f"{_TRACKER_GLOBAL}_RESULT = {_TRACKER_GLOBAL}.finalize(\n"
        "    production_data_root=globals().get('TEST_DIR'),\n"
        "    production_test_names=globals().get('test_stems'),\n"
        ")\n"
        f"print('E0_NOTEBOOK_TELEMETRY', {_TRACKER_GLOBAL}_RESULT, flush=True)\n"
    )
    final_hash = _sha256_bytes(final_source.encode("utf-8"))
    allowed = tuple(config.allowed_auxiliary_cell_sha256)
    if final_hash not in allowed:
        allowed = (*allowed, final_hash)
    embedded_config = NotebookTelemetryConfig(
        **{
            **asdict(config),
            "allowed_auxiliary_cell_sha256": allowed,
        }
    ).validated()
    runtime_source = standalone_runtime_source()
    runtime_hash = _sha256_bytes(runtime_source.encode("utf-8"))
    config_json = json.dumps(
        embedded_config.to_json_value(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    prepended_source = f'''# E0 additive telemetry bootstrap; no torch import or CUDA context creation.
import json as _e0_json
import hashlib as _e0_hashlib
import sys as _e0_sys
import types as _e0_types

_E0_TELEMETRY_RUNTIME_SOURCE = {runtime_source!r}
_E0_TELEMETRY_RUNTIME_SHA256 = {runtime_hash!r}
if _e0_hashlib.sha256(_E0_TELEMETRY_RUNTIME_SOURCE.encode("utf-8")).hexdigest() != _E0_TELEMETRY_RUNTIME_SHA256:
    raise RuntimeError("embedded E0 telemetry runtime source hash mismatch")
if {_RUNTIME_MODULE_NAME!r} in _e0_sys.modules:
    raise RuntimeError("embedded E0 telemetry runtime module is already registered")
_E0_TELEMETRY_MODULE = _e0_types.ModuleType({_RUNTIME_MODULE_NAME!r})
_E0_TELEMETRY_MODULE.__file__ = "<embedded-e0-notebook-telemetry>"
_e0_sys.modules[{_RUNTIME_MODULE_NAME!r}] = _E0_TELEMETRY_MODULE
exec(compile(
    _E0_TELEMETRY_RUNTIME_SOURCE,
    _E0_TELEMETRY_MODULE.__file__,
    "exec",
), _E0_TELEMETRY_MODULE.__dict__)
_E0_TELEMETRY_CONFIG = _E0_TELEMETRY_MODULE.NotebookTelemetryConfig.from_json_value(
    _e0_json.loads({config_json!r})
)
{_TRACKER_GLOBAL} = _E0_TELEMETRY_MODULE.NotebookTelemetryTracker(_E0_TELEMETRY_CONFIG)
{_TRACKER_GLOBAL}.start(get_ipython())
print("E0_NOTEBOOK_TELEMETRY_STARTED", {{
    "runtime_source_sha256": _E0_TELEMETRY_RUNTIME_SHA256,
    "expected_public_cells": len(_E0_TELEMETRY_CONFIG.expected_public_cell_sha256),
}}, flush=True)
'''
    return NotebookTelemetrySources(
        prepended_source=prepended_source,
        final_source=final_source,
        runtime_source_sha256=runtime_hash,
        prepended_source_sha256=_sha256_bytes(prepended_source.encode("utf-8")),
        final_source_sha256=final_hash,
    )
