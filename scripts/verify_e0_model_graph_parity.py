#!/usr/bin/env python3
"""Bounded real-model parity check for additive E0 support telemetry.

This executes the pinned public support source twice on the same eight full-spatial
frames: once unchanged and once with the reviewed telemetry patch.  It compares
detector coordinates, ordered candidate edges, pre/post-ILP graphs, and the
logical arrays and attributes written to GEFF.  This is operational parity
evidence only.  It does not execute the full notebook, output post-processing,
quality evaluation, submission construction, or public-score reproduction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import sys
import tarfile
import time
import types
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "44b6_0113de3b"
FRAMES = tuple(range(8))
EXPECTED_RAW_SHAPE = (100, 64, 256, 256)
MAX_APPLICATION_SECONDS = 300.0
MAX_OUTPUT_BYTES = 256 * 1024**2
PUBLIC_SUPPORT_SHA256 = "49613ad0b50ac90c3e07e3f8a803f2adf0926203be3f4a97d577c60b8f756178"
EXPECTED_CHECKPOINT_SHA256 = (
    "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
    "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f",
)
EXPECTED_CONFIG_SHA256 = "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a"
EXPECTED_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
    "pool_kernel_um": 5.0,
}
EXPECTED_SUPPORT_FILES = {
    "repo/scripts/predict_unet_transformer.py": "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9",
    "repo/scripts/train_unet_transformer.py": "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea",
    "repo/src/biohub_tracking/models/temporal_unet.py": "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac",
    "repo/src/biohub_tracking/models/simple_node_transformer.py": "b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e",
    "repo/src/biohub_tracking/__init__.py": "26a18d8da84e40da73281a48ebc3017d847a2e57431ab63e8629d2109e6e8571",
    "repo/src/biohub_tracking/models/__init__.py": "ab7587ef79856bae50d24b62e5805092d0459ee1c586522b763f9ef70c093e1d",
}
PUBLIC_CONFIG = {
    "det_threshold": 0.965,
    "det_tta": True,
    "pool_kernel_um": 3.0,
    "edge_activation": "softmax",
    "edge_candidate_threshold": 0.48,
    "use_ilp": True,
    "ilp_edge_weight": -1.0,
    "ilp_appearance_weight": 0.0,
    "ilp_disappearance_weight": 2.0,
    "ilp_division_weight": 1.2,
    "secondary_edge_weight": 0.15,
    "secondary_detection_weight": 0.80,
    "secondary_link_mode": "low_margin_consensus",
    "secondary_mix_temperature": 1.0,
    "secondary_low_margin_max": 0.35,
    "bidirectional_edge_weight": 0.15,
    "edge_feature_tta": True,
    "minimum_candidate_retention": 0.90,
}
CAMPAIGN_ENVIRONMENT = (
    "BIOHUB_RUN_ID",
    "BIOHUB_RUN_SPEC_SHA256",
    "BIOHUB_INTENT_ID",
    "BIOHUB_FENCING_TOKEN",
    "BIOHUB_ATTEMPT_DIR",
    "BIOHUB_PROGRESS_PATH",
)
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA_RE = re.compile(r"[0-9a-f]{64}")


class ContractError(RuntimeError):
    """The run cannot establish its pinned operational contract."""


class DeadlineExceeded(RuntimeError):
    """The single shared application deadline expired."""


class ParityMismatch(RuntimeError):
    """Both arms completed but produced different behavior."""


@dataclass(frozen=True)
class Request:
    source: Path
    support_root: Path
    checkpoints: tuple[Path, Path]
    data_root: Path
    input_manifest: Path
    output: Path
    max_wall_seconds: float


@dataclass(frozen=True)
class CampaignContext:
    run_id: str
    run_spec_sha256: str
    intent_id: str
    fencing_token: int
    attempt_dir: Path
    progress_path: Path
    result_manifest_path: Path

    def identity(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_spec_sha256": self.run_spec_sha256,
            "intent_id": self.intent_id,
            "fencing_token": self.fencing_token,
        }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"missing {label}: {path}") from exc
    if not resolved.is_file():
        raise ContractError(f"{label} is not a file: {resolved}")
    return resolved


def verify_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    resolved = require_file(path, label)
    actual = sha256_file(resolved)
    if actual != expected:
        raise ContractError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": actual}


def check_deadline(started: float, limit: float, stage: str) -> None:
    elapsed = time.monotonic() - started
    if elapsed >= limit:
        raise DeadlineExceeded(
            f"shared {limit:.3f}s application deadline reached during {stage}; elapsed={elapsed:.3f}s"
        )


@contextmanager
def alarm_deadline(seconds: float) -> Iterator[None]:
    if os.name == "nt" or not hasattr(signal, "setitimer"):
        yield
        return

    def handler(_signum: int, _frame: Any) -> None:
        raise DeadlineExceeded(f"shared {seconds:.3f}s application wall alarm fired")

    old_handler = signal.signal(signal.SIGALRM, handler)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)


def campaign_context(request: Request) -> CampaignContext | None:
    values = {name: os.environ.get(name) for name in CAMPAIGN_ENVIRONMENT}
    present = {name for name, value in values.items() if value is not None}
    if not present:
        return None
    missing = set(CAMPAIGN_ENVIRONMENT) - present
    if missing:
        raise ContractError(f"partial campaign environment; missing {sorted(missing)}")
    run_id = str(values["BIOHUB_RUN_ID"])
    run_spec_sha256 = str(values["BIOHUB_RUN_SPEC_SHA256"])
    intent_id = str(values["BIOHUB_INTENT_ID"])
    if _ID_RE.fullmatch(run_id) is None or _ID_RE.fullmatch(intent_id) is None:
        raise ContractError("campaign run/intent identifiers are invalid")
    if _SHA_RE.fullmatch(run_spec_sha256) is None:
        raise ContractError("BIOHUB_RUN_SPEC_SHA256 is not a lowercase SHA-256")
    try:
        fencing_token = int(str(values["BIOHUB_FENCING_TOKEN"]))
    except ValueError as exc:
        raise ContractError("BIOHUB_FENCING_TOKEN is not an integer") from exc
    if fencing_token <= 0:
        raise ContractError("BIOHUB_FENCING_TOKEN must be positive")
    attempt_dir = Path(str(values["BIOHUB_ATTEMPT_DIR"])).resolve(strict=True)
    progress = Path(str(values["BIOHUB_PROGRESS_PATH"]))
    if not progress.is_absolute():
        raise ContractError("BIOHUB_PROGRESS_PATH must be absolute")
    progress = progress.resolve()
    if request.output.resolve().parent != attempt_dir or not progress.is_relative_to(attempt_dir):
        raise ContractError("parity output and progress must be inside the campaign attempt")
    if attempt_dir.name != run_id:
        raise ContractError("campaign attempt directory must end in BIOHUB_RUN_ID")
    result = attempt_dir / "result.json"
    if result.exists():
        raise ContractError("campaign result already exists")
    return CampaignContext(
        run_id, run_spec_sha256, intent_id, fencing_token, attempt_dir, progress, result
    )


def write_progress(context: CampaignContext, completed: int, error: str | None) -> None:
    atomic_json(
        context.progress_path,
        {
            **context.identity(),
            "completed_units": completed,
            "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": error,
        },
    )


def load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(require_file(path, label).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON in {label}: {path}") from exc


def verify_staged_manifests(request: Request) -> dict[str, Any]:
    manifest_dir = request.input_manifest.resolve().parent
    source_manifest_path = manifest_dir / "source.json"
    dependency_manifest_path = manifest_dir / "dependencies.json"
    effective_config_path = manifest_dir / "config.json"
    split_manifest_path = manifest_dir / "split.json"
    root = Path(__file__).resolve().parents[1]

    source_manifest = load_json(source_manifest_path, "source manifest")
    if not isinstance(source_manifest, dict) or not source_manifest:
        raise ContractError("source manifest must be a nonempty object")
    for relative, expected in source_manifest.items():
        if not isinstance(relative, str) or _SHA_RE.fullmatch(str(expected)) is None:
            raise ContractError("invalid source manifest entry")
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            raise ContractError(f"source manifest path escapes project root: {relative}")
        verify_hash(candidate, str(expected), f"source manifest entry {relative}")

    dependency_manifest = load_json(dependency_manifest_path, "dependency manifest")
    expected_python = dependency_manifest.get("python")
    if expected_python != sys.version:
        raise ContractError("Python version differs from the staged dependency manifest")
    executable_expected = dependency_manifest.get("executable_sha256")
    if _SHA_RE.fullmatch(str(executable_expected)) is None:
        raise ContractError("dependency manifest executable SHA-256 is invalid")
    verify_hash(Path(sys.executable), str(executable_expected), "Python executable")
    packages = dependency_manifest.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ContractError("dependency manifest packages must be a nonempty object")
    observed_packages: dict[str, str] = {}
    for name, expected in packages.items():
        try:
            actual = importlib.metadata.version(str(name))
        except importlib.metadata.PackageNotFoundError as exc:
            raise ContractError(f"required package is unavailable: {name}") from exc
        if actual != expected:
            raise ContractError(f"package {name} differs: expected {expected!r}, got {actual!r}")
        observed_packages[str(name)] = actual

    staged_config = load_json(effective_config_path, "effective config manifest")
    if staged_config.get("dataset") != DATASET or staged_config.get("frames") != list(FRAMES):
        raise ContractError("effective config dataset or frames differ")
    if staged_config.get("full_spatial") is not True:
        raise ContractError("effective config must require full-spatial inference")
    if float(staged_config.get("application_deadline_seconds", -1)) != request.max_wall_seconds:
        raise ContractError("effective config application deadline differs from argv")
    if staged_config.get("full_notebook_release_claim") is not False:
        raise ContractError("effective config improperly claims full-notebook evidence")
    expected_checkpoint_paths = [str(path.resolve()) for path in request.checkpoints]
    if staged_config.get("checkpoints") != expected_checkpoint_paths:
        raise ContractError("effective config checkpoint paths differ from argv")
    if staged_config.get("checkpoint_sha256") != list(EXPECTED_CHECKPOINT_SHA256):
        raise ContractError("effective config checkpoint hashes differ")

    split_manifest = load_json(split_manifest_path, "split manifest")
    if split_manifest.get("quality_evaluation") is not False:
        raise ContractError("split manifest must disable quality evaluation")

    return {
        "source_manifest": verify_hash(
            source_manifest_path, sha256_file(source_manifest_path), "source manifest"
        ),
        "dependency_manifest": verify_hash(
            dependency_manifest_path, sha256_file(dependency_manifest_path), "dependency manifest"
        ),
        "effective_config_manifest": verify_hash(
            effective_config_path, sha256_file(effective_config_path), "effective config manifest"
        ),
        "split_manifest": verify_hash(
            split_manifest_path, sha256_file(split_manifest_path), "split manifest"
        ),
        "packages": observed_packages,
        "split_classification": split_manifest,
    }


def verify_static_inputs(request: Request) -> dict[str, Any]:
    source = verify_hash(request.source, PUBLIC_SUPPORT_SHA256, "public-patched support source")
    support_root = request.support_root.resolve(strict=True)
    support = {
        relative: verify_hash(support_root / relative, digest, relative)
        for relative, digest in EXPECTED_SUPPORT_FILES.items()
    }
    checkpoints = []
    for index, (path, digest) in enumerate(zip(request.checkpoints, EXPECTED_CHECKPOINT_SHA256)):
        config_path = path.resolve().parent / "config.json"
        config = load_json(config_path, f"checkpoint {index} config")
        if config != EXPECTED_CONFIG:
            raise ContractError(f"checkpoint {index} config contents differ")
        checkpoints.append(
            {
                "checkpoint": verify_hash(path, digest, f"checkpoint {index}"),
                "config": verify_hash(
                    config_path, EXPECTED_CONFIG_SHA256, f"checkpoint {index} config"
                ),
            }
        )
    return {"source": source, "support": support, "checkpoints": checkpoints}


def verify_input_snapshot(request: Request, np: Any, zarr: Any, label: str) -> dict[str, Any]:
    manifest_path = require_file(request.input_manifest, "input manifest")
    manifest_hash_before = sha256_file(manifest_path)
    manifest = load_json(manifest_path, "input manifest")
    expected_path = (request.data_root / f"{DATASET}.zarr").resolve(strict=True)
    if Path(str(manifest.get("path", ""))).resolve(strict=True) != expected_path:
        raise ContractError("input manifest path differs from fixed data path")
    if manifest.get("shape") != list(EXPECTED_RAW_SHAPE):
        raise ContractError("input manifest shape differs from fixed TZYX shape")
    if manifest.get("frames") != list(FRAMES):
        raise ContractError("input manifest frames must be exactly 0..7")
    if manifest.get("frame_memory_order") != "C":
        raise ContractError("input manifest frame memory order must be C")

    group = zarr.open_group(str(expected_path), mode="r")
    array = group["0"]
    if tuple(int(value) for value in array.shape) != EXPECTED_RAW_SHAPE:
        raise ContractError("live raw array shape differs from fixed TZYX shape")
    if str(array.dtype) != str(manifest.get("raw_dtype")):
        raise ContractError("live raw dtype differs from input manifest")
    expected_hashes = manifest.get("frame_sha256")
    if not isinstance(expected_hashes, list) or len(expected_hashes) != len(FRAMES):
        raise ContractError("input manifest must contain eight frame hashes")
    frame_nbytes = []
    observed_hashes = []
    for frame, expected in zip(FRAMES, expected_hashes):
        raw = np.ascontiguousarray(array[frame])
        if tuple(int(value) for value in raw.shape) != EXPECTED_RAW_SHAPE[1:]:
            raise ContractError(f"raw frame {frame} shape differs")
        payload = raw.tobytes(order="C")
        actual = sha256_bytes(payload)
        if actual != expected:
            raise ContractError(f"raw frame {frame} changed at {label}")
        observed_hashes.append(actual)
        frame_nbytes.append(len(payload))
    expected_nbytes = manifest.get("frame_nbytes")
    if isinstance(expected_nbytes, int):
        expected_nbytes = [expected_nbytes] * len(FRAMES)
    if expected_nbytes != frame_nbytes:
        raise ContractError("input manifest frame byte counts differ")

    metadata = manifest.get("metadata_files")
    if not isinstance(metadata, dict) or not metadata:
        raise ContractError("input manifest metadata_files must be nonempty")
    observed_metadata = {}
    for relative, expected in metadata.items():
        candidate = (expected_path / str(relative)).resolve()
        if not candidate.is_relative_to(expected_path):
            raise ContractError(f"metadata path escapes Zarr root: {relative}")
        observed_metadata[str(relative)] = verify_hash(
            candidate, str(expected), f"Zarr metadata {relative}"
        )["sha256"]
    manifest_hash_after = sha256_file(manifest_path)
    if manifest_hash_after != manifest_hash_before:
        raise ContractError(f"input manifest changed while verifying {label}")
    return {
        "label": label,
        "manifest_sha256": manifest_hash_after,
        "path": str(expected_path),
        "shape": list(EXPECTED_RAW_SHAPE),
        "dtype": str(array.dtype),
        "frame_sha256": observed_hashes,
        "frame_nbytes": frame_nbytes,
        "metadata_sha256": observed_metadata,
    }


def install_support_paths(support_root: Path) -> None:
    paths = [PROJECT_ROOT / "src", support_root / "repo/scripts", support_root / "repo/src"]
    for path in reversed(paths):
        value = str(path.resolve(strict=True))
        if value not in sys.path:
            sys.path.insert(0, value)


def load_source_module(name: str, source: str, filename: Path) -> types.ModuleType:
    if name in sys.modules:
        raise ContractError(f"module name collision: {name}")
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(  # noqa: S102 - executes a SHA-pinned, locally staged source under comparison
            compile(source, str(filename), "exec"), module.__dict__
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def verify_support_import_origins(support_root: Path) -> dict[str, str]:
    repo = (support_root / "repo").resolve(strict=True)
    observed = {}
    for name in (
        "biohub_tracking",
        "biohub_tracking.io",
        "biohub_tracking.metrics",
        "biohub_tracking.models",
        "biohub_tracking.models.temporal_unet",
        "train_unet_transformer",
        "dataspec",
        "evaluate",
    ):
        imported = sys.modules.get(name)
        origin_text = getattr(imported, "__file__", None)
        if imported is None or not origin_text:
            raise ContractError(f"expected pinned support import is absent: {name}")
        origin = Path(origin_text).resolve(strict=True)
        if not origin.is_relative_to(repo):
            raise ContractError(f"support import {name} resolved outside staged repo: {origin}")
        observed[name] = str(origin)
    return observed


def redirect_diagnostic_path(module: types.ModuleType, destination: Path) -> None:
    real_path = Path
    destination.mkdir(parents=True, exist_ok=False)

    def arm_path(value: Any = ".") -> Path:
        path = real_path(value)
        return destination if str(path) == "/kaggle/working" else path

    module.Path = arm_path


def public_environment(secondary_checkpoint: Path) -> dict[str, str]:
    return {
        "BIOHUB_SECONDARY_WEIGHTS": str(secondary_checkpoint.resolve()),
        "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.15",
        "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.80",
        "BIOHUB_SECONDARY_LINK_MODE": "low_margin_consensus",
        "BIOHUB_SECONDARY_MIX_TEMPERATURE": "1",
        "BIOHUB_SECONDARY_LOW_MARGIN_MAX": "0.35",
        "BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.48",
        "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",
        "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.15",
        "BIOHUB_BIDIRECTIONAL_FUSION_MODE": "harmonic_probability",
        "BIOHUB_EDGE_FEATURE_TTA": "1",
        "BIOHUB_DIAGNOSTIC_ARM": "parity",
        "BIOHUB_GPU_SHARD": "single",
    }


@contextmanager
def temporary_environment(updates: dict[str, str], removals: Sequence[str] = ()) -> Iterator[None]:
    keys = set(updates) | set(removals)
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key in removals:
            os.environ.pop(key, None)
        os.environ.update(updates)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def scalar_token(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("graph contains a nonfinite float")
        return {"type": "float64", "hex": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if hasattr(value, "item"):
        return scalar_token(value.item())
    raise ContractError(f"unsupported graph scalar type: {type(value).__name__}")


def canonical_dataframe(frame: Any, identity_columns: Sequence[str]) -> dict[str, Any]:
    available = {str(column) for column in frame.columns}
    columns = [column for column in identity_columns if column in available]
    columns.extend(sorted(available - set(columns)))
    schema = {column: str(frame.schema[column]) for column in columns}
    rows = []
    for row in frame.iter_rows(named=True):
        rows.append({column: scalar_token(row[column]) for column in columns})

    def row_key(row: dict[str, Any]) -> str:
        identity = [row[column] for column in identity_columns if column in row]
        return json.dumps(identity or row, sort_keys=True, separators=(",", ":"))

    rows.sort(key=row_key)
    return {"columns": columns, "schema": schema, "rows": rows}


def canonical_graph(graph: Any) -> dict[str, Any]:
    nodes = canonical_dataframe(graph.node_attrs(), ("node_id",))
    edges = canonical_dataframe(graph.edge_attrs(), ("source_id", "target_id", "edge_id"))
    return {
        "num_nodes": int(graph.num_nodes()),
        "num_edges": int(graph.num_edges()),
        "nodes": nodes,
        "edges": edges,
    }


def canonical_graph_semantics(graph_canonical: dict[str, Any]) -> dict[str, Any]:
    """Project a graph onto IDs, topology, and user attributes for GEFF readback.

    ``edge_id`` is a tracksdata storage identity, rather than tracking topology.
    Schemas are verified independently in the full canonical graph and the GEFF
    array inventory; this projection compares values without treating a reader's
    integer-width presentation as a biological graph change.
    """

    def project(table: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
        columns = [column for column in table["columns"] if column not in excluded]
        rows = [{column: row[column] for column in columns} for row in table["rows"]]
        rows.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
        return {"columns": columns, "rows": rows}

    return {
        "num_nodes": graph_canonical["num_nodes"],
        "num_edges": graph_canonical["num_edges"],
        "nodes": project(graph_canonical["nodes"], set()),
        "edges": project(graph_canonical["edges"], {"edge_id"}),
    }


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_canonical_json(path: Path, value: Any) -> dict[str, Any]:
    atomic_json(path, value)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def array_identity(array: Any, np: Any) -> dict[str, Any]:
    materialized = np.asarray(array)
    contiguous = np.ascontiguousarray(materialized)
    if contiguous.dtype.hasobject:
        payload = canonical_bytes(contiguous.tolist())
        encoding = "canonical_json"
    else:
        payload = contiguous.tobytes(order="C")
        encoding = "c_order_bytes"
    return {
        "shape": [int(value) for value in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "encoding": encoding,
        "sha256": sha256_bytes(payload),
        "nbytes": len(payload),
    }


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("GEFF attribute contains a nonfinite float")
        return {"__float_hex__": value.hex()}
    if isinstance(value, dict):
        return {
            str(key): json_value(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return json_value(value.item())
    raise ContractError(f"unsupported GEFF attribute type: {type(value).__name__}")


def canonical_geff(path: Path, np: Any, zarr: Any) -> dict[str, Any]:
    root = zarr.open_group(str(path), mode="r")
    groups: dict[str, Any] = {}
    arrays: dict[str, Any] = {}

    def walk(group: Any, prefix: str) -> None:
        groups[prefix or "/"] = json_value(dict(group.attrs))
        for name, array in sorted(group.arrays(), key=lambda item: item[0]):
            key = f"{prefix}/{name}" if prefix else str(name)
            arrays[key] = {
                **array_identity(array[...], np),
                "attrs": json_value(dict(array.attrs)),
            }
        for name, child in sorted(group.groups(), key=lambda item: item[0]):
            key = f"{prefix}/{name}" if prefix else str(name)
            walk(child, key)

    walk(root, "")
    if not arrays:
        raise ContractError(f"GEFF contains no arrays: {path}")
    return {"groups": groups, "arrays": arrays}


def graph_from_geff(td: Any, path: Path) -> Any:
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def compare_exact(left: Any, right: Any, label: str) -> None:
    if canonical_bytes(left) != canonical_bytes(right):
        raise ParityMismatch(f"{label} differs between control and instrumented arms")


def candidate_arrays(
    coords: Any, edges: list[tuple[int, int, float, float]], np: Any
) -> dict[str, Any]:
    if not isinstance(coords, np.ndarray) or coords.dtype != np.dtype(np.int16):
        raise ContractError(
            f"predict_video coordinates must be native int16, found {getattr(coords, 'dtype', None)}"
        )
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ContractError(
            f"predict_video coordinates must have native shape (N, 4), found {coords.shape}"
        )
    for index, edge in enumerate(edges):
        if len(edge) != 4:
            raise ContractError(f"candidate edge {index} does not have four fields")
        source, target, probability, distance = edge
        if not isinstance(source, int) or not isinstance(target, int):
            raise ContractError(f"candidate edge {index} endpoints are not native integers")
        if not math.isfinite(float(probability)) or not math.isfinite(float(distance)):
            raise ContractError(f"candidate edge {index} contains a nonfinite value")
        if not 0.0 <= float(probability) <= 1.0 or float(distance) < 0.0:
            raise ContractError(f"candidate edge {index} has an invalid probability or distance")
    coordinates = np.ascontiguousarray(coords.astype("<i2", copy=False))
    source = np.asarray([edge[0] for edge in edges], dtype="<i8")
    target = np.asarray([edge[1] for edge in edges], dtype="<i8")
    probability = np.asarray([edge[2] for edge in edges], dtype="<f8")
    distance = np.asarray([edge[3] for edge in edges], dtype="<f8")
    return {
        "coordinates": coordinates,
        "edge_source": source,
        "edge_target": target,
        "edge_probability": probability,
        "edge_distance": distance,
    }


def candidate_identity(arrays: dict[str, Any], np: Any) -> dict[str, Any]:
    return {name: array_identity(value, np) for name, value in arrays.items()}


def save_stage_npz(path: Path, arrays: dict[str, Any], np: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as loaded:
        if loaded.files != list(arrays):
            raise ContractError("NPZ stage array order changed")
        for name, expected in arrays.items():
            if not np.array_equal(loaded[name], expected):
                raise ContractError(f"NPZ stage array changed on reload: {name}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def max_component_span(graph_canonical: dict[str, Any]) -> dict[str, int]:
    node_rows = graph_canonical["nodes"]["rows"]
    edge_rows = graph_canonical["edges"]["rows"]

    def token_int(token: dict[str, Any]) -> int:
        if token.get("type") != "int":
            raise ContractError("expected integral graph identity/time")
        return int(token["value"])

    times = {token_int(row["node_id"]): token_int(row["t"]) for row in node_rows}
    neighbors = {node_id: set() for node_id in times}
    for row in edge_rows:
        source = token_int(row["source_id"])
        target = token_int(row["target_id"])
        if source not in neighbors or target not in neighbors:
            raise ContractError("graph edge references a missing node")
        neighbors[source].add(target)
        neighbors[target].add(source)
    seen: set[int] = set()
    max_inclusive = 0
    max_unique = 0
    for start in neighbors:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component_times = []
        while stack:
            node = stack.pop()
            component_times.append(times[node])
            for adjacent in neighbors[node]:
                if adjacent not in seen:
                    seen.add(adjacent)
                    stack.append(adjacent)
        max_inclusive = max(max_inclusive, max(component_times) - min(component_times) + 1)
        max_unique = max(max_unique, len(set(component_times)))
    return {"maximum_inclusive_frame_span": max_inclusive, "maximum_unique_frames": max_unique}


def finalize_telemetry(runtime_module: types.ModuleType | None) -> None:
    if runtime_module is not None:
        runtime_module.telemetry.finalize_process()


def validate_telemetry(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    event_paths = list(root.glob("invocation-*/events.jsonl"))
    if len(event_paths) != 1:
        raise ContractError(f"expected one telemetry event file, found {len(event_paths)}")
    records = [json.loads(line) for line in event_paths[0].read_text().splitlines() if line]
    types_seen = [record.get("record_type") for record in records]
    for required in (
        "invocation_start",
        "coordinate_artifact",
        "dataset_summary",
        "invocation_finish",
        "process_summary",
    ):
        if types_seen.count(required) != 1:
            raise ContractError(f"telemetry must contain one {required} record")
    summary = next(record for record in records if record.get("record_type") == "dataset_summary")
    if summary.get("outcome") != "returned" or summary.get("ilp", {}).get("status") != "returned":
        raise ContractError("telemetry dataset or ILP did not return")
    counts = summary.get("counts", {})
    expected_counts = {
        "detected_nodes": expected["num_coordinates"],
        "pre_ilp_edges": expected["num_candidate_edges"],
        "output_edges": expected["num_post_ilp_edges"],
    }
    for key, value in expected_counts.items():
        if counts.get(key, {}).get("value") != value:
            raise ContractError(f"telemetry count differs for {key}")
    coordinate = summary.get("coordinate_artifact", {})
    if coordinate.get("status") != "available":
        raise ContractError("telemetry coordinate artifact is unavailable")
    if coordinate.get("coordinate_sha256") != expected["coordinate_sha256"]:
        raise ContractError("telemetry coordinate hash differs from model output")
    unavailable = [
        stage
        for stage, record in summary.get("durations", {}).items()
        if record.get("status") != "available"
    ]
    if unavailable:
        raise ContractError(f"telemetry stages unavailable: {unavailable}")
    return {
        "event_path": str(event_paths[0]),
        "event_sha256": sha256_file(event_paths[0]),
        "record_types": types_seen,
        "dataset_summary": summary,
    }


def run_arm(
    label: str,
    request: Request,
    public_source: str,
    evidence_root: Path,
    np: Any,
    torch: Any,
    zarr: Any,
    td: Any,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    check_deadline(started, request.max_wall_seconds, f"{label} setup")
    arm_root = evidence_root / label
    arm_root.mkdir(parents=True, exist_ok=False)
    diagnostic_root = arm_root / "diagnostics"
    telemetry_root = arm_root / "telemetry"
    runtime_module = None
    source_chain = None
    source_to_execute = public_source

    environment = public_environment(request.checkpoints[1])
    environment["BIOHUB_E0_TELEMETRY_DIR"] = str(telemetry_root) if label == "instrumented" else ""
    with temporary_environment(environment):
        if label == "instrumented":
            from biohub_ct.campaign import e0_support_telemetry

            helper_source = e0_support_telemetry.runtime_helper_source()
            runtime_module = load_source_module(
                e0_support_telemetry.RUNTIME_MODULE_NAME,
                helper_source,
                Path("<embedded-e0-support-runtime>"),
            )
            patched = e0_support_telemetry.patch_support_source(
                public_source, expected_public_patched_sha256=PUBLIC_SUPPORT_SHA256
            )
            source_to_execute = patched.source
            source_chain = patched.source_chain()
            telemetry_source_path = Path(e0_support_telemetry.__file__).resolve(strict=True)
            if not telemetry_source_path.is_relative_to((PROJECT_ROOT / "src").resolve()):
                raise ContractError(
                    "e0_support_telemetry resolved outside the staged project source: "
                    f"{telemetry_source_path}"
                )
        module = load_source_module(
            f"_e0_parity_{label}", source_to_execute, request.source.resolve()
        )
        support_import_origins = verify_support_import_origins(request.support_root)
        redirect_diagnostic_path(module, diagnostic_root)

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        device = torch.device("cuda:0")
        primary, window_size, downsample = module.load_model(request.checkpoints[0], device)
        secondary, secondary_window, secondary_downsample = module.load_model(
            request.checkpoints[1], device
        )
        if secondary_window != window_size or tuple(secondary_downsample) != tuple(downsample):
            raise ContractError("primary and secondary inference grids differ")
        if window_size != 2 or tuple(downsample) != (1, 4, 4):
            raise ContractError("loaded inference grid differs from public config")
        cfg = module.PredictConfig(
            det_threshold=PUBLIC_CONFIG["det_threshold"],
            det_tta=PUBLIC_CONFIG["det_tta"],
            pool_kernel_um=PUBLIC_CONFIG["pool_kernel_um"],
            edge_activation=PUBLIC_CONFIG["edge_activation"],
            threshold=PUBLIC_CONFIG["edge_candidate_threshold"],
            use_ilp=True,
            ilp_edge_weight=PUBLIC_CONFIG["ilp_edge_weight"],
            ilp_appearance_weight=PUBLIC_CONFIG["ilp_appearance_weight"],
            ilp_disappearance_weight=PUBLIC_CONFIG["ilp_disappearance_weight"],
            ilp_division_weight=PUBLIC_CONFIG["ilp_division_weight"],
        )
        dataset_path = request.data_root / f"{DATASET}.zarr"
        if runtime_module is not None:
            runtime_module.telemetry.begin_invocation(
                data_root=str(request.data_root.resolve()),
                output_dir=str(arm_root.resolve()),
                method="unet_transformer",
                fold=0,
                test_names=[dataset_path.name],
            )

        arm_started = time.perf_counter()
        coords, edges = module.predict_video(
            primary,
            dataset_path,
            device,
            cfg=cfg,
            window_size=window_size,
            max_frames=8,
            unet_batch_size=4,
            downsample=downsample,
            secondary_model=secondary,
            secondary_edge_weight=PUBLIC_CONFIG["secondary_edge_weight"],
            secondary_detection_weight=PUBLIC_CONFIG["secondary_detection_weight"],
            secondary_link_mode=PUBLIC_CONFIG["secondary_link_mode"],
            secondary_mix_temperature=PUBLIC_CONFIG["secondary_mix_temperature"],
            secondary_low_margin_max=PUBLIC_CONFIG["secondary_low_margin_max"],
        )
        check_deadline(started, request.max_wall_seconds, f"{label} prediction")
        arrays = candidate_arrays(coords, edges, np)
        if len(coords) == 0 or len(edges) == 0:
            raise ContractError(f"{label} did not produce nonempty detections and candidate edges")
        frame_counts = {
            int(frame): int((arrays["coordinates"][:, 0] == frame).sum()) for frame in FRAMES
        }
        if {int(value) for value in arrays["coordinates"][:, 0]} != set(FRAMES):
            raise ContractError(f"{label} detections do not cover exactly frames 0..7")
        if any(count <= 0 for count in frame_counts.values()):
            raise ContractError(f"{label} has an empty detection frame: {frame_counts}")
        stage_identity = candidate_identity(arrays, np)
        stage_npz = save_stage_npz(arm_root / "stages.npz", arrays, np)

        graph = (
            module._e0_build_graph(coords, edges)
            if runtime_module
            else module.build_graph(coords, edges)
        )
        if graph.num_nodes() == 0 or graph.num_edges() == 0:
            raise ContractError(f"{label} pre-ILP graph is empty")
        pre_graph = canonical_graph(graph)
        pre_artifact = write_canonical_json(arm_root / "pre-ilp-graph.json", pre_graph)
        solver = td.solvers.ILPSolver(
            edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
            appearance_weight=cfg.ilp_appearance_weight,
            disappearance_weight=cfg.ilp_disappearance_weight,
            division_weight=cfg.ilp_division_weight,
        )
        with module.suppress_output():
            graph = module._e0_solve(solver, graph) if runtime_module else solver.solve(graph)
        solver_invoked = True
        post_graph = canonical_graph(graph)
        if post_graph["num_nodes"] == 0 or post_graph["num_edges"] == 0:
            raise ContractError(f"{label} post-ILP graph is empty")
        component_span = max_component_span(post_graph)
        if component_span["maximum_unique_frames"] < 6:
            raise ContractError(f"{label} has no connected component spanning six observed frames")
        post_artifact = write_canonical_json(arm_root / "post-ilp-graph.json", post_graph)

        geff_path = arm_root / "output.geff"
        if runtime_module:
            module._e0_save_graph(graph, geff_path)
            runtime_module.telemetry.finish_dataset(
                detected_nodes=len(coords),
                pre_ilp_edges=len(edges),
                output_edges=graph.num_edges(),
            )
            runtime_module.telemetry.finish_invocation()
            runtime_module.telemetry.finalize_process()
        else:
            module.save_graph(graph, geff_path)
        loaded_graph = canonical_graph(graph_from_geff(td, geff_path))
        post_semantics = canonical_graph_semantics(post_graph)
        loaded_semantics = canonical_graph_semantics(loaded_graph)
        compare_exact(
            post_semantics,
            loaded_semantics,
            f"{label} in-memory versus GEFF-reloaded graph semantics",
        )
        loaded_artifact = write_canonical_json(arm_root / "reloaded-geff-graph.json", loaded_graph)
        geff = canonical_geff(geff_path, np, zarr)
        geff_artifact = write_canonical_json(arm_root / "canonical-geff.json", geff)
        check_deadline(started, request.max_wall_seconds, f"{label} serialization")

        summary = {
            "label": label,
            "elapsed_seconds": time.perf_counter() - arm_started,
            "source_sha256": sha256_bytes(source_to_execute.encode()),
            "source_chain": source_chain,
            "support_import_origins": support_import_origins,
            "num_coordinates": len(coords),
            "num_candidate_edges": len(edges),
            "num_pre_ilp_nodes": pre_graph["num_nodes"],
            "num_pre_ilp_edges": pre_graph["num_edges"],
            "num_post_ilp_nodes": post_graph["num_nodes"],
            "num_post_ilp_edges": post_graph["num_edges"],
            "solver_invoked": solver_invoked,
            "component_span": component_span,
            "coordinate_sha256": stage_identity["coordinates"]["sha256"],
            "candidate_identity": stage_identity,
            "detection_frame_counts": frame_counts,
            "pre_graph_sha256": sha256_bytes(canonical_bytes(pre_graph)),
            "post_graph_sha256": sha256_bytes(canonical_bytes(post_graph)),
            "reloaded_graph_sha256": sha256_bytes(canonical_bytes(loaded_graph)),
            "post_graph_semantic_sha256": sha256_bytes(canonical_bytes(post_semantics)),
            "reloaded_graph_semantic_sha256": sha256_bytes(canonical_bytes(loaded_semantics)),
            "geff_logical_sha256": sha256_bytes(canonical_bytes(geff)),
            "artifacts": [stage_npz, pre_artifact, post_artifact, loaded_artifact, geff_artifact],
        }
        if runtime_module is not None:
            summary["telemetry"] = validate_telemetry(telemetry_root, summary)

        del graph, solver, primary, secondary, coords, edges
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return summary, {
            "pre": pre_graph,
            "post": post_graph,
            "loaded": loaded_graph,
            "post_semantics": post_semantics,
            "loaded_semantics": loaded_semantics,
            "geff": geff,
            "candidates": arrays,
        }


def evidence_files(evidence_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in evidence_root.rglob("*") if item.is_file()):
        records.append(
            {
                "archive_member": path.relative_to(evidence_root.parent).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def deterministic_tar(evidence_root: Path, destination: Path) -> None:
    temporary = destination.with_name("." + destination.name + ".tmp")
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for path in sorted(evidence_root.rglob("*")):
                relative = path.relative_to(evidence_root.parent).as_posix()
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o755 if path.is_dir() else 0o644
                if path.is_file():
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                else:
                    archive.addfile(info)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run(request: Request, started: float) -> tuple[dict[str, Any], Path]:
    remaining = request.max_wall_seconds - (time.monotonic() - started)
    if remaining <= 0:
        raise DeadlineExceeded("application deadline elapsed before parity preflight")
    with alarm_deadline(remaining):
        if not (0 < request.max_wall_seconds <= MAX_APPLICATION_SECONDS):
            raise ContractError("max-wall-seconds must be in (0, 300]")
        if request.output.name != "parity.json":
            raise ContractError("output filename must be parity.json")
        evidence_root = request.output.parent / "evidence"
        archive_path = request.output.parent / "parity-artifacts.tar"
        if evidence_root.exists() or archive_path.exists() or request.output.exists():
            raise ContractError("parity outputs already exist")

        static = verify_static_inputs(request)
        staged = verify_staged_manifests(request)
        install_support_paths(request.support_root)
        import numpy as np
        import torch
        import tracksdata as td
        import zarr

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ContractError("exactly one visible CUDA device is required")
        device_name = str(torch.cuda.get_device_name(0))
        if "4070" not in device_name:
            raise ContractError(f"single RTX 4070 required, found {device_name!r}")
        torch.cuda.set_device(0)
        input_checks = [verify_input_snapshot(request, np, zarr, "before_control")]
        evidence_root.mkdir(parents=True, exist_ok=False)
        public_source = request.source.read_text(encoding="utf-8")
        control, control_values = run_arm(
            "control", request, public_source, evidence_root, np, torch, zarr, td, started
        )
        input_checks.append(verify_input_snapshot(request, np, zarr, "after_control"))
        instrumented, instrumented_values = run_arm(
            "instrumented", request, public_source, evidence_root, np, torch, zarr, td, started
        )
        input_checks.append(verify_input_snapshot(request, np, zarr, "after_instrumented"))

        if len({check["manifest_sha256"] for check in input_checks}) != 1:
            raise ContractError("input manifest identity changed between arm checks")
        input_identities = [
            {key: value for key, value in check.items() if key != "label"} for check in input_checks
        ]
        if any(identity != input_identities[0] for identity in input_identities[1:]):
            raise ContractError("raw frame or Zarr metadata identity changed between arm checks")

        for name in control_values["candidates"]:
            left = control_values["candidates"][name]
            right = instrumented_values["candidates"][name]
            if left.dtype != right.dtype or left.shape != right.shape:
                raise ParityMismatch(f"ordered candidate array contract differs: {name}")
            if left.tobytes(order="C") != right.tobytes(order="C"):
                raise ParityMismatch(f"ordered candidate array bytes differ: {name}")
        for stage in ("pre", "post", "loaded", "geff"):
            compare_exact(control_values[stage], instrumented_values[stage], stage)
        if not control["solver_invoked"] or not instrumented["solver_invoked"]:
            raise ContractError("both arms must invoke the ILP solver")
        if control["component_span"]["maximum_unique_frames"] < 6:
            raise ContractError("control component-span criterion failed")

        files = evidence_files(evidence_root)
        deterministic_tar(evidence_root, archive_path)
        if archive_path.stat().st_size > MAX_OUTPUT_BYTES:
            raise ContractError(
                f"evidence archive exceeds {MAX_OUTPUT_BYTES} bytes: {archive_path.stat().st_size}"
            )
        shutil.rmtree(evidence_root)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "scope": "bounded_e0_real_model_graph_telemetry_parity",
            "request": {
                "source": str(request.source.resolve()),
                "support_root": str(request.support_root.resolve()),
                "checkpoints": [str(path.resolve()) for path in request.checkpoints],
                "data_root": str(request.data_root.resolve()),
                "input_manifest": str(request.input_manifest.resolve()),
                "frames": list(FRAMES),
                "full_spatial": True,
                "max_wall_seconds": request.max_wall_seconds,
            },
            "public_configuration": PUBLIC_CONFIG,
            "effective_pool_kernel_evidence": (
                "3.0 micrometers from the exact public PredictConfig default; the public launch "
                "argv supplies no pool-kernel override. Checkpoint config pool_kernel_um=5.0 is "
                "unused by load_model/predict_video in this source."
            ),
            "static_identity": static,
            "staged_manifests": staged,
            "input_identity_checks": input_checks,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "cuda_device": device_name,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "zarr": zarr.__version__,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            },
            "arms": [control, instrumented],
            "parity": {
                "coordinates_exact": True,
                "ordered_candidate_edges_and_float_bits_exact": True,
                "canonical_pre_ilp_graph_exact": True,
                "canonical_post_ilp_graph_exact": True,
                "in_memory_to_reloaded_geff_semantics_exact_each_arm": True,
                "canonical_geff_arrays_and_attributes_exact": True,
                "solver_invoked_each_arm": True,
                "minimum_connected_unique_frames": 6,
            },
            "evidence_files": files,
            "artifact_archive": {
                "path": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
                "format": "deterministic_uncompressed_ustar",
                "contains_only": "evidence/ output tree",
            },
            "elapsed_seconds": time.monotonic() - started,
            "interpretation": (
                "Operational parity for one fixed eight-frame real-model graph slice. "
                "This is not a clean holdout or quality evaluation and makes no claim about "
                "the full notebook, later graph post-processing, submission validity, runtime "
                "on Kaggle, model quality, or reproduction of the displayed public score."
            ),
        }
        check_deadline(started, request.max_wall_seconds, "receipt construction")
        return receipt, archive_path


def write_result(context: CampaignContext, artifacts: Sequence[Path]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    hashes = {}
    for path in artifacts:
        resolved = path.resolve(strict=True)
        try:
            key = resolved.relative_to(project_root).as_posix()
        except ValueError as exc:
            raise ContractError(f"campaign artifact is outside project root: {resolved}") from exc
        hashes[key] = sha256_file(resolved)
    atomic_json(
        context.result_manifest_path,
        {
            **context.identity(),
            "status": "COMPLETE",
            "completed_units": 1,
            "artifact_sha256": hashes,
        },
    )


def base_receipt(request: Request, status: str, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "scope": "bounded_e0_real_model_graph_telemetry_parity",
        "error": {"type": type(error).__name__, "message": str(error)},
        "request": {
            "source": str(request.source),
            "support_root": str(request.support_root),
            "checkpoints": [str(path) for path in request.checkpoints],
            "data_root": str(request.data_root),
            "input_manifest": str(request.input_manifest),
            "frames": list(FRAMES),
            "full_spatial": True,
            "max_wall_seconds": request.max_wall_seconds,
        },
        "interpretation": (
            "No parity claim. The bounded operational check did not pass. No full-notebook, "
            "post-processing, quality, submission, Kaggle-runtime, or public-score claim is made."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=2, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=MAX_APPLICATION_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = Request(
        source=args.source,
        support_root=args.support_root,
        checkpoints=tuple(args.checkpoints),
        data_root=args.data_root,
        input_manifest=args.input_manifest,
        output=args.output,
        max_wall_seconds=args.max_wall_seconds,
    )
    collisions = [
        path
        for path in (
            request.output,
            request.output.parent / "evidence",
            request.output.parent / "parity-artifacts.tar",
            request.output.parent / "result.json",
        )
        if path.exists()
    ]
    if collisions:
        print(
            json.dumps(
                {
                    "status": "INCONCLUSIVE",
                    "error": "refusing to overwrite pre-existing parity outputs",
                    "paths": [str(path) for path in collisions],
                }
            )
        )
        return 2
    started = time.monotonic()
    context = None
    try:
        context = campaign_context(request)
        if context:
            write_progress(context, 0, None)
        receipt, archive_path = run(request, started)
        exit_code = 0
    except ParityMismatch as exc:
        receipt = base_receipt(request, "FAIL", exc)
        archive_path = request.output.parent / "parity-artifacts.tar"
        exit_code = 1
    except Exception as exc:  # noqa: BLE001 - every runtime/dependency failure is INCONCLUSIVE
        receipt = base_receipt(request, "INCONCLUSIVE", exc)
        archive_path = request.output.parent / "parity-artifacts.tar"
        exit_code = 2
    receipt["elapsed_seconds"] = time.monotonic() - started
    if context:
        receipt["campaign_identity"] = context.identity()
    atomic_json(request.output, receipt)
    if context:
        if exit_code == 0:
            write_progress(context, 1, None)
            write_result(context, (request.output, context.progress_path, archive_path))
        else:
            error = receipt["error"]
            write_progress(context, 0, f"{error['type']}: {error['message']}")
    print(json.dumps({"status": receipt["status"], "output": str(request.output)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
