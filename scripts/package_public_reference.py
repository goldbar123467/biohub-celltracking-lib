"""Build the pinned E0 public-reference notebook without changing its algorithm.

The public notebook's original cell sources are preserved byte-for-byte. This
builder clears stale outputs, adds a fail-closed input-integrity cell before the
public cells, and adds a release-validation cell after them. It never contacts
Kaggle or launches a notebook.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

PINNED_REFERENCE: dict[str, Any] = {
    "schema_version": 1,
    "evidence_class": "overlap_unknown",
    "notebook": {
        "ref": "redoctopusk/biohub-942tta",
        "version": 1,
        "script_version_id": 347821442,
        "source_sha256": "521cb97f0f457643379a51b60c4f71e3f4cc7d1823fd98cbb97633ffaa515ec4",
        "displayed_public_score": 0.946,
        "displayed_runtime": "1h 21m 41s",
        "machine_shape": "NvidiaTeslaT4",
        "displayed_accelerators": 2,
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@sha256:"
            "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        ),
        "enable_internet": False,
    },
    "datasets": {
        "pilkwang/biohub-tracking-support-pack-50ep-v1": {
            "dataset_id": 10999845,
            "version": 10,
            "last_updated_utc": "2026-07-08T22:42:12.497Z",
            "license": "CC0-1.0",
            "archive_sha256": "2ea8f16d8e2df6781f3d48713004b18075ae3e462266773670de05a56f908c8a",
            "required_members": {
                "ARTIFACT_MANIFEST.json": None,
                "repo/scripts/predict_unet_transformer.py": None,
                "weights/unet_transformer/split_0/edge_predictor_best.pth": (
                    "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
                ),
            },
            "minimum_wheel_count": 62,
        },
        "pilkwang/biohub-temporal-unet3d-seed314159-v1": {
            "dataset_id": 11184174,
            "version": 2,
            "last_updated_utc": "2026-07-20T08:26:25.767Z",
            "license": "CC0-1.0",
            "archive_sha256": "b1fe1b4636fb7510d247c95a59ed16c523346388659e11285ffe8eea343c98bf",
            "required_members": {
                "ARTIFACT_MANIFEST.json": None,
                "weights/unet_transformer/split_0/config.json": None,
                "weights/unet_transformer/split_0/edge_predictor_best.pth": (
                    "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
                ),
            },
            "minimum_wheel_count": 62,
        },
        "pilkwang/biohub-deepcenter-unet3d-center-prior-v1": {
            "dataset_id": 11061989,
            "version": 5,
            "last_updated_utc": "2026-07-07T11:53:58.670Z",
            "license": "CC0-1.0",
            "archive_sha256": "c0fddb8ae879e6e6742fa0a6921e226265bb2f5306858c952ded3314cc2bdaf1",
            "required_members": {
                "ARTIFACT_MANIFEST.json": None,
                "weights/full_frame_center/best.pt": (
                    "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0"
                ),
            },
            "minimum_wheel_count": 0,
        },
    },
    "competition": "biohub-cell-tracking-during-development",
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

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_pin(pin: Mapping[str, Any]) -> None:
    notebook = pin.get("notebook")
    datasets = pin.get("datasets")
    if not isinstance(notebook, Mapping) or not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("Reference pin must contain notebook and nonempty datasets mappings")
    if not isinstance(notebook.get("version"), int) or notebook["version"] < 1:
        raise ValueError("Pinned notebook version must be a positive integer")
    if not isinstance(notebook.get("script_version_id"), int):
        raise TypeError("Pinned script version ID must be an integer")
    for label, digest in (
        ("notebook source", notebook.get("source_sha256")),
        *[(f"dataset archive {ref}", spec.get("archive_sha256")) for ref, spec in datasets.items()],
    ):
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"Invalid SHA-256 for {label}")
    for ref, spec in datasets.items():
        if not isinstance(ref, str) or ref.count("/") != 1 or not isinstance(spec, Mapping):
            raise ValueError(f"Malformed dataset reference: {ref!r}")
        if not isinstance(spec.get("version"), int) or spec["version"] < 1:
            raise ValueError(f"Invalid dataset version for {ref}")
        required = spec.get("required_members")
        if not isinstance(required, Mapping) or not required:
            raise ValueError(f"No required members pinned for {ref}")
        for name, digest in required.items():
            _validate_member_name(name)
            if digest is not None and (
                not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None
            ):
                raise ValueError(f"Invalid member SHA-256 for {ref}:{name}")


def _validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member name: {name!r}")


def inspect_archive(path: Path, ref: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = path.resolve(strict=True)
    actual_archive_sha256 = sha256_file(path)
    if actual_archive_sha256 != spec["archive_sha256"]:
        raise ValueError(
            f"Archive checksum mismatch for {ref}: expected {spec['archive_sha256']}, "
            f"got {actual_archive_sha256}"
        )
    members: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Corrupt ZIP member for {ref}: {corrupt}")
        infos = [info for info in archive.infolist() if not info.is_dir()]
        for info in infos:
            _validate_member_name(info.filename)
            if info.filename in members:
                raise ValueError(f"Duplicate ZIP member for {ref}: {info.filename}")
            with archive.open(info, "r") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            members[info.filename] = {"bytes": info.file_size, "sha256": digest}
        manifest_record = members.get("ARTIFACT_MANIFEST.json")
        if manifest_record is None:
            raise ValueError(f"Missing ARTIFACT_MANIFEST.json in {ref}")
        try:
            artifact_manifest = json.loads(archive.read("ARTIFACT_MANIFEST.json"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed artifact manifest in {ref}") from exc

    for name, expected_digest in spec["required_members"].items():
        actual = members.get(name)
        if actual is None:
            raise ValueError(f"Missing pinned member for {ref}: {name}")
        if expected_digest is not None and actual["sha256"] != expected_digest:
            raise ValueError(
                f"Member checksum mismatch for {ref}:{name}: expected {expected_digest}, "
                f"got {actual['sha256']}"
            )
    wheel_count = sum(name.startswith("wheels/") and name.endswith(".whl") for name in members)
    if wheel_count < int(spec.get("minimum_wheel_count", 0)):
        raise ValueError(
            f"Offline wheel coverage mismatch for {ref}: found {wheel_count}, "
            f"expected at least {spec['minimum_wheel_count']}"
        )
    return {
        "ref": ref,
        "dataset_id": spec["dataset_id"],
        "version": spec["version"],
        "last_updated_utc": spec["last_updated_utc"],
        "license": spec["license"],
        "archive_file": path.name,
        "archive_bytes": path.stat().st_size,
        "archive_sha256": actual_archive_sha256,
        "artifact_manifest": artifact_manifest,
        "files": members,
        "file_count": len(members),
        "wheel_count": wheel_count,
    }


def verify_reference_inputs(
    source_notebook: Path,
    archives: Mapping[str, Path],
    *,
    pin: Mapping[str, Any] = PINNED_REFERENCE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_pin(pin)
    source_notebook = source_notebook.resolve(strict=True)
    actual_source_sha256 = sha256_file(source_notebook)
    if actual_source_sha256 != pin["notebook"]["source_sha256"]:
        raise ValueError(
            "Notebook source checksum mismatch: expected "
            f"{pin['notebook']['source_sha256']}, got {actual_source_sha256}"
        )
    expected_refs = set(pin["datasets"])
    actual_refs = set(archives)
    if actual_refs != expected_refs:
        raise ValueError(
            "Dataset reference mismatch: "
            f"missing={sorted(expected_refs - actual_refs)}, extra={sorted(actual_refs - expected_refs)}"
        )
    try:
        notebook = json.loads(source_notebook.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Source notebook is malformed JSON") from exc
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("Source must be a valid nbformat 4 notebook")
    inspected = {
        ref: inspect_archive(Path(archives[ref]), ref, pin["datasets"][ref])
        for ref in sorted(expected_refs)
    }
    return notebook, inspected


def _code_cell(source: str, purpose: str) -> dict[str, Any]:
    compile(source, purpose, "exec")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"biohub_e0_instrumentation": purpose},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _preflight_source(lock: Mapping[str, Any]) -> str:
    embedded = json.dumps(lock, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"""# E0 packaging instrumentation: exact public inputs, before any public code executes.
import hashlib as _e0_hashlib
import json as _e0_json
import time as _e0_time
from pathlib import Path as _E0Path

_E0_STARTED_MONOTONIC = _e0_time.monotonic()
_E0_LOCK = _e0_json.loads({embedded!r})

def _e0_sha256(path):
    with _E0Path(path).open("rb") as stream:
        return _e0_hashlib.file_digest(stream, "sha256").hexdigest()

def _e0_dataset_root(ref):
    owner, slug = ref.split("/", 1)
    candidates = [
        _E0Path("/kaggle/input/datasets") / owner / slug,
        _E0Path("/kaggle/input") / slug,
    ]
    roots = {{path.resolve() for path in candidates if path.is_dir()}}
    if len(roots) != 1:
        raise RuntimeError({{"dataset": ref, "resolved_roots": sorted(map(str, roots))}})
    return roots.pop()

_E0_DATASET_ROOTS = {{}}
_E0_INPUT_INTEGRITY = {{}}
for _e0_ref, _e0_spec in sorted(_E0_LOCK["datasets"].items()):
    _e0_root = _e0_dataset_root(_e0_ref)
    _e0_actual_names = {{
        path.relative_to(_e0_root).as_posix()
        for path in _e0_root.rglob("*") if path.is_file()
    }}
    _e0_expected_names = set(_e0_spec["files"])
    if _e0_actual_names != _e0_expected_names:
        raise RuntimeError({{
            "dataset": _e0_ref,
            "missing": sorted(_e0_expected_names - _e0_actual_names),
            "extra": sorted(_e0_actual_names - _e0_expected_names),
        }})
    _e0_mismatches = {{}}
    for _e0_name in sorted(_e0_expected_names):
        _e0_path = _e0_root / _e0_name
        _e0_actual_size = _e0_path.stat().st_size
        _e0_actual_hash = _e0_sha256(_e0_path)
        _e0_expected = _e0_spec["files"][_e0_name]
        if (_e0_actual_size != _e0_expected["bytes"] or
                _e0_actual_hash != _e0_expected["sha256"]):
            _e0_mismatches[_e0_name] = {{
                "expected": _e0_expected,
                "actual": {{"bytes": _e0_actual_size, "sha256": _e0_actual_hash}},
            }}
    if _e0_mismatches:
        raise RuntimeError({{"dataset": _e0_ref, "integrity_mismatches": _e0_mismatches}})
    _E0_DATASET_ROOTS[_e0_ref] = str(_e0_root)
    _E0_INPUT_INTEGRITY[_e0_ref] = {{
        "status": "verified",
        "root": str(_e0_root),
        "version": _e0_spec["version"],
        "files": len(_e0_expected_names),
        "bytes": sum(record["bytes"] for record in _e0_spec["files"].values()),
    }}
print("E0_INPUT_INTEGRITY", _e0_json.dumps(_E0_INPUT_INTEGRITY, sort_keys=True))
"""


def _release_validation_source(lock: Mapping[str, Any]) -> str:
    columns = repr(lock["submission_columns"])
    competition = lock["competition"]
    return f"""# E0 packaging instrumentation: final hidden-input and CSV release evidence.
import csv as _e0_csv
import importlib.metadata as _e0_metadata
import math as _e0_math
import os as _e0_os
import platform as _e0_platform
import sys as _e0_sys

_E0_COLUMNS = {columns}
_e0_submission = _E0Path("/kaggle/working/submission.csv")
if not _e0_submission.is_file():
    raise FileNotFoundError(_e0_submission)
_e0_competition = {competition!r}
_e0_competition_roots = [
    _E0Path("/kaggle/input/competitions") / _e0_competition,
    _E0Path("/kaggle/input") / _e0_competition,
]
_e0_test_roots = {{(path / "test").resolve() for path in _e0_competition_roots
                  if (path / "test").is_dir()}}
if len(_e0_test_roots) != 1:
    raise RuntimeError({{"competition_test_roots": sorted(map(str, _e0_test_roots))}})
_e0_test_root = _e0_test_roots.pop()
_e0_shapes = {{}}
for _e0_zarr in sorted(_e0_test_root.glob("*.zarr")):
    _e0_metadata_path = _e0_zarr / "0" / "zarr.json"
    if not _e0_metadata_path.is_file():
        raise FileNotFoundError(_e0_metadata_path)
    _e0_array_metadata = _e0_json.loads(_e0_metadata_path.read_text())
    _e0_shape = _e0_array_metadata.get("shape")
    if (not isinstance(_e0_shape, list) or len(_e0_shape) != 4 or
            any(not isinstance(v, int) or v <= 0 for v in _e0_shape)):
        raise RuntimeError({{"dataset": _e0_zarr.stem, "invalid_shape": _e0_shape}})
    _e0_shapes[_e0_zarr.stem] = _e0_shape
if not _e0_shapes:
    raise RuntimeError("No test Zarr inputs discovered")

def _e0_int(raw, column, row_id):
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError({{"row": row_id, "column": column, "value": raw}}) from exc
    if str(value) != str(raw):
        raise RuntimeError({{"row": row_id, "noncanonical_integer": column, "value": raw}})
    return value

_e0_nodes = {{dataset: {{}} for dataset in _e0_shapes}}
_e0_counts = {{dataset: {{"nodes": 0, "edges": 0}} for dataset in _e0_shapes}}
_e0_rows = 0
_e0_current_dataset = None
_e0_seen_dataset_blocks = set()
with _e0_submission.open("r", encoding="utf-8", newline="") as _e0_stream:
    _e0_reader = _e0_csv.DictReader(_e0_stream)
    if _e0_reader.fieldnames != _E0_COLUMNS:
        raise RuntimeError({{"expected_header": _E0_COLUMNS, "actual_header": _e0_reader.fieldnames}})
    for _e0_row in _e0_reader:
        if None in _e0_row or any(value is None or value == "" for value in _e0_row.values()):
            raise RuntimeError({{"unexpected_blank_missing_or_extra_field": _e0_rows}})
        _e0_row_id = _e0_int(_e0_row["id"], "id", _e0_rows)
        if _e0_row_id != _e0_rows:
            raise RuntimeError({{"expected_id": _e0_rows, "actual_id": _e0_row_id}})
        _e0_dataset = _e0_row["dataset"]
        if _e0_dataset not in _e0_shapes:
            raise RuntimeError({{"unexpected_dataset": _e0_dataset, "row": _e0_row_id}})
        if _e0_dataset != _e0_current_dataset:
            if _e0_dataset in _e0_seen_dataset_blocks:
                raise RuntimeError({{"noncontiguous_dataset": _e0_dataset, "row": _e0_row_id}})
            _e0_seen_dataset_blocks.add(_e0_dataset)
            _e0_current_dataset = _e0_dataset
        _e0_kind = _e0_row["row_type"]
        if _e0_kind == "node":
            if _e0_int(_e0_row["source_id"], "source_id", _e0_row_id) != -1 or \
                    _e0_int(_e0_row["target_id"], "target_id", _e0_row_id) != -1:
                raise RuntimeError({{"node_sentinel_error": _e0_row_id}})
            _e0_node_id = _e0_int(_e0_row["node_id"], "node_id", _e0_row_id)
            if _e0_node_id < 0:
                raise RuntimeError({{"negative_node_id": [_e0_dataset, _e0_node_id]}})
            _e0_coords = [_e0_int(_e0_row[name], name, _e0_row_id) for name in ("t", "z", "y", "x")]
            if any(value < 0 or value >= limit for value, limit in zip(_e0_coords, _e0_shapes[_e0_dataset])):
                raise RuntimeError({{"out_of_bounds_node": _e0_row_id, "coords": _e0_coords,
                                     "shape": _e0_shapes[_e0_dataset]}})
            if _e0_node_id in _e0_nodes[_e0_dataset]:
                raise RuntimeError({{"duplicate_node": [_e0_dataset, _e0_node_id]}})
            _e0_nodes[_e0_dataset][_e0_node_id] = _e0_coords[0]
            _e0_counts[_e0_dataset]["nodes"] += 1
        elif _e0_kind == "edge":
            if any(_e0_int(_e0_row[name], name, _e0_row_id) != -1
                   for name in ("node_id", "t", "z", "y", "x")):
                raise RuntimeError({{"edge_sentinel_error": _e0_row_id}})
            _e0_counts[_e0_dataset]["edges"] += 1
        else:
            raise RuntimeError({{"invalid_row_type": _e0_kind, "row": _e0_row_id}})
        _e0_rows += 1
if _e0_rows == 0 or any(record["nodes"] == 0 for record in _e0_counts.values()):
    raise RuntimeError({{"empty_submission_or_dataset": _e0_counts}})

_e0_indegree = {{dataset: {{}} for dataset in _e0_shapes}}
_e0_outdegree = {{dataset: {{}} for dataset in _e0_shapes}}
with _e0_submission.open("r", encoding="utf-8", newline="") as _e0_stream:
    for _e0_row_id, _e0_row in enumerate(_e0_csv.DictReader(_e0_stream)):
        if _e0_row["row_type"] != "edge":
            continue
        _e0_dataset = _e0_row["dataset"]
        _e0_source = _e0_int(_e0_row["source_id"], "source_id", _e0_row_id)
        _e0_target = _e0_int(_e0_row["target_id"], "target_id", _e0_row_id)
        if _e0_source < 0 or _e0_target < 0:
            raise RuntimeError({{"negative_edge_endpoint": [_e0_dataset, _e0_source, _e0_target]}})
        _e0_times = _e0_nodes[_e0_dataset]
        if _e0_source not in _e0_times or _e0_target not in _e0_times:
            raise RuntimeError({{"dangling_edge": [_e0_dataset, _e0_source, _e0_target]}})
        if _e0_times[_e0_target] != _e0_times[_e0_source] + 1:
            raise RuntimeError({{"nonconsecutive_edge": [_e0_dataset, _e0_source, _e0_target]}})
        _e0_outdegree[_e0_dataset][_e0_source] = _e0_outdegree[_e0_dataset].get(_e0_source, 0) + 1
        _e0_indegree[_e0_dataset][_e0_target] = _e0_indegree[_e0_dataset].get(_e0_target, 0) + 1
        if _e0_outdegree[_e0_dataset][_e0_source] > 2 or _e0_indegree[_e0_dataset][_e0_target] > 1:
            raise RuntimeError({{"invalid_degree": [_e0_dataset, _e0_source, _e0_target]}})

_e0_distributions = [
    "torch", "numpy", "pandas", "scipy", "scikit-image", "zarr", "geff", "geff-spec",
    "tracksdata", "polars", "pyscipopt", "ilpy", "dask", "imagecodecs", "pyarrow",
    "rustworkx", "sqlalchemy", "blosc2", "numcodecs",
]
_e0_versions = {{}}
for _e0_distribution in _e0_distributions:
    try:
        _e0_versions[_e0_distribution] = _e0_metadata.version(_e0_distribution)
    except _e0_metadata.PackageNotFoundError:
        _e0_versions[_e0_distribution] = None

_e0_run_manifest = {{
    "schema_version": 1,
    "status": "COMPLETE",
    "reference": _E0_LOCK["reference"],
    "release_digest": _E0_LOCK["release_digest"],
    "input_integrity": _E0_INPUT_INTEGRITY,
    "actual_input_shapes_tzyx": _e0_shapes,
    "dataset_shapes": _e0_shapes,
    "expected_dataset_ids": sorted(_e0_shapes),
    "completed_dataset_ids": sorted(_e0_counts),
    "csv_sha256": _e0_sha256(_e0_submission),
    "coordinate_bounds_valid": True,
    "graph_valid": True,
    "hidden_discovery_supported": True,
    "internet_enabled_configured": _E0_LOCK["reference"]["notebook"]["enable_internet"],
    "submission": {{
        "path": str(_e0_submission),
        "sha256": _e0_sha256(_e0_submission),
        "rows": _e0_rows,
        "datasets": _e0_counts,
        "validation": "PASS",
    }},
    "effective_biohub_environment": {{
        key: value for key, value in sorted(_e0_os.environ.items()) if key.startswith("BIOHUB_")
    }},
    "environment": {{
        "python": _e0_sys.version,
        "platform": _e0_platform.platform(),
        "distributions": _e0_versions,
    }},
    "elapsed_seconds": _e0_time.monotonic() - _E0_STARTED_MONOTONIC,
}}
_e0_run_manifest["full_runtime_seconds"] = _e0_run_manifest["elapsed_seconds"]
try:
    import torch as _e0_torch
    _e0_run_manifest["environment"]["cuda_available"] = _e0_torch.cuda.is_available()
    _e0_run_manifest["environment"]["cuda_devices"] = [
        _e0_torch.cuda.get_device_name(index) for index in range(_e0_torch.cuda.device_count())
    ]
except Exception as _e0_cuda_error:
    _e0_run_manifest["environment"]["cuda_probe_error"] = repr(_e0_cuda_error)

_e0_manifest_path = _E0Path("/kaggle/working/public_reference_run_manifest.json")
_e0_manifest_tmp = _e0_manifest_path.with_suffix(".json.tmp")
_e0_manifest_tmp.write_text(_e0_json.dumps(
    _e0_run_manifest, indent=2, sort_keys=True, allow_nan=False
) + "\\n")
_e0_manifest_tmp.replace(_e0_manifest_path)
print("E0_RELEASE_VALIDATION", _e0_json.dumps({{
    "status": "PASS", "manifest": str(_e0_manifest_path),
    "submission_sha256": _e0_run_manifest["submission"]["sha256"],
    "rows": _e0_rows, "input_shapes": _e0_shapes,
}}, sort_keys=True))
"""


def build_package(
    source_notebook: Path,
    archives: Mapping[str, Path],
    output_dir: Path,
    kernel_id: str,
    *,
    pin: Mapping[str, Any] = PINNED_REFERENCE,
) -> dict[str, Any]:
    if kernel_id.count("/") != 1 or any(not part for part in kernel_id.split("/")):
        raise ValueError("Kernel ID must have owner/slug form")
    source, inspected = verify_reference_inputs(source_notebook, archives, pin=pin)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_cell_hashes = [
        hashlib.sha256("".join(cell.get("source", [])).encode("utf-8")).hexdigest()
        for cell in source["cells"]
    ]
    artifact_lock = {
        "schema_version": 1,
        "reference": copy.deepcopy(dict(pin)),
        "competition": pin["competition"],
        "submission_columns": copy.deepcopy(pin["submission_columns"]),
        "datasets": inspected,
    }
    artifact_lock["release_digest"] = hashlib.sha256(
        canonical_json_bytes(artifact_lock)
    ).hexdigest()

    packaged = copy.deepcopy(source)
    packaged["cells"] = (
        [_code_cell(_preflight_source(artifact_lock), "e0-input-integrity")]
        + packaged["cells"]
        + [_code_cell(_release_validation_source(artifact_lock), "e0-release-validation")]
    )
    for cell in packaged["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            compile("".join(cell.get("source", [])), "packaged-notebook-cell", "exec")
    packaged.setdefault("metadata", {})["biohub_e0"] = {
        "release_digest": artifact_lock["release_digest"],
        "source_ref": pin["notebook"]["ref"],
        "source_version": pin["notebook"]["version"],
        "source_sha256": pin["notebook"]["source_sha256"],
        "algorithm_cells_preserved": len(source["cells"]),
        "evidence_class": pin["evidence_class"],
    }
    notebook_path = output_dir / "submission.ipynb"
    write_json(notebook_path, packaged)

    metadata = {
        "id": kernel_id,
        "title": "Biohub E0 Public Reference Reproduction",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": sorted(
            f"{ref}/{spec['version']}" for ref, spec in pin["datasets"].items()
        ),
        "competition_sources": [pin["competition"]],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": pin["notebook"]["machine_shape"],
    }
    write_json(output_dir / "kernel-metadata.json", metadata)
    write_json(output_dir / "artifact-lock.json", artifact_lock)

    manifest = {
        "schema_version": 1,
        "status": "ready_for_root_review_not_launched",
        "release_digest": artifact_lock["release_digest"],
        "source_notebook": {
            **copy.deepcopy(pin["notebook"]),
            "cell_count": len(source["cells"]),
            "cell_source_sha256": source_cell_hashes,
        },
        "packaged_notebook": {
            "path": notebook_path.name,
            "sha256": sha256_file(notebook_path),
            "cell_count": len(packaged["cells"]),
            "algorithm_cell_positions": [1, len(source["cells"])],
            "algorithm_cell_source_sha256": source_cell_hashes,
            "instrumentation": ["e0-input-integrity", "e0-release-validation"],
        },
        "kernel_metadata_sha256": sha256_file(output_dir / "kernel-metadata.json"),
        "artifact_lock_sha256": sha256_file(output_dir / "artifact-lock.json"),
        "dataset_archives": {
            ref: {
                key: record[key]
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
            for ref, record in inspected.items()
        },
        "algorithm_changes": [],
        "packaging_changes": [
            "prepend exact mounted-file integrity verification",
            "append actual TZYX shape capture and strict submission.csv validation",
            "append environment, effective BIOHUB settings, and output manifest capture",
            "set private GPU notebook metadata with internet disabled",
            "pin every Kaggle dataset source to its reviewed numeric version",
            "clear execution outputs and counts",
        ],
        "evidence_class": pin["evidence_class"],
        "known_limitations": [
            "Public score 0.946 is Kaggle-displayed upstream evidence, not our reproduction result.",
            "Model training includes both competition-training embryos; local clean holdout status is unavailable.",
            "Package has not been pushed or executed on Kaggle.",
        ],
    }
    write_json(output_dir / "package-manifest.json", manifest)
    return manifest


def parse_archive_arguments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Archive must use REF=PATH form: {value!r}")
        ref, raw_path = value.split("=", 1)
        if not ref or not raw_path or ref in result:
            raise ValueError(f"Malformed or duplicate archive argument: {value!r}")
        result[ref] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--archive", action="append", default=[], metavar="REF=PATH", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    args = parser.parse_args()
    manifest = build_package(
        args.source_notebook,
        parse_archive_arguments(args.archive),
        args.output_dir,
        args.kernel_id,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
