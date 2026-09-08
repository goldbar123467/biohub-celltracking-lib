"""Independent validation of an exact-version E0 Kaggle rehearsal output.

The notebook's manifest is evidence only after an authenticated exact-version
transport binds its bytes to the reviewed package.  This module then validates
that manifest and streams the raw CSV independently of the notebook code.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from biohub_ct.campaign.kaggle_rehearsal import (
    E0_R3_PACKAGE_IDENTITY,
    PackageIdentity,
    RehearsalError,
    preflight_package,
)
from biohub_ct.config import SUBMISSION_COLUMNS
from biohub_ct.submission.validator import SubmissionError, validate_submission


class ReleaseValidationError(ValueError):
    """Downloaded release evidence is malformed, inconsistent, or inadmissible."""


@dataclass(frozen=True)
class ExactVersionOutputProof:
    """Narrow boundary supplied by an authenticated exact-version downloader."""

    schema_version: int
    kind: str
    status: str
    transport_status: str
    notebook_slug: str
    notebook_version: int
    version_label: str
    release_digest: str
    run_manifest_sha256: str
    submission_sha256: str
    embedded_release_identity_verified: bool
    embedded_reference_verified: bool
    embedded_submission_hash_verified: bool
    output_files: Sequence[Mapping[str, object]]
    expected_input_shapes_tzyx: Mapping[str, object] | None
    shape_identity_status: str
    release_validation_status: str

    @classmethod
    def from_value(
        cls, value: ExactVersionOutputProof | Mapping[str, object]
    ) -> ExactVersionOutputProof:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ReleaseValidationError("exact-version output proof must be a mapping")
        try:
            return cls(**{field: value[field] for field in cls.__dataclass_fields__})  # type: ignore[arg-type]
        except (KeyError, TypeError) as exc:
            raise ReleaseValidationError("exact-version output proof has the wrong schema") from exc


@dataclass(frozen=True)
class DatasetReleaseSummary:
    dataset: str
    shape_tzyx: tuple[int, int, int, int]
    nodes: int
    edges: int
    forks: int
    max_indegree: int
    max_outdegree: int


@dataclass(frozen=True)
class ReleaseValidationResult:
    """Strictly JSON-serializable successful structural and identity result."""

    schema_version: int
    status: str
    release_digest: str
    notebook_slug: str
    notebook_version: int
    submission_sha256: str
    run_manifest_sha256: str
    package_manifest_sha256: str
    artifact_lock_sha256: str
    row_count: int
    datasets: tuple[DatasetReleaseSummary, ...]
    official_format_status: str
    scorer_compatibility_status: str
    e0_lineage_contract_status: str
    identity_status: str
    quality_admission: str
    resource_admission: str
    missing_quality_evidence: tuple[str, ...]
    missing_resource_evidence: tuple[str, ...]
    full_runtime_seconds: float

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["datasets"] = [asdict(dataset) for dataset in self.datasets]
        return value


@dataclass(frozen=True)
class _ParsedCSV:
    sha256: str
    row_count: int
    datasets: tuple[DatasetReleaseSummary, ...]


_SHA256_LENGTH = 64
_QUALITY_EVIDENCE_PATHS = (
    "peak_extraction.pre_cap_candidates",
    "peak_extraction.capped_frame_count",
    "peak_extraction.capped_frame_denominator",
    "peak_extraction.capped_clip_count",
    "peak_extraction.capped_clip_denominator",
    "execution.output_fallback_count",
    "execution.retry_count",
)
_RESOURCE_EVIDENCE_PATHS = (
    "stage_times_seconds",
    "peak_ram_bytes",
    "peak_vram_bytes",
    "verified_runtime_limit_seconds",
    "runtime_headroom_seconds",
    "measured_quota_debit",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_object(path: Path) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReleaseValidationError(f"cannot read {path.name}") from exc

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseValidationError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ReleaseValidationError(f"nonfinite JSON value in {path.name}: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"cannot parse strict JSON from {path.name}") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"{path.name} must contain a JSON object")
    return value, _sha256_bytes(raw)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReleaseValidationError(f"{label} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise ReleaseValidationError(f"{label} keys must be strings")
    return value  # type: ignore[return-value]


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReleaseValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _require_positive_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseValidationError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ReleaseValidationError(f"{label} must be a finite positive number")
    return result


def _canonical_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _validate_proof(
    value: ExactVersionOutputProof | Mapping[str, object],
    *,
    identity: PackageIdentity,
    manifest_sha256: str,
    submission_sha256: str,
    manifest_bytes: int,
    submission_bytes: int,
    canonicalized_notebook: tuple[str, int] | None = None,
) -> tuple[ExactVersionOutputProof, dict[str, tuple[int, int, int, int]]]:
    proof = ExactVersionOutputProof.from_value(value)
    if (
        proof.schema_version != 1
        or proof.kind != "KAGGLE_EXACT_VERSION_OUTPUT_PROOF"
        or proof.status != "PASS"
        or proof.transport_status != "VERIFIED"
    ):
        raise ReleaseValidationError(
            "exact-version output proof is not verified PASS schema version 1"
        )
    expected_slug = identity.notebook_slug
    if canonicalized_notebook is not None:
        expected_slug, expected_version = canonicalized_notebook
        if proof.notebook_version != expected_version:
            raise ReleaseValidationError("output version differs from canonicalization proof")
    if proof.notebook_slug != expected_slug:
        raise ReleaseValidationError("exact-version proof notebook slug differs from the package")
    _require_int(proof.notebook_version, "exact-version proof notebook_version", minimum=1)
    if proof.version_label != f"v{proof.notebook_version}":
        raise ReleaseValidationError("exact-version proof version label is inconsistent")
    if proof.release_digest != identity.release_digest:
        raise ReleaseValidationError("exact-version proof release digest differs from the package")
    for label, actual, expected in (
        ("run manifest", proof.run_manifest_sha256, manifest_sha256),
        ("submission", proof.submission_sha256, submission_sha256),
    ):
        if not _is_sha256(actual) or actual != expected:
            raise ReleaseValidationError(f"exact-version proof {label} hash mismatch")
    if (
        proof.embedded_release_identity_verified is not True
        or proof.embedded_reference_verified is not True
        or proof.embedded_submission_hash_verified is not True
    ):
        raise ReleaseValidationError("exact-version proof did not verify all embedded identities")
    if proof.release_validation_status != "UNRESOLVED_PENDING_INDEPENDENT_VALIDATOR":
        raise ReleaseValidationError("transport proof pre-claimed independent release validation")
    if proof.shape_identity_status != "VERIFIED" or proof.expected_input_shapes_tzyx is None:
        raise ReleaseValidationError("exact-version proof has no verified input shape identity")
    expected_shapes = _validate_shape_map(
        proof.expected_input_shapes_tzyx, "proof expected input shapes"
    )

    if not isinstance(proof.output_files, Sequence) or isinstance(proof.output_files, (str, bytes)):
        raise ReleaseValidationError("exact-version proof output_files must be a list")
    files_by_basename: dict[str, list[Mapping[str, object]]] = {}
    seen_paths: set[str] = set()
    for index, raw_file in enumerate(proof.output_files):
        record = _require_mapping(raw_file, f"exact-version proof output_files[{index}]")
        basename = record.get("basename")
        path = record.get("path")
        size = record.get("bytes")
        digest = record.get("sha256")
        if (
            not isinstance(basename, str)
            or not basename
            or not isinstance(path, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not _is_sha256(digest)
        ):
            raise ReleaseValidationError("exact-version proof output file inventory is invalid")
        # Zarr outputs routinely contain the same basename in many directories.
        # Bind inventory uniqueness to the full path, while keeping the two
        # required release artifacts unambiguous across the complete inventory.
        normalized_path = PurePosixPath(path.replace("\\", "/"))
        if normalized_path.name != basename:
            raise ReleaseValidationError("exact-version proof output file inventory is invalid")
        if ".." in normalized_path.parts or "\x00" in path:
            raise ReleaseValidationError("exact-version proof output path is invalid")
        path_key = normalized_path.as_posix()
        if path_key in seen_paths:
            raise ReleaseValidationError("exact-version proof contains a duplicate output path")
        seen_paths.add(path_key)
        files_by_basename.setdefault(basename, []).append(record)
    for basename, expected_hash, expected_bytes in (
        ("public_reference_run_manifest.json", manifest_sha256, manifest_bytes),
        ("submission.csv", submission_sha256, submission_bytes),
    ):
        records = files_by_basename.get(basename, [])
        if (
            len(records) != 1
            or records[0].get("sha256") != expected_hash
            or records[0].get("bytes") != expected_bytes
        ):
            raise ReleaseValidationError(f"exact-version proof inventory does not bind {basename}")
    return proof, expected_shapes


def _validate_package(
    package_dir: Path, identity: PackageIdentity
) -> tuple[dict[str, object], dict[str, object], str, str]:
    try:
        preflight_package(package_dir, identity)
    except (OSError, TypeError, ValueError, RehearsalError) as exc:
        raise ReleaseValidationError("reviewed E0 package preflight failed") from exc

    package_manifest, package_hash = _strict_json_object(package_dir / "package-manifest.json")
    artifact_lock, artifact_hash = _strict_json_object(package_dir / "artifact-lock.json")
    if package_hash != identity.package_manifest_sha256:
        raise ReleaseValidationError("package manifest bytes differ from the immutable identity")
    if artifact_hash != identity.artifact_lock_sha256:
        raise ReleaseValidationError("artifact lock bytes differ from the immutable identity")

    release_digest = artifact_lock.get("release_digest")
    lock_preimage = dict(artifact_lock)
    lock_preimage.pop("release_digest", None)
    if (
        release_digest != identity.release_digest
        or _canonical_digest(lock_preimage) != release_digest
    ):
        raise ReleaseValidationError("artifact lock release digest is invalid")
    if package_manifest.get("release_digest") != release_digest:
        raise ReleaseValidationError(
            "package manifest release digest differs from the artifact lock"
        )
    if package_manifest.get("artifact_lock_sha256") != artifact_hash:
        raise ReleaseValidationError("package manifest does not bind the artifact lock bytes")
    if package_manifest.get("algorithm_changes") != []:
        raise ReleaseValidationError("E0 reproduction package declares algorithm changes")

    reference = _require_mapping(artifact_lock.get("reference"), "artifact lock reference")
    source_notebook = _require_mapping(
        package_manifest.get("source_notebook"), "package source_notebook"
    )
    reference_notebook = _require_mapping(reference.get("notebook"), "reference notebook")
    for key, expected in reference_notebook.items():
        if source_notebook.get(key) != expected:
            raise ReleaseValidationError(f"package source notebook identity mismatch: {key}")
    if package_manifest.get("evidence_class") != reference.get("evidence_class"):
        raise ReleaseValidationError("package evidence class differs from the reference")

    lock_datasets = _require_mapping(artifact_lock.get("datasets"), "artifact lock datasets")
    package_datasets = _require_mapping(
        package_manifest.get("dataset_archives"), "package dataset_archives"
    )
    if set(package_datasets) != set(lock_datasets):
        raise ReleaseValidationError("package and artifact lock dataset sets differ")
    for ref, raw_lock_entry in lock_datasets.items():
        lock_entry = _require_mapping(raw_lock_entry, f"artifact lock dataset {ref}")
        package_entry = _require_mapping(package_datasets[ref], f"package dataset {ref}")
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
        ):
            if package_entry.get(key) != lock_entry.get(key):
                raise ReleaseValidationError(f"package dataset identity mismatch: {ref}:{key}")
    return package_manifest, artifact_lock, package_hash, artifact_hash


def _validate_shape_map(value: object, label: str) -> dict[str, tuple[int, int, int, int]]:
    mapping = _require_mapping(value, label)
    if not mapping:
        raise ReleaseValidationError(f"{label} must not be empty")
    result: dict[str, tuple[int, int, int, int]] = {}
    for dataset, raw_shape in mapping.items():
        if (
            not dataset
            or not isinstance(raw_shape, Sequence)
            or isinstance(raw_shape, (str, bytes))
        ):
            raise ReleaseValidationError(f"{label} has an invalid entry for {dataset!r}")
        shape = tuple(raw_shape)
        if len(shape) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
        ):
            raise ReleaseValidationError(f"{label}.{dataset} must be positive integer TZYX")
        result[dataset] = (shape[0], shape[1], shape[2], shape[3])
    return result


def _canonical_csv_int(raw: object, column: str, row_id: int) -> int:
    if not isinstance(raw, str):
        raise ReleaseValidationError(f"row {row_id} {column} is missing")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ReleaseValidationError(f"row {row_id} {column} must be an integer") from exc
    if str(value) != raw:
        raise ReleaseValidationError(f"row {row_id} {column} must be a canonical integer")
    return value


def _finish_dataset(
    dataset: str,
    shape: tuple[int, int, int, int],
    nodes: Mapping[int, int],
    edges: Sequence[tuple[int, int, int]],
) -> DatasetReleaseSummary:
    if not nodes:
        raise ReleaseValidationError(f"E0 dataset {dataset} contains no node rows")
    seen_edges: set[tuple[int, int]] = set()
    indegree: dict[int, int] = {}
    outdegree: dict[int, int] = {}
    children: dict[int, set[int]] = {}
    for source, target, row_id in edges:
        if source not in nodes or target not in nodes:
            raise ReleaseValidationError(f"row {row_id} has an endpoint outside dataset {dataset}")
        edge = (source, target)
        if edge in seen_edges:
            raise ReleaseValidationError(
                f"row {row_id} duplicates edge {dataset}:{source}->{target}"
            )
        seen_edges.add(edge)
        delta = nodes[target] - nodes[source]
        if delta != 1:
            raise ReleaseValidationError(
                f"row {row_id} violates E0 scorer-compatible frame gap: delta={delta}"
            )
        indegree[target] = indegree.get(target, 0) + 1
        outdegree[source] = outdegree.get(source, 0) + 1
        children.setdefault(source, set()).add(target)
        if indegree[target] > 1:
            raise ReleaseValidationError(
                f"E0 lineage merge is forbidden: {dataset} node {target} has multiple parents"
            )
        if outdegree[source] > 2:
            raise ReleaseValidationError(
                f"E0 lineage fork is invalid: {dataset} node {source} has more than two children"
            )
    forks = sum(degree == 2 for degree in outdegree.values())
    if any(len(children[source]) != 2 for source, degree in outdegree.items() if degree == 2):
        raise ReleaseValidationError(f"E0 division topology is malformed in {dataset}")
    return DatasetReleaseSummary(
        dataset=dataset,
        shape_tzyx=shape,
        nodes=len(nodes),
        edges=len(edges),
        forks=forks,
        max_indegree=max(indegree.values(), default=0),
        max_outdegree=max(outdegree.values(), default=0),
    )


def _parse_submission(path: Path, shapes: Mapping[str, tuple[int, int, int, int]]) -> _ParsedCSV:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as binary:
            for block in iter(lambda: binary.read(1024 * 1024), b""):
                digest.update(block)
            binary.seek(0)
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != SUBMISSION_COLUMNS:
                    raise ReleaseValidationError(
                        f"submission columns must be exactly {SUBMISSION_COLUMNS}"
                    )
                seen_datasets: set[str] = set()
                summaries: list[DatasetReleaseSummary] = []
                current: str | None = None
                nodes: dict[int, int] = {}
                edges: list[tuple[int, int, int]] = []
                row_count = 0
                for row in reader:
                    if None in row or any(value is None or value == "" for value in row.values()):
                        raise ReleaseValidationError(
                            f"row {row_count} has blank, missing, or extra fields"
                        )
                    row_id = _canonical_csv_int(row["id"], "id", row_count)
                    if row_id != row_count:
                        raise ReleaseValidationError(
                            f"submission id must equal row index {row_count}, got {row_id}"
                        )
                    dataset = row["dataset"]
                    if dataset not in shapes:
                        raise ReleaseValidationError(
                            f"row {row_id} has unexpected dataset {dataset!r}"
                        )
                    if dataset != current:
                        if current is not None:
                            summaries.append(
                                _finish_dataset(current, shapes[current], nodes, edges)
                            )
                        if dataset in seen_datasets:
                            raise ReleaseValidationError(
                                f"dataset rows are not contiguous: {dataset}"
                            )
                        seen_datasets.add(dataset)
                        current = dataset
                        nodes, edges = {}, []

                    values = {
                        column: _canonical_csv_int(row[column], column, row_id)
                        for column in SUBMISSION_COLUMNS
                        if column not in {"id", "dataset", "row_type"}
                    }
                    if row["row_type"] == "node":
                        if values["source_id"] != -1 or values["target_id"] != -1:
                            raise ReleaseValidationError(f"row {row_id} node endpoints must be -1")
                        node_id = values["node_id"]
                        coords = tuple(values[column] for column in ("t", "z", "y", "x"))
                        if node_id < 0 or any(
                            value < 0 or value >= limit
                            for value, limit in zip(coords, shapes[dataset], strict=True)
                        ):
                            raise ReleaseValidationError(
                                f"row {row_id} node id or TZYX coordinate is out of bounds"
                            )
                        if node_id in nodes:
                            raise ReleaseValidationError(
                                f"duplicate node id in dataset {dataset}: {node_id}"
                            )
                        nodes[node_id] = coords[0]
                    elif row["row_type"] == "edge":
                        if any(values[column] != -1 for column in ("node_id", "t", "z", "y", "x")):
                            raise ReleaseValidationError(
                                f"row {row_id} edge node fields must be -1"
                            )
                        source, target = values["source_id"], values["target_id"]
                        if source < 0 or target < 0:
                            raise ReleaseValidationError(
                                f"row {row_id} edge endpoints must be nonnegative"
                            )
                        edges.append((source, target, row_id))
                    else:
                        raise ReleaseValidationError(f"row {row_id} has invalid row_type")
                    row_count += 1
                if current is not None:
                    summaries.append(_finish_dataset(current, shapes[current], nodes, edges))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReleaseValidationError("cannot read strict submission CSV") from exc

    if not summaries:
        raise ReleaseValidationError("submission contains no datasets")
    actual = {summary.dataset for summary in summaries}
    if actual != set(shapes):
        raise ReleaseValidationError(
            f"dataset coverage mismatch: expected={sorted(shapes)}, actual={sorted(actual)}"
        )
    return _ParsedCSV(digest.hexdigest(), row_count, tuple(summaries))


def _validate_run_manifest(
    manifest: Mapping[str, object],
    *,
    artifact_lock: Mapping[str, object],
    identity: PackageIdentity,
    parsed: _ParsedCSV,
    expected_shapes: Mapping[str, tuple[int, int, int, int]],
) -> tuple[dict[str, tuple[int, int, int, int]], float]:
    exact = {
        "schema_version": 1,
        "status": "COMPLETE",
        "release_digest": identity.release_digest,
        "reference": artifact_lock["reference"],
        "coordinate_bounds_valid": True,
        "graph_valid": True,
        "hidden_discovery_supported": True,
        "internet_enabled_configured": False,
        "csv_sha256": parsed.sha256,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise ReleaseValidationError(f"run manifest violates exact {key} contract")

    shapes = _validate_shape_map(manifest.get("actual_input_shapes_tzyx"), "actual shapes")
    duplicate_shapes = _validate_shape_map(manifest.get("dataset_shapes"), "dataset shapes")
    if duplicate_shapes != shapes:
        raise ReleaseValidationError("run manifest shape maps disagree")
    if shapes != expected_shapes:
        raise ReleaseValidationError("run manifest shapes differ from exact-version input evidence")
    if shapes != {summary.dataset: summary.shape_tzyx for summary in parsed.datasets}:
        raise ReleaseValidationError("submission validation did not use the manifest TZYX shapes")
    expected_ids = sorted(shapes)
    if manifest.get("expected_dataset_ids") != expected_ids:
        raise ReleaseValidationError("run manifest expected dataset IDs differ from discovery")
    if manifest.get("completed_dataset_ids") != expected_ids:
        raise ReleaseValidationError("run manifest completed dataset IDs differ from discovery")

    lock_datasets = _require_mapping(artifact_lock.get("datasets"), "artifact lock datasets")
    integrity = _require_mapping(manifest.get("input_integrity"), "run input_integrity")
    if set(integrity) != set(lock_datasets):
        raise ReleaseValidationError("mounted input set differs from the artifact lock")
    roots: set[str] = set()
    for ref, raw_lock_entry in lock_datasets.items():
        lock_entry = _require_mapping(raw_lock_entry, f"artifact lock dataset {ref}")
        record = _require_mapping(integrity[ref], f"input integrity {ref}")
        files = _require_mapping(lock_entry.get("files"), f"artifact lock files {ref}")
        expected_bytes = 0
        for name, raw_file in files.items():
            file_record = _require_mapping(raw_file, f"artifact lock file {ref}:{name}")
            expected_bytes += _require_int(
                file_record.get("bytes"), f"artifact lock bytes {ref}:{name}"
            )
            if not _is_sha256(file_record.get("sha256")):
                raise ReleaseValidationError(f"artifact lock file hash is invalid: {ref}:{name}")
        root = record.get("root")
        if not isinstance(root, str) or not PurePosixPath(root).is_absolute():
            raise ReleaseValidationError(f"mounted input root is not absolute: {ref}")
        if PurePosixPath(root).name != ref.split("/", 1)[1] or root in roots:
            raise ReleaseValidationError(f"mounted input root identity mismatch: {ref}")
        roots.add(root)
        expected_record = {
            "status": "verified",
            "version": lock_entry.get("version"),
            "files": len(files),
            "bytes": expected_bytes,
        }
        for key, expected in expected_record.items():
            if record.get(key) != expected:
                raise ReleaseValidationError(f"mounted input identity mismatch: {ref}:{key}")

    submission = _require_mapping(manifest.get("submission"), "run submission")
    submission_path = submission.get("path")
    if (
        not isinstance(submission_path, str)
        or PurePosixPath(submission_path).name != "submission.csv"
    ):
        raise ReleaseValidationError("run manifest submission path is not submission.csv")
    if submission.get("sha256") != parsed.sha256 or submission.get("validation") != "PASS":
        raise ReleaseValidationError("run manifest submission hash or validation status is invalid")
    if submission.get("rows") != parsed.row_count:
        raise ReleaseValidationError("run manifest submission row count differs from the CSV")
    counts = _require_mapping(submission.get("datasets"), "run submission datasets")
    expected_counts = {
        summary.dataset: {"nodes": summary.nodes, "edges": summary.edges}
        for summary in parsed.datasets
    }
    if counts != expected_counts:
        raise ReleaseValidationError("run manifest dataset counts differ from the CSV")
    elapsed = _require_positive_finite(manifest.get("elapsed_seconds"), "elapsed_seconds")
    full_runtime = _require_positive_finite(
        manifest.get("full_runtime_seconds"), "full_runtime_seconds"
    )
    if elapsed != full_runtime:
        raise ReleaseValidationError("E0 manifest elapsed and full runtime disagree")
    return shapes, full_runtime


def _missing_paths(manifest: Mapping[str, object], paths: Sequence[str]) -> tuple[str, ...]:
    missing: list[str] = []
    for path in paths:
        value: object = manifest
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                missing.append(path)
                break
            value = value[part]
        else:
            if value is None:
                missing.append(path)
    return tuple(missing)


def validate_downloaded_e0_release(
    submission_csv: Path | str,
    run_manifest_json: Path | str,
    package_dir: Path | str,
    *,
    exact_version_proof: ExactVersionOutputProof | Mapping[str, object],
    package_identity: PackageIdentity = E0_R3_PACKAGE_IDENTITY,
    canonicalization_receipt: Path | str | None = None,
) -> ReleaseValidationResult:
    """Validate exact downloaded E0 bytes without trusting notebook assertions alone.

    Successful return proves identity and the strict E0 structural contract.  It
    deliberately does not imply scientific quality or resource admission when
    the upstream notebook omitted the necessary instrumentation.
    """

    submission_path = Path(submission_csv)
    manifest_path = Path(run_manifest_json)
    package_path = Path(package_dir)
    _, artifact_lock, package_hash, artifact_hash = _validate_package(
        package_path, package_identity
    )
    canonicalized_notebook = None
    if canonicalization_receipt is not None:
        from biohub_ct.campaign.kaggle_canonicalization import verify_canonicalization_receipt

        mapping = verify_canonicalization_receipt(
            Path(canonicalization_receipt), identity=package_identity
        )
        canonicalized_notebook = (mapping.actual_notebook_slug, mapping.notebook_version)
    manifest, manifest_hash = _strict_json_object(manifest_path)
    shapes = _validate_shape_map(manifest.get("actual_input_shapes_tzyx"), "actual shapes")
    parsed = _parse_submission(submission_path, shapes)
    proof, expected_shapes = _validate_proof(
        exact_version_proof,
        identity=package_identity,
        manifest_sha256=manifest_hash,
        submission_sha256=parsed.sha256,
        manifest_bytes=manifest_path.stat().st_size,
        submission_bytes=submission_path.stat().st_size,
        canonicalized_notebook=canonicalized_notebook,
    )
    _, full_runtime = _validate_run_manifest(
        manifest,
        artifact_lock=artifact_lock,
        identity=package_identity,
        parsed=parsed,
        expected_shapes=expected_shapes,
    )

    try:
        existing = validate_submission(
            submission_path,
            expected_datasets=sorted(shapes),
            shapes=shapes,
        )
    except SubmissionError as exc:
        raise ReleaseValidationError(
            "existing project submission validator rejected the CSV"
        ) from exc
    if existing.row_count != parsed.row_count:
        raise ReleaseValidationError("independent validators disagree on row count")
    if _sha256_file(submission_path) != parsed.sha256:
        raise ReleaseValidationError("submission changed during validation")
    if _sha256_file(manifest_path) != manifest_hash:
        raise ReleaseValidationError("run manifest changed during validation")

    missing_quality = _missing_paths(manifest, _QUALITY_EVIDENCE_PATHS)
    missing_resource = _missing_paths(manifest, _RESOURCE_EVIDENCE_PATHS)
    return ReleaseValidationResult(
        schema_version=1,
        status="STRUCTURAL_AND_IDENTITY_PASS_ADMISSION_BLOCKED",
        release_digest=package_identity.release_digest,
        notebook_slug=proof.notebook_slug,
        notebook_version=proof.notebook_version,
        submission_sha256=parsed.sha256,
        run_manifest_sha256=manifest_hash,
        package_manifest_sha256=package_hash,
        artifact_lock_sha256=artifact_hash,
        row_count=parsed.row_count,
        datasets=parsed.datasets,
        official_format_status="PASS",
        scorer_compatibility_status="PASS",
        e0_lineage_contract_status="PASS",
        identity_status="PASS",
        quality_admission=(
            "BLOCKED_MISSING_UPSTREAM_EVIDENCE" if missing_quality else "NOT_EVALUATED"
        ),
        resource_admission=(
            "BLOCKED_MISSING_UPSTREAM_EVIDENCE" if missing_resource else "NOT_EVALUATED"
        ),
        missing_quality_evidence=missing_quality,
        missing_resource_evidence=missing_resource,
        full_runtime_seconds=full_runtime,
    )


__all__ = [
    "DatasetReleaseSummary",
    "ExactVersionOutputProof",
    "ReleaseValidationError",
    "ReleaseValidationResult",
    "validate_downloaded_e0_release",
]
