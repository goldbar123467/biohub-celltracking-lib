"""Fail-closed operational admission, independent of model quality claims."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


class AdmissionError(ValueError):
    """A concrete requirement for an external action is missing."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decimal_nonnegative(value: object, field: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise AdmissionError(f"{field} must be known and finite")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise AdmissionError(f"{field} must be numeric") from exc
    if not number.is_finite() or number < 0:
        raise AdmissionError(f"{field} must be finite and nonnegative")
    return number


def require_fresh(observed_at: str, *, now: datetime, max_age_seconds: float) -> None:
    try:
        observed = datetime.fromisoformat(observed_at)
    except (TypeError, ValueError, AttributeError) as exc:
        raise AdmissionError("Observation needs a valid UTC timestamp") from exc
    if observed.tzinfo is None or observed.utcoffset().total_seconds() != 0:
        raise AdmissionError("Observation must use timezone-aware UTC")
    age = (now - observed).total_seconds()
    if age < -5 or age > max_age_seconds:
        raise AdmissionError("Resource observation is stale or from the future")


def verify_file_manifest(root: Path, manifest: Mapping[str, str]) -> dict[str, str]:
    """Verify the bytes used by a worker, rejecting paths outside its root."""
    root = root.resolve(strict=True)
    if not manifest:
        raise AdmissionError("An explicit nonempty file manifest is required")
    checked = {}
    for relative, expected in manifest.items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise AdmissionError("Manifest paths must stay inside the artifact root")
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise AdmissionError("Manifest path resolves outside the artifact root or is not a file")
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise AdmissionError("Manifest contains an invalid SHA-256")
        actual = file_sha256(path)
        if actual != expected:
            raise AdmissionError(f"Artifact hash mismatch: {relative}")
        checked[relative] = actual
    return checked


def verify_run_files(spec: Mapping, root: Path) -> dict:
    """The hash-bound specification names each actual identity-bearing file."""
    files = spec.get("verified_files")
    bindings = spec.get("identity_file_bindings")
    if not isinstance(files, dict) or not isinstance(bindings, dict):
        raise AdmissionError("Run requires verified_files and identity_file_bindings")
    required = {
        "source.source_bundle_sha256", "source.dependency_manifest_sha256",
        "source.effective_config_sha256", "data.input_manifest_sha256",
        "data.split_manifest_sha256",
    }
    if spec.get("model", {}).get("weight_sha256") is not None:
        required.add("model.weight_sha256")
    if set(bindings) != required:
        raise AdmissionError("Identity file bindings must cover the exact run identity")
    for field, relative in bindings.items():
        group, key = field.split(".")
        expected = spec.get(group, {}).get(key)
        if relative not in files or files[relative] != expected:
            raise AdmissionError(f"File manifest does not bind {field}")
    checked = verify_file_manifest(root, files)
    # Source bundle is a manifest of source files, not merely an unchecked label.
    source_manifest_path = root / bindings["source.source_bundle_sha256"]
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_files = verify_file_manifest(root, source_manifest)
    return {"verified_files": checked, "verified_source_files": source_files}


def release_quota_reserve(*, full_runtime_hours: object | None,
                          quota_hours_per_wall_hour: object,
                          verified_runtime_limit_hours: object,
                          attempts: int = 2) -> Decimal:
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 2:
        raise AdmissionError("At least two final release attempts must remain protected")
    limit = decimal_nonnegative(verified_runtime_limit_hours, "verified platform runtime limit")
    # Before a credible complete runtime exists, reserve the live verified limit.
    runtime = limit if full_runtime_hours is None else decimal_nonnegative(full_runtime_hours, "full runtime")
    rate = decimal_nonnegative(quota_hours_per_wall_hour, "measured quota rate")
    if limit <= 0 or runtime <= 0 or runtime > limit or rate <= 0:
        raise AdmissionError("Release runtime/rate is outside the supported envelope")
    return attempts * runtime * rate


def admit_resource_action(*, observation: Mapping, now: datetime,
                          reservation: object, protected_reserve: object,
                          active_gpu_jobs: int, is_release: bool = False) -> dict:
    require_fresh(observation.get("observed_at"), now=now, max_age_seconds=120)
    if isinstance(active_gpu_jobs, bool) or not isinstance(active_gpu_jobs, int) or active_gpu_jobs < 0:
        raise AdmissionError("Active job count must be verified")
    if active_gpu_jobs:
        raise AdmissionError("This platform already has a GPU job or unresolved launch")
    remaining = decimal_nonnegative(observation.get("remaining"), "remaining resource")
    requested = decimal_nonnegative(reservation, "worst-case reservation")
    protected = decimal_nonnegative(protected_reserve, "protected release reserve")
    if requested <= 0:
        raise AdmissionError("A positive worst-case reservation is required")
    available = remaining if is_release else remaining - protected
    if requested > available:
        raise AdmissionError("Worst-case work does not fit the remaining resource and release reserve")
    return {"reservation": str(requested), "remaining_after": str(remaining - requested),
            "protected_reserve": str(protected), "observed_at": observation["observed_at"]}


def admit_submission_slots(*, allowed_now: int, campaign_today: int,
                           exploration_today: int, submission_class: str,
                           final_release: bool = False) -> None:
    for value in (allowed_now, campaign_today, exploration_today):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AdmissionError("Submission counts must be known nonnegative integers")
    if submission_class not in {"public_reference_reproduction", "validated_candidate", "experimental_candidate"}:
        raise AdmissionError("Unknown submission evidence class")
    if (campaign_today >= 3 and not final_release) or allowed_now < 1:
        raise AdmissionError("Daily campaign or platform submission allowance exhausted")
    if not final_release and allowed_now <= 2:
        raise AdmissionError("Two final submission slots are protected")
    if (not final_release
            and submission_class in {"experimental_candidate", "public_reference_reproduction"}
            and exploration_today >= 2):
        raise AdmissionError("Daily exploratory submission limit exhausted")


def verify_release_artifacts(candidate: Mapping, root: Path) -> dict:
    """Validate frozen identity plus a downloaded, exact-version rehearsal receipt."""
    from biohub_ct.campaign.contracts import validate_candidate

    validate_candidate(candidate, for_approval=True)
    files = candidate.get("verified_files", {})
    checked = verify_file_manifest(root, files)
    reserve_reason = candidate.get("reserve_release_justification")
    if candidate.get("final_release") is True:
        if not isinstance(reserve_reason, Mapping):
            raise AdmissionError("Final reserve use requires a recorded justification")
        if reserve_reason.get("category") not in {
            "justified_repair", "superior_frozen_candidate", "deadline_recovery"
        }:
            raise AdmissionError("Final reserve justification has an unsupported category")
        rationale = reserve_reason.get("rationale")
        evidence_path = reserve_reason.get("evidence_path")
        if not isinstance(rationale, str) or not rationale.strip():
            raise AdmissionError("Final reserve justification must explain the decision")
        if not isinstance(evidence_path, str) or evidence_path not in checked:
            raise AdmissionError("Final reserve justification requires hash-verified evidence")
    manifest_name = candidate.get("rehearsal_manifest_path")
    csv_name = candidate.get("downloaded_csv_path")
    eligibility_name = candidate["eligibility_evidence_path"]
    for name, expected in ((manifest_name, candidate["rehearsal_manifest_sha256"]),
                           (csv_name, candidate["visible_csv_sha256"])):
        if name not in checked or checked[name] != expected:
            raise AdmissionError("Rehearsal output identity is not bound to verified files")
    if eligibility_name not in checked:
        raise AdmissionError("Eligibility evidence must be hash-verified")
    receipt = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    if receipt.get("notebook_slug") != candidate["notebook_slug"] or receipt.get("notebook_version") != candidate["notebook_version"]:
        raise AdmissionError("Rehearsal receipt belongs to a different notebook version")
    if receipt.get("identity") != candidate["identity"]:
        raise AdmissionError("Rehearsal identity differs from frozen release")
    if receipt.get("status") != "COMPLETE" or receipt.get("internet_enabled") is not False:
        raise AdmissionError("A completed offline rehearsal is required")
    if receipt.get("csv_sha256") != candidate["visible_csv_sha256"]:
        raise AdmissionError("Rehearsal CSV checksum differs from downloaded file")
    expected = receipt.get("expected_dataset_ids")
    if not isinstance(expected, list) or not expected or len(expected) != len(set(expected)):
        raise AdmissionError("Rehearsal requires explicit unique dataset coverage")
    if set(receipt.get("completed_dataset_ids", [])) != set(expected):
        raise AdmissionError("Rehearsal dataset coverage is incomplete")
    for field in ("coordinate_bounds_valid", "graph_valid", "hidden_discovery_supported"):
        if receipt.get(field) is not True:
            raise AdmissionError(f"Rehearsal missing {field}")
    if receipt.get("fallback_count") != 0:
        raise AdmissionError("Fallback outputs cannot satisfy the release evidence gate")
    runtime = decimal_nonnegative(receipt.get("full_runtime_seconds"), "full runtime")
    if runtime <= 0 or runtime > 43200:
        raise AdmissionError("Rehearsal runtime is outside the offline limit")
    eligibility = json.loads((root / eligibility_name).read_text(encoding="utf-8"))
    if eligibility.get("eligible") is not True or eligibility.get("identity") != candidate["identity"]:
        raise AdmissionError("Eligibility review does not bind this candidate")
    if (candidate["submission_class"] == "public_reference_reproduction"
            and eligibility.get("training_overlap") == "unknown"
            and eligibility.get("overlap_uncertainty_disclosed") is not True):
        raise AdmissionError("Reference overlap uncertainty must be disclosed")
    # Run the actual CSV validator; worker-authored booleans alone are insufficient.
    from biohub_ct.submission.validator import SubmissionError, validate_submission
    shapes = receipt.get("dataset_shapes")
    if not isinstance(shapes, dict) or set(shapes) != set(expected):
        raise AdmissionError("Exact dataset shapes are required for independent bounds validation")
    for shape in shapes.values():
        if not isinstance(shape, list) or len(shape) != 4 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape):
            raise AdmissionError("Dataset shapes must be positive integer TZYX arrays")
    try:
        validate_submission(root / csv_name, expected_datasets=expected, shapes=shapes)
    except SubmissionError as exc:
        raise AdmissionError(f"Downloaded CSV is invalid: {exc}") from exc
    return {"verified_files": checked, "rehearsal": receipt,
            "reserve_release_justification": reserve_reason}
