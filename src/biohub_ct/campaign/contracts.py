"""Canonical identities and state-machine contracts for campaign operations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any


class ContractError(ValueError):
    """Raised when a persisted campaign record is unsafe or incomplete."""


class StateTransitionError(ContractError):
    """Raised when a state-machine transition is not allowed."""


class JobState(StrEnum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    LAUNCH_INTENT = "LAUNCH_INTENT"
    LAUNCH_UNKNOWN = "LAUNCH_UNKNOWN"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class ReleaseState(StrEnum):
    DRAFT = "DRAFT"
    FROZEN = "FROZEN"
    REHEARSAL_RUNNING = "REHEARSAL_RUNNING"
    REHEARSAL_COMPLETE = "REHEARSAL_COMPLETE"
    OUTPUT_VALIDATED = "OUTPUT_VALIDATED"
    APPROVED = "APPROVED"
    SUBMIT_INTENT = "SUBMIT_INTENT"
    SUBMISSION_UNKNOWN = "SUBMISSION_UNKNOWN"
    ACCEPTED = "ACCEPTED"
    SCORED = "SCORED"
    FAILED = "FAILED"


class IntentState(StrEnum):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    UNKNOWN = "UNKNOWN"
    CONFIRMED = "CONFIRMED"
    ABSENT = "ABSENT"
    FAILED = "FAILED"


JOB_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.PROPOSED: frozenset({JobState.APPROVED, JobState.STOPPED}),
    JobState.APPROVED: frozenset({JobState.LAUNCH_INTENT, JobState.STOPPED}),
    JobState.LAUNCH_INTENT: frozenset(
        {JobState.RUNNING, JobState.LAUNCH_UNKNOWN, JobState.FAILED, JobState.STOPPED}
    ),
    JobState.LAUNCH_UNKNOWN: frozenset(
        {JobState.RUNNING, JobState.FAILED, JobState.STOPPED}
    ),
    JobState.RUNNING: frozenset(
        {JobState.COMPLETE, JobState.PARTIAL, JobState.FAILED, JobState.STOPPED}
    ),
    JobState.COMPLETE: frozenset(),
    JobState.PARTIAL: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.STOPPED: frozenset(),
}

RELEASE_TRANSITIONS: dict[ReleaseState, frozenset[ReleaseState]] = {
    ReleaseState.DRAFT: frozenset({ReleaseState.FROZEN, ReleaseState.FAILED}),
    ReleaseState.FROZEN: frozenset(
        {ReleaseState.REHEARSAL_RUNNING, ReleaseState.FAILED}
    ),
    ReleaseState.REHEARSAL_RUNNING: frozenset(
        {ReleaseState.REHEARSAL_COMPLETE, ReleaseState.FAILED}
    ),
    ReleaseState.REHEARSAL_COMPLETE: frozenset(
        {ReleaseState.OUTPUT_VALIDATED, ReleaseState.FAILED}
    ),
    ReleaseState.OUTPUT_VALIDATED: frozenset(
        {ReleaseState.APPROVED, ReleaseState.FAILED}
    ),
    ReleaseState.APPROVED: frozenset(
        {ReleaseState.SUBMIT_INTENT, ReleaseState.FAILED}
    ),
    ReleaseState.SUBMIT_INTENT: frozenset(
        {ReleaseState.ACCEPTED, ReleaseState.SUBMISSION_UNKNOWN, ReleaseState.FAILED}
    ),
    ReleaseState.SUBMISSION_UNKNOWN: frozenset(
        {ReleaseState.ACCEPTED, ReleaseState.FAILED}
    ),
    ReleaseState.ACCEPTED: frozenset({ReleaseState.SCORED, ReleaseState.FAILED}),
    ReleaseState.SCORED: frozenset(),
    ReleaseState.FAILED: frozenset(),
}

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PLACEHOLDER = re.compile(r"\s*<[^>]+>\s*\Z")

# These are receipts produced by execution, not requested execution identity.
_MUTABLE_RUN_FIELDS = frozenset({"state", "reviewer_decision_id", "run_spec_sha256"})
_MUTABLE_EXECUTION_FIELDS = frozenset(
    {
        "provider_job_id",
        "pid",
        "process_start_time",
        "observed_at",
        "stage",
        "completed_units",
        "elapsed_wall_seconds",
        "last_checkpoint_path",
        "last_checkpoint_sha256",
        "error",
    }
)
_MUTABLE_CANDIDATE_FIELDS = frozenset(
    {
        "approval_decision_id",
        "intent_id",
        "intent_persisted_at",
        "submitted_at",
        "submission_id",
        "status",
        "last_status_checked_at",
        "public_score",
        "private_score",
        "score_receipt_path",
        "selected_for_final",
        "selection_receipt_path",
        "candidate_sha256",
    }
)
def _normalise_json(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{path} must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path} contains a non-string key")
            result[key] = _normalise_json(item, f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise ContractError(f"{path} is not a JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the sole canonical JSON encoding used by campaign digests."""

    return json.dumps(
        _normalise_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _without(mapping: Mapping[str, Any], excluded: frozenset[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in excluded}


def run_spec_preimage(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Select immutable run proposal fields, including all requested worker settings."""

    if not isinstance(spec, Mapping):
        raise ContractError("run specification must be an object")
    value = _without(spec, _MUTABLE_RUN_FIELDS)
    execution = value.get("execution")
    if isinstance(execution, Mapping):
        value["execution"] = _without(execution, _MUTABLE_EXECUTION_FIELDS)
    return value


def run_spec_digest(spec: Mapping[str, Any]) -> str:
    return sha256_digest(run_spec_preimage(spec))


def candidate_preimage(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ContractError("candidate must be an object")
    identity = candidate.get("identity")
    if not isinstance(identity, Mapping):
        # Drafts can still be stored and updated before freeze. Keeping an
        # explicit null identity makes their provisional digest deterministic.
        identity = None
    return {"schema_version": candidate.get("schema_version"), "identity": identity}


def candidate_document(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen release record, excluding mutable receipt/status fields."""
    if not isinstance(candidate, Mapping):
        raise ContractError("candidate must be an object")
    return _without(candidate, _MUTABLE_CANDIDATE_FIELDS)


def candidate_digest(candidate: Mapping[str, Any]) -> str:
    return sha256_digest(candidate_preimage(candidate))


def intent_digest(kind: str, payload: Mapping[str, Any]) -> str:
    if kind not in {"launch", "submission"}:
        raise ContractError(f"unsupported intent kind: {kind!r}")
    return sha256_digest({"kind": kind, "payload": payload})


def validate_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ContractError(f"{field} must be a resolved identifier")
    return value


def validate_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def require_resolved(value: Any, field: str) -> Any:
    if value is None:
        raise ContractError(f"{field} is unresolved")
    if isinstance(value, str) and (not value.strip() or _PLACEHOLDER.fullmatch(value)):
        raise ContractError(f"{field} is unresolved")
    return value


def _required_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = require_resolved(parent.get(key), key)
    if not isinstance(value, Mapping):
        raise ContractError(f"{key} must be an object")
    return value


def validate_run_spec(spec: Mapping[str, Any], *, for_approval: bool = False) -> None:
    """Validate a draft or the stricter action-ready run specification."""

    validate_identifier(spec.get("run_id"), "run_id")
    canonical_json(run_spec_preimage(spec))
    supplied = spec.get("run_spec_sha256")
    if supplied is not None and supplied != run_spec_digest(spec):
        raise ContractError("run_spec_sha256 does not match the immutable specification")
    if not for_approval:
        return

    require_resolved(spec.get("experiment_id"), "experiment_id")
    require_resolved(spec.get("work_kind"), "work_kind")
    source = _required_mapping(spec, "source")
    if source.get("git_commit") is None and source.get("source_bundle_sha256") is None:
        raise ContractError("source identity requires git_commit or source_bundle_sha256")
    if source.get("source_bundle_sha256") is not None:
        validate_sha256(source["source_bundle_sha256"], "source.source_bundle_sha256")
    validate_sha256(
        require_resolved(source.get("dependency_manifest_sha256"), "source.dependency_manifest_sha256"),
        "source.dependency_manifest_sha256",
    )
    validate_sha256(
        require_resolved(source.get("effective_config_sha256"), "source.effective_config_sha256"),
        "source.effective_config_sha256",
    )
    data = _required_mapping(spec, "data")
    validate_sha256(
        require_resolved(data.get("input_manifest_sha256"), "data.input_manifest_sha256"),
        "data.input_manifest_sha256",
    )
    validate_sha256(
        require_resolved(data.get("split_manifest_sha256"), "data.split_manifest_sha256"),
        "data.split_manifest_sha256",
    )
    execution = _required_mapping(spec, "execution")
    for field in ("host", "working_directory", "deadline_utc"):
        require_resolved(execution.get(field), f"execution.{field}")
    argv = execution.get("argv")
    if not isinstance(argv, list) or not argv:
        raise ContractError("execution.argv must be a non-empty argument array")
    for index, arg in enumerate(argv):
        require_resolved(arg, f"execution.argv[{index}]")
        if not isinstance(arg, str):
            raise ContractError(f"execution.argv[{index}] must be a string")
    number = execution.get("max_wall_seconds")
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise ContractError("execution.max_wall_seconds must be a positive finite number")
    if not math.isfinite(number) or number <= 0:
        raise ContractError("execution.max_wall_seconds must be a positive finite number")
    number = execution.get("max_steps_or_clips")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise ContractError("execution.max_steps_or_clips must be a positive integer")
    try:
        deadline = datetime.fromisoformat(str(execution["deadline_utc"]))
    except ValueError as exc:
        raise ContractError("execution.deadline_utc must be an ISO 8601 timestamp") from exc
    if (
        deadline.tzinfo is None
        or deadline.utcoffset() is None
        or deadline.utcoffset().total_seconds() != 0
    ):
        raise ContractError("execution.deadline_utc must be timezone-aware UTC")
    progress_path = execution.get("worker_progress_path")
    if progress_path is not None:
        require_resolved(progress_path, "execution.worker_progress_path")
        if not isinstance(progress_path, str):
            raise ContractError("execution.worker_progress_path must be a string")
    for field in ("stop_rules", "success_rules", "expected_artifacts"):
        value = spec.get(field)
        if not isinstance(value, list) or not value:
            raise ContractError(f"{field} must be a non-empty list")


def validate_candidate(candidate: Mapping[str, Any], *, for_approval: bool = False) -> None:
    validate_identifier(candidate.get("candidate_id"), "candidate_id")
    canonical_json(candidate_preimage(candidate))
    supplied = candidate.get("candidate_sha256")
    if supplied is not None and supplied != candidate_digest(candidate):
        raise ContractError("candidate_sha256 does not match the immutable candidate")
    if not for_approval:
        return
    validate_candidate_identity(candidate)
    for field in (
        "rehearsal_manifest_sha256",
        "visible_csv_sha256",
        "eligibility_evidence_path",
    ):
        require_resolved(candidate.get(field), field)
    validate_sha256(candidate["rehearsal_manifest_sha256"], "rehearsal_manifest_sha256")
    validate_sha256(candidate["visible_csv_sha256"], "visible_csv_sha256")


def validate_candidate_identity(candidate: Mapping[str, Any]) -> None:
    """Validate fields that must be fixed before a release enters FROZEN."""
    for field in ("competition", "submission_class", "notebook_slug", "output_filename"):
        require_resolved(candidate.get(field), field)
    version = candidate.get("notebook_version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ContractError("notebook_version must be a positive integer")
    identity = _required_mapping(candidate, "identity")
    for field in (
        "source_sha256",
        "weight_sha256",
        "dependency_manifest_sha256",
        "effective_config_sha256",
        "input_contract_sha256",
    ):
        validate_sha256(require_resolved(identity.get(field), f"identity.{field}"), f"identity.{field}")


def validate_job_transition(current: str | JobState, target: str | JobState) -> None:
    before, after = JobState(current), JobState(target)
    if after not in JOB_TRANSITIONS[before]:
        raise StateTransitionError(f"invalid job transition: {before.value} -> {after.value}")


def validate_release_transition(current: str | ReleaseState, target: str | ReleaseState) -> None:
    before, after = ReleaseState(current), ReleaseState(target)
    if after not in RELEASE_TRANSITIONS[before]:
        raise StateTransitionError(f"invalid release transition: {before.value} -> {after.value}")


__all__ = [
    "JOB_TRANSITIONS",
    "RELEASE_TRANSITIONS",
    "ContractError",
    "IntentState",
    "JobState",
    "ReleaseState",
    "StateTransitionError",
    "candidate_digest",
    "candidate_document",
    "candidate_preimage",
    "canonical_json",
    "intent_digest",
    "require_resolved",
    "run_spec_digest",
    "run_spec_preimage",
    "sha256_digest",
    "validate_candidate",
    "validate_candidate_identity",
    "validate_identifier",
    "validate_job_transition",
    "validate_release_transition",
    "validate_run_spec",
    "validate_sha256",
]
