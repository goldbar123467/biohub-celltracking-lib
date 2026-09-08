"""Validate downloaded worker receipts and finalize operational run state."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from biohub_ct.campaign.admission import (
    AdmissionError,
    decimal_nonnegative,
    file_sha256,
    verify_file_manifest,
)
from biohub_ct.campaign.contracts import JobState, validate_sha256
from biohub_ct.campaign.state import CampaignStore


class CompletionRecoveryError(AdmissionError):
    """Downloaded terminal evidence is incomplete, altered, or misbound."""


def _load_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise CompletionRecoveryError(f"{field} is not valid finite JSON") from exc
    if not isinstance(value, dict):
        raise CompletionRecoveryError(f"{field} must be a JSON object")
    return value


def _inside(root: Path, value: str | Path, field: str, *, strict: bool = True) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve(strict=strict)
    except OSError as exc:
        raise CompletionRecoveryError(f"{field} is missing") from exc
    if not resolved.is_relative_to(root):
        raise CompletionRecoveryError(f"{field} must stay inside the downloaded artifact root")
    return resolved


def _confirmed_launch_intent(store: CampaignStore, run_id: str) -> dict[str, Any]:
    matches = [
        intent
        for intent in store.list_intents(kind="launch")
        if intent["subject_id"] == run_id and intent["state"] == "CONFIRMED"
    ]
    if len(matches) != 1:
        raise CompletionRecoveryError("run requires one confirmed launch intent")
    return matches[0]


def _verify_complete_result(
    completion: Mapping[str, Any],
    spec: Mapping[str, Any],
    intent: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("complete") is not True:
        raise CompletionRecoveryError("COMPLETE lacks an operational artifact-gate receipt")
    if artifacts.get("quality_review_required") is not True:
        raise CompletionRecoveryError("COMPLETE must remain pending separate quality review")
    result_name = spec.get("result_manifest_path")
    if not isinstance(result_name, str) or not result_name:
        raise CompletionRecoveryError("run specification has no result manifest path")
    result_path = _inside(root, result_name, "result manifest")
    result_digest = artifacts.get("result_manifest_sha256")
    try:
        validate_sha256(result_digest, "result_manifest_sha256")
    except ValueError as exc:
        raise CompletionRecoveryError("result manifest digest is invalid") from exc
    if file_sha256(result_path) != result_digest:
        raise CompletionRecoveryError("result manifest hash mismatch")
    result = _load_object(result_path, "result manifest")
    result_token = result.get("fencing_token")
    if type(result_token) is not int or result_token <= 0:
        raise CompletionRecoveryError("result manifest has an invalid fencing token")
    identity = (
        result.get("run_id"),
        result.get("run_spec_sha256"),
        result.get("intent_id"),
        result.get("fencing_token"),
    )
    expected_identity = (
        spec["run_id"],
        intent["subject_digest"],
        intent["intent_id"],
        intent["fencing_token"],
    )
    if identity != expected_identity or result.get("status") != "COMPLETE":
        raise CompletionRecoveryError("result manifest is not bound to the exact launch intent")
    expected_units = spec.get("expected_completed_units")
    if isinstance(expected_units, bool) or not isinstance(expected_units, int) or expected_units <= 0:
        raise CompletionRecoveryError("run specification has no positive exact coverage bound")
    result_units = result.get("completed_units")
    artifact_units = artifacts.get("completed_units")
    if (
        type(result_units) is not int
        or type(artifact_units) is not int
        or result_units != expected_units
        or artifact_units != expected_units
        or expected_units > spec["execution"]["max_steps_or_clips"]
    ):
        raise CompletionRecoveryError("COMPLETE result does not cover the exact planned units")
    result_files = result.get("artifact_sha256")
    expected_files = spec.get("expected_artifacts")
    if not isinstance(result_files, dict) or not isinstance(expected_files, list):
        raise CompletionRecoveryError("COMPLETE result has no exact artifact manifest")
    if set(result_files) != set(expected_files):
        raise CompletionRecoveryError("COMPLETE result has missing or unexpected artifacts")
    if artifacts.get("artifact_sha256") != result_files:
        raise CompletionRecoveryError("completion and result artifact manifests differ")
    try:
        checked = verify_file_manifest(root, result_files)
    except AdmissionError as exc:
        raise CompletionRecoveryError(str(exc)) from exc
    return {
        "result_manifest_path": str(result_path),
        "result_manifest_sha256": result_digest,
        "artifact_sha256": checked,
        "completed_units": expected_units,
    }


def reconcile_completed_run(
    store: CampaignStore,
    run_id: str,
    completion_path: Path,
    artifact_root: Path,
    *,
    settlements: list[dict] | None,
) -> dict[str, Any]:
    """Validate one downloaded terminal receipt and atomically finalize its run.

    Instance-hour reservations can be settled from verified supervisor elapsed
    time. USD reservations remain active when authenticated actual billing was
    not supplied by the caller.
    """
    try:
        root = Path(artifact_root).resolve(strict=True)
    except OSError as exc:
        raise CompletionRecoveryError("downloaded artifact root is missing") from exc
    receipt_path = _inside(root, completion_path, "completion receipt")
    completion = _load_object(receipt_path, "completion receipt")
    run = store.get_run(run_id)
    # finalize_run performs the exact idempotency check after full evidence
    # validation; other non-running states still fail there.
    if run["state"] != JobState.RUNNING.value and run["state"] not in {
            JobState.COMPLETE.value,
            JobState.PARTIAL.value,
            JobState.FAILED.value,
            JobState.STOPPED.value,
    }:
        raise CompletionRecoveryError("only a running or identically finalized run can recover")
    intent = _confirmed_launch_intent(store, run_id)
    completion_token = completion.get("fencing_token")
    if type(completion_token) is not int or completion_token <= 0:
        raise CompletionRecoveryError("completion receipt has an invalid fencing token")
    identity = (
        completion.get("run_id"),
        completion.get("run_spec_sha256"),
        completion.get("fencing_token"),
    )
    expected_identity = (run_id, run["run_spec_sha256"], intent["fencing_token"])
    if identity != expected_identity:
        raise CompletionRecoveryError("completion receipt is not bound to the exact run and launch")
    status = completion.get("status")
    terminal = {"COMPLETE", "PARTIAL", "FAILED", "STOPPED"}
    if status not in terminal:
        raise CompletionRecoveryError("completion receipt is not operationally terminal")
    exit_code = completion.get("exit_code")
    failure = completion.get("error")
    failed_before_child = (
        status == "FAILED"
        and exit_code is None
        and isinstance(failure, Mapping)
        and completion.get("process_identity") is None
    )
    if (isinstance(exit_code, bool) or not isinstance(exit_code, int)) and not failed_before_child:
        raise CompletionRecoveryError("completion receipt has no integer exit status")
    stop_reason = completion.get("stop_reason")
    if status in {"COMPLETE", "PARTIAL"} and (
        exit_code != 0 or stop_reason is not None or failure is not None
    ):
        raise CompletionRecoveryError("successful terminal status contradicts process outcome")
    if status == "FAILED" and exit_code == 0 and stop_reason is None and failure is None:
        raise CompletionRecoveryError("FAILED receipt has no recorded failure")
    if status == "STOPPED" and stop_reason not in {"deadline", "work_unit_limit"}:
        raise CompletionRecoveryError("STOPPED receipt has no bounded stop reason")
    elapsed = decimal_nonnegative(completion.get("elapsed_wall_seconds"), "elapsed wall seconds")
    output_digest = completion.get("output_log_sha256")
    try:
        validate_sha256(output_digest, "output_log_sha256")
    except ValueError as exc:
        raise CompletionRecoveryError("output log digest is invalid") from exc
    output_path = _inside(root, receipt_path.parent / "output.log", "worker output log")
    if file_sha256(output_path) != output_digest:
        raise CompletionRecoveryError("worker output log hash mismatch")

    complete_evidence = None
    if status == "COMPLETE":
        complete_evidence = _verify_complete_result(completion, run["spec"], intent, root)
    elif completion.get("artifacts") is not None:
        raise CompletionRecoveryError("non-COMPLETE receipt cannot carry a complete artifact claim")

    completion_digest = file_sha256(receipt_path)
    evidence = {
        "completion_path": str(receipt_path),
        "completion_sha256": completion_digest,
        "output_log_path": str(output_path),
        "output_log_sha256": output_digest,
        "elapsed_wall_seconds": format(elapsed, "f"),
        "operational_status": status,
        "quality_review_required": status == "COMPLETE",
    }
    if complete_evidence is not None:
        evidence.update(complete_evidence)

    effective_settlements: Sequence[Mapping[str, Any]] | None = settlements
    reservations = store.list_intent_reservations(intent["intent_id"])
    active = [record for record in reservations if record["status"] == "ACTIVE"]
    if (
        settlements is None
        and len(active) == 1
        and active[0]["unit"] == "instance_hours"
    ):
        hours = elapsed / Decimal(3600)
        effective_settlements = [
            {"reservation_id": record["reservation_id"], "actual_amount": format(hours, "f")}
            for record in active
        ]
    elif settlements is not None:
        effective_settlements = list(settlements)

    finalized = store.finalize_run(
        run_id,
        status,
        evidence=evidence,
        settlements=effective_settlements,
        receipt_path=str(receipt_path),
    )
    result = {
        **finalized,
        "status": status,
        "completion_sha256": completion_digest,
        "evidence_path": str(receipt_path),
        "quality_review_required": status == "COMPLETE",
    }
    if finalized["settlement_needed"]:
        if any(record["unit"] == "USD" for record in finalized["retained_reservations"]):
            result["settlement_reason"] = "authenticated actual billing is required"
        else:
            result["settlement_reason"] = "explicit allocation across reservations is required"
    return result


__all__ = ["CompletionRecoveryError", "reconcile_completed_run"]
