"""One bounded controller iteration. Workers and scoring continue independently."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from biohub_ct.campaign.admission import (
    AdmissionError,
    admit_submission_slots,
    require_fresh,
    verify_release_artifacts,
)
from biohub_ct.campaign.budget import BudgetError
from biohub_ct.campaign.contracts import canonical_json
from biohub_ct.campaign.kaggle_cli import KaggleCLI, KaggleError
from biohub_ct.campaign.state import ApprovalError, CampaignStore, DispatchError
from biohub_ct.campaign.watchdog import atomic_json


@dataclass(frozen=True)
class ReviewConfig:
    owner: str
    artifact_root: Path
    receipt_root: Path
    mutation_enabled: bool = False
    max_iteration_seconds: int = 540


def _state_snapshot(store: CampaignStore, destination: Path) -> dict:
    store.export_json(destination)
    return json.loads(destination.read_text(encoding="utf-8"))


def _submission_decision(receipt) -> str:
    if receipt.status != "COMPLETE":
        return "PENDING_SCORE"
    if receipt.public_score is None and receipt.private_score is None:
        return "SCORE_UNAVAILABLE"
    return "SCORE_RECEIVED"


def _daily_submission_usage(snapshot, history, today) -> dict:
    """Union account history with unobserved intents without freeing unknown slots."""
    def utc_day(raw):
        try:
            value = datetime.fromisoformat(raw)
        except (TypeError, ValueError) as exc:
            raise AdmissionError("Submission accounting needs an exact timestamp") from exc
        # The installed Kaggle history CSV formats its UTC API date without a zone.
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).date()

    releases = {row["candidate_id"]: row["candidate"] for row in snapshot["releases"]}
    intents = [row for row in snapshot["intents"]
               if row["intent_kind"] == "submission" and row["state"] != "ABSENT"]
    intent_by_submission = {}
    for intent in intents:
        external_id = intent.get("external_id")
        if external_id is not None:
            key = str(external_id)
            if key in intent_by_submission:
                raise AdmissionError("One account submission is bound to multiple intents")
            intent_by_submission[key] = intent

    all_history_ids = set()
    total = exploratory = unclassified = 0
    for row in history:
        key = str(row.submission_id)
        if key in all_history_ids:
            raise AdmissionError("Duplicate account submission in daily accounting")
        all_history_ids.add(key)
        if utc_day(row.submitted_at_raw) != today:
            continue
        total += 1
        intent = intent_by_submission.get(key)
        candidate = releases.get(intent["subject_id"], {}) if intent else {}
        classification = candidate.get("submission_class")
        if classification == "validated_candidate":
            continue
        if classification == "public_reference_reproduction" and row.status == "COMPLETE" and (
            row.public_score is not None or row.private_score is not None
        ):
            continue
        exploratory += 1
        if classification not in {"public_reference_reproduction", "experimental_candidate"}:
            unclassified += 1

    for intent in intents:
        if str(intent.get("external_id")) in all_history_ids:
            continue
        if utc_day(intent["created_at"]) != today:
            continue
        total += 1
        classification = releases.get(intent["subject_id"], {}).get("submission_class")
        if classification != "validated_candidate":
            exploratory += 1
            if classification not in {"public_reference_reproduction", "experimental_candidate"}:
                unclassified += 1
    return {"campaign_today": total, "exploration_today": exploratory,
            "unclassified_counted_as_exploratory": unclassified}


def review_once(store: CampaignStore, kaggle: KaggleCLI, config: ReviewConfig,
                *, inventory: Callable[[], dict], release_id: str | None = None,
                launch_request: dict | None = None,
                download_completion: Callable[[dict, dict], tuple[Path, Path]] | None = None) -> dict:
    """Reconcile first and optionally submit one explicitly named frozen release.

    Read-only mode means no external mutations. Local observation/decision records
    are still durable. The inventory callback must use bounded service calls; the
    entrypoint additionally runs this iteration under a hard process deadline.
    """
    if not 1 <= config.max_iteration_seconds <= 540:
        raise ValueError("Review iteration must finish within nine minutes")
    started = time.monotonic()
    now = datetime.now(UTC)
    review_id = "review-" + now.strftime("%Y%m%dT%H%M%S%fZ")
    directory = config.receipt_root / review_id
    decisions = []

    def check_time():
        if time.monotonic() - started >= config.max_iteration_seconds:
            raise AdmissionError("Iteration time bound reached; defer further actions")

    with store.review_lock(config.owner) as lease:
        directory.mkdir(parents=True, exist_ok=False)
        campaign = store.get_campaign()
        snapshot = _state_snapshot(store, directory / "before.json")
        live = inventory()
        require_fresh(live.get("observed_at"), now=datetime.now(UTC), max_age_seconds=120)
        check_time()
        history = kaggle.submissions(campaign["competition"])
        limits = kaggle.submission_limits(campaign["competition"])
        quota = kaggle.gpu_quota()
        observations = {"provider": live, "submissions": [r.as_dict() for r in history],
                        "submission_limits": limits, "quota": quota}
        atomic_json(directory / "observations.json", observations)
        check_time()

        # A process interruption can leave DISPATCHED before UNKNOWN was written.
        # Treat it as uncertain, preserving its reservation and prohibiting retries.
        for intent in snapshot["intents"]:
            intent = store.get_intent(intent["intent_id"])
            if intent["state"] == "DISPATCHED":
                try:
                    intent = store.mark_intent_unknown(intent["intent_id"], receipt={"reason": "prior dispatch has no confirmed receipt"})
                except DispatchError:
                    intent = store.get_intent(intent["intent_id"])
            if intent["state"] != "UNKNOWN":
                continue
            if intent["intent_kind"] == "submission":
                matches = [r for r in history if r.description == intent["description_tag"]]
                if len(matches) == 1 and matches[0].filename == intent["payload"]["output_filename"]:
                    row = matches[0]
                    store.reconcile_submission(intent["intent_id"], outcome="ACCEPTED",
                        submission_id=row.submission_id, receipt=row.as_dict())
                    decisions.append({"decision": _submission_decision(row),
                                      "submission_id": row.submission_id, "status": row.status,
                                      "public_score": row.public_score})
                else:
                    decisions.append({"decision": "BLOCKED", "intent_id": intent["intent_id"],
                        "reason": "Submission is uncertain; no unique matching account receipt. Reservation retained."})
            else:
                run = store.get_run(intent["subject_id"])
                if run["spec"].get("execution", {}).get("host") == "kaggle":
                    decisions.append({
                        "decision": "DEFER_PROVIDER_RECONCILIATION",
                        "intent_id": intent["intent_id"],
                        "run_id": run["run_id"],
                        "reason": (
                            "Kaggle launch uncertainty requires its exact-version provider "
                            "operator; generic worker receipts were ignored and the reservation "
                            "retained"
                        ),
                    })
                    continue
                # The provider observation must contain an authenticated, exact
                # intent receipt. No process-name matching or guessed PID ownership.
                receipts = live.get("launch_receipts", {})
                receipt = receipts.get(intent["intent_id"])
                if receipt and receipt.get("run_spec_sha256") == intent["subject_digest"] and receipt.get("fencing_token") == intent["fencing_token"] and receipt.get("provider_job_id"):
                    store.reconcile_launch(intent["intent_id"], outcome="RUNNING",
                        provider_job_id=receipt["provider_job_id"], receipt=receipt)
                    decisions.append({"decision": "CONTINUE", "run_id": intent["subject_id"]})
                else:
                    decisions.append({"decision": "BLOCKED", "intent_id": intent["intent_id"],
                        "reason": "Launch remains uncertain; compute reservation retained."})

        recorded = campaign.get("submission_receipts", {})
        changes = {}
        tracked_ids = set(campaign.get("tracked_submission_ids", []))
        tracked_ids.update(int(i["external_id"]) for i in snapshot["intents"]
                           if i["intent_kind"] == "submission" and i.get("external_id"))
        for row in history:
            if row.submission_id in tracked_ids:
                value = {"status": row.status, "public_score": row.public_score,
                         "private_score": row.private_score, "filename": row.filename}
                if recorded.get(str(row.submission_id)) != value:
                    changes[str(row.submission_id)] = value
                    decisions.append({"decision": _submission_decision(row),
                                      "submission_id": row.submission_id, **value})
        if changes:
            store.update_campaign({"submission_receipts": changes}, reviewer=config.owner,
                reason="Refreshed tracked numeric submission IDs from authenticated account history",
                fencing_token=lease.fencing_token)

        for run in store.list_runs():
            if run["state"] != "RUNNING":
                continue
            if run["spec"].get("execution", {}).get("host") == "kaggle":
                decisions.append({
                    "decision": "DEFER_PROVIDER_RECONCILIATION",
                    "run_id": run["run_id"],
                    "reason": (
                        "Kaggle run state requires its exact-version provider operator and "
                        "terminal reconciler; generic worker evidence was ignored and the "
                        "reservation retained"
                    ),
                })
                continue
            completion = live.get("worker_completions", {}).get(run["run_id"])
            if completion is not None:
                if completion.get("run_spec_sha256") != run["run_spec_sha256"]:
                    decisions.append({"decision": "REJECT", "run_id": run["run_id"],
                                      "reason": "Worker completion digest does not match admitted run"})
                else:
                    if download_completion is None:
                        decisions.append({"decision": "REVIEW_ARTIFACTS", "run_id": run["run_id"],
                            "worker_status": completion.get("status"),
                            "reason": "Worker stopped; independent artifact download/verification is required before settlement"})
                    else:
                        try:
                            check_time()
                            receipt_path, artifact_root = download_completion(run["spec"], completion)
                            from biohub_ct.campaign.recovery import reconcile_completed_run
                            recovered = reconcile_completed_run(store, run["run_id"], receipt_path, artifact_root, settlements=None)
                            decisions.append({"decision": "RUN_FINALIZED", "run_id": run["run_id"],
                                **{key: value for key, value in recovered.items() if key != "run"}})
                        except (AdmissionError, OSError, subprocess.TimeoutExpired) as exc:
                            reason = str(exc) if isinstance(exc, AdmissionError) else (
                                f"Artifact retrieval failed ({type(exc).__name__}); reservation retained")
                            decisions.append({"decision": "BLOCKED", "run_id": run["run_id"], "reason": reason})
                continue
            launch = next((i for i in store.list_intents() if i["intent_kind"] == "launch" and i["subject_id"] == run["run_id"]), None)
            receipt = live.get("launch_receipts", {}).get(launch["intent_id"]) if launch else None
            heartbeat = live.get("worker_heartbeats", {}).get(run["run_id"], {})
            healthy = False
            if receipt and receipt.get("process_alive") is True and heartbeat.get("run_spec_sha256") == run["run_spec_sha256"] and launch and heartbeat.get("fencing_token") == launch["fencing_token"]:
                try:
                    require_fresh(heartbeat.get("observed_at"), now=datetime.now(UTC), max_age_seconds=90)
                    healthy = datetime.now(UTC) < datetime.fromisoformat(run["spec"]["execution"]["deadline_utc"])
                except (AdmissionError, ValueError):
                    healthy = False
            decisions.append({"decision": "CONTINUE" if healthy else "BLOCKED",
                "run_id": run["run_id"], "reason": "Verified supervisor remains alive within its independent deadline"
                if healthy else "Missing/stale bound heartbeat, expired deadline, or absent exact process identity; reservation retained"})

        if release_id is not None:
            check_time()
            snapshot = _state_snapshot(store, directory / "reconciled.json")
            try:
                _submit_candidate(store, kaggle, config, campaign, snapshot, history,
                                  limits, release_id, lease.fencing_token, decisions)
            except (AdmissionError, ApprovalError, BudgetError) as exc:
                decisions.append({"decision": "BLOCKED", "candidate_id": release_id, "reason": str(exc)})
        if launch_request is not None:
            check_time()
            try:
                if not config.mutation_enabled:
                    raise AdmissionError("Read-only iteration: launch was not attempted")
                from biohub_ct.campaign.dispatch import launch_existing_vast
                receipt = launch_existing_vast(store, launch_request["transport"], launch_request["spec"],
                    owner=config.owner, ledger_id=launch_request["ledger_id"],
                    reservation_amount=launch_request["reservation_amount"], receipts=config.receipt_root / "launches",
                    inventory=inventory, review_lease=lease)
                decisions.append({"decision": "APPROVE_RUN" if receipt["status"] == "DISPATCHED" else "BLOCKED",
                                  "run_id": launch_request["spec"]["run_id"], "receipt": receipt})
            except (AdmissionError, ApprovalError, BudgetError) as exc:
                decisions.append({"decision": "BLOCKED", "run_id": launch_request["spec"]["run_id"], "reason": str(exc)})
        if not decisions:
            decisions.append({"decision": "NO_CHANGE", "reason": "No new tracked evidence or explicitly eligible action"})
        after = _state_snapshot(store, directory / "after.json")
        result = {"review_id": review_id, "campaign_id": campaign["campaign_id"],
            "reviewed_at": now.isoformat(), "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "mutation_enabled": config.mutation_enabled, "fencing_token": lease.fencing_token,
            "decisions": decisions, "state_version": after["state_version"],
            "observation_sha256": hashlib.sha256(canonical_json(observations).encode()).hexdigest()}
        # Only a changed actionable decision warrants another user notification.
        stable = hashlib.sha256(canonical_json(decisions).encode()).hexdigest()
        result["notify"] = any(d["decision"] != "NO_CHANGE" for d in decisions) and stable != campaign.get("last_decision_sha256")
        store.update_campaign({"last_review_id": review_id, "last_decision_sha256": stable,
                               "last_review_receipt": str(directory / "decision.json")},
            reviewer=config.owner, reason="Persisted bounded reviewer decision", fencing_token=lease.fencing_token)
        atomic_json(directory / "decision.json", result)
    return result


def _submit_candidate(store, kaggle, config, campaign, snapshot, history, limits,
                      candidate_id, token, decisions):
    record = store.get_candidate(candidate_id)
    candidate = {**record["candidate"], "candidate_sha256": record["candidate_sha256"]}
    if not config.mutation_enabled:
        raise AdmissionError("Read-only iteration: submission was not attempted")
    authorization = campaign["authorization"]
    if campaign.get("stop_requested") is True:
        raise AdmissionError("Campaign stop flag is set")
    if authorization.get("routine_runs_and_submissions") is not True:
        raise AdmissionError("Campaign has no recorded authorization for routine submissions")
    if authorization.get("expires_at") and datetime.now(UTC) >= datetime.fromisoformat(authorization["expires_at"]):
        raise AdmissionError("Campaign authorization expired")
    if record["state"] != "OUTPUT_VALIDATED":
        raise AdmissionError("Candidate is not in OUTPUT_VALIDATED state")
    if any(i["state"] in {"PENDING", "DISPATCHED", "UNKNOWN"} for i in snapshot["intents"]):
        raise AdmissionError("An unresolved prior intent must be reconciled before a new submission")
    verified = verify_release_artifacts(candidate, config.artifact_root)
    require_fresh(limits["observed_at"], now=datetime.now(UTC), max_age_seconds=120)
    today = datetime.now(UTC).date()
    # Account for manual entries too. A confirmed intent and its history row are
    # one attempt; an unobserved intent retains its slot. An unscored first public
    # reference and unclassified manual entries consume exploratory allowance.
    usage = _daily_submission_usage(snapshot, history, today)
    admit_submission_slots(allowed_now=limits["num_allowed_now"], campaign_today=usage["campaign_today"],
        exploration_today=usage["exploration_today"], submission_class=candidate["submission_class"],
        final_release=candidate.get("final_release") is True)
    tag = "biohub-" + candidate["candidate_sha256"][:32]
    ledger_id = f"submission-slots-{today.isoformat()}"
    # Daily ledger must already be reconciled/created from actual account policy.
    try:
        store.budget_snapshot(ledger_id)
    except KeyError as exc:
        raise AdmissionError("Daily submission ledger has not been initialized and reconciled") from exc
    intent = store.authorize_submission(candidate_id, decision_id="decision-" + tag,
        reviewer=config.owner, reservations=[{"ledger_id": ledger_id, "reservation_id": "slot-" + tag, "amount": "1"}],
        reason="Exact frozen offline version and downloaded CSV independently verified",
        quality_class=candidate["submission_class"], evidence_paths=list(verified["verified_files"]),
        intent_id="submit-" + tag, request_id="request-" + tag, description_tag=tag, fencing_token=token)
    try:
        receipt = kaggle.submit_exact(competition=candidate["competition"], slug=candidate["notebook_slug"],
            version=candidate["notebook_version"], filename=candidate["output_filename"], unique_tag=tag,
            verify_persisted_intent=lambda: store.validate_submission_dispatch(intent["intent_id"], candidate_id,
                candidate["candidate_sha256"], token))
    except Exception as exc:  # noqa: BLE001 - external mutation boundary must persist uncertainty for any failure
        # Persist exception type only: transport errors may contain credentials.
        receipt = {"status": "RECONCILIATION_REQUIRED", "error_type": type(exc).__name__}
    current = store.get_intent(intent["intent_id"])
    if current["state"] == "PENDING":
        decisions.append({"decision": "BLOCKED", "candidate_id": candidate_id,
                          "intent_id": intent["intent_id"], "reason": "Dispatch was not claimed; no submission attempt confirmed"})
        return
    if current["state"] == "DISPATCHED":
        store.mark_intent_unknown(intent["intent_id"], receipt=receipt)
    if current["state"] in {"DISPATCHED", "UNKNOWN"}:
        try:
            matched = kaggle.reconcile_intent(candidate["competition"], tag, candidate["output_filename"])
        except (KaggleError, OSError, subprocess.TimeoutExpired):
            matched = None
        if matched is not None:
            store.reconcile_submission(intent["intent_id"], outcome="ACCEPTED",
                submission_id=matched.submission_id, receipt=matched.as_dict())
            decisions.append({"decision": _submission_decision(matched),
                "candidate_id": candidate_id, "submission_id": matched.submission_id,
                "status": matched.status, "public_score": matched.public_score})
            return
    decisions.append({"decision": "PENDING_SCORE", "candidate_id": candidate_id,
                      "intent_id": intent["intent_id"], "status": "SUBMISSION_UNKNOWN",
                      "reason": "One exact-version attempt persisted; reconcile numeric receipt before any retry"})
