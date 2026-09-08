from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from biohub_ct.campaign import (
    ApprovalError,
    CampaignStore,
    ContractError,
    DispatchError,
    DuplicateIntentError,
    JobState,
    ReviewLockBusy,
    StateTransitionError,
    canonical_json,
    run_spec_digest,
)

DIGESTS = [f"{index:x}" * 64 for index in range(1, 10)]


def run_spec(run_id: str = "run-1") -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": "E1",
        "hypothesis": "bounded diagnostic",
        "control_run_id": None,
        "changed_factors": ["precision"],
        "work_kind": "inference",
        "source": {
            "git_commit": "1" * 40,
            "source_bundle_sha256": DIGESTS[0],
            "dependency_manifest_sha256": DIGESTS[1],
            "effective_config_sha256": DIGESTS[2],
        },
        "model": {"weight_sha256": DIGESTS[3], "training_membership": "unknown"},
        "data": {
            "input_manifest_sha256": DIGESTS[4],
            "split_manifest_sha256": DIGESTS[5],
            "training_ids": ["6bba"],
            "development_ids": ["6bba-dev"],
            "evaluation_ids": [],
            "evidence_class": "previously_inspected_diagnostic",
        },
        "execution": {
            "host": "existing-vast-4070-super",
            "provider_instance_id": "existing-allocation",
            "working_directory": "/workspace/biohub-cell-tracking",
            "argv": ["python", "worker.py", "--run-id", run_id],
            "nonsecret_environment": {"PYTHONHASHSEED": "0"},
            "gpu_devices": [0],
            "random_seed": 7,
            "max_wall_seconds": 600,
            "max_steps_or_clips": 8,
            "max_incremental_cost": "1.00",
            "max_quota_hours": None,
            "deadline_utc": "2030-09-08T03:00:00Z",
            "checkpoint_interval_seconds": 60,
            "worker_progress_path": "reports/campaigns/run-1/heartbeat.json",
        },
        "stop_rules": ["deadline", "nonfinite output"],
        "success_rules": ["all eight clips complete"],
        "expected_artifacts": ["metrics.json"],
        "state": "PROPOSED",
        "reviewer_decision_id": None,
        "run_spec_sha256": None,
    }


def reservations(reservation_id: str = "reserve-1") -> list[dict]:
    return [{"ledger_id": "vast-hours", "reservation_id": reservation_id, "amount": "1.5"}]


def ready_store(tmp_path) -> CampaignStore:
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    store.initialize_campaign(
        "biohub-2026-09-08",
        "biohub-cell-tracking-during-development",
        status="ACTIVE",
        authorization={
            "routine_runs_and_submissions": False,
            "existing_allocation_authorized": True,
            "source": "explicit user authorization for the bounded campaign plan",
        },
        prior_campaign_refs=[
            {"campaign_id": "longrun-20260906-02", "ledger_status": "exhausted"}
        ],
    )
    store.create_ledger(
        "vast-hours",
        resource_scope="existing Vast RTX 4070 SUPER allocation",
        unit="instance_hours",
        authorized_total="4",
    )
    return store


def candidate(candidate_id: str = "candidate-1") -> dict:
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "identity": {
            "source_sha256": DIGESTS[0],
            "weight_sha256": DIGESTS[1],
            "dependency_manifest_sha256": DIGESTS[2],
            "effective_config_sha256": DIGESTS[3],
            "input_contract_sha256": DIGESTS[4],
        },
        "competition": "biohub-cell-tracking-during-development",
        "submission_class": "reproduction",
        "notebook_slug": "clarkkitchen/biohub-candidate-1",
        "notebook_version": 2,
        "notebook_complete_receipt": "receipts/notebook.json",
        "rehearsal_manifest_sha256": DIGESTS[5],
        "visible_csv_sha256": DIGESTS[6],
        "output_filename": "submission.csv",
        "eligibility_evidence_path": "evidence/eligibility.json",
    }


def ready_candidate(store: CampaignStore) -> dict:
    registered = store.register_candidate(candidate())
    for state in ("FROZEN", "REHEARSAL_RUNNING", "REHEARSAL_COMPLETE", "OUTPUT_VALIDATED"):
        store.transition_release("candidate-1", state, evidence={"checked": True})
    return registered


def test_canonical_hash_is_order_independent_and_rejects_nonfinite():
    assert canonical_json({"b": [2, 3], "a": 1}) == '{"a":1,"b":[2,3]}'
    assert run_spec_digest(run_spec()) == run_spec_digest(dict(reversed(list(run_spec().items()))))
    with pytest.raises(ContractError, match="finite"):
        canonical_json({"bad": float("nan")})


def test_campaign_defaults_deny_authorization_and_preserve_prior_reference(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    document = store.initialize_campaign(
        "fresh-campaign",
        "biohub-cell-tracking-during-development",
        prior_campaign_refs=[{"campaign_id": "old", "ledger_status": "exhausted"}],
    )
    assert document["authorization"]["routine_runs_and_submissions"] is False
    assert document["authorization"]["new_rental_purchase_authorized"] is False
    with pytest.raises(ContractError, match="append-only"):
        store.update_campaign(
            {"prior_campaign_refs": []}, reviewer="reviewer", reason="must not reset history"
        )


def test_legacy_prior_campaign_status_remains_valid_during_update(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    reference = {
        "campaign_id": "longrun-20260906-02",
        "status": "budget_exhausted",
        "remaining_seconds": 0,
        "terminal_receipt_path": "reports/longrun/terminal.json",
        "terminal_receipt_sha256": "a" * 64,
    }
    store.initialize_campaign(
        "fresh-campaign",
        "biohub-cell-tracking-during-development",
        prior_campaign_refs=[reference],
    )

    updated = store.update_campaign(
        {"status": "ACTIVE"}, reviewer="reviewer", reason="verified live inventory"
    )

    assert updated["status"] == "ACTIVE"
    assert updated["prior_campaign_refs"] == [reference]


def test_review_lock_is_exclusive_and_tokens_increase(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    with store.review_lock("reviewer-a") as first:
        assert first.fencing_token == 1
        with pytest.raises(ReviewLockBusy), store.review_lock("reviewer-b"):
            pass
    with store.review_lock("reviewer-b") as second:
        assert second.fencing_token == 2


def test_authorized_launch_claim_is_digest_bound_and_one_shot(tmp_path):
    store = ready_store(tmp_path)
    registered = store.register_run(run_spec())
    with store.review_lock("reviewer") as lease:
        intent = store.authorize_launch(
            "run-1",
            decision_id="decision-1",
            reviewer="hourly-helper",
            reservations=reservations(),
            reason="bounded run fits reconciled existing allocation",
            intent_id="launch-1",
            request_id="request-1",
            description_tag="biohub-run-1",
            fencing_token=lease.fencing_token,
        )
    altered = run_spec()
    altered["execution"]["max_wall_seconds"] = 601
    with pytest.raises(DispatchError, match="identity"):
        store.validate_dispatch(
            intent["intent_id"], "run-1", run_spec_digest(altered), intent["fencing_token"]
        )
    claimed = store.validate_dispatch(
        intent["intent_id"], "run-1", registered["run_spec_sha256"], intent["fencing_token"]
    )
    assert claimed["state"] == "DISPATCHED"
    with pytest.raises(DispatchError, match="DISPATCHED"):
        store.validate_dispatch(
            intent["intent_id"], "run-1", registered["run_spec_sha256"], intent["fencing_token"]
        )


def test_newer_review_fences_unclaimed_launch(tmp_path):
    store = ready_store(tmp_path)
    registered = store.register_run(run_spec())
    with store.review_lock("reviewer") as lease:
        intent = store.authorize_launch(
            "run-1", decision_id="decision-1", reviewer="hourly-helper",
            reservations=reservations(), reason="bounded", intent_id="launch-1",
            request_id="request-1", description_tag="biohub-run-1",
            fencing_token=lease.fencing_token,
        )
    with store.review_lock("next-review"):
        pass
    with pytest.raises(DispatchError, match="stale"):
        store.validate_dispatch(
            "launch-1", "run-1", registered["run_spec_sha256"], intent["fencing_token"]
        )
    with store.review_lock("cleanup-review") as cleanup:
        cancelled = store.cancel_pending_intent(
            "launch-1",
            fencing_token=cleanup.fencing_token,
            reason="newer review superseded an unclaimed launch",
        )
    assert cancelled["state"] == "ABSENT"
    assert store.budget_snapshot("vast-hours").outstanding_reservations == 0


def test_pending_intent_cannot_outlive_approval_window(tmp_path):
    store = ready_store(tmp_path)
    registered = store.register_run(run_spec())
    reviewed = datetime(2026, 9, 8, tzinfo=UTC)
    with store.review_lock("reviewer") as lease:
        intent = store.authorize_launch(
            "run-1", decision_id="decision-1", reviewer="hourly-helper",
            reservations=reservations(), reason="bounded", intent_id="launch-1",
            request_id="request-1", description_tag="biohub-run-1",
            fencing_token=lease.fencing_token, now=reviewed,
        )
    with pytest.raises(DispatchError, match="approval dispatch window"):
        store.validate_dispatch(
            "launch-1", "run-1", registered["run_spec_sha256"], intent["fencing_token"],
            now=reviewed + timedelta(minutes=60),
        )


def test_expired_approval_cannot_create_intent(tmp_path):
    store = ready_store(tmp_path)
    store.register_run(run_spec())
    reviewed = datetime(2026, 9, 8, tzinfo=UTC)
    with store.review_lock("reviewer") as lease:
        store.approve_run(
            "run-1",
            decision_id="decision-1",
            reviewer="hourly-helper",
            reservations=reservations(),
            reason="bounded",
            fencing_token=lease.fencing_token,
            now=reviewed,
        )
        with pytest.raises(ApprovalError, match="expired"):
            store.create_launch_intent(
                "run-1",
                decision_id="decision-1",
                intent_id="launch-1",
                request_id="request-1",
                description_tag="biohub-run-1",
                fencing_token=lease.fencing_token,
                now=reviewed + timedelta(minutes=60, microseconds=1),
            )
    assert store.get_approval("decision-1")["consumed_at"] is None
    assert store.budget_snapshot("vast-hours").outstanding_reservations == 1.5
    assert store.expire_approvals(now=reviewed + timedelta(minutes=60)) == ["decision-1"]
    assert store.get_approval("decision-1")["superseded_by"] == "EXPIRED"
    assert store.budget_snapshot("vast-hours").outstanding_reservations == 0


def test_float_work_unit_bound_fails_before_reservation(tmp_path):
    store = ready_store(tmp_path)
    invalid = run_spec()
    invalid["execution"]["max_steps_or_clips"] = 8.0
    store.register_run(invalid)
    with store.review_lock("reviewer") as lease, pytest.raises(
        ContractError, match="positive integer"
    ):
        store.approve_run(
            "run-1", decision_id="decision-1", reviewer="hourly-helper",
            reservations=reservations(), reason="must fail",
            fencing_token=lease.fencing_token,
        )
    assert store.budget_snapshot("vast-hours").outstanding_reservations == 0


def test_duplicate_request_id_rolls_back_second_authorization(tmp_path):
    store = ready_store(tmp_path)
    store.register_run(run_spec("run-1"))
    store.register_run(run_spec("run-2"))
    with store.review_lock("reviewer") as lease:
        store.authorize_launch(
            "run-1", decision_id="decision-1", reviewer="hourly-helper",
            reservations=reservations("reserve-1"), reason="bounded",
            intent_id="launch-1", request_id="request-shared",
            description_tag="biohub-run-1", fencing_token=lease.fencing_token,
        )
        with pytest.raises(DuplicateIntentError):
            store.authorize_launch(
                "run-2", decision_id="decision-2", reviewer="hourly-helper",
                reservations=reservations("reserve-2"), reason="bounded",
                intent_id="launch-2", request_id="request-shared",
                description_tag="biohub-run-2", fencing_token=lease.fencing_token,
            )
    assert store.get_run("run-2")["state"] == JobState.PROPOSED
    assert store.budget_snapshot("vast-hours").outstanding_reservations == 1.5
    with pytest.raises(KeyError):
        store.get_approval("decision-2")


def test_specs_freeze_and_transitions_are_validated(tmp_path):
    store = ready_store(tmp_path)
    store.register_run(run_spec())
    changed = run_spec()
    changed["hypothesis"] = "a corrected draft"
    assert store.update_run_spec("run-1", changed)["spec"]["hypothesis"] == "a corrected draft"
    with store.review_lock("reviewer") as lease:
        store.approve_run(
            "run-1", decision_id="decision-1", reviewer="hourly-helper",
            reservations=reservations(), reason="bounded", fencing_token=lease.fencing_token,
        )
    with pytest.raises(ContractError, match="immutable"):
        store.update_run_spec("run-1", run_spec())
    with pytest.raises((StateTransitionError, DispatchError)):
        store.transition_job("run-1", JobState.COMPLETE)


def test_atomic_json_export_contains_append_only_events(tmp_path):
    store = ready_store(tmp_path)
    store.register_run(run_spec())
    exported = store.export_json(tmp_path / "campaign.json")
    payload = __import__("json").loads(exported.read_text())
    assert payload["campaign"]["campaign_id"] == "biohub-2026-09-08"
    assert payload["state_version"] == payload["events"][-1]["state_version_after"]
    assert payload["jobs"][0]["run_spec_sha256"] == run_spec_digest(run_spec())


def test_submission_intent_requires_positive_numeric_receipt_and_lists_state(tmp_path):
    store = ready_store(tmp_path)
    with store.review_lock("test-reviewer") as authorization_lease:
        store.update_campaign(
            {"authorization": {"routine_runs_and_submissions": True}},
            reviewer="test-reviewer",
            reason="explicit submission authorization fixture",
            fencing_token=authorization_lease.fencing_token,
        )
    store.create_ledger(
        "submission-slots", resource_scope="verified account daily allowance",
        unit="submission_slots", authorized_total=3,
    )
    registered = ready_candidate(store)
    with store.review_lock("reviewer") as lease:
        intent = store.authorize_submission(
            "candidate-1", decision_id="submission-decision-1", reviewer="hourly-helper",
            reservations=[{"ledger_id": "submission-slots",
                           "reservation_id": "submission-slot-1", "amount": 1}],
            reason="eligible frozen exact version", intent_id="submission-intent-1",
            request_id="submission-request-1", description_tag="biohub-candidate-1-v2",
            fencing_token=lease.fencing_token,
        )
    store.validate_submission_dispatch(
        intent["intent_id"], "candidate-1", registered["candidate_sha256"],
        intent["fencing_token"],
    )
    for invalid in (True, 0, -1, "abc", "01", "1.0"):
        with pytest.raises(DispatchError, match="positive integer"):
            store.confirm_submission(
                intent["intent_id"], submission_id=invalid, receipt={"accepted": True}
            )
    accepted = store.confirm_submission(
        intent["intent_id"], submission_id=56086172, receipt={"accepted": True}
    )
    assert accepted["external_id"] == "56086172"
    assert store.list_candidates(states=["ACCEPTED"])[0]["candidate_id"] == "candidate-1"
    assert store.list_intents(kind="submission", states=["CONFIRMED"])[0]["intent_id"] == intent["intent_id"]
    assert store.list_runs() == []
