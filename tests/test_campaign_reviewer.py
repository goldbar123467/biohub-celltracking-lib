import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

import biohub_ct.campaign.reviewer as reviewer_module
from biohub_ct.campaign.kaggle_cli import SubmissionReceipt
from biohub_ct.campaign.reviewer import ReviewConfig, review_once
from biohub_ct.campaign.state import CampaignStore, ReviewLockBusy


class Account:
    def __init__(self):
        self.rows = [SubmissionReceipt(123, "submission.csv", "2026-09-08T00:00:00Z", "prior", "PENDING", None, None)]
        self.mutations = 0

    def submissions(self, competition):
        return self.rows

    def submission_limits(self, competition):
        return {"observed_at": datetime.now(UTC).isoformat(), "num_allowed_now": 5}

    def gpu_quota(self):
        return {"remaining_hours": 29.82, "observed_at": datetime.now(UTC).isoformat()}

    def submit_exact(self, **kwargs):
        self.mutations += 1
        raise AssertionError("Read-only iteration attempted a submission")


def setup(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    store.initialize_campaign("campaign-test", "biohub", fields={"tracked_submission_ids": [123]})
    account = Account()
    config = ReviewConfig("reviewer", tmp_path, tmp_path / "reviews")
    return store, account, config


def inventory():
    return {"observed_at": datetime.now(UTC).isoformat(), "launch_receipts": {}}


def test_read_only_records_numeric_status_once_and_stays_quiet_unchanged(tmp_path):
    store, account, config = setup(tmp_path)
    first = review_once(store, account, config, inventory=inventory)
    assert first["notify"]
    assert first["decisions"][0]["submission_id"] == 123
    second = review_once(store, account, config, inventory=inventory)
    assert not second["notify"]
    assert second["decisions"][0]["decision"] == "NO_CHANGE"
    assert account.mutations == 0
    receipt = Path(store.get_campaign()["last_review_receipt"])
    assert receipt.is_file()
    handoff = tmp_path / "handoff.md"
    text = handoff.read_text(encoding="utf-8")
    assert f"State version: `{second['state_version']}`" in text
    assert f"Decision receipt: `{receipt}`" in text
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() in text
    assert "## Approved action\n\n- None." in text


def test_pending_to_complete_records_score_without_promotion(tmp_path):
    store, account, config = setup(tmp_path)
    review_once(store, account, config, inventory=inventory)
    pending_handoff = (tmp_path / "handoff.md").read_text(encoding="utf-8")
    assert "PENDING_SCORE" in pending_handoff
    account.rows = [SubmissionReceipt(123, "submission.csv", "2026-09-08T00:00:00Z", "prior", "COMPLETE", .9, None)]
    result = review_once(store, account, config, inventory=inventory)
    assert result["notify"]
    assert result["decisions"][0]["public_score"] == .9
    assert "incumbent" not in store.get_campaign()
    current_handoff = (tmp_path / "handoff.md").read_text(encoding="utf-8")
    assert current_handoff != pending_handoff
    assert "SCORE_RECEIVED" in current_handoff
    assert '"public_score":0.9' in current_handoff
    assert "PENDING_SCORE" not in current_handoff


def test_second_reviewer_cannot_acquire_live_lock(tmp_path):
    store, account, config = setup(tmp_path)
    with store.review_lock("first"), pytest.raises(ReviewLockBusy):
        review_once(store, account, config, inventory=inventory)
    assert account.mutations == 0


def test_complete_without_numeric_score_is_unavailable_and_rechecked_later(tmp_path):
    store, account, config = setup(tmp_path)
    review_once(store, account, config, inventory=inventory)
    account.rows = [SubmissionReceipt(123, "submission.csv", "2026-09-08T00:00:00Z", "prior", "COMPLETE", None, None)]
    complete = review_once(store, account, config, inventory=inventory)
    assert complete["decisions"][0]["decision"] == "SCORE_UNAVAILABLE"
    assert complete["notify"]
    assert not review_once(store, account, config, inventory=inventory)["notify"]
    account.rows = [SubmissionReceipt(123, "submission.csv", "2026-09-08T00:00:00Z", "prior", "COMPLETE", 0.0, None)]
    scored = review_once(store, account, config, inventory=inventory)
    assert scored["decisions"][0]["decision"] == "SCORE_RECEIVED"
    assert scored["decisions"][0]["public_score"] == 0.0
    assert account.mutations == 0


def test_unchanged_blocker_not_repeated_as_notification(tmp_path):
    store, account, config = setup(tmp_path)
    # A draft is enough to prove mutation-disabled release requests remain local.
    store.register_candidate({"candidate_id": "draft"})
    review_once(store, account, config, inventory=inventory)
    first = review_once(store, account, config, inventory=inventory, release_id="draft")
    second = review_once(store, account, config, inventory=inventory, release_id="draft")
    assert first["notify"] and not second["notify"]
    assert account.mutations == 0


@pytest.mark.parametrize(
    ("unit", "worker_status", "confirmed", "outstanding"),
    [
        ("instance_hours", "STOPPED", Decimal("0.05"), Decimal(0)),
        ("USD", "COMPLETE", Decimal(0), Decimal(1)),
    ],
)
def test_review_downloads_and_recovers_terminal_run_without_invoice_claim(
    tmp_path, unit, worker_status, confirmed, outstanding
):
    from test_campaign_recovery import campaign_fixture

    store, artifact_root, completion_path, completion = campaign_fixture(
        tmp_path, unit=unit, status=worker_status
    )
    account = Account()
    config = ReviewConfig(
        "recovery-reviewer", artifact_root, tmp_path / "review-receipts"
    )
    downloads = []

    def download(spec, observed):
        downloads.append((spec["run_id"], observed["run_spec_sha256"]))
        assert observed == completion
        return completion_path, artifact_root

    result = review_once(
        store,
        account,
        config,
        inventory=lambda: {
            "observed_at": datetime.now(UTC).isoformat(),
            "launch_receipts": {},
            "worker_completions": {"run-1": completion},
        },
        download_completion=download,
    )

    recovered = next(row for row in result["decisions"] if row["decision"] == "RUN_FINALIZED")
    assert recovered["status"] == worker_status
    assert recovered["quality_review_required"] is (worker_status == "COMPLETE")
    assert store.get_run("run-1")["state"] == worker_status
    ledger = store.budget_snapshot("compute")
    assert ledger.confirmed_spend == confirmed
    assert ledger.outstanding_reservations == outstanding
    assert recovered["settlement_needed"] is (unit == "USD")
    assert "invoice" not in recovered
    assert store.list_candidates() == []
    assert account.mutations == 0
    assert len(downloads) == 1
    handoff = (store.path.parent / "handoff.md").read_text(encoding="utf-8")
    assert '"decision":"RUN_FINALIZED"' in handoff
    assert f'"status":"{worker_status}"' in handoff
    assert "## Active jobs and deadlines\n\n- None." in handoff
    assert f"confirmed/settled spend `{confirmed}` `{unit}`; active reserved `{outstanding}`" in handoff


def test_download_filesystem_failure_retains_run_and_reservation(tmp_path):
    from test_campaign_recovery import campaign_fixture

    store, artifact_root, _, completion = campaign_fixture(tmp_path)
    account = Account()

    def unavailable_destination(_spec, _completion):
        raise FileNotFoundError("Unusable Windows path")

    result = review_once(store, account,
        ReviewConfig("recovery-reviewer", artifact_root, tmp_path / "receipts"),
        inventory=lambda: {"observed_at": datetime.now(UTC).isoformat(),
            "launch_receipts": {}, "worker_completions": {"run-1": completion}},
        download_completion=unavailable_destination)
    assert result["decisions"][0]["decision"] == "BLOCKED"
    assert "FileNotFoundError" in result["decisions"][0]["reason"]
    assert store.get_run("run-1")["state"] == "RUNNING"
    assert store.budget_snapshot("compute").outstanding_reservations == Decimal(1)
    assert account.mutations == 0


def _kaggle_store(tmp_path, *, confirm: bool):
    from test_campaign_dispatch import ready_store, run_spec

    store = ready_store(tmp_path)
    spec = run_spec()
    spec["execution"]["host"] = "kaggle"
    spec["execution"].pop("provider_instance_id")
    spec["kaggle_rehearsal"] = {
        "notebook_slug": "owner/exact-version-rehearsal",
        "notebook_version": 1,
    }
    registered = store.register_run(spec)
    with store.review_lock("launch-reviewer") as lease:
        intent = store.authorize_launch(
            "run-1",
            decision_id="decision-run-1",
            reviewer="launch-reviewer",
            reservations=[{
                "ledger_id": "compute",
                "reservation_id": "compute-run-1",
                "amount": "1",
            }],
            reason="bounded provider-specific reviewer fixture",
            intent_id="launch-run-1",
            request_id="request-run-1",
            description_tag="run-1",
            fencing_token=lease.fencing_token,
        )
        store.validate_dispatch(
            intent["intent_id"],
            "run-1",
            registered["run_spec_sha256"],
            lease.fencing_token,
        )
        if confirm:
            store.confirm_launch(
                intent["intent_id"],
                provider_job_id="kaggle-kernel-owner/exact-version-rehearsal/1",
                receipt={"provider": "kaggle", "notebook_version": 1},
            )
        else:
            store.mark_intent_unknown(
                intent["intent_id"], receipt={"reason": "provider receipt unavailable"}
            )
    return store, registered, store.get_intent(intent["intent_id"])


def test_kaggle_run_defers_to_provider_reconciler_and_ignores_worker_evidence(tmp_path):
    store, registered, intent = _kaggle_store(tmp_path, confirm=True)
    account = Account()
    downloads = []
    forged_completion = {
        "run_id": "run-1",
        "run_spec_sha256": registered["run_spec_sha256"],
        "fencing_token": intent["fencing_token"],
        "status": "COMPLETE",
    }

    result = review_once(
        store,
        account,
        ReviewConfig("reviewer", tmp_path, tmp_path / "reviews"),
        inventory=lambda: {
            "observed_at": datetime.now(UTC).isoformat(),
            "launch_receipts": {
                intent["intent_id"]: {
                    "run_spec_sha256": registered["run_spec_sha256"],
                    "fencing_token": intent["fencing_token"],
                    "process_alive": True,
                },
            },
            "worker_heartbeats": {
                "run-1": {
                    "run_spec_sha256": registered["run_spec_sha256"],
                    "fencing_token": intent["fencing_token"],
                    "observed_at": datetime.now(UTC).isoformat(),
                },
            },
            "worker_completions": {"run-1": forged_completion},
        },
        download_completion=lambda *_args: downloads.append(_args),
    )

    decision = next(row for row in result["decisions"] if row["run_id"] == "run-1")
    assert decision["decision"] == "DEFER_PROVIDER_RECONCILIATION"
    assert "exact-version provider operator" in decision["reason"]
    assert downloads == []
    assert store.get_run("run-1")["state"] == "RUNNING"
    assert store.budget_snapshot("compute").outstanding_reservations == Decimal(1)
    assert account.mutations == 0


def test_unchanged_kaggle_provider_deferral_notifies_only_once(tmp_path):
    store, _registered, _intent = _kaggle_store(tmp_path, confirm=True)
    account = Account()
    config = ReviewConfig("reviewer", tmp_path, tmp_path / "reviews")
    provider_inventory = lambda: {
        "observed_at": datetime.now(UTC).isoformat(),
        "launch_receipts": {},
    }

    first = review_once(store, account, config, inventory=provider_inventory)
    second = review_once(store, account, config, inventory=provider_inventory)

    assert first["notify"] is True
    assert second["notify"] is False
    assert first["decisions"] == second["decisions"]
    assert store.get_run("run-1")["state"] == "RUNNING"
    assert store.budget_snapshot("compute").outstanding_reservations == Decimal(1)
    assert account.mutations == 0


def test_unknown_kaggle_launch_ignores_generic_worker_receipt(tmp_path):
    store, registered, intent = _kaggle_store(tmp_path, confirm=False)
    account = Account()

    result = review_once(
        store,
        account,
        ReviewConfig("reviewer", tmp_path, tmp_path / "reviews"),
        inventory=lambda: {
            "observed_at": datetime.now(UTC).isoformat(),
            "launch_receipts": {
                intent["intent_id"]: {
                    "run_spec_sha256": registered["run_spec_sha256"],
                    "fencing_token": intent["fencing_token"],
                    "provider_job_id": "linux-supervisor-forged",
                    "process_alive": True,
                },
            },
        },
    )

    decision = next(row for row in result["decisions"] if row.get("intent_id") == intent["intent_id"])
    assert decision["decision"] == "DEFER_PROVIDER_RECONCILIATION"
    assert "generic worker receipts were ignored" in decision["reason"]
    assert store.get_intent(intent["intent_id"])["state"] == "UNKNOWN"
    assert store.get_run("run-1")["state"] == "LAUNCH_UNKNOWN"
    assert store.budget_snapshot("compute").outstanding_reservations == Decimal(1)
    assert account.mutations == 0


def test_handoff_stays_with_store_when_receipts_are_isolated(tmp_path):
    controller_root = tmp_path / "controller"
    store, registered, intent = _kaggle_store(controller_root, confirm=False)
    account = Account()
    receipt_root = tmp_path / "isolated-receipts"

    result = review_once(
        store,
        account,
        ReviewConfig("reviewer", controller_root, receipt_root),
        inventory=lambda: {
            "observed_at": datetime.now(UTC).isoformat(),
            "launch_receipts": {},
        },
    )

    handoff = controller_root / "handoff.md"
    text = handoff.read_text(encoding="utf-8")
    assert not (receipt_root / "handoff.md").exists()
    assert f"Authoritative store: `{store.path}`" in text
    assert f"State version: `{result['state_version']}`" in text
    assert f"`{registered['run_id']}`: state `LAUNCH_UNKNOWN`" in text
    assert "deadline `2030-09-08T00:00:00Z`" in text
    assert f"`{intent['intent_id']}`: `launch`" in text
    assert "active reserved `1`" in text
    assert "At least one external intent is unresolved; retry is prohibited." in text
    assert account.mutations == 0


def test_handoff_path_cannot_overwrite_campaign_database(tmp_path):
    store = CampaignStore(tmp_path / "handoff.md")
    store.initialize_campaign("campaign-test", "biohub")
    account = Account()

    with pytest.raises(RuntimeError, match="conflicts with the campaign store or lock"):
        review_once(
            store,
            account,
            ReviewConfig("reviewer", tmp_path, tmp_path / "reviews"),
            inventory=inventory,
        )

    assert store.get_campaign()["campaign_id"] == "campaign-test"
    assert account.mutations == 0
    assert not (tmp_path / "reviews").exists()


def test_handoff_write_failure_is_not_silenced(tmp_path, monkeypatch):
    store, account, config = setup(tmp_path)

    def fail_write(_path, _text):
        raise OSError("handoff storage unavailable")

    monkeypatch.setattr(reviewer_module, "_atomic_text", fail_write)
    with pytest.raises(OSError, match="handoff storage unavailable"):
        review_once(store, account, config, inventory=inventory)

    assert account.mutations == 0
