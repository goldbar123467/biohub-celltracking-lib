from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

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


def test_pending_to_complete_records_score_without_promotion(tmp_path):
    store, account, config = setup(tmp_path)
    review_once(store, account, config, inventory=inventory)
    account.rows = [SubmissionReceipt(123, "submission.csv", "2026-09-08T00:00:00Z", "prior", "COMPLETE", .9, None)]
    result = review_once(store, account, config, inventory=inventory)
    assert result["notify"]
    assert result["decisions"][0]["public_score"] == .9
    assert "incumbent" not in store.get_campaign()


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
