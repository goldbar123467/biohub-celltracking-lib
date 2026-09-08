from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from biohub_ct.campaign import (
    BudgetExceeded,
    BudgetLedger,
    CampaignStore,
    UnknownBudgetError,
)
from test_campaign_state import ready_store, reservations, run_spec


def test_unknown_numerical_authority_fails_closed(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    ledger = BudgetLedger.create(
        store,
        "vast-hours",
        resource_scope="authorized existing Vast allocation; numerical remainder unresolved",
        unit="instance_hours",
        authorized_total=None,
    )
    with pytest.raises(UnknownBudgetError, match="no numerical authorization"):
        ledger.reserve(
            "reserve-1", subject_kind="run", subject_id="run-1", amount="0.25"
        )
    assert ledger.snapshot().available is None


def test_separate_ledgers_do_not_substitute_for_each_other(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    rental = BudgetLedger.create(
        store, "rental-usd", resource_scope="existing rental", unit="USD",
        currency="USD", authorized_total="5",
    )
    kaggle = BudgetLedger.create(
        store, "kaggle-hours", resource_scope="Kaggle account quota", unit="quota_hours",
        authorized_total="2",
    )
    rental.reserve("rental-r", subject_kind="run", subject_id="run-1", amount="4")
    with pytest.raises(BudgetExceeded):
        rental.reserve("rental-r2", subject_kind="run", subject_id="run-2", amount="2")
    kaggle.reserve("kaggle-r", subject_kind="run", subject_id="run-2", amount="2")
    assert rental.snapshot().outstanding_reservations == Decimal(4)
    assert kaggle.snapshot().outstanding_reservations == Decimal(2)


def test_settlement_charges_actual_once_and_releases_unused_amount(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    ledger = BudgetLedger.create(
        store, "rental-usd", resource_scope="existing rental", unit="USD",
        currency="USD", authorized_total="100",
    )
    ledger.reserve("reserve-1", subject_kind="run", subject_id="run-1", amount="40")
    assert ledger.snapshot().available == Decimal(60)
    ledger.settle("reserve-1", "25")
    snapshot = ledger.snapshot()
    assert snapshot.confirmed_spend == Decimal(25)
    assert snapshot.outstanding_reservations == 0
    assert snapshot.available == Decimal(75)


def test_protected_reserve_requires_explicit_release_work(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    ledger = BudgetLedger.create(
        store, "kaggle-hours", resource_scope="Kaggle account quota", unit="quota_hours",
        authorized_total="10", protected_reserve="4",
    )
    with pytest.raises(BudgetExceeded):
        ledger.reserve("research", subject_kind="run", subject_id="run-1", amount="7")
    ledger.reserve(
        "release", subject_kind="release", subject_id="candidate-1", amount="7",
        allow_protected=True,
    )
    assert ledger.snapshot().outstanding_reservations == Decimal(7)


def test_concurrent_reservations_cannot_oversubscribe(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    ledger = BudgetLedger.create(
        store, "rental-usd", resource_scope="existing rental", unit="USD",
        currency="USD", authorized_total=10,
    )

    def reserve(index):
        try:
            ledger.reserve(
                f"reserve-{index}", subject_kind="run", subject_id=f"run-{index}", amount=7
            )
            return "reserved"
        except BudgetExceeded:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(reserve, (1, 2)))
    assert outcomes == ["rejected", "reserved"]
    assert ledger.snapshot().outstanding_reservations == Decimal(7)


def test_unknown_launch_receipt_retains_reservation(tmp_path):
    store = ready_store(tmp_path)
    registered = store.register_run(run_spec())
    with store.review_lock("reviewer") as lease:
        intent = store.authorize_launch(
            "run-1", decision_id="decision-1", reviewer="hourly-helper",
            reservations=reservations(), reason="bounded",
            intent_id="launch-1", request_id="request-1",
            description_tag="biohub-run-1", fencing_token=lease.fencing_token,
        )
    store.validate_dispatch(
        "launch-1", "run-1", registered["run_spec_sha256"], intent["fencing_token"]
    )
    unknown = store.mark_intent_unknown("launch-1", receipt={"timeout": True})
    assert unknown["state"] == "UNKNOWN"
    assert store.get_run("run-1")["state"] == "LAUNCH_UNKNOWN"
    assert store.budget_snapshot("vast-hours").outstanding_reservations == Decimal("1.5")
    store.reconcile_launch("launch-1", outcome="ABSENT", receipt={"matched": []})
    assert store.budget_snapshot("vast-hours").outstanding_reservations == 0
