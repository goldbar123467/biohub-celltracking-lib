"""Daily allocation includes manual entries, uncertain intents and the first reference."""

from datetime import date

import pytest

from biohub_ct.campaign.admission import AdmissionError, admit_submission_slots
from biohub_ct.campaign.kaggle_cli import SubmissionReceipt
from biohub_ct.campaign.reviewer import _daily_submission_usage

TODAY = date(2026, 9, 8)


def snapshot():
    return {
        "releases": [
            {
                "candidate_id": "reference",
                "candidate": {"submission_class": "public_reference_reproduction"},
            },
            {"candidate_id": "trial", "candidate": {"submission_class": "experimental_candidate"}},
            {"candidate_id": "validated", "candidate": {"submission_class": "validated_candidate"}},
        ],
        "intents": [],
    }


def intent(candidate_id, external_id, *, state="CONFIRMED", created_at="2026-09-08T00:00:00Z"):
    return {
        "intent_kind": "submission",
        "subject_id": candidate_id,
        "external_id": external_id,
        "state": state,
        "created_at": created_at,
    }


def receipt(submission_id, *, status="PENDING", score=None, timestamp="2026-09-08 00:01:00"):
    return SubmissionReceipt(
        submission_id, "submission.csv", timestamp, "test", status, score, None
    )


def test_first_reference_and_manual_submission_consume_two_exploratory_slots():
    state = snapshot()
    state["intents"] = [intent("reference", "1")]
    usage = _daily_submission_usage(state, [receipt(1), receipt(2)], TODAY)
    assert usage == {
        "campaign_today": 2,
        "exploration_today": 2,
        "unclassified_counted_as_exploratory": 1,
    }
    with pytest.raises(AdmissionError, match="exploratory"):
        admit_submission_slots(
            allowed_now=3,
            campaign_today=usage["campaign_today"],
            exploration_today=usage["exploration_today"],
            submission_class="experimental_candidate",
        )


def test_only_numeric_completed_reference_result_releases_its_exploratory_classification():
    state = snapshot()
    state["intents"] = [intent("reference", "1")]
    unscored = _daily_submission_usage(state, [receipt(1, status="COMPLETE")], TODAY)
    scored = _daily_submission_usage(state, [receipt(1, status="COMPLETE", score=0.0)], TODAY)
    assert unscored["exploration_today"] == 1
    assert scored["exploration_today"] == 0
    assert scored["campaign_today"] == 1


def test_account_history_and_uncertain_intents_are_unioned_without_double_counting():
    state = snapshot()
    state["intents"] = [
        intent("trial", "1"),
        intent("reference", None, state="UNKNOWN"),
        intent("validated", None, state="ABSENT"),
    ]
    usage = _daily_submission_usage(state, [receipt(1)], TODAY)
    assert usage["campaign_today"] == 2
    assert usage["exploration_today"] == 2


def test_actual_account_day_takes_precedence_over_intent_day_at_midnight():
    state = snapshot()
    state["intents"] = [intent("validated", "1", created_at="2026-09-07T23:59:59Z")]
    usage = _daily_submission_usage(state, [receipt(1)], TODAY)
    assert usage["campaign_today"] == 1
    assert usage["exploration_today"] == 0
    assert _daily_submission_usage(state, [receipt(1)], date(2026, 9, 7))["campaign_today"] == 0


def test_unparseable_timestamp_and_duplicate_receipt_cannot_free_daily_allowance():
    with pytest.raises(AdmissionError, match="timestamp"):
        _daily_submission_usage(snapshot(), [receipt(1, timestamp="unknown")], TODAY)
    with pytest.raises(AdmissionError, match="Duplicate"):
        _daily_submission_usage(snapshot(), [receipt(1), receipt(1)], TODAY)
