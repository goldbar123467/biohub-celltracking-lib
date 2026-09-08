import subprocess
from types import SimpleNamespace

import pytest

from biohub_ct.campaign.kaggle_cli import (
    KaggleCLI,
    KaggleError,
    SubmissionUnknown,
    parse_submissions,
)

HEADER = "ref,fileName,date,description,status,publicScore,privateScore\n"


def runner_for(output, *, code=0, stderr=""):
    def run(argv, **kwargs):
        assert kwargs["shell"] is False
        return SimpleNamespace(stdout=output, stderr=stderr, returncode=code)
    return run


def test_history_pending_scores_and_exact_id_not_latest():
    text = HEADER + "2,submission.csv,2026-09-08 00:10:00,new,SubmissionStatus.PENDING,,\n1,submission.csv,2026-09-07 20:10:00,old,SubmissionStatus.COMPLETE,1.01,\n"
    client = KaggleCLI(["/usr/bin/kaggle"], runner=runner_for(text))
    assert client.submission("test", 1).public_score == 1.01
    assert client.submission("test", 2).public_score is None
    assert client.submission("test", 100) is None


@pytest.mark.parametrize("output", [
    "usage: kaggle competitions\nerror: invalid choice", "{}",
    HEADER + "1,submission.csv,2026-09-07 20:10:00,old,SubmissionStatus.COMPLETE,NaN,\n",
    HEADER + "1,submission.csv,2026-09-07 20:10:00,old,SubmissionStatus.PENDING,0.9,\n",
])
def test_cli_success_exit_is_not_validated_receipt(output):
    with pytest.raises(KaggleError):
        KaggleCLI(["kaggle"], runner=runner_for(output)).submissions("test")


def test_duplicate_numeric_ids_and_invalid_timestamps_rejected():
    row = "1,submission.csv,2026-09-07 20:10:00,x,SubmissionStatus.COMPLETE,0.9,\n"
    with pytest.raises(KaggleError, match="duplicate"):
        parse_submissions(HEADER + row + row)
    with pytest.raises(KaggleError, match="timestamp"):
        parse_submissions(HEADER + row.replace("2026-09-07 20:10:00", "unknown"))


def test_live_allowance_not_inferred_from_lifetime_total():
    client = KaggleCLI(["kaggle"], runner=runner_for('{"numTotal": 200, "numAllowedNow": 5}'))
    assert client.submission_limits("test")["num_allowed_now"] == 5
    for invalid in ('{"numAllowedNow": null}', '{"numAllowedNow": true}', '{}'):
        with pytest.raises(KaggleError):
            KaggleCLI(["kaggle"], runner=runner_for(invalid)).submission_limits("test")


def test_quota_units_and_unknown_gpu_fail_closed():
    client = KaggleCLI(["kaggle"], runner=runner_for('[{"resource":"GPU","used":"0.18h","remaining":"29.82h","total":"30.00h","refreshAt":"2026-09-12T00:00:00"}]'))
    assert client.gpu_quota()["remaining_hours"] == 29.82
    with pytest.raises(KaggleError):
        KaggleCLI(["kaggle"], runner=runner_for('[]')).gpu_quota()


def test_timeout_after_submission_is_unknown_and_never_retried():
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 60)
    client = KaggleCLI(["kaggle"], runner=runner)
    checked = []
    with pytest.raises(SubmissionUnknown):
        client.submit_exact(competition="test", slug="owner/slug", version=3,
            filename="submission.csv", unique_tag="biohub-" + "a" * 16,
            verify_persisted_intent=lambda: checked.append(True))
    assert checked == [True]
    assert len(calls) == 1
    assert calls[0][calls[0].index("-v") + 1] == "3"


def test_intent_rejection_prevents_external_mutation():
    calls = []
    client = KaggleCLI(["kaggle"], runner=lambda *a, **k: calls.append(a))
    def reject():
        raise KaggleError("No persisted approval")
    with pytest.raises(KaggleError, match="persisted approval"):
        client.submit_exact(competition="test", slug="owner/slug", version=3,
            filename="submission.csv", unique_tag="biohub-" + "a" * 16,
            verify_persisted_intent=reject)
    assert calls == []


def test_reconcile_unique_tag_requires_single_matching_receipt():
    tag = "biohub-" + "a" * 16
    row = f"1,submission.csv,2026-09-08 00:10:00,{tag},SubmissionStatus.PENDING,,\n"
    client = KaggleCLI(["kaggle"], runner=runner_for(HEADER + row))
    assert client.reconcile_intent("test", tag, "submission.csv").submission_id == 1
    client = KaggleCLI(["kaggle"], runner=runner_for(HEADER))
    assert client.reconcile_intent("test", tag, "submission.csv") is None
    client = KaggleCLI(["kaggle"], runner=runner_for(HEADER + row + row.replace("1,submission", "2,submission")))
    with pytest.raises(SubmissionUnknown):
        client.reconcile_intent("test", tag, "submission.csv")
