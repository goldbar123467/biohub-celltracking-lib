import json
from decimal import Decimal
from pathlib import Path

import pytest

from biohub_ct.campaign.admission import file_sha256
from biohub_ct.campaign.recovery import CompletionRecoveryError, reconcile_completed_run
from biohub_ct.campaign.state import ReservationError
from test_campaign_dispatch import ready_store as ready_campaign_store
from test_campaign_dispatch import run_spec as campaign_run_spec


def recovery_plan(*args, **kwargs):
    pytest.importorskip("torch")
    from biohub_ct.training.recovery import recovery_plan as implementation

    return implementation(*args, **kwargs)


def fixture_run(tmp_path, *, prior=0, elapsed=360, terminal=True):
    root, jobs = tmp_path / "campaigns", tmp_path / "jobs"
    (root / "failed").mkdir(parents=True)
    (jobs / "failed").mkdir(parents=True)
    (root / "failed/manifest.json").write_text(json.dumps({"budget_spent_before_seconds": prior}))
    (root / "failed/status.json").write_text(
        json.dumps({"stage": "failed", "elapsed_seconds": elapsed})
    )
    (jobs / "failed/started_at.txt").write_text("2026-09-06T00:00:00Z")
    (jobs / "failed/finished_at.txt").write_text("2026-09-06T00:06:00Z")
    if terminal:
        (jobs / "failed/exit-code.txt").write_text("1")
    return root, jobs


def test_restart_budget_carries_prior_attempts_and_repair_reserve(tmp_path):
    root, jobs = fixture_run(tmp_path, prior=1200, elapsed=400)
    plan = recovery_plan("failed", root=root, jobs=jobs)
    assert plan["budget_spent_before_seconds"] == 2200
    assert plan["remaining_seconds"] == 24800


def test_launcher_duration_catches_stale_stage_elapsed(tmp_path):
    root, jobs = fixture_run(tmp_path, elapsed=6)
    plan = recovery_plan("failed", root=root, jobs=jobs)
    assert plan["parent_elapsed_seconds"] == 360


def test_running_parent_cannot_be_restarted(tmp_path):
    root, jobs = fixture_run(tmp_path, terminal=False)
    with pytest.raises(RuntimeError, match="terminal exit"):
        recovery_plan("failed", root=root, jobs=jobs)


def test_repeated_failures_cannot_reset_budget(tmp_path):
    root, jobs = fixture_run(tmp_path, prior=26000)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        recovery_plan("failed", root=root, jobs=jobs)


def test_repair_migration_allows_source_change_but_rejects_data_change(tmp_path):
    pytest.importorskip("torch")
    from dataclasses import replace

    import numpy as np

    from biohub_ct.training.checkpoint import load_checkpoint
    from biohub_ct.training.model import ModelConfig
    from biohub_ct.training.trainer import TrainConfig, train_detector

    class Sampler:
        train_ids, val_ids, dev_ids = ["44b6_fit"], ["6bba_val"], []

        def sample(self, step, batch_size):
            image = np.ones((batch_size, 1, 8, 8, 8), np.float32)
            return {"image": image, "target": image, "weight": image}

    config = TrainConfig(
        max_steps=1,
        max_seconds=60,
        device="cpu",
        amp=False,
        batch_size=1,
        checkpoint_every_steps=1,
        cpu_threads=1,
    )
    identity = {"source_digest": "old", "data_audit_sha256": "same-data"}
    train_detector(config, Sampler(), tmp_path / "old", ModelConfig(4), identity)
    state = load_checkpoint(tmp_path / "old/checkpoints")
    result = train_detector(
        replace(config, max_steps=2),
        Sampler(),
        tmp_path / "new",
        ModelConfig(4),
        {**identity, "source_digest": "repaired"},
        resume=tmp_path / "old/checkpoints",
        resume_expected_identity=state["identity"],
    )
    assert result["step"] == 2
    with pytest.raises(ValueError, match="data or split identity"):
        train_detector(
            replace(config, max_steps=2),
            Sampler(),
            tmp_path / "rejected",
            ModelConfig(4),
            {**identity, "data_audit_sha256": "changed"},
            resume=tmp_path / "old/checkpoints",
            resume_expected_identity=state["identity"],
        )


def test_recovery_copy_verifies_and_preserves_checkpoint_pointers(tmp_path):
    pytest.importorskip("torch")
    from biohub_ct.training.checkpoint import load_checkpoint, save_checkpoint
    from biohub_ct.training.recovery import copy_verified_checkpoints

    source = tmp_path / "source"
    save_checkpoint(source, {"step": 2, "marker": "best"}, is_best=True)
    save_checkpoint(source, {"step": 3, "marker": "latest"})
    copied = copy_verified_checkpoints(source, tmp_path / "copied")
    assert copied["marker"] == "latest"
    manifest = json.loads((tmp_path / "copied/manifest.json").read_text())
    assert load_checkpoint(tmp_path / "copied" / manifest["best"]["file"])["marker"] == "best"
    original_manifest = json.loads((source / "manifest.json").read_text())
    original_manifest["best"]["sha256"] = "0" * 64
    (source / "manifest.json").write_text(json.dumps(original_manifest))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        copy_verified_checkpoints(source, tmp_path / "rejected")


def campaign_fixture(
    tmp_path, *, unit="instance_hours", status="COMPLETE", reservation_count=1
):
    store = ready_campaign_store(tmp_path, unit=unit)
    spec = campaign_run_spec()
    attempt_name = "reports/campaign-workers/run-1"
    spec["expected_artifacts"] = [f"{attempt_name}/answer.txt"]
    spec["expected_completed_units"] = 1
    spec["result_manifest_path"] = f"{attempt_name}/result.json"
    registered = store.register_run(spec)
    with store.review_lock("launch-reviewer") as lease:
        intent = store.authorize_launch(
            "run-1",
            decision_id="decision-run-1",
            reviewer="launch-reviewer",
            reservations=[
                {
                    "ledger_id": "compute",
                    "reservation_id": f"compute-run-{index}",
                    "amount": "1",
                }
                for index in range(1, reservation_count + 1)
            ],
            reason="legitimate bounded recovery fixture",
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
        store.confirm_launch(
            intent["intent_id"], provider_job_id="linux-supervisor-123", receipt={}
        )

    root = tmp_path / "downloaded"
    attempt = root / attempt_name
    worker = attempt / "worker"
    worker.mkdir(parents=True)
    (worker / "output.log").write_text("bounded worker output\n")
    artifacts = None
    if status == "COMPLETE":
        (attempt / "answer.txt").write_text("140")
        result = {
            "run_id": "run-1",
            "run_spec_sha256": registered["run_spec_sha256"],
            "intent_id": intent["intent_id"],
            "fencing_token": intent["fencing_token"],
            "status": "COMPLETE",
            "completed_units": 1,
            "artifact_sha256": {
                f"{attempt_name}/answer.txt": file_sha256(attempt / "answer.txt")
            },
        }
        (attempt / "result.json").write_text(json.dumps(result))
        artifacts = {
            "complete": True,
            "result_manifest_sha256": file_sha256(attempt / "result.json"),
            "artifact_sha256": result["artifact_sha256"],
            "completed_units": 1,
            "quality_review_required": True,
        }
    completion = {
        "run_id": "run-1",
        "run_spec_sha256": registered["run_spec_sha256"],
        "fencing_token": intent["fencing_token"],
        "status": status,
        "exit_code": 0 if status in {"COMPLETE", "PARTIAL"} else 124,
        "stop_reason": "deadline" if status == "STOPPED" else None,
        "error": {"type": "WorkerError"} if status == "FAILED" else None,
        "elapsed_wall_seconds": 180,
        "artifacts": artifacts,
        "output_log_sha256": file_sha256(worker / "output.log"),
    }
    completion_path = worker / "completion.json"
    completion_path.write_text(json.dumps(completion))
    return store, root, completion_path, completion


def rewrite_completion(path: Path, completion: dict) -> None:
    path.write_text(json.dumps(completion))


def test_complete_recovery_verifies_artifacts_and_settles_instance_hours(tmp_path):
    store, root, path, _completion = campaign_fixture(tmp_path)

    result = reconcile_completed_run(store, "run-1", path, root, settlements=None)

    assert result["status"] == "COMPLETE"
    assert result["quality_review_required"] is True
    assert result["settlement_needed"] is False
    assert store.get_run("run-1")["state"] == "COMPLETE"
    ledger = store.budget_snapshot("compute")
    assert ledger.confirmed_spend == Decimal("0.05")
    assert ledger.outstanding_reservations == 0
    assert not store.list_candidates()
    event = [row for row in store.list_events() if row["event_type"] == "RUN_FINALIZED"][-1]
    assert event["receipt_path"] == str(path.resolve())
    assert event["payload"]["evidence"]["quality_review_required"] is True


@pytest.mark.parametrize("status", ["PARTIAL", "FAILED", "STOPPED"])
def test_interrupted_or_partial_run_is_terminal_without_complete_claim(tmp_path, status):
    store, root, path, _completion = campaign_fixture(tmp_path, status=status)

    result = reconcile_completed_run(store, "run-1", path, root, settlements=None)

    assert result["status"] == status
    assert result["quality_review_required"] is False
    assert store.get_run("run-1")["state"] == status
    assert store.budget_snapshot("compute").outstanding_reservations == 0


def test_failed_before_child_process_is_recoverable(tmp_path):
    store, root, path, completion = campaign_fixture(tmp_path, status="FAILED")
    completion["exit_code"] = None
    completion["process_identity"] = None
    rewrite_completion(path, completion)

    result = reconcile_completed_run(store, "run-1", path, root, settlements=None)

    assert result["status"] == "FAILED"
    assert store.get_run("run-1")["state"] == "FAILED"


def test_changed_log_hash_rolls_back_run_and_reservation(tmp_path):
    store, root, path, _completion = campaign_fixture(tmp_path)
    (path.parent / "output.log").write_text("changed after receipt")

    with pytest.raises(CompletionRecoveryError, match="log hash mismatch"):
        reconcile_completed_run(store, "run-1", path, root, settlements=None)

    assert store.get_run("run-1")["state"] == "RUNNING"
    assert store.budget_snapshot("compute").outstanding_reservations == 1


@pytest.mark.parametrize("field", ["intent_id", "fencing_token", "run_spec_sha256"])
def test_result_manifest_must_match_exact_launch_identity(tmp_path, field):
    store, root, path, completion = campaign_fixture(tmp_path)
    result_path = root / store.get_run("run-1")["spec"]["result_manifest_path"]
    result = json.loads(result_path.read_text())
    result[field] = "wrong" if field != "fencing_token" else result[field] + 1
    result_path.write_text(json.dumps(result))
    completion["artifacts"]["result_manifest_sha256"] = file_sha256(result_path)
    rewrite_completion(path, completion)

    with pytest.raises(CompletionRecoveryError, match="exact launch intent"):
        reconcile_completed_run(store, "run-1", path, root, settlements=None)

    assert store.get_run("run-1")["state"] == "RUNNING"


def test_complete_missing_exact_coverage_is_rejected(tmp_path):
    store, root, path, completion = campaign_fixture(tmp_path)
    result_path = root / store.get_run("run-1")["spec"]["result_manifest_path"]
    result = json.loads(result_path.read_text())
    result["completed_units"] = 0
    result_path.write_text(json.dumps(result))
    completion["artifacts"]["result_manifest_sha256"] = file_sha256(result_path)
    rewrite_completion(path, completion)

    with pytest.raises(CompletionRecoveryError, match="exact planned units"):
        reconcile_completed_run(store, "run-1", path, root, settlements=None)

    assert store.get_run("run-1")["state"] == "RUNNING"
    assert store.budget_snapshot("compute").outstanding_reservations == 1


@pytest.mark.parametrize("location", ["result", "artifacts"])
def test_boolean_completed_units_cannot_alias_integer_coverage(tmp_path, location):
    store, root, path, completion = campaign_fixture(tmp_path)
    result_path = root / store.get_run("run-1")["spec"]["result_manifest_path"]
    if location == "result":
        result = json.loads(result_path.read_text())
        result["completed_units"] = True
        result_path.write_text(json.dumps(result))
        completion["artifacts"]["result_manifest_sha256"] = file_sha256(result_path)
    else:
        completion["artifacts"]["completed_units"] = True
    rewrite_completion(path, completion)

    with pytest.raises(CompletionRecoveryError, match="planned units"):
        reconcile_completed_run(store, "run-1", path, root, settlements=None)

    assert store.get_run("run-1")["state"] == "RUNNING"


def test_unknown_usd_billing_finalizes_but_retains_reservation(tmp_path):
    store, root, path, _completion = campaign_fixture(tmp_path, unit="USD")

    result = reconcile_completed_run(store, "run-1", path, root, settlements=None)

    assert result["status"] == "COMPLETE"
    assert result["settlement_needed"] is True
    assert result["settlement_reason"] == "authenticated actual billing is required"
    assert result["retained_reservations"][0]["unit"] == "USD"
    ledger = store.budget_snapshot("compute")
    assert ledger.confirmed_spend == 0
    assert ledger.outstanding_reservations == 1


def test_multiple_instance_hour_reservations_require_explicit_allocation(tmp_path):
    store, root, path, _completion = campaign_fixture(tmp_path, reservation_count=2)

    result = reconcile_completed_run(store, "run-1", path, root, settlements=None)

    assert result["settlement_needed"] is True
    assert result["settlement_reason"] == "explicit allocation across reservations is required"
    assert store.budget_snapshot("compute").outstanding_reservations == 2


def test_known_usd_billing_settles_with_terminal_state_atomically(tmp_path):
    store, root, path, _completion = campaign_fixture(tmp_path, unit="USD")

    result = reconcile_completed_run(
        store,
        "run-1",
        path,
        root,
        settlements=[{"reservation_id": "compute-run-1", "actual_amount": "0.04"}],
    )

    assert result["settlement_needed"] is False
    ledger = store.budget_snapshot("compute")
    assert ledger.confirmed_spend == Decimal("0.04")
    assert ledger.outstanding_reservations == 0
    assert store.get_run("run-1")["state"] == "COMPLETE"


def test_invalid_known_settlement_rolls_back_terminal_transition(tmp_path):
    store, root, path, _completion = campaign_fixture(tmp_path, unit="USD")

    with pytest.raises(ReservationError, match="cover every active"):
        reconcile_completed_run(
            store,
            "run-1",
            path,
            root,
            settlements=[{"reservation_id": "wrong-reservation", "actual_amount": "0.10"}],
        )

    assert store.get_run("run-1")["state"] == "RUNNING"
    ledger = store.budget_snapshot("compute")
    assert ledger.confirmed_spend == 0
    assert ledger.outstanding_reservations == 1
    assert not [row for row in store.list_events() if row["event_type"] == "RUN_FINALIZED"]


def test_repeating_identical_recovery_is_idempotent(tmp_path):
    store, root, path, _completion = campaign_fixture(tmp_path)
    first = reconcile_completed_run(store, "run-1", path, root, settlements=None)
    second = reconcile_completed_run(store, "run-1", path, root, settlements=None)

    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert len([row for row in store.list_events() if row["event_type"] == "RUN_FINALIZED"]) == 1
