import json

import pytest

pytest.importorskip("torch")
from biohub_ct.training.recovery import recovery_plan


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
