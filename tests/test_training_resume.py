"""Recovery tests compare observable training continuation, not serialization alone."""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biohub_ct.training import checkpoint


def _assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            _assert_equal(av, bv)
    else:
        assert a == b


def _components():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.Dropout(0.2), torch.nn.SiLU(), torch.nn.Linear(8, 1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    return model, optimizer, scheduler, scaler


def _update(parts):
    model, optimizer, scheduler, scaler = parts
    # Exercise Python/NumPy/Torch RNGs, dropout and optimizer history.
    image = torch.randn(5, 3) + float(np.random.random()) + random.random()
    target = torch.arange(5, dtype=torch.float32).reshape(-1, 1) / 5
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(image), target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.detach().clone()


def _state(parts, step=1):
    model, optimizer, scheduler, scaler = parts
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": checkpoint.capture_rng_state(),
        "step": step,
        "elapsed_seconds": 1.0,
        "config": {"learning_rate": 0.002},
        "model_config": {},
        "data_config": {},
        "identity": {"config": "config-A", "split": "split-A", "source": "source-A"},
        "versions": {"torch": str(torch.__version__)},
    }


def test_checkpoint_resume_matches_uninterrupted_next_update(tmp_path):
    random.seed(302)
    np.random.seed(302)
    torch.manual_seed(302)
    uninterrupted = _components()
    _update(uninterrupted)
    saved = _state(uninterrupted)
    checkpoint.save_checkpoint(tmp_path, saved, is_best=True)
    expected_loss = _update(uninterrupted)
    expected_parts = [copy.deepcopy(part.state_dict()) for part in uninterrupted]
    expected_draws = (random.random(), float(np.random.random()), torch.rand(3))

    resumed = _components()  # Constructor intentionally consumes the wrong RNG state.
    restored = checkpoint.load_checkpoint(tmp_path, expected_identity=saved["identity"])
    for part, key in zip(resumed, ("model", "optimizer", "scheduler", "scaler")):
        part.load_state_dict(restored[key])
    checkpoint.restore_rng_state(restored["rng"])
    actual_loss = _update(resumed)
    _assert_equal(expected_loss, actual_loss)
    for part, expected in zip(resumed, expected_parts):
        _assert_equal(expected, part.state_dict())
    _assert_equal(expected_draws, (random.random(), float(np.random.random()), torch.rand(3)))
    assert restored["step"] == 1


@pytest.mark.parametrize("identity_key", ["config", "split", "source"])
def test_resume_rejects_changed_identity(tmp_path, identity_key):
    state = _state(_components())
    checkpoint.save_checkpoint(tmp_path, state)
    changed = dict(state["identity"], **{identity_key: "different"})
    with pytest.raises((ValueError, RuntimeError), match="[Ii]dentity|mismatch"):
        checkpoint.load_checkpoint(tmp_path, expected_identity=changed)


def test_corruption_is_rejected_before_deserialization(tmp_path, monkeypatch):
    path = checkpoint.save_checkpoint(tmp_path, _state(_components()))
    contents = bytearray(path.read_bytes())
    contents[len(contents) // 2] ^= 0xFF
    path.write_bytes(contents)

    def forbidden(*args, **kwargs):
        pytest.fail("Corrupted checkpoint reached torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises((ValueError, RuntimeError), match="[Cc]hecksum|SHA|hash"):
        checkpoint.load_checkpoint(path)


def test_failed_save_preserves_latest_and_validation_best(tmp_path, monkeypatch):
    initial = _state(_components(), step=1)
    checkpoint.save_checkpoint(tmp_path, initial, is_best=True)
    manifest_before = json.loads((tmp_path / "manifest.json").read_text())

    def disk_failure(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(torch, "save", disk_failure)
    with pytest.raises(OSError, match="full disk"):
        checkpoint.save_checkpoint(tmp_path, _state(_components(), step=2))
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest_before
    restored = checkpoint.load_checkpoint(tmp_path)
    assert restored["step"] == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_production_trainer_resumes_same_updates_after_sampler_failure(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable; CPU result does not establish GPU recovery")
    from biohub_ct.training.data import DataConfig
    from biohub_ct.training.model import ModelConfig
    from biohub_ct.training.trainer import TrainConfig, train_detector

    class Sampler:
        config = DataConfig(patch_shape=(8, 8, 8))
        train_ids, val_ids, dev_ids = ["44b6_fit"], ["6bba_outer"], ["44b6_dev"]

        def __init__(self, fail_at=None):
            self.fail_at, self.steps = fail_at, []

        def sample(self, step, batch_size):
            if step == self.fail_at:
                raise RuntimeError("simulated loader interruption")
            self.steps.append(step)
            rng = np.random.default_rng(813 + step)
            image = rng.random((batch_size, 1, 8, 8, 8), dtype=np.float32)
            target = np.zeros_like(image)
            target[..., 4, 4, 4] = 1
            return {
                "image": image,
                "target": target,
                "weight": np.ones_like(image) / 512,
                "loss_normalizer": float(batch_size),
            }

    def validation(model, step):
        # A diagnostic callback must not perturb subsequent training RNG.
        random.random()
        np.random.random()
        torch.rand(2, device=device)
        return {"score": float(step)}

    config = TrainConfig(
        max_steps=4,
        max_seconds=120,
        batch_size=1,
        device=device,
        amp=device == "cuda",
        checkpoint_every_steps=1,
        validation_every_steps=1,
        log_every_steps=1,
        cpu_threads=1,
    )
    common = {
        "config": config,
        "model_config": ModelConfig(base_channels=4),
        "identity": {"source": "test-source", "input": "test-input"},
        "validation_callback": validation,
    }
    complete = tmp_path / "complete"
    interrupted = tmp_path / "interrupted"
    full_report = train_detector(sampler=Sampler(), output_dir=complete, **common)
    full = checkpoint.load_checkpoint(complete / "checkpoints")
    with pytest.raises(RuntimeError, match="loader interruption"):
        train_detector(sampler=Sampler(fail_at=2), output_dir=interrupted, **common)
    partial = checkpoint.load_checkpoint(interrupted / "checkpoints")
    assert partial["step"] == 2
    assert json.loads((interrupted / "status.json").read_text())["status"] == "failed"
    resumed_sampler = Sampler()
    resumed_report = train_detector(
        sampler=resumed_sampler,
        output_dir=interrupted,
        resume=interrupted / "checkpoints",
        **common,
    )
    restored = checkpoint.load_checkpoint(interrupted / "checkpoints")
    assert resumed_sampler.steps == [2, 3]
    assert full_report["step"] == resumed_report["step"] == 4
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "best_score"):
        _assert_equal(full[key], restored[key])


def test_production_resume_rejects_changed_learning_rate_before_sampling(tmp_path):
    from dataclasses import replace

    from biohub_ct.training.data import DataConfig
    from biohub_ct.training.model import ModelConfig
    from biohub_ct.training.trainer import TrainConfig, train_detector

    class Sampler:
        config = DataConfig(patch_shape=(8, 8, 8))
        train_ids, val_ids, dev_ids = ["44b6_fit"], ["6bba_outer"], []

        def sample(self, step, batch_size):
            image = np.ones((batch_size, 1, 8, 8, 8), dtype=np.float32)
            return {"image": image, "target": image, "weight": image}

    config = TrainConfig(
        max_steps=1,
        max_seconds=60,
        batch_size=1,
        device="cpu",
        amp=False,
        checkpoint_every_steps=1,
        cpu_threads=1,
    )
    train_detector(config, Sampler(), tmp_path, ModelConfig(base_channels=4))
    manifest_before = (tmp_path / "checkpoints" / "manifest.json").read_bytes()

    class ForbiddenSampler(Sampler):
        def sample(self, step, batch_size):
            pytest.fail("Incompatible resume reached training data")

    with pytest.raises(ValueError, match="config mismatch"):
        train_detector(
            replace(config, learning_rate=config.learning_rate * 2),
            ForbiddenSampler(),
            tmp_path,
            ModelConfig(base_channels=4),
            resume=tmp_path / "checkpoints",
        )
    assert (tmp_path / "checkpoints" / "manifest.json").read_bytes() == manifest_before
