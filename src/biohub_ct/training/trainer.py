from __future__ import annotations

import json
import math
import random
import signal
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from biohub_ct.training.checkpoint import (
    atomic_json,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    versions,
)
from biohub_ct.training.model import ModelConfig, PointDetector3D, masked_heatmap_loss


def scaled_optimizer_update(
    model, optimizer, scaler, forward_loss, gradient_clip, *, max_retries=8, on_overflow=None
):
    """Retry overflowed AMP backward passes without consuming an optimizer step.

    This detector has no mutable running-statistic buffers. Restore RNG so a retry
    uses the same stochastic forward pass. Non-AMP or persistent failures still stop.
    """
    rng = capture_rng_state()
    for retry in range(max_retries + 1):
        if retry:
            restore_rng_state(rng)
        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
        if finite:
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            return loss, norm, retry
        if not scaler.is_enabled() or retry == max_retries:
            raise FloatingPointError(f"Nonfinite gradients after {retry} AMP retries")
        old_scale = scaler.get_scale()
        # unscale_ recorded nonfinite gradients, so GradScaler skips optimizer.step.
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        if not 0 < new_scale < old_scale:
            raise FloatingPointError("AMP overflow did not reduce the loss scale")
        if on_overflow is not None:
            on_overflow(
                {
                    "event": "amp_overflow_retry",
                    "retry": retry + 1,
                    "old_scale": old_scale,
                    "new_scale": new_scale,
                    "finite_loss": float(loss.detach()),
                }
            )
    raise AssertionError("Unreachable AMP retry state")


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 100000
    max_seconds: float = 7200
    batch_size: int = 2
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    seed: int = 20260905
    device: str = "cuda"
    amp: bool = True
    deterministic: bool = True
    checkpoint_every_steps: int = 1000
    checkpoint_every_seconds: float = 300
    log_every_steps: int = 10
    validation_every_steps: int = 2000
    gradient_clip: float = 10.0
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        if (
            min(
                self.max_steps,
                self.batch_size,
                self.checkpoint_every_steps,
                self.log_every_steps,
                self.validation_every_steps,
                self.cpu_threads,
            )
            < 1
        ):
            raise ValueError("Training step/count settings must be positive")
        if not all(
            math.isfinite(x) and x > 0
            for x in (
                self.max_seconds,
                self.learning_rate,
                self.checkpoint_every_seconds,
                self.gradient_clip,
            )
        ):
            raise ValueError("Training numerical settings must be finite and positive")
        if self.checkpoint_every_seconds > 600 or self.weight_decay < 0 or self.seed < 0:
            raise ValueError(
                "Checkpoint interval <=600 seconds and nonnegative weight decay/seed required"
            )


def train_detector(
    config: TrainConfig,
    sampler,
    output_dir: Path | str,
    model_config: ModelConfig = ModelConfig(),
    identity: dict | None = None,
    validation_callback: Callable[[PointDetector3D, int], dict] | None = None,
    resume: Path | str | None = None,
    resume_expected_identity: dict | None = None,
) -> dict:
    """One fold/refit run. Callback returns a finite `score` (higher is better).

    Sampler contract: sample(step,batch_size) -> image,target,weight float32 B1ZYX.
    Step-indexed sampling plus complete RNG state supports exact same-stack resume.
    Walltime includes data loading, validation and checkpointing in this invocation.
    The caller must enforce the campaign-wide budget across runs.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and (output / "checkpoints" / "manifest.json").exists():
        raise FileExistsError("Run already has checkpoints; pass resume explicitly")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = PointDetector3D(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # Constant schedule does not change when a walltime-bounded run resumes.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    data_config = asdict(sampler.config) if hasattr(sampler, "config") else {}
    run_identity = dict(identity or {})
    split = {name: list(getattr(sampler, name, [])) for name in ("train_ids", "val_ids", "dev_ids")}
    run_identity["split"] = split
    step, elapsed_prior, best_score = 0, 0.0, None
    overflow_retries = 0
    resume_parent = None
    if resume is not None:
        expected = resume_expected_identity or run_identity
        # Explicit repair migration permits only source identity changes.
        source_keys = {"source_digest", "campaign_script_sha256", "git_commit"}
        if {k: v for k, v in expected.items() if k not in source_keys} != {
            k: v for k, v in run_identity.items() if k not in source_keys
        }:
            raise ValueError("Repair resume changed data or split identity")
        state = load_checkpoint(resume, expected_identity=expected)
        # Only budget extensions may differ. Numerical/runtime/sampling settings remain fixed.
        old_config, current_config = dict(state["config"]), asdict(config)
        for key in ("max_steps", "max_seconds"):
            old_config.pop(key)
            current_config.pop(key)
        if (
            old_config != current_config
            or state["model_config"] != asdict(model_config)
            or state["data_config"] != data_config
        ):
            raise ValueError("Checkpoint training/model/data config mismatch")
        if state["versions"] != versions():
            raise ValueError("Exact training resume requires matching dependency versions")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng_state(state["rng"])
        step, elapsed_prior = int(state["step"]), float(state["elapsed_seconds"])
        best_score = state.get("best_score")
        overflow_retries = int(state.get("amp_overflow_retries", 0))
        resume_parent = {"checkpoint": str(resume), "identity": state["identity"]}
    start = time.monotonic()
    last_save = start
    stopped = {"signal": None}
    previous_handlers = {}

    def stop_handler(number, _frame) -> None:
        stopped["signal"] = int(number)

    for number in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[number] = signal.signal(number, stop_handler)
        except ValueError:
            pass  # Main campaign uses main thread; unit callers may use a worker thread.

    def save(is_best: bool = False) -> Path:
        return save_checkpoint(
            output / "checkpoints",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": capture_rng_state(),
                "step": step,
                "elapsed_seconds": elapsed_prior + time.monotonic() - start,
                "best_score": best_score,
                "config": asdict(config),
                "model_config": asdict(model_config),
                "data_config": data_config,
                "identity": run_identity,
                "versions": versions(),
                "amp_overflow_retries": overflow_retries,
                "resume_parent": resume_parent,
            },
            is_best=is_best,
        )

    last_loss = None
    last_checkpoint = None
    atomic_json(
        output / "status.json",
        {
            "status": "running",
            "step": step,
            "identity": run_identity,
            "config": asdict(config),
            "versions": versions(),
        },
    )
    try:
        with (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as metrics:
            while step < config.max_steps:
                if stopped["signal"] is not None or time.monotonic() - start >= config.max_seconds:
                    break
                iteration_start = time.monotonic()
                batch = sampler.sample(step, config.batch_size)
                tensors = [
                    torch.as_tensor(batch[key], device=device)
                    for key in ("image", "target", "weight")
                ]
                if any(t.dtype != torch.float32 for t in tensors):
                    raise ValueError("Sampler tensors must be float32")
                model.train()

                def forward_loss(tensors=tensors, batch=batch):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=config.amp and device.type == "cuda",
                    ):
                        return masked_heatmap_loss(
                            model(tensors[0]),
                            tensors[1],
                            tensors[2],
                            normalizer=batch.get("loss_normalizer"),
                        )

                def log_overflow(event, step=step):
                    metrics.write(json.dumps({"step": step, **event}, allow_nan=False) + "\n")

                loss, grad_norm, retries = scaled_optimizer_update(
                    model,
                    optimizer,
                    scaler,
                    forward_loss,
                    config.gradient_clip,
                    on_overflow=log_overflow,
                )
                overflow_retries += retries
                scheduler.step()
                step += 1
                last_loss = float(loss.detach())
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic() - start
                if step % config.log_every_steps == 0 or step == 1:
                    event = {
                        "step": step,
                        "loss": last_loss,
                        "amp_overflow_retries": overflow_retries,
                        "amp_scale": scaler.get_scale(),
                        "gradient_norm": float(grad_norm),
                        "elapsed_seconds": elapsed_prior + elapsed,
                        "step_seconds": time.monotonic() - iteration_start,
                        "metadata": batch.get("metadata", []),
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0,
                    }
                    metrics.write(json.dumps(event, allow_nan=False) + "\n")
                    atomic_json(output / "status.json", {"status": "running", **event})
                improved = False
                if validation_callback is not None and step % config.validation_every_steps == 0:
                    rng_before = capture_rng_state()
                    model.eval()
                    try:
                        with torch.inference_mode():
                            validation = validation_callback(model, step)
                    finally:
                        restore_rng_state(rng_before)
                    score = float(validation["score"])
                    if not math.isfinite(score):
                        raise FloatingPointError("Nonfinite validation score")
                    metrics.write(
                        json.dumps({"step": step, "validation": validation}, allow_nan=False) + "\n"
                    )
                    improved = best_score is None or score > best_score
                    if improved:
                        best_score = score
                if (
                    improved
                    or step % config.checkpoint_every_steps == 0
                    or time.monotonic() - last_save >= config.checkpoint_every_seconds
                ):
                    last_checkpoint = save(improved)
                    last_save = time.monotonic()
            last_checkpoint = save()
        reason = (
            "signal"
            if stopped["signal"] is not None
            else ("max_steps" if step >= config.max_steps else "walltime")
        )
        report = {
            "status": "interrupted" if stopped["signal"] else "completed",
            "reason": reason,
            "signal": stopped["signal"],
            "step": step,
            "amp_overflow_retries": overflow_retries,
            "last_loss": last_loss,
            "best_score": best_score,
            "checkpoint": str(last_checkpoint),
            "elapsed_seconds": elapsed_prior + time.monotonic() - start,
        }
        atomic_json(output / "status.json", report)
        return report
    except BaseException as exc:
        # Do not overwrite the last valid model after a nonfinite update or arbitrary error.
        atomic_json(
            output / "status.json",
            {
                "status": "failed",
                "step": step,
                "elapsed_seconds": elapsed_prior + time.monotonic() - start,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "last_valid_checkpoint": str(last_checkpoint) if last_checkpoint else None,
            },
        )
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
