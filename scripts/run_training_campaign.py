#!/usr/bin/env python
"""Frozen two-embryo campaign with a shared wall-clock compute ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from biohub_ct.data.geff_io import read_geff_graph
from biohub_ct.data.paths import discover_datasets
from biohub_ct.data.splits import validate_split
from biohub_ct.data.zarr_io import open_zarr_volume
from biohub_ct.metrics.official_adapter import evaluate_official
from biohub_ct.pipelines.baseline_classical import ClassicalConfig, run_classical_baseline
from biohub_ct.pipelines.evaluate import _json_safe, official_module
from biohub_ct.pipelines.learned import (
    LearnedConfig,
    frame_probabilities,
    predict_record,
    predict_record_thresholds,
)
from biohub_ct.pipelines.submission_pipeline import environment_info, source_digest
from biohub_ct.submission.validator import iter_submission_graphs
from biohub_ct.submission.writer import write_submission
from biohub_ct.training.checkpoint import atomic_json, load_checkpoint
from biohub_ct.training.data import DataConfig, SparsePatchSampler
from biohub_ct.training.model import ModelConfig, PointDetector3D, masked_heatmap_loss
from biohub_ct.training.trainer import TrainConfig, train_detector


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def development_callback(sampler):
    def callback(model, step):
        device = next(model.parameters()).device
        losses = []
        with torch.inference_mode():
            for index in range(8):
                batch = sampler.sample(index, 2)
                image, target, weight = [
                    torch.as_tensor(batch[k], device=device) for k in ("image", "target", "weight")
                ]
                losses.append(
                    float(
                        masked_heatmap_loss(
                            model(image), target, weight, normalizer=batch.get("loss_normalizer")
                        )
                    )
                )
        return {
            "score": -float(np.mean(losses)),
            "masked_dev_loss": float(np.mean(losses)),
            "patch_batches": 8,
            "selection": "within-training-embryo development only",
        }

    return callback


def load_model(run_dir):
    manifest = json.loads((run_dir / "checkpoints" / "manifest.json").read_text())
    entry = manifest.get("best") or manifest["latest"]
    checkpoint = run_dir / "checkpoints" / entry["file"]
    state = load_checkpoint(checkpoint)
    model = PointDetector3D(ModelConfig(**state["model_config"])).cuda().eval()
    model.load_state_dict(state["model"])
    return model, {"path": str(checkpoint), "sha256": digest(checkpoint), "step": state["step"]}


def evaluate_record(record, graph, directory, runtime):
    directory.mkdir(parents=True, exist_ok=True)
    volume = open_zarr_volume(record.zarr_path, require_complete_chunks=True)
    truth, metadata = read_geff_graph(record.geff_path)
    csv = directory / (record.name + ".csv")
    write_submission(
        {record.name: graph},
        csv,
        expected_datasets=[record.name],
        shapes={record.name: volume.shape},
    )
    _, restored = next(iter_submission_graphs(csv))
    if restored.nodes_list != graph.nodes_list or restored.edges_set() != graph.edges_set():
        raise RuntimeError("Serialized prediction differs from inference graph")
    row = asdict(
        evaluate_official(
            restored, truth, scale=volume.scale, total_true_nodes=metadata.estimated_number_of_nodes
        )
    )
    row["adj_edge_jaccard"] = row.pop("adjusted_edge_jaccard")
    row.update(
        dataset=record.name,
        runtime_s=runtime,
        node_count_ratio=graph.num_nodes / metadata.estimated_number_of_nodes,
        diagnostics=graph.inference_diagnostics,
        csv_sha256=digest(csv),
    )
    atomic_json(directory / (record.name + ".json"), _json_safe(row))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--splits", default="configs/embryo-splits.json")
    parser.add_argument("--total-seconds", type=float, default=27000)
    parser.add_argument("--fold-seconds", type=float, default=9000)
    parser.add_argument("--refit-seconds", type=float, default=3600)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.total_seconds <= 0 or args.total_seconds > 27000 or args.fold_seconds <= 0:
        raise ValueError("Positive budget required; campaign maximum is 7.5 hours")
    output = Path("reports/campaigns") / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + args.total_seconds
    audit_path = Path("reports/data-verification.json")
    audit = json.loads(audit_path.read_text())
    if not args.smoke and (not audit["verified"] or not audit["sha256_checked"]):
        raise RuntimeError("Full dataset checksum audit required")
    if shutil.disk_usage(".").free < 15 * 1024**3:
        raise RuntimeError("15 GiB disk reserve required")
    metric = official_module()
    records = {r.name: r for r in discover_datasets(args.data_dir, require_geff=True)}
    splits = json.loads(Path(args.splits).read_text())
    folds = {key: splits[key] for key in ("fold0", "fold1")}
    for split in folds.values():
        validate_split(split, set(records))
    # Frozen lexicographic decile, selected without examining images or labels.
    panels = {key: sorted(split["train"])[::10] for key, split in folds.items()}
    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig(
        max_steps=1000000, max_seconds=args.fold_seconds, batch_size=4, validation_every_steps=2000
    )
    identity = {
        "source_digest": source_digest(),
        "git_commit": os.environ.get("BIOHUB_SOURCE_COMMIT"),
        "campaign_script_sha256": digest(__file__),
        "split_sha256": digest(args.splits),
        "data_audit_sha256": digest(audit_path),
    }
    manifest = {
        "run_id": args.run_id,
        "started_unix": time.time(),
        "args": vars(args),
        "identity": identity,
        "full_data_audit_passed": bool(audit["verified"] and audit["sha256_checked"]),
        "folds": folds,
        "development_ids": panels,
        "training": asdict(train_config),
        "data": asdict(data_config),
        "model": asdict(model_config),
        "threshold_candidates": [0.3, 0.5, 0.7],
        "threshold_panel": {k: v[:2] for k, v in panels.items()},
        "selection_rule": "best negative masked internal dev loss; threshold best internal panel official score",
        "inference": asdict(LearnedConfig()),
        "environment": environment_info(),
        "warning": "Dark background is a heuristic; unknown bright cells are ignored.",
    }
    atomic_json(output / "manifest.json", manifest)

    def status(stage, **fields):
        event = {
            "stage": stage,
            "elapsed_seconds": time.monotonic() - start,
            "remaining_seconds": max(0, deadline - time.monotonic()),
            **fields,
        }
        atomic_json(output / "status.json", _json_safe(event))
        print(json.dumps(_json_safe(event), allow_nan=False), flush=True)

    outcomes = {}
    try:
        for key, split in folds.items():
            if args.smoke and key != "fold0":
                continue
            if deadline - time.monotonic() < 120:
                raise TimeoutError("Insufficient campaign time to begin training")
            fit_ids = sorted(set(split["train"]) - set(panels[key]))
            sampler = SparsePatchSampler(
                list(records.values()), fit_ids, split["val"], data_config, dev_ids=panels[key]
            )
            dev_sampler = SparsePatchSampler(
                list(records.values()),
                panels[key],
                split["val"],
                replace(data_config, augment=False, steps_per_frame=1),
            )
            callback = development_callback(dev_sampler)
            status(
                "training", fold=key, train_clips=len(fit_ids), development_clips=len(panels[key])
            )
            config = replace(
                train_config, max_seconds=min(args.fold_seconds, deadline - time.monotonic() - 60)
            )
            if args.smoke:
                config = replace(
                    config,
                    max_steps=20,
                    checkpoint_every_steps=10,
                    validation_every_steps=10,
                    log_every_steps=5,
                )
            result = train_detector(
                config, sampler, output / key, model_config, {**identity, "fold": key}, callback
            )
            if result["status"] != "completed":
                raise RuntimeError(f"Training stopped: {result}")
            outcomes[key] = {"training": result}
            if args.smoke:
                # Real-data checkpoint readback and continued optimizer updates on GPU.
                resume_result = train_detector(
                    replace(config, max_steps=30),
                    sampler,
                    output / key,
                    model_config,
                    {**identity, "fold": key},
                    callback,
                    resume=output / key / "checkpoints",
                )
                model, checkpoint = load_model(output / key)
                volume = open_zarr_volume(
                    records[panels[key][0]].zarr_path, require_complete_chunks=True
                )
                inference_start = time.monotonic()
                raw = volume.read_frame(0)
                probability = frame_probabilities(model, raw, deadline_at=deadline)
                outcomes[key].update(
                    resumed=resume_result,
                    checkpoint=checkpoint,
                    frame_shape=list(raw.shape),
                    probability_shape=list(probability.shape),
                    inference_seconds=time.monotonic() - inference_start,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    probability_range=[float(probability.min()), float(probability.max())],
                )
                atomic_json(output / "summary.json", _json_safe(outcomes))
                status("smoke_completed", outcomes=outcomes)
                return
            del sampler, dev_sampler

        # Both models finish fitting before either outer fold is examined.
        for key, split in folds.items():
            model, checkpoint = load_model(output / key)
            outcomes[key]["checkpoint"] = checkpoint
            configurations = [
                replace(LearnedConfig(), threshold=t) for t in manifest["threshold_candidates"]
            ]
            candidate_rows = [[] for _ in configurations]
            for name in panels[key][:2]:
                status("threshold_selection", fold=key, dataset=name)
                then = time.monotonic()
                graphs = predict_record_thresholds(
                    model, records[name], configurations, deadline_at=deadline
                )
                for config, prediction, rows in zip(configurations, graphs, candidate_rows):
                    rows.append(
                        evaluate_record(
                            records[name],
                            prediction,
                            output / key / f"development-{config.threshold}",
                            time.monotonic() - then,
                        )
                    )
            candidates = [
                {"threshold": config.threshold, "aggregate": metric.summarise(rows)}
                for config, rows in zip(configurations, candidate_rows)
            ]
            selected = max(candidates, key=lambda row: row["aggregate"]["score"])
            config = replace(LearnedConfig(), threshold=selected["threshold"])
            outcomes[key].update(
                threshold_candidates=candidates, selected_threshold=config.threshold
            )
            atomic_json(output / key / "selection.json", _json_safe(outcomes[key]))
            learned_rows, classical_rows = [], []
            for index, name in enumerate(split["val"]):
                if shutil.disk_usage(".").free < 15 * 1024**3:
                    raise RuntimeError("Disk reserve reached")
                status(
                    "outer_evaluation",
                    fold=key,
                    dataset=name,
                    completed=index,
                    total=len(split["val"]),
                )
                then = time.monotonic()
                graph = predict_record(model, records[name], config, deadline_at=deadline)
                learned_rows.append(
                    evaluate_record(
                        records[name],
                        graph,
                        output / key / "outer-learned",
                        time.monotonic() - then,
                    )
                )
                then = time.monotonic()
                graph = run_classical_baseline(
                    records[name], config=ClassicalConfig(), deadline_at=deadline
                )
                classical_rows.append(
                    evaluate_record(
                        records[name],
                        graph,
                        output / key / "outer-classical",
                        time.monotonic() - then,
                    )
                )
                atomic_json(
                    output / key / "outer-progress.json",
                    _json_safe(
                        {
                            "complete": len(learned_rows) == len(split["val"]),
                            "learned": learned_rows,
                            "classical": classical_rows,
                        }
                    ),
                )
            outcomes[key].update(
                learned=metric.summarise(learned_rows),
                classical=metric.summarise(classical_rows),
                complete_outer_fold=True,
                inference_clean=all(
                    not r["diagnostics"]["fallback"] and not r["diagnostics"]["capped_frames"]
                    for r in learned_rows
                ),
            )
            atomic_json(output / "summary.json", _json_safe(outcomes))
            del model
            torch.cuda.empty_cache()
        # Refit only after complete paired validation improves in both directions.
        promotable = all(
            row["inference_clean"] and row["learned"]["score"] > row["classical"]["score"]
            for row in outcomes.values()
        )
        remaining = deadline - time.monotonic() - 60
        if promotable and remaining >= 600 and args.refit_seconds > 0:
            status("all_data_refit")
            refit_inference = replace(
                LearnedConfig(),
                threshold=float(
                    np.median([row["selected_threshold"] for row in outcomes.values()])
                ),
            )
            atomic_json(output / "refit-inference.json", asdict(refit_inference))
            sampler = SparsePatchSampler(list(records.values()), sorted(records), [], data_config)
            result = train_detector(
                replace(train_config, max_seconds=min(args.refit_seconds, remaining)),
                sampler,
                output / "refit",
                model_config,
                {**identity, "fold": "all_data"},
            )
            outcomes["refit"] = result
        atomic_json(output / "summary.json", _json_safe(outcomes))
        status(
            "completed",
            improved_both_folds=promotable,
            release_status="Requires private checkpoint packaging and offline Kaggle rehearsal",
        )
    except TimeoutError as exc:
        atomic_json(output / "summary.json", _json_safe(outcomes))
        status(
            "budget_exhausted",
            reason=str(exc),
            release_status="Not eligible for promotion without full gates",
        )
    except BaseException as exc:
        status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
