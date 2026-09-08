"""Render fixed-control E1 overlays and verify every displayed detection identity."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.evaluate_detection_diagnostics import (
    EvaluationProgress,
    _artifact_path,
    _candidate_points_legacy,
    _candidate_points_plateau,
    _expected_identity,
    _run_for,
    _verify_cache_index,
    atomic_json,
    enforce_wall_deadline,
    exact_initial_grid,
    verify_cache_precision_identity,
    verify_local_contract_files,
)

from biohub_ct.campaign.detection_diagnostics import hash_array, hash_file
from biohub_ct.campaign.learned_logit_adapter import (
    load_learned_tile_cache,
    reconstruct_probabilities,
)
from biohub_ct.data.geff_io import read_geff_graph
from biohub_ct.data.zarr_io import open_zarr_volume


def validate_evaluation_report(report, plan):
    """Require actual grid and frame coverage, rather than trusting count fields."""
    if report.get("status") != "COMPLETE" or report.get("completed_initial_configurations") != 54:
        raise ValueError("Overlays require the completed preregistered grid")
    initial = report.get("initial_rows", [])
    expected_cells = {cell.cell_id for cell in exact_initial_grid(plan)}
    actual_cells = [row.get("cell_id") for row in initial]
    if len(initial) != 54 or len(set(actual_cells)) != 54 or set(actual_cells) != expected_cells:
        raise ValueError("Evaluation grid coverage is incomplete or duplicated")
    population = {
        (dataset["dataset_id"], frame["frame"])
        for dataset in plan["population"]["datasets"]
        for frame in dataset["frames"]
    }
    for cell in initial:
        frames = cell.get("frame_rows", [])
        actual = [(row.get("dataset_id"), row.get("frame")) for row in frames]
        if (
            cell.get("status") != "COMPLETE"
            or len(actual) != len(population)
            or set(actual) != population
        ):
            raise ValueError("Evaluation frame coverage is incomplete or duplicated")
    if report.get("fixed_high_cap_diagnostic", {}).get("status") != "COMPLETE":
        raise ValueError("Fixed high-cap diagnostic is incomplete")


def render_sheet(raw, scale, truth, point_sets, *, title, destination):
    """Show full-depth MIPs and thin orthogonal slabs through a fixed GT anchor."""
    panel_size = 300
    label_height = 34
    canvas = Image.new("RGB", (3 * panel_size, 4 * (panel_size + label_height) + 70), "#10151c")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="white")
    draw.text(
        (10, 26),
        "Cyan circles: predictions. Yellow crosses: sparse annotations. MIPs collapse depth.",
        fill="white",
    )
    draw.text(
        (10, 44),
        "Slabs: +/-1.625 um; displayed points within same slab. 40 um crop at nearest central annotation.",
        fill="white",
    )
    scale = np.asarray(scale, dtype=float)
    center = (np.asarray(raw.shape) - 1) / 2
    anchor = (
        truth[np.argmin(np.linalg.norm((truth - center) * scale, axis=1))] if len(truth) else center
    )
    axes = [(0, 1, 2, "XY"), (1, 0, 2, "XZ"), (2, 0, 1, "YZ")]
    bounds = []
    for coord, spacing, size in zip(anchor, scale, raw.shape):
        bounds.append(
            (
                max(0, int(np.floor(coord - 20 / spacing))),
                min(size, int(np.ceil(coord + 20 / spacing)) + 1),
            )
        )
    for row, (variant, points, is_crop) in enumerate(
        [
            ("control", point_sets[0], False),
            ("plateau", point_sets[1], False),
            ("control", point_sets[0], True),
            ("plateau", point_sets[1], True),
        ]
    ):
        for col, (depth, vertical, horizontal, plane) in enumerate(axes):
            lows = np.zeros(3, dtype=int)
            highs = np.asarray(raw.shape)
            if is_crop:
                lows = np.asarray([pair[0] for pair in bounds])
                highs = np.asarray([pair[1] for pair in bounds])
                lows[depth] = max(0, int(np.ceil(anchor[depth] - 1.625 / scale[depth])))
                highs[depth] = min(
                    raw.shape[depth], int(np.floor(anchor[depth] + 1.625 / scale[depth])) + 1
                )
            slab = raw[tuple(slice(int(lo), int(hi)) for lo, hi in zip(lows, highs))]
            projection = slab.max(axis=depth).astype(np.float32)
            low, high = np.quantile(projection, [0.01, 0.995])
            gray = np.clip((projection - low) / max(1.0, high - low) * 255, 0, 255).astype(np.uint8)
            physical_size = (highs - lows) * scale
            ratio = panel_size / max(physical_size[horizontal], physical_size[vertical])
            width = max(1, round(physical_size[horizontal] * ratio))
            height = max(1, round(physical_size[vertical] * ratio))
            offset_x, offset_y = (panel_size - width) // 2, (panel_size - height) // 2
            panel = Image.new("RGB", (panel_size, panel_size), "#202020")
            picture = (
                Image.fromarray(gray)
                .convert("RGB")
                .resize((width, height), Image.Resampling.NEAREST)
            )
            panel.paste(picture, (offset_x, offset_y))
            marks = ImageDraw.Draw(panel)
            for coordinates, color, symbol in [
                (points, "#00eaff", "circle"),
                (truth, "#ffed4a", "cross"),
            ]:
                chosen = coordinates[np.all((coordinates >= lows) & (coordinates < highs), axis=1)]
                for point in chosen:
                    x = (
                        offset_x
                        + (point[horizontal] - lows[horizontal] + 0.5)
                        / (highs[horizontal] - lows[horizontal])
                        * width
                    )
                    y = (
                        offset_y
                        + (point[vertical] - lows[vertical] + 0.5)
                        / (highs[vertical] - lows[vertical])
                        * height
                    )
                    if symbol == "circle":
                        marks.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=1)
                    else:
                        marks.line((x - 4, y, x + 4, y), fill=color)
                        marks.line((x, y - 4, x, y + 4), fill=color)
            marks.line(
                (12, panel_size - 14, 12 + 10 * ratio, panel_size - 14), fill="white", width=2
            )
            marks.text((12, panel_size - 28), "10 um", fill="white")
            x0, y0 = col * panel_size, 70 + row * (panel_size + label_height)
            draw.text(
                (x0 + 4, y0 + 8),
                f"{variant} {plane} {'slab crop' if is_crop else 'full MIP'}",
                fill="white",
            )
            canvas.paste(panel, (x0, y0 + label_height))
    canvas.save(destination)
    return {
        "anchor_raw_zyx": anchor.tolist(),
        "crop_bounds_zyx": bounds,
        "slab_half_width_um": 1.625,
        "mip_depth_ambiguity": True,
        "annotation_completeness": "sparse; unmatched predictions are not automatically false positives",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("plan-json", "evaluation-json", "artifact-root", "data-root"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    args = parser.parse_args()
    run_id = os.environ["BIOHUB_RUN_ID"]
    digest = os.environ["BIOHUB_RUN_SPEC_SHA256"]
    intent = os.environ["BIOHUB_INTENT_ID"]
    token = int(os.environ["BIOHUB_FENCING_TOKEN"])
    attempt = Path(os.environ["BIOHUB_ATTEMPT_DIR"]).resolve(strict=True)
    if attempt != (ROOT / "reports/campaign-workers" / run_id).resolve() or token < 0:
        raise ValueError("Invalid exclusive overlay attempt identity")
    progress_path = Path(os.environ["BIOHUB_PROGRESS_PATH"]).resolve()
    if not progress_path.is_relative_to(attempt):
        raise ValueError("Progress must live in this attempt")
    plan = json.loads(args.plan_json.read_text())
    report = json.loads(args.evaluation_json.read_text())
    validate_evaluation_report(report, plan)
    if report["plan_file_sha256"] != hash_file(args.plan_json):
        raise ValueError("Evaluation plan identity changed")
    if report["evaluator_file_sha256"] != hash_file(
        ROOT / "scripts/evaluate_detection_diagnostics.py"
    ):
        raise ValueError("Evaluation implementation identity changed")
    verify_local_contract_files(plan, source_root=ROOT)
    deadline = time.monotonic() + args.max_wall_seconds
    progress = EvaluationProgress(progress_path, run_id, digest)
    progress.write()
    artifacts = {}
    rows = []
    try:
        with progress.heartbeat(), enforce_wall_deadline(deadline):
            for dataset in plan["population"]["datasets"]:
                dataset_id = dataset["dataset_id"]
                run = _run_for(plan, "native-amp", dataset_id)
                index = _verify_cache_index(args.artifact_root, plan, run)
                volume = open_zarr_volume(
                    args.data_root / f"{dataset_id}.zarr", require_complete_chunks=True
                )
                graph, _ = read_geff_graph(args.data_root / f"{dataset_id}.geff")
                nodes = graph.nodes_by_time()
                for frame in dataset["frames"]:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Overlay rendering deadline reached")
                    frame_id = frame["frame"]
                    raw = volume.read_frame(frame_id)
                    if hash_array(raw) != frame["input_frame_typed_array_sha256"]:
                        raise ValueError("Overlay raw frame identity changed")
                    cache_row = next(row for row in index["frames"] if row["frame"] == frame_id)
                    cache_path = _artifact_path(
                        args.artifact_root, run, "/" + cache_row["logits_path"]
                    )
                    payload, _ = load_learned_tile_cache(
                        cache_path,
                        expected_identity=_expected_identity(plan, dataset, frame, "native-amp"),
                    )
                    verify_cache_precision_identity(payload, "native-amp")
                    if payload.inference_device_type != "cuda":
                        raise ValueError("Overlay cache must retain CUDA device identity")
                    probability = reconstruct_probabilities(payload, activation="native")
                    truth = np.asarray(
                        [node.coord for node in nodes.get(frame_id, [])], dtype=float
                    ).reshape(-1, 3)
                    annotation_identity = plan["evaluator_contract"]["frozen_annotation_frames"][
                        dataset_id
                    ][str(frame_id)]
                    if hash_array(truth) != annotation_identity["coordinate_float64_zyx_sha256"]:
                        raise ValueError(
                            "Displayed annotation coordinates differ from preregistered truth"
                        )
                    point_sets, point_hashes = [], {}
                    for variant, extract in [
                        ("legacy_voxel_maxima", _candidate_points_legacy),
                        ("connected_plateau", _candidate_points_plateau),
                    ]:
                        points = extract(
                            probability,
                            raw_shape=raw.shape,
                            raw_scale_zyx_um=volume.scale,
                            xy_stride=4,
                            threshold=0.3,
                            radius_um=3.0,
                            max_nodes=2000,
                        )["points_raw_zyx"]
                        cell = next(
                            row
                            for row in report["initial_rows"]
                            if row["precision_variant"] == "amp_native_sigmoid"
                            and row["extraction_variant"] == variant
                            and row["threshold_probability"] == 0.3
                            and row["radius_um"] == 3.0
                        )
                        reference = next(
                            row
                            for row in cell["frame_rows"]
                            if row["dataset_id"] == dataset_id and row["frame"] == frame_id
                        )
                        point_hash = hash_array(np.asarray(points, dtype=np.int64).reshape(-1, 3))
                        if point_hash != reference["predicted_points_raw_zyx_sha256"]:
                            raise ValueError("Displayed points differ from evaluated nodes")
                        point_sets.append(points)
                        point_hashes[variant] = point_hash
                    output = attempt / f"{dataset_id}-frame-{frame_id:05d}.png"
                    geometry = render_sheet(
                        raw,
                        volume.scale,
                        truth,
                        point_sets,
                        title=f"{dataset_id} frame {frame_id}; native CUDA sigmoid; p=0.3, radius=3um, cap=2000",
                        destination=output,
                    )
                    artifacts[output.relative_to(ROOT).as_posix()] = hash_file(output)
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "frame": frame_id,
                            "image": output.name,
                            "input_frame_sha256": hash_array(raw),
                            "predicted_point_sha256": point_hashes,
                            **geometry,
                        }
                    )
            if len(rows) != 8:
                raise ValueError("Overlay population is incomplete")
            index_path = attempt / "overlay-index.json"
            atomic_json(
                index_path,
                {
                    "status": "COMPLETE",
                    "run_id": run_id,
                    "run_spec_sha256": digest,
                    "evaluation_sha256": hash_file(args.evaluation_json),
                    "plan_sha256": hash_file(args.plan_json),
                    "overlays": rows,
                    "manual_review_performed": False,
                    "advancement_eligible": False,
                },
            )
            artifacts[index_path.relative_to(ROOT).as_posix()] = hash_file(index_path)
        progress.completed_units = 1
        progress.write()
        artifacts[progress_path.relative_to(ROOT).as_posix()] = hash_file(progress_path)
        result = {
            "run_id": run_id,
            "run_spec_sha256": digest,
            "intent_id": intent,
            "fencing_token": token,
            "status": "COMPLETE",
            "completed_units": 1,
            "artifact_sha256": artifacts,
        }
        atomic_json(attempt / "result.json", result)
    except BaseException as exc:
        progress.write(error=f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
