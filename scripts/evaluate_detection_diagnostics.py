"""Evaluate the preregistered E1 grid from strict learned-logit caches."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.campaign.detection_diagnostics import (
    CacheIdentity,
    PlateauCandidate,
    cache_manifest_path,
    hash_array,
    hash_file,
    hash_json,
)
from biohub_ct.campaign.learned_logit_adapter import (
    create_adapter,
    load_learned_tile_cache,
    reconstruct_probabilities,
)
from biohub_ct.config import MAX_MATCH_DISTANCE_MICRONS
from biohub_ct.data.geff_io import read_geff_graph
from biohub_ct.data.zarr_io import open_zarr_volume
from biohub_ct.detection.peaks import anisotropic_nms
from biohub_ct.pipelines.learned import frame_probabilities, probability_nodes
from biohub_ct.training.data import model_to_raw_points, normalize_frame

TEMPORAL_MISSING_REASON = "isolated_frame_panel_has_no_complete_temporal_clip"
MANUAL_MISSING_REASON = "manual_overlay_review_not_performed_by_evaluator"


def operational_completed_units(report_status: str) -> int:
    """Expose one unit only after the whole initial-plus-refinement operation."""
    if report_status == "RUNNING":
        return 0
    if report_status == "COMPLETE":
        return 1
    raise ValueError("Unknown E1 evaluator report status")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class EvaluationProgress:
    def __init__(self, path: Path, run_id: str, run_spec_sha256: str) -> None:
        if not path.is_absolute():
            raise ValueError("BIOHUB_PROGRESS_PATH must be absolute")
        if len(run_spec_sha256) != 64 or any(c not in "0123456789abcdef" for c in run_spec_sha256):
            raise ValueError("Run-spec identity must be a lowercase SHA-256")
        self.path = path
        self.run_id = run_id
        self.run_spec_sha256 = run_spec_sha256
        self.completed_units = 0
        self._lock = threading.Lock()

    def write(self, *, error: str | None = None) -> None:
        with self._lock:
            atomic_json(
                self.path,
                {
                    "run_id": self.run_id,
                    "run_spec_sha256": self.run_spec_sha256,
                    "completed_units": self.completed_units,
                    "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "error": error,
                },
            )

    @contextmanager
    def heartbeat(self):
        stopped = threading.Event()

        def emit() -> None:
            while not stopped.wait(30):
                self.write()

        thread = threading.Thread(target=emit, name="e1-evaluator-heartbeat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=2)


@contextmanager
def enforce_wall_deadline(deadline_at: float):
    remaining = deadline_at - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("E1 evaluation wall-time cap reached")
    can_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    previous_handler = None
    if can_alarm:
        previous_handler = signal.getsignal(signal.SIGALRM)

        def timeout_handler(signum, frame):
            del signum, frame
            raise TimeoutError("E1 evaluation wall-time cap reached")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        if can_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class GridCell:
    precision_variant: str
    extraction_variant: str
    threshold_probability: float
    radius_um: float
    stage: str = "initial"

    @property
    def cell_id(self) -> str:
        return (
            f"{self.stage}:{self.precision_variant}:{self.extraction_variant}:"
            f"p={self.threshold_probability:.17g}:r={self.radius_um:.17g}"
        )


def _integer_histogram(values: Sequence[int]) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def _finite_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    if not np.isfinite(array).all():
        raise ValueError("Diagnostic distribution contains a nonfinite value")
    return {
        "count": len(array),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def connected_plateau_candidates_fast(
    scores: np.ndarray,
    *,
    threshold: float,
    scale_zyx_um: Sequence[float],
) -> tuple[list[PlateauCandidate], int]:
    """Find exact equal-score plateaus without labeling once per singleton value."""
    from scipy.ndimage import label, maximum_filter

    values = np.asarray(scores)
    scale = np.asarray(scale_zyx_um, dtype=np.float64)
    if values.dtype != np.float32 or values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Plateau input must be finite float32 ZYX")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Plateau scale must contain three positive finite values")
    if not math.isfinite(threshold):
        raise ValueError("Plateau threshold must be finite")
    maxima = (values >= threshold) & (
        values == maximum_filter(values, size=3, mode="constant", cval=-np.inf)
    )
    raw_maxima = int(maxima.sum())
    tied = np.zeros(values.shape, dtype=bool)
    # Visit one orientation from every 26-neighbor pair, then mark both ends.
    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) > (0, 0, 0)
    ]
    for offset in offsets:
        left = tuple(
            slice(0, size - delta) if delta > 0 else slice(-delta, size)
            for size, delta in zip(values.shape, offset)
        )
        right = tuple(
            slice(delta, size) if delta > 0 else slice(0, size + delta)
            for size, delta in zip(values.shape, offset)
        )
        equal_pair = maxima[left] & maxima[right] & (values[left] == values[right])
        tied[left] |= equal_pair
        tied[right] |= equal_pair
    candidates = [
        PlateauCandidate(tuple(map(int, coord)), float(values[tuple(coord)]), 1, tuple(coord), 0.0)
        for coord in np.argwhere(maxima & ~tied)
    ]
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for value in np.unique(values[tied]):
        components, count = label(tied & (values == value), structure=structure)
        for component_id in range(1, count + 1):
            coords = np.argwhere(components == component_id)
            centroid = coords.mean(axis=0)
            physical_delta = (coords - centroid) * scale
            squared = np.einsum("ij,ij->i", physical_delta, physical_delta)
            best = float(squared.min())
            representative = min(map(tuple, coords[np.flatnonzero(squared == best)].tolist()))
            candidates.append(
                PlateauCandidate(
                    tuple(map(int, representative)),
                    float(value),
                    len(coords),
                    tuple(float(item) for item in centroid),
                    math.sqrt(best),
                )
            )
    candidates.sort(key=lambda candidate: (-candidate.logit, candidate.coord_zyx))
    return candidates, raw_maxima


def physical_nms_fast(
    candidates: Sequence[PlateauCandidate],
    *,
    scale_zyx_um: Sequence[float],
    radius_um: float,
) -> list[PlateauCandidate]:
    """Use the spatial-bin NMS while preserving score/coordinate tie order."""
    ordered = sorted(candidates, key=lambda candidate: (-candidate.logit, candidate.coord_zyx))
    keep = anisotropic_nms(
        [candidate.coord_zyx for candidate in ordered],
        [candidate.logit for candidate in ordered],
        scale=tuple(float(value) for value in scale_zyx_um),
        radius_um=radius_um,
    )
    return [ordered[index] for index in keep]


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def exact_initial_grid(plan: Mapping[str, Any]) -> tuple[GridCell, ...]:
    """Expand and verify the frozen 3 x 2 x 3 x 3 initial grid."""
    grid = _require_mapping(plan.get("initial_grid"), "initial_grid")
    precision = [item["id"] for item in plan.get("precision_variants", [])]
    extraction = [item["id"] for item in plan.get("extraction_variants", [])]
    thresholds = grid.get("thresholds_probability")
    radii = grid.get("radii_um")
    if (
        precision != ["amp_native_sigmoid", "amp_logits_fp32_sigmoid", "full_fp32"]
        or extraction != ["legacy_voxel_maxima", "connected_plateau"]
        or thresholds != [0.3, 0.5, 0.7]
        or radii != [2.0, 3.0, 4.0]
        or grid.get("total_initial_configurations") != 54
        or grid.get("maximum_allowed_per_variant") != 15
    ):
        raise ValueError("E1 initial grid differs from the preregistered 54-cell contract")
    cells = tuple(
        GridCell(p, e, float(t), float(r))
        for p in precision
        for e in extraction
        for t in thresholds
        for r in radii
    )
    if len(cells) != 54 or len({cell.cell_id for cell in cells}) != 54:
        raise ValueError("E1 initial grid is incomplete or duplicated")
    return cells


def _candidate_points_legacy(
    probability: np.ndarray,
    *,
    raw_shape: Sequence[int],
    raw_scale_zyx_um: Sequence[float],
    xy_stride: int,
    threshold: float,
    radius_um: float,
    max_nodes: int,
) -> dict[str, Any]:
    from scipy.ndimage import maximum_filter

    maxima = (probability >= threshold) & (
        probability == maximum_filter(probability, size=3, mode="nearest")
    )
    model_points = np.argwhere(maxima)
    scores = probability[tuple(model_points.T)] if len(model_points) else np.empty(0)
    candidate_limit = max_nodes * 10
    order = np.argsort(-scores, kind="stable")[:candidate_limit]
    bounded = model_points[order]
    raw_points = model_to_raw_points(bounded, xy_stride)
    raw_points = np.clip(np.rint(raw_points), 0, np.asarray(raw_shape) - 1).astype(int)
    kept = anisotropic_nms(
        raw_points.tolist(),
        scores[order].tolist(),
        scale=tuple(float(v) for v in raw_scale_zyx_um),
        radius_um=radius_um,
    )
    retained = raw_points[kept[:max_nodes]] if kept else np.empty((0, 3), dtype=int)
    model_scale = np.asarray(raw_scale_zyx_um, dtype=np.float64) * np.asarray(
        [1.0, xy_stride, xy_stride]
    )
    plateaus, _ = connected_plateau_candidates_fast(
        probability, threshold=threshold, scale_zyx_um=model_scale
    )
    return {
        "points_raw_zyx": retained,
        "extraction_representative_kind": "legacy_equal_maximum_voxel",
        "raw_local_maximum_voxels": int(maxima.sum()),
        "connected_plateau_count": len(plateaus),
        "plateau_voxel_count_distribution": _integer_histogram(
            [candidate.plateau_voxels for candidate in plateaus]
        ),
        "plateau_representative_displacement_um": _finite_summary(
            [candidate.representative_displacement_um for candidate in plateaus]
        ),
        "pre_cap_candidate_count": len(kept),
        "candidate_pool_truncated": len(model_points) > candidate_limit,
        "capped": len(model_points) > candidate_limit or len(kept) > max_nodes,
    }


def _candidate_points_plateau(
    probability: np.ndarray,
    *,
    raw_shape: Sequence[int],
    raw_scale_zyx_um: Sequence[float],
    xy_stride: int,
    threshold: float,
    radius_um: float,
    max_nodes: int,
) -> dict[str, Any]:
    model_scale = np.asarray(raw_scale_zyx_um, dtype=np.float64) * np.asarray(
        [1.0, xy_stride, xy_stride]
    )
    candidates, raw_maxima = connected_plateau_candidates_fast(
        probability,
        threshold=threshold,
        scale_zyx_um=model_scale,
    )
    nms_candidates = physical_nms_fast(candidates, scale_zyx_um=model_scale, radius_um=radius_um)
    retained_candidates = nms_candidates[:max_nodes]
    model_points = np.asarray(
        [candidate.coord_zyx for candidate in retained_candidates], dtype=np.float32
    ).reshape(-1, 3)
    raw_points = model_to_raw_points(model_points, xy_stride)
    raw_points = np.clip(np.rint(raw_points), 0, np.asarray(raw_shape) - 1).astype(int)
    return {
        "points_raw_zyx": raw_points,
        "extraction_representative_kind": "connected_plateau_physical_centroid",
        "raw_local_maximum_voxels": raw_maxima,
        "connected_plateau_count": len(candidates),
        "plateau_voxel_count_distribution": _integer_histogram(
            [c.plateau_voxels for c in candidates]
        ),
        "plateau_representative_displacement_um": _finite_summary(
            [c.representative_displacement_um for c in candidates]
        ),
        "pre_cap_candidate_count": len(nms_candidates),
        "candidate_pool_truncated": False,
        "capped": len(nms_candidates) > max_nodes,
    }


def diagnostic_match_points(
    predicted_raw_zyx: np.ndarray,
    truth_raw_zyx: np.ndarray,
    *,
    raw_scale_zyx_um: Sequence[float],
    max_distance_um: float,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Apply the repo's explicitly diagnostic matcher without all-pairs Python loops.

    This reproduces ``biohub_ct.metrics.edge``: exact dynamic programming when
    both sets have at most 20 nodes, then distance/pred-index/truth-index greedy
    assignment. It is not the pinned organizer matcher and cannot support an
    official-score claim.
    """
    from scipy.spatial import cKDTree

    predicted = np.asarray(predicted_raw_zyx, dtype=np.float64).reshape(-1, 3)
    truth = np.asarray(truth_raw_zyx, dtype=np.float64).reshape(-1, 3)
    scale = np.asarray(raw_scale_zyx_um, dtype=np.float64)
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("Matching coordinates must be finite")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Matching scale must contain three positive finite values")
    if not math.isfinite(max_distance_um) or max_distance_um <= 0:
        raise ValueError("Matching radius must be positive and finite")
    if not len(predicted) or not len(truth):
        return [], []
    predicted_physical = predicted * scale
    truth_physical = truth * scale
    if len(predicted) <= 20 and len(truth) <= 20:
        from biohub_ct.data.schema import Graph, Node
        from biohub_ct.metrics.edge import match_nodes

        pred_graph = Graph(Node(i, 0, *map(int, p)) for i, p in enumerate(predicted))
        truth_graph = Graph(Node(i, 0, *map(int, p)) for i, p in enumerate(truth))
        matches = match_nodes(
            pred_graph,
            truth_graph,
            scale=tuple(float(v) for v in scale),
            max_distance=max_distance_um,
        )
        pairs = sorted(matches.pred_to_gt.items())
    else:
        tree = cKDTree(truth_physical)
        neighbors = tree.query_ball_point(predicted_physical, r=max_distance_um)
        candidates = []
        for pred_index, truth_indices in enumerate(neighbors):
            if not truth_indices:
                continue
            indices = np.asarray(truth_indices, dtype=np.int64)
            distances = np.linalg.norm(
                truth_physical[indices] - predicted_physical[pred_index], axis=1
            )
            candidates.extend(
                (float(distance), pred_index, int(truth_index))
                for truth_index, distance in zip(indices, distances)
                if distance <= max_distance_um
            )
        candidates.sort(key=lambda value: (value[0], value[1], value[2]))
        used_predicted: set[int] = set()
        used_truth: set[int] = set()
        pairs = []
        for _, pred_index, truth_index in candidates:
            if pred_index in used_predicted or truth_index in used_truth:
                continue
            pairs.append((pred_index, truth_index))
            used_predicted.add(pred_index)
            used_truth.add(truth_index)
        pairs.sort()
    distances = [
        float(np.linalg.norm(predicted_physical[pred] - truth_physical[gt])) for pred, gt in pairs
    ]
    return pairs, sorted(distances)


def frame_metrics(
    probability: np.ndarray,
    truth_points_raw_zyx: np.ndarray,
    *,
    raw_shape: Sequence[int],
    raw_scale_zyx_um: Sequence[float],
    xy_stride: int,
    extraction_variant: str,
    threshold: float,
    radius_um: float,
    max_nodes: int = 2000,
    match_distance_um: float = MAX_MATCH_DISTANCE_MICRONS,
) -> dict[str, Any]:
    """Compute deterministic frame metrics; temporal metrics are intentionally absent."""
    values = np.asarray(probability)
    truth = np.asarray(truth_points_raw_zyx, dtype=np.float64).reshape(-1, 3)
    if values.dtype != np.float32 or values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Probability field must be finite float32 ZYX")
    if extraction_variant == "legacy_voxel_maxima":
        extracted = _candidate_points_legacy(
            values,
            raw_shape=raw_shape,
            raw_scale_zyx_um=raw_scale_zyx_um,
            xy_stride=xy_stride,
            threshold=threshold,
            radius_um=radius_um,
            max_nodes=max_nodes,
        )
    elif extraction_variant == "connected_plateau":
        extracted = _candidate_points_plateau(
            values,
            raw_shape=raw_shape,
            raw_scale_zyx_um=raw_scale_zyx_um,
            xy_stride=xy_stride,
            threshold=threshold,
            radius_um=radius_um,
            max_nodes=max_nodes,
        )
    else:
        raise ValueError("Unknown extraction variant")
    points = extracted.pop("points_raw_zyx")
    pairs, distances = diagnostic_match_points(
        points,
        truth,
        raw_scale_zyx_um=raw_scale_zyx_um,
        max_distance_um=match_distance_um,
    )
    pre_cap = extracted["pre_cap_candidate_count"]
    retained = len(points)
    return {
        **extracted,
        "predicted_node_count": retained,
        "annotated_node_count": len(truth),
        "matching_backend": "local_diagnostic_dp_le20_else_greedy_not_organizer_official",
        "matched_annotated_node_count": len(pairs),
        "annotated_recall": len(pairs) / len(truth) if len(truth) else None,
        "matched_localization_um": distances,
        "clipped_candidate_fraction": (
            None
            if extracted["candidate_pool_truncated"]
            else (max(0, pre_cap - retained) / pre_cap if pre_cap else 0.0)
        ),
        "clipped_candidate_fraction_reason": (
            "pre_nms_candidate_pool_truncated" if extracted["candidate_pool_truncated"] else None
        ),
        "predicted_points_raw_zyx": points.tolist(),
    }


def aggregate_frame_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    annotated = sum(int(row["annotated_node_count"]) for row in rows)
    matched = sum(int(row["matched_annotated_node_count"]) for row in rows)
    distances = [float(v) for row in rows for v in row["matched_localization_um"]]
    predicted = sum(int(row["predicted_node_count"]) for row in rows)
    pre_cap = sum(int(row["pre_cap_candidate_count"]) for row in rows)
    plateau_histogram: dict[str, int] = {}
    for row in rows:
        for size, count in row["plateau_voxel_count_distribution"].items():
            plateau_histogram[size] = plateau_histogram.get(size, 0) + int(count)
    pool_truncated = sum(bool(r["candidate_pool_truncated"]) for r in rows)
    return {
        "frame_count": len(rows),
        "predicted_node_count": predicted,
        "annotated_node_count": annotated,
        "matched_annotated_node_count": matched,
        "annotated_recall": matched / annotated if annotated else None,
        "matched_localization_median_um": float(np.median(distances)) if distances else None,
        "matched_localization_p95_um": float(np.quantile(distances, 0.95)) if distances else None,
        "raw_local_maximum_voxels": sum(int(r["raw_local_maximum_voxels"]) for r in rows),
        "connected_plateau_count": sum(int(r["connected_plateau_count"]) for r in rows),
        "plateau_voxel_count_distribution": plateau_histogram,
        "plateau_representative_displacement_um": {
            "aggregation": "per_frame_summaries_preserved_below",
            "per_frame": [r["plateau_representative_displacement_um"] for r in rows],
        },
        "pre_cap_candidate_count": pre_cap,
        "clipped_candidate_fraction": (
            None if pool_truncated else (max(0, pre_cap - predicted) / pre_cap if pre_cap else 0.0)
        ),
        "clipped_candidate_fraction_reason": (
            "one_or_more_frames_pre_nms_candidate_pool_truncated" if pool_truncated else None
        ),
        "per_frame_cap_rate": sum(bool(r["capped"]) for r in rows) / len(rows),
        "candidate_pool_truncated_frames": pool_truncated,
    }


def choose_refinement(
    initial_rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> tuple[GridCell, ...]:
    """Choose at most one preregistered midpoint stage from complete initial rows."""
    if len(initial_rows) != 54 or any(row.get("status") != "COMPLETE" for row in initial_rows):
        raise ValueError("Refinement requires all 54 complete initial rows")
    required_finite_metrics = (
        "annotated_recall",
        "matched_localization_median_um",
        "matched_localization_p95_um",
        "predicted_node_count",
        "clipped_candidate_fraction",
        "per_frame_cap_rate",
    )
    for row in initial_rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            return ()
        values = [metrics.get(field) for field in required_finite_metrics]
        if any(
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ) or metrics.get("candidate_pool_truncated_frames"):
            return ()
    control = next(
        row
        for row in initial_rows
        if row["precision_variant"] == "amp_native_sigmoid"
        and row["extraction_variant"] == "legacy_voxel_maxima"
        and row["threshold_probability"] == 0.3
        and row["radius_um"] == 3.0
    )
    control_count = int(control["metrics"]["predicted_node_count"])
    control_recall = control["metrics"]["annotated_recall"]
    control_cap = float(control["metrics"]["per_frame_cap_rate"])
    if not control_count or control_recall is None:
        return ()
    eligible = []
    for row in initial_rows:
        metrics = row["metrics"]
        recall = metrics["annotated_recall"]
        if (
            recall is not None
            and control_recall - recall <= 0.01
            and float(metrics["per_frame_cap_rate"]) <= control_cap
        ):
            reduction = (control_count - int(metrics["predicted_node_count"])) / control_count
            eligible.append((row, reduction))
    if not eligible:
        return ()
    eligible.sort(
        key=lambda item: (
            -float(item[0]["metrics"]["annotated_recall"]),
            int(item[0]["metrics"]["predicted_node_count"]),
            -float(item[0]["threshold_probability"]),
            float(item[0]["radius_um"]),
            str(item[0]["precision_variant"]),
            str(item[0]["extraction_variant"]),
        )
    )
    anchor, reduction = eligible[0]
    target = float(plan["advancement_rule"]["count_ratio_relative_reduction_min"])
    if reduction >= target:
        return ()
    thresholds = list(map(float, plan["initial_grid"]["thresholds_probability"]))
    radii = list(map(float, plan["initial_grid"]["radii_um"]))
    t, r = float(anchor["threshold_probability"]), float(anchor["radius_um"])
    candidates = []
    ti, ri = thresholds.index(t), radii.index(r)
    for neighbor in (ti - 1, ti + 1):
        if 0 <= neighbor < len(thresholds):
            candidates.append((0.5 * (t + thresholds[neighbor]), r))
    for neighbor in (ri - 1, ri + 1):
        if 0 <= neighbor < len(radii):
            candidates.append((t, 0.5 * (r + radii[neighbor])))
    cells = tuple(
        GridCell(
            str(anchor["precision_variant"]),
            str(anchor["extraction_variant"]),
            threshold,
            radius,
            "refinement_1",
        )
        for threshold, radius in dict.fromkeys(candidates)
    )
    maximum = int(plan["refinement_policy"]["maximum_configurations"])
    if len(cells) > maximum or int(plan["refinement_policy"]["maximum_stages"]) != 1:
        raise ValueError("Refinement exceeds the preregistered single-stage bound")
    return cells


def unavailable_metrics() -> dict[str, dict[str, Any]]:
    return {
        "estimated_node_count": {"value": None, "reason": TEMPORAL_MISSING_REASON},
        "node_count_ratio": {"value": None, "reason": TEMPORAL_MISSING_REASON},
        "per_clip_cap_rate": {"value": None, "reason": TEMPORAL_MISSING_REASON},
        "fixed_linker_adjusted_edge_score": {
            "value": None,
            "reason": TEMPORAL_MISSING_REASON,
        },
        "overlay_review": {"value": None, "reason": MANUAL_MISSING_REASON},
    }


def make_partial_report(
    *,
    stage: str,
    fixed_high_cap_diagnostic: Mapping[str, Any],
    initial_rows: Sequence[Mapping[str, Any]],
    refinement_rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    completed_initial = len(initial_rows)
    if stage == "refinement_1" and completed_initial != 54:
        raise RuntimeError("Refinement checkpoint cannot precede all 54 initial cells")
    if completed_initial > 54:
        raise RuntimeError("Initial checkpoint exceeds the 54-cell grid")
    return {
        "schema_version": 1,
        "status": "RUNNING",
        "kind": "e1_preregistered_isolated_frame_diagnostic",
        **identity,
        "completed_initial_configurations": completed_initial,
        "completed_refinement_configurations": len(refinement_rows),
        "checkpoint_stage": stage,
        "fixed_high_cap_diagnostic": dict(fixed_high_cap_diagnostic),
        "initial_rows": list(initial_rows),
        "refinement_rows": list(refinement_rows),
        "advancement_eligible": False,
        "advancement_rejection_reasons": [
            "evaluation_incomplete",
            TEMPORAL_MISSING_REASON,
            MANUAL_MISSING_REASON,
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def verify_local_contract_files(
    plan: Mapping[str, Any], *, source_root: Path = ROOT
) -> dict[str, Mapping[str, Any]]:
    """Verify every local source/config byte identity consumed by evaluation."""
    for relative, expected_sha256 in plan["source_contract"]["files"].items():
        if hash_file(source_root / relative) != expected_sha256:
            raise ValueError("A frozen R5 capture source file changed")
    config_root = source_root / plan["config_contract"]["root"]
    configs = {}
    for name, identity in plan["config_contract"]["files"].items():
        path = config_root / name
        if hash_file(path) != identity["file_sha256"]:
            raise ValueError("An E1 config file byte hash changed")
        config = _read_json(path)
        if hash_json(config) != identity["canonical_json_sha256"]:
            raise ValueError("An E1 config canonical JSON hash changed")
        configs[name] = config
    if (
        configs["adapter-native-amp.json"].get("inference_config", {}).get("xy_stride") != 4
        or configs["adapter-full-fp32.json"].get("inference_config", {}).get("xy_stride") != 4
    ):
        raise ValueError("E1 evaluator requires the frozen XY stride of four")
    return configs


def verify_capture_plan_extension(plan: Mapping[str, Any], capture_plan: Mapping[str, Any]) -> None:
    """Allow exactly one additive evaluator_contract over the frozen R5 plan."""
    expanded = dict(plan)
    evaluator = expanded.pop("evaluator_contract", None)
    if not isinstance(evaluator, Mapping) or expanded != dict(capture_plan):
        raise ValueError(
            "Current E1 plan must equal the frozen R5 capture plan except evaluator_contract"
        )


def verify_evaluator_metric_contract(plan: Mapping[str, Any]) -> None:
    """Reject a plan that describes a different matching radius than the evaluator consumes."""
    evaluator = plan["evaluator_contract"]
    if (
        evaluator.get("annotation_match_distance_um") != MAX_MATCH_DISTANCE_MICRONS
        or evaluator.get("annotation_matching", {}).get("distance_um") != MAX_MATCH_DISTANCE_MICRONS
    ):
        raise ValueError("E1 annotation matching distance differs from evaluator semantics")


def verify_capture_run_index_contract(plan: Mapping[str, Any], *, source_root: Path = ROOT) -> None:
    """Bind all downloaded receipts to the admitted R5 run-spec index."""
    evaluator = plan["evaluator_contract"]
    contract = evaluator["frozen_capture_run_index"]
    index_path = source_root / contract["path"]
    if hash_file(index_path) != contract["file_sha256"]:
        raise ValueError("Frozen R5 cache run-spec index hash changed")
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) != contract["run_count"]:
        raise ValueError("Frozen R5 cache run-spec index coverage changed")
    indexed = {entry["run_id"]: entry["run_spec_sha256"] for entry in entries}
    expected_runs = evaluator["frozen_capture_runs"]
    planned_ids = {run["run_id"] for run in plan["worker_runs"]}
    if (
        set(indexed) != planned_ids
        or set(expected_runs) != planned_ids
        or indexed
        != {run_id: identity["run_spec_sha256"] for run_id, identity in expected_runs.items()}
    ):
        raise ValueError("Frozen capture run identities differ from the R5 index or plan")
    for identity in expected_runs.values():
        for field in (
            "run_spec_sha256",
            "result_manifest_sha256",
            "completion_sha256",
        ):
            value = identity.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("Frozen capture run identity has an invalid SHA-256")
        if not isinstance(identity.get("intent_id"), str) or not identity["intent_id"]:
            raise ValueError("Frozen capture run identity has no intent")
        if (
            isinstance(identity.get("fencing_token"), bool)
            or not isinstance(identity.get("fencing_token"), int)
            or identity["fencing_token"] < 0
        ):
            raise ValueError("Frozen capture run identity has an invalid fencing token")


def _expected_identity(
    plan: Mapping[str, Any], dataset: Mapping[str, Any], frame: Mapping[str, Any], capture: str
) -> CacheIdentity:
    suffix = "native-amp" if capture == "native-amp" else "full-fp32"
    files = plan["config_contract"]["files"]
    return CacheIdentity(
        model_sha256=plan["frozen_model"]["sha256"],
        source_sha256=plan["source_contract"]["combined_source_sha256"],
        config_sha256=files[f"adapter-{suffix}.json"]["canonical_json_sha256"],
        input_frame_sha256=frame["input_frame_typed_array_sha256"],
        transform_sha256=files["transform.json"]["canonical_json_sha256"],
        precision_sha256=files[f"precision-{suffix}.json"]["canonical_json_sha256"],
        tta_sha256=files["tta-disabled.json"]["canonical_json_sha256"],
        output_schema=plan["cache_contract"]["output_schema"],
    )


def _run_for(plan: Mapping[str, Any], capture: str, dataset_id: str) -> Mapping[str, Any]:
    return next(
        run
        for run in plan["worker_runs"]
        if run["precision_capture"] == capture and run["dataset_id"] == dataset_id
    )


def _artifact_path(root: Path, run: Mapping[str, Any], suffix: str) -> Path:
    matches = [name for name in run["run_spec_expected_artifacts"] if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one run artifact ending with {suffix}")
    return _resolve_downloaded_path(root, run, matches[0])


def _resolve_downloaded_path(root: Path, run: Mapping[str, Any], project_relative: str) -> Path:
    """Resolve either an in-place project tree or controller per-run download tree."""
    direct = root / project_relative
    nested = root / str(run["run_id"]) / project_relative
    existing = [path for path in (direct, nested) if path.exists()]
    if len(existing) != 1:
        raise ValueError("Expected one unambiguous downloaded artifact path")
    return existing[0]


def verify_downloaded_run(
    root: Path, run: Mapping[str, Any], expected_run: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Recheck the worker result and every controller-downloaded cache artifact."""
    run_id = str(run["run_id"])
    expected = run.get("run_spec_expected_artifacts")
    prefix = f"reports/campaign-workers/{run_id}/"
    if (
        not isinstance(expected, list)
        or len(expected) != 7
        or len(set(expected)) != 7
        or any(not isinstance(path, str) or not path.startswith(prefix) for path in expected)
    ):
        raise ValueError("Run does not declare the exact seven attempt-local artifacts")
    result_relative = f"{prefix}result.json"
    result_path = _resolve_downloaded_path(root, run, result_relative)
    result = _read_json(result_path)
    artifact_sha256 = result.get("artifact_sha256")
    if (
        result.get("run_id") != run_id
        or result.get("run_spec_sha256") != expected_run.get("run_spec_sha256")
        or result.get("intent_id") != expected_run.get("intent_id")
        or result.get("fencing_token") != expected_run.get("fencing_token")
        or result.get("status") != "COMPLETE"
        or result.get("completed_units") != 2
        or not isinstance(artifact_sha256, dict)
        or set(artifact_sha256) != set(expected)
        or hash_file(result_path) != expected_run.get("result_manifest_sha256")
    ):
        raise ValueError("Downloaded worker result is incomplete or has an artifact-set mismatch")
    for relative, expected_sha256 in artifact_sha256.items():
        if hash_file(_resolve_downloaded_path(root, run, relative)) != expected_sha256:
            raise ValueError("Downloaded worker artifact hash mismatch")
    identity = {
        "run_id": run_id,
        "run_spec_sha256": result.get("run_spec_sha256"),
    }
    progress = _read_json(_artifact_path(root, run, "/progress.json"))
    reference = _read_json(_artifact_path(root, run, "/cache-reference.json"))
    if (
        any(progress.get(key) != value for key, value in identity.items())
        or progress.get("completed_units") != 2
        or progress.get("error") is not None
        or any(reference.get(key) != value for key, value in identity.items())
        or reference.get("intent_id") != result.get("intent_id")
        or reference.get("fencing_token") != result.get("fencing_token")
    ):
        raise ValueError("Downloaded progress/reference identity differs from the result")
    index_path = _artifact_path(root, run, "/cache-index.json")
    if reference.get("cache_index", {}).get("sha256") != hash_file(index_path):
        raise ValueError("Cache reference does not bind the downloaded cache index")
    completion_relative = f"{prefix}worker/completion.json"
    completion_path = _resolve_downloaded_path(root, run, completion_relative)
    completion = _read_json(completion_path)
    completion_artifacts = completion.get("artifacts", {})
    worker_progress = completion.get("worker_progress", {})
    if (
        hash_file(completion_path) != expected_run.get("completion_sha256")
        or completion.get("status") != "COMPLETE"
        or completion.get("exit_code") != 0
        or completion.get("error") is not None
        or completion.get("run_id") != run_id
        or completion.get("run_spec_sha256") != expected_run.get("run_spec_sha256")
        or completion.get("fencing_token") != expected_run.get("fencing_token")
        or completion_artifacts.get("complete") is not True
        or completion_artifacts.get("completed_units") != 2
        or completion_artifacts.get("result_manifest_sha256") != hash_file(result_path)
        or completion_artifacts.get("artifact_sha256") != artifact_sha256
        or worker_progress.get("run_id") != run_id
        or worker_progress.get("run_spec_sha256") != expected_run.get("run_spec_sha256")
        or worker_progress.get("completed_units") != 2
        or worker_progress.get("error") is not None
    ):
        raise ValueError("Downloaded completion receipt differs from the admitted capture run")
    return result


def _verify_cache_index(
    root: Path, plan: Mapping[str, Any], run: Mapping[str, Any]
) -> Mapping[str, Any]:
    expected_run = plan["evaluator_contract"]["frozen_capture_runs"][run["run_id"]]
    verify_downloaded_run(root, run, expected_run)
    index = _read_json(_artifact_path(root, run, "/cache-index.json"))
    suffix = "native-amp" if run["precision_capture"] == "native-amp" else "full-fp32"
    config_files = plan["config_contract"]["files"]
    expected = {
        "status": "complete",
        "run_id": run["run_id"],
        "dataset_id": run["dataset_id"],
        "planned_frames": 2,
        "planned_frame_indices": [0, 50],
        "model_sha256": plan["frozen_model"]["sha256"],
        "source_sha256": plan["source_contract"]["combined_source_sha256"],
        "output_schema": plan["cache_contract"]["output_schema"],
        "config_sha256": config_files[f"adapter-{suffix}.json"]["canonical_json_sha256"],
        "transform_sha256": config_files["transform.json"]["canonical_json_sha256"],
        "precision_sha256": config_files[f"precision-{suffix}.json"]["canonical_json_sha256"],
        "tta_sha256": config_files["tta-disabled.json"]["canonical_json_sha256"],
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise ValueError(f"Cache index {key} differs from the E1 plan")
    if index.get("completed_frames") != 2 or len(index.get("frames", [])) != 2:
        raise ValueError("Cache index does not contain the exact two-frame panel")
    return index


def verify_cache_precision_identity(payload: Any, capture: str) -> None:
    """Bind both the precision label and native-logit dtype to the capture arm."""
    expected = {
        "native-amp": ("native_amp", "torch.float16"),
        "full-fp32": ("full_float32", "torch.float32"),
    }
    try:
        expected_precision, expected_dtype = expected[capture]
    except KeyError as exc:
        raise ValueError("Unknown E1 precision capture arm") from exc
    if payload.inference_precision != expected_precision:
        raise ValueError("E1 cache precision label differs from its precision arm")
    if payload.native_logit_dtype != expected_dtype:
        raise ValueError("E1 cache native dtype differs from its precision arm")


def verify_annotation_frame_identity(
    nodes: Sequence[Any], expected: Mapping[str, Any]
) -> np.ndarray:
    """Hash the exact ordered GEFF node IDs and coordinates consumed by metrics."""
    coordinates = np.asarray([node.coord for node in nodes], dtype=np.float64).reshape(-1, 3)
    node_ids = np.asarray([node.node_id for node in nodes], dtype=np.int64).reshape(-1)
    if (
        len(nodes) != expected.get("annotated_nodes")
        or hash_array(coordinates) != expected.get("coordinate_float64_zyx_sha256")
        or hash_array(node_ids) != expected.get("node_ids_int64_sha256")
    ):
        raise ValueError("Actual GEFF annotation identity differs from preregistration")
    return coordinates


def _load_probabilities(
    *,
    root: Path,
    data_root: Path,
    model_file: Path,
    plan: Mapping[str, Any],
    deadline_at: float,
) -> tuple[
    dict[tuple[str, int, str], np.ndarray], dict[tuple[str, int], np.ndarray], dict[str, float]
]:
    config_root = ROOT / plan["config_contract"]["root"]
    native_config = _read_json(config_root / "adapter-native-amp.json")
    adapter = create_adapter(native_config, model_file=model_file)
    fields: dict[tuple[str, int, str], np.ndarray] = {}
    truth_by_frame: dict[tuple[str, int], np.ndarray] = {}
    timings = {"cache_load_seconds": 0.0, "control_parity_seconds": 0.0}
    for dataset in plan["population"]["datasets"]:
        if time.monotonic() >= deadline_at:
            raise TimeoutError("E1 evaluation expired while loading caches")
        dataset_id = dataset["dataset_id"]
        volume = open_zarr_volume(data_root / f"{dataset_id}.zarr", require_complete_chunks=True)
        if list(volume.shape) != dataset["shape_tzyx"]:
            raise ValueError("Actual Zarr shape differs from preregistered metadata")
        if list(volume.scale) != dataset["scale_zyx_um"]:
            raise ValueError("Actual Zarr scale differs from preregistered metadata")
        truth_graph, _ = read_geff_graph(data_root / f"{dataset_id}.geff")
        truth_by_time = truth_graph.nodes_by_time()
        indices = {
            capture: _verify_cache_index(root, plan, _run_for(plan, capture, dataset_id))
            for capture in ("native-amp", "full-fp32")
        }
        for frame_spec in dataset["frames"]:
            if time.monotonic() >= deadline_at:
                raise TimeoutError("E1 evaluation expired while loading a frame")
            frame_index = int(frame_spec["frame"])
            raw = volume.read_frame(frame_index)
            if (
                list(raw.shape) != frame_spec["raw_shape_zyx"]
                or raw.dtype.str != frame_spec["raw_dtype"]
                or raw.nbytes != frame_spec["raw_bytes"]
            ):
                raise ValueError("Actual frame array contract differs from preregistration")
            if hash_array(raw) != frame_spec["input_frame_typed_array_sha256"]:
                raise ValueError("Actual frame hash differs from the preregistered panel")
            truth_nodes = truth_by_time.get(frame_index, [])
            annotation_contract = plan["evaluator_contract"]["frozen_annotation_frames"]
            truth_by_frame[(dataset_id, frame_index)] = verify_annotation_frame_identity(
                truth_nodes, annotation_contract[dataset_id][str(frame_index)]
            )
            payloads = {}
            for capture in ("native-amp", "full-fp32"):
                started = time.monotonic()
                index_frame = next(
                    row for row in indices[capture]["frames"] if row["frame"] == frame_index
                )
                cache_path = _artifact_path(
                    root, _run_for(plan, capture, dataset_id), f"/{index_frame['logits_path']}"
                )
                identity = _expected_identity(plan, dataset, frame_spec, capture)
                payload, manifest = load_learned_tile_cache(cache_path, expected_identity=identity)
                if (
                    hash_file(cache_manifest_path(cache_path))
                    != index_frame["manifest_file_sha256"]
                ):
                    raise ValueError("Cache sidecar hash differs from its index")
                if manifest["file_sha256"] != index_frame["logits_file_sha256"]:
                    raise ValueError("Cache payload hash differs from its index")
                if payload.inference_device_type != "cuda":
                    raise ValueError("E1 caches must preserve CUDA device identity")
                verify_cache_precision_identity(payload, capture)
                if payload.normalization["normalized_frame_sha256"] != hash_array(
                    normalize_frame(raw)
                ):
                    raise ValueError("Cached normalization differs from the actual frame")
                payloads[capture] = payload
                timings["cache_load_seconds"] += time.monotonic() - started
            started = time.monotonic()
            production = frame_probabilities(
                adapter.model, raw, adapter.config, deadline_at=deadline_at
            )
            native = reconstruct_probabilities(payloads["native-amp"], activation="native")
            if not np.array_equal(native, production):
                raise ValueError("Native cache replay is not bitwise equal to production")
            production_nodes, production_capped = probability_nodes(
                production,
                raw.shape,
                volume.scale,
                frame_index,
                0,
                replace(adapter.config, threshold=0.3, nms_radius_um=3.0),
            )
            replay_nodes = _candidate_points_legacy(
                native,
                raw_shape=raw.shape,
                raw_scale_zyx_um=volume.scale,
                xy_stride=adapter.config.xy_stride,
                threshold=0.3,
                radius_um=3.0,
                max_nodes=adapter.config.max_nodes_per_frame,
            )
            if (
                replay_nodes["points_raw_zyx"].tolist()
                != [list(node.coord) for node in production_nodes]
                or replay_nodes["capped"] != production_capped
            ):
                raise ValueError("Frozen-control cache extraction differs from probability_nodes")
            fields[(dataset_id, frame_index, "amp_native_sigmoid")] = native
            fields[(dataset_id, frame_index, "amp_logits_fp32_sigmoid")] = (
                reconstruct_probabilities(payloads["native-amp"], activation="float32")
            )
            fields[(dataset_id, frame_index, "full_fp32")] = reconstruct_probabilities(
                payloads["full-fp32"], activation="float32"
            )
            timings["control_parity_seconds"] += time.monotonic() - started
    return fields, truth_by_frame, timings


def _evaluate_cells(
    cells: Iterable[GridCell],
    *,
    plan: Mapping[str, Any],
    fields: Mapping[tuple[str, int, str], np.ndarray],
    truth: Mapping[tuple[str, int], np.ndarray],
    deadline_at: float,
    checkpoint: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        if time.monotonic() >= deadline_at:
            raise TimeoutError("E1 evaluation expired before the next grid cell")
        started = time.monotonic()
        frame_rows = []
        for dataset in plan["population"]["datasets"]:
            dataset_id = dataset["dataset_id"]
            for frame_spec in dataset["frames"]:
                if time.monotonic() >= deadline_at:
                    raise TimeoutError("E1 evaluation expired inside a grid cell")
                frame_index = int(frame_spec["frame"])
                metrics = frame_metrics(
                    fields[(dataset_id, frame_index, cell.precision_variant)],
                    truth[(dataset_id, frame_index)],
                    raw_shape=frame_spec["raw_shape_zyx"],
                    raw_scale_zyx_um=dataset["scale_zyx_um"],
                    xy_stride=4,
                    extraction_variant=cell.extraction_variant,
                    threshold=cell.threshold_probability,
                    radius_um=cell.radius_um,
                )
                predicted_points = np.asarray(
                    metrics.pop("predicted_points_raw_zyx"), dtype=np.int64
                ).reshape(-1, 3)
                metrics["predicted_points_raw_zyx_sha256"] = hash_array(predicted_points)
                frame_rows.append({"dataset_id": dataset_id, "frame": frame_index, **metrics})
        aggregate = aggregate_frame_rows(frame_rows)
        rows.append(
            {
                **asdict(cell),
                "cell_id": cell.cell_id,
                "status": "COMPLETE",
                "coverage": "exact_eight_preregistered_isolated_frames",
                "metrics": aggregate,
                "unavailable_metrics": unavailable_metrics(),
                "frame_rows": frame_rows,
                "extraction_seconds": time.monotonic() - started,
            }
        )
        if checkpoint is not None:
            checkpoint(rows)
    return rows


def run_fixed_high_cap_diagnostic(
    plan: Mapping[str, Any],
    fields: Mapping[tuple[str, int, str], np.ndarray],
    *,
    deadline_at: float,
) -> dict[str, Any]:
    """Expose legacy-control truncation on one fixed frame without another forward."""
    contract = plan["evaluator_contract"]["fixed_high_cap_diagnostic"]
    expected_contract = {
        "dataset_id": "44b6_0113de3b",
        "frame": 0,
        "precision_variant": "amp_native_sigmoid",
        "extraction_variant": "legacy_voxel_maxima",
        "threshold_probability": 0.3,
        "radius_um": 3.0,
        "default_max_nodes": 2000,
        "high_max_nodes": 262144,
        "probability_shape_zyx": [64, 64, 64],
        "role": "fixed_truncation_diagnostic_not_grid_search_or_advancement",
        "selection_effect": "none",
    }
    if contract != expected_contract:
        raise ValueError("High-cap diagnostic frame differs from preregistration")
    dataset = next(
        item
        for item in plan["population"]["datasets"]
        if item["dataset_id"] == contract["dataset_id"]
    )
    frame_spec = next(item for item in dataset["frames"] if item["frame"] == contract["frame"])
    if time.monotonic() >= deadline_at:
        raise TimeoutError("E1 evaluation expired before the fixed high-cap diagnostic")
    dataset_id = contract["dataset_id"]
    frame = contract["frame"]
    probability = fields[(dataset_id, frame, contract["precision_variant"])]
    if list(probability.shape) != contract["probability_shape_zyx"]:
        raise ValueError("Fixed high-cap diagnostic probability geometry changed")
    default_cap = contract["default_max_nodes"]
    high_cap = contract["high_max_nodes"]
    if high_cap != int(probability.size):
        raise ValueError("Fixed high-cap diagnostic no longer covers every probability voxel")
    common = {
        "raw_shape": frame_spec["raw_shape_zyx"],
        "raw_scale_zyx_um": dataset["scale_zyx_um"],
        "xy_stride": 4,
        "threshold": contract["threshold_probability"],
        "radius_um": contract["radius_um"],
    }
    started = time.monotonic()
    default = _candidate_points_legacy(probability, max_nodes=default_cap, **common)
    if time.monotonic() >= deadline_at:
        raise TimeoutError("E1 evaluation expired during the fixed high-cap diagnostic")
    high = _candidate_points_legacy(probability, max_nodes=high_cap, **common)
    high_incomplete = bool(high["candidate_pool_truncated"] or high["capped"])
    return {
        "status": "INCOMPLETE_TRUNCATED" if high_incomplete else "COMPLETE",
        "role": contract["role"],
        "selection_effect": contract["selection_effect"],
        "dataset_id": dataset_id,
        "frame": frame,
        "precision_variant": contract["precision_variant"],
        "extraction_variant": contract["extraction_variant"],
        "threshold_probability": contract["threshold_probability"],
        "radius_um": contract["radius_um"],
        "default_max_nodes": default_cap,
        "high_max_nodes": high_cap,
        "default_predicted_node_count": len(default["points_raw_zyx"]),
        "high_cap_predicted_node_count": len(high["points_raw_zyx"]),
        "default_pre_cap_candidate_count": default["pre_cap_candidate_count"],
        "high_cap_pre_cap_candidate_count": high["pre_cap_candidate_count"],
        "default_candidate_pool_truncated": default["candidate_pool_truncated"],
        "high_cap_candidate_pool_truncated": high["candidate_pool_truncated"],
        "default_capped": default["capped"],
        "high_cap_capped": high["capped"],
        "default_points_raw_zyx_sha256": hash_array(
            np.asarray(default["points_raw_zyx"], dtype=np.int64).reshape(-1, 3)
        ),
        "high_cap_points_raw_zyx_sha256": hash_array(
            np.asarray(high["points_raw_zyx"], dtype=np.int64).reshape(-1, 3)
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


def evaluate(
    plan_path: Path,
    artifact_root: Path,
    data_root: Path,
    model_file: Path,
    *,
    max_wall_seconds: float,
    checkpoint: Callable[
        [str, Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        None,
    ]
    | None = None,
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    if plan.get("status") != "PREREGISTERED_NOT_EXECUTED":
        raise ValueError("E1 plan status is not the frozen preregistered state")
    verify_local_contract_files(plan)
    if hash_file(model_file) != plan["frozen_model"]["sha256"]:
        raise ValueError("Model hash differs from the frozen E1 checkpoint")
    evaluator = plan.get("evaluator_contract", {})
    verify_evaluator_metric_contract(plan)
    if evaluator.get("file_sha256") != hash_file(Path(__file__)):
        raise ValueError("Evaluator source differs from the additive plan contract")
    capture_contract = evaluator.get("frozen_capture_plan", {})
    capture_path = ROOT / str(capture_contract.get("path", ""))
    if not capture_path.is_file() or hash_file(capture_path) != capture_contract.get("sha256"):
        raise ValueError("Frozen R5 capture plan identity differs from the evaluator contract")
    capture_plan = _read_json(capture_path)
    verify_capture_plan_extension(plan, capture_plan)
    verify_capture_run_index_contract(plan)
    if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("Evaluation wall-time bound must be positive and finite")
    started = time.monotonic()
    deadline_at = started + max_wall_seconds
    fields, truth, timings = _load_probabilities(
        root=artifact_root,
        data_root=data_root,
        model_file=model_file,
        plan=plan,
        deadline_at=deadline_at,
    )
    high_cap_diagnostic = run_fixed_high_cap_diagnostic(plan, fields, deadline_at=deadline_at)
    if checkpoint is not None:
        checkpoint("fixed_high_cap", high_cap_diagnostic, (), ())

    def initial_checkpoint(rows: Sequence[Mapping[str, Any]]) -> None:
        if checkpoint is not None:
            checkpoint("initial", high_cap_diagnostic, rows, ())

    initial = _evaluate_cells(
        exact_initial_grid(plan),
        plan=plan,
        fields=fields,
        truth=truth,
        deadline_at=deadline_at,
        checkpoint=initial_checkpoint,
    )
    if len(initial) != 54:
        raise RuntimeError("E1 initial grid did not complete all 54 configurations")
    refinement_cells = choose_refinement(initial, plan)

    def refinement_checkpoint(rows: Sequence[Mapping[str, Any]]) -> None:
        if checkpoint is not None:
            checkpoint("refinement_1", high_cap_diagnostic, initial, rows)

    refinement = _evaluate_cells(
        refinement_cells,
        plan=plan,
        fields=fields,
        truth=truth,
        deadline_at=deadline_at,
        checkpoint=refinement_checkpoint,
    )
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "kind": "e1_preregistered_isolated_frame_diagnostic",
        "plan_id": plan["plan_id"],
        "plan_file_sha256": hash_file(plan_path),
        "evaluator_file_sha256": hash_file(Path(__file__)),
        "model_sha256": hash_file(model_file),
        "population_evidence_class": plan["evidence_class"],
        "control_probability_bitwise_parity": True,
        "control_node_extraction_parity": True,
        "matching_backend": "local_diagnostic_dp_le20_else_greedy_not_organizer_official",
        "completed_initial_configurations": len(initial),
        "completed_refinement_configurations": len(refinement),
        "fixed_high_cap_diagnostic": high_cap_diagnostic,
        "initial_rows": initial,
        "refinement_rows": refinement,
        "refinement_stage_count": 1 if refinement else 0,
        "unavailable_metrics": unavailable_metrics(),
        "advancement_eligible": False,
        "advancement_rejection_reasons": [
            TEMPORAL_MISSING_REASON,
            MANUAL_MISSING_REASON,
            "diagnostic_reuse_is_not_heldout_or_release_evidence",
        ],
        "stage_runtime_seconds": {
            **timings,
            "initial_extraction": sum(row["extraction_seconds"] for row in initial),
            "refinement_extraction": sum(row["extraction_seconds"] for row in refinement),
            "total": time.monotonic() - started,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--run-id", default=os.environ.get("BIOHUB_RUN_ID"))
    parser.add_argument("--run-spec-sha256", default=os.environ.get("BIOHUB_RUN_SPEC_SHA256"))
    parser.add_argument("--intent-id", default=os.environ.get("BIOHUB_INTENT_ID"))
    parser.add_argument("--fencing-token", default=os.environ.get("BIOHUB_FENCING_TOKEN"))
    parser.add_argument("--attempt-dir", type=Path, default=os.environ.get("BIOHUB_ATTEMPT_DIR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_id or not args.run_spec_sha256 or not args.intent_id:
        raise ValueError("Run ID, run-spec SHA-256, and intent ID are required")
    try:
        fencing_token = int(args.fencing_token)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fencing token must be an integer") from exc
    if fencing_token < 0 or args.attempt_dir is None:
        raise ValueError("Nonnegative fencing token and attempt directory are required")
    attempt_dir = args.attempt_dir.resolve(strict=True)
    expected_attempt_root = (ROOT / "reports" / "campaign-workers").resolve()
    if not attempt_dir.is_relative_to(expected_attempt_root) or attempt_dir.name != args.run_id:
        raise ValueError("Attempt directory must be inside reports/campaign-workers")
    progress_raw = os.environ.get("BIOHUB_PROGRESS_PATH")
    if not progress_raw:
        raise ValueError("BIOHUB_PROGRESS_PATH is required")
    progress_path = Path(progress_raw).resolve()
    output = args.output_json.resolve()
    if not progress_path.is_relative_to(attempt_dir) or not output.is_relative_to(attempt_dir):
        raise ValueError("Progress and evaluation output must live inside the attempt")
    result_path = attempt_dir / "result.json"
    if result_path.exists():
        raise ValueError("Existing result.json cannot be reused")
    progress = EvaluationProgress(progress_path, args.run_id, args.run_spec_sha256)
    progress.write()
    plan_path = args.plan_json.resolve(strict=True)
    plan = _read_json(plan_path)

    def checkpoint(
        stage: str,
        fixed_high_cap_diagnostic: Mapping[str, Any],
        initial_rows: Sequence[Mapping[str, Any]],
        refinement_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        partial = make_partial_report(
            stage=stage,
            fixed_high_cap_diagnostic=fixed_high_cap_diagnostic,
            initial_rows=initial_rows,
            refinement_rows=refinement_rows,
            identity={
                "run_id": args.run_id,
                "run_spec_sha256": args.run_spec_sha256,
                "intent_id": args.intent_id,
                "fencing_token": fencing_token,
                "plan_id": plan["plan_id"],
                "plan_file_sha256": hash_file(plan_path),
                "evaluator_file_sha256": hash_file(Path(__file__)),
            },
        )
        atomic_json(output, partial)
        # One operational unit is the complete initial grid plus the optional
        # refinement. Per-configuration coverage lives in the checkpoint.
        progress.completed_units = operational_completed_units(partial["status"])
        progress.write()

    deadline_at = time.monotonic() + args.max_wall_seconds
    try:
        with progress.heartbeat(), enforce_wall_deadline(deadline_at):
            report = evaluate(
                plan_path,
                args.artifact_root.resolve(strict=True),
                args.data_root.resolve(strict=True),
                args.model_file.resolve(strict=True),
                max_wall_seconds=max(0.0, deadline_at - time.monotonic()),
                checkpoint=checkpoint,
            )
        if report["completed_initial_configurations"] != 54:
            raise RuntimeError("A COMPLETE E1 report requires exactly 54 initial cells")
        report.update(
            run_id=args.run_id,
            run_spec_sha256=args.run_spec_sha256,
            intent_id=args.intent_id,
            fencing_token=fencing_token,
        )
        atomic_json(output, report)
        progress.completed_units = operational_completed_units(report["status"])
        progress.write()
        artifacts = {
            output.relative_to(ROOT).as_posix(): hash_file(output),
            progress_path.relative_to(ROOT).as_posix(): hash_file(progress_path),
        }
        result = {
            "run_id": args.run_id,
            "run_spec_sha256": args.run_spec_sha256,
            "intent_id": args.intent_id,
            "fencing_token": fencing_token,
            "status": "COMPLETE",
            "completed_units": operational_completed_units(report["status"]),
            "artifact_sha256": artifacts,
        }
        atomic_json(result_path, result)
    except BaseException as exc:
        progress.write(error=f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
