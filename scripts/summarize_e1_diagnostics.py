"""Reaggregate the completed E1 report and export a diagnostic table and figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.evaluate_detection_diagnostics import aggregate_frame_rows, choose_refinement
from scripts.render_e1_diagnostic_overlays import validate_evaluation_report


def summarize(report_path: Path, plan_path: Path, output: Path):
    report = json.loads(report_path.read_text())
    plan = json.loads(plan_path.read_text())
    validate_evaluation_report(report, plan)
    expected_refinement = {cell.cell_id for cell in choose_refinement(report["initial_rows"], plan)}
    if {row["cell_id"] for row in report["refinement_rows"]} != expected_refinement:
        raise ValueError("Refinement differs from the preregistered rule")
    rows = report["initial_rows"] + report["refinement_rows"]
    for row in rows:
        recomputed = aggregate_frame_rows(row["frame_rows"])
        for key, value in recomputed.items():
            actual = row["metrics"][key]
            if isinstance(value, float):
                if not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Aggregate mismatch: " + key)
            elif actual != value:
                raise ValueError("Aggregate mismatch: " + key)
    control = next(
        row
        for row in rows
        if row["precision_variant"] == "amp_native_sigmoid"
        and row["extraction_variant"] == "legacy_voxel_maxima"
        and row["threshold_probability"] == 0.3
        and row["radius_um"] == 3
    )
    baseline = control["metrics"]
    fields = [
        "cell_id",
        "stage",
        "precision_variant",
        "extraction_variant",
        "threshold_probability",
        "radius_um",
        "predicted_node_count",
        "relative_predicted_count_reduction",
        "annotated_node_count",
        "matched_annotated_node_count",
        "annotated_recall",
        "matched_localization_median_um",
        "matched_localization_p95_um",
        "per_frame_cap_rate",
        "pre_cap_candidate_count",
        "clipped_candidate_fraction",
        "raw_local_maximum_voxels",
        "connected_plateau_count",
        "candidate_pool_truncated_frames",
        "extraction_seconds",
    ]
    flat = []
    for row in rows:
        combined = {
            **row,
            **row["metrics"],
            "relative_predicted_count_reduction": 1
            - row["metrics"]["predicted_node_count"] / baseline["predicted_node_count"],
        }
        flat.append({field: combined[field] for field in fields})
    acceptable = [
        row
        for row in flat
        if row["annotated_recall"] >= baseline["annotated_recall"] - 0.01
        and row["matched_localization_median_um"]
        <= baseline["matched_localization_median_um"] + 0.40625
        and row["matched_localization_p95_um"] <= baseline["matched_localization_p95_um"] + 1.625
        and row["per_frame_cap_rate"] <= baseline["per_frame_cap_rate"]
    ]
    lowest_count = min(acceptable, key=lambda row: (row["predicted_node_count"], row["cell_id"]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "e1-diagnostic-frontier.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    fixed = [
        row
        for row in flat
        if row["stage"] == "initial"
        and row["threshold_probability"] == 0.3
        and row["radius_um"] == 3
    ]
    summary = {
        "status": "INDEPENDENT_AGGREGATE_CHECK_PASS",
        "initial_rows": 54,
        "refinement_rows": len(report["refinement_rows"]),
        "annotated_nodes": baseline["annotated_node_count"],
        "matching_scope": report["matching_backend"],
        "fixed_control_metrics": baseline,
        "same_threshold_radius_comparisons": fixed,
        "lowest_count_with_diagnostic_recall_localization_cap_constraints": lowest_count,
        "twenty_percent_relative_predicted_count_reduction_achieved": lowest_count[
            "relative_predicted_count_reduction"
        ]
        >= 0.2,
        "estimated_node_count_ratio_available": False,
        "official_temporal_score_available": False,
        "advancement_eligible": False,
        "input_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    (output / "e1-diagnostic-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    colors = {
        "amp_native_sigmoid": "#2878b5",
        "amp_logits_fp32_sigmoid": "#da8530",
        "full_fp32": "#3f995b",
    }
    labels = {
        "amp_native_sigmoid": "AMP / native sigmoid",
        "amp_logits_fp32_sigmoid": "AMP / FP32 sigmoid",
        "full_fp32": "Full FP32",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for precision, color in colors.items():
        for extraction, marker in [("legacy_voxel_maxima", "o"), ("connected_plateau", "s")]:
            group = [
                row
                for row in flat
                if row["precision_variant"] == precision and row["extraction_variant"] == extraction
            ]
            axes[0].scatter(
                [row["predicted_node_count"] for row in group],
                [100 * row["per_frame_cap_rate"] for row in group],
                marker=marker,
                color=color,
                facecolors=color if extraction == "legacy_voxel_maxima" else "none",
                s=65,
                alpha=0.8,
                label=labels[precision]
                + (" / voxel" if extraction == "legacy_voxel_maxima" else " / plateau"),
            )
    axes[0].axvline(
        0.8 * baseline["predicted_node_count"],
        color="#b53c3c",
        linestyle=":",
        label="20% fewer than control",
    )
    axes[0].set(
        xlabel="Total retained predictions across 8 frames",
        ylabel="Frames hitting a node cap (%)",
        title="All 56 preregistered comparisons",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="lower right")
    if all(
        row["matched_annotated_node_count"] == row["annotated_node_count"] == 26 for row in flat
    ):
        axes[0].text(
            0.03,
            0.97,
            "All configurations match 26/26 sparse annotations.\nSeveral grid points overlap.",
            transform=axes[0].transAxes,
            va="top",
            fontsize=8,
        )
    short_labels = {
        "amp_native_sigmoid": "AMP\nnative sigmoid",
        "amp_logits_fp32_sigmoid": "AMP\nFP32 sigmoid",
        "full_fp32": "Full FP32",
    }
    names = [
        short_labels[row["precision_variant"]]
        + "\n"
        + ("voxel" if row["extraction_variant"] == "legacy_voxel_maxima" else "plateau")
        for row in fixed
    ]
    bars = axes[1].bar(
        range(len(fixed)),
        [row["predicted_node_count"] for row in fixed],
        color=[colors[row["precision_variant"]] for row in fixed],
    )
    axes[1].bar_label(bars, fontsize=8, padding=3)
    axes[1].set(
        xticks=range(len(fixed)),
        xticklabels=names,
        ylabel="Retained predictions",
        title="Fixed threshold 0.3 and NMS radius 3 um",
    )
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].set_ylim(0, baseline["predicted_node_count"] * 1.12)
    fig.suptitle(
        f"E1 diagnostic reuse: {baseline['annotated_node_count']} sparse annotations; no complete temporal clips",
        fontsize=12,
    )
    fig.savefig(output / "e1-diagnostic-frontier.png", dpi=180)
    fig.savefig(output / "e1-diagnostic-frontier.svg")
    plt.close(fig)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "lowest_count": lowest_count,
                "twenty_percent_relative_predicted_count_reduction_achieved": summary[
                    "twenty_percent_relative_predicted_count_reduction_achieved"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.report, args.plan, args.output)
