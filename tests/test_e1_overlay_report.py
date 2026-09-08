"""Reject falsely complete analysis input before visual evidence is rendered."""

from copy import deepcopy

import pytest

pytest.importorskip("PIL")
from scripts.evaluate_detection_diagnostics import exact_initial_grid
from scripts.render_e1_diagnostic_overlays import validate_evaluation_report


def complete_report():
    plan = {
        "precision_variants": [
            {"id": value}
            for value in ["amp_native_sigmoid", "amp_logits_fp32_sigmoid", "full_fp32"]
        ],
        "extraction_variants": [
            {"id": value} for value in ["legacy_voxel_maxima", "connected_plateau"]
        ],
        "initial_grid": {
            "thresholds_probability": [0.3, 0.5, 0.7],
            "radii_um": [2.0, 3.0, 4.0],
            "total_initial_configurations": 54,
            "maximum_allowed_per_variant": 15,
        },
        "population": {
            "datasets": [
                {"dataset_id": f"clip-{i}", "frames": [{"frame": 0}, {"frame": 50}]}
                for i in range(4)
            ]
        },
    }
    frames = [
        {"dataset_id": dataset["dataset_id"], "frame": frame["frame"]}
        for dataset in plan["population"]["datasets"]
        for frame in dataset["frames"]
    ]
    report = {
        "status": "COMPLETE",
        "completed_initial_configurations": 54,
        "fixed_high_cap_diagnostic": {"status": "COMPLETE"},
        "initial_rows": [
            {"cell_id": cell.cell_id, "status": "COMPLETE", "frame_rows": deepcopy(frames)}
            for cell in exact_initial_grid(plan)
        ],
    }
    return report, plan


def test_declared_completion_cannot_hide_missing_or_duplicated_grid_cells():
    report, plan = complete_report()
    validate_evaluation_report(report, plan)
    report["initial_rows"] = report["initial_rows"][:2]
    with pytest.raises(ValueError, match="grid coverage"):
        validate_evaluation_report(report, plan)
    report, plan = complete_report()
    report["initial_rows"][-1] = deepcopy(report["initial_rows"][0])
    with pytest.raises(ValueError, match="grid coverage"):
        validate_evaluation_report(report, plan)


def test_duplicate_frame_and_unfinished_high_cap_cannot_be_rendered_as_complete():
    report, plan = complete_report()
    report["initial_rows"][0]["frame_rows"][-1] = deepcopy(
        report["initial_rows"][0]["frame_rows"][0]
    )
    with pytest.raises(ValueError, match="frame coverage"):
        validate_evaluation_report(report, plan)
    report, plan = complete_report()
    report["fixed_high_cap_diagnostic"]["status"] = "RUNNING"
    with pytest.raises(ValueError, match="high-cap"):
        validate_evaluation_report(report, plan)
