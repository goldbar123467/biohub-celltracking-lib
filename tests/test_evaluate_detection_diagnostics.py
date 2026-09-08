import hashlib
import json
import shutil
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scripts.evaluate_detection_diagnostics import (
    TEMPORAL_MISSING_REASON,
    aggregate_frame_rows,
    choose_refinement,
    connected_plateau_candidates_fast,
    diagnostic_match_points,
    exact_initial_grid,
    frame_metrics,
    make_partial_report,
    operational_completed_units,
    physical_nms_fast,
    run_fixed_high_cap_diagnostic,
    unavailable_metrics,
    verify_annotation_frame_identity,
    verify_cache_precision_identity,
    verify_capture_plan_extension,
    verify_capture_run_index_contract,
    verify_downloaded_run,
    verify_evaluator_metric_contract,
    verify_local_contract_files,
)

from biohub_ct.campaign.detection_diagnostics import (
    connected_plateau_candidates,
    hash_array,
    hash_file,
    hash_json,
    physical_nms,
)
from biohub_ct.data.schema import Graph, Node
from biohub_ct.metrics.edge import match_nodes
from biohub_ct.pipelines.learned import LearnedConfig, probability_nodes


def plan_contract():
    return {
        "precision_variants": [
            {"id": "amp_native_sigmoid"},
            {"id": "amp_logits_fp32_sigmoid"},
            {"id": "full_fp32"},
        ],
        "extraction_variants": [
            {"id": "legacy_voxel_maxima"},
            {"id": "connected_plateau"},
        ],
        "initial_grid": {
            "thresholds_probability": [0.3, 0.5, 0.7],
            "radii_um": [2.0, 3.0, 4.0],
            "total_initial_configurations": 54,
            "maximum_allowed_per_variant": 15,
        },
        "advancement_rule": {"count_ratio_relative_reduction_min": 0.2},
        "refinement_policy": {"maximum_configurations": 6, "maximum_stages": 1},
    }


def test_exact_initial_grid_is_complete_and_fails_on_drift():
    plan = plan_contract()
    grid = exact_initial_grid(plan)
    assert len(grid) == 54
    assert len({cell.cell_id for cell in grid}) == 54

    changed = deepcopy(plan)
    changed["initial_grid"]["thresholds_probability"][-1] = 0.8
    with pytest.raises(ValueError, match="54-cell"):
        exact_initial_grid(changed)


def test_plateau_and_legacy_frame_metrics_expose_tie_behavior():
    pytest.importorskip("scipy", reason="local-maxima extraction requires optional SciPy")
    probability = np.zeros((3, 3, 3), dtype=np.float32)
    probability[1, 1, 1] = 0.9
    probability[1, 1, 2] = 0.9
    truth = np.asarray([[1, 1, 1]], dtype=float)
    common = {
        "raw_shape": (3, 3, 3),
        "raw_scale_zyx_um": (1.0, 1.0, 1.0),
        "xy_stride": 1,
        "threshold": 0.5,
        "radius_um": 0.1,
        "max_nodes": 10,
    }

    plateau = frame_metrics(probability, truth, extraction_variant="connected_plateau", **common)
    legacy = frame_metrics(probability, truth, extraction_variant="legacy_voxel_maxima", **common)

    assert plateau["raw_local_maximum_voxels"] == 2
    assert plateau["connected_plateau_count"] == 1
    assert plateau["plateau_voxel_count_distribution"] == {"2": 1}
    assert plateau["predicted_node_count"] == 1
    assert plateau["annotated_recall"] == 1.0
    assert legacy["raw_local_maximum_voxels"] == 2
    assert legacy["connected_plateau_count"] == 1
    assert legacy["plateau_voxel_count_distribution"] == {"2": 1}
    assert legacy["predicted_node_count"] == 2


def test_vectorized_diagnostic_matcher_parity_with_existing_greedy_branch():
    pytest.importorskip("scipy", reason="reference node matching requires optional SciPy")
    rng = np.random.default_rng(20260908)
    predicted = rng.integers(0, 50, size=(31, 3))
    truth = rng.integers(0, 50, size=(29, 3))
    scale = (1.625, 0.40625, 0.40625)
    pred_graph = Graph(Node(i, 0, *map(int, p)) for i, p in enumerate(predicted))
    truth_graph = Graph(Node(i, 0, *map(int, p)) for i, p in enumerate(truth))
    expected = match_nodes(pred_graph, truth_graph, scale=scale, max_distance=7.0)

    pairs, _ = diagnostic_match_points(
        predicted, truth, raw_scale_zyx_um=scale, max_distance_um=7.0
    )

    assert dict(pairs) == expected.pred_to_gt

    tied_predicted = np.zeros((21, 3), dtype=int)
    tied_truth = np.zeros((22, 3), dtype=int)
    tied_expected = match_nodes(
        Graph(Node(i, 0, *p) for i, p in enumerate(tied_predicted)),
        Graph(Node(i, 0, *p) for i, p in enumerate(tied_truth)),
        scale=(1.0, 1.0, 1.0),
        max_distance=1.0,
    )
    tied_pairs, _ = diagnostic_match_points(
        tied_predicted,
        tied_truth,
        raw_scale_zyx_um=(1.0, 1.0, 1.0),
        max_distance_um=1.0,
    )
    assert dict(tied_pairs) == tied_expected.pred_to_gt


def test_fast_plateau_components_match_frozen_reference_implementation():
    pytest.importorskip("scipy", reason="reference plateau labeling requires optional SciPy")
    scores = np.full((5, 5, 5), -2.0, dtype=np.float32)
    scores[0, 0, 0] = 4.0
    scores[2, 2, 2] = 3.0
    scores[2, 2, 3] = 3.0
    scores[4, 4, 4] = 2.0
    scale = (1.625, 1.625, 1.625)

    expected, expected_raw = connected_plateau_candidates(
        scores, threshold_logit=0.0, scale_zyx_um=scale
    )
    observed, observed_raw = connected_plateau_candidates_fast(
        scores, threshold=0.0, scale_zyx_um=scale
    )

    assert observed_raw == expected_raw
    assert observed == expected
    assert physical_nms_fast(observed, scale_zyx_um=scale, radius_um=2.0) == physical_nms(
        expected, scale_zyx_um=scale, radius_um=2.0
    )


def test_legacy_extraction_matches_production_probability_nodes():
    pytest.importorskip(
        "scipy", reason="production local-maxima extraction requires optional SciPy"
    )
    probability = np.zeros((5, 5, 5), dtype=np.float32)
    probability[1, 1, 1] = 0.9
    probability[1, 1, 2] = 0.9
    probability[4, 4, 4] = 0.8
    config = LearnedConfig(
        threshold=0.5,
        nms_radius_um=1.0,
        xy_stride=1,
        max_nodes_per_frame=10,
    )
    expected, capped = probability_nodes(
        probability, probability.shape, (1.0, 1.0, 1.0), 0, 0, config
    )
    observed = frame_metrics(
        probability,
        np.empty((0, 3)),
        raw_shape=probability.shape,
        raw_scale_zyx_um=(1.0, 1.0, 1.0),
        xy_stride=1,
        extraction_variant="legacy_voxel_maxima",
        threshold=0.5,
        radius_um=1.0,
        max_nodes=10,
    )

    assert observed["predicted_points_raw_zyx"] == [list(node.coord) for node in expected]
    assert observed["capped"] is capped


def test_aggregate_and_temporal_fields_stay_explicitly_missing():
    frame = {
        "predicted_node_count": 2,
        "annotated_node_count": 2,
        "matched_annotated_node_count": 1,
        "matched_localization_um": [1.0],
        "raw_local_maximum_voxels": 3,
        "connected_plateau_count": 2,
        "plateau_voxel_count_distribution": {"1": 1, "2": 1},
        "plateau_representative_displacement_um": {
            "count": 2,
            "min": 0.0,
            "median": 0.25,
            "p95": 0.475,
            "max": 0.5,
        },
        "pre_cap_candidate_count": 4,
        "capped": True,
        "candidate_pool_truncated": False,
    }
    result = aggregate_frame_rows([frame, frame])
    assert result["annotated_recall"] == 0.5
    assert result["matched_localization_median_um"] == 1.0
    assert result["per_frame_cap_rate"] == 1.0
    missing = unavailable_metrics()
    assert missing["fixed_linker_adjusted_edge_score"] == {
        "value": None,
        "reason": TEMPORAL_MISSING_REASON,
    }


def test_refinement_is_one_bounded_midpoint_stage_after_complete_grid():
    plan = plan_contract()
    rows = []
    for cell in exact_initial_grid(plan):
        rows.append(
            {
                "status": "COMPLETE",
                "precision_variant": cell.precision_variant,
                "extraction_variant": cell.extraction_variant,
                "threshold_probability": cell.threshold_probability,
                "radius_um": cell.radius_um,
                "metrics": {
                    "annotated_recall": 1.0,
                    "matched_localization_median_um": 0.5,
                    "matched_localization_p95_um": 1.0,
                    "predicted_node_count": 100,
                    "clipped_candidate_fraction": 0.0,
                    "per_frame_cap_rate": 0.0,
                    "candidate_pool_truncated_frames": 0,
                },
            }
        )
    refinement = choose_refinement(rows, plan)
    assert 1 <= len(refinement) <= 4
    assert {cell.stage for cell in refinement} == {"refinement_1"}
    assert len({(cell.precision_variant, cell.extraction_variant) for cell in refinement}) == 1
    assert all(cell.threshold_probability in {0.6, 0.7} for cell in refinement)
    assert all(cell.radius_um in {2.0, 2.5, 3.0} for cell in refinement)

    with pytest.raises(ValueError, match="54 complete"):
        choose_refinement(rows[:-1], plan)

    rows[0]["metrics"]["clipped_candidate_fraction"] = None
    assert choose_refinement(rows, plan) == ()


def test_partial_checkpoint_never_claims_complete_or_refines_before_54():
    partial = make_partial_report(
        stage="initial",
        fixed_high_cap_diagnostic={"status": "COMPLETE"},
        initial_rows=[{"cell": 1}],
        refinement_rows=[],
        identity={"run": "x"},
    )
    assert partial["status"] == "RUNNING"
    assert operational_completed_units(partial["status"]) == 0
    assert operational_completed_units("COMPLETE") == 1
    assert partial["completed_initial_configurations"] == 1
    assert partial["advancement_eligible"] is False
    with pytest.raises(RuntimeError, match="cannot precede"):
        make_partial_report(
            stage="refinement_1",
            fixed_high_cap_diagnostic={"status": "COMPLETE"},
            initial_rows=[{}] * 53,
            refinement_rows=[{}],
            identity={},
        )


def test_downloaded_run_requires_exact_hashed_attempt_artifacts(tmp_path):
    run_id = "e1-pilot-native-amp-example"
    prefix = f"reports/campaign-workers/{run_id}"
    names = [
        f"{prefix}/cache/cache-index.json",
        f"{prefix}/cache/frame-0.npz",
        f"{prefix}/cache/frame-50.npz",
        f"{prefix}/cache/frame-0.npz.manifest.json",
        f"{prefix}/cache/frame-50.npz.manifest.json",
        f"{prefix}/cache-reference.json",
        f"{prefix}/progress.json",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    digest = "d" * 64
    index = tmp_path / names[0]
    reference = tmp_path / names[-2]
    progress = tmp_path / names[-1]
    reference.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_spec_sha256": digest,
                "intent_id": "intent",
                "fencing_token": 3,
                "cache_index": {"sha256": hashlib.sha256(index.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    progress.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_spec_sha256": digest,
                "completed_units": 2,
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    artifact_sha256 = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names
    }
    result = tmp_path / prefix / "result.json"
    result.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_spec_sha256": digest,
                "intent_id": "intent",
                "fencing_token": 3,
                "status": "COMPLETE",
                "completed_units": 2,
                "artifact_sha256": artifact_sha256,
            }
        ),
        encoding="utf-8",
    )
    run = {"run_id": run_id, "run_spec_expected_artifacts": names}
    completion = tmp_path / prefix / "worker" / "completion.json"
    completion.parent.mkdir(parents=True)
    completion.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_spec_sha256": digest,
                "fencing_token": 3,
                "status": "COMPLETE",
                "exit_code": 0,
                "error": None,
                "artifacts": {
                    "complete": True,
                    "completed_units": 2,
                    "result_manifest_sha256": hash_file(result),
                    "artifact_sha256": artifact_sha256,
                },
                "worker_progress": {
                    "run_id": run_id,
                    "run_spec_sha256": digest,
                    "completed_units": 2,
                    "error": None,
                },
            }
        ),
        encoding="utf-8",
    )
    expected_run = {
        "run_spec_sha256": digest,
        "intent_id": "intent",
        "fencing_token": 3,
        "result_manifest_sha256": hash_file(result),
        "completion_sha256": hash_file(completion),
    }

    assert verify_downloaded_run(tmp_path, run, expected_run)["status"] == "COMPLETE"
    download_root = tmp_path / "downloads"
    shutil.copytree(tmp_path / "reports", download_root / run_id / "reports")
    assert verify_downloaded_run(download_root, run, expected_run)["status"] == "COMPLETE"
    (tmp_path / names[1]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_downloaded_run(tmp_path, run, expected_run)


def test_all_local_source_and_config_contract_files_are_verified(tmp_path):
    source = tmp_path / "capture.py"
    source.write_text("capture = True\n", encoding="utf-8")
    config_root = tmp_path / "work" / "configs"
    config_root.mkdir(parents=True)
    names = [
        "adapter-native-amp.json",
        "adapter-full-fp32.json",
        "precision-native-amp.json",
        "precision-full-fp32.json",
        "transform.json",
        "tta-disabled.json",
    ]
    identities = {}
    for name in names:
        value = {"inference_config": {"xy_stride": 4}, "name": name}
        path = config_root / name
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        identities[name] = {
            "file_sha256": hash_file(path),
            "canonical_json_sha256": hash_json(value),
        }
    plan = {
        "source_contract": {"files": {"capture.py": hash_file(source)}},
        "config_contract": {"root": "work/configs", "files": identities},
    }

    assert set(verify_local_contract_files(plan, source_root=tmp_path)) == set(names)
    (config_root / "transform.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte hash"):
        verify_local_contract_files(plan, source_root=tmp_path)


@pytest.mark.parametrize(
    ("section", "key", "changed"),
    [
        ("population", "frame_hash", "changed"),
        ("initial_grid", "total", 55),
        ("frozen_model", "sha256", "b" * 64),
    ],
)
def test_evaluator_extension_rejects_any_frozen_capture_plan_change(section, key, changed):
    capture = {
        "population": {"frame_hash": "a" * 64},
        "initial_grid": {"total": 54},
        "frozen_model": {"sha256": "c" * 64},
    }
    expanded = {**deepcopy(capture), "evaluator_contract": {"file_sha256": "d" * 64}}
    verify_capture_plan_extension(expanded, capture)
    expanded[section][key] = changed
    with pytest.raises(ValueError, match="except evaluator_contract"):
        verify_capture_plan_extension(expanded, capture)


def test_capture_receipts_are_bound_to_exact_admitted_run_index(tmp_path):
    run_id = "capture-1"
    digest = "a" * 64
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps([{"run_id": run_id, "run_spec_sha256": digest}]) + "\n",
        encoding="utf-8",
    )
    plan = {
        "worker_runs": [{"run_id": run_id}],
        "evaluator_contract": {
            "frozen_capture_run_index": {
                "path": "index.json",
                "file_sha256": hash_file(index_path),
                "run_count": 1,
            },
            "frozen_capture_runs": {
                run_id: {
                    "run_spec_sha256": digest,
                    "intent_id": "intent-1",
                    "fencing_token": 2,
                    "result_manifest_sha256": "b" * 64,
                    "completion_sha256": "c" * 64,
                }
            },
        },
    }
    verify_capture_run_index_contract(plan, source_root=tmp_path)
    plan["evaluator_contract"]["frozen_capture_runs"][run_id]["run_spec_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="differ"):
        verify_capture_run_index_contract(plan, source_root=tmp_path)


def test_fixed_high_cap_diagnostic_is_separate_and_removes_pool_truncation():
    pytest.importorskip("scipy", reason="local-maxima extraction requires optional SciPy")
    probability = np.zeros((64, 64, 64), dtype=np.float32)
    probability[0, 0, 0] = 0.9
    probability[0, 2, 2] = 0.8
    probability[2, 0, 2] = 0.7
    plan = {
        "population": {
            "datasets": [
                {
                    "dataset_id": "44b6_0113de3b",
                    "scale_zyx_um": [1.0, 1.0, 1.0],
                    "frames": [{"frame": 0, "raw_shape_zyx": [64, 256, 256]}],
                }
            ]
        },
        "evaluator_contract": {
            "fixed_high_cap_diagnostic": {
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
        },
    }
    result = run_fixed_high_cap_diagnostic(
        plan,
        {("44b6_0113de3b", 0, "amp_native_sigmoid"): probability},
        deadline_at=float("inf"),
    )
    assert result["role"] == "fixed_truncation_diagnostic_not_grid_search_or_advancement"
    assert result["default_predicted_node_count"] == 3
    assert result["high_cap_predicted_node_count"] == 3
    assert result["high_cap_candidate_pool_truncated"] is False
    assert result["high_cap_capped"] is False


def test_cache_precision_identity_rejects_mislabeled_payload():
    payload = SimpleNamespace(inference_precision="native_amp", native_logit_dtype="torch.float16")
    verify_cache_precision_identity(payload, "native-amp")
    payload.inference_precision = "full_float32"
    with pytest.raises(ValueError, match="precision label"):
        verify_cache_precision_identity(payload, "native-amp")


def test_matching_distance_contract_is_bound_to_consumed_constant():
    plan = {
        "evaluator_contract": {
            "annotation_match_distance_um": 7.0,
            "annotation_matching": {"distance_um": 7.0},
        }
    }
    verify_evaluator_metric_contract(plan)
    plan["evaluator_contract"]["annotation_matching"]["distance_um"] = 8.0
    with pytest.raises(ValueError, match="matching distance"):
        verify_evaluator_metric_contract(plan)


def test_annotation_identity_binds_ordered_coordinates_and_node_ids():
    nodes = [Node(9, 0, 1, 2, 3), Node(12, 0, 4, 5, 6)]
    coordinates = np.asarray([node.coord for node in nodes], dtype=np.float64)
    node_ids = np.asarray([node.node_id for node in nodes], dtype=np.int64)
    expected = {
        "annotated_nodes": 2,
        "coordinate_float64_zyx_sha256": hash_array(coordinates),
        "node_ids_int64_sha256": hash_array(node_ids),
    }
    assert np.array_equal(verify_annotation_frame_identity(nodes, expected), coordinates)
    with pytest.raises(ValueError, match="annotation identity"):
        verify_annotation_frame_identity(list(reversed(nodes)), expected)
