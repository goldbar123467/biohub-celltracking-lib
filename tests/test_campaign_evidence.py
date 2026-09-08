from __future__ import annotations

import hashlib
import json

import pytest

from biohub_ct.campaign.evidence import (
    PINNED_SCORER_COMMIT,
    PINNED_SCORER_SHA256,
    EvidenceValidationError,
    aggregate_evidence,
    audit_historical_outer_progress,
    build_per_clip_evidence,
    compare_paired_evidence,
    strict_json_dumps,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _raw_record(
    tmp_path,
    clip_id: str,
    *,
    candidate_id: str = "candidate-a",
    direction: str = "fit_6bba_eval_44b6",
    predicted_nodes: int = 10,
    estimated_nodes: int = 10,
    annotated_nodes: int = 2,
    matched_annotated_nodes: int = 2,
    edge: tuple[int, int, int] = (1, 0, 0),
    division: tuple[int, int, int] = (0, 0, 0),
    membership: str = "excluded",
    evidence_class: str = "locked_evaluation",
    input_manifest: str = "same-input",
    split_manifest: str = "same-split",
    runtime: dict[str, float] | None = None,
):
    artifact = tmp_path / f"{candidate_id}-{clip_id}.json"
    if not artifact.exists():
        artifact.write_text(json.dumps({"candidate": candidate_id, "clip": clip_id}))
    if matched_annotated_nodes:
        localization = {
            "p50": {"value": 0.5, "reason": None},
            "p90": {"value": 0.9, "reason": None},
            "p95": {"value": 1.0, "reason": None},
        }
    else:
        localization = {
            key: {"value": None, "reason": "matched_annotated_nodes_zero"}
            for key in ("p50", "p90", "p95")
        }
    embryo = clip_id.split("_", 1)[0]
    membership_ids = ["6bba_train"] if membership == "excluded" else []
    if membership == "included":
        membership_ids = [clip_id]
    return {
        "schema_version": 1,
        "run_id": f"run-{candidate_id}-{direction}",
        "candidate_id": candidate_id,
        "clip_id": clip_id,
        "embryo": embryo,
        "direction": direction,
        "frame_ids": [0, 1],
        "evidence_class": evidence_class,
        "provenance": {
            "candidate_sha256": _digest(candidate_id),
            "git_commit": "1" * 40,
            "source_bundle_sha256": _digest(f"source-{candidate_id}"),
            "dependency_manifest_sha256": _digest("deps"),
            "effective_config_sha256": _digest(f"config-{candidate_id}"),
            "model_weight_sha256": _digest(f"weights-{candidate_id}"),
            "model_weight_sha256_reason": None,
            "input_manifest_sha256": _digest(input_manifest),
            "split_manifest_sha256": _digest(split_manifest),
            "scorer_commit": PINNED_SCORER_COMMIT,
            "scorer_sha256": PINNED_SCORER_SHA256,
            "training_membership": membership,
            "training_ids": membership_ids,
            "development_ids": [],
            "used_for_development_selection": False,
            "previously_inspected": False,
        },
        "predicted_nodes": predicted_nodes,
        "estimated_nodes": estimated_nodes,
        "annotated_nodes": annotated_nodes,
        "matched_annotated_nodes": matched_annotated_nodes,
        "localization_um_quantiles": localization,
        "pre_cap_candidates": predicted_nodes,
        "capped_frames": 0,
        "total_frames": 2,
        "edge_tp": edge[0],
        "edge_fp": edge[1],
        "edge_fn": edge[2],
        "division_tp": division[0],
        "division_fp": division[1],
        "division_fn": division[2],
        "predicted_forks_after_solver": division[0] + division[1],
        "predicted_forks_after_postprocess": division[0] + division[1],
        "runtime_seconds_by_stage": runtime if runtime is not None else {"scoring": 0.25},
        "peak_ram_bytes": 1024,
        "peak_vram_bytes": 0,
        "fallback_count": 0,
        "graph_path": artifact.name,
        "graph_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "evaluation_complete": True,
    }


def _record(tmp_path, clip_id: str, **kwargs):
    return build_per_clip_evidence(_raw_record(tmp_path, clip_id, **kwargs))


def test_complete_rejects_missing_planned_clip_ids(tmp_path):
    record = _record(tmp_path, "44b6_a")
    plan = {"fit_6bba_eval_44b6": ["44b6_a", "44b6_b"]}

    with pytest.raises(EvidenceValidationError, match="missing clip IDs.*44b6_b"):
        aggregate_evidence(
            [record],
            planned_clip_ids_by_direction=plan,
            status="COMPLETE",
            artifact_root=tmp_path,
        )

    partial = aggregate_evidence(
        [record], plan, status="PARTIAL", artifact_root=tmp_path, verify_artifacts=True
    )
    assert partial["status"] == "PARTIAL"
    assert partial["coverage_complete"] is False
    assert partial["missing_clip_ids"] == ["44b6_b"]


def test_duplicate_evaluated_and_planned_ids_are_rejected(tmp_path):
    record = _record(tmp_path, "44b6_a")
    with pytest.raises(EvidenceValidationError, match="duplicate evaluated clip_id"):
        aggregate_evidence(
            [record, record],
            {"fit_6bba_eval_44b6": ["44b6_a"]},
            status="PARTIAL",
        )
    with pytest.raises(EvidenceValidationError, match="duplicate planned clip IDs"):
        aggregate_evidence(
            [record],
            {
                "fit_6bba_eval_44b6": ["44b6_a"],
                "fit_44b6_eval_6bba": ["44b6_a"],
            },
            status="PARTIAL",
        )


def test_absent_counts_and_missing_diagnostics_are_rejected(tmp_path):
    missing_count = _raw_record(tmp_path, "44b6_a")
    del missing_count["estimated_nodes"]
    with pytest.raises(EvidenceValidationError, match="estimated_nodes"):
        build_per_clip_evidence(missing_count)

    missing_diagnostic = _raw_record(tmp_path, "44b6_b")
    del missing_diagnostic["pre_cap_candidates"]
    with pytest.raises(EvidenceValidationError, match="pre_cap_candidates"):
        build_per_clip_evidence(missing_diagnostic)

    empty_runtime = _raw_record(tmp_path, "44b6_c", runtime={})
    with pytest.raises(EvidenceValidationError, match="runtime_seconds_by_stage cannot be empty"):
        build_per_clip_evidence(empty_runtime)


def test_empty_denominators_use_json_null_plus_reason(tmp_path):
    record = _record(
        tmp_path,
        "44b6_empty",
        predicted_nodes=0,
        estimated_nodes=0,
        annotated_nodes=0,
        matched_annotated_nodes=0,
        edge=(0, 0, 0),
        division=(0, 0, 0),
    )
    result = aggregate_evidence(
        [record],
        {"fit_6bba_eval_44b6": ["44b6_empty"]},
        status="PARTIAL",
    )
    official = result["overall"]["official"]
    assert official["raw_edge_jaccard"] == {
        "value": None,
        "reason": "edge_union_zero",
    }
    assert official["division_jaccard"] == {
        "value": None,
        "reason": "division_union_zero",
    }
    assert result["overall"]["recall"]["pooled"]["value"] is None
    encoded = strict_json_dumps(result)
    assert "NaN" not in encoded and "Infinity" not in encoded
    with pytest.raises(EvidenceValidationError, match="non-finite"):
        strict_json_dumps({"metric": float("nan")})


def test_official_aggregate_uses_each_candidates_own_edge_union_weights(tmp_path):
    plan = {"fit_6bba_eval_44b6": ["44b6_a", "44b6_b"]}
    control = [
        _record(tmp_path, "44b6_a", candidate_id="control", edge=(1, 0, 0)),
        _record(tmp_path, "44b6_b", candidate_id="control", edge=(1, 8, 0)),
    ]
    candidate = [
        _record(tmp_path, "44b6_a", candidate_id="candidate", edge=(9, 0, 0)),
        _record(tmp_path, "44b6_b", candidate_id="candidate", edge=(0, 1, 0)),
    ]

    comparison = compare_paired_evidence(
        control,
        candidate,
        plan,
        control_artifact_root=tmp_path,
        candidate_artifact_root=tmp_path,
    )

    assert comparison["control"]["overall"]["official"]["adjusted_edge_jaccard"][
        "value"
    ] == pytest.approx(0.2)
    assert comparison["candidate"]["overall"]["official"]["adjusted_edge_jaccard"][
        "value"
    ] == pytest.approx(0.9)
    assert comparison["candidate_minus_control"]["adjusted_edge_jaccard"]["value"] == pytest.approx(
        0.7
    )
    assert "own per-clip edge union" in comparison["adjusted_weight_policy"]


def test_unknown_training_membership_cannot_be_labeled_locked_holdout(tmp_path):
    raw = _raw_record(
        tmp_path,
        "44b6_a",
        membership="unknown",
        evidence_class="locked_evaluation",
    )
    with pytest.raises(EvidenceValidationError, match="cannot be labeled as a clean/locked"):
        build_per_clip_evidence(raw)

    raw["evidence_class"] = "overlap_unknown"
    assert build_per_clip_evidence(raw).evidence_class.value == "overlap_unknown"


def test_changed_graph_artifact_invalidates_complete_evidence(tmp_path):
    record = _record(tmp_path, "44b6_a")
    plan = {"fit_6bba_eval_44b6": ["44b6_a"]}
    aggregate_evidence([record], plan, status="COMPLETE", artifact_root=tmp_path)

    (tmp_path / record.graph_path).write_text("changed after manifest")
    with pytest.raises(EvidenceValidationError, match="artifact SHA256 mismatch"):
        aggregate_evidence([record], plan, status="COMPLETE", artifact_root=tmp_path)


def test_two_directions_report_mean_and_pooled_recall_and_division_subsets(tmp_path):
    forward = _record(
        tmp_path,
        "44b6_a",
        direction="fit_6bba_eval_44b6",
        annotated_nodes=1,
        matched_annotated_nodes=1,
        division=(1, 0, 1),
    )
    reverse = _record(
        tmp_path,
        "6bba_b",
        direction="fit_44b6_eval_6bba",
        annotated_nodes=9,
        matched_annotated_nodes=0,
        division=(0, 2, 0),
    )
    plan = {
        "fit_6bba_eval_44b6": ["44b6_a"],
        "fit_44b6_eval_6bba": ["6bba_b"],
    }
    result = aggregate_evidence([forward, reverse], plan, status="COMPLETE", artifact_root=tmp_path)

    assert set(result["directions"]) == set(plan)
    recall = result["overall"]["recall"]
    assert recall["mean_per_clip"]["value"] == pytest.approx(0.5)
    assert recall["mean_per_clip_denominator_clips"] == 2
    assert recall["pooled"]["value"] == pytest.approx(0.1)
    assert recall["pooled_numerator_matched_annotated_nodes"] == 1
    assert recall["pooled_denominator_annotated_nodes"] == 10
    divisions = result["overall"]["divisions"]
    assert divisions["all_clips"]["division_jaccard"]["value"] == pytest.approx(1 / 4)
    assert divisions["all_clips"]["division_fp"] == 2
    assert divisions["division_bearing_subset"]["clip_ids"] == ["44b6_a"]
    assert divisions["division_bearing_subset"]["division_jaccard"]["value"] == pytest.approx(1 / 2)


def test_pairing_rejects_data_or_split_population_mismatch(tmp_path):
    control = [_record(tmp_path, "44b6_a", candidate_id="control")]
    candidate = [
        _record(
            tmp_path,
            "44b6_a",
            candidate_id="candidate",
            input_manifest="different-input",
        )
    ]
    with pytest.raises(EvidenceValidationError, match="input_manifest_sha256"):
        compare_paired_evidence(
            control,
            candidate,
            {"fit_6bba_eval_44b6": ["44b6_a"]},
            status="PARTIAL",
            verify_artifacts=False,
        )


def _historical_row(
    clip_id: str,
    *,
    edge_tp: int,
    edge_fp: int,
    division: tuple[int, int, int] = (0, 0, 0),
    capped=False,
):
    edge_union = edge_tp + edge_fp
    raw = edge_tp / edge_union
    node_count_ratio = 2.0
    return {
        "dataset": clip_id,
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": 0,
        "division_tp": division[0],
        "division_fp": division[1],
        "division_fn": division[2],
        "node_recall": 0.5,
        "node_count_ratio": node_count_ratio,
        "total_node_ratio": node_count_ratio - 1,
        "edge_jaccard": raw,
        "adj_edge_jaccard": raw * 0.9,
        "diagnostics": {"capped_frames": [1] if capped else []},
    }


def test_historical_outer_audit_stays_partial_and_records_original_membership(tmp_path):
    outer = {
        "complete": False,
        "classical": [
            _historical_row("44b6_a", edge_tp=1, edge_fp=0, division=(1, 0, 1)),
            _historical_row("44b6_b", edge_tp=1, edge_fp=8, division=(0, 2, 0)),
        ],
        "learned": [
            _historical_row("44b6_a", edge_tp=9, edge_fp=0, division=(1, 0, 1), capped=True),
            _historical_row("44b6_b", edge_tp=0, edge_fp=1, division=(0, 2, 0)),
        ],
    }
    splits = {
        "fold0": {
            "train": ["6bba_train_a", "6bba_train_b"],
            "val": ["44b6_a", "44b6_b", "44b6_missing"],
        }
    }
    outer_path = tmp_path / "outer-progress.json"
    split_path = tmp_path / "splits.json"
    outer_path.write_text(json.dumps(outer))
    split_path.write_text(json.dumps(splits))

    result = audit_historical_outer_progress(outer_path, split_path)

    assert result["status"] == "PARTIAL"
    assert result["new_contract_eligible"] is False
    assert result["direction"] == "fit_6bba_eval_44b6"
    assert result["evaluated_clip_count"] == 2
    assert result["planned_clip_count"] == 3
    assert result["missing_clip_ids"] == ["44b6_missing"]
    membership = result["original_dataset_membership"]
    assert membership["training_embryos"] == ["6bba"]
    assert membership["evaluation_embryos"] == ["44b6"]
    assert membership["training_ids"] == ["6bba_train_a", "6bba_train_b"]
    assert "two_direction_evaluation" in result["claims_not_supported"]
    assert result["candidates"]["learned"]["adjusted_edge_jaccard"]["value"] == pytest.approx(0.81)
    learned = result["candidates"]["learned"]
    assert learned["division_jaccard_all_clips"]["value"] == pytest.approx(1 / 4)
    assert learned["division_bearing_subset"]["clip_ids"] == ["44b6_a"]
    assert learned["division_bearing_subset"]["division_jaccard"]["value"] == pytest.approx(1 / 2)
    assert learned["count_ratio_distribution"]["pooled_ratio"]["value"] is None
