import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from biohub_ct.campaign.admission import (
    AdmissionError,
    admit_resource_action,
    admit_submission_slots,
    file_sha256,
    release_quota_reserve,
    require_fresh,
    verify_file_manifest,
    verify_release_artifacts,
    verify_run_files,
)


def test_release_reserve_is_conservative_and_not_device_count():
    assert release_quota_reserve(full_runtime_hours=None, quota_hours_per_wall_hour=1, verified_runtime_limit_hours=12) == 24
    assert release_quota_reserve(full_runtime_hours="1.5", quota_hours_per_wall_hour="1.1", verified_runtime_limit_hours=12) == Decimal("3.30")
    for value in (None, "NaN", "Infinity", -1, True):
        with pytest.raises(AdmissionError):
            release_quota_reserve(full_runtime_hours=1, quota_hours_per_wall_hour=value, verified_runtime_limit_hours=12)


def test_resource_freshness_concurrency_and_reserved_hours():
    now = datetime.now(UTC)
    observation = {"observed_at": now.isoformat(), "remaining": "29.82"}
    args = {"observation": observation, "now": now, "reservation": "5.8", "protected_reserve": 24, "active_gpu_jobs": 0}
    assert admit_resource_action(**args)["remaining_after"] == "24.02"
    with pytest.raises(AdmissionError, match="does not fit"):
        admit_resource_action(**{**args, "reservation": 6})
    with pytest.raises(AdmissionError, match="already has"):
        admit_resource_action(**{**args, "active_gpu_jobs": 1})
    with pytest.raises(AdmissionError, match="stale"):
        admit_resource_action(**{**args, "now": now + timedelta(seconds=121)})
    with pytest.raises(AdmissionError, match="UTC"):
        require_fresh("2026-09-08T00:00:00", now=now, max_age_seconds=120)


def test_daily_submission_policy_preserves_final_slots():
    args = {"allowed_now": 5, "campaign_today": 0, "exploration_today": 0,
            "submission_class": "public_reference_reproduction"}
    admit_submission_slots(**args)
    with pytest.raises(AdmissionError, match="protected"):
        admit_submission_slots(**{**args, "allowed_now": 2})
    admit_submission_slots(**{**args, "allowed_now": 2, "final_release": True})
    admit_submission_slots(**{**args, "allowed_now": 2, "campaign_today": 3, "final_release": True})
    with pytest.raises(AdmissionError, match="exploratory"):
        admit_submission_slots(**{**args, "exploration_today": 2, "submission_class": "experimental_candidate"})
    with pytest.raises(AdmissionError, match="exploratory"):
        admit_submission_slots(**{**args, "exploration_today": 2})
    # A separately justified reserve release is outside the default three-entry
    # allocation and its two exploratory entries, but still obeys platform quota.
    admit_submission_slots(**{**args, "allowed_now": 1, "campaign_today": 4,
                              "exploration_today": 2, "final_release": True})
    with pytest.raises(AdmissionError, match="platform submission allowance"):
        admit_submission_slots(**{**args, "allowed_now": 0, "final_release": True})


def test_actual_file_changes_and_path_escape_rejected(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("x = 1")
    manifest = {"source.py": file_sha256(path)}
    assert verify_file_manifest(tmp_path, manifest) == manifest
    path.write_text("x = 2")
    with pytest.raises(AdmissionError, match="hash mismatch"):
        verify_file_manifest(tmp_path, manifest)
    with pytest.raises(AdmissionError, match="inside"):
        verify_file_manifest(tmp_path, {"../outside": "a" * 64})


def test_source_identity_manifest_covers_actual_worker_bytes(tmp_path):
    (tmp_path / "worker.py").write_text("print('bounded')")
    (tmp_path / "source.json").write_text(json.dumps({"worker.py": file_sha256(tmp_path / "worker.py")}))
    (tmp_path / "config.json").write_text("{}")
    source_hash = file_sha256(tmp_path / "source.json")
    other_hash = file_sha256(tmp_path / "config.json")
    spec = {"source": {"source_bundle_sha256": source_hash, "dependency_manifest_sha256": other_hash,
                       "effective_config_sha256": other_hash},
            "data": {"input_manifest_sha256": other_hash, "split_manifest_sha256": other_hash},
            "verified_files": {"source.json": source_hash, "config.json": other_hash},
            "identity_file_bindings": {"source.source_bundle_sha256": "source.json",
                "source.dependency_manifest_sha256": "config.json", "source.effective_config_sha256": "config.json",
                "data.input_manifest_sha256": "config.json", "data.split_manifest_sha256": "config.json"}}
    verify_run_files(spec, tmp_path)
    (tmp_path / "worker.py").write_text("print('changed after approval')")
    with pytest.raises(AdmissionError, match="hash mismatch"):
        verify_run_files(spec, tmp_path)


def candidate_fixture(tmp_path, *, outside=False):
    csv_path = tmp_path / "submission.csv"
    csv_path.write_text("id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n"
                        f"0,sample,node,0,0,0,0,{10 if outside else 0},-1,-1\n")
    identity = {key: "a" * 64 for key in ("source_sha256", "weight_sha256", "dependency_manifest_sha256",
                                          "effective_config_sha256", "input_contract_sha256")}
    receipt = {"notebook_slug": "owner/notebook", "notebook_version": 1, "identity": identity,
        "status": "COMPLETE", "internet_enabled": False, "csv_sha256": file_sha256(csv_path),
        "expected_dataset_ids": ["sample"], "completed_dataset_ids": ["sample"],
        "dataset_shapes": {"sample": [1, 2, 2, 2]}, "coordinate_bounds_valid": True,
        "graph_valid": True, "hidden_discovery_supported": True, "fallback_count": 0,
        "full_runtime_seconds": 2}
    (tmp_path / "rehearsal.json").write_text(json.dumps(receipt))
    (tmp_path / "eligibility.json").write_text(json.dumps({"eligible": True, "identity": identity,
        "training_overlap": "unknown", "overlap_uncertainty_disclosed": True}))
    files = {name: file_sha256(tmp_path / name) for name in ("submission.csv", "rehearsal.json", "eligibility.json")}
    return {"candidate_id": "candidate-1", "competition": "biohub", "submission_class": "public_reference_reproduction",
        "notebook_slug": "owner/notebook", "notebook_version": 1, "rehearsal_manifest_sha256": files["rehearsal.json"],
        "visible_csv_sha256": files["submission.csv"], "output_filename": "submission.csv", "identity": identity,
        "eligibility_evidence_path": "eligibility.json", "verified_files": files,
        "rehearsal_manifest_path": "rehearsal.json", "downloaded_csv_path": "submission.csv"}


def test_release_gate_revalidates_csv_bounds_despite_worker_success_flags(tmp_path):
    candidate = candidate_fixture(tmp_path, outside=True)
    with pytest.raises(AdmissionError, match="outside image bounds"):
        verify_release_artifacts(candidate, tmp_path)


def test_release_gate_exact_version_and_complete_output(tmp_path):
    candidate = candidate_fixture(tmp_path)
    assert verify_release_artifacts(candidate, tmp_path)["rehearsal"]["notebook_version"] == 1
    candidate["notebook_version"] = 2
    with pytest.raises(AdmissionError, match="different notebook"):
        verify_release_artifacts(candidate, tmp_path)


def test_final_reserve_requires_bound_decision_and_unchanged_evidence(tmp_path):
    from biohub_ct.campaign.contracts import ContractError
    from biohub_ct.campaign.state import CampaignStore

    candidate = candidate_fixture(tmp_path)
    candidate["final_release"] = True
    with pytest.raises(AdmissionError, match="recorded justification"):
        verify_release_artifacts(candidate, tmp_path)
    decision = tmp_path / "recovery-review.txt"
    decision.write_text("Fixture: previous execution failed; corrected artifact is rehearsed.")
    candidate["reserve_release_justification"] = {
        "category": "justified_repair",
        "rationale": "Recover a reconciled failed execution with the frozen repaired artifact.",
        "evidence_path": decision.name,
    }
    with pytest.raises(AdmissionError, match="hash-verified evidence"):
        verify_release_artifacts(candidate, tmp_path)
    candidate["verified_files"][decision.name] = file_sha256(decision)
    verified = verify_release_artifacts(candidate, tmp_path)
    assert verified["reserve_release_justification"] == candidate["reserve_release_justification"]
    store = CampaignStore(tmp_path / "isolated.sqlite3")
    store.register_candidate(candidate)
    store.transition_release(candidate["candidate_id"], "FROZEN")
    candidate["reserve_release_justification"]["category"] = "deadline_recovery"
    with pytest.raises(ContractError, match="immutable after freeze"):
        store.update_candidate_spec(candidate["candidate_id"], candidate)
    candidate["reserve_release_justification"]["category"] = "justified_repair"
    decision.write_text("Changed after review")
    with pytest.raises(AdmissionError, match="hash mismatch"):
        verify_release_artifacts(candidate, tmp_path)
