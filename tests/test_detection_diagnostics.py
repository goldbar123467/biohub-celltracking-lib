import json
from dataclasses import replace

import numpy as np
import pytest
from scripts.cache_detection_diagnostics import (
    Progress,
    hash_attempt_artifacts,
    select_frame_indices,
)

from biohub_ct.campaign.detection_diagnostics import (
    CacheValidationError,
    PlateauCandidate,
    connected_plateau_candidates,
    extract_plateau_peaks,
    freeze_ablation_grid,
    load_logit_cache,
    make_cache_identity,
    physical_nms,
    save_logit_cache,
    sigmoid_scores,
    validate_adapter_output,
)

DIGEST = "a" * 64


def test_explicit_frame_panel_is_bounded_unique_and_ordered():
    assert select_frame_indices(100, 2, [0, 50]) == [0, 50]
    assert select_frame_indices(100, 2, None) == [0, 1]
    with pytest.raises(ValueError, match="exceeds"):
        select_frame_indices(100, 1, [0, 50])
    with pytest.raises(ValueError, match="unique"):
        select_frame_indices(100, 2, [50, 50])
    with pytest.raises(ValueError, match="outside"):
        select_frame_indices(100, 2, [0, 100])


def identity(frame: np.ndarray, *, precision: str = "full-float32"):
    return make_cache_identity(
        model_sha256=DIGEST,
        source_sha256="b" * 64,
        config={"tile": 64, "overlap": 16},
        input_frame=frame,
        transform={"axes": "ZYX", "xy_stride": 4},
        precision={"inference": precision, "activation": "not-applied"},
        tta={"enabled": False, "transforms": []},
    )


def test_float32_raw_logits_round_trip_without_probability_clipping(tmp_path):
    frame = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    logits = np.array([[[-12.5, 0.0], [1.25, 19.0]]], dtype=np.float32)
    path = tmp_path / "frame.npy"
    expected = identity(frame)

    manifest = save_logit_cache(path, logits, expected, adapter="test.raw-logits")
    restored, loaded_manifest = load_logit_cache(path, expected_identity=expected)

    assert manifest == loaded_manifest
    assert loaded_manifest["output_kind"] == "raw_pre_sigmoid_logits"
    assert np.array_equal(restored, logits)
    assert restored.min() < 0 and restored.max() > 1


def test_cache_corruption_and_identity_change_fail_closed(tmp_path):
    frame = np.ones((2, 2, 2), dtype=np.uint16)
    logits = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    path = tmp_path / "frame.npy"
    expected = identity(frame)
    save_logit_cache(path, logits, expected, adapter="test.raw-logits")

    changed = replace(expected, precision_sha256="c" * 64)
    with pytest.raises(CacheValidationError, match="invalidated"):
        load_logit_cache(path, expected_identity=changed)

    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(CacheValidationError, match="checksum"):
        load_logit_cache(path, expected_identity=expected)


@pytest.mark.parametrize(
    "value,error",
    [
        (np.zeros((2, 2, 2), dtype=np.float16), TypeError),
        (np.full((2, 2, 2), np.nan, dtype=np.float32), ValueError),
        (np.zeros((1, 2, 2, 2), dtype=np.float32), ValueError),
    ],
)
def test_adapter_output_never_casts_clips_or_accepts_nonfinite(value, error):
    with pytest.raises(error):
        validate_adapter_output(value)


def test_sigmoid_precision_is_explicit_and_full_float_requires_new_cache_identity():
    logits = np.array([[[-8.001, 8.001]]], dtype=np.float32)
    low = sigmoid_scores(logits, arithmetic="float16")
    full = sigmoid_scores(logits, arithmetic="float32")
    assert low.dtype == full.dtype == np.dtype("float32")
    assert not np.array_equal(low, full)
    with pytest.raises(ValueError, match="float16 or float32"):
        sigmoid_scores(logits, arithmetic="implicit")


def test_connected_same_logit_plateau_has_one_deterministic_representative():
    pytest.importorskip("scipy")
    logits = np.full((3, 3, 4), -5, dtype=np.float32)
    logits[1, 1, 1:3] = 7

    candidates, raw_maxima = connected_plateau_candidates(
        logits, threshold_logit=0, scale_zyx_um=(3.0, 1.0, 1.0)
    )

    assert raw_maxima == 2
    assert len(candidates) == 1
    assert candidates[0].coord_zyx == (1, 1, 1)
    assert candidates[0].plateau_voxels == 2
    assert candidates[0].centroid_zyx == (1.0, 1.0, 1.5)
    assert candidates[0].representative_displacement_um == pytest.approx(0.5)


def test_disconnected_equal_maxima_and_border_are_retained():
    pytest.importorskip("scipy")
    logits = np.full((5, 5, 5), -4, dtype=np.float32)
    logits[0, 0, 0] = 6
    logits[4, 4, 4] = 6

    result = extract_plateau_peaks(
        logits,
        threshold_logit=0,
        scale_zyx_um=(2.0, 1.0, 1.0),
        radius_um=2.0,
    )

    assert result.raw_maximum_voxels == 2
    assert result.plateau_count == 2
    assert [candidate.coord_zyx for candidate in result.retained] == [
        (0, 0, 0),
        (4, 4, 4),
    ]
    assert not result.capped


def test_physical_nms_uses_anisotropic_distance_and_stable_equal_score_order():
    candidates = [
        PlateauCandidate((0, 1, 0), 2.0, 1, (0.0, 1.0, 0.0), 0.0),
        PlateauCandidate((1, 0, 0), 2.0, 1, (1.0, 0.0, 0.0), 0.0),
        PlateauCandidate((0, 0, 0), 3.0, 1, (0.0, 0.0, 0.0), 0.0),
    ]

    kept = physical_nms(candidates, scale_zyx_um=(3.0, 1.0, 1.0), radius_um=2.0)

    assert [candidate.coord_zyx for candidate in kept] == [(0, 0, 0), (1, 0, 0)]


def test_candidate_cap_is_disclosed_after_nms():
    pytest.importorskip("scipy")
    logits = np.full((5, 5, 5), -5, dtype=np.float32)
    logits[0, 0, 0] = 4
    logits[4, 4, 4] = 3
    result = extract_plateau_peaks(
        logits,
        threshold_logit=0,
        scale_zyx_um=(1, 1, 1),
        radius_um=1,
        max_candidates=1,
    )
    assert result.pre_cap_count == 2
    assert result.capped
    assert len(result.retained) == 1


def test_grid_is_frozen_at_fifteen_per_variant_with_one_refinement():
    plan = freeze_ablation_grid(
        precision_variant="amp-logits-float32-sigmoid",
        tie_variant="connected-plateau",
        thresholds_logit=[-1, 0, 1, 2, 3],
        radii_um=[2, 3, 4],
        refinement=[(0.5, 2.5), (0.5, 3.0)],
    )
    assert len(plan.initial) == 15
    assert len(plan.refinement) == 2
    assert not hasattr(plan.initial[0], "precision")

    with pytest.raises(ValueError, match="1 to 15"):
        freeze_ablation_grid(
            precision_variant="full-float32",
            tie_variant="connected-plateau",
            thresholds_logit=[0, 1, 2, 3],
            radii_um=[1, 2, 3, 4],
        )


def test_progress_contract_is_atomic_and_exact(tmp_path):
    path = (tmp_path / "progress.json").resolve()
    progress = Progress(path, "e1-run", "d" * 64)
    progress.write()
    first = json.loads(path.read_text(encoding="utf-8"))
    assert set(first) == {
        "run_id",
        "run_spec_sha256",
        "completed_units",
        "observed_at",
        "error",
    }
    assert first["completed_units"] == 0
    assert first["error"] is None
    progress.completed_units = 1
    progress.write()
    assert json.loads(path.read_text(encoding="utf-8"))["completed_units"] == 1


def test_attempt_artifact_manifest_is_exact_and_rejects_escape(tmp_path):
    root = tmp_path / "root"
    attempt = root / "reports" / "campaign-workers" / "e1-run"
    cache = attempt / "cache"
    cache.mkdir(parents=True)
    payload = cache / "frame.npz"
    sidecar = cache / "frame.npz.manifest.json"
    payload.write_bytes(b"payload")
    sidecar.write_text("{}\n", encoding="utf-8")

    manifest = hash_attempt_artifacts(root, attempt, [payload, sidecar])

    assert set(manifest) == {
        "reports/campaign-workers/e1-run/cache/frame.npz",
        "reports/campaign-workers/e1-run/cache/frame.npz.manifest.json",
    }
    outside = root / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the attempt"):
        hash_attempt_artifacts(root, attempt, [outside])
    with pytest.raises(ValueError, match="unique"):
        hash_attempt_artifacts(root, attempt, [payload, payload])
