import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biohub_ct.campaign.detection_diagnostics import CacheValidationError, make_cache_identity
from biohub_ct.campaign.learned_logit_adapter import (
    LearnedLogitAdapter,
    create_adapter,
    load_learned_tile_cache,
    reconstruct_probabilities,
    save_learned_tile_cache,
)
from biohub_ct.pipelines.learned import LearnedConfig, frame_probabilities


def _model():
    model = torch.nn.Conv3d(1, 1, 3, padding=1)
    with torch.no_grad():
        model.weight.copy_(torch.linspace(-0.2, 0.2, 27).reshape(1, 1, 3, 3, 3))
        model.bias.fill_(-0.1)
    return model


def _frame():
    return np.arange(9 * 37 * 41, dtype=np.uint16).reshape(9, 37, 41)


def _identity(frame):
    digest = "1" * 64
    return make_cache_identity(
        model_sha256=digest,
        source_sha256="2" * 64,
        config={"adapter": "learned-per-tile"},
        input_frame=frame,
        transform={"xy_stride": 4, "tile_size": 8, "overlap": 3},
        precision={"model": "float32", "activation": "native"},
        tta={"enabled": False},
        output_schema="biohub.learned.per_tile_logits.v1",
    )


def test_cached_reconstruction_matches_actual_production_forward_and_blending():
    raw = _frame()
    config = LearnedConfig(tile_size=8, overlap=3)
    model = _model().train()
    control = frame_probabilities(model, raw, config)
    progress = []
    payload = LearnedLogitAdapter(
        model, config, inference_precision="full_float32"
    ).capture_payload(
        raw,
        deadline_at=time.monotonic() + 30,
        progress=lambda: progress.append(1),
    )
    replay = reconstruct_probabilities(payload, activation="native", device="cpu")

    np.testing.assert_array_equal(replay, control)
    assert model.training
    assert len(progress) == payload.logits.shape[0] == 8
    assert payload.logits.min() < 0 < payload.logits.max()
    assert payload.normalization["normalized_frame_sha256"]


def test_payload_preserves_production_tile_order_bounds_and_full_weights():
    raw = _frame()
    config = LearnedConfig(tile_size=8, overlap=3)
    payload = LearnedLogitAdapter(
        _model(), config, inference_precision="full_float32"
    ).capture_payload(raw, deadline_at=time.monotonic() + 30, progress=lambda: None)

    assert payload.normalized_shape_zyx == (9, 10, 11)
    assert payload.origins_zyx.tolist() == [
        [0, 0, 0], [0, 0, 3], [0, 2, 0], [0, 2, 3],
        [1, 0, 0], [1, 0, 3], [1, 2, 0], [1, 2, 3],
    ]
    assert payload.stops_zyx.tolist() == [
        [8, 8, 8], [8, 8, 11], [8, 10, 8], [8, 10, 11],
        [9, 8, 8], [9, 8, 11], [9, 10, 8], [9, 10, 11],
    ]
    assert payload.logits.shape == payload.blend_weights.shape == (8, 8, 8, 8)
    assert payload.tiling["tile_input_padding_zyx"] == [[0, 0], [0, 0], [0, 0]]
    assert payload.tiling["configured_overlap_zyx"] == [3, 3, 3]
    assert np.all(payload.blend_weights > 0)
    assert payload.blend_weights[0, 0, 0, 0] == 1
    assert payload.blend_weights[0, 3, 3, 3] == 27


def test_float32_activation_and_explicit_clamp_are_separate_replays():
    raw = _frame()
    payload = LearnedLogitAdapter(
        _model(), LearnedConfig(tile_size=8, overlap=3), inference_precision="full_float32"
    ).capture_payload(raw, deadline_at=time.monotonic() + 30, progress=lambda: None)
    unclamped = reconstruct_probabilities(payload, activation="float32", device="cpu")
    clamped = reconstruct_probabilities(
        payload, activation="float32", logit_clamp=(-0.25, 0.25), device="cpu"
    )
    assert np.any(unclamped != clamped)
    lower = float(torch.sigmoid(torch.tensor(-0.25)))
    upper = float(torch.sigmoid(torch.tensor(0.25)))
    assert float(clamped.min()) >= lower - 1e-7
    assert float(clamped.max()) <= upper + 1e-7


def test_learned_tile_cache_round_trip_and_identity_rejection(tmp_path):
    raw = _frame()
    payload = LearnedLogitAdapter(
        _model(), LearnedConfig(tile_size=8, overlap=3), inference_precision="full_float32"
    ).capture_payload(raw, deadline_at=time.monotonic() + 30, progress=lambda: None)
    identity = _identity(raw)
    path = tmp_path / "frame.learned-tiles.npz"
    manifest = save_learned_tile_cache(path, payload, identity, adapter="test-adapter")
    restored, checked = load_learned_tile_cache(path, expected_identity=identity)

    assert checked == manifest
    np.testing.assert_array_equal(restored.logits, payload.logits)
    np.testing.assert_array_equal(restored.blend_weights, payload.blend_weights)
    np.testing.assert_array_equal(
        reconstruct_probabilities(restored, device="cpu"),
        reconstruct_probabilities(payload, device="cpu"),
    )
    with pytest.raises(CacheValidationError, match="identity"):
        load_learned_tile_cache(path, expected_identity=_identity(raw.copy() + 1))


def test_capture_checks_deadline_before_model_forward():
    with pytest.raises(TimeoutError, match="deadline"):
        LearnedLogitAdapter(
            _model(), LearnedConfig(tile_size=8, overlap=3), inference_precision="full_float32"
        ).capture_payload(_frame(), deadline_at=0, progress=lambda: None)


def test_cli_factory_loads_exported_state_dict_on_cpu(tmp_path):
    from biohub_ct.training.model import ModelConfig, PointDetector3D

    weights = tmp_path / "weights.pt"
    torch.save(PointDetector3D(ModelConfig(base_channels=4)).state_dict(), weights)
    adapter = create_adapter(
        {
            "model_format": "state_dict",
            "model_config": {"base_channels": 4},
            "inference_config": {"tile_size": 8, "overlap": 3},
            "device": "cpu",
            "inference_precision": "full_float32",
        },
        model_file=weights,
    )
    payload = adapter.capture_payload(
        _frame(), deadline_at=time.monotonic() + 30, progress=lambda: None
    )
    assert adapter.output_schema == "biohub.learned.per_tile_logits.v1"
    assert payload.logits.shape == (8, 8, 8, 8)
