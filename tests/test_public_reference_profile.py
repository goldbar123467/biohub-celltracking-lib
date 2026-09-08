import hashlib
import json
from pathlib import Path

import pytest
import scripts.profile_public_reference as profile_module
from scripts.profile_public_reference import (
    EXPECTED_CONFIG,
    EXPECTED_RAW_SHAPE,
    MAX_WALL_SECONDS,
    TTA_EFFECTIVE_TRANSFORMS,
    TTA_VIEWS,
    CompatibilityMismatch,
    ProfileRequest,
    UnavailableDependency,
    apply_planar_view,
    centered_strided_slice,
    enforce_pass_criteria,
    import_runtime_dependencies,
    invert_planar_view,
    load_and_verify_config,
    make_model_class,
    validate_request,
)


def request(tmp_path: Path, **overrides: object) -> ProfileRequest:
    values = {
        "data_root": tmp_path / "data",
        "support_root": tmp_path / "support",
        "checkpoints": (tmp_path / "primary.pth", tmp_path / "secondary.pth"),
        "output": tmp_path / "profile.json",
        "tile_yx": 64,
        "max_wall_seconds": MAX_WALL_SECONDS,
    }
    values.update(overrides)
    return ProfileRequest(**values)  # type: ignore[arg-type]


def test_profile_bounds_prevent_full_frame_or_unbounded_run(tmp_path: Path) -> None:
    validate_request(request(tmp_path))
    with pytest.raises(CompatibilityMismatch, match="tile_yx"):
        validate_request(request(tmp_path, tile_yx=512))
    with pytest.raises(CompatibilityMismatch, match="divisible by four"):
        validate_request(request(tmp_path, tile_yx=63))
    with pytest.raises(CompatibilityMismatch, match="max_wall_seconds"):
        validate_request(request(tmp_path, max_wall_seconds=MAX_WALL_SECONDS + 0.001))


def test_centered_strided_slice_reads_exact_downsampled_tile() -> None:
    assert EXPECTED_RAW_SHAPE == (100, 64, 256, 256)
    result = centered_strided_slice(length=256, stride=4, output_size=64)
    assert result.step == 4
    assert len(range(*result.indices(256))) == 64
    assert result.start == 0
    assert result.stop == 256


def test_missing_runtime_is_reported_without_install_attempt() -> None:
    attempted = []

    def missing_import(name: str):
        attempted.append(name)
        raise ModuleNotFoundError(name)

    with pytest.raises(UnavailableDependency, match="numpy, torch, zarr"):
        import_runtime_dependencies(missing_import)
    assert attempted == ["numpy", "torch", "zarr"]


def test_main_writes_blocked_dependency_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This exercises the standalone CLI; a supervising campaign's identity
    # must not turn its temporary output into an unrelated campaign attempt.
    for name in profile_module.CAMPAIGN_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    def blocked(_request: ProfileRequest):
        raise UnavailableDependency("torch unavailable; no install attempted")

    output = tmp_path / "receipt.json"
    monkeypatch.setattr(profile_module, "run_profile", blocked)
    exit_code = profile_module.main(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--support-root",
            str(tmp_path / "support"),
            "--checkpoints",
            str(tmp_path / "primary.pth"),
            str(tmp_path / "secondary.pth"),
            "--output",
            str(output),
        ]
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert receipt["status"] == "BLOCKED_UNAVAILABLE_DEPENDENCY"
    assert "no package installation was attempted" in receipt["uncertainty"]


def test_atomic_json_rejects_nonfinite_values(tmp_path: Path) -> None:
    output = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="JSON compliant"):
        profile_module._atomic_write_json(output, {"value": float("nan")})
    assert not output.exists()


def test_campaign_adapter_writes_receipt_result_then_final_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "e0-profile"
    attempt = tmp_path / "reports" / "campaign-workers" / run_id
    attempt.mkdir(parents=True)
    output = attempt / "profile.json"
    progress = attempt / "progress.json"
    environment = {
        "BIOHUB_RUN_ID": run_id,
        "BIOHUB_RUN_SPEC_SHA256": "a" * 64,
        "BIOHUB_INTENT_ID": "intent-1",
        "BIOHUB_FENCING_TOKEN": "7",
        "BIOHUB_ATTEMPT_DIR": str(attempt),
        "BIOHUB_PROGRESS_PATH": str(progress),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(profile_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        profile_module,
        "run_profile",
        lambda _request: {"schema_version": 1, "status": "PASS"},
    )
    writes = []
    real_atomic_write = profile_module._atomic_write_json

    def record_write(path: Path, payload: dict) -> None:
        writes.append((Path(path).name, payload.get("completed_units")))
        real_atomic_write(path, payload)

    monkeypatch.setattr(profile_module, "_atomic_write_json", record_write)
    exit_code = profile_module.main(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--support-root",
            str(tmp_path / "support"),
            "--checkpoints",
            str(tmp_path / "primary.pth"),
            str(tmp_path / "secondary.pth"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert writes == [
        ("progress.json", 0),
        ("profile.json", None),
        ("result.json", 1),
        ("progress.json", 1),
    ]
    result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
    artifact_key = output.relative_to(tmp_path).as_posix()
    assert result == {
        "run_id": run_id,
        "run_spec_sha256": "a" * 64,
        "intent_id": "intent-1",
        "fencing_token": 7,
        "status": "COMPLETE",
        "completed_units": 1,
        "artifact_sha256": {artifact_key: hashlib.sha256(output.read_bytes()).hexdigest()},
    }
    final_progress = json.loads(progress.read_text(encoding="utf-8"))
    assert final_progress["completed_units"] == 1
    assert final_progress["error"] is None


def test_config_content_must_match_even_when_json_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "edge_predictor_best.pth"
    checkpoint.write_bytes(b"checkpoint")
    config_path = tmp_path / "config.json"
    changed = {**EXPECTED_CONFIG, "window_size": 3}
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.profile_public_reference.EXPECTED_SHA256",
        {
            **profile_module.EXPECTED_SHA256,
            "config": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
    )
    with pytest.raises(CompatibilityMismatch, match="differs"):
        load_and_verify_config(checkpoint)


def test_public_tta_has_exactly_eight_named_views() -> None:
    assert TTA_VIEWS == (
        "identity",
        "flip_x",
        "flip_y",
        "flip_xy",
        "rot90",
        "rot270",
        "transpose",
        "anti_transpose",
    )
    assert len(set(TTA_EFFECTIVE_TRANSFORMS.values())) == 7
    assert TTA_EFFECTIVE_TRANSFORMS["anti_transpose"] == "flip_x"


def test_planar_views_round_trip_and_preserve_wzyx_shape_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    source = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    for view in TTA_VIEWS:
        transformed = apply_planar_view(source, view)
        restored = invert_planar_view(transformed, view)
        assert transformed.shape == source.shape
        assert torch.equal(restored, source), view
    assert torch.equal(
        apply_planar_view(source, "anti_transpose"),
        apply_planar_view(source, "flip_x"),
    )


def finite_summary(shape: list[int]) -> dict:
    return {
        "shape": shape,
        "finite": True,
        "min": 0.0,
        "max": 1.0,
        "mean": 0.5,
    }


def passing_receipt() -> dict:
    spatial = [3, 8, 8]
    views = [
        {"name": name, "effective_transform": TTA_EFFECTIVE_TRANSFORMS[name]} for name in TTA_VIEWS
    ]
    return {
        "data_identity": {
            "frames": [finite_summary(spatial), finite_summary(spatial)],
            "normalized_pair": finite_summary([2, *spatial]),
        },
        "model_profiles": [
            {
                "label": "primary",
                "executed_view_count": 8,
                "unique_effective_transform_count": 7,
                "views": views,
                "feature_output": finite_summary([1, 2, 32, *spatial]),
                "detection_outputs": [
                    finite_summary([1, 1, *spatial]),
                    finite_summary([1, 1, *spatial]),
                ],
                "mean_abs_feature_delta_from_identity": 0.01,
            },
            {
                "label": "secondary",
                "executed_view_count": 8,
                "unique_effective_transform_count": 7,
                "views": views,
                "feature_output": finite_summary([1, 2, 32, *spatial]),
                "detection_outputs": [
                    finite_summary([1, 1, *spatial]),
                    finite_summary([1, 1, *spatial]),
                ],
                "mean_abs_feature_delta_from_identity": 0.0,
            },
        ],
        "secondary_detection_fusion": {
            "outputs": [
                finite_summary([1, 1, *spatial]),
                finite_summary([1, 1, *spatial]),
            ]
        },
    }


def test_pass_criteria_reject_nonfinite_shape_mismatch_and_noop() -> None:
    receipt = passing_receipt()
    enforce_pass_criteria(receipt)

    receipt = passing_receipt()
    receipt["model_profiles"][0]["feature_output"]["finite"] = False
    receipt["model_profiles"][0]["feature_output"]["min"] = float("nan")
    with pytest.raises(CompatibilityMismatch, match="NaN or infinity"):
        enforce_pass_criteria(receipt)

    receipt = passing_receipt()
    receipt["model_profiles"][1]["feature_output"]["shape"][-1] = 7
    with pytest.raises(CompatibilityMismatch, match="feature shapes differ"):
        enforce_pass_criteria(receipt)

    receipt = passing_receipt()
    receipt["model_profiles"][0]["mean_abs_feature_delta_from_identity"] = 0.0
    with pytest.raises(CompatibilityMismatch, match="no-op"):
        enforce_pass_criteria(receipt)


def test_checkpoint_wrapper_encode_shape_contract_on_cpu() -> None:
    torch = pytest.importorskip("torch")

    class FakeUNet(torch.nn.Module):
        def forward(self, value):
            return value.repeat(1, 1, 4, 1, 1, 1)

    class FakeTransformer(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

    wrapper = make_model_class(torch, FakeTransformer)
    model = wrapper(FakeUNet(), unet_out_channels=4).eval()
    inputs = torch.zeros((1, 2, 3, 8, 8), dtype=torch.float32)
    with torch.inference_mode():
        features, detections = model.encode(inputs)
    assert features.shape == (1, 2, 4, 3, 8, 8)
    assert len(detections) == 2
    assert all(item.shape == (1, 1, 3, 8, 8) for item in detections)
    with pytest.raises(ValueError, match="B, W, Z, Y, X"):
        model.encode(torch.zeros((2, 3, 8, 8)))
