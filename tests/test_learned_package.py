from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from biohub_ct.pipelines.learned import LearnedConfig
from biohub_ct.training.model import ModelConfig, PointDetector3D


@pytest.fixture
def exported(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "learned_packager", Path(__file__).resolve().parents[1] / "scripts/package_learned_kaggle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = ModelConfig(base_channels=4)
    torch.manual_seed(7)
    model = PointDetector3D(config).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "model_config": asdict(config), "step": 32}, checkpoint)
    snapshot = tmp_path / "snapshot.json"
    module.write_json(
        snapshot,
        {
            "run_id": "test-only",
            "reports": {
                "fold0/selection.json": {
                    "checkpoint": {"sha256": module.sha256(checkpoint), "step": 32},
                    "selected_threshold": 0.3,
                },
                "manifest.json": {
                    "model": asdict(config),
                    "inference": asdict(LearnedConfig()),
                    "identity": {"source_digest": "test"},
                },
            },
        },
    )
    output = tmp_path / "model"
    manifest = module.export_model(checkpoint, snapshot, output, "owner/test-model")
    return module, model, checkpoint, snapshot, output, manifest


def test_export_preserves_inference_and_selected_threshold(exported):
    _, model, _, _, output, manifest = exported
    restored = PointDetector3D(ModelConfig(**manifest["model_config"])).eval()
    restored.load_state_dict(torch.load(output / "detector-weights.pt", weights_only=True))
    image = torch.arange(512, dtype=torch.float32).reshape(1, 1, 8, 8, 8) / 511
    with torch.inference_mode():
        assert torch.equal(model(image), restored(image))
    assert manifest["inference_config"]["threshold"] == 0.3


def test_export_rejects_unselected_checkpoint(exported, tmp_path):
    module, _, checkpoint, snapshot, _, _ = exported
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="frozen selection"):
        module.export_model(checkpoint, snapshot, tmp_path / "bad", "owner/test-model")


def test_build_is_frozen_offline_and_checks_weights(exported, tmp_path):
    module, _, _, _, model_dir, _ = exported
    first, second = tmp_path / "first", tmp_path / "second"
    module.build_learned(first, "owner/test", model_dir, "owner/test-model")
    module.build_learned(second, "owner/test", model_dir, "owner/test-model")
    for filename in ("submission.ipynb", "kernel-metadata.json", "package-manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    metadata = json.loads((first / "kernel-metadata.json").read_text())
    assert metadata["enable_internet"] is False
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == ["owner/test-model"]
    notebook = json.loads((first / "submission.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook-cell", "exec")
    weights = model_dir / "detector-weights.pt"
    weights.write_bytes(weights.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        module.build_learned(tmp_path / "bad", "owner/test", model_dir, "owner/test-model")
