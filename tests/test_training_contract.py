"""Independent behavioral checks for sparse supervision and data exclusion."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biohub_ct.data.paths import DatasetRecord
from biohub_ct.data.schema import Graph, Node
from biohub_ct.training import data


def test_xy_downsampling_and_point_mapping_share_block_centers():
    raw = np.zeros((8, 16, 16), dtype=np.uint16)
    raw[3, 4:8, 8:12] = 1000
    image = data.normalize_frame(raw, xy_stride=4)
    assert image.shape == (8, 4, 4)
    assert np.unravel_index(image.argmax(), image.shape) == (3, 1, 2)
    raw_center = np.array([[3.0, 5.5, 9.5]], dtype=np.float32)
    model_center = data.raw_to_model_points(raw_center, xy_stride=4)
    np.testing.assert_array_equal(model_center, [[3, 1, 2]])
    target, _ = data.make_sparse_targets(
        image,
        model_center,
        data.DataConfig(),
        background_threshold=0,
    )
    assert np.unravel_index(target.argmax(), target.shape) == (3, 1, 2)
    # Integer annotations can map to fractional model coordinates; retain them.
    raw_points = np.array([[0, 0, 0], [7, 15, 15], [4, 6, 10]], dtype=np.float32)
    np.testing.assert_array_equal(
        data.model_to_raw_points(data.raw_to_model_points(raw_points, 4), 4),
        raw_points,
    )
    model_delta = np.array([[1, 1, 1]], dtype=np.float32)
    raw_delta = data.model_to_raw_points(model_delta, 4) - data.model_to_raw_points(
        model_delta * 0, 4
    )
    np.testing.assert_array_equal(raw_delta * [1.625, 0.40625, 0.40625], [[1.625] * 3])


def test_bright_unknown_voxel_has_no_negative_supervision():
    config = data.DataConfig(patch_shape=(16, 16, 16), sigma_voxels=(1, 1, 1))
    image = np.ones((16, 16, 16), dtype=np.float32)
    image[:3, :3, :3] = 0
    target, weight = data.make_sparse_targets(
        image,
        np.array([[4, 4, 4]], dtype=np.float32),
        config,
        background_threshold=0.1,
    )
    assert target[4, 4, 4] == 1 and weight[4, 4, 4] > 0
    assert target[0, 0, 0] == 0 and weight[0, 0, 0] > 0
    assert weight[12, 12, 12] == 0
    assert np.isfinite(target).all() and np.isfinite(weight).all()


def test_unknown_voxels_cannot_change_loss_or_receive_gradient():
    torch = pytest.importorskip("torch")
    from biohub_ct.training.model import masked_heatmap_loss

    logits = torch.tensor([0.0, 100.0, -100.0, 0.0]).reshape(1, 1, 1, 1, 4)
    logits.requires_grad_()
    target = torch.tensor([1.0, 0.0, 1.0, 0.0]).reshape_as(logits)
    weight = torch.tensor([1.0, 0.0, 0.0, 3.0]).reshape_as(logits)
    loss = masked_heatmap_loss(logits, target, weight)
    assert loss.item() == pytest.approx(0.25, abs=1e-7)
    loss.backward()
    torch.testing.assert_close(
        logits.grad.flatten(),
        torch.tensor([-0.0625, 0.0, 0.0, 0.1875]),
        rtol=0,
        atol=0,
    )
    changed = logits.detach().clone()
    changed[..., 1:3] *= -1
    changed_target = target.clone()
    changed_target[..., 1:3] = 1 - changed_target[..., 1:3]
    torch.testing.assert_close(masked_heatmap_loss(changed, changed_target, weight), loss)


def test_batch_with_no_supervision_fails_without_gradients():
    torch = pytest.importorskip("torch")
    from biohub_ct.training.model import masked_heatmap_loss

    logits = torch.zeros((1, 1, 2, 2, 2), requires_grad=True)
    with pytest.raises(ValueError, match="supervised"):
        masked_heatmap_loss(logits, torch.zeros_like(logits), torch.zeros_like(logits))
    assert logits.grad is None


def test_background_only_patch_keeps_low_weight_after_loss_normalization():
    torch = pytest.importorskip("torch")
    from biohub_ct.training.model import masked_heatmap_loss

    config = data.DataConfig(patch_shape=(8, 8, 8))
    target, weight = data.make_sparse_targets(
        np.zeros((8, 8, 8), np.float32),
        np.empty((0, 3), np.float32),
        config,
        background_threshold=0.05,
    )
    assert float(weight.sum()) == pytest.approx(1 / 21, rel=1e-6)
    logits = torch.zeros((1, 1, 8, 8, 8), requires_grad=True)
    target_t, weight_t = (torch.from_numpy(x)[None, None] for x in (target, weight))
    loss = masked_heatmap_loss(logits, target_t, weight_t, normalizer=1.0)
    assert loss.item() == pytest.approx(0.25 / 21, rel=1e-6)
    loss.backward()
    assert logits.grad.sum().item() == pytest.approx(0.25 / 21, rel=1e-6)
    duplicated = masked_heatmap_loss(
        logits.detach().repeat(2, 1, 1, 1, 1),
        target_t.repeat(2, 1, 1, 1, 1),
        weight_t.repeat(2, 1, 1, 1, 1),
        normalizer=2.0,
    )
    torch.testing.assert_close(duplicated, loss)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_loss_rejects_corruption_even_in_unknown_voxels(invalid):
    torch = pytest.importorskip("torch")
    from biohub_ct.training.model import masked_heatmap_loss

    logits = torch.zeros((1, 1, 1, 1, 2), requires_grad=True)
    target = torch.tensor([0.0, invalid]).reshape_as(logits)
    weight = torch.tensor([1.0, 0.0]).reshape_as(logits)
    with pytest.raises(ValueError, match="[Nn]onfinite"):
        masked_heatmap_loss(logits, target, weight)
    assert logits.grad is None


def test_sampler_excludes_internal_dev_and_outer_embryo_and_is_step_reproducible(monkeypatch):
    records = [
        DatasetRecord(name, Path(name + ".zarr"), Path(name + ".geff"))
        for name in ("44b6_fit", "44b6_dev", "6bba_outer")
    ]
    opened, read_times = [], []
    raw = np.arange(16 * 32 * 32, dtype=np.uint16).reshape(16, 32, 32)

    def open_volume(path, **kwargs):
        opened.append(Path(path).stem)
        assert kwargs["require_complete_chunks"]

        def read_frame(t):
            read_times.append(t)
            return raw.copy()

        return SimpleNamespace(
            shape=(2, 16, 32, 32), scale=(1.625, 0.40625, 0.40625), read_frame=read_frame
        )

    monkeypatch.setattr(data, "open_zarr_volume", open_volume)
    monkeypatch.setattr(
        data,
        "read_geff_graph",
        lambda path: (
            Graph([Node(1, 1, 8, 16, 16)]),
            None,
        ),
    )
    sampler = data.SparsePatchSampler(
        records,
        ["44b6_fit"],
        ["6bba_outer"],
        data.DataConfig(patch_shape=(8, 8, 8), seed=7),
        dev_ids=["44b6_dev"],
    )
    first = sampler.sample(step=19, batch_size=2)
    sampler.sample(step=2, batch_size=1)
    resumed = sampler.sample(step=19, batch_size=2)
    assert opened == ["44b6_fit"]
    assert read_times and set(read_times) == {1}
    for key in ("image", "target", "weight"):
        assert first[key].shape == (2, 1, 8, 8, 8)
        assert first[key].dtype == np.float32
        np.testing.assert_array_equal(first[key], resumed[key])
    assert first["metadata"] == resumed["metadata"]
    assert first["loss_normalizer"] == resumed["loss_normalizer"] == 2.0
    assert {row["dataset"] for row in first["metadata"]} == {"44b6_fit"}


@pytest.mark.parametrize("val", [["44b6_other"], ["44b6_fit"]])
def test_sampler_rejects_shared_embryo_before_reading_data(monkeypatch, val):
    def forbidden(*args, **kwargs):
        pytest.fail("Data was opened before leakage validation")

    monkeypatch.setattr(data, "open_zarr_volume", forbidden)
    with pytest.raises(ValueError, match="overlap"):
        data.SparsePatchSampler([], ["44b6_fit"], val)


def test_sampler_rejects_development_ids_in_fit_partition():
    with pytest.raises(ValueError, match="overlap"):
        data.SparsePatchSampler([], ["44b6_fit"], ["6bba_outer"], dev_ids=["44b6_fit"])


def test_model_preserves_odd_patch_shape_and_updates_parameters():
    torch = pytest.importorskip("torch")
    from biohub_ct.training.model import ModelConfig, PointDetector3D, masked_heatmap_loss

    torch.manual_seed(17)
    model = PointDetector3D(ModelConfig(base_channels=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    image = torch.rand((1, 1, 9, 11, 13))
    target = torch.zeros_like(image)
    target[..., 4, 5, 6] = 1
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    logits = model(image)
    assert logits.shape == image.shape
    loss = masked_heatmap_loss(logits, target, torch.ones_like(image))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[key], value) for key, value in model.named_parameters())
