import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from biohub_ct.pipelines.learned import LearnedConfig, frame_probabilities, probability_nodes


def test_overlapping_tiles_cover_odd_frame_and_restore_model_mode():
    model = torch.nn.Conv3d(1, 1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(0.7)
    model.train()
    raw = np.arange(11 * 53 * 61, dtype=np.uint16).reshape(11, 53, 61)
    result = frame_probabilities(model, raw, LearnedConfig(tile_size=8, overlap=3))
    assert result.shape == (11, 14, 16)
    np.testing.assert_allclose(result, torch.sigmoid(torch.tensor(0.7)).item(), atol=2e-7)
    assert model.training


def test_peak_coordinates_use_downsample_centers_and_clip_padding():
    probability = np.zeros((8, 8, 8), np.float32)
    probability[2, 3, 4] = 0.9
    probability[7, 7, 7] = 0.8
    nodes, capped = probability_nodes(
        probability, (8, 29, 29), (1.625, 0.40625, 0.40625), 5, 17, LearnedConfig()
    )
    assert [(n.node_id, n.t, n.coord) for n in nodes] == [
        (17, 5, (2, 14, 18)),
        (18, 5, (7, 28, 28)),
    ]
    assert not capped


def test_deadline_interrupts_before_forward():
    model = torch.nn.Conv3d(1, 1, 1)
    with pytest.raises(TimeoutError):
        frame_probabilities(model, np.zeros((8, 32, 32), np.uint16), deadline_at=0)
