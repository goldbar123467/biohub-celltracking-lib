"""Overflow backoff must leave optimization and data-order semantics intact."""

import copy

import pytest

torch = pytest.importorskip("torch")
from biohub_ct.training.checkpoint import capture_rng_state, restore_rng_state
from biohub_ct.training.trainer import scaled_optimizer_update


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_overflow_retry_matches_one_finite_update(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(37)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.3)).to(device)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler(device, init_scale=65536)
    reference_scaler = torch.amp.GradScaler(device, init_scale=32768)
    images = torch.ones(2, 4, device=device)
    calls = []

    def inject_first_overflow(gradient):
        calls.append(1)
        return torch.full_like(gradient, float("inf")) if len(calls) == 1 else gradient

    hook = next(model.parameters()).register_hook(inject_first_overflow)
    rng = capture_rng_state()
    events = []
    _, _, retries = scaled_optimizer_update(
        model,
        optimizer,
        scaler,
        lambda: model(images).square().mean(),
        10,
        on_overflow=events.append,
    )
    hook.remove()
    actual_rng = capture_rng_state()
    restore_rng_state(rng)
    scaled_optimizer_update(
        reference,
        reference_optimizer,
        reference_scaler,
        lambda: reference(images).square().mean(),
        10,
    )
    assert retries == 1 and len(events) == 1
    assert events[0]["new_scale"] == 32768
    assert scaler.state_dict() == reference_scaler.state_dict()
    for left, right in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert optimizer.state[left]["step"].item() == 1
    assert torch.equal(actual_rng["torch"], capture_rng_state()["torch"])
    for left, right in zip(actual_rng["cuda"], capture_rng_state()["cuda"]):
        assert torch.equal(left, right)


@pytest.mark.parametrize("amp", [False, True])
def test_unrecoverable_gradient_preserves_weights_and_fails(amp):
    model = torch.nn.Linear(2, 1)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=amp)
    next(model.parameters()).register_hook(lambda gradient: gradient * float("inf"))
    with pytest.raises(FloatingPointError, match="Nonfinite gradients"):
        scaled_optimizer_update(
            model,
            optimizer,
            scaler,
            lambda: model(torch.ones(1, 2)).square().mean(),
            10,
            max_retries=2,
        )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    assert not optimizer.state
