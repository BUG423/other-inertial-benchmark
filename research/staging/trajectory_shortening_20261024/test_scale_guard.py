"""Scientific contract tests for the unregistered SAGE-IO proposal."""

import torch

from scale_guard import MultiScaleTemporalBlock, ScaleGuardHead, scale_guard_loss


def test_multiscale_block_preserves_window_shape_and_is_finite():
    block = MultiScaleTemporalBlock(in_channels=6, width=32)
    x = torch.randn(4, 6, 200)
    y = block(x)
    assert y.shape == (4, 32, 200)
    assert torch.isfinite(y).all()


def test_scale_guard_starts_with_identity_gain_and_bounded_gain():
    head = ScaleGuardHead(32, dims=3, initial_speed=0.5, max_gain=0.5)
    output = head(torch.randn(8, 32))
    assert torch.allclose(output["gain"], torch.ones_like(output["gain"]))
    assert torch.all(output["speed"] > 0)
    assert torch.all(output["gain"] >= 0.5)
    assert torch.all(output["gain"] <= 1.5)
    assert torch.allclose(torch.linalg.vector_norm(output["direction"], dim=-1),
                          torch.ones(8), atol=1e-5)


def test_scale_guard_loss_has_finite_gradients_and_masks_zero_speed_direction():
    prediction = torch.randn(6, 3, requires_grad=True)
    target = torch.vstack([torch.zeros(1, 3), torch.randn(5, 3)])
    loss = scale_guard_loss(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
