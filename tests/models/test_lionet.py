"""L-IONet causal gates and genuine XYZ polar-head contract."""

import pytest
import torch

from inertial_benchmark.nn.base import InputSpec
from inertial_benchmark.models.lionet.model import CausalGatedBlock, LIONet


def test_causal_gate_has_no_future_dependency():
    torch.manual_seed(91)
    block = CausalGatedBlock(4, 2, 8).eval()
    x = torch.randn(2, 4, 32)
    perturbed = x.clone()
    perturbed[:, :, 16:] += 100
    for a, b in zip(block(x), block(perturbed)):
        torch.testing.assert_close(a[:, :, :16], b[:, :, :16], rtol=0, atol=0)


@pytest.mark.parametrize("dims", [2, 3])
def test_polar_loss_and_learned_vertical(dims):
    model = LIONet(InputSpec(window=400, dims=dims))
    imu = torch.randn(2, 6, 400)
    out = model(imu)
    assert out["vel"].shape == (2, dims)
    torch.testing.assert_close(out["vel"][:, :2].norm(dim=-1), out["speed"].abs())
    target = torch.tensor([[0.8, -0.3, 1.7], [-0.4, 0.2, -0.8]])[:, :dims]
    loss, items = model.loss(out, {"imu": imu, "target": target}, epoch=0)
    loss.backward()
    assert torch.isfinite(loss)
    if dims == 3:
        assert "vertical" in items
        assert model.head.weight.grad[2].abs().sum() > 0
        assert model.head.bias.grad[2].abs() > 0
    clone = LIONet(model.input_spec).eval()
    clone.load_state_dict(model.state_dict())
    model.eval()
    torch.testing.assert_close(model(imu)["vel"], clone(imu)["vel"], rtol=0, atol=0)
