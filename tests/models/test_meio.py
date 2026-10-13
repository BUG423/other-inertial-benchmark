"""MEIO scientific loss phases and actual reduced-depth computation."""

import torch
from torch import nn

from inertial_benchmark.models.meio.loss import MEIOMultiExitLoss
from inertial_benchmark.models.meio.model import MEIONet
from inertial_benchmark.nn.base import InputSpec


def _model(**kwargs):
    return MEIONet(InputSpec(window=200, dims=3, target="displacement"),
                   feature_dim=16, layers=3, **kwargs)


def test_exit_weight_reduction_and_covariance_entropy_weighting():
    means = torch.tensor([[[.1, .2, .3], [.4, .5, .6]]], requires_grad=True)
    sigma = torch.tensor([[[.2, .2, .2], [.5, .5, .5]]], requires_grad=True)
    target = torch.tensor([[.2, .4, .8]])
    out = {"vel": means[:, -1], "logstd": sigma[:, -1],
           "all_vel": means, "all_logstd": sigma}
    fn = MEIOMultiExitLoss(switch_epoch=10)
    early, _ = fn(out, target, epoch=9)
    per_mse = ((means - target[:, None]) ** 2).mean((0, 2))
    torch.testing.assert_close(early, (per_mse * torch.tensor([1/3, 2/3])).mean())
    early.backward(retain_graph=True)
    assert sigma.grad is None
    means.grad = None
    late, _ = fn(out, target, epoch=10)
    weights = torch.sqrt(torch.tensor([.5, 1.]))
    weights /= weights.sum()
    expected = (.5 * (means - target[:, None]) ** 2 * torch.exp(-2 * sigma)
                + weights[None, :, None] * sigma).mean()
    torch.testing.assert_close(late, expected)
    late.backward()
    assert sigma.grad is not None and sigma.grad.abs().sum() > 0
    assert fn.phase(9) == "weighted_multiexit_mse"
    assert fn.phase(10) == "weighted_multiexit_nll"


def test_warmup_and_target_masks():
    model = _model()
    out = model(torch.randn(2, 6, 200))
    target = torch.randn(2, 3)
    fn = MEIOMultiExitLoss(switch_epoch=10, warmup_epochs=4)
    at_switch, _ = fn(out, target, epoch=10)
    before, _ = fn(out, target, epoch=9)
    torch.testing.assert_close(at_switch, before)
    value, _ = fn(out, target, epoch=12, mask=torch.zeros(2, 1))
    value.backward()
    assert value.item() == 0
    assert fn.phase(12) == "mse_nll_blend"
    assert fn.phase(14) == "weighted_multiexit_nll"


def test_every_exit_has_real_xyz_and_postswitch_covariance_gradients():
    model = _model()
    imu = torch.randn(2, 6, 200)
    out = model(imu)
    assert out["all_vel"].shape == out["all_logstd"].shape == (2, 3, 3)
    target = torch.tensor([[.3, -.2, .7], [-.4, .1, -.8]])
    value, _ = model.loss(out, {"imu": imu, "target": target}, epoch=11)
    value.backward()
    for head in model.heads:
        assert head.mean.weight.grad[2].abs().sum() > 0
        assert head.logstd.weight.grad.abs().sum() > 0


def test_early_exit_really_skips_deeper_compute_for_accepted_samples():
    model = _model(variance_threshold=1.1).eval()
    sizes = []

    class CountBlock(nn.Module):
        def forward(self, tokens):
            sizes.append(len(tokens))
            return tokens

    class ControlledHead(nn.Module):
        def forward(self, tokens):
            return tokens.mean(1)[:, :3], tokens[:, 0, :1]

    model.blocks = nn.ModuleList([CountBlock() for _ in range(3)])
    model.affines = nn.ModuleList([nn.Identity() for _ in range(3)])
    model.heads = nn.ModuleList([ControlledHead() for _ in range(3)])
    tokens = torch.ones(2, 20, 16)
    tokens[0] = 0
    out = model.forward_early_exit(tokens)
    assert sizes == [2, 1, 1]
    assert out["exit_depth"].tolist() == [1, 3]
    torch.testing.assert_close(out["vel"], torch.tensor([[0., 0., 0.], [1., 1., 1.]]))


def test_checkpoint_reload_and_full_depth_reference_are_exact():
    model = _model(force_full_depth=True).eval()
    clone = _model(force_full_depth=True).eval()
    clone.load_state_dict(model.state_dict())
    imu = torch.randn(2, 6, 200)
    for key, output in model(imu).items():
        torch.testing.assert_close(output, clone(imu)[key], rtol=0, atol=0)
    assert model(imu)["exit_depth"].tolist() == [3, 3]
