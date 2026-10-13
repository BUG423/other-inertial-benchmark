"""Explicit server-only staging tests, outside the default accepted-model suite."""

import importlib.util
from pathlib import Path

import pytest
import torch

from inertial_benchmark.nn.base import InputSpec

spec = importlib.util.spec_from_file_location("staging_m2eit", Path(__file__).with_name("model.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_haar_constant_impulse_energy_and_inverse():
    x = torch.tensor([[[2., 2., 0., 4., -1., 3.]]], dtype=torch.float64)
    low, high = module.haar_pair(x)
    assert high[0, 0, 0] == 0  # constant pair has no detail component
    rebuilt = torch.stack(((low + high) / 2 ** .5, (low - high) / 2 ** .5), -1).flatten(-2)
    torch.testing.assert_close(rebuilt, x, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(low.square().sum() + high.square().sum(), x.square().sum())
    with pytest.raises(ValueError):
        module.haar_pair(x[..., :5])


def test_selective_scan_matches_independent_three_step_solution_and_gradients():
    # Analytic scalar solution with input-dependent B/C/delta: not a replay of a scan loop.
    x = torch.tensor([[[2.], [-1.], [3.]]], dtype=torch.float64, requires_grad=True)
    dt = torch.tensor([[[.1], [.3], [.2]]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[[1.], [2.], [-.5]]], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([[[2.], [-1.], [3.]]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[-2.]], dtype=torch.float64, requires_grad=True)
    skip = torch.tensor([.4], dtype=torch.float64, requires_grad=True)
    actual = module.selective_scan_reference(x, dt, a, b, c, skip)
    h0 = dt[0, 0, 0] * b[0, 0, 0] * x[0, 0, 0]
    h1 = torch.exp(a[0, 0] * dt[0, 1, 0]) * h0 + dt[0, 1, 0] * b[0, 1, 0] * x[0, 1, 0]
    h2 = torch.exp(a[0, 0] * (dt[0, 1, 0] + dt[0, 2, 0])) * h0
    h2 = h2 + torch.exp(a[0, 0] * dt[0, 2, 0]) * dt[0, 1, 0] * b[0, 1, 0] * x[0, 1, 0]
    h2 = h2 + dt[0, 2, 0] * b[0, 2, 0] * x[0, 2, 0]
    expected = torch.stack((h0, h1, h2)).view(1, 3, 1) * c + skip * x
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    parameters = (x, dt, a, b, c, skip)
    ga = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
    ge = torch.autograd.grad(expected.sum(), parameters)
    for actual_grad, expected_grad in zip(ga, ge):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


def test_one_direction_reference_is_causal_and_input_dependent():
    model = module.SelectiveMambaReference(8, state_dim=3).eval()
    x = torch.randn(2, 9, 8)
    changed = x.clone()
    changed[:, 5:] += 100
    torch.testing.assert_close(model(x)[:, :5], model(changed)[:, :5], rtol=0, atol=0)
    assert not torch.equal(model(x)[:, 5:], model(changed)[:, 5:])


@pytest.mark.parametrize("dims", [2, 3])
def test_independent_experts_router_real_xyz_and_default_window(dims):
    model = module.M2EITDraft(InputSpec(window=200, dims=dims), feature_dim=16,
                             state_dim=3, attention_heads=4, head_hidden=8)
    imu = torch.randn(2, 6, 200)
    out = model(imu)
    assert out["vel"].shape == (2, dims)
    assert out["expert_features"].shape == (2, 3, 16)
    torch.testing.assert_close(out["router_weights"].sum(1), torch.ones(2))
    assert (out["router_weights"] > 0).all()
    target = torch.tensor([[.5, -.7, 1.2], [-.3, .4, -1.1]])[:, :dims]
    loss, _ = model.loss(out, {"imu": imu, "target": target})
    if dims == 3:
        # XY is taken from this exact raw XYZ task, rather than a fabricated Z=0 run.
        raw_xyz_gradient = torch.autograd.grad(loss, out["vel"], retain_graph=True)[0]
        assert raw_xyz_gradient[:, 2].abs().sum() > 0
        xy_error = (out["vel"][:, :2] - target[:, :2]).square().sum(-1).mean()
        xy_gradient = torch.autograd.grad(xy_error, out["vel"], retain_graph=True)[0]
        assert xy_gradient[:, :2].abs().sum() > 0 and xy_gradient[:, 2].abs().sum() == 0
        assert target[:, 2].abs().sum() > 0
    loss.backward()
    for expert in (model.spatial, model.temporal, model.frequency,
                   model.temporal_to_spatial, model.frequency_to_spatial, model.router):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())
    assert model.head[-1].weight.grad[dims - 1].abs().sum() > 0
    assert torch.isfinite(loss)


def test_exact_eval_checkpoint_reload_mask_and_no_registry_side_effect(tmp_path):
    from inertial_benchmark.nn.registry import list_models

    assert "m2eit" not in list_models()
    args = dict(feature_dim=16, state_dim=3, attention_heads=4, head_hidden=8)
    model = module.M2EITDraft(InputSpec(window=200, dims=3), **args).eval()
    clone = module.M2EITDraft(model.input_spec, **args).eval()
    checkpoint = tmp_path / "staging-m2eit.pt"
    torch.save({"state_dict": model.state_dict(), "input_spec": model.input_spec.to_dict()}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert saved["input_spec"] == model.input_spec.to_dict()
    clone.load_state_dict(saved["state_dict"])
    imu = torch.randn(2, 6, 200)
    first, second = model(imu), clone(imu)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    loss, _ = model.loss(first, {"imu": imu, "target": torch.randn(2, 3),
                               "mask": torch.zeros(2, 1)})
    assert loss == 0
    loss.backward()


def test_paper_default_profile_b2_real_xyz_backward():
    # This default-width/original-window probe is distinct from reduced-width math tests.
    model = module.M2EITDraft(InputSpec(window=200, dims=3))
    imu = torch.randn(2, 6, 200)
    out = model(imu)
    assert out["vel"].shape == (2, 3)
    loss, _ = model.loss(out, {"imu": imu, "target": torch.tensor([[.4, -.3, 1.], [-.8, .2, -1.]])})
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head[-1].weight.grad[2].abs().sum() > 0
    assert len(model.spatial) == 6 and len(model.temporal) == len(model.frequency) == 2


@pytest.mark.parametrize("changes", [{"dims": 1}, {"window": 199},
                                     {"target": "displacement"}, {"frame": "body"}])
def test_reject_incompatible_task_contract(changes):
    with pytest.raises(ValueError):
        module.M2EITDraft(InputSpec(**({"window": 200, "dims": 3} | changes)))
