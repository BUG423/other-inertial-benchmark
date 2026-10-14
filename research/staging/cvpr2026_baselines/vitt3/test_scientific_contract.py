"""UNEXECUTED AutoDL-only scientific contract drafts; not paper-fidelity evidence.

Parent conftest fails closed before collection without original-task tail receipt.
No local framework import, model execution or pytest invocation is authorized.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from inertial_benchmark.cfg import INPUT_KEYS
from inertial_benchmark.data.views import CHANNELS
from inertial_benchmark.nn.base import BaseModel, InputSpec

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cvpr2026_science_vitt3", HERE / "model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Model = getattr(module, "ViT3Velocity")


def make_model(**kwargs):
    torch.manual_seed(1043)
    return Model(InputSpec(window=200, dims=3), **kwargs)


def test_exact_xyz_and_plain_mse_from_yaml():
    profile = yaml.safe_load((HERE / "model_3d.yaml").read_text())
    assert profile["registered"] is False and profile["loss"] == "mse"
    assert set(profile["input"]) <= set(INPUT_KEYS) and "channels" not in profile["input"]
    model = Model(InputSpec.from_dict(profile["input"]), **profile["args"])
    assert isinstance(model, BaseModel) and model.model_cfg["registered"] is False
    assert model.model_cfg["args"] == profile["args"]
    assert tuple(model.model_cfg["input_channels"]) == CHANNELS
    assert set(model.model_cfg["input"]) <= set(INPUT_KEYS)
    imu = torch.randn(2, 6, 200)
    output = model(imu)
    assert set(output) == {"vel"} and output["vel"].shape == (2, 3)
    assert torch.isfinite(output["vel"]).all()
    target = torch.randn(2, 3)
    loss, logs = model.loss(output, {"imu": imu, "target": target})
    assert torch.equal(loss, (output["vel"] - target).square().mean())
    assert set(logs) == {"mse"}
    assert isinstance(model.velocity_head, nn.Linear) and model.velocity_head.out_features == 3
    model.set_loss("mse")
    for name, kwargs in (("mse_sum", {}), ("mse_l1", {}), ("mse", {"alpha": 0})):
        with pytest.raises(ValueError):
            model.set_loss(name, **kwargs)
    assert model.loss_name == "mse" and model.loss_kwargs == {}


@pytest.mark.parametrize("changes", [
    {"window": 100}, {"window": 200.0}, {"dims": 2}, {"rate": 100},
    {"frame": "body"}, {"orientation": "estimated"}, {"remove_gravity": True},
    {"target": "displacement"}, {"history": 2}, {"history_stride": 10},
    {"output_steps": 2}, {"extra_inputs": ("init_velocity",)}, {"overlap": "center"},
    {"channels": tuple(reversed(CHANNELS))},
])
def test_reject_expanded_or_different_scientific_contract(changes):
    values = {"window": 200, "dims": 3, **changes}
    with pytest.raises(ValueError):
        Model(InputSpec(**values))


def test_reject_pretraining_and_input_shape():
    with pytest.raises(ValueError):
        make_model(pretrained=True)
    model = make_model()
    for shape in ((2, 6, 199), (2, 3, 200), (2, 1, 6, 200)):
        with pytest.raises(ValueError):
            model(torch.zeros(shape))


def test_real_z_gradient_and_update_without_padding():
    model = make_model().train()
    imu = torch.randn(2, 6, 200, requires_grad=True)
    output = model(imu)
    # Only the third component has a residual; XY padding would fail this gate.
    target = output["vel"].detach().clone()
    target[:, 2] += 1.0
    loss, _ = model.loss(output, {"imu": imu, "target": target})
    loss.backward()
    assert imu.grad is not None and torch.isfinite(imu.grad).all()
    assert imu.grad[:, 2].abs().sum() > 0 and imu.grad[:, 5].abs().sum() > 0
    assert model.velocity_head.weight.grad[2].abs().sum() > 0
    assert model.velocity_head.bias.grad[2].abs() > 0
    assert torch.equal(model.velocity_head.weight.grad[:2],
                       torch.zeros_like(model.velocity_head.weight.grad[:2]))
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    before = model.velocity_head.weight.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert not torch.equal(before[2], model.velocity_head.weight[2])


def test_eval_independence_no_parameter_mutation_and_exact_reload():
    model = make_model().eval()
    first, distractor = torch.randn(2, 6, 200), torch.randn(2, 6, 200)
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        a = model(first)["vel"]
        model(distractor)
        b = model(first)["vel"]
        split = torch.cat([model(sample[None])["vel"] for sample in first])
    assert torch.equal(a, b)
    assert torch.allclose(a, split, atol=2e-6, rtol=2e-5)
    assert all(torch.equal(tensor, state[name]) for name, tensor in model.state_dict().items())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        assert torch.equal(a, restored(first)["vel"])


def test_provenance_remains_static_unregistered_not_capacity_accepted():
    record = json.loads((HERE / "provenance.json").read_text())
    assert record["implementation_status"] == "static_draft"
    assert record["class_name"] == "ViT3Velocity"
    for field in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled", "registered"):
        assert record[field] is False


def test_negative_dot_swiglu_inner_gradient_against_autograd():
    core = module.TemporalTTT(16, 2).double()
    key, value = torch.randn(1, 2, 7, 8, dtype=torch.float64), torch.randn(1, 2, 7, 8, dtype=torch.float64)
    first = core.w1.detach().clone().requires_grad_()
    second = core.w2.detach().clone().requires_grad_()
    inner = -((key @ first) * F.silu(key @ second) * value).sum() * core.scale / 7
    gradients = torch.autograd.grad(inner, (first, second))
    expected = tuple(weight - gradient / (gradient.norm(dim=-2, keepdim=True) + 1)
                     for weight, gradient in zip((first, second), gradients))
    actual = core.inner_swiglu(key, value, first, second)
    for result, oracle in zip(actual, expected):
        assert torch.allclose(result, oracle, atol=1e-12, rtol=1e-10)


def test_three_tap_inner_gradient_against_autograd():
    core = module.TemporalTTT(16, 2).double()
    key, value = torch.randn(1, 8, 7, dtype=torch.float64), torch.randn(1, 8, 7, dtype=torch.float64)
    weight = core.w3.detach().clone().requires_grad_()
    inner = -(F.conv1d(key, weight, padding=1, groups=8) * value).sum() * core.scale / 7
    gradient, = torch.autograd.grad(inner, (weight,))
    expected = weight - gradient / (gradient.norm(dim=(-2, -1), keepdim=True) + 1)
    result = core.inner_conv(key, value, weight)
    assert torch.allclose(result[0], expected, atol=1e-12, rtol=1e-10)
    assert core.scale == 3.0 ** -0.5 and core.w3.shape[-1] == 3


def test_per_sample_inner_updates_and_rotary_norm_without_state_carry():
    core = module.TemporalTTT(16, 2).double()
    inputs = torch.randn(2, 9, 16, dtype=torch.float64)
    state = {name: value.clone() for name, value in core.state_dict().items()}
    together = core(inputs)
    apart = torch.cat([core(row[None]) for row in inputs])
    assert torch.allclose(together, apart, atol=1e-12, rtol=1e-10)
    assert all(torch.equal(state[name], value) for name, value in core.state_dict().items())
    tensor = torch.randn(2, 2, 9, 8, dtype=torch.float64)
    assert torch.allclose(core.time_position(tensor).square().sum(-1), tensor.square().sum(-1),
                          atol=1e-12, rtol=1e-10)
    assert not torch.equal(core.time_position(tensor)[..., 1:, :], tensor[..., 1:, :])
