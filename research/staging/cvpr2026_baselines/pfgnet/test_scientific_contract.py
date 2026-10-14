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
spec = importlib.util.spec_from_file_location("cvpr2026_science_pfgnet", HERE / "model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Model = getattr(module, "PFGNetVelocity")


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
    assert record["class_name"] == "PFGNetVelocity"
    for field in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled", "registered"):
        assert record[field] is False


def test_temporal_frequency_cues_literal_derivative_laplacian_and_variance():
    gate = module.PeripheralFrequencyGate1D(4).double()
    x = torch.randn(2, 4, 19, dtype=torch.float64)
    first = F.conv1d(x, torch.tensor([-1., 0., 1.], dtype=x.dtype).view(1, 1, 3).expand(4, 1, 3),
                     padding=1, groups=4)
    second = F.conv1d(x, torch.tensor([1., -2., 1.], dtype=x.dtype).view(1, 1, 3).expand(4, 1, 3),
                      padding=1, groups=4)
    mean = F.avg_pool1d(x, 3, stride=1, padding=1)
    variance = (F.avg_pool1d(x.square(), 3, stride=1, padding=1) - mean.square()).clamp_min(0)
    oracle = torch.cat(((first.square() + 1e-6).sqrt().mean(1, keepdim=True),
                        second.abs().mean(1, keepdim=True), variance.mean(1, keepdim=True)), 1)
    assert torch.equal(gate.frequency_cues(x), oracle)
    constant = gate.frequency_cues(torch.ones(1, 4, 19, dtype=x.dtype))
    assert torch.equal(constant[:, 1:, 2:-2], torch.zeros_like(constant[:, 1:, 2:-2]))
    assert gate.kernels == (9, 15, 31)


def test_zero_scale_gate_is_uniform_and_center_suppression_literal():
    gate = module.PeripheralFrequencyGate1D(4).double()
    x = torch.randn(2, 4, 19, dtype=torch.float64)
    with torch.no_grad():
        gate.gate_head.weight.zero_()
        gate.gate_head.bias.zero_()
    weights = gate.weights(x)
    assert torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]))
    oracle = sum(branch(x) for branch in gate.branches) / 3
    assert torch.allclose(gate(x), oracle, atol=1e-12, rtol=1e-10)
    branch = gate.branches[0]
    with torch.no_grad():
        branch.beta.fill_(0.5)
    assert torch.equal(branch(x), branch.large(x) - torch.tanh(branch.beta) * branch.center(x))
    assert branch.large.groups == 4 and branch.center.groups == 4


def test_released_grn_formula_and_all_channel_identity_adaptation():
    grn = module.TemporalGRN(4).double()
    x = torch.randn(2, 4, 19, dtype=torch.float64)
    assert torch.equal(grn(x), x + x / (x.norm(dim=-1, keepdim=True) + 1e-6))
    assert torch.equal(grn.gamma, torch.ones_like(grn.gamma))
    rep = module.TemporalRepDW(4, 7).double()
    assert torch.equal(rep.identity(x), x)
    assert rep.large.kernel_size == (7,) and rep.large.groups == 4
