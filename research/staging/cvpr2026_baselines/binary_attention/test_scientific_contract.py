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
spec = importlib.util.spec_from_file_location("cvpr2026_science_binary_attention", HERE / "model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Model = getattr(module, "BinaryAttentionVelocity")


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
    assert record["class_name"] == "BinaryAttentionVelocity"
    for field in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled", "registered"):
        assert record[field] is False


def test_published_sign_zero_and_symmetric_ste_closed_boundaries():
    x = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
    result = module.PublishedSTESign.apply(x)
    assert torch.equal(result, torch.tensor([-1., -1., 1., 1., 1.]))
    result.sum().backward()
    assert torch.equal(x.grad, torch.tensor([0., 1., 1., 1., 0.]))


def test_sign_token_channel_scale_and_pv_quantizer_source_formula():
    x = torch.tensor([[[[-2., -1.], [0., 1.], [2., 3.]]]])
    expected = x.abs().mean(-2, keepdim=True).mean(-1, keepdim=True) * (x.sign() + (x == 0))
    assert torch.equal(module.scaled_sign(x), expected)
    value = torch.tensor([[[-3., -2., -1.], [1., 2., 3.]]], requires_grad=True)
    maximum = value.detach().abs().amax(-2, keepdim=True)
    scale = 127 / (maximum + 1e-6)
    output = module.PublishedSymmetricQuant.apply(value)
    assert torch.equal(output, (value * scale).round() / (scale + 1e-6))
    assert output.abs().max() > 2  # This is NOT forward clipping to [-2,2].
    output.sum().backward()
    assert torch.equal(value.grad, torch.tensor([[[0., 0., 1.], [1., 0., 0.]]]))
    probabilities = torch.tensor([[-.1, .0, .2, .9, 1.2]], requires_grad=True)
    quantized = module.quantize_probability(probabilities)
    assert torch.allclose(quantized * 255, (probabilities.detach() * 255).round().clamp(0, 255))
    quantized.sum().backward()
    assert torch.equal(probabilities.grad, torch.tensor([[0., 1., 1., 1., 0.]]))


@pytest.mark.parametrize("pv_quant", [False, True])
def test_floating_attention_exact_core_equation_and_1d_cls_bias(pv_quant):
    core = module.BinaryQKAttention1D(16, 2, 4, pv_quant).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    q, k, v = core.qkv(x).reshape(2, 5, 3, 2, 8).permute(2, 0, 3, 1, 4).unbind(0)
    logits = module.scaled_sign(q) @ module.scaled_sign(k).transpose(-2, -1) * 8 ** -0.5
    bias = core.relative_bias[core.relative_index].permute(2, 0, 1)
    probability = (logits + bias[None]).softmax(-1)
    if pv_quant:
        probability = module.quantize_probability(probability)
        v = module.PublishedSymmetricQuant.apply(v)
    oracle = core.proj((probability @ v).transpose(1, 2).reshape(2, 5, 16))
    assert torch.equal(core(x), oracle)
    index = core.relative_index
    assert torch.equal(index[1:, 1:], torch.arange(4)[:, None] - torch.arange(4)[None, :] + 3)
    assert len({index[0, 0].item(), index[0, 1].item(), index[1, 0].item()}) == 3
    assert make_model().hardware_implementation == "floating_fake_quant_reference"
