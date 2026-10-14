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
spec = importlib.util.spec_from_file_location("cvpr2026_science_ucan", HERE / "model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Model = getattr(module, "UCANVelocity")


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
    assert record["class_name"] == "UCANVelocity"
    for field in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled", "registered"):
        assert record[field] is False


def test_hedgehog_identity_positive_feature_map_and_signed_fourier_guard():
    feature = module.HedgehogFeatureMap(8).double()
    x = torch.randn(2, 2, 7, 8, dtype=torch.float64)
    oracle = torch.cat((2*x, -2*x), -1).softmax(-1)
    assert torch.equal(feature(x), oracle)
    assert (feature(x) > 0).all() and torch.allclose(feature(x).sum(-1), torch.ones_like(x[..., 0]))
    coefficient = torch.tensor([[[2., 0.], [-1., 0.]]], dtype=torch.float64)
    normalized = module.FourierPosition1D.normalized(coefficient)
    assert torch.equal(normalized[..., 0], coefficient[..., 0])  # signed sum, NOT abs/L1.
    assert torch.isfinite(normalized).all()


def test_fourier_position_exact_minus_rotation_and_padded_one_convention():
    position = module.FourierPosition1D(16, 2).double()
    x = torch.randn(2, 2, 7, 16, dtype=torch.float64)
    phase = torch.arange(7, dtype=x.dtype)[:, None] * position.inv_frequency[None]
    sine = torch.einsum("tf,hfg->htg", phase.sin(), position.normalized(position.sin_coefficient))
    cosine = torch.einsum("tf,hfg->htg", phase.cos(), position.normalized(position.cos_coefficient))
    sine = F.pad(sine, (0, 8-sine.shape[-1]), value=1)
    cosine = F.pad(cosine, (0, 8-cosine.shape[-1]), value=1)
    sine, cosine = torch.cat((sine, sine), -1), torch.cat((cosine, cosine), -1)
    first, second = x.chunk(2, -1)
    assert torch.equal(position(x), x*cosine - torch.cat((-second, first), -1)*sine)
    assert "inv_frequency" in dict(position.named_buffers())


def test_window_map_sharing_own_value_and_padded_key_mask():
    sender = module.WindowAttention1D(16, 2, 4, False).double()
    receiver = module.WindowAttention1D(16, 2, 4, True).double()
    x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
    output, attention = sender(x)
    received, returned = receiver(x, attention)
    assert returned is attention and output.shape == received.shape == x.shape
    assert not hasattr(receiver, "qkv") and receiver.value is not sender.qkv
    assert torch.equal(attention[[1, 3], :, :, -1], torch.zeros_like(attention[[1, 3], :, :, -1]))
    assert torch.allclose(attention.sum(-1), torch.ones_like(attention[..., 0]))
    received.square().mean().backward()
    assert sender.qkv.weight.grad is not None and sender.qkv.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        receiver(x)


def test_dual_global_channel_literal_and_receiver_uses_raw_same_forward_qk():
    core = module.DualFusion1D(32, 2, False).double()
    receiver = module.DualFusion1D(32, 2, True).double()
    x = torch.randn(2, 7, 32, dtype=torch.float64)
    output, (q, k) = core(x)
    v = core.value(x)
    heads = lambda t: t.reshape(2, 7, 2, 8).transpose(1, 2)
    qp, kp = core.query_map(heads(q)), core.key_map(heads(k))
    denominator = qp @ kp.mean(-2, keepdim=True).transpose(-2, -1) + 1e-6
    qpos, kpos = core.position(qp), core.position(kp)
    context = (kpos.transpose(-2, -1)*7**-.5) @ (heads(v)*7**-.5)
    global_part = ((qpos @ context)/denominator).transpose(1, 2).reshape(2, 7, 16)
    global_part += core.local_value(v.transpose(1, 2)).transpose(1, 2)
    qc, kc = F.normalize(q.transpose(1, 2), dim=-1), F.normalize(k.transpose(1, 2), dim=-1)
    channel = (((qc @ kc.transpose(-2, -1))*core.temperature).softmax(-1)
               @ v.transpose(1, 2)).transpose(1, 2)
    assert torch.equal(output, core.proj(torch.cat((global_part, channel), -1)))
    received, returned = receiver(x, (q, k))
    assert returned[0] is q and returned[1] is k and received.shape == output.shape
    assert not hasattr(receiver, "query") and not hasattr(receiver, "key")
    assert receiver.value is not core.value
    with pytest.raises(ValueError):
        receiver(x)


def test_selective_first_sixteen_large_kernel_literal_and_temporal_field():
    module_ = module.SelectiveLargeKernel1D(32, 2).double()
    x = torch.randn(2, 9, 32, dtype=torch.float64)
    mixed = module_.ffn(x).transpose(1, 2)
    oracle = torch.cat((module_.large(mixed[:, :16]), mixed[:, 16:]), 1)
    assert torch.equal(module_(x), x + module_.merge(oracle).transpose(1, 2))
    first, second = module_.large.large
    assert first.kernel_size == second.kernel_size == (5,)
    assert first.dilation == (1,) and second.dilation == (2,)
    assert 1 + 4 + 4*2 == 13
    assert module_.fine_dim == 16


def test_pair_cache_never_persisted_and_receivers_cannot_create_own_cache():
    model = make_model()
    seen = []
    hook = model.receiving[0].register_forward_pre_hook(lambda block, args: seen.append(args[1]))
    model(torch.randn(2, 6, 200))
    hook.remove()
    assert len(seen) == 1 and seen[0][0].grad_fn is not None
    assert seen[0][1][0].grad_fn is not None and seen[0][1][1].grad_fn is not None
    for block in (model.sharing[0], model.receiving[0]):
        assert not hasattr(block, "cache") and not hasattr(block, "shared_qk")
    with pytest.raises(ValueError):
        model.receiving[0](torch.randn(2, 50, 32))
