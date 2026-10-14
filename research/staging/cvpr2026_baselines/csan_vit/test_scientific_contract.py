"""SERVER-ONLY pending tests; never executed/imported during static drafting.

The parent collection guard requires an accepted AutoDL original-task tail
receipt. These are scientific mechanism tests, not accuracy/capacity claims.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from inertial_benchmark.cfg import INPUT_KEYS
from inertial_benchmark.data.views import CHANNELS
from inertial_benchmark.nn.base import InputSpec


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("staging_cvpr2026_csan_model", HERE / "model.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
CSaNVelocity = _module.CSaNVelocity
CSaNAttention = _module.CSaNAttention


def canonical(**changes):
    fields = dict(window=200, rate=200, frame="gravity_world", orientation="reference",
                  remove_gravity=False, target="avg_velocity", dims=3, channels=CHANNELS,
                  history=0, history_stride=0, output_steps=0, extra_inputs=())
    fields.update(changes)
    return InputSpec(**fields)


def reference_attention(layer, x, *, include_private=True, early_tanh=False):
    """Independent equations: separate QKV chunks, explicit head layouts."""
    batch, tokens, dim = x.shape
    heads, width = layer.num_heads, layer.head_dim
    q, k, v = F.linear(x, layer.qkv.weight, layer.qkv.bias).chunk(3, dim=-1)
    q, k, v = [value.reshape(batch, tokens, heads, width).transpose(1, 2)
               for value in (q, k, v)]
    scores = torch.einsum("bhtd,bhsd->bhts", q, k) / (width ** 0.5)
    eta = 0.25
    statistic = torch.where(scores > eta, scores,
                            eta * (scores.clamp(max=eta) / eta - 1).exp()).mean(-1)
    u = F.linear(x, layer.u.weight, layer.u.bias).reshape(batch, tokens, heads, -1)
    importance = (statistic.transpose(1, 2)[..., None] * u.sigmoid()).reshape(batch, tokens, -1)
    reduced = F.linear(importance, layer.reduce.weight, layer.reduce.bias)
    hidden = reduced * F.linear(reduced, layer.act.proj.weight, layer.act.proj.bias).sigmoid()
    calibration = {}
    for name in ("ka", "kv", "temp", "magk"):
        linear = getattr(layer, name)
        value = F.linear(hidden, linear.weight, linear.bias)
        calibration[name] = value.tanh() if early_tanh else value
    ka = (calibration["ka"] + 1).reshape(batch, tokens, heads, width).transpose(1, 2)
    kv = calibration["kv"].reshape(batch, tokens, heads, width).transpose(1, 2)
    temp = calibration["temp"].transpose(1, 2)[..., None] + 1
    magk = calibration["magk"].transpose(1, 2)[:, :, None, :] + 1
    public_weights = (scores * temp).softmax(-1) * magk
    public_value = torch.einsum("bhts,bhsd->bhtd", public_weights, v) * ka
    result = public_value + v * kv if include_private else public_value
    result = result.transpose(1, 2).reshape(batch, tokens, dim)
    return F.linear(result, layer.proj.weight, layer.proj.bias)


def test_compensation_full_values_and_all_input_parameter_gradients():
    torch.manual_seed(7081)
    layer = CSaNAttention(24, 3).double()
    # Nontrivial compensators (not the almost-neutral initialization) expose all
    # row/column/private terms and distinguish an ordinary attention substitute.
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(mean=0.02, std=0.15)
    x = torch.randn(2, 7, 24, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    probe = torch.randn_like(x)
    actual = layer(x)
    expected = reference_attention(layer, other)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    parameters = tuple(layer.parameters())
    actual_grad = torch.autograd.grad((actual * probe).sum(), (x, *parameters))
    expected_grad = torch.autograd.grad((expected * probe).sum(), (other, *parameters))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value, rtol=2e-9, atol=1e-10)
        assert torch.isfinite(actual_value).all()
        assert actual_value.abs().sum() > 0
    no_private = reference_attention(layer, x.detach(), include_private=False)
    earlier = reference_attention(layer, x.detach(), early_tanh=True)
    assert not torch.allclose(actual.detach(), no_private)
    assert not torch.allclose(actual.detach(), earlier)


def test_last_effective_linear_plus_one_not_shadowed_tanh_definition():
    layer = CSaNAttention(8, 2).double()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.qkv.weight[16:].copy_(torch.eye(8, dtype=torch.float64))
        layer.proj.weight.copy_(torch.eye(8, dtype=torch.float64))
        layer.ka.bias.fill_(4)
        layer.kv.bias.fill_(5)
        layer.temp.bias.fill_(3)
        layer.magk.bias.fill_(2)
    x = torch.arange(24, dtype=torch.float64).reshape(1, 3, 8) / 10
    # scores=0: row temperature irrelevant, col gain=3, public gain=5,
    # private gain=5. No tanh or post-hoc budget normalization is permissible.
    expected = 15 * x.mean(dim=1, keepdim=True) + 5 * x
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=1e-14)
    assert not torch.allclose(layer(x), reference_attention(layer, x, early_tanh=True))


def test_softgshift_and_original_quasilinear_values_gradients():
    torch.manual_seed(7082)
    activation = _module.SoftGShift(5).double()
    x = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(activation(x), x * F.linear(x, activation.proj.weight,
                                                          activation.proj.bias).sigmoid())
    points = torch.tensor([-1.0, 0.0, 0.2, 0.3, 1.0], dtype=torch.float64, requires_grad=True)
    actual = CSaNAttention.quasi_linear(points)
    expected = torch.where(points > 0.25, points, 0.25 * (points / 0.25 - 1).exp())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), points)[0],
                               torch.where(points > 0.25, torch.ones_like(points),
                                           (points / 0.25 - 1).exp()))


def test_default_t200_scratch_finite_backward_and_update():
    torch.manual_seed(7083)
    model = CSaNVelocity(canonical())
    model.train()
    imu = torch.randn(2, 6, 200)
    batch = {"imu": imu, "target": torch.randn(2, 3)}
    output = model(imu)
    assert set(output) == {"vel"} and output["vel"].shape == (2, 3)
    loss, parts = model.loss(output, batch)
    torch.testing.assert_close(loss, F.mse_loss(output["vel"], batch["target"]))
    assert set(parts) == {"mse"} and torch.isfinite(loss)
    before = model.velocity_head.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.velocity_head.weight)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_real_z_only_supervision_and_one_shared_linear_head():
    torch.manual_seed(7084)
    model = CSaNVelocity(canonical(), d_model=24, n_layers=1, num_heads=3)
    imu = torch.randn(2, 6, 200)
    output = model(imu)
    target = output["vel"].detach().clone()
    target[:, 2] += 1
    loss, _ = model.loss(output, {"imu": imu, "target": target})
    loss.backward()
    assert model.velocity_head.weight.grad[2].abs().sum() > 0
    torch.testing.assert_close(model.velocity_head.weight.grad[:2],
                               torch.zeros_like(model.velocity_head.weight.grad[:2]))
    assert model.token_projection.weight.grad.abs().sum() > 0
    features = model.encode_window(imu)
    torch.testing.assert_close(model(imu)["vel"], F.linear(features, model.velocity_head.weight,
                                                         model.velocity_head.bias))
    assert isinstance(model.velocity_head, nn.Linear) and model.velocity_head.out_features == 3


def test_eval_independent_windows_batches_axis_order_and_reload():
    torch.manual_seed(7085)
    model = CSaNVelocity(canonical(), d_model=24, n_layers=1, num_heads=3, dropout=0.2).eval()
    imu = torch.randn(2, 6, 200)
    with torch.no_grad():
        first = model(imu)["vel"]
        model(torch.randn_like(imu))
        torch.testing.assert_close(first, model(imu)["vel"], rtol=0, atol=0)
        separate = torch.cat([model(window[None])["vel"] for window in imu])
        torch.testing.assert_close(first, separate, rtol=1e-5, atol=1e-6)
        assert not torch.allclose(first, model(imu[:, [3, 4, 5, 0, 1, 2]])["vel"])
        restored = CSaNVelocity(canonical(), **model.model_cfg).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(first, restored(imu)["vel"], rtol=0, atol=0)
    assert not hasattr(model, "state") and not hasattr(model, "hidden_state")
    with pytest.raises(TypeError):
        model(imu, {"target": torch.randn(2, 3)})


@pytest.mark.parametrize("changes", [
    {"dims": 2}, {"window": 199}, {"rate": 100}, {"frame": "body"},
    {"frame": "gravity_yaw_local"}, {"orientation": "device"}, {"remove_gravity": True},
    {"target": "displacement"}, {"target": "frame_velocity"}, {"output_steps": 2},
    {"overlap": "center"}, {"history": 1}, {"history_stride": 1}, {"extra_inputs": ("orientation",)},
    {"extra_inputs": ("init_velocity",)}, {"channels": CHANNELS[::-1]},
])
def test_rejects_noncanonical_input_contract(changes):
    with pytest.raises(ValueError):
        CSaNVelocity(canonical(**changes))


@pytest.mark.parametrize("args", [
    {"pretrained": True}, {"pretrained": 0}, {"d_model": 0}, {"d_model": 25},
    {"n_layers": 0}, {"n_layers": True}, {"num_heads": 0},
    {"mlp_ratio": 0}, {"mlp_ratio": float("nan")},
    {"dropout": -0.1}, {"dropout": 1}, {"dropout": float("nan")},
])
def test_rejects_invalid_recipe(args):
    with pytest.raises(ValueError):
        CSaNVelocity(canonical(), **args)


@pytest.mark.parametrize("shape", [(2, 6, 199), (2, 200, 6), (2, 5, 200), (2, 1, 6, 200)])
def test_runtime_shape_rejected(shape):
    model = CSaNVelocity(canonical(), d_model=24, n_layers=1, num_heads=3)
    with pytest.raises(ValueError, match="expected input"):
        model(torch.zeros(shape))


def test_config_and_provenance_stay_unregistered_draft_plain_mse():
    cfg = yaml.safe_load((HERE / "model_3d.yaml").read_text())
    provenance = json.loads((HERE / "provenance.json").read_text())
    model = CSaNVelocity(InputSpec.from_dict(cfg["input"]), **cfg["args"])
    assert model.model_cfg == cfg["args"]
    assert cfg["registered"] is False and cfg["loss"] == "mse"
    assert cfg["input_channels"] == list(CHANNELS)
    assert "channels" not in cfg["input"] and set(cfg["input"]) <= set(INPUT_KEYS)
    assert cfg["args"]["pretrained"] is False
    assert provenance["implementation_status"] == "static_draft"
    for key in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled", "registered"):
        assert provenance[key] is False
    assert model.inference_contract == "window_noncausal"
    assert model.training_initialization == "scratch_supervised"
    model.set_loss("mse")
    with pytest.raises(ValueError, match="MSE"):
        model.set_loss("mse_sum")
    with pytest.raises(ValueError, match="MSE"):
        model.set_loss("mse", auxiliary_weight=1)
    assert "register_model" not in (HERE / "model.py").read_text()
