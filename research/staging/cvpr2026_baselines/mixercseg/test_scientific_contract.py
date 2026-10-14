"""SERVER-ONLY pending tests, never executed/imported during static drafting.

Root's collection guard requires accepted AutoDL original-task tail readiness.
These verify the narrow TransMixer1D architecture port, not full MixerCSeg,
source CUDA throughput, capacity acceptance, or inertial accuracy.
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
_spec = importlib.util.spec_from_file_location("staging_cvpr2026_mixercseg_model", HERE / "model.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
TransMixerVelocity = _module.TransMixerVelocity


def canonical(**changes):
    fields = dict(window=200, rate=200, frame="gravity_world", orientation="reference",
                  remove_gravity=False, target="avg_velocity", dims=3, channels=CHANNELS,
                  history=0, history_stride=0, output_steps=0, extra_inputs=())
    fields.update(changes)
    return InputSpec(**fields)


def independent_scan(u, delta, A, B, C, D, bias):
    """Independent sample/group/channel indexed unroll, no port helper calls."""
    batch, groups, state_size, steps = B.shape
    per_group = u.shape[1] // groups
    samples = []
    for sample in range(batch):
        channels = []
        for group in range(groups):
            for channel in range(per_group):
                flat = group * per_group + channel
                state = u.new_zeros(state_size)
                values = []
                for step in range(steps):
                    dt = F.softplus(delta[sample, flat, step] + bias[flat])
                    state = (torch.exp(dt * A[flat]) * state
                             + dt * B[sample, group, :, step] * u[sample, flat, step])
                    values.append((state * C[sample, group, :, step]).sum()
                                  + D[flat] * u[sample, flat, step])
                channels.append(torch.stack(values))
        samples.append(torch.stack(channels))
    return torch.stack(samples)


def independent_global(layer, x):
    batch, steps, dim = x.shape
    x = F.group_norm(x.transpose(1, 2), layer.in_norm.num_groups,
                     layer.in_norm.weight, layer.in_norm.bias, layer.in_norm.eps).transpose(1, 2)
    q, k, v = F.linear(x, layer.qkv.weight, layer.qkv.bias).chunk(3, -1)
    q, k, v = [value.reshape(batch, steps, layer.num_heads, layer.head_dim).transpose(1, 2)
               for value in (q, k, v)]
    probability = (torch.einsum("bhtd,bhsd->bhts", q, k) / layer.head_dim ** 0.5).softmax(-1)
    y = torch.einsum("bhts,bhsd->bhtd", probability, v)
    y = y.transpose(1, 2).reshape(batch, steps, dim)
    return F.linear(y, layer.proj.weight, layer.proj.bias)


def independent_local(layer, x):
    x = F.group_norm(x.transpose(1, 2), layer.in_norm.num_groups,
                     layer.in_norm.weight, layer.in_norm.bias, layer.in_norm.eps)
    gate = F.conv1d(F.max_pool1d(x, 3, stride=1, padding=1),
                    layer.conv1.weight, layer.conv1.bias).sigmoid()
    return F.conv1d(x * gate, layer.conv2.weight, layer.conv2.bias).transpose(1, 2)


def independent_route(y, delta, bias, ratio, global_fn, local_fn):
    """Reference ordered gather and exact one-to-one original-position return."""
    batch, channels, steps = y.shape
    sorted_indices = F.softplus(delta + bias[None, :, None]).argsort(1, descending=True)
    n_global = int(channels * ratio)
    g_index = sorted_indices[:, :n_global].transpose(1, 2)
    l_index = sorted_indices[:, n_global:].transpose(1, 2)
    rows = torch.arange(batch, device=y.device)[:, None, None]
    times = torch.arange(steps, device=y.device)[None, :, None]
    global_y = global_fn(y[rows, g_index, times])
    local_y = local_fn(y[rows, l_index, times])
    # Invert the complete permutation rather than calling source gather/scatter.
    ordered = torch.cat((global_y, local_y), dim=-1)
    inverse = sorted_indices.transpose(1, 2).argsort(dim=-1)
    return ordered.gather(-1, inverse).transpose(1, 2)


def independent_mixer(core, tokens):
    batch, steps, _ = tokens.shape
    x, z = F.linear(tokens, core.in_proj.weight, core.in_proj.bias).chunk(2, dim=-1)
    z = F.silu(z)
    x = F.silu(F.conv1d(x.transpose(1, 2), core.conv1d.weight, core.conv1d.bias,
                       padding=1, groups=core.d_inner))
    inputs = torch.stack((x, x.flip(-1)), dim=1)
    coefficients = torch.stack([F.linear(inputs[:, group].transpose(1, 2),
                                         core.x_proj_weight[group]).transpose(1, 2)
                                for group in range(2)], dim=1)
    dt_rank, B, C = coefficients.split((core.dt_rank, core.d_state, core.d_state), dim=2)
    delta = torch.stack([F.linear(dt_rank[:, group].transpose(1, 2),
                                  core.dt_projs_weight[group]).transpose(1, 2)
                         for group in range(2)], dim=1).reshape(batch, -1, steps)
    inputs = inputs.reshape(batch, -1, steps)
    bias = core.dt_projs_bias.reshape(-1)
    y = independent_scan(inputs, delta, -core.A_logs.exp(), B, C, core.Ds, bias)
    y = independent_route(y, delta, bias, core.global_ratio,
                          lambda value: independent_global(core.global_module, value),
                          lambda value: independent_local(core.local_module, value))
    forward, reverse = y.reshape(batch, 2, core.d_inner, steps).unbind(dim=1)
    y = (forward + reverse.flip(-1)).transpose(1, 2)
    y = F.layer_norm(y, (core.d_inner,), core.out_norm.weight, core.out_norm.bias,
                     core.out_norm.eps)
    return F.linear(y * z, core.out_proj.weight, core.out_proj.bias)


def test_selective_recurrence_values_and_all_input_gradients():
    torch.manual_seed(7161)
    batch, groups, channels, state, steps = 2, 2, 3, 4, 6
    arguments = [
        torch.randn(batch, groups * channels, steps, dtype=torch.float64),
        torch.randn(batch, groups * channels, steps, dtype=torch.float64) * 0.2,
        -torch.rand(groups * channels, state, dtype=torch.float64) - 0.2,
        torch.randn(batch, groups, state, steps, dtype=torch.float64) * 0.3,
        torch.randn(batch, groups, state, steps, dtype=torch.float64) * 0.3,
        torch.randn(groups * channels, dtype=torch.float64),
        torch.randn(groups * channels, dtype=torch.float64) - 2,
    ]
    arguments = [value.requires_grad_() for value in arguments]
    other = [value.detach().clone().requires_grad_() for value in arguments]
    actual = _module.selective_scan_reference(*arguments)
    expected = independent_scan(*other)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * probe).sum(), arguments)
    expected_grad = torch.autograd.grad((expected * probe).sum(), other)
    for grad, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(grad, expected_value, rtol=1e-10, atol=1e-11)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    # Repeated calls must start from zero, not saved terminal state.
    torch.testing.assert_close(actual, _module.selective_scan_reference(*arguments), rtol=0, atol=0)


def test_per_token_descending_delta_channel_routing_not_fixed_half_or_time_pruning():
    # Four channels exchange ordering at each of three tokens, and batch 2 has
    # a different ordering. Exact descending channel order is source semantics.
    delta = torch.tensor([[[4., 1., 2.], [3., 4., 1.], [2., 3., 4.], [1., 2., 3.]],
                          [[1., 2., 3.], [2., 1., 4.], [4., 3., 1.], [3., 4., 2.]]])
    bias = torch.zeros(4)
    global_indices, local_indices = _module.get_global_local_index(delta, bias)
    assert global_indices.tolist() == [[[0, 1], [1, 2], [2, 3]],
                                        [[2, 3], [3, 2], [1, 0]]]
    assert local_indices.tolist() == [[[2, 3], [3, 0], [0, 1]],
                                       [[1, 0], [0, 1], [3, 2]]]
    assert global_indices.shape == (2, 3, 2)  # all tokens retained
    y = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3).requires_grad_()
    class Affine(nn.Module):
        def __init__(self, scale, offset):
            super().__init__()
            self.scale, self.offset = scale, offset
        def forward(self, values):
            return self.scale * values + self.offset
    global_branch, local_branch = Affine(2, 100), Affine(3, -100)
    actual = _module.route_channels(y, delta, bias, 0.5, global_branch, local_branch)
    expected = independent_route(y, delta, bias, 0.5, global_branch, local_branch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grads = torch.autograd.grad(actual.sum(), y)[0]
    expected_grads = torch.full_like(y.transpose(1, 2), 3).scatter(2, global_indices, 2).transpose(1, 2)
    torch.testing.assert_close(grads, expected_grads, rtol=0, atol=0)
    assert not torch.equal(actual[:, :2], 2 * y[:, :2] + 100)


def test_delta_bias_participates_and_ties_are_complete_partition_not_claimed_invariant():
    delta = torch.zeros(2, 16, 5)
    bias = torch.arange(16, dtype=torch.float32)
    g, l = _module.get_global_local_index(delta, bias)
    assert g[0, 0].tolist() == list(range(15, 7, -1))
    assert torch.all(_module.positive_delta(delta, bias) > 0)
    tied_g, tied_l = _module.get_global_local_index(delta, torch.zeros(16))
    partition = torch.cat((tied_g, tied_l), dim=-1).sort(dim=-1).values
    torch.testing.assert_close(partition, torch.arange(16).expand(2, 5, 16))
    # Source hard indices do not invent a surrogate gradient. Learned delta
    # still has gradients through selective recurrence, tested above/below.
    assert not tied_g.requires_grad and not tied_l.requires_grad


@pytest.mark.parametrize("component,reference", [("global", independent_global), ("local", independent_local)])
def test_distinct_branch_values_and_gradients(component, reference):
    torch.manual_seed(7162)
    layer = (_module.GlobalAttention(16, 4) if component == "global"
             else _module.LocalRefiner1D(16)).double()
    x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual, expected = layer(x), reference(layer, other)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    parameters = tuple(layer.parameters())
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * probe).sum(), (x, *parameters))
    expected_grad = torch.autograd.grad((expected * probe).sum(), (other, *parameters))
    for grad, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(grad, expected_value, rtol=2e-9, atol=1e-10)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_full_dynamic_mixer_core_values_and_parameter_gradients():
    torch.manual_seed(7163)
    core = _module.DynamicMixer1D(16, 3).double()
    x = torch.randn(1, 5, 16, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual, expected = core(x), independent_mixer(core, other)
    torch.testing.assert_close(actual, expected, rtol=2e-9, atol=1e-10)
    parameters = tuple(core.parameters())
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * probe).sum(), (x, *parameters))
    expected_grad = torch.autograd.grad((expected * probe).sum(), (other, *parameters))
    for grad, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(grad, expected_value, rtol=1e-7, atol=1e-9)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_source_v0_initialization_and_input_conditioned_positive_delta():
    torch.manual_seed(7164)
    core = _module.DynamicMixer1D(16, 4)
    torch.testing.assert_close(-core.A_logs.exp(),
                               -torch.arange(1, 5).float().expand(64, 4))
    torch.testing.assert_close(core.Ds, torch.ones(64))
    initial_dt = F.softplus(core.dt_projs_bias.detach())
    assert initial_dt.min() >= 0.001 - 1e-7 and initial_dt.max() <= 0.1 + 1e-7
    assert core.dt_projs_weight.abs().max() <= core.dt_rank ** -0.5
    def projected_delta(tokens):
        content, _ = core.in_proj(tokens).chunk(2, -1)
        content = F.silu(core.conv1d(content.transpose(1, 2)))
        scans = torch.stack((content, content.flip(-1)), dim=1)
        coefficients = torch.einsum("bkdt,knd->bknt", scans, core.x_proj_weight)
        low_rank = coefficients[:, :, :core.dt_rank]
        return torch.einsum("bkrt,kdr->bkdt", low_rank, core.dt_projs_weight).flatten(1, 2)
    first = projected_delta(torch.randn(2, 7, 16))
    second = projected_delta(torch.randn(2, 7, 16))
    assert not torch.allclose(first, second)
    assert torch.all(_module.positive_delta(first, core.dt_projs_bias.flatten()) > 0)
    assert core.k_group == 2


def test_default_t200_scratch_finite_backward_and_update():
    torch.manual_seed(7165)
    model = TransMixerVelocity(canonical()).train()
    imu = torch.randn(2, 6, 200)
    target = torch.randn(2, 3)
    output = model(imu)
    assert set(output) == {"vel"} and output["vel"].shape == (2, 3)
    loss, parts = model.loss(output, {"imu": imu, "target": target})
    torch.testing.assert_close(loss, F.mse_loss(output["vel"], target))
    assert set(parts) == {"mse"} and torch.isfinite(loss)
    before = model.velocity_head.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for block in model.blocks:
        assert block.op.global_module.qkv.weight.grad.abs().sum() > 0
        assert block.op.local_module.conv1.weight.grad.abs().sum() > 0
        assert block.op.dt_projs_weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.velocity_head.weight)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_real_z_only_supervision_and_one_shared_xyz_head():
    torch.manual_seed(7166)
    model = TransMixerVelocity(canonical(), d_model=16, n_layers=1, d_state=4)
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
    assert model.blocks[0].op.global_module.qkv.weight.grad.abs().sum() > 0
    assert model.blocks[0].op.local_module.conv2.weight.grad.abs().sum() > 0
    features = model.encode_window(imu)
    torch.testing.assert_close(model(imu)["vel"], F.linear(features, model.velocity_head.weight,
                                                         model.velocity_head.bias))
    assert isinstance(model.velocity_head, nn.Linear) and model.velocity_head.out_features == 3


def test_nonzero_dropout_eval_fix_window_batch_independence_axis_order_and_reload():
    torch.manual_seed(7167)
    model = TransMixerVelocity(canonical(), d_model=16, n_layers=1, d_state=4, dropout=0.2).eval()
    imu = torch.randn(2, 6, 200)
    with torch.no_grad():
        first = model(imu)["vel"]
        model(torch.randn_like(imu))
        torch.testing.assert_close(first, model(imu)["vel"], rtol=0, atol=0)
        separate = torch.cat([model(window[None])["vel"] for window in imu])
        torch.testing.assert_close(first, separate, rtol=1e-5, atol=1e-6)
        assert not torch.allclose(first, model(imu[:, [3, 4, 5, 0, 1, 2]])["vel"])
        restored = TransMixerVelocity(canonical(), **model.model_cfg).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(first, restored(imu)["vel"], rtol=0, atol=0)
    # Stronger explicit dropout equation check, not only testing default p=0.
    attention = _module.GlobalAttention(16, 4, dropout=0.6).double().eval()
    values = torch.randn(2, 5, 16, dtype=torch.float64)
    torch.testing.assert_close(attention(values), independent_global(attention, values))
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
        TransMixerVelocity(canonical(**changes))


@pytest.mark.parametrize("args", [
    {"pretrained": True}, {"pretrained": 0}, {"d_model": 0}, {"d_model": 17},
    {"n_layers": 0}, {"n_layers": True}, {"d_state": 0}, {"expansion": 0},
    {"num_heads": 0}, {"num_heads": 7}, {"global_ratio": 0}, {"global_ratio": 1},
    {"global_ratio": 0.3}, {"global_ratio": float("nan")},
    {"mlp_ratio": 0}, {"mlp_ratio": float("nan")},
    {"dropout": -0.1}, {"dropout": 1}, {"dropout": float("nan")},
])
def test_rejects_invalid_recipe(args):
    with pytest.raises(ValueError):
        TransMixerVelocity(canonical(), **args)


@pytest.mark.parametrize("shape", [(2, 6, 199), (2, 200, 6), (2, 5, 200), (2, 1, 6, 200)])
def test_runtime_shape_rejected(shape):
    model = TransMixerVelocity(canonical(), d_model=16, n_layers=1, d_state=4)
    with pytest.raises(ValueError, match="expected input"):
        model(torch.zeros(shape))


def test_config_provenance_unregistered_draft_plain_mse_narrow_core_identity():
    cfg = yaml.safe_load((HERE / "model_3d.yaml").read_text())
    provenance = json.loads((HERE / "provenance.json").read_text())
    model = TransMixerVelocity(InputSpec.from_dict(cfg["input"]), **cfg["args"])
    assert model.model_cfg == cfg["args"]
    assert cfg["registered"] is False and cfg["loss"] == "mse"
    assert cfg["input_channels"] == list(CHANNELS)
    assert "channels" not in cfg["input"] and set(cfg["input"]) <= set(INPUT_KEYS)
    assert cfg["args"]["pretrained"] is False
    assert provenance["implementation_status"] == "static_draft"
    for key in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled", "registered"):
        assert provenance[key] is False
    assert model.inference_contract == "window_noncausal"
    assert model.implementation_kind == "architecture_core_port"
    assert model.training_initialization == "scratch_supervised"
    assert "not_full_mixercseg" in cfg["fidelity"]
    assert "eval_dropout" in provenance["identity"]["explicit_semantic_changes"]
    model.set_loss("mse")
    with pytest.raises(ValueError, match="MSE"):
        model.set_loss("mse_sum")
    with pytest.raises(ValueError, match="MSE"):
        model.set_loss("mse", auxiliary_weight=1)
    assert "register_model" not in (HERE / "model.py").read_text()
