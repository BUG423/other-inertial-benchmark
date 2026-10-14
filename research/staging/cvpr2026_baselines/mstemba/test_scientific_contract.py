"""Server-only tests; parent conftest denies collection before tail handoff.

Written, not executed. Small profiles below are diagnostic identities, NOT
capacity acceptance of source256/384/576 widths, B128, or any training stride.
No CUDA extension, pretrained weights, action/C-alignment loss or registry use.
"""

import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from inertial_benchmark.cfg import INPUT_KEYS, get_cfg
from inertial_benchmark.nn.base import BaseModel, InputSpec
from inertial_benchmark.nn.registry import MODELS


HERE = Path(__file__).parent
BEFORE_REGISTRY = dict(MODELS)
SPEC = importlib.util.spec_from_file_location("cvpr2026_staging_mstemba", HERE / "model.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
AFTER_IMPORT_REGISTRY = dict(MODELS)


def _small(spec=None, **overrides):
    args = dict(embed_dims=(8, 12, 18), state_size=2, conv_size=4, drop_path=0.0)
    args.update(overrides)
    return module.MSTembaVelocity(spec or InputSpec(dims=3), **args)


def test_draft_profile_source_widths_schema_and_no_registry_side_effect():
    assert AFTER_IMPORT_REGISTRY == BEFORE_REGISTRY
    profile = yaml.safe_load((HERE / "model_3d.yaml").read_text())
    provenance = json.loads((HERE / "provenance.json").read_text())
    assert profile["arch"] == "mstemba_architecture_core_velocity"
    assert profile["args"]["embed_dims"] == [256, 384, 576]
    defaults = inspect.signature(module.MSTembaVelocity).parameters
    assert defaults["embed_dims"].default == (256, 384, 576)
    assert defaults["state_size"].default == 16
    assert defaults["conv_size"].default == 4
    assert defaults["backend"].default == "torch_reference"
    assert set(profile["input"]) <= set(INPUT_KEYS)
    assert "channels" not in profile["input"]
    assert tuple(profile["input_channels"]) == module.CHANNELS
    cfg = get_cfg({"model": profile})
    assert cfg.window == 200 and cfg.rate == 200 and cfg.dims == 3
    assert cfg.orientation == "reference" and cfg.remove_gravity is False
    assert cfg.history == cfg.history_stride == cfg.output_steps == 0
    assert cfg.extra_inputs == [] and profile["loss"] == "mse"
    assert provenance["implementation_status"] == "static_draft"
    for key in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled", "registered"):
        assert provenance[key] is False and profile[key] is False
    assert provenance["repository"]["commit"] == module.MSTembaVelocity.source_commit
    assert profile["causality"] == "window_noncausal"


@pytest.mark.parametrize("fields", [
    {"dims": 2}, {"window": 64}, {"window": 199}, {"window": 201},
    {"rate": 100}, {"rate": float("nan")}, {"rate": float("inf")},
    {"frame": "body"}, {"frame": "gravity_yaw_local"},
    {"orientation": "device"}, {"remove_gravity": True},
    {"target": "displacement"}, {"target": "velocity_at_end"},
    {"target": "frame_velocity"}, {"output_steps": 2}, {"overlap": "center"},
    {"history": 1}, {"history": 2, "history_stride": 200}, {"history_stride": 1},
    {"extra_inputs": ("orientation",)}, {"extra_inputs": ("gravity",)},
    {"extra_inputs": ("init_velocity",)},
    {"channels": ("gyro_y", "gyro_x", "gyro_z", "acc_x", "acc_y", "acc_z")},
    {"channels": ("acc_x", "gyro_y", "gyro_z", "gyro_x", "acc_y", "acc_z")},
    {"channels": ("gyro_x",) * 6},
])
def test_reject_permission_layout_axis_rate_and_target_expansion(fields):
    with pytest.raises(ValueError):
        _small(InputSpec(**dict({"dims": 3}, **fields)))


@pytest.mark.parametrize("args", [
    {"pretrained": True}, {"backend": "cuda"}, {"backend": "mamba"},
    {"embed_dims": (8, 12)}, {"embed_dims": (8, 0, 18)},
    {"embed_dims": (8, True, 18)}, {"state_size": 0}, {"conv_size": 0},
    {"drop_path": float("nan")}, {"drop_path": float("inf")},
    {"drop_path": -0.1}, {"drop_path": 1.0},
])
def test_reject_unsupported_backend_pretraining_and_invalid_hyperparameters(args):
    with pytest.raises(ValueError):
        _small(**args)


def test_selective_scan_exact_scalar_euler_recurrence_not_zoh():
    u = torch.tensor([[[1.0, 2.0, -0.5]]], dtype=torch.float64)
    delta = torch.tensor([[[0.2, -0.3, 0.5]]], dtype=torch.float64)
    A = torch.tensor([[-1.5]], dtype=torch.float64)
    B = torch.tensor([[[0.7, 1.2, 0.4]]], dtype=torch.float64)
    C = torch.tensor([[[1.0, -0.3, 0.9]]], dtype=torch.float64)
    D = torch.tensor([0.25], dtype=torch.float64)
    bias = torch.tensor([-0.1], dtype=torch.float64)
    z = torch.tensor([[[0.8, -0.2, 1.1]]], dtype=torch.float64)
    expected, wrong_zoh = [], []
    state = zoh_state = u.new_zeros(())
    for index in range(3):
        dt = F.softplus(delta[0, 0, index] + bias[0])
        decay = (dt * A[0, 0]).exp()
        state = decay * state + dt * B[0, 0, index] * u[0, 0, index]
        zoh_state = decay * zoh_state + ((dt * A[0, 0]).expm1() / A[0, 0]) * B[0, 0, index] * u[0, 0, index]
        gate = F.silu(z[0, 0, index])
        expected.append((state * C[0, 0, index] + D[0] * u[0, 0, index]) * gate)
        wrong_zoh.append((zoh_state * C[0, 0, index] + D[0] * u[0, 0, index]) * gate)
    actual = module.selective_scan_reference(u, delta, A, B, C, D, z, bias)
    torch.testing.assert_close(actual[0, 0], torch.stack(expected))
    assert not torch.allclose(actual[0, 0], torch.stack(wrong_zoh))
    assert actual.dtype == torch.float64


def test_selective_scan_input_dependent_terms_and_gradients():
    torch.manual_seed(11)
    u = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    delta = torch.randn_like(u, requires_grad=True)
    A = (-torch.rand(3, 2, dtype=torch.float64) - 0.5).requires_grad_()
    B = torch.randn(2, 2, 5, dtype=torch.float64, requires_grad=True)
    C = torch.randn_like(B, requires_grad=True)
    D = torch.ones(3, dtype=torch.float64, requires_grad=True)
    z = torch.ones_like(u, requires_grad=True)
    bias = torch.zeros_like(D, requires_grad=True)
    output = module.selective_scan_reference(u, delta, A, B, C, D, z, bias)
    changed_B = B.detach().clone()
    changed_B[:, :, 2] += 1.0
    altered = module.selective_scan_reference(u, delta, A, changed_B, C, D, z, bias)
    torch.testing.assert_close(output[..., :2], altered[..., :2])
    assert not torch.allclose(output[..., 2:], altered[..., 2:])
    output.square().mean().backward()
    for operand in (u, delta, A, B, C, D, z, bias):
        assert operand.grad is not None and torch.isfinite(operand.grad).all()
        assert operand.grad.abs().sum() > 0
    torch.testing.assert_close(
        output.detach(), module.selective_scan_reference(u, delta, A, B, C, D, z, bias).detach())


def test_direction_has_causal_depthwise_conv_then_projected_dt_B_C():
    direction = module._SelectiveDirection(4, 2, 1, 4).double().eval()
    x, z = torch.randn(2, 4, 9, dtype=torch.float64), torch.randn(2, 4, 9, dtype=torch.float64)
    activated = F.silu(F.conv1d(F.pad(x, (3, 0)), direction.conv.weight,
                                direction.conv.bias, groups=4))
    dt, B, C = direction.x_proj(activated.transpose(1, 2)).split((1, 2, 2), dim=-1)
    expected = module.selective_scan_reference(
        activated, F.linear(dt, direction.dt_proj.weight).transpose(1, 2),
        -direction.A_log.exp(), B.transpose(1, 2), C.transpose(1, 2),
        direction.D, z, direction.dt_proj.bias)
    torch.testing.assert_close(direction(x, z), expected)
    future = x.clone()
    future[..., 5:] += 3.0
    torch.testing.assert_close(direction(x, z)[..., :5], direction(future, z)[..., :5])


def test_bimamba_v2_reverse_shared_xz_independent_cores_mean_before_output():
    mixer = module.BidirectionalSelectiveSSM(8, state_size=2, conv_size=4).double().eval()
    tokens = torch.randn(2, 7, 8, dtype=torch.float64)
    x, z = mixer.in_proj(tokens).transpose(1, 2).chunk(2, dim=1)
    expected_forward = mixer.forward_direction(x, z)
    expected_backward = mixer.backward_direction(x.flip(-1), z.flip(-1)).flip(-1)
    forward, backward = mixer.directional_outputs(tokens)
    torch.testing.assert_close(forward, expected_forward)
    torch.testing.assert_close(backward, expected_backward)
    torch.testing.assert_close(mixer(tokens), mixer.out_proj(((forward + backward) / 2).transpose(1, 2)))
    for name in ("A_log", "D"):
        assert getattr(mixer.forward_direction, name).data_ptr() != getattr(mixer.backward_direction, name).data_ptr()
    assert mixer.forward_direction.dt_proj.weight.data_ptr() != mixer.backward_direction.dt_proj.weight.data_ptr()
    assert not hasattr(mixer, "inference_cache")


@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("length", [199, 200, 201])
def test_phase_zero_pad_reinterleave_remove_preserves_order_tail_and_gradients(dilation, length):
    class Phase(nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset, self.seen = offset, None

        def forward(self, values):
            self.seen = values.detach().clone()
            return values + 1000 * self.offset

    scale = module.DilatedTembaScale(2, 2, dilation, 1, 2, 0.0)
    scale.projection = nn.Identity()
    scale.phases = nn.ModuleList([Phase(offset) for offset in range(dilation)])
    tokens = torch.arange(length * 2.0).reshape(1, length, 2).requires_grad_()
    output = scale(tokens)
    phases = (torch.arange(length) % dilation).view(1, length, 1)
    torch.testing.assert_close(output, tokens + 1000 * phases)
    assert output.shape == tokens.shape
    padding = (-length) % dilation
    for offset, phase in enumerate(scale.phases):
        assert phase.seen.shape[1] == (length + padding) // dilation
        original = tokens.detach()[:, offset::dilation]
        torch.testing.assert_close(phase.seen[:, :original.shape[1]], original)
        if phase.seen.shape[1] > original.shape[1]:
            assert phase.seen[:, -1].count_nonzero() == 0
    output[:, -1].sum().backward()
    torch.testing.assert_close(tokens.grad[:, -1], torch.ones(1, 2))
    assert tokens.grad[:, :-1].count_nonzero() == 0


def test_phase_parameters_are_independent_and_source_final_dt_bias_is_zero():
    model = _small()
    assert [scale.dilation for scale in model.scales] == [1, 2, 3]
    pointers = [phase.mixer.in_proj.weight.data_ptr() for phase in model.scales[2].phases]
    assert len(set(pointers)) == 3
    for component in model.modules():
        if isinstance(component, nn.Linear) and component.bias is not None:
            assert component.bias.count_nonzero() == 0
        if isinstance(component, module._SelectiveDirection):
            expected = torch.arange(1, 3.0).log().expand_as(component.A_log)
            torch.testing.assert_close(component.A_log, expected)
            torch.testing.assert_close(component.D, torch.ones_like(component.D))


def test_sequential_scales_project_shared_norm_sum_then_ssm_fuser():
    torch.manual_seed(4)
    model = _small().eval()
    imu = torch.randn(1, 6, 200)
    captured, incoming = [], []
    handles = [scale.register_forward_hook(lambda _, args, out: captured.append((args[0], out)))
               for scale in model.scales]
    handles.append(model.fuser.register_forward_pre_hook(lambda _, args: incoming.append(args[0])))
    try:
        output = model.encode_window(imu)
    finally:
        for handle in handles:
            handle.remove()
    assert output.shape == (1, 200, 18)
    torch.testing.assert_close(captured[1][0], captured[0][1])
    torch.testing.assert_close(captured[2][0], captured[1][1])
    expected = sum(model.fusion_norm(projection(values[1]))
                   for projection, values in zip(model.scale_projections, captured))
    torch.testing.assert_close(incoming[0], expected)


def test_XYZ_mse_backprop_includes_Z_and_both_sensor_Z_axes():
    torch.manual_seed(3)
    model = _small().eval()
    imu = torch.randn(2, 6, 200, requires_grad=True)
    output = model(imu)
    assert isinstance(model, BaseModel) and set(output) == {"vel"}
    assert output["vel"].shape == (2, 3)
    target = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    loss, items = model.loss(output, {"imu": imu, "target": target})
    torch.testing.assert_close(loss, (output["vel"] - target).square().mean())
    assert set(items) == {"mse"}
    loss.backward()
    assert model.velocity_head.weight.grad[2].abs().sum() > 0
    assert imu.grad[:, 2].abs().sum() > 0 and imu.grad[:, 5].abs().sum() > 0
    assert torch.isfinite(imu.grad).all()
    model.set_loss("mse")
    with pytest.raises(ValueError):
        model.set_loss("mse", auxiliary_weight=0.1)
    with pytest.raises(ValueError):
        model.set_loss("mse_l1")


@pytest.mark.parametrize("order", [[1, 0, 2, 3, 4, 5], [3, 1, 2, 0, 4, 5]])
def test_ordered_input_axis_and_sensor_units_can_change_final_velocity(order):
    torch.manual_seed(29)
    model = _small().eval()
    imu = torch.randn(1, 6, 200)
    with torch.no_grad():
        original = model.encode_window(imu).mean(1)
        changed = model.encode_window(imu[:, order]).mean(1)
        difference = original - changed
        selected = difference.abs().argmax().item()
        assert difference[0, selected].abs() > 1e-7
        model.velocity_head.weight.zero_()
        model.velocity_head.bias.zero_()
        model.velocity_head.weight[0, selected] = 1.0
        before, after = model(imu)["vel"], model(imu[:, order])["vel"]
    torch.testing.assert_close(before[0, 0] - after[0, 0], difference[0, selected])
    assert not torch.allclose(before, after)


def test_eval_has_no_cross_window_or_batch_state_and_no_extra_forward_input():
    model = _small().eval()
    imu = torch.randn(2, 6, 200)
    with torch.no_grad():
        first = model(imu)["vel"]
        model(torch.randn_like(imu))
        repeated = model(imu)["vel"]
        separated = torch.cat([model(imu[index:index + 1])["vel"] for index in range(2)])
    torch.testing.assert_close(first, repeated)
    torch.testing.assert_close(first, separated, rtol=1e-4, atol=1e-6)
    with pytest.raises(TypeError):
        model(imu, {"init_velocity": torch.zeros(2, 3)})
    for invalid in (torch.zeros(2, 6, 199), torch.zeros(2, 6, 201), torch.zeros(2, 1, 6, 200)):
        with pytest.raises(ValueError):
            model(invalid)
