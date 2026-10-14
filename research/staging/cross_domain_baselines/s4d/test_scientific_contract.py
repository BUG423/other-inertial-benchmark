"""Server-only scientific tests for the unregistered S4D velocity port.

Run on AutoDL as required by tools/ops_v1/README.md. Convolution is checked
against independent explicit causal sums and state recurrence, including their
gradients; a random forward smoke alone would not test the SSM equations.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from inertial_benchmark.nn.base import InputSpec


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("staging_s4d", HERE / "model.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _small(dims=3, prenorm=False):
    return module.S4DVelocity(
        InputSpec(window=32, dims=dims), d_model=8, d_state=8,
        n_layers=2, dropout=0.0, prenorm=prenorm)


def _reference_parameters(kernel):
    # Deliberately do not call continuous_parameters or discrete_parameters.
    # This spells out the ZOH equation independently from the production path.
    A = -torch.exp(kernel.log_A_real) + 1j * kernel.A_imag
    dt = torch.exp(kernel.log_dt).unsqueeze(-1)
    C = kernel.C[..., 0] + 1j * kernel.C[..., 1]
    A_bar = torch.exp(dt * A)
    B_bar = (torch.exp(dt * A) - 1.0) / A
    return A_bar, B_bar, C


def _recurrence(layer, u):
    A_bar, B_bar, C = _reference_parameters(layer.kernel)
    state = torch.zeros(u.shape[0], *A_bar.shape, dtype=A_bar.dtype, device=u.device)
    outputs = []
    for step in range(u.shape[-1]):
        state = A_bar.unsqueeze(0) * state + B_bar.unsqueeze(0) * u[..., step, None]
        outputs.append(2.0 * (state * C.unsqueeze(0)).sum(dim=-1).real
                       + layer.D.unsqueeze(0) * u[..., step])
    return torch.stack(outputs, dim=-1)


def _explicit_convolution(layer, u):
    kernel = layer.kernel(u.shape[-1])
    return torch.stack([
        sum(u[..., time - lag] * kernel[:, lag] for lag in range(time + 1))
        + u[..., time] * layer.D
        for time in range(u.shape[-1])
    ], dim=-1)


def _direct_grouped_convolution(layer, u):
    length = u.shape[-1]
    # conv1d uses correlation: reversing the kernel gives causal convolution.
    kernel = layer.kernel(length).flip(-1).unsqueeze(1)
    return F.conv1d(F.pad(u, (length - 1, 0)), kernel, groups=u.shape[1]) + (
        u * layer.D.view(1, -1, 1))


def test_complex_diagonal_kernel_matches_independent_zoh_impulse_recurrence():
    torch.manual_seed(101)
    kernel = module.DiagonalS4DKernel(3, 8, dt_min=0.01, dt_max=0.08).double()
    A_bar, B_bar, C = _reference_parameters(kernel)
    state = B_bar
    expected = []
    for _ in range(17):
        expected.append(2.0 * (C * state).sum(dim=-1).real)
        state = A_bar * state
    torch.testing.assert_close(kernel(17), torch.stack(expected, dim=-1),
                               rtol=1e-11, atol=1e-12)
    actual = kernel.discrete_parameters()
    for left, right in zip(actual, (A_bar, B_bar, C)):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("reference", [_recurrence, _explicit_convolution,
                                         _direct_grouped_convolution])
def test_fft_values_and_all_input_ssm_parameter_gradients_match_reference(reference):
    torch.manual_seed(102)
    layer = module.S4DLayer(3, 8, dropout=0.0, dt_min=0.01, dt_max=0.08).double()
    u = torch.randn(2, 3, 13, dtype=torch.float64, requires_grad=True)
    weights = torch.randn_like(u)
    parameters = (u, layer.D, layer.kernel.log_dt, layer.kernel.log_A_real,
                  layer.kernel.A_imag, layer.kernel.C)
    actual = layer.linear_response(u)
    expected = reference(layer, u)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    actual_gradients = torch.autograd.grad((actual.square() * weights).sum(), parameters)
    expected_gradients = torch.autograd.grad((expected.square() * weights).sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all()
        assert actual_gradient.abs().sum() > 0
        torch.testing.assert_close(actual_gradient, expected_gradient,
                                   rtol=1e-8, atol=1e-10)


def test_fft_padding_preserves_causality_and_has_no_circular_wraparound():
    torch.manual_seed(103)
    layer = module.S4DLayer(2, 8, dropout=0.0).double()
    impulse = torch.zeros(1, 2, 19, dtype=torch.float64)
    impulse[..., -1] = 1.0
    response = layer.linear_response(impulse)
    torch.testing.assert_close(response[..., :-1], torch.zeros_like(response[..., :-1]),
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(response[0, :, -1], layer.kernel(19)[:, 0] + layer.D)
    original = torch.randn(2, 2, 19, dtype=torch.float64)
    modified = original.clone()
    modified[..., 10:] = torch.randn_like(modified[..., 10:]) * 7.0
    torch.testing.assert_close(layer.linear_response(original)[..., :10],
                               layer.linear_response(modified)[..., :10],
                               rtol=1e-10, atol=1e-11)


def test_s4d_lin_initialization_and_stable_diagonal_are_source_faithful():
    kernel = module.DiagonalS4DKernel(3, 8)
    dt, A, C = kernel.continuous_parameters()
    torch.testing.assert_close(A.real, torch.full((3, 4), -0.5))
    torch.testing.assert_close(A.imag, (torch.arange(4) * torch.pi).repeat(3, 1))
    assert C.shape == (3, 4) and C.is_complex()
    assert torch.all(dt >= 0.001 - 1e-8) and torch.all(dt <= 0.1 + 1e-8)
    assert torch.all(A.real < 0)
    A_bar, _, _ = kernel.discrete_parameters()
    assert torch.all(A_bar.abs() < 1.0)


@pytest.mark.parametrize("dims,prenorm", [(2, False), (3, False), (3, True)])
def test_base_model_contract_and_single_linear_task_head(dims, prenorm):
    model = _small(dims, prenorm)
    imu = torch.randn(2, 6, 32)
    output = model(imu)
    assert isinstance(model, module.BaseModel)
    assert output.keys() == {"vel"}
    assert output["vel"].shape == (2, dims)
    assert isinstance(model.velocity_head, torch.nn.Linear)
    assert model.velocity_head.in_features == 8
    assert model.velocity_head.out_features == dims
    assert len(model.layers) == len(model.norms) == 2
    assert all(isinstance(norm, torch.nn.LayerNorm) for norm in model.norms)
    assert model.inference_contract == "window_noncausal"
    loss, items = model.loss(output, {"imu": imu, "target": torch.randn(2, dims)})
    assert torch.isfinite(loss) and loss.ndim == 0
    assert items.keys() == {"mse"}


def test_default_width_t200_finite_backward_and_optimizer_update():
    torch.manual_seed(104)
    model = module.S4DVelocity(InputSpec(window=200, dims=3))
    imu = torch.randn(2, 6, 200)
    target = torch.tensor([[0.3, -0.2, 1.0], [-0.4, 0.8, -1.1]])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model.layers[0].kernel.C.detach().clone()
    output = model(imu)
    loss, _ = model.loss(output, {"imu": imu, "target": target})
    loss.backward()
    assert output["vel"].shape == (2, 3) and torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter).all() and torch.isfinite(parameter.grad).all()
    assert model.velocity_head.weight.grad[2].abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.layers[0].kernel.C)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_z_only_supervision_reaches_real_z_head_ssm_and_both_z_input_channels():
    torch.manual_seed(105)
    model = _small().eval()
    imu = torch.randn(2, 6, 32, requires_grad=True)
    output = model(imu)
    target = output["vel"].detach().clone()
    target[:, 2] += torch.tensor([1.0, -1.5])
    loss, _ = model.loss(output, {"imu": imu, "target": target})
    loss.backward()
    torch.testing.assert_close(model.velocity_head.weight.grad[:2],
                               torch.zeros_like(model.velocity_head.weight.grad[:2]))
    assert model.velocity_head.weight.grad[2].abs().sum() > 0
    assert model.layers[0].kernel.C.grad.abs().sum() > 0
    assert imu.grad[:, 2].abs().sum() > 0
    assert imu.grad[:, 5].abs().sum() > 0
    assert torch.isfinite(imu.grad).all()


def test_eval_is_deterministic_state_reset_and_state_dict_reload_equivalent():
    torch.manual_seed(106)
    model = _small().eval()
    imu = torch.randn(2, 6, 32)
    expected = model(imu)["vel"]
    _ = model(torch.randn(2, 6, 32))
    torch.testing.assert_close(model(imu)["vel"], expected, rtol=0, atol=0)
    restored = _small().eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(imu)["vel"], expected, rtol=0, atol=0)


def test_yaml_drafts_scratch_contract_and_no_active_registration():
    from inertial_benchmark.nn.registry import list_models

    assert "s4d_lin_zoh_velocity" not in list_models()
    for filename, dims in (("model_2d.yaml", 2), ("model_3d.yaml", 3)):
        draft = yaml.safe_load((HERE / filename).read_text())
        assert draft["registered"] is False
        assert draft["training"] == "supervised_from_scratch"
        assert draft["args"]["pretrained"] is False
        assert draft["args"]["d_model"] == 128 and draft["args"]["d_state"] == 64
        assert draft["input"]["dims"] == dims
        assert draft["input"]["window"] == draft["input"]["rate"] == 200
        assert draft["input"]["target"] == "avg_velocity" and draft["loss"] == "mse"


@pytest.mark.parametrize("changes", [
    {"channels": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")},
    {"target": "displacement"}, {"frame": "body"},
    {"target": "frame_velocity"}, {"history": 2, "history_stride": 4},
    {"extra_inputs": ("init_velocity",)}, {"rate": 100},
    {"orientation": "device"}, {"remove_gravity": True}, {"output_steps": 1},
])
def test_rejects_incompatible_input_contract(changes):
    with pytest.raises(ValueError):
        module.S4DVelocity(InputSpec(window=32, dims=3, **changes))


@pytest.mark.parametrize("changes", [
    {"d_state": 7}, {"d_state": 0}, {"n_layers": 0}, {"dropout": 1.0},
    {"dropout": float("nan")}, {"dt_min": 0.0}, {"dt_min": 0.2, "dt_max": 0.1},
    {"dt_max": float("inf")}, {"pretrained": True},
])
def test_rejects_invalid_architecture_settings(changes):
    with pytest.raises(ValueError):
        module.S4DVelocity(InputSpec(window=32, dims=3), **changes)


def test_rejects_invalid_runtime_shape_and_kernel_length():
    model = _small()
    with pytest.raises(ValueError, match="expected input"):
        model(torch.randn(2, 6, 31))
    for length in (0, -1, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            model.layers[0].kernel(length)
