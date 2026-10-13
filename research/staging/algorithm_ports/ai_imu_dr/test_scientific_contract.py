"""Explicit server-only scientific tests; default public testpaths exclude staging.

Synthetic formula/engineering evidence only, never KITTI accuracy or native
admission. The last two tests require the already acquired, hash-verified
official SOURCE-only oracle directory via AI_IMU_DR_OFFICIAL_SOURCE.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg
import torch

from .filter import (State, inject, nominal_propagate, observation,
                     observation_jacobian, propagate, released_linearization,
                     run_filter, update)
from .geometry import delta_t_seconds, left_jacobian, se23_exp, skew, so3_exp
from .model import AIIMUDR, PhysicalParameters, augment_released
from .relative_pose import NoEligiblePairs, build_pairs, relative_pose_sum
from .training import SupervisedRecording, released_epoch_step, released_optimizer


DTYPE = torch.float64


def tensor(value):
    return torch.as_tensor(value, dtype=DTYPE)


def np_hat(x):
    x, y, z = x
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def np_exp(x):
    return scipy.linalg.expm(np_hat(x))


def np_inject(s, dx):
    algebra = np.zeros((5, 5))
    algebra[:3, :3] = np_hat(dx[:3])
    algebra[:3, 3] = dx[3:6]
    algebra[:3, 4] = dx[6:9]
    group = scipy.linalg.expm(algebra)
    dR = group[:3, :3]
    return (dR @ s[0], dR @ s[1] + group[:3, 3], dR @ s[2] + group[:3, 4],
            s[3] + dx[9:12], s[4] + dx[12:15], np_exp(dx[15:18]) @ s[5], s[6] + dx[18:21])


def np_error(truth, estimate):
    def group(s):
        matrix = np.eye(5)
        matrix[:3, :3], matrix[:3, 3], matrix[:3, 4] = s[:3]
        return matrix
    logarithm = np.real_if_close(scipy.linalg.logm(group(truth) @ np.linalg.inv(group(estimate)))).real
    angular = np.array([logarithm[2, 1], logarithm[0, 2], logarithm[1, 0]])
    extrinsic = np.real_if_close(scipy.linalg.logm(truth[5] @ estimate[5].T)).real
    return np.concatenate((angular, logarithm[:3, 3], logarithm[:3, 4],
                           truth[3] - estimate[3], truth[4] - estimate[4],
                           [extrinsic[2, 1], extrinsic[0, 2], extrinsic[1, 0]], truth[6] - estimate[6]))


def np_nominal(s, u, dt, g):
    R, v, p, bg, ba, Rci, t = s
    acceleration = R @ (u[3:6] - ba) + g
    return (R @ np_exp((u[:3] - bg) * dt), v + acceleration * dt,
            p + v * dt + .5 * acceleration * dt ** 2, bg.copy(), ba.copy(), Rci.copy(), t.copy())


def np_observation(s, u):
    R, v, _, bg, _, Rci, t = s
    return (Rci.T @ R.T @ v + np.cross(t, u[:3] - bg))[1:]


def np_FG(s, g):
    R, v, p, *_ = s
    F, G = np.zeros((21, 21)), np.zeros((21, 18))
    F[3:6, :3], F[6:9, 3:6] = np_hat(g), np.eye(3)
    F[:3, 9:12] = -R
    F[3:6, 9:12], F[6:9, 9:12] = -np_hat(v) @ R, -np_hat(p) @ R
    F[3:6, 12:15] = -R
    G[:3, :3], G[3:6, :3], G[6:9, :3] = R, np_hat(v) @ R, np_hat(p) @ R
    G[3:6, 3:6] = R
    for row, col in ((9, 6), (12, 9), (15, 12), (18, 15)):
        G[row:row + 3, col:col + 3] = np.eye(3)
    return F, G


def np_propagate(s, P, u, dt, g, Q):
    F, G = np_FG(s, g)
    F, G = F * dt, G * dt
    Phi = np.eye(21) + F + F @ F / 2 + F @ F @ F / 6
    return np_nominal(s, u, dt, g), Phi @ (P + G @ Q @ G.T) @ Phi.T


def np_update(s, P, u, variances):
    R, v, _, _, _, Rci, t = s
    H = np.zeros((2, 21))
    H[:, 3:6] = (Rci.T @ R.T)[1:]
    H[:, 9:12] = -np_hat(t)[1:]
    H[:, 15:18] = (Rci.T @ np_hat(R.T @ v))[1:]
    H[:, 18:21] = -np_hat(u[:3] - s[3])[1:]
    N = np.diag(variances)
    K = np.linalg.solve(H @ P @ H.T + N, H @ P).T
    dx = K @ (-np_observation(s, u))
    A = np.eye(21) - K @ H
    joseph = A @ P @ A.T + K @ N @ K.T
    return np_inject(s, dx), (joseph + joseph.T) / 2


def sample_state(batch=1):
    R = np_exp(np.array([.16, -.12, .24]))
    Rci = np_exp(np.array([-.04, .08, -.06]))
    s = (R, np.array([13., 2., 1.]), np.array([7., -3., 4.]),
         np.array([.003, -.004, .006]), np.array([.02, -.01, .03]),
         Rci, np.array([.7, -.4, .25]))
    return State(*(tensor(x).unsqueeze(0).repeat((batch,) + (1,) * np.asarray(x).ndim) for x in s))


def np_state(state, index=0):
    return tuple(value[index].detach().numpy().copy() for value in state.tensors())


def sample_input(batch=1, length=9):
    clock = torch.arange(length, dtype=torch.int64).unsqueeze(0).repeat(batch, 1) * 10_000_000
    clock += 1_600_000_000_000_000_000
    # Synthetic irregular ns jitter, not a recorded acquisition clock.
    clock[:, 1::2] += 31
    u = tensor([.08, -.03, .04, .4, -.2, 9.9]).repeat(batch, length, 1)
    u[:, :, 0] += torch.arange(length, dtype=DTYPE) * .001
    return clock, u


def make_model():
    return AIIMUDR(torch.zeros(6, dtype=DTYPE), torch.ones(6, dtype=DTYPE),
                   PhysicalParameters.released_kitti())


@pytest.mark.parametrize("vector", [[0., 0., 0.], [1e-10, -2e-10, 3e-10], [.4, -.2, .7], [np.pi - 1e-8, 0., 0.]])
def test_geometry_independent_expm_and_gradients(vector):
    x = tensor(vector).requires_grad_()
    np.testing.assert_allclose(so3_exp(x).detach().numpy(), np_exp(np.array(vector)), atol=2e-12, rtol=0)
    delta = torch.cat((x, tensor([1., -2., 3., .4, .5, -.6])))
    algebra = np.zeros((5, 5)); algebra[:3, :3] = np_hat(np.array(vector))
    algebra[:3, 3:] = delta.detach().numpy()[3:].reshape(2, 3).T
    exp = scipy.linalg.expm(algebra)
    R, v, p = se23_exp(delta)
    np.testing.assert_allclose(v.detach().numpy(), exp[:3, 3], atol=3e-12, rtol=0)
    np.testing.assert_allclose(p.detach().numpy(), exp[:3, 4], atol=3e-12, rtol=0)
    (R.square().sum() + v.sum() + p.sum()).backward()
    assert torch.isfinite(x.grad).all()
    assert torch.autograd.gradcheck(so3_exp, (x.detach().requires_grad_(),), eps=1e-6, atol=2e-6, rtol=1e-4)


def test_left_group_injection_rotates_position_and_velocity():
    state = sample_state()
    dx = tensor(np.linspace(-.2, .3, 21)).unsqueeze(0)
    result = inject(state, dx)
    oracle = np_inject(np_state(state), dx[0].numpy())
    for actual, expected in zip(result.tensors(), oracle):
        np.testing.assert_allclose(actual[0].numpy(), expected, atol=2e-12, rtol=0)
    assert not torch.allclose(result.p, state.p + dx[:, 6:9])


@pytest.mark.parametrize("column", range(21))
def test_all21_measurement_jacobian_columns_independent_finite_difference(column):
    s, eps = sample_state(), 2e-6
    _, u = sample_input()
    delta = np.zeros(21); delta[column] = eps
    plus = np_observation(np_inject(np_state(s), delta), u[0, 2].numpy())
    minus = np_observation(np_inject(np_state(s), -delta), u[0, 2].numpy())
    finite_difference = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(observation_jacobian(s, u[:, 2])[0, :, column].numpy(),
                               finite_difference, atol=3e-9, rtol=2e-7)


def test_released_torch_bias_sign_is_a_counterexample_not_the_active_track():
    s = sample_state(); _, u = sample_input()
    correct = observation_jacobian(s, u[:, 2])[0, :, 9:12]
    literal_torch = skew(s.tci)[0, 1:]
    assert torch.equal(correct, -literal_torch)
    assert not torch.allclose(correct, literal_torch)


def test_released_propagation_numpy_parity_full_crosscovariance_and_psd():
    s = sample_state(2); _, u = sample_input(2)
    rng = np.random.default_rng(773)
    A = rng.normal(size=(21, 21)) * .02
    P = tensor(A @ A.T + np.eye(21) * .001).repeat(2, 1, 1)
    Q = torch.diag(tensor(np.linspace(1e-8, 1e-4, 18))).repeat(2, 1, 1)
    dt, g = tensor([.013, .008]), tensor([0., 0., -9.80655])
    result, covariance = propagate(s, P, u[:, 3], dt, g, Q)
    for b in range(2):
        expected, expected_P = np_propagate(np_state(s, b), P[b].numpy(), u[b, 3].numpy(), float(dt[b]), g.numpy(), Q[b].numpy())
        for actual, oracle in zip(result.tensors(), expected):
            np.testing.assert_allclose(actual[b].numpy(), oracle, atol=2e-12, rtol=0)
        np.testing.assert_allclose(covariance[b].numpy(), expected_P, atol=2e-12, rtol=0)
    assert torch.linalg.eigvalsh(covariance).min() > 0
    assert covariance[:, :3, 9:12].abs().sum() > 0


def test_approximate_phi_converges_to_independent_discrete21_fd_not_exact_identity():
    s = sample_state(); _, inputs = sample_input()
    source, u, g = np_state(s), inputs[0, 3].numpy(), np.array([0., 0., -9.80655])
    F, _ = np_FG(source, g)
    errors = []
    for dt in (.02, .01, .005):
        base = np_nominal(source, u, dt, g)
        fd = np.zeros((21, 21))
        for column in range(21):
            delta = np.zeros(21); delta[column] = 2e-6
            plus = np_nominal(np_inject(source, delta), u, dt, g)
            minus = np_nominal(np_inject(source, -delta), u, dt, g)
            fd[:, column] = (np_error(plus, base) - np_error(minus, base)) / 4e-6
        step = F * dt
        Phi = np.eye(21) + step + step @ step / 2 + step @ step @ step / 6
        errors.append(np.linalg.norm(fd - Phi))
    assert errors[0] > 1e-7  # honest difference from exact held-input propagation
    assert errors[1] < errors[0] * .4 and errors[2] < errors[1] * .4


def test_joseph_update_independent_numpy_and_explicit_omitted_reset():
    s = sample_state(2); _, u = sample_input(2)
    P = torch.eye(21, dtype=DTYPE).repeat(2, 1, 1) * .03
    variances = tensor([[.7, 2.3], [1.5, .9]])
    result, covariance = update(s, P, u[:, 2], variances)
    for b in range(2):
        oracle, expected = np_update(np_state(s, b), P[b].numpy(), u[b, 2].numpy(), variances[b].numpy())
        for actual, value in zip(result.tensors(), oracle):
            np.testing.assert_allclose(actual[b].numpy(), value, atol=2e-11, rtol=0)
        np.testing.assert_allclose(covariance[b].numpy(), expected, atol=2e-12, rtol=0)
    assert torch.linalg.eigvalsh(covariance).min() > 0


def test_complete_b2_filter_matches_independent_numpy_recursion():
    s = sample_state(2); clock, imu = sample_input(2, 11)
    model = make_model().eval()
    with torch.no_grad():
        result = model(clock, imu, s)
        P0, Q = model.covariance_matrices(2)
    for b in range(2):
        state, P = np_state(s, b), P0[b].numpy()
        for i in range(1, 11):
            dt = int(clock[b, i] - clock[b, i - 1]) * 1e-9
            state, P = np_propagate(state, P, imu[b, i].numpy(), dt, model.gravity.numpy(), Q[b].numpy())
            state, P = np_update(state, P, imu[b, i].numpy(), result.measurement_variances[b, i].numpy())
            for name, expected in zip(("R", "v", "p", "bg", "ba", "Rci", "tci"), state):
                np.testing.assert_allclose(getattr(result, name)[b, i].numpy(), expected, atol=2e-10, rtol=1e-11)
            np.testing.assert_allclose(result.P[b, i].numpy(), P, atol=1e-10, rtol=1e-11)
    assert torch.linalg.eigvalsh(result.P).min() >= -1e-12


def test_structure_width_6210_plus12_and_covariance_bounds_psd():
    model = make_model()
    assert sum(p.numel() for p in model.mes_net.parameters()) == 6210
    assert sum(p.numel() for p in model.initprocesscov_net.parameters()) == 12
    assert model.mes_net.cov_net[0].in_channels == 6
    assert model.mes_net.cov_net[4].dilation == (3,)
    P, Q = model.covariance_matrices(2)
    assert torch.equal(torch.diagonal(P, dim1=-2, dim2=-1)[:, [2, 5, 6, 7, 8]], torch.zeros(2, 5, dtype=DTYPE))
    assert torch.linalg.eigvalsh(P).min() == 0 and torch.linalg.eigvalsh(Q).min() > 0
    scales = model.initprocesscov_net.scales()
    assert all(bool(((value > .1) & (value < 10)).all()) for value in scales)
    variances = model.mes_net(torch.zeros(2, 6, 21, dtype=DTYPE), model.measurement_base)
    assert variances.shape == (2, 21, 2)
    assert bool((variances > model.measurement_base / 1000).all())
    assert bool((variances < model.measurement_base * 1000).all())


def test_rf17_future8_released_padding_and_no_later_dependencies():
    model = make_model().eval()
    with torch.no_grad():
        for layer in (model.mes_net.cov_net[0], model.mes_net.cov_net[4], model.mes_net.cov_lin[0]):
            layer.weight.fill_(.02); layer.bias.fill_(.2)
    x = torch.ones(1, 6, 41, dtype=DTYPE, requires_grad=True)
    y = model.mes_net(x, model.measurement_base)[0, 20, 0]
    (gradient,) = torch.autograd.grad(y, x)
    support = torch.nonzero(gradient.abs().sum((0, 1)) > 0).flatten().tolist()
    assert support == list(range(12, 29))
    assert gradient[:, :, 28].abs().sum() > 0  # eight-sample future dependence


def test_eval_deterministic_training_dropout_and_rng_replay():
    model = make_model(); x = torch.ones(2, 6, 25, dtype=DTYPE)
    model.eval()
    assert torch.equal(model.mes_net(x, model.measurement_base), model.mes_net(x, model.measurement_base))
    model.train(); rng = torch.get_rng_state()
    first = model.mes_net(x, model.measurement_base); next_draw = torch.randn(7)
    torch.set_rng_state(rng)
    assert torch.equal(first, model.mes_net(x, model.measurement_base))
    assert torch.equal(next_draw, torch.randn(7))
    assert not torch.equal(first, model.mes_net(x, model.measurement_base))


def test_clock_jitter_integer_offset_and_overflow_guards():
    clock = torch.tensor([[0, 10_000_001, 20_000_003]], dtype=torch.int64)
    torch.testing.assert_close(delta_t_seconds(clock), tensor([[.010000001, .010000002]]), rtol=0, atol=1e-17)
    assert torch.equal(delta_t_seconds(clock), delta_t_seconds(clock + 1_600_000_000_000_000_000))
    with pytest.raises(ValueError, match="strictly"):
        delta_t_seconds(torch.tensor([[torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).min]]))
    with pytest.raises(ValueError, match="overflows"):
        delta_t_seconds(torch.tensor([[torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max]]))
    with pytest.raises(ValueError, match="int64"):
        delta_t_seconds(clock.double())


def test_world_uphill_z_is_physical_not_body_vertical_padding():
    pitch = tensor([0., -.2, 0.])
    R = so3_exp(pitch).unsqueeze(0)
    velocity = (R @ tensor([20., 0., 0.])[:, None]).squeeze(-1)
    zero = torch.zeros(1, 3, dtype=DTYPE)
    s = State(R, velocity, zero, zero, zero, torch.eye(3, dtype=DTYPE).unsqueeze(0), zero)
    clock, _ = sample_input(1, 15)
    imu = torch.cat((zero, (R.mT @ (-make_model().gravity)[None, :, None]).squeeze(-1)), -1)[:, None].repeat(1, 15, 1)
    output = make_model().eval()(clock, imu, s)
    assert output.p[0, -1, 2] > .5
    assert output.v[0, -1, 2] > 3
    torch.testing.assert_close(output.p[0, -1], velocity[0] * .14, atol=2e-6, rtol=0)


def synthetic_long_reference(batch=2, length=191):
    clock, imu = sample_input(batch, length)
    t = torch.arange(length, dtype=DTYPE) * .01
    p = torch.stack((t * 1000, t * 12, t * 7), -1).unsqueeze(0).repeat(batch, 1, 1)
    R = torch.eye(3, dtype=DTYPE).repeat(batch, length, 1, 1)
    return clock, imu, R, p


def test_short_recordings_rejected_not_silent_alternative_loss():
    _, _, R, p = synthetic_long_reference(1)
    with pytest.raises(NoEligiblePairs):
        build_pairs(R[:, :30], p[:, :30])
    with pytest.raises(NoEligiblePairs):
        build_pairs(R, p * .001)


def test_original_pair_indices_gt_chord_denominator_and_sum_not_mean():
    _, _, R, p = synthetic_long_reference(1)
    pairs = build_pairs(R, p)
    assert pairs[0].path_lengths.tolist() == list(range(100, 801, 100))
    assert all(i % 10 == 0 for i in pairs[0].start.tolist() + pairs[0].end.tolist())
    assert not pairs[0].target_local.requires_grad
    loss = relative_pose_sum(R, p * 1.1, pairs)
    torch.testing.assert_close(loss, tensor(.08), atol=1e-12, rtol=0)
    # A curved/U-shaped path has path length > chord. The source denominator
    # is the reference LOCAL chord norm, not the requested distance threshold.
    curved = p.clone(); t = torch.arange(p.shape[1], dtype=DTYPE) * .03
    curved[0, :, 0] = 500 * torch.sin(t); curved[0, :, 1] = 500 * (1 - torch.cos(t))
    other = build_pairs(R, curved)[0]
    assert bool((torch.linalg.vector_norm(other.target_local, dim=-1) < other.path_lengths).any())


def test_source_augmentation_noise_std_and_rng_nonmutating():
    raw = torch.zeros(2, 30000, 6, dtype=DTYPE); original = raw.clone()
    rng = torch.get_rng_state(); noisy = augment_released(raw); next_draw = torch.rand(9)
    torch.set_rng_state(rng)
    assert torch.equal(noisy, augment_released(raw)) and torch.equal(next_draw, torch.rand(9))
    assert torch.equal(raw, original)
    torch.testing.assert_close(noisy.std(1), torch.full((2, 6), 1e-4, dtype=DTYPE), rtol=.04, atol=0)


def test_full_cnn_filter_sum_loss_backprop_p0_q_measurement_gradients_and_two_updates():
    torch.manual_seed(617)
    model = make_model(); clock, imu, reference_R, reference_p = synthetic_long_reference()
    s = sample_state(2)
    s = State(s.R, tensor([[1000., 12., 7.], [1000., 11., 6.]]), s.p, s.bg, s.ba, s.Rci, s.tci)
    pairs = build_pairs(reference_R, reference_p)
    optimizer = released_optimizer(model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    for _ in range(2):
        optimizer.zero_grad()
        result = model(clock, imu, s)
        loss = relative_pose_sum(result.R, result.p, pairs)
        assert torch.isfinite(loss) and loss > 0
        loss.backward()
        for block in (model.initprocesscov_net.factor_initial_covariance,
                      model.initprocesscov_net.factor_process_covariance,
                      model.mes_net.cov_net, model.mes_net.cov_lin):
            gradients = [p.grad for p in block.parameters()]
            assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
            assert sum(float(g.abs().sum()) for g in gradients) > 0
        assert bool((model.initprocesscov_net.factor_initial_covariance.weight.grad != 0).all())
        assert bool((model.initprocesscov_net.factor_process_covariance.weight.grad != 0).all())
        optimizer.step()
    assert all(int(value["step"]) == 2 for value in optimizer.state.values())
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    assert result.p.shape[-1] == 3 and result.p[..., 2].abs().sum() > 0


def test_numeric_state_and_rng_checkpoint_replay_without_claiming_full_nio(tmp_path):
    model = make_model(); model.train()
    clock, imu = sample_input(2, 15); s = sample_state(2)
    path = tmp_path / "science.pt"
    rng = torch.get_rng_state()
    torch.save({"state": model.state_dict(), "rng": rng, "track": "released_discrete_port"}, path)
    with torch.no_grad():
        first = model(clock, imu, s); draw = torch.randn(5)
    saved = torch.load(path, weights_only=True)
    restored = make_model(); restored.load_state_dict(saved["state"]); restored.train()
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    torch.set_rng_state(saved["rng"])
    with torch.no_grad():
        second = restored(clock, imu, s)
    for name in ("R", "v", "p", "bg", "ba", "Rci", "tci", "P", "measurement_variances"):
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert torch.equal(draw, torch.randn(5))


def test_released_epoch_step_no_pairs_is_admission_failure():
    model = make_model(); clock, imu = sample_input()
    R = torch.eye(3, dtype=DTYPE).repeat(1, 9, 1, 1); p = torch.zeros(1, 9, 3, dtype=DTYPE)
    recording = SupervisedRecording(clock, imu, sample_state(), R, p)
    with pytest.raises(NoEligiblePairs):
        released_epoch_step(model, released_optimizer(model), (recording,), augmentation=False)


def test_released_recording_gate_clip_and_two_real_optimizer_updates():
    torch.manual_seed(419)
    model = make_model(); clock, imu, R, p = synthetic_long_reference(1)
    state = sample_state()
    state = State(state.R, tensor([[1000., 12., 7.]]), state.p, state.bg,
                  state.ba, state.Rci, state.tci)
    recording = SupervisedRecording(clock, imu, state, R, p)
    optimizer = released_optimizer(model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    for _ in range(2):
        receipt = released_epoch_step(model, optimizer, (recording,), augmentation=True)
        assert receipt["optimizer_updates"] == 1, receipt
        assert receipt["accepted_recordings"] == 1 and not receipt["rejected"]
        assert receipt["loss"] > 0 and receipt["preclip_grad_norm"] <= 3
    assert all(int(value["step"]) == 2 for value in optimizer.state.values())
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())


@pytest.mark.parametrize("bad_variances", [[0., 1.], [-1., 1.], [np.nan, 1.]])
def test_measurement_covariance_invalid_rejected_without_fake_jitter(bad_variances):
    state = sample_state(); _, imu = sample_input()
    with pytest.raises(ValueError):
        update(state, torch.eye(21, dtype=DTYPE)[None], imu[:, 1], tensor(bad_variances)[None])


def test_filter_rejects_indefinite_covariance_no_psd_repair():
    state = sample_state(); clock, imu = sample_input()
    model = make_model(); P, Q = model.covariance_matrices(1)
    bad = P.detach().clone(); bad[:, 0, 0] = -.1
    with pytest.raises(ValueError, match="PSD"):
        run_filter(clock, imu, state, bad, Q, torch.ones(1, 9, 2, dtype=DTYPE), model.gravity)


def test_inference_cannot_accept_later_reference_labels():
    model = make_model(); clock, imu = sample_input()
    with pytest.raises(TypeError):
        model(clock, imu, sample_state(), reference_p=torch.zeros(1, 9, 3))


def test_torch_geometry_and_update_nonzero_leverarm_gradcheck():
    state = sample_state(); _, imu = sample_input()
    delta = torch.zeros(1, 21, dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(lambda increment: observation(inject(state, increment), imu[:, 2]),
                                    (delta,), eps=1e-6, atol=2e-6, rtol=1e-4)
    P = torch.eye(21, dtype=DTYPE)[None] * .03
    N = tensor([[.8, 1.7]]).requires_grad_()
    assert torch.autograd.gradcheck(lambda noise: update(state, P, imu[:, 2], noise)[0].p,
                                    (N,), eps=1e-6, atol=2e-6, rtol=1e-4)


def _official_class(filename, class_names, sha):
    root = os.environ.get("AI_IMU_DR_OFFICIAL_SOURCE")
    if not root:
        raise RuntimeError("operator must supply acquired official source oracle directory; no silent skip")
    path = Path(root) / "src" / filename
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == sha
    parsed = ast.parse(content)
    chosen = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in class_names]
    assert {node.name for node in chosen} == set(class_names)
    namespace = {"torch": torch, "np": np, "scipy": scipy}
    # Execute only pinned class definitions. Legacy top-level demo/import,
    # package installation, pretrained loading and data are never run.
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), "exec"), namespace)
    return [namespace[name] for name in class_names]


def test_actual_official_cnn_b1_eval_rng_and_parameter_parity():
    Init, Mes = _official_class("utils_torch_filter.py", ("InitProcessCovNet", "MesNet"),
        "05b3aa23ef0719b816a9ba13215f21a3920258a758ac907b8e3fa32d3024ffdb")
    rng = torch.get_rng_state(); official_init, official_mes = Init(), Mes()
    next_draw = torch.rand(7)
    torch.set_rng_state(rng); port = make_model()
    assert torch.equal(next_draw, torch.rand(7))
    for key, value in official_init.state_dict().items():
        assert torch.equal(value, port.initprocesscov_net.state_dict()[key])
    for key, value in official_mes.state_dict().items():
        assert torch.equal(value, port.mes_net.state_dict()[key])
    official_mes.eval(); port.eval(); _, u = sample_input(1, 41)
    old = official_mes(u.transpose(1, 2), SimpleNamespace(cov0_measurement=port.measurement_base))
    new = port.mes_net(u.transpose(1, 2), port.measurement_base)
    assert torch.equal(old, new[0])
    for oldscale, newscale in zip((official_init.init_cov(None), official_init.init_processcov(None).flatten()),
                                 port.initprocesscov_net.scales()):
        assert torch.equal(oldscale, newscale)


def test_actual_official_numpy_b1_propagation_and_negative_bias_update_parity():
    (Original,) = _official_class("utils_numpy_filter.py", ("NUMPYIEKF",),
        "ea855d25a7748a2d4c6e8134a7d1120cacb6e4225d408845effeed6eff1de001")
    source = Original(); port = make_model()
    state = sample_state(); _, u = sample_input()
    _, Q = port.covariance_matrices(1); source.Q = Q[0].detach().numpy(); source.g = port.gravity.numpy()
    P = np.eye(21) * .01
    old = source.propagate(*np_state(state), P, u[0, 3].numpy(), .013)
    newstate, newP = propagate(state, tensor(P).unsqueeze(0), u[:, 3], tensor([.013]), port.gravity, Q)
    for actual, expected in zip(newstate.tensors(), old[:7]):
        np.testing.assert_allclose(actual[0].detach().numpy(), expected, atol=3e-12, rtol=0)
    np.testing.assert_allclose(newP[0].detach().numpy(), old[7], atol=3e-12, rtol=0)
    variances = np.array([.8, 1.4])
    oldupdate = source.update(*old[:7], old[7], u[0, 3].numpy(), 3, variances)
    newstate, newP = update(newstate, newP, u[:, 3], tensor(variances).unsqueeze(0))
    for actual, expected in zip(newstate.tensors(), oldupdate[:7]):
        np.testing.assert_allclose(actual[0].detach().numpy(), expected, atol=3e-11, rtol=0)
    np.testing.assert_allclose(newP[0].detach().numpy(), oldupdate[7], atol=3e-12, rtol=0)
