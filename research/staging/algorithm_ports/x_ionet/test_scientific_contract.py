"""Server-only engineering core tests; no neural X-IONet or benchmark acceptance."""

import importlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

# Isolated namespace for relative leaf imports; nothing enters the active registry.
package = types.ModuleType("xionet_scientific_staging")
package.__path__ = [str(Path(__file__).parent)]
sys.modules[package.__name__] = package
geo = importlib.import_module(package.__name__ + ".geometry")
flt = importlib.import_module(package.__name__ + ".filter")
cov = importlib.import_module(package.__name__ + ".covariance")


def state():
    return flt.NavigationState(Rotation.from_rotvec([0.6, -0.4, 0.3]).as_matrix(),
                               [0.1, -0.2, 0.5], [1.0, 2.0, 3.0],
                               [0.01, 0.02, -0.03], [0.1, 0.2, -0.1])


def make_filter(*, current=None, p0=None, noise=None, **kwargs):
    # All constants are SYNTHETIC engineering choices, never author defaults.
    args = dict(initial_timestamp_ns=0, gravity=[0, 0, -9.81],
                noise=noise or flt.NoiseSpectralDensity(1e-5, 1e-4, 1e-7, 1e-6),
                discretization="piecewise_constant_measurement_average",
                frame="full_body_anchor", reset="left_additive_tangent",
                max_dt_s=1.0, rotation_chart_limit=3.0)
    args.update(kwargs)
    return flt.ErrorStateFilter(current or state(), np.eye(15) if p0 is None else p0, **args)


def scipy_retract(current, delta):
    return flt.NavigationState(Rotation.from_rotvec(delta[:3]).as_matrix() @ current.rotation,
                               current.velocity + delta[3:6], current.position + delta[6:9],
                               current.gyro_bias + delta[9:12], current.accel_bias + delta[12:15])


def scipy_reference_propagation(current, gyro, acc, dt, gravity):
    r = current.rotation @ Rotation.from_rotvec((gyro - current.gyro_bias) * dt).as_matrix()
    acceleration = gravity + current.rotation @ (acc - current.accel_bias)
    v = current.velocity + acceleration * dt
    p = current.position + (current.velocity + v) * (dt / 2)
    return flt.NavigationState(r, v, p, current.gyro_bias, current.accel_bias)


def left_state_error(true, nominal):
    return np.concatenate((Rotation.from_matrix(true.rotation @ nominal.rotation.T).as_rotvec(),
                           true.velocity - nominal.velocity, true.position - nominal.position,
                           true.gyro_bias - nominal.gyro_bias, true.accel_bias - nominal.accel_bias))


def finite_difference(function, dimension, eps=1e-6):
    columns = []
    for i in range(dimension):
        d = np.zeros(dimension)
        d[i] = eps
        columns.append((function(d) - function(-d)) / (2 * eps))
    return np.stack(columns, axis=1)


@pytest.mark.parametrize("theta", [
    [0.0, 0, 0], [1e-10, -2e-10, 3e-10], [0.3, -0.2, 0.7],
    [math.pi - 1e-7, 0, 0], [0, -math.pi, 0],
    [v * (math.pi - 1e-8) / math.sqrt(14) for v in (1, 2, -3)],
])
def test_exp_log_zero_small_and_near_pi_against_independent_scipy(theta):
    r = geo.so3_exp(theta)
    expected = Rotation.from_rotvec(theta).as_matrix()
    np.testing.assert_allclose(r, expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(geo.so3_exp(geo.so3_log(r)), expected, rtol=0, atol=2e-12)
    np.testing.assert_allclose(r.T @ r, np.eye(3), rtol=0, atol=2e-14)
    assert abs(np.linalg.det(r) - 1) < 2e-14


def test_exact_pi_axis_ambiguity_uses_a_canonical_positive_largest_component():
    principal = geo.so3_log(geo.so3_exp([0, -math.pi, 0]))
    assert principal[1] > 0
    np.testing.assert_allclose(principal, [0, math.pi, 0], rtol=0, atol=1e-14)


@pytest.mark.parametrize("theta", [[0, 0, 0], [1e-5, -2e-5, 3e-5],
                                   [0.4, -0.3, 0.2], [2.7, 0.2, -0.1]])
def test_left_reset_jacobian_independent_exp_sum_exp_minus_derivative(theta):
    theta = np.asarray(theta, dtype=float)
    undo = Rotation.from_rotvec(-theta).as_matrix()

    def error(eta):
        return Rotation.from_matrix(Rotation.from_rotvec(theta + eta).as_matrix() @ undo).as_rotvec()

    expected = finite_difference(error, 3)
    np.testing.assert_allclose(geo.left_jacobian(theta), expected, rtol=0, atol=2e-9)
    if np.linalg.norm(theta) > 0.1:
        assert np.linalg.norm(geo.left_jacobian(theta) - geo.left_jacobian(-theta)) > 0.1


def test_left_small_angle_has_positive_half_hat_not_right_reset_sign():
    theta = np.array([1e-7, -2e-7, 3e-7])
    independent_skew = np.array([[0, -theta[2], theta[1]],
                                [theta[2], 0, -theta[0]], [-theta[1], theta[0], 0]])
    np.testing.assert_allclose(geo.left_jacobian(theta), np.eye(3) + 0.5 * independent_skew,
                               rtol=0, atol=3e-14)


def test_propagation_full15_jacobian_and_world_gyro_bias_coupling():
    current = state()
    gyro, acc, dt = np.array([0.3, -0.1, 0.2]), np.array([2, -0.5, 9.4]), 0.023
    gravity = np.array([0, 0, -9.81])
    nominal = scipy_reference_propagation(current, gyro, acc, dt, gravity)
    f, _ = flt.propagation_linearization(current, gyro, acc, dt)

    def propagated_error(delta):
        true = scipy_reference_propagation(scipy_retract(current, delta), gyro, acc, dt, gravity)
        return left_state_error(true, nominal)

    expected = finite_difference(propagated_error, 15)
    np.testing.assert_allclose(f, expected, rtol=0, atol=5e-10)
    assert np.linalg.norm(f[:3, 9:12] + np.eye(3) * dt) > 0.01
    actual = flt.nominal_propagation(current, gyro, acc, dt, gravity)
    np.testing.assert_allclose(left_state_error(actual, nominal), np.zeros(15), rtol=0, atol=1e-14)


def test_noise_map_matches_independent_interval_average_perturbations_and_units():
    current = state()
    gyro, acc, dt = np.array([0.3, -0.1, 0.2]), np.array([2, -0.5, 9.4]), 0.02
    gravity = np.array([0, 0, -9.81])
    nominal = scipy_reference_propagation(current, gyro, acc, dt, gravity)
    _, g = flt.propagation_linearization(current, gyro, acc, dt)

    def error(noise_increment):
        true = scipy_reference_propagation(current, gyro - noise_increment[:3] / dt,
                                          acc - noise_increment[3:6] / dt, dt, gravity)
        true.gyro_bias += noise_increment[6:9]
        true.accel_bias += noise_increment[9:12]
        return left_state_error(true, nominal)

    np.testing.assert_allclose(g, finite_difference(error, 12), rtol=0, atol=5e-10)
    p0 = np.zeros((15, 15))
    f = make_filter(current=current, p0=p0)
    report = f.propagate(gyro, acc, next_timestamp_ns=20_000_000)
    qc = np.diag(np.repeat(f.noise.values(), 3))
    np.testing.assert_allclose(report["Q"], report["G"] @ qc @ report["G"].T * dt,
                               rtol=0, atol=0)
    # Held average-acceleration noise gives .25*q_a*dt^3, not an implicit Van Loan model.
    np.testing.assert_allclose(np.diag(report["Q"])[6:9],
                               np.full(3, 0.25 * f.noise.accel_m2_per_s3 * dt ** 3),
                               rtol=1e-13, atol=1e-20)


def test_stationary_specific_force_at_nonidentity_attitude_and_known_linear_motion():
    current = state()
    current.velocity[:] = 0
    current.position[:] = 0
    zero_noise = flt.NoiseSpectralDensity(0, 0, 0, 0)
    f = make_filter(current=current, noise=zero_noise)
    specific = current.rotation.T @ (-f.gravity) + current.accel_bias
    for i in range(1, 51):
        f.propagate(current.gyro_bias, specific, next_timestamp_ns=i * 10_000_000)
    np.testing.assert_allclose(f.state.rotation, current.rotation, rtol=0, atol=1e-14)
    np.testing.assert_allclose(f.state.velocity, np.zeros(3), rtol=0, atol=1e-14)
    np.testing.assert_allclose(f.state.position, np.zeros(3), rtol=0, atol=1e-14)
    initial = flt.NavigationState(np.eye(3), [0.2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
    moving = make_filter(current=initial, noise=zero_noise)
    for i in range(1, 11):
        moving.propagate([0, 0, 0], [1, 0, 9.81], next_timestamp_ns=i * 100_000_000)
    np.testing.assert_allclose(moving.state.velocity, [1.2, 0, 0], rtol=0, atol=5e-15)
    np.testing.assert_allclose(moving.state.position, [0.7, 0, 0], rtol=0, atol=5e-15)


def test_clone_augmentation_cross_covariance_singular_psd_without_fake_jitter():
    rng = np.random.default_rng(241)
    a = rng.normal(size=(15, 15))
    p = a @ a.T
    f = make_filter(p0=p)
    report = f.clone_current()
    j = report["augmentation_jacobian"]
    np.testing.assert_array_equal(f.covariance, j @ p @ j.T)
    np.testing.assert_array_equal(f.covariance[:15, 15:], p[:, [0, 1, 2, 6, 7, 8]])
    assert f.dimension == 21 and np.linalg.matrix_rank(f.covariance, tol=1e-9) == 15
    assert np.linalg.eigvalsh(f.covariance).min() >= -1e-10
    prior = f.covariance.copy()
    r = f.propagate([0.3, 0.1, 0.2], [0.5, 1.0, 9.7], next_timestamp_ns=10_000_000)
    np.testing.assert_allclose(f.covariance, r["F"] @ prior @ r["F"].T + r["Q"],
                               rtol=0, atol=1e-13)
    np.testing.assert_array_equal(f.covariance[15:, 15:], prior[15:, 15:])


def test_clone_marginalization_retains_other_cross_correlations_and_clock_order():
    f = make_filter()
    f.clone_current()
    f.propagate([0.2, 0, 0.1], [1, 0, 9.81], next_timestamp_ns=100_000_000)
    f.clone_current()
    f.propagate([0.2, 0, 0.1], [1, 0, 9.81], next_timestamp_ns=200_000_000)
    old = f.covariance.copy()
    keep = np.r_[0:15, 21:27]
    f.marginalize_clone(0)
    np.testing.assert_array_equal(f.covariance, old[np.ix_(keep, keep)])
    assert f.dimension == 21 and f.clones[0].timestamp_ns == 100_000_000
    assert np.abs(f.covariance[:15, 15:]).sum() > 0


def test_full_body_measurement_jacobian_independent_all_current_and_clone_coordinates():
    f = make_filter()
    f.clone_current()
    f.propagate([0.3, 0.1, 0.2], [1, -2, 9.81], next_timestamp_ns=100_000_000)
    predicted, h = f.measurement_model(0)
    anchor = f.clones[0]

    def observation(delta):
        true = scipy_retract(f.state, delta[:15])
        clone_r = Rotation.from_rotvec(delta[15:18]).as_matrix() @ anchor.rotation
        clone_p = anchor.position + delta[18:21]
        return clone_r.T @ (true.position - clone_p)

    np.testing.assert_allclose(predicted, observation(np.zeros(21)), rtol=0, atol=1e-15)
    np.testing.assert_allclose(h, finite_difference(observation, 21), rtol=0, atol=5e-10)
    np.testing.assert_array_equal(h[:, :3], np.zeros((3, 3)))  # R_i, not current R_j


def test_same_convention_printed_sign_counterexample_and_consistent_update():
    current = flt.NavigationState(np.eye(3), [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
    p0 = np.zeros((15, 15))
    p0[3, 3] = 1
    f = make_filter(current=current, p0=p0, noise=flt.NoiseSpectralDensity(0, 0, 0, 0))
    f.clone_current()
    f.propagate([0, 0, 0], [0, 0, 9.81], next_timestamp_ns=1_000_000_000)
    before = f.state.position.copy()
    report = f.update([0, 0, 0], np.eye(3), anchor_timestamp_ns=0)
    np.testing.assert_allclose(report["K"][6, 0], 0.5, rtol=0, atol=1e-15)
    diagnostic = flt.paper_literal_sign_diagnostic(report["predicted"], np.zeros(3), report["K"])
    assert diagnostic["operational_filter_updated"] is False
    assert abs(before[0] + diagnostic["paper_literal_correction"][6] - 1.5) < 1e-14
    np.testing.assert_allclose(f.state.position, [0.5, 0, 0], rtol=0, atol=1e-14)
    assert np.linalg.norm(f.state.position) < np.linalg.norm(before)


def test_zero_residual_preserves_nominal_state_while_covariance_gains_information():
    f = make_filter()
    f.clone_current()
    f.propagate([0.1, 0.3, -0.2], [1, 0, 9.9], next_timestamp_ns=100_000_000)
    before = f.state_dict()
    measurement, _ = f.measurement_model(0)
    report = f.update(measurement, np.eye(3) * 0.1, anchor_timestamp_ns=0)
    np.testing.assert_array_equal(report["correction"], np.zeros(21))
    assert f.state_dict()["current"] == before["current"]
    assert f.state_dict()["clones"] == before["clones"]
    assert np.trace(f.covariance) < np.trace(np.asarray(before["covariance"]))


def test_joseph_full_cross_covariance_and_all_left_rotation_reset_blocks():
    f = make_filter()
    f.clone_current()
    f.propagate([0.1, 0.3, -0.2], [1, 0, 9.9], next_timestamp_ns=100_000_000)
    f.clone_current()
    f.propagate([0.2, -0.1, 0.3], [1, 0.4, 9.8], next_timestamp_ns=200_000_000)
    before = f.covariance.copy()
    old_rotations = [f.state.rotation.copy()] + [c.rotation.copy() for c in f.clones]
    measurement, _ = f.measurement_model(0)
    sigma = np.array([[0.1, 0.02, -0.01], [0.02, 0.2, 0.03], [-0.01, 0.03, 0.15]])
    report = f.update(measurement + [0.02, -0.03, 0.01], sigma, anchor_timestamp_ns=0)
    a = np.eye(f.dimension) - report["K"] @ report["H"]
    joseph = a @ before @ a.T + report["K"] @ sigma @ report["K"].T
    np.testing.assert_allclose(report["joseph_before_reset"], joseph, rtol=0, atol=1e-15)
    reset = np.eye(f.dimension)
    for lo in (0, 15, 21):
        theta = report["correction"][lo:lo + 3]
        undo = Rotation.from_rotvec(-theta).as_matrix()
        reset[lo:lo + 3, lo:lo + 3] = finite_difference(
            lambda eta: Rotation.from_matrix(
                Rotation.from_rotvec(theta + eta).as_matrix() @ undo).as_rotvec(), 3)
    np.testing.assert_allclose(report["reset_jacobian"], reset, rtol=0, atol=2e-10)
    np.testing.assert_allclose(f.covariance, reset @ joseph @ reset.T, rtol=0, atol=5e-10)
    np.testing.assert_allclose(f.state.rotation,
                               Rotation.from_rotvec(report["correction"][:3]).as_matrix() @ old_rotations[0],
                               rtol=0, atol=1e-14)
    for i, clone in enumerate(f.clones):
        lo = 15 + 6 * i
        np.testing.assert_allclose(clone.rotation,
                                   Rotation.from_rotvec(report["correction"][lo:lo + 3]).as_matrix()
                                   @ old_rotations[i + 1], rtol=0, atol=1e-14)
    assert np.linalg.eigvalsh(f.covariance).min() >= -1e-10


def test_serialized_complete_state_contract_replays_propagation_and_update_exactly():
    f = make_filter()
    f.clone_current()
    f.propagate([0.1, 0.3, -0.2], [1, 0, 9.9], next_timestamp_ns=100_000_000)
    saved = json.loads(json.dumps(f.state_dict(), allow_nan=False))
    restored = flt.ErrorStateFilter.from_state_dict(saved)
    assert restored.state_dict() == f.state_dict()
    for estimator in (f, restored):
        estimator.propagate([0.2, 0.1, 0.3], [1, 0.5, 9.8], next_timestamp_ns=200_000_000)
        predicted, _ = estimator.measurement_model(0)
        estimator.update(predicted + [0.01, -0.02, 0.03], np.eye(3) * 0.1, anchor_timestamp_ns=0)
    assert restored.state_dict() == f.state_dict()


def test_full_spd_off_diagonal_huber_gaussian_matches_numpy_and_has_real_xyz_gradients():
    raw = torch.tensor([[-0.2, 0.3, 0.7, -0.1, 0.2, 0.4]], dtype=torch.float64,
                       requires_grad=True)
    sigma = cov.full_spd_from_lower(raw, minimum_std_m=1e-4)["covariance"]
    assert torch.linalg.eigvalsh(sigma).min() > 0 and sigma[0, 0, 1].abs() > 0
    prediction = torch.tensor([[0.001, -0.02, 0.3]], dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(prediction)
    actual = cov.huber_gaussian_loss(prediction, target, sigma, reduction="sum")
    error = prediction.detach().numpy()[0]
    matrix = sigma.detach().numpy()[0]
    expected_h = sum(0.5 * e ** 2 if abs(e) <= 0.005 else 0.005 * abs(e) - 0.5 * 0.005 ** 2
                     for e in error)
    expected_g = 0.5 * error @ np.linalg.solve(matrix, error) + 0.5 * np.linalg.slogdet(matrix)[1]
    assert abs(actual["loss"].item() - (expected_h + 1e-4 * expected_g)) < 1e-14
    diagonal_g = (0.5 * error @ np.linalg.solve(np.diag(np.diag(matrix)), error)
                  + 0.5 * np.linalg.slogdet(np.diag(np.diag(matrix)))[1])
    assert abs(expected_g - diagonal_g) > 1e-4
    actual["loss"].backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    assert torch.isfinite(prediction.grad).all() and prediction.grad[0, 2].abs() > 0


def test_full_covariance_and_prediction_gradients_match_central_difference():
    raw = torch.tensor([[-0.2, 0.3, 0.7, -0.1, 0.2, 0.4]], dtype=torch.float64,
                       requires_grad=True)
    prediction = torch.tensor([[0.001, -0.02, 0.3]], dtype=torch.float64, requires_grad=True)

    def loss(predicted, factor_entries):
        sigma = cov.full_spd_from_lower(factor_entries, minimum_std_m=1e-4)["covariance"]
        return cov.huber_gaussian_loss(predicted, torch.zeros_like(predicted), sigma,
                                       reduction="sum")["loss"]

    assert torch.autograd.gradcheck(loss, (prediction, raw), eps=1e-6, atol=1e-8, rtol=1e-5)


def test_covariance_loss_reduction_is_named_and_not_a_hidden_phase_switch():
    predicted = torch.tensor([[0.2, 0.1, -0.3], [0.4, -0.2, 0.1]], dtype=torch.float64)
    sigma = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    summed = cov.huber_gaussian_loss(predicted, torch.zeros_like(predicted), sigma, reduction="sum")
    mean = cov.huber_gaussian_loss(predicted, torch.zeros_like(predicted), sigma,
                                     reduction="mean_samples")
    torch.testing.assert_close(summed["loss"], 2 * mean["loss"], rtol=0, atol=0)
    torch.testing.assert_close(summed["loss"], summed["huber"] + 1e-4 * summed["gaussian"],
                              rtol=0, atol=0)


@pytest.mark.parametrize("defect", ["nonsymmetric", "singular", "negative", "nan"])
def test_covariance_rejection_never_repairs_with_fake_diagonal_jitter(defect):
    sigma = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    if defect == "nonsymmetric":
        sigma[0, 0, 1] = 0.1
    elif defect == "singular":
        sigma[0, 0, 0] = 0
    elif defect == "negative":
        sigma[0, 0, 0] = -1
    else:
        sigma[0, 0, 0] = float("nan")
    before = sigma.clone()
    with pytest.raises(ValueError):
        cov.huber_gaussian_loss(torch.zeros(1, 3, dtype=torch.float64),
                                torch.zeros(1, 3, dtype=torch.float64), sigma, reduction="sum")
    torch.testing.assert_close(sigma, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("defect", ["clock_order", "clock_range", "dt_limit", "missing_clone", "sigma"])
def test_filter_failure_leaves_complete_state_unmodified(defect):
    f = make_filter()
    f.clone_current()
    f.propagate([0, 0, 0], [0, 0, 9.81], next_timestamp_ns=100_000_000)
    before = f.state_dict()
    with pytest.raises(ValueError):
        if defect == "clock_order":
            f.propagate([0, 0, 0], [0, 0, 9.81], next_timestamp_ns=100_000_000)
        elif defect == "clock_range":
            f.propagate([0, 0, 0], [0, 0, 9.81], next_timestamp_ns=(1 << 63))
        elif defect == "dt_limit":
            f.propagate([0, 0, 0], [0, 0, 9.81], next_timestamp_ns=2_000_000_000)
        elif defect == "missing_clone":
            f.update([0, 0, 0], np.eye(3), anchor_timestamp_ns=1)
        else:
            f.update([0, 0, 0], np.zeros((3, 3)), anchor_timestamp_ns=0)
    assert f.state_dict() == before


def test_signed_int64_endpoints_cannot_hide_an_unrepresentable_positive_interval():
    f = make_filter(initial_timestamp_ns=-(1 << 63))
    before = f.state_dict()
    with pytest.raises(ValueError, match="representable int64 interval"):
        f.propagate([0, 0, 0], [0, 0, 9.81], next_timestamp_ns=(1 << 63) - 1)
    assert f.state_dict() == before


@pytest.mark.parametrize("choice", ["gravity_yaw_local", "right_error", "van_loan", "negative_noise"])
def test_unspecified_frame_reset_discretization_and_noise_are_not_claimed_supported(choice):
    kw = {}
    if choice == "gravity_yaw_local":
        kw["frame"] = choice
    elif choice == "right_error":
        kw["reset"] = choice
    elif choice == "van_loan":
        kw["discretization"] = choice
    else:
        with pytest.raises(ValueError):
            flt.NoiseSpectralDensity(-1, 0, 0, 0)
        return
    with pytest.raises(ValueError):
        make_filter(**kw)
