"""Native-task Gaussian statistics checked against analytic / SciPy densities."""

import math

import numpy as np
import pytest

from inertial_benchmark.metrics.uncertainty import gaussian_uncertainty_metrics


def test_gaussian_nll_and_nees_match_multivariate_density():
    from scipy.stats import multivariate_normal

    target = np.array([[1.0, -2.0, 0.5], [-3.0, 4.0, -1.0]])
    mean = np.zeros_like(target)
    sigma = np.array([[2.0, 1.0, 0.5], [3.0, 2.0, 4.0]])
    out = gaussian_uncertainty_metrics(mean, target, np.log(sigma))
    density = [-multivariate_normal.logpdf(t, mean=m, cov=np.diag(s*s))
               for m, t, s in zip(mean, target, sigma)]
    assert out["uncertainty_task_nll"] == pytest.approx(np.mean(density))
    expected_nees = (0.25 + 4 + 1 + 1 + 4 + 0.0625) / 2
    assert out["uncertainty_task_nees"] == pytest.approx(expected_nees)
    assert out["uncertainty_task_nees_per_dim"] == pytest.approx(expected_nees / 3)


def test_component_and_box_coverages_are_distinct():
    errors = np.array([[0.0, 0.5, 3.0], [1.5, 0.0, -0.5]])
    out = gaussian_uncertainty_metrics(errors, np.zeros_like(errors), np.zeros_like(errors))
    assert out["uncertainty_task_coverage_1sigma"] == pytest.approx(4/6)
    assert out["uncertainty_task_coverage_2sigma"] == pytest.approx(5/6)
    assert out["uncertainty_task_box_coverage_1sigma"] == 0
    assert out["uncertainty_task_box_coverage_2sigma"] == 0.5


def test_central_interval_calibration_uses_declared_probability_levels():
    from scipy.stats import norm

    mean = np.array([[0.0, 1.0], [2.0, 3.0]])
    levels = (0.5, 0.95)
    out = gaussian_uncertainty_metrics(mean, np.zeros_like(mean), np.zeros_like(mean),
                                       calibration_levels=levels)
    empirical = [np.mean(np.abs(mean) <= norm.ppf((1 + level) / 2)) for level in levels]
    assert out["uncertainty_task_coverage_ece"] == pytest.approx(np.mean(np.abs(np.array(empirical)-levels)))


def test_explicit_isotropic_head_and_xyz_marginal_xy():
    mean = np.array([[0.0, 2.0, 4.0], [2.0, 0.0, 2.0]])
    target = np.zeros_like(mean)
    logstd = np.full((2, 1), np.log(2))
    isotropic = gaussian_uncertainty_metrics(mean, target, logstd)
    explicit = gaussian_uncertainty_metrics(mean, target, np.broadcast_to(logstd, mean.shape))
    assert isotropic["uncertainty_task_nll"] == explicit["uncertainty_task_nll"]
    assert isotropic["uncertainty_task_nees"] == explicit["uncertainty_task_nees"]
    assert isotropic["uncertainty_task_isotropic"] == 1
    xy = gaussian_uncertainty_metrics(mean[:, :2], target[:, :2], logstd)
    assert xy["uncertainty_task_dims"] == 2
    assert xy["uncertainty_task_nees"] == pytest.approx(1)
    assert isotropic["uncertainty_task_nees"] == pytest.approx(3.5)


def test_missing_uncertainty_and_all_invalid_are_not_unit_variance():
    mean = target = np.zeros((2, 3))
    missing = gaussian_uncertainty_metrics(mean, target, None)
    assert missing["uncertainty_task_available"] == 0
    assert math.isnan(missing["uncertainty_task_nll"])
    masked = gaussian_uncertainty_metrics(np.full_like(mean, np.nan), target,
                                         np.full_like(mean, np.nan), [False, False])
    assert masked["uncertainty_task_available"] == 1
    assert masked["uncertainty_task_num_rows"] == 0
    assert math.isnan(masked["uncertainty_task_nees"])


def test_masked_invalid_parameters_and_scalar_standard_deviation_units():
    mean = np.array([[0.0, 0.5], [np.nan, np.nan]])
    target = np.zeros_like(mean)
    logstd = np.array([[0.0, 0.0], [np.nan, np.nan]])
    out = gaussian_uncertainty_metrics(mean, target, logstd, [True, False])
    assert out["uncertainty_task_num_rows"] == 1
    assert out["uncertainty_task_nees"] == 0.25
    # Consistent metre -> centimetre conversion leaves NEES and coverage invariant,
    # while the differential-density NLL shifts by D*log(100).
    converted = gaussian_uncertainty_metrics(mean * 100, target * 100,
                                            logstd + np.log(100), [True, False])
    assert converted["uncertainty_task_nees"] == pytest.approx(out["uncertainty_task_nees"])
    assert converted["uncertainty_task_coverage_1sigma"] == out["uncertainty_task_coverage_1sigma"]
    assert converted["uncertainty_task_nll"] == pytest.approx(out["uncertainty_task_nll"] + 2*np.log(100))


@pytest.mark.parametrize("kind", ["layout", "nan", "overflow", "underflow", "levels"])
def test_invalid_gaussian_contract_is_rejected(kind):
    mean = target = np.zeros((2, 3))
    logstd = np.zeros_like(mean)
    if kind == "layout":
        logstd = np.zeros((2, 2))
    elif kind == "nan":
        logstd[0, 2] = np.nan
    elif kind == "overflow":
        logstd[0, 0] = 1000
    elif kind == "underflow":
        logstd[0, 0] = -1000
    with pytest.raises(ValueError):
        gaussian_uncertainty_metrics(mean, target, logstd,
                                     calibration_levels=() if kind == "levels" else (0.5, 0.9))
