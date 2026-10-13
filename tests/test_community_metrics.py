"""Analytic, gap-aware metrics without model or dataset dependencies."""

import math

import numpy as np
import pytest

from inertial_benchmark.metrics import (
    community_trajectory_metrics,
    contiguous_spans,
    fitness_keys,
    relative_translation_statistics,
    trajectory_metrics,
)


def test_xyz_position_statistics_and_axis_errors():
    gt = np.zeros((5, 3))
    pred = np.column_stack((np.zeros(5), np.zeros(5), np.arange(5)))
    out = community_trajectory_metrics(pred, gt, rate=1, dims=3)
    assert out["position_mae"] == pytest.approx(2)
    assert out["position_rmse"] == pytest.approx(math.sqrt(6))
    assert out["position_p50"] == pytest.approx(2)
    assert out["position_p90"] == pytest.approx(3.6)
    assert out["position_p95"] == pytest.approx(3.8)
    assert out["endpoint_error"] == out["position_max"] == 4
    assert out["position_z_mae"] == 2 and out["position_x_rmse"] == 0
    xy = community_trajectory_metrics(pred[:, :2], gt[:, :2], rate=1, dims=2)
    assert xy["position_mae"] == 0 and "position_z_mae" not in xy


def test_contiguous_lengths_and_relative_pairs_never_bridge_gaps():
    gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [np.nan]*3,
                   [100, 0, 10], [101, 0, 11], [102, 0, 12]], dtype=float)
    pred = 2 * gt
    pred[:3] += [10, 0, 0]
    pred[4:] += [0, 0, 40]
    valid = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)
    out = community_trajectory_metrics(pred, gt, valid, rate=1)
    length = 2 + 2 * math.sqrt(2)
    assert contiguous_spans(valid) == [(0, 3), (4, 7)]
    assert out["reference_length_contiguous_m"] == pytest.approx(length)
    assert out["predicted_length_contiguous_m"] == pytest.approx(2 * length)
    assert out["plr_contiguous"] == 2
    assert out["plr_contiguous_dense"] == 2
    assert out["segment_endpoint_error_sum"] == pytest.approx(length)
    assert out["normalized_drift"] == pytest.approx(100)
    assert out["valid_duration_s"] == 4 and out["valid_segments"] == 2
    assert out["valid_position_samples"] == 6
    assert out["rpe_translation_1s_pairs"] == 4
    assert out["rpe_translation_1s_rmse"] == pytest.approx(math.sqrt(1.5))
    long = relative_translation_statistics(pred, gt, valid, rate=1, delta=3)
    assert long["pairs"] == 0 and math.isnan(long["rmse"])


def test_partial_reference_endpoints_and_no_short_span_extrapolation():
    gt = np.column_stack((np.arange(8), np.zeros(8), np.zeros(8)))
    pred = 2 * gt
    valid = np.array([False, True, True, False, False, True, True, False])
    out = community_trajectory_metrics(pred, gt, valid, rate=1, intervals=(1, 2))
    assert out["endpoint_error"] == 6  # last valid index, not last array element
    assert out["rpe_translation_1s_pairs"] == 2
    assert out["rpe_translation_1s_rmse"] == 1
    assert out["rpe_translation_2s_pairs"] == 0
    assert math.isnan(out["rpe_translation_2s_rmse"])


def test_point_distance_relation_is_rotation_independent():
    gt = np.column_stack((np.arange(4), np.zeros(4), np.zeros(4)))
    pred = gt[:, [1, 0, 2]]
    out = community_trajectory_metrics(pred, gt, rate=1)
    assert out["rpe_translation_1s_rmse"] == pytest.approx(math.sqrt(2))
    assert out["rpe_point_distance_1s_rmse"] == 0
    assert math.isnan(out["rpe_rotation_rmse"])
    assert out["rpe_rotation_available"] == 0


def test_rpe_3d_scale_and_timestamp_interval():
    rate, duration = 20, 12
    t = np.arange(rate * duration + 1) / rate
    velocity = np.array([1.0, 2.0, 2.0])
    gt = t[:, None] * velocity
    out = community_trajectory_metrics(1.2 * gt, gt, rate=rate)
    assert out["rpe_translation_10s_rmse"] == pytest.approx(6)
    assert out["rpe_translation_10s_pairs"] == 41
    assert out["normalized_drift"] == pytest.approx(20)
    assert out["plr_contiguous"] == pytest.approx(1.2)


def test_undefined_empty_static_and_singleton_metrics():
    empty = community_trajectory_metrics(np.empty((0, 3)), np.empty((0, 3)), rate=1)
    assert empty["valid_segments"] == empty["rpe_translation_1s_pairs"] == 0
    assert math.isnan(empty["position_rmse"]) and math.isnan(empty["endpoint_error"])
    masked = community_trajectory_metrics(np.full((3, 3), np.nan), np.full((3, 3), np.nan),
                                         np.zeros(3, dtype=bool), rate=1)
    assert math.isnan(masked["position_p95"])
    still = community_trajectory_metrics(np.zeros((3, 3)), np.zeros((3, 3)), rate=1)
    assert still["position_rmse"] == still["rpe_translation_1s_rmse"] == 0
    assert math.isnan(still["normalized_drift"]) and math.isnan(still["plr_contiguous"])
    singleton = community_trajectory_metrics(np.ones((1, 3)), np.zeros((1, 3)), rate=1)
    assert singleton["endpoint_error"] == pytest.approx(math.sqrt(3))
    assert singleton["valid_duration_s"] == 0
    assert math.isnan(singleton["normalized_drift"])


@pytest.mark.parametrize("which", ["pred", "gt"])
def test_nonfinite_valid_position_is_a_failure_not_a_better_score(which):
    pred, gt = np.zeros((3, 3)), np.zeros((3, 3))
    (pred if which == "pred" else gt)[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        community_trajectory_metrics(pred, gt, rate=1)
    out = community_trajectory_metrics(pred, gt, [True, False, True], rate=1)
    assert out["position_rmse"] == 0


@pytest.mark.parametrize("kind", ["axes", "mask", "rate", "delta"])
def test_input_contract_rejects_missing_axes_and_invalid_time(kind):
    pred = gt = np.zeros((3, 3))
    if kind == "axes":
        with pytest.raises(ValueError, match="axes"):
            community_trajectory_metrics(pred[:, :2], gt, dims=3)
    elif kind == "mask":
        with pytest.raises(ValueError, match="shape"):
            community_trajectory_metrics(pred, gt, [True])
    elif kind == "rate":
        with pytest.raises(ValueError, match="rate"):
            community_trajectory_metrics(pred, gt, rate=0)
    else:
        with pytest.raises(ValueError, match="one sample"):
            relative_translation_statistics(pred, gt, rate=1, delta=0.1)


def test_additive_integration_retains_legacy_rte_and_plr_values():
    t = np.arange(81)
    gt = np.column_stack((t, np.zeros(81), np.zeros(81)))
    pred = 1.1 * gt
    valid = np.ones(81, dtype=bool)
    valid[1:70] = False
    out = trajectory_metrics(pred, gt, valid, rate=1, dims=3, oracle=gt,
                             t_rte=(tau for tau in (1, 10)))
    assert out["rte"] == pytest.approx(6)
    assert out["rte_scaled"] == 1
    assert out["valid_span_s"] == 80  # the published historical span includes the gap
    assert out["valid_duration_s"] == 10
    assert out["rpe_translation_60s_pairs"] == 0
    assert math.isnan(out["rpe_translation_60s_rmse"])
    assert out["plr_dense"] == pytest.approx(out["plr_oracle"] * out["oracle_ratio"])
    assert "endpoint_error" in fitness_keys()
    assert "rpe_rotation_rmse" not in fitness_keys()
    assert "position_std" not in fitness_keys()
    assert "rpe_translation_1s_pairs" not in fitness_keys()
