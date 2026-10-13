"""Conditional diagonal-Gaussian calibration in matched native task coordinates.

The caller must supply mean, target and log standard deviation in the same
frame, units and row layout. World velocity merged across overlapping windows
does not satisfy that contract with an untransformed per-window logstd.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Iterable, Optional

import numpy as np

CALIBRATION_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
_STATISTICS = ("nll", "nees", "nees_per_dim", "standardized_rmse", "sigma_mean",
               "coverage_1sigma", "coverage_2sigma", "box_coverage_1sigma",
               "box_coverage_2sigma", "coverage_ece")


def gaussian_uncertainty_metrics(
    mean: np.ndarray,
    target: np.ndarray,
    logstd: Optional[np.ndarray],
    valid: Optional[np.ndarray] = None,
    *,
    calibration_levels: Iterable[float] = CALIBRATION_LEVELS,
) -> dict:
    """NLL, normalized squared error and marginal central-interval calibration.

    Inputs are (N, D); logstd may be (N, D) or an explicit isotropic (N, 1)
    scalar head. A missing head is unavailable, never replaced by unit variance.
    NLL includes the Gaussian normalizing constant and sums over D components.
    NEES has expectation D under the stated diagonal model; NEES/D expects 1.
    Coverage averages components; box coverage requires all D components. ECE
    averages absolute empirical-minus-nominal central-interval coverages, and
    is not classification ECE. No clipping or calibration fit is performed.
    """
    mean, target = np.asarray(mean, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if mean.ndim != 2 or target.shape != mean.shape or mean.shape[1] < 1:
        raise ValueError("mean and target must have matching (N, D) shapes")
    valid = np.ones(len(mean), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != (len(mean),):
        raise ValueError("valid must have shape (N,)")
    levels = tuple(float(value) for value in calibration_levels)
    if not levels or any(not np.isfinite(value) or not 0 < value < 1 for value in levels):
        raise ValueError("calibration_levels must be nonempty finite probabilities in (0, 1)")
    if len(set(levels)) != len(levels):
        raise ValueError("calibration_levels must not contain duplicate probabilities")
    out = {f"uncertainty_task_{key}": float("nan") for key in _STATISTICS}
    out.update(uncertainty_task_available=0, uncertainty_task_num_rows=0,
               uncertainty_task_dims=int(mean.shape[1]), uncertainty_task_isotropic=0)
    if logstd is None:
        return out
    logstd = np.asarray(logstd, dtype=np.float64)
    if logstd.shape not in (mean.shape, (len(mean), 1)):
        raise ValueError("logstd must have shape (N, D) or isotropic (N, 1)")
    out["uncertainty_task_isotropic"] = int(logstd.shape[1] == 1)
    out["uncertainty_task_available"] = 1
    out["uncertainty_task_num_rows"] = int(valid.sum())
    if not valid.any():
        return out
    mean, target, logstd = mean[valid], target[valid], logstd[valid]
    if not all(np.isfinite(values).all() for values in (mean, target, logstd)):
        raise ValueError("Gaussian parameters and target must be finite on valid rows")
    logstd = np.broadcast_to(logstd, mean.shape)
    try:
        with np.errstate(over="raise", under="ignore", divide="raise", invalid="raise"):
            sigma = np.exp(logstd)
            if not np.isfinite(sigma).all() or (sigma <= 0).any():
                raise ValueError("logstd must define finite positive standard deviations")
            standardized = (mean - target) / sigma
            squares = standardized ** 2
            nees = np.sum(squares, axis=1)
            nll = np.sum(logstd + 0.5 * squares + 0.5 * np.log(2 * np.pi), axis=1)
    except FloatingPointError as error:
        raise ValueError("Gaussian statistics overflow; predictions are not valid scores") from error
    if not np.isfinite(nees).all() or not np.isfinite(nll).all():
        raise ValueError("Gaussian statistics must remain finite")
    absolute = np.abs(standardized)
    calibration_errors = []
    for level in levels:
        radius = NormalDist().inv_cdf((1 + level) / 2)
        calibration_errors.append(abs(float(np.mean(absolute <= radius)) - level))
    stats = {
        "nll": float(np.mean(nll)),
        "nees": float(np.mean(nees)),
        "nees_per_dim": float(np.mean(nees) / mean.shape[1]),
        "standardized_rmse": float(np.sqrt(np.mean(squares))),
        "sigma_mean": float(np.mean(sigma)),
        "coverage_1sigma": float(np.mean(absolute <= 1)),
        "coverage_2sigma": float(np.mean(absolute <= 2)),
        "box_coverage_1sigma": float(np.mean(np.all(absolute <= 1, axis=1))),
        "box_coverage_2sigma": float(np.mean(np.all(absolute <= 2, axis=1))),
        "coverage_ece": float(np.mean(calibration_errors)),
    }
    if not all(np.isfinite(value) for value in stats.values()):
        raise ValueError("Aggregated Gaussian statistics must remain finite")
    out.update({f"uncertainty_task_{key}": value for key, value in stats.items()})
    return out
