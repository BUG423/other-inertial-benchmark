"""Position statistics and relative errors over contiguous valid reference spans.

These additive metrics do not change the historical RoNIN RTE/PLR protocol.
Positions alone cannot establish a full SE(3) relative pose error: translation
below compares world-frame displacement vectors, and rotation remains unknown.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

NAN = float("nan")


def _prepare(pred, gt, valid, dims):
    if isinstance(dims, bool) or not isinstance(dims, (int, np.integer)) or dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[0] != gt.shape[0]:
        raise ValueError("positions must have matching (N, D) shapes")
    if pred.shape[1] < dims or gt.shape[1] < dims:
        raise ValueError(f"requested {dims} axes must be present in both positions")
    pred, gt = pred[:, :dims], gt[:, :dims]
    valid = np.ones(len(gt), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != (len(gt),):
        raise ValueError("valid must have shape (N,)")
    # Excluding a failed model prediction would bias its measured accuracy.
    if not np.isfinite(pred[valid]).all() or not np.isfinite(gt[valid]).all():
        raise ValueError("positions must be finite at every valid reference sample")
    return pred, gt, valid


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def contiguous_spans(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open spans [start, stop) without bridging invalid samples."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 1:
        raise ValueError("valid must be one-dimensional")
    bounds = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    return [(int(start), int(stop)) for start, stop in bounds.reshape(-1, 2)]


def _statistics(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    if not len(errors):
        return dict.fromkeys(("mae", "rmse", "p50", "p90", "p95", "max", "std"), NAN)
    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "p50": float(np.percentile(errors, 50)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
        "std": float(np.std(errors)),
    }


def _length(pos, spans, step):
    length = 0.0
    for start, stop in spans:
        if stop - start < 2:
            continue
        indices = np.arange(start, stop, step)
        if indices[-1] != stop - 1:
            indices = np.r_[indices, stop - 1]
        length += float(np.linalg.norm(np.diff(pos[indices], axis=0), axis=1).sum())
    return length


def relative_translation_statistics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: Optional[np.ndarray] = None,
    *,
    rate: float = 200.0,
    delta: float = 1.0,
    dims: int = 3,
) -> dict:
    """World-displacement relative error; all samples in each pair must be valid.

    Returns vector error statistics and the RMSE of displacement-length errors
    (the orientation-independent point-distance relation). No short-span scaling
    is performed. Missing full predicted poses prevent SE(3) / rotation claims.
    """
    pred, gt, valid = _prepare(pred, gt, valid, dims)
    rate, delta = _positive(rate, "rate"), _positive(delta, "delta")
    distance = int(round(rate * delta))
    if distance < 1:
        raise ValueError("delta must span at least one sample at this rate")
    vectors, length_errors = [], []
    for start, stop in contiguous_spans(valid):
        if stop - start <= distance:
            continue
        predicted = pred[start + distance:stop] - pred[start:stop - distance]
        reference = gt[start + distance:stop] - gt[start:stop - distance]
        vectors.append(np.linalg.norm(predicted - reference, axis=1))
        length_errors.append(np.abs(np.linalg.norm(predicted, axis=1)
                                    - np.linalg.norm(reference, axis=1)))
    errors = np.concatenate(vectors) if vectors else np.empty(0)
    point_errors = np.concatenate(length_errors) if length_errors else np.empty(0)
    return {**_statistics(errors), "pairs": int(len(errors)),
            "point_distance_rmse": _statistics(point_errors)["rmse"]}


def community_trajectory_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: Optional[np.ndarray] = None,
    *,
    rate: float = 200.0,
    dims: int = 3,
    intervals: Iterable[float] = (1.0, 10.0, 60.0),
) -> dict:
    """Additive unaligned statistics in the requested real XYZ or XY axes.

    MAE here means the mean Euclidean position error; component MAE is named
    separately. Paths are sampled independently within each contiguous span.
    Normalized drift sums endpoint *displacement* errors per span and divides
    by their summed 1 s reference path, cancelling each span's starting offset.
    """
    pred, gt, valid = _prepare(pred, gt, valid, dims)
    rate = _positive(rate, "rate")
    spans = contiguous_spans(valid)
    errors = np.linalg.norm(pred[valid] - gt[valid], axis=1)
    out = {f"position_{key}": value for key, value in _statistics(errors).items()}
    indices = np.flatnonzero(valid)
    out["endpoint_error"] = float(errors[-1]) if len(indices) else NAN
    for axis, name in enumerate("xyz"[:dims]):
        component = pred[valid, axis] - gt[valid, axis]
        out[f"position_{name}_mae"] = float(np.mean(np.abs(component))) if len(component) else NAN
        out[f"position_{name}_rmse"] = float(np.sqrt(np.mean(component ** 2))) if len(component) else NAN
    out["valid_position_samples"] = int(valid.sum())
    out["valid_segments"] = len(spans)
    out["valid_duration_s"] = float(sum(stop - start - 1 for start, stop in spans) / rate)
    step = max(1, int(round(rate)))
    ref_length, pred_length = _length(gt, spans, step), _length(pred, spans, step)
    ref_dense, pred_dense = _length(gt, spans, 1), _length(pred, spans, 1)
    out.update(reference_length_contiguous_m=ref_length, predicted_length_contiguous_m=pred_length,
               reference_length_contiguous_dense_m=ref_dense, predicted_length_contiguous_dense_m=pred_dense,
               plr_contiguous=pred_length / ref_length if ref_length > 0 else NAN,
               plr_contiguous_dense=pred_dense / ref_dense if ref_dense > 0 else NAN)
    endpoint_sum = sum(float(np.linalg.norm((pred[stop - 1] - pred[start])
                                            - (gt[stop - 1] - gt[start])))
                       for start, stop in spans if stop - start >= 2)
    out["segment_endpoint_error_sum"] = endpoint_sum if spans else NAN
    out["normalized_drift"] = 100.0 * endpoint_sum / ref_length if ref_length >= 1.0 else NAN
    # Avoid implying an orientation estimate from velocity headings or reference poses.
    out.update(rpe_rotation_rmse=NAN, rpe_rotation_available=0)
    for interval in dict.fromkeys(float(value) for value in intervals):
        detail = relative_translation_statistics(pred, gt, valid, rate=rate, delta=interval, dims=dims)
        prefix = f"rpe_translation_{interval:g}s"
        out.update({f"{prefix}_{key}": value for key, value in detail.items()
                    if key != "point_distance_rmse"})
        out[f"rpe_point_distance_{interval:g}s_rmse"] = detail["point_distance_rmse"]
    return out
