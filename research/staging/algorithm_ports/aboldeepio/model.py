"""Unregistered AbolDeepIO scalar reconstruction from the first-author thesis.

This is not a verified journal-version reproduction or an XYZ odometry model.
Input is body acceleration/gyroscope plus TRUE elapsed nanoseconds. Its two
outputs are prefix path length and an R4 quaternion-coordinate chord sum. Neither
output contains direction. See source-card.md for indexing/dropout/head assumptions.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn


def delta_t_nanoseconds(timestamps_ns: torch.Tensor) -> torch.Tensor:
    """Adjacent true clock differences, before any model-dtype conversion.

    Int64 timestamp subtraction retains small jitter on large epoch clocks.
    Float64 is allowed only when the caller already has an appropriate clock;
    precision lost before this function cannot be recovered. Float32 clocks are
    rejected. Original acquisition-clock provenance is a CALLER admission
    requirement: numeric checks cannot distinguish a real uniform clock from
    artificial nominal-rate fill. No conversion to seconds is made.
    """
    if (not isinstance(timestamps_ns, torch.Tensor) or timestamps_ns.ndim != 2
            or timestamps_ns.shape[1] < 2
            or timestamps_ns.dtype not in (torch.int64, torch.float64)):
        raise ValueError("timestamps_ns must be a batched int64/float64 nanosecond clock")
    if not torch.isfinite(timestamps_ns).all():
        raise ValueError("timestamp clock must be finite")
    # Test the original order BEFORE subtraction: int64 [max,min] can wrap to
    # a positive difference even though the actual timestamp clock decreases.
    if not (timestamps_ns[:, 1:] > timestamps_ns[:, :-1]).all():
        raise ValueError("original timestamp clock must increase strictly")
    delta = timestamps_ns[:, 1:] - timestamps_ns[:, :-1]
    if not (torch.isfinite(delta).all() and (delta > 0).all()):
        raise ValueError("timestamp differences must be finite and positive without overflow")
    return delta


def prepare_original_window(
    accelerometer: torch.Tensor,
    gyroscope: torch.Tensor,
    timestamps_ns: torch.Tensor,
    *,
    start: int,
    window_n: int,
) -> dict:
    """Freeze one honest N-versus-N+1 convention (unverified against journal VoR).

    N denotes N supervised increments at readings ``start .. start+N-1``.
    Observe N+1 IMU readings ``start .. start+N``: one future reading for the
    final target. A previous timestamp at ``start-1`` provides the true backward
    dt for the first input; it is clock context, not an extra IMU input. Feature
    row i+1 predicts target row i. Label creation is a separate training-only API
    using N+1 reference poses ``start-1 .. start+N-1``; no future pose is required.
    The caller separately proves this is the original acquisition clock, not a
    nominal grid; a well-formed tensor does not provide that provenance proof.
    """
    if (isinstance(start, bool) or not isinstance(start, int) or start < 1
            or isinstance(window_n, bool) or not isinstance(window_n, int) or window_n < 1):
        raise ValueError("window needs a positive N and a real previous timestamp (start >= 1)")
    if (accelerometer.ndim != 3 or accelerometer.shape[-1] != 3
            or gyroscope.shape != accelerometer.shape
            or timestamps_ns.shape != accelerometer.shape[:2]):
        raise ValueError("expected matched (B,T,3) acc/gyro and (B,T) timestamps")
    if start + window_n >= accelerometer.shape[1]:
        raise ValueError("window has no genuine one-sample lookahead")
    if not (accelerometer.device == gyroscope.device == timestamps_ns.device):
        raise ValueError("input streams must share their device")
    # Check the whole selected clock including previous context; no pose access.
    clock = timestamps_ns[:, start - 1:start + window_n + 1]
    dt = delta_t_nanoseconds(clock).unsqueeze(-1)
    target_clock = timestamps_ns[:, start:start + window_n]
    observed_clock = timestamps_ns[:, start:start + window_n + 1]
    return {"acc": accelerometer[:, start:start + window_n + 1],
            "gyro": gyroscope[:, start:start + window_n + 1], "delta_t_ns": dt,
            "target_timestamp_ns": target_clock, "observation_timestamp_ns": observed_clock,
            "lookahead_ns": observed_clock[:, 1:] - target_clock}


def scalar_prefix_targets(
    positions: torch.Tensor,
    quaternions_wxyz: torch.Tensor,
    *,
    hemisphere: str = "raw",
) -> dict:
    """Training labels, equations 6.6/6.7, with a real preceding reference pose.

    Input ``(B,N+1,3/4)`` holds the preceding anchor and N supervised poses.
    Prefix sums of step lengths/chords produce ``(B,N)`` targets. ``raw`` retains
    the published coordinate-difference expression, including sign-flip artifacts.
    ``continuous`` is an explicitly different hemisphere-continuity adaptation;
    no automatic normalization, SO(3) conversion or heading substitution occurs.
    """
    if (positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[1] < 2
            or quaternions_wxyz.shape != (*positions.shape[:2], 4)):
        raise ValueError("labels need matching (B,N+1,3) positions and (B,N+1,4) quaternions")
    if not (positions.is_floating_point() and quaternions_wxyz.is_floating_point()):
        raise ValueError("reference pose arrays must be floating point")
    if positions.device != quaternions_wxyz.device:
        raise ValueError("reference pose arrays must share a device")
    if not (torch.isfinite(positions).all() and torch.isfinite(quaternions_wxyz).all()):
        raise ValueError("reference scalar labels need finite valid poses")
    q = quaternions_wxyz
    if not (torch.abs(torch.linalg.vector_norm(q, dim=-1) - 1) < 1e-3).all():
        raise ValueError("quaternion labels must already be unit wxyz; no silent normalization")
    if hemisphere == "continuous":
        dot = (q[:, 1:] * q[:, :-1]).sum(-1)
        signs = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot)).cumprod(1)
        q = torch.cat((q[:, :1], q[:, 1:] * signs.unsqueeze(-1)), dim=1)
    elif hemisphere != "raw":
        raise ValueError("hemisphere must be explicitly raw or continuous")
    length = torch.linalg.vector_norm(positions[:, 1:] - positions[:, :-1], dim=-1).cumsum(1)
    chord = torch.linalg.vector_norm(q[:, 1:] - q[:, :-1], dim=-1).cumsum(1)
    return {"path_length": length, "orientation_chord": chord}


def weighted_l1_sum(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    lambda_length: float = 1.0,
    lambda_orientation: float = 1.0,
    supervision: str = "many_to_many",
) -> torch.Tensor:
    """Weighted absolute-error SUM (Eq6.10), never MSE/time or batch mean.

    Numeric lambda values and batch reduction are unreported: 1/1 and SUM over
    batch as well as supervised time are declared staging defaults. Many-to-one
    uses only the final supervised row, not the final future observation node.
    """
    weights = (lambda_length, lambda_orientation)
    if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
        raise ValueError("loss weights must be finite, nonnegative and not both zero")
    if supervision not in ("many_to_many", "many_to_one"):
        raise ValueError("unknown scalar supervision variant")
    total = None
    reference_shape = None
    for name, weight in zip(("path_length", "orientation_chord"), weights):
        predicted, target = outputs[name], targets[name]
        if (predicted.ndim != 2 or not predicted.numel() or target.shape != predicted.shape
                or predicted.device != target.device or not predicted.is_floating_point()
                or not target.is_floating_point()):
            raise ValueError(
                "scalar outputs/labels must be matching nonempty floating (B,N) arrays")
        if reference_shape is not None and predicted.shape != reference_shape:
            raise ValueError("the two scalar tasks must have the same B,N layout")
        reference_shape = predicted.shape
        if not (torch.isfinite(predicted).all() and torch.isfinite(target).all()):
            raise ValueError("scalar loss inputs must be finite")
        error = predicted - target
        if supervision == "many_to_one":
            error = error[:, -1:]
        term = weight * error.abs().sum()
        total = term if total is None else total + term
    return total


class TrainingGaussianNoise(nn.Module):
    """Printed sigma-squared is a variance; Gaussian draws use sqrt(variance)."""

    def __init__(self, variance: float) -> None:
        super().__init__()
        if not math.isfinite(variance) or variance < 0:
            raise ValueError("noise variance must be finite and nonnegative")
        self.variance = float(variance)
        self.stddev = math.sqrt(variance)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training or self.variance == 0:
            return values
        return values + torch.randn_like(values) * self.stddev


class _LSTMStack(nn.Module):
    """Independent unidirectional widths with explicitly assumed output dropout.

    Keras input/recurrent gate-mask details are not recovered. These are separate
    PyTorch LSTMs with element-wise post-layer dropout, not a claimed conversion
    of Keras recurrent dropout. Framework initialization/gates may differ too.
    """

    def __init__(self, channels: int, widths: tuple) -> None:
        super().__init__()
        self.widths = widths
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for width in widths:
            self.layers.append(nn.LSTM(channels, width, batch_first=True))
            self.dropouts.append(nn.Dropout(0.25))
            channels = width

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer, dropout in zip(self.layers, self.dropouts):
            values, _ = layer(values)
            values = dropout(values)
        return values


class AbolDeepIOOriginalScalar(nn.Module):
    """Full reported AbolDeepIO1 widths; two scalar sequences, no XYZ head/API.

    Only acc, gyro, and true dt enter inference. ``window_n`` is 10 or 20
    supervised increments with N+1 readings, under the explicit shift convention.
    Output heads are assumed linear and may produce negative predictions before
    learning. Positive-head enforcement would be a separate scientific variant.
    Raw nanoseconds are cast to the network's floating dtype but not rescaled;
    very large raw dt can saturate LSTM gates. That numerical issue is retained
    and reported rather than silently introducing a normalized-time method.
    """

    def __init__(self, window_n: int = 10) -> None:
        super().__init__()
        if (isinstance(window_n, bool) or not isinstance(window_n, int)
                or window_n not in (10, 20)):
            raise ValueError("reported scalar profiles use N=10 or N=20")
        self.window_n = int(window_n)
        self.acc_noise = TrainingGaussianNoise(0.2)
        self.gyro_noise = TrainingGaussianNoise(0.02)
        self.acc_stream = _LSTMStack(3, (100, 150, 100))
        self.gyro_stream = _LSTMStack(3, (100, 150, 100))
        self.time_stream = _LSTMStack(1, (100,))
        self.fusion = _LSTMStack(300, (150, 100, 50))
        self.head = nn.Linear(50, 2)

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor,
                delta_t_ns: torch.Tensor) -> dict:
        expected_time = self.window_n + 1
        if (acc.ndim != 3 or acc.shape[-1] != 3 or acc.shape[1] != expected_time
                or gyro.shape != acc.shape or delta_t_ns.shape != (*acc.shape[:2], 1)
                or not acc.shape[0]):
            raise ValueError(
                "expected N+1 readings: acc/gyro (B,N+1,3), true dt_ns (B,N+1,1)")
        if not (acc.is_floating_point() and gyro.dtype == acc.dtype
                and acc.dtype == self.head.weight.dtype
                and delta_t_ns.dtype in (torch.int64, torch.float64)):
            raise ValueError(
                "IMU must match model dtype; raw dt must retain int64/float64 precision")
        if not (acc.device == gyro.device == delta_t_ns.device == self.head.weight.device):
            raise ValueError("all inference streams must share the model device")
        if not (torch.isfinite(acc).all() and torch.isfinite(gyro).all()
                and torch.isfinite(delta_t_ns).all() and (delta_t_ns > 0).all()):
            raise ValueError("inference inputs must be finite with positive true dt")
        # Explicit RNG order assumption: acceleration noise, gyro noise, stream
        # dropout acc->gyro->time, fusion dropout. No time-noise/GT-derived input.
        acc, gyro = self.acc_noise(acc), self.gyro_noise(gyro)
        floating_dt = delta_t_ns.to(dtype=acc.dtype)
        if not torch.isfinite(floating_dt).all():
            raise ValueError("raw nanosecond dt overflows the model floating dtype")
        features = torch.cat((self.acc_stream(acc), self.gyro_stream(gyro),
                              self.time_stream(floating_dt)), dim=-1)
        raw = self.head(self.fusion(features))
        supervised = raw[:, 1:, :]  # one observed sample of lookahead per output
        return {"path_length": supervised[..., 0], "orientation_chord": supervised[..., 1]}
