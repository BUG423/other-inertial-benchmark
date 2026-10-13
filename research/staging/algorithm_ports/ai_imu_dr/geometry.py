"""Differentiable float64 geometry for the released AI-IMU-DR group update.

Source-derived engineering port, not an active benchmark model. Attribution
and the upstream MIT terms are in LICENSE.upstream. The tangent is SE_2(3)
left multiplication, followed by additive biases/lever arm and left Rci.
"""

from __future__ import annotations

import torch
from torch import Tensor


def skew(x: Tensor) -> Tensor:
    if x.shape[-1] != 3:
        raise ValueError("skew requires (...,3)")
    a, b, c = x.unbind(-1)
    z = torch.zeros_like(a)
    return torch.stack((z, -c, b, c, z, -a, -b, a, z), -1).reshape(*x.shape[:-1], 3, 3)


def _coefficients(theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    q = theta.square().sum(-1)
    # Clamp only the unselected closed-form branch. Both branches remain
    # differentiable at zero; no Tensor([...tensor scalars...]) detachment.
    safe = q.clamp_min(1e-8)
    angle = safe.sqrt()
    small = q < 1e-8
    a = torch.where(small, 1 - q / 6 + q.square() / 120, angle.sin() / angle)
    b = torch.where(small, .5 - q / 24 + q.square() / 720, (1 - angle.cos()) / safe)
    c = torch.where(small, 1 / 6 - q / 120 + q.square() / 5040,
                    (angle - angle.sin()) / (safe * angle))
    return a[..., None, None], b[..., None, None], c[..., None, None]


def so3_exp(theta: Tensor) -> Tensor:
    a, b, _ = _coefficients(theta)
    h = skew(theta)
    eye = torch.eye(3, dtype=theta.dtype, device=theta.device)
    return eye + a * h + b * (h @ h)


def left_jacobian(theta: Tensor) -> Tensor:
    _, b, c = _coefficients(theta)
    h = skew(theta)
    return torch.eye(3, dtype=theta.dtype, device=theta.device) + b * h + c * (h @ h)


def se23_exp(delta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return Exp([theta,xi_v,xi_p]); translation is J_left(theta) xi."""
    if delta.shape[-1] != 9:
        raise ValueError("SE_2(3) tangent requires nine components")
    rotation = so3_exp(delta[..., :3])
    jacobian = left_jacobian(delta[..., :3])
    velocity = (jacobian @ delta[..., 3:6, None]).squeeze(-1)
    position = (jacobian @ delta[..., 6:9, None]).squeeze(-1)
    return rotation, velocity, position


def delta_t_seconds(timestamps_ns: Tensor) -> Tensor:
    """Check original int64 ordering before subtraction; preserve clock jitter.

    Tensor checks cannot prove acquisition provenance. The caller must supply
    the original sensor clock, not a nominal-rate reconstruction. Absolute
    offsets are removed in integer arithmetic before conversion to seconds.
    """
    if timestamps_ns.dtype != torch.int64 or timestamps_ns.ndim != 2:
        raise ValueError("clock must be int64 nanoseconds with shape (B,T)")
    if timestamps_ns.shape[1] < 2 or not bool((timestamps_ns[:, 1:] > timestamps_ns[:, :-1]).all()):
        raise ValueError("original clock must be strictly increasing")
    difference = timestamps_ns[:, 1:] - timestamps_ns[:, :-1]
    if not bool((difference > 0).all()):
        raise ValueError("clock difference overflows signed int64")
    return difference.to(torch.float64) * 1e-9
