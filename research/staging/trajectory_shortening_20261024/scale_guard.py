"""Isolated SAGE-IO modules for a new, unregistered recipe.

This file is deliberately independent of the frozen benchmark registry.  The
modules are a proposal for a multi-scale temporal trunk and an
amplitude/direction velocity head; they are not a reproduction of any one
CVPR paper and have not passed project CPU/GPU admission.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("initial_speed must be finite and positive")
    return math.log(math.expm1(value))


class MultiScaleTemporalBlock(nn.Module):
    """Gated 1D multi-scale residual block for B,C,T IMU windows.

    The branches are depthwise temporal operators at three receptive-field
    scales.  The input-conditioned gate mixes branches after their pointwise
    projections; it never persists state between windows.
    """

    def __init__(self, in_channels: int = 6, width: int = 128,
                 kernels: tuple[int, ...] = (3, 7, 15),
                 dilations: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        if type(in_channels) is not int or in_channels < 1:
            raise ValueError("in_channels must be positive")
        if type(width) is not int or width < 4:
            raise ValueError("width must be at least four")
        if len(kernels) != len(dilations) or not kernels:
            raise ValueError("kernels and dilations must be non-empty and paired")
        if any(type(k) is not int or k < 3 or k % 2 == 0 for k in kernels):
            raise ValueError("kernels must be odd integers >= 3")
        if any(type(d) is not int or d < 1 for d in dilations):
            raise ValueError("dilations must be positive integers")
        self.in_channels, self.width = in_channels, width
        self.input_projection = nn.Conv1d(in_channels, width, kernel_size=1)
        self.branches = nn.ModuleList()
        for kernel, dilation in zip(kernels, dilations):
            self.branches.append(nn.Sequential(
                nn.Conv1d(width, width, kernel, padding=dilation * (kernel // 2),
                          dilation=dilation, groups=width, bias=False),
                nn.GroupNorm(1, width),
                nn.GELU(),
                nn.Conv1d(width, width, kernel_size=1, bias=False),
            ))
        hidden = max(4, width // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(width, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, len(self.branches), kernel_size=1),
        )
        self.output_norm = nn.GroupNorm(1, width)
        self.residual = nn.Identity()
        self.output_projection = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        if imu.ndim != 3 or imu.shape[1] != self.in_channels or imu.shape[-1] < 3:
            raise ValueError("expected B,C,T with the configured input channels")
        base = self.input_projection(imu)
        branches = torch.stack([branch(base) for branch in self.branches], dim=1)
        weights = self.gate(base).softmax(dim=1).unsqueeze(2)
        mixed = (branches * weights).sum(dim=1)
        return self.output_projection(self.output_norm(mixed + self.residual(base)))


class ScaleGuardHead(nn.Module):
    """Bounded amplitude/direction readout for a new recipe.

    ``gain`` starts at one and is bounded to ``[1-max_gain, 1+max_gain]``.
    The forward pass uses only learned features; target speed enters only in
    ``scale_guard_loss`` during training.
    """

    def __init__(self, feature_dim: int, dims: int = 3,
                 initial_speed: float = 0.5, max_gain: float = 0.5,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if type(feature_dim) is not int or feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if type(dims) is not int or dims < 2:
            raise ValueError("dims must be at least two")
        if not math.isfinite(max_gain) or not 0 <= max_gain < 1:
            raise ValueError("max_gain must be finite and in [0,1)")
        self.dims, self.max_gain, self.eps = dims, float(max_gain), float(eps)
        self.direction = nn.Linear(feature_dim, dims)
        self.speed = nn.Linear(feature_dim, 1)
        self.gain = nn.Linear(feature_dim, 1)
        nn.init.constant_(self.speed.bias, _inverse_softplus(initial_speed))
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.direction.in_features:
            raise ValueError("expected B,F feature matrix")
        direction = F.normalize(self.direction(features), dim=-1, eps=self.eps)
        speed = F.softplus(self.speed(features))
        gain = 1.0 + self.max_gain * torch.tanh(self.gain(features))
        velocity = direction * speed * gain
        return {"velocity": velocity, "direction": direction,
                "speed": speed, "gain": gain}


def scale_guard_loss(prediction: torch.Tensor, target: torch.Tensor,
                     speed_weight: float = 0.25,
                     direction_weight: float = 0.10,
                     eps: float = 1e-6) -> torch.Tensor:
    """New-recipe loss; it does not alter the frozen MSE protocol."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have the same B,D shape")
    if speed_weight < 0 or direction_weight < 0:
        raise ValueError("loss weights must be non-negative")
    vector = F.smooth_l1_loss(prediction, target)
    prediction_speed = torch.linalg.vector_norm(prediction, dim=-1)
    target_speed = torch.linalg.vector_norm(target, dim=-1)
    speed = F.smooth_l1_loss(prediction_speed, target_speed)
    moving = target_speed > eps
    if moving.any():
        pred_direction = prediction / prediction_speed.clamp_min(eps).unsqueeze(-1)
        target_direction = target / target_speed.clamp_min(eps).unsqueeze(-1)
        direction = (1.0 - (pred_direction[moving] * target_direction[moving]).sum(-1)).mean()
    else:
        direction = prediction.new_zeros(())
    return vector + speed_weight * speed + direction_weight * direction
