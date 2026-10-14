"""PFGNet peripheral-frequency core transferred from space to IMU time.

fhjdqaq/PFGNet@9db02476d14bfc457558ebd7fd0b2d68f47dd6fa,
Apache-2.0; LICENSE.upstream retained. This isolated architecture_core_port
does not reproduce the SimVP frame-packing/decoder or its forecasting recipe.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from inertial_benchmark.data.views import CHANNELS
from inertial_benchmark.nn.base import BaseModel, InputSpec


def _contract(spec: InputSpec) -> None:
    if (type(spec.window) is not int or spec.window != 200 or spec.dims != 3
            or tuple(spec.channels) != CHANNELS or spec.rate != 200
            or spec.frame != "gravity_world" or spec.orientation != "reference"
            or spec.remove_gravity or spec.target != "avg_velocity"
            or spec.output_layout != "window" or spec.overlap != "mean" or spec.output_steps
            or spec.history or spec.history_stride or spec.extra_inputs):
        raise ValueError("requires exact canonical B6T200, 200Hz reference gravity_world "
                         "XYZ average velocity, gravity retained, no history/extra inputs")


class TemporalGRN(nn.Module):
    """Source GRN formula, not the differently defined ConvNeXt-V2 GRN."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * x / (x.norm(dim=-1, keepdim=True) + 1e-6) + self.beta


class TemporalRepDW(nn.Module):
    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        # Two orthogonal spatial axes become one temporal axis, not two serial
        # temporal large kernels with a silently doubled effective field.
        self.large = nn.Conv1d(dim, dim, kernel, padding=kernel // 2,
                               groups=dim, bias=False)
        self.small = nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.identity = nn.Conv1d(dim, dim, 1, groups=dim, bias=False)
        # Explicit 1D adaptation: identity for every channel, not only group 0.
        nn.init.ones_(self.identity.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.large(x) + self.small(x) + self.identity(x)


class MSInit1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        share, remainder = divmod(dim, 3)
        self.branches = nn.ModuleList([
            nn.Sequential(TemporalRepDW(dim, kernel), nn.Conv1d(dim, share, 1, bias=False))
            for kernel in (3, 5, 7)
        ])
        self.tail = (nn.Sequential(TemporalRepDW(dim, 3),
                                  nn.Conv1d(dim, remainder, 1, bias=False))
                     if remainder else None)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [branch(x) for branch in self.branches]
        if self.tail is not None:
            parts.append(self.tail(x))
        return F.gelu(self.norm(torch.cat(parts, dim=1)))


class PeripheralBranch1D(nn.Module):
    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        self.large = nn.Conv1d(dim, dim, kernel, padding=kernel // 2,
                               groups=dim, bias=False)
        self.center = nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.large(x) - self.beta.tanh() * self.center(x)


class PeripheralFrequencyGate1D(nn.Module):
    """Gradient/Laplacian/local-variance cues select peripheral scales."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.kernels = (9, 15, 31)
        self.branches = nn.ModuleList([PeripheralBranch1D(dim, kernel)
                                      for kernel in self.kernels])
        self.register_buffer("derivative", torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 3),
                             persistent=False)
        self.register_buffer("laplacian", torch.tensor([1.0, -2.0, 1.0]).view(1, 1, 3),
                             persistent=False)
        self.gate_head = nn.Conv1d(3, len(self.kernels), 1)

    def frequency_cues(self, x: torch.Tensor) -> torch.Tensor:
        first = F.conv1d(x, self.derivative.expand(self.dim, 1, 3),
                         padding=1, groups=self.dim)
        second = F.conv1d(x, self.laplacian.expand(self.dim, 1, 3),
                          padding=1, groups=self.dim)
        mean = F.avg_pool1d(x, 3, stride=1, padding=1)
        variance = (F.avg_pool1d(x.square(), 3, stride=1, padding=1)
                    - mean.square()).clamp_min(0.0)
        return torch.cat((torch.sqrt(first.square() + 1e-6).mean(1, keepdim=True),
                          second.abs().mean(1, keepdim=True),
                          variance.mean(1, keepdim=True)), dim=1)

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate_head(self.frequency_cues(x)).softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.weights(x)
        return sum(branch(x) * alpha[:, index:index + 1]
                   for index, branch in enumerate(self.branches))


class PFGBlock1D(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int) -> None:
        super().__init__()
        groups = min(32, dim)
        self.norm1, self.norm2 = nn.GroupNorm(groups, dim), nn.GroupNorm(groups, dim)
        self.mixer = PeripheralFrequencyGate1D(dim)
        self.grn1, self.grn2 = TemporalGRN(dim), TemporalGRN(dim)
        expanded = dim * mlp_ratio
        self.pw_in = nn.Conv1d(dim, 2 * expanded, 1)
        self.dw_value = nn.Conv1d(expanded, expanded, 3, padding=1,
                                  groups=expanded, bias=False)
        self.pw_out = nn.Conv1d(expanded, dim, 1)
        self.gamma1 = nn.Parameter(torch.full((1, dim, 1), 1e-6))
        self.gamma2 = nn.Parameter(torch.full((1, dim, 1), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma1 * self.grn1(F.gelu(self.mixer(self.norm1(x))))
        gate, value = self.pw_in(self.norm2(x)).chunk(2, dim=1)
        mixed = self.pw_out(F.silu(gate) * self.dw_value(value))
        return x + self.gamma2 * self.grn2(mixed)


class PFGNetVelocity(BaseModel):
    default_loss = "mse"
    training_initialization = "scratch_supervised"
    implementation_kind = "architecture_core_port"
    causality = "window_noncausal"

    def __init__(self, input_spec: InputSpec, d_model: int = 32,
                 depth: int = 2, mlp_ratio: int = 2, pretrained: bool = False) -> None:
        super().__init__(input_spec)
        _contract(input_spec)
        if pretrained is not False:
            raise ValueError("scratch-only profile; pretrained must be False")
        if (any(type(value) is not int or value < 1 for value in (d_model, depth, mlp_ratio))
                or d_model < 3 or d_model % min(32, d_model)):
            raise ValueError("positive integer width/depth/ratio and valid source GroupNorm groups required")
        self.stem = nn.Conv1d(6, d_model, 3, padding=1)
        self.ms_init = MSInit1D(d_model)
        self.blocks = nn.ModuleList([PFGBlock1D(d_model, mlp_ratio) for _ in range(depth)])
        self.velocity_head = nn.Linear(d_model, 3)
        self.model_cfg = {
            "name": "pfgnet_velocity_3d_staging", "arch": "pfgnet_velocity",
            "args": {"d_model": d_model, "depth": depth, "mlp_ratio": mlp_ratio,
                     "pretrained": False},
            "input": {key: value for key, value in input_spec.to_dict().items()
                      if key != "channels"}, "input_channels": list(CHANNELS),
            "registered": False, "implementation_kind": self.implementation_kind,
        }

    def set_loss(self, name: str, **kwargs) -> None:
        if name != "mse" or kwargs:
            raise ValueError("this declared scratch profile accepts plain MSE only")
        super().set_loss(name)

    def encode_window(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        features = self.ms_init(self.stem(imu))
        for block in self.blocks:
            features = block(features)
        return features

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu).mean(dim=-1))}
