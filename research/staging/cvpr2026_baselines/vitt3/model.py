"""Unregistered ViT³ input-only fast-weight core, adapted to 1D IMU windows.

Source: LeapLabTHU/ViTTT@36eb23bfe2eca5cd4ad9ce01ef93bf8a89c6020f,
ttt_block.py, MIT (upstream notice in LICENSE.upstream). This is an
architecture_core_port, not the author's official inertial implementation.
No model execution or capacity acceptance is implied by this static draft.
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


class TemporalTTT(nn.Module):
    """One negative-dot-product inner step, independently for each sample.

    Local adapted weights are tensors, never assigned back to nn.Parameters.
    The local branch is a three-tap temporal replacement for source 3x3 DWConv.
    Scale=1/sqrt(3) explicitly replaces source equivalent-neighborhood scale 9.
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if (type(dim) is not int or type(num_heads) is not int or num_heads < 1
                or dim < 1 or dim % num_heads):
            raise ValueError("dim must be positive and divisible by num_heads")
        self.dim, self.num_heads = dim, num_heads
        d = dim // num_heads
        self.head_dim = d
        if d % 2:
            raise ValueError("temporal RoPE requires an even head dimension")
        self.scale = 3.0 ** -0.5
        self.register_buffer("inv_frequency", 10000.0 ** (
            -torch.arange(0, d, 2, dtype=torch.float32) / d), persistent=False)
        self.qkv = nn.Linear(dim, 3 * dim + 3 * d)
        self.w1 = nn.Parameter(torch.empty(1, num_heads, d, d))
        self.w2 = nn.Parameter(torch.empty(1, num_heads, d, d))
        self.w3 = nn.Parameter(torch.empty(d, 1, 3))
        for weight in (self.w1, self.w2, self.w3):
            nn.init.trunc_normal_(weight, std=0.02)
        self.proj = nn.Linear(dim + d, dim)

    def time_position(self, x: torch.Tensor) -> torch.Tensor:
        """Explicit 1D half-pair RoPE replacing the optional spatial source RoPE."""
        position = torch.arange(x.shape[-2], dtype=x.dtype, device=x.device)
        phase = position[:, None] * self.inv_frequency.to(dtype=x.dtype)[None, :]
        cosine = torch.cat((phase.cos(), phase.cos()), dim=-1)
        sine = torch.cat((phase.sin(), phase.sin()), dim=-1)
        first, second = x.chunk(2, dim=-1)
        return x * cosine + torch.cat((-second, first), dim=-1) * sine

    def inner_swiglu(self, k: torch.Tensor, v: torch.Tensor,
                     w1: torch.Tensor, w2: torch.Tensor) -> tuple:
        z1, z2 = k @ w1, k @ w2
        sig = torch.sigmoid(z2)
        activation = z2 * sig
        # L=-mean_N(sum_d(F_w(k)*v))*scale. No velocity target is accessed.
        e = -v * (self.scale / k.shape[-2])
        g1 = k.transpose(-2, -1) @ (e * activation)
        g2 = k.transpose(-2, -1) @ (
            e * z1 * sig * (1.0 + z2 * (1.0 - sig)))
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        return w1 - g1, w2 - g2

    def inner_conv(self, k: torch.Tensor, v: torch.Tensor,
                   weight: torch.Tensor) -> torch.Tensor:
        # k/v: B,d,N; source spatial gradient calculation reduced to time.
        batch, channels, length = k.shape
        e = -v * (self.scale / length)
        padded = F.pad(k, (1, 1))
        gradient = torch.stack([
            (padded[..., offset:offset + length] * e).sum(dim=-1)
            for offset in range(3)
        ], dim=-1).unsqueeze(-2)
        gradient = gradient / (gradient.norm(dim=(-2, -1), keepdim=True) + 1.0)
        return weight.unsqueeze(0).expand(batch, channels, 1, 3) - gradient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        d = self.head_dim
        q1, k1, v1, q2, k2, v2 = torch.split(
            self.qkv(x), [dim, dim, dim, d, d, d], dim=-1)

        def heads(tensor):
            return tensor.reshape(batch, length, self.num_heads, d).transpose(1, 2)

        q1, k1, v1 = self.time_position(heads(q1)), self.time_position(heads(k1)), heads(v1)
        q2, k2, v2 = (tensor.transpose(1, 2) for tensor in (q2, k2, v2))
        adapted1, adapted2 = self.inner_swiglu(k1, v1, self.w1, self.w2)
        adapted3 = self.inner_conv(k2, v2, self.w3)
        global_part = ((q1 @ adapted1) * F.silu(q1 @ adapted2))
        global_part = global_part.transpose(1, 2).reshape(batch, length, dim)
        local_part = F.conv1d(
            q2.reshape(1, batch * d, length), adapted3.reshape(batch * d, 1, 3),
            padding=1, groups=batch * d)
        local_part = local_part.reshape(batch, d, length).transpose(1, 2)
        return self.proj(torch.cat((global_part, local_part), dim=-1))


class ViT3Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = TemporalTTT(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT3Velocity(BaseModel):
    default_loss = "mse"
    training_initialization = "scratch_supervised"
    implementation_kind = "architecture_core_port"
    causality = "window_noncausal"

    def __init__(self, input_spec: InputSpec, d_model: int = 32,
                 num_heads: int = 4, depth: int = 2, patch_size: int = 4,
                 mlp_ratio: int = 2, pretrained: bool = False) -> None:
        super().__init__(input_spec)
        _contract(input_spec)
        if pretrained is not False:
            raise ValueError("scratch-only profile; pretrained must be False")
        if any(type(value) is not int or value < 1
               for value in (d_model, num_heads, depth, patch_size, mlp_ratio)):
            raise ValueError("width/head/depth/patch/MLP settings must be positive integers")
        if d_model % num_heads or 200 % patch_size:
            raise ValueError("heads must divide width and patch_size must divide T200")
        self.stem = nn.Conv1d(6, d_model, patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([ViT3Block(d_model, num_heads, mlp_ratio)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.velocity_head = nn.Linear(d_model, 3)
        self.model_cfg = {
            "name": "vitt3_velocity_3d_staging", "arch": "vitt3_velocity",
            "args": {"d_model": d_model, "num_heads": num_heads, "depth": depth,
                     "patch_size": patch_size, "mlp_ratio": mlp_ratio, "pretrained": False},
            "input": {key: value for key, value in input_spec.to_dict().items()
                      if key != "channels"},
            "input_channels": list(CHANNELS), "registered": False,
            "implementation_kind": self.implementation_kind,
        }

    def set_loss(self, name: str, **kwargs) -> None:
        if name != "mse" or kwargs:
            raise ValueError("this declared scratch profile accepts plain MSE only")
        super().set_loss(name)

    def encode_window(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        features = self.stem(imu).transpose(1, 2)
        for block in self.blocks:
            features = block(features)
        return self.norm(features)

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu).mean(dim=1))}
