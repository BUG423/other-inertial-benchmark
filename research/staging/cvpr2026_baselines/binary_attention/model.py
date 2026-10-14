"""Floating fake-quant BinaryAttention core; no binary CUDA kernel or speed claim.

EdwardChasel/BinaryAttention@a8e0268ffc63ed4bb26bceac5fa015d358732055,
Apache-2.0. Quantizer source notice: Copyright (c) 2015-present, Facebook, Inc.
All rights reserved; modified upstream by Chaodong Xiao. LICENSE.upstream kept.
This is an unregistered six-axis 1D architecture_core_port, not paper reproduction.
"""

from __future__ import annotations

import torch
from torch import nn

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


class PublishedSTESign(torch.autograd.Function):
    """Retain source backward: zero outside the closed interval [-1, +1]."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign() + (x == 0).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return gradient * ((x <= 1.0) & (x >= -1.0)).to(dtype=gradient.dtype)


class PublishedSymmetricQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        maximum = x.abs().amax(dim=-2, keepdim=True).detach()
        scale = 127.0 / (maximum + 1e-6)
        return (x * scale).round() / (scale + 1e-6)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        # Source clipping belongs to backward only, not forward tensor clipping.
        return gradient * ((x < 2.0) & (x > -2.0)).to(dtype=gradient.dtype)


def scaled_sign(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
    return scale * PublishedSTESign.apply(x)


def quantize_probability(x: torch.Tensor) -> torch.Tensor:
    # Same round_ste + UINT8 bounds as source, still represented by floating tensors.
    scaled = x * 255.0
    rounded = scaled + (scaled.round() - scaled).detach()
    return rounded.clamp(0.0, 255.0) / 255.0


class BinaryQKAttention1D(nn.Module):
    def __init__(self, dim: int, num_heads: int, tokens: int,
                 pv_quant: bool = True) -> None:
        super().__init__()
        self.num_heads, self.tokens = num_heads, tokens
        self.scale = (dim // num_heads) ** -0.5
        self.pv_quant = pv_quant
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        distances = 2 * tokens - 1 + 3
        self.relative_bias = nn.Parameter(torch.empty(distances, num_heads))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        index = torch.full((tokens + 1, tokens + 1), distances - 3, dtype=torch.long)
        positions = torch.arange(tokens)
        index[1:, 1:] = positions[:, None] - positions[None, :] + tokens - 1
        index[1:, 0] = distances - 2
        index[0, 0] = distances - 1
        self.register_buffer("relative_index", index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        if length != self.tokens + 1:
            raise ValueError("attention token count must match fixed relative-time bias")
        q, k, v = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (scaled_sign(q) @ scaled_sign(k).transpose(-2, -1)) * self.scale
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)
        probability = (scores + bias.unsqueeze(0)).softmax(dim=-1)
        if self.pv_quant:
            probability = quantize_probability(probability)
            v = PublishedSymmetricQuant.apply(v)
        features = (probability @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.proj(features)


class BinaryTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, tokens: int,
                 mlp_ratio: int, pv_quant: bool) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = BinaryQKAttention1D(dim, heads, tokens, pv_quant)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class BinaryAttentionVelocity(BaseModel):
    default_loss = "mse"
    training_initialization = "scratch_supervised"
    implementation_kind = "architecture_core_port"
    causality = "window_noncausal"
    hardware_implementation = "floating_fake_quant_reference"

    def __init__(self, input_spec: InputSpec, d_model: int = 32,
                 num_heads: int = 4, depth: int = 2, patch_size: int = 4,
                 mlp_ratio: int = 2, pv_quant: bool = True,
                 pretrained: bool = False) -> None:
        super().__init__(input_spec)
        _contract(input_spec)
        if pretrained is not False:
            raise ValueError("scratch-only profile; pretrained must be False")
        if (any(type(value) is not int or value < 1
                for value in (d_model, num_heads, depth, patch_size, mlp_ratio))
                or d_model % num_heads or 200 % patch_size or type(pv_quant) is not bool):
            raise ValueError("positive divisible width/head/patch settings and bool pv_quant required")
        tokens = 200 // patch_size
        self.stem = nn.Conv1d(6, d_model, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.position = nn.Parameter(torch.empty(1, tokens + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([BinaryTransformerBlock(
            d_model, num_heads, tokens, mlp_ratio, pv_quant) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.velocity_head = nn.Linear(d_model, 3)
        self.model_cfg = {
            "name": "binary_attention_velocity_3d_staging", "arch": "binary_attention_velocity",
            "args": {"d_model": d_model, "num_heads": num_heads, "depth": depth,
                     "patch_size": patch_size, "mlp_ratio": mlp_ratio,
                     "pv_quant": pv_quant, "pretrained": False},
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
        tokens = self.stem(imu).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(imu.shape[0], -1, -1), tokens), dim=1) + self.position
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu)[:, 0])}
