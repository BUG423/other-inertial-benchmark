"""Edge-RecViT released routing/recursive core, explicitly adapted to IMU time.

RicardoLee510520/Edge-RecViT@9d0d34a8b3029d62955c9a1cf54ff7e4409ff934,
Apache-2.0. Original source notice: Copyright 2024 The HuggingFace Team.
All rights reserved. LICENSE.upstream and provenance retain source identity.
Released three-layer router/ST estimator is used, not paper two-layer EARR.
Dense masked execution makes no sparse-FLOPs or speedup claim.
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


class RecursiveTransformerUnit(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        d = dim // self.num_heads
        q, k, v = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.num_heads, d).permute(2, 0, 3, 1, 4).unbind(0)
        weights = ((q @ k.transpose(-2, -1)) * (d ** -0.5)).softmax(-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        x = x + self.proj(context)
        return x + self.mlp(self.norm2(x))


class TemporalDepthRouter(nn.Module):
    """Source three-layer MLP and clamped-uniform Gumbel straight-through."""

    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.network = nn.Sequential(
            nn.Linear(dim, 2 * dim, bias=False), nn.GELU(),
            nn.Linear(2 * dim, 2 * dim, bias=False), nn.ReLU(),
            nn.Linear(2 * dim, depth, bias=False))
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)

    def choices(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        if self.training:
            uniform = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            noise = -torch.log(-torch.log(uniform))
            soft = (logits + noise).softmax(-1)  # source temperature fixed to 1
            hard = F.one_hot(soft.argmax(-1), self.depth).to(dtype=soft.dtype)
            selected = (hard - soft).detach() + soft
        else:
            selected = F.one_hot(logits.argmax(-1), self.depth).to(dtype=logits.dtype)
        # No in-place modification of the estimator; CLS always traverses all L.
        full = torch.zeros_like(selected[:, :1])
        full[..., -1] = 1.0
        return torch.cat((full, selected[:, 1:]), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected = self.choices(x)
        return 1.0 - selected.cumsum(-1) + selected


class EdgeRecViTVelocity(BaseModel):
    default_loss = "mse"
    training_initialization = "scratch_supervised"
    implementation_kind = "architecture_core_port"
    causality = "window_noncausal"

    def __init__(self, input_spec: InputSpec, d_model: int = 32,
                 num_heads: int = 4, depth: int = 4, patch_size: int = 4,
                 mlp_ratio: int = 2, pretrained: bool = False) -> None:
        super().__init__(input_spec)
        _contract(input_spec)
        if pretrained is not False:
            raise ValueError("scratch-only profile; pretrained must be False")
        if (any(type(value) is not int or value < 1
                for value in (d_model, num_heads, depth, patch_size, mlp_ratio))
                or depth < 3 or d_model % num_heads or 200 % patch_size):
            raise ValueError("positive settings, depth>=3, width/head and T200/patch divisibility required")
        self.depth = depth
        self.stem = nn.Conv1d(6, d_model, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.position = nn.Parameter(torch.empty(1, 200 // patch_size + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        self.router = TemporalDepthRouter(d_model, depth)
        self.head_layer = RecursiveTransformerUnit(d_model, num_heads, mlp_ratio)
        self.middle_layer = RecursiveTransformerUnit(d_model, num_heads, mlp_ratio)
        self.tail_layer = RecursiveTransformerUnit(d_model, num_heads, mlp_ratio)
        self.norm = nn.LayerNorm(d_model)
        self.velocity_head = nn.Linear(d_model, 3)
        self.model_cfg = {
            "name": "edge_recvit_velocity_3d_staging", "arch": "edge_recvit_velocity",
            "args": {"d_model": d_model, "num_heads": num_heads, "depth": depth,
                     "patch_size": patch_size, "mlp_ratio": mlp_ratio, "pretrained": False},
            "input": {key: value for key, value in input_spec.to_dict().items()
                      if key != "channels"},
            "input_channels": list(CHANNELS), "registered": False,
            "implementation_kind": self.implementation_kind,
        }

    def tokenize(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        tokens = self.stem(imu).transpose(1, 2)
        return torch.cat((self.cls_token.expand(imu.shape[0], -1, -1), tokens), dim=1) + self.position

    def recurse(self, tokens: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        if gates.shape != (*tokens.shape[:2], self.depth):
            raise ValueError("one survival gate for each token and recursive step required")
        for step in range(self.depth):
            layer = (self.head_layer if step == 0 else
                     self.tail_layer if step == self.depth - 1 else self.middle_layer)
            # Identical released masked-residual update; every step computes a
            # dense block, even if only some token state writes remain active.
            tokens = tokens + gates[..., step:step + 1] * (layer(tokens) - tokens)
        return self.norm(tokens)

    def set_loss(self, name: str, **kwargs) -> None:
        if name != "mse" or kwargs:
            raise ValueError("this declared scratch profile accepts plain MSE only")
        super().set_loss(name)

    def encode_window(self, imu: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenize(imu)
        return self.recurse(tokens, self.router(tokens))

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu)[:, 0])}
