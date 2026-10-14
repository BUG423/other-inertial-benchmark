"""UCAN sharing/receiving dual-fusion core, explicitly adapted to 1D IMU.

hokiyoshi/UCAN@b812c0adb8ce70d442973db5c87bab7d1e8c3333,
basicsr/archs/ucan_arch.py; Apache-2.0, LICENSE.upstream retained.
Unregistered architecture_core_port, not the complete SR paper architecture.
All attention caches are local to one forward; no pretrained or persistent state.
"""

from __future__ import annotations

import math

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


class HedgehogFeatureMap(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.layer = nn.Linear(head_dim, head_dim)
        nn.init.eye_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = self.layer(x)
        return torch.cat((2 * projection, -2 * projection), dim=-1).softmax(dim=-1)


class FourierPosition1D(nn.Module):
    """Source learnable Fourier mixing reduced to one time coordinate.

    Retains source signed coefficient-sum normalization and minus rotation.
    A signed epsilon protects near-zero sums; inverse frequencies are buffers,
    instead of the released unregistered tensor and automatic CUDA placement.
    """

    def __init__(self, head_dim: int, heads: int) -> None:
        super().__init__()
        if head_dim < 4 or head_dim % 4:
            raise ValueError("Fourier feature width must be a positive multiple of four")
        inverse = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        inverse = inverse[inverse >= 2 * math.pi / 4096]
        self.register_buffer("inv_frequency", inverse, persistent=False)
        self.half_width = head_dim // 2
        count, output = len(inverse), min(len(inverse), head_dim // 4)
        self.sin_coefficient = nn.Parameter(torch.empty(heads, count, output))
        self.cos_coefficient = nn.Parameter(torch.empty(heads, count, output))
        for coefficient in (self.sin_coefficient, self.cos_coefficient):
            nn.init.xavier_normal_(coefficient, gain=0.5)
            with torch.no_grad():
                for column in range(output):
                    row = column * math.ceil(count / output)
                    if row < count:
                        coefficient[:, row, column] += 1

    @staticmethod
    def normalized(coefficient: torch.Tensor) -> torch.Tensor:
        total = coefficient.sum(dim=-2, keepdim=True)
        sign = torch.where(total < 0, -torch.ones_like(total), torch.ones_like(total))
        return coefficient / (sign * total.abs().clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position = torch.arange(x.shape[-2], dtype=x.dtype, device=x.device)
        phase = position[:, None] * self.inv_frequency.to(dtype=x.dtype)[None, :]
        sine = torch.einsum("tf,hfg->htg", phase.sin(),
                            self.normalized(self.sin_coefficient))
        cosine = torch.einsum("tf,hfg->htg", phase.cos(),
                              self.normalized(self.cos_coefficient))
        # Source pads missing frequencies with ones in BOTH sine/cosine arrays.
        sine = F.pad(sine, (0, self.half_width - sine.shape[-1]), value=1)
        cosine = F.pad(cosine, (0, self.half_width - cosine.shape[-1]), value=1)
        sine, cosine = torch.cat((sine, sine), -1), torch.cat((cosine, cosine), -1)
        first, second = x.chunk(2, dim=-1)
        return x * cosine - torch.cat((-second, first), dim=-1) * sine


class SpatialGateFFN1D(nn.Module):
    def __init__(self, dim: int, ratio: int) -> None:
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.norm = nn.LayerNorm(hidden // 2)
        self.depthwise = nn.Conv1d(hidden // 2, hidden // 2, 7, padding=3,
                                   groups=hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.fc1(x).chunk(2, dim=-1)
        second = self.depthwise(self.norm(second).transpose(1, 2)).transpose(1, 2)
        return self.fc2(F.gelu(first) * second)


class WindowAttention1D(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, receiving: bool) -> None:
        super().__init__()
        self.heads, self.window, self.receiving = heads, window, receiving
        if receiving:
            self.value = nn.Linear(dim, dim)
        else:
            self.qkv = nn.Linear(dim, 3 * dim)
            self.relative_bias = nn.Embedding(2 * window - 1, heads)
            position = torch.arange(window)
            self.register_buffer("relative_index", position[:, None] - position[None, :]
                                 + window - 1, persistent=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, shared_map=None) -> tuple:
        batch, length, dim = x.shape
        padding = (-length) % self.window
        windows = F.pad(x, (0, 0, 0, padding)).reshape(
            batch * ((length + padding) // self.window), self.window, dim)
        head_dim = dim // self.heads

        def heads(value):
            return value.reshape(-1, self.window, self.heads, head_dim).transpose(1, 2)

        if self.receiving:
            if shared_map is None:
                raise ValueError("receiving requires a same-forward local attention map")
            value, attention = heads(self.value(windows)), shared_map
        else:
            query, key, value = (heads(tensor) for tensor in self.qkv(windows).chunk(3, -1))
            valid = (torch.arange(length + padding, device=x.device) < length)
            valid = valid.reshape(-1, self.window).repeat(batch, 1)
            logits = (query @ key.transpose(-2, -1)) * head_dim ** -0.5
            bias = self.relative_bias(self.relative_index).permute(2, 0, 1)
            logits = logits + bias
            logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
            attention = logits.softmax(dim=-1)
        expected = (windows.shape[0], self.heads, self.window, self.window)
        if tuple(attention.shape) != expected:
            raise ValueError("shared attention map does not belong to this window layout")
        result = (attention @ value).transpose(1, 2).reshape(-1, self.window, dim)
        result = self.proj(result).reshape(batch, length + padding, dim)[:, :length]
        return result, attention


class DualFusion1D(nn.Module):
    """Hedgehog global linear fusion plus channel attention, each with own V."""

    def __init__(self, dim: int, heads: int, receiving: bool) -> None:
        super().__init__()
        self.half, self.heads, self.receiving = dim // 2, heads, receiving
        head_dim = self.half // heads
        if not receiving:
            self.query = nn.Linear(dim, self.half)
            self.key = nn.Linear(dim, self.half)
        self.value = nn.Linear(dim, self.half)
        self.query_map, self.key_map = HedgehogFeatureMap(head_dim), HedgehogFeatureMap(head_dim)
        self.position = FourierPosition1D(2 * head_dim, heads)
        self.local_value = nn.Conv1d(self.half, self.half, 3, padding=1, groups=self.half)
        self.temperature = nn.Parameter(torch.ones(self.half, 1))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, shared_qk=None) -> tuple:
        batch, length, _ = x.shape
        if self.receiving:
            if shared_qk is None:
                raise ValueError("receiving requires same-forward unmodified Q/K")
            query, key = shared_qk
        else:
            query, key = self.query(x), self.key(x)
        if tuple(query.shape) != (batch, length, self.half) or key.shape != query.shape:
            raise ValueError("shared Q/K do not belong to this window layout")
        value = self.value(x)

        def heads(tensor):
            return tensor.reshape(batch, length, self.heads, self.half // self.heads).transpose(1, 2)

        query_phi, key_phi = self.query_map(heads(query)), self.key_map(heads(key))
        denominator = (query_phi @ key_phi.mean(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6
        query_position, key_position = self.position(query_phi), self.position(key_phi)
        context = ((key_position.transpose(-2, -1) * length ** -0.5)
                   @ (heads(value) * length ** -0.5))
        global_part = ((query_position @ context) / denominator)
        global_part = global_part.transpose(1, 2).reshape(batch, length, self.half)
        global_part = global_part + self.local_value(value.transpose(1, 2)).transpose(1, 2)
        query_channel = F.normalize(query.transpose(1, 2), dim=-1)
        key_channel = F.normalize(key.transpose(1, 2), dim=-1)
        channel_map = ((query_channel @ key_channel.transpose(-2, -1))
                       * self.temperature).softmax(dim=-1)
        channel_part = (channel_map @ value.transpose(1, 2)).transpose(1, 2)
        return self.proj(torch.cat((global_part, channel_part), dim=-1)), (query, key)


class SharedHybrid1D(nn.Module):
    def __init__(self, dim: int, local_heads: int, global_heads: int,
                 window: int, ratio: int, receiving: bool) -> None:
        super().__init__()
        self.receiving = receiving
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.norm3, self.norm4 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.local = WindowAttention1D(dim, local_heads, window, receiving)
        self.dual = DualFusion1D(dim, global_heads, receiving)
        self.ffn1, self.ffn2 = SpatialGateFFN1D(dim, ratio), SpatialGateFFN1D(dim, ratio)
        self.scale1 = nn.Parameter(torch.full((dim,), 1e-4))
        self.scale2 = nn.Parameter(torch.full((dim,), 1e-4))

    def forward(self, x: torch.Tensor, cache=None) -> tuple:
        if self.receiving and cache is None:
            raise ValueError("receiver cannot synthesize a map/QK from another window")
        local_cache, qk_cache = cache if self.receiving else (None, None)
        shortcut = x
        local, attention = self.local(self.norm1(x), local_cache)
        x = shortcut + local
        x = x + self.ffn1(self.norm2(x)) + shortcut * self.scale1
        shortcut = x
        fused, qk = self.dual(x, qk_cache)
        x = shortcut + self.norm3(fused)
        x = x + self.norm4(self.ffn2(x)) + shortcut * self.scale2
        return x, (attention, qk)


class LargeKernelSpatial1D(nn.Module):
    """Source k=23's 5/dilation2 factors collapsed to time: effective RF 13."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.large = nn.Sequential(nn.Conv1d(dim, dim, 5, padding=2, groups=dim),
            nn.Conv1d(dim, dim, 5, padding=4, dilation=2, groups=dim))
        self.small = nn.Sequential(nn.Conv1d(dim, dim // 4, 1), nn.GELU(),
            nn.Conv1d(dim // 4, dim // 4, 3, padding=1), nn.GELU(),
            nn.Conv1d(dim // 4, dim, 1))
        self.channel = nn.Conv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel(x) * (self.large(x) + self.small(x))


class SelectiveLargeKernel1D(nn.Module):
    def __init__(self, dim: int, ratio: int) -> None:
        super().__init__()
        self.fine_dim = 16  # Actual released BasicLayer uses fixed first sixteen.
        self.ffn = SpatialGateFFN1D(dim, ratio)
        self.large = LargeKernelSpatial1D(self.fine_dim)
        self.merge = nn.Conv1d(dim, dim, 1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.ffn(x).transpose(1, 2)
        fine, remainder = mixed[:, :self.fine_dim], mixed[:, self.fine_dim:]
        mixed = torch.cat((self.large(fine), remainder), dim=1)
        return x + self.merge(mixed).transpose(1, 2)


class UCANVelocity(BaseModel):
    default_loss = "mse"
    training_initialization = "scratch_supervised"
    implementation_kind = "architecture_core_port"
    causality = "window_noncausal"

    def __init__(self, input_spec: InputSpec, d_model: int = 32,
                 local_heads: int = 4, global_heads: int = 2, pairs: int = 1,
                 patch_size: int = 4, window_size: int = 10, mlp_ratio: int = 2,
                 pretrained: bool = False) -> None:
        super().__init__(input_spec)
        _contract(input_spec)
        if pretrained is not False:
            raise ValueError("scratch-only profile; pretrained must be False")
        values = (d_model, local_heads, global_heads, pairs, patch_size, window_size, mlp_ratio)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("width/head/pair/patch/window/MLP settings must be positive integers")
        if (d_model < 16 or d_model % local_heads or d_model % (4 * global_heads)
                or 200 % patch_size or (d_model * mlp_ratio) % 2):
            raise ValueError("incompatible selective/attention/Fourier/patch/GLU widths")
        self.stem = nn.Conv1d(6, d_model, patch_size, stride=patch_size)
        self.sharing = nn.ModuleList([SharedHybrid1D(d_model, local_heads, global_heads,
            window_size, mlp_ratio, False) for _ in range(pairs)])
        self.receiving = nn.ModuleList([SharedHybrid1D(d_model, local_heads, global_heads,
            window_size, mlp_ratio, True) for _ in range(pairs)])
        self.sharing_large = nn.ModuleList([SelectiveLargeKernel1D(d_model, mlp_ratio)
                                             for _ in range(pairs)])
        self.receiving_large = nn.ModuleList([SelectiveLargeKernel1D(d_model, mlp_ratio)
                                               for _ in range(pairs)])
        self.norm = nn.LayerNorm(d_model)
        self.velocity_head = nn.Linear(d_model, 3)
        self.model_cfg = {
            "name": "ucan_velocity_3d_staging", "arch": "ucan_velocity",
            "args": {"d_model": d_model, "local_heads": local_heads,
                     "global_heads": global_heads, "pairs": pairs,
                     "patch_size": patch_size, "window_size": window_size,
                     "mlp_ratio": mlp_ratio, "pretrained": False},
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
        for sender, receiver, sender_large, receiver_large in zip(
                self.sharing, self.receiving, self.sharing_large, self.receiving_large):
            features, cache = sender(features)
            features = sender_large(features)
            features, _ = receiver(features, cache)
            features = receiver_large(features)
        return self.norm(features)

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu).mean(dim=1))}
