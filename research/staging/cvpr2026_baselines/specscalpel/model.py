"""Unregistered SpectralScalpel MASF/FACM inertial architecture-core port.

HaoyuJi/SpecScalpel@01adb1ed708d410023b8482c52270cac358c8525 (MIT).
Independent compact adaptation of the2026 spectral modules, not the complete
skeleton action segmentation model. V=1 is a singleton latent feature location,
NOT a body joint or an image made from heterogeneous IMU axes. All action/text/
boundary/AADL objectives and graph/ASRF stages are omitted from this MSE identity.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from inertial_benchmark.data.views import CHANNELS
from inertial_benchmark.nn.base import BaseModel, InputSpec


def _validate_spec(spec: InputSpec) -> None:
    if (tuple(spec.channels) != CHANNELS or spec.window != 200 or spec.dims != 3
            or spec.rate != 200 or not math.isfinite(float(spec.rate))
            or spec.target != "avg_velocity" or spec.output_layout != "window"
            or spec.frame != "gravity_world" or spec.orientation != "reference"
            or spec.remove_gravity is not False or spec.history != 0
            or spec.history_stride != 0 or spec.output_steps != 0
            or spec.overlap != "mean" or spec.extra_inputs):
        raise ValueError("SpectralScalpel requires ordered gyro_xyz/acc_xyz, T200/200Hz, "
                         "XYZ average velocity, gravity_world/reference, retained gravity "
                         "and no history/extra inputs")


class DynamicSpectralWeights(nn.Module):
    """Source CustomProcessing with V=1; emits (four_filters,B,channels)."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((64, 1))
        self.spatial_linear = nn.Linear(1, 4)
        self.temporal_linear = nn.Linear(64, 4)
        self.filter_linear = nn.Linear(16, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.spatial_linear(self.pool(features))
        values = self.temporal_linear(values.transpose(2, 3)).transpose(2, 3)
        return self.filter_linear(values.flatten(2)).permute(2, 0, 1)


class MultiScaleAdaptiveSpectralFilter(nn.Module):
    """MASF: source40/48/56/64 filters, nearest interpolation and dual weighting."""

    filter_lengths = (40, 48, 56, 64)

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.filters = nn.ParameterList([nn.Parameter(torch.ones(1, 1, length, 1))
                                         for length in self.filter_lengths])
        # Source hardcodes64 feature channels. Generalizing that axis is only a
        # separately recorded small draft; the main source-width profile stays64.
        self.static_logits = nn.Parameter(torch.zeros(4, 1, channels, 1, 1))
        self.dynamic_logits = DynamicSpectralWeights()

    def weighting(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dynamic = self.dynamic_logits(features.abs()).softmax(dim=0)[..., None, None]
        return dynamic, self.static_logits.softmax(dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.channels or features.shape[-1] != 1:
            raise ValueError("MASF expects (B,channels,T,1), not skeleton coordinates")
        spectrum = torch.fft.rfft(features, dim=2)
        dynamic, static = self.weighting(features)
        filtered = []
        for filter_ in self.filters:
            expanded = F.interpolate(filter_, size=(spectrum.shape[2], 1), mode="nearest")
            filtered.append(torch.fft.irfft(spectrum * expanded, n=features.shape[2], dim=2))
        values = torch.stack(filtered, dim=0)
        mixed = ((values * dynamic).sum(dim=0) + (values * static).sum(dim=0)) / 2
        return (mixed + features) / 2


class FrequencyAwareChannelMixer(nn.Module):
    """FACM: identical pointwise transforms for real/imag, then irFFT + residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv1d(channels, channels, 1)
        self.second = nn.Conv1d(channels, channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        length = features.shape[-1]
        spectrum = torch.fft.rfft(features, dim=-1)
        # Source stacks along batch, NOT along feature channels. It executes no
        # activation/BN despite allocating unused BN/ReLU modules upstream.
        pairs = torch.cat((spectrum.real, spectrum.imag), dim=0)
        real, imag = self.second(self.first(pairs)).chunk(2, dim=0)
        return torch.fft.irfft(torch.complex(real, imag), n=length, dim=-1) + features


class SigmoidLinearAttention(nn.Module):
    """Source positive feature map is sigmoid (despite upstream method name elu)."""

    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.heads = heads
        self.query, self.key, self.value = (nn.Linear(channels, channels) for _ in range(3))
        self.output = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, channels = tokens.shape
        def split(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, length, self.heads, channels // self.heads).transpose(1, 2)
        query, key = split(self.query(tokens)).sigmoid(), split(self.key(tokens)).sigmoid()
        value = split(self.value(tokens))
        kv = torch.einsum("bhsd,bhse->bhde", key, value)
        # Source epsilon is inside the summed-key vector, not a denominator clamp.
        inverse = torch.einsum("bhld,bhd->bhl", query, key.sum(dim=-2) + 1e-6).reciprocal()
        attended = torch.einsum("bhde,bhld,bhl->bhle", kv, query, inverse)
        attended = attended.transpose(1, 2).reshape(batch, length, channels)
        return self.dropout(self.output(attended))


class SpectralTemporalBlock(nn.Module):
    """Source encoder AttModule followed by FACM; no skeleton SFI/refinement."""

    def __init__(self, channels: int, dilation: int, heads: int,
                 dropout: float, attention_dropout: float) -> None:
        super().__init__()
        self.feed_forward = nn.Sequential(nn.Conv1d(channels, channels, 3,
                                                   padding=dilation, dilation=dilation), nn.ReLU())
        self.instance_norm = nn.InstanceNorm1d(channels, track_running_stats=False)
        self.attention = SigmoidLinearAttention(channels, heads, attention_dropout)
        self.output = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.spectral_mixer = FrequencyAwareChannelMixer(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.feed_forward(features)
        normalized = self.instance_norm(values).transpose(1, 2)
        values = values + self.attention(normalized).transpose(1, 2)
        values = features + self.dropout(self.output(values))
        return self.spectral_mixer(values)


class SpectralScalpelVelocity(BaseModel):
    default_loss = "mse"
    implementation_kind = "spectral_architecture_core_port"
    causality = "window_noncausal"
    source_commit = "01adb1ed708d410023b8482c52270cac358c8525"
    training_initialization = "scratch_supervised"

    def __init__(self, input_spec: InputSpec, channels: int = 64, layers: int = 10,
                 heads: int = 4, dropout: float = 0.5, attention_dropout: float = 0.05,
                 pretrained: bool = False) -> None:
        super().__init__(input_spec)
        _validate_spec(input_spec)
        if pretrained is not False:
            raise ValueError("only scratch MSE architecture-core adaptation is implemented")
        for value, name in ((channels, "channels"), (layers, "layers"), (heads, "heads")):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        if not all(math.isfinite(value) and 0 <= value < 1 for value in (dropout, attention_dropout)):
            raise ValueError("dropout values must be finite and in [0,1)")
        self.input_projection = nn.Conv1d(6, channels, 1)
        self.spectral_filter = MultiScaleAdaptiveSpectralFilter(channels)
        self.temporal_blocks = nn.ModuleList([
            SpectralTemporalBlock(channels, 2 ** index, heads, dropout, attention_dropout)
            for index in range(layers)])
        self.velocity_head = nn.Linear(channels, 3)
        self.model_cfg = dict(name="specscalpel_spectral_core_velocity_3d_staging",
                              arch="specscalpel_spectral_core_velocity", loss="mse", registered=False,
                              args=dict(channels=channels, layers=layers, heads=heads, dropout=dropout,
                                        attention_dropout=attention_dropout, pretrained=False),
                              causality=self.causality, singleton_feature_location=True,
                              implementation_kind=self.implementation_kind,
                              omitted_objectives="action/text/boundary/AADL")

    def set_loss(self, name: str, **kwargs) -> None:
        if name != "mse" or kwargs:
            raise ValueError("this architecture identity is MSE-only")
        super().set_loss(name)

    def encode_window(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        features = self.input_projection(imu)
        features = self.spectral_filter(features.unsqueeze(-1)).squeeze(-1)
        for block in self.temporal_blocks:
            features = block(features)
        return features

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu).mean(dim=-1))}
