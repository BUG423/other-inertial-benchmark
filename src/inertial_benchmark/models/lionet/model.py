"""L-IONet: eight causal gated WaveNet blocks and a horizontal polar head.

Paper: Chen et al., IEEE IoT Journal 7(5), 2020, §IV-C, Eq. (5)-(6).
https://arxiv.org/html/2001.04061v1
The paper specifies gates, dilations, residual/skip connections and 32 filters.
Kernel size 2, 1x1 residual/skip projections and mean pooling are explicit IPB
assumptions; no unavailable official parameter fixture is claimed. The benchmark
adapts relative polar increments to absolute horizontal speed/heading, like its
IONet profile. XYZ adds an independently learned vertical component.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ...nn.base import BaseModel, InputSpec
from ...nn.registry import register_model
from ..ionet.model import CHANNEL_PERMUTATION


class CausalGatedBlock(nn.Module):
    """WaveNet tanh/sigmoid gate; left padding cannot expose future samples."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left = (kernel_size - 1) * dilation
        self.gate = nn.Conv1d(channels, 2 * channels, kernel_size, dilation=dilation)
        self.residual = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        f, g = self.gate(F.pad(x, (self.left, 0))).chunk(2, dim=1)
        z = f.tanh() * g.sigmoid()
        return x + self.residual(z), self.skip(z)


@register_model("lionet")
class LIONet(BaseModel):
    """Paper WaveNet feature extractor with declared unified polar adaptation."""

    default_loss = "ionet_polar"

    def __init__(self, input_spec: InputSpec, filters: int = 32, layers: int = 8,
                 kernel_size: int = 2, decimation: int = 2) -> None:
        super().__init__(input_spec)
        if input_spec.output_layout != "window" or input_spec.frame != "gravity_world":
            raise ValueError("lionet unified polar head needs window targets in gravity_world")
        if input_spec.dims not in (2, 3):
            raise ValueError("lionet supports horizontal 2D or explicitly adapted XYZ")
        if min(filters, layers, decimation) < 1 or kernel_size < 2:
            raise ValueError("filters/layers/decimation must be positive and kernel_size >= 2")
        if input_spec.window % decimation:
            raise ValueError("lionet window must be divisible by decimation")
        self.decimation = int(decimation)
        self.stem = nn.Conv1d(input_spec.num_channels, filters, 1)
        self.blocks = nn.ModuleList([
            CausalGatedBlock(filters, kernel_size, 2 ** i) for i in range(layers)
        ])
        self.head = nn.Linear(filters, input_spec.dims)

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        x = imu[:, CHANNEL_PERMUTATION, :]
        if self.decimation > 1:
            x = F.avg_pool1d(x, self.decimation, self.decimation)
        x = self.stem(x)
        skips = []
        for block in self.blocks:
            x, skip = block(x)
            skips.append(skip)
        polar = self.head(torch.stack(skips).sum(0).mean(dim=-1))
        speed, heading = polar[:, 0], polar[:, 1]
        xy = torch.stack([speed * heading.cos(), speed * heading.sin()], dim=-1)
        velocity = xy if self.input_spec.dims == 2 else torch.cat([xy, polar[:, 2:3]], -1)
        return {"vel": velocity, "speed": speed, "heading": heading}
