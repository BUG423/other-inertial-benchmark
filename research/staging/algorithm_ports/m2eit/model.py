"""Unregistered M2EIT reconstruction draft; see source-card.md before admission.

Independent mathematical implementation. No author network code was available.
The selective scan is a deliberately explicit, differentiable CPU reference, not
an assertion of official Mamba kernel equivalence or GPU capacity acceptance.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from inertial_benchmark.nn.base import BaseModel, InputSpec


def haar_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal Haar approximation/detail; requires a complete even pair."""
    if x.shape[-1] % 2:
        raise ValueError("Haar expert requires an even temporal length")
    even, odd = x[..., ::2], x[..., 1::2]
    return (even + odd) / math.sqrt(2), (even - odd) / math.sqrt(2)


def selective_scan_reference(
    x: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    skip: torch.Tensor,
) -> torch.Tensor:
    """Mamba selective recurrence, x/delta=(B,L,E), b/c=(B,L,N).

    Stable negative diagonal A and input-dependent B,C,delta are retained. The
    input term uses standard Mamba's delta*B*x discretization, not an ordinary
    fixed-coefficient RNN. Loops preserve autograd and need no compiled extension.
    """
    if x.ndim != 3 or delta.shape != x.shape:
        raise ValueError("scan x and delta must both have shape (B,L,E)")
    if b.shape != c.shape or b.shape[:2] != x.shape[:2]:
        raise ValueError("scan B/C must have shape (B,L,N)")
    if a.shape != (x.shape[-1], b.shape[-1]) or skip.shape != (x.shape[-1],):
        raise ValueError("scan A/skip state dimensions disagree")
    state = x.new_zeros(x.shape[0], x.shape[-1], b.shape[-1])
    outputs = []
    for index in range(x.shape[1]):
        dt = delta[:, index, :, None]
        state = torch.exp(dt * a) * state + dt * b[:, index, None, :] * x[:, index, :, None]
        y = (state * c[:, index, None, :]).sum(-1) + skip * x[:, index]
        outputs.append(y)
    return torch.stack(outputs, dim=1)


class SelectiveMambaReference(nn.Module):
    """One-direction selective SSM with causal depthwise convolution and SiLU gate."""

    def __init__(self, width: int, state_dim: int = 16, conv_kernel: int = 2,
                 expansion: int = 2) -> None:
        super().__init__()
        self.inner = width * expansion
        self.dt_rank = math.ceil(width / 16)
        self.conv_kernel = conv_kernel
        self.in_proj = nn.Linear(width, 2 * self.inner, bias=False)
        self.conv = nn.Conv1d(self.inner, self.inner, conv_kernel,
                              groups=self.inner, padding=conv_kernel - 1)
        self.select = nn.Linear(self.inner, self.dt_rank + 2 * state_dim, bias=False)
        self.delta = nn.Linear(self.dt_rank, self.inner)
        self.a_log = nn.Parameter(torch.arange(1, state_dim + 1).float().log()
                                  .repeat(self.inner, 1))
        self.skip = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, width, bias=False)
        nn.init.uniform_(self.delta.weight, -self.dt_rank ** -.5, self.dt_rank ** -.5)
        with torch.no_grad():
            dt = torch.exp(torch.empty(self.inner).uniform_(math.log(.001), math.log(.1)))
            self.delta.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.state_dim = state_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x, gate = self.in_proj(tokens).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :tokens.shape[1]].transpose(1, 2))
        dt, b, c = self.select(x).split((self.dt_rank, self.state_dim, self.state_dim), dim=-1)
        dt = F.softplus(self.delta(dt))
        scanned = selective_scan_reference(x, dt, -self.a_log.exp(), b, c, self.skip)
        return self.out_proj(scanned * F.silu(gate))


class BiMambaBlock(nn.Module):
    def __init__(self, width: int, state_dim: int, conv_kernel: int, expansion: int) -> None:
        super().__init__()
        self.forward_scan = SelectiveMambaReference(width, state_dim, conv_kernel, expansion)
        self.backward_scan = SelectiveMambaReference(width, state_dim, conv_kernel, expansion)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        forward = self.forward_scan(tokens)
        backward = self.backward_scan(tokens.flip(1)).flip(1)
        return tokens + self.norm(forward + backward)


class SpatialBasicBlock(nn.Module):
    def __init__(self, channels: int, width: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, width, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(width)
        self.skip = (nn.Identity() if channels == width and stride == 1 else
                     nn.Sequential(nn.Conv1d(channels, width, 1, stride, bias=False),
                                   nn.BatchNorm1d(width)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))


class HaarResidualBlock(nn.Module):
    """Eq.11 draft: concatenation along time preserves the residual shape.

    The paper omits the concat axis/residual reconciliation. This choice is an
    explicit assumption, not purported official source behavior. Both frequency
    bands remain separate until their independent learned pointwise mappings.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.low = nn.Sequential(nn.Conv1d(width, width, 1), nn.BatchNorm1d(width))
        self.high = nn.Sequential(nn.Conv1d(width, width, 1), nn.BatchNorm1d(width))
        self.fused_norm = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, high = haar_pair(x)
        return x + self.fused_norm(torch.cat((self.low(low), self.high(high)), dim=-1))


class M2EITDraft(BaseModel):
    """Three independent experts, two cross-domain translations, learned soft router."""

    default_loss = "mse_sum"
    saved_outputs = ("router_weights", "expert_features")

    def __init__(self, input_spec: InputSpec, feature_dim: int = 256,
                 temporal_layers: int = 2, frequency_layers: int = 2,
                 state_dim: int = 16, conv_kernel: int = 2, expansion: int = 2,
                 attention_heads: int = 4, head_hidden: int = 128) -> None:
        super().__init__(input_spec)
        if (input_spec.dims not in (2, 3) or input_spec.num_channels != 6 or
                input_spec.sub_windows or input_spec.output_layout != "window" or
                input_spec.target != "avg_velocity" or input_spec.frame != "gravity_world"):
            raise ValueError("draft requires six-axis gravity_world window average velocity, dims 2/3")
        if input_spec.window % 2 or feature_dim % attention_heads:
            raise ValueError("even window and divisible multihead width are required")
        if min(temporal_layers, frequency_layers, state_dim, expansion) < 1:
            raise ValueError("all expert depths and state dimensions must be positive")
        stem_width = feature_dim // 4
        if stem_width < 1:
            raise ValueError("feature_dim must be >=4")
        # Native IPB gyro-first order is converted to the paper's acc-first order.
        self.spatial_stem = nn.Sequential(nn.Conv1d(6, stem_width, 7, 2, 3, bias=False),
                                         nn.BatchNorm1d(stem_width), nn.ReLU(),
                                         nn.MaxPool1d(3, 2, 1))
        widths = [stem_width, stem_width, 2 * stem_width, 2 * stem_width,
                  feature_dim, feature_dim]
        channels = stem_width
        spatial = []
        for index, width in enumerate(widths):
            spatial.append(SpatialBasicBlock(channels, width, 2 if index in (2, 4) else 1))
            channels = width
        self.spatial = nn.Sequential(*spatial)
        self.sequence_stem = nn.Sequential(nn.Conv1d(6, feature_dim, 1), nn.ReLU())
        self.temporal = nn.Sequential(*[BiMambaBlock(feature_dim, state_dim, conv_kernel, expansion)
                                        for _ in range(temporal_layers)])
        self.frequency = nn.Sequential(*[HaarResidualBlock(feature_dim)
                                         for _ in range(frequency_layers)])
        self.temporal_to_spatial = nn.MultiheadAttention(feature_dim, attention_heads,
                                                       dropout=0, batch_first=True)
        self.frequency_to_spatial = nn.MultiheadAttention(feature_dim, attention_heads,
                                                        dropout=0, batch_first=True)
        self.router = nn.Linear(3 * feature_dim, 3)
        self.head = nn.Sequential(nn.Linear(feature_dim, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, input_spec.dims))

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        x = imu[:, [3, 4, 5, 0, 1, 2]]
        spatial = self.spatial(self.spatial_stem(x)).transpose(1, 2)
        sequence = self.sequence_stem(x)
        temporal = self.temporal(sequence.transpose(1, 2))
        frequency = self.frequency(sequence).transpose(1, 2)
        temporal, _ = self.temporal_to_spatial(temporal, spatial, spatial, need_weights=False)
        frequency, _ = self.frequency_to_spatial(frequency, spatial, spatial, need_weights=False)
        experts = torch.stack((spatial.mean(1), temporal.mean(1), frequency.mean(1)), dim=1)
        weights = self.router(experts.flatten(1)).softmax(-1)
        fused = (weights[..., None] * experts).sum(1)
        return {"vel": self.head(fused), "router_weights": weights, "expert_features": experts}
