"""Unregistered MS-Temba architecture-core velocity port (pure Torch reference).

Source: thearkaprava/MS-Temba@d42b4606d03884bfe2591d0bed1ed7a1dd01ab22.
Retains the three sequential dilated BiMamba-v2 scales and SSM fuser. The
selective recurrence is an eager reference, NOT a fused CUDA kernel, S4D, or
ordinary fixed-parameter convolution. Source C-alignment/scale classification
losses are intentionally absent from this scratch/MSE-only architecture identity.
Upstream Apache-2.0 notices are recorded in provenance and LICENSE.upstream.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from inertial_benchmark.data.views import CHANNELS
from inertial_benchmark.nn.base import BaseModel, InputSpec


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_spec(spec: InputSpec) -> None:
    if (tuple(spec.channels) != CHANNELS or spec.window != 200 or spec.dims != 3
            or spec.rate != 200 or not math.isfinite(float(spec.rate))
            or spec.target != "avg_velocity" or spec.output_layout != "window"
            or spec.frame != "gravity_world" or spec.orientation != "reference"
            or spec.remove_gravity is not False or spec.history != 0
            or spec.history_stride != 0 or spec.output_steps != 0
            or spec.overlap != "mean" or spec.extra_inputs):
        raise ValueError("MS-Temba staging requires ordered gyro_xyz/acc_xyz, T200/200Hz, "
                         "XYZ average velocity, gravity_world/reference, retained gravity, "
                         "mean overlap and no history/extra inputs")


def selective_scan_reference(u: torch.Tensor, delta: torch.Tensor,
                             A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                             D: torch.Tensor, z: torch.Tensor,
                             delta_bias: torch.Tensor) -> torch.Tensor:
    """Real input-dependent selective scan, zero state on EVERY call.

    Layouts: u/delta/z=(B,H,T), A=(H,N), B/C=(B,N,T), D/bias=(H,).
    h[t]=exp(softplus(dt[t]+bias)*A)*h[t-1]+dt[t]*B[t]*u[t].
    y[t]=(C[t] dot h[t]+D*u[t])*silu(z[t]). This is the upstream
    selective_scan_ref Euler input term, NOT the S4D/ZOH expm1(A*dt)/A term.
    FP32 accumulation is used for FP16/BF16/FP32, FP64 for explicit doubles.
    No inference cache, persistent state, or implicit streaming interface exists.
    """
    if (u.ndim != 3 or delta.shape != u.shape or z.shape != u.shape
            or A.ndim != 2 or A.shape[0] != u.shape[1]
            or B.shape != (u.shape[0], A.shape[1], u.shape[2]) or C.shape != B.shape
            or D.shape != (u.shape[1],) or delta_bias.shape != D.shape
            or u.shape[2] < 1):
        raise ValueError("selective scan tensor layouts are inconsistent")
    dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    values, gates = u.to(dtype), z.to(dtype)
    dt = F.softplus(delta.to(dtype) + delta_bias.to(dtype)[None, :, None])
    A, B, C, D = A.to(dtype), B.to(dtype), C.to(dtype), D.to(dtype)
    state = values.new_zeros((u.shape[0], u.shape[1], A.shape[1]))
    outputs = []
    for index in range(u.shape[-1]):
        step = dt[:, :, index]
        state = (torch.exp(step.unsqueeze(-1) * A.unsqueeze(0)) * state
                 + step.unsqueeze(-1) * B[:, None, :, index]
                 * values[:, :, index].unsqueeze(-1))
        readout = (state * C[:, None, :, index]).sum(dim=-1)
        readout = readout + D.unsqueeze(0) * values[:, :, index]
        outputs.append(readout * F.silu(gates[:, :, index]))
    return torch.stack(outputs, dim=-1).to(u.dtype)


class _SelectiveDirection(nn.Module):
    """One source BiMamba-v2 direction: depthwise causal conv + selective scan."""

    def __init__(self, inner_width: int, state_size: int, dt_rank: int,
                 conv_size: int) -> None:
        super().__init__()
        self.state_size, self.dt_rank = state_size, dt_rank
        self.conv = nn.Conv1d(inner_width, inner_width, conv_size,
                              groups=inner_width, padding=conv_size - 1, bias=True)
        self.x_proj = nn.Linear(inner_width, dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner_width, bias=True)
        self.A_log = nn.Parameter(torch.arange(1, state_size + 1, dtype=torch.float32)
                                  .log().repeat(inner_width, 1))
        self.D = nn.Parameter(torch.ones(inner_width))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.conv(x)[..., :x.shape[-1]])
        parameters = self.x_proj(x.transpose(1, 2))
        dt, B, C = parameters.split((self.dt_rank, self.state_size, self.state_size), dim=-1)
        # Source keeps dt bias separate and applies softplus inside the scan.
        delta = F.linear(dt, self.dt_proj.weight).transpose(1, 2)
        return selective_scan_reference(x, delta, -self.A_log.exp(),
                                        B.transpose(1, 2), C.transpose(1, 2),
                                        self.D, z, self.dt_proj.bias)


class BidirectionalSelectiveSSM(nn.Module):
    """Source BiMamba-v2: shared input projection, independent directional cores."""

    def __init__(self, width: int, state_size: int = 16, conv_size: int = 4) -> None:
        super().__init__()
        for value, name in ((width, "width"), (state_size, "state_size"), (conv_size, "conv_size")):
            _positive_int(value, name)
        inner_width = 2 * width
        self.in_proj = nn.Linear(width, 2 * inner_width, bias=False)
        rank = math.ceil(width / 16)
        self.forward_direction = _SelectiveDirection(inner_width, state_size, rank, conv_size)
        self.backward_direction = _SelectiveDirection(inner_width, state_size, rank, conv_size)
        self.out_proj = nn.Linear(inner_width, width, bias=False)

    def directional_outputs(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, z = self.in_proj(tokens).transpose(1, 2).chunk(2, dim=1)
        forward = self.forward_direction(x, z)
        backward = self.backward_direction(x.flip(-1), z.flip(-1)).flip(-1)
        return forward, backward

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        forward, backward = self.directional_outputs(tokens)
        # Source if_divide_out=True combines both directions BEFORE output projection.
        return self.out_proj(((forward + backward) / 2).transpose(1, 2))


class _RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        dtype = torch.float64 if tokens.dtype == torch.float64 else torch.float32
        values = tokens.to(dtype)
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(tokens.dtype) * self.weight


class _DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return tokens
        keep = 1 - self.probability
        mask = tokens.new_empty((tokens.shape[0], 1, 1)).bernoulli_(keep)
        return tokens * mask / keep


class TembaSSMBlock(nn.Module):
    """One source depth=1 VisionMamba block, including final add+RMSNorm.

    Default scientific/native validation is FP32 with AMP disabled. This eager
    block has no explicit FP32 residual cast. FP32 selective-scan accumulation
    does NOT establish upstream residual_in_fp32 or mixed-precision parity.
    Low-precision/fused addnorm behavior is unverified and separately gated.
    """

    def __init__(self, width: int, state_size: int, conv_size: int,
                 drop_path: float) -> None:
        super().__init__()
        self.norm = _RMSNorm(width)
        self.mixer = BidirectionalSelectiveSSM(width, state_size, conv_size)
        self.final_drop_path = _DropPath(drop_path)
        self.norm_f = _RMSNorm(width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # The first source layer has no stochastic depth. The final norm applies
        # the separate source VisionMamba drop_path (default0.1 even at depth1).
        mixed = self.mixer(self.norm(tokens))
        return self.norm_f(tokens + self.final_drop_path(mixed))


class DilatedTembaScale(nn.Module):
    """Source sequential scale, with independent phase modules and zero padding."""

    def __init__(self, in_width: int, width: int, dilation: int, state_size: int,
                 conv_size: int, drop_path: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.projection = nn.Sequential(nn.Linear(in_width, width), nn.LayerNorm(width), nn.GELU())
        self.phases = nn.ModuleList([TembaSSMBlock(width, state_size, conv_size, drop_path)
                                    for _ in range(dilation)])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(tokens)
        length = tokens.shape[1]
        padding = (-length) % self.dilation
        if padding:
            tokens = F.pad(tokens, (0, 0, 0, padding))
        outputs = [phase(tokens[:, offset::self.dilation])
                   for offset, phase in enumerate(self.phases)]
        # Stack by time-within-phase, then phase, restoring exact original order.
        restored = torch.stack(outputs, dim=2).flatten(1, 2)
        return restored[:, :length]


class MSTembaVelocity(BaseModel):
    default_loss = "mse"
    implementation_kind = "architecture_core_torch_reference"
    causality = "window_noncausal"
    source_commit = "d42b4606d03884bfe2591d0bed1ed7a1dd01ab22"
    training_initialization = "scratch_supervised"

    def __init__(self, input_spec: InputSpec, embed_dims: tuple = (256, 384, 576),
                 state_size: int = 16, conv_size: int = 4, drop_path: float = 0.1,
                 backend: str = "torch_reference", pretrained: bool = False) -> None:
        super().__init__(input_spec)
        _validate_spec(input_spec)
        if pretrained is not False or backend != "torch_reference":
            raise ValueError("only scratch pure-Torch selective-scan reference is implemented")
        if (not isinstance(embed_dims, (tuple, list)) or len(embed_dims) != 3
                or any(type(value) is not int or value < 1 for value in embed_dims)):
            raise ValueError("embed_dims requires three positive integer widths")
        _positive_int(state_size, "state_size")
        _positive_int(conv_size, "conv_size")
        if not math.isfinite(drop_path) or not 0 <= drop_path < 1:
            raise ValueError("drop_path must be finite and in [0,1)")
        widths = tuple(embed_dims)
        self.input_projection = nn.Linear(6, widths[0])
        self.scales = nn.ModuleList([
            DilatedTembaScale(widths[0] if index == 0 else widths[index - 1],
                             widths[index], index + 1, state_size, conv_size, drop_path)
            for index in range(3)])
        self.scale_projections = nn.ModuleList([nn.Linear(width, widths[-1]) for width in widths])
        self.fusion_norm = nn.LayerNorm(widths[-1])  # same norm shared across all three scales
        self.fuser = TembaSSMBlock(widths[-1], state_size, conv_size, drop_path)
        self.velocity_head = nn.Linear(widths[-1], 3)
        # Upstream MSTemba.apply(_init_weights) overrides ALL linear weights/biases,
        # including Mamba dt_proj initialization: std0.02 truncated normal + zero
        # bias. Do not silently substitute generic Mamba log-uniform dt init.
        self.apply(self._source_final_init)
        args = dict(embed_dims=list(widths), state_size=state_size, conv_size=conv_size,
                    drop_path=drop_path, backend=backend, pretrained=False)
        self.model_cfg = dict(name="mstemba_architecture_core_velocity_3d_staging",
                              arch="mstemba_architecture_core_velocity", args=args,
                              loss="mse", causality=self.causality,
                              implementation_kind=self.implementation_kind,
                              auxiliary_losses="omitted", registered=False)

    @staticmethod
    def _source_final_init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def set_loss(self, name: str, **kwargs) -> None:
        if name != "mse" or kwargs:
            raise ValueError("this architecture identity is MSE-only; use a new identity for auxiliary losses")
        super().set_loss(name)

    def encode_window(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        tokens = self.input_projection(imu.transpose(1, 2))
        outputs = []
        for scale in self.scales:
            tokens = scale(tokens)
            outputs.append(tokens)
        fused = sum(self.fusion_norm(projection(value))
                    for projection, value in zip(self.scale_projections, outputs))
        return self.fuser(fused)

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu).mean(dim=1))}
