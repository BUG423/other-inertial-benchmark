"""Unregistered PatchTST velocity PoC.

This is a compact, independent adaptation of the supervised PatchTST
architecture.  Its fidelity boundary is deliberate: overlapping patch tokens
and one shared channel-independent Transformer are retained, while the
forecasting head is replaced by an ordered linear window-average-velocity
adapter.  Only that task adapter mixes channel features; the encoder retains
shared channel-wise paths.  There is no self-supervised/pretrained path.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from inertial_benchmark.nn.base import BaseModel, InputSpec
from inertial_benchmark.data.views import CHANNELS


class _ResidualSelfAttention(nn.Module):
    """Shared multi-head attention with PatchTST's residual score path."""

    def __init__(self, width: int, heads: int, attention_dropout: float,
                 projection_dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.heads = heads
        self.head_width = width // heads
        self.scale = self.head_width ** -0.5
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output = nn.Sequential(nn.Linear(width, width),
                                    nn.Dropout(projection_dropout))

    def forward(self, tokens: torch.Tensor,
                previous_scores: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = tokens.shape

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, length, self.heads, self.head_width).transpose(1, 2)

        query, key, value = heads(self.query(tokens)), heads(self.key(tokens)), heads(
            self.value(tokens))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if previous_scores is not None:
            scores = scores + previous_scores
        attention = self.attention_dropout(scores.softmax(dim=-1))
        encoded = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, width)
        return self.output(encoded), scores


class _PatchEncoderLayer(nn.Module):
    """Post-norm vanilla encoder layer using temporal BatchNorm as PatchTST does."""

    def __init__(self, width: int, heads: int, feedforward_width: int,
                 dropout: float, attention_dropout: float) -> None:
        super().__init__()
        self.attention = _ResidualSelfAttention(
            width, heads, attention_dropout, dropout)
        # Upstream has both projection dropout (inside attention) and a second
        # residual dropout before Add+Norm. They are distinct random masks.
        self.attention_residual_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.BatchNorm1d(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_width, width),
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.BatchNorm1d(width)

    @staticmethod
    def _batch_norm(norm: nn.BatchNorm1d, tokens: torch.Tensor) -> torch.Tensor:
        return norm(tokens.transpose(1, 2)).transpose(1, 2)

    def forward(self, tokens: torch.Tensor,
                previous_scores: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, scores = self.attention(tokens, previous_scores)
        tokens = self._batch_norm(
            self.attention_norm, tokens + self.attention_residual_dropout(attended))
        transformed = self.feedforward(tokens)
        tokens = self._batch_norm(
            self.feedforward_norm, tokens + self.feedforward_dropout(transformed))
        return tokens, scores


class SharedChannelPatchEncoder(nn.Module):
    """Patch/encode channels with shared weights; return upstream ``(B,C,D,N)``.

    No attention or feed-forward operator mixes channels. In train mode,
    BatchNorm statistics pool over ``B*C`` sequences and their patch positions,
    as upstream does; channel independence is not statistical independence.
    In eval mode fixed running statistics give separate channel forwards.
    """

    def __init__(self, window: int, patch_len: int, stride: int, d_model: int,
                 n_heads: int, n_layers: int, d_ff: int, dropout: float,
                 attention_dropout: float, pad_end: bool) -> None:
        super().__init__()
        if patch_len < 1 or stride < 1 or patch_len > window:
            raise ValueError("patch_len/stride must be positive and patch_len <= window")
        if min(d_model, n_heads, n_layers, d_ff) < 1:
            raise ValueError("Transformer dimensions and depth must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.window = int(window)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.pad_end = bool(pad_end)
        effective = self.window + (self.stride if self.pad_end else 0)
        self.patch_count = (effective - self.patch_len) // self.stride + 1
        self.patch_projection = nn.Linear(self.patch_len, d_model)
        self.position = nn.Parameter(torch.empty(1, self.patch_count, d_model))
        nn.init.uniform_(self.position, -0.02, 0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _PatchEncoderLayer(d_model, n_heads, d_ff, dropout, attention_dropout)
            for _ in range(n_layers)
        ])

    def patchify(self, imu: torch.Tensor) -> torch.Tensor:
        """Return ``(B,C,N,P)`` overlapping patch tokens."""
        if self.pad_end:
            imu = F.pad(imu, (0, self.stride), mode="replicate")
        patches = imu.unfold(-1, self.patch_len, self.stride)
        if patches.shape[-2] != self.patch_count:
            raise RuntimeError("patch count differs from the construction-time contract")
        return patches

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(imu)
        batch, channels, patch_count, _ = patches.shape
        # Folding C into the batch is the paper's channel-independent operation:
        # no token attends to another channel. Train-mode BatchNorm still pools
        # batch statistics over the folded B*C dimension.
        tokens = self.patch_projection(patches).reshape(
            batch * channels, patch_count, -1)
        tokens = self.dropout(tokens + self.position)
        scores = None
        for layer in self.layers:
            tokens, scores = layer(tokens, scores)
        return tokens.reshape(batch, channels, patch_count, -1).transpose(-2, -1)


class PatchTSTVelocity(BaseModel):
    """PatchTST CI encoder with a separate ordered linear velocity adapter.

    The encoder's ``(B,C,D,N)`` output is flattened in channel, feature, patch
    order. A learned linear map keeps gyro/acc units and x/y/z identities
    available to the velocity task. The adapter is an explicit inertial change
    from upstream's per-channel forecasts, not part of the source CI claim.
    """

    default_loss = "mse"
    encoder_channel_independent = True
    task_adapter = "ordered_channel_feature_patch_linear_velocity"
    causality = "window_noncausal"
    training_initialization = "scratch_supervised"

    def __init__(self, input_spec: InputSpec, patch_len: int = 16, stride: int = 8,
                 d_model: int = 128, n_heads: int = 16, n_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.2,
                 attention_dropout: float = 0.0, head_dropout: float = 0.0,
                 pad_end: bool = True, pretrained: bool = False) -> None:
        super().__init__(input_spec)
        if (tuple(input_spec.channels) != CHANNELS or input_spec.dims not in (2, 3)
                or input_spec.sub_windows or input_spec.output_layout != "window"
                or input_spec.target != "avg_velocity"
                or input_spec.frame != "gravity_world"
                or input_spec.rate != 200
                or input_spec.orientation != "reference"
                or input_spec.remove_gravity
                or input_spec.extra_inputs):
            raise ValueError(
                "PatchTSTVelocity requires ordered gyro_xyz/acc_xyz at 200 Hz, "
                "gravity_world/reference orientation with gravity retained, "
                "window average velocity with dims 2 or 3 and no extra_inputs")
        if pretrained:
            raise ValueError("the staging fidelity profile is supervised from scratch")
        if not all(math.isfinite(value) for value in
                   (dropout, attention_dropout, head_dropout)):
            raise ValueError("dropout values must be finite")
        if not 0 <= min(dropout, attention_dropout, head_dropout):
            raise ValueError("dropout values must be non-negative")
        if max(dropout, attention_dropout, head_dropout) >= 1:
            raise ValueError("dropout values must be less than one")
        self.encoder = SharedChannelPatchEncoder(
            int(input_spec.window), patch_len, stride, d_model, n_heads,
            n_layers, d_ff, dropout, attention_dropout, pad_end)
        self.velocity_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(head_dropout),
            nn.Linear(6 * d_model * self.encoder.patch_count, input_spec.dims),
        )
        self.model_cfg = {
            "patch_len": patch_len, "stride": stride, "d_model": d_model,
            "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff,
            "dropout": dropout, "attention_dropout": attention_dropout,
            "head_dropout": head_dropout, "pad_end": pad_end,
            "pretrained": False, "encoder_channel_independent": True,
            "task_adapter": self.task_adapter,
            "causality": self.causality,
            "encoder_output_layout": "B,C,D,N",
            "head_flatten_order": "C,D,N",
        }

    @property
    def patch_count(self) -> int:
        return self.encoder.patch_count

    def encode_channels(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        return self.encoder(imu)

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_channels(imu))}

