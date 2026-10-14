"""Unregistered TESSERA-v1 dual-stream architecture velocity baseline.

ucam-eo/tessera@e368539f7c4711e5f16fdd09801573830e34baf9, tessera_infer v1.
Preserves separate temporal encoders, GRU-conditioned attention pooling and
ordered concatenation/single reducer from the released MIT code. Scratch MSE
only: BT/projector/views/mix/geospatial/calendar inputs are intentionally absent.
This is NOT the complete TESSERA paper, SSL reproduction, v2, or a weight port.
"""

from __future__ import annotations

import math

import torch
from torch import nn

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
        raise ValueError("TESSERA-v1 requires ordered gyro_xyz/acc_xyz, T200/200Hz, "
                         "XYZ average velocity, gravity_world/reference, retained gravity "
                         "and no history/extra inputs")


class RelativeWindowPosition(nn.Module):
    """Source sinusoidal encoding, with internally derived sample-index position.

    The calendar DOY operand is replaced by q=t_relative_seconds*200=0..199.
    This is a declared dimensionless task adaptation, not an extra clock input,
    acquired date, learned external feature, or temporal view augmentation.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        dtype = torch.float64 if tokens.dtype == torch.float64 else torch.float32
        positions = torch.arange(tokens.shape[1], device=tokens.device, dtype=dtype)
        frequencies = torch.exp(torch.arange(0, self.width, 2, device=tokens.device, dtype=dtype)
                                * (-math.log(10000.0) / self.width))
        angles = positions[:, None] * frequencies[None, :]
        position = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        return position.to(tokens.dtype).unsqueeze(0)


class GRUConditionedAttentionPool(nn.Module):
    """Weights depend on GRU context; pooled VALUES are the original tokens."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.temporal_context = nn.GRU(width, width, batch_first=True)
        self.query = nn.Linear(width, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        context, _ = self.temporal_context(tokens)  # implicit zero hidden state on every call
        weights = self.query(context).softmax(dim=1)
        return (weights * tokens).sum(dim=1)


class SensorTemporalEncoder(nn.Module):
    def __init__(self, width: int, heads: int, layers: int,
                 feedforward_width: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(3, width), nn.ReLU(), nn.Linear(width, width))
        self.position = RelativeWindowPosition(width)
        layer = nn.TransformerEncoderLayer(width, heads, feedforward_width,
                                           dropout, activation="relu", batch_first=True,
                                           norm_first=False)
        # Retain source TransformerEncoder cloning/initialization; no made-up
        # per-layer reinitialization, causal mask, shared modality weights or CLS.
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool = GRUConditionedAttentionPool(width)

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(sensor)
        return self.pool(self.transformer(tokens + self.position(tokens)))


class TesseraV1DualStreamVelocity(BaseModel):
    default_loss = "mse"
    implementation_kind = "dualstream_architecture_core_port"
    causality = "window_noncausal"
    source_commit = "e368539f7c4711e5f16fdd09801573830e34baf9"
    training_initialization = "scratch_supervised"
    ssl_objectives = "omitted"

    def __init__(self, input_spec: InputSpec, width: int = 512, layers: int = 8,
                 heads: int = 8, feedforward_width: int = 4096,
                 latent_dim: int = 128, dropout: float = 0.1,
                 pretrained: bool = False) -> None:
        super().__init__(input_spec)
        _validate_spec(input_spec)
        if pretrained is not False:
            raise ValueError("this architecture identity is scratch MSE; no upstream weights/SSL")
        for value, name in ((width, "width"), (layers, "layers"), (heads, "heads"),
                            (feedforward_width, "feedforward_width"), (latent_dim, "latent_dim")):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if width % heads or width % 2 or width != 4 * latent_dim:
            raise ValueError("source width must be even, divisible by heads and4*latent_dim")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        self.gyro_encoder = SensorTemporalEncoder(width, heads, layers, feedforward_width, dropout)
        self.acc_encoder = SensorTemporalEncoder(width, heads, layers, feedforward_width, dropout)
        self.reducer = nn.Linear(2 * width, latent_dim)
        self.velocity_head = nn.Linear(latent_dim, 3)
        self.model_cfg = dict(name="tessera_v1_dualstream_architecture_velocity_3d_staging",
                              arch="tessera_v1_dualstream_architecture_velocity", loss="mse",
                              args=dict(width=width, layers=layers, heads=heads,
                                        feedforward_width=feedforward_width, latent_dim=latent_dim,
                                        dropout=dropout, pretrained=False), registered=False,
                              causality=self.causality, implementation_kind=self.implementation_kind,
                              ssl_objectives=self.ssl_objectives,
                              fusion_order="gyro_features,acc_features",
                              source_configuration="v1_code512wide_8layers_ff4096_not_paper4layers")

    def set_loss(self, name: str, **kwargs) -> None:
        if name != "mse" or kwargs:
            raise ValueError("this architecture identity is MSE-only; SSL requires a separate identity")
        super().set_loss(name)

    def encode_streams(self, imu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.check_input(imu)
        return (self.gyro_encoder(imu[:, :3].transpose(1, 2)),
                self.acc_encoder(imu[:, 3:].transpose(1, 2)))

    def encode_window(self, imu: torch.Tensor) -> torch.Tensor:
        gyro, acc = self.encode_streams(imu)
        return self.reducer(torch.cat((gyro, acc), dim=-1))

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu))}
