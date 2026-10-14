"""Unregistered CSaN-ViT temporal architecture port; not a paper reproduction.

Source: SudongCAI/CSaN@a07175ffb9fa82ec58eca397e80aa63027c5d73f,
MODELS/vit_csan.py, LAST Attention definition (lines 297-390), Apache-2.0.
The earlier same-name tanh/GELU implementation is deliberately NOT selected.
Changes: six-axis token projection, 1D positions, temporal mean, one XYZ head;
the width/depth recipe is an unmeasured adapter hypothesis. See provenance.json.
No registry mutation, pretrained loading, custom kernels, or persistent state.
Runtime tests may run only on the server, after the original-task tail barrier.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from inertial_benchmark.data.views import CHANNELS
from inertial_benchmark.nn.base import BaseModel, InputSpec


def _canonical_spec(spec: InputSpec) -> None:
    if (tuple(spec.channels) != CHANNELS or spec.window != 200 or spec.rate != 200
            or spec.dims != 3 or spec.frame != "gravity_world"
            or spec.orientation != "reference" or spec.remove_gravity
            or spec.target != "avg_velocity" or spec.output_layout != "window"
            or spec.output_steps or spec.overlap != "mean" or spec.history or spec.history_stride
            or spec.extra_inputs):
        raise ValueError("requires canonical ordered six-axis T200@200Hz reference "
                         "gravity_world retained-gravity XYZ avg_velocity, no extras/history")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _dropout(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("dropout must be finite in [0, 1)")


def _source_linear_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SoftGShift(nn.Module):
    """Source compensator activation, x * sigmoid(Linear(x))."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.apply(_source_linear_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.proj(x))


class CSaNAttention(nn.Module):
    """Last source Attention: row temperature, column magnitude, private budget.

    Calibration is linear (+1 for ka/temp/magk), NOT bounded with tanh.
    Attention remains quadratic in tokens. No clamp is added to its gains.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        _positive_int("dim", dim)
        _positive_int("num_heads", num_heads)
        _dropout(dropout)
        if dim % (4 * num_heads):
            raise ValueError("dim must be divisible by 4*num_heads for source u reshape")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        hidden_dim = max(dim // 8, 24)
        self.u = nn.Linear(dim, dim // 4)
        self.reduce = nn.Linear(dim // 4, hidden_dim)
        self.act = SoftGShift(hidden_dim)
        self.ka = nn.Linear(hidden_dim, dim)
        self.kv = nn.Linear(hidden_dim, dim)
        self.temp = nn.Linear(hidden_dim, num_heads)
        self.magk = nn.Linear(hidden_dim, num_heads)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.apply(_source_linear_init)

    @staticmethod
    def quasi_linear(x: torch.Tensor, eta: float = 0.25) -> torch.Tensor:
        return torch.where(x > eta, x, eta * torch.exp(x.clamp(max=eta) / eta - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        scores = (q * self.scale) @ k.transpose(-2, -1)
        row_importance = self.quasi_linear(scores).mean(dim=-1, keepdim=True)
        u = self.u(x).reshape(batch, tokens, self.num_heads, -1).transpose(1, 2)
        importance = (row_importance * u.sigmoid()).transpose(1, 2).reshape(batch, tokens, -1)
        compensator = self.act(self.reduce(importance))
        ka = self.ka(compensator).reshape(batch, tokens, self.num_heads, self.head_dim)
        ka = ka.transpose(1, 2) + 1.0
        kv = self.kv(compensator).reshape(batch, tokens, self.num_heads, self.head_dim)
        kv = kv.transpose(1, 2)
        row_temperature = self.temp(compensator).transpose(1, 2).unsqueeze(-1) + 1.0
        column_magnitude = self.magk(compensator).transpose(1, 2).unsqueeze(-2) + 1.0
        attention = (scores * row_temperature).softmax(dim=-1) * column_magnitude
        attention = self.attn_drop(attention)
        y = (attention @ v) * ka + v * kv
        y = y.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(y))


class CSaNBlock(nn.Module):
    """Ordinary source pre-LN Block; LayerScale/DropPath disabled as declared."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CSaNAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CSaNVelocity(BaseModel):
    """B,6,200 -> B,3; independently reset, scratch-supervised MSE adapter."""

    default_loss = "mse"
    training_initialization = "scratch_supervised"
    implementation_kind = "architecture_port"
    source_architecture = "csan_vit_last_attention_softgshift_linear_calibration"
    inference_contract = "window_noncausal"

    def __init__(self, input_spec: InputSpec, d_model: int = 96, n_layers: int = 4,
                 num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 pretrained: bool = False) -> None:
        _canonical_spec(input_spec)
        if pretrained is not False:
            raise ValueError("only pretrained=False (supervised from scratch) is permitted")
        _positive_int("d_model", d_model)
        _positive_int("n_layers", n_layers)
        _positive_int("num_heads", num_heads)
        _dropout(dropout)
        if (isinstance(mlp_ratio, bool) or not math.isfinite(mlp_ratio)
                or mlp_ratio <= 0 or int(d_model * mlp_ratio) < 1):
            raise ValueError("mlp_ratio must give a positive finite hidden width")
        if d_model % (4 * num_heads):
            raise ValueError("d_model must be divisible by 4*num_heads")
        super().__init__(input_spec)
        self.model_cfg = dict(d_model=d_model, n_layers=n_layers, num_heads=num_heads,
                              mlp_ratio=mlp_ratio, dropout=dropout, pretrained=False)
        self.token_projection = nn.Linear(6, d_model)
        self.position = nn.Parameter(torch.empty(1, 200, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.position_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(CSaNBlock(d_model, num_heads, mlp_ratio, dropout)
                                    for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.velocity_head = nn.Linear(d_model, 3)
        self.apply(_source_linear_init)

    def set_loss(self, name: str, **kwargs: object) -> None:
        if name != "mse" or kwargs:
            raise ValueError("this staged scientific profile permits plain MSE only")
        super().set_loss(name)

    def encode_window(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        x = self.position_drop(self.token_projection(imu.transpose(1, 2)) + self.position)
        for block in self.blocks:
            x = block(x)
        return self.norm(x).mean(dim=1)

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu))}
