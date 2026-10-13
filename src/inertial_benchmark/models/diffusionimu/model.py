"""DiffusionIMU paper-equation reconstruction; assumptions are recorded in README.md.

The paper's method-specific Eqs.12--22, rather than its background DDPM Eqs.6--9,
define this draft. Unspecified encoder attention layout is explicitly assumed.
Softplus u is a regularized positive head, NOT a Gaussian standard deviation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from inertial_benchmark.nn.base import BaseModel, InputSpec, check_loss_batch
from inertial_benchmark.nn.registry import register_model
from inertial_benchmark.nn.losses import masked_output_mean, register_loss


class LocalGlobalAttentionBlock(nn.Module):
    """Attention bottleneck draft replacing local convolution, then global attention."""

    def __init__(self, channels: int, width: int, stride: int,
                 heads: int = 4, local_radius: int = 3, dropout: float = .2) -> None:
        super().__init__()
        inner = max(heads, width // 4)
        if inner % heads:
            raise ValueError("bottleneck width must divide into attention heads")
        self.input_projection = nn.Conv1d(channels, inner, 1, bias=False)
        self.input_norm = nn.BatchNorm1d(inner)
        self.local_attention = nn.MultiheadAttention(inner, heads, dropout=dropout, batch_first=True)
        self.global_attention = nn.MultiheadAttention(inner, heads, dropout=dropout, batch_first=True)
        self.local_norm = nn.LayerNorm(inner)
        self.global_norm = nn.LayerNorm(inner)
        self.output_projection = nn.Conv1d(inner, width, 1, stride=stride, bias=False)
        self.output_norm = nn.BatchNorm1d(width)
        self.skip = (nn.Identity() if channels == width and stride == 1 else
                     nn.Sequential(nn.Conv1d(channels, width, 1, stride=stride, bias=False),
                                   nn.BatchNorm1d(width)))
        self.radius = int(local_radius)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = F.relu(self.input_norm(self.input_projection(x))).transpose(1, 2)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        excluded = (positions[:, None] - positions[None, :]).abs() > self.radius
        local, _ = self.local_attention(tokens, tokens, tokens, attn_mask=excluded, need_weights=False)
        tokens = self.local_norm(tokens + local)
        global_features, _ = self.global_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.global_norm(tokens + global_features)
        features = self.output_norm(self.output_projection(tokens.transpose(1, 2)))
        return F.relu(features + self.skip(x))


@register_loss("diffusionimu_multitask")
class DiffusionIMUMultiTaskLoss:
    """Method Eq.22: vector squared error + positive-head L2/2 + alignment L2.

    This does not train a likelihood or noise-prediction objective. Changing it to
    NLL/DDPM would create a distinct ablation, rather than faithfully filling a gap.
    """

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: torch.Tensor | None = None) -> tuple:
        # Eq.21 is interpreted as mean over samples of squared vector error;
        # component-mean scaling would change relative multi-task weights by D.
        velocity = masked_output_mean((out["vel"] - target).square().sum(-1), mask)
        uncertainty = masked_output_mean(.5 * out["uncertainty_regularizer_head"].square().sum(-1), mask)
        alignment = masked_output_mean(out["alignment_squared_norm"], mask)
        total = velocity + uncertainty + alignment
        return total, {"mse_sum": velocity.detach(), "uncertainty_regularization": uncertainty.detach(),
                       "feature_alignment": alignment.detach()}


@register_model("diffusionimu")
class DiffusionIMU(BaseModel):
    """Attention encoder, adaptive feature noise, alignment and K-step decoder."""

    default_loss = "diffusionimu_multitask"
    saved_outputs = ("uncertainty_regularizer_head", "diffusion_timestep", "alignment_squared_norm")

    def __init__(self, input_spec: InputSpec, hidden: int = 128, decoder_steps: int = 3,
                 diffusion_timesteps: int = 100, dropout: float = .2,
                 attention_heads: int = 4, local_radius: int = 3,
                 stem_width: int = 64, initial_alpha: float = .5,
                 evaluation_timestep: int = 0) -> None:
        super().__init__(input_spec)
        if (input_spec.dims not in (2, 3) or input_spec.num_channels != 6 or
                input_spec.sub_windows or input_spec.output_layout != "window" or
                input_spec.target != "avg_velocity" or input_spec.frame != "gravity_world"):
            raise ValueError("model requires six-axis gravity_world window average velocity, dims 2/3")
        if decoder_steps < 1 or diffusion_timesteps < 1 or hidden < 1 or local_radius < 0:
            raise ValueError("invalid decoder/time/hidden/local attention size")
        if not 0 <= evaluation_timestep < diffusion_timesteps or not 0 < initial_alpha < 1:
            raise ValueError("invalid evaluation timestep or initial noise coefficient")
        self.stem = nn.Sequential(nn.Conv1d(6, stem_width, 7, 2, 3, bias=False),
                                  nn.BatchNorm1d(stem_width), nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        channels = stem_width
        blocks = []
        for stage in range(4):
            width = stem_width * 2 ** stage
            for index in range(2):
                stride = 2 if stage > 0 and index == 0 else 1
                blocks.append(LocalGlobalAttentionBlock(channels, width, stride,
                              attention_heads, local_radius, dropout))
                channels = width
        self.encoder = nn.Sequential(*blocks)
        self.feature_projection = nn.Conv1d(channels, hidden, 1)
        # A sigmoid coefficient is an explicit bounded-mixture assumption.
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(initial_alpha)))
        self.alignment_projection = nn.Linear(hidden, hidden, bias=False)
        self.diffusion_encoder = nn.Linear(hidden, hidden, bias=False)
        self.time_embedding = nn.Embedding(diffusion_timesteps, hidden)
        self.refinement = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(decoder_steps)])
        self.velocity_head = nn.Linear(hidden, input_spec.dims, bias=False)
        self.uncertainty_head = nn.Linear(hidden, input_spec.dims, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.diffusion_timesteps = int(diffusion_timesteps)
        self.evaluation_timestep = int(evaluation_timestep)

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        # Paper lists acceleration before angular velocity; IPB retains its own layout externally.
        h = self.feature_projection(self.encoder(self.stem(imu[:, [3, 4, 5, 0, 1, 2]])))
        h = h.transpose(1, 2)
        noise = torch.randn_like(h)  # retained at evaluation; no silent deterministic zero-noise ablation
        alpha = self.alpha_logit.sigmoid()
        modulated = alpha * h + (1 - alpha) * noise
        alignment = (self.alignment_projection(h) - modulated).square().sum((1, 2))
        if self.training:
            timestep = torch.randint(self.diffusion_timesteps, (len(imu),), device=imu.device)
        else:
            # The paper does not specify eval t: record this policy and noise seed in inference identity.
            timestep = torch.full((len(imu),), self.evaluation_timestep,
                                  dtype=torch.long, device=imu.device)
        latent = F.relu(self.diffusion_encoder(modulated) + self.time_embedding(timestep)[:, None, :])
        for layer in self.refinement:
            latent = F.relu(layer(latent))
        pooled = self.dropout(latent.mean(1))
        return {"vel": self.velocity_head(pooled),
                "uncertainty_regularizer_head": F.softplus(self.uncertainty_head(pooled)),
                "alignment_squared_norm": alignment, "diffusion_timestep": timestep}

    def set_loss(self, name: str, **kwargs) -> None:
        # Keep the paper objective fixed; likelihood/noise objectives require a named ablation.
        if name != self.default_loss or kwargs:
            raise ValueError("DiffusionIMU preserves its paper multi-task loss; overrides require a named ablation")
        self.loss_name, self.loss_kwargs = name, {}

    def loss(self, out: dict, batch: dict, epoch: int = 0) -> tuple:
        check_loss_batch(batch)
        if self.loss_name != self.default_loss:
            raise ValueError("scientific loss identity does not match DiffusionIMU")
        return DiffusionIMUMultiTaskLoss()(out, batch["target"], epoch, batch.get("mask"))
