"""MEIO-Net: actual covariance-controlled early exit, not all-head selection after compute.

Primary released training specification: SunXuehang/MEIO-Net/EarlyExitPDRModule.py.
The official repository omits NeuralNetwork/EarlyExit/EarlyExitDense.py. This
independent ResMLP reconstruction therefore declares head placement, head sharing
and the variance-threshold rule as assumptions, rather than claiming parameter
equivalence or a complete original reproduction. See docs/algorithms/meio.md.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.registry import register_model
from ..llio.model import Affine, PatchFlatten, PreAffinePostLayerScale
from .loss import MEIOMultiExitLoss  # noqa: F401: registers the scientific phase loss


class ExitHead(nn.Module):
    """Mean-pooling MLP with XYZ mean and one isotropic log-sigma (released [3,1])."""

    def __init__(self, feature_dim: int, dims: int, layers: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.extend([nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Dropout(dropout)])
        self.trunk = nn.Sequential(*blocks)
        self.mean = nn.Linear(feature_dim, dims)
        self.logstd = nn.Linear(feature_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple:
        z = self.trunk(tokens.mean(1))
        return self.mean(z), self.logstd(z)


@register_model("meio")
class MEIONet(BaseModel):
    """Joint multi-exit training; independently shrinking active inference batches."""

    default_loss = "meio_multiexit"
    saved_outputs = ("exit_depth",)

    def __init__(self, input_spec: InputSpec, patch_len: int = 10,
                 feature_dim: int = 512, layers: int = 6, expansion: int = 2,
                 dropout: float = .5, head_layers: int = 2, head_dropout: float = .5,
                 variance_threshold: float = 1e-3, min_logstd: float = None,
                 force_full_depth: bool = False) -> None:
        super().__init__(input_spec)
        if input_spec.output_layout != "window" or input_spec.target != "displacement":
            raise ValueError("MEIO-Net's phase loss is on window displacement")
        if input_spec.sub_windows:
            raise ValueError("MEIO-Net uses one contiguous window, not stacked history")
        if min(patch_len, feature_dim, layers, expansion) < 1 or head_layers < 0:
            raise ValueError("MEIO-Net sizes must be positive; head_layers >= 0")
        if input_spec.window % patch_len:
            raise ValueError("MEIO-Net input window must be divisible by patch_len")
        if not 0 <= dropout < 1 or not 0 <= head_dropout < 1:
            raise ValueError("MEIO-Net dropout must be in [0,1)")
        if not math.isfinite(variance_threshold) or variance_threshold < 0:
            raise ValueError("variance_threshold must be finite and nonnegative (m^2 per axis)")
        if min_logstd is not None and not math.isfinite(min_logstd):
            raise ValueError("min_logstd must be finite or null")
        self.variance_threshold, self.min_logstd = float(variance_threshold), min_logstd
        self.force_full_depth = bool(force_full_depth)
        tokens = input_spec.window // patch_len
        self.embedding = nn.Sequential(PatchFlatten(patch_len),
                                       nn.Linear(patch_len * input_spec.num_channels, feature_dim))
        self.blocks = nn.ModuleList()
        self.affines = nn.ModuleList()
        self.heads = nn.ModuleList()
        for _ in range(layers):
            token = PreAffinePostLayerScale(feature_dim, nn.Conv1d(tokens, tokens, 1, bias=False))
            channel = PreAffinePostLayerScale(feature_dim, nn.Sequential(
                nn.Linear(feature_dim, feature_dim * expansion, bias=False), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(feature_dim * expansion, feature_dim, bias=False)))
            self.blocks.append(nn.Sequential(token, channel))
            self.affines.append(Affine(feature_dim))
            self.heads.append(ExitHead(feature_dim, input_spec.dims, head_layers, head_dropout))

    def _head(self, depth: int, tokens: torch.Tensor) -> tuple:
        mean, sigma = self.heads[depth](self.affines[depth](tokens))
        if self.min_logstd is not None:
            sigma = sigma.clamp_min(self.min_logstd)
        return mean, sigma

    def forward_all(self, tokens: torch.Tensor) -> dict:
        means, sigmas = [], []
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            mean, sigma = self._head(i, tokens)
            means.append(mean)
            sigmas.append(sigma.expand_as(mean))
        all_mean, all_sigma = torch.stack(means, 1), torch.stack(sigmas, 1)
        return {"vel": all_mean[:, -1], "logstd": all_sigma[:, -1],
                "all_vel": all_mean, "all_logstd": all_sigma,
                "exit_depth": torch.full((len(tokens),), len(self.blocks), device=tokens.device,
                                         dtype=torch.int64)}

    def forward_early_exit(self, tokens: torch.Tensor) -> dict:
        batch, dims = len(tokens), self.input_spec.dims
        mean_out = tokens.new_zeros((batch, dims))
        sigma_out = tokens.new_zeros((batch, dims))
        depth_out = torch.zeros(batch, device=tokens.device, dtype=torch.int64)
        active = torch.arange(batch, device=tokens.device)
        for i, block in enumerate(self.blocks):
            if not len(active):
                break
            tokens = block(tokens)
            mean, sigma = self._head(i, tokens)
            # Covariance is predicted from IMU. Neither target motion nor labels decide the exit.
            accepted = torch.exp(2 * sigma[:, 0]) <= self.variance_threshold
            if i == len(self.blocks) - 1:
                accepted = torch.ones_like(accepted)
            ids = active[accepted]
            mean_out[ids] = mean[accepted]
            sigma_out[ids] = sigma[accepted].expand(-1, dims)
            depth_out[ids] = i + 1
            keep = ~accepted
            active, tokens = active[keep], tokens[keep]
        return {"vel": mean_out, "logstd": sigma_out, "exit_depth": depth_out}

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        tokens = self.embedding(imu)
        if self.training or self.force_full_depth:
            return self.forward_all(tokens)
        return self.forward_early_exit(tokens)
