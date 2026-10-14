"""Unregistered ModernTCN architecture port for window-average IMU velocity.

Backbone adapted from luodhhh/ModernTCN at
56a9a2c018385cd5acef015378cae7f084d1b11c (MIT, copyright 2024 luodhhh).
The accompanying LICENSE.upstream preserves that notice.  The official
classification feature path is retained; pooling and velocity supervision are
benchmark adaptations.  This file has no registry or training side effects.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from inertial_benchmark.data.views import CHANNELS
from inertial_benchmark.nn.base import BaseModel, InputSpec


class LargeKernelDepthwise(nn.Module):
    """Unmerged large/small depthwise Conv-BN branches used during training.

Structural reparameterization/export is deliberately outside this PoC.  Both
branches use zero padding, bias-free convolutions, and their own BatchNorm,
as in the upstream training graph.
    """

    def __init__(self, channels: int, large_kernel: int,
                 small_kernel: int | None) -> None:
        super().__init__()
        if large_kernel < 1 or large_kernel % 2 != 1:
            raise ValueError("large_kernel must be a positive odd integer")
        if small_kernel is not None and (
                small_kernel < 1 or small_kernel % 2 != 1
                or small_kernel > large_kernel):
            raise ValueError("small_kernel must be odd, positive and <= large_kernel")
        self.large_kernel = large_kernel
        self.small_kernel = small_kernel

        def branch(kernel: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(channels, channels, kernel, padding=kernel // 2,
                          groups=channels, bias=False),
                nn.BatchNorm1d(channels),
            )

        self.large_branch = branch(large_kernel)
        self.small_branch = branch(small_kernel) if small_kernel is not None else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        result = self.large_branch(features)
        if self.small_branch is not None:
            result = result + self.small_branch(features)
        return result


class ModernTCNBlock(nn.Module):
    """Time mixing -> feature mixing -> variable mixing -> one residual.

The semantic layout is always ``(B,M,D,N)``: variables M, features D,
patches N.  Feature FFN groups are M; after permutation, variable FFN
groups are D.  Removing that permutation would change the architecture.
    """

    def __init__(self, variables: int, width: int, ffn_ratio: int,
                 large_kernel: int, small_kernel: int | None,
                 dropout: float) -> None:
        super().__init__()
        self.variables = variables
        self.width = width
        self.dw = LargeKernelDepthwise(variables * width, large_kernel, small_kernel)
        # Shared feature normalization folds the variables into the batch,
        # exactly as the official classifier's Block.forward does.
        self.norm = nn.BatchNorm1d(width)
        hidden_width = width * ffn_ratio

        def feedforward(groups: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(variables * width, variables * hidden_width, 1, groups=groups),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Conv1d(variables * hidden_width, variables * width, 1, groups=groups),
                nn.Dropout(dropout),
            )

        self.feature_ffn = feedforward(variables)
        self.variable_ffn = feedforward(width)

    def mix_features(self, features: torch.Tensor) -> torch.Tensor:
        batch, variables, width, patches = features.shape
        return self.feature_ffn(features.reshape(batch, variables * width, patches)).reshape(
            batch, variables, width, patches)

    def mix_variables(self, features: torch.Tensor) -> torch.Tensor:
        batch, variables, width, patches = features.shape
        permuted = features.permute(0, 2, 1, 3).reshape(batch, width * variables, patches)
        mixed = self.variable_ffn(permuted).reshape(batch, width, variables, patches)
        return mixed.permute(0, 2, 1, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, variables, width, patches = features.shape
        mixed = self.dw(features.reshape(batch, variables * width, patches))
        mixed = self.norm(mixed.reshape(batch * variables, width, patches))
        mixed = mixed.reshape(batch, variables, width, patches)
        mixed = self.mix_features(mixed)
        mixed = self.mix_variables(mixed)
        return features + mixed


class ModernTCNVelocity(BaseModel):
    """ModernTCN classification backbone with a deterministic linear velocity head.

Defaults anchor the released EthanolConcentration classification backbone:
patch 8/stride 4, two stages 32/64 with one block each, kernels 13/5,
FFN ratio 1, downsample 2 and backbone dropout 0.1.  These settings are a
traceable starting point, not an official IMU recipe.  The original class
GELU/dropout/flatten head becomes temporal mean -> ordered variable/feature
flatten -> Linear(6 * final_width, D), and shared benchmark MSE.

Window length can vary for contract tests or a separately identified profile;
the supplied drafts freeze T=200 at 200 Hz.  Symmetric temporal padding and
window pooling make this a window-noncausal model, not a streaming model.
    """

    default_loss = "mse"
    training_initialization = "scratch_supervised"
    causality = "window_noncausal"

    def __init__(self, input_spec: InputSpec, patch_size: int = 8,
                 patch_stride: int = 4, widths: Sequence[int] = (32, 64),
                 depths: Sequence[int] = (1, 1),
                 large_kernels: Sequence[int] = (13, 13),
                 small_kernels: Sequence[int | None] = (5, 5),
                 ffn_ratio: int = 1, downsample_ratio: int = 2,
                 dropout: float = 0.1, pretrained: bool = False) -> None:
        super().__init__(input_spec)
        if (tuple(input_spec.channels) != CHANNELS or input_spec.dims not in (2, 3)
                or input_spec.sub_windows or input_spec.extra_inputs
                or input_spec.target != "avg_velocity"
                or input_spec.frame != "gravity_world"
                or input_spec.orientation != "reference"
                or input_spec.rate != 200.0 or input_spec.remove_gravity
                or input_spec.output_steps):
            raise ValueError(
                "ModernTCNVelocity requires ordered [gyro_xyz, acc_xyz], gravity_world "
                "average velocity with reference orientation at 200 Hz, dims 2 or 3, no history/extra inputs "
                "and gravity retained")
        if pretrained:
            raise ValueError("the staging profile is supervised from scratch")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if (type(patch_size) is not int or type(patch_stride) is not int
                or not 1 <= patch_stride <= patch_size <= int(input_spec.window)):
            raise ValueError("patch_size/patch_stride must be integers with 1 <= stride <= size <= window")
        if type(ffn_ratio) is not int or ffn_ratio < 1:
            raise ValueError("ffn_ratio must be a positive integer")
        # At ratio two the official tail-copy padding is precisely repeat-last.
        # Other ratios need their own reviewed boundary behavior/profile.
        if type(downsample_ratio) is not int or downsample_ratio != 2:
            raise ValueError("the staging source contract fixes downsample_ratio=2")
        if not widths or not (len(widths) == len(depths) == len(large_kernels) == len(small_kernels)):
            raise ValueError("widths, depths and both kernel lists must have equal nonzero length")
        if any(type(value) is not int or value < 1 for value in (*widths, *depths, *large_kernels)):
            raise ValueError("stage widths, depths and large kernels must be positive integers")
        if any(value is not None and type(value) is not int for value in small_kernels):
            raise ValueError("small kernels must be integers or None")
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio
        self.widths = tuple(widths)
        self.depths = tuple(depths)
        self.patch_count = int(input_spec.window) // patch_stride
        self.stage_patch_counts = [self.patch_count]
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, widths[0], patch_size, patch_stride),
                          nn.BatchNorm1d(widths[0]))
        ])
        for before, after in zip(widths[:-1], widths[1:]):
            self.downsample_layers.append(nn.Sequential(
                nn.BatchNorm1d(before),
                nn.Conv1d(before, after, downsample_ratio, downsample_ratio),
            ))
            self.stage_patch_counts.append((self.stage_patch_counts[-1] + 1) // 2)
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                ModernTCNBlock(6, width, ffn_ratio, large_kernel, small_kernel, dropout)
                for _ in range(depth)
            ])
            for width, depth, large_kernel, small_kernel
            in zip(widths, depths, large_kernels, small_kernels)
        ])
        self.velocity_head = nn.Linear(6 * widths[-1], input_spec.dims)
        args = {
            "patch_size": patch_size, "patch_stride": patch_stride,
            "widths": list(widths), "depths": list(depths),
            "large_kernels": list(large_kernels), "small_kernels": list(small_kernels),
            "ffn_ratio": ffn_ratio, "downsample_ratio": downsample_ratio,
            "dropout": dropout, "pretrained": False,
        }
        self.model_cfg = {
            "name": f"moderntcn_velocity_{input_spec.dims}d_staging",
            "arch": "moderntcn_velocity", "args": args,
            "input": input_spec.to_dict(), "registered": False,
            "implementation_kind": "architecture_port",
        }

    def encode(self, imu: torch.Tensor) -> torch.Tensor:
        """Return the final ``(B,6,width,patches)`` backbone representation."""
        self.check_input(imu)
        features = imu.unsqueeze(2)
        for index, (downsample, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            batch, variables, width, patches = features.shape
            flattened = features.reshape(batch * variables, width, patches)
            if index == 0 and self.patch_size != self.patch_stride:
                flattened = F.pad(flattened, (0, self.patch_size - self.patch_stride), mode="replicate")
            elif index and patches % 2:
                flattened = torch.cat([flattened, flattened[:, :, -1:]], dim=-1)
            flattened = downsample(flattened)
            features = flattened.reshape(batch, variables, flattened.shape[1], flattened.shape[2])
            features = stage(features)
        return features

    def forward(self, imu: torch.Tensor) -> dict:
        encoded = self.encode(imu)
        # Ordered pooling preserves gyro/accelerometer and axis identities;
        # there is no zero-filled Z or mean over the six physical variables.
        pooled = encoded.mean(dim=-1).flatten(start_dim=1)
        return {"vel": self.velocity_head(pooled)}
