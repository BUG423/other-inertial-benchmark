"""Independent IMU candidate inspired by CVPR2026 PFGNet/Spectral Scalpel.

No upstream adapter imports, public registration, pretrained weights or
cross-window state. CPU qualification is separate from the existing ten-model
tail queue. This file alone does not establish accuracy or novelty.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DCPreservingSpectralGate(nn.Module):
    """Positive real spectral gain; DC exactly one, identity at initialization.

    DC preservation holds for these latent features, not physical velocity.
    The rFFT uses only the current observed window; it is window-noncausal.
    """
    def __init__(self, channels, bands=8, max_change=.25):
        super().__init__()
        if not 0 < max_change < 1 or bands < 2:
            raise ValueError('Bounded positive spectral gain required')
        self.max_change = max_change
        self.static = nn.Parameter(torch.zeros(channels, bands))
        self.condition = nn.Linear(channels, channels * bands)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, x):
        b, c, t = x.shape
        cues = (x[..., 1:] - x[..., :-1]).square().mean(-1).log1p()
        band_logits = self.static[None] + self.condition(cues).reshape(b, c, -1)
        logits = F.interpolate(band_logits, size=t // 2 + 1, mode='linear', align_corners=True)
        gain = 1 + self.max_change * logits.tanh()
        gain = torch.cat((torch.ones_like(gain[..., :1]), gain[..., 1:]), -1)
        return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * gain, n=t, dim=-1)


class CrossSensorLagFusion(nn.Module):
    """Within-window gyro/acc latent products at fixed lags; no circular wrap."""
    def __init__(self, half_width):
        super().__init__()
        self.lags = (-2, 0, 2)
        self.route = nn.Linear(3 * half_width, 3)
        self.project = nn.Conv1d(half_width, 2 * half_width, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, gyro, accel):
        g = gyro - gyro.mean(-1, keepdim=True)
        a = accel - accel.mean(-1, keepdim=True)
        g = g / (g.square().mean(-1, keepdim=True) + 1e-4).sqrt()
        a = a / (a.square().mean(-1, keepdim=True) + 1e-4).sqrt()
        products = []
        for lag in self.lags:
            if lag > 0:
                shifted = F.pad(a[..., :-lag], (lag, 0))
            elif lag < 0:
                shifted = F.pad(a[..., -lag:], (0, -lag))
            else:
                shifted = a
            products.append(g * shifted)
        weights = self.route(torch.cat([p.mean(-1) for p in products], -1)).softmax(-1)
        mixed = sum(p * weights[:, i, None, None] for i, p in enumerate(products))
        return self.project(mixed)


class TemporalPeripheralBlock(nn.Module):
    """Compact temporal center/surround mixing, inspired by PFGNet.

    This is a separately declared recipe, not the source-faithful PFG adapter.
    """
    def __init__(self, width):
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.branches = nn.ModuleList(nn.Conv1d(width, width, k, padding=k // 2,
                                               groups=width) for k in (5, 15, 31))
        self.center = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.suppression = nn.Parameter(torch.zeros(3))
        self.route = nn.Conv1d(3, 3, 1)
        self.mlp = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(),
                                 nn.Conv1d(2 * width, width, 1))
        self.scale = nn.Parameter(torch.full((1, width, 1), .01))

    def forward(self, x):
        z = self.norm(x)
        diff = F.pad(z[..., 1:] - z[..., :-1], (1, 0))
        lap = F.pad(diff[..., 1:] - diff[..., :-1], (1, 0))
        cues = torch.cat((diff.abs().mean(1, keepdim=True), lap.abs().mean(1, keepdim=True),
                          z.square().mean(1, keepdim=True)), 1)
        weights = self.route(cues).softmax(1)
        center = self.center(z)
        mixed = sum((branch(z) - self.suppression[i].tanh() * center) * weights[:, i:i + 1]
                    for i, branch in enumerate(self.branches))
        return x + self.scale * self.mlp(mixed)


class MomentResidualReadout(nn.Module):
    """Signed vector correction from temporal spread; no mandatory expansion.

    A zero residual starts at the matched GAP+Linear control. A direction/speed
    product is deliberately avoided: independent speed and gain are redundant.
    """
    def __init__(self, width, temporal_moments=True):
        super().__init__()
        self.temporal_moments = temporal_moments
        self.base = nn.Linear(width, 3)
        self.residual = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 3))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x, enabled=True):
        mean = x.mean(-1)
        if not enabled:
            return self.base(mean)
        if self.temporal_moments:
            spread = ((x - mean[..., None]).square().mean(-1) + 1e-6).sqrt()
            change = ((x[..., 1:] - x[..., :-1]).square().mean(-1) + 1e-6).sqrt()
        else:
            # Equal parameter count and MLP depth; only GAP features supplied.
            spread = change = mean
        return self.base(mean) + self.residual(torch.cat((mean, spread, change), -1))


class CoherentSpectralIO(nn.Module):
    """B,6,200 gyro/acc -> B,3 mean velocity; no labels in forward.

    Independent ablations share width/depth/input/loss and initial base weights.
    A real-data adapter must enforce reference_oracle versus deployable tracks.
    """
    def __init__(self, width=48, depth=3, spectral=True, cross_sensor=True, moments=True, temporal_moments=True):
        super().__init__()
        if width < 8 or width % 2 or depth < 1:
            raise ValueError('Even width >=8 and positive depth required')
        self.spectral_enabled, self.cross_enabled, self.moments_enabled = spectral, cross_sensor, moments
        self.gyro_stem = nn.Conv1d(3, width // 2, 3, padding=1)
        self.accel_stem = nn.Conv1d(3, width // 2, 3, padding=1)
        self.cross = CrossSensorLagFusion(width // 2)
        self.blocks = nn.Sequential(*(TemporalPeripheralBlock(width) for _ in range(depth)))
        self.spectral = DCPreservingSpectralGate(width)
        self.readout = MomentResidualReadout(width, temporal_moments)

    def forward(self, imu):
        if imu.ndim != 3 or tuple(imu.shape[1:]) != (6, 200):
            raise ValueError('Expected B,6,200, ordered gyro_xyz/acc_xyz at 200Hz')
        gyro, accel = self.gyro_stem(imu[:, :3]), self.accel_stem(imu[:, 3:])
        x = torch.cat((gyro, accel), 1)
        if self.cross_enabled:
            x = x + self.cross(gyro, accel)
        if self.spectral_enabled:
            x = self.spectral(x)
        # Place the gate BEFORE nonlinear temporal mixing. Immediately before
        # GAP a DC-preserving filter would have no effect on the linear control.
        x = self.blocks(x)
        return {'vel': self.readout(x, self.moments_enabled)}
