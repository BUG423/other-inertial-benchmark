"""Official-source-derived AI-IMU-DR noise adapter plus 21D filter.

Unregistered scientific staging. This is a vehicle filter, not a renamed
velocity regressor. See source-card.md for source/release discrepancies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .filter import State, Trajectory, run_filter


@dataclass(frozen=True)
class PhysicalParameters:
    gravity: tuple[float, float, float]
    process_base: tuple[float, float, float, float, float, float]
    initial_base: tuple[float, float, float, float, float, float]
    measurement_base: tuple[float, float]

    @classmethod
    def released_kitti(cls) -> "PhysicalParameters":
        """Pinned main_kitti.py::KITTIParameters facts, not universal defaults."""
        return cls((0., 0., -9.80655), (2e-4, 1e-3, 1e-8, 1e-6, 1e-8, 1e-8),
                   (1e-6, .1, 1e-8, 1e-3, 1e-5, .01), (1., 10.))


class InitProcessCovNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.factor_initial_covariance = nn.Linear(1, 6, bias=False).double()
        with torch.no_grad():
            self.factor_initial_covariance.weight.div_(10)
        self.factor_process_covariance = nn.Linear(1, 6, bias=False).double()
        with torch.no_grad():
            self.factor_process_covariance.weight.div_(10)

    def scales(self) -> tuple[Tensor, Tensor]:
        one = self.factor_initial_covariance.weight.new_ones(1)
        return (torch.pow(10., torch.tanh(self.factor_initial_covariance(one))),
                torch.pow(10., torch.tanh(self.factor_process_covariance(one))))


class MesNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cov_net = nn.Sequential(nn.Conv1d(6, 32, 5), nn.ReplicationPad1d(4),
                                     nn.ReLU(), nn.Dropout(.5),
                                     nn.Conv1d(32, 32, 5, dilation=3), nn.ReplicationPad1d(4),
                                     nn.ReLU(), nn.Dropout(.5)).double()
        self.cov_lin = nn.Sequential(nn.Linear(32, 2), nn.Tanh()).double()
        with torch.no_grad():
            self.cov_lin[0].bias.div_(100)
            self.cov_lin[0].weight.div_(100)

    def forward(self, normalized_body: Tensor, base_variances: Tensor) -> Tensor:
        if normalized_body.ndim != 3 or normalized_body.shape[1] != 6 or normalized_body.shape[2] < 9:
            raise ValueError("released CNN needs (B,6,T), T>=9")
        features = self.cov_net(normalized_body).transpose(1, 2)  # source B1 squeeze repaired
        logits = self.cov_lin(features)
        return base_variances * torch.pow(10., 3 * logits)


class AIIMUDR(nn.Module):
    """Body six-axis, original clock -> full world trajectory and 21D covariance.

    Initial R/v/p are explicit inputs, with their reference assistance disclosed
    by the caller. No later reference pose/velocity is an inference argument.
    mean/std must be fitted on declared training IMU only and saved with weights.
    """

    def __init__(self, mean: Tensor, std: Tensor, parameters: PhysicalParameters) -> None:
        super().__init__()
        if mean.shape != (6,) or std.shape != (6,) or not bool(torch.isfinite(mean).all()):
            raise ValueError("explicit six-axis train-only normalization required")
        if not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
            raise ValueError("normalization standard deviations must be finite positive")
        # Construction order matches released TORCHIEKF, including float32 RNG
        # initialization of layers before .double(), then source rescaling.
        self.initprocesscov_net = InitProcessCovNet()
        self.mes_net = MesNet()
        self.register_buffer("u_loc", mean.detach().clone().double())
        self.register_buffer("u_std", std.detach().clone().double())
        self.register_buffer("gravity", torch.tensor(parameters.gravity, dtype=torch.float64))
        self.register_buffer("process_base", torch.tensor(parameters.process_base, dtype=torch.float64))
        self.register_buffer("initial_base", torch.tensor(parameters.initial_base, dtype=torch.float64))
        self.register_buffer("measurement_base", torch.tensor(parameters.measurement_base, dtype=torch.float64))
        if self.process_base.shape != (6,) or self.initial_base.shape != (6,) or self.measurement_base.shape != (2,):
            raise ValueError("physics requires six Q groups, six P0 groups and two N variances")
        for name in ("process_base", "initial_base", "measurement_base"):
            value = getattr(self, name)
            if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
                raise ValueError(f"{name} must be finite positive")
        if self.gravity.shape != (3,) or not bool(torch.isfinite(self.gravity).all()):
            raise ValueError("explicit finite gravity required")

    def covariance_matrices(self, batch: int) -> tuple[Tensor, Tensor]:
        initial_scale, process_scale = self.initprocesscov_net.scales()
        groups = self.initial_base * initial_scale
        zero = groups.new_zeros(())
        # Released known yaw, all initial position, vertical velocity = zero
        # uncertainty; PSD, not SPD. This privilege is not repaired by jitter.
        diagonal = torch.stack((groups[0], groups[0], zero, groups[1], groups[1], zero,
                                zero, zero, zero, groups[2], groups[2], groups[2],
                                groups[3], groups[3], groups[3], groups[4], groups[4], groups[4],
                                groups[5], groups[5], groups[5]))
        P0 = torch.diag(diagonal).expand(batch, 21, 21)
        Q = torch.diag((self.process_base * process_scale).repeat_interleave(3)).expand(batch, 18, 18)
        return P0, Q

    def forward(self, timestamps_ns: Tensor, body_imu: Tensor, initial: State) -> Trajectory:
        if body_imu.ndim != 3 or body_imu.shape[-1] != 6 or body_imu.dtype != torch.float64:
            raise ValueError("float64 (B,T,6) gyro-then-specific-force input required")
        if body_imu.device != self.u_loc.device:
            raise ValueError("model/IMU device mismatch")
        normalized = ((body_imu - self.u_loc) / self.u_std).transpose(1, 2)
        variances = self.mes_net(normalized, self.measurement_base)
        P0, Q = self.covariance_matrices(body_imu.shape[0])
        return run_filter(timestamps_ns, body_imu, initial, P0, Q, variances, self.gravity)


def augment_released(body_imu: Tensor) -> Tensor:
    """Training-only caller hook: source per-sample noise plus constant bias.

    No implicit augmentation in forward/eval. B1 random draw order follows
    dataset.py::add_noise; B2 draws one bias per recording (new batch support).
    Units are standard deviations, not variances. Does not mutate caller data.
    """
    if body_imu.ndim != 3 or body_imu.shape[-1] != 6:
        raise ValueError("augmentation requires (B,T,6)")
    w = torch.randn_like(body_imu)
    bias = torch.randn_like(body_imu[:, 0])
    scale = body_imu.new_tensor((1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4))
    bias_scale = body_imu.new_tensor((1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4))
    return body_imu + w * scale + (bias * bias_scale)[:, None]
