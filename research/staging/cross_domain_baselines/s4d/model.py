"""Unregistered S4D-Lin/ZOH architecture port for window-average velocity.

Independent compact implementation of the diagonal SSM equations and residual
backbone in state-spaces/s4@e757cef, ``models/s4/s4d.py`` and ``example.py``.
Upstream is Apache-2.0. No upstream file, custom kernel or weights are vendored.
This velocity adapter is not an official inertial-navigation implementation.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from inertial_benchmark.data.views import CHANNELS
from inertial_benchmark.nn.base import BaseModel, InputSpec


class DiagonalS4DKernel(nn.Module):
    """Conjugate-paired S4D-Lin kernel with zero-order-hold discretization.

    Each feature has ``d_state/2`` complex states, representing ``d_state``
    real dimensions. B is fixed to one, as in the upstream minimal kernel.
    The continuous diagonal is ``A = -exp(log_A_real) + i*A_imag``. Learned
    dt is the SSM discretization parameter, not the IMU sampling interval.
    """

    def __init__(self, d_model: int, d_state: int = 64,
                 dt_min: float = 0.001, dt_max: float = 0.1) -> None:
        super().__init__()
        if type(d_model) is not int or d_model < 1:
            raise ValueError("d_model must be a positive integer")
        if type(d_state) is not int or d_state < 2 or d_state % 2:
            raise ValueError("d_state must be a positive even integer")
        if not (math.isfinite(dt_min) and math.isfinite(dt_max)
                and 0 < dt_min <= dt_max):
            raise ValueError("dt limits must be finite and 0 < dt_min <= dt_max")
        self.d_model = d_model
        self.d_state = d_state
        half_state = d_state // 2
        log_dt = torch.empty(d_model).uniform_(math.log(dt_min), math.log(dt_max))
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.full((d_model, half_state), math.log(0.5)))
        frequencies = math.pi * torch.arange(half_state, dtype=torch.float32)
        self.A_imag = nn.Parameter(frequencies.unsqueeze(0).repeat(d_model, 1))
        readout = torch.randn(d_model, half_state, dtype=torch.complex64)
        # Real-pair storage preserves complex precision when module.double() is used.
        self.C = nn.Parameter(torch.view_as_real(readout))

    def continuous_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = self.log_dt.exp()
        A = torch.complex(-self.log_A_real.exp(), self.A_imag)
        C = torch.view_as_complex(self.C.contiguous())
        return dt, A, C

    def discrete_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return diagonal Abar, Bbar and C for ``x[t]=Abar*x[t-1]+Bbar*u[t]``."""
        dt, A, C = self.continuous_parameters()
        dt_A = dt.unsqueeze(-1) * A
        return dt_A.exp(), torch.expm1(dt_A) / A, C

    def forward(self, length: int) -> torch.Tensor:
        if type(length) is not int or length < 1:
            raise ValueError("kernel length must be a positive integer")
        dt, A, C = self.continuous_parameters()
        dt_A = dt.unsqueeze(-1) * A
        # expm1 is algebraically identical to exp(dt*A)-1 but avoids cancellation.
        C_bar = C * torch.expm1(dt_A) / A
        time = torch.arange(length, device=A.device, dtype=dt.dtype)
        vandermonde = torch.exp(dt_A.unsqueeze(-1) * time)
        return 2.0 * torch.einsum("hn,hnl->hl", C_bar, vandermonde).real


class S4DLayer(nn.Module):
    """Causal diagonal SSM + D skip + GELU + pointwise Conv/GLU."""

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1,
                 dt_min: float = 0.001, dt_max: float = 0.1) -> None:
        super().__init__()
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.kernel = DiagonalS4DKernel(d_model, d_state, dt_min, dt_max)
        self.D = nn.Parameter(torch.randn(d_model))
        self.activation = nn.GELU()
        # Dropout1d on B,H,T has the upstream DropoutNd(tie=True) semantics.
        self.dropout = nn.Dropout1d(dropout)
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1), nn.GLU(dim=1))

    def linear_response(self, u: torch.Tensor) -> torch.Tensor:
        """Zero-initial-state causal convolution including the direct D term."""
        if u.ndim != 3 or u.shape[1] != self.kernel.d_model or u.shape[-1] < 1:
            raise ValueError("S4DLayer expects (B,d_model,T) with T >= 1")
        length = u.shape[-1]
        kernel = self.kernel(length)
        # Padding to 2T makes this linear convolution, rather than circular wraparound.
        spectrum = torch.fft.rfft(u, n=2 * length) * torch.fft.rfft(
            kernel, n=2 * length).unsqueeze(0)
        response = torch.fft.irfft(spectrum, n=2 * length)[..., :length]
        return response + u * self.D.view(1, -1, 1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.output_linear(self.dropout(self.activation(self.linear_response(u))))


class S4DVelocity(BaseModel):
    """Source-qualified S4D-Lin/ZOH port with a single linear velocity head.

    The temporal SSM layers are causal inside each window, while the pooled
    window-average prediction consumes the complete window. State is reset for
    every window. This adapter does not expose a streaming recurrent interface.
    """

    default_loss = "mse"
    training_initialization = "scratch_supervised"
    implementation_kind = "architecture_port"
    source_architecture = "official_minimal_s4d_lin_zoh"
    inference_contract = "window_noncausal"

    def __init__(self, input_spec: InputSpec, d_model: int = 128,
                 d_state: int = 64, n_layers: int = 4, dropout: float = 0.1,
                 prenorm: bool = False, dt_min: float = 0.001,
                 dt_max: float = 0.1, pretrained: bool = False) -> None:
        super().__init__(input_spec)
        if (tuple(input_spec.channels) != CHANNELS or input_spec.dims not in (2, 3)
                or input_spec.sub_windows or input_spec.extra_inputs
                or input_spec.output_layout != "window"
                or input_spec.target != "avg_velocity"
                or input_spec.frame != "gravity_world" or input_spec.rate != 200
                or input_spec.orientation != "reference" or input_spec.remove_gravity
                or input_spec.output_steps):
            raise ValueError("S4DVelocity requires [gyro_xyz,acc_xyz] at 200 Hz, "
                             "gravity_world window average velocity, dims 2 or 3, "
                             "reference orientation, gravity retained and no history/extra inputs")
        if pretrained:
            raise ValueError("the staging profile requires supervised training from scratch")
        if type(n_layers) is not int or n_layers < 1:
            raise ValueError("n_layers must be a positive integer")
        if type(prenorm) is not bool:
            raise ValueError("prenorm must be bool")
        self.prenorm = prenorm
        self.encoder = nn.Linear(6, d_model)
        self.layers = nn.ModuleList([
            S4DLayer(d_model, d_state, dropout, dt_min, dt_max)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.residual_dropout = nn.ModuleList([nn.Dropout1d(dropout) for _ in range(n_layers)])
        self.velocity_head = nn.Linear(d_model, input_spec.dims)
        self.model_cfg = {
            "d_model": d_model, "d_state": d_state, "n_layers": n_layers,
            "dropout": dropout, "prenorm": prenorm, "dt_min": dt_min,
            "dt_max": dt_max, "pretrained": False,
        }

    def encode_window(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        encoded = self.encoder(imu.transpose(1, 2)).transpose(1, 2)
        for layer, norm, dropout in zip(self.layers, self.norms, self.residual_dropout):
            residual = encoded
            if self.prenorm:
                encoded = norm(encoded.transpose(1, 2)).transpose(1, 2)
            encoded = residual + dropout(layer(encoded))
            if not self.prenorm:
                encoded = norm(encoded.transpose(1, 2)).transpose(1, 2)
        return encoded

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu).mean(dim=-1))}
