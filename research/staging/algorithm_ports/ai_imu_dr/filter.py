"""21D released-discretization invariant filter, with modern differentiability.

Deliberately preserves Taylor3 Phi, dt-scaled G and omitted post-injection
covariance reset. The measurement bias Jacobian follows the released NumPy
implementation, not the opposite sign in the released Torch implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .geometry import delta_t_seconds, se23_exp, skew, so3_exp


@dataclass(frozen=True)
class State:
    R: Tensor
    v: Tensor
    p: Tensor
    bg: Tensor
    ba: Tensor
    Rci: Tensor
    tci: Tensor

    def tensors(self) -> tuple[Tensor, ...]:
        return tuple(getattr(self, item.name) for item in fields(self))

    def clone(self) -> "State":
        return State(*(value.clone() for value in self.tensors()))


@dataclass(frozen=True)
class Trajectory:
    R: Tensor
    v: Tensor
    p: Tensor
    bg: Tensor
    ba: Tensor
    Rci: Tensor
    tci: Tensor
    P: Tensor
    measurement_variances: Tensor


def validate_state(state: State, batch: int, *, device: torch.device) -> None:
    for name, value in zip((f.name for f in fields(state)), state.tensors()):
        shape = (batch, 3, 3) if name in {"R", "Rci"} else (batch, 3)
        if value.shape != shape or value.dtype != torch.float64 or value.device != device:
            raise ValueError(f"{name} must be float64 {shape} on the IMU device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite initial {name}")
    eye = torch.eye(3, dtype=torch.float64, device=device)
    for rotation in (state.R, state.Rci):
        if not torch.allclose(rotation.mT @ rotation, eye.expand_as(rotation), atol=1e-8, rtol=0):
            raise ValueError("initial orientation must be SO(3), not projected silently")
        if not torch.allclose(torch.linalg.det(rotation), torch.ones(batch, device=device, dtype=torch.float64),
                              atol=1e-8, rtol=0):
            raise ValueError("initial orientation must have determinant +1")


def inject(state: State, delta: Tensor) -> State:
    if delta.shape != (*state.v.shape[:-1], 21):
        raise ValueError("21D group increment shape mismatch")
    dR, dv, dp = se23_exp(delta[..., :9])
    return State(dR @ state.R, (dR @ state.v[..., None]).squeeze(-1) + dv,
                 (dR @ state.p[..., None]).squeeze(-1) + dp,
                 state.bg + delta[..., 9:12], state.ba + delta[..., 12:15],
                 so3_exp(delta[..., 15:18]) @ state.Rci, state.tci + delta[..., 18:21])


def nominal_propagate(state: State, imu: Tensor, dt: Tensor, gravity: Tensor) -> State:
    acceleration = (state.R @ (imu[..., 3:6] - state.ba)[..., None]).squeeze(-1) + gravity
    delta = dt[..., None]
    return State(state.R @ so3_exp((imu[..., :3] - state.bg) * delta),
                 state.v + acceleration * delta,
                 state.p + state.v * delta + .5 * acceleration * delta.square(),
                 state.bg, state.ba, state.Rci, state.tci)


def released_linearization(state: State, gravity: Tensor) -> tuple[Tensor, Tensor]:
    """Continuous F/G exactly as released; step scaling is applied separately."""
    batch = state.v.shape[0]
    F = state.v.new_zeros(batch, 21, 21)
    G = state.v.new_zeros(batch, 21, 18)
    eye = torch.eye(3, dtype=state.v.dtype, device=state.v.device)
    vR, pR = skew(state.v) @ state.R, skew(state.p) @ state.R
    F[:, 3:6, :3] = skew(gravity)
    F[:, 6:9, 3:6] = eye
    F[:, :3, 9:12] = -state.R
    F[:, 3:6, 9:12] = -vR
    F[:, 6:9, 9:12] = -pR
    F[:, 3:6, 12:15] = -state.R
    G[:, :3, :3] = state.R
    G[:, 3:6, :3] = vR
    G[:, 6:9, :3] = pR
    G[:, 3:6, 3:6] = state.R
    for row, column in ((9, 6), (12, 9), (15, 12), (18, 15)):
        G[:, row:row + 3, column:column + 3] = eye
    return F, G


def propagate(state: State, P: Tensor, imu: Tensor, dt: Tensor, gravity: Tensor,
              Q: Tensor) -> tuple[State, Tensor]:
    F, G = released_linearization(state, gravity)
    F, G = F * dt[:, None, None], G * dt[:, None, None]
    F2 = F @ F
    Phi = torch.eye(21, dtype=P.dtype, device=P.device) + F + .5 * F2 + (F2 @ F) / 6
    covariance = Phi @ (P + G @ Q @ G.mT) @ Phi.mT
    return nominal_propagate(state, imu, dt, gravity), covariance


def observation(state: State, imu: Tensor) -> Tensor:
    v_imu = (state.R.mT @ state.v[..., None]).squeeze(-1)
    body = (state.Rci.mT @ v_imu[..., None]).squeeze(-1)
    body = body + (skew(state.tci) @ (imu[..., :3] - state.bg)[..., None]).squeeze(-1)
    return body[..., 1:3]


def observation_jacobian(state: State, imu: Tensor) -> Tensor:
    H = state.v.new_zeros(state.v.shape[0], 2, 21)
    v_imu = (state.R.mT @ state.v[..., None]).squeeze(-1)
    H[:, :, 3:6] = (state.Rci.mT @ state.R.mT)[:, 1:3]
    H[:, :, 9:12] = -skew(state.tci)[:, 1:3]  # released NumPy sign
    H[:, :, 15:18] = (state.Rci.mT @ skew(v_imu))[:, 1:3]
    H[:, :, 18:21] = -skew(imu[..., :3] - state.bg)[:, 1:3]
    return H


def update(state: State, P: Tensor, imu: Tensor, variances: Tensor) -> tuple[State, Tensor]:
    if variances.shape != (state.v.shape[0], 2) or not bool((variances > 0).all()):
        raise ValueError("two strictly positive car-frame measurement variances required")
    if not bool(torch.isfinite(variances).all()):
        raise ValueError("non-finite measurement variance")
    H = observation_jacobian(state, imu)
    N = torch.diag_embed(variances)
    S = H @ P @ H.mT + N
    # Strict factorization. No fake diagonal jitter or silent PSD repair.
    chol = torch.linalg.cholesky(S)
    K = torch.cholesky_solve((P @ H.mT).mT, chol).mT
    delta = (K @ (-observation(state, imu))[..., None]).squeeze(-1)
    A = torch.eye(21, dtype=P.dtype, device=P.device) - K @ H
    joseph = A @ P @ A.mT + K @ N @ K.mT
    # Released behavior: symmetrize Joseph, OMIT tangent reset intentionally.
    return inject(state, delta), (joseph + joseph.mT) / 2


def run_filter(timestamps_ns: Tensor, imu: Tensor, initial: State, P0: Tensor,
               Q: Tensor, variances: Tensor, gravity: Tensor) -> Trajectory:
    if imu.ndim != 3 or imu.shape[-1] != 6 or imu.dtype != torch.float64:
        raise ValueError("body IMU must be float64 (B,T,6), gyro(rad/s) then specific force(m/s²)")
    if not bool(torch.isfinite(imu).all()):
        raise ValueError("IMU must be finite")
    batch, length, _ = imu.shape
    if batch < 1 or length < 2:
        raise ValueError("non-empty batch and at least two clock samples required")
    if timestamps_ns.shape != (batch, length) or timestamps_ns.device != imu.device:
        raise ValueError("clock/IMU shape or device mismatch")
    dt = delta_t_seconds(timestamps_ns)
    validate_state(initial, batch, device=imu.device)
    if P0.shape != (batch, 21, 21) or Q.shape != (batch, 18, 18):
        raise ValueError("full P0/Q batched shape mismatch")
    for name, matrix in (("P0", P0), ("Q", Q)):
        if matrix.dtype != imu.dtype or matrix.device != imu.device or not bool(torch.isfinite(matrix).all()):
            raise ValueError(f"invalid {name} dtype/device/finiteness")
        if not torch.allclose(matrix, matrix.mT, atol=1e-12, rtol=0):
            raise ValueError(f"{name} must be symmetric")
        if bool((torch.linalg.eigvalsh(matrix.detach()) < -1e-12).any()):
            raise ValueError(f"{name} must be PSD; no jitter is supplied")
    if variances.shape != (batch, length, 2) or variances.dtype != imu.dtype:
        raise ValueError("measurement variances must be (B,T,2) float64")
    if variances.device != imu.device or not bool(torch.isfinite(variances).all()) or not bool((variances > 0).all()):
        raise ValueError("measurement variances must be finite positive on IMU device")
    if gravity.shape != (3,) or gravity.dtype != imu.dtype or gravity.device != imu.device:
        raise ValueError("gravity must be explicit float64 world three-vector")
    if not bool(torch.isfinite(gravity).all()):
        raise ValueError("non-finite gravity")
    states, covariances = [initial], [P0]
    for i in range(1, length):  # source uses the right-end IMU sample u[i]
        state, P = propagate(states[-1], covariances[-1], imu[:, i], dt[:, i - 1], gravity, Q)
        state, P = update(state, P, imu[:, i], variances[:, i])
        states.append(state)
        covariances.append(P)
    components = [torch.stack([getattr(state, field.name) for state in states], 1) for field in fields(State)]
    return Trajectory(*components, torch.stack(covariances, 1), variances)
