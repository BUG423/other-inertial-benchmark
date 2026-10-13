"""Independent engineering ESKF core; NOT a complete original X-IONet system.

Current error order is theta,dv,dp,dbg,dba (15), followed by theta_i,dp_i
for each historical pose clone (6). R_true=Exp(theta) R_est; v/p errors
are world-additive, sensor biases are additive in sensor coordinates. The
declared full-body anchor differs from an unresolved gravity-yaw-local frame.
No reference pose, selector, neural model, engine, registry or RNG is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import left_jacobian, rotation, skew, so3_exp, vector3


def timestamp_ns(value) -> int:
    if (isinstance(value, bool) or not isinstance(value, int)
            or not -(1 << 63) <= value < (1 << 63)):
        raise ValueError("timestamps must be explicit signed-int64 nanoseconds")
    return value


def symmetric_psd(values, dimension: int, name: str = "P") -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (dimension, dimension) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite {dimension}x{dimension} matrix")
    scale = max(1.0, float(np.linalg.norm(values, ord=2)))
    if not np.allclose(values, values.T, rtol=0, atol=1e-10 * scale):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(values).min() < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite; no added jitter")
    return values.copy()


@dataclass
class NavigationState:
    rotation: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray

    def __post_init__(self) -> None:
        self.rotation = rotation(self.rotation)
        for key in ("velocity", "position", "gyro_bias", "accel_bias"):
            setattr(self, key, vector3(getattr(self, key), key))

    def copy(self):
        return NavigationState(self.rotation, self.velocity, self.position,
                               self.gyro_bias, self.accel_bias)


@dataclass
class PoseClone:
    timestamp_ns: int
    rotation: np.ndarray
    position: np.ndarray

    def __post_init__(self) -> None:
        self.timestamp_ns = timestamp_ns(self.timestamp_ns)
        self.rotation = rotation(self.rotation)
        self.position = vector3(self.position, "clone position")


@dataclass(frozen=True)
class NoiseSpectralDensity:
    """Explicit isotropic INDEPENDENT variance PSDs, not stddev or per-step noise.

    gyro measurement: rad²/s; acceleration measurement: m²/s³;
    gyro-bias random walk: rad²/s³; accel-bias random walk: m²/s⁵.
    No author values or defaults are supplied. The measurement-noise discretization
    below treats each interval's average noise as constant over that interval.
    """

    gyro_rad2_per_s: float
    accel_m2_per_s3: float
    gyro_bias_rad2_per_s3: float
    accel_bias_m2_per_s5: float

    def __post_init__(self) -> None:
        for value in self.values():
            if not math.isfinite(value) or value < 0:
                raise ValueError("noise spectral variance densities must be finite and nonnegative")

    def values(self) -> tuple:
        return (self.gyro_rad2_per_s, self.accel_m2_per_s3,
                self.gyro_bias_rad2_per_s3, self.accel_bias_m2_per_s5)


def nominal_propagation(state: NavigationState, gyro, acc, dt: float,
                        gravity) -> NavigationState:
    """Held old attitude/force, Euler velocity, trapezoid position; bias means fixed."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("propagation interval must be finite and positive")
    omega = vector3(gyro, "gyro") - state.gyro_bias
    specific_force = vector3(acc, "accelerometer") - state.accel_bias
    world_acc = vector3(gravity, "world gravity") + state.rotation @ specific_force
    new_v = state.velocity + world_acc * dt
    return NavigationState(state.rotation @ so3_exp(omega * dt), new_v,
                           state.position + 0.5 * dt * (state.velocity + new_v),
                           state.gyro_bias, state.accel_bias)


def propagation_linearization(state: NavigationState, gyro, acc, dt: float) -> tuple:
    """Analytic discrete left-error F and average-measurement noise map G.

    Ftheta-bg=-Rold*Jl((omega-bg)dt)*dt, not -I*dt. Exact first derivative
    of this declared held-attitude nominal integrator; noise approximation is an
    explicit interval-average contract, not an author tuning/discretization claim.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("linearization interval must be finite and positive")
    r = state.rotation
    omega = vector3(gyro) - state.gyro_bias
    world_force = r @ (vector3(acc) - state.accel_bias)
    gyro_map = r @ left_jacobian(omega * dt)
    f = np.eye(15)
    f[0:3, 9:12] = -gyro_map * dt
    f[3:6, 0:3] = -skew(world_force) * dt
    f[3:6, 12:15] = -r * dt
    f[6:9, 0:3] = -skew(world_force) * (0.5 * dt ** 2)
    f[6:9, 3:6] = np.eye(3) * dt
    f[6:9, 12:15] = -r * (0.5 * dt ** 2)
    g = np.zeros((15, 12))
    g[0:3, 0:3] = -gyro_map
    g[3:6, 3:6] = -r
    g[6:9, 3:6] = -r * (0.5 * dt)
    g[9:12, 6:9] = np.eye(3)
    g[12:15, 9:12] = np.eye(3)
    return f, g


def paper_literal_sign_diagnostic(predicted: np.ndarray, measured: np.ndarray,
                                  gain: np.ndarray) -> dict:
    """Paper's h-z under the SAME positive H/plus convention, without state update.

    This diagnoses an unresolved convention; it does not label the author text a
    typo, or offer a secretly corrected/full-paper operational implementation.
    """
    h, z = vector3(predicted), vector3(measured)
    gain = np.asarray(gain, dtype=np.float64)
    if gain.ndim != 2 or gain.shape[1] != 3 or not np.isfinite(gain).all():
        raise ValueError("diagnostic gain must be finite (state_dimension,3)")
    return {"paper_literal_correction": gain @ (h - z),
            "consistent_correction": gain @ (z - h),
            "operational_filter_updated": False}


class ErrorStateFilter:
    """Explicit-parameter 15+6n engineering filter, no inferred original defaults."""

    def __init__(self, state: NavigationState, covariance, *, initial_timestamp_ns: int,
                 gravity, noise: NoiseSpectralDensity, discretization: str,
                 frame: str, reset: str, max_dt_s: float, rotation_chart_limit: float) -> None:
        if not isinstance(state, NavigationState) or not isinstance(noise, NoiseSpectralDensity):
            raise ValueError("explicit navigation state and spectral-density contract required")
        if discretization != "piecewise_constant_measurement_average":
            raise ValueError("only the explicitly declared interval-average discretization is supported")
        if frame != "full_body_anchor" or reset != "left_additive_tangent":
            raise ValueError("core supports only declared full-body anchor and left-tangent reset")
        if (not math.isfinite(max_dt_s) or max_dt_s <= 0
                or not math.isfinite(rotation_chart_limit) or not 0 < rotation_chart_limit < math.pi):
            raise ValueError("explicit positive max dt and a principal rotation chart limit < pi required")
        self.state = state.copy()
        self.covariance = symmetric_psd(covariance, 15)
        self.timestamp_ns = timestamp_ns(initial_timestamp_ns)
        self.gravity = vector3(gravity, "world gravity")
        self.noise = noise
        self.discretization, self.frame, self.reset = discretization, frame, reset
        self.max_dt_s, self.rotation_chart_limit = float(max_dt_s), float(rotation_chart_limit)
        self.clones: list = []

    @property
    def dimension(self) -> int:
        return 15 + 6 * len(self.clones)

    def propagate(self, gyro, acc, *, next_timestamp_ns: int) -> dict:
        next_time = timestamp_ns(next_timestamp_ns)
        interval = next_time - self.timestamp_ns  # Python int: check range before float conversion
        if interval <= 0 or interval >= (1 << 63):
            raise ValueError("clock must increase with a representable int64 interval")
        dt = interval * 1e-9
        if dt > self.max_dt_s:
            raise ValueError("recorded interval exceeds the explicit propagation contract")
        f_current, g_current = propagation_linearization(self.state, gyro, acc, dt)
        f = np.eye(self.dimension)
        f[:15, :15] = f_current
        g = np.zeros((self.dimension, 12))
        g[:15] = g_current
        qc = np.diag(np.repeat(self.noise.values(), 3))
        qd = (g @ qc @ g.T) * dt
        candidate_p = f @ self.covariance @ f.T + qd
        candidate_p = 0.5 * (candidate_p + candidate_p.T)
        candidate_state = nominal_propagation(self.state, gyro, acc, dt, self.gravity)
        symmetric_psd(candidate_p, self.dimension)
        self.state, self.covariance, self.timestamp_ns = candidate_state, candidate_p, next_time
        return {"F": f, "G": g, "Q": qd, "dt_s": dt}

    def clone_current(self) -> dict:
        if self.clones and self.clones[-1].timestamp_ns >= self.timestamp_ns:
            raise ValueError("clones need distinct monotonically increasing current timestamps")
        old = self.dimension
        j = np.zeros((old + 6, old))
        j[:old] = np.eye(old)
        j[old:old + 3, 0:3] = np.eye(3)
        j[old + 3:old + 6, 6:9] = np.eye(3)
        candidate = j @ self.covariance @ j.T
        symmetric_psd(candidate, old + 6)
        clone = PoseClone(self.timestamp_ns, self.state.rotation, self.state.position)
        self.clones.append(clone)
        self.covariance = candidate
        return {"augmentation_jacobian": j, "timestamp_ns": clone.timestamp_ns}

    def _clone_index(self, anchor_timestamp_ns: int) -> int:
        anchor_timestamp_ns = timestamp_ns(anchor_timestamp_ns)
        for i, clone in enumerate(self.clones):
            if clone.timestamp_ns == anchor_timestamp_ns:
                return i
        raise ValueError("measurement anchor must be an actual retained pose clone")

    def marginalize_clone(self, anchor_timestamp_ns: int) -> None:
        index = self._clone_index(anchor_timestamp_ns)
        lo = 15 + 6 * index
        keep = np.concatenate((np.arange(lo), np.arange(lo + 6, self.dimension)))
        candidate = self.covariance[np.ix_(keep, keep)]
        symmetric_psd(candidate, self.dimension - 6)
        self.clones.pop(index)
        self.covariance = candidate.copy()

    def measurement_model(self, anchor_timestamp_ns: int) -> tuple:
        index = self._clone_index(anchor_timestamp_ns)
        anchor = self.clones[index]
        if anchor.timestamp_ns >= self.timestamp_ns:
            raise ValueError("displacement needs a historical anchor, not a zero-duration current clone")
        dp = self.state.position - anchor.position
        local = anchor.rotation.T @ dp
        h = np.zeros((3, self.dimension))
        offset = 15 + 6 * index
        h[:, offset:offset + 3] = anchor.rotation.T @ skew(dp)
        h[:, offset + 3:offset + 6] = -anchor.rotation.T
        h[:, 6:9] = anchor.rotation.T
        return local, h

    def _retracted(self, delta: np.ndarray) -> tuple:
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != (self.dimension,) or not np.isfinite(delta).all():
            raise ValueError("correction must be a finite full error-state vector")
        starts = [0] + [15 + 6 * i for i in range(len(self.clones))]
        if any(np.linalg.norm(delta[s:s + 3]) >= self.rotation_chart_limit for s in starts):
            raise ValueError("rotation correction exceeds the explicitly admitted tangent chart")
        state = NavigationState(so3_exp(delta[:3]) @ self.state.rotation,
                                self.state.velocity + delta[3:6],
                                self.state.position + delta[6:9],
                                self.state.gyro_bias + delta[9:12],
                                self.state.accel_bias + delta[12:15])
        clones = []
        reset = np.eye(self.dimension)
        reset[:3, :3] = left_jacobian(delta[:3])
        for i, clone in enumerate(self.clones):
            offset = 15 + 6 * i
            clones.append(PoseClone(clone.timestamp_ns,
                                    so3_exp(delta[offset:offset + 3]) @ clone.rotation,
                                    clone.position + delta[offset + 3:offset + 6]))
            reset[offset:offset + 3, offset:offset + 3] = left_jacobian(delta[offset:offset + 3])
        return state, clones, reset

    def update(self, measurement, covariance, *, anchor_timestamp_ns: int) -> dict:
        """Consistent engineering residual z-h only; Joseph then left-coordinate reset."""
        z = vector3(measurement, "local displacement")
        sigma = symmetric_psd(covariance, 3, "measurement Sigma")
        try:
            np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement Sigma must be SPD without fake jitter") from exc
        predicted, h = self.measurement_model(anchor_timestamp_ns)
        innovation_cov = h @ self.covariance @ h.T + sigma
        lower = np.linalg.cholesky(innovation_cov)
        gain = np.linalg.solve(lower.T, np.linalg.solve(lower, h @ self.covariance)).T
        residual = z - predicted
        delta = gain @ residual
        candidate_state, candidate_clones, reset = self._retracted(delta)
        a = np.eye(self.dimension) - gain @ h
        posterior = a @ self.covariance @ a.T + gain @ sigma @ gain.T
        posterior = 0.5 * (posterior + posterior.T)
        candidate_p = reset @ posterior @ reset.T
        candidate_p = 0.5 * (candidate_p + candidate_p.T)
        symmetric_psd(candidate_p, self.dimension)
        self.state, self.clones, self.covariance = candidate_state, candidate_clones, candidate_p
        return {"predicted": predicted, "residual": residual, "H": h, "K": gain,
                "innovation_covariance": innovation_cov, "correction": delta,
                "joseph_before_reset": posterior, "reset_jacobian": reset}

    def state_dict(self) -> dict:
        """JSON basic/numeric types only; no estimator objects, implicit defaults or RNG."""
        current = {key: getattr(self.state, key).tolist() for key in
                   ("rotation", "velocity", "position", "gyro_bias", "accel_bias")}
        return {"schema": "xionet-left-engineering-core-v1", "current": current,
                "timestamp_ns": self.timestamp_ns, "covariance": self.covariance.tolist(),
                "clones": [{"timestamp_ns": c.timestamp_ns, "rotation": c.rotation.tolist(),
                            "position": c.position.tolist()} for c in self.clones],
                "gravity": self.gravity.tolist(), "noise_psd": list(self.noise.values()),
                "discretization": self.discretization, "frame": self.frame, "reset": self.reset,
                "max_dt_s": self.max_dt_s, "rotation_chart_limit": self.rotation_chart_limit}

    @classmethod
    def from_state_dict(cls, saved: dict):
        if not isinstance(saved, dict) or saved.get("schema") != "xionet-left-engineering-core-v1":
            raise ValueError("unknown filter scientific state schema")
        state = NavigationState(**saved["current"])
        covariance = np.asarray(saved["covariance"], dtype=np.float64)
        result = cls(state, covariance[:15, :15], initial_timestamp_ns=saved["timestamp_ns"],
                     gravity=saved["gravity"], noise=NoiseSpectralDensity(*saved["noise_psd"]),
                     discretization=saved["discretization"], frame=saved["frame"], reset=saved["reset"],
                     max_dt_s=saved["max_dt_s"], rotation_chart_limit=saved["rotation_chart_limit"])
        result.clones = [PoseClone(**c) for c in saved["clones"]]
        times = [c.timestamp_ns for c in result.clones]
        if times != sorted(set(times)) or any(t > result.timestamp_ns for t in times):
            raise ValueError("checkpoint clone clocks must be unique, ordered and not future")
        result.covariance = symmetric_psd(covariance, result.dimension)
        return result
