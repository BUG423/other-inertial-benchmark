"""Research-only typed transport for original-clock AI-IMU-DR filter output.

No NIO wiring, reconstruction, world rotation, GT join or covariance mapping.
Metadata records caller evidence; this codec does not verify raw data/permissions.
Cartesian uncertainty is explicitly unavailable until separate qualification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import re
from typing import Any, Optional

import numpy as np


SCHEMA = "ipb-ai-imu-dr-full-sequence-research-v2"
ERROR_COORDINATES = "left_SE2_3_theta_xiv_xip_bg_ba_thetaci_tci"
DISCRETIZATION = "released_Taylor3_Gdt_omitted_error_reset"


def scientific_units() -> dict:
    """Fixed SI declarations, not an automatic test of acquisition units.

    P is a mixed-coordinate covariance: each element has the outer-product
    units of its two error coordinates. It has no single Cartesian unit.
    A fresh nested mapping is returned so callers cannot mutate the contract.
    """
    return {"timestamp_ns": "ns", "p_world": "m", "v_world": "m_s",
            "R_world_from_IMU": "dimensionless", "R_IMU_from_car": "dimensionless",
            "gyro_bias": "rad_s", "accelerometer_bias": "m_s2", "lever_arm": "m",
            "measurement_variances": {"axes": ["car_lateral", "car_up"], "unit": "m2_s2"},
            "error_covariance": {
                "kind": "mixed_error_coordinate_outer_products",
                "coordinate_order": ["theta", "xiv", "xip", "bg", "ba", "thetaci", "tci"],
                "coordinate_block_units": ["rad", "m_s", "m", "rad_s", "m_s2", "rad", "m"],
                "block_size": 3,
                "entry_rule": "P[i,j] has unit(error[i]) times unit(error[j])"}}


class CartesianUncertaintyUnavailable(ValueError):
    """State tangent covariance is not a qualified Cartesian distribution."""


@dataclass(frozen=True)
class FullSequenceContract:
    sequence_id: str
    filter_source_manifest_sha256: str
    clock_provenance: str = "unverified"
    raw_source_sha256: Optional[str] = None
    raw_clock_receipt_sha256: Optional[str] = None
    initialization_receipt_sha256: Optional[str] = None
    world_frame: str = "gravity_aligned_z_up"
    input_frame: str = "sensor_body"
    position_origin: str = "local_recording_IMU_origin"
    initialization_privilege: str = "reference_initial_R_v"
    error_coordinates: str = ERROR_COORDINATES
    discretization: str = DISCRETIZATION
    measurement_variance_frame: str = "car_lateral_up"
    future_input_samples: int = 8
    nominal_raw_rate_hz: float = 100.
    unit_system: str = "SI"
    timestamp_unit: str = "ns"
    position_unit: str = "m"
    velocity_unit: str = "m_s"
    rotation_unit: str = "dimensionless"
    gyro_bias_unit: str = "rad_s"
    accelerometer_bias_unit: str = "m_s2"
    lever_arm_unit: str = "m"
    measurement_variance_unit: str = "m2_s2"
    error_covariance_units: str = "mixed_error_coordinate_outer_products"

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("explicit recording identity required")
        for name in ("filter_source_manifest_sha256", "raw_source_sha256", "raw_clock_receipt_sha256",
                     "initialization_receipt_sha256"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)):
                raise ValueError(f"{name} must be SHA256 or an explicitly unavailable optional receipt")
        if self.filter_source_manifest_sha256 is None:
            raise ValueError("filter code identity is required")
        if self.clock_provenance not in ("unverified", "synthetic_engineering", "caller_acquisition_receipts"):
            raise ValueError("unknown original-clock evidence declaration")
        if self.clock_provenance == "caller_acquisition_receipts" and any(
                getattr(self, name) is None for name in ("raw_source_sha256", "raw_clock_receipt_sha256",
                                                         "initialization_receipt_sha256")):
            raise ValueError("caller acquisition declaration needs raw/clock/initialization receipts")
        expected = {"world_frame": "gravity_aligned_z_up", "input_frame": "sensor_body",
                    "position_origin": "local_recording_IMU_origin",
                    "initialization_privilege": "reference_initial_R_v", "error_coordinates": ERROR_COORDINATES,
                    "discretization": DISCRETIZATION, "measurement_variance_frame": "car_lateral_up",
                    "unit_system": "SI", "timestamp_unit": "ns", "position_unit": "m",
                    "velocity_unit": "m_s", "rotation_unit": "dimensionless",
                    "gyro_bias_unit": "rad_s", "accelerometer_bias_unit": "m_s2",
                    "lever_arm_unit": "m", "measurement_variance_unit": "m2_s2",
                    "error_covariance_units": "mixed_error_coordinate_outer_products"}
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("transport must explicitly preserve the accepted scientific frame/state contract")
        if type(self.future_input_samples) is not int or self.future_input_samples != 8:
            raise ValueError("released CNN uses eight future samples")
        if isinstance(self.nominal_raw_rate_hz, bool) or self.nominal_raw_rate_hz != 100.:
            raise ValueError("this scientific track declares original nominal100Hz input")


def _copy_readonly(value: Any, shape: tuple, dtype: np.dtype, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"{name} requires {shape} {dtype}, got {value.shape} {value.dtype}")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite state values")
    result = value.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FullSequencePrediction:
    contract: FullSequenceContract
    timestamp_ns: np.ndarray
    p_world: np.ndarray
    v_world: np.ndarray
    R_world_from_IMU: np.ndarray
    gyro_bias: np.ndarray
    accelerometer_bias: np.ndarray
    R_IMU_from_car: np.ndarray
    lever_arm: np.ndarray
    error_covariance: np.ndarray
    measurement_variances: np.ndarray
    valid_input: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.contract, FullSequenceContract):
            raise TypeError("typed full-sequence scientific contract required")
        clock = np.asarray(self.timestamp_ns)
        if clock.ndim != 1 or clock.dtype != np.dtype("int64") or len(clock) < 2:
            raise ValueError("original clock needs >=2 int64 nanosecond samples")
        if not np.all(clock[1:] > clock[:-1]):
            raise ValueError("original clock must be strictly increasing before subtraction")
        # Python integer checks cannot overflow even at opposite int64 bounds.
        if any(int(b) - int(a) > np.iinfo(np.int64).max for a, b in zip(clock[:-1], clock[1:])):
            raise ValueError("adjacent source clock difference overflows signed int64")
        n = len(clock)
        shapes = {"timestamp_ns": ((n,), np.dtype("int64")),
                  "p_world": ((n, 3), np.dtype("float64")), "v_world": ((n, 3), np.dtype("float64")),
                  "R_world_from_IMU": ((n, 3, 3), np.dtype("float64")),
                  "gyro_bias": ((n, 3), np.dtype("float64")),
                  "accelerometer_bias": ((n, 3), np.dtype("float64")),
                  "R_IMU_from_car": ((n, 3, 3), np.dtype("float64")),
                  "lever_arm": ((n, 3), np.dtype("float64")),
                  "error_covariance": ((n, 21, 21), np.dtype("float64")),
                  "measurement_variances": ((n, 2), np.dtype("float64")),
                  "valid_input": ((n,), np.dtype("bool"))}
        for name, (shape, dtype) in shapes.items():
            object.__setattr__(self, name, _copy_readonly(getattr(self, name), shape, dtype, name))
        if not self.valid_input.any():
            raise ValueError("no valid filter inputs; do not manufacture a trajectory")
        for rotation in (self.R_world_from_IMU, self.R_IMU_from_car):
            if (not np.allclose(rotation.swapaxes(-2, -1) @ rotation, np.eye(3), rtol=0, atol=1e-8)
                    or not np.allclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-8)):
                raise ValueError("state orientation must already be SO(3), never silently projected")
        P = self.error_covariance
        if not np.allclose(P, P.swapaxes(-2, -1), rtol=0, atol=1e-10):
            raise ValueError("reported tangent covariance is not symmetric")
        if np.linalg.eigvalsh(P).min() < -1e-10:
            raise ValueError("reported tangent covariance is not PSD; no jitter/repair")
        if not (self.measurement_variances > 0).all():
            raise ValueError("car constraint variances must be positive")

    @property
    def relative_seconds(self) -> np.ndarray:
        origin = int(self.timestamp_ns[0])
        result = np.fromiter(((int(v) - origin) * 1e-9 for v in self.timestamp_ns),
                             dtype=np.float64, count=len(self.timestamp_ns))
        result.setflags(write=False)
        return result

    def metric_arrays(self, dims: int) -> dict:
        """Reference-free, same-XYZ slices. No interpolation or trajectory integration."""
        if type(dims) is not int or dims not in (2, 3):
            raise ValueError("metric dimensions must be integer2 or3")
        return {"t": self.relative_seconds, "pos_pred": self.p_world[:, :dims],
                "vel_pred": self.v_world[:, :dims], "valid_input": self.valid_input}

    def uncertainty_support(self) -> dict:
        return {"supported": False, "reason": "unqualified_state_covariance_to_Cartesian_mapping",
                "measurement_variances_are_prediction_sigma": False,
                "position_tangent_block_is_world_covariance": False,
                "error_coordinates": self.contract.error_coordinates,
                "mapping_required": "Jp=[-skew(p),0,I,0...]; Cp=Jp P Jp^T including cross terms",
                "future_qualification": "independent_FD_and_MonteCarlo_then_statistics",
                "current_metrics": "NLL_NEES_coverage_ECE_unavailable"}

    def position_covariance_world(self) -> np.ndarray:
        raise CartesianUncertaintyUnavailable("P[6:9,6:9] is left-SE2(3) tangent covariance; "
                                               "Cartesian Jp P Jp^T mapping is not qualified")

    def save_npz(self, path: Any) -> Path:
        """Research transport only, not public NIO native export/resume."""
        path = Path(path)
        arrays = {field.name: getattr(self, field.name) for field in fields(self) if field.name != "contract"}
        payload = {"schema": SCHEMA, "contract": asdict(self.contract), "units": scientific_units(),
                   "Cartesian_uncertainty_supported": False, "resume_training_supported": False,
                   "codec_verifies_raw_acquisition_permissions": False}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(payload, sort_keys=True)), **arrays)
        return path

    @classmethod
    def load_npz(cls, path: Any) -> "FullSequencePrediction":
        expected = {field.name for field in fields(cls) if field.name != "contract"}
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != expected | {"metadata"}:
                raise ValueError("research full-state transport fields differ")
            meta = json.loads(saved["metadata"].item())
            if (meta.get("schema") != SCHEMA or meta.get("Cartesian_uncertainty_supported") is not False
                    or meta.get("resume_training_supported") is not False
                    or meta.get("codec_verifies_raw_acquisition_permissions") is not False):
                raise ValueError("research full-state transport must not claim extra qualified capabilities")
            if meta.get("units") != scientific_units():
                raise ValueError("research full-state SI unit declarations differ or are missing")
            contract = FullSequenceContract(**meta["contract"])
            return cls(contract, **{name: saved[name] for name in expected})


def from_filter_trajectory(timestamps_ns: Any, trajectory: Any, contract: FullSequenceContract,
                           *, batch_index: int, valid_input: Any) -> FullSequencePrediction:
    """Detach one accepted filter result without modifying any returned value.

    Only transports filter outputs. No raw data is read, GT is joined, grid is
    resampled, frame is rotated, covariance is projected or velocity integrated.
    """
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("explicit nonnegative batch index required")

    def host(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    clock = host(timestamps_ns)
    if clock.ndim != 2 or batch_index >= clock.shape[0]:
        raise ValueError("filter clock requires (B,T) and an existing batch item")
    mapping = {"p_world": "p", "v_world": "v", "R_world_from_IMU": "R",
               "gyro_bias": "bg", "accelerometer_bias": "ba", "R_IMU_from_car": "Rci",
               "lever_arm": "tci", "error_covariance": "P", "measurement_variances": "measurement_variances"}
    converted = {}
    for dest, source in mapping.items():
        array = host(getattr(trajectory, source))
        if array.shape[:2] != clock.shape:
            raise ValueError("full filter state and clock batch/time axes differ")
        converted[dest] = array[batch_index]
    return FullSequencePrediction(contract, clock[batch_index], valid_input=valid_input, **converted)
