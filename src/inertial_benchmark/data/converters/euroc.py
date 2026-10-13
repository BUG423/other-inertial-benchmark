"""EuRoC MAV IMU/state release, including the AirIO image-free derivative.

Both released T_BS matrices must identify the same IMU/body frame. Integer ns
are subtracted before conversion to seconds. State biases are labels and are
never applied to the input. The shared v1 pipeline handles clocks and gaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Iterator, Optional

import numpy as np
import yaml

from . import _rig_utils as rig
from ._external_utils import AIRIO_COMMIT, AIRIO_SOURCE, evidence, numeric_csv, safe_orientation
from .base import RawSequence

NAME = "euroc"
VERSION = "1.0.0"
LICENSE = (
    "EuRoC original dataset terms unverified (https://doi.org/10.3929/ethz-b-000690084); "
    "AirIO code license does not relicense data"
)
TRAIN_POOL = ("MH_01_easy", "MH_03_medium", "MH_05_difficult", "V1_02_medium",
              "V2_01_easy", "V2_03_difficult")
TEST = ("MH_02_easy", "MH_04_difficult", "V1_03_difficult", "V2_02_medium", "V1_01_easy")
OFFICIAL_SPLITS_LEAK = True
OFFICIAL_SPLITS_LEAK_REASON = (
    "AirIO's published train_list and val_list are identical; default validation is the "
    "whole V2_03_difficult recording withheld from the six-recording train pool. "
    "The original AirIO lists are preserved only as official_* provenance."
)


def _discover(source: Path) -> dict:
    source = Path(source)
    roots = (source, source / "EuRoC-Dataset")
    for root in roots:
        found = {p.parent.parent.parent.name: p.parent.parent.parent
                 for p in root.glob("*/mav0/imu0/data.csv")}
        if found:
            return found
    if (source / "mav0/imu0/data.csv").is_file():
        return {source.name: source}
    return {}


def list_sequences(source: Path) -> list:
    return sorted(_discover(source))


def official_splits(source: Path) -> dict:
    local = set(list_sequences(source))
    return {"train": [s for s in TRAIN_POOL if s in local],
            "val": [s for s in TRAIN_POOL if s in local],
            "test": [s for s in TEST if s in local]}


def grouped_splits(source: Path) -> dict:
    local = set(list_sequences(source))
    return {"train": [s for s in TRAIN_POOL if s != "V2_03_difficult" and s in local],
            "val": [s for s in ("V2_03_difficult",) if s in local],
            "test": [s for s in TEST if s in local]}


def parse_sequence(folder: Path) -> RawSequence:
    folder = Path(folder)
    imu = folder / "mav0/imu0/data.csv"
    pose = folder / "mav0/state_groundtruth_estimate0/data.csv"
    calibrations = [folder / "mav0/imu0/sensor.yaml",
                    folder / "mav0/state_groundtruth_estimate0/sensor.yaml"]
    yaml_text = {}
    for path in calibrations:
        text = path.read_text()
        matrix = np.asarray(yaml.safe_load(text)["T_BS"]["data"], dtype=float).reshape(4, 4)
        if not np.allclose(matrix, np.eye(4), atol=1e-12, rtol=0):
            raise ValueError(
                "nonidentity T_BS: explicit sensor/state rotation and lever arm required"
            )
        yaml_text[str(path.relative_to(folder))] = text
    ti_ns, imu_data, imu_header = numeric_csv(imu, 7, integer_clock=True)
    tp_ns, pose_data, pose_header = numeric_csv(pose, 17, integer_clock=True)
    origin = int(ti_ns[0])
    ti, tp = (ti_ns - origin) * 1e-9, (tp_ns - origin) * 1e-9
    q, pose_valid = safe_orientation(pose_data[:, 3:7])
    pose_valid &= np.isfinite(pose_data[:, :3]).all(1) & np.isfinite(pose_data[:, 7:10]).all(1)
    imu_valid = np.isfinite(imu_data).all(1)
    paths = [imu, pose, *calibrations]
    machine_hall = folder.name.startswith("MH_")
    attrs = {
        "subject_id": "not_applicable", "device_id": "VI-Sensor_ADIS16448",
        "placement": "body", "group_id": folder.name, "recording_id": folder.name,
        "body_frame": "euroc_imu_S_identity_T_BS",
        "position_source": "Leica_plus_IMU_batch_state_at_IMU_origin" if machine_hall
        else "Vicon_plus_IMU_batch_state_at_IMU_origin",
        "orientation_source": ("EuRoC_batch_IMU_state" if machine_hall
                               else "EuRoC_batch_Vicon_IMU_state"),
        "device_orientation_source": "none",
        "source_files": ",".join(str(p.relative_to(folder)) for p in paths),
        "source_license": LICENSE, "source_sha256": evidence(paths, folder),
        "source_csv_headers": {"imu": imu_header, "state": pose_header},
        "source_sensor_yaml": yaml_text, "source_timestamp_origin_ns": origin,
        "start_time_unix": origin * 1e-9,
        "timestamp_precision_policy": "int64_subtract_before_seconds",
        "reference_variant": "original_released_batch_state",
        "reference_bias_used_for_input": False,
        "reference_position_is_imu_origin": True, "reference_orientation_is_privileged": True,
        "benchmark_track": "uav_reference_orientation", "airio_source_commit": AIRIO_COMMIT,
        "source_description": AIRIO_SOURCE, "capabilities": {"reference_orientation": True,
        "device_orientation": False, "independent_mocap_orientation": False,
        "reference_uses_external_Vicon": not machine_hall},
    }
    notes = [
        "state q_RS is IMU S->reference R (wxyz), position is S origin in R; "
        "both source T_BS are identity",
        "source clocks are exact int64 ns; subtract shared integer origin before seconds; "
        "no parser resampling",
        "raw IMU SI body axes retained; estimated gyro/accel biases and velocities "
        "remain reference labels",
        "no deployable attitude stream; orientation=reference is a privileged oracle protocol; "
        "device mode unavailable",
        "source V1_01 orientation correction archive is not applied; "
        "original reference version is recorded",
    ]
    return RawSequence(folder.name, ti, np.nan_to_num(imu_data[:, :3]),
                       np.nan_to_num(imu_data[:, 3:6]),
                       tp, pose_data[:, :3], q, velocity=pose_data[:, 7:10], imu_valid=imu_valid,
                       pose_valid=pose_valid, attrs=attrs, notes=notes)


def iter_raw_sequences(
    source: Path, only: Optional[Collection[str]] = None
) -> Iterator[RawSequence]:
    found = _discover(source)
    for name in sorted(found) if only is None else sorted(only):
        try:
            yield parse_sequence(found[name])
        except (OSError, ValueError, KeyError) as exc:
            yield rig.rejected_sequence(name, f"parse/calibration error: {exc}",
                                        {"source_license": LICENSE})

