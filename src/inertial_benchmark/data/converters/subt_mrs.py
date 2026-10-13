"""AirIMU's SubT-MRS CSV release, preserving the independent source lineage.

The released reference is map/fusion based and incorporates IMU measurements.
Reference bias columns and orientation are privileged and do not calibrate the
input or become a device attitude stream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Collection, Iterator, Optional

import numpy as np

from . import _rig_utils as rig
from ._external_utils import evidence, numeric_csv
from .base import RawSequence

NAME = "subt_mrs"
VERSION = "1.0.1"
LICENSE = "original SubT-MRS and derivative release terms; verify for redistribution"
AUTHOR = "https://github.com/haleqiu/AirIMU"
SOURCE_CONTRACT = AUTHOR + "/blob/main/datasets/SubTdataset.py"
REFERENCE_CONTRACT = "https://arxiv.org/html/2307.07607v4#S3.SS2"


def _discover(source):
    root = Path(source)
    found = {}
    for imu in sorted(root.glob("**/imu_data/imu_data.csv")):
        folder = imu.parent.parent
        relative = folder.relative_to(root)
        # Only the real UGV release; never include simulated TartanAir records.
        parts = relative.parts
        events = [i for i, p in enumerate(parts)
                  if p in ("Final_Challenge", "Urban_Challenge")]
        if len(events) != 1:
            continue
        p = parts[events[0]:]
        if len(p) != 3 or p[1] not in ("UGV1", "UGV2", "UGV3"):
            continue
        gt = folder / "ground_truth/ground_truth_imu.csv"
        if gt.is_file():
            key = "_".join(p)
            if key in found:
                raise ValueError("duplicate SubT-MRS source recording identifier")
            found[key] = (folder, p)
    return found


def list_sequences(source):
    return sorted(_discover(source))


def official_splits(source):
    # The derivative's loader/docs do not publish train/test assignments.
    # This explicit initial catalogue is evaluation-only, not an author split.
    # Event + robot groups remain intact for a future derived training protocol.
    return {"test": list_sequences(source)}


def parse_sequence(folder, sequence_id, identity):
    folder = Path(folder)
    paths = [folder / "imu_data/imu_data.csv",
             folder / "ground_truth/ground_truth_imu.csv"]
    if any(p.is_symlink() for p in paths):
        raise ValueError("raw CSV symlink refused")
    ti, imu, hi = numeric_csv(paths[0], 11, integer_clock=True)
    tp, pose, hp = numeric_csv(paths[1], 17, integer_clock=True)
    if np.any(np.diff(ti) <= 0) or np.any(np.diff(tp) <= 0):
        raise ValueError("nonmonotonic integer nanosecond source clock")
    origin = int(ti[0])
    # Subtract as integer nanoseconds first to preserve a 5 ms sample interval
    # even when the absolute Unix epoch exceeds float64's integer precision.
    imu_time = (ti-origin).astype(np.float64) / 1e9
    pose_time = (tp-origin).astype(np.float64) / 1e9
    gyro, accel = imu[:, 4:7], imu[:, 7:10]
    position, q_xyzw, velocity = pose[:, :3], pose[:, 3:7], pose[:, 7:10]
    q = rig.xyzw_to_wxyz(q_xyzw)
    pose_valid = np.isfinite(position).all(1) & np.isfinite(q).all(1)
    pose_valid &= np.abs(np.linalg.norm(q, axis=1)-1) < 1e-3
    q_safe = rig.normalize_quaternions(np.where(pose_valid[:, None], q,
                                              [1., 0., 0., 0.]))
    event, robot, segment = identity
    attrs = {
        "subject_id": "not_applicable", "device_id": "epson_mg365_"+robot,
        "placement": "body", "source_placement": "robot_body", "source_platform": "ugv",
        "body_frame": "released_imu_frame",
        "group_id": event+"_"+robot, "recording_id": event+"_"+robot,
        "sequence_id_source": segment, "source_event": event,
        "position_source": "SubT_MRS_map_registration_multisensor_fusion_IMU_origin_release",
        "orientation_source": "SubT_MRS_map_fusion_reference_body_to_world",
        "device_orientation_source": "none",
        "source_files": ",".join(str(p.relative_to(folder)) for p in paths),
        "source_sha256": evidence(paths, folder), "source_license": LICENSE,
        "source_timestamp_origin_ns": str(origin), "start_time_unix": origin/1e9,
        "source_csv_headers": {"imu": hi, "pose": hp},
        "source_contract_evidence": SOURCE_CONTRACT,
        "reference_contract_evidence": REFERENCE_CONTRACT,
        "reference_is_estimator": True, "reference_incorporates_IMU": True,
        "reference_orientation_is_privileged": True,
        "reference_bias_columns_used": False,
        "reference_origin": "imu_origin_as_released_by_AirIMU",
        "derivative_release": "AirIMU segmented and adjusted SubT-MRS UGV CSV",
        "independent_source_id": "subt_mrs",
        "training_enabled": False, "split_is_author_official": False,
        "benchmark_track": "ugv_reference_catalogue",
        "capabilities": {"reference_orientation": True,
                         "device_orientation": False, "training": False},
    }
    notes = [
        "AirIMU loader specifies integer-ns clocks, xyzw reference, rad/s gyro and SI specific force",
        "IMU-origin reference is accepted as released; physical gravity/gyro/time checks remain mandatory",
        "Map-constrained visual/LiDAR/IMU fusion reference is correlated with input; it is not independent mocap",
        "No reference gyroscope/accelerometer bias subtraction or GT attitude input substitution",
        "Initial benchmark-derived test-only catalogue; all segments from one event/robot share a recording group",
    ]
    raw = RawSequence(sequence_id, imu_time, gyro, accel, pose_time,
                      position, q_safe, velocity=velocity,
                      imu_valid=np.isfinite(gyro).all(1)&np.isfinite(accel).all(1),
                      pose_valid=pose_valid, attrs=attrs, notes=notes)
    stats = rig.physical_checks(raw)
    failures, warnings = rig.evaluate_checks(stats)
    rig.apply_checks(raw, stats, failures, warnings)
    return raw


def iter_raw_sequences(source, only: Optional[Collection[str]] = None) -> Iterator[RawSequence]:
    found = _discover(source)
    for name in sorted(found) if only is None else sorted(only):
        try:
            folder, identity = found[name]
            yield parse_sequence(folder, name, identity)
        except (OSError, ValueError, KeyError) as exc:
            yield rig.rejected_sequence(name, f"source/time/frame contract: {exc}",
                                        {"source_license": LICENSE})
