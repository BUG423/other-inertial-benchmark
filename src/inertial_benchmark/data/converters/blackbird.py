"""Blackbird AirIO IMU/Vicon derivative; conversion/evaluation catalogue only.

The pinned author transform is Rx(pi) * q_NED_body * Rz(pi/2), with zero
lever arm. IMU order follows the author loader rather than the contradictory
CSV header. Published partitions from one route are never independent flights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Iterator, Optional

import numpy as np

from . import _rig_utils as rig
from ._external_utils import AIRIO_COMMIT, AIRIO_SOURCE, evidence, numeric_csv, safe_orientation
from .base import RawSequence

NAME = "blackbird"
VERSION = "1.0.0"
LICENSE = "Blackbird original dataset terms; AirIO code BSD license does not relicense data"
AUTHOR_LOADER = f"https://github.com/Air-IO/Air-IO/blob/{AIRIO_COMMIT}/datasets/BlackBirddataset.py"


def _root(source: Path) -> Path:
    source = Path(source)
    return source / "Blackbird" if (source / "Blackbird").is_dir() else source


def _discover(source: Path) -> dict:
    root = _root(source)
    return {"__".join(p.parent.relative_to(root).parts): p.parent
            for p in sorted(root.glob("**/imu_data.csv"))
            if (p.parent / "groundTruthPoses.csv").is_file()}


def list_sequences(source: Path) -> list:
    return sorted(_discover(source))


def official_splits(source: Path) -> dict:
    """All converted segments are a test catalogue; training is deliberately disabled."""
    return {"test": list_sequences(source)}


def parse_sequence(folder: Path, sequence_id: str, root: Path) -> RawSequence:
    imu, pose = folder / "imu_data.csv", folder / "groundTruthPoses.csv"
    ti, data, header = numeric_csv(imu, 7)
    tp_us, gt, pose_header = numeric_csv(pose, 8, integer_clock=True)
    origin = float(ti[0])
    ti_relative = ti - origin
    # Reference integer microseconds are offset before float conversion.
    origin_us = int(round(origin * 1e6))
    tp = (tp_us - origin_us) * 1e-6 + (origin_us * 1e-6 - origin)
    q_body, pose_valid = safe_orientation(gt[:, 3:7])
    q = rig.quat_multiply(rig.quat_multiply(np.array([0., 1., 0., 0.]), q_body),
                          np.array([np.sqrt(0.5), 0., 0., np.sqrt(0.5)]))
    position = gt[:, :3] * [1., -1., -1.]
    pose_valid &= np.isfinite(position).all(1)
    parts = folder.relative_to(root).parts
    partition = parts[0] if parts else "unknown"
    # Route-level grouping is more conservative than geometry+speed grouping.
    route = parts[1] if len(parts) > 1 else folder.name
    attrs = {
        "subject_id": "not_applicable", "device_id": "blackbird_quadrotor",
        "placement": "body", "group_id": f"route_{route}", "recording_id": f"route_{route}",
        "route_id": route, "source_partition": partition,
        "position_source": "Vicon_drone_body_zero_lever_arm_IMU_proxy",
        "orientation_source": "Vicon_body_with_pinned_author_body_IMU_rotation",
        "device_orientation_source": "none", "body_frame": "blackbird_imu",
        "source_files": ",".join(str(p.relative_to(root)) for p in [imu, pose]),
        "source_license": LICENSE, "source_sha256": evidence([imu, pose], root),
        "source_imu_header": header, "source_pose_header": pose_header,
        "source_timestamp_origin_s": origin, "start_time_unix": origin,
        "source_imu_column_policy": "gyro_then_accel_per_pinned_loader_header_is_inconsistent",
        "reference_lever_arm_m": [0., 0., 0.], "reference_lever_arm_exact": False,
        "author_loader": AUTHOR_LOADER, "author_loader_evidence_lines": "54-88,125-126,177-190",
        "author_loader_sha256": "cbd6b7f7a46488b56ba419b8820b37ed4d8fb2f41578037b7c81e5eb66ed7295",
        "airio_source_commit": AIRIO_COMMIT, "source_description": AIRIO_SOURCE,
        "training_enabled": False, "benchmark_track": "uav_conversion_catalogue",
        "reference_orientation_is_privileged": True,
        "capabilities": {"reference_orientation": True, "device_orientation": False,
                         "training": False, "control_thrust_used_as_input": False},
    }
    notes = [
        "IMU gyro3/accel3 follows pinned AirIO loader; source header labels accel3/gyro3 "
        "and is contradicted by values",
        "NED->z-up Rx(pi), q_IMU_to_world=Rx(pi)*q_body_to_NED*Rz(pi/2); "
        "zero lever arm is an author proxy",
        "native IMU seconds and Vicon integer microseconds retained; "
        "resampling delegated to v1 pipeline",
        "no source finite-difference/smoothed velocity injected; "
        "targets derive from positions in the selected v1 view",
        "all released partitions of a route share group_id; "
        "catalogue has no train/val and is not training-ready",
        "reference attitude is privileged; no deployable device orientation, "
        "no thrust/control input",
    ]
    return RawSequence(sequence_id, ti_relative, np.nan_to_num(data[:, :3]),
                       np.nan_to_num(data[:, 3:6]),
                       tp, position, q, imu_valid=np.isfinite(data).all(1), pose_valid=pose_valid,
                       attrs=attrs, notes=notes)


def iter_raw_sequences(
    source: Path, only: Optional[Collection[str]] = None
) -> Iterator[RawSequence]:
    root, found = _root(source), _discover(source)
    for name in sorted(found) if only is None else sorted(only):
        try:
            yield parse_sequence(found[name], name, root)
        except (OSError, ValueError, KeyError) as exc:
            yield rig.rejected_sequence(name, f"parse/frame error: {exc}",
                                        {"source_license": LICENSE})
