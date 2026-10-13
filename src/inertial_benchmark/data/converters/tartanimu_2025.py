"""Pinned CVPR2025 TartanIMU retargetted NPZ release (official test only).

This is the labelled trajectory format, not the IROS2026 window challenge.
The author loader establishes seconds, SI IMU, xyzw body-to-world and z-up.
Published samples have no reliable original recording IDs: grouping is
conservatively by platform and never manufactures independent train/val data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Iterator, Optional

import numpy as np

from . import _rig_utils as rig
from ._external_utils import evidence, safe_orientation
from .base import RawSequence

NAME = "tartanimu_2025"
VERSION = "1.0.0"
LICENSE = "TartanIMU CVPR2025 sample data terms unspecified; code Apache-2.0 is not data relicensing"
SOURCE_REVISION = "3577944d7fcac2882d60c03fc2bfc83718a5b60e"
AUTHOR_COMMIT = "8a8025000f85d2c6b0692b88e15c2c91e761f7d7"
AUTHOR_LOADER = (
    f"https://github.com/superxslam/TartanIMU/blob/{AUTHOR_COMMIT}/"
    "tartan_imu/dataloader/dataset_AirLab.py"
)
_KEYS = ("retargetted_ts", "retargetted_imu", "retargetted_pos", "retargetted_quat")
_PLATFORMS = ("car", "dog", "drone", "human")


def _root(source: Path) -> Path:
    source = Path(source)
    for root in (source / "data/samples", source / "samples", source):
        if any((root / p / "test").is_dir() for p in _PLATFORMS):
            return root
    return source


def _discover(source: Path) -> dict:
    root = _root(source)
    return {
        "__".join(p.relative_to(root).with_suffix("").parts): p
        for platform in _PLATFORMS
        for p in sorted((root / platform / "test").glob("*.npz"))
    }


def list_sequences(source: Path) -> list:
    return sorted(_discover(source))


def official_splits(source: Path) -> dict:
    return {"test": list_sequences(source)}


def parse_sequence(path: Path, sequence_id: str, root: Path) -> RawSequence:
    with np.load(path, allow_pickle=False) as arrays:
        missing = set(_KEYS) - set(arrays.files)
        if missing:
            raise ValueError(f"missing labelled trajectory arrays: {sorted(missing)}")
        time = np.asarray(arrays[_KEYS[0]], dtype=np.float64)
        if time.ndim == 2 and time.shape[1] == 1:
            time = time[:, 0]
        imu = np.asarray(arrays[_KEYS[1]], dtype=np.float64)
        position = np.asarray(arrays[_KEYS[2]], dtype=np.float64)
        quaternion = np.asarray(arrays[_KEYS[3]], dtype=np.float64)
    n = len(time)
    if time.shape != (n,) or imu.shape != (n, 6) or position.shape != (n, 3):
        raise ValueError("expected ts(N,) or (N,1), imu(N,6), position(N,3)")
    if quaternion.shape != (n, 4) or n < 2 or not np.isfinite(time).all():
        raise ValueError("invalid timestamp/quaternion shape or nonfinite clock")
    positive_dt = np.diff(time)
    positive_dt = positive_dt[positive_dt > 0]
    if not len(positive_dt) or not 0.001 <= float(np.median(positive_dt)) <= 0.02:
        raise ValueError("clock does not follow the source seconds/100-250Hz contract")
    orientation, pose_valid = safe_orientation(rig.xyzw_to_wxyz(quaternion))
    pose_valid &= np.isfinite(position).all(axis=1)
    imu_valid = np.isfinite(imu).all(axis=1)
    platform = path.relative_to(root).parts[0]
    origin = float(time[0])
    attrs = {
        "subject_id": "unknown", "device_id": f"tartanimu_{platform}_retargetted",
        "placement": "retargetted_body", "group_id": f"platform_{platform}",
        "recording_id": "unknown", "motion_platform": platform,
        "position_source": "author_retargetted_reference_position",
        "orientation_source": "author_retargetted_reference_body_to_world",
        "device_orientation_source": "none", "body_frame": "author_retargetted_FLU",
        "source_files": path.relative_to(root).as_posix(),
        "source_sha256": evidence([path], root), "source_license": LICENSE,
        "source_revision": SOURCE_REVISION, "author_loader": AUTHOR_LOADER,
        "author_loader_evidence_lines": "79-119",
        "source_timestamp_origin_s": origin, "source_partition": "test",
        "source_retarged_preprocessing_preserved": True,
        "reference_origin_exact_imu_lever_arm_verified": False,
        "reference_orientation_is_privileged": True,
        "training_enabled": False, "benchmark_track": "multiplatform_external_test",
        "capabilities": {"reference_orientation": True, "device_orientation": False,
                         "training": False},
    }
    if origin > 1e8:
        attrs["start_time_unix"] = origin
    notes = [
        "Author retargetted IMU is accelerometer3 then gyro3; SI and gravity retained.",
        "Source xyzw body-to-world becomes wxyz; no reference velocity clipping applied.",
        "All source samples are official test; no fabricated train/val or recording IDs.",
        "Original recording identity is unknown; platform-level grouping is conservative.",
        "Retargetting/reference-origin provenance is retained; no exact IMU lever arm claimed.",
        "Source overlapping adapt clips and position jumps are not independent recordings.",
        "Native clock duplicate removal, gap handling and resampling belong to the pipeline.",
    ]
    return RawSequence(sequence_id, time - origin, np.nan_to_num(imu[:, 3:6]),
                       np.nan_to_num(imu[:, :3]), time - origin, position, orientation,
                       imu_valid=imu_valid, pose_valid=pose_valid, attrs=attrs, notes=notes)


def iter_raw_sequences(
    source: Path, only: Optional[Collection[str]] = None,
) -> Iterator[RawSequence]:
    root, found = _root(source), _discover(source)
    for name in sorted(found) if only is None else sorted(only):
        try:
            yield parse_sequence(found[name], name, root)
        except (OSError, ValueError, KeyError) as exc:
            yield rig.rejected_sequence(name, f"TartanIMU source contract: {exc}",
                                        {"source_license": LICENSE})
