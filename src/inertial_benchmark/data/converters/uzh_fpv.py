"""Official UZH-FPV Snapdragon text ZIPs with published v3 6DoF reference.

Only IMU/reference text is read; cameras and archive code are never extracted
or used as model input. The estimated reference attitude stays privileged.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Collection, Iterator, Optional
import zipfile

import numpy as np

from . import _rig_utils as rig
from ._external_utils import sha256_file
from .base import RawSequence

NAME = "uzh_fpv"
VERSION = "1.0.0"
LICENSE = "CC-BY-NC-SA-3.0"
FORMAT = "https://fpv.ifi.uzh.ch/2019-2020-submission-format/"
DATA_SOURCE = "https://fpv.ifi.uzh.ch/datasets/"
REFERENCE = "https://fpv.ifi.uzh.ch/#publication"


def _discover(source):
    root = Path(source)
    candidates = [root] if root.is_file() else sorted(root.glob("*_snapdragon_with_gt.zip"))
    return {p.name.removesuffix("_with_gt.zip"): p for p in candidates
            if re.fullmatch(r"(?:indoor|outdoor)_(?:forward|45)_\d+_snapdragon_with_gt\.zip", p.name)}


def list_sequences(source):
    return sorted(_discover(source))


def official_splits(source):
    # A public-reference recording catalogue is a benchmark-derived test track.
    # A single flight is never split into nominally independent train and test.
    return {"test": list_sequences(source)}


def text_arrays(text, columns, *, indexed=False):
    """Read decimal Unix seconds without rounding the epoch before subtraction."""
    times, values, headers = [], [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            headers.append(line)
            continue
        row = line.split()
        if len(row) != columns+(1 if indexed else 0):
            raise ValueError("unexpected text columns")
        if indexed:
            index = Decimal(row[0])
            if not index.is_finite() or index != len(times):
                raise ValueError("source sample index is not a consecutive integer")
            row = row[1:]
        try:
            t = Decimal(row[0])
            if not t.is_finite():
                raise ValueError("nonfinite source clock")
            times.append(t)
            values.append([float(v) for v in row[1:]])
        except (InvalidOperation, OverflowError) as exc:
            raise ValueError("invalid source number") from exc
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("missing or nonmonotonic source clock")
    return times, np.asarray(values, float), "\n".join(headers)


def source_text(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("raw ZIP symlink refused")
    with zipfile.ZipFile(path) as bundle:
        seen = set()
        for item in bundle.infolist():
            p = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (p.is_absolute() or ".." in p.parts or "\\" in item.filename
                    or re.match(r"^[A-Za-z]:", item.filename)
                    or stat.S_ISLNK(mode) or item.flag_bits & 1
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or str(p) in seen):
                raise ValueError("unsafe ZIP member")
            seen.add(str(p))
        out = {}
        for name in ("imu.txt", "groundtruth.txt"):
            item = bundle.getinfo(name)
            if item.file_size > 32*1024**2 or item.is_dir():
                raise ValueError("oversized or nonfile numeric text")
            out[name] = bundle.read(item).decode("utf-8")
    return out


def parse_sequence(path, sequence_id):
    text = source_text(path)
    # The source imu.txt header omits the leading sample index. Actual released
    # rows contain index + Unix timestamp + gyro3 + accel3; preserve that fact.
    ti, imu, hi = text_arrays(text["imu.txt"], 7, indexed=True)
    tp, pose, hp = text_arrays(text["groundtruth.txt"], 8)
    origin = ti[0]
    imu_time = np.asarray([float(t-origin) for t in ti])
    pose_time = np.asarray([float(t-origin) for t in tp])
    q = rig.xyzw_to_wxyz(pose[:, 3:7])
    valid_pose = np.isfinite(pose[:, :3]).all(1) & np.isfinite(q).all(1)
    valid_pose &= np.abs(np.linalg.norm(q, axis=1)-1) < 1e-3
    q_safe = rig.normalize_quaternions(np.where(valid_pose[:, None], q,
                                              [1., 0., 0., 0.]))
    attrs = {
        "subject_id": "not_applicable", "device_id": "qualcomm_snapdragon_flight",
        "placement": "body", "source_platform": "fpv_quadrotor",
        "group_id": sequence_id, "recording_id": sequence_id,
        "position_source": "UZH_FPV_v3_vision_based_6DoF_reference_Leica_validated",
        "orientation_source": "UZH_FPV_v3_vision_based_reference_IMU_to_world",
        "device_orientation_source": "none", "body_frame": "snapdragon_imu",
        "source_files": Path(path).name+":imu.txt,groundtruth.txt",
        "source_sha256": {Path(path).name: sha256_file(Path(path))},
        "source_member_sha256": {n: hashlib.sha256(t.encode()).hexdigest()
                                 for n, t in text.items()},
        "source_license": LICENSE, "start_time_unix": float(origin),
        "source_timestamp_origin_decimal_s": str(origin),
        "source_txt_headers": {"imu": hi, "pose": hp},
        "source_imu_header_omits_sample_index": True,
        "source_pose_convention_evidence": FORMAT,
        "reference_version": "official_2022_updated_v3_with_gt",
        "reference_is_estimator": True, "reference_orientation_is_privileged": True,
        "reference_is_independent_mocap_orientation": False,
        "reference_origin": "Snapdragon_IMU_as_specified_by_official_pose_format",
        "source_reference_provenance": REFERENCE,
        "training_enabled": False, "split_is_author_official": False,
        "independent_source_id": "uzh_fpv", "benchmark_track": "uav_reference_catalogue",
        "capabilities": {"reference_orientation": True,
                         "device_orientation": False, "training": False},
    }
    notes = [
        "Official text ZIP: indexed IMU Unix seconds, rad/s and SI specific force; reference seconds/metres/xyzw",
        "GT IMU origin/body-to-world follows the official pose format; no extra camera or prism lever arm applied",
        "Updated v3 reference is vision-based full-pose estimation validated against Leica positions, not independent mocap attitude",
        "No reference orientation or fitted bias substituted as deployable IMU attitude/calibration",
        "One complete public flight is one recording group; initial benchmark-derived evaluation-only catalogue",
        "Drone-racing speed bounds replace pedestrian limits; published Snapdragon sequences reach 20.73m/s",
    ]
    raw = RawSequence(sequence_id, imu_time, imu[:, :3], imu[:, 3:6],
                      pose_time, pose[:, :3], q_safe,
                      imu_valid=np.isfinite(imu).all(1), pose_valid=valid_pose,
                      attrs=attrs, notes=notes)
    stats = rig.physical_checks(raw)
    failures, warnings = rig.evaluate_checks(stats, limits={
        "speed_median_max": 25., "speed_p95_max": 35.})
    rig.apply_checks(raw, stats, failures, warnings)
    return raw


def iter_raw_sequences(source, only: Optional[Collection[str]] = None) -> Iterator[RawSequence]:
    found = _discover(source)
    for name in sorted(found) if only is None else sorted(only):
        try:
            yield parse_sequence(found[name], name)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            yield rig.rejected_sequence(name, f"source/time/frame contract: {exc}",
                                        {"source_license": LICENSE})
