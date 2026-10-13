"""Restricted target-train inputs for a future unpaired-domain training protocol.

This leaf is deliberately separate from ``Sequence`` and labelled task views. It
does not resolve dataset YAML aliases, read ``dataset.json``/QC reports, or hash
whole HDF5 files. Such provenance admission belongs to a separately authorized
caller. The audit lists every field read here, including structural attributes.
The NumPy items collate directly with PyTorch's ordinary map-style DataLoader.
No augmentation, target labels, pose-validity mask, or device yaw alignment is
constructed. This module is not wired into the ordinary training pipeline.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import h5py
import numpy as np

from ..utils.geometry import GRAVITY, quat_rotate

PathLike = Union[str, Path]
INPUT_MODES = ("body_imu", "reference_attitude_assisted")
GROUP_KINDS = ("recording", "session", "subject")
_FIELDS = ("timestamp", "imu/gyroscope", "imu/accelerometer", "valid/imu")
_ATTRS = (
    "schema_version", "dataset", "sequence_id", "sample_rate_hz", "timestamp_type",
    "accelerometer_type", "body_frame", "subject_id", "group_id", "recording_id", "session_id",
)
_ATTITUDE_ATTRS = ("world_frame", "orientation_convention", "orientation_source")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class TargetInputError(ValueError):
    """Target input access/admission is outside the declared scientific contract."""


@dataclass
class TargetAccessAudit:
    """Operation-level audit; not a sandbox against arbitrary user Python code."""

    events: list = field(default_factory=list)

    def record(self, path: Path, operation: str, field_name: str) -> None:
        self.events.append({"path": str(path), "operation": operation, "field": field_name})


@dataclass(frozen=True)
class TargetInputConfig:
    """Six-channel ``[gyro_xyz, acc_xyz]`` windows; no target/frame-label options.

    ``body_imu`` is raw specific force and cannot remove gravity without attitude.
    The assisted mode is explicitly privileged world-frame input, never pure IMU.
    """

    window: int = 400
    rate: float = 200.0
    mode: str = "body_imu"
    remove_gravity: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 2:
            raise TargetInputError("window must be an integer >= 2")
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise TargetInputError("rate must be finite and positive")
        if self.mode not in INPUT_MODES:
            raise TargetInputError(f"mode must be one of {INPUT_MODES}")
        if self.mode == "body_imu" and self.remove_gravity:
            raise TargetInputError("body_imu cannot remove gravity using hidden attitude")

    @property
    def allowed_fields(self) -> tuple:
        return _FIELDS + (("pose/orientation",) if self.mode != "body_imu" else ())


@dataclass(frozen=True)
class ReferenceAttitudeContract:
    """Caller-admitted reference-quaternion coverage, independent of pose labels.

    ``intervals`` are half-open sample intervals proven to contain actual reference
    attitude, not finite placeholders. ``evidence_sha256`` binds the caller's
    admission receipt; this reader does NOT open it or infer coverage from finite
    quaternion values/``valid/pose``. The caller must validate its provenance before
    constructing this contract. This is an attestation, not cryptographic trust.
    """

    sequence_id: str
    sample_count: int
    orientation_source: str
    intervals: tuple
    evidence_sha256: str

    def __post_init__(self) -> None:
        _sequence_id(self.sequence_id)
        if (isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int)
                or self.sample_count < 1):
            raise TargetInputError("attitude contract needs a positive sample_count")
        if not self.orientation_source or self.orientation_source.lower() in ("none", "unknown"):
            raise TargetInputError("attitude source must be a declared reference source")
        if not isinstance(self.evidence_sha256, str) or not _SHA.fullmatch(self.evidence_sha256):
            raise TargetInputError("attitude contract needs an admission-receipt SHA256")
        previous = 0
        for interval in self.intervals:
            if (not isinstance(interval, (tuple, list)) or len(interval) != 2
                    or any(isinstance(i, bool) or not isinstance(i, int) for i in interval)):
                raise TargetInputError("attitude intervals must be integer [start, stop) pairs")
            lo, hi = interval
            if not (previous <= lo < hi <= self.sample_count):
                raise TargetInputError("attitude intervals must be ordered, disjoint and in bounds")
            previous = hi
        if not self.intervals:
            raise TargetInputError("attitude contract has no proven reference coverage")
        object.__setattr__(self, "intervals", tuple(tuple(i) for i in self.intervals))


@dataclass(frozen=True)
class RecordingIdentity:
    """Explicit recording/session/subject identity from admitted source metadata.

    ``group_kind`` classifies a dataset's group_id; it is never inferred to be a
    subject or recording. IDs are namespaced by dataset. IMU hashes detect exact
    duplicated recordings across datasets, including changed sequence names.
    """

    dataset: str
    sequence_id: str
    group_id: str
    group_kind: str
    subject_id: str = ""
    recording_id: str = ""
    session_id: str = ""
    imu_sha256: str = ""

    def __post_init__(self) -> None:
        for key in ("dataset", "sequence_id", "group_id", "group_kind", "subject_id",
                    "recording_id", "session_id", "imu_sha256"):
            if not isinstance(getattr(self, key), str):
                raise TargetInputError(f"recording identity {key} must be text")
        if not self.dataset or not self.sequence_id or not self.group_id:
            raise TargetInputError("recording identity needs dataset, sequence and group IDs")
        if self.group_kind not in GROUP_KINDS:
            raise TargetInputError(
                "group_id requires explicit recording/session/subject classification")
        if self.imu_sha256 and not _SHA.fullmatch(self.imu_sha256):
            raise TargetInputError("invalid IMU content SHA256")


@dataclass(frozen=True)
class SourcePartitionGuard:
    """Source identities supplied without opening any source held-out HDF5 file.

    Same-subject *different* training recordings may be allowed explicitly. A
    recording/session match, duplicate IMU content, or any held-out group match is
    always rejected. The caller must supply all relevant held-out partitions;
    this leaf cannot prove completeness of an omitted source partition.
    """

    train: tuple
    held_out: tuple
    allow_shared_training_subjects: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", tuple(self.train))
        object.__setattr__(self, "held_out", tuple(self.held_out))
        if not isinstance(self.allow_shared_training_subjects, bool):
            raise TargetInputError("shared training subjects need an explicit boolean declaration")
        if not self.train or not self.held_out:
            raise TargetInputError("source_guard needs source train and held-out identities")
        if any(not isinstance(i, RecordingIdentity) for i in self.train + self.held_out):
            raise TargetInputError("source_guard entries must be RecordingIdentity values")
        if any(not i.imu_sha256 for i in self.train + self.held_out):
            raise TargetInputError(
                "source_guard needs admitted IMU hashes for cross-dataset duplicate checks")

    def check(self, target: RecordingIdentity) -> None:
        if not isinstance(target, RecordingIdentity) or not target.imu_sha256:
            raise TargetInputError("target guard check requires its actual admitted IMU hash")
        for source in self.train + self.held_out:
            same_namespace = source.dataset == target.dataset
            if same_namespace and source.sequence_id == target.sequence_id:
                raise TargetInputError("target recording is already in a source partition")
            if source.imu_sha256 and source.imu_sha256 == target.imu_sha256:
                raise TargetInputError("duplicate source/target IMU recording content")
            if same_namespace:
                for key in ("recording_id", "session_id"):
                    value = getattr(source, key)
                    if value and value == getattr(target, key):
                        raise TargetInputError(f"source/target share {key}")
                if source.group_kind != target.group_kind and source.group_id == target.group_id:
                    raise TargetInputError(
                        "matching group_id has inconsistent group classification")
                group_match = (source.group_kind == target.group_kind
                               and source.group_id == target.group_id)
                if group_match and source.group_kind in ("recording", "session"):
                    raise TargetInputError("source/target share a recording/session group")
                if source in self.held_out and (group_match or (
                        source.subject_id and source.subject_id == target.subject_id
                        and source.group_kind == "subject")):
                    raise TargetInputError("target crosses a protected source held-out group")
                same_subject = bool(source.subject_id and source.subject_id == target.subject_id)
                if (source in self.train and (same_subject or (
                        group_match and source.group_kind == "subject"))
                        and not self.allow_shared_training_subjects):
                    raise TargetInputError(
                        "shared training subject needs an explicit experiment declaration")


def _sequence_id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value) or ".." in value:
        raise TargetInputError(f"unsafe sequence ID: {value!r}")
    return value


def _inside(root: Path, relative: str) -> Path:
    """Require a real in-root path without symlink aliases (including train->val)."""
    candidate = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise TargetInputError(f"target path cannot contain a symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise TargetInputError(f"target path escapes admitted root: {relative}")
    return resolved


def _dataset(handle: h5py.File, name: str, allowed: tuple) -> h5py.Dataset:
    if name not in allowed:
        raise TargetInputError(f"forbidden target field: {name}")
    current: Any = handle
    for part in name.split("/"):
        if (not isinstance(current, (h5py.File, h5py.Group))
                or not isinstance(current.get(part, getlink=True), h5py.HardLink)):
            raise TargetInputError(f"target input must not follow soft/external HDF5 links: {name}")
        current = current[part]
    if not isinstance(current, h5py.Dataset) or current.is_virtual or current.external:
        raise TargetInputError(
            f"target input must be an internal materialized HDF5 dataset: {name}")
    return current


class UnlabelledInertialDataset:
    """Canonical target-train windows, with access isolation and no label keys.

    ``source_guard`` and ``group_kinds`` make separation policy explicit. Metadata
    reads are the canonical train split and the attribute allowlist only. Numeric
    QC and IMU content hashing read only allowed input fields; no whole-file hash,
    dataset manifest, conversion report, ``valid/pose`` or labelled view is read.
    Target val/test files are never opened, even for QC. The admitted root should
    be immutable for the run; stat guards reject accidental file replacement.
    """

    def __init__(
        self,
        root: PathLike,
        view: TargetInputConfig,
        *,
        source_guard: SourcePartitionGuard,
        group_kinds: Mapping[str, str],
        only: Optional[Sequence[str]] = None,
        stride: int = 200,
        attitude: Optional[Mapping[str, ReferenceAttitudeContract]] = None,
        audit: Optional[TargetAccessAudit] = None,
    ) -> None:
        if (not isinstance(view, TargetInputConfig)
                or not isinstance(source_guard, SourcePartitionGuard)):
            raise TargetInputError(
                "explicit target input config and source partition guard are required")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise TargetInputError("stride must be a positive integer")
        self.root = Path(root).expanduser().resolve(strict=True)
        self.view, self.stride = view, stride
        self.audit = audit if audit is not None else TargetAccessAudit()
        split = _inside(self.root, "splits/train.txt")
        self.audit.record(split, "metadata_read", "canonical_train_split")
        split_bytes = split.read_bytes()
        ids = [_sequence_id(s.strip()) for s in split_bytes.decode("utf-8").splitlines()
               if s.strip() and not s.lstrip().startswith("#")]
        if not ids or len(set(ids)) != len(ids):
            raise TargetInputError("canonical train split is empty or has duplicate sequence IDs")
        selected = ids if only is None else [_sequence_id(s) for s in only]
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(ids):
            raise TargetInputError("selected target IDs must be unique members of canonical train")
        self.train_split_sha256 = hashlib.sha256(split_bytes).hexdigest()
        self.sequence_ids = tuple(selected)
        self.paths, self.identities, self._signatures, self._valid = [], [], [], []
        self.access_contract = {
            "input_mode": view.mode, "input_fields": list(view.allowed_fields),
            "structural_attributes": list(
                _ATTRS + (_ATTITUDE_ATTRS if view.mode != "body_imu" else ())),
            "metadata_files": ["splits/train.txt"], "whole_file_hash": False,
            "pose_labels_or_pose_validity": False,
            "reference_attitude_assisted": view.mode != "body_imu",
        }
        self._attitude = dict(attitude or {})
        if view.mode == "body_imu" and self._attitude:
            raise TargetInputError("body_imu must not consume an attitude admission contract")
        if set(self._attitude) - set(selected):
            raise TargetInputError("attitude contract contains non-selected target IDs")
        index = []
        for k, sid in enumerate(selected):
            if sid not in group_kinds:
                raise TargetInputError(f"{sid}: explicit group classification is missing")
            path = _inside(self.root, f"sequences/{sid}.h5")
            before = self._signature(path)
            with h5py.File(path, "r") as handle:
                attrs = self._attrs(handle, path)
                if attrs.get("sequence_id") != sid:
                    raise TargetInputError(
                        f"{sid}: HDF5 sequence_id differs from canonical train ID")
                if (attrs.get("schema_version") != "1.0"
                        or attrs.get("timestamp_type") != "relative"):
                    raise TargetInputError(
                        f"{sid}: target reader requires native IPB v1 relative timestamps")
                if attrs.get("accelerometer_type") != "specific_force":
                    raise TargetInputError(f"{sid}: expected native specific-force accelerometer")
                rate = float(attrs.get("sample_rate_hz", float("nan")))
                if not math.isfinite(rate) or abs(rate - view.rate) > 1e-6:
                    raise TargetInputError(f"{sid}: sample rate differs from target input contract")
                valid, digest = self._scan_inputs(handle, path, attrs)
                identity = RecordingIdentity(
                    dataset=attrs.get("dataset", ""), sequence_id=sid,
                    group_id=attrs.get("group_id", ""), group_kind=group_kinds[sid],
                    subject_id=attrs.get("subject_id", ""),
                    recording_id=attrs.get("recording_id", ""),
                    session_id=attrs.get("session_id", ""), imu_sha256=digest,
                )
                source_guard.check(identity)
                for previous in self.identities:
                    if previous.imu_sha256 == digest:
                        raise TargetInputError("selected target sequences duplicate IMU content")
            if self._signature(path) != before:
                raise TargetInputError(f"{sid}: target file changed during admission")
            starts = np.arange(0, max(len(valid) - view.window + 1, 0), stride, dtype=np.int64)
            bad = np.concatenate(([0], np.cumsum(~valid)))
            starts = starts[(bad[starts + view.window] - bad[starts]) == 0]
            self.paths.append(path)
            self.identities.append(identity)
            self._signatures.append(before)
            self._valid.append(valid)
            index.extend((k, int(start)) for start in starts)
        self.index = np.asarray(index, dtype=np.int64).reshape(-1, 2)
        if not len(self.index):
            raise TargetInputError("target train has no admissible input windows")

    @staticmethod
    def _signature(path: Path) -> tuple:
        st = path.stat()
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def _attrs(self, handle: h5py.File, path: Path) -> dict:
        allowed = _ATTRS + (_ATTITUDE_ATTRS if self.view.mode != "body_imu" else ())
        attrs = {}
        for key in allowed:
            self.audit.record(path, "attribute_read", key)
            value = handle.attrs.get(key, "")
            attrs[key] = value.decode("utf-8") if isinstance(value, bytes) else value
        return attrs

    def _read(self, handle: h5py.File, path: Path, name: str, sl: slice) -> np.ndarray:
        dataset = _dataset(handle, name, self.view.allowed_fields)
        self.audit.record(path, "input_read", name)
        return np.asarray(dataset[sl])

    def _scan_inputs(self, handle: h5py.File, path: Path, attrs: dict) -> tuple:
        """QC/hash granted input bytes in chunks; never derive mask from pose labels."""
        shapes = {name: _dataset(handle, name, self.view.allowed_fields).shape
                  for name in self.view.allowed_fields}
        if len(shapes["timestamp"]) != 1:
            raise TargetInputError("timestamp must have shape (N,)")
        n = shapes["timestamp"][0]
        if n == 0:
            raise TargetInputError("target input sequence is empty")
        expected = {"timestamp": (n,), "valid/imu": (n,), "imu/gyroscope": (n, 3),
                    "imu/accelerometer": (n, 3), "pose/orientation": (n, 4)}
        if any(shapes[key] != expected[key] for key in shapes):
            raise TargetInputError("target input arrays have inconsistent shapes")
        valid = np.zeros(n, dtype=bool)
        previous_time = None
        for start in range(0, n, 65536):
            sl = slice(start, min(start + 65536, n))
            t = np.asarray(self._read(handle, path, "timestamp", sl), dtype=np.float64)
            v = np.asarray(self._read(handle, path, "valid/imu", sl), dtype=bool)
            if (not np.isfinite(t).all() or (previous_time is not None and
                    not np.isclose(t[0] - previous_time, 1 / self.view.rate, rtol=1e-6, atol=1e-9))
                    or not np.allclose(np.diff(t), 1 / self.view.rate, rtol=1e-6, atol=1e-9)):
                raise TargetInputError("target timestamp is not a finite uniform native grid")
            if start == 0 and (not len(t) or abs(t[0]) > 1e-9):
                raise TargetInputError("native target timestamp must start at zero")
            previous_time = t[-1]
            valid[sl] = v
        digest = hashlib.sha256()
        # Same float32 little-endian shape/bytes definition as native IMU hashes.
        for name in ("imu/gyroscope", "imu/accelerometer"):
            digest.update(repr(shapes[name]).encode())
            for start in range(0, n, 65536):
                sl = slice(start, min(start + 65536, n))
                values = np.ascontiguousarray(self._read(handle, path, name, sl), dtype="<f4")
                valid[sl] &= np.isfinite(values).all(axis=1)
                digest.update(values.tobytes())
        if self.view.mode != "body_imu":
            sid = path.stem
            contract = self._attitude.get(sid)
            if not isinstance(contract, ReferenceAttitudeContract) or contract.sequence_id != sid:
                raise TargetInputError(
                    f"{sid}: assisted input needs its admitted attitude coverage")
            if (contract.sample_count != n
                    or contract.orientation_source != attrs["orientation_source"]
                    or attrs["world_frame"] != "gravity_aligned_z_up"
                    or attrs["orientation_convention"] != "body_to_world_wxyz"):
                raise TargetInputError(
                    f"{sid}: attitude admission differs from native reference contract")
            coverage = np.zeros(n, dtype=bool)
            for lo, hi in contract.intervals:
                coverage[lo:hi] = True
            valid &= coverage
            for start in range(0, n, 65536):
                sl = slice(start, min(start + 65536, n))
                q = np.asarray(self._read(handle, path, "pose/orientation", sl), dtype=np.float64)
                healthy = (np.isfinite(q).all(axis=1)
                           & (np.abs(np.linalg.norm(q, axis=1) - 1) < 1e-3))
                valid[sl] &= healthy  # numerical health is not the provenance coverage proof
        return valid, digest.hexdigest()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        k, start = (int(v) for v in self.index[index])
        path = self.paths[k]
        if self._signature(path) != self._signatures[k]:
            raise TargetInputError(
                "admitted target source changed; rebuild under a new run identity")
        sl = slice(start, start + self.view.window)
        with h5py.File(path, "r") as handle:
            gyro = np.asarray(self._read(handle, path, "imu/gyroscope", sl), dtype=np.float64)
            acc = np.asarray(self._read(handle, path, "imu/accelerometer", sl), dtype=np.float64)
            t = np.asarray(self._read(handle, path, "timestamp", sl), dtype=np.float64)
            if self.view.mode != "body_imu":
                q = np.asarray(self._read(handle, path, "pose/orientation", sl), dtype=np.float64)
                gyro, acc = quat_rotate(q, gyro), quat_rotate(q, acc)
                if self.view.remove_gravity:
                    acc[:, 2] -= GRAVITY
        if self._signature(path) != self._signatures[k]:
            raise TargetInputError("target input changed during window read")
        imu = np.ascontiguousarray(np.concatenate((gyro, acc), axis=1).T, dtype=np.float32)
        return {"imu": imu,
                "timestamp": t, "input_mask": self._valid[k][sl].copy(), "seq": k, "start": start}
