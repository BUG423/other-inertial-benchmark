"""GrandTour ADIS IMU with mission-specific CPT7 GNSS/INS reference.

Zarr-v2 arrays are read directly from the author's tar releases without
extracting files or requiring ROS. The released ADIS orientation is absent;
reference orientation remains privileged and never becomes device attitude.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
from typing import Collection, Iterator, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from . import _rig_utils as rig
from .base import RawSequence

NAME = "grandtour"
VERSION = "1.0.0"
LICENSE = "MIT (author Hugging Face dataset card)"
AUTHOR = "https://github.com/leggedrobotics/grand_tour_dataset"
TF_CONVENTION = AUTHOR + "/blob/main/examples_ros1/grand_tour_ros1/README.md"
ZARR_CONVENTION = AUTHOR + "/blob/main/examples_hugging_face/src/grand_tour/zarr_transforms.py"


class TarArrays:
    """Bounded numeric Zarr-v2 reader; links, traversal and object codecs fail closed."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("raw tar symlink refused")
        self.tar = tarfile.open(self.path, "r:*")
        self.members = {}
        for member in self.tar.getmembers():
            p = PurePosixPath(member.name)
            if p.is_absolute() or ".." in p.parts or member.issym() or member.islnk():
                self.close()
                raise ValueError("unsafe tar member")
            if not member.isfile():
                continue
            name = p.as_posix()
            if name in self.members or member.size > 512 * 1024**2:
                self.close()
                raise ValueError("duplicate or oversized tar member")
            self.members[name] = member

    def close(self):
        self.tar.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def bytes(self, name):
        with self.tar.extractfile(self.members[name]) as stream:
            return stream.read()

    def metadata(self, topic):
        return json.loads(self.bytes(topic + "/.zattrs"))

    def array(self, topic, field):
        prefix = topic + "/" + field + "/"
        meta = json.loads(self.bytes(prefix + ".zarray"))
        dtype = np.dtype(meta["dtype"])
        shape, chunks = tuple(meta["shape"]), tuple(meta["chunks"])
        if (meta.get("zarr_format") != 2 or dtype.hasobject
                or dtype.kind not in "fiu" or meta.get("filters")
                or meta.get("order") != "C" or len(shape) != len(chunks)
                or not 1 <= len(shape) <= 3 or any(s <= 0 for s in shape + chunks)
                or math.prod(shape) * dtype.itemsize > 512 * 1024**2
                or math.prod(chunks) * dtype.itemsize > 512 * 1024**2):
            raise ValueError("unsupported/unsafe numeric Zarr layout")
        from numcodecs import get_codec

        compressor = meta.get("compressor")
        if compressor and compressor.get("id") not in ("blosc", "zlib", "gzip"):
            raise ValueError("unsupported compressor")
        codec = get_codec(compressor) if compressor else None
        out = np.empty(shape, dtype=dtype)
        grid = [range((s + c - 1) // c) for s, c in zip(shape, chunks)]
        separator = meta.get("dimension_separator", ".")
        if separator not in (".", "/"):
            raise ValueError("unsupported dimension separator")
        for index in itertools.product(*grid):
            raw = self.bytes(prefix + separator.join(map(str, index)))
            decoded = codec.decode(raw) if codec else raw
            values = np.frombuffer(decoded, dtype=dtype)
            if values.size != math.prod(chunks):
                raise ValueError("chunk decoded size differs from declared padded shape")
            values = values.reshape(chunks)
            sizes = [min(c, s - i*c) for i, s, c in zip(index, shape, chunks)]
            src = tuple(slice(0, n) for n in sizes)
            dst = tuple(slice(i*c, i*c+n) for i, c, n in zip(index, chunks, sizes))
            out[dst] = values[src]
        return out


def _discover(source):
    root = Path(source)
    folders = [root, *sorted(root.iterdir())] if root.is_dir() else []
    return {p.name: p for p in folders
            if p.is_dir() and (p / "data/adis_imu.tar").is_file()
            and (p / "data/cpt7_ie_tc_odometry.tar").is_file()
            and (p / "data/tf.tar").is_file()}


def list_sequences(source):
    return sorted(_discover(source))


def official_splits(source):
    # A representative mission catalogue provides held-out evaluation only.
    # Never manufacture a training split by partitioning one trajectory.
    return {"test": list_sequences(source)}


def _edge(tf, name):
    edge = tf[name]
    if edge["base_frame_id"] != "box_base" or edge["child_frame_id"] != name:
        raise ValueError("only explicit common box_base extrinsics are supported")
    q = np.array([edge["rotation"][k] for k in ("x", "y", "z", "w")], float)
    p = np.array([edge["translation"][k] for k in ("x", "y", "z")], float)
    if not np.isfinite(q).all() or not np.isfinite(p).all() or abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError("invalid mission TF")
    return Rotation.from_quat(q), p


def compose_reference(position, quaternion_xyzw, tf):
    """T_world_ADIS = T_world_CPT7 inv(T_box_CPT7) T_box_ADIS.

    ROS TF encodes the child origin/axes in the parent, so the rotation maps
    child coordinates into parent coordinates. Apply the rotating translation
    as well as the mounting rotation; position is the actual ADIS origin.
    """
    r_box_cpt, p_box_cpt = _edge(tf, "cpt7_imu")
    r_box_adis, p_box_adis = _edge(tf, "adis16475_imu")
    r_cpt_adis = r_box_cpt.inv() * r_box_adis
    p_cpt_adis = r_box_cpt.inv().apply(p_box_adis-p_box_cpt)
    r_world_cpt = Rotation.from_quat(quaternion_xyzw)
    p_world_adis = np.asarray(position, float) + r_world_cpt.apply(p_cpt_adis)
    q_world_adis = rig.from_rotation(r_world_cpt * r_cpt_adis)
    return p_world_adis, q_world_adis, p_cpt_adis


def parse_mission(folder, sequence_id):
    folder = Path(folder)
    with TarArrays(folder / "data/adis_imu.tar") as imu:
        imu_meta = imu.metadata("adis_imu")
        ti = imu.array("adis_imu", "timestamp").astype(float)
        gyro = imu.array("adis_imu", "ang_vel").astype(float)
        accel = imu.array("adis_imu", "lin_acc").astype(float)
    with TarArrays(folder / "data/cpt7_ie_tc_odometry.tar") as pose:
        pose_meta = pose.metadata("cpt7_ie_tc_odometry")
        tp = pose.array("cpt7_ie_tc_odometry", "timestamp").astype(float)
        p = pose.array("cpt7_ie_tc_odometry", "pose_pos").astype(float)
        q = pose.array("cpt7_ie_tc_odometry", "pose_orien").astype(float)
    with TarArrays(folder / "data/tf.tar") as transforms:
        tf = transforms.metadata("tf")["tf"]
    if imu_meta["frame_id"] != "adis16475_imu" or pose_meta["frame_id"] != "enu_origin":
        raise ValueError("unsupported source IMU/reference frame")
    if not np.isfinite(ti).all() or not np.isfinite(tp).all() or len(ti)<2 or len(tp)<2:
        raise ValueError("missing finite source timestamps")
    if np.any(np.diff(ti)<=0) or np.any(np.diff(tp)<=0):
        raise ValueError("nonmonotonic source timestamps")
    valid_pose = np.isfinite(p).all(1) & np.isfinite(q).all(1)
    valid_pose &= np.abs(np.linalg.norm(q,axis=1)-1)<1e-3
    # Mask invalid poses, using a harmless temporary quaternion only in the
    # calculation; it remains pose_valid=False and is never a valid label.
    q_safe=q.copy();q_safe[~valid_pose]=[0,0,0,1]
    position, orientation, lever = compose_reference(p, q_safe, tf)
    origin=float(ti[0])
    paths=[folder/'data'/name for name in ('adis_imu.tar','cpt7_ie_tc_odometry.tar','tf.tar')]
    attrs={
        "subject_id":"not_applicable","device_id":"anymal_d_adis16475_2",
        "placement":"body","group_id":sequence_id,"recording_id":sequence_id,
        "position_source":"CPT7_Inertial_Explorer_GNSS_INS_tightly_coupled_ADIS_lever_arm",
        "orientation_source":"CPT7_GNSS_INS_reference_with_mission_TF_ADIS_mounting",
        "device_orientation_source":"none","body_frame":"adis16475_imu",
        "source_files":','.join('data/'+p.name for p in paths),"source_license":LICENSE,
        "source_sha256":json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},sort_keys=True),
        "start_time_unix":origin,"source_timestamp_origin_s":origin,
        "reference_lever_arm_m":lever.tolist(),"reference_lever_arm_exact":True,
        "reference_origin":"adis16475_imu","source_reference_origin":"cpt7_imu",
        "reference_orientation_is_privileged":True,"reference_is_estimator":True,
        "training_enabled":False,"benchmark_track":"legged_robot_reference_catalogue",
        "tf_convention_evidence":TF_CONVENTION,"quaternion_order_evidence":ZARR_CONVENTION,
        "capabilities":{"reference_orientation":True,"device_orientation":False,"training":False},
    }
    notes=["ADIS SI specific force and rad/s kept on original Unix-second clock; native resampling delegated to v1 pipeline",
           "CPT7 GNSS/INS pose is an estimator reference, not the independent Leica prism trajectory",
           "ROS xyzw reference becomes wxyz; T_world_ADIS=T_world_CPT7*inv(T_box_CPT7)*T_box_ADIS with rotating lever arm",
           "ADIS all-zero source orientation is unavailable; no device orientation supplied",
           "representative missions are test-only, with whole-mission group_id"]
    raw=RawSequence(sequence_id,ti-origin,gyro,accel,tp-origin,position,orientation,
                    imu_valid=np.isfinite(gyro).all(1)&np.isfinite(accel).all(1),
                    pose_valid=valid_pose,attrs=attrs,notes=notes)
    stats=rig.physical_checks(raw);failures,warnings=rig.evaluate_checks(stats)
    rig.apply_checks(raw,stats,failures,warnings)
    return raw


def iter_raw_sequences(source, only: Optional[Collection[str]]=None) -> Iterator[RawSequence]:
    found=_discover(source)
    for name in sorted(found) if only is None else sorted(only):
        try:
            yield parse_mission(found[name],name)
        except (OSError,ValueError,KeyError,tarfile.TarError) as exc:
            yield rig.rejected_sequence(name,f"source/time/frame contract: {exc}",{"source_license":LICENSE})
