"""KITTI estimated-reference native derivative of the audited original OXTS sidecar.

Source is the lossless candidate audit directory, containing quality-manifest.json
and candidate/*.h5. Backward clocks create separate sequences in the SAME original
recording group. The shared dev/v1 pipeline supplies the 200 Hz grid, short-gap
interpolation, quaternion SLERP, long-gap masks and IMU FIR-support invalidation.
The original 100 Hz sidecar is never overwritten or substituted for actual-dt AI.
No vehicle-to-IMU lever arm is inferred; this is an OXTS GPS/INS reference track.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Collection, Iterator

import h5py
import numpy as np

from . import _kitti_raw100hz as raw
from .base import RawSequence

NAME = "kitti"
VERSION = "1.0.0"
LICENSE = "KITTI dataset terms: https://www.cvlibs.net/datasets/kitti/ (data not relicensed by AI code)"
SPLIT_DESCRIPTION = "preregistered engineering train0042/val0033/test0034, whole recording groups, not the complete paper protocol"


def load_candidate(source: Path, name: str) -> raw.Candidate:
    source = Path(source)
    report = json.loads((source / "quality-manifest.json").read_text())
    if report.get("state") != "complete_candidate_not_formally_admitted" or report.get("rows") != 41547:
        raise ValueError("An actually audited original three-recording sidecar is required")
    record = next(r for r in report["records"] if r["recording"] == name)
    path = source / "candidate" / (name + ".h5")
    if raw.sha256(path) != record["candidate_sha256"]:
        raise ValueError("Audited sidecar hash changed")
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema") != raw.SCHEMA or h5.attrs.get("formal_native_ready"):
            raise ValueError("Not the audited lossless sidecar schema")
        c = raw.analyze(h5["raw/packet30"][:], h5["raw/civil_time_ns"][:], h5["raw/source_index"][:])
        if not np.array_equal(c.clock_breaks, h5["quality/clock_break_bits"][:]):
            raise ValueError("Sidecar clock segmentation differs")
    return c


def branches(candidate: raw.Candidate) -> list[tuple[int, int]]:
    # Long positive gaps remain INSIDE a branch and are invalid on the output
    # grid. A nonpositive clock creates a new branch instead of dropping rows.
    split = np.flatnonzero(candidate.clock_breaks & (1 | 4 | 8)) + 1
    edges = [0, *split.tolist(), len(candidate.packet)]
    return list(zip(edges[:-1], edges[1:]))


def sequence_ids(source: Path) -> dict[str, tuple[str, int, int]]:
    result = {}
    for name in raw.SPLITS.values():
        c = load_candidate(source, name)
        for index, (start, stop) in enumerate(branches(c)):
            result[f"{name}_clock{index:02d}"] = (name, start, stop)
    return result


def list_sequences(source: Path) -> list[str]:
    return sorted(sequence_ids(source))


def official_splits(source: Path) -> dict[str, list[str]]:
    """Framework ABI name; these are explicitly preregistered, not official lists."""
    ids = sequence_ids(source)
    return {split: [key for key, (record, _, _) in ids.items() if record == name]
            for split, name in raw.SPLITS.items()}


def make_raw(candidate: raw.Candidate, name: str, start: int, stop: int,
             sequence_id: str, source: Path) -> RawSequence:
    c = candidate
    selected = slice(start, stop)
    origin = int(c.clock_ns[start])
    # Integer subtraction before float seconds preserves fractional ns. This
    # clock is RELATIVE and never automatically labelled as UTC Unix time.
    t = (c.clock_ns[selected] - origin).astype(np.float64) * 1e-9
    attrs = {"subject_id": "not_applicable", "device_id": "OXTS_RT3003",
      "placement": "body", "group_id": name, "recording_id": name,
      "body_frame": "KITTI_GPS_IMU_front_left_up",
      "position_source": "OXTS_GPS_INS_Mercator_estimate_shared_IMU",
      "orientation_source": "OXTS_GPS_INS_reference_RzRyRx",
      "device_orientation_source": "none", "source_license": LICENSE,
      "source_files": f"candidate/{name}.h5;quality-manifest.json",
      "source_sha256": {f"candidate/{name}.h5": raw.sha256(Path(source)/"candidate"/(name+".h5")),
                        "quality-manifest.json": raw.sha256(Path(source)/"quality-manifest.json")},
      "source_original_index_span_half_open": [int(c.source_index[start]), int(c.source_index[stop-1])+1],
      "source_original_civil_clock_origin_ns": origin,
      "source_timestamp_timezone": "unknown; no absolute UTC claim", "source_original_rate": "nonuniform original100Hz",
      "split_description": SPLIT_DESCRIPTION, "native_domain": "ground_vehicle",
      "source_world_chart": "translated Mercator east_north_up; not first-pose-rotated",
      "velocity_source_frame": "ve east, vn north, vu perpendicular to Earth surface (local tangent up)",
      "source_altitude_datum": "unknown", "physical_pose_output_anchor": "OXTS unit; antenna/output correction configuration unavailable",
      "reference_is_independent_ground_truth": False, "reference_shares_input_IMU": True,
      "reference_orientation_is_privileged": True, "reference_initial_velocity_is_privileged": True,
      "car_to_IMU_lever_arm": "unknown; not substituted from IMU-to-Velodyne calibration",
      "AI_vehicle_NHC_input_admission": "pending explicit car/IMU and original-clock protocol",
      "capabilities": {"reference_orientation": True, "device_orientation": False, "independent_reference": False},
      "quality_policy": "navstat4 and finite/nonnegative modes; primary GNSS mode0 retained and separately flagged; -1 upstream interpolation invalid",
    }
    notes = [SPLIT_DESCRIPTION, "original100Hz sidecar and civil ns retained separately; derive200Hz via shared v1 resampler",
      "clock backjumps split sequences within original recording group; positive gaps use valid masks, not valid trajectory bridges",
      "same-IMU GPS/INS estimated reference; privileged orientation/initial velocity; no independent truth claim",
      "IMU->Velodyne calibration does not provide vehicle->IMU R/t; no guessed mounting or lever-arm correction",
      "ENU tangent vu is source-provided; derivative-vs-p diagnostic is recorded, no silent fit/correction",
    ]
    return RawSequence(sequence_id, t, c.packet[selected, 17:20], c.packet[selected, 11:14],
                       t.copy(), c.position[selected], c.quaternion[selected],
                       velocity=c.packet[selected][:,[7,6,10]],
                       imu_valid=c.input_eligible[selected], pose_valid=c.reference_candidate[selected],
                       attrs=attrs, notes=notes)


def iter_raw_sequences(source: Path, only: Collection[str] | None = None) -> Iterator[RawSequence]:
    ids = sequence_ids(source)
    selected = set(ids) if only is None else set(only)
    cache = {}
    for sequence_id in sorted(selected):
        if sequence_id not in ids:
            raise ValueError("Unknown preregistered original recording/clock branch")
        name, start, stop = ids[sequence_id]
        if name not in cache:
            cache[name] = load_candidate(source, name)
        yield make_raw(cache[name], name, start, stop, sequence_id, source)


def grid_provenance(candidate: raw.Candidate, start: int, stop: int,
                    native_timestamp: np.ndarray) -> dict[str, np.ndarray]:
    """Discrete source endpoints/quality; no numerical interpolation of flags.

    native_timestamp is the absolute offset from this original clock branch's
    relative origin, including the common pipeline's leading trim offset.
    """
    ns = candidate.clock_ns[start:stop]
    t = (ns-int(ns[0])).astype(np.float64)*1e-9
    grid = np.asarray(native_timestamp,dtype=np.float64)
    right = np.searchsorted(t,grid,side="left").clip(0,len(t)-1)
    left = np.maximum(right-1,0)
    exact_right = np.abs(t[right]-grid)<=1e-6
    exact_left = np.abs(t[left]-grid)<=1e-6
    left = np.where(exact_right,right,left)
    right = np.where(exact_left,left,right)
    span = t[right]-t[left]
    alpha = np.divide(grid-t[left],span,out=np.zeros_like(grid),where=span>0)
    return {"source_left_index":candidate.source_index[start:stop][left],
            "source_right_index":candidate.source_index[start:stop][right],
            "source_left_civil_ns":ns[left], "source_right_civil_ns":ns[right],
            "linear_reference_alpha":alpha,
            "is_between_original_samples":left!=right,
            "bracket_gap_exceeds50ms":span>.05,
            "categorical_flags_left":candidate.packet[start:stop][left,25:30].astype(np.int32),
            "categorical_flags_right":candidate.packet[start:stop][right,25:30].astype(np.int32)}


def convert_derived(source: Path, output: Path) -> dict:
    """Standard 200 Hz IPB conversion plus discrete provenance, before final hashes.

    This writes a NEW native directory, never the existing formal registry.
    Source quality flags are stored as categorical endpoint pairs. Their
    values are never passed to the signal interpolator or SLERP.
    """
    import sys
    from ..convert import convert_dataset, write_dataset_files
    from ..format import load_sequence
    from ..manifest import sha256_file
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Derived native output already exists; no silent overwrite")
    convert_dataset(NAME, source, output, converter=sys.modules[__name__], workers=1,
                    rate=200., gap_threshold=.05, min_duration=2., strata=[])
    report = json.loads((output / "conversion_report.json").read_text())
    ids, cache = sequence_ids(source), {}
    for record in report["sequences"]:
        if record["status"] != "accepted":
            continue
        name, start, stop = ids[record["sequence_id"]]
        if name not in cache:
            cache[name] = load_candidate(source,name)
        path = output / record["file"]
        seq = load_sequence(path)
        offset = record.get("trimmed_s",[0.,0.])[0]
        provenance = grid_provenance(cache[name],start,stop,seq.timestamp+offset)
        with h5py.File(path,"r+") as h5:
            for key,value in provenance.items():
                h5.create_dataset("provenance/"+key,data=value,compression="gzip",shuffle=True)
            h5.create_dataset("provenance/effective_input_reference_valid",data=seq.valid,compression="gzip")
            h5.attrs["categorical_quality_interpolation"] = "none; discrete original left/right endpoint values"
            h5.attrs["original_sidecar_sha256"] = raw.sha256(source/"candidate"/(name+".h5"))
            h5.attrs["original_sidecar_retained"] = True
            h5.attrs["source_branch_grid_origin_civil_ns"] = int(cache[name].clock_ns[start])
            h5.attrs["derived_leading_trim_seconds"] = offset
        record.update(sha256=sha256_file(path),size_bytes=path.stat().st_size)
    opts={"rate":200.,"gap_threshold":.05,"compression":"gzip","min_duration":2.,"edge_min_run":1.}
    policy={"default":"preregistered_whole_recording_engineering",
            "official_splits_leak":False,"description":SPLIT_DESCRIPTION,
            "formal_registry_modified":False,"estimated_reference_shared_IMU":True,
            "raw100Hz_sidecar_retained":True,"resampling":"shared dev/v1; linear nominal-source uniformization + anti-imaging polyphase IMU; linear position/velocity + quaternion SLERP; invalid long gaps/FIR support"}
    return write_dataset_files(output,NAME,{"converter":NAME+"@"+VERSION,"license":LICENSE},
                               official_splits(source),report["sequences"],opts,
                               val_fraction=.1,seed=0,source=source,policy=policy,strata=[])
