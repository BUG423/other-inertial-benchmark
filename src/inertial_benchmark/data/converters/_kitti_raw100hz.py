"""Lossless KITTI OXTS candidate conversion, outside the uniform IPB registry.

The original unsynchronised 100 Hz packets, civil nanosecond clock, source
indices and bad intervals are retained. This module does not register a dataset,
resample, sort timestamps, fix gaps, estimate a mounting transform or grant an
AI-IMU-DR input permission. GPS/INS reference and initial state are privileged.

Field units/axes: KITTI's official devkit readme and Geiger et al., IJRR 2013.
Mercator + Rz(yaw) Ry(pitch) Rx(roll) follows the released AI-IMU-DR parser;
the world is east/north/up, translated to the first finite geographic sample.
It is not the official MATLAB demo's first-pose-rotated coordinate system.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

VERSION = "0.1.0-candidate"
SCHEMA = "ib.kitti.raw100hz.candidate.v1"
FIELDS = ("lat", "lon", "alt", "roll", "pitch", "yaw", "vn", "ve", "vf", "vl", "vu",
          "ax", "ay", "az", "af", "al", "au", "wx", "wy", "wz", "wf", "wl", "wu",
          "pos_accuracy", "vel_accuracy", "navstat", "numsats", "posmode", "velmode", "orimode")
SPLITS = {"train": "2011_10_03_drive_0042_extract",
          "val": "2011_09_30_drive_0033_extract", "test": "2011_09_30_drive_0034_extract"}
REASONS = {1: "invalid_packet_or_clock", 2: "nonfinite_packet",
           4: "invalid_geographic_reference", 8: "nonintegral_quality_flags",
           16: "upstream_interpolated_or_invalid_mode", 32: "navigation_not_locked"}
CLOCK_BREAKS = {1: "nonpositive_clock_interval", 2: "clock_gap_above50ms",
                4: "source_index_gap", 8: "invalid_clock_endpoint"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def civil_nanoseconds(text: str) -> int:
    """Lossless naive civil coordinate; NOT an asserted UTC Unix timestamp."""
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d{1,9})", text)
    if match is None:
        raise ValueError("Original civil timestamp must have 1..9 fractional digits")
    second = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S")
    delta = second - datetime(1970, 1, 1)
    ns = (delta.days * 86400 + delta.seconds) * 10**9 + int(match[2].ljust(9, "0"))
    if not np.iinfo(np.int64).min < ns <= np.iinfo(np.int64).max:
        raise ValueError("Civil coordinate does not fit int64")
    return ns


def pose_enu(packet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ENU Mercator position and wxyz body-to-ENU quaternion; no lever-arm fix."""
    packet = np.asarray(packet, dtype=np.float64)
    valid = np.isfinite(packet[:, :6]).all(1) & (np.abs(packet[:, 0]) < 90)
    if not valid.any():
        raise ValueError("No finite geographic reference origin")
    origin = int(np.flatnonzero(valid)[0])
    scale = np.cos(np.deg2rad(packet[origin, 0]))
    p = np.full((len(packet), 3), np.nan)
    lat, lon, alt = packet[valid, :3].T
    p[valid] = np.stack((scale * 6378137.0 * np.deg2rad(lon),
                        scale * 6378137.0 * np.log(np.tan(np.deg2rad(90 + lat) / 2)), alt), 1)
    p -= p[origin]
    roll, pitch, yaw = (packet[:, 3:6] / 2).T
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    q = np.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                  cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), 1)
    return p, q


def rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), 1).reshape(-1, 3, 3)


def spans(mask: np.ndarray, breaks: np.ndarray) -> list[tuple[int, int]]:
    """Half-open row spans; no join across bad rows, clocks or source-index gaps."""
    result, start = [], None
    for index, valid in enumerate(mask):
        if start is not None and (not valid or (index and breaks[index - 1])):
            result.append((start, index))
            start = None
        if valid and start is None:
            start = index
    if start is not None:
        result.append((start, len(mask)))
    return result


def distance_pairs(position: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Released pair indexing: downsample10, floor(N_down/10)-1 starts, 100..800m.

    Output columns are source ROW offsets start/end, requested path metres,
    actual chord metres. No cross-segment pairs or empty-objective substitution.
    """
    p = np.asarray(position[start:stop:10], dtype=np.float64)
    if len(p) == 0 or not np.isfinite(p).all():
        return np.empty((0, 4), dtype=np.float64)
    distance = np.r_[0., np.linalg.norm(np.diff(p, axis=0), axis=1).cumsum()]
    rows = []
    for k in range(max(0, len(p) // 10 - 1)):
        first = 10 * k
        for length in range(100, 801, 100):
            threshold = distance[first] + length
            if threshold > distance[-1]:
                continue
            last = int(np.searchsorted(distance[first:], threshold)) + first
            chord = float(np.linalg.norm(p[last] - p[first]))
            if chord > 0:
                rows.append((start + first * 10, start + last * 10, length, chord))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


@dataclass
class Candidate:
    packet: np.ndarray
    clock_ns: np.ndarray
    clock_text: list[str]
    source_index: np.ndarray
    packet_text: list[str]
    packet_sha256: list[str]
    reasons: np.ndarray
    clock_breaks: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    input_eligible: np.ndarray
    reference_candidate: np.ndarray
    gnss_supported: np.ndarray


def analyze(packet: np.ndarray, clock_ns: np.ndarray, source_index: np.ndarray,
            *, parse_ok: np.ndarray | None = None) -> Candidate:
    packet = np.asarray(packet, dtype=np.float64)
    clock_ns, source_index = np.asarray(clock_ns), np.asarray(source_index)
    n = len(packet)
    if packet.shape != (n, 30) or clock_ns.shape != (n,) or source_index.shape != (n,):
        raise ValueError("Exactly thirty fields and aligned original clock/source indices required")
    if clock_ns.dtype != np.int64 or source_index.dtype != np.int64 or not n:
        raise ValueError("Nonempty int64 original clock and indices required")
    if np.any(source_index < 0) or np.any(np.diff(source_index) <= 0):
        raise ValueError("Original source indices must be unique and increasing; never sort/reindex")
    reasons = np.zeros(n, np.uint16)
    parse_ok = np.ones(n, bool) if parse_ok is None else np.asarray(parse_ok, bool)
    if parse_ok.shape != (n,):
        raise ValueError("Parse validity layout mismatch")
    reasons[~parse_ok | (clock_ns == np.iinfo(np.int64).min)] |= 1
    finite = np.isfinite(packet).all(1)
    reasons[~finite] |= 2
    geo = np.isfinite(packet[:, :3]).all(1) & (np.abs(packet[:, 0]) < 90) & (np.abs(packet[:, 1]) <= 180)
    reasons[~geo] |= 4
    flags = packet[:, 25:]
    integral = np.isfinite(flags).all(1) & (flags == np.floor(flags)).all(1)
    reasons[~integral] |= 8
    # KITTI documents that -1 in the final three fields marks linear upstream
    # interpolation of ALL values; original packet files are not necessarily raw IMU.
    reasons[(packet[:, 27:30] < 0).any(1)] |= 16
    reasons[packet[:, 25] != 4] |= 32
    breaks = np.zeros(max(0, n-1), np.uint8)
    clock_valid = clock_ns != np.iinfo(np.int64).min
    good_edge = clock_valid[:-1] & clock_valid[1:]
    delta = np.zeros(max(0, n-1), np.int64)
    delta[good_edge] = clock_ns[1:][good_edge] - clock_ns[:-1][good_edge]
    breaks[good_edge & (delta <= 0)] |= 1
    breaks[good_edge & (delta > 50_000_000)] |= 2
    breaks[np.diff(source_index) != 1] |= 4
    breaks[~good_edge] |= 8
    p, q = pose_enu(packet)
    input_eligible = parse_ok & clock_valid & integral & finite & ((reasons & 16) == 0)
    reference = input_eligible & geo & (packet[:, 25] == 4)
    # Mode0 is lack of the PRIMARY GNSS measurement, not proof that the locked
    # INS solution is invalid. Expose this stricter sensitivity mask separately.
    gnss = reference & (packet[:, 27:30] > 0).all(1)
    return Candidate(packet, clock_ns.copy(), [], source_index.copy(), [], [], reasons, breaks,
                     p, q, input_eligible, reference, gnss)


def read_record(raw_root: Path, name: str, member_manifest: dict) -> Candidate:
    if name not in SPLITS.values():
        raise ValueError("Recording is not in the preregistered train/val/test whitelist")
    date = name[:10]
    root = raw_root / date / name / "oxts"
    paths = sorted((root / "data").glob("*.txt"))
    indices = np.array([int(p.stem) for p in paths], dtype=np.int64)
    clock_path = root / "timestamps.txt"
    clocks = clock_path.read_bytes()
    rel = clock_path.relative_to(raw_root).as_posix()
    proof = member_manifest[rel]
    if len(clocks) != proof["bytes"] or hashlib.sha256(clocks).hexdigest() != proof["sha256"]:
        raise ValueError("Original clock differs from acquisition member proof")
    texts = clocks.decode("ascii").splitlines()
    if len(paths) != len(texts) or not paths:
        raise ValueError("Packet/clock row alignment differs; refuse guessed alignment")
    packets = np.full((len(paths), 30), np.nan)
    ns = np.full(len(paths), np.iinfo(np.int64).min, dtype=np.int64)
    parse_ok, hashes, rawtexts = np.ones(len(paths), bool), [], []
    for i, path in enumerate(paths):
        raw = path.read_bytes()
        rel = path.relative_to(raw_root).as_posix()
        proof = member_manifest[rel]
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != proof["bytes"] or digest != proof["sha256"]:
            raise ValueError("Original packet differs from acquisition proof: " + rel)
        hashes.append(digest)
        text = raw.decode("ascii")
        rawtexts.append(text)
        try:
            values = [float(x) for x in text.split()]
            if len(values) != 30:
                raise ValueError("Not thirty fields")
            packets[i] = values
            ns[i] = civil_nanoseconds(texts[i])
        except ValueError:
            parse_ok[i] = False
    result = analyze(packets, ns, indices, parse_ok=parse_ok)
    result.clock_text, result.packet_text, result.packet_sha256 = texts, rawtexts, hashes
    return result


def _segment_reports(candidate: Candidate, mask: np.ndarray) -> list[dict]:
    result = []
    for start, stop in spans(mask, candidate.clock_breaks):
        p = candidate.position[start:stop]
        pairs = distance_pairs(candidate.position, start, stop)
        # Fixed 6000-sample chunks, every 100 source rows (one nominal second).
        # These are feasibility statistics only, not an approved training sampler.
        chunks = {str(length): 0 for length in range(100, 801, 100)}
        distance = np.r_[0., np.linalg.norm(np.diff(p, axis=0), axis=1).cumsum()]
        for offset in range(0, max(0, stop-start-6000+1), 100):
            item = distance_pairs(candidate.position, start+offset, start+offset+6000)
            for length in chunks:
                chunks[length] += int(np.any(item[:, 2] == int(length)))
        result.append({"row_span_half_open": [start, stop],
            "source_index_first_last": [int(candidate.source_index[start]), int(candidate.source_index[stop-1])],
            "samples": stop-start, "duration_seconds": int(candidate.clock_ns[stop-1]-candidate.clock_ns[start])/1e9,
            "path_m": float(distance[-1]), "has_at_least6000_original_samples": stop-start >= 6000,
            "released_pair_counts_by_requested_m": {str(k): int(np.sum(pairs[:, 2] == k)) for k in range(100,801,100)},
            "fixed6000_chunks_with_pairs_by_requested_m": chunks})
    return result


def quality_report(candidate: Candidate) -> dict:
    c = candidate
    delta = np.diff(c.clock_ns)
    positive = delta[delta > 0]
    finite_edges = (c.clock_breaks == 0) & c.reference_candidate[:-1] & c.reference_candidate[1:]
    R = rotation(c.quaternion)
    world_acc = np.einsum("nij,nj->ni", R, c.packet[:, 11:14])
    velocity = c.packet[:, [7, 6, 10]]
    keep = finite_edges & (delta >= 5_000_000)
    dt = delta[keep] / 1e9
    vel_residual = np.linalg.norm(np.diff(c.position, axis=0)[keep]/dt[:,None] - (velocity[:-1][keep]+velocity[1:][keep])/2, axis=1)
    acc_residual = np.linalg.norm(np.diff(velocity, axis=0)[keep]/dt[:,None] - (world_acc[:-1][keep]+world_acc[1:][keep])/2 + [0,0,9.80655], axis=1)
    def percentiles(values):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        return {"median": float(np.median(values)), "p95": float(np.percentile(values,95))} if len(values) else None
    return {"rows": len(c.packet), "field_count": 30,
        "clock": {"civil_timezone": "unknown; no absolute UTC claim", "int64_nanoseconds_retained": True,
          "median_positive_dt_ns": float(np.median(positive)) if len(positive) else None,
          "median_positive_rate_hz": 1e9/float(np.median(positive)) if len(positive) else None,
          "break_counts": {v: int(np.sum((c.clock_breaks & k) != 0)) for k,v in CLOCK_BREAKS.items()},
          "broken_edges": [{"left_source_index": int(c.source_index[i]), "right_source_index": int(c.source_index[i+1]),
                            "delta_ns": int(delta[i]), "reason_bits": int(c.clock_breaks[i])}
                           for i in np.flatnonzero(c.clock_breaks)]},
        "row_reason_counts": {v:int(np.sum((c.reasons & k) != 0)) for k,v in REASONS.items()},
        "quality_field_counts": {FIELDS[i]: {str(int(v)): int(np.sum(c.packet[:, i] == v))
                                   for v in np.unique(c.packet[:, i]) if np.isfinite(v)} for i in range(25,30)},
        "accuracy_fields": {FIELDS[i]: percentiles(c.packet[:, i]) for i in (23,24)},
        "input_eligible_rows": int(c.input_eligible.sum()), "ins_reference_candidate_rows": int(c.reference_candidate.sum()),
        "primary_gnss_all_modes_positive_rows": int(c.gnss_supported.sum()),
        "clock_contiguous_segments": _segment_reports(c, np.ones(len(c.packet), bool)),
        "ins_candidate_contiguous_segments": _segment_reports(c, c.reference_candidate),
        "primary_gnss_supported_contiguous_segments": _segment_reports(c, c.gnss_supported),
        "physical_diagnostics_not_accuracy": {"world_rotated_acc_median_m_s2": np.nanmedian(world_acc[c.reference_candidate],axis=0).tolist(),
          "position_derivative_vs_same_INS_velocity_residual_m_s": percentiles(vel_residual),
          "velocity_derivative_vs_rotated_acc_minus_g_residual_m_s2": percentiles(acc_residual),
          "body_specific_force_interpretation": "released AI parser uses ax/ay/az directly with gravity; original SI retained, no fitted scale/bias"},
        "formal_native_ready": False, "AI_full_sequence_input_admitted": False,
        "pending": ["OXTS estimated reference is not independent ground truth",
          "physical pose output anchor/GPS antenna correction configuration not supplied",
          "car-to-IMU R/t is not established by IMU-to-Velodyne calibration; no guessed transform",
          "quality modes/INS accuracy are recorded, not an independent reference error guarantee",
          "nonuniform source clock requires an explicit raw100Hz full-sequence reader; no uniform200Hz conversion"]}


def write_candidate(path: Path, candidate: Candidate, metadata: dict) -> None:
    """Write a candidate schema, never a formal IPB sequence or zero lever arm."""
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Candidate output already exists")
    with h5py.File(path, "x") as h5:
        h5.attrs.update(schema=SCHEMA, converter_version=VERSION, formal_native_ready=False,
                        time_basis="original civil nanosecond coordinate; timezone unknown",
                        world_frame="translated Mercator east_north_up", body_frame="KITTI GPS_IMU front_left_up",
                        pose_anchor="OXTS unit per official readme; physical output/antenna correction unknown",
                        reference_is_independent_ground_truth=False,
                        metadata_json=json.dumps(metadata, sort_keys=True),
                        raw_field_names_json=json.dumps(FIELDS), row_reason_bits_json=json.dumps(REASONS),
                        clock_break_bits_json=json.dumps(CLOCK_BREAKS))
        arrays = {"raw/packet30":candidate.packet, "raw/source_index":candidate.source_index,
          "raw/civil_time_ns":candidate.clock_ns, "quality/row_reason_bits":candidate.reasons,
          "quality/clock_break_bits":candidate.clock_breaks, "quality/input_eligible":candidate.input_eligible,
          "quality/ins_reference_candidate":candidate.reference_candidate, "quality/primary_gnss_supported":candidate.gnss_supported,
          "imu/gyro_rad_s":candidate.packet[:,17:20], "imu/acc_m_s2":candidate.packet[:,11:14],
          "reference/position_enu_m":candidate.position, "reference/orientation_wxyz":candidate.quaternion,
          "reference/velocity_enu_m_s":candidate.packet[:,[7,6,10]]}
        for name, value in arrays.items():
            h5.create_dataset(name, data=value, compression="gzip", shuffle=True)
        for name, values in (("raw/civil_time_text",candidate.clock_text), ("raw/packet_text",candidate.packet_text),
                             ("raw/packet_sha256",candidate.packet_sha256)):
            if values:
                h5.create_dataset(name, data=values, dtype=h5py.string_dtype("ascii"))
        pairs = [distance_pairs(candidate.position,a,b) for a,b in spans(candidate.reference_candidate,candidate.clock_breaks)]
        h5.create_dataset("supervision_candidate/pairs_row_start_end_path_chord", data=np.concatenate(pairs) if pairs else np.empty((0,4)))
