"""Small source readers for the UAV additions; resampling stays in data.convert."""

from __future__ import annotations

import csv
import hashlib
from decimal import Decimal
from pathlib import Path

import numpy as np

from . import _rig_utils as rig

AIRIO_COMMIT = "7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73"
AIRIO_SOURCE = f"https://github.com/Air-IO/Air-IO/tree/{AIRIO_COMMIT}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_csv(path: Path, columns: int, *, integer_clock: bool = False):
    """Keep nanosecond/microsecond clocks as integers, including scientific notation."""
    timestamps, values, headers = [], [], []
    with Path(path).open(newline="") as stream:
        for row in csv.reader(stream):
            if not row or not row[0].strip():
                continue
            if row[0].lstrip().startswith("#"):
                headers.append(",".join(row))
                continue
            try:
                value = Decimal(row[0]) if integer_clock else float(row[0])
            except (ValueError, ArithmeticError):
                if timestamps:
                    raise ValueError(f"non-numeric row within {path}") from None
                headers.append(",".join(row))
                continue
            if len(row) != columns:
                raise ValueError(f"{path}: expected {columns} columns, got {len(row)}")
            if integer_clock:
                if not value.is_finite() or value != value.to_integral_value():
                    raise ValueError(f"{path}: integer source clock required")
                value = int(value)
            timestamps.append(value)
            values.append([float(item) for item in row[1:]])
    if len(timestamps) < 2:
        raise ValueError(f"{path}: fewer than two source samples")
    time = np.asarray(timestamps, dtype=np.int64 if integer_clock else np.float64)
    return time, np.asarray(values, dtype=np.float64), "\n".join(headers)


def safe_orientation(q):
    q = np.asarray(q, dtype=np.float64)
    valid = np.isfinite(q).all(1) & (np.linalg.norm(np.nan_to_num(q), axis=1) > 0.5)
    return rig.normalize_quaternions(np.where(valid[:, None], q, [1., 0., 0., 0.])), valid


def evidence(paths, root):
    return {str(p.relative_to(root)): sha256_file(p) for p in paths}

