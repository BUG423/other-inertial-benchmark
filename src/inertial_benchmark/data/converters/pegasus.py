"""AirIO Pegasus simulator catalogue; native v1 conversion fails closed."""

from pathlib import Path
from typing import Collection, Iterator, Optional

from . import _rig_utils as rig
from ._external_utils import AIRIO_COMMIT, AIRIO_SOURCE
from ._source_audit import discover
from .base import RawSequence

NAME = "pegasus"
VERSION = "1.0.0-audit"
LICENSE = "AirIO Pegasus dataset license unspecified; do not infer from the code license"
REJECTION = (
    "audit-only: source IMU is FRD and the author rotates it into FLU; the ground-truth "
    "quaternion's body frame (FRD or FLU) and position lever arm have not been verified. "
    "A legal IMU body_to_world pose cannot be produced; training is disabled."
)


def _discover(source):
    return discover(source, "imu_data.csv")


def list_sequences(source):
    return sorted(_discover(source))


def official_splits(source):
    return {}  # no invented train/val/test for a rejected source


def iter_raw_sequences(
    source: Path, only: Optional[Collection[str]] = None
) -> Iterator[RawSequence]:
    found = _discover(source)
    for name in sorted(found) if only is None else sorted(only):
        meta = found.get(name, {"source_files": "unknown"})
        meta.update(source_license=LICENSE, simulation=True, training_enabled=False,
                    source_description=AIRIO_SOURCE, airio_source_commit=AIRIO_COMMIT,
                    capabilities={"reference_orientation": False, "training": False})
        yield rig.rejected_sequence(name, REJECTION, meta,
                                    [f"source evidence: {meta}",
                                     "No placeholder quaternion or v1 canonical is written."])

