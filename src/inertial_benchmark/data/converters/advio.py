"""ADVIO public iPhone/manual-fixpoint reference catalogue; audit-only in v1."""

from pathlib import Path
from typing import Collection, Iterator, Optional

from . import _rig_utils as rig
from ._source_audit import discover
from .base import RawSequence

NAME = "advio"
VERSION = "1.0.0-audit"
LICENSE = "CC-BY-NC-4.0 (https://zenodo.org/records/1476931)"
SOURCE_COMMIT = "346e5968b49d361b87142f66c5c665cc6a8e8dfe"
REJECTION = (
    "audit-only: pose.csv is a camera/proxy reference built from IMU and manual fixpoints; "
    "camera-to-iPhone-IMU rotation direction and lever arm remain unverified. The source "
    "quaternion is not a verified IMU body_to_world pose, so native v1 training is disabled."
)


def _discover(source):
    # The real 2018 release says pose.csv/gyro.csv, despite README examples.
    return discover(source, "ground-truth/pose.csv")


def list_sequences(source):
    return sorted(_discover(source))


def official_splits(source):
    return {}


def iter_raw_sequences(
    source: Path, only: Optional[Collection[str]] = None
) -> Iterator[RawSequence]:
    found = _discover(source)
    for name in sorted(found) if only is None else sorted(only):
        meta = found.get(name, {"source_files": "unknown"})
        meta.update(source_license=LICENSE, training_enabled=False,
                    source_description=f"https://github.com/AaltoVision/ADVIO/tree/{SOURCE_COMMIT}",
                    source_record="https://zenodo.org/records/1476931",
                    reference_quality="IMU_correlated_manual_fixpoint_camera_proxy",
                    published_accelerometer_units="SI_as_recorded_no_extra_g_multiplier",
                    capabilities={"reference_orientation": False, "training": False})
        yield rig.rejected_sequence(name, REJECTION, meta,
                                    [f"source evidence: {meta}",
                                     "No placeholder quaternion or v1 canonical is written."])

