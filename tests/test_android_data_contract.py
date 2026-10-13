"""Independent APP exchange contract; no embedded Android source is required."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from inertial_benchmark import Sequence, SequenceError, load_sequence, validate
from inertial_benchmark.data.views import SequenceView, ViewConfig
from inertial_benchmark.utils.geometry import GRAVITY, quat_rotate


@pytest.fixture
def android_exchange(tmp_path: Path) -> Path:
    """A physically consistent phone tilted 90 degrees around body x."""
    path = tmp_path / "android_exchange.h5"
    n = 401
    t = np.arange(n, dtype=np.float64) / 200
    attrs = {
        "archive_format": "inertial-lab/1",
        "schema_version": "0.1",
        "dataset": "InertialLab",
        "sequence_id": "android-contract-test",
        "world_frame": "arcore_local_x_right_y_forward_z_up",
        "timestamp_type": "relative",
        "orientation_convention": "body_to_world_wxyz",
        "accelerometer_type": "specific_force",
        "position_source": "ARCore visual-inertial odometry",
        "orientation_source": "ARCore Android sensor pose",
        "subject_id": "synthetic",
        "device_id": "synthetic",
        "source_license": "generated",
        "sample_rate_hz": 200,
    }
    velocity = np.tile([1.5, 0.5, 0.0], (n, 1))
    arrays = {
        "timestamp": t,
        "imu/gyroscope": np.zeros((n, 3)),
        "imu/accelerometer": np.tile([0.0, GRAVITY, 0.0], (n, 1)),
        "pose/orientation": np.tile([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], (n, 1)),
        "pose/position": velocity * t[:, None],
        "pose/velocity": velocity,
        "valid/imu": np.ones(n, bool),
        "valid/orientation": np.ones(n, bool),
        "valid/position": np.ones(n, bool),
    }
    with h5py.File(path, "w") as f:
        f.attrs.update(attrs)
        for key, values in arrays.items():
            f.create_dataset(key, data=values)
    return path


def test_exchange_preserves_provenance_and_physics_without_silent_relabeling(android_exchange):
    original = android_exchange.read_bytes()
    seq = load_sequence(android_exchange)
    assert isinstance(seq, Sequence)
    assert seq.attrs["archive_format"] == "inertial-lab/1"
    assert seq.attrs["legacy_schema_version"] == "0.1"
    assert seq.attrs["world_frame"] == "arcore_local_x_right_y_forward_z_up"
    assert seq.attrs["position_source"] == "ARCore visual-inertial odometry"
    assert seq.attrs["orientation_source"] == "ARCore Android sensor pose"
    assert seq.sample_rate == 200 and len(seq) == 401 and seq.valid_pose.all()
    np.testing.assert_allclose(seq.accelerometer, np.tile([0, GRAVITY, 0], (401, 1)))
    np.testing.assert_allclose(
        quat_rotate(seq.orientation, seq.accelerometer),
        np.tile([0, 0, GRAVITY], (401, 1)), atol=1e-6,
    )
    view = SequenceView(seq, ViewConfig(window=5, dims=3, frame="gravity_world"))
    starts = view.starts(2)
    assert len(starts) > 0
    np.testing.assert_allclose(view.targets(starts[:1]), [[1.5, 0.5, 0]], atol=1e-6)
    np.testing.assert_allclose(view.imu_windows(starts[:1])[0, 5], GRAVITY, atol=1e-6)
    report = validate(seq)
    assert not report.ok
    assert len(report.errors) == 1 and "world_frame" in report.errors[0]
    assert report.info["gravity"]["tilt_deg"] < 1e-4
    with pytest.raises(SequenceError, match="world_frame"):
        load_sequence(android_exchange, validate=True)
    assert android_exchange.read_bytes() == original


def test_exchange_inverted_specific_force_is_rejected_without_changing_source(android_exchange):
    with h5py.File(android_exchange, "r+") as f:
        f["imu/accelerometer"][:] *= -1
    original = android_exchange.read_bytes()
    seq = load_sequence(android_exchange)
    report = validate(seq)
    assert any("gravity:" in error and "-z" in error for error in report.errors)
    np.testing.assert_allclose(seq.accelerometer[:, 1], -GRAVITY)
    assert android_exchange.read_bytes() == original
