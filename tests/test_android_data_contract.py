from __future__ import annotations

import h5py
import numpy as np
import pytest
from pathlib import Path

from inertial_benchmark import CanonicalSequence, SequenceValidationError, WindowDataset


ANDROID_CONTRACT_ATTRIBUTES = {
    "archive_format": "inertial-lab/1",
    "schema_version": "0.1",
    "dataset": "InertialLab",
    "sequence_id": "android-contract-test",
    "display_name": "Android Contract Test",
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
    "started_at_utc": "2026-08-19T00:00:00Z",
}


def test_android_contract_is_accepted_by_benchmark_loader(tmp_path: Path):
    """验证符合 InertialLab 标准转换契约的 Android 序列能被 Benchmark 正确加载与校验。"""
    h5_path = tmp_path / "android_canonical.h5"
    n_samples = 10
    timestamps = np.arange(n_samples, dtype=np.float64) * 0.005  # 200 Hz
    gyroscope = np.zeros((n_samples, 3), dtype=np.float32)
    accelerometer = np.tile([0.0, 0.0, 9.80665], (n_samples, 1)).astype(np.float32)
    orientation = np.tile([1.0, 0.0, 0.0, 0.0], (n_samples, 1)).astype(np.float32)
    position = np.column_stack((
        timestamps * 1.5,
        timestamps * 0.5,
        np.zeros(n_samples),
    )).astype(np.float32)
    velocity = np.tile([1.5, 0.5, 0.0], (n_samples, 1)).astype(np.float32)

    valid_mask = np.ones(n_samples, dtype=bool)

    with h5py.File(h5_path, "w") as handle:
        for k, v in ANDROID_CONTRACT_ATTRIBUTES.items():
            handle.attrs[k] = v

        handle.create_dataset("timestamp", data=timestamps)
        handle.create_dataset("imu/gyroscope", data=gyroscope)
        handle.create_dataset("imu/accelerometer", data=accelerometer)
        handle.create_dataset("pose/orientation", data=orientation)
        handle.create_dataset("pose/position", data=position)
        handle.create_dataset("pose/velocity", data=velocity)
        handle.create_dataset("valid/imu", data=valid_mask)
        handle.create_dataset("valid/orientation", data=valid_mask)
        handle.create_dataset("valid/position", data=valid_mask)

    sequence = CanonicalSequence.from_hdf5(h5_path)

    assert sequence.attributes["world_frame"] == "arcore_local_x_right_y_forward_z_up"
    assert sequence.attributes["position_source"] == "ARCore visual-inertial odometry"
    assert sequence.attributes["orientation_source"] == "ARCore Android sensor pose"
    assert sequence.attributes["archive_format"] == "inertial-lab/1"
    assert sequence.valid_position is not None and sequence.valid_position.all()

    # 验证能生成滑窗视图
    dataset = WindowDataset(sequence, window_size=5, stride=2, target="velocity")
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample.features.shape == (6, 5)
    np.testing.assert_allclose(sample.target, [1.5, 0.5, 0.0], atol=1e-5)
