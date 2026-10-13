"""Blackbird author transforms and conservative route catalogue grouping."""

import numpy as np
import pytest

from inertial_benchmark.data.convert import convert_dataset
from inertial_benchmark.data.converters import blackbird
from inertial_benchmark.data.format import load_sequence
from inertial_benchmark.data.manifest import resolve_dataset
from inertial_benchmark.data.views import SequenceView, ViewConfig


def write_blackbird(root, partition="train", route="egg"):
    folder = root / partition / route / "yawForward/maxSpeed8p0"
    folder.mkdir(parents=True)
    t = np.arange(401) * .01
    epoch = 1525745895.
    imu = np.column_stack([epoch + t, np.zeros((len(t), 5)), np.full(len(t), 9.80665)])
    # q_NED_body = Rx(pi)^-1 * identity * Rz(pi/2)^-1, wxyz.
    q = np.tile([0., -np.sqrt(.5), -np.sqrt(.5), 0.], (len(t), 1))
    pose = np.column_stack([(epoch * 1e6 + np.round(t * 1e6)), np.zeros((len(t), 3)), q])
    np.savetxt(folder / "imu_data.csv", imu, delimiter=",", fmt="%.9f",
               header="timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z")
    np.savetxt(folder / "groundTruthPoses.csv", pose, delimiter=",", fmt=["%.0f"] + ["%.12f"] * 7)
    return folder


def test_author_layout_and_ned_body_imu_transform(tmp_path):
    folder = write_blackbird(tmp_path)
    name = blackbird.list_sequences(tmp_path)[0]
    raw = blackbird.parse_sequence(folder, name, tmp_path)
    np.testing.assert_allclose(np.abs(raw.orientation[:, 0]), 1., atol=1e-9)
    np.testing.assert_allclose(raw.orientation[:, 1:], 0., atol=1e-9)
    np.testing.assert_allclose(raw.accelerometer[:, 2], 9.80665)
    np.testing.assert_array_equal(raw.gyroscope, 0.)
    assert raw.attrs["reference_lever_arm_exact"] is False
    assert raw.attrs["training_enabled"] is False


def test_published_partitions_of_same_route_are_one_group(tmp_path):
    for partition in ["train", "test", "eval"]:
        write_blackbird(tmp_path, partition)
    raws = list(blackbird.iter_raw_sequences(tmp_path))
    assert len(raws) == 3 and {r.attrs["group_id"] for r in raws} == {"route_egg"}
    assert set(blackbird.official_splits(tmp_path)) == {"test"}


def test_native_v1_catalogue_has_no_train_or_device_attitude(tmp_path):
    write_blackbird(tmp_path / "raw")
    out = tmp_path / "converted"
    manifest = convert_dataset("blackbird", tmp_path / "raw", out, workers=1)
    assert manifest["statistics"]["num_sequences"] == 1
    assert not (out / "splits/train.txt").exists()
    with pytest.raises(FileNotFoundError, match="train"):
        resolve_dataset(out).sequence_ids("train")
    seq = load_sequence(out / next(iter(manifest["sequences"].values()))["file"], validate=True)
    assert seq.device_orientation is None
    assert seq.attrs["benchmark_track"] == "uav_conversion_catalogue"
    # Oracle reference task can read the catalogue; deployment attitude is unavailable.
    SequenceView(seq, ViewConfig(orientation="reference"))
    with pytest.raises(ValueError, match="device"):
        SequenceView(seq, ViewConfig(orientation="device"))

