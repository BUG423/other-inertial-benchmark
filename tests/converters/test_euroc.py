"""EuRoC exact clocks, sensor/state frame checks, source masks and v1 conversion."""

import json

import numpy as np
import pytest
import yaml

from inertial_benchmark.data.convert import convert_dataset
from inertial_benchmark.data.converters import euroc
from inertial_benchmark.data.format import load_sequence
from inertial_benchmark.data.views import SequenceView, ViewConfig


def write_euroc(root, name="MH_01_easy", *, transform=None):
    folder = root / name
    imu, pose = folder / "mav0/imu0", folder / "mav0/state_groundtruth_estimate0"
    imu.mkdir(parents=True)
    pose.mkdir(parents=True)
    matrix = np.eye(4) if transform is None else transform
    for directory in [imu, pose]:
        (directory / "sensor.yaml").write_text(
            yaml.safe_dump({"T_BS": {"data": matrix.ravel().tolist()}})
        )
    origin = 1403636580000000007
    times = origin + np.arange(601, dtype=np.int64) * 5000000
    with (imu / "data.csv").open("w") as f:
        f.write("#timestamp,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z\n")
        for t in times:
            f.write(f"{t},0,0,0,0,0,9.80665\n")
    with (pose / "data.csv").open("w") as f:
        for t in times:
            f.write(f"{t},0,0,0,1,0,0,0,0,0,0,0.1,0.2,0.3,0.4,0.5,0.6\n")
    return folder


def test_clock_preserves_low_nanoseconds_and_biases_are_not_inputs(tmp_path):
    folder = write_euroc(tmp_path)
    raw = euroc.parse_sequence(folder)
    np.testing.assert_allclose(raw.imu_time, np.arange(601) * .005, atol=1e-14)
    assert raw.attrs["source_timestamp_origin_ns"] == 1403636580000000007
    np.testing.assert_array_equal(raw.gyroscope, 0.)
    assert raw.device_orientation is None
    assert raw.attrs["reference_bias_used_for_input"] is False
    assert raw.attrs["reference_position_is_imu_origin"] is True


def test_nonidentity_or_missing_calibration_is_rejected(tmp_path):
    matrix = np.eye(4)
    matrix[0, 3] = .1
    write_euroc(tmp_path, transform=matrix)
    raw = next(euroc.iter_raw_sequences(tmp_path))
    assert raw.rejected and "nonidentity T_BS" in raw.rejected
    (tmp_path / "MH_01_easy/mav0/imu0/sensor.yaml").unlink()
    assert next(euroc.iter_raw_sequences(tmp_path)).rejected


def test_derived_validation_is_recording_disjoint(tmp_path):
    for name in euroc.TRAIN_POOL + euroc.TEST:
        write_euroc(tmp_path, name)
    split = euroc.grouped_splits(tmp_path)
    assert [len(split[s]) for s in ["train", "val", "test"]] == [5, 1, 5]
    assert not set(split["train"]) & set(split["val"])
    assert not set(split["train"]) & set(split["test"])
    assert euroc.official_splits(tmp_path)["train"] == euroc.official_splits(tmp_path)["val"]


def test_reference_gap_remains_invalid_on_v1_grid_and_device_view_fails(tmp_path):
    folder = write_euroc(tmp_path / "raw")
    path = folder / "mav0/state_groundtruth_estimate0/data.csv"
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:200] + rows[230:]) + "\n")
    out = tmp_path / "converted"
    manifest = convert_dataset("euroc", folder.parent, out, workers=1)
    assert manifest["statistics"]["num_sequences"] == 1
    seq = load_sequence(out / "sequences/MH_01_easy.h5", validate=True)
    assert (~seq.valid_pose).sum() >= 29
    with pytest.raises(ValueError, match="device"):
        SequenceView(seq, ViewConfig(orientation="device"))
    report = json.loads((out / "conversion_report.json").read_text())
    assert report["accepted"] == 1


def test_invalid_reference_position_is_masked_not_used_as_supervision(tmp_path):
    folder = write_euroc(tmp_path / "raw")
    path = folder / "mav0/state_groundtruth_estimate0/data.csv"
    rows = path.read_text().splitlines()
    for index in range(200, 210):
        parts = rows[index].split(",")
        parts[1] = "nan"
        rows[index] = ",".join(parts)
    path.write_text("\n".join(rows) + "\n")
    raw = euroc.parse_sequence(folder)
    assert (~raw.pose_valid).sum() == 10
    out = tmp_path / "converted"
    convert_dataset("euroc", folder.parent, out, workers=1)
    seq = load_sequence(out / "sequences/MH_01_easy.h5", validate=True)
    assert np.isfinite(seq.position).all() and (~seq.valid_pose).sum() >= 10

