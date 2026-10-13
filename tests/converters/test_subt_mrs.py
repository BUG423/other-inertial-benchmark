"""Scientific source-contract tests, executed on AutoDL with real v1 helpers."""
from pathlib import Path

import numpy as np
import pytest

from inertial_benchmark.data.converters import subt_mrs


def source_csvs(tmp_path, segment="tunnel_1", robot="UGV2"):
    folder = tmp_path / "SubT-MRS/Final_Challenge" / robot / segment
    (folder / "imu_data").mkdir(parents=True)
    (folder / "ground_truth").mkdir()
    origin = 1_700_000_000_123_456_789
    times = [origin+i*5_000_000 for i in range(4_401)]
    imu = folder / "imu_data/imu_data.csv"
    gt = folder / "ground_truth/ground_truth_imu.csv"
    with imu.open("w") as f:
        f.write("timestamp,qx,qy,qz,qw,wx,wy,wz,ax,ay,az\n")
        for t in times:
            # This deliberately unrelated source attitude must never become
            # device attitude. Reference biases must never alter the input.
            f.write(f"{t},0,1,0,0,0,0,0,0,0,9.80665\n")
    with gt.open("w") as f:
        f.write("timestamp,x,y,z,qx,qy,qz,qw,vx,vy,vz,bwx,bwy,bwz,bax,bay,baz\n")
        for t in times:
            f.write(f"{t},1,2,3,0,0,0,1,0,0,0,1,2,3,4,5,6\n")
    return folder, times


def test_exact_nanosecond_origin_and_privileged_bias_separation(tmp_path):
    folder, times = source_csvs(tmp_path)
    raw = subt_mrs.parse_sequence(folder, "test", ("Final_Challenge", "UGV2", "tunnel_1"))
    assert raw.rejected is None
    np.testing.assert_allclose(raw.imu_time, np.arange(len(times))*0.005, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(raw.pose_time, raw.imu_time)
    np.testing.assert_array_equal(raw.gyroscope, 0)
    np.testing.assert_array_equal(raw.accelerometer[:, 2], 9.80665)
    np.testing.assert_array_equal(raw.orientation[:, 0], 1)
    assert raw.device_orientation is None
    assert raw.attrs["reference_bias_columns_used"] is False
    assert raw.attrs["reference_incorporates_IMU"] is True
    assert raw.attrs["placement"] == "body"
    assert raw.attrs["source_placement"] == "robot_body"
    assert raw.attrs["source_platform"] == "ugv"
    assert raw.attrs["source_timestamp_origin_ns"] == str(times[0])


def test_segments_keep_event_robot_group_and_catalogue_is_explicit(tmp_path):
    source_csvs(tmp_path, "tunnel_1")
    source_csvs(tmp_path, "tunnel_2")
    ids = subt_mrs.list_sequences(tmp_path)
    assert len(ids) == 2
    assert subt_mrs.official_splits(tmp_path) == {"test": ids}
    raws = list(subt_mrs.iter_raw_sequences(tmp_path))
    assert {r.attrs["group_id"] for r in raws} == {"Final_Challenge_UGV2"}
    assert all(r.attrs["split_is_author_official"] is False for r in raws)
    assert all(r.attrs["training_enabled"] is False for r in raws)


@pytest.mark.parametrize("delta", [0, -5_000_000])
def test_duplicate_or_reversed_clock_rejected(tmp_path, delta):
    folder, times = source_csvs(tmp_path)
    p = folder / "imu_data/imu_data.csv"
    lines = p.read_text().splitlines()
    lines[2] = str(times[0]+delta)+","+lines[2].split(",", 1)[1]
    p.write_text("\n".join(lines)+"\n")
    raw = list(subt_mrs.iter_raw_sequences(tmp_path))[0]
    assert "nonmonotonic" in raw.rejected


def test_simulation_is_not_counted_as_real_independent_source(tmp_path):
    folder, _ = source_csvs(tmp_path)
    real = tmp_path / "SubT-MRS/Final_Challenge"
    real.rename(tmp_path / "SubT-MRS/TartanAir")
    assert subt_mrs.list_sequences(tmp_path) == []


def test_symlink_source_is_refused(tmp_path):
    folder, _ = source_csvs(tmp_path)
    imu = folder / "imu_data/imu_data.csv"
    real = imu.with_suffix(".original")
    imu.rename(real)
    imu.symlink_to(real)
    raw = list(subt_mrs.iter_raw_sequences(tmp_path))[0]
    assert "symlink" in raw.rejected
