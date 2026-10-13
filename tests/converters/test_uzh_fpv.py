"""UZH source clocks, indexed IMU schema, privileged attitude and ZIP safety."""
from decimal import Decimal
import stat
import zipfile

import numpy as np
import pytest

from inertial_benchmark.data.converters import uzh_fpv


def test_indexed_epoch_decimal_clock_keeps_subnanosecond_difference():
    text = ("# header omits sample index\n"
            "0 1540821358.195877075195 1 2 3 4 5 6\n"
            "1 1540821358.195877075196 7 8 9 10 11 12\n")
    times, values, header = uzh_fpv.text_arrays(text, 7, indexed=True)
    assert times[1]-times[0] == Decimal("0.000000000001")
    np.testing.assert_array_equal(values[0], [1, 2, 3, 4, 5, 6])
    assert "omits" in header


@pytest.mark.parametrize("index", ["0", "-1", "1.5"])
def test_duplicate_reversed_or_noninteger_sample_index_rejected(index):
    text = "0 100 0 0 0 0 0 9.8\n"+index+" 100.01 0 0 0 0 0 9.8\n"
    with pytest.raises(ValueError, match="sample index"):
        uzh_fpv.text_arrays(text, 7, indexed=True)


@pytest.mark.parametrize("member,mode", [("../escape", 0), ("/absolute", 0),
                                         ("link", stat.S_IFLNK | 0o777)])
def test_archive_escape_or_symlink_is_rejected(tmp_path, member, mode):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo(member)
        info.create_system = 3
        info.external_attr = mode << 16
        z.writestr(info, "outside")
    with pytest.raises(ValueError, match="unsafe ZIP"):
        uzh_fpv.source_text(archive)


def test_static_source_remains_si_xyz_reference_only_and_one_flight_group(tmp_path):
    path = tmp_path / "indoor_forward_6_snapdragon_with_gt.zip"
    imu = ["# timestamp ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z"]
    pose = ["# timestamp tx ty tz qx qy qz qw"]
    origin = Decimal("1540821358.195877075195")
    for i in range(2201):
        t = origin+Decimal(i)/100
        imu.append(f"{i} {t} 0 0 0 0 0 9.80665")
        pose.append(f"{t} 1 2 3 0 0 0 1")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("imu.txt", "\n".join(imu)+"\n")
        z.writestr("groundtruth.txt", "\n".join(pose)+"\n")
        z.writestr("unrelated_camera.png", b"camera is not an input")
    raw = list(uzh_fpv.iter_raw_sequences(tmp_path))[0]
    assert raw.rejected is None
    assert raw.attrs["placement"] == "body"
    assert raw.attrs["group_id"] == "indoor_forward_6_snapdragon"
    assert raw.attrs["training_enabled"] is False
    assert raw.device_orientation is None
    assert raw.attrs["reference_orientation_is_privileged"] is True
    assert raw.attrs["reference_is_independent_mocap_orientation"] is False
    np.testing.assert_array_equal(raw.orientation[:, 0], 1)
    np.testing.assert_array_equal(raw.position[0], [1, 2, 3])
    np.testing.assert_array_equal(raw.accelerometer[:, 2], 9.80665)
    np.testing.assert_array_equal(raw.imu_time, raw.pose_time)
    assert uzh_fpv.official_splits(tmp_path) == {"test": [raw.sequence_id]}
