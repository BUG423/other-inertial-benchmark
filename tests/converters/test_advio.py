"""Actual ADVIO release filenames are catalogued; camera q is never an IMU q."""

import json
import zipfile

from inertial_benchmark.data.convert import convert_dataset
from inertial_benchmark.data.converters import advio


def test_actual_release_archive_is_discovered_and_rejected(tmp_path):
    with zipfile.ZipFile(tmp_path / "advio-15.zip", "w") as f:
        f.writestr("advio-15/iphone/gyro.csv", "0,0,0,0\n")
        f.writestr("advio-15/iphone/accelerometer.csv", "0,0,9.8,0\n")
        f.writestr("advio-15/ground-truth/pose.csv", "0,0,0,0,1,0,0,0\n")
    assert advio.list_sequences(tmp_path) == ["advio-15"]
    raw = next(advio.iter_raw_sequences(tmp_path))
    assert raw.rejected and "camera-to-iPhone" in raw.rejected
    assert raw.attrs["published_accelerometer_units"] == "SI_as_recorded_no_extra_g_multiplier"
    assert len(raw.orientation) == 0
    out = tmp_path / "audit"
    convert_dataset("advio", tmp_path, out, workers=1)
    report = json.loads((out / "conversion_report.json").read_text())
    assert (report["accepted"], report["rejected"]) == (0, 1)
    assert not list(out.glob("sequences/*.h5"))
