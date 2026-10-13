"""Plausible simulator labels never bypass an unresolved body-frame audit."""

import json
import zipfile

from inertial_benchmark.data.convert import convert_dataset
from inertial_benchmark.data.converters import pegasus


def test_zip_catalogue_rejects_without_making_pose_or_splits(tmp_path):
    archive = tmp_path / "PegasusDataset.zip"
    with zipfile.ZipFile(archive, "w") as f:
        for part in ["TRAIN/a", "TEST/b"]:
            f.writestr("PegasusDataset/" + part + "/imu_data.csv", "0,0,0,-9.81,0,0,0\n")
            f.writestr("PegasusDataset/" + part + "/ground_truth.csv",
                       "0,0,0,0,1,0,0,0,0,0,0,0,0,0,0\n")
    assert len(pegasus.list_sequences(tmp_path)) == 2
    raws = list(pegasus.iter_raw_sequences(tmp_path))
    assert all(r.rejected and "FRD or FLU" in r.rejected for r in raws)
    assert all(len(r.orientation) == 0 for r in raws)
    out = tmp_path / "audit"
    convert_dataset("pegasus", tmp_path, out, workers=1)
    report = json.loads((out / "conversion_report.json").read_text())
    assert (report["accepted"], report["rejected"]) == (0, 2)
    assert not list(out.glob("sequences/*.h5"))
    assert pegasus.official_splits(tmp_path) == {}

