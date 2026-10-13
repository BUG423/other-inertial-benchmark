"""Focused input-only target admission tests; execute on the server, not locally."""

from dataclasses import replace
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from inertial_benchmark.data.unlabelled import (
    RecordingIdentity,
    ReferenceAttitudeContract,
    SourcePartitionGuard,
    TargetAccessAudit,
    TargetInputConfig,
    TargetInputError,
    UnlabelledInertialDataset,
)
from inertial_benchmark.utils.geometry import quat_from_yaw, quat_rotate


def write_native(root, sid="target", *, n=24, group="target-recording", subject="p1",
                 shift=0.0, poison=False):
    path = root / "sequences" / f"{sid}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(n, dtype=np.float64) / 200
    gyro = np.column_stack((t + 0.1 + shift, t * 2 + 0.2, t * 3 + 0.3)).astype("f4")
    acc = np.column_stack((t + 1 + shift, t + 2, t + 9.81)).astype("f4")
    with h5py.File(path, "w") as f:
        f.attrs.update(schema_version="1.0", dataset="native-target", sequence_id=sid,
                       sample_rate_hz=200.0, timestamp_type="relative",
                       accelerometer_type="specific_force", body_frame="phone",
                       group_id=group, subject_id=subject, orientation_source="synthetic-reference",
                       orientation_convention="body_to_world_wxyz",
                       world_frame="gravity_aligned_z_up")
        f["timestamp"], f["imu/gyroscope"], f["imu/accelerometer"] = t, gyro, acc
        f["valid/imu"] = np.ones(n, bool)
        f["pose/orientation"] = quat_from_yaw(np.full(n, 0.4)).astype("f4")
        f["pose/position"] = np.full((n, 3), np.nan if poison else 123.0)
        f["pose/velocity"] = np.full((n, 3), np.nan if poison else 456.0)
        f["valid/pose"] = np.full(n, not poison)
    return path


def write_split(root, ids):
    (root / "splits").mkdir(exist_ok=True)
    (root / "splits" / "train.txt").write_text("\n".join(ids) + "\n")


def identity(sid, *, dataset="native-target", group=None, kind="recording", subject="p1",
             digest="1" * 64, **kw):
    return RecordingIdentity(dataset, sid, group or sid, kind, subject_id=subject,
                             imu_sha256=digest, **kw)


def guard(*, train=None, held_out=None, shared=True):
    return SourcePartitionGuard(
        tuple(train or [identity("source-train")]),
        tuple(held_out or [identity("source-val", subject="p2", digest="2" * 64)]),
        allow_shared_training_subjects=shared,
    )


def build(root, *, mode="body_imu", only=None, audit=None, source_guard=None, attitude=None,
          window=8, remove_gravity=False, group_kinds=None):
    return UnlabelledInertialDataset(
        root, TargetInputConfig(window=window, mode=mode, remove_gravity=remove_gravity),
        source_guard=source_guard or guard(), group_kinds=group_kinds or {"target": "recording"},
        only=only, stride=4, attitude=attitude, audit=audit,
    )


def contract(sid="target", n=24, intervals=None):
    return ReferenceAttitudeContract(sid, n, "synthetic-reference",
                                     tuple(intervals or [(0, n)]), "a" * 64)


def forbid_labels(monkeypatch, *, quaternion=False):
    original = h5py.Dataset.__getitem__
    reads = []

    def checked(dataset, item):
        reads.append(dataset.name)
        blocked = {"/pose/position", "/pose/velocity", "/valid/pose"}
        if quaternion:
            blocked.add("/pose/orientation")
        if dataset.name in blocked:
            raise AssertionError(f"forbidden numeric target read: {dataset.name}")
        return original(dataset, item)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked)
    return reads


def test_body_no_pose_access_or_hidden_metadata(tmp_path, monkeypatch):
    write_native(tmp_path, poison=True)
    write_split(tmp_path, ["target"])
    # These are intentionally unreadable as metadata. The leaf never opens them.
    (tmp_path / "dataset.json").write_text("not JSON; forbidden reader metadata")
    (tmp_path / "conversion_report.json").write_text("not JSON")
    reads = forbid_labels(monkeypatch, quaternion=True)
    audit = TargetAccessAudit()
    ds = build(tmp_path, audit=audit)
    sample = ds[0]
    assert set(sample) == {"imu", "timestamp", "input_mask", "seq", "start"}
    assert sample["imu"].shape == (6, 8) and sample["imu"].dtype == np.float32
    assert sample["input_mask"].all() and len(ds) == 5
    assert set(reads) <= {"/timestamp", "/imu/gyroscope", "/imu/accelerometer", "/valid/imu"}
    assert not any(e["field"] in ("orientation_source", "valid/pose") for e in audit.events)
    assert {Path(e["path"]).name for e in audit.events} == {"train.txt", "target.h5"}
    assert ds.access_contract["whole_file_hash"] is False


@pytest.mark.parametrize("mode", ["body_imu", "reference_attitude_assisted"])
def test_target_position_velocity_validpose_poison_does_not_change_inputs(
        tmp_path, mode, monkeypatch):
    original, poison = tmp_path / "original", tmp_path / "poison"
    for root, flag in ((original, False), (poison, True)):
        write_native(root, poison=flag)
        write_split(root, ["target"])
    forbid_labels(monkeypatch, quaternion=mode == "body_imu")
    attitude = None if mode == "body_imu" else {"target": contract()}
    a = build(original, mode=mode, attitude=attitude)
    b = build(poison, mode=mode, attitude=attitude)
    np.testing.assert_array_equal(a.index, b.index)
    assert a.identities == b.identities
    for i in range(len(a)):
        for key in ("imu", "timestamp", "input_mask"):
            np.testing.assert_array_equal(a[i][key], b[i][key])


def test_assisted_quaternion_is_declared_and_changes_rotated_input(tmp_path, monkeypatch):
    write_native(tmp_path)
    write_split(tmp_path, ["target"])
    reads = forbid_labels(monkeypatch)
    ds = build(tmp_path, mode="reference_attitude_assisted", attitude={"target": contract()})
    q = quat_from_yaw(np.full(8, 0.4)).astype("f4").astype("f8")
    body = build(tmp_path)[0]["imu"].T.astype("f8")
    expected = np.concatenate((quat_rotate(q, body[:, :3]), quat_rotate(q, body[:, 3:])), axis=1)
    np.testing.assert_allclose(ds[0]["imu"], expected.T.astype("f4"), atol=1e-6)
    assert "/pose/orientation" in reads
    assert ds.access_contract["reference_attitude_assisted"] is True
    before = ds[0]["imu"]
    with h5py.File(tmp_path / "sequences" / "target.h5", "r+") as f:
        f["pose/orientation"][:] = quat_from_yaw(np.full(24, 0.9)).astype("f4")
    changed = build(tmp_path, mode="reference_attitude_assisted", attitude={"target": contract()})
    assert not np.array_equal(before, changed[0]["imu"])


def test_body_survives_missing_pose_group(tmp_path):
    path = write_native(tmp_path)
    write_split(tmp_path, ["target"])
    with h5py.File(path, "r+") as f:
        del f["pose"]
        del f["valid/pose"]
    assert build(tmp_path)[0]["imu"].shape == (6, 8)


def test_assisted_requires_coverage_not_finite_placeholders(tmp_path):
    write_native(tmp_path)
    write_split(tmp_path, ["target"])
    with pytest.raises(TargetInputError, match="admitted attitude coverage"):
        build(tmp_path, mode="reference_attitude_assisted")
    # Only a proven interval is usable, even though every quaternion is finite.
    ds = build(tmp_path, mode="reference_attitude_assisted",
               attitude={"target": contract(intervals=[(8, 20)])})
    assert ds.index[:, 1].tolist() == [8, 12]
    with pytest.raises(TargetInputError, match="differs"):
        build(tmp_path, mode="reference_attitude_assisted",
              attitude={"target": replace(contract(), orientation_source="different-reference")})


def test_invalid_input_windows_and_quaternion_health(tmp_path):
    path = write_native(tmp_path)
    write_split(tmp_path, ["target"])
    with h5py.File(path, "r+") as f:
        f["valid/imu"][8:12] = False
        f["imu/gyroscope"][20] = np.nan
        f["pose/orientation"][0:4] = 0
    body = build(tmp_path)
    assert body.index[:, 1].tolist() == [0, 12]
    assisted = build(tmp_path, mode="reference_attitude_assisted", attitude={"target": contract()})
    assert assisted.index[:, 1].tolist() == [12]


@pytest.mark.parametrize("ids", [["../escape"], ["/tmp/escape"], ["nested/name"], [".."],
                                 ["target", "target"], [], ["target", "val"]])
def test_only_canonical_train_ids_and_no_path_escape(tmp_path, ids):
    write_native(tmp_path)
    write_native(tmp_path, "val", shift=1)
    write_split(tmp_path, ["target"])
    with pytest.raises(TargetInputError):
        build(tmp_path, only=ids)


@pytest.mark.parametrize("which", ["train_alias", "sequence_escape"])
def test_symlink_alias_rejected(tmp_path, which):
    path = write_native(tmp_path)
    write_split(tmp_path, ["target"])
    if which == "train_alias":
        train = tmp_path / "splits" / "train.txt"
        train.rename(train.with_name("val.txt"))
        train.symlink_to("val.txt")
    else:
        outside = tmp_path.parent / f"{tmp_path.name}-outside.h5"
        path.rename(outside)
        path.symlink_to(outside)
    with pytest.raises(TargetInputError, match="symlink"):
        build(tmp_path)


@pytest.mark.parametrize("kind", ["soft", "external", "virtual", "external_storage"])
def test_hdf5_input_links_cannot_escape_allowlist(tmp_path, kind):
    path = write_native(tmp_path)
    write_split(tmp_path, ["target"])
    with h5py.File(path, "r+") as f:
        del f["imu/gyroscope"]
        if kind == "soft":
            f["imu/gyroscope"] = h5py.SoftLink("/pose/velocity")
        elif kind == "external":
            f["imu/gyroscope"] = h5py.ExternalLink("other.h5", "/gyro")
        elif kind == "virtual":
            layout = h5py.VirtualLayout(shape=(24, 3), dtype="f4")
            layout[:] = h5py.VirtualSource(str(path), "pose/velocity", shape=(24, 3))
            f.create_virtual_dataset("imu/gyroscope", layout)
        else:
            f.create_dataset("imu/gyroscope", shape=(24, 3), dtype="f4",
                             external=[("outside.bin", 0, 24 * 3 * 4)])
    with pytest.raises(TargetInputError, match="links|materialized"):
        build(tmp_path)


def test_val_test_never_opened_and_empty_selection_refused(tmp_path, monkeypatch):
    write_native(tmp_path)
    write_split(tmp_path, ["target"])
    (tmp_path / "splits" / "val.txt").write_text("malicious\n")
    original_init = h5py.File.__init__
    opened = []

    def checked(self, name, *args, **kwargs):
        if isinstance(name, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(name))
            opened.append(path.stem)
        else:
            # h5py wraps existing FileID/GroupID/DatasetID handles internally.
            # Check their actual owning file without counting a new physical open.
            path = Path(os.fsdecode(h5py.h5f.get_name(name)))
        assert path.stem == "target"
        original_init(self, name, *args, **kwargs)

    # Preserve the real File class: production checks File/Group types and links.
    monkeypatch.setattr(h5py.File, "__init__", checked)
    assert len(build(tmp_path)) == 5 and opened == ["target"]
    with pytest.raises(TargetInputError):
        build(tmp_path, only=[])


def test_source_guard_recording_session_subject_semantics(tmp_path):
    path = write_native(tmp_path, group="shared-person")
    write_split(tmp_path, ["target"])
    common = dict(group="shared-person", kind="subject")
    source = identity("source-train", **common)
    ds = build(tmp_path, group_kinds={"target": "subject"}, source_guard=guard(train=[source]))
    assert len(ds) == 5  # same participant, distinct recording, explicitly declared
    with pytest.raises(TargetInputError, match="declaration"):
        build(tmp_path, group_kinds={"target": "subject"},
              source_guard=guard(train=[source], shared=False))
    with pytest.raises(TargetInputError, match="held-out"):
        build(tmp_path, group_kinds={"target": "subject"},
              source_guard=guard(held_out=[identity("held", **common)]))
    with h5py.File(path, "r+") as f:
        f.attrs["session_id"] = "session-5"
    with pytest.raises(TargetInputError, match="session_id"):
        build(tmp_path, group_kinds={"target": "subject"},
              source_guard=guard(train=[identity("other", session_id="session-5")]))


def test_source_guard_duplicate_content_across_namespaces(tmp_path):
    write_native(tmp_path)
    write_split(tmp_path, ["target"])
    admitted = build(tmp_path)
    duplicate = identity("renamed", dataset="other-dataset",
                         digest=admitted.identities[0].imu_sha256)
    with pytest.raises(TargetInputError, match="duplicate"):
        build(tmp_path, source_guard=guard(train=[duplicate]))


@pytest.mark.parametrize("kind", ["recording", "session"])
def test_shared_training_subject_declaration_never_allows_shared_recording_group(kind):
    protected = identity("source", kind=kind, group="same-physical-group")
    target = identity("target", kind=kind, group="same-physical-group", digest="4" * 64)
    with pytest.raises(TargetInputError, match="recording/session"):
        guard(train=[protected], shared=True).check(target)


@pytest.mark.parametrize("defect", ["short", "all_invalid", "nonuniform_time"])
def test_no_admissible_windows_or_bad_time_grid_is_refused(tmp_path, defect):
    path = write_native(tmp_path, n=4 if defect == "short" else 24)
    write_split(tmp_path, ["target"])
    with h5py.File(path, "r+") as f:
        if defect == "all_invalid":
            f["valid/imu"][:] = False
        elif defect == "nonuniform_time":
            f["timestamp"][12] += 0.003
    with pytest.raises(TargetInputError, match="no admissible|uniform"):
        build(tmp_path)


def test_target_changed_after_admission_refused(tmp_path):
    path = write_native(tmp_path)
    write_split(tmp_path, ["target"])
    ds = build(tmp_path)
    with h5py.File(path, "r+") as f:
        f["imu/accelerometer"][0] = 0
    with pytest.raises(TargetInputError, match="changed"):
        ds[0]


@pytest.mark.parametrize("kwargs", [
    {"mode": "device"}, {"mode": "body_imu", "remove_gravity": True},
    {"window": 1}, {"rate": float("nan")},
])
def test_input_config_refuses_hidden_privilege_or_invalid_shape(kwargs):
    with pytest.raises(TargetInputError):
        TargetInputConfig(**kwargs)


def test_admission_contracts_reject_unproven_identity_and_mutable_intervals():
    with pytest.raises(TargetInputError, match="classification"):
        identity("s", kind="unspecified")
    with pytest.raises(TargetInputError, match="IMU hashes"):
        guard(train=[identity("s", digest="")])
    with pytest.raises(TargetInputError, match="intervals"):
        contract(intervals=[(12, 20), (4, 8)])
    intervals = [[0, 24]]
    c = ReferenceAttitudeContract("target", 24, "synthetic-reference", intervals, "a" * 64)
    intervals[0][1] = 1
    assert c.intervals == ((0, 24),)
