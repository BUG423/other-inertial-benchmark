"""Evaluation dimensions are explicit; XY diagnostics reuse the saved XYZ prediction."""

import csv
import json

import numpy as np
import pytest

from inertial_benchmark.cfg import load_default
from inertial_benchmark.engine.results import RunResult, SequenceResult
from inertial_benchmark.utils import json_load, json_save
from inertial_benchmark.utils.reports import ProtocolMismatch, check_protocols, load_runs


def triangle_result():
    t = np.arange(5, dtype=float)
    gt = t[:, None] * np.array([1.0, 0.0, 2.0])
    return SequenceResult(
        sequence_id="triangle", t=t, rate=1.0,
        pos_pred=gt + [4.0, 0.0, 3.0], pos_gt=gt, pos_oracle=gt.copy(),
        valid_pose=np.array([True, True, False, True, True]),
        vel_pred=np.tile([1.0, 0.0, 1.0], (4, 1)),
        vel_target=np.tile([1.0, 0.0, 2.0], (4, 1)),
        window_valid=np.array([True, False, True, True]),
    )


def save_triangle(root, dims=3):
    seq = triangle_result()
    seq.compute_metrics(dims=dims)
    run = RunResult([seq, SequenceResult("short", skipped="too short")],
                    cfg={**load_default(), "metric_dims": dims}, model="triangle")
    return run.save(root, predictions=True, plots=False)


def rewrite_csv(path, transform):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    fields, rows = transform(fields, rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_xyz_metrics_and_xy_diagnostics_share_one_prediction():
    seq = triangle_result()
    before = {name: getattr(seq, name).copy() for name in
              ("pos_pred", "pos_gt", "pos_oracle", "vel_pred", "vel_target", "valid_pose")}
    metrics = seq.compute_metrics(dims=3, t_rte=iter([1.0, 2.0]), d_rte=iter([1.0]))
    assert metrics["ate"] == pytest.approx(5.0)
    assert metrics["xy_ate"] == pytest.approx(4.0)
    assert metrics["vel_rmse"] == pytest.approx(1.0)
    assert metrics["xy_vel_rmse"] == pytest.approx(0.0)
    assert metrics["xy_speed_ratio"] == pytest.approx(1.0)
    assert metrics["speed_ratio"] == pytest.approx(np.sqrt(2.0 / 5.0))
    assert metrics["xy_t_rte_2s"] == pytest.approx(0.0)
    assert metrics["xy_d_rte_1m"] == pytest.approx(0.0)
    np.testing.assert_allclose(seq.position_error(3), [5, 5, np.nan, 5, 5], equal_nan=True)
    xy = np.linalg.norm(seq.pos_pred[:, :2] - seq.pos_gt[:, :2], axis=1)
    np.testing.assert_allclose(seq.position_error(2), np.where(seq.valid_pose, xy, np.nan),
                               equal_nan=True)
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(seq, name), values)


def test_reference_distance_uses_the_evaluated_dimensions():
    seq = triangle_result()
    assert seq.distance == pytest.approx(2.0)  # Before evaluation, preserve the historical XY view.
    seq.compute_metrics(3)
    assert seq.distance == pytest.approx(2.0 * np.sqrt(5.0))
    assert seq.to_dict()["distance_m"] == pytest.approx(2.0 * np.sqrt(5.0))
    seq.compute_metrics(2)
    assert seq.distance == pytest.approx(2.0)


def test_three_dimensional_positions_allow_unavailable_window_velocities():
    seq = triangle_result()
    seq.vel_pred = np.zeros((0, 2))
    seq.vel_target = np.zeros((0, 2))
    seq.window_valid = np.zeros(0, bool)
    metrics = seq.compute_metrics(3)
    assert metrics["ate"] == pytest.approx(5.0)
    assert metrics["xy_ate"] == pytest.approx(4.0)
    assert np.isnan(metrics["vel_rmse"]) and np.isnan(metrics["xy_vel_rmse"])
    assert metrics["num_windows"] == metrics["xy_num_windows"] == 0


@pytest.mark.parametrize("dims", [1, 4, True, "3"])
def test_invalid_requested_dimensions_are_rejected(dims):
    seq = triangle_result()
    with pytest.raises(ValueError, match="metric_dims must be 2 or 3"):
        seq.compute_metrics(dims=dims)
    with pytest.raises(ValueError, match="metric_dims must be 2 or 3"):
        seq.position_error(dims)


@pytest.mark.parametrize("array", ["pos_pred", "pos_gt", "pos_oracle", "vel_pred", "vel_target"])
def test_three_dimensions_require_real_xyz_arrays(array, tmp_path):
    seq = triangle_result()
    setattr(seq, array, getattr(seq, array)[:, :2])
    with pytest.raises(ValueError, match=f"metric_dims=3 requires {array}"):
        seq.compute_metrics(dims=3)
    assert seq.skipped is None  # This is an incompatible protocol, not a skipped recording.
    if array in ("pos_pred", "pos_gt"):
        with pytest.raises(ValueError, match=f"metric_dims=3 requires {array}"):
            seq.position_error(3)
        with pytest.raises(ValueError, match=f"metric_dims=3 requires {array}"):
            RunResult([seq], cfg={"metric_dims": 3}).plot(tmp_path)
        assert list(tmp_path.iterdir()) == []


def test_dimension_evidence_survives_npz_csv_and_run_json(tmp_path):
    pd = pytest.importorskip("pandas")
    run = save_triangle(tmp_path / "new")
    restored = SequenceResult.load(run / "predictions" / "triangle.npz")
    assert restored.metric_dims_requested == restored.metric_dims_effective == 3
    assert restored.predicted_axes == ["x", "y", "z"]
    assert restored.metrics["ate"] == pytest.approx(5.0)
    assert restored.metrics["xy_ate"] == pytest.approx(4.0)
    meta = json_load(run / "metrics.json")
    assert meta["metric_dims_requested"] == meta["metric_dims_effective"] == 3
    assert meta["predicted_axes"] == ["x", "y", "z"]
    rows = pd.read_csv(run / "sequences.csv")
    assert rows.loc[0, "metric_dims_effective"] == 3
    assert rows.loc[0, "predicted_axes"] == "x,y,z"
    assert check_protocols([run])["metric_dims"] == 3
    assert len(load_runs(run)) == 1  # The genuinely short recording remains skipped.


@pytest.mark.parametrize("mutation", ["missing", "effective", "axes", "csv"])
def test_report_rejects_unsupported_or_inconsistent_three_dimensions(tmp_path, mutation):
    run = save_triangle(tmp_path / "bad")
    meta = json_load(run / "metrics.json")
    if mutation == "missing":
        for key in ("metric_dims_requested", "metric_dims_effective", "predicted_axes"):
            meta.pop(key)
    elif mutation == "effective":
        meta["metric_dims_effective"] = 2
    elif mutation == "axes":
        meta["predicted_axes"] = ["x", "y"]
    else:
        def modify(fields, rows):
            rows[0]["metric_dims_effective"] = "2"
            return fields, rows
        rewrite_csv(run / "sequences.csv", modify)
    json_save(run / "metrics.json", meta)
    with pytest.raises(ProtocolMismatch, match="dimension|predicted_axes"):
        check_protocols([run])
    pytest.importorskip("pandas")
    with pytest.raises(ProtocolMismatch, match="dimension|predicted_axes"):
        load_runs(run)


def test_old_two_dimensional_npz_and_report_remain_compatible(tmp_path):
    seq = triangle_result()
    seq.pos_pred, seq.pos_oracle = seq.pos_pred[:, :2], seq.pos_oracle[:, :2]
    seq.vel_pred, seq.vel_target = seq.vel_pred[:, :2], seq.vel_target[:, :2]
    # Write the historical NPZ schema: no dimension fields, no inference of old metric evidence.
    path = tmp_path / "old.npz"
    np.savez(path, meta=np.array(json.dumps({"sequence_id": "old", "metrics": {"ate": 4.0}})),
             t=seq.t, pos_pred=seq.pos_pred, pos_gt=seq.pos_gt, pos_oracle=seq.pos_oracle,
             valid_pose=seq.valid_pose, vel_pred=seq.vel_pred, vel_target=seq.vel_target,
             window_valid=seq.window_valid)
    old = SequenceResult.load(path)
    assert old.metric_dims_requested is None and old.metric_dims_effective is None
    assert old.predicted_axes is None
    assert old.compute_metrics(2)["ate"] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="metric_dims=3"):
        old.compute_metrics(3)

    run = save_triangle(tmp_path / "legacy", dims=2)
    meta = json_load(run / "metrics.json")
    fields = {"metric_dims_requested", "metric_dims_effective", "predicted_axes"}
    for key in fields:
        meta.pop(key)
    json_save(run / "metrics.json", meta)
    rewrite_csv(run / "sequences.csv", lambda names, rows: (
        [key for key in names if key not in fields],
        [{key: value for key, value in row.items() if key not in fields} for row in rows]))
    assert check_protocols([run])["metric_dims"] == 2
    pytest.importorskip("pandas")
    assert len(load_runs(run)) == 1


def test_run_protocol_cannot_relabel_computed_metrics(tmp_path):
    seq = triangle_result()
    seq.compute_metrics(2)
    run = RunResult([seq], cfg={"metric_dims": 3})
    with pytest.raises(ValueError, match="dimensions actually evaluated"):
        run.to_dict()
    with pytest.raises(ValueError, match="dimensions actually evaluated"):
        run.plot(tmp_path)
