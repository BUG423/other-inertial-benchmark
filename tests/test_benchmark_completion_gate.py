"""Focused guards for future campaign split execution and completed-state reuse."""

import csv
import copy
import hashlib
import json
from pathlib import Path
from typing import Optional

import pytest

from inertial_benchmark.engine.benchmark import (
    BenchmarkItem,
    _sha,
    _run_item,
    bind_completed_results,
    matrix_row_id,
    matrix_state,
    save_matrix_state,
    verify_completed_state,
)


@pytest.fixture
def native(tmp_path):
    root = tmp_path / "native"
    (root / "splits").mkdir(parents=True)
    (root / "sequences").mkdir()
    (root / "dataset.json").write_text(json.dumps({"dataset": "fake"}))
    (root / "splits" / "test.txt").write_text("s0\n")
    return root


def _write_result(root: Path, row: dict, split: str = "test", *, num_sequences: int = 1,
                  model: Optional[str] = None, dataset: Optional[str] = None,
                  dims: Optional[int] = None, reported_split: Optional[str] = None,
                  seed: Optional[int] = None,
                  sequence_ids: Optional[list[str]] = None) -> None:
    dims = dims or row["dims"]
    sequence_ids = sequence_ids or [f"s{index}" for index in range(max(num_sequences, 0))]
    output = root / split
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mode": "val",
        "model": model or row["profile"],
        "dataset": dataset or row["dataset"],
        "split": reported_split or split,
        "seed": row["seed"] if seed is None else seed,
        "protocol": {"metric_dims": dims},
        "metric_dims_requested": dims,
        "metric_dims_effective": dims,
        "predicted_axes": list("xyz"[:dims]),
        "num_sequences": num_sequences,
        "num_skipped": 0,
        "metrics": {"ate": 1.0 if num_sequences else None},
        "median": {"ate": 1.0 if num_sequences else None},
        "std": {"ate": 0.0 if num_sequences >= 2 else None},
        "count": {"ate": max(num_sequences, 0)},
    }
    (output / "metrics.json").write_text(json.dumps(metrics))
    with (output / "sequences.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "sequence_id", "skipped", "metric_dims_requested",
            "metric_dims_effective", "predicted_axes", "ate"))
        writer.writeheader()
        for sequence_id in sequence_ids:
            writer.writerow({"sequence_id": sequence_id, "skipped": "",
                             "metric_dims_requested": dims,
                             "metric_dims_effective": dims,
                             "predicted_axes": ",".join("xyz"[:dims]), "ate": 1.0})


def _row(native_root, **overrides):
    split_path = Path(native_root) / "splits" / "test.txt"
    base = {"profile": "fixture-model", "dataset": "fake", "data_root": str(native_root),
            "seed": 1735231569, "dims": 2,
            "split_sha256": {"test.txt": hashlib.sha256(split_path.read_bytes()).hexdigest()},
            "taskview_receipt": {"status": "passed", "splits": {
                "test": {"status": "passed", "valid_windows": 1}}}}
    return {**base, **overrides}


def test_explicit_all_na_taskview_fails_before_execution_and_preserves_old_directory(
        native, tmp_path):
    (native / "splits" / "test.txt").write_text("")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not-loaded-for-na")
    item = BenchmarkItem("fixture-model", str(weights), str(native), 1735231569,
                         splits=["test"], root=tmp_path / "runs",
                         split_receipts={"test": {
                             "status": "not_applicable_missing_split"}})
    historical = item.run_dir / "test"
    historical.mkdir(parents=True)
    marker = historical / "historical-empty-directory"
    marker.write_text("preserve")
    with pytest.raises(RuntimeError, match="no applicable nonempty evaluation split"):
        _run_item(item)
    assert marker.read_text() == "preserve"
    assert not (historical / "metrics.json").exists()


def test_taskview_passed_but_empty_split_fails_before_validator(native, tmp_path):
    (native / "splits" / "test.txt").write_text("")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not-loaded-for-empty")
    item = BenchmarkItem("fixture-model", str(weights), str(native), 1735231569,
                         splits=["test"], root=tmp_path / "runs",
                         split_receipts={"test": {"status": "passed", "valid_windows": 1}})
    with pytest.raises(RuntimeError, match="qualified applicable split.*empty"):
        _run_item(item)


def test_completed_state_binds_nonempty_results_and_rechecks_hash(native, tmp_path):
    row = _row(native)
    run = tmp_path / "run"
    _write_result(run, row)
    checkpoint = run / "train" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    identity = run / "run-identity.json"
    identity.write_text("{}")
    artifacts = bind_completed_results(
        row, run, ["test"], row["taskview_receipt"]["splits"])
    completed = {**row, "state": "completed", "checkpoint": str(checkpoint),
                 "checkpoint_sha256": _sha(checkpoint), "run_identity_path": str(identity),
                 "run_identity_sha256": _sha(identity), "result_artifacts": artifacts,
                 "result": {"splits": {"test": "evaluated"}}}

    verify_completed_state(completed)
    assert artifacts["splits"]["test"]["num_sequences"] == 1
    assert artifacts["splits"]["test"]["csv_rows"] == 1
    assert artifacts["not_applicable_splits"] == {}

    with (run / "test" / "sequences.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(RuntimeError, match="sequences.csv changed"):
        verify_completed_state(completed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("zero", "num_sequences"),
     ("model", "model identity mismatch"),
     ("dataset", "dataset identity mismatch"),
     ("split", "split identity mismatch"),
     ("seed", "seed identity mismatch"),
     ("dims", "dimension evidence mismatch"),
     ("missing_csv", "lacks sequences.csv"),
     ("header", "lacks required headers"),
     ("rows", "CSV row count")),
)
def test_completed_result_gate_rejects_empty_or_inconsistent_evidence(
        native, tmp_path, mutation, message):
    row = _row(native)
    run = tmp_path / mutation
    kwargs = {}
    if mutation == "zero":
        kwargs["num_sequences"] = 0
    elif mutation == "model":
        kwargs["model"] = "wrong-model"
    elif mutation == "dataset":
        kwargs["dataset"] = "wrong-dataset"
    elif mutation == "split":
        kwargs["reported_split"] = "wrong-split"
    elif mutation == "seed":
        kwargs["seed"] = 951150512
    elif mutation == "dims":
        kwargs["dims"] = 3
    _write_result(run, row, **kwargs)
    if mutation == "missing_csv":
        (run / "test" / "sequences.csv").unlink()
    elif mutation == "header":
        (run / "test" / "sequences.csv").write_text("sequence_id,skipped\ns0,\n")
    elif mutation == "rows":
        with (run / "test" / "sequences.csv").open("a") as stream:
            stream.write('s1,,2,2,"x,y"\n')

    with pytest.raises(RuntimeError, match=message):
        bind_completed_results(row, run, ["test"],
                               {"test": {"status": "passed", "valid_windows": 1}})


def test_completed_result_gate_rejects_all_na_cell(native, tmp_path):
    (native / "splits" / "test.txt").write_text("")
    row = _row(native)
    row["taskview_receipt"]["splits"]["test"] = {
        "status": "not_applicable_missing_split"}
    with pytest.raises(RuntimeError, match="no applicable nonempty evaluation split"):
        bind_completed_results(row, tmp_path / "run", ["test"],
                               row["taskview_receipt"]["splits"])


def _completed(row: dict, run: Path, *, artifacts: bool = True) -> dict:
    checkpoint = run / "train" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    identity = run / "run-identity.json"
    identity.write_text("{}")
    completed = {**row, "state": "completed", "checkpoint": str(checkpoint),
                 "checkpoint_sha256": _sha(checkpoint), "run_identity_path": str(identity),
                 "run_identity_sha256": _sha(identity),
                 "result": {"splits": {"test": "evaluated"}}}
    if artifacts:
        completed["result_artifacts"] = bind_completed_results(
            row, run, ["test"], row["taskview_receipt"]["splits"])
    return completed


def test_legacy_completed_state_is_live_validated_without_backfill(native, tmp_path):
    row = _row(native)
    run = tmp_path / "legacy"
    _write_result(run, row)
    completed = _completed(row, run, artifacts=False)
    before = dict(completed)

    verify_completed_state(completed)

    assert completed == before
    assert "result_artifacts" not in completed
    metrics = json.loads((run / "test" / "metrics.json").read_text())
    metrics["num_sequences"] = 0
    (run / "test" / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="num_sequences"):
        verify_completed_state(completed)


def test_expected_eval_split_omission_is_rejected(native, tmp_path):
    (native / "splits" / "test_extra.txt").write_text("s1\n")
    row = _row(native)
    row["taskview_receipt"]["splits"]["test_extra"] = {
        "status": "passed", "valid_windows": 1}
    _write_result(tmp_path / "run", row)
    with pytest.raises(RuntimeError, match="exactly match dataset test_splits"):
        bind_completed_results(row, tmp_path / "run", ["test"],
                               row["taskview_receipt"]["splits"])


def test_nonempty_split_cannot_be_declared_not_applicable(native, tmp_path):
    row = _row(native)
    row["taskview_receipt"]["splits"]["test"] = {
        "status": "not_applicable_missing_split"}
    with pytest.raises(RuntimeError, match="nonempty but taskview marks it N/A"):
        bind_completed_results(row, tmp_path / "run", ["test"],
                               row["taskview_receipt"]["splits"])


@pytest.mark.parametrize("split", ["", ".", "..", "../outside", "nested/test", "nested\\test"])
def test_unsafe_split_names_are_rejected_before_output_access(native, tmp_path, split):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"unused")
    item = BenchmarkItem("fixture-model", str(weights), str(native), 1735231569,
                         splits=[split], root=tmp_path / "runs")
    with pytest.raises(RuntimeError, match="unsafe evaluation split name"):
        _run_item(item)


def test_result_symlink_cannot_escape_run_root(native, tmp_path):
    row = _row(native)
    outside = tmp_path / "outside"
    _write_result(outside, row)
    run = tmp_path / "run"
    run.mkdir()
    (run / "test").symlink_to(outside / "test", target_is_directory=True)
    with pytest.raises(RuntimeError, match="resolves outside the run root"):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


def test_result_file_symlink_is_rejected(native, tmp_path):
    row = _row(native)
    run = tmp_path / "run"
    outside = tmp_path / "outside"
    _write_result(outside, row)
    (run / "test").mkdir(parents=True)
    (run / "test" / "metrics.json").symlink_to(outside / "test" / "metrics.json")
    (run / "test" / "sequences.csv").write_bytes(
        (outside / "test" / "sequences.csv").read_bytes())
    with pytest.raises(RuntimeError, match="metrics.json must not be a symlink"):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


def test_matrix_state_cannot_override_taskview_qualification(native, tmp_path):
    admission = tmp_path / "coverage.json"
    row = {**_row(native), "input_track": "deployment", "domain": "walking",
           "trainable": True, "checkpoint_training_source": None,
           "scientific_input_spec": {"window": 100}, "sampling_spec": {"stride": 10},
           "dataset_fingerprint": "fp", "dataset_manifest_sha256": "manifest",
           "split_sha256": {"test.txt": "split"}, "profile_config_sha256": "profile",
           "model_cpu_accepted": True, "model_cpu_receipt_sha256": "cpu"}
    row["id"] = matrix_row_id(row["profile"], row["dataset"], row["seed"], row["input_track"])
    save_matrix_state(admission, row)
    state_path = admission.parent / "run-states" / f'{row["id"]}.json'
    stale = json.loads(state_path.read_text())
    stale["taskview_receipt"]["splits"]["test"] = {
        "status": "not_applicable_missing_split"}
    state_path.write_text(json.dumps(stale))
    with pytest.raises(RuntimeError, match="scientific row identity changed"):
        matrix_state(admission, row)


def test_result_sequence_ids_must_match_dataset_split(native, tmp_path):
    (native / "splits" / "test.txt").write_text("expected_s0\nexpected_s1\n")
    row = _row(native)
    run = tmp_path / "fabricated"
    _write_result(run, row, sequence_ids=["fabricated_not_in_split"])
    with pytest.raises(RuntimeError, match="sequence IDs do not exactly match"):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


def test_result_cannot_truncate_dataset_split(native, tmp_path):
    (native / "splits" / "test.txt").write_text("s0\ns1\n")
    row = _row(native)
    run = tmp_path / "truncated"
    _write_result(run, row, sequence_ids=["s0"])
    with pytest.raises(RuntimeError, match="sequence IDs do not exactly match"):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


def test_result_sequence_order_is_irrelevant_but_schema_and_null_metrics_are_valid(
        native, tmp_path):
    (native / "splits" / "test.txt").write_text("s0\ns1\n")
    row = _row(native)
    run = tmp_path / "real-schema"
    _write_result(run, row, num_sequences=2, sequence_ids=["s1", "s0"])
    metrics_path = run / "test" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    for name in ("metrics", "median", "std"):
        metrics[name]["rte"] = None
    metrics["count"]["rte"] = 0
    metrics_path.write_text(json.dumps(metrics))
    csv_path = run / "test" / "sequences.csv"
    lines = csv_path.read_text().splitlines()
    csv_path.write_text("\n".join([lines[0] + ",rte"] + [line + "," for line in lines[1:]]) + "\n")

    artifacts = bind_completed_results(
        row, run, ["test"], row["taskview_receipt"]["splits"])

    assert artifacts["splits"]["test"]["num_sequences"] == 2


@pytest.mark.parametrize("case", ["empty", "missing_main"])
def test_result_requires_complete_publishable_aggregate_schema(native, tmp_path, case):
    row = _row(native)
    run = tmp_path / case
    _write_result(run, row)
    metrics_path = run / "test" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if case == "empty":
        for name in ("metrics", "median", "std", "count"):
            metrics[name] = {}
        message = "aggregate metric keys are inconsistent or empty"
    else:
        metrics.update(metrics={"loss": 0.5}, median={"loss": 0.5},
                       std={"loss": None}, count={"loss": 1})
        csv_path = run / "test" / "sequences.csv"
        lines = csv_path.read_text().splitlines()
        csv_path.write_text("\n".join([lines[0] + ",loss"]
                            + [line + ",0.5" for line in lines[1:]]) + "\n")
        message = "no finite publishable main metric"
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match=message):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


def test_atomic_result_replacement_during_read_is_rejected(native, tmp_path, monkeypatch):
    from inertial_benchmark.engine import benchmark as benchmark_module

    row = _row(native)
    run = tmp_path / "replace-race"
    _write_result(run, row)
    metrics_path = run / "test" / "metrics.json"
    old_bytes = metrics_path.read_bytes()
    replacement = metrics_path.with_name("replacement.json")
    changed = json.loads(old_bytes)
    changed["model"] = "replacement-model"
    replacement.write_text(json.dumps(changed))
    real_fdopen = benchmark_module.os.fdopen
    opens = 0

    def replace_after_open(fd, *args, **kwargs):
        nonlocal opens
        opens += 1
        if opens == 2:  # split snapshot is first; metrics.json is second
            replacement.replace(metrics_path)
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(benchmark_module.os, "fdopen", replace_after_open)
    with pytest.raises(RuntimeError, match="metrics.json pathname changed"):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])
    assert metrics_path.read_bytes() != old_bytes


@pytest.mark.parametrize("aggregate", ["metrics", "median", "std"])
def test_result_aggregate_values_are_recomputed_from_csv(native, tmp_path, aggregate):
    (native / "splits" / "test.txt").write_text("s0\ns1\n")
    row = _row(native)
    run = tmp_path / aggregate
    _write_result(run, row, num_sequences=2, sequence_ids=["s0", "s1"])
    path = run / "test" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics[aggregate]["ate"] = 999.0
    path.write_text(json.dumps(metrics))

    with pytest.raises(RuntimeError, match=r"disagrees with CSV"):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


def test_result_aggregate_allows_json_roundtrip_tolerance(native, tmp_path):
    (native / "splits" / "test.txt").write_text("s0\ns1\n")
    row = _row(native)
    run = tmp_path / "roundtrip"
    _write_result(run, row, num_sequences=2, sequence_ids=["s0", "s1"])
    path = run / "test" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["metrics"]["ate"] += 5e-13
    path.write_text(json.dumps(metrics))

    bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonempty_nonfinite_csv_metric_is_never_treated_as_null(
        native, tmp_path, value):
    row = _row(native)
    run = tmp_path / value.replace("-", "minus-")
    _write_result(run, row)
    metrics_path = run / "test" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    for aggregate in ("metrics", "median"):
        metrics[aggregate]["rte"] = 2.0
    metrics["std"]["rte"] = None
    metrics["count"]["rte"] = 1
    metrics_path.write_text(json.dumps(metrics))
    csv_path = run / "test" / "sequences.csv"
    lines = csv_path.read_text().splitlines()
    lines[0] += ",rte"
    lines[1] = lines[1].rsplit(",", 1)[0] + f",{value},2.0"
    csv_path.write_text("\n".join(lines) + "\n")

    with pytest.raises(RuntimeError, match="CSV metric 'ate' is not finite"):
        bind_completed_results(
            row, run, ["test"], row["taskview_receipt"]["splits"])


@pytest.mark.parametrize("case", ["zero_count_nan", "one_count_nan_std"])
def test_result_aggregate_requires_json_null_for_undefined_statistics(
        native, tmp_path, case):
    row = _row(native)
    run = tmp_path / case
    _write_result(run, row)
    path = run / "test" / "metrics.json"
    metrics = json.loads(path.read_text())
    if case == "zero_count_nan":
        for name in ("metrics", "median", "std"):
            metrics[name]["rte"] = float("nan")
        metrics["count"]["rte"] = 0
        csv_path = run / "test" / "sequences.csv"
        lines = csv_path.read_text().splitlines()
        csv_path.write_text("\n".join([lines[0] + ",rte", lines[1] + ","]) + "\n")
    else:
        metrics["std"]["ate"] = float("nan")
    path.write_text(json.dumps(metrics))

    with pytest.raises(RuntimeError, match="zero count|one sample"):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


@pytest.mark.parametrize("mutation", ["duplicate_header", "extra_field"])
def test_result_rejects_malformed_csv_shape(native, tmp_path, mutation):
    row = _row(native)
    run = tmp_path / mutation
    _write_result(run, row)
    path = run / "test" / "sequences.csv"
    lines = path.read_text().splitlines()
    if mutation == "duplicate_header":
        lines[0] += ",ate"
        lines[1] += ",1.0"
    else:
        lines[1] += ",unexpected"
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(RuntimeError, match="required headers|malformed extra fields"):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


def test_changed_split_bytes_cannot_be_rebound(native, tmp_path):
    row = _row(native)
    run = tmp_path / "changed-split"
    _write_result(run, row, sequence_ids=["changed"])
    (native / "splits" / "test.txt").write_text("changed\n")

    with pytest.raises(RuntimeError, match="changed from qualified split_sha256"):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


def test_qualified_empty_na_split_cannot_disappear_before_binding(native, tmp_path):
    empty = native / "splits" / "test_empty.txt"
    empty.write_text("")
    row = _row(native, split_sha256={
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((native / "splits").glob("*.txt"))
    })
    row["data_root"] = {
        "name": "fake", "path": str(native),
        "test_splits": ["test", "test_empty"],
    }
    row["taskview_receipt"]["splits"]["test_empty"] = {
        "status": "not_applicable_missing_split"}
    run = tmp_path / "deleted-empty"
    _write_result(run, row)
    empty.unlink()

    with pytest.raises(RuntimeError, match="disappeared after qualified split_sha256"):
        bind_completed_results(
            row, run, ["test", "test_empty"], row["taskview_receipt"]["splits"])


def test_atomic_split_replacement_during_read_is_rejected(native, tmp_path, monkeypatch):
    from inertial_benchmark.engine import benchmark as benchmark_module

    row = _row(native)
    run = tmp_path / "split-replace-race"
    _write_result(run, row)
    split_path = native / "splits" / "test.txt"
    replacement = split_path.with_name("replacement.txt")
    replacement.write_text("s0\n")
    real_fdopen = benchmark_module.os.fdopen
    replaced = False

    def replace_after_open(fd, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            replacement.replace(split_path)
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(benchmark_module.os, "fdopen", replace_after_open)
    with pytest.raises(RuntimeError, match="split 'test' pathname changed"):
        bind_completed_results(row, run, ["test"], row["taskview_receipt"]["splits"])


@pytest.mark.parametrize("mutation", ["run_id", "path", "boolean"])
def test_transfer_completed_provenance_is_bound_to_authoritative_origin(
        native, tmp_path, mutation):
    target = _row(
        native, input_track="deployment", domain="target-domain", trainable=False,
        checkpoint_training_source="source-dataset")
    run = tmp_path / "transfer"
    _write_result(run, target)
    completed = _completed(target, run)
    origin_row = _row(
        native, dataset="source-dataset", input_track=target["input_track"],
        domain="source-domain", trainable=True, checkpoint_training_source=None)
    origin_run = tmp_path / "origin"
    _write_result(origin_run, origin_row)
    origin = _completed(origin_row, origin_run)
    origin["id"] = matrix_row_id(
        target["profile"], origin["dataset"], target["seed"], target["input_track"])
    identity_payload = {key: origin[key] for key in ("profile", "seed", "input_track")}
    Path(origin["run_identity_path"]).write_text(json.dumps(identity_payload))
    origin["run_identity_sha256"] = _sha(origin["run_identity_path"])
    completed["checkpoint"] = origin["checkpoint"]
    completed["checkpoint_sha256"] = origin["checkpoint_sha256"]
    completed["checkpoint_source"] = {
        "run_id": origin["id"], "path": completed["checkpoint"],
        "sha256": completed["checkpoint_sha256"],
    }
    completed["cross_domain_transfer"] = True
    forged = copy.deepcopy(completed)
    if mutation == "run_id":
        forged["checkpoint_source"]["run_id"] = "FORGED"
        message = "run_id mismatch"
    elif mutation == "path":
        other = tmp_path / "other.pt"
        other.write_bytes(Path(completed["checkpoint"]).read_bytes())
        forged["checkpoint_source"]["path"] = str(other)
        message = "path mismatch"
    else:
        forged["cross_domain_transfer"] = False
        message = "cross_domain_transfer mismatch"

    with pytest.raises(RuntimeError, match=message):
        verify_completed_state(forged, [origin])
    verify_completed_state(completed, [origin])


def test_transfer_target_self_consistent_checkpoint_must_equal_authoritative_origin(
        native, tmp_path):
    origin_row = _row(
        native, dataset="source-dataset", input_track="deployment",
        domain="source-domain", trainable=True, checkpoint_training_source=None)
    origin_run = tmp_path / "origin-a"
    _write_result(origin_run, origin_row)
    origin = _completed(origin_row, origin_run)
    origin["id"] = matrix_row_id(
        origin["profile"], origin["dataset"], origin["seed"], origin["input_track"])
    identity_payload = {key: origin[key] for key in ("profile", "seed", "input_track")}
    Path(origin["run_identity_path"]).write_text(json.dumps(identity_payload))
    origin["run_identity_sha256"] = _sha(origin["run_identity_path"])

    target = _row(
        native, input_track="deployment", domain="target-domain", trainable=False,
        checkpoint_training_source=origin["dataset"])
    target_run = tmp_path / "target-b"
    _write_result(target_run, target)
    completed = _completed(target, target_run)
    completed["checkpoint_source"] = {
        "run_id": origin["id"], "path": completed["checkpoint"],
        "sha256": completed["checkpoint_sha256"],
    }
    completed["cross_domain_transfer"] = True

    with pytest.raises(RuntimeError, match="differs from authoritative origin"):
        verify_completed_state(completed, [origin])


def test_trainable_completed_row_rejects_transfer_only_provenance(native, tmp_path):
    row = _row(native, trainable=True)
    run = tmp_path / "trainable"
    _write_result(run, row)
    completed = _completed(row, run)
    completed["checkpoint_source"] = {
        "run_id": "FORGED", "path": completed["checkpoint"],
        "sha256": completed["checkpoint_sha256"],
    }
    completed["cross_domain_transfer"] = False
    with pytest.raises(RuntimeError, match="transfer-only provenance"):
        verify_completed_state(completed)


@pytest.mark.parametrize("mutation", ["omission", "invention"])
def test_completed_binding_applicable_manifest_must_match_current_qualification(
        native, tmp_path, mutation):
    row = _row(native)
    run = tmp_path / mutation
    _write_result(run, row)
    completed = _completed(row, run)
    if mutation == "omission":
        completed["result_artifacts"]["applicable_splits"] = []
    else:
        completed["result_artifacts"]["applicable_splits"].append("invented")

    with pytest.raises(RuntimeError, match="verified evaluation splits|manifest is inconsistent"):
        verify_completed_state(completed)
