import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from inertial_benchmark.cfg import EVAL_PROTOCOL_KEYS, load_default  # noqa: E402
from inertial_benchmark.cli import entrypoint  # noqa: E402
from inertial_benchmark.utils import json_load, json_save  # noqa: E402
from inertial_benchmark.utils import reports as reports_module  # noqa: E402
from inertial_benchmark.utils.reports import (  # noqa: E402
    ProtocolMismatch,
    collect_matrix_results,
    validate_matrix_evaluation,
)


def row_id(row):
    value = {key: row[key] for key in ("profile", "dataset", "seed", "input_track")}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def matrix_row(profile, state="qualified", taskview=True, seed=7, **overrides):
    row = {"profile": profile, "dataset": "syn", "seed": seed,
           "input_track": "deployment", "data_root": "/native/syn", "dims": 3,
           "split_sha256": {},
           "main_xyz": True, "domain": "human", "checkpoint_training_source": None,
           "state": state, "reasons": []}
    if taskview:
        row["taskview_receipt"] = {
            "status": "passed",
            "splits": {"train": {"status": "passed"}, "val": {"status": "passed"},
                       "test": {"status": "passed"},
                       "test_empty": {"status": "not_applicable_missing_split"}},
        }
    row.update(overrides)
    row["id"] = row_id(row)
    return row


def write_native(root, splits):
    root = Path(root)
    hashes = {}
    for split, sequence_ids in splits.items():
        path = root / "splits" / f"{split}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{value}\n" for value in sequence_ids), encoding="utf-8")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return root, hashes


def write_eval(matrix, row, split="test", num_sequences=1):
    run = (matrix / "runs" / row["id"] / row["profile"] / Path(row["data_root"]).name
           / f"seed{row['seed']}" / split)
    run.mkdir(parents=True, exist_ok=True)
    default = load_default()
    protocol = {key: default[key] for key in EVAL_PROTOCOL_KEYS}
    protocol["metric_dims"] = row["dims"]
    axes = list("xyz"[:row["dims"]])
    json_save(run / "metrics.json", {
        "mode": "val", "model": row["profile"], "dataset": row["dataset"],
        "split": split, "seed": row["seed"], "num_sequences": num_sequences,
        "num_skipped": 0, "protocol": protocol, "metric_dims_requested": row["dims"],
        "metric_dims_effective": row["dims"], "predicted_axes": axes,
        "metrics": {"ate": 1.25, "rte": 0.5, "d_rte_10m": None},
        "median": {"ate": 1.25, "rte": 0.5, "d_rte_10m": None},
        "std": {"ate": None, "rte": None, "d_rte_10m": None},
        "count": {"ate": 1, "rte": 1, "d_rte_10m": 0},
        "privileged_inputs": [], "efficiency": {"params": 12, "flops": 34.0},
    })
    fields = ["sequence_id", "group_id", "skipped", "metric_dims_requested",
              "metric_dims_effective", "predicted_axes", "ate", "rte", "d_rte_10m"]
    with (run / "sequences.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"sequence_id": "s0", "group_id": "g0", "skipped": "",
                         "metric_dims_requested": row["dims"],
                         "metric_dims_effective": row["dims"],
                         "predicted_axes": ",".join(axes), "ate": 1.25, "rte": 0.5,
                         "d_rte_10m": ""})
    return run


def make_matrix(tmp_path):
    matrix = tmp_path / "matrix"
    native, split_hashes = write_native(
        tmp_path / "native" / "syn", {"test": ["s0"], "test_empty": []})
    source = {"data_root": str(native), "split_sha256": split_hashes}
    available = matrix_row("available", state="qualified", **source)
    invalid = matrix_row("invalid", state="qualified", seed=11, **source)
    running = matrix_row("running", state="running", seed=8, **source)
    unsupported = matrix_row("unsupported", state="unsupported_taskview", seed=9, **source)
    unsupported["reasons"] = ["required orientation is unavailable"]
    unsupported["taskview_receipt"] = {
        "status": "failed", "error": "missing device orientation",
        "splits": {"test": {"status": "passed"}},
    }
    pending = matrix_row(
        "pending", state="pending_qualification", taskview=False, seed=10, **source)
    coverage = {"schema": "ib.dynamic-matrix.v1",
                "source": {"source_manifest_sha256": "source-sha"},
                "coverage": [available, invalid, running, unsupported, pending]}
    json_save(matrix / "coverage.json", coverage)
    for row in (available, invalid):
        json_save(matrix / "run-states" / f"{row['id']}.json", {**row, "state": "completed"})
    run = write_eval(matrix, available)
    write_eval(matrix, invalid, num_sequences=0)
    write_eval(matrix, running)  # A valid-looking residual must still be ignored while running.
    # This legacy zero-sequence artifact is deliberately invalid, but its taskview split is N/A
    # and therefore it must never be admitted or validated.
    write_eval(matrix, available, split="test_empty", num_sequences=0)
    return matrix, available, run


def test_matrix_collection_has_strict_results_and_explicit_unavailable_ledger(tmp_path):
    matrix, available, run = make_matrix(tmp_path)
    before = (matrix / "coverage.json").read_bytes()
    out = tmp_path / "report"
    info = collect_matrix_results(matrix, out)

    assert (matrix / "coverage.json").read_bytes() == before
    results = pd.read_csv(out / "per_sequence.csv")
    assert len(results) == 1
    assert results.loc[0, ["matrix_row_id", "model", "dataset", "split", "seed"]].tolist() == [
        available["id"], "available", "syn", "test", 7]
    ledger = pd.read_csv(out / "availability.csv", keep_default_na=False)
    states = set(zip(ledger["model"], ledger["split"], ledger["availability"]))
    assert ("available", "test", "available") in states
    assert ("invalid", "test", "invalid") in states
    assert ("available", "test_empty", "not_applicable") in states
    assert ("running", "test", "running") in states
    assert ("unsupported", "test", "unsupported") in states
    assert ("pending", "*", "pending") in states
    assert all(ledger.loc[ledger["availability"] != "available", "reason"].str.len() > 0)
    assert info["available_evaluations"] == 1 and info["available_sequence_rows"] == 1
    assert info["availability_counts"] == {
        "available": 1, "invalid": 1, "not_applicable": 3, "running": 1,
        "unsupported": 1, "pending": 1}

    manifest = json_load(out / "artifacts-sha256.json")
    accepted = manifest["accepted_evaluations"]
    assert len(accepted) == 1
    assert accepted[0]["metrics"]["path"] == str(
        (run / "metrics.json").relative_to(matrix))
    assert accepted[0]["sequences"]["sha256"] == hashlib.sha256(
        (run / "sequences.csv").read_bytes()).hexdigest()
    assert accepted[0]["native_split"]["sha256"] == hashlib.sha256(
        (Path(available["data_root"]) / "splits" / "test.txt").read_bytes()).hexdigest()
    assert all("test_empty" not in json.dumps(item) for item in accepted)
    assert any(item["model"] == "running" and item["artifacts"]
               for item in manifest["ignored_evaluations"])
    assert any(item["model"] == "invalid" and item["artifacts"]
               for item in manifest["rejected_evaluations"])
    running_ledger = ledger[(ledger["model"] == "running") & (ledger["split"] == "test")]
    assert running_ledger.iloc[0]["artifact_disposition"] == "ignored"
    assert running_ledger.iloc[0]["metrics_sha256"]
    report = json_load(out / "report.json")
    assert report["artifact_manifest_sha256"] == hashlib.sha256(
        (out / "artifacts-sha256.json").read_bytes()).hexdigest()
    assert (out / "summary.csv").is_file() and (out / "report.json").is_file()


@pytest.mark.parametrize(
    "mutation", ["zero", "identity_model", "identity_dataset", "identity_split",
                 "identity_seed", "json_dims", "csv_dims", "count", "empty", "no_header"])
def test_matrix_evaluation_rejects_invalid_or_inconsistent_small_results(tmp_path, mutation):
    native, split_hashes = write_native(tmp_path / "native", {"test": ["s0"]})
    row = matrix_row("strict", data_root=str(native), split_sha256=split_hashes)
    matrix = tmp_path / "matrix"
    run = write_eval(matrix, row)
    metrics = json_load(run / "metrics.json")
    if mutation == "zero":
        metrics["num_sequences"] = 0
        json_save(run / "metrics.json", metrics)
    elif mutation.startswith("identity_"):
        field = mutation.removeprefix("identity_")
        metrics[field] = -1 if field == "seed" else "other"
        json_save(run / "metrics.json", metrics)
    elif mutation == "json_dims":
        metrics["metric_dims_effective"] = 2
        json_save(run / "metrics.json", metrics)
    elif mutation == "count":
        metrics["num_sequences"] = 2
        json_save(run / "metrics.json", metrics)
    elif mutation == "empty":
        header = (run / "sequences.csv").read_text().splitlines()[0]
        (run / "sequences.csv").write_text(header + "\n")
    elif mutation == "no_header":
        (run / "sequences.csv").write_text("")
    else:
        rows = list(csv.DictReader((run / "sequences.csv").open(newline="", encoding="utf-8")))
        fields = list(rows[0])
        rows[0]["metric_dims_effective"] = "2"
        with (run / "sequences.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(ProtocolMismatch):
        validate_matrix_evaluation(run, row, "test")


@pytest.mark.parametrize("mutation", ["forged", "truncated"])
def test_matrix_collection_rejects_sequence_ids_not_exactly_bound_to_native_split(
        tmp_path, mutation):
    matrix = tmp_path / "matrix"
    expected = ["s0"] if mutation == "forged" else ["s0", "s1"]
    native, split_hashes = write_native(tmp_path / "native", {"test": expected})
    row = matrix_row(
        "identity-bound", state="completed", data_root=str(native),
        split_sha256=split_hashes)
    row["taskview_receipt"]["splits"].pop("test_empty")
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row]})
    run = write_eval(matrix, row)
    if mutation == "forged":
        rows = list(csv.DictReader((run / "sequences.csv").open(newline="", encoding="utf-8")))
        fields = list(rows[0])
        rows[0]["sequence_id"] = "forged"
        with (run / "sequences.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    out = tmp_path / "report"
    info = collect_matrix_results(matrix, out)
    assert info["available_evaluations"] == 0
    ledger = pd.read_csv(out / "availability.csv", keep_default_na=False)
    assert ledger.loc[0, "availability"] == "invalid"
    assert "native split" in ledger.loc[0, "reason"]
    rejected = json_load(out / "artifacts-sha256.json")["rejected_evaluations"]
    assert any(item["path"] == "splits/test.txt" for item in rejected[0]["artifacts"])


@pytest.mark.parametrize(
    "mutation", ["empty_metrics", "missing_median", "missing_std", "missing_count",
                 "missing_core", "count_mismatch", "core_null"])
def test_matrix_collection_rejects_invalid_aggregate_metric_contract(tmp_path, mutation):
    matrix = tmp_path / "matrix"
    native, split_hashes = write_native(tmp_path / "native", {"test": ["s0"]})
    row = matrix_row(
        "aggregate-bound", state="completed", data_root=str(native),
        split_sha256=split_hashes)
    row["taskview_receipt"]["splits"].pop("test_empty")
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row]})
    run = write_eval(matrix, row)
    metrics = json_load(run / "metrics.json")
    if mutation == "empty_metrics":
        metrics["metrics"] = {}
    elif mutation in ("missing_median", "missing_std", "missing_count"):
        metrics.pop(mutation.removeprefix("missing_"))
    elif mutation == "missing_core":
        for name in ("metrics", "median", "std", "count"):
            metrics[name].pop("rte")
    elif mutation == "count_mismatch":
        metrics["count"]["ate"] = 0
    else:
        metrics["metrics"]["ate"] = None
        metrics["median"]["ate"] = None
        metrics["count"]["ate"] = 0
        rows = list(csv.DictReader((run / "sequences.csv").open(newline="", encoding="utf-8")))
        fields = list(rows[0])
        rows[0]["ate"] = ""
        with (run / "sequences.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    json_save(run / "metrics.json", metrics)

    info = collect_matrix_results(matrix, tmp_path / "report")
    assert info["available_evaluations"] == 0
    ledger = pd.read_csv(tmp_path / "report" / "availability.csv", keep_default_na=False)
    assert ledger.loc[0, "availability"] == "invalid"
    assert ledger.loc[0, "artifact_disposition"] == "rejected"


@pytest.mark.parametrize("mutation", ["changed", "symlink"])
def test_matrix_collection_fails_closed_on_native_split_path_or_hash_change(tmp_path, mutation):
    matrix = tmp_path / "matrix"
    native, split_hashes = write_native(tmp_path / "native", {"test": ["s0"]})
    row = matrix_row(
        "source-bound", state="completed", data_root=str(native),
        split_sha256=split_hashes)
    row["taskview_receipt"]["splits"].pop("test_empty")
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row]})
    write_eval(matrix, row)
    split_path = native / "splits" / "test.txt"
    if mutation == "changed":
        split_path.write_text("s0\ns1\n", encoding="utf-8")
        message = "changed from coverage"
    else:
        outside = tmp_path / "outside-split.txt"
        outside.write_text("s0\n", encoding="utf-8")
        split_path.unlink()
        split_path.symlink_to(outside)
        message = "symlink"
    with pytest.raises(ProtocolMismatch, match=message):
        collect_matrix_results(matrix, tmp_path / "report")


@pytest.mark.parametrize(
    "split", ["", ".", "..", "nested/test", "/absolute", "test\\escape", "nul\x00escape"])
def test_matrix_collection_rejects_unsafe_split_components(tmp_path, split):
    matrix = tmp_path / "matrix"
    row = matrix_row("unsafe")
    row["taskview_receipt"]["splits"][split] = {"status": "passed"}
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row]})
    with pytest.raises(ProtocolMismatch, match="safe path component"):
        collect_matrix_results(matrix, tmp_path / "report")


@pytest.mark.parametrize(
    "mutation", ["nonempty", "changed", "deleted_bound", "exists_unbound"])
def test_matrix_collection_fails_closed_on_false_not_applicable_split(tmp_path, mutation):
    matrix = tmp_path / "matrix"
    native, split_hashes = write_native(tmp_path / "native", {"test_empty": []})
    row = matrix_row(
        "na-bound", state="completed", data_root=str(native), split_sha256=split_hashes)
    row["taskview_receipt"]["splits"] = {
        "test_empty": {"status": "not_applicable_missing_split"}}
    path = native / "splits" / "test_empty.txt"
    if mutation == "nonempty":
        path.write_text("s0\n", encoding="utf-8")
        row["split_sha256"]["test_empty.txt"] = hashlib.sha256(path.read_bytes()).hexdigest()
        message = "nonempty but taskview marks it N/A"
    elif mutation == "changed":
        path.write_text("# changed bytes\n", encoding="utf-8")
        message = "changed from coverage split_sha256"
    elif mutation == "deleted_bound":
        path.unlink()
        message = "disappeared after coverage split_sha256"
    else:
        row["split_sha256"].pop("test_empty.txt")
        message = "exists but coverage split_sha256 did not bind it"
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row]})
    with pytest.raises(ProtocolMismatch, match=message):
        collect_matrix_results(matrix, tmp_path / "report")


def test_matrix_collection_accepts_only_proven_empty_or_unbound_missing_na_split(tmp_path):
    matrix = tmp_path / "matrix"
    native, split_hashes = write_native(tmp_path / "native", {"bound_empty": []})
    row = matrix_row(
        "na-proof", state="completed", data_root=str(native), split_sha256=split_hashes)
    row["taskview_receipt"]["splits"] = {
        "bound_empty": {"status": "not_applicable_missing_split"},
        "missing_unbound": {"status": "not_applicable_missing_split"},
    }
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row]})

    out = tmp_path / "report"
    info = collect_matrix_results(matrix, out)
    assert info["availability_counts"] == {"not_applicable": 2}
    ledger = pd.read_csv(out / "availability.csv", keep_default_na=False)
    bound = ledger.loc[ledger["split"] == "bound_empty"].iloc[0]
    missing = ledger.loc[ledger["split"] == "missing_unbound"].iloc[0]
    assert bound["native_split_sha256"] == split_hashes["bound_empty.txt"]
    assert missing["native_split_sha256"] == ""
    evidence = json_load(out / "artifacts-sha256.json")["not_applicable_evaluations"]
    records = {item["split"]: item["native_split"] for item in evidence}
    assert records["bound_empty"]["coverage_hash_bound"] is True
    assert records["bound_empty"]["sha256"] == split_hashes["bound_empty.txt"]
    assert records["missing_unbound"]["coverage_hash_bound"] is False
    assert records["missing_unbound"]["status"] == "missing_unbound"


def test_matrix_collection_rejects_symlinked_result_path(tmp_path):
    matrix = tmp_path / "matrix"
    row = matrix_row("linked", state="completed")
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row]})
    run = write_eval(matrix, row)
    outside = tmp_path / "outside-result"
    shutil.move(str(run), outside)
    run.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProtocolMismatch, match="symlink"):
        collect_matrix_results(matrix, tmp_path / "report")


def test_matrix_collection_rejects_duplicate_coverage_id(tmp_path):
    matrix = tmp_path / "matrix"
    row = matrix_row("duplicate")
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row, dict(row)]})
    with pytest.raises(ProtocolMismatch, match="duplicate coverage row id"):
        collect_matrix_results(matrix, tmp_path / "report")


@pytest.mark.parametrize("schema", [None, "ib.dynamic-matrix.v0"])
def test_matrix_collection_rejects_missing_or_wrong_coverage_schema(tmp_path, schema):
    matrix = tmp_path / "matrix"
    document = {"coverage": [matrix_row("wrong-schema")]}
    if schema is not None:
        document["schema"] = schema
    json_save(matrix / "coverage.json", document)
    with pytest.raises(ProtocolMismatch, match="expected schema"):
        collect_matrix_results(matrix, tmp_path / "report")


@pytest.mark.parametrize("field,wrong", [
    ("dims", 2),
    ("main_xyz", False),
    ("data_root", "/other/native"),
    ("domain", "uav"),
    ("checkpoint_training_source", "euroc"),
    ("taskview_receipt", {"status": "passed", "splits": {
        "test": {"status": "not_applicable_missing_split"}}}),
    ("future_science_field", {"changed": True}),
])
def test_matrix_state_cannot_override_coverage_science(tmp_path, field, wrong):
    matrix = tmp_path / field
    row = matrix_row("locked", future_science_field={"changed": False})
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row]})
    state = {**row, "state": "completed", field: wrong}
    json_save(matrix / "run-states" / f"{row['id']}.json", state)
    with pytest.raises(ProtocolMismatch, match="differs|changed"):
        collect_matrix_results(matrix, tmp_path / f"report-{field}")


def test_matrix_state_nulls_do_not_erase_coverage_science(tmp_path):
    matrix = tmp_path / "matrix"
    native, split_hashes = write_native(tmp_path / "native", {"test": ["s0"]})
    row = matrix_row(
        "null-safe", data_root=str(native), split_sha256=split_hashes)
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row]})
    state = {**row, "state": "completed"}
    for key in ("dims", "main_xyz", "data_root", "domain", "checkpoint_training_source",
                "taskview_receipt"):
        state[key] = None
    json_save(matrix / "run-states" / f"{row['id']}.json", state)
    write_eval(matrix, row)
    info = collect_matrix_results(matrix, tmp_path / "report")
    assert info["available_evaluations"] == 1


@pytest.mark.parametrize("status", ["applicable", "pending", "failed", None])
def test_matrix_collection_rejects_noncanonical_taskview_split_status(tmp_path, status):
    matrix = tmp_path / "matrix"
    row = matrix_row("bad-status")
    row["taskview_receipt"]["splits"]["test"]["status"] = status
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": [row]})
    with pytest.raises(ProtocolMismatch, match="unsupported status/schema"):
        collect_matrix_results(matrix, tmp_path / "report")


def test_matrix_summary_stratum_requires_same_protocol_across_seeds(tmp_path):
    matrix = tmp_path / "matrix"
    native, split_hashes = write_native(tmp_path / "native", {"test": ["s0"]})
    rows = [matrix_row("same", state="completed", seed=seed, data_root=str(native),
                       split_sha256=split_hashes) for seed in (7, 8)]
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": rows})
    write_eval(matrix, rows[0])
    second = write_eval(matrix, rows[1])
    metrics = json_load(second / "metrics.json")
    metrics["protocol"]["eval_stride"] += 1
    json_save(second / "metrics.json", metrics)
    with pytest.raises(ProtocolMismatch, match="summary stratum across seeds"):
        collect_matrix_results(matrix, tmp_path / "report")


def test_matrix_summary_preserves_scientific_strata(tmp_path):
    matrix = tmp_path / "matrix"
    native, split_hashes = write_native(tmp_path / "native", {"test": ["s0"]})
    source = {"data_root": str(native), "split_sha256": split_hashes}
    rows = [
        matrix_row("same", state="completed", seed=7, **source),
        matrix_row("same", state="completed", seed=8, input_track="reference_oracle",
                   dims=2, main_xyz=False, domain="uav", checkpoint_training_source="euroc",
                   **source),
    ]
    json_save(matrix / "coverage.json", {
        "schema": "ib.dynamic-matrix.v1", "coverage": rows})
    for row in rows:
        write_eval(matrix, row)
    collect_matrix_results(matrix, tmp_path / "report")
    summary = pd.read_csv(tmp_path / "report" / "summary.csv", keep_default_na=False)
    assert len(summary) == 2
    assert {"input_track", "matrix_dims", "main_xyz", "domain",
            "checkpoint_training_source"} <= set(summary.columns)
    assert set(summary["input_track"]) == {"deployment", "reference_oracle"}
    assert set(summary["matrix_dims"]) == {2, 3}


def test_matrix_validation_hashes_the_exact_bytes_it_parses(tmp_path, monkeypatch):
    native, split_hashes = write_native(tmp_path / "native", {"test": ["s0"]})
    row = matrix_row("snapshot", data_root=str(native), split_sha256=split_hashes)
    run = write_eval(tmp_path / "matrix", row)
    original = (run / "metrics.json").read_bytes()
    real_snapshot = reports_module._snapshot_file
    changed = False

    def snapshot_then_replace(path, root):
        nonlocal changed
        snapshot = real_snapshot(path, root)
        if Path(path).name == "metrics.json" and not changed:
            changed = True
            metrics = json.loads(original)
            metrics["model"] = "replacement-after-snapshot"
            json_save(path, metrics)
        return snapshot

    monkeypatch.setattr(reports_module, "_snapshot_file", snapshot_then_replace)
    accepted = validate_matrix_evaluation(run, row, "test")
    assert accepted["metadata"]["model"] == "snapshot"
    assert accepted["metrics"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert accepted["metrics"]["sha256"] != hashlib.sha256(
        (run / "metrics.json").read_bytes()).hexdigest()


def test_matrix_report_cli_is_separate_from_strict_legacy_report(tmp_path):
    matrix, _, _ = make_matrix(tmp_path)
    out = tmp_path / "cli-report"
    assert entrypoint(["matrix-report", f"matrix={matrix}", f"out={out}"]) == 0
    assert json_load(out / "report.json")["schema"] == "ib.matrix-results.v1"
