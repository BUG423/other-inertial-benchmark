#!/usr/bin/env python3
"""Record server acceptance with optional per-file pytest process isolation.

The caller holds the shared CPU flock. Isolation preserves the original collected
nodeids and assertions; each test module runs in a new process to release memory.
Missing or duplicated nodeids invalidate acceptance. Source hashes, individual
logs, JUnit files and node reports make interrupted runs reviewable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def resource_snapshot():
    cgroup = Path("/sys/fs/cgroup")
    events = cgroup / "memory.events"
    current = cgroup / "memory.current"
    maximum = cgroup / "memory.max"
    cpu = cgroup / "cpu.max"
    return {"parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "memory_current_bytes": int(current.read_text()) if current.exists() else None,
            "memory_max": maximum.read_text().strip() if maximum.exists() else None,
            "cpu_max": cpu.read_text().strip() if cpu.exists() else None,
            "memory_events": dict((key, int(value)) for key, value in
                                  (line.split() for line in events.read_text().splitlines()))
            if events.exists() else {}}


def execute(root, env, report, label, name, argv, node_path=None):
    log = report / f"{label}-{name}.log"
    selected_env = dict(env)
    if node_path:
        selected_env["IPB_TEST_NODE_REPORT"] = str(node_path)
    before = resource_snapshot()
    start = time.monotonic()
    with log.open("wb") as stream:
        process = subprocess.run(argv, cwd=root, env=selected_env, stdout=stream,
                                 stderr=subprocess.STDOUT)
    after = resource_snapshot()
    item = {"check": name, "argv": argv, "returncode": process.returncode,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "log": str(log), "log_sha256": digest(log), "resources_before": before,
            "resources_after": after, "memory_events_delta": {
                key: value - before["memory_events"].get(key, value)
                for key, value in after["memory_events"].items()}}
    if node_path:
        item["node_report"] = str(node_path)
        item["node_report_sha256"] = digest(node_path) if node_path.exists() else None
    print(json.dumps(item), flush=True)
    return item


def summarize_nodes(expected, evidence):
    collected, records, duplicates = [], {}, []
    invalid_shards = []
    for item, payload in evidence:
        if payload is None:
            invalid_shards.append(item["check"])
            continue
        shard_collected = payload.get("collected_nodeids", [])
        collected.extend(shard_collected)
        if payload.get("duplicate_collection"):
            invalid_shards.append(item["check"])
        for node, entry in payload.get("reports", {}).items():
            if node in records or entry.get("duplicate_phase"):
                duplicates.append(node)
            records[node] = entry
    expected_set, collected_set, recorded_set = set(expected), set(collected), set(records)
    missing = sorted(expected_set - recorded_set)
    pending = sorted(node for node, item in records.items() if item["outcome"] == "pending")
    unexpected = sorted((collected_set | recorded_set) - expected_set)
    complete = (bool(expected) and len(expected) == len(expected_set)
                and len(collected) == len(collected_set)
                and collected_set == expected_set and not missing and not pending
                and not unexpected and not duplicates and not invalid_shards)
    summary = {"tests": len(records), "passed": 0, "failures": 0, "errors": 0, "skipped": 0}
    skipped = []
    for node, item in records.items():
        outcome = item["outcome"]
        if outcome == "passed":
            summary["passed"] += 1
        elif outcome == "skipped":
            summary["skipped"] += 1
            skipped.append({"test": node, "reason": item.get("reason", "")})
        elif outcome == "failed":
            key = "failures" if item["phases"].get("call") == "failed" else "errors"
            summary[key] += 1
    coverage = {"complete": complete, "expected_count": len(expected),
                "unique_collected_count": len(collected_set), "unique_reported_count": len(records),
                "missing_nodeids": missing, "pending_nodeids": pending,
                "unexpected_nodeids": unexpected, "duplicate_nodeids": sorted(set(duplicates)),
                "duplicate_collection": len(collected) != len(collected_set),
                "invalid_shards": invalid_shards}
    return summary, skipped, coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", default="integrated-v1")
    parser.add_argument("--pytest-isolation", choices=("session", "file"), default="session")
    parser.add_argument("--pytest-path", action="append", default=[],
                        help="Explicit test scope; repeat for affected modules. "
                             "Default: full tests.")
    args = parser.parse_args()
    root = args.root.resolve()
    report = root / ".local/reports"
    report.mkdir(parents=True, exist_ok=True)
    if not args.stage or Path(args.stage).name != args.stage:
        parser.error("stage must be a nonempty filename component")
    for selected in args.pytest_path:
        path = (root / selected).resolve()
        if not path.is_relative_to(root / "tests") or not path.exists():
            parser.error("pytest paths must exist inside this deployment's tests directory")
    target = report / f"{args.stage}-checks.json"
    progress = report / f"{args.stage}-progress.json"
    if target.exists() or progress.exists():
        parser.error("stage receipt already exists; select a new stage to preserve prior evidence")
    manifest = root / "source_manifest.json"
    before = digest(manifest)
    source = read_json(manifest)
    files = source["files"]
    if any(digest(root / name) != expected for name, expected in files.items()):
        raise RuntimeError("source manifest mismatch before checks")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("IPB_TEST_NODE_REPORT", None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", MALLOC_ARENA_MAX="2",
               IPB_DATASETS=str(root / "datasets/ipb-v1"))
    python, cli = str(root / ".venv/bin/python"), str(root / ".venv/bin/ipb")
    results, evidence = [], []
    expected = []

    def save_progress(active_check):
        summary, _, coverage = summarize_nodes(expected, evidence)
        payload = {"schema": "ib.dev-v1.server-check-progress.v1", "stage": args.stage,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "source_manifest_sha256": before, "pytest_isolation": args.pytest_isolation,
                   "active_check": active_check, "checks": results, "pytest": summary,
                   "coverage": coverage}
        temporary = progress.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(progress)

    collection = report / f"{args.stage}-collection.json"
    save_progress("collection")
    collect = execute(root, env, report, args.stage, "collection",
                      [python, "tools/ops_v1/cpu_pytest.py", "--collect-only", "-q",
                       *args.pytest_path], collection)
    results.append(collect)
    payload = read_json(collection)
    expected = payload.get("collected_nodeids", []) if payload else []
    save_progress(None)
    if collect["returncode"] == 0 and expected:
        scopes = list(dict.fromkeys(node.split("::", 1)[0] for node in expected)) \
            if args.pytest_isolation == "file" else [args.pytest_path]
        for index, scope in enumerate(scopes, 1):
            name = f"pytest-{index:02d}" if args.pytest_isolation == "file" else "pytest"
            selected = [scope] if isinstance(scope, str) else scope
            node_path = report / f"{args.stage}-{name}-nodes.json"
            junit = report / f"{args.stage}-{name}.xml"
            save_progress(name)
            item = execute(root, env, report, args.stage, name,
                           [python, "tools/ops_v1/cpu_pytest.py", "-vv", *selected,
                            f"--junitxml={junit}"], node_path)
            item["junit"] = str(junit)
            item["junit_sha256"] = digest(junit) if junit.exists() else None
            results.append(item)
            evidence.append((item, read_json(node_path)))
            save_progress(None)
    commands = [
        ("ruff", [python, "-m", "ruff", "check", "src", "tests"]),
        ("compileall", [python, "-m", "compileall", "-q", "src", "tests", "tools"]),
        ("import_location", [python, "-c",
                             "import inertial_benchmark as ib; from pathlib import Path; "
                             "p = Path(ib.__file__).resolve(); print(p); "
                             f"assert p.is_relative_to(Path({str(root / 'src')!r}).resolve()), "
                             "'package must resolve inside this deployment src'"]),
        ("ipb_info", [cli, "info"]),
        ("ipb_cfg", [cli, "cfg", "model=ronin_resnet18", "recipe=unified"]),
        ("ipb_benchmark_dryrun", [cli, "benchmark", "cfg=benchmarks/main.yaml", "dry_run=true"]),
    ]
    for name, argv in commands:
        save_progress(name)
        results.append(execute(root, env, report, args.stage, name, argv))
        save_progress(None)
    stable = digest(manifest) == before and all(digest(root / name) == expected_hash
                                               for name, expected_hash in files.items())
    summary, skipped, coverage = summarize_nodes(expected, evidence)
    receipt = {"schema": "ib.dev-v1.server-checks.v2", "stage": args.stage,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "source_manifest_sha256": before,
               "deployment_id": source.get("deployment_id"), "source_stable_during_checks": stable,
               "checks": results, "pytest": summary, "skipped_tests": skipped,
               "pytest_isolation": args.pytest_isolation,
               "pytest_scope": args.pytest_path or ["tests"],
               "coverage": coverage,
               "collection_sha256": digest(collection) if collection.exists() else None,
               "collection_events": payload.get("collection_events", []) if payload else [],
               "input_roots": {key: value for key, value in env.items()
                               if key == "IPB_DATASETS" or key.startswith("IPB_RAW_")},
               "resource_environment": {key: env.get(key) for key in
                                        ("MALLOC_ARENA_MAX", "OMP_NUM_THREADS",
                                         "OPENBLAS_NUM_THREADS",
                                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
               "source_file_count": len(files), "python": python, "implicit_pythonpath": False,
               "resources": resource_snapshot(),
               "cpu_memory_scope": "actual cgroup limits in resources; one Torch compute thread"}
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    return (0 if stable and coverage["complete"]
            and all(item["returncode"] == 0 for item in results) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
