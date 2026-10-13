#!/usr/bin/env python3
"""Record server pytest, Ruff, compileall and public CLI checks in one receipt.

The parent process must hold the shared CPU flock. The v1 virtual environment
selects the package without PYTHONPATH. Scientific source changes during checks
invalidate the receipt, even if the individual commands return zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", default="integrated-v1")
    args = parser.parse_args()
    root = args.root.resolve()
    report = root / ".local/reports"
    report.mkdir(parents=True, exist_ok=True)
    source_manifest = root / "source_manifest.json"
    before = digest(source_manifest)
    source = json.loads(source_manifest.read_text())
    files = source["files"]
    for name, expected in files.items():
        if digest(root / name) != expected:
            raise RuntimeError(f"source manifest mismatch before checks: {name}")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", IPB_DATASETS=str(root / "datasets/ipb-v1"))
    python = str(root / ".venv/bin/python")
    cli = str(root / ".venv/bin/ipb")
    junit = report / f"{args.stage}-pytest.xml"
    commands = [
        ("pytest", [python, "tools/ops_v1/cpu_pytest.py", "-q", f"--junitxml={junit}"]),
        ("ruff", [python, "-m", "ruff", "check", "src", "tests"]),
        ("compileall", [python, "-m", "compileall", "-q", "src", "tests", "tools"]),
        ("import_location", [python, "-c", "import inertial_benchmark; print(inertial_benchmark.__file__)"]),
        ("ipb_info", [cli, "info"]),
        ("ipb_cfg", [cli, "cfg", "model=ronin_resnet18", "recipe=unified"]),
        ("ipb_benchmark_dryrun", [cli, "benchmark", "cfg=benchmarks/main.yaml", "dry_run=true"]),
    ]
    results = []
    for name, argv in commands:
        log = report / f"{args.stage}-{name}.log"
        start = time.monotonic()
        with log.open("wb") as stream:
            process = subprocess.run(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        item = {"check": name, "argv": argv, "returncode": process.returncode,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "log": str(log), "log_sha256": digest(log)}
        results.append(item)
        print(json.dumps(item), flush=True)
    stable = digest(source_manifest) == before and all(digest(root / name) == expected
                                                    for name, expected in files.items())
    summary = {}
    skipped = []
    if junit.exists():
        document = ET.parse(junit).getroot()
        suites = [document] if document.tag == "testsuite" else list(document.iter("testsuite"))
        summary = {key: sum(int(s.get(key, 0)) for s in suites)
                   for key in ("tests", "failures", "errors", "skipped")}
        summary["passed"] = summary["tests"] - summary["failures"] - summary["errors"] - summary["skipped"]
        skipped = [{"test": case.get("classname", "") + "." + case.get("name", ""),
                    "reason": item.get("message", "")} for case in document.iter("testcase")
                   for item in case.findall("skipped")]
    receipt = {"schema": "ib.dev-v1.server-checks.v1", "stage": args.stage,
               "created_utc": datetime.now(timezone.utc).isoformat(), "source_manifest_sha256": before,
               "deployment_id": source.get("deployment_id"), "source_stable_during_checks": stable,
               "checks": results, "pytest": summary, "skipped_tests": skipped,
               "junit_sha256": digest(junit) if junit.exists() else None,
               "source_file_count": len(files), "python": python, "implicit_pythonpath": False,
               "child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
               "cpu_memory_scope": "0.5 CPU / 2 GiB; synthetic tests and available data only"}
    target = report / f"{args.stage}-checks.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if stable and all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
