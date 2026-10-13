#!/usr/bin/env python3
"""Run pytest with one Torch thread and optional durable nodeid evidence."""

import json
import os
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1")

import pytest


class CPUProfile:
    def __init__(self):
        destination = os.environ.get("IPB_TEST_NODE_REPORT")
        self.destination = Path(destination) if destination else None
        self.collected = []
        self.reports = {}
        self.collection_events = []
        self.exitstatus = None
        self.execution_profile = {}

    @pytest.hookimpl(trylast=True)
    def pytest_configure(self, config):
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.execution_profile = {"torch_num_threads": torch.get_num_threads(),
                                  "torch_num_interop_threads": torch.get_num_interop_threads()}

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]
        self.save()

    def pytest_collectreport(self, report):
        if self.destination is not None and (report.failed or report.skipped):
            self.collection_events.append({"nodeid": report.nodeid, "outcome": report.outcome,
                                           "reason": str(report.longrepr)})

    def pytest_runtest_logreport(self, report):
        if self.destination is None:
            return
        item = self.reports.setdefault(report.nodeid, {"phases": {}, "outcome": "pending"})
        if report.when in item["phases"]:
            item["duplicate_phase"] = True
        item["phases"][report.when] = report.outcome
        if report.skipped:
            item["reason"] = str(report.longrepr[2] if isinstance(report.longrepr, tuple)
                                 else report.longrepr)
        phases = item["phases"]
        if "failed" in phases.values():
            item["outcome"] = "failed"
        elif "skipped" in phases.values():
            item["outcome"] = "skipped"
        elif phases.get("call") == phases.get("teardown") == "passed":
            item["outcome"] = "passed"
        if report.when == "teardown":
            self.save()

    def pytest_sessionfinish(self, session, exitstatus):
        self.exitstatus = int(exitstatus)
        self.save()

    def save(self):
        if self.destination is None:
            return
        payload = {"schema": "ib.pytest-node-evidence.v1", "collected_nodeids": self.collected,
                   "duplicate_collection": len(self.collected) != len(set(self.collected)),
                   "reports": self.reports, "exitstatus": self.exitstatus,
                   "collection_events": self.collection_events,
                   "execution_profile": self.execution_profile}
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.destination)


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:], plugins=[CPUProfile()]))
