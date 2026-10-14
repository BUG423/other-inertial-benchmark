"""Fail closed before collecting any new model tests without a tail receipt.

These tests are intentionally not part of the active ``tests/`` suite.  A
receipt is an explicit audited handoff, not a generated permission or proof
that a scheduler has already finished.  This file does not import torch.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def pytest_collect_file(file_path, parent):
    if file_path.suffix != ".py" or not file_path.name.startswith("test_"):
        return None
    value = os.environ.get("IPB_CVPR2026_TAIL_READINESS")
    if not value:
        raise pytest.UsageError(
            "CVPR2026 model tests are deferred: set IPB_CVPR2026_TAIL_READINESS "
            "only after audited original-queue terminal disposition and resource handoff."
        )
    try:
        receipt_path = Path(value).resolve(strict=True)
        if not str(receipt_path).startswith(
            "/root/autodl-tmp/other-inertial-benchmark-v1/.local/"
        ):
            raise ValueError("tail receipt must reside in the isolated AutoDL private evidence scope")
        receipt = json.loads(receipt_path.read_text())
        plan_path = HERE / "tail_queue.json"
        plan = json.loads(plan_path.read_text())
        if receipt.get("schema") != "ib.research.cvpr2026-tail-readiness.v1":
            raise ValueError("wrong receipt schema")
        if receipt.get("status") != "ready_for_model_validation":
            raise ValueError("not ready for model validation")
        if receipt.get("tail_plan_sha256") != hashlib.sha256(plan_path.read_bytes()).hexdigest():
            raise ValueError("tail receipt does not bind these exact preparation bytes")
        if receipt.get("existing_source_manifest_sha256") != plan["existing_source_manifest_sha256"]:
            raise ValueError("original frozen source mismatch")
        conditions = receipt.get("conditions", {})
        if any(conditions.get(name) is not True for name in plan["tail_barrier"]["required_conditions"]):
            raise ValueError("all original-queue disposition and resource handoff conditions required")
        if not receipt.get("reviewed_utc") or not receipt.get("reviewer"):
            raise ValueError("explicit audited handoff reviewer/time missing")
        references = receipt.get("evidence_artifacts", [])
        if not references:
            raise ValueError("handoff must bind terminal and resource evidence artifacts")
        for artifact in references:
            target = Path(artifact["path"]).resolve(strict=True)
            if not str(target).startswith("/root/autodl-tmp/other-inertial-benchmark-v1/.local/"):
                raise ValueError("handoff evidence outside isolated server scope")
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != artifact["sha256"]:
                raise ValueError("handoff evidence hash mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise pytest.UsageError(f"CVPR2026 tail admission denied: {exc}") from exc
    # Returning None permits normal pytest collection; no registration,
    # deployment, training or mutation of the original plan happens here.
    return None
