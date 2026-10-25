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
SMALL_ARTIFACT_LIMIT = 8 * 1024 * 1024
HANDOFF_TOTAL_LIMIT = 32 * 1024 * 1024
ADAPTER_TOTAL_LIMIT = 16 * 1024 * 1024


def _bounded_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f"evidence exceeds explicit small-file budget: {path.name}")
    return value


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
        private_evidence_root = os.environ.get(
            "IPB_V1_PRIVATE_EVIDENCE_ROOT", "<agent-project-root>/.local"
        )
        if not str(receipt_path).startswith(private_evidence_root.rstrip("/") + "/"):
            raise ValueError("tail receipt must reside in the isolated AutoDL private evidence scope")
        receipt = json.loads(_bounded_bytes(receipt_path, 256 * 1024))
        plan_path = HERE / "tail_queue.json"
        plan_bytes = _bounded_bytes(plan_path, 2 * 1024 * 1024)
        plan = json.loads(plan_bytes)
        if receipt.get("schema") != "ib.research.cvpr2026-tail-readiness.v1":
            raise ValueError("wrong receipt schema")
        if receipt.get("status") != "ready_for_model_validation":
            raise ValueError("not ready for model validation")
        if receipt.get("tail_plan_sha256") != hashlib.sha256(plan_bytes).hexdigest():
            raise ValueError("tail receipt does not bind these exact preparation bytes")
        if receipt.get("existing_source_manifest_sha256") != plan["existing_source_manifest_sha256"]:
            raise ValueError("original frozen source mismatch")
        candidates = plan.get("method_candidates", [])
        if len(candidates) != 10:
            raise ValueError("tail plan must bind exactly ten independent candidates")
        adapter_artifacts = list(plan.get("shared_adapter_artifacts", []))
        if len(adapter_artifacts) != 3:
            raise ValueError("shared checker/guard/lifecycle byte bindings missing")
        for candidate in candidates:
            if candidate.get("license_notice_status") == "hold_notice_review":
                raise ValueError("upstream copyright-notice review must close before batch execution")
            artifacts = candidate.get("adapter_artifacts", [])
            if len(artifacts) != 5:
                raise ValueError("model/profile/provenance/license/test byte bindings missing")
            adapter_artifacts.extend(artifacts)
            adapter_artifacts.append({"path": candidate["evidence_file"],
                                      "sha256": candidate["evidence_file_sha256"]})
        seen = set()
        used = 0
        for artifact in adapter_artifacts:
            relative = Path(artifact["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid adapter artifact relative path")
            target = (HERE / relative).resolve(strict=True)
            if not target.is_relative_to(HERE) or not target.is_file():
                raise ValueError("adapter artifact escapes admitted source scope")
            if target in seen:
                continue  # repeated source-evidence file; verified once below
            seen.add(target)
            value = _bounded_bytes(target, min(SMALL_ARTIFACT_LIMIT, ADAPTER_TOTAL_LIMIT - used))
            used += len(value)
            if hashlib.sha256(value).hexdigest() != artifact["sha256"]:
                raise ValueError("adapter bytes changed after tail plan/receipt binding")
        conditions = receipt.get("conditions", {})
        if any(conditions.get(name) is not True for name in plan["tail_barrier"]["required_conditions"]):
            raise ValueError("all original-queue disposition and resource handoff conditions required")
        if not receipt.get("reviewed_utc") or not receipt.get("reviewer"):
            raise ValueError("explicit audited handoff reviewer/time missing")
        references = receipt.get("evidence_artifacts", [])
        if not references or len(references) > 32:
            raise ValueError("handoff must bind terminal and resource evidence artifacts")
        used = 0
        for artifact in references:
            target = Path(artifact["path"]).resolve(strict=True)
            if not str(target).startswith(private_evidence_root.rstrip("/") + "/"):
                raise ValueError("handoff evidence outside isolated server scope")
            if target.suffix.lower() not in {".json", ".csv", ".md", ".txt"}:
                raise ValueError("handoff artifacts must be small metadata, never checkpoints/datasets")
            value = _bounded_bytes(target, min(SMALL_ARTIFACT_LIMIT, HANDOFF_TOTAL_LIMIT - used))
            used += len(value)
            actual = hashlib.sha256(value).hexdigest()
            if actual != artifact["sha256"]:
                raise ValueError("handoff evidence hash mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise pytest.UsageError(f"CVPR2026 tail admission denied: {exc}") from exc
    # Returning None permits normal pytest collection; no registration,
    # deployment, training or mutation of the original plan happens here.
    return None
