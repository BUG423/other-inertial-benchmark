"""Validate preparation metadata/AST only; NEVER import a model or framework.

This is not a scientific, CPU, capacity or training acceptance test.  It checks
the disabled tail plan and evidence bindings, including the selected TESSERA
architecture identity rather than its conditional upstream SSL proposal.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_SHA = "1897c9ccfa959199ef3f8dbf8c7863dd40d5472d4da9b5551154366458650949"
SEEDS = [1735231569, 843628571, 951150512]
CHANNELS = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
FOLDERS = {
    "vitt3", "edge_recvit", "pfgnet", "binary_attention", "ucan", "mstemba",
    "specscalpel", "tessera_v1", "csan_vit", "mixercseg",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(require_adapters: bool = False) -> dict:
    plan = json.loads((HERE / "tail_queue.json").read_text())
    require(plan["schema"] == "ib.research.cvpr2026-tail-queue.plan.v1", "plan schema")
    require((plan["venue"], plan["venue_year"], plan["venue_track"]) ==
            ("CVPR", 2026, "main_conference"), "wrong venue/year/track")
    require(set(plan["excluded_venue_tracks"]) == {"workshops", "findings"}, "track exclusions")
    for field in ("execution_enabled", "active_queue_enrolled", "formal_2067_rows_changed",
                  "model_runtime_started"):
        require(plan[field] is False, f"preparation must not set {field}")
    require(plan["accepted_method_count_increment"] == 0 and not plan["runtime_receipts"],
            "preparation cannot claim accepted methods or runtime receipts")
    require(plan["existing_source_manifest_sha256"] == SOURCE_SHA, "frozen source binding")
    barrier = plan["tail_barrier"]
    require(barrier["satisfied"] is False and barrier["readiness_receipt"] is None,
            "no evaluated tail readiness exists in this preparation plan")
    require(barrier["status"] == "not_evaluated" and len(barrier["required_conditions"]) == 4,
            "tail conditions incomplete or declared as facts")
    require(barrier["no_stop_restart_or_preemption_of_existing_work"] is True,
            "original work must not be interrupted")
    runtime_steps = {"framework_model_import", "cpu_model_tests", "native_cpu_lifecycle",
                     "gpu_capacity", "training", "validation_and_test"}
    require(runtime_steps <= set(plan["deferred_until_tail"]), "runtime is not fully deferred")
    input_contract = plan["planned_input"]
    require(input_contract["shape"] == "B,6,200" and input_contract["sampling_rate_hz"] == 200,
            "input shape/rate")
    require(input_contract["channels"] == CHANNELS, "ordered six axes")
    require(input_contract["frame"] == "gravity_world" and
            input_contract["orientation_permission"] == "reference_oracle", "input track")
    require(input_contract["gravity_retained"] is True and
            input_contract["extra_inputs"] == [] and
            input_contract["outside_window_future"] is False, "input permission expansion")
    require(plan["planned_output"]["dimensions"] == 3 and
            plan["planned_output"]["z_is_learned_not_padded"] is True and
            plan["planned_output"]["xy_is_slice_of_same_xyz_prediction"] is True,
            "real XYZ/same-prediction XY")
    require(plan["planned_training"]["seeds"] == SEEDS and
            plan["planned_training"]["loss"] == "mse" and
            plan["planned_training"]["initialization"] == "from_scratch", "shared training")

    candidates = plan["method_candidates"]
    require(len(candidates) == plan["expected_independent_methods"] ==
            plan["selected_independent_source_count"] == 10, "exactly ten sources")
    for key in ("id", "paper_title", "cvf_url", "folder"):
        require(len({row[key] for row in candidates}) == 10, f"duplicate {key}")
    require({row["folder"] for row in candidates} == FOLDERS, "unexpected candidate set")
    checked_sources = set()
    excluded = json.loads((REPO / "docs/models/excluded-methods.json").read_text())
    # Owner exclusion concerns papers proposed by Shanshan Zhang, not every
    # unrelated coauthor of an excluded paper.
    require(any("Shanshan Zhang" in row.get("authors", []) for row in excluded["methods"]),
            "owner author exclusion record missing")
    for row in candidates:
        require(not ({"Shanshan Zhang", "张珊珊"} & set(row["paper_authors"])),
                f"excluded author: {row['id']}")
        evidence_path = HERE / row["evidence_file"]
        require(evidence_path.parent == HERE and evidence_path.is_file(), "evidence path")
        require(digest(evidence_path) == row["evidence_file_sha256"], "source evidence drift")
        evidence = json.loads(evidence_path.read_text())
        matches = [c for c in evidence["candidates"] if c["id"] == row["evidence_candidate_id"]]
        require(len(matches) == 1, "candidate does not bind exactly one source record")
        source = matches[0]
        paper = source.get("paper", source)
        repo = source.get("repository", {})
        require(paper["venue"] == "CVPR 2026 main conference", "main paper qualification")
        require(paper["title"] == row["paper_title"] and paper["authors"] == row["paper_authors"]
                and paper["cvf_url"] == row["cvf_url"], "paper identity mismatch")
        require(row["cvf_url"].startswith("https://openaccess.thecvf.com/content/CVPR2026/html/")
                and row["cvf_url"].endswith("_CVPR_2026_paper.html"), "non-main CVF URL")
        require(HEX64.fullmatch(row["pdf_sha256"]) is not None and
                paper["pdf_sha256"] == row["pdf_sha256"] and paper["pdf_bytes"] > 0,
                "complete primary PDF binding")
        commit = repo.get("commit", source.get("commit"))
        require(commit == row["upstream_commit"] and HEX40.fullmatch(commit) is not None,
                "immutable upstream commit")
        require(repo.get("url", source.get("official_repo")) == row["official_repo"],
                "author repository identity")
        license_record = repo.get("license", source.get("license"))
        require(license_record == row["license"] and license_record["spdx"] in
                {"MIT", "Apache-2.0"}, "fixed license record")
        license_files = license_record.get("files", [license_record])
        require(bool(license_files), "empty license files")
        for item in license_files:
            require(item["bytes"] > 0 and HEX64.fullmatch(item["sha256"]) is not None,
                    "license bytes/hash")
        require(row["core_sources"] == source["core_sources"] and bool(row["core_sources"]),
                "actual core binding")
        for core in row["core_sources"]:
            require(core["bytes"] > 0 and HEX64.fullmatch(core["sha256"]) is not None,
                    "core bytes/hash")
        for field in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled",
                      "source_specific_auxiliary_losses"):
            require(row[field] is False, f"undeclared runtime/loss in {row['id']}")
        require(row["selected_recipe"] == "scratch_supervised_mse", "selected training recipe")
        checked_sources.add(row["evidence_file"])

    tessera = next(row for row in candidates if row["folder"] == "tessera_v1")
    require(tessera["id"] == "tessera_v1_dualstream_architecture_velocity" and
            tessera["id"] != tessera["evidence_candidate_id"] and
            "omits BT" in tessera["selected_scope"], "TESSERA architecture/SSL identity bridge")
    legacy = plan["legacy_three_ports"]
    require(legacy["artifacts_retained"] is True and legacy["count_toward_cvpr2026_target"]
            is False and legacy["old_pilot_is_current_execution_plan"] is False,
            "legacy three ports must not count or run as current pilot")
    pilot = plan["proposed_pilot"]
    require(pilot["candidate_cells"] == 60 and pilot["status"] == "proposal_not_frozen_not_enrolled"
            and pilot["full_13_source_expansion_authorized"] is False, "pilot proposal boundary")

    parsed = []
    for path in sorted(HERE.rglob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
        parsed.append(str(path.relative_to(HERE)))
    for path in sorted(HERE.rglob("*.json")):
        json.loads(path.read_text())
    if require_adapters:
        cfg_tree = ast.parse((REPO / "src/inertial_benchmark/cfg/__init__.py").read_text())
        allowed_input_keys = None
        for node in cfg_tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                    target.id == "INPUT_KEYS" for target in node.targets):
                allowed_input_keys = set(ast.literal_eval(node.value))
        require(allowed_input_keys is not None, "cannot statically read canonical cfg INPUT_KEYS")
        lifecycle_tree = ast.parse((HERE / "test_native_lifecycle.py").read_text())
        lifecycle_classes = None
        for node in lifecycle_tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                    target.id == "CLASSES" for target in node.targets):
                lifecycle_classes = ast.literal_eval(node.value)
        require(lifecycle_classes is not None and set(lifecycle_classes) == FOLDERS,
                "native lifecycle class mapping must cover exactly these ten adapters")
        for row in candidates:
            for filename in ("model.py", "model_3d.yaml", "provenance.json",
                             "LICENSE.upstream", "test_scientific_contract.py"):
                require((HERE / row["folder"] / filename).is_file(),
                        f"missing adapter artifact: {row['folder']}/{filename}")
            provenance = json.loads((HERE / row["folder"] / "provenance.json").read_text())
            require(provenance["implementation_status"] == "static_draft", "adapter stage")
            for field in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled",
                          "registered"):
                require(provenance[field] is False, "adapter must not claim runtime admission")
            for filename, expected_sha in provenance.get("local_artifact_sha256", {}).items():
                require(filename in {"model.py", "model_3d.yaml", "LICENSE.upstream",
                                     "test_scientific_contract.py"}, "unknown local artifact hash path")
                require(digest(HERE / row["folder"] / filename) == expected_sha,
                        "provenance internal local-artifact hash drift")
            source_reference = provenance.get("source_evidence")
            if source_reference:
                require((HERE / row["folder"] / source_reference["path"]).resolve(strict=True) ==
                        (HERE / row["evidence_file"]).resolve(strict=True) and
                        source_reference["sha256"] == row["evidence_file_sha256"],
                        "provenance source-evidence reference drift")
            model_tree = ast.parse((HERE / row["folder"] / "model.py").read_text())
            base_classes = [node.name for node in model_tree.body if isinstance(node, ast.ClassDef)
                            and any(isinstance(base, ast.Name) and base.id == "BaseModel"
                                    for base in node.bases)]
            require(base_classes == [lifecycle_classes[row["folder"]]] ==
                    [provenance["class_name"]], "exported adapter/native-lifecycle class mismatch")
            require(not any(isinstance(node, ast.Name) and node.id == "register_model"
                            for node in ast.walk(model_tree)), "unexpected model registration")
            # Limited line-level check, not a complete YAML/schema parser. All
            # drafts use a flat input mapping; never import cfg (it uses Torch).
            in_input = False
            input_keys = set()
            for line in (HERE / row["folder"] / "model_3d.yaml").read_text().splitlines():
                if line == "input:":
                    in_input = True
                    continue
                if in_input and line and not line[0].isspace() and not line.startswith("#"):
                    in_input = False
                if in_input:
                    match = re.match(r"^  ([a-z_]+):", line)
                    if match:
                        input_keys.add(match.group(1))
            require(bool(input_keys) and input_keys <= allowed_input_keys,
                    f"{row['folder']} draft input has unknown cfg keys {input_keys - allowed_input_keys}")
            artifacts = row.get("adapter_artifacts", [])
            expected = {f"{row['folder']}/{name}" for name in
                        ("model.py", "model_3d.yaml", "provenance.json", "LICENSE.upstream",
                         "test_scientific_contract.py")}
            require(len(artifacts) == 5 and {a["path"] for a in artifacts} == expected,
                    "adapter byte bindings must cover the exact five files")
            for artifact in artifacts:
                target = (HERE / artifact["path"]).resolve(strict=True)
                require(target.is_relative_to(HERE) and target.is_file(), "adapter path escape")
                require(HEX64.fullmatch(artifact["sha256"]) is not None and
                        digest(target) == artifact["sha256"], "adapter byte drift")
        shared = plan.get("shared_adapter_artifacts", [])
        require(len(shared) == 3 and {a["path"] for a in shared} ==
                {"conftest.py", "test_native_lifecycle.py", "check_static_manifest.py"},
                "shared runtime-gate/lifecycle/checker bindings")
        for artifact in shared:
            require(digest(HERE / artifact["path"]) == artifact["sha256"], "shared artifact drift")
    return {
        "schema": "ib.research.cvpr2026.static-metadata-check.v1",
        "status": "passed",
        "qualified_primary_sources": 10,
        "evidence_files_checked": sorted(checked_sources),
        "python_ast_files": parsed,
        "adapter_file_presence_required": require_adapters,
        "license_notice_holds": [row["folder"] for row in candidates
                                 if row.get("license_notice_status") == "hold_notice_review"],
        "legal_reuse_or_distribution_accepted": False,
        "framework_imported": False,
        "model_runtime_or_scientific_acceptance": False,
        "execution_enabled": False,
        "tail_barrier_evaluated": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-adapters", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.require_adapters), indent=2))
