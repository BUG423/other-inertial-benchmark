"""Read-only decomposition of the observed trajectory-shortening phenomenon.

This tool consumes completed ``metrics.json`` and run-state metadata only.  It
does not load IMU arrays, predictions or checkpoints, and it never changes a
formal run.  The output is a hypothesis audit, not a causal estimate from a
randomized intervention.

The decomposition separates four mechanisms that are otherwise conflated by
one PLR number:

* ``oracle_ratio``: loss caused by the reference/oracle trajectory protocol;
* ``plr_oracle`` and ``speed_ratio``: model output magnitude relative to that
  oracle;
* direction error versus speed bias: radial underestimation versus directional
  ambiguity;
* train/test speed-ratio gap: a possible train/eval mode or domain effect.

Target and loss tables are descriptive because profiles do not randomize these
factors.  They must not be reported as causal effects.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


FIELDS = (
    "plr", "plr_dense", "plr_oracle", "oracle_ratio", "speed_ratio", "speed_bias",
    "along_bias", "cross_rmse", "dir_err_mean", "vel_rmse", "ate",
    "t_rte_10s",
)


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def median(values):
    values = sorted(value for value in values if value is not None)
    return statistics.median(values) if values else None


def fraction(values, predicate):
    values = [value for value in values if value is not None]
    return sum(predicate(value) for value in values) / len(values) if values else None


def describe(rows, key):
    values = [row.get(key) for row in rows]
    return {
        "n": sum(value is not None for value in values),
        "median": median(values),
        "mean": statistics.fmean(value for value in values if value is not None)
        if any(value is not None for value in values) else None,
        "fraction_below_one": fraction(values, lambda value: value < 1.0),
    }


def correlation(rows, left, right):
    pairs = [(row[left], row[right]) for row in rows
             if row.get(left) is not None and row.get(right) is not None]
    if len(pairs) < 3:
        return None
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    mx, my = statistics.fmean(x), statistics.fmean(y)
    xx = sum((value - mx) ** 2 for value in x)
    yy = sum((value - my) ** 2 for value in y)
    return sum((a - mx) * (b - my) for a, b in pairs) / math.sqrt(xx * yy) if xx and yy else None


def load_json(path, limit=8 * 1024 * 1024):
    if path.stat().st_size > limit:
        raise ValueError(f"metadata exceeds bound: {path}")
    return json.loads(path.read_bytes())


def read_cell(matrix, row):
    state_path = matrix / "run-states" / f"{row['id']}.json"
    state = load_json(state_path, 2 * 1024 * 1024) if state_path.is_file() else {}
    if state.get("state", row.get("state")) != "completed":
        return None
    if state.get("domain") != "human" or state.get("dims") != 3:
        return None
    if state.get("classic") or not state.get("trainable", row.get("trainable", False)):
        return None
    spec = state.get("scientific_input_spec") or {}
    run_root = matrix / "runs" / row["id"] / row["profile"] / row["dataset"] / f"seed{row['seed']}"
    test_path = run_root / "test" / "metrics.json"
    if not test_path.is_file():
        return None
    test = load_json(test_path)
    if (test.get("model") != row["profile"] or test.get("dataset") != row["dataset"]
            or test.get("seed") != row["seed"] or test.get("split") != "test"):
        raise ValueError(f"result identity mismatch: {row['id']}")
    metrics = test.get("metrics") or {}
    result = {
        "id": row["id"], "profile": row["profile"],
        "architecture": state.get("architecture", row.get("architecture")),
        "dataset": row["dataset"], "seed": row["seed"],
        "target": spec.get("target"), "window": spec.get("window"),
        "input_track": state.get("input_track"),
        "scientific_loss": state.get("scientific_loss"),
    }
    for key in FIELDS:
        result[key] = finite(metrics.get(key))
    train_path = run_root / "train" / "metrics.json"
    if train_path.is_file():
        train = load_json(train_path)
        train_metrics = train.get("metrics") or {}
        result["train_speed_ratio"] = finite(train_metrics.get("speed_ratio"))
        if result["train_speed_ratio"] is not None and result["speed_ratio"] is not None:
            result["train_test_speed_gap"] = result["speed_ratio"] - result["train_speed_ratio"]
    else:
        result["train_speed_ratio"] = result["train_test_speed_gap"] = None
    return result


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return {
        name: {field: describe(items, field) for field in FIELDS}
        for name, items in sorted(groups.items()) if len(items) >= 5
    }


def audit(matrix, out):
    coverage_path = matrix / "coverage.json"
    coverage_bytes = coverage_path.read_bytes()
    coverage = json.loads(coverage_bytes)
    if len(coverage.get("coverage", [])) != 2067:
        raise ValueError("formal scope is not the frozen 2067 rows")
    rows, rejected = [], []
    for row in coverage["coverage"]:
        try:
            value = read_cell(matrix, row)
            if value is not None:
                rows.append(value)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            rejected.append({"id": row.get("id"), "error": str(exc)})

    canonical = [row for row in rows if row["window"] == 200
                 and row["target"] == "avg_velocity"
                 and row["input_track"] == "reference_oracle"]
    low_direction = [row for row in canonical if row.get("dir_err_mean") is not None
                     and row["dir_err_mean"] < 20]
    high_direction = [row for row in canonical if row.get("dir_err_mean") is not None
                      and row["dir_err_mean"] >= 45]
    protocol = describe(canonical, "oracle_ratio")
    model = describe(canonical, "plr_oracle")
    speed = describe(canonical, "speed_ratio")
    train_test = describe(canonical, "train_test_speed_gap")
    same_direction = {
        "low_dir_err_lt20": {"n": len(low_direction), "speed_ratio": describe(low_direction, "speed_ratio"),
                              "speed_bias": describe(low_direction, "speed_bias"),
                              "along_bias": describe(low_direction, "along_bias")},
        "high_dir_err_ge45": {"n": len(high_direction), "speed_ratio": describe(high_direction, "speed_ratio"),
                               "speed_bias": describe(high_direction, "speed_bias"),
                               "along_bias": describe(high_direction, "along_bias")},
    }
    # The framework's exact identity uses dense PLR.  ``plr`` is sampled at a
    # different reference resolution and is intentionally not used here.
    decomposition_errors = []
    for row in rows:
        if all(row.get(key) is not None for key in ("plr_dense", "plr_oracle", "oracle_ratio")):
            decomposition_errors.append(abs(row["plr_dense"] - row["plr_oracle"] * row["oracle_ratio"]))
    result = {
        "schema": "ib.trajectory-cause-decomposition.v1",
        "coverage_sha256": hashlib.sha256(coverage_bytes).hexdigest(),
        "completed_human_xyz_trainable_cells": len(rows),
        "rejected": rejected,
        "canonical_t200_avg_velocity_reference_oracle": {
            "n": len(canonical), "protocol_oracle_ratio": protocol,
            "model_plr_oracle": model, "speed_ratio": speed,
            "train_test_speed_gap": train_test,
            "direction_strata": same_direction,
            "correlations": {
                "speed_ratio_vs_direction_error": correlation(canonical, "speed_ratio", "dir_err_mean"),
                "speed_ratio_vs_speed_bias": correlation(canonical, "speed_ratio", "speed_bias"),
                "speed_ratio_vs_cross_rmse": correlation(canonical, "speed_ratio", "cross_rmse"),
                "plr_oracle_vs_speed_ratio": correlation(canonical, "plr_oracle", "speed_ratio"),
            },
        },
        "by_target_descriptive": grouped(rows, "target"),
        "by_loss_descriptive": grouped(rows, "scientific_loss"),
        "identity": {
            "rows_checked": len(decomposition_errors),
            "max_abs_plr_identity_error": max(decomposition_errors) if decomposition_errors else None,
            "formula": "plr_dense = plr_oracle * oracle_ratio",
            "caveat": "metrics.json values are aggregate summaries; exact identity is checked per recording by audit_recordings.py, so aggregate ratios need not multiply exactly.",
        },
        "interpretation": {
            "protocol": "oracle_ratio measures reference/metric smoothing; it cannot explain model-side plr_oracle.",
            "radial": "speed_bias and along_bias below zero with low direction error support radial magnitude underestimation.",
            "direction": "direction strata and speed_ratio-vs-direction correlation test whether shortening is merely vector cancellation.",
            "mode_gap": "train_test_speed_gap tests train/eval mode or domain shift; a small gap with train-side shortening argues against dropout as the primary cause.",
            "confounding": "target and loss tables are descriptive; profiles do not randomize target or loss, so they are modifiers/hypotheses rather than causal effects.",
        },
        "full_chain_admission": False,
    }
    out.mkdir(parents=True, exist_ok=False)
    summary_bytes = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    (out / "summary.json").write_bytes(summary_bytes)
    (out / "input-hash.json").write_text(json.dumps({"coverage_sha256": result["coverage_sha256"]}, indent=2) + "\n")
    status_path = out.parent.parent / "cause-status.json"
    status_tmp = status_path.with_suffix(".tmp")
    status_tmp.write_text(json.dumps({
        "schema": "ib.trajectory-cause-status.v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "leaf": str(out.parent),
        "summary": str(out / "summary.json"),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "canonical": result["canonical_t200_avg_velocity_reference_oracle"],
        "full_chain_admission": False,
    }, indent=2, ensure_ascii=False) + "\n")
    status_tmp.replace(status_path)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.matrix, args.out)
