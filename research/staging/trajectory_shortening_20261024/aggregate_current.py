"""Read-only aggregation of current formal PLR results on the Agent.

Usage on the target runtime:
  python aggregate_current.py --matrix /path/to/.local/matrix/main

It consumes only completed ``run-states`` and primary ``test/metrics.json``
files.  It never trains, edits coverage, fits a gain, or selects a model.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path


HUMAN = {"ridi", "ronin", "oxiod", "idol", "imunet", "tlio", "rnin"}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def read_rows(matrix: Path):
    coverage = json.loads((matrix / "coverage.json").read_text())["coverage"]
    for row in coverage:
        state_path = matrix / "run-states" / f"{row['id']}.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if state.get("state") != "completed":
            continue
        metrics_path = (matrix / "runs" / row["id"] / row["profile"] / row["dataset"]
                        / f"seed{row['seed']}" / "test/metrics.json")
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text()).get("metrics", {})
        if not finite(metrics.get("plr")):
            continue
        yield {**row, **state, "metrics": metrics}


def summary(rows):
    values = [float(row["metrics"]["plr"]) for row in rows]
    return {
        "n": len(values),
        "below_one": sum(value < 1 for value in values),
        "fraction_below_one": sum(value < 1 for value in values) / len(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    rows = list(read_rows(args.matrix))
    paired = collections.defaultdict(list)
    for row in rows:
        paired[(row["profile"], row["dataset"])].append(row["metrics"]["plr"])
    result = {
        "schema": "ib.trajectory-shortening.current-readonly.v1",
        "rows": summary(rows),
        "human_in_domain": summary([row for row in rows if row["dataset"] in HUMAN]),
        "uav_robot_test_only": summary([row for row in rows if row["dataset"] not in HUMAN]),
        "paired_groups": {
            "n": len(paired),
            "groups_with_at_least_three_seeds": sum(len(values) >= 3 for values in paired.values()),
            "fraction_group_mean_below_one": (
                sum(statistics.fmean(values) < 1 for values in paired.values()) / len(paired)
                if paired else None
            ),
        },
        "claims_proven": [],
        "read_only": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
