#!/usr/bin/env python3
"""Post a bounded, read-only formal-matrix status report to GitHub Issue #1.

The reporter reads the current matrix and run-state files over the existing
private SSH config. It never launches or signals benchmark work. A stale
matrix snapshot is reported as stale evidence instead of being presented as a
live result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "BUG423/other-inertial-benchmark"
ISSUE = "1"
CONFIG = ROOT / ".local/autodl-dual4090.json"
REMOTE_HELPER = ROOT / ".local/acquisition/ops/.venv/bin/python"
STATE_DIR = Path.home() / ".local/state/other-inertial-benchmark-issue-reporter"
LOCK_PATH = STATE_DIR / "reporter.lock"
MENTIONS = ("@BUG423", "@RicardoCDUT", "@Sarah-123886")
MAX_METADATA_BYTES = 32 * 1024 * 1024


REMOTE_SCRIPT = dedent(
    r"""
    import datetime, hashlib, json, os, socket, subprocess
    from pathlib import Path

    root = Path('/root/autodl-tmp/other-inertial-benchmark-v1')
    matrix = root / '.local/matrix/main'
    coverage_path = matrix / 'coverage.json'
    if not coverage_path.is_file():
        raise RuntimeError('coverage.json missing')
    if coverage_path.stat().st_size > 32 * 1024 * 1024:
        raise RuntimeError('coverage.json exceeds bound')
    coverage_bytes = coverage_path.read_bytes()
    coverage = json.loads(coverage_bytes)
    rows = coverage.get('coverage')
    if not isinstance(rows, list) or len(rows) != 2067:
        raise RuntimeError('formal coverage is not exactly 2067 rows')

    counts = {}
    active = []
    states = matrix / 'run-states'
    for row in rows:
        identifier = row.get('id')
        if not isinstance(identifier, str) or not identifier:
            raise RuntimeError('row id missing')
        state_path = states / (identifier + '.json')
        value = row
        if state_path.is_file():
            if state_path.stat().st_size > 2 * 1024 * 1024:
                raise RuntimeError('run-state exceeds bound')
            value = {**row, **json.loads(state_path.read_bytes())}
        state = value.get('state', 'unknown')
        counts[state] = counts.get(state, 0) + 1
        if state == 'running':
            active.append({key: value.get(key) for key in ('id', 'profile', 'dataset', 'seed', 'state')})

    process_text = subprocess.run(
        ['bash', '-lc', "ps -eo pid=,args= | egrep '(full_matrix|benchmark|train)' | egrep -v 'egrep|grep' | head -40"],
        text=True, capture_output=True, check=False,
    ).stdout.splitlines()
    now = datetime.datetime.now(datetime.timezone.utc)
    mtime = datetime.datetime.fromtimestamp(coverage_path.stat().st_mtime, datetime.timezone.utc)
    print(json.dumps({
        'observed_utc': now.isoformat(),
        'hostname': socket.gethostname(),
        'coverage_mtime_utc': mtime.isoformat(),
        'coverage_sha256': hashlib.sha256(coverage_bytes).hexdigest(),
        'full_scope': len(rows),
        'counts': counts,
        'active': active,
        'matching_processes': process_text,
    }, ensure_ascii=False))
    """
).strip()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run(argv: list[str], timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=ROOT, stdin=subprocess.DEVNULL, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        check=False,
    )


def fetch_remote() -> dict:
    if not CONFIG.is_file() or CONFIG.stat().st_mode & 0o077:
        raise RuntimeError("private SSH config is missing or not mode 0600")
    helper = REMOTE_HELPER if REMOTE_HELPER.is_file() else Path(sys.executable)
    command = [
        str(helper), str(ROOT / "tools/ops_v1/remote.py"),
        "--config", str(CONFIG), "--timeout", "60", "exec",
        "/root/miniconda3/bin/python -c " + shlex.quote(REMOTE_SCRIPT),
    ]
    result = run(command, timeout=90)
    if result.returncode:
        raise RuntimeError(f"remote read failed with exit {result.returncode}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("remote read returned no metadata")
    value = json.loads(lines[-1])
    if value.get("full_scope") != 2067:
        raise RuntimeError("remote formal scope changed")
    return value


def write_receipt(value: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / "last-read.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def render(value: dict) -> tuple[str, str]:
    now = utc_now()
    observed = datetime.fromisoformat(value["observed_utc"])
    mtime = datetime.fromisoformat(value["coverage_mtime_utc"])
    age_hours = max(0.0, (now - mtime).total_seconds() / 3600.0)
    fresh = age_hours <= 6.0
    status = "fresh" if fresh else "stale_snapshot"
    marker = f"<!-- ib-live-matrix-reporter/{now:%Y%m%dT%H%M%SZ} -->"
    counts = value.get("counts", {})
    active = value.get("active", [])
    lines = [
        marker,
        f"### 正式矩阵只读状态 · {now.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        *MENTIONS,
        "",
        f"状态源：{status}；主机 `{value.get('hostname')}`；coverage SHA256 `{value.get('coverage_sha256')}`。",
        f"coverage 文件最后修改：{value.get('coverage_mtime_utc')}（约 {age_hours:.1f} 小时前）。",
        f"正式范围：{value.get('full_scope')}；文件中记录的 completed={counts.get('completed', 0)}、"
        f"running={counts.get('running', 0)}、qualified={counts.get('qualified', 0)}、"
        f"failed={counts.get('failed', 0)}、unsupported={counts.get('unsupported_taskview', 0)}。",
    ]
    if not fresh:
        lines += [
            "",
            "该 coverage 快照已超过 6 小时，以上计数只作为过期文件证据，不能解释为当前服务器实时进度。",
        ]
    if active:
        lines += ["", "文件中标为 running 的单元："]
        lines.extend(f"- `{item.get('profile')} / {item.get('dataset')} / seed{item.get('seed')}`"
                     for item in active[:16])
    processes = value.get("matching_processes") or []
    lines += ["", f"远端匹配到的 benchmark 进程行数：{len(processes)}。"]
    if processes:
        lines.extend(f"- `{line[:240]}`" for line in processes[:8])
    lines += [
        "",
        "本 reporter 只读 coverage/run-states 并发布 Issue；不启动、停止、重试、下载或修改训练。",
    ]
    return "\n".join(lines), marker


def already_posted(marker: str) -> bool:
    result = run(["gh", "issue", "view", ISSUE, "--repo", REPOSITORY,
                  "--json", "comments", "--jq", ".comments[].body"], timeout=45)
    if result.returncode:
        raise RuntimeError("cannot verify existing Issue marker")
    return marker in result.stdout


def post(body: str, marker: str) -> str:
    if already_posted(marker):
        return "duplicate_prevented"
    result = run(["gh", "issue", "comment", ISSUE, "--repo", REPOSITORY,
                  "--body", body], timeout=60)
    if result.returncode:
        raise RuntimeError("GitHub Issue comment failed")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            value = fetch_remote()
            write_receipt(value)
            body, marker = render(value)
            if args.dry_run:
                print(body)
            else:
                print(json.dumps({"marker": marker, "result": post(body, marker)}, ensure_ascii=False))
            return 0
        except Exception as exc:
            receipt = {"observed_utc": utc_now().isoformat(),
                       "status": "report_failed", "error_type": type(exc).__name__}
            write_receipt(receipt)
            print(json.dumps(receipt, ensure_ascii=False), file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
