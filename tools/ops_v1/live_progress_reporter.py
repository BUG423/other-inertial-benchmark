#!/usr/bin/env python3
"""Post a bounded, read-only formal-matrix status report to GitHub Issue #1.

The reporter reads the authoritative matrix on the live Agent through the
app-remote-terminal WSS relay. It never launches or signals benchmark work,
and it never uses the old SSH snapshot as a fallback.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "BUG423/other-inertial-benchmark"
ISSUE = "1"
PAIRING_PROFILE = Path(os.environ.get(
    "REMOTE_TERMINAL_APP_PROFILE",
    str(ROOT.parent / "app-remote-terminal/pairing/20261023/remote-pc-1.app.json"),
))
RELAY_HELPER = ROOT.parent / "app-remote-terminal/scripts/relay-exec.js"
STATE_DIR = Path.home() / ".local/state/other-inertial-benchmark-issue-reporter"
LOCK_PATH = STATE_DIR / "reporter.lock"
MENTIONS = ("@BUG423", "@RicardoCDUT", "@Sarah-123886")
MAX_METADATA_BYTES = 32 * 1024 * 1024


REMOTE_SCRIPT = dedent(
    r"""
    import datetime, hashlib, json, os, socket, subprocess
    from pathlib import Path

    matrix = Path('/home/ps/RAL/cloud-import-20261023/runtime/primary-local/matrix/main')
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

    state_mtimes = [
        path.stat().st_mtime for path in states.glob('*.json')
        if path.is_file()
    ]
    controller_text = subprocess.run(
        ['bash', '-lc', "ps -eo pid=,args= | egrep '(agent/index\\.js|continue-formal\\.py|formal-bridge\\.py)' | egrep -v 'egrep|grep' | head -40"],
        text=True, capture_output=True, check=False,
    ).stdout.splitlines()
    now = datetime.datetime.now(datetime.timezone.utc)
    mtime = datetime.datetime.fromtimestamp(coverage_path.stat().st_mtime, datetime.timezone.utc)
    research_path = Path('/home/ps/RAL/research-20261024/research-status.json')
    research = None
    if research_path.is_file() and research_path.stat().st_size <= 1024 * 1024:
        research = json.loads(research_path.read_bytes())
    print(json.dumps({
        'observed_utc': now.isoformat(),
        'hostname': socket.gethostname(),
        'coverage_mtime_utc': mtime.isoformat(),
        'coverage_sha256': hashlib.sha256(coverage_bytes).hexdigest(),
        'full_scope': len(rows),
        'counts': counts,
        'active': active,
        'run_state_files': len(state_mtimes),
        'latest_run_state_mtime_utc': (
            datetime.datetime.fromtimestamp(max(state_mtimes), datetime.timezone.utc).isoformat()
            if state_mtimes else None
        ),
        'matching_processes': controller_text,
        'controller_present': any('continue-formal.py' in line for line in controller_text),
        'research': research,
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
    if not PAIRING_PROFILE.is_file() or PAIRING_PROFILE.stat().st_mode & 0o077:
        raise RuntimeError("private relay APP pairing profile is missing or not mode 0600")
    if not RELAY_HELPER.is_file():
        raise RuntimeError("relay browser helper is missing")
    command = [
        "node", str(RELAY_HELPER), "--profile", str(PAIRING_PROFILE),
        "--timeout", "90", "--command",
        "/home/ps/anaconda3/bin/python3 -c " + shlex.quote(REMOTE_SCRIPT),
    ]
    result = run(command, timeout=105)
    if result.returncode:
        raise RuntimeError(f"remote read failed with exit {result.returncode}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("remote read returned no metadata")
    relay_result = json.loads(lines[-1])
    if relay_result.get("exitCode") != 0 or relay_result.get('cleanupConfirmed') is not True:
        raise RuntimeError("relay command failed")
    output_lines = [line.strip() for line in relay_result.get("output", "").splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("relay command returned no metadata")
    value = json.loads(output_lines[-1])
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
    latest_state = value.get("latest_run_state_mtime_utc")
    state_age_hours = None
    if latest_state:
        state_age_hours = max(0.0, (now - datetime.fromisoformat(latest_state)).total_seconds() / 3600.0)
    live = value.get("controller_present") is True and state_age_hours is not None and state_age_hours <= 6.0
    status = "live_agent" if live else "agent_state_needs_review"
    marker = f"<!-- ib-live-matrix-reporter/{now:%Y%m%dT%H%M%SZ} -->"
    counts = value.get("counts", {})
    active = value.get("active", [])
    lines = [
        marker,
        f"### 正式矩阵只读状态 · {now.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        *MENTIONS,
        "",
        f"状态源：{status}；通过阿里云 WSS Agent 读取；主机 `{value.get('hostname')}`；coverage SHA256 `{value.get('coverage_sha256')}`。",
        f"run-states 文件数：{value.get('run_state_files', 0)}；最新修改：{latest_state or 'unknown'}"
        + (f"（约 {state_age_hours:.1f} 小时前）。" if state_age_hours is not None else "。"),
        f"coverage 文件最后修改：{value.get('coverage_mtime_utc')}（冻结范围文件，不作为实时进度依据）。",
        f"正式范围：{value.get('full_scope')}；文件中记录的 completed={counts.get('completed', 0)}、"
        f"running={counts.get('running', 0)}、qualified={counts.get('qualified', 0)}、"
        f"failed={counts.get('failed', 0)}、unsupported={counts.get('unsupported_taskview', 0)}。",
    ]
    if not live:
        lines += [
            "",
            "未检测到近期 run-state 更新或正式控制器；本次计数需要人工复核，不能解释为实时进度。",
        ]
    if active:
        lines += ["", "文件中标为 running 的单元："]
        lines.extend(f"- `{item.get('profile')} / {item.get('dataset')} / seed{item.get('seed')}`"
                     for item in active[:16])
    processes = value.get("matching_processes") or []
    lines += ["", f"远端匹配到的 benchmark 进程行数：{len(processes)}。"]
    if processes:
        lines.extend(f"- `{line[:240]}`" for line in processes[:8])
    research = value.get('research') or {}
    length = research.get('length') or {}
    if length:
        lines += ['', f"轨迹研究快照：{length.get('observed_utc')}；当时 completed={length.get('completed')}。",
                  f"人体 XYZ 神经可训练配置主 test：{length.get('human_cells')} 个单元；"
                  f"PLR<1 比例 {length.get('human_fraction_below_one'):.2%}。"]
        canonical = (length.get('canonical_cluster') or {}).get('metrics') or {}
        for metric in ('plr', 'plr_oracle', 'speed_ratio'):
            entry = canonical.get(metric)
            if entry:
                low, high = entry['conditional_subject_bootstrap_95']
                lines.append(f"- T200/reference/平均速度、三 seed 齐全组的 `{metric}`："
                             f"{entry['estimate']:.4f}，条件性受试者分组 bootstrap 95% CI [{low:.4f}, {high:.4f}]。")
        lines += ["以上为已观察算法/来源的探索性诊断；不代表算法总体区间、因果证明或完整结果验收。"]
    candidate = research.get('candidate') or {}
    if candidate:
        lines += ['', f"新网络候选：{candidate.get('status')}；参数 {candidate.get('parameters')}；"
                  f"CPU 科学检查 {candidate.get('tests_passed')} 项；原生生命周期 {candidate.get('native_lifecycle')}。",
                  "完整数据三 seed 精度与 CUDA 容量待验证；当前 CPU 工程结果不能作为精度提升证据。"]
    lines += [
        "",
        "本 reporter 只读 Agent 上的 coverage/run-states 和进程列表并发布 Issue；不启动、停止、重试、下载或修改训练。",
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', encoding='utf-8') as draft:
        draft.write(body)
        draft.flush()
        result = run(["gh", "issue", "comment", ISSUE, "--repo", REPOSITORY,
                      "--body-file", draft.name], timeout=60)
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
