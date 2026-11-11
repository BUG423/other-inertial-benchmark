#!/usr/bin/env python3
"""Post a bounded, read-only formal-matrix status report to a GitHub Issue.

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
ISSUE = os.environ.get("IPB_REPORT_ISSUE", "1")
PAIRING_PROFILE = Path(os.environ.get(
    "REMOTE_TERMINAL_APP_PROFILE",
    str(ROOT.parent / "app-remote-terminal/pairing/20261023/remote-pc-1.app.json"),
))
RELAY_HELPER = ROOT.parent / "app-remote-terminal/scripts/relay-exec.js"
STATE_DIR = Path.home() / ".local/state/other-inertial-benchmark-issue-reporter"
LOCK_PATH = STATE_DIR / "reporter.lock"
MENTIONS = ("@BUG423", "@RicardoCDUT", "@Sarah-123886")
MAX_METADATA_BYTES = 32 * 1024 * 1024

# These values describe the private Agent runtime and must be supplied by the
# host that owns that runtime.  Keeping them out of the public source prevents
# the repository from disclosing usernames, directory layouts and interpreter
# installations.  The matrix/research roots are intentionally separate so a
# reporter cannot silently read an unrelated checkout.
REMOTE_AGENT_MATRIX_ROOT = os.environ.get("REMOTE_AGENT_MATRIX_ROOT", "").strip()
REMOTE_AGENT_RESEARCH_ROOT = os.environ.get("REMOTE_AGENT_RESEARCH_ROOT", "").strip()
REMOTE_AGENT_PYTHON = os.environ.get("REMOTE_AGENT_PYTHON", "python3").strip()


REMOTE_SCRIPT = dedent(
    r"""
    import datetime, hashlib, json, subprocess
    from pathlib import Path

    matrix = Path('__REMOTE_AGENT_MATRIX_ROOT__')
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
    research_path = Path('__REMOTE_AGENT_RESEARCH_STATUS_ROOT__')
    research = None
    if research_path.is_file() and research_path.stat().st_size <= 1024 * 1024:
        research = json.loads(research_path.read_bytes())
    cause_path = Path('__REMOTE_AGENT_CAUSE_STATUS_ROOT__')
    cause = None
    if cause_path.is_file() and cause_path.stat().st_size <= 1024 * 1024:
        cause = json.loads(cause_path.read_bytes())
    print(json.dumps({
        'observed_utc': now.isoformat(),
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
        'matching_process_count': len(controller_text),
        'controller_present': any('continue-formal.py' in line for line in controller_text),
        'research': research,
        'cause': cause,
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
    if not REMOTE_AGENT_MATRIX_ROOT:
        raise RuntimeError("REMOTE_AGENT_MATRIX_ROOT must point to the private Agent matrix")
    if not REMOTE_AGENT_RESEARCH_ROOT:
        raise RuntimeError("REMOTE_AGENT_RESEARCH_ROOT must point to the private Agent research root")
    if not REMOTE_AGENT_PYTHON:
        raise RuntimeError("REMOTE_AGENT_PYTHON must name the private Agent Python executable")
    remote_script = REMOTE_SCRIPT.replace(
        "Path('__REMOTE_AGENT_MATRIX_ROOT__')",
        f"Path({REMOTE_AGENT_MATRIX_ROOT!r})",
    ).replace(
        "Path('__REMOTE_AGENT_RESEARCH_STATUS_ROOT__')",
        f"Path({REMOTE_AGENT_RESEARCH_ROOT!r}) / 'research-status.json'",
    ).replace(
        "Path('__REMOTE_AGENT_CAUSE_STATUS_ROOT__')",
        f"Path({REMOTE_AGENT_RESEARCH_ROOT!r}) / 'cause-status.json'",
    )
    command = [
        "node", str(RELAY_HELPER), "--profile", str(PAIRING_PROFILE),
        "--timeout", "90", "--command",
        shlex.quote(REMOTE_AGENT_PYTHON) + " -c " + shlex.quote(remote_script),
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
        f"状态源：{status}；通过阿里云 WSS Agent 读取。",
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
    lines += ["", f"远端匹配到的 benchmark 进程数：{value.get('matching_process_count', 0)}。"]
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
    cause = value.get('cause') or {}
    canonical = cause.get('canonical') or {}
    if canonical:
        direction = canonical.get('direction_strata') or {}
        low_direction = direction.get('low_dir_err_lt20') or {}
        high_direction = direction.get('high_dir_err_ge45') or {}
        lines += ['', f"缩短原因分解：{cause.get('updated_utc')}。",
                   f"T200/reference/avg_velocity：oracle_ratio 中位数 {canonical.get('protocol_oracle_ratio', {}).get('median', float('nan')):.4f}；"
                   f"plr_oracle 中位数 {canonical.get('model_plr_oracle', {}).get('median', float('nan')):.4f}；"
                   f"speed_ratio 中位数 {canonical.get('speed_ratio', {}).get('median', float('nan')):.4f}。",
                   f"方向误差<20°组 speed_ratio 中位数 {(low_direction.get('speed_ratio') or {}).get('median', float('nan')):.4f}；"
                   f"方向误差≥45°组 {(high_direction.get('speed_ratio') or {}).get('median', float('nan')):.4f}；"
                   f"train/test speed_ratio 差中位数 {canonical.get('train_test_speed_gap', {}).get('median', float('nan')):.6f}。",
                   "该分解描述模型幅值低估与方向误差的关联；目标/损失机制仍待受控干预验证。",
                   "train/test 差异同时混合数据和模式效应，不能单独用来排除 Dropout。"]
        sensitivity = canonical.get('direction_sensitivity_descriptive') or {}
        if sensitivity:
            def coefficient(entry):
                value = entry.get('correlation')
                return f"{value:.4f}" if isinstance(value, (int, float)) else "unknown"

            pooled = sensitivity.get('all') or {}
            trimmed = sensitivity.get('direction_lt45') or {}
            within = sensitivity.get('within_profile_dataset') or {}
            lines += [
                f"速度/方向相关敏感性：全部 {pooled.get('n')} 单元 r={coefficient(pooled)}；"
                f"方向误差<45° 的 {trimmed.get('n')} 单元 r={coefficient(trimmed)}；"
                f"{within.get('groups')} 个 profile/dataset 组内去均值 r={coefficient(within)}。",
                "排除尾部会改变样本和范围；组内去均值也不能消除所有混杂，以上均为描述性结果。",
            ]
    capacity = cause.get('capacity_probe') or {}
    if (capacity.get('complete') is True and capacity.get('training_only') is True
            and capacity.get('dataset') == 'ridi' and capacity.get('recordings') == 1
            and capacity.get('windows') == 16 and capacity.get('steps_per_arm') == 192
            and len(capacity.get('seeds') or []) == 3):
        medians = capacity.get('median_over_three_seeds') or {}

        def capacity_ratio(arm, mode):
            entry = (medians.get(arm) or {}).get(mode) or {}
            ratio = entry.get('speed_ratio')
            return f"{ratio:.4f}" if isinstance(ratio, (int, float)) else "unknown"

        lines += [
            '', 'CPU 固定训练窗对照（RIDI 单录制、16 窗、三 seed、每组 192 步）：',
            f"关闭 Dropout：推理 BN 的速度幅值比中位数 {capacity_ratio('0.0', 'running_bn')}；"
            f"同窗 batch BN 为 {capacity_ratio('0.0', 'batch_bn')}。"
            f"默认 Dropout：推理 BN 为 {capacity_ratio('0.5', 'running_bn')}。",
            '相同输入/权重的 BN 模式对照与有限预算 Dropout 对照仅用于机制排查；'
            '无测试集拟合，不代表轨迹 PLR、全矩阵因果解释或定位精度提升。',
        ]
    checkpoint_bn = cause.get('checkpoint_bn_probe') or {}
    if (checkpoint_bn.get('complete') is True and checkpoint_bn.get('unique_windows') == 16
            and checkpoint_bn.get('profile') == 'ronin_resnet18_3d'
            and checkpoint_bn.get('dataset') == 'ridi'):
        paired = (checkpoint_bn.get('paired_summary') or {}).get('128') or {}
        delta = paired.get('median_variance_convention_speed_ratio_delta')
        worse = paired.get('biased_calibration_relative_l2_worse_than_original')
        if isinstance(delta, (int, float)) and type(worse) is int:
            lines += [
                f"已完成 checkpoint 的只读 BN 复核：同一组 16 窗重复到 128 条，"
                f"方差约定的配对幅值比差中位数 {delta:.5f}；"
                f"校准后 {worse}/3 seed 的向量相对 L2 误差更大。",
                '重复 batch 没有增加观测，也不重现对应 batch 的训练；本次不接受 BN 校准为修复方案。',
            ]
    velocity_state = cause.get('velocity_state_probe') or {}
    if (velocity_state.get('complete') is True and velocity_state.get('dataset') == 'ridi'
            and velocity_state.get('split') == 'train' and velocity_state.get('recordings') == 45
            and velocity_state.get('total_windows') == 720 and velocity_state.get('optimizer_updates') == 0):
        summary = velocity_state.get('recording_summary') or {}

        def state_median(method, metric='relative_l2'):
            entry = (summary.get(method) or {}).get(metric) or {}
            number = entry.get('median')
            return f"{number:.4f}" if type(number) in (int, float) else 'unknown'

        lines += [
            '', f"初速度边界诊断：{velocity_state.get('observed_utc')}；45 条训练录制、720 窗，零优化更新。",
            f"录制级向量相对 L2 中位数：零初速 {state_median('zero_state')}；"
            f"参考初速+积分 {state_median('oracle_state')}；打乱初速 {state_median('shuffled_state')}；"
            f"只保留参考初速 {state_median('oracle_persistence')}。",
            f"打乱初速的幅值比仍为 {state_median('shuffled_state', 'speed_ratio')}，幅值接近 1 不等于运动预测正确。"
            f"三个原 checkpoint 的误差分别为 {state_median('checkpoint_1735231569')}、"
            f"{state_median('checkpoint_843628571')}、{state_median('checkpoint_951150512')}。",
            '恒定速度扰动是构造的可辨识性检查；参考初速是特权信息。'
            '这不证明真实数据的歧义比例、全矩阵缩短主因或 ATE/RTE 改善。',
        ]
    radial = cause.get('radial_loss_probe') or {}
    radial_splits = radial.get('splits') or {}
    if (radial.get('complete') is True and radial.get('dataset') == 'ridi'
            and radial.get('gain_fit_split') == 'train' and radial.get('test_accessed') is False
            and radial.get('optimizer_updates') == 0 and len(radial.get('seeds') or []) == 3
            and radial_splits.get('train') == dict(recordings=45, windows=720, groups=8)
            and radial_splits.get('val') == dict(recordings=4, windows=64, groups=1)):
        counts = radial.get('summary') or {}
        improved = counts.get('validation_seeds_vector_mse_gain_improves')
        worse = counts.get('validation_seeds_norm_sum_gain_worsens')
        if type(improved) is int and type(worse) is int:
            lines += [
                '', '径向损失诊断：45 条训练录制拟合标量，4 条验证录制检查；验证仅一个 group。',
                f"按向量 MSE 拟合的增益使 {improved}/3 seed 的验证窗口误差降低；"
                f"按训练模长和校准的增益使 {worse}/3 seed 的误差变大。",
                '低方向误差窗仍有负的预测方向投影残差；统一放大不能解释或解决全部缩短。'
                '增益差异还受窗口权重影响，不能单独归因于方向歧义；该窗口诊断无测试集或 ATE/RTE 结论。',
            ]
    dynamic = cause.get('dynamic_lag_probe') or {}
    dynamic_splits = dynamic.get('split_shapes') or {}
    dynamic_seeds = dynamic.get('seeds') or []
    dynamic_scope = (
        dynamic.get('complete') is True and dynamic.get('dataset') == 'ridi'
        and dynamic.get('profile') == 'ronin_resnet18_3d'
        and dynamic.get('optimizer_updates') == 0 and dynamic.get('test_accessed') is False
        and dynamic.get('input_files_unchanged') is True
        and dynamic.get('weights_and_buffers_unchanged') is True
        and dynamic.get('cuda_initialized') is False
        and dynamic.get('group_overlap') == 0
        and dynamic_splits.get('train') == dict(recordings=45, windows=720, groups=8)
        and dynamic_splits.get('val') == dict(recordings=4, windows=64, groups=1)
        and len(dynamic.get('train_quantile_edges_mps') or []) == 3
        and all(type(edge) in (int, float) for edge in dynamic.get('train_quantile_edges_mps') or [])
        and len(dynamic_seeds) == 3
    )
    if dynamic_scope:
        def dynamic_row(row):
            return (isinstance(row, dict)
                    and all(_dynamic_split_ready(row.get(split)) for split in ('train', 'val')))

        def _dynamic_split_ready(split):
            return (isinstance(split, dict) and type(split.get('speed_ratio')) in (int, float)
                    and type(split.get('delta_speed_correlation')) in (int, float)
                    and type(split.get('roughness_speed_correlation')) in (int, float)
                    and isinstance(split.get('bins'), list) and len(split['bins']) == 4
                    and all(type(value) in (int, float) for value in split['bins']))

        dynamic_scope = all(dynamic_row(row) for row in dynamic_seeds)
    if dynamic_scope:
        edges = dynamic['train_quantile_edges_mps']
        lines += [
            '', '动态变化分层诊断：RIDI 训练 45 条录制/720 窗，验证 4 条录制/64 窗；'
            f"训练端点速度变化四分位边界 {', '.join(f'{edge:.5f}' for edge in edges)} m/s。",
            '三个 seed 的训练/验证整体速度比：' + '; '.join(
                f"seed {index} {row['train']['speed_ratio']:.4f}/{row['val']['speed_ratio']:.4f}"
                for index, row in enumerate(dynamic_seeds, 1)) + '。',
            '验证四个变化区间速度比：' + '; '.join(
                f"seed {index} [{', '.join(f'{value:.4f}' for value in row['val']['bins'])}]"
                for index, row in enumerate(dynamic_seeds, 1)) + '。',
            '验证端点变化/速度误差相关：' + ', '.join(
                f"{row['val']['delta_speed_correlation']:.3f}" for row in dynamic_seeds)
            + '；IMU 粗糙度相关：' + ', '.join(
                f"{row['val']['roughness_speed_correlation']:.3f}" for row in dynamic_seeds) + '。',
            '动态分层在 seed 与 train/val 间没有稳定单调关系；当前不能称为动态滞后已被证实。'
            '分层阈值只用于诊断，不是部署特征；该分层诊断没有测试集、ATE/RTE 或因果干预结论。',
        ]
    trajectory = cause.get('trajectory_gain_probe') or {}
    gain_counts = trajectory.get('paired_summary') or {}
    count_bounds = dict(ate_seed_medians_worse=3, rte_10s_seed_medians_worse=3,
                        position_p95_seed_medians_worse=3, ate_recording_seed_pairs_worse=12)
    trajectory_scope = (
        trajectory.get('schema') == 'ib.trajectory-gain-diagnostic.v1'
        and trajectory.get('complete') is True and trajectory.get('dataset') == 'ridi'
        and trajectory.get('profile') == 'ronin_resnet18_3d' and trajectory.get('split') == 'val'
        and trajectory.get('recordings') == 4 and trajectory.get('groups') == 1
        and trajectory.get('gain_fit_split') == 'train' and trajectory.get('gains_refitted') is False
        and trajectory.get('eval_stride') == 200 and trajectory.get('optimizer_updates') == 0
        and trajectory.get('test_accessed') is False and trajectory.get('cuda_initialized') is False
        and all(trajectory.get(k) is True for k in ('input_files_unchanged',
                'weights_and_buffers_unchanged', 'framework_files_unchanged', 'inputs_read_only'))
        and len(trajectory.get('seeds') or []) == 3
        and all(isinstance(gain_counts.get(arm), dict) and all(
            type(gain_counts[arm].get(k)) is int and 0 <= gain_counts[arm][k] <= bound
            for k, bound in count_bounds.items()) for arm in ('vector_mse', 'norm_sum'))
    )
    if trajectory_scope:
        lines += ['', '完整验证轨迹增益控制：RIDI 4 条录制/1 个 group；训练增益已冻结，验证不重拟合。']
        for arm, label in (('vector_mse', '向量 MSE 增益'), ('norm_sum', '模长和增益')):
            counts = gain_counts[arm]
            lines += [f"{label}：{counts['ate_seed_medians_worse']}/3 seed 的未对齐 ATE 中位数恶化；"
                      f"{counts['rte_10s_seed_medians_worse']}/3 的 10s RTE 中位数恶化；"
                      f"{counts['position_p95_seed_medians_worse']}/3 的位置 P95 中位数恶化；"
                      f"{counts['ate_recording_seed_pairs_worse']}/12 录制/seed 配对的 ATE 恶化。"]
        lines += ['窗口或短时误差改善不能代替完整轨迹与尾部验收；刚体对齐指标另列，不拟合尺度。'
                  '验证已用于 checkpoint 选择，12 个配对不是独立受试者；无测试精度或因果结论。']
    head = cause.get('head_intervention_probe') or {}
    head_summary = head.get('paired_summary') or {}
    required_counts = {
        'vector_mse_vs_identity': ('ate_seed_medians_improved', 'rte_10s_seed_medians_improved',
                                   'position_p95_seed_medians_improved'),
        'magnitude_aux_vs_vector_mse': ('ate_seed_medians_worse',),
        'least_squares_vs_identity': ('ate_seed_medians_worse', 'rte_10s_seed_medians_worse'),
    }
    head_scope = (
        head.get('schema') == 'ib.frozen-head-intervention.v1' and head.get('complete') is True
        and head.get('dataset') == 'ridi' and head.get('profile') == 'ronin_resnet18_3d'
        and head.get('fit_split') == 'train' and head.get('test_accessed') is False
        and head.get('head_only') is True and head.get('steps_per_optimization_arm') == 256
        and head.get('train_windows') == 720 and head.get('train_recordings') == 45
        and head.get('train_groups') == 8 and head.get('validation_recordings') == 4
        and head.get('validation_groups') == 1 and head.get('group_overlap') == 0
        and head.get('cuda_initialized') is False and head.get('saves_weights') is False
        and head.get('formal_matrix_changed') is False and len(head.get('seeds') or []) == 3
        and all(head.get(k) is True for k in ('final_step_preselected', 'original_model_restored',
                'frozen_features_and_buffers_unchanged', 'source_checkpoint_and_data_files_unchanged',
                'framework_and_probe_unchanged', 'framework_matches_current_checkout'))
        and all(isinstance(head_summary.get(arm), dict) and all(
            type(head_summary[arm].get(k)) is int and 0 <= head_summary[arm][k] <= 3
            for k in keys) for arm, keys in required_counts.items())
    )
    pair_count = (head_summary.get('vector_mse_vs_identity') or {}).get('ate_recording_seed_pairs_worse')
    if head_scope and type(pair_count) is int and 0 <= pair_count <= 12:
        mse = head_summary['vector_mse_vs_identity']
        auxiliary = head_summary['magnitude_aux_vs_vector_mse']
        ls = head_summary['least_squares_vs_identity']
        lines += ['', '冻结回归层干预：RIDI 720 训练窗、4 条完整验证录制/1 个 group；两组相同初始回归层、256 步。',
                  f"保持原向量 MSE：{mse['ate_seed_medians_improved']}/3 seed 的 ATE、"
                  f"{mse['rte_10s_seed_medians_improved']}/3 的 10s RTE、"
                  f"{mse['position_p95_seed_medians_improved']}/3 的位置 P95 中位数改善；"
                  f"仍有 {pair_count}/12 录制/seed 配对的 ATE 恶化。",
                  f"幅值辅助项相对同预算原 MSE 使 {auxiliary['ate_seed_medians_worse']}/3 seed 的 ATE 中位数恶化；"
                  f"截断最小二乘相对原 checkpoint 有 {ls['ate_seed_medians_worse']}/3 的 ATE、"
                  f"{ls['rte_10s_seed_medians_worse']}/3 的 10s RTE 中位数恶化。",
                  '存在可通过回归层重新拟合改善的偏差；本轮同时改变了原训练的采样权重、Dropout 和优化约束，'
                  '尚不能单独归因于优化未收敛或正则化。验证已用于 checkpoint 选择，无独立测试精度或通用修复结论。']
    dropout = cause.get('expected_dropout_probe') or {}
    dropout_summary = dropout.get('paired_summary') or {}
    reductions = dropout_summary.get('train_speed_ratio_reduction') or []
    plr_reductions = dropout_summary.get('validation_plr_reduction') or []
    count_keys = ('dropout_half_vs_none_ate_seed_medians_improved',
                  'dropout_half_vs_none_ate_seed_medians_worse',
                  'dropout_half_vs_none_rte_seed_medians_improved')
    dropout_scope = (
        dropout.get('schema') == 'ib.expected-head-dropout.v1' and dropout.get('complete') is True
        and dropout.get('dataset') == 'ridi' and dropout.get('profile') == 'ronin_resnet18_3d'
        and dropout.get('fit_split') == 'train' and dropout.get('head_only') is True
        and dropout.get('test_accessed') is False and dropout.get('steps_per_optimization_arm') == 0
        and dropout.get('train_windows') == 720 and dropout.get('train_recordings') == 45
        and dropout.get('train_groups') == 8 and dropout.get('validation_recordings') == 4
        and dropout.get('validation_groups') == 1 and dropout.get('group_overlap') == 0
        and dropout.get('cuda_initialized') is False and dropout.get('saves_weights') is False
        and dropout.get('formal_matrix_changed') is False and len(dropout.get('seeds') or []) == 3
        and all(dropout.get(k) is True for k in ('original_model_restored',
                'frozen_features_and_buffers_unchanged', 'source_checkpoint_and_data_files_unchanged',
                'framework_and_probe_unchanged', 'framework_matches_current_checkout'))
        and all(len(values) == 3 and all(type(v) in (int, float) and 0 <= v <= 1 for v in values)
                for values in (reductions, plr_reductions))
        and all(type(dropout_summary.get(k)) is int and 0 <= dropout_summary[k] <= 3 for k in count_keys)
    )
    if dropout_scope:
        lines += ['', '最后特征层 Dropout 期望损失对照：固定特征、采样权重及求解器，仅比较 p=0 与 p=0.5。',
                  'p=0.5 的训练速度幅值比分别降低 ' + '、'.join(f'{100*v:.2f}' for v in reductions)
                  + ' 个百分点；验证 PLR 中位数分别降低 '
                  + '、'.join(f'{100*v:.2f}' for v in plr_reductions) + ' 个百分点。',
                  f"保留期望正则项时 {dropout_summary[count_keys[0]]}/3 seed 的验证 ATE 中位数改善、"
                  f"{dropout_summary[count_keys[1]]}/3 恶化；"
                  f"{dropout_summary[count_keys[2]]}/3 的 10s RTE 中位数改善。",
                  '局部回归层存在正则收缩效应，但关闭 Dropout 不是通用修复；'
                  '此前层的 Dropout 与特征自适应未干预，不能外推为完整网络或全矩阵根因。'
                  '验证仅一个 group，未访问独立测试集。']
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
