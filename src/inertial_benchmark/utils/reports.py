"""结果汇总（定义见 ``docs/METRICS.md`` 第 5 节）：跨序列统计、分组 bootstrap、多种子聚合、
配对 Wilcoxon，导出 CSV / Markdown / LaTeX 表格与汇总图。依赖 pandas。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import stat
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from ..cfg import EVAL_PROTOCOL_KEYS, ConfigError
from ..metrics import MAIN_METRICS, display_name
from . import LOGGER, json_load, json_save

PathLike = Union[str, Path]
KEYS = ("dataset", "split", "model")
DIMENSION_FIELDS = ("metric_dims_requested", "metric_dims_effective", "predicted_axes")
MATRIX_ID_FIELDS = ("profile", "dataset", "seed", "input_track")
MATRIX_SCIENTIFIC_FIELDS = (
    "profile", "dataset", "seed", "input_track", "dataset_fingerprint",
    "dataset_manifest_sha256", "split_sha256", "profile_config_sha256", "sampling_spec",
    "scientific_input_spec", "dims", "main_xyz", "native_xyz", "data_root", "domain",
    "checkpoint_training_source", "taskview_receipt", "classic", "trainable",
)
MATRIX_STATE_MUTABLE_FIELDS = frozenset((
    "id", "state", "error", "dependency_reason", "resource_reason",
    "model_cpu_pending_reason", "model_cpu_accepted", "model_cpu_receipt_sha256",
    "checkpoint", "checkpoint_sha256", "checkpoint_retention", "checkpoint_source",
    "cross_domain_transfer", "result", "result_artifacts", "run_identity_path",
    "run_identity_sha256",
))
MATRIX_SPLIT_STATUSES = frozenset(("passed", "not_applicable_missing_split"))
MATRIX_UNSUPPORTED_STATES = frozenset((
    "unsupported_taskview", "unsupported_missing_validation",
))
MATRIX_AGGREGATE_FIELDS = ("metrics", "median", "std", "count")
MATRIX_CORE_METRICS = ("ate", "rte")
MATRIX_SUMMARY_KEYS = (
    "dataset", "split", "model", "input_track", "matrix_dims", "main_xyz", "domain",
    "checkpoint_training_source", "privileged",
)


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reports need pandas: pip install 'other-inertial-benchmark"
                          "[train]'") from exc
    return pd


def find_runs(root: PathLike) -> list:
    """``root`` 下所有同时含 ``metrics.json`` 与 ``sequences.csv`` 的评测目录（排除训练目录）。"""
    runs = []
    for path in sorted(Path(root).rglob("metrics.json")):
        if (path.parent / "sequences.csv").exists():
            meta = json_load(path)
            if meta.get("mode", "val") == "val":
                runs.append(path.parent)
    return runs


class ProtocolMismatch(ConfigError):
    """同一份报表中的 run 使用了不同的评测协议，指标不可比（CLI 以退出码 2 报错）。"""


class MatrixEvaluationInvalid(ProtocolMismatch):
    """A completed result is unusable, but the matrix schema itself remains readable."""


def _check_dimension_evidence(run: PathLike, meta: dict) -> None:
    requested = (meta.get("protocol") or {}).get("metric_dims")
    evidence = [meta.get(key) for key in DIMENSION_FIELDS]
    if all(value is None for value in evidence):
        if requested == 3:
            raise ProtocolMismatch(f"{run}: metric_dims=3 has no actual dimension evidence; "
                                   "re-evaluate saved XYZ predictions before reporting")
        return  # Historical 2D results never used the 3D-to-2D fallback.
    req, effective, axes = evidence
    if (any(value is None for value in evidence) or req != requested or effective != requested
            or axes not in (["x", "y"], ["x", "y", "z"]) or len(axes) < requested):
        raise ProtocolMismatch(f"{run}: actual evaluation dimensions disagree with metric_dims "
                               f"or predicted_axes; re-evaluate with metric_dims={requested}")
    # JSON evidence alone is insufficient: every evaluated CSV row must agree with it.
    with open(Path(run) / "sequences.csv", newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if not set(DIMENSION_FIELDS) <= set(rows.fieldnames or ()):
            raise ProtocolMismatch(f"{run}: sequences.csv lacks actual dimension evidence; "
                                   "re-evaluate all sequences before reporting")
        count = 0
        for row in rows:
            if row.get("skipped"):
                continue
            try:
                dims = [float(row[key]) for key in DIMENSION_FIELDS[:2]]
            except (ValueError, TypeError) as exc:
                raise ProtocolMismatch(f"{run}: missing sequence dimension evidence for "
                                       f"{row.get('sequence_id')}") from exc
            if dims != [requested, requested] or row["predicted_axes"] != ",".join(axes):
                raise ProtocolMismatch(f"{run}: sequence {row.get('sequence_id')} has "
                                       "inconsistent evaluation dimensions or predicted_axes; "
                                       "re-evaluate before reporting")
            count += 1
    if meta.get("num_sequences") is not None and count != meta["num_sequences"]:
        raise ProtocolMismatch(f"{run}: dimension evidence sequence count differs from JSON")


def check_protocols(runs: Iterable[PathLike]) -> dict:
    """核对各 run 的评测协议是否一致，不一致时抛出 :class:`ProtocolMismatch`。

    只比较**评测协议**键（``EVAL_PROTOCOL_KEYS``：步长、指标维度、RTE/D-RTE 参数、速度阈值）：
    这些键决定指标怎么算，必须全表一致；输入规格键（窗口、坐标系、目标…）本来就随模型不同，
    只记录不比较。缺少 ``protocol`` 的旧结果视为未知，直接报错（宁可让人重跑也不混表）。
    """
    seen: dict = {}
    metadata = []
    for run in runs:
        meta = json_load(Path(run) / "metrics.json")
        protocol = meta.get("protocol")
        if not protocol:
            raise ProtocolMismatch(f"{run}/metrics.json has no 'protocol' block; re-run the "
                                   "evaluation with the current version before reporting")
        missing = [k for k in EVAL_PROTOCOL_KEYS if protocol.get(k) is None]
        if missing:
            raise ProtocolMismatch(f"{run}: evaluation protocol is incomplete, missing {missing}")
        metadata.append((run, meta))
        for key in EVAL_PROTOCOL_KEYS:
            value = protocol[key]
            value = tuple(value) if isinstance(value, list) else value
            first_run, first = seen.setdefault(key, (run, value))
            if first != value:
                raise ProtocolMismatch(
                    f"evaluation protocol differs between runs: {key}={first!r} in {first_run} "
                    f"but {key}={value!r} in {run}. Re-evaluate both with the same protocol "
                    "(eval keys come from default.yaml / the benchmark config / the CLI).")
    for run, meta in metadata:
        _check_dimension_evidence(run, meta)
    return {key: value for key, (_, value) in seen.items()}


def load_runs(root: PathLike):
    """把评测结果展开为逐序列表：``dataset, split, model, seed, run, sequence_id, group_id, …``。"""
    pd = _pd()
    frames = []
    for run in find_runs(root):
        meta = json_load(run / "metrics.json")
        _check_dimension_evidence(run, meta)
        df = pd.read_csv(run / "sequences.csv")
        if "skipped" in df:
            df = df[df["skipped"].isna() | (df["skipped"] == "")]
        eff = meta.get("efficiency") or {}
        privileged = meta.get("privileged_inputs") or []
        df.insert(0, "run", str(run))
        df.insert(0, "privileged", ",".join(privileged))
        df.insert(0, "seed", meta.get("seed", 0))
        df.insert(0, "model", meta.get("model", "unknown"))
        df.insert(0, "split", meta.get("split", "unknown"))
        df.insert(0, "dataset", meta.get("dataset", "unknown"))
        for key in ("params", "flops"):
            df[key] = eff.get(key, np.nan)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(KEYS) + ["seed", "privileged", "run", "sequence_id",
                                                  "group_id"])
    df = pd.concat(frames, ignore_index=True)
    df["privileged"] = df["privileged"].fillna("").astype(str)
    df["group_id"] = df.get("group_id", df["sequence_id"]).astype(str)
    df["sequence_id"] = df["sequence_id"].astype(str)
    return df


def bootstrap_ci(values: Sequence[float], groups: Optional[Sequence[str]] = None,
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple:
    """按组有放回重采样的均值置信区间；组为空时按单个样本重采样。"""
    values = np.asarray(values, float)
    groups = np.asarray(groups if groups is not None else np.arange(len(values))).astype(str)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    if len(values) == 0:
        return (math.nan, math.nan)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) == 1:
        m = float(values.mean())
        return (m, m)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def summarize(df, metrics: Iterable[str] = MAIN_METRICS, n_boot: int = 2000,
              group_keys: Sequence[str] = KEYS):
    """按 ``group_keys`` 汇总：跨序列均值/中位数/标准差、分组 CI、种子间标准差。"""
    pd = _pd()
    metrics = [m for m in metrics if m in df.columns]
    group_keys = tuple(group_keys)
    rows = []
    for key, part in df.groupby(list(group_keys), sort=True):
        # 逐序列先在种子间平均，再跨序列统计
        seq = part.groupby(["sequence_id", "group_id"], as_index=False)[metrics].mean()
        per_seed = part.groupby("seed")[metrics].mean()
        row = dict(zip(group_keys, key))
        row.update(n_sequences=len(seq), n_seeds=int(part["seed"].nunique()),
                   privileged=(sorted(set(part["privileged"]) - {""})[0]
                               if "privileged" in part and set(part["privileged"]) - {""}
                               else ""),
                   params=part["params"].iloc[0] if "params" in part else np.nan,
                   flops=part["flops"].iloc[0] if "flops" in part else np.nan)
        for m in metrics:
            v = seq[m].to_numpy(float)
            finite = v[np.isfinite(v)]
            row[f"{m}_mean"] = float(np.nanmean(per_seed[m])) if len(finite) else np.nan
            row[f"{m}_median"] = float(np.median(finite)) if len(finite) else np.nan
            row[f"{m}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
            row[f"{m}_seed_std"] = float(np.nanstd(per_seed[m], ddof=1)) \
                if per_seed[m].notna().sum() > 1 else np.nan
            lo, hi = bootstrap_ci(v, seq["group_id"], n_boot=n_boot)
            row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = lo, hi
            row[f"{m}_n"] = int(len(finite))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_wilcoxon(df, metric: str, model_a: str, model_b: str, dataset: str,
                    split: str, min_pairs: int = 5) -> dict:
    """同一数据集/划分上两模型的逐序列配对 Wilcoxon 检验（值先在种子间平均）。"""
    from scipy.stats import wilcoxon

    part = df[(df["dataset"] == dataset) & (df["split"] == split)]
    a = part[part["model"] == model_a].groupby("sequence_id")[metric].mean()
    b = part[part["model"] == model_b].groupby("sequence_id")[metric].mean()
    both = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    diff = (both["a"] - both["b"]).to_numpy(float)
    out = {"dataset": dataset, "split": split, "metric": metric, "model_a": model_a,
           "model_b": model_b, "n": int(len(diff)),
           "median_diff": float(np.median(diff)) if len(diff) else math.nan,
           "statistic": math.nan, "p_value": math.nan}
    if len(diff) >= min_pairs:
        if np.all(diff == 0):
            out.update(statistic=0.0, p_value=1.0)
        else:
            res = wilcoxon(both["a"], both["b"], zero_method="wilcox", alternative="two-sided")
            out.update(statistic=float(res.statistic), p_value=float(res.pvalue))
    return out


def all_pairwise_tests(df, metrics: Iterable[str] = ("ate", "rte")):
    pd = _pd()
    rows = []
    for (dataset, split), part in df.groupby(["dataset", "split"]):
        models = sorted(part["model"].unique())
        for a, b in combinations(models, 2):
            for m in metrics:
                if m in df.columns:
                    rows.append(paired_wilcoxon(df, m, a, b, dataset, split))
    return pd.DataFrame(rows)


def _fmt(value: float, digits: int = 3) -> str:
    return "–" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _cell(row, m: str, digits: int, pm: str) -> str:
    mean = row[f"{m}_mean"]
    if row["n_seeds"] > 1 and np.isfinite(row[f"{m}_seed_std"]):
        return f"{_fmt(mean, digits)} {pm} {_fmt(row[f'{m}_seed_std'], digits)}"
    return _fmt(mean, digits)


def split_privileged(summary) -> tuple:
    """按是否使用特权输入拆成两张表：主表只放纯 IMU 方法。"""
    if "privileged" not in summary.columns:
        return summary, summary.iloc[0:0]
    flag = summary["privileged"].fillna("").astype(str) != ""
    return summary[~flag], summary[flag]


def table_rows(summary, metrics: Sequence[str], digits: int = 3, pm: str = "±",
               privileged: bool = False) -> tuple:
    metrics = [m for m in metrics if f"{m}_mean" in summary.columns]
    header = ["dataset", "split", "model", "n"]
    if privileged:
        header.append("privileged inputs")
    header += [display_name(m) for m in metrics]
    rows = []
    for _, r in summary.iterrows():
        row = [str(r["dataset"]), str(r["split"]), str(r["model"]), str(int(r["n_sequences"]))]
        if privileged:
            row.append(str(r.get("privileged", "")))
        rows.append(row + [_cell(r, m, digits, pm) for m in metrics])
    return header, rows


def _markdown_table(summary, metrics: Sequence[str], digits: int, privileged: bool) -> str:
    header, rows = table_rows(summary, metrics, digits, privileged=privileged)
    fixed = 5 if privileged else 3
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * fixed + ["---:"] * (len(header) - fixed)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def to_markdown(summary, metrics: Sequence[str] = MAIN_METRICS, digits: int = 3) -> str:
    """主表（纯 IMU 方法）；存在特权输入的方法单列一张表，不与主表混排。"""
    plain, privileged = split_privileged(summary)
    out = _markdown_table(plain, metrics, digits, privileged=False)
    if len(privileged):
        out += ("\n**使用特权输入（来自参考真值）的方法** — 不与上表直接比较：\n\n"
                + _markdown_table(privileged, metrics, digits, privileged=True))
    return out


def _tex(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#")):
        text = text.replace(a, b)
    return text


def to_latex(summary, metrics: Sequence[str] = MAIN_METRICS, digits: int = 3) -> str:
    """booktabs 风格表格（只含纯 IMU 方法）；多种子时单元格为 ``均值 $\\pm$ 种子标准差``。"""
    summary, _ = split_privileged(summary)
    header, rows = table_rows(summary, metrics, digits, pm="$\\pm$")
    spec = "lll" + "r" * (len(header) - 3)
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
           " & ".join(_tex(h) for h in header) + " \\\\", "\\midrule"]
    out += [" & ".join(_tex(c) if "$" not in c else c for c in r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def make_plots(df, summary, out: Path, metric: str = "ate") -> list:
    """每个 (dataset, split) 的箱线图、逐序列误差 CDF、长度比散点，以及全局帕累托图。"""
    from .plotting import (
        PALETTE,
        plot_boxplot,
        plot_error_cdf,
        plot_length_ratio,
        plot_pareto,
    )

    files = []
    for (dataset, split), part in df.groupby(["dataset", "split"]):
        seq = part.groupby(["model", "sequence_id"], as_index=False).mean(numeric_only=True)
        tag = f"{dataset}_{split}"
        models = sorted(seq["model"].unique())
        pages = [models[k:k + len(PALETTE)] for k in range(0, len(models), len(PALETTE))]
        for page, names in enumerate(pages, 1):
            page_tag = tag if len(pages) == 1 else f"{tag}_page{page:02d}"
            values = {m: seq.loc[seq["model"] == m, metric].to_numpy() for m in names}
            files.append(out / f"box_{metric}_{page_tag}.png")
            plot_boxplot(values, files[-1], ylabel=display_name(metric))
            files.append(out / f"cdf_{metric}_{page_tag}.png")
            plot_error_cdf(values, files[-1], xlabel=f"per-sequence {display_name(metric)}")
            if {"plr", "distance_m"} <= set(seq.columns):
                pts = {m: (seq.loc[seq["model"] == m, "distance_m"],
                           seq.loc[seq["model"] == m, "plr"]) for m in names}
                files.append(out / f"plr_{page_tag}.png")
                plot_length_ratio(pts, files[-1])
        # The Pareto front uses every model in this dataset/split, never one page.
        sub = summary[(summary["dataset"] == dataset) & (summary["split"] == split)]
        if f"{metric}_mean" in sub and sub["params"].notna().any():
            files.append(out / f"pareto_params_{metric}_{tag}.png")
            plot_pareto(list(sub["model"]), sub["params"], sub[f"{metric}_mean"], files[-1],
                        xlabel="parameters", ylabel=display_name(metric))
    return files


def _sha256(path: PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _matrix_row_id(row: Mapping) -> str:
    identity = {key: row[key] for key in MATRIX_ID_FIELDS}
    encoded = json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _safe_component(value: object, label: str) -> str:
    text = str(value) if isinstance(value, str) else ""
    if (not text or any(char in text for char in ("/", "\\", "\x00"))
            or text in (".", "..") or Path(text).is_absolute()
            or len(Path(text).parts) != 1 or Path(text).name != text):
        raise ProtocolMismatch(f"matrix {label} is not a safe path component: {value!r}")
    return text


def _checked_matrix_path(path: Path, root: Path) -> tuple:
    """Reject lexical/canonical escape and every symlink below the trusted matrix root."""
    root = root.resolve(strict=True)
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ProtocolMismatch(f"matrix artifact path escapes root: {path}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ProtocolMismatch(f"matrix artifact path contains a symlink: {current}")
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ProtocolMismatch(f"matrix artifact path resolves outside root: {path}")
    return path, root


def _snapshot_file(path: Path, root: Path) -> dict:
    """Read one regular file once; parsing and the recorded SHA consume these exact bytes."""
    path, root = _checked_matrix_path(path, root)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ProtocolMismatch(f"cannot safely open matrix artifact {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProtocolMismatch(f"matrix artifact is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(data) != after.st_size:
        raise ProtocolMismatch(f"matrix artifact changed while being read: {path}")
    _checked_matrix_path(path, root)
    try:
        linked = path.lstat()
    except FileNotFoundError as exc:
        raise ProtocolMismatch(f"matrix artifact was replaced while being read: {path}") from exc
    if (stat.S_ISLNK(linked.st_mode)
            or (linked.st_dev, linked.st_ino) != (after.st_dev, after.st_ino)):
        raise ProtocolMismatch(f"matrix artifact path was replaced while being read: {path}")
    record = {"path": str(path.relative_to(root)), "bytes": len(data),
              "sha256": _sha256_bytes(data)}
    return {"path": path, "data": data, "record": record}


def _snapshot_json(path: Path, root: Path, label: str) -> tuple:
    snapshot = _snapshot_file(path, root)
    try:
        value = json.loads(snapshot["data"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolMismatch(f"{path}: invalid {label} JSON") from exc
    return value, snapshot["record"]


def _artifact_record(path: Path, relative_to: Path) -> dict:
    return _snapshot_file(path, relative_to)["record"]


def _atomic_dataframe_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _matrix_reason(row: Mapping, fallback: str) -> str:
    parts = [fallback]
    reasons = row.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    parts.extend(str(value) for value in reasons if value)
    for key in ("error", "dependency_reason", "resource_reason", "model_cpu_pending_reason"):
        if row.get(key):
            parts.append(f"{key}: {row[key]}")
    return "; ".join(dict.fromkeys(parts))


def _matrix_unavailable_status(row: Mapping) -> str:
    state = str(row.get("state") or "pending")
    if state in MATRIX_UNSUPPORTED_STATES or state.startswith("unsupported"):
        return "unsupported"
    if state == "running":
        return "running"
    if state == "failed":
        return "failed"
    return "pending"


def _load_matrix_rows(matrix: Path) -> tuple:
    coverage_path = matrix / "coverage.json"
    if not coverage_path.is_file():
        raise FileNotFoundError(f"matrix coverage not found: {coverage_path}")
    coverage, coverage_record = _snapshot_json(coverage_path, matrix, "coverage")
    if not isinstance(coverage, dict) or coverage.get("schema") != "ib.dynamic-matrix.v1":
        raise ProtocolMismatch(
            f"{coverage_path}: expected schema 'ib.dynamic-matrix.v1'")
    base_rows = coverage.get("coverage") if isinstance(coverage, dict) else None
    if not isinstance(base_rows, list):
        raise ProtocolMismatch(f"{coverage_path}: expected a coverage row list")
    inputs = {str(coverage_path): coverage_record}
    rows, identifiers = [], set()
    for base in base_rows:
        if not isinstance(base, dict) or any(key not in base for key in MATRIX_ID_FIELDS):
            raise ProtocolMismatch(f"{coverage_path}: matrix row lacks {MATRIX_ID_FIELDS}")
        if (isinstance(base["seed"], bool) or not isinstance(base["seed"], int)
                or not isinstance(base["input_track"], str) or not base["input_track"]):
            raise ProtocolMismatch(f"{coverage_path}: matrix seed/track identity is invalid")
        identifier = _matrix_row_id(base)
        if base.get("id", identifier) != identifier:
            raise ProtocolMismatch(
                f"{coverage_path}: coverage row id does not match scientific identity")
        if identifier in identifiers:
            raise ProtocolMismatch(f"{coverage_path}: duplicate coverage row id {identifier}")
        identifiers.add(identifier)
        for value, label in ((base["profile"], "profile"), (base["dataset"], "dataset")):
            _safe_component(value, label)
        required = ("data_root", "split_sha256", "dims", "main_xyz", "domain",
                    "checkpoint_training_source")
        if any(key not in base for key in required):
            raise ProtocolMismatch(
                f"{coverage_path}: coverage row lacks scientific fields {required}")
        split_hashes = base["split_sha256"]
        if (not isinstance(base["data_root"], str) or not base["data_root"]
                or not isinstance(split_hashes, Mapping)
                or any(not isinstance(name, str) or Path(name).name != name
                       or not name.endswith(".txt") or not isinstance(digest, str)
                       for name, digest in split_hashes.items())
                or base["dims"] not in (2, 3) or type(base["main_xyz"]) is not bool
                or base["main_xyz"] != (base["dims"] == 3)
                or not isinstance(base["domain"], str) or not base["domain"]
                or (base["checkpoint_training_source"] is not None
                    and not isinstance(base["checkpoint_training_source"], str))):
            raise ProtocolMismatch(f"{coverage_path}: coverage scientific fields are invalid")
        _safe_component(Path(base["data_root"]).name, "data directory")
        row = dict(base)
        state_path = matrix / "run-states" / f"{identifier}.json"
        try:
            state_path.lstat()
        except FileNotFoundError:
            pass
        else:
            state, state_record = _snapshot_json(state_path, matrix, "run-state")
            if (not isinstance(state, dict)
                    or state.get("id", identifier) not in (None, identifier)):
                raise ProtocolMismatch(f"{state_path}: run-state id does not match coverage")
            for key in MATRIX_SCIENTIFIC_FIELDS:
                if key in state and state[key] is not None and state[key] != base.get(key):
                    raise ProtocolMismatch(f"{state_path}: run-state {key} differs from coverage")
            for key, value in state.items():
                if (key in base and key not in MATRIX_STATE_MUTABLE_FIELDS and value is not None
                        and value != base[key]):
                    raise ProtocolMismatch(f"{state_path}: run-state coverage field {key} changed")
            row.update({key: value for key, value in state.items() if value is not None})
            inputs[str(state_path)] = state_record
        authoritative_receipt = base.get("taskview_receipt")
        if authoritative_receipt is None:
            taskview_path = matrix / "taskviews" / f"{row['profile']}__{row['dataset']}.json"
            try:
                taskview_path.lstat()
            except FileNotFoundError:
                pass
            else:
                authoritative_receipt, taskview_record = _snapshot_json(
                    taskview_path, matrix, "taskview")
                inputs[str(taskview_path)] = taskview_record
        state_receipt = row.get("taskview_receipt")
        if (authoritative_receipt is not None and state_receipt is not None
                and state_receipt != authoritative_receipt):
            raise ProtocolMismatch(
                f"{state_path}: taskview receipt differs from authoritative input")
        if authoritative_receipt is None:
            row.pop("taskview_receipt", None)
        else:
            row["taskview_receipt"] = authoritative_receipt
        row["id"] = identifier
        rows.append(row)
    return coverage, rows, list(inputs.values())


def _evaluation_snapshots(run: Path, root: Path, *, require_both: bool) -> dict:
    artifacts = {}
    for name in ("metrics.json", "sequences.csv"):
        path, _ = _checked_matrix_path(run / name, root)
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        artifacts[name] = _snapshot_file(path, root)
    if require_both and set(artifacts) != {"metrics.json", "sequences.csv"}:
        missing = sorted({"metrics.json", "sequences.csv"} - set(artifacts))
        raise FileNotFoundError(f"{run}: complete metrics.json + sequences.csv are required; "
                                f"missing {missing}")
    return artifacts


def _evaluation_protocol(meta: Mapping, metrics_path: Path) -> dict:
    protocol = meta.get("protocol")
    if not isinstance(protocol, Mapping):
        raise MatrixEvaluationInvalid(f"{metrics_path}: matrix result lacks a protocol mapping")
    missing = [key for key in EVAL_PROTOCOL_KEYS if protocol.get(key) is None]
    if missing:
        raise MatrixEvaluationInvalid(
            f"{metrics_path}: evaluation protocol is incomplete, missing {missing}")
    return {key: protocol[key] for key in EVAL_PROTOCOL_KEYS}


def _native_split_snapshot(row: Mapping, split: str, *, not_applicable: bool = False) -> dict:
    """Snapshot one native split and enforce its qualified applicability.

    An applicable split must exist, be hash-bound by coverage, and contain IDs.  A
    ``not_applicable_missing_split`` split may instead be absent only when coverage never
    bound that filename; if coverage did bind an empty file, that exact empty file must still
    exist.  Thus a taskview receipt cannot turn a populated, changed, or deleted qualified
    split into N/A.
    """
    data_root = Path(str(row["data_root"])).expanduser().resolve(strict=True)
    if not data_root.is_dir():
        raise ProtocolMismatch(f"matrix data_root is not a directory: {data_root}")
    split_name = f"{_safe_component(split, 'split')}.txt"
    split_hashes = row.get("split_sha256")
    if not isinstance(split_hashes, Mapping):
        raise ProtocolMismatch(
            f"matrix row {row.get('id')} has no split_sha256 mapping")
    bound_when_qualified = split_name in split_hashes
    split_path = data_root / "splits" / split_name
    try:
        snapshot = _snapshot_file(split_path, data_root)
    except FileNotFoundError as exc:
        if bound_when_qualified:
            raise ProtocolMismatch(
                f"native split {split_name} disappeared after coverage split_sha256") from exc
        if not_applicable:
            return {
                "path": split_path,
                "data": None,
                "record": {
                    "dataset_root": str(data_root),
                    "path": str(split_path.relative_to(data_root)),
                    "status": "missing_unbound",
                    "coverage_hash_bound": False,
                },
                "sequence_ids": [],
            }
        raise ProtocolMismatch(
            f"matrix row {row.get('id')} has no declared source hash for {split_name}") from exc
    if not bound_when_qualified:
        raise ProtocolMismatch(
            f"native split {split_name} exists but coverage split_sha256 did not bind it")
    expected_sha = split_hashes[split_name]
    if snapshot["record"]["sha256"] != expected_sha:
        raise ProtocolMismatch(
            f"native split {split_name} changed from coverage split_sha256")
    try:
        text = snapshot["data"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolMismatch(f"native split {split_name} is not UTF-8") from exc
    sequence_ids = []
    for line in text.splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            sequence_ids.append(value)
    if not_applicable and sequence_ids:
        raise ProtocolMismatch(
            f"native split {split_name} is nonempty but taskview marks it N/A")
    if not not_applicable and not sequence_ids:
        raise MatrixEvaluationInvalid(
            f"taskview-passed native split {split_name} has no sequence IDs")
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ProtocolMismatch(f"native split {split_name} contains duplicate sequence IDs")
    snapshot["record"] = {"dataset_root": str(data_root),
                          "coverage_hash_bound": True, **snapshot["record"]}
    snapshot["sequence_ids"] = sequence_ids
    return snapshot


def _json_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixEvaluationInvalid(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise MatrixEvaluationInvalid(f"{label} must be a finite number")
    return number


def _validate_metric_aggregates(meta: Mapping, evaluated: Sequence[Mapping], fields: Sequence[str],
                                metrics_path: Path, sequences_path: Path,
                                num_sequences: int) -> None:
    aggregates = {name: meta.get(name) for name in MATRIX_AGGREGATE_FIELDS}
    invalid = [name for name, value in aggregates.items() if not isinstance(value, Mapping)]
    if invalid:
        raise MatrixEvaluationInvalid(
            f"{metrics_path}: aggregate fields must be mappings, invalid {invalid}")
    metric_keys = set(aggregates["metrics"])
    if not metric_keys or any(set(aggregates[name]) != metric_keys
                              for name in MATRIX_AGGREGATE_FIELDS[1:]):
        raise MatrixEvaluationInvalid(
            f"{metrics_path}: metrics/median/std/count must have one nonempty key contract")
    missing_core = sorted(set(MATRIX_CORE_METRICS) - metric_keys)
    if missing_core:
        raise MatrixEvaluationInvalid(
            f"{metrics_path}: aggregate contract lacks core metrics {missing_core}")
    missing_columns = sorted(metric_keys - set(fields))
    if missing_columns:
        raise MatrixEvaluationInvalid(
            f"{sequences_path}: CSV lacks aggregate metric columns {missing_columns}")

    for metric in sorted(metric_keys):
        values, value_rows = [], []
        for index, entry in enumerate(evaluated):
            raw = entry.get(metric)
            if raw is None or not str(raw).strip():
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise MatrixEvaluationInvalid(
                    f"{sequences_path}: metric {metric!r} is not numeric") from exc
            if not math.isfinite(value):
                raise MatrixEvaluationInvalid(
                    f"{sequences_path}: metric {metric!r} is not finite")
            values.append(value)
            value_rows.append(index)

        count = aggregates["count"][metric]
        if (isinstance(count, bool) or not isinstance(count, int)
                or count < 0 or count > num_sequences or count != len(values)):
            raise MatrixEvaluationInvalid(
                f"{metrics_path}: count[{metric!r}] disagrees with finite CSV values")
        if metric in MATRIX_CORE_METRICS and count != num_sequences:
            raise MatrixEvaluationInvalid(
                f"{metrics_path}: core metric {metric!r} must be finite for every sequence")

        reported = {name: aggregates[name][metric] for name in ("metrics", "median", "std")}
        if count == 0:
            if any(value is not None for value in reported.values()):
                raise MatrixEvaluationInvalid(
                    f"{metrics_path}: zero-count metric {metric!r} must use null aggregates")
            continue
        mean = float(np.mean(values))
        if metric == "loss" and "num_windows" in fields:
            weights = []
            for index in value_rows:
                raw_weight = evaluated[index].get("num_windows")
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError) as exc:
                    raise MatrixEvaluationInvalid(
                        f"{sequences_path}: loss requires numeric num_windows weights") from exc
                if not math.isfinite(weight) or weight < 0:
                    raise MatrixEvaluationInvalid(
                        f"{sequences_path}: loss has an invalid num_windows weight")
                weights.append(weight)
            if sum(weights) > 0:
                mean = float(np.average(values, weights=weights))
        expected = {
            "metrics": mean,
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if count > 1 else None,
        }
        for name, target in expected.items():
            actual = reported[name]
            if target is None:
                if actual is not None:
                    raise MatrixEvaluationInvalid(
                        f"{metrics_path}: {name}[{metric!r}] must be null for count={count}")
                continue
            number = _json_finite_number(actual, f"{metrics_path}: {name}[{metric!r}]")
            if not math.isclose(number, target, rel_tol=1e-9, abs_tol=1e-12):
                raise MatrixEvaluationInvalid(
                    f"{metrics_path}: {name}[{metric!r}] disagrees with sequences.csv")


def _validate_evaluation_snapshots(run: Path, row: Mapping, split: str,
                                   artifacts: Mapping) -> dict:
    metrics_snapshot, sequences_snapshot = artifacts["metrics.json"], artifacts["sequences.csv"]
    metrics_path, sequences_path = metrics_snapshot["path"], sequences_snapshot["path"]
    try:
        meta = json.loads(metrics_snapshot["data"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatrixEvaluationInvalid(f"{metrics_path}: invalid metrics JSON") from exc
    if not isinstance(meta, dict) or meta.get("mode") != "val":
        raise MatrixEvaluationInvalid(
            f"{metrics_path}: matrix result must be a mode=val evaluation")
    expected = {"model": row.get("profile"), "dataset": row.get("dataset"),
                "split": split, "seed": row.get("seed")}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise MatrixEvaluationInvalid(
                f"{metrics_path}: {key}={meta.get(key)!r} does not match matrix {value!r}")
    num_sequences = meta.get("num_sequences")
    if isinstance(num_sequences, bool) or not isinstance(num_sequences, int) or num_sequences <= 0:
        raise MatrixEvaluationInvalid(f"{metrics_path}: num_sequences must be a positive integer")
    num_skipped = meta.get("num_skipped")
    if isinstance(num_skipped, bool) or not isinstance(num_skipped, int) or num_skipped < 0:
        raise MatrixEvaluationInvalid(f"{metrics_path}: num_skipped must be a nonnegative integer")
    protocol = _evaluation_protocol(meta, metrics_path)
    if any(meta.get(key) is None for key in DIMENSION_FIELDS):
        raise MatrixEvaluationInvalid(
            f"{metrics_path}: explicit JSON dimension evidence is required")
    requested, effective, axes = (meta[key] for key in DIMENSION_FIELDS)
    if (isinstance(requested, bool) or requested not in (2, 3) or effective != requested
            or axes not in (["x", "y"], ["x", "y", "z"]) or len(axes) < requested):
        raise MatrixEvaluationInvalid(f"{metrics_path}: invalid JSON dimension evidence")
    if row.get("dims") is not None and requested != row["dims"]:
        raise MatrixEvaluationInvalid(
            f"{metrics_path}: evaluated dimensions differ from matrix row dims")
    if protocol.get("metric_dims") != requested:
        raise MatrixEvaluationInvalid(
            f"{metrics_path}: protocol metric_dims differs from JSON evidence")

    try:
        csv_text = sequences_snapshot["data"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MatrixEvaluationInvalid(f"{sequences_path}: CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    fields = list(reader.fieldnames or ())
    if not fields:
        raise MatrixEvaluationInvalid(f"{sequences_path}: CSV has no header")
    if len(fields) != len(set(fields)):
        raise MatrixEvaluationInvalid(f"{sequences_path}: CSV has duplicate header fields")
    required = {"sequence_id", "skipped", *DIMENSION_FIELDS}
    if not required <= set(fields):
        raise MatrixEvaluationInvalid(
            f"{sequences_path}: CSV lacks required evidence {sorted(required - set(fields))}")
    rows = list(reader)
    if not rows:
        raise MatrixEvaluationInvalid(f"{sequences_path}: CSV has a header but no data rows")
    evaluated = [item for item in rows if not item.get("skipped")]
    if not evaluated:
        raise MatrixEvaluationInvalid(f"{sequences_path}: CSV has no non-skipped evaluation rows")
    if len(evaluated) != num_sequences:
        raise MatrixEvaluationInvalid(
            f"{run}: metrics num_sequences={num_sequences} but CSV has "
            f"{len(evaluated)} evaluated rows")
    if num_skipped != len(rows) - len(evaluated):
        raise MatrixEvaluationInvalid(f"{run}: metrics num_skipped disagrees with CSV row count")
    sequence_ids = [item.get("sequence_id", "") for item in rows]
    if any(not value for value in sequence_ids) or len(set(sequence_ids)) != len(sequence_ids):
        raise MatrixEvaluationInvalid(
            f"{sequences_path}: sequence_id values must be nonempty and unique")
    native_split = artifacts.get("native_split")
    if not isinstance(native_split, Mapping):
        raise ProtocolMismatch(f"{run}: native split was not safely snapshotted")
    # CSV order is not scientific identity; uniqueness above makes exact set equality
    # cardinality-preserving while allowing a validator to schedule sequences differently.
    if set(sequence_ids) != set(native_split["sequence_ids"]):
        raise MatrixEvaluationInvalid(
            f"{sequences_path}: evaluated+skipped sequence IDs do not exactly match native split")
    for item in evaluated:
        try:
            csv_requested = float(item["metric_dims_requested"])
            csv_effective = float(item["metric_dims_effective"])
        except (TypeError, ValueError) as exc:
            raise MatrixEvaluationInvalid(
                f"{sequences_path}: invalid dimension evidence for "
                f"{item.get('sequence_id')}") from exc
        if ([csv_requested, csv_effective] != [float(requested), float(effective)]
                or item["predicted_axes"] != ",".join(axes)):
            raise MatrixEvaluationInvalid(
                f"{sequences_path}: dimension evidence disagrees for {item.get('sequence_id')}")
    _validate_metric_aggregates(
        meta, evaluated, fields, metrics_path, sequences_path, num_sequences)

    pd = _pd()
    try:
        frame = pd.read_csv(io.BytesIO(sequences_snapshot["data"]))
    except Exception as exc:  # pandas parser exceptions vary by supported version
        raise MatrixEvaluationInvalid(f"{sequences_path}: CSV cannot be parsed") from exc
    if "skipped" in frame:
        frame = frame[frame["skipped"].isna() | (frame["skipped"] == "")]
    if len(frame) != num_sequences:
        raise MatrixEvaluationInvalid(f"{sequences_path}: parsed CSV row count is unstable")
    frame = frame.copy()
    frame.insert(0, "matrix_row_id", str(row["id"]))
    frame.insert(1, "dataset", str(row["dataset"]))
    frame.insert(2, "split", str(split))
    frame.insert(3, "model", str(row["profile"]))
    frame.insert(4, "seed", int(row["seed"]))
    frame.insert(5, "input_track", str(row.get("input_track", "")))
    frame.insert(6, "matrix_dims", int(row.get("dims", requested)))
    frame.insert(7, "main_xyz", bool(row.get("main_xyz", requested == 3)))
    frame.insert(8, "domain", str(row.get("domain", "")))
    frame.insert(9, "checkpoint_training_source", str(
        row.get("checkpoint_training_source") or ""))
    privileged_inputs = meta.get("privileged_inputs") or []
    if (not isinstance(privileged_inputs, list)
            or any(not isinstance(value, str) for value in privileged_inputs)):
        raise MatrixEvaluationInvalid(f"{metrics_path}: privileged_inputs must be a string list")
    privileged = ",".join(privileged_inputs)
    frame.insert(10, "privileged", privileged)
    frame.insert(11, "run", str(run))
    if "group_id" not in frame:
        frame["group_id"] = frame["sequence_id"]
    else:
        frame["group_id"] = frame["group_id"].fillna(frame["sequence_id"])
    frame["group_id"] = frame["group_id"].astype(str)
    frame["sequence_id"] = frame["sequence_id"].astype(str)
    efficiency = meta.get("efficiency") or {}
    if not isinstance(efficiency, Mapping):
        raise MatrixEvaluationInvalid(f"{metrics_path}: efficiency must be a mapping")
    for key in ("params", "flops"):
        frame[key] = efficiency.get(key, np.nan)
    return {"metadata": meta, "protocol": protocol, "rows": frame,
            "privileged": privileged,
            "metrics": metrics_snapshot["record"],
            "sequences": sequences_snapshot["record"],
            "native_split": native_split["record"]}


def validate_matrix_evaluation(run: PathLike, row: Mapping, split: str) -> dict:
    """Strictly validate one matrix evaluation from a single immutable byte snapshot."""
    split = _safe_component(split, "split")
    run = Path(run)
    if run.is_symlink():
        raise ProtocolMismatch(f"matrix evaluation directory is a symlink: {run}")
    run = run.resolve(strict=True)
    artifacts = _evaluation_snapshots(run, run, require_both=True)
    artifacts["native_split"] = _native_split_snapshot(row, split)
    return _validate_evaluation_snapshots(run, row, split, artifacts)


def _ledger_artifact_fields(artifacts: Mapping) -> dict:
    metrics = artifacts.get("metrics.json", {}).get("record", {})
    sequences = artifacts.get("sequences.csv", {}).get("record", {})
    native = artifacts.get("native_split", {}).get("record", {})
    return {"metrics_sha256": metrics.get("sha256", ""),
            "sequences_sha256": sequences.get("sha256", ""),
            "native_split_sha256": native.get("sha256", "")}


def _evidence_entry(base: Mapping, reason: str, artifacts: Mapping) -> dict:
    return {**base, "reason": reason,
            "artifacts": [artifacts[name]["record"] for name in sorted(artifacts)]}


def _canonical_protocol(protocol: Mapping) -> tuple:
    def freeze(value):
        if isinstance(value, list):
            return tuple(freeze(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((key, freeze(item)) for key, item in value.items()))
        return value
    return tuple((key, freeze(protocol[key])) for key in EVAL_PROTOCOL_KEYS)


def _check_matrix_summary_protocols(evaluations: Sequence[Mapping]) -> None:
    seen = {}
    stratum_keys = tuple(
        key for key in MATRIX_SUMMARY_KEYS if key != "privileged"
    ) + ("privileged",)
    for evaluation in evaluations:
        stratum = tuple(evaluation.get(key, "") for key in stratum_keys)
        protocol = _canonical_protocol(evaluation["protocol"])
        previous = seen.setdefault(stratum, (evaluation["seed"], protocol))
        if previous[1] != protocol:
            raise ProtocolMismatch(
                "matrix evaluation protocol differs within one summary stratum across seeds: "
                f"{dict(zip(stratum_keys, stratum))}; seed {previous[0]} != "
                f"seed {evaluation['seed']}")


def collect_matrix_results(matrix: PathLike, out: PathLike) -> dict:
    """Collect a dynamic matrix without mutating its coverage, run-state, or taskview files.

    Only ``state=completed`` evaluations admitted by a passed taskview are eligible.  Residual
    artifacts from every other state are hashed as ignored evidence.  Invalid completed results
    remain visible as rejected evidence and never enter the per-sequence table.
    """
    matrix, out = Path(matrix).resolve(strict=True), Path(out)
    coverage, rows, input_files = _load_matrix_rows(matrix)
    frames, ledger, evaluations = [], [], []
    ignored, rejected, not_applicable = [], [], []
    for row in rows:
        receipt = row.get("taskview_receipt")
        if receipt is not None and (not isinstance(receipt, Mapping)
                                    or receipt.get("status") not in ("passed", "failed", "pending")
                                    or not isinstance(receipt.get("splits"), Mapping)):
            raise ProtocolMismatch(f"matrix row {row['id']} has an invalid taskview receipt schema")
        taskview_status = receipt.get("status") if isinstance(receipt, Mapping) else None
        split_receipts = receipt.get("splits") if isinstance(receipt, Mapping) else None
        if not isinstance(split_receipts, dict):
            split_receipts = {}
        for split, split_receipt in split_receipts.items():
            _safe_component(split, "split")
            if (not isinstance(split_receipt, Mapping)
                    or split_receipt.get("status") not in MATRIX_SPLIT_STATUSES):
                raise ProtocolMismatch(
                    f"matrix taskview split {split!r} has unsupported status/schema")
        eval_splits = {name: value for name, value in split_receipts.items()
                       if name not in ("train", "val") and not name.startswith("official_")}
        if not eval_splits:
            if row.get("state") == "completed":
                raise ProtocolMismatch(
                    f"completed matrix row {row['id']} has no declared evaluation splits")
            status = ("unsupported" if taskview_status not in (None, "pending") else "pending")
            reason = _matrix_reason(
                row, "taskview receipt has no declared applicable evaluation splits")
            ledger.append({"matrix_row_id": row["id"], "model": row["profile"],
                           "dataset": row["dataset"], "split": "*", "seed": row["seed"],
                           "input_track": row.get("input_track", ""),
                           "matrix_dims": row["dims"], "main_xyz": row["main_xyz"],
                           "domain": row["domain"], "checkpoint_training_source": (
                               row.get("checkpoint_training_source") or ""),
                           "row_state": row.get("state", ""),
                           "taskview_status": taskview_status or "missing",
                           "split_taskview_status": "missing", "availability": status,
                           "artifact_disposition": "none", "reason": reason,
                           "run": "", "num_sequences": "",
                           "metrics_sha256": "", "sequences_sha256": ""})
            continue
        for split, split_receipt in sorted(eval_splits.items()):
            split_status = split_receipt["status"]
            base = {"matrix_row_id": row["id"], "model": row["profile"],
                    "dataset": row["dataset"], "split": split, "seed": row["seed"],
                    "input_track": row.get("input_track", ""),
                    "matrix_dims": row["dims"], "main_xyz": row["main_xyz"],
                    "domain": row["domain"], "checkpoint_training_source": (
                        row.get("checkpoint_training_source") or ""),
                    "row_state": row.get("state", ""),
                    "taskview_status": taskview_status or "missing",
                    "split_taskview_status": split_status or "missing"}
            data_dir = _safe_component(Path(row["data_root"]).name, "data directory")
            run = (matrix / "runs" / row["id"] / row["profile"] / data_dir
                   / f"seed{row['seed']}" / split)
            _checked_matrix_path(run, matrix)
            artifacts = _evaluation_snapshots(run, matrix, require_both=False)
            hashes = _ledger_artifact_fields(artifacts)
            if split_status == "not_applicable_missing_split":
                reason = "taskview split status: not_applicable_missing_split"
                has_residual_results = bool(artifacts)
                native_split = _native_split_snapshot(row, split, not_applicable=True)
                artifacts["native_split"] = native_split
                hashes = _ledger_artifact_fields(artifacts)
                not_applicable.append({**base, "reason": reason,
                                       "native_split": native_split["record"]})
                if has_residual_results:
                    ignored.append(_evidence_entry(base, reason, artifacts))
                ledger.append({**base, "availability": "not_applicable",
                               "artifact_disposition": (
                                   "ignored" if has_residual_results else "none"),
                               "reason": reason,
                               "run": str(run) if has_residual_results else "",
                               "num_sequences": "", **hashes})
                continue
            if taskview_status != "passed":
                if row.get("state") == "completed":
                    raise ProtocolMismatch(
                        f"completed matrix row {row['id']} lacks a passed taskview receipt")
                availability = (
                    "unsupported" if taskview_status not in (None, "pending") else "pending")
                detail = receipt.get("error") if isinstance(receipt, Mapping) else None
                reason = _matrix_reason(
                    row, f"taskview={taskview_status or 'missing'}, "
                    f"split={split_status or 'missing'}" + (f", error={detail}" if detail else ""))
                if artifacts:
                    ignored.append(_evidence_entry(base, reason, artifacts))
                ledger.append({**base, "availability": availability, "reason": reason,
                               "artifact_disposition": "ignored" if artifacts else "none",
                               "run": str(run) if artifacts else "", "num_sequences": "",
                               **hashes})
                continue
            if row.get("state") != "completed":
                availability = _matrix_unavailable_status(row)
                reason = _matrix_reason(row, f"matrix row state: {row.get('state', 'pending')}")
                if artifacts:
                    ignored.append(_evidence_entry(base, reason, artifacts))
                ledger.append({**base, "availability": availability,
                               "artifact_disposition": "ignored" if artifacts else "none",
                               "reason": reason, "run": str(run) if artifacts else "",
                               "num_sequences": "", **hashes})
                continue
            if set(artifacts) != {"metrics.json", "sequences.csv"}:
                missing = sorted({"metrics.json", "sequences.csv"} - set(artifacts))
                reason = f"completed applicable split lacks result artifacts: {missing}"
                rejected.append(_evidence_entry(base, reason, artifacts))
                ledger.append({**base, "availability": "invalid",
                               "artifact_disposition": "rejected", "reason": reason,
                               "run": str(run), "num_sequences": "", **hashes})
                continue
            try:
                artifacts["native_split"] = _native_split_snapshot(row, split)
                accepted = _validate_evaluation_snapshots(run, row, split, artifacts)
            except MatrixEvaluationInvalid as exc:
                hashes = _ledger_artifact_fields(artifacts)
                reason = f"{type(exc).__name__}: {exc}"
                rejected.append(_evidence_entry(base, reason, artifacts))
                ledger.append({**base, "availability": "invalid",
                               "artifact_disposition": "rejected", "reason": reason,
                               "run": str(run), "num_sequences": "", **hashes})
                continue
            except ProtocolMismatch:
                # Source/schema/path trust failures are matrix-wide and must not be
                # downgraded to one invalid completed result merely because
                # ConfigError inherits ValueError.
                raise
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
                hashes = _ledger_artifact_fields(artifacts)
                reason = f"invalid completed evaluation content: {type(exc).__name__}: {exc}"
                rejected.append(_evidence_entry(base, reason, artifacts))
                ledger.append({**base, "availability": "invalid",
                               "artifact_disposition": "rejected", "reason": reason,
                               "run": str(run), "num_sequences": "", **hashes})
                continue
            else:
                hashes = _ledger_artifact_fields(artifacts)
                frames.append(accepted["rows"])
                evaluation = {**base, "privileged": accepted["privileged"],
                              "num_sequences": int(len(accepted["rows"])),
                              "protocol": accepted["protocol"],
                              "metrics": accepted["metrics"],
                              "sequences": accepted["sequences"],
                              "native_split": accepted["native_split"]}
                evaluations.append(evaluation)
                ledger.append({**base, "availability": "available", "reason": "",
                               "artifact_disposition": "accepted",
                               "run": str(run), "num_sequences": int(len(accepted["rows"])),
                               **hashes})

    _check_matrix_summary_protocols(evaluations)

    pd = _pd()
    if frames:
        per_sequence = pd.concat(frames, ignore_index=True)
        summary_metrics = [name for name in MAIN_METRICS if name in per_sequence.columns]
        summary = summarize(per_sequence, summary_metrics, group_keys=MATRIX_SUMMARY_KEYS)
    else:
        summary_metrics = []
        per_sequence = pd.DataFrame(columns=["matrix_row_id", *KEYS, "seed", "input_track",
                                             "matrix_dims", "main_xyz", "domain",
                                             "checkpoint_training_source", "privileged",
                                             "run", "sequence_id", "group_id"])
        summary = pd.DataFrame(columns=[*MATRIX_SUMMARY_KEYS, "n_sequences", "n_seeds",
                                        "params", "flops"])
    availability = pd.DataFrame(ledger, columns=[
        "matrix_row_id", "model", "dataset", "split", "seed", "input_track", "matrix_dims",
        "main_xyz", "domain", "checkpoint_training_source", "row_state", "taskview_status",
        "split_taskview_status", "availability", "artifact_disposition", "reason", "run",
        "num_sequences", "metrics_sha256", "sequences_sha256", "native_split_sha256",
    ])
    out.mkdir(parents=True, exist_ok=True)
    _atomic_dataframe_csv(per_sequence, out / "per_sequence.csv")
    _atomic_dataframe_csv(summary, out / "summary.csv")
    _atomic_dataframe_csv(availability, out / "availability.csv")
    counts = {str(key): int(value)
              for key, value in availability["availability"].value_counts().items()}
    created = datetime.now(timezone.utc).isoformat()
    info = {"schema": "ib.matrix-results.v1", "created_utc": created,
            "matrix": str(matrix), "coverage_sha256": input_files[0]["sha256"],
            "matrix_rows": len(rows), "evaluation_splits": len(ledger),
            "available_evaluations": len(evaluations),
            "available_sequence_rows": int(len(per_sequence)),
            "summary_metrics": summary_metrics,
            "availability_counts": counts,
            "coverage_source_manifest_sha256": (coverage.get("source") or {}).get(
                "source_manifest_sha256"),
            "files": ["per_sequence.csv", "summary.csv", "availability.csv",
                      "report.json", "artifacts-sha256.json"],
            "note": ("summary.csv contains completed, valid evaluations only and is already "
                     "stratified by track, dimensions, domain, and checkpoint training source")}
    generated = [_artifact_record(out / name, out) for name in
                 ("per_sequence.csv", "summary.csv", "availability.csv")]
    manifest = {"schema": "ib.matrix-results-artifacts.v1",
                "created_utc": created, "matrix_inputs": input_files,
                "accepted_evaluations": evaluations, "ignored_evaluations": ignored,
                "rejected_evaluations": rejected,
                "not_applicable_evaluations": not_applicable, "generated": generated,
                "report_excluded_from_manifest": (
                    "report.json stores this manifest's root SHA and is excluded to avoid "
                    "a hash cycle"),
                "scope": "small result metadata/CSV only; checkpoints and predictions excluded"}
    json_save(out / "artifacts-sha256.json", manifest)
    info["artifact_manifest_sha256"] = _sha256(out / "artifacts-sha256.json")
    json_save(out / "report.json", info)
    LOGGER.info(f"matrix report: {len(evaluations)} evaluations, {len(per_sequence)} rows -> {out}")
    return info


def build_report(runs: PathLike, out: PathLike, metrics: Sequence[str] = MAIN_METRICS,
                 plots: bool = True, n_boot: int = 2000) -> dict:
    """汇总 ``runs`` 下全部评测结果，写出 CSV / Markdown / LaTeX 表格与图。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = check_protocols(find_runs(runs))
    df = load_runs(runs)
    if df.empty:
        raise FileNotFoundError(f"no evaluation results (metrics.json + sequences.csv) "
                                f"under {runs}")
    # 所有浮点列都是逐序列指标（含 duration_s / distance_m），效率指标单独处理
    all_metrics = [c for c in df.columns
                   if df[c].dtype.kind == "f" and c not in ("params", "flops", *DIMENSION_FIELDS)]
    summary = summarize(df, all_metrics, n_boot=n_boot)
    tests = all_pairwise_tests(df, [m for m in ("ate", "rte") if m in df.columns])
    df.to_csv(out / "per_sequence.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    tests.to_csv(out / "wilcoxon.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summary, metrics), encoding="utf-8")
    (out / "summary.tex").write_text(to_latex(summary, metrics), encoding="utf-8")
    files = make_plots(df, summary, out) if plots else []
    info = {"runs": str(runs), "num_runs": int(df["run"].nunique()),
            "num_rows": int(len(df)), "datasets": sorted(df["dataset"].unique()),
            "models": sorted(df["model"].unique()), "protocol": protocol,
            "files": sorted(str(p.relative_to(out)) for p in out.iterdir()),
            "plots": [str(p) for p in files]}
    json_save(out / "report.json", info)
    LOGGER.info(f"report: {info['num_runs']} runs → {out}")
    LOGGER.info("\n" + to_markdown(summary, metrics))
    return info
