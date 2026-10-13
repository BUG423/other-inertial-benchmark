"""结果对象：``Trajectory``（逐帧轨迹）、``SequenceResult``（一条序列的全部输出与指标）、
``RunResult``（一次评测的聚合结果与文件写出）。
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from ..metrics import MAIN_METRICS, path_length, sequence_metrics
from ..metrics.uncertainty import gaussian_uncertainty_metrics
from ..utils import LOGGER, json_save, to_builtin

PathLike = Union[str, Path]


@dataclass
class Trajectory:
    """统一时间轴上的预测轨迹。"""

    t: np.ndarray  # (N,)
    pos: np.ndarray  # (N, D)
    vel: Optional[np.ndarray] = None  # (N, D) 逐帧速度（积分输入）
    sequence_id: str = ""

    def __len__(self) -> int:
        return len(self.t)

    def length(self, resolution: Optional[float] = 1.0) -> float:
        rate = 1.0 / float(np.median(np.diff(self.t))) if len(self.t) > 1 else 200.0
        return path_length(self.pos, None, rate, resolution, self.pos.shape[1])

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"t": self.t, "pos": self.pos}
        if self.vel is not None:
            arrays["vel"] = self.vel
        np.savez_compressed(path, sequence_id=np.array(self.sequence_id), **arrays)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "Trajectory":
        with np.load(path) as f:
            return cls(t=f["t"], pos=f["pos"], vel=f["vel"] if "vel" in f else None,
                       sequence_id=str(f["sequence_id"]))

    def plot(self, path: Optional[PathLike] = None, gt: Optional[np.ndarray] = None):
        from ..utils.plotting import plot_trajectory

        return plot_trajectory(self.pos, gt if gt is not None else np.full_like(self.pos, np.nan),
                               title=self.sequence_id, path=path)


ARRAYS = ("t", "pos_pred", "pos_gt", "pos_oracle", "valid_pose", "t_window", "starts",
          "vel_pred", "vel_target", "window_valid", "window_valid_input",
          "window_valid_target", "logstd", "uncertainty_mean_raw", "uncertainty_target_raw",
          "uncertainty_logstd_raw", "uncertainty_valid_raw", "uncertainty_basis_raw")
OUTPUT_PREFIX = "out_"  # 其他逐窗口模型输出在 .npz 中的键前缀
DIMENSION_FIELDS = ("metric_dims_requested", "metric_dims_effective", "predicted_axes")


@dataclass
class SequenceResult:
    """一条序列的推理结果（世界系）与指标。"""

    sequence_id: str
    dataset: str = "unknown"
    group_id: str = "unknown"
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pos_pred: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pos_gt: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pos_oracle: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    valid_pose: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    t_window: np.ndarray = field(default_factory=lambda: np.zeros(0))
    starts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    vel_pred: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    vel_target: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    window_valid: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    # 拆分的窗口有效性：输入（IMU + 所需姿态）与目标（参考位姿），见 DESIGN 第 5 节
    window_valid_input: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    window_valid_target: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    logstd: Optional[np.ndarray] = None
    # 模型的其他逐窗口输出（键 → (K, ...)），保持模型输出的视图坐标系与**逐窗口**布局
    # （不做重叠合并，也不做无效窗口插值）；``logstd`` 同理
    outputs: dict = field(default_factory=dict)
    frame: str = "gravity_world"
    rate: float = 200.0
    loss: Optional[float] = None
    skipped: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    # 只在成功计算指标后写入；旧 NPZ 缺少这些字段时保持未知，不推断历史计算维度。
    metric_dims_requested: Optional[int] = None
    metric_dims_effective: Optional[int] = None
    predicted_axes: Optional[list] = None
    # Optional additive raw-task fields. Legacy vel_pred/vel_target are world-frame,
    # overlap-merged velocities; legacy logstd remains the untouched model output.
    uncertainty_mean_raw: Optional[np.ndarray] = None
    uncertainty_target_raw: Optional[np.ndarray] = None
    uncertainty_logstd_raw: Optional[np.ndarray] = None
    uncertainty_valid_raw: Optional[np.ndarray] = None
    uncertainty_basis_raw: Optional[np.ndarray] = None
    uncertainty: dict = field(default_factory=lambda: {
        "schema": "ib.uncertainty.raw-task.v1", "supported": False,
        "head_trained": False, "reason": "legacy_or_missing_raw_task_contract"})

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory(self.t, self.pos_pred, sequence_id=self.sequence_id)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def distance(self) -> float:
        dims = self.metric_dims_effective or 2
        return path_length(self.pos_gt, self.valid_pose, self.rate, dims=dims) \
            if len(self.t) else 0.0

    def position_error(self, dims: int = 2) -> np.ndarray:
        """按所选维度计算逐帧位置误差（无效参考处为 NaN）。"""
        self._check_dimensions(dims)
        err = np.linalg.norm(self.pos_pred[:, :dims] - self.pos_gt[:, :dims], axis=1)
        return np.where(self.valid_pose, err, np.nan)

    def _check_dimensions(self, dims: int, include_velocity: bool = False) -> None:
        if isinstance(dims, bool) or not isinstance(dims, (int, np.integer)) or dims not in (2, 3):
            raise ValueError(f"metric_dims must be 2 or 3, got {dims!r}")
        arrays = {"pos_pred": self.pos_pred, "pos_gt": self.pos_gt}
        if include_velocity:
            arrays.update({name: getattr(self, name) for name in
                           ("pos_oracle", "vel_pred", "vel_target")
                           if getattr(self, name) is not None and np.size(getattr(self, name))})
        for name, values in arrays.items():
            shape = np.shape(values)
            if len(shape) != 2 or shape[1] not in (2, 3) or shape[1] < dims:
                raise ValueError(
                    f"{self.sequence_id}: metric_dims={dims} requires {name} with at least "
                    f"{dims} position/velocity axes, got shape {shape}; "
                    "use metric_dims=2 for a model with only XY outputs (2D reference protocol)")

    def dimension_metadata(self) -> dict:
        return {key: getattr(self, key) for key in DIMENSION_FIELDS}

    def uncertainty_metrics(self, dims: int) -> dict:
        """Score matched unmerged task rows; unavailable distributions stay NA.

        Scores are conditional network-head diagnostics in the declared task frame
        and units, not uncertainty of the integrated world trajectory. XYZ and XY
        use the identical raw outputs/mask, with XY slicing the last axis only.
        """
        contract = self.uncertainty
        empty = np.empty((0, dims), dtype=float)
        unavailable = lambda: gaussian_uncertainty_metrics(empty, empty, None)
        if not contract.get("supported") or not contract.get("head_trained"):
            return unavailable()
        arrays = (self.uncertainty_mean_raw, self.uncertainty_target_raw,
                  self.uncertainty_logstd_raw, self.uncertainty_valid_raw)
        if any(value is None for value in arrays):
            contract["reason"] = "missing_raw_task_arrays"
            return unavailable()
        mean, target, logstd, valid = (np.asarray(value) for value in arrays)
        if (mean.ndim not in (2, 3) or target.shape != mean.shape or mean.shape[-1] < dims
                or mean.shape[-1] != contract.get("dims")):
            raise ValueError("uncertainty raw mean/target must match declared task axes/layout")
        expected_layout = "window" if mean.ndim == 2 else contract.get("layout")
        if expected_layout != contract.get("layout") or expected_layout not in (
                "window", "frame", "steps"):
            raise ValueError("uncertainty raw arrays disagree with declared task layout")
        expected_mask = (mean.shape[0], 1) if mean.ndim == 2 else mean.shape[:-1]
        if valid.shape != expected_mask:
            raise ValueError("uncertainty valid mask must match raw output rows")
        if logstd.shape not in (mean.shape, mean.shape[:-1] + (1,)):
            raise ValueError("uncertainty raw logstd must match mean layout or be isotropic")
        valid = valid.astype(bool, copy=False).reshape(-1)
        if not valid.any():
            contract["reason"] = "no_valid_raw_outputs"
            return unavailable()
        # Apply the actual loss's distribution parameterization to a copy. Raw
        # arrays and all legacy prediction values remain unchanged, including floors.
        sigma = logstd.astype(np.float64, copy=True) + float(contract.get("logstd_offset", 0))
        lo, hi = contract.get("min_logstd"), contract.get("max_logstd")
        if lo is not None or hi is not None:
            sigma = np.clip(sigma, lo, hi)
        channels = sigma.shape[-1]
        return gaussian_uncertainty_metrics(
            mean[..., :dims].reshape(-1, dims), target[..., :dims].reshape(-1, dims),
            sigma[..., :dims].reshape(-1, min(channels, dims)), valid)

    def compute_metrics(self, dims: int = 2, rte_delta: float = 60.0,
                        t_rte: Iterable[float] = (1.0, 10.0),
                        d_rte: Iterable[float] = (10.0,), min_speed: float = 0.2) -> dict:
        if self.skipped:
            self.metrics = {}
            return self.metrics
        self._check_dimensions(dims, include_velocity=True)
        # Materialize optional iterables once: XYZ and XY use the same requested windows.
        t_rte, d_rte = tuple(t_rte), tuple(d_rte)
        self.metrics = sequence_metrics(
            self.pos_pred, self.pos_gt, self.valid_pose, self.vel_pred, self.vel_target,
            self.window_valid, rate=self.rate, pos_oracle=self.pos_oracle, dims=dims,
            rte_delta=rte_delta, t_rte=t_rte, d_rte_distance=d_rte, min_speed=min_speed)
        self.metrics.update(self.uncertainty_metrics(dims))
        if dims == 3:
            xy = sequence_metrics(
                self.pos_pred[:, :2], self.pos_gt[:, :2], self.valid_pose,
                self.vel_pred[:, :2], self.vel_target[:, :2], self.window_valid,
                rate=self.rate,
                pos_oracle=self.pos_oracle[:, :2] if self.pos_oracle is not None else None,
                dims=2, rte_delta=rte_delta, t_rte=t_rte, d_rte_distance=d_rte,
                min_speed=min_speed)
            self.metrics.update({f"xy_{key}": value for key, value in xy.items()})
            self.metrics.update({f"xy_{key}": value for key, value in
                                 self.uncertainty_metrics(2).items()})
        self.metric_dims_requested = self.metric_dims_effective = int(dims)
        self.predicted_axes = list("xyz"[:self.pos_pred.shape[1]])
        if self.loss is not None:
            self.metrics["loss"] = self.loss
        return self.metrics

    def to_dict(self) -> dict:
        """一行 ``sequences.csv``。"""
        row = {"sequence_id": self.sequence_id, "group_id": self.group_id,
               "duration_s": self.duration, "distance_m": self.distance,
               "skipped": self.skipped or ""}
        row.update(self.dimension_metadata())
        row["predicted_axes"] = ",".join(self.predicted_axes) if self.predicted_axes else ""
        row.update(self.metrics)
        row["uncertainty_support"] = json.dumps(to_builtin(self.uncertainty), sort_keys=True)
        return row

    def save(self, path: PathLike) -> Path:
        """保存为 ``.npz``（数组 + ``meta`` JSON 字符串；其他模型输出的键加 ``out_`` 前缀）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in ARRAYS if getattr(self, k) is not None}
        arrays.update({OUTPUT_PREFIX + k: np.asarray(v) for k, v in self.outputs.items()})
        meta = {"sequence_id": self.sequence_id, "dataset": self.dataset,
                "group_id": self.group_id, "frame": self.frame, "rate": self.rate,
                "loss": self.loss, "skipped": self.skipped, "metrics": self.metrics,
                "uncertainty": self.uncertainty}
        meta.update(self.dimension_metadata())
        np.savez_compressed(path, meta=np.array(json.dumps(to_builtin(meta))), **arrays)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "SequenceResult":
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            arrays = {k: f[k] for k in ARRAYS if k in f}
            outputs = {k[len(OUTPUT_PREFIX):]: f[k] for k in f.files
                       if k.startswith(OUTPUT_PREFIX)}
        metrics = {k: (math.nan if v is None else v) for k, v in meta.pop("metrics").items()}
        return cls(**meta, **arrays, outputs=outputs, metrics=metrics)

    def plot(self, path: Optional[PathLike] = None):
        from ..utils.plotting import plot_trajectory

        title = self.sequence_id
        if "ate" in self.metrics:
            title += f"  ATE {self.metrics['ate']:.2f} m"
        return plot_trajectory(self.pos_pred, self.pos_gt, self.valid_pose, self.pos_oracle,
                               title, path)


def _stats(values: list) -> tuple:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan, math.nan, math.nan, 0
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else math.nan
    return float(arr.mean()), float(np.median(arr)), std, int(len(arr))


@dataclass
class RunResult:
    """一次评测（一个模型 × 一个数据集划分）的全部序列结果。"""

    sequences: list
    dataset: str = "unknown"
    split: str = "unknown"
    model: str = "unknown"
    weights: Optional[str] = None
    cfg: dict = field(default_factory=dict)
    efficiency: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    # 序列级/标定型模型在 train 划分上拟合的标定量（随结果一起发布）
    calibration: dict = field(default_factory=dict)
    save_dir: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def evaluated(self) -> list:
        return [s for s in self.sequences if not s.skipped]

    def aggregate(self) -> dict:
        keys: list = []
        for s in self.evaluated:
            keys += [k for k in s.metrics if k not in keys]
        out = {"mean": {}, "median": {}, "std": {}, "count": {}}
        for k in keys:
            mean, median, std, n = _stats([s.metrics.get(k) for s in self.evaluated])
            out["mean"][k], out["median"][k], out["std"][k], out["count"][k] = mean, median, \
                std, n
        # 窗口级损失按窗口数加权
        weights = [(s.metrics.get("loss"), s.metrics.get("num_windows", 0))
                   for s in self.evaluated if s.metrics.get("loss") is not None]
        total = sum(w for _, w in weights)
        if total:
            out["mean"]["loss"] = float(sum(v * w for v, w in weights) / total)
        return out

    @property
    def metrics(self) -> dict:
        """各指标跨序列均值（``fitness`` 的缺省来源）。"""
        return self.aggregate()["mean"]

    def value(self, key: str, stat: str = "mean") -> Optional[float]:
        """取某个指标的跨序列聚合值；``stat`` 为 ``"mean"``（缺省）或 ``"median"``。

        未知指标名返回 ``None``（调用方给出可用键列表）。
        """
        from ..metrics import FITNESS_STATS

        if stat not in FITNESS_STATS:
            raise ValueError(f"stat must be one of {FITNESS_STATS}, got {stat!r}")
        return self.aggregate()[stat].get(key)

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]

    @property
    def privileged_inputs(self) -> list:
        """本次评测使用的特权输入（来自参考真值），报表中必须单列。"""
        from ..data.views import PRIVILEGED_INPUTS

        used = self.cfg.get("extra_inputs") or ()
        return [name for name in used if name in PRIVILEGED_INPUTS]

    def dimension_metadata(self) -> dict:
        """Dimensions actually evaluated; historical metric evidence is never inferred."""
        metadata = [s.dimension_metadata() for s in self.evaluated]
        if not metadata or all(all(value is None for value in m.values()) for m in metadata):
            return dict.fromkeys(DIMENSION_FIELDS)
        first = metadata[0]
        if any(any(value is None for value in m.values()) or m != first for m in metadata):
            raise ValueError("run contains inconsistent or missing evaluation dimensions; "
                             "re-evaluate all sequences with one metric_dims protocol")
        requested = self.cfg.get("metric_dims")
        if requested is not None and requested != first["metric_dims_requested"]:
            raise ValueError("run metric_dims differs from the dimensions actually evaluated; "
                             "re-evaluate before saving results")
        return first

    def to_dict(self) -> dict:
        from ..cfg import protocol_from_cfg

        agg = self.aggregate()
        cfg = self.cfg
        return {
            "mode": "val",
            "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
            "model": self.model,
            "weights": self.weights,
            "dataset": self.dataset,
            "split": self.split,
            "seed": cfg.get("seed"),
            "dataset_fingerprint": (self.env.get("dataset") or {}).get("fingerprint"),
            "git": (self.env.get("git") or {}).get("commit"),
            "protocol": protocol_from_cfg(cfg),
            **self.dimension_metadata(),
            # 特权输入（如来自参考真值的初速度）必须显式标记，报表中单列、不与纯 IMU 方法混排
            "privileged_inputs": self.privileged_inputs,
            "num_sequences": len(self.evaluated),
            "num_skipped": len(self.sequences) - len(self.evaluated),
            "skipped": {s.sequence_id: s.skipped for s in self.sequences if s.skipped},
            "metrics": agg["mean"],
            "median": agg["median"],
            "std": agg["std"],
            "count": agg["count"],
            "efficiency": self.efficiency,
            "calibration": self.calibration,
            # Present even when predictions=False: support/NA and coordinates are
            # scientific evidence rather than optional visualization payloads.
            "uncertainty_support": {s.sequence_id: s.uncertainty for s in self.evaluated},
        }

    def summary(self, keys: Iterable[str] = MAIN_METRICS) -> str:
        m = self.metrics
        parts = [f"{k}={m[k]:.4g}" for k in keys if k in m and m[k] == m[k]]
        return (f"{self.model} on {self.dataset}/{self.split} "
                f"({len(self.evaluated)} seq): " + ", ".join(parts))

    def save(self, save_dir: Optional[PathLike] = None, predictions: bool = True,
             plots: bool = True, max_plots: int = 12) -> Path:
        """写出 ``metrics.json``、``sequences.csv``、``predictions/*.npz``、``plots/*.png``。"""
        save_dir = Path(save_dir or self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        rows = [s.to_dict() for s in self.sequences]
        header: list = []
        for r in rows:
            header += [k for k in r if k not in header]
        with open(save_dir / "sequences.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: _csv_value(r.get(k)) for k in header})
        json_save(save_dir / "metrics.json", self.to_dict())
        if predictions:
            for s in self.evaluated:
                s.save(save_dir / "predictions" / f"{s.sequence_id}.npz")
        if plots and self.evaluated:
            try:
                self.plot(save_dir / "plots", max_plots)
            except Exception as exc:  # noqa: BLE001 - 绘图失败不影响结果文件
                LOGGER.warning(f"plotting failed: {exc}")
        return save_dir

    def plot(self, out: PathLike, max_plots: int = 12) -> list:
        from ..utils import plotting

        out = Path(out)
        seqs = sorted(self.evaluated, key=lambda s: s.sequence_id)
        evidence = self.dimension_metadata()
        dims = int(self.cfg.get("metric_dims") or evidence["metric_dims_requested"] or 2)
        for s in seqs:
            s._check_dimensions(dims)
        if not seqs:
            return []
        error_label = f"{dims}D position error (m)"
        files = []
        for s in seqs[:max_plots]:
            files.append(out / f"traj_{s.sequence_id}.png")
            s.plot(files[-1])
        items = [{"pred": s.pos_pred, "gt": s.pos_gt, "valid": s.valid_pose,
                  "oracle": s.pos_oracle, "title": s.sequence_id} for s in seqs]
        files.append(out / "trajectories.png")
        plotting.plot_trajectory_grid(items, files[-1], max_plots)
        errors = np.concatenate([s.position_error(dims=dims) for s in seqs])
        files.append(out / "error_cdf.png")
        plotting.plot_error_cdf({self.model: errors}, files[-1], xlabel=error_label)
        curves = [{"t": s.t, "error": s.position_error(dims=dims), "title": s.sequence_id}
                  for s in seqs]
        files.append(out / "error_over_time.png")
        plotting.plot_error_over_time(curves, files[-1], max_plots, ylabel=error_label)
        files.append(out / "ate_box.png")
        plotting.plot_boxplot({self.model: [s.metrics.get("ate", math.nan) for s in seqs]},
                              files[-1])
        files.append(out / "length_ratio.png")
        plotting.plot_length_ratio({self.model: ([s.distance for s in seqs],
                                                 [s.metrics.get("plr", math.nan) for s in seqs])},
                                   files[-1])
        return files


def _csv_value(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return ""
    return v
