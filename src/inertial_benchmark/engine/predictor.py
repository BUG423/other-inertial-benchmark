"""推理与轨迹重建（DESIGN 第 5 节）。

1. 以 ``eval_stride`` 滑窗推理，窗口速度的时间戳为目标对应时刻（平均速度/位移为窗口中心，
   ``velocity_at_end`` 为窗口末端）；
2. 含无效样本的窗口照常推理；**预测**只在输入（IMU 与所需姿态）无效的窗口上按相邻有效窗口插值替换，
   **目标**只在参考位姿无效的窗口上替换；
3. 逐帧速度为窗口速度的分段线性插值，首尾半窗内常值外推；
4. 梯形积分，起点锚定首个有效参考位置，不做任何对齐；
5. 同时积分窗口目标得到 oracle 轨迹。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch

from ..cfg import get_cfg, is_checkpoint
from ..data.format import Sequence, load_sequence
from ..data.views import SequenceView, ViewConfig
from ..nn.base import InputSpec, SequenceModel
from ..nn.registry import build_model
from ..utils import LOGGER
from ..utils.callbacks import default_callbacks, run_callbacks
from ..utils.torch_utils import load_checkpoint, select_device
from .results import SequenceResult, Trajectory

PathLike = Union[str, Path]


def load_model(cfg: Any, device: Optional[torch.device] = None):
    """按配置构建模型。

    ``cfg.model`` 为 checkpoint 时加载其结构与权重；``cfg.pretrained`` 另行加载（非严格）权重。
    """
    if is_checkpoint(cfg.model):
        ckpt = load_checkpoint(cfg.model)
        model = build_model(cfg, InputSpec.from_dict(ckpt["input_spec"]), ckpt["model_cfg"])
        model.load_state_dict(ckpt["model"])
        # Runtime evidence only; no parameters/state_dict entries are added. An epoch
        # argument alone cannot certify that a fresh uncertainty head was trained.
        if type(ckpt.get("epoch")) is int and ckpt["epoch"] >= 0:
            model._ipb_checkpoint_epoch = ckpt["epoch"]
            model._ipb_checkpoint_source = str(cfg.model)
            model._ipb_uncertainty_checkpoint_evidence = dict(ckpt.get("training_evidence") or {})
            saved_cfg = ckpt.get("model_cfg") or {}
            model._ipb_uncertainty_checkpoint_loss = {
                "loss": saved_cfg.get("loss"),
                "loss_kwargs": dict(saved_cfg.get("loss_kwargs") or {})}
    else:
        model = build_model(cfg)
    if getattr(cfg, "pretrained", None):
        ckpt = load_checkpoint(cfg.pretrained)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        # Non-strict transfer does not certify the destination covariance head.
        model._ipb_checkpoint_epoch = None
        model._ipb_uncertainty_checkpoint_evidence = {}
        model._ipb_uncertainty_checkpoint_loss = {}
        LOGGER.info(f"loaded pretrained weights from {cfg.pretrained} "
                    f"(missing {len(missing)}, unexpected {len(unexpected)})")
    return model.to(device) if device is not None else model


def uncertainty_contract(model: Any, cfg: ViewConfig, epoch: int) -> dict:
    """Explicit raw-task Gaussian contracts; unsupported heads remain unavailable.

    These contracts describe the independently implemented network/loss, not the
    complete paper's filtering system. EqNIO's world covariance is not diagonal:
    its likelihood is evaluated using the canonical mean/target/logstd instead.
    New models must extend this reviewed table or provide an explicit callable
    ``raw_uncertainty_contract(epoch)`` returning the same contract fields.
    """
    cls = f"{type(model).__module__}.{type(model).__name__}"
    ordinary = {
        "inertial_benchmark.models.tlio.model.TLIOResNet",
        "inertial_benchmark.models.llio.model.LLIONet",
        "inertial_benchmark.models.dive.model.DIVE",
        "inertial_benchmark.models.meio.model.MEIONet",
    }
    canonical = cls == "inertial_benchmark.models.eqnio.tlio.EqNIOTLIO"
    loss_name = getattr(model, "loss_name", None)
    kwargs = dict(getattr(model, "loss_kwargs", {}) or {})
    contract = {"schema": "ib.uncertainty.raw-task.v1", "supported": False,
                "head_trained": False, "reason": "unsupported_model_contract",
                "model_class": cls, "loss_name": loss_name, "evaluation_epoch": int(epoch),
                "frame": cfg.frame, "target": cfg.target, "layout": cfg.output_layout,
                "dims": int(cfg.dims),
                "unit": "m" if cfg.target in ("displacement", "multi_displacement") else "m/s",
                "overlap_policy": "raw_outputs_before_world_conversion_and_merge",
                "mean_key": "vel", "logstd_key": "logstd", "canonical": False,
                "logstd_offset": 0.0, "min_logstd": None, "max_logstd": None}
    hook = getattr(model, "raw_uncertainty_contract", None)
    if callable(hook):
        declared = dict(hook(epoch))
        for key in ("frame", "target", "layout", "unit", "mean_key", "logstd_key",
                    "training_start_epoch"):
            if key not in declared:
                raise ValueError(f"raw_uncertainty_contract is missing {key}")
        if (declared["target"] != cfg.target or declared["layout"] != cfg.output_layout
                or declared.get("canonical", False)):
            raise ValueError("custom raw uncertainty hook must match the task layout/target; "
                             "canonical transforms require an explicitly implemented adapter")
        if declared["frame"] != cfg.frame:
            raise ValueError("custom raw uncertainty hook must declare the same task frame")
        if declared["unit"] != contract["unit"]:
            raise ValueError("custom raw uncertainty hook has inconsistent task units")
        contract.update(declared, supported=True, contract_source="model_explicit_hook")
        start = declared["training_start_epoch"]
    elif cls in ordinary or canonical:
        accepted = {"gaussian_nll", "mse_then_nll", "nll_detach_then_nll"}
        if cls.endswith(".DIVE"):
            accepted.add("dive_mse_then_nll")
        if cls.endswith(".MEIONet"):
            accepted.add("meio_multiexit")
        if canonical:
            # out['logstd'] is only a world-frame marginal report. It cannot be
            # paired with a diagonal world Gaussian, even under a loss override.
            accepted = {"eqnio_canonical_nll"}
        if loss_name not in accepted:
            contract["reason"] = "unsupported_or_nonprobabilistic_scientific_loss"
            return contract
        contract.update(supported=True, contract_source="reviewed_network_and_loss")
        start = 0 if loss_name == "gaussian_nll" else kwargs.get(
            "switch_epoch", 9 if canonical else 10)
        if loss_name == "meio_multiexit":
            # At the first blend epoch alpha=1, so the sigma branch still has
            # zero NLL weight. The next epoch is the first trained sigma phase.
            start += int(int(kwargs.get("warmup_epochs", 0)) > 0)
        if loss_name == "dive_mse_then_nll":
            contract["logstd_offset"] = float(kwargs.get("offset", 1e-7))
        if loss_name in ("gaussian_nll", "mse_then_nll", "nll_detach_then_nll",
                         "eqnio_canonical_nll"):
            contract["min_logstd"] = kwargs.get("min_logstd", float(np.log(1e-3)))
            contract["max_logstd"] = kwargs.get(
                "max_logstd", float(np.log(1e3)) if loss_name in (
                    "gaussian_nll", "mse_then_nll") else None)
        elif loss_name == "dive_mse_then_nll":
            contract.update(min_logstd=kwargs.get("min_logstd"),
                            max_logstd=kwargs.get("max_logstd"))
        if canonical:
            contract.update(mean_key="disp_canonical", logstd_key="logstd_canonical",
                            canonical=True, frame=f"o2_canonical_per_window:{cfg.frame}")
    else:
        return contract
    if type(start) is not int or start < 0:
        raise ValueError("uncertainty training_start_epoch must be a nonnegative integer")
    contract["training_start_epoch"] = start
    trained = getattr(model, "_ipb_trained_epoch", None)
    updates = getattr(model, "_ipb_epoch_optimizer_updates", None)
    total = getattr(model, "_ipb_optimizer_updates", None)
    checkpoint = getattr(model, "_ipb_checkpoint_epoch", None)
    proof = dict(getattr(model, "_ipb_uncertainty_checkpoint_evidence", {}) or {})
    valid_updates = lambda count, all_count: (
        type(count) is int and count > 0 and type(all_count) is int and all_count >= count)
    if type(trained) is int and trained >= 0 and valid_updates(updates, total):
        evidence, source = trained, "completed_optimizer_epoch"
    elif (type(checkpoint) is int and checkpoint >= 0
          and type(proof.get("completed_epoch")) is int
          and proof["completed_epoch"] == checkpoint
          and valid_updates(proof.get("epoch_optimizer_updates"),
                            proof.get("total_optimizer_updates"))):
        evidence, source = checkpoint, "checkpoint_epoch"
        updates, total = proof["epoch_optimizer_updates"], proof["total_optimizer_updates"]
    else:
        evidence, source = None, "none"
    contract.update(training_evidence_epoch=evidence, training_evidence_source=source,
                    training_epoch_optimizer_updates=updates if evidence is not None else 0,
                    training_total_optimizer_updates=total if evidence is not None else 0)
    saved_loss = dict(getattr(model, "_ipb_uncertainty_checkpoint_loss", {}) or {})
    if source == "checkpoint_epoch":
        contract.update(checkpoint_loss_name=saved_loss.get("loss"),
                        checkpoint_loss_kwargs=saved_loss.get("loss_kwargs"))
        if saved_loss.get("loss") is None:
            contract["reason"] = "missing_checkpoint_scientific_loss_identity"
            return contract
        if (saved_loss["loss"] != loss_name
                or dict(saved_loss.get("loss_kwargs") or {}) != kwargs):
            # Loading weights does not retrain sigma under an evaluation override.
            # A genuine subsequent optimizer epoch uses live evidence above instead.
            contract["reason"] = "checkpoint_scientific_loss_contract_overridden"
            return contract
    if epoch < start:
        contract["reason"] = "uncertainty_head_before_training_phase"
    elif evidence is None:
        contract["reason"] = "missing_completed_optimizer_training_evidence"
    elif evidence < epoch or evidence < start:
        contract["reason"] = "training_evidence_does_not_cover_evaluation_phase"
    else:
        contract.update(head_trained=True, reason="available")
    return contract


def spec_to_view(spec: Any) -> ViewConfig:
    """模型输入规格 → 任务视图配置（只取视图字段）。"""
    return ViewConfig(**{k: getattr(spec, k) for k in ViewConfig.__dataclass_fields__})


def integrate(t: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """逐帧速度的梯形积分，``pos[0] = 0``。"""
    dt = np.diff(t)[:, None]
    steps = 0.5 * (vel[1:] + vel[:-1]) * dt
    return np.concatenate([np.zeros((1, vel.shape[1])), np.cumsum(steps, axis=0)], axis=0)


def fill_invalid_windows(t_window: np.ndarray, values: np.ndarray,
                         valid: np.ndarray) -> np.ndarray:
    """无效窗口的值用有效窗口按时间线性插值（两端常值外推）替换。"""
    if valid.all():
        return values
    out = values.copy()
    for c in range(values.shape[1]):
        out[~valid, c] = np.interp(t_window[~valid], t_window[valid], values[valid, c])
    return out


def reconstruct(t: np.ndarray, t_window: np.ndarray, vel_window: np.ndarray,
                anchor_index: int, anchor: np.ndarray) -> np.ndarray:
    """窗口速度 → 逐帧速度（分段线性 + 首尾常值外推）→ 梯形积分 → 锚定。"""
    vel = np.stack([np.interp(t, t_window, vel_window[:, c])
                    for c in range(vel_window.shape[1])], axis=1)
    pos = integrate(t, vel)
    return pos + (anchor[: pos.shape[1]] - pos[anchor_index])


def merge_outputs(times: np.ndarray, values: np.ndarray, valid: np.ndarray,
                  distance: np.ndarray, overlap: str = "mean",
                  resolution: float = 0.0) -> tuple:
    """把（可能重叠的）逐输出预测合并到唯一时间轴上，返回 ``(t, values, valid)``。

    逐帧/多步目标在滑窗下会让同一时刻出现多个预测（``eval_stride < window`` 时必然重叠）。
    合并策略：

    * ``mean``（**默认**）：同一时刻的全部**有效**输出取算术平均。默认取平均是因为各重叠预测来自
      同一模型、同一时刻、不同上下文窗口，误差近似独立同分布，平均能降低方差且不引入偏置；
      而且它对 ``eval_stride`` 的取值最不敏感（换步长时结果稳定），便于跨模型比较。
    * ``center``：只保留“窗口中心距该时刻最近”的那个输出，即最少依赖窗口边界外推的那个；
      带因果/单向结构（如流式 LSTM）或边界效应明显的模型适合用它。

    两种策略都优先使用有效输出；某一时刻的全部输出都无效时，退化为对全部输出合并并标记为无效
    （随后由 :func:`fill_invalid_windows` 按相邻有效时刻插值替换）。

    时间偏移一律是半个采样间隔的整数倍（见 ``ViewConfig.output_offsets``），因此按半采样分辨率
    分组是精确的。
    """
    if overlap not in ("mean", "center"):
        raise ValueError(f"unknown overlap strategy {overlap!r}; expected 'mean' or 'center'")
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(len(times), -1)
    valid = np.asarray(valid, bool).reshape(-1)
    distance = np.asarray(distance, dtype=np.float64).reshape(-1)
    if len(times) == 0:
        return times, values, valid
    step = float(resolution) if resolution > 0 else 1e-9
    key = np.round(times / step).astype(np.int64)
    uniq, inverse = np.unique(key, return_inverse=True)
    groups = np.arange(len(uniq))
    t_out = uniq * step
    any_valid = np.zeros(len(uniq), bool)
    np.logical_or.at(any_valid, inverse, valid)
    use = valid | ~any_valid[inverse]  # 该时刻没有任何有效输出时退化为使用全部输出
    if overlap == "mean":
        counts = np.bincount(inverse[use], minlength=len(uniq)).astype(np.float64)
        merged = np.stack([np.bincount(inverse[use], weights=values[use, c], minlength=len(uniq))
                           / np.maximum(counts, 1.0) for c in range(values.shape[1])], axis=1)
    else:
        rows = np.flatnonzero(use)
        rows = rows[np.lexsort((distance[rows], inverse[rows]))]
        merged = values[rows[np.searchsorted(inverse[rows], groups)]]
    return t_out, merged, any_valid


class Predictor:
    """把模型应用到整条序列，输出 :class:`SequenceResult`。"""

    def __init__(self, cfg: Any = None, model: Optional[torch.nn.Module] = None,
                 device: Optional[torch.device] = None, callbacks: Optional[dict] = None,
                 overrides: Optional[dict] = None) -> None:
        self.args = cfg if cfg is not None else get_cfg(overrides)
        self.device = device or select_device(self.args.device, verbose=False)
        self.model = model if model is not None else load_model(self.args, self.device)
        self.model.to(self.device)
        spec = getattr(self.model, "input_spec", None)
        self.view_cfg = spec_to_view(spec) if spec is not None else ViewConfig.from_cfg(self.args)
        self.callbacks = callbacks or default_callbacks()
        self.amp = bool(self.args.amp) and self.device.type == "cuda"

    @torch.inference_mode()
    def infer(self, view: SequenceView, collect_loss: bool = False, epoch: int = 0) -> dict:
        """返回 ``starts``、``window_valid``、视图坐标系下的 ``vel``/``logstd``、其他逐窗口输出
        ``outputs`` 与损失累计。

        ``outputs`` 收集模型返回的其余张量中首维等于批大小的项（逐窗口输出，例如速度大小、
        协方差）；模型可用 ``saved_outputs`` 属性（键的元组）限定范围，缺省为全部。
        """
        model = self.model
        was_training = model.training
        model.eval()
        starts = view.starts(int(self.args.eval_stride), require_valid=False)
        valid_input = view.window_valid_input(starts)
        valid_target = view.window_valid_target(starts)
        valid = valid_input & valid_target
        keep = getattr(model, "saved_outputs", None)
        needs_extra = bool(self.view_cfg.extra_inputs)
        outs: dict = {}
        extras: dict = {}
        contract = uncertainty_contract(model, self.view_cfg, epoch)
        raw: dict = {}
        loss_sum, loss_n = 0.0, 0
        batch = int(self.args.val_batch)
        for i in range(0, len(starts), batch):
            s = starts[i:i + batch]
            x = torch.from_numpy(view.imu_windows(s)).to(self.device, non_blocking=True)
            extra = {k: torch.from_numpy(v).to(self.device)
                     for k, v in view.extra_inputs(s).items()} if needs_extra else {}
            with torch.autocast(self.device.type, enabled=self.amp):
                out = model(x, extra) if needs_extra else model(x)
            out = {k: v.float() for k, v in out.items() if torch.is_tensor(v)}
            if contract["supported"]:
                mean_key, sigma_key = contract["mean_key"], contract["logstd_key"]
                if mean_key not in out or sigma_key not in out:
                    contract.update(head_trained=False, reason="missing_model_gaussian_output")
                else:
                    mean, sigma = out[mean_key].cpu().numpy(), out[sigma_key].cpu().numpy()
                    target = view.targets(s)
                    if contract["canonical"]:
                        if "frame_columns" not in out:
                            raise ValueError("EqNIO canonical Gaussian needs frame_columns")
                        basis = out["frame_columns"].cpu().numpy()
                        target = np.concatenate([
                            np.einsum("bji,bj->bi", basis, target[:, :2]), target[:, 2:]], -1)
                        raw.setdefault("basis", []).append(basis)
                    expected = (len(s),) + self.view_cfg.output_shape
                    if mean.shape != expected or target.shape != expected:
                        raise ValueError("raw uncertainty mean/target must match InputSpec output_shape")
                    if sigma.shape not in (expected, expected[:-1] + (1,)):
                        raise ValueError("raw uncertainty logstd must match mean layout or be isotropic")
                    mask = view.target_mask(s) & valid_input[i:i + batch, None]
                    raw.setdefault("mean", []).append(mean)
                    raw.setdefault("target", []).append(target)
                    raw.setdefault("logstd", []).append(sigma)
                    raw.setdefault("valid", []).append(mask)
            for k, v in out.items():
                if k in ("vel", "logstd"):
                    outs.setdefault(k, []).append(v.cpu().numpy())
                elif v.ndim >= 1 and v.shape[0] == len(s) and (keep is None or k in keep):
                    extras.setdefault(k, []).append(v.cpu().numpy())
            v = valid[i:i + batch]
            if collect_loss and v.any():
                idx = torch.from_numpy(np.flatnonzero(v)).to(self.device)
                target = torch.from_numpy(view.targets(s[v])).to(self.device)
                mask = torch.from_numpy(view.target_mask(s[v])).to(self.device)
                sub = {k: t.index_select(0, idx) for k, t in out.items()
                       if t.ndim >= 1 and t.shape[0] == len(s)}
                # batch 的键与 Trainer 完全一致（``target``/``imu``/``mask``，见
                # nn.base.LOSS_BATCH_KEYS）：否则用到 ``batch["imu"]`` 或 ``batch["mask"]``
                # 的损失只能在训练中工作，验证时会 KeyError
                loss, _ = model.loss(sub, {"target": target, "imu": x.index_select(0, idx),
                                           "mask": mask}, epoch)
                loss_sum += float(loss) * int(v.sum())
                loss_n += int(v.sum())
        model.train(was_training)
        result = {"starts": starts, "window_valid": valid, "window_valid_input": valid_input,
                  "window_valid_target": valid_target, "loss_sum": loss_sum, "loss_n": loss_n}
        empty = (0,) + self.view_cfg.output_shape[:-1] + (self.view_cfg.dims,)
        for k in ("vel", "logstd"):
            result[k] = np.concatenate(outs[k]) if k in outs else (
                np.zeros(empty) if k == "vel" else None)
        result["outputs"] = {k: np.concatenate(v) for k, v in extras.items()}
        result["uncertainty"] = contract
        result["uncertainty_raw"] = {k: np.concatenate(v) for k, v in raw.items()}
        return result

    def infer_sequence(self, view: SequenceView) -> dict:
        """序列级模型（:class:`~inertial_benchmark.nn.base.SequenceModel`）的推理入口。

        模型返回的时间戳必须落在同一窗口网格上，才能与逐窗口模型共用轨迹重建与指标。
        """
        starts = view.starts(int(self.args.eval_stride), require_valid=False)
        grid = view.target_times(starts)
        result = self.model.predict_sequence(view.seq, view, starts)
        times, velocities = result[0], result[1]
        outputs = dict(result[2]) if len(result) > 2 and result[2] else {}
        times = np.asarray(times, dtype=np.float64)
        expected = np.asarray(grid, dtype=np.float64).reshape(-1)
        if times.shape != expected.shape or not np.allclose(times, expected, atol=1e-6):
            raise ValueError(
                f"{type(self.model).__name__}.predict_sequence returned {len(times)} timestamps "
                f"that do not match the window grid ({len(expected)} expected): a SequenceModel "
                "must report velocities on view.target_times(view.starts(eval_stride))")
        return {"starts": starts, "window_valid": view.window_valid(starts),
                "window_valid_input": view.window_valid_input(starts),
                "window_valid_target": view.window_valid_target(starts),
                "vel": np.asarray(velocities, dtype=np.float64), "logstd": None,
                "outputs": outputs, "loss_sum": 0.0, "loss_n": 0}

    def predict_view(self, view: SequenceView, collect_loss: bool = False,
                     epoch: Optional[int] = None) -> SequenceResult:
        if epoch is None:
            epoch = getattr(self.model, "_ipb_checkpoint_epoch", None)
            epoch = int(epoch) if type(epoch) is int else 0
        seq = view.seq
        res = SequenceResult(sequence_id=seq.sequence_id, dataset=seq.dataset,
                             group_id=seq.group_id, frame=self.view_cfg.frame,
                             rate=self.view_cfg.rate)
        res.t = seq.timestamp
        res.pos_gt = seq.position
        res.valid_pose = seq.valid_pose
        cfg = self.view_cfg
        if len(seq) < cfg.input_span:
            res.skipped = f"shorter than one input span ({len(seq)} < {cfg.input_span})"
            return res
        sequence_model = isinstance(self.model, SequenceModel)
        out = (self.infer_sequence(view) if sequence_model
               else self.infer(view, collect_loss, epoch))
        res.uncertainty = out.get("uncertainty", {
            "schema": "ib.uncertainty.raw-task.v1", "supported": False,
            "head_trained": False, "reason": "sequence_model_without_gaussian_contract"})
        for key, values in out.get("uncertainty_raw", {}).items():
            setattr(res, f"uncertainty_{key}_raw", values)
        starts = out["starts"]
        w_input, w_target = out["window_valid_input"], out["window_valid_target"]
        anchors = np.flatnonzero(seq.valid_pose)
        if not w_input.any() or not w_target.any() or len(anchors) == 0:
            res.skipped = "no valid input window, no valid target window or no reference pose"
            return res
        # 每个输出映射到自己的时间戳，重叠预测按 overlap 策略合并（DESIGN 第 5 节）
        times = view.output_times(starts)
        centers = seq.timestamp[starts] + 0.5 * (cfg.window - 1) * cfg.dt
        distance = np.abs(times - centers[:, None])
        rows = times.shape[1]
        vel_rows = view.to_world_velocity(out["vel"], starts).reshape(len(starts), rows, -1)
        tgt_rows = view.to_world_velocity(view.targets(starts), starts).reshape(
            len(starts), rows, -1)
        row_input = np.repeat(w_input[:, None], rows, axis=1)
        row_target = view.target_mask(starts)
        merge = dict(overlap=cfg.overlap, resolution=0.5 * cfg.dt)
        t_window, vel, valid_pred = merge_outputs(times, vel_rows, row_input, distance, **merge)
        _, tgt, valid_tgt = merge_outputs(times, tgt_rows, row_target, distance, **merge)
        # 预测只在**输入**无效处填补，目标只在**位姿**无效处填补：位姿有缺口而 IMU 正常时，
        # 模型仍然自己穿越缺口，其漂移会体现在轨迹指标里（旧实现把这些预测也插值掉了）
        vel = fill_invalid_windows(t_window, vel, valid_pred)
        tgt = fill_invalid_windows(t_window, tgt, valid_tgt)
        i0 = int(anchors[0])
        res.pos_pred = reconstruct(seq.timestamp, t_window, vel, i0, seq.position[i0])
        res.pos_oracle = reconstruct(seq.timestamp, t_window, tgt, i0, seq.position[i0])
        res.t_window, res.starts = t_window, starts
        res.window_valid = valid_pred & valid_tgt
        res.window_valid_input, res.window_valid_target = valid_pred, valid_tgt
        res.vel_pred, res.vel_target, res.logstd = vel, tgt, out["logstd"]
        res.outputs = out["outputs"]
        if collect_loss and out["loss_n"]:
            res.loss = out["loss_sum"] / out["loss_n"]
        return res

    def predict_sequence(self, seq: Union[Sequence, PathLike], collect_loss: bool = False,
                         epoch: Optional[int] = None) -> SequenceResult:
        if not isinstance(seq, Sequence):
            seq = load_sequence(seq)
        return self.predict_view(SequenceView(seq, self.view_cfg), collect_loss, epoch)

    def __call__(self, source: Union[PathLike, Sequence, Iterable]) -> Any:
        """``source`` 为单个 h5/``Sequence`` 时返回 :class:`Trajectory`，目录或列表时返回列表。"""
        run_callbacks(self.callbacks, "on_predict_start", self)
        if isinstance(source, Sequence):
            items, single = [source], True
        elif isinstance(source, (str, Path)):
            path = Path(source)
            single = not path.is_dir()
            items = [path] if single else (sorted(path.glob("*.h5"))
                                            or sorted((path / "sequences").glob("*.h5")))
        else:
            items, single = list(source), False
        self.results = [self.predict_sequence(s) for s in items]
        run_callbacks(self.callbacks, "on_predict_end", self)
        trajs = [Trajectory(r.t, r.pos_pred, sequence_id=r.sequence_id) for r in self.results]
        return trajs[0] if single else trajs
