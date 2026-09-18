"""训练/推理模式失配守卫的纯函数与模块状态切换。

损失的 train/eval 比值只作为双侧报告量：验证集上的 BatchNorm batch 统计会让健康模型也出现
很大的损失比，不能拿它直接门禁。主探针比较常规 ``eval`` 与只打开 dropout 后做 MC 平均的
速度幅值；后者直接测量训练期 dropout 期望是否改变了推理期输出。
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

DROP_PROBE_WARN = 0.02
DROP_PROBE_ERROR = 0.05
SPEED_RATIO_WARN_LOW = 0.78
SPEED_RATIO_ERROR_LOW = 0.65
SPEED_RATIO_WARN_HIGH = 1.15
MC_SAMPLES = 8


def two_sided_ratio(loss_eval: float, loss_train: float) -> float:
    """返回 ``max(r, 1/r)``；异号或非有限损失没有有意义的比值，返回 NaN。"""
    a, b = float(loss_eval), float(loss_train)
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    if a == 0.0 or b == 0.0:
        return float("inf") if a != b else 1.0
    ratio = a / b
    return max(ratio, 1.0 / ratio) if ratio > 0.0 else float("nan")


def symmetric_gap(loss_eval: float, loss_train: float) -> float:
    """``|a-b| / (|a|+|b|)``，对可取负值、可跨零的损失安全。"""
    a, b = float(loss_eval), float(loss_train)
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    scale = abs(a) + abs(b)
    return abs(a - b) / scale if scale > 0.0 else 0.0


def gap_verdict(speed_ratio_eval: float, speed_ratio_dropout: float,
                loss_eval: float | None = None, loss_train: float | None = None) -> dict:
    """根据 MC-dropout 幅值恢复量与 eval 幅值给出 ``ok/warning/error``。

    val 损失的双侧比与有符号差值会写入 ``loss_report``，但实测表明它们不能区分健康模型与
    AMP 幅值塌陷，故不参与级别判定。
    """
    sr_eval, sr_dropout = float(speed_ratio_eval), float(speed_ratio_dropout)
    delta = sr_dropout - sr_eval
    level, reasons = "ok", []

    def escalate(target: str) -> None:
        nonlocal level
        if {"ok": 0, "warning": 1, "error": 2}[target] > \
                {"ok": 0, "warning": 1, "error": 2}[level]:
            level = target

    if math.isfinite(delta):
        if delta > DROP_PROBE_ERROR:
            escalate("error")
            reasons.append(f"dropout probe delta={delta:+.3f} > {DROP_PROBE_ERROR:.2f}")
        elif delta > DROP_PROBE_WARN:
            escalate("warning")
            reasons.append(f"dropout probe delta={delta:+.3f} > {DROP_PROBE_WARN:.2f}")
    if math.isfinite(sr_eval):
        if sr_eval < SPEED_RATIO_ERROR_LOW:
            escalate("error")
            reasons.append(
                f"speed_ratio(eval)={sr_eval:.3f} < {SPEED_RATIO_ERROR_LOW:.2f}")
        elif sr_eval < SPEED_RATIO_WARN_LOW:
            escalate("warning")
            reasons.append(
                f"speed_ratio(eval)={sr_eval:.3f} < {SPEED_RATIO_WARN_LOW:.2f}")
        elif sr_eval > SPEED_RATIO_WARN_HIGH:
            escalate("warning")
            reasons.append(
                f"speed_ratio(eval)={sr_eval:.3f} > {SPEED_RATIO_WARN_HIGH:.2f}")
    out = {
        "level": level,
        "reasons": reasons,
        "dropout_probe_delta": delta,
        "speed_ratio_eval": sr_eval,
        "speed_ratio_dropout": sr_dropout,
        "thresholds": {
            "dropout_probe_warning": DROP_PROBE_WARN,
            "dropout_probe_error": DROP_PROBE_ERROR,
            "speed_ratio_warning_low": SPEED_RATIO_WARN_LOW,
            "speed_ratio_error_low": SPEED_RATIO_ERROR_LOW,
            "speed_ratio_warning_high": SPEED_RATIO_WARN_HIGH,
        },
    }
    if loss_eval is not None and loss_train is not None:
        a, b = float(loss_eval), float(loss_train)
        out["loss_report"] = {
            "loss_eval": a,
            "loss_train": b,
            "delta": a - b,
            "ratio": a / b if b > 0.0 else None,
            "two_sided_ratio": two_sided_ratio(a, b),
            "symmetric_gap": symmetric_gap(a, b),
            "gates_verdict": False,
        }
    return out


def attribute_channels(loss_eval: float, loss_train: float, loss_dropout_only: float,
                       loss_bn_only: float) -> dict:
    """以四种模式的损失把总失配归因到 dropout 与 BatchNorm 通道。"""
    total = float(loss_eval) - float(loss_train)
    if not math.isfinite(total) or abs(total) < 1e-12:
        return {"share_dropout": None, "share_batchnorm": None, "dominant": None,
                "total": total, "degenerate": True}
    share_dropout = (float(loss_eval) - float(loss_dropout_only)) / total
    share_batchnorm = (float(loss_eval) - float(loss_bn_only)) / total
    return {
        "share_dropout": share_dropout,
        "share_batchnorm": share_batchnorm,
        "dominant": "dropout" if share_dropout > share_batchnorm else "batchnorm",
        "total": total,
        "degenerate": False,
    }


@dataclass(frozen=True)
class GuardMode:
    """需要独立控制的两个训练态通道。"""

    name: str
    dropout: bool = False
    batchnorm: bool = False
    mc_samples: int = 1


GUARD_MODES = (
    GuardMode("eval"),
    GuardMode("dropout_only", dropout=True, mc_samples=MC_SAMPLES),
    GuardMode("bn_only", batchnorm=True),
    GuardMode("train", dropout=True, batchnorm=True, mc_samples=MC_SAMPLES),
)


def dropout_modules(model: nn.Module) -> list:
    """按注册顺序返回全部 dropout 模块。"""
    kind = nn.modules.dropout._DropoutNd
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, kind)]


def functional_dropout_modules(model: nn.Module) -> list:
    """返回内部用函数式 dropout、但没有 ``nn.Dropout`` 子模块的标准模块。"""
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention) and float(module.dropout) > 0.0:
            modules.append((name, module, float(module.dropout)))
        elif (isinstance(module, nn.modules.rnn.RNNBase)
              and float(module.dropout) > 0.0 and int(module.num_layers) > 1):
            modules.append((name, module, float(module.dropout)))
    return modules


def batchnorm_modules(model: nn.Module) -> list:
    """返回全部 BatchNorm 模块（包括其维度变体）。"""
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, _BatchNorm)]


@contextlib.contextmanager
def mode_context(model: nn.Module, mode: GuardMode):
    """临时切换 dropout/BN，退出时复原逐模块标志和 BN running 状态。"""
    flags = {id(module): module.training for module in model.modules()}
    bns = batchnorm_modules(model)
    bn_state = [
        (module, module.momentum,
         None if module.running_mean is None else module.running_mean.clone(),
         None if module.running_var is None else module.running_var.clone(),
         None if module.num_batches_tracked is None else module.num_batches_tracked.clone())
        for _, module in bns
    ]
    try:
        model.eval()
        for _, module in dropout_modules(model):
            module.train(bool(mode.dropout))
        for _, module, _ in functional_dropout_modules(model):
            module.train(bool(mode.dropout))
        for _, module in bns:
            module.momentum = 0.0
            module.train(bool(mode.batchnorm))
        yield model
    finally:
        for module, momentum, mean, variance, tracked in bn_state:
            module.momentum = momentum
            if mean is not None:
                module.running_mean.copy_(mean)
            if variance is not None:
                module.running_var.copy_(variance)
            if tracked is not None:
                module.num_batches_tracked.copy_(tracked)
        for module in model.modules():
            module.train(flags[id(module)])


def mean_shift_risk(model: nn.Module) -> dict:
    """保守标注每个 dropout 后是否还有非线性，因此可能改变输出均值。

    PyTorch 模块注册顺序并不等于任意网络的完整计算图；这里故意取保守近似。只有 dropout 概率
    为零，或其后直到输出只剩仿射/线性操作时才豁免。
    """
    affine = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.Identity, nn.Flatten, nn.Dropout, nn.Dropout1d, nn.Dropout2d,
        nn.Dropout3d, _BatchNorm, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    )
    leaves = [(name, module) for name, module in model.named_modules()
              if not list(module.children())]
    report = {}
    for index, (name, module) in enumerate(leaves):
        if not isinstance(module, nn.modules.dropout._DropoutNd):
            continue
        probability = float(getattr(module, "p", 0.0))
        risky = probability > 0.0 and any(
            not isinstance(tail, affine) for _, tail in leaves[index + 1:])
        reason = ("dropout probability is zero" if probability <= 0.0 else
                  "a nonlinearity follows" if risky else "only affine operations follow")
        report[name] = {"p": probability, "mean_shift_risk": risky, "reason": reason}
    # MultiheadAttention 与多层 RNN 的 dropout 是函数式实现，不会作为叶子模块出现；其输出仍会
    # 进入注意力归一化/循环门，保守地视为均值位移风险。
    for name, _, probability in functional_dropout_modules(model):
        report[name] = {"p": probability, "mean_shift_risk": True,
                        "reason": "functional dropout inside a nonlinear module"}
    return report


__all__ = [
    "DROP_PROBE_ERROR", "DROP_PROBE_WARN", "GUARD_MODES", "MC_SAMPLES",
    "GuardMode", "attribute_channels", "batchnorm_modules", "dropout_modules",
    "functional_dropout_modules", "gap_verdict", "mean_shift_risk", "mode_context", "symmetric_gap",
    "two_sided_ratio",
]
