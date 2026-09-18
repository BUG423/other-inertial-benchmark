"""Train/eval 模式守卫：双侧报告量、dropout 探针与状态隔离。"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from inertial_benchmark.engine.mode_guard import (  # noqa: E402
    DROP_PROBE_WARN,
    GUARD_MODES,
    attribute_channels,
    gap_verdict,
    mean_shift_risk,
    mode_context,
    symmetric_gap,
    two_sided_ratio,
)


def test_loss_report_is_two_sided_and_signed_safe():
    assert two_sided_ratio(0.8, 0.2) == pytest.approx(4.0)
    assert two_sided_ratio(0.2, 0.8) == pytest.approx(4.0)
    assert two_sided_ratio(0.0, 1.0) == math.inf
    assert math.isnan(two_sided_ratio(-1.0, 1.0))
    assert symmetric_gap(-1.0, 1.0) == pytest.approx(1.0)
    assert symmetric_gap(-2.0, -2.0) == pytest.approx(0.0)


def test_dropout_probe_thresholds_match_the_reproduced_runs():
    # (speed_ratio eval, speed_ratio MC-dropout, loss eval, loss train)
    healthy = [
        (0.887, 0.893, 0.6900, 0.7527),
        (0.909, 0.918, 0.0335, 0.2193),  # 健康但双侧损失比为 6.6
    ]
    collapsing = [
        (0.832, 0.856, 0.8116, 0.6698),
        (0.477, 0.617, 0.6229, 0.5658),  # 塌陷但双侧损失比仅 1.10
    ]
    for sr, sr_drop, loss_eval, loss_train in healthy:
        verdict = gap_verdict(sr, sr_drop, loss_eval, loss_train)
        assert verdict["level"] == "ok"
        assert verdict["dropout_probe_delta"] <= DROP_PROBE_WARN
    for sr, sr_drop, loss_eval, loss_train in collapsing:
        verdict = gap_verdict(sr, sr_drop, loss_eval, loss_train)
        assert verdict["level"] in ("warning", "error")
        assert verdict["dropout_probe_delta"] > DROP_PROBE_WARN


class _Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.net = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(16, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(self.bn(x))


def test_mode_context_restores_flags_and_batchnorm_state():
    model = _Head()
    model.train()
    model.bn.eval()  # 刻意制造混合状态，要求逐模块原样恢复
    flags = {name: module.training for name, module in model.named_modules()}
    state = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.randn(16, 8)
    with torch.no_grad():
        for mode in GUARD_MODES:
            with mode_context(model, mode):
                model(x)
    assert {name: module.training for name, module in model.named_modules()} == flags
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())


def test_risk_and_channel_attribution_identify_dropout_path():
    risk = mean_shift_risk(_Head())
    assert risk["net.2"]["mean_shift_risk"] is True
    assert risk["net.5"]["mean_shift_risk"] is False
    assert risk["net.5"]["reason"] == "only affine operations follow"

    # dropout_only≈eval、bn_only≈train：失配由 BN 通道主导。
    attr = attribute_channels(0.7872, 0.5578, 0.7899, 0.5529)
    assert attr["dominant"] == "batchnorm"
    assert attr["share_batchnorm"] > 0.9
    assert abs(attr["share_dropout"]) < 0.1
    assert attribute_channels(0.5, 0.5, 0.5, 0.5)["degenerate"] is True


def test_functional_attention_dropout_is_switched_and_reported():
    model = nn.MultiheadAttention(8, 2, dropout=0.25, batch_first=True)
    model.eval()
    risk = mean_shift_risk(model)
    assert risk[""]["mean_shift_risk"] is True
    assert risk[""]["p"] == pytest.approx(0.25)
    with mode_context(model, GUARD_MODES[1]):
        assert model.training is True
    assert model.training is False
