"""Server-only MASF/FACM scientific tests, written but NOT executed.

Parent conftest blocks all collection until the original-task tail handoff.
Reduced channels/depths below are diagnostics, NOT full-profile capacity GO.
No synthetic skeleton, action/boundary labels, text, graph, BERT or ASRF.
"""

import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from inertial_benchmark.cfg import INPUT_KEYS, get_cfg
from inertial_benchmark.nn.base import BaseModel, InputSpec
from inertial_benchmark.nn.registry import MODELS


HERE = Path(__file__).parent
BEFORE_REGISTRY = dict(MODELS)
SPEC = importlib.util.spec_from_file_location("cvpr2026_staging_specscalpel", HERE / "model.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
AFTER_IMPORT_REGISTRY = dict(MODELS)


def _small(spec=None, **overrides):
    args = dict(channels=8, layers=2, heads=2, dropout=0.0, attention_dropout=0.0)
    args.update(overrides)
    return module.SpectralScalpelVelocity(spec or InputSpec(dims=3), **args)


def test_source_profile_schema_defaults_and_isolated_registry():
    assert AFTER_IMPORT_REGISTRY == BEFORE_REGISTRY
    profile = yaml.safe_load((HERE / "model_3d.yaml").read_text())
    provenance = json.loads((HERE / "provenance.json").read_text())
    assert profile["arch"] == "specscalpel_spectral_core_velocity"
    assert profile["args"]["channels"] == 64 and profile["args"]["layers"] == 10
    defaults = inspect.signature(module.SpectralScalpelVelocity).parameters
    assert defaults["channels"].default == 64 and defaults["layers"].default == 10
    assert defaults["heads"].default == 4 and defaults["dropout"].default == 0.5
    assert defaults["attention_dropout"].default == 0.05
    assert set(profile["input"]) <= set(INPUT_KEYS)
    assert "channels" not in profile["input"]
    assert tuple(profile["input_channels"]) == module.CHANNELS
    cfg = get_cfg({"model": profile})
    assert cfg.window == 200 and cfg.rate == 200 and cfg.dims == 3
    assert cfg.frame == "gravity_world" and cfg.orientation == "reference"
    assert cfg.remove_gravity is False and cfg.extra_inputs == []
    assert cfg.history == cfg.history_stride == cfg.output_steps == 0
    assert profile["loss"] == "mse" and profile["causality"] == "window_noncausal"
    assert provenance["implementation_status"] == "static_draft"
    for key in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled", "registered"):
        assert provenance[key] is False and profile[key] is False
    assert provenance["repository"]["commit"] == module.SpectralScalpelVelocity.source_commit


@pytest.mark.parametrize("fields", [
    {"dims": 2}, {"window": 64}, {"window": 199}, {"window": 201},
    {"rate": 100}, {"rate": float("nan")}, {"rate": float("inf")},
    {"frame": "body"}, {"frame": "gravity_yaw_local"},
    {"orientation": "device"}, {"remove_gravity": True},
    {"target": "displacement"}, {"target": "velocity_at_end"},
    {"target": "frame_velocity"}, {"output_steps": 1}, {"overlap": "center"},
    {"history": 1}, {"history": 2, "history_stride": 200}, {"history_stride": 1},
    {"extra_inputs": ("orientation",)}, {"extra_inputs": ("gravity",)},
    {"extra_inputs": ("init_velocity",)},
    {"channels": ("gyro_y", "gyro_x", "gyro_z", "acc_x", "acc_y", "acc_z")},
    {"channels": ("acc_x", "gyro_y", "gyro_z", "gyro_x", "acc_y", "acc_z")},
    {"channels": ("gyro_x",) * 6},
])
def test_reject_hidden_input_axis_layout_rate_and_target_expansion(fields):
    with pytest.raises(ValueError):
        _small(InputSpec(**dict({"dims": 3}, **fields)))


@pytest.mark.parametrize("args", [
    {"pretrained": True}, {"channels": 0}, {"channels": True},
    {"layers": 0}, {"heads": 0}, {"channels": 7, "heads": 2},
    {"dropout": float("nan")}, {"attention_dropout": float("nan")},
    {"dropout": float("inf")}, {"attention_dropout": float("inf")},
    {"dropout": 1.0}, {"attention_dropout": -0.1},
])
def test_reject_pretraining_nonfinite_dropout_and_inconsistent_hyperparameters(args):
    with pytest.raises(ValueError):
        _small(**args)


@pytest.mark.parametrize("length", [199, 200, 201])
def test_MASF_all_ones_filters_start_as_identity_and_preserve_exact_time_length(length):
    filter_ = module.MultiScaleAdaptiveSpectralFilter(8).double()
    features = torch.randn(2, 8, length, 1, dtype=torch.float64)
    output = filter_(features)
    assert output.shape == features.shape
    torch.testing.assert_close(output, features, rtol=1e-10, atol=1e-10)
    assert [value.shape[2] for value in filter_.filters] == [40, 48, 56, 64]
    assert filter_.static_logits.shape == (4, 1, 8, 1, 1)
    assert filter_.dynamic_logits.pool.output_size == (64, 1)
    assert all(value.shape[-1] == 1 for value in filter_.filters)
    with pytest.raises(ValueError):
        filter_(features.expand(-1, -1, -1, 17))


def test_dynamic_selector_is_source_pool_linear_permute_pipeline():
    selector = module.DynamicSpectralWeights().double()
    features = torch.randn(2, 8, 200, 1, dtype=torch.float64)
    expected = selector.spatial_linear(F.adaptive_avg_pool2d(features, (64, 1)))
    expected = selector.temporal_linear(expected.transpose(2, 3)).transpose(2, 3)
    expected = selector.filter_linear(expected.flatten(2)).permute(2, 0, 1)
    assert expected.shape == (4, 2, 8)
    torch.testing.assert_close(selector(features), expected)


def test_MASF_exact_fft_nearest_filters_dynamic_static_and_residual_fusion():
    torch.manual_seed(17)
    filter_ = module.MultiScaleAdaptiveSpectralFilter(8).double()
    features = torch.randn(2, 8, 200, 1, dtype=torch.float64, requires_grad=True)
    # Identical source all-ones filters produce zero selector gradient at init.
    # Deliberately different filters make the selector mechanism observable.
    with torch.no_grad():
        for index, weights in enumerate(filter_.filters):
            weights.copy_(torch.linspace(0.2 + index, 1.0 + index, weights.shape[2]).view(1, 1, -1, 1))
        filter_.static_logits.copy_(torch.linspace(-1, 1, 32).reshape(4, 1, 8, 1, 1))
    spectrum = torch.fft.rfft(features, dim=2)
    outputs = torch.stack([
        torch.fft.irfft(spectrum * F.interpolate(weights, size=(101, 1), mode="nearest"),
                        n=200, dim=2) for weights in filter_.filters
    ])
    dynamic, static = filter_.weighting(features)
    assert dynamic.shape == (4, 2, 8, 1, 1)
    torch.testing.assert_close(dynamic.sum(0), torch.ones(2, 8, 1, 1, dtype=torch.float64))
    torch.testing.assert_close(static.sum(0), torch.ones(1, 8, 1, 1, dtype=torch.float64))
    expected = (((outputs * dynamic).sum(0) + (outputs * static).sum(0)) / 2 + features) / 2
    actual = filter_(features)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    for parameter in (filter_.static_logits, *filter_.filters,
                      filter_.dynamic_logits.spatial_linear.weight,
                      filter_.dynamic_logits.temporal_linear.weight,
                      filter_.dynamic_logits.filter_linear.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert features.grad.abs().sum() > 0


@pytest.mark.parametrize("length", [199, 200, 201])
def test_FACM_real_imag_share_batch_stacked_pointwise_transforms_and_nT(length):
    mixer = module.FrequencyAwareChannelMixer(4).double()
    features = torch.randn(2, 4, length, dtype=torch.float64)
    spectrum = torch.fft.rfft(features, dim=2)
    real = mixer.second(mixer.first(spectrum.real))
    imag = mixer.second(mixer.first(spectrum.imag))
    expected = torch.fft.irfft(torch.complex(real, imag), n=length, dim=2) + features
    actual = mixer(features)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == features.shape
    assert not any(isinstance(child, (nn.ReLU, nn.BatchNorm1d)) for child in mixer.modules())
    with torch.no_grad():
        for layer in (mixer.first, mixer.second):
            layer.weight.copy_(torch.eye(4).unsqueeze(-1))
            layer.bias.zero_()
    torch.testing.assert_close(mixer(features), 2 * features, rtol=1e-10, atol=1e-10)


def test_source_sigmoid_linear_attention_matches_explicit_positive_kernel():
    attention = module.SigmoidLinearAttention(8, heads=2, dropout=0.0).double().eval()
    tokens = torch.randn(2, 7, 8, dtype=torch.float64)
    def split(values):
        return values.view(2, 7, 2, 4).transpose(1, 2)
    query, key = split(attention.query(tokens)).sigmoid(), split(attention.key(tokens)).sigmoid()
    value = split(attention.value(tokens))
    kernel = torch.einsum("bhld,bhsd->bhls", query, key)
    denominator = kernel.sum(-1) + 1e-6 * query.sum(-1)
    expected = torch.einsum("bhls,bhse->bhle", kernel, value) / denominator.unsqueeze(-1)
    expected = attention.output(expected.transpose(1, 2).reshape(2, 7, 8))
    torch.testing.assert_close(attention(tokens), expected)
    # No softmax over a T-by-T logits matrix, ELU map or attention mask labels.
    assert isinstance(attention.dropout, nn.Dropout)


def test_temporal_block_preserves_source_conv_attention_residual_then_FACM():
    block = module.SpectralTemporalBlock(8, 2, 2, 0.0, 0.0).eval()
    features = torch.randn(2, 8, 200)
    values = block.feed_forward(features)
    normalized = block.instance_norm(values).transpose(1, 2)
    values = values + block.attention(normalized).transpose(1, 2)
    expected = block.spectral_mixer(features + block.output(values))
    torch.testing.assert_close(block(features), expected)
    assert block.instance_norm.track_running_stats is False
    assert block.feed_forward[0].dilation == (2,)


def test_XYZ_mse_and_gradients_reach_Z_sensor_axes_and_both_spectral_modules():
    torch.manual_seed(5)
    model = _small().eval()
    # Nonidentical filters expose dynamic/static weighting rather than asserting
    # a false nonzero gradient at the exact source initialization.
    with torch.no_grad():
        for index, weights in enumerate(model.spectral_filter.filters):
            weights.fill_(0.5 + index * 0.3)
    imu = torch.randn(2, 6, 200, requires_grad=True)
    output = model(imu)
    assert isinstance(model, BaseModel) and set(output) == {"vel"}
    assert output["vel"].shape == (2, 3)
    target = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    loss, items = model.loss(output, {"imu": imu, "target": target})
    torch.testing.assert_close(loss, (output["vel"] - target).square().mean())
    assert set(items) == {"mse"}
    loss.backward()
    assert torch.isfinite(imu.grad).all()
    assert imu.grad[:, 2].abs().sum() > 0 and imu.grad[:, 5].abs().sum() > 0
    for weights in (model.velocity_head.weight, model.spectral_filter.filters[0],
                    model.spectral_filter.static_logits,
                    model.temporal_blocks[0].spectral_mixer.first.weight):
        assert weights.grad is not None and weights.grad.abs().sum() > 0
    assert model.velocity_head.weight.grad[2].abs().sum() > 0
    model.set_loss("mse")
    for name, kwargs in (("mse_l1", {}), ("mse", {"aadl_weight": 1.0})):
        with pytest.raises(ValueError):
            model.set_loss(name, **kwargs)


@pytest.mark.parametrize("order", [[1, 0, 2, 3, 4, 5], [3, 1, 2, 0, 4, 5]])
def test_ordered_axis_and_sensor_identity_is_not_arbitrarily_permutation_invariant(order):
    torch.manual_seed(31)
    model = _small().eval()
    imu = torch.randn(1, 6, 200)
    with torch.no_grad():
        original, changed = model.encode_window(imu).mean(-1), model.encode_window(imu[:, order]).mean(-1)
        difference = original - changed
        selected = difference.abs().argmax().item()
        assert difference[0, selected].abs() > 1e-6
        model.velocity_head.weight.zero_()
        model.velocity_head.bias.zero_()
        model.velocity_head.weight[0, selected] = 1.0
        before, after = model(imu)["vel"], model(imu[:, order])["vel"]
    torch.testing.assert_close(before[0, 0] - after[0, 0], difference[0, selected])
    assert not torch.allclose(before, after)


def test_eval_has_no_cross_window_batch_statistics_state_or_extra_forward_inputs():
    model = _small().eval()
    imu = torch.randn(2, 6, 200)
    with torch.no_grad():
        first = model(imu)["vel"]
        model(torch.randn_like(imu))
        repeated = model(imu)["vel"]
        independent = torch.cat([model(imu[index:index + 1])["vel"] for index in range(2)])
    torch.testing.assert_close(first, repeated)
    torch.testing.assert_close(first, independent, rtol=1e-4, atol=1e-6)
    with pytest.raises(TypeError):
        model(imu, {"gravity": torch.zeros(2, 200, 3)})
    for invalid in (torch.zeros(2, 6, 199), torch.zeros(2, 6, 201), torch.zeros(2, 1, 6, 200)):
        with pytest.raises(ValueError):
            model(invalid)
