"""Server-only scientific contract tests for the isolated ModernTCN port.

Run on AutoDL with the agreed source bundle and CPU slot.  Passing these tests
does not establish original-paper reproduction or benchmark accuracy.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from inertial_benchmark.data.views import CHANNELS
from inertial_benchmark.nn.base import BaseModel, InputSpec


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("staging_moderntcn", HERE / "model.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _small(dims=3, window=64):
    return module.ModernTCNVelocity(
        InputSpec(window=window, dims=dims), widths=(8, 16), depths=(1, 1),
        large_kernels=(13, 13), small_kernels=(5, 5), dropout=0.0)


@pytest.mark.parametrize("dims", [2, 3])
def test_basemodel_shapes_and_shared_mse(dims):
    model = _small(dims)
    imu = torch.randn(2, 6, 64)
    target = torch.randn(2, dims)
    out = model(imu)
    assert isinstance(model, BaseModel)
    assert set(out) == {"vel"}
    assert out["vel"].shape == (2, dims)
    loss, items = model.loss(out, {"imu": imu, "target": target})
    torch.testing.assert_close(loss, (out["vel"] - target).square().mean())
    assert set(items) == {"mse"}


def test_default_xyz_backward_updates_real_z_and_both_sensor_z_inputs():
    torch.manual_seed(18)
    model = module.ModernTCNVelocity(InputSpec(window=200, dims=3))
    imu = torch.randn(2, 6, 200, requires_grad=True)
    prediction = model(imu)["vel"]
    # Only Z has a residual, proving genuine Z supervision propagates through
    # the ordered linear head and the shared six-axis representation.
    target = prediction.detach().clone()
    target[:, 2] += torch.tensor([1.0, -0.75])
    loss, _ = model.loss({"vel": prediction}, {"imu": imu, "target": target})
    loss.backward()
    assert prediction.shape == (2, 3) and torch.isfinite(loss)
    assert model.velocity_head.weight.grad[2].abs().sum() > 0
    torch.testing.assert_close(model.velocity_head.weight.grad[:2],
                               torch.zeros_like(model.velocity_head.weight.grad[:2]))
    assert imu.grad[:, 2].abs().sum() > 0  # actual gyro-z, not a synthetic channel
    assert imu.grad[:, 5].abs().sum() > 0  # actual acc-z
    assert torch.isfinite(imu.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    before = model.velocity_head.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=1e-3).step()
    assert not torch.equal(model.velocity_head.weight[2], before[2])


@pytest.mark.parametrize("dims,expected", [(2, 88866), (3, 89251)])
def test_default_parameter_and_patch_fixtures(dims, expected):
    model = module.ModernTCNVelocity(InputSpec(window=200, dims=dims))
    assert model.num_params == expected
    assert model.patch_count == 50
    assert model.stage_patch_counts == [50, 25]
    assert model.downsample_layers[0][0].in_channels == 1
    assert model.downsample_layers[0][0].kernel_size == (8,)
    assert model.downsample_layers[0][0].stride == (4,)
    assert model.velocity_head.in_features == 6 * 64
    assert model.velocity_head.out_features == dims
    assert len(model.stages) == 2


def test_stem_repeats_last_sample_and_odd_downsample_copies_tail():
    model = _small(window=68).eval()  # 17 patches -> repeat last -> 9 patches
    observed = []
    hooks = [
        layer.register_forward_pre_hook(
            lambda _layer, args: observed.append(args[0].detach().clone()))
        for layer in model.downsample_layers
    ]
    imu = torch.arange(68.0).reshape(1, 1, 68).repeat(2, 6, 1)
    try:
        with torch.no_grad():
            encoded = model.encode(imu)
    finally:
        for hook in hooks:
            hook.remove()
    assert observed[0].shape == (12, 1, 72)
    torch.testing.assert_close(observed[0][0, 0, -4:], torch.full((4,), 67.0))
    assert observed[1].shape[-1] == 18
    torch.testing.assert_close(observed[1][:, :, -1], observed[1][:, :, -2])
    assert encoded.shape == (2, 6, 16, 9)


def test_depthwise_temporal_kernel_preserves_channel_isolation():
    temporal = module.LargeKernelDepthwise(12, 13, 5).eval()
    assert temporal.large_branch[0].groups == 12
    assert temporal.small_branch[0].groups == 12
    with torch.no_grad():
        temporal.large_branch[0].weight.fill_(1.0)
        temporal.small_branch[0].weight.zero_()
    impulse = torch.zeros(1, 12, 31)
    impulse[0, 3, 15] = 1.0
    with torch.no_grad():
        response = temporal(impulse)
    assert torch.count_nonzero(response[0, 3]) == 13
    assert torch.count_nonzero(response[0, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]]) == 0


def _positive_ffn_weights(block):
    with torch.no_grad():
        for ffn in (block.feature_ffn, block.variable_ffn):
            for layer in ffn:
                if isinstance(layer, torch.nn.Conv1d):
                    layer.weight.fill_(0.1)
                    layer.bias.zero_()


def test_feature_ffn_mixes_features_without_cross_variable_dependency():
    block = module.ModernTCNBlock(6, 8, 2, 13, 5, 0.0).eval()
    _positive_ffn_weights(block)
    assert block.feature_ffn[0].groups == 6
    assert block.feature_ffn[3].groups == 6
    features = torch.ones(2, 6, 8, 7)
    changed = features.clone()
    changed[:, 2, 0] += 1
    with torch.no_grad():
        original = block.mix_features(features)
        perturbed = block.mix_features(changed)
    torch.testing.assert_close(original[:, [0, 1, 3, 4, 5]],
                               perturbed[:, [0, 1, 3, 4, 5]])
    assert (original[:, 2, 1:] - perturbed[:, 2, 1:]).abs().min() > 0


def test_variable_ffn_mixes_variables_without_cross_feature_dependency():
    block = module.ModernTCNBlock(6, 8, 2, 13, 5, 0.0).eval()
    _positive_ffn_weights(block)
    assert block.variable_ffn[0].groups == 8
    assert block.variable_ffn[3].groups == 8
    features = torch.ones(2, 6, 8, 7)
    changed = features.clone()
    changed[:, 2, 0] += 1
    with torch.no_grad():
        original = block.mix_variables(features)
        perturbed = block.mix_variables(changed)
    torch.testing.assert_close(original[:, :, 1:], perturbed[:, :, 1:])
    assert (original[:, [0, 1, 3, 4, 5], 0]
            - perturbed[:, [0, 1, 3, 4, 5], 0]).abs().min() > 0


def test_one_whole_block_residual_instead_of_extra_ffn_residuals():
    block = module.ModernTCNBlock(6, 8, 1, 13, 5, 0.0).eval()
    with torch.no_grad():
        block.variable_ffn[3].weight.zero_()
        block.variable_ffn[3].bias.zero_()
    features = torch.randn(2, 6, 8, 7)
    torch.testing.assert_close(block(features), features)


def test_ordered_temporal_mean_and_linear_head_is_deterministic():
    model = _small().eval()
    imu = torch.randn(2, 6, 64)
    with torch.no_grad():
        encoded = model.encode(imu)
        expected = model.velocity_head(encoded.mean(dim=-1).flatten(start_dim=1))
        first = model(imu)["vel"]
        second = model(imu)["vel"]
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, first)
    assert model.causality == "window_noncausal"
    assert model.training_initialization == "scratch_supervised"


def test_drafts_provenance_and_import_remain_unregistered():
    from inertial_benchmark.nn.registry import list_models

    assert "moderntcn_velocity" not in list_models()
    for filename, dims in (("model_2d.yaml", 2), ("model_3d.yaml", 3)):
        profile = yaml.safe_load((HERE / filename).read_text())
        spec = InputSpec.from_dict(profile["input"])
        model = module.ModernTCNVelocity(spec, **profile["args"])
        assert model.input_spec.channels == CHANNELS
        assert spec.window == 200 and spec.rate == 200
        assert spec.dims == dims and spec.extra_inputs == ()
        assert profile["registered"] is False
        assert profile["training"] == "supervised_from_scratch"
        assert profile["loss"] == "mse" and profile["args"]["pretrained"] is False
    provenance = json.loads((HERE / "provenance.json").read_text())
    assert provenance["active_registered"] is False
    assert provenance["accepted_method_count_increment"] == 0
    assert provenance["official_source"]["license"] == "MIT"
    assert provenance["formal_2067_rows_changed"] is False


@pytest.mark.parametrize("changes", [
    {"channels": tuple(reversed(CHANNELS))},
    {"channels": ("gyro_x",) * 6},
    {"extra_inputs": ("init_velocity",)},
    {"frame": "gravity_yaw_local"}, {"target": "displacement"},
    {"history": 2, "history_stride": 4}, {"remove_gravity": True}, {"rate": 100},
    {"orientation": "device"}, {"output_steps": 1},
])
def test_rejects_incompatible_scientific_input(changes):
    with pytest.raises(ValueError, match="requires ordered"):
        module.ModernTCNVelocity(InputSpec(**({"window": 64, "dims": 3} | changes)))


@pytest.mark.parametrize("changes", [
    {"patch_size": 2, "patch_stride": 4}, {"patch_size": 65},
    {"widths": (8,), "depths": (1, 1)},
    {"large_kernels": (12, 13)}, {"small_kernels": (15, 5)},
    {"ffn_ratio": 1.5}, {"downsample_ratio": 3}, {"pretrained": True},
])
def test_rejects_unreviewed_or_invalid_architecture(changes):
    with pytest.raises(ValueError):
        module.ModernTCNVelocity(InputSpec(window=64, dims=3), **changes)


def test_rejects_wrong_runtime_input_shape():
    model = _small()
    with pytest.raises(ValueError, match="expected input"):
        model(torch.randn(2, 6, 65))
