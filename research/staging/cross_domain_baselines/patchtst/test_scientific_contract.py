"""CPU scientific-contract tests for the unregistered PatchTST staging PoC."""

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from inertial_benchmark.nn.base import InputSpec


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("staging_patchtst", HERE / "model.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _small(dims: int = 3, window: int = 64):
    return module.PatchTSTVelocity(
        InputSpec(window=window, dims=dims), patch_len=8, stride=4,
        d_model=16, n_heads=4, n_layers=2, d_ff=32,
        dropout=0.0, attention_dropout=0.0, head_dropout=0.0,
    )


@pytest.mark.parametrize("dims", [2, 3])
def test_basemodel_input_output_and_loss_shapes(dims):
    model = _small(dims)
    imu = torch.randn(3, 6, 64)
    out = model(imu)
    assert isinstance(model, module.BaseModel)
    assert model.input_spec == InputSpec(window=64, dims=dims)
    assert out["vel"].shape == (3, dims)
    assert model.encode_channels(imu).shape == (3, 6, 16, model.patch_count)
    loss, items = model.loss(out, {"imu": imu, "target": torch.randn(3, dims)})
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert items.keys() == {"mse"}


def test_patch_tokens_overlap_and_repeat_last_end_padding():
    model = module.PatchTSTVelocity(
        InputSpec(window=20, dims=2), patch_len=4, stride=3,
        d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0)
    imu = torch.arange(20.0).view(1, 1, 20).repeat(1, 6, 1)
    patches = model.encoder.patchify(imu)
    assert patches.shape == (1, 6, 7, 4)
    torch.testing.assert_close(patches[0, 0, 0], torch.tensor([0., 1., 2., 3.]))
    torch.testing.assert_close(patches[0, 0, 1], torch.tensor([3., 4., 5., 6.]))
    torch.testing.assert_close(patches[0, 0, -1], torch.tensor([18., 19., 19., 19.]))


def test_channel_independence_permutation_and_shared_weights():
    torch.manual_seed(7)
    model = _small().eval()
    imu = torch.randn(2, 6, 64)
    order = torch.tensor([5, 2, 0, 4, 1, 3])
    with torch.no_grad():
        encoded = model.encode_channels(imu)
        permuted = model.encode_channels(imu[:, order])
    torch.testing.assert_close(permuted, encoded[:, order])

    changed = imu.clone()
    changed[:, 2] += torch.linspace(-2, 2, 64)
    with torch.no_grad():
        changed_encoded = model.encode_channels(changed)
    torch.testing.assert_close(changed_encoded[:, [0, 1, 3, 4, 5]],
                               encoded[:, [0, 1, 3, 4, 5]])
    assert not torch.equal(changed_encoded[:, 2], encoded[:, 2])

    identical = torch.randn(2, 1, 64).expand(-1, 6, -1).clone()
    with torch.no_grad():
        same_encoded = model.encode_channels(identical)
    for channel in range(1, 6):
        torch.testing.assert_close(same_encoded[:, 0], same_encoded[:, channel])
    assert model.encoder_channel_independent is True
    assert not hasattr(model, "channel_independent")
    assert isinstance(model.encoder.patch_projection, torch.nn.Linear)
    assert isinstance(model.velocity_head[-1], torch.nn.Linear)


def test_eval_encoder_channel_gradient_isolation():
    model = _small().eval()
    imu = torch.randn(2, 6, 64, requires_grad=True)
    encoded = model.encode_channels(imu)
    encoded[:, 2].square().sum().backward()
    assert imu.grad[:, 2].abs().sum() > 0
    torch.testing.assert_close(imu.grad[:, [0, 1, 3, 4, 5]],
                               torch.zeros_like(imu.grad[:, [0, 1, 3, 4, 5]]))


def test_encoder_preserves_upstream_feature_then_patch_layout():
    model = _small().eval()
    captured = []
    handle = model.encoder.layers[-1].register_forward_hook(
        lambda _layer, _inputs, output: captured.append(output[0]))
    try:
        with torch.no_grad():
            encoded = model.encode_channels(torch.randn(2, 6, 64))
    finally:
        handle.remove()
    expected = captured[0].reshape(2, 6, model.patch_count, 16).transpose(-2, -1)
    torch.testing.assert_close(encoded, expected)
    assert model.model_cfg["encoder_output_layout"] == "B,C,D,N"
    assert model.model_cfg["head_flatten_order"] == "C,D,N"


def test_ordered_task_adapter_flatten_keeps_channel_feature_patch_identity():
    model = _small().eval()
    head = model.velocity_head[-1]
    shape = (1, 6, 16, model.patch_count)
    features = torch.arange(head.in_features, dtype=torch.float32).reshape(shape)
    selected = ((0, 3, 2), (3, 1, 4), (5, 7, 1))
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        for output_axis, (channel, feature, patch) in enumerate(selected):
            flat_index = (channel * 16 + feature) * model.patch_count + patch
            head.weight[output_axis, flat_index] = 1
        velocity = model.velocity_head(features)
    expected = torch.tensor([[features[0, c, d, n] for c, d, n in selected]])
    torch.testing.assert_close(velocity, expected)


@pytest.mark.parametrize("order", [[1, 0, 2, 3, 4, 5], [3, 1, 2, 0, 4, 5]])
def test_final_velocity_can_distinguish_axis_and_sensor_units(order):
    torch.manual_seed(19)
    model = _small().eval()
    imu = torch.randn(2, 6, 64)
    with torch.no_grad():
        encoded = model.encode_channels(imu)
        permuted = model.encode_channels(imu[:, order])
        difference = (encoded[0, 0] - permuted[0, 0]).flatten()
        selected = difference.abs().argmax().item()
        assert difference[selected].abs() > 1e-4
        # A deterministic adapter selecting a channel-0 feature establishes
        # that the architecture can distinguish gyro-x from gyro-y or acc-x.
        # This avoids a probabilistic assertion about random head weights.
        head = model.velocity_head[-1]
        head.weight.zero_()
        head.bias.zero_()
        head.weight[0, selected] = 1
        original_velocity = model(imu)["vel"]
        permuted_velocity = model(imu[:, order])["vel"]
    torch.testing.assert_close(
        original_velocity[0, 0] - permuted_velocity[0, 0], difference[selected])
    assert not torch.allclose(original_velocity, permuted_velocity)


def test_training_batchnorm_pools_statistics_over_folded_channels():
    tokens = torch.arange(12 * 5 * 4, dtype=torch.float32).reshape(12, 5, 4)
    norm = torch.nn.BatchNorm1d(4, affine=False, track_running_stats=False).train()
    actual = module._PatchEncoderLayer._batch_norm(norm, tokens)
    expected = (tokens - tokens.mean(dim=(0, 1))) / torch.sqrt(
        tokens.var(dim=(0, 1), unbiased=False) + norm.eps)
    torch.testing.assert_close(actual, expected)
    # B=2 and C=6 were folded into 12; train statistics are shared, not
    # separately estimated within each sensor channel.
    per_sequence = (tokens - tokens.mean(dim=1, keepdim=True)) / torch.sqrt(
        tokens.var(dim=1, keepdim=True, unbiased=False) + norm.eps)
    assert not torch.allclose(actual, per_sequence)


def test_attention_scores_and_residual_path_match_source_mechanics():
    torch.manual_seed(23)
    attention = module._ResidualSelfAttention(8, 2, 0.0, 0.0).eval()
    tokens = torch.randn(3, 4, 8)
    previous = torch.randn(3, 2, 4, 4)
    with torch.no_grad():
        actual, scores = attention(tokens, previous)
        query = attention.query(tokens).reshape(3, 4, 2, 4).transpose(1, 2)
        key = attention.key(tokens).reshape(3, 4, 2, 4).transpose(1, 2)
        value = attention.value(tokens).reshape(3, 4, 2, 4).transpose(1, 2)
        expected_scores = query @ key.transpose(-2, -1) / 2.0 + previous
        weighted = (expected_scores.softmax(dim=-1) @ value).transpose(1, 2)
        expected = attention.output(weighted.reshape(3, 4, 8))
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(actual, expected)
    layer = module._PatchEncoderLayer(8, 2, 16, 0.2, 0.0)
    assert layer.attention.output[-1].p == 0.2
    assert layer.attention_residual_dropout.p == 0.2


def test_position_embedding_is_shared_learned_and_source_initialized():
    model = _small()
    position = model.encoder.position
    assert position.shape == (1, model.patch_count, 16)
    assert position.requires_grad
    assert position.min() >= -0.02 and position.max() <= 0.02


def test_default_xyz_cpu_forward_loss_backward_and_real_z_gradients():
    torch.manual_seed(11)
    model = module.PatchTSTVelocity(InputSpec(window=200, dims=3))
    imu = torch.randn(2, 6, 200, requires_grad=True)
    target = torch.tensor([[0.7, -0.2, 1.1], [-0.5, 0.4, -1.3]])
    out = model(imu)
    loss, _ = model.loss(out, {"imu": imu, "target": target})
    loss.backward()

    assert out["vel"].shape == (2, 3)
    assert model.patch_count == 25
    assert model.velocity_head[-1].in_features == 6 * 128 * 25
    assert model.num_params == 460419
    assert torch.isfinite(loss)
    assert model.velocity_head[-1].weight.grad[2].abs().sum() > 0
    assert imu.grad[:, 2].abs().sum() > 0  # gyro-z channel participates
    assert imu.grad[:, 5].abs().sum() > 0  # accelerometer-z channel participates
    parameters = list(model.parameters())
    assert parameters and all(torch.isfinite(parameter).all() for parameter in parameters)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in parameters)


def test_yaml_drafts_are_scratch_only_and_not_registered():
    from inertial_benchmark.cfg import get_cfg
    from inertial_benchmark.nn.registry import list_models

    assert "patchtst_ci_ordered_velocity" not in list_models()
    for name, dims in (("model_2d.yaml", 2), ("model_3d.yaml", 3)):
        profile = yaml.safe_load((HERE / name).read_text())
        assert profile["registered"] is False
        assert profile["training"] == "supervised_from_scratch"
        assert profile["args"]["pretrained"] is False
        assert profile["arch"] == "patchtst_ci_ordered_velocity"
        assert profile["task_adapter"] == module.PatchTSTVelocity.task_adapter
        assert profile["causality"] == module.PatchTSTVelocity.causality == "window_noncausal"
        assert profile["encoder_output_layout"] == "B,C,D,N"
        assert profile["head_flatten_order"] == "C,D,N"
        assert profile["input_channels"] == [
            "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
        assert profile["input"]["extra_inputs"] == []
        assert profile["input"]["dims"] == dims
        assert profile["input"]["rate"] == 200
        assert profile["input"]["target"] == "avg_velocity"
        assert profile["input"]["frame"] == "gravity_world"
        assert profile["input"]["orientation"] == "reference"
        assert profile["input"]["remove_gravity"] is False
        effective = get_cfg({"model": profile})
        spec = InputSpec.from_cfg(effective)
        assert spec == InputSpec(window=200, dims=dims)


@pytest.mark.parametrize(
    "changes",
    [
        {"dims": 1}, {"channels": ("gx",) * 5}, {"target": "displacement"},
        {"frame": "body"}, {"history": 2, "history_stride": 4},
        {"channels": ("gyro_y", "gyro_x", "gyro_z", "acc_x", "acc_y", "acc_z")},
        {"channels": ("gyro_x",) * 6}, {"rate": 100}, {"rate": float("nan")},
        {"orientation": "device"}, {"remove_gravity": True},
        {"extra_inputs": ("init_velocity",)}, {"extra_inputs": ("orientation",)},
        {"extra_inputs": ("gravity",)},
    ],
)
def test_rejects_incompatible_task_contract(changes):
    values = {"window": 64, "dims": 3} | changes
    with pytest.raises(ValueError):
        _ = module.PatchTSTVelocity(
            InputSpec(**values), patch_len=8, stride=4,
            d_model=16, n_heads=4, n_layers=1, d_ff=32)


def test_rejects_pretraining_and_invalid_patch_geometry():
    spec = InputSpec(window=64, dims=3)
    with pytest.raises(ValueError, match="supervised from scratch"):
        module.PatchTSTVelocity(spec, pretrained=True)
    with pytest.raises(ValueError, match="patch_len"):
        module.PatchTSTVelocity(spec, patch_len=65)
    with pytest.raises(ValueError, match="divisible"):
        module.PatchTSTVelocity(spec, d_model=15, n_heads=4)


@pytest.mark.parametrize("argument", ["dropout", "attention_dropout", "head_dropout"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_dropout_at_every_position(argument, value):
    with pytest.raises(ValueError, match="finite"):
        module.PatchTSTVelocity(InputSpec(window=64, dims=3), **{argument: value})

