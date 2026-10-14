"""Server-only TESSERA-v1 architecture contracts, written NOT executed.

Parent conftest forbids collection before audited original-task tail handoff.
Width16/layer1/FF32 below are explicit diagnostic identities, not acceptance or
silent substitution of source defaults512/8/4096. SSL is intentionally absent.
"""

import importlib.util
import inspect
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from inertial_benchmark.cfg import INPUT_KEYS, get_cfg
from inertial_benchmark.nn.base import BaseModel, InputSpec
from inertial_benchmark.nn.registry import MODELS


HERE = Path(__file__).parent
BEFORE_REGISTRY = dict(MODELS)
SPEC = importlib.util.spec_from_file_location("cvpr2026_staging_tessera_v1", HERE / "model.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
AFTER_IMPORT_REGISTRY = dict(MODELS)


def _small(spec=None, **overrides):
    args = dict(width=16, layers=1, heads=2, feedforward_width=32, latent_dim=4, dropout=0.0)
    args.update(overrides)
    return module.TesseraV1DualStreamVelocity(spec or InputSpec(dims=3), **args)


def test_profile_source_defaults_schema_SSL_boundary_and_no_registration():
    assert AFTER_IMPORT_REGISTRY == BEFORE_REGISTRY
    profile = yaml.safe_load((HERE / "model_3d.yaml").read_text())
    provenance = json.loads((HERE / "provenance.json").read_text())
    assert profile["arch"] == "tessera_v1_dualstream_architecture_velocity"
    defaults = inspect.signature(module.TesseraV1DualStreamVelocity).parameters
    for key, expected in (("width", 512), ("layers", 8), ("heads", 8),
                          ("feedforward_width", 4096), ("latent_dim", 128)):
        assert defaults[key].default == expected and profile["args"][key] == expected
    assert set(profile["input"]) <= set(INPUT_KEYS) and "channels" not in profile["input"]
    assert tuple(profile["input_channels"]) == module.CHANNELS
    cfg = get_cfg({"model": profile})
    assert cfg.window == 200 and cfg.rate == 200 and cfg.dims == 3
    assert cfg.frame == "gravity_world" and cfg.orientation == "reference"
    assert cfg.remove_gravity is False and cfg.extra_inputs == []
    assert cfg.history == cfg.history_stride == cfg.output_steps == 0
    assert profile["loss"] == "mse" and profile["ssl_objectives"] == "omitted"
    assert profile["causality"] == "window_noncausal"
    assert provenance["implementation_status"] == "static_draft"
    for key in ("runtime_verified", "capacity_accepted", "formal_matrix_enrolled", "registered"):
        assert provenance[key] is False and profile[key] is False
    assert provenance["repository"]["commit"] == module.TesseraV1DualStreamVelocity.source_commit
    assert provenance["adaptation"]["complete_paper_reproduction"] is False
    assert "4-layer" in " ".join(provenance["adaptation"]["paper_code_differences"])


@pytest.mark.parametrize("fields", [
    {"dims": 2}, {"window": 64}, {"window": 199}, {"window": 201},
    {"rate": 100}, {"rate": float("nan")}, {"rate": float("inf")},
    {"frame": "body"}, {"frame": "gravity_yaw_local"},
    {"orientation": "device"}, {"remove_gravity": True},
    {"target": "displacement"}, {"target": "velocity_at_end"},
    {"target": "frame_velocity"}, {"output_steps": 2}, {"overlap": "center"},
    {"history": 1}, {"history": 2, "history_stride": 200}, {"history_stride": 1},
    {"extra_inputs": ("orientation",)}, {"extra_inputs": ("gravity",)},
    {"extra_inputs": ("init_velocity",)},
    {"channels": ("gyro_y", "gyro_x", "gyro_z", "acc_x", "acc_y", "acc_z")},
    {"channels": ("acc_x", "gyro_y", "gyro_z", "gyro_x", "acc_y", "acc_z")},
    {"channels": ("gyro_x",) * 6},
])
def test_reject_extra_modalities_clock_permissions_layout_and_target(fields):
    with pytest.raises(ValueError):
        _small(InputSpec(**dict({"dims": 3}, **fields)))


@pytest.mark.parametrize("args", [
    {"pretrained": True}, {"width": 0}, {"width": True}, {"layers": 0},
    {"heads": 0}, {"width": 15}, {"width": 18}, {"heads": 3},
    {"feedforward_width": 0}, {"latent_dim": 8},
    {"dropout": float("nan")}, {"dropout": float("inf")},
    {"dropout": -0.1}, {"dropout": 1.0},
])
def test_reject_pretraining_nonfinite_dropout_and_unsupported_source_shape(args):
    with pytest.raises(ValueError):
        _small(**args)


def test_relative_positions_exact_source_sinusoid_derived_inside_window():
    position = module.RelativeWindowPosition(16)
    tokens = torch.randn(2, 200, 16, dtype=torch.float64)
    actual = position(tokens)
    expected = tokens.new_zeros((1, 200, 16))
    indices = torch.arange(200, dtype=torch.float64)
    frequencies = torch.exp(torch.arange(0, 16, 2, dtype=torch.float64) * (-math.log(10000.0) / 16))
    expected[0, :, 0::2] = torch.sin(indices[:, None] * frequencies[None])
    expected[0, :, 1::2] = torch.cos(indices[:, None] * frequencies[None])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, position(tokens * 500))
    assert list(position.parameters()) == []
    assert actual.shape == (1, 200, 16) and actual.dtype == tokens.dtype
    assert actual[0, 0, 0::2].count_nonzero() == 0
    torch.testing.assert_close(actual[0, 0, 1::2], torch.ones(8, dtype=torch.float64))


def test_pool_GRU_conditioned_weights_but_original_token_values():
    pool = module.GRUConditionedAttentionPool(8).double().eval()
    tokens = torch.randn(2, 13, 8, dtype=torch.float64, requires_grad=True)
    context, _ = pool.temporal_context(tokens)
    weights = pool.query(context).softmax(dim=1)
    expected = (weights * tokens).sum(dim=1)
    actual = pool(tokens)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, 1, dtype=torch.float64))
    assert not torch.allclose(actual, (weights * context).sum(dim=1))
    actual.square().sum().backward()
    for parameter in (pool.query.weight, pool.temporal_context.weight_ih_l0,
                      pool.temporal_context.weight_hh_l0):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert tokens.grad.abs().sum() > 0 and torch.isfinite(tokens.grad).all()
    with torch.no_grad():
        pool(torch.randn_like(tokens))
        torch.testing.assert_close(actual.detach(), pool(tokens))


def test_two_encoders_are_unshared_but_layers_retain_source_clone_initialization():
    model = _small(layers=2)
    assert model.gyro_encoder is not model.acc_encoder
    gyro = dict(model.gyro_encoder.named_parameters())
    acc = dict(model.acc_encoder.named_parameters())
    assert set(gyro) == set(acc)
    assert all(gyro[key].data_ptr() != acc[key].data_ptr() for key in gyro)
    for encoder in (model.gyro_encoder, model.acc_encoder):
        first, second = encoder.transformer.layers
        assert first.norm_first is False and first.self_attn.batch_first is True
        assert first.self_attn.num_heads == 2
        assert first.linear1.out_features == 32
        assert first.activation is torch.nn.functional.relu
        for name, parameter in first.named_parameters():
            other = dict(second.named_parameters())[name]
            assert parameter.data_ptr() != other.data_ptr()
            torch.testing.assert_close(parameter, other)
        assert encoder.embedding[0].in_features == 3
        assert encoder.embedding[0].out_features == 4 * model.reducer.out_features
    assert isinstance(model.reducer, nn.Linear) and not isinstance(model.reducer, nn.Sequential)
    assert not any(isinstance(child, nn.BatchNorm1d) for child in model.modules())
    assert not hasattr(model, "projector") and not hasattr(model, "ssl_loss")


def test_modality_encoder_gradient_isolation_and_ordered_channel_split():
    torch.manual_seed(17)
    model = _small().double().eval()
    imu = torch.randn(2, 6, 200, dtype=torch.float64, requires_grad=True)
    gyro, acc = model.encode_streams(imu)
    torch.testing.assert_close(gyro, model.gyro_encoder(imu[:, :3].transpose(1, 2)))
    torch.testing.assert_close(acc, model.acc_encoder(imu[:, 3:].transpose(1, 2)))
    gyro_gradient = torch.autograd.grad(gyro[0, 0], imu, retain_graph=True)[0]
    acc_gradient = torch.autograd.grad(acc[0, 0], imu)[0]
    assert gyro_gradient[:, :3].abs().sum() > 0 and gyro_gradient[:, 3:].count_nonzero() == 0
    assert acc_gradient[:, 3:].abs().sum() > 0 and acc_gradient[:, :3].count_nonzero() == 0
    changed = imu.detach().clone()
    changed[:, 3:] += 20.0
    with torch.no_grad():
        changed_gyro, _ = model.encode_streams(changed)
    torch.testing.assert_close(changed_gyro, gyro.detach())


def test_ordered_concat_single_reducer_and_XYZ_adapter_are_axis_sensitive():
    class SensorProbe(nn.Module):
        def forward(self, sensor):
            return torch.cat((sensor.mean(1), sensor.new_zeros((sensor.shape[0], 13))), dim=-1)

    model = _small().eval()
    model.gyro_encoder, model.acc_encoder = SensorProbe(), SensorProbe()
    with torch.no_grad():
        model.reducer.weight.zero_()
        model.reducer.bias.zero_()
        # Preserve gyro-x, gyro-z, acc-x, acc-z in that explicit order.
        for output, selected in enumerate((0, 2, 16, 18)):
            model.reducer.weight[output, selected] = 1.0
        model.velocity_head.weight.zero_()
        model.velocity_head.bias.zero_()
        model.velocity_head.weight[0, 0] = 1.0
        model.velocity_head.weight[1, 2] = 1.0
        model.velocity_head.weight[2, 1] = 1.0
        model.velocity_head.weight[2, 3] = 1.0
    imu = torch.arange(1.0, 7.0).view(1, 6, 1).expand(1, 6, 200)
    torch.testing.assert_close(model(imu)["vel"], torch.tensor([[1.0, 4.0, 9.0]]))
    torch.testing.assert_close(model(imu[:, [3, 1, 2, 0, 4, 5]])["vel"], torch.tensor([[4.0, 1.0, 9.0]]))
    torch.testing.assert_close(model(imu[:, [2, 1, 0, 3, 4, 5]])["vel"], torch.tensor([[3.0, 4.0, 7.0]]))


def test_true_XYZ_mse_Z_sensor_axes_and_all_branch_pool_gradients():
    torch.manual_seed(9)
    model = _small().eval()
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
    for encoder in (model.gyro_encoder, model.acc_encoder):
        for weights in (encoder.embedding[0].weight,
                        encoder.transformer.layers[0].self_attn.in_proj_weight,
                        encoder.pool.query.weight,
                        encoder.pool.temporal_context.weight_ih_l0):
            assert weights.grad is not None and torch.isfinite(weights.grad).all()
            assert weights.grad.abs().sum() > 0
    assert model.velocity_head.weight.grad[2].abs().sum() > 0
    assert model.reducer.weight.grad.abs().sum() > 0
    model.set_loss("mse")
    for name, kwargs in (("mse_l1", {}), ("mse", {"bt_lambda": 0.005})):
        with pytest.raises(ValueError):
            model.set_loss(name, **kwargs)


def test_eval_resets_GRU_between_windows_no_external_position_or_extra_inputs():
    model = _small().eval()
    imu = torch.randn(2, 6, 200)
    with torch.no_grad():
        first = model(imu)["vel"]
        model(torch.randn_like(imu))
        repeated = model(imu)["vel"]
        independent = torch.cat([model(imu[index:index + 1])["vel"] for index in range(2)])
    torch.testing.assert_close(first, repeated)
    torch.testing.assert_close(first, independent, rtol=1e-4, atol=1e-6)
    for extra in ({"init_velocity": torch.zeros(2, 3)}, {"calendar_doy": torch.zeros(2, 200)}):
        with pytest.raises(TypeError):
            model(imu, extra)
    for invalid in (torch.zeros(2, 6, 199), torch.zeros(2, 6, 201), torch.zeros(2, 7, 200),
                    torch.zeros(2, 1, 6, 200)):
        with pytest.raises(ValueError):
            model(invalid)
