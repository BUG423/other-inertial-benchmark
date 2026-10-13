"""Learned XYZ heads, vertical supervision and checkpoint contracts.

These engineering checks use analytical 3D trajectories, not paper accuracy scores.
The original 2D configurations and their fixtures remain separate references.
"""

import numpy as np
import pytest
from synthetic import make_sequence

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import ConfigError, get_cfg, load_model_cfg  # noqa: E402
from inertial_benchmark.data.views import SequenceView  # noqa: E402
from inertial_benchmark.engine.predictor import load_model  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.utils.torch_utils import save_checkpoint  # noqa: E402


@pytest.mark.parametrize("name", [
    "ronin_resnet18_3d", "ronin_lstm_3d", "imunet_3d", "tlio",
])
def test_xyz_head_target_loss_and_checkpoint(name, tmp_path):
    """Real model Z receives gradients; a saved 3D input spec cannot become 2D."""
    torch.manual_seed(0)
    cfg = get_cfg({"model": name, "device": "cpu", "metric_dims": 3, "amp": False})
    model = build_model(cfg).eval()
    spec = model.input_spec
    assert spec.dims == 3
    if name != "tlio":
        assert model.model_cfg["name"] == name
        assert model.model_cfg["fidelity"].startswith("3d_adapted")
    seq = make_sequence(duration=3.0, seed=7)
    view = SequenceView(seq, spec)
    starts = np.array([0, 100])
    imu = torch.from_numpy(view.imu_windows(starts)).float()
    target = torch.from_numpy(view.targets(starts)).float()
    mask = torch.from_numpy(view.target_mask(starts))
    assert target.shape == (2, *spec.output_shape)
    assert torch.count_nonzero(target[..., 2]) > 0
    out = model(imu)
    assert out["vel"].shape == target.shape
    out["vel"].retain_grad()
    loss, _ = model.loss(out, {"imu": imu, "target": target, "mask": mask})
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(out["vel"].grad).all()
    assert out["vel"].grad[..., 2].abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    # An error confined to Z must produce a nonzero Z gradient and no XY gradient.
    probe = torch.zeros_like(target, requires_grad=True)
    vertical = torch.zeros_like(target)
    vertical[..., 2] = 1.0
    probe_out = {"vel": probe}
    if "logstd" in out:
        probe_out["logstd"] = torch.zeros_like(target, requires_grad=True)
    vertical_loss, _ = model.loss(probe_out, {"imu": imu, "target": vertical, "mask": mask})
    assert vertical_loss > 0
    vertical_loss.backward()
    assert probe.grad[..., 2].abs().sum() > 0
    assert torch.count_nonzero(probe.grad[..., :2]) == 0

    with torch.no_grad():
        expected = model(imu)["vel"]
    checkpoint = tmp_path / "xyz.pt"
    save_checkpoint(checkpoint, {"model": model.state_dict(), "model_cfg": model.model_cfg,
                                 "input_spec": spec.to_dict()})
    loaded_cfg = get_cfg({"model": str(checkpoint), "metric_dims": 3, "device": "cpu"})
    loaded = load_model(loaded_cfg).eval()
    assert loaded.input_spec == spec and loaded.model_cfg["name"] == name
    assert loaded.loss_name == model.loss_name and loaded.loss_kwargs == model.loss_kwargs
    with torch.no_grad():
        torch.testing.assert_close(loaded(imu)["vel"], expected, rtol=0, atol=0)
    with pytest.raises(ConfigError, match="conflicts with checkpoint"):
        get_cfg({"model": str(checkpoint), "dims": 2})


@pytest.mark.parametrize("name", ["ronin_resnet18", "ronin_lstm", "imunet"])
def test_native_2d_config_remains_a_separate_protocol(name):
    original = get_cfg({"model": name})
    adapted = get_cfg({"model": name + "_3d"})
    assert original.dims == 2 and adapted.dims == 3
    assert load_model_cfg(name)["arch"] == load_model_cfg(name + "_3d")["arch"]
    with pytest.raises(ConfigError, match="no recipe"):
        get_cfg({"model": name + "_3d", "recipe": "official"})


@pytest.mark.parametrize("name,args", [
    ("pdr", {}),
    ("mean_speed_heading", {"variant": "constant"}),
    ("mean_speed_heading", {"variant": "gated"}),
])
def test_planar_baselines_refuse_fabricated_vertical_predictions(name, args):
    with pytest.raises(ValueError, match="planar.*dims=2"):
        build_model(get_cfg({"model": name, "dims": 3, "model_args": args}))


@pytest.mark.parametrize("variant", ["zero_velocity", "train_mean_velocity"])
def test_explicit_xyz_constant_baselines_preserve_their_definition(variant):
    cfg = get_cfg({"model": "mean_speed_heading", "dims": 3, "orientation": "reference",
                   "model_args": {"variant": variant, "calibrate_delta": False}})
    model = build_model(cfg)
    train_view = SequenceView(make_sequence(duration=3.0, seed=7), model.input_spec)
    stats = model.calibrate([train_view], split="train")
    evaluation = make_sequence(duration=3.0, seed=19)
    evaluation.position[:, 2] += 0.2 * evaluation.timestamp
    evaluation.velocity[:, 2] += 0.2  # Constant vertical velocity does not change acceleration.
    eval_view = SequenceView(evaluation, model.input_spec)
    starts = eval_view.starts(100)
    _, prediction, _ = model.predict_sequence(evaluation, eval_view, starts)
    assert prediction.shape == (len(starts), 3)
    expected = np.zeros(3) if variant == "zero_velocity" else np.array(stats["mean_velocity"])
    np.testing.assert_allclose(prediction, np.tile(expected, (len(starts), 1)), atol=1e-6)
    if variant == "train_mean_velocity":
        assert abs(expected[2]) > 0
        evaluation_mean = eval_view.targets_world(starts).mean(axis=0)
        assert not np.isclose(expected[2], evaluation_mean[2])
