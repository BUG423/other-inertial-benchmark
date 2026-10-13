"""Raw-task likelihood bridge: coordinates, masks, phase evidence and persistence.

Run on AutoDL only. Small deterministic test heads verify adapters, not published
performance or scientific reproduction of the models.
"""

import json
import math
from types import MethodType

import numpy as np
import pytest
import torch

from inertial_benchmark.cfg import get_cfg
from inertial_benchmark.data.views import SequenceView
from inertial_benchmark.engine.predictor import Predictor, load_model, uncertainty_contract
from inertial_benchmark.engine.results import RunResult, SequenceResult
from inertial_benchmark.metrics.uncertainty import gaussian_uncertainty_metrics
from inertial_benchmark.nn.base import BaseModel, InputSpec
from inertial_benchmark.models.eqnio.tlio import EqNIOTLIO
from inertial_benchmark.models.dive.model import DIVE
from inertial_benchmark.models.meio.model import MEIONet
from inertial_benchmark.models.tlio.model import TLIOResNet
from synthetic import make_sequence


class RawGaussian(BaseModel):
    """Explicit test/plugin contract on the original task, never merged velocities."""

    def __init__(self, spec):
        super().__init__(spec)
        self.mean = torch.nn.Parameter(torch.tensor([.4, -.7, .2]))
        self.loss_name = "gaussian_nll"
        self._ipb_trained_epoch = 12
        self._ipb_epoch_optimizer_updates = 2
        self._ipb_optimizer_updates = 26

    def raw_uncertainty_contract(self, epoch):
        cfg = self.input_spec
        return {"frame": cfg.frame, "target": cfg.target, "layout": cfg.output_layout,
                "unit": "m" if cfg.target in ("displacement", "multi_displacement") else "m/s",
                "mean_key": "vel", "logstd_key": "logstd", "training_start_epoch": 0}

    def forward(self, imu):
        shape = (len(imu),) + self.input_spec.output_shape
        mean = self.mean[:self.input_spec.dims].expand(shape)
        return {"vel": mean, "logstd": torch.full_like(mean, math.log(.6))}


def _predictor(spec):
    cfg = get_cfg({"model": "tlio", "device": "cpu", "val_batch": 3,
                   "eval_stride": 7, "amp": False})
    return Predictor(cfg, model=RawGaussian(spec), device=torch.device("cpu"))


def _result(mean, target=None, logstd=None, valid=None, **contract):
    mean = np.asarray(mean, float)
    target = np.zeros_like(mean) if target is None else np.asarray(target, float)
    logstd = np.zeros_like(mean) if logstd is None else np.asarray(logstd, float)
    valid_shape = (len(mean), 1) if mean.ndim == 2 else mean.shape[:-1]
    valid = np.ones(valid_shape, bool) if valid is None else np.asarray(valid, bool)
    metadata = {"supported": True, "head_trained": True, "reason": "available",
                "dims": mean.shape[-1], "frame": "gravity_yaw_local", "target": "displacement",
                "unit": "m", "layout": "window" if mean.ndim == 2 else "steps",
                "logstd_offset": 0., "min_logstd": None, "max_logstd": None,
                **contract}
    return SequenceResult("s", uncertainty_mean_raw=mean, uncertainty_target_raw=target,
                          uncertainty_logstd_raw=logstd, uncertainty_valid_raw=valid,
                          uncertainty=metadata)


def _checkpoint_evidence(model, epoch):
    """Unit fixture for the serialized contract; not a real training receipt."""
    model._ipb_checkpoint_epoch = epoch
    model._ipb_uncertainty_checkpoint_evidence = {
        "completed_epoch": epoch, "epoch_optimizer_updates": 2,
        "total_optimizer_updates": 2 * (epoch + 1)}
    model._ipb_uncertainty_checkpoint_loss = {
        "loss": model.loss_name, "loss_kwargs": dict(model.loss_kwargs)}


@pytest.mark.parametrize("target,steps", [("displacement", 0), ("multi_displacement", 2),
                                        ("frame_velocity", 0)])
def test_raw_task_layout_coordinates_and_xyz_to_xy(target, steps):
    spec = InputSpec(window=21, dims=3, frame="gravity_yaw_local", target=target,
                     output_steps=steps)
    predictor = _predictor(spec)
    seq = make_sequence(duration=.6, seed=17)
    view = SequenceView(seq, predictor.view_cfg)
    result = predictor.predict_view(view, epoch=12)
    assert not result.skipped
    np.testing.assert_array_equal(result.uncertainty_target_raw, view.targets(result.starts))
    assert result.uncertainty["unit"] == ("m" if "displacement" in target else "m/s")
    assert result.uncertainty["layout"] == spec.output_layout
    assert result.uncertainty_valid_raw.shape == (len(result.starts), spec.num_outputs)
    metrics = result.compute_metrics(dims=3)
    mean = result.uncertainty_mean_raw.reshape(-1, 3)
    truth = result.uncertainty_target_raw.reshape(-1, 3)
    sigma = result.uncertainty_logstd_raw.reshape(-1, 3)
    valid = result.uncertainty_valid_raw.reshape(-1)
    xyz = gaussian_uncertainty_metrics(mean, truth, sigma, valid)
    xy = gaussian_uncertainty_metrics(mean[:, :2], truth[:, :2], sigma[:, :2], valid)
    for key, value in xyz.items():
        assert metrics[key] == pytest.approx(value)
        assert metrics["xy_" + key] == pytest.approx(xy[key])
    # Displacement units and overlaps differ from world velocity reconstruction.
    assert result.uncertainty_mean_raw.shape != result.vel_pred.shape or not np.allclose(
        result.uncertainty_mean_raw, result.vel_pred)


def test_bridge_preserves_every_legacy_prediction_value():
    spec = InputSpec(window=21, dims=3, frame="gravity_yaw_local", target="displacement")
    predictor = _predictor(spec)
    seq = make_sequence(duration=.6, seed=2)
    with_bridge = predictor.predict_sequence(seq, epoch=12)
    predictor.model.raw_uncertainty_contract = None
    legacy = predictor.predict_sequence(seq, epoch=12)
    for key in ("t", "t_window", "starts", "pos_pred", "pos_gt", "pos_oracle", "vel_pred",
                "vel_target", "logstd", "window_valid", "window_valid_input",
                "window_valid_target"):
        np.testing.assert_array_equal(getattr(with_bridge, key), getattr(legacy, key))
    metrics = legacy.compute_metrics(dims=3)
    assert metrics["uncertainty_task_available"] == 0
    assert math.isnan(metrics["uncertainty_task_nll"])
    assert legacy.uncertainty["reason"] == "unsupported_model_contract"


def test_tlio_phase_requires_actual_training_or_checkpoint_evidence():
    spec = InputSpec(window=200, dims=3, target="displacement")
    model = TLIOResNet(spec, group_sizes=(1, 1, 1, 1), base_plane=4, fc_dim=8,
                       trans_planes=4, dropout=0.)
    model.set_loss("nll_detach_then_nll", switch_epoch=9, min_logstd=math.log(.001))
    assert uncertainty_contract(model, spec, 9)["reason"].startswith("missing_completed")
    _checkpoint_evidence(model, 8)
    assert not uncertainty_contract(model, spec, 8)["head_trained"]
    assert not uncertainty_contract(model, spec, 9)["head_trained"]
    _checkpoint_evidence(model, 9)
    assert uncertainty_contract(model, spec, 9)["head_trained"]
    assert not uncertainty_contract(model, spec, 10)["head_trained"]
    model._ipb_trained_epoch = 10
    model._ipb_epoch_optimizer_updates = 2
    model._ipb_optimizer_updates = 22
    assert uncertainty_contract(model, spec, 10)["training_evidence_source"] == "completed_optimizer_epoch"
    model.set_loss("mse")
    assert uncertainty_contract(model, spec, 10)["reason"] == "unsupported_or_nonprobabilistic_scientific_loss"


def test_meio_first_blend_epoch_does_not_train_sigma():
    spec = InputSpec(window=20, dims=3, target="displacement")
    model = MEIONet(spec, feature_dim=8, layers=2, head_layers=0, dropout=0., head_dropout=0.)
    model.set_loss("meio_multiexit", switch_epoch=9, warmup_epochs=2)
    _checkpoint_evidence(model, 10)
    assert not uncertainty_contract(model, spec, 9)["head_trained"]
    assert uncertainty_contract(model, spec, 10)["head_trained"]


def test_dive_sigma_offset_and_untrained_phase_are_explicit():
    spec = InputSpec(window=200, dims=3, target="velocity_at_end",
                     frame="gravity_yaw_local", extra_inputs=("orientation",))
    model = DIVE(spec, group_sizes=(1, 1, 1, 1), base_plane=4, fc_dim=8,
                 trans_planes=4, dropout=0.)
    model.set_loss("dive_mse_then_nll", switch_epoch=10, offset=1e-7)
    _checkpoint_evidence(model, 10)
    assert not uncertainty_contract(model, spec, 9)["head_trained"]
    contract = uncertainty_contract(model, spec, 10)
    assert contract["head_trained"] and contract["logstd_offset"] == 1e-7
    assert contract["unit"] == "m/s" and contract["min_logstd"] is None


@pytest.mark.parametrize("key,value", [("frame", "body"), ("unit", "km/h"),
                                     ("layout", "steps"), ("training_start_epoch", True)])
def test_explicit_extension_hook_cannot_silently_change_frame_units_or_layout(key, value):
    spec = InputSpec(window=21, dims=3, target="displacement")
    model = RawGaussian(spec)
    declared = model.raw_uncertainty_contract(12)
    declared[key] = value
    model.raw_uncertainty_contract = lambda epoch: declared
    with pytest.raises(ValueError):
        uncertainty_contract(model, spec, 12)


def test_eqnio_uses_canonical_distribution_not_world_marginals():
    spec = InputSpec(window=200, dims=3, frame="gravity_world", target="displacement")
    model = EqNIOTLIO(spec, hidden=8, depth=1, frame_kernel=4,
                       group_sizes=(1, 1, 1, 1), base_plane=4, fc_dim=8, trans_planes=4,
                       dropout=0.)
    model.set_loss("eqnio_canonical_nll", switch_epoch=9, min_logstd=None)
    _checkpoint_evidence(model, 9)

    def known_distribution(self, imu):
        count = len(imu)
        a = math.sqrt(.5)
        frame = imu.new_tensor([[a, -a], [a, a]]).expand(count, 2, 2)
        mean = imu.new_tensor([1., 2., 3.]).expand(count, 3)
        sigma = imu.new_tensor([math.log(.1), math.log(1.), math.log(2.)]).expand(count, 3)
        world = torch.cat([torch.einsum("bij,bj->bi", frame, mean[:, :2]), mean[:, 2:]], -1)
        var = torch.exp(2 * sigma)
        cov = torch.zeros(count, 3, 3, device=imu.device)
        cov[:, :2, :2] = torch.einsum("bij,bj,bkj->bik", frame, var[:, :2], frame)
        cov[:, 2, 2] = var[:, 2]
        return {"vel": world, "logstd": .5 * torch.log(torch.diagonal(cov, dim1=1, dim2=2)),
                "cov": cov, "disp_canonical": mean, "logstd_canonical": sigma,
                "frame_columns": frame}

    # Deterministic adapter fixture; no model-reproduction/performance claim.
    model.forward = MethodType(known_distribution, model)
    cfg = get_cfg({"model": "eqnio_tlio", "device": "cpu", "val_batch": 2,
                   "eval_stride": 50, "amp": False})
    predictor = Predictor(cfg, model=model, device=torch.device("cpu"))
    view = SequenceView(make_sequence(duration=1.6, seed=3), predictor.view_cfg)
    result = predictor.predict_view(view)
    assert result.uncertainty["frame"] == "o2_canonical_per_window:gravity_world"
    basis = result.uncertainty_basis_raw
    truth = view.targets(result.starts)
    canonical = np.concatenate([np.einsum("bji,bj->bi", basis, truth[:, :2]), truth[:, 2:]], -1)
    np.testing.assert_array_equal(result.uncertainty_target_raw, canonical)
    expected = gaussian_uncertainty_metrics(result.uncertainty_mean_raw, canonical,
                                            result.uncertainty_logstd_raw)
    assert result.uncertainty_metrics(3)["uncertainty_task_nll"] == pytest.approx(expected["uncertainty_task_nll"])
    raw = predictor.infer(view, epoch=9)
    incorrect = gaussian_uncertainty_metrics(raw["vel"], truth, raw["logstd"])
    assert not np.isclose(expected["uncertainty_task_nll"], incorrect["uncertainty_task_nll"])


def test_effective_loss_floor_offset_does_not_change_raw_arrays():
    raw = np.full((2, 3), -20.)
    result = _result(np.zeros((2, 3)), logstd=raw, logstd_offset=1e-7,
                     min_logstd=math.log(.001))
    metrics = result.uncertainty_metrics(3)
    assert metrics["uncertainty_task_sigma_mean"] == pytest.approx(.001)
    np.testing.assert_array_equal(result.uncertainty_logstd_raw, raw)


def test_isotropic_logstd_is_same_distribution_for_xyz_and_xy():
    result = _result(np.array([[1., 2., 3.], [.5, -.5, 1.]]),
                     logstd=np.full((2, 1), math.log(2.)))
    xyz, xy = result.uncertainty_metrics(3), result.uncertainty_metrics(2)
    assert xyz["uncertainty_task_isotropic"] == xy["uncertainty_task_isotropic"] == 1
    expected = gaussian_uncertainty_metrics(result.uncertainty_mean_raw[:, :2],
                                            result.uncertainty_target_raw[:, :2],
                                            result.uncertainty_logstd_raw)
    assert xy["uncertainty_task_nll"] == expected["uncertainty_task_nll"]


def test_mask_excludes_invalid_raw_rows_without_world_interpolation():
    mean = np.zeros((2, 2, 3));mean[0, 1] = np.nan
    valid = np.array([[True, False], [True, True]])
    result = _result(mean, valid=valid)
    score = result.uncertainty_metrics(3)
    assert score["uncertainty_task_num_rows"] == 3
    assert score["uncertainty_task_nll"] == pytest.approx(1.5 * math.log(2 * math.pi))
    result.uncertainty_valid_raw[0, 1] = True
    with pytest.raises(ValueError, match="finite"):
        result.uncertainty_metrics(3)


@pytest.mark.parametrize("which", ["mask", "sigma", "layout", "dims"])
def test_malformed_raw_contract_rejected(which):
    result = _result(np.zeros((2, 3)))
    if which == "mask":result.uncertainty_valid_raw = np.ones(2, bool)
    elif which == "sigma":result.uncertainty_logstd_raw = np.zeros((2, 2))
    elif which == "layout":result.uncertainty["layout"] = "steps"
    else:result.uncertainty["dims"] = 2
    with pytest.raises(ValueError):result.uncertainty_metrics(3)


def test_empty_or_all_invalid_raw_support_is_unavailable():
    for result in (_result(np.empty((0, 3))),
                   _result(np.zeros((2, 3)), valid=np.zeros((2, 1), bool))):
        metrics = result.uncertainty_metrics(3)
        assert metrics["uncertainty_task_available"] == 0
        assert math.isnan(metrics["uncertainty_task_nll"])
        assert result.uncertainty["reason"] == "no_valid_raw_outputs"


def test_npz_raw_roundtrip_legacy_compat_and_no_prediction_artifact(tmp_path):
    predictor = _predictor(InputSpec(window=21, dims=3, target="displacement"))
    result = predictor.predict_sequence(make_sequence(duration=.6, seed=4), epoch=12)
    result.compute_metrics(dims=3)
    loaded = SequenceResult.load(result.save(tmp_path / "new.npz"))
    for name in ("uncertainty_mean_raw", "uncertainty_target_raw", "uncertainty_logstd_raw",
                 "uncertainty_valid_raw", "vel_pred", "vel_target", "logstd"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
    assert loaded.uncertainty == result.uncertainty
    assert loaded.compute_metrics(dims=3)["uncertainty_task_nll"] == result.metrics["uncertainty_task_nll"]
    with np.load(tmp_path / "new.npz") as archive:
        legacy = {name: archive[name] for name in archive.files if not name.startswith("uncertainty_")}
    meta = json.loads(str(legacy["meta"]));meta.pop("uncertainty")
    legacy["meta"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "legacy.npz", **legacy)
    old = SequenceResult.load(tmp_path / "legacy.npz")
    assert old.uncertainty_mean_raw is None
    assert old.compute_metrics(dims=3)["uncertainty_task_available"] == 0
    run = RunResult([result], cfg={"metric_dims": 3})
    run.save(tmp_path / "report", predictions=False, plots=False)
    assert not (tmp_path / "report" / "predictions").exists()
    report = json.loads((tmp_path / "report" / "metrics.json").read_text())
    assert report["metrics"]["uncertainty_task_available"] == 1
    assert report["uncertainty_support"][result.sequence_id]["unit"] == "m"


def test_checkpoint_epoch_loaded_for_direct_prediction(tmp_path):
    spec = InputSpec(window=200, dims=3, target="displacement")
    args = dict(group_sizes=(1, 1, 1, 1), base_plane=4, fc_dim=8, trans_planes=4, dropout=0.)
    model = TLIOResNet(spec, **args)
    model_cfg = {"name": "tiny_tlio", "arch": "tlio", "args": args,
                 "loss": "nll_detach_then_nll", "loss_kwargs": {"switch_epoch": 9}}
    checkpoint = tmp_path / "trained.pt"
    torch.save({"model": model.state_dict(), "input_spec": spec.to_dict(),
                "model_cfg": model_cfg, "epoch": 9, "cfg": {},
                "training_evidence": {"completed_epoch": 9, "epoch_optimizer_updates": 2,
                                      "total_optimizer_updates": 20}}, checkpoint)
    cfg = get_cfg({"model": str(checkpoint), "device": "cpu", "val_batch": 2, "amp": False})
    loaded = load_model(cfg, torch.device("cpu"))
    assert loaded._ipb_checkpoint_epoch == 9
    predictor = Predictor(cfg, model=loaded, device=torch.device("cpu"))
    result = predictor.predict_sequence(make_sequence(duration=1.2, seed=6))
    assert result.uncertainty["evaluation_epoch"] == 9
    assert result.uncertainty["head_trained"]
    # An override to a later phase cannot invent missing training epochs.
    result = predictor.predict_sequence(make_sequence(duration=1.2, seed=6), epoch=10)
    assert not result.uncertainty["head_trained"]


def test_epoch_without_optimizer_evidence_cannot_certify_head_training():
    spec = InputSpec(window=200, dims=3, target="displacement")
    model = TLIOResNet(spec, group_sizes=(1, 1, 1, 1), base_plane=4, fc_dim=8,
                       trans_planes=4, dropout=0.)
    model.set_loss("nll_detach_then_nll", switch_epoch=9)
    model._ipb_checkpoint_epoch = 10
    assert not uncertainty_contract(model, spec, 10)["head_trained"]
    _checkpoint_evidence(model, 10)
    model._ipb_uncertainty_checkpoint_evidence["epoch_optimizer_updates"] = 0
    assert not uncertainty_contract(model, spec, 10)["head_trained"]
    _checkpoint_evidence(model, 10)
    model._ipb_uncertainty_checkpoint_evidence["completed_epoch"] = 9
    assert not uncertainty_contract(model, spec, 10)["head_trained"]


@pytest.mark.parametrize("override", ["different_loss", "different_loss_kwargs"])
def test_checkpoint_loss_override_does_not_retrain_or_certify_sigma(tmp_path, override):
    spec = InputSpec(window=200, dims=3, target="displacement")
    args = dict(group_sizes=(1, 1, 1, 1), base_plane=4, fc_dim=8, trans_planes=4, dropout=0.)
    model = TLIOResNet(spec, **args)
    model_cfg = {"name": "tiny_tlio", "arch": "tlio", "args": args,
                 "loss": "nll_detach_then_nll", "loss_kwargs": {"switch_epoch": 9}}
    checkpoint = tmp_path / "trained.pt"
    torch.save({"model": model.state_dict(), "input_spec": spec.to_dict(),
                "model_cfg": model_cfg, "epoch": 10, "cfg": {},
                "training_evidence": {"completed_epoch": 10, "epoch_optimizer_updates": 2,
                                      "total_optimizer_updates": 22}}, checkpoint)
    cfg = get_cfg({"model": str(checkpoint), "device": "cpu", "val_batch": 2, "amp": False})
    loaded = load_model(cfg, torch.device("cpu"))
    assert uncertainty_contract(loaded, spec, 10)["head_trained"]
    before = {name: value.clone() for name, value in loaded.state_dict().items()}
    if override == "different_loss":
        loaded.set_loss("gaussian_nll")
    else:
        loaded.set_loss("nll_detach_then_nll", switch_epoch=0)
    contract = uncertainty_contract(loaded, spec, 10)
    assert not contract["head_trained"]
    assert contract["reason"] == "checkpoint_scientific_loss_contract_overridden"
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    # NA is reportable alongside the exact original raw arrays.
    result = _result(np.zeros((2, 3)))
    result.uncertainty = contract
    assert result.uncertainty_metrics(3)["uncertainty_task_available"] == 0
    # Completed subsequent training, rather than loading/overriding, can certify
    # the new loss. This is a contract fixture, not a real scientific training run.
    loaded._ipb_trained_epoch = 11
    loaded._ipb_epoch_optimizer_updates = 2
    loaded._ipb_optimizer_updates = 24
    assert uncertainty_contract(loaded, spec, 11)["head_trained"]
