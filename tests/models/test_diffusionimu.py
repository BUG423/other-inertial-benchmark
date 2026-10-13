"""DiffusionIMU mathematical contracts; positive u is not Gaussian covariance."""

import pytest
import torch

from inertial_benchmark.nn.base import InputSpec
from inertial_benchmark.models.diffusionimu import model as module

def _model(dims=3, **kwargs):
    return module.DiffusionIMU(InputSpec(window=200, dims=dims),
                                   hidden=16, stem_width=16, **kwargs)


def test_eq22_positive_regularizer_is_not_gaussian_nll():
    pred = torch.tensor([[1., 2., -3.], [4., 0., 2.]], requires_grad=True)
    u = torch.tensor([[2., 3., 4.], [1., 5., 2.]], requires_grad=True)
    alignment = torch.tensor([.5, 99.], requires_grad=True)
    target = torch.tensor([[0., 1., 1.], [4., 1., 3.]])
    out = {"vel": pred, "uncertainty_regularizer_head": u, "alignment_squared_norm": alignment}
    actual, items = module.DiffusionIMUMultiTaskLoss()(out, target, mask=torch.tensor([[1.], [0.]]))
    expected = ((pred[0] - target[0]) ** 2).sum() + .5 * u[0].square().sum() + .5
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(u.grad[0], u.detach()[0])
    assert u.grad[1].abs().sum() == 0
    assert set(items) == {"mse_sum", "uncertainty_regularization", "feature_alignment"}


@pytest.mark.parametrize("dims", [2, 3])
def test_real_xyz_multitask_gradients_all_three_refinement_steps(dims):
    model = _model(dims)
    imu = torch.randn(2, 6, 200)
    out = model(imu)
    assert out["vel"].shape == out["uncertainty_regularizer_head"].shape == (2, dims)
    assert "logstd" not in out and "cov" not in out
    assert (out["uncertainty_regularizer_head"] > 0).all()
    target = torch.tensor([[.5, -.7, 1.2], [-.3, .4, -1.1]])[:, :dims]
    loss, _ = model.loss(out, {"imu": imu, "target": target})
    if dims == 3:
        raw_xyz_gradient = torch.autograd.grad(loss, out["vel"], retain_graph=True)[0]
        assert raw_xyz_gradient[:, 2].abs().sum() > 0
        xy_error = (out["vel"][:, :2] - target[:, :2]).square().sum(-1).mean()
        xy_gradient = torch.autograd.grad(xy_error, out["vel"], retain_graph=True)[0]
        assert xy_gradient[:, :2].abs().sum() > 0 and xy_gradient[:, 2].abs().sum() == 0
        assert target[:, 2].abs().sum() > 0
    loss.backward()
    assert model.velocity_head.weight.grad[dims - 1].abs().sum() > 0
    assert model.uncertainty_head.weight.grad.abs().sum() > 0
    assert model.alpha_logit.grad.abs() > 0
    assert model.alignment_projection.weight.grad.abs().sum() > 0
    assert model.time_embedding.weight.grad.abs().sum() > 0
    for layer in model.refinement:
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    assert torch.isfinite(loss)


def test_eval_noise_replay_exact_checkpoint_reload_and_next_rng_draw(tmp_path):
    model = _model().eval()
    clone = _model().eval()
    checkpoint = tmp_path / "staging-diffusionimu.pt"
    torch.save({"state_dict": model.state_dict(), "input_spec": model.input_spec.to_dict()}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert saved["input_spec"] == model.input_spec.to_dict()
    clone.load_state_dict(saved["state_dict"])
    imu = torch.randn(2, 6, 200)
    torch.manual_seed(981)
    first = model(imu)
    next_first = torch.rand(7)
    torch.manual_seed(981)
    second = clone(imu)
    next_second = torch.rand(7)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert torch.equal(next_first, next_second)
    assert first["diffusion_timestep"].tolist() == [0, 0]
    torch.manual_seed(982)
    third = model(imu)
    assert not torch.equal(first["vel"], third["vel"])


def test_mask_empty_loss_still_has_grad_and_custom_loss_override_is_rejected():
    model = _model()
    imu = torch.randn(2, 6, 200)
    loss, _ = model.loss(model(imu), {"imu": imu, "target": torch.randn(2, 3),
                                    "mask": torch.zeros(2, 1)})
    assert loss == 0
    loss.backward()
    assert model.velocity_head.weight.grad.abs().sum() == 0
    with pytest.raises(ValueError):
        model.set_loss("gaussian_nll")


def test_training_timestep_policy_in_range_and_untrained_head_has_no_std_contract():
    model = _model(diffusion_timesteps=7)
    out = model(torch.randn(2, 6, 200))
    assert ((out["diffusion_timestep"] >= 0) & (out["diffusion_timestep"] < 7)).all()
    assert not hasattr(model, "raw_uncertainty_contract")
    assert "logstd" not in out


def test_paper_default_profile_b2_real_xyz_backward():
    model = module.DiffusionIMU(InputSpec(window=200, dims=3))
    imu = torch.randn(2, 6, 200)
    out = model(imu)
    loss, _ = model.loss(out, {"imu": imu, "target": torch.tensor([[.4, -.3, 1.], [-.8, .2, -1.]])})
    loss.backward()
    assert torch.isfinite(loss) and out["vel"].shape == (2, 3)
    assert model.velocity_head.weight.grad[2].abs().sum() > 0
    assert len(model.encoder) == 8 and len(model.refinement) == 3


@pytest.mark.parametrize("changes", [{"dims": 1}, {"target": "displacement"},
                                     {"frame": "body"}])
def test_reject_incompatible_task_contract(changes):
    with pytest.raises(ValueError):
        module.DiffusionIMU(InputSpec(**({"window": 200, "dims": 3} | changes)))


@pytest.mark.parametrize("profile,dims", [("diffusionimu", 2), ("diffusionimu_3d", 3)])
def test_registered_profile_loss_and_auxiliary_head_contract(profile, dims):
    from inertial_benchmark.cfg import get_cfg, load_model_cfg
    from inertial_benchmark.engine.predictor import uncertainty_contract
    from inertial_benchmark.nn.losses import build_loss
    from inertial_benchmark.nn.registry import build_model, list_models

    cfg = get_cfg({"model": profile, "recipe": "unified"})
    described = load_model_cfg(profile)
    model = build_model(cfg)
    assert "diffusionimu" in list_models() and type(model) is module.DiffusionIMU
    assert model.input_spec == InputSpec(window=200, dims=dims)
    assert model.loss_name == "diffusionimu_multitask" and model.loss_kwargs == {}
    assert isinstance(build_loss(model.loss_name), module.DiffusionIMUMultiTaskLoss)
    assert described["args"] == {
        "hidden": 128, "decoder_steps": 3, "diffusion_timesteps": 100,
        "dropout": .2, "attention_heads": 4, "local_radius": 3,
        "stem_width": 64, "initial_alpha": .5, "evaluation_timestep": 0}
    assert len(model.encoder) == 8 and len(model.refinement) == 3
    assert "official" not in described["recipes"] and described["code"] is None
    assert not uncertainty_contract(model, model.input_spec, 12)["supported"]
    assert not hasattr(model, "raw_uncertainty_contract")
    with pytest.raises(ValueError, match="paper multi-task loss"):
        build_model(get_cfg({"model": profile, "loss": "gaussian_nll"}))


@pytest.mark.parametrize("profile,dims", [("diffusionimu", 2), ("diffusionimu_3d", 3)])
def test_integrated_default_forward_loss_gradients_and_rng_match_staging(profile, dims):
    import importlib.util
    from pathlib import Path

    from inertial_benchmark.cfg import get_cfg, load_model_cfg
    from inertial_benchmark.nn.registry import build_model

    path = Path(__file__).resolve().parents[2] / "research/staging/algorithm_ports/diffusionimu/model.py"
    spec = importlib.util.spec_from_file_location("diffusionimu_accepted_staging_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    integrated = build_model(get_cfg({"model": profile}))
    original = oracle.DiffusionIMUDraft(integrated.input_spec, **load_model_cfg(profile)["args"])
    original.load_state_dict(integrated.state_dict(), strict=True)
    imu = torch.randn(2, 6, 200)
    target = torch.tensor([[.5, -.7, 1.2], [-.3, .4, -1.1]])[:, :dims]
    torch.manual_seed(612)
    actual = integrated(imu)
    actual_loss, actual_items = integrated.loss(actual, {"imu": imu, "target": target}, 12)
    actual_loss.backward()
    actual_next = torch.rand(7)
    torch.manual_seed(612)
    expected = original(imu)
    expected_loss, expected_items = original.loss(expected, {"imu": imu, "target": target}, 12)
    expected_loss.backward()
    assert torch.equal(actual_next, torch.rand(7))
    assert actual.keys() == expected.keys() and actual_items.keys() == expected_items.keys()
    assert torch.equal(actual_loss, expected_loss)
    assert "logstd" not in actual and "cov" not in actual
    for key in actual:
        assert torch.equal(actual[key], expected[key])
    for key in actual_items:
        assert torch.equal(actual_items[key], expected_items[key])
    assert dict(integrated.named_parameters()).keys() == dict(original.named_parameters()).keys()
    for key, parameter in integrated.named_parameters():
        other = dict(original.named_parameters())[key]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, other.grad)
