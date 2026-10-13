"""Server-only acceptance proposals; importing this file is not a PASS receipt.

Run explicitly outside default testpaths after a separate CPU-budget approval.
Tiny hidden widths are mathematical fixtures; default-width cases remain marked.
"""

from __future__ import annotations

import copy
import inspect
import io
import math

import pytest
import torch
from torch.nn import functional as F

from inertial_benchmark.nn.base import InputSpec
from research.staging.algorithm_ports.motiontransformer.model import (
    MotionTransformerAdaptation,
    MotionTransformerCore,
    PAPER_LOSS_WEIGHTS,
    absolute_distance,
    domain_step,
    frozen_discriminators,
    integrate_polar,
    squared_distance,
)


def inputs(source_batch=2, target_batch=3):
    generator = torch.Generator().manual_seed(1729)
    return (torch.randn(source_batch, 200, 6, generator=generator),
            torch.randn(target_batch, 200, 6, generator=generator),
            torch.tensor([[1.1, -0.3], [0.7, 0.2]])[:source_batch])


def tiny_core(label_dims=2):
    return MotionTransformerCore(hidden=8, label_dims=label_dims)


def adaptation(dims=3, hidden=8):
    return MotionTransformerAdaptation(InputSpec(window=400, rate=200, dims=dims), hidden=hidden)


def test_norm_reductions_have_independent_hand_computed_values():
    a = torch.tensor([[1., -2.], [3., -4.]])
    b = torch.zeros_like(a)
    assert squared_distance(a, b).item() == 15
    assert absolute_distance(a, b).item() == 5


def test_paper_onehot_domain_is_time_constant_and_distinct():
    source, _, _ = inputs()
    source_theta = MotionTransformerCore.domain_vector(source, 0)
    target_theta = MotionTransformerCore.domain_vector(source, 1)
    assert torch.equal(source_theta[..., 0], torch.ones(2, 200))
    assert torch.equal(source_theta[..., 1], torch.zeros(2, 200))
    assert torch.equal(source_theta.flip(-1), target_theta)


def test_original_polar_path_chains_predicted_heading_without_later_gt():
    labels = torch.tensor([[[1., math.pi / 2], [2., 0.], [1., -math.pi / 2]]], dtype=torch.float64)
    pos, heading = integrate_polar(labels, torch.tensor([[2., 3.]], dtype=torch.float64),
                                   torch.zeros(1, dtype=torch.float64))
    expected = torch.tensor([[[2., 3.], [2., 4.], [2., 6.], [3., 6.]]], dtype=torch.float64)
    torch.testing.assert_close(pos, expected, rtol=0, atol=1e-14)
    torch.testing.assert_close(heading, torch.tensor([[math.pi / 2, math.pi / 2, 0.]], dtype=torch.float64))


def test_default_paper_width_is_real_two_second_six_axis_forward():
    source, target, _ = inputs(target_batch=2)
    core = MotionTransformerCore()
    assert core.hidden == 128  # Explicit architecture assumption, not an author parameter match.
    assert core.predict(source, 0).shape == (2, 2)
    assert core.translate(target, 1, 0).shape == (2, 200, 6)


def test_default_width_b2_has_all_five_finite_losses_and_real_gradients():
    core = MotionTransformerCore()
    source, target, labels = inputs(target_batch=2)
    with frozen_discriminators(core):
        loss, terms = core.generator_objective(source, labels, target)
        loss.backward()
    assert all(torch.isfinite(value) for value in terms.values())
    assert len(terms) == 5 and torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in core.generator_parameters())
    assert core.predictor.readout.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("domain", [-1, 2])
def test_bad_domains_are_rejected(domain):
    source, _, _ = inputs()
    with pytest.raises(ValueError, match="domain"):
        tiny_core().encode(source, domain)


@pytest.mark.parametrize("shape", [(2, 199, 6), (2, 200, 9), (0, 200, 6)])
def test_no_short_window_extra_axes_or_empty_paper_input(shape):
    with pytest.raises(ValueError, match="200,6"):
        tiny_core().predict(torch.zeros(shape), 0)


def test_unit_latent_noise_cannot_be_silently_disabled():
    with pytest.raises(ValueError, match="unit normal"):
        MotionTransformerCore(noise_std=0)


def test_discriminator_objective_is_both_domains_with_detached_generation():
    core = tiny_core()
    source, target, _ = inputs()
    actual = core.discriminator_loss(source, target)
    with torch.no_grad():
        fake_target = core.translate(source, 0, 1)
        fake_source = core.translate(target, 1, 0)
        expected = source.new_zeros(())
        for index, real, fake in ((0, source, fake_source), (1, target, fake_target)):
            real_logits, fake_logits = core.discriminators[index](real), core.discriminators[index](fake)
            expected += F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            expected += F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert all(p.grad is None for p in core.generator_parameters())
    assert all(p.grad is not None for p in core.discriminator_parameters())


def test_every_joint_term_matches_equations_with_replayed_latent_noise():
    core = tiny_core()
    source, target, labels = inputs()
    torch.manual_seed(991)
    state = torch.random.get_rng_state()
    total, terms = core.generator_objective(source, labels, target)
    torch.random.set_rng_state(state)
    zs, zt = core.encode(source, 0), core.encode(target, 1)
    fake_t, fake_s = core.generators[1](zs), core.generators[0](zt)
    translated_t, translated_s = core.encode(fake_t, 1), core.encode(fake_s, 0)
    logit_t, logit_s = core.discriminators[1](fake_t), core.discriminators[0](fake_s)
    expected = {
        "gan": -F.binary_cross_entropy_with_logits(logit_t, torch.zeros_like(logit_t))
               - F.binary_cross_entropy_with_logits(logit_s, torch.zeros_like(logit_s)),
        "reconstruction": squared_distance(core.reconstruct(zs, 0), source)
                          + squared_distance(core.reconstruct(zt, 1), target),
        "prediction": squared_distance(core.predictor(zs), labels)
                      + squared_distance(core.predictor(translated_t), labels),
        "cycle": absolute_distance(core.generators[0](translated_t), source)
                 + absolute_distance(core.generators[1](translated_s), target),
        "perceptual": squared_distance(zs, translated_t) + squared_distance(zt, translated_s),
    }
    assert set(terms) == {"gan", "reconstruction", "prediction", "cycle", "perceptual"}
    for name in expected:
        torch.testing.assert_close(terms[name], expected[name], rtol=0, atol=0)
    w_ae, w_pred, w_cycle, w_perceptual = PAPER_LOSS_WEIGHTS
    reference = expected["gan"] + w_ae * expected["reconstruction"] + w_pred * expected["prediction"]
    reference = reference + w_cycle * expected["cycle"]
    reference = reference + w_perceptual * expected["perceptual"]
    torch.testing.assert_close(total, reference, rtol=0, atol=0)


def test_all_generator_branches_receive_joint_gradients_and_discs_are_frozen():
    core = tiny_core()
    source, target, labels = inputs()
    with frozen_discriminators(core):
        loss, _ = core.generator_objective(source, labels, target)
        loss.backward()
    for module in (core.encoder, *core.generators, core.decoder, core.predictor):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())
    assert all(p.grad is None and p.requires_grad for p in core.discriminator_parameters())


def test_unpaired_target_batch_sizes_and_order_do_not_require_alignment():
    core = tiny_core()
    source, target, labels = inputs(source_batch=2, target_batch=3)
    loss, _ = core.generator_objective(source, labels, target)
    assert torch.isfinite(loss)
    torch.testing.assert_close(core.discriminator_loss(source, target),
                               core.discriminator_loss(source, target.flip(0)), rtol=1e-6, atol=1e-6)
    assert "target_labels" not in inspect.signature(core.generator_objective).parameters
    with pytest.raises(TypeError):
        core.generator_objective(source, labels, target, target_labels=torch.randn(3, 2))


def test_domain_step_really_updates_disjoint_optimizers_and_all_modules():
    core = tiny_core()
    source, target, labels = inputs()
    before = {name: p.detach().clone() for name, p in core.named_parameters()}
    generator = torch.optim.Adam(core.generator_parameters(), lr=1e-4)
    discriminator = torch.optim.Adam(core.discriminator_parameters(), lr=1e-4)
    result = domain_step(core, source, labels, target, generator, discriminator)
    assert set(result) == {"total", "discriminator", "gan", "reconstruction", "prediction", "cycle", "perceptual"}
    assert all(value.ndim == 0 and torch.isfinite(value) for value in result.values())
    for prefix in ("encoder.", "generators.0.", "generators.1.", "decoder.", "predictor.",
                   "discriminators.0.", "discriminators.1."):
        assert any(not torch.equal(before[name], p) for name, p in core.named_parameters()
                   if name.startswith(prefix))
    for optimizer in (generator, discriminator):
        assert all(state["step"].item() == 1 for state in optimizer.state.values())
    assert all(p.grad is None for p in core.discriminator_parameters())


def test_wrong_optimizer_partition_fails_before_mutating_any_weight():
    core = tiny_core()
    source, target, labels = inputs()
    before = copy.deepcopy(core.state_dict())
    wrong = torch.optim.Adam(core.parameters(), lr=1e-4)
    discriminator = torch.optim.Adam(core.discriminator_parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="exactly the disjoint"):
        domain_step(core, source, labels, target, wrong, discriminator)
    assert all(torch.equal(before[name], value) for name, value in core.state_dict().items())


def test_two_stage_checkpoint_resumes_optimizer_and_rng_exactly():
    core = tiny_core()
    source, target, labels = inputs()
    generator = torch.optim.Adam(core.generator_parameters(), lr=1e-4)
    discriminator = torch.optim.Adam(core.discriminator_parameters(), lr=1e-4)
    domain_step(core, source, labels, target, generator, discriminator)
    payload = io.BytesIO()
    torch.save({"model": core.state_dict(), "generator_optimizer": generator.state_dict(),
                "discriminator_optimizer": discriminator.state_dict(),
                "rng": torch.random.get_rng_state()}, payload)
    reference = domain_step(core, source, labels, target, generator, discriminator)
    reference_next = torch.randn(9)
    restored = tiny_core()
    restored_g = torch.optim.Adam(restored.generator_parameters(), lr=1e-4)
    restored_d = torch.optim.Adam(restored.discriminator_parameters(), lr=1e-4)
    payload.seek(0)
    checkpoint = torch.load(payload, weights_only=True)
    restored.load_state_dict(checkpoint["model"])
    restored_g.load_state_dict(checkpoint["generator_optimizer"])
    restored_d.load_state_dict(checkpoint["discriminator_optimizer"])
    torch.random.set_rng_state(checkpoint["rng"])
    replay = domain_step(restored, source, labels, target, restored_g, restored_d)
    assert torch.equal(reference_next, torch.randn(9))
    assert all(torch.equal(reference[name], replay[name]) for name in reference)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in core.state_dict().items())
    for optimizer in (generator, discriminator, restored_g, restored_d):
        assert all(state["step"].item() == 2 for state in optimizer.state.values())


def test_autoencoder_rng_replay_exact_but_fresh_noise_changes_objective():
    core = tiny_core()
    source, target, labels = inputs()
    torch.manual_seed(801)
    state = torch.random.get_rng_state()
    first, terms_first = core.generator_objective(source, labels, target)
    next_first = torch.randn(7)
    torch.random.set_rng_state(state)
    second, terms_second = core.generator_objective(source, labels, target)
    next_second = torch.randn(7)
    assert torch.equal(first, second) and torch.equal(next_first, next_second)
    assert all(torch.equal(terms_first[k], terms_second[k]) for k in terms_first)
    _, fresh = core.generator_objective(source, labels, target)
    assert not torch.equal(terms_first["reconstruction"], fresh["reconstruction"])


def test_paper_inference_needs_no_rng_and_state_reload_is_exact():
    core = tiny_core().eval()
    source, _, _ = inputs()
    buffer = io.BytesIO()
    torch.save(core.state_dict(), buffer)
    loaded = tiny_core().eval()
    buffer.seek(0)
    loaded.load_state_dict(torch.load(buffer, weights_only=True))
    before = torch.random.get_rng_state()
    with torch.no_grad():
        original, restored = core.predict(source, 1), loaded.predict(source, 1)
    assert torch.equal(original, restored)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("dims", [2, 3])
def test_ipb_adapter_preserves_raw_channel_order_rate_and_duration(dims):
    model = adaptation(dims)
    imu = torch.arange(6, dtype=torch.float32)[None, :, None].expand(2, 6, 400)
    prepared = model.prepare(imu)
    assert prepared.shape == (2, 200, 6)
    assert torch.equal(prepared[0, 0], torch.tensor([3., 4., 5., 0., 1., 2.]))
    out = model(imu)
    assert out["vel"].shape == (2, dims)
    expected_xy = out["speed"][:, None] * torch.stack([out["heading"].cos(), out["heading"].sin()], -1)
    assert torch.equal(out["vel"][:, :2], expected_xy)
    if dims == 3:
        assert torch.equal(out["vel"][:, 2], out["vertical"])


def test_xyz_vertical_is_real_independent_trainable_target_with_five_losses():
    model = adaptation(3)
    # Engineering tensors, not a downloaded data fixture or an accuracy result.
    batch = {"imu": torch.randn(2, 6, 400), "target_domain_imu": torch.randn(3, 6, 400),
             "target": torch.tensor([[.3, -.4, 1.7], [-.8, .2, -2.1]])}
    assert torch.equal(model.source_labels(batch["target"])[:, 2], batch["target"][:, 2])
    loss, terms = model.loss(model(batch["imu"]), batch)
    assert len(terms) == 5 and torch.isfinite(loss)
    loss.backward()
    assert model.core.predictor.readout.weight.grad[2].abs().sum() > 0
    assert model.core.predictor.readout.bias.grad[2].abs() > 0
    assert all(p.grad is None for p in model.core.discriminator_parameters())


def test_xyz_adapter_state_spec_loss_raw_velocity_and_rng_reload_are_exact():
    model = adaptation().eval()
    imu = torch.randn(2, 6, 400)
    payload = io.BytesIO()
    torch.save({"state": model.state_dict(), "input_spec": model.input_spec.to_dict(),
                "loss": model.loss_name}, payload)
    payload.seek(0)
    checkpoint = torch.load(payload, weights_only=True)
    restored = MotionTransformerAdaptation(InputSpec.from_dict(checkpoint["input_spec"]), hidden=8).eval()
    restored.load_state_dict(checkpoint["state"])
    restored.set_loss(checkpoint["loss"])
    assert restored.input_spec == model.input_spec and restored.loss_name == model.loss_name
    before = torch.random.get_rng_state()
    with torch.no_grad():
        original, recovered = model(imu), restored(imu)
    assert set(original) == set(recovered)
    assert all(torch.equal(original[name], recovered[name]) for name in original)
    assert torch.equal(before, torch.random.get_rng_state())


def test_single_domain_training_refuses_to_masquerade_as_motiontransformer():
    model = adaptation()
    imu = torch.randn(2, 6, 400)
    with pytest.raises(ValueError, match="unpaired unlabelled"):
        model.loss(model(imu), {"imu": imu, "target": torch.zeros(2, 3)})
    with pytest.raises(ValueError, match="single-domain MSE"):
        model.set_loss("mse")


@pytest.mark.parametrize("spec", [InputSpec(window=200, rate=200, dims=3),
                                   InputSpec(window=400, rate=100, dims=3),
                                   InputSpec(window=400, rate=200, dims=3, frame="body")])
def test_adapter_rejects_hidden_scientific_window_rate_frame_changes(spec):
    with pytest.raises(ValueError):
        MotionTransformerAdaptation(spec, hidden=8)
