"""Exact regression against the private, pinned pre-optimisation source.

Operator-invoked server-only acceptance, outside the default pytest testpaths.
Set IPB_LIE_EVENTS_ORACLE_PATH to the archived, hash-verified historical repo
file and IPB_LIE_EVENTS_TEST_DEVICE to cpu or cuda (see tools/ops_v1/README.md).
Missing evidence/device fails; this suite never silently skips an untested
optimisation. The normal public model tests remain self-contained.
"""

import hashlib
import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch

from inertial_benchmark.cfg import get_cfg
from inertial_benchmark.nn import build_model
from inertial_benchmark.nn.modules import lie_events as current
from inertial_benchmark.models.nio_lie_events import model as model_module


ORACLE_SHA256 = "c8158b700566d8cc7015741099097b6e3f480fcd71383eff13ce4fec10c11518"


@pytest.fixture(scope="module")
def oracle():
    path = Path(os.environ.get("IPB_LIE_EVENTS_ORACLE_PATH", ""))
    assert path.is_file(), "provide the private pinned pre-optimisation oracle"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ORACLE_SHA256
    spec = importlib.util.spec_from_file_location("ipb_lie_events_private_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def device():
    name = os.environ.get("IPB_LIE_EVENTS_TEST_DEVICE", "cpu")
    assert name in ("cpu", "cuda"), "test device must be explicitly cpu or cuda"
    if name == "cuda":
        assert torch.cuda.is_available(), "CUDA comparison requested without a visible GPU"
    return torch.device(name)


def _window(device, dtype, velocities, steps=13, rotating=False):
    velocity = torch.tensor(velocities, dtype=dtype, device=device)
    batch = len(velocity)
    t = torch.arange(steps, dtype=dtype, device=device) * .005
    rotation = torch.eye(3, dtype=dtype, device=device).expand(batch, steps, 3, 3).clone()
    gyro = torch.zeros(batch, steps, 3, dtype=dtype, device=device)
    acc = torch.zeros_like(gyro)
    acc[..., 2] = 9.81
    if rotating:
        phase = torch.arange(batch, dtype=dtype, device=device)[:, None] * .17
        gyro[..., 0] = .3 * torch.sin(2 * t + phase)
        gyro[..., 1] = -.2 * torch.cos(3 * t - phase)
        gyro[..., 2] = .8
        phi = gyro * t[None, :, None]
        rotation = current.so3_exp(phi)
        acc[..., 0] = torch.sin(7 * t + phase)
        acc[..., 1] = -.4 * torch.cos(5 * t - phase)
    return gyro, acc, rotation, velocity


CASES = (
    "no_events", "one_sample", "two_samples", "mixed_xyz", "threshold_boundary",
    "max64", "noncommuting_rotation",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", CASES)
def test_event_counts_pose_every_stack_element_exact(oracle, device, case, dtype):
    threshold, steps, rotating = .1, 13, False
    velocities = [[0., 0., 0.], [1.1, -.4, .7], [-3.4, .9, -1.2]]
    if case == "no_events":
        velocities = [[0., 0., 0.], [0., 0., 0.]]
    elif case in ("one_sample", "two_samples"):
        steps = 1 if case == "one_sample" else 2
    elif case == "threshold_boundary":
        speed = threshold / .005
        velocities = [[math.nextafter(speed, 0.), 0., 0.], [speed, 0., 0.],
                      [math.nextafter(speed, math.inf), 0., 0.]]
        steps = 4
    elif case == "max64":
        velocities, steps, threshold = [[100., 0., 0.], [0., 0., 0.]], 3, .001
    elif case == "noncommuting_rotation":
        rotating, threshold, steps = True, .01, 17
    args = _window(device, dtype, velocities, steps, rotating)
    if case == "threshold_boundary":
        # Use neighbours of the tested dtype, not Python float neighbours that
        # all round to the same float32 value.
        speed = torch.tensor(threshold / .005, dtype=dtype, device=device)
        args[3][:, 0] = torch.stack([
            torch.nextafter(speed, torch.zeros_like(speed)), speed,
            torch.nextafter(speed, torch.full_like(speed, math.inf))])
    before = oracle.lie_events(*args, .005, threshold, bins=11, max_events_per_step=64)
    after = current.lie_events(*args, .005, threshold, bins=11, max_events_per_step=64)
    assert set(before) == set(after)
    for key in before:
        assert torch.equal(after[key], before[key]), (case, dtype, key)
        assert torch.isfinite(after[key]).all()
    if case == "no_events":
        assert int(after["counts"].sum()) == 0
    if case == "max64":
        assert int(after["counts"][0]) == 128
        assert int(after["counts"][1]) == 0


def test_rank_bins_keep_negative_outside_times_and_tie_order(oracle, device):
    # beta is deliberately not clipped; negative/outside timestamps and ties
    # must reach the original argsort/rank-bin rule unchanged.
    times = torch.tensor([1.2, 0., -.03, .0125, .0125, .5],
                         dtype=torch.float64, device=device)
    measurement = torch.arange(36, dtype=torch.float64, device=device).reshape(6, 6)
    polarity = (measurement - 13.) / 7.
    before = oracle.generate_event_stack(times, measurement, polarity, 4)
    after = current.generate_event_stack(times, measurement, polarity, 4)
    assert torch.equal(after, before)
    order = torch.argsort(times)
    index = torch.linspace(0., 3., 6, dtype=torch.float64, device=device).to(torch.int64)
    # The first bin averages the first two ranked events; it is not a time bin.
    assert torch.equal(after[:6, 0], measurement[order][index == 0].mean(0))


def test_unclipped_interpolation_and_pseudo_inputs_exact(oracle, device, monkeypatch):
    """Controlled Exp stress fixture tests metadata transport, not physical accuracy.

    Reversing event translation intentionally creates negative interpolated
    timestamps. Both implementations see the same untouched recursion otherwise.
    Captured pre-bin pseudo/event inputs must match, including stable real ties.
    """
    captures = {"before": [], "after": []}
    for name, module in (("before", oracle), ("after", current)):
        original_exp, original_stack = module.se3_exp, module.generate_event_stack

        def reversed_translation(xi, original=original_exp):
            rotation, translation = original(xi)
            return rotation, -translation

        def capture(times, measurement, polarity, bins, original=original_stack, name=name):
            captures[name].append(tuple(x.detach().clone() for x in
                                        (times, measurement, polarity)))
            return original(times, measurement, polarity, bins)

        monkeypatch.setattr(module, "se3_exp", reversed_translation)
        monkeypatch.setattr(module, "generate_event_stack", capture)
    args = _window(device, torch.float64, [[3., -.2, .7], [0., 0., 0.]], steps=5)
    before = oracle.lie_events(*args, .005, .01, bins=9)
    after = current.lie_events(*args, .005, .01, bins=9)
    assert torch.equal(before["stack"], after["stack"])
    assert len(captures["before"]) == len(captures["after"]) == 2
    assert any(bool((t < 0).any()) for t, _, _ in captures["before"])
    for b, (before_rows, after_rows) in enumerate(zip(captures["before"], captures["after"])):
        for old, new in zip(before_rows, after_rows):
            assert torch.equal(old, new)
        times, measurement, polarity = after_rows
        assert float(times[0]) == 0. and float(times[-1]) == .02
        assert torch.equal(measurement[0], torch.cat([args[0][b, 0], args[1][b, 0]]))
        assert torch.equal(measurement[-1], torch.cat([args[0][b, -1], args[1][b, -1]]))
        assert torch.isfinite(polarity).all()


def _rng_state(device):
    return (torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None)


def _restore_rng(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


@pytest.mark.parametrize("profile", ["nio_lie_events_ronin", "nio_lie_events_ronin_3d",
                                     "nio_lie_events_tlio"])
def test_training_noise_rng_forward_loss_gradient_exact(oracle, device, monkeypatch, profile):
    torch.manual_seed(8127)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(8127)
    model = build_model(get_cfg({"model": profile})).to(device).train()
    # Full scientific window/backbone/threshold/noise remain the actual profile.
    gyro, acc, _, velocity = _window(device, torch.float32,
                                    [[.9, -.2, .6], [-.4, .7, -.5]], steps=200)
    imu = torch.cat([gyro, acc], -1).transpose(1, 2).requires_grad_()
    extra = model.input_spec.dummy_extra(2, device)
    extra["init_velocity"] = velocity[:, :model.input_spec.dims]
    target = torch.full((2,) + model.input_spec.output_shape, .2, device=device)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state = _rng_state(device)
    results = []
    sampled_velocity = []
    for function in (oracle.lie_events, current.lie_events):
        model.load_state_dict(original)
        model.zero_grad(set_to_none=True)
        imu.grad = None
        _restore_rng(state, device)
        def capture_velocity(*args, function=function, **kwargs):
            sampled_velocity.append(args[3].detach().clone())
            return function(*args, **kwargs)

        monkeypatch.setattr(model_module, "lie_events", capture_velocity)
        stack = model.event_stack(imu, extra)
        assert not stack.requires_grad  # the representation remains no_grad
        out = model.backbone(stack)
        loss, _ = model.loss(out, {"target": target, "imu": imu, "extra": extra}, epoch=10)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = {name: None if p.grad is None else p.grad.detach().clone()
                     for name, p in model.named_parameters()}
        assert any(value is not None and bool((value != 0).any()) for value in gradients.values())
        assert all(value is None or bool(torch.isfinite(value).all()) for value in gradients.values())
        assert imu.grad is None
        next_draw_cpu = torch.rand(11)
        next_draw_device = torch.rand(11, device=device)
        results.append((stack.detach().clone(), {k: v.detach().clone() for k, v in out.items()},
                        loss.detach().clone(), gradients, next_draw_cpu, next_draw_device,
                        _rng_state(device)))
    old, new = results
    assert torch.equal(sampled_velocity[0], sampled_velocity[1])
    if model.input_spec.dims == 3:
        assert model.zero_v0_z is False
        assert not torch.equal(sampled_velocity[0][:, 2], velocity[:, 2].double())
    else:
        assert torch.equal(sampled_velocity[0][:, 2], torch.zeros_like(sampled_velocity[0][:, 2]))
    assert torch.equal(old[0], new[0])
    assert old[1].keys() == new[1].keys()
    assert all(torch.equal(old[1][key], new[1][key]) for key in old[1])
    assert torch.equal(old[2], new[2])
    for name in old[3]:
        assert (old[3][name] is None) == (new[3][name] is None)
        if old[3][name] is not None:
            assert torch.equal(old[3][name], new[3][name]), name
    assert torch.equal(old[4], new[4]) and torch.equal(old[5], new[5])
    assert torch.equal(old[6][0], new[6][0])
    if old[6][1] is not None:
        assert torch.equal(old[6][1], new[6][1])
