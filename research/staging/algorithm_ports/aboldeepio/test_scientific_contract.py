"""Explicit server-only scientific staging tests; not an active benchmark suite."""

import importlib.util
import io
import math
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "staging_aboldeepio", Path(__file__).with_name("model.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inference_inputs(n=10, batch=2):
    t = torch.arange((n + 1) * 3, dtype=torch.float32).reshape(1, n + 1, 3) / 20
    acc = t.repeat(batch, 1, 1) + torch.tensor([1.0, -0.5, 9.81])
    gyro = t.repeat(batch, 1, 1) / 10
    dt = torch.full((batch, n + 1, 1), 5_000_000, dtype=torch.int64)
    dt[:, 3, 0] += 2  # synthetic nonuniform ns perturbation, before the floating cast
    return acc, gyro, dt


@pytest.mark.parametrize("n", [10, 20])
def test_reported_full_widths_independent_three_streams_and_two_scalar_sequences(n):
    model = module.AbolDeepIOOriginalScalar(n).eval()
    for stack, channels, widths in (
        (model.acc_stream, 3, (100, 150, 100)),
        (model.gyro_stream, 3, (100, 150, 100)),
        (model.time_stream, 1, (100,)),
        (model.fusion, 300, (150, 100, 50)),
    ):
        for layer, dropout, width in zip(stack.layers, stack.dropouts, widths):
            assert layer.input_size == channels and layer.hidden_size == width
            assert layer.num_layers == 1 and not layer.bidirectional and layer.batch_first
            assert dropout.p == 0.25
            channels = width
        assert len(stack.layers) == len(widths)
    assert set(map(id, model.acc_stream.parameters())).isdisjoint(
        set(map(id, model.gyro_stream.parameters())))
    assert model.head.in_features == 50 and model.head.out_features == 2
    output = model(*inference_inputs(n))
    assert set(output) == {"path_length", "orientation_chord"}
    assert all(value.shape == (2, n) and torch.isfinite(value).all() for value in output.values())
    assert model.acc_noise.variance == 0.2 and model.gyro_noise.variance == 0.02


def test_true_nanosecond_jitter_large_epoch_and_frozen_lookahead_alignment():
    epoch = 1_690_000_000_000_000_000
    increments = torch.tensor([[11, 13, 17, 19, 23, 29], [31, 37, 41, 43, 47, 53]])
    clock = torch.cat((torch.zeros(2, 1, dtype=torch.int64), increments.cumsum(1)), dim=1) + epoch
    torch.testing.assert_close(module.delta_t_nanoseconds(clock), increments, rtol=0, atol=0)
    acc = torch.arange(2 * 7 * 3, dtype=torch.float64).reshape(2, 7, 3)
    gyro = acc / 10
    prepared = module.prepare_original_window(acc, gyro, clock, start=2, window_n=2)
    torch.testing.assert_close(prepared["acc"], acc[:, 2:5], rtol=0, atol=0)
    torch.testing.assert_close(prepared["gyro"], gyro[:, 2:5], rtol=0, atol=0)
    torch.testing.assert_close(prepared["delta_t_ns"].squeeze(-1), increments[:, 1:4],
                              rtol=0, atol=0)
    torch.testing.assert_close(prepared["target_timestamp_ns"], clock[:, 2:4], rtol=0, atol=0)
    torch.testing.assert_close(prepared["observation_timestamp_ns"], clock[:, 2:5], rtol=0, atol=0)
    torch.testing.assert_close(prepared["lookahead_ns"], increments[:, 2:4], rtol=0, atol=0)
    assert prepared["lookahead_ns"][0, -1] == 19


@pytest.mark.parametrize("order", ["max_then_min", "min_then_max"])
def test_int64_clock_order_and_difference_overflow_are_both_rejected(order):
    limits = torch.iinfo(torch.int64)
    clock = ([limits.max, limits.min] if order == "max_then_min"
             else [limits.min, limits.max])
    timestamps = torch.tensor([clock], dtype=torch.int64)
    if order == "max_then_min":
        # Positive wrapped subtraction does NOT make the original clock monotonic.
        assert (timestamps[:, 1:] - timestamps[:, :-1]).item() == 1
        assert timestamps[0, 1] < timestamps[0, 0]
        expected = "original timestamp"
    else:
        # Actual order increases, but the positive elapsed interval exceeds int64.
        assert timestamps[0, 1] > timestamps[0, 0]
        assert (timestamps[:, 1:] - timestamps[:, :-1]).item() == -1
        expected = "differences.*overflow"
    with pytest.raises(ValueError, match=expected):
        module.delta_t_nanoseconds(timestamps)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_dt_branch_retains_actual_nanoseconds_without_nominal_or_seconds_fill(dtype):
    model = module.AbolDeepIOOriginalScalar().eval()
    acc, gyro, dt = inference_inputs()
    dt = dt.to(dtype)
    captured = []
    hook = model.time_stream.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].clone()))
    model(acc, gyro, dt)
    hook.remove()
    torch.testing.assert_close(captured[0], dt.to(acc.dtype), rtol=0, atol=0)
    assert captured[0][0, 3, 0] == 5_000_002
    assert captured[0][0, 2, 0] != captured[0][0, 3, 0]


def test_one_future_sample_changes_only_last_supervised_output():
    torch.manual_seed(1840)
    # FP64 removes rounding of a small propagated change through the deep stacks;
    # this is a scientific causality test, not a new default training recipe.
    model = module.AbolDeepIOOriginalScalar().double().eval()
    acc, gyro, dt = inference_inputs()
    acc, gyro = acc.double(), gyro.double()
    changed = acc.clone()
    changed[:, -1] += 10
    a, b = model(acc, gyro, dt), model(changed, gyro, dt)
    for key in a:
        torch.testing.assert_close(a[key][:, :-1], b[key][:, :-1], rtol=0, atol=0)
    assert any(not torch.equal(a[key][:, -1], b[key][:, -1]) for key in a)


@pytest.mark.parametrize("variance", [0.2, 0.02])
def test_noise_uses_sqrt_of_variance_and_training_only_rng_draw(variance):
    noise = module.TrainingGaussianNoise(variance).train()
    values = torch.zeros(128, 11, 3, dtype=torch.float64)
    torch.manual_seed(1904)
    expected = torch.randn_like(values) * math.sqrt(variance)
    next_draw = torch.rand(4)
    torch.manual_seed(1904)
    actual = noise(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(4), next_draw, rtol=0, atol=0)
    assert abs(actual.var(unbiased=False).item() - variance) < variance * 0.1
    noise.eval()
    state = torch.get_rng_state().clone()
    torch.testing.assert_close(noise(values), values, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), state)


def test_noise_order_acc_then_gyro_and_no_dt_noise():
    model = module.AbolDeepIOOriginalScalar().train()
    acc, gyro, dt = inference_inputs()
    torch.manual_seed(1153)
    expected_acc = acc + torch.randn_like(acc) * math.sqrt(0.2)
    expected_gyro = gyro + torch.randn_like(gyro) * math.sqrt(0.02)
    next_draw = torch.rand(3)
    torch.manual_seed(1153)
    actual_acc, actual_gyro = model.acc_noise(acc), model.gyro_noise(gyro)
    torch.testing.assert_close(actual_acc, expected_acc, rtol=0, atol=0)
    torch.testing.assert_close(actual_gyro, expected_gyro, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(3), next_draw, rtol=0, atol=0)
    assert torch.equal(dt, inference_inputs()[2])


def test_prefix_path_is_not_signed_displacement_or_endpoint_norm():
    p = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0]]])
    q = torch.tensor([[[1.0, 0, 0, 0]]]).repeat(1, 3, 1)
    output = module.scalar_prefix_targets(p, q)
    torch.testing.assert_close(output["path_length"], torch.tensor([[1.0, 2.0]]), rtol=0, atol=0)
    assert torch.linalg.vector_norm(p[:, -1] - p[:, 0]) == 0
    opposite = module.scalar_prefix_targets(-p, q)
    assert torch.equal(output["path_length"], opposite["path_length"])
    assert not torch.equal(p[:, 1] - p[:, 0], -p[:, 1] + p[:, 0])


def test_quaternion_chord_loses_axis_and_is_not_so3_angle():
    angle = 1.0
    half = angle / 2
    qx = torch.tensor([[[1.0, 0, 0, 0], [math.cos(half), math.sin(half), 0, 0]]],
                      dtype=torch.float64)
    qy = torch.tensor([[[1.0, 0, 0, 0], [math.cos(half), 0, math.sin(half), 0]]],
                      dtype=torch.float64)
    p = torch.zeros(1, 2, 3, dtype=torch.float64)
    x, y = module.scalar_prefix_targets(p, qx), module.scalar_prefix_targets(p, qy)
    torch.testing.assert_close(x["orientation_chord"], y["orientation_chord"], rtol=0, atol=0)
    torch.testing.assert_close(x["orientation_chord"], torch.tensor([[2 * math.sin(angle / 4)]],
                              dtype=torch.float64), rtol=1e-14, atol=1e-14)
    assert abs(x["orientation_chord"].item() - angle) > 0.4


def test_raw_sign_flip_and_explicit_continuity_are_distinct_scientific_protocols():
    q = torch.tensor([[[1.0, 0, 0, 0], [-1.0, 0, 0, 0], [1.0, 0, 0, 0]]])
    p = torch.zeros(1, 3, 3)
    raw = module.scalar_prefix_targets(p, q, hemisphere="raw")
    continuous = module.scalar_prefix_targets(p, q, hemisphere="continuous")
    torch.testing.assert_close(raw["orientation_chord"], torch.tensor([[2.0, 4.0]]), rtol=0, atol=0)
    torch.testing.assert_close(continuous["orientation_chord"], torch.zeros(1, 2), rtol=0, atol=0)


def test_weighted_l1_is_sum_and_many_to_one_is_final_supervised_row():
    output = {"path_length": torch.tensor([[1.0, 2.0], [4.0, 1.0]], requires_grad=True),
              "orientation_chord": torch.tensor([[0.5, -1.0], [2.0, 4.0]], requires_grad=True)}
    target = {"path_length": torch.tensor([[0.0, 1.0], [1.0, 3.0]]),
              "orientation_chord": torch.zeros(2, 2)}
    loss = module.weighted_l1_sum(output, target, lambda_length=2, lambda_orientation=0.5)
    torch.testing.assert_close(loss, torch.tensor(17.75), rtol=0, atol=0)
    final = module.weighted_l1_sum(output, target, lambda_length=2, lambda_orientation=0.5,
                                   supervision="many_to_one")
    torch.testing.assert_close(final, torch.tensor(8.5), rtol=0, atol=0)
    duplicated = {k: v.repeat(2, 1) for k, v in output.items()}
    duplicated_target = {k: v.repeat(2, 1) for k, v in target.items()}
    torch.testing.assert_close(module.weighted_l1_sum(duplicated, duplicated_target,
                              lambda_length=2, lambda_orientation=0.5), 2 * loss, rtol=0, atol=0)
    loss.backward()
    torch.testing.assert_close(output["path_length"].grad, torch.tensor([[2.0, 2.0], [2.0, -2.0]]),
                              rtol=0, atol=0)
    torch.testing.assert_close(output["orientation_chord"].grad,
                              torch.tensor([[0.5, -0.5], [0.5, 0.5]]), rtol=0, atol=0)


def test_full_width_finite_gradients_without_adding_fake_time_normalization():
    model = module.AbolDeepIOOriginalScalar().train()
    acc, gyro, dt = inference_inputs()
    output = model(acc, gyro, dt)
    targets = {key: torch.ones_like(value) * 2 for key, value in output.items()}
    loss = module.weighted_l1_sum(output, targets)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for block in (model.acc_stream, model.gyro_stream, model.fusion, model.head):
        assert any(parameter.grad.abs().sum() > 0 for parameter in block.parameters())
    # Raw 5e6 ns can saturate the time gates. Finite (possibly zero) time gradients
    # are recorded as a boundary, NOT silently fixed using a seconds-input model.
    assert all(torch.isfinite(p.grad).all() for p in model.time_stream.parameters())


@pytest.mark.parametrize("training", [False, True])
def test_state_reload_and_replayed_rng_preserve_both_scalar_sequences_and_next_draw(training):
    model = module.AbolDeepIOOriginalScalar()
    model.train(training)
    inputs = inference_inputs()
    torch.manual_seed(7319)
    state = torch.get_rng_state().clone()
    expected = model(*inputs)
    next_draw = torch.rand(7)
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "training": model.training, "rng": state}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    reloaded = module.AbolDeepIOOriginalScalar()
    reloaded.load_state_dict(saved["model"], strict=True)
    reloaded.train(saved["training"])
    torch.set_rng_state(saved["rng"])
    actual = reloaded(*inputs)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(7), next_draw, rtol=0, atol=0)


def test_eval_consumes_no_random_draws_and_linear_head_has_no_hidden_positivity_clamp():
    model = module.AbolDeepIOOriginalScalar().eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([-2.0, -3.0]))
    state = torch.get_rng_state().clone()
    actual = model(*inference_inputs())
    assert torch.equal(torch.get_rng_state(), state)
    assert (actual["path_length"] == -2).all() and (actual["orientation_chord"] == -3).all()


def test_gt_pose_is_not_an_inference_argument_or_a_trajectory_assembly_api():
    model = module.AbolDeepIOOriginalScalar().eval()
    with pytest.raises(TypeError):
        model(*inference_inputs(), position=torch.zeros(2, 10, 3))
    assert not hasattr(model, "integrate_using_gt_direction")


@pytest.mark.parametrize("defect", ["missing_future", "clock_float32", "zero_dt", "nan_dt"])
def test_forward_refuses_missing_future_or_invalid_true_clock(defect):
    model = module.AbolDeepIOOriginalScalar().eval()
    acc, gyro, dt = inference_inputs()
    if defect == "missing_future":
        acc, gyro, dt = acc[:, :-1], gyro[:, :-1], dt[:, :-1]
    elif defect == "clock_float32":
        dt = dt.float()
    elif defect == "zero_dt":
        dt[0, 0] = 0
    else:
        dt = dt.double()
        dt[0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(acc, gyro, dt)


@pytest.mark.parametrize("defect", ["start_zero", "no_future", "clock_float32", "nonincreasing"])
def test_time_adapter_refuses_missing_context_or_invalid_clock(defect):
    acc = torch.zeros(1, 15, 3)
    clock = torch.arange(15, dtype=torch.int64).unsqueeze(0) * 5_000_000
    start, n = 1, 10
    if defect == "start_zero":
        start = 0
    elif defect == "no_future":
        start = 5
    elif defect == "clock_float32":
        clock = clock.float()
    else:
        clock[0, 5] = clock[0, 4]
    with pytest.raises(ValueError):
        module.prepare_original_window(acc, acc, clock, start=start, window_n=n)


@pytest.mark.parametrize("defect", ["short", "nan", "nonunit", "wrong_hemisphere"])
def test_labels_require_explicit_finite_reference_and_sign_protocol(defect):
    p = torch.zeros(1, 3, 3)
    q = torch.tensor([[[1.0, 0, 0, 0]]]).repeat(1, 3, 1)
    hemisphere = "raw"
    if defect == "short":
        p, q = p[:, :1], q[:, :1]
    elif defect == "nan":
        p[0, 1, 0] = float("nan")
    elif defect == "nonunit":
        q *= 2
    else:
        hemisphere = "automatic_unreported"
    with pytest.raises(ValueError):
        module.scalar_prefix_targets(p, q, hemisphere=hemisphere)


@pytest.mark.parametrize("defect", ["weights", "variant", "task_shape", "nan"])
def test_scalar_loss_contract_rejects_unreported_or_nonfinite_inputs(defect):
    output = {"path_length": torch.zeros(2, 10), "orientation_chord": torch.zeros(2, 10)}
    target = {key: value.clone() for key, value in output.items()}
    kwargs = {}
    if defect == "weights":
        kwargs["lambda_length"] = -1
    elif defect == "variant":
        kwargs["supervision"] = "mse"
    elif defect == "task_shape":
        output["orientation_chord"] = torch.zeros(2, 1)
        target["orientation_chord"] = torch.zeros(2, 1)
    else:
        output["path_length"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        module.weighted_l1_sum(output, target, **kwargs)
