"""RIDI scientific correction, train-only fitting and tensor-only reload contracts."""

import numpy as np
import pytest
import torch

from inertial_benchmark.data.views import SequenceView, ViewConfig
from inertial_benchmark.models.ridi.model import RIDI, correct_block
from inertial_benchmark.nn.base import InputSpec
from inertial_benchmark.utils.geometry import quat_from_euler_zyx, quat_to_matrix
from tests.synthetic import make_sequence


def _view(seed=1, placement="handheld"):
    seq = make_sequence(duration=1.5, seed=seed)
    seq.attrs["placement"] = placement
    cfg = ViewConfig(window=200, frame="gravity_world", orientation="reference",
                     remove_gravity=True, target="velocity_at_end", dims=3)
    return SequenceView(seq, cfg)


def _model(**kwargs):
    return RIDI(InputSpec(window=200, frame="gravity_world", remove_gravity=True,
                          target="velocity_at_end", dims=3), **kwargs)


def test_device_bias_correction_uses_actual_rotating_xyz_frame():
    n = 251
    phase = np.linspace(0, 1, n)
    q = quat_from_euler_zyx(phase, .6 * phase, -.4 * phase)
    rotations = quat_to_matrix(q)
    body_bias = np.array([.3, -.2, .7])
    acc = np.einsum("nij,j->ni", rotations, body_bias)
    idx = np.arange(50, n, 50)
    velocity, correction = correct_block(acc, rotations, idx, np.zeros((len(idx), 3)),
                                         np.zeros(3), .005, regularization=1e-10)
    np.testing.assert_allclose(velocity[idx], 0, atol=2e-8)
    assert np.max(np.abs(correction[:, 2])) > .1
    # Biases live in the device frame, not in a fixed world-axis correction.
    corrected_acc = acc + np.einsum("nij,nj->ni", rotations, correction)
    assert np.linalg.norm(corrected_acc) < np.linalg.norm(acc) / 10


def test_validation_fitting_is_rejected_without_accessing_sklearn():
    with pytest.raises(ValueError, match="only on train"):
        _model().calibrate([], split="val")


def test_numeric_rbf_buffers_match_sklearn_decisions():
    from sklearn.svm import SVC, SVR
    rng = np.random.default_rng(8)
    x = rng.normal(size=(16, 1200))
    model = _model()
    reg = SVR(C=10, epsilon=.01, gamma=model.gamma).fit(x, rng.normal(size=16))
    svc = SVC(C=10, gamma=model.gamma).fit(x, np.arange(16) % 2)
    model._pack([reg], "reg")
    model._pack([svc], "svc")
    np.testing.assert_allclose(model._decisions(x, "reg")[:, 0], reg.predict(x), atol=2e-6)
    np.testing.assert_allclose(model._decisions(x, "svc")[:, 0], svc.decision_function(x), atol=2e-6)


def test_train_only_cap_xyz_prediction_and_tensor_checkpoint_reload():
    views = [_view(1, "handheld"), _view(2, "pocket")]
    model = _model(max_fit_windows=12, train_stride=1, seed=71)
    stats = model.calibrate(views)
    assert stats["windows_available"] > stats["windows_fit"] == 12
    assert stats["placement_classes"] == [0, 2]
    assert stats["regressors"] == 6
    assert all(isinstance(value, torch.Tensor) for value in model.state_dict().values())
    view = _view(3)
    starts = view.starts(10)
    prediction = model.predict_sequence(view.seq, view, starts)
    assert prediction[1].shape == (len(starts), 3)
    assert np.isfinite(prediction[1]).all()
    assert np.any(prediction[1][:, 2] != 0)
    clone = _model()
    clone.load_state_dict(model.state_dict())
    repeated = clone.predict_sequence(view.seq, view, starts)
    np.testing.assert_array_equal(prediction[0], repeated[0])
    np.testing.assert_array_equal(prediction[1], repeated[1])
    # Regression and physical correction use IMU + orientation at prediction, not GT motion.
    view.seq.position[:] += np.arange(len(view.seq))[:, None] * 100
    view.seq.velocity[:] = 123
    changed_reference = model.predict_sequence(view.seq, view, starts)
    np.testing.assert_array_equal(prediction[1], changed_reference[1])


def test_empty_classifier_features_and_single_sample_correction():
    model, view = _model(), _view()
    assert model._features(view, np.empty(0, dtype=np.int64)).shape == (0, 1200)
    initial = np.array([.2, -.3, .7])
    velocity, bias = correct_block(np.ones((1, 3)), np.eye(3)[None], np.array([], int),
                                   np.empty((0, 3)), initial, .005)
    np.testing.assert_array_equal(velocity[0], initial)
    np.testing.assert_array_equal(bias, np.zeros((1, 3)))
