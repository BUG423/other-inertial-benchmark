"""Independent RIDI implementation, ECCV 2018 §4 and Eq. (1).

Train-only RBF placement SVM -> per-placement epsilon-SVR -> piecewise-linear
device-frame acceleration correction -> double integration. The original method
is planar. XYZ fits an additional real vertical SVR, rather than filling Z with
zero. Bounded offline correction blocks differ from the original asynchronous
sliding implementation. The exact contract and assumptions are in its model card.
Only numeric support-vector tensors enter checkpoints; no sklearn pickle loads.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ...data.views import end_velocity
from ...nn.base import InputSpec, SequenceModel
from ...nn.registry import register_model
from ...utils.geometry import heading_from_quat, quat_to_matrix, rotate_z


PLACEMENT = {"leg": 0, "pocket": 0, "bag": 1, "handbag": 1,
             "handheld": 2, "hand": 2, "body": 3}
_FLOAT_BUFFERS = ("reg_support", "reg_coeff", "reg_bias", "svc_support", "svc_coeff", "svc_bias")
_INDEX_BUFFERS = ("reg_offsets", "svc_offsets", "svc_pairs", "class_values")


def correction_basis(length: int, spacing: int) -> np.ndarray:
    """Piecewise linear device-frame bias values, one knot every 50 samples."""
    nodes = np.unique(np.r_[np.arange(0, length, spacing), length - 1])
    if len(nodes) == 1:
        return np.ones((length, 1))
    return np.stack([np.interp(np.arange(length), nodes, np.eye(len(nodes))[k])
                     for k in range(len(nodes))], axis=1)


def correct_block(acc_world: np.ndarray, rotations: np.ndarray, measurement_idx: np.ndarray,
                  measured_world: np.ndarray, initial_velocity: np.ndarray, dt: float,
                  knot_spacing: int = 50, regularization: float = 0.1) -> tuple:
    """Solve RIDI Eq. (1) in world coordinates (orthogonal residual invariance).

Unknowns are biases in the *device* frame, mapped by each sample's actual rotation.
The integral uses the trapezoid rule, also used for the returned corrected velocity.
No reference position/velocity is accessed during this physical correction.
"""
    n = len(acc_world)
    basis = correction_basis(n, knot_spacing)
    if n == 1:
        return np.asarray(initial_velocity)[None, :], np.zeros((1, 3))
    increment = 0.5 * (acc_world[1:] + acc_world[:-1]) * dt
    raw_velocity = np.vstack([initial_velocity, initial_velocity + np.cumsum(increment, axis=0)])
    if len(measurement_idx) == 0:
        return raw_velocity, np.zeros((n, 3))
    # J[t, output_axis, knot, body_axis] = d velocity_world[t] / d bias_body[knot].
    sample_jac = rotations[:, :, None, :] * basis[:, None, :, None]
    integrated_jac = np.concatenate([
        np.zeros_like(sample_jac[:1]),
        np.cumsum(0.5 * (sample_jac[1:] + sample_jac[:-1]) * dt, axis=0)
    ], axis=0)
    a = integrated_jac[measurement_idx].reshape(-1, basis.shape[1] * 3)
    b = (measured_world - raw_velocity[measurement_idx]).reshape(-1)
    # Augmented least squares remains defined even for short/rank-deficient smoke windows.
    a_aug = np.vstack([a, np.sqrt(regularization) * np.eye(a.shape[1])])
    b_aug = np.r_[b, np.zeros(a.shape[1])]
    nodes = np.linalg.lstsq(a_aug, b_aug, rcond=None)[0].reshape(-1, 3)
    bias_body = basis @ nodes
    corrected = acc_world + np.einsum("nij,nj->ni", rotations, bias_body)
    velocity = np.vstack([initial_velocity, initial_velocity + np.cumsum(
        0.5 * (corrected[1:] + corrected[:-1]) * dt, axis=0)])
    return velocity, bias_body


@register_model("ridi")
class RIDI(SequenceModel):
    """SVM/SVR fitting is calibration, not fictitious neural optimizer updates."""

    def __init__(self, input_spec: InputSpec, train_stride: int = 10,
                 svm_c: float = 10.0, svr_c: float = 10.0, epsilon: float = 0.01,
                 feature_sigma: float = 2.0, target_sigma: float = 30.0,
                 knot_spacing: int = 50, correction_block: int = 1000,
                 regularization: float = 0.1, max_fit_windows: int = 20000,
                 cache_mb: float = 256, seed: int = None) -> None:
        super().__init__(input_spec)
        if input_spec.frame != "gravity_world" or not input_spec.remove_gravity:
            raise ValueError("RIDI needs gravity_world linear acceleration (remove_gravity=true)")
        if input_spec.target != "velocity_at_end" or input_spec.output_layout != "window":
            raise ValueError("RIDI regresses instantaneous velocity_at_end, not average displacement")
        if input_spec.sub_windows:
            raise ValueError("RIDI uses one contiguous IMU history window")
        if input_spec.dims not in (2, 3):
            raise ValueError("RIDI supports its planar reference or explicitly adapted XYZ")
        if min(train_stride, knot_spacing, correction_block) < 1:
            raise ValueError("RIDI stride, knot spacing and correction block must be positive")
        if min(svm_c, svr_c, regularization, cache_mb) <= 0 or epsilon < 0:
            raise ValueError("RIDI C/regularization/cache must be positive; epsilon >= 0")
        if max_fit_windows < 0:
            raise ValueError("max_fit_windows=0 is uncapped; negative values are invalid")
        self.train_stride = int(train_stride)
        self.svm_c, self.svr_c, self.epsilon = float(svm_c), float(svr_c), float(epsilon)
        self.feature_sigma, self.target_sigma = float(feature_sigma), float(target_sigma)
        self.knot_spacing, self.correction_block = int(knot_spacing), int(correction_block)
        self.regularization, self.max_fit_windows = float(regularization), int(max_fit_windows)
        self.cache_mb = float(cache_mb)
        self.seed = None if seed is None else int(seed)
        self.gamma = 1.0 / (input_spec.num_channels * input_spec.window)
        width = input_spec.num_channels * input_spec.window
        for key in _FLOAT_BUFFERS:
            shape = (0, width) if key.endswith("support") else (0,)
            self.register_buffer(key, torch.empty(shape, dtype=torch.float32))
        for key in _INDEX_BUFFERS:
            shape = (0, 2) if key == "svc_pairs" else (0,)
            self.register_buffer(key, torch.empty(shape, dtype=torch.long))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key in _FLOAT_BUFFERS + _INDEX_BUFFERS:
            if prefix + key in state_dict:
                incoming = state_dict[prefix + key]
                self._buffers[key] = torch.empty(incoming.shape, dtype=self._buffers[key].dtype,
                                                 device=self._buffers[key].device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _features(self, view, starts: np.ndarray) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.int64)
        if not len(starts):
            return np.empty((0, self.input_spec.window * self.input_spec.num_channels))
        yaw = heading_from_quat(view.q)
        # Rz(-yaw) R(q) is the minimal-tilt stabilized IMU frame, with world Z up.
        stabilized = np.c_[rotate_z(view.imu[:, :3], -yaw),
                           rotate_z(view.imu[:, 3:6], -yaw)]
        idx = starts[:, None] + np.arange(self.input_spec.window)
        x = stabilized[idx]
        if self.feature_sigma > 0:
            x = gaussian_filter1d(x, self.feature_sigma, axis=1)
        return x.reshape(len(starts), -1)

    def calibrate(self, views, split: str = "train") -> dict:
        if split != "train":
            raise ValueError("RIDI may fit SVM/SVR only on train sequences")
        try:
            from sklearn.svm import SVC, SVR
        except ImportError as exc:
            raise ImportError("RIDI calibration requires the optional scikit-learn dependency") from exc
        prepared = []
        for view in views:
            starts = view.starts(self.train_stride)
            if not len(starts):
                continue
            prepared.append((view, starts))
        total_windows = sum(len(starts) for _, starts in prepared)
        if not total_windows:
            raise ValueError("RIDI train split contains no valid history windows")
        actual_seed = (torch.initial_seed() % (2 ** 32)) if self.seed is None else self.seed
        rng = np.random.default_rng(actual_seed)
        capped = self.max_fit_windows and total_windows > self.max_fit_windows
        if capped and self.max_fit_windows < len(prepared):
            raise ValueError("max_fit_windows must cover at least one window per train sequence")
        features, targets, labels = [], [], []
        remaining = self.max_fit_windows if capped else total_windows
        remaining_population = total_windows
        for sequence_index, (view, all_starts) in enumerate(prepared):
            count = len(all_starts)
            if capped:
                # Reserve one sample for every remaining train sequence; cap *before*
                # materializing 1200-D features, avoiding an unbounded dense allocation.
                reserve = len(prepared) - sequence_index - 1
                proportional = max(1, int(remaining * count / remaining_population))
                count = min(count, remaining - reserve, proportional)
                starts = np.sort(rng.choice(all_starts, count, replace=False))
            else:
                starts = all_starts
            remaining -= count
            remaining_population -= len(all_starts)
            idx = np.arange(len(view.seq))
            world = end_velocity(view.seq.position, view.seq.velocity, idx, view.cfg.rate)
            stabilized = rotate_z(world, -heading_from_quat(view.q))
            if self.target_sigma > 0:
                stabilized = gaussian_filter1d(stabilized, self.target_sigma, axis=0)
            ends = starts + self.input_spec.window - 1
            features.append(self._features(view, starts))
            targets.append(stabilized[ends, :self.input_spec.dims])
            placement = str(view.seq.attrs.get("ridi_placement", view.seq.attrs.get("placement", "unknown")))
            if placement == "unknown":
                placement = str(view.seq.attrs.get("placement", "unknown"))
            labels.append(np.full(len(starts), PLACEMENT.get(placement, 4), dtype=np.int64))
        x, y, label = np.concatenate(features), np.concatenate(targets), np.concatenate(labels)
        classes = np.unique(label)
        regs = []
        for cls in classes:
            for axis in range(self.input_spec.dims):
                reg = SVR(C=self.svr_c, epsilon=self.epsilon, gamma=self.gamma,
                          kernel="rbf", cache_size=self.cache_mb)
                reg.fit(x[label == cls], y[label == cls, axis])
                regs.append(reg)
        pairs, classifiers = [], []
        for a, b in combinations(classes.tolist(), 2):
            keep = (label == a) | (label == b)
            svc = SVC(C=self.svm_c, gamma=self.gamma, kernel="rbf", cache_size=self.cache_mb)
            svc.fit(x[keep], label[keep])
            pairs.append((a, b))
            classifiers.append(svc)
        self._pack(regs, "reg")
        self._pack(classifiers, "svc")
        self._set_buffer("class_values", classes)
        self._set_buffer("svc_pairs", np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
        return {"windows_available": total_windows, "windows_fit": len(x),
                "placement_classes": classes.tolist(), "unknown_placement_windows": int((label == 4).sum()),
                "regressors": len(regs), "pairwise_classifiers": len(classifiers),
                "max_fit_windows": self.max_fit_windows, "sampling_seed": actual_seed,
                "gamma": self.gamma,
                "fidelity": "RBF SVM/SVR and bias LS; bounded offline correction; XYZ adapted"}

    def _set_buffer(self, key: str, value) -> None:
        self._buffers[key] = torch.as_tensor(value, dtype=self._buffers[key].dtype,
                                            device=self._buffers[key].device)

    def _pack(self, estimators: list, prefix: str) -> None:
        width = self.input_spec.window * self.input_spec.num_channels
        support, coefficients, offsets, bias = [], [], [0], []
        for estimator in estimators:
            support.append(estimator.support_vectors_)
            coefficients.append(estimator.dual_coef_[0])
            offsets.append(offsets[-1] + len(estimator.support_vectors_))
            bias.append(float(estimator.intercept_[0]))
        self._set_buffer(prefix + "_support", np.concatenate(support) if support else np.empty((0, width)))
        self._set_buffer(prefix + "_coeff", np.concatenate(coefficients) if coefficients else np.empty(0))
        self._set_buffer(prefix + "_offsets", offsets)
        self._set_buffer(prefix + "_bias", bias)

    def _decisions(self, x: np.ndarray, prefix: str) -> np.ndarray:
        support = getattr(self, prefix + "_support").detach().cpu().numpy().astype(np.float64)
        coefficients = getattr(self, prefix + "_coeff").detach().cpu().numpy()
        offsets = getattr(self, prefix + "_offsets").detach().cpu().numpy()
        bias = getattr(self, prefix + "_bias").detach().cpu().numpy()
        result = np.empty((len(x), len(bias)))
        for k, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            for first in range(0, len(x), 128):
                queries = x[first:first + 128]
                prediction = np.full(len(queries), bias[k], dtype=np.float64)
                for begin in range(lo, hi, 4096):
                    end = min(begin + 4096, hi)
                    s = support[begin:end]
                    distance = np.maximum((queries * queries).sum(1)[:, None] +
                                          (s * s).sum(1)[None, :] - 2 * queries @ s.T, 0)
                    prediction += np.exp(-self.gamma * distance) @ coefficients[begin:end]
                result[first:first + len(queries), k] = prediction
        return result

    def regress(self, x: np.ndarray) -> tuple:
        classes = self.class_values.detach().cpu().numpy()
        if not len(classes):
            raise RuntimeError("RIDI must be calibrated on train before prediction")
        votes = np.zeros((len(x), len(classes)), dtype=np.int64)
        pairs = self.svc_pairs.detach().cpu().numpy()
        decisions = self._decisions(x, "svc")
        for k, (a, b) in enumerate(pairs):
            votes[:, np.flatnonzero(classes == a)[0]] += decisions[:, k] <= 0
            votes[:, np.flatnonzero(classes == b)[0]] += decisions[:, k] > 0
        selected = votes.argmax(1)
        reg = self._decisions(x, "reg").reshape(len(x), len(classes), self.input_spec.dims)
        return classes[selected], reg[np.arange(len(x)), selected]

    def predict_sequence(self, seq, view, starts: np.ndarray) -> tuple:
        starts = np.asarray(starts, dtype=np.int64)
        if not len(starts):
            return view.target_times(starts), np.empty((0, self.input_spec.dims)), {}
        # Dense SVR cues use 50-sample spacing independently of the common evaluation grid.
        cues = view.starts(self.knot_spacing, require_valid=False)
        cues = cues[view.window_valid_input(cues)]
        if not len(cues):
            raise ValueError("RIDI sequence has no valid input history window")
        placement, local = self.regress(self._features(view, cues))
        cue_idx = cues + self.input_spec.window - 1
        # Paper transition detector: >2 placement changes in ten 0.15-s observations.
        cls_starts = view.starts(max(1, round(0.15 * view.cfg.rate)), require_valid=False)
        cls_starts = cls_starts[view.window_valid_input(cls_starts)]
        cls_ids, _ = self.regress(self._features(view, cls_starts))
        cls_idx = cls_starts + self.input_spec.window - 1
        transition = np.zeros(len(cues), dtype=bool)
        for k, endpoint in enumerate(cue_idx):
            stop = int(np.searchsorted(cls_idx, endpoint, side="right"))
            recent = cls_ids[max(0, stop - 10):stop]
            transition[k] = int(np.count_nonzero(np.diff(recent))) > 2
        local[transition] = 0.0
        # The original flat-floor reference fixes vertical regression to zero. XYZ fits it.
        local_xyz = local if local.shape[1] == 3 else np.c_[local, np.zeros(len(local))]
        measured = rotate_z(local_xyz, heading_from_quat(view.q[cue_idx]))
        rotations = quat_to_matrix(view.q)
        acc_world = np.asarray(view.imu[:, 3:6], dtype=np.float64)
        velocity, bias = np.zeros((len(seq), 3)), np.zeros((len(seq), 3))
        initial = np.zeros(3)  # Original RIDI assumes zero velocity at sequence start.
        for lo in range(0, len(seq) - 1, self.correction_block):
            hi = min(lo + self.correction_block + 1, len(seq))
            keep = (cue_idx >= lo) & (cue_idx < hi)
            v, b = correct_block(acc_world[lo:hi], rotations[lo:hi], cue_idx[keep] - lo,
                                 measured[keep], initial, view.cfg.dt,
                                 self.knot_spacing, self.regularization)
            velocity[lo:hi], bias[lo:hi] = v, b
            initial = v[-1]
        ends = starts + self.input_spec.window - 1
        interpolated_placement = placement[np.minimum(np.searchsorted(cue_idx, ends), len(placement) - 1)]
        return view.target_times(starts), velocity[ends, :self.input_spec.dims], {
            "acceleration_bias_body": bias[ends], "placement_class": interpolated_placement,
        }
