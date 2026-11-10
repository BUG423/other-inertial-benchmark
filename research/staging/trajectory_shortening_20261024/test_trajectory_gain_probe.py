"""Analytical controls for trajectory scale, anchoring and misleading lengths."""
import unittest

import numpy as np

from trajectory_gain_probe import readout, scale_trajectory
from inertial_benchmark.engine.predictor import reconstruct


class TrajectoryGainTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(3001) / 200.
        self.anchor = np.array([7., -3., 2.])
        self.target = self.anchor + self.t[:, None] * np.array([[1., 0., 0.]])
        self.valid = np.ones(len(self.t), bool)

    def test_scaling_displacement_equals_reintegrating_scaled_velocity(self):
        times = np.array([.5, 7.5, 14.5])
        velocity = np.array([[.5, .1, 0.], [.9, -.2, .1], [1.1, .2, -.1]])
        baseline = reconstruct(self.t, times, velocity, 100, self.anchor)
        for gain in (.7, 1., 1.2):
            np.testing.assert_allclose(scale_trajectory(baseline, self.anchor, gain),
                reconstruct(self.t, times, velocity * gain, 100, self.anchor), atol=1e-12)
        np.testing.assert_array_equal(scale_trajectory(baseline, self.anchor, 1.2)[100], self.anchor)

    def test_straight_underestimate_has_known_ate_and_ten_second_error(self):
        pred = scale_trajectory(self.target, self.anchor, .9)
        result = readout(pred, self.target, self.valid, 200., self.target)
        self.assertAlmostEqual(result['ate_m'], float(np.sqrt(np.mean((.1*self.t)**2))))
        self.assertAlmostEqual(result['rte_10s_m'], 1.)
        self.assertEqual(result['rte_10s_scaled'], 0.)
        self.assertEqual(result['rte_10s_pairs'], 1001)
        corrected = readout(scale_trajectory(pred, self.anchor, 1/.9),
                            self.target, self.valid, 200., self.target)
        self.assertLess(corrected['ate_m'], 1e-12)
        self.assertLess(corrected['rte_10s_p95_m'], 1e-12)

    def test_unit_length_can_worsen_position_and_relative_errors(self):
        pred = self.anchor + self.t[:, None] * np.array([[0., .5, 0.]])
        before = readout(pred, self.target, self.valid, 200., self.target)
        after = readout(scale_trajectory(pred, self.anchor, 2.),
                        self.target, self.valid, 200., self.target)
        self.assertAlmostEqual(after['plr'], 1.)
        for key in ('ate_m', 'rte_10s_m', 'position_error_p95_m', 'rte_10s_p95_m'):
            self.assertGreater(after[key], before[key])

    def test_invalid_gains_are_rejected(self):
        for gain in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                scale_trajectory(self.target, self.anchor, gain)


if __name__ == '__main__':
    unittest.main()
