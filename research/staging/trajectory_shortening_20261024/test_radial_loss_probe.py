"""Analytical controls, executed with NumPy/framework on the remote Agent."""
import unittest

import numpy as np

from radial_loss_probe import gain_controls, radial_statistics


class RadialLossTests(unittest.TestCase):
    def test_collinear_underfit_needs_equal_mse_and_norm_gain(self):
        target = np.array([[1., 0., 0.], [2., 0., 0.]])
        prediction = .5 * target
        self.assertEqual(gain_controls(prediction, target), dict(vector_mse=2., norm_sum=2.))
        result = radial_statistics(prediction, target)
        self.assertAlmostEqual(result['directional_mse'], 0)
        self.assertAlmostEqual(result['oracle_norm_relative_l2'], 0)
        self.assertAlmostEqual(result['fraction_below_directional_optimum'], 1)

    def test_ambiguous_direction_mean_is_shorter_but_norm_restoration_hurts(self):
        target = np.array([[.5, np.sqrt(.75), 0.], [.5, -np.sqrt(.75), 0.]])
        prediction = np.array([[.5, 0., 0.], [.5, 0., 0.]])
        gain = gain_controls(prediction, target)
        self.assertAlmostEqual(gain['vector_mse'], 1)
        self.assertAlmostEqual(gain['norm_sum'], 2)
        result = radial_statistics(prediction, target)
        self.assertAlmostEqual(result['mean_prediction_direction_residual_mps'], 0)
        self.assertAlmostEqual(result['oracle_norm_worse_window_fraction'], 1)
        self.assertGreater(result['oracle_norm_relative_l2'], result['relative_l2'])

    def test_collinear_amplitude_uncertainty_does_not_create_norm_gap(self):
        target = np.array([[.5, 0., 0.], [1.5, 0., 0.]])
        prediction = np.array([[1., 0., 0.], [1., 0., 0.]])
        self.assertEqual(gain_controls(prediction, target), dict(vector_mse=1., norm_sum=1.))
        self.assertAlmostEqual(radial_statistics(prediction, target)['speed_ratio'], 1)

    def test_opposite_direction_projects_to_zero_and_zero_predictions_are_counted(self):
        prediction = np.array([[-1., 0., 0.], [0., 0., 0.]])
        target = np.array([[1., 0., 0.], [1., 0., 0.]])
        result = radial_statistics(prediction, target)
        self.assertEqual(result['windows'], 2)
        self.assertEqual(result['usable_windows'], 1)
        self.assertAlmostEqual(result['oracle_projection_relative_l2'], 1)
        self.assertEqual(radial_statistics(prediction[1:], target[1:])['usable_windows'], 0)


if __name__ == '__main__':
    unittest.main()
