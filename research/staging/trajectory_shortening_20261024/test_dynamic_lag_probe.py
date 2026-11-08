"""Analytical checks for the dynamics stratification."""
import unittest

import numpy as np

from dynamic_lag_probe import bins, correlation


class DynamicLagTests(unittest.TestCase):
    def test_bins_use_fixed_edges_and_clip(self):
        np.testing.assert_array_equal(bins([-1., .2, .5, 1.5], [.2, .5, 1.]), [0, 1, 2, 3])

    def test_constant_or_short_vectors_have_undefined_correlation(self):
        self.assertIsNone(correlation([1., 1., 1.], [1., 2., 3.]))
        self.assertIsNone(correlation([1., 2.], [1., 2.]))

    def test_correlation_uses_only_finite_pairs(self):
        self.assertAlmostEqual(correlation([1., 2., np.nan, 4.], [2., 4., 8., 8.]), 1.)


if __name__ == '__main__':
    unittest.main()
