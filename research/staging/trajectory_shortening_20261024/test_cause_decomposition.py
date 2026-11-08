import json
from pathlib import Path
import tempfile
import unittest

from cause_decomposition import correlation, describe, direction_sensitivity


class CauseDecompositionTests(unittest.TestCase):
    def test_describe_ignores_missing_values(self):
        result = describe([{"x": .8}, {"x": None}, {"x": 1.2}], "x")
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["fraction_below_one"], .5)

    def test_correlation_rejects_constant_or_short_inputs(self):
        self.assertIsNone(correlation([{"a": 1, "b": 2}], "a", "b"))
        self.assertIsNone(correlation([{"a": 1, "b": 2}, {"a": 2, "b": 2}, {"a": 3, "b": 2}], "a", "b"))
        self.assertAlmostEqual(correlation([{"a": 1, "b": 1}, {"a": 2, "b": 2}, {"a": 3, "b": 3}], "a", "b"), 1.0)

    def test_group_offsets_can_reverse_the_pooled_association(self):
        rows = []
        for profile, speed, direction in (("a", .8, 10), ("b", .1, 40)):
            rows.extend({"profile": profile, "dataset": "same", "seed": seed,
                         "speed_ratio": speed + seed * .1, "dir_err_mean": direction + seed}
                        for seed in range(3))
        result = direction_sensitivity(rows)
        self.assertLess(result["all"]["correlation"], 0)
        self.assertEqual(result["within_profile_dataset"]["groups"], 2)
        self.assertAlmostEqual(result["within_profile_dataset"]["correlation"], 1)

    def test_tail_association_is_reported_separately(self):
        rows = [{"speed_ratio": speed, "dir_err_mean": direction}
                for speed, direction in ((.8, 10), (.9, 11), (1., 12), (.01, 45))]
        result = direction_sensitivity(rows)
        self.assertLess(result["all"]["correlation"], 0)
        self.assertEqual(result["direction_lt45"]["n"], 3)
        self.assertAlmostEqual(result["direction_lt45"]["correlation"], 1)

    def test_missing_pairs_and_duplicate_seeds_do_not_create_replication(self):
        rows = [{"profile": "a", "dataset": "same", "seed": seed,
                 "speed_ratio": speed, "dir_err_mean": direction}
                for seed, speed, direction in ((1, .8, 10), (1, .9, 11), (2, 1., 12),
                                               (3, float("nan"), 13), (4, .9, None))]
        result = direction_sensitivity(rows)
        self.assertEqual(result["all"]["n"], 3)
        self.assertEqual(result["within_profile_dataset"]["groups"], 0)
        self.assertIsNone(result["within_profile_dataset"]["correlation"])


if __name__ == "__main__":
    unittest.main()
