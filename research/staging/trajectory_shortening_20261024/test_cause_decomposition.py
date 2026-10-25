import json
from pathlib import Path
import tempfile
import unittest

from cause_decomposition import correlation, describe


class CauseDecompositionTests(unittest.TestCase):
    def test_describe_ignores_missing_values(self):
        result = describe([{"x": .8}, {"x": None}, {"x": 1.2}], "x")
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["fraction_below_one"], .5)

    def test_correlation_rejects_constant_or_short_inputs(self):
        self.assertIsNone(correlation([{"a": 1, "b": 2}], "a", "b"))
        self.assertIsNone(correlation([{"a": 1, "b": 2}, {"a": 2, "b": 2}, {"a": 3, "b": 2}], "a", "b"))
        self.assertAlmostEqual(correlation([{"a": 1, "b": 1}, {"a": 2, "b": 2}, {"a": 3, "b": 3}], "a", "b"), 1.0)


if __name__ == "__main__":
    unittest.main()
