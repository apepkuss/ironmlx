#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_b1_path_qualification.py")
SPEC = importlib.util.spec_from_file_location("b1_analysis", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class B1AnalysisTests(unittest.TestCase):
    def test_geometric_mean(self) -> None:
        self.assertAlmostEqual(MODULE.geometric_mean([1.0, 4.0]), 2.0)

    def test_constant_bootstrap_interval(self) -> None:
        result = MODULE.bootstrap_geomean_ci([1.2] * 7, samples=100, seed=7)
        self.assertAlmostEqual(result["estimate"], 1.2)
        self.assertAlmostEqual(result["ci95_low"], 1.2)
        self.assertAlmostEqual(result["ci95_high"], 1.2)

    def test_percentile_interpolates(self) -> None:
        self.assertAlmostEqual(MODULE.percentile([1.0, 3.0], 0.25), 1.5)


if __name__ == "__main__":
    unittest.main()
