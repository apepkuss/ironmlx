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

    @staticmethod
    def interpretation_report(*, strict_gate: bool) -> dict:
        ratio = {"estimate": 1.0, "ci95_low": 0.99, "ci95_high": 1.01}
        return {
            "aggregate": {
                "greedy": {"actor_over_direct_generation_tps": ratio},
                "sampled": {"actor_over_direct_generation_tps": ratio},
            },
            "gates": {
                "token_exact_and_all_valid": True,
                "strict_2pct_per_cell": strict_gate,
                "safety_5pct_per_cell": True,
                "peak_memory_no_more_than_10pct": True,
            },
        }

    def test_interpretation_reports_passing_qualification(self) -> None:
        lines = MODULE.interpretation_lines(
            self.interpretation_report(strict_gate=True)
        )
        rendered = "\n".join(lines)
        self.assertIn("strict 2% per-cell parity gate passes", rendered)
        self.assertIn("Performance qualification passes all declared gates", rendered)
        self.assertNotIn("Keep the GenerationStream route", rendered)

    def test_interpretation_reports_failed_gate(self) -> None:
        lines = MODULE.interpretation_lines(
            self.interpretation_report(strict_gate=False)
        )
        rendered = "\n".join(lines)
        self.assertIn("strict 2% per-cell parity gate fails", rendered)
        self.assertIn("strict_2pct_per_cell", rendered)
        self.assertIn("Keep the GenerationStream route", rendered)


if __name__ == "__main__":
    unittest.main()
