#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_dflash2_q8_q16_qualification.py")
SPEC = importlib.util.spec_from_file_location("qualification", SCRIPT)
assert SPEC and SPEC.loader
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def record(mode: str, context: int, replicate: int, block: int, elapsed: int):
    tokens = [1, 2, 3]
    return {
        "mode": mode,
        "context_tokens": context,
        "replicate": replicate,
        "block_size": block,
        "order_in_pair": block == 16,
        "wall_total_us": elapsed + 1_000,
        "token_ids": tokens,
        "metrics": {
            "generated_tokens": len(tokens),
            "generation_us": elapsed,
            "acceptance_rate": 0.75,
        },
    }


class QualificationAnalysisTests(unittest.TestCase):
    def test_bootstrap_ci_is_deterministic_and_contains_constant_ratio(self):
        first = qualification.bootstrap_geomean_ci([1.2] * 7, samples=1000, seed=9)
        second = qualification.bootstrap_geomean_ci([1.2] * 7, samples=1000, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 1.2)
        self.assertAlmostEqual(first["ci95_low"], 1.2)
        self.assertAlmostEqual(first["ci95_high"], 1.2)

    def test_stratified_bootstrap_keeps_cells_equally_weighted(self):
        result = qualification.bootstrap_stratified_geomean_ci(
            [[4.0, 4.0], [0.25] * 8], samples=1000, seed=9
        )
        self.assertAlmostEqual(result["estimate"], 1.0)
        self.assertAlmostEqual(result["ci95_low"], 1.0)
        self.assertAlmostEqual(result["ci95_high"], 1.0)

    def test_complete_faster_q16_matrix_promotes_default(self):
        records = []
        for mode in ("greedy", "sampled"):
            for context in (2048, 8192, 32768):
                for replicate in range(5):
                    records.append(record(mode, context, replicate, 8, 120_000))
                    records.append(record(mode, context, replicate, 16, 100_000))
        report = qualification.analyze(
            {
                "schema_version": 1,
                "frozen_p2_sha": "f" * 40,
                "execution_scope": "B1",
                "target_dir": "/local/target",
                "draft_dir": "/local/draft",
                "max_new_tokens": 3,
                "records": records,
            },
            bootstrap_samples=1000,
        )
        self.assertTrue(report["correctness"]["all_q8_q16_tokens_exact"])
        self.assertTrue(report["gates"]["promote_q16_default"])

    def test_incomplete_pair_is_rejected(self):
        raw = {
            "schema_version": 1,
            "frozen_p2_sha": "f" * 40,
            "execution_scope": "B1",
            "target_dir": "/local/target",
            "draft_dir": "/local/draft",
            "max_new_tokens": 3,
            "records": [record("greedy", 2048, 0, 8, 100_000)],
        }
        with self.assertRaisesRegex(ValueError, "incomplete"):
            qualification.analyze(raw, bootstrap_samples=1000)


if __name__ == "__main__":
    unittest.main()
