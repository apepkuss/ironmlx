#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_dflash2_batched_q8_q16_gate.py")
SPEC = importlib.util.spec_from_file_location("batched_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def report(block_size: int, elapsed: int):
    records = []
    for batch_width in (2, 4):
        for replicate in range(5):
            records.append(
                {
                    "mode": "greedy",
                    "context_tokens": 2048,
                    "batch_width": batch_width,
                    "replicate": replicate,
                    "block_size": block_size,
                    "full_wall_us": elapsed,
                    "generation_wall_us": elapsed,
                    "token_ids": [[1, 2, 3]] * batch_width,
                    "tensor_batch_windows": 1,
                }
            )
    return {
        "schema_version": 1,
        "frozen_p2_sha": "f" * 40,
        "block_size": block_size,
        "max_new_tokens": 3,
        "records": records,
    }


class BatchedGateTests(unittest.TestCase):
    def test_constant_twenty_percent_win_continues(self):
        result = gate.analyze(
            report(8, 120_000),
            report(16, 100_000),
            report(8, 120_000),
            bootstrap_samples=1000,
        )
        self.assertTrue(result["gates"]["continue_p4"])
        self.assertAlmostEqual(
            result["aggregate"]["generation_q16_over_q8_bracket"]["estimate"], 1.2
        )

    def test_equal_performance_stops(self):
        result = gate.analyze(
            report(8, 100_000),
            report(16, 100_000),
            report(8, 100_000),
            bootstrap_samples=1000,
        )
        self.assertFalse(result["gates"]["continue_p4"])


if __name__ == "__main__":
    unittest.main()
