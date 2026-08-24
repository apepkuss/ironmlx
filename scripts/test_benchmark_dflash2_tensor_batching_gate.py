import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("benchmark_dflash2_tensor_batching_gate.py")
SPEC = importlib.util.spec_from_file_location("dflash2_tensor_batching_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TensorBatchingGateTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank_in_sorted_values(self) -> None:
        self.assertEqual(MODULE.percentile([3.0, 1.0, 2.0], 0.50), 2.0)
        self.assertEqual(MODULE.percentile([3.0, 1.0, 2.0], 0.95), 3.0)

    def test_summary_uses_whole_batch_throughput(self) -> None:
        summary = MODULE.summarize_batches(
            [
                {"aggregate_tps": 20.0, "wall_s": 2.0},
                {"aggregate_tps": 30.0, "wall_s": 1.0},
                {"aggregate_tps": 25.0, "wall_s": 1.5},
            ]
        )
        self.assertEqual(summary["aggregate_tps_median"], 25.0)
        self.assertEqual(summary["batch_wall_s_median"], 1.5)


if __name__ == "__main__":
    unittest.main()
