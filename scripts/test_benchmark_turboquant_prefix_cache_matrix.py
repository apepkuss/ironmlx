import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("benchmark_turboquant_prefix_cache_matrix.py")


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkTurboQuantPrefixCacheMatrixTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_matrix_contains_baseline_turboquant_prefix_and_combo(self):
        variants = self.module.build_matrix("k3v4")

        self.assertEqual(
            [variant.name for variant in variants],
            [
                "baseline_dense",
                "turboquant_only",
                "prefix_cache_only",
                "turboquant_prefix_cache",
            ],
        )
        self.assertEqual(variants[1].kv_quant, "k3v4")
        self.assertFalse(variants[1].prefix_cache)
        self.assertIsNone(variants[2].kv_quant)
        self.assertTrue(variants[2].prefix_cache)
        self.assertEqual(variants[3].kv_quant, "k3v4")
        self.assertTrue(variants[3].prefix_cache)

    def test_combo_serve_command_enables_turboquant_and_prefix_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.module.MatrixConfig(
                root=Path("/repo"),
                model_dir=Path("/models/qwen"),
                out_root=Path(tmp),
                serve_bin=Path("/repo/target/release/ironmlx"),
                iron_bench_bin=Path("/repo/target/release/iron-bench"),
                prompt_lens=(128,),
                max_tokens=8,
                runs=3,
                ssd_prefix_cache_max_gb=10.0,
            )
            combo = self.module.build_matrix("k3v4")[-1]
            cmd = self.module.build_serve_command(
                cfg,
                combo,
                port=19003,
                variant_dir=Path(tmp) / "turboquant_prefix_cache",
            )

        self.assertEqual(cmd[:2], ["/repo/target/release/ironmlx", "serve"])
        self.assertIn("--model", cmd)
        self.assertIn("/models/qwen", cmd)
        self.assertIn("--port", cmd)
        self.assertIn("19003", cmd)
        self.assertIn("--kv-quant", cmd)
        self.assertEqual(cmd[cmd.index("--kv-quant") + 1], "k3v4")
        self.assertIn("--paged-prefix-cache-dir", cmd)
        cache_dir = cmd[cmd.index("--paged-prefix-cache-dir") + 1]
        self.assertTrue(cache_dir.endswith("turboquant_prefix_cache/prefix-cache"))
        self.assertIn("--ssd-prefix-cache-max-gb", cmd)
        self.assertEqual(cmd[cmd.index("--ssd-prefix-cache-max-gb") + 1], "10")

    def test_bench_command_uses_prefix_probe_json_output_shape(self):
        cfg = self.module.MatrixConfig(
            root=Path("/repo"),
            model_dir=Path("/models/qwen"),
            out_root=Path("/tmp/out"),
            serve_bin=Path("/repo/target/release/ironmlx"),
            iron_bench_bin=Path("/repo/target/release/iron-bench"),
            prompt_lens=(256,),
            max_tokens=16,
            runs=4,
        )

        cmd = self.module.build_bench_command(cfg, port=19001, prompt_len=256)

        self.assertEqual(cmd[0], "/repo/target/release/iron-bench")
        self.assertIn("--target", cmd)
        self.assertIn("ironmlx=http://127.0.0.1:19001", cmd)
        self.assertIn("--prefix-cache-probe", cmd)
        self.assertIn("--format", cmd)
        self.assertEqual(cmd[cmd.index("--format") + 1], "json")
        self.assertEqual(cmd[cmd.index("--runs") + 1], "4")

    def test_run_plan_can_filter_to_combo_variant_only(self):
        cfg = self.module.MatrixConfig(
            root=Path("/repo"),
            model_dir=Path("/models/qwen"),
            out_root=Path("/tmp/out"),
            serve_bin=Path("/repo/target/release/ironmlx"),
            iron_bench_bin=Path("/repo/target/release/iron-bench"),
            prompt_lens=(128,),
            max_tokens=8,
            runs=3,
            variant_names=("turboquant_prefix_cache",),
        )

        plan = self.module.build_run_plan(cfg)

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["variant"].name, "turboquant_prefix_cache")
        self.assertIn("--kv-quant", plan[0]["serve_cmd"])
        self.assertIn("--paged-prefix-cache-dir", plan[0]["serve_cmd"])

    def test_summarize_payload_splits_cold_and_warm_probe_runs(self):
        variant = self.module.build_matrix("k3v4")[-1]
        payload = {
            "metadata": {"prefix_cache_probe": True},
            "raw_runs": [
                {
                    "target": "ironmlx",
                    "pp_target": 128,
                    "tg_target": 8,
                    "run_idx": 0,
                    "ttft_ms": 100.0,
                    "tg_tps": 20.0,
                    "tpot_ms": 50.0,
                    "pp_tps": 1280.0,
                    "e2e_s": 0.5,
                    "prompt_tokens_local": 128,
                    "prompt_tokens_server": 130,
                    "completion_tokens_server": 8,
                    "cached_tokens": None,
                    "finish_reason": "stop",
                    "prefix_cache_probe_phase": "cold_or_miss_candidate",
                },
                {
                    "target": "ironmlx",
                    "pp_target": 128,
                    "tg_target": 8,
                    "run_idx": 1,
                    "ttft_ms": 40.0,
                    "tg_tps": 25.0,
                    "tpot_ms": 40.0,
                    "pp_tps": 3200.0,
                    "e2e_s": 0.25,
                    "prompt_tokens_local": 128,
                    "prompt_tokens_server": 130,
                    "completion_tokens_server": 8,
                    "cached_tokens": 120,
                    "finish_reason": "stop",
                    "prefix_cache_probe_phase": "warm_hit_candidate",
                },
                {
                    "target": "ironmlx",
                    "pp_target": 128,
                    "tg_target": 8,
                    "run_idx": 2,
                    "ttft_ms": 50.0,
                    "tg_tps": 30.0,
                    "tpot_ms": 33.0,
                    "pp_tps": 2560.0,
                    "e2e_s": 0.3,
                    "prompt_tokens_local": 128,
                    "prompt_tokens_server": 130,
                    "completion_tokens_server": 8,
                    "cached_tokens": 120,
                    "finish_reason": "stop",
                    "prefix_cache_probe_phase": "warm_hit_candidate",
                },
            ],
        }

        rows = self.module.summarize_bench_payload(
            variant=variant,
            prompt_len=128,
            payload=payload,
            cache_bytes=4096,
            health_payload={"memory": {"mlx_peak_mb": 2048.5}},
            cache_dir=Path("/tmp/cache"),
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["variant"], "turboquant_prefix_cache")
        self.assertEqual(row["prompt_len"], 128)
        self.assertEqual(row["cold_ttft_ms"], 100.0)
        self.assertEqual(row["warm_ttft_ms_median"], 45.0)
        self.assertEqual(row["warm_tg_tps_median"], 27.5)
        self.assertEqual(row["cache_bytes"], 4096)
        self.assertEqual(row["memory_peak_mb"], 2048.5)

    def test_summary_files_are_written_as_json_csv_and_markdown(self):
        rows = [
            {
                "variant": "turboquant_prefix_cache",
                "prompt_len": 128,
                "kv_quant": "k3v4",
                "prefix_cache": True,
                "cold_ttft_ms": 100.0,
                "warm_ttft_ms_median": 45.0,
                "warm_tg_tps_median": 27.5,
                "warm_e2e_s_median": 0.275,
                "warm_cached_tokens_median": 120.0,
                "cache_bytes": 4096,
                "memory_peak_mb": 2048.5,
                "status": "ok",
                "notes": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.module.write_summary_files(out, rows)

            self.assertEqual(json.loads((out / "summary.json").read_text()), rows)
            csv_text = (out / "summary.csv").read_text()
            md_text = (out / "summary.md").read_text()

        self.assertIn("turboquant_prefix_cache", csv_text)
        self.assertIn("warm_ttft_ms_median", csv_text)
        self.assertIn("| turboquant_prefix_cache |", md_text)


if __name__ == "__main__":
    unittest.main()
