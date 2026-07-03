import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("gemma4_drafter_active_kv_regression.py")


def load_module():
    spec = importlib.util.spec_from_file_location("gemma4_regression", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config(out_root: Path):
    module = load_module()
    return module.RegressionConfig(
        root=Path("/repo"),
        out_root=out_root,
        serve_bin=Path("/repo/target/release/ironmlx"),
        iron_bench_bin=Path("/repo/target/release/iron-bench"),
        e4b_model_dir=Path("/models/gemma4-e4b"),
        e4b_drafter_dir=Path("/models/gemma4-e4b-assistant"),
        gemma12b_model_dir=Path("/models/gemma4-12b"),
        gemma12b_drafter_dir=Path("/models/gemma4-12b-assistant"),
        prompt_lens=(2048, 24576),
        max_tokens=32,
        duration_secs=20,
        warmup_duration_secs=5,
        startup_timeout_secs=120,
        request_timeout_secs=900,
    )


class Gemma4DrafterActiveKvRegressionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_default_variants_cover_e4b_b2_e4b_b4_and_12b_b2(self):
        cfg = test_config(Path("/tmp/out"))

        variants = self.module.build_default_variants(cfg)

        self.assertEqual(
            [variant.name for variant in variants],
            ["e4b_b2", "e4b_b4", "12b_b2"],
        )
        self.assertEqual([variant.b_max for variant in variants], [2, 4, 2])
        self.assertEqual([variant.concurrent for variant in variants], [2, 4, 2])
        self.assertTrue(all(variant.max_cache_cap == 262144 for variant in variants))
        self.assertTrue(all(variant.mtp_draft_tokens == 2 for variant in variants))
        self.assertTrue(all(variant.kv_quant == "k3v4" for variant in variants))
        self.assertEqual(variants[0].model_dir, Path("/models/gemma4-e4b"))
        self.assertEqual(variants[2].drafter_dir, Path("/models/gemma4-12b-assistant"))

    def test_serve_command_starts_app_daemon_with_active_kv_and_paged_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = test_config(Path(tmp))
            variant = self.module.build_default_variants(cfg)[0]
            variant_dir = Path(tmp) / variant.name

            cmd = self.module.build_serve_command(cfg, variant, 19080, variant_dir)

        self.assertEqual(cmd[:2], ["/repo/target/release/ironmlx", "serve"])
        self.assertNotIn("--model", cmd)
        self.assertIn("--host", cmd)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertIn("--port", cmd)
        self.assertEqual(cmd[cmd.index("--port") + 1], "19080")
        self.assertIn("--max-sequences", cmd)
        self.assertEqual(cmd[cmd.index("--max-sequences") + 1], "2")
        self.assertIn("--max-cache-cap", cmd)
        self.assertEqual(cmd[cmd.index("--max-cache-cap") + 1], "262144")
        self.assertIn("--paged-prefix-cache-dir", cmd)
        self.assertTrue(
            cmd[cmd.index("--paged-prefix-cache-dir") + 1].endswith(
                "e4b_b2/prefix-cache"
            )
        )
        self.assertIn("--active-kv-offload", cmd)
        self.assertIn("--active-kv-offload-dir", cmd)
        self.assertTrue(
            cmd[cmd.index("--active-kv-offload-dir") + 1].endswith(
                "e4b_b2/active-kv"
            )
        )
        self.assertIn("--kv-quant", cmd)
        self.assertEqual(cmd[cmd.index("--kv-quant") + 1], "k3v4")

    def test_load_payload_uses_dynamic_app_model_api_with_drafter(self):
        cfg = test_config(Path("/tmp/out"))
        variant = self.module.build_default_variants(cfg)[2]

        payload = self.module.build_load_payload(cfg, variant)

        self.assertEqual(payload["model"], "gemma4-12b-b2")
        self.assertEqual(payload["model_dir"], "/models/gemma4-12b")
        self.assertEqual(payload["mtp_model_dir"], "/models/gemma4-12b-assistant")
        self.assertEqual(payload["mtp_draft_tokens"], 2)
        self.assertEqual(payload["max_cache_cap"], 262144)
        self.assertTrue(payload["set_default"])

    def test_bench_command_runs_concurrent_json_probe_against_loaded_model(self):
        cfg = test_config(Path("/tmp/out"))
        variant = self.module.build_default_variants(cfg)[1]

        cmd = self.module.build_bench_command(cfg, variant, 19081, prompt_len=24576)

        self.assertEqual(cmd[0], "/repo/target/release/iron-bench")
        self.assertIn("--target", cmd)
        self.assertIn("ironmlx=http://127.0.0.1:19081", cmd)
        self.assertIn("--model-dir", cmd)
        self.assertEqual(cmd[cmd.index("--model-dir") + 1], "/models/gemma4-e4b")
        self.assertIn("--model", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "gemma4-e4b-b4")
        self.assertIn("--prompt-len", cmd)
        self.assertEqual(cmd[cmd.index("--prompt-len") + 1], "24576")
        self.assertIn("--concurrent", cmd)
        self.assertEqual(cmd[cmd.index("--concurrent") + 1], "4")
        self.assertIn("--duration", cmd)
        self.assertEqual(cmd[cmd.index("--duration") + 1], "20")
        self.assertIn("--warmup-duration", cmd)
        self.assertEqual(cmd[cmd.index("--warmup-duration") + 1], "5")
        self.assertIn("--format", cmd)
        self.assertEqual(cmd[cmd.index("--format") + 1], "json")

    def test_health_delta_accepts_active_kv_budget_and_mtp_progress(self):
        variant = self.module.RegressionVariant(
            name="12b_b2",
            model_label="gemma4-12b-b2",
            model_dir=Path("/models/gemma4-12b"),
            drafter_dir=Path("/models/gemma4-12b-assistant"),
            b_max=2,
            concurrent=2,
        )
        before = health(memory_budget_exceeded=3, prefill_count=1, step_count=2)
        after = health(memory_budget_exceeded=3, prefill_count=2, step_count=5)

        self.module.assert_health_delta(variant, before, after)

    def test_health_delta_accepts_model_context_cap_below_requested_cap(self):
        variant = self.module.RegressionVariant(
            name="e4b_b2",
            model_label="gemma4-e4b-b2",
            model_dir=Path("/models/gemma4-e4b"),
            drafter_dir=Path("/models/gemma4-e4b-assistant"),
            b_max=2,
            concurrent=2,
        )
        before = health(
            memory_budget_exceeded=0,
            prefill_count=1,
            step_count=1,
            logical_cap=131072,
            model_context=131072,
        )
        after = health(
            memory_budget_exceeded=0,
            prefill_count=2,
            step_count=3,
            logical_cap=131072,
            model_context=131072,
        )

        self.module.assert_health_delta(variant, before, after)

    def test_health_delta_rejects_budget_growth_and_degraded_active_kv(self):
        variant = self.module.RegressionVariant(
            name="12b_b2",
            model_label="gemma4-12b-b2",
            model_dir=Path("/models/gemma4-12b"),
            drafter_dir=Path("/models/gemma4-12b-assistant"),
            b_max=2,
            concurrent=2,
        )
        before = health(memory_budget_exceeded=3, prefill_count=1, step_count=2)
        after = health(
            memory_budget_exceeded=4,
            prefill_count=1,
            step_count=2,
            active_kv_degraded=True,
            swap_error_count=1,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "memory_budget_exceeded_count increased.*active_kv_offload.degraded",
        ):
            self.module.assert_health_delta(variant, before, after)

    def test_rolling_mid_admit_profile_accepts_mid_admit_and_batched_active_count(self):
        self.module.assert_rolling_mid_admit_profile(
            "\n".join(
                [
                    "[chunked-rolling-profile] event=mid_begin active_before=1 active_after=2",
                    "[chunked-rolling-profile] event=mid_chunk active_count=2",
                    "[chunked-rolling-profile] event=mid_finalize active_before=2 active_after=2",
                ]
            )
        )

    def test_rolling_mid_admit_profile_rejects_b1_like_queue_only_run(self):
        log_text = "\n".join(
            [
                "[chunked-rolling-profile] event=fresh_prefill active_count=1 fresh_batch_limit=1",
                "[chunked-rolling-profile] event=queue_enqueue queue_len=3",
                "[chunked-rolling-profile] event=decode_step active_before=1 active_after=1",
            ]
        )

        with self.assertRaisesRegex(AssertionError, "did not start rolling mid-admit"):
            self.module.assert_rolling_mid_admit_profile(log_text)

    def test_rolling_mid_admit_profile_rejects_decode_step_errors(self):
        log_text = "\n".join(
            [
                "[chunked-rolling-profile] event=mid_begin active_before=1 active_after=2",
                "[chunked-rolling-profile] event=decode_step_error active_before=2 active_after=2",
                "[chunked-rolling-profile] event=mid_finalize active_before=2 active_after=2",
            ]
        )

        with self.assertRaisesRegex(AssertionError, "hit decode step errors"):
            self.module.assert_rolling_mid_admit_profile(log_text)

    def test_summary_payload_records_concurrent_metrics_and_health(self):
        variant = self.module.RegressionVariant(
            name="e4b_b4",
            model_label="gemma4-e4b-b4",
            model_dir=Path("/models/gemma4-e4b"),
            drafter_dir=Path("/models/gemma4-e4b-assistant"),
            b_max=4,
            concurrent=4,
        )
        payload = {
            "mode": "concurrent",
            "cells": [
                {
                    "pp_target": 24576,
                    "tg_target": 32,
                    "concurrent": 4,
                    "n_requests": 8,
                    "ttft_ms": {"p50": 1200.0, "p95": 1800.0},
                    "itl_ms": {"p50": 42.0, "p95": 70.0},
                    "early_itl_ms": {"p50": 35.0, "p95": 60.0},
                    "aggregate": {"tokens_per_sec": 78.5, "req_per_sec": 0.4},
                    "finish_reason_summary": "length=8",
                }
            ],
        }
        after = health(memory_budget_exceeded=0, prefill_count=2, step_count=7)

        rows = self.module.summarize_bench_payload(variant, payload, after)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["variant"], "e4b_b4")
        self.assertEqual(row["prompt_len"], 24576)
        self.assertEqual(row["concurrent"], 4)
        self.assertEqual(row["n_requests"], 8)
        self.assertEqual(row["ttft_ms_p50"], 1200.0)
        self.assertEqual(row["itl_ms_p95"], 70.0)
        self.assertEqual(row["tokens_per_sec"], 78.5)
        self.assertEqual(row["memory_budget_exceeded_count"], 0)
        self.assertEqual(row["active_kv_degraded"], False)

    def test_dry_run_writes_commands_metadata_and_empty_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "dry"
            exit_code = self.module.main(
                [
                    "--dry-run",
                    "--out-root",
                    str(out_root),
                    "--variant",
                    "12b_b2",
                    "--no-build",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((out_root / "run_commands.sh").is_file())
            self.assertTrue((out_root / "metadata.json").is_file())
            self.assertTrue((out_root / "summary.json").is_file())
            self.assertTrue((out_root / "summary.csv").is_file())
            self.assertTrue((out_root / "summary.md").is_file())
            metadata = json.loads((out_root / "metadata.json").read_text())
            commands = (out_root / "run_commands.sh").read_text()
            summary = (out_root / "summary.md").read_text()

        self.assertEqual(metadata["variants"], ["12b_b2"])
        self.assertIn("/admin/api/models/load", commands)
        self.assertIn("--active-kv-offload", commands)
        self.assertIn("12b_b2", summary)


def health(
    *,
    memory_budget_exceeded: int,
    prefill_count: int,
    step_count: int,
    active_kv_degraded: bool = False,
    swap_error_count: int = 0,
    logical_cap: int = 262144,
    model_context: int = 262144,
):
    return {
        "model": {
            "max_position_embeddings": model_context,
        },
        "scheduler": {
            "b_max": 2,
            "memory_budget_exceeded_count": memory_budget_exceeded,
        },
        "memory": {
            "kv_cache_budget_policy": "active_kv_offload",
            "kv_cache_logical_cap_tokens": logical_cap,
            "kv_cache_resident_cap_tokens": 1024,
            "mlx_peak_bytes": 2 * 1024 * 1024 * 1024,
        },
        "mtp": {
            "enabled": True,
            "draft_tokens": 2,
            "prefill_count": prefill_count,
            "step_count": step_count,
            "fallback_prefill_count": 0,
            "drafted_tokens": 8,
            "accepted_draft_tokens": 5,
        },
        "active_kv_offload": {
            "enabled": True,
            "degraded": active_kv_degraded,
            "swap_error_count": swap_error_count,
            "swap_out_count": 1,
            "swap_in_count": 1,
            "parked_requests": 0,
        },
    }


if __name__ == "__main__":
    unittest.main()
