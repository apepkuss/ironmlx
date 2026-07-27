import importlib.util
import json
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from email.message import Message
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("benchmark_prompt_lookup_matrix.py")


def load_module():
    spec = importlib.util.spec_from_file_location("prompt_lookup_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BenchmarkPromptLookupMatrixTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def config(self, out_root=Path("/tmp/prompt-lookup-test"), **overrides):
        values = {
            "root": Path("/repo"),
            "model_dir": Path("/models/qwen"),
            "out_root": out_root,
            "serve_bin": Path("/repo/target/release/ironmlx"),
            "tokenizer_bin": Path("/repo/target/release/iron-bench-tokenizer"),
        }
        values.update(overrides)
        return self.module.MatrixConfig(**values)

    def test_versioned_corpus_has_four_positive_categories_and_controls(self):
        cases = self.module.load_corpus(self.module.DEFAULT_CORPUS)

        positives = {case.category for case in cases if case.polarity == "positive"}
        controls = {case.polarity for case in cases if case.polarity != "positive"}

        self.assertEqual(positives, {"rag", "code", "json", "long_copy"})
        self.assertEqual(controls, {"negative", "adversarial"})
        self.assertEqual(len({case.case_id for case in cases}), len(cases))

    def test_lookup_config_parser_rejects_invalid_range(self):
        parsed = self.module.LookupConfig.parse("wide:2:6:8:65536:100000")
        self.assertEqual(parsed.name, "wide")
        self.assertEqual(parsed.max_draft_tokens, 8)

        with self.assertRaises(Exception):
            self.module.LookupConfig.parse("bad:5:2:4:100:100")

    def test_tokenizer_context_does_not_mask_existing_exception(self):
        sidecar = object.__new__(self.module.TokenizerSidecar)
        sidecar._process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        sidecar._lock = None
        sidecar._process.send_signal(signal.SIGINT)
        sidecar._process.wait(timeout=5)

        sidecar.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)

    def test_balanced_variants_use_abba_order_per_cache_mode(self):
        cfg = self.config(
            balanced=True,
            lookup_configs=(self.module.LookupConfig(name="default"),),
        )

        variants = self.module.build_variants(cfg)

        self.assertEqual(
            [variant.name for variant in variants],
            [
                "baseline_off",
                "lookup_default_off",
                "lookup_default_off",
                "baseline_off",
            ],
        )
        self.assertEqual([variant.round_index for variant in variants], [0, 1, 2, 3])

    def test_serve_command_has_exact_prompt_lookup_limits(self):
        lookup = self.module.LookupConfig(
            name="calibrated",
            min_ngram=3,
            max_ngram=7,
            max_draft_tokens=6,
            history_window_tokens=8192,
            max_index_entries=12345,
        )
        cfg = self.config(
            prompt_tokens=(1024, 8192),
            max_tokens=(128, 512),
            concurrency=(1, 8),
        )
        variant = self.module.Variant("lookup", "off", lookup, 0)

        command = self.module.build_serve_command(cfg, variant, Path("/tmp/variant"))

        self.assertEqual(command[0:2], ["/repo/target/release/ironmlx", "serve"])
        self.assertIn("--prompt-lookup", command)
        self.assertEqual(command[command.index("--max-sequences") + 1], "8")
        self.assertEqual(command[command.index("--max-cache-cap") + 1], "9728")
        self.assertEqual(command[command.index("--prefill-chunk-size") + 1], "2048")
        self.assertIn("--force-scheduler", command)
        self.assertEqual(
            command[command.index("--prompt-lookup-max-index-entries") + 1],
            "12345",
        )

    def test_serve_command_can_force_b1_for_controlled_concurrent_clients(self):
        cfg = self.config(concurrency=(1, 2, 8), max_sequences=1)
        variant = self.module.Variant("baseline", "prefix", None, 0)

        command = self.module.build_serve_command(cfg, variant, Path("/tmp/variant"))

        self.assertEqual(command[command.index("--max-sequences") + 1], "1")

    def test_comparison_marks_only_shape_controlled_scheduler_cells(self):
        rows = []
        for lookup_name in (None, "default"):
            for concurrency in (1, 2):
                rows.append(
                    {
                        "variant": "baseline" if lookup_name is None else "lookup",
                        "lookup_name": lookup_name,
                        "cache_mode": "prefix",
                        "case_id": "case",
                        "category": "code",
                        "polarity": "positive",
                        "target_prompt_tokens": 8192,
                        "max_tokens": 16,
                        "concurrency": concurrency,
                        "output_match_ratio": 1.0,
                        "baseline_consistent_ratio": 1.0,
                        "expected_prefix_match_ratio": 1.0,
                        "ttft_ms_median": 10.0,
                        "e2e_s_median": 1.0,
                        "e2e_s_p95": 1.0,
                        "tg_tps_median": 10.0,
                        "aggregate_tps_median": 10.0,
                        "itl_ms_p95": 1.0,
                        "lookup_queries": 1,
                        "lookup_hits": 1,
                        "lookup_drafted_tokens": 1,
                        "lookup_accepted_tokens": 1,
                        "lookup_rejected_tokens": 0,
                        "lookup_acceptance_ratio": 1.0,
                        "lookup_rollbacks": 0,
                        "index_entries_current_max": 0,
                        "index_entries_peak_max": 1,
                        "server_healthy": True,
                        "scheduler_path_observed": concurrency == 1,
                    }
                )

        comparisons = self.module.build_comparisons(self.config(), rows)
        controlled = {
            row["concurrency"]: row["scheduler_path_controlled"]
            for row in comparisons
        }
        self.assertEqual(controlled, {1: True, 2: False})

        for row in rows:
            row["scheduler_path_observed"] = True
        controlled_scheduler = self.module.build_comparisons(self.config(), rows)
        self.assertTrue(
            all(row["scheduler_path_controlled"] for row in controlled_scheduler)
        )

    def test_rendered_requests_are_request_local_and_copy_target_is_deep(self):
        case = next(
            case
            for case in self.module.load_corpus(self.module.DEFAULT_CORPUS)
            if case.category == "rag"
        )

        prompt_a, expected_a = self.module.render_case(case, 20, 1, 64)
        prompt_b, expected_b = self.module.render_case(case, 20, 2, 64)

        self.assertNotEqual(prompt_a, prompt_b)
        self.assertNotEqual(expected_a, expected_b)
        self.assertIn(expected_a.split()[0], prompt_a)
        self.assertNotIn(expected_a.split()[0], prompt_b)
        target_position = prompt_a.index(expected_a.split()[0])
        self.assertGreater(target_position, len(prompt_a) // 2)

    def test_prompt_resolution_chooses_nearest_unit_count(self):
        case = self.module.load_corpus(self.module.DEFAULT_CORPUS)[0]

        def fake_counter(text):
            return len(text) // 8

        resolved = self.module.resolve_prompt(case, 1200, 64, fake_counter)
        prompt, _ = self.module.render_case(case, resolved.context_units, 0, 64)
        current = fake_counter(prompt)
        next_prompt, _ = self.module.render_case(
            case, resolved.context_units + 1, 0, 64
        )
        previous_prompt, _ = self.module.render_case(
            case, max(1, resolved.context_units - 1), 0, 64
        )

        self.assertLessEqual(abs(current - 1200), abs(fake_counter(next_prompt) - 1200))
        self.assertLessEqual(
            abs(current - 1200), abs(fake_counter(previous_prompt) - 1200)
        )

    def test_output_parity_detects_cross_variant_mismatch(self):
        base = {
            "cache_mode": "off",
            "case_id": "rag",
            "target_prompt_tokens": 1024,
            "max_tokens": 128,
            "concurrency": 2,
            "batch_idx": 0,
            "worker_id": 1,
            "lookup_name": None,
            "output_token_hash": "baseline",
        }
        matching = dict(base, lookup_name="default")
        mismatching = dict(base, lookup_name="wide", output_token_hash="different")
        rows = [base, matching, mismatching]

        self.module.attach_output_parity(rows)

        self.assertTrue(rows[0]["baseline_consistent"])
        self.assertTrue(rows[1]["baseline_match"])
        self.assertFalse(rows[2]["baseline_match"])

    def test_prompt_lookup_health_uses_monotonic_process_delta(self):
        before = {
            "prompt_lookup": {
                "queries": 100,
                "drafted_tokens": 300,
                "accepted_tokens": 250,
                "verify_round_us": 1000,
                "exact_batched_verify_windows": 10,
                "sequential_verify_windows": 20,
            }
        }
        after = {
            "status": "healthy",
            "prompt_lookup": {
                "enabled": True,
                "queries": 104,
                "drafted_tokens": 312,
                "accepted_tokens": 259,
                "verify_round_us": 1600,
                "exact_batched_verify_windows": 13,
                "sequential_verify_windows": 22,
                "index_entries_current": 0,
                "index_entries_peak": 80,
            },
        }

        delta = self.module.health_delta(before, after)

        self.assertEqual(delta["prompt_lookup"]["queries"], 4)
        self.assertEqual(delta["prompt_lookup"]["drafted_tokens"], 12)
        self.assertEqual(delta["prompt_lookup"]["accepted_tokens"], 9)
        self.assertEqual(delta["prompt_lookup"]["verify_round_us"], 600)
        self.assertEqual(delta["prompt_lookup"]["exact_batched_verify_windows"], 3)
        self.assertEqual(delta["prompt_lookup"]["sequential_verify_windows"], 2)

    def test_built_requests_include_stable_prompt_hashes(self):
        case = self.module.load_corpus(self.module.DEFAULT_CORPUS)[0]
        resolved = self.module.ResolvedPrompt(case, 1024, 128, 12, 1000)

        first = self.module.build_requests(resolved, 2, 3, warmup=False)
        second = self.module.build_requests(resolved, 2, 3, warmup=False)

        self.assertEqual(first[0][2]["prompt_hash"], second[0][2]["prompt_hash"])
        self.assertNotEqual(first[0][2]["prompt_hash"], first[1][2]["prompt_hash"])

    def test_run_batch_submits_in_worker_order_without_serializing_responses(self):
        submitted = []
        submitted_lock = threading.Lock()
        all_submitted = threading.Event()

        class FakeResponse:
            status = 200

            def __init__(self):
                self.headers = Message()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def __iter__(self):
                if not all_submitted.wait(timeout=2):
                    raise AssertionError("response bodies were serialized")
                event = {
                    "choices": [
                        {"delta": {"content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"completion_tokens": 1},
                }
                yield ("data: " + json.dumps(event) + "\n").encode("utf-8")
                yield b"data: [DONE]\n"

        class FakeConnection:
            def __init__(self, host, port, timeout):
                del host, port, timeout

            def request(self, method, path, body, headers):
                del method, path, headers
                prompt = json.loads(body)["messages"][0]["content"]
                with submitted_lock:
                    submitted.append(prompt)
                    if len(submitted) == 4:
                        all_submitted.set()

            def getresponse(self):
                return FakeResponse()

            def close(self):
                pass

        requests = [
            ("worker-{}".format(index), None, {"worker_id": index})
            for index in range(4)
        ]
        with mock.patch.object(
            self.module.http.client,
            "HTTPConnection",
            FakeConnection,
        ):
            results, _ = self.module.run_batch(self.config(), requests, 1)

        self.assertEqual(submitted, ["worker-0", "worker-1", "worker-2", "worker-3"])
        self.assertEqual([row["worker_id"] for row in results], [0, 1, 2, 3])

    def test_batch_checkpoints_round_trip_atomically(self):
        case = self.module.load_corpus(self.module.DEFAULT_CORPUS)[0]
        resolved = self.module.ResolvedPrompt(case, 1024, 128, 12, 1000)
        rows = [{"output_token_ids": [1, 2, 3]}]
        health = {
            "case_id": case.case_id,
            "target_prompt_tokens": 1024,
            "max_tokens": 128,
            "concurrency": 2,
            "batch_idx": 4,
        }

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            self.module.write_batch_checkpoint(
                checkpoint_dir, resolved, 2, 4, rows, health
            )
            loaded_rows, loaded_health = self.module.load_batch_checkpoints(
                checkpoint_dir
            )

            self.assertEqual(loaded_rows, rows)
            self.assertEqual(loaded_health, [health])
            self.assertFalse(list(checkpoint_dir.glob("*.tmp")))

    def test_full_gate_requires_clean_lifecycle_and_four_dimensional_coverage(self):
        cfg = self.config(
            prompt_tokens=(1024, 8192, 32768, 65536),
            max_tokens=(128, 512),
            concurrency=(1, 2, 4, 8),
            runs=5,
            balanced=True,
        )
        comparisons = []
        for category in ("rag", "code", "json", "long_copy"):
            comparisons.append(
                {
                    "lookup_name": "default",
                    "category": category,
                    "polarity": "positive",
                    "output_match_ratio": 1.0,
                    "baseline_consistent_ratio": 1.0,
                    "index_entries_current_max": 0,
                    "index_entries_peak_max": 100,
                    "server_healthy": True,
                    "baseline_ttft_ms": 100.0,
                    "ttft_change_ms": 1.0,
                    "tg_change_pct": 12.0,
                    "concurrency": 1,
                    "e2e_p95_change_pct": -5.0,
                }
            )
        comparisons.extend(
            [
                {
                    "lookup_name": "default",
                    "category": "ordinary_chat",
                    "polarity": "negative",
                    "output_match_ratio": 1.0,
                    "baseline_consistent_ratio": 1.0,
                    "index_entries_current_max": 0,
                    "index_entries_peak_max": 100,
                    "server_healthy": True,
                    "baseline_ttft_ms": 100.0,
                    "ttft_change_ms": 1.0,
                    "tg_change_pct": -2.0,
                    "concurrency": 1,
                    "e2e_p95_change_pct": 0.0,
                },
                {
                    "lookup_name": "default",
                    "category": "rag",
                    "polarity": "positive",
                    "output_match_ratio": 1.0,
                    "baseline_consistent_ratio": 1.0,
                    "index_entries_current_max": 0,
                    "index_entries_peak_max": 100,
                    "server_healthy": True,
                    "baseline_ttft_ms": 100.0,
                    "ttft_change_ms": 1.0,
                    "tg_change_pct": 12.0,
                    "concurrency": 8,
                    "e2e_p95_change_pct": 1.0,
                },
            ]
        )
        for comparison in comparisons:
            comparison.update(
                lookup_queries=10,
                lookup_hits=5,
                lookup_drafted_tokens=20,
                lookup_accepted_tokens=15,
                lookup_rejected_tokens=5,
                target_prompt_tokens=1024,
                max_tokens=128,
            )
        template = comparisons[0]
        for prompt_tokens in (1024, 8192, 32768, 65536):
            for max_tokens in (128, 512):
                for concurrency in (1, 2, 4, 8):
                    comparisons.append(
                        dict(
                            template,
                            target_prompt_tokens=prompt_tokens,
                            max_tokens=max_tokens,
                            concurrency=concurrency,
                        )
                    )

        gates = self.module.evaluate_gates(cfg, comparisons)

        self.assertEqual(gates["status"], "pass")
        self.assertTrue(gates["output_token_parity_100pct"])
        self.assertTrue(gates["request_local_lifecycle_clean"])
        self.assertTrue(gates["lookup_dimension_coverage"])

    def test_gates_report_non_exercised_fallback_cells_separately(self):
        cfg = self.config()
        active = {
            "lookup_name": "default",
            "category": "rag",
            "polarity": "positive",
            "output_match_ratio": 1.0,
            "baseline_consistent_ratio": 1.0,
            "index_entries_current_max": 0,
            "index_entries_peak_max": 10,
            "server_healthy": True,
            "baseline_ttft_ms": 100.0,
            "ttft_change_ms": 0.0,
            "tg_change_pct": 12.0,
            "concurrency": 1,
            "e2e_p95_change_pct": 0.0,
            "lookup_queries": 4,
            "lookup_hits": 2,
            "lookup_drafted_tokens": 6,
            "lookup_accepted_tokens": 4,
            "lookup_rejected_tokens": 2,
        }
        fallback = dict(active)
        fallback.update(
            output_match_ratio=0.5,
            concurrency=4,
            lookup_queries=0,
            lookup_hits=0,
            lookup_drafted_tokens=0,
            lookup_accepted_tokens=0,
            lookup_rejected_tokens=0,
        )

        gates = self.module.evaluate_gates(cfg, [active, fallback])

        self.assertTrue(gates["output_token_parity_100pct"])
        self.assertTrue(gates["lookup_path_exercised"])
        self.assertEqual(gates["lookup_path_exercised_cells"], 1)
        self.assertEqual(gates["fallback_cells"], 1)
        self.assertFalse(gates["fallback_output_token_parity_100pct"])

    def test_gates_separate_scheduler_control_from_feature_toggle_and_check_counters(self):
        cfg = self.config()
        common = {
            "lookup_name": "default",
            "category": "ordinary_chat",
            "polarity": "negative",
            "baseline_consistent_ratio": 1.0,
            "index_entries_current_max": 0,
            "index_entries_peak_max": 10,
            "server_healthy": True,
            "baseline_ttft_ms": 100.0,
            "ttft_change_ms": 0.0,
            "tg_change_pct": 0.0,
            "concurrency": 1,
            "e2e_p95_change_pct": 0.0,
            "lookup_queries": 4,
            "lookup_hits": 2,
            "lookup_drafted_tokens": 6,
            "lookup_accepted_tokens": 4,
            "lookup_rejected_tokens": 2,
        }
        comparisons = [
            dict(
                common,
                cache_mode="off",
                scheduler_path_controlled=False,
                output_match_ratio=0.0,
            ),
            dict(
                common,
                cache_mode="prefix",
                scheduler_path_controlled=True,
                output_match_ratio=1.0,
            ),
        ]

        gates = self.module.evaluate_gates(cfg, comparisons)

        self.assertFalse(gates["output_token_parity_100pct"])
        self.assertTrue(gates["scheduler_path_output_token_parity_100pct"])
        self.assertTrue(gates["lookup_counter_invariants_hold"])

        comparisons[1]["lookup_rejected_tokens"] = 1
        gates = self.module.evaluate_gates(cfg, comparisons)
        self.assertFalse(gates["lookup_counter_invariants_hold"])

    def test_lifecycle_gate_is_independent_from_degraded_server_health(self):
        cfg = self.config()
        comparison = {
            "lookup_name": "default",
            "category": "long_copy",
            "polarity": "positive",
            "output_match_ratio": 1.0,
            "baseline_consistent_ratio": 1.0,
            "index_entries_current_max": 0,
            "index_entries_peak_max": 10,
            "server_healthy": False,
            "baseline_ttft_ms": 100.0,
            "ttft_change_ms": 0.0,
            "tg_change_pct": 10.0,
            "concurrency": 1,
            "e2e_p95_change_pct": 0.0,
            "lookup_queries": 8,
            "lookup_hits": 5,
            "lookup_drafted_tokens": 20,
            "lookup_accepted_tokens": 17,
            "lookup_rejected_tokens": 3,
        }

        gates = self.module.evaluate_gates(cfg, [comparison])

        self.assertTrue(gates["request_local_lifecycle_clean"])
        self.assertFalse(gates["server_health_healthy"])
        self.assertEqual(gates["server_health_degraded_cells"], 1)

    def test_dry_run_writes_plan_without_requiring_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            cfg = self.config(out_root=out, dry_run=True)

            result = self.module.run_matrix(cfg)

            self.assertEqual(result, 0)
            self.assertTrue((out / "metadata.json").is_file())
            self.assertTrue((out / "server-plan.sh").is_file())
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertEqual(metadata["variants"][0]["name"], "baseline_off")


if __name__ == "__main__":
    unittest.main()
