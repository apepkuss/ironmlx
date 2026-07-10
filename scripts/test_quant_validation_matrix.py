import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import quant_validation_matrix as qvm


class QuantValidationMatrixArgsTests(unittest.TestCase):
    def test_parse_args_defaults_to_ignored_reports_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            argv = [
                "quant_validation_matrix.py",
                "--model",
                f"toy={model_dir}",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = qvm.parse_args()

        self.assertEqual(args.out_root, "reports/quant-validation")

    def test_parse_args_accepts_multiple_concurrency_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            argv = [
                "quant_validation_matrix.py",
                "--model",
                f"toy={model_dir}",
                "--concurrent",
                "4,8",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = qvm.parse_args()

        self.assertEqual(args.concurrent, [4, 8])

    def test_parse_args_accepts_dedicated_concurrent_prompt_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            argv = [
                "quant_validation_matrix.py",
                "--model",
                f"toy={model_dir}",
                "--concurrent-prompt-lens",
                "8192,32768",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = qvm.parse_args()

        self.assertEqual(args.concurrent_prompt_lens, [8192, 32768])

    def test_server_command_pins_every_scheduler_config_value(self) -> None:
        args = SimpleNamespace(
            serve_prefill_chunk_size=2048,
            serve_admission_deadline_ms=5,
            serve_admission_queue_max=32,
            serve_max_cache_cap=65536,
            serve_decode_cadence_mid_chunk_cap=256,
        )

        command = qvm.build_server_command(
            Path("/repo"),
            qvm.ModelSpec(label="toy", path=Path("/model")),
            port=18740,
            max_sequences=8,
            args=args,
        )

        expected = {
            "--max-sequences": "8",
            "--prefill-chunk-size": "2048",
            "--admission-deadline-ms": "5",
            "--admission-queue-max": "32",
            "--max-cache-cap": "65536",
            "--decode-cadence-mid-chunk-cap": "256",
        }
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)

        self.assertEqual(
            qvm.scheduler_config_from_args(args, max_sequences=8),
            {
                "b_max": 8,
                "prefill_chunk_size": 2048,
                "admission_deadline_ms": 5,
                "admission_queue_max": 32,
                "max_cache_cap": 65536,
                "decode_cadence_mid_chunk_cap": 256,
            },
        )


class QuantValidationMatrixEvidenceTests(unittest.TestCase):
    def test_checkpoint_identity_records_revision_and_quant_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "models--mlx-community--toy" / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(
                '{"quantization":{"mode":"mxfp4","bits":4,"group_size":32}}',
                encoding="utf-8",
            )
            model = qvm.ModelSpec(label="toy", path=snapshot)

            identity = qvm.checkpoint_identity(model)

        self.assertEqual(identity["revision"], "abc123")
        self.assertEqual(identity["quantization_mode"], "mxfp4")
        self.assertEqual(identity["bits"], 4)
        self.assertEqual(identity["group_size"], 32)
        self.assertEqual(identity["expected_bits"], 4)
        self.assertEqual(identity["expected_group_size"], 32)
        self.assertTrue(identity["contract_matches"])

    def test_checkpoint_identity_accepts_quantization_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "models--mlx-community--toy" / "snapshots" / "def456"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(
                '{"quantization_config":{"mode":"mxfp8","bits":8,"group_size":32}}',
                encoding="utf-8",
            )
            model = qvm.ModelSpec(label="toy", path=snapshot)

            identity = qvm.checkpoint_identity(model)

        self.assertEqual(identity["quantization_mode"], "mxfp8")
        self.assertTrue(identity["contract_matches"])

    def test_completion_counts_cover_sequential_and_concurrent_payloads(self) -> None:
        sequential = {"stats": [{"n_runs": 2}, {"n_runs": 3}]}
        concurrent = {"cells": [{"n_requests": 7}, {"n_requests": 11}]}

        self.assertEqual(
            qvm.benchmark_completion_counts(sequential),
            {"completed_requests": 5, "failed_requests": 0},
        )
        self.assertEqual(
            qvm.benchmark_completion_counts(concurrent),
            {"completed_requests": 18, "failed_requests": 0},
        )

    def test_check_status_is_explicit(self) -> None:
        self.assertEqual(qvm.with_check_status({"ok": True})["status"], "passed")
        self.assertEqual(qvm.with_check_status({"ok": False})["status"], "failed")

    def test_summary_preserves_decode_and_concurrent_latency_metrics(self) -> None:
        sequential = {
            "stats": [
                {
                    "pp_target": 512,
                    "tg_target": 128,
                    "n_runs": 2,
                    "tpot_ms_median": 12.5,
                    "pp_tps_median": 3200.0,
                }
            ]
        }
        concurrent = {
            "cells": [
                {
                    "pp_target": 32768,
                    "tg_target": 512,
                    "concurrent": 8,
                    "n_requests": 8,
                    "itl_ms": {"p95": 21.0},
                }
            ]
        }

        seq_row = qvm.summarize_sequential("toy", "http_e2e", sequential)[0]
        concurrent_row = qvm.summarize_concurrent("toy", concurrent)[0]
        self.assertEqual(seq_row["tpot_ms_median"], 12.5)
        self.assertEqual(seq_row["prefill_tokens_per_sec"], 3200.0)
        self.assertEqual(concurrent_row["itl_ms_p95"], 21.0)

    def test_summary_csv_uses_lf_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {"overall_status": "passed", "models": []}
            qvm.write_summary(run_dir, [], manifest)

            raw = (run_dir / "summary.csv").read_bytes()

        self.assertNotIn(b"\r\n", raw)


if __name__ == "__main__":
    unittest.main()
