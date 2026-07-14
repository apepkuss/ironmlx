import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import mxfp_performance_gate as gate


SCHEDULER_CONFIG = {
    "b_max": 8,
    "prefill_chunk_size": 2048,
    "admission_deadline_ms": 5,
    "admission_queue_max": 32,
    "max_cache_cap": 65536,
    "decode_cadence_mid_chunk_cap": 256,
}


def write_matrix(root: Path, candidate: str, baseline: str) -> None:
    rows = []
    for model, multiplier in ((candidate, 1.1), (baseline, 1.0)):
        rows.extend(
            [
                {
                    "model": model,
                    "category": "http_e2e",
                    "pp_target": 128,
                    "tg_target": 512,
                    "concurrency": 1,
                    "tpot_ms_median": 10.0 * multiplier,
                    "e2e_s_p95": 1.0 * multiplier,
                    "itl_ms_p95": 10.0 * multiplier,
                    "ok": True,
                },
                {
                    "model": model,
                    "category": "long_context",
                    "pp_target": 8192,
                    "tg_target": 512,
                    "concurrency": 1,
                    "tpot_ms_median": 10.0 * multiplier,
                    "e2e_s_p95": 5.0 * multiplier,
                    "itl_ms_p95": 10.0 * multiplier,
                    "ok": True,
                },
                {
                    "model": model,
                    "category": "concurrent",
                    "pp_target": 8192,
                    "tg_target": 512,
                    "concurrency": 8,
                    "tpot_ms_median": "",
                    "e2e_s_p95": 20.0 * multiplier,
                    "itl_ms_p95": 12.0 * multiplier,
                    "ok": True,
                },
            ]
        )
    fieldnames = list(rows[0])
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_scheduler_manifest(root: Path, config: Optional[dict[str, int]]) -> None:
    manifest = {} if config is None else {"scheduler_config": config}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class PerformanceGateTests(unittest.TestCase):
    def test_parse_args_defaults_to_ignored_reports_root(self) -> None:
        argv = [
            "mxfp_performance_gate.py",
            "--matrix",
            "matrix",
            "--strict-decode",
            "strict",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = gate.parse_args()

        self.assertEqual(args.out_root, Path("reports/mxfp-performance"))

    def test_scheduler_config_validation_rejects_legacy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = root / "matrix"
            strict = root / "strict"
            matrix.mkdir()
            strict.mkdir()
            write_scheduler_manifest(matrix, None)
            write_scheduler_manifest(strict, None)

            with self.assertRaisesRegex(gate.GateError, "scheduler_config"):
                gate.validate_scheduler_configurations([matrix], strict)

    def test_scheduler_config_validation_requires_identical_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = root / "matrix"
            strict = root / "strict"
            matrix.mkdir()
            strict.mkdir()
            write_scheduler_manifest(matrix, SCHEDULER_CONFIG)
            write_scheduler_manifest(strict, SCHEDULER_CONFIG)

            self.assertEqual(
                gate.validate_scheduler_configurations([matrix], strict),
                SCHEDULER_CONFIG,
            )

            mismatched = dict(SCHEDULER_CONFIG)
            mismatched["decode_cadence_mid_chunk_cap"] = 128
            write_scheduler_manifest(strict, mismatched)
            with self.assertRaisesRegex(gate.GateError, "scheduler configuration mismatch"):
                gate.validate_scheduler_configurations([matrix], strict)

    def test_matrix_comparisons_cover_required_metric_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_matrix(root, "candidate", "baseline")

            comparisons = gate.matrix_comparisons(
                [root], "mxfp4", "candidate", "baseline", threshold=1.25
            )

        families = {item["metric_family"] for item in comparisons}
        self.assertEqual(families, {"sequential_tpot", "long_e2e_c1", "long_e2e_c8", "long_itl_c8"})
        self.assertTrue(all(item["passed"] for item in comparisons))

    def test_strict_decode_comparison_requires_valid_full_length_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "models": [
                    {
                        "label": "candidate",
                        "validation": {"ok": True},
                        "benchmark_json": "candidate.json",
                    },
                    {
                        "label": "baseline",
                        "validation": {"ok": True},
                        "benchmark_json": "baseline.json",
                    },
                ]
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for label, value in (("candidate", 12.0), ("baseline", 10.0)):
                (root / f"{label}.json").write_text(
                    json.dumps(
                        {
                            "cells": [
                                {
                                    "pp_target": 52,
                                    "tg_target": 512,
                                    "concurrent": 8,
                                    "itl_ms": {"p95": value},
                                }
                            ],
                            "raw_runs": [
                                {
                                    "worker_id": worker,
                                    "finish_reason": "length",
                                    "completion_tokens": 40,
                                    "completion_tokens_server": 512,
                                }
                                for worker in range(8)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            comparison = gate.strict_decode_comparison(
                root, "mxfp4", "candidate", "baseline", threshold=1.25
            )

        self.assertEqual(comparison["metric_family"], "strict_decode_itl_c8")
        self.assertAlmostEqual(comparison["ratio"], 1.2)
        self.assertTrue(comparison["passed"])

    def test_strict_decode_comparison_rejects_manifest_only_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "models": [
                    {
                        "label": label,
                        "validation": {"ok": True},
                        "benchmark_json": f"{label}.json",
                    }
                    for label in ("candidate", "baseline")
                ]
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for label in ("candidate", "baseline"):
                (root / f"{label}.json").write_text(
                    json.dumps(
                        {
                            "cells": [
                                {
                                    "pp_target": 52,
                                    "tg_target": 512,
                                    "concurrent": 8,
                                    "itl_ms": {"p95": 10.0},
                                }
                            ],
                            "raw_runs": [
                                {
                                    "worker_id": worker,
                                    "finish_reason": "stop" if worker == 0 else "length",
                                    "completion_tokens": 12 if worker == 0 else 40,
                                    "completion_tokens_server": 12 if worker == 0 else 512,
                                }
                                for worker in range(8)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(gate.GateError, "full-length"):
                gate.strict_decode_comparison(
                    root, "mxfp4", "candidate", "baseline", threshold=1.25
                )

    def test_comparison_fails_above_threshold(self) -> None:
        comparison = gate.comparison(
            mode="mxfp8",
            metric_family="strict_decode_itl_c8",
            source="fixture",
            pp_target=64,
            tg_target=512,
            concurrency=8,
            candidate=13.0,
            baseline=10.0,
            threshold=1.25,
        )

        self.assertFalse(comparison["passed"])
        self.assertAlmostEqual(comparison["ratio"], 1.3)


if __name__ == "__main__":
    unittest.main()
