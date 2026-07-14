from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import affine56_performance_gate as gate


SCHEDULER_CONFIG = {
    "b_max": 8,
    "prefill_chunk_size": 2048,
    "admission_deadline_ms": 5,
    "admission_queue_max": 32,
    "max_cache_cap": 65536,
    "decode_cadence_mid_chunk_cap": 256,
}


def labels(architecture: str) -> dict[int, str]:
    return gate.ARCHITECTURES[architecture]


def write_health(path: Path, active_bytes: int) -> None:
    path.write_text(
        json.dumps({"memory": {"mlx_active_bytes": active_bytes}}),
        encoding="utf-8",
    )


def write_matrix(
    root: Path,
    architecture: str,
    tg_target: int,
    multipliers: dict[int, float] | None = None,
    *,
    scheduler: dict[str, int] = SCHEDULER_CONFIG,
    failed_bit: int | None = None,
    include_memory: bool = True,
) -> None:
    multipliers = multipliers or {4: 1.0, 5: 1.2, 6: 1.3}
    rows: list[dict[str, object]] = []
    models = []
    for bits, label in labels(architecture).items():
        ok = bits != failed_bit
        models.append(
            {
                "label": label,
                "ok": ok,
                "completed_requests": 12 if ok else 11,
                "failed_requests": 0 if ok else 1,
            }
        )
        multiplier = multipliers[bits]
        rows.append(
            {
                "model": label,
                "category": "http_e2e",
                "pp_target": 2048,
                "tg_target": tg_target,
                "concurrency": 1,
                "requests": 2,
                "ttft_ms_p95": 500.0 * multiplier,
                "e2e_s_p95": 5.0 * multiplier,
                "tpot_ms_median": 10.0 * multiplier,
                "itl_ms_p95": "",
                "ok": ok,
            }
        )
        for pp_target in (8192, 32768):
            rows.extend(
                [
                    {
                        "model": label,
                        "category": "long_context",
                        "pp_target": pp_target,
                        "tg_target": tg_target,
                        "concurrency": 1,
                        "requests": 2,
                        "ttft_ms_p95": 2000.0 * multiplier,
                        "e2e_s_p95": 20.0 * multiplier,
                        "tpot_ms_median": 10.0 * multiplier,
                        "itl_ms_p95": "",
                        "ok": ok,
                    },
                    {
                        "model": label,
                        "category": "concurrent",
                        "pp_target": pp_target,
                        "tg_target": tg_target,
                        "concurrency": 8,
                        "requests": 8,
                        "ttft_ms_p95": 3000.0 * multiplier,
                        "e2e_s_p95": 40.0 * multiplier,
                        "tpot_ms_median": "",
                        "itl_ms_p95": 12.0 * multiplier,
                        "ok": ok,
                    },
                ]
            )
        for category in ("multi_turn", "stability"):
            rows.append(
                {
                    "model": label,
                    "category": category,
                    "pp_target": "",
                    "tg_target": "",
                    "concurrency": 1,
                    "requests": 3,
                    "e2e_s_p95": 1.0,
                    "tpot_ms_median": "",
                    "itl_ms_p95": "",
                    "ok": ok,
                }
            )
        model_dir = root / label
        model_dir.mkdir(parents=True, exist_ok=True)
        if include_memory:
            write_health(model_dir / "health_before.json", {4: 100, 5: 120, 6: 140}[bits])

    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "overall_status": "failed" if failed_bit is not None else "passed",
                "scheduler_config": scheduler,
                "args": {
                    "max_tokens": tg_target,
                    "long_prompt_lens": [8192, 32768],
                    "concurrent_prompt_lens": [8192, 32768],
                    "concurrent": [8],
                },
                "models": models,
            }
        ),
        encoding="utf-8",
    )


def write_strict(
    root: Path,
    architecture: str,
    multipliers: dict[int, float] | None = None,
    *,
    scheduler: dict[str, int] = SCHEDULER_CONFIG,
    premature_bit: int | None = None,
) -> None:
    multipliers = multipliers or {4: 1.0, 5: 1.2, 6: 1.3}
    records = []
    for bits, label in labels(architecture).items():
        benchmarks = {}
        for concurrency in (1, 8):
            filename = f"{label}-c{concurrency}.json"
            raw_runs = [
                {
                    "worker_id": worker,
                    "finish_reason": "stop"
                    if bits == premature_bit and worker == 0
                    else "length",
                    "completion_tokens": 8
                    if bits == premature_bit and worker == 0
                    else 40,
                    "completion_tokens_server": 8
                    if bits == premature_bit and worker == 0
                    else 512,
                }
                for worker in range(concurrency)
            ]
            (root / filename).write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "pp_target": 52,
                                "tg_target": 512,
                                "concurrent": concurrency,
                                "n_requests": concurrency,
                                "itl_ms": {"p95": 12.0 * multipliers[bits]},
                            }
                        ],
                        "raw_runs": raw_runs,
                    }
                ),
                encoding="utf-8",
            )
            benchmarks[f"c{concurrency}"] = {
                "benchmark_json": filename,
                "validation": {"ok": bits != premature_bit},
            }
        records.append({"label": label, "benchmarks": benchmarks})
    (root / "manifest.json").write_text(
        json.dumps({"scheduler_config": scheduler, "models": records}),
        encoding="utf-8",
    )


def write_prefill(
    root: Path,
    architecture: str,
    multipliers: dict[int, float] | None = None,
    *,
    scheduler: dict[str, int] = SCHEDULER_CONFIG,
    reverse: bool = False,
) -> None:
    multipliers = multipliers or {4: 1.0, 5: 1.2, 6: 1.3}
    records = []
    model_items = list(labels(architecture).items())
    if reverse:
        model_items.reverse()
    for bits, label in model_items:
        filename = f"{label}.json"
        stats = []
        raw_runs = []
        for prompt in (2048, 8192, 32768):
            stats.append(
                {
                    "pp_target": prompt,
                    "tg_target": 1,
                    "n_runs": 5,
                    "ttft_ms_median": prompt * multipliers[bits],
                    "ttft_ms_p95": prompt * multipliers[bits] * 1.05,
                }
            )
            raw_runs.extend(
                {
                    "pp_target": prompt,
                    "tg_target": 1,
                    "finish_reason": "length",
                    "completion_tokens_server": 1,
                    "ttft_ms": prompt * multipliers[bits],
                }
                for _ in range(5)
            )
        (root / filename).write_text(
            json.dumps(
                {
                    "metadata": {"runs_measured": 5, "warmup": 2},
                    "stats": stats,
                    "raw_runs": raw_runs,
                }
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "label": label,
                "ok": True,
                "checkpoint": {
                    "repo_id": label,
                    "revision": f"{label}-revision",
                    "bits": bits,
                    "group_size": 64,
                    "quantization_mode": "affine",
                    "contract_matches": True,
                },
                "benchmark": {
                    "ok": True,
                    "benchmark_json": filename,
                    "validation": {"ok": True},
                },
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "overall_status": "passed",
                "scheduler_config": scheduler,
                "args": {
                    "prompt_lens": [2048, 8192, 32768],
                    "max_tokens": 1,
                    "ignore_eos": True,
                    "runs": 5,
                    "warmup": 2,
                    "inter_run_cooldown_secs": 1,
                    "nonce_seed": 5606,
                },
                "models": records,
            }
        ),
        encoding="utf-8",
    )


class Affine56PerformanceGateTests(unittest.TestCase):
    def make_evidence(
        self,
        root: Path,
        architecture: str,
        multipliers: dict[int, float] | None = None,
    ) -> tuple[list[Path], Path, list[Path]]:
        matrix_dirs = []
        for tg_target in (128, 512):
            matrix = root / f"matrix-{tg_target}"
            matrix.mkdir()
            write_matrix(matrix, architecture, tg_target, multipliers)
            matrix_dirs.append(matrix)
        strict = root / "strict"
        strict.mkdir()
        write_strict(strict, architecture, multipliers)
        prefill_dirs = []
        for name, reverse in (("prefill-forward", False), ("prefill-reverse", True)):
            prefill = root / name
            prefill.mkdir()
            write_prefill(prefill, architecture, multipliers, reverse=reverse)
            prefill_dirs.append(prefill)
        return matrix_dirs, strict, prefill_dirs

    def test_parse_args_defaults_to_ignored_reports_root(self) -> None:
        argv = [
            "affine56_performance_gate.py",
            "--matrix",
            "matrix",
            "--strict-decode",
            "strict",
            "--strict-prefill",
            "prefill-forward",
            "--strict-prefill",
            "prefill-reverse",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = gate.parse_args()
        self.assertEqual(args.out_root, Path("reports/affine56-performance"))
        self.assertEqual(
            args.strict_prefill,
            [Path("prefill-forward"), Path("prefill-reverse")],
        )

    def test_complete_evidence_passes_independent_bit_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "gemma4")
            report = gate.build_report(
                matrix_dirs, strict, prefill_dirs, architectures=("gemma4",)
            )

        self.assertEqual(report["overall_status"], "passed")
        self.assertEqual(report["statuses"]["gemma4-5bit"], "passed")
        self.assertEqual(report["statuses"]["gemma4-6bit"], "passed")
        self.assertTrue(any(item["baseline_bits"] == 6 for item in report["comparisons"]))
        self.assertEqual(gate.LATENCY_THRESHOLDS, {5: 1.375, 6: 1.65})
        self.assertEqual(gate.FIVE_VS_SIX_THRESHOLD, 1.10)

    def test_clean_prefill_requires_two_reverse_order_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "gemma4")
            with self.assertRaisesRegex(gate.GateError, "two.*prefill rounds"):
                gate.build_report(
                    matrix_dirs,
                    strict,
                    prefill_dirs[:1],
                    architectures=("gemma4",),
                )

            manifest_path = prefill_dirs[1] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["models"].reverse()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "reverse model order"):
                gate.build_report(
                    matrix_dirs, strict, prefill_dirs, architectures=("gemma4",)
                )

    def test_clean_prefill_pools_raw_samples_across_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(
                root, "gemma4", {4: 1.0, 5: 1.0, 6: 1.0}
            )
            label = labels("gemma4")[5]
            for prefill, multiplier in zip(prefill_dirs, (1.20, 0.90)):
                artifact_path = prefill / f"{label}.json"
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                for row in payload["stats"]:
                    row["ttft_ms_median"] = row["pp_target"] * multiplier
                    row["ttft_ms_p95"] = row["pp_target"] * multiplier
                for row in payload["raw_runs"]:
                    row["ttft_ms"] = row["pp_target"] * multiplier
                artifact_path.write_text(json.dumps(payload), encoding="utf-8")

            report = gate.build_report(
                matrix_dirs, strict, prefill_dirs, architectures=("gemma4",)
            )

        ratios = {
            item["pp_target"]: item["ratio"]
            for item in report["comparisons"]
            if item["metric_family"] == "strict_prefill_ttft_median_c1"
            and item["candidate_bits"] == 5
            and item["baseline_bits"] == 4
        }
        self.assertEqual(ratios, {2048: 1.05, 8192: 1.05, 32768: 1.05})

    def test_clean_prefill_checkpoint_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "qwen35")
            manifest_path = prefill_dirs[1] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["models"][0]["checkpoint"]["revision"] = "different"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "checkpoint mismatch"):
                gate.build_report(
                    matrix_dirs, strict, prefill_dirs, architectures=("qwen35",)
                )

    def test_clean_prefill_rejects_wrong_checkpoint_contract_in_both_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "gemma4")
            label = labels("gemma4")[5]
            for prefill in prefill_dirs:
                manifest_path = prefill / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                record = next(item for item in manifest["models"] if item["label"] == label)
                record["checkpoint"]["bits"] = 6
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "invalid checkpoint identity"):
                gate.build_report(
                    matrix_dirs, strict, prefill_dirs, architectures=("gemma4",)
                )

    def test_matrix_latency_metrics_do_not_replace_clean_prefill_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "gemma4")
            for matrix in matrix_dirs:
                summary = matrix / "summary.csv"
                with summary.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    if row["model"] == labels("gemma4")[5]:
                        row["e2e_s_p95"] = "10000"
                        row["tpot_ms_median"] = "10000"
                        row["itl_ms_p95"] = "10000"
                with summary.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            report = gate.build_report(
                matrix_dirs, strict, prefill_dirs, architectures=("gemma4",)
            )

        self.assertEqual(report["overall_status"], "passed")
        matrix_metrics = {
            item["metric_family"]
            for item in report["comparisons"]
            if item["source"] in {str(path) for path in matrix_dirs}
            and item["metric_family"] != "active_memory"
        }
        self.assertEqual(matrix_metrics, set())
        prefill_metrics = {
            item["metric_family"]
            for item in report["comparisons"]
            if item["source"] == " + ".join(str(path) for path in prefill_dirs)
        }
        self.assertEqual(prefill_metrics, {"strict_prefill_ttft_median_c1"})

    def test_five_vs_six_failure_does_not_mask_other_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(
                root, "gemma4", {4: 1.0, 5: 1.30, 6: 1.0}
            )
            report = gate.build_report(
                matrix_dirs, strict, prefill_dirs, architectures=("gemma4",)
            )

        self.assertEqual(report["overall_status"], "failed")
        self.assertEqual(report["statuses"]["gemma4-5bit"], "failed")
        self.assertEqual(report["statuses"]["gemma4-6bit"], "passed")

    def test_missing_target_length_fails_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "qwen35")
            with self.assertRaisesRegex(gate.GateError, "target lengths"):
                gate.build_report(
                    matrix_dirs[:1], strict, prefill_dirs, architectures=("qwen35",)
                )

    def test_mismatched_scheduler_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "qwen35")
            mismatched = dict(SCHEDULER_CONFIG)
            mismatched["prefill_chunk_size"] = 1024
            manifest = json.loads((strict / "manifest.json").read_text(encoding="utf-8"))
            manifest["scheduler_config"] = mismatched
            (strict / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "scheduler configuration mismatch"):
                gate.build_report(
                    matrix_dirs, strict, prefill_dirs, architectures=("qwen35",)
                )

    def test_failed_requests_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "gemma4")
            write_matrix(matrix_dirs[0], "gemma4", 128, failed_bit=5)
            with self.assertRaisesRegex(gate.GateError, "failed validation"):
                gate.build_report(
                    matrix_dirs, strict, prefill_dirs, architectures=("gemma4",)
                )

    def test_premature_strict_decode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "gemma4")
            write_strict(strict, "gemma4", premature_bit=6)
            with self.assertRaisesRegex(gate.GateError, "strict decode validation failed|full-length"):
                gate.build_report(
                    matrix_dirs, strict, prefill_dirs, architectures=("gemma4",)
                )

    def test_under_warmed_prefill_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "gemma4")
            manifest_path = prefill_dirs[0] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["args"]["warmup"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "warmup"):
                gate.build_report(
                    matrix_dirs, strict, prefill_dirs, architectures=("gemma4",)
                )

    def test_non_positive_prefill_ttft_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "qwen35")
            artifact_path = prefill_dirs[0] / f"{labels('qwen35')[5]}.json"
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            payload["raw_runs"][0]["ttft_ms"] = 0.0
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "clean prefill request"):
                gate.build_report(
                    matrix_dirs, strict, prefill_dirs, architectures=("qwen35",)
                )

    def test_missing_memory_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dirs, strict, prefill_dirs = self.make_evidence(root, "qwen35")
            (matrix_dirs[-1] / labels("qwen35")[5] / "health_before.json").unlink()
            with self.assertRaisesRegex(gate.GateError, "memory"):
                gate.build_report(
                    matrix_dirs, strict, prefill_dirs, architectures=("qwen35",)
                )


if __name__ == "__main__":
    unittest.main()
