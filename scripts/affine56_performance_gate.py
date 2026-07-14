#!/usr/bin/env python3
"""Evaluate production performance gates for affine 5-bit and 6-bit models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ARCHITECTURES = {
    "gemma4": {
        4: "gemma-4-e2b-it-4bit",
        5: "gemma-4-e2b-it-5bit",
        6: "gemma-4-e2b-it-6bit",
    },
    "qwen35": {
        4: "Qwen3.5-2B-4bit",
        5: "Qwen3.5-2B-5bit",
        6: "Qwen3.5-2B-6bit",
    },
}
EXPECTED_BITS_BY_LABEL = {
    label: bits for labels in ARCHITECTURES.values() for bits, label in labels.items()
}
LATENCY_THRESHOLDS = {5: 1.375, 6: 1.650}
FIVE_VS_SIX_THRESHOLD = 1.10
REQUIRED_TARGET_LENGTHS = {128, 512}
REQUIRED_LONG_PROMPTS = {8192, 32768}
REQUIRED_PREFILL_PROMPTS = {2048, 8192, 32768}
MIN_PREFILL_RUNS = 5
MIN_PREFILL_WARMUP = 2
MIN_PREFILL_COOLDOWN_SECS = 1
REQUIRED_PREFILL_ROUNDS = 2
SCHEDULER_CONFIG_FIELDS = (
    "b_max",
    "prefill_chunk_size",
    "admission_deadline_ms",
    "admission_queue_max",
    "max_cache_cap",
    "decode_cadence_mid_chunk_cap",
)


class GateError(RuntimeError):
    pass


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GateError(f"missing artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"failed to read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{path}: expected a JSON object")
    return value


def read_scheduler_configuration(run_dir: Path) -> dict[str, int]:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    config = manifest.get("scheduler_config")
    if not isinstance(config, dict):
        raise GateError(f"{manifest_path}: missing scheduler_config")
    normalized: dict[str, int] = {}
    for field in SCHEDULER_CONFIG_FIELDS:
        value = config.get(field)
        if not isinstance(value, int) or value <= 0:
            raise GateError(f"{manifest_path}: invalid scheduler_config.{field}={value!r}")
        normalized[field] = value
    return normalized


def validate_scheduler_configurations(
    matrix_dirs: list[Path], strict_dir: Path, prefill_dirs: list[Path]
) -> dict[str, int]:
    sources = [*matrix_dirs, strict_dir, *prefill_dirs]
    if not matrix_dirs:
        raise GateError("at least one matrix directory is required")
    expected_source = sources[0]
    expected = read_scheduler_configuration(expected_source)
    for source in sources[1:]:
        actual = read_scheduler_configuration(source)
        if actual != expected:
            raise GateError(
                "scheduler configuration mismatch: "
                f"expected {expected} from {expected_source}, got {actual} from {source}"
            )
    return expected


def read_matrix_rows(matrix_dir: Path) -> list[dict[str, Any]]:
    summary = matrix_dir / "summary.csv"
    if not summary.is_file():
        raise GateError(f"missing matrix summary: {summary}")
    rows: list[dict[str, Any]] = []
    with summary.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for field in ("pp_target", "tg_target", "concurrency", "requests"):
                row[field] = optional_int(row.get(field))
            for field in (
                "ttft_ms_p95",
                "e2e_s_p95",
                "tpot_ms_median",
                "itl_ms_p95",
            ):
                row[field] = optional_float(row.get(field))
            row["ok"] = str(row.get("ok", "")).lower() == "true"
            rows.append(row)
    if not rows:
        raise GateError(f"{summary}: no summary rows")
    return rows


def matrix_cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("category"),
        row.get("pp_target"),
        row.get("tg_target"),
        row.get("concurrency"),
    )


def validate_matrix_evidence(
    matrix_dir: Path, required_labels: set[str]
) -> tuple[int, list[dict[str, Any]]]:
    manifest_path = matrix_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("overall_status") != "passed":
        raise GateError(f"{matrix_dir}: matrix failed validation")
    args = manifest.get("args")
    if not isinstance(args, dict):
        raise GateError(f"{manifest_path}: missing args")
    tg_target = optional_int(args.get("max_tokens"))
    if tg_target not in REQUIRED_TARGET_LENGTHS:
        raise GateError(f"{manifest_path}: unsupported target length {tg_target}")
    if set(args.get("long_prompt_lens") or []) != REQUIRED_LONG_PROMPTS:
        raise GateError(f"{manifest_path}: incomplete long prompt coverage")
    if set(args.get("concurrent_prompt_lens") or []) != REQUIRED_LONG_PROMPTS:
        raise GateError(f"{manifest_path}: incomplete concurrent prompt coverage")
    if 8 not in set(args.get("concurrent") or []):
        raise GateError(f"{manifest_path}: missing concurrency 8")

    records = {
        record.get("label"): record
        for record in manifest.get("models") or []
        if isinstance(record, dict)
    }
    for label in required_labels:
        record = records.get(label)
        if record is None:
            raise GateError(f"{matrix_dir}: missing model record {label}")
        if (
            not record.get("ok")
            or record.get("failed_requests") != 0
            or not isinstance(record.get("completed_requests"), int)
            or record["completed_requests"] <= 0
        ):
            raise GateError(f"{matrix_dir}: {label} failed validation")

    rows = read_matrix_rows(matrix_dir)
    by_model: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for label in required_labels:
        model_rows = [row for row in rows if row.get("model") == label]
        if any(not row["ok"] or not row.get("requests") for row in model_rows):
            raise GateError(f"{matrix_dir}: {label} has failed validation rows")
        cells: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in model_rows:
            key = matrix_cell_key(row)
            if key in cells:
                raise GateError(f"{matrix_dir}: duplicate row for {label}/{key}")
            cells[key] = row
        required_keys = {
            ("long_context", prompt, tg_target, 1)
            for prompt in REQUIRED_LONG_PROMPTS
        } | {
            ("concurrent", prompt, tg_target, 8)
            for prompt in REQUIRED_LONG_PROMPTS
        }
        missing = required_keys - cells.keys()
        if missing:
            raise GateError(f"{matrix_dir}: {label} missing required cells {sorted(missing)}")
        if not any(
            key[0] == "http_e2e" and key[2] == tg_target and key[3] == 1
            for key in cells
        ):
            raise GateError(f"{matrix_dir}: {label} missing sequential HTTP cell")
        for category in ("multi_turn", "stability"):
            if not any(key[0] == category for key in cells):
                raise GateError(f"{matrix_dir}: {label} missing {category} validation")
        by_model[label] = cells
    return tg_target, rows


def comparison(
    *,
    architecture: str,
    metric_family: str,
    source: str,
    pp_target: int | None,
    tg_target: int | None,
    concurrency: int,
    candidate_bits: int,
    baseline_bits: int,
    candidate: float,
    baseline: float,
    threshold: float,
) -> dict[str, Any]:
    if candidate < 0 or baseline <= 0:
        raise GateError(
            f"invalid metric values for {architecture}/{candidate_bits}/{metric_family}: "
            f"{candidate}, {baseline}"
        )
    ratio = candidate / baseline
    return {
        "architecture": architecture,
        "metric_family": metric_family,
        "source": source,
        "pp_target": pp_target,
        "tg_target": tg_target,
        "concurrency": concurrency,
        "candidate_bits": candidate_bits,
        "baseline_bits": baseline_bits,
        "candidate": candidate,
        "baseline": baseline,
        "ratio": ratio,
        "threshold": threshold,
        "passed": ratio <= threshold,
    }


def metric_specs(row: dict[str, Any]) -> list[tuple[str, str]]:
    category = row.get("category")
    concurrency = row.get("concurrency")
    if category in {"http_e2e", "long_context"} and concurrency == 1:
        return [("prefill_ttft_c1", "ttft_ms_p95")]
    return []


def matrix_comparisons(
    matrix_dirs: list[Path], architecture: str
) -> list[dict[str, Any]]:
    labels = ARCHITECTURES[architecture]
    results: list[dict[str, Any]] = []
    for matrix_dir in matrix_dirs:
        rows = read_matrix_rows(matrix_dir)
        cells = {
            bits: {
                matrix_cell_key(row): row
                for row in rows
                if row.get("model") == label and row["ok"]
            }
            for bits, label in labels.items()
        }
        for candidate_bits, baseline_bits, threshold in (
            (5, 4, LATENCY_THRESHOLDS[5]),
            (6, 4, LATENCY_THRESHOLDS[6]),
            (5, 6, FIVE_VS_SIX_THRESHOLD),
        ):
            for key, candidate_row in sorted(
                cells[candidate_bits].items(), key=lambda item: str(item[0])
            ):
                baseline_row = cells[baseline_bits].get(key)
                if baseline_row is None:
                    raise GateError(
                        f"{matrix_dir}: missing {baseline_bits}-bit baseline for {architecture}/{key}"
                    )
                for family, field in metric_specs(candidate_row):
                    candidate = candidate_row.get(field)
                    baseline = baseline_row.get(field)
                    if candidate is None or baseline is None:
                        raise GateError(
                            f"{matrix_dir}: missing {field} for {architecture}/{key}"
                        )
                    results.append(
                        comparison(
                            architecture=architecture,
                            metric_family=family,
                            source=str(matrix_dir),
                            pp_target=key[1],
                            tg_target=key[2],
                            concurrency=key[3],
                            candidate_bits=candidate_bits,
                            baseline_bits=baseline_bits,
                            candidate=candidate,
                            baseline=baseline,
                            threshold=threshold,
                        )
                    )
    return results


def strict_payload(
    strict_dir: Path, record: dict[str, Any], label: str, concurrency: int
) -> dict[str, Any]:
    benchmark = (record.get("benchmarks") or {}).get(f"c{concurrency}")
    if not isinstance(benchmark, dict):
        raise GateError(f"{label}: missing c{concurrency} strict decode artifact")
    if not (benchmark.get("validation") or {}).get("ok"):
        raise GateError(f"strict decode validation failed for {label}/c{concurrency}")
    raw_path = benchmark.get("benchmark_json")
    if not raw_path:
        raise GateError(f"{label}: missing strict decode JSON path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = strict_dir / path
    payload = read_json(path)
    cells = payload.get("cells") or []
    raw_runs = payload.get("raw_runs") or []
    if len(cells) != 1 or not raw_runs:
        raise GateError(f"{label}/c{concurrency}: incomplete strict decode payload")
    cell = cells[0]
    if (
        optional_int(cell.get("tg_target")) != 512
        or optional_int(cell.get("concurrent")) != concurrency
        or optional_int(cell.get("n_requests")) != len(raw_runs)
    ):
        raise GateError(f"{label}/c{concurrency}: invalid strict decode cell")
    invalid = [
        index
        for index, row in enumerate(raw_runs)
        if row.get("finish_reason") != "length"
        or row.get("completion_tokens_server") != 512
    ]
    if invalid:
        raise GateError(f"{label}/c{concurrency}: strict decode is not full-length: {invalid}")
    if {row.get("worker_id") for row in raw_runs} != set(range(concurrency)):
        raise GateError(f"{label}/c{concurrency}: incomplete strict worker coverage")
    return cell


def strict_comparisons(strict_dir: Path, architecture: str) -> list[dict[str, Any]]:
    manifest = read_json(strict_dir / "manifest.json")
    records = {
        record.get("label"): record
        for record in manifest.get("models") or []
        if isinstance(record, dict)
    }
    labels = ARCHITECTURES[architecture]
    cells: dict[int, dict[int, dict[str, Any]]] = {}
    for bits, label in labels.items():
        record = records.get(label)
        if record is None:
            raise GateError(f"strict decode manifest missing model {label}")
        cells[bits] = {
            concurrency: strict_payload(strict_dir, record, label, concurrency)
            for concurrency in (1, 8)
        }
    results = []
    for candidate_bits, baseline_bits, threshold in (
        (5, 4, LATENCY_THRESHOLDS[5]),
        (6, 4, LATENCY_THRESHOLDS[6]),
        (5, 6, FIVE_VS_SIX_THRESHOLD),
    ):
        for concurrency, family, field, statistic in (
            (1, "strict_decode_itl_c1", "itl_ms", "p95"),
            (8, "strict_decode_itl_c8", "itl_ms", "p95"),
        ):
            candidate = optional_float(
                (cells[candidate_bits][concurrency].get(field) or {}).get(statistic)
            )
            baseline = optional_float(
                (cells[baseline_bits][concurrency].get(field) or {}).get(statistic)
            )
            if candidate is None or baseline is None:
                raise GateError(f"missing strict decode {family} for {architecture}")
            results.append(
                comparison(
                    architecture=architecture,
                    metric_family=family,
                    source=str(strict_dir),
                    pp_target=optional_int(
                        cells[candidate_bits][concurrency].get("pp_target")
                    ),
                    tg_target=512,
                    concurrency=concurrency,
                    candidate_bits=candidate_bits,
                    baseline_bits=baseline_bits,
                    candidate=candidate,
                    baseline=baseline,
                    threshold=threshold,
                )
            )
    return results


def read_prefill_round(
    prefill_dir: Path, required_labels: set[str]
) -> tuple[
    list[str],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, dict[int, list[float]]],
]:
    manifest_path = prefill_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("overall_status") != "passed":
        raise GateError(f"{prefill_dir}: clean prefill validation failed")
    args = manifest.get("args") or {}
    if set(args.get("prompt_lens") or []) != REQUIRED_PREFILL_PROMPTS:
        raise GateError(f"{manifest_path}: incomplete clean prefill prompt coverage")
    runs = optional_int(args.get("runs"))
    if runs is None or runs < MIN_PREFILL_RUNS:
        raise GateError(
            f"{manifest_path}: clean prefill requires at least {MIN_PREFILL_RUNS} runs"
        )
    warmup = optional_int(args.get("warmup"))
    if warmup is None or warmup < MIN_PREFILL_WARMUP:
        raise GateError(
            f"{manifest_path}: clean prefill requires at least "
            f"{MIN_PREFILL_WARMUP} warmup runs"
        )
    cooldown = optional_int(args.get("inter_run_cooldown_secs"))
    if cooldown is None or cooldown < MIN_PREFILL_COOLDOWN_SECS:
        raise GateError(
            f"{manifest_path}: clean prefill requires at least "
            f"{MIN_PREFILL_COOLDOWN_SECS}s inter-run cooldown"
        )
    if (
        optional_int(args.get("max_tokens")) != 1
        or args.get("ignore_eos") is not True
        or optional_int(args.get("nonce_seed")) is None
    ):
        raise GateError(f"{manifest_path}: invalid clean prefill request configuration")

    model_records = manifest.get("models") or []
    if not all(isinstance(record, dict) for record in model_records):
        raise GateError(f"{manifest_path}: invalid clean prefill model records")
    order = [record.get("label") for record in model_records]
    if len(order) != len(required_labels) or set(order) != required_labels:
        raise GateError(f"{manifest_path}: incomplete clean prefill model order")
    records = {record["label"]: record for record in model_records}
    checkpoints: dict[str, dict[str, Any]] = {}
    result: dict[str, dict[int, list[float]]] = {}
    for label in required_labels:
        record = records.get(label)
        benchmark = (record or {}).get("benchmark") or {}
        if (
            not record
            or not record.get("ok")
            or benchmark.get("ok") is not True
            or not (benchmark.get("validation") or {}).get("ok")
        ):
            raise GateError(f"{prefill_dir}: {label} failed clean prefill validation")
        checkpoint = record.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("contract_matches") is not True
            or checkpoint.get("quantization_mode") != "affine"
            or optional_int(checkpoint.get("bits")) != EXPECTED_BITS_BY_LABEL[label]
            or optional_int(checkpoint.get("expected_bits"))
            not in (None, EXPECTED_BITS_BY_LABEL[label])
            or optional_int(checkpoint.get("group_size")) != 64
            or optional_int(checkpoint.get("expected_group_size")) not in (None, 64)
            or not checkpoint.get("repo_id")
            or not checkpoint.get("revision")
        ):
            raise GateError(f"{prefill_dir}: {label} has invalid checkpoint identity")
        checkpoints[label] = checkpoint
        raw_path = benchmark.get("benchmark_json")
        if not raw_path:
            raise GateError(f"{prefill_dir}: {label} missing clean prefill artifact")
        path = Path(raw_path)
        if not path.is_absolute():
            path = prefill_dir / path
        payload = read_json(path)
        metadata = payload.get("metadata") or {}
        if (
            optional_int(metadata.get("runs_measured")) != runs
            or optional_int(metadata.get("warmup")) != warmup
        ):
            raise GateError(f"{label}: inconsistent clean prefill metadata")
        stats = payload.get("stats") or []
        raw_runs = payload.get("raw_runs") or []
        cells: dict[int, dict[str, Any]] = {}
        for row in stats:
            prompt = optional_int(row.get("pp_target"))
            median = optional_float(row.get("ttft_ms_median"))
            p95 = optional_float(row.get("ttft_ms_p95"))
            if (
                prompt not in REQUIRED_PREFILL_PROMPTS
                or prompt in cells
                or optional_int(row.get("tg_target")) != 1
                or optional_int(row.get("n_runs")) != runs
                or median is None
                or median <= 0
                or not math.isfinite(median)
                or p95 is None
                or p95 <= 0
                or not math.isfinite(p95)
            ):
                raise GateError(f"{label}: invalid clean prefill stats for PP={prompt}")
            cells[prompt] = {
                "ttft_ms_median": median,
                "ttft_ms_p95": p95,
            }
        if set(cells) != REQUIRED_PREFILL_PROMPTS:
            raise GateError(f"{label}: incomplete clean prefill stats")
        samples = {prompt: [] for prompt in REQUIRED_PREFILL_PROMPTS}
        for index, row in enumerate(raw_runs):
            prompt = optional_int(row.get("pp_target"))
            ttft = optional_float(row.get("ttft_ms"))
            if (
                prompt not in samples
                or optional_int(row.get("tg_target")) != 1
                or row.get("finish_reason") != "length"
                or optional_int(row.get("completion_tokens_server")) != 1
                or ttft is None
                or ttft <= 0
                or not math.isfinite(ttft)
            ):
                raise GateError(f"{label}: invalid clean prefill request {index}")
            samples[prompt].append(ttft)
        if any(len(values) != runs for values in samples.values()):
            raise GateError(f"{label}: incomplete clean prefill raw request coverage")
        result[label] = samples
    normalized_args = {
        key: args.get(key)
        for key in (
            "prompt_lens",
            "max_tokens",
            "ignore_eos",
            "runs",
            "warmup",
            "inter_run_cooldown_secs",
            "nonce_seed",
        )
    }
    return order, checkpoints, normalized_args, result


def read_prefill_evidence(
    prefill_dirs: list[Path], required_labels: set[str]
) -> dict[str, dict[int, dict[str, Any]]]:
    if len(prefill_dirs) != REQUIRED_PREFILL_ROUNDS:
        raise GateError(
            f"clean performance gate requires exactly two prefill rounds, got {len(prefill_dirs)}"
        )
    rounds = [read_prefill_round(path, required_labels) for path in prefill_dirs]
    if rounds[1][0] != list(reversed(rounds[0][0])):
        raise GateError("clean prefill rounds must use exact reverse model order")
    if rounds[1][1] != rounds[0][1]:
        raise GateError("clean prefill checkpoint mismatch between rounds")
    if rounds[1][2] != rounds[0][2]:
        raise GateError("clean prefill request configuration mismatch between rounds")

    result: dict[str, dict[int, dict[str, Any]]] = {}
    for label in required_labels:
        cells = {}
        for prompt in REQUIRED_PREFILL_PROMPTS:
            samples = [
                value
                for _, _, _, round_samples in rounds
                for value in round_samples[label][prompt]
            ]
            cells[prompt] = {
                "ttft_ms_median": statistics.median(samples),
                "n_samples": len(samples),
            }
        result[label] = cells
    return result


def prefill_comparisons(
    prefill_dirs: list[Path],
    architecture: str,
    cells_by_label: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    labels = ARCHITECTURES[architecture]
    source = " + ".join(str(path) for path in prefill_dirs)
    results = []
    for candidate_bits, baseline_bits, threshold in (
        (5, 4, LATENCY_THRESHOLDS[5]),
        (6, 4, LATENCY_THRESHOLDS[6]),
        (5, 6, FIVE_VS_SIX_THRESHOLD),
    ):
        for prompt in sorted(REQUIRED_PREFILL_PROMPTS):
            candidate = cells_by_label[labels[candidate_bits]][prompt][
                "ttft_ms_median"
            ]
            baseline = cells_by_label[labels[baseline_bits]][prompt][
                "ttft_ms_median"
            ]
            results.append(
                comparison(
                    architecture=architecture,
                    metric_family="strict_prefill_ttft_median_c1",
                    source=source,
                    pp_target=prompt,
                    tg_target=1,
                    concurrency=1,
                    candidate_bits=candidate_bits,
                    baseline_bits=baseline_bits,
                    candidate=candidate,
                    baseline=baseline,
                    threshold=threshold,
                )
            )
    return results


def active_memory(matrix_dir: Path, label: str) -> int:
    path = matrix_dir / label / "health_before.json"
    if not path.is_file():
        raise GateError(f"missing active memory artifact: {path}")
    payload = read_json(path)
    value = (payload.get("memory") or {}).get("mlx_active_bytes")
    if not isinstance(value, int) or value <= 0:
        raise GateError(f"{path}: invalid active memory value {value!r}")
    return value


def memory_comparisons(
    matrix_dirs: list[Path], architecture: str
) -> list[dict[str, Any]]:
    labels = ARCHITECTURES[architecture]
    results = []
    for matrix_dir in matrix_dirs:
        values = {bits: active_memory(matrix_dir, label) for bits, label in labels.items()}
        for candidate_bits, baseline_bits, threshold in (
            (5, 4, LATENCY_THRESHOLDS[5]),
            (6, 4, LATENCY_THRESHOLDS[6]),
            (5, 6, FIVE_VS_SIX_THRESHOLD),
        ):
            item = comparison(
                architecture=architecture,
                metric_family="active_memory",
                source=str(matrix_dir),
                pp_target=None,
                tg_target=None,
                concurrency=0,
                candidate_bits=candidate_bits,
                baseline_bits=baseline_bits,
                candidate=float(values[candidate_bits]),
                baseline=float(values[baseline_bits]),
                threshold=threshold,
            )
            if baseline_bits == 4:
                item["passed"] = item["passed"] and values[candidate_bits] >= values[4]
            results.append(item)
    return results


def build_report(
    matrix_dirs: list[Path],
    strict_dir: Path,
    prefill_dirs: list[Path],
    *,
    architectures: tuple[str, ...] = ("gemma4", "qwen35"),
) -> dict[str, Any]:
    unknown = set(architectures) - ARCHITECTURES.keys()
    if unknown:
        raise GateError(f"unknown architectures: {sorted(unknown)}")
    required_labels = {
        label for architecture in architectures for label in ARCHITECTURES[architecture].values()
    }
    scheduler = validate_scheduler_configurations(matrix_dirs, strict_dir, prefill_dirs)
    target_lengths = set()
    for matrix_dir in matrix_dirs:
        target, _ = validate_matrix_evidence(matrix_dir, required_labels)
        target_lengths.add(target)
    if target_lengths != REQUIRED_TARGET_LENGTHS:
        raise GateError(
            f"incomplete target lengths: got {sorted(target_lengths)}, "
            f"expected {sorted(REQUIRED_TARGET_LENGTHS)}"
        )

    prefill_cells = read_prefill_evidence(prefill_dirs, required_labels)
    comparisons: list[dict[str, Any]] = []
    for architecture in architectures:
        comparisons.extend(
            prefill_comparisons(prefill_dirs, architecture, prefill_cells)
        )
        comparisons.extend(strict_comparisons(strict_dir, architecture))
        comparisons.extend(memory_comparisons(matrix_dirs, architecture))
    statuses = {
        f"{architecture}-{bits}bit": "passed"
        for architecture in architectures
        for bits in (5, 6)
    }
    for item in comparisons:
        if not item["passed"]:
            statuses[f"{item['architecture']}-{item['candidate_bits']}bit"] = "failed"
    return {
        "overall_status": "passed"
        if statuses and all(status == "passed" for status in statuses.values())
        else "failed",
        "statuses": statuses,
        "scheduler_config": scheduler,
        "target_lengths": sorted(target_lengths),
        "architectures": list(architectures),
        "matrix_dirs": [str(path) for path in matrix_dirs],
        "strict_decode_dir": str(strict_dir),
        "strict_prefill_dirs": [str(path) for path in prefill_dirs],
        "comparisons": comparisons,
    }


def write_summary(run_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Affine 5-bit and 6-bit Performance Gate",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Scheduler configuration: `{report['scheduler_config']}`",
        f"- Target lengths: `{report['target_lengths']}`",
        "",
        "## Independent Status",
        "",
        "| target | status |",
        "|---|---|",
    ]
    for target, status in sorted(report["statuses"].items()):
        lines.append(f"| {target} | {status} |")
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| architecture | candidate | baseline | metric | PP | TG | C | ratio | limit | status |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report["comparisons"]:
        lines.append(
            "| {architecture} | {candidate_bits} | {baseline_bits} | {metric} | {pp} | {tg} | {concurrency} | {ratio:.3f} | {threshold:.3f} | {status} |".format(
                architecture=item["architecture"],
                candidate_bits=item["candidate_bits"],
                baseline_bits=item["baseline_bits"],
                metric=item["metric_family"],
                pp=item["pp_target"] if item["pp_target"] is not None else "",
                tg=item["tg_target"] if item["tg_target"] is not None else "",
                concurrency=item["concurrency"],
                ratio=item["ratio"],
                threshold=item["threshold"],
                status="passed" if item["passed"] else "failed",
            )
        )
    lines.extend(["", f"- Machine-readable gate: `{run_dir / 'gate.json'}`", ""])
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="append", type=Path, required=True)
    parser.add_argument("--strict-decode", type=Path, required=True)
    parser.add_argument(
        "--strict-prefill", action="append", type=Path, required=True
    )
    parser.add_argument(
        "--architecture",
        action="append",
        choices=tuple(ARCHITECTURES),
        dest="architectures",
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("reports/affine56-performance")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    matrix_dirs = [path.resolve() for path in args.matrix]
    strict_dir = args.strict_decode.resolve()
    prefill_dirs = [path.resolve() for path in args.strict_prefill]
    architectures = tuple(args.architectures or ARCHITECTURES.keys())
    report = build_report(
        matrix_dirs, strict_dir, prefill_dirs, architectures=architectures
    )
    out_root = args.out_root if args.out_root.is_absolute() else repo / args.out_root
    run_dir = out_root.resolve() / time.strftime("%Y-%m-%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir.parent / "latest.txt").write_text(f"{run_dir}\n", encoding="utf-8")
    (run_dir / "gate.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(run_dir, report)
    print(run_dir)
    return 0 if report["overall_status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"affine56 performance gate: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
