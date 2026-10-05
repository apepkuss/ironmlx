#!/usr/bin/env python3
"""Analyze an ABA-bracketed DFlash2 B2/B4 Q8/Q16 feasibility gate."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def bootstrap_ci(
    ratios: list[float], *, samples: int = 20_000, seed: int = 20_260_928
) -> dict[str, float]:
    if len(ratios) < 2:
        raise ValueError("CI95 requires at least two ratios")
    logs = [math.log(value) for value in ratios]
    rng = random.Random(seed)
    estimates = [
        math.exp(statistics.fmean(rng.choice(logs) for _ in logs))
        for _ in range(samples)
    ]
    return {
        "estimate": geometric_mean(ratios),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def record_key(record: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(record["mode"]),
        int(record["context_tokens"]),
        int(record["batch_width"]),
        int(record["replicate"]),
    )


def index_records(report: dict[str, Any]) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    indexed = {}
    for record in report.get("records", []):
        key = record_key(record)
        if key in indexed:
            raise ValueError(f"duplicate record {key}")
        indexed[key] = record
    return indexed


def record_tps(record: dict[str, Any], field: str, max_new_tokens: int) -> float:
    elapsed_us = int(record[field])
    tokens = int(record["batch_width"]) * max_new_tokens
    if elapsed_us <= 0:
        raise ValueError("elapsed time must be positive")
    return tokens * 1_000_000.0 / elapsed_us


def analyze(
    q8_before: dict[str, Any],
    q16: dict[str, Any],
    q8_after: dict[str, Any],
    *,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    for report in (q8_before, q16, q8_after):
        if int(report.get("schema_version", 0)) != 1:
            raise ValueError("unsupported gate schema")
    if int(q8_before["block_size"]) != 8 or int(q8_after["block_size"]) != 8:
        raise ValueError("Q8 brackets must use block_size=8")
    if int(q16["block_size"]) != 16:
        raise ValueError("candidate must use block_size=16")
    max_new_tokens = int(q16["max_new_tokens"])
    if any(int(report["max_new_tokens"]) != max_new_tokens for report in (q8_before, q8_after)):
        raise ValueError("max_new_tokens mismatch")

    before = index_records(q8_before)
    candidate = index_records(q16)
    after = index_records(q8_after)
    keys = sorted(candidate)
    if not keys or any(key not in before or key not in after for key in keys):
        raise ValueError("Q16 records do not have complete Q8 brackets")

    cells: dict[tuple[str, int, int], list[tuple[dict[str, Any], ...]]] = defaultdict(list)
    all_exact = True
    tensor_path = True
    for key in keys:
        triple = (before[key], candidate[key], after[key])
        all_exact &= triple[0]["token_ids"] == triple[1]["token_ids"] == triple[2]["token_ids"]
        tensor_path &= all(int(record["tensor_batch_windows"]) > 0 for record in triple)
        cells[key[:3]].append(triple)

    cell_reports = []
    aggregate_generation = []
    aggregate_full = []
    enough_pairs = True
    for index, (cell_key, triples) in enumerate(sorted(cells.items())):
        enough_pairs &= len(triples) >= 5
        metrics = {}
        for metric_name, field in (
            ("generation_wall", "generation_wall_us"),
            ("full_wall", "full_wall_us"),
        ):
            q8_bracket = [
                math.sqrt(
                    record_tps(left, field, max_new_tokens)
                    * record_tps(right, field, max_new_tokens)
                )
                for left, _, right in triples
            ]
            q16_tps = [
                record_tps(middle, field, max_new_tokens) for _, middle, _ in triples
            ]
            ratios = [right / left for left, right in zip(q8_bracket, q16_tps)]
            summary = {
                "q8_bracket_tps_median": statistics.median(q8_bracket),
                "q16_tps_median": statistics.median(q16_tps),
                "q16_over_q8_bracket": bootstrap_ci(
                    ratios, samples=bootstrap_samples, seed=31_337 + index
                ),
                "paired_ratios": ratios,
            }
            metrics[metric_name] = summary
            (aggregate_generation if metric_name == "generation_wall" else aggregate_full).extend(
                ratios
            )
        mode, context_tokens, batch_width = cell_key
        cell_reports.append(
            {
                "mode": mode,
                "context_tokens": context_tokens,
                "batch_width": batch_width,
                "paired_runs": len(triples),
                **metrics,
            }
        )

    generation = bootstrap_ci(aggregate_generation, samples=bootstrap_samples, seed=91_337)
    full = bootstrap_ci(aggregate_full, samples=bootstrap_samples, seed=91_338)
    continue_p4 = (
        all_exact
        and tensor_path
        and enough_pairs
        and generation["estimate"] >= 1.10
        and generation["ci95_low"] > 1.0
        and full["ci95_low"] >= 0.98
    )
    return {
        "schema_version": 1,
        "frozen_p2_sha": q16["frozen_p2_sha"],
        "scope": "ABA-bracketed 2K greedy B2/B4 Q8/Q16 tensor-batch feasibility gate",
        "correctness": {
            "all_q8_q16_tokens_exact": all_exact,
            "all_runs_used_tensor_batch_windows": tensor_path,
        },
        "aggregate": {
            "generation_q16_over_q8_bracket": generation,
            "full_wall_q16_over_q8_bracket": full,
        },
        "gates": {
            "at_least_five_pairs_per_cell": enough_pairs,
            "required_point_improvement": 1.10,
            "continue_p4": continue_p4,
            "decision": "continue_batched_q16" if continue_p4 else "stop_batched_q16",
        },
        "cells": cell_reports,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DFlash2 B2/B4 Q8/Q16 feasibility gate",
        "",
        f"- Scope: {report['scope']}",
        f"- Decision: **{report['gates']['decision']}**",
        f"- Token exactness: `{report['correctness']['all_q8_q16_tokens_exact']}`",
        "",
        "| Mode | Context | Batch | Pairs | Generation Q8/Q16 TPS | Generation ratio CI95 | Full-wall ratio CI95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        generation = cell["generation_wall"]
        full = cell["full_wall"]
        generation_ratio = generation["q16_over_q8_bracket"]
        full_ratio = full["q16_over_q8_bracket"]
        lines.append(
            f"| {cell['mode']} | {cell['context_tokens']} | {cell['batch_width']} | "
            f"{cell['paired_runs']} | {generation['q8_bracket_tps_median']:.3f}/"
            f"{generation['q16_tps_median']:.3f} | {generation_ratio['estimate']:.4f} "
            f"[{generation_ratio['ci95_low']:.4f}, {generation_ratio['ci95_high']:.4f}] | "
            f"{full_ratio['estimate']:.4f} [{full_ratio['ci95_low']:.4f}, "
            f"{full_ratio['ci95_high']:.4f}] |"
        )
    generation = report["aggregate"]["generation_q16_over_q8_bracket"]
    full = report["aggregate"]["full_wall_q16_over_q8_bracket"]
    lines.extend(
        [
            "",
            f"- Aggregate generation ratio: {generation['estimate']:.4f} "
            f"(CI95 {generation['ci95_low']:.4f}–{generation['ci95_high']:.4f}).",
            f"- Aggregate full-wall ratio: {full['estimate']:.4f} "
            f"(CI95 {full['ci95_low']:.4f}–{full['ci95_high']:.4f}).",
            "- Ratios above 1.0 favor Q16; continuation requires a 1.10 point estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q8-before", type=Path, required=True)
    parser.add_argument("--q16", type=Path, required=True)
    parser.add_argument("--q8-after", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1_000:
        parser.error("--bootstrap-samples must be at least 1000")
    return args


def main() -> int:
    args = parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    report = analyze(
        load(args.q8_before),
        load(args.q16),
        load(args.q8_after),
        bootstrap_samples=args.bootstrap_samples,
    )
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
