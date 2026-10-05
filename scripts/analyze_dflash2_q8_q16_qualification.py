#!/usr/bin/env python3
"""Analyze paired Q8/Q16 DFlash2 runs with deterministic 95% bootstrap CIs."""

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
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def bootstrap_geomean_ci(
    ratios: list[float], *, samples: int = 20_000, seed: int = 20_260_928
) -> dict[str, float]:
    if len(ratios) < 2:
        raise ValueError("CI95 requires at least two paired ratios")
    rng = random.Random(seed)
    logs = [math.log(value) for value in ratios]
    estimates = [
        math.exp(statistics.fmean(rng.choice(logs) for _ in logs))
        for _ in range(samples)
    ]
    return {
        "estimate": geometric_mean(ratios),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def bootstrap_stratified_geomean_ci(
    ratio_cells: list[list[float]], *, samples: int = 20_000, seed: int = 20_260_928
) -> dict[str, float]:
    """Bootstrap within each qualification cell, preserving matrix balance."""
    if not ratio_cells or any(len(cell) < 2 for cell in ratio_cells):
        raise ValueError("stratified CI95 requires at least two ratios per cell")
    if any(value <= 0.0 or not math.isfinite(value) for cell in ratio_cells for value in cell):
        raise ValueError("stratified geometric mean requires finite positive values")
    log_cells = [[math.log(value) for value in cell] for cell in ratio_cells]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        cell_means = [
            statistics.fmean(rng.choice(cell) for _ in cell) for cell in log_cells
        ]
        estimates.append(math.exp(statistics.fmean(cell_means)))
    point = math.exp(statistics.fmean(statistics.fmean(cell) for cell in log_cells))
    return {
        "estimate": point,
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def record_tps(record: dict[str, Any], timing: str) -> float:
    generated = int(record["metrics"]["generated_tokens"])
    if timing == "wall_total":
        elapsed_us = int(record["wall_total_us"])
    elif timing == "runtime_generation":
        elapsed_us = int(record["metrics"]["generation_us"])
    else:
        raise ValueError(f"unknown timing {timing}")
    if generated <= 0 or elapsed_us <= 0:
        raise ValueError("generated tokens and elapsed time must be positive")
    return generated * 1_000_000.0 / elapsed_us


def pair_records(records: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (str(record["mode"]), int(record["context_tokens"]), int(record["replicate"]))
        block_size = int(record["block_size"])
        if block_size in grouped[key]:
            raise ValueError(f"duplicate Q{block_size} record for {key}")
        grouped[key][block_size] = record

    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for (mode, context, replicate), widths in sorted(grouped.items()):
        if set(widths) != {8, 16}:
            raise ValueError(f"incomplete Q8/Q16 pair for {(mode, context, replicate)}")
        q8, q16 = widths[8], widths[16]
        cells[(mode, context)].append(
            {
                "replicate": replicate,
                "tokens_equal": q8["token_ids"] == q16["token_ids"],
                "q8": q8,
                "q16": q16,
            }
        )
    return cells


def summarize_metric(
    pairs: list[dict[str, Any]], timing: str, *, samples: int, seed: int
) -> dict[str, Any]:
    q8 = [record_tps(pair["q8"], timing) for pair in pairs]
    q16 = [record_tps(pair["q16"], timing) for pair in pairs]
    ratios = [right / left for left, right in zip(q8, q16)]
    return {
        "q8_tps_median": statistics.median(q8),
        "q16_tps_median": statistics.median(q16),
        "q16_over_q8": bootstrap_geomean_ci(ratios, samples=samples, seed=seed),
        "paired_ratios": ratios,
    }


def analyze(raw: dict[str, Any], *, bootstrap_samples: int = 20_000) -> dict[str, Any]:
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("unsupported raw qualification schema")
    cells = pair_records(list(raw.get("records", [])))
    if not cells:
        raise ValueError("qualification report has no records")

    cell_reports = []
    generation_ratio_cells: list[list[float]] = []
    wall_ratio_cells: list[list[float]] = []
    correctness = True
    enough_pairs = True
    for index, ((mode, context), pairs) in enumerate(sorted(cells.items())):
        correctness &= all(pair["tokens_equal"] for pair in pairs)
        enough_pairs &= len(pairs) >= 5
        generation = summarize_metric(
            pairs,
            "runtime_generation",
            samples=bootstrap_samples,
            seed=31_337 + index,
        )
        wall = summarize_metric(
            pairs,
            "wall_total",
            samples=bootstrap_samples,
            seed=91_337 + index,
        )
        generation_ratio_cells.append(generation["paired_ratios"])
        wall_ratio_cells.append(wall["paired_ratios"])
        acceptance_q8 = [float(pair["q8"]["metrics"]["acceptance_rate"]) for pair in pairs]
        acceptance_q16 = [float(pair["q16"]["metrics"]["acceptance_rate"]) for pair in pairs]
        cell_reports.append(
            {
                "mode": mode,
                "context_tokens": context,
                "paired_runs": len(pairs),
                "token_exact": all(pair["tokens_equal"] for pair in pairs),
                "runtime_generation": generation,
                "wall_total": wall,
                "acceptance_rate_median": {
                    "q8": statistics.median(acceptance_q8),
                    "q16": statistics.median(acceptance_q16),
                },
            }
        )

    aggregate_generation = bootstrap_stratified_geomean_ci(
        generation_ratio_cells, samples=bootstrap_samples, seed=20_260_928
    )
    aggregate_wall = bootstrap_stratified_geomean_ci(
        wall_ratio_cells, samples=bootstrap_samples, seed=20_260_929
    )
    no_generation_cell_regression = all(
        cell["runtime_generation"]["q16_over_q8"]["ci95_low"] >= 0.95
        for cell in cell_reports
    )
    promote = (
        correctness
        and enough_pairs
        and aggregate_generation["ci95_low"] > 1.0
        and aggregate_wall["ci95_low"] >= 0.98
        and no_generation_cell_regression
    )
    return {
        "schema_version": 1,
        "frozen_p2_sha": raw["frozen_p2_sha"],
        "scope": raw["execution_scope"],
        "target_dir": raw["target_dir"],
        "draft_dir": raw["draft_dir"],
        "max_new_tokens": raw["max_new_tokens"],
        "correctness": {"all_q8_q16_tokens_exact": correctness},
        "aggregate": {
            "runtime_generation_q16_over_q8": aggregate_generation,
            "wall_total_q16_over_q8": aggregate_wall,
        },
        "gates": {
            "at_least_five_pairs_per_cell": enough_pairs,
            "no_cell_generation_ci95_below_0_95": no_generation_cell_regression,
            "promote_q16_default": promote,
            "decision": "promote_q16" if promote else "keep_q8_default_q16_opt_in",
        },
        "cells": cell_reports,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DFlash2 Q8/Q16 CI95 qualification",
        "",
        f"- Frozen P2 baseline: `{report['frozen_p2_sha']}`",
        f"- Scope: {report['scope']}",
        f"- Decision: **{report['gates']['decision']}**",
        f"- Token exactness: `{report['correctness']['all_q8_q16_tokens_exact']}`",
        "",
        "| Mode | Context | Pairs | Runtime Q8 TPS | Runtime Q16 TPS | Runtime ratio CI95 | Total-wall ratio CI95 | Acceptance Q8/Q16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        runtime = cell["runtime_generation"]
        wall = cell["wall_total"]
        runtime_ratio = runtime["q16_over_q8"]
        wall_ratio = wall["q16_over_q8"]
        acceptance = cell["acceptance_rate_median"]
        lines.append(
            f"| {cell['mode']} | {cell['context_tokens']} | {cell['paired_runs']} | "
            f"{runtime['q8_tps_median']:.3f} | {runtime['q16_tps_median']:.3f} | "
            f"{runtime_ratio['estimate']:.4f} [{runtime_ratio['ci95_low']:.4f}, {runtime_ratio['ci95_high']:.4f}] | "
            f"{wall_ratio['estimate']:.4f} [{wall_ratio['ci95_low']:.4f}, {wall_ratio['ci95_high']:.4f}] | "
            f"{acceptance['q8']:.3f}/{acceptance['q16']:.3f} |"
        )
    aggregate_runtime = report["aggregate"]["runtime_generation_q16_over_q8"]
    aggregate_wall = report["aggregate"]["wall_total_q16_over_q8"]
    lines.extend(
        [
            "",
            "## Aggregate paired result",
            "",
            f"- Runtime generation ratio: {aggregate_runtime['estimate']:.4f} "
            f"(CI95 {aggregate_runtime['ci95_low']:.4f}–{aggregate_runtime['ci95_high']:.4f}).",
            f"- Total-wall ratio: {aggregate_wall['estimate']:.4f} "
            f"(CI95 {aggregate_wall['ci95_low']:.4f}–{aggregate_wall['ci95_high']:.4f}).",
            "- Ratios above 1.0 favor Q16.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1_000:
        parser.error("--bootstrap-samples must be at least 1000")
    return args


def main() -> int:
    args = parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyze(raw, bootstrap_samples=args.bootstrap_samples)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
