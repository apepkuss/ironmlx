#!/usr/bin/env python3
"""Analyze paired GenerationStream/SchedulerActor B=1 qualification runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
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
    ratios: list[float], *, samples: int, seed: int
) -> dict[str, float]:
    if len(ratios) < 2:
        raise ValueError("CI95 requires at least two paired ratios")
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


def bootstrap_stratified_geomean_ci(
    cells: list[list[float]], *, samples: int, seed: int
) -> dict[str, float]:
    if not cells or any(len(cell) < 2 for cell in cells):
        raise ValueError("stratified CI95 requires at least two ratios per cell")
    log_cells = [[math.log(value) for value in cell] for cell in cells]
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def paired_cell(path: Path, *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    raw = load_json(path)
    meta = raw["meta"]
    if meta["path"] != "paired":
        raise ValueError(f"{path}: expected paired path")
    records = raw["records"]
    if len(records) < 5:
        raise ValueError(f"{path}: CI95 requires at least five measured pairs")

    tps_ratios = []
    e2e_speed_ratios = []
    ttft_speed_ratios = []
    direct_tps = []
    actor_tps = []
    direct_e2e = []
    actor_e2e = []
    direct_ttft = []
    actor_ttft = []
    exact = True
    valid = True
    orders = set()
    for record in records:
        direct = record["direct"]
        actor = record["actor"]
        if direct is None or actor is None:
            raise ValueError(f"{path}: incomplete paired record")
        orders.add(record["order"])
        exact &= bool(record["token_exact"])
        valid &= bool(direct["valid"]) and bool(actor["valid"])
        direct_tps.append(float(direct["generation_tps"]))
        actor_tps.append(float(actor["generation_tps"]))
        direct_e2e.append(float(direct["e2e_ms"]))
        actor_e2e.append(float(actor["e2e_ms"]))
        direct_ttft.append(float(direct["ttft_ms"]))
        actor_ttft.append(float(actor["ttft_ms"]))
        tps_ratios.append(actor_tps[-1] / direct_tps[-1])
        e2e_speed_ratios.append(direct_e2e[-1] / actor_e2e[-1])
        ttft_speed_ratios.append(direct_ttft[-1] / actor_ttft[-1])

    if orders != {"direct-actor", "actor-direct"}:
        raise ValueError(f"{path}: paired order was not counterbalanced")
    return {
        "sampling": str(meta["sampling"]),
        "context_tokens": int(meta["prompt_tokens"]),
        "pairs": len(records),
        "token_exact": exact,
        "all_valid": valid,
        "median": {
            "direct_ttft_ms": statistics.median(direct_ttft),
            "actor_ttft_ms": statistics.median(actor_ttft),
            "direct_e2e_ms": statistics.median(direct_e2e),
            "actor_e2e_ms": statistics.median(actor_e2e),
            "direct_generation_tps": statistics.median(direct_tps),
            "actor_generation_tps": statistics.median(actor_tps),
        },
        "actor_over_direct_generation_tps": bootstrap_geomean_ci(
            tps_ratios, samples=bootstrap_samples, seed=seed
        ),
        "actor_over_direct_e2e_speed": bootstrap_geomean_ci(
            e2e_speed_ratios, samples=bootstrap_samples, seed=seed + 10_000
        ),
        "actor_over_direct_ttft_speed": bootstrap_geomean_ci(
            ttft_speed_ratios, samples=bootstrap_samples, seed=seed + 20_000
        ),
        "paired_ratios": {
            "generation_tps": tps_ratios,
            "e2e_speed": e2e_speed_ratios,
            "ttft_speed": ttft_speed_ratios,
        },
    }


def isolated_peak(memory_dir: Path, sampling: str, context: int) -> dict[str, Any]:
    direct = load_json(memory_dir / f"{sampling}-{context}-direct.json")
    actor = load_json(memory_dir / f"{sampling}-{context}-actor.json")
    direct_peak = int(direct["final_memory"]["peak_bytes"])
    actor_peak = int(actor["final_memory"]["peak_bytes"])
    return {
        "direct_bytes": direct_peak,
        "actor_bytes": actor_peak,
        "actor_over_direct": actor_peak / direct_peak,
    }


def analyze(
    paired_dir: Path, memory_dir: Path, *, bootstrap_samples: int = 20_000
) -> dict[str, Any]:
    paths = sorted(paired_dir.glob("*.json"))
    if not paths:
        raise ValueError("paired directory contains no JSON files")
    first_meta = load_json(paths[0])["meta"]
    cells = [
        paired_cell(path, bootstrap_samples=bootstrap_samples, seed=31_337 + index)
        for index, path in enumerate(paths)
    ]
    cells.sort(key=lambda cell: (cell["sampling"], cell["context_tokens"]))
    for cell in cells:
        cell["isolated_peak_memory"] = isolated_peak(
            memory_dir, cell["sampling"], cell["context_tokens"]
        )

    aggregate: dict[str, Any] = {}
    for label, selected in [
        ("all", cells),
        ("greedy", [cell for cell in cells if cell["sampling"] == "greedy"]),
        ("sampled", [cell for cell in cells if cell["sampling"] == "sampled"]),
    ]:
        aggregate[label] = {
            "actor_over_direct_generation_tps": bootstrap_stratified_geomean_ci(
                [cell["paired_ratios"]["generation_tps"] for cell in selected],
                samples=bootstrap_samples,
                seed=80_000 + len(aggregate),
            ),
            "actor_over_direct_e2e_speed": bootstrap_stratified_geomean_ci(
                [cell["paired_ratios"]["e2e_speed"] for cell in selected],
                samples=bootstrap_samples,
                seed=90_000 + len(aggregate),
            ),
            "actor_over_direct_ttft_speed": bootstrap_stratified_geomean_ci(
                [cell["paired_ratios"]["ttft_speed"] for cell in selected],
                samples=bootstrap_samples,
                seed=100_000 + len(aggregate),
            ),
        }

    correctness = all(cell["token_exact"] and cell["all_valid"] for cell in cells)
    parity_2pct = all(
        cell["actor_over_direct_generation_tps"]["ci95_low"] >= 0.98
        and cell["actor_over_direct_e2e_speed"]["ci95_low"] >= 0.98
        for cell in cells
    )
    safety_5pct = all(
        cell["actor_over_direct_generation_tps"]["ci95_low"] >= 0.95
        and cell["actor_over_direct_e2e_speed"]["ci95_low"] >= 0.95
        for cell in cells
    )
    memory_10pct = all(
        cell["isolated_peak_memory"]["actor_over_direct"] <= 1.10 for cell in cells
    )
    return {
        "schema_version": 1,
        "qualification": {
            "model_dir": first_meta["model_dir"],
            "mlx_metallib": first_meta["mlx_metallib"],
            "device_name": first_meta["device_name"],
            "max_tokens": first_meta["max_tokens"],
            "warmup_runs_per_cell": first_meta["warmup_runs"],
            "measured_pairs_per_cell": first_meta["measured_runs"],
            "cooldown_ms": first_meta["cooldown_ms"],
            "prefill_chunk_size": first_meta["prefill_chunk_size"],
            "b_max": first_meta["b_max"],
            "admission_deadline_ms": first_meta["admission_deadline_ms"],
        },
        "cells": cells,
        "aggregate": aggregate,
        "gates": {
            "token_exact_and_all_valid": correctness,
            "strict_2pct_per_cell": parity_2pct,
            "safety_5pct_per_cell": safety_5pct,
            "peak_memory_no_more_than_10pct": memory_10pct,
        },
    }


def interpretation_lines(report: dict[str, Any]) -> list[str]:
    greedy = report["aggregate"]["greedy"]
    sampled = report["aggregate"]["sampled"]
    gates = report["gates"]
    strict_gate = gates["strict_2pct_per_cell"]
    all_gates_pass = all(gates.values())
    lines = [
        f"- Greedy aggregate generation TPS ratio is {greedy['actor_over_direct_generation_tps']['estimate']:.4f}; the strict 2% per-cell parity gate {'passes' if strict_gate else 'fails'}.",
        f"- Sampled aggregate generation TPS ratio is {sampled['actor_over_direct_generation_tps']['estimate']:.4f}.",
    ]
    if all_gates_pass:
        lines.extend(
            [
                "- Token output is exact, every measured run is valid, and isolated peak memory stays within the 10% limit.",
                "- Performance qualification passes all declared gates. This clears the B1 performance gate for capability coverage and route unification; it does not by itself complete those later stages.",
            ]
        )
    else:
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        lines.extend(
            [
                f"- Performance qualification remains incomplete; failed gates: {failed}.",
                "- Keep the GenerationStream route until the failed gates are addressed and this qualification is repeated.",
            ]
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    qualification = report["qualification"]
    lines = [
        "# B1 GenerationStream / SchedulerActor qualification",
        "",
        "Ratios above 1.0 favor SchedulerActor. CI95 uses deterministic paired bootstrap resampling.",
        "",
        f"- Device: {qualification['device_name']}",
        f"- Model: `{qualification['model_dir']}`",
        f"- Workload: 2K/8K/32K prompt tokens, {qualification['max_tokens']} generated tokens, greedy and sampled",
        f"- Pairing: {qualification['warmup_runs_per_cell']} warmup + {qualification['measured_pairs_per_cell']} measured counterbalanced pairs per cell",
        f"- Scheduler: B={qualification['b_max']}, {qualification['admission_deadline_ms']} ms admission window, prefill chunk {qualification['prefill_chunk_size']}",
        "- Peak memory: isolated process per path and cell",
        "",
        "| Mode | Context | Pairs | TTFT direct/actor ms | Actor TTFT speed CI95 | TPS direct/actor | Actor TPS CI95 | E2E direct/actor ms | Actor E2E speed CI95 | Peak GiB direct/actor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gib = 1024**3
    for cell in report["cells"]:
        median = cell["median"]
        ttft = cell["actor_over_direct_ttft_speed"]
        tps = cell["actor_over_direct_generation_tps"]
        e2e = cell["actor_over_direct_e2e_speed"]
        peak = cell["isolated_peak_memory"]
        lines.append(
            f"| {cell['sampling']} | {cell['context_tokens']} | {cell['pairs']} | "
            f"{median['direct_ttft_ms']:.2f}/{median['actor_ttft_ms']:.2f} | "
            f"{ttft['estimate']:.4f} [{ttft['ci95_low']:.4f}, {ttft['ci95_high']:.4f}] | "
            f"{median['direct_generation_tps']:.2f}/{median['actor_generation_tps']:.2f} | "
            f"{tps['estimate']:.4f} [{tps['ci95_low']:.4f}, {tps['ci95_high']:.4f}] | "
            f"{median['direct_e2e_ms']:.2f}/{median['actor_e2e_ms']:.2f} | "
            f"{e2e['estimate']:.4f} [{e2e['ci95_low']:.4f}, {e2e['ci95_high']:.4f}] | "
            f"{peak['direct_bytes'] / gib:.3f}/{peak['actor_bytes'] / gib:.3f} |"
        )
    lines.extend(["", "## Aggregate", ""])
    for label in ("greedy", "sampled", "all"):
        item = report["aggregate"][label]
        tps = item["actor_over_direct_generation_tps"]
        e2e = item["actor_over_direct_e2e_speed"]
        ttft = item["actor_over_direct_ttft_speed"]
        lines.append(
            f"- {label}: TPS {tps['estimate']:.4f} "
            f"(CI95 {tps['ci95_low']:.4f}-{tps['ci95_high']:.4f}); "
            f"E2E speed {e2e['estimate']:.4f} "
            f"(CI95 {e2e['ci95_low']:.4f}-{e2e['ci95_high']:.4f}); "
            f"TTFT speed {ttft['estimate']:.4f} "
            f"(CI95 {ttft['ci95_low']:.4f}-{ttft['ci95_high']:.4f})."
        )
    lines.extend(["", "## Gates", ""])
    for name, passed in report["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *interpretation_lines(report),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-dir", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    report = analyze(
        args.paired_dir,
        args.memory_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
