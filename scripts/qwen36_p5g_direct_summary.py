#!/usr/bin/env python3
"""Summarize direct GatedDeltaNet p5g layer2 profile logs.

The direct Rust benchmark emits regular JSON for whole-forward latency. When it
is built with `--features p5g-profile` and run with
`IRONMLX_P5G_PROFILE_MODE=layer2`, `GatedDeltaNet::forward_on` also writes
`[p5g-profile] ... step_breakdown=...` records to stderr. This script converts
those logs into stable JSON summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


STEP_NAMES = [
    "step_1a_in_proj_qkvz",
    "step_1b_in_proj_ba",
    "step_2a_prepend_conv_state",
    "step_2b_conv1d_silu",
    "step_2c_update_conv_state",
    "step_3_split_reshape_per_head",
    "step_4_qk_rmsnorm",
    "step_5_compute_g",
    "step_6_sigmoid_beta",
    "step_7_kernel_cache",
    "step_8_norm_proj",
]


def pct(values: list[float], p: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50_us": pct(values, 50),
        "p95_us": pct(values, 95),
        "mean_us": sum(values) / len(values) if values else None,
    }


def parse_profile_log(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in path.read_text().splitlines():
        if "[p5g-profile] " not in line:
            continue
        rest = line.split("[p5g-profile] ", 1)[1]
        record: dict[str, str] = {}
        for item in rest.split():
            if "=" in item:
                key, value = item.split("=", 1)
                record[key] = value
        if record.get("mode") == "layer2":
            records.append(record)
    return records


def build_summary(log: Path, skip_first_per_seq: int) -> dict[str, Any]:
    records = parse_profile_log(log)
    by_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, record in enumerate(records):
        if "step_breakdown" not in record:
            raise RuntimeError(f"record {idx} missing step_breakdown: {record}")
        values = [int(v) for v in record["step_breakdown"].split(",") if v]
        if len(values) != len(STEP_NAMES):
            raise RuntimeError(
                f"record {idx} step_breakdown has {len(values)} fields; "
                f"expected {len(STEP_NAMES)}"
            )
        seq = int(record["seq"])
        by_seq[seq].append(
            {
                "elapsed_us": int(record["elapsed_us"]),
                "steps_us": values,
                "layer": int(record.get("layer", -1)),
                "offset_before": int(record.get("offset_before", 0)),
                "offset_after": int(record.get("offset_after", 0)),
            }
        )

    seq_summaries: dict[str, Any] = {}
    for seq, seq_records in sorted(by_seq.items()):
        if skip_first_per_seq:
            seq_records = seq_records[skip_first_per_seq:]
        elapsed = [float(r["elapsed_us"]) for r in seq_records]
        step_summary: dict[str, Any] = {}
        for pos, name in enumerate(STEP_NAMES):
            step_values = [float(r["steps_us"][pos]) for r in seq_records]
            step_summary[name] = summarize(step_values)
        top_steps = sorted(
            (
                {
                    "step": name,
                    "p50_us": stats["p50_us"],
                    "mean_us": stats["mean_us"],
                }
                for name, stats in step_summary.items()
            ),
            key=lambda item: item["p50_us"] or 0.0,
            reverse=True,
        )
        seq_summaries[str(seq)] = {
            "records": len(seq_records),
            "elapsed_us": summarize(elapsed),
            "steps": step_summary,
            "top_steps_by_p50_us": top_steps,
        }

    return {
        "meta": {
            "log": str(log),
            "step_names": STEP_NAMES,
            "records": len(records),
            "skip_first_per_seq": skip_first_per_seq,
        },
        "by_seq": seq_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--skip-first-per-seq",
        type=int,
        default=0,
        help="Drop warmup records from each seq group before summarizing.",
    )
    args = parser.parse_args()

    output = build_summary(args.log, args.skip_first_per_seq)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
