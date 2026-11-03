#!/usr/bin/env python3
"""Summarize ANE_REAL_OUTPUT pairs without treating execution as qualification.

Usage: python3 scripts/analyze-ane-qualification.py DIRECTORY --output REPORT.json
Inputs named FAMILY-FIXTURE-ROUTE-{auto,off}.json are produced by the explicitly
ignored ane_dense_real integration test. Raw arrays remain local.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def summarize(directory):
    rows = []
    for baseline in sorted(directory.glob("*-off.json")):
        candidate = baseline.with_name(baseline.name.replace("-off.json", "-auto.json"))
        if not candidate.exists():
            continue
        off = json.loads(baseline.read_text())
        auto = json.loads(candidate.read_text())
        assert off["prompt_tokens"] == auto["prompt_tokens"], baseline.name
        pairs = list(zip(off["teacher_forced"], auto["teacher_forced"]))
        assert len(off["teacher_forced"]) == len(auto["teacher_forced"])
        assert all(a["target"] == b["target"] for a, b in pairs)
        assert off["completed_ane_jobs"] == 0
        assert auto["completed_ane_jobs"] > 0
        assert (
            off["surface_bytes_after_unload"] == auto["surface_bytes_after_unload"] == 0
        )
        deltas = [b["nll"] - a["nll"] for a, b in pairs]
        assert all(math.isfinite(x) for x in deltas)
        rows.append(
            dict(
                case=baseline.stem.removesuffix("-off"),
                samples=len(pairs),
                mean_nll_delta=sum(deltas) / len(deltas),
                max_abs_nll_delta=max(map(abs, deltas)),
                top1_agreement=sum(a["top1"] == b["top1"] for a, b in pairs)
                / len(pairs),
                completed_ane_jobs=auto["completed_ane_jobs"],
                ane_surface_bytes=auto["ane_surface_bytes"],
                gpu_prefill_seconds=off["prefill_seconds"],
                ane_prefill_seconds=auto["prefill_seconds"],
                baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            )
        )
    assert rows, "No complete auto/off pairs"
    return dict(
        qualification="measurements_only",
        cases=rows,
        limits=[
            "Teacher-forced samples are not a broad perplexity evaluation.",
            "Cross-process cold timings do not qualify warm acceleration.",
            "Top-1 agreement is not a claim of exact generation parity.",
        ],
    )


def summarize_paired(directory):
    cases = []
    for path in sorted(directory.glob("*-paired-64.json")):
        data = json.loads(path.read_text())
        rows = data["paired_stats"]
        assert len(rows) == 64
        assert all((r["jobs"] > 0) == r["enabled"] for r in rows)
        ratios = []
        for start in range(4, len(rows), 4):
            block = rows[start : start + 4]
            assert [r["enabled"] for r in block] == [False, True, True, False]
            gpu = statistics.mean(r["seconds"] for r in block if not r["enabled"])
            ane = statistics.mean(r["seconds"] for r in block if r["enabled"])
            ratios.append(math.log(gpu / ane))
        rng = random.Random(1701)
        boot = sorted(
            math.expm1(statistics.mean(rng.choices(ratios, k=len(ratios))))
            for _ in range(5000)
        )
        cases.append(
            dict(
                case=path.stem,
                warm_trials=60,
                paired_speedup=math.expm1(statistics.mean(ratios)),
                bootstrap_ci95=[boot[125], boot[4874]],
                gpu_mean_seconds=statistics.mean(
                    r["seconds"] for r in rows[4:] if not r["enabled"]
                ),
                ane_mean_seconds=statistics.mean(
                    r["seconds"] for r in rows[4:] if r["enabled"]
                ),
                first_four=rows[:4],
                binary_sha256=data.get("binary_sha256"),
            )
        )
    assert cases, "No paired benchmark files"
    return dict(
        qualification="measurements_only",
        cases=cases,
        limits=[
            "Each case uses one fixed 2048-token B1 prompt; this is not App-wide throughput.",
            "Only the first four cold/preparation trials are excluded; all 60 remaining trials are retained.",
            "Deterministic bootstrap resamples 15 ABBA blocks; it does not cover other machines or inputs.",
            "Warm speedup does not amortize first-use preparation or private-runtime cache storage.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--paired", action="store_true", help="Analyze 64-trial ABBA fixtures"
    )
    args = parser.parse_args()
    result = (
        summarize_paired(args.directory) if args.paired else summarize(args.directory)
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["cases"]:
        if args.paired:
            print(
                f"{row['case']}: speedup {row['paired_speedup']:.2%}, CI95 {row['bootstrap_ci95']}"
            )
            continue
        print(
            f"{row['case']}: NLL delta {row['mean_nll_delta']:+.6f}; "
            f"top-1 {row['top1_agreement']:.2%}; ANE jobs {row['completed_ane_jobs']}"
        )


if __name__ == "__main__":
    main()
