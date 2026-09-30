#!/usr/bin/env python3
"""Paired-session bootstrap acceptance; never filter failed measured rows."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics

METRICS = ("ttft_s", "decode_tps", "e2e_s", "common_decode_tps")


def geometric(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def load_sessions(paths):
    sessions = {}
    for path in paths:
        if not path.stem.rsplit("-s", 1)[-1].isdigit():
            continue
        data = json.loads(path.read_text())
        sid = data["session"]
        if sid in sessions:
            raise ValueError(f"duplicate session {sid}")
        rows = {r["id"]: r for r in data["runs"] if r["category"] != "warmup"}
        if len(rows) != 6 or any(not r["valid"] for r in rows.values()):
            raise ValueError(f"incomplete or invalid session: {path}")
        sessions[sid] = rows
    return sessions


def compare(ours, rival, metric, draws=10000):
    if set(ours) != set(rival):
        raise ValueError("session ids must be paired")
    sessions = sorted(ours)
    if not sessions:
        raise ValueError("no data")
    prompts = sorted(ours[sessions[0]])
    for s in sessions:
        if sorted(ours[s]) != prompts or sorted(rival[s]) != prompts:
            raise ValueError("prompt sets differ")

    def ratios(indices, category=None):
        return [
            statistics.median(ours[s][p][metric] for s in indices)
            / statistics.median(rival[s][p][metric] for s in indices)
            for p in prompts
            if category is None or ours[sessions[0]][p]["category"] == category
        ]

    point = geometric(ratios(sessions))
    rng = random.Random(20260929)
    boot = sorted(
        geometric(ratios(rng.choices(sessions, k=len(sessions)))) for _ in range(draws)
    )
    lo, hi = boot[int(draws * 0.025)], boot[min(draws - 1, int(draws * 0.975))]
    high = metric.endswith("tps")
    category = {c: geometric(ratios(sessions, c)) for c in ("code", "knowledge")}
    categories_ok = all(v >= 1 / 1.03 if high else v <= 1.03 for v in category.values())
    sufficient = len(sessions) >= 8
    better = sufficient and categories_ok and (lo > 1 if high else hi < 1)
    no_worse = (
        sufficient
        and categories_ok
        and ((point >= 1 and lo >= 1 / 1.03) if high else (point <= 1 and hi <= 1.03))
    )
    return dict(
        ratio=point,
        ci95=[lo, hi],
        categories=category,
        better=better,
        no_worse=no_worse,
        sessions=len(sessions),
        per_prompt=dict(zip(prompts, ratios(sessions))),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    apps = {
        name: load_sessions(
            sorted(args.directory.glob(f"{args.label}-{name}-s[0-9]*.json"))
        )
        for name in ("ironmlx", "omlx", "splash", "tensorfold")
    }
    comparisons = {
        name: {m: compare(apps["ironmlx"], apps[name], m) for m in METRICS}
        for name in ("omlx", "splash", "tensorfold")
    }
    gates = dict(
        omlx=all(comparisons["omlx"][m]["better"] for m in METRICS),
        tensorfold=sum(comparisons["tensorfold"][m]["no_worse"] for m in METRICS[:3])
        >= 2,
    )
    # The common-tokenizer audit must support any TPS claim.
    tf = comparisons["tensorfold"]
    gates["token_count_audit"] = (
        not tf["decode_tps"]["no_worse"] or tf["common_decode_tps"]["no_worse"]
    )
    report = dict(
        comparisons=comparisons,
        performance_gate=all(gates.values()),
        gates=gates,
        note="Performance only: numerical, output-quality and run-environment audits remain separate required gates.",
    )
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
