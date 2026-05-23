"""P5i.a baseline aggregator: read ironmlx + omlx iron-bench CSVs, produce per-PP
summary JSON with medians + delta_pct + +10%-target threshold.

CSV header (from iron-bench --format csv; P5i.a does not use request-id join):
target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,
prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason
"""

from __future__ import annotations
import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Local import — tools/ is the package root for this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p5h_2a_se_analysis import bootstrap_median_ci  # noqa: E402

EXPECTED_PPS = (128, 512)
# Per-PP expected RUNS (iron-bench --runs argument). PP=512 bumped to 15 per
# P5h+2.a T1 to absorb fresh-spawn JIT/cache fill-in variance that RUNS=7 left
# at ~6.85% final envelope; RUNS=15 brings the final uncertainty envelope to
# 1.94% (within the ±2% spec § 7.2 noise band). See
# docs/p5h+2-a-pp512-protocol.md.
EXPECTED_RUNS_PER_PP: dict[int, int] = {
    128: 7,
    512: 15,  # P5h+2.a T1: bumped from 7 to absorb fresh-spawn JIT variance
}


def expected_runs_for_pp(pp: int) -> int:
    try:
        return EXPECTED_RUNS_PER_PP[pp]
    except KeyError as exc:
        raise SystemExit(f"unexpected pp_target={pp}; expected {EXPECTED_PPS}") from exc


def expected_run_idx_set_for_pp(pp: int) -> frozenset[int]:
    return frozenset(range(expected_runs_for_pp(pp)))


def load_pp_tps_by_pp(csv_path: Path) -> dict[int, list[float]]:
    """Parse iron-bench CSV at `csv_path` and return {pp_target: [pp_tps,...]}.

    Per Codex P5i.a P2 finding #4, this function fails HARD on any malformed
    row (missing column, unparseable int/float, non-positive pp_tps, unknown
    pp_target) and validates that each PP has exactly the expected run_idx
    set with no duplicates and no gaps. Silent skip-on-error is intentionally
    NOT used: that behavior could mask missing or duplicated measurement runs
    and produce a summary that misreports the underlying sweep.
    """
    by_pp_runs: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        required = {"target", "pp_target", "run_idx", "pp_tps"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{csv_path}: missing required CSV columns: {sorted(missing)}"
            )
        # csv.DictReader starts data rows at line 2 (line 1 = header).
        for line_no, row in enumerate(reader, start=2):
            try:
                pp_raw = row["pp_target"]
                run_idx_raw = row["run_idx"]
                pp_tps_raw = row["pp_tps"]
            except KeyError as exc:
                raise SystemExit(
                    f"{csv_path}: malformed row at line {line_no}: missing column {exc}; row={row!r}"
                ) from exc
            try:
                pp = int(pp_raw)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"{csv_path}: malformed row at line {line_no}: pp_target={pp_raw!r} not int; row={row!r}"
                ) from exc
            try:
                run_idx = int(run_idx_raw)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"{csv_path}: malformed row at line {line_no}: run_idx={run_idx_raw!r} not int; row={row!r}"
                ) from exc
            try:
                pp_tps = float(pp_tps_raw)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"{csv_path}: malformed row at line {line_no}: pp_tps={pp_tps_raw!r} not float; row={row!r}"
                ) from exc
            if pp not in EXPECTED_PPS:
                raise SystemExit(
                    f"{csv_path}: line {line_no}: unexpected pp_target={pp}; expected {EXPECTED_PPS}"
                )
            if pp_tps <= 0:
                raise SystemExit(
                    f"{csv_path}: line {line_no}: non-positive pp_tps={pp_tps} for pp={pp}"
                )
            by_pp_runs[pp].append((run_idx, pp_tps))
    # Per-PP validation: exact row count + run_idx set must match the per-PP
    # expected set (no duplicates, no gaps, no extras).
    by_pp: dict[int, list[float]] = {}
    for pp in EXPECTED_PPS:
        rows = by_pp_runs.get(pp, [])
        got = len(rows)
        expected_runs = expected_runs_for_pp(pp)
        if got != expected_runs:
            raise SystemExit(
                f"{csv_path}: expected {expected_runs} measured rows for PP={pp}, got {got}"
            )
        run_idxs = [r for r, _ in rows]
        observed_set = set(run_idxs)
        if len(observed_set) != len(run_idxs):
            # Duplicate run_idx detected.
            dupes = sorted(idx for idx, count in Counter(run_idxs).items() if count > 1)
            raise SystemExit(
                f"{csv_path}: PP={pp} has duplicate run_idx values: {dupes}; observed={sorted(run_idxs)}"
            )
        expected_set = expected_run_idx_set_for_pp(pp)
        if observed_set != expected_set:
            raise SystemExit(
                f"{csv_path}: PP={pp} run_idx set was {sorted(observed_set)} "
                f"but expected {sorted(expected_set)}"
            )
        by_pp[pp] = [tps for _, tps in rows]
    return by_pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ironmlx-csv", required=True, type=Path)
    ap.add_argument("--omlx-csv", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    ironmlx = load_pp_tps_by_pp(args.ironmlx_csv)
    omlx = load_pp_tps_by_pp(args.omlx_csv)

    summary = {"per_pp": {}}
    # Fixed seed so the per-PP bootstrap CI is reproducible across runs on the
    # same inputs (P5h+2.a T3). Bootstrap-resample is a SCREENING metric (see
    # p5h_2a_se_analysis docstring); caller must still pair it with the
    # between-sweep half-range when deciding final uncertainty envelope.
    rng = random.Random(42)
    for pp in EXPECTED_PPS:
        i_tps = ironmlx[pp]
        o_tps = omlx[pp]
        i_med = statistics.median(i_tps)
        o_med = statistics.median(o_tps)
        delta_pct = (i_med - o_med) / o_med * 100.0
        passes_plus10 = (delta_pct is not None) and (delta_pct >= 10.0)
        i_ci = bootstrap_median_ci(
            i_tps, subset_size=len(i_tps), iterations=1000, rng=rng
        )
        o_ci = bootstrap_median_ci(
            o_tps, subset_size=len(o_tps), iterations=1000, rng=rng
        )
        summary["per_pp"][str(pp)] = {
            "ironmlx_runs": len(i_tps),
            "omlx_runs": len(o_tps),
            "ironmlx_pp_tps_median": i_med,
            "omlx_pp_tps_median": o_med,
            "delta_pct": delta_pct,
            "passes_plus10_target": passes_plus10,
            "ironmlx_pp_tps_ci95_low": i_ci["ci95_low"],
            "ironmlx_pp_tps_ci95_high": i_ci["ci95_high"],
            "ironmlx_pp_tps_ci95_half_width_pct": i_ci["ci95_half_width_pct"],
            "omlx_pp_tps_ci95_low": o_ci["ci95_low"],
            "omlx_pp_tps_ci95_high": o_ci["ci95_high"],
            "omlx_pp_tps_ci95_half_width_pct": o_ci["ci95_half_width_pct"],
        }

    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"OK: wrote {args.out_json}")
    for pp, row in summary["per_pp"].items():
        i, o, d = (
            row["ironmlx_pp_tps_median"],
            row["omlx_pp_tps_median"],
            row["delta_pct"],
        )
        flag = "PASS" if row["passes_plus10_target"] else "MISS"
        i_hw = row["ironmlx_pp_tps_ci95_half_width_pct"]
        o_hw = row["omlx_pp_tps_ci95_half_width_pct"]
        print(
            f"  PP={pp}: ironmlx={i:.2f} (±{i_hw:.2f}%) "
            f"omlx={o:.2f} (±{o_hw:.2f}%) delta={d:+.2f}% {flag}"
        )


if __name__ == "__main__":
    main()
