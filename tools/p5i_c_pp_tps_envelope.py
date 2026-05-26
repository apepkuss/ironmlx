"""P5i.c Phase 0 — per-PP pp_tps envelope across >=3 independent sweep repeats.

Per spec § 4.2.3 + plan Step 5.1: final pp_tps envelope =
MAX(within-sweep bootstrap CI95 half-width, between-sweep half-range),
reusing P5h+2.a methodology. Hard-fails on per-PP row count mismatch.

When --compare-repeat-csv inputs are supplied, additionally emits ironmlx-vs-omlx
delta median + conservative CI bounds.

CLI:
    python tools/p5i_c_pp_tps_envelope.py \\
        --pp 128 \\
        --repeat-csv /tmp/p5i-c-phase-0-r1-pp128-production/bench.csv \\
        --repeat-csv /tmp/p5i-c-phase-0-r2-pp128-production/bench.csv \\
        --repeat-csv /tmp/p5i-c-phase-0-r3-pp128-production/bench.csv \\
        [--compare-repeat-csv /tmp/p5i-c-phase-0-omlx-r1-pp128/bench.csv ...] \\
        --out-json /tmp/p5i-c-phase-0-pp128-envelope.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p5h_2a_se_analysis import bootstrap_median_ci  # noqa: E402

# Legacy default; can be overridden via --expected-runs at call sites
# (P5h+2.d uses 15 for both PPs).
DEFAULT_EXPECTED_RUNS: dict[int, int] = {128: 7, 512: 15}
STANDARD_ACCEPTANCE_THRESHOLD_PCT = 2.0
SMALL_PP_ACCEPTANCE_THRESHOLD_PCT = 2.5
SMALL_PP_ACCEPTANCE_PP = 128


def acceptance_target_for_pp(pp: int) -> tuple[float, str]:
    """Return the per-PP envelope target and policy label.

    PP=128 uses the small-PP acceptance threshold because fixed TTFT jitter is
    a larger fraction of short-prefill latency. All other PPs keep the standard
    threshold.
    """
    if pp == SMALL_PP_ACCEPTANCE_PP:
        return SMALL_PP_ACCEPTANCE_THRESHOLD_PCT, "small_pp_acceptance_threshold"
    return STANDARD_ACCEPTANCE_THRESHOLD_PCT, "standard_acceptance_threshold"


def load_pp_tps(csv_path: Path, pp: int, expected_runs: int) -> list[float]:
    """Read pp_tps column from iron-bench CSV; hard-fail on shape mismatch.

    Validates: pp_target == pp on every row, pp_tps is finite-float, row count
    matches expected_runs.
    """
    if not csv_path.exists():
        raise SystemExit(f"{csv_path}: not found")
    rows: list[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            try:
                row_pp = int(row["pp_target"])
            except (KeyError, ValueError) as exc:
                raise SystemExit(
                    f"{csv_path}:{line_no}: bad pp_target: {row}: {exc}"
                ) from exc
            if row_pp != pp:
                raise SystemExit(
                    f"{csv_path}:{line_no}: pp_target={row_pp} but expected {pp}"
                )
            try:
                tps = float(row["pp_tps"])
            except (KeyError, ValueError) as exc:
                raise SystemExit(
                    f"{csv_path}:{line_no}: bad pp_tps: {row}: {exc}"
                ) from exc
            if tps <= 0:
                raise SystemExit(f"{csv_path}:{line_no}: non-positive pp_tps={tps}")
            rows.append(tps)
    if len(rows) != expected_runs:
        raise SystemExit(
            f"{csv_path}: expected {expected_runs} rows for PP={pp}, got {len(rows)}"
        )
    return rows


def compute_pp_tps_envelope(
    repeat_csvs: list[Path], pp: int, expected_runs: int
) -> dict:
    """Per-repeat medians + within bootstrap CI95 + between-sweep half-range.

    Final envelope = MAX(within CI95 half-width max, between half-range pct).
    Verdict PASS if envelope <= the per-PP acceptance target. PP=128 uses the
    small-PP acceptance threshold (2.5%); all other PPs use the standard 2.0%
    threshold.
    """
    if len(repeat_csvs) < 3:
        raise SystemExit("need >=3 repeat-csv inputs for between-sweep envelope")
    per_repeat = []
    for path in repeat_csvs:
        tps = load_pp_tps(path, pp, expected_runs)
        med = median(tps)
        rng = random.Random(42)
        ci = bootstrap_median_ci(tps, subset_size=len(tps), iterations=1000, rng=rng)
        # P5h+2.d spec § 4.1: per-repeat diagnostic fields (no gate logic).
        # NOTE: when len(tps) == 3, first_3 and last_3 are identical,
        # so trailing_slowdown_pct is always 0% and fast_start_drop_pct
        # reduces to (max(tps) / median(tps) - 1) * 100. Diagnostic fields
        # are most meaningful when len(tps) >= 6.
        if len(tps) >= 3:
            first_3 = tps[:3]
            last_3 = tps[-3:]
            first_3_med = median(first_3)
            last_3_med = median(last_3)
            trailing_slowdown_pct = (last_3_med / first_3_med - 1) * 100
            fast_start_drop_pct = (max(first_3) / last_3_med - 1) * 100
        else:
            first_3_med = None
            last_3_med = None
            trailing_slowdown_pct = None
            fast_start_drop_pct = None
        per_repeat.append(
            {
                "path": str(path),
                "n": len(tps),
                "median": med,
                "ci95_low": ci["ci95_low"],
                "ci95_high": ci["ci95_high"],
                "ci95_half_width_pct": ci["ci95_half_width_pct"],
                "first_3_runs_median_pp_tps": first_3_med,
                "last_3_runs_median_pp_tps": last_3_med,
                "trailing_slowdown_pct": trailing_slowdown_pct,
                "fast_start_drop_pct": fast_start_drop_pct,
            }
        )

    medians = [r["median"] for r in per_repeat]
    mean_med = sum(medians) / len(medians)
    if mean_med == 0:
        raise SystemExit("mean of medians is 0; cannot compute envelope")
    between_half_range_pct = (max(medians) - min(medians)) / mean_med * 100 / 2
    within_max_pct = max(r["ci95_half_width_pct"] for r in per_repeat)
    final_envelope_pct = max(within_max_pct, between_half_range_pct)
    target_pct, target_policy = acceptance_target_for_pp(pp)
    verdict = "PASS" if final_envelope_pct <= target_pct else "FAIL"

    return {
        "pp": pp,
        "per_repeat": per_repeat,
        "medians": medians,
        "mean_median": mean_med,
        "between_sweep_half_range_pct": between_half_range_pct,
        "within_sweep_ci95_max_pct": within_max_pct,
        "final_uncertainty_envelope_pct": final_envelope_pct,
        "target_pct": target_pct,
        "target_policy": target_policy,
        "verdict": verdict,
    }


def compute_vs_omlx_delta(ironmlx_envelope: dict, omlx_envelope: dict, pp: int) -> dict:
    """Delta median + conservative CI bounds combining both sides' envelopes.

    delta_pct_median = (ironmlx_median / omlx_median - 1) * 100
    Conservative bounds: combine the worst-case envelope from each side.
    """
    iron_med = ironmlx_envelope["mean_median"]
    omlx_med = omlx_envelope["mean_median"]
    if omlx_med == 0:
        raise SystemExit("omlx mean of medians is 0; cannot compute delta")
    delta_pct = (iron_med / omlx_med - 1) * 100
    # Conservative CI: worst-case envelope from either side (each is half-width
    # in percentage points of its own baseline; sum is upper bound on combined
    # uncertainty since the two distributions are independent).
    iron_env = ironmlx_envelope["final_uncertainty_envelope_pct"]
    omlx_env = omlx_envelope["final_uncertainty_envelope_pct"]
    conservative_half_width_pct = iron_env + omlx_env
    return {
        "pp": pp,
        "ironmlx_mean_median": iron_med,
        "omlx_mean_median": omlx_med,
        "delta_pct_median": delta_pct,
        "delta_pct_conservative_low": delta_pct - conservative_half_width_pct,
        "delta_pct_conservative_high": delta_pct + conservative_half_width_pct,
        "conservative_half_width_pct": conservative_half_width_pct,
        "ironmlx_envelope_pct": iron_env,
        "omlx_envelope_pct": omlx_env,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeat-csv", type=Path, action="append", required=True)
    p.add_argument(
        "--compare-repeat-csv",
        type=Path,
        action="append",
        default=None,
        help="optional omlx (or other comparator) repeats for vs-comparator delta",
    )
    p.add_argument("--pp", type=int, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument(
        "--expected-runs",
        type=int,
        default=None,
        help="Per-PP expected row count (overrides DEFAULT_EXPECTED_RUNS). "
        "Required when PP not in default map (e.g., P5h+2.d uses 15 for both).",
    )
    args = p.parse_args()
    expected = (
        args.expected_runs
        if args.expected_runs is not None
        else DEFAULT_EXPECTED_RUNS.get(args.pp)
    )
    if expected is None:
        raise SystemExit(
            f"--expected-runs required for pp={args.pp} (no default registered)"
        )
    ironmlx = compute_pp_tps_envelope(args.repeat_csv, args.pp, expected)
    result: dict = {"ironmlx": ironmlx}
    if args.compare_repeat_csv:
        comparator = compute_pp_tps_envelope(
            args.compare_repeat_csv, args.pp, expected
        )
        result["comparator"] = comparator
        result["delta_vs_comparator"] = compute_vs_omlx_delta(
            ironmlx, comparator, args.pp
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
