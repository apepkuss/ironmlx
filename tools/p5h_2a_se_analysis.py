"""P5h+2.a Phase 0 bootstrap-resample + drift diagnostics.

Reads iron-bench CSV (--format csv) and computes:
- For each RUNS subset size N in {7, 15, 21, 30}: 95% CI of median pp_tps
  via 1000 bootstrap samples with replacement
- Per-run drift diagnostics: linear regression of pp_tps vs run_idx and
  ttft_ms vs run_idx, with slope + r_squared + normal-approx p-value
- Output JSON with per-N SE and drift diagnostics

Per P5h+2.a spec § 5.1 + § 4.1. Bootstrap-resample is a SCREENING metric
only; between-sweep validation (T1 repeat sweeps) is required for
Outcome (a) per spec § 3.1.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

BOOTSTRAP_ITERATIONS = 1000
SUBSET_SIZES = (7, 15, 21, 30)


def load_csv(csv_path: Path) -> tuple[list[float], list[float], list[int]]:
    """Read iron-bench CSV. Return (pp_tps_list, ttft_ms_list, run_idx_list)."""
    pp_tps_list: list[float] = []
    ttft_ms_list: list[float] = []
    run_idx_list: list[int] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        required = {"pp_tps", "ttft_ms", "run_idx"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{csv_path}: missing required CSV columns: {sorted(missing)}"
            )
        for row_num, row in enumerate(reader, start=2):
            try:
                pp_tps = float(row["pp_tps"])
                ttft_ms = float(row["ttft_ms"])
                run_idx = int(row["run_idx"])
            except (KeyError, ValueError) as e:
                raise SystemExit(
                    f"{csv_path}:{row_num}: malformed row: {e}; row={dict(row)}"
                )
            if pp_tps <= 0:
                raise SystemExit(f"{csv_path}:{row_num}: non-positive pp_tps={pp_tps}")
            pp_tps_list.append(pp_tps)
            ttft_ms_list.append(ttft_ms)
            run_idx_list.append(run_idx)
    return pp_tps_list, ttft_ms_list, run_idx_list


def bootstrap_median_ci(
    values: list[float], subset_size: int, iterations: int, rng: random.Random
) -> dict:
    """Bootstrap 95% CI of median for subset of given size.

    For 1000 iterations: draw random subset with replacement, compute median.
    Return dict with point_median, ci95_low, ci95_high, ci95_half_width_pct.
    """
    if subset_size > len(values):
        raise SystemExit(
            f"subset_size={subset_size} exceeds available data N={len(values)}"
        )
    medians = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in range(subset_size)]
        medians.append(statistics.median(sample))
    medians.sort()
    point_median = statistics.median(values)
    ci95_low = medians[int(0.025 * iterations)]
    ci95_high = medians[int(0.975 * iterations)]
    ci95_half_width_pct = (ci95_high - ci95_low) / 2.0 / point_median * 100.0
    return {
        "subset_size": subset_size,
        "point_median": point_median,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "ci95_half_width_pct": ci95_half_width_pct,
        "bootstrap_iterations": iterations,
    }


def _normal_two_sided_p_value(z_score: float) -> float:
    """Two-sided normal-approx p-value from a z/t-like statistic."""
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def linear_regression(x_vals: list[float], y_vals: list[float]) -> dict:
    """Simple linear regression. Returns slope, intercept, r_squared, p_value.

    No scipy dependency (stdlib only). p_value is a normal approximation from
    the slope t-statistic; n is small, so downstream interpretation must still
    pair it with slope magnitude and r_squared.
    """
    n = len(x_vals)
    if n < 2:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "p_value": 1.0,
            "n": n,
        }
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    num = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    den_x = sum((x_vals[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((y_vals[i] - mean_y) ** 2 for i in range(n))
    if den_x == 0.0 or den_y == 0.0:
        return {
            "slope": 0.0,
            "intercept": mean_y,
            "r_squared": 0.0,
            "p_value": 1.0,
            "n": n,
        }
    slope = num / den_x
    intercept = mean_y - slope * mean_x
    r_squared = (num**2) / (den_x * den_y)
    if n < 3:
        p_value = 1.0
    elif r_squared >= 1.0:
        p_value = 0.0
    else:
        residual_ss = den_y * (1.0 - r_squared)
        if residual_ss <= 0.0:
            p_value = 0.0
        else:
            stderr_slope = math.sqrt((residual_ss / (n - 2)) / den_x)
            p_value = (
                _normal_two_sided_p_value(slope / stderr_slope)
                if stderr_slope > 0.0
                else 0.0
            )
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "p_value_method": "normal_approx_slope_t",
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input", required=True, type=Path, help="iron-bench --format csv input"
    )
    ap.add_argument("--out-json", required=True, type=Path, help="output JSON")
    ap.add_argument(
        "--seed", type=int, default=42, help="bootstrap RNG seed (default 42)"
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pp_tps, ttft_ms, run_idx = load_csv(args.input)
    print(f"loaded N={len(pp_tps)} runs from {args.input}")

    se_per_subset = {}
    for n in SUBSET_SIZES:
        if n > len(pp_tps):
            print(f"  skipping subset_size={n} (data N={len(pp_tps)} too small)")
            continue
        result = bootstrap_median_ci(pp_tps, n, BOOTSTRAP_ITERATIONS, rng)
        se_per_subset[str(n)] = result
        print(
            f"  N={n}: point_median={result['point_median']:.2f} "
            f"ci95=[{result['ci95_low']:.2f}, {result['ci95_high']:.2f}] "
            f"half_width={result['ci95_half_width_pct']:.2f}%"
        )

    pp_tps_drift = linear_regression([float(i) for i in run_idx], pp_tps)
    ttft_drift = linear_regression([float(i) for i in run_idx], ttft_ms)
    print(
        f"pp_tps_vs_run_idx: slope={pp_tps_drift['slope']:.4f} "
        f"r_squared={pp_tps_drift['r_squared']:.4f} p={pp_tps_drift['p_value']:.4g}"
    )
    print(
        f"ttft_ms_vs_run_idx: slope={ttft_drift['slope']:.4f} "
        f"r_squared={ttft_drift['r_squared']:.4f} p={ttft_drift['p_value']:.4g}"
    )

    output = {
        "input_csv": str(args.input),
        "input_n_runs": len(pp_tps),
        "bootstrap_seed": args.seed,
        "se_per_subset": se_per_subset,
        "drift_diagnostics": {
            "pp_tps_vs_run_idx": pp_tps_drift,
            "ttft_ms_vs_run_idx": ttft_drift,
        },
    }
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
