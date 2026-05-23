# P5h+2.a PP=512 Measurement Protocol Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine a PP=512 iron-bench measurement protocol that supports ±2% decision-making (Approach A), or document why ±2% is unachievable and redefine spec § 7.2 noise band for PP=512 with root cause / quantified ambiguity + aggregator per-PP 95% CI emission (Approach C fallback).

**Architecture:** 6-task exploration/characterization phase, measurement-only (no production ironmlx src changes). T0 captures RUNS=30 baseline + bootstrap-resample SE + drift diagnostics. T1 tries 2-3 alternate (RUNS, cooldown, preheat) configs; near-target ones promoted to ≥3 independent spawn+sweep repeats for between-sweep half-range validation; selection = lowest wall × MAX(within-sweep bootstrap CI, between-sweep median half-range). T2 (conditional, only if T1 misses ±2%) does powermetrics + per-run TTFT drift + request-state determinism control to identify variance root cause OR bound quantified ambiguity. T3 drafts the chosen protocol (Path A) or band redefinition (Path C) and updates aggregator for per-PP RUNS override + per-PP 95% CI emission. T4 validates with ≥3 independent repeats + omlx repeats if external-target decisions in scope, then commits only the validated protocol/band + aggregator changes. T5 close-out.

**Tech Stack:** Python 3.13 + uv + ruff for analysis script; iron-bench HTTP sweep harness (workspace member at `iron-bench/`); aggregator `tools/p5i_a_baseline_aggregate.py` extension; bootstrap-resample statistics via `statistics` + `random` stdlib (no scipy/numpy dep). No Rust src changes expected.

---

## File structure

**New branch:** `ironmlx-p5h+2-a-pp512-measurement` (fork from `ironmlx-p5i-a-gather-qmm-feasibility` HEAD `a90a85c`).

**Python tooling (T0/T3 create; T3 modify):**
- `tools/p5h_2a_se_analysis.py` (new in T0; bootstrap-resample SE + per-run drift diagnostics; CLI `--input <csv> --out-json <path>`)
- `tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py` (new in T0; pytest for bootstrap + drift functions with synthetic fixtures)
- `tools/p5i_a_baseline_aggregate.py` (modify in T3 — per-PP RUNS expectation enforced for each backend input; per-PP 95% CI emission integrated from `tools/p5h_2a_se_analysis.py`)
- `tools/p5h_aggregator/tests/test_p5i_a_baseline_aggregate_ci.py` (new in T3; pytest for CI emission)

**Docs (T3/T5 commit; T2 gitignored bench log):**
- `docs/p5h+2-a-pp512-protocol.md` (T3 drafts; T4 commits after validation — Path A protocol params OR Path C band redefinition)
- `docs/p5h+2-a-close-out.md` (T5 committed)
- `reports/p5h+2-a-bench-log.md` (gitignored — per-task experimental notes + per-config CI tables + T2 investigation findings)

**Spec (T3 conditional update — Path C):**
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` § 7.2.2 (T3 only if Approach C path — band amendment for PP=512)

**Memory (T5; outside repo):**
- `~/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_findings.md` (T5 — extend with P5h+2.a closure section)

**Output (gitignored, machine-local):**
- `/tmp/p5h-2a-env.sh` (env file pattern; SNAP + IRONMLX_PID + OMLX_PID + OMLX_MODEL_ID persist across steps)
- `/tmp/p5h-2a-t0-pp512-30runs.csv` (T0 RUNS=30 raw)
- `/tmp/p5h-2a-t0-se-analysis.json` (T0 bootstrap output)
- `/tmp/p5h-2a-t1-<config>-pp512.csv` (T1 per-config raw; e.g. `runs21-cooldown1x`)
- `/tmp/p5h-2a-t1-<config>-repeat<N>.csv` (T1 independent repeat per near-target candidate; N ∈ {1,2,3})
- `/tmp/p5h-2a-t1-<config>-se-analysis.json` (T1 per-config bootstrap + between-sweep half-range)
- `/tmp/p5h-2a-t2-powermetrics.log` (T2 conditional)
- `/tmp/p5h-2a-t4-validate-repeat<N>.csv` (T4 ≥3 validation repeats)
- `/tmp/p5h-2a-t4-validate-summary.json` (T4 final validation CI)

---

## Task 1: T0 — Phase 0 characterization (Approach A)

**Files:**
- Create: `tools/p5h_2a_se_analysis.py` (~120 lines; bootstrap-resample + drift diagnostics)
- Create: `tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py` (~80 lines; pytest fixtures)
- Output (gitignored): `/tmp/p5h-2a-pp512-30runs.csv`, `/tmp/p5h-2a-t0-se-analysis.json`, `/tmp/p5h-2a-t0-preheat.log`, `/tmp/p5h-2a-env.sh`, `reports/p5h+2-a-bench-log.md` (T0 section)

### Step 1.1: Branch + spec verification

- [ ] Create + checkout new branch:

```bash
cd /Users/xin/workspace/ironmlx-backend
git fetch
git checkout -b ironmlx-p5h+2-a-pp512-measurement ironmlx-p5i-a-gather-qmm-feasibility
git log --oneline -3
```

Expected: HEAD `a90a85c` (P5h+2.a spec commit) at top.

- [ ] Verify spec + plan committed on this branch:

```bash
ls docs/superpowers/specs/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol-design.md
ls docs/superpowers/plans/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol.md
```

Expected: both files present.

### Step 1.2: Pre-sweep cleanup + port check + env file init

- [ ] Clear stale outputs:

```bash
rm -f /tmp/p5h-2a-pp512-30runs.csv /tmp/p5h-2a-t0-se-analysis.json \
      /tmp/p5h-2a-t0-preheat.log /tmp/p5h-2a-env.sh \
      /tmp/p5h-2a-ironmlx-serve.log
```

- [ ] Verify port 18099 free:

```bash
lsof -i :18099 2>&1 || echo "PORT_18099_FREE"
```

Expected: PORT_18099_FREE. If occupied: `pkill -f "ironmlx serve" 2>/dev/null; sleep 3`.

### Step 1.3: Build ironmlx + spawn serve background + write env file

- [ ] Build release binary:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx
```

Expected: `Finished release` clean.

- [ ] Spawn ironmlx serve (use Bash run_in_background: true):

```bash
SNAP=$(ls -d ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/*/ | head -1)
echo "snap=$SNAP"
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx serve \
  --model "$SNAP" --port 18099 --host 127.0.0.1 > /tmp/p5h-2a-ironmlx-serve.log 2>&1 &
IRONMLX_PID=$!
echo "ironmlx_pid=$IRONMLX_PID"
{
  printf 'export SNAP=%q\n' "$SNAP"
  printf 'export IRONMLX_PID=%q\n' "$IRONMLX_PID"
} > /tmp/p5h-2a-env.sh
cat /tmp/p5h-2a-env.sh
```

- [ ] Wait healthz (up to 5min for model load):

```bash
source /tmp/p5h-2a-env.sh
for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:18099/healthz 2>/dev/null | grep -q ok; then
    echo "ready_after=${i}*5s"; break
  fi
  sleep 5
done
curl -sf http://127.0.0.1:18099/healthz || (echo "ironmlx not ready"; exit 1)
```

Expected: `ready_after=Nx5s` with N ≤ 60 + healthz returns ok.

### Step 1.4: 5-min thermal preheat (per P5h T0b H1 binding)

- [ ] Preheat with `iron-bench --runs 800` at PP=512 (per T4 finding default --runs 20 = 7s, insufficient for 5min):

```bash
source /tmp/p5h-2a-env.sh
cd /Users/xin/workspace/ironmlx-backend
PREHEAT_START=$(date +%s)
cargo run --release -p iron-bench -- \
  --target ironmlx_preheat=http://127.0.0.1:18099 \
  --model qwen3.5-moe --model-dir "$SNAP" \
  --prompt-len 512 --max-tokens 1 --runs 800 --warmup 0 --format csv > /tmp/p5h-2a-t0-preheat.log 2>&1
PREHEAT_EXIT=$?
PREHEAT_END=$(date +%s)
PREHEAT_WALL=$((PREHEAT_END - PREHEAT_START))
echo "preheat_exit=$PREHEAT_EXIT"
echo "preheat_wall_seconds=$PREHEAT_WALL"
```

Expected: `preheat_exit=0`; `preheat_wall_seconds >= 240` (≥4min; ideally ~300s). If wall < 240s: report BLOCKED — preheat insufficient; per spec § 7.5 must be ≥5min.

### Step 1.5: RUNS=30 measurement sweep at PP=512

- [ ] Run single sweep (warmup=1; NO --capture-server-request-id):

```bash
source /tmp/p5h-2a-env.sh
cd /Users/xin/workspace/ironmlx-backend
cargo run --release -p iron-bench -- \
  --target ironmlx=http://127.0.0.1:18099 \
  --model qwen3.5-moe --model-dir "$SNAP" \
  --prompt-len 512 --max-tokens 1 --runs 30 --warmup 1 --format csv > /tmp/p5h-2a-pp512-30runs.csv 2>>/tmp/p5h-2a-ironmlx-serve.log
echo "sweep_exit=$?"
wc -l /tmp/p5h-2a-pp512-30runs.csv
head -2 /tmp/p5h-2a-pp512-30runs.csv
tail -3 /tmp/p5h-2a-pp512-30runs.csv
```

Expected: `sweep_exit=0`; CSV has 31 lines (1 header + 30 data); header has columns `target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,...`.

### Step 1.6: Kill ironmlx + verify port free

- [ ] Cleanup:

```bash
source /tmp/p5h-2a-env.sh
kill $IRONMLX_PID 2>/dev/null
wait $IRONMLX_PID 2>/dev/null
sleep 3
lsof -i :18099 2>&1 || echo "PORT_18099_FREE_AFTER_KILL"
```

Expected: PORT_18099_FREE_AFTER_KILL.

### Step 1.7: Create `tools/p5h_2a_se_analysis.py`

- [ ] Create the file with exact content:

```python
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
```

- [ ] Lint clean:

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check tools/p5h_2a_se_analysis.py
uv run --with ruff ruff format --check tools/p5h_2a_se_analysis.py
```

Expected: both clean. If format fails, run `ruff format` to fix; re-check.

### Step 1.8: Create pytest for analysis script

- [ ] Create `tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py`:

```python
"""Pytest for tools/p5h_2a_se_analysis.py bootstrap + drift functions."""

from __future__ import annotations
import csv
import random
import sys
import tempfile
from pathlib import Path

# Ensure tools/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from p5h_2a_se_analysis import bootstrap_median_ci, linear_regression, load_csv  # noqa: E402


def write_csv(rows: list[dict], path: Path) -> None:
    """Write iron-bench-format CSV with required fields for the analysis."""
    fieldnames = [
        "target",
        "pp_target",
        "tg_target",
        "run_idx",
        "ttft_ms",
        "tg_tps",
        "tpot_ms",
        "pp_tps",
        "e2e_s",
        "prompt_tokens_local",
        "prompt_tokens_server",
        "completion_tokens_server",
        "cached_tokens",
        "finish_reason",
    ]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            full_row = {fn: row.get(fn, "") for fn in fieldnames}
            w.writerow(full_row)


def test_load_csv_basic():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.csv"
        rows = [
            {
                "target": "x",
                "pp_target": "512",
                "run_idx": "0",
                "pp_tps": "100.0",
                "ttft_ms": "5000.0",
            },
            {
                "target": "x",
                "pp_target": "512",
                "run_idx": "1",
                "pp_tps": "105.0",
                "ttft_ms": "5100.0",
            },
            {
                "target": "x",
                "pp_target": "512",
                "run_idx": "2",
                "pp_tps": "98.0",
                "ttft_ms": "5050.0",
            },
        ]
        write_csv(rows, p)
        pp_tps, ttft_ms, run_idx = load_csv(p)
        assert pp_tps == [100.0, 105.0, 98.0]
        assert ttft_ms == [5000.0, 5100.0, 5050.0]
        assert run_idx == [0, 1, 2]


def test_load_csv_rejects_negative_pp_tps():
    import pytest

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.csv"
        rows = [
            {
                "target": "x",
                "pp_target": "512",
                "run_idx": "0",
                "pp_tps": "-1.0",
                "ttft_ms": "100",
            }
        ]
        write_csv(rows, p)
        with pytest.raises(SystemExit):
            load_csv(p)


def test_bootstrap_median_ci_zero_variance():
    rng = random.Random(42)
    values = [100.0] * 30
    result = bootstrap_median_ci(values, subset_size=7, iterations=200, rng=rng)
    assert result["point_median"] == 100.0
    assert result["ci95_low"] == 100.0
    assert result["ci95_high"] == 100.0
    assert result["ci95_half_width_pct"] == 0.0


def test_bootstrap_median_ci_uniform_spread():
    rng = random.Random(42)
    values = [
        95.0,
        96.0,
        97.0,
        98.0,
        99.0,
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
    ]
    result = bootstrap_median_ci(values, subset_size=7, iterations=1000, rng=rng)
    assert 95.0 <= result["ci95_low"] <= result["point_median"]
    assert result["point_median"] <= result["ci95_high"] <= 105.0
    # Spread should be non-trivial but bounded
    assert 0.0 < result["ci95_half_width_pct"] < 10.0


def test_linear_regression_perfect_increasing():
    x = [float(i) for i in range(10)]
    y = [2.0 * i + 1.0 for i in range(10)]  # slope=2, intercept=1
    result = linear_regression(x, y)
    assert abs(result["slope"] - 2.0) < 1e-9
    assert abs(result["intercept"] - 1.0) < 1e-9
    assert abs(result["r_squared"] - 1.0) < 1e-9
    assert result["p_value"] == 0.0


def test_linear_regression_no_correlation():
    x = [float(i) for i in range(10)]
    y = [50.0] * 10  # constant
    result = linear_regression(x, y)
    assert result["slope"] == 0.0
    assert result["intercept"] == 50.0
    assert result["r_squared"] == 0.0
    assert result["p_value"] == 1.0
```

- [ ] Run pytest + lint clean:

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py -v
uv run --with ruff ruff check tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py
uv run --with ruff ruff format --check tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py
```

Expected: 6 tests pass; ruff clean.

### Step 1.9: Run analysis on T0 RUNS=30 data + record findings

- [ ] Run analysis script:

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run python tools/p5h_2a_se_analysis.py \
  --input /tmp/p5h-2a-pp512-30runs.csv \
  --out-json /tmp/p5h-2a-t0-se-analysis.json
cat /tmp/p5h-2a-t0-se-analysis.json
```

Expected output includes per-N (7, 15, 21, 30) bootstrap CI + drift diagnostics (slope + r_squared + p_value) for both pp_tps and ttft_ms.

- [ ] Create or append `reports/p5h+2-a-bench-log.md` T0 section (gitignored):

```bash
mkdir -p /Users/xin/workspace/ironmlx-backend/reports
cat >> /Users/xin/workspace/ironmlx-backend/reports/p5h+2-a-bench-log.md << 'BENCHLOG'

# P5h+2.a T0 — PP=512 RUNS=30 characterization

**Date**: $(date +%Y-%m-%d)
**Branch**: ironmlx-p5h+2-a-pp512-measurement
**Spawn**: 1 cycle (preheat ~5min + 30 measured runs)

## Within-sweep bootstrap CI per subset size (T0 raw /tmp/p5h-2a-pp512-30runs.csv)

| N (subset) | point_median pp_tps | ci95_low | ci95_high | half_width_pct |
|---|---|---|---|---|
| 7 | <FILL from JSON> | <FILL> | <FILL> | <FILL>% |
| 15 | <FILL> | <FILL> | <FILL> | <FILL>% |
| 21 | <FILL> | <FILL> | <FILL> | <FILL>% |
| 30 | <FILL> | <FILL> | <FILL> | <FILL>% |

## Drift diagnostics

- pp_tps vs run_idx: slope=<FILL>, r_squared=<FILL>
- ttft_ms vs run_idx: slope=<FILL>, r_squared=<FILL>

## Initial observations

- Bootstrap CI is a SCREENING metric ONLY (per spec § 5.1). Outcome (a) requires between-sweep repeat validation in T1.
- Drift signal interpretation: |r_squared| > 0.1 + significant slope → thermal/scheduler-state non-stationarity likely; T2 Approach B may be needed even if T1 bootstrap-CI hits target.
- T1 candidates to try based on T0 CI: <FILL after reading JSON>.
BENCHLOG
```

Fill `<FILL>` markers from the JSON output. Use `python3 -c "import json; d=json.load(open('/tmp/p5h-2a-t0-se-analysis.json')); ..."` style if convenient.

### Step 1.10: Hygiene + commit T0

- [ ] Final hygiene + commit:

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check tools/p5h_2a_se_analysis.py tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py
uv run --with ruff ruff format --check tools/p5h_2a_se_analysis.py tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py -v

git add tools/p5h_2a_se_analysis.py tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py

git commit -m "$(cat <<'COMMIT'
feat(p5h+2-a-t0): bootstrap-resample SE + drift diagnostics for PP=512 protocol fix

Per docs/superpowers/specs/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol-design.md § 4.1
and docs/superpowers/plans/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol.md T0.

New analysis script tools/p5h_2a_se_analysis.py:
* Reads iron-bench --format csv input
* For each RUNS subset size N in {7, 15, 21, 30}: 1000-iteration bootstrap
  resample of median pp_tps with replacement; computes 95% CI half-width
* Per-run drift diagnostics: linear regression of pp_tps vs run_idx and
  ttft_ms vs run_idx (slope + r_squared); surfaces non-stationarity
  before T1 candidate selection
* Stdlib only (no scipy/numpy dep); 1000 bootstrap iterations + seeded
  RNG for reproducibility
* Per spec § 5.1: bootstrap CI is a SCREENING metric ONLY; Outcome (a)
  requires between-sweep repeat validation in T1

New pytest tools/p5h_aggregator/tests/test_p5h_2a_se_analysis.py covers:
* CSV load + missing-column SystemExit + non-positive pp_tps rejection
* Bootstrap zero-variance and uniform-spread sanity
* Linear regression perfect-increasing and zero-correlation sanity

T0 RUNS=30 raw data + analysis JSON saved to /tmp/p5h-2a-* (gitignored);
findings + initial candidate hints in reports/p5h+2-a-bench-log.md
(gitignored per [feedback_no_reports_commit]).

Spec § 4.1 acceptance: T0 closes when RUNS=30 CSV + SE analysis JSON
+ per-RUNS within-sweep CI + drift diagnostics all reported. T0 is
characterization only and NOT allowed to prove Outcome (a) by itself.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
COMMIT
)"
git log --oneline -3
```

---

## Task 2: T1 — Phase 1A protocol candidates (Approach A)

**Files:**
- Output (gitignored): `/tmp/p5h-2a-t1-<config>-pp512.csv` per config; `/tmp/p5h-2a-t1-<config>-repeat<N>.csv` per repeat if near-target; `/tmp/p5h-2a-t1-<config>-se-analysis.json`; `reports/p5h+2-a-bench-log.md` (T1 section append)
- No committed files unless Path A landing requires (T3 handles spec)

### Step 2.1: Read T0 JSON + select 2-3 candidate configs

- [ ] Read T0 SE data:

```bash
cat /tmp/p5h-2a-t0-se-analysis.json
```

- [ ] Choose 2-3 candidate `(RUNS, cooldown_seconds, preheat_seconds)` configs based on T0 bootstrap CI per N. Suggested defaults if T0 doesn't dictate otherwise:
  - **Candidate A**: `RUNS=21, cooldown=current (~3s inter-PP), preheat=300s` (longer median; same cooldown)
  - **Candidate B**: `RUNS=15, cooldown=10s, preheat=300s` (medium median; 3× cooldown)
  - **Candidate C**: `RUNS=7, cooldown=current, preheat=600s` (current median; 2× preheat — thermal-saturation test)

Document choice + rationale in `reports/p5h+2-a-bench-log.md` T1 section.

### Step 2.2: Per-candidate sweep loop

For each candidate `<config>` ∈ {A, B, C}:

- [ ] Pre-sweep cleanup + port check:

```bash
lsof -i :18099 2>&1 || echo "PORT_FREE"
pkill -f "ironmlx serve" 2>/dev/null
sleep 3
```

- [ ] Spawn ironmlx + wait healthz (mirror Step 1.3-1.3 pattern; reuse env file):

```bash
source /tmp/p5h-2a-env.sh
MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx serve \
  --model "$SNAP" --port 18099 --host 127.0.0.1 > /tmp/p5h-2a-t1-<config>-serve.log 2>&1 &
IRONMLX_PID=$!
printf 'export IRONMLX_PID=%q\n' "$IRONMLX_PID" >> /tmp/p5h-2a-env.sh

for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:18099/healthz 2>/dev/null | grep -q ok; then
    echo "ready_after=${i}*5s"; break
  fi
  sleep 5
done
curl -sf http://127.0.0.1:18099/healthz || (echo "ironmlx not ready"; exit 1)
```

- [ ] Preheat for `<config>.preheat_seconds`:

```bash
source /tmp/p5h-2a-env.sh
cd /Users/xin/workspace/ironmlx-backend
PREHEAT_START=$(date +%s)
# Adjust --runs to approximate preheat target seconds (~5s per run at PP=512 → 800 runs for ~4min, 1200 for ~6min)
PREHEAT_RUNS_FOR_CONFIG=<800 for 300s target, 1200 for 600s target>
cargo run --release -p iron-bench -- \
  --target ironmlx_preheat=http://127.0.0.1:18099 \
  --model qwen3.5-moe --model-dir "$SNAP" \
  --prompt-len 512 --max-tokens 1 --runs $PREHEAT_RUNS_FOR_CONFIG --warmup 0 --format csv > /tmp/p5h-2a-t1-<config>-preheat.log 2>&1
PREHEAT_WALL=$(($(date +%s) - PREHEAT_START))
echo "preheat_wall=${PREHEAT_WALL}s (target=${TARGET}s)"
```

Expected: `preheat_wall >= 0.8 * <config>.preheat_seconds`. If short → log BLOCKED.

- [ ] Sweep with `<config>.RUNS, warmup=1`:

```bash
source /tmp/p5h-2a-env.sh
cd /Users/xin/workspace/ironmlx-backend
cargo run --release -p iron-bench -- \
  --target ironmlx=http://127.0.0.1:18099 \
  --model qwen3.5-moe --model-dir "$SNAP" \
  --prompt-len 512 --max-tokens 1 --runs <config.RUNS> --warmup 1 --format csv > /tmp/p5h-2a-t1-<config>-pp512.csv 2>>/tmp/p5h-2a-t1-<config>-serve.log
echo "sweep_exit=$?"
wc -l /tmp/p5h-2a-t1-<config>-pp512.csv
```

Expected: CSV has `<config.RUNS> + 1` lines.

- [ ] Kill ironmlx + sleep cooldown:

```bash
source /tmp/p5h-2a-env.sh
kill $IRONMLX_PID 2>/dev/null
wait $IRONMLX_PID 2>/dev/null
sleep <config.cooldown_seconds>
```

- [ ] Run bootstrap analysis:

```bash
uv run python tools/p5h_2a_se_analysis.py \
  --input /tmp/p5h-2a-t1-<config>-pp512.csv \
  --out-json /tmp/p5h-2a-t1-<config>-se-analysis.json
cat /tmp/p5h-2a-t1-<config>-se-analysis.json
```

- [ ] Extract within-sweep CI half-width for full N (i.e. the subset_size == `<config.RUNS>`):

```bash
python3 -c "
import json
d = json.load(open('/tmp/p5h-2a-t1-<config>-se-analysis.json'))
ci = d['se_per_subset'][str(<config.RUNS>)]['ci95_half_width_pct']
print(f'<config> within-sweep CI half-width = {ci:.2f}%')
print('PROMOTE_TO_REPEAT' if ci <= 2.5 else 'REJECT_CANDIDATE')
"
```

- [ ] Decision: if within-sweep CI ≤ ±2.5% → promote to repeat validation (Step 2.3 below). If > ±2.5% → reject this candidate; document in bench log; move to next config.

### Step 2.3: Independent repeat validation for promoted candidate(s)

For each candidate promoted from Step 2.2 (≤ ±2.5% within-sweep CI):

- [ ] Run **3 independent spawn+preheat+sweep repeats**. For each repeat N ∈ {1, 2, 3}:

```bash
# Mirror Step 2.2 spawn+preheat+sweep+kill cycle exactly, but save to
# /tmp/p5h-2a-t1-<config>-repeat<N>.csv
# Use independent spawn each time (no cached state shared)
```

Each repeat is a full lifecycle: spawn → wait healthz → preheat → sweep → kill. **Do NOT skip preheat between repeats** (between-spawn thermal state is the entire point of repeat validation).

- [ ] After 3 repeats complete, compute within-sweep CI max + between-sweep median half-range:

```bash
python3 << 'PY'
import json, statistics
medians = []
within_cis = []
for n in [1, 2, 3]:
    # Re-use analysis script to get the per-repeat median
    import subprocess
    out = subprocess.run([
        'uv', 'run', 'python', 'tools/p5h_2a_se_analysis.py',
        '--input', f'/tmp/p5h-2a-t1-<config>-repeat{n}.csv',
        '--out-json', f'/tmp/p5h-2a-t1-<config>-repeat{n}-se.json',
    ], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f'analysis failed for repeat {n}: stdout={out.stdout} stderr={out.stderr}')
    d = json.load(open(f'/tmp/p5h-2a-t1-<config>-repeat{n}-se.json'))
    # Use the full-N subset (matching config.RUNS)
    full = d['se_per_subset'][str(<config.RUNS>)]
    medians.append(full['point_median'])
    within_cis.append(full['ci95_half_width_pct'])
print(f'<config> per-repeat medians: {medians}')

# Between-sweep half-range = range / 2 / mean. This is a spread envelope,
# not a 95% CI.
mean_median = statistics.mean(medians)
range_half = (max(medians) - min(medians)) / 2.0
between_sweep_half_range_pct = range_half / mean_median * 100.0

# Final uncertainty envelope = max(within-sweep bootstrap CI across repeats,
# between-sweep half-range).
within_ci_max = max(within_cis)
final_uncertainty = max(within_ci_max, between_sweep_half_range_pct)

print(f'<config> within-sweep CI max = {within_ci_max:.2f}%')
print(f'<config> between-sweep half-range = {between_sweep_half_range_pct:.2f}%')
print(f'<config> final uncertainty envelope = {final_uncertainty:.2f}%')
print('OUTCOME_A_PASS' if final_uncertainty <= 2.0 else 'OUTCOME_A_FAIL')
PY
```

- [ ] Document candidate's final uncertainty envelope + verdict in `reports/p5h+2-a-bench-log.md` T1 section.

### Step 2.4: T1 selection + close-out

- [ ] Across all candidates that completed repeat validation:
  - If ≥1 hit `final_uncertainty ≤ 2.0%` → **Outcome (a) candidate found**. Pick winner = lowest `(wall_time × final_uncertainty)` per spec § 5.2, then T4 must independently validate before any protocol commit.
  - If 0 hit `final_uncertainty ≤ 2.0%` → **T2 Approach B triggers**.

- [ ] Document selection (or T2 trigger) in `reports/p5h+2-a-bench-log.md`. Include:
  - Per-candidate (within-sweep CI max, between-sweep half-range, final uncertainty envelope, wall_time estimate)
  - Selection: candidate name + protocol params (`RUNS, cooldown, preheat`)
  - Or: "no candidate met ±2% → T2 fallback triggered"

- [ ] T1 has NO source code commit. Bench log is gitignored. T1 closes when documentation is written + selection or T2 trigger recorded.

---

## Task 3: T2 — Phase 1B fallback (CONDITIONAL — only if T1 doesn't hit ±2%)

**Files:**
- Output (gitignored): `/tmp/p5h-2a-t2-powermetrics.log` (if sudo -n succeeds); `reports/p5h+2-a-bench-log.md` (T2 section)
- No commits in T2

### Step 3.1: Skip-check — execute T2 only if T1 outcome was "no candidate met ±2%"

- [ ] If T1 selected a winning candidate → SKIP T2; document "T2 skipped (T1 outcome a)"; jump to Task 4 (T3).

- [ ] Else continue.

### Step 3.2: Per-run TTFT drift analysis (from T0 data; no GPU needed)

- [ ] Read T0 drift diagnostics:

```bash
python3 -c "
import json
d = json.load(open('/tmp/p5h-2a-t0-se-analysis.json'))
pp_tps_drift = d['drift_diagnostics']['pp_tps_vs_run_idx']
ttft_drift = d['drift_diagnostics']['ttft_ms_vs_run_idx']
span = max(d['input_n_runs'] - 1, 1)
pp_tps_total_drift = abs(pp_tps_drift['slope']) * span
ttft_total_drift_ms = abs(ttft_drift['slope']) * span
print(f'pp_tps_vs_run_idx: slope={pp_tps_drift[\"slope\"]:.4f}, '
      f'r_squared={pp_tps_drift[\"r_squared\"]:.4f}, '
      f'p={pp_tps_drift[\"p_value\"]:.4g}, total_drift={pp_tps_total_drift:.2f} pp_tps')
print(f'ttft_ms_vs_run_idx: slope={ttft_drift[\"slope\"]:.4f}, '
      f'r_squared={ttft_drift[\"r_squared\"]:.4f}, '
      f'p={ttft_drift[\"p_value\"]:.4g}, total_drift={ttft_total_drift_ms:.2f} ms')
# Heuristic: p < 0.05 AND r_squared > 0.1 AND total drift crosses threshold
"
```

- [ ] Document interpretation in `reports/p5h+2-a-bench-log.md` T2 section:
  - If `p_value < 0.05` AND `r_squared > 0.1` AND total drift across the sweep is `≥10ms` for `ttft_ms` OR `≥1 pp_tps` for `pp_tps` → drift confirmed; thermal/scheduler state likely a variance source

### Step 3.3: powermetrics correlation (non-blocking)

- [ ] Try non-interactive sudo:

```bash
sudo -n true 2>/dev/null && echo "SUDO_NONINTERACTIVE_AVAILABLE" || echo "SUDO_NOT_AVAILABLE"
```

- [ ] If SUDO_NOT_AVAILABLE → document "powermetrics unavailable in this execution context" in bench log; skip to Step 3.4.

- [ ] If SUDO_NONINTERACTIVE_AVAILABLE → spawn powermetrics concurrent with a fresh RUNS=30 sweep:

```bash
sudo -n powermetrics --sample-interval 1000 --samplers gpu_power,thermal -i 1000 > /tmp/p5h-2a-t2-powermetrics.log 2>&1 &
PMETRICS_PID=$!

# Concurrent: spawn ironmlx + preheat + RUNS=30 sweep + kill (mirror Step 1.3-1.6)
# ...

kill $PMETRICS_PID 2>/dev/null
wait $PMETRICS_PID 2>/dev/null
```

- [ ] Correlate thermal trace with per-run pp_tps:
  - Extract GPU temperature timeline from `/tmp/p5h-2a-t2-powermetrics.log`
  - Map run start times → pp_tps; check if thermal saturates partway through sweep + pp_tps drops in correlation
  - Document findings in bench log

### Step 3.4: Request-state determinism control

Per spec § 6.3 — MoE router top-k is deterministic for fixed weights + fixed input. Do NOT chase a "routing seed" knob. Instead verify/pin protocol state:

- [ ] Verify across all T1 sweeps (read each `/tmp/p5h-2a-t1-<config>-pp512.csv`):
  - Identical `pp_target` column values (should all be 512)
  - Identical `target` column (per-config; ironmlx target name)
  - Identical iron-bench invocation args (re-check command history)
  - No concurrent client during any sweep (already enforced by `[feedback_serial_perf_experiments]`)
  - Fresh server spawn before each sweep (Step 2.2 enforces this)

- [ ] Check MLX environment variables (existing baseline; do not modify production):

```bash
env | grep -iE 'MLX|METAL|GPU' | sort
```

Document the environment in bench log.

- [ ] If any suspected state variable differs across sweeps, re-bench with that variable pinned and compare CI.

### Step 3.5: T2 verdict

- [ ] Write T2 verdict in `reports/p5h+2-a-bench-log.md`:
  - Drift analysis: confirmed / not confirmed / inconclusive
  - powermetrics correlation: confirmed / not available / inconclusive
  - Request-state determinism: all variables fixed / found difference (named)
  - **Identified root cause** OR **quantified ambiguity** ("multiple sources contribute; e.g. estimated <X>% from drift + <Y>% unexplained")
  - **Recommended T3 path**: A (still try a config based on root cause; may need to revisit T1 with different params) OR C (redefine band per Path C)

### Step 3.6: T2 close-out

- [ ] T2 has NO source code commit. Bench log is gitignored. T2 closes when documentation is written + path A/C decision recorded.

---

## Task 4: T3 — Spec new PP=512 protocol (Path A) OR band redefinition (Path C)

**Files:**
- Create: `docs/p5h+2-a-pp512-protocol.md` (drafted in T3; committed in T4 only after validation)
- Modify: `tools/p5i_a_baseline_aggregate.py` (per-PP RUNS expectation enforced for each backend input + per-PP 95% CI emission)
- Create: `tools/p5h_aggregator/tests/test_p5i_a_baseline_aggregate_ci.py` (pytest for new CI emission)
- Modify (Path C only): `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` § 7.2.2 (drafted in T3; committed in T4 after validation)

### Step 4.1: Decide Path A or Path C based on T1/T2 verdict

- [ ] Read T1+T2 verdict from `reports/p5h+2-a-bench-log.md`.
  - **Path A**: T1 selected a winning candidate (Outcome a) → spec the protocol
  - **Path C**: T2 verdict requires band redefinition → amend spec § 7.2 + emit CI

### Step 4.2: Write `docs/p5h+2-a-pp512-protocol.md` (Path A)

- [ ] If Path A, create the doc with these sections:

```markdown
# P5h+2.a PP=512 Protocol (Path A — Outcome a achieved)

**Status:** Validated protocol; supersedes the implicit RUNS=7 protocol for PP=512 after T4 validation commit.
**Date:** YYYY-MM-DD (fill from `date +%Y-%m-%d`)
**Branch:** ironmlx-p5h+2-a-pp512-measurement
**Spec ref:** docs/superpowers/specs/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol-design.md § 3.1 Outcome (a)

## Selected protocol

- **RUNS**: <FILL from selected candidate>
- **cooldown**: <FILL seconds between PPs>
- **preheat**: <FILL seconds before first measured run>
- **independent repeat count**: 3 (minimum for between-sweep half-range validation)

## Empirical SE achieved

| Sweep | within-sweep CI half-width | between-sweep half-range | final uncertainty envelope (max) |
|---|---|---|---|
| candidate <FILL name> | <FILL>% | <FILL>% (from 3 repeats) | <FILL>% (≤ 2.0%) |

## Comparison scope

This protocol is validated for: <choose one>
- (i) **ironmlx-only pre/post regression decisions** — only ironmlx repeats collected
- (ii) **ironmlx-vs-omlx external target decisions** — both ironmlx AND omlx PP=512 sweeps use this protocol; aggregator emits CI for both backends

If (i): downstream decisions claiming ironmlx-vs-omlx +X% target are NOT supported by this protocol. T4 may extend with omlx repeats to upgrade to (ii); document if so.

## Rationale for selection

<FILL: chosen for lowest wall_time × final uncertainty envelope; cite specific candidates considered + rejected and reasons>

## Reproducibility

```bash
# Spawn ironmlx serve, wait healthz, then:
cargo run --release -p iron-bench -- \
  --target ironmlx=http://127.0.0.1:18099 \
  --model qwen3.5-moe --model-dir "$SNAP" \
  --prompt-len 512 --max-tokens 1 \
  --runs <FILL RUNS> --warmup 1 --format csv > <output>.csv

# For independent repeat: kill + sleep <FILL cooldown>s + respawn + preheat <FILL> seconds + sweep
# Aggregate with /tmp/p5i-a-baseline-aggregate.py extended for per-PP CI (see T3 aggregator changes)
```

## References

- Spec § 4.1-4.6 for the task-by-task derivation
- reports/p5h+2-a-bench-log.md T0/T1 sections for raw data per candidate
- tools/p5h_2a_se_analysis.py for bootstrap-resample methodology
```

Fill `<FILL>` markers from T1 results.

### Step 4.3: Write `docs/p5h+2-a-pp512-protocol.md` (Path C)

- [ ] If Path C, create the doc with these sections (different from Path A):

```markdown
# P5h+2.a PP=512 Protocol (Path C — band redefined)

**Status:** Validated band redefinition; spec § 7.2 noise band for PP=512 amended per § 7.2.2 after T4 validation commit.
**Date:** YYYY-MM-DD
**Branch:** ironmlx-p5h+2-a-pp512-measurement
**Spec ref:** docs/superpowers/specs/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol-design.md § 3.2 Outcome (b)

## Empirical max-achievable SE at PP=512

<FILL: best final uncertainty envelope achieved in T1 (within-sweep CI + between-sweep half-range)>%

Per T1 candidates evaluated:
| candidate | RUNS | cooldown | preheat | final uncertainty envelope |
|---|---|---|---|---|
| <FILL all candidates + verdicts> |

## Root cause or quantified ambiguity (T2 verdict)

<FILL from T2 bench log>:
- Drift analysis: <pp_tps slope + r_squared / ttft_ms slope + r_squared>
- powermetrics correlation: <thermal trace finding OR "unavailable">
- Request-state determinism: <all pinned / variable identified>
- Identified root cause OR ambiguity: <FILL>

## Why ±2% is unachievable in scope

<FILL: specific quantitative reason; e.g. "thermal drift contributes >3% per-sweep variance even at preheat=600s; closing requires hardware cooldown beyond iron-bench scope">

## Recommended new spec § 7.2 noise band for PP=512

±<FILL X>% (e.g. ±5% or ±10% based on empirical achievable level rounded up)

## Reproducibility (use this protocol for any future PP=512 sweep)

<same as Path A reproducibility section but with the chosen RUNS even if it doesn't achieve ±2%>

## References (same as Path A)
```

### Step 4.4: Modify `tools/p5i_a_baseline_aggregate.py` (per-PP RUNS + per-PP CI emission)

- [ ] Read current aggregator:

```bash
cat /Users/xin/workspace/ironmlx-backend/tools/p5i_a_baseline_aggregate.py
```

- [ ] Update `EXPECTED_RUNS_PER_PP` from a single constant to a per-PP dict, and update the run_idx set validation to derive from the same per-PP value:

```python
# OLD:
# EXPECTED_RUNS_PER_PP = 7
# EXPECTED_RUN_IDX_SET = frozenset(range(EXPECTED_RUNS_PER_PP))
# NEW:
EXPECTED_RUNS_PER_PP: dict[int, int] = {
    128: 7,
    512: <FILL chosen RUNS from T1 if Path A, else keep 7>,
}


def expected_runs_for_pp(pp: int) -> int:
    try:
        return EXPECTED_RUNS_PER_PP[pp]
    except KeyError as exc:
        raise SystemExit(f"unexpected pp_target={pp}; expected {EXPECTED_PPS}") from exc


def expected_run_idx_set_for_pp(pp: int) -> frozenset[int]:
    return frozenset(range(expected_runs_for_pp(pp)))
```

Update `load_pp_tps_by_pp` to look up per-PP RUNS and per-PP run_idx set:

```python
# OLD:
# for pp in EXPECTED_PPS:
#     got = len(by_pp.get(pp, []))
#     if got != EXPECTED_RUNS_PER_PP:
#         raise SystemExit(...)
# NEW:
for pp in EXPECTED_PPS:
    got = len(by_pp.get(pp, []))
    expected = expected_runs_for_pp(pp)
    if got != expected:
        raise SystemExit(
            f"{csv_path}: expected {expected} measured rows for PP={pp}, got {got}"
        )
    run_idxs = [r for r, _ in by_pp_runs.get(pp, [])]
    observed_set = set(run_idxs)
    expected_set = expected_run_idx_set_for_pp(pp)
    if len(observed_set) != len(run_idxs):
        dupes = sorted(idx for idx, count in Counter(run_idxs).items() if count > 1)
        raise SystemExit(
            f"{csv_path}: PP={pp} has duplicate run_idx values: {dupes}; observed={sorted(run_idxs)}"
        )
    if observed_set != expected_set:
        raise SystemExit(
            f"{csv_path}: PP={pp} run_idx set was {sorted(observed_set)} "
            f"but expected {sorted(expected_set)}"
        )
```

External ironmlx-vs-omlx comparisons require both backend CSVs to use the same PP=512 protocol. If T4 collects only ironmlx repeats, do not run or publish PP=512 external-target aggregator conclusions until matching omlx repeats are available.

- [ ] Import bootstrap CI computation from `tools.p5h_2a_se_analysis`:

```python
# At top of file:
import random

from p5h_2a_se_analysis import bootstrap_median_ci
```

- [ ] In `main()`, after computing per-PP medians, add CI emission:

```python
# After existing per-PP median computation, before write_text:
rng = random.Random(42)  # fixed seed for reproducibility
for pp in EXPECTED_PPS:
    i_tps = ironmlx[pp]
    o_tps = omlx[pp]
    n_i = len(i_tps)
    n_o = len(o_tps)
    i_ci = bootstrap_median_ci(i_tps, subset_size=n_i, iterations=1000, rng=rng)
    o_ci = bootstrap_median_ci(o_tps, subset_size=n_o, iterations=1000, rng=rng)
    summary["per_pp"][str(pp)].update({
        "ironmlx_pp_tps_ci95_low": i_ci["ci95_low"],
        "ironmlx_pp_tps_ci95_high": i_ci["ci95_high"],
        "ironmlx_pp_tps_ci95_half_width_pct": i_ci["ci95_half_width_pct"],
        "omlx_pp_tps_ci95_low": o_ci["ci95_low"],
        "omlx_pp_tps_ci95_high": o_ci["ci95_high"],
        "omlx_pp_tps_ci95_half_width_pct": o_ci["ci95_half_width_pct"],
    })
```

- [ ] Update stdout print to include CI:

```python
# OLD:
# print(f"  PP={pp}: ironmlx={i:.2f} omlx={o:.2f} delta={d:+.2f}% {flag}")
# NEW:
print(f"  PP={pp}: ironmlx={i:.2f} (±{row['ironmlx_pp_tps_ci95_half_width_pct']:.2f}%) omlx={o:.2f} (±{row['omlx_pp_tps_ci95_half_width_pct']:.2f}%) delta={d:+.2f}% {flag}")
```

### Step 4.5: Create pytest for aggregator CI emission

- [ ] Create `tools/p5h_aggregator/tests/test_p5i_a_baseline_aggregate_ci.py`:

```python
"""Pytest for tools/p5i_a_baseline_aggregate.py per-PP CI emission (P5h+2.a T3)."""

from __future__ import annotations
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from p5i_a_baseline_aggregate import EXPECTED_RUNS_PER_PP  # noqa: E402


def write_iron_bench_csv(rows: list[dict], path: Path) -> None:
    """Write iron-bench-format CSV (matches T0 protocol)."""
    fieldnames = [
        "target",
        "pp_target",
        "tg_target",
        "run_idx",
        "ttft_ms",
        "tg_tps",
        "tpot_ms",
        "pp_tps",
        "e2e_s",
        "prompt_tokens_local",
        "prompt_tokens_server",
        "completion_tokens_server",
        "cached_tokens",
        "finish_reason",
    ]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            full_row = {fn: row.get(fn, "") for fn in fieldnames}
            w.writerow(full_row)


def gen_synthetic_rows(
    target: str, pp: int, runs: int, base_pp_tps: float, spread: float = 0.05
) -> list[dict]:
    """Generate synthetic iron-bench rows with controlled pp_tps spread."""
    rows = []
    for i in range(runs):
        pp_tps = base_pp_tps * (1 + spread * ((i - runs // 2) / (runs // 2)))
        rows.append(
            {
                "target": target,
                "pp_target": str(pp),
                "tg_target": "1",
                "run_idx": str(i),
                "ttft_ms": "5000.0",
                "tg_tps": "100.0",
                "tpot_ms": "10.0",
                "pp_tps": f"{pp_tps:.4f}",
                "e2e_s": "5.0",
                "prompt_tokens_local": str(pp),
                "prompt_tokens_server": "",
                "completion_tokens_server": "",
                "cached_tokens": "",
                "finish_reason": "length",
            }
        )
    return rows


def test_aggregator_emits_per_pp_ci_fields():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ironmlx_csv = td_path / "ironmlx.csv"
        omlx_csv = td_path / "omlx.csv"
        out_json = td_path / "summary.json"

        # PP=128: 7 runs (per EXPECTED_RUNS_PER_PP[128])
        rows128_iron = gen_synthetic_rows("ironmlx", 128, 7, 1000.0)
        rows128_omlx = gen_synthetic_rows("omlx", 128, 7, 1100.0)
        # PP=512: fixture follows the configured protocol RUNS.
        rows512_iron = gen_synthetic_rows(
            "ironmlx", 512, EXPECTED_RUNS_PER_PP[512], 1500.0
        )
        rows512_omlx = gen_synthetic_rows(
            "omlx", 512, EXPECTED_RUNS_PER_PP[512], 2000.0
        )

        write_iron_bench_csv(rows128_iron + rows512_iron, ironmlx_csv)
        write_iron_bench_csv(rows128_omlx + rows512_omlx, omlx_csv)

        # Run aggregator
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS_DIR / "p5i_a_baseline_aggregate.py"),
                "--ironmlx-csv",
                str(ironmlx_csv),
                "--omlx-csv",
                str(omlx_csv),
                "--out-json",
                str(out_json),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"aggregator failed: {result.stderr}"

        # Verify CI fields present
        summary = json.loads(out_json.read_text())
        for pp in ("128", "512"):
            row = summary["per_pp"][pp]
            assert "ironmlx_pp_tps_ci95_low" in row
            assert "ironmlx_pp_tps_ci95_high" in row
            assert "ironmlx_pp_tps_ci95_half_width_pct" in row
            assert "omlx_pp_tps_ci95_low" in row
            assert "omlx_pp_tps_ci95_high" in row
            assert "omlx_pp_tps_ci95_half_width_pct" in row
            # Sanity: CI low <= median <= high
            assert row["ironmlx_pp_tps_ci95_low"] <= row["ironmlx_pp_tps_median"]
            assert row["ironmlx_pp_tps_median"] <= row["ironmlx_pp_tps_ci95_high"]
```

### Step 4.6: Modify spec § 7.2 (Path C only)

- [ ] If Path C, append `§ 7.2.2` to `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md`. Find § 7.2 section + add subsection:

```markdown
### 7.2.2 PP=512 noise band amendment (post-P5h+2.a)

Per P5h+2.a validation (`docs/p5h+2-a-pp512-protocol.md`, commit <FILL T4 SHA>), the ±2% noise band of § 7.2 is amended for PP=512 to ±<FILL X>% based on empirical uncertainty of the canonical RUNS-<FILL N> protocol on M5 Max. Root cause or bounded ambiguity: <FILL thermal drift / request-state drift / allocator non-determinism / scheduler state / multiple sources>. P5h+2.a Approach B documented the path forward — see `docs/p5h+2-a-pp512-protocol.md` for full details + reproducibility.
```

Skip if Path A.

### Step 4.7: Hygiene before validation (no commit yet)

- [ ] Hygiene:

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check tools/p5i_a_baseline_aggregate.py tools/p5h_aggregator/tests/test_p5i_a_baseline_aggregate_ci.py
uv run --with ruff ruff format --check tools/p5i_a_baseline_aggregate.py tools/p5h_aggregator/tests/test_p5i_a_baseline_aggregate_ci.py
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/ -v
```

- [ ] Conditional aggregator regression check:

If `EXPECTED_RUNS_PER_PP[512] == 7`, the old P5i.a canonical baseline remains valid input; revalidate medians + delta fields:

```bash
uv run python tools/p5i_a_baseline_aggregate.py \
  --ironmlx-csv /tmp/p5i-a-baseline-ironmlx.csv \
  --omlx-csv /tmp/p5i-a-baseline-omlx.csv \
  --out-json /tmp/p5i-a-baseline-revalidated.json
# Compare: medians unchanged; new CI fields present
diff <(jq 'del(.per_pp."128".ironmlx_pp_tps_ci95_low, .per_pp."128".ironmlx_pp_tps_ci95_high, .per_pp."128".ironmlx_pp_tps_ci95_half_width_pct, .per_pp."128".omlx_pp_tps_ci95_low, .per_pp."128".omlx_pp_tps_ci95_high, .per_pp."128".omlx_pp_tps_ci95_half_width_pct, .per_pp."512".ironmlx_pp_tps_ci95_low, .per_pp."512".ironmlx_pp_tps_ci95_high, .per_pp."512".ironmlx_pp_tps_ci95_half_width_pct, .per_pp."512".omlx_pp_tps_ci95_low, .per_pp."512".omlx_pp_tps_ci95_high, .per_pp."512".omlx_pp_tps_ci95_half_width_pct)' /tmp/p5i-a-baseline-summary.json) <(jq 'del(.per_pp."128".ironmlx_pp_tps_ci95_low, .per_pp."128".ironmlx_pp_tps_ci95_high, .per_pp."128".ironmlx_pp_tps_ci95_half_width_pct, .per_pp."128".omlx_pp_tps_ci95_low, .per_pp."128".omlx_pp_tps_ci95_high, .per_pp."128".omlx_pp_tps_ci95_half_width_pct, .per_pp."512".ironmlx_pp_tps_ci95_low, .per_pp."512".ironmlx_pp_tps_ci95_high, .per_pp."512".ironmlx_pp_tps_ci95_half_width_pct, .per_pp."512".omlx_pp_tps_ci95_low, .per_pp."512".omlx_pp_tps_ci95_high, .per_pp."512".omlx_pp_tps_ci95_half_width_pct)' /tmp/p5i-a-baseline-revalidated.json)
# Expected: empty diff (medians + delta_pct + passes_plus10_target identical)
```

If `EXPECTED_RUNS_PER_PP[512] != 7`, the old P5i.a canonical baseline is intentionally invalid for the new PP=512 protocol. Do not use it as a positive regression input. Instead:
- keep the synthetic pytest in Step 4.5 as the positive aggregator validation for configured per-PP RUNS;
- optionally run the old baseline once and record that it fails with the expected PP=512 row-count error;
- do not publish PP=512 external-target conclusions until matching ironmlx and omlx CSVs exist under the same selected protocol.

- [ ] Do **not** commit in T3. T4 must validate the drafted protocol/band first; the commit happens in Step 5.5 after validation passes or after Path C fallback is drafted and validated.

---

## Task 5: T4 — Validate new protocol (≥3 independent repeats)

**Files:**
- Output (gitignored): `/tmp/p5h-2a-t4-validate-repeat<N>.csv` (N=1..3); `/tmp/p5h-2a-t4-validate-summary.json`; `reports/p5h+2-a-bench-log.md` (T4 section)
- Commit validated T3 docs/aggregator changes in Step 5.5 after acceptance passes

### Step 5.1: Run ≥3 independent ironmlx spawn+sweep repeats under selected protocol

For each repeat N ∈ {1, 2, 3}:

- [ ] Cleanup + spawn + healthz + preheat + sweep + kill (mirror Step 1.3-1.6, but use the SELECTED protocol params from T3):

```bash
# Use exact RUNS / cooldown / preheat values from docs/p5h+2-a-pp512-protocol.md
# Save sweep to /tmp/p5h-2a-t4-validate-repeat<N>.csv
```

Each repeat is a FRESH spawn (independent process; no shared state).

### Step 5.2: Compute final uncertainty envelope for validation

- [ ] Run analysis script per repeat + compute between-sweep half-range:

```bash
for n in 1 2 3; do
  uv run python tools/p5h_2a_se_analysis.py \
    --input /tmp/p5h-2a-t4-validate-repeat${n}.csv \
    --out-json /tmp/p5h-2a-t4-validate-repeat${n}-se.json
done

python3 << 'PY'
import json, statistics
RUNS = <FILL chosen RUNS>
medians = []
within_ci_max = 0.0
for n in [1, 2, 3]:
    d = json.load(open(f'/tmp/p5h-2a-t4-validate-repeat{n}-se.json'))
    full = d['se_per_subset'][str(RUNS)]
    medians.append(full['point_median'])
    within_ci_max = max(within_ci_max, full['ci95_half_width_pct'])

mean_median = statistics.mean(medians)
between_half_range = (max(medians) - min(medians)) / 2.0 / mean_median * 100.0
final_uncertainty = max(within_ci_max, between_half_range)

print(f'within-sweep CI (max across repeats): {within_ci_max:.2f}%')
print(f'between-sweep half-range (from {len(medians)} repeats): {between_half_range:.2f}%')
print(f'final uncertainty envelope (max): {final_uncertainty:.2f}%')
# Save validation summary
json.dump({
    'repeats': len(medians),
    'medians': medians,
    'within_sweep_ci95_pct': within_ci_max,
    'between_sweep_half_range_pct': between_half_range,
    'final_uncertainty_pct': final_uncertainty,
    'protocol_target_pct': 2.0,
    'outcome': 'VALIDATED' if final_uncertainty <= 2.0 else 'BELOW_TARGET',
}, open('/tmp/p5h-2a-t4-validate-summary.json', 'w'), indent=2)
print('wrote /tmp/p5h-2a-t4-validate-summary.json')
PY
```

### Step 5.3: omlx repeats (CONDITIONAL — only if comparison scope is ironmlx-vs-omlx external target)

- [ ] If `docs/p5h+2-a-pp512-protocol.md` § Comparison Scope = (ii) → also run ≥3 omlx spawn+sweep repeats:

```bash
# Same protocol but spawn omlx instead of ironmlx; same preheat + RUNS
# Save to /tmp/p5h-2a-t4-validate-omlx-repeat<N>.csv
```

- [ ] Compute omlx final uncertainty envelope (same script as Step 5.2).

- [ ] If (i) only → omlx repeats SKIPPED; document explicitly in close-out that external-target decisions remain blocked until omlx repeats are collected.

### Step 5.4: T4 acceptance + record

- [ ] Verify:
  - For Path A: ironmlx `final_uncertainty_pct ≤ 2.0%`; if comparison scope is (ii), omlx final uncertainty must also be `≤ 2.0%` → T4 acceptance PASS
  - For Path A failure: do **not** commit the Path A protocol. If T2 was skipped, run Task 3 first to document root cause or quantified ambiguity. Then convert to Path C by returning to Step 4.3/4.6, set the PP=512 band from the validated empirical uncertainty envelope, rerun Step 4.7 hygiene, and repeat T4 acceptance against the Path C band.
  - For Path C: `final_uncertainty_pct` fits the T3-stated band (e.g. if T3 said ±5%, T4 confirms within ±5%) → T4 acceptance PASS

- [ ] Document validation in `reports/p5h+2-a-bench-log.md` T4 section:
  - Per-repeat medians
  - within-sweep CI / between-sweep half-range / final uncertainty envelope
  - Whether omlx repeats were collected
  - Final comparison scope (i or ii)

### Step 5.5: Commit validated protocol/band + aggregator changes

- [ ] Commit only after Step 5.4 acceptance passes:

```bash
cd /Users/xin/workspace/ironmlx-backend
git add docs/p5h+2-a-pp512-protocol.md tools/p5i_a_baseline_aggregate.py tools/p5h_aggregator/tests/test_p5i_a_baseline_aggregate_ci.py
# If Path C: also add docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md

git commit -m "$(cat <<COMMIT
docs(p5h+2-a-t4): validated PP=512 protocol <Path A / Path C> + aggregator CI emission

Per docs/superpowers/specs/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol-design.md § 4.4-4.5
and docs/superpowers/plans/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol.md T3-T4.

Path: <A: protocol validated / C: band redefined and validated>

New docs/p5h+2-a-pp512-protocol.md:
* Selected (RUNS, cooldown, preheat) values <FILL>
* T4 final uncertainty envelope achieved <FILL>
* Comparison scope: <(i) ironmlx-only pre/post / (ii) ironmlx-vs-omlx>
* Reproducibility command + per-repeat protocol

Aggregator tools/p5i_a_baseline_aggregate.py:
* EXPECTED_RUNS_PER_PP from single constant to per-PP dict (PP=128 still 7;
  PP=512 set to <FILL chosen RUNS>)
* PP-specific run_idx validation derived from EXPECTED_RUNS_PER_PP
* Per-PP 95% CI emission (ironmlx + omlx) via bootstrap from
  tools/p5h_2a_se_analysis.py (imported)
* New pytest tools/p5h_aggregator/tests/test_p5i_a_baseline_aggregate_ci.py
  covers configured RUNS, CI field emission, and median bounds sanity

<IF Path C: also amends docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md
§ 7.2.2 with PP=512 band amendment + root cause / quantified ambiguity>

Validation:
* T4 ironmlx final uncertainty envelope: <FILL>%
* T4 omlx final uncertainty envelope: <FILL or skipped; if skipped, external target claims remain blocked>
* Step 4.7 hygiene and pytest passed before validation commit

Downstream consumers (P5i.c PP=512 land conditions; future P5h+2.b ROI gate)
should use CI half-width for noise-bound comparisons. ironmlx-only repeats
only unblock ironmlx pre/post regression decisions; external omlx-target
claims require both backends measured under this protocol.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
COMMIT
)"
git log --oneline -3
```

Validation data + bench log remain gitignored.

---

## Task 6: T5 — Close-out

**Files:**
- Create: `docs/p5h+2-a-close-out.md` (committed)
- Modify (outside repo): `~/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_findings.md` (extend with P5h+2.a closure section)

### Step 6.1: Determine close-out status

Per spec § 3.3 vocabulary:
- **Full PASS**: Outcome (a) achieved (Path A); T4 validated final uncertainty envelope ≤ ±2%
- **Feasibility PASS**: Outcome (b) achieved (Path C); band redefined + root cause / quantified ambiguity documented + aggregator emits CI
- **Blocked**: A + B + C all failed (should not happen given fallback exists)

### Step 6.2: Write `docs/p5h+2-a-close-out.md`

- [ ] Compose with these sections:
  - Status (Full PASS / Feasibility PASS / Blocked) + date + branch + commit chain
  - § 1 Close Gate result (cite per-RUNS CI measurements; per-repeat half-range; comparison scope)
  - § 2 What landed (T3 protocol/band doc; aggregator per-PP CI emission; spec § 7.2.2 amendment if Path C)
  - § 3 P5i.c PP=512 arm unblocked status — explicitly state:
    - Whether new protocol supports ironmlx-only pre/post / ironmlx-vs-omlx target / both
    - Required iron-bench invocation (RUNS / cooldown / preheat / repeat count)
  - § 4 P5h+3 follow-up items:
    - Iron-bench tool enhancement for trimmed mean / IQR / per-iter throughput (Approach C from brainstorming, deferred)
    - If Path C: longer-term path to true ±2% (hardware-level thermal control / different metric / different protocol)
  - § 5 Memory update — link to `[project_p5h_findings]` extension
  - § 6 References

### Step 6.3: Update memory `project_p5h_findings.md`

- [ ] Edit memory file. Append new section:

```markdown
## P5h+2.a closure update (YYYY-MM-DD)

P5h+2.a closed as **<Full PASS / Feasibility PASS / Blocked>**. Branch
`ironmlx-p5h+2-a-pp512-measurement` (forked from
ironmlx-p5i-a-gather-qmm-feasibility a90a85c).

PP=512 measurement protocol:
- RUNS: <FILL>
- cooldown: <FILL>s
- preheat: <FILL>s
- independent repeat count: 3 (minimum for between-sweep half-range)
- final uncertainty envelope achieved: <FILL>% (target ±2%)
- comparison scope: <(i) ironmlx-only / (ii) ironmlx-vs-omlx>

If Path C: spec § 7.2 noise band for PP=512 amended to ±<FILL>% per spec
§ 7.2.2 + docs/p5h+2-a-pp512-protocol.md. Root cause: <FILL>.

Aggregator tools/p5i_a_baseline_aggregate.py extended:
- EXPECTED_RUNS_PER_PP per-PP dict (PP=128: 7; PP=512: <FILL>)
- Per-PP 95% CI emission for ironmlx + omlx

Downstream P5i.c PP=512 arm now <unblocked / partially unblocked (ironmlx pre/post only)>.
P5h+3 follow-up: iron-bench tool enhancement for trimmed mean / IQR (deferred).
```

Update frontmatter `description` if Full PASS achieves ±2%.

### Step 6.4: Final hygiene + commit T5

- [ ] Hygiene:

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check tools/p5h_2a_se_analysis.py tools/p5i_a_baseline_aggregate.py
uv run --with ruff ruff format --check tools/p5h_2a_se_analysis.py tools/p5i_a_baseline_aggregate.py
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/ -v
```

All clean.

- [ ] Commit T5:

```bash
git add docs/p5h+2-a-close-out.md
git commit -m "$(cat <<'COMMIT'
docs(p5h+2-a-t5): close-out — <Full PASS / Feasibility PASS / Blocked>

Per docs/superpowers/specs/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol-design.md § 4.6
and docs/superpowers/plans/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol.md T5.

P5h+2.a closes as <status>:
* PP=512 measurement protocol established at RUNS=<FILL>, cooldown=<FILL>s,
  preheat=<FILL>s, ≥3 independent repeat validation
* Final uncertainty envelope achieved: <FILL>% (target ±2%)
* Comparison scope: <(i) ironmlx-only pre/post / (ii) ironmlx-vs-omlx>

<IF Full PASS:>
* Spec § 7.2 noise band ±2% retained; new RUNS/protocol committed to
  docs/p5h+2-a-pp512-protocol.md
* Aggregator per-PP CI emission unchanged from T3

<IF Feasibility PASS:>
* Spec § 7.2 noise band for PP=512 amended to ±<FILL>% per § 7.2.2
* Root cause / quantified ambiguity: <FILL>
* P5h+3 follow-up: <FILL>

Downstream:
* P5i.c PP=512 arm <unblocked for selected comparison mode>
* future P5h+2.b ROI gate inherits CI envelope from aggregator output

Memory project_p5h_findings.md extended with P5h+2.a closure section.

Commit chain: <T0 SHA> -> <T3 SHA> -> this commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
COMMIT
)"
git log --oneline -5
```

---

## Self-Review

### Spec coverage (against spec § 1 / § 3 / § 4 / § 5 / § 6 / § 7 / § 8 / § 9 / § 10)

- Spec § 1 Goal (4 sub-points) — Plan T0+T1 cover characterization + iteration; T1 has between-sweep validation; T2 has root cause/ambiguity investigation; T3 Path A or C delivers spec/band update.
- Spec § 3 Close Gate (Outcome a + Outcome b + status vocab) — Plan T4 + T5 implement final uncertainty-envelope validation + status classification.
- Spec § 4 Tasks T0-T5 — Plan tasks 1-6 mirror 1:1.
- Spec § 5 Approach A details (bootstrap methodology + selection rule) — Plan T0 Step 1.7 implements bootstrap + T1 Step 2.4 implements selection rule.
- Spec § 6 Approach B fallback details — Plan T2 Steps 3.2-3.4 implement TTFT drift / powermetrics non-blocking / request-state determinism.
- Spec § 7 Approach C fallback (band redefinition + aggregator CI) — Plan T3 Steps 4.3 (Path C doc), 4.4 (aggregator CI emission), 4.6 (spec § 7.2.2) implement.
- Spec § 8 Out of scope — Plan does not include deferred items (other PPs / iron-bench tool / P5i.c / P5h+2.b/c / T1-only / spec § 1.2 / P5i.a T2 revert).
- Spec § 9 Validation gates — Plan tasks enforce no-production-src / statistical rigor / reproducibility / production parity / serial GPU / python hygiene.
- Spec § 10 Branch + sequencing — Plan T0 Step 1.1 creates new branch from a90a85c.

### Placeholder scan

- `<FILL>` markers in commit messages (Step 5.5, 6.4), close-out doc template (T5), protocol doc templates (T3), bench log T0 (Step 1.9) — all are runtime substitutions for measurement data filled by implementer at execution. Not "TBD / implement later" markers.
- `<config>` in T1 Step 2.2 — placeholder for per-iteration candidate name (A/B/C); implementer substitutes per loop iteration.
- `<config.RUNS>` etc. in T1 Step 2.2 — placeholder for per-candidate parameter; implementer substitutes per loop iteration.
- `<800 for 300s target, 1200 for 600s target>` in T1 Step 2.2 — explicit guidance for implementer to compute per-candidate preheat runs count.
- `<line>` / similar reading-output substitutions in T4 Step 5.2 — runtime data.

All placeholders are runtime substitutions tied to per-task experimental data, not "TBD".

### Type consistency

- `EXPECTED_RUNS_PER_PP` semantics: per-PP dict (PP=128: 7; PP=512: chosen RUNS per T1) — consistent in spec § 4.4 + plan T3 Step 4.4 + pytest fixture in Step 4.5.
- JSON schema for `tools/p5h_2a_se_analysis.py` output: `{se_per_subset: {N: {point_median, ci95_low, ci95_high, ci95_half_width_pct, bootstrap_iterations}}, drift_diagnostics: {pp_tps_vs_run_idx: {slope, intercept, r_squared, p_value, p_value_method, n}, ttft_ms_vs_run_idx: ...}, input_csv, input_n_runs, bootstrap_seed}` — consistent in T0 Step 1.7 (creation), T1 Step 2.2 (consumption), T2 Step 3.2 (consumption), T4 Step 5.2 (consumption).
- final uncertainty definition (max of within-sweep bootstrap CI and between-sweep median half-range) — consistent in spec § 5.1 + § 5.2 + plan T1 Step 2.3 + T4 Step 5.2.
- comparison-scope vocab `(i) ironmlx-only pre/post` vs `(ii) ironmlx-vs-omlx external target` — consistent in spec § 3.1 + § 4.4 + § 7.3 + plan T3 Step 4.2 + T4 Step 5.3 + T5 Step 6.2/6.3.

No issues to fix inline.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-23-ironmlx-p5h+2-a-pp512-measurement-protocol.md`. Two execution options:

1. **Subagent-Driven (recommended)** — Fresh subagent per task + two-stage review (spec compliance + code quality) after each. Established pattern from P5h+1 T1-T5 and P5i.a T0-T5.

2. **Inline Execution** — Execute tasks in this session with executing-plans skill; batch execution with checkpoints.

Which approach?
