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
