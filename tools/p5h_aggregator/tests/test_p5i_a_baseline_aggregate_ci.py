"""Pytest for tools/p5i_a_baseline_aggregate.py per-PP CI emission (P5h+2.a T3)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile

import pytest
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
        if runs > 1:
            pp_tps = base_pp_tps * (1 + spread * ((i - runs // 2) / max(runs // 2, 1)))
        else:
            pp_tps = base_pp_tps
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
        # PP=512: per EXPECTED_RUNS_PER_PP[512] (post-P5h+2.a typically 15)
        rows128_iron = gen_synthetic_rows(
            "ironmlx", 128, EXPECTED_RUNS_PER_PP[128], 1000.0
        )
        rows128_omlx = gen_synthetic_rows(
            "omlx", 128, EXPECTED_RUNS_PER_PP[128], 1100.0
        )
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
        assert result.returncode == 0, (
            f"aggregator failed: stdout={result.stdout} stderr={result.stderr}"
        )

        # Verify CI fields present
        summary = json.loads(out_json.read_text())
        for pp in ("128", "512"):
            row = summary["per_pp"][pp]
            assert "ironmlx_pp_tps_ci95_low" in row, (
                f"PP={pp} missing ironmlx CI fields"
            )
            assert "ironmlx_pp_tps_ci95_high" in row
            assert "ironmlx_pp_tps_ci95_half_width_pct" in row
            assert "omlx_pp_tps_ci95_low" in row
            assert "omlx_pp_tps_ci95_high" in row
            assert "omlx_pp_tps_ci95_half_width_pct" in row
            # Sanity: CI low <= median <= high
            assert row["ironmlx_pp_tps_ci95_low"] <= row["ironmlx_pp_tps_median"]
            assert row["ironmlx_pp_tps_median"] <= row["ironmlx_pp_tps_ci95_high"]


def test_aggregator_rejects_wrong_runs_per_pp_512():
    """If PP=512 CSV has 7 rows instead of EXPECTED_RUNS_PER_PP[512]=15, aggregator must fail."""
    if EXPECTED_RUNS_PER_PP[512] == 7:
        pytest.skip(
            "EXPECTED_RUNS_PER_PP[512]==7; test only applies when bumped above 7"
        )
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ironmlx_csv = td_path / "ironmlx.csv"
        omlx_csv = td_path / "omlx.csv"
        out_json = td_path / "summary.json"

        rows128_iron = gen_synthetic_rows(
            "ironmlx", 128, EXPECTED_RUNS_PER_PP[128], 1000.0
        )
        rows128_omlx = gen_synthetic_rows(
            "omlx", 128, EXPECTED_RUNS_PER_PP[128], 1100.0
        )
        # Intentionally wrong: PP=512 with only 7 rows
        rows512_iron = gen_synthetic_rows("ironmlx", 512, 7, 1500.0)
        rows512_omlx = gen_synthetic_rows(
            "omlx", 512, EXPECTED_RUNS_PER_PP[512], 2000.0
        )

        write_iron_bench_csv(rows128_iron + rows512_iron, ironmlx_csv)
        write_iron_bench_csv(rows128_omlx + rows512_omlx, omlx_csv)

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
        assert result.returncode != 0, (
            f"aggregator should have failed on PP=512 row count mismatch but exited 0; "
            f"stdout={result.stdout}"
        )
        assert "PP=512" in result.stderr or "PP=512" in result.stdout, (
            f"expected error message about PP=512; got stdout={result.stdout} stderr={result.stderr}"
        )
