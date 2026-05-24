"""Pytest for tools/p5i_c_pp_tps_envelope.py (P5i.c Phase 0)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
ENVELOPE_SCRIPT = TOOLS_DIR / "p5i_c_pp_tps_envelope.py"

FIELDNAMES = [
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


def write_iron_bench_csv(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            full = {fn: r.get(fn, "") for fn in FIELDNAMES}
            w.writerow(full)


def gen_pp_tps_rows(
    target: str, pp: int, runs: int, base_pp_tps: float, jitter: float = 0.01
) -> list[dict]:
    rows = []
    for i in range(runs):
        delta = jitter * (i - runs // 2) / max(runs // 2, 1)
        rows.append(
            {
                "target": target,
                "pp_target": str(pp),
                "tg_target": "1",
                "run_idx": str(i),
                "ttft_ms": "100.0",
                "tg_tps": "100.0",
                "tpot_ms": "10.0",
                "pp_tps": f"{base_pp_tps * (1 + delta):.4f}",
                "e2e_s": "1.0",
                "prompt_tokens_local": str(pp),
                "prompt_tokens_server": str(pp + 12),
                "completion_tokens_server": "1",
                "cached_tokens": "0",
                "finish_reason": "length",
            }
        )
    return rows


def test_envelope_pass_pp128():
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        csvs = []
        for r in range(3):
            csv_p = td_p / f"r{r}.csv"
            # stable across repeats: ±0.5% between-sweep variation
            write_iron_bench_csv(
                gen_pp_tps_rows("ironmlx", 128, 7, 1000.0 * (1 + 0.005 * r)), csv_p
            )
            csvs.append(csv_p)
        out_json = td_p / "out.json"
        args = []
        for c in csvs:
            args.extend(["--repeat-csv", str(c)])
        args.extend(["--pp", "128", "--out-json", str(out_json)])
        result = subprocess.run(
            [sys.executable, str(ENVELOPE_SCRIPT), *args],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(out_json.read_text())
        iron = data["ironmlx"]
        assert iron["verdict"] == "PASS"
        assert iron["final_uncertainty_envelope_pct"] <= 2.0
        assert len(iron["per_repeat"]) == 3
        for rep in iron["per_repeat"]:
            assert rep["n"] == 7


def test_envelope_rejects_too_few_repeats():
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        csv_p = td_p / "r0.csv"
        write_iron_bench_csv(gen_pp_tps_rows("ironmlx", 128, 7, 1000.0), csv_p)
        out_json = td_p / "out.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ENVELOPE_SCRIPT),
                "--repeat-csv",
                str(csv_p),
                "--pp",
                "128",
                "--out-json",
                str(out_json),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert ">=3" in result.stderr or "3" in result.stderr


def test_envelope_rejects_wrong_pp_target_in_row():
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        csvs = []
        for r in range(3):
            csv_p = td_p / f"r{r}.csv"
            # Inject wrong pp_target on first row only
            rows = gen_pp_tps_rows("ironmlx", 128, 7, 1000.0)
            if r == 0:
                rows[0]["pp_target"] = "256"
            write_iron_bench_csv(rows, csv_p)
            csvs.append(csv_p)
        out_json = td_p / "out.json"
        args = []
        for c in csvs:
            args.extend(["--repeat-csv", str(c)])
        args.extend(["--pp", "128", "--out-json", str(out_json)])
        result = subprocess.run(
            [sys.executable, str(ENVELOPE_SCRIPT), *args],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "pp_target" in result.stderr


def test_envelope_rejects_wrong_row_count():
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        csvs = []
        for r in range(3):
            csv_p = td_p / f"r{r}.csv"
            # PP=128 expects 7 rows; first sweep has 6
            rows = gen_pp_tps_rows("ironmlx", 128, 7 if r > 0 else 6, 1000.0)
            write_iron_bench_csv(rows, csv_p)
            csvs.append(csv_p)
        out_json = td_p / "out.json"
        args = []
        for c in csvs:
            args.extend(["--repeat-csv", str(c)])
        args.extend(["--pp", "128", "--out-json", str(out_json)])
        result = subprocess.run(
            [sys.executable, str(ENVELOPE_SCRIPT), *args],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "expected 7 rows" in result.stderr or "got 6" in result.stderr


def test_envelope_with_comparator_emits_delta():
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        ironmlx_csvs = []
        omlx_csvs = []
        for r in range(3):
            iron_p = td_p / f"iron-r{r}.csv"
            omlx_p = td_p / f"omlx-r{r}.csv"
            write_iron_bench_csv(
                gen_pp_tps_rows("ironmlx", 128, 7, 1000.0 * (1 + 0.005 * r)), iron_p
            )
            write_iron_bench_csv(
                gen_pp_tps_rows("omlx", 128, 7, 1100.0 * (1 + 0.005 * r)), omlx_p
            )
            ironmlx_csvs.append(iron_p)
            omlx_csvs.append(omlx_p)
        out_json = td_p / "out.json"
        args = []
        for c in ironmlx_csvs:
            args.extend(["--repeat-csv", str(c)])
        for c in omlx_csvs:
            args.extend(["--compare-repeat-csv", str(c)])
        args.extend(["--pp", "128", "--out-json", str(out_json)])
        result = subprocess.run(
            [sys.executable, str(ENVELOPE_SCRIPT), *args],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(out_json.read_text())
        assert "comparator" in data
        assert "delta_vs_comparator" in data
        delta = data["delta_vs_comparator"]
        # ironmlx ~1000, omlx ~1100 → delta ~ -9%
        assert -12.0 < delta["delta_pct_median"] < -7.0
