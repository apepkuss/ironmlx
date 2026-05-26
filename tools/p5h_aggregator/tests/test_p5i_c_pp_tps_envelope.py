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


def test_pp128_uses_small_pp_acceptance_threshold():
    """PP=128 uses the small-PP acceptance threshold: an envelope between
    2.0% and 2.5% should pass and be explicitly labelled."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        csvs = []
        for r, base_pp_tps in enumerate([1000.0, 1000.0, 1046.0]):
            csv_p = td_p / f"r{r}.csv"
            write_iron_bench_csv(
                gen_pp_tps_rows(
                    "ironmlx", 128, 7, base_pp_tps, jitter=0.0
                ),
                csv_p,
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
        iron = json.loads(out_json.read_text())["ironmlx"]
        assert 2.0 < iron["final_uncertainty_envelope_pct"] < 2.5
        assert iron["target_pct"] == 2.5
        assert iron["target_policy"] == "small_pp_acceptance_threshold"
        assert iron["verdict"] == "PASS"


def test_pp512_keeps_standard_acceptance_threshold():
    """PP=512 keeps the standard 2.0% threshold; the same envelope that
    PP=128 accepts should still fail here."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        csvs = []
        for r, base_pp_tps in enumerate([1000.0, 1000.0, 1046.0]):
            csv_p = td_p / f"r{r}.csv"
            write_iron_bench_csv(
                gen_pp_tps_rows(
                    "ironmlx", 512, 15, base_pp_tps, jitter=0.0
                ),
                csv_p,
            )
            csvs.append(csv_p)
        out_json = td_p / "out.json"
        args = []
        for c in csvs:
            args.extend(["--repeat-csv", str(c)])
        args.extend(["--pp", "512", "--out-json", str(out_json)])
        result = subprocess.run(
            [sys.executable, str(ENVELOPE_SCRIPT), *args],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        iron = json.loads(out_json.read_text())["ironmlx"]
        assert 2.0 < iron["final_uncertainty_envelope_pct"] < 2.5
        assert iron["target_pct"] == 2.0
        assert iron["target_policy"] == "standard_acceptance_threshold"
        assert iron["verdict"] == "FAIL"


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


def _make_cell_csv(path: Path, pp_tps_values: list[float], pp: int) -> None:
    """Helper: write iron-bench CSV with given pp_tps series; pp_target fixed."""
    rows = []
    for idx, v in enumerate(pp_tps_values):
        rows.append({
            "target": "p5i_c",
            "pp_target": str(pp),
            "tg_target": "1",
            "run_idx": str(idx),
            "ttft_ms": "100",
            "tg_tps": "1.0",
            "tpot_ms": "1.0",
            "pp_tps": str(v),
            "e2e_s": "1.0",
            "prompt_tokens_local": str(pp),
            "prompt_tokens_server": str(pp),
            "completion_tokens_server": "1",
            "cached_tokens": "0",
            "finish_reason": "length",
        })
    write_iron_bench_csv(rows, path)


def test_trailing_slowdown_pct_emitted_per_repeat():
    """Positive trailing-slowdown case: last-3 median < first-3 median.

    Series: [1000]*12 + [800,800,800]. first3=1000, last3=800.
    trailing_slowdown_pct = 800/1000 - 1 = -0.20 = -20.0%
    """
    pp = 128
    series = [1000.0] * 12 + [800.0, 800.0, 800.0]
    assert len(series) == 15
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        csv_paths = [tmp_p / f"r{r}_pp{pp}.csv" for r in (1, 2, 3)]
        for p_csv in csv_paths:
            _make_cell_csv(p_csv, series, pp)
        out_json = tmp_p / "out.json"
        cmd = [
            sys.executable, str(ENVELOPE_SCRIPT), "--pp", str(pp),
            "--out-json", str(out_json), "--expected-runs", str(len(series)),
        ]
        for c in csv_paths:
            cmd.extend(["--repeat-csv", str(c)])
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"stderr={r.stderr}"
        result = json.loads(out_json.read_text())
        for per_rep in result["ironmlx"]["per_repeat"]:
            assert "trailing_slowdown_pct" in per_rep
            assert abs(per_rep["trailing_slowdown_pct"] - (-20.0)) < 0.01
            assert "first_3_runs_median_pp_tps" in per_rep
            assert "last_3_runs_median_pp_tps" in per_rep
            assert abs(per_rep["first_3_runs_median_pp_tps"] - 1000.0) < 0.01
            assert abs(per_rep["last_3_runs_median_pp_tps"] - 800.0) < 0.01


def test_fast_start_drop_pct_emitted_per_repeat():
    """Positive fast-start-drop case: max(first-3) > median(last-3).

    Series: [1500,1500,1500] + [1200]*12. first3_max=1500, last3_med=1200.
    fast_start_drop_pct = 1500/1200 - 1 = +0.25 = +25.0%
    """
    pp = 512
    series = [1500.0, 1500.0, 1500.0] + [1200.0] * 12
    assert len(series) == 15
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        csv_paths = [tmp_p / f"r{r}_pp{pp}.csv" for r in (1, 2, 3)]
        for p_csv in csv_paths:
            _make_cell_csv(p_csv, series, pp)
        out_json = tmp_p / "out.json"
        cmd = [
            sys.executable, str(ENVELOPE_SCRIPT), "--pp", str(pp),
            "--out-json", str(out_json), "--expected-runs", str(len(series)),
        ]
        for c in csv_paths:
            cmd.extend(["--repeat-csv", str(c)])
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"stderr={r.stderr}"
        result = json.loads(out_json.read_text())
        for per_rep in result["ironmlx"]["per_repeat"]:
            assert abs(per_rep["fast_start_drop_pct"] - 25.0) < 0.01


def test_diagnostic_fields_present_for_short_series_degenerate():
    """N < 3 runs: trailing_slowdown_pct / fast_start_drop_pct fields MUST
    still be present in JSON (null), so downstream tooling does not KeyError.
    Override --expected-runs to 2 to bypass the per-PP row-count guard.
    """
    pp = 128
    series = [1000.0, 1000.0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        csv_paths = [tmp_p / f"r{r}_pp{pp}.csv" for r in (1, 2, 3)]
        for p_csv in csv_paths:
            _make_cell_csv(p_csv, series, pp)
        out_json = tmp_p / "out.json"
        cmd = [
            sys.executable, str(ENVELOPE_SCRIPT), "--pp", str(pp),
            "--out-json", str(out_json), "--expected-runs", "2",
        ]
        for c in csv_paths:
            cmd.extend(["--repeat-csv", str(c)])
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"stderr={r.stderr}"
        result = json.loads(out_json.read_text())
        for per_rep in result["ironmlx"]["per_repeat"]:
            assert "trailing_slowdown_pct" in per_rep
            assert "fast_start_drop_pct" in per_rep
            assert per_rep["trailing_slowdown_pct"] is None
            assert per_rep["fast_start_drop_pct"] is None


def test_diagnostic_fields_at_exactly_3_runs_boundary():
    """Regression guard for the N=3 edge case: first_3 and last_3 are the same
    3 elements, so trailing_slowdown_pct is always 0% (not meaningful as a
    "trailing slowdown" signal). Verify the fields are computed without crash
    and produce the documented degenerate values.
    """
    pp = 128
    series = [1000.0, 900.0, 800.0]  # downward trend within first/last 3
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        csv_paths = [tmp_p / f"r{r}_pp{pp}.csv" for r in (1, 2, 3)]
        for p_csv in csv_paths:
            _make_cell_csv(p_csv, series, pp)
        out_json = tmp_p / "out.json"
        cmd = [
            sys.executable, str(ENVELOPE_SCRIPT), "--pp", str(pp),
            "--out-json", str(out_json), "--expected-runs", str(len(series)),
        ]
        for c in csv_paths:
            cmd.extend(["--repeat-csv", str(c)])
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"stderr={r.stderr}"
        result = json.loads(out_json.read_text())
        for per_rep in result["ironmlx"]["per_repeat"]:
            # first_3 == last_3 == series, so:
            # first_3_med = median([1000, 900, 800]) = 900
            # last_3_med = median([1000, 900, 800]) = 900
            # trailing_slowdown_pct = 900/900 - 1 = 0%
            assert abs(per_rep["trailing_slowdown_pct"]) < 0.01
            # fast_start_drop_pct = max([1000, 900, 800]) / 900 - 1 = 0.111 -> 11.11%
            assert abs(per_rep["fast_start_drop_pct"] - 11.11) < 0.1
            assert abs(per_rep["first_3_runs_median_pp_tps"] - 900.0) < 0.01
            assert abs(per_rep["last_3_runs_median_pp_tps"] - 900.0) < 0.01
