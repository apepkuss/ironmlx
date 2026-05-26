"""Pytest for tools/p5h_2b_thermal_overlay.py."""

from __future__ import annotations
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from p5h_2b_thermal_overlay import (  # noqa: E402
    _infer_timestamp_field,
    join_overlay,
    parse_powermetrics_samples,
)


def test_parse_powermetrics_jsonl(tmp_path):
    p = tmp_path / "thermal.json"
    p.write_text(
        '{"timestamp": 1000, "gpu_die_temp_c": 60}\n'
        '{"timestamp": 2000, "gpu_die_temp_c": 65}\n'
    )
    samples = parse_powermetrics_samples(p)
    assert len(samples) == 2
    assert _infer_timestamp_field(samples) == "timestamp"


def test_join_outlier_runs_hot(tmp_path):
    # Create cell with 3 runs; run 2 is outlier (slow) and runs hot.
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "bench.csv").write_text(
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason,run_start_unix_ns,run_end_unix_ns\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,1000000000,1100000000\n"
        "x,128,1,1,100.0,1.0,1.0,500.0,1.0,128,140,1,0,length,1200000000,1500000000\n"
        "x,128,1,2,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,1600000000,1700000000\n"
    )
    (cell / "meta.json").write_text(json.dumps({"mode": "production", "pp": 128}))
    samples = [
        {"timestamp": 1000, "gpu_die_temp_c": 60},
        {"timestamp": 1050, "gpu_die_temp_c": 61},
        {"timestamp": 1100, "gpu_die_temp_c": 62},
        {"timestamp": 1250, "gpu_die_temp_c": 80},  # in outlier window
        {"timestamp": 1400, "gpu_die_temp_c": 82},  # in outlier window
        {"timestamp": 1650, "gpu_die_temp_c": 63},
    ]
    result = join_overlay(samples, cell)
    assert result["correlation"] == "outliers_run_hot"
    outlier_runs = [o for o in result["overlay"] if o["is_outlier"]]
    assert len(outlier_runs) == 1
    assert outlier_runs[0]["run_idx"] == 1


def test_join_overlay_missing_run_end_unix_ns_raises(tmp_path):
    """Fix #2: missing run_end_unix_ns must raise SystemExit with both column names."""
    cell = tmp_path / "cell"
    cell.mkdir()
    # Only run_start_unix_ns present; run_end_unix_ns absent
    (cell / "bench.csv").write_text(
        "run_idx,pp_tps,run_start_unix_ns\n0,1000.0,1000000000\n"
    )
    (cell / "meta.json").write_text(json.dumps({}))
    samples = [{"timestamp": 1000, "gpu_die_temp_c": 60}]
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        join_overlay(samples, cell)
    msg = str(exc_info.value)
    assert "run_end_unix_ns" in msg, f"expected column name in error: {msg}"


def test_join_overlay_none_timestamp_field_no_error(tmp_path):
    """Fix #1: samples with ts_field value=None must not raise TypeError."""
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "bench.csv").write_text(
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason,run_start_unix_ns,run_end_unix_ns\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,1000000000,1100000000\n"
    )
    (cell / "meta.json").write_text(json.dumps({}))
    # Sample with timestamp=None simulates powermetrics null field
    samples = [
        {"timestamp": None, "gpu_die_temp_c": 60},
        {"timestamp": 1050, "gpu_die_temp_c": 61},
    ]
    # Should not raise; None sample simply falls outside window
    result = join_overlay(samples, cell)
    assert result["n_overlay_runs"] == 1
