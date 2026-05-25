"""Pytest for tools/p5h_2d_thermal_experiment.py Mechanism gate analyzer
(spec § 2.4)."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

import p5h_2d_thermal_experiment as drv  # noqa: E402


def _envelope_with_diagnostic(
    medians_pp128_per_repeat: list[dict], medians_pp512_per_repeat: list[dict]
) -> dict:
    """Build a synthetic combined envelope-like dict the analyzer consumes.

    Each per_repeat entry needs: trailing_slowdown_pct, fast_start_drop_pct.
    """
    return {
        "pp_envelopes": {
            "0s": {
                "128": {"ironmlx": {"per_repeat": medians_pp128_per_repeat}},
                "512": {"ironmlx": {"per_repeat": medians_pp512_per_repeat}},
            },
            "60s": {
                "128": {"ironmlx": {"per_repeat": medians_pp128_per_repeat}},
                "512": {"ironmlx": {"per_repeat": medians_pp512_per_repeat}},
            },
            "120s": {
                "128": {"ironmlx": {"per_repeat": medians_pp128_per_repeat}},
                "512": {"ironmlx": {"per_repeat": medians_pp512_per_repeat}},
            },
        }
    }


def test_strong_yes_when_both_pps_show_50pct_reduction_with_residual_under_10pct():
    """0s has 20% trailing on PP=128; 60s has 5% trailing -> 75% reduction.
    0s has 30% fast-start-drop on PP=512; 60s has 8% -> 73% reduction.
    Both PP-specific BEST residuals <= 10%.
    """
    pp128_0s = [{"trailing_slowdown_pct": -20.0, "fast_start_drop_pct": 5.0}] * 3
    pp128_60s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 2.0}] * 3
    pp128_120s = [{"trailing_slowdown_pct": -3.0, "fast_start_drop_pct": 1.0}] * 3
    pp512_0s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 30.0}] * 3
    pp512_60s = [{"trailing_slowdown_pct": -2.0, "fast_start_drop_pct": 8.0}] * 3
    pp512_120s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 6.0}] * 3
    matrix = {
        "0s": {"128": pp128_0s, "512": pp512_0s},
        "60s": {"128": pp128_60s, "512": pp512_60s},
        "120s": {"128": pp128_120s, "512": pp512_120s},
    }
    verdict = drv.compute_mechanism_gate(matrix)
    assert verdict["verdict"] == "strong_yes", verdict
    assert verdict["best_cooldown_per_pp"]["128"] in ("60s", "120s")
    assert verdict["best_cooldown_per_pp"]["512"] in ("60s", "120s")


def test_weak_yes_when_only_one_pp_reduction():
    """PP=128 reduces 75%; PP=512 unchanged."""
    pp128_0s = [{"trailing_slowdown_pct": -20.0, "fast_start_drop_pct": 5.0}] * 3
    pp128_60s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 2.0}] * 3
    pp128_120s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 2.0}] * 3
    pp512_0s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 30.0}] * 3
    pp512_60s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 28.0}] * 3
    pp512_120s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 27.0}] * 3
    matrix = {
        "0s": {"128": pp128_0s, "512": pp512_0s},
        "60s": {"128": pp128_60s, "512": pp512_60s},
        "120s": {"128": pp128_120s, "512": pp512_120s},
    }
    verdict = drv.compute_mechanism_gate(matrix)
    assert verdict["verdict"] == "weak_yes", verdict


def test_weak_yes_when_both_reduce_but_one_residual_above_10pct():
    """Both PPs reduce by >=50%, but PP=128 BEST residual remains >10%.
    Spec § 2.4 classifies this as weak_yes, not no.
    """
    pp128_0s = [{"trailing_slowdown_pct": -40.0, "fast_start_drop_pct": 5.0}] * 3
    pp128_60s = [{"trailing_slowdown_pct": -12.0, "fast_start_drop_pct": 2.0}] * 3
    pp128_120s = [{"trailing_slowdown_pct": -11.0, "fast_start_drop_pct": 1.0}] * 3
    pp512_0s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 30.0}] * 3
    pp512_60s = [{"trailing_slowdown_pct": -2.0, "fast_start_drop_pct": 8.0}] * 3
    pp512_120s = [{"trailing_slowdown_pct": -2.0, "fast_start_drop_pct": 7.0}] * 3
    matrix = {
        "0s": {"128": pp128_0s, "512": pp512_0s},
        "60s": {"128": pp128_60s, "512": pp512_60s},
        "120s": {"128": pp128_120s, "512": pp512_120s},
    }
    verdict = drv.compute_mechanism_gate(matrix)
    assert verdict["verdict"] == "weak_yes", verdict
    assert verdict["details"]["128"]["reduced_by_50pct"] is True
    assert verdict["details"]["128"]["residual_le_10pct"] is False


def test_no_when_neither_pp_reduces_50pct():
    """Both PPs show < 50% reduction across all cooldowns."""
    pp128_0s = [{"trailing_slowdown_pct": -20.0, "fast_start_drop_pct": 5.0}] * 3
    pp128_60s = [{"trailing_slowdown_pct": -18.0, "fast_start_drop_pct": 4.0}] * 3
    pp128_120s = [{"trailing_slowdown_pct": -17.0, "fast_start_drop_pct": 4.0}] * 3
    pp512_0s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 30.0}] * 3
    pp512_60s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 28.0}] * 3
    pp512_120s = [{"trailing_slowdown_pct": -5.0, "fast_start_drop_pct": 27.0}] * 3
    matrix = {
        "0s": {"128": pp128_0s, "512": pp512_0s},
        "60s": {"128": pp128_60s, "512": pp512_60s},
        "120s": {"128": pp128_120s, "512": pp512_120s},
    }
    verdict = drv.compute_mechanism_gate(matrix)
    assert verdict["verdict"] == "no", verdict


def test_no_when_0s_baseline_already_clean():
    """0s baseline residual <= 10% for both PPs -> mechanism not demonstrated.
    Spec § 2.4 last clause: classify as no/inconclusive."""
    pp128_0s = [{"trailing_slowdown_pct": -3.0, "fast_start_drop_pct": 2.0}] * 3
    pp128_60s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 1.0}] * 3
    pp128_120s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 1.0}] * 3
    pp512_0s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 5.0}] * 3
    pp512_60s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 2.0}] * 3
    pp512_120s = [{"trailing_slowdown_pct": -1.0, "fast_start_drop_pct": 2.0}] * 3
    matrix = {
        "0s": {"128": pp128_0s, "512": pp512_0s},
        "60s": {"128": pp128_60s, "512": pp512_60s},
        "120s": {"128": pp128_120s, "512": pp512_120s},
    }
    verdict = drv.compute_mechanism_gate(matrix)
    assert verdict["verdict"] == "no", verdict
    assert "baseline_already_clean" in verdict.get("reason", ""), verdict
