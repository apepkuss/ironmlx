"""Pytest for P5i.c Phase 0 roi_ranking extensions (spec § 4.2.4 + § 8/9/10)."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from p5h_aggregator.roi_ranking import (  # noqa: E402
    emit_category_coverage,
    emit_phase_1_default_rule,
    evaluate_dense_diagnostic_trigger,
    identify_tied_tiers,
)


# --- Tied-tier algorithm (spec § 8) ---


def test_identify_tied_tiers_separate():
    ranking = [("a", 25.0), ("b", 10.0), ("c", 5.0)]
    ci95 = {"a": (23.0, 27.0), "b": (8.0, 12.0), "c": (4.0, 6.0)}
    tiers = identify_tied_tiers(ranking, ci95)
    assert tiers == [["a"], ["b"], ["c"]]


def test_identify_tied_tiers_merge_adjacent():
    ranking = [("a", 25.0), ("b", 23.0), ("c", 5.0)]
    ci95 = {"a": (22.0, 28.0), "b": (20.0, 26.0), "c": (4.0, 6.0)}
    tiers = identify_tied_tiers(ranking, ci95)
    # a CI95 low=22 <= b CI95 high=26 → merge; b low=20 > c high=6 → split
    assert tiers == [["a", "b"], ["c"]]


def test_identify_tied_tiers_chain():
    # A overlaps B, B overlaps C, A may not overlap C — chain merges all 3
    ranking = [("a", 25.0), ("b", 20.0), ("c", 15.0)]
    ci95 = {"a": (22.0, 28.0), "b": (18.0, 23.0), "c": (13.0, 18.5)}
    tiers = identify_tied_tiers(ranking, ci95)
    assert tiers == [["a", "b", "c"]]


def test_identify_tied_tiers_empty():
    assert identify_tied_tiers([], {}) == []


# --- 4-category coverage (spec § 4.2.4) ---


def test_emit_category_coverage_all_measured():
    audit = {
        "scheduler": "measured",
        "kv_cache": "measured",
        "attention": "measured",
        "moe": "measured",
    }
    measured_spans = {
        "scheduler_admission",
        "cache_state_update",
        "fused_sdpa",
        "gather_qmm_gate_up",
    }
    result = emit_category_coverage(audit, measured_spans)
    assert result == {
        "scheduler": "measured",
        "kv_cache": "measured",
        "attention": "measured",
        "moe": "measured",
    }


def test_emit_category_coverage_proxy_only_preserved():
    """Per spec § 3.2 + Codex round-2: proxy-only flows through as limitation."""
    audit = {"kv_cache": "proxy-only", "moe": "measured"}
    measured_spans = {"cache_state_update", "gather_qmm_gate_up"}
    result = emit_category_coverage(audit, measured_spans)
    assert result["kv_cache"] == "proxy-only"
    assert result["moe"] == "measured"


def test_emit_category_coverage_measured_independent_of_ranking_presence():
    """Per Codex round-2: declared `measured` means schema-level coverage; a
    span below MIN_SHARE_PCT can still be `measured`. Ranking-presence is
    surfaced separately via the per-PP top-N table, NOT by downgrading
    coverage status."""
    audit = {"scheduler": "measured"}
    measured_spans = {"gather_qmm_gate_up"}  # no scheduler-category span in top-N
    result = emit_category_coverage(audit, measured_spans)
    assert result == {"scheduler": "measured"}


def test_emit_category_coverage_unmeasured_declared_stays_unmeasured():
    audit = {"scheduler": "unmeasured"}
    measured_spans = {"gather_qmm_gate_up"}
    result = emit_category_coverage(audit, measured_spans)
    assert result == {"scheduler": "unmeasured"}


# --- Phase 1 default rule R1/R2/R3 (spec § 9) ---


def test_emit_phase_1_default_rule_R1():
    ranking_per_pp = {
        128: [("gather_qmm_gate_up", 25.0)],
        512: [("gather_qmm_gate_up", 23.0)],
    }
    tiers_per_pp = {128: [["gather_qmm_gate_up"]], 512: [["gather_qmm_gate_up"]]}
    coverage = {"moe": "measured"}
    result = emit_phase_1_default_rule(ranking_per_pp, tiers_per_pp, coverage)
    assert result["triggered_rule"] == "R1"
    assert result["suggested_phase_1_candidates"] == ["gather_qmm_gate_up"]


def test_emit_phase_1_default_rule_R2_pp_divergence():
    ranking_per_pp = {
        128: [("gather_qmm_gate_up", 25.0)],
        512: [("fused_sdpa", 22.0)],
    }
    tiers_per_pp = {128: [["gather_qmm_gate_up"]], 512: [["fused_sdpa"]]}
    coverage = {"moe": "measured", "attention": "measured"}
    result = emit_phase_1_default_rule(ranking_per_pp, tiers_per_pp, coverage)
    assert result["triggered_rule"] == "R2"
    assert set(result["suggested_phase_1_candidates"]) == {
        "gather_qmm_gate_up",
        "fused_sdpa",
    }


def test_emit_phase_1_default_rule_R3_tied_tier():
    ranking_per_pp = {
        128: [("a", 22.0), ("b", 21.0)],
        512: [("a", 22.0), ("b", 21.0)],
    }
    tiers_per_pp = {128: [["a", "b"]], 512: [["a", "b"]]}
    coverage = {"moe": "measured"}
    result = emit_phase_1_default_rule(ranking_per_pp, tiers_per_pp, coverage)
    assert result["triggered_rule"] == "R3"


def test_emit_phase_1_default_rule_data_insufficient_empty_rankings():
    result = emit_phase_1_default_rule({}, {}, {})
    assert result["triggered_rule"] == "data_insufficient"


def test_emit_phase_1_default_rule_data_insufficient_no_tier_1():
    ranking_per_pp = {128: [("a", 25.0)]}
    tiers_per_pp = {128: []}  # empty tiers
    result = emit_phase_1_default_rule(ranking_per_pp, tiers_per_pp, {})
    assert result["triggered_rule"] == "data_insufficient"


# --- Dense diagnostic triggers (spec § 10) ---


def test_dense_trigger_A_non_moe_top_above_threshold():
    tiers = {128: [["fused_sdpa"]]}
    medians = {128: {"fused_sdpa": 18.0}}
    result = evaluate_dense_diagnostic_trigger(tiers, medians)
    assert result["triggered"] is True
    assert "trigger-A" in result["reason"]


def test_dense_trigger_A_non_moe_top_below_threshold():
    tiers = {128: [["fused_sdpa"]]}
    medians = {128: {"fused_sdpa": 10.0}}  # below 15% threshold
    result = evaluate_dense_diagnostic_trigger(tiers, medians)
    assert result["triggered"] is False


def test_dense_trigger_skip_moe_dominant():
    tiers = {128: [["gather_qmm_gate_up"]]}
    medians = {128: {"gather_qmm_gate_up": 25.0}}
    result = evaluate_dense_diagnostic_trigger(tiers, medians)
    assert result["triggered"] is False
    assert "MoE" in result["reason"]


def test_dense_trigger_B_mixed_tier_fires():
    """Mixed MoE + non-MoE in tier-1 fires trigger-B (or trigger-A first if magnitude crosses)."""
    tiers = {128: [["gather_qmm_gate_up", "fused_sdpa"]]}
    medians = {128: {"gather_qmm_gate_up": 22.0, "fused_sdpa": 18.0}}
    result = evaluate_dense_diagnostic_trigger(tiers, medians)
    assert result["triggered"] is True
    # Either trigger-A (fused_sdpa above 15%) or trigger-B (mixed); both honored.
    assert "trigger-A" in result["reason"] or "trigger-B" in result["reason"]
