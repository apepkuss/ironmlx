"""Tests for tools.p5h_aggregator.roi_ranking.

Covers:
* compute_gap_weight per spec § 1.2 PP target gain table.
* is_kernel_bound covers T0b/T2/T3 kernel-bound spans.
* aggregate_per_pp builds per-PP per-span median exclusive.
* rank_top3_bottlenecks orders by max_gain_pct.
* rank_p5i / rank_p5j PP-set selection.
* feasibility_verdict 4-tier output.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.p5h_aggregator.roi_ranking import (
    KERNEL_BOUND_SPANS,
    KERNEL_REWRITE_REALISTIC_HIGH,
    OP_LEVEL_REALISTIC_HIGH,
    PP_TARGET_GAINS,
    Candidate,
    FeasibilityVerdict,
    PerPpAggregate,
    aggregate_per_pp,
    compute_gap_weight,
    feasibility_verdict,
    is_kernel_bound,
    rank_p5i,
    rank_p5j,
    rank_top3_bottlenecks,
    write_ranking_csv,
    write_verdict_json,
)


# --- compute_gap_weight ---


def test_compute_gap_weight_zero_current_gain_is_1():
    """At current_gain=0, gap_weight = target/target = 1.0."""
    for pp, _ in PP_TARGET_GAINS.items():
        assert compute_gap_weight(pp) == pytest.approx(1.0)


def test_compute_gap_weight_full_target_met_is_0():
    """current_gain >= target → gap_weight = 0 (no urgency)."""
    assert compute_gap_weight(128, current_gain=0.24) == pytest.approx(0.0)
    assert compute_gap_weight(128, current_gain=0.50) == pytest.approx(0.0)


def test_compute_gap_weight_partial_progress():
    """current_gain = 0.12 toward target 0.24 → weight = 0.5."""
    assert compute_gap_weight(128, current_gain=0.12) == pytest.approx(0.5)


def test_compute_gap_weight_unknown_pp_returns_0():
    assert compute_gap_weight(999) == 0.0


# --- is_kernel_bound ---


def test_is_kernel_bound_covers_t0b_h4():
    assert is_kernel_bound("gda_step_7_kernel_and_cache_update")
    assert is_kernel_bound("gda_step_8_out_proj")


def test_is_kernel_bound_covers_gated_attention_t2_4():
    assert is_kernel_bound("kv_mask_update")
    assert is_kernel_bound("fused_sdpa")


def test_is_kernel_bound_covers_moe_t3_4():
    assert is_kernel_bound("routing_sort_pack")
    assert is_kernel_bound("gather_qmm_gate_up")
    assert is_kernel_bound("gather_qmm_down")
    assert is_kernel_bound("routing_unsort_weighted_reduce")


def test_is_kernel_bound_rejects_non_kernel_spans():
    assert not is_kernel_bound("http_parse_render_tokenize")
    assert not is_kernel_bound("scheduler_admission")
    assert not is_kernel_bound("model_prefill_forward")


# --- aggregate_per_pp ---


def _attribution_row(
    *,
    pp,
    request_id,
    span_name,
    span_kind="tree",
    parent_span_id="1",
    span_id="2",
    inclusive_us="100.00",
    exclusive_us="100.00",
    routing_path="scheduler",
):
    return {
        "pp": str(pp),
        "request_id": request_id,
        "routing_path": routing_path,
        "span_name": span_name,
        "span_kind": span_kind,
        "parent_span_id": parent_span_id,
        "span_id": span_id,
        "inclusive_us": inclusive_us,
        "exclusive_us": exclusive_us,
    }


def _root_row(pp, rid, inclusive_us="1000.00", routing_path="scheduler"):
    return _attribution_row(
        pp=pp,
        request_id=rid,
        span_name="server_request_recv_to_first_content_sse_write",
        parent_span_id="",
        span_id="1",
        inclusive_us=inclusive_us,
        exclusive_us="500.00",
        routing_path=routing_path,
    )


def test_aggregate_per_pp_computes_root_median_per_pp():
    rows = [
        _root_row(pp=128, rid="r1", inclusive_us="1000.00"),
        _root_row(pp=128, rid="r2", inclusive_us="2000.00"),
        _root_row(pp=128, rid="r3", inclusive_us="3000.00"),
    ]
    agg = aggregate_per_pp(rows)
    assert 128 in agg
    assert agg[128].root_inclusive_us_median == pytest.approx(2000.0)


def test_aggregate_per_pp_excludes_diagnostic_rows():
    rows = [
        _root_row(pp=128, rid="r1"),
        _attribution_row(
            pp=128,
            request_id="r1",
            span_name="sse_write_role_chunk_diagnostic",
            span_kind="diagnostic",
            exclusive_us="",
        ),
        _attribution_row(
            pp=128,
            request_id="r1",
            span_name="http_parse_render_tokenize",
            inclusive_us="100.00",
            exclusive_us="100.00",
        ),
    ]
    agg = aggregate_per_pp(rows)
    assert "http_parse_render_tokenize" in agg[128].by_span_exclusive_us
    assert "sse_write_role_chunk_diagnostic" not in agg[128].by_span_exclusive_us


def test_aggregate_per_pp_excludes_root_self_from_candidates():
    rows = [
        _root_row(pp=128, rid="r1"),
    ]
    agg = aggregate_per_pp(rows)
    # Root span itself is excluded from candidates (it's the denominator).
    assert (
        "server_request_recv_to_first_content_sse_write"
        not in agg[128].by_span_exclusive_us
    )


def test_aggregate_per_pp_includes_synthesized_rows():
    rows = [
        _root_row(pp=128, rid="r1", inclusive_us="1000.00"),
        _attribution_row(
            pp=128,
            request_id="r1",
            span_name="unattributed_server_request_recv_to_first_content_sse_write",
            span_kind="synthesized",
            inclusive_us="500.00",
            exclusive_us="500.00",
        ),
    ]
    agg = aggregate_per_pp(rows)
    assert (
        "unattributed_server_request_recv_to_first_content_sse_write"
        in agg[128].by_span_exclusive_us
    )


# --- rank_top3_bottlenecks ---


def test_rank_top3_bottlenecks_orders_by_share():
    per_pp = {
        128: PerPpAggregate(
            pp=128,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={
                "small_op": 10.0,
                "medium_op": 100.0,
                "huge_op": 500.0,
                "another": 200.0,
                "tiny": 1.0,
            },
        )
    }
    top3 = rank_top3_bottlenecks(per_pp)
    names = [c.span_name for c in top3[128]]
    assert names == ["huge_op", "another", "medium_op"]


# --- rank_p5i / rank_p5j ---


def test_rank_p5i_only_selects_pp_128_and_512():
    per_pp = {
        128: PerPpAggregate(
            pp=128,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={"op_a": 100.0},
        ),
        512: PerPpAggregate(
            pp=512,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={"op_b": 200.0},
        ),
        2048: PerPpAggregate(
            pp=2048,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={"op_c": 300.0},
        ),
    }
    p5i = rank_p5i(per_pp)
    pps = {c.pp for c in p5i}
    assert pps == {128, 512}


def test_rank_p5j_includes_pp_2048_through_16384():
    per_pp = {
        pp: PerPpAggregate(
            pp=pp,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={"op_x": 100.0},
        )
        for pp in (128, 512, 2048, 4096, 8192, 16384)
    }
    p5j = rank_p5j(per_pp)
    pps = {c.pp for c in p5j}
    assert pps == {2048, 4096, 8192, 16384}


def test_rank_p5j_lane_b_caveat_in_notes():
    per_pp = {
        4096: PerPpAggregate(
            pp=4096,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={"op_x": 100.0},
        )
    }
    p5j = rank_p5j(per_pp)
    assert all(c.lane == "B" for c in p5j)
    assert any("lane_b_top_level_only" in c.notes for c in p5j)


def test_rank_p5i_lane_a_no_caveat():
    per_pp = {
        128: PerPpAggregate(
            pp=128,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={"op_x": 100.0},
        )
    }
    p5i = rank_p5i(per_pp)
    assert all(c.lane == "A" for c in p5i)
    assert not any("lane_b_top_level_only" in c.notes for c in p5i)


def test_kernel_bound_candidate_gets_scope_gate_flag_and_higher_realistic():
    per_pp = {
        128: PerPpAggregate(
            pp=128,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={
                "fused_sdpa": 100.0,  # kernel bound
                "scheduler_admission": 100.0,  # NOT kernel bound
            },
        )
    }
    p5i = rank_p5i(per_pp)
    kernel = next(c for c in p5i if c.span_name == "fused_sdpa")
    non_kernel = next(c for c in p5i if c.span_name == "scheduler_admission")
    assert kernel.scope_gate_trigger
    assert not non_kernel.scope_gate_trigger
    # Kernel rewrite gets higher realistic range (50-70% vs 30-50%).
    assert kernel.realistic_high_gain_pct > non_kernel.realistic_high_gain_pct
    # Both have same max_gain_pct (same measured exclusive / root).
    assert kernel.max_gain_pct == pytest.approx(non_kernel.max_gain_pct)


# --- feasibility_verdict ---


def _candidate(span_name, pp, max_gain_pct, scope_gate=False) -> Candidate:
    realistic_high = max_gain_pct * (
        KERNEL_REWRITE_REALISTIC_HIGH if scope_gate else OP_LEVEL_REALISTIC_HIGH
    )
    return Candidate(
        span_name=span_name,
        pp=pp,
        measured_exclusive_us=max_gain_pct * 1000.0,
        root_inclusive_us=1000.0,
        max_gain_pct=max_gain_pct,
        realistic_low_gain_pct=max_gain_pct * 0.3,
        realistic_high_gain_pct=realistic_high,
        gap_weight=1.0,
        score=max_gain_pct,
        scope_gate_trigger=scope_gate,
        lane="A" if pp in (128, 512, 2048) else "B",
    )


def test_feasibility_verdict_yes_when_op_only_meets_target():
    # PP=128 target=0.24. Two op-level cands at 0.30 each, realistic_high
    # = 0.30 * 0.5 = 0.15 each → sum 0.30 >= 0.24 → YES.
    cands = [_candidate("a", 128, 0.30), _candidate("b", 128, 0.30)]
    v = feasibility_verdict(cands, [], pp=128)
    assert v == FeasibilityVerdict.YES


def test_feasibility_verdict_yes_with_scope_gate():
    # PP=128 target=0.24. Op-only realistic_high sum = 0.30*0.5 = 0.15 < 0.24.
    # Add kernel cand 0.30 → realistic_high = 0.30*0.7 = 0.21; total
    # = 0.15 + 0.21 = 0.36 >= 0.24 → YES_WITH_SCOPE_GATE.
    cands = [
        _candidate("a", 128, 0.30, scope_gate=False),
        _candidate("b", 128, 0.30, scope_gate=True),
    ]
    v = feasibility_verdict(cands, [], pp=128)
    assert v == FeasibilityVerdict.YES_WITH_SCOPE_GATE


def test_feasibility_verdict_no_under_measured_cap():
    # PP=2048 target=1.10. Single op cand 0.30 + kernel 0.30. Sum realistic
    # = 0.15 + 0.21 = 0.36 << 1.10 → NO_UNDER_MEASURED_CAP.
    cands = [
        _candidate("a", 2048, 0.30, scope_gate=False),
        _candidate("b", 2048, 0.30, scope_gate=True),
    ]
    v = feasibility_verdict([], cands, pp=2048)
    assert v == FeasibilityVerdict.NO_UNDER_MEASURED_CAP


def test_feasibility_verdict_data_insufficient_when_no_candidates():
    v = feasibility_verdict([], [], pp=128)
    assert v == FeasibilityVerdict.DATA_INSUFFICIENT


def test_feasibility_verdict_data_insufficient_unknown_pp():
    v = feasibility_verdict([_candidate("a", 999, 1.0)], [], pp=999)
    assert v == FeasibilityVerdict.DATA_INSUFFICIENT


# --- CSV / JSON emission ---


def test_write_ranking_csv_emits_top3_and_p5i_and_p5j(tmp_path: Path):
    top3 = {
        128: [
            Candidate(
                span_name="a",
                pp=128,
                measured_exclusive_us=100.0,
                root_inclusive_us=1000.0,
                max_gain_pct=0.1,
                realistic_low_gain_pct=0.03,
                realistic_high_gain_pct=0.05,
                gap_weight=1.0,
                score=0.1,
                scope_gate_trigger=False,
                lane="A",
            )
        ]
    }
    p5i = [
        Candidate(
            span_name="b",
            pp=512,
            measured_exclusive_us=200.0,
            root_inclusive_us=1000.0,
            max_gain_pct=0.2,
            realistic_low_gain_pct=0.06,
            realistic_high_gain_pct=0.10,
            gap_weight=1.0,
            score=0.2,
            scope_gate_trigger=False,
            lane="A",
        )
    ]
    p5j = [
        Candidate(
            span_name="c",
            pp=4096,
            measured_exclusive_us=300.0,
            root_inclusive_us=1000.0,
            max_gain_pct=0.3,
            realistic_low_gain_pct=0.15,
            realistic_high_gain_pct=0.21,
            gap_weight=1.0,
            score=0.3,
            scope_gate_trigger=True,
            lane="B",
        )
    ]
    out = tmp_path / "ranking.csv"
    write_ranking_csv(top3, p5i, p5j, out)
    rows = list(csv.DictReader(out.open()))
    cats = {r["category"] for r in rows}
    assert cats == {"top3", "p5i", "p5j"}
    p5j_row = next(r for r in rows if r["category"] == "p5j")
    assert p5j_row["scope_gate_trigger"] == "True"
    assert p5j_row["lane"] == "B"


def test_write_verdict_json_structure(tmp_path: Path):
    per_pp = {
        128: PerPpAggregate(
            pp=128,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={"op_a": 300.0},
        )
    }
    p5i = rank_p5i(per_pp)
    out = tmp_path / "verdict.json"
    verdict = write_verdict_json(per_pp, p5i, [], out)
    payload = json.loads(out.read_text())
    assert payload["128"]["target_gain_pct"] == 0.24
    assert payload["128"]["lane"] == "A"
    assert payload["128"]["verdict"] in {v.value for v in FeasibilityVerdict}
    assert verdict["128"]["candidate_count"] == 1


# --- end-to-end CLI ---


def test_roi_ranking_cli_runs(tmp_path: Path):
    """End-to-end CLI smoke test: build attribution CSV by hand, run module,
    assert outputs exist + verdict JSON is valid."""
    attribution = tmp_path / "attribution.csv"
    summary = tmp_path / "summary.csv"
    ranking = tmp_path / "ranking.csv"
    verdict = tmp_path / "verdict.json"

    with attribution.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "pp",
                "request_id",
                "routing_path",
                "span_name",
                "span_kind",
                "parent_span_id",
                "span_id",
                "inclusive_us",
                "exclusive_us",
            ],
        )
        w.writeheader()
        w.writerow(_root_row(pp=128, rid="r1", inclusive_us="1000.00"))
        w.writerow(
            _attribution_row(
                pp=128,
                request_id="r1",
                span_name="fused_sdpa",
                inclusive_us="500.00",
                exclusive_us="500.00",
            )
        )

    summary.write_text(
        "pp,request_count,root_inclusive_us_median,coverage_pct_median,"
        "coverage_pct_min,top1_span_name,top1_share,top2_span_name,top2_share,"
        "top3_span_name,top3_share\n"
        "128,1,1000.00,1.0000,1.0000,fused_sdpa,0.5000,,,,\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.p5h_aggregator.roi_ranking",
            "--attribution-csv",
            str(attribution),
            "--summary-csv",
            str(summary),
            "--out-ranking",
            str(ranking),
            "--out-verdict",
            str(verdict),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}\nstderr:\n{result.stderr}"
    )
    assert ranking.exists()
    assert verdict.exists()
    v = json.loads(verdict.read_text())
    assert "128" in v
    assert v["128"]["candidate_count"] >= 1


def test_kernel_bound_spans_set_is_documented():
    """Smoke check: spec-declared kernel-bound spans are in the set."""
    expected_min = {
        "gda_step_7_kernel_and_cache_update",
        "gda_step_8_out_proj",
        "kv_mask_update",
        "fused_sdpa",
        "routing_sort_pack",
        "gather_qmm_gate_up",
        "gather_qmm_down",
        "routing_unsort_weighted_reduce",
    }
    assert expected_min <= KERNEL_BOUND_SPANS
