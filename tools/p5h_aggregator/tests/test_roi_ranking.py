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
    SPEC_LANE_A_PP_SET,
    SPEC_LANE_B_PP_SET,
    Candidate,
    FeasibilityVerdict,
    PerPpAggregate,
    aggregate_per_pp,
    compute_gap_weight,
    feasibility_verdict,
    is_kernel_bound,
    observed_lane_for_pp,
    rank_p5i,
    rank_p5j,
    rank_top3_bottlenecks,
    warn_lane_divergence,
    wrapper_dominated_verdict_explanation,
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


def test_aggregate_per_pp_excludes_synthesized_rows():
    """Fix C: ``unattributed_<span>`` synthesized residual rows are EXCLUDED
    from the ROI candidate pool. They are not actionable optimization targets
    and double-count the parent's exclusive_us (which already = residual)."""
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
        not in agg[128].by_span_exclusive_us
    )


def test_aggregate_per_pp_excludes_all_unattributed_residuals():
    """Fix C: every synthesized residual must be excluded regardless of parent."""
    rows = [
        _root_row(pp=128, rid="r1", inclusive_us="1000.00"),
        _attribution_row(
            pp=128,
            request_id="r1",
            span_name="real_op",
            inclusive_us="100.00",
            exclusive_us="100.00",
        ),
        _attribution_row(
            pp=128,
            request_id="r1",
            span_name="unattributed_real_op",
            span_kind="synthesized",
            inclusive_us="50.00",
            exclusive_us="50.00",
        ),
        _attribution_row(
            pp=128,
            request_id="r1",
            span_name="unattributed_http_parse_render_tokenize",
            span_kind="synthesized",
            inclusive_us="42.00",
            exclusive_us="42.00",
        ),
    ]
    agg = aggregate_per_pp(rows)
    assert "real_op" in agg[128].by_span_exclusive_us
    for name in agg[128].by_span_exclusive_us:
        assert not name.startswith("unattributed_"), (
            f"synthesized residual {name} should not appear in ROI candidates"
        )


def test_aggregate_per_pp_sums_multi_emit_span_per_request():
    """Fix A: multi-emit spans (gs_chunk_N, decoder_layer_N) are summed PER
    REQUEST before median across requests, NOT per-record median."""
    # Two requests at PP=8192, each emits gs_chunk_N 5 times of 100us each.
    # Per-request total = 500us; root = 1000us; share = 0.5.
    # WRONG (per-record median): 100us → share 0.1.
    # RIGHT (per-request total median): 500us → share 0.5.
    rows: list[dict] = []
    for rid in ("r1", "r2"):
        rows.append(_root_row(pp=8192, rid=rid, inclusive_us="1000.00"))
        for _ in range(5):
            rows.append(
                _attribution_row(
                    pp=8192,
                    request_id=rid,
                    span_name="gs_chunk_N",
                    inclusive_us="100.00",
                    exclusive_us="100.00",
                )
            )
    agg = aggregate_per_pp(rows)
    assert agg[8192].by_span_exclusive_us["gs_chunk_N"] == pytest.approx(500.0)
    assert agg[8192].by_span["gs_chunk_N"] == pytest.approx(0.5)


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
    """Fix B: lane comes from OBSERVED routing_path (gs_chunked → 'B'), not
    spec partition. Caller must pass observed_lane dict."""
    per_pp = {
        4096: PerPpAggregate(
            pp=4096,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={"op_x": 100.0},
        )
    }
    p5j = rank_p5j(per_pp, observed_lane={4096: "B"})
    assert all(c.lane == "B" for c in p5j)
    assert any("lane_b_top_level_only" in c.notes for c in p5j)


def test_rank_p5i_lane_a_no_caveat():
    """Fix B: scheduler routing → 'A'."""
    per_pp = {
        128: PerPpAggregate(
            pp=128,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={"op_x": 100.0},
        )
    }
    p5i = rank_p5i(per_pp, observed_lane={128: "A"})
    assert all(c.lane == "A" for c in p5i)
    assert not any("lane_b_top_level_only" in c.notes for c in p5i)


def test_rank_p5i_unknown_observed_lane_marks_question():
    """When no observed_lane supplied, lane defaults to '?'."""
    per_pp = {
        128: PerPpAggregate(
            pp=128,
            root_inclusive_us_median=1000.0,
            by_span_exclusive_us={"op_x": 100.0},
        )
    }
    p5i = rank_p5i(per_pp)
    assert all(c.lane == "?" for c in p5i)


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
    p5i = rank_p5i(per_pp, observed_lane={128: "A"})
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
    obs = {128: "A"}
    p5i = rank_p5i(per_pp, observed_lane=obs)
    out = tmp_path / "verdict.json"
    verdict = write_verdict_json(per_pp, p5i, [], out, observed_lane=obs)
    payload = json.loads(out.read_text())
    assert payload["128"]["target_gain_pct"] == 0.24
    # Fix B: per-PP entry carries spec_lane (partition) + observed_lane.
    assert payload["128"]["spec_lane"] == "A"
    assert payload["128"]["observed_lane"] == "A"
    assert payload["128"]["lane"] == "A"  # backward-compat alias
    assert payload["128"]["verdict"] in {v.value for v in FeasibilityVerdict}
    assert verdict["128"]["candidate_count"] == 1


# --- Fix B: observed_lane_for_pp + warn_lane_divergence ---


def test_observed_lane_for_pp_scheduler_only():
    rows = [
        _root_row(pp=128, rid="r1", routing_path="scheduler"),
        _attribution_row(
            pp=128,
            request_id="r1",
            span_name="op_a",
            routing_path="scheduler",
        ),
    ]
    obs = observed_lane_for_pp(rows)
    assert obs == {128: "A"}


def test_observed_lane_for_pp_gs_chunked_at_lane_a_pp():
    """Fix B canonical case: PP=2048 spec partition Lane A, but chat-template
    overhead pushes prompt > prefill_chunk_size → observed gs_chunked = Lane B."""
    rows = [
        _root_row(pp=2048, rid="r1", routing_path="gs_chunked"),
        _attribution_row(
            pp=2048,
            request_id="r1",
            span_name="gs_chunk_N",
            routing_path="gs_chunked",
        ),
    ]
    obs = observed_lane_for_pp(rows)
    assert obs == {2048: "B"}


def test_observed_lane_for_pp_mixed():
    rows = [
        _root_row(pp=2048, rid="r1", routing_path="scheduler"),
        _root_row(pp=2048, rid="r2", routing_path="gs_chunked"),
    ]
    obs = observed_lane_for_pp(rows)
    assert obs == {2048: "mixed"}


def test_warn_lane_divergence_emits_for_pp_2048():
    """PP=2048 spec Lane A; observed Lane B → must warn."""
    import io

    buf = io.StringIO()
    warn_lane_divergence({2048: "B"}, stream=buf)
    out = buf.getvalue()
    assert "PP=2048" in out
    assert "observed_lane=B" in out
    assert "spec partition lane=A" in out


def test_warn_lane_divergence_silent_when_match():
    import io

    buf = io.StringIO()
    warn_lane_divergence({128: "A", 4096: "B"}, stream=buf)
    assert buf.getvalue() == ""


def test_spec_lane_partitions_distinct():
    """Sanity: SPEC_LANE_A / SPEC_LANE_B together cover the 6 measurement PPs."""
    assert SPEC_LANE_A_PP_SET == {128, 512, 2048}
    assert SPEC_LANE_B_PP_SET == {4096, 8192, 16384}
    assert SPEC_LANE_A_PP_SET.isdisjoint(SPEC_LANE_B_PP_SET)


# --- Fix D: wrapper-dominance verdict ---


def test_wrapper_dominance_lane_b_gs_chunk_n_returns_explanation():
    """Fix D: gs_chunk_N > 50% of root on Lane B → explanation populated."""
    agg = PerPpAggregate(
        pp=8192,
        root_inclusive_us_median=1000.0,
        by_span_exclusive_us={"gs_chunk_N": 900.0},
    )
    agg.by_span = {"gs_chunk_N": 0.90}
    explanation = wrapper_dominated_verdict_explanation(8192, "B", agg)
    assert explanation is not None
    assert "gs_chunk_N" in explanation
    assert "P5h+1" in explanation


def test_wrapper_dominance_lane_a_first_token_sampling_returns_explanation():
    """Fix D: first_token_sampling > 50% on Lane A → MLX lazy materialization
    wrapper explanation."""
    agg = PerPpAggregate(
        pp=128,
        root_inclusive_us_median=1000.0,
        by_span_exclusive_us={"first_token_sampling": 968.0},
    )
    agg.by_span = {"first_token_sampling": 0.968}
    explanation = wrapper_dominated_verdict_explanation(128, "A", agg)
    assert explanation is not None
    assert "first_token_sampling" in explanation
    assert "lazy materialization" in explanation


def test_wrapper_dominance_below_threshold_no_explanation():
    agg = PerPpAggregate(
        pp=128,
        root_inclusive_us_median=1000.0,
        by_span_exclusive_us={"first_token_sampling": 400.0},
    )
    agg.by_span = {"first_token_sampling": 0.40}
    explanation = wrapper_dominated_verdict_explanation(128, "A", agg)
    assert explanation is None


def test_feasibility_verdict_wrapper_dominance_returns_data_insufficient():
    """Fix D: when gs_chunk_N dominates Lane B PP, verdict = data_insufficient
    even if candidates exist (no actionable target inside the wrapper)."""
    agg = PerPpAggregate(
        pp=8192,
        root_inclusive_us_median=1000.0,
        by_span_exclusive_us={"gs_chunk_N": 900.0},
    )
    agg.by_span = {"gs_chunk_N": 0.90}
    cands = [_candidate("gs_chunk_N", 8192, 0.90, scope_gate=False)]
    v = feasibility_verdict([], cands, pp=8192, per_pp_agg=agg, observed_lane="B")
    assert v == FeasibilityVerdict.DATA_INSUFFICIENT


def test_feasibility_verdict_no_wrapper_uses_existing_4_tier_logic():
    """Without wrapper dominance, the 4-tier logic still applies."""
    agg = PerPpAggregate(
        pp=128,
        root_inclusive_us_median=1000.0,
        by_span_exclusive_us={"op_a": 300.0, "op_b": 300.0},
    )
    agg.by_span = {"op_a": 0.30, "op_b": 0.30}
    cands = [
        _candidate("op_a", 128, 0.30, scope_gate=False),
        _candidate("op_b", 128, 0.30, scope_gate=False),
    ]
    v = feasibility_verdict(cands, [], pp=128, per_pp_agg=agg, observed_lane="A")
    # Same as test_feasibility_verdict_yes_when_op_only_meets_target.
    assert v == FeasibilityVerdict.YES


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
