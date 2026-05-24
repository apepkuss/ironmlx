"""Pytest for tools/p5h_aggregator/multi_repeat.py (P5i.c Phase 0)."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from p5h_aggregator.multi_repeat import parse_attribution_csv  # noqa: E402


def test_parse_attribution_csv_root_via_empty_parent_span_id(tmp_path):
    """Root row identified by empty parent_span_id + span_kind=tree; root span_name
    is EXCLUDED from substep shares (denominator only)."""
    csv_path = tmp_path / "attr.csv"
    csv_path.write_text(
        "pp,request_id,routing_path,chunk_idx,span_name,span_kind,parent_span_id,span_id,inclusive_us,exclusive_us\n"
        "128,r1,scheduler,,server_request_recv_to_first_content_sse_write,tree,,1,1000.00,10.00\n"
        "128,r1,scheduler,,gather_qmm_gate_up,tree,1,2,250.00,250.00\n"
        "128,r1,scheduler,,fused_sdpa,tree,1,3,100.00,100.00\n"
        "128,r2,scheduler,,server_request_recv_to_first_content_sse_write,tree,,4,1100.00,10.00\n"
        "128,r2,scheduler,,gather_qmm_gate_up,tree,4,5,275.00,275.00\n"
        "128,r2,scheduler,,fused_sdpa,tree,4,6,110.00,110.00\n"
    )
    result = parse_attribution_csv(csv_path)
    assert 128 in result
    pp_data = result[128]
    # Root span name MUST be excluded
    assert "server_request_recv_to_first_content_sse_write" not in pp_data
    # gather_qmm_gate_up median = median(250, 275) = 262.5
    # root median = median(1000, 1100) = 1050
    # share = 262.5 / 1050 = 0.25
    assert abs(pp_data["gather_qmm_gate_up"] - 0.25) < 0.01
    # fused_sdpa median = 105, share = 105 / 1050 = 0.1
    assert abs(pp_data["fused_sdpa"] - 0.1) < 0.01


def test_parse_attribution_csv_skips_diagnostic_rows(tmp_path):
    csv_path = tmp_path / "attr.csv"
    csv_path.write_text(
        "pp,request_id,routing_path,chunk_idx,span_name,span_kind,parent_span_id,span_id,inclusive_us,exclusive_us\n"
        "128,r1,scheduler,,root_span,tree,,1,1000.00,10.00\n"
        "128,r1,scheduler,,gather_qmm_gate_up,tree,1,2,250.00,250.00\n"
        "128,r1,scheduler,,sse_write_role_chunk_diagnostic,diagnostic,1,3,5.00,\n"
    )
    result = parse_attribution_csv(csv_path)
    # Diagnostic row should be omitted (exclusive_us empty)
    assert "sse_write_role_chunk_diagnostic" not in result[128]
    assert "gather_qmm_gate_up" in result[128]


def test_parse_attribution_csv_sums_per_request_multi_emit(tmp_path):
    """Per-request SUM first (Fix A discipline): spans emitting multiple times
    per request (e.g. gather_qmm_gate_up across 28 MoE layers) must sum within
    each request before cross-request median."""
    csv_path = tmp_path / "attr.csv"
    # r1: gather_qmm_gate_up emitted 4× with 50 each → sum=200
    # r2: gather_qmm_gate_up emitted 4× with 60 each → sum=240
    # root: r1=1000, r2=1000 → median=1000
    # expected share = median(200, 240) / 1000 = 220 / 1000 = 0.22
    csv_path.write_text(
        "pp,request_id,routing_path,chunk_idx,span_name,span_kind,parent_span_id,span_id,inclusive_us,exclusive_us\n"
        "128,r1,scheduler,,root,tree,,1,1000.00,10.00\n"
        "128,r1,scheduler,,gather_qmm_gate_up,tree,1,2,50.00,50.00\n"
        "128,r1,scheduler,,gather_qmm_gate_up,tree,1,3,50.00,50.00\n"
        "128,r1,scheduler,,gather_qmm_gate_up,tree,1,4,50.00,50.00\n"
        "128,r1,scheduler,,gather_qmm_gate_up,tree,1,5,50.00,50.00\n"
        "128,r2,scheduler,,root,tree,,6,1000.00,10.00\n"
        "128,r2,scheduler,,gather_qmm_gate_up,tree,6,7,60.00,60.00\n"
        "128,r2,scheduler,,gather_qmm_gate_up,tree,6,8,60.00,60.00\n"
        "128,r2,scheduler,,gather_qmm_gate_up,tree,6,9,60.00,60.00\n"
        "128,r2,scheduler,,gather_qmm_gate_up,tree,6,10,60.00,60.00\n"
    )
    result = parse_attribution_csv(csv_path)
    # share = median(200, 240) / median(1000, 1000) = 220 / 1000 = 0.22
    assert abs(result[128]["gather_qmm_gate_up"] - 0.22) < 0.001


def test_parse_attribution_csv_raises_on_missing_root(tmp_path):
    csv_path = tmp_path / "attr.csv"
    # No row with empty parent_span_id → no root
    csv_path.write_text(
        "pp,request_id,routing_path,chunk_idx,span_name,span_kind,parent_span_id,span_id,inclusive_us,exclusive_us\n"
        "128,r1,scheduler,,gather_qmm_gate_up,tree,1,2,250.00,250.00\n"
    )
    import pytest

    with pytest.raises(SystemExit):
        parse_attribution_csv(csv_path)
