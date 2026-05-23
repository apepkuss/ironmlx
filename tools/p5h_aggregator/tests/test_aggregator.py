"""Tests for tools.p5h_aggregator.aggregator T5 extension functions.

Covers (per spec § 2.5a pseudocode):
* compute_exclusive — per-span exclusive_us + sum-to-root invariant + negative
  exclusive bound.
* synthesize_residual_leaves — only emits when residual > 1us.
* coverage_pct — 1 - Σ residuals / root.inclusive_us formula.
* build_attribution end-to-end on a Lane-A and Lane-B fixture.
* Per-PP CSV emission + coverage gate exit code.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from tools.p5h_aggregator.aggregator import (
    COVERAGE_GATE_PCT,
    ResidualLeaf,
    build_attribution,
    compute_exclusive,
    coverage_pct,
    diagnostic_columns,
    synthesize_residual_leaves,
    write_attribution_csv,
    write_summary_csv,
)
from tools.p5h_aggregator.schema_validator import parse_line


def _build_line(
    *,
    request_id="rid",
    routing_path="scheduler",
    prompt_tokens=128,
    span_id,
    parent_span_id="null",
    span_name,
    parent_span="null",
    start_ns=0,
    end_ns=1_000,
    mode="off",
    span_kind="tree",
    chunk_idx="null",
):
    return (
        "  2026-05-22T03:00:00Z  INFO ironmlx::core::p5h: "
        f"[p5h-profile] request_id={request_id} routing_path={routing_path} "
        f"prompt_tokens={prompt_tokens} seq=128 layer_idx=-1 chunk_idx={chunk_idx} "
        f"span_id={span_id} parent_span_id={parent_span_id} "
        f"span_name={span_name} parent_span={parent_span} "
        f"start_ns={start_ns} end_ns={end_ns} mode={mode} span_kind={span_kind}"
    )


def _lane_a_full_fixture(rid="rid", end_root=100_000) -> list:
    """Lane-A fixture with full required tree + one diagnostic.

    All required spans windowed strictly inside the root interval. Total of
    child inclusive_us = sum across 5 required tree children — leaves a
    residual large enough to test synthesize_residual_leaves.
    """
    spans = []
    # Root: [0, end_root]
    spans.append(
        parse_line(
            _build_line(
                request_id=rid,
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=end_root,
            )
        )
    )
    # 6 required tree children at distinct windows, each 500ns wide.
    # P5h+1 T1: `first_token_sampling` is now split into two siblings
    # (`_prepare` + `_materialize_and_sample`).
    for sid, name in enumerate(
        [
            "http_parse_render_tokenize",
            "scheduler_admission",
            "model_prefill_forward",
            "first_token_sampling_prepare",
            "first_token_sampling_materialize_and_sample",
            "detok_format_first_content_chunk",
        ],
        start=2,
    ):
        start = 1_000 * sid
        spans.append(
            parse_line(
                _build_line(
                    request_id=rid,
                    span_id=sid,
                    parent_span_id="1",
                    span_name=name,
                    parent_span="server_request_recv_to_first_content_sse_write",
                    start_ns=start,
                    end_ns=start + 500,
                )
            )
        )
    # Required diagnostic
    spans.append(
        parse_line(
            _build_line(
                request_id=rid,
                span_id=100,
                parent_span_id="1",
                span_name="sse_write_role_chunk_diagnostic",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=10_000,
                end_ns=10_500,
                span_kind="diagnostic",
            )
        )
    )
    return spans


# --- compute_exclusive ---


def test_compute_exclusive_single_root_no_children():
    spans = [
        parse_line(
            _build_line(
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=5_000,
            )
        )
    ]
    excl = compute_exclusive(spans)
    # 5_000 ns = 5.0 us
    assert excl == {1: 5.0}


def test_compute_exclusive_parent_minus_children():
    spans = _lane_a_full_fixture()
    excl = compute_exclusive(spans)
    # Root has 6 children with inclusive_us=0.5 each (500ns / 1000). Root
    # inclusive = 100us; exclusive = 100 - 6*0.5 = 97.0us.
    assert excl[1] == pytest.approx(100.0 - 6 * 0.5, abs=0.01)
    # Each child has no children, exclusive = inclusive.
    for sid in (2, 3, 4, 5, 6, 7):
        assert excl[sid] == pytest.approx(0.5, abs=0.01)
    # Diagnostic span (id=100) is NOT in the exclusive tree.
    assert 100 not in excl


def test_compute_exclusive_sum_to_root_invariant():
    spans = _lane_a_full_fixture()
    excl = compute_exclusive(spans)
    tree_sum = sum(excl.values())
    root_inclusive = next(s.inclusive_us for s in spans if s.span_id == 1)
    assert abs(tree_sum - root_inclusive) < 1.0


def test_compute_exclusive_negative_exclusive_raises():
    """Child windowed beyond parent triggers exclusive < -1us — raises."""
    spans = _lane_a_full_fixture()
    # Force a child span that's wider than the root: span_id=2 made 200us wide
    # (parent has 100us total). To trigger the negative-exclusive assertion we
    # need the SUM of child inclusive_us to exceed root.inclusive_us by > 1us.
    # Patch span_id=2 inclusive from 0.5us to 200us by re-parsing.
    spans[1] = parse_line(
        _build_line(
            span_id=2,
            parent_span_id="1",
            span_name="http_parse_render_tokenize",
            parent_span="server_request_recv_to_first_content_sse_write",
            start_ns=0,
            end_ns=200_000,  # 200us
        )
    )
    with pytest.raises(AssertionError, match="negative exclusive"):
        compute_exclusive(spans)


# --- synthesize_residual_leaves ---


def test_synthesize_residual_leaves_emits_unattributed_when_gap_exceeds_threshold():
    spans = _lane_a_full_fixture()
    residuals = synthesize_residual_leaves(spans)
    # Root: 100us - 6*0.5us = 97.0us residual under root.
    root_residuals = [r for r in residuals if r.parent_span_id == 1]
    assert len(root_residuals) == 1
    assert (
        root_residuals[0].span_name
        == "unattributed_server_request_recv_to_first_content_sse_write"
    )
    assert root_residuals[0].inclusive_us == pytest.approx(97.0, abs=0.01)
    assert root_residuals[0].span_kind == "synthesized"


def test_synthesize_residual_leaves_omits_when_residual_under_threshold():
    """Spans whose children exactly account for parent inclusive: no residual."""
    # Root [0, 10000], single child [0, 10000].
    spans = [
        parse_line(
            _build_line(
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=10_000,
            )
        ),
        parse_line(
            _build_line(
                span_id=2,
                parent_span_id="1",
                span_name="http_parse_render_tokenize",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=0,
                end_ns=10_000,
            )
        ),
    ]
    residuals = synthesize_residual_leaves(spans)
    assert residuals == []


def test_synthesize_residual_leaves_skips_leaf_spans():
    """Leaves (no children) never produce residual rows."""
    spans = _lane_a_full_fixture()
    residuals = synthesize_residual_leaves(spans)
    # No residual should have parent_span_id in {2,3,4,5,6} (those are leaves).
    leaf_parent_ids = {2, 3, 4, 5, 6}
    for r in residuals:
        assert r.parent_span_id not in leaf_parent_ids


# --- coverage_pct ---


def test_coverage_pct_zero_residuals_returns_one():
    spans = [
        parse_line(
            _build_line(
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=1_000,
            )
        )
    ]
    root = spans[0]
    assert coverage_pct(root, []) == 1.0


def test_coverage_pct_residual_half_returns_half():
    """Residual = 50% of root inclusive."""
    spans = [
        parse_line(
            _build_line(
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=10_000,  # 10us
            )
        )
    ]
    residual = ResidualLeaf(
        span_name="unattributed_server_request_recv_to_first_content_sse_write",
        inclusive_us=5.0,  # 50% of 10us
        parent_span_id=1,
        parent_span_name="server_request_recv_to_first_content_sse_write",
        chunk_idx=None,
    )
    assert coverage_pct(spans[0], [residual]) == pytest.approx(0.5)


def test_coverage_pct_lane_a_full_fixture_below_gate():
    """Lane-A fixture has 97.0% residual → coverage 3.0%, far below 95% gate."""
    spans = _lane_a_full_fixture()
    root = next(s for s in spans if s.span_id == 1)
    residuals = synthesize_residual_leaves(spans)
    cov = coverage_pct(root, residuals)
    assert cov < COVERAGE_GATE_PCT
    assert cov == pytest.approx(0.030, abs=0.01)


# --- diagnostic_columns ---


def test_diagnostic_columns_excludes_tree_spans():
    spans = _lane_a_full_fixture()
    cols = diagnostic_columns(spans)
    assert "sse_write_role_chunk_diagnostic_us" in cols
    # 500ns = 0.5us
    assert cols["sse_write_role_chunk_diagnostic_us"] == pytest.approx(0.5)
    assert "http_parse_render_tokenize_us" not in cols


# --- build_attribution ---


def test_build_attribution_returns_full_request_state():
    spans = _lane_a_full_fixture()
    attr = build_attribution(spans, pp="128")
    assert attr.request_id == "rid"
    assert attr.pp == "128"
    assert attr.routing_path == "scheduler"
    assert attr.root.span_id == 1
    assert len(attr.tree_spans) == 7  # root + 6 children
    assert len(attr.diagnostics) == 1
    # Residual under root
    assert any(r.parent_span_id == 1 for r in attr.residuals)


# --- CSV emission ---


def test_write_attribution_csv_emits_rows_per_kind(tmp_path: Path):
    spans = _lane_a_full_fixture()
    attr = build_attribution(spans, pp="128")
    out = tmp_path / "attribution.csv"
    write_attribution_csv([attr], out)

    rows = list(csv.DictReader(out.open()))
    kinds = {r["span_kind"] for r in rows}
    assert "tree" in kinds
    assert "synthesized" in kinds
    assert "diagnostic" in kinds

    # Diagnostic row carries empty exclusive_us per § 2.5a.
    diag_rows = [r for r in rows if r["span_kind"] == "diagnostic"]
    assert diag_rows
    assert all(r["exclusive_us"] == "" for r in diag_rows)

    # Synthesized row: exclusive == inclusive (residual leaf, no children).
    synth_rows = [r for r in rows if r["span_kind"] == "synthesized"]
    assert synth_rows
    for r in synth_rows:
        assert r["exclusive_us"] == r["inclusive_us"]
        assert r["span_name"].startswith("unattributed_")


def test_write_summary_csv_top3_ordering(tmp_path: Path):
    spans = _lane_a_full_fixture()
    attr = build_attribution(spans, pp="128")
    out = tmp_path / "summary.csv"
    median_cov = write_summary_csv([attr], out)
    assert "128" in median_cov
    # Below 95% gate (fixture has 97.0% residual).
    assert median_cov["128"] < COVERAGE_GATE_PCT

    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["pp"] == "128"
    assert int(row["request_count"]) == 1
    # Top-1 in this fixture should be the residual unattributed root (97.0%
    # share) since every other child is 0.5%.
    assert row["top1_span_name"].startswith("unattributed_")
    assert float(row["top1_share"]) > 0.9


def test_top3_bottlenecks_sums_multi_emit_span_per_request(tmp_path: Path):
    """Fix A: multi-emit spans (e.g. gs_chunk_N with N records/request) must
    be summed per-request BEFORE median across requests, otherwise the per-
    record median drastically under-reports per-request cost.

    Fixture: root [0, 100us]. 2 children of same span_name 'multi_emit' each
    20us → per-request total exclusive = 40us → share = 0.40.
    WRONG (per-record median): single child = 20us → share = 0.20.
    """
    rid = "rid_multi"
    spans = [
        parse_line(
            _build_line(
                request_id=rid,
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=100_000,
            )
        ),
        # Two children of same name, each 20us = 40us total exclusive.
        parse_line(
            _build_line(
                request_id=rid,
                span_id=10,
                parent_span_id="1",
                span_name="multi_emit",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=10_000,
                end_ns=30_000,
            )
        ),
        parse_line(
            _build_line(
                request_id=rid,
                span_id=11,
                parent_span_id="1",
                span_name="multi_emit",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=40_000,
                end_ns=60_000,
            )
        ),
    ]
    attr = build_attribution(spans, pp="2048")
    out = tmp_path / "summary.csv"
    write_summary_csv([attr], out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    row = rows[0]
    # Top-1 must be 'unattributed_root' (~0.6 share) — but 'multi_emit' must
    # report PER-REQUEST total share = 0.40 (= 40us / 100us), NOT 0.20.
    # Walk all top columns to find 'multi_emit'.
    multi_share = None
    for i in (1, 2, 3):
        if row[f"top{i}_span_name"] == "multi_emit":
            multi_share = float(row[f"top{i}_share"])
    assert multi_share is not None, (
        "multi_emit span should appear in top-3; "
        f"row keys: {dict((k, row[k]) for k in row if 'top' in k)}"
    )
    assert multi_share == pytest.approx(0.40, abs=0.01), (
        f"per-request sum aggregation expected share=0.40, got {multi_share}"
    )


# --- end-to-end aggregator CLI: coverage gate exit ---


def _write_log(path: Path, spans: list) -> None:
    """Write spans back as [p5h-profile] log lines via _build_line clone."""
    lines = []
    for s in spans:
        # Re-emit the span as a log line; parse_line is forgiving with the
        # leading prefix text.
        lines.append(
            _build_line(
                request_id=s.request_id,
                routing_path=s.routing_path,
                prompt_tokens=s.prompt_tokens,
                span_id=s.span_id,
                parent_span_id=(
                    "null" if s.parent_span_id is None else str(s.parent_span_id)
                ),
                span_name=s.span_name,
                parent_span=("null" if s.parent_span is None else s.parent_span),
                start_ns=s.start_ns,
                end_ns=s.end_ns,
                mode=s.mode,
                span_kind=s.span_kind,
                chunk_idx=("null" if s.chunk_idx is None else str(s.chunk_idx)),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def test_aggregator_cli_coverage_gate_failure_exits_7(tmp_path: Path):
    """Lane-A fixture has ~97.0% residual under root → coverage_pct ~3.0% →
    aggregator must exit 7 (COVERAGE GATE FAILURE)."""
    server_log = tmp_path / "server.log"
    bench_csv = tmp_path / "bench.csv"
    out = tmp_path / "attribution.csv"
    summary = tmp_path / "summary.csv"

    spans = _lane_a_full_fixture(rid="abc")
    _write_log(server_log, spans)
    bench_csv.write_text("request_id,pp_target\nabc,128\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.p5h_aggregator.aggregator",
            "--server-log",
            str(server_log),
            "--bench-csv",
            str(bench_csv),
            "--out",
            str(out),
            "--summary-out",
            str(summary),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7, (
        f"expected exit 7 (COVERAGE GATE FAILURE), got {result.returncode}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "COVERAGE GATE FAILURE" in result.stderr
    assert out.exists()
    assert summary.exists()


def test_aggregator_cli_coverage_gate_pass_exits_0(tmp_path: Path):
    """Fixture where children exactly account for root + 5us slack (5% residual)
    → coverage_pct ~95% → aggregator exits 0."""
    server_log = tmp_path / "server.log"
    bench_csv = tmp_path / "bench.csv"
    out = tmp_path / "attribution.csv"
    summary = tmp_path / "summary.csv"

    # Build fixture: root [0, 100_000ns]. Children inclusive sum = 95_000ns
    # → 5_000ns residual = 5%. Coverage = 95%.
    rid = "rid_pass"
    spans = [
        parse_line(
            _build_line(
                request_id=rid,
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=100_000,
            )
        )
    ]
    # 6 required children. P5h+1 T1: `first_token_sampling` is split into
    # `_prepare` + `_materialize_and_sample` siblings; the original 19_000ns
    # budget for `first_token_sampling` is split evenly (9_500ns each) so
    # the children inclusive_us sum is unchanged at 95_000ns → coverage 95%.
    widths = {
        "http_parse_render_tokenize": 19_000,
        "scheduler_admission": 19_000,
        "model_prefill_forward": 19_000,
        "first_token_sampling_prepare": 9_500,
        "first_token_sampling_materialize_and_sample": 9_500,
        "detok_format_first_content_chunk": 19_000,
    }
    cursor = 1_000
    for sid, name in enumerate(
        [
            "http_parse_render_tokenize",
            "scheduler_admission",
            "model_prefill_forward",
            "first_token_sampling_prepare",
            "first_token_sampling_materialize_and_sample",
            "detok_format_first_content_chunk",
        ],
        start=2,
    ):
        width = widths[name]
        spans.append(
            parse_line(
                _build_line(
                    request_id=rid,
                    span_id=sid,
                    parent_span_id="1",
                    span_name=name,
                    parent_span="server_request_recv_to_first_content_sse_write",
                    start_ns=cursor,
                    end_ns=cursor + width,
                )
            )
        )
        cursor += width + 500
    # Required diagnostic
    spans.append(
        parse_line(
            _build_line(
                request_id=rid,
                span_id=100,
                parent_span_id="1",
                span_name="sse_write_role_chunk_diagnostic",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=500,
                end_ns=900,
                span_kind="diagnostic",
            )
        )
    )

    _write_log(server_log, spans)
    bench_csv.write_text(f"request_id,pp_target\n{rid},128\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.p5h_aggregator.aggregator",
            "--server-log",
            str(server_log),
            "--bench-csv",
            str(bench_csv),
            "--out",
            str(out),
            "--summary-out",
            str(summary),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"expected exit 0 (gate pass), got {result.returncode}\n"
        f"stderr:\n{result.stderr}"
    )
    assert out.exists()
    assert summary.exists()


def test_aggregator_cli_aborted_request_skipped(tmp_path: Path):
    """Aborted request (root.mode=aborted) must be skipped per § 7.1."""
    server_log = tmp_path / "server.log"
    bench_csv = tmp_path / "bench.csv"
    out = tmp_path / "attribution.csv"
    summary = tmp_path / "summary.csv"

    rid = "aborted_only"
    spans = [
        parse_line(
            _build_line(
                request_id=rid,
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=10_000,
                mode="aborted",
            )
        ),
        parse_line(
            _build_line(
                request_id=rid,
                span_id=2,
                parent_span_id="1",
                span_name="http_parse_render_tokenize",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=100,
                end_ns=200,
            )
        ),
    ]
    _write_log(server_log, spans)
    bench_csv.write_text(f"request_id,pp_target\n{rid},128\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.p5h_aggregator.aggregator",
            "--server-log",
            str(server_log),
            "--bench-csv",
            str(bench_csv),
            "--out",
            str(out),
            "--summary-out",
            str(summary),
        ],
        capture_output=True,
        text=True,
    )
    # All requests aborted → exit 6 (zero non-aborted requests).
    assert result.returncode == 6, (
        f"expected exit 6 (zero non-aborted requests), got {result.returncode}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "zero non-aborted requests" in result.stderr


# --- P5h+1 T2: chunk_idx column in attribution CSV ---


def test_attribution_csv_has_chunk_idx_column_after_routing_path(tmp_path: Path):
    """write_attribution_csv inserts `chunk_idx` immediately after
    `routing_path`. No existing columns reordered or removed.
    """
    spans = _lane_a_full_fixture()
    attr = build_attribution(spans, pp="128")
    out = tmp_path / "attribution.csv"
    write_attribution_csv([attr], out)

    with out.open() as f:
        header = f.readline().strip().split(",")
    expected = [
        "pp",
        "request_id",
        "routing_path",
        "chunk_idx",
        "span_name",
        "span_kind",
        "parent_span_id",
        "span_id",
        "inclusive_us",
        "exclusive_us",
    ]
    assert header == expected, f"expected {expected}, got {header}"


def test_attribution_csv_preserves_tree_and_diagnostic_chunk_idx(tmp_path: Path):
    """Tree + diagnostic rows must round-trip the parsed `chunk_idx`. The
    Lane-A fixture's rows are all chunk_idx=null → empty string in CSV."""
    spans = _lane_a_full_fixture()
    attr = build_attribution(spans, pp="128")
    out = tmp_path / "attribution.csv"
    write_attribution_csv([attr], out)

    rows = list(csv.DictReader(out.open()))
    for r in rows:
        if r["span_kind"] in ("tree", "diagnostic"):
            # Lane-A → all chunk_idx=null → empty cell.
            assert r["chunk_idx"] == "", (
                f"Lane-A tree/diagnostic row chunk_idx must be empty, "
                f"got {r['chunk_idx']!r} on span {r['span_name']}"
            )


def test_attribution_csv_synthesized_inherits_chunk_idx(tmp_path: Path):
    """ResidualLeaf rows inherit `chunk_idx` from the parent tree span.
    Build a Lane-B-style fixture where a non-leaf span carries chunk_idx=3
    and emits a residual leaf — the CSV row for the residual must carry "3".
    """
    rid = "rid_synth"
    # Root [0, 100us]. Parent span (`gs_chunk_N` with chunk_idx=3) inside
    # root; ONE narrow child of that parent → residual under gs_chunk_N
    # inherits chunk_idx=3.
    spans = [
        parse_line(
            _build_line(
                request_id=rid,
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=100_000,
            )
        ),
        parse_line(
            _build_line(
                request_id=rid,
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=10,
                parent_span_id="1",
                span_name="gs_chunk_N",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=10_000,
                end_ns=90_000,  # 80us window
                chunk_idx="3",
            )
        ),
        # Narrow child under gs_chunk_N (5us) — residual = 80-5 = 75us,
        # synthesized as unattributed_gs_chunk_N.
        parse_line(
            _build_line(
                request_id=rid,
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=11,
                parent_span_id="10",
                span_name="decoder_layer_N",
                parent_span="gs_chunk_N",
                start_ns=11_000,
                end_ns=16_000,
                chunk_idx="3",
            )
        ),
    ]
    attr = build_attribution(spans, pp="2048")
    out = tmp_path / "attribution.csv"
    write_attribution_csv([attr], out)

    rows = list(csv.DictReader(out.open()))
    synth_rows = [r for r in rows if r["span_kind"] == "synthesized"]
    assert synth_rows, "expected at least one synthesized residual row"
    # The synthesized row under gs_chunk_N (id=10) must inherit chunk_idx="3".
    matching = [r for r in synth_rows if r["span_name"] == "unattributed_gs_chunk_N"]
    assert matching, f"expected unattributed_gs_chunk_N row, got rows: {synth_rows}"
    assert matching[0]["chunk_idx"] == "3", (
        f"synthesized chunk_idx must inherit parent (3), got {matching[0]['chunk_idx']!r}"
    )
