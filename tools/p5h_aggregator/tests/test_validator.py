import pytest
from tools.p5h_aggregator.schema_validator import (
    parse_line,
    validate_request,
    group_by_request,
)

LINE_OK = (
    "  2026-05-21T03:00:00Z  INFO ironmlx::core::p5h: "
    "[p5h-profile] request_id=abc routing_path=scheduler prompt_tokens=128 "
    "seq=128 layer_idx=-1 span_id=1 parent_span_id=null "
    "span_name=server_request_recv_to_first_content_sse_write parent_span=null "
    "start_ns=1000 end_ns=2000 mode=off span_kind=tree"
)

def test_parse_line_root():
    s = parse_line(LINE_OK)
    assert s is not None
    assert s.span_name == "server_request_recv_to_first_content_sse_write"
    assert s.parent_span_id is None
    assert s.span_kind == "tree"

def test_validate_missing_required_fails():
    s = parse_line(LINE_OK)
    rep = validate_request([s])
    assert not rep.ok
    assert any("missing required" in f for f in rep.failures)

def test_duplicate_span_id_fails():
    a = parse_line(LINE_OK)
    b = parse_line(LINE_OK)  # same span_id=1
    rep = validate_request([a, b])
    assert not rep.ok
    assert any("duplicate" in f for f in rep.failures)

# --- Hard-path fixtures per Codex plan review v3 P2 #4 ---

def _build_line(
    *,
    request_id="abc",
    routing_path="scheduler",
    prompt_tokens=128,
    span_id,
    parent_span_id="null",
    span_name,
    parent_span="null",
    start_ns=1_000_000,
    end_ns=2_000_000,
    mode="off",
    span_kind="tree",
):
    """Build a synthetic [p5h-profile] log line with field overrides."""
    return (
        f"  2026-05-21T03:00:00Z  INFO ironmlx::core::p5h: "
        f"[p5h-profile] request_id={request_id} routing_path={routing_path} "
        f"prompt_tokens={prompt_tokens} seq=128 layer_idx=-1 "
        f"span_id={span_id} parent_span_id={parent_span_id} "
        f"span_name={span_name} parent_span={parent_span} "
        f"start_ns={start_ns} end_ns={end_ns} mode={mode} span_kind={span_kind}"
    )

def _lane_a_pass_fixture() -> list:
    """Minimal Lane-A request: root + all 6 required tree spans + 1 required diagnostic."""
    spans = []
    # Root: contains all children in [0, 100_000_000]
    spans.append(parse_line(_build_line(
        span_id=1, parent_span_id="null",
        span_name="server_request_recv_to_first_content_sse_write",
        parent_span="null",
        start_ns=0, end_ns=100_000_000,
    )))
    # Required tree children
    for sid, name in enumerate([
        "http_parse_render_tokenize",
        "scheduler_admission",
        "model_prefill_forward",
        "first_token_sampling",
        "detok_format_first_content_chunk",
    ], start=2):
        spans.append(parse_line(_build_line(
            span_id=sid, parent_span_id="1",
            span_name=name, parent_span="server_request_recv_to_first_content_sse_write",
            start_ns=1_000 * sid, end_ns=1_000 * sid + 500,
        )))
    # Required diagnostic (under root span_id=1, but span_kind=diagnostic)
    spans.append(parse_line(_build_line(
        span_id=100, parent_span_id="1",
        span_name="sse_write_role_chunk_diagnostic",
        parent_span="server_request_recv_to_first_content_sse_write",
        start_ns=10_000, end_ns=10_500, span_kind="diagnostic",
    )))
    return spans

def test_lane_a_full_fixture_passes():
    """Per Codex plan review v3 P2 #4: a well-formed Lane-A request must
    PASS all structural checks, including the diagnostic presence subset."""
    spans = _lane_a_pass_fixture()
    rep = validate_request(spans)
    assert rep.ok, f"unexpected failures: {rep.failures}"

def test_lane_a_missing_diagnostic_fails():
    """Drop the diagnostic span; presence check on LANE_A_REQUIRED_DIAGNOSTIC must fail."""
    spans = [s for s in _lane_a_pass_fixture() if s.span_kind != "diagnostic"]
    rep = validate_request(spans)
    assert not rep.ok
    assert any("missing required diagnostic spans" in f for f in rep.failures), rep.failures

def test_diagnostic_parent_not_root_fails():
    """Diagnostic span with parent_span_id != root.span_id and not None must fail
    (per § 2.5a Diagnostic span checks)."""
    spans = _lane_a_pass_fixture()
    # Mutate diagnostic span's parent_span_id to a non-root id (id=2 is http_parse).
    for i, s in enumerate(spans):
        if s.span_kind == "diagnostic":
            spans[i] = parse_line(_build_line(
                span_id=100, parent_span_id="2",
                span_name="sse_write_role_chunk_diagnostic",
                parent_span="http_parse_render_tokenize",  # not root
                start_ns=10_000, end_ns=10_500, span_kind="diagnostic",
            ))
            break
    rep = validate_request(spans)
    assert not rep.ok
    assert any("parent_span_id" in f and "must be null or root" in f for f in rep.failures), rep.failures

def test_aborted_root_only_skips_required_set_and_passes():
    """Per Codex plan review v12 P2 #6 + v13 P2 #5: a request whose root carries
    `mode="aborted"` (closed via `RootSpanHandle::close_at_aborted` on a
    pre-first-content terminal path) intentionally lacks
    `detok_format_first_content_chunk` and other downstream tree spans. The
    validator MUST skip the per-lane required-set check + pre_content_decode_steps
    gate for such requests, and report.aborted MUST be True."""
    # Root only + http_parse + scheduler_admission (no first_token_sampling,
    # no detok, no diagnostic role chunk). Mode="aborted" on root marks it.
    spans = [
        parse_line(_build_line(
            span_id=1, parent_span_id="null",
            span_name="server_request_recv_to_first_content_sse_write",
            parent_span="null",
            start_ns=0, end_ns=2_000_000, mode="aborted",
        )),
        parse_line(_build_line(
            span_id=2, parent_span_id="1",
            span_name="http_parse_render_tokenize",
            parent_span="server_request_recv_to_first_content_sse_write",
            start_ns=1_000, end_ns=1_500,
        )),
        parse_line(_build_line(
            span_id=3, parent_span_id="1",
            span_name="scheduler_admission",
            parent_span="server_request_recv_to_first_content_sse_write",
            start_ns=2_000, end_ns=2_500,
        )),
    ]
    rep = validate_request(spans)
    assert rep.aborted, "report.aborted must be True when root.mode=aborted"
    assert rep.ok, f"aborted request must skip required-set check, got failures: {rep.failures}"

def test_aborted_request_with_pre_content_decode_steps_passes():
    """Per Codex plan review v13 P1 #1: aborted requests may emit
    `pre_content_decode_steps` before the closure-scope guard fires
    (Lane-B per-iteration loop opened the span, then stream.next_token Err).
    The validator MUST skip the `pre_content_decode_steps count > 0` gate
    for aborted requests."""
    spans = [
        parse_line(_build_line(
            span_id=1, parent_span_id="null",
            span_name="server_request_recv_to_first_content_sse_write",
            parent_span="null",
            start_ns=0, end_ns=2_000_000, mode="aborted",
        )),
        parse_line(_build_line(
            span_id=2, parent_span_id="1",
            span_name="pre_content_decode_steps",
            parent_span="server_request_recv_to_first_content_sse_write",
            start_ns=500_000, end_ns=600_000,
        )),
    ]
    rep = validate_request(spans)
    assert rep.aborted
    assert rep.ok, f"aborted request with pre_content_decode_steps must pass, got: {rep.failures}"

def test_non_aborted_root_does_not_set_report_aborted():
    """report.aborted is True ONLY when root.mode == 'aborted'."""
    spans = _lane_a_pass_fixture()  # root.mode = "off"
    rep = validate_request(spans)
    assert not rep.aborted

# --- Lane-B chunk-count fixtures (per Codex plan review v21 P1) ---

def _lane_b_pass_fixture(*, prompt_tokens=4096, chunk_size=2048) -> list:
    """Minimal well-formed Lane-B request: root + all 9 LANE_B_REQUIRED_TREE
    spans + the expected `ceil(prompt_tokens / chunk_size)` count of
    gs_chunk_N children under gs_stream_init_and_chunk_loop. Per v21 P1.

    Per self-review of v21 fix: ALL parent-child timing windows respect
    spec § 2.5a interval containment: `parent.start_ns ≤ child.start_ns ≤
    child.end_ns ≤ parent.end_ns`. Earlier draft had children at start_ns
    10_400+ under a parent windowed [3_000, 3_500] — violated containment.
    """
    spans = []
    # Root span — wide window [0, 100_000_000] containing every other span.
    spans.append(parse_line(_build_line(
        request_id="lb-req",
        routing_path="gs_chunked",
        prompt_tokens=prompt_tokens,
        span_id=1, parent_span_id="null",
        span_name="server_request_recv_to_first_content_sse_write",
        parent_span="null",
        start_ns=0, end_ns=100_000_000,
    )))
    # http_parse_render_tokenize: tight window early in the root.
    spans.append(parse_line(_build_line(
        request_id="lb-req",
        routing_path="gs_chunked",
        prompt_tokens=prompt_tokens,
        span_id=2, parent_span_id="1",
        span_name="http_parse_render_tokenize",
        parent_span="server_request_recv_to_first_content_sse_write",
        start_ns=1_000, end_ns=2_000,
    )))
    # gs_stream_init_and_chunk_loop: wide window [3_000, 49_999] to contain
    # all five children below (gs_kv_cache_alloc, gs_first_token_sample_dispatch,
    # and `expected_chunks` × gs_chunk_N).
    spans.append(parse_line(_build_line(
        request_id="lb-req",
        routing_path="gs_chunked",
        prompt_tokens=prompt_tokens,
        span_id=3, parent_span_id="1",
        span_name="gs_stream_init_and_chunk_loop",
        parent_span="server_request_recv_to_first_content_sse_write",
        start_ns=3_000, end_ns=49_999,
    )))
    # gs_kv_cache_alloc — earliest child of gs_stream_init, inside [3_000, 49_999].
    spans.append(parse_line(_build_line(
        request_id="lb-req",
        routing_path="gs_chunked",
        prompt_tokens=prompt_tokens,
        span_id=4, parent_span_id="3",
        span_name="gs_kv_cache_alloc",
        parent_span="gs_stream_init_and_chunk_loop",
        start_ns=3_100, end_ns=3_200,
    )))
    # gs_chunk_N: exactly ceil(prompt_tokens / chunk_size) instances,
    # serialized inside gs_stream_init's window.
    expected_chunks = (prompt_tokens + chunk_size - 1) // chunk_size
    for i in range(expected_chunks):
        sid = 100 + i
        chunk_start = 10_000 + 1_000 * i
        spans.append(parse_line(_build_line(
            request_id="lb-req",
            routing_path="gs_chunked",
            prompt_tokens=prompt_tokens,
            span_id=sid, parent_span_id="3",
            span_name="gs_chunk_N",
            parent_span="gs_stream_init_and_chunk_loop",
            start_ns=chunk_start, end_ns=chunk_start + 500,
        )))
    # gs_first_token_sample_dispatch — last child of gs_stream_init.
    spans.append(parse_line(_build_line(
        request_id="lb-req",
        routing_path="gs_chunked",
        prompt_tokens=prompt_tokens,
        span_id=5, parent_span_id="3",
        span_name="gs_first_token_sample_dispatch",
        parent_span="gs_stream_init_and_chunk_loop",
        start_ns=40_000, end_ns=40_100,
    )))
    # Three post-prefill root children: sse_write_role_chunk +
    # gs_first_token_materialize_and_predispatch + detok_format_first_content_chunk.
    # Each at distinct start_ns to keep sibling ordering deterministic.
    for sid, name, start in [
        (200, "sse_write_role_chunk", 60_000),
        (201, "gs_first_token_materialize_and_predispatch", 70_000),
        (202, "detok_format_first_content_chunk", 80_000),
    ]:
        spans.append(parse_line(_build_line(
            request_id="lb-req",
            routing_path="gs_chunked",
            prompt_tokens=prompt_tokens,
            span_id=sid, parent_span_id="1",
            span_name=name,
            parent_span="server_request_recv_to_first_content_sse_write",
            start_ns=start, end_ns=start + 1_000,
        )))
    return spans

def test_lane_b_full_fixture_passes():
    """Per Codex v21 P1: a well-formed Lane-B request with all required
    children + expected gs_chunk_N count must PASS."""
    spans = _lane_b_pass_fixture(prompt_tokens=4096, chunk_size=2048)  # expect 2 chunks
    rep = validate_request(spans)
    assert rep.ok, f"unexpected failures: {rep.failures}"

def test_lane_b_diagnostic_span_fails():
    """Per Codex v23 P3: Lane-B currently has no allowed diagnostic spans.
    Accidentally emitting Lane-A's role diagnostic under gs_chunked must fail
    even when all Lane-B tree buckets are present."""
    spans = _lane_b_pass_fixture(prompt_tokens=4096, chunk_size=2048)
    spans.append(parse_line(_build_line(
        request_id="lb-req",
        routing_path="gs_chunked",
        prompt_tokens=4096,
        span_id=800, parent_span_id="1",
        span_name="sse_write_role_chunk_diagnostic",
        parent_span="server_request_recv_to_first_content_sse_write",
        start_ns=50_000, end_ns=50_100, span_kind="diagnostic",
    )))
    rep = validate_request(spans)
    assert not rep.ok
    assert any(
        "unexpected diagnostic span_name for gs_chunked" in f and "sse_write_role_chunk_diagnostic" in f
        for f in rep.failures
    ), f"expected Lane-B diagnostic rejection, got: {rep.failures}"

def test_mixed_routing_within_request_fails():
    """Per Codex v24 P3: route-specific validation must use the root route
    and reject any child carrying the opposite routing_path."""
    spans = _lane_b_pass_fixture(prompt_tokens=4096, chunk_size=2048)
    spans[1].routing_path = "scheduler"  # http_parse child disagrees with root
    rep = validate_request(spans)
    assert not rep.ok
    assert any(
        "routing_path mismatch: root has gs_chunked" in f and "http_parse_render_tokenize" in f
        for f in rep.failures
    ), f"expected mixed-routing rejection, got: {rep.failures}"

def test_lane_b_missing_gs_chunk_n_fails():
    """Per Codex v21 P1: Lane-B request that emits NO `gs_chunk_N` must
    fail validation — the chunked-prefill loop's try_ wrapper did not run."""
    spans = [s for s in _lane_b_pass_fixture() if s.span_name != "gs_chunk_N"]
    rep = validate_request(spans)
    assert not rep.ok
    assert any(
        "gs_chunk_N" in f and ("missing required tree" in f or "emitted 0" in f)
        for f in rep.failures
    ), f"expected missing-gs_chunk_N failure, got: {rep.failures}"

def test_lane_b_missing_gs_kv_cache_alloc_fails():
    """Per Codex v21 P1: Lane-B request that emits NO `gs_kv_cache_alloc`
    must fail validation."""
    spans = [s for s in _lane_b_pass_fixture() if s.span_name != "gs_kv_cache_alloc"]
    rep = validate_request(spans)
    assert not rep.ok
    assert any(
        "gs_kv_cache_alloc" in f and "missing required tree" in f
        for f in rep.failures
    ), f"expected missing-gs_kv_cache_alloc failure, got: {rep.failures}"

def test_lane_b_gs_chunk_n_count_mismatch_fails():
    """Per Codex v21 P1: if gs_chunk_N count doesn't match
    ceil(prompt_tokens / chunk_size), validation must fail with a count-mismatch
    message. Fixture builds 2 chunks for 4096 tokens at chunk_size 2048; we
    drop one to force a mismatch."""
    spans = _lane_b_pass_fixture(prompt_tokens=4096, chunk_size=2048)
    chunk_spans = [s for s in spans if s.span_name == "gs_chunk_N"]
    assert len(chunk_spans) == 2, "fixture sanity check"
    # Drop the second chunk span — now count = 1, expected = 2.
    spans = [s for s in spans if s.span_id != chunk_spans[1].span_id]
    rep = validate_request(spans)
    assert not rep.ok
    assert any(
        "gs_chunk_N count mismatch" in f and "got 1" in f and "expected 2" in f
        for f in rep.failures
    ), f"expected gs_chunk_N count-mismatch failure, got: {rep.failures}"

def test_lane_b_unexpected_deep_span_fails():
    """Per Codex v22 P1: Lane-B is top-level-only. A request that contains
    a deep Lane-A span name under the chunk loop must fail validation even if
    all required Lane-B buckets are present and coverage would otherwise look
    healthy."""
    spans = _lane_b_pass_fixture(prompt_tokens=4096, chunk_size=2048)
    first_chunk = next(s for s in spans if s.span_name == "gs_chunk_N")
    spans.append(parse_line(_build_line(
        request_id="lb-req",
        routing_path="gs_chunked",
        prompt_tokens=4096,
        span_id=900, parent_span_id=str(first_chunk.span_id),
        span_name="decoder_layer_N",
        parent_span="gs_chunk_N",
        start_ns=first_chunk.start_ns + 10,
        end_ns=first_chunk.start_ns + 20,
    )))
    rep = validate_request(spans)
    assert not rep.ok
    assert any(
        "unexpected Lane-B tree spans" in f and "decoder_layer_N" in f
        for f in rep.failures
    ), f"expected unexpected-deep-span failure, got: {rep.failures}"

def test_join_orphan_aggregator_hard_fail(tmp_path):
    """Per Codex plan review v3 P2 #4 + § 2.5a Join key: aggregator MUST exit
    non-zero when server log has a request_id absent from iron-bench CSV.
    This test invokes the aggregator entry point with a curated input.
    """
    import subprocess
    import sys
    # Server log with a request_id "abc"
    server_log = tmp_path / "server.log"
    server_log.write_text("\n".join(
        _build_line(span_id=sid, parent_span_id=("null" if sid == 1 else "1"),
                    span_name=name,
                    parent_span=("null" if sid == 1 else "server_request_recv_to_first_content_sse_write"),
                    start_ns=1_000 * sid, end_ns=1_000 * sid + 500)
        for sid, name in [
            (1, "server_request_recv_to_first_content_sse_write"),
            (2, "http_parse_render_tokenize"),
        ]
    ) + "\n")
    # Bench CSV with a DIFFERENT request_id "xyz" — server "abc" is orphan
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text("request_id,pp_target\nxyz,128\n")
    out = tmp_path / "out.csv"
    result = subprocess.run(
        [sys.executable, "-m", "tools.p5h_aggregator.aggregator",
         "--server-log", str(server_log),
         "--bench-csv", str(bench_csv),
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 4, f"expected exit 4 (JOIN HARD-FAIL), got {result.returncode}\nstderr:\n{result.stderr}"
    assert "JOIN HARD-FAIL" in result.stderr
