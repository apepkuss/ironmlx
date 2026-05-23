import re
from pathlib import Path

from tools.p5h_aggregator.schema_validator import (
    LANE_A_REQUIRED_TREE,
    LANE_B_ALLOWED_TREE,
    LANE_B_REQUIRED_TREE,
    parse_line,
    validate_chunk_ancestry,
    validate_request,
)

LINE_OK = (
    "  2026-05-21T03:00:00Z  INFO ironmlx::core::p5h: "
    "[p5h-profile] request_id=abc routing_path=scheduler prompt_tokens=128 "
    "seq=128 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
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
    chunk_idx="null",
):
    """Build a synthetic [p5h-profile] log line with field overrides."""
    return (
        f"  2026-05-21T03:00:00Z  INFO ironmlx::core::p5h: "
        f"[p5h-profile] request_id={request_id} routing_path={routing_path} "
        f"prompt_tokens={prompt_tokens} seq=128 layer_idx=-1 chunk_idx={chunk_idx} "
        f"span_id={span_id} parent_span_id={parent_span_id} "
        f"span_name={span_name} parent_span={parent_span} "
        f"start_ns={start_ns} end_ns={end_ns} mode={mode} span_kind={span_kind}"
    )


def _lane_a_pass_fixture() -> list:
    """Minimal Lane-A request: root + all required tree spans + 1 required diagnostic.

    P5h+1 T1: `first_token_sampling` was split into sibling pair
    `first_token_sampling_prepare` + `first_token_sampling_materialize_and_sample`.
    """
    spans = []
    # Root: contains all children in [0, 100_000_000]
    spans.append(
        parse_line(
            _build_line(
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=100_000_000,
            )
        )
    )
    # Required tree children
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
        spans.append(
            parse_line(
                _build_line(
                    span_id=sid,
                    parent_span_id="1",
                    span_name=name,
                    parent_span="server_request_recv_to_first_content_sse_write",
                    start_ns=1_000 * sid,
                    end_ns=1_000 * sid + 500,
                )
            )
        )
    # Required diagnostic (under root span_id=1, but span_kind=diagnostic)
    spans.append(
        parse_line(
            _build_line(
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
    assert any("missing required diagnostic spans" in f for f in rep.failures), (
        rep.failures
    )


def test_diagnostic_parent_not_root_fails():
    """Diagnostic span with parent_span_id != root.span_id and not None must fail
    (per § 2.5a Diagnostic span checks)."""
    spans = _lane_a_pass_fixture()
    # Mutate diagnostic span's parent_span_id to a non-root id (id=2 is http_parse).
    for i, s in enumerate(spans):
        if s.span_kind == "diagnostic":
            spans[i] = parse_line(
                _build_line(
                    span_id=100,
                    parent_span_id="2",
                    span_name="sse_write_role_chunk_diagnostic",
                    parent_span="http_parse_render_tokenize",  # not root
                    start_ns=10_000,
                    end_ns=10_500,
                    span_kind="diagnostic",
                )
            )
            break
    rep = validate_request(spans)
    assert not rep.ok
    assert any(
        "parent_span_id" in f and "must be null or root" in f for f in rep.failures
    ), rep.failures


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
        parse_line(
            _build_line(
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=2_000_000,
                mode="aborted",
            )
        ),
        parse_line(
            _build_line(
                span_id=2,
                parent_span_id="1",
                span_name="http_parse_render_tokenize",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=1_000,
                end_ns=1_500,
            )
        ),
        parse_line(
            _build_line(
                span_id=3,
                parent_span_id="1",
                span_name="scheduler_admission",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=2_000,
                end_ns=2_500,
            )
        ),
    ]
    rep = validate_request(spans)
    assert rep.aborted, "report.aborted must be True when root.mode=aborted"
    assert rep.ok, (
        f"aborted request must skip required-set check, got failures: {rep.failures}"
    )


def test_aborted_request_with_pre_content_decode_steps_passes():
    """Per Codex plan review v13 P1 #1: aborted requests may emit
    `pre_content_decode_steps` before the closure-scope guard fires
    (Lane-B per-iteration loop opened the span, then stream.next_token Err).
    The validator MUST skip the `pre_content_decode_steps count > 0` gate
    for aborted requests."""
    spans = [
        parse_line(
            _build_line(
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=2_000_000,
                mode="aborted",
            )
        ),
        parse_line(
            _build_line(
                span_id=2,
                parent_span_id="1",
                span_name="pre_content_decode_steps",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=500_000,
                end_ns=600_000,
            )
        ),
    ]
    rep = validate_request(spans)
    assert rep.aborted
    assert rep.ok, (
        f"aborted request with pre_content_decode_steps must pass, got: {rep.failures}"
    )


def test_non_aborted_root_does_not_set_report_aborted():
    """report.aborted is True ONLY when root.mode == 'aborted'."""
    spans = _lane_a_pass_fixture()  # root.mode = "off"
    rep = validate_request(spans)
    assert not rep.aborted


# --- Lane-B chunk-count fixtures (per Codex plan review v21 P1) ---


def _lane_b_pass_fixture(*, prompt_tokens=4096, chunk_size=2048) -> list:
    """Well-formed Lane-B request fixture for P5h+1 T2.

    Schema: root + http_parse + gs_stream_init wrapper with
    `ceil(prompt_tokens / chunk_size)` gs_chunk_N children. Each chunk
    emits the full decoder hierarchy required by LANE_B_REQUIRED_TREE
    (decoder_layer_N + input_norm + attention_path + GDN substeps +
    residual_overhead + post_attention_norm + mlp_path + MoE substeps +
    cache_state_update + slice_last_and_project_lm_head) with the
    matching chunk_idx propagated from the chunk ancestor (per
    validate_chunk_ancestry).

    All parent-child timing windows respect spec § 2.5a interval
    containment: `parent.start_ns ≤ child.start_ns ≤ child.end_ns
    ≤ parent.end_ns`.
    """
    spans = []
    # Root span — wide window [0, 100_000_000] containing every other span.
    spans.append(
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=prompt_tokens,
                span_id=1,
                parent_span_id="null",
                span_name="server_request_recv_to_first_content_sse_write",
                parent_span="null",
                start_ns=0,
                end_ns=100_000_000,
            )
        )
    )
    # http_parse_render_tokenize: tight window early in the root.
    spans.append(
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=prompt_tokens,
                span_id=2,
                parent_span_id="1",
                span_name="http_parse_render_tokenize",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=1_000,
                end_ns=2_000,
            )
        )
    )
    # gs_stream_init_and_chunk_loop: wide window [3_000, 9_000_000] to contain
    # gs_kv_cache_alloc + all gs_chunk_N + gs_first_token_sample_dispatch.
    spans.append(
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=prompt_tokens,
                span_id=3,
                parent_span_id="1",
                span_name="gs_stream_init_and_chunk_loop",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=3_000,
                end_ns=9_000_000,
            )
        )
    )
    # gs_kv_cache_alloc — earliest child of gs_stream_init. chunk_idx=null
    # (pre-loop site, outside any chunk).
    spans.append(
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=prompt_tokens,
                span_id=4,
                parent_span_id="3",
                span_name="gs_kv_cache_alloc",
                parent_span="gs_stream_init_and_chunk_loop",
                start_ns=3_100,
                end_ns=3_200,
            )
        )
    )

    expected_chunks = (prompt_tokens + chunk_size - 1) // chunk_size
    # Each chunk gets a 1_000_000ns window. Inside each window we emit
    # the full decoder hierarchy as a single layer (layer_idx=0).
    next_id = 1_000
    for i in range(expected_chunks):
        chunk_id = 100 + i
        chunk_start = 10_000 + 1_000_000 * i
        chunk_end = chunk_start + 900_000
        spans.append(
            parse_line(
                _build_line(
                    request_id="lb-req",
                    routing_path="gs_chunked",
                    prompt_tokens=prompt_tokens,
                    span_id=chunk_id,
                    parent_span_id="3",
                    span_name="gs_chunk_N",
                    parent_span="gs_stream_init_and_chunk_loop",
                    start_ns=chunk_start,
                    end_ns=chunk_end,
                    chunk_idx=str(i),
                )
            )
        )
        # decoder_layer_N under gs_chunk_N. layer_idx=0 satisfies
        # decoder-descendant layer_idx >= 0 check.
        decoder_id = next_id
        next_id += 1
        # Window for decoder_layer is large enough to hold input_norm,
        # attention_path (with substeps), residual_overhead, post_norm,
        # mlp_path (with substeps).
        decoder_start = chunk_start + 1_000
        decoder_end = chunk_start + 800_000
        spans.append(
            parse_line(
                f"  2026-05-22T03:00:00Z  INFO ironmlx::core::p5h: "
                f"[p5h-profile] request_id=lb-req routing_path=gs_chunked "
                f"prompt_tokens={prompt_tokens} seq=128 layer_idx=0 chunk_idx={i} "
                f"span_id={decoder_id} parent_span_id={chunk_id} "
                f"span_name=decoder_layer_N parent_span=gs_chunk_N "
                f"start_ns={decoder_start} end_ns={decoder_end} mode=off span_kind=tree"
            )
        )

        # input_norm — narrow window.
        sid = next_id
        next_id += 1
        spans.append(
            parse_line(
                f"  2026-05-22T03:00:00Z  INFO ironmlx::core::p5h: "
                f"[p5h-profile] request_id=lb-req routing_path=gs_chunked "
                f"prompt_tokens={prompt_tokens} seq=128 layer_idx=0 chunk_idx={i} "
                f"span_id={sid} parent_span_id={decoder_id} "
                f"span_name=input_norm parent_span=decoder_layer_N "
                f"start_ns={decoder_start + 100} end_ns={decoder_start + 200} mode=off span_kind=tree"
            )
        )

        # attention_path — wide window to hold GDN + GatedAttention substeps.
        attn_id = next_id
        next_id += 1
        attn_start = decoder_start + 300
        attn_end = decoder_start + 300_000
        spans.append(
            parse_line(
                f"  2026-05-22T03:00:00Z  INFO ironmlx::core::p5h: "
                f"[p5h-profile] request_id=lb-req routing_path=gs_chunked "
                f"prompt_tokens={prompt_tokens} seq=128 layer_idx=0 chunk_idx={i} "
                f"span_id={attn_id} parent_span_id={decoder_id} "
                f"span_name=attention_path parent_span=decoder_layer_N "
                f"start_ns={attn_start} end_ns={attn_end} mode=off span_kind=tree"
            )
        )

        # GatedAttention substeps + GDN substeps + cache_state_update +
        # slice_last_and_project_lm_head under attention_path (all narrow
        # windows, serialized).
        attn_substeps = [
            "q_gate_k_v_proj",
            "q_split_norm_reshape",
            "mrope_apply",
            "kv_mask_update",
            "fused_sdpa",
            "gate_sigmoid_mul",
            "o_proj",
            "gda_step_1a_in_proj_qkvz",
            "gda_step_1b_in_proj_ba",
            "gda_step_2a_prepend_conv_state",
            "gda_step_2b_conv1d_silu",
            "gda_step_2c_update_conv_state",
            "gda_step_3_split_reshape_per_head",
            "gda_step_4_qk_rmsnorm",
            "gda_step_5_compute_g",
            "gda_step_6_sigmoid_beta",
            "gda_step_7_kernel_and_cache_update",
            # P5h+1 T1.5 (Codex B-lite): required Lane-B child of
            # gda_step_7_kernel_and_cache_update.
            "gda_step_7_kernel_dispatch_and_materialize",
            "gda_step_8_norm_proj",
            "cache_state_update",
            "slice_last_and_project_lm_head",
        ]
        cursor = attn_start + 100
        for name in attn_substeps:
            sid = next_id
            next_id += 1
            spans.append(
                parse_line(
                    f"  2026-05-22T03:00:00Z  INFO ironmlx::core::p5h: "
                    f"[p5h-profile] request_id=lb-req routing_path=gs_chunked "
                    f"prompt_tokens={prompt_tokens} seq=128 layer_idx=0 chunk_idx={i} "
                    f"span_id={sid} parent_span_id={attn_id} "
                    f"span_name={name} parent_span=attention_path "
                    f"start_ns={cursor} end_ns={cursor + 100} mode=off span_kind=tree"
                )
            )
            cursor += 200

        # residual_overhead under decoder_layer.
        sid = next_id
        next_id += 1
        res_start = attn_end + 100
        spans.append(
            parse_line(
                f"  2026-05-22T03:00:00Z  INFO ironmlx::core::p5h: "
                f"[p5h-profile] request_id=lb-req routing_path=gs_chunked "
                f"prompt_tokens={prompt_tokens} seq=128 layer_idx=0 chunk_idx={i} "
                f"span_id={sid} parent_span_id={decoder_id} "
                f"span_name=residual_overhead parent_span=decoder_layer_N "
                f"start_ns={res_start} end_ns={res_start + 100} mode=off span_kind=tree"
            )
        )

        # post_attention_norm under decoder_layer.
        sid = next_id
        next_id += 1
        post_start = res_start + 200
        spans.append(
            parse_line(
                f"  2026-05-22T03:00:00Z  INFO ironmlx::core::p5h: "
                f"[p5h-profile] request_id=lb-req routing_path=gs_chunked "
                f"prompt_tokens={prompt_tokens} seq=128 layer_idx=0 chunk_idx={i} "
                f"span_id={sid} parent_span_id={decoder_id} "
                f"span_name=post_attention_norm parent_span=decoder_layer_N "
                f"start_ns={post_start} end_ns={post_start + 100} mode=off span_kind=tree"
            )
        )

        # mlp_path under decoder_layer with MoE substeps.
        mlp_id = next_id
        next_id += 1
        mlp_start = post_start + 200
        mlp_end = decoder_end - 100
        spans.append(
            parse_line(
                f"  2026-05-22T03:00:00Z  INFO ironmlx::core::p5h: "
                f"[p5h-profile] request_id=lb-req routing_path=gs_chunked "
                f"prompt_tokens={prompt_tokens} seq=128 layer_idx=0 chunk_idx={i} "
                f"span_id={mlp_id} parent_span_id={decoder_id} "
                f"span_name=mlp_path parent_span=decoder_layer_N "
                f"start_ns={mlp_start} end_ns={mlp_end} mode=off span_kind=tree"
            )
        )

        moe_substeps = [
            "router_logits_softmax_topk",
            "routing_sort_pack",
            "gather_qmm_gate_up",
            "swiglu_activation",
            "gather_qmm_down",
            "routing_unsort_weighted_reduce",
            "shared_expert",
            "moe_output_sum",
        ]
        cursor = mlp_start + 100
        for name in moe_substeps:
            sid = next_id
            next_id += 1
            spans.append(
                parse_line(
                    f"  2026-05-22T03:00:00Z  INFO ironmlx::core::p5h: "
                    f"[p5h-profile] request_id=lb-req routing_path=gs_chunked "
                    f"prompt_tokens={prompt_tokens} seq=128 layer_idx=0 chunk_idx={i} "
                    f"span_id={sid} parent_span_id={mlp_id} "
                    f"span_name={name} parent_span=mlp_path "
                    f"start_ns={cursor} end_ns={cursor + 100} mode=off span_kind=tree"
                )
            )
            cursor += 200

    # gs_first_token_sample_dispatch — last child of gs_stream_init.
    # chunk_idx=null (post-loop site, outside any chunk).
    spans.append(
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=prompt_tokens,
                span_id=5,
                parent_span_id="3",
                span_name="gs_first_token_sample_dispatch",
                parent_span="gs_stream_init_and_chunk_loop",
                start_ns=8_900_000,
                end_ns=8_900_100,
            )
        )
    )
    # Three post-prefill root children: sse_write_role_chunk +
    # gs_first_token_materialize_and_predispatch + detok_format_first_content_chunk.
    # All chunk_idx=null (outside any chunk).
    for sid, name, start in [
        (200, "sse_write_role_chunk", 9_500_000),
        (201, "gs_first_token_materialize_and_predispatch", 9_600_000),
        (202, "detok_format_first_content_chunk", 9_700_000),
    ]:
        spans.append(
            parse_line(
                _build_line(
                    request_id="lb-req",
                    routing_path="gs_chunked",
                    prompt_tokens=prompt_tokens,
                    span_id=sid,
                    parent_span_id="1",
                    span_name=name,
                    parent_span="server_request_recv_to_first_content_sse_write",
                    start_ns=start,
                    end_ns=start + 1_000,
                )
            )
        )
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
    spans.append(
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=800,
                parent_span_id="1",
                span_name="sse_write_role_chunk_diagnostic",
                parent_span="server_request_recv_to_first_content_sse_write",
                start_ns=50_000,
                end_ns=50_100,
                span_kind="diagnostic",
            )
        )
    )
    rep = validate_request(spans)
    assert not rep.ok
    assert any(
        "unexpected diagnostic span_name for gs_chunked" in f
        and "sse_write_role_chunk_diagnostic" in f
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
        "routing_path mismatch: root has gs_chunked" in f
        and "http_parse_render_tokenize" in f
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
        "gs_kv_cache_alloc" in f and "missing required tree" in f for f in rep.failures
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


def test_lane_b_unknown_span_name_fails():
    """P5h+1 T2: Lane-B is no longer top-level-only — the full decoder
    hierarchy is on LANE_B_ALLOWED_TREE. But a span name OUTSIDE the
    allow-list (typo, accidental Lane-A name leak, unrelated probe) must
    still fail the closed-set rejection check. This replaces the obsolete
    P5h v22 P1 top-level-only test."""
    spans = _lane_b_pass_fixture(prompt_tokens=4096, chunk_size=2048)
    first_chunk = next(s for s in spans if s.span_name == "gs_chunk_N")
    spans.append(
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=9000,
                parent_span_id=str(first_chunk.span_id),
                span_name="totally_unknown_probe",
                parent_span="gs_chunk_N",
                start_ns=first_chunk.start_ns + 10,
                end_ns=first_chunk.start_ns + 20,
                chunk_idx=(
                    "null"
                    if first_chunk.chunk_idx is None
                    else str(first_chunk.chunk_idx)
                ),
            )
        )
    )
    rep = validate_request(spans)
    assert not rep.ok
    assert any(
        "unexpected Lane-B tree spans" in f and "totally_unknown_probe" in f
        for f in rep.failures
    ), f"expected unknown-name rejection, got: {rep.failures}"


def test_join_orphan_aggregator_hard_fail(tmp_path):
    """Per Codex plan review v3 P2 #4 + § 2.5a Join key: aggregator MUST exit
    non-zero when server log has a request_id absent from iron-bench CSV.
    This test invokes the aggregator entry point with a curated input.
    """
    import subprocess
    import sys

    # Server log with a request_id "abc"
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "\n".join(
            _build_line(
                span_id=sid,
                parent_span_id=("null" if sid == 1 else "1"),
                span_name=name,
                parent_span=(
                    "null"
                    if sid == 1
                    else "server_request_recv_to_first_content_sse_write"
                ),
                start_ns=1_000 * sid,
                end_ns=1_000 * sid + 500,
            )
            for sid, name in [
                (1, "server_request_recv_to_first_content_sse_write"),
                (2, "http_parse_render_tokenize"),
            ]
        )
        + "\n"
    )
    # Bench CSV with a DIFFERENT request_id "xyz" — server "abc" is orphan
    bench_csv = tmp_path / "bench.csv"
    bench_csv.write_text("request_id,pp_target\nxyz,128\n")
    out = tmp_path / "out.csv"
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
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4, (
        f"expected exit 4 (JOIN HARD-FAIL), got {result.returncode}\nstderr:\n{result.stderr}"
    )
    assert "JOIN HARD-FAIL" in result.stderr


# --- P5h+1 T1: Lane-A `first_token_sampling` split into sibling pair ---


def test_lane_a_required_tree_excludes_first_token_sampling():
    assert "first_token_sampling" not in LANE_A_REQUIRED_TREE


def test_lane_a_required_tree_includes_first_token_sampling_prepare():
    assert "first_token_sampling_prepare" in LANE_A_REQUIRED_TREE


def test_lane_a_required_tree_includes_first_token_sampling_materialize_and_sample():
    assert "first_token_sampling_materialize_and_sample" in LANE_A_REQUIRED_TREE


# --- P5h+1 T2: chunk_idx schema + structural ancestry rule ---


def test_parse_line_chunk_idx_null_lane_a():
    """Lane-A spans carry chunk_idx=null which parses to Python None."""
    line = _build_line(
        span_id=1,
        parent_span_id="null",
        span_name="server_request_recv_to_first_content_sse_write",
        parent_span="null",
        chunk_idx="null",
    )
    s = parse_line(line)
    assert s is not None
    assert s.chunk_idx is None


def test_parse_line_chunk_idx_int_lane_b():
    """Lane-B spans under gs_chunk_N carry chunk_idx=<int> parsed to int."""
    line = _build_line(
        request_id="lb-req",
        routing_path="gs_chunked",
        prompt_tokens=4096,
        span_id=42,
        parent_span_id="100",
        span_name="decoder_layer_N",
        parent_span="gs_chunk_N",
        chunk_idx="2",
    )
    s = parse_line(line)
    assert s is not None
    assert s.chunk_idx == 2


def test_lane_b_required_tree_includes_decoder_wrappers():
    """P5h+1 T2: LANE_B_REQUIRED_TREE must include the decoder wrappers."""
    for name in (
        "decoder_layer_N",
        "input_norm",
        "attention_path",
        "residual_overhead",
        "post_attention_norm",
        "mlp_path",
    ):
        assert name in LANE_B_REQUIRED_TREE, name


def test_lane_b_required_tree_includes_attention_substeps():
    for name in (
        "q_gate_k_v_proj",
        "q_split_norm_reshape",
        "mrope_apply",
        "kv_mask_update",
        "fused_sdpa",
        "gate_sigmoid_mul",
        "o_proj",
    ):
        assert name in LANE_B_REQUIRED_TREE, name


def test_lane_b_required_tree_includes_moe_substeps():
    for name in (
        "router_logits_softmax_topk",
        "routing_sort_pack",
        "gather_qmm_gate_up",
        "swiglu_activation",
        "gather_qmm_down",
        "routing_unsort_weighted_reduce",
        "shared_expert",
        "moe_output_sum",
    ):
        assert name in LANE_B_REQUIRED_TREE, name


def test_lane_b_required_tree_includes_all_11_gda_steps():
    for name in (
        "gda_step_1a_in_proj_qkvz",
        "gda_step_1b_in_proj_ba",
        "gda_step_2a_prepend_conv_state",
        "gda_step_2b_conv1d_silu",
        "gda_step_2c_update_conv_state",
        "gda_step_3_split_reshape_per_head",
        "gda_step_4_qk_rmsnorm",
        "gda_step_5_compute_g",
        "gda_step_6_sigmoid_beta",
        "gda_step_7_kernel_and_cache_update",
        "gda_step_8_norm_proj",
    ):
        assert name in LANE_B_REQUIRED_TREE, name


def test_lane_b_required_tree_includes_gda_step_7_kernel_dispatch_and_materialize():
    """P5h+1 T1.5 (Codex B-lite): the new sub-span MUST be a required Lane-B
    tree span so the close-gate validator enforces emission. Without this
    presence-check the aggregator would silently fall back to synthesizing
    `unattributed_gda_step_7_kernel_and_cache_update` whenever the sub-span
    fails to emit, masking emitter regressions."""
    assert "gda_step_7_kernel_dispatch_and_materialize" in LANE_B_REQUIRED_TREE


def test_lane_b_required_tree_includes_cache_and_lm_head():
    for name in ("cache_state_update", "slice_last_and_project_lm_head"):
        assert name in LANE_B_REQUIRED_TREE, name


def test_lane_b_allowed_tree_keeps_tokenizer_encode():
    """tokenizer_encode stays in ALLOWED (not REQUIRED) — fires on both lanes
    pre-routing in the HTTP handler, presence on Lane-B not enforced."""
    assert "tokenizer_encode" in LANE_B_ALLOWED_TREE
    assert "tokenizer_encode" not in LANE_B_REQUIRED_TREE


def test_validate_chunk_ancestry_passes_for_matching_chunk_ids():
    """Two-deep chain (gs_chunk_N → decoder_layer_N → input_norm) with
    consistent chunk_idx must pass."""
    spans = [
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=100,
                parent_span_id="3",
                span_name="gs_chunk_N",
                parent_span="gs_stream_init_and_chunk_loop",
                chunk_idx="0",
            )
        ),
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=101,
                parent_span_id="100",
                span_name="decoder_layer_N",
                parent_span="gs_chunk_N",
                chunk_idx="0",
            )
        ),
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=102,
                parent_span_id="101",
                span_name="input_norm",
                parent_span="decoder_layer_N",
                chunk_idx="0",
            )
        ),
    ]
    assert validate_chunk_ancestry(spans) == []


def test_validate_chunk_ancestry_fails_when_descendant_chunk_idx_mismatches():
    """Descendant with chunk_idx=1 under a gs_chunk_N ancestor with
    chunk_idx=0 must fail."""
    spans = [
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=100,
                parent_span_id="3",
                span_name="gs_chunk_N",
                parent_span="gs_stream_init_and_chunk_loop",
                chunk_idx="0",
            )
        ),
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=101,
                parent_span_id="100",
                span_name="decoder_layer_N",
                parent_span="gs_chunk_N",
                chunk_idx="1",  # MISMATCH
            )
        ),
    ]
    failures = validate_chunk_ancestry(spans)
    assert any("chunk_idx=1" in f and "ancestor" in f for f in failures), failures


def test_validate_chunk_ancestry_fails_when_gs_chunk_n_has_null_chunk_idx():
    spans = [
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=100,
                parent_span_id="3",
                span_name="gs_chunk_N",
                parent_span="gs_stream_init_and_chunk_loop",
                chunk_idx="null",  # MUST NOT be null on a gs_chunk_N span
            )
        ),
    ]
    failures = validate_chunk_ancestry(spans)
    assert any("gs_chunk_N" in f and "null chunk_idx" in f for f in failures), failures


def test_validate_chunk_ancestry_fails_when_outside_chunk_has_non_null_chunk_idx():
    """A span emitted outside any gs_chunk_N ancestor must carry chunk_idx=null."""
    spans = [
        parse_line(
            _build_line(
                request_id="lb-req",
                routing_path="gs_chunked",
                prompt_tokens=4096,
                span_id=4,
                parent_span_id="3",
                span_name="gs_kv_cache_alloc",
                parent_span="gs_stream_init_and_chunk_loop",
                chunk_idx="0",  # Should be null — gs_kv_cache_alloc is pre-loop
            )
        ),
    ]
    failures = validate_chunk_ancestry(spans)
    assert any(
        "outside gs_chunk_N" in f and "gs_kv_cache_alloc" in f for f in failures
    ), failures


def test_rust_lane_b_allow_list_subset_of_python():
    """Cross-check: every name in core/p5h.rs::LANE_B_ALLOWED_TRY_SPAN_NAMES
    must appear in Python LANE_B_ALLOWED_TREE (otherwise the Rust emit would
    succeed but the Python validator would reject as unexpected)."""
    p5h_path = (
        Path(__file__).resolve().parents[3] / "ironmlx" / "src" / "core" / "p5h.rs"
    )
    source = p5h_path.read_text()
    match = re.search(
        r"const\s+LANE_B_ALLOWED_TRY_SPAN_NAMES:\s*&\[\&str\]\s*=\s*&\[(.*?)\];",
        source,
        re.DOTALL,
    )
    assert match is not None, "could not locate LANE_B_ALLOWED_TRY_SPAN_NAMES in p5h.rs"
    names = re.findall(r'"([^"]+)"', match.group(1))
    assert names, "no names extracted from LANE_B_ALLOWED_TRY_SPAN_NAMES"
    rust_set = set(names)
    missing = rust_set - LANE_B_ALLOWED_TREE
    assert not missing, (
        f"Rust try-helper allow-list has names not in Python LANE_B_ALLOWED_TREE: {missing}"
    )
