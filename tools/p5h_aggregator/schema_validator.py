"""P5h schema validator — implements § 2.5a structural checks.

Single source of truth: docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md § 2.5a.
DO NOT re-derive semantics here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

P5H_LOG_RE = re.compile(
    r"\[p5h-profile\] "
    r"request_id=(?P<request_id>\S+) "
    r"routing_path=(?P<routing_path>\S+) "
    r"prompt_tokens=(?P<prompt_tokens>\d+) "
    r"seq=(?P<seq>\d+) "
    r"layer_idx=(?P<layer_idx>-?\d+) "
    r"chunk_idx=(?P<chunk_idx>null|\d+) "
    r"span_id=(?P<span_id>\d+) "
    r"parent_span_id=(?P<parent_span_id>\S+) "
    r"span_name=(?P<span_name>\S+) "
    r"parent_span=(?P<parent_span>\S+) "
    r"start_ns=(?P<start_ns>\d+) "
    r"end_ns=(?P<end_ns>\d+) "
    r"mode=(?P<mode>\S+) "
    r"span_kind=(?P<span_kind>\S+)"
)

# Per Codex plan review v1 P1 #2: split required sets by span_kind so the
# presence check doesn't fail on Lane-A diagnostic spans being absent from
# tree_spans. Each lane's required set is split into a tree subset and a
# diagnostic subset, checked against the corresponding span_kind partition.
#
# Per Codex T4 review (P1.1 + P1.2): the validator must also expose a wider
# ALLOWED set distinct from REQUIRED:
#   * REQUIRED → presence-checked (every non-aborted request MUST emit each).
#   * ALLOWED  → closed-set rejection (any emitted span not in ALLOWED fails).
# `tokenizer_encode` (T4.4) fires on both lanes in the HTTP handler pre-
# routing — must be ALLOWED on Lane-B (not REQUIRED, since presence is
# already covered by Lane-A's tree set and Lane-B fixture doesn't enforce it).
# `first_eval_amortized_cost` (T4.5) is a static OnceLock diagnostic that
# fires at most ONCE per process via close_p5h_span_diagnostic — cannot be
# REQUIRED per-request, but MUST be ALLOWED on Lane-A.

LANE_A_REQUIRED_TREE = {
    "server_request_recv_to_first_content_sse_write",
    "http_parse_render_tokenize",
    "scheduler_admission",
    "model_prefill_forward",
    "first_token_sampling_prepare",
    "first_token_sampling_materialize_and_sample",
    "detok_format_first_content_chunk",
}
LANE_A_REQUIRED_DIAGNOSTIC = {
    "sse_write_role_chunk_diagnostic",
}
LANE_A_ALLOWED_DIAGNOSTIC = LANE_A_REQUIRED_DIAGNOSTIC | {
    # T4.5 retroactive: static OnceLock fires once per process via
    # close_p5h_span_diagnostic at p5h.rs; allowed on Lane-A but not required.
    "first_eval_amortized_cost",
}

LANE_B_REQUIRED_TREE = {
    "server_request_recv_to_first_content_sse_write",
    "http_parse_render_tokenize",
    "gs_stream_init_and_chunk_loop",
    "gs_kv_cache_alloc",  # per Codex plan review v21 P1 — children of
    "gs_chunk_N",  # gs_stream_init_and_chunk_loop on Lane-B,
    "gs_first_token_sample_dispatch",  # all three were allow-listed for emission
    "sse_write_role_chunk",  # in v20 but only the third was required.
    "gs_first_token_materialize_and_predispatch",
    "detok_format_first_content_chunk",
    # P5h+1 T2: Lane-B is no longer top-level-only. The full decoder
    # hierarchy emits under `gs_chunk_N` ancestors (Rust try-helper
    # allow-list `LANE_B_ALLOWED_TRY_SPAN_NAMES` extended in lockstep).
    # Required-presence enforces that the chunked-prefill loop actually
    # opened the wrappers so T1's per-substep eval probes light up.
    # Decoder wrappers (children of gs_chunk_N → input transitions):
    "decoder_layer_N",
    "input_norm",
    "attention_path",
    "residual_overhead",
    "post_attention_norm",
    "mlp_path",
    # T2 GatedAttention substeps under `attention_path`:
    "q_gate_k_v_proj",
    "q_split_norm_reshape",
    "mrope_apply",
    "kv_mask_update",
    "fused_sdpa",
    "gate_sigmoid_mul",
    "o_proj",
    # T3 MoE substeps under `mlp_path` (incl. shared expert + routing):
    "router_logits_softmax_topk",
    "routing_sort_pack",
    "gather_qmm_gate_up",
    "swiglu_activation",
    "gather_qmm_down",
    "routing_unsort_weighted_reduce",
    "shared_expert",
    "moe_output_sum",
    # GDN 11 substeps under `attention_path` (hybrid model):
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
    # P5h+1 T1.5 (Codex B-lite): child of
    # `gda_step_7_kernel_and_cache_update` that owns the kernel
    # select/build/state_in/t_arr/dispatch/take_at/materialize-eval work
    # (everything except the cache mutation). Required on Lane-B so the
    # parent's self-time is fully attributed and stops being synthesized
    # as `unattributed_gda_step_7_kernel_and_cache_update`.
    "gda_step_7_kernel_dispatch_and_materialize",
    "gda_step_8_norm_proj",
    # Cache + lm_head emitted once per chunk:
    "cache_state_update",
    "slice_last_and_project_lm_head",
}
LANE_B_ALLOWED_TREE = LANE_B_REQUIRED_TREE | {
    # T4.4 retroactive subspan of http_parse_render_tokenize; fires on both
    # lanes because handler entry is pre-routing. Allowed on Lane-B but not
    # required (Lane-B presence not enforced for tokenizer_encode), so it
    # stays in ALLOWED but not REQUIRED.
    "tokenizer_encode",
    # P5i.c Phase 1 Stage α opt-in children of gather_qmm_gate_up.
    "gate_up_input_shape_prep",
    "gate_up_gather_qmm_call",
    "gate_up_slice_outputs",
}
LANE_B_REQUIRED_DIAGNOSTIC: set[str] = set()  # no Lane-B diagnostic spans currently
LANE_B_ALLOWED_DIAGNOSTIC: set[str] = set()  # no Lane-B diagnostic spans currently


def required_sets_for_routing(routing: str) -> tuple[set[str], set[str]]:
    if routing == "scheduler":
        return LANE_A_REQUIRED_TREE, LANE_A_REQUIRED_DIAGNOSTIC
    if routing == "gs_chunked":
        return LANE_B_REQUIRED_TREE, LANE_B_REQUIRED_DIAGNOSTIC
    return set(), set()


def allowed_diagnostic_for_routing(routing: str) -> set[str]:
    """Closed allow-set for diagnostic span_names. Superset of REQUIRED to
    accommodate one-shot diagnostics (e.g. first_eval_amortized_cost) that
    cannot be presence-checked per-request."""
    if routing == "scheduler":
        return LANE_A_ALLOWED_DIAGNOSTIC
    if routing == "gs_chunked":
        return LANE_B_ALLOWED_DIAGNOSTIC
    return set()


@dataclass
class Span:
    request_id: str
    routing_path: str
    prompt_tokens: int
    seq: int
    layer_idx: int
    # P5h+1 T2: Lane-B chunk index (zero-based). Non-null only for spans
    # emitted inside a `gs_chunk_N` ancestor (see validate_chunk_ancestry).
    chunk_idx: int | None
    span_id: int
    parent_span_id: int | None
    span_name: str
    parent_span: str | None
    start_ns: int
    end_ns: int
    mode: str
    span_kind: str
    inclusive_us: float = 0.0
    exclusive_us: float = 0.0

    def __post_init__(self):
        self.inclusive_us = (self.end_ns - self.start_ns) / 1000.0


def parse_line(line: str) -> Span | None:
    m = P5H_LOG_RE.search(line)
    if not m:
        return None
    g = m.groupdict()
    pid = g["parent_span_id"]
    chunk_idx_raw = g["chunk_idx"]
    return Span(
        request_id=g["request_id"],
        routing_path=g["routing_path"],
        prompt_tokens=int(g["prompt_tokens"]),
        seq=int(g["seq"]),
        layer_idx=int(g["layer_idx"]),
        chunk_idx=None if chunk_idx_raw == "null" else int(chunk_idx_raw),
        span_id=int(g["span_id"]),
        parent_span_id=None if pid == "null" else int(pid),
        span_name=g["span_name"],
        parent_span=None if g["parent_span"] == "null" else g["parent_span"],
        start_ns=int(g["start_ns"]),
        end_ns=int(g["end_ns"]),
        mode=g["mode"],
        span_kind=g["span_kind"],
    )


@dataclass
class ValidationReport:
    failures: list[str] = field(default_factory=list)
    request_count: int = 0
    tree_span_count: int = 0
    diagnostic_span_count: int = 0
    # Per Codex plan review v12 P2 #6: True iff the root span carries
    # mode="aborted" (pre-first-content terminal close). Consumers (T0a.14
    # verifier, T5 aggregator) exclude these from coverage gates.
    aborted: bool = False

    def fail(self, msg: str):
        self.failures.append(msg)

    @property
    def ok(self) -> bool:
        return not self.failures


def validate_request(
    spans: list[Span], *, prefill_chunk_size: int = 2048
) -> ValidationReport:
    """Run § 2.5a structural checks on one request's worth of spans."""
    report = ValidationReport(request_count=1)
    tree = [s for s in spans if s.span_kind == "tree"]
    diag = [s for s in spans if s.span_kind == "diagnostic"]
    report.tree_span_count = len(tree)
    report.diagnostic_span_count = len(diag)

    # Per Codex plan review v12 P2 #6: pre-first-content abort requests
    # (root closed via RootSpanHandle::close_at_aborted, mode="aborted")
    # intentionally lack `detok_format_first_content_chunk` and downstream
    # spans. Skip the per-lane required-set check + interval containment
    # check for these requests; still run id-uniqueness + closure + single-root.
    aborted = any(s.parent_span_id is None and s.mode == "aborted" for s in tree)
    report.aborted = aborted

    if not tree:
        report.fail("no tree spans emitted")
        return report

    # Per-record validity
    for s in spans:
        if not s.request_id:
            report.fail(f"empty request_id on span {s.span_name}")
        if s.prompt_tokens == 0:
            report.fail(f"prompt_tokens=0 on span {s.span_name}")
        if s.routing_path not in ("scheduler", "gs_chunked"):
            report.fail(f"invalid routing_path={s.routing_path} on span {s.span_name}")

    # Id uniqueness within request
    ids = [s.span_id for s in spans]
    if len(set(ids)) != len(ids):
        report.fail("duplicate span_id within request")

    # Exactly one root with span_name = server_request_recv_to_first_content_sse_write
    roots = [s for s in tree if s.parent_span_id is None]
    if len(roots) != 1:
        report.fail(f"expected exactly 1 root, found {len(roots)}")
    elif roots[0].span_name != "server_request_recv_to_first_content_sse_write":
        report.fail(
            f"root span_name is {roots[0].span_name}, expected server_request_recv_to_first_content_sse_write"
        )

    # Per-request identity consistency (per Codex review v24 P3): root is the
    # source of truth for routing. Root is emitted at close time, so `tree[0]`
    # is not a safe proxy for request routing when logs are unsorted.
    req_id = roots[0].request_id if len(roots) == 1 else "<unknown-root-request>"
    routing = roots[0].routing_path if len(roots) == 1 else "<unknown-root-routing>"
    if len(roots) == 1:
        for s in spans:
            if s.request_id != req_id:
                report.fail(
                    f"request_id mismatch: root has {req_id}, span {s.span_name} has {s.request_id}"
                )
            if s.routing_path != routing:
                report.fail(
                    f"routing_path mismatch: root has {routing}, span {s.span_name} has {s.routing_path}"
                )

    # No orphan top-level (non-root tree span with null parent)
    for s in tree:
        if (
            s.parent_span_id is None
            and s.span_name != "server_request_recv_to_first_content_sse_write"
        ):
            report.fail(f"orphan top-level tree span: {s.span_name}")

    # Closure: every non-null parent_span_id resolves
    by_id = {s.span_id: s for s in tree}
    for s in tree:
        if s.parent_span_id is not None and s.parent_span_id not in by_id:
            report.fail(f"orphan parent_span_id={s.parent_span_id} on {s.span_name}")

    # Label self-consistency
    for s in tree:
        if (s.parent_span_id is None) != (s.parent_span is None):
            report.fail(
                f"label inconsistency on {s.span_name}: parent_span_id={s.parent_span_id}, parent_span={s.parent_span}"
            )
        if s.parent_span_id is not None and s.parent_span_id in by_id:
            if by_id[s.parent_span_id].span_name != s.parent_span:
                report.fail(
                    f"parent_span label mismatch on {s.span_name}: parent_span_id resolves to {by_id[s.parent_span_id].span_name} but parent_span={s.parent_span}"
                )

    # Interval containment
    for s in tree:
        if s.parent_span_id is not None and s.parent_span_id in by_id:
            p = by_id[s.parent_span_id]
            if not (p.start_ns <= s.start_ns and s.end_ns <= p.end_ns):
                report.fail(
                    f"interval not contained on {s.span_name}: parent [{p.start_ns}, {p.end_ns}], child [{s.start_ns}, {s.end_ns}]"
                )

    # Reachability + no cycle
    if len(roots) == 1:
        children_by_parent: dict[int, list[Span]] = {}
        for s in tree:
            if s.parent_span_id is not None:
                children_by_parent.setdefault(s.parent_span_id, []).append(s)
        visited: set[int] = set()
        stack = [roots[0]]
        while stack:
            cur = stack.pop()
            if cur.span_id in visited:
                report.fail(
                    f"cycle detected at span {cur.span_name} (id={cur.span_id})"
                )
                break
            visited.add(cur.span_id)
            for c in children_by_parent.get(cur.span_id, []):
                stack.append(c)
        unreachable = set(by_id.keys()) - visited
        if unreachable:
            report.fail(
                f"unreachable tree spans from root: {[by_id[i].span_name for i in unreachable]}"
            )

    # Route-aware required span_names (per Codex plan review v1 P1 #2 — check
    # tree subset against tree_spans, diagnostic subset against diagnostic_spans).
    # Per Codex v12 P2 #6: skip required-set check for aborted requests — they
    # legitimately lack downstream spans because first content was never sent.
    if not aborted:
        tree_names = {s.span_name for s in tree}
        diag_names = {s.span_name for s in diag}
        required_tree, required_diag = required_sets_for_routing(routing)
        missing_tree = required_tree - tree_names
        missing_diag = required_diag - diag_names
        if missing_tree:
            report.fail(f"missing required tree spans for {routing}: {missing_tree}")
        if missing_diag:
            report.fail(
                f"missing required diagnostic spans for {routing}: {missing_diag}"
            )
        # Per Codex plan review v22 P1: Lane-B is top-level-only in P5h.
        # Presence checks alone are insufficient: a buggy route-aware `try_`
        # helper could emit deep Lane-A names under chunked GS while still
        # satisfying all required Lane-B buckets. Reject any non-aborted
        # Lane-B tree span whose name is outside the allowed top-level set
        # (with repeated `gs_chunk_N` represented once in the set).
        if routing == "gs_chunked":
            # Reject against LANE_B_ALLOWED_TREE (REQUIRED + retroactive
            # additions like `tokenizer_encode` per Codex T4 P1.1 review).
            unexpected_tree = tree_names - LANE_B_ALLOWED_TREE
            if unexpected_tree:
                report.fail(
                    f"unexpected Lane-B tree spans (deep emission forbidden in P5h): {unexpected_tree}"
                )

    # Diagnostic checks (per § 2.5a + Codex plan review v1 P2 #4 + v23 P3 +
    # T4 P1.2): diagnostic span names are route-specific closed sets. Lane A
    # allows `sse_write_role_chunk_diagnostic` + `first_eval_amortized_cost`;
    # Lane B allows none.
    root_span_id = roots[0].span_id if len(roots) == 1 else None
    allowed_diag = allowed_diagnostic_for_routing(routing)
    for d in diag:
        if d.span_name not in allowed_diag:
            report.fail(f"unexpected diagnostic span_name for {routing}: {d.span_name}")
        # Per § 2.5a "Diagnostic span checks": parent_span_id MUST be None OR
        # point at root.span_id. Anything else = emitter bug.
        if d.parent_span_id is not None and d.parent_span_id != root_span_id:
            report.fail(
                f"diagnostic span {d.span_name} parent_span_id={d.parent_span_id} — "
                f"must be null or root's span_id ({root_span_id})"
            )

    # Decoder-descendant layer_idx sanity (per Codex plan review v13 P1 #2):
    # any span transitively under `decoder_layer_N` MUST have layer_idx >= 0
    # (the real decoder layer index plumbed via decoder_layer.rs → gated_*.rs
    # → substep SpanFields). layer_idx == -1 on a decoder-descendant means
    # the plumbing missed that site and the span will be unattributable
    # across the 40 decoder layers. Skip this check for aborted requests
    # (their tree may be partially populated).
    if not aborted:
        by_id = {s.span_id: s for s in tree}

        def under_decoder_layer(span):
            cur = span
            while cur.parent_span_id is not None and cur.parent_span_id in by_id:
                cur = by_id[cur.parent_span_id]
                if cur.span_name == "decoder_layer_N":
                    return True
            return False

        for s in tree:
            if s.span_name == "decoder_layer_N":
                if s.layer_idx < 0:
                    report.fail(
                        f"decoder_layer_N has layer_idx={s.layer_idx} (must be 0..num_layers-1)"
                    )
                continue
            if under_decoder_layer(s) and s.layer_idx < 0:
                report.fail(
                    f"decoder-descendant span {s.span_name} has layer_idx=-1 — "
                    f"layer_idx plumbing missing in gated_delta_net.rs / gated_attention.rs / sparse_moe.rs"
                )

    # pre_content_decode_steps hard gate (per § 2.5a).
    # Per Codex plan review v13 P1 #1: aborted requests legitimately may have
    # emitted `pre_content_decode_steps` before hitting the abort terminal
    # (e.g. Lane-B per-iteration loop opened a `pre_content_decode_steps` span,
    # then `stream.next_token()` returned Err, and the closure-scope guard
    # closed root via close_at_aborted). Skip this hard gate for aborted
    # requests — they intentionally diverge from the happy-path span shape.
    if not aborted:
        pcds_count = sum(1 for s in tree if s.span_name == "pre_content_decode_steps")
        if pcds_count > 0:
            report.fail(
                f"pre_content_decode_steps count={pcds_count} > 0 — first prefill token did not detokenize non-empty; adjust benchmark prompts"
            )

    # Lane-B chunk-count check (per Codex plan review v21 P1):
    # `gs_chunk_N` is REPEATED — emitted once per chunk inside the
    # `GenerationStream::new(...)` chunked prefill loop. The required-tree
    # presence check above only asserts >= 1 instance; here we additionally
    # validate that the count matches the expected number of chunks per
    # request, computed from the request's `prompt_tokens` and the bench's
    # `prefill_chunk_size` (which the validator can read off the active root
    # span's ctx). When chunk_size is unavailable (older fixtures), require
    # at least 1 gs_chunk_N for any Lane-B request — silently emitting zero
    # chunks would mean the entire chunked-prefill loop body never ran the
    # try_ wrapper, which is a real instrumentation failure.
    if not aborted and routing == "gs_chunked":
        chunk_count = sum(1 for s in tree if s.span_name == "gs_chunk_N")
        if chunk_count < 1:
            report.fail(
                f"Lane-B request emitted {chunk_count} gs_chunk_N spans — "
                f"`GenerationStream::new` chunked-prefill loop body did not "
                f"reach try_with_p5h_span_from_current_trace (per Codex v21 P1)"
            )
        # If prefill_chunk_size + prompt_tokens are known, also assert exact
        # expected chunk count. Default ironmlx server uses prefill_chunk_size
        # = 2048 (per `serve.rs` default); root span's prompt_tokens field is
        # the join key for this check. ceil(prompt_tokens / chunk_size) is
        # the expected emission count when the request entered Lane-B
        # (PP > chunk_size).
        if tree:
            prompt_tokens = tree[0].prompt_tokens
            # `prefill_chunk_size` is a per-call kwarg (default 2048 = ironmlx
            # server default per `serve.rs`). The T0a.14 harness reads the
            # actual `--prefill-chunk-size` from the spawn args and passes it
            # in; the standalone validator tests use the 2048 default.
            expected_chunks = (
                prompt_tokens + prefill_chunk_size - 1
            ) // prefill_chunk_size
            if expected_chunks > 0 and chunk_count != expected_chunks:
                report.fail(
                    f"Lane-B gs_chunk_N count mismatch: got {chunk_count}, "
                    f"expected {expected_chunks} = ceil({prompt_tokens}/{prefill_chunk_size}). "
                    f"Either the chunk loop's try_ wrapper missed an iteration "
                    f"or the bench's prefill_chunk_size differs from {prefill_chunk_size} "
                    f"(per Codex v21 P1). Pass the actual chunk_size via "
                    f"`validate_request(spans, prefill_chunk_size=...)`."
                )

    # P5h+1 T2: chunk_idx propagation under gs_chunk_N ancestors. Run on every
    # request (including aborted) because a malformed chunk_idx implies a
    # schema bug, not a happy-path coverage gap.
    for failure in validate_chunk_ancestry(spans):
        report.fail(failure)

    return report


def validate_chunk_ancestry(spans: list[Span]) -> list[str]:
    """P5h+1 T2 structural rule for chunk_idx propagation.

    Every span emitted under a `gs_chunk_N` ancestor MUST carry the same
    non-null `chunk_idx` as the ancestor (set by the Rust RAII
    `P5hChunkContextGuard` in `core::generate::GenerationStream::new`).
    Spans emitted outside any `gs_chunk_N` ancestor (Lane-A entirely;
    Lane-B pre/post-loop sites such as `gs_kv_cache_alloc` and
    `gs_first_token_sample_dispatch`) MUST carry `chunk_idx=null` (which
    parses to Python `None`).

    Returns a list of human-readable failure messages; empty list = pass.
    """
    by_id = {s.span_id: s for s in spans}
    failures: list[str] = []

    for span in spans:
        if span.span_name == "gs_chunk_N" and span.chunk_idx is None:
            failures.append(f"gs_chunk_N span_id={span.span_id} has null chunk_idx")
            continue

        ancestor = (
            by_id.get(span.parent_span_id) if span.parent_span_id is not None else None
        )
        chunk_ancestor: Span | None = None
        while ancestor is not None:
            if ancestor.span_name == "gs_chunk_N":
                chunk_ancestor = ancestor
                break
            ancestor = (
                by_id.get(ancestor.parent_span_id)
                if ancestor.parent_span_id is not None
                else None
            )

        if chunk_ancestor is None:
            if span.span_name != "gs_chunk_N" and span.chunk_idx is not None:
                failures.append(
                    f"span_id={span.span_id} ({span.span_name}) is outside gs_chunk_N "
                    f"but has chunk_idx={span.chunk_idx}"
                )
            continue

        if chunk_ancestor.chunk_idx is None:
            failures.append(
                f"gs_chunk_N ancestor span_id={chunk_ancestor.span_id} has null chunk_idx"
            )
        elif span.chunk_idx != chunk_ancestor.chunk_idx:
            failures.append(
                f"span_id={span.span_id} ({span.span_name}) has chunk_idx={span.chunk_idx} "
                f"but gs_chunk_N ancestor span_id={chunk_ancestor.span_id} has chunk_idx={chunk_ancestor.chunk_idx}"
            )

    return failures


def group_by_request(spans: Iterable[Span]) -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    for s in spans:
        out.setdefault(s.request_id, []).append(s)
    return out
