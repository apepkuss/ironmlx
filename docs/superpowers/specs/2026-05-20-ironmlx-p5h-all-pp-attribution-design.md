# P5h — ironmlx 全 PP Prefill Gap Attribution (Design Spec)

| Field | Value |
|---|---|
| Phase | P5h (post-P5g) |
| Date | 2026-05-20 |
| Branch | `ironmlx-p5h-perf` (from `ironmlx-p5g-perf` HEAD `31c01db`) |
| Hardware | M5 Max 128 GB |
| Model | `mlx-community/Qwen3.5-35B-A3B-4bit` (40 layers: 30 linear-attention/GDN + 10 full-attention; `num_experts=256`, `num_experts_per_tok=8`, `attn_output_gate=true`, hidden 2048, head_dim 256, moe_intermediate 512, shared_expert_intermediate 512 per config.json) |
| Status | Design / brainstorming complete, awaiting writing-plans |
| Prior phase | P5g — no opt promoted; T0 v2 + T1 revert + close-out at `31c01db` (see `reports/p5g-final-results.md`) |

## § 1 调研依据与决策摘要

### 1.1 P5g leave-off state + P5h trigger

P5g (GatedDeltaNet deep refactor) close-out 标 "no optimization promoted; T0 数据 + Layer 3 upper bound 数据归 P5h scope refresh" per § 7.3 success bar fallback。具体 leave-off:

- **Ship state**: P5g HEAD == P5f baseline (no GDN-internal opt; T1 fused projection reverted)
- **3-way bench measured (cool-restart-verified)**: ironmlx prefill 是 omlx 的 **49-52% at PP=2048-16384**, **88.7% at PP=128**
- **PP=16384 decode TG**: ironmlx +19.2% over omlx (P5f shipped +10.3% advantage preserved/strengthened)
- **GDN occupancy 实测**: 38-46% of prefill wall-time (vs P5g spec § 1.3 prior 假设 ~20%) — GDN 是 prefill primary cost slot
- **Phase C top-3 (GDN-internal)**: 1a_in_proj_qkvz 44-46% / 8_norm_proj 20-21% / 7_kernel 16-17% — 全 quantized matmul + kernel work, op-level fusion saturated
- **Phase D ablation 全 negative** (3 modes -2.89% to -10.59% vs Phase A): plan § 7.1 ablation upper-bound 假设被推翻,根因未确定
- **Linear-family saturation 已证**: T1 fused projection 实测 geomean -0.12%, PP=16384 -2.15% → revert
- **未测 areas**: GatedAttention layers (10/40 in MoE-A3B, full-attn O(S²) suspected long-PP dominant), MoE expert dispatch + LinearMLP (30 layers), HTTP / scheduler / admission (P5e/P5f 改动后没重测), tokenization / first-eval (短 PP 固定开销 suspect), lm_head + MLX eval/cache state

### 1.2 P5h 起点目标 (per Boss directive + chatgpt review v1)

ironmlx 性能目标 (spec § 1.1 P5f/P5g shared): **全 PP 段 prefill / decode / e2e 超过 omlx +10%**。P5g 3-way bench 揭示 gap 远比预期大:

| PP | ironmlx (P5g ship) | omlx | ironmlx/omlx | 达 omlx+10% 需提升 |
|---:|---:|---:|---:|---:|
| 128 | 948 | 1069 | 88.7% | +24% |
| 512 | 1578 | 2498 | 63.2% | +74% |
| 2048 | 1843 | 3515 | 52.4% | +110% |
| 4096 | 1834 | 3590 | 51.1% | +115% |
| 8192 | 1735 | 3542 | 49.0% | +124% |
| 16384 | 1599 | 3310 | 48.3% | +126% |

**关键观察**: gap 跨 PP 形态分化:
- **短 PP (128/512)**: gap 24-74%, 可能由固定开销 (scheduler / admission / tokenization / first-eval / fixed recurrent cost) 主导
- **长 PP (2048+)**: gap 110-128% (i.e., **2.1-2.3×** improvement needed), 必须 kernel/memory 层 (op-level 已证 saturated)

P5g 是 "measure-then-attempt-then-fail-fast" pattern。**P5h 是 "measure-only then re-prioritize" pattern** — 不优化,只把瓶颈拆清楚,然后决定 P5i (短 PP attack) + P5j (长 PP attack) 实施顺序。

### 1.3 Honest target feasibility caveat

PP=2048+ 需 **2.1-2.3× improvement on 4-bit quantized MoE** 是 stretch target。在 omlx (Python + mlx) 已高度优化的 kernel 上拿 2× 改进需要:

- Kernel rewrite (Step 7 GDN Metal kernel + Step 8 out_proj + quant Linear gather_qmm)
- Memory layout / KV cache state 改造
- 可能 device-aware tile selection (per memory `[device_aware_tile]`)
- Possibly graph reuse / pre-allocated buffer (per memory `[mlx_graph_reuse_stage11]`)

**P5h close-out 必须 include target feasibility assessment** — 出 attribution 后 honest 评估 "全 PP omlx+10% 在 P5i+P5j+P5h+1 内是否可达"。If not feasible within reasonable scope, Boss 决策 partial target (e.g., 短 PP +24% & 长 PP +30-50% partial close)。

## § 2 P5h Architecture

### 2.1 Measure-only phase + reusable infra

P5h 不 ship 任何 optimization。所有 instrument code feature-gated (复用 P5g `p5g-profile` feature 或新加 `p5h-profile`),default build byte-identical to P5f/P5g baseline。

复用 P5g 基础设施:
- `p5g_t0_gated_delta_profile.rs` harness pattern (per-PP server spawn, line-by-line stderr drainer, `wait_ready_or_fail` helper, `/tmp/p5g-env.sh` env persistence)
- 3-layer profile protocol (Layer 1 boundary, Layer 2 step breakdown, Layer 3 shape-preserving ablation)
- `tracing_subscriber::fmt().with_writer(std::io::stderr)` (P5g commit `5e35ab2`)
- `validate_request_group` Python helper + composite request marker `offset_before==0 AND layer==L_MIN`
- Off-line aggregator pattern (machine-generated reports, not hand-fill placeholders)

新增 P5h-specific infrastructure:
- **UMA cache state hardening protocol** (§ 2.4) — P5g T4 暴露的 noise source
- **GatedAttention instrumentation** (3-edit pattern, similar to P5g GDN)
- **Multi-layer attribution synthesis** (aggregator cross-component: HTTP + scheduler + GDN + GatedAttention + MoE + lm_head + MLX eval/cache)
- **Phase D root cause investigation** infrastructure (phase-order randomized harness control + substitute self-cost measurement protocol)

### 2.2 全链路 7-layer scope

Per Boss directive (chatgpt 建议 fully accepted),P5h 覆盖 7 areas:

| # | Layer | P5g status | P5h instrument scope |
|---:|---|---|---|
| 1 | **HTTP path** | P5f close-out measured baseline; P5e/P5f scheduler 改动后无重测 | iron-bench client-side ttft + server-side request entry/dispatch boundary timing |
| 2 | **Scheduler / admission** | P5e/P5f admit queue + b_max=1 default; per-request batch construction overhead 未细测 | Per-request admission latency, batch construction cost, slot allocation |
| 3 | **Tokenization / first-eval** | 短 PP 固定开销 prime suspect, P5e/P5f 未隔离 | Tokenizer Encode 时间, first-eval (JIT compile + kernel warmup) 一次性成本 |
| 4 | **GDN sub-step** | P5g T0 已测 11-step Layer 2 breakdown (commit `52c39bd`),但 `[p5g-profile]` 旧 schema 缺 § 2.5a server-emitted fields (`request_id / prompt_tokens / routing_path / seq / layer_idx / span_id / parent_span_id / span_name / parent_span / start_ns / end_ns / mode`) — **不能直接参与 § 7.1 exclusive coverage tree** | **新增 P5h substep 仪表化 (Codex review v2 P2 #1 + v4 P1 + v12 P1 + v13 P1 + v13 P2 #4)** — modest code change: gated_delta_net.rs 11-step substeps 各自 wrap 一个 `try_with_p5h_span_from_current_trace(...)` (None-tolerant per Codex v12 P1 #1,从 CLI/tests 调用时 no-op),substep 自然挂在 wrapper `attention_path` 下 (label `parent_span = "attention_path"`,id 自动取栈顶 from `P5H_CURRENT_SPAN_STACK`)。`attention_path` wrapper 由 `decoder_layer.rs::DecoderLayerMoe::forward_on` 统一打开 (T0a.11 Step 1),NOT 在 gated_delta_net.rs 内部。**仍复用 P5g 已 verified 的 instrumentation 位置 + Phase D 抗污染设计** (Layer1\|Layer2 mode-gate, AblateX barrier-free)。`[p5g-profile]` 现有 `tracing::info!` 行不动 (back-compat for P5g harness);`[p5h-profile]` 行来自 substep 包装器的 close emission,不是与 `[p5g-profile]` formatter 并列的手写行 (per Codex v11 P2 #6 + v13 P2 #4 — 双 emit 来自 DIFFERENT call sites)。T0a 实施 + rerun 同 PP set 下 UMA cold/warm pair。Cost ~11 min rerun wall (P5g T0 663s precedent) + ~2-4h code-change wall。P5g existing data 保留作 prior reference but not in P5h coverage gate。Kernel-level per-tile/per-shape timing 是 P5j (kernel rewrite) scope,不是 P5h attribution scope。 |
| 5 | **GatedAttention** | **P5g 完全未测**; 10/40 layers in `Qwen3.5-35B-A3B-4bit` (config.json `layer_types` "linear,linear,linear,full" × 10), full-attn O(S²), `attn_output_gate=true` 让 `q_proj` 输出含 gate (`Hq × D × 2`), `k_proj`/`v_proj` 用 KV-head dim (num_key_value_heads=2) | New 3-layer profile. Layer 1: entry/exit barrier. **Layer 2: 7-step code-backed taxonomy** (per T2 § 3 — `q_gate_k_v_proj`(`q_proj` with gate + `k_proj` + `v_proj`, separate not fused) / q_split_norm_reshape / mrope_apply / kv_mask_update / fused_sdpa / gate_sigmoid_mul / o_proj; `fused_sdpa` is `mlx::fast::scaled_dot_product_attention_on` and its internals — softmax / value matmul — are NOT separately measurable on production path). Layer 3 ablation: **conditional on T0b Phase D outcome** per § 2.5 / T2 conditional gate. |
| 6 | **MoE expert dispatch + LinearMLP** | P5g 未测; 40 decoder layers (10 含 full-attention, 30 含 linear-attention/GDN), 每层 SparseMoeBlock 含 `num_experts=256` + `num_experts_per_tok=8` (config.json verified) + 1 shared expert (`shared_expert_intermediate=512`) + gather_qmm routed compute + **sorted-routing pack/unpack 是 P5e shipped optimization, 不能藏在 gather bucket 内** (Codex review v2 P2 #2) | Per-layer 8-step code-backed taxonomy (per T3 § 3 — router_logits_softmax_topk / routing_sort_pack / gather_qmm_gate_up / swiglu_activation / gather_qmm_down / routing_unsort_weighted_reduce / shared_expert / moe_output_sum). **ROI math 必须从 `Qwen35MoeConfig` runtime values 来,不 hardcode** — routed work scales as `BS × num_experts_per_tok = BS × 8`。 |
| 7 | **lm_head + MLX eval/cache state** | P5e/P5f shipped lm_head fix; MLX eval barrier costs 未细测 | lm_head time + MLX `eval()` barrier latency + KVCache + GatedDeltaCache state-update cost |

### 2.3 3-layer profile protocol per layer (复用 P5g pattern + extend)

每个 layer 重复 P5g 的 Layer 1 / Layer 2 / Layer 3 protocol,具体扩展:

- **Layer 1 (boundary-isolated)**: entry barrier + exit barrier + emit `[p5h-profile] layer=<name> ...elapsed_us=N`. 估算该 layer 总占比。
- **Layer 2 (per-step breakdown)**: 每个 sub-op 用 `mlx::transforms::eval(&[&intermediate])?` materialize + timer push。append step_breakdown CSV 到 Layer 1 log line。
- **Layer 3 (shape-preserving ablation)**: 每个 sub-step 提 substitute (e.g., GatedDeltaNet step 5 compute_g substitute = zeros_like passthrough; GatedAttention `o_proj` substitute = identity-on-gated-output if scope permits). Mode-gated entry barriers off for AblateX (per P5g § 4.1a barrier-free invariant). **Layer 3 ablation 是 conditional on T0b Phase D root cause outcome** — per § 2.5 decision tree + T2/T3 conditional gates; H2/H4 verified 时 Layer 3 skip or replace with real-path microbenchmarks，multi-primary 时应用所有 verified mitigations。**特别注意**: `fused_sdpa` (`mlx::fast::scaled_dot_product_attention_on`) 的 softmax/value matmul internals 是 fused MLX call,**不能在 production path 上单独 ablation**;只能 ablate 整个 `fused_sdpa` 步骤 (用 e.g. zeros tensor 替换其输出)。

**Critical**: Phase D ablation 在 P5g 全 negative (反常)。P5h T0b 必须先 investigate root cause —— substitute self-cost / cache divergence / kernel materialization-dispatch variance / phase order thermal —— 否则 P5h 任何 ablation reading 也会被同样 anomaly 污染。

### 2.4 UMA cache state hardening protocol

P5g T4 暴露: sweep_full Qwen3.5-4B 之后跑 ironmlx serve restart, 3-way bench 测 ironmlx 数据 -20% vs T1-start baseline (same HEAD, ~25 min earlier)。Cool 5 min 后重测完全恢复匹配 P5f baseline。

**Hypothesis**: sweep_full 加载 Qwen3.5-4B 4-bit weights, evict Qwen3.5-MoE-35B 在 Apple Silicon UMA 中的 weight layout / page-table state。ironmlx serve 后续 load 17.5GB 进 sub-optimal cache state。

**P5h hardening protocol**:

1. **Phase-isolated spawn**: 不同 model 的 inference (e.g., sweep_full Qwen3.5-4B vs MoE-A3B bench) 不能背靠背跑;之间 cool ≥ 5 min。
2. **Cold-start baseline + warm reading 双值报告**: 每 bench iteration 跑 2 次,第 1 次 "cold" (cache 可能不最优), 第 2 次 "warm"。Report 标 both。Variance > ±2% 触发 cool-then-retry。
3. **UMA pressure probe**: T0a 加一个 sanity check — 测 ironmlx PP=2048 在 cold-start vs warm-reading,确认 ≤ ±2%。否则 abort + investigate。
4. **Strict serial 跨 server** (per `feedback_serial_perf_experiments`): 一次只起一个 server, 完全 kill + cool ≥ 30s + lsof port-free 再起下一个。
5. **Document in spec**: 任何后续 phase / report 引用 measurement, must annotate cold/warm state + hardening protocol applied。

### 2.5a Exclusive timing span schema (foundation for T5 attribution gate)

**Problem**: P5g 的 boundary-isolated medians per layer 不构成 mutually-exclusive timing tree。HTTP TTFT contains scheduler; scheduler contains forward dispatch; forward contains decoder layers; decoder layers contain GatedAttention/GDN/MoE; lm_head + MLX eval/cache 可能 overlap parent spans。若 T5 简单 sum medians, 要么 double-count (nested spans 重复算) 要么 leave gaps (同级 spans 之间未被 instrument 的边缘)。"95% wall-time accounted" gate 在此 schema 下 trivially mismeasure。

**Solution**: 单一 exclusive parent-child span tree。

Schema fields are split into **server-emitted** (written into each `[p5h-profile]` log line directly by `ironmlx serve`, since server owns this data) vs **aggregator-injected** (added by T5 Python aggregator from the iron-bench client-side sweep CSV, since server has no way to know iron-bench's `--prompt-len` or warmup/measured run index without a metadata channel that v3 P2 explicitly defers out of scope). This split is per Codex review v3 P2 — without it, the server would need iron-bench header propagation, which is out of P5h scope.

**Trace context propagation through call chain** (per Codex review v5 P1 #2 + v6 P2 #3 + v8 P1 #2 — without this, deep log sites in GDN/GatedAttention/MoE cannot emit `request_id` or `prompt_tokens` because HTTP-layer state doesn't reach them today; v9 corrects v7/v8's unsound async/thread-local lifetime design):

v5 originally proposed a thread-local around `model.batched_prefill(...)` only. v6/v7 extended scope to "ENTIRE root window from axum handler entry to first content SSE write". v8 review correctly flagged this is unsound — `chat_completion` is an **async axum handler** with `.await` points (`state.model.lock().await`, etc.). Tokio may move the future between worker threads between awaits, so `thread_local!` is NOT request-local OR task-local; the streaming-response root close happens in a `tokio::spawn`/`spawn_blocking` task that runs AFTER the handler future has returned — the original handler thread doesn't even live until first content SSE. v9 redesigns: explicit-value primary, thread-local as short-lived RAII guard only.

**Primary mechanism — explicit value, NOT thread-local-as-store:**

```rust
#[derive(Clone)]
struct P5hTraceContext {
    request_id: String,
    prompt_tokens: u32,
    routing_path: &'static str,  // "scheduler" | "gs_chunked"
}
```

The context is a plain value owned by the request handle (`GenerateRequest`, `RequestState`, spawned-task closure that needs it). It is **explicitly cloned into each thread/task** that owns a span. No part of this design relies on a thread-local outliving an `.await` boundary.

**Thread-local `P5H_CURRENT_TRACE` is a short-lived RAII guard inside sync regions, NOT a request-scoped store:**

```rust
thread_local! {
    static P5H_CURRENT_TRACE: RefCell<Option<P5hTraceContext>> = RefCell::new(None);
    static P5H_CURRENT_SPAN_STACK: RefCell<Vec<SpanHandle>> = RefCell::new(Vec::new());
}

struct P5hTraceGuard;
impl P5hTraceGuard {
    // Per Codex review v15 P1: enter() takes a `base_parent` SpanHandle that
    // seeds the span stack. Without it, the first `with_p5h_span_from_current_trace`
    // inside the guard would see an empty stack, emit `parent_span_id = None`,
    // and become a second "root" — failing T0a's single-root / no-orphan-top-level
    // / reachability structural checks.
    //
    // The base_parent is the EXPLICIT-CONTEXT span the caller has already opened
    // and is about to delegate sync work into. Examples (per § 2.5a propagation
    // chain below):
    //   - Lane-A `prefill_admitted_inner` (per Codex v16 P1 + v18 P2 #2 — NOT
    //     actor scope; actor cannot wedge spans between batched_prefill and
    //     sample_batch fused in this function): the inner function itself opens
    //     `model_prefill_forward` via open_p5h_span, enters guard with that
    //     span as base_parent, calls model.batched_prefill[_vl](...), drops
    //     guard, closes model_prefill_forward. Deep substeps inside
    //     (embed_lookup, decoder_layer_N, etc.) chain under
    //     model_prefill_forward via the stack. The actor itself does NOT enter
    //     a guard around sched.prefill_admitted(...).
    //   - Lane-B spawn_blocking: caller opens `gs_stream_init_and_chunk_loop`,
    //     then enters guard with that span as base_parent before
    //     GenerationStream::new(...). Deep substeps (gs_kv_cache_alloc,
    //     gs_chunk_N, gs_first_token_sample_dispatch) chain under it.
    //   - Lane-B per-iteration: caller opens `gs_first_token_materialize_and_predispatch`
    //     (or `pre_content_decode_steps`), then enters guard with that as
    //     base_parent before stream.next_token().
    fn enter(ctx: P5hTraceContext, base_parent: SpanHandle) -> Self {
        P5H_CURRENT_TRACE.with(|c| {
            let mut slot = c.borrow_mut();
            assert!(
                slot.is_none(),
                "P5hTraceGuard::enter while another guard is active — nested guards are forbidden \
                 (helpers must READ via P5H_CURRENT_TRACE, not enter their own guard); \
                 fix the guard set/drop sites in the calling task/thread"
            );
            *slot = Some(ctx);
        });
        P5H_CURRENT_SPAN_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            assert!(stack.is_empty(), "P5hTraceGuard::enter with non-empty span stack — prior guard leaked");
            stack.push(base_parent);  // sentinel: first with_p5h_span_from_current_trace sees this as parent
        });
        P5hTraceGuard
    }
}
impl Drop for P5hTraceGuard {
    fn drop(&mut self) {
        P5H_CURRENT_SPAN_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            assert_eq!(
                stack.len(), 1,
                "P5hTraceGuard::drop with span stack length {} — expected 1 (only base_parent sentinel). \
                 Either an inner span was opened without close, or close was called more times than open.",
                stack.len(),
            );
            stack.clear();  // pop the base_parent sentinel
        });
        P5H_CURRENT_TRACE.with(|c| *c.borrow_mut() = None);
    }
}
```

The guard MUST only wrap **synchronous, no-`.await` instrumentation regions** — code blocks where the executing thread is pinned for the duration. **Nesting is forbidden** (per Codex review v9 P3): instrumentation helpers and deep model log sites are READ-ONLY against `P5H_CURRENT_TRACE`; they MUST NOT enter their own guard. The `base_parent` argument to `enter` IS how deep work gets a meaningful parent without nesting another guard (per Codex review v15 P1).

**Dual emission API — span lifecycle model** (per Codex review v10 P1 #1 + P1 #2 + v13 P1 — earlier v11/v12 "one-shot emit_p5h_span(record)" model could not produce `parent_span_id` because the parent's `span_id` did not exist yet when nested children were being emitted; v13 adds id fields but kept the one-shot API; v14 redesigns to open/close lifecycle):

```rust
// Returned by open_*; opaque to caller other than passing as parent.
// Clone is REQUIRED (per Codex review v17 P1) so callers can plumb the
// handle through multiple uses without borrow-checker conflicts: e.g.,
// `Scheduler::cloned_active_row_p5h_trace_and_root` hands out an owned
// SpanHandle clone so prefill_admitted_inner can pass &root_span to
// open_p5h_span AND simultaneously hold &mut self.cache / &mut self.slots
// for the prefill call. P5hTraceContext is also Clone for the same reason.
#[derive(Clone)]
pub struct SpanHandle {
    span_id: u64,                 // atomic-counter generated at open
    span_name: &'static str,
    parent_span_id: Option<u64>,
    start_ns: u64,
}

// Optional per-span annotations supplied at close (e.g., layer_idx, seq, mode).
pub struct SpanFields { /* layer_idx, seq, mode, etc. */ }

// ============================================================================
// EXPLICIT-CONTEXT API — for async spans, cross-task spans, spans whose
// start/end live on different threads or straddle .await points.
// Used by: root (via RootSpanHandle), http_parse_render_tokenize,
// scheduler_admission, sse_write_role_chunk, detok_format_first_content_chunk.
// Caller passes both the trace context AND the parent SpanHandle as values.
// Never reads P5H_CURRENT_TRACE or P5H_CURRENT_SPAN_STACK.
// ============================================================================
// open at "now" (start_ns = monotonic_ns()). Use when the span start coincides
// with the call site.
fn open_p5h_span(
    ctx: &P5hTraceContext,
    parent: Option<&SpanHandle>,  // None only for the root
    span_name: &'static str,
) -> SpanHandle;

// open at an explicitly captured timestamp. Required (per Codex review v14 P1)
// for spans whose true start_ns is captured BEFORE the full ctx exists — root
// and `http_parse_render_tokenize` both fall in this category: handler entry
// must capture root_start_ns + http_parse_start_ns immediately, but ctx is
// only complete after tokenize + routing decision produce prompt_tokens and
// routing_path. Caller passes the earlier-captured timestamp.
fn open_p5h_span_at(
    ctx: &P5hTraceContext,
    parent: Option<&SpanHandle>,
    span_name: &'static str,
    start_ns: u64,
) -> SpanHandle;

fn close_p5h_span(
    ctx: &P5hTraceContext,
    handle: SpanHandle,
    end_ns: u64,
    fields: SpanFields,
);

// RootSpanHandle wraps SpanHandle with the ctx so the closing site (forwarder
// task / spawn_blocking body) can close root without re-plumbing ctx as a
// separate value. Clone is REQUIRED (per Codex review v18 P1) so handler can
// clone the handle into both Lane-A forwarder spawn AND Lane-B blocking task
// captures, AND into per-iteration loop bodies that need to access span as
// parent multiple times before the final close.
#[derive(Clone)]
pub struct RootSpanHandle {
    ctx: P5hTraceContext,
    span: SpanHandle,
}
impl RootSpanHandle {
    // crate-private accessors so cross-module emission sites (e.g., openai.rs)
    // can read fields without exposing private struct members.
    pub(crate) fn ctx(&self) -> &P5hTraceContext { &self.ctx }
    pub(crate) fn span(&self) -> &SpanHandle { &self.span }

    // close_at consumes self so each *clone* can close exactly once on its own
    // thread. Per Codex plan review v14 P1 #1: callers wrap root in
    // `P5hRootCloseGuard::new(root_handle)` (owns the Option, exposes
    // `.span()` for child parent lookup, `.close_success(end_ns)` for
    // happy-path once-close, `.is_open()` for "pre-first-content phase"
    // gating, and Drop runs `close_at_aborted(monotonic_ns())` for any
    // pre-first-content terminal path). The v13 design held
    // `&'a mut Option<RootSpanHandle>` borrowed from an outer variable —
    // that pattern fails to compile because the mutable borrow blocks every
    // subsequent `.as_ref()` / `.take()` callsite. Owning-Option pattern:
    //   let mut root_guard = P5hRootCloseGuard::new(root_handle);  // outside loop
    //   ...
    //   if first_non_empty_content {
    //       root_guard.close_success(end_ns);  // panics if called twice
    //   }
    // T0a tree structural check also asserts exactly one close record per
    // (request_id, root span_id) — duplicate close = double-close bug, fail.
    //
    // close_at is pub(crate) (per Codex review v20 P2) so cross-module call
    // sites — Lane-A forwarder + Lane-B spawn_blocking body, both in
    // openai.rs — can invoke it; ctx() / span() are already pub(crate) for
    // the same reason.
    pub(crate) fn close_at(self, end_ns: u64);  // calls close_p5h_span(&self.ctx, self.span, end_ns, ..)
}

// ============================================================================
// IMPLICIT-GUARD API — for deep, sync, no-`.await` instrumentation that runs
// inside an authorized P5hTraceGuard region. Used by: GDN entry/exit barriers,
// GatedAttention substeps, MoE substeps, lm_head span, model_prefill_forward
// sub-children, gs_stream_init_and_chunk_loop sub-children,
// gs_first_token_materialize_and_predispatch sub-work.
// Internally: opens span (parent = top of P5H_CURRENT_SPAN_STACK), pushes
// handle onto stack, runs `f`, pops handle, closes span. Panics if
// P5H_CURRENT_TRACE is None (no active guard) — that panic IS the guard-set/
// drop validation.
// ============================================================================
fn with_p5h_span_from_current_trace<T>(
    span_name: &'static str,
    fields_fn: impl FnOnce() -> SpanFields,
    body: impl FnOnce() -> T,
) -> T;
```

**`P5H_CURRENT_SPAN_STACK` discipline**:

- `P5H_CURRENT_SPAN_STACK` is declared in the `thread_local!` block at the top of this subsection (alongside `P5H_CURRENT_TRACE`).
- `P5hTraceGuard::enter(ctx, base_parent)` push-seeds the stack with `base_parent`; `Drop` asserts stack length == 1 (only sentinel remains) then clears.
- `with_p5h_span_from_current_trace` and its None-tolerant wrapper `try_with_p5h_span_from_current_trace` (per Codex plan review v12 P1 #1) are the ONLY APIs that push/pop additional entries. When `P5H_CURRENT_TRACE` is None, `try_` skips the push/pop entirely and just runs body. Manual stack manipulation by instrumentation code is forbidden — same fail-fast discipline as guard nesting.
- Explicit-context spans (root + top-level + SSE) do NOT touch the stack; they plumb parent via the `Option<&SpanHandle>` parameter to `open_p5h_span[_at]`.

**Authorized `P5hTraceGuard::enter(ctx, base_parent)` sites** (only these — no others permitted; each requires the caller to FIRST open the corresponding top-level span via explicit API, THEN enter the guard with that span as `base_parent`):

- (Lane B prefill, in `spawn_blocking` closure) Caller opens `gs_stream_init_and_chunk_loop` via `open_p5h_span(&ctx, Some(root_guard.span()), ...)` (per Codex plan review v14 P1 #1 + v15 P2 #2 — root is held in `P5hRootCloseGuard` declared at closure top, not a raw `root_handle` local). Then enters `P5hTraceGuard` with that span as `base_parent` and calls `GenerationStream::new(...)`. **Lane-B top-level-only emission (per Codex plan review v20 P1 #1):** while the guard is active, `try_with_p5h_span_from_current_trace` checks the active `routing_path`; on `"gs_chunked"` it emits ONLY for span_names in the Lane-B allow-list `{gs_kv_cache_alloc, gs_chunk_N, gs_first_token_sample_dispatch}` and no-ops all decoder / GDN / GatedAttention / MoE / lm_head deep names. So the guard's chain-via-stack mechanism gives those three top-level spans their `parent_span_id = gs_stream_init_and_chunk_loop.span_id`, while every deep `try_` call inside `model.forward_*` / `make_cache` / `sample_async_greedy` runs body directly with no emission. Lane-B's chunked GS path is top-level-only in P5h per spec § 5; per-chunk substep attribution is deferred to P5h+1 (requires `chunk_idx` schema extension to disambiguate N records per request). On `new()` return: drop guard, close `gs_stream_init_and_chunk_loop`.
- (Lane A, INSIDE `prefill_admitted_inner` — per Codex v16 P1 — NOT at actor scope, because actor cannot wedge between prefill and sampling) Caller (the inner function itself) opens `model_prefill_forward` via `open_p5h_span(&ctx, Some(&root_span), ...)`. Then enters guard with `model_prefill_forward` as `base_parent` and calls `model.batched_prefill[_vl](...)`. Deep substeps (`embed_lookup`, `decoder_layer_N`, etc.) chain under `model_prefill_forward`. On return: drop guard, close `model_prefill_forward`. (`first_token_sampling` is opened/closed separately later in the same function — no guard needed unless T4 adds deep sampling spans.)
- (Lane A, INSIDE `Scheduler::step` IF that function also fuses model-forward + sample like `prefill_admitted_inner`) Same SINK pattern: open `pre_content_decode_steps` inside `step`, enter guard with that as `base_parent`, run model-forward + sample, drop guard, close span. T0a verifies whether `step` fuses or not; if `step` is purely sync at the actor level with no fused phases, the simpler actor-scope guard (open span at actor, enter guard, call `step`, drop, close) works.
- (Lane B per-iteration, in `spawn_blocking` closure) For each `stream.next_token()` call inside the `spawn_blocking` body's loop: caller opens the corresponding top-level explicit span (`gs_first_token_materialize_and_predispatch` for the first iteration, `pre_content_decode_steps` if first detok was empty) via explicit API, enters guard with that as `base_parent`, calls `next_token()`, drops guard, closes the span. **Per Codex plan review v20 P1 #1 + Lane-B top-level-only:** the `try_with_p5h_span_from_current_trace` allow-list (defined in `p5h.rs`) does NOT include `gs_first_token_materialize_and_predispatch` or `pre_content_decode_steps` — these per-iteration top-level spans are opened/closed via the EXPLICIT `open_p5h_span` + `close_p5h_span` API in the closure body, not via the `try_` helper. The guard's role is to seed the stack so that IF the suppression allow-list permitted decoder substeps under Lane-B (it doesn't), they would chain correctly. With the allow-list in force, the only `try_` emissions during `next_token()` are the three Lane-B-allowed gs_* spans from `GenerationStream::new`'s first iteration (subsequent iterations don't re-enter `new()`); all decoder / GDN / etc deep names no-op. This preserves the Lane-B top-level-only invariant per spec § 5.

This pattern ensures every deep span has a non-null `parent_span_id` chain reaching back to the root, so T0a's id-based structural checks (single-root, no-orphan-top-level, reachability) hold.

**Spans that MUST use the explicit-context API (do NOT enter a guard)**:

- `server_request_recv_to_first_content_sse_write` (root) — opens in async axum handler, closes in spawned forwarder task / `spawn_blocking` body. Implementation: handler captures `root_start_ns` at entry, opens the span via `open_p5h_span_at(&ctx, None, "server_request_recv_to_first_content_sse_write", root_start_ns)` once `ctx` is complete, and wraps the returned `SpanHandle` in `RootSpanHandle { ctx, span }`; clones this handle into both Lane-A forwarder spawn AND Lane-B blocking task. Each spawn body holds its clone as `let mut root_guard = P5hRootCloseGuard::new(root_handle_clone);` (per Codex plan review v14 P1 #1 — owning-Option guard exposes `.span()` / `.close_success(end_ns)` / `.is_open()` / Drop runs `close_at_aborted` for any pre-first-content terminal path). The iteration that emits first-content SSE calls `root_guard.close_success(end_ns)` (per Codex review v18 P1 once-close pattern + v14 P1 #1 RAII redesign). `close_success` internally calls `close_p5h_span(&self.ctx, self.span, end_ns, SpanFields::default())`. T0a tree structural check asserts exactly one close record per (request_id, root span_id) to catch double-close bugs.
- `http_parse_render_tokenize` — runs in async handler. Per Codex review v12 P2 #2, the full `P5hTraceContext` cannot exist at span start (because `prompt_tokens` comes from `prompt_ids.len()` produced by `render_and_encode(...)`, and `routing_path` comes from the `use_scheduler` predicate at `openai.rs:404` which needs `prompt_len`). Required handler ordering:
  1. Handler entry: generate `request_id` uuid; capture `root_start_ns = monotonic_ns()` and `http_parse_render_tokenize.start_ns = monotonic_ns()`. Both timestamps captured BEFORE any parse/render/tokenize work begins.
  2. Run request parse + `render_chat_template(...)` + `tokenizer.encode(...)` to produce `prompt_ids`.
  3. Compute `prompt_tokens = prompt_ids.len() as u32` and evaluate `use_scheduler = state.prefill_chunk_size == 0 || prompt_ids.len() <= state.prefill_chunk_size` per `openai.rs:404`; set `routing_path = if use_scheduler { "scheduler" } else { "gs_chunked" }`.
  4. Construct full `ctx = P5hTraceContext { request_id, prompt_tokens, routing_path }`. Open root via `let root_span = open_p5h_span_at(&ctx, None, "server_request_recv_to_first_content_sse_write", root_start_ns);` and wrap as `let root_handle = RootSpanHandle { ctx: ctx.clone(), span: root_span.clone() };` (per Codex v14 P1 — `open_p5h_span_at` is the API that preserves the early-captured `root_start_ns`; `open_p5h_span` would use `monotonic_ns()` at call time, missing the parse/tokenize cost).
  5. Open + close the `http_parse_render_tokenize` span at its captured start: `let http_span = open_p5h_span_at(&ctx, Some(&root_span), "http_parse_render_tokenize", http_parse_start_ns); close_p5h_span(&ctx, http_span, monotonic_ns(), SpanFields::default());` — span uses the start_ns captured in step 1 (the real beginning) and the current time as end_ns (immediately after ctx construction).
  6. Write `request.p5h_trace = Some(ctx.clone())` AND `request.p5h_root_span = Some(root_span.clone())` (per Codex v15 P1) for plumbing into `RequestState` / `spawn_blocking` closure captures (Lane B) / forwarder closure captures (Lane A). Under Lane A, `prefill_admitted_inner` reads these via `Scheduler::cloned_active_row_p5h_trace_and_root` to open `model_prefill_forward` + `first_token_sampling` INSIDE the function (per Codex v16 P1 — NOT at actor scope). Lane-B `spawn_blocking` captures `root_handle` directly (it already has both `ctx` and `span`); does not need `p5h_root_span` field separately.
- `scheduler_admission` (Lane A) — wraps `cmd_tx.send(...).await + reply_rx.await` from handler to actor; the start_ns is captured in handler before the await, end_ns is captured in handler after the await; both use the same explicit ctx.
- `sse_write_role_chunk` (Lane A and Lane B) — Lane A's `tx.send(...).await` at `openai.rs:562` is async; Lane B's `tx.blocking_send(...)` at `openai.rs:455` is sync and runs AFTER `GenerationStream::new(...)` / `gs_stream_init_and_chunk_loop` has completed, but BEFORE the post-prefill `stream.next_token()` materialization loop. Use explicit ctx, captured from the spawn closure; do not enter a guard for the role-chunk emission.
- `detok_format_first_content_chunk` (Lane A and Lane B) — same reasoning. Lane A's `tx.send(...).await` at `openai.rs:589` is async; Lane B's `tx.blocking_send(...)` at `openai.rs:473` is sync but emits the root close event, which must use explicit ctx so it can be paired with the explicit RootSpanHandle.
- `first_token_sampling` (Lane A) — happens inside `sched.prefill_admitted_inner`'s post-`batched_prefill` "three-stage dispatch" (per `scheduler.rs:784`). Per Codex review v16 P1 + v17 P2 #1: `first_token_sampling` is an explicit top-level **sibling** of `model_prefill_forward`, opened/closed INSIDE `prefill_admitted_inner` (NOT at actor scope; NOT via implicit API which would chain it under `model_prefill_forward`). Vanilla case uses no guard (no deep sampling instrumentation in scope); if T4 later adds deep sampling spans, wrap with a guard using `first_token_sampling` as base_parent.

**Rationale**: `P5H_CURRENT_TRACE` thread-local is a convenience for the bulk of instrumentation (deep model spans that run inside one of the four guard sites). Top-level spans that cross task/await boundaries plumb the context as an explicit value — never read the thread-local. This makes the unsoundness pattern Codex v8 P1 #2 caught (thread-local outliving an `.await`) structurally impossible: code that needs context across an await has no thread-local API to misuse.

The `enter()` panic-on-nest is the enforcement mechanism for the implicit API: a wrongly-placed inner guard fails fast at the first sweep request instead of silently clearing the outer context on `drop`. Code that mistakenly calls `with_p5h_span_from_current_trace(...)` outside a guard region panics with the span name — same fail-fast discipline.

**Propagation chain (explicit, no thread-local across thread/await boundaries):**

1. `GenerateRequest` adds **two** fields (per Codex review v15 P1 — `p5h_trace` alone is insufficient because the scheduler actor needs the root `SpanHandle` to seed the guard's `base_parent`):
   - `p5h_trace: Option<P5hTraceContext>` (uuid + prompt_tokens + routing_path)
   - `p5h_root_span: Option<SpanHandle>` (the root span the handler opened via `open_p5h_span_at`; cloned so `Scheduler::admit` can stash it)
   Both gated by `p5h-profile` feature; default `None`. HTTP handler populates them per the § 2.5a "Server-only root" handler ordering (steps 1-6).
2. `RequestState` (in `scheduler.rs`) carries both fields; `Scheduler::admit` copies them from `GenerateRequest` at admit time. `Scheduler::prefill_admitted_inner` (NOT the actor; per Codex review v16 P1 + v17 P2 #1) reads them via the `cloned_active_row_p5h_trace_and_root` helper to open `model_prefill_forward` + `first_token_sampling` inside the function. `Scheduler::step` does the same for `pre_content_decode_steps` if T0a confirms `step` similarly fuses model-forward + sample (otherwise actor-scope wrap suffices for `step`).
3. **Lane-B `spawn_blocking`** body (`openai.rs::serve_via_gs_stream`): the closure captures **both** `ctx: P5hTraceContext` (clone of `request.p5h_trace`) AND `root_handle: RootSpanHandle` (clone of the handle the handler opened) — Rust move/borrow rules force both into the closure. Per Codex plan review v14 P1 #1 + v15 P2 #2: at closure top, wrap root via `let mut root_guard = P5hRootCloseGuard::new(root_handle);` (owning Option exposes `.span()` / `.close_success(end_ns)` / `.is_open()`; Drop runs `close_at_aborted` on any pre-first-content terminal path). All subsequent `root_handle.span()` references go through `root_guard.span()`. Inside the closure, the wall-clock order MUST be:
   1. Open `gs_stream_init_and_chunk_loop`, enter guard with that span as base_parent, run `GenerationStream::new(...)`, drop guard, close `gs_stream_init_and_chunk_loop`.
   2. Open/close `sse_write_role_chunk` around the role `tx.blocking_send(...)`.
   3. Enter the post-prefill loop; open `gs_first_token_materialize_and_predispatch` for the first `stream.next_token()` call, `pre_content_decode_steps` only for later pre-content iterations, then open/close `detok_format_first_content_chunk` and call `root_guard.close_success(end_ns)` on the first non-empty content send.

   Open the chunk-loop top-level explicit span first, THEN enter guard with that span as base_parent:
     ```
     let gs_top = open_p5h_span(&ctx, Some(root_guard.span()), "gs_stream_init_and_chunk_loop");
     {
         let _guard = P5hTraceGuard::enter(ctx.clone(), gs_top.clone());
         let stream = GenerationStream::new(...);  // deep substeps chain under gs_top
     }  // _guard drops; stack returns to empty
     close_p5h_span(&ctx, gs_top, monotonic_ns(), SpanFields::default());
     ```
     Deep substeps inside `GenerationStream::new()` (`gs_kv_cache_alloc`, `gs_chunk_N`, `gs_first_token_sample_dispatch`) use `try_with_p5h_span_from_current_trace(...)` (None-tolerant per Codex plan review v12 P1 #1 — same code path also runs from CLI/tests where no guard is active) and chain under `gs_top` via the seeded stack.
   - After `GenerationStream::new(...)` returns and `gs_top` is closed, `sse_write_role_chunk` for the `tx.blocking_send` at `openai.rs:455` is opened/closed via `let h = open_p5h_span(&ctx, Some(root_guard.span()), "sse_write_role_chunk"); tx.blocking_send(...); close_p5h_span(&ctx, h, monotonic_ns(), ..);` — explicit API, no guard.
   - The post-prefill loop opens a per-iteration top-level explicit span ONLY while root is still open (`root_guard.is_open()`), enters guard with it as base_parent, calls `stream.next_token()`, exits guard, closes the top-level span, then handles SSE emission. Once first non-empty content sends (`root_guard.close_success(end_ns)`), the per-iteration P5h emission ceases — follow-on tokens stream without any P5h tree spans. Per Codex plan review v12 P1 #2 + v14 P1 #1 + v14 P2 #4: P5h root-tree ends at TTFT (first content chunk) by design.
     ```
     loop {
         let iter_top_span = if root_guard.is_open() {
             let name = span_name_for_this_iteration;
             Some(open_p5h_span(&ctx, Some(root_guard.span()), name))
         } else {
             None
         };
         let _iter_guard = iter_top_span.as_ref().map(|s| P5hTraceGuard::enter(ctx.clone(), s.clone()));
         let ev = stream.next_token();
         drop(_iter_guard);
         if let Some(span) = iter_top_span {
             close_p5h_span(&ctx, span, monotonic_ns(), ..);
         }
         // SSE emission per event:
         //   first_non_empty_content = !ev.text.is_empty() && root_guard.is_open()
         //   if first_non_empty_content {
         //     let h = open_p5h_span(&ctx, Some(root_guard.span()), "detok_format_first_content_chunk");
         //     /* send chunk */
         //     close_p5h_span(&ctx, h, end_ns, ..);
         //     root_guard.close_success(end_ns);  // RAII Drop becomes no-op after this
         //   }
     }
     ```
     `span_name_for_this_iteration` is `gs_first_token_materialize_and_predispatch` for the first iteration; `pre_content_decode_steps` for subsequent iterations while root is still open (first detok was empty and we're looping for non-empty content). `root_guard` is the `P5hRootCloseGuard` declared at the top of the `spawn_blocking` closure (per Codex v14 P1 #1) — its Drop closes the root via `close_at_aborted(...)` if execution exits the closure with root still open (any pre-first-content terminal path).
4. **Lane-A scheduler — span sites SINK into `prefill_admitted_inner`** (per Codex review v16 P1 — actor-level wrapping is wrong because `prefill_admitted_inner` runs `model.batched_prefill[_vl](...)` AND `sample_batch(...)` in one function call, with no boundary the actor can wedge a span close + new open into):
   - **Wrong (v15/v16 pre-fix design)**: actor opens `model_prefill_forward` around the whole `sched.prefill_admitted(...)` call, then opens `first_token_sampling` after return. Problem: `prefill_admitted_inner` (per `scheduler.rs:959-1025`) runs prefill at lines 973-981 then immediately runs sampler reshape + Stage A/B at lines 996-1025, all before returning to actor. Actor-level `model_prefill_forward` would silently absorb sampling cost; `first_token_sampling` opened after return would be zero-width.
   - **Correct design**: the open/close of both spans happens INSIDE `prefill_admitted_inner`, not at actor scope. Concretely (T0a code change inside `scheduler.rs::prefill_admitted_inner`):
     ```
     // Read the lone active row's p5h_trace + p5h_root_span (under --b-max 1
     // exactly one row is active; if multiple, p5h-profile already panics at
     // startup per § 2.5a single-active-row invariant). Returns OWNED clones
     // (per Codex review v17 P1) — references borrowed from self.slots would
     // collide with the subsequent &mut self.cache / &mut self.slots /
     // &mut self.prng_state needed by batched_prefill_vl + Stage A + sample_batch.
     // Returns Ok(None) when both fields are None (per Codex v10 P1 #2 + v11 P1 #4):
     // every non-openai.rs entry path (anthropic.rs / CLI / tests / scheduler_actor
     // internals) keeps both fields None, and the SINK below quietly no-ops on None
     // so those paths keep working under --features p5h-profile.
     let p5h_trace = self.cloned_active_row_p5h_trace_and_root()?;
     // p5h_trace: Option<(P5hTraceContext, SpanHandle)>

     // Span 1: model_prefill_forward — wraps ONLY the model.batched_prefill[_vl] call.
     // Both span open and guard enter are gated on p5h_trace.as_ref(); under None the
     // SINK no-ops and the call runs exactly as the non-feature build does.
     let mpf_span = p5h_trace.as_ref().map(|(ctx, root_span)| {
         open_p5h_span(ctx, Some(root_span), "model_prefill_forward")
     });
     let logits_result = {
         let _guard = match (p5h_trace.as_ref(), mpf_span.as_ref()) {
             (Some((ctx, _)), Some(mpf)) => Some(P5hTraceGuard::enter(ctx.clone(), mpf.clone())),
             _ => None,
         };
         if is_vl { model.batched_prefill_vl(...) } else { model.batched_prefill(...) }
         // deep substeps (embed_lookup / decoder_layer_N / ... / slice_last_and_project_lm_head)
         // chain under mpf via the seeded stack (only when guard entered, i.e. Some path).
     };
     if let (Some((ctx, _)), Some(mpf)) = (p5h_trace.as_ref(), mpf_span) {
         close_p5h_span(ctx, mpf, monotonic_ns(), ..);
     }
     let logits = logits_result?;

     // Span 2: first_token_sampling — wraps the logits reshape + Stage A (sampler refs +
     // histories) + Stage B (sample_batch). NO guard needed for vanilla case (sample_batch
     // currently has no deep instrumentation candidates); if T4 adds deep sampling
     // breakdown later, wrap with a guard using fts as base_parent.
     // Per Codex plan review v12 P2 #5: capture the FULL fts body's Result, close
     // the span, THEN `?`. Earlier shape had `logits.reshape(..)?` between open
     // and close, which would leak the fts span on a reshape Err — same anti-pattern
     // v11 P2 #5 fixed for model_prefill_forward.
     let fts_span = p5h_trace.as_ref().map(|(ctx, root_span)| {
         open_p5h_span(ctx, Some(root_span), "first_token_sampling")
     });
     let tokens_result = (|| -> anyhow::Result<_> {
         let logits_bv = logits.reshape(..)?;
         let (row_samplers, row_histories) = collect_sampler_refs_and_histories(..);
         sample_batch(&row_samplers, &logits_bv, ..)
     })();
     if let (Some((ctx, _)), Some(fts)) = (p5h_trace.as_ref(), fts_span) {
         close_p5h_span(ctx, fts, monotonic_ns(), ..);
     }
     let tokens = tokens_result?;

     // Stage C distribution per row continues as today (not in the top-level tree).
     ```
   - Actor scope keeps zero P5h instrumentation around `sched.prefill_admitted(...)` itself — actor just calls it.
   - `pre_content_decode_steps`: actor opens this explicit span around any `sched.step(...)` calls needed for pre-content decode, but per § 2.5a authorized list, the actor-level guard inside that block opens via the same SINK pattern — actually `step` is simpler since it doesn't fuse prefill + sampling, so an outer guard around `sched.step(...)` with `pre_content_decode_steps` as base_parent works. (Alternative if `step` itself fuses model-forward + sample like `prefill_admitted_inner`, the same SINK pattern applies: open subspans inside `step` rather than at actor scope. T0a verifies which.)
5. **Lane-A streaming forwarder** (`openai.rs:546` `tokio::spawn` body): handler clones BOTH `ctx` AND `root_handle` into the spawn closure. Per Codex plan review v14 P1 #1 + v15 P2 #2: at closure top, wrap root via `let mut root_guard = P5hRootCloseGuard::new(root_handle);`. All `root_handle.span()` references in the snippets below go through `root_guard.span()`. The closure **never calls `P5hTraceGuard::enter(...)`** — Lane-A forwarder is not on the authorized guard sites list (per Codex review v10 P1 #2 + v11 P1 #2: `tx.send(...).await` is async, a guard around it would have to span the await, which is the v8 unsoundness pattern). All SSE emission uses the explicit-context API:
   - For `sse_write_role_chunk_diagnostic` (per Codex v18 P2 #3 + v19 P1 — Lane-A only; emitted with `span_kind="diagnostic"`, NOT in exclusive tree): `let h = open_p5h_span(&ctx, Some(root_guard.span()), "sse_write_role_chunk_diagnostic"); format_sse(...); tx.send(...).await; close_p5h_span(&ctx, h, monotonic_ns(), ..);` — span end_ns captured immediately after `.await` returns. Emitter sets `span_kind = "diagnostic"` per § 2.5a server-emitted fields table (fixed set of diagnostic span_names).
   - Per content-event iteration (outer scope holds `let mut root_guard = P5hRootCloseGuard::new(root_handle_clone);` per Codex plan review v14 P1 #1): per spec § 2.5a Lane-A bucket 7, `detok_format_first_content_chunk` MUST cover `detok stream step + ChunkResponse serialize + first content SSE write` — the span starts BEFORE detok runs, not after. Per Codex plan review v18 P2 #4 + v19 P2 #4: capture `detok_start_ns = monotonic_ns()` BEFORE `detok.step(ev.token)`; run detok; only when the resulting `text` is first non-empty content AND root is still open, retroactively open the span via `open_p5h_span_at(&ctx, Some(root_guard.span()), "detok_format_first_content_chunk", detok_start_ns)`. Otherwise emit no span and continue iterating.
     ```
     let detok_start_ns = monotonic_ns();
     let text = match detok.step(ev.token) { /* Ok/Err */ };
     let first_non_empty_content = !text.is_empty() && root_guard.is_open();
     if first_non_empty_content {
         let h = open_p5h_span_at(&ctx, Some(root_guard.span()),
                                  "detok_format_first_content_chunk", detok_start_ns);
         format_sse(...);
         tx.send(...).await;
         let end_ns = monotonic_ns();
         close_p5h_span(&ctx, h, end_ns, ..);
         root_guard.close_success(end_ns);
     }
     ```
     All explicit, no `P5hTraceGuard` guard around the send. After `close_success`, `root_guard.is_open()` returns false and subsequent iterations emit no P5h spans. Drop of `root_guard` at closure exit fires `close_at_aborted(monotonic_ns())` if root is still open (any pre-first-content terminal path: role-send fail, detok err, event_rx end, async cancel). Per Codex v18 P1 once-close pattern + v14 P1 #1 RAII redesign + v18 P2 #4 / v19 P2 #4 detok coverage.
6. Deep instrumentation sites (decoder layer body, GDN/GatedAttention/MoE substeps, lm_head span) wrap their work via `try_with_p5h_span_from_current_trace(span_name, fields_fn, body)` (None-tolerant variant added per Codex plan review v12 P1 #1, made route-aware per Codex plan review v20 P1 #1 + v21 P2 #3). The helper resolves to exactly one of four cases based on the active `P5H_CURRENT_TRACE` and (when present) its `routing_path`:

   1. **No active trace** (`P5H_CURRENT_TRACE == None`) — non-OpenAI entry paths (anthropic.rs / CLI / tests / `prefill_admitted_inner` SINK with `None` row): run body directly with no span emission. Required because `model.batched_prefill[_vl](...)` still flows through `DecoderLayerMoe::forward_on` → GDN/GatedAttention/MoE on these entry paths.
   2. **Lane A** (`routing_path == "scheduler"`) — forward to the strict `with_p5h_span_from_current_trace(...)`; emit every span_name the deep instrumentation reaches. Full per-substep attribution.
   3. **Lane B** (`routing_path == "gs_chunked"`) — emit ONLY for span_names in the compile-time const `LANE_B_ALLOWED_TRY_SPAN_NAMES = {"gs_kv_cache_alloc", "gs_chunk_N", "gs_first_token_sample_dispatch"}` (the three top-level chunked-GS substeps inside `GenerationStream::new(...)`); any other span_name (decoder_layer_N, gda_step_*, q_gate_*, router_logits_*, slice_last_*, mlx_eval_barrier, etc.) no-ops. The guard remains active for stack-seeding so the three allow-listed names chain under `gs_stream_init_and_chunk_loop`, but deep decoder/GDN/MoE/lm_head substep emission is suppressed. Lane-B chunked GS is top-level-only in P5h per spec § 5; per-chunk substep attribution is deferred to P5h+1 (needs `chunk_idx` schema extension to disambiguate N records per request).
   4. **Unknown routing_path** — panic with span_name in the message. Only the two values in case 2 + case 3 are legal per `P5hTraceContext.routing_path`. Any other value is an emitter bug and must fail fast.

   Sampling spans (`first_token_sampling`) are top-level explicit, not deep — they use the explicit-context `open_p5h_span` API in the vanilla case (per Codex v17 P2 #1).

**Async-safety rationale** (replaces v6/v7 "axum handler → scheduler driver same thread chain" claim, per Codex v8 P1 #2): Tokio worker threads may execute the same async future on different OS threads between `.await` points. `thread_local!` is not request-local OR task-local. The correct discipline is: (a) plumb the context explicitly across thread/task boundaries as a clonable value; (b) only enter the thread-local guard inside synchronous regions where the executing thread is pinned. This mirrors the pattern Tokio's own `tracing` crate uses for span attachment.

**Hard-fail under `p5h-profile` + `--b-max 1`** (per Codex review v7 P3 + v10 P2):

- The strict `with_p5h_span_from_current_trace(...)` variant — reserved for sites where ctx is provably populated — MUST `panic!` with the span_name in the message when `P5H_CURRENT_TRACE` is None. This catches missing/wrong guard set/drop sites on the implicit-API path. The None-tolerant `try_with_p5h_span_from_current_trace(...)` variant (added per Codex plan review v12 P1 #1, used by ALL deep callers) instead runs the body directly when no trace is active, so non-OpenAI entry paths under `--features p5h-profile` still execute. Either way, an unbalanced `P5H_CURRENT_SPAN_STACK` at any open/close transition (stack would underflow on pop, or parent on stack doesn't match the expected enclosing context) MUST panic on the strict path; on the `try_` path the stack is untouched when no trace is active.
- Any `open_p5h_span[_at](...)` call with a default-constructed / empty `P5hTraceContext` MUST `panic!` with the span_name. Likewise `close_p5h_span(...)` MUST panic if the passed `SpanHandle.span_id` does not match the open record (catches handle reuse / cross-request leakage). This catches forgotten ctx-plumbing and handle misuse on the explicit-API path.
- T0a verifies via the route-aware fixture documented in § 3 T0a — per-record field validation, route-aware schema presence check (all required top-level + root spans emitted per lane), and parent-child tree closure check. A first-record-only check is insufficient (v10 P2 — would pass even if top-level async spans never emit).

**Single-active-row hard gate** (per Codex review v5 P1 #2): the design ONLY works if exactly one in-flight row exists during `prefill_admitted_inner` + any pre-content decode steps. P5h harness MUST start the server with `--b-max 1` (production default per `serve.rs:38`; P5h enforces). On `p5h-profile` feature, server panics at startup if `b_max > 1` OR if the scheduler ever observes `active_count() > 1` during a `[p5h-profile]`-emitting forward. Strict serial sweep (per memory `[feedback_serial_perf_experiments]`) also enforces only one in-flight request server-side.

**Streaming-only scope** (per Codex plan review v16 P1 #2 + v17 P1 #1): P5h instrumentation targets TTFT (time-to-first-content-token) on the streaming SSE path. Only `chat_completions` requests with `req.stream == true` (where `req: ChatRequest` is the handler's `Json(req)` extraction; `ChatRequest.stream` is a plain `bool` with `#[serde(default)]`, NOT `Option<bool>`) open the root span, populate `GenerateRequest.p5h_trace` / `p5h_root_span`, emit the `X-Ironmlx-Request-Id` header, and fire the `[p5h-profile]` log records. Non-streaming (`req.stream == false`) requests skip all P5h side effects entirely — the `serve_via_*_unary` paths have no root terminal (no first-content SSE write), so opening a root in `chat_completions` and dispatching to a unary path would leak the root span in `OPEN_SPAN_REGISTRY`. The iron-bench `--capture-server-request-id` flag is correspondingly only meaningful for streaming requests. P5h sweeps use streaming exclusively.

**Admission early-return root cleanup** (per Codex plan review v16 P1 #2): the Lane-A admission flow (`cmd_tx.send` → `reply_rx.await` → `AdmitReply` match) has three error branches that all `return` BEFORE the forwarder `tokio::spawn`. The forwarder is where `P5hRootCloseGuard` would normally pick up abort cleanup. When admission fails, `serve_via_scheduler_stream` MUST close the `scheduler_admission` span AND explicitly call `RootSpanHandle::new(ctx_clone, root_span_clone).close_at_aborted(admission_close_end_ns)` before returning the error response. Otherwise the root opened in `chat_completions` (Step 3 of § 2.5a Server-only root handler ordering) leaks indefinitely.

**Server-emitted fields** (per `[p5h-profile]` log line, written by ironmlx):

**Field source per emission API** (per Codex review v11 P2): every record's `request_id` / `routing_path` / `prompt_tokens` come from the **active `P5hTraceContext`**, but how that context reaches the emitter depends on the API:

- `open_p5h_span[_at](&ctx, ...)` + `close_p5h_span(&ctx, ...)` (explicit-context API — root + http_parse_render_tokenize + scheduler_admission + sse_write_role_chunk + detok_format_first_content_chunk): reads fields directly from the `&ctx` argument. **Never reads `P5H_CURRENT_TRACE`.**
- `with_p5h_span_from_current_trace(...)` (strict, panics if no active guard) / `try_with_p5h_span_from_current_trace(...)` (None-tolerant; runs body directly if no active guard — used by ALL deep callers per Codex plan review v12 P1 #1) — implicit-guard API for deep sync spans inside an authorized guard region: reads fields from `P5H_CURRENT_TRACE` thread-local (populated by the active `P5hTraceGuard`) and parent from `P5H_CURRENT_SPAN_STACK` top. When no guard is active (non-OpenAI entry: anthropic.rs / CLI / tests), `try_` emits nothing.

The underlying value is identical in both cases (`GenerateRequest.p5h_trace` is the canonical source, plumbed via explicit clone to handler ctx and via `P5hTraceGuard::enter(ctx.clone())` to thread-local).

| Field | Type | Semantics |
|---|---|---|
| `request_id` | string (uuid) | server-generated per request, T5 group-by key; sourced per the dual-API rule above |
| `routing_path` | string | "scheduler" \| "gs_chunked"; sourced per the dual-API rule above; needed so T5 can partition records into Lane A vs Lane B |
| `prompt_tokens` | int | server-measured (post-chat-template, post-tokenize); sourced per the dual-API rule above; proxy for iron-bench `--prompt-len` once correlated |
| `seq` | int | sequence length at this forward (chunk size or 1 for decode) |
| `layer_idx` | int | GDN/full-attn layer 0..39 (-1 for non-decoder spans) |
| `span_id` | u64 | **stable unique id per emitted span instance** (per Codex review v12 P1 — `span_name` alone cannot identify a span instance because `attention_path`/`mlp_path`/GDN/GatedAttention/MoE substep names repeat across 40 decoder layers + N chunks under Lane B + N pre-content-decode iterations). Generated by the emitter at span open; recommended impl = `(request_id_hash ^ atomic_counter)` or just `atomic_counter` since `(request_id, span_id)` is the global key. **The T5 aggregator MUST build the exclusive tree from `span_id`/`parent_span_id` — NOT from `span_name`/`parent_span` strings.** |
| `parent_span_id` | u64 \| null | id of the lexically-enclosing span (null = root). Set by the emitter from the active enclosing span's `span_id` (explicit API: passed via `SpanHandle`; implicit API: read from a per-thread `P5H_CURRENT_SPAN_STACK: RefCell<Vec<SpanHandle>>` maintained alongside `P5H_CURRENT_TRACE`, pushed by the emitter on span open + popped on span close). |
| `span_name` | string | 'http_parse_render_tokenize' / 'scheduler_admission' / 'attention_path' / 'gda_step_1a_in_proj_qkvz' / etc. — human-readable label retained for log readability + § 4 schema-presence validation + ROI ranking grouping; NOT used to identify span instances or rebuild tree (per Codex v12 P1). |
| `parent_span` | string \| null | human-readable parent label retained for log readability; NOT used to rebuild tree. T5 fixture asserts `(parent_span_id is None) == (parent_span is null)` and `parent_span_id resolves to a span_name string equal to parent_span` as a self-consistency check. |
| `start_ns` | u64 | monotonic clock start (ns) |
| `end_ns` | u64 | monotonic clock end (ns) |
| `mode` | string | Span-time mode tag: `'off'` (default) / `'layer1'` / `'layer2'` / `'ablate-X'` (P5g compatibility) / `'aborted'` (set on the ROOT span by `RootSpanHandle::close_at_aborted` when a pre-first-content terminal path closes the request — Lane-A role-send fail, Lane-B `GenerationStream::new` Err, Lane-B role-send fail, `stream.next_token` Err / Ok(None), empty-content + finish_reason break, detok Err, panic-in-spawn_blocking; see Codex plan review v12 P2 #6 + v13 P1 #1). The aggregator + T0a.14 verifier exclude requests whose root carries `mode="aborted"` from coverage + structural gates. |
| `span_kind` | string | `"tree"` (default, participates in exclusive parent-child tree + coverage gate + structural checks) \| `"diagnostic"` (per Codex review v19 P1 — emitted as a recorded interval for T5 reporting but **excluded** from tree build, exclusive_us computation, sum-to-root invariant, coverage_pct, reachability/cycle checks, and T5-synthesized residual rows such as `unattributed_server_root`. Used for spans whose execution overlaps a tree span across threads/tasks, e.g., Lane-A `sse_write_role_chunk_diagnostic` overlapping `model_prefill_forward` on different threads.) The set of `span_kind="diagnostic"` span_names is fixed and enumerated in § 2.5a per-lane bucket lists. Any other raw emitted span_name MUST emit `span_kind="tree"`. |

**Aggregator-injected fields** (added by T5 Python aggregator when joining server log records with the iron-bench client sweep CSV):

| Field | Type | Source |
|---|---|---|
| `pp` | int | iron-bench `--prompt-len` for the sweep cell that produced this record; sourced by `(request_id)` join into iron-bench CSV (per Codex review v12 P3 — `request_id` is the sole join key per "Join key" subsection below; v4 timestamp-window fallback was dropped in v5 and MUST NOT reappear) |
| `run_id` | int | iron-bench warmup/measured run index within the sweep cell; same `(request_id)` join |
| `bench_session_id` | string | optional T5 group-by for multi-session sweeps; same `(request_id)` join |

**Join key — single committed path (per Codex review v4 P2)**: prior v4 wording left "header OR wallclock fallback" as implementer choice. Both paths were under-specified (header path didn't actually exist end-to-end; wallclock path required a `server_request_start_wallclock` field that wasn't in the server-emitted schema). v5 commits to ONE concrete path; T0a delivers all three edits below or T0a does not close:

1. **server emit** (`openai.rs`, gated on `p5h-profile` feature): chat-completion response builder MUST set response header `X-Ironmlx-Request-Id: <uuid>` (same uuid that anchors every `[p5h-profile]` log record's `request_id` field). Per Codex plan review v16 P1 #2 + v17 P2 #3: emitted ONLY on streaming paths (`serve_via_scheduler_stream`, `serve_via_gs_stream`) — non-streaming `serve_via_*_unary` paths skip P5h entirely (no root, no header). Small addition (~5 lines) to each of the two streaming response constructions.
2. **iron-bench capture** (`iron-bench/src/client.rs` + `iron-bench/src/report.rs`): both capture AND serializer schema gated on new `--capture-server-request-id` CLI flag (default off). Per Codex plan review v18 P2 #5 + v19 P2 #3 — scope is **CSV-only**; `render_markdown` and `render_json` are untouched (the T5 aggregator reads CSV via `csv.DictReader`, so JSON output gaining `request_id` is wasted work and creates a second source of truth):
   - **Flag off** (default, non-P5h runs): `RequestResult.request_id` not set; `render_csv` header + body emit zero `request_id`-related bytes; CSV output is **byte-identical** to current iron-bench (per Codex review v5 P2 #3 + v19 P1 #2 — verified via deterministic in-memory golden test on `render_csv(&[fixture], false)`, NOT two live CLI runs).
   - **Flag on** (P5h sweeps): `run_chat_completion` captures `X-Ironmlx-Request-Id` from `resp.headers()` BEFORE entering `resp.bytes_stream()`, populates `RequestResult.request_id: Option<String>`; `render_csv` adds `request_id` column to CSV header + row tail (appended after existing `finish_reason` to keep prior-column byte ordering intact for tools that read by name). `render_json` and `render_markdown` are unchanged in both flag states.
3. **aggregator join** (T5 Python): `(request_id)` is the sole join key — no wallclock fallback. T5 aggregator hard-fails if any P5h sweep cell shows < 100% 1:1 server↔client request_id match. Orphan rate > 0% per PP fails T5 gate (the gate threshold was "< 1%" in v4 — v5 tightens to "= 0%" since deterministic header propagation should never lose records under per-PP serial sweep).

**Wallclock fallback explicitly DROPPED**: v4 listed wallclock as fallback. Codex v4 P2 correctly noted server-emitted schema had no `server_request_start_wallclock` field. Rather than add a fragile fallback, v5 makes the header path the only path. Boss memory `[feedback_iron_bench_priority]` cautions against modifying iron-bench casually — the 3 edits above are scoped, behind a new CLI flag, and ship/revert independently.

**Routing precondition — dual-lane design** (per Codex review v5 P1 #1 + v6 P1 fact-check):

v5 proposed forcing `--prefill-chunk-size 0` to keep all PP on the scheduler path. v6 review correctly flagged this would divorce P5h from the production-default config (P5g actually measured the long-PP gap on the chunked GS path — see `reports/p5g-final-results.md:62`: "PP=4096 → 3 chunks (two 2048 + one 12); PP=16384 → 9 chunks"). Forcing single-shot would test a config production users never hit, and the resulting attribution would not transfer to the default-config gap that motivates P5h.

v7 switches to **dual-lane with production-default server config preserved** (no `--prefill-chunk-size` override):

- **Lane A — Scheduler path, PP ≤ default `prefill_chunk_size` (2048)**: PP ∈ {128, 512, 2048} routes through `serve_via_scheduler_stream` (per `openai.rs:404` — `prompt_len <= prefill_chunk_size` predicate). Full § 2.5a deep substep attribution applies (GDN/GatedAttention/MoE substep breakdown via wrapper spans `attention_path`/`mlp_path`). T0a/T1/T2/T3/T4 deep instrumentation is meaningful on this lane.
- **Lane B — Chunked GS path, PP > default `prefill_chunk_size`**: PP ∈ {4096, 8192, 16384} routes through `serve_via_gs_stream` (per `openai.rs:408`). Each request emits N chunks (PP=4096 → 3 chunks, PP=16384 → 9 chunks). P5h covers ONLY top-level chunked-path attribution: server-side root + `gs_stream_init_and_chunk_loop` wall-time (including `gs_kv_cache_alloc`, repeated `gs_chunk_N`, and `gs_first_token_sample_dispatch`) + post-`new()` role SSE + first-token materialization/predispatch + first-content SSE formatting/write. Deep substep attribution (per-chunk GDN/GatedAttention/MoE breakdown, with `chunk_idx` schema extension) is **out of scope for P5h** (deferred to P5h+1) — would multiply records per request by N and require schema additions.

T0a profile gate validates per-PP routing via a `routing_path: "scheduler" | "gs_chunked"` annotation on the root span; T5 aggregator partitions output into two attribution tables (Lane A deep, Lane B top-level). Long-PP P5j candidate ranking comes with explicit caveat: P5j ROI estimates on PP > 2048 are bounded by Lane-B granularity; if a P5j candidate needs per-substep evidence at long PP, P5h+1 chunked deep-attribution must run first.

**Server-only root** (per Codex review v2 P1 #2 + v3 P1 fact-check; iron-bench TTFT cross-process correlation deferred — would require iron-bench → ironmlx request-id propagation, out of P5h scope):

- **Root span**: `server_request_recv_to_first_content_sse_write` — from server's `axum` request-handler entry to the moment when the **first non-empty `delta.content` SSE chunk** is sent into the body channel. All `[p5h-profile]` records anchor under this server-side root.
- **Why not "first SSE write" (Codex v3 P1)**: the forwarder task spawned right after `AdmitReply` issues a synthetic role chunk (`delta.role = "assistant"`, `delta.content = ""`) BEFORE the first-batch prefill runs (per `openai.rs:546-564` + `scheduler_actor.rs:276-313`: admit reply is sent before prefill). If root ended at "first SSE write", prefill + first-token sampling would fall **outside** the root, defeating the entire attribution exercise. The role chunk write is emitted as a diagnostic span (`sse_write_role_chunk_diagnostic` under Lane A — see top-level bucket list + Codex v18 P2 #3 for why Lane-A's is diagnostic not exclusive; `sse_write_role_chunk` under Lane B is a proper exclusive child since Lane-B execution is sequential inside `spawn_blocking`).
- **Root terminal definition — lane-specific callsites** (per Codex review v9 P2 #2; both lanes share semantics, different code paths):
  - **Lane A (scheduler path)**: `end_ns = monotonic_ns()` captured at the instruction immediately after the first successful `tx.send(Ok(format_sse_data(&chunk)))` in `openai.rs::serve_via_scheduler_stream`'s detok loop (`openai.rs:589` area) where `chunk.choices[0].delta.content` is non-empty.
  - **Lane B (chunked GS path)**: `end_ns = monotonic_ns()` captured at the instruction immediately after the first successful `tx.blocking_send(Ok(format_sse_data(&chunk)))` in `openai.rs::serve_via_gs_stream`'s `spawn_blocking` body loop (`openai.rs:473`) where `chunk.choices[0].delta.content` is non-empty. (Lane B uses `blocking_send` not `send` because the closure is inside `spawn_blocking`.)
- **`finish_reason` is irrelevant** (both lanes): if the model's first token simultaneously carries a stop/length finish reason (e.g., `max_tokens=1` request, or first-token sentinel), the chunk still closes the root, because iron-bench TTFT counts that same chunk (per `iron-bench/src/client.rs:106-116` — TTFT only inspects `delta.content` non-empty, not `finish_reason`). Excluding finish-reason chunks would create a server-side root that is wider than client-side TTFT — wrong direction. Empty-content role/keepalive chunks still do NOT close the root.
- **Implementation note**: the `detok_format_first_content_chunk` span should still record `finish_reason_present: bool` as an annotation (useful diagnostic), but this annotation does NOT gate root closure.
- **Client transport residual** is computed as a SEPARATE diagnostic: `client_transport_residual_us = iron_bench_ttft_us - server_root_inclusive_us`. Not part of the exclusive tree; reported alongside in `reports/p5h-attribution.md` as a transport-overhead column.

**Top-level buckets under `server_request_recv_to_first_content_sse_write`** (mutually exclusive children; Lane-A scheduler-path schema — Lane-B chunked-GS schema in the dedicated subsection below):

1. `http_parse_render_tokenize` (server-side request parsing + chat template + tokenizer Encode)
2. `scheduler_admission` (admit queue + slot allocation + batch construction; ends at `AdmitReply` send)
3. `sse_write_role_chunk_diagnostic` (Lane A — **diagnostic span, `span_kind="diagnostic"`, NOT in the exclusive tree**; per Codex review v18 P2 #3 + v19 P1): the role chunk is written by the streaming forwarder task (`tokio::spawn` body) on a different thread than the actor running `sched.prefill_admitted(...)`. After the actor sends `AdmitReply` (`scheduler_actor.rs:276`), it immediately proceeds into `sched.prefill_admitted(...)` (`scheduler_actor.rs:302-307`) — at the same time, the handler-side forwarder spawn receives `AdmitReply` and writes the role chunk. The two events can overlap in wall-clock time, so a mutually-exclusive sibling structure would compute negative residual and trip the `exclusive_us ≥ -1µs` hard invariant. v19 emits Lane-A role chunk with `span_kind="diagnostic"`: T5 aggregator filters it out of `tree_spans` before any exclusive_us / sum-to-root / coverage_pct / reachability / interval-containment computation per § 2.5a pseudocode. It is reported as a separate `role_chunk_diagnostic_us` column in T5 output. Its time does NOT fall under `unattributed_server_root` (which T5 synthesizes from raw tree spans only — v19 corrected an earlier wording that said otherwise). Lane B does NOT have this issue — its `sse_write_role_chunk` runs sequentially inside `spawn_blocking` after `GenerationStream::new(...)` returns, so it remains a true `span_kind="tree"` exclusive child there.
4. `model_prefill_forward` (the full `model.batched_prefill(...)` call — embed + 40 decoder layers + final norm + `slice_last_and_project` lm_head; per `qwen3_5_moe/model.rs::batched_prefill` lines 240-258, this single call covers everything through producing first-token logits)
5. `first_token_sampling` (per-row sampler invocation after `batched_prefill` returns; per `scheduler.rs::prefill_admitted_inner` "three-stage dispatch")
6. `pre_content_decode_steps` (per Codex review v6 P2 #2 — if detok returns `Ok(None)` or empty string for the first prefill token, server does not send a content chunk yet and iron-bench does not record TTFT; scheduler may then run additional `Scheduler::step()` decode forwards + sample + detok until detok yields a non-empty string. This bucket covers all such pre-first-content decode iterations. Expected `inclusive_us == 0` for well-formed benchmark prompts where the first prefill token detokenizes to a visible character; if non-zero, T0a/T5 must surface it.)
7. `detok_format_first_content_chunk` (detok stream step + ChunkResponse serialize + first content SSE write — for the iteration that actually produces non-empty content, whether that came from prefill or from a pre-content decode step)
8. `unattributed_server_root` (T5-synthesized residual output row — NOT a raw server-emitted `[p5h-profile]` record; see "Residual leaves" below)

**Lane-B chunked-GS top-level buckets** (per "Routing precondition" — PP > `prefill_chunk_size`):

Lane-B uses a shallower tree (no deep substep nesting in P5h; deferred to P5h+1). **Bucket ordering MUST match actual `serve_via_gs_stream` flow** (per `openai.rs:416-470` + `generate.rs:945-1055` — per Codex review v7 P1): the chunked prefill loop lives inside `GenerationStream::new()`, which runs BEFORE the role chunk SSE is sent. The Lane-B order differs from Lane A here (Lane A's role chunk happens before prefill via the scheduler `AdmitReply` forwarder spawn; Lane B's role chunk happens after the whole prefill loop finishes because `GenerationStream::new()` is synchronous inside `spawn_blocking`). Carrying the Lane-A order over to Lane B (as v7 spec did, now fixed in v8) would leak the entire chunked prefill into `unattributed_server_root` and break long-PP attribution — the exact opposite of what P5h needs.

Children of `server_request_recv_to_first_content_sse_write` under Lane B, in actual wall-clock order (per `openai.rs:416-470` + `generate.rs:1085-1370` — per Codex review v8 P1 #1, the first-token sampling is done INSIDE `GenerationStream::new()`, NOT by the first `stream.next_token()` call; v7/v8's sibling `first_token_sampling` was misplaced):

1. `http_parse_render_tokenize` (same as Lane A — runs in the axum handler before `spawn_blocking`)
2. `gs_stream_init_and_chunk_loop` (the entire `GenerationStream::new(...)` call inside `spawn_blocking` — covers KV cache allocation + chunked prefill loop body + final chunk's full forward producing first-token logits + first-token sample dispatch). Children of this bucket:
   - `gs_kv_cache_alloc` (`model.make_cache(...)` call)
   - `gs_chunk_N` × ceil(prompt_len / prefill_chunk_size) (each chunk covers `[forward_text_hidden + cache update + eval]`; the final chunk covers `[batched_prefill-equivalent forward + lm_head]` producing first-token logits)
   - `gs_first_token_sample_dispatch` (the `sample_async_greedy(&last_logits)` + `async_eval(&[&pending])` calls at `generate.rs:1097-1098` for the pipelined path, OR the synchronous `request.sampler.sample(&last_logits, ...)` call at `generate.rs:1123-1125` for the non-pipelined path. In the pipelined case this only **dispatches** the GPU work — actual GPU completion happens later, observable when the materialize step waits.)
   - `unattributed_gs_stream_init_and_chunk_loop` (T5-synthesized residual output row — NOT a raw server-emitted `[p5h-profile]` record)
3. `sse_write_role_chunk` (post-`GenerationStream::new` role chunk send — `openai.rs:441-457`)
4. `gs_first_token_materialize_and_predispatch` (the first `stream.next_token()` call — per `generate.rs:1319-1366`, in pipelined path this does THREE things: (a) `pending.item()?` waits on the pre-dispatched GPU sample to land — this is where GPU sample latency actually shows up if it wasn't already hidden behind chunk loop; (b) detok + termination check; (c) if not terminating, builds + async_eval-dispatches the next decode step's sample. Even in the unit case where first-token detok is non-empty, this span covers all three sub-activities — they execute synchronously inside the same `next_token()` call before its return enables the SSE write that closes root. T5 aggregator notes whether (c) ran by checking `finish_reason == None` on the first-content event.)
5. `pre_content_decode_steps` (only emitted when first-token detok was empty/None and additional `stream.next_token()` iterations ran before yielding non-empty content; same semantics as Lane A bucket 6. Expected `inclusive_us == 0` for well-formed benchmark prompts.)
6. `detok_format_first_content_chunk` (only the SSE format + `tx.blocking_send` for the iteration that yielded non-empty content — the `next_token()` work itself was already accounted for in bucket 4 or bucket 5)
7. `unattributed_server_root` (T5-synthesized residual output row — NOT a raw server-emitted `[p5h-profile]` record)

**Lane-B root closure invariant** (per Codex review v7 P1): the root span MUST cover the entire `GenerationStream::new(...)` wall-time. Root start = axum handler entry; root end = first non-empty content SSE write (per "Root terminal definition" above). Implementation MUST NOT close the root on role chunk send under Lane B, or `gs_stream_init_and_chunk_loop` falls outside the tree.

**Lane-A `first_token_sampling` semantics preserved**: the Lane-A top-level bucket `first_token_sampling` (bucket 5 above) IS valid — Lane A goes through `scheduler.prefill_admitted_inner`, which per `scheduler.rs:784` runs first-token sampling as a distinct "three-stage dispatch" step AFTER `model.batched_prefill(...)` returns logits (different from Lane B's pipelined-greedy GenerationStream flow where sample-dispatch is fused inside `new()`).

**T0a/T5 gate for pre_content_decode_steps — hard-gate first prefill token detok non-empty** (per Codex review v6 P2 #2 + v17 P2 #2):

The cross-thread question "did detok yield non-empty content yet?" is decidable only inside the Lane-A streaming forwarder task / Lane-B `spawn_blocking` body — the scheduler actor cannot observe detok state without a new cross-thread state-sync mechanism (additional design surface; out of P5h scope per Codex v17 P2 #2). Rather than build that sync, P5h hard-gates the upstream invariant:

- **T0a fixture HARD GATE**: every measured P5h sweep request MUST satisfy `pre_content_decode_steps` records count == 0 per request (i.e., the very first prefill token detokenized to a non-empty string, so the forwarder/blocking-task emitted `detok_format_first_content_chunk` on its first iteration without falling into the decode loop). If T0a observes any `pre_content_decode_steps` record on the benchmark prompts, T0a HARD GATE FAILS and either:
  - (a) T0a swaps the benchmark prompts for ones whose first prefill token is guaranteed to decode to a visible character (e.g., prompts ending with a clear question mark or whose model output reliably begins with a printable token; verifiable by a one-shot smoke run before the sweep), OR
  - (b) T0a documents the prompt-output combination as requiring cross-thread first-content state sync, which becomes P5h+1 scope (`actor ↔ forwarder ↔ blocking-task` synchronization channel for `first_content_emitted: bool`). Do NOT proceed with `pre_content_decode_steps` instrumentation under (b) — the P5h scope ends.
- **T5 gate**: same `pre_content_decode_steps.count == 0` per request as T0a. Any non-zero count post-T0a (e.g., introduced by a Tn-side regression in prompt selection) fails T5.
- Forwarder/blocking-task implementation hint: detok yields `Ok(None)` or empty string detection happens at `openai.rs:570-578` (Lane A) / equivalent in `spawn_blocking` body (Lane B). Both sides are local-scope, so emitting `pre_content_decode_steps` is technically possible — but its presence triggers the T0a fail-fast above before any sweep data is consumed.

**`model_prefill_forward` children** (mutually exclusive; matches `batched_prefill` call chain):
- `embed_lookup` (token id → hidden_states; `text.embed_on(...)`)
- `decoder_layer_{0..39}` × 40 (one span per decoder layer inside `forward_post_embedding_on`)
- `final_norm_in_text_model` (the model-level RMSNorm before lm_head, inside `forward_post_embedding_on` tail)
- `slice_last_and_project_lm_head` (`slice_last_and_project` — slicing + `lm_head.forward_on`)
- `unattributed_model_prefill_forward` (T5-synthesized residual output row — NOT a raw server-emitted `[p5h-profile]` record)

**`decoder_layer_N` children** (mutually exclusive):
- `input_norm` (pre-attention RmsNorm)
- `attention_path` — wrapper span for GatedAttention OR GatedDeltaNet (whichever this layer is, per config `layer_types[N]`). The substep breakdown lives **under** `attention_path`, NOT directly under `decoder_layer_N`.
- `post_attention_norm` (post-attention RmsNorm)
- `mlp_path` — wrapper span for SparseMoeBlock OR shared LinearMLP (per config `mlp_only_layers`). The substep breakdown lives **under** `mlp_path`, NOT directly under `decoder_layer_N`.
- `residual_overhead` (the two residual adds + any layout shuffle around them)
- `unattributed_decoder_layer_N` (T5-synthesized residual output row, only present in T5 output if non-zero — NOT a raw server-emitted `[p5h-profile]` record)

**`attention_path` (GatedAttention) children** = 7 substeps per § 2.2 #5 code-backed taxonomy; substep records MUST set `parent_span_id = <this attention_path span's id>` (with label `parent_span = "attention_path"`). Per Codex review v4 P1: substeps must NOT set parent to `decoder_layer_N`'s span, otherwise the wrapper span goes empty and `attention_path` exclusive_us becomes its full inclusive_us, double-counting under coverage gate. Per Codex review v12 P1: tree is rebuilt from `parent_span_id`, not the `attention_path` string label (the same label appears in every full-attn decoder layer).
**`attention_path` (GatedDeltaNet) children** = P5g T0 11-step breakdown; substep records MUST set `parent_span_id = <this attention_path span's id>` (label `parent_span = "attention_path"`). P5g instrumentation must be **extended** to emit P5h span-schema fields including `span_id` + `parent_span_id` (per § 2.5a server-emitted table). NOT `decoder_layer_N` as v3/v4 incorrectly said.
**`mlp_path` (SparseMoeBlock) children** = 8 substeps per § 2.2 #6 code-backed taxonomy; substep records MUST set `parent_span_id = <this mlp_path span's id>` (label `parent_span = "mlp_path"`). Same Codex v4 P1 reasoning + v12 P1 id-based tree rebuild.

**Residual leaves**:

Raw server logs MUST NOT emit `unattributed_*` `[p5h-profile]` records. Residual is a T5 aggregator output concept: after raw tree structural validation passes, T5 MUST synthesize at most one `unattributed_<span_name>` output row for each non-leaf raw tree span whose `inclusive_us - Σ raw_child.inclusive_us > 1µs`. If the residual is `0` (within ±1 µs noise), no synthesized row is emitted. The synthesized residual row is a leaf in the attribution output (no further children), and counts as **NOT-accountable** in the coverage gate (see § 7.1).

This explicit-residual pattern is what makes the coverage gate non-trivial:
- Without it: `Σ exclusive_us = root.inclusive_us` by tree identity (Codex P1 #1 — trivially passes even when no useful attribution emitted).
- With it: `coverage_gate = 1 - (Σ synthesized_unattributed_*.inclusive_us / root.inclusive_us) ≥ 95%`. If instrumentation only emits the root, T5 synthesizes `unattributed_server_root.inclusive_us = root.inclusive_us`, coverage = 0%, gate FAILS loudly.

**Exclusive time computation** (T5 aggregator):

```python
# Pseudocode (per Codex review v19 P1 — diagnostic spans are EXCLUDED from
# every tree-property computation; they are reported as separate columns only):
tree_spans = [s for s in spans if s.span_kind == "tree"]
diagnostic_spans = [s for s in spans if s.span_kind == "diagnostic"]

# 1. Compute inclusive on ALL spans (cheap; same formula).
for span in spans:
    span.inclusive_us = (span.end_ns - span.start_ns) / 1000

# 2. Exclusive + structural invariants run on tree_spans ONLY.
build_tree(tree_spans)  # uses (span_id, parent_span_id), NOT span_name strings
for span in tree_spans (depth-first, children-first):
    span.exclusive_us = span.inclusive_us - sum(child.inclusive_us for child in span.children)
    assert span.exclusive_us >= -1.0, f"{span.span_name}: negative exclusive {span.exclusive_us}us — broken parent_span attribution"

# 3. Structural invariant (always true if instrumentation correct — sanity check, NOT coverage gate):
root = find_root_span(tree_spans)  # server_request_recv_to_first_content_sse_write
tree_exclusive_sum = sum(s.exclusive_us for s in tree_spans)
assert abs(tree_exclusive_sum - root.inclusive_us) < 1.0  # tree identity (Codex P1 #1: this alone is trivial)

# 4. T5 synthesizes residual output rows AFTER raw structural validation.
#    Raw server logs never contain `unattributed_*` spans, so Lane-B closed-set
#    validation applies to pre-synthesis emitted tree spans only.
synth_residual_rows = []
for span in tree_spans:
    if not span.children:
        continue
    residual_us = span.inclusive_us - sum(child.inclusive_us for child in span.children)
    if residual_us > 1.0:
        synth_residual_rows.append(SynthRow(
            span_name=f"unattributed_{span.span_name}",
            parent_span_id=span.span_id,
            inclusive_us=residual_us,
        ))

# 5. Real coverage gate per § 7.1: only NON-residual leaf time counts as "accountable":
unattributed_total = sum(s.inclusive_us for s in synth_residual_rows)
accountable_total = root.inclusive_us - unattributed_total
coverage_pct = accountable_total / root.inclusive_us
assert coverage_pct >= 0.95, f"coverage {coverage_pct:.1%} < 95% — instrumentation gaps in {[s.span_name for s in synth_residual_rows if s.inclusive_us / root.inclusive_us > 0.01]}"

# 6. Diagnostic spans validated separately — same request_id/routing_path checks
#    as tree_spans plus a route-specific closed span_name set (Lane A currently
#    allows sse_write_role_chunk_diagnostic; Lane B currently allows none),
#    optional root-interval-containment for diagnostics that should happen within
#    the root window, but NO containment check vs tree siblings (they're
#    explicitly allowed to overlap). Reported as separate T5 columns (e.g.,
#    role_chunk_diagnostic_us). Do NOT add their inclusive_us to coverage_pct or
#    unattributed_total.
diagnostic_allowed_by_routing = {
    "scheduler": {"sse_write_role_chunk_diagnostic"},
    "gs_chunked": set(),
}
for d in diagnostic_spans:
    assert d.request_id != "" and d.routing_path in {"scheduler", "gs_chunked"}
    assert d.span_name in diagnostic_allowed_by_routing[d.routing_path]
    # Optional containment per diagnostic semantics; T5 emits as report column.
```

**Hard invariants** (per Codex review v19 P1 — explicitly tree-only):
- `Σ tree_spans' exclusive_us ≡ root.inclusive_us` (tree identity — sanity check, alone insufficient per Codex P1 #1). Diagnostic spans NOT in the sum.
- `span.exclusive_us ≥ -1µs` for every **tree** span (negative = broken parent_span_id attribution). Diagnostic spans have no exclusive_us field.
- `coverage_pct = 1 - Σ synthesized_unattributed_*.inclusive_us / root.inclusive_us ≥ 95%` is the real gate (residual `unattributed_*` rows are computed from raw tree spans only; diagnostic durations do NOT contribute to or shrink unattributed).
- T5 aggregator MUST synthesize an explicit `unattributed_<span_name>` output row for every non-leaf raw tree span whose residual > 1µs. Server emitters MUST NOT emit raw `unattributed_*` spans.

**Out of scope (P5h+1 if needed)**:
- Per-MLX-kernel internal timing (e.g. softmax inside SDPA). Production-path can't expose this without changing production code; document as P5h+1 MLX-kernel investigation if T5 attribution shows attention `fused_sdpa` as unattributable hotspot.

### 2.5 Phase D root cause investigation (T0b of P5h)

P5g flagged 4 个 hypothesis:

| # | Hypothesis | P5h investigation method |
|---|---|---|
| H1 | GPU thermal drift across 24 spawns | Phase order rerun with identical cooldown policy: compare Phase D cells in A→B→C→D vs D→C→B→A. If H1 verifies, confirm the proposed randomized-order + cool-gate mitigation separately. |
| H2 | Substitute 自身有成本 | Use `p5g-profile` in-place timing with matching real/substitute windows. Do not use a pure Phase-A-with-ablate-X pp_tps comparison as proof. Step 7c `t_arr` requires its own real timer or is excluded from the H2 ratio gate. |
| H3 | Cache state divergence (AblateConv 不更新 conv_state) | Add diagnostic-only `ablate-conv-with-manual-cache-update`: keep qkv passthrough for Step 2b, but still build real Step 2a `conv_input` and run the real Step 2c conv_state update from the `conv_input` tail. |
| H4 | Kernel materialization / dispatch-path variance | Compare Step 7d forced-eval output materialization under AblateComputeG (g=zeros) vs Phase A (g=normal). Timer includes dispatch + taking both outputs + `eval(&[y, new_state])`, and excludes pre/post processing/cache mutation. |

**Decision tree**:
- H1 verified → P5h all phases adopt randomized order + cool gates between phases.
- H2 verified → discard ablation upper-bound concept; use Phase B/C ranking only for candidate priority.
- H3 verified → cache state must be carefully preserved across ablation; substitute design 需新 guard pattern。
- H4 verified → ablation invalid for kernel-dispatch-time hotspots; must use real candidate impl benchmark instead。
- If 2+ hypotheses verify, apply every verified mitigation that affects T2/T3 validity; ranking is for explanation only. If a hypothesis remains inconclusive after one same-protocol rerun, mark it unresolved and escalate to Boss rather than silently treating it as rejected.

**Out-of-scope**: P5h T0b 只 identify root cause + propose mitigation. Actual substitute redesign 在 P5h T1+ 各 layer profile 应用。

## § 3 Tasks decomposition (7 tasks, T0 split into T0a+T0b per Codex review v2 residual risk)

### T0a — Foundation: exclusive span schema + UMA hardening + GDN rerun (HARD GATE before T0b/T2/T3)

Per Codex review v2 residual: T0 was sprawling 4 distinct work-streams (schema infra + UMA hardening + 4 Phase D investigations + GDN rerun). **Codex recommends active split into T0a + T0b execution checkpoint**, not passive "sprawl-only-then-split" note. T0a proves trace schema works on a known component (GDN) before any Phase D investigation. If T0a's exclusive coverage gate fails on the GDN rerun alone, the schema is broken and Phase D investigations would emit non-schema records — wasted work.

- Branch verify + Cargo feature `p5h-profile` add (alongside `p5g-profile`, both can be on simultaneously)
- **Harness server-launch contract** (per Codex review v5 P1 #1 + v6 P1 dual-lane revision + v6 P2 #4 env-var split): all P5h sweep server processes MUST launch with **production-default `--prefill-chunk-size`** (do NOT override to 0 — v6 P1 showed this would divorce P5h from the config production users actually run). Single-active-row precondition is enforced via `--b-max 1` (also production default per `serve.rs:38`). The `p5h-profile` feature is a **Cargo build flag**, not a `serve` runtime flag (per Codex v6 P2 #4 — `serve.rs` has no `--features` arg). Harness exports two distinct env vars:

  ```bash
  # Cargo build-time features (controls compilation, gates p5h-profile instrumentation)
  IRONMLX_P5H_CARGO_FEATURES="p5h-profile"

  # Server runtime flags (do NOT include --prefill-chunk-size override; default 2048 is the production lane boundary)
  IRONMLX_P5H_SERVER_FLAGS="--b-max 1"
  # Add other production-default flags here only if a sweep needs to vary them (do not change for baseline P5h)

  MLX_DIR=$HOME/.local/mlx cargo run --release \
      --features "$IRONMLX_P5H_CARGO_FEATURES" \
      -p ironmlx -- serve $IRONMLX_P5H_SERVER_FLAGS \
      --model "$IRONMLX_MOE_MODEL_DIR" \
      --port 8080
  ```

  T0a profile gate annotates every root span with `routing_path: "scheduler" | "gs_chunked"` so T5 can partition records into Lane A (PP ≤ chunk_size — full deep attribution) vs Lane B (PP > chunk_size — top-level only). Any routing mismatch (e.g., PP=2048 expected scheduler but observed gs_chunked) fails the per-PP gate.
- **Exclusive span schema infrastructure** per § 2.5a — Rust span tracker + log emission format follows the § 2.5a **server-emitted fields** table as the single source of truth (per Codex review v4 P3 + v7 P2: do NOT restate the field list here — v3/v4 restated, then drifted; v5 restated again and immediately drifted in v6 by missing `routing_path` which v7 added to the table). `pp` and `run_id` are NOT server-emitted — they are aggregator-injected from iron-bench CSV per the § 2.5a aggregator-injected fields table.
- Python aggregator computes `exclusive_us = inclusive_us - sum(children_us)`; assert sum-to-root invariant + per-span `exclusive_us ≥ -1µs`
- **Trace context propagation infrastructure** (per § 2.5a "Trace context propagation through call chain"): **Implement exactly the § 2.5a per-thread guard contract — do NOT summarize or re-paraphrase the design here** (per Codex review v8 P2 + v12 P2: every prior round that restated guard semantics drifted from the source within one review round, and v11/v12 specifically caught re-paraphrases that re-introduced unsound Lane-A forwarder guard / closure-wide Lane-B guard). § 2.5a is the single source of truth.

  T0a delivers these concrete artifacts (deliverables list, not semantics restatement):
  - `P5hTraceContext` struct in `core/generate.rs` (fields per § 2.5a)
  - `P5hTraceGuard` RAII type with panic-on-nest (per § 2.5a)
  - `SpanHandle { span_id, span_name, parent_span_id, start_ns }` value type
  - `RootSpanHandle { ctx, span }` value type with `close_at(end_ns)` method (per Codex review v14 P2 — earlier `{ ctx, start_ns }` shape was stale, never matched § 2.5a code block)
  - Explicit-context lifecycle API: `open_p5h_span(&ctx, parent, span_name) -> SpanHandle` + `open_p5h_span_at(&ctx, parent, span_name, start_ns) -> SpanHandle` (the `_at` variant is required for root + http_parse_render_tokenize per Codex v14 P1, since their true start_ns is captured before `ctx` is complete) + `close_p5h_span(&ctx, handle, end_ns, fields)`
  - Implicit-guard API (two variants per Codex plan review v12 P1 #1):
    - `with_p5h_span_from_current_trace(span_name, fields_fn, body) -> T` — strict; handles open + push `P5H_CURRENT_SPAN_STACK` + run body + pop + close. Panics if `P5H_CURRENT_TRACE` is None or stack inconsistent. Reserved for sites where ctx is provably populated.
    - `try_with_p5h_span_from_current_trace(span_name, fields_fn, body) -> T` — None-tolerant + route-aware (per Codex plan review v12 P1 #1 + v20 P1 #1 + v21 P2 #3). Four-case dispatch on `P5H_CURRENT_TRACE`: (a) None → run body, no emit; (b) `Some(routing_path = "scheduler")` → forward to strict + emit; (c) `Some(routing_path = "gs_chunked")` → emit ONLY if `span_name ∈ LANE_B_ALLOWED_TRY_SPAN_NAMES = {"gs_kv_cache_alloc", "gs_chunk_N", "gs_first_token_sample_dispatch"}`, else no-op (Lane-B is top-level-only in P5h per spec § 5; deep emission deferred to P5h+1); (d) `Some(unknown routing_path)` → panic. Used by ALL current deep callers (decoder_layer.rs / gated_delta_net.rs / gated_attention.rs / sparse_moe.rs / model.rs lm_head / mlx_eval_barrier / GS chunked-prefill substeps) because `model.batched_prefill[_vl](...)` flows through these sites from both Lane-A (must emit) and Lane-B / non-OpenAI (must no-op via case (c)/(a)).
  - Per-thread `P5H_CURRENT_SPAN_STACK: RefCell<Vec<SpanHandle>>` for `parent_span_id` propagation (per § 2.5a v13 P1 fix)
  - `GenerateRequest.p5h_trace: Option<P5hTraceContext>` + `GenerateRequest.p5h_root_span: Option<SpanHandle>` fields (gated on `p5h-profile`; per Codex v15 P1 — the root SpanHandle MUST flow to the actor so the guard's `base_parent` can be set, otherwise deep spans orphan as second roots)
  - `RequestState.p5h_trace` + `RequestState.p5h_root_span` fields + `Scheduler::admit` copies both from `GenerateRequest`
  - Helper `Scheduler::cloned_active_row_p5h_trace_and_root(&self) -> Result<Option<(P5hTraceContext, SpanHandle)>>` (per Codex review v17 P1 + v10 P1 #2 + v11 P1 #4 — returns OWNED clones, NOT references; reference-returning signature would borrow `self.slots` and collide with the subsequent `&mut self.cache` / `&mut self.slots` / `&mut self.prng_state` that `prefill_admitted_inner` needs for `batched_prefill_vl` + Stage A + `sample_batch`). Requires `SpanHandle: Clone` (`P5hTraceContext: Clone` already). Under `--b-max 1` reads from the lone active `RequestState`. **Hard-fails on the `p5h-profile` feature only if active row count != 1**, OR if the row's `p5h_trace`/`p5h_root_span` are in mixed state (exactly one Some). **Returns `Ok(None)` when both are `None`** — every non-`openai.rs` entry path (anthropic.rs / CLI / tests / scheduler_actor internals) keeps both fields `None`, and the SINK callsite in `prefill_admitted_inner` quietly no-ops on None so those paths keep working under `--features p5h-profile`. Returns `Ok(Some((ctx, root_span)))` when both populated (only the `openai.rs` handler populates either, and it always populates both).
  - Per Codex v16 P1 — `prefill_admitted_inner` code change (`scheduler.rs:794-1025` area): open `model_prefill_forward` + enter guard around the `model.batched_prefill[_vl](...)` call; close mpf; then open `first_token_sampling` + (no guard for vanilla case) around the logits reshape + Stage A + `sample_batch(...)` interval; close `first_token_sampling`. Stage C distribution stays untouched.
  - Guard `enter()` call sites — EXACTLY the authorized list in § 2.5a, nothing more
  - Server startup panic when `p5h-profile` is active AND `b_max > 1` (single-active-row invariant)
  - **Fixture validation** (T0a HARD GATE precondition, per Codex review v10 P2 — first-record-only fixture is insufficient because it would pass even when top-level async spans never emit at all):
    - **Per-record check** (every emitted `[p5h-profile]` record, not just the first): `request_id != ""` AND `prompt_tokens > 0` AND `routing_path ∈ {"scheduler", "gs_chunked"}`. Any field empty/invalid = a guard set/drop site or explicit-context emission site is missing or wrong.
    - **Per-request consistency check**: after finding the single root, treat the root's `request_id` and `routing_path` as the request source of truth. Every tree and diagnostic span in the request group MUST carry the same `request_id` and `routing_path`; mixed routing within one request is a hard-gate failure. Do not infer route from `tree_spans[0]`, because log order is not guaranteed to put the root first.
    - **Route-aware schema presence check** (per non-aborted sweep request; required sets are split by `span_kind`):
      - If `routing_path == "scheduler"` (Lane A): assert tree span_names ⊇ `{server_request_recv_to_first_content_sse_write, http_parse_render_tokenize, scheduler_admission, model_prefill_forward, first_token_sampling, detok_format_first_content_chunk}` AND diagnostic span_names ⊇ `{sse_write_role_chunk_diagnostic}`. Per Codex v18 P2 #3, Lane-A role chunk is diagnostic because it can overlap with `model_prefill_forward` across threads; it is emitted but NOT included in exclusive child sums.
      - If `routing_path == "gs_chunked"` (Lane B): assert tree span_names ⊇ `{server_request_recv_to_first_content_sse_write, http_parse_render_tokenize, gs_stream_init_and_chunk_loop, gs_kv_cache_alloc, gs_chunk_N, gs_first_token_sample_dispatch, sse_write_role_chunk, gs_first_token_materialize_and_predispatch, detok_format_first_content_chunk}` AND diagnostic span_names ⊇ `{}`. `gs_kv_cache_alloc`, repeated `gs_chunk_N`, and `gs_first_token_sample_dispatch` are required children of `gs_stream_init_and_chunk_loop`; missing any of them makes the Lane-B top-level coverage gate meaningless.
      - `pre_content_decode_steps` is NOT an optional happy-path bucket for measured T0a/T5 requests. For non-aborted requests, any emitted `pre_content_decode_steps` record is a hard-gate failure per the § 2.5a T0a/T5 gate above.
      - If `routing_path == "gs_chunked"` (Lane B), reject any raw emitted tree span_name outside the Lane-B required tree set above. P5h Lane B is top-level-only; decoder/GDN/GatedAttention/MoE/lm_head deep substep attribution is deferred to P5h+1. This closed-set check runs before T5 residual synthesis, so synthesized `unattributed_*` output rows are not validator input.
    - **Id-based tree structural checks** (per Codex review v12 P1 + v13 P2 + v19 P1 — **all checks below operate on `tree_spans = [s for s in spans if s.span_kind == "tree"]` ONLY**; diagnostic spans validated separately, see "Diagnostic span checks" below):
      - **Id uniqueness**: `(request_id, span_id)` is unique across all records (no duplicate span_id within a request — atomic-counter emitter should never collide, but assert defensively)
      - **Exactly one root per request**: exactly one span with `parent_span_id is None` per `request_id`; its `span_name` MUST equal `"server_request_recv_to_first_content_sse_write"`. More than one root = forwarder/blocking task each opened its own root (forgot to clone `RootSpanHandle`); zero root = root open site missing.
      - **No orphan top-level**: no span other than root has `parent_span_id is None`. A non-root span with null parent = explicit-context emitter forgot to pass parent handle.
      - **Closure**: every non-null `parent_span_id` resolves to an emitted span's `span_id` within the same request (orphan parent_span_id = missing parent emission site).
      - **Label self-consistency**: `(parent_span_id is None) == (parent_span string is null)` AND `parent_span_id resolves to a span whose span_name equals the parent_span label` (mismatch = emitter wrote inconsistent id vs label).
      - **Interval containment**: for every non-root span, `parent.start_ns ≤ child.start_ns ≤ child.end_ns ≤ parent.end_ns`. Violation = child opened before parent or closed after parent (parent_span_id pointing at the wrong ancestor, or guard/handle leaked across requests).
      - **Reachability + no cycle**: starting from root, DFS over `parent_span_id` → child edges must reach every span in `tree_spans` exactly once (no cycles, no disconnected subtrees). Diagnostic spans NOT required to be reachable from root.
    - **Diagnostic span checks** (per Codex review v19 P1 — operate on `diagnostic_spans = [s for s in spans if s.span_kind == "diagnostic"]`):
      - Per-record field validity: same as tree (request_id != "", prompt_tokens > 0, routing_path ∈ {"scheduler", "gs_chunked"}).
      - The `span_name` MUST be in the closed set enumerated in § 2.5a per-lane bucket lists: Lane A currently allows only `sse_write_role_chunk_diagnostic`; Lane B currently allows no diagnostic spans. Any other `span_name` with `span_kind="diagnostic"` is an emitter bug = fail.
      - `parent_span_id` for diagnostic spans MUST point at root's span_id (annotates the request scope) OR be null (loose diagnostic). Either is acceptable; do NOT include in the tree DFS.
      - NO interval containment check vs tree siblings (diagnostic spans explicitly ALLOWED to overlap tree spans across threads; that overlap is the entire reason `span_kind="diagnostic"` exists).
    - Any failure (tree or diagnostic) = T0a fails before close (do NOT proceed to T0b / T2 / T3 / T4 with a broken schema).
- **Request-correlation infrastructure** (per § 2.5a "Join key" — single committed path):
  - `openai.rs` (chat-completion response builder): emit `X-Ironmlx-Request-Id: <uuid>` header on STREAMING responses ONLY (`serve_via_scheduler_stream`, `serve_via_gs_stream`), gated on `p5h-profile` feature; same uuid used as `GenerateRequest.p5h_trace.request_id` and every `[p5h-profile]` log record's `request_id` field. Per Codex plan review v16 P1 #2 + v17 P2 #3: non-streaming `serve_via_*_unary` paths skip P5h entirely (no root span opened, no header emitted) — iron-bench `--capture-server-request-id` is only meaningful for streaming sweeps.
  - `iron-bench/src/client.rs`: capture `X-Ironmlx-Request-Id` from `resp.headers()` BEFORE entering `bytes_stream()`; add `request_id: Option<String>` to `RequestResult`. Capture path gated on new CLI flag.
  - `iron-bench/src/report.rs::render_csv`: CSV serializer signature changes to `render_csv(cells, capture_request_id: bool) -> String` and writes new `request_id` column **only when flag is on** (per § 2.5a P2 #3 fix + Codex plan review v18 P2 #5 + v19 P2 #3 — flag off keeps CSV schema byte-identical to current; column appended after existing `finish_reason`). `render_markdown` and `render_json` are unchanged in both flag states (CSV-only scope — aggregator consumes CSV).
  - New iron-bench CLI flag `--capture-server-request-id` gates BOTH capture path AND serializer schema (default off, on for P5h sweeps; off-state output is byte-identical to current iron-bench)
- **UMA hardening protocol** implementation: cold/warm pair measurement + variance check + automatic retry (per § 2.4)
- **GDN harness code extension** to emit `[p5h-profile]` log lines with new schema (per § 2.2 #4 + Codex review v2 P2 #1 + v12 P2 #6 + v14 P2 #5) — `[p5h-profile]` lines come from each of the 11 GDN substeps being wrapped in `try_with_p5h_span_from_current_trace(...)` (T0a.11 Step 2), which emits one record on span close. The existing `[p5g-profile]` `tracing::info!` line at `gated_delta_net.rs:1059-1077` stays untouched (back-compat for the P5g harness consumer); there is NO hand-written parallel `[p5h-profile]` formatter call alongside it. Both line shapes appear in a `--features p5h-profile` rerun because they originate at DIFFERENT call sites and use DIFFERENT formatters — NOT because anything was double-emitted at the existing barrier site.
- **GDN rerun** under P5h UMA protocol — same PP set as P5g T0, cold/warm pair per PP, exclusive span tree where GDN substeps' `parent_span_id = <enclosing attention_path span's span_id>` (label `parent_span = "attention_path"`), per § 2.5a id-based tree (v12 P1) + wrapper structure (v4 P1; NOT `decoder_layer_N`)
- **Schema validation on GDN rerun** (T0a's hard gate, must pass before T0b starts):
  - Sum-to-root identity holds within ±1µs
  - All `exclusive_us ≥ -1µs`
  - Lane-A GDN `attention_path` emit-limited coverage regression guard (per § 7.1 + T0a.14 Codex review): per-PP **median** `coverage_pct ≥ 50%` AND per-instance **min** `coverage_pct ≥ 35%`. T0a.14 empirical sweep showed per-substep `tracing::info!` dispatch overhead + return-path gaps cap raw substep coverage at 53-55% median (37-41% min) regardless of legitimate body wrap expansion (slicing attribution, Option E end_ns capture, etc.). The original ≥95% target is deferred to **[p5h+1_emit_cost_reduction]** (buffered/binary emit, or equivalent low-overhead collection path); T0a's gate is a regression guard, not exact wall-time completeness. Full `decoder_layer_N` coverage gate still applies only at T5 after T1-T4 all land.
  - Lane-B `gs_stream_init_and_chunk_loop` top-level coverage_pct ≥ 95% (per Codex plan review v21 P1 + v22 P2 — P5h Lane-B is top-level-only, but the dominant chunk-loop bucket still needs a residual floor; deep decoder/GDN/GatedAttention/MoE/lm_head span_names remain out of scope and must be suppressed/rejected on Lane-B. Lane-B's 95% gate holds because gs_top wraps only 3 direct children with no per-emit drift accumulation.)
  - UMA cold/warm variance per-PP threshold (per § 2.4 + T0a.14 thermal observation): default ±2% for PP ∈ {128, 512, 2048, 4096, 8192}; **±4% for PP=16384** because a 7-run iron-bench warm batch at PP=16384 runs ~70s of continuous GPU dispatch, accumulating heat past the 5min intra-PP cool gate's recovery capacity on M5 Max.
  - iron-bench↔server `request_id` join rate = 100% across all sweep cells
- Output: P5h schema infrastructure (incl. span lifecycle API per § 2.5a), request-correlation infra (server header + iron-bench capture + CSV column), GDN protocol-consistent data under id-based exclusive tree (GDN substeps' `parent_span_id` = enclosing `attention_path` span; label `parent_span = "attention_path"`)
- **GATE**: T0a must close before T0b dispatches. If schema gate fails, fix schema first; do NOT proceed to Phase D investigations until GDN data demonstrates schema works end-to-end.
- Commit: `feat(p5h-t0a): exclusive span schema + UMA hardening + GDN P5h-protocol rerun`

### T0b — Phase D root cause investigation (4 hypotheses, depends on T0a)

T0b only starts AFTER T0a closes (schema proven on GDN rerun). T0a remains the sequencing gate, but T0b Phase D root-cause measurement intentionally uses the lower-overhead existing P5g profiling path (`p5g-profile` ON, `p5h-profile` OFF) rather than emitting P5h schema records; otherwise per-span P5h logging overhead would contaminate substitute and kernel timing.

- Phase D root cause investigation (4 hypotheses per § 2.5 decision tree):
  - H1 (thermal drift): phase-order rerun with identical cooldown policy; compare Phase D values across A→B→C→D vs D→C→B→A orderings
  - H2 (substitute self-cost): in-place `p5g-profile` timers with matching real/substitute timing windows; Step 7c `t_arr` needs its own real timer or is excluded from H2 ratio gating
  - H3 (cache state divergence): add `ablate-conv-with-manual-cache-update` variant; isolate cache-divergence effect from substitute effect by preserving the real Step 2c `conv_input`-tail cache update
  - H4 (kernel materialization / dispatch-path variance): Step 7d forced-eval timing under AblateComputeG vs Phase A; include dispatch outputs' `eval`, exclude pre/post processing and cache mutation
- Decision-tree mapping (per § 2.5):
  - H1 verified → P5h all phases adopt randomized order + cool gates
  - H2 verified → discard ablation upper-bound; use Layer 2 ranking only for candidate priority
  - H3 verified → ablation requires cache-state-preserving substitute design (new guard pattern)
  - H4 verified → ablation invalid for kernel-dispatch-time hotspots; use real candidate impl benchmark instead
  - Multi-primary → apply every verified mitigation that affects T2/T3 validity; rank causes only for reporting
  - Inconclusive after one same-protocol rerun → mark unresolved and escalate to Boss with numeric data
- Output: `reports/p5h-phase-d-root-cause.md` documenting verified root-cause hypotheses, ranked primary cause, applied mitigations, unresolved hypotheses if any, and decision-tree binding for T2/T3 conditional gates
- Commit: `feat(p5h-t0b): Phase D root cause investigation + decision-tree resolution`

### T1 — HTTP path + scheduler/admission profile

- Instrument: HTTP request entry / response exit, Scheduler admit queue dequeue, batch construction, slot allocation
- Layer 1: per-request boundary timing
- Layer 2: per-step breakdown (request parse / chat-template / admission wait / batch construct / forward dispatch / response serialize)
- Run sweep PP=128-16384, 5 runs warm + cold pair per UMA hardening
- Output: HTTP+scheduler attribution per PP, short-PP fixed-overhead identification
- Commit: `test(p5h-t1): HTTP + scheduler admission profile`

### T2 — GatedAttention layer instrumentation (new harness)

- Read `ironmlx/src/nn/gated_attention.rs` (10/40 full-attn layers in MoE-A3B; verified via config.json `layer_types` pattern "linear,linear,linear,full" × 10)
- Add 3-edit instrumentation pattern (mirror P5g GDN):
  - Edit 1: entry barrier (input + cache materialize) gated on Layer1|Layer2
  - Edit 2: cache update sites use `as_deref_mut()` (preserve borrow)
  - Edit 3: tail refactor + exit barrier + emission. Decoder layer (`decoder_layer.rs::DecoderLayerMoe::forward_on`) opens the wrapper `attention_path` span via `try_with_p5h_span_from_current_trace("attention_path", ..)` (T0a.11 Step 1); each GatedAttention substep inside `gated_attention.rs::forward_on` opens its own span via the same `try_` API (T2.2 Step 2) so substep `parent_span_id` = wrapper span's `span_id` automatically (stack top), label `parent_span = "attention_path"`. Per § 2.5a + Codex v4 P1 + v12 P1 + v13 P2 #4. Substep `SpanFields` carry the real decoder `layer_idx` plumbed via the new `layer_idx: i32` parameter on `GatedAttention::forward_on` (per Codex v13 P1 #2).
- **Layer 2 step breakdown — code-backed taxonomy** (7 sub-steps matching actual `gated_attention.rs:120-276` production path with `attn_output_gate=true` per config; **not** decomposing the fused SDPA internals which are inside `mlx::fast::scaled_dot_product_attention_on`):
  1. `q_gate_k_v_proj` — three separate Linear projections (NOT fused QKV): `q_proj` outputs `Hq × D × 2` (queries concatenated with gate, since `attn_output_gate=true`); `k_proj` outputs `Hkv × D` (KV-head dim, GQA); `v_proj` outputs `Hkv × D` (KV-head dim). Single span covers all three.
  2. `q_split_norm_reshape` — split q output back into queries + gate halves + `q_norm`/`k_norm` (per-head RmsNorm) + reshapes/transposes to SDPA layout
  3. `mrope_apply` — `mrope.apply(&queries, &k, cos, sin)` rotary
  4. `kv_mask_update` — KV validity mask construction + `KVCache::update_and_fetch_on(k, v, lens, target)`
  5. `fused_sdpa` — `mlx::fast::scaled_dot_product_attention_on(...)` (fused MLX op; **softmax/value matmul internals inside, not separately measurable on production path** — if T5 shows this as unattributable hotspot, P5h+1 may investigate MLX-kernel-level)
  6. `gate_sigmoid_mul` — gate sigmoid + elementwise multiply on SDPA output (gate tensor came from `q_proj` second half)
  7. `o_proj` — `Linear::forward_on(&gated, target)` output projection
- **Layer 3 ablations — conditional on T0b Phase D outcome** (per § 2.5 decision tree):
  - **If T0b verifies H1** (thermal drift): ablations OK, T2 runs Layer 3 with randomized phase order + cool gates。
  - **If T0b verifies H2** (substitute self-cost): Layer 3 **skipped** for T2; replace with real-path microbenchmarks (e.g., swap `o_proj` with smaller dim variant compiled separately, measure end-to-end delta against baseline). Layer 1/2 still emitted.
  - **If T0b verifies H3** (cache state divergence): Layer 3 requires cache-state-preserving substitute design — for GatedAttention, KV cache must remain valid across ablation (e.g., `ablate_fused_sdpa` substitute returns shape-preserving zeros but still calls `KVCache::update_and_fetch_on` to keep cache consistent for subsequent forwards).
  - **If T0b verifies H4** (kernel materialization / dispatch-path variance): Layer 3 invalid for any step that touches Metal kernels (especially `fused_sdpa`); skip Layer 3 for steps 4-5 (kv_mask_update + fused_sdpa); Layer 3 OK for pure op-level steps (q_gate_k_v_proj, q_split_norm_reshape, mrope_apply, gate_sigmoid_mul, o_proj).
  - **If multiple hypotheses verify**: apply every verified mitigation that affects validity; ranking is for reporting only.
- Run sweep + aggregate under id-based exclusive span schema (per § 2.5a): `decoder_layer.rs::DecoderLayerMoe::forward_on` opens the wrapper `attention_path` span via `try_with_p5h_span_from_current_trace` (None-tolerant per Codex v12 P1 #1); each of the 7 GatedAttention substeps in `gated_attention.rs::forward_on` opens its own span inside that wrapper using the same `try_` API, so each substep's `parent_span_id` = wrapper span's `span_id` (label `parent_span = "attention_path"`). Substep `SpanFields.layer_idx` is set from the new `layer_idx: i32` parameter (per Codex v13 P1 #2). Wrapper site is `decoder_layer.rs`, NOT `text_model.rs` (per Codex v12 P2 #4).
- Output: GatedAttention per-PP occupancy table (7-step breakdown), top-3 step ranking, long-PP O(S²) growth verification (PP=128 to PP=16384 step ratios)
- Commit: `test(p5h-t2): GatedAttention 3-layer profile (code-backed taxonomy + conditional ablation)`

### T3 — MoE expert dispatch + LinearMLP profile

- Read `ironmlx/src/nn/sparse_moe.rs` (or equivalent SparseMoeBlock module) — 40 decoder layers each contain MoE (`num_experts=256`, `num_experts_per_tok=8`, `moe_intermediate=512`, `shared_expert_intermediate=512` per config.json verified)
- Add 3-edit instrumentation pattern (mirror P5g GDN + T2 GatedAttention)
- **Layer 2 step breakdown — code-backed taxonomy** (8 sub-steps reflecting sorted-routing path shipped in P5e; verify against current `sparse_moe.rs` at instrumentation time, exclusive parent = `mlp_path`):
  1. `router_logits_softmax_topk` — gating linear (`hidden → num_experts`) + softmax + top-`num_experts_per_tok` index/weight selection
  2. `routing_sort_pack` — sorted-routing pack (P5e shipped): sort tokens by expert id, gather token-features into per-expert contiguous slabs. Step exists only on the sorted path; if T0a observes `sparse_moe.rs` falls into a non-sorted branch for some sequence length, emit `routing_sort_pack` with `inclusive_us = 0` and document the branch in T3 close-out.
  3. `gather_qmm_gate_up` — quantized matmul gate + up projections for packed token slabs (`packed_tokens × moe_intermediate × 2`)
  4. `swiglu_activation` — SwiGLU elementwise (gate · silu(up) or equivalent fused op) on the gate/up output
  5. `gather_qmm_down` — quantized matmul down projection (`packed_tokens × hidden`)
  6. `routing_unsort_weighted_reduce` — unpack from per-expert slabs back to original token order + weight by routing probability + scatter-reduce contributions across the `num_experts_per_tok` experts of each token
  7. `shared_expert` — separate LinearMLP for the shared expert (`BS × shared_expert_intermediate × 2 + BS × hidden`)
  8. `moe_output_sum` — final residual combining routed + shared outputs into the layer's MLP output tensor
- **Layer 3 ablations — conditional on T0b Phase D outcome** (same gating logic as T2):
  - **H1 verified**: ablations OK with randomized order + cool gates
  - **H2 verified**: Layer 3 skipped, replace with real-path microbenchmarks (e.g., reduce experts_per_tok from 8 → 4 in a controlled fork, measure delta)
  - **H3 verified**: ablation must preserve routing index validity (don't break downstream attention KV slot allocation); pack/unpack steps must remain consistent (substitute can no-op compute but must still produce shape-correct outputs)
  - **H4 verified**: Layer 3 invalid for `gather_qmm_*` steps + `routing_sort_pack`/`routing_unsort_weighted_reduce` (all kernel-dispatch dependent); skip Layer 3 for steps 2-3 + 5-6; OK for steps 1, 4, 7-8
  - **Multiple verified hypotheses**: apply every verified mitigation that affects validity; ranking is for reporting only.
- Run sweep + aggregate under id-based exclusive span schema (per § 2.5a): `decoder_layer.rs::DecoderLayerMoe::forward_on` opens the wrapper `mlp_path` span via `try_with_p5h_span_from_current_trace` (None-tolerant per Codex v12 P1 #1); each of the 8 MoE substeps in `sparse_moe.rs::SparseMoeBlock::forward_on` opens its own span inside that wrapper using the same `try_` API, so each substep's `parent_span_id` = wrapper span's `span_id` (label `parent_span = "mlp_path"`). NOT `decoder_layer_N` per Codex v4 P1. Substep `SpanFields.layer_idx` is set from the new `layer_idx: i32` parameter (per Codex v13 P1 #2). Wrapper site is `decoder_layer.rs`, NOT `text_model.rs` (per Codex v12 P2 #4).
- **ROI math source**: derive `num_experts_per_tok = 8`, `moe_intermediate = 512`, `num_experts = 256` from `Qwen35MoeConfig` runtime values, NOT hardcoded constants in spec/report (which could drift if model config changes)
- Output: MoE per-PP attribution, router top-8 cost ratio, gather_qmm dominance check, shared expert vs routed cost split
- Commit: `test(p5h-t3): MoE expert + LinearMLP profile (code-backed taxonomy + conditional ablation)`

### T4 — lm_head + MLX eval/cache state + tokenization/first-eval profile

- `slice_last_and_project_lm_head` Linear quantized matmul timing (this is the `slice_last_and_project` child of `model_prefill_forward` per § 2.5a, NOT a sibling — model_prefill_forward boundary fix per Codex v4 P2 #4; single Linear not split)
- **Lane A** `first_token_sampling` per-row sampler invocation timing (sibling of `model_prefill_forward` under root, per § 2.5a Lane-A top-level buckets); **Lane B** does NOT emit this bucket — sample dispatch lives inside `gs_stream_init_and_chunk_loop` as `gs_first_token_sample_dispatch`, and the post-`new()` work is `gs_first_token_materialize_and_predispatch` (per § 2.5a Lane-B bucket list)
- MLX `eval()` barrier latency at major sync points
- KVCache + GatedDeltaCache state-update cost (per-forward)
- Tokenization: tokenizer Encode time per prompt length (subspan of `http_parse_render_tokenize`)
- First-eval (JIT compile + kernel warmup) one-shot cost per (model, prompt_shape) pair
- Run sweep, aggregate
- Output: `slice_last_and_project_lm_head` occupancy (Lane A) / `gs_first_token_sample_dispatch` cost (Lane B), `first_token_sampling` (Lane A) / `gs_first_token_materialize_and_predispatch` (Lane B) cost, first-eval amortization (短 PP suspect), tokenization fixed cost
- Commit: `test(p5h-t4): lm_head + tokenization + MLX state profile`

### T5 — Cross-layer attribution synthesis + P5i/P5j candidate ranking + close-out report

- Aggregate T0a (GDN rerun) + T0b (Phase D resolution) + T1-T4 measurements into per-PP exclusive attribution table per § 2.5a span schema
- Implement § 2.5a's tree/diagnostic split + exclusive computation per § 2.5a pseudocode (`tree_spans = [s for s in spans if s.span_kind == "tree"]`; all tree-property computation operates on tree_spans only; diagnostic_spans validated separately, reported as columns). DO NOT re-derive the formulas here per Codex review v20 P1 — restating triggered drift in v19 (and previously v3/v4/v6/v7).
- **Exclusive coverage gate** per § 7.1 (residual-based, single source of truth — DO NOT redefine here). If a future revision changes the gate, change § 7.1 first and reference it from here.
- Identify per-PP top-3 bottleneck across all measured spans
- Rank P5i candidates (短 PP focus, +24-74% target) by ROI estimate
- Rank P5j candidates (长 PP focus, +110-128% target) by ROI estimate + Scope gate trigger (kernel rewrite = trigger Boss approval)
- **Target feasibility assessment**: honest evaluate "全 PP omlx+10% 在 P5i+P5j 内是否可达" given measured + estimated upper-bound caps
- Write `reports/p5h-attribution.md` (self-contained)
- Lock P5h spec § 7.2 final state (P5g `spec § 7.2` pattern — TBD → locked) + commit
- Commit: `chore(p5h-t5): close-out — all-PP exclusive attribution + P5i/P5j candidate ranking`

## § 4 Validation gates per task (Tn 共用)

**任何 task 触碰 Rust 代码** (T0a 必触碰: schema infra + GDN harness extension; T0b 可能触碰: Phase D substitute mode adds; T1-T4 必触碰: 各 layer instrumentation; T5 一般只触碰 Python aggregator + Markdown report, 但若 T5 修 Rust 也适用),都必须严格按 CLAUDE.md "Rust 代码检测"规定执行:

```bash
cargo fmt                                                                          # Rust 自动格式化 (CLAUDE.md mandate)
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check                       # format check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings  # 0 Rust warnings (mlx-sys C++ warnings ok)
MLX_DIR=$HOME/.local/mlx cargo build --release                                     # release build PASS
```

**Python-only tasks** (T5 aggregator + report build, if no Rust touched): run `ruff check` / `ruff format --check` if a Python lint config exists in repo; markdown link/heading validity verified via repo's standard pre-commit (if configured). No cargo gates required for pure Python/Markdown changes.

**Schema-touching tasks** (T0a, T5 aggregator): T5 aggregator MUST emit a schema-conformance report — every emitted `[p5h-profile]` record validated against § 2.5a server-emitted fields. **iron-bench↔server `request_id` join rate must = 100% per PP (orphan rate = 0%)** — per Codex review v5 P2 #4, this matches T0a hard gate + § 2.5a deterministic header join (wallclock fallback was dropped in v5, so any orphan == broken header propagation == fix-before-close-out). The earlier "orphan rate > 1% fails" threshold (v4) is superseded.

Sentinel suite (MoE-A3B-4bit; per § 5):

```bash
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored --test-threads=1         # argmax=11
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --test-threads=1      # B=2 row-eq
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --test-threads=1   # SSE PASS
```

Profile-gate invariant (must verify per task):

- Default build (no `--features p5h-profile`): 0 `[p5h-profile]` log lines emitted by `ironmlx serve` under any sweep (byte-for-byte identity with P5f baseline)
- Feature build (`--features p5h-profile`): instrumentation active, exclusive span schema records emitted per § 2.5a, UMA cold/warm pair variance ≤ ±2% on the layer's sweep

T0a + T0b + T5 额外:

- T0a exclusive span schema validator (tree spans only per Codex v20 P1; diagnostic spans validated separately per § 2.5a): assert `sum(child.inclusive) ≤ parent.inclusive` for all parent tree spans in test fixture; assert sum-to-root identity within ±1µs over `tree_spans`; assert per-tree-span `exclusive_us ≥ -1µs`
- T0a GDN rerun: P5h-protocol GDN data emitted under id-based exclusive tree with GDN substeps' `parent_span_id = <enclosing attention_path span id>` (label `parent_span = "attention_path"`, per § 2.5a + Codex v4 P1 + v12 P1); Lane-A GDN `attention_path` emit-limited coverage regression guard (per-PP median ≥ 50% AND per-instance min ≥ 35%, per T0a.14 Codex review — ≥95% target deferred to **[p5h+1_emit_cost_reduction]**) and Lane-B `gs_stream_init_and_chunk_loop` top-level coverage_pct ≥ 95% under § 7.1 residual-based gates (full `decoder_layer_N` coverage gate applies only at T5; Lane-B deep substep attribution is suppressed/rejected and deferred to P5h+1); UMA cold/warm variance ≤ ±2% per PP (PP=16384 widens to ±4% per § 2.4 + T0a.14 thermal observation); iron-bench↔server `request_id` join rate = 100%
- **T0a HARD GATE**: T0a's coverage + schema invariants must pass before T0b dispatches (per § 3 T0a). If schema gate fails on GDN rerun, fix schema before any Phase D investigation work.
- T0b Phase D root cause: 4 hypotheses (H1-H4) resolved per § 2.5 decision tree, OR explicit unresolved-list documented in `reports/p5h-phase-d-root-cause.md`; T2/T3 conditional ablation gates bound per T0b outcome
- T5 attribution: per § 7.1 residual-based exclusive coverage gate (`coverage_pct = 1 - Σ synthesized_unattributed_*.inclusive_us / root.inclusive_us ≥ 95%` per PP, root = `server_request_recv_to_first_content_sse_write`; residual rows synthesized after raw structural validation); P5i/P5j candidate ranking emitted with ROI estimate ranges + Scope gate trigger status; `client_transport_residual_us` reported as separate diagnostic column

## § 5 Numerical Safety

复用 P5g sentinel suite (P5e/P5f shared):
- `p5_qwen35_moe_smoke::p5b_first_token_argmax_regression_sentinel`: argmax = 11
- `p5_qwen35_moe_batched::p5c_batched_prefill_b2_equals_b1_per_row`: B=2 vs B=1 per-row identical (LOGITS_TOL=1.0; if any P5h instrumentation drifts argmax → STOP per § 4.3)
- `p5_qwen35_moe_http_smoke::p5c_http_smoke_chat_completion_non_stream`: PASS

P5h 不引入新 numerical correctness 风险 (measure-only, no algorithmic change). Ablation substitutes mode-gated, default build identity preserved.

## § 6 Out of Scope / Non-Goals (P5h)

- Any optimization implementation (kernel rewrite / tile tuning / memory layout / fusion). Optimizations 在 P5i + P5j + 可能 P5k 实施。
- GDN Step 7 `gated_delta_step` Metal kernel rewrite (P5g § 4.1 Scope gate trigger, P5j candidate driver)
- Step 8 out_proj kernel optimization (same family, P5j companion)
- Multi-request batching default change (`--b-max 1` 保留 P5f shipped config)
- **Chunked GS path (`serve_via_gs_stream`) DEEP substep attribution** (per Codex review v6 P1 — v7 keeps production-default `--prefill-chunk-size`, so PP > 2048 routes through GS chunked path as Lane B; Lane B measures top-level only). Per-substep GDN/GatedAttention/MoE breakdown under chunked path requires a `chunk_idx` schema extension (each substep span emits N times per request) plus chunked-tree extension to § 2.5a. Deferred to P5h+1 if Lane B top-level results show a long-PP P5j candidate needs per-substep evidence.
- **Multi-row in-flight attribution** (per § 2.5a "Single-active-row hard gate"): the `P5H_CURRENT_TRACE` RAII guard design assumes exactly one active row when any sync instrumentation region runs. Multi-row attribution would need per-row mlx::Stream / per-row span context; out of P5h scope.
- prefill_chunk_size sweep (becomes P5j candidate if T2 GatedAttention long-PP O(S²) supports it)
- omlx PagedCache style port (out per memory `[feedback_design_philosophy]`)
- mlx::compile wrap (still blocked by 4 safe-wrapper API gaps from P5e T2)
- GatedAttention algorithmic redesign (P5h measures it; redesign 在 P5j 如果 measured 必要)

## § 7 Success Criteria

### 7.1 Exclusive attribution coverage gate (P1-fix from Codex review v1 + v2)

T5 must produce a per-PP exclusive attribution table built **only** from same-protocol P5h measurements (P5g existing data excluded; see § 2.2 #4 + T0a GDN rerun). Coverage is **residual-based** (Codex review v2 P1 #1 — the naive `Σ exclusive ≡ root.inclusive` formulation is a tree identity, trivially true and useless as a gate). Coverage computed as:

```
tree_spans = [s for s in spans if s.span_kind == "tree"]   # per Codex v19 P1 + v20 P1
root_wall_us = root_span(tree_spans).inclusive_us           # root.span_name == "server_request_recv_to_first_content_sse_write"
synth_residual_rows = synthesize_unattributed_rows(tree_spans)  # per § 2.5a Residual leaves
unattributed_total_us = Σ s.inclusive_us  for s in synth_residual_rows
                        where s.span_name.startswith("unattributed_")
accountable_us = root_wall_us - unattributed_total_us
coverage_pct = accountable_us / root_wall_us
            = 1 - (unattributed_total_us / root_wall_us)
```

The root is **server-side only** (`server_request_recv_to_first_content_sse_write`, per § 2.5a Codex v2 P1 #2). Client/transport latency is reported as a separate `client_transport_residual_us = iron_bench_ttft_us - root_wall_us` diagnostic column, NOT included in the coverage gate. Diagnostic spans (`span_kind == "diagnostic"`, e.g., Lane-A `sse_write_role_chunk_diagnostic`) are reported as separate diagnostic columns and NEVER enter the coverage_pct numerator or denominator (per Codex v19 P1 + v20 P1).

**Hard invariants** (per § 2.5a):
- `coverage_pct ≥ 95%` per PP (else identify which `unattributed_<span>` dominates → add instrumentation for that span's children → re-run before close-out)
- `span.exclusive_us ≥ -1µs` for every emitted **tree** span (per Codex v19 P1 — diagnostic spans have no exclusive_us; negative beyond noise = broken parent_span_id attribution, MUST fix)
- `Σ tree_spans' exclusive_us ≡ root_wall_us` within ±1µs (tree identity sanity check; tree_spans only per Codex v19 P1 + v20 P1; alone INSUFFICIENT as a coverage gate per Codex review v2 P1 #1)
- No bucket can be counted under two different parents (mutually exclusive tree)
- T5 aggregator MUST synthesize an explicit `unattributed_<span_name>` output row for every non-leaf raw tree span whose residual > 1µs; server emitters MUST NOT emit raw `unattributed_*` spans (per § 2.5a "Residual leaves")

This **replaces** prior naive "sum medians ≥ 95%" gate which double-counted nested spans (Codex review v1 P1 #1) and the equally-naive "`Σ exclusive_us / root.inclusive_us`" formulation (Codex review v2 P1 #1, which is a tree identity).

### 7.2 P5h ship gate (T5 close-out gate)

1. **Exclusive attribution coverage** per § 7.1: `coverage_pct = 1 - Σ synthesized_unattributed_*.inclusive_us / root.inclusive_us ≥ 95%` per PP (residual-based; root = `server_request_recv_to_first_content_sse_write`; raw `tree_spans` only per Codex v19 P1; residual rows synthesized by T5 after structural validation); `exclusive_us ≥ -1µs` for every emitted tree span; diagnostic spans validated separately and reported as columns (e.g., `role_chunk_diagnostic_us`); `client_transport_residual_us` reported separately (not part of gate)
2. **Protocol-consistent data — dual-lane explicit** (per Codex review v7 P2 + § 2.5a "Routing precondition"):
   - **Lane A** (PP ∈ {128, 512, 2048}, scheduler path): full deep substep attribution — HTTP/scheduler/admission (T1) + GDN (T0a rerun; substeps' `parent_span_id` = enclosing `attention_path` span) + GatedAttention (T2, 7 substeps under `attention_path` wrapper) + MoE (T3, 8 substeps under `mlp_path` wrapper) + lm_head/MLX state (T4) — all measured under same UMA hardening + id-based exclusive span schema with trace context correlation; § 7.1 residual coverage ≥ 95% per PP. Phase D resolution (T0b) is a separate `p5g-profile` root-cause decision input that binds Layer-3 ablation validity; it is not included as a raw P5h coverage-tree data source.
   - **Lane B** (PP ∈ {4096, 8192, 16384}, chunked GS path): top-level only per the § 2.5a Lane-B bucket list (single source of truth — DO NOT re-enumerate here per Codex v9 P2 #1; v8 re-enumeration carried over the stale `first_token_sampling` bucket that was removed in v9). Lane-B buckets include `gs_first_token_sample_dispatch` nested inside `gs_stream_init_and_chunk_loop` and a top-level `gs_first_token_materialize_and_predispatch` sibling (NOT a `first_token_sampling` sibling). Deep GDN/GatedAttention/MoE/lm_head substep attribution under chunked path is **explicitly out of scope** (deferred to P5h+1 per § 6); Lane B coverage gate measured only against top-level buckets, still ≥ 95%.
   - P5g existing data remains as prior reference only, excluded from both lanes' coverage gates.
   - P5j long-PP candidate ranking from Lane B carries explicit "bounded by Lane-B granularity" caveat; any P5j candidate requiring per-substep evidence at long PP must defer to P5h+1 chunked deep-attribution before P5j dispatch.
3. **UMA hardening verified**: cross-repeat (cold/warm pair) measurement variance per-PP threshold (default ±2%, **PP=16384 ±4%** per § 2.4 + T0a.14 thermal observation: 7-run warm batch at PP=16384 ~70s of continuous GPU dispatch outpaces 5min intra-PP cool gate's recovery on M5 Max)
4. **Phase D root cause** (T0b output): H1-H4 verdicts documented, verified hypotheses' mitigations applied, primary cause ranked for explanation, OR explicit unresolved hypothesis list with proposed next investigation path (per § 2.5 decision tree); T2/T3 Layer 3 conditional ablation gates bound per T0b outcome
5. **P5i + P5j candidate ranking**: each candidate has expected ROI range (number-anchored), Scope gate trigger status, 实施优先级
6. **Target feasibility assessment**: honest verdict on "全 PP omlx+10% achievable in P5i+P5j" — if not, partial-target proposal for Boss decision
7. **Reusable infra delivered**: exclusive span schema infrastructure (per § 2.5a) + UMA hardening protocol (per § 2.4) + GatedAttention 3-layer profile harness (per § 2.2 #5 code-backed taxonomy with `attn_output_gate=true`) + MoE 8-step profile harness (per § 2.2 #6 sorted-routing path + `Qwen35MoeConfig` runtime values) — all usable in P5i+P5j+P5h+1
8. **Validation gates pass per task** (T0a/T0b/T1-T5 each independently green; per § 4)
9. **T0a HARD GATE passed** before T0b/T2/T3 dispatched: schema sum-to-root invariant (tree_spans only per Codex v20 P1) + per-tree-span exclusive_us ≥ -1µs + Lane-A GDN `attention_path` emit-limited coverage regression guard (per-PP median ≥ 50% AND per-instance min ≥ 35%, per T0a.14 Codex review; the ≥95% wall-time-completeness target is deferred to **[p5h+1_emit_cost_reduction]** — buffered/binary emit or equivalent low-overhead collection path) + Lane-B `gs_stream_init_and_chunk_loop` top-level coverage ≥ 95% with Lane-B deep span names suppressed/rejected + diagnostic spans validated separately per § 2.5a + UMA cold/warm variance per-PP threshold (default ±2%, PP=16384 ±4% per § 2.4 + T0a.14 thermal observation) + iron-bench↔server `request_id` join rate = 100% all verified on GDN rerun data (per § 3 T0a + § 4)

P5h 整体 success = all 9 gates PASS, output (attribution report + P5i/P5j candidate list) is actionable for Boss to authorize P5i and/or P5j.

## § 8 References

- P5g close-out: `reports/p5g-final-results.md`
- P5g findings memory: `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5g_findings.md`
- P5g design spec: `docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md` (§ 4.1a / § 7.1a / § 7.2 post-T0 amendments)
- P5g implementation plan: `docs/superpowers/plans/2026-05-20-ironmlx-p5g-gated-delta-net.md`
- Boss memory: `[feedback_design_rigor]`, `[feedback_serial_perf_experiments]`, `[feedback_no_spec_from_competitors]`, `[feedback_performance_stability_priority]`, `[feedback_design_philosophy]`, `[feedback_task_breakdown_bounded]`, `[feedback_iron_bench_priority]`, `[feedback_no_unnecessary_docs]`
- Reusable infra from P5g: `ironmlx/tests/p5g_t0_gated_delta_profile.rs` (HTTP-path harness), `ironmlx/src/main.rs` (tracing→stderr fix), `/tmp/p5g-env.sh` pattern
