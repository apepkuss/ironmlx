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
| 4 | **GDN sub-step** | P5g T0 已测 11-step Layer 2 breakdown (commit `52c39bd`),但 `[p5g-profile]` 旧 schema 缺 § 2.5a server-emitted fields (`request_id / prompt_tokens / seq / layer_idx / span_name / parent_span / start_ns / end_ns / mode`) — **不能直接参与 § 7.1 exclusive coverage tree** | **扩展 P5g harness emit 新 schema (Codex review v2 P2 #1 + v4 P1)** — modest code change: gated_delta_net.rs entry/exit barrier 的 `tracing::info!` format string 加新字段; harness Python aggregator 解析新字段 + substep `parent_span = attention_path` (NOT `decoder_layer_N` per Codex v4 P1)。**仍复用 P5g 已 verified 的 instrumentation 位置 + Phase D 抗污染设计** (Layer1\|Layer2 mode-gate, AblateX barrier-free)。新增 `[p5h-profile]` log line (跟 `[p5g-profile]` 平行 emit, 旧 P5g pipeline 不破)。T0a 实施 + rerun 同 PP set 下 UMA cold/warm pair。Cost ~11 min rerun wall (P5g T0 663s precedent) + ~2-4h code-change wall。P5g existing data 保留作 prior reference but not in P5h coverage gate。Kernel-level per-tile/per-shape timing 是 P5j (kernel rewrite) scope,不是 P5h attribution scope。 |
| 5 | **GatedAttention** | **P5g 完全未测**; 10/40 layers in `Qwen3.5-35B-A3B-4bit` (config.json `layer_types` "linear,linear,linear,full" × 10), full-attn O(S²), `attn_output_gate=true` 让 `q_proj` 输出含 gate (`Hq × D × 2`), `k_proj`/`v_proj` 用 KV-head dim (num_key_value_heads=2) | New 3-layer profile. Layer 1: entry/exit barrier. **Layer 2: 7-step code-backed taxonomy** (per T2 § 3 — `q_gate_k_v_proj`(`q_proj` with gate + `k_proj` + `v_proj`, separate not fused) / q_split_norm_reshape / mrope_apply / kv_mask_update / fused_sdpa / gate_sigmoid_mul / o_proj; `fused_sdpa` is `mlx::fast::scaled_dot_product_attention_on` and its internals — softmax / value matmul — are NOT separately measurable on production path). Layer 3 ablation: **conditional on T0b Phase D outcome** per § 2.5 / T2 conditional gate. |
| 6 | **MoE expert dispatch + LinearMLP** | P5g 未测; 40 decoder layers (10 含 full-attention, 30 含 linear-attention/GDN), 每层 SparseMoeBlock 含 `num_experts=256` + `num_experts_per_tok=8` (config.json verified) + 1 shared expert (`shared_expert_intermediate=512`) + gather_qmm routed compute + **sorted-routing pack/unpack 是 P5e shipped optimization, 不能藏在 gather bucket 内** (Codex review v2 P2 #2) | Per-layer 8-step code-backed taxonomy (per T3 § 3 — router_logits_softmax_topk / routing_sort_pack / gather_qmm_gate_up / swiglu_activation / gather_qmm_down / routing_unsort_weighted_reduce / shared_expert / moe_output_sum). **ROI math 必须从 `Qwen35MoeConfig` runtime values 来,不 hardcode** — routed work scales as `BS × num_experts_per_tok = BS × 8`。 |
| 7 | **lm_head + MLX eval/cache state** | P5e/P5f shipped lm_head fix; MLX eval barrier costs 未细测 | lm_head time + MLX `eval()` barrier latency + KVCache + GatedDeltaCache state-update cost |

### 2.3 3-layer profile protocol per layer (复用 P5g pattern + extend)

每个 layer 重复 P5g 的 Layer 1 / Layer 2 / Layer 3 protocol,具体扩展:

- **Layer 1 (boundary-isolated)**: entry barrier + exit barrier + emit `[p5h-profile] layer=<name> ...elapsed_us=N`. 估算该 layer 总占比。
- **Layer 2 (per-step breakdown)**: 每个 sub-op 用 `mlx::transforms::eval(&[&intermediate])?` materialize + timer push。append step_breakdown CSV 到 Layer 1 log line。
- **Layer 3 (shape-preserving ablation)**: 每个 sub-step 提 substitute (e.g., GatedDeltaNet step 5 compute_g substitute = zeros_like passthrough; GatedAttention `o_proj` substitute = identity-on-gated-output if scope permits). Mode-gated entry barriers off for AblateX (per P5g § 4.1a barrier-free invariant). **Layer 3 ablation 是 conditional on T0b Phase D root cause outcome** — per § 2.5 decision tree + T2/T3 conditional gates; H2/H4 verified primary 时 Layer 3 skip or replace with real-path microbenchmarks。**特别注意**: `fused_sdpa` (`mlx::fast::scaled_dot_product_attention_on`) 的 softmax/value matmul internals 是 fused MLX call,**不能在 production path 上单独 ablation**;只能 ablate 整个 `fused_sdpa` 步骤 (用 e.g. zeros tensor 替换其输出)。

**Critical**: Phase D ablation 在 P5g 全 negative (反常)。P5h T0b 必须先 investigate root cause —— substitute self-cost / cache divergence / kernel template variance / phase order thermal —— 否则 P5h 任何 ablation reading 也会被同样 anomaly 污染。

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

**Trace context propagation through call chain** (per Codex review v5 P1 #2 + v6 P2 #3 — without this, deep log sites in GDN/GatedAttention/MoE cannot emit `request_id` or `prompt_tokens` because HTTP-layer state doesn't reach them today):

v5 originally proposed a thread-local that carried only `request_id` and was set/cleared only around `model.batched_prefill(...)`. v6 review correctly flagged two problems: (a) `first_token_sampling` + `detok_format_first_content_chunk` + `pre_content_decode_steps` spans live outside `batched_prefill` but still need `request_id` for the schema, and (b) schema also requires `prompt_tokens` per record but a request_id-only thread-local can't carry it. v7 fixes by upgrading to a full trace context:

```rust
#[derive(Clone)]
struct P5hTraceContext {
    request_id: String,
    prompt_tokens: u32,
    routing_path: &'static str,  // "scheduler" | "gs_chunked"
}
thread_local! { static P5H_CURRENT_TRACE: RefCell<Option<P5hTraceContext>> = RefCell::new(None); }
```

1. `GenerateRequest` adds field `p5h_trace: Option<P5hTraceContext>` (gated by `p5h-profile` feature; default `None`). HTTP handler in `openai.rs` populates it (uuid for `request_id`, post-tokenize `prompt_ids.len()` for `prompt_tokens`, routing-decision result for `routing_path`).
2. `RequestState` (in `scheduler.rs`) carries the same field; `Scheduler::admit` copies it from `GenerateRequest` at admit time.
3. `P5H_CURRENT_TRACE` is set at the **root span entry** (start of `chat_completion` handler in `openai.rs`) and cleared at the **root span exit** (immediately after first non-empty content SSE write, or on error). Scope covers the ENTIRE root window — not just `batched_prefill`. This way `model_prefill_forward`, `first_token_sampling`, `pre_content_decode_steps`, `detok_format_first_content_chunk`, and any deep substep span all read the same context.

   **Per-thread set/clear contract** (per Codex review v7 P3 — the axum handler does not directly call deep model code; instrumentation runs on multiple threads):

   - **axum handler thread** (root start/end): sets `P5H_CURRENT_TRACE` at `chat_completion` entry; clears at root close. Used by `http_parse_render_tokenize` + (Lane A) the `AdmitReply` send timestamp inside `scheduler_admission`.
   - **`spawn_blocking` thread** (Lane B): `serve_via_gs_stream` body runs inside `tokio::task::spawn_blocking`. The handler MUST clone the trace context and `P5H_CURRENT_TRACE.set(cloned)` as the FIRST statement of the `spawn_blocking` closure, BEFORE `GenerationStream::new(...)`. Cleared as the LAST statement of the closure. Covers `gs_stream_init_and_chunk_loop` (including all `gs_chunk_N` + any deep substep inside the chunked prefill — though deep substeps are out of scope for Lane B, the context still propagates for any T5 sanity check).
   - **scheduler actor `driver_loop` thread** (Lane A): the actor thread is long-lived (one per `SchedulerActor`), not per-request. T0a MUST add explicit set/clear around each per-request operation that the actor performs:
     - Around `sched.prefill_admitted(&model_lock)` in `scheduler_actor::driver_loop` (currently `scheduler_actor.rs:304-307`): read the lone active row's `RequestState.p5h_trace`, set `P5H_CURRENT_TRACE`, run `prefill_admitted`, route events, then clear. This is the set/clear point that v7 left implicit.
     - Around every `sched.step(...)` call that contributes to `pre_content_decode_steps` (the same actor loop iteration set, before the step, cleared after event routing).
     - On `handle_admit_mid_*` calls (mid-batch admit): not P5h-relevant under `--b-max 1` but if any code path reaches them while `p5h-profile` is active, hard-fail (multi-row invariant violated).
   - **streaming forwarder thread** (Lane A `tokio::spawn` after `AdmitReply`): the forwarder reads `event_rx`, formats SSE chunks, writes via `tx.send`. It owns `sse_write_role_chunk` + `detok_format_first_content_chunk`. The axum handler MUST clone the trace context into the spawned task before spawn, and the spawn body sets `P5H_CURRENT_TRACE.set(cloned)` as its first statement (clears on task return).

   **Hard-fail under `p5h-profile` + `--b-max 1`** (per Codex review v7 P3): any instrumented site that attempts to read `P5H_CURRENT_TRACE` and finds `None` MUST `panic!` with the span_name in the message (NOT silently log empty fields). T0a verifies via a logging fixture: first emitted `[p5h-profile]` record per fresh request MUST have non-empty `request_id` AND `prompt_tokens > 0`; if either is missing, a cross-thread set/clear point is wrong and the harness fails before T0a closes.
4. Deep instrumentation sites (GDN entry/exit barriers, GatedAttention substep emit, MoE substep emit) read `P5H_CURRENT_TRACE` to populate `request_id` + `prompt_tokens` + (optionally) `routing_path` on `[p5h-profile]` log lines.

**Thread-local safety** (memory `[feedback_ffi_runtime_semantics]` caution — MLX `thread_local` encoder gotchas): the thread-local is set/cleared on the request-handler thread (axum + scheduler driver). MLX kernel dispatch + GPU execution happen on different threads, but those threads do NOT read `P5H_CURRENT_TRACE` — only the CPU-side entry/exit barriers do, on the same thread chain (axum handler → scheduler driver, which is `block_on`-pinned per request under `--b-max 1`). MLX `eval()` barriers are caller-side `array::wait()`, which executes back on the caller thread. No cross-thread propagation needed. If under `b_max=1` the scheduler driver dispatches work to a different thread (e.g., spawn for streaming forwarder), the thread-local must be cloned and re-set on the new thread before any instrumented span runs — T0a implementation must verify this end-to-end via a logging fixture (first log line under a fresh request → assert context populated; if not, the thread-local crossing point is missing).

**Single-active-row hard gate** (per Codex review v5 P1 #2): the thread-local design ONLY works if exactly one in-flight row exists during `prefill_admitted_inner` + any pre-content decode steps. P5h harness MUST start the server with `--b-max 1` (production default per `serve.rs:38`; P5h enforces). On `p5h-profile` feature, server panics at startup if `b_max > 1` OR if the scheduler ever observes `active_count() > 1` during a `[p5h-profile]`-emitting forward. Strict serial sweep (per memory `[feedback_serial_perf_experiments]`) also enforces only one in-flight request server-side.

**Server-emitted fields** (per `[p5h-profile]` log line, written by ironmlx):

| Field | Type | Semantics |
|---|---|---|
| `request_id` | string (uuid) | server-generated per request, T5 group-by key; populated from `P5H_CURRENT_TRACE.request_id` thread-local (see "Trace context propagation" above) |
| `routing_path` | string | "scheduler" \| "gs_chunked"; populated from `P5H_CURRENT_TRACE.routing_path` — needed so T5 can partition records into Lane A vs Lane B |
| `prompt_tokens` | int | server-measured (post-chat-template, post-tokenize); populated from `P5H_CURRENT_TRACE.prompt_tokens`; proxy for iron-bench `--prompt-len` once correlated |
| `seq` | int | sequence length at this forward (chunk size or 1 for decode) |
| `layer_idx` | int | GDN/full-attn layer 0..39 (-1 for non-decoder spans) |
| `span_name` | string | 'http_request_recv' / 'sched_admit' / 'gda_step_1a_in_proj_qkvz' / etc. |
| `parent_span` | string \| null | 上一级 span 名 (null = top-level root) |
| `start_ns` | u64 | monotonic clock start (ns) |
| `end_ns` | u64 | monotonic clock end (ns) |
| `mode` | string | 'off' / 'layer1' / 'layer2' / 'ablate-X' |

**Aggregator-injected fields** (added by T5 Python aggregator when joining server log records with the iron-bench client sweep CSV):

| Field | Type | Source |
|---|---|---|
| `pp` | int | iron-bench `--prompt-len` for the sweep cell that produced this record; joined via `(request_id, sweep_cell_timestamp_window)` ↔ iron-bench request log |
| `run_id` | int | iron-bench warmup/measured run index within the sweep cell; joined the same way |
| `bench_session_id` | string | optional T5 group-by for multi-session sweeps |

**Join key — single committed path (per Codex review v4 P2)**: prior v4 wording left "header OR wallclock fallback" as implementer choice. Both paths were under-specified (header path didn't actually exist end-to-end; wallclock path required a `server_request_start_wallclock` field that wasn't in the server-emitted schema). v5 commits to ONE concrete path; T0a delivers all three edits below or T0a does not close:

1. **server emit** (`openai.rs`, gated on `p5h-profile` feature): chat-completion response builder MUST set response header `X-Ironmlx-Request-Id: <uuid>` (same uuid that anchors every `[p5h-profile]` log record's `request_id` field). Streaming and non-streaming paths both set it. Small addition (~5 lines) to `serve_via_scheduler_stream`'s response construction.
2. **iron-bench capture** (`iron-bench/src/client.rs` + `iron-bench/src/report.rs`): both capture AND serializer schema gated on new `--capture-server-request-id` CLI flag (default off):
   - **Flag off** (default, non-P5h runs): `RequestResult.request_id` not set; `report.rs` CSV/JSON header + body emit zero `request_id`-related bytes; output is **byte-identical** to current iron-bench (per Codex review v5 P2 #3 — earlier wording was contradictory because it claimed byte-identical while always adding a column).
   - **Flag on** (P5h sweeps): `run_chat_completion` captures `X-Ironmlx-Request-Id` from `resp.headers()` BEFORE entering `resp.bytes_stream()`, populates `RequestResult.request_id: Option<String>`; serializer adds `request_id` column to CSV header + JSON object (appended after existing `finish_reason` to keep prior-column byte ordering intact for tools that read by name).
3. **aggregator join** (T5 Python): `(request_id)` is the sole join key — no wallclock fallback. T5 aggregator hard-fails if any P5h sweep cell shows < 100% 1:1 server↔client request_id match. Orphan rate > 0% per PP fails T5 gate (the gate threshold was "< 1%" in v4 — v5 tightens to "= 0%" since deterministic header propagation should never lose records under per-PP serial sweep).

**Wallclock fallback explicitly DROPPED**: v4 listed wallclock as fallback. Codex v4 P2 correctly noted server-emitted schema had no `server_request_start_wallclock` field. Rather than add a fragile fallback, v5 makes the header path the only path. Boss memory `[feedback_iron_bench_priority]` cautions against modifying iron-bench casually — the 3 edits above are scoped, behind a new CLI flag, and ship/revert independently.

**Routing precondition — dual-lane design** (per Codex review v5 P1 #1 + v6 P1 fact-check):

v5 proposed forcing `--prefill-chunk-size 0` to keep all PP on the scheduler path. v6 review correctly flagged this would divorce P5h from the production-default config (P5g actually measured the long-PP gap on the chunked GS path — see `reports/p5g-final-results.md:62`: "PP=4096 → 3 chunks (two 2048 + one 12); PP=16384 → 9 chunks"). Forcing single-shot would test a config production users never hit, and the resulting attribution would not transfer to the default-config gap that motivates P5h.

v7 switches to **dual-lane with production-default server config preserved** (no `--prefill-chunk-size` override):

- **Lane A — Scheduler path, PP ≤ default `prefill_chunk_size` (2048)**: PP ∈ {128, 512, 2048} routes through `serve_via_scheduler_stream` (per `openai.rs:404` — `prompt_len <= prefill_chunk_size` predicate). Full § 2.5a deep substep attribution applies (GDN/GatedAttention/MoE substep breakdown via wrapper spans `attention_path`/`mlp_path`). T0a/T1/T2/T3/T4 deep instrumentation is meaningful on this lane.
- **Lane B — Chunked GS path, PP > default `prefill_chunk_size`**: PP ∈ {4096, 8192, 16384} routes through `serve_via_gs_stream` (per `openai.rs:408`). Each request emits N chunks (PP=4096 → 3 chunks, PP=16384 → 9 chunks). P5h covers ONLY top-level chunked-path attribution: server-side root + chunk loop wall-time + per-chunk forward total + first-token sampling + sse_write. Deep substep attribution (per-chunk GDN/GatedAttention/MoE breakdown, with `chunk_idx` schema extension) is **out of scope for P5h** (deferred to P5h+1) — would multiply records per request by N and require schema additions.

T0a profile gate validates per-PP routing via a `routing_path: "scheduler" | "gs_chunked"` annotation on the root span; T5 aggregator partitions output into two attribution tables (Lane A deep, Lane B top-level). Long-PP P5j candidate ranking comes with explicit caveat: P5j ROI estimates on PP > 2048 are bounded by Lane-B granularity; if a P5j candidate needs per-substep evidence at long PP, P5h+1 chunked deep-attribution must run first.

**Server-only root** (per Codex review v2 P1 #2 + v3 P1 fact-check; iron-bench TTFT cross-process correlation deferred — would require iron-bench → ironmlx request-id propagation, out of P5h scope):

- **Root span**: `server_request_recv_to_first_content_sse_write` — from server's `axum` request-handler entry to the moment when the **first non-empty `delta.content` SSE chunk** is sent into the body channel (`tx.send(Ok(format_sse_data(&content_chunk)))` in `openai.rs::serve_via_scheduler_stream`'s detok loop). All `[p5h-profile]` records anchor under this server-side root.
- **Why not "first SSE write" (Codex v3 P1)**: the forwarder task spawned right after `AdmitReply` issues a synthetic role chunk (`delta.role = "assistant"`, `delta.content = ""`) BEFORE the first-batch prefill runs (per `openai.rs:546-564` + `scheduler_actor.rs:276-313`: admit reply is sent before prefill). If root ended at "first SSE write", prefill + first-token sampling would fall **outside** the root, defeating the entire attribution exercise. The role chunk write is itself a small child span (`sse_write_role_chunk`) under the root, not the root's terminal point.
- **Root terminal definition (exact)**: `end_ns = monotonic_ns()` captured at the instruction immediately after the first successful `tx.send(Ok(format_sse_data(&content_chunk)))` where `content_chunk.choices[0].delta.content` is non-empty. **`finish_reason` is irrelevant** — if the model's first token simultaneously carries a stop/length finish reason (e.g., `max_tokens=1` request, or first-token sentinel), the chunk still closes the root, because iron-bench TTFT counts that same chunk (per `iron-bench/src/client.rs:106-116` — TTFT only inspects `delta.content` non-empty, not `finish_reason`). Excluding finish-reason chunks would create a server-side root that is wider than client-side TTFT — wrong direction. Empty-content role/keepalive chunks still do NOT close the root.
- **Implementation note**: the `detok_format_first_content_chunk` span should still record `finish_reason_present: bool` as an annotation (useful diagnostic), but this annotation does NOT gate root closure.
- **Client transport residual** is computed as a SEPARATE diagnostic: `client_transport_residual_us = iron_bench_ttft_us - server_root_inclusive_us`. Not part of the exclusive tree; reported alongside in `reports/p5h-attribution.md` as a transport-overhead column.

**Top-level buckets under `server_request_recv_to_first_content_sse_write`** (mutually exclusive children; Lane-A scheduler-path schema — Lane-B chunked-GS schema in the dedicated subsection below):

1. `http_parse_render_tokenize` (server-side request parsing + chat template + tokenizer Encode)
2. `scheduler_admission` (admit queue + slot allocation + batch construction; ends at `AdmitReply` send)
3. `sse_write_role_chunk` (forwarder spawn + initial role chunk write; happens before prefill in current `openai.rs` flow)
4. `model_prefill_forward` (the full `model.batched_prefill(...)` call — embed + 40 decoder layers + final norm + `slice_last_and_project` lm_head; per `qwen3_5_moe/model.rs::batched_prefill` lines 240-258, this single call covers everything through producing first-token logits)
5. `first_token_sampling` (per-row sampler invocation after `batched_prefill` returns; per `scheduler.rs::prefill_admitted_inner` "three-stage dispatch")
6. `pre_content_decode_steps` (per Codex review v6 P2 #2 — if detok returns `Ok(None)` or empty string for the first prefill token, server does not send a content chunk yet and iron-bench does not record TTFT; scheduler may then run additional `Scheduler::step()` decode forwards + sample + detok until detok yields a non-empty string. This bucket covers all such pre-first-content decode iterations. Expected `inclusive_us == 0` for well-formed benchmark prompts where the first prefill token detokenizes to a visible character; if non-zero, T0a/T5 must surface it.)
7. `detok_format_first_content_chunk` (detok stream step + ChunkResponse serialize + first content SSE write — for the iteration that actually produces non-empty content, whether that came from prefill or from a pre-content decode step)
8. `unattributed_server_root` (explicit residual leaf — see "Residual leaves" below)

**Lane-B chunked-GS top-level buckets** (per "Routing precondition" — PP > `prefill_chunk_size`):

Lane-B uses a shallower tree (no deep substep nesting in P5h; deferred to P5h+1). **Bucket ordering MUST match actual `serve_via_gs_stream` flow** (per `openai.rs:416-470` + `generate.rs:945-1055` — per Codex review v7 P1): the chunked prefill loop lives inside `GenerationStream::new()`, which runs BEFORE the role chunk SSE is sent. The Lane-B order differs from Lane A here (Lane A's role chunk happens before prefill via the scheduler `AdmitReply` forwarder spawn; Lane B's role chunk happens after the whole prefill loop finishes because `GenerationStream::new()` is synchronous inside `spawn_blocking`). Carrying the Lane-A order over to Lane B (as v7 spec did, now fixed in v8) would leak the entire chunked prefill into `unattributed_server_root` and break long-PP attribution — the exact opposite of what P5h needs.

Children of `server_request_recv_to_first_content_sse_write` under Lane B, in actual wall-clock order:

1. `http_parse_render_tokenize` (same as Lane A — runs in the axum handler before `spawn_blocking`)
2. `gs_stream_init_and_chunk_loop` (the entire `GenerationStream::new(...)` call inside `spawn_blocking` — covers KV cache allocation + chunked prefill loop body + final chunk's full forward producing first-token logits). Children of this bucket:
   - `gs_kv_cache_alloc` (`model.make_cache(...)` call)
   - `gs_chunk_N` × ceil(prompt_len / prefill_chunk_size) (each chunk covers `[forward_text_hidden + cache update + eval]`; the final chunk covers `[batched_prefill-equivalent forward + lm_head]` producing first-token logits)
   - `unattributed_gs_stream_init_and_chunk_loop` (residual leaf)
3. `sse_write_role_chunk` (post-`GenerationStream::new` role chunk send — `openai.rs:441-457`. v7 had this before the chunk loop; v8 fix places it after)
4. `first_token_sampling` (first `stream.next_token()` call — runs sampler on the logits the final chunk produced)
5. `pre_content_decode_steps` (same semantics as Lane A bucket 6 — additional `stream.next_token()` iterations if first-token detok was empty)
6. `detok_format_first_content_chunk` (the `stream.next_token()` iteration that yields non-empty content + ChunkResponse serialize + SSE write — same as Lane A)
7. `unattributed_server_root` (residual leaf)

**Lane-B root closure invariant** (per Codex review v7 P1): the root span MUST cover the entire `GenerationStream::new(...)` wall-time. Root start = axum handler entry; root end = first non-empty content SSE write (per "Root terminal definition" above). Implementation MUST NOT close the root on role chunk send under Lane B, or `gs_stream_init_and_chunk_loop` falls outside the tree.

**T0a/T5 gate for pre_content_decode_steps**: per Codex review v6 P2 #2, every measured request MUST satisfy `pre_content_decode_steps.inclusive_us < 1ms` (within noise of "first prefill token detokenized to non-empty string"). If any sweep cell shows non-trivial pre-content decode time, T0a flags it before T0b dispatches; T5 close-out reports the count + investigates whether benchmark prompts need adjustment OR whether instrumentation needs to subdivide this bucket.

**`model_prefill_forward` children** (mutually exclusive; matches `batched_prefill` call chain):
- `embed_lookup` (token id → hidden_states; `text.embed_on(...)`)
- `decoder_layer_{0..39}` × 40 (one span per decoder layer inside `forward_post_embedding_on`)
- `final_norm_in_text_model` (the model-level RMSNorm before lm_head, inside `forward_post_embedding_on` tail)
- `slice_last_and_project_lm_head` (`slice_last_and_project` — slicing + `lm_head.forward_on`)
- `unattributed_model_prefill_forward` (explicit residual leaf)

**`decoder_layer_N` children** (mutually exclusive):
- `input_norm` (pre-attention RmsNorm)
- `attention_path` — wrapper span for GatedAttention OR GatedDeltaNet (whichever this layer is, per config `layer_types[N]`). The substep breakdown lives **under** `attention_path`, NOT directly under `decoder_layer_N`.
- `post_attention_norm` (post-attention RmsNorm)
- `mlp_path` — wrapper span for SparseMoeBlock OR shared LinearMLP (per config `mlp_only_layers`). The substep breakdown lives **under** `mlp_path`, NOT directly under `decoder_layer_N`.
- `residual_overhead` (the two residual adds + any layout shuffle around them)
- `unattributed_decoder_layer_N` (explicit residual leaf, only emitted if non-zero)

**`attention_path` (GatedAttention) children** = 7 substeps per § 2.2 #5 code-backed taxonomy; substep records MUST set `parent_span = attention_path` (NOT `decoder_layer_N` — per Codex review v4 P1 fix, otherwise the wrapper span goes missing and `attention_path` exclusive_us becomes its full inclusive_us, double-counting under coverage gate).
**`attention_path` (GatedDeltaNet) children** = P5g T0 11-step breakdown; substep records MUST set `parent_span = attention_path`. P5g instrumentation must be **extended** to emit P5h span-schema fields (per § 2.5a server-emitted table) + use `parent_span = attention_path` (NOT the old `parent_span = decoder_layer_N` proposal in v3/v4; see Codex review v4 P1 + v2 P2 #1, § 2.2 #4 + T0a).
**`mlp_path` (SparseMoeBlock) children** = 8 substeps per § 2.2 #6 code-backed taxonomy; substep records MUST set `parent_span = mlp_path` (NOT `decoder_layer_N` — same Codex v4 P1 reasoning).

**Residual leaves**:

Every non-leaf span MUST emit at most one `unattributed_<span_name>` child whose `inclusive_us = parent.inclusive_us - Σ accountable_children.inclusive_us`. If the residual is `0` (within ±1 µs noise), emission is OPTIONAL. The residual leaf is itself a leaf (no further children), and counts as **NOT-accountable** in the coverage gate (see § 7.1).

This explicit-residual pattern is what makes the coverage gate non-trivial:
- Without it: `Σ exclusive_us = root.inclusive_us` by tree identity (Codex P1 #1 — trivially passes even when no useful attribution emitted).
- With it: `coverage_gate = 1 - (Σ unattributed_*.inclusive_us / root.inclusive_us) ≥ 95%`. If instrumentation only emits the root, all of root's time becomes `unattributed_server_root.inclusive_us`, coverage = 0%, gate FAILS loudly.

**Exclusive time computation** (T5 aggregator):

```python
# Pseudocode:
for span in spans:
    span.inclusive_us = (span.end_ns - span.start_ns) / 1000
for span in spans (depth-first, children-first):
    span.exclusive_us = span.inclusive_us - sum(child.inclusive_us for child in span.children)
    assert span.exclusive_us >= -1.0, f"{span.span_name}: negative exclusive {span.exclusive_us}us — broken parent_span attribution"

# Structural invariant (always true if instrumentation correct — sanity check, NOT coverage gate):
root = find_root_span(spans)  # server_request_recv_to_first_content_sse_write
all_exclusive_sum = sum(s.exclusive_us for s in spans)
assert abs(all_exclusive_sum - root.inclusive_us) < 1.0  # tree identity (Codex P1 #1: this alone is trivial)

# Real coverage gate per § 7.1: only NON-residual leaf time counts as "accountable":
unattributed_total = sum(s.inclusive_us for s in spans if s.span_name.startswith("unattributed_"))
accountable_total = root.inclusive_us - unattributed_total
coverage_pct = accountable_total / root.inclusive_us
assert coverage_pct >= 0.95, f"coverage {coverage_pct:.1%} < 95% — instrumentation gaps in {[s.span_name for s in spans if s.span_name.startswith('unattributed_') and s.inclusive_us / root.inclusive_us > 0.01]}"
```

**Hard invariants**:
- `Σ all exclusive_us ≡ root inclusive_us` (tree identity — sanity check, alone insufficient per Codex P1 #1).
- `span.exclusive_us ≥ -1µs` for every span (negative = broken parent_span attribution).
- `coverage_pct = 1 - Σ unattributed_*.inclusive_us / root.inclusive_us ≥ 95%` is the real gate.
- Every non-leaf span MUST emit explicit `unattributed_<span_name>` if its residual > 1µs.

**Out of scope (P5h+1 if needed)**:
- Per-MLX-kernel internal timing (e.g. softmax inside SDPA). Production-path can't expose this without changing production code; document as P5h+1 MLX-kernel investigation if T5 attribution shows attention `fused_sdpa` as unattributable hotspot.

### 2.5 Phase D root cause investigation (T0b of P5h)

P5g flagged 4 个 hypothesis:

| # | Hypothesis | P5h investigation method |
|---|---|---|
| H1 | GPU thermal drift across 24 spawns | Phase order randomized rerun: Phase D first, then A/B/C. Compare Phase D values across orderings. |
| H2 | Substitute 自身有成本 | Substitute self-cost: 跑 Phase A (no profile) WITH `IRONMLX_P5G_PROFILE_MODE=ablate-X` enabled — measure substitute path vs original path direct comparison。 |
| H3 | Cache state divergence (AblateConv 不更新 conv_state) | Add new ablation variant: AblateConv + manual conv_state update (same as AblateNone but with substitute on Step 2b). Isolate cache-divergence effect from substitute effect. |
| H4 | Kernel template variance (g=0 input 触发 slow path) | Compare kernel dispatch under AblateComputeG (g=zeros) vs Phase A (g=normal) — measure kernel-dispatch elapsed time only, exclude pre/post processing. |

**Decision tree**:
- H1 verified → P5h all phases adopt randomized order + cool gates between phases.
- H2 verified → discard ablation upper-bound concept; use Phase B/C ranking only for candidate priority.
- H3 verified → cache state must be carefully preserved across ablation; substitute design 需新 guard pattern。
- H4 verified → ablation invalid for kernel-dispatch-time hotspots; must use real candidate impl benchmark instead。

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
- **Trace context propagation infrastructure** (per § 2.5a "Trace context propagation through call chain" — required for deep model log sites AND post-prefill spans to emit `request_id` + `prompt_tokens` + `routing_path`):
  - `GenerateRequest` (in `core/generate.rs`): add `p5h_trace: Option<P5hTraceContext>`, gated on `p5h-profile` feature; struct has `request_id` + `prompt_tokens` + `routing_path`
  - `RequestState` (in `core/scheduler.rs`): add same field; `Scheduler::admit` copies from `GenerateRequest`
  - `openai.rs` `chat_completion` handler (root span entry): set `P5H_CURRENT_TRACE` thread-local immediately on request entry; clear on root span exit (after first non-empty content SSE write OR on error). Scope covers ENTIRE root window (not just `batched_prefill`) — per v6 P2 #3 fix, this lets `first_token_sampling` / `pre_content_decode_steps` / `detok_format_first_content_chunk` spans also populate the context fields
  - If root handler dispatches to additional threads (e.g., scheduler driver `block_on`, streaming forwarder spawn), the trace context must be cloned + re-set on the new thread before any instrumented span runs. T0a verifies this via a logging fixture: first emitted `[p5h-profile]` record under a fresh request MUST have non-empty `request_id`; if any record's `request_id` is empty, the cross-thread propagation is missing
  - Deep instrumentation sites (GDN/GatedAttention/MoE entry/exit barriers) read the thread-local for `request_id` + `prompt_tokens` + `routing_path` fields on `[p5h-profile]` log lines
  - Server startup panic if `b_max > 1` when `p5h-profile` feature is active (single-active-row invariant)
- **Request-correlation infrastructure** (per § 2.5a "Join key" — single committed path):
  - `openai.rs` (chat-completion response builder): emit `X-Ironmlx-Request-Id: <uuid>` header on streaming + non-streaming responses, gated on `p5h-profile` feature; same uuid used as the `request_id` field in `GenerateRequest.p5h_request_id` + every `[p5h-profile]` log record
  - `iron-bench/src/client.rs`: capture `X-Ironmlx-Request-Id` from `resp.headers()` BEFORE entering `bytes_stream()`; add `request_id: Option<String>` to `RequestResult`. Capture path gated on new CLI flag.
  - `iron-bench/src/report.rs`: CSV/JSON serializer writes new `request_id` column **only when flag is on** (per § 2.5a P2 #3 fix — flag off keeps schema byte-identical to current; column appended after existing `finish_reason`)
  - New iron-bench CLI flag `--capture-server-request-id` gates BOTH capture path AND serializer schema (default off, on for P5h sweeps; off-state output is byte-identical to current iron-bench)
- **UMA hardening protocol** implementation: cold/warm pair measurement + variance check + automatic retry (per § 2.4)
- **GDN harness code extension** to emit `[p5h-profile]` log lines with new schema (per § 2.2 #4 + Codex review v2 P2 #1) — modest format-string change to existing entry/exit barriers in `gated_delta_net.rs`; `[p5g-profile]` lines kept in parallel for back-compat
- **GDN rerun** under P5h UMA protocol — same PP set as P5g T0, cold/warm pair per PP, exclusive span tree with `parent_span = attention_path` for GDN substeps (per § 2.5a wrapper-span structure, per Codex review v4 P1 fix; NOT `parent_span = decoder_layer_N` as v3/v4 incorrectly said)
- **Schema validation on GDN rerun** (T0a's hard gate, must pass before T0b starts):
  - Sum-to-root identity holds within ±1µs
  - All `exclusive_us ≥ -1µs`
  - GDN `attention_path` coverage_pct ≥ 95% (per § 7.1; per Codex review v4 P1 — `decoder_layer_N` coverage at T0a stage is meaningless because input_norm/post_attention_norm/mlp_path/residual_overhead are not yet instrumented and would all flow into `unattributed_decoder_layer_N`. Full `decoder_layer_N` coverage gate applies only at T5 after T1-T4 all land.)
  - UMA cold/warm variance ≤ ±2% per PP
  - iron-bench↔server `request_id` join rate = 100% across all sweep cells
- Output: P5h schema infrastructure, request-correlation infra (server header + iron-bench capture + CSV column), GDN protocol-consistent data under exclusive tree with `parent_span = attention_path`
- **GATE**: T0a must close before T0b dispatches. If schema gate fails, fix schema first; do NOT proceed to Phase D investigations until GDN data demonstrates schema works end-to-end.
- Commit: `feat(p5h-t0a): exclusive span schema + UMA hardening + GDN P5h-protocol rerun`

### T0b — Phase D root cause investigation (4 hypotheses, depends on T0a)

T0b only starts AFTER T0a closes (schema proven on GDN rerun). T0b reuses the now-validated schema infrastructure to emit Phase D records consistent with later T2/T3 ablation work.

- Phase D root cause investigation (4 hypotheses per § 2.5 decision tree):
  - H1 (thermal drift): phase-order randomized rerun (Phase D first, then A/B/C); compare values across orderings
  - H2 (substitute self-cost): run Phase A with `IRONMLX_P5G_PROFILE_MODE=ablate-X` enabled, directly compare substitute path vs original path
  - H3 (cache state divergence): add `ablate-conv-with-manual-cache-update` variant; isolate cache-divergence effect from substitute effect
  - H4 (kernel template variance): kernel-dispatch-only timing under AblateComputeG vs Phase A; exclude pre/post processing
- Decision-tree mapping (per § 2.5):
  - H1 primary → P5h all phases adopt randomized order + cool gates
  - H2 primary → discard ablation upper-bound; use Layer 2 ranking only for candidate priority
  - H3 primary → ablation requires cache-state-preserving substitute design (new guard pattern)
  - H4 primary → ablation invalid for kernel-dispatch-time hotspots; use real candidate impl benchmark instead
- Output: `reports/p5h-phase-d-root-cause.md` documenting primary root cause + mitigation + decision-tree binding for T2/T3 conditional gates
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
  - Edit 3: tail refactor + exit barrier + log emission. Wrapper span `attention_path` is opened by the decoder layer; substep records inside GatedAttention set `parent_span = attention_path` (per § 2.5a + Codex v4 P1, NOT `decoder_layer_N`).
- **Layer 2 step breakdown — code-backed taxonomy** (7 sub-steps matching actual `gated_attention.rs:120-276` production path with `attn_output_gate=true` per config; **not** decomposing the fused SDPA internals which are inside `mlx::fast::scaled_dot_product_attention_on`):
  1. `q_gate_k_v_proj` — three separate Linear projections (NOT fused QKV): `q_proj` outputs `Hq × D × 2` (queries concatenated with gate, since `attn_output_gate=true`); `k_proj` outputs `Hkv × D` (KV-head dim, GQA); `v_proj` outputs `Hkv × D` (KV-head dim). Single span covers all three.
  2. `q_split_norm_reshape` — split q output back into queries + gate halves + `q_norm`/`k_norm` (per-head RmsNorm) + reshapes/transposes to SDPA layout
  3. `mrope_apply` — `mrope.apply(&queries, &k, cos, sin)` rotary
  4. `kv_mask_update` — KV validity mask construction + `KVCache::update_and_fetch_on(k, v, lens, target)`
  5. `fused_sdpa` — `mlx::fast::scaled_dot_product_attention_on(...)` (fused MLX op; **softmax/value matmul internals inside, not separately measurable on production path** — if T5 shows this as unattributable hotspot, P5h+1 may investigate MLX-kernel-level)
  6. `gate_sigmoid_mul` — gate sigmoid + elementwise multiply on SDPA output (gate tensor came from `q_proj` second half)
  7. `o_proj` — `Linear::forward_on(&gated, target)` output projection
- **Layer 3 ablations — conditional on T0b Phase D outcome** (per § 2.5 decision tree):
  - **If T0b verifies H1 primary** (thermal drift): ablations OK, T2 runs Layer 3 with randomized phase order + cool gates。
  - **If T0b verifies H2 primary** (substitute self-cost): Layer 3 **skipped** for T2; replace with real-path microbenchmarks (e.g., swap `o_proj` with smaller dim variant compiled separately, measure end-to-end delta against baseline). Layer 1/2 still emitted.
  - **If T0b verifies H3 primary** (cache state divergence): Layer 3 requires cache-state-preserving substitute design — for GatedAttention, KV cache must remain valid across ablation (e.g., `ablate_fused_sdpa` substitute returns shape-preserving zeros but still calls `KVCache::update_and_fetch_on` to keep cache consistent for subsequent forwards).
  - **If T0b verifies H4 primary** (kernel template variance): Layer 3 invalid for any step that touches Metal kernels (especially `fused_sdpa`); skip Layer 3 for steps 4-5 (kv_mask_update + fused_sdpa); Layer 3 OK for pure op-level steps (q_gate_k_v_proj, q_split_norm_reshape, mrope_apply, gate_sigmoid_mul, o_proj).
- Run sweep + aggregate under exclusive span schema (per § 2.5a): wrapper span `attention_path` is the parent emitted by the decoder layer; the 7 GatedAttention substeps each set `parent_span = attention_path`
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
  - **H1 primary**: ablations OK with randomized order + cool gates
  - **H2 primary**: Layer 3 skipped, replace with real-path microbenchmarks (e.g., reduce experts_per_tok from 8 → 4 in a controlled fork, measure delta)
  - **H3 primary**: ablation must preserve routing index validity (don't break downstream attention KV slot allocation); pack/unpack steps must remain consistent (substitute can no-op compute but must still produce shape-correct outputs)
  - **H4 primary**: Layer 3 invalid for `gather_qmm_*` steps + `routing_sort_pack`/`routing_unsort_weighted_reduce` (all kernel-dispatch dependent); skip Layer 3 for steps 2-3 + 5-6; OK for steps 1, 4, 7-8
- Run sweep + aggregate under exclusive span schema (per § 2.5a): wrapper span `mlp_path` is the parent emitted by the decoder layer; the 8 MoE substeps each set `parent_span = mlp_path` (NOT `decoder_layer_N` per Codex v4 P1)
- **ROI math source**: derive `num_experts_per_tok = 8`, `moe_intermediate = 512`, `num_experts = 256` from `Qwen35MoeConfig` runtime values, NOT hardcoded constants in spec/report (which could drift if model config changes)
- Output: MoE per-PP attribution, router top-8 cost ratio, gather_qmm dominance check, shared expert vs routed cost split
- Commit: `test(p5h-t3): MoE expert + LinearMLP profile (code-backed taxonomy + conditional ablation)`

### T4 — lm_head + MLX eval/cache state + tokenization/first-eval profile

- `slice_last_and_project_lm_head` Linear quantized matmul timing (this is the `slice_last_and_project` child of `model_prefill_forward` per § 2.5a, NOT a sibling — model_prefill_forward boundary fix per Codex v4 P2 #4; single Linear not split)
- `first_token_sampling` per-row sampler invocation timing (sibling of `model_prefill_forward` under root, per § 2.5a top-level buckets)
- MLX `eval()` barrier latency at major sync points
- KVCache + GatedDeltaCache state-update cost (per-forward)
- Tokenization: tokenizer Encode time per prompt length (subspan of `http_parse_render_tokenize`)
- First-eval (JIT compile + kernel warmup) one-shot cost per (model, prompt_shape) pair
- Run sweep, aggregate
- Output: `slice_last_and_project_lm_head` occupancy, `first_token_sampling` cost, first-eval amortization (短 PP suspect), tokenization fixed cost
- Commit: `test(p5h-t4): lm_head + tokenization + MLX state profile`

### T5 — Cross-layer attribution synthesis + P5i/P5j candidate ranking + close-out report

- Aggregate T0a (GDN rerun) + T0b (Phase D resolution) + T1-T4 measurements into per-PP exclusive attribution table per § 2.5a span schema
- Compute `exclusive_us = inclusive_us - sum(children.inclusive_us)` for every span; verify `span.exclusive_us ≥ -1µs` invariant per § 2.5a
- Sum-to-root identity (`Σ all spans' exclusive_us ≡ root.inclusive_us within ±1µs`) is a **tree-property sanity check only**, NOT a coverage gate (per Codex review v2 P1 #1 — this identity is trivially true and useless as a quality bar)
- **Exclusive coverage gate** per § 7.1 (residual-based, single source of truth — DO NOT redefine here):

  ```
  root_wall_us = root_span("server_request_recv_to_first_content_sse_write").inclusive_us
  unattributed_total_us = Σ s.inclusive_us  for s in all_spans where s.span_name.startswith("unattributed_")
  coverage_pct = 1 - (unattributed_total_us / root_wall_us)
  gate: coverage_pct ≥ 95%  per PP
  ```

  If a future revision changes the gate, change § 7.1 first and reference it from here — do not duplicate the formula in two places. (Codex review v3 P2 caught duplicate trivial formula in this bullet during the v2 round.)
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

- T0a exclusive span schema validator: assert `sum(child.inclusive) ≤ parent.inclusive` for all parent spans in test fixture; assert sum-to-root identity within ±1µs; assert per-span `exclusive_us ≥ -1µs`
- T0a GDN rerun: P5h-protocol GDN data emitted under exclusive tree with GDN substeps' `parent_span = attention_path` (per § 2.5a + Codex v4 P1); GDN `attention_path` coverage_pct ≥ 95% under § 7.1 residual-based gate (full `decoder_layer_N` coverage gate applies only at T5); UMA cold/warm variance ≤ ±2% per PP; iron-bench↔server `request_id` join rate = 100%
- **T0a HARD GATE**: T0a's coverage + schema invariants must pass before T0b dispatches (per § 3 T0a). If schema gate fails on GDN rerun, fix schema before any Phase D investigation work.
- T0b Phase D root cause: 4 hypotheses (H1-H4) resolved per § 2.5 decision tree, OR explicit unresolved-list documented in `reports/p5h-phase-d-root-cause.md`; T2/T3 conditional ablation gates bound per T0b outcome
- T5 attribution: per § 7.1 residual-based exclusive coverage gate (`coverage_pct = 1 - Σ unattributed_*.inclusive_us / root.inclusive_us ≥ 95%` per PP, root = `server_request_recv_to_first_content_sse_write`); P5i/P5j candidate ranking emitted with ROI estimate ranges + Scope gate trigger status; `client_transport_residual_us` reported as separate diagnostic column

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
- **Multi-row in-flight attribution** (per § 2.5a "Single-active-row hard gate"): `P5H_CURRENT_TRACE` thread-local design assumes exactly one active row. Multi-row attribution would need per-row mlx::Stream / per-row span context; out of P5h scope.
- prefill_chunk_size sweep (becomes P5j candidate if T2 GatedAttention long-PP O(S²) supports it)
- omlx PagedCache style port (out per memory `[feedback_design_philosophy]`)
- mlx::compile wrap (still blocked by 4 safe-wrapper API gaps from P5e T2)
- GatedAttention algorithmic redesign (P5h measures it; redesign 在 P5j 如果 measured 必要)

## § 7 Success Criteria

### 7.1 Exclusive attribution coverage gate (P1-fix from Codex review v1 + v2)

T5 must produce a per-PP exclusive attribution table built **only** from same-protocol P5h measurements (P5g existing data excluded; see § 2.2 #4 + T0a GDN rerun). Coverage is **residual-based** (Codex review v2 P1 #1 — the naive `Σ exclusive ≡ root.inclusive` formulation is a tree identity, trivially true and useless as a gate). Coverage computed as:

```
root_wall_us = root_span("server_request_recv_to_first_content_sse_write").inclusive_us
unattributed_total_us = Σ s.inclusive_us  for s in all_spans where s.span_name.startswith("unattributed_")
accountable_us = root_wall_us - unattributed_total_us
coverage_pct = accountable_us / root_wall_us
            = 1 - (unattributed_total_us / root_wall_us)
```

The root is **server-side only** (`server_request_recv_to_first_content_sse_write`, per § 2.5a Codex v2 P1 #2). Client/transport latency is reported as a separate `client_transport_residual_us = iron_bench_ttft_us - root_wall_us` diagnostic column, NOT included in the coverage gate.

**Hard invariants** (per § 2.5a):
- `coverage_pct ≥ 95%` per PP (else identify which `unattributed_<span>` dominates → add instrumentation for that span's children → re-run before close-out)
- `span.exclusive_us ≥ -1µs` for every emitted span (negative beyond noise = broken parent_span attribution, MUST fix)
- `Σ all spans' exclusive_us ≡ root_wall_us` within ±1µs (tree identity sanity check; alone INSUFFICIENT as a coverage gate per Codex review v2 P1 #1)
- No bucket can be counted under two different parents (mutually exclusive tree)
- Every non-leaf span MUST emit an explicit `unattributed_<span_name>` leaf if its residual > 1µs (per § 2.5a "Residual leaves")

This **replaces** prior naive "sum medians ≥ 95%" gate which double-counted nested spans (Codex review v1 P1 #1) and the equally-naive "`Σ exclusive_us / root.inclusive_us`" formulation (Codex review v2 P1 #1, which is a tree identity).

### 7.2 P5h ship gate (T5 close-out gate)

1. **Exclusive attribution coverage** per § 7.1: `coverage_pct = 1 - Σ unattributed_*.inclusive_us / root.inclusive_us ≥ 95%` per PP (residual-based; root = `server_request_recv_to_first_content_sse_write`); `exclusive_us ≥ -1µs` for every emitted span; `client_transport_residual_us` reported separately (not part of gate)
2. **Protocol-consistent data — dual-lane explicit** (per Codex review v7 P2 + § 2.5a "Routing precondition"):
   - **Lane A** (PP ∈ {128, 512, 2048}, scheduler path): full deep substep attribution — HTTP/scheduler/admission (T1) + GDN (T0a rerun with `parent_span = attention_path`) + GatedAttention (T2, 7 substeps under `attention_path`) + MoE (T3, 8 substeps under `mlp_path`) + lm_head/MLX state (T4) + Phase D resolution (T0b) — all measured under same UMA hardening + exclusive span schema with trace context correlation; § 7.1 residual coverage ≥ 95% per PP.
   - **Lane B** (PP ∈ {4096, 8192, 16384}, chunked GS path): top-level only per § 2.5a Lane-B bucket list — server root + `gs_stream_init_and_chunk_loop` (with per-`gs_chunk_N` timing) + `sse_write_role_chunk` + `first_token_sampling` + `pre_content_decode_steps` + `detok_format_first_content_chunk`. Deep GDN/GatedAttention/MoE/lm_head substep attribution under chunked path is **explicitly out of scope** (deferred to P5h+1 per § 6); Lane B coverage gate measured only against top-level buckets, still ≥ 95%.
   - P5g existing data remains as prior reference only, excluded from both lanes' coverage gates.
   - P5j long-PP candidate ranking from Lane B carries explicit "bounded by Lane-B granularity" caveat; any P5j candidate requiring per-substep evidence at long PP must defer to P5h+1 chunked deep-attribution before P5j dispatch.
3. **UMA hardening verified**: cross-repeat (cold/warm pair) measurement variance ≤ ±2% per PP per metric per layer (per § 2.4 protocol)
4. **Phase D root cause** (T0b output): one of H1-H4 identified primary (mitigation proposed) OR explicit unresolved hypothesis list with proposed next investigation path (per § 2.5 decision tree); T2/T3 Layer 3 conditional ablation gates bound per T0b outcome
5. **P5i + P5j candidate ranking**: each candidate has expected ROI range (number-anchored), Scope gate trigger status, 实施优先级
6. **Target feasibility assessment**: honest verdict on "全 PP omlx+10% achievable in P5i+P5j" — if not, partial-target proposal for Boss decision
7. **Reusable infra delivered**: exclusive span schema infrastructure (per § 2.5a) + UMA hardening protocol (per § 2.4) + GatedAttention 3-layer profile harness (per § 2.2 #5 code-backed taxonomy with `attn_output_gate=true`) + MoE 8-step profile harness (per § 2.2 #6 sorted-routing path + `Qwen35MoeConfig` runtime values) — all usable in P5i+P5j+P5h+1
8. **Validation gates pass per task** (T0a/T0b/T1-T5 each independently green; per § 4)
9. **T0a HARD GATE passed** before T0b/T2/T3 dispatched: schema sum-to-root invariant + per-span exclusive_us ≥ -1µs + GDN `attention_path` coverage ≥ 95% (per Codex v4 P1 — `decoder_layer_N` coverage at T0a is meaningless since norms/mlp/residual are not yet instrumented) + UMA cold/warm variance ≤ ±2% per PP + iron-bench↔server `request_id` join rate = 100% all verified on GDN rerun data (per § 3 T0a + § 4)

P5h 整体 success = all 9 gates PASS, output (attribution report + P5i/P5j candidate list) is actionable for Boss to authorize P5i and/or P5j.

## § 8 References

- P5g close-out: `reports/p5g-final-results.md`
- P5g findings memory: `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5g_findings.md`
- P5g design spec: `docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md` (§ 4.1a / § 7.1a / § 7.2 post-T0 amendments)
- P5g implementation plan: `docs/superpowers/plans/2026-05-20-ironmlx-p5g-gated-delta-net.md`
- Boss memory: `[feedback_design_rigor]`, `[feedback_serial_perf_experiments]`, `[feedback_no_spec_from_competitors]`, `[feedback_performance_stability_priority]`, `[feedback_design_philosophy]`, `[feedback_task_breakdown_bounded]`, `[feedback_iron_bench_priority]`, `[feedback_no_unnecessary_docs]`
- Reusable infra from P5g: `ironmlx/tests/p5g_t0_gated_delta_profile.rs` (HTTP-path harness), `ironmlx/src/main.rs` (tracing→stderr fix), `/tmp/p5g-env.sh` pattern
