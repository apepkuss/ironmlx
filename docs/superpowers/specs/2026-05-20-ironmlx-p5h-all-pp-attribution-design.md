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
| 4 | **GDN sub-step** | P5g T0 已测 11-step Layer 2 breakdown + per-step eval barriers (commit `52c39bd`); Step 7 kernel dispatch + Step 2c cache update 已 eagerly evaluated。**但 P5g data 测于 no-UMA-hardening protocol,不能直接放入 P5h 95% coverage gate (per § 7.1)** | **复用 P5g harness (instrumentation code, `p5g-profile` feature) 但 P5h protocol 下 rerun** — 同 PP set, UMA cold/warm pair, exclusive span schema (`parent_span = decoder_layer_N`)。Code 不动 (P5g 已 ship),只 rerun + 重新 aggregate。Cost ~11 min wall (P5g T0 663s precedent)。**Output**: GDN sub-step under P5h protocol, comparable with other layers in T5 attribution. P5g existing data 保留作 prior reference but not in coverage gate。Kernel-level per-tile/per-shape timing 是 P5j (kernel rewrite) scope,不是 P5h attribution scope。 |
| 5 | **GatedAttention** | **P5g 完全未测**; 10/40 layers in `Qwen3.5-35B-A3B-4bit` (config.json `layer_types` "linear,linear,linear,full" × 10), full-attn O(S²), `attn_output_gate=true` | New 3-layer profile. Layer 1: entry/exit barrier. **Layer 2: 7-step code-backed taxonomy** (see T2 § 3 for full list — qkv_proj / q_split_norm_reshape / mrope_apply / kv_mask_update / fused_sdpa / gate_sigmoid_mul / o_proj; `fused_sdpa` is `mlx::fast::scaled_dot_product_attention_on` and its internals — softmax / value matmul — are NOT separately measurable on production path). Layer 3 ablation: **conditional on T0 Phase D outcome** per § 2.3 / T2 conditional gate. |
| 6 | **MoE expert dispatch + LinearMLP** | P5g 未测; 40 decoder layers (10 含 full-attention, 30 含 linear-attention/GDN), 每层 SparseMoeBlock 含 `num_experts=256` + `num_experts_per_tok=8` (config.json verified) + 1 shared expert (`shared_expert_intermediate=512`) + gather_qmm routed compute | Per-layer router top-8 selection cost + gather_qmm routed matmul + shared expert LinearMLP + final combine. **ROI math 必须从 `Qwen35MoeConfig` runtime values 来,不 hardcode** — routed work scales as `BS × num_experts_per_tok = BS × 8`, 不是 prior memory 误记的 `BS × 4`。 |
| 7 | **lm_head + MLX eval/cache state** | P5e/P5f shipped lm_head fix; MLX eval barrier costs 未细测 | lm_head time + MLX `eval()` barrier latency + KVCache + GatedDeltaCache state-update cost |

### 2.3 3-layer profile protocol per layer (复用 P5g pattern + extend)

每个 layer 重复 P5g 的 Layer 1 / Layer 2 / Layer 3 protocol,具体扩展:

- **Layer 1 (boundary-isolated)**: entry barrier + exit barrier + emit `[p5h-profile] layer=<name> ...elapsed_us=N`. 估算该 layer 总占比。
- **Layer 2 (per-step breakdown)**: 每个 sub-op 用 `mlx::transforms::eval(&[&intermediate])?` materialize + timer push。append step_breakdown CSV 到 Layer 1 log line。
- **Layer 3 (shape-preserving ablation)**: 每个 sub-step 提 substitute (e.g., GatedDeltaNet step 5 compute_g substitute = zeros_like passthrough; GatedAttention `o_proj` substitute = identity-on-gated-output if scope permits). Mode-gated entry barriers off for AblateX (per P5g § 4.1a barrier-free invariant). **Layer 3 ablation 是 conditional on T0 Phase D root cause outcome** — per § 2.5 decision tree + T2/T3 conditional gates; H2/H4 verified primary 时 Layer 3 skip or replace with real-path microbenchmarks。**特别注意**: `fused_sdpa` (`mlx::fast::scaled_dot_product_attention_on`) 的 softmax/value matmul internals 是 fused MLX call,**不能在 production path 上单独 ablation**;只能 ablate 整个 `fused_sdpa` 步骤 (用 e.g. zeros tensor 替换其输出)。

**Critical**: Phase D ablation 在 P5g 全 negative (反常)。P5h T0 必须先 investigate root cause —— substitute self-cost / cache divergence / kernel template variance / phase order thermal —— 否则 P5h 任何 ablation reading 也会被同样 anomaly 污染。

### 2.4 UMA cache state hardening protocol

P5g T4 暴露: sweep_full Qwen3.5-4B 之后跑 ironmlx serve restart, 3-way bench 测 ironmlx 数据 -20% vs T1-start baseline (same HEAD, ~25 min earlier)。Cool 5 min 后重测完全恢复匹配 P5f baseline。

**Hypothesis**: sweep_full 加载 Qwen3.5-4B 4-bit weights, evict Qwen3.5-MoE-35B 在 Apple Silicon UMA 中的 weight layout / page-table state。ironmlx serve 后续 load 17.5GB 进 sub-optimal cache state。

**P5h hardening protocol**:

1. **Phase-isolated spawn**: 不同 model 的 inference (e.g., sweep_full Qwen3.5-4B vs MoE-A3B bench) 不能背靠背跑;之间 cool ≥ 5 min。
2. **Cold-start baseline + warm reading 双值报告**: 每 bench iteration 跑 2 次,第 1 次 "cold" (cache 可能不最优), 第 2 次 "warm"。Report 标 both。Variance > ±2% 触发 cool-then-retry。
3. **UMA pressure probe**: T0 加一个 sanity check — 测 ironmlx PP=2048 在 cold-start vs warm-reading,确认 ≤ ±2%。否则 abort + investigate。
4. **Strict serial 跨 server** (per `feedback_serial_perf_experiments`): 一次只起一个 server, 完全 kill + cool ≥ 30s + lsof port-free 再起下一个。
5. **Document in spec**: 任何后续 phase / report 引用 measurement, must annotate cold/warm state + hardening protocol applied。

### 2.5a Exclusive timing span schema (foundation for T5 attribution gate)

**Problem**: P5g 的 boundary-isolated medians per layer 不构成 mutually-exclusive timing tree。HTTP TTFT contains scheduler; scheduler contains forward dispatch; forward contains decoder layers; decoder layers contain GatedAttention/GDN/MoE; lm_head + MLX eval/cache 可能 overlap parent spans。若 T5 简单 sum medians, 要么 double-count (nested spans 重复算) 要么 leave gaps (同级 spans 之间未被 instrument 的边缘)。"95% wall-time accounted" gate 在此 schema 下 trivially mismeasure。

**Solution**: 单一 exclusive parent-child span tree。每个 `[p5h-profile]` record 必须含 schema fields:

| Field | Type | Semantics |
|---|---|---|
| `request_id` | string (uuid or seq#) | 唯一 request 标识, T5 group-by |
| `run_id` | int | warmup/measured run index within request |
| `pp` | int | iron-bench `--prompt-len`, T5 group-by |
| `seq` | int | sequence length at this forward (chunk size or 1 for decode) |
| `layer_idx` | int | GDN/full-attn layer 0..39 (-1 for non-decoder spans) |
| `span_name` | string | 'http_request_recv' / 'sched_admit' / 'gda_step_1a_in_proj_qkvz' / etc. |
| `parent_span` | string \| null | 上一级 span 名 (null = top-level root) |
| `start_ns` | u64 | monotonic clock start (ns) |
| `end_ns` | u64 | monotonic clock end (ns) |
| `mode` | string | 'off' / 'layer1' / 'layer2' / 'ablate-X' |

**Top-level span buckets** (mutually exclusive, sum = end-to-end wall time):
1. `client_network` (iron-bench client side — TTFT measurement boundary outside ironmlx)
2. `http_parse_render_tokenize` (server-side request parsing + chat template + tokenizer Encode)
3. `scheduler_admission` (admit queue + slot allocation + batch construction)
4. `model_prefill_forward` (top-level forward call into TextModel)
5. `final_norm_lm_head_first_token` (post-decoder norm + lm_head Linear + first-token sampling)
6. `response_serialization` (SSE format + body write)

**`model_prefill_forward` children** (mutually exclusive, sum = forward time):
- `embed_lookup` (token id → hidden_states)
- `decoder_layer_{0..39}` × 40 (one span per decoder layer)
- `final_norm_in_text_model` (the model-level RMSNorm before lm_head, if any)

**`decoder_layer_N` children** (mutually exclusive):
- `input_norm` (pre-attention RmsNorm)
- `attention_path` — GatedAttention OR GatedDeltaNet (whichever this layer is, per config `layer_types[N]`)
- `post_attention_norm` (post-attention RmsNorm)
- `mlp_path` — SparseMoeBlock OR shared LinearMLP (per config `mlp_only_layers`)
- `residual_overhead` (the two residual adds + any layout shuffle around them)

**`attention_path` (GatedAttention) children** (per § 2.2 #5 code-backed taxonomy below).
**`attention_path` (GatedDeltaNet) children** = P5g T0 11-step breakdown (复用 schema).
**`mlp_path` (SparseMoeBlock) children** (per § 2.2 #6 code-backed taxonomy below).

**Exclusive time computation** (T5 aggregator):

```python
# Pseudocode:
for span in spans:
    span.inclusive_us = (span.end_ns - span.start_ns) / 1000
for span in spans (depth-first, children-first):
    span.exclusive_us = span.inclusive_us - sum(child.inclusive_us for child in span.children)
# T5 95% gate:
total_wall_us = root_span.inclusive_us
sum_buckets = sum(span.exclusive_us for span in spans)
assert abs(total_wall_us - sum_buckets) / total_wall_us < 0.05  # ≥95% coverage
```

**Hard invariants**:
- Sum of all exclusive_us ≡ root inclusive_us (mathematical identity if instrumentation is correct).
- Any span's exclusive_us ≥ 0 (negative = child span exceeded parent, indicates broken parent_span attribution).
- 95% gate is therefore not "≥95% of buckets emitted records" but "≥95% of root wall-time covered by sub-buckets that share root as ancestor".

**Out of scope (P5h+1 if needed)**:
- Per-MLX-kernel internal timing (e.g. softmax inside SDPA). Production-path can't expose this without changing production code; document as P5h+1 MLX-kernel investigation if T5 attribution shows attention `fused_sdpa` as unattributable hotspot.

### 2.5 Phase D root cause investigation (T0 of P5h)

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

**Out-of-scope**: P5h T0 只 identify root cause + propose mitigation. Actual substitute redesign 在 P5h T1+ 各 layer profile 应用。

## § 3 Tasks decomposition (6 tasks per writing-plans guideline)

### T0 — Pre-flight + exclusive span schema + UMA hardening protocol + Phase D root cause investigation + GDN rerun under P5h protocol

- Branch verify + Cargo feature `p5h-profile` add (alongside `p5g-profile`, both can be on simultaneously)
- **Exclusive span schema infrastructure** per § 2.5a — Rust span tracker + log emission format (`request_id` / `run_id` / `parent_span` / `start_ns` / `end_ns`); Python aggregator computes `exclusive_us = inclusive_us - sum(children_us)`; assert sum to root invariant
- UMA hardening protocol implementation: cold/warm pair measurement + variance check + automatic retry (per § 2.4 protocol)
- Phase D root cause: 4 investigation sub-tasks (H1 randomized order / H2 substitute self-cost / H3 cache divergence / H4 kernel template variance per § 2.5 decision tree)
- **GDN rerun**: rerun P5g GDN T0 sweep (复用 commit `52c39bd` instrumentation, `p5g-profile` feature) **under P5h UMA protocol + exclusive span schema** so GDN data is comparable with other layers in T5 attribution gate. P5g existing data 保留作 prior reference but excluded from coverage gate. Cost ~11 min (P5g T0 663s precedent).
- Output: hardening protocol spec, Phase D root-cause report (`reports/p5h-phase-d-root-cause.md`), GDN protocol-consistent data, reusable infra in test harness
- **T0 sprawl risk note** (per Codex review residual risk): T0 combines schema infra + UMA retry + 4 Phase D investigations + GDN rerun. If subagent execution shows sprawl (e.g. > 8 hours wall, or > 800 lines of new instrument code), split into T0a (schema + UMA + GDN rerun) + T0b (Phase D root cause) before touching T2/T3 GatedAttention/MoE instrumentation.
- Commit: `feat(p5h-t0): exclusive span schema + UMA hardening + Phase D root cause + GDN P5h-protocol rerun`

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
  - Edit 3: tail refactor + exit barrier + log emission, `parent_span = decoder_layer_N`
- **Layer 2 step breakdown — code-backed taxonomy** (7 sub-steps matching actual `gated_attention.rs:120-276` production path; **not** decomposing the fused SDPA internals which are inside `mlx::fast::scaled_dot_product_attention_on`):
  1. `qkv_proj` — Linear(hidden→3 × num_heads × head_dim) projection split into q/k/v
  2. `q_split_norm_reshape` — split + `q_norm`/`k_norm` (per-head RmsNorm) + reshapes/transposes to SDPA layout
  3. `mrope_apply` — `mrope.apply(&queries, &k, cos, sin)` rotary
  4. `kv_mask_update` — KV validity mask construction + `KVCache::update_and_fetch_on(k, v, lens, target)`
  5. `fused_sdpa` — `mlx::fast::scaled_dot_product_attention_on(...)` (fused MLX op; **softmax/value matmul internals inside, not separately measurable on production path** — if T5 shows this as unattributable hotspot, P5h+1 may investigate MLX-kernel-level)
  6. `gate_sigmoid_mul` — `attn_output_gate=true` per config → gate sigmoid + elementwise multiply on SDPA output
  7. `o_proj` — `Linear::forward_on(&gated, target)` output projection
- **Layer 3 ablations — conditional on T0 Phase D outcome** (per § 2.5 decision tree):
  - **If T0 verifies H1 primary** (thermal drift): ablations OK, T2 runs Layer 3 with randomized phase order + cool gates。
  - **If T0 verifies H2 primary** (substitute self-cost): Layer 3 **skipped** for T2; replace with real-path microbenchmarks (e.g., swap `o_proj` with smaller dim variant compiled separately, measure end-to-end delta against baseline). Layer 1/2 still emitted.
  - **If T0 verifies H3 primary** (cache state divergence): Layer 3 requires cache-state-preserving substitute design — for GatedAttention, KV cache must remain valid across ablation (e.g., `ablate_fused_sdpa` substitute returns shape-preserving zeros but still calls `KVCache::update_and_fetch_on` to keep cache consistent for subsequent forwards).
  - **If T0 verifies H4 primary** (kernel template variance): Layer 3 invalid for any step that touches Metal kernels (especially `fused_sdpa`); skip Layer 3 for steps 4-5 (kv_mask_update + fused_sdpa); Layer 3 OK for pure op-level steps (qkv_proj, q_split_norm_reshape, mrope_apply, gate_sigmoid_mul, o_proj).
- Run sweep + aggregate under exclusive span schema (per § 2.5a) with `parent_span = decoder_layer_N`
- Output: GatedAttention per-PP occupancy table (7-step breakdown), top-3 step ranking, long-PP O(S²) growth verification (PP=128 to PP=16384 step ratios)
- Commit: `test(p5h-t2): GatedAttention 3-layer profile (code-backed taxonomy + conditional ablation)`

### T3 — MoE expert dispatch + LinearMLP profile

- Read `ironmlx/src/nn/sparse_moe.rs` (or equivalent SparseMoeBlock module) — 40 decoder layers each contain MoE (`num_experts=256`, `num_experts_per_tok=8`, `moe_intermediate=512`, `shared_expert_intermediate=512` per config.json verified)
- Add 3-edit instrumentation pattern (mirror P5g GDN + T2 GatedAttention)
- **Layer 2 step breakdown — code-backed taxonomy**:
  1. `router_topk` — gating linear + softmax + top-8 selection (`BS × 256 → BS × 8` indices + weights)
  2. `gather_qmm_gate_up` — gather + quantized matmul gate/up projections for routed tokens (`BS × 8 × moe_intermediate`)
  3. `gather_qmm_down` — gather + quantized matmul down projection (`BS × 8 × hidden`)
  4. `expert_combine` — scatter-add by routing indices + weight
  5. `shared_expert` — separate LinearMLP for the shared expert (`BS × shared_expert_intermediate × 2 + BS × hidden`)
  6. `moe_output_sum` — final residual combining routed + shared outputs
- **Layer 3 ablations — conditional on T0 Phase D outcome** (same gating logic as T2):
  - **H1 primary**: ablations OK with randomized order + cool gates
  - **H2 primary**: Layer 3 skipped, replace with real-path microbenchmarks (e.g., reduce experts_per_tok from 8 → 4 in a controlled fork, measure delta)
  - **H3 primary**: ablation must preserve routing index validity (don't break downstream attention KV slot allocation)
  - **H4 primary**: Layer 3 invalid for `gather_qmm_*` steps (kernel-dispatch dependent); skip Layer 3 for steps 2-3; OK for steps 1, 4-6
- Run sweep + aggregate under exclusive span schema (per § 2.5a) with `parent_span = decoder_layer_N` (sub-parent `mlp_path`)
- **ROI math source**: derive `num_experts_per_tok = 8`, `moe_intermediate = 512`, `num_experts = 256` from `Qwen35MoeConfig` runtime values, NOT hardcoded constants in spec/report (which could drift if model config changes)
- Output: MoE per-PP attribution, router top-8 cost ratio, gather_qmm dominance check, shared expert vs routed cost split
- Commit: `test(p5h-t3): MoE expert + LinearMLP profile (code-backed taxonomy + conditional ablation)`

### T4 — lm_head + MLX eval/cache state + tokenization/first-eval profile

- lm_head Linear quantized matmul timing (Step 8-like; for lm_head, single Linear not split)
- MLX `eval()` barrier latency at major sync points
- KVCache + GatedDeltaCache state-update cost (per-forward)
- Tokenization: tokenizer Encode time per prompt length
- First-eval (JIT compile + kernel warmup) one-shot cost per (model, prompt_shape) pair
- Run sweep, aggregate
- Output: lm_head occupancy, first-eval amortization (短 PP suspect), tokenization fixed cost
- Commit: `test(p5h-t4): lm_head + tokenization + MLX state profile`

### T5 — Cross-layer attribution synthesis + P5i/P5j candidate ranking + close-out report

- Aggregate T0 (GDN rerun) + T1-T4 measurements into per-PP exclusive attribution table per § 2.5a span schema
- Compute `exclusive_us = inclusive_us - sum(children.inclusive_us)` for every span; verify `span.exclusive_us ≥ 0` invariant per § 2.5a; assert sum to root identity
- **Exclusive coverage gate** per § 7.1: `coverage_pct = Σ span.exclusive_us / root_span.inclusive_us ≥ 95%` per PP (NOT the prior naive "sum medians" gate, which double-counted nested spans per Codex review P1 #1)
- Identify per-PP top-3 bottleneck across all measured spans
- Rank P5i candidates (短 PP focus, +24-74% target) by ROI estimate
- Rank P5j candidates (长 PP focus, +110-128% target) by ROI estimate + Scope gate trigger (kernel rewrite = trigger Boss approval)
- **Target feasibility assessment**: honest evaluate "全 PP omlx+10% 在 P5i+P5j 内是否可达" given measured + estimated upper-bound caps
- Write `reports/p5h-attribution.md` (self-contained)
- Lock P5h spec § 7.2 final state (P5g `spec § 7.2` pattern — TBD → locked) + commit
- Commit: `chore(p5h-t5): close-out — all-PP exclusive attribution + P5i/P5j candidate ranking`

## § 4 Validation gates per task (Tn 共用)

每个 T1-T4 instrumentation task 完成时,严格按 CLAUDE.md "Rust 代码检测"规定执行:

```bash
cargo fmt                                                                          # Rust 自动格式化 (CLAUDE.md mandate)
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check                       # format check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings  # 0 Rust warnings (mlx-sys C++ warnings ok)
MLX_DIR=$HOME/.local/mlx cargo build --release                                     # release build PASS
```

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

T0 + T5 额外:

- T0 Phase D root cause: 4 hypotheses (H1-H4) resolved per § 2.5 decision tree, OR explicit unresolved-list documented in `reports/p5h-phase-d-root-cause.md`
- T0 exclusive span schema validator: assert `sum(child.inclusive) ≤ parent.inclusive` for all parent spans in test fixture; assert sum-to-root identity within 5% tolerance
- T0 GDN rerun: P5h-protocol GDN data emitted to `/tmp/p5h-gdn-rerun.json` with `parent_span = decoder_layer_N`
- T5 attribution: per § 7.1 exclusive coverage gate, P5i/P5j candidate ranking emitted with ROI estimate ranges + Scope gate trigger status

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
- prefill_chunk_size sweep (becomes P5j candidate if T2 GatedAttention long-PP O(S²) supports it)
- omlx PagedCache style port (out per memory `[feedback_design_philosophy]`)
- mlx::compile wrap (still blocked by 4 safe-wrapper API gaps from P5e T2)
- GatedAttention algorithmic redesign (P5h measures it; redesign 在 P5j 如果 measured 必要)

## § 7 Success Criteria

### 7.1 Exclusive attribution coverage gate (P1-fix from Codex review)

T5 must produce a per-PP exclusive attribution table built **only** from same-protocol P5h measurements (P5g existing data excluded; see § 2.2 #4 + T0 GDN rerun). Coverage computed as:

```
total_wall_us = root_span("client_to_response").inclusive_us
sum_exclusive = Σ span.exclusive_us  for span in all_emitted_spans
coverage_pct = sum_exclusive / total_wall_us
```

**Hard invariants** (per § 2.5a):
- `coverage_pct ≥ 95%` per PP (else report unaccounted residual + investigate before close-out)
- `span.exclusive_us ≥ 0` for every emitted span (negative = broken parent_span attribution, MUST fix)
- No bucket can be counted under two different parents (mutually exclusive tree)

This **replaces** prior naive "sum medians ≥ 95%" gate which double-counted nested spans (Codex review P1 #1).

### 7.2 P5h ship gate (T5 close-out gate)

1. **Exclusive attribution coverage** per § 7.1: ≥ 95% per PP, exclusive_us ≥ 0 for all spans
2. **Protocol-consistent data**: GDN (T0 rerun under P5h protocol) + HTTP/scheduler/admission (T1) + GatedAttention (T2) + MoE (T3) + lm_head/MLX state (T4) — all measured under same UMA hardening + exclusive span schema. P5g existing data remains as prior reference only, excluded from coverage gate.
3. **UMA hardening verified**: cross-repeat (cold/warm pair) measurement variance ≤ ±2% per PP per metric per layer (per § 2.4 protocol)
4. **Phase D root cause**: one of H1-H4 identified primary (mitigation proposed) OR explicit unresolved hypothesis list with proposed next investigation path (per § 2.5 decision tree)
5. **P5i + P5j candidate ranking**: each candidate has expected ROI range (number-anchored), Scope gate trigger status, 实施优先级
6. **Target feasibility assessment**: honest verdict on "全 PP omlx+10% achievable in P5i+P5j" — if not, partial-target proposal for Boss decision
7. **Reusable infra delivered**: GatedAttention 3-layer profile harness (per § 2.2 #5 code-backed taxonomy), MoE profile extension (per § 2.2 #6 + `Qwen35MoeConfig` runtime values), UMA hardening protocol, exclusive span schema infra — all usable in P5i+P5j+P5h+1
8. **Validation gates pass per task** (T0-T5 each independently green; per § 4)

P5h 整体 success = all 8 gates PASS, output (attribution report + P5i/P5j candidate list) is actionable for Boss to authorize P5i and/or P5j.

## § 8 References

- P5g close-out: `reports/p5g-final-results.md`
- P5g findings memory: `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5g_findings.md`
- P5g design spec: `docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md` (§ 4.1a / § 7.1a / § 7.2 post-T0 amendments)
- P5g implementation plan: `docs/superpowers/plans/2026-05-20-ironmlx-p5g-gated-delta-net.md`
- Boss memory: `[feedback_design_rigor]`, `[feedback_serial_perf_experiments]`, `[feedback_no_spec_from_competitors]`, `[feedback_performance_stability_priority]`, `[feedback_design_philosophy]`, `[feedback_task_breakdown_bounded]`, `[feedback_iron_bench_priority]`, `[feedback_no_unnecessary_docs]`
- Reusable infra from P5g: `ironmlx/tests/p5g_t0_gated_delta_profile.rs` (HTTP-path harness), `ironmlx/src/main.rs` (tracing→stderr fix), `/tmp/p5g-env.sh` pattern
