# P5h — ironmlx 全 PP Prefill Gap Attribution (Design Spec)

| Field | Value |
|---|---|
| Phase | P5h (post-P5g) |
| Date | 2026-05-20 |
| Branch | `ironmlx-p5h-perf` (from `ironmlx-p5g-perf` HEAD `31c01db`) |
| Hardware | M5 Max 128 GB |
| Model | mlx-community/Qwen3.5-MoE-A3B-4bit |
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
| 4 | **GDN sub-step** | P5g T0 已测 11-step Layer 2 breakdown + per-step eval barriers (commit `52c39bd`); Step 7 kernel dispatch + Step 2c cache update 已 eagerly evaluated | **复用 P5g T0 measurement** (commit `52c39bd` 数据 in `/tmp/p5g-t0-phases.json` + `reports/p5g-final-results.md`). P5h T5 cross-layer synthesis 时 cite 这些数据,不重新 instrument。Kernel-level per-tile/per-shape timing 是 P5j (kernel rewrite) scope,不是 P5h attribution scope。 |
| 5 | **GatedAttention** | **P5g 完全未测**; 10/40 layers in MoE-A3B, full-attn O(S²) | New 3-layer profile (Layer 1 entry/exit, Layer 2 forward-step breakdown — SDPA dispatch + KV layout + rope + softmax + output proj, Layer 3 ablation per step) |
| 6 | **MoE expert dispatch + LinearMLP** | P5g 未测; 30 layers, 128 experts top-4, gather_qmm matmul | Per-layer expert dispatch overhead + gather_qmm matmul time + LinearMLP routing latency |
| 7 | **lm_head + MLX eval/cache state** | P5e/P5f shipped lm_head fix; MLX eval barrier costs 未细测 | lm_head time + MLX `eval()` barrier latency + KVCache + GatedDeltaCache state-update cost |

### 2.3 3-layer profile protocol per layer (复用 P5g pattern + extend)

每个 layer 重复 P5g 的 Layer 1 / Layer 2 / Layer 3 protocol,具体扩展:

- **Layer 1 (boundary-isolated)**: entry barrier + exit barrier + emit `[p5h-profile] layer=<name> ...elapsed_us=N`. 估算该 layer 总占比。
- **Layer 2 (per-step breakdown)**: 每个 sub-op 用 `mlx::transforms::eval(&[&intermediate])?` materialize + timer push。append step_breakdown CSV 到 Layer 1 log line。
- **Layer 3 (shape-preserving ablation)**: 每个 sub-step 提 substitute (e.g., GatedAttention SDPA substitute = identity passthrough), mode-gated entry barriers off for AblateX (per P5g § 4.1a barrier-free invariant)。

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

### T0 — Pre-flight + UMA hardening protocol + Phase D root cause investigation

- Branch verify + Cargo feature `p5h-profile` add (alongside `p5g-profile`, both can be on simultaneously)
- UMA hardening protocol implementation: cold/warm pair measurement + variance check + automatic retry
- Phase D root cause: 4 investigation sub-tasks (H1 randomized order / H2 substitute self-cost / H3 cache divergence / H4 kernel template variance)
- Output: hardening protocol spec, Phase D root-cause report (`reports/p5h-phase-d-root-cause.md`), reusable infra in test harness
- Commit: `feat(p5h-t0): UMA hardening + Phase D root cause`

### T1 — HTTP path + scheduler/admission profile

- Instrument: HTTP request entry / response exit, Scheduler admit queue dequeue, batch construction, slot allocation
- Layer 1: per-request boundary timing
- Layer 2: per-step breakdown (request parse / chat-template / admission wait / batch construct / forward dispatch / response serialize)
- Run sweep PP=128-16384, 5 runs warm + cold pair per UMA hardening
- Output: HTTP+scheduler attribution per PP, short-PP fixed-overhead identification
- Commit: `test(p5h-t1): HTTP + scheduler admission profile`

### T2 — GatedAttention layer instrumentation (new harness)

- Read `ironmlx/src/nn/gated_attention.rs` (suspected 10/40 full-attn layers)
- Add 3-edit instrumentation pattern (mirror P5g GDN):
  - Edit 1: entry barrier (input + cache materialize) gated on Layer1|Layer2
  - Edit 2: cache update sites use `as_deref_mut()` (preserve borrow)
  - Edit 3: tail refactor + exit barrier + log emission
- Layer 2 step breakdown: SDPA dispatch / KV layout / RoPE / softmax / output proj
- Layer 3 ablations: SDPA passthrough, RoPE skip, output proj zeros — applying Phase D root cause lessons
- Run sweep + aggregate
- Output: GatedAttention per-PP occupancy table, top-3 step ranking, long-PP O(S²) verification
- Commit: `test(p5h-t2): GatedAttention 3-layer profile`

### T3 — MoE expert dispatch + LinearMLP profile

- Instrument MoE layer (30/40 in A3B): expert routing (top-4 selection), gather_qmm dispatch, expert LinearMLP, output combine
- Layer 1 + Layer 2 breakdown
- Layer 3 ablations: routing identity (always top-4 first experts), gather_qmm zeros — applying Phase D root cause lessons
- Output: MoE per-PP attribution, expert dispatch overhead, gather_qmm dominance check
- Commit: `test(p5h-t3): MoE expert + LinearMLP profile`

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

- Aggregate T1-T4 measurements into per-PP attribution table
- Verify attribution sum ≥ 95% wall-time (else report unaccounted residual + investigate)
- Identify per-PP top-3 bottleneck across 7 layers
- Rank P5i candidates (短 PP focus, +24-74% target) by ROI estimate
- Rank P5j candidates (长 PP focus, +110-128% target) by ROI estimate + Scope gate trigger (kernel rewrite = trigger Boss approval)
- **Target feasibility assessment**: honest evaluate "全 PP omlx+10% 在 P5i+P5j 内是否可达" given measured + estimated upper-bound caps
- Write `reports/p5h-attribution.md` (self-contained)
- Lock P5h spec § 7.2 final state + commit
- Commit: `chore(p5h-t5): close-out — all-PP attribution + P5i/P5j candidate ranking`

## § 4 Validation gates per task (Tn 共用)

每个 T1-T4 instrumentation task 完成时:

- `cargo build --release`: PASS, 0 Rust warnings (mlx-sys C++ warnings ok)
- `cargo +nightly fmt --all -- --check`: clean
- `cargo +nightly clippy --all-features --workspace -- -D warnings`: 0 warnings
- `p5_qwen35_moe_smoke` (argmax=11): PASS
- `p5_qwen35_moe_batched` (B=2 row-eq): PASS
- `p5_qwen35_moe_http_smoke`: PASS
- Profile feature truly gated: default build produces zero `[p5h-profile]` log lines
- UMA hardening: cold/warm pair variance ≤ ±2% on the layer's sweep

T0 + T5 额外:
- T0 Phase D root cause: 4 hypothesis resolved (per H1-H4 decision tree) or explicit unresolved-list documented
- T5 attribution sum ≥ 95% wall-time accounted; P5i/P5j candidate ranking emitted

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

P5h ship gate (T5 close-out gate):

1. **Attribution coverage**: per-PP top-N components sum ≥ 95% wall-time accounted (no large unidentified residual)
2. **UMA hardening verified**: cross-repeat measurement variance ≤ ±2% per PP per metric per layer
3. **Phase D root cause**: resolved (one of H1-H4 identified primary, mitigation proposed) OR explicit unresolved hypothesis list with proposed next investigation path
4. **P5i + P5j candidate ranking**: each candidate has expected ROI range, Scope gate trigger status, 实施优先级
5. **Target feasibility assessment**: honest verdict on "全 PP omlx+10% achievable in P5i+P5j" — if not, partial-target proposal for Boss decision
6. **Reusable infra delivered**: GatedAttention 3-layer profile harness, MoE profile extension, UMA hardening protocol — all usable in P5i+P5j
7. **Validation gates pass per task** (T0-T5 each independently green)

P5h 整体 success = all 7 gates PASS, output (attribution report + P5i/P5j candidate list) is actionable for Boss to authorize P5i and/or P5j.

## § 8 References

- P5g close-out: `reports/p5g-final-results.md`
- P5g findings memory: `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5g_findings.md`
- P5g design spec: `docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md` (§ 4.1a / § 7.1a / § 7.2 post-T0 amendments)
- P5g implementation plan: `docs/superpowers/plans/2026-05-20-ironmlx-p5g-gated-delta-net.md`
- Boss memory: `[feedback_design_rigor]`, `[feedback_serial_perf_experiments]`, `[feedback_no_spec_from_competitors]`, `[feedback_performance_stability_priority]`, `[feedback_design_philosophy]`, `[feedback_task_breakdown_bounded]`, `[feedback_iron_bench_priority]`, `[feedback_no_unnecessary_docs]`
- Reusable infra from P5g: `ironmlx/tests/p5g_t0_gated_delta_profile.rs` (HTTP-path harness), `ironmlx/src/main.rs` (tracing→stderr fix), `/tmp/p5g-env.sh` pattern
