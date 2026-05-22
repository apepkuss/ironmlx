# P5h T3.4 — MoE Layer 3 Ablation Binding (per T0b H4 verified)

**Status:** Committed binding decision per plan T3.4 + Boss + Codex Option A precedent from T2.4 (`f2153b6`). Replaces the empty-commit pattern in plan T3 close-out per `feedback_no_empty_commits` memory note (carrier file required for close-out narratives; Boss preference established 2026-05-22).

**Date:** 2026-05-22.
**Branch:** `ironmlx-p5h-perf`.
**Predecessor commits:** `67e131a` (T3.2 — MoE 8-substep instrumentation in sparse_moe.rs) → `2ba47de` (T3.3 — MoE sweep harness) → T3.3 GPU sweep pass (94.9s wall, /tmp/p5h-t3.json verdict=pass) → this commit.

**Source docs:**
- Spec § 3 T3 conditional table (H4 verified row): `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` lines 895-899.
- Plan T3.1-T3.4: `docs/superpowers/plans/2026-05-20-ironmlx-p5h-all-pp-attribution.md` line 4641 ("Mirrors T2 with the 8-step MoE breakdown").
- T0b binding (memory): `project_p5h_t0b_findings.md` — H4 verified → kernel-bound Layer-3 ablation SKIPPED at all PPs.
- T2.4 close-out: `docs/p5h-t2-layer3-binding.md` (the structural template this doc follows).
- T3.3 sweep result JSON (machine-local, lost on reboot): `/tmp/p5h-t3.json`.

---

## 1. T0b verdict recap (binding inputs — identical to T2.4)

| Hypothesis | Verdict | Impact on T3 |
|---|---|---|
| H1 (cross-test thermal accumulation) | inconclusive within-cycle; ~17% cross-test drift | **Preheat protocol MANDATORY for T3 (already enforced by T3.3 harness via p5h_common::preheat_to_saturation).** |
| H2 (substitute self-cost > real) | rejected (substitutes 0%/33%/62% of real on T0b GDN substeps) | Substitutes always cheaper → ablation reading approximates real cost from below. |
| H3 (cache state divergence) | unresolved/inconclusive | No T3 binding from H3. |
| H4 (small-PP kernel materialization variance) | verified (PP=2048 +22.19%, PP=4096 +9.51%; ±3% noise at larger PP) | **Layer-3 ablation INVALID for kernel-dispatch-touching steps. Per Boss + Codex Option A (T0b): SKIP at all PPs (no PP-bucket carve-out).** |

---

## 2. T3 8-substep classification (kernel-bound vs op-level)

Per spec § 3 T3 H4 verified row (lines 895-899) **verbatim**: "Layer 3 invalid for `gather_qmm_*` steps + `routing_sort_pack`/`routing_unsort_weighted_reduce` (all kernel-dispatch dependent); skip Layer 3 for steps 2-3 + 5-6; OK for steps 1, 4, 7-8."

| # | Substep | Class | Touches Metal kernel? | Layer 3 ablation eligibility |
|---|---|---|---|---|
| 1 | `router_logits_softmax_topk` | op-level | No (gating Linear + softmax + topK selection — softmax/topK are CPU-ops; Linear is quantized matmul which is MLX-internal but does not hit the H4 dispatch-variance pattern at the gating scale). | Eligible (deferred per § 4) |
| 2 | `routing_sort_pack` | **kernel-bound** | **Yes** — argsort + gather + reshape involve Metal kernel dispatch + materialization. T3.3 data confirms this is also the only PP-scaling substep (3.4→7.9 us as PP grows). | **SKIPPED at all PPs.** |
| 3 | `gather_qmm_gate_up` | **kernel-bound** | **Yes** — fused gather + quantized matmul (2 projections: gate + up) via Metal kernel dispatch. Same H4 risk pattern as gated_delta_step. | **SKIPPED at all PPs.** |
| 4 | `swiglu_activation` | op-level | No (sigmoid + elementwise multiplications — no kernel-dispatch boundary). | Eligible (deferred per § 4) |
| 5 | `gather_qmm_down` | **kernel-bound** | **Yes** — fused gather + quantized matmul (down projection). Same H4 risk as step 3. | **SKIPPED at all PPs.** |
| 6 | `routing_unsort_weighted_reduce` | **kernel-bound** | **Yes** — unsort (argsort inverse), gather, weighted reduce — Metal kernel dispatch. | **SKIPPED at all PPs.** |
| 7 | `shared_expert` | op-level | No (LinearMLP forward: gate + up + sigmoid + multiply + down — same Linear quantized matmul as in_proj/o_proj; same kernel-dispatch reasoning as T2's q_gate_k_v_proj — does not hit the H4 dispatch-variance pattern at this scale). | Eligible (deferred per § 4) |
| 8 | `moe_output_sum` | op-level | No (residual sum + reshape — pure elementwise). | Eligible (deferred per § 4) |

---

## 3. T3.3 sweep evidence (PP=128, Lane A medians)

From `/tmp/p5h-t3.json` cells (280 records per cell = 40 decoder layers × RUNS=7):

| Rank | Substep | median_inclusive_us | Notes |
|---|---|---|---|
| 1 | **routing_sort_pack** | **3.42** | **kernel-bound — SKIP per H4. ONLY PP-scaling substep (3.4→7.9 us at PP=1024).** |
| 2 | shared_expert | 2.17 | op-level — biggest non-routing op-level cost; constant per layer |
| 3 | router_logits_softmax_topk | 1.83 | op-level — Linear (hidden→num_experts=256) + softmax + topK |
| 4 | routing_unsort_weighted_reduce | 1.67 | **kernel-bound — SKIP per H4** |
| 5 | gather_qmm_gate_up | 1.25 | **kernel-bound — SKIP per H4** |
| 6 | swiglu_activation | 0.75 | op-level — sigmoid + elementwise |
| 7 | gather_qmm_down | 0.63 | **kernel-bound — SKIP per H4** |
| 8 | moe_output_sum | 0.50 | op-level — residual sum |

`mlp_path` wrapper = 28.06 us (averaged across all 40 decoder layers per request).

**Key observations:**

- **gather_qmm is NOT dominant at small PP.** Spec § 3 T3 anticipated `gather_qmm_*` to be the biggest MoE hotspot ("gather_qmm dominance check"). Actual data shows gather_qmm gate_up (1.25 us) + down (0.63 us) = 1.88 us combined, which is LESS than routing_sort_pack alone at PP=1024 (7.90 us) and LESS than shared_expert (2.17 us). Root cause: at PP ≤ 1024 with `--max-tokens 1`, `packed_tokens = PP × 8 / 256 ≈ 4-32 per expert` — tiny matmul dimensions; the quantized matmul kernel is heavily amortized at small batch.
- **gather_qmm dominance would emerge at larger PP / batch sizes** where `packed_tokens` per expert grows substantially. T3.3 does NOT measure this regime (Lane A capped at PP ≤ 2036). P5h+1 should consider a higher-PP or batched sweep if gather_qmm characterization is desired.
- **routing_sort_pack is the only PP-scaling kernel-bound step** — 3.42 → 5.69 → 7.90 us across PP=128/512/1024. Linear scaling matches expected sort + gather complexity.
- **Total op-level substep cost per layer = 1.83 + 0.75 + 2.17 + 0.50 = 5.25 us** (sum of steps 1, 4, 7, 8). Even SMALLER than T2 GatedAttention's 9 us op-level total (per `docs/p5h-t2-layer3-binding.md` § 3).
- **Total kernel-bound substep cost per layer = 3.42 + 1.25 + 0.63 + 1.67 = 6.97 us** (sum of steps 2, 3, 5, 6). Routed-kernel work dominates op-level by ~33%.
- **Wrapper / substep gap** = mlp_path (28.06) - sum(substeps) (12.21) = ~15.85 us. Most of this is `[p5h+1_emit_cost_reduction]` follow-up territory — emit overhead at ~1.8 us per span × 9 spans = 16 us, dominates the gap. (See memory `project_p5h_emit_cost_followup`.)

This is the second op-level cost surface (after T2) showing modest absolute substep costs at small PP. The data argues the same conclusion as T2.4 Option A: **op-level ablation ROI is bounded by ~5 us per layer per token — disproportionate to the implementation cost** (~5 new ProfileMode variants + harness + GPU sweep ≈ 1 day implementer + reviews).

---

## 4. Layer 3 binding decisions

### 4.1 Kernel-bound steps (2, 3, 5, 6): SKIPPED at ALL PPs

`routing_sort_pack`, `gather_qmm_gate_up`, `gather_qmm_down`, `routing_unsort_weighted_reduce` Layer 3 ablation is SKIPPED at every PP per Boss + Codex Option A T0b decision (no PP-bucket carve-out).

If a future P5h+1 or P5i task needs to estimate the savings from optimizing any kernel-bound step, use **real candidate implementation benchmarks**:
- For `gather_qmm_*`: implement an alternative quantized matmul kernel (e.g., different tile size, fused gather+matmul, different quantization scheme); measure end-to-end pp_tps delta against the current baseline.
- For `routing_sort_pack` / `routing_unsort_weighted_reduce`: implement alternative sort/scatter approach (e.g., different argsort variant, different gather strategy); measure pp_tps delta.

Do NOT add ablation variants in `sparse_moe.rs` — H4 will contaminate the readings.

This binding mirrors T0b's primary decision for GDN Step 7 `gated_delta_step` and Step 8 `out_proj` per `docs/p5h-t0b-close-out.md` § 3.1 + T2.4's kernel-bound `kv_mask_update` + `fused_sdpa` skip per `docs/p5h-t2-layer3-binding.md` § 4.1.

### 4.2 Op-level steps (1, 4, 7, 8): ELIGIBLE but DEFERRED

`router_logits_softmax_topk`, `swiglu_activation`, `shared_expert`, `moe_output_sum` are spec-eligible for Layer 3 ablation under H4 verified.

**Decision: defer Layer 3 ablation implementation to P5h+1 or post-T5 cross-layer attribution.**

Rationale:
- T3.3 sweep data already provides per-substep `inclusive_us` — T5's P5i candidate ranking can proceed from this directly without ablation upper-bounds (per Codex Q-T2-2 answer applied uniformly across T2/T3).
- Op-level substep medians at small PP are all sub-3us (largest = shared_expert at 2.17 us); op-level total ≈ 5.25 us per layer.
- Implementation cost of 4 new ProfileMode variants + harness + GPU sweep ≈ 1 day implementer + reviews — disproportionate to the analytical value at this stage.
- Op-level total budget (5.25 us × 40 layers = 210 us per request) is smaller than the single-layer routing_sort_pack at PP=1024 (7.90 us, scaling) work, which T0b H4 says cannot be ablated anyway.

**Trigger condition for future op-level ablation work** (P5h+1 follow-up):
- T5 cross-layer attribution synthesis identifies a dominant op-level MoE substep (e.g., if T5 shows op-level shared_expert contributes >10% of total per-PP wall time at any Lane A PP).
- OR a real candidate implementation lands for a specific op-level step and we want to estimate baseline savings before merging.

**If triggered, recommended candidate substeps** (per T3.3 cost ranking):
- Priority 1: `shared_expert` (2.17 us — biggest op-level cost; LinearMLP same shape as the FFN MLP layers that P5g flagged as quantized matmul hotspots).
- Priority 2: `router_logits_softmax_topk` (1.83 us — Linear `hidden→num_experts=256` + softmax; the gating Linear is a candidate for fused or quantized optimization).
- Skip: `swiglu_activation` (0.75 us — pure elementwise; no kernel to optimize), `moe_output_sum` (0.50 us — residual sum; no separately optimizable kernel).

### 4.3 H1 binding: 5min preheat MANDATORY

Already enforced in T3.3 harness via `p5h_common::preheat_to_saturation`. Records `preheat_protocol` field in `/tmp/p5h-t3.json`.

### 4.4 H3 binding: none

H3 unresolved/inconclusive per T0b; no T3 binding derived.

---

## 5. Out of T3 scope / deferred follow-ups

- **Op-level Layer 3 ablation** (per § 4.2) — defer until T5 identifies a dominant op-level hotspot or a real candidate impl lands.
- **Lane B MoE attribution** — T3.3 PP=4096 confirmed Lane B's deep substep attribution is suppressed (0 records for all 9 T3 spans, exempt from PASS criterion per spec § 3 T0a line 963). Lane B coverage is a P5h+1 task per the spec's existing deferral.
- **High-PP / batched gather_qmm characterization** — T3.3 measures only Lane A PP ≤ 1024 with B=1. At these dimensions gather_qmm is amortized small and does NOT show its expected dominance. A separate higher-PP or batched sweep would reveal the gather_qmm scaling characteristic, but that's outside T3's Lane A focus.
- **routing_sort_pack non-sorted path** — T3.2 instrumented the non-sorted branch with zero-op closure (inclusive_us ≈ 0). For 35B model with `--max-tokens 1` and prefill PP ≥ 64, the sorted-path records dominate; the non-sorted-path zero-op records contribute to record_count but don't bias the median. If T5 needs separate sorted vs non-sorted breakdown, add a `bs_k` field to the emission schema (P5h+1 work).
- **Per-layer-idx MoE breakdown** — T3.3 sweep aggregates across 40 decoder layers. If T5 needs per-layer attribution, the `[p5h-profile]` records already carry `layer_idx` — the T5 aggregator can group by it without further harness work.

---

## 6. T3 closure summary

| Sub-task | Status | Artifact |
|---|---|---|
| T3.1 — Read 8 substep boundaries | done (in-session reading; no commit) | source map vs spec § 3 T3 confirmed |
| T3.2 — Add 8-substep instrumentation | done (commit `67e131a`) | `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` +621/-262 |
| T3.3 — Sweep harness + GPU sweep | done (commit `2ba47de` + sweep 94.9s wall verdict pass) | `ironmlx/tests/p5h_t3_moe_sweep.rs` +1012; `/tmp/p5h-t3.json` |
| T3.4 — Layer 3 conditional binding | this commit | `docs/p5h-t3-layer3-binding.md` |

T3 closed. Next: T4 (lm_head + tokenization + MLX state profile) or T5 (cross-layer attribution synthesis + P5i/P5j candidate ranking) per plan sequencing.
