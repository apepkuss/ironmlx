# P5h T2.4 — GatedAttention Layer 3 Ablation Binding (per T0b H4 verified)

**Status:** Committed binding decision per plan T2.4 + Codex T2 review (Option A: docs-only binding; no new ProfileMode / ablation harness). Replaces the empty-commit pattern in plan T2.4 Step 3 per `feedback_no_empty_commits` memory note (carrier file required for close-out narratives; Boss preference established 2026-05-22).

**Date:** 2026-05-22.
**Branch:** `ironmlx-p5h-perf`.
**Predecessor commits:** `1aa179a` (T2.2 — GatedAttention 7-substep instrumentation) → `b78e10b` (T2.3 — sweep harness) → T2.3 GPU sweep pass (97s wall, /tmp/p5h-t2.json verdict=pass) → this commit.

**Source docs:**
- Spec § 3 T2 conditional table (H4 verified row): `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md` lines 872-877.
- Plan T2.4 Step 2 "Apply binding per spec § 3 T2 conditional table": `docs/superpowers/plans/2026-05-20-ironmlx-p5h-all-pp-attribution.md` lines 4614-4633.
- T0b binding (memory): `project_p5h_t0b_findings.md` — H4 verified (small-PP kernel-dispatch variance) → kernel-bound Layer-3 ablation SKIPPED at all PPs.
- T0b close-out: `docs/p5h-t0b-close-out.md`.
- Codex T2 review questions doc (gitignored, working tree only): `reports/p5h-t2-codex-review-questions.md`.
- T2.3 sweep result JSON (machine-local, lost on reboot): `/tmp/p5h-t2.json`.

---

## 1. T0b verdict recap (binding inputs)

| Hypothesis | Verdict | Impact on T2 |
|---|---|---|
| H1 (cross-test thermal accumulation) | inconclusive within-cycle; ~17% cross-test drift | **Preheat protocol MANDATORY for T2 (already enforced by T2.3 harness via p5h_common::preheat_to_saturation).** |
| H2 (substitute self-cost > real) | rejected (substitutes 0%/33%/62% of real on T0b GDN substeps) | No structural block on ablation. But substitutes always cheaper → ablation reading approximates real cost from below, not loose upper bound. |
| H3 (cache state divergence) | unresolved/inconclusive (4/4 N/A both runs) | No T2 binding from H3. |
| H4 (small-PP kernel materialization variance) | verified (PP=2048 +22.19%, PP=4096 +9.51%; ±3% noise at larger PP) | **Layer-3 ablation INVALID for kernel-dispatch-touching steps. Per Boss + Codex Option A: SKIP at all PPs (no PP-bucket carve-out).** |

---

## 2. T2 7-substep classification (kernel-bound vs op-level)

Per spec § 3 T2 H4 verified row: "skip Layer 3 for steps 4-5 (kv_mask_update + fused_sdpa); Layer 3 OK for pure op-level steps (q_gate_k_v_proj, q_split_norm_reshape, mrope_apply, gate_sigmoid_mul, o_proj)."

| # | Substep | Class | Touches Metal kernel? | Layer 3 ablation eligibility |
|---|---|---|---|---|
| 1 | `q_gate_k_v_proj` | op-level | No (3 Linear quantized matmul: q_proj + k_proj + v_proj). Linear quantized matmul kernels are MLX-internal but instrumentation timer doesn't cross the kernel-dispatch boundary the way fused_sdpa does. | Eligible (deferred per § 4) |
| 2 | `q_split_norm_reshape` | op-level | No (reshape + split + RMSNorm + transpose). | Eligible (deferred per § 4) |
| 3 | `mrope_apply` | op-level | No (fused MetalKernel call internally per P3b1, but cost dominated by mrope's amortized internal work — does not hit the H4 small-PP dispatch-variance pattern at the GDN+Step 7d level). | Eligible (deferred per § 4) |
| 4 | `kv_mask_update` | **kernel-bound** | **Yes** — `cache.update_and_fetch_on` writes KV cache via Metal kernel; H4 verified the cache-update path is the exact kind of "dispatch + materialize" cost that varies with input value patterns. | **SKIPPED at all PPs.** |
| 5 | `fused_sdpa` | **kernel-bound** | **Yes** — `mlx::fast::scaled_dot_product_attention_on` is a fused Metal kernel; softmax / value matmul internals are not separately measurable on the production path. H4's exact mechanism (kernel materialization + dispatch-path variance) directly applies. | **SKIPPED at all PPs.** |
| 6 | `gate_sigmoid_mul` | op-level | No (sigmoid + elementwise multiply — no kernel-dispatch boundary). | Eligible (deferred per § 4) |
| 7 | `o_proj` | op-level | No (1 Linear quantized matmul; same kernel-dispatch reasoning as q_gate_k_v_proj). | Eligible (deferred per § 4) |

---

## 3. T2.3 sweep evidence (PP=128, Lane A medians)

From `/tmp/p5h-t2.json` cells:

| Rank | Substep | median_inclusive_us | Notes |
|---|---|---|---|
| 1 | mrope_apply | **5.29** | Dominant op-level cost — single mrope.apply call (fused MetalKernel) |
| 2 | kv_mask_update | 3.21 | **SKIP per H4 binding** |
| 3 | fused_sdpa | 2.54 | **SKIP per H4 binding** |
| 4 | q_split_norm_reshape | 2.06 | reshape + RMSNorm + transpose |
| 5 | q_gate_k_v_proj | 1.04 | 3 Linear quantized matmul |
| 6 | gate_sigmoid_mul | 0.75 | sigmoid + multiply |
| 7 | o_proj | 0.33 | 1 Linear quantized matmul |

Wrapper `attention_path` median = 41 us (averaged across 30 GDN + 10 full-attn layers per request).

**Observation**: all 7 op-level / kernel-bound substep medians at small PP are sub-6us per layer. Total GatedAttention substep sum per full-attn layer ≈ 15 us. Of that ~6 us is the 2 kernel-bound steps (40% of substep budget), and ~9 us is op-level. The op-level portion is small in absolute terms.

This is the key data point supporting the Option A deferral decision: even if op-level ablation produced tight upper bounds (per H2 reject reframing), the ROI is bounded by ~9 us per full-attn layer per token. Across 10 full-attn layers × prefill tokens, the candidate savings are modest relative to the dominant GDN / MoE budgets that T3 / T5 will measure.

---

## 4. Layer 3 binding decisions

### 4.1 Kernel-bound steps (4-5): SKIPPED at ALL PPs

`kv_mask_update` and `fused_sdpa` Layer 3 ablation is SKIPPED at every PP. Both touch Metal kernels via cache updates or fused SDPA dispatch — both exact instances of the H4 verified "kernel materialization / dispatch-path variance" pattern.

If a future P5h+1 or P5i task needs to estimate the savings from optimizing either kernel-bound step, use **real candidate implementation benchmarks** (build the candidate kernel variant, measure end-to-end pp_tps delta against the baseline). Do NOT add ablation variants.

This binding mirrors T0b's primary decision for GDN Step 7 `gated_delta_step` and Step 8 `out_proj` per `docs/p5h-t0b-close-out.md` § 3.1.

### 4.2 Op-level steps (1, 2, 3, 6, 7): ELIGIBLE but DEFERRED

`q_gate_k_v_proj`, `q_split_norm_reshape`, `mrope_apply`, `gate_sigmoid_mul`, `o_proj` are spec-eligible for Layer 3 ablation under H4 verified.

**Decision: defer Layer 3 ablation implementation to P5h+1 or after T5 cross-layer attribution.**

Rationale:
- T2.3 sweep data already provides per-substep inclusive_us — T5's P5i candidate ranking can proceed from this directly without ablation upper-bounds (per Codex Q-T2-2 answer).
- Op-level substep medians at small PP are all sub-6us; the dominant op-level cost (mrope_apply 5.29 us) is itself a fused MetalKernel internally, which may itself be sensitive to a milder version of the H4 pattern.
- H2 reject reframing argues substitute < real → ablation reading approximates real cost from below. This supports OPTION B in principle, but T2.3 data shows op-level totals are small enough that the additional measurement signal value is modest.
- Implementation cost of 5 new ProfileMode variants + harness + GPU sweep ≈ 1 day implementer + reviews. Disproportionate to the analytical value at this stage.

**Trigger condition for future op-level ablation work** (P5h+1 follow-up):
- T5 cross-layer attribution synthesis identifies a dominant GatedAttention op-level substep (e.g., if T5 shows that op-level mrope_apply contributes >10% of total per-PP wall time at any Lane A PP).
- OR: a real candidate implementation lands for a specific op-level step and we want to estimate baseline savings before merging.

**If triggered, recommended candidate substeps for op-level ablation** (per Codex Q-T2-1 fallback to Option C):
- Priority 1: `q_gate_k_v_proj` (3 Linear quantized matmul) — likely quantization-path hotspot per P5g findings on quantized Linear.
- Priority 2: `o_proj` (1 Linear quantized matmul) — same reasoning.
- Skip: `mrope_apply`, `q_split_norm_reshape`, `gate_sigmoid_mul` (low absolute cost; high implementation overhead per insight).

### 4.3 H1 binding: 5min preheat MANDATORY

Already enforced in T2.3 harness via `p5h_common::preheat_to_saturation`. Records `preheat_protocol` field in `/tmp/p5h-t2.json`.

### 4.4 H3 binding: none

H3 unresolved/inconclusive per T0b; no T2 binding derived.

---

## 5. Out of T2 scope / deferred follow-ups

- **Op-level Layer 3 ablation** (per § 4.2) — defer until T5 identifies a dominant op-level hotspot or a real candidate impl lands.
- **Lane B GatedAttention attribution** — T2.3 PP=4096 confirmed Lane B's deep substep attribution is suppressed (0 records for all 8 T2 spans, exempt from PASS criterion per spec § 3 T0a line 963). Lane B coverage is a P5h+1 task per the spec's existing deferral.
- **Per-layer-idx GatedAttention breakdown** — T2.3 sweep aggregates across 10 full-attn layers. If T5 needs per-layer attribution, the `[p5h-profile]` records already carry `layer_idx` — the T5 aggregator can group by it without further harness work.
- **mrope_apply internal MetalKernel timing** — if mrope_apply becomes a confirmed hotspot, P5h+1 should profile its internal kernel separately (similar to how T0b probed Step 7d for gated_delta_step).

---

## 6. T2 closure summary

| Sub-task | Status | Artifact |
|---|---|---|
| T2.1 — Read 7 substep boundaries | done (in-session reading; no commit) | source map vs spec § 2.2 #5 confirmed |
| T2.2 — Add 7-substep instrumentation | done (commit `1aa179a`) | `ironmlx/src/nn/gated_attention.rs` +273/-102 |
| T2.3 — Sweep harness + GPU sweep | done (commit `b78e10b` + sweep 97s wall verdict pass) | `ironmlx/tests/p5h_t2_gated_attention_sweep.rs` +971; `/tmp/p5h-t2.json` |
| T2.4 — Layer 3 conditional binding | this commit | `docs/p5h-t2-layer3-binding.md` |

T2 closed. Next: T3 (MoE 8-step) or T4 (lm_head + MLX state) per plan sequencing.
