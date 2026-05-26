# P5i.c Phase 0 Ranking Snapshot

**Status:** measure-only close (γ-lite) per Codex round-2 — ranking PASS, production envelope FAIL/DEFERRED, Phase 1 acceptance still blocked. See `docs/p5i-c-phase-0-close-out.md` for the close-out framing + dependency chain. **2026-05-25 update:** P5h+2.b re-attempt (post-P5h+2.c scheduler fix) closed FAIL/DEFERRED with substantial improvement (counter==0 confirmed; PP=512 between-half-range 9.79%→2.16%); envelope still > ±2% gate. Next phase was **P5h+2.d thermal investigation** per Codex round-4 Option α. **2026-05-25 P5h+2.d update:** Closed **Mechanism-only** (Mechanism gate `strong_yes`; BEST cooldown=120s; PP=512 envelope PASS 0.91% / PP=128 envelope FAIL 4.71% — within-CI residual NOT thermal at sweep-boundary scale). Phase 0 § 7 #4 production envelope **NOT** backfilled. **2026-05-26 update:** `small-PP acceptance threshold` support is prepared for PP=128, but P5h+2.e is still running under Claude Code; no close-out/backfill is recorded here yet. Ranking section below unchanged (Phase 0 probe ranking not affected by production envelope status).

**Date:** 2026-05-24

**Spec ref:** `docs/superpowers/specs/2026-05-24-ironmlx-p5i-c-phase-0-gap-decomposition-design.md` (commit `fb2d1c0`).

**Audit ref:** `reports/p5i-c-phase-0-audit.json` (gitignored; T0 output).

**Branch HEAD measured:** `c3d92e1` (T1 harness + P5i.a T1+T2 + P5h+2.a closure; code-equivalent to plan baseline `6a593c4`).

## Phase 1 default rule

**Triggered:** `R1`

**Suggested candidates:** ['gather_qmm_gate_up']

**Rationale:** cross-PP tier-1 identical: gather_qmm_gate_up

## 4-category coverage status

| Category | Status |
|---|---|
| scheduler | `measured` |
| kv_cache | `proxy-only` |
| attention | `measured` |
| moe | `measured` |

## Per-PP top-N ranking with CI95 + tier

### PP=128

| Tier | Candidates (probe-share + CI95 half-width) |
|---|---|
| tier-1 | gather_qmm_gate_up (23.38%, ±0.06%) |
| tier-2 | gather_qmm_down (12.34%, ±0.08%) |
| tier-3 | gda_step_1a_in_proj_qkvz (10.19%, ±0.04%) |
| tier-4 | gda_step_8_norm_proj (5.07%, ±0.07%) |
| tier-5 | shared_expert (4.76%, ±0.02%) |
| tier-6 | gda_step_7_kernel_dispatch_and_materialize (4.35%, ±0.03%) |
| tier-7 | routing_unsort_weighted_reduce (3.73%, ±0.04%) |
| tier-8 | router_logits_softmax_topk (3.56%, ±0.02%) |
| tier-9 | swiglu_activation (3.21%, ±0.02%) |
| tier-10 | routing_sort_pack (3.01%, ±0.02%) |
| tier-11 | q_gate_k_v_proj (2.80%, ±0.04%) |
| tier-12 | moe_output_sum (2.49%, ±0.03%) |
| tier-13 | gda_step_5_compute_g (2.30%, ±0.05%), gda_step_1b_in_proj_ba (2.24%, ±0.01%) |
| tier-14 | gda_step_4_qk_rmsnorm (2.18%, ±0.02%), gda_step_2b_conv1d_silu (2.15%, ±0.02%) |
| tier-15 | gda_step_2a_prepend_conv_state (2.05%, ±0.02%) |
| tier-16 | gda_step_6_sigmoid_beta (1.87%, ±0.03%) |
| tier-17 | o_proj (1.57%, ±0.03%) |

### PP=512

| Tier | Candidates (probe-share + CI95 half-width) |
|---|---|
| tier-1 | gather_qmm_gate_up (22.84%, ±0.35%) |
| tier-2 | gda_step_1a_in_proj_qkvz (14.94%, ±0.22%) |
| tier-3 | gather_qmm_down (11.69%, ±0.11%) |
| tier-4 | gda_step_8_norm_proj (7.28%, ±0.02%) |
| tier-5 | gda_step_7_kernel_dispatch_and_materialize (6.16%, ±0.02%) |
| tier-6 | shared_expert (5.09%, ±0.05%) |
| tier-7 | q_gate_k_v_proj (3.76%, ±0.06%) |
| tier-8 | routing_unsort_weighted_reduce (3.24%, ±0.02%) |
| tier-9 | router_logits_softmax_topk (2.72%, ±0.03%) |
| tier-10 | routing_sort_pack (2.41%, ±0.03%) |
| tier-11 | swiglu_activation (2.19%, ±0.00%) |
| tier-12 | o_proj (2.03%, ±0.05%) |
| tier-13 | gda_step_2b_conv1d_silu (1.71%, ±0.04%) |
| tier-14 | moe_output_sum (1.60%, ±0.05%) |
| tier-15 | gda_step_4_qk_rmsnorm (1.54%, ±0.02%) |
| tier-16 | gda_step_1b_in_proj_ba (1.48%, ±0.02%) |
| tier-17 | fused_sdpa (1.39%, ±0.02%), gda_step_2a_prepend_conv_state (1.36%, ±0.05%), gda_step_5_compute_g (1.32%, ±0.03%) |
| tier-18 | gda_step_6_sigmoid_beta (1.08%, ±0.02%) |

## Dense diagnostic

**Skipped:** tier-1 dominated by MoE candidates per current ranking

## vs-omlx delta (P5h+2.a scope ii baseline)

| PP | ironmlx_median | omlx_median | delta_pct | ironmlx_envelope | omlx_envelope |
|---|---|---|---|---|---|
| 128 | 964.31 | 1053.71 | -8.48% | ±11.98% | ±3.47% |
| 512 | 1377.46 | 2199.53 | -37.37% | ±11.88% | ±5.07% |
