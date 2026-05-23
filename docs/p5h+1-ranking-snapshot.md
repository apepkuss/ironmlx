# P5h+1 Ranking Snapshot

**Status:** Committed concise summary. Full detail in `reports/p5h+1-ranking-snapshot.md` (gitignored).
**Date:** 2026-05-23
**Branch:** `ironmlx-p5h-perf`
**Implementation commits:** T1 `d57fbfb` + T2 `dfac330` + T3 `9f2c06c` + T1.5 `b14285a`

## TL;DR

**P5h+1 Close Gate 4/4 PASS.** Wrapper-dominance instrumentation gap closed: post-P5h+1 top-1 candidate is now `gather_qmm_gate_up` at 20-25% share across every PP (vs P5h T5 wrappers at 96-99%). 5 stable kernel-bound candidates surfaced across all PPs: `gather_qmm_gate_up`, `gather_qmm_down`, `gda_step_8_norm_proj`, `fused_sdpa` (long-PP), `gda_step_7_kernel_dispatch_and_materialize`. Top op-level candidate is `gda_step_1a_in_proj_qkvz` (10-18%). PP=128 reachable per current candidate set with kernel rewrites; PP=512-16384 require additional optimization sources beyond the current pool (gap +13% to +63% short).

## Close Gate result

| # | Condition | Status |
|---|---|---|
| 1 | Lane A wrapper share ≤ 50% | ✅ PASS (`first_token_sampling_materialize_and_sample` not in top-5) |
| 2 | Lane B wrapper share ≤ 50% | ✅ PASS (`gs_chunk_N` not in top-5) |
| 3 | coverage_pct ≥ 0.95 per PP | ✅ PASS (0.9796-0.9944) |
| 4 | Verdict ≠ data_insufficient ≥ 3 PPs | ✅ PASS (all 6 PPs) |

## Per-PP table (probe-mode attribution + production root reference)

⚠️ Probe-mode attribution data was generated with `--p5h-measurement-eval-probes` ON, which forces per-substep `mlx::transforms::eval` materialization. Probe-mode root inclusive time is **larger** than production root inclusive time (per spec § 6.5). Substep ROI shares are correct; production target feasibility uses `production_root_us` (prior P5h T5 baseline) as denominator, not probe_attribution_root_us.

| PP | observed lane | probe_attribution_root_us | production_root_us (P5h T5) | probe overhead | top-1 non-wrapper (probe share) | verdict |
|---|---|---|---|---|---|---|
| 128 | A | 260,886 | 137,414 | +89.8% | gather_qmm_gate_up (25.02%) | `yes_with_scope_gate` |
| 512 | A | 453,318 | 327,167 | +38.6% | gather_qmm_gate_up (23.57%) | `no_under_measured_cap` |
| 2048 | B ⚠️ spec § 1.2 says A | 1,365,489 | 1,104,580 | +23.6% | gather_qmm_gate_up (21.87%) | `no_under_measured_cap` |
| 4096 | B | 2,718,657 | 2,252,123 | +20.7% | gather_qmm_gate_up (22.16%) | `no_under_measured_cap` |
| 8192 | B | 5,752,183 | 4,900,643 | +17.4% | gather_qmm_gate_up (21.94%) | `no_under_measured_cap` |
| 16384 | B | 13,195,451 | 11,284,775 | +16.9% | gather_qmm_gate_up (20.10%) | `no_under_measured_cap` |

## P5i / P5j candidate list (cross-PP stable)

🔧 = `scope_gate_trigger=true` (kernel-rewrite candidate; Boss Scope gate approval required).

1. 🔧 **gather_qmm_gate_up** — 20-25% all PPs (MoE quantized gate+up matmul). Top P5i + P5j candidate.
2. 🔧 **gather_qmm_down** — 10-12% all PPs (MoE quantized down matmul). Same kernel family as gate_up.
3. **gda_step_1a_in_proj_qkvz** — 10-18% all PPs (GDN qkvz projection, op-level). Op-level optimization candidate (no Scope gate).
4. 🔧 **gda_step_8_norm_proj** — 5-8% all PPs (GDN final norm+projection).
5. 🔧 **fused_sdpa** — 6-16% (long-PP O(S²) growth; emerges as top-2 at PP=16384).
6. 🔧 **gda_step_7_kernel_dispatch_and_materialize** — 5-7% (GDN kernel dispatch; new sub-span from T1.5).

Combined gather_qmm_{gate_up + down} share = ~32-35% all PPs — dominant kernel family.

## Feasibility (production target gain coverage)

| PP | target gain | op_only_sum | with_kernel_sum | gap to target |
|---|---|---|---|---|
| 128 | +24% | 22.97% | 60.91% | +37% surplus (with kernel) |
| 512 | +74% | 22.22% | 61.10% | -13% short |
| 2048 | +110% | 21.46% | 61.31% | -49% short |
| 4096 | +115% | 20.43% | 61.66% | -53% short |
| 8192 | +124% | 19.16% | 62.21% | -62% short |
| 16384 | +126% | 18.05% | 62.66% | -63% short |

**Interpretation**: PP=128 reachable with current kernel-bound candidates. PP=512-16384 require additional optimization beyond the current candidate pool (likely candidates: higher realistic gain assumptions for first-pass quant kernel work; or new optimization opportunities not yet measured; or partial-target outcome with explicit Boss approval).

## Reproduction

See `reports/p5h+1-ranking-snapshot.md` § 9 for exact commands. Full per-PP top-5 + cross-PP observations + P5h+2 follow-up list in that document.
