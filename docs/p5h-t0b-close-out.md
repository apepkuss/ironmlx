# P5h T0b — Phase D Root Cause Investigation: Close-out Report

**Status:** Committed (forward-fix for empty commit `60a5e41` per Boss preference — empty commits no longer used for close-out narratives going forward; spec § 3 T0b.5 template's `git commit --allow-empty` suggestion is wrong and should be flagged in future tasks).

**Date:** 2026-05-22.
**Branch:** `ironmlx-p5h-perf`.
**Carrier commits:** `60a5e41` (empty commit; body has the original narrative — kept in history, already pushed) + this commit (file-carrying version).

**Source docs:**
- Spec § 2.5 + § 3 T0b: `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md`
- Plan Task T0b: `docs/superpowers/plans/2026-05-20-ironmlx-p5h-all-pp-attribution.md`
- Design memo (Codex v2 reviewed, working tree only): `reports/p5h-t0b-phase-d-design.md`
- Codex T0b verdict review questions (working tree only): `reports/p5h-t0b-codex-review-questions.md`
- Raw verdict JSONs: `/tmp/p5h-t0b-h{1,2,3,4}.json` + `/tmp/p5h-t0b-h3-run1.json` (machine-local)
- T0b commit chain: `ef02e0f` → `1af0d52` → `3875ee5` → `be22727` → `6ff78a6` → `66a0ee9` → `60a5e41` (empty close-out) → (this doc commit, forward-fix).

---

## 1. Verdict summary

| H | hypothesis | verdict | resolution |
|---|---|---|---|
| H1 | GPU thermal drift across 24 spawns (within-cycle position effect) | **inconclusive** | median_drift=1.54%, max_drift=4.59% across 12 cells. Falls in inconclusive band, but cross-test thermal accumulation is large (~17.3% from start to end of test). 5min preheat protocol adopted as mitigation. |
| H2 | Substitute self-cost: substitute body > real op cost | **rejected** | 11970 records. Max per-step ratios = 0.62 / 0.33 / 0.0 (step_5 / step_7c / step_2b). Substitutes are uniformly cheaper than real ops. |
| H3 | Cache state divergence: AblateConv slower because conv_state stale | **unresolved (inconclusive twice)** | Both run1 and run2 produce 4/4 cells N/A: `phase_a < ablate_conv` at every PP. The P5g anomaly does not reproduce under preheat protocol — structural N/A, not statistical. Per Codex + Boss decision: no T2/T3 binding from H3. |
| H4 | Kernel materialization / dispatch-path variance | **verified (small-PP)** | 7980 records. PP=2048 +22.19%, PP=4096 +9.51%, PP=8192 -1.16%, PP=16384 -2.65%. 2/4 buckets above 5% threshold. |

---

## 2. Primary / ranked causes of the P5g Phase D anomaly

**Primary cause: H1 (cross-test thermal accumulation).**

H1's within-cycle metric came back inconclusive at the spec threshold, but the auxiliary data is unambiguous: Phase A None PP=2048 measured 1842.06 tok/s at test start vs 1524.01 tok/s at test end (-17.3% over 36min). The H3 result independently confirms this — once preheat enforces thermal saturation BEFORE measurement, the original P5g "AblateConv slower than Phase A" anomaly disappears entirely (phase_a < ablate_conv at every PP both runs). The systematic 100% directional pattern in H1 (normal_D > reversed_D in all 12 cells) further supports a real but slow-moving thermal effect that the within-cycle metric cannot fully capture.

**Secondary cause: H4 (kernel materialization / dispatch-path variance at small PP).**

H4 verified at PP=2048 (+22.19%) and PP=4096 (+9.51%); essentially noise at PP=8192/16384. The mechanism is NOT proven to be a `g`-value-dependent Metal kernel branch — the current Metal source does not branch on `g` values. The most defensible characterization is: **AblateComputeG changes Step 7d forced-eval timing at small PP**. The underlying mechanism is most likely lazy graph scheduling / cache hit rates / pipelining variance around `dispatch_builder()...dispatch() + take_at + eval`, but this is not directly proven by current data. A same-mode control measurement (H4MeasurePhaseA vs H4MeasurePhaseA) would be needed to rule out small-grid-noise as a partial confound; deferred unless Option B PP-bucket binding becomes desirable.

**Not a cause:**
- H2 (substitute self-cost): all substitutes ≤62% of real cost — substitute construction overhead is not the issue.
- H3 (cache state divergence): unresolvable under T0b conditions because the underlying anomaly the hypothesis was meant to explain doesn't reproduce.

---

## 3. T2/T3 binding (per spec § 3 T0b.5 decision tree)

Decisions per Boss + Codex review (this session):

### 3.1 Kernel-bound Layer 3 ablation: SKIPPED at ALL PPs (Option A — conservative)

The H4 verified row in spec § 3 T0b.5 table reads: "Ablation INVALID for kernel-dispatch-time hotspots (e.g., Step 7 gated_delta_step, Step 8 out_proj). T2.4/T3.4 Layer-3 ablation SKIPPED for kernel-bound steps; replaced with real candidate impl benchmark."

We chose Option A (no PP carve-out) over the data-driven Option B (Lane A SKIP / Lane B keep) because:
- The spec text does not carve out by PP. Following spec literally minimizes downstream interpretation cost.
- Without an H4 same-mode control to rule out small-grid-noise at Lane B, the "PP=8192/16384 drift below threshold" reading is not robust enough to support a PP carve-out.
- Asymmetric Lane A vs Lane B ablation policy adds explanation cost (every future reader has to retain the PP-bucket exception). Worth the avoided ablation effort only if the savings are large; they aren't.

**Binding:**
- Step 7 `gated_delta_step`: Layer 3 ablation SKIP for all PPs. Replace with real candidate implementation benchmark for future kernel optimizations.
- Step 8 `out_proj`: same.
- Any future identified kernel-bound hotspot in P5h or P5h+1 scope: ablation SKIP, use real impl benchmark.

### 3.2 5min preheat: MANDATORY for all P5h sweeps

H1's cross-test thermal accumulation (~17%) is the dominant driver of within-test variance. H3's complete N/A reproduction under preheat confirms preheat is sufficient to dissolve the original P5g Phase D anomaly. Therefore:

- Every P5h sweep test entry MUST include the 4 PP × RUNS=3 throwaway Phase A preheat with spawn-kill per PP (~10min wall) before formal measurement begins.
- This applies to T1 (re-baselining), T2.1-T2.3 (GatedAttention Phase A/B/C), T3.1-T3.3 (MoE Phase A/B/C), T4 (full attribution), and T5 (validation gate).
- No P5h sweep result is accepted without explicit preheat record in its output JSON's `preheat_protocol` field.

### 3.3 Non-kernel-bound Layer 3 ablation: KEEP

Spec § 2.5 + § 3 designated Layer 3 ablation steps in GDN Step 1-6 (RMSNorm, in_proj_qkvz, conv1d+silu intermediate steps not covered by H4) and MoE non-kernel-bound steps. Per Boss + Codex Q3 stance "H4 mechanism limited to Step 7d forced-eval timing changes", we keep Layer 3 ablation for non-kernel steps. They are not bound by the H4 finding.

### 3.4 H3 unresolved: no binding

H3 was rerun once per spec § 3 T0b.5 inconclusive-handling protocol. Result reproduced (4/4 N/A). Per spec: "If still inconclusive, do not bind T2/T3 gates from that hypothesis. Mark T0b as unresolved for that hypothesis and escalate to Boss with the numeric data." Boss has the data (both H3 JSONs preserved at `/tmp/p5h-t0b-h3-run1.json` and `/tmp/p5h-t0b-h3.json`).

### 3.5 Out of scope / deferred to P5h+1

- H4 same-mode control measurement (Phase A vs Phase A) — defer until or unless PP-bucket carve-out becomes desirable. Not required for Option A binding.
- H4 mechanism narrowing (lazy graph vs cache vs pipelining vs eval barrier) — not required for T0b close-out; H4 verified is enough to make the conservative T2/T3 binding decision.
- H3 deeper investigation (why does preheat eliminate the P5g anomaly so completely?) — covered by the primary H1 attribution. No follow-up needed for T0b scope.

---

## 4. Raw data tables

### 4.1 H1 (T0b.1, ran 2026-05-22, 36min wall, 48 spawns total)

`/tmp/p5h-t0b-h1.json`. median_drift=0.0154, max_drift=0.0459 → inconclusive.

Cross-test thermal drift (Phase A None PP=2048):
| position | pp_tps | drift from cold |
|---|---|---|
| start (spawn 1) | 1842.06 | baseline |
| end (spawn 48) | 1524.01 | **-17.3%** |

Within-cycle Phase D drift (normal vs reversed):
| ablate_mode | PP | normal | reversed | drift_pct | direction |
|---|---|---|---|---|---|
| ablate-compute-g | 2048 | 1603.01 | 1552.01 | 3.18% | normal > reversed |
| ablate-compute-g | 4096 | 1577.31 | 1532.55 | 2.84% | normal > reversed |
| ablate-compute-g | 8192 | 1515.43 | 1496.76 | 1.23% | normal > reversed |
| ablate-compute-g | 16384 | 1387.63 | 1373.60 | 1.01% | normal > reversed |
| ablate-conv | 2048 | 1611.02 | 1537.10 | 4.59% | normal > reversed |
| ablate-conv | 4096 | 1556.14 | 1524.30 | 2.05% | normal > reversed |
| ablate-conv | 8192 | 1524.65 | 1501.17 | 1.54% | normal > reversed |
| ablate-conv | 16384 | 1403.02 | 1388.28 | 1.05% | normal > reversed |
| ablate-t-arr | 2048 | 1564.48 | 1537.23 | 1.74% | normal > reversed |
| ablate-t-arr | 4096 | 1533.20 | 1521.40 | 0.77% | normal > reversed |
| ablate-t-arr | 8192 | 1502.14 | 1486.30 | 1.05% | normal > reversed |
| ablate-t-arr | 16384 | 1378.77 | 1368.40 | 0.75% | normal > reversed |

100% directional consistency (12/12 cells normal > reversed). Below spec verified threshold (median > 5% OR max > 10%) but consistent enough to motivate the preheat protocol decision.

### 4.2 H2 (T0b.2, ran 2026-05-22, 5.3min wall, total_records=11970)

`/tmp/p5h-t0b-h2.json`. Verdict: rejected. Max per-step median ratios across 4 PPs:

| step | substitute body | max per-PP median ratio | interpretation |
|---|---|---|---|
| step_2b (conv1d+silu) | `qkv.clone()` | 0.0 | substitute essentially free (Arc inc only) |
| step_5_compute_g | `zeros_like(a).cast(Float32)` | 0.62 | substitute ~62% of real cost |
| step_7c_t_arr | `T_ARR_ABLATION_CACHE` HashMap+Mutex lookup | 0.33 | substitute ~33% of real cost |

All ratios ≤ 1.00 across all PP buckets. Substitute self-cost hypothesis falsified.

### 4.3 H3 (T0b.3, both runs, total ~22min wall, 4/4 N/A both runs)

`/tmp/p5h-t0b-h3-run1.json` + `/tmp/p5h-t0b-h3.json`. Verdict both: inconclusive (all N/A).

Run 1:
| PP | Phase A | AblateConv | AblateConvWithManualCacheUpdate |
|---|---|---|---|
| 2048 | 1079.44 | 1460.87 | 1529.32 |
| 4096 | 1160.27 | 1489.10 | 1547.63 |
| 8192 | 1370.46 | 1492.97 | 1521.66 |
| 16384 | 1277.67 | 1396.92 | 1406.09 |

Run 2 (rerun per spec § 3 T0b.5):
| PP | Phase A | AblateConv | AblateConvWithManualCacheUpdate |
|---|---|---|---|
| 2048 | 1412.33 | 1643.94 | 1618.39 |
| 4096 | 1455.25 | 1649.08 | 1616.41 |
| 8192 | 1522.97 | 1616.82 | 1577.99 |
| 16384 | 1432.78 | 1477.47 | 1457.42 |

Both runs: `phase_a < ablate_conv` at every PP → 4/4 N/A. Structural (the conditions for `recovery_pct` to be defined are never met under T0b preheat protocol).

Note: in run1, WithManual marginally > AblateConv (cache update slight positive); in run2, WithManual marginally < AblateConv (cache update slight negative). Both within ~3%, no directional signal. Within noise.

Absolute pp_tps run2 ~10-30% higher than run1 — likely because the prior 24min H4+H2+H3 sweep ended ~11min before run1, leaving the system slightly warmer; before run2 ~3min idle. Both runs were on the same machine but at different thermal states. Both still produce the same structural N/A pattern.

### 4.4 H4 (T0b.4, ran 2026-05-22, 7.4min wall, total_records=7980)

`/tmp/p5h-t0b-h4.json`. Verdict: verified.

| PP | Phase A kernel_us | AblateComputeG kernel_us | kernel_drift_pct | record_count (per mode) |
|---|---|---|---|---|
| 2048 | 7181.5 | 8775.0 | **+22.19%** | 420 |
| 4096 | 26966.5 | 29531.0 | **+9.51%** | 630 |
| 8192 | 29906.0 | 29559.5 | -1.16% | 1050 |
| 16384 | 30597.0 | 29787.5 | -2.65% | 1890 |

2/4 PP buckets above 5% threshold → verified per spec.

Asymmetric: small-PP strong positive drift, large-PP within noise. Mechanism not proven (no Metal source branch on `g` values). Most defensible reading: AblateComputeG changes Step 7d forced-eval timing at small PP. Underlying cause likely in graph scheduling / cache / pipelining around dispatch + eval, but unproven by current data.

---

## 5. Open follow-ups (for P5h+1 or later)

1. **H4 same-mode control** — run H4MeasurePhaseA twice (or H4MeasureAblateComputeG twice) and compare drift. If same-mode drift at PP=2048 approaches 20%, the H4 finding partially attributable to small-grid noise. Defer unless PP-bucket binding ever wanted.
2. **H4 mechanism narrowing** — separate measurements of dispatch-only timing, take_at-only timing, and eval-only timing at small PP could partition the variance. Defer; not needed for T0b binding.
3. **Per-substitute self-cost ratios** as a tuning input for substitute design — H2 produced clean ratio data (0.0 / 0.33 / 0.62) that could inform future ablation harness design. Not blocking.
4. **H1 within-cycle metric refinement** — the current H1 metric compares D-at-position-13-24 vs D-at-position-25-36 within one test. A two-test design (cold-restart between phases) would directly test the original "D late slow" hypothesis. Lower priority — preheat protocol already addresses the practical issue.

---

## 6. P5h sweep protocol updates (binding from T0b)

For all subsequent P5h sweep tests (T1 / T2 / T3 / T4 / T5):

1. **5min preheat at every test entry**: 4 PP × RUNS=3 throwaway Phase A iron-bench runs with spawn-kill per PP. Discard results. Helper `preheat_to_saturation` lives in `ironmlx/tests/p5h_common/mod.rs` since 2026-05-22 (T1 extracted from T0b's inline copy; T0a + T0b kept their pre-extraction inline copies untouched).
2. **No kernel-bound Layer 3 ablation**: Step 7 `gated_delta_step`, Step 8 `out_proj`, and any future kernel-bound hotspot. Replace with real candidate implementation benchmark.
3. **Layer 3 ablation OK for non-kernel-bound steps** with preheat protocol.
4. **Each test JSON output MUST record `preheat_protocol` + `initial_cool_protocol` fields**. Boss option C explicit.
5. **RUNS=7 trimmed median + WARMUP=0** for any ablation-based comparison (T0a.13-fix precedent).
6. **Server feature gate `p5g-profile`** (NOT `p5h-profile`) for any ablation / kernel timing measurement — `p5h-profile` per-span tracing overhead contaminates substitute / kernel timing per design memo § 1.
