# P5i.c Phase 0 — Gap Decomposition: Close-out (γ-lite, measure-only)

**Status:** Phase 0 closed as **measure-only** per Codex round-2 γ-lite decision (`reports/p5i-c-phase-0-acceptance-codex-review.md` § 13).
**Date:** 2026-05-24.
**Branch:** `ironmlx-p5h+2-a-pp512-measurement`.
**Commit chain on this branch (P5i.c Phase 0):**
- `fb2d1c0` design spec (Codex-revised, Boss approved)
- `36cffb3` implementation plan (Codex-revised, Boss approved)
- `c3d92e1` T1 dual-mode Phase 0 capture harness
- `2535c34` T3 multi-repeat ranking pipeline (multi_repeat + pp_tps_envelope + phase0_compose + roi_ranking extensions + 17 pytests)
- this commit (T5 close-out)

**Sources:**
- Spec: `docs/superpowers/specs/2026-05-24-ironmlx-p5i-c-phase-0-gap-decomposition-design.md` (commit `fb2d1c0`).
- Plan: `docs/superpowers/plans/2026-05-24-ironmlx-p5i-c-phase-0-gap-decomposition.md` (commit `36cffb3`).
- Ranking snapshot: `docs/p5i-c-phase-0-ranking-snapshot.md` (this commit).
- T0 audit + T1/T2 bench log + acceptance review: `reports/p5i-c-phase-0-{audit.json,audit.md,bench-log.md,ranking-summary.md}` + `reports/p5i-c-{t0-5,phase-0-acceptance}-codex-review-questions.md` (all gitignored per `[feedback_no_reports_commit]`).
- Predecessor: `docs/p5i-a-close-out.md` (P5i.a Feasibility PASS); `docs/p5h+2-a-pp512-protocol.md` (RUNS=15 + monolithic preheat).

---

## § 1 Acceptance per spec § 7 — partial PASS

| # | Criterion | Verdict | Note |
|---|---|---|---|
| 1 | coverage_pct ≥ 0.95 per PP per repeat | ✓ PASS | aggregator runs all 6 probe cells without coverage error |
| 2 | `first_token_sampling_materialize_and_sample` ∉ top-5 | ✓ PASS | both PPs |
| 3 | verdict ≠ data_insufficient | ✓ PASS | both PPs (R1 triggered) |
| 4 | ironmlx production pp_tps envelope ≤ ±2% per PP | **✗ FAIL/DEFERRED** | PP=128 11.98% (r1+r4 trailing outliers); PP=512 11.88% (r1+r4 fast cluster vs r2+r3 slow) — **deferred to P5h+2.b per Codex round-2 § 13** |
| 5 | substep CI surfaced (not over-gated) | ✓ PASS | per-substep CI95 half-widths sub-percent both PPs |
| 6 | 4-category coverage status | ✓ PASS | scheduler measured / kv_cache proxy-only / attention measured / moe measured |
| 7 | tied-tier honesty | ✓ PASS | output is list-of-lists; tied tiers at lower ranks documented |
| 8 | vs-omlx baseline (T2 envelope, caveat allowed) | partial | omlx PP=128 env 3.47%; PP=512 env 5.07%; both > 2% — caveat per criterion #8 wording |

**Headline**: 7/8 acceptance criteria PASS; criterion #4 FAIL/DEFERRED with explicit P5h+2.b dependency for envelope re-validation.

---

## § 2 What landed — ranking + infrastructure

### Ranking (per `docs/p5i-c-phase-0-ranking-snapshot.md`)

Phase 1 default rule **R1 triggered** — cross-PP tier-1 candidate identical:

| PP | tier-1 candidate | probe-mode median share | CI95 half-width | P5h+1 reference |
|---|---|---|---|---|
| 128 | `gather_qmm_gate_up` | 23.38% | ±0.06% | 25.02% (P5h+1) — Δ -1.6pp |
| 512 | `gather_qmm_gate_up` | 22.84% | ±0.35% | 23.57% (P5h+1) — Δ -0.7pp |

Suggested Phase 1 candidate (per default rule R1): **`gather_qmm_gate_up`** for both PPs. Cross-PP single Phase 1 (e.g., P5i.c.1) attacks this candidate. **Boss decision required** at Phase 1 brainstorm.

Top-5 cross-PP stable post-Phase-0 (substep CI95 columns omitted for brevity; full in snapshot):

| Rank | Candidate | PP=128 share | PP=512 share | Category |
|---|---|---|---|---|
| 1 | gather_qmm_gate_up | 23.38% | 22.84% | MoE |
| 2 | gather_qmm_down | 12.34% | 11.69% | MoE |
| 3 | gda_step_1a_in_proj_qkvz | 10.19% | 14.94% | attention (GDN) |
| 4 | gda_step_8_norm_proj | 5.07% | 7.28% | attention (GDN) |
| 5 | gda_step_7_kernel_dispatch_and_materialize | 4.35% | 6.16% | attention (GDN) |

Matches P5h+1 ranking shape within ±2pp at top-3 (gather_qmm_{gate_up + down} combined 32-35% cross-PP; gda_step_1a 10-18%).

### Infrastructure shipped (commits c3d92e1 + 2535c34)

- **`ironmlx/tests/p5i_c_phase_0_capture.rs`** — env-var driven dual-mode (probe + production) capture harness. Configurable PP list / RUNS per PP / monolithic preheat seconds / repeat index. Per-cell output `/tmp/p5i-c-phase-0-r${R}-pp${PP}-${MODE}/{server.log,bench.csv,meta.json}`. Does NOT mutate validated `p5h_t5_attribution_capture.rs`.
- **`tools/p5h_aggregator/multi_repeat.py`** — per-substep bootstrap CI95 across ≥3 probe repeats; per-request SUM first (Fix A discipline for multi-emit spans like `gather_qmm_*` × 28 MoE layers); production-mode `root_us` extraction from flag-OFF server.log root spans.
- **`tools/p5i_c_pp_tps_envelope.py`** — per-PP pp_tps envelope `MAX(within bootstrap CI, between-sweep half-range)` + optional `--compare-repeat-csv` for vs-comparator delta with conservative CI bounds.
- **`tools/p5i_c_phase0_compose.py`** — final ranking JSON composer + markdown templates (summary / close-out / memory).
- **`tools/p5h_aggregator/roi_ranking.py`** extensions: `identify_tied_tiers` (greedy adjacent-overlap chain per spec § 8); `emit_category_coverage` (schema-level measured/proxy-only/unmeasured per Codex round-2 — NOT contingent on ranking presence); `emit_phase_1_default_rule` (R1/R2/R3 per spec § 9); `evaluate_dense_diagnostic_trigger` (trigger-A/B per spec § 10); `PHASE_0_CATEGORIES` span mapping.
- 17 new pytests covering aggregator math + tied-tier / coverage / rule / trigger logic.

---

## § 3 What FAILED — ironmlx production `pp_tps` envelope (§ 7 #4)

### Raw evidence

| PP | r1 median | r2 median | r3 median | r4 median (validation) | within CI95 max | between half-range | envelope |
|---|---|---|---|---|---|---|---|
| 128 | 953.69 (last 2 outliers 740/707) | 969.81 | 969.45 | 967.46 (last 2 outliers 881/640) | **11.98%** (r1 + r4 trailing) | 0.84% | **11.98% FAIL** |
| 512 | **1590.43** | 1278.85 | 1263.10 | **1393.57** | 0.89% | **11.88%** (r1+r4 fast cluster vs r2+r3 slow) | **11.88% FAIL** |

### Failure mechanism — measurement state inconsistency (NOT thermal noise)

- **PP=128**: r4 reproduced r1's last-2-rows trailing outliers (881, 640 vs typical ≥950). Pattern repeats across independent spawns → **harness-side / fan-cycling artifact, NOT single-point thermal noise** (Codex Q5 hypothesis confirmed).
- **PP=512**: r4 median 1394 falls in same family as r1 (1590), NOT same family as r2/r3 (1279/1263). Cross-spawn medians cluster into 2 groups ("fast" {r1, r4} ≈ 1490 / "slow" {r2, r3} ≈ 1271). **Protocol state-machine issue, not thermal**: every spawn's measurement state is consistent internally but bimodal across spawns.

### r4 predeclared validation (per Codex round-1 § 10)

Predeclared rules applied to {r2, r3, r4}:
1. r4 within-sweep CI95 ≤ 2% per PP → PP=128 FAIL (4.58%); PP=512 PASS (0.98%)
2. r4 median in same family as r2/r3 (≤5% deviation from avg) → PP=128 PASS (0.22%); PP=512 FAIL (9.65%)
3. {r2, r3, r4} envelope ≤ 2% per PP → PP=128 FAIL (4.58%); PP=512 FAIL (4.97%)

**ALL three rules failed at least once across PPs → r1 stays as evidence; escalate to P5h+2.b per Codex Q2.**

---

## § 4 vs-omlx delta — informational only (per Codex Q8)

Current noisy measurements are **directionally consistent with P5i.a, but exact vs-omlx delta is deferred until P5h+2.b**.

| PP | ironmlx median (mean of repeats) | omlx median | nominal delta | conservative half-width | P5i.a reference |
|---|---|---|---|---|---|
| 128 | 964.31 (r1-3 mean) | 1053.71 | -8.48% | ±15.46% | -7.31% (P5i.a T4 first) |
| 512 | 1377.46 (r1-3 mean) | 2199.53 | -37.37% | ±16.95% | -42.83% (P5i.a T4 first) |

**PP=128**: directionally consistent with P5i.a (-7 to -9pp range); precise delta NOT asserted until envelope passes.
**PP=512**: conservative range [-54%, -20%] still clearly indicates ironmlx behind omlx; precise delta deferred to P5h+2.b.

omlx PP=128 envelope 3.47% (slightly over 2% on within-sweep r2 CI); PP=512 envelope 5.07% — both treated as comparator caveat per Codex Q8.

---

## § 5 Phase 1 readiness — brainstorm allowed; implementation BLOCKED

Per Codex round-2 Q9:

- **Phase 1 brainstorm/design** may start in parallel with P5h+2.b. Tier-1 ranking is clean + cross-source consistent (P5h+1 within ±2pp) — brainstorming candidate `gather_qmm_gate_up` is well-founded.
- **Phase 1 implementation benchmark / acceptance gate / +10% target verification** MUST wait on P5h+2.b stable production envelope. Without ≤±2% envelope, ±2% optimization landing decisions are not statistically defensible.

---

## § 6 P5h+2.b — protocol fix hard bindings (per Codex round-2 Q10)

P5h+2.b spec MUST include:

1. **Hard acceptance**: PP=128 + PP=512 ironmlx production `pp_tps` envelope ≤ ±2% on ≥3 fresh-spawn repeats (replicate P5h+2.a binding but for both target PPs).
2. **Mechanism investigation**: explain or eliminate
   - PP=128 trailing-outlier pattern (last 2-3 rows of sweep dropping ~25-40%)
   - PP=512 bimodal cross-spawn medians (fast ~1490 / slow ~1270 clusters)
3. **Raw data preservation**: all raw per-run pp_tps + server log timestamps retained for forensic analysis.
4. **Predeclared outlier exclusion**: any rule excluding measurements MUST be written before looking at data (Codex round-1 pattern).
5. **Phase 0 backfill**: P5h+2.b outcome explicitly backfills `docs/p5i-c-phase-0-close-out.md` § 1 criterion #4 + `docs/p5i-c-phase-0-ranking-snapshot.md` envelope numbers.
6. **Per Codex round-2 Q11 instrumentation**: P5h+2.b must record per-run time series, run order within sweep, server lifecycle (spawn/kill timestamps), preheat placement (which spawn, which PP), and whether each PP shares the same server spawn or has its own. Likely protocol state-machine issue, not pure thermal noise — capture data accordingly.
7. **No PP-unification assumption**: PP=128 trailing outliers and PP=512 fast/slow cluster may be DIFFERENT mechanisms; P5h+2.b must not force them into one explanation.

---

## § 7 Dense diagnostic — SKIPPED (spec § 10 trigger evaluation)

`dense_diagnostic_triggered = false` per ranking output. Reason: MoE candidates dominate tier-1 at both PPs (`gather_qmm_gate_up` 23%); no non-MoE candidate at ≥15% magnitude threshold (trigger-A); no mixed MoE/non-MoE tier-1 (trigger-B). Per spec § 10 → Dense diagnostic skipped.

---

## § 8 Memory update

Extends MEMORY.md with new entry `project_p5i_c_phase_0_findings.md` documenting γ-lite close + tier-1 ranking + P5h+2.b dependency.

---

## § 9 Next phase

1. **P5h+2.b brainstorm** — Boss + Codex review questions doc → spec → plan → execute → close-out envelope at PASS.
2. **Phase 1 brainstorm (parallel allowed)** — design Phase 1 around `gather_qmm_gate_up` candidate; implementation/acceptance gated by P5h+2.b close.
3. After P5h+2.b PASS:
   - Re-run Phase 0 ironmlx production cells with fixed protocol; recompute envelope; backfill § 1 criterion #4
   - Recompute vs-omlx delta with confidence
   - Phase 1 implementation may proceed

---

## § 10 References

- Spec: `docs/superpowers/specs/2026-05-24-ironmlx-p5i-c-phase-0-gap-decomposition-design.md` (commit `fb2d1c0`)
- Plan: `docs/superpowers/plans/2026-05-24-ironmlx-p5i-c-phase-0-gap-decomposition.md` (commit `36cffb3`)
- Ranking snapshot: `docs/p5i-c-phase-0-ranking-snapshot.md` (this commit)
- Codex acceptance review (gitignored): `reports/p5i-c-phase-0-acceptance-codex-review.md` (§ 10/11/13 = decisions)
- T0/T0.5 audit + Codex review (gitignored): `reports/p5i-c-phase-0-audit.{json,md}` + `reports/p5i-c-t0-5-codex-review-questions.md`
- T1/T2/T3 bench log (gitignored): `reports/p5i-c-phase-0-bench-log.md` + per-cell artifacts under `/tmp/p5i-c-phase-0-*`
- Predecessor: `docs/p5i-a-close-out.md` (P5i.a Feasibility PASS), `docs/p5h+2-a-pp512-protocol.md` (RUNS=15 + monolithic preheat), `docs/p5h+1-close-out.md` (ranking infrastructure baseline)
- Memory: `[project-p5h-findings]` + `[project-p5h-2a-findings]` + new `[project-p5i-c-phase-0-findings]` (this commit)
