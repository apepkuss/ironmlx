# P5i.a — gather_qmm Feasibility + Short-PP Baseline: Close-out

**Status:** **Feasibility PASS** per spec § 3.2.
**Date:** 2026-05-23.
**Branch:** `ironmlx-p5i-a-gather-qmm-feasibility` (forked from `ironmlx-p5h-perf` @ `6579633`).
**T5 commit chain:** `99dfb93` (T0 baseline) → `9496d61` (T1 C1 sorted-MoE rank-3) → `5f9a269` (T2 gate+up fusion) → `7671581` (T3 feasibility memo) → (no T4 commit; Outcome C documented finding in bench log) → this commit (T5 close-out).

**Sources:**
- Spec: `docs/superpowers/specs/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility-design.md` (commit `6579633`).
- Plan: `docs/superpowers/plans/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility.md` (commit `85bb595`).
- T0 baseline doc: `docs/p5i-a-baseline.md` (commit `99dfb93`).
- T3 feasibility memo: `docs/p5i-a-gather-kernel-feasibility.md` (commit `7671581`).
- T4 canonical measurements: `/tmp/p5i-a-t4-summary.json` + `/tmp/p5i-a-t4-confirm-summary.json` (Outcome C; not committed).
- Predecessor P5h+1 close-out: `docs/p5h+1-close-out.md`.
- Memory: `project_p5h_findings.md` (P5h overall + P5h+1 + this P5i.a closure section).

---

## § 1 Close Gate 4-condition result

Per spec § 3.2 P5i.a first-phase gate.

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | External baseline established (T0 canonical denominator) | ✓ PASS | T0 commit `99dfb93`; ironmlx PP=128 919.68 / PP=512 1562.84 vs omlx PP=128 1060.90 / PP=512 2513.74; per `docs/p5i-a-baseline.md` |
| 2 | PP=128 outcome resolved | ✓ Feasibility PASS | Gap quantified at -7.10% / -7.51% canonical (T4 first + confirm); in-scope Level (b)/(c)/(d) candidates exhausted per spec § 3.2; remaining +17.1pp gap tied to P5i.c follow-up |
| 3 | PP=512 gap quantified | ✓ PASS (with noise-band caveat) | -41.33% / -44.32% canonical; within-sweep variance 20-25% per-run; 7-run median ±5-10% standard error; measurement protocol cannot reliably detect ±2% effects at PP=512 |
| 4 | P5i.b 立项/否决 verdict delivered | ✓ PASS | **暂缓 (defer)** per T3 commit `7671581` + `docs/p5i-a-gather-kernel-feasibility.md` |

**Spec § 7.2 update**: NOT applicable. Per spec § 4.6 acceptance criteria: spec § 7.2 update IF P5i.a delivers PP=128 +10% achievement (otherwise stays unchanged from P5h+1 post state). Status is Feasibility PASS (not Full PASS) → spec § 7.2 stays unchanged at **9/9 LOCKED PASS** per P5h+1 close-out § 1.

---

## § 2 Per-PP final state (T0 baseline → post-P5i.a canonical)

5-min thermal preheat per spec § 7.5 H1 binding. iron-bench `--prompt-len 128` / `--prompt-len 512` exact per spec § 7.6 (no chat-template adjustment at short PPs). Serial execution per spec § 7.4 (ironmlx + omlx never concurrent).

| PP | T0 baseline (ironmlx) | T4 canonical first | T4 canonical confirm | omlx baseline | Δ vs T0 (first) | Δ vs T0 (confirm) | Δ vs omlx (first) | Δ vs omlx (confirm) | passes +10% target |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 919.68 | 985.56 | 981.19 | 1060.90 | **+7.16%** | +6.69% | -7.10% | -7.51% | ✗ |
| 512 | 1562.84 | 1474.84 | 1399.56 | 2513.74 | -5.63% | -10.45% | -41.33% | -44.32% | ✗ |

**PP=128 reading**: T1+T2 cumulative landed gain over T0 = +6.9% (median of first/confirm). Gap to omlx+10% target = **-17.1pp to -17.5pp remaining** (closed +6.0pp via T1+T2 from initial -13.3pp).

**PP=512 reading**: canonical direction signal is **noise-bound** (per `[project_p5h_t0b_findings]` and T4 observed within-sweep variance). The PP=512 measurement protocol per current configuration cannot reliably resolve a ±2% optimization effect. Both T2-standalone "+3.55% under noisy preheat" and T4 canonical "-8.0%" readings fall inside the protocol noise band. PP=512 should not be characterized as either a confirmed regression or confirmed gain from T2 alone. The relevant honest statement: post-P5i.a PP=512 remains **-41.33% to -44.32%** below omlx with gap **-51.33pp to -54.32pp** to omlx+10% target, and protocol noise reduction is itself a P5h+2 prerequisite before re-attributing PP=512 changes ≤±5%.

---

## § 3 What landed (per-task summary)

### T0 baseline — `99dfb93`

Established canonical ironmlx (flag-OFF) vs omlx CLI denominator for PP=128/512 with 7-run median per spec § 7.3-7.6. T0 also delivered `tools/p5i_a_baseline_aggregate.py` (ruff-clean per § 7.7) and `docs/p5i-a-baseline.md` documenting the comparison methodology. **Gap at T0**: PP=128 -13.31%; PP=512 -37.83%.

### T1 simplifications — `9496d61` + 5 documented non-landings

Level (b) lazy-graph simplifications inventory: 6 candidates explored. Disposition:

| ID | Description | Disposition | Reason |
|---|---|---|---|
| C1 | sorted-MoE path rank-3 (drop one `expand_dims` + one `reshape`) | **LANDED** | Commit `9496d61`; bit-exact correctness; +3.16% PP=128 standalone over T0 |
| (absorbed) | 2 candidates folded into C1's rank-3 simplification | ABSORBED | Implementation merge — same code path; redundant |
| (rejected) | 3 candidates rejected a-priori | REJECTED | Documented rationale per T1 close-out (no ≥1% repeatable gain expected per per-candidate analysis) |

T1 ≥1% per-candidate gate per spec § 4.2 fully satisfied (C1 cleared; non-landings have documented rationale, not silent drops).

### T2 gate+up fusion — `5f9a269`

Level (c) MoE op-level fusion: `gate_proj` + `up_proj` two `gather_qmm` calls fused into single `gather_qmm` with concatenated weight tile.

- **Correctness**: bit-exact vs split path (production parity smoke `p5_qwen35_moe_smoke` argmax sentinel + pp_tps within ±2%, per spec § 7.1).
- **Architectural soundness**: cross-thread MLX stream binding fix via `OnceLock` lazy build (avoids static init at module load time; defers weight tile assembly to first inference call on the runtime stream).
- **PP=128 measured**: +6.9% cumulative over T0 (canonical 5-min preheat per spec § 7.5; T4 first + confirm median).
- **PP=512 measured**: noise-bound per § 2 above. T2 standalone reading "+3.55%" reported under earlier non-preheat protocol; canonical T4 reading "-8.0%" sits inside ±5-10% standard-error band. **Direction signal cannot be asserted in either polarity at PP=512 from current protocol**.

**Decision**: KEPT for PP=128 gain + architectural soundness (single-call gather_qmm reduces dispatch overhead by ~50% for the gate+up sites; combines with future Boss-approved P5i.b kernel rewrite if revisited).

### T3 self-quant gather Metal kernel feasibility — `7671581`

Level (a) feasibility-only design memo (no kernel implementation; per spec § 5 Scope gate, Level (a) work requires explicit Boss approval before P5i.b dispatch).

**Verdict: 暂缓 (defer P5i.b)** per `docs/p5i-a-gather-kernel-feasibility.md`. Four reasons:

1. Gather indirection cost is **structurally unknown**. self_qmm's +35% gain came on contiguous (M,K) × (N,K) matmul (every loader thread reads sequential memory). Gather mode adds `expert_indices[token]` lookup that fragments weight-load pattern across L2 cache lines — may halve or quarter achievable speedup.
2. PP=128 closeable other ways (T1+T2 already closed -13.3pp → -7.1pp; further wrapper/op-level work may close remaining -17.1pp without 2-4 week kernel rewrite).
3. PP=512 -35.62% gap (post-T2) may not be purely kernel-bound. T0b H4 was small-PP (≤128) hypothesis; PP=512 gap could equally come from launch-count overhead (8 gather_qmm calls/layer × 28 layers × 2 chunks), KV-cache layout transitions, or MLX upstream gather indirection.
4. Existing self_qmm is M1-Pro-only tile-tuned. M5 Max tile params not tuned per `[project_cross_device_tuning_deferred]`. Gather rewrite would need both expert-aware AND device-aware tile lookup before peak gain.

### T4 gda_step_1a tile tuning — Outcome C (no commit)

Level (d) op-level / kernel-bound investigation per P5h+1 ranking 10-18% all PPs. Two paths attempted; both eliminated:

- **Path (i): self_qmm tile-tune via `validate_tile`** — **BLOCKED**. `ironmlx/src/nn/self_qmm/lookup.rs` hard-panics on non-(32,64,32) tile parameters. Lookup mechanism is an empty shell on M5 Max; the validate_tile gate prevents any alternative tile from being attempted without first extending the lookup arity (out of P5i.a scope; surfaced as P5h+2 prerequisite).
- **Path (iii): op-level micro-optimization** — **SATURATED**. Per `[project_p5g_findings]` T1 fuse-revert evidence: `in_proj_qkvz` Linear 4-bit quant matmul is op-level saturated. No remaining single-op gain ≥1% per the existing operator graph.

T4 closed as **Outcome C documented negative finding** per spec § 4.5; no source change committed. T4 canonical 7-run sweeps (first + confirm) provide the post-P5i.a cumulative measurement for T5 close-out.

---

## § 4 Self-quant gather Metal kernel verdict

**暂缓 (defer P5i.b)** per T3 commit `7671581`.

**Upper-bound ROI estimate** (per T3 memo): ~12% root_inclusive at PP=128. This would close PP=128 to ~+5.4% over omlx, still short of +10% target. PP=512 -23% post-rewrite (still missing target by -33pp).

**Why now is not the right time**:

- PP=128 still closeable via P5i.c wrapper/scheduler discovery (not yet exhausted at the wrapper/scheduler level, only at the Level b/c/d MoE-substep level).
- PP=512 root attribution needs sharper measurement before 2-4 week kernel commit. P5i.c new-candidate discovery (scheduler overhead, KV cache layout, attention path) should attribute the residual PP=512 gap to specific spans before locking in gather kernel rewrite ROI.
- Gather indirection cost is uncertain enough that an upper-bound 12% estimate may collapse to 3-5% under realistic L2-fragmentation conditions; without a prototype, the project carries 2-4 week opportunity cost against a payoff that may not materialize.

**Reconsideration trigger**: after P5i.c sharpens PP=512 attribution, if gather_qmm_{gate_up + down} 32-35% root share remains the top blocker with no alternative non-kernel path, then re-evaluate P5i.b立项 with Boss Scope gate per spec § 5.

---

## § 5 P5i+ follow-up

### Next phase: **P5i.c new candidate discovery** (since T3 verdict was 暂缓, not 立项)

Per spec § 4.6 P5i.b alternative path: T3 verdict 暂缓 → P5i.c new-candidate discovery within P5i.a follow-up scope, NOT P5i.b.

P5i.c candidate inventory (initial; expanded during P5i.c brainstorming):

1. **PP=512 measurement protocol noise reduction** (precondition for any PP=512 attribution work). T4 found within-sweep variance 20-25% per-run; 7-run median has ±5-10% standard error; cannot detect ±2% effects. Investigate: longer run length / thermal envelope verification per request / per-cycle drift compensation.
2. **Scheduler overhead investigation** (off-MoE candidate). Per P5h+1 ranking, scheduler spans were not enumerated in the 6-candidate top-list; may hide gap-explaining cost at PP=512.
3. **KV cache layout investigation** (off-MoE candidate). Long-PP cache layout transitions at the PP=512 boundary may explain part of the PP=128 → PP=512 ironmlx-vs-omlx slope divergence.
4. **Attention path / fused_sdpa at short PP** (currently ranked top-5 at PP=16384 in P5h+1; verify short-PP share before deprioritizing).

After P5i.c sharper attribution → **reconsider P5i.b立项** with Boss Scope gate per spec § 5.

### P5h+2 prerequisites surfaced by P5i.a

- **`self_qmm` lookup arity extension** (T3+T4 blocker). Current `ironmlx/src/nn/self_qmm/lookup.rs` `validate_tile` hard-panics on non-(32,64,32) tile parameters; lookup mechanism is an empty shell on M5 Max. Any device-aware or quant-aware tile alternative requires lookup arity extension first.
- **PP=512 measurement protocol noise reduction** (T4 finding). Same item as P5i.c#1 — listed here to make P5h+2 dependency explicit.

### P5h+2 follow-up list (carried unchanged from P5h+1 § 7.2.1.5)

Unchanged from P5h+1 close-out § 8 (Boss decision deferred per `[feedback_task_breakdown_bounded]`):

- `validate_chunk_ancestry` cycle vulnerability (T2 review Important; 1-line `visited` set fix).
- `P5hChunkContextGuard.active: bool` dead field (T2 review Minor; remove or add `.disarm()`).
- `roi_ranking.py::LANE_A_WRAPPER_SPAN` stale `first_token_sampling` literal (T1 review note; cleanup).
- GA `kv_mask_update` outer probe duplicate-eval on cache=Some (T1 review Minor; comment).
- Emit cost reduction (T0a 95% gate via buffered/binary emit) — `[project_p5h_emit_cost_followup]`.
- T0b H4 same-mode control (Phase A × 2) — data confirmation per `[project_p5h_t0b_findings]`.
- T2/T3 op-level ablation (TRIGGERED by `gda_step_1a_in_proj_qkvz` 10-18%; pending; per T2.4/T3.4 binding doc).
- T4.2 mid-admit P5h ctx plumbing — low priority.
- Spec § 1.2 PP=2048 partition (already addressed in P5h+1 § 7.2.1.5 via note).

---

## § 6 Optional Dense diagnostic — skipped (Step 6.1a)

Per plan Step 6.1a: "Run only if MoE T0/final numbers are surprising or the residual gap is hard to attribute."

**Skip rationale**: MoE pattern observed in T0-T4 measurements matches the T0b H4 small-PP kernel-dispatch hypothesis + T2 op-level saturation evidence. The PP=128 closeable / PP=512 kernel-bound split is consistent with the `[project_p5h_t0b_findings]` H4 binding + P5h+1 ranking gather_qmm 32-35% share. Pattern is not surprising in the "unexpected" sense — only the residual attribution magnitude is uncertain (Dense diagnostic would tell us Dense pipeline gap vs MoE-specific gap, but that's a P5i.c-scoped question, NOT a P5i.a closure prerequisite).

Dense diagnostic deferred to P5i.c candidate inventory (§ 5 above) if scheduler/KV cache investigations don't sharpen PP=512 attribution sufficiently.

---

## § 7 Memory update

Extends `project_p5h_findings.md` with new section "P5i.a closure update (2026-05-23, same-day after P5h+1)". See memory file for cumulative state.

---

## § 8 References

- **Spec**: `docs/superpowers/specs/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility-design.md` (commit `6579633`)
- **Plan**: `docs/superpowers/plans/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility.md` (commit `85bb595`)
- **T0 baseline doc**: `docs/p5i-a-baseline.md` (commit `99dfb93`)
- **T3 feasibility memo**: `docs/p5i-a-gather-kernel-feasibility.md` (commit `7671581`)
- **T0 aggregator**: `tools/p5i_a_baseline_aggregate.py`
- **T4 canonical sweep summaries** (not committed; bench artifacts):
  - `/tmp/p5i-a-t4-summary.json` (first 7-run sweep)
  - `/tmp/p5i-a-t4-confirm-summary.json` (confirm 7-run sweep)
- **Commit chain on `ironmlx-p5i-a-gather-qmm-feasibility`**:
  - `99dfb93` T0 controlled iron-bench baseline (ironmlx flag-OFF vs omlx CLI; PP=128/512)
  - `9496d61` T1 C1 sorted-MoE path rank-3 (drop one expand_dims + one reshape)
  - `5f9a269` T2 gate+up fusion (single gather_qmm; bit-exact correctness; OnceLock lazy stream binding)
  - `7671581` T3 self-quant gather Metal kernel feasibility memo (Level a; design only; 暂缓 verdict)
  - (T4 Outcome C — no commit; canonical cumulative measurement documented in bench log + this close-out § 2)
  - this commit (T5 close-out)
- **Predecessor close-outs**: `docs/p5h+1-close-out.md` (P5h+1 attribution gap closure), `docs/p5h-t5-close-out.md` (P5h ship state)
- **Memory keys**:
  - `[project_p5h_findings]` — P5h overall + P5h+1 + P5i.a closure section (extended this commit)
  - `[project_p5h_t0b_findings]` — H1 thermal + H4 small-PP kernel-bound bindings
  - `[project_p5g_findings]` — T1 fuse-revert (op-level saturation evidence for T4 Path iii rejection)
  - `[project_cross_device_tuning_deferred]` — M5 Max tile tuning deferral (T3 reason #4)
  - `[project_p8a_stage9_findings]` — self_qmm dense +35% MLX baseline (T3 precedent)
  - `[feedback_no_empty_commits]` — close-out doc commit pattern (not `--allow-empty`)
  - `[feedback_task_breakdown_bounded]` — P5h+2 follow-up list bounded scope
  - `[feedback_honest_answers_no_sycophancy]` — PP=512 noise-bound framing per Boss preference (not asserting +3.55% nor -8.0% as truth)
  - `[feedback_serial_perf_experiments]` — spec § 7.4 serial execution binding
  - `[feedback_iron_bench_priority]` — T0/T4 use iron-bench (not custom perf scripts)
