# P5h+2.b — PP=128/512 Production Envelope Protocol Fix: Close-out (FAIL/DEFERRED)

**Status:** FAIL/DEFERRED per spec § 5.7. T4 acceptance under the predeclared protocol failed; root cause identified as production SchedulerActor Finished-phase ERROR path. Next phase must fix the production scheduler bug before P5h+2.b can be re-attempted.

**Date:** 2026-05-25.
**Branch:** `ironmlx-p5h+2-a-pp512-measurement`.
**Commit chain on this branch (P5h+2.b):**
- `aabf21f` design spec (Codex-revised, Boss approved)
- `ecc97cb` implementation plan (Codex-revised, Boss approved)
- this commit (T5F close-out — single commit attaching all infra)

**Sources:**
- Spec: `docs/superpowers/specs/2026-05-24-ironmlx-p5h+2-b-pp128-512-envelope-protocol-fix-design.md` (commit `aabf21f`)
- Plan: `docs/superpowers/plans/2026-05-24-ironmlx-p5h+2-b-pp128-512-envelope-protocol-fix.md` (commit `ecc97cb`)
- T0-T4 bench log + Codex review docs: `reports/p5h+2-b-*` (all gitignored per `[feedback-no-reports-commit]`)
- Predeclared exclusions (T4.1 locked): `reports/p5h+2-b-t4-predeclared-exclusions.md`
- Predecessor: `docs/p5i-c-phase-0-close-out.md` (Phase 0 γ-lite; § 7 #4 remains FAIL/DEFERRED)

---

## § 1 Acceptance per spec § 7.2 FAIL/DEFERRED — ALL satisfied

| # | Criterion | Status |
|---|---|---|
| 1 | T4 failure documented with raw envelope JSON + per-run preservation | ✓ |
| 2 | Status FAIL/DEFERRED (not PASS) | ✓ |
| 3 | Phase 0 § 7 #4 remains FAIL/DEFERRED (no PASS backfill) | ✓ |
| 4 | Next design questions explicit for new Boss + Codex round | ✓ § 4 below |
| 5 | Committed Rust/Python tooling passes the gates | ✓ |

---

## § 2 Failure summary

T4.3 acceptance sweep with predeclared protocol (lifecycle=`same_spawn_per_pp`, logging=`quiet_acceptance`, PP=128 RUNS=15, PP=512 RUNS=15, Rules B+C applied):

| PP | within-CI max | between half-range | final envelope | verdict |
|---|---|---|---|---|
| 128 | 4.17% | 1.18% | **4.17%** | FAIL (gate ≤ ±2%) |
| 512 | 7.19% | **10.71%** | **10.71%** | FAIL |

Per-repeat medians:
- PP=128: 896 / 918 / 910 pp_tps
- PP=512: 1589 / 1289 / 1323 pp_tps

T4R bounded retry SKIPPED — Codex round-3 (`reports/p5h+2-b-t4-3-codex-review.md` § 10) found root cause makes any α/β/γ T4R candidate ineffective; budget preserved for T5F evidence preservation.

---

## § 3 Root cause identified — production scheduler bug

Per Codex round-3 review of T4.3 server.log: every T4 cell (also T1/T2 cells) emits **1116 ERROR lines** of `[SchedulerActor] step error: step illegal in Finished phase: call prefill_admitted first`. Count `1116 = 1100 preheat + 1 warmup + 15 measured runs` = exactly per-request.

Code path:
- After `prefill_admitted()` request completes, rolling loop calls `sched.step()` at `ironmlx/src/core/server/scheduler_actor.rs:336`
- `max_tokens=1` → request finished → `Scheduler::step()` raises `step illegal in Finished phase` at `ironmlx/src/core/scheduler.rs:1285`
- Actor logs ERROR + calls `evict_all` at `scheduler_actor.rs:404`

**Mechanism**: per-request ERROR emission + state reset (evict_all) adds non-deterministic overhead → within-sweep pp_tps jitter + cross-spawn state-decay variance.

**Critical correction to T4 analysis**: "GPU/MLX state evolves DURING measurement" is overreach — T4 meta shows measurement is seconds-level, not minutes. Correct framing: "per-request scheduler ERROR/logging/evict path amplifies short-window pp_tps variance + state difference". Thermal/hardware NOT primary suspect; production scheduler ERROR path IS.

This is a PRODUCTION bug independent of `--p5h-profile` instrumentation — it fires in all modes (default_profile/quiet_acceptance/buffered_profile equally).

---

## § 4 Next design questions (per Codex round-3)

5 explicit questions for follow-up phase (e.g. P5h+2.c):

1. **Fix `max_tokens=1` / prefill-finished `step()` path** in `ironmlx/src/core/server/scheduler_actor.rs` and `ironmlx/src/core/scheduler.rs`. After `prefill_admitted` completes a `max_tokens=1` request, the actor's rolling loop must NOT call `sched.step()` again (request is Finished).
2. **Regression test**: prefill-only complete requests MUST NOT emit `step illegal in Finished phase` in `server.log`. Add to `ironmlx/tests/` covering the production HTTP path.
3. **New acceptance precondition** for measurement protocol work: `quiet_acceptance` mode requires scheduler/server ERROR count = 0 in `server.log` (modulo an explicit whitelist). Add a sanity check to `tools/p5h_2b_protocol_experiment.py` driver.
4. After scheduler fix → re-run PP=128 + PP=512 acceptance under same protocol (`same_spawn_per_pp` + `quiet_acceptance` + Rules A+B+C) → reassess whether ±2% gate is reachable.
5. **Defer** thermal / fan curve / cross-machine investigation until AFTER scheduler ERROR fix. T4 evidence was likely contaminated by the ERROR path; previously-attributed "GPU degradation" patterns may disappear once scheduler bug fixed.

---

## § 5 Phase 0 backfill (limited per spec § 5.7)

Phase 0 close-out + ranking snapshot get an ADDITIVE failed-attempt note ONLY. Criterion #4 stays FAIL/DEFERRED. Per Codex round-3 wording template:

> "P5h+2.b attempted resolution FAILED; Phase 0 §7 #4 remains FAIL/DEFERRED. T4 acceptance under the predeclared protocol failed with PP=128 envelope 4.17% and PP=512 envelope 10.71%. Additional log review found a per-request SchedulerActor Finished-phase step error present across T1/T2/T4 runs; the next phase must fix that production scheduler path and rerun acceptance before any PASS backfill."

Files modified for backfill:
- `docs/p5i-c-phase-0-close-out.md` — append P5h+2.b failed-attempt note to § 1 row 4 + § 6
- `docs/p5i-c-phase-0-ranking-snapshot.md` — append failed-attempt note to preamble; envelope section unchanged

---

## § 6 Reusable infrastructure committed (regardless of outcome per spec § 7.2 #5)

Even though P5h+2.b closes FAIL/DEFERRED, reusable infra remains committed for the follow-up scheduler-fix phase:

- `tools/p5h_2b_t0_outlier_source.py` + tests — offline outlier-source decomposition (4 pytests)
- `tools/p5h_2b_protocol_experiment.py` — T1/T2/T4 driver
- `tools/p5h_2b_thermal_overlay.py` + tests — powermetrics overlay (4 pytests including defensive fixes)
- `ironmlx/tests/p5i_c_phase_0_capture.rs` — extended with `P5I_C_SERVER_LIFECYCLE` (3 modes) + `P5I_C_PP_ORDER` + `P5I_C_LOGGING_MODE` (3 modes) + lifecycle Unix timestamps
- `iron-bench/src/{main,runner,report}.rs` — `--capture-run-timestamps` CLI flag (composable with `--capture-server-request-id`)

All 8 new pytests + existing 131 = 139 PASS.

---

## § 7 Wall summary

Cumulative wall: ~13 hr (T0 1.5 + T1-build 3 + T1-run 2 + T3 0.5 + T2 1.2 + T4.3 0.75 + T4 analysis 0.75 + T5F 1.3) — within 15 hr cap per Codex Q7 D.

---

## § 8 References

- T0 verdict: `reports/p5h+2-b-t0-outlier-source.md` (gitignored)
- T1-T4 bench log + Codex review chain: `reports/p5h+2-b-bench-log.md` + `reports/p5h+2-b-{t1-powermetrics,t4-3}-codex-review.md` (all gitignored)
- Predeclared exclusions: `reports/p5h+2-b-t4-predeclared-exclusions.md` (gitignored)
- Memory: `[project-p5h-2b-findings]` (new; outside repo)
- Predecessor: `docs/p5i-c-phase-0-close-out.md`, `docs/p5h+2-a-pp512-protocol.md`, `[project-p5i-c-phase-0-findings]`, `[project-p5h-2a-findings]`

---

## § 9 Re-attempt outcome (post-P5h+2.c, 2026-05-25)

**Status:** Re-attempt closed **FAIL/DEFERRED with substantial partial progress** per Codex round-4 (`reports/p5h+2-b-rerun-codex-review.md` § 6 Option α). Phase 0 § 7 #4 STILL FAIL/DEFERRED.

**Branch:** re-attempt + this update commit hosted on `ironmlx-p5h+2-c-scheduler-finished-fix` (P5h+2.c shipped at `9da48f5`; re-attempt sweep + close-out follow on same branch).

### § 9.1 Production-scale validation of P5h+2.c

`grep -c "step illegal in"` across all 6 cells (3 repeats × {PP=128, PP=512}) of the re-attempt sweep → **ZERO**. Driver `--allow-server-errors` default OFF did NOT abort. P5h+2.c fix CONFIRMED at production scale (beyond T2 integration smoke).

### § 9.2 Envelope verdict (Codex Rules B+C re-applied; identical protocol to T4.3)

| PP | pre-fix envelope (T4.3) | post-fix envelope (re-attempt) | Δ | gate ≤ ±2% |
|---|---|---|---|---|
| **128** | 4.17% (within-CI dominant) | **4.78%** (between-half-range dominant) | -0.6pp | FAIL |
| **512** | 10.71% (between-half-range dominant) | **5.06%** (within-CI dominant) | **+5.6pp improvement** | FAIL |

### § 9.3 What P5h+2.c DID prove

1. **PP=512 between-sweep half-range collapsed** 9.79% → 2.16% (76% reduction; cross-spawn medians 1391/1332/1335 vs pre-fix 1589/1278/1263).
2. **r1 PP=128 first-ever clean sweep**: raw `992 985 987 990 994 990 995 990 997 993 995 997 989 992 997` — 15 stable runs at ~993 plateau. Best PP=128 data in any P5h+2.x run to date.
3. **PP=512 cross-spawn medians tightened** — spawn-state inconsistency dramatically reduced.

### § 9.4 What P5h+2.c did NOT solve (residual variance)

1. **PP=128 trailing slowdown persists in r2 + r3**: r2 last 5 = `924 908 779 843` (Rule C fired); r3 last 5 = `744 812 877 881` (Rule C did NOT fire — only 1 run < 0.9 threshold). Gradual within-sweep degradation NOT scheduler-caused.
2. **PP=512 within-sweep fast-start-then-plateau** in ALL 3 repeats: ~1580 for first 2-3 runs → drop to ~1290-1390 plateau; r2 within-CI 5.06%. NOT scheduler-caused.
3. **PP=128 between-half-range INCREASED** 0.84% → 4.78% because r1 ran clean while r2/r3 degraded — exposes underlying degradation mechanism is sometimes suppressed (r1) and sometimes not (r2/r3).

### § 9.5 Narrative (Codex round-4 Q3, option a)

**P5h+2.c eliminated one variance source; exposed another residual mechanism that was previously masked.** Evidence supports this framing over "P5h+2.c side-effect made PP=128 worse":

- scheduler ERROR clear (counter == 0 verified)
- PP=512 envelope substantially improved (5.06% < 10.71%)
- r1 PP=128 plateau cleanest ever seen
- PP=128 between-half-range increase arises from r1's clean run exposing the spread between clean and degraded spawns — the degradation is residual, NOT side-effect-induced.

### § 9.6 Codex round-4 forward-binding

Outcome matched Codex round-3 pre-experiment prediction (b): "clear improvement, scheduler bug was real contributor, but NOT dominant for ±2% gate". Per Codex round-3 Q3 #4 + #5: now at the post-fix reassessment point with thermal/cross-machine deferral unblocked.

### § 9.7 β (gate relaxation) explicitly rejected

Per Codex round-4 Q5 #1: relaxing gate to ±3% does NOT pass either. PP=512 within-sweep CI 5.06% (r2 specifically) FAILS ±3% gate. Relaxation is moving the goalpost without resolving underlying variance and is not statistically defensible.

### § 9.8 Wall (re-attempt)

Re-attempt sweep (3 × {PP=128, PP=512} cells) + post-experiment analysis + close-out backfill: ~1.5 hr incremental. Cumulative on P5h+2.b workstream now ~14.5 hr.

---

## § 10 Next phase — P5h+2.d (thermal investigation) per Codex round-4 Option α

### § 10.1 Residual-mechanism hypothesis ranking (post-fix evidence)

| Rank | Hypothesis | Evidence post-fix |
|---|---|---|
| H1 | **Thermal / fan curve at sweep boundary** | r1 PP=128 clean = "cold-enough-at-start"; r2/r3 PP=128 degradation = "GPU heat-soak"; PP=512 fast-start = "first 2 runs before throttle". Reproducible + temporal pattern. |
| H2 | **MLX internal state (allocator / JIT / scheduler state-decay)** | First-spawn r1 sometimes clean; subsequent spawns inherit warm MLX internals which differ from fresh-spawn. Not separable from H1 without intervention. |
| H3 | **iron-bench client-side jitter** | Phase 0 T0 outlier-source analyzer ruled this out; re-confirmed by within-spawn pattern reproducibility. **Rejected.** |
| H4 | **Cross-machine variability** | Single-machine M5 Max baseline only; not testable solo. Out of scope. |

H1 is the clear top candidate (was H4-last before P5h+2.c; promoted by elimination of scheduler ERROR variance).

### § 10.2 P5h+2.d phase design constraints (per Codex round-4 Q2)

**Two-stage entry protocol — non-sudo first, sudo second:**

1. **Stage 1 (NO sudo): Thermal-protocol probe matrix.** Vary cooldown intervals between runs/repeats, vary PP order (fixed/reversed/alternating), vary long-cooldown placement before PP=128/512 small matrix. Goal: confirm "changing thermal state changes r2/r3 degradation + PP=512 fast-start drop".
2. **Stage 2 (sudo powermetrics): causal evidence backstop.** Requires Boss-approved settings rule for sudo. powermetrics sidecar as causal corroboration after Stage 1 establishes correlation, NOT as primary entry.

**Second-tier control: omlx baseline re-test.** After ironmlx thermal pattern is confirmed in Stage 1, re-run omlx PP=128/512 under same protocol. If both shift proportionally → system-level thermal; if only ironmlx shifts → ironmlx-specific (MLX state decay candidate). omlx control MUST NOT block P5h+2.d main line.

### § 10.3 Predeclared P5h+2.d success criteria (per Codex round-4 Q5 #2)

Must declare BEFORE running experiments:

- **Strong PASS**: with cooldown/thermal protocol modification, PP=128 + PP=512 envelope returns to ≤ ±2% on ≥3 fresh-spawn repeats — backfill Phase 0 § 7 #4 PASS, unblock Phase 1 implementation.
- **Weak PASS**: degradation pattern reproducibly ELIMINATED (e.g., extending cooldown to N seconds removes r2/r3 trailing slowdown) but envelope between 2-3% — would require Boss+Codex explicit decision on gate or further work.
- **FAIL / next-phase escalation**: degradation pattern persists regardless of protocol → mechanism is not thermal at sweep boundary; reassess H2 (MLX state decay) or H4 (cross-machine).

### § 10.4 Non-scheduler ERROR/WARN total tracking (per Codex round-4 Q5 #3)

P5h+2.d driver MUST extend `--allow-server-errors`-style check to also scan for non-scheduler ERROR/WARN classes and report counts per cell. Goal: avoid misattributing a NEW server-side anomaly as thermal-mechanism evidence.

### § 10.5 Phase 1 brainstorm — γ-lite parallel allowed (per Codex round-4 Q4)

Phase 1 BRAINSTORM/DESIGN may start in parallel with P5h+2.d. Tier-1 candidate `gather_qmm_gate_up` is stable + cross-source consistent (P5h+1 + Phase 0 + within ±2pp).

**Bindings reaffirmed:**
- Phase 1 implementation, performance acceptance, and +10% target verification **STILL BLOCKED** on envelope PASS.
- No premature acceptance via gate relaxation.

### § 10.6 Phase 0 backfill summary

`docs/p5i-c-phase-0-close-out.md` § 1 #4 + `docs/p5i-c-phase-0-ranking-snapshot.md` preamble both extended with re-attempt FAIL note + P5h+2.d dependency pointer (this commit).

### § 10.7 Re-attempt + close-out references

- Re-attempt raw data: `/tmp/p5h+2-b-rerun-t4-rerun-acceptance-r{1,2,3}-pp{128,512}/{bench.csv,server.log}` (host-side; gitignored)
- Re-attempt sweep log: `/tmp/p5h+2-b-rerun-t4-acceptance.log` (gitignored)
- Codex round-4 review doc: `reports/p5h+2-b-rerun-codex-review.md` (gitignored per `[feedback-no-reports-commit]`)
- Predecessor P5h+2.c scheduler fix: `docs/p5h+2-c-close-out.md` (PASS); `[project-p5h-2c-findings]`
