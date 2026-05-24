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
