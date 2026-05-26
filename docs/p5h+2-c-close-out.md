# P5h+2.c — Scheduler Finished-phase ERROR Fix: Close-out

**Status:** PASS — `step illegal in Finished phase` ERROR eliminated; bug-surface integration test (`test_scheduler_actor_max_tokens_1_no_finished_phase_error`) PASSES with counter == 0; scheduler.rs fail-fast semantics preserved (unit test asserts `step(Finished)` still returns Err); full Rust + Python gates clean.

**Date:** 2026-05-25.
**Branch:** `ironmlx-p5h+2-c-scheduler-finished-fix`.
**Implementation commit:** T3 commit containing scheduler_actor.rs + scheduler.rs tests + integration test + protocol driver guard + close-out doc (this commit).

**Sources:**
- Spec: `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-c-scheduler-finished-fix-design.md`
- Plan: `docs/superpowers/plans/2026-05-25-ironmlx-p5h+2-c-scheduler-finished-fix.md`
- Codex review: `reports/p5h-scheduler-bug-fix-codex-review.md` § 9 (gitignored)
- Predecessor: `docs/p5h+2-b-close-out.md` (P5h+2.b T5F where root cause was identified)

## § 1 Acceptance per spec § 7 — ALL PASS

| # | Criterion | Verdict |
|---|---|---|
| 1 | Bug surface eliminated (counter == 0 on 3× max_new_tokens=1 smoke) | ✓ PASS |
| 2 | Scheduler fail-fast preserved (step(Finished) still Err) | ✓ PASS |
| 3 | No regression (full cargo test PASSES; 3 mrope pre-existing failures documented in § notes) | ✓ PASS |
| 4 | Rust gate (fmt + nightly fmt --check + clippy + build) | ✓ CLEAN |
| 5 | Python gate (ruff + pytest) | ✓ CLEAN (139 PASS) |
| 6 | Driver guard active (--allow-server-errors opt-in) | ✓ |
| 7 | Close-out doc + memory + commit per `[feedback-*]` | ✓ |

## § 2 What landed

- `ironmlx/src/core/server/scheduler_actor.rs`: new `RollingControl` enum + `finalize_finished_batch_if_any` helper + `drive_empty_scheduler_handoff` helper (lifted from existing duplicated empty-batch handoff block); rolling-loop top hook + outer-loop top defensive hook; `p5h-profile` `STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT` counter.
- `ironmlx/src/core/scheduler.rs::tests`: 2 new unit tests lock fail-fast semantic (`test_max_new_tokens_1_transitions_to_finished_after_prefill` + `test_step_finished_phase_still_returns_err`).
- `ironmlx/tests/p5h_2c_scheduler_finished_smoke.rs`: new actor integration test (3× max_new_tokens=1 → counter == 0); gated under `#[cfg(feature = "p5h-profile")]`.
- `tools/p5h_2b_protocol_experiment.py`: `check_no_scheduler_errors` guard + `--allow-server-errors` CLI flag.

## § 3 Mechanism summary

P5h+2.b root cause (per `[project-p5h-2b-findings]`): `prefill_admitted` leaves `phase=Finished` for `max_tokens=1` workload (line 1247-1250); rolling loop's biased `tokio::select!` falls through to `RollingEvent::Step` when cmd_rx empty; `sched.step()` rejects via phase guard at line 1286; actor logs ERROR + `evict_all` per request → 1116 ERROR/cell in P5h+2.b T4.3.

P5h+2.c fix: pre-event handoff at rolling-loop top — when `phase == Finished` after previous prefill, call `drive_empty_scheduler_handoff`, which first runs `finalize_finished_batch_if_any` (evicts batch + clears event_txs) and then handles queued admits / try_recv / break / return. The biased select never sees a `Finished` state.

Scheduler core semantics preserved: `step_inner` phase guard untouched at scheduler.rs:1286; `step(Phase::Finished)` still returns Err; unit test locks this.

## § 4 P5h+2.b re-attempt readiness

P5h+2.b T4 acceptance sweep can now be re-run with this fix. T0-T3 infrastructure (protocol_experiment.py with --allow-server-errors default off → strict precondition, lifecycle harness, multi_repeat aggregator, pp_tps_envelope tool) is reusable as-is. Expected outcome: 0 `step illegal in <phase>` ERROR per cell + envelope re-measurement.

## § 5 Memory update

Extends MEMORY.md with `project-p5h-2c-findings` entry pointing at `project_p5h_2c_findings.md`. Phase 0 + P5h+2.b memory entries remain unchanged (this fix doesn't backfill P5h+2.b/Phase 0; that's a separate re-run task).

## § 6 Notes

### Env var deviation in integration test

Integration test `ironmlx/tests/p5h_2c_scheduler_finished_smoke.rs` uses `QWEN35_MODEL=...Qwen3.5-4B-MLX-4bit...` per existing `b1_p2_*` test convention, **not** the plan template's `IRONMLX_MOE_MODEL_DIR=...35B-A3B...`.

Reason: `Qwen35MoeModel` (the MoE 35B-A3B model) requires a different loader path (`Qwen35MoeModel::from_loader`) than `Qwen35Model::from_loader`. The integration test's actor-side bug surface (rolling-loop phase control) is independent of model size and MoE routing; the dense 4B model is fully sufficient to exercise the `Phase::Finished` transition via `max_new_tokens=1` prefill. Using the 4B model also reduces test wall time (~0.9s vs. ~30s+ for 35B).

The plan's original template specified 35B-A3B under the assumption that `Qwen35MoeModel` fits `Qwen35Model::from_loader`. Verified at T2: it does not. Decision: use dense 4B, document deviation here.

### Pre-existing NEXT_SPAN_ID race in `core::p5h::tests`

Two tests (`try_with_span_lane_b_*`) share a static `AtomicU64 NEXT_SPAN_ID`; parallel `cargo test` execution causes non-deterministic ordering. This race was **NOT introduced by P5h+2.c** — reproduced by stashing all P5h+2.c changes and running baseline HEAD `c4bfa92`. Out-of-scope for P5h+2.c (separate future task if it becomes flaky).

Note: these tests do not appear in the `cargo test --release -p ironmlx` run summary as failures in this gate run; the race only manifests when test parallelism hits a specific timing window.

### Pre-existing mrope test failures

Three tests in `ironmlx/tests/p3b1_mrope.rs` (`rotary_plus_sdpa_matches_fixture`, `apply_matches_python_fixture`, `cos_sin_matches_python_fixture`) fail on HEAD `c4bfa92` (baseline without P5h+2.c changes). **NOT introduced by P5h+2.c.** The first `cargo test --release -p ironmlx` summary line reports 293 PASS / 0 FAIL for the lib crate; the mrope binary is a separate test binary with a different result line.

## § 7 References

- Spec: `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-c-scheduler-finished-fix-design.md`
- Plan: `docs/superpowers/plans/2026-05-25-ironmlx-p5h+2-c-scheduler-finished-fix.md`
- Codex review: `reports/p5h-scheduler-bug-fix-codex-review.md` § 9
- Predecessor: `docs/p5h+2-b-close-out.md`
- Memory: `[project-p5h-2c-findings]` (new), `[project-p5h-2b-findings]` (root cause source), `[project-p5i-c-phase-0-findings]` (Phase 0 still pending P5h+2.b re-run)
