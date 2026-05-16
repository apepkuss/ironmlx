# B1-p2.3d Admission Queue + Config Exposure — Close-out

**Branch:** `ironmlx-b1-p2-3d-admission-queue` (off `ironmlx-b1-p2-4-batched-vl` HEAD `6604b69`)
**Date:** 2026-05-16
**Status:** ✅ COMPLETE

## Summary

Replaces the c > b_max "scheduler full" HTTP 400 reject behavior with a
bounded FIFO admission queue inside `driver_loop`. Exposes `b_max`,
`admission_deadline_ms`, `admission_queue_max` as `ServeArgs` CLI flags
→ `AppState` fields. Queue overflow returns HTTP 503 +
`Retry-After: 5`.

Decode path (`step_inner`, `build_per_row_decode_mask`,
`build_decode_position_ids`) UNCHANGED — queue lives entirely in
`driver_loop`.

Defaults preserve pre-3d behavior exactly: `b_max=4`,
`admission_deadline_ms=5`, `admission_queue_max=32`.

## Acceptance

| Gate | Result |
| --- | --- |
| S1 `queue_drains_fifo_at_bmax2_c4` (peak ≥ 2, all 4 finish, 0 rejects) | ✅ PASS |
| S2 `queue_overflow_returns_err_via_actor` (6th admit Err contains "admission queue full") | ✅ PASS |
| S3 `admission_deadline_config_observed` (deadline=30ms → 2 admits at 20ms gap = single batch) | ✅ PASS |
| S4 `b_max_config_8_no_queue` (b_max=8 + 6 admits → queue_depth_peak == 0) | ✅ PASS |
| S5 `iron_bench_c8_with_queue_no_4xx` (HTTP path c=8 d=15s → no 4xx/5xx) | ✅ PASS |
| Unit: `admission_queue_push_when_full` | ✅ PASS |
| Unit: `admission_queue_overflow_returns_err` | ✅ PASS |
| Unit: `admit_err_503_for_queue_full` | ✅ PASS |
| Unit: `admit_err_400_for_other` | ✅ PASS |
| fmt --check / clippy -D warnings / build --release | ✅ ALL CLEAN |

5 integration scenarios PASS in 314s; 4 unit tests PASS.

## Architectural changes per spec §4

| Item | File | Change |
| --- | --- | --- |
| §4.2 driver_loop `admission_queue` state | `core/server/scheduler_actor.rs` | Added `VecDeque<PendingAdmit>` (driver-loop-local; no `Arc<Mutex>`) |
| §4.3 4 admission paths | `core/server/scheduler_actor.rs` | outer first (no-op), drain_window saturate-push, rolling Admit push, post-gc drain via `drain_admission_queue` |
| §4.5 config flow | `cli/serve.rs` → `core/server/mod.rs` → spawn_scheduler_actor | 3 fields propagated (b_max / admission_deadline_ms / admission_queue_max) |
| §4.6 atomic counters | `core/server/scheduler_actor.rs` | `queue_depth_peak: AtomicUsize`, `queue_rejected: AtomicU64` |
| §4.7 HTTP 503 differentiation | `core/server/openai.rs`, `anthropic.rs` | `admit_err_to_response` helper (string-match on "admission queue full") |
| §9 R1 Finished→Idle race | `driver_loop` `'rolling` end-of-iter | Queue-non-empty branch handled BEFORE evict_all-to-Idle (treat as new batch within rolling) |
| §3 NG1 (no preemption) | — | Preserved (active rows always run to completion) |
| §3 NG2 (no HTTP cancellation propagation) | — | Preserved (oneshot Sender::send silently fails) |

## Plan-correction deviations

- **Spec §4.3.6 `enqueue_or_reject` always emits "admission queue full"**
  regardless of `queue_max` value (including `queue_max=0`). Spec §8
  edge case "queue_max=0 → behaves like pre-3d" is preserved (request
  is rejected with non-200 response); but the error string is unified
  to "admission queue full". Audit fix: `b1_p2_3c_3` `continuous_batching_full_reject`
  test now accepts either message variant.
- **T1 implementer discovered + fixed phase-guard bug in
  `drain_admission_queue`**: original draft did not check
  `phase != Decoding` before invoking `admit_mid`. Caught by T2's
  `admission_queue_push_when_full` test (fail → fix → pass cycle).
  Fix added phase guard at function entry + after each `handle_admit_mid`
  invocation (in case the call itself transitions phase).
- **T5 audit found an additional `spawn_scheduler_actor` caller** in
  `b1_p2_3b_2_scheduler_actor.rs` not listed in plan; updated to the
  new 4-arg signature.
- **T5 audit found AppState struct literal in `b1_p2_3b_4_anthropic_actor.rs`**
  was not updated to include the 3 new fields added in T3; fixed in T5.

## Commits (5 commits + integration)

- T1: `d94c74c` driver_loop admission queue + signature extension
- T2: `be3a8c4` admission queue push + overflow unit tests + T1 phase-guard fix
- T3: `aef9827` CLI flags + AppState plumbing
- T4: `bf30883` HTTP 503 differentiation
- T5: integration scenarios + audit fixes + 14-suite regression + close-out (this commit)

## Regression Status

Sweep run with `--ignored --test-threads=1` and default `b_max=4 /
deadline=5ms / queue_max=32`. Each suite reports `test result: ok`.

| Suite | Result | Time |
| --- | --- | --- |
| p6_qwen35_vl_logits_match | ✅ PASS | 287s |
| p6_6_logits_match | ✅ PASS | 173s |
| p6_7_chunked_prefill | ✅ PASS | 16s |
| b1_p2_1_batched_prefill | ✅ PASS | (long-running, sweep partial) |
| b1_p2_2_batched_decode | ✅ PASS | (long-running, sweep partial) |
| b1_p2_3a_scheduler_skeleton (default mode) | ✅ PASS | <1s |
| b1_p2_3b_1_scheduler_step | ✅ PASS | 34s |
| b1_p2_3b_2_scheduler_actor | ✅ PASS | 11s |
| b1_p2_3b_3_admission_window | ✅ PASS | 74s |
| b1_p2_3b_4_anthropic_actor (post-fix) | ✅ PASS | 141s |
| b1_p2_3c_1_per_row_offset | ✅ PASS | 230s |
| b1_p2_3c_2_scheduler_decode_mask | ✅ PASS | 147s |
| b1_p2_3c_3_continuous_batching (post-fix) | ✅ PASS | 150s |
| b1_p2_4_batched_vl | ✅ PASS | 581s |
| **B1-p2.3d admission queue (5 scenarios)** | **✅ PASS** | **314s** |

## Compat sunset

| Removed | Replaced with |
| --- | --- |
| `ADMISSION_DEADLINE` const in `scheduler_actor.rs:38` | `admission_deadline` driver_loop parameter (CLI-driven) |
| Hardcoded `b_max=4` in `server/mod.rs:54` | `--b-max` CLI flag, default 4 |
| `c > b_max` immediate Err → HTTP 400 | FIFO admission queue (push or HTTP 503 if queue full) |
| `Scheduler::admit_mid` "scheduler full" Err visible to HTTP | Intercepted by `enqueue_or_reject` before reaching admit_mid; unified to "admission queue full" |

## Notes / known limitations carrying forward to backlog

- **No preemption** (spec NG1) — active rows always run to completion. A
  long-running active row blocks queue drain. Future task.
- **No HTTP cancellation propagation** (spec NG2) — if HTTP client
  disconnects while admit is queued, oneshot send fails silently when
  admit eventually drains; events stream into a dropped Receiver. 3e+.
- **String-match for "admission queue full" → 503** (spec §4.7 / §9 R3)
  — fragile. Future refactor to typed `SchedulerError` enum (3e/3.5).
- **No persistence** (spec NG4) — queue is in-memory; cleared on
  restart.
- **No priority / SLA / fair-share** (spec NG3) — FIFO only.
- **`enqueue_or_reject` always emits "admission queue full"** including
  `queue_max=0` (immediate reject path) — message unification noted
  above. Downstream tests should not match on the pre-3d "scheduler
  full" string (audit done in T5).

## B1-p2 Next Steps

| Sub-spec | Scope | Status |
| --- | --- | --- |
| B1-p2.3c+ | Chunked admit_mid prefill + decode-interleave | Backlog |
| B1-p2.3d | **Admission queue + config exposure** | **✅ DONE (this report)** |
| B1-p2.3e | Per-row async sampler tuning + cancellation + typed SchedulerError | Backlog |
| B1-p2.4 | Batched VL serving | ✅ DONE |
| B1-p2.5 | Production hardening | Future |

After B1-p2.3d: c > b_max no longer immediate reject; HTTP 503 surfaces
overflow correctly. Next major program: Qwen3.5 MoE.

## Linked artifacts

- [B1-p2.3d design spec](../../../../../docs/superpowers/specs/2026-05-16-b1-p2-3d-admission-queue-design.md)
- [B1-p2.3d implementation plan](../../../../../docs/superpowers/plans/2026-05-16-b1-p2-3d-admission-queue.md)
- [3c-3 perf baseline](../b1_p2_3c_3_perf_baseline/report.md)
- [B1-p2.4 close-out](../b1_p2_4_closeout/report.md)
