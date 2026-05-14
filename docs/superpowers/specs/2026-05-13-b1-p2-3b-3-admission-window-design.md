# B1-p2.3b-3 — Admission window + multi-request batching activation (design)

**Date:** 2026-05-13
**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3b-2 head `7c69919`)
**Predecessor sub-phase:** B1-p2.3b-2 — SchedulerActor skeleton + OpenAI text-path swap (closed at commit `7862b1f` + `7c69919`)
**Sibling sub-phase:** 3b-4 — Anthropic handler refactor (separate spec; depends on 3b-2 not on 3b-3)
**Successor sub-phases:** 3c (per-row offset), 3c+ (chunked batched prefill), 3d (admission queue + preemption), 3e (per-row sampler tuning), B1-p2.4 (VL B>1)

---

## §1 Goals

1. Replace 3b-2's `driver_loop` "one-admit-per-batch" with a hybrid admission-window: drain additional `SchedulerCommand::Admit` messages for up to `5 ms` after the first admit, OR until `active_count() == b_max` (saturate-trigger), whichever happens first.
2. Add `batch_count: Arc<AtomicU64>` to `SchedulerActorHandle` (alongside `admit_count`) as a test hook so integration tests can assert `batch_count < admit_count` when multi-admit batching genuinely happens.
3. Provide 4 integration scenarios that prove multi-request batching, b_max-saturate path, deadline-fires-with-one-admit, and scheduler↔GS concurrent-lock non-deadlock.
4. Fold in 3b-2's final-review Minors (M1 — `evict_all` Err logging; M2 — `chunk_size == 0` routing-semantic comment; M3 — concurrent-scheduler+GS lock test) as part of the same sub-phase.

## §2 Non-goals

- **Admission queue.** Defer to 3d. When the scheduler is full mid-window and `admit` returns `Err("scheduler full: no row available")`, 3b-3 still propagates the error to the caller — it does not queue and retry.
- **Configurable deadline.** 5 ms is hardcoded for 3b-3. Surfacing this as `AppConfig::admission_window_ms` or a CLI flag is deferred to 3d/3e.
- **Anthropic handler refactor.** Defer to 3b-4. `core/server/anthropic.rs` continues using GenerationStream regardless of admission-window behavior.
- **Request cancellation.** No `SchedulerCommand::Cancel { id }` variant. If a handler drops its `event_rx`, the actor's `route_event` will silently discard subsequent events for that row (until the batch finishes naturally). 3d may add cancellation alongside preemption.
- **HTTP/axum end-to-end concurrent tests.** Spec §5 integration tests run at the actor level (skip axum). The wire format is unchanged from 3b-2; existing 3b-2 Scenario A already validates HTTP-to-actor wiring for single requests.
- **Per-row sampler config (temperature/top_k).** 3e responsibility.

## §3 Background

### 3.1 Where 3b-2 left off

3b-2 (commits `8dd3590` → `7c69919`) shipped:

- `SchedulerActor` module (`core/server/scheduler_actor.rs`) with `driver_loop`, `run_batch_once`, `route_event` and `SchedulerActorHandle { cmd_tx, admit_count }`
- `AppState.scheduler_handle` field, populated by `serve()` calling `spawn_scheduler_actor(model.clone(), 4)`
- OpenAI handler routing — text-only short-prompt requests go to actor; VL / long-prompt fall back to `GenerationStream` (sunset-tracked compat)
- 3 integration scenarios at bit_id=1.0000 for the swap-in

Crucially, 3b-2's `driver_loop` body is exactly the line being replaced:

```rust
// scheduler_actor.rs:96-103 (at HEAD 7c69919)
while let Some(cmd) = cmd_rx.blocking_recv() {
    match cmd {
        SchedulerCommand::Admit { request, reply_tx } => {
            ...
            // 3b-2: one-admit-per-batch. 3b-3 replaces this
            // with admission-window logic that drains additional
            // SchedulerCommand::Admit messages before batching.
            if let Err(e) = run_batch_once(...)
```

The sunset marker at `scheduler_actor.rs:103` precisely names 3b-3's job: drain additional admits before batching.

### 3.2 Multi-request event routing is already production-ready

3b-2 designed `event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>` to support multi-row batches even though the driver only ever ran 1-row batches. The per-row routing in `route_event` and the `event_txs.clear()` cleanup in `run_batch_once` already work for batches of size N — 3b-3 only changes when `run_batch_once` is invoked, not how events flow.

This is the load-bearing inheritance from 3b-2: the change in 3b-3 is small (replace one blocking_recv with select!), but the impact (multi-request throughput unlock) is large because the rest of the pipeline already supports it.

### 3.3 Industry references

- **vLLM**: `max_batch_delay_ms` typically 1-2 s default but tuned to single-digit ms for interactive workloads. Deadline-based admission window.
- **TensorRT-LLM (Triton)**: `batch_wait_timeout_ms`, configurable in ms; balances latency vs. batch size. Deadline-based.
- **SGLang**: continuous batching with saturate-leaning policies; less explicit "single admission window timeout" constant published.

3b-3 picks **hybrid deadline + saturate** matching the dominant production pattern (vLLM/TRT-LLM): the first admit starts a deadline timer; further admits accumulate; either `active_count() == b_max` (saturate) or the deadline expires triggers `prefill_admitted`. The deadline is a hard limit — additional admits during the window do NOT reset it.

### 3.4 First Tokio `select!` usage in the codebase

The codebase has no production `tokio::select!` usage today. 3b-3 introduces the pattern. The spec calls it out explicitly so future reviewers know the precedent.

### 3.5 Lock strategy: inherited from 3b-2

`model.blocking_lock()` is held only during `run_batch_once`. During the admission window (`select! { admit | sleep }`), the lock is **not** held — GS-path handlers can acquire it freely. Once the batch starts, GS-path handlers block on the lock until the batch finishes (typically 100 ms - 30 s depending on max_new_tokens). This is unchanged from 3b-2.

## §4 Architecture

### 4.1 Driver state machine (after 3b-3)

```text
[Idle] ── first admit arrives ──> [Admitting]
                                       │
                            ┌──────────┼──────────┐
                            │          │          │
                  (deadline expires)  (more)  (saturate:
                            │       admits     active_count
                            │       arrive       == b_max)
                            │       inside        │
                            │       window        │
                            │          │          │
                            v          v          v
                      [Run batch: prefill_admitted + step* + evict_all]
                            │
                            v
                          [Idle]
```

The driver is single-threaded — only one batch active at a time. The Scheduler instance is owned by the driver and not exposed.

### 4.2 `SchedulerActorHandle` extension

3b-2 form:

```rust
#[derive(Clone)]
pub struct SchedulerActorHandle {
    pub cmd_tx: mpsc::Sender<SchedulerCommand>,
    #[doc(hidden)]
    pub admit_count: Arc<AtomicU64>,
}
```

3b-3 adds two fields:

```rust
#[derive(Clone)]
pub struct SchedulerActorHandle {
    pub cmd_tx: mpsc::Sender<SchedulerCommand>,
    #[doc(hidden)]
    pub admit_count: Arc<AtomicU64>,
    #[doc(hidden)]
    pub batch_count: Arc<AtomicU64>,        // NEW: incremented per run_batch_once
    #[doc(hidden)]
    pub saturate_triggered: Arc<AtomicU64>, // NEW: incremented when drain_window exits via active_count == b_max
}
```

Test invariants:

- `batch_count == 1` after a single Admit command (regardless of how many rows in the batch)
- `batch_count == admit_count` if every admit triggered its own batch (no batching happening)
- `batch_count < admit_count` if multi-admit batching is occurring (3b-3 success signal)
- `saturate_triggered` increments only when `drain_window` exits because `active_count() >= b_max` (NOT when the deadline expires). Used by Scenario 2 to verify the saturate path was taken without relying on wall-time measurement.

### 4.3 Driver loop after 3b-3

```rust
const ADMISSION_DEADLINE: Duration = Duration::from_millis(5);

fn driver_loop(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    mut cmd_rx: mpsc::Receiver<SchedulerCommand>,
    admit_count: Arc<AtomicU64>,
    batch_count: Arc<AtomicU64>,
) {
    let mut sched = Scheduler::new(b_max);
    let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
    let rt = tokio::runtime::Handle::current();

    loop {
        // Idle: wait for first admit (or shutdown if cmd_rx closes).
        let Some(first_cmd) = rt.block_on(cmd_rx.recv()) else { return };
        handle_admit(first_cmd, &mut sched, &mut event_txs, &admit_count);

        // Admitting: drain additional admits until deadline or saturate.
        // (Skip if first admit already saturated.)
        if sched.active_count() < b_max {
            rt.block_on(drain_window(
                &mut cmd_rx,
                &mut sched,
                &mut event_txs,
                &admit_count,
                b_max,
                ADMISSION_DEADLINE,
            ));
        }

        // Run the batch.
        batch_count.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = run_batch_once(&mut sched, &model, &mut event_txs) {
            tracing::error!("[SchedulerActor] batch error: {e:?}");
            if let Err(evict_err) = sched.evict_all() {
                // M1 fix: surface evict_all failure; rely on 3b-1 poison
                // flag to reject subsequent admits.
                tracing::warn!(
                    "[SchedulerActor] evict_all after batch error also failed: {evict_err:?}; \
                     relying on 3b-1 poison flag to reject subsequent admits"
                );
            }
            event_txs.clear();
        }
    }
}

async fn drain_window(
    cmd_rx: &mut mpsc::Receiver<SchedulerCommand>,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    b_max: usize,
    deadline: Duration,
) {
    let timer = tokio::time::sleep(deadline);
    tokio::pin!(timer);
    loop {
        tokio::select! {
            biased;
            _ = &mut timer => return, // hard deadline — no extension on new admits
            maybe = cmd_rx.recv() => {
                let Some(cmd) = maybe else { return }; // channel closed
                handle_admit(cmd, sched, event_txs, admit_count);
                if sched.active_count() >= b_max {
                    return; // saturate-trigger
                }
            }
        }
    }
}

fn handle_admit(
    cmd: SchedulerCommand,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
) {
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    match sched.admit(request) {
        Ok(id) => {
            admit_count.fetch_add(1, Ordering::Relaxed);
            event_txs.insert(id, event_tx);
            if reply_tx
                .send(Ok(AdmitReply { request_id: id, event_rx }))
                .is_err()
            {
                let _ = sched.evict(id);
                event_txs.remove(&id);
            }
        }
        Err(e) => {
            let _ = reply_tx.send(Err(e));
        }
    }
}
```

Key implementation notes:

- **`biased;` in `select!`** — gives the deadline branch priority. Without it, if both branches are ready at the same Tokio scheduler tick, the result is non-deterministic. `biased;` makes the deadline check first, guaranteeing the hard-limit semantic.
- **`tokio::runtime::Handle::current().block_on(...)`** — `driver_loop` runs on `tokio::task::spawn_blocking` (because Scheduler is `!Send` and model lock is sync). `block_on` lets the blocking driver drive async operations (`cmd_rx.recv()`, `tokio::time::sleep`). The runtime handle is captured from the spawning context.
- **`handle_admit` factored out** — same Admit handling logic used for both "first admit" and "in-window admits". DRY.

### 4.4 `run_batch_once` and `route_event` — unchanged

3b-2's `run_batch_once` already handles N-row batches (calls `prefill_admitted` once for the whole batch, loops `step` until all rows finished, evicts all). 3b-3 invokes it exactly the same way; the difference is just that `sched.active_count()` may be > 1 at the moment of invocation.

`route_event` already dispatches `StepEvent`s to per-`RequestId` channels. Unchanged.

### 4.5 OpenAI handler — M2 fix

Add a 1-line comment near `openai.rs:364` explaining the `chunk_size == 0` routing semantic:

```rust
// COMPAT(3b-2): long-prompt fallback to GS sunsets in 3c+ chunked-prefill phase.
//
// Note: when prefill_chunk_size == 0 (chunking disabled by config), this
// predicate routes ALL text-only requests to the SchedulerActor regardless
// of length. The scheduler's prefill_admitted internally calls
// batched_prefill in one shot — equivalent to the GS path's behavior when
// chunking is also disabled there. The 3c+ chunked-prefill phase will need
// to revisit this semantic.
let has_images = request.pixel_values.is_some();
let prompt_len = request.prompt_ids.len();
let use_scheduler =
    !has_images && (state.prefill_chunk_size == 0 || prompt_len <= state.prefill_chunk_size);
```

### 4.6 Module surface summary

```text
ironmlx/src/core/server/scheduler_actor.rs   — MODIFY
  + Add `batch_count` field to SchedulerActorHandle
  + Initialize in spawn_scheduler_actor
  + Replace driver_loop body with select!-based admission window
  + Add drain_window async helper
  + Extract handle_admit helper (already in driver body — extract for DRY)
  + ADMISSION_DEADLINE = Duration::from_millis(5) const
  + M1 fix: tracing::warn! on evict_all Err

ironmlx/src/core/server/openai.rs            — MODIFY
  + M2 fix: 4-line comment block on chunk_size==0 routing semantics

ironmlx/tests/b1_p2_3b_3_admission_window.rs — NEW (~400 lines)
  + 4 integration scenarios

ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/
    b1_p2_3b_3_closeout/report.md             — NEW close-out
```

Zero modifications to: `core/server/anthropic.rs`, `core/server/mod.rs`, `core/server/chat_format.rs`, `core/scheduler.rs`, `core/generate.rs`, `core/sampler.rs`, `core/tokenizer.rs`, `core/cache/`, `models/`, `nn/`.

## §5 Tests

### 5.1 Unit tests

No new unit tests in `scheduler_actor.rs`. The 3b-2 `driver_shuts_down_when_cmd_channel_closes` test still applies — the new driver loop has the same shutdown semantic (`cmd_rx.recv()` returning `None` → driver exits via the `let Some(first_cmd) = ... else { return }` early return).

### 5.2 Integration test `tests/b1_p2_3b_3_admission_window.rs`

4 `#[ignore]` `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]` scenarios:

**Scenario 1 — `admission_window_two_concurrent_admits_batch_together`**
1. Load model + tokenizer
2. Run B=1 GS baseline for prompt A and prompt B
3. Spawn actor with `b_max = 4`
4. `tokio::task::JoinSet` spawn 2 admits CONCURRENTLY (no sleep between)
5. Each spawned task: sends `Admit` cmd, awaits reply, drains `event_rx`, collects tokens
6. After JoinSet completes:
   - Assert `admit_count == 2`
   - **Assert `batch_count == 1`** (the load-bearing invariant — proves batching)
   - Per-row tokens vs baseline: `argmax_bit_id_ratio >= 0.95` for both

**Scenario 2 — `admission_window_b_max_saturate_triggers_immediate_prefill`**
1. Spawn actor with `b_max = 4`
2. JoinSet spawn 4 admits concurrently
3. Record wall-time start before spawn, end after all complete
4. Assertions:
   - `admit_count == 4`
   - `batch_count == 1`
   - Wall time NOT inflated by 5ms deadline — saturate path used (admit elapsed should be < 5ms before prefill; if model.forward dominates the total, just verify total time is reasonable)
   - Per-row bit_id ≥ 0.95 vs respective baselines

   (The wall-time assertion is hard to make precise because Tokio scheduling has jitter. A reliable proxy: instrument the driver with an additional `#[doc(hidden)]` counter `saturate_triggered: Arc<AtomicU64>` that increments when the saturate path exits `drain_window`. Spec adds this counter alongside `batch_count`. Assert `saturate_triggered == 1` after Scenario 2.)

**Scenario 3 — `admission_window_deadline_fires_with_single_admit`**
1. Spawn actor with `b_max = 4`
2. Submit 1 admit
3. Drain `event_rx` to completion
4. Assertions:
   - `admit_count == 1`
   - `batch_count == 1`
   - `saturate_triggered == 0` (deadline path used, not saturate)
   - (Optional, fragile due to timer precision: total elapsed ≥ 5ms — skip if Tokio scheduler can't guarantee on test runners)

**Scenario 4 — `admission_window_concurrent_scheduler_and_gs_no_deadlock`** (M3 fix)
1. Spawn actor with `b_max = 4`
2. JoinSet:
   - Task A: send 1 admit to scheduler → drain events
   - Task B: directly invoke `GenerationStream` via `model.blocking_lock()` from a `spawn_blocking` task (simulating GS path)
3. Both tasks must complete within reasonable time (e.g., 60s total — pass-through model.forward dominates)
4. Verify both produced sensible token output (non-empty)
5. Assertions:
   - `admit_count == 1` (only scheduler task incremented)
   - `batch_count == 1`
   - Both `task_a_tokens.len() > 0` and `task_b_tokens.len() > 0`
   - No `tokio::time::timeout` panic (proxies the no-deadlock property)

### 5.3 Acceptance gates

- All 4 integration scenarios pass
- Existing regression suite (P6.3 / P6.6 / P6.7 / B1-p2.1 / B1-p2.2 / B1-p2.3b-1 / B1-p2.3b-2): green
- `cargo +nightly fmt --all -- --check`, `clippy -D warnings`, `cargo build --release -p ironmlx`: clean
- Lib test count: 188 (3b-2 baseline) + 0 (no new unit tests in this phase)

## §6 Estimate

**3-5 working days:**
- D1 — `driver_loop` rewrite + `batch_count` / `saturate_triggered` hooks + M1/M2 fixes
- D2 — Scenario 1 + 2 (basic batching activation + saturate-path proof)
- D3 — Scenario 3 + 4 (deadline path + concurrent-with-GS proof)
- D4 — Full regression sweep + close-out
- D5 (buffer) — review fixes

## §7 Compat sunset notes

3b-3 ships no new compat. The 3 sunset markers from 3b-2 remain:
- VL → GS path: sunsets in B1-p2.4
- Long prompt → GS path: sunsets in 3c+ chunked-prefill phase
- Anthropic → GS path: sunsets in 3b-4

3b-3 itself adds one item to the future-cleanup list: the **hardcoded 5 ms `ADMISSION_DEADLINE`**. A future phase (3d or 3e) should expose this via `AppConfig` and a CLI flag. Spec §2 marks this explicitly so the next-phase implementer is aware.

## §8 Alternatives considered

| Decision | Selected | Rejected |
| --- | --- | --- |
| Window form | Hybrid deadline + saturate | Saturate-only (effectively no batching unless admits land in same tokio tick); Pure deadline (wasted latency when batch already full); Configurable deadline (3b-3 scope creep) |
| Default deadline | 5 ms hardcoded | 0 ms (greedy = same as 3b-2 behavior); 10 ms (higher TTFT cost without proportional throughput gain at small b_max); configurable now (3b-3 scope creep) |
| Fairness | Hard deadline (no reset on new admits) | Soft deadline (resets on each admit) — starvation risk |
| Test layer | Actor-level + `batch_count` hook | HTTP/axum end-to-end (heavy setup, fragile under Tokio scheduler jitter); both layers (scope creep); deadlock-only (under-tested) |
| `select!` flavor | `tokio::select!` with `biased;` | Manual `poll_fn` (more code); `futures::select!` (extra dependency) |
| `block_on` bridge | `tokio::runtime::Handle::current().block_on(...)` inside spawn_blocking | Move driver to async task with `task::yield_now` (Scheduler is `!Send`); rewrite Scheduler to be `Send` (out of scope, would force Sampler refactor) |

## §9 Risk register

| Risk | Mitigation |
| --- | --- |
| Tokio scheduler jitter makes wall-time-based assertions in Scenario 2/3 flaky | Use atomic counters (`saturate_triggered`) as primary signal; wall-time only as supplementary diagnostic |
| `select!` with `biased;` is unfamiliar pattern; engineers may miss the deadline-hard-limit semantic | Spec §4.3 explicit comment; in-code comment near the `biased;` directive |
| Multi-admit batch hits Scheduler::prefill_admitted edge cases not covered by 3b-1 single-row tests | 3b-1 explicitly tested B=2 and B=4 batches at bit_id=1.0000; the runtime path is identical |
| Concurrent GS-path + scheduler-path lock contention exposes deadlock | Scenario 4 directly tests this; timer-based `tokio::time::timeout` wraps the test to surface deadlocks as test failures |
| `batch_count` increment under racy admission edge case (e.g., driver shuts down mid-window) | Increment AFTER drain_window returns, BEFORE `run_batch_once`; if `run_batch_once` errors, batch_count still reflects a started-but-failed batch — acceptable for diagnostic purpose |

## §10 Linked artifacts

- Predecessor spec: [`docs/superpowers/specs/2026-05-13-b1-p2-3b-2-scheduler-actor-skeleton-design.md`](2026-05-13-b1-p2-3b-2-scheduler-actor-skeleton-design.md)
- Predecessor plan: [`docs/superpowers/plans/2026-05-13-b1-p2-3b-2-scheduler-actor-skeleton.md`](../plans/2026-05-13-b1-p2-3b-2-scheduler-actor-skeleton.md)
- Predecessor close-out: [`ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_2_closeout/report.md`](../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_2_closeout/report.md)
- 3b-2's driver_loop sunset marker: [`ironmlx/src/core/server/scheduler_actor.rs:103`](../../ironmlx/src/core/server/scheduler_actor.rs#L103)
- Scheduler API (unchanged): [`ironmlx/src/core/scheduler.rs`](../../ironmlx/src/core/scheduler.rs)
- Test pattern reference: [`ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs`](../../ironmlx/tests/b1_p2_3b_2_scheduler_actor.rs)
