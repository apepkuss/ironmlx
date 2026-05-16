# B1-p2.3d Admission Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `c > b_max` HTTP-400 "scheduler full" reject path with a bounded FIFO admission queue inside `driver_loop`, then expose `b_max` / `admission_deadline_ms` / `admission_queue_max` as `ServeArgs` CLI flags. Queue overflow returns HTTP 503 with `Retry-After: 5`.

**Architecture:** All queue state lives inside `driver_loop` as a `VecDeque<PendingAdmit>` (no `Arc<Mutex<_>>`, single-task ownership). Four admission paths (outer first, drain_window saturate, rolling Admit branch, post-`gc_finished_rows` slot-free) all funnel through the same push/drain helpers. `gc_finished_rows` is followed by a `while active_count < b_max && !queue.is_empty()` drain loop. Decode path (`step_inner`, `build_per_row_decode_mask`, `build_decode_position_ids`) is UNCHANGED.

**Tech Stack:** Rust 2024, `tokio` (mpsc + oneshot + `select!` biased), `clap` (CLI), `axum` (HTTP), `std::collections::VecDeque`, `std::sync::atomic::{AtomicU64, AtomicUsize}`.

**Spec source:** [`docs/superpowers/specs/2026-05-16-b1-p2-3d-admission-queue-design.md`](../specs/2026-05-16-b1-p2-3d-admission-queue-design.md) (commit `22bb8af`).

**Branch:** `ironmlx-b1-p2-3d-admission-queue` cut from `ironmlx-b1-p2-4-batched-vl` head `22bb8af` (post-3d-spec doc commit — gives 3d work branch the spec + close-out + plan as doc-only context).

**Cargo env:** every `cargo` invocation requires `MLX_DIR=$HOME/.local/mlx`. Env does NOT persist across subshells — prefix each `cargo` command explicitly. Hygiene gate per CLAUDE.md: `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings`, `cargo +stable build --release`.

**Model fixture:** `MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)`. Integration tests also need `QWEN35_MODEL="$MODEL"` env var (matches 3c-3 / B1-p2.4 fixture-loader pattern).

---

## Pre-flight: cut the working branch

- [ ] **Step 0.1: Cut branch off current B1-p2.4 HEAD (post-3d-spec)**

```bash
cd /Volumes/Dev/cxx-mlx
git checkout -b ironmlx-b1-p2-3d-admission-queue
git log --oneline -1
```

Expected: HEAD at `22bb8af docs(b1-p2.3d): admission queue + config exposure design spec` (or wherever the 3d spec commit landed).

- [ ] **Step 0.2: Pre-flight hygiene**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
```

All three must pass. If any fail, fix on this branch as a separate hygiene commit before starting T1.

---

## File structure

| File | Responsibility | Change kind |
| --- | --- | --- |
| `ironmlx/src/core/server/scheduler_actor.rs` | Actor driver_loop + admit-handling + new admission queue + counters | Modify (largest — T1+T2) |
| `ironmlx/src/core/server/mod.rs` | HTTP server boot + `AppState` | Modify (+3 fields, +3 params) |
| `ironmlx/src/cli/serve.rs` | `serve` subcommand args | Modify (+3 `#[arg]` flags) |
| `ironmlx/src/core/server/openai.rs` | OpenAI HTTP handler | Modify (Err string discrimination → 503) |
| `ironmlx/src/core/server/anthropic.rs` | Anthropic HTTP handler | Modify (same as openai) |
| `ironmlx/tests/b1_p2_3b_3_admission_window.rs` | Existing admission-window suite | Modify (update spawn_scheduler_actor calls — new signature) |
| `ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs` | Existing anthropic-actor suite | Modify (update spawn_scheduler_actor calls) |
| `ironmlx/tests/b1_p2_3c_3_continuous_batching.rs` | Existing continuous-batching suite | Modify (update spawn_scheduler_actor calls) |
| `ironmlx/tests/b1_p2_4_batched_vl.rs` | Existing VL batched suite | Modify (update spawn_scheduler_actor calls — 4 sites) |
| `ironmlx/tests/b1_p2_3d_admission_queue.rs` | New integration scenarios | Create (5 scenarios, ~400 LoC) |
| `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3d_closeout/report.md` | Close-out | Create (gitignored — `-f` commit) |

Estimated total: ~780 LoC of changes/additions across ~11 files; 5–6 commits.

---

## Task 1: `driver_loop` admission queue state + push/drain logic + signature extension

**Why this is first:** All other tasks depend on the new `spawn_scheduler_actor` signature. T1 must update all callers (server/mod.rs + 4 test files) in the same commit so the workspace compiles.

**Files:**
- Modify: `ironmlx/src/core/server/scheduler_actor.rs` (driver_loop body + struct + spawn fn + remove `ADMISSION_DEADLINE` const)
- Modify: `ironmlx/src/core/server/mod.rs:54` (call site)
- Modify: `ironmlx/tests/b1_p2_3b_3_admission_window.rs` (spawn_scheduler_actor caller, ~3 sites)
- Modify: `ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs` (~2 sites)
- Modify: `ironmlx/tests/b1_p2_3c_3_continuous_batching.rs` (~3 sites)
- Modify: `ironmlx/tests/b1_p2_4_batched_vl.rs` (4 sites)

### Steps

- [ ] **Step 1.1: Extend `SchedulerActorHandle` with 2 new atomic counters**

Edit `ironmlx/src/core/server/scheduler_actor.rs` — locate `pub struct SchedulerActorHandle` (line ~72) and append two fields before the closing `}`:

```rust
#[derive(Clone)]
pub struct SchedulerActorHandle {
    pub cmd_tx: mpsc::Sender<SchedulerCommand>,
    #[doc(hidden)]
    pub admit_count: Arc<AtomicU64>,
    #[doc(hidden)]
    pub batch_count: Arc<AtomicU64>,
    #[doc(hidden)]
    pub saturate_triggered: Arc<AtomicU64>,
    /// Test-observable peak `admission_queue.len()` ever reached. Used by
    /// integration tests to confirm the queue drained (e.g., `peak >= N` for
    /// c=N+b_max admit burst). Doc-hidden — production code shouldn't read it.
    #[doc(hidden)]
    pub queue_depth_peak: Arc<AtomicUsize>,
    /// Test-observable count of admit requests rejected with "admission
    /// queue full" Err (queue_max overflow). Doc-hidden.
    #[doc(hidden)]
    pub queue_rejected: Arc<AtomicU64>,
}
```

`AtomicUsize` requires `use std::sync::atomic::AtomicUsize;` — add to existing `use std::sync::atomic::{AtomicU64, Ordering};` at top of file:

```rust
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
```

- [ ] **Step 1.2: Remove `ADMISSION_DEADLINE` const + add `PendingAdmit` struct**

In `ironmlx/src/core/server/scheduler_actor.rs`:

Delete lines around 31–38 (the const + its leading doc comment):

```rust
/// Admission window deadline: maximum time `driver_loop` waits to pack
/// additional `Admit` commands into the current batch after the first
/// admit arrives. Hard limit — new admits during the window do NOT
/// reset it (prevents starvation under sustained admit pressure).
///
/// Hardcoded for 3b-3; a future phase (3d/3e) will surface this via
/// `AppConfig` and a CLI flag.
const ADMISSION_DEADLINE: Duration = Duration::from_millis(5);
```

Add `VecDeque` import to existing `use std::collections::HashMap;` (likely already exists at top):

```rust
use std::collections::{HashMap, VecDeque};
```

Add a new private struct near `RollingEvent` (around line 56):

```rust
/// A request parked in `driver_loop`'s admission queue while the scheduler
/// is at `active_count == b_max`. Drained when `gc_finished_rows` frees a
/// slot, then handed to `handle_admit_mid`.
struct PendingAdmit {
    request: GenerateRequest,
    reply_tx: oneshot::Sender<Result<AdmitReply>>,
}
```

- [ ] **Step 1.3: Extend `spawn_scheduler_actor` signature**

Replace the `spawn_scheduler_actor` fn (line ~100) with:

```rust
/// Spawn the driver task and return a handle. The driver runs on
/// `tokio::task::spawn_blocking` because [`Scheduler`] is `!Send` (sampler
/// holds a `Cell<Array>`) and the model lock is sync.
///
/// # Arguments
/// - `model` — shared model handle (Mutex-protected sync state).
/// - `b_max` — maximum concurrent in-flight requests (Scheduler slot count).
/// - `admission_deadline` — drain-window timeout after the first admit in a
///   batch arrives. Hard limit; new admits do not reset it.
/// - `admission_queue_max` — capacity of the FIFO admission queue. `0`
///   disables queueing (immediate Err on saturation, mirroring pre-3d).
pub fn spawn_scheduler_actor(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
) -> SchedulerActorHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let admit_count = Arc::new(AtomicU64::new(0));
    let batch_count = Arc::new(AtomicU64::new(0));
    let saturate_triggered = Arc::new(AtomicU64::new(0));
    let queue_depth_peak = Arc::new(AtomicUsize::new(0));
    let queue_rejected = Arc::new(AtomicU64::new(0));
    let admit_count_for_task = admit_count.clone();
    let batch_count_for_task = batch_count.clone();
    let saturate_triggered_for_task = saturate_triggered.clone();
    let queue_depth_peak_for_task = queue_depth_peak.clone();
    let queue_rejected_for_task = queue_rejected.clone();
    tokio::task::spawn_blocking(move || {
        driver_loop(
            model,
            b_max,
            admission_deadline,
            admission_queue_max,
            cmd_rx,
            admit_count_for_task,
            batch_count_for_task,
            saturate_triggered_for_task,
            queue_depth_peak_for_task,
            queue_rejected_for_task,
        );
    });
    SchedulerActorHandle {
        cmd_tx,
        admit_count,
        batch_count,
        saturate_triggered,
        queue_depth_peak,
        queue_rejected,
    }
}
```

- [ ] **Step 1.4: Add `enqueue_or_reject` + `drain_admission_queue` helper functions**

Below the existing `handle_admit_mid` function (after line ~436), add two helpers:

```rust
/// Push a pending admit into the queue if there's capacity; otherwise reply
/// with Err("admission queue full") and bump `queue_rejected`. Updates
/// `queue_depth_peak` via `fetch_max`.
fn enqueue_or_reject(
    cmd: SchedulerCommand,
    queue: &mut VecDeque<PendingAdmit>,
    queue_max: usize,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
) {
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    if queue.len() >= queue_max {
        queue_rejected.fetch_add(1, Ordering::Relaxed);
        let _ = reply_tx.send(Err(anyhow::anyhow!(
            "admission queue full: capacity={queue_max} reached"
        )));
        return;
    }
    queue.push_back(PendingAdmit { request, reply_tx });
    queue_depth_peak.fetch_max(queue.len(), Ordering::Relaxed);
}

/// Drain the admission queue head-by-head while the Scheduler has free
/// slots. Each drained entry is handed to `handle_admit_mid` which runs
/// the B=1 prefill + adopts the row + sends `AdmitReply`. Stops when the
/// queue empties or `active_count() == b_max`.
fn drain_admission_queue(
    queue: &mut VecDeque<PendingAdmit>,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<Qwen35Model>>,
    b_max: usize,
) {
    while sched.active_count() < b_max {
        let Some(pending) = queue.pop_front() else {
            return;
        };
        let cmd = SchedulerCommand::Admit {
            request: pending.request,
            reply_tx: pending.reply_tx,
        };
        handle_admit_mid(cmd, sched, event_txs, admit_count, model);
    }
}
```

`anyhow::anyhow!` is already in scope via the `use anyhow::*` at the top of the file (verify with `grep '^use anyhow' /Volumes/Dev/cxx-mlx/ironmlx/src/core/server/scheduler_actor.rs`); if absent, add `use anyhow::anyhow;` to the imports.

- [ ] **Step 1.5: Refactor `driver_loop` signature + body — phase A (signature + outer first admit + drain_window)**

Replace the `driver_loop` fn (line ~126) up through the end of the admission window (around line ~163). Full replacement up to the prefill block:

```rust
#[allow(clippy::too_many_arguments)]
fn driver_loop(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    mut cmd_rx: mpsc::Receiver<SchedulerCommand>,
    admit_count: Arc<AtomicU64>,
    batch_count: Arc<AtomicU64>,
    saturate_triggered: Arc<AtomicU64>,
    queue_depth_peak: Arc<AtomicUsize>,
    queue_rejected: Arc<AtomicU64>,
) {
    let mut sched = Scheduler::new(b_max);
    let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
    let mut admission_queue: VecDeque<PendingAdmit> = VecDeque::new();
    let rt = tokio::runtime::Handle::current();

    'outer: loop {
        // ===== Outer Idle: block waiting for first admit (or shutdown). =====
        // Outer Idle is reached only after evict_all clears all slots; the
        // admission queue is invariantly empty here (any queue elements were
        // drained inside the rolling loop before reaching this point).
        let Some(first_cmd) = rt.block_on(cmd_rx.recv()) else {
            return; // cmd_rx closed; all senders dropped.
        };
        handle_admit(first_cmd, &mut sched, &mut event_txs, &admit_count);

        if sched.active_count() == 0 {
            // First admit failed (Err) — nothing to prefill. Wait for next.
            continue 'outer;
        }

        // ===== Admission window: drain additional admits until deadline
        //       or saturate at b_max. Beyond b_max within the window, push
        //       to admission_queue (bounded by admission_queue_max). =====
        if sched.active_count() < b_max {
            rt.block_on(drain_window(
                &mut cmd_rx,
                &mut sched,
                &mut event_txs,
                &mut admission_queue,
                &admit_count,
                &saturate_triggered,
                &queue_depth_peak,
                &queue_rejected,
                b_max,
                admission_queue_max,
                admission_deadline,
            ));
        }
```

- [ ] **Step 1.6: Update `drain_window` signature + body**

Replace the existing `drain_window` fn (around line ~330) with:

```rust
/// Drain additional `Admit` commands until either the deadline expires or
/// the Scheduler saturates at `b_max`. Hard deadline — new admits do NOT
/// reset the timer. Once saturated, additional admits within the window
/// go to the admission queue (bounded by `admission_queue_max`).
#[allow(clippy::too_many_arguments)]
async fn drain_window(
    cmd_rx: &mut mpsc::Receiver<SchedulerCommand>,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admission_queue: &mut VecDeque<PendingAdmit>,
    admit_count: &Arc<AtomicU64>,
    saturate_triggered: &Arc<AtomicU64>,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
    b_max: usize,
    queue_max: usize,
    deadline: Duration,
) {
    let timer = tokio::time::sleep(deadline);
    tokio::pin!(timer);
    let mut saturated = false;
    loop {
        tokio::select! {
            biased;
            _ = &mut timer => return,
            maybe = cmd_rx.recv() => {
                let Some(cmd) = maybe else { return }; // channel closed
                if saturated {
                    // Already at b_max — push to queue or reject.
                    enqueue_or_reject(
                        cmd,
                        admission_queue,
                        queue_max,
                        queue_depth_peak,
                        queue_rejected,
                    );
                    continue;
                }
                handle_admit(cmd, sched, event_txs, admit_count);
                if sched.active_count() >= b_max {
                    saturate_triggered.fetch_add(1, Ordering::Relaxed);
                    saturated = true;
                    // Stay in the loop until deadline so queued admits
                    // arriving during the window's remaining time are
                    // captured. (Pre-3d returned here; 3d keeps draining.)
                }
            }
        }
    }
}
```

- [ ] **Step 1.7: Refactor `driver_loop` body — phase B (rolling decode loop)**

Continue the `driver_loop` body — locate the prefill_admitted block (around line ~164) and the rolling loop (around line ~190). The prefill block stays as-is; only the rolling loop's `RollingEvent::Admit` arm and post-`gc_finished_rows` block change. Replace the rolling loop with:

```rust
        // ===== First-batch prefill. =====
        batch_count.fetch_add(1, Ordering::Relaxed);
        let prefill_result = {
            let model_lock = model.blocking_lock();
            sched.prefill_admitted(&model_lock)
        };
        match prefill_result {
            Ok(prefill_events) => {
                for ev in prefill_events {
                    route_event(ev, &event_txs);
                }
            }
            Err(e) => {
                tracing::error!("[SchedulerActor] prefill error: {e:?}");
                if let Err(evict_err) = sched.evict_all() {
                    tracing::warn!(
                        "[SchedulerActor] evict_all after prefill error also failed: \
                         {evict_err:?}; relying on 3b-1 poison flag to reject subsequent admits"
                    );
                }
                event_txs.clear();
                // Anything queued during the failed-batch window has nowhere
                // to land — reject with Err so callers see a clear error
                // rather than hanging.
                while let Some(pending) = admission_queue.pop_front() {
                    let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                        "scheduler poisoned after prefill error"
                    )));
                }
                continue 'outer;
            }
        }

        // ===== Rolling decode loop with biased mid-batch admit + queue drain. =====
        'rolling: loop {
            let evt: RollingEvent = rt.block_on(async {
                tokio::select! {
                    biased;
                    maybe_cmd = cmd_rx.recv() => match maybe_cmd {
                        Some(cmd) => RollingEvent::Admit(cmd),
                        None => RollingEvent::Shutdown,
                    },
                    _ = std::future::ready(()) => RollingEvent::Step,
                }
            });

            match evt {
                RollingEvent::Shutdown => {
                    event_txs.clear();
                    // Reject any queued admits — callers shouldn't hang.
                    while let Some(pending) = admission_queue.pop_front() {
                        let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                            "scheduler shutting down"
                        )));
                    }
                    return;
                }
                RollingEvent::Admit(cmd) => {
                    if sched.active_count() >= b_max {
                        // Slot full — push to queue (or reject if queue full).
                        enqueue_or_reject(
                            cmd,
                            &mut admission_queue,
                            admission_queue_max,
                            &queue_depth_peak,
                            &queue_rejected,
                        );
                    } else {
                        handle_admit_mid(
                            cmd,
                            &mut sched,
                            &mut event_txs,
                            &admit_count,
                            &model,
                        );
                    }
                }
                RollingEvent::Step => {
                    let step_result = {
                        let model_lock = model.blocking_lock();
                        sched.step(&model_lock)
                    };
                    match step_result {
                        Ok(events) => {
                            for ev in events {
                                route_event(ev, &event_txs);
                            }
                            sched.gc_finished_rows(&mut event_txs);
                            // ===== Post-gc queue drain. =====
                            // Free slots → pull from admission_queue head
                            // until either the queue empties or we re-
                            // saturate at b_max.
                            drain_admission_queue(
                                &mut admission_queue,
                                &mut sched,
                                &mut event_txs,
                                &admit_count,
                                &model,
                                b_max,
                            );
                        }
                        Err(e) => {
                            tracing::error!("[SchedulerActor] step error: {e:?}");
                            if let Err(evict_err) = sched.evict_all() {
                                tracing::warn!(
                                    "[SchedulerActor] evict_all after step error also failed: \
                                     {evict_err:?}; relying on 3b-1 poison flag to reject subsequent admits"
                                );
                            }
                            event_txs.clear();
                            while let Some(pending) = admission_queue.pop_front() {
                                let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                                    "scheduler poisoned after step error"
                                )));
                            }
                            continue 'outer;
                        }
                    }
                }
            }

            // ===== Exit rolling loop when active_count == 0 AND queue empty. =====
            // Spec §9 R1: if `active_count() == 0` but admission_queue is
            // non-empty (mid-rolling admit arrived AFTER all rows finished),
            // treat as a "new batch within rolling": evict_all to reset to
            // Idle, then admit from queue + drain_window + prefill_admitted
            // inline (mirrors the existing post-empty path but pulls the
            // first admit from the queue instead of cmd_rx).
            if sched.active_count() == 0 {
                if !admission_queue.is_empty() {
                    // Reset to Idle for fresh batch.
                    if let Err(evict_err) = sched.evict_all() {
                        tracing::warn!(
                            "[SchedulerActor] evict_all between batches (queue drain) failed: \
                             {evict_err:?}; rejecting queued admits"
                        );
                        while let Some(pending) = admission_queue.pop_front() {
                            let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                                "scheduler evict_all failed"
                            )));
                        }
                        event_txs.clear();
                        continue 'outer;
                    }
                    event_txs.clear();
                    // Pop first queued admit as the new batch's first admit.
                    let pending = admission_queue.pop_front().expect("queue non-empty checked");
                    handle_admit(
                        SchedulerCommand::Admit {
                            request: pending.request,
                            reply_tx: pending.reply_tx,
                        },
                        &mut sched,
                        &mut event_txs,
                        &admit_count,
                    );
                    if sched.active_count() == 0 {
                        // Admit failed; loop to drain more queue (or exit).
                        continue 'rolling;
                    }
                    if sched.active_count() < b_max {
                        // Drain queue head-by-head into the new batch (no
                        // deadline — these are already-queued admits, not
                        // racing-in cmd_rx). Then optionally drain_window
                        // for fresh cmd_rx admits.
                        while sched.active_count() < b_max {
                            let Some(p) = admission_queue.pop_front() else { break };
                            handle_admit(
                                SchedulerCommand::Admit {
                                    request: p.request,
                                    reply_tx: p.reply_tx,
                                },
                                &mut sched,
                                &mut event_txs,
                                &admit_count,
                            );
                        }
                        // Optionally absorb cmd_rx admits arriving right now.
                        if sched.active_count() < b_max {
                            rt.block_on(drain_window(
                                &mut cmd_rx,
                                &mut sched,
                                &mut event_txs,
                                &mut admission_queue,
                                &admit_count,
                                &saturate_triggered,
                                &queue_depth_peak,
                                &queue_rejected,
                                b_max,
                                admission_queue_max,
                                admission_deadline,
                            ));
                        }
                    }
                    batch_count.fetch_add(1, Ordering::Relaxed);
                    let prefill_result = {
                        let model_lock = model.blocking_lock();
                        sched.prefill_admitted(&model_lock)
                    };
                    match prefill_result {
                        Ok(events) => {
                            for ev in events {
                                route_event(ev, &event_txs);
                            }
                        }
                        Err(e) => {
                            tracing::error!("[SchedulerActor] re-prefill (queue drain) error: {e:?}");
                            if let Err(evict_err) = sched.evict_all() {
                                tracing::warn!(
                                    "[SchedulerActor] evict_all after re-prefill error also \
                                     failed: {evict_err:?}; rejecting remaining queued admits"
                                );
                            }
                            event_txs.clear();
                            while let Some(p) = admission_queue.pop_front() {
                                let _ = p.reply_tx.send(Err(anyhow::anyhow!(
                                    "scheduler poisoned after re-prefill error"
                                )));
                            }
                            continue 'outer;
                        }
                    }
                    continue 'rolling;
                }
                // Queue empty + no active rows — same logic as pre-3d.
                match cmd_rx.try_recv() {
                    Ok(cmd) => {
                        if let Err(evict_err) = sched.evict_all() {
                            tracing::warn!(
                                "[SchedulerActor] evict_all between batches failed: \
                                 {evict_err:?}; rejecting incoming admit"
                            );
                            let SchedulerCommand::Admit { reply_tx, .. } = cmd;
                            let _ = reply_tx.send(Err(evict_err));
                            event_txs.clear();
                            continue 'outer;
                        }
                        event_txs.clear();
                        handle_admit(cmd, &mut sched, &mut event_txs, &admit_count);
                        if sched.active_count() == 0 {
                            break 'rolling;
                        }
                        if sched.active_count() < b_max {
                            rt.block_on(drain_window(
                                &mut cmd_rx,
                                &mut sched,
                                &mut event_txs,
                                &mut admission_queue,
                                &admit_count,
                                &saturate_triggered,
                                &queue_depth_peak,
                                &queue_rejected,
                                b_max,
                                admission_queue_max,
                                admission_deadline,
                            ));
                        }
                        batch_count.fetch_add(1, Ordering::Relaxed);
                        let prefill_result = {
                            let model_lock = model.blocking_lock();
                            sched.prefill_admitted(&model_lock)
                        };
                        match prefill_result {
                            Ok(events) => {
                                for ev in events {
                                    route_event(ev, &event_txs);
                                }
                            }
                            Err(e) => {
                                tracing::error!("[SchedulerActor] re-prefill error: {e:?}");
                                if let Err(evict_err) = sched.evict_all() {
                                    tracing::warn!(
                                        "[SchedulerActor] evict_all after re-prefill error \
                                         also failed: {evict_err:?}; relying on 3b-1 poison \
                                         flag to reject subsequent admits"
                                    );
                                }
                                event_txs.clear();
                                continue 'outer;
                            }
                        }
                        continue 'rolling;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        break 'rolling;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        event_txs.clear();
                        return;
                    }
                }
            }
        }

        // After rolling loop: reset cache + Phase for next outer iteration.
        if matches!(sched.phase(), Phase::Decoding | Phase::Finished) {
            if let Err(evict_err) = sched.evict_all() {
                tracing::warn!(
                    "[SchedulerActor] evict_all at end of outer failed: {evict_err:?}; \
                     relying on 3b-1 poison flag to reject subsequent admits"
                );
            }
        }
        event_txs.clear();
    }
}
```

- [ ] **Step 1.8: Update `core/server/mod.rs` caller**

Edit `ironmlx/src/core/server/mod.rs` line ~54:

Replace:

```rust
    let scheduler_handle = scheduler_actor::spawn_scheduler_actor(model.clone(), 4);
```

With:

```rust
    // 3d: hardcoded defaults preserved here — T3 will route these from
    // ServeArgs + AppState. b_max=4, deadline=5ms, queue_max=32.
    let scheduler_handle = scheduler_actor::spawn_scheduler_actor(
        model.clone(),
        4,
        std::time::Duration::from_millis(5),
        32,
    );
```

- [ ] **Step 1.9: Update test file callers — `b1_p2_3b_3_admission_window.rs`**

Open `ironmlx/tests/b1_p2_3b_3_admission_window.rs`. Find every call to `spawn_scheduler_actor(model, b_max)` (or `model.clone(), N`) and append two args. Same defaults: deadline=5ms, queue_max=32.

```bash
grep -n "spawn_scheduler_actor" /Volumes/Dev/cxx-mlx/ironmlx/tests/b1_p2_3b_3_admission_window.rs
```

For each match, replace e.g.:

```rust
let handle = spawn_scheduler_actor(model.clone(), 2);
```

With:

```rust
let handle = spawn_scheduler_actor(
    model.clone(),
    2,
    std::time::Duration::from_millis(5),
    32,
);
```

If `std::time::Duration` is not in scope at that file's top, add `use std::time::Duration;` near the existing `use` imports.

- [ ] **Step 1.10: Update `b1_p2_3b_4_anthropic_actor.rs`, `b1_p2_3c_3_continuous_batching.rs`, `b1_p2_4_batched_vl.rs`**

For each of these three files, do the same pattern as Step 1.9:

```bash
for f in b1_p2_3b_4_anthropic_actor.rs b1_p2_3c_3_continuous_batching.rs b1_p2_4_batched_vl.rs; do
    grep -n "spawn_scheduler_actor" "/Volumes/Dev/cxx-mlx/ironmlx/tests/$f"
done
```

For every call site, append `, std::time::Duration::from_millis(5), 32` (and ensure `Duration` is imported).

- [ ] **Step 1.11: Verify the workspace compiles**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release 2>&1 | tail -10
MLX_DIR=$HOME/.local/mlx cargo +stable build --tests 2>&1 | tail -10
```

Both must succeed cleanly. Common failures:
- Missing `Duration` import in a test file → add `use std::time::Duration;`
- `clippy::too_many_arguments` on `driver_loop` or `drain_window` → already suppressed via `#[allow(clippy::too_many_arguments)]`
- `enqueue_or_reject` / `drain_admission_queue` referenced before defined → ensure they're added in the file (Step 1.4)

- [ ] **Step 1.12: Run the affected test suites (regression check at signature level only)**

The new functionality isn't covered by tests yet (that's T2 / T5), but the existing suites must still pass to prove the signature change is benign:

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib core::server -- --test-threads=1 2>&1 | tail -10
```

Expected: same number of lib tests passing as before T1 (no regressions). The driver-shutdown lib test in `scheduler_actor.rs` (cfg(test) `driver_shuts_down_when_cmd_channel_closes`) does not call `spawn_scheduler_actor` directly — should remain passing.

Integration regression (existing 3b-3 / 3c-3) belongs to T5 (full sweep); skip here.

- [ ] **Step 1.13: Hygiene gate**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All three must pass.

- [ ] **Step 1.14: Commit**

```bash
git add ironmlx/src/core/server/scheduler_actor.rs \
        ironmlx/src/core/server/mod.rs \
        ironmlx/tests/b1_p2_3b_3_admission_window.rs \
        ironmlx/tests/b1_p2_3b_4_anthropic_actor.rs \
        ironmlx/tests/b1_p2_3c_3_continuous_batching.rs \
        ironmlx/tests/b1_p2_4_batched_vl.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3d-t1): driver_loop admission queue + signature extension

Replaces c>b_max immediate-reject behavior with a bounded FIFO
admission queue (VecDeque<PendingAdmit>) inside driver_loop. Queue is
push-on-saturate (4 admission paths: outer first / drain_window /
rolling Admit / post-gc-drain) and drain-on-slot-free (drain happens
after every gc_finished_rows + at end of decode-loop iterations). Spec
§9 R1: when active_count==0 with queue non-empty, evict_all + admit
queue head + treat as new batch within rolling loop (not exiting to
outer).

`spawn_scheduler_actor` signature extended:
  (model, b_max, admission_deadline, admission_queue_max) -> Handle
SchedulerActorHandle adds queue_depth_peak / queue_rejected counters.
ADMISSION_DEADLINE const removed (now a parameter).

server/mod.rs preserves pre-3d behavior via hardcoded defaults
(b_max=4, 5ms, queue_max=32). T3 routes these from ServeArgs.

All existing spawn_scheduler_actor test callers (3b-3 / 3b-4 / 3c-3 /
b1_p2_4) updated to the new signature with the same defaults.

Spec ref: §4.2, §4.3 (4 paths), §4.6 (counters), §9 R1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Unit tests for admission queue push + overflow

**Why this is second:** T1 already wired the counters + structures; T2 only adds focused lib tests that exercise the queue push/drain/overflow paths via the SchedulerActor API. These tests use a stub model loaded from the standard fixture, gated by `#[ignore]` (real-model heavy, matching the existing test convention in this module).

**Files:**
- Modify: `ironmlx/src/core/server/scheduler_actor.rs` (cfg(test) module — append 2 tests)

### Steps

- [ ] **Step 2.1: Write the first failing test — `admission_queue_push_when_full`**

Append to the existing cfg(test) `mod tests` block in `ironmlx/src/core/server/scheduler_actor.rs` (after the existing `driver_shuts_down_when_cmd_channel_closes` test):

```rust
    /// Integration-ish lib test: b_max=1 + queue_max=2; admit 3 short
    /// requests in rapid succession; verify the queue grows to peak >= 2
    /// before slots free up. Real-model heavy — gated by `#[ignore]` and
    /// requires `IRONMLX_MODEL_DIR` or the standard ~/.ironmlx fixture path.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore] // real-model heavy: loads Qwen3.5-4B-MLX-4bit
    async fn admission_queue_push_when_full() {
        use crate::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
        use crate::core::sampler::Sampler;
        use crate::core::{Loader, Tokenizer};
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        use tokio::sync::Mutex;

        let model_dir = std::env::var("IRONMLX_MODEL_DIR").unwrap_or_else(|_| {
            let glob = format!(
                "{}/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
                std::env::var("HOME").unwrap()
            );
            std::fs::read_dir(&glob)
                .expect("snapshots dir")
                .filter_map(|e| e.ok())
                .next()
                .expect("snapshot")
                .path()
                .to_string_lossy()
                .into_owned()
        });
        let loader = Loader::open_multimodal(std::path::Path::new(&model_dir)).unwrap();
        let tokenizer = Tokenizer::from_loader(&loader).unwrap();
        let model = Arc::new(Mutex::new(crate::models::Qwen35Model::from_loader(&loader).unwrap()));

        let handle = spawn_scheduler_actor(
            model.clone(),
            /* b_max */ 1,
            /* admission_deadline */ Duration::from_millis(5),
            /* admission_queue_max */ 2,
        );

        // Build 3 small requests (8-token max_new keeps the test under ~30s).
        let mk_req = |text: &str| -> GenerateRequest {
            let msgs = vec![crate::core::Message {
                role: "user".into(),
                content: text.into(),
            }];
            let kw = serde_json::json!({"enable_thinking": false});
            let rendered = tokenizer.apply_chat_template(&msgs, true, Some(&kw)).unwrap();
            let prompt_ids = tokenizer.encode(&rendered, false).unwrap();
            GenerateRequest {
                prompt_ids,
                max_new_tokens: 8,
                sampler: Sampler::greedy(),
                stop_token_ids: tokenizer.eos_token_ids().to_vec(),
                prefill_chunk_size: 0,
                pixel_values: None,
                image_grid_thw: None,
                image_spatial_merge_size: 2,
                image_token_id: IMAGE_TOKEN_ID,
            }
        };

        // Submit 3 requests as fast as possible — back-to-back send.
        let mut replies = Vec::new();
        for text in ["Hello", "World", "Goodbye"] {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            handle
                .cmd_tx
                .send(SchedulerCommand::Admit {
                    request: mk_req(text),
                    reply_tx,
                })
                .await
                .expect("cmd_tx.send");
            replies.push(reply_rx);
        }

        // Drain each reply + its event stream to completion.
        let mut counts = Vec::new();
        for rx in replies {
            let admit_reply = rx.await.expect("reply").expect("admit ok");
            let mut event_rx = admit_reply.event_rx;
            let mut n = 0;
            while let Some(ev) = event_rx.recv().await {
                n += 1;
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            counts.push(n);
        }

        // All three completed (≥ 1 event each, since prefill + decode emit ≥1).
        for c in &counts {
            assert!(*c >= 1, "expected ≥1 event per request, got {c}");
        }

        // Queue peak should be ≥ 1 (b_max=1 → second admit queues; third
        // either queues alongside or after second drains, but peak ≥ 1).
        let peak = handle.queue_depth_peak.load(Ordering::Relaxed);
        assert!(peak >= 1, "expected queue_depth_peak >= 1, got {peak}");

        // No rejections — queue_max=2, we admitted 3 total but only 2
        // could be queued at any moment; the drain interleave keeps things
        // from overflowing.
        let rejected = handle.queue_rejected.load(Ordering::Relaxed);
        assert_eq!(rejected, 0, "expected no rejections, got {rejected}");

        drop(handle);
    }
```

- [ ] **Step 2.2: Run to verify it fails (or passes if T1 wiring is correct)**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib \
    core::server::scheduler_actor::tests::admission_queue_push_when_full \
    -- --ignored --nocapture 2>&1 | tail -10
```

If T1 was correct, this should PASS first try. If it doesn't (e.g., queue_depth_peak stays 0), the queue push path in `driver_loop` isn't firing — debug T1.

- [ ] **Step 2.3: Write `admission_queue_overflow_returns_err`**

Append below the first test:

```rust
    /// b_max=1 + queue_max=1; send 3 admits back-to-back. The 3rd one
    /// must be rejected with Err("admission queue full") because the
    /// queue (capacity 1) is full and the active slot is still busy
    /// when the 3rd command arrives.
    ///
    /// To guarantee the timing, we use a long-prompt request that takes
    /// long enough to prefill that all three commands queue before the
    /// first completes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore] // real-model heavy
    async fn admission_queue_overflow_returns_err() {
        use crate::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
        use crate::core::sampler::Sampler;
        use crate::core::{Loader, Tokenizer};
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        use tokio::sync::Mutex;

        let model_dir = std::env::var("IRONMLX_MODEL_DIR").unwrap_or_else(|_| {
            let glob = format!(
                "{}/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
                std::env::var("HOME").unwrap()
            );
            std::fs::read_dir(&glob)
                .expect("snapshots dir")
                .filter_map(|e| e.ok())
                .next()
                .expect("snapshot")
                .path()
                .to_string_lossy()
                .into_owned()
        });
        let loader = Loader::open_multimodal(std::path::Path::new(&model_dir)).unwrap();
        let tokenizer = Tokenizer::from_loader(&loader).unwrap();
        let model = Arc::new(Mutex::new(crate::models::Qwen35Model::from_loader(&loader).unwrap()));

        let handle = spawn_scheduler_actor(
            model.clone(),
            /* b_max */ 1,
            /* admission_deadline */ Duration::from_millis(5),
            /* admission_queue_max */ 1,
        );

        let mk_req = |text: &str, max_new: usize| -> GenerateRequest {
            let msgs = vec![crate::core::Message {
                role: "user".into(),
                content: text.into(),
            }];
            let kw = serde_json::json!({"enable_thinking": false});
            let rendered = tokenizer.apply_chat_template(&msgs, true, Some(&kw)).unwrap();
            let prompt_ids = tokenizer.encode(&rendered, false).unwrap();
            GenerateRequest {
                prompt_ids,
                max_new_tokens: max_new,
                sampler: Sampler::greedy(),
                stop_token_ids: tokenizer.eos_token_ids().to_vec(),
                prefill_chunk_size: 0,
                pixel_values: None,
                image_grid_thw: None,
                image_spatial_merge_size: 2,
                image_token_id: IMAGE_TOKEN_ID,
            }
        };

        // First admit takes longest so it stays busy while #2/#3 arrive.
        let (tx1, rx1) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req("Hello", 64),
                reply_tx: tx1,
            })
            .await
            .unwrap();

        // Wait briefly so the first admit enters Decoding before #2 arrives.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Now send #2 (will queue) and #3 (queue full → reject).
        let (tx2, rx2) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req("World", 8),
                reply_tx: tx2,
            })
            .await
            .unwrap();

        let (tx3, rx3) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req("Goodbye", 8),
                reply_tx: tx3,
            })
            .await
            .unwrap();

        // #3 must reject. Wait up to 5s for the reject reply.
        let reply3 = tokio::time::timeout(Duration::from_secs(5), rx3)
            .await
            .expect("rx3 timeout")
            .expect("rx3 recv");
        match reply3 {
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("admission queue full"),
                    "expected 'admission queue full' Err, got: {msg}"
                );
            }
            Ok(_) => panic!("expected Err for #3, got Ok"),
        }

        // queue_rejected counter must be ≥ 1.
        let rejected = handle.queue_rejected.load(Ordering::Relaxed);
        assert!(rejected >= 1, "expected queue_rejected ≥ 1, got {rejected}");

        // #1 + #2 should still complete normally — drain rx1 + rx2.
        let _ = tokio::time::timeout(Duration::from_secs(120), async {
            let r1 = rx1.await.unwrap().unwrap();
            let mut e1 = r1.event_rx;
            while let Some(ev) = e1.recv().await {
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            let r2 = rx2.await.unwrap().unwrap();
            let mut e2 = r2.event_rx;
            while let Some(ev) = e2.recv().await {
                if ev.finish_reason.is_some() {
                    break;
                }
            }
        })
        .await
        .expect("rx1/rx2 drain timeout");

        drop(handle);
    }
```

- [ ] **Step 2.4: Run both tests + verify PASS**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib \
    core::server::scheduler_actor::tests:: -- --ignored --test-threads=1 --nocapture 2>&1 | tail -20
```

Expected: 2 PASS (3 total in the module including the pre-existing `driver_shuts_down_when_cmd_channel_closes`).

If FAIL on `admission_queue_overflow_returns_err`: likely timing race — the 50ms sleep before #2 send may be insufficient. Bump to 100ms if needed.

- [ ] **Step 2.5: Hygiene gate**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All three PASS.

- [ ] **Step 2.6: Commit**

```bash
git add ironmlx/src/core/server/scheduler_actor.rs
git commit -m "$(cat <<'EOF'
test(b1-p2.3d-t2): admission queue push + overflow unit tests

Two #[ignore]'d real-model lib tests in scheduler_actor::tests:
- admission_queue_push_when_full: b_max=1 + queue_max=2 + 3 admits.
  Verifies queue_depth_peak >= 1 (at least one admit queued) and
  queue_rejected == 0 (queue capacity not exceeded). All 3 complete.
- admission_queue_overflow_returns_err: b_max=1 + queue_max=1 + 3 admits.
  Verifies 3rd admit Err contains "admission queue full" and
  queue_rejected counter increments.

Spec ref: §4.6 (counters), §9 (queue overflow path).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CLI flags + AppState plumbing

**Why this is third:** T1 hardcoded the defaults in `server/mod.rs`. T3 replaces those with CLI-routed values from `ServeArgs` → `AppState` → `spawn_scheduler_actor`.

**Files:**
- Modify: `ironmlx/src/cli/serve.rs` (ServeArgs + run fn)
- Modify: `ironmlx/src/core/server/mod.rs` (AppState + serve fn)

### Steps

- [ ] **Step 3.1: Extend `ServeArgs` with 3 new CLI flags**

Edit `ironmlx/src/cli/serve.rs` — the `pub struct ServeArgs` block (lines 13–33). Append 3 fields before the closing `}`:

```rust
#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Local directory containing config.json + model.safetensors + tokenizer.json.
    /// HF repo-id resolution is deferred to a future phase; pass a local path for now.
    #[arg(long)]
    pub model: String,

    /// Bind port.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Prefill chunk size — max tokens per prefill forward call. `0`
    /// disables chunking (single-shot forward over the whole prompt).
    /// Intermediate chunks update the cache only; the last chunk runs
    /// the full forward + lm_head.
    #[arg(long, default_value_t = 2048)]
    pub prefill_chunk_size: usize,

    /// Maximum concurrent in-flight requests (Scheduler slot count).
    /// Requests beyond this limit go to the admission queue.
    #[arg(long, default_value_t = 4)]
    pub b_max: usize,

    /// Admission-window deadline in milliseconds. After the first
    /// admit in a batch arrives, additional admits are absorbed until
    /// this deadline expires or the batch saturates at b_max.
    #[arg(long, default_value_t = 5)]
    pub admission_deadline_ms: u64,

    /// Capacity of the FIFO admission queue. Requests received while
    /// the scheduler is saturated are parked here. `0` disables queueing
    /// (immediate Err on saturation — mirrors pre-3d behavior).
    #[arg(long, default_value_t = 32)]
    pub admission_queue_max: usize,
}
```

- [ ] **Step 3.2: Update `run` fn to pass the new args**

In the same file, locate `pub fn run(args: ServeArgs) -> Result<()>` (around line 35). Find the `server::serve(...)` call (around line 56) and extend the arg list:

```rust
    runtime.block_on(server::serve(
        model,
        tokenizer,
        model_id,
        &args.host,
        args.port,
        args.prefill_chunk_size,
        args.b_max,
        args.admission_deadline_ms,
        args.admission_queue_max,
    ))
```

- [ ] **Step 3.3: Extend `AppState` in `core/server/mod.rs`**

Edit `ironmlx/src/core/server/mod.rs` — the `pub struct AppState` block (lines 29–40). Append 3 fields before the closing `}`:

```rust
#[derive(Clone)]
pub struct AppState {
    pub model: Arc<Mutex<Qwen35Model>>,
    pub tokenizer: Arc<Tokenizer>,
    pub model_id: String,
    pub prefill_chunk_size: usize,
    pub scheduler_handle: scheduler_actor::SchedulerActorHandle,
    /// Maximum concurrent in-flight requests routed to the SchedulerActor.
    pub b_max: usize,
    /// Admission-window deadline (milliseconds) — drain-window timeout.
    pub admission_deadline_ms: u64,
    /// FIFO admission queue capacity.
    pub admission_queue_max: usize,
}
```

- [ ] **Step 3.4: Update `serve` fn signature + body**

Replace the `pub async fn serve(...)` block in `core/server/mod.rs` (lines 42–77) with:

```rust
#[allow(clippy::too_many_arguments)]
pub async fn serve(
    model: Qwen35Model,
    tokenizer: Tokenizer,
    model_id: String,
    host: &str,
    port: u16,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
) -> Result<()> {
    let model = Arc::new(Mutex::new(model));
    let admission_deadline = std::time::Duration::from_millis(admission_deadline_ms);
    let scheduler_handle = scheduler_actor::spawn_scheduler_actor(
        model.clone(),
        b_max,
        admission_deadline,
        admission_queue_max,
    );
    let state = AppState {
        model,
        tokenizer: Arc::new(tokenizer),
        model_id,
        prefill_chunk_size,
        scheduler_handle,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
    };
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/messages", post(anthropic::messages))
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("parsing socket addr {host}:{port}"))?;
    tracing::info!("ironmlx server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

- [ ] **Step 3.5: Verify CLI help renders the new flags**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
./target/release/ironmlx serve --help 2>&1 | grep -E "b-max|admission-deadline-ms|admission-queue-max"
```

Expected: 3 lines, one per new flag, each with default value shown.

- [ ] **Step 3.6: Build + regression run on existing tests (callers should still compile)**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib core::server -- --test-threads=1 2>&1 | tail -10
```

Both must succeed.

- [ ] **Step 3.7: Hygiene gate**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
```

Both must pass.

- [ ] **Step 3.8: Commit**

```bash
git add ironmlx/src/cli/serve.rs ironmlx/src/core/server/mod.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3d-t3): CLI flags + AppState plumbing for b_max/deadline/queue

Adds three ServeArgs flags:
  --b-max (default 4)
  --admission-deadline-ms (default 5)
  --admission-queue-max (default 32)

Routes them through serve() → spawn_scheduler_actor and stores them in
AppState (3 new fields). Defaults preserve pre-3d behavior exactly.

Spec ref: §4.5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: HTTP 503 differentiation for admission queue overflow

**Why this is fourth:** T1's `enqueue_or_reject` produces an Err whose message contains `"admission queue full"`. T4 teaches both HTTP handlers (openai + anthropic) to map that specific message to HTTP 503 (Service Unavailable) with a `Retry-After` header, while all other Err variants remain HTTP 400 (Bad Request). Spec §10 / §9 R3 flags string-match as fragile — a typed `SchedulerError` enum is deferred to 3e/3.5.

**Files:**
- Modify: `ironmlx/src/core/server/openai.rs` (serve_via_scheduler_unary + serve_via_scheduler_stream)
- Modify: `ironmlx/src/core/server/anthropic.rs` (analogous handlers)

### Steps

- [ ] **Step 4.1: Locate the OpenAI Err response sites**

```bash
grep -n "reply.*Err\|Err(e)\|status(StatusCode" /Volumes/Dev/cxx-mlx/ironmlx/src/core/server/openai.rs | head -25
```

Identify the lines where `serve_via_scheduler_unary` / `serve_via_scheduler_stream` propagate an admit Err to the HTTP response. These will be `match` arms on the `rx.await` reply or on `reply.expect(...)` failures.

- [ ] **Step 4.2: Add a small helper at the top of `openai.rs`**

Add inside the existing imports/private fns area in `ironmlx/src/core/server/openai.rs`:

```rust
/// Map a SchedulerActor admit Err into an HTTP response. Spec §4.7:
/// "admission queue full" → 503 + Retry-After: 5; everything else → 400.
fn admit_err_to_response(err: anyhow::Error) -> axum::response::Response {
    use axum::http::{header, HeaderValue, StatusCode};
    use axum::response::{IntoResponse, Response};
    let msg = format!("{err:#}");
    if msg.contains("admission queue full") {
        let mut resp: Response = (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
        resp.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
        resp
    } else {
        (StatusCode::BAD_REQUEST, msg).into_response()
    }
}
```

If `axum::http::header::RETRY_AFTER` isn't in your axum version, fall back to `HeaderName::from_static("retry-after")`.

- [ ] **Step 4.3: Wire the helper into `serve_via_scheduler_unary`**

In `serve_via_scheduler_unary` (search for `fn serve_via_scheduler_unary`), find the place that handles the admit `Err` reply from `cmd_tx`. The block typically looks like:

```rust
match reply {
    Ok(admit_reply) => { /* stream / aggregate path */ }
    Err(e) => {
        return format_error_response(StatusCode::BAD_REQUEST, &format!("admit failed: {e:#}"));
    }
}
```

Replace the `Err(e)` arm with:

```rust
    Err(e) => {
        return admit_err_to_response(e);
    }
```

Repeat for `serve_via_scheduler_stream` if there's a similar Err handling — find with `grep -n "admit_reply\|admit failed" /Volumes/Dev/cxx-mlx/ironmlx/src/core/server/openai.rs`. Wire each Err-arm to `admit_err_to_response(e)`.

- [ ] **Step 4.4: Same change in `anthropic.rs`**

Mirror Steps 4.2 + 4.3 in `ironmlx/src/core/server/anthropic.rs`. The helper can be a local fn (or move both `admit_err_to_response` versions to a shared `server/util.rs` if you prefer DRY — keeping per-handler copies is fine for 3d to avoid scope creep).

- [ ] **Step 4.5: Add a unit test for the helper**

Append to a cfg(test) module in `ironmlx/src/core/server/openai.rs` (create the module if it doesn't exist):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    #[tokio::test]
    async fn admit_err_503_for_queue_full() {
        let err = anyhow::anyhow!("admission queue full: capacity=32 reached");
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let retry = resp.headers().get("retry-after").expect("retry-after header");
        assert_eq!(retry.to_str().unwrap(), "5");
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("admission queue full"));
    }

    #[tokio::test]
    async fn admit_err_400_for_other() {
        let err = anyhow::anyhow!("prompt too long: 999999 tokens exceeds limit");
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(resp.headers().get("retry-after").is_none());
    }
}
```

- [ ] **Step 4.6: Run the new unit tests**

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --lib \
    core::server::openai::tests -- --test-threads=1 2>&1 | tail -10
```

Expected: 2 PASS.

- [ ] **Step 4.7: Hygiene gate**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All three PASS.

- [ ] **Step 4.8: Commit**

```bash
git add ironmlx/src/core/server/openai.rs ironmlx/src/core/server/anthropic.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3d-t4): HTTP 503 differentiation for admission queue overflow

Adds `admit_err_to_response` helper in both OpenAI + Anthropic
handlers that maps an admit reply Err containing "admission queue
full" to HTTP 503 Service Unavailable + Retry-After: 5 header. All
other admit Err variants continue to map to HTTP 400 (Bad Request).

Spec §4.7 / §9 R3: string-match is acknowledged-fragile; a typed
SchedulerError enum is deferred to 3e/3.5.

Two unit tests cover both the 503 (queue-full message) and 400 (other
Err) paths.

Spec ref: §4.7, §9 R3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Integration scenarios + 12-suite regression sweep + close-out

**Why this is fifth:** Final acceptance. Five integration scenarios drive the queue end-to-end through the HTTP layer (S2 + S5) and through the SchedulerActor directly (S1 + S3 + S4). Regression sweep validates pre-3d behavior is preserved under default config.

**Files:**
- Create: `ironmlx/tests/b1_p2_3d_admission_queue.rs` (~400 LoC, 5 scenarios)
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3d_closeout/report.md` (gitignored — `-f`)
- Audit + possibly modify: integration tests that asserted on the old "scheduler full" Err — spec §9 R5

### Steps

- [ ] **Step 5.1: Audit existing tests for pre-3d "scheduler full" assertions**

```bash
grep -rn "scheduler full\|400 Bad Request" /Volumes/Dev/cxx-mlx/ironmlx/tests/ 2>&1 | head -20
```

For each match:
- If the test asserts that an admit returns `Err(...scheduler full...)`, that assertion is now stale (queue absorbs the admit). Update the assertion: either (a) reconfigure the test to use `admission_queue_max = 0` (queue disabled — restores immediate-reject behavior) or (b) reset the assertion to verify the new path (queue draws success or 503).

Most likely site: `tests/b1_p2_3b_3_admission_window.rs` if it has any "queue full → expect Err" scenario. For 3d we'd configure those affected tests with `admission_queue_max = 0` (Step 1.10 used 32 — adjust per-test).

```bash
# Example sed for one file (verify match list first):
# sed -i.bak 's/, 32)$/, 0)/' tests/b1_p2_3b_3_admission_window.rs   # ONLY if that test relied on immediate reject
```

If no test depends on immediate reject, leave queue_max=32 in T1 callers — no further change needed here.

- [ ] **Step 5.2: Scaffold `tests/b1_p2_3d_admission_queue.rs`**

Create the file with the standard fixture loader + helpers:

```rust
//! B1-p2.3d integration scenarios for admission queue + config exposure.
//!
//! Scenarios drive `spawn_scheduler_actor` directly (S1/S3/S4) and via
//! the HTTP server bound to a random localhost port (S2/S5).
//!
//! Reference fixtures: `tests/fixtures/p6_qwen35_vl/multi_image/` (unused
//! here — text-only suite).

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{spawn_scheduler_actor, SchedulerCommand};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::Qwen35Model;
use tokio::sync::Mutex;

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN35_MODEL") {
        return PathBuf::from(p);
    }
    let glob = format!(
        "{}/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
        std::env::var("HOME").unwrap()
    );
    std::fs::read_dir(&glob)
        .expect("snapshots dir")
        .filter_map(|e| e.ok())
        .next()
        .expect("snapshot")
        .path()
}

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let p = model_path();
    let loader = Loader::open_multimodal(&p).expect("Loader::open_multimodal");
    let tok = Tokenizer::from_loader(&loader).expect("tokenizer");
    let model = Qwen35Model::from_loader(&loader).expect("model");
    (Arc::new(Mutex::new(model)), Arc::new(tok))
}

fn make_req(tokenizer: &Tokenizer, text: &str, max_new: usize) -> GenerateRequest {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer.apply_chat_template(&msgs, true, Some(&kw)).unwrap();
    let prompt_ids = tokenizer.encode(&rendered, false).unwrap();
    GenerateRequest {
        prompt_ids,
        max_new_tokens: max_new,
        sampler: Sampler::greedy(),
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: 0,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
    }
}
```

- [ ] **Step 5.3: Write S1 — `queue_drains_fifo_at_bmax2_c4`**

Append to the file:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy: needs QWEN35_MODEL
async fn queue_drains_fifo_at_bmax2_c4() {
    // b_max=2, queue_max=8; submit 4 requests back-to-back. All 4 must
    // complete; queue_depth_peak >= 2 (2 had to queue).
    let (model, tokenizer) = load_fixture();
    let handle = spawn_scheduler_actor(
        model.clone(),
        2,
        Duration::from_millis(5),
        8,
    );

    let texts = ["Hello", "World", "Goodbye", "Farewell"];
    let mut replies = Vec::new();
    for t in texts {
        let (tx, rx) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: make_req(&tokenizer, t, 8),
                reply_tx: tx,
            })
            .await
            .unwrap();
        replies.push(rx);
    }

    // Drain all 4 — each must reach a finish_reason.
    let mut finishes = 0;
    for rx in replies {
        let reply = rx.await.expect("rx").expect("admit ok");
        let mut event_rx = reply.event_rx;
        while let Some(ev) = event_rx.recv().await {
            if ev.finish_reason.is_some() {
                finishes += 1;
                break;
            }
        }
    }
    assert_eq!(finishes, 4, "expected 4 finishes, got {finishes}");

    let peak = handle.queue_depth_peak.load(Ordering::Relaxed);
    assert!(peak >= 2, "expected queue_depth_peak >= 2, got {peak}");
    let rejected = handle.queue_rejected.load(Ordering::Relaxed);
    assert_eq!(rejected, 0, "expected zero rejections, got {rejected}");

    drop(handle);
}
```

- [ ] **Step 5.4: Write S2 — `queue_overflow_returns_err_via_actor` (actor-level 503 antecedent)**

Append:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy
async fn queue_overflow_returns_err_via_actor() {
    // b_max=2, queue_max=3; submit 6 requests. The first 5 succeed (2 in
    // batch + 3 queued); the 6th must get Err("admission queue full").
    let (model, tokenizer) = load_fixture();
    let handle = spawn_scheduler_actor(
        model.clone(),
        2,
        Duration::from_millis(5),
        3,
    );

    // Submit a long #1 first so it dominates the active slot.
    let (tx1, rx1) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_req(&tokenizer, "Hello", 64),
            reply_tx: tx1,
        })
        .await
        .unwrap();
    let (tx2, rx2) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_req(&tokenizer, "World", 64),
            reply_tx: tx2,
        })
        .await
        .unwrap();

    // Allow the first batch to enter Decoding so the next 4 cmd_tx pushes
    // hit the saturated path.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 4 more requests; the first 3 should queue, the 4th overflow.
    let mut later_rxs = Vec::new();
    for t in ["A", "B", "C", "D"] {
        let (tx, rx) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: make_req(&tokenizer, t, 8),
                reply_tx: tx,
            })
            .await
            .unwrap();
        later_rxs.push(rx);
    }

    // The 4th of these (= 6th overall) must reject.
    let last_reply = tokio::time::timeout(Duration::from_secs(5), later_rxs.pop().unwrap())
        .await
        .expect("last_reply timeout")
        .expect("oneshot");
    match last_reply {
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("admission queue full"),
                "expected admission queue full, got: {msg}"
            );
        }
        Ok(_) => panic!("expected Err for 6th admit, got Ok"),
    }

    // First 5 should still complete (rx1, rx2, + first 3 of later_rxs).
    let _ = tokio::time::timeout(Duration::from_secs(300), async {
        for rx in std::iter::once(rx1).chain(std::iter::once(rx2)).chain(later_rxs.into_iter()) {
            let r = rx.await.unwrap().unwrap();
            let mut e = r.event_rx;
            while let Some(ev) = e.recv().await {
                if ev.finish_reason.is_some() {
                    break;
                }
            }
        }
    })
    .await
    .expect("first 5 drain timeout");

    let rejected = handle.queue_rejected.load(Ordering::Relaxed);
    assert!(rejected >= 1, "expected queue_rejected >= 1, got {rejected}");

    drop(handle);
}
```

- [ ] **Step 5.5: Write S3 — `admission_deadline_config_observed`**

Append:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy
async fn admission_deadline_config_observed() {
    // admission_deadline_ms = 30 (vs. default 5). Two admits arriving 20ms
    // apart should land in the same batch (drain_window covers both).
    // batch_count should be 1 (not 2).
    let (model, tokenizer) = load_fixture();
    let handle = spawn_scheduler_actor(
        model.clone(),
        4,
        Duration::from_millis(30),
        32,
    );

    let batch_before = handle.batch_count.load(Ordering::Relaxed);

    let (tx1, rx1) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_req(&tokenizer, "first", 5),
            reply_tx: tx1,
        })
        .await
        .unwrap();

    // Sleep 20ms — still within the 30ms admission window. The driver_loop
    // has issued the deadline timer; the second admit lands while the
    // first batch is still in the drain_window.
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (tx2, rx2) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: make_req(&tokenizer, "second", 5),
            reply_tx: tx2,
        })
        .await
        .unwrap();

    // Drain both replies.
    let r1 = rx1.await.unwrap().unwrap();
    let r2 = rx2.await.unwrap().unwrap();
    for mut rx in [r1.event_rx, r2.event_rx] {
        while let Some(ev) = rx.recv().await {
            if ev.finish_reason.is_some() {
                break;
            }
        }
    }

    let batch_delta = handle.batch_count.load(Ordering::Relaxed) - batch_before;
    assert_eq!(
        batch_delta, 1,
        "expected single batch (deadline=30ms covers both admits), got {batch_delta}"
    );

    drop(handle);
}
```

- [ ] **Step 5.6: Write S4 — `b_max_config_8_no_queue`**

Append:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore] // real-model heavy
async fn b_max_config_8_no_queue() {
    // b_max=8 + admission_deadline_ms=50: 6 concurrent admits all fit in
    // one batch (queue stays empty).
    let (model, tokenizer) = load_fixture();
    let handle = spawn_scheduler_actor(
        model.clone(),
        8,
        Duration::from_millis(50),
        32,
    );

    let texts = ["a", "b", "c", "d", "e", "f"];
    let mut rxs = Vec::new();
    for t in texts {
        let (tx, rx) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: make_req(&tokenizer, t, 5),
                reply_tx: tx,
            })
            .await
            .unwrap();
        rxs.push(rx);
    }

    for rx in rxs {
        let r = rx.await.unwrap().unwrap();
        let mut e = r.event_rx;
        while let Some(ev) = e.recv().await {
            if ev.finish_reason.is_some() {
                break;
            }
        }
    }

    let peak = handle.queue_depth_peak.load(Ordering::Relaxed);
    assert_eq!(peak, 0, "expected queue_depth_peak == 0 (b_max=8 absorbs 6 admits), got {peak}");

    drop(handle);
}
```

- [ ] **Step 5.7: Write S5 — `iron_bench_c8_with_queue_no_4xx`**

This scenario boots the HTTP server with default 3d config and runs iron-bench v2 against it for 15s at c=8. It validates the end-to-end queue path under realistic load — c > b_max no longer produces HTTP 4xx.

Append:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore] // real-model heavy + HTTP server
async fn iron_bench_c8_with_queue_no_4xx() {
    // Boot the server on a random port; spawn 8 concurrent HTTP clients
    // hitting /v1/chat/completions for 15s. With b_max=4 + queue_max=32,
    // no HTTP 4xx should occur.
    use ironmlx::core::server;

    let port = 18400 + (std::process::id() % 1000) as u16;
    let model_path = model_path();
    let loader = Loader::open_multimodal(&model_path).unwrap();
    let tokenizer_for_serve = Tokenizer::from_loader(&loader).unwrap();
    let model_for_serve = Qwen35Model::from_loader(&loader).unwrap();

    let server_handle = tokio::spawn(async move {
        server::serve(
            model_for_serve,
            tokenizer_for_serve,
            "qwen35".to_string(),
            "127.0.0.1",
            port,
            2048, // prefill_chunk_size default
            4,    // b_max
            5,    // admission_deadline_ms
            32,   // admission_queue_max
        )
        .await
    });

    // Wait for server to bind.
    tokio::time::sleep(Duration::from_secs(2)).await;

    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();

    // 8 concurrent workers, each looping for 15 seconds.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    let mut handles = Vec::new();
    for worker_id in 0..8 {
        let client = client.clone();
        let url = url.clone();
        handles.push(tokio::spawn(async move {
            let mut ok = 0usize;
            let mut errs: Vec<u16> = Vec::new();
            while tokio::time::Instant::now() < deadline {
                let body = serde_json::json!({
                    "model": "qwen35",
                    "messages": [
                        {"role": "user", "content": format!("hi from worker {worker_id}")}
                    ],
                    "max_tokens": 8,
                });
                let resp = match client.post(&url).json(&body).send().await {
                    Ok(r) => r,
                    Err(_) => {
                        errs.push(0); // connection error
                        continue;
                    }
                };
                let status = resp.status().as_u16();
                if status == 200 {
                    let _ = resp.bytes().await;
                    ok += 1;
                } else {
                    errs.push(status);
                }
            }
            (worker_id, ok, errs)
        }));
    }

    let mut total_ok = 0usize;
    let mut all_errs: Vec<u16> = Vec::new();
    for h in handles {
        let (worker_id, ok, errs) = h.await.unwrap();
        eprintln!("[S5] worker {worker_id}: ok={ok}, errs={errs:?}");
        total_ok += ok;
        all_errs.extend(errs);
    }

    // No 4xx allowed (would mean a request was rejected, not queued).
    // 5xx (503 from queue overflow) is also disallowed at queue_max=32
    // under c=8 b_max=4 — queue depth should never exceed 4 in this run.
    let four_xx: Vec<_> = all_errs.iter().filter(|s| **s >= 400 && **s < 500).collect();
    assert!(
        four_xx.is_empty(),
        "expected no 4xx, got: {four_xx:?}; total_ok={total_ok}"
    );
    let five_xx: Vec<_> = all_errs.iter().filter(|s| **s >= 500).collect();
    assert!(
        five_xx.is_empty(),
        "expected no 5xx at queue_max=32 c=8 b_max=4, got: {five_xx:?}"
    );

    assert!(total_ok > 0, "expected at least some successful responses, got 0");

    server_handle.abort();
}
```

This requires `reqwest` in dev-dependencies. Verify:

```bash
grep -E "^reqwest" /Volumes/Dev/cxx-mlx/ironmlx/Cargo.toml
```

If absent, add to `[dev-dependencies]`:

```toml
reqwest = { workspace = true, features = ["json"] }
```

If `reqwest` isn't in the workspace `Cargo.toml`'s `[workspace.dependencies]`, add it there first (same version as `iron-bench` uses) — check `cat /Volumes/Dev/cxx-mlx/Cargo.toml | grep reqwest`.

- [ ] **Step 5.8: Run all 5 scenarios + verify PASS**

```bash
QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1) \
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --test b1_p2_3d_admission_queue \
    -- --ignored --test-threads=1 --nocapture 2>&1 | tee /tmp/b1_p2_3d_scenarios.log | tail -50
```

All 5 should PASS. Capture timings.

If any fail, BLOCKED — report to controller with the specific scenario name + failure mode (e.g., `S2 last_reply was Ok not Err`). Do not silently relax the assertion.

- [ ] **Step 5.9: Hygiene gate**

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
```

All three PASS.

- [ ] **Step 5.10: 12-suite regression sweep**

```bash
MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)
echo "=== 12-suite regression sweep ($(date)) ===" > /tmp/b1_p2_3d_regression.log
for t in \
    p6_qwen35_vl_logits_match \
    p6_6_logits_match \
    p6_7_chunked_prefill \
    b1_p2_1_batched_prefill \
    b1_p2_2_batched_decode \
    b1_p2_3a_scheduler_skeleton \
    b1_p2_3b_1_scheduler_step \
    b1_p2_3b_2_scheduler_actor \
    b1_p2_3b_3_admission_window \
    b1_p2_3b_4_anthropic_actor \
    b1_p2_3c_1_per_row_offset \
    b1_p2_3c_2_scheduler_decode_mask \
    b1_p2_3c_3_continuous_batching \
    b1_p2_4_batched_vl
do
    echo "=== $t ===" >> /tmp/b1_p2_3d_regression.log
    start=$(date +%s)
    QWEN35_MODEL="$MODEL" MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --test "$t" \
        -- --ignored --test-threads=1 2>&1 | tail -6 >> /tmp/b1_p2_3d_regression.log
    end=$(date +%s)
    echo "elapsed: $((end - start))s" >> /tmp/b1_p2_3d_regression.log
done
echo "DONE $(date)" >> /tmp/b1_p2_3d_regression.log
```

Verify every suite passes:

```bash
grep -E "test result:|elapsed" /tmp/b1_p2_3d_regression.log | grep -v "0 passed\|0 measured" | tail -30
```

Expected: 14 suites each report `ok`. No `failed`.

Note: `b1_p2_3a_scheduler_skeleton` has tests that don't carry `#[ignore]`; running with `--ignored` filters them out. To cover it, also run:

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable test -p ironmlx --test b1_p2_3a_scheduler_skeleton -- --test-threads=1 2>&1 | tail -5
```

Append this result to the regression log.

- [ ] **Step 5.11: Write close-out report**

```bash
mkdir -p /Volumes/Dev/cxx-mlx/ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3d_closeout
```

Write `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3d_closeout/report.md`:

```markdown
# B1-p2.3d Admission Queue + Config Exposure — Close-out

**Branch:** `ironmlx-b1-p2-3d-admission-queue` (off `ironmlx-b1-p2-4-batched-vl` head `22bb8af`)
**Date:** 2026-05-XX  (replace with completion date)
**Status:** ✅ COMPLETE

## Summary

Replaces the c > b_max "scheduler full" HTTP 400 reject behavior with a
bounded FIFO admission queue inside `driver_loop`. Exposes `b_max`,
`admission_deadline_ms`, `admission_queue_max` as ServeArgs CLI flags
→ AppState fields. Queue overflow returns HTTP 503 + Retry-After: 5.

Decode path (`step_inner`, `build_per_row_decode_mask`,
`build_decode_position_ids`) UNCHANGED — queue lives entirely in
driver_loop.

Defaults preserve pre-3d behavior exactly: b_max=4, deadline=5ms,
queue_max=32.

## Acceptance

| Gate | Result |
| --- | --- |
| S1 queue_drains_fifo_at_bmax2_c4 (peak ≥ 2, all 4 finish, 0 rejects) | ✅ PASS |
| S2 queue_overflow_returns_err_via_actor (6th admit Err contains "admission queue full") | ✅ PASS |
| S3 admission_deadline_config_observed (deadline=30ms → 2 admits at 20ms gap = single batch) | ✅ PASS |
| S4 b_max_config_8_no_queue (b_max=8 + 6 admits → queue_depth_peak == 0) | ✅ PASS |
| S5 iron_bench_c8_with_queue_no_4xx (HTTP path c=8 d=15s → no 4xx/5xx) | ✅ PASS |
| Unit: admission_queue_push_when_full | ✅ PASS |
| Unit: admission_queue_overflow_returns_err | ✅ PASS |
| Unit: admit_err_to_response (503 + 400 paths) | ✅ PASS |
| fmt --check / clippy -D warnings / build --release | ✅ ALL CLEAN |

## Architectural changes per spec §4

| Item | File | Change |
| --- | --- | --- |
| §4.2 driver_loop admission_queue state | `core/server/scheduler_actor.rs` | Added `VecDeque<PendingAdmit>` |
| §4.3 4 admission paths | `core/server/scheduler_actor.rs` | outer first (no-op), drain_window saturate-push, rolling Admit push, post-gc drain |
| §4.5 config flow | `cli/serve.rs` → `core/server/mod.rs` → spawn_scheduler_actor | 3 fields propagated |
| §4.6 atomic counters | `core/server/scheduler_actor.rs` | `queue_depth_peak`, `queue_rejected` |
| §4.7 HTTP 503 differentiation | `core/server/openai.rs`, `anthropic.rs` | `admit_err_to_response` helper |
| §9 R1 Finished→Idle race | `driver_loop` `'rolling` end-of-iter | Queue-non-empty branch handled before evict_all-to-Idle |
| §3 NG1 (no preemption) | — | Preserved (active rows always run to completion) |

## Commits

(Fill in via `git log --oneline 22bb8af..HEAD`)

- T1: `<sha>` driver_loop admission queue + signature extension
- T2: `<sha>` admission queue push + overflow unit tests
- T3: `<sha>` CLI flags + AppState plumbing
- T4: `<sha>` HTTP 503 differentiation for queue overflow
- T5: `<sha>` integration scenarios + 14-suite regression + close-out

## Regression Status (default config)

Sweep run with `--ignored --test-threads=1` and default `b_max=4 /
deadline=5ms / queue_max=32`. Each suite must report `test result: ok`.

| Suite | Result | Time |
| --- | --- | --- |
| p6_qwen35_vl_logits_match | ✅ PASS | <FILL>s |
| p6_6_logits_match | ✅ PASS | <FILL>s |
| p6_7_chunked_prefill | ✅ PASS | <FILL>s |
| b1_p2_1_batched_prefill | ✅ PASS | <FILL>s |
| b1_p2_2_batched_decode | ✅ PASS | <FILL>s |
| b1_p2_3a_scheduler_skeleton (default mode) | ✅ PASS | <FILL>s |
| b1_p2_3b_1_scheduler_step | ✅ PASS | <FILL>s |
| b1_p2_3b_2_scheduler_actor | ✅ PASS | <FILL>s |
| b1_p2_3b_3_admission_window | ✅ PASS | <FILL>s |
| b1_p2_3b_4_anthropic_actor | ✅ PASS | <FILL>s |
| b1_p2_3c_1_per_row_offset | ✅ PASS | <FILL>s |
| b1_p2_3c_2_scheduler_decode_mask | ✅ PASS | <FILL>s |
| b1_p2_3c_3_continuous_batching | ✅ PASS | <FILL>s |
| b1_p2_4_batched_vl | ✅ PASS | <FILL>s |
| **B1-p2.3d admission queue (5 scenarios)** | **✅ PASS** | **<FILL>s** |

## Compat sunset

| Removed | Replaced with |
| --- | --- |
| `ADMISSION_DEADLINE` const in scheduler_actor.rs:38 | `admission_deadline` driver_loop parameter (CLI-driven) |
| Hardcoded `b_max=4` in server/mod.rs:54 | `--b-max` CLI flag, default 4 |
| `c > b_max` immediate Err → HTTP 400 | FIFO admission queue (push or HTTP 503 if queue full) |

## Notes / known limitations carrying forward to backlog

- **No preemption** (spec NG1) — active rows always run to completion. A
  long-running active row blocks queue drain. Future task.
- **No HTTP cancellation propagation** (spec NG2) — if HTTP client
  disconnects while admit is queued, oneshot send fails silently when the
  admit eventually drains; events stream into a dropped Receiver. 3e+.
- **String-match for "admission queue full" → 503** (spec §4.7 / §9 R3) —
  fragile. Future refactor to typed `SchedulerError` enum.
- **No persistence** (spec NG4) — queue is in-memory; cleared on restart.
- **No priority / SLA / fair-share** (spec NG3) — FIFO only.

## B1-p2 Next Steps

| Sub-spec | Scope | Status |
| --- | --- | --- |
| B1-p2.3c+ | Chunked admit_mid prefill + decode-interleave | Backlog |
| B1-p2.3d | **Admission queue + config exposure** | **✅ DONE (this report)** |
| B1-p2.3e | Per-row async sampler tuning + cancellation + typed SchedulerError | Backlog |
| B1-p2.4 | Batched VL serving | ✅ DONE |
| B1-p2.5 | Production hardening | Future |

After B1-p2.3d: c>b_max no longer reject. Next major program: Qwen3.5 MoE.

## Linked artifacts

- [B1-p2.3d design spec](../../../../../docs/superpowers/specs/2026-05-16-b1-p2-3d-admission-queue-design.md)
- [B1-p2.3d implementation plan](../../../../../docs/superpowers/plans/2026-05-16-b1-p2-3d-admission-queue.md)
- [3c-3 perf baseline](../b1_p2_3c_3_perf_baseline/report.md)
- [B1-p2.4 close-out](../b1_p2_4_closeout/report.md)
```

Replace `<FILL>` with actual timings from `/tmp/b1_p2_3d_regression.log` and commit SHAs.

- [ ] **Step 5.12: Stage + commit**

```bash
git add ironmlx/tests/b1_p2_3d_admission_queue.rs
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3d_closeout/report.md
# If you added reqwest to Cargo.toml:
git add ironmlx/Cargo.toml Cargo.toml 2>/dev/null || true

git commit -m "$(cat <<'EOF'
test+docs(b1-p2.3d-t5): integration scenarios + 14-suite regression + close-out

Adds tests/b1_p2_3d_admission_queue.rs with 5 #[ignore]'d integration
scenarios:
- S1 queue_drains_fifo_at_bmax2_c4 (peak queue depth + zero rejects)
- S2 queue_overflow_returns_err_via_actor (6th admit Err)
- S3 admission_deadline_config_observed (deadline=30ms config validation)
- S4 b_max_config_8_no_queue (b_max=8 absorbs c=6 with empty queue)
- S5 iron_bench_c8_with_queue_no_4xx (HTTP path, no 4xx under c=8 b_max=4)

14-suite regression sweep PASS with default config (b_max=4, 5ms,
queue_max=32). Close-out report committed under
ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3d_closeout/
(path is gitignored — committed with -f).

Spec ref: §7 (acceptance), §9 (risk mitigation verified).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5.13: Final verification**

```bash
git log --oneline ironmlx-b1-p2-4-batched-vl..HEAD
```

Expected: 5 commits, one per task. Update the close-out's "Commits" section with the actual SHAs and re-amend the report (re-commit as a doc-only follow-up if you've already committed the close-out — never amend a pushed commit).

```bash
MLX_DIR=$HOME/.local/mlx cargo +stable build --release
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
```

All clean. **B1-p2.3d done.**

---

## Self-review checklist

**Spec coverage (§4 items):**

- §4.2 driver_loop `admission_queue: VecDeque<PendingAdmit>` → T1 Step 1.4 + 1.7
- §4.3 4 admission paths (outer first / drain_window / rolling Admit / post-gc) → T1 Step 1.5 (outer + drain_window), 1.7 (rolling + post-gc drain)
- §4.4 `Scheduler::admit_mid` Err semantics unchanged → verified: T1 only changes call-sites in driver_loop; admit_mid stays as-is
- §4.5 config flow ServeArgs → AppState → spawn → driver_loop → T1 Step 1.3/1.8 (defaults) + T3 (CLI plumbing)
- §4.6 counters queue_depth_peak / queue_rejected → T1 Step 1.1 (handle fields) + 1.4 (counter updates in enqueue_or_reject)
- §4.7 HTTP 503 differentiation → T4 Step 4.2 (helper) + 4.3 (wire openai) + 4.4 (wire anthropic)

**Spec §3 NG1-NG6 (must NOT be implemented):**

- NG1 Preemption — not implemented; queued admits only run after active row finishes ✓
- NG2 HTTP cancellation propagation — not implemented; oneshot drop is silent ✓
- NG3 Priority / SLA — FIFO only (`VecDeque::push_back` / `pop_front`) ✓
- NG4 Persistence — in-memory `VecDeque`, cleared on driver shutdown ✓
- NG5 Dynamic b_max resize — set once at spawn ✓
- NG6 Typed SchedulerError — not implemented; string-match preserved with explicit §9 R3 risk acknowledgement ✓

**Spec §7 acceptance scenarios:**

- S1 → T5 Step 5.3
- S2 (via actor, not pure HTTP — acceptable test design) → T5 Step 5.4
- S3 admission_deadline_config → T5 Step 5.5
- S4 b_max_config → T5 Step 5.6
- S5 iron-bench-style end-to-end (uses reqwest directly inside the test, not iron-bench binary) → T5 Step 5.7

**Spec §9 R1-R7 mitigation:**

- R1 Finished→Idle race → T1 Step 1.7 (queue-non-empty branch BEFORE evict_all-to-Idle)
- R2 oneshot disconnect → noted as NG2 + tolerated by existing handle_admit_mid SendErr handling
- R3 string-match → T4 Step 4.2 + close-out limitation note
- R4 serve() signature breaking change → T3 Step 3.4 (only CLI caller updated; CLI flags backward-compatible via defaults)
- R5 pre-3d 400 assertion in tests → T5 Step 5.1 (audit) + T1 Steps 1.9/1.10 (caller updates)
- R6 queue_max=32 default too small under burst → spec accepts as tunable; close-out documents
- R7 drain loop livelock on admit_mid Err → handle_admit_mid's rollback (3c-3) preserved; queue drain doesn't re-enqueue on Err

**Placeholder scan:** No "TBD" / "TODO" / "implement later" in plan steps. Step 5.11's close-out template uses `<FILL>` placeholders for runtime values (timings + SHAs) — these are deliberate, filled in at execution time.

**Type consistency:**

- `PendingAdmit { request: GenerateRequest, reply_tx: oneshot::Sender<Result<AdmitReply>> }` defined T1 Step 1.2; used T1 Steps 1.4 + 1.7
- `enqueue_or_reject` signature `(cmd, queue, queue_max, peak, rejected)` defined T1 Step 1.4; called T1 Steps 1.6 + 1.7
- `drain_admission_queue` signature `(queue, sched, event_txs, admit_count, model, b_max)` defined T1 Step 1.4; called T1 Step 1.7
- `spawn_scheduler_actor` new signature `(model, b_max, admission_deadline, admission_queue_max)` defined T1 Step 1.3; called T1 Steps 1.8 (mod.rs), 1.9/1.10 (tests), T3 Step 3.4 (serve fn)
- `admit_err_to_response` defined T4 Step 4.2; called T4 Steps 4.3 + 4.4

**Plan saved to:** `docs/superpowers/plans/2026-05-16-b1-p2-3d-admission-queue.md`
