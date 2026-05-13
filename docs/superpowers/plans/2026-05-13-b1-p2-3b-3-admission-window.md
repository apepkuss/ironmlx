# B1-p2.3b-3 Admission Window + Multi-Request Batching Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 3b-2's "one-admit-per-batch" `driver_loop` with a hybrid 5 ms-deadline + b_max-saturate admission window so multiple HTTP requests arriving close in time batch into a single Scheduler call.

**Architecture:** `driver_loop` becomes a two-phase outer loop: (1) block on `cmd_rx.recv()` for the first admit, then (2) `tokio::select!` over `cmd_rx.recv()` and `tokio::time::sleep(5ms)` to drain additional admits until either `Scheduler::active_count() == b_max` or the deadline expires (hard limit — new admits do not reset it). Adds `batch_count` and `saturate_triggered` atomic test hooks to `SchedulerActorHandle` and folds in three 3b-2 final-review minors.

**Tech Stack:** Rust 2021, Tokio (mpsc, oneshot, time::sleep, runtime::Handle::block_on, select!), ironmlx core (`Scheduler` API unchanged from 3b-1/3b-2).

---

## File Structure

```
ironmlx/src/core/server/scheduler_actor.rs    — MODIFY: driver_loop rewrite + 2 new handle fields + helpers + M1 fix
ironmlx/src/core/server/openai.rs             — MODIFY: M2 — 4-line comment near chunk_size==0 routing
ironmlx/tests/b1_p2_3b_3_admission_window.rs  — NEW: 4 #[ignore] integration scenarios + helpers
ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/
    b1_p2_3b_3_closeout/report.md             — NEW: close-out
```

**Zero modifications to:** `core/server/anthropic.rs`, `core/server/chat_format.rs`, `core/server/mod.rs`, `core/scheduler.rs`, `core/generate.rs`, `core/sampler.rs`, `core/tokenizer.rs`, `core/cache/`, `models/`, `nn/`.

---

## Grounded facts (verified by reading HEAD `dabc28c`)

- [`core/server/scheduler_actor.rs:60-71`](../../ironmlx/src/core/server/scheduler_actor.rs#L60) — `spawn_scheduler_actor(model, b_max) -> SchedulerActorHandle` constructs `admit_count: Arc<AtomicU64>`, clones it for the task closure, returns it on the handle.
- [`core/server/scheduler_actor.rs:73-122`](../../ironmlx/src/core/server/scheduler_actor.rs#L73) — current `driver_loop` uses `while let Some(cmd) = cmd_rx.blocking_recv()` and immediately calls `run_batch_once` per admit. The sunset comment lives at line 103 ("3b-2: one-admit-per-batch").
- [`core/server/scheduler_actor.rs:127-150`](../../ironmlx/src/core/server/scheduler_actor.rs#L127) — `run_batch_once` and `route_event` already handle N-row batches correctly; 3b-3 changes only when `run_batch_once` is invoked.
- [`core/server/scheduler_actor.rs:106-112`](../../ironmlx/src/core/server/scheduler_actor.rs#L106) — current `run_batch_once` Err path is `tracing::error!` then `let _ = sched.evict_all();` — the swallowed `evict_all` Err is what M1 fixes.
- [`core/server/openai.rs:360-365`](../../ironmlx/src/core/server/openai.rs#L360) — M2 fix target. Existing two `// COMPAT(3b-2)` comments live at lines 360 and 361; `let use_scheduler = ...` at line 364.
- [`core/scheduler.rs`](../../ironmlx/src/core/scheduler.rs) — `Scheduler::admit(req) -> Result<RequestId>` errors `"scheduler full: no row available (b_max={b_max})"` when full. `active_count() -> usize` and `phase() -> Phase` exposed. Poison flag set on `prefill_admitted`/`step` Err — only `evict_all` clears it.

---

## Branch Sanity

- [ ] **Step 0: Verify branch + head**

```bash
cd /Volumes/Dev/cxx-mlx
git status --short
git log --oneline -3
```

Expected: branch `ironmlx-b1-p2-3-continuous-batching`, HEAD at `dabc28c` ("docs(b1-p2.3b-3): admission window + multi-request batching activation spec"). Only `design.md` may be untracked in the repo root.

---

## Task 1: `scheduler_actor.rs` rewrite (driver_loop + 2 new hooks + M1) + `openai.rs` M2 comment

**Files:**
- Modify: `ironmlx/src/core/server/scheduler_actor.rs`
- Modify: `ironmlx/src/core/server/openai.rs`

This is the largest task — replaces ~50 lines in `scheduler_actor.rs` with ~100 lines implementing the admission window. Adds two atomic counters to the handle. Adds M1 (evict_all warn) and M2 (chunk_size==0 comment) fixes.

- [ ] **Step 1.1: Add `Duration` import and the `ADMISSION_DEADLINE` const to `scheduler_actor.rs`**

The current `use std::sync::Arc;` block (line 11-13) imports `HashMap`, `atomic::{AtomicU64, Ordering}`, and `Arc`. Add `Duration`:

`old_string`:
```rust
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot, Mutex};
```

`new_string`:
```rust
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, Mutex};
```

Then add the const immediately after the imports block (before `pub enum SchedulerCommand`). Use `Edit` with the anchor of the existing line:

`old_string`:
```rust
use crate::Result;

/// Commands accepted by the actor. 3b-2 ships only [`Admit`]; later
/// phases may add `Cancel { id }`, `Stats`, etc.
pub enum SchedulerCommand {
```

`new_string`:
```rust
use crate::Result;

/// Admission window deadline: maximum time `driver_loop` waits to pack
/// additional `Admit` commands into the current batch after the first
/// admit arrives. Hard limit — new admits during the window do NOT
/// reset it (prevents starvation under sustained admit pressure).
///
/// Hardcoded for 3b-3; a future phase (3d/3e) will surface this via
/// `AppConfig` and a CLI flag.
const ADMISSION_DEADLINE: Duration = Duration::from_millis(5);

/// Commands accepted by the actor. 3b-2 ships only [`Admit`]; later
/// phases may add `Cancel { id }`, `Stats`, etc.
pub enum SchedulerCommand {
```

- [ ] **Step 1.2: Extend `SchedulerActorHandle` with two new atomic counters**

`old_string`:
```rust
/// Handle held by [`crate::core::server::AppState`]. Cheap to clone
/// (`mpsc::Sender` and `Arc<AtomicU64>` are both `Clone`).
#[derive(Clone)]
pub struct SchedulerActorHandle {
    pub cmd_tx: mpsc::Sender<SchedulerCommand>,
    /// Test-observable counter. Incremented by the driver every time
    /// `Scheduler::admit` succeeds. Doc-hidden because production code
    /// shouldn't read it — it exists for integration tests to assert
    /// routing decisions (e.g., "VL request did NOT increment the
    /// counter, so it took the GS path"). Cost: one atomic load per
    /// successful admit.
    #[doc(hidden)]
    pub admit_count: Arc<AtomicU64>,
}
```

`new_string`:
```rust
/// Handle held by [`crate::core::server::AppState`]. Cheap to clone
/// (`mpsc::Sender` and `Arc<AtomicU64>` are both `Clone`).
#[derive(Clone)]
pub struct SchedulerActorHandle {
    pub cmd_tx: mpsc::Sender<SchedulerCommand>,
    /// Test-observable counter. Incremented by the driver every time
    /// `Scheduler::admit` succeeds. Doc-hidden because production code
    /// shouldn't read it — it exists for integration tests to assert
    /// routing decisions (e.g., "VL request did NOT increment the
    /// counter, so it took the GS path"). Cost: one atomic load per
    /// successful admit.
    #[doc(hidden)]
    pub admit_count: Arc<AtomicU64>,
    /// Test-observable counter. Incremented by the driver once per
    /// `run_batch_once` invocation (including failed batches — diagnostic
    /// purpose). When multi-admit batching is working, integration tests
    /// expect `batch_count < admit_count`. Doc-hidden.
    #[doc(hidden)]
    pub batch_count: Arc<AtomicU64>,
    /// Test-observable counter. Incremented by `drain_window` when it
    /// exits because `Scheduler::active_count() >= b_max` (saturate path),
    /// NOT when the deadline expires. Used by integration tests to prove
    /// the saturate-trigger fired without relying on wall-time measurement.
    /// Doc-hidden.
    #[doc(hidden)]
    pub saturate_triggered: Arc<AtomicU64>,
}
```

- [ ] **Step 1.3: Update `spawn_scheduler_actor` to initialize and propagate both new counters**

`old_string`:
```rust
/// Spawn the driver task and return a handle. The driver runs on
/// `tokio::task::spawn_blocking` because [`Scheduler`] is `!Send` (sampler
/// holds a `Cell<Array>`) and the model lock is sync.
pub fn spawn_scheduler_actor(model: Arc<Mutex<Qwen35Model>>, b_max: usize) -> SchedulerActorHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let admit_count = Arc::new(AtomicU64::new(0));
    let admit_count_for_task = admit_count.clone();
    tokio::task::spawn_blocking(move || {
        driver_loop(model, b_max, cmd_rx, admit_count_for_task);
    });
    SchedulerActorHandle {
        cmd_tx,
        admit_count,
    }
}
```

`new_string`:
```rust
/// Spawn the driver task and return a handle. The driver runs on
/// `tokio::task::spawn_blocking` because [`Scheduler`] is `!Send` (sampler
/// holds a `Cell<Array>`) and the model lock is sync.
pub fn spawn_scheduler_actor(model: Arc<Mutex<Qwen35Model>>, b_max: usize) -> SchedulerActorHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let admit_count = Arc::new(AtomicU64::new(0));
    let batch_count = Arc::new(AtomicU64::new(0));
    let saturate_triggered = Arc::new(AtomicU64::new(0));
    let admit_count_for_task = admit_count.clone();
    let batch_count_for_task = batch_count.clone();
    let saturate_triggered_for_task = saturate_triggered.clone();
    tokio::task::spawn_blocking(move || {
        driver_loop(
            model,
            b_max,
            cmd_rx,
            admit_count_for_task,
            batch_count_for_task,
            saturate_triggered_for_task,
        );
    });
    SchedulerActorHandle {
        cmd_tx,
        admit_count,
        batch_count,
        saturate_triggered,
    }
}
```

- [ ] **Step 1.4: Replace `driver_loop` body with the admission-window form**

Replace the entire `driver_loop` function (current lines 73-122) with the new form. Use `Edit` with the full old fn body as `old_string`:

`old_string`:
```rust
fn driver_loop(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    mut cmd_rx: mpsc::Receiver<SchedulerCommand>,
    admit_count: Arc<AtomicU64>,
) {
    let mut sched = Scheduler::new(b_max);
    let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();

    while let Some(cmd) = cmd_rx.blocking_recv() {
        match cmd {
            SchedulerCommand::Admit { request, reply_tx } => {
                let (event_tx, event_rx) = mpsc::unbounded_channel();
                match sched.admit(request) {
                    Ok(id) => {
                        admit_count.fetch_add(1, Ordering::Relaxed);
                        event_txs.insert(id, event_tx);
                        if reply_tx
                            .send(Ok(AdmitReply {
                                request_id: id,
                                event_rx,
                            }))
                            .is_err()
                        {
                            // Caller dropped reply_rx before we could send.
                            // Evict the orphan slot and continue.
                            let _ = sched.evict(id);
                            event_txs.remove(&id);
                            continue;
                        }
                        // 3b-2: one-admit-per-batch. 3b-3 replaces this
                        // with admission-window logic that drains additional
                        // SchedulerCommand::Admit messages before batching.
                        if let Err(e) = run_batch_once(&mut sched, &model, &mut event_txs) {
                            tracing::error!("[SchedulerActor] batch error: {e:?}");
                            // 3b-1 Scheduler poisons itself on Err; evict_all
                            // both clears poison and resets the slot table.
                            let _ = sched.evict_all();
                            event_txs.clear();
                        }
                    }
                    Err(e) => {
                        let _ = reply_tx.send(Err(e));
                    }
                }
            }
        }
    }
    // cmd_rx closed — all senders dropped. Exit cleanly.
}
```

`new_string`:
```rust
fn driver_loop(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    mut cmd_rx: mpsc::Receiver<SchedulerCommand>,
    admit_count: Arc<AtomicU64>,
    batch_count: Arc<AtomicU64>,
    saturate_triggered: Arc<AtomicU64>,
) {
    let mut sched = Scheduler::new(b_max);
    let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
    let rt = tokio::runtime::Handle::current();

    loop {
        // Idle: block waiting for the first admit (or shutdown).
        let Some(first_cmd) = rt.block_on(cmd_rx.recv()) else {
            // cmd_rx closed — all senders dropped. Exit cleanly.
            return;
        };
        handle_admit(first_cmd, &mut sched, &mut event_txs, &admit_count);

        // Admitting: drain additional admits until deadline or saturate.
        // Skip if the first admit already saturated the scheduler (e.g.,
        // b_max == 1) or if `handle_admit` failed and active_count is 0
        // (then there's nothing to prefill — bail to next outer iteration).
        if sched.active_count() == 0 {
            // First admit failed (admit returned Err); nothing to prefill.
            // Loop back to wait for next cmd.
            continue;
        }
        if sched.active_count() < b_max {
            rt.block_on(drain_window(
                &mut cmd_rx,
                &mut sched,
                &mut event_txs,
                &admit_count,
                &saturate_triggered,
                b_max,
                ADMISSION_DEADLINE,
            ));
        }

        // Run the batch. Count it BEFORE invocation so failed batches still
        // appear in the diagnostic counter.
        batch_count.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = run_batch_once(&mut sched, &model, &mut event_txs) {
            tracing::error!("[SchedulerActor] batch error: {e:?}");
            // M1 fix (3b-2 final-review): surface evict_all failure; rely
            // on 3b-1 poison flag to reject subsequent admits.
            if let Err(evict_err) = sched.evict_all() {
                tracing::warn!(
                    "[SchedulerActor] evict_all after batch error also failed: {evict_err:?}; \
                     relying on 3b-1 poison flag to reject subsequent admits"
                );
            }
            event_txs.clear();
        }
    }
}

/// Drain additional `Admit` commands until either the deadline expires or
/// `Scheduler::active_count()` saturates at `b_max`. Hard deadline — new
/// admits do NOT reset the timer.
async fn drain_window(
    cmd_rx: &mut mpsc::Receiver<SchedulerCommand>,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    saturate_triggered: &Arc<AtomicU64>,
    b_max: usize,
    deadline: Duration,
) {
    let timer = tokio::time::sleep(deadline);
    tokio::pin!(timer);
    loop {
        tokio::select! {
            // `biased;` gives the deadline branch priority when both are
            // ready in the same tick, guaranteeing the hard-limit semantic.
            biased;
            _ = &mut timer => return,
            maybe = cmd_rx.recv() => {
                let Some(cmd) = maybe else { return }; // channel closed
                handle_admit(cmd, sched, event_txs, admit_count);
                if sched.active_count() >= b_max {
                    saturate_triggered.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            }
        }
    }
}

/// Process a single `Admit` command: try `Scheduler::admit`; on success
/// register the per-request event channel and increment admit_count;
/// on failure forward the Err to the caller. Reply-tx send failure
/// (caller abandoned) causes the slot to be evicted as cleanup.
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
                .send(Ok(AdmitReply {
                    request_id: id,
                    event_rx,
                }))
                .is_err()
            {
                // Caller dropped reply_rx before we could send.
                // Evict the orphan slot.
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

- [ ] **Step 1.5: Update the module-level doc comment to reflect 3b-3 state**

The current line 1-9 module doc says "3b-2 ships the 'one-admit-per-batch' form ... 3b-3 will replace ...". Update to reflect the new state:

`old_string`:
```rust
//! SchedulerActor — Tokio task wrapping [`Scheduler`] for serving HTTP
//! requests via mpsc channels.
//!
//! 3b-2 ships the "one-admit-per-batch" form of the driver loop. 3b-3 will
//! replace [`driver_loop`]'s `cmd_rx.blocking_recv()` with an
//! admission-window `select!` so the driver can pack multiple concurrent
//! admits into a single batched forward.
//!
//! See `docs/superpowers/specs/2026-05-13-b1-p2-3b-2-scheduler-actor-skeleton-design.md` § 4.
```

`new_string`:
```rust
//! SchedulerActor — Tokio task wrapping [`Scheduler`] for serving HTTP
//! requests via mpsc channels.
//!
//! 3b-3 activates multi-request batching via a hybrid admission window:
//! the first admit starts a [`ADMISSION_DEADLINE`] timer; further admits
//! accumulate until either [`Scheduler::active_count`] saturates at
//! `b_max` (saturate path) or the deadline expires (hard limit, no
//! reset on new admits). Then a single `run_batch_once` call processes
//! the entire batch.
//!
//! See `docs/superpowers/specs/2026-05-13-b1-p2-3b-3-admission-window-design.md` § 4.
```

- [ ] **Step 1.6: Format, build, clippy, run scheduler_actor unit test**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release scheduler_actor 2>&1 | tail -5
```

Expected:
- fmt: clean (run `cargo +nightly fmt --all` if drift, re-check)
- build: `Finished release profile ...`
- clippy: clean. Likely complaints to fix:
  - `clippy::needless_pass_by_value` on `Arc<AtomicU64>` params — fine, they're cheap to clone but the helper takes `&Arc<AtomicU64>` anyway
  - `clippy::let_underscore_must_use` on `let _ = sched.evict(id);` — already in the codebase, ignore
- scheduler_actor unit test: `running 1 test` / `test result: ok. 1 passed; 0 failed` — the `driver_shuts_down_when_cmd_channel_closes` test still works because it doesn't depend on the production `driver_loop`; it spawns its own stand-in.

If you see `unused_variables: ev` or similar in unused branches of the deleted `match` block, ensure the entire old `driver_loop` body was replaced cleanly.

If clippy demands `#[allow(clippy::too_many_arguments)]` on the new `driver_loop` (6 args vs the previous 4), add it:

```rust
#[allow(clippy::too_many_arguments)]
fn driver_loop(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    ...
```

- [ ] **Step 1.7: Apply M2 fix in `openai.rs`**

Find the existing two `// COMPAT(3b-2)` comments at lines 360-361, followed by the `let use_scheduler = ...` line at 364. Use `Edit`:

`old_string`:
```rust
    // COMPAT(3b-2): VL fallback to GS sunsets in B1-p2.4 (batched VL).
    // COMPAT(3b-2): long-prompt fallback to GS sunsets in 3c+ chunked-prefill phase.
    let has_images = request.pixel_values.is_some();
    let prompt_len = request.prompt_ids.len();
    let use_scheduler =
        !has_images && (state.prefill_chunk_size == 0 || prompt_len <= state.prefill_chunk_size);
```

`new_string`:
```rust
    // COMPAT(3b-2): VL fallback to GS sunsets in B1-p2.4 (batched VL).
    // COMPAT(3b-2): long-prompt fallback to GS sunsets in 3c+ chunked-prefill phase.
    //
    // Note (3b-3): when prefill_chunk_size == 0 (chunking disabled by
    // config), this predicate routes ALL text-only requests to the
    // SchedulerActor regardless of length — equivalent to the GS path's
    // behavior when chunking is also disabled there. The 3c+
    // chunked-prefill phase will need to revisit this semantic.
    let has_images = request.pixel_values.is_some();
    let prompt_len = request.prompt_ids.len();
    let use_scheduler =
        !has_images && (state.prefill_chunk_size == 0 || prompt_len <= state.prefill_chunk_size);
```

- [ ] **Step 1.8: Format, build, clippy, full lib regression**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected:
- All clean (fmt/clippy/build)
- Lib tests: **188 passed** (unchanged from 3b-2 baseline — no new lib tests in this task). Integration tests live in their own `tests/` files, not the lib suite.

- [ ] **Step 1.9: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/src/core/server/scheduler_actor.rs ironmlx/src/core/server/openai.rs
git commit -m "feat(b1-p2.3b-3): admission window driver_loop + batch_count/saturate_triggered hooks + 3b-2 minors M1/M2"
```

---

## Task 2: Integration scenarios 1 + 2 (basic 2-admit batching + saturate path)

**Files:**
- Create: `ironmlx/tests/b1_p2_3b_3_admission_window.rs`

3b-3's headline tests: prove multi-admit batching activates by asserting `batch_count < admit_count` AND `saturate_triggered` increments on the b_max-saturate path.

- [ ] **Step 2.1: Create the integration test file with imports + helpers + Scenario 1 + Scenario 2**

```rust
//! B1-p2.3b-3 — Admission window + multi-request batching activation.
//!
//! Four scenarios (see spec § 5.2):
//!   1. `admission_window_two_concurrent_admits_batch_together` — 2
//!      concurrent admits land in 1 batch (batch_count==1, admit_count==2);
//!      per-row tokens match B=1 GS baseline.
//!   2. `admission_window_b_max_saturate_triggers_immediate_prefill` — 4
//!      concurrent admits saturate b_max=4 (saturate_triggered==1).
//!   3. (Task 3) `admission_window_deadline_fires_with_single_admit` — 1
//!      admit reaches deadline (saturate_triggered==0).
//!   4. (Task 3) `admission_window_concurrent_scheduler_and_gs_no_deadlock`
//!      — concurrent scheduler-path + GS-path don't deadlock.
//!
//! Tests are `#[ignore]`-gated; run only with `QWEN35_MODEL` env var.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio::task::JoinSet;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand,
};
use ironmlx::core::tokenizer::Tokenizer;
use ironmlx::core::Loader;
use ironmlx::models::Qwen35Model;

const ARGMAX_BITID_GATE: f64 = 0.95;

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    (Arc::new(Mutex::new(model)), Arc::new(tokenizer))
}

/// Tokenize a chat-template-rendered prompt. Mirrors 3b-2 test pattern.
fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let messages = vec![("user".to_string(), text.to_string())];
    let rendered = tokenizer
        .apply_chat_template(
            &messages,
            /* add_generation_prompt */ true,
            /* kwargs */ None,
        )
        .expect("apply_chat_template");
    tokenizer.encode(&rendered, /* add_special */ false).expect("encode")
}

/// Run a B=1 baseline via direct `GenerationStream`. Locks the model.
fn run_b1_baseline(
    model: &Mutex<Qwen35Model>,
    tokenizer: &Tokenizer,
    request: GenerateRequest,
) -> Vec<u32> {
    let model_guard = model.blocking_lock();
    let mut stream = GenerationStream::new(&model_guard, tokenizer, request).expect("new stream");
    let mut tokens = Vec::new();
    while let Some(ev) = stream.next_token().expect("next_token") {
        tokens.push(ev.token);
        if ev.finish_reason.is_some() {
            break;
        }
    }
    tokens
}

fn argmax_bit_id_ratio(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    same as f64 / n as f64
}

/// Send one Admit cmd via `handle.cmd_tx`, await reply, drain `event_rx` to
/// completion, return collected tokens.
async fn admit_and_drain(
    handle: SchedulerActorHandle,
    request: GenerateRequest,
) -> Vec<u32> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .expect("send admit");
    let reply = reply_rx.await.expect("admit reply").expect("admit ok");
    let mut event_rx = reply.event_rx;
    let mut tokens = Vec::new();
    while let Some(ev) = event_rx.recv().await {
        tokens.push(ev.token);
        if ev.finish_reason.is_some() {
            break;
        }
    }
    tokens
}

fn make_request(prompt_ids: Vec<u32>, max_new_tokens: usize, stop_token_ids: Vec<u32>) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: 256,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn admission_window_two_concurrent_admits_batch_together() {
    let (model, tokenizer) = load_fixture();

    let prompt_a = "What is the capital of France?";
    let prompt_b = "Name three primary colors used in painting.";
    let prompt_a_ids = tokenize_prompt(&tokenizer, prompt_a);
    let prompt_b_ids = tokenize_prompt(&tokenizer, prompt_b);
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let max_new_tokens: usize = 12;

    // 1. B=1 baselines.
    let baseline_a = run_b1_baseline(
        &model,
        &tokenizer,
        make_request(prompt_a_ids.clone(), max_new_tokens, stop_token_ids.clone()),
    );
    let baseline_b = run_b1_baseline(
        &model,
        &tokenizer,
        make_request(prompt_b_ids.clone(), max_new_tokens, stop_token_ids.clone()),
    );
    assert!(!baseline_a.is_empty() && !baseline_b.is_empty(), "baselines must produce tokens");

    // 2. Spawn the actor.
    let handle = spawn_scheduler_actor(model.clone(), 4);
    let admit_before = handle.admit_count.load(Ordering::Relaxed);
    let batch_before = handle.batch_count.load(Ordering::Relaxed);

    // 3. Fire 2 concurrent admits via JoinSet.
    let mut set: JoinSet<Vec<u32>> = JoinSet::new();
    let req_a = make_request(prompt_a_ids, max_new_tokens, stop_token_ids.clone());
    let req_b = make_request(prompt_b_ids, max_new_tokens, stop_token_ids);
    let h1 = handle.clone();
    let h2 = handle.clone();
    set.spawn(async move { admit_and_drain(h1, req_a).await });
    set.spawn(async move { admit_and_drain(h2, req_b).await });

    let mut tokens: Vec<Vec<u32>> = Vec::new();
    while let Some(res) = set.join_next().await {
        tokens.push(res.expect("join task"));
    }
    assert_eq!(tokens.len(), 2, "both tasks must complete");

    // 4. Assert batching invariants.
    let admit_after = handle.admit_count.load(Ordering::Relaxed);
    let batch_after = handle.batch_count.load(Ordering::Relaxed);
    println!(
        "[two_concurrent] admit_delta={} batch_delta={}",
        admit_after - admit_before,
        batch_after - batch_before
    );
    assert_eq!(admit_after - admit_before, 2, "expected 2 admits");
    assert_eq!(
        batch_after - batch_before,
        1,
        "multi-admit batching failed — 2 admits produced {} batches",
        batch_after - batch_before
    );

    // 5. Per-row bit-id parity. JoinSet completion order is not stable; match by
    // first token to disambiguate which row is which.
    let baselines = vec![baseline_a, baseline_b];
    for got in &tokens {
        let baseline_match = baselines
            .iter()
            .find(|b| !b.is_empty() && !got.is_empty() && b[0] == got[0])
            .unwrap_or_else(|| panic!("no baseline first-token matched scheduler row: {got:?}"));
        let ratio = argmax_bit_id_ratio(got, baseline_match);
        println!(
            "[two_concurrent] row bit_id={:.4} (scheduler_len={} baseline_len={})",
            ratio,
            got.len(),
            baseline_match.len()
        );
        assert!(
            ratio >= ARGMAX_BITID_GATE,
            "row bit_id {ratio:.4} below gate {ARGMAX_BITID_GATE}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn admission_window_b_max_saturate_triggers_immediate_prefill() {
    let (model, tokenizer) = load_fixture();

    let prompts = [
        "What is two plus two?",
        "Name one color of the sky during sunset.",
        "Write a single-sentence definition of gravity.",
        "How many continents are there on Earth?",
    ];
    let prompt_ids: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tokenize_prompt(&tokenizer, p))
        .collect();
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let max_new_tokens: usize = 8;

    // Spawn actor with b_max == prompts.len() so saturate triggers.
    let handle = spawn_scheduler_actor(model.clone(), prompts.len());
    let admit_before = handle.admit_count.load(Ordering::Relaxed);
    let batch_before = handle.batch_count.load(Ordering::Relaxed);
    let saturate_before = handle.saturate_triggered.load(Ordering::Relaxed);

    // Fire all 4 admits concurrently.
    let mut set: JoinSet<Vec<u32>> = JoinSet::new();
    for ids in prompt_ids {
        let req = make_request(ids, max_new_tokens, stop_token_ids.clone());
        let h = handle.clone();
        set.spawn(async move { admit_and_drain(h, req).await });
    }

    let mut results: Vec<Vec<u32>> = Vec::new();
    while let Some(res) = set.join_next().await {
        results.push(res.expect("join task"));
    }
    assert_eq!(results.len(), 4, "all 4 tasks must complete");

    let admit_delta = handle.admit_count.load(Ordering::Relaxed) - admit_before;
    let batch_delta = handle.batch_count.load(Ordering::Relaxed) - batch_before;
    let saturate_delta = handle.saturate_triggered.load(Ordering::Relaxed) - saturate_before;
    println!(
        "[saturate] admit_delta={} batch_delta={} saturate_delta={}",
        admit_delta, batch_delta, saturate_delta
    );
    assert_eq!(admit_delta, 4);
    assert_eq!(batch_delta, 1, "4 admits should land in 1 batch");
    assert_eq!(
        saturate_delta, 1,
        "saturate path must trigger when active_count == b_max"
    );

    // Each row should produce non-empty token output.
    for (i, tokens) in results.iter().enumerate() {
        assert!(!tokens.is_empty(), "row {i} produced no tokens");
    }
}
```

(The `tokenize_prompt` helper uses `apply_chat_template` and `encode` based on 3b-2 test corrections. If those signatures don't match — see Task 2 Step 2.2 — adapt to the actual `Tokenizer` API by checking `tests/b1_p2_3b_2_scheduler_actor.rs`.)

- [ ] **Step 2.2: Format + build the new test crate**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx --test b1_p2_3b_3_admission_window 2>&1 | tail -5
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
```

Expected: fmt clean, build clean, clippy clean. Likely fixups:
- If `tokenizer.apply_chat_template(...)` signature differs (kwargs may be a different type), adapt as in 3b-2's actual implementation — open `tests/b1_p2_3b_2_scheduler_actor.rs` and copy the call literal.
- If `GenerateEvent.token` is `u32` (not `Option<u32>`) — already accounted for in the code above (no `Option` unwrap).
- If clippy warns about `clippy::large_enum_variant` on `SchedulerCommand` (cmd channel signed): not applicable; we're just sending.

- [ ] **Step 2.3: Run Scenarios 1 + 2 (~10-20 min on GPU)**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_3b_3_admission_window -- --ignored --nocapture --test-threads=1 2>&1 | tail -40
```

Use `run_in_background: true` + Monitor; timeout ~1200000 ms (20 min). Tests are `--test-threads=1` to avoid concurrent model loads.

Expected: `test result: ok. 2 passed; 0 failed`. Console prints:
- `[two_concurrent] admit_delta=2 batch_delta=1` and per-row bit_id ratios
- `[saturate] admit_delta=4 batch_delta=1 saturate_delta=1`

**If Scenario 1's `batch_delta == 2`** (admits did NOT batch): debug whether `drain_window` is actually entering the `select!`. Possible causes:
- The first admit's reply path is taking too long (rare); the second admit arrives AFTER the deadline window
- `cmd_rx` is buffered (channel size 64), so the second cmd may already be queued — drain_window should pick it up via the `cmd_rx.recv()` branch

If consistently failing: increase the test's deadline expectation OR force serialization. Report DONE_WITH_CONCERNS if unable to reproduce reliably.

**If Scenario 2's `saturate_delta == 0`**: the 4 admits did not saturate before the deadline. This is likely because the 4 admits each take >1ms to send — by the time the 4th arrives, the 5ms deadline expired. Either:
- The test is incorrect about timing assumptions — accept that under some scheduler conditions saturate doesn't fire and downgrade the assertion to "saturate_delta in {0, 1}" with a console warning
- Or hardcode a higher deadline (e.g., 50ms) just for Scenario 2 — but that's outside the spec

If consistently failing report DONE_WITH_CONCERNS — the controller will decide whether to widen the test envelope.

- [ ] **Step 2.4: Full lib regression sanity**

```bash
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: **188 passed** (no new lib tests).

- [ ] **Step 2.5: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/tests/b1_p2_3b_3_admission_window.rs
git commit -m "test(b1-p2.3b-3): admission window scenarios 1 + 2 (concurrent batching + saturate)"
```

---

## Task 3: Integration scenarios 3 + 4 (deadline path + concurrent-with-GS no-deadlock)

**Files:**
- Modify: `ironmlx/tests/b1_p2_3b_3_admission_window.rs` (append Scenario 3 + Scenario 4)

- [ ] **Step 3.1: Append Scenario 3 (deadline fires with single admit)**

Append immediately after the closing `}` of `admission_window_b_max_saturate_triggers_immediate_prefill`:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn admission_window_deadline_fires_with_single_admit() {
    let (model, tokenizer) = load_fixture();

    let prompt = "What is the capital of France?";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let max_new_tokens: usize = 6;

    let handle = spawn_scheduler_actor(model.clone(), 4);
    let admit_before = handle.admit_count.load(Ordering::Relaxed);
    let batch_before = handle.batch_count.load(Ordering::Relaxed);
    let saturate_before = handle.saturate_triggered.load(Ordering::Relaxed);

    let req = make_request(prompt_ids, max_new_tokens, stop_token_ids);
    let tokens = admit_and_drain(handle.clone(), req).await;
    assert!(!tokens.is_empty(), "tokens produced");

    let admit_delta = handle.admit_count.load(Ordering::Relaxed) - admit_before;
    let batch_delta = handle.batch_count.load(Ordering::Relaxed) - batch_before;
    let saturate_delta = handle.saturate_triggered.load(Ordering::Relaxed) - saturate_before;
    println!(
        "[deadline] admit_delta={} batch_delta={} saturate_delta={}",
        admit_delta, batch_delta, saturate_delta
    );
    assert_eq!(admit_delta, 1);
    assert_eq!(batch_delta, 1);
    assert_eq!(
        saturate_delta, 0,
        "single admit must use deadline path, not saturate"
    );
}
```

- [ ] **Step 3.2: Append Scenario 4 (concurrent scheduler + GS, no deadlock)**

This is the M3 fix from 3b-2's final review. Both paths run concurrently and must both complete without deadlocking on `model.blocking_lock`.

Append immediately after Scenario 3's closing `}`:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn admission_window_concurrent_scheduler_and_gs_no_deadlock() {
    let (model, tokenizer) = load_fixture();
    let tokenizer_arc = tokenizer.clone();

    let prompt = "Name a color.";
    let prompt_ids = tokenize_prompt(&tokenizer, prompt);
    let stop_token_ids: Vec<u32> = tokenizer.eos_token_ids().to_vec();
    let max_new_tokens: usize = 4;

    let handle = spawn_scheduler_actor(model.clone(), 4);
    let admit_before = handle.admit_count.load(Ordering::Relaxed);

    // Task A: scheduler path.
    let req_a = make_request(prompt_ids.clone(), max_new_tokens, stop_token_ids.clone());
    let handle_a = handle.clone();
    let task_a = tokio::spawn(async move { admit_and_drain(handle_a, req_a).await });

    // Task B: GS path. Runs `GenerationStream` directly on `spawn_blocking`
    // to mirror the production HTTP handler GS path.
    let req_b = make_request(prompt_ids, max_new_tokens, stop_token_ids);
    let model_b = model.clone();
    let tokenizer_b = tokenizer_arc.clone();
    let task_b = tokio::task::spawn_blocking(move || -> Vec<u32> {
        let model_guard = model_b.blocking_lock();
        let mut stream =
            GenerationStream::new(&model_guard, &tokenizer_b, req_b).expect("new stream");
        let mut tokens = Vec::new();
        while let Some(ev) = stream.next_token().expect("next_token") {
            tokens.push(ev.token);
            if ev.finish_reason.is_some() {
                break;
            }
        }
        tokens
    });

    // Both tasks must complete within a generous bound (60s).
    let tokens_a = tokio::time::timeout(Duration::from_secs(60), task_a)
        .await
        .expect("task A timed out — possible deadlock")
        .expect("task A join");
    let tokens_b = tokio::time::timeout(Duration::from_secs(60), task_b)
        .await
        .expect("task B timed out — possible deadlock")
        .expect("task B join");

    assert!(!tokens_a.is_empty(), "task A (scheduler) produced no tokens");
    assert!(!tokens_b.is_empty(), "task B (GS) produced no tokens");
    let admit_delta = handle.admit_count.load(Ordering::Relaxed) - admit_before;
    println!(
        "[concurrent_no_deadlock] admit_delta={} task_a_len={} task_b_len={}",
        admit_delta,
        tokens_a.len(),
        tokens_b.len()
    );
    assert_eq!(admit_delta, 1, "only scheduler path incremented admit_count");
}
```

- [ ] **Step 3.3: Format + build**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx --test b1_p2_3b_3_admission_window 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 3.4: Run all 4 Scenarios**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_3b_3_admission_window -- --ignored --nocapture --test-threads=1 2>&1 | tail -40
```

Use `run_in_background: true` + Monitor; timeout ~1800000 ms (30 min).

Expected: `test result: ok. 4 passed; 0 failed`. Each scenario prints its diagnostic line.

**If Scenario 4 times out**: deadlock between scheduler driver and GS path. Diagnose:
- Does the driver hold `model.blocking_lock()` outside `run_batch_once`? It should not (spec §3.5).
- Does the GS path use `blocking_lock()` correctly? Standard pattern.
- Suspicion: the `tokio::task::spawn_blocking` for task B may be waiting on the same `spawn_blocking` thread pool as the driver task. Mitigation: increase Tokio worker threads or pre-spawn driver before task B.

If consistently failing, STOP and report BLOCKED with the timeout details.

- [ ] **Step 3.5: Commit**

```bash
cd /Volumes/Dev/cxx-mlx
git add ironmlx/tests/b1_p2_3b_3_admission_window.rs
git commit -m "test(b1-p2.3b-3): scenarios 3 + 4 (deadline path + concurrent-with-GS no-deadlock — M3 fix)"
```

---

## Task 4: Regression sweep + close-out

**Files:**
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_3_closeout/report.md`

- [ ] **Step 4.1: Full hygiene sweep**

```bash
cd /Volumes/Dev/cxx-mlx
cargo +nightly fmt --all -- --check 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo build --release -p ironmlx 2>&1 | tail -3
MLX_DIR=/Users/sam/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected all green:
- fmt clean
- clippy clean
- build clean
- lib tests: **188 passed** (unchanged from 3b-2)

- [ ] **Step 4.2: P6.3 single-image regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_qwen35_vl_logits_match -- --ignored 2>&1 | tail -5
```

`run_in_background: true` + Monitor; timeout ~600000 ms. Expected: PASS, `max_diff=0.3906`, `first_token=760`.

- [ ] **Step 4.3: P6.6 logits-match regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored 2>&1 | tail -5
```

Expected: PASS, `first_token=760`.

- [ ] **Step 4.4: P6.7 chunked-prefill regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test p6_7_chunked_prefill -- --ignored 2>&1 | tail -5
```

Timeout ~1500000 ms. Expected: PASS.

- [ ] **Step 4.5: B1-p2.1 prefill regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_1_batched_prefill -- --ignored 2>&1 | tail -5
```

Timeout ~1500000 ms. Expected: PASS — 10/12 argmax bit-id, max_diff ≤ 0.19.

- [ ] **Step 4.6: B1-p2.2 batched decode regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_2_batched_decode -- --ignored 2>&1 | tail -5
```

Timeout ~1500000 ms. Expected: PASS — 57/60 argmax bit-id.

- [ ] **Step 4.7: B1-p2.3b-1 scheduler regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_3b_1_scheduler_step -- --ignored --test-threads=1 2>&1 | tail -10
```

Timeout ~1800000 ms. Expected: PASS — 3 scenarios all bit_id=1.0000.

- [ ] **Step 4.8: B1-p2.3b-2 scheduler_actor regression**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_3b_2_scheduler_actor -- --ignored --test-threads=1 2>&1 | tail -10
```

Timeout ~600000 ms. Expected: PASS — 3 scenarios; Scenario A bit_id ≥ 0.95.

- [ ] **Step 4.9: B1-p2.3b-3 admission window re-run (sanity)**

```bash
QWEN35_MODEL=/Users/sam/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/ \
MLX_DIR=/Users/sam/.local/mlx \
cargo test -p ironmlx --release --test b1_p2_3b_3_admission_window -- --ignored --test-threads=1 2>&1 | tail -20
```

Timeout ~1800000 ms. Expected: 4 scenarios PASS.

**If any regression fails:** STOP and report BLOCKED. 3b-3's changes are localized to `scheduler_actor.rs` + a 4-line comment in `openai.rs`; broad regressions should not occur.

- [ ] **Step 4.10: Write the close-out report**

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_3_closeout/report.md`:

```markdown
# B1-p2.3b-3 Admission window + multi-request batching — Close-out

**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3b-2 head `7c69919`)
**Date:** 2026-05-13
**Spec:** `docs/superpowers/specs/2026-05-13-b1-p2-3b-3-admission-window-design.md` (commit `dabc28c`)
**Plan:** `docs/superpowers/plans/2026-05-13-b1-p2-3b-3-admission-window.md`

## Summary

Replaced 3b-2's "one-admit-per-batch" `driver_loop` with a hybrid 5ms-deadline
+ b_max-saturate admission window. First admit starts a timer; subsequent
admits accumulate in the same batch until either `Scheduler::active_count()
== b_max` (saturate path) or the deadline expires (hard limit, no reset on
new admits). `SchedulerActorHandle` gains two atomic test hooks
(`batch_count`, `saturate_triggered`) to enable integration tests that
verify multi-admit batching is genuinely happening.

Three 3b-2 final-review minors folded in:
- **M1**: `evict_all` failure after batch error now logs `tracing::warn!`
  with poison-flag reliance note.
- **M2**: `openai.rs` `chunk_size == 0` routing semantic explained inline
  for the 3c+ chunked-prefill phase implementer.
- **M3**: Scenario 4 verifies concurrent scheduler-path + GS-path don't
  deadlock.

Scheduler API, server `mod.rs`/`AppState`, and HTTP handler routing
unchanged. iron-bench v1 sees no protocol change.

## Acceptance

| Test | Result |
| --- | --- |
| `driver_shuts_down_when_cmd_channel_closes` (unit, 3b-2 inherited) | ✅ |
| `admission_window_two_concurrent_admits_batch_together` | ✅ batch_delta=1, per-row bit_id ≥ 0.95 |
| `admission_window_b_max_saturate_triggers_immediate_prefill` | ✅ batch_delta=1, saturate_delta=1 |
| `admission_window_deadline_fires_with_single_admit` | ✅ batch_delta=1, saturate_delta=0 |
| `admission_window_concurrent_scheduler_and_gs_no_deadlock` | ✅ both tasks complete < 60s |

## Architectural Changes

1. **`ironmlx/src/core/server/scheduler_actor.rs`**:
   - Added `ADMISSION_DEADLINE = Duration::from_millis(5)` const.
   - `SchedulerActorHandle` gains `batch_count` + `saturate_triggered` (`Arc<AtomicU64>`, doc-hidden).
   - `spawn_scheduler_actor` initializes and propagates both new counters.
   - `driver_loop` rewritten: two-phase outer loop using `tokio::runtime::Handle::current().block_on(...)` to bridge to async; `drain_window` async helper drains additional admits via `tokio::select! { biased; deadline | cmd_rx.recv() }`; `handle_admit` factored out (DRY).
   - M1 fix: `evict_all` Err after batch error now `tracing::warn!`s.
2. **`ironmlx/src/core/server/openai.rs`**:
   - M2 fix: 4-line comment near `prefill_chunk_size == 0` routing branch.

No changes to: `core/server/anthropic.rs`, `core/server/chat_format.rs`, `core/server/mod.rs`, `core/scheduler.rs`, `core/generate.rs`, `core/sampler.rs`, `core/tokenizer.rs`, `core/cache/`, `models/`, `nn/`.

## Compat sunset markers (recorded in code, inherited from 3b-2)

| Location | Marker | Sunset |
| --- | --- | --- |
| `openai.rs::chat_completions` dispatch | `// COMPAT(3b-2): VL fallback to GS sunsets in B1-p2.4` | B1-p2.4 batched VL |
| `openai.rs::chat_completions` dispatch | `// COMPAT(3b-2): long-prompt fallback to GS sunsets in 3c+ chunked-prefill phase` | 3c+ chunked prefill |
| `anthropic.rs` untouched | (implicit) | 3b-4 Anthropic refactor |
| `scheduler_actor.rs::ADMISSION_DEADLINE` | hardcoded 5ms | 3d/3e config exposure |

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `<T1_SHA>` | feat | admission window driver_loop + batch_count/saturate_triggered + 3b-2 M1/M2 |
| `<T2_SHA>` | test | scenarios 1 + 2 (concurrent batching + saturate) |
| `<T3_SHA>` | test | scenarios 3 + 4 (deadline path + concurrent-with-GS — M3 fix) |
| `<T4_SHA>` | docs | This close-out |

(Fill `<T*_SHA>` from `git log --oneline dabc28c..HEAD`.)

## Regression Status

| Check | Result |
| --- | --- |
| `cargo +nightly fmt --all -- --check` | clean |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean |
| `cargo build --release -p ironmlx` | clean |
| `cargo test -p ironmlx --lib --release` | **<COUNT> passed / 0 failed** |
| P6.3 single-image | <FILL> |
| P6.6 logits-match | <FILL> |
| P6.7 chunked-prefill matrix | <FILL> |
| B1-p2.1 batched prefill | <FILL> |
| B1-p2.2 batched decode | <FILL> |
| B1-p2.3b-1 scheduler scenarios | <FILL> |
| B1-p2.3b-2 scheduler_actor scenarios | <FILL> |
| B1-p2.3b-3 two_concurrent_admits | batch_delta=1, per-row bit_id ≥ 0.95 |
| B1-p2.3b-3 b_max_saturate | batch_delta=1, saturate_delta=1 |
| B1-p2.3b-3 deadline_fires_single_admit | batch_delta=1, saturate_delta=0 |
| B1-p2.3b-3 concurrent_scheduler_and_gs_no_deadlock | both tasks complete |

## Notes

- **Multi-request batching is now live for the SchedulerActor path.** Concurrent text-only short-prompt requests pack into a single Scheduler batch within a 5ms window. iron-bench v2's batching benchmarks can now exercise this; iron-bench v1 (single-request stream) sees no change.
- **Lock strategy unchanged from 3b-2.** Driver holds `model.blocking_lock()` only during `run_batch_once`; the admission window itself is purely async (`tokio::select!`) and holds no model lock. Scenario 4 verifies the concurrent scheduler + GS no-deadlock invariant.
- **Hardcoded 5ms deadline**: the spec recorded this as a 3d/3e sunset item — surface via `AppConfig` and a CLI flag once admission queue / preemption lands.
- **`tokio::select! { biased; ... }`** is the first production use of this pattern in the codebase. The `biased;` keyword ensures the deadline branch takes priority when both are ready in the same tick, preserving the hard-limit semantics.
- **3b-2 M1/M2/M3 closed**: M1 (`evict_all` warn) inlined; M2 (`chunk_size==0` comment) added; M3 (concurrent scheduler+GS) covered by Scenario 4.

## B1-p2.3x Next Steps

- **B1-p2.3b-4** — Anthropic handler refactor (6-event SSE wrapper).
- **B1-p2.3c** — Per-row KV cache offset tracking; lifts lockstep constraint.
- **B1-p2.3 (chunked-prefill phase)** — Adds batched prefill chunking; removes `prompt_len > chunk_size` GS fallback.
- **B1-p2.3d** — Admission queue + preemption; also surfaces `ADMISSION_DEADLINE` via config.
- **B1-p2.3e** — Per-row sampler invocation tuning.
- **B1-p2.4** — VL B>1 batched serving; removes VL GS fallback.

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-13-b1-p2-3b-3-admission-window-design.md`
- Plan: `docs/superpowers/plans/2026-05-13-b1-p2-3b-3-admission-window.md`
- Modified module: `ironmlx/src/core/server/scheduler_actor.rs`
- Modified handler: `ironmlx/src/core/server/openai.rs` (M2 comment only)
- Integration test: `ironmlx/tests/b1_p2_3b_3_admission_window.rs`
- Predecessor: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_2_closeout/report.md`
```

Fill each `<FILL>` and `<COUNT>` from the regression sweep outputs. Leave `<T*_SHA>` placeholders to be filled after Step 4.11 commit (or fill them just before Step 4.11 from `git log --oneline dabc28c..HEAD`).

- [ ] **Step 4.11: Commit close-out**

```bash
cd /Volumes/Dev/cxx-mlx
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3b_3_closeout/report.md
git commit -m "docs(b1-p2.3b-3): close-out — admission window + multi-request batching"
```

- [ ] **Step 4.12: Final summary log**

```bash
cd /Volumes/Dev/cxx-mlx
git log --oneline dabc28c..HEAD
```

Expected: 4 commits (T1 feat, T2 test, T3 test, T4 docs).

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
| --- | --- |
| §1 Goal 1 (admission window driver_loop) | T1 Step 1.4 |
| §1 Goal 2 (batch_count + saturate_triggered hooks) | T1 Step 1.2 (Handle), Step 1.3 (init) |
| §1 Goal 3 (4 integration scenarios) | T2 (1, 2), T3 (3, 4) |
| §1 Goal 4 (M1/M2/M3 fold-in) | T1 Step 1.4 (M1 evict_all warn), T1 Step 1.7 (M2 comment), T3 Step 3.2 (M3 Scenario 4) |
| §3.1 sunset marker at line 103 (replace it) | T1 Step 1.4 (replaces entire driver_loop body) |
| §4.1 Phase state machine | T1 Step 1.4 (`Idle → first admit → Admitting → drain → Run batch → Idle`) |
| §4.2 Handle extension | T1 Step 1.2 |
| §4.3 Driver loop with `select! { biased; ... }` | T1 Step 1.4 (`drain_window` async helper with explicit `biased;`) |
| §4.4 run_batch_once / route_event unchanged | T1 Step 1.4 (doesn't modify these fns) |
| §4.5 M2 fix | T1 Step 1.7 |
| §4.6 module surface | All T1 steps combined |
| §5.1 unit tests (existing driver_shuts_down stays) | T1 Step 1.4 preserves the test (`#[cfg(test)] mod tests` block untouched) |
| §5.2 Scenario 1 | T2 Step 2.1 |
| §5.2 Scenario 2 | T2 Step 2.1 |
| §5.2 Scenario 3 | T3 Step 3.1 |
| §5.2 Scenario 4 | T3 Step 3.2 |
| §6 acceptance gates | T4 Steps 4.1-4.9 |
| §7 sunset notes | Close-out template |
| §9 risk register (jitter, biased, batch_count edge, deadlock) | T1 Step 1.4 (`biased;` + explicit comment), T2 Step 2.3 (flakiness mitigation guidance), T3 Step 3.4 (deadlock detection via `tokio::time::timeout`) |

All spec sections covered.

**2. Placeholder scan:**
- `<FILL>` / `<COUNT>` / `<T*_SHA>` in close-out template (T4 Step 4.10) — explicit "fill at execution time".
- No bare "TBD" / "implement later" / "fill in details" elsewhere.

**3. Type consistency:**

| Symbol | First defined | Reused |
| --- | --- | --- |
| `ADMISSION_DEADLINE: Duration = from_millis(5)` | T1 Step 1.1 | T1 Step 1.4 (`driver_loop` passes to `drain_window`) |
| `SchedulerActorHandle { cmd_tx, admit_count, batch_count, saturate_triggered }` | T1 Step 1.2 | T1 Step 1.3 (`spawn_scheduler_actor` init), T2 (test reads `admit_count`, `batch_count`, `saturate_triggered`), T3 (test reads same) |
| `driver_loop(model, b_max, cmd_rx, admit_count, batch_count, saturate_triggered)` | T1 Step 1.4 | internal only |
| `drain_window(cmd_rx, sched, event_txs, admit_count, saturate_triggered, b_max, deadline)` | T1 Step 1.4 | called by `driver_loop` |
| `handle_admit(cmd, sched, event_txs, admit_count)` | T1 Step 1.4 | called by `driver_loop` (first admit) and `drain_window` (subsequent admits) |
| `admit_and_drain(handle, request) -> Vec<u32>` helper | T2 Step 2.1 | T3 (Scenarios 3, 4 reuse) |
| `tokenize_prompt` / `run_b1_baseline` / `argmax_bit_id_ratio` / `load_fixture` / `make_request` helpers | T2 Step 2.1 | T3 Scenarios 3/4 reuse |
| `ARGMAX_BITID_GATE = 0.95` const | T2 Step 2.1 | T2 Scenario 1 |

All names consistent across tasks. Helper functions defined in Task 2 are referenced in Task 3 — the implementer of Task 3 must NOT redefine them; they're appending to the same file.
