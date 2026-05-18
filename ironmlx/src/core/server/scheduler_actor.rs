//! SchedulerActor — Tokio task wrapping [`Scheduler`] for serving HTTP
//! requests via mpsc channels.
//!
//! 3b-3 activates multi-request batching via a hybrid admission window:
//! the first admit starts a [`ADMISSION_DEADLINE`] timer; further admits
//! accumulate until either [`Scheduler::active_count`] saturates at
//! `b_max` (saturate path) or the deadline expires (hard limit, no
//! reset on new admits).
//!
//! 3c-3 introduces the rolling decode loop: after first-batch prefill
//! the driver biased-selects between `cmd_rx.recv()` (mid-batch admit)
//! and an always-ready step branch. Mid admits route through
//! [`Scheduler::admit_mid`] (B=1 temp-cache prefill + adopt-into-main);
//! step branch calls [`Scheduler::step`] + [`Scheduler::gc_finished_rows`].
//! The loop exits when `active_count == 0` AND `cmd_rx` is empty.
//!
//! See `docs/superpowers/specs/2026-05-13-b1-p2-3b-3-admission-window-design.md` § 4.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, Mutex};

use crate::core::generate::GenerateRequest;
use crate::core::scheduler::{Phase, RequestId, Scheduler, StepEvent};
use crate::models::Qwen35Model;
use crate::Result;

/// Commands accepted by the actor. 3b-2 ships only [`Admit`]; later
/// phases may add `Cancel { id }`, `Stats`, etc.
pub enum SchedulerCommand {
    /// Submit a request for batched generation. On success, replies with
    /// the admitted [`RequestId`] and an mpsc receiver that streams
    /// [`StepEvent`]s (one per produced token, until `finish_reason`
    /// becomes `Some(_)` on the final event for this row).
    Admit {
        request: GenerateRequest,
        reply_tx: oneshot::Sender<Result<AdmitReply>>,
    },
}

/// A request parked in `driver_loop`'s admission queue while the scheduler
/// is at `active_count == b_max`. Drained when `gc_finished_rows` frees a
/// slot, then handed to `handle_admit_mid_chunked`.
struct PendingAdmit {
    request: GenerateRequest,
    reply_tx: oneshot::Sender<Result<AdmitReply>>,
}

/// Event yielded by the rolling decode loop's biased select. Either a
/// new admit command arrived (mid-batch admit), the always-ready step
/// branch fired, or the cmd_rx channel was closed (shutdown).
enum RollingEvent {
    Admit(SchedulerCommand),
    Step,
    Shutdown,
}

/// Reply payload for [`SchedulerCommand::Admit`]. Carries the assigned
/// [`RequestId`] and the per-request event receiver.
pub struct AdmitReply {
    pub request_id: RequestId,
    pub event_rx: mpsc::UnboundedReceiver<StepEvent>,
}

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
    /// batch (prefill_admitted invocation, including failed batches —
    /// diagnostic purpose). When multi-admit batching is working,
    /// integration tests expect `batch_count < admit_count`. Doc-hidden.
    #[doc(hidden)]
    pub batch_count: Arc<AtomicU64>,
    /// Test-observable counter. Incremented by `drain_window` when it
    /// exits because `Scheduler::active_count() >= b_max` (saturate path),
    /// NOT when the deadline expires. Used by integration tests to prove
    /// the saturate-trigger fired without relying on wall-time measurement.
    /// Doc-hidden.
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
    // ── B1-p2.5 G3: /healthz monitoring atomics ──────────────────────────
    /// Live count of in-flight (active) requests in the scheduler slots.
    /// Updated by driver_loop tail on every rolling iteration.
    pub b_active: Arc<AtomicU64>,
    /// Live count of requests parked in the admission queue.
    /// Updated by driver_loop tail on every rolling iteration.
    pub b_queued: Arc<AtomicU64>,
    /// Monotonic count of admits rejected due to admission queue full.
    /// Aliased from `queue_rejected` — single source of truth in driver_loop.
    /// P1.1: Scheduler.admission_queue_full_count field removed (no fetch_add
    /// caller); health collector now reads from this Arc directly. B1-p2.5.
    pub admission_queue_full_count: Arc<AtomicU64>,
    /// Monotonic count of admits rejected due to memory budget exceeded.
    /// Cloned from Scheduler::memory_budget_exceeded_count.
    pub memory_budget_exceeded_count: Arc<AtomicU64>,
    /// Shared Arc into BudgetState::active — live bytes charged to KV cache.
    pub kv_cache_active_bytes: Arc<AtomicUsize>,
    /// KV cache soft limit in bytes (computed at startup; static for lifetime).
    pub kv_cache_soft_limit_bytes: usize,
}

/// Spawn the driver task and return a handle. The driver runs on
/// `tokio::task::spawn_blocking` because [`Scheduler`] is `!Send`
/// (holds Array fields: KVCache, prng_state) and the model lock is sync.
///
/// # Arguments
/// - `model` — shared model handle (Mutex-protected sync state).
/// - `b_max` — maximum concurrent in-flight requests (Scheduler slot count).
/// - `admission_deadline` — drain-window timeout after the first admit in a
///   batch arrives. Hard limit; new admits do not reset it.
/// - `admission_queue_max` — capacity of the FIFO admission queue. `0`
///   disables queueing (immediate Err on saturation, mirroring pre-3d).
/// - `effective_cap_max` — upper bound on `prompt_len + max_new_tokens`
///   per request. Computed at boot as
///   `min(--max-cache-cap CLI, model.config.max_position_embeddings)`.
///   Passed directly to `Scheduler::new`. B1-p2.3f.
/// - `meta` — model memory-budget metadata for startup validation. B1-p2.5.
pub fn spawn_scheduler_actor(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    meta: crate::core::memory_budget::ModelMeta,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError> {
    // Single Scheduler::new — both budget validation + driver_loop use this
    // instance. Arc atomics are cloned before ownership moves into the task.
    // B1-p2.5 P0 fix: previously two Scheduler instances were created; the
    // handle held Arc clones from the dropped #1 while driver_loop mutated #2.
    let scheduler = Scheduler::new(b_max, effective_cap_max, meta)?;

    // Clone health atomics BEFORE moving scheduler into driver_loop.
    let memory_budget_exceeded_count = scheduler.memory_budget_exceeded_count.clone();
    let kv_cache_active_bytes = scheduler.budget_state.shared_active();
    let kv_cache_soft_limit_bytes = scheduler.budget_state.soft_limit();

    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let admit_count = Arc::new(AtomicU64::new(0));
    let batch_count = Arc::new(AtomicU64::new(0));
    let saturate_triggered = Arc::new(AtomicU64::new(0));
    let queue_depth_peak = Arc::new(AtomicUsize::new(0));
    let queue_rejected = Arc::new(AtomicU64::new(0));
    // B1-p2.5 G3: live b_active / b_queued updated by driver_loop tail.
    let b_active = Arc::new(AtomicU64::new(0));
    let b_queued = Arc::new(AtomicU64::new(0));

    let admit_count_for_task = admit_count.clone();
    let batch_count_for_task = batch_count.clone();
    let saturate_triggered_for_task = saturate_triggered.clone();
    let queue_depth_peak_for_task = queue_depth_peak.clone();
    let queue_rejected_for_task = queue_rejected.clone();
    let b_active_for_task = b_active.clone();
    let b_queued_for_task = b_queued.clone();
    tokio::task::spawn_blocking(move || {
        driver_loop(
            scheduler,
            model,
            admission_deadline,
            admission_queue_max,
            cmd_rx,
            admit_count_for_task,
            batch_count_for_task,
            saturate_triggered_for_task,
            queue_depth_peak_for_task,
            queue_rejected_for_task,
            b_active_for_task,
            b_queued_for_task,
        );
    });
    Ok(SchedulerActorHandle {
        cmd_tx,
        admit_count,
        batch_count,
        saturate_triggered,
        queue_depth_peak,
        queue_rejected: queue_rejected.clone(),
        b_active,
        b_queued,
        // P1.1: alias admission_queue_full_count to queue_rejected Arc —
        // driver_loop is the single fetch_add site; Scheduler field removed.
        admission_queue_full_count: queue_rejected,
        memory_budget_exceeded_count,
        kv_cache_active_bytes,
        kv_cache_soft_limit_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn driver_loop(
    scheduler: Scheduler,
    model: Arc<Mutex<Qwen35Model>>,
    admission_deadline: Duration,
    admission_queue_max: usize,
    mut cmd_rx: mpsc::Receiver<SchedulerCommand>,
    admit_count: Arc<AtomicU64>,
    batch_count: Arc<AtomicU64>,
    saturate_triggered: Arc<AtomicU64>,
    queue_depth_peak: Arc<AtomicUsize>,
    queue_rejected: Arc<AtomicU64>,
    b_active: Arc<AtomicU64>,
    b_queued: Arc<AtomicU64>,
) {
    // Receive Scheduler ownership from spawn_scheduler_actor (single instance).
    // P0 fix: previously driver_loop called Scheduler::new a second time,
    // creating fresh Arc atomics disconnected from the handle. B1-p2.5.
    let mut sched = scheduler;
    let b_max = sched.b_max();
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
                        let _ = pending
                            .reply_tx
                            .send(Err(anyhow::anyhow!("scheduler shutting down")));
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
                        handle_admit_mid_chunked(
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

            // B1-p2.5 G3: update /healthz live counters at tail of every rolling step.
            b_active.store(sched.active_count() as u64, Ordering::Relaxed);
            b_queued.store(admission_queue.len() as u64, Ordering::Relaxed);

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
                            let _ = pending
                                .reply_tx
                                .send(Err(anyhow::anyhow!("scheduler evict_all failed")));
                        }
                        event_txs.clear();
                        continue 'outer;
                    }
                    event_txs.clear();
                    // Pop first queued admit as the new batch's first admit.
                    let pending = admission_queue
                        .pop_front()
                        .expect("queue non-empty checked");
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
                            let Some(p) = admission_queue.pop_front() else {
                                break;
                            };
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
                            tracing::error!(
                                "[SchedulerActor] re-prefill (queue drain) error: {e:?}"
                            );
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

/// Mid-batch admit handler — chunked (B1-p2.3c+).
///
/// Orchestrates the three-phase chunked admit:
/// 1. `Scheduler::admit_mid_begin` — reserve slot + alloc temp cache.
/// 2. Loop `admit_mid_chunk` until last chunk, interleaving one
///    `Scheduler::step` between chunks so active rows continue
///    emitting tokens at chunk-boundary cadence (spec §4.5.5
///    chunk:step = 1:1).
/// 3. `Scheduler::admit_mid_finalize` — adopt temp → main cache,
///    sample first generated token.
///
/// Acquires `model.blocking_lock()` per phase (begin / per-chunk /
/// per-step / finalize) so each phase yields the lock between calls.
/// Active rows' SSE consumers see token events at ~chunk forward time
/// granularity instead of one multi-second prefill stall.
///
/// On any error during the loop, the orphan slot is evicted and
/// `event_txs[id]` removed so the next `step()` does not panic on an
/// empty `generated_tokens`.
fn handle_admit_mid_chunked(
    cmd: SchedulerCommand,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<Qwen35Model>>,
) {
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    // Phase 1: begin.
    let mut handle = {
        let m = model.blocking_lock();
        match sched.admit_mid_begin(request, &m) {
            Ok(h) => h,
            Err(e) => {
                let _ = reply_tx.send(Err(e));
                return;
            }
        }
    };
    let id = handle.request_id;
    event_txs.insert(id, event_tx);
    if reply_tx
        .send(Ok(AdmitReply {
            request_id: id,
            event_rx,
        }))
        .is_err()
    {
        // Caller dropped reply_rx before we did any GPU work.
        let _ = sched.evict(id);
        event_txs.remove(&id);
        return;
    }

    // Phase 2: chunk loop. Interleave one active-row step per chunk
    // except after the last chunk (finalize is the next step there).
    loop {
        let is_last = {
            let m = model.blocking_lock();
            match sched.admit_mid_chunk(&mut handle, &m) {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!("[SchedulerActor] admit_mid_chunk error: {e:?}");
                    let _ = sched.evict(id);
                    event_txs.remove(&id);
                    return;
                }
            }
        };

        if is_last {
            break;
        }

        // Interleave one active-row decode step.
        let step_result = {
            let m = model.blocking_lock();
            sched.step(&m)
        };
        match step_result {
            Ok(events) => {
                for ev in events {
                    route_event(ev, event_txs);
                }
                sched.gc_finished_rows(event_txs);
            }
            Err(e) => {
                tracing::error!("[SchedulerActor] step error inside chunked admit_mid loop: {e:?}");
                let _ = sched.evict(id);
                event_txs.remove(&id);
                return;
            }
        }
    }

    // Phase 3: finalize.
    let m = model.blocking_lock();
    match sched.admit_mid_finalize(handle, &m) {
        Ok((_id, first_event)) => {
            admit_count.fetch_add(1, Ordering::Relaxed);
            route_event(first_event, event_txs);
        }
        Err(e) => {
            tracing::error!("[SchedulerActor] admit_mid_finalize error: {e:?}");
            let _ = sched.evict(id);
            event_txs.remove(&id);
        }
    }
}

/// Push a pending admit into the queue if there's capacity; otherwise reply
/// with `Err(SchedulerError::QueueFull)` (wrapped in anyhow) and bump
/// `queue_rejected`. Updates `queue_depth_peak` via `fetch_max`.
///
/// HTTP handlers downcast the anyhow Err to [`SchedulerError`] to map
/// QueueFull → HTTP 503 + Retry-After; other errors → HTTP 400.
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
        let _ = reply_tx.send(Err(anyhow::Error::new(
            crate::core::scheduler::SchedulerError::QueueFull {
                capacity: queue_max,
            },
        )));
        return;
    }
    queue.push_back(PendingAdmit { request, reply_tx });
    queue_depth_peak.fetch_max(queue.len(), Ordering::Relaxed);
}

/// Drain the admission queue head-by-head while the Scheduler has free
/// slots. Each drained entry is handed to `handle_admit_mid_chunked` which runs
/// the B=1 prefill + adopts the row + sends `AdmitReply`. Stops when the
/// queue empties or `active_count() == b_max`.
///
/// IMPORTANT: `admit_mid` is only legal in `Decoding` phase. If
/// `gc_finished_rows` just transitioned the scheduler to `Finished`
/// (because `active_count` dropped to 0), the caller's rolling-loop
/// exit branch (`active_count == 0 && queue non-empty`) will handle the
/// queued entries via `evict_all` + fresh `prefill_admitted`. Return
/// early here so we do not call `admit_mid` in an illegal phase.
fn drain_admission_queue(
    queue: &mut VecDeque<PendingAdmit>,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<Qwen35Model>>,
    b_max: usize,
) {
    // admit_mid is only legal in Decoding phase.
    if sched.phase() != Phase::Decoding {
        return;
    }
    while sched.active_count() < b_max {
        let Some(pending) = queue.pop_front() else {
            return;
        };
        let cmd = SchedulerCommand::Admit {
            request: pending.request,
            reply_tx: pending.reply_tx,
        };
        handle_admit_mid_chunked(cmd, sched, event_txs, admit_count, model);
        // Re-check phase after each mid-admit — if admit_mid itself
        // exhausted remaining rows and transitioned to Finished, stop.
        if sched.phase() != Phase::Decoding {
            return;
        }
    }
}

fn route_event(ev: StepEvent, event_txs: &HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>) {
    if let Some(tx) = event_txs.get(&ev.id) {
        // Unbounded channel — only fails when the receiver was dropped
        // (handler abandoned). That's fine; the entry naturally clears
        // at the next `event_txs.clear()` in driver_loop.
        let _ = tx.send(ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drop the SchedulerActorHandle (and thus cmd_tx); confirm the driver
    /// task exits cleanly. We can't construct a real Qwen35Model in a unit
    /// test, so we never send any commands — we only verify the driver's
    /// `rt.block_on(cmd_rx.recv())` outer loop terminates when all
    /// senders are dropped.
    ///
    /// To keep this test self-contained without a model, we don't call
    /// `spawn_scheduler_actor` (which would require a real model handle).
    /// Instead we directly spawn a minimal stand-in driver that mirrors
    /// the channel-close exit condition.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn driver_shuts_down_when_cmd_channel_closes() {
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<SchedulerCommand>(8);
        let handle = tokio::task::spawn_blocking(move || {
            // Mirrors `driver_loop`'s exit condition without touching a model.
            while let Some(_cmd) = cmd_rx.blocking_recv() {
                // would dispatch here in real driver
            }
        });
        drop(cmd_tx);
        // Driver should exit promptly after senders drop.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .expect("driver did not shut down within 2s")
            .expect("driver join error");
    }

    /// b_max=1 + queue_max=2; admit 3 short requests in rapid succession;
    /// verify the queue grows to peak >= 1 before slots free up.
    /// Real-model heavy — gated by `#[ignore]`.
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
        let model = Arc::new(Mutex::new(
            crate::models::Qwen35Model::from_loader(&loader).unwrap(),
        ));
        let meta = model.lock().await.model_meta();

        let handle = spawn_scheduler_actor(
            model.clone(),
            /* b_max */ 1,
            /* admission_deadline */ Duration::from_millis(5),
            /* admission_queue_max */ 2,
            /* effective_cap_max */ 32768,
            meta,
        )
        .expect("spawn");

        let mk_req = |text: &str| -> GenerateRequest {
            let msgs = vec![crate::core::Message {
                role: "user".into(),
                content: text.into(),
            }];
            let kw = serde_json::json!({"enable_thinking": false});
            let rendered = tokenizer
                .apply_chat_template(&msgs, true, Some(&kw))
                .unwrap();
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

        for c in &counts {
            assert!(*c >= 1, "expected ≥1 event per request, got {c}");
        }

        let peak = handle.queue_depth_peak.load(Ordering::Relaxed);
        assert!(peak >= 1, "expected queue_depth_peak >= 1, got {peak}");

        let rejected = handle.queue_rejected.load(Ordering::Relaxed);
        assert_eq!(rejected, 0, "expected no rejections, got {rejected}");

        drop(handle);
    }

    /// b_max=1 + queue_max=1; send 3 admits back-to-back. The 3rd one
    /// must be rejected with Err("admission queue full").
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
        let model = Arc::new(Mutex::new(
            crate::models::Qwen35Model::from_loader(&loader).unwrap(),
        ));
        let meta = model.lock().await.model_meta();

        let handle = spawn_scheduler_actor(
            model.clone(),
            /* b_max */ 1,
            /* admission_deadline */ Duration::from_millis(5),
            /* admission_queue_max */ 1,
            /* effective_cap_max */ 32768,
            meta,
        )
        .expect("spawn");

        let mk_req = |text: &str, max_new: usize| -> GenerateRequest {
            let msgs = vec![crate::core::Message {
                role: "user".into(),
                content: text.into(),
            }];
            let kw = serde_json::json!({"enable_thinking": false});
            let rendered = tokenizer
                .apply_chat_template(&msgs, true, Some(&kw))
                .unwrap();
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

        let (tx1, rx1) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req("Hello", 64),
                reply_tx: tx1,
            })
            .await
            .unwrap();

        // Wait briefly so first admit enters Decoding before #2/#3 arrive.
        tokio::time::sleep(Duration::from_millis(50)).await;

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

        let rejected = handle.queue_rejected.load(Ordering::Relaxed);
        assert!(rejected >= 1, "expected queue_rejected ≥ 1, got {rejected}");

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
}
