//! SchedulerActor — Tokio task wrapping [`Scheduler`] for serving HTTP
//! requests via mpsc channels.
//!
//! 3b-3 activates multi-request batching via a hybrid admission window:
//! the first admit starts a [`ADMISSION_DEADLINE`] timer; further admits
//! accumulate until either [`Scheduler::active_count`] saturates at
//! `b_max` (saturate path) or the deadline expires (hard limit, no
//! reset on new admits).
//!
//! 3c-3 introduced the rolling decode loop: after first-batch prefill
//! the driver usually biased-selects between `cmd_rx.recv()` (mid-batch
//! admit) and an always-ready step branch. Admission work marks the next
//! decode step as due, so active rows take one [`Scheduler::step`] before
//! the actor accepts more optional mid-batch admission work. Mid admits
//! route through [`Scheduler::admit_mid`] (B=1 temp-cache prefill +
//! adopt-into-main); step branch calls [`Scheduler::step`] +
//! [`Scheduler::gc_finished_rows`]. The loop exits when
//! `active_count == 0` AND `cmd_rx` is empty.
//!
//! See `docs/superpowers/specs/2026-05-13-b1-p2-3b-3-admission-window-design.md` § 4.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, Mutex};

use crate::core::generate::GenerateRequest;
use crate::core::model::Model;
use crate::core::scheduler::{
    AdmitMidHandle, DenseVlMethods, Phase, RequestId, Scheduler, StepEvent,
};
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
/// slot, then handed to the rolling mid-admit chunk path.
struct PendingAdmit {
    request: GenerateRequest,
    reply_tx: oneshot::Sender<Result<AdmitReply>>,
}

fn fresh_prefill_batch_limit_for_request<M: Model>(
    request: &GenerateRequest,
    b_max: usize,
) -> usize {
    M::fresh_prefill_batch_limit(request.prompt_ids.len(), b_max).clamp(1, b_max)
}

fn fresh_prefill_batch_limit_for_command<M: Model>(cmd: &SchedulerCommand, b_max: usize) -> usize {
    let SchedulerCommand::Admit { request, .. } = cmd;
    fresh_prefill_batch_limit_for_request::<M>(request, b_max)
}

/// Event yielded by the rolling decode loop. Either a new admit command
/// arrived (mid-batch admit), a decode step is due, or the cmd_rx channel
/// was closed (shutdown).
#[allow(clippy::large_enum_variant)] // Admit(SchedulerCommand) intentionally large; boxing would add allocation on hot path
enum RollingEvent {
    Admit(SchedulerCommand),
    AdvanceMidAdmit,
    Step,
    Shutdown,
}

/// Rolling-loop admission fairness policy.
///
/// Any completed admission work (initial prefill or mid-batch prefill)
/// makes one decode step due before the actor accepts more optional
/// admission work. This bounds active streams' token gap under sustained
/// arrivals while keeping the existing initial admission window and FIFO
/// queue semantics intact.
#[derive(Debug, Default)]
struct RollingAdmissionPolicy {
    decode_due_after_admission: bool,
}

impl RollingAdmissionPolicy {
    fn record_admission_work(&mut self) {
        self.decode_due_after_admission = true;
    }

    fn record_decode_step(&mut self) {
        self.decode_due_after_admission = false;
    }

    fn should_force_decode(&self, phase: Phase, active_count: usize) -> bool {
        self.decode_due_after_admission && phase == Phase::Decoding && active_count > 0
    }
}

/// P5h+2.c regression counter: incremented every time the actor's Step
/// branch observes an `Err` whose Debug output contains
/// `step illegal in Finished phase`. The integration test
/// `ironmlx/tests/p5h_2c_scheduler_finished_smoke.rs` resets this counter
/// at test start, runs 3× `max_new_tokens=1` admit cmds, and asserts it
/// stays at 0 (proving the pre-event finalization hook eliminated the
/// bug surface).
///
/// Gated under `cfg(feature = "p5h-profile")` so default release builds
/// pay zero cost. `cfg(test)` items are not visible to `ironmlx/tests/*`
/// integration targets (library compiles as a dependency), which is why
/// the feature flag is required instead of `cfg(test)`.
#[cfg(feature = "p5h-profile")]
#[doc(hidden)]
pub static STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Result returned by [`drive_empty_scheduler_handoff`] encoding what the
/// caller's rolling loop should do next. Matches the existing `continue
/// 'rolling` / `break 'rolling` / `continue 'outer` / `return` patterns
/// without exposing label control to the helper.
///
/// Added by P5h+2.c to make the empty-batch handoff path reusable from
/// (a) the existing post-step empty-handoff site and (b) the new
/// pre-event Finished-batch finalization at the rolling-loop top.
enum RollingControl {
    /// Re-enter the rolling loop (a new batch was admitted + prefilled).
    ContinueRolling,
    /// Exit the rolling loop into the outer-loop tail cleanup (no
    /// queued or pending admits; outer will block on `cmd_rx.recv()`).
    BreakRolling,
    /// `continue 'outer` — outer loop body resumes from its top
    /// (e.g., poisoned-state recovery).
    ContinueOuter,
    /// `return` from the actor (cmd_rx disconnected; all senders dropped).
    ReturnActor,
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
pub fn spawn_scheduler_actor<M>(
    model: Arc<Mutex<M>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    meta: crate::core::memory_budget::ModelMeta,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    // ── Step 1: Budget validation on the calling thread. ──────────────────
    // No Scheduler / Array is allocated here — just pure arithmetic + RAM
    // check. Returns Err early if the budget is too tight.
    let budget_state =
        crate::core::memory_budget::validate_startup_budget(b_max, effective_cap_max, &meta)?;

    // ── Step 2: Shared atomics created on the calling thread. ─────────────
    // Cloned for both the handle (returned to caller) and the driver thread.
    // This is the single source of truth — handle + driver share the same
    // Arc instances, so /healthz reads the live values the driver updates.
    //
    // B1-p2.5 P0 fix v2: the previous fix (c043ce9) created Scheduler::new
    // on the calling thread then moved it into spawn_blocking. That caused
    // "MLX runtime error: There is no Stream(gpu, N) in current thread"
    // because Array::zeros (prng_state) was bound to the calling thread's
    // Metal Stream. This fix keeps budget validation + Arc creation on the
    // calling thread while deferring Scheduler::new_with_state (and thus
    // Array allocation) to the spawn_blocking worker thread.
    let memory_budget_exceeded_count = Arc::new(AtomicU64::new(0));

    // Healthz observables cloned from BudgetState (Arc<AtomicUsize> inside).
    let kv_cache_active_bytes = budget_state.shared_active();
    let kv_cache_soft_limit_bytes = budget_state.soft_limit();

    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let admit_count = Arc::new(AtomicU64::new(0));
    let batch_count = Arc::new(AtomicU64::new(0));
    let saturate_triggered = Arc::new(AtomicU64::new(0));
    let queue_depth_peak = Arc::new(AtomicUsize::new(0));
    let queue_rejected = Arc::new(AtomicU64::new(0));
    // B1-p2.5 G3: live b_active / b_queued updated by driver_loop tail.
    let b_active = Arc::new(AtomicU64::new(0));
    let b_queued = Arc::new(AtomicU64::new(0));

    // Clone Arcs for the driver thread.
    let driver_budget_state = budget_state.clone();
    let driver_mb_exceeded = memory_budget_exceeded_count.clone();
    let admit_count_for_task = admit_count.clone();
    let batch_count_for_task = batch_count.clone();
    let saturate_triggered_for_task = saturate_triggered.clone();
    let queue_depth_peak_for_task = queue_depth_peak.clone();
    let queue_rejected_for_task = queue_rejected.clone();
    let b_active_for_task = b_active.clone();
    let b_queued_for_task = b_queued.clone();

    // ── Step 3: Spawn driver — Scheduler::new_with_state constructed INSIDE
    //    spawn_blocking so MLX Array fields (prng_state) are allocated on the
    //    worker thread's Metal Stream. Thread affinity preserved.
    tokio::task::spawn_blocking(move || {
        let scheduler = Scheduler::<M>::new_with_state(
            b_max,
            effective_cap_max,
            driver_budget_state,
            driver_mb_exceeded,
            meta,
        )
        .expect("budget already validated above; new_with_state must not fail");
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
fn driver_loop<M>(
    scheduler: Scheduler<M>,
    model: Arc<Mutex<M>>,
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
) where
    M: Model + DenseVlMethods + Send + 'static,
{
    // Receive Scheduler ownership from spawn_scheduler_actor (single instance).
    // P0 fix: previously driver_loop called Scheduler::new a second time,
    // creating fresh Arc atomics disconnected from the handle. B1-p2.5.
    let mut sched = scheduler;
    let b_max = sched.b_max();
    let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
    let mut admission_queue: VecDeque<PendingAdmit> = VecDeque::new();
    let mut in_flight_mid_admit: Option<AdmitMidHandle> = None;
    let rt = tokio::runtime::Handle::current();

    'outer: loop {
        // P5h+2.c defensive: ensure scheduler is in Phase::Idle before
        // blocking on next admit. Most error paths already call evict_all,
        // but this guards any future code path that leaves phase=Finished.
        // If finalize fails, the actor cannot safely admit more requests
        // (the scheduler would be in an unrecoverable state); terminate
        // cleanly rather than emit ERROR per request.
        if sched.phase() == Phase::Finished {
            if let Err(e) = finalize_finished_batch_if_any(&mut sched, &mut event_txs) {
                tracing::error!(
                    "[SchedulerActor] outer-loop finalize failed: {e:?}; \
                     actor cannot reset Finished batch safely — terminating"
                );
                event_txs.clear();
                return;
            }
        }

        // ===== Outer Idle: block waiting for first admit (or shutdown). =====
        // Outer Idle is reached only after evict_all clears all slots; the
        // admission queue is invariantly empty here (any queue elements were
        // drained inside the rolling loop before reaching this point).
        let Some(first_cmd) = rt.block_on(cmd_rx.recv()) else {
            return; // cmd_rx closed; all senders dropped.
        };
        let fresh_batch_limit = fresh_prefill_batch_limit_for_command::<M>(&first_cmd, b_max);
        handle_admit(first_cmd, &mut sched, &mut event_txs, &admit_count);

        if sched.active_count() == 0 {
            // First admit failed (Err) — nothing to prefill. Wait for next.
            continue 'outer;
        }

        // ===== Admission window: drain additional admits until deadline
        //       or the model's fresh-prefill batch limit. Beyond the limit,
        //       push to admission_queue (bounded by admission_queue_max). =====
        if sched.active_count() < fresh_batch_limit {
            rt.block_on(drain_window(
                &mut cmd_rx,
                &mut sched,
                &mut event_txs,
                &mut admission_queue,
                &admit_count,
                &saturate_triggered,
                &queue_depth_peak,
                &queue_rejected,
                fresh_batch_limit,
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

        // ===== Rolling decode loop with bounded mid-batch admit + queue drain. =====
        let mut admission_policy = RollingAdmissionPolicy::default();
        admission_policy.record_admission_work();
        'rolling: loop {
            // P5h+2.c: pre-event Finished-batch finalization + handoff. If
            // previous iteration's prefill_admitted/step left phase=Finished
            // (e.g. max_tokens=1 workload), handle the completed batch BEFORE
            // dispatching another event. Per Codex Q6: biased select may pick
            // Admit over Step, so this must run before the event pick — or the
            // actor could call admit_mid_begin() in Phase::Finished.
            //
            // `drive_empty_scheduler_handoff` itself calls
            // `finalize_finished_batch_if_any`; do not duplicate finalization
            // here. This avoids two divergent finalize/error paths.
            if sched.phase() == Phase::Finished {
                match drive_empty_scheduler_handoff(
                    &mut sched,
                    &mut cmd_rx,
                    &mut event_txs,
                    &mut admission_queue,
                    &model,
                    &admit_count,
                    &saturate_triggered,
                    &queue_depth_peak,
                    &queue_rejected,
                    &batch_count,
                    b_max,
                    admission_queue_max,
                    admission_deadline,
                    &rt,
                ) {
                    RollingControl::ContinueRolling => {
                        admission_policy.record_admission_work();
                        continue 'rolling;
                    }
                    RollingControl::BreakRolling => break 'rolling,
                    RollingControl::ContinueOuter => continue 'outer,
                    RollingControl::ReturnActor => return,
                }
            }

            let evt: RollingEvent =
                if admission_policy.should_force_decode(sched.phase(), sched.active_count()) {
                    RollingEvent::Step
                } else if in_flight_mid_admit.is_some() {
                    RollingEvent::AdvanceMidAdmit
                } else {
                    rt.block_on(async {
                        tokio::select! {
                            biased;
                            maybe_cmd = cmd_rx.recv() => match maybe_cmd {
                                Some(cmd) => RollingEvent::Admit(cmd),
                                None => RollingEvent::Shutdown,
                            },
                            _ = std::future::ready(()) => RollingEvent::Step,
                        }
                    })
                };

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
                    if !can_start_rolling_mid_admit_for_command::<M>(
                        &cmd,
                        sched.active_count(),
                        b_max,
                    ) {
                        // Rolling admission limit reached — queue for a later decode turn.
                        enqueue_or_reject(
                            cmd,
                            &mut admission_queue,
                            admission_queue_max,
                            &queue_depth_peak,
                            &queue_rejected,
                        );
                    } else if start_mid_admit_one_chunk(
                        cmd,
                        &mut in_flight_mid_admit,
                        &mut sched,
                        &mut event_txs,
                        &admit_count,
                        &model,
                    ) {
                        admission_policy.record_admission_work();
                    }
                }
                RollingEvent::AdvanceMidAdmit => {
                    if advance_mid_admit_one_chunk(
                        &mut in_flight_mid_admit,
                        &mut sched,
                        &mut event_txs,
                        &admit_count,
                        &model,
                    ) {
                        admission_policy.record_admission_work();
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
                            admission_policy.record_decode_step();
                            // ===== Post-gc queue drain. =====
                            // Free slots → pull from admission_queue head
                            // for one bounded mid-admit. Further queued
                            // requests wait for the next decode turn so
                            // active streams keep making progress.
                            if in_flight_mid_admit.is_none()
                                && drain_admission_queue(
                                    &mut admission_queue,
                                    &mut in_flight_mid_admit,
                                    &mut sched,
                                    &mut event_txs,
                                    &admit_count,
                                    &model,
                                    b_max,
                                )
                            {
                                admission_policy.record_admission_work();
                            }
                        }
                        Err(e) => {
                            tracing::error!("[SchedulerActor] step error: {e:?}");
                            #[cfg(feature = "p5h-profile")]
                            if format!("{e:?}").contains("step illegal in Finished phase") {
                                STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                            if let Err(evict_err) = sched.evict_all() {
                                tracing::warn!(
                                    "[SchedulerActor] evict_all after step error also failed: \
                                     {evict_err:?}; relying on 3b-1 poison flag to reject subsequent admits"
                                );
                            }
                            in_flight_mid_admit = None;
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
            //
            // P5h+2.c: extracted into `drive_empty_scheduler_handoff` so the
            // same logic backs the pre-event Finished-batch finalization hook
            // at the rolling-loop top. The helper finalizes any leftover
            // `Phase::Finished` state first, then performs the queued-admit
            // / try_recv / break handoff.
            if sched.active_count() == 0 {
                match drive_empty_scheduler_handoff(
                    &mut sched,
                    &mut cmd_rx,
                    &mut event_txs,
                    &mut admission_queue,
                    &model,
                    &admit_count,
                    &saturate_triggered,
                    &queue_depth_peak,
                    &queue_rejected,
                    &batch_count,
                    b_max,
                    admission_queue_max,
                    admission_deadline,
                    &rt,
                ) {
                    RollingControl::ContinueRolling => {
                        admission_policy.record_admission_work();
                        continue 'rolling;
                    }
                    RollingControl::BreakRolling => break 'rolling,
                    RollingControl::ContinueOuter => continue 'outer,
                    RollingControl::ReturnActor => return,
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
        in_flight_mid_admit = None;
        event_txs.clear();
    }
}

/// Drain additional `Admit` commands until either the deadline expires or the
/// fresh-batch admission limit is reached. Hard deadline — new admits do NOT
/// reset the timer. Once the limit is reached, additional admits within the
/// window go to the admission queue (bounded by `admission_queue_max`).
#[allow(clippy::too_many_arguments)]
async fn drain_window<M>(
    cmd_rx: &mut mpsc::Receiver<SchedulerCommand>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admission_queue: &mut VecDeque<PendingAdmit>,
    admit_count: &Arc<AtomicU64>,
    saturate_triggered: &Arc<AtomicU64>,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
    batch_limit: usize,
    b_max: usize,
    queue_max: usize,
    deadline: Duration,
) where
    M: Model + DenseVlMethods + Send + 'static,
{
    let batch_limit = batch_limit.clamp(1, b_max);
    let timer = tokio::time::sleep(deadline);
    tokio::pin!(timer);
    let mut limit_reached = false;
    loop {
        tokio::select! {
            biased;
            _ = &mut timer => return,
            maybe = cmd_rx.recv() => {
                let Some(cmd) = maybe else { return }; // channel closed
                if limit_reached {
                    // Fresh batch is full for this model/prompt policy — push
                    // to queue or reject.
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
                if sched.active_count() >= batch_limit {
                    if batch_limit >= b_max {
                        saturate_triggered.fetch_add(1, Ordering::Relaxed);
                    }
                    limit_reached = true;
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
fn handle_admit<M>(
    cmd: SchedulerCommand,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
) where
    M: Model + DenseVlMethods + Send + 'static,
{
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

fn uses_multi_chunk_prefill(request: &GenerateRequest) -> bool {
    request.prefill_chunk_size > 0 && request.prompt_ids.len() > request.prefill_chunk_size
}

fn can_start_rolling_mid_admit_for_request<M: Model>(
    request: &GenerateRequest,
    active_count: usize,
    b_max: usize,
) -> bool {
    if active_count >= b_max {
        return false;
    }
    let rolling_limit = fresh_prefill_batch_limit_for_request::<M>(request, b_max);
    active_count < rolling_limit || uses_multi_chunk_prefill(request)
}

fn can_start_rolling_mid_admit_for_command<M: Model>(
    cmd: &SchedulerCommand,
    active_count: usize,
    b_max: usize,
) -> bool {
    let SchedulerCommand::Admit { request, .. } = cmd;
    can_start_rolling_mid_admit_for_request::<M>(request, active_count, b_max)
}

fn begin_mid_admit<M>(
    cmd: SchedulerCommand,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    model: &Arc<Mutex<M>>,
) -> Option<AdmitMidHandle>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    // Phase 1: begin.
    let handle = {
        let m = model.blocking_lock();
        match sched.admit_mid_begin(request, &m) {
            Ok(h) => h,
            Err(e) => {
                let _ = reply_tx.send(Err(e));
                return None;
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
        // Caller dropped reply_rx before the prefill chunks completed.
        let _ = sched.evict(id);
        event_txs.remove(&id);
        return None;
    }

    Some(handle)
}

fn start_mid_admit_one_chunk<M>(
    cmd: SchedulerCommand,
    in_flight_mid_admit: &mut Option<AdmitMidHandle>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<M>>,
) -> bool
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if in_flight_mid_admit.is_some() {
        return false;
    }
    let Some(handle) = begin_mid_admit(cmd, sched, event_txs, model) else {
        return false;
    };
    *in_flight_mid_admit = Some(handle);
    advance_mid_admit_one_chunk(in_flight_mid_admit, sched, event_txs, admit_count, model)
}

fn advance_mid_admit_one_chunk<M>(
    in_flight_mid_admit: &mut Option<AdmitMidHandle>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<M>>,
) -> bool
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let Some(mut handle) = in_flight_mid_admit.take() else {
        return false;
    };
    let id = handle.request_id;

    let is_last = {
        let m = model.blocking_lock();
        match sched.admit_mid_chunk(&mut handle, &m) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("[SchedulerActor] admit_mid_chunk error: {e:?}");
                let _ = sched.evict(id);
                event_txs.remove(&id);
                return true;
            }
        }
    };

    if !is_last {
        *in_flight_mid_admit = Some(handle);
        return true;
    }

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
    true
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

/// Drain at most one mid-batch admit chunk from the admission queue.
/// Full-prompt rolling admits obey the model's `fresh_prefill_batch_limit`.
/// Multi-chunk admits may start in a spare slot beyond that limit because
/// each chunk yields back to the rolling loop before the next chunk runs.
/// Once any admission work happens the caller must return to decode before
/// draining more queue.
///
/// IMPORTANT: `admit_mid` is only legal in `Decoding` phase. If
/// `gc_finished_rows` just transitioned the scheduler to `Finished`
/// (because `active_count` dropped to 0), the caller's rolling-loop
/// exit branch (`active_count == 0 && queue non-empty`) will handle the
/// queued entries via `evict_all` + fresh `prefill_admitted`. Return
/// early here so we do not call `admit_mid` in an illegal phase.
fn drain_admission_queue<M>(
    queue: &mut VecDeque<PendingAdmit>,
    in_flight_mid_admit: &mut Option<AdmitMidHandle>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<M>>,
    b_max: usize,
) -> bool
where
    M: Model + DenseVlMethods + Send + 'static,
{
    // admit_mid is only legal in Decoding phase.
    if sched.phase() != Phase::Decoding {
        return false;
    }
    if in_flight_mid_admit.is_some() {
        return false;
    }
    while sched.active_count() < b_max {
        let Some(pending) = queue.front() else {
            return false;
        };
        if !can_start_rolling_mid_admit_for_request::<M>(
            &pending.request,
            sched.active_count(),
            b_max,
        ) {
            return false;
        }
        let pending = queue
            .pop_front()
            .expect("queue.front returned Some immediately before pop_front");
        let cmd = SchedulerCommand::Admit {
            request: pending.request,
            reply_tx: pending.reply_tx,
        };
        let did_admission_work = start_mid_admit_one_chunk(
            cmd,
            in_flight_mid_admit,
            sched,
            event_txs,
            admit_count,
            model,
        );
        // Re-check phase after each mid-admit — if admit_mid itself
        // exhausted remaining rows and transitioned to Finished, stop.
        if did_admission_work || sched.phase() != Phase::Decoding {
            return did_admission_work;
        }
    }
    false
}

fn route_event(ev: StepEvent, event_txs: &HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>) {
    if let Some(tx) = event_txs.get(&ev.id) {
        // Unbounded channel — only fails when the receiver was dropped
        // (handler abandoned). That's fine; the entry naturally clears
        // at the next `event_txs.clear()` in driver_loop.
        let _ = tx.send(ev);
    }
}

/// Finalize a `Phase::Finished` batch: evict slots + release budget +
/// reset to `Phase::Idle`, then close per-request event channels.
///
/// Returns `Ok(true)` if finalization happened (caller MUST go to the
/// empty-scheduler handoff path, NOT continue the normal event pick;
/// per spec § 4.2.1 hard binding).
/// This binding applies in the rolling-loop context; the outer-loop hook calls this directly because admission_queue is invariantly empty at that point.
/// Returns `Ok(false)` if `phase != Finished` (no-op; safe to continue).
/// Returns `Err` if `evict_all` failed (caller should reject queued
/// admits + `continue 'outer` per existing pattern).
///
/// Added by P5h+2.c. The `Phase::Finished` state arises naturally when
/// `prefill_admitted` completes a batch where every request has
/// `max_new_tokens=1` (the prefill samples first+last token in one
/// pass), which is the standard `iron-bench --max-tokens 1` perf
/// measurement workload.
fn finalize_finished_batch_if_any<M: Model>(
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
) -> Result<bool> {
    if sched.phase() != Phase::Finished {
        return Ok(false);
    }
    match sched.evict_all() {
        Ok(()) => {
            event_txs.clear();
            Ok(true)
        }
        Err(e) => {
            tracing::warn!("[SchedulerActor] finalize_finished_batch: evict_all failed: {e:?}");
            Err(e)
        }
    }
}

/// Finalize a just-finished batch if needed, then drain queued admits
/// (or a single pending `cmd_rx.try_recv` admit) into a fresh batch, run
/// `prefill_admitted`, and return how the caller's rolling loop should
/// proceed. Lifts the existing empty-batch transition logic at the
/// rolling-loop tail so it can also be invoked from the new pre-event
/// Finished-batch finalization at the rolling-loop top.
///
/// This helper is the single empty-batch handoff path. It first calls
/// [`finalize_finished_batch_if_any`], so callers must not separately
/// finalize before invoking it. After that call the scheduler is either
/// `Phase::Idle` or `Phase::Decoding` (the legacy post-step empty-handoff
/// path used to encounter `Phase::Finished`; the new pre-event hook now
/// shoulders that case via finalize). The helper preserves the current
/// reset semantics for `Decoding`-with-zero-active-rows before starting
/// the next batch but never calls `evict_all` in `Idle` (which is itself
/// an error per scheduler.rs:775-780).
///
/// Behavior per branch:
/// - Queued admit present → pop head, fresh batch via `handle_admit` +
///   `drain_window` + `prefill_admitted`; returns `ContinueRolling`.
/// - Queue empty + `cmd_rx.try_recv()` returns `Ok(cmd)` → fresh batch
///   via the same path; returns `ContinueRolling`.
/// - Queue empty + `try_recv` returns `Empty` → returns `BreakRolling`.
/// - Queue empty + `try_recv` returns `Disconnected` → clear `event_txs`,
///   returns `ReturnActor`.
/// - Any `finalize`, legacy reset, or `prefill_admitted` failure →
///   reject queued admits, clear `event_txs`, returns `ContinueOuter`.
///
/// Added by P5h+2.c. Replaces the existing `if sched.active_count() == 0
/// { ... }` block at rolling-loop tail to avoid divergent copies.
#[allow(clippy::too_many_arguments)]
fn drive_empty_scheduler_handoff<M>(
    sched: &mut Scheduler<M>,
    cmd_rx: &mut mpsc::Receiver<SchedulerCommand>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admission_queue: &mut VecDeque<PendingAdmit>,
    model: &Arc<Mutex<M>>,
    admit_count: &Arc<AtomicU64>,
    saturate_triggered: &Arc<AtomicU64>,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
    batch_count: &Arc<AtomicU64>,
    b_max: usize,
    admission_queue_max: usize,
    admission_deadline: Duration,
    rt: &tokio::runtime::Handle,
) -> RollingControl
where
    M: Model + DenseVlMethods + Send + 'static,
{
    // P5h+2.c: finalize any Finished batch BEFORE re-admitting. After
    // this, phase is one of {Idle, Decoding}; never Finished. Callers
    // must not separately finalize.
    match finalize_finished_batch_if_any(sched, event_txs) {
        Ok(_) => {}
        Err(_e) => {
            while let Some(pending) = admission_queue.pop_front() {
                let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                    "scheduler poisoned during Finished-batch finalize"
                )));
            }
            event_txs.clear();
            return RollingControl::ContinueOuter;
        }
    }

    if !admission_queue.is_empty() {
        // Reset Decoding-with-zero-active-rows to Idle for fresh batch.
        // (Finished was already handled by finalize above; Idle would
        // itself be an error for `evict_all`.)
        if sched.phase() == Phase::Decoding {
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
                return RollingControl::ContinueOuter;
            }
            event_txs.clear();
        }
        // Pop first queued admit as the new batch's first admit.
        let pending = admission_queue
            .pop_front()
            .expect("queue non-empty checked");
        let fresh_batch_limit = fresh_prefill_batch_limit_for_request::<M>(&pending.request, b_max);
        handle_admit(
            SchedulerCommand::Admit {
                request: pending.request,
                reply_tx: pending.reply_tx,
            },
            sched,
            event_txs,
            admit_count,
        );
        if sched.active_count() == 0 {
            // Admit failed; loop to drain more queue (or exit).
            return RollingControl::ContinueRolling;
        }
        if sched.active_count() < fresh_batch_limit {
            // Drain queue head-by-head into the new batch (no deadline —
            // these are already-queued admits, not racing-in cmd_rx).
            // Then optionally drain_window for fresh cmd_rx admits.
            while sched.active_count() < fresh_batch_limit {
                let Some(p) = admission_queue.pop_front() else {
                    break;
                };
                handle_admit(
                    SchedulerCommand::Admit {
                        request: p.request,
                        reply_tx: p.reply_tx,
                    },
                    sched,
                    event_txs,
                    admit_count,
                );
            }
            // Optionally absorb cmd_rx admits arriving right now.
            if sched.active_count() < fresh_batch_limit {
                rt.block_on(drain_window(
                    cmd_rx,
                    sched,
                    event_txs,
                    admission_queue,
                    admit_count,
                    saturate_triggered,
                    queue_depth_peak,
                    queue_rejected,
                    fresh_batch_limit,
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
                    route_event(ev, event_txs);
                }
            }
            Err(e) => {
                tracing::error!("[SchedulerActor] re-prefill (queue drain) error: {e:?}");
                if let Err(evict_err) = sched.evict_all() {
                    tracing::warn!(
                        "[SchedulerActor] evict_all after re-prefill error also failed: \
                         {evict_err:?}; rejecting remaining queued admits"
                    );
                }
                event_txs.clear();
                while let Some(p) = admission_queue.pop_front() {
                    let _ = p.reply_tx.send(Err(anyhow::anyhow!(
                        "scheduler poisoned after re-prefill error"
                    )));
                }
                return RollingControl::ContinueOuter;
            }
        }
        return RollingControl::ContinueRolling;
    }
    // Queue empty + no active rows — same logic as pre-3d.
    match cmd_rx.try_recv() {
        Ok(cmd) => {
            let fresh_batch_limit = fresh_prefill_batch_limit_for_command::<M>(&cmd, b_max);
            if sched.phase() == Phase::Decoding {
                if let Err(evict_err) = sched.evict_all() {
                    tracing::warn!(
                        "[SchedulerActor] evict_all between batches failed: \
                         {evict_err:?}; rejecting incoming admit"
                    );
                    let SchedulerCommand::Admit { reply_tx, .. } = cmd;
                    let _ = reply_tx.send(Err(evict_err));
                    event_txs.clear();
                    return RollingControl::ContinueOuter;
                }
                event_txs.clear();
            }
            handle_admit(cmd, sched, event_txs, admit_count);
            if sched.active_count() == 0 {
                return RollingControl::BreakRolling;
            }
            if sched.active_count() < fresh_batch_limit {
                rt.block_on(drain_window(
                    cmd_rx,
                    sched,
                    event_txs,
                    admission_queue,
                    admit_count,
                    saturate_triggered,
                    queue_depth_peak,
                    queue_rejected,
                    fresh_batch_limit,
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
                        route_event(ev, event_txs);
                    }
                }
                Err(e) => {
                    tracing::error!("[SchedulerActor] re-prefill error: {e:?}");
                    if let Err(evict_err) = sched.evict_all() {
                        tracing::warn!(
                            "[SchedulerActor] evict_all after re-prefill error also failed: \
                             {evict_err:?}; relying on 3b-1 poison flag to reject subsequent admits"
                        );
                    }
                    event_txs.clear();
                    return RollingControl::ContinueOuter;
                }
            }
            RollingControl::ContinueRolling
        }
        Err(tokio::sync::mpsc::error::TryRecvError::Empty) => RollingControl::BreakRolling,
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
            event_txs.clear();
            RollingControl::ReturnActor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
    use crate::core::sampler::Sampler;

    struct SchedulerActorFakeModel;

    impl Model for SchedulerActorFakeModel {
        fn make_cache(
            &self,
            batch: i32,
            cap: i32,
            dtype: mlx::Dtype,
        ) -> Result<Vec<crate::nn::LayerCache>> {
            Ok(vec![crate::nn::LayerCache::Full(
                crate::core::KVCache::new(batch, 1, 1, 1, dtype, cap),
            )])
        }

        fn forward_on(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            fake_logits(input_ids.shape().as_slice()[0] as usize)
        }

        fn batched_prefill(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            _per_row_lens: &[i32],
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            fake_logits(input_ids.shape().as_slice()[0] as usize)
        }

        fn forward_text_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            let shape = input_ids.shape();
            let b = shape.as_slice()[0] as usize;
            let s = shape.as_slice()[1] as usize;
            let hidden = 4_usize;
            let flat = vec![0.0_f32; b * s * hidden];
            (&flat[..], &[b as i32, s as i32, hidden as i32][..])
                .try_into()
                .map_err(|e| anyhow::anyhow!("fake hidden Array failed: {e:?}"))
        }

        fn fresh_prefill_batch_limit(_prompt_len: usize, b_max: usize) -> usize
        where
            Self: Sized,
        {
            b_max.min(2)
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    impl DenseVlMethods for SchedulerActorFakeModel {
        fn batched_prefill_vl(
            &self,
            _input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            _per_row_lens: &[i32],
            _per_row_pixel_values: &[Option<&[mlx::Array]>],
            _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
            _image_token_id: i32,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            unreachable!("scheduler_actor policy unit tests are text-only")
        }

        fn compute_vision_embeds(
            &self,
            _pixel_values: &[mlx::Array],
            _grid_thw: &[(i32, i32, i32)],
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            unreachable!("scheduler_actor policy unit tests are text-only")
        }

        fn forward_vl_chunk(
            &self,
            _input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            unreachable!("scheduler_actor policy unit tests are text-only")
        }

        fn forward_vl_hidden(
            &self,
            _input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            unreachable!("scheduler_actor policy unit tests are text-only")
        }
    }

    fn fake_logits(batch: usize) -> Result<mlx::Array> {
        let vocab = 8_usize;
        let mut flat = vec![0.0_f32; batch * vocab];
        for row in 0..batch {
            flat[row * vocab + 3] = 100.0;
        }
        let logits_bv: mlx::Array = (&flat[..], &[batch as i32, vocab as i32][..])
            .try_into()
            .map_err(|e| anyhow::anyhow!("fake logits Array failed: {e:?}"))?;
        logits_bv
            .reshape(&[batch as i32, 1, vocab as i32][..])
            .map_err(|e| anyhow::anyhow!("fake logits reshape failed: {e:?}"))
    }

    fn mk_req(prompt_token: u32) -> GenerateRequest {
        GenerateRequest {
            prompt_ids: vec![prompt_token],
            max_new_tokens: 16,
            sampler: Sampler::greedy(),
            stop_token_ids: vec![2],
            prefill_chunk_size: 0,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: IMAGE_TOKEN_ID,
            #[cfg(feature = "p5h-profile")]
            p5h_trace: None,
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: None,
        }
    }

    fn queued_pending(prompt_token: u32) -> (PendingAdmit, oneshot::Receiver<Result<AdmitReply>>) {
        let (reply_tx, reply_rx) = oneshot::channel();
        (
            PendingAdmit {
                request: mk_req(prompt_token),
                reply_tx,
            },
            reply_rx,
        )
    }

    #[test]
    fn rolling_policy_forces_one_decode_after_admission_work() {
        let mut policy = RollingAdmissionPolicy::default();

        assert!(!policy.should_force_decode(Phase::Decoding, 1));
        policy.record_admission_work();
        assert!(policy.should_force_decode(Phase::Decoding, 1));

        policy.record_decode_step();
        assert!(!policy.should_force_decode(Phase::Decoding, 1));
    }

    #[test]
    fn rolling_policy_does_not_force_decode_without_active_decoding_rows() {
        let mut policy = RollingAdmissionPolicy::default();

        policy.record_admission_work();
        assert!(!policy.should_force_decode(Phase::Idle, 1));
        assert!(!policy.should_force_decode(Phase::Finished, 1));
        assert!(!policy.should_force_decode(Phase::Decoding, 0));
    }

    #[test]
    fn drain_admission_queue_limits_successful_mid_admit_to_one_per_turn() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = Scheduler::<SchedulerActorFakeModel>::new(
            4,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        sched.admit(mk_req(11)).expect("initial admit");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 1);
        assert_eq!(sched.phase(), Phase::Decoding);

        let (pending_1, reply_rx_1) = queued_pending(21);
        let (pending_2, reply_rx_2) = queued_pending(22);
        let (pending_3, reply_rx_3) = queued_pending(23);
        let _reply_rxs = [reply_rx_1, reply_rx_2, reply_rx_3];
        let mut queue = VecDeque::from([pending_1, pending_2, pending_3]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            4,
        );

        assert!(did_admit, "expected one queued request to be admitted");
        assert_eq!(
            queue.len(),
            2,
            "queue drain should leave remaining queued requests for later decode turns"
        );
        assert_eq!(
            sched.active_count(),
            2,
            "one active row plus exactly one mid-admitted row"
        );
        assert!(in_flight_mid_admit.is_none());
    }

    #[test]
    fn drain_admission_queue_respects_rolling_prefill_batch_limit() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = Scheduler::<SchedulerActorFakeModel>::new(
            4,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        sched.admit(mk_req(11)).expect("initial admit 1");
        sched.admit(mk_req(12)).expect("initial admit 2");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 2);
        assert_eq!(sched.phase(), Phase::Decoding);
        assert_eq!(sched.active_count(), 2);

        let (pending_1, reply_rx_1) = queued_pending(21);
        let (pending_2, reply_rx_2) = queued_pending(22);
        let _reply_rxs = [reply_rx_1, reply_rx_2];
        let mut queue = VecDeque::from([pending_1, pending_2]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            4,
        );

        assert!(
            !did_admit,
            "active rows already reached the model's rolling prefill batch limit"
        );
        assert_eq!(
            queue.len(),
            2,
            "queued requests should wait for decode progress instead of growing active batch"
        );
        assert_eq!(sched.active_count(), 2);
        assert_eq!(admit_count.load(Ordering::Relaxed), 0);
        assert!(in_flight_mid_admit.is_none());
    }

    #[test]
    fn drain_admission_queue_starts_chunked_mid_admit_beyond_rolling_limit() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = Scheduler::<SchedulerActorFakeModel>::new(
            4,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        sched.admit(mk_req(11)).expect("initial admit 1");
        sched.admit(mk_req(12)).expect("initial admit 2");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 2);
        assert_eq!(sched.phase(), Phase::Decoding);
        assert_eq!(sched.active_count(), 2);

        let (reply_tx, reply_rx) = oneshot::channel();
        let mut chunked_req = mk_req(21);
        chunked_req.prompt_ids = vec![21, 22, 23, 24];
        chunked_req.prefill_chunk_size = 2;
        let _reply_rx = reply_rx;
        let mut queue = VecDeque::from([PendingAdmit {
            request: chunked_req,
            reply_tx,
        }]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            4,
        );

        assert!(
            did_admit,
            "chunked queued requests may start prefill under decode-cadence protection"
        );
        assert_eq!(queue.len(), 0);
        assert_eq!(sched.active_count(), 3);
        assert_eq!(
            admit_count.load(Ordering::Relaxed),
            0,
            "the request should not count as admitted until the final chunk samples its first token"
        );
        assert!(
            in_flight_mid_admit.is_some(),
            "multi-chunk mid-admit should yield after one chunk"
        );
    }

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
                #[cfg(feature = "p5h-profile")]
                p5h_trace: None,
                #[cfg(feature = "p5h-profile")]
                p5h_root_span: None,
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
                #[cfg(feature = "p5h-profile")]
                p5h_trace: None,
                #[cfg(feature = "p5h-profile")]
                p5h_root_span: None,
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
