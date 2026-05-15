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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, Mutex};

use crate::core::generate::GenerateRequest;
use crate::core::scheduler::{RequestId, Scheduler, StepEvent};
use crate::models::Qwen35Model;
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
    /// Submit a request for batched generation. On success, replies with
    /// the admitted [`RequestId`] and an mpsc receiver that streams
    /// [`StepEvent`]s (one per produced token, until `finish_reason`
    /// becomes `Some(_)` on the final event for this row).
    Admit {
        request: GenerateRequest,
        reply_tx: oneshot::Sender<Result<AdmitReply>>,
    },
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
}

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

    'outer: loop {
        // ===== Outer Idle: block waiting for first admit (or shutdown). =====
        let Some(first_cmd) = rt.block_on(cmd_rx.recv()) else {
            return; // cmd_rx closed; all senders dropped.
        };
        handle_admit(first_cmd, &mut sched, &mut event_txs, &admit_count);

        if sched.active_count() == 0 {
            // First admit failed (Err) — nothing to prefill. Wait for next.
            continue 'outer;
        }

        // ===== Admission window: drain additional admits until deadline
        //       or saturate at b_max. =====
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
                continue 'outer;
            }
        }

        // ===== Rolling decode loop with biased mid-batch admit. =====
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
                    // cmd_rx closed. Drop event_txs (handlers see EOF), return.
                    event_txs.clear();
                    return;
                }
                RollingEvent::Admit(cmd) => {
                    handle_admit_mid(cmd, &mut sched, &mut event_txs, &admit_count, &model);
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
                            continue 'outer;
                        }
                    }
                }
            }

            // ===== Exit rolling loop when active_count == 0. =====
            if sched.active_count() == 0 {
                match cmd_rx.try_recv() {
                    Ok(cmd) => {
                        // Pending command arrived after last row finished but
                        // before the next select tick. Treat as start of a new
                        // outer batch: drop Finished->Idle via evict_all, then
                        // handle_admit + drain_window + prefill_admitted
                        // inline (cannot requeue into mpsc::Receiver).
                        if let Err(evict_err) = sched.evict_all() {
                            tracing::warn!(
                                "[SchedulerActor] evict_all between batches failed: \
                                 {evict_err:?}; rejecting incoming admit"
                            );
                            // Surface the failure to the caller (best effort).
                            let SchedulerCommand::Admit { reply_tx, .. } = cmd;
                            let _ = reply_tx.send(Err(evict_err));
                            event_txs.clear();
                            continue 'outer;
                        }
                        event_txs.clear();
                        handle_admit(cmd, &mut sched, &mut event_txs, &admit_count);
                        if sched.active_count() == 0 {
                            // admit failed; nothing more to do.
                            break 'rolling;
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
        if let Err(evict_err) = sched.evict_all() {
            tracing::warn!(
                "[SchedulerActor] evict_all at end of batch failed: {evict_err:?}; \
                 relying on 3b-1 poison flag to reject subsequent admits"
            );
        }
        event_txs.clear();
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

/// Mid-batch admit handler. Acquires the model lock, calls
/// [`Scheduler::admit_mid`] (which runs B=1 prefill into a temp cache
/// and adopts the row into the main cache), then registers the
/// per-request event channel and routes the first generated token's
/// event. Lock is held only for the duration of `admit_mid`.
fn handle_admit_mid(
    cmd: SchedulerCommand,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<Qwen35Model>>,
) {
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let admit_result = {
        let model_lock = model.blocking_lock();
        sched.admit_mid(request, &model_lock)
    };
    match admit_result {
        Ok((id, prefill_event)) => {
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
                return;
            }
            // Route the first generated token event.
            route_event(prefill_event, event_txs);
        }
        Err(e) => {
            let _ = reply_tx.send(Err(e));
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
}
