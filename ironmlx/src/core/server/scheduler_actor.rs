//! SchedulerActor — Tokio task wrapping [`Scheduler`] for serving HTTP
//! requests via mpsc channels.
//!
//! 3b-2 ships the "one-admit-per-batch" form of the driver loop. 3b-3 will
//! replace [`driver_loop`]'s `cmd_rx.blocking_recv()` with an
//! admission-window `select!` so the driver can pack multiple concurrent
//! admits into a single batched forward.
//!
//! See `docs/superpowers/specs/2026-05-13-b1-p2-3b-2-scheduler-actor-skeleton-design.md` § 4.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
}

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

/// Acquire the model lock, drive prefill + step loop to completion, evict
/// the batch, and release the lock. Lock held only for the duration of
/// this call.
fn run_batch_once(
    sched: &mut Scheduler,
    model: &Arc<Mutex<Qwen35Model>>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
) -> Result<()> {
    let model = model.blocking_lock();

    let prefill_events = sched.prefill_admitted(&model)?;
    for ev in prefill_events {
        route_event(ev, event_txs);
    }

    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model)?;
        for ev in events {
            route_event(ev, event_txs);
        }
    }

    sched.evict_all()?;
    // Drop all per-request senders → handlers see channel close (EOF).
    event_txs.clear();
    Ok(())
}

fn route_event(ev: StepEvent, event_txs: &HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>) {
    if let Some(tx) = event_txs.get(&ev.id) {
        // Unbounded channel — only fails when the receiver was dropped
        // (handler abandoned). That's fine; the entry naturally clears
        // at the next `event_txs.clear()` in run_batch_once.
        let _ = tx.send(ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drop the SchedulerActorHandle (and thus cmd_tx); confirm the driver
    /// task exits cleanly. We can't construct a real Qwen35Model in a unit
    /// test, so we never send any commands — we only verify the driver's
    /// `while let Some(cmd) = cmd_rx.blocking_recv()` loop terminates when
    /// all senders are dropped.
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
