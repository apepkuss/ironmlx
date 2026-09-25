//! Worker-owned MLX decision inference using the shared model lifecycle.
use super::{engine_pool::EngineLease, process_memory::global_process_memory_governor};
use ironmlx_decision::{DecisionRequest, DecisionResponse, Laya};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

pub const DECISION_QUEUE_CAPACITY: usize = 16;

#[derive(Debug, thiserror::Error)]
pub enum DecisionExecutionError {
    #[error("decision request queue is full")]
    QueueFull,
    #[error("decision worker is unavailable")]
    WorkerStopped,
    #[error("decision request timed out")]
    Timeout,
    #[error("decision execution requires a lease for the same worker")]
    WrongEngine,
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Inference(String),
}

#[derive(Default)]
struct Counts {
    pending: AtomicUsize,
    active: AtomicUsize,
}
struct Pending(Arc<Counts>);
impl Drop for Pending {
    fn drop(&mut self) {
        self.0.pending.fetch_sub(1, Ordering::SeqCst);
    }
}
struct Active(Arc<Counts>);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}
type Reply = Result<DecisionResponse, DecisionExecutionError>;
struct Job {
    request: DecisionRequest,
    reply: oneshot::Sender<Reply>,
    started: Instant,
    _pending: Pending,
    // Cancellation of HTTP must not release the model during GPU work.
    _lease: EngineLease,
}

#[derive(Clone)]
pub struct DecisionRuntime {
    inner: Arc<Worker>,
}
struct Worker {
    sender: Option<mpsc::SyncSender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
    counts: Arc<Counts>,
    weight_bytes: usize,
    usage: Arc<super::runtime_usage::ModelRuntimeUsageCounters>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            if thread.thread().id() != std::thread::current().id() {
                let _ = thread.join();
            }
        }
    }
}
impl DecisionRuntime {
    pub(crate) async fn load(
        path: PathBuf,
        settings: ironmlx_decision::DecisionSettings,
    ) -> anyhow::Result<Self> {
        settings.validate()?;
        let (sender, receiver) = mpsc::sync_channel::<Job>(DECISION_QUEUE_CAPACITY);
        let (ready_tx, ready_rx) = oneshot::channel();
        let counts = Arc::new(Counts::default());
        let worker_counts = counts.clone();
        let usage = Arc::new(super::runtime_usage::ModelRuntimeUsageCounters::default());
        let worker_usage = usage.clone();
        let thread = std::thread::Builder::new()
            .name("ironmlx-decision".into())
            .spawn(move || {
                let load = || -> anyhow::Result<_> {
                    let device = settings.device.resolve()?;
                    mlx::set_default_device(device);
                    mlx::set_default_stream(mlx::new_stream(device)?);
                    let model = Laya::load_with_settings(&path, settings)?;
                    let bytes =
                        usize::try_from(std::fs::metadata(path.join("model.safetensors"))?.len())?;
                    Ok((model, bytes * settings.dtype.weight_multiplier()))
                };
                match load() {
                    Ok((model, bytes)) => {
                        if ready_tx.send(Ok(bytes)).is_ok() {
                            while let Ok(job) = receiver.recv() {
                                if job.reply.is_closed() {
                                    continue;
                                }
                                if job.started.elapsed() > Duration::from_secs(60) {
                                    let _ = job.reply.send(Err(DecisionExecutionError::Timeout));
                                    continue;
                                }
                                worker_counts.active.fetch_add(1, Ordering::SeqCst);
                                let _active = Active(worker_counts.clone());
                                let result = (|| -> Reply {
                                    let governor = global_process_memory_governor();
                                    let _memory = governor
                                        .try_reserve(
                                            256 * 1024
                                                * 1024
                                                * settings
                                                    .batch_size
                                                    .min(job.request.questions.len())
                                                * settings.dtype.weight_multiplier(),
                                            "Laya inference",
                                        )
                                        .map_err(|e| {
                                            DecisionExecutionError::Inference(e.to_string())
                                        })?;
                                    let result = model.predict(&job.request).map_err(|e| {
                                        if e.to_string().contains("question options exceed") {
                                            DecisionExecutionError::InvalidInput(e.to_string())
                                        } else {
                                            DecisionExecutionError::Inference(e.to_string())
                                        }
                                    });
                                    mlx::synchronize().map_err(|e| {
                                        DecisionExecutionError::Inference(e.to_string())
                                    })?;
                                    result
                                })();
                                if let Ok(response) = &result {
                                    worker_usage
                                        .record_input_tokens(response.usage.input_tokens as u64);
                                }
                                let _ = job.reply.send(result);
                            }
                        }
                        let _ = mlx::synchronize();
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                }
            })?;
        let weight_bytes = ready_rx.await??;
        Ok(Self {
            inner: Arc::new(Worker {
                sender: Some(sender),
                thread: Some(thread),
                counts,
                weight_bytes,
                usage,
            }),
        })
    }

    pub fn model_weight_bytes(&self) -> usize {
        self.inner.weight_bytes
    }
    pub fn active_and_queued(&self) -> (usize, usize) {
        let active = self.inner.counts.active.load(Ordering::SeqCst);
        (
            active,
            self.inner
                .counts
                .pending
                .load(Ordering::SeqCst)
                .saturating_sub(active),
        )
    }
    pub fn usage(&self) -> super::runtime_usage::ModelRuntimeUsageSnapshot {
        self.inner.usage.snapshot(false)
    }

    pub async fn predict(&self, request: DecisionRequest, lease: EngineLease) -> Reply {
        if !matches!(lease.engine(), super::engine_pool::EngineVariant::Decision(runtime) if Arc::ptr_eq(&self.inner, &runtime.inner))
        {
            return Err(DecisionExecutionError::WrongEngine);
        }
        request
            .validate()
            .map_err(|e| DecisionExecutionError::InvalidInput(e.to_string()))?;
        let (reply, result) = oneshot::channel();
        self.inner.counts.pending.fetch_add(1, Ordering::SeqCst);
        let job = Job {
            request,
            reply,
            started: Instant::now(),
            _pending: Pending(self.inner.counts.clone()),
            _lease: lease,
        };
        self.inner
            .sender
            .as_ref()
            .expect("live worker")
            .try_send(job)
            .map_err(|e| match e {
                mpsc::TrySendError::Full(_) => DecisionExecutionError::QueueFull,
                mpsc::TrySendError::Disconnected(_) => DecisionExecutionError::WorkerStopped,
            })?;
        tokio::time::timeout(Duration::from_secs(120), result)
            .await
            .map_err(|_| DecisionExecutionError::Timeout)?
            .map_err(|_| DecisionExecutionError::WorkerStopped)?
    }
}
