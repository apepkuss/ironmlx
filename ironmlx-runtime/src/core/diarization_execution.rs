//! Worker-local speaker diarization on the managed model lifecycle.
use super::{
    engine_pool::{EngineLease, EngineVariant},
    process_memory::global_process_memory_governor,
};
use ironmlx_audio::{
    diarization::{Diarization, NemotronDiarization, Options},
    AudioError, PcmBuffer, SessionControl,
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

pub const DIARIZATION_QUEUE_CAPACITY: usize = 4;
#[derive(Debug, thiserror::Error)]
pub enum DiarizationExecutionError {
    #[error("Diarization request queue is full")]
    QueueFull,
    #[error("Diarization worker is unavailable")]
    WorkerStopped,
    #[error("Diarization requires a lease for the same model")]
    WrongEngine,
    #[error("Diarization request timed out")]
    Timeout,
    #[error("{0}")]
    Memory(String),
    #[error("{0}")]
    Inference(#[from] anyhow::Error),
}

pub struct DiarizationRequest {
    pub pcm: PcmBuffer,
    pub options: Options,
    pub progress: Option<tokio::sync::mpsc::Sender<ironmlx_audio::diarization::ProbabilityChunk>>,
    pub input_memory: Option<Arc<super::process_memory::MemoryReservation>>,
    pub input_slot: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
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
struct Cancellation(Arc<AtomicBool>);
impl Drop for Cancellation {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
struct Control {
    cancelled: Arc<AtomicBool>,
    deadline: Instant,
}
impl SessionControl for Control {
    fn check(&self) -> ironmlx_audio::Result<()> {
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(AudioError::Cancelled);
        }
        if Instant::now() >= self.deadline {
            return Err(AudioError::DeadlineExceeded);
        }
        Ok(())
    }
}
type Reply = Result<Diarization, DiarizationExecutionError>;
struct Job {
    request: DiarizationRequest,
    reply: oneshot::Sender<Reply>,
    control: Control,
    _pending: Pending,
    _lease: Arc<EngineLease>,
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
#[derive(Clone)]
pub struct DiarizationRuntime {
    inner: Arc<Worker>,
}
impl DiarizationRuntime {
    pub(crate) async fn load(path: PathBuf) -> anyhow::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<Job>(DIARIZATION_QUEUE_CAPACITY);
        let (ready_tx, ready_rx) = oneshot::channel();
        let counts = Arc::new(Counts::default());
        let worker_counts = counts.clone();
        let usage = Arc::new(super::runtime_usage::ModelRuntimeUsageCounters::default());
        let worker_usage = usage.clone();
        let thread = std::thread::Builder::new()
            .name("ironmlx-diarization".into())
            .spawn(move || {
                let load = || -> anyhow::Result<_> {
                    let device = mlx::Device::gpu(0);
                    mlx::set_default_device(device);
                    mlx::set_default_stream(mlx::new_stream(device)?);
                    NemotronDiarization::load(&path)
                };
                match load() {
                    Ok(model) => {
                        if ready_tx.send(Ok(model.weight_bytes() as usize)).is_ok() {
                            while let Ok(job) = receiver.recv() {
                                if job.reply.is_closed() {
                                    continue;
                                }
                                worker_counts.active.fetch_add(1, Ordering::SeqCst);
                                let _active = Active(worker_counts.clone());
                                let result = (|| -> Reply {
                                    job.control.check().map_err(anyhow::Error::from)?;
                                    let governor = global_process_memory_governor();
                                    governor.sample_process();
                                    let _reservation = governor
                                        .try_reserve(
                                            2 * 1024 * 1024 * 1024,
                                            "Nemotron diarization inference",
                                        )
                                        .map_err(|e| {
                                            DiarizationExecutionError::Memory(e.to_string())
                                        })?;
                                    let result = model.diarize(
                                        &job.request.pcm,
                                        job.request.options,
                                        &job.control,
                                        &mut |offset, probabilities| {
                                            if let Some(progress) = &job.request.progress {
                                                progress
                                                    .blocking_send((offset, probabilities.to_vec()))
                                                    .map_err(|_| AudioError::Cancelled)?;
                                            }
                                            Ok(())
                                        },
                                    );
                                    // Finish submitted GPU work before releasing the lease/reservation.
                                    mlx::synchronize().map_err(anyhow::Error::from)?;
                                    let output = result?;
                                    worker_usage.record_output_tokens(output.segments.len() as u64);
                                    Ok(output)
                                })();
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
    pub async fn diarize(&self, request: DiarizationRequest, lease: EngineLease) -> Reply {
        if !matches!(lease.engine(),EngineVariant::Diarization(runtime) if Arc::ptr_eq(&self.inner,&runtime.inner))
        {
            return Err(DiarizationExecutionError::WrongEngine);
        }
        let lease = Arc::new(lease);
        let (reply, result) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let _cancellation = Cancellation(cancelled.clone());
        self.inner.counts.pending.fetch_add(1, Ordering::SeqCst);
        let deadline = Duration::from_secs(
            180 + (request.pcm.samples.len() as u64
                / u64::from(request.pcm.format.channels.max(1))
                / u64::from(request.pcm.format.sample_rate.max(1)))
                * 2,
        );
        let job = Job {
            request,
            reply,
            control: Control {
                cancelled,
                deadline: Instant::now() + deadline,
            },
            _pending: Pending(self.inner.counts.clone()),
            _lease: lease,
        };
        self.inner
            .sender
            .as_ref()
            .ok_or(DiarizationExecutionError::WorkerStopped)?
            .try_send(job)
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => DiarizationExecutionError::QueueFull,
                mpsc::TrySendError::Disconnected(_) => DiarizationExecutionError::WorkerStopped,
            })?;
        tokio::time::timeout(deadline, result)
            .await
            .map_err(|_| DiarizationExecutionError::Timeout)?
            .map_err(|_| DiarizationExecutionError::WorkerStopped)?
    }
}
