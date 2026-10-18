//! Worker-local speaker diarization on the managed model lifecycle.
use super::{
    engine_pool::{EngineLease, EngineVariant},
    process_memory::global_process_memory_governor,
};
use ironmlx_audio::{
    diarization::{Diarization, NemotronDiarization, Options, ProbabilityChunk, StreamingState},
    AudioError, PcmBuffer, SessionControl,
};
use std::{
    collections::HashMap,
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
struct FeedCancellation {
    flag: Arc<AtomicBool>,
    armed: bool,
}
impl Drop for FeedCancellation {
    fn drop(&mut self) {
        if self.armed {
            self.flag.store(true, Ordering::SeqCst);
        }
    }
}
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
type StreamReply = Result<ProbabilityChunk, DiarizationExecutionError>;
struct StreamState {
    state: StreamingState,
    cancelled: Arc<AtomicBool>,
    _lease: EngineLease,
    _memory: super::process_memory::MemoryReservation,
}
struct StartStream {
    options: Options,
    cancelled: Arc<AtomicBool>,
    lease: EngineLease,
    reply: oneshot::Sender<Result<u64, DiarizationExecutionError>>,
}
struct FeedStream {
    id: u64,
    samples: Vec<f32>,
    snapshot: bool,
    reply: oneshot::Sender<StreamReply>,
    control: Control,
    _pending: Pending,
}
enum Command {
    File(Job),
    Start(StartStream),
    Feed(FeedStream),
}
/// A session owns its cancellation flag. Dropping it releases worker-local state,
/// including its model lease, even when the bounded command queue is full.
pub struct DiarizationStream {
    runtime: DiarizationRuntime,
    id: u64,
    cancellation: Cancellation,
}
impl DiarizationStream {
    /// Feed finite mono 16 kHz PCM; only owned probabilities cross the worker boundary.
    pub async fn feed(&self, samples: Vec<f32>) -> StreamReply {
        self.request(samples, false).await
    }
    /// Flush a copy using the currently available right context. Live speaker
    /// caches and PCM are retained, so a committed turn does not renumber speakers.
    /// Optional samples supply the input resampler's final right-edge tail.
    pub async fn snapshot(&self, tail: Vec<f32>) -> StreamReply {
        self.request(tail, true).await
    }
    async fn request(&self, samples: Vec<f32>, snapshot: bool) -> StreamReply {
        if samples.len() > 5 * 16000 || samples.iter().any(|v| !v.is_finite()) {
            return Err(DiarizationExecutionError::Inference(anyhow::anyhow!(
                "stream feed requires at most five seconds of finite mono 16 kHz PCM"
            )));
        }
        let (reply, result) = oneshot::channel();
        // A cancelled/timed-out feed invalidates this session; partially updated
        // caches must never be reused. Disarming happens only after a full reply.
        let mut cancellation = FeedCancellation {
            flag: self.cancellation.0.clone(),
            armed: true,
        };
        self.runtime
            .inner
            .counts
            .pending
            .fetch_add(1, Ordering::SeqCst);
        self.runtime.send(Command::Feed(FeedStream {
            id: self.id,
            samples,
            snapshot,
            reply,
            control: Control {
                cancelled: cancellation.flag.clone(),
                deadline: Instant::now() + Duration::from_secs(90),
            },
            _pending: Pending(self.runtime.inner.counts.clone()),
        }))?;
        let output = tokio::time::timeout(Duration::from_secs(90), result)
            .await
            .map_err(|_| DiarizationExecutionError::Timeout)?
            .map_err(|_| DiarizationExecutionError::WorkerStopped)?;
        // Do not cancel the live stream after a successful job.
        cancellation.armed = false;
        output
    }
}

struct Job {
    request: DiarizationRequest,
    reply: oneshot::Sender<Reply>,
    control: Control,
    _pending: Pending,
    _lease: Arc<EngineLease>,
}
struct Worker {
    sender: Option<mpsc::SyncSender<Command>>,
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
        let (sender, receiver) = mpsc::sync_channel::<Command>(DIARIZATION_QUEUE_CAPACITY);
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
                            let mut streams: HashMap<u64, StreamState> = HashMap::new();
                            let mut next_id = 0u64;
                            loop {
                                // Periodic pruning works even without a close command and
                                // breaks the lease/runtime cycle after the client disappears.
                                let _ = mlx::synchronize();
                                streams.retain(|_, state| !state.cancelled.load(Ordering::SeqCst));
                                let command = match receiver.recv_timeout(Duration::from_millis(50))
                                {
                                    Ok(command) => command,
                                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                                };
                                let job = match command {
                                    Command::File(job) => job,
                                    Command::Start(start) => {
                                        if start.reply.is_closed()
                                            || start.cancelled.load(Ordering::SeqCst)
                                        {
                                            continue;
                                        }
                                        streams.retain(|_, state| {
                                            !state.cancelled.load(Ordering::SeqCst)
                                        });
                                        let output = (|| {
                                            if streams.len() >= 8 {
                                                return Err(DiarizationExecutionError::QueueFull);
                                            }
                                            let memory = global_process_memory_governor()
                                                .try_reserve(
                                                    16 * 1024 * 1024,
                                                    "Nemotron streaming cache",
                                                )
                                                .map_err(|e| {
                                                    DiarizationExecutionError::Memory(e.to_string())
                                                })?;
                                            let state = model.init_stream(start.options)?;
                                            next_id += 1;
                                            streams.insert(
                                                next_id,
                                                StreamState {
                                                    state,
                                                    cancelled: start.cancelled,
                                                    _lease: start.lease,
                                                    _memory: memory,
                                                },
                                            );
                                            Ok(next_id)
                                        })();
                                        let _ = start.reply.send(output);
                                        continue;
                                    }
                                    Command::Feed(feed) => {
                                        if feed.reply.is_closed() {
                                            continue;
                                        }
                                        worker_counts.active.fetch_add(1, Ordering::SeqCst);
                                        let _active = Active(worker_counts.clone());
                                        let output = (|| -> StreamReply {
                                            feed.control.check().map_err(anyhow::Error::from)?;
                                            let stream = streams
                                                .get_mut(&feed.id)
                                                .ok_or(DiarizationExecutionError::WorkerStopped)?;
                                            let governor = global_process_memory_governor();
                                            governor.sample_process();
                                            let _memory = governor
                                                .try_reserve(
                                                    2 * 1024 * 1024 * 1024,
                                                    "Nemotron streaming inference",
                                                )
                                                .map_err(|e| {
                                                    DiarizationExecutionError::Memory(e.to_string())
                                                })?;
                                            let offset = stream.state.frames_processed;
                                            let output = if feed.snapshot {
                                                model.feed(
                                                    &mut stream.state.clone(),
                                                    &feed.samples,
                                                    true,
                                                    &feed.control,
                                                )
                                            } else {
                                                model.feed(
                                                    &mut stream.state,
                                                    &feed.samples,
                                                    false,
                                                    &feed.control,
                                                )
                                            };
                                            mlx::synchronize().map_err(anyhow::Error::from)?;
                                            Ok((offset, output?))
                                        })();
                                        if output.is_err() {
                                            streams.remove(&feed.id);
                                        }
                                        let _ = feed.reply.send(output);
                                        continue;
                                    }
                                };
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
    fn send(&self, command: Command) -> Result<(), DiarizationExecutionError> {
        self.inner
            .sender
            .as_ref()
            .ok_or(DiarizationExecutionError::WorkerStopped)?
            .try_send(command)
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => DiarizationExecutionError::QueueFull,
                mpsc::TrySendError::Disconnected(_) => DiarizationExecutionError::WorkerStopped,
            })
    }
    pub async fn start_stream(
        &self,
        options: Options,
        lease: EngineLease,
    ) -> Result<DiarizationStream, DiarizationExecutionError> {
        if !matches!(lease.engine(), EngineVariant::Diarization(runtime) if Arc::ptr_eq(&self.inner, &runtime.inner))
        {
            return Err(DiarizationExecutionError::WrongEngine);
        }
        options.validate()?;
        let cancellation = Cancellation(Arc::new(AtomicBool::new(false)));
        let (reply, result) = oneshot::channel();
        self.send(Command::Start(StartStream {
            options,
            cancelled: cancellation.0.clone(),
            lease,
            reply,
        }))?;
        let id = tokio::time::timeout(Duration::from_secs(90), result)
            .await
            .map_err(|_| DiarizationExecutionError::Timeout)?
            .map_err(|_| DiarizationExecutionError::WorkerStopped)??;
        Ok(DiarizationStream {
            runtime: self.clone(),
            id,
            cancellation,
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
        self.send(Command::File(job))?;
        tokio::time::timeout(deadline, result)
            .await
            .map_err(|_| DiarizationExecutionError::Timeout)?
            .map_err(|_| DiarizationExecutionError::WorkerStopped)?
    }
}
