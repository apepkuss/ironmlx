//! Bounded, worker-local audio execution sharing the process memory governor.
//! Only owned host PCM crosses the worker boundary; model/session objects never do.
use super::{
    engine_pool::EngineLease,
    process_memory::{global_process_memory_governor, MemoryReservation},
};
use ironmlx_audio::{
    indextts25::IndexTts25Loader, AudioError, Language, OutputPolicy, PcmChunk,
    ResolvedModelResources, SessionControl, TtsModel, TtsRequest, TtsStep, TtsSummary,
};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore};

pub const AUDIO_QUEUE_CAPACITY: usize = 4;
pub const AUDIO_CHUNK_FRAMES: usize = 8192;
pub const AUDIO_CHANNEL_CAPACITY: usize = 8;

/// Explicit local resources, outside the immutable source snapshot. No downloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioModelResources {
    pub derived_resources: PathBuf,
    pub resource_lock: PathBuf,
    pub wetext_fsts: PathBuf,
    pub unidic_dir: PathBuf,
    #[serde(default)]
    pub execution: AudioExecutionLimits,
}
impl AudioModelResources {
    fn resolved(&self, snapshot: PathBuf) -> ResolvedModelResources {
        ResolvedModelResources {
            source_snapshot: snapshot,
            derived_resources: self.derived_resources.clone(),
            resource_lock: self.resource_lock.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioExecutionLimits {
    #[serde(with = "duration_millis", rename = "queue_timeout_ms")]
    pub queue_timeout: Duration,
    #[serde(with = "duration_millis", rename = "first_audio_timeout_ms")]
    pub first_audio_timeout: Duration,
    #[serde(with = "duration_millis", rename = "execution_timeout_ms")]
    pub execution_timeout: Duration,
    #[serde(with = "duration_millis", rename = "slow_consumer_timeout_ms")]
    pub slow_consumer_timeout: Duration,
    /// Output service policy, including inter-segment silence. May only lower
    /// the v1 maximum of 600 seconds at 22050 Hz.
    pub max_output_frames: u64,
    pub segment_tokens: usize,
}
impl Default for AudioExecutionLimits {
    fn default() -> Self {
        Self {
            queue_timeout: Duration::from_secs(60),
            first_audio_timeout: Duration::from_secs(120),
            execution_timeout: Duration::from_secs(900),
            slow_consumer_timeout: Duration::from_secs(30),
            max_output_frames: 600 * 22050,
            segment_tokens: 120,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AudioExecutionError {
    #[error("audio execution requires a lease for the same model worker")]
    WrongEngine,
    #[error("audio request queue is full")]
    QueueFull,
    #[error("audio {0} deadline exceeded")]
    Timeout(&'static str),
    #[error("audio worker terminated unexpectedly")]
    WorkerStopped,
    #[error("audio request cancelled")]
    Cancelled,
    #[error(transparent)]
    Memory(#[from] super::process_memory::MemoryReservationError),
    #[error(transparent)]
    Model(#[from] AudioError),
}
pub type AudioExecutionResult<T> = Result<T, AudioExecutionError>;

/// The allocation's owner must retain this reservation until consumption/drop.
/// Transport can place it inside a reference-counted byte owner.
pub struct AudioOutputChunk {
    pub chunk: PcmChunk,
    pub reservation: MemoryReservation,
}
pub enum AudioOutput {
    Chunk(AudioOutputChunk),
    Finished(TtsSummary),
}

#[derive(Default)]
struct Counts {
    active: AtomicUsize,
    queued: AtomicUsize,
}
struct Queued(Arc<Counts>);
impl Drop for Queued {
    fn drop(&mut self) {
        self.0.queued.fetch_sub(1, Ordering::SeqCst);
    }
}
struct Active(Arc<Counts>);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

struct Job {
    request: TtsRequest,
    control: Arc<RequestControl>,
    tx: mpsc::Sender<AudioOutputChunk>,
    terminal: oneshot::Sender<AudioExecutionResult<TtsSummary>>,
    _permit: OwnedSemaphorePermit,
    _active: Active,
    _input: MemoryReservation,
    _lease: EngineLease,
}
struct RequestControl {
    cancelled: AtomicBool,
    started: Instant,
    first_audio: AtomicBool,
    limits: AudioExecutionLimits,
}
impl RequestControl {
    fn status(&self) -> AudioExecutionResult<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(AudioExecutionError::Cancelled);
        }
        if self.started.elapsed() >= self.limits.execution_timeout {
            return Err(AudioExecutionError::Timeout("execution"));
        }
        if !self.first_audio.load(Ordering::Acquire)
            && self.started.elapsed() >= self.limits.first_audio_timeout
        {
            return Err(AudioExecutionError::Timeout("first audio"));
        }
        Ok(())
    }
}
impl SessionControl for RequestControl {
    fn check(&self) -> ironmlx_audio::Result<()> {
        self.status().map_err(|e| match e {
            AudioExecutionError::Cancelled => AudioError::Cancelled,
            _ => AudioError::DeadlineExceeded,
        })
    }
}

/// Dropping the stream signals cancellation. Worker-owned permits, leases and
/// computation memory remain alive through the final GPU synchronization.
pub struct AudioResponse {
    rx: mpsc::Receiver<AudioOutputChunk>,
    terminal: Option<oneshot::Receiver<AudioExecutionResult<TtsSummary>>>,
    control: Arc<RequestControl>,
    completed: Option<TtsSummary>,
    done: bool,
}
impl Drop for AudioResponse {
    fn drop(&mut self) {
        self.control.cancelled.store(true, Ordering::Release);
    }
}
impl AudioResponse {
    pub async fn next(&mut self) -> AudioExecutionResult<Option<AudioOutput>> {
        if self.done {
            return Ok(None);
        }
        let first = !self.control.first_audio.load(Ordering::Acquire);
        let duration = if first {
            self.control
                .limits
                .first_audio_timeout
                .min(self.control.limits.execution_timeout)
        } else {
            self.control.limits.execution_timeout
        };
        let deadline = tokio::time::Instant::from_std(self.control.started + duration);
        match tokio::time::timeout_at(deadline, self.next_inner()).await {
            Ok(result) => result,
            Err(_) => {
                self.done = true;
                self.control.cancelled.store(true, Ordering::Release);
                Err(AudioExecutionError::Timeout(if first {
                    "first audio"
                } else {
                    "execution"
                }))
            }
        }
    }
    async fn next_inner(&mut self) -> AudioExecutionResult<Option<AudioOutput>> {
        if self.done {
            return Ok(None);
        }
        loop {
            // Failure has priority over buffered, unsent audio. Success is held
            // until every preceding chunk has been received.
            if let Some(terminal) = self.terminal.as_mut() {
                match terminal.try_recv() {
                    Ok(Ok(summary)) => {
                        self.completed = Some(summary);
                        self.terminal = None;
                    }
                    Ok(Err(error)) => {
                        self.done = true;
                        self.rx.close();
                        return Err(error);
                    }
                    Err(oneshot::error::TryRecvError::Closed) => {
                        self.done = true;
                        return Err(AudioExecutionError::WorkerStopped);
                    }
                    Err(oneshot::error::TryRecvError::Empty) => {}
                }
            }
            if self.completed.is_some() {
                if let Some(chunk) = self.rx.recv().await {
                    return Ok(Some(AudioOutput::Chunk(chunk)));
                }
                self.done = true;
                return Ok(Some(AudioOutput::Finished(self.completed.take().unwrap())));
            }
            tokio::select! {
                biased;
                result = self.terminal.as_mut().expect("pending terminal") => {
                    self.terminal = None;
                    match result {
                        Ok(Ok(summary)) => self.completed = Some(summary),
                        Ok(Err(error)) => { self.done = true; self.rx.close(); return Err(error); }
                        Err(_) => { self.done = true; return Err(AudioExecutionError::WorkerStopped); }
                    }
                }
                chunk = self.rx.recv() => {
                    if let Some(chunk) = chunk { return Ok(Some(AudioOutput::Chunk(chunk))); }
                    // Sender closes just before publishing the terminal result.
                    match self.terminal.take().expect("pending terminal").await {
                        Ok(Ok(summary)) => { self.done = true; return Ok(Some(AudioOutput::Finished(summary))); }
                        Ok(Err(error)) => { self.done = true; return Err(error); }
                        Err(_) => { self.done = true; return Err(AudioExecutionError::WorkerStopped); }
                    }
                }
            }
        }
    }
}

/// Cloneable handle to one dedicated native worker and its serial admission lane.
#[derive(Clone)]
pub struct AudioRuntime {
    inner: Arc<WorkerHandle>,
}
struct WorkerHandle {
    sender: Option<std::sync::mpsc::Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
    execution: Arc<Semaphore>,
    counts: Arc<Counts>,
    model_weight_bytes: usize,
    limits: AudioExecutionLimits,
}
impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            // A draining pool can release its last reference in a completed job.
            // In that case the worker exits on its next recv, on the same thread.
            if thread.thread().id() != std::thread::current().id() {
                let _ = thread.join();
            }
        }
    }
}
impl AudioRuntime {
    pub(crate) async fn load(
        snapshot: PathBuf,
        resources: AudioModelResources,
    ) -> anyhow::Result<Self> {
        let limits = resources.execution.clone();
        limits.validate()?;
        let max_output_frames = limits.max_output_frames;
        let segment_tokens = limits.segment_tokens;
        let (sender, receiver) = std::sync::mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("ironmlx-audio".into())
            .spawn(move || {
                let load = || -> anyhow::Result<_> {
                    let stream = mlx::new_stream(mlx::Device::gpu(0))?;
                    mlx::set_default_stream(stream);
                    let loader = IndexTts25Loader::new(
                        resources.wetext_fsts.clone(),
                        resources.unidic_dir.clone(),
                    )
                    .with_limits(
                        ironmlx_audio::text::TextLimits {
                            segment_tokens,
                            ..Default::default()
                        },
                        max_output_frames,
                    )?;
                    let resolved = resources.resolved(snapshot);
                    let model = loader.load_model(&resolved)?;
                    // Loading already validates every weight. Count the verified
                    // backing files without repeating multi-GB integrity hashing.
                    let mut bytes = 0usize;
                    for name in [
                        "gpt.safetensors",
                        "codec.safetensors",
                        "s2mel.safetensors",
                        "bigvgan.safetensors",
                        "model.safetensors",
                    ] {
                        bytes = bytes
                            .checked_add(usize::try_from(
                                std::fs::metadata(resolved.source_snapshot.join(name))?.len(),
                            )?)
                            .ok_or_else(|| anyhow::anyhow!("audio weight size overflow"))?;
                    }
                    for name in ["auxiliary.safetensors", "campplus.safetensors"] {
                        bytes = bytes
                            .checked_add(usize::try_from(
                                std::fs::metadata(resolved.derived_resources.join(name))?.len(),
                            )?)
                            .ok_or_else(|| anyhow::anyhow!("audio weight size overflow"))?;
                    }
                    Ok((model, bytes))
                };
                match load() {
                    Ok((mut model, bytes)) => {
                        if ready_tx.send(Ok(bytes)).is_ok() {
                            while let Ok(job) = receiver.recv() {
                                run_job(&mut model, job);
                            }
                        }
                        let _ = mlx::synchronize();
                        drop(model);
                        mlx::clear_streams();
                    }
                    Err(error) => {
                        let _ = mlx::synchronize();
                        mlx::clear_streams();
                        let _ = ready_tx.send(Err(error));
                    }
                }
            })?;
        // The pool owns this load future; a disconnected HTTP waiter must not
        // abort it. It retains snapshot and load reservations until completion.
        let bytes = ready_rx
            .await
            .map_err(|_| anyhow::anyhow!("audio loader stopped"))??;
        Ok(Self {
            inner: Arc::new(WorkerHandle {
                sender: Some(sender),
                thread: Some(thread),
                execution: Arc::new(Semaphore::new(1)),
                counts: Arc::new(Counts::default()),
                model_weight_bytes: bytes,
                limits,
            }),
        })
    }
    pub fn model_weight_bytes(&self) -> usize {
        self.inner.model_weight_bytes
    }
    pub fn active_and_queued(&self) -> (usize, usize) {
        (
            self.inner.counts.active.load(Ordering::SeqCst),
            self.inner.counts.queued.load(Ordering::SeqCst),
        )
    }

    pub async fn submit(
        &self,
        request: TtsRequest,
        input: MemoryReservation,
        lease: EngineLease,
    ) -> AudioExecutionResult<AudioResponse> {
        match lease.engine() {
            super::engine_pool::EngineVariant::Audio(runtime)
                if Arc::ptr_eq(&runtime.inner, &self.inner) => {}
            _ => return Err(AudioExecutionError::WrongEngine),
        }
        let permit = match self.inner.execution.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                self.inner
                    .counts
                    .queued
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                        (n < AUDIO_QUEUE_CAPACITY).then_some(n + 1)
                    })
                    .map_err(|_| AudioExecutionError::QueueFull)?;
                let queued = Queued(self.inner.counts.clone());
                let permit = tokio::time::timeout(
                    self.inner.limits.queue_timeout,
                    self.inner.execution.clone().acquire_owned(),
                )
                .await
                .map_err(|_| AudioExecutionError::Timeout("queue"))?
                .map_err(|_| AudioExecutionError::WorkerStopped)?;
                drop(queued);
                permit
            }
        };
        self.inner.counts.active.fetch_add(1, Ordering::SeqCst);
        let control = Arc::new(RequestControl {
            cancelled: AtomicBool::new(false),
            started: Instant::now(),
            first_audio: AtomicBool::new(false),
            limits: self.inner.limits.clone(),
        });
        let (tx, rx) = mpsc::channel(AUDIO_CHANNEL_CAPACITY);
        let (terminal, terminal_rx) = oneshot::channel();
        let response = AudioResponse {
            rx,
            terminal: Some(terminal_rx),
            control: control.clone(),
            completed: None,
            done: false,
        };
        let job = Job {
            request,
            control,
            tx,
            terminal,
            _permit: permit,
            _active: Active(self.inner.counts.clone()),
            _input: input,
            _lease: lease,
        };
        self.inner
            .sender
            .as_ref()
            .ok_or(AudioExecutionError::WorkerStopped)?
            .send(job)
            .map_err(|_| AudioExecutionError::WorkerStopped)?;
        Ok(response)
    }
}

fn run_job(model: &mut dyn TtsModel, job: Job) {
    let Job {
        request,
        control,
        tx,
        terminal,
        _permit,
        _active,
        _input,
        _lease,
    } = job;
    let governor = global_process_memory_governor();
    // Keep reservations outside the unwind boundary through synchronization.
    governor.sample_process();
    let reservation = governor.try_reserve_prefill(32 * 1024 * 1024 * 1024, "audio computation");
    let result = match reservation.as_ref() {
        Ok(_) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> AudioExecutionResult<TtsSummary> {
                control.status()?;
                let mut session = model.start(request, control.as_ref())?;
                let mut frames = 0u64;
                loop {
                    control.status()?;
                    let step = session.advance().map_err(|e| {
                        control
                            .status()
                            .err()
                            .unwrap_or(AudioExecutionError::Model(e))
                    })?;
                    match step {
                        TtsStep::Progress => {}
                        TtsStep::Finished(summary) => {
                            if frames == 0 || frames != summary.total_frames {
                                return Err(AudioExecutionError::WorkerStopped);
                            }
                            return Ok(summary);
                        }
                        TtsStep::Audio(chunk) => {
                            if chunk.start_frame != frames
                                || chunk.pcm.samples.is_empty()
                                || chunk.pcm.samples.len() > AUDIO_CHUNK_FRAMES
                                || chunk.pcm.format.channels != 1
                                || chunk.pcm.format.sample_rate != 22050
                            {
                                return Err(AudioExecutionError::WorkerStopped);
                            }
                            frames += chunk.pcm.samples.len() as u64;
                            governor.sample_process();
                            // f32 PCM plus its s16 transport representation, including
                            // Vec allocation capacity, remains reserved across send.
                            let reservation = governor.try_reserve(
                                chunk.pcm.samples.capacity() * 4 + chunk.pcm.samples.len() * 2,
                                "audio output chunk",
                            )?;
                            let mut output = AudioOutputChunk { chunk, reservation };
                            let started = Instant::now();
                            loop {
                                control.status()?;
                                match tx.try_send(output) {
                                    Ok(()) => {
                                        control.first_audio.store(true, Ordering::Release);
                                        break;
                                    }
                                    Err(mpsc::error::TrySendError::Closed(_)) => {
                                        return Err(AudioExecutionError::Cancelled)
                                    }
                                    Err(mpsc::error::TrySendError::Full(value)) => output = value,
                                }
                                if started.elapsed() >= control.limits.slow_consumer_timeout {
                                    return Err(AudioExecutionError::Timeout("slow consumer"));
                                }
                                std::thread::park_timeout(Duration::from_millis(5));
                            }
                        }
                    }
                }
            },
        ))
        .unwrap_or(Err(AudioExecutionError::WorkerStopped)),
        Err(_) => Err(AudioExecutionError::Memory(
            reservation.as_ref().err().unwrap().clone(),
        )),
    };
    let result = match mlx::synchronize() {
        Ok(()) => result,
        Err(e) => Err(AudioExecutionError::Model(e.into())),
    };
    drop(reservation);
    governor.sample_process();
    drop(tx);
    if let Err(error) = &result {
        tracing::warn!(%error, "audio request failed");
    }
    let _ = terminal.send(result);
    // Drop the lease away from the model-owning thread, so draining/unload can
    // join that thread without self-join. GPU work has already synchronized.
    std::thread::spawn(move || drop((_lease, _input, _active, _permit)));
}

/// Construct the agreed request after bounded native input decoding.
pub fn speech_request(
    text: String,
    reference: ironmlx_audio::PcmBuffer,
    stream: bool,
) -> TtsRequest {
    TtsRequest {
        text,
        reference,
        language: Language::Auto,
        seed: None,
        output_policy: if stream {
            OutputPolicy::Chunks
        } else {
            OutputPolicy::Collect
        },
    }
}

impl AudioExecutionLimits {
    pub fn validate(&self) -> anyhow::Result<()> {
        let maximum_deadline = Duration::from_secs(24 * 60 * 60);
        if [
            self.queue_timeout,
            self.first_audio_timeout,
            self.execution_timeout,
            self.slow_consumer_timeout,
        ]
        .iter()
        .any(|value| *value > maximum_deadline)
            || self.queue_timeout.is_zero()
            || self.first_audio_timeout.is_zero()
            || self.execution_timeout.is_zero()
            || self.slow_consumer_timeout.is_zero()
            || self.max_output_frames == 0
            || self.max_output_frames > 600 * 22050
            || !(6..=120).contains(&self.segment_tokens)
        {
            anyhow::bail!("audio deadlines must be positive and max_output_frames must be within 1..=13230000");
        }
        Ok(())
    }
}
mod duration_millis {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;
    pub fn serialize<S: Serializer>(value: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(
            value
                .as_millis()
                .try_into()
                .map_err(serde::ser::Error::custom)?,
        )
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_millis(u64::deserialize(deserializer)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn response(
        limits: AudioExecutionLimits,
    ) -> (
        AudioResponse,
        mpsc::Sender<AudioOutputChunk>,
        oneshot::Sender<AudioExecutionResult<TtsSummary>>,
    ) {
        let control = Arc::new(RequestControl {
            cancelled: AtomicBool::new(false),
            started: Instant::now(),
            first_audio: AtomicBool::new(false),
            limits,
        });
        let (tx, rx) = mpsc::channel(8);
        let (terminal, terminal_rx) = oneshot::channel();
        (
            AudioResponse {
                rx,
                terminal: Some(terminal_rx),
                control,
                completed: None,
                done: false,
            },
            tx,
            terminal,
        )
    }
    #[tokio::test]
    async fn unexpected_worker_exit_is_an_error_not_eof() {
        let (mut output, tx, terminal) = response(Default::default());
        drop((tx, terminal));
        assert!(matches!(
            output.next().await,
            Err(AudioExecutionError::WorkerStopped)
        ));
        assert!(output.next().await.unwrap().is_none());
    }
    #[tokio::test]
    async fn first_audio_deadline_cancels_a_worker_without_releasing_its_resources() {
        let limits = AudioExecutionLimits {
            first_audio_timeout: Duration::from_millis(5),
            ..Default::default()
        };
        let (mut output, _tx, _terminal) = response(limits);
        let control = output.control.clone();
        assert!(matches!(
            output.next().await,
            Err(AudioExecutionError::Timeout("first audio"))
        ));
        assert!(control.cancelled.load(Ordering::Acquire));
    }
    #[tokio::test]
    async fn execution_deadline_remains_active_after_first_audio() {
        let limits = AudioExecutionLimits {
            execution_timeout: Duration::from_millis(5),
            ..Default::default()
        };
        let (mut output, _tx, _terminal) = response(limits);
        output.control.first_audio.store(true, Ordering::Release);
        assert!(matches!(
            output.next().await,
            Err(AudioExecutionError::Timeout("execution"))
        ));
    }
    #[test]
    fn dropping_consumer_cancels_its_worker() {
        let (output, _tx, _terminal) = response(Default::default());
        let control = output.control.clone();
        drop(output);
        assert!(control.cancelled.load(Ordering::Acquire));
    }
    #[test]
    fn policy_configuration_uses_milliseconds_and_rejects_invalid_limits() {
        let limits: AudioExecutionLimits =
            serde_json::from_str(r#"{"queue_timeout_ms":1234,"max_output_frames":22050}"#).unwrap();
        assert_eq!(limits.queue_timeout, Duration::from_millis(1234));
        assert!(limits.validate().is_ok());
        assert!(
            serde_json::from_str::<AudioExecutionLimits>(r#"{"queue_timeout_ms":null}"#).is_err()
        );
        assert!(serde_json::from_str::<AudioExecutionLimits>(r#"{"unknown":1}"#).is_err());
        assert!(AudioExecutionLimits {
            max_output_frames: 600 * 22050 + 1,
            ..limits
        }
        .validate()
        .is_err());
    }
}
