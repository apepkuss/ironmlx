//! Worker-local speech recognition on the managed model lifecycle.
use super::{
    engine_pool::{EngineLease, EngineVariant},
    process_memory::global_process_memory_governor,
};
use ironmlx_audio::{
    qwen3_asr::{Qwen3Asr, Transcription},
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

pub const ASR_QUEUE_CAPACITY: usize = 4;

/// Per-model admission and generation defaults, independent of chat sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsrSettings {
    pub max_audio_duration_seconds: u32,
    /// Budget for each bounded recognition window, not the merged recording.
    pub max_output_tokens: usize,
}

impl Default for AsrSettings {
    fn default() -> Self {
        // Both Qwen3-ASR profiles share the official 20-minute chunk policy and
        // the 1,024-token evaluation setting. Native execution still uses <=30s windows.
        Self {
            max_audio_duration_seconds: 1200,
            max_output_tokens: 1024,
        }
    }
}

impl AsrSettings {
    pub fn validate(self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            (1..=7200).contains(&self.max_audio_duration_seconds),
            "ASR max_audio_duration_seconds must be an integer from 1 to 7200"
        );
        anyhow::ensure!(
            (1..=8192).contains(&self.max_output_tokens),
            "ASR max_output_tokens must be an integer from 1 to 8192"
        );
        Ok(self)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AsrExecutionError {
    #[error("ASR request queue is full")]
    QueueFull,
    #[error("ASR worker is unavailable")]
    WorkerStopped,
    #[error("ASR execution requires a lease for the same model")]
    WrongEngine,
    #[error("ASR request timed out")]
    Timeout,
    #[error("{0}")]
    Memory(String),
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Inference(#[from] anyhow::Error),
}

impl AsrExecutionError {
    pub fn is_output_limit(&self) -> bool {
        matches!(self, Self::Inference(error)
            if matches!(error.downcast_ref::<AudioError>(), Some(AudioError::GenerationLimitExceeded)))
    }
}

#[derive(Default)]
pub struct AsrStreamingContext {
    pub raw_text: String,
    pub update_index: usize,
}

pub struct AsrRequest {
    pub pcm: PcmBuffer,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub max_tokens: usize,
    pub streaming_context: Option<AsrStreamingContext>,
    pub progress: Option<tokio::sync::mpsc::Sender<String>>,
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
    progress: Option<tokio::sync::mpsc::Sender<String>>,
}
impl SessionControl for Control {
    fn check(&self) -> ironmlx_audio::Result<()> {
        if self.cancelled.load(Ordering::SeqCst)
            || self.progress.as_ref().is_some_and(|p| p.is_closed())
        {
            return Err(AudioError::Cancelled);
        }
        if Instant::now() >= self.deadline {
            return Err(AudioError::DeadlineExceeded);
        }
        Ok(())
    }
}
type Reply = Result<Transcription, AsrExecutionError>;
struct Job {
    request: AsrRequest,
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
pub struct AsrRuntime {
    inner: Arc<Worker>,
}
impl AsrRuntime {
    pub(crate) async fn load(path: PathBuf) -> anyhow::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<Job>(ASR_QUEUE_CAPACITY);
        let (ready_tx, ready_rx) = oneshot::channel();
        let counts = Arc::new(Counts::default());
        let worker_counts = counts.clone();
        let usage = Arc::new(super::runtime_usage::ModelRuntimeUsageCounters::default());
        let worker_usage = usage.clone();
        let thread = std::thread::Builder::new()
            .name("ironmlx-asr".into())
            .spawn(move || {
                let load = || -> anyhow::Result<_> {
                    let device = mlx::Device::gpu(0);
                    mlx::set_default_device(device);
                    mlx::set_default_stream(mlx::new_stream(device)?);
                    Qwen3Asr::load(&path)
                };
                match load() {
                    Ok(mut model) => {
                        if ready_tx.send(Ok(model.weight_bytes() as usize)).is_ok() {
                            while let Ok(job) = receiver.recv() {
                                if job.reply.is_closed() {
                                    continue;
                                }
                                worker_counts.active.fetch_add(1, Ordering::SeqCst);
                                let _active = Active(worker_counts.clone());
                                let result = (|| -> Reply {
                                    job.control.check().map_err(anyhow::Error::from)?;
                                    model
                                        .validate_language(job.request.language.as_deref())
                                        .map_err(|e| {
                                            AsrExecutionError::InvalidInput(e.to_string())
                                        })?;
                                    // Byte-level BPE produces at most one token per prompt byte.
                                    // Include prompt/audio prefix KV, not only generated tokens.
                                    let context_bound =
                                        job.request.prompt.as_ref().map_or(0, String::len) + 512;
                                    let bytes = 512usize * 1024 * 1024
                                        + (context_bound + job.request.max_tokens.min(8192))
                                            * 114_688;
                                    let governor = global_process_memory_governor();
                                    governor.sample_process();
                                    let _reservation = governor.try_reserve(bytes, "Qwen3-ASR inference")
                                        .map_err(|e| AsrExecutionError::Memory(e.to_string()))?;
                                    let result = if let Some(context) = &job.request.streaming_context {
                                        model.transcribe_incremental(
                                            &job.request.pcm, job.request.language.as_deref(),
                                            job.request.prompt.as_deref(), job.request.max_tokens,
                                            &context.raw_text, context.update_index, &job.control,
                                        )
                                    } else {
                                        model.transcribe_with_progress(
                                            &job.request.pcm, job.request.language.as_deref(),
                                            job.request.prompt.as_deref(), job.request.max_tokens,
                                            &job.control, &mut |segment| {
                                                if let Some(progress) = &job.request.progress {
                                                    progress.try_send(segment.text.clone()).map_err(|_| anyhow::anyhow!("ASR stream consumer disconnected or too slow"))?;
                                                }
                                                Ok(())
                                            },
                                        )
                                    };
                                    // Finish submitted GPU work before releasing the lease/reservation.
                                    mlx::synchronize().map_err(anyhow::Error::from)?;
                                    let output = result?;
                                    worker_usage
                                        .record_output_tokens(output.generated_tokens as u64);
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
    pub async fn transcribe(&self, request: AsrRequest, lease: EngineLease) -> Reply {
        let lease = Arc::new(lease);
        let frames = request.pcm.frames().map_err(anyhow::Error::from)?;
        let rate = request.pcm.format.sample_rate as usize;
        if frames <= 30 * rate || request.streaming_context.is_some() {
            return self.transcribe_leased(request, lease).await;
        }
        // Yield the GPU worker between file segments so realtime jobs are not
        // trapped behind a whole recording. Shared guards retain input memory
        // and the model lease across cancellation and submitted GPU operations.
        let deadline = Duration::from_secs(180 + (frames / rate) as u64 * 2);
        let cancelled = Arc::new(AtomicBool::new(false));
        let _cancellation = Cancellation(cancelled.clone());
        let control = Control {
            cancelled,
            deadline: Instant::now() + deadline,
            progress: request.progress.clone(),
        };
        let memory = request.input_memory.clone();
        let slot = request.input_slot.clone();
        let pcm = request.pcm;
        let prepared = tokio::task::spawn_blocking(move || {
            let (_memory, _slot) = (memory, slot);
            ironmlx_audio::qwen3_asr::prepare_audio(&pcm, &control)
        })
        .await
        .map_err(|e| AsrExecutionError::Inference(e.into()))??;
        let duration = prepared.samples.len() as f64 / 16000.;
        let run = async {
            let mut output = Transcription {
                text: String::new(),
                language: String::new(),
                duration,
                generated_tokens: 0,
                segments: Vec::new(),
                raw_text: String::new(),
                stable_text: String::new(),
            };
            for range in prepared.segments {
                let start = range.start as f64 / 16000.;
                let end = range.end as f64 / 16000.;
                let mut samples = prepared.samples[range].to_vec();
                samples.resize(samples.len().max(201), 0.);
                let result = self
                    .transcribe_leased(
                        AsrRequest {
                            pcm: PcmBuffer {
                                format: ironmlx_audio::PcmFormat {
                                    sample_rate: 16000,
                                    channels: 1,
                                },
                                samples,
                            },
                            language: request.language.clone(),
                            prompt: request.prompt.clone(),
                            max_tokens: request.max_tokens,
                            streaming_context: None,
                            progress: None,
                            input_memory: request.input_memory.clone(),
                            input_slot: request.input_slot.clone(),
                        },
                        lease.clone(),
                    )
                    .await?;
                let delta =
                    ironmlx_audio::qwen3_asr::append_transcript(&mut output.text, &result.text);
                output.generated_tokens += result.generated_tokens;
                output
                    .segments
                    .push(ironmlx_audio::qwen3_asr::TranscriptSegment {
                        start,
                        end,
                        text: result.text.clone(),
                        language: result.language.clone(),
                    });
                if !result.language.is_empty()
                    && !output.language.split(',').any(|v| v == result.language)
                {
                    if !output.language.is_empty() {
                        output.language.push(',');
                    }
                    output.language.push_str(&result.language);
                }
                if let Some(progress) = &request.progress {
                    progress.try_send(delta).map_err(|_| {
                        AsrExecutionError::InvalidInput(
                            "Transcription stream disconnected or too slow".into(),
                        )
                    })?;
                }
                tokio::task::yield_now().await;
            }
            Ok(output)
        };
        tokio::time::timeout(deadline, run)
            .await
            .map_err(|_| AsrExecutionError::Timeout)?
    }

    pub async fn transcribe_leased(&self, request: AsrRequest, lease: Arc<EngineLease>) -> Reply {
        if !matches!(lease.engine(),EngineVariant::Asr(runtime) if Arc::ptr_eq(&self.inner,&runtime.inner))
        {
            return Err(AsrExecutionError::WrongEngine);
        }
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
        let progress = request.progress.clone();
        let job = Job {
            request,
            reply,
            control: Control {
                cancelled,
                deadline: Instant::now() + deadline,
                progress,
            },
            _pending: Pending(self.inner.counts.clone()),
            _lease: lease,
        };
        self.inner
            .sender
            .as_ref()
            .ok_or(AsrExecutionError::WorkerStopped)?
            .try_send(job)
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => AsrExecutionError::QueueFull,
                mpsc::TrySendError::Disconnected(_) => AsrExecutionError::WorkerStopped,
            })?;
        tokio::time::timeout(deadline, result)
            .await
            .map_err(|_| AsrExecutionError::Timeout)?
            .map_err(|_| AsrExecutionError::WorkerStopped)?
    }
}

#[cfg(test)]
mod settings_tests {
    use super::AsrSettings;
    #[test]
    fn asr_settings_enforce_service_bounds_and_strict_schema() {
        assert_eq!(AsrSettings::default().max_audio_duration_seconds, 1200);
        assert_eq!(AsrSettings::default().max_output_tokens, 1024);
        for (duration, tokens, valid) in [
            (1, 1, true),
            (7200, 8192, true),
            (0, 1024, false),
            (7201, 1024, false),
            (1200, 0, false),
            (1200, 8193, false),
        ] {
            assert_eq!(
                AsrSettings {
                    max_audio_duration_seconds: duration,
                    max_output_tokens: tokens
                }
                .validate()
                .is_ok(),
                valid
            );
        }
        for raw in [
            r#"{"max_audio_duration_seconds":1200,"max_output_tokens":1024,"temperature":1}"#,
            r#"{"max_audio_duration_seconds":1.5,"max_output_tokens":1024}"#,
        ] {
            assert!(serde_json::from_str::<AsrSettings>(raw).is_err());
        }
    }
}
