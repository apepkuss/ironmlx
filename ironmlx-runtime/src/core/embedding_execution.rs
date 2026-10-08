//! Worker-owned text and image embeddings on the managed model lifecycle.
use super::{
    engine_pool::{EngineLease, EngineVariant},
    process_memory::global_process_memory_governor,
};
use ironmlx_lm::models::embedding_gemma2::{
    EmbeddingContent, EmbeddingGemma2Model, EmbeddingOutput, EmbeddingSample, MAX_BATCH_SIZE,
    SUPPORTED_DIMENSIONS,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc, Mutex, MutexGuard,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::oneshot;

pub const EMBEDDING_QUEUE_CAPACITY: usize = 8;
const EMBEDDING_METRICS_WINDOW: Duration = Duration::from_secs(60);
const EMBEDDING_METRICS_SAMPLE_LIMIT: usize = 4_096;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct EmbeddingMetricsSnapshot {
    pub window_seconds: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub recent_completed_requests: usize,
    pub latency_ms_p50: Option<f64>,
    pub input_tokens_per_second: Option<f64>,
    pub vectors_per_second: Option<f64>,
    pub last_request_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct EmbeddingPerformanceSample {
    completed_at: Instant,
    duration: Duration,
    processing_duration: Duration,
    input_tokens: u64,
    vectors: u64,
}

#[derive(Debug, Default)]
struct EmbeddingMetrics {
    completed_requests: AtomicU64,
    failed_requests: AtomicU64,
    last_request_unix_ms: AtomicU64,
    samples: Mutex<VecDeque<EmbeddingPerformanceSample>>,
}

impl EmbeddingMetrics {
    fn record_success(
        &self,
        input_tokens: u64,
        vectors: u64,
        duration: Duration,
        processing_duration: Duration,
    ) {
        self.record_success_at(
            EmbeddingPerformanceSample {
                completed_at: Instant::now(),
                duration,
                processing_duration,
                input_tokens,
                vectors,
            },
            unix_time_ms(),
        );
    }

    fn record_success_at(&self, sample: EmbeddingPerformanceSample, completed_unix_ms: u64) {
        self.completed_requests.fetch_add(1, Ordering::Relaxed);
        self.last_request_unix_ms
            .store(completed_unix_ms, Ordering::Relaxed);
        let mut samples = self.samples();
        prune_embedding_samples(&mut samples, sample.completed_at);
        samples.push_back(sample);
        while samples.len() > EMBEDDING_METRICS_SAMPLE_LIMIT {
            samples.pop_front();
        }
    }

    fn record_failure(&self) {
        self.record_failure_at(unix_time_ms());
    }

    fn record_failure_at(&self, completed_unix_ms: u64) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.last_request_unix_ms
            .store(completed_unix_ms, Ordering::Relaxed);
    }

    fn snapshot(&self) -> EmbeddingMetricsSnapshot {
        self.snapshot_at(Instant::now())
    }

    fn snapshot_at(&self, now: Instant) -> EmbeddingMetricsSnapshot {
        let mut samples = self.samples();
        prune_embedding_samples(&mut samples, now);
        let latency_ms_p50 = median(
            samples
                .iter()
                .map(|sample| sample.duration.as_secs_f64() * 1_000.0),
        );
        let input_tokens_per_second =
            median(samples.iter().filter_map(|sample| {
                rate_per_second(sample.input_tokens, sample.processing_duration)
            }));
        let vectors_per_second = median(
            samples
                .iter()
                .filter_map(|sample| rate_per_second(sample.vectors, sample.processing_duration)),
        );
        let last_request_unix_ms = self.last_request_unix_ms.load(Ordering::Relaxed);
        EmbeddingMetricsSnapshot {
            window_seconds: EMBEDDING_METRICS_WINDOW.as_secs(),
            completed_requests: self.completed_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            recent_completed_requests: samples.len(),
            latency_ms_p50,
            input_tokens_per_second,
            vectors_per_second,
            last_request_unix_ms: (last_request_unix_ms > 0).then_some(last_request_unix_ms),
        }
    }

    fn samples(&self) -> MutexGuard<'_, VecDeque<EmbeddingPerformanceSample>> {
        self.samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

fn rate_per_second(units: u64, duration: Duration) -> Option<f64> {
    let seconds = duration.as_secs_f64();
    (units > 0 && seconds > 0.0).then_some(units as f64 / seconds)
}

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn prune_embedding_samples(samples: &mut VecDeque<EmbeddingPerformanceSample>, now: Instant) {
    while samples.front().is_some_and(|sample| {
        now.saturating_duration_since(sample.completed_at) > EMBEDDING_METRICS_WINDOW
    }) {
        samples.pop_front();
    }
}

// Caller timeouts and the GPU worker can finish the same job concurrently.
// Record exactly one outcome while retaining the job's lease through GPU work.
struct CompletionMetrics {
    metrics: Arc<EmbeddingMetrics>,
    recorded: AtomicBool,
}
impl CompletionMetrics {
    fn failure(&self) {
        if !self.recorded.swap(true, Ordering::SeqCst) {
            self.metrics.record_failure();
        }
    }
    fn success(&self, output: &EmbeddingOutput, latency: Duration, processing: Duration) {
        if !self.recorded.swap(true, Ordering::SeqCst) {
            self.metrics.record_success(
                output.input_tokens as u64,
                output.embeddings.len() as u64,
                latency,
                processing,
            );
        }
    }
}
/// Legacy text inputs and ordered multimodal samples on the same endpoint.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Text(String),
    Batch(Vec<String>),
    Document(EmbeddingDocument),
    Documents(Vec<EmbeddingItem>),
    #[serde(skip)]
    Prepared(Vec<EmbeddingSample>),
}
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingItem {
    Text(String),
    Document(EmbeddingDocument),
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingDocument {
    pub content: Vec<EmbeddingPart>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum EmbeddingPart {
    Text { text: String },
    ImageUrl { image_url: EmbeddingImageUrl },
    InputAudio { input_audio: EmbeddingAudioInput },
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingImageUrl {
    pub url: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingAudioInput {
    pub data: String,
    pub format: String,
}
impl EmbeddingInput {
    pub fn into_items(self) -> anyhow::Result<Vec<EmbeddingItem>> {
        Ok(match self {
            Self::Text(text) => vec![EmbeddingItem::Text(text)],
            Self::Batch(texts) => texts.into_iter().map(EmbeddingItem::Text).collect(),
            Self::Document(document) => vec![EmbeddingItem::Document(document)],
            Self::Documents(documents) => documents,
            Self::Prepared(_) => {
                anyhow::bail!("decoded embedding input cannot be used as an HTTP payload")
            }
        })
    }
    fn into_samples(self) -> anyhow::Result<Vec<EmbeddingSample>> {
        match self {
            Self::Prepared(samples) => Ok(samples),
            Self::Text(text) => Ok(vec![EmbeddingSample {
                content: vec![EmbeddingContent::Text(text)],
            }]),
            Self::Batch(texts) => Ok(texts
                .into_iter()
                .map(|text| EmbeddingSample {
                    content: vec![EmbeddingContent::Text(text)],
                })
                .collect()),
            _ => anyhow::bail!("media input must be decoded before embedding execution"),
        }
    }
}
impl EmbeddingDocument {
    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.content.is_empty() && self.content.len() <= 64,
            "input content requires 1..=64 parts"
        );
        let mut text_bytes = 0usize;
        let mut has_content = false;
        for part in &self.content {
            match part {
                EmbeddingPart::Text { text } => {
                    text_bytes = text_bytes.saturating_add(text.len());
                    has_content |= !text.trim().is_empty();
                }
                EmbeddingPart::ImageUrl { image_url } => {
                    anyhow::ensure!(
                        !image_url.url.is_empty(),
                        "image input URL must not be empty"
                    );
                    has_content = true;
                }
                EmbeddingPart::InputAudio { input_audio } => {
                    anyhow::ensure!(
                        !input_audio.data.is_empty()
                            && ["wav", "flac", "mp3"].contains(&input_audio.format.as_str()),
                        "audio input requires base64 data and format wav, flac or mp3"
                    );
                    has_content = true;
                }
            }
        }
        anyhow::ensure!(
            has_content && text_bytes <= 128 * 1024,
            "input content must be nonempty with at most 128 KiB text"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    pub dimensions: Option<usize>,
    pub encoding_format: Option<String>,
}
impl EmbeddingRequest {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.model.trim().is_empty(), "model is required");
        let validate_text = |text: &str| -> anyhow::Result<()> {
            anyhow::ensure!(
                !text.trim().is_empty() && text.len() <= 128 * 1024,
                "input texts must be nonempty and at most 128 KiB each"
            );
            Ok(())
        };
        let count = match &self.input {
            EmbeddingInput::Text(text) => {
                validate_text(text)?;
                1
            }
            EmbeddingInput::Batch(texts) => {
                for text in texts {
                    validate_text(text)?;
                }
                texts.len()
            }
            EmbeddingInput::Document(document) => {
                document.validate()?;
                1
            }
            EmbeddingInput::Documents(documents) => {
                for item in documents {
                    match item {
                        EmbeddingItem::Text(text) => validate_text(text)?,
                        EmbeddingItem::Document(document) => document.validate()?,
                    }
                }
                documents.len()
            }
            EmbeddingInput::Prepared(samples) => samples.len(),
        };
        anyhow::ensure!(
            count > 0 && count <= MAX_BATCH_SIZE,
            "input requires 1..=32 samples"
        );
        anyhow::ensure!(
            SUPPORTED_DIMENSIONS.contains(&self.dimensions.unwrap_or(768)),
            "dimensions must be 128, 256, 512 or 768"
        );
        anyhow::ensure!(
            self.encoding_format
                .as_deref()
                .is_none_or(|s| s == "float" || s == "base64"),
            "encoding_format must be float or base64"
        );
        Ok(())
    }
}
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingExecutionError {
    #[error("embedding queue is full")]
    QueueFull,
    #[error("embedding worker is unavailable")]
    WorkerStopped,
    #[error("embedding request timed out")]
    Timeout,
    #[error("embedding execution requires a lease for the same worker")]
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
type Reply = Result<EmbeddingOutput, EmbeddingExecutionError>;
struct Job {
    request: EmbeddingRequest,
    reply: oneshot::Sender<Reply>,
    started: Instant,
    completion: Arc<CompletionMetrics>,
    _pending: Pending,
    _lease: EngineLease,
}
#[derive(Clone)]
pub struct EmbeddingRuntime {
    inner: Arc<Worker>,
}
struct Worker {
    sender: Option<mpsc::SyncSender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
    counts: Arc<Counts>,
    weight_bytes: usize,
    usage: Arc<super::runtime_usage::ModelRuntimeUsageCounters>,
    metrics: Arc<EmbeddingMetrics>,
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
impl EmbeddingRuntime {
    pub(crate) async fn load(path: PathBuf) -> anyhow::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<Job>(EMBEDDING_QUEUE_CAPACITY);
        let (ready_tx, ready_rx) = oneshot::channel();
        let counts = Arc::new(Counts::default());
        let worker_counts = counts.clone();
        let usage = Arc::new(super::runtime_usage::ModelRuntimeUsageCounters::default());
        let worker_usage = usage.clone();
        let metrics = Arc::new(EmbeddingMetrics::default());
        let thread = std::thread::Builder::new()
            .name("ironmlx-embedding".into())
            .spawn(move || {
                let load = || -> anyhow::Result<_> {
                    let device = mlx::Device::gpu(0);
                    mlx::set_default_device(device);
                    mlx::set_default_stream(mlx::new_stream(device)?);
                    EmbeddingGemma2Model::load(&path)
                };
                match load() {
                    Ok(model) => {
                        if ready_tx.send(Ok(model.weight_bytes())).is_ok() {
                            while let Ok(job) = receiver.recv() {
                                if job.reply.is_closed() {
                                    job.completion.failure();
                                    continue;
                                }
                                if job.started.elapsed() > Duration::from_secs(60) {
                                    job.completion.failure();
                                    let _ = job.reply.send(Err(EmbeddingExecutionError::Timeout));
                                    continue;
                                }
                                worker_counts.active.fetch_add(1, Ordering::SeqCst);
                                let _active = Active(worker_counts.clone());
                                let processing_started = Instant::now();
                                let result = (|| -> Reply {
                                    // Serial sequences cap peak memory independently of the request batch size.
                                    let governor = global_process_memory_governor();
                                    governor.refresh_process();
                                    let _memory = governor
                                        .try_reserve(
                                            1536 * 1024 * 1024,
                                            "EmbeddingGemma 2 inference",
                                        )
                                        .map_err(|e| {
                                            EmbeddingExecutionError::Inference(e.to_string())
                                        })?;
                                    let samples =
                                        job.request.input.into_samples().map_err(|e| {
                                            EmbeddingExecutionError::InvalidInput(e.to_string())
                                        })?;
                                    let result = model
                                        .encode_inputs(
                                            &samples,
                                            job.request.dimensions.unwrap_or(768),
                                        )
                                        .map_err(|e| {
                                            let message = e.to_string();
                                            if message.contains("input")
                                                || message.contains("token ID")
                                            {
                                                EmbeddingExecutionError::InvalidInput(message)
                                            } else {
                                                EmbeddingExecutionError::Inference(message)
                                            }
                                        });
                                    mlx::synchronize().map_err(|e| {
                                        EmbeddingExecutionError::Inference(e.to_string())
                                    })?;
                                    result
                                })();
                                if let Ok(output) = &result {
                                    worker_usage.record_input_tokens(output.input_tokens as u64);
                                    job.completion.success(
                                        output,
                                        job.started.elapsed(),
                                        processing_started.elapsed(),
                                    );
                                } else {
                                    job.completion.failure();
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
                metrics,
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
    pub fn metrics(&self) -> EmbeddingMetricsSnapshot {
        self.inner.metrics.snapshot()
    }
    pub async fn encode(&self, request: EmbeddingRequest, lease: EngineLease) -> Reply {
        if !matches!(lease.engine(), EngineVariant::Embedding(runtime) if Arc::ptr_eq(&self.inner,&runtime.inner))
        {
            self.inner.metrics.record_failure();
            return Err(EmbeddingExecutionError::WrongEngine);
        }
        if let Err(e) = request.validate() {
            self.inner.metrics.record_failure();
            return Err(EmbeddingExecutionError::InvalidInput(e.to_string()));
        }
        let (reply, result) = oneshot::channel();
        let completion = Arc::new(CompletionMetrics {
            metrics: self.inner.metrics.clone(),
            recorded: AtomicBool::new(false),
        });
        self.inner.counts.pending.fetch_add(1, Ordering::SeqCst);
        let job = Job {
            request,
            reply,
            started: Instant::now(),
            completion: completion.clone(),
            _pending: Pending(self.inner.counts.clone()),
            _lease: lease,
        };
        if let Err(error) = self
            .inner
            .sender
            .as_ref()
            .expect("live embedding worker")
            .try_send(job)
        {
            completion.failure();
            return Err(match error {
                mpsc::TrySendError::Full(_) => EmbeddingExecutionError::QueueFull,
                mpsc::TrySendError::Disconnected(_) => EmbeddingExecutionError::WorkerStopped,
            });
        }
        match tokio::time::timeout(Duration::from_secs(120), result).await {
            Err(_) => {
                completion.failure();
                Err(EmbeddingExecutionError::Timeout)
            }
            Ok(Err(_)) => {
                completion.failure();
                Err(EmbeddingExecutionError::WorkerStopped)
            }
            Ok(Ok(reply)) => reply,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn embedding_metrics_separate_queue_latency_from_batch_throughput_and_expire() {
        let metrics = EmbeddingMetrics::default();
        let now = Instant::now();
        for (tokens, vectors, latency, processing) in [(120, 3, 90, 30), (200, 5, 150, 50)] {
            metrics.record_success_at(
                EmbeddingPerformanceSample {
                    completed_at: now,
                    duration: Duration::from_millis(latency),
                    processing_duration: Duration::from_millis(processing),
                    input_tokens: tokens,
                    vectors,
                },
                1700,
            );
        }
        metrics.record_failure_at(1800);
        let snapshot = metrics.snapshot_at(now);
        assert_eq!(snapshot.completed_requests, 2);
        assert_eq!(snapshot.failed_requests, 1);
        assert_eq!(snapshot.recent_completed_requests, 2);
        assert_eq!(snapshot.latency_ms_p50, Some(120.0));
        assert_eq!(snapshot.input_tokens_per_second, Some(4000.0));
        assert_eq!(snapshot.vectors_per_second, Some(100.0));
        let expired =
            metrics.snapshot_at(now + EMBEDDING_METRICS_WINDOW + Duration::from_millis(1));
        assert_eq!(expired.completed_requests, 2);
        assert_eq!(expired.failed_requests, 1);
        assert_eq!(expired.last_request_unix_ms, Some(1800));
        assert_eq!(expired.recent_completed_requests, 0);
        assert_eq!(expired.latency_ms_p50, None);
        assert_eq!(expired.input_tokens_per_second, None);
        assert_eq!(expired.vectors_per_second, None);
        // No throughput sample may divide by zero, even for an immediate result.
        assert_eq!(rate_per_second(3, Duration::ZERO), None);
    }
    #[test]
    fn embedding_completion_records_only_one_outcome_when_timeout_races_worker() {
        for timeout_first in [false, true] {
            let metrics = Arc::new(EmbeddingMetrics::default());
            let completion = CompletionMetrics {
                metrics: metrics.clone(),
                recorded: AtomicBool::new(false),
            };
            let output = EmbeddingOutput {
                embeddings: vec![vec![0.0]; 3],
                input_tokens: 120,
            };
            if timeout_first {
                completion.failure();
            }
            completion.success(
                &output,
                Duration::from_millis(90),
                Duration::from_millis(30),
            );
            completion.failure();
            completion.failure();
            let snapshot = metrics.snapshot();
            assert_eq!(snapshot.completed_requests, u64::from(!timeout_first));
            assert_eq!(snapshot.failed_requests, u64::from(timeout_first));
            assert_eq!(
                snapshot.recent_completed_requests,
                usize::from(!timeout_first)
            );
        }
    }
    #[test]
    fn embedding_metrics_bound_sample_retention() {
        let metrics = EmbeddingMetrics::default();
        let now = Instant::now();
        for _ in 0..EMBEDDING_METRICS_SAMPLE_LIMIT + 10 {
            metrics.record_success_at(
                EmbeddingPerformanceSample {
                    completed_at: now,
                    duration: Duration::from_millis(1),
                    processing_duration: Duration::from_millis(1),
                    input_tokens: 1,
                    vectors: 1,
                },
                1700,
            );
        }
        let snapshot = metrics.snapshot_at(now);
        assert_eq!(
            snapshot.completed_requests,
            (EMBEDDING_METRICS_SAMPLE_LIMIT + 10) as u64
        );
        assert_eq!(
            snapshot.recent_completed_requests,
            EMBEDDING_METRICS_SAMPLE_LIMIT
        );
    }
    #[test]
    fn validates_embedding_request_bounds_and_formats() {
        let mut r: EmbeddingRequest = serde_json::from_value(
            serde_json::json!({"model":"test","input":["hello","你好"],"dimensions":128}),
        )
        .unwrap();
        assert!(r.validate().is_ok());
        r.dimensions = Some(129);
        assert!(r.validate().is_err());
        r.dimensions = None;
        r.encoding_format = Some("hex".into());
        assert!(r.validate().is_err());
        r.encoding_format = None;
        r.input = EmbeddingInput::Batch(vec![]);
        assert!(r.validate().is_err());
        r.input = EmbeddingInput::Batch(vec!["x".into(); 33]);
        assert!(r.validate().is_err());
        r.input = EmbeddingInput::Text(" ".into());
        assert!(r.validate().is_err());
    }
    #[test]
    fn validates_ordered_embedding_content_and_mixed_batches() {
        for value in [
            serde_json::json!({"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}),
            serde_json::json!(["hello",{"content":[{"type":"text","text":"label"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]),
        ] {
            let request: EmbeddingRequest =
                serde_json::from_value(serde_json::json!({"model":"test","input":value})).unwrap();
            assert!(request.validate().is_ok());
        }
        for value in [
            serde_json::json!({"content":[]}),
            serde_json::json!({"content":[{"type":"text","text":" "}]}),
            serde_json::json!({"content":[{"type":"audio","url":"test"}]}),
            serde_json::json!({"content":[{"type":"image_url","image_url":{"url":"test","detail":"high"}}]}),
            serde_json::json!({"content":[{"type":"text","text":"x","unexpected":true}]}),
        ] {
            let request = serde_json::from_value::<EmbeddingRequest>(
                serde_json::json!({"model":"test","input":value}),
            );
            assert!(request.is_err() || request.unwrap().validate().is_err());
        }
    }
}
