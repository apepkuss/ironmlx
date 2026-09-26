//! Worker-owned MLX decision inference using the shared model lifecycle.
use super::{engine_pool::EngineLease, process_memory::global_process_memory_governor};
use ironmlx_decision::{DecisionRequest, DecisionResponse, Laya};
use serde::Serialize;
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

pub const DECISION_QUEUE_CAPACITY: usize = 16;
const DECISION_METRICS_WINDOW: Duration = Duration::from_secs(60);
const DECISION_METRICS_SAMPLE_LIMIT: usize = 4_096;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct DecisionMetricsSnapshot {
    pub window_seconds: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub recent_completed_requests: usize,
    pub latency_ms_p50: Option<f64>,
    pub input_tokens_per_second: Option<f64>,
    pub questions_per_second: Option<f64>,
    pub last_request_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct DecisionPerformanceSample {
    completed_at: Instant,
    duration: Duration,
    input_tokens: u64,
    questions: u64,
}

#[derive(Debug, Default)]
struct DecisionMetrics {
    completed_requests: AtomicU64,
    failed_requests: AtomicU64,
    last_request_unix_ms: AtomicU64,
    samples: Mutex<VecDeque<DecisionPerformanceSample>>,
}

impl DecisionMetrics {
    fn record_success(&self, input_tokens: u64, questions: u64, duration: Duration) {
        self.record_success_at(
            input_tokens,
            questions,
            duration,
            Instant::now(),
            unix_time_ms(),
        );
    }

    fn record_success_at(
        &self,
        input_tokens: u64,
        questions: u64,
        duration: Duration,
        completed_at: Instant,
        completed_unix_ms: u64,
    ) {
        self.completed_requests.fetch_add(1, Ordering::Relaxed);
        self.last_request_unix_ms
            .store(completed_unix_ms, Ordering::Relaxed);
        let mut samples = self.samples();
        prune_decision_samples(&mut samples, completed_at);
        samples.push_back(DecisionPerformanceSample {
            completed_at,
            duration,
            input_tokens,
            questions,
        });
        while samples.len() > DECISION_METRICS_SAMPLE_LIMIT {
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

    fn snapshot(&self) -> DecisionMetricsSnapshot {
        self.snapshot_at(Instant::now())
    }

    fn snapshot_at(&self, now: Instant) -> DecisionMetricsSnapshot {
        let mut samples = self.samples();
        prune_decision_samples(&mut samples, now);
        let latency_ms_p50 = median(
            samples
                .iter()
                .map(|sample| sample.duration.as_secs_f64() * 1_000.0),
        );
        let input_tokens_per_second = median(
            samples
                .iter()
                .filter_map(|sample| rate_per_second(sample.input_tokens, sample.duration)),
        );
        let questions_per_second = median(
            samples
                .iter()
                .filter_map(|sample| rate_per_second(sample.questions, sample.duration)),
        );
        let last_request_unix_ms = self.last_request_unix_ms.load(Ordering::Relaxed);
        DecisionMetricsSnapshot {
            window_seconds: DECISION_METRICS_WINDOW.as_secs(),
            completed_requests: self.completed_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            recent_completed_requests: samples.len(),
            latency_ms_p50,
            input_tokens_per_second,
            questions_per_second,
            last_request_unix_ms: (last_request_unix_ms > 0).then_some(last_request_unix_ms),
        }
    }

    fn samples(&self) -> MutexGuard<'_, VecDeque<DecisionPerformanceSample>> {
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

fn prune_decision_samples(samples: &mut VecDeque<DecisionPerformanceSample>, now: Instant) {
    while samples.front().is_some_and(|sample| {
        now.saturating_duration_since(sample.completed_at) > DECISION_METRICS_WINDOW
    }) {
        samples.pop_front();
    }
}

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
    metrics_recorded: Arc<AtomicBool>,
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
    metrics: Arc<DecisionMetrics>,
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
        let metrics = Arc::new(DecisionMetrics::default());
        let worker_metrics = metrics.clone();
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
                                    if !job.metrics_recorded.swap(true, Ordering::SeqCst) {
                                        worker_metrics.record_failure();
                                    }
                                    continue;
                                }
                                if job.started.elapsed() > Duration::from_secs(60) {
                                    if !job.metrics_recorded.swap(true, Ordering::SeqCst) {
                                        worker_metrics.record_failure();
                                    }
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
                                match &result {
                                    Ok(response) => {
                                        let input_tokens = response.usage.input_tokens as u64;
                                        worker_usage.record_input_tokens(input_tokens);
                                        if !job.metrics_recorded.swap(true, Ordering::SeqCst) {
                                            worker_metrics.record_success(
                                                input_tokens,
                                                job.request.questions.len() as u64,
                                                job.started.elapsed(),
                                            );
                                        }
                                    }
                                    Err(_) => {
                                        if !job.metrics_recorded.swap(true, Ordering::SeqCst) {
                                            worker_metrics.record_failure();
                                        }
                                    }
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

    pub fn metrics(&self) -> DecisionMetricsSnapshot {
        self.inner.metrics.snapshot()
    }

    pub async fn predict(&self, request: DecisionRequest, lease: EngineLease) -> Reply {
        if !matches!(lease.engine(), super::engine_pool::EngineVariant::Decision(runtime) if Arc::ptr_eq(&self.inner, &runtime.inner))
        {
            self.inner.metrics.record_failure();
            return Err(DecisionExecutionError::WrongEngine);
        }
        if let Err(error) = request.validate() {
            self.inner.metrics.record_failure();
            return Err(DecisionExecutionError::InvalidInput(error.to_string()));
        }
        let (reply, result) = oneshot::channel();
        let metrics_recorded = Arc::new(AtomicBool::new(false));
        self.inner.counts.pending.fetch_add(1, Ordering::SeqCst);
        let job = Job {
            request,
            reply,
            started: Instant::now(),
            metrics_recorded: metrics_recorded.clone(),
            _pending: Pending(self.inner.counts.clone()),
            _lease: lease,
        };
        if let Err(error) = self
            .inner
            .sender
            .as_ref()
            .expect("live worker")
            .try_send(job)
        {
            self.inner.metrics.record_failure();
            return Err(match error {
                mpsc::TrySendError::Full(_) => DecisionExecutionError::QueueFull,
                mpsc::TrySendError::Disconnected(_) => DecisionExecutionError::WorkerStopped,
            });
        }
        match tokio::time::timeout(Duration::from_secs(120), result).await {
            Err(_) => {
                if !metrics_recorded.swap(true, Ordering::SeqCst) {
                    self.inner.metrics.record_failure();
                }
                Err(DecisionExecutionError::Timeout)
            }
            Ok(Err(_)) => {
                if !metrics_recorded.swap(true, Ordering::SeqCst) {
                    self.inner.metrics.record_failure();
                }
                Err(DecisionExecutionError::WorkerStopped)
            }
            Ok(Ok(reply)) => reply,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_metrics_report_recent_throughput_latency_and_totals() {
        let metrics = DecisionMetrics::default();
        let now = Instant::now();
        metrics.record_success_at(120, 3, Duration::from_millis(30), now, 1_700);
        metrics.record_success_at(
            200,
            5,
            Duration::from_millis(50),
            now + Duration::from_millis(10),
            1_800,
        );
        metrics.record_failure_at(1_900);

        let snapshot = metrics.snapshot_at(now + Duration::from_millis(10));
        assert_eq!(snapshot.window_seconds, 60);
        assert_eq!(snapshot.completed_requests, 2);
        assert_eq!(snapshot.failed_requests, 1);
        assert_eq!(snapshot.recent_completed_requests, 2);
        assert_eq!(snapshot.latency_ms_p50, Some(40.0));
        assert_eq!(snapshot.input_tokens_per_second, Some(4_000.0));
        assert_eq!(snapshot.questions_per_second, Some(100.0));
        assert_eq!(snapshot.last_request_unix_ms, Some(1_900));
    }

    #[test]
    fn decision_metrics_expire_recent_samples_without_losing_totals() {
        let metrics = DecisionMetrics::default();
        let now = Instant::now();
        metrics.record_success_at(10, 1, Duration::from_millis(10), now, 1_700);

        let snapshot =
            metrics.snapshot_at(now + DECISION_METRICS_WINDOW + Duration::from_millis(1));
        assert_eq!(snapshot.completed_requests, 1);
        assert_eq!(snapshot.recent_completed_requests, 0);
        assert_eq!(snapshot.latency_ms_p50, None);
        assert_eq!(snapshot.input_tokens_per_second, None);
        assert_eq!(snapshot.questions_per_second, None);
        assert_eq!(snapshot.last_request_unix_ms, Some(1_700));
    }
}
