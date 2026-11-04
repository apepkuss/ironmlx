use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, MutexGuard, OnceLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use tokio::sync::Notify;

use crate::core::runtime_activity::{FinishedRequest, RequestOutcome, RuntimeActivityRecorder};

use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PrefixCacheUsageSnapshot {
    pub hit_tokens: u64,
    pub eligible_tokens: u64,
}

const PERFORMANCE_WINDOW: Duration = Duration::from_secs(60);
const PERFORMANCE_SAMPLE_LIMIT: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ModelRuntimePerformanceSnapshot {
    pub window_seconds: u64,
    pub completed_requests: usize,
    pub live_decode_tokens_per_second: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub session_decode_tokens_per_second: Option<f64>,
    pub ttft_ms: Option<f64>,
}

impl Default for ModelRuntimePerformanceSnapshot {
    fn default() -> Self {
        Self {
            window_seconds: PERFORMANCE_WINDOW.as_secs(),
            completed_requests: 0,
            live_decode_tokens_per_second: None,
            prefill_tokens_per_second: None,
            decode_tokens_per_second: None,
            session_decode_tokens_per_second: None,
            ttft_ms: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct ModelRuntimeUsageSnapshot {
    pub cumulative_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache: Option<PrefixCacheUsageSnapshot>,
    pub performance: ModelRuntimePerformanceSnapshot,
}

#[derive(Debug, Clone, Copy)]
struct ModelRuntimePerformanceSample {
    completed_at: Instant,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: Option<f64>,
    ttft_ms: f64,
}

#[derive(Debug, Clone, Copy)]
struct ActiveDecodeRequest {
    first_token_at: Instant,
    decode_tokens: u64,
}

#[derive(Debug, Default)]
struct ModelRuntimeDecodeState {
    active_requests: HashMap<u64, ActiveDecodeRequest>,
    session_decode_tokens: u64,
    session_decode_duration: Duration,
}

/// Process-wide recorder these counters feed, with the model they belong to.
struct ActivityLink {
    recorder: Arc<RuntimeActivityRecorder>,
    model_id: Arc<str>,
}

impl std::fmt::Debug for ActivityLink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivityLink")
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default)]
pub struct ModelRuntimeUsageCounters {
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    prefix_cache_hit_tokens: AtomicU64,
    prefix_cache_eligible_tokens: AtomicU64,
    next_request_id: AtomicU64,
    /// Requests past their first token and not yet settled. One counter,
    /// not in-flight minus awaiting: two separate loads can interleave with
    /// a start or settle and momentarily report one decoding request too many.
    decoding_requests: AtomicU64,
    request_waker: OnceLock<Arc<Notify>>,
    activity: OnceLock<ActivityLink>,
    decode_state: Mutex<ModelRuntimeDecodeState>,
    performance_samples: Mutex<VecDeque<ModelRuntimePerformanceSample>>,
}

#[derive(Debug)]
#[must_use = "request performance samples must be completed or intentionally dropped"]
pub struct ModelRuntimeRequestTracker {
    counters: Arc<ModelRuntimeUsageCounters>,
    request_id: u64,
    input_tokens: u64,
    started_at: Instant,
    first_token_at: Option<Instant>,
    last_output_at: Option<Instant>,
    max_output_event_gap: Duration,
    output_tokens: u64,
    finished: bool,
    endpoint: Option<&'static str>,
    persistence: Option<PersistenceRequest>,
}

#[derive(Debug)]
struct PersistenceRequest {
    model: String,
    route: &'static str,
    concurrency: usize,
    queued_at_start: u64,
    hits_at_start: u64,
    overlapped: bool,
}

impl ModelRuntimeUsageCounters {
    pub fn record_input_tokens(&self, tokens: u64) {
        self.input_tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn record_output_tokens(&self, tokens: u64) {
        self.output_tokens.fetch_add(tokens, Ordering::Relaxed);
        if let Some(link) = self.activity.get() {
            link.recorder.record_output_tokens(tokens);
        }
    }

    /// `endpoint` is the API path the request came in on, shown in the
    /// recent-request log.
    pub fn start_request(
        self: &Arc<Self>,
        input_tokens: u64,
        started_at: Instant,
        endpoint: Option<&'static str>,
    ) -> ModelRuntimeRequestTracker {
        self.record_input_tokens(input_tokens);
        if let Some(waker) = self.request_waker.get() {
            waker.notify_one();
        }
        if let Some(link) = self.activity.get() {
            link.recorder.record_request_started();
        }
        ModelRuntimeRequestTracker {
            counters: self.clone(),
            request_id: self.next_request_id.fetch_add(1, Ordering::Relaxed),
            input_tokens,
            started_at,
            first_token_at: None,
            last_output_at: None,
            max_output_event_gap: Duration::ZERO,
            output_tokens: 0,
            finished: false,
            persistence: None,
            endpoint,
        }
    }

    /// Wakes `waker` once per started request. Only the first attachment takes
    /// effect.
    pub fn attach_request_waker(&self, waker: Arc<Notify>) {
        let _ = self.request_waker.set(waker);
    }

    /// Feeds output tokens, request starts, first tokens and finished
    /// requests of `model_id` to `recorder`. Only the first attachment takes
    /// effect.
    pub fn attach_activity(&self, recorder: Arc<RuntimeActivityRecorder>, model_id: Arc<str>) {
        let _ = self.activity.set(ActivityLink { recorder, model_id });
    }

    /// Requests that have produced their first token and are not finished.
    pub fn decoding_requests(&self) -> u64 {
        self.decoding_requests.load(Ordering::Relaxed)
    }

    fn settle_request(&self, had_first_token: bool) {
        if had_first_token {
            self.decoding_requests.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub fn record_prefix_cache_lookup(&self, eligible_tokens: u64, hit_tokens: u64) {
        self.record_prefix_cache_eligible_tokens(eligible_tokens);
        self.record_prefix_cache_hit_tokens(hit_tokens.min(eligible_tokens));
    }

    pub fn record_prefix_cache_eligible_tokens(&self, tokens: u64) {
        self.prefix_cache_eligible_tokens
            .fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn record_prefix_cache_hit_tokens(&self, tokens: u64) {
        self.prefix_cache_hit_tokens
            .fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn snapshot(&self, prefix_cache_applicable: bool) -> ModelRuntimeUsageSnapshot {
        self.snapshot_at(prefix_cache_applicable, Instant::now())
    }

    fn snapshot_at(
        &self,
        prefix_cache_applicable: bool,
        now: Instant,
    ) -> ModelRuntimeUsageSnapshot {
        let input_tokens = self.input_tokens.load(Ordering::Relaxed);
        let output_tokens = self.output_tokens.load(Ordering::Relaxed);
        ModelRuntimeUsageSnapshot {
            cumulative_tokens: input_tokens.saturating_add(output_tokens),
            input_tokens,
            output_tokens,
            prefix_cache: prefix_cache_applicable.then(|| PrefixCacheUsageSnapshot {
                hit_tokens: self.prefix_cache_hit_tokens.load(Ordering::Relaxed),
                eligible_tokens: self.prefix_cache_eligible_tokens.load(Ordering::Relaxed),
            }),
            performance: self.performance_snapshot_at(now),
        }
    }

    fn record_performance_sample(&self, sample: ModelRuntimePerformanceSample) {
        let mut samples = self.performance_samples();
        prune_performance_samples(&mut samples, sample.completed_at);
        samples.push_back(sample);
        while samples.len() > PERFORMANCE_SAMPLE_LIMIT {
            samples.pop_front();
        }
    }

    fn performance_snapshot_at(&self, now: Instant) -> ModelRuntimePerformanceSnapshot {
        let mut samples = self.performance_samples();
        prune_performance_samples(&mut samples, now);
        let completed_requests = samples.len();
        let prefill_tokens_per_second = median(
            samples
                .iter()
                .map(|sample| sample.prefill_tokens_per_second),
        );
        let decode_tokens_per_second = median(
            samples
                .iter()
                .filter_map(|sample| sample.decode_tokens_per_second),
        );
        let ttft_ms = median(samples.iter().map(|sample| sample.ttft_ms));
        drop(samples);

        let decode_state = self.decode_state();
        let (live_decode_sum, live_decode_samples) = decode_state
            .active_requests
            .values()
            .filter_map(|request| {
                tokens_per_second(
                    request.decode_tokens,
                    now.saturating_duration_since(request.first_token_at),
                )
            })
            .fold((0.0, 0_usize), |(sum, count), rate| (sum + rate, count + 1));
        let live_decode_tokens_per_second = (live_decode_samples > 0).then_some(live_decode_sum);
        let session_decode_tokens_per_second = tokens_per_second(
            decode_state.session_decode_tokens,
            decode_state.session_decode_duration,
        );

        ModelRuntimePerformanceSnapshot {
            window_seconds: PERFORMANCE_WINDOW.as_secs(),
            completed_requests,
            live_decode_tokens_per_second,
            prefill_tokens_per_second,
            decode_tokens_per_second,
            session_decode_tokens_per_second,
            ttft_ms,
        }
    }

    fn update_active_decode_request(
        &self,
        request_id: u64,
        first_token_at: Instant,
        decode_tokens: u64,
    ) {
        self.decode_state().active_requests.insert(
            request_id,
            ActiveDecodeRequest {
                first_token_at,
                decode_tokens,
            },
        );
    }

    fn finish_decode_request(
        &self,
        request_id: u64,
        decode_tokens: u64,
        decode_duration: Duration,
    ) {
        let mut state = self.decode_state();
        state.active_requests.remove(&request_id);
        if decode_tokens > 0 && !decode_duration.is_zero() {
            state.session_decode_tokens = state.session_decode_tokens.saturating_add(decode_tokens);
            state.session_decode_duration = state
                .session_decode_duration
                .saturating_add(decode_duration);
        }
    }

    fn remove_active_decode_request(&self, request_id: u64) {
        self.decode_state().active_requests.remove(&request_id);
    }

    fn decode_state(&self) -> MutexGuard<'_, ModelRuntimeDecodeState> {
        self.decode_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn performance_samples(&self) -> MutexGuard<'_, VecDeque<ModelRuntimePerformanceSample>> {
        self.performance_samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl ModelRuntimeRequestTracker {
    pub(crate) fn with_persistence_context(
        mut self,
        model: &str,
        route: &'static str,
        concurrency: usize,
    ) -> Self {
        self.persistence = Some(PersistenceRequest {
            model: model.into(),
            route,
            concurrency: concurrency.max(1),
            queued_at_start: crate::core::cache::prefix_save::snapshot()
                .stats
                .queued_total,
            hits_at_start: self
                .counters
                .prefix_cache_hit_tokens
                .load(Ordering::Relaxed),
            overlapped: crate::core::cache::prefix_save::busy(),
        });
        self
    }

    pub fn record_output_tokens(&mut self, tokens: u64) {
        self.record_output_tokens_at(tokens, Instant::now());
    }

    pub fn complete(mut self) {
        self.complete_at(Instant::now());
    }

    fn record_output_tokens_at(&mut self, tokens: u64, now: Instant) {
        if tokens == 0 {
            return;
        }
        // These are output events, which can contain several speculative tokens.
        // Keep prefill and time after the final output out of the event gap.
        if let Some(previous) = self.last_output_at {
            self.max_output_event_gap = self
                .max_output_event_gap
                .max(now.saturating_duration_since(previous));
        }
        self.last_output_at = Some(now);
        if let Some(p) = self.persistence.as_mut() {
            p.overlapped |= crate::core::cache::prefix_save::busy();
            let state = self.counters.decode_state();
            let active = state.active_requests.len()
                + usize::from(!state.active_requests.contains_key(&self.request_id));
            p.concurrency = p.concurrency.max(active);
        }
        if self.first_token_at.is_none() {
            self.counters
                .decoding_requests
                .fetch_add(1, Ordering::Relaxed);
            if let Some(link) = self.counters.activity.get() {
                link.recorder.record_first_token(
                    self.input_tokens,
                    now.saturating_duration_since(self.started_at),
                );
            }
        }
        let first_token_at = *self.first_token_at.get_or_insert(now);
        self.output_tokens = self.output_tokens.saturating_add(tokens);
        self.counters.record_output_tokens(tokens);
        self.counters.update_active_decode_request(
            self.request_id,
            first_token_at,
            self.output_tokens.saturating_sub(1),
        );
    }

    fn complete_at(&mut self, completed_at: Instant) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.counters.settle_request(self.first_token_at.is_some());
        self.record_finished(RequestOutcome::Completed, completed_at);
        let Some(first_token_at) = self.first_token_at else {
            self.counters.remove_active_decode_request(self.request_id);
            return;
        };
        let decode_tokens = self.output_tokens.saturating_sub(1);
        let decode_duration = completed_at.saturating_duration_since(first_token_at);
        self.counters
            .finish_decode_request(self.request_id, decode_tokens, decode_duration);
        let ttft = first_token_at.saturating_duration_since(self.started_at);
        let Some(prefill_tokens_per_second) = tokens_per_second(self.input_tokens, ttft) else {
            return;
        };
        if let Some(p) = &self.persistence {
            let persistence = crate::core::cache::prefix_save::snapshot();
            let overlapped = p.overlapped || persistence.stats.queued_total > p.queued_at_start;
            tracing::debug!(
                model = %p.model,
                route = p.route,
                request_id = self.request_id,
                input_tokens = self.input_tokens,
                output_tokens = self.output_tokens,
                concurrency = p.concurrency,
                ttft_ms = ttft.as_secs_f64() * 1000.,
                decode_ms = decode_duration.as_secs_f64() * 1000.,
                max_output_event_gap_ms = self.max_output_event_gap.as_secs_f64() * 1000.,
                completion_tail_ms = completed_at
                    .saturating_duration_since(self.last_output_at.unwrap_or(first_token_at))
                    .as_secs_f64() * 1000.,
                persistence_overlapped = overlapped,
                persistence_pending = persistence.stats.pending_jobs,
                persistence_status = %persistence.stats.last_status,
                "request latency breakdown"
            );
            crate::core::cache::prefix_save::observe_request(
                &p.model,
                p.route,
                self.input_tokens,
                p.concurrency,
                ttft.as_secs_f64() * 1000.,
                (decode_tokens > 0)
                    .then(|| decode_duration.as_secs_f64() * 1000. / decode_tokens as f64),
                overlapped,
                self.counters
                    .prefix_cache_hit_tokens
                    .load(Ordering::Relaxed)
                    > p.hits_at_start,
            );
        }
        self.counters
            .record_performance_sample(ModelRuntimePerformanceSample {
                completed_at,
                prefill_tokens_per_second,
                decode_tokens_per_second: tokens_per_second(decode_tokens, decode_duration),
                ttft_ms: ttft.as_secs_f64() * 1_000.0,
            });
    }

    /// Adds this request to the attached recorder's recent-request log.
    fn record_finished(&self, outcome: RequestOutcome, finished_at: Instant) {
        let Some(link) = self.counters.activity.get() else {
            return;
        };
        link.recorder.record_finished(
            FinishedRequest {
                model: &link.model_id,
                endpoint: self.endpoint,
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                ttft: self
                    .first_token_at
                    .map(|first| first.saturating_duration_since(self.started_at)),
                decode_duration: self
                    .first_token_at
                    .map(|first| finished_at.saturating_duration_since(first)),
                duration: finished_at.saturating_duration_since(self.started_at),
                outcome,
            },
            unix_time_ms(),
        );
    }
}

impl Drop for ModelRuntimeRequestTracker {
    fn drop(&mut self) {
        if !self.finished {
            self.counters.settle_request(self.first_token_at.is_some());
            self.counters.remove_active_decode_request(self.request_id);
            self.record_finished(RequestOutcome::Aborted, Instant::now());
        }
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn prune_performance_samples(samples: &mut VecDeque<ModelRuntimePerformanceSample>, now: Instant) {
    while samples.front().is_some_and(|sample| {
        now.saturating_duration_since(sample.completed_at) > PERFORMANCE_WINDOW
    }) {
        samples.pop_front();
    }
}

fn tokens_per_second(tokens: u64, duration: Duration) -> Option<f64> {
    (tokens > 0 && !duration.is_zero()).then(|| tokens as f64 / duration.as_secs_f64())
}

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[middle - 1] + values[middle]) / 2.0)
    } else {
        Some(values[middle])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::runtime_activity::{ActivitySample, RequestOutcome, RuntimeActivityRecorder};

    #[test]
    fn attached_counters_feed_recorder() {
        let recorder = RuntimeActivityRecorder::new();
        let counters = Arc::new(ModelRuntimeUsageCounters::default());
        counters.attach_activity(recorder.clone(), Arc::from("model-a"));
        // Only the first attachment takes effect.
        let other = RuntimeActivityRecorder::new();
        counters.attach_activity(other.clone(), Arc::from("model-b"));

        let mut request = counters.start_request(12, Instant::now(), Some("/v1/messages"));
        request.record_output_tokens(1);
        request.record_output_tokens(2);
        request.complete();

        let point = recorder.close_bucket(0, ActivitySample::default());
        assert_eq!(point.output_tokens, 3);
        assert_eq!(point.requests_started, 1);
        assert_eq!(point.requests_finished, 1);
        assert_eq!(point.first_tokens, 1);
        let recent = recorder.recent().requests;
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].model, "model-a");
        assert_eq!(recent[0].endpoint.as_deref(), Some("/v1/messages"));
        assert_eq!(recent[0].input_tokens, 12);
        assert_eq!(recent[0].output_tokens, 3);
        assert_eq!(recent[0].outcome, RequestOutcome::Completed);
        assert!(recent[0].ttft_ms.is_some());
        assert!(other.recent().requests.is_empty());
    }

    #[test]
    fn dropped_tracker_records_aborted() {
        let recorder = RuntimeActivityRecorder::new();
        let counters = Arc::new(ModelRuntimeUsageCounters::default());
        counters.attach_activity(recorder.clone(), Arc::from("m"));

        let mut request = counters.start_request(5, Instant::now(), None);
        request.record_output_tokens(1);
        drop(request);
        let silent = counters.start_request(5, Instant::now(), None);
        drop(silent);

        let recent = recorder.recent().requests;
        assert_eq!(recent.len(), 2);
        assert!(recent
            .iter()
            .all(|r| r.outcome == RequestOutcome::Aborted && r.endpoint.is_none()));
        assert_eq!(recent[0].output_tokens, 0);
        assert_eq!(recent[0].ttft_ms, None);
        assert_eq!(recent[1].output_tokens, 1);
        let point = recorder.close_bucket(0, ActivitySample::default());
        assert_eq!(point.requests_started, 2);
        assert_eq!(point.requests_finished, 2);
        assert_eq!(point.output_tokens, 1);
    }

    #[test]
    fn unattached_counters_do_not_record() {
        let recorder = RuntimeActivityRecorder::new();
        let counters = Arc::new(ModelRuntimeUsageCounters::default());
        let mut request = counters.start_request(5, Instant::now(), Some("/v1/messages"));
        request.record_output_tokens(4);
        request.complete();
        counters.record_output_tokens(2);
        drop(counters.start_request(5, Instant::now(), None));

        let point = recorder.close_bucket(0, ActivitySample::default());
        assert_eq!(point.output_tokens, 0);
        assert_eq!(point.requests_started, 0);
        assert_eq!(point.requests_finished, 0);
        assert!(recorder.recent().requests.is_empty());
        assert_eq!(counters.snapshot(false).output_tokens, 6);
    }

    #[test]
    fn decoding_requests_counts_requests_after_first_token_only() {
        let counters = Arc::new(ModelRuntimeUsageCounters::default());
        let t0 = Instant::now();
        let mut a = counters.start_request(10, t0, None);
        let b = counters.start_request(10, t0, None);
        assert_eq!(counters.decoding_requests(), 0);
        a.record_output_tokens(1);
        assert_eq!(counters.decoding_requests(), 1);
        a.complete();
        drop(b); // dropped before first token
        assert_eq!(counters.decoding_requests(), 0);
    }

    #[tokio::test]
    async fn start_request_wakes_attached_waker() {
        let counters = Arc::new(ModelRuntimeUsageCounters::default());
        let waker = Arc::new(tokio::sync::Notify::new());
        counters.attach_request_waker(waker.clone());
        let _r = counters.start_request(1, Instant::now(), None);
        tokio::time::timeout(Duration::from_millis(50), waker.notified())
            .await
            .expect("woken");
    }

    #[test]
    fn output_event_gap_excludes_prefill_empty_events_and_completion_tail() {
        let counters = Arc::new(ModelRuntimeUsageCounters::default());
        let start = Instant::now();
        let mut request = counters.start_request(100, start);
        request.record_output_tokens_at(1, start + Duration::from_secs(10));
        assert_eq!(request.max_output_event_gap, Duration::ZERO);
        request.record_output_tokens_at(0, start + Duration::from_secs(11));
        request.record_output_tokens_at(4, start + Duration::from_secs(13));
        request.record_output_tokens_at(1, start + Duration::from_secs(14));
        assert_eq!(request.max_output_event_gap, Duration::from_secs(3));
        request.complete_at(start + Duration::from_secs(20));
        assert_eq!(request.max_output_event_gap, Duration::from_secs(3));
        assert_eq!(
            request.last_output_at,
            Some(start + Duration::from_secs(14))
        );
        assert_eq!(counters.snapshot(true).output_tokens, 6);
    }

    #[test]
    fn snapshot_saturates_hits_to_eligible_tokens() {
        let counters = ModelRuntimeUsageCounters::default();
        counters.record_input_tokens(11);
        counters.record_output_tokens(7);
        counters.record_prefix_cache_lookup(5, 8);

        assert_eq!(
            counters.snapshot(true),
            ModelRuntimeUsageSnapshot {
                cumulative_tokens: 18,
                input_tokens: 11,
                output_tokens: 7,
                prefix_cache: Some(PrefixCacheUsageSnapshot {
                    hit_tokens: 5,
                    eligible_tokens: 5,
                }),
                performance: ModelRuntimePerformanceSnapshot::default(),
            }
        );
        assert_eq!(counters.snapshot(false).prefix_cache, None);
    }

    #[test]
    fn completed_request_records_deterministic_performance_sample() {
        let counters = Arc::new(ModelRuntimeUsageCounters::default());
        let started_at = Instant::now();
        let mut request = counters.start_request(100, started_at, None);
        request.record_output_tokens_at(1, started_at + Duration::from_millis(200));
        request.record_output_tokens_at(9, started_at + Duration::from_millis(1_100));
        let live_snapshot = counters.snapshot_at(true, started_at + Duration::from_millis(1_100));
        assert_eq!(
            live_snapshot.performance.live_decode_tokens_per_second,
            Some(10.0)
        );
        assert_eq!(
            live_snapshot.performance.session_decode_tokens_per_second,
            None
        );
        request.complete_at(started_at + Duration::from_millis(1_100));

        let snapshot = counters.snapshot_at(true, started_at + Duration::from_millis(1_100));
        assert_eq!(snapshot.cumulative_tokens, 110);
        assert_eq!(snapshot.performance.completed_requests, 1);
        assert_eq!(snapshot.performance.live_decode_tokens_per_second, None);
        assert_eq!(snapshot.performance.prefill_tokens_per_second, Some(500.0));
        assert_eq!(snapshot.performance.decode_tokens_per_second, Some(10.0));
        assert_eq!(
            snapshot.performance.session_decode_tokens_per_second,
            Some(10.0)
        );
        assert_eq!(snapshot.performance.ttft_ms, Some(200.0));
    }

    #[test]
    fn live_decode_sums_active_model_requests_and_session_average_is_weighted() {
        let counters = Arc::new(ModelRuntimeUsageCounters::default());
        let started_at = Instant::now();
        let mut first = counters.start_request(10, started_at, None);
        first.record_output_tokens_at(1, started_at + Duration::from_millis(100));
        first.record_output_tokens_at(4, started_at + Duration::from_millis(500));

        let mut second = counters.start_request(10, started_at, None);
        second.record_output_tokens_at(1, started_at + Duration::from_millis(300));
        second.record_output_tokens_at(2, started_at + Duration::from_millis(500));

        let concurrent = counters.snapshot_at(true, started_at + Duration::from_millis(500));
        assert_eq!(
            concurrent.performance.live_decode_tokens_per_second,
            Some(20.0)
        );

        first.complete_at(started_at + Duration::from_millis(500));
        let one_active = counters.snapshot_at(true, started_at + Duration::from_millis(500));
        assert_eq!(
            one_active.performance.live_decode_tokens_per_second,
            Some(10.0)
        );
        assert_eq!(
            one_active.performance.session_decode_tokens_per_second,
            Some(10.0)
        );

        second.complete_at(started_at + Duration::from_millis(700));
        let completed = counters.snapshot_at(true, started_at + Duration::from_millis(700));
        assert_eq!(completed.performance.live_decode_tokens_per_second, None);
        assert_eq!(
            completed.performance.session_decode_tokens_per_second,
            Some(7.5)
        );
    }

    #[test]
    fn performance_counters_are_independent_per_loaded_model() {
        let started_at = Instant::now();
        let first_model = Arc::new(ModelRuntimeUsageCounters::default());
        let second_model = Arc::new(ModelRuntimeUsageCounters::default());

        let mut first_request = first_model.start_request(10, started_at, None);
        first_request.record_output_tokens_at(1, started_at + Duration::from_millis(100));
        first_request.record_output_tokens_at(4, started_at + Duration::from_millis(500));

        assert_eq!(
            first_model
                .snapshot_at(true, started_at + Duration::from_millis(500))
                .performance
                .live_decode_tokens_per_second,
            Some(10.0)
        );
        assert_eq!(
            second_model
                .snapshot_at(true, started_at + Duration::from_millis(500))
                .performance,
            ModelRuntimePerformanceSnapshot::default()
        );

        first_request.complete_at(started_at + Duration::from_millis(500));
        assert_eq!(
            first_model
                .snapshot_at(true, started_at + Duration::from_millis(500))
                .performance
                .session_decode_tokens_per_second,
            Some(10.0)
        );
        assert_eq!(
            second_model
                .snapshot_at(true, started_at + Duration::from_millis(500))
                .performance
                .session_decode_tokens_per_second,
            None
        );
    }

    #[test]
    fn incomplete_and_expired_requests_are_excluded() {
        let counters = Arc::new(ModelRuntimeUsageCounters::default());
        let started_at = Instant::now();
        let mut incomplete = counters.start_request(10, started_at, None);
        incomplete.record_output_tokens_at(1, started_at + Duration::from_millis(10));
        incomplete.record_output_tokens_at(1, started_at + Duration::from_millis(20));
        assert_eq!(
            counters
                .snapshot_at(true, started_at + Duration::from_millis(20))
                .performance
                .live_decode_tokens_per_second,
            Some(100.0)
        );
        drop(incomplete);
        assert_eq!(
            counters
                .snapshot_at(true, started_at + Duration::from_secs(1))
                .performance,
            ModelRuntimePerformanceSnapshot::default()
        );

        let mut completed = counters.start_request(10, started_at, None);
        completed.record_output_tokens_at(1, started_at + Duration::from_millis(10));
        completed.record_output_tokens_at(1, started_at + Duration::from_millis(20));
        completed.complete_at(started_at + Duration::from_millis(20));
        let expired = counters.snapshot_at(true, started_at + Duration::from_secs(61));
        assert_eq!(expired.performance.completed_requests, 0);
        assert_eq!(expired.performance.decode_tokens_per_second, None);
        assert_eq!(
            expired.performance.session_decode_tokens_per_second,
            Some(100.0)
        );
    }
}
