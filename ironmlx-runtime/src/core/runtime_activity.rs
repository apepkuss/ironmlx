//! Process-wide runtime activity history and recent-request log shown by the
//! Dashboard status page.
//!
//! The recorder lives for the whole process (it is not rebuilt when a model is
//! reloaded). The per-token path is a single atomic add; everything else takes a
//! short `std::sync::Mutex` that never awaits.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde::Serialize;
use tokio::sync::broadcast;

pub const BUCKET_MS: u64 = 5_000;
pub const TIER_30S_MS: u64 = 30_000;
pub const TIER_5M_MS: u64 = 300_000;
pub const RECENT_REQUEST_CAPACITY: usize = 100;

const SCHEMA_VERSION: u32 = 1;
const TIER_5S_CAPACITY: usize = 120;
const TIER_30S_CAPACITY: usize = 120;
const TIER_5M_CAPACITY: usize = 288;
const EVENT_CHANNEL_CAPACITY: usize = 256;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HistoryPoint {
    /// Bucket start, unix ms.
    pub t: u64,
    pub decode_tps: f64,
    pub prefill_tps: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub output_tokens: u64,
    /// Requests whose first token fell in the bucket.
    pub first_tokens: u64,
    pub requests_started: u64,
    pub requests_finished: u64,
    pub active_max: u64,
    pub queued_max: u64,
    pub process_bytes: u64,
    pub kv_cache_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub ceiling_bytes: u64,
    #[serde(skip)]
    pub prefill_input_tokens: u64,
    #[serde(skip)]
    pub ttft_sum_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestOutcome {
    Completed,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecentRequest {
    pub id: u64,
    pub finished_unix_ms: u64,
    pub model: String,
    pub endpoint: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub ttft_ms: Option<f64>,
    pub decode_tps: Option<f64>,
    pub duration_ms: u64,
    pub outcome: RequestOutcome,
}

/// Tracker-side input; the id and finish time are assigned by the recorder.
pub struct FinishedRequest<'a> {
    pub model: &'a str,
    pub endpoint: Option<&'static str>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub ttft: Option<Duration>,
    pub decode_duration: Option<Duration>,
    pub duration: Duration,
    pub outcome: RequestOutcome,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ActivitySample {
    pub active: u64,
    pub queued: u64,
    pub process_bytes: u64,
    pub kv_cache_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub ceiling_bytes: u64,
}

#[derive(Debug, Clone)]
pub enum ActivityEvent {
    History(HistoryPoint),
    Request(RecentRequest),
}

#[derive(Debug, Clone, Serialize)]
pub struct BucketSizes {
    #[serde(rename = "5s")]
    pub s5: u64,
    #[serde(rename = "30s")]
    pub s30: u64,
    #[serde(rename = "5m")]
    pub m5: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryTiers {
    #[serde(rename = "5s")]
    pub s5: Vec<HistoryPoint>,
    #[serde(rename = "30s")]
    pub s30: Vec<HistoryPoint>,
    #[serde(rename = "5m")]
    pub m5: Vec<HistoryPoint>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryResponse {
    pub schema_version: u32,
    pub bucket_ms: BucketSizes,
    pub points: HistoryTiers,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecentRequestsResponse {
    pub schema_version: u32,
    /// Newest first.
    pub requests: Vec<RecentRequest>,
}

/// One coarse tier: a bounded ring of finished points plus the finer points of
/// the window currently being filled.
struct Tier {
    window_ms: u64,
    child_ms: u64,
    points: VecDeque<HistoryPoint>,
    capacity: usize,
    pending: Vec<HistoryPoint>,
}

impl Tier {
    fn new(window_ms: u64, child_ms: u64, capacity: usize) -> Self {
        Self {
            window_ms,
            child_ms,
            points: VecDeque::with_capacity(capacity),
            capacity,
            pending: Vec::new(),
        }
    }

    fn window_start(&self, t: u64) -> u64 {
        t / self.window_ms * self.window_ms
    }

    /// Feeds one finer point. Returns the coarse points this tier emitted: the
    /// previous window (when `child` starts a new one) and/or the current window
    /// (when `child` fills its last slot). A window is never rolled up twice.
    fn push_child(&mut self, child: &HistoryPoint) -> [Option<HistoryPoint>; 2] {
        let window = self.window_start(child.t);
        let previous = match self.pending.first() {
            Some(first) if self.window_start(first.t) != window => Some(self.roll_up()),
            _ => None,
        };
        self.pending.push(child.clone());
        let current = (child.t + self.child_ms >= window + self.window_ms).then(|| self.roll_up());
        [previous, current]
    }

    fn roll_up(&mut self) -> HistoryPoint {
        let window = self.window_start(self.pending[0].t);
        let point = aggregate(window, self.window_ms, &self.pending);
        self.pending.clear();
        if self.points.len() == self.capacity {
            self.points.pop_front();
        }
        self.points.push_back(point.clone());
        point
    }
}

fn aggregate(t: u64, window_ms: u64, children: &[HistoryPoint]) -> HistoryPoint {
    let mut point = HistoryPoint {
        t,
        decode_tps: 0.0,
        prefill_tps: None,
        ttft_ms: None,
        output_tokens: 0,
        first_tokens: 0,
        requests_started: 0,
        requests_finished: 0,
        active_max: 0,
        queued_max: 0,
        process_bytes: 0,
        kv_cache_bytes: 0,
        mlx_cache_bytes: 0,
        ceiling_bytes: 0,
        prefill_input_tokens: 0,
        ttft_sum_ms: 0.0,
    };
    for child in children {
        point.output_tokens += child.output_tokens;
        point.first_tokens += child.first_tokens;
        point.requests_started += child.requests_started;
        point.requests_finished += child.requests_finished;
        point.active_max = point.active_max.max(child.active_max);
        point.queued_max = point.queued_max.max(child.queued_max);
        point.process_bytes = point.process_bytes.max(child.process_bytes);
        point.kv_cache_bytes = point.kv_cache_bytes.max(child.kv_cache_bytes);
        point.mlx_cache_bytes = point.mlx_cache_bytes.max(child.mlx_cache_bytes);
        point.ceiling_bytes = child.ceiling_bytes;
        point.prefill_input_tokens += child.prefill_input_tokens;
        point.ttft_sum_ms += child.ttft_sum_ms;
    }
    point.decode_tps = point.output_tokens as f64 / (window_ms as f64 / 1000.0);
    set_latency_fields(&mut point);
    point
}

fn set_latency_fields(point: &mut HistoryPoint) {
    if point.first_tokens == 0 {
        point.prefill_tps = None;
        point.ttft_ms = None;
        return;
    }
    point.ttft_ms = Some(point.ttft_sum_ms / point.first_tokens as f64);
    point.prefill_tps = if point.ttft_sum_ms > 0.0 {
        Some(point.prefill_input_tokens as f64 / (point.ttft_sum_ms / 1000.0))
    } else {
        None
    };
}

struct Inner {
    first_tokens: u64,
    prefill_input_tokens: u64,
    ttft_sum_ms: f64,
    requests_finished: u64,
    active_max: u64,
    queued_max: u64,
    process_bytes: u64,
    kv_cache_bytes: u64,
    mlx_cache_bytes: u64,
    s5: VecDeque<HistoryPoint>,
    s30: Tier,
    m5: Tier,
    recent: VecDeque<RecentRequest>,
    next_request_id: u64,
}

impl Inner {
    fn new() -> Self {
        Self {
            first_tokens: 0,
            prefill_input_tokens: 0,
            ttft_sum_ms: 0.0,
            requests_finished: 0,
            active_max: 0,
            queued_max: 0,
            process_bytes: 0,
            kv_cache_bytes: 0,
            mlx_cache_bytes: 0,
            s5: VecDeque::with_capacity(TIER_5S_CAPACITY),
            s30: Tier::new(TIER_30S_MS, BUCKET_MS, TIER_30S_CAPACITY),
            m5: Tier::new(TIER_5M_MS, TIER_30S_MS, TIER_5M_CAPACITY),
            recent: VecDeque::with_capacity(RECENT_REQUEST_CAPACITY),
            next_request_id: 1,
        }
    }

    fn observe(&mut self, sample: ActivitySample) {
        self.active_max = self.active_max.max(sample.active);
        self.queued_max = self.queued_max.max(sample.queued);
        self.process_bytes = self.process_bytes.max(sample.process_bytes);
        self.kv_cache_bytes = self.kv_cache_bytes.max(sample.kv_cache_bytes);
        self.mlx_cache_bytes = self.mlx_cache_bytes.max(sample.mlx_cache_bytes);
    }
}

pub struct RuntimeActivityRecorder {
    output_tokens: AtomicU64,
    requests_started: AtomicU64,
    inner: Mutex<Inner>,
    events: broadcast::Sender<ActivityEvent>,
}

impl RuntimeActivityRecorder {
    pub fn new() -> Arc<Self> {
        let (events, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Arc::new(Self {
            output_tokens: AtomicU64::new(0),
            requests_started: AtomicU64::new(0),
            inner: Mutex::new(Inner::new()),
            events,
        })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn record_output_tokens(&self, tokens: u64) {
        self.output_tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn record_request_started(&self) {
        self.requests_started.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_first_token(&self, input_tokens: u64, ttft: Duration) {
        let mut inner = self.lock();
        inner.first_tokens += 1;
        inner.prefill_input_tokens += input_tokens;
        inner.ttft_sum_ms += ttft.as_secs_f64() * 1000.0;
    }

    pub fn record_finished(&self, request: FinishedRequest<'_>, now_unix_ms: u64) -> RecentRequest {
        let decode_tps = match request.decode_duration {
            Some(duration) if request.output_tokens >= 2 && !duration.is_zero() => {
                Some((request.output_tokens - 1) as f64 / duration.as_secs_f64())
            }
            _ => None,
        };
        let mut recent = RecentRequest {
            id: 0,
            finished_unix_ms: now_unix_ms,
            model: request.model.to_string(),
            endpoint: request.endpoint.map(str::to_string),
            input_tokens: request.input_tokens,
            output_tokens: request.output_tokens,
            ttft_ms: request.ttft.map(|d| d.as_secs_f64() * 1000.0),
            decode_tps,
            duration_ms: request.duration.as_millis() as u64,
            outcome: request.outcome,
        };
        let mut inner = self.lock();
        recent.id = inner.next_request_id;
        inner.next_request_id += 1;
        inner.requests_finished += 1;
        if inner.recent.len() == RECENT_REQUEST_CAPACITY {
            inner.recent.pop_front();
        }
        inner.recent.push_back(recent.clone());
        let _ = self.events.send(ActivityEvent::Request(recent.clone()));
        recent
    }

    pub fn observe(&self, sample: ActivitySample) {
        self.lock().observe(sample);
    }

    /// Closes the bucket starting at `bucket_start_unix_ms`. If the wall clock
    /// stepped backwards (the start is not after the last closed bucket), nothing
    /// is reset, stored or broadcast: the returned point is only a view of the
    /// current counters, which roll into the next valid bucket.
    ///
    /// The counters cover everything since the previous stored point. When the
    /// clock fired late and that is more than one bucket, `decode_tps` divides by
    /// that whole span, so the late point does not show a spike.
    pub fn close_bucket(&self, bucket_start_unix_ms: u64, sample: ActivitySample) -> HistoryPoint {
        let mut inner = self.lock();
        inner.observe(sample);
        let previous_start = inner.s5.back().map(|last| last.t);
        let stale = previous_start.is_some_and(|last| bucket_start_unix_ms <= last);
        let span_ms = previous_start
            .map_or(BUCKET_MS, |last| bucket_start_unix_ms.saturating_sub(last))
            .max(BUCKET_MS);
        let (output_tokens, requests_started) = if stale {
            (
                self.output_tokens.load(Ordering::Relaxed),
                self.requests_started.load(Ordering::Relaxed),
            )
        } else {
            (
                self.output_tokens.swap(0, Ordering::Relaxed),
                self.requests_started.swap(0, Ordering::Relaxed),
            )
        };
        let mut point = HistoryPoint {
            t: bucket_start_unix_ms,
            decode_tps: output_tokens as f64 / (span_ms as f64 / 1000.0),
            prefill_tps: None,
            ttft_ms: None,
            output_tokens,
            first_tokens: inner.first_tokens,
            requests_started,
            requests_finished: inner.requests_finished,
            active_max: inner.active_max,
            queued_max: inner.queued_max,
            process_bytes: inner.process_bytes,
            kv_cache_bytes: inner.kv_cache_bytes,
            mlx_cache_bytes: inner.mlx_cache_bytes,
            ceiling_bytes: sample.ceiling_bytes,
            prefill_input_tokens: inner.prefill_input_tokens,
            ttft_sum_ms: inner.ttft_sum_ms,
        };
        set_latency_fields(&mut point);
        if stale {
            return point;
        }
        inner.first_tokens = 0;
        inner.prefill_input_tokens = 0;
        inner.ttft_sum_ms = 0.0;
        inner.requests_finished = 0;
        inner.active_max = 0;
        inner.queued_max = 0;
        inner.process_bytes = 0;
        inner.kv_cache_bytes = 0;
        inner.mlx_cache_bytes = 0;

        if inner.s5.len() == TIER_5S_CAPACITY {
            inner.s5.pop_front();
        }
        inner.s5.push_back(point.clone());
        for coarse in inner.s30.push_child(&point).into_iter().flatten() {
            inner.m5.push_child(&coarse);
        }
        let _ = self.events.send(ActivityEvent::History(point.clone()));
        point
    }

    pub fn history(&self) -> HistoryResponse {
        let inner = self.lock();
        HistoryResponse {
            schema_version: SCHEMA_VERSION,
            bucket_ms: BucketSizes {
                s5: BUCKET_MS,
                s30: TIER_30S_MS,
                m5: TIER_5M_MS,
            },
            points: HistoryTiers {
                s5: inner.s5.iter().cloned().collect(),
                s30: inner.s30.points.iter().cloned().collect(),
                m5: inner.m5.points.iter().cloned().collect(),
            },
        }
    }

    pub fn recent(&self) -> RecentRequestsResponse {
        let inner = self.lock();
        RecentRequestsResponse {
            schema_version: SCHEMA_VERSION,
            requests: inner.recent.iter().rev().cloned().collect(),
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ActivityEvent> {
        self.events.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn finished(output_tokens: u64, decode: Option<Duration>) -> FinishedRequest<'static> {
        FinishedRequest {
            model: "m",
            endpoint: Some("/v1/chat/completions"),
            input_tokens: 10,
            output_tokens,
            ttft: Some(Duration::from_millis(250)),
            decode_duration: decode,
            duration: Duration::from_secs(3),
            outcome: RequestOutcome::Completed,
        }
    }

    fn close_with_tokens(recorder: &RuntimeActivityRecorder, t: u64, tokens: u64) {
        recorder.record_output_tokens(tokens);
        recorder.close_bucket(t, ActivitySample::default());
    }

    #[test]
    fn close_bucket_computes_rates() {
        let recorder = RuntimeActivityRecorder::new();
        recorder.record_output_tokens(500);
        recorder.record_first_token(1000, Duration::from_millis(500));
        recorder.record_first_token(3000, Duration::from_millis(1500));
        let point = recorder.close_bucket(0, ActivitySample::default());
        assert_eq!(point.decode_tps, 100.0);
        assert_eq!(point.prefill_tps, Some(2000.0));
        assert_eq!(point.ttft_ms, Some(1000.0));
        assert_eq!(point.first_tokens, 2);

        let empty = recorder.close_bucket(5000, ActivitySample::default());
        assert_eq!(empty.decode_tps, 0.0);
        assert_eq!(empty.prefill_tps, None);
        assert_eq!(empty.ttft_ms, None);
        assert_eq!(empty.output_tokens, 0);
        assert_eq!(empty.first_tokens, 0);
        assert_eq!(empty.requests_started, 0);
        assert_eq!(empty.requests_finished, 0);
    }

    #[test]
    fn observe_keeps_bucket_maxima() {
        let recorder = RuntimeActivityRecorder::new();
        recorder.observe(ActivitySample {
            active: 3,
            process_bytes: 10 * GIB,
            ..ActivitySample::default()
        });
        let point = recorder.close_bucket(
            0,
            ActivitySample {
                active: 1,
                process_bytes: 8 * GIB,
                ceiling_bytes: 100 * GIB,
                ..ActivitySample::default()
            },
        );
        assert_eq!(point.active_max, 3);
        assert_eq!(point.process_bytes, 10 * GIB);
        assert_eq!(point.ceiling_bytes, 100 * GIB);
        let next = recorder.close_bucket(5000, ActivitySample::default());
        assert_eq!(next.active_max, 0);
        assert_eq!(next.process_bytes, 0);
    }

    #[test]
    fn thirty_second_rollup_from_six_points() {
        let recorder = RuntimeActivityRecorder::new();
        for i in 0..6 {
            close_with_tokens(&recorder, i * 5000, 50);
        }
        let history = recorder.history();
        assert_eq!(history.points.s30.len(), 1);
        assert_eq!(history.points.s30[0].t, 0);
        assert_eq!(history.points.s30[0].decode_tps, 10.0);
        // Entering the next window must not roll the same window up again.
        close_with_tokens(&recorder, 30_000, 50);
        assert_eq!(recorder.history().points.s30.len(), 1);
    }

    #[test]
    fn rollup_with_missing_slots_divides_by_window() {
        let recorder = RuntimeActivityRecorder::new();
        close_with_tokens(&recorder, 0, 300);
        close_with_tokens(&recorder, 5000, 300);
        assert!(recorder.history().points.s30.is_empty());
        close_with_tokens(&recorder, 30_000, 0);
        let s30 = recorder.history().points.s30;
        assert_eq!(s30.len(), 1);
        assert_eq!(s30[0].t, 0);
        assert_eq!(s30[0].decode_tps, 20.0);
    }

    #[test]
    fn five_minute_tier_rolls_up_from_thirty_second_points() {
        let recorder = RuntimeActivityRecorder::new();
        recorder.record_first_token(600, Duration::from_millis(300));
        for i in 0..60 {
            close_with_tokens(&recorder, i * 5000, 30);
        }
        let m5 = recorder.history().points.m5;
        assert_eq!(m5.len(), 1);
        assert_eq!(m5[0].t, 0);
        assert_eq!(m5[0].output_tokens, 1800);
        assert_eq!(m5[0].decode_tps, 6.0);
        assert_eq!(m5[0].ttft_ms, Some(300.0));
        assert_eq!(m5[0].prefill_tps, Some(2000.0));
    }

    #[test]
    fn late_tick_closes_one_bucket_and_leaves_gap() {
        let recorder = RuntimeActivityRecorder::new();
        close_with_tokens(&recorder, 0, 100);
        close_with_tokens(&recorder, 600_000, 200);
        let s5 = recorder.history().points.s5;
        assert_eq!(s5.len(), 2);
        assert_eq!((s5[0].t, s5[0].output_tokens), (0, 100));
        assert_eq!((s5[1].t, s5[1].output_tokens), (600_000, 200));
    }

    #[test]
    fn late_close_spreads_decode_rate_over_elapsed_span() {
        let recorder = RuntimeActivityRecorder::new();
        recorder.record_output_tokens(100);
        let first = recorder.close_bucket(0, ActivitySample::default());
        assert_eq!(first.decode_tps, 20.0);
        // The clock fired late: this close covers 20 s since the previous point.
        recorder.record_output_tokens(100);
        let late = recorder.close_bucket(20_000, ActivitySample::default());
        assert_eq!(late.decode_tps, 5.0);
        assert_eq!(late.output_tokens, 100);
        // Back on schedule: one bucket again; this point also fills the 30 s window.
        recorder.record_output_tokens(100);
        let next = recorder.close_bucket(25_000, ActivitySample::default());
        assert_eq!(next.decode_tps, 20.0);
        // The roll-up still divides the window's token total by the window.
        let s30 = recorder.history().points.s30;
        assert_eq!(s30.len(), 1);
        assert_eq!(s30[0].output_tokens, 300);
        assert_eq!(s30[0].decode_tps, 10.0);
    }

    #[test]
    fn tiers_are_bounded() {
        let recorder = RuntimeActivityRecorder::new();
        for i in 0..200 {
            close_with_tokens(&recorder, i * 5000, 1);
        }
        let s5 = recorder.history().points.s5;
        assert_eq!(s5.len(), 120);
        assert_eq!(s5[0].t, 80 * 5000);
        assert_eq!(s5[119].t, 199 * 5000);
    }

    #[test]
    fn non_increasing_bucket_start_is_ignored() {
        let recorder = RuntimeActivityRecorder::new();
        let mut rx = recorder.subscribe();
        close_with_tokens(&recorder, 10_000, 50);
        close_with_tokens(&recorder, 5_000, 30);
        assert_eq!(recorder.history().points.s5.len(), 1);
        assert!(matches!(rx.try_recv(), Ok(ActivityEvent::History(_))));
        assert!(rx.try_recv().is_err());
        let next = recorder.close_bucket(15_000, ActivitySample::default());
        assert_eq!(next.output_tokens, 30);
    }

    #[test]
    fn coarse_tier_is_bounded_and_aggregates_maxima() {
        let recorder = RuntimeActivityRecorder::new();
        for i in 0..(121 * 6) {
            recorder.close_bucket(
                i * 5000,
                ActivitySample {
                    active: i % 6,
                    ceiling_bytes: i,
                    ..ActivitySample::default()
                },
            );
        }
        let s30 = recorder.history().points.s30;
        assert_eq!(s30.len(), 120);
        assert_eq!(s30[0].t, TIER_30S_MS);
        let last = s30.last().unwrap();
        assert_eq!(last.active_max, 5);
        assert_eq!(last.ceiling_bytes, 121 * 6 - 1);
    }

    #[test]
    fn recent_requests_are_bounded_and_newest_first() {
        let recorder = RuntimeActivityRecorder::new();
        for i in 0..105 {
            recorder.record_finished(finished(10, None), 1000 + i);
        }
        let requests = recorder.recent().requests;
        assert_eq!(requests.len(), 100);
        assert_eq!(requests[0].id, 105);
        assert_eq!(requests[99].id, 6);
    }

    #[test]
    fn finished_request_rates() {
        let recorder = RuntimeActivityRecorder::new();
        let rate = recorder.record_finished(finished(101, Some(Duration::from_secs(2))), 1);
        assert_eq!(rate.decode_tps, Some(50.0));
        assert_eq!(rate.ttft_ms, Some(250.0));
        assert_eq!(rate.duration_ms, 3000);
        let single = recorder.record_finished(finished(1, Some(Duration::from_secs(2))), 2);
        assert_eq!(single.decode_tps, None);
        let zero = recorder.record_finished(finished(50, Some(Duration::ZERO)), 3);
        assert_eq!(zero.decode_tps, None);
        let mut aborted = finished(5, None);
        aborted.outcome = RequestOutcome::Aborted;
        let aborted = recorder.record_finished(aborted, 4);
        assert_eq!(aborted.outcome, RequestOutcome::Aborted);
        assert_eq!(aborted.finished_unix_ms, 4);
    }

    #[test]
    fn history_json_shape() {
        let recorder = RuntimeActivityRecorder::new();
        recorder.record_first_token(10, Duration::from_millis(5));
        recorder.close_bucket(0, ActivitySample::default());
        let value = serde_json::to_value(recorder.history()).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["bucket_ms"]["5s"], 5000);
        assert_eq!(value["bucket_ms"]["30s"], 30000);
        assert_eq!(value["bucket_ms"]["5m"], 300000);
        for tier in ["5s", "30s", "5m"] {
            assert!(value["points"][tier].is_array(), "{tier}");
        }
        let point = &value["points"]["5s"][0];
        assert!(point.get("prefill_input_tokens").is_none());
        assert!(point.get("ttft_sum_ms").is_none());
        assert!(point.get("decode_tps").is_some());
        let recent = serde_json::to_value(recorder.recent()).unwrap();
        assert_eq!(recent["schema_version"], 1);
    }

    #[test]
    fn events_are_broadcast() {
        let recorder = RuntimeActivityRecorder::new();
        let mut rx = recorder.subscribe();
        recorder.close_bucket(0, ActivitySample::default());
        recorder.record_finished(finished(10, None), 7);
        assert!(matches!(rx.try_recv(), Ok(ActivityEvent::History(p)) if p.t == 0));
        assert!(matches!(rx.try_recv(), Ok(ActivityEvent::Request(r)) if r.id == 1));
    }

    #[test]
    fn sending_without_subscribers_is_ignored() {
        let recorder = RuntimeActivityRecorder::new();
        recorder.close_bucket(0, ActivitySample::default());
        recorder.record_finished(finished(10, None), 1);
    }
}
