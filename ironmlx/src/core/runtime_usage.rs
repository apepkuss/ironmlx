use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, MutexGuard,
    },
    time::{Duration, Instant},
};

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
    pub prefill_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub ttft_ms: Option<f64>,
}

impl Default for ModelRuntimePerformanceSnapshot {
    fn default() -> Self {
        Self {
            window_seconds: PERFORMANCE_WINDOW.as_secs(),
            completed_requests: 0,
            prefill_tokens_per_second: None,
            decode_tokens_per_second: None,
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

#[derive(Debug, Default)]
pub struct ModelRuntimeUsageCounters {
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    prefix_cache_hit_tokens: AtomicU64,
    prefix_cache_eligible_tokens: AtomicU64,
    performance_samples: Mutex<VecDeque<ModelRuntimePerformanceSample>>,
}

#[derive(Debug)]
#[must_use = "request performance samples must be completed or intentionally dropped"]
pub struct ModelRuntimeRequestTracker {
    counters: Arc<ModelRuntimeUsageCounters>,
    input_tokens: u64,
    started_at: Instant,
    first_token_at: Option<Instant>,
    output_tokens: u64,
}

impl ModelRuntimeUsageCounters {
    pub fn record_input_tokens(&self, tokens: u64) {
        self.input_tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn record_output_tokens(&self, tokens: u64) {
        self.output_tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn start_request(
        self: &Arc<Self>,
        input_tokens: u64,
        started_at: Instant,
    ) -> ModelRuntimeRequestTracker {
        self.record_input_tokens(input_tokens);
        ModelRuntimeRequestTracker {
            counters: self.clone(),
            input_tokens,
            started_at,
            first_token_at: None,
            output_tokens: 0,
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
        ModelRuntimePerformanceSnapshot {
            window_seconds: PERFORMANCE_WINDOW.as_secs(),
            completed_requests: samples.len(),
            prefill_tokens_per_second: median(
                samples
                    .iter()
                    .map(|sample| sample.prefill_tokens_per_second),
            ),
            decode_tokens_per_second: median(
                samples
                    .iter()
                    .filter_map(|sample| sample.decode_tokens_per_second),
            ),
            ttft_ms: median(samples.iter().map(|sample| sample.ttft_ms)),
        }
    }

    fn performance_samples(&self) -> MutexGuard<'_, VecDeque<ModelRuntimePerformanceSample>> {
        self.performance_samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl ModelRuntimeRequestTracker {
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
        self.first_token_at.get_or_insert(now);
        self.output_tokens = self.output_tokens.saturating_add(tokens);
        self.counters.record_output_tokens(tokens);
    }

    fn complete_at(&mut self, completed_at: Instant) {
        let Some(first_token_at) = self.first_token_at else {
            return;
        };
        let ttft = first_token_at.saturating_duration_since(self.started_at);
        let Some(prefill_tokens_per_second) = tokens_per_second(self.input_tokens, ttft) else {
            return;
        };
        let decode_tokens = self.output_tokens.saturating_sub(1);
        let decode_duration = completed_at.saturating_duration_since(first_token_at);
        self.counters
            .record_performance_sample(ModelRuntimePerformanceSample {
                completed_at,
                prefill_tokens_per_second,
                decode_tokens_per_second: tokens_per_second(decode_tokens, decode_duration),
                ttft_ms: ttft.as_secs_f64() * 1_000.0,
            });
    }
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
        let mut request = counters.start_request(100, started_at);
        request.record_output_tokens_at(1, started_at + Duration::from_millis(200));
        request.record_output_tokens_at(9, started_at + Duration::from_millis(1_100));
        request.complete_at(started_at + Duration::from_millis(1_100));

        let snapshot = counters.snapshot_at(true, started_at + Duration::from_millis(1_100));
        assert_eq!(snapshot.cumulative_tokens, 110);
        assert_eq!(snapshot.performance.completed_requests, 1);
        assert_eq!(snapshot.performance.prefill_tokens_per_second, Some(500.0));
        assert_eq!(snapshot.performance.decode_tokens_per_second, Some(10.0));
        assert_eq!(snapshot.performance.ttft_ms, Some(200.0));
    }

    #[test]
    fn incomplete_and_expired_requests_are_excluded() {
        let counters = Arc::new(ModelRuntimeUsageCounters::default());
        let started_at = Instant::now();
        let mut incomplete = counters.start_request(10, started_at);
        incomplete.record_output_tokens_at(1, started_at + Duration::from_millis(10));
        drop(incomplete);
        assert_eq!(
            counters
                .snapshot_at(true, started_at + Duration::from_secs(1))
                .performance,
            ModelRuntimePerformanceSnapshot::default()
        );

        let mut completed = counters.start_request(10, started_at);
        completed.record_output_tokens_at(1, started_at + Duration::from_millis(10));
        completed.complete_at(started_at + Duration::from_millis(20));
        assert_eq!(
            counters
                .snapshot_at(true, started_at + Duration::from_secs(61))
                .performance
                .completed_requests,
            0
        );
    }
}
