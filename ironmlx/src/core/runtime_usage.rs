use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PrefixCacheUsageSnapshot {
    pub hit_tokens: u64,
    pub eligible_tokens: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ModelRuntimeUsageSnapshot {
    pub cumulative_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache: Option<PrefixCacheUsageSnapshot>,
}

#[derive(Debug, Default)]
pub struct ModelRuntimeUsageCounters {
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    prefix_cache_hit_tokens: AtomicU64,
    prefix_cache_eligible_tokens: AtomicU64,
}

impl ModelRuntimeUsageCounters {
    pub fn record_input_tokens(&self, tokens: u64) {
        self.input_tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn record_output_tokens(&self, tokens: u64) {
        self.output_tokens.fetch_add(tokens, Ordering::Relaxed);
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
        }
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
            }
        );
        assert_eq!(counters.snapshot(false).prefix_cache, None);
    }
}
