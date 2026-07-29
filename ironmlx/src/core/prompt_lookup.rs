use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context};
use serde::{Deserialize, Serialize};

use crate::core::sampler::Sampler;
use crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile;
use crate::core::speculative::MtpDraftPolicySnapshot;
use crate::Result;

const POSITIONS_PER_NGRAM: usize = 2;
const QUALIFICATION_SCHEMA_VERSION: u32 = 6;
const QUALIFICATION_BASELINE_SAMPLES: usize = 8;
const QUALIFICATION_PROBE_SAMPLES: usize = 8;
const QUALIFICATION_MIN_GAIN_BPS: u64 = 300;
const QUALIFICATION_MAX_REPROBE_OVERHEAD_BPS: u64 = 300;
const QUALIFICATION_MULTI_BATCH_INITIAL_DELAY_TOKENS: u64 = 512;
const QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS: u64 = 512;
const QUALIFICATION_REJECTED_MAX_COOLDOWN_TOKENS: u64 = 32 * 1_024;
const QUALIFICATION_REVALIDATE_TOKENS: u64 = 512;
const QUALIFICATION_PROFILE_TTL_MS: u64 = 7 * 24 * 60 * 60 * 1_000;
pub(crate) const SHARED_PROMPT_LOOKUP_TTL_MS: u64 = 60 * 60 * 1_000;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptLookupSourceStats {
    pub queries: u64,
    pub hits: u64,
    pub misses: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub zero_accept_windows: u64,
    pub wasted_verify_tokens: u64,
    pub propose_us: u64,
    pub verify_us: u64,
    pub rollback_us: u64,
}

impl PromptLookupSourceStats {
    fn saturating_delta_since(self, before: Self) -> Self {
        Self {
            queries: self.queries.saturating_sub(before.queries),
            hits: self.hits.saturating_sub(before.hits),
            misses: self.misses.saturating_sub(before.misses),
            drafted_tokens: self.drafted_tokens.saturating_sub(before.drafted_tokens),
            accepted_tokens: self.accepted_tokens.saturating_sub(before.accepted_tokens),
            zero_accept_windows: self
                .zero_accept_windows
                .saturating_sub(before.zero_accept_windows),
            wasted_verify_tokens: self
                .wasted_verify_tokens
                .saturating_sub(before.wasted_verify_tokens),
            propose_us: self.propose_us.saturating_sub(before.propose_us),
            verify_us: self.verify_us.saturating_sub(before.verify_us),
            rollback_us: self.rollback_us.saturating_sub(before.rollback_us),
        }
    }

    fn accumulate_delta(&mut self, delta: Self) {
        self.queries = self.queries.saturating_add(delta.queries);
        self.hits = self.hits.saturating_add(delta.hits);
        self.misses = self.misses.saturating_add(delta.misses);
        self.drafted_tokens = self.drafted_tokens.saturating_add(delta.drafted_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(delta.accepted_tokens);
        self.zero_accept_windows = self
            .zero_accept_windows
            .saturating_add(delta.zero_accept_windows);
        self.wasted_verify_tokens = self
            .wasted_verify_tokens
            .saturating_add(delta.wasted_verify_tokens);
        self.propose_us = self.propose_us.saturating_add(delta.propose_us);
        self.verify_us = self.verify_us.saturating_add(delta.verify_us);
        self.rollback_us = self.rollback_us.saturating_add(delta.rollback_us);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptLookupProposalSource {
    Local,
    Shared,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptLookupConfig {
    pub min_ngram: usize,
    pub max_ngram: usize,
    pub max_draft_tokens: usize,
    pub history_window_tokens: usize,
    pub max_index_entries: usize,
    /// Share immutable histories from normally completed requests within one
    /// model-engine trust domain.
    pub cross_request: bool,
}

impl PromptLookupConfig {
    pub const DEFAULT_MIN_NGRAM: usize = 2;
    pub const DEFAULT_MAX_NGRAM: usize = 4;
    pub const DEFAULT_MAX_DRAFT_TOKENS: usize = 4;
    pub const DEFAULT_HISTORY_WINDOW_TOKENS: usize = 32 * 1024;
    pub const DEFAULT_MAX_INDEX_ENTRIES: usize = 64 * 1024;

    pub fn validate(self) -> Result<Self> {
        if self.min_ngram == 0 {
            bail!("prompt lookup min_ngram must be >= 1");
        }
        if self.max_ngram < self.min_ngram {
            bail!(
                "prompt lookup max_ngram {} must be >= min_ngram {}",
                self.max_ngram,
                self.min_ngram
            );
        }
        if self.max_draft_tokens == 0 {
            bail!("prompt lookup max_draft_tokens must be >= 1");
        }
        if self.history_window_tokens <= self.max_ngram {
            bail!(
                "prompt lookup history_window_tokens {} must exceed max_ngram {}",
                self.history_window_tokens,
                self.max_ngram
            );
        }
        if self.max_index_entries == 0 {
            bail!("prompt lookup max_index_entries must be >= 1");
        }
        Ok(self)
    }
}

impl Default for PromptLookupConfig {
    fn default() -> Self {
        Self {
            min_ngram: Self::DEFAULT_MIN_NGRAM,
            max_ngram: Self::DEFAULT_MAX_NGRAM,
            max_draft_tokens: Self::DEFAULT_MAX_DRAFT_TOKENS,
            history_window_tokens: Self::DEFAULT_HISTORY_WINDOW_TOKENS,
            max_index_entries: Self::DEFAULT_MAX_INDEX_ENTRIES,
            cross_request: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptLookupStats {
    pub queries: u64,
    pub hits: u64,
    pub misses: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub rejected_tokens: u64,
    pub zero_accept_windows: u64,
    pub exact_sampling_windows: u64,
    pub exact_acceptance_draws: u64,
    pub exact_residual_corrections: u64,
    pub exact_bonus_samples: u64,
    pub propose_us: u64,
    pub index_build_us: u64,
    pub index_update_us: u64,
    pub index_entries_current: u64,
    pub index_entries_peak: u64,
    pub index_evictions: u64,
    /// End-to-end verify window time through the acceptance materialization
    /// boundary. Unlike the submit-only phase timers below, this includes
    /// pending lazy MLX work forced by the packed acceptance readback.
    pub verify_round_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub exact_batched_verify_windows: u64,
    pub sequential_verify_windows: u64,
    pub verify_accept_host_sync_count: u64,
    pub verify_accept_host_sync_us: u64,
    pub rollback_count: u64,
    pub rollback_us: u64,
    pub mtp_shadow_commit_windows: u64,
    pub mtp_shadow_commit_tokens: u64,
    pub mtp_shadow_commit_us: u64,
    pub miss_fast_path_steps: u64,
    pub ordinary_cost_samples: u64,
    pub lookup_cost_samples: u64,
    pub ordinary_cost_us: u64,
    pub lookup_cost_us: u64,
    pub qualified_regimes_current: u64,
    pub rejected_regimes_current: u64,
    pub qualification_changes: u64,
    pub qualification_profile_loads: u64,
    pub qualification_profile_writes: u64,
    pub qualification_profile_write_drops: u64,
    pub qualification_query_gate_skips: u64,
    pub miss_query_gate_skips: u64,
    pub miss_query_reprobes: u64,
    pub hybrid_neural_windows: u64,
    pub hybrid_lookup_windows: u64,
    pub hybrid_source_switches: u64,
    pub hybrid_lookup_miss_fallbacks: u64,
    pub hybrid_neural_rebases: u64,
    pub hybrid_neural_rebase_us: u64,
    pub local_source: PromptLookupSourceStats,
    pub shared_source: PromptLookupSourceStats,
    pub shared_queries: u64,
    pub shared_hits: u64,
    pub shared_misses: u64,
    pub shared_mtp_certified_published_windows: u64,
    pub shared_mtp_certified_published_tokens: u64,
    pub shared_mtp_certified_hits: u64,
    pub shared_mtp_canonical_validation_windows: u64,
    pub shared_mtp_canonical_validation_tokens: u64,
    pub shared_mtp_canonical_validation_us: u64,
    pub shared_mtp_canonical_validation_mismatches: u64,
    pub shared_mtp_canonical_fallbacks: u64,
    pub shared_published_requests: u64,
    pub shared_published_tokens: u64,
    pub shared_entries_current: u64,
    pub shared_entries_peak: u64,
    pub shared_evictions: u64,
    pub shared_pressure_evictions: u64,
    pub shared_clear_count: u64,
    pub shared_cleared_entries: u64,
    pub shared_estimated_bytes_current: u64,
    pub shared_estimated_bytes_peak: u64,
}

impl PromptLookupStats {
    pub fn saturating_delta_since(self, before: Self) -> Self {
        Self {
            queries: self.queries.saturating_sub(before.queries),
            hits: self.hits.saturating_sub(before.hits),
            misses: self.misses.saturating_sub(before.misses),
            drafted_tokens: self.drafted_tokens.saturating_sub(before.drafted_tokens),
            accepted_tokens: self.accepted_tokens.saturating_sub(before.accepted_tokens),
            rejected_tokens: self.rejected_tokens.saturating_sub(before.rejected_tokens),
            zero_accept_windows: self
                .zero_accept_windows
                .saturating_sub(before.zero_accept_windows),
            exact_sampling_windows: self
                .exact_sampling_windows
                .saturating_sub(before.exact_sampling_windows),
            exact_acceptance_draws: self
                .exact_acceptance_draws
                .saturating_sub(before.exact_acceptance_draws),
            exact_residual_corrections: self
                .exact_residual_corrections
                .saturating_sub(before.exact_residual_corrections),
            exact_bonus_samples: self
                .exact_bonus_samples
                .saturating_sub(before.exact_bonus_samples),
            propose_us: self.propose_us.saturating_sub(before.propose_us),
            index_build_us: self.index_build_us.saturating_sub(before.index_build_us),
            index_update_us: self.index_update_us.saturating_sub(before.index_update_us),
            index_entries_current: self.index_entries_current,
            index_entries_peak: self.index_entries_peak.max(before.index_entries_peak),
            index_evictions: self.index_evictions.saturating_sub(before.index_evictions),
            verify_round_us: self.verify_round_us.saturating_sub(before.verify_round_us),
            verify_forward_us: self
                .verify_forward_us
                .saturating_sub(before.verify_forward_us),
            projection_us: self.projection_us.saturating_sub(before.projection_us),
            exact_batched_verify_windows: self
                .exact_batched_verify_windows
                .saturating_sub(before.exact_batched_verify_windows),
            sequential_verify_windows: self
                .sequential_verify_windows
                .saturating_sub(before.sequential_verify_windows),
            verify_accept_host_sync_count: self
                .verify_accept_host_sync_count
                .saturating_sub(before.verify_accept_host_sync_count),
            verify_accept_host_sync_us: self
                .verify_accept_host_sync_us
                .saturating_sub(before.verify_accept_host_sync_us),
            rollback_count: self.rollback_count.saturating_sub(before.rollback_count),
            rollback_us: self.rollback_us.saturating_sub(before.rollback_us),
            mtp_shadow_commit_windows: self
                .mtp_shadow_commit_windows
                .saturating_sub(before.mtp_shadow_commit_windows),
            mtp_shadow_commit_tokens: self
                .mtp_shadow_commit_tokens
                .saturating_sub(before.mtp_shadow_commit_tokens),
            mtp_shadow_commit_us: self
                .mtp_shadow_commit_us
                .saturating_sub(before.mtp_shadow_commit_us),
            miss_fast_path_steps: self
                .miss_fast_path_steps
                .saturating_sub(before.miss_fast_path_steps),
            ordinary_cost_samples: self
                .ordinary_cost_samples
                .saturating_sub(before.ordinary_cost_samples),
            lookup_cost_samples: self
                .lookup_cost_samples
                .saturating_sub(before.lookup_cost_samples),
            ordinary_cost_us: self
                .ordinary_cost_us
                .saturating_sub(before.ordinary_cost_us),
            lookup_cost_us: self.lookup_cost_us.saturating_sub(before.lookup_cost_us),
            qualified_regimes_current: self.qualified_regimes_current,
            rejected_regimes_current: self.rejected_regimes_current,
            qualification_changes: self
                .qualification_changes
                .saturating_sub(before.qualification_changes),
            qualification_profile_loads: self
                .qualification_profile_loads
                .saturating_sub(before.qualification_profile_loads),
            qualification_profile_writes: self
                .qualification_profile_writes
                .saturating_sub(before.qualification_profile_writes),
            qualification_profile_write_drops: self
                .qualification_profile_write_drops
                .saturating_sub(before.qualification_profile_write_drops),
            qualification_query_gate_skips: self
                .qualification_query_gate_skips
                .saturating_sub(before.qualification_query_gate_skips),
            miss_query_gate_skips: self
                .miss_query_gate_skips
                .saturating_sub(before.miss_query_gate_skips),
            miss_query_reprobes: self
                .miss_query_reprobes
                .saturating_sub(before.miss_query_reprobes),
            hybrid_neural_windows: self
                .hybrid_neural_windows
                .saturating_sub(before.hybrid_neural_windows),
            hybrid_lookup_windows: self
                .hybrid_lookup_windows
                .saturating_sub(before.hybrid_lookup_windows),
            hybrid_source_switches: self
                .hybrid_source_switches
                .saturating_sub(before.hybrid_source_switches),
            hybrid_lookup_miss_fallbacks: self
                .hybrid_lookup_miss_fallbacks
                .saturating_sub(before.hybrid_lookup_miss_fallbacks),
            hybrid_neural_rebases: self
                .hybrid_neural_rebases
                .saturating_sub(before.hybrid_neural_rebases),
            hybrid_neural_rebase_us: self
                .hybrid_neural_rebase_us
                .saturating_sub(before.hybrid_neural_rebase_us),
            local_source: self
                .local_source
                .saturating_delta_since(before.local_source),
            shared_source: self
                .shared_source
                .saturating_delta_since(before.shared_source),
            shared_queries: self.shared_queries.saturating_sub(before.shared_queries),
            shared_hits: self.shared_hits.saturating_sub(before.shared_hits),
            shared_misses: self.shared_misses.saturating_sub(before.shared_misses),
            shared_mtp_certified_published_windows: self
                .shared_mtp_certified_published_windows
                .saturating_sub(before.shared_mtp_certified_published_windows),
            shared_mtp_certified_published_tokens: self
                .shared_mtp_certified_published_tokens
                .saturating_sub(before.shared_mtp_certified_published_tokens),
            shared_mtp_certified_hits: self
                .shared_mtp_certified_hits
                .saturating_sub(before.shared_mtp_certified_hits),
            shared_mtp_canonical_validation_windows: self
                .shared_mtp_canonical_validation_windows
                .saturating_sub(before.shared_mtp_canonical_validation_windows),
            shared_mtp_canonical_validation_tokens: self
                .shared_mtp_canonical_validation_tokens
                .saturating_sub(before.shared_mtp_canonical_validation_tokens),
            shared_mtp_canonical_validation_us: self
                .shared_mtp_canonical_validation_us
                .saturating_sub(before.shared_mtp_canonical_validation_us),
            shared_mtp_canonical_validation_mismatches: self
                .shared_mtp_canonical_validation_mismatches
                .saturating_sub(before.shared_mtp_canonical_validation_mismatches),
            shared_mtp_canonical_fallbacks: self
                .shared_mtp_canonical_fallbacks
                .saturating_sub(before.shared_mtp_canonical_fallbacks),
            shared_published_requests: self
                .shared_published_requests
                .saturating_sub(before.shared_published_requests),
            shared_published_tokens: self
                .shared_published_tokens
                .saturating_sub(before.shared_published_tokens),
            shared_entries_current: self.shared_entries_current,
            shared_entries_peak: self.shared_entries_peak.max(before.shared_entries_peak),
            shared_evictions: self
                .shared_evictions
                .saturating_sub(before.shared_evictions),
            shared_pressure_evictions: self
                .shared_pressure_evictions
                .saturating_sub(before.shared_pressure_evictions),
            shared_clear_count: self
                .shared_clear_count
                .saturating_sub(before.shared_clear_count),
            shared_cleared_entries: self
                .shared_cleared_entries
                .saturating_sub(before.shared_cleared_entries),
            shared_estimated_bytes_current: self.shared_estimated_bytes_current,
            shared_estimated_bytes_peak: self
                .shared_estimated_bytes_peak
                .max(before.shared_estimated_bytes_peak),
        }
    }

    pub(crate) fn accumulate_delta(&mut self, delta: Self) {
        self.queries = self.queries.saturating_add(delta.queries);
        self.hits = self.hits.saturating_add(delta.hits);
        self.misses = self.misses.saturating_add(delta.misses);
        self.drafted_tokens = self.drafted_tokens.saturating_add(delta.drafted_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(delta.accepted_tokens);
        self.rejected_tokens = self.rejected_tokens.saturating_add(delta.rejected_tokens);
        self.zero_accept_windows = self
            .zero_accept_windows
            .saturating_add(delta.zero_accept_windows);
        self.exact_sampling_windows = self
            .exact_sampling_windows
            .saturating_add(delta.exact_sampling_windows);
        self.exact_acceptance_draws = self
            .exact_acceptance_draws
            .saturating_add(delta.exact_acceptance_draws);
        self.exact_residual_corrections = self
            .exact_residual_corrections
            .saturating_add(delta.exact_residual_corrections);
        self.exact_bonus_samples = self
            .exact_bonus_samples
            .saturating_add(delta.exact_bonus_samples);
        self.propose_us = self.propose_us.saturating_add(delta.propose_us);
        self.index_build_us = self.index_build_us.saturating_add(delta.index_build_us);
        self.index_update_us = self.index_update_us.saturating_add(delta.index_update_us);
        self.index_entries_current = delta.index_entries_current;
        self.index_entries_peak = self.index_entries_peak.max(delta.index_entries_peak);
        self.index_evictions = self.index_evictions.saturating_add(delta.index_evictions);
        self.verify_round_us = self.verify_round_us.saturating_add(delta.verify_round_us);
        self.verify_forward_us = self
            .verify_forward_us
            .saturating_add(delta.verify_forward_us);
        self.projection_us = self.projection_us.saturating_add(delta.projection_us);
        self.exact_batched_verify_windows = self
            .exact_batched_verify_windows
            .saturating_add(delta.exact_batched_verify_windows);
        self.sequential_verify_windows = self
            .sequential_verify_windows
            .saturating_add(delta.sequential_verify_windows);
        self.verify_accept_host_sync_count = self
            .verify_accept_host_sync_count
            .saturating_add(delta.verify_accept_host_sync_count);
        self.verify_accept_host_sync_us = self
            .verify_accept_host_sync_us
            .saturating_add(delta.verify_accept_host_sync_us);
        self.rollback_count = self.rollback_count.saturating_add(delta.rollback_count);
        self.rollback_us = self.rollback_us.saturating_add(delta.rollback_us);
        self.mtp_shadow_commit_windows = self
            .mtp_shadow_commit_windows
            .saturating_add(delta.mtp_shadow_commit_windows);
        self.mtp_shadow_commit_tokens = self
            .mtp_shadow_commit_tokens
            .saturating_add(delta.mtp_shadow_commit_tokens);
        self.mtp_shadow_commit_us = self
            .mtp_shadow_commit_us
            .saturating_add(delta.mtp_shadow_commit_us);
        self.miss_fast_path_steps = self
            .miss_fast_path_steps
            .saturating_add(delta.miss_fast_path_steps);
        self.ordinary_cost_samples = self
            .ordinary_cost_samples
            .saturating_add(delta.ordinary_cost_samples);
        self.lookup_cost_samples = self
            .lookup_cost_samples
            .saturating_add(delta.lookup_cost_samples);
        self.ordinary_cost_us = self.ordinary_cost_us.saturating_add(delta.ordinary_cost_us);
        self.lookup_cost_us = self.lookup_cost_us.saturating_add(delta.lookup_cost_us);
        self.qualification_changes = self
            .qualification_changes
            .saturating_add(delta.qualification_changes);
        self.qualification_profile_loads = self
            .qualification_profile_loads
            .saturating_add(delta.qualification_profile_loads);
        self.qualification_profile_writes = self
            .qualification_profile_writes
            .saturating_add(delta.qualification_profile_writes);
        self.qualification_profile_write_drops = self
            .qualification_profile_write_drops
            .saturating_add(delta.qualification_profile_write_drops);
        self.qualification_query_gate_skips = self
            .qualification_query_gate_skips
            .saturating_add(delta.qualification_query_gate_skips);
        self.miss_query_gate_skips = self
            .miss_query_gate_skips
            .saturating_add(delta.miss_query_gate_skips);
        self.miss_query_reprobes = self
            .miss_query_reprobes
            .saturating_add(delta.miss_query_reprobes);
        self.hybrid_neural_windows = self
            .hybrid_neural_windows
            .saturating_add(delta.hybrid_neural_windows);
        self.hybrid_lookup_windows = self
            .hybrid_lookup_windows
            .saturating_add(delta.hybrid_lookup_windows);
        self.hybrid_source_switches = self
            .hybrid_source_switches
            .saturating_add(delta.hybrid_source_switches);
        self.hybrid_lookup_miss_fallbacks = self
            .hybrid_lookup_miss_fallbacks
            .saturating_add(delta.hybrid_lookup_miss_fallbacks);
        self.hybrid_neural_rebases = self
            .hybrid_neural_rebases
            .saturating_add(delta.hybrid_neural_rebases);
        self.hybrid_neural_rebase_us = self
            .hybrid_neural_rebase_us
            .saturating_add(delta.hybrid_neural_rebase_us);
        self.local_source.accumulate_delta(delta.local_source);
        self.shared_source.accumulate_delta(delta.shared_source);
        self.shared_queries = self.shared_queries.saturating_add(delta.shared_queries);
        self.shared_hits = self.shared_hits.saturating_add(delta.shared_hits);
        self.shared_misses = self.shared_misses.saturating_add(delta.shared_misses);
        self.shared_mtp_certified_published_windows = self
            .shared_mtp_certified_published_windows
            .saturating_add(delta.shared_mtp_certified_published_windows);
        self.shared_mtp_certified_published_tokens = self
            .shared_mtp_certified_published_tokens
            .saturating_add(delta.shared_mtp_certified_published_tokens);
        self.shared_mtp_certified_hits = self
            .shared_mtp_certified_hits
            .saturating_add(delta.shared_mtp_certified_hits);
        self.shared_mtp_canonical_validation_windows = self
            .shared_mtp_canonical_validation_windows
            .saturating_add(delta.shared_mtp_canonical_validation_windows);
        self.shared_mtp_canonical_validation_tokens = self
            .shared_mtp_canonical_validation_tokens
            .saturating_add(delta.shared_mtp_canonical_validation_tokens);
        self.shared_mtp_canonical_validation_us = self
            .shared_mtp_canonical_validation_us
            .saturating_add(delta.shared_mtp_canonical_validation_us);
        self.shared_mtp_canonical_validation_mismatches = self
            .shared_mtp_canonical_validation_mismatches
            .saturating_add(delta.shared_mtp_canonical_validation_mismatches);
        self.shared_mtp_canonical_fallbacks = self
            .shared_mtp_canonical_fallbacks
            .saturating_add(delta.shared_mtp_canonical_fallbacks);
        self.shared_published_requests = self
            .shared_published_requests
            .saturating_add(delta.shared_published_requests);
        self.shared_published_tokens = self
            .shared_published_tokens
            .saturating_add(delta.shared_published_tokens);
        self.shared_entries_current = delta.shared_entries_current;
        self.shared_entries_peak = self.shared_entries_peak.max(delta.shared_entries_peak);
        self.shared_evictions = self.shared_evictions.saturating_add(delta.shared_evictions);
        self.shared_pressure_evictions = self
            .shared_pressure_evictions
            .saturating_add(delta.shared_pressure_evictions);
        self.shared_clear_count = self
            .shared_clear_count
            .saturating_add(delta.shared_clear_count);
        self.shared_cleared_entries = self
            .shared_cleared_entries
            .saturating_add(delta.shared_cleared_entries);
        self.shared_estimated_bytes_current = delta.shared_estimated_bytes_current;
        self.shared_estimated_bytes_peak = self
            .shared_estimated_bytes_peak
            .max(delta.shared_estimated_bytes_peak);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptLookupCostAction {
    Ordinary,
    Lookup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct PromptLookupQualificationRegime {
    pub batch_width: usize,
    pub context_bucket_tokens: usize,
    pub sampler: PromptLookupSamplerFingerprint,
    pub proposal_source: Option<PromptLookupProposalSource>,
    pub verify_width: usize,
}

impl PromptLookupQualificationRegime {
    pub(crate) fn new(batch_width: usize, context_tokens: usize, sampler: Sampler) -> Self {
        Self {
            batch_width: batch_width.max(1),
            context_bucket_tokens: context_bucket(context_tokens),
            sampler: sampler.into(),
            proposal_source: None,
            verify_width: 1,
        }
    }

    pub(crate) fn with_proposal(
        mut self,
        proposal_source: PromptLookupProposalSource,
        verify_width: usize,
    ) -> Self {
        self.proposal_source = Some(proposal_source);
        self.verify_width = verify_width.max(2);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct PromptLookupSamplerFingerprint {
    temperature_bits: u32,
    top_k: Option<i32>,
    top_p_bits: Option<u32>,
    min_p_bits: Option<u32>,
    repetition_penalty_bits: Option<u32>,
    frequency_penalty_bits: Option<u32>,
    presence_penalty_bits: Option<u32>,
}

impl From<Sampler> for PromptLookupSamplerFingerprint {
    fn from(sampler: Sampler) -> Self {
        Self {
            temperature_bits: sampler.temperature.to_bits(),
            top_k: sampler.top_k,
            top_p_bits: sampler.top_p.map(f32::to_bits),
            min_p_bits: sampler.min_p.map(f32::to_bits),
            repetition_penalty_bits: sampler.repetition_penalty.map(f32::to_bits),
            frequency_penalty_bits: sampler.frequency_penalty.map(f32::to_bits),
            presence_penalty_bits: sampler.presence_penalty.map(f32::to_bits),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PromptLookupQualificationRuntimeConfig {
    context_fingerprint: String,
    profile_path: PathBuf,
    baseline: PromptLookupQualificationBaseline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptLookupQualificationBaseline {
    Ordinary,
    QwenMtp,
    Gemma4Assistant,
}

impl PromptLookupQualificationRuntimeConfig {
    pub(crate) fn for_scheduler_profile(profile: &SchedulerAutotuneRuntimeProfile) -> Result<Self> {
        Self::for_scheduler_profile_with_baseline(
            profile,
            PromptLookupQualificationBaseline::Ordinary,
        )
    }

    pub(crate) fn for_scheduler_profile_with_baseline(
        profile: &SchedulerAutotuneRuntimeProfile,
        baseline: PromptLookupQualificationBaseline,
    ) -> Result<Self> {
        let context_fingerprint = qualification_context_fingerprint(profile, baseline)?;
        let home = dirs::home_dir()
            .context("locating home directory for PromptLookup qualification profiles")?;
        Ok(Self {
            profile_path: home
                .join(".ironmlx")
                .join("prompt-lookup-qualifications")
                .join("profiles")
                .join(format!("{context_fingerprint}.json")),
            context_fingerprint,
            baseline,
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(context_fingerprint: &str, profile_path: PathBuf) -> Self {
        Self {
            context_fingerprint: context_fingerprint.to_string(),
            profile_path,
            baseline: PromptLookupQualificationBaseline::Ordinary,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PromptLookupQualificationStats {
    pub ordinary_cost_samples: u64,
    pub lookup_cost_samples: u64,
    pub ordinary_cost_us: u64,
    pub lookup_cost_us: u64,
    pub qualified_regimes_current: u64,
    pub rejected_regimes_current: u64,
    pub qualification_changes: u64,
    pub profile_loads: u64,
    pub profile_writes: u64,
    pub profile_write_drops: u64,
    pub query_gate_skips: u64,
    pub miss_query_gate_skips: u64,
    pub miss_query_reprobes: u64,
}

#[derive(Debug)]
pub(crate) struct PromptLookupCostController {
    runtime: PromptLookupQualificationRuntimeConfig,
    regimes: HashMap<PromptLookupQualificationRegime, QualificationRegimeState>,
    writer: QualificationProfileWriter,
    stats: PromptLookupQualificationStats,
}

#[derive(Debug)]
struct QualificationRegimeState {
    phase: QualificationPhase,
    last_evidence: Option<PromptLookupQualificationEvidence>,
    next_rejected_cooldown_tokens: u64,
    transition_cost_per_token_ns: u64,
}

#[derive(Debug)]
enum QualificationPhase {
    Delayed {
        samples: VecDeque<u64>,
        remaining_tokens: u64,
    },
    Baseline {
        samples: Vec<u64>,
    },
    Probe {
        baseline_cost_per_token_ns: u64,
        samples: Vec<u64>,
        counters: PromptLookupQualificationCounters,
    },
    Qualified {
        baseline_cost_per_token_ns: u64,
        rolling_lookup_samples: VecDeque<u64>,
        rolling_counters: VecDeque<PromptLookupQualificationCounters>,
        tokens_until_revalidate: u64,
    },
    Rejected {
        cooldown_tokens: u64,
    },
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct PromptLookupQualificationCounters {
    queries: u64,
    hits: u64,
    misses: u64,
    drafted_tokens: u64,
    accepted_tokens: u64,
    rollback_count: u64,
}

impl PromptLookupQualificationCounters {
    fn accumulate(&mut self, delta: PromptLookupStats, source: Option<PromptLookupProposalSource>) {
        let source_delta = match source {
            Some(PromptLookupProposalSource::Local) => delta.local_source,
            Some(PromptLookupProposalSource::Shared) => delta.shared_source,
            Some(PromptLookupProposalSource::Mixed) | None => PromptLookupSourceStats {
                queries: delta.queries,
                hits: delta.hits,
                misses: delta.misses,
                drafted_tokens: delta.drafted_tokens,
                accepted_tokens: delta.accepted_tokens,
                zero_accept_windows: delta.zero_accept_windows,
                wasted_verify_tokens: delta.rejected_tokens,
                propose_us: delta.propose_us,
                verify_us: delta.verify_round_us,
                rollback_us: delta.rollback_us,
            },
        };
        self.queries = self.queries.saturating_add(source_delta.queries);
        self.hits = self.hits.saturating_add(source_delta.hits);
        self.misses = self.misses.saturating_add(source_delta.misses);
        self.drafted_tokens = self
            .drafted_tokens
            .saturating_add(source_delta.drafted_tokens);
        self.accepted_tokens = self
            .accepted_tokens
            .saturating_add(source_delta.accepted_tokens);
        self.rollback_count = self.rollback_count.saturating_add(delta.rollback_count);
    }

    fn accumulate_counters(&mut self, delta: Self) {
        self.queries = self.queries.saturating_add(delta.queries);
        self.hits = self.hits.saturating_add(delta.hits);
        self.misses = self.misses.saturating_add(delta.misses);
        self.drafted_tokens = self.drafted_tokens.saturating_add(delta.drafted_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(delta.accepted_tokens);
        self.rollback_count = self.rollback_count.saturating_add(delta.rollback_count);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PromptLookupQualificationDecision {
    Qualified,
    Rejected,
    Ineligible,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PromptLookupQualificationEvidence {
    regime: PromptLookupQualificationRegime,
    decision: PromptLookupQualificationDecision,
    baseline_cost_per_token_ns: u64,
    lookup_cost_per_token_ns: u64,
    transition_cost_per_token_ns: u64,
    estimated_gain_bps: i64,
    baseline_samples: usize,
    lookup_samples: usize,
    counters: PromptLookupQualificationCounters,
    rejected_cooldown_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PromptLookupQualificationProfile {
    schema_version: u32,
    context_fingerprint: String,
    baseline: PromptLookupQualificationBaseline,
    updated_at_unix_ms: u64,
    entries: Vec<PromptLookupQualificationEvidence>,
}

#[derive(Debug)]
struct QualificationProfileWriter {
    mailbox: Arc<QualificationProfileMailbox>,
    worker: Option<std::thread::JoinHandle<()>>,
}

#[derive(Debug)]
struct QualificationProfileMailbox {
    state: Mutex<QualificationProfileMailboxState>,
    wake: Condvar,
}

#[derive(Debug, Default)]
struct QualificationProfileMailboxState {
    pending: Option<PromptLookupQualificationProfile>,
    closed: bool,
}

impl QualificationProfileWriter {
    fn new(path: PathBuf) -> Result<Self> {
        let mailbox = Arc::new(QualificationProfileMailbox {
            state: Mutex::new(QualificationProfileMailboxState::default()),
            wake: Condvar::new(),
        });
        let worker_mailbox = Arc::clone(&mailbox);
        let worker = std::thread::Builder::new()
            .name("prompt-lookup-profile-writer".to_string())
            .spawn(move || loop {
                let profile = {
                    let mut state = worker_mailbox
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    while state.pending.is_none() && !state.closed {
                        state = worker_mailbox
                            .wake
                            .wait(state)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    match state.pending.take() {
                        Some(profile) => profile,
                        None if state.closed => break,
                        None => continue,
                    }
                };
                if let Err(error) = persist_qualification_profile(&path, &profile) {
                    tracing::warn!(
                        target: "ironmlx::prompt_lookup",
                        path = %path.display(),
                        error = %error,
                        "failed to persist PromptLookup qualification profile"
                    );
                }
            })
            .context("spawning PromptLookup qualification profile writer")?;
        Ok(Self {
            mailbox,
            worker: Some(worker),
        })
    }

    fn queue_latest(&self, profile: PromptLookupQualificationProfile) -> bool {
        let mut state = self
            .mailbox
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let replaced = state.pending.replace(profile).is_some();
        self.mailbox.wake.notify_one();
        replaced
    }
}

impl Drop for QualificationProfileWriter {
    fn drop(&mut self) {
        {
            let mut state = self
                .mailbox
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.closed = true;
            self.mailbox.wake.notify_one();
        }
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                tracing::warn!(
                    target: "ironmlx::prompt_lookup",
                    "PromptLookup qualification profile writer panicked during shutdown"
                );
            }
        }
    }
}

impl PromptLookupCostController {
    pub(crate) fn new(runtime: PromptLookupQualificationRuntimeConfig) -> Result<Self> {
        let mut stats = PromptLookupQualificationStats::default();
        let regimes = match load_qualification_profile(&runtime) {
            Ok(Some(profile)) => {
                stats.profile_loads = 1;
                profile
                    .entries
                    .into_iter()
                    .map(|evidence| {
                        let (phase, next_rejected_cooldown_tokens) = match evidence.decision {
                            PromptLookupQualificationDecision::Qualified => (
                                QualificationPhase::Qualified {
                                    baseline_cost_per_token_ns: evidence.baseline_cost_per_token_ns,
                                    rolling_lookup_samples: VecDeque::new(),
                                    rolling_counters: VecDeque::new(),
                                    tokens_until_revalidate: QUALIFICATION_REVALIDATE_TOKENS,
                                },
                                QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS,
                            ),
                            PromptLookupQualificationDecision::Rejected
                            | PromptLookupQualificationDecision::Ineligible => (
                                QualificationPhase::Rejected {
                                    cooldown_tokens: evidence.rejected_cooldown_tokens,
                                },
                                next_rejected_cooldown(evidence.rejected_cooldown_tokens),
                            ),
                        };
                        (
                            evidence.regime,
                            QualificationRegimeState {
                                transition_cost_per_token_ns: evidence.transition_cost_per_token_ns,
                                phase,
                                last_evidence: Some(evidence),
                                next_rejected_cooldown_tokens,
                            },
                        )
                    })
                    .collect()
            }
            Ok(None) => HashMap::new(),
            Err(error) => {
                tracing::warn!(
                    target: "ironmlx::prompt_lookup",
                    path = %runtime.profile_path.display(),
                    error = %error,
                    "ignoring invalid PromptLookup qualification profile"
                );
                HashMap::new()
            }
        };
        let writer = QualificationProfileWriter::new(runtime.profile_path.clone())?;
        let mut controller = Self {
            runtime,
            regimes,
            writer,
            stats,
        };
        controller.refresh_regime_gauges();
        Ok(controller)
    }

    pub(crate) fn next_action(
        &mut self,
        regime: PromptLookupQualificationRegime,
    ) -> PromptLookupCostAction {
        let state = self
            .regimes
            .entry(regime)
            .or_insert_with(|| initial_regime_state(regime));
        match state.phase {
            QualificationPhase::Delayed { .. }
            | QualificationPhase::Baseline { .. }
            | QualificationPhase::Rejected { .. } => PromptLookupCostAction::Ordinary,
            QualificationPhase::Probe { .. } | QualificationPhase::Qualified { .. } => {
                PromptLookupCostAction::Lookup
            }
        }
    }

    pub(crate) fn record_sample(
        &mut self,
        regime: PromptLookupQualificationRegime,
        action: PromptLookupCostAction,
        elapsed_ns: u64,
        committed_tokens: usize,
        prompt_lookup_delta: PromptLookupStats,
    ) {
        if committed_tokens == 0 {
            return;
        }
        let state = self
            .regimes
            .entry(regime)
            .or_insert_with(|| initial_regime_state(regime));
        let cost_per_token_ns = (elapsed_ns / committed_tokens as u64).saturating_add(
            if action == PromptLookupCostAction::Lookup {
                state.transition_cost_per_token_ns
            } else {
                0
            },
        );
        let progress_tokens = normalized_progress_tokens(regime, committed_tokens);
        let mut persist = false;
        match &mut state.phase {
            QualificationPhase::Delayed {
                samples,
                remaining_tokens,
            } => {
                if action != PromptLookupCostAction::Ordinary {
                    return;
                }
                self.stats.ordinary_cost_samples =
                    self.stats.ordinary_cost_samples.saturating_add(1);
                self.stats.ordinary_cost_us = self
                    .stats
                    .ordinary_cost_us
                    .saturating_add(elapsed_ns / 1_000);
                samples.push_back(cost_per_token_ns);
                while samples.len() > QUALIFICATION_BASELINE_SAMPLES {
                    samples.pop_front();
                }
                *remaining_tokens = remaining_tokens.saturating_sub(progress_tokens);
                if *remaining_tokens == 0 {
                    if samples.len() == QUALIFICATION_BASELINE_SAMPLES {
                        state.phase = QualificationPhase::Probe {
                            baseline_cost_per_token_ns: median_deque(samples),
                            samples: Vec::with_capacity(QUALIFICATION_PROBE_SAMPLES),
                            counters: PromptLookupQualificationCounters::default(),
                        };
                    } else {
                        state.phase = QualificationPhase::Baseline {
                            samples: samples.iter().copied().collect(),
                        };
                    }
                }
            }
            QualificationPhase::Baseline { samples } => {
                if action != PromptLookupCostAction::Ordinary {
                    return;
                }
                self.stats.ordinary_cost_samples =
                    self.stats.ordinary_cost_samples.saturating_add(1);
                self.stats.ordinary_cost_us = self
                    .stats
                    .ordinary_cost_us
                    .saturating_add(elapsed_ns / 1_000);
                samples.push(cost_per_token_ns);
                if samples.len() >= QUALIFICATION_BASELINE_SAMPLES {
                    let baseline_cost_per_token_ns = median(samples);
                    state.phase = QualificationPhase::Probe {
                        baseline_cost_per_token_ns,
                        samples: Vec::with_capacity(QUALIFICATION_PROBE_SAMPLES),
                        counters: PromptLookupQualificationCounters::default(),
                    };
                }
            }
            QualificationPhase::Probe {
                baseline_cost_per_token_ns,
                samples,
                counters,
            } => {
                if action != PromptLookupCostAction::Lookup {
                    return;
                }
                self.stats.lookup_cost_samples = self.stats.lookup_cost_samples.saturating_add(1);
                self.stats.lookup_cost_us =
                    self.stats.lookup_cost_us.saturating_add(elapsed_ns / 1_000);
                samples.push(cost_per_token_ns);
                counters.accumulate(prompt_lookup_delta, regime.proposal_source);
                if samples.len() >= QUALIFICATION_PROBE_SAMPLES {
                    let baseline = *baseline_cost_per_token_ns;
                    let lookup = median(samples);
                    let decision = qualification_decision(baseline, lookup);
                    let rejected_cooldown_tokens = match decision {
                        PromptLookupQualificationDecision::Qualified => 0,
                        PromptLookupQualificationDecision::Rejected
                        | PromptLookupQualificationDecision::Ineligible => {
                            state.next_rejected_cooldown_tokens
                        }
                    };
                    let evidence = PromptLookupQualificationEvidence {
                        regime,
                        decision,
                        baseline_cost_per_token_ns: baseline,
                        lookup_cost_per_token_ns: lookup,
                        transition_cost_per_token_ns: state.transition_cost_per_token_ns,
                        estimated_gain_bps: estimated_gain_bps(baseline, lookup),
                        baseline_samples: QUALIFICATION_BASELINE_SAMPLES,
                        lookup_samples: QUALIFICATION_PROBE_SAMPLES,
                        counters: *counters,
                        rejected_cooldown_tokens,
                    };
                    state.last_evidence = Some(evidence);
                    state.phase = match decision {
                        PromptLookupQualificationDecision::Qualified => {
                            state.next_rejected_cooldown_tokens =
                                QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS;
                            QualificationPhase::Qualified {
                                baseline_cost_per_token_ns: baseline,
                                rolling_lookup_samples: VecDeque::new(),
                                rolling_counters: VecDeque::new(),
                                tokens_until_revalidate: QUALIFICATION_REVALIDATE_TOKENS,
                            }
                        }
                        PromptLookupQualificationDecision::Rejected
                        | PromptLookupQualificationDecision::Ineligible => {
                            state.next_rejected_cooldown_tokens =
                                next_rejected_cooldown(rejected_cooldown_tokens);
                            QualificationPhase::Rejected {
                                cooldown_tokens: rejected_cooldown_tokens,
                            }
                        }
                    };
                    self.stats.qualification_changes =
                        self.stats.qualification_changes.saturating_add(1);
                    persist = true;
                }
            }
            QualificationPhase::Qualified {
                baseline_cost_per_token_ns,
                rolling_lookup_samples,
                rolling_counters,
                tokens_until_revalidate,
            } => {
                if action != PromptLookupCostAction::Lookup {
                    return;
                }
                self.stats.lookup_cost_samples = self.stats.lookup_cost_samples.saturating_add(1);
                self.stats.lookup_cost_us =
                    self.stats.lookup_cost_us.saturating_add(elapsed_ns / 1_000);
                rolling_lookup_samples.push_back(cost_per_token_ns);
                let mut counter_delta = PromptLookupQualificationCounters::default();
                counter_delta.accumulate(prompt_lookup_delta, regime.proposal_source);
                rolling_counters.push_back(counter_delta);
                while rolling_lookup_samples.len() > QUALIFICATION_PROBE_SAMPLES {
                    rolling_lookup_samples.pop_front();
                    rolling_counters.pop_front();
                }
                *tokens_until_revalidate = tokens_until_revalidate.saturating_sub(progress_tokens);
                let drifted = rolling_lookup_samples.len() == QUALIFICATION_PROBE_SAMPLES
                    && qualification_decision(
                        *baseline_cost_per_token_ns,
                        median_deque(rolling_lookup_samples),
                    ) == PromptLookupQualificationDecision::Rejected;
                if drifted {
                    let lookup = median_deque(rolling_lookup_samples);
                    let counters = rolling_counters.iter().copied().fold(
                        PromptLookupQualificationCounters::default(),
                        |mut total, delta| {
                            total.accumulate_counters(delta);
                            total
                        },
                    );
                    state.last_evidence = Some(PromptLookupQualificationEvidence {
                        regime,
                        decision: PromptLookupQualificationDecision::Rejected,
                        baseline_cost_per_token_ns: *baseline_cost_per_token_ns,
                        lookup_cost_per_token_ns: lookup,
                        transition_cost_per_token_ns: state.transition_cost_per_token_ns,
                        estimated_gain_bps: estimated_gain_bps(*baseline_cost_per_token_ns, lookup),
                        baseline_samples: QUALIFICATION_BASELINE_SAMPLES,
                        lookup_samples: QUALIFICATION_PROBE_SAMPLES,
                        counters,
                        rejected_cooldown_tokens: state.next_rejected_cooldown_tokens,
                    });
                    let rejected_cooldown_tokens = state.next_rejected_cooldown_tokens;
                    state.next_rejected_cooldown_tokens =
                        next_rejected_cooldown(rejected_cooldown_tokens);
                    state.phase = QualificationPhase::Rejected {
                        cooldown_tokens: rejected_cooldown_tokens,
                    };
                    self.stats.qualification_changes =
                        self.stats.qualification_changes.saturating_add(1);
                    persist = true;
                } else if *tokens_until_revalidate == 0 {
                    state.phase = QualificationPhase::Baseline {
                        samples: Vec::with_capacity(QUALIFICATION_BASELINE_SAMPLES),
                    };
                    state.last_evidence = None;
                    self.stats.qualification_changes =
                        self.stats.qualification_changes.saturating_add(1);
                    persist = true;
                }
            }
            QualificationPhase::Rejected { cooldown_tokens } => {
                if action != PromptLookupCostAction::Ordinary {
                    return;
                }
                self.stats.ordinary_cost_samples =
                    self.stats.ordinary_cost_samples.saturating_add(1);
                self.stats.ordinary_cost_us = self
                    .stats
                    .ordinary_cost_us
                    .saturating_add(elapsed_ns / 1_000);
                *cooldown_tokens = cooldown_tokens.saturating_sub(progress_tokens);
                if *cooldown_tokens == 0 {
                    state.phase = QualificationPhase::Baseline {
                        samples: Vec::with_capacity(QUALIFICATION_BASELINE_SAMPLES),
                    };
                    self.stats.qualification_changes =
                        self.stats.qualification_changes.saturating_add(1);
                }
            }
        }
        self.refresh_regime_gauges();
        if persist {
            self.queue_profile_write();
        }
    }

    pub(crate) fn record_lookup_transition(
        &mut self,
        regimes: &[PromptLookupQualificationRegime],
        elapsed_ns: u64,
        episode_committed_tokens: usize,
    ) {
        if regimes.is_empty() || elapsed_ns == 0 || episode_committed_tokens == 0 {
            return;
        }
        let transition_cost_per_token_ns = elapsed_ns / episode_committed_tokens as u64;
        self.stats.lookup_cost_us = self.stats.lookup_cost_us.saturating_add(elapsed_ns / 1_000);
        let mut persist = false;

        for regime in regimes {
            let state = self
                .regimes
                .entry(*regime)
                .or_insert_with(|| initial_regime_state(*regime));
            let previous_transition_cost = state.transition_cost_per_token_ns;
            state.transition_cost_per_token_ns =
                previous_transition_cost.max(transition_cost_per_token_ns);
            let Some(mut evidence) = state.last_evidence.clone() else {
                continue;
            };

            let lookup_execution_cost = evidence
                .lookup_cost_per_token_ns
                .saturating_sub(evidence.transition_cost_per_token_ns);
            evidence.transition_cost_per_token_ns = state.transition_cost_per_token_ns;
            evidence.lookup_cost_per_token_ns =
                lookup_execution_cost.saturating_add(state.transition_cost_per_token_ns);
            evidence.estimated_gain_bps = estimated_gain_bps(
                evidence.baseline_cost_per_token_ns,
                evidence.lookup_cost_per_token_ns,
            );
            evidence.decision = qualification_decision(
                evidence.baseline_cost_per_token_ns,
                evidence.lookup_cost_per_token_ns,
            );

            let minimum_cooldown = minimum_rejected_cooldown_tokens(
                *regime,
                evidence.baseline_cost_per_token_ns,
                elapsed_ns,
            );
            match evidence.decision {
                PromptLookupQualificationDecision::Qualified => {
                    evidence.rejected_cooldown_tokens = 0;
                }
                PromptLookupQualificationDecision::Rejected
                | PromptLookupQualificationDecision::Ineligible => {
                    let current_cooldown = match state.phase {
                        QualificationPhase::Rejected { cooldown_tokens } => cooldown_tokens,
                        _ => state.next_rejected_cooldown_tokens,
                    };
                    let cooldown_tokens = current_cooldown.max(minimum_cooldown);
                    evidence.rejected_cooldown_tokens = cooldown_tokens;
                    state.phase = QualificationPhase::Rejected { cooldown_tokens };
                    state.next_rejected_cooldown_tokens = state
                        .next_rejected_cooldown_tokens
                        .max(next_rejected_cooldown(cooldown_tokens));
                    self.stats.qualification_changes =
                        self.stats.qualification_changes.saturating_add(1);
                }
            }
            state.last_evidence = Some(evidence);
            persist = true;
        }

        self.refresh_regime_gauges();
        if persist {
            self.queue_profile_write();
        }
    }

    pub(crate) fn record_lookup_ineligible(&mut self, regime: PromptLookupQualificationRegime) {
        let state = self
            .regimes
            .entry(regime)
            .or_insert_with(|| initial_regime_state(regime));
        let baseline_cost_per_token_ns = match &state.phase {
            QualificationPhase::Probe {
                baseline_cost_per_token_ns,
                ..
            }
            | QualificationPhase::Qualified {
                baseline_cost_per_token_ns,
                ..
            } => *baseline_cost_per_token_ns,
            _ => state
                .last_evidence
                .as_ref()
                .map_or(0, |evidence| evidence.baseline_cost_per_token_ns),
        };
        state.last_evidence = Some(PromptLookupQualificationEvidence {
            regime,
            decision: PromptLookupQualificationDecision::Ineligible,
            baseline_cost_per_token_ns,
            lookup_cost_per_token_ns: 0,
            transition_cost_per_token_ns: 0,
            estimated_gain_bps: 0,
            baseline_samples: QUALIFICATION_BASELINE_SAMPLES,
            lookup_samples: 0,
            counters: PromptLookupQualificationCounters::default(),
            rejected_cooldown_tokens: QUALIFICATION_REJECTED_MAX_COOLDOWN_TOKENS,
        });
        state.phase = QualificationPhase::Rejected {
            cooldown_tokens: QUALIFICATION_REJECTED_MAX_COOLDOWN_TOKENS,
        };
        state.next_rejected_cooldown_tokens = QUALIFICATION_REJECTED_MAX_COOLDOWN_TOKENS;
        self.stats.qualification_changes = self.stats.qualification_changes.saturating_add(1);
        self.refresh_regime_gauges();
        self.queue_profile_write();
    }

    pub(crate) fn record_query_gate_skip(&mut self) {
        self.stats.query_gate_skips = self.stats.query_gate_skips.saturating_add(1);
    }

    pub(crate) fn record_miss_query_gate_skip(&mut self) {
        self.stats.miss_query_gate_skips = self.stats.miss_query_gate_skips.saturating_add(1);
    }

    pub(crate) fn record_miss_query_reprobe(&mut self) {
        self.stats.miss_query_reprobes = self.stats.miss_query_reprobes.saturating_add(1);
    }

    pub(crate) fn stats(&self) -> PromptLookupQualificationStats {
        self.stats
    }

    fn refresh_regime_gauges(&mut self) {
        self.stats.qualified_regimes_current = self
            .regimes
            .values()
            .filter(|state| matches!(state.phase, QualificationPhase::Qualified { .. }))
            .count() as u64;
        self.stats.rejected_regimes_current = self
            .regimes
            .values()
            .filter(|state| matches!(state.phase, QualificationPhase::Rejected { .. }))
            .count() as u64;
    }

    fn queue_profile_write(&mut self) {
        let profile = PromptLookupQualificationProfile {
            schema_version: QUALIFICATION_SCHEMA_VERSION,
            context_fingerprint: self.runtime.context_fingerprint.clone(),
            baseline: self.runtime.baseline,
            updated_at_unix_ms: unix_time_ms(),
            entries: self
                .regimes
                .values()
                .filter_map(|state| state.last_evidence.clone())
                .collect(),
        };
        let replaced = self.writer.queue_latest(profile);
        self.stats.profile_writes = self.stats.profile_writes.saturating_add(1);
        if replaced {
            self.stats.profile_write_drops = self.stats.profile_write_drops.saturating_add(1);
        }
    }
}

fn qualification_context_fingerprint(
    profile: &SchedulerAutotuneRuntimeProfile,
    baseline: PromptLookupQualificationBaseline,
) -> Result<String> {
    let encoded = serde_json::to_vec(&(
        QUALIFICATION_SCHEMA_VERSION,
        env!("CARGO_PKG_VERSION"),
        baseline,
        &profile.hardware_label,
        profile.runtime_context.fingerprint(),
        profile.config,
        &profile.rules,
    ))?;
    Ok(fnv1a_hex(&encoded))
}

fn initial_regime_state(regime: PromptLookupQualificationRegime) -> QualificationRegimeState {
    let phase = if regime.batch_width > 1 {
        QualificationPhase::Delayed {
            samples: VecDeque::with_capacity(QUALIFICATION_BASELINE_SAMPLES),
            remaining_tokens: QUALIFICATION_MULTI_BATCH_INITIAL_DELAY_TOKENS,
        }
    } else {
        QualificationPhase::Baseline {
            samples: Vec::with_capacity(QUALIFICATION_BASELINE_SAMPLES),
        }
    };
    QualificationRegimeState {
        phase,
        last_evidence: None,
        next_rejected_cooldown_tokens: QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS,
        transition_cost_per_token_ns: 0,
    }
}

fn normalized_progress_tokens(
    regime: PromptLookupQualificationRegime,
    committed_tokens: usize,
) -> u64 {
    committed_tokens
        .div_ceil(regime.batch_width)
        .try_into()
        .unwrap_or(u64::MAX)
}

fn next_rejected_cooldown(current: u64) -> u64 {
    current.saturating_mul(4).clamp(
        QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS,
        QUALIFICATION_REJECTED_MAX_COOLDOWN_TOKENS,
    )
}

fn minimum_rejected_cooldown_tokens(
    regime: PromptLookupQualificationRegime,
    baseline_cost_per_token_ns: u64,
    transition_elapsed_ns: u64,
) -> u64 {
    let normalized_token_cost_ns = baseline_cost_per_token_ns
        .saturating_mul(regime.batch_width.try_into().unwrap_or(u64::MAX));
    let allowed_overhead_ns =
        normalized_token_cost_ns.saturating_mul(QUALIFICATION_MAX_REPROBE_OVERHEAD_BPS) / 10_000;
    if allowed_overhead_ns == 0 {
        return QUALIFICATION_REJECTED_MAX_COOLDOWN_TOKENS;
    }
    let required_tokens = transition_elapsed_ns.div_ceil(allowed_overhead_ns);
    let mut cooldown = QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS;
    while cooldown < required_tokens && cooldown < QUALIFICATION_REJECTED_MAX_COOLDOWN_TOKENS {
        cooldown = next_rejected_cooldown(cooldown);
    }
    cooldown
}

fn context_bucket(context_tokens: usize) -> usize {
    context_tokens
        .max(1_024)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
}

fn qualification_decision(
    baseline_cost_per_token_ns: u64,
    lookup_cost_per_token_ns: u64,
) -> PromptLookupQualificationDecision {
    if lookup_cost_per_token_ns.saturating_mul(10_000)
        <= baseline_cost_per_token_ns.saturating_mul(10_000 - QUALIFICATION_MIN_GAIN_BPS)
    {
        PromptLookupQualificationDecision::Qualified
    } else {
        PromptLookupQualificationDecision::Rejected
    }
}

fn estimated_gain_bps(baseline_cost_per_token_ns: u64, lookup_cost_per_token_ns: u64) -> i64 {
    if baseline_cost_per_token_ns == 0 {
        return 0;
    }
    let baseline = i128::from(baseline_cost_per_token_ns);
    let lookup = i128::from(lookup_cost_per_token_ns);
    (((baseline - lookup) * 10_000) / baseline).clamp(i128::from(i64::MIN), i128::from(i64::MAX))
        as i64
}

fn median(samples: &[u64]) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        sorted[middle - 1].saturating_add(sorted[middle]) / 2
    } else {
        sorted[middle]
    }
}

fn median_deque(samples: &VecDeque<u64>) -> u64 {
    median(&samples.iter().copied().collect::<Vec<_>>())
}

fn load_qualification_profile(
    runtime: &PromptLookupQualificationRuntimeConfig,
) -> Result<Option<PromptLookupQualificationProfile>> {
    let raw = match std::fs::read_to_string(&runtime.profile_path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("reading {}", runtime.profile_path.display()));
        }
    };
    let profile: PromptLookupQualificationProfile = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {}", runtime.profile_path.display()))?;
    if profile.schema_version != QUALIFICATION_SCHEMA_VERSION {
        bail!(
            "PromptLookup qualification schema mismatch: expected {}, got {}",
            QUALIFICATION_SCHEMA_VERSION,
            profile.schema_version
        );
    }
    if profile.context_fingerprint != runtime.context_fingerprint {
        bail!("PromptLookup qualification context fingerprint mismatch");
    }
    if profile.baseline != runtime.baseline {
        bail!("PromptLookup qualification baseline mismatch");
    }
    if unix_time_ms().saturating_sub(profile.updated_at_unix_ms) > QUALIFICATION_PROFILE_TTL_MS {
        return Ok(None);
    }
    for evidence in &profile.entries {
        match evidence.decision {
            PromptLookupQualificationDecision::Qualified
                if evidence.rejected_cooldown_tokens != 0 =>
            {
                bail!("qualified PromptLookup evidence contains a rejected cooldown");
            }
            PromptLookupQualificationDecision::Rejected
            | PromptLookupQualificationDecision::Ineligible
                if !valid_rejected_cooldown(evidence.rejected_cooldown_tokens) =>
            {
                bail!(
                    "invalid PromptLookup rejected cooldown: {}",
                    evidence.rejected_cooldown_tokens
                );
            }
            _ => {}
        }
    }
    Ok(Some(profile))
}

fn valid_rejected_cooldown(cooldown_tokens: u64) -> bool {
    let mut candidate = QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS;
    loop {
        if cooldown_tokens == candidate {
            return true;
        }
        if candidate == QUALIFICATION_REJECTED_MAX_COOLDOWN_TOKENS {
            return false;
        }
        candidate = next_rejected_cooldown(candidate);
    }
}

fn persist_qualification_profile(
    path: &Path,
    profile: &PromptLookupQualificationProfile,
) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("PromptLookup qualification profile path has no parent"))?;
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let temp_path = path.with_extension(format!("tmp-{}-{}", std::process::id(), unix_time_ms()));
    let encoded = serde_json::to_vec_pretty(profile)?;
    let mut temp = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp_path)
        .with_context(|| format!("creating {}", temp_path.display()))?;
    temp.write_all(&encoded)?;
    temp.write_all(b"\n")?;
    temp.sync_all()?;
    std::fs::rename(&temp_path, path)
        .with_context(|| format!("renaming {} to {}", temp_path.display(), path.display()))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NgramKey(Box<[u32]>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PromptLookupHistoryFingerprint {
    len: usize,
    lane0: u64,
    lane1: u64,
}

impl PromptLookupHistoryFingerprint {
    const LANE0_SEED: u64 = 0xcbf2_9ce4_8422_2325;
    const LANE1_SEED: u64 = 0x6c62_272e_07bb_0142;
    const LANE0_PRIME: u64 = 0x0000_0100_0000_01b3;
    const LANE1_PRIME: u64 = 0x9e37_79b1_85eb_ca87;

    fn new() -> Self {
        Self {
            len: 0,
            lane0: Self::LANE0_SEED,
            lane1: Self::LANE1_SEED,
        }
    }

    fn push(&mut self, token: u32) {
        let position = self.len as u64;
        let value = u64::from(token) | (position.rotate_left(17) << 32);
        self.lane0 ^= value;
        self.lane0 = self.lane0.wrapping_mul(Self::LANE0_PRIME);
        self.lane0 ^= self.lane0 >> 32;
        self.lane1 ^= value.rotate_left(29) ^ position.wrapping_mul(Self::LANE0_PRIME);
        self.lane1 = self.lane1.rotate_left(23).wrapping_mul(Self::LANE1_PRIME);
        self.lane1 ^= self.lane1 >> 29;
        self.len = self.len.saturating_add(1);
    }

    #[cfg(test)]
    pub(crate) fn from_history(history: &[u32]) -> Self {
        let mut fingerprint = Self::new();
        for &token in history {
            fingerprint.push(token);
        }
        fingerprint
    }
}

#[derive(Debug, Clone)]
struct IndexLedgerEntry {
    key: NgramKey,
    continuation: usize,
}

#[derive(Debug, Clone)]
struct SharedPromptLookupCandidate {
    id: u64,
    draft: Box<[u32]>,
    mtp_certified_draft_len: usize,
    mtp_certified_history: Option<PromptLookupHistoryFingerprint>,
    mtp_policy_snapshot: Option<MtpDraftPolicySnapshot>,
    expires_at_ms: u64,
    last_access: u64,
}

#[derive(Debug, Clone)]
struct SharedPromptLookupLedgerEntry {
    id: u64,
    key: NgramKey,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SharedPromptLookupPublishResult {
    pub indexed_tokens: usize,
    pub evicted_entries: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SharedPromptLookupMtpCertification {
    pub continuation: usize,
    pub draft_len: usize,
    pub policy_snapshot: MtpDraftPolicySnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SharedPromptLookupProposal {
    pub ngram_size: usize,
    pub tokens: Vec<u32>,
    pub mtp_certified_draft_len: usize,
    pub mtp_certified_bonus_token: Option<u32>,
    pub mtp_policy_snapshot: Option<MtpDraftPolicySnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LocalPromptLookupProposal {
    pub ngram_size: usize,
    pub continuation: usize,
    pub tokens: Vec<u32>,
}

/// Bounded CPU-only n-gram pool for immutable histories from normally
/// completed requests. A scheduler instance is the trust-domain boundary.
#[derive(Debug)]
pub(crate) struct SharedPromptLookupPool {
    config: PromptLookupConfig,
    entries: HashMap<NgramKey, SharedPromptLookupCandidate>,
    ledger: VecDeque<SharedPromptLookupLedgerEntry>,
    next_id: u64,
    access_clock: u64,
    entries_peak: usize,
    estimated_bytes_peak: usize,
    availability_epoch: u64,
}

impl SharedPromptLookupPool {
    pub(crate) fn new(config: PromptLookupConfig) -> Result<Self> {
        let config = config.validate()?;
        anyhow::ensure!(
            config.cross_request,
            "shared PromptLookup pool requires cross_request=true"
        );
        Ok(Self {
            config,
            entries: HashMap::new(),
            ledger: VecDeque::new(),
            next_id: 1,
            access_clock: 0,
            entries_peak: 0,
            estimated_bytes_peak: 0,
            availability_epoch: 0,
        })
    }

    pub(crate) fn config(&self) -> PromptLookupConfig {
        self.config
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn entries_peak(&self) -> usize {
        self.entries_peak
    }

    pub(crate) fn availability_epoch(&self) -> u64 {
        self.availability_epoch
    }

    pub(crate) fn estimated_bytes(&self) -> usize {
        let entry_bytes = self
            .entries
            .iter()
            .map(|(key, candidate)| {
                std::mem::size_of::<(NgramKey, SharedPromptLookupCandidate)>()
                    .saturating_add(key.0.len().saturating_mul(std::mem::size_of::<u32>()))
                    .saturating_add(
                        candidate
                            .draft
                            .len()
                            .saturating_mul(std::mem::size_of::<u32>()),
                    )
            })
            .sum::<usize>();
        let ledger_bytes = self
            .ledger
            .iter()
            .map(|entry| {
                std::mem::size_of::<SharedPromptLookupLedgerEntry>()
                    .saturating_add(entry.key.0.len().saturating_mul(std::mem::size_of::<u32>()))
            })
            .sum::<usize>();
        entry_bytes.saturating_add(ledger_bytes)
    }

    pub(crate) fn estimated_bytes_peak(&self) -> usize {
        self.estimated_bytes_peak
    }

    pub(crate) fn clear(&mut self) -> usize {
        let cleared = self.entries.len();
        self.entries.clear();
        self.ledger.clear();
        if cleared > 0 {
            self.bump_availability_epoch();
        }
        cleared
    }

    pub(crate) fn publish_history_with_mtp_certifications(
        &mut self,
        history: &[u32],
        mtp_certifications: &[SharedPromptLookupMtpCertification],
    ) -> SharedPromptLookupPublishResult {
        let now_ms = unix_time_ms();
        let mut evicted_entries = self.prune_expired(now_ms);
        let prefix_fingerprints = (!mtp_certifications.is_empty()).then(|| {
            let mut fingerprints = Vec::with_capacity(history.len().saturating_add(1));
            let mut fingerprint = PromptLookupHistoryFingerprint::new();
            fingerprints.push(fingerprint);
            for &token in history {
                fingerprint.push(token);
                fingerprints.push(fingerprint);
            }
            fingerprints
        });
        let window_start = history
            .len()
            .saturating_sub(self.config.history_window_tokens);
        for continuation in window_start..history.len() {
            let draft_end = history
                .len()
                .min(continuation.saturating_add(self.config.max_draft_tokens));
            if continuation == draft_end {
                continue;
            }
            let mtp_certified_draft_len = mtp_certifications
                .iter()
                .filter(|certification| certification.continuation == continuation)
                .map(|certification| certification.draft_len)
                .max()
                .unwrap_or(0)
                .min(draft_end.saturating_sub(continuation));
            let mtp_certified_history = (mtp_certified_draft_len > 0)
                .then(|| {
                    prefix_fingerprints
                        .as_ref()
                        .and_then(|fingerprints| fingerprints.get(continuation))
                        .copied()
                })
                .flatten();
            let mtp_policy_snapshot = mtp_certifications
                .iter()
                .filter(|certification| certification.continuation == continuation)
                .max_by_key(|certification| certification.draft_len)
                .map(|certification| certification.policy_snapshot);
            for n in self.config.min_ngram..=self.config.max_ngram {
                if continuation < n || continuation - n < window_start {
                    continue;
                }
                let key = NgramKey(history[continuation - n..continuation].into());
                self.insert(
                    key,
                    history[continuation..draft_end].into(),
                    mtp_certified_draft_len,
                    mtp_certified_history,
                    mtp_policy_snapshot,
                    now_ms,
                );
            }
        }
        evicted_entries =
            evicted_entries.saturating_add(self.reclaim_to(self.config.max_index_entries, now_ms));
        self.entries_peak = self.entries_peak.max(self.entries.len());
        self.compact_ledger_if_needed();
        self.refresh_estimated_bytes_peak();
        if history.len() > self.config.min_ngram || evicted_entries > 0 {
            self.bump_availability_epoch();
        }
        SharedPromptLookupPublishResult {
            indexed_tokens: history.len().saturating_sub(window_start),
            evicted_entries,
        }
    }

    pub(crate) fn propose(
        &mut self,
        history: &[u32],
        history_fingerprint: PromptLookupHistoryFingerprint,
        limit: usize,
    ) -> Option<SharedPromptLookupProposal> {
        let max_draft = limit.min(self.config.max_draft_tokens);
        if max_draft == 0 {
            return None;
        }
        let now_ms = unix_time_ms();
        let max_ngram = self.config.max_ngram.min(history.len());
        for n in (self.config.min_ngram..=max_ngram).rev() {
            let key = NgramKey(history[history.len() - n..].into());
            let expired = self
                .entries
                .get(&key)
                .is_some_and(|candidate| candidate.expires_at_ms <= now_ms);
            if expired {
                if self.entries.remove(&key).is_some() {
                    self.bump_availability_epoch();
                }
                continue;
            }
            let Some(candidate) = self.entries.get_mut(&key) else {
                continue;
            };
            self.access_clock = self.access_clock.saturating_add(1);
            self.next_id = self.next_id.saturating_add(1);
            candidate.id = self.next_id;
            candidate.last_access = self.access_clock;
            self.ledger.push_back(SharedPromptLookupLedgerEntry {
                id: candidate.id,
                key,
            });
            let draft = candidate
                .draft
                .iter()
                .copied()
                .take(max_draft)
                .collect::<Vec<_>>();
            let mtp_certified_draft_len = candidate
                .mtp_certified_history
                .filter(|fingerprint| *fingerprint == history_fingerprint)
                .map_or(0, |_| candidate.mtp_certified_draft_len.min(draft.len()));
            let mtp_policy_snapshot = (mtp_certified_draft_len > 0)
                .then_some(candidate.mtp_policy_snapshot)
                .flatten();
            let mtp_certified_bonus_token = (mtp_certified_draft_len > 0)
                .then(|| draft.get(mtp_certified_draft_len).copied())
                .flatten();
            self.compact_ledger_if_needed();
            self.refresh_estimated_bytes_peak();
            return (!draft.is_empty()).then_some(SharedPromptLookupProposal {
                ngram_size: n,
                tokens: draft,
                mtp_certified_draft_len,
                mtp_certified_bonus_token,
                mtp_policy_snapshot,
            });
        }
        self.compact_ledger_if_needed();
        None
    }

    pub(crate) fn reclaim_fraction(&mut self, numerator: usize, denominator: usize) -> usize {
        let target = self
            .entries
            .len()
            .saturating_mul(numerator)
            .div_ceil(denominator.max(1));
        let evicted = self.reclaim_to(target, unix_time_ms());
        if evicted > 0 {
            self.bump_availability_epoch();
        }
        self.refresh_estimated_bytes_peak();
        evicted
    }

    fn insert(
        &mut self,
        key: NgramKey,
        draft: Box<[u32]>,
        mtp_certified_draft_len: usize,
        mtp_certified_history: Option<PromptLookupHistoryFingerprint>,
        mtp_policy_snapshot: Option<MtpDraftPolicySnapshot>,
        now_ms: u64,
    ) {
        self.access_clock = self.access_clock.saturating_add(1);
        self.next_id = self.next_id.saturating_add(1);
        let id = self.next_id;
        let candidate = SharedPromptLookupCandidate {
            id,
            draft,
            mtp_certified_draft_len,
            mtp_certified_history,
            mtp_policy_snapshot,
            expires_at_ms: now_ms.saturating_add(SHARED_PROMPT_LOOKUP_TTL_MS),
            last_access: self.access_clock,
        };
        match self.entries.entry(key.clone()) {
            std::collections::hash_map::Entry::Occupied(mut occupied)
                if occupied.get().draft.len() > candidate.draft.len() =>
            {
                let existing = occupied.get_mut();
                existing.id = candidate.id;
                existing.expires_at_ms = candidate.expires_at_ms;
                existing.last_access = candidate.last_access;
            }
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                occupied.insert(candidate);
            }
            std::collections::hash_map::Entry::Vacant(vacant) => {
                vacant.insert(candidate);
            }
        }
        self.ledger
            .push_back(SharedPromptLookupLedgerEntry { id, key });
    }

    fn prune_expired(&mut self, now_ms: u64) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, candidate| candidate.expires_at_ms > now_ms);
        before.saturating_sub(self.entries.len())
    }

    fn reclaim_to(&mut self, target: usize, now_ms: u64) -> usize {
        let mut evicted = self.prune_expired(now_ms);
        while self.entries.len() > target {
            let Some(oldest) = self.ledger.pop_front() else {
                break;
            };
            if self
                .entries
                .get(&oldest.key)
                .is_some_and(|candidate| candidate.id == oldest.id)
            {
                self.entries.remove(&oldest.key);
                evicted = evicted.saturating_add(1);
            }
        }
        self.compact_ledger_if_needed();
        evicted
    }

    fn compact_ledger_if_needed(&mut self) {
        let limit = self.entries.len().saturating_mul(4).saturating_add(1_024);
        if self.ledger.len() <= limit {
            return;
        }
        let mut current = self
            .entries
            .iter()
            .map(|(key, candidate)| {
                (
                    candidate.last_access,
                    SharedPromptLookupLedgerEntry {
                        id: candidate.id,
                        key: key.clone(),
                    },
                )
            })
            .collect::<Vec<_>>();
        current.sort_unstable_by_key(|(last_access, _)| *last_access);
        self.ledger = current.into_iter().map(|(_, entry)| entry).collect();
    }

    fn refresh_estimated_bytes_peak(&mut self) {
        self.estimated_bytes_peak = self.estimated_bytes_peak.max(self.estimated_bytes());
    }

    fn bump_availability_epoch(&mut self) {
        self.availability_epoch = self.availability_epoch.saturating_add(1);
    }
}

#[derive(Debug, Clone)]
pub struct PromptLookupRowState {
    config: PromptLookupConfig,
    history: Vec<u32>,
    history_fingerprint: PromptLookupHistoryFingerprint,
    index: HashMap<NgramKey, VecDeque<usize>>,
    ledger: VecDeque<IndexLedgerEntry>,
    index_entries_peak: usize,
    index_evictions: u64,
}

impl PromptLookupRowState {
    pub fn new(history: &[u32], config: PromptLookupConfig) -> Result<Self> {
        let config = config.validate()?;
        let mut state = Self {
            config,
            history: Vec::with_capacity(history.len()),
            history_fingerprint: PromptLookupHistoryFingerprint::new(),
            index: HashMap::new(),
            ledger: VecDeque::new(),
            index_entries_peak: 0,
            index_evictions: 0,
        };
        for &token in history {
            state.commit(token);
        }
        Ok(state)
    }

    pub fn config(&self) -> PromptLookupConfig {
        self.config
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }

    pub(crate) fn history_fingerprint(&self) -> PromptLookupHistoryFingerprint {
        self.history_fingerprint
    }

    pub fn index_entries(&self) -> usize {
        self.index.len()
    }

    pub fn index_entries_peak(&self) -> usize {
        self.index_entries_peak
    }

    pub fn index_evictions(&self) -> u64 {
        self.index_evictions
    }

    pub(crate) fn propose(&self, limit: usize) -> Option<LocalPromptLookupProposal> {
        let max_draft = limit.min(self.config.max_draft_tokens);
        if max_draft == 0 {
            return None;
        }
        let history_len = self.history.len();
        let window_start = history_len.saturating_sub(self.config.history_window_tokens);
        let max_ngram = self.config.max_ngram.min(history_len);
        for n in (self.config.min_ngram..=max_ngram).rev() {
            let suffix_start = history_len - n;
            let key = NgramKey(self.history[suffix_start..].into());
            let Some(positions) = self.index.get(&key) else {
                continue;
            };
            let mut best: Option<(usize, usize)> = None;
            for &continuation in positions.iter().rev() {
                if continuation < window_start || continuation >= suffix_start {
                    continue;
                }
                let available = history_len.saturating_sub(continuation).min(max_draft);
                if available == 0 {
                    continue;
                }
                match best {
                    Some((best_len, best_pos))
                        if best_len > available
                            || (best_len == available && best_pos > continuation) => {}
                    _ => best = Some((available, continuation)),
                }
            }
            if let Some((draft_len, continuation)) = best {
                return Some(LocalPromptLookupProposal {
                    ngram_size: n,
                    continuation,
                    tokens: self.history[continuation..continuation + draft_len].to_vec(),
                });
            }
        }
        None
    }

    pub fn commit(&mut self, token: u32) {
        self.history.push(token);
        self.history_fingerprint.push(token);
        let continuation = self.history.len() - 1;
        for n in self.config.min_ngram..=self.config.max_ngram {
            if continuation < n {
                continue;
            }
            let key = NgramKey(self.history[continuation - n..continuation].into());
            let positions = self.index.entry(key.clone()).or_default();
            positions.push_back(continuation);
            while positions.len() > POSITIONS_PER_NGRAM {
                positions.pop_front();
            }
            self.ledger
                .push_back(IndexLedgerEntry { key, continuation });
        }
        self.evict_stale();
        self.evict_to_entry_cap();
        self.index_entries_peak = self.index_entries_peak.max(self.index.len());
    }

    fn evict_stale(&mut self) {
        let min_continuation = self
            .history
            .len()
            .saturating_sub(self.config.history_window_tokens);
        while self
            .ledger
            .front()
            .is_some_and(|entry| entry.continuation < min_continuation)
        {
            self.evict_oldest_ledger_entry();
        }
    }

    fn evict_to_entry_cap(&mut self) {
        while self.index.len() > self.config.max_index_entries {
            if self.ledger.is_empty() {
                break;
            }
            self.evict_oldest_ledger_entry();
        }
    }

    fn evict_oldest_ledger_entry(&mut self) {
        let Some(entry) = self.ledger.pop_front() else {
            return;
        };
        let mut remove_key = false;
        if let Some(positions) = self.index.get_mut(&entry.key) {
            if positions.front() == Some(&entry.continuation) {
                positions.pop_front();
            }
            remove_key = positions.is_empty();
        }
        if remove_key {
            self.index.remove(&entry.key);
            self.index_evictions = self.index_evictions.saturating_add(1);
        }
    }

    pub fn validate_history(&self, expected: &[u32]) -> Result<()> {
        if self.history != expected {
            return Err(anyhow!(
                "prompt lookup history diverged: indexed {} tokens, request has {}",
                self.history.len(),
                expected.len()
            ));
        }
        Ok(())
    }

    pub fn validate_committed_tail(
        &self,
        expected_len: usize,
        expected_last_token: u32,
    ) -> Result<()> {
        if self.history.len() != expected_len
            || self.history.last().copied() != Some(expected_last_token)
        {
            return Err(anyhow!(
                "prompt lookup history tail diverged: indexed len={} last={:?}, request len={} last={}",
                self.history.len(),
                self.history.last(),
                expected_len,
                expected_last_token
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PromptLookupConfig {
        PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 3,
            max_draft_tokens: 4,
            history_window_tokens: 64,
            max_index_entries: 64,
            cross_request: false,
        }
    }

    #[test]
    fn verify_round_time_survives_delta_and_accumulation() {
        let before = PromptLookupStats {
            verify_round_us: 100,
            local_source: PromptLookupSourceStats {
                queries: 2,
                hits: 1,
                ..PromptLookupSourceStats::default()
            },
            ..PromptLookupStats::default()
        };
        let after = PromptLookupStats {
            verify_round_us: 175,
            local_source: PromptLookupSourceStats {
                queries: 5,
                hits: 3,
                drafted_tokens: 8,
                accepted_tokens: 6,
                wasted_verify_tokens: 2,
                ..PromptLookupSourceStats::default()
            },
            shared_source: PromptLookupSourceStats {
                queries: 4,
                hits: 2,
                zero_accept_windows: 1,
                ..PromptLookupSourceStats::default()
            },
            ..PromptLookupStats::default()
        };
        let delta = after.saturating_delta_since(before);
        assert_eq!(delta.verify_round_us, 75);
        assert_eq!(delta.local_source.queries, 3);
        assert_eq!(delta.local_source.hits, 2);
        assert_eq!(delta.local_source.drafted_tokens, 8);
        assert_eq!(delta.local_source.accepted_tokens, 6);
        assert_eq!(delta.local_source.wasted_verify_tokens, 2);
        assert_eq!(delta.shared_source.queries, 4);
        assert_eq!(delta.shared_source.hits, 2);
        assert_eq!(delta.shared_source.zero_accept_windows, 1);

        let mut aggregate = PromptLookupStats {
            verify_round_us: 25,
            local_source: PromptLookupSourceStats {
                queries: 7,
                ..PromptLookupSourceStats::default()
            },
            ..PromptLookupStats::default()
        };
        aggregate.accumulate_delta(delta);
        assert_eq!(aggregate.verify_round_us, 100);
        assert_eq!(aggregate.local_source.queries, 10);
        assert_eq!(aggregate.local_source.hits, 2);
        assert_eq!(aggregate.shared_source.queries, 4);
        assert_eq!(aggregate.shared_source.zero_accept_windows, 1);
    }

    #[test]
    fn proposes_continuation_for_longest_suffix_match() {
        let state = PromptLookupRowState::new(&[1, 2, 3, 4, 1, 2, 3], cfg()).unwrap();
        assert_eq!(
            state.propose(4),
            Some(LocalPromptLookupProposal {
                ngram_size: 3,
                continuation: 3,
                tokens: vec![4, 1, 2, 3],
            })
        );
    }

    #[test]
    fn does_not_match_current_suffix_to_itself() {
        let state = PromptLookupRowState::new(&[1, 2, 3], cfg()).unwrap();
        assert_eq!(state.propose(4), None);
    }

    #[test]
    fn rejected_draft_is_not_committed() {
        let mut state = PromptLookupRowState::new(&[1, 2, 3, 4, 1, 2, 3], cfg()).unwrap();
        let before = state.history().to_vec();
        let _draft = state.propose(4).unwrap();
        assert_eq!(state.history(), before);
        state.commit(9);
        assert_eq!(state.history().last(), Some(&9));
    }

    #[test]
    fn index_entry_cap_is_enforced() {
        let config = PromptLookupConfig {
            max_index_entries: 3,
            ..cfg()
        };
        let state = PromptLookupRowState::new(&(0..32).collect::<Vec<_>>(), config).unwrap();
        assert!(state.index_entries() <= 3);
        assert!(state.index_evictions() > 0);
    }

    #[test]
    fn repetitive_history_keeps_ledger_bounded_by_window() {
        let config = PromptLookupConfig {
            history_window_tokens: 16,
            ..cfg()
        };
        let state = PromptLookupRowState::new(&vec![7; 256], config).unwrap();
        let variants = config.max_ngram - config.min_ngram + 1;
        assert!(state.ledger.len() <= config.history_window_tokens * variants);
        assert!(state.index_entries() <= variants);
    }

    #[test]
    fn invalid_config_is_rejected() {
        let config = PromptLookupConfig {
            min_ngram: 4,
            max_ngram: 3,
            ..cfg()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validates_committed_tail_without_full_history_comparison() {
        let state = PromptLookupRowState::new(&[1, 2, 3, 4], cfg()).unwrap();
        state.validate_committed_tail(4, 4).unwrap();
        assert!(state.validate_committed_tail(3, 4).is_err());
        assert!(state.validate_committed_tail(4, 3).is_err());
    }

    #[test]
    fn shared_pool_reuses_only_published_immutable_history() {
        let config = PromptLookupConfig {
            cross_request: true,
            ..cfg()
        };
        let mut pool = SharedPromptLookupPool::new(config).unwrap();
        assert_eq!(
            pool.propose(
                &[1, 2, 3],
                PromptLookupHistoryFingerprint::from_history(&[1, 2, 3]),
                4,
            ),
            None
        );

        let published = pool.publish_history_with_mtp_certifications(&[1, 2, 3, 4, 5, 6], &[]);
        assert_eq!(published.indexed_tokens, 6);
        assert_eq!(
            pool.propose(
                &[9, 1, 2, 3],
                PromptLookupHistoryFingerprint::from_history(&[9, 1, 2, 3]),
                4,
            ),
            Some(SharedPromptLookupProposal {
                ngram_size: 3,
                tokens: vec![4, 5, 6],
                mtp_certified_draft_len: 0,
                mtp_certified_bonus_token: None,
                mtp_policy_snapshot: None,
            })
        );
    }

    #[test]
    fn shared_pool_preserves_mtp_certification_for_the_exact_continuation() {
        let config = PromptLookupConfig {
            cross_request: true,
            ..cfg()
        };
        let mut pool = SharedPromptLookupPool::new(config).unwrap();
        let policy_snapshot = crate::core::speculative::MtpDraftPolicyState::new(4).snapshot();
        pool.publish_history_with_mtp_certifications(
            &[1, 2, 3, 4, 5, 6],
            &[SharedPromptLookupMtpCertification {
                continuation: 3,
                draft_len: 2,
                policy_snapshot,
            }],
        );

        assert_eq!(
            pool.propose(
                &[1, 2, 3],
                PromptLookupHistoryFingerprint::from_history(&[1, 2, 3]),
                4,
            ),
            Some(SharedPromptLookupProposal {
                ngram_size: 3,
                tokens: vec![4, 5, 6],
                mtp_certified_draft_len: 2,
                mtp_certified_bonus_token: Some(6),
                mtp_policy_snapshot: Some(policy_snapshot),
            })
        );
        assert_eq!(
            pool.propose(
                &[9, 1, 2, 3],
                PromptLookupHistoryFingerprint::from_history(&[9, 1, 2, 3]),
                4,
            ),
            Some(SharedPromptLookupProposal {
                ngram_size: 3,
                tokens: vec![4, 5, 6],
                mtp_certified_draft_len: 0,
                mtp_certified_bonus_token: None,
                mtp_policy_snapshot: None,
            }),
            "matching only the n-gram must not inherit MTP certification from another history"
        );
    }

    #[test]
    fn shared_pool_enforces_global_entry_cap() {
        let config = PromptLookupConfig {
            max_index_entries: 3,
            cross_request: true,
            ..cfg()
        };
        let mut pool = SharedPromptLookupPool::new(config).unwrap();
        let published =
            pool.publish_history_with_mtp_certifications(&(0..32).collect::<Vec<_>>(), &[]);
        assert!(pool.len() <= config.max_index_entries);
        assert!(published.evicted_entries > 0);
    }

    #[test]
    fn shared_pool_pressure_reclaims_lru_entries() {
        let config = PromptLookupConfig {
            cross_request: true,
            ..cfg()
        };
        let mut pool = SharedPromptLookupPool::new(config).unwrap();
        pool.publish_history_with_mtp_certifications(&(0..32).collect::<Vec<_>>(), &[]);
        let before = pool.len();
        let evicted = pool.reclaim_fraction(1, 4);
        assert!(evicted > 0);
        assert!(pool.len() <= before.div_ceil(4));
    }

    #[test]
    fn shared_pool_availability_epoch_tracks_publish_clear_and_pressure() {
        let config = PromptLookupConfig {
            cross_request: true,
            ..cfg()
        };
        let mut pool = SharedPromptLookupPool::new(config).unwrap();
        assert_eq!(pool.availability_epoch(), 0);

        pool.publish_history_with_mtp_certifications(&(0..32).collect::<Vec<_>>(), &[]);
        let published_epoch = pool.availability_epoch();
        assert!(published_epoch > 0);

        assert!(pool.reclaim_fraction(1, 2) > 0);
        let reclaimed_epoch = pool.availability_epoch();
        assert!(reclaimed_epoch > published_epoch);

        assert!(pool.clear() > 0);
        assert!(pool.availability_epoch() > reclaimed_epoch);
    }

    #[test]
    fn shared_pool_prefers_longer_candidate_and_expires_it() {
        let config = PromptLookupConfig {
            cross_request: true,
            ..cfg()
        };
        let mut pool = SharedPromptLookupPool::new(config).unwrap();
        pool.publish_history_with_mtp_certifications(&[1, 2, 3, 4, 5, 6], &[]);
        pool.publish_history_with_mtp_certifications(&[9, 1, 2, 3, 7], &[]);

        assert_eq!(
            pool.propose(
                &[8, 1, 2, 3],
                PromptLookupHistoryFingerprint::from_history(&[8, 1, 2, 3]),
                4,
            ),
            Some(SharedPromptLookupProposal {
                ngram_size: 3,
                tokens: vec![4, 5, 6],
                mtp_certified_draft_len: 0,
                mtp_certified_bonus_token: None,
                mtp_policy_snapshot: None,
            }),
            "a recent short tail must not replace a longer candidate"
        );

        let expires_at = pool
            .entries
            .values()
            .map(|candidate| candidate.expires_at_ms)
            .max()
            .expect("published entries");
        assert!(pool.prune_expired(expires_at) > 0);
        assert_eq!(
            pool.propose(
                &[8, 1, 2, 3],
                PromptLookupHistoryFingerprint::from_history(&[8, 1, 2, 3]),
                4,
            ),
            None
        );
    }

    #[test]
    fn cost_controller_qualifies_only_after_measured_gain() {
        let path = test_profile_path("qualifies");
        let runtime = PromptLookupQualificationRuntimeConfig::for_test("ctx-a", path.clone());
        let mut controller = PromptLookupCostController::new(runtime).unwrap();
        let regime = PromptLookupQualificationRegime::new(1, 8_000, Sampler::greedy());

        for _ in 0..QUALIFICATION_BASELINE_SAMPLES {
            assert_eq!(
                controller.next_action(regime),
                PromptLookupCostAction::Ordinary
            );
            controller.record_sample(
                regime,
                PromptLookupCostAction::Ordinary,
                100_000,
                1,
                PromptLookupStats::default(),
            );
        }
        for _ in 0..QUALIFICATION_PROBE_SAMPLES {
            assert_eq!(
                controller.next_action(regime),
                PromptLookupCostAction::Lookup
            );
            controller.record_sample(
                regime,
                PromptLookupCostAction::Lookup,
                80_000,
                1,
                PromptLookupStats {
                    queries: 1,
                    hits: 1,
                    drafted_tokens: 4,
                    accepted_tokens: 4,
                    ..PromptLookupStats::default()
                },
            );
        }

        assert_eq!(
            controller.next_action(regime),
            PromptLookupCostAction::Lookup
        );
        assert_eq!(controller.stats().qualified_regimes_current, 1);
        controller.record_sample(
            regime,
            PromptLookupCostAction::Lookup,
            80_000,
            QUALIFICATION_REVALIDATE_TOKENS as usize,
            PromptLookupStats::default(),
        );
        assert_eq!(
            controller.next_action(regime),
            PromptLookupCostAction::Ordinary
        );
        assert!(controller
            .regimes
            .get(&regime)
            .expect("qualified regime")
            .last_evidence
            .is_none());
        drop(controller);
        let persisted: PromptLookupQualificationProfile =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(persisted.entries.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cost_controller_rejects_unprofitable_lookup_and_cools_down() {
        let path = test_profile_path("rejects");
        let runtime = PromptLookupQualificationRuntimeConfig::for_test("ctx-b", path.clone());
        let mut controller = PromptLookupCostController::new(runtime).unwrap();
        let regime = PromptLookupQualificationRegime::new(1, 64_000, Sampler::greedy());

        drive_unprofitable_qualification(&mut controller, regime);

        assert_eq!(
            controller.next_action(regime),
            PromptLookupCostAction::Ordinary
        );
        assert_eq!(controller.stats().rejected_regimes_current, 1);
        controller.record_sample(
            regime,
            PromptLookupCostAction::Ordinary,
            100_000,
            QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS as usize,
            PromptLookupStats::default(),
        );
        assert_eq!(
            controller.next_action(regime),
            PromptLookupCostAction::Ordinary
        );
        assert_eq!(controller.stats().rejected_regimes_current, 0);
        assert!(controller
            .regimes
            .get(&regime)
            .expect("rejected regime")
            .last_evidence
            .is_some());
        drop(controller);
        let persisted: PromptLookupQualificationProfile =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(persisted.entries.len(), 1);
        assert_eq!(
            persisted.entries[0].rejected_cooldown_tokens,
            QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn lookup_transition_cost_is_charged_to_lookup_and_can_revoke_qualification() {
        let path = test_profile_path("transition-cost");
        let runtime =
            PromptLookupQualificationRuntimeConfig::for_test("ctx-transition", path.clone());
        let mut controller = PromptLookupCostController::new(runtime).unwrap();
        let regime = PromptLookupQualificationRegime::new(1, 8_000, Sampler::greedy());

        for _ in 0..QUALIFICATION_BASELINE_SAMPLES {
            controller.record_sample(
                regime,
                PromptLookupCostAction::Ordinary,
                100_000,
                1,
                PromptLookupStats::default(),
            );
        }
        for _ in 0..QUALIFICATION_PROBE_SAMPLES {
            controller.record_sample(
                regime,
                PromptLookupCostAction::Lookup,
                80_000,
                1,
                PromptLookupStats::default(),
            );
        }
        assert_eq!(controller.stats().qualified_regimes_current, 1);

        controller.record_lookup_transition(&[regime], 400_000, 8);

        let state = controller.regimes.get(&regime).expect("transition regime");
        let QualificationPhase::Rejected { cooldown_tokens } = state.phase else {
            panic!("transition cost should revoke qualification");
        };
        assert_eq!(cooldown_tokens, 512);
        let evidence = state.last_evidence.as_ref().expect("transition evidence");
        assert_eq!(evidence.transition_cost_per_token_ns, 50_000);
        assert_eq!(evidence.lookup_cost_per_token_ns, 130_000);
        assert_eq!(
            evidence.decision,
            PromptLookupQualificationDecision::Rejected
        );
        assert_eq!(controller.stats().lookup_cost_us, 1_040);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn lookup_transition_cost_expands_reprobe_cooldown_with_batch_normalization() {
        let b1 = PromptLookupQualificationRegime::new(1, 8_000, Sampler::greedy());
        let b4 = PromptLookupQualificationRegime::new(4, 8_000, Sampler::greedy());
        assert_eq!(
            minimum_rejected_cooldown_tokens(b1, 100_000, 10_000_000),
            8_192
        );
        assert_eq!(
            minimum_rejected_cooldown_tokens(b4, 100_000, 10_000_000),
            2_048
        );
    }

    #[test]
    fn ineligible_lookup_regime_is_persisted_fail_closed() {
        let path = test_profile_path("ineligible");
        let runtime =
            PromptLookupQualificationRuntimeConfig::for_test("ctx-ineligible", path.clone());
        let mut controller = PromptLookupCostController::new(runtime.clone()).unwrap();
        let regime = PromptLookupQualificationRegime::new(2, 8_000, Sampler::greedy());

        controller.record_lookup_ineligible(regime);
        assert_eq!(
            controller.next_action(regime),
            PromptLookupCostAction::Ordinary
        );
        let evidence = controller
            .regimes
            .get(&regime)
            .and_then(|state| state.last_evidence.as_ref())
            .expect("ineligible evidence");
        assert_eq!(
            evidence.decision,
            PromptLookupQualificationDecision::Ineligible
        );
        assert_eq!(
            evidence.rejected_cooldown_tokens,
            QUALIFICATION_REJECTED_MAX_COOLDOWN_TOKENS
        );
        drop(controller);

        let mut reloaded = PromptLookupCostController::new(runtime).unwrap();
        assert_eq!(
            reloaded.next_action(regime),
            PromptLookupCostAction::Ordinary
        );
        assert_eq!(reloaded.stats().profile_loads, 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn multi_batch_regime_delays_probe_by_per_request_tokens() {
        let path = test_profile_path("multi-delay");
        let runtime = PromptLookupQualificationRuntimeConfig::for_test("ctx-delay", path.clone());
        let mut controller = PromptLookupCostController::new(runtime).unwrap();
        let regime = PromptLookupQualificationRegime::new(8, 8_000, Sampler::greedy());

        for _ in 0..QUALIFICATION_MULTI_BATCH_INITIAL_DELAY_TOKENS - 1 {
            assert_eq!(
                controller.next_action(regime),
                PromptLookupCostAction::Ordinary
            );
            controller.record_sample(
                regime,
                PromptLookupCostAction::Ordinary,
                800_000,
                regime.batch_width,
                PromptLookupStats::default(),
            );
        }
        assert_eq!(
            controller.next_action(regime),
            PromptLookupCostAction::Ordinary
        );
        controller.record_sample(
            regime,
            PromptLookupCostAction::Ordinary,
            800_000,
            regime.batch_width,
            PromptLookupStats::default(),
        );

        assert_eq!(
            controller.next_action(regime),
            PromptLookupCostAction::Lookup
        );
        assert_eq!(
            controller.stats().ordinary_cost_samples,
            QUALIFICATION_MULTI_BATCH_INITIAL_DELAY_TOKENS
        );
        drop(controller);
        assert!(!path.exists());
    }

    #[test]
    fn rejected_regime_uses_persisted_exponential_backoff() {
        let path = test_profile_path("reject-backoff");
        let runtime = PromptLookupQualificationRuntimeConfig::for_test("ctx-backoff", path.clone());
        let mut controller = PromptLookupCostController::new(runtime.clone()).unwrap();
        let regime = PromptLookupQualificationRegime::new(1, 64_000, Sampler::greedy());

        drive_unprofitable_qualification(&mut controller, regime);
        controller.record_sample(
            regime,
            PromptLookupCostAction::Ordinary,
            100_000,
            QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS as usize,
            PromptLookupStats::default(),
        );
        drive_unprofitable_qualification(&mut controller, regime);

        let state = controller.regimes.get(&regime).expect("rejected regime");
        let QualificationPhase::Rejected { cooldown_tokens } = state.phase else {
            panic!("expected rejected phase");
        };
        assert_eq!(cooldown_tokens, 2_048);
        assert_eq!(state.next_rejected_cooldown_tokens, 8_192);
        assert_eq!(
            state
                .last_evidence
                .as_ref()
                .expect("rejected evidence")
                .rejected_cooldown_tokens,
            2_048
        );
        drop(controller);

        let reloaded = PromptLookupCostController::new(runtime).unwrap();
        let state = reloaded.regimes.get(&regime).expect("reloaded regime");
        let QualificationPhase::Rejected { cooldown_tokens } = state.phase else {
            panic!("expected reloaded rejected phase");
        };
        assert_eq!(cooldown_tokens, 2_048);
        assert_eq!(state.next_rejected_cooldown_tokens, 8_192);
        assert_eq!(reloaded.stats().profile_loads, 1);
        drop(reloaded);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn qualification_profile_requires_exact_context_and_fresh_schema() {
        let path = test_profile_path("profile");
        let regime = PromptLookupQualificationRegime::new(2, 32_000, Sampler::greedy());
        let mut profile = PromptLookupQualificationProfile {
            schema_version: QUALIFICATION_SCHEMA_VERSION,
            context_fingerprint: "ctx-c".to_string(),
            baseline: PromptLookupQualificationBaseline::Ordinary,
            updated_at_unix_ms: unix_time_ms(),
            entries: vec![PromptLookupQualificationEvidence {
                regime,
                decision: PromptLookupQualificationDecision::Qualified,
                baseline_cost_per_token_ns: 100,
                lookup_cost_per_token_ns: 80,
                transition_cost_per_token_ns: 0,
                estimated_gain_bps: 2_000,
                baseline_samples: QUALIFICATION_BASELINE_SAMPLES,
                lookup_samples: QUALIFICATION_PROBE_SAMPLES,
                counters: PromptLookupQualificationCounters::default(),
                rejected_cooldown_tokens: 0,
            }],
        };
        persist_qualification_profile(&path, &profile).unwrap();
        let runtime = PromptLookupQualificationRuntimeConfig::for_test("ctx-c", path.clone());
        assert!(load_qualification_profile(&runtime).unwrap().is_some());
        let mismatch = PromptLookupQualificationRuntimeConfig::for_test("ctx-other", path.clone());
        assert!(load_qualification_profile(&mismatch).is_err());
        let baseline_mismatch = PromptLookupQualificationRuntimeConfig {
            baseline: PromptLookupQualificationBaseline::QwenMtp,
            ..runtime.clone()
        };
        assert!(load_qualification_profile(&baseline_mismatch).is_err());

        profile.schema_version = QUALIFICATION_SCHEMA_VERSION - 1;
        persist_qualification_profile(&path, &profile).unwrap();
        assert!(load_qualification_profile(&runtime).is_err());

        profile.schema_version = QUALIFICATION_SCHEMA_VERSION;
        profile.entries[0].rejected_cooldown_tokens =
            QUALIFICATION_REJECTED_INITIAL_COOLDOWN_TOKENS;
        persist_qualification_profile(&path, &profile).unwrap();
        assert!(load_qualification_profile(&runtime).is_err());

        profile.entries[0].decision = PromptLookupQualificationDecision::Rejected;
        profile.entries[0].rejected_cooldown_tokens = 1_024;
        persist_qualification_profile(&path, &profile).unwrap();
        assert!(load_qualification_profile(&runtime).is_err());

        profile.entries[0].rejected_cooldown_tokens = 2_048;
        persist_qualification_profile(&path, &profile).unwrap();
        assert!(load_qualification_profile(&runtime).unwrap().is_some());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn qualification_regime_uses_power_of_two_context_buckets() {
        assert_eq!(
            PromptLookupQualificationRegime::new(1, 1, Sampler::greedy()).context_bucket_tokens,
            1_024
        );
        assert_eq!(
            PromptLookupQualificationRegime::new(4, 8_001, Sampler::greedy()).context_bucket_tokens,
            8_192
        );
        assert_eq!(
            PromptLookupQualificationRegime::new(8, 64 * 1_024, Sampler::greedy())
                .context_bucket_tokens,
            64 * 1_024
        );
        assert_ne!(
            PromptLookupQualificationRegime::new(1, 8_000, Sampler::greedy()),
            PromptLookupQualificationRegime::new(
                1,
                8_000,
                Sampler::greedy().with_temperature(0.7).with_top_p(0.9)
            )
        );
        let base = PromptLookupQualificationRegime::new(2, 8_000, Sampler::greedy());
        assert_ne!(
            base.with_proposal(PromptLookupProposalSource::Local, 3),
            base.with_proposal(PromptLookupProposalSource::Shared, 3),
            "local and shared proposal evidence must use independent qualification regimes"
        );
        assert_ne!(
            base.with_proposal(PromptLookupProposalSource::Shared, 3),
            base.with_proposal(PromptLookupProposalSource::Shared, 5),
            "actual verify width must remain part of the cost regime"
        );
    }

    fn test_profile_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ironmlx-prompt-lookup-{name}-{}-{}.json",
            std::process::id(),
            unix_time_ms()
        ))
    }

    fn drive_unprofitable_qualification(
        controller: &mut PromptLookupCostController,
        regime: PromptLookupQualificationRegime,
    ) {
        for _ in 0..QUALIFICATION_BASELINE_SAMPLES {
            assert_eq!(
                controller.next_action(regime),
                PromptLookupCostAction::Ordinary
            );
            controller.record_sample(
                regime,
                PromptLookupCostAction::Ordinary,
                100_000,
                regime.batch_width,
                PromptLookupStats::default(),
            );
        }
        for _ in 0..QUALIFICATION_PROBE_SAMPLES {
            assert_eq!(
                controller.next_action(regime),
                PromptLookupCostAction::Lookup
            );
            controller.record_sample(
                regime,
                PromptLookupCostAction::Lookup,
                110_000,
                regime.batch_width,
                PromptLookupStats {
                    queries: 1,
                    misses: 1,
                    ..PromptLookupStats::default()
                },
            );
        }
    }
}
