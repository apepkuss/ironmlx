//! Speculative decoding helpers shared by MTP generation paths.

use std::cell::Cell;
use std::collections::VecDeque;
use std::time::Instant;

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};
use serde::{Deserialize, Serialize};

use crate::core::cache::layer::{enable_turboquant_kv_caches, LayerCache};
use crate::core::cache::{MtpCache, MtpCacheSnapshot};
use crate::core::constrained::{apply_speculative_token_masks, ConstraintSession};
use crate::core::generation_types::{GenerateEvent, GenerateRequest};
use crate::core::model_input::build_position_ids;
use crate::core::sampler::draw_uniforms;
use crate::core::tokenizer::{DecodeStream, Tokenizer};
use crate::core::Sampler;
#[cfg(test)]
use crate::core::{Loader, Model};
#[cfg(test)]
use crate::nn::MtpStepOutput;
use crate::Result;
#[cfg(test)]
use mlx::Dtype;

/// Runtime limits for a single-request MTP speculative generation stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpSpeculativeConfig {
    pub max_draft_tokens: usize,
}

thread_local! {
    static QWEN_FIXED_DRAFT_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Benchmark-only scope that freezes Qwen's adaptive MTP draft policy at the
/// configured maximum depth. Production callers must retain the adaptive
/// policy so an unprofitable drafter can fail closed to ordinary decode.
#[doc(hidden)]
pub struct QwenFixedMtpDraftDepthScope;

impl Drop for QwenFixedMtpDraftDepthScope {
    fn drop(&mut self) {
        QWEN_FIXED_DRAFT_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

#[doc(hidden)]
pub fn qwen_fixed_mtp_draft_depth_scope() -> QwenFixedMtpDraftDepthScope {
    QWEN_FIXED_DRAFT_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    QwenFixedMtpDraftDepthScope
}

fn qwen_fixed_mtp_draft_depth_is_armed() -> bool {
    QWEN_FIXED_DRAFT_DEPTH.with(|depth| depth.get() > 0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpDraftTokensArg {
    Explicit(usize),
    Omitted,
}

pub fn resolve_mtp_draft_tokens(raw_config: &serde_json::Value, arg: MtpDraftTokensArg) -> usize {
    match arg {
        MtpDraftTokensArg::Explicit(value) => value,
        MtpDraftTokensArg::Omitted => default_mtp_draft_tokens_for_config(raw_config),
    }
}

pub fn default_mtp_draft_tokens_for_config(raw_config: &serde_json::Value) -> usize {
    let model_type = raw_config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let text = raw_config
        .get("text_config")
        .and_then(serde_json::Value::as_object);
    let hidden_size = text
        .and_then(|value| value.get("hidden_size"))
        .and_then(serde_json::Value::as_i64);
    let layers = text
        .and_then(|value| value.get("num_hidden_layers"))
        .and_then(serde_json::Value::as_i64);
    let experts = text
        .and_then(|value| value.get("num_experts"))
        .and_then(serde_json::Value::as_i64);
    let experts_per_tok = text
        .and_then(|value| value.get("num_experts_per_tok"))
        .and_then(serde_json::Value::as_i64);

    match (model_type, hidden_size, layers, experts, experts_per_tok) {
        // Qwen3.6-27B Dense and Qwen3.8-27B Dense share this text
        // architecture and retain their pre-Gemma d=2 default.
        ("qwen3_5", Some(5120), Some(64), None, None) => 2,
        ("qwen3_5_moe", Some(2048), Some(40), Some(256), Some(8)) => 2,
        // Qwen3.5-4B and Gemma4 keep the conservative d=1 default.
        _ => 1,
    }
}

impl MtpSpeculativeConfig {
    pub fn new(max_draft_tokens: usize, sampler: Sampler) -> Result<Self> {
        if max_draft_tokens == 0 {
            return Err(anyhow!(
                "MtpSpeculativeConfig::new: max_draft_tokens must be > 0"
            ));
        }
        anyhow::ensure!(
            sampler.temperature.is_finite(),
            "sampling temperature must be finite"
        );
        Ok(Self { max_draft_tokens })
    }
}

const MAX_DRAFT_CAP_OBSERVATION_REGIMES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtpDraftCapContextBucket {
    UpTo2k,
    UpTo8k,
    UpTo32k,
    UpTo128k,
    Above128k,
}

impl MtpDraftCapContextBucket {
    pub fn for_tokens(tokens: usize) -> Self {
        match tokens {
            0..=2_048 => Self::UpTo2k,
            2_049..=8_192 => Self::UpTo8k,
            8_193..=32_768 => Self::UpTo32k,
            32_769..=131_072 => Self::UpTo128k,
            _ => Self::Above128k,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MtpDraftCapObservation {
    pub configured_max_draft_tokens: usize,
    pub min_draft_tokens: usize,
    pub max_draft_tokens: usize,
    pub batch_width: usize,
    pub context_bucket: MtpDraftCapContextBucket,
    pub mixed_context_buckets: bool,
    pub windows: usize,
    pub drafted_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub committed_tokens: usize,
    pub rollback_count: usize,
    pub total_us: u64,
    pub draft_forward_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub sampling_us: u64,
    pub main_rollback_us: u64,
    pub decode_cache_commit_us: u64,
    pub cache_restore_us: u64,
}

impl MtpDraftCapObservation {
    fn same_regime(&self, other: &Self) -> bool {
        self.configured_max_draft_tokens == other.configured_max_draft_tokens
            && self.min_draft_tokens == other.min_draft_tokens
            && self.max_draft_tokens == other.max_draft_tokens
            && self.batch_width == other.batch_width
            && self.context_bucket == other.context_bucket
            && self.mixed_context_buckets == other.mixed_context_buckets
    }

    fn add_assign(&mut self, other: &Self) {
        debug_assert!(self.same_regime(other));
        self.windows = self.windows.saturating_add(other.windows);
        self.drafted_tokens = self.drafted_tokens.saturating_add(other.drafted_tokens);
        self.accepted_draft_tokens = self
            .accepted_draft_tokens
            .saturating_add(other.accepted_draft_tokens);
        self.committed_tokens = self.committed_tokens.saturating_add(other.committed_tokens);
        self.rollback_count = self.rollback_count.saturating_add(other.rollback_count);
        self.total_us = self.total_us.saturating_add(other.total_us);
        self.draft_forward_us = self.draft_forward_us.saturating_add(other.draft_forward_us);
        self.verify_forward_us = self
            .verify_forward_us
            .saturating_add(other.verify_forward_us);
        self.projection_us = self.projection_us.saturating_add(other.projection_us);
        self.sampling_us = self.sampling_us.saturating_add(other.sampling_us);
        self.main_rollback_us = self.main_rollback_us.saturating_add(other.main_rollback_us);
        self.decode_cache_commit_us = self
            .decode_cache_commit_us
            .saturating_add(other.decode_cache_commit_us);
        self.cache_restore_us = self.cache_restore_us.saturating_add(other.cache_restore_us);
    }

    fn saturating_delta_since(&self, before: Option<&Self>) -> Self {
        let before = before.filter(|value| self.same_regime(value));
        Self {
            configured_max_draft_tokens: self.configured_max_draft_tokens,
            min_draft_tokens: self.min_draft_tokens,
            max_draft_tokens: self.max_draft_tokens,
            batch_width: self.batch_width,
            context_bucket: self.context_bucket,
            mixed_context_buckets: self.mixed_context_buckets,
            windows: self
                .windows
                .saturating_sub(before.map_or(0, |value| value.windows)),
            drafted_tokens: self
                .drafted_tokens
                .saturating_sub(before.map_or(0, |value| value.drafted_tokens)),
            accepted_draft_tokens: self
                .accepted_draft_tokens
                .saturating_sub(before.map_or(0, |value| value.accepted_draft_tokens)),
            committed_tokens: self
                .committed_tokens
                .saturating_sub(before.map_or(0, |value| value.committed_tokens)),
            rollback_count: self
                .rollback_count
                .saturating_sub(before.map_or(0, |value| value.rollback_count)),
            total_us: self
                .total_us
                .saturating_sub(before.map_or(0, |value| value.total_us)),
            draft_forward_us: self
                .draft_forward_us
                .saturating_sub(before.map_or(0, |value| value.draft_forward_us)),
            verify_forward_us: self
                .verify_forward_us
                .saturating_sub(before.map_or(0, |value| value.verify_forward_us)),
            projection_us: self
                .projection_us
                .saturating_sub(before.map_or(0, |value| value.projection_us)),
            sampling_us: self
                .sampling_us
                .saturating_sub(before.map_or(0, |value| value.sampling_us)),
            main_rollback_us: self
                .main_rollback_us
                .saturating_sub(before.map_or(0, |value| value.main_rollback_us)),
            decode_cache_commit_us: self
                .decode_cache_commit_us
                .saturating_sub(before.map_or(0, |value| value.decode_cache_commit_us)),
            cache_restore_us: self
                .cache_restore_us
                .saturating_sub(before.map_or(0, |value| value.cache_restore_us)),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MtpDraftCapTiming {
    draft_forward_us: u64,
    verify_forward_us: u64,
    projection_us: u64,
    sampling_us: u64,
    main_rollback_us: u64,
    decode_cache_commit_us: u64,
    cache_restore_us: u64,
}

/// Runtime counters collected by [`MtpTextGenerationStream`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MtpSpeculativeStats {
    /// Speculative windows verified by the main model.
    pub windows: usize,
    /// Draft tokens proposed by the MTP head.
    pub drafted_tokens: usize,
    /// Draft tokens accepted before mismatch.
    pub accepted_draft_tokens: usize,
    /// Non-greedy windows resolved with exact speculative sampling.
    pub exact_sampling_windows: usize,
    /// Acceptance Bernoulli draws consumed by exact sampling.
    pub exact_acceptance_draws: usize,
    /// Rejections corrected from the normalized positive residual `(p - q)+`.
    pub exact_residual_corrections: usize,
    /// Target-distribution bonus samples emitted after full draft acceptance.
    pub exact_bonus_samples: usize,
    /// Draft windows that attempted each zero-based draft position.
    pub draft_attempts_by_position: Vec<usize>,
    /// Draft windows that accepted each zero-based draft position.
    pub draft_accepts_by_position: Vec<usize>,
    /// Windows that required committing only an accepted main-cache prefix.
    pub rollback_count: usize,
    /// Windows that reused the temporary draft MTP cache after full acceptance.
    pub mtp_cache_reuse_count: usize,
    /// MTP cache token positions kept from the temporary draft cache.
    pub mtp_cache_reused_tokens: usize,
    /// Number of times adaptive draft budget decreased after a low-acceptance window.
    pub draft_budget_reductions: usize,
    /// Number of times adaptive draft budget increased after a full-acceptance window.
    pub draft_budget_increases: usize,
    /// Microseconds spent in MTP draft hidden forward passes.
    pub draft_forward_us: u64,
    /// Microseconds spent in main-model verify and fallback replay hidden forwards.
    pub verify_forward_us: u64,
    /// Microseconds spent projecting hidden states to logits.
    pub projection_us: u64,
    /// Microseconds spent sampling logits.
    pub sampling_us: u64,
    /// Host synchronizations performed while constructing neural draft chains.
    pub draft_host_sync_count: usize,
    /// Microseconds blocked on host synchronization while constructing draft chains.
    pub draft_host_sync_us: u64,
    /// Host synchronizations performed to resolve a verified speculative window.
    pub verify_accept_host_sync_count: usize,
    /// Microseconds blocked on the compact verify-acceptance result.
    pub verify_accept_host_sync_us: u64,
    /// Microseconds spent trimming, restoring, or replaying main KV after mismatch.
    pub main_rollback_us: u64,
    /// Microseconds spent committing accepted tokens into the MTP KV cache.
    pub mtp_cache_commit_us: u64,
    /// Microseconds spent building MTP KV cache entries during prompt prefill.
    pub mtp_prefill_cache_commit_us: u64,
    /// Microseconds spent committing accepted decode tokens into the MTP KV cache.
    pub mtp_decode_cache_commit_us: u64,
    /// Microseconds spent restoring the MTP KV cache after temporary draft.
    pub mtp_cache_restore_us: u64,
    /// Bounded, regime-level observations used only by offline draft-cap calibration.
    pub draft_cap_observations: Vec<MtpDraftCapObservation>,
    /// Windows omitted after the bounded observation table reached capacity.
    pub draft_cap_observation_dropped_windows: usize,
}

impl MtpSpeculativeStats {
    /// Speculative windows that attempted at least two draft tokens.
    pub fn multi_token_windows(&self) -> usize {
        self.draft_attempts_by_position
            .get(1)
            .copied()
            .unwrap_or_default()
    }

    pub(crate) fn draft_cap_timing(&self) -> MtpDraftCapTiming {
        MtpDraftCapTiming {
            draft_forward_us: self.draft_forward_us,
            verify_forward_us: self.verify_forward_us,
            projection_us: self.projection_us,
            sampling_us: self.sampling_us,
            main_rollback_us: self.main_rollback_us,
            decode_cache_commit_us: self.mtp_decode_cache_commit_us,
            cache_restore_us: self.mtp_cache_restore_us,
        }
    }

    pub fn saturating_delta_since(&self, before: &Self) -> Self {
        fn vec_delta(current: &[usize], before: &[usize]) -> Vec<usize> {
            let len = current.len().max(before.len());
            (0..len)
                .map(|idx| {
                    current
                        .get(idx)
                        .copied()
                        .unwrap_or_default()
                        .saturating_sub(before.get(idx).copied().unwrap_or_default())
                })
                .collect()
        }

        let draft_cap_observations = self
            .draft_cap_observations
            .iter()
            .map(|current| {
                let before = before
                    .draft_cap_observations
                    .iter()
                    .find(|value| current.same_regime(value));
                current.saturating_delta_since(before)
            })
            .filter(|value| value.windows > 0)
            .collect();

        Self {
            windows: self.windows.saturating_sub(before.windows),
            drafted_tokens: self.drafted_tokens.saturating_sub(before.drafted_tokens),
            accepted_draft_tokens: self
                .accepted_draft_tokens
                .saturating_sub(before.accepted_draft_tokens),
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
            draft_attempts_by_position: vec_delta(
                &self.draft_attempts_by_position,
                &before.draft_attempts_by_position,
            ),
            draft_accepts_by_position: vec_delta(
                &self.draft_accepts_by_position,
                &before.draft_accepts_by_position,
            ),
            rollback_count: self.rollback_count.saturating_sub(before.rollback_count),
            mtp_cache_reuse_count: self
                .mtp_cache_reuse_count
                .saturating_sub(before.mtp_cache_reuse_count),
            mtp_cache_reused_tokens: self
                .mtp_cache_reused_tokens
                .saturating_sub(before.mtp_cache_reused_tokens),
            draft_budget_reductions: self
                .draft_budget_reductions
                .saturating_sub(before.draft_budget_reductions),
            draft_budget_increases: self
                .draft_budget_increases
                .saturating_sub(before.draft_budget_increases),
            draft_forward_us: self
                .draft_forward_us
                .saturating_sub(before.draft_forward_us),
            verify_forward_us: self
                .verify_forward_us
                .saturating_sub(before.verify_forward_us),
            projection_us: self.projection_us.saturating_sub(before.projection_us),
            sampling_us: self.sampling_us.saturating_sub(before.sampling_us),
            draft_host_sync_count: self
                .draft_host_sync_count
                .saturating_sub(before.draft_host_sync_count),
            draft_host_sync_us: self
                .draft_host_sync_us
                .saturating_sub(before.draft_host_sync_us),
            verify_accept_host_sync_count: self
                .verify_accept_host_sync_count
                .saturating_sub(before.verify_accept_host_sync_count),
            verify_accept_host_sync_us: self
                .verify_accept_host_sync_us
                .saturating_sub(before.verify_accept_host_sync_us),
            main_rollback_us: self
                .main_rollback_us
                .saturating_sub(before.main_rollback_us),
            mtp_cache_commit_us: self
                .mtp_cache_commit_us
                .saturating_sub(before.mtp_cache_commit_us),
            mtp_prefill_cache_commit_us: self
                .mtp_prefill_cache_commit_us
                .saturating_sub(before.mtp_prefill_cache_commit_us),
            mtp_decode_cache_commit_us: self
                .mtp_decode_cache_commit_us
                .saturating_sub(before.mtp_decode_cache_commit_us),
            mtp_cache_restore_us: self
                .mtp_cache_restore_us
                .saturating_sub(before.mtp_cache_restore_us),
            draft_cap_observations,
            draft_cap_observation_dropped_windows: self
                .draft_cap_observation_dropped_windows
                .saturating_sub(before.draft_cap_observation_dropped_windows),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_draft_cap_observation(
        &mut self,
        configured_max_draft_tokens: usize,
        draft_tokens_by_row: &[usize],
        context_tokens_by_row: &[usize],
        accepted_draft_tokens: usize,
        committed_tokens: usize,
        rollback_count: usize,
        total_us: u64,
        timing_delta: MtpDraftCapTiming,
    ) {
        if draft_tokens_by_row.is_empty()
            || draft_tokens_by_row.len() != context_tokens_by_row.len()
        {
            return;
        }
        let min_draft_tokens = draft_tokens_by_row.iter().copied().min().unwrap_or(0);
        let max_draft_tokens = draft_tokens_by_row.iter().copied().max().unwrap_or(0);
        if min_draft_tokens == 0 {
            return;
        }
        let first_context_bucket = MtpDraftCapContextBucket::for_tokens(context_tokens_by_row[0]);
        let mixed_context_buckets = context_tokens_by_row
            .iter()
            .copied()
            .map(MtpDraftCapContextBucket::for_tokens)
            .any(|bucket| bucket != first_context_bucket);
        let observation = MtpDraftCapObservation {
            configured_max_draft_tokens,
            min_draft_tokens,
            max_draft_tokens,
            batch_width: draft_tokens_by_row.len(),
            context_bucket: context_tokens_by_row
                .iter()
                .copied()
                .map(MtpDraftCapContextBucket::for_tokens)
                .max()
                .unwrap_or(first_context_bucket),
            mixed_context_buckets,
            windows: draft_tokens_by_row.len(),
            drafted_tokens: draft_tokens_by_row.iter().copied().sum(),
            accepted_draft_tokens,
            committed_tokens,
            rollback_count,
            total_us,
            draft_forward_us: timing_delta.draft_forward_us,
            verify_forward_us: timing_delta.verify_forward_us,
            projection_us: timing_delta.projection_us,
            sampling_us: timing_delta.sampling_us,
            main_rollback_us: timing_delta.main_rollback_us,
            decode_cache_commit_us: timing_delta.decode_cache_commit_us,
            cache_restore_us: timing_delta.cache_restore_us,
        };
        if let Some(current) = self
            .draft_cap_observations
            .iter_mut()
            .find(|value| value.same_regime(&observation))
        {
            current.add_assign(&observation);
        } else if self.draft_cap_observations.len() < MAX_DRAFT_CAP_OBSERVATION_REGIMES {
            self.draft_cap_observations.push(observation);
        } else {
            self.draft_cap_observation_dropped_windows = self
                .draft_cap_observation_dropped_windows
                .saturating_add(observation.windows);
        }
    }

    pub(crate) fn merge_from(&mut self, other: Self) {
        self.windows = self.windows.saturating_add(other.windows);
        self.drafted_tokens = self.drafted_tokens.saturating_add(other.drafted_tokens);
        self.accepted_draft_tokens = self
            .accepted_draft_tokens
            .saturating_add(other.accepted_draft_tokens);
        self.exact_sampling_windows = self
            .exact_sampling_windows
            .saturating_add(other.exact_sampling_windows);
        self.exact_acceptance_draws = self
            .exact_acceptance_draws
            .saturating_add(other.exact_acceptance_draws);
        self.exact_residual_corrections = self
            .exact_residual_corrections
            .saturating_add(other.exact_residual_corrections);
        self.exact_bonus_samples = self
            .exact_bonus_samples
            .saturating_add(other.exact_bonus_samples);
        merge_counter_vec(
            &mut self.draft_attempts_by_position,
            other.draft_attempts_by_position,
        );
        merge_counter_vec(
            &mut self.draft_accepts_by_position,
            other.draft_accepts_by_position,
        );
        self.rollback_count = self.rollback_count.saturating_add(other.rollback_count);
        self.mtp_cache_reuse_count = self
            .mtp_cache_reuse_count
            .saturating_add(other.mtp_cache_reuse_count);
        self.mtp_cache_reused_tokens = self
            .mtp_cache_reused_tokens
            .saturating_add(other.mtp_cache_reused_tokens);
        self.draft_budget_reductions = self
            .draft_budget_reductions
            .saturating_add(other.draft_budget_reductions);
        self.draft_budget_increases = self
            .draft_budget_increases
            .saturating_add(other.draft_budget_increases);
        self.draft_forward_us = self.draft_forward_us.saturating_add(other.draft_forward_us);
        self.verify_forward_us = self
            .verify_forward_us
            .saturating_add(other.verify_forward_us);
        self.projection_us = self.projection_us.saturating_add(other.projection_us);
        self.sampling_us = self.sampling_us.saturating_add(other.sampling_us);
        self.draft_host_sync_count = self
            .draft_host_sync_count
            .saturating_add(other.draft_host_sync_count);
        self.draft_host_sync_us = self
            .draft_host_sync_us
            .saturating_add(other.draft_host_sync_us);
        self.verify_accept_host_sync_count = self
            .verify_accept_host_sync_count
            .saturating_add(other.verify_accept_host_sync_count);
        self.verify_accept_host_sync_us = self
            .verify_accept_host_sync_us
            .saturating_add(other.verify_accept_host_sync_us);
        self.main_rollback_us = self.main_rollback_us.saturating_add(other.main_rollback_us);
        self.mtp_cache_commit_us = self
            .mtp_cache_commit_us
            .saturating_add(other.mtp_cache_commit_us);
        self.mtp_prefill_cache_commit_us = self
            .mtp_prefill_cache_commit_us
            .saturating_add(other.mtp_prefill_cache_commit_us);
        self.mtp_decode_cache_commit_us = self
            .mtp_decode_cache_commit_us
            .saturating_add(other.mtp_decode_cache_commit_us);
        self.mtp_cache_restore_us = self
            .mtp_cache_restore_us
            .saturating_add(other.mtp_cache_restore_us);
        for observation in other.draft_cap_observations {
            if let Some(current) = self
                .draft_cap_observations
                .iter_mut()
                .find(|value| value.same_regime(&observation))
            {
                current.add_assign(&observation);
            } else if self.draft_cap_observations.len() < MAX_DRAFT_CAP_OBSERVATION_REGIMES {
                self.draft_cap_observations.push(observation);
            } else {
                self.draft_cap_observation_dropped_windows = self
                    .draft_cap_observation_dropped_windows
                    .saturating_add(observation.windows);
            }
        }
        self.draft_cap_observation_dropped_windows = self
            .draft_cap_observation_dropped_windows
            .saturating_add(other.draft_cap_observation_dropped_windows);
    }

    pub fn record_window_acceptance(
        &mut self,
        attempted_draft_tokens: usize,
        accepted_draft_tokens: usize,
    ) {
        if attempted_draft_tokens == 0 {
            return;
        }
        let accepted = accepted_draft_tokens.min(attempted_draft_tokens);
        if self.draft_attempts_by_position.len() < attempted_draft_tokens {
            self.draft_attempts_by_position
                .resize(attempted_draft_tokens, 0);
            self.draft_accepts_by_position
                .resize(attempted_draft_tokens, 0);
        }
        for idx in 0..attempted_draft_tokens {
            self.draft_attempts_by_position[idx] =
                self.draft_attempts_by_position[idx].saturating_add(1);
            if idx < accepted {
                self.draft_accepts_by_position[idx] =
                    self.draft_accepts_by_position[idx].saturating_add(1);
            }
        }
    }

    pub(crate) fn record_exact_sampling(&mut self, counters: ExactSamplingCounters) {
        self.exact_sampling_windows = self.exact_sampling_windows.saturating_add(counters.windows);
        self.exact_acceptance_draws = self
            .exact_acceptance_draws
            .saturating_add(counters.acceptance_draws);
        self.exact_residual_corrections = self
            .exact_residual_corrections
            .saturating_add(counters.residual_corrections);
        self.exact_bonus_samples = self
            .exact_bonus_samples
            .saturating_add(counters.bonus_samples);
    }
}

impl MtpDraftCapTiming {
    pub(crate) fn saturating_delta_since(self, before: Self) -> Self {
        Self {
            draft_forward_us: self
                .draft_forward_us
                .saturating_sub(before.draft_forward_us),
            verify_forward_us: self
                .verify_forward_us
                .saturating_sub(before.verify_forward_us),
            projection_us: self.projection_us.saturating_sub(before.projection_us),
            sampling_us: self.sampling_us.saturating_sub(before.sampling_us),
            main_rollback_us: self
                .main_rollback_us
                .saturating_sub(before.main_rollback_us),
            decode_cache_commit_us: self
                .decode_cache_commit_us
                .saturating_sub(before.decode_cache_commit_us),
            cache_restore_us: self
                .cache_restore_us
                .saturating_sub(before.cache_restore_us),
        }
    }
}

fn merge_counter_vec(dst: &mut Vec<usize>, src: Vec<usize>) {
    if dst.len() < src.len() {
        dst.resize(src.len(), 0);
    }
    for (idx, value) in src.into_iter().enumerate() {
        dst[idx] = dst[idx].saturating_add(value);
    }
}

pub use super::speculative_model::MtpSpeculativeModel;

pub(crate) fn elapsed_us_since(start: Instant) -> u64 {
    start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
}

pub(crate) fn add_elapsed_us(counter: &mut u64, start: Instant) {
    *counter = counter.saturating_add(elapsed_us_since(start));
}

pub(crate) fn add_mtp_prefill_cache_commit_us(stats: &mut MtpSpeculativeStats, start: Instant) {
    let elapsed = elapsed_us_since(start);
    stats.mtp_cache_commit_us = stats.mtp_cache_commit_us.saturating_add(elapsed);
    stats.mtp_prefill_cache_commit_us = stats.mtp_prefill_cache_commit_us.saturating_add(elapsed);
}

pub(crate) fn add_mtp_decode_cache_commit_us(stats: &mut MtpSpeculativeStats, start: Instant) {
    let elapsed = elapsed_us_since(start);
    stats.mtp_cache_commit_us = stats.mtp_cache_commit_us.saturating_add(elapsed);
    stats.mtp_decode_cache_commit_us = stats.mtp_decode_cache_commit_us.saturating_add(elapsed);
}

#[derive(Debug)]
pub(crate) struct MtpDraftResult {
    pub tokens: Vec<u32>,
    pub distributions: Vec<DraftTokenDistribution>,
    pub cache_snapshot: MtpCacheSnapshot,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MtpDraftPolicyWindow {
    pub attempted_draft_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub committed_tokens: usize,
    pub total_us: u64,
    pub context_tokens: usize,
    pub batch_width: usize,
    pub kv_state: MtpDraftPolicyKvState,
    pub draft_forward_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub sampling_us: u64,
    pub verify_accept_host_sync_us: u64,
    pub main_rollback_us: u64,
    pub mtp_cache_commit_us: u64,
    pub mtp_prefill_cache_commit_us: u64,
    pub mtp_decode_cache_commit_us: u64,
    pub mtp_cache_restore_us: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(crate) enum MtpDraftPolicyKvState {
    #[default]
    Contiguous,
    Paged,
    PagedActiveKv,
}

impl MtpDraftPolicyKvState {
    pub(crate) fn from_runtime(paged: bool, active_kv: bool) -> Self {
        if active_kv {
            Self::PagedActiveKv
        } else if paged {
            Self::Paged
        } else {
            Self::Contiguous
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MtpDraftPolicyRegime {
    context_bucket: MtpDraftCapContextBucket,
    batch_width: usize,
    kv_state: MtpDraftPolicyKvState,
}

impl MtpDraftPolicyWindow {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_stats_delta(
        attempted_draft_tokens: usize,
        accepted_draft_tokens: usize,
        committed_tokens: usize,
        total_us: u64,
        context_tokens: usize,
        batch_width: usize,
        kv_state: MtpDraftPolicyKvState,
        delta: &MtpSpeculativeStats,
    ) -> Self {
        Self {
            attempted_draft_tokens,
            accepted_draft_tokens,
            committed_tokens,
            total_us,
            context_tokens,
            batch_width,
            kv_state,
            draft_forward_us: delta.draft_forward_us,
            verify_forward_us: delta.verify_forward_us,
            projection_us: delta.projection_us,
            sampling_us: delta.sampling_us,
            verify_accept_host_sync_us: delta.verify_accept_host_sync_us,
            main_rollback_us: delta.main_rollback_us,
            mtp_cache_commit_us: delta.mtp_cache_commit_us,
            mtp_prefill_cache_commit_us: delta.mtp_prefill_cache_commit_us,
            mtp_decode_cache_commit_us: delta.mtp_decode_cache_commit_us,
            mtp_cache_restore_us: delta.mtp_cache_restore_us,
        }
    }

    fn measured_components_us(self) -> u64 {
        self.draft_forward_us
            .saturating_add(self.verify_forward_us)
            .saturating_add(self.projection_us)
            .saturating_add(self.sampling_us)
            .saturating_add(self.verify_accept_host_sync_us)
            .saturating_add(self.main_rollback_us)
            .saturating_add(self.mtp_decode_cache_commit_us)
            .saturating_add(self.mtp_cache_restore_us)
    }

    fn gemma4_cost_per_committed_token_us(self) -> f64 {
        let measured_components_us = self.measured_components_us();
        let comparable_us = if self.attempted_draft_tokens == 0 && measured_components_us > 0 {
            // A zero-draft control window still runs inside speculative
            // bookkeeping so the drafter cache can be resumed if it wins.
            // Snapshot/resolve/state-maintenance overhead disappears after a
            // permanent switch to the ordinary scheduler and must not make
            // that control path look artificially expensive.
            measured_components_us
        } else {
            self.total_us.max(measured_components_us)
        };
        comparable_us as f64 / self.committed_tokens.max(1) as f64
    }

    fn qwen_cost_per_committed_token_us(self) -> f64 {
        self.total_us.max(self.measured_components_us()) as f64
            / self.committed_tokens.max(1) as f64
    }

    fn regime(self) -> MtpDraftPolicyRegime {
        MtpDraftPolicyRegime {
            context_bucket: MtpDraftCapContextBucket::for_tokens(self.context_tokens),
            batch_width: self.batch_width.max(1),
            kv_state: self.kv_state,
        }
    }

    fn acceptance_rate(self) -> f64 {
        if self.attempted_draft_tokens == 0 {
            1.0
        } else {
            self.accepted_draft_tokens.min(self.attempted_draft_tokens) as f64
                / self.attempted_draft_tokens as f64
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MtpDraftBudgetChange {
    pub reduced: bool,
    pub increased: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Gemma4DrafterPolicyState {
    max_draft_tokens: usize,
    current_budget: usize,
    acceptance_ewma: Option<f64>,
    active_regime: Option<MtpDraftPolicyRegime>,
    cost_estimates: Vec<MtpDraftCostEstimate>,
    probe_budget: Option<usize>,
    probe_origin_budget: Option<usize>,
    probe_windows_remaining: usize,
    cooldown_windows: usize,
    next_probe_cooldown_windows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct Gemma4DrafterPolicySnapshot {
    max_draft_tokens: usize,
    current_budget: usize,
    acceptance_ewma_bits: Option<u64>,
    active_regime: Option<MtpDraftPolicyRegime>,
    cost_estimates: Vec<MtpDraftCostEstimateSnapshot>,
    probe_budget: Option<usize>,
    probe_origin_budget: Option<usize>,
    probe_windows_remaining: usize,
    cooldown_windows: usize,
    next_probe_cooldown_windows: usize,
}

#[derive(Debug, Clone)]
struct MtpDraftCostEstimate {
    regime: MtpDraftPolicyRegime,
    draft_tokens: usize,
    cost_ewma: f64,
    acceptance_ewma: f64,
    samples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MtpDraftCostEstimateSnapshot {
    regime: MtpDraftPolicyRegime,
    draft_tokens: usize,
    cost_ewma_bits: u64,
    acceptance_ewma_bits: u64,
    samples: usize,
}

impl Gemma4DrafterPolicyState {
    const EWMA_ALPHA: f64 = 0.35;
    const LOW_ACCEPTANCE: f64 = 0.50;
    const HIGH_ACCEPTANCE: f64 = 0.85;
    const MIN_COST_SAMPLES: usize = 2;
    const PROBE_WINDOWS: usize = 2;
    const ZERO_DRAFT_MIN_COST_SAMPLES: usize = 8;
    // At >32K, waiting for eight single-draft windows before measuring
    // ordinary decode allows a costly path to consume most of a typical
    // response. Two complete MTP windows already include target, draft,
    // sampling, and cache costs; the separate four-window ordinary probe
    // remains the noise filter for the final decision.
    const LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES: usize = 2;
    const ZERO_DRAFT_PROBE_WINDOWS: usize = 4;
    const INITIAL_PROBE_COOLDOWN_WINDOWS: usize = 8;
    const MAX_PROBE_COOLDOWN_WINDOWS: usize = 64;
    const COST_IMPROVEMENT_RATIO: f64 = 0.95;
    // Ordinary decode is the safe control path. Require only a small measured
    // margin before bypassing MTP so a 5-10% speculative regression is not
    // hidden by overly conservative hysteresis. The four-window probe and
    // EWMA still absorb single-window timing noise.
    const ZERO_DRAFT_COST_IMPROVEMENT_RATIO: f64 = 0.98;

    pub(crate) fn new(max_draft_tokens: usize) -> Self {
        let max_draft_tokens = max_draft_tokens.max(1);
        Self {
            max_draft_tokens,
            current_budget: max_draft_tokens,
            acceptance_ewma: None,
            active_regime: None,
            cost_estimates: Vec::new(),
            probe_budget: None,
            probe_origin_budget: None,
            probe_windows_remaining: 0,
            cooldown_windows: 0,
            next_probe_cooldown_windows: (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2)
                .min(Self::MAX_PROBE_COOLDOWN_WINDOWS),
        }
    }

    pub(crate) fn current_budget(&self) -> usize {
        self.current_budget.min(self.max_draft_tokens)
    }

    pub(crate) fn seed_initial_budget(&mut self, budget: usize) -> bool {
        if self.active_regime.is_some()
            || !self.cost_estimates.is_empty()
            || self.probe_budget.is_some()
        {
            return false;
        }
        let seeded = budget.clamp(1, self.max_draft_tokens);
        let changed = seeded != self.current_budget;
        self.current_budget = seeded;
        changed
    }

    #[cfg(test)]
    pub(crate) fn should_maintain_mtp_cache(&self) -> bool {
        self.current_budget() > 0 || self.probe_budget.is_some()
    }

    pub(crate) fn uses_ordinary_decode(&self) -> bool {
        self.current_budget() == 0 && self.probe_budget.is_none()
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> Gemma4DrafterPolicySnapshot {
        Gemma4DrafterPolicySnapshot {
            max_draft_tokens: self.max_draft_tokens,
            current_budget: self.current_budget,
            acceptance_ewma_bits: self.acceptance_ewma.map(f64::to_bits),
            active_regime: self.active_regime,
            cost_estimates: self
                .cost_estimates
                .iter()
                .map(|estimate| MtpDraftCostEstimateSnapshot {
                    regime: estimate.regime,
                    draft_tokens: estimate.draft_tokens,
                    cost_ewma_bits: estimate.cost_ewma.to_bits(),
                    acceptance_ewma_bits: estimate.acceptance_ewma.to_bits(),
                    samples: estimate.samples,
                })
                .collect(),
            probe_budget: self.probe_budget,
            probe_origin_budget: self.probe_origin_budget,
            probe_windows_remaining: self.probe_windows_remaining,
            cooldown_windows: self.cooldown_windows,
            next_probe_cooldown_windows: self.next_probe_cooldown_windows,
        }
    }

    #[cfg(test)]
    pub(crate) fn restore_snapshot(&mut self, snapshot: Gemma4DrafterPolicySnapshot) -> Result<()> {
        anyhow::ensure!(
            snapshot.max_draft_tokens == self.max_draft_tokens,
            "MTP draft policy snapshot max {} != destination max {}",
            snapshot.max_draft_tokens,
            self.max_draft_tokens
        );
        anyhow::ensure!(
            snapshot.current_budget <= snapshot.max_draft_tokens,
            "MTP draft policy snapshot budget {} is outside [0, {}]",
            snapshot.current_budget,
            snapshot.max_draft_tokens
        );
        self.current_budget = snapshot.current_budget;
        self.acceptance_ewma = snapshot.acceptance_ewma_bits.map(f64::from_bits);
        self.active_regime = snapshot.active_regime;
        self.cost_estimates = snapshot
            .cost_estimates
            .into_iter()
            .map(|estimate| MtpDraftCostEstimate {
                regime: estimate.regime,
                draft_tokens: estimate.draft_tokens,
                cost_ewma: f64::from_bits(estimate.cost_ewma_bits),
                acceptance_ewma: f64::from_bits(estimate.acceptance_ewma_bits),
                samples: estimate.samples,
            })
            .collect();
        self.probe_budget = snapshot.probe_budget;
        self.probe_origin_budget = snapshot.probe_origin_budget;
        self.probe_windows_remaining = snapshot.probe_windows_remaining;
        self.cooldown_windows = snapshot.cooldown_windows;
        self.next_probe_cooldown_windows = snapshot.next_probe_cooldown_windows;
        Ok(())
    }

    pub(crate) fn observe_window(&mut self, window: MtpDraftPolicyWindow) -> MtpDraftBudgetChange {
        let regime = window.regime();
        if self.active_regime != Some(regime) {
            self.active_regime = Some(regime);
            self.acceptance_ewma = None;
            self.cost_estimates.clear();
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            self.cooldown_windows = 0;
            self.next_probe_cooldown_windows =
                (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2).min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
        }
        let old = self.current_budget();

        let acceptance = window.acceptance_rate();
        self.acceptance_ewma = Some(update_ewma(
            self.acceptance_ewma,
            acceptance,
            Self::EWMA_ALPHA,
        ));
        self.record_cost(
            regime,
            window.attempted_draft_tokens,
            window.gemma4_cost_per_committed_token_us(),
            acceptance,
        );
        if window.attempted_draft_tokens != old {
            return MtpDraftBudgetChange::default();
        }

        let zero_draft_probe_min_samples =
            if self.acceptance_ewma.unwrap_or(acceptance) < Self::LOW_ACCEPTANCE {
                Self::MIN_COST_SAMPLES
            } else if matches!(
                regime.context_bucket,
                MtpDraftCapContextBucket::UpTo128k | MtpDraftCapContextBucket::Above128k
            ) {
                Self::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES
            } else {
                Self::ZERO_DRAFT_MIN_COST_SAMPLES
            };
        if old == 1
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_none()
            && self
                .cost_estimate(regime, old)
                .is_some_and(|estimate| estimate.samples >= zero_draft_probe_min_samples)
        {
            self.current_budget = 0;
            if self.acceptance_ewma.unwrap_or(acceptance) < Self::LOW_ACCEPTANCE {
                // At persistently low acceptance, a one-token drafter cannot
                // amortize target verification. Commit directly to ordinary
                // decode so architecture-specific schedulers can leave the
                // speculative path instead of benchmarking a slower
                // zero-draft emulation of it.
                self.probe_budget = None;
                self.probe_origin_budget = None;
                self.probe_windows_remaining = 0;
            } else {
                self.probe_budget = Some(0);
                self.probe_origin_budget = Some(old);
                self.probe_windows_remaining = Self::ZERO_DRAFT_PROBE_WINDOWS;
            }
            return budget_change(old, self.current_budget);
        }
        if old == 0 && self.probe_budget.is_none() {
            return MtpDraftBudgetChange::default();
        }
        if old == 1
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_some()
            && self.cooldown_windows == 0
            && self
                .cost_estimate(regime, 1)
                .is_some_and(|estimate| estimate.samples >= Self::ZERO_DRAFT_MIN_COST_SAMPLES)
        {
            // A previous zero-draft sample is stale once the response enters
            // a different acceptance/cost phase. Re-test ordinary decode only
            // while the MTP cache is still synchronized, and discard the old
            // control estimate so it cannot drive an uncontrolled switch.
            self.cost_estimates
                .retain(|estimate| estimate.regime != regime || estimate.draft_tokens != 0);
            self.current_budget = 0;
            self.probe_budget = Some(0);
            self.probe_origin_budget = Some(old);
            self.probe_windows_remaining = Self::ZERO_DRAFT_PROBE_WINDOWS;
            return budget_change(old, self.current_budget);
        }

        let mut next = old;
        let full_accept = window.accepted_draft_tokens == window.attempted_draft_tokens;
        if !full_accept {
            let rejected_probe = self.probe_budget == Some(old);
            next = window.accepted_draft_tokens.saturating_add(1).min(old);
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            if rejected_probe {
                self.back_off_next_probe();
            } else if old == 1 {
                self.cooldown_windows = self.cooldown_windows.saturating_sub(1);
            } else {
                self.arm_initial_probe_cooldown();
            }
        } else if self.probe_budget == Some(old) {
            self.probe_windows_remaining = self.probe_windows_remaining.saturating_sub(1);
            if self.probe_windows_remaining == 0 {
                let origin = self.probe_origin_budget.take();
                self.probe_budget = None;
                next = origin.map_or(old, |origin| {
                    self.preferred_probe_budget(regime, origin, old)
                });
                if origin.is_some_and(|origin| next == origin) && origin != Some(0) {
                    self.back_off_next_probe();
                } else {
                    self.arm_initial_probe_cooldown();
                }
            }
        } else {
            next = self.best_measured_budget(regime, old);
            if next > old && self.acceptance_ewma.unwrap_or(acceptance) < Self::HIGH_ACCEPTANCE {
                next = old;
            }
            if next == old {
                if self.cooldown_windows > 0 {
                    self.cooldown_windows -= 1;
                } else if self
                    .cost_estimate(regime, old)
                    .is_some_and(|estimate| estimate.samples >= Self::MIN_COST_SAMPLES)
                {
                    if let Some(probe) = self.next_probe_budget(regime, old) {
                        if probe < old
                            || self.acceptance_ewma.unwrap_or(acceptance) >= Self::HIGH_ACCEPTANCE
                        {
                            next = probe;
                            self.probe_budget = Some(probe);
                            self.probe_origin_budget = Some(old);
                            self.probe_windows_remaining = Self::PROBE_WINDOWS;
                        }
                    }
                }
            }
        }

        let smoothed_acceptance = self.acceptance_ewma.unwrap_or(acceptance);
        if smoothed_acceptance < Self::LOW_ACCEPTANCE && old > 1 {
            next = next.min(old - 1);
        }
        if acceptance == 0.0 {
            next = next.min(1);
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            // A rejected wider probe should back off before it is retried.
            // At budget 1, however, repeatedly re-arming this cooldown makes
            // an unprofitable MTP path impossible to compare with ordinary
            // decode: every zero-acceptance window resets the countdown. Keep
            // the existing countdown moving toward a fresh zero-draft probe.
            if old > 1 {
                self.arm_initial_probe_cooldown();
            }
        }

        self.current_budget = next.min(self.max_draft_tokens);
        budget_change(old, self.current_budget)
    }

    fn arm_initial_probe_cooldown(&mut self) {
        self.cooldown_windows = Self::INITIAL_PROBE_COOLDOWN_WINDOWS;
        self.next_probe_cooldown_windows =
            (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2).min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
    }

    fn back_off_next_probe(&mut self) {
        self.cooldown_windows = self.next_probe_cooldown_windows.clamp(
            Self::INITIAL_PROBE_COOLDOWN_WINDOWS,
            Self::MAX_PROBE_COOLDOWN_WINDOWS,
        );
        self.next_probe_cooldown_windows = self
            .cooldown_windows
            .saturating_mul(2)
            .min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
    }

    fn record_cost(
        &mut self,
        regime: MtpDraftPolicyRegime,
        draft_tokens: usize,
        cost: f64,
        acceptance: f64,
    ) {
        if let Some(estimate) = self
            .cost_estimates
            .iter_mut()
            .find(|estimate| estimate.regime == regime && estimate.draft_tokens == draft_tokens)
        {
            if estimate.samples == 0 {
                estimate.cost_ewma = cost;
                estimate.acceptance_ewma = acceptance;
            } else {
                estimate.cost_ewma = update_ewma(Some(estimate.cost_ewma), cost, Self::EWMA_ALPHA);
                estimate.acceptance_ewma =
                    update_ewma(Some(estimate.acceptance_ewma), acceptance, Self::EWMA_ALPHA);
            }
            estimate.samples = estimate.samples.saturating_add(1);
        } else {
            // Draft=0 activates the ordinary Q=1 target shape for the first
            // time in this response. That first control window can include a
            // one-off Metal graph compilation which has already been paid by
            // the time the policy makes its decision. Keep the marker but do
            // not let cold compilation bias the steady-state comparison.
            let samples = usize::from(draft_tokens != 0);
            self.cost_estimates.push(MtpDraftCostEstimate {
                regime,
                draft_tokens,
                cost_ewma: cost,
                acceptance_ewma: acceptance,
                samples,
            });
        }
    }

    fn cost_estimate(
        &self,
        regime: MtpDraftPolicyRegime,
        draft_tokens: usize,
    ) -> Option<&MtpDraftCostEstimate> {
        self.cost_estimates
            .iter()
            .find(|estimate| estimate.regime == regime && estimate.draft_tokens == draft_tokens)
    }

    fn best_measured_budget(&self, regime: MtpDraftPolicyRegime, current: usize) -> usize {
        let Some(current_estimate) = self.cost_estimate(regime, current) else {
            return current;
        };
        if current_estimate.samples < Self::MIN_COST_SAMPLES {
            return current;
        }
        self.cost_estimates
            .iter()
            .filter(|estimate| {
                estimate.regime == regime
                    && estimate.draft_tokens > 0
                    && estimate.samples >= Self::MIN_COST_SAMPLES
                    && (estimate.draft_tokens <= current
                        || estimate.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
                    && estimate.cost_ewma
                        < current_estimate.cost_ewma * Self::COST_IMPROVEMENT_RATIO
            })
            .min_by(|left, right| left.cost_ewma.total_cmp(&right.cost_ewma))
            .map_or(current, |estimate| estimate.draft_tokens)
    }

    fn next_probe_budget(&self, regime: MtpDraftPolicyRegime, current: usize) -> Option<usize> {
        let lower = current.checked_sub(1);
        let upper = current
            .checked_add(1)
            .filter(|&budget| budget <= self.max_draft_tokens);
        [lower, upper]
            .into_iter()
            .flatten()
            .filter(|&budget| {
                if budget == 0 && current > 0 {
                    // Wait for a controlled zero-draft probe. Once sampled,
                    // refreshes are scheduled explicitly after cooldown rather
                    // than by the generic adjacent-budget probe path.
                    if self.cost_estimate(regime, 0).is_some()
                        || self.cost_estimate(regime, current).is_none_or(|estimate| {
                            estimate.samples < Self::ZERO_DRAFT_MIN_COST_SAMPLES
                        })
                    {
                        return false;
                    }
                }
                budget <= current
                    || current == 0
                    || self
                        .cost_estimate(regime, budget)
                        .is_none_or(|estimate| estimate.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
            })
            .min_by_key(|&budget| {
                self.cost_estimate(regime, budget)
                    .map_or(0, |estimate| estimate.samples)
            })
    }

    fn preferred_probe_budget(
        &self,
        regime: MtpDraftPolicyRegime,
        origin: usize,
        probe: usize,
    ) -> usize {
        let Some(origin_cost) = self.cost_estimate(regime, origin) else {
            return probe;
        };
        let Some(probe_cost) = self.cost_estimate(regime, probe) else {
            return origin;
        };
        let improvement_ratio = if probe == 0 {
            Self::ZERO_DRAFT_COST_IMPROVEMENT_RATIO
        } else {
            Self::COST_IMPROVEMENT_RATIO
        };
        if probe_cost.samples >= Self::MIN_COST_SAMPLES
            && (probe <= origin
                || origin == 0
                || probe_cost.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
            && probe_cost.cost_ewma < origin_cost.cost_ewma * improvement_ratio
        {
            probe
        } else {
            origin
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct QwenMtpDraftPolicyState {
    max_draft_tokens: usize,
    current_budget: usize,
    acceptance_ewma: Option<f64>,
    active_regime: Option<MtpDraftPolicyRegime>,
    cost_estimates: Vec<MtpDraftCostEstimate>,
    probe_budget: Option<usize>,
    probe_origin_budget: Option<usize>,
    probe_windows_remaining: usize,
    long_context_mtp_warmup_windows_remaining: usize,
    cooldown_windows: usize,
    next_probe_cooldown_windows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QwenMtpDraftPolicySnapshot {
    max_draft_tokens: usize,
    current_budget: usize,
    acceptance_ewma_bits: Option<u64>,
    active_regime: Option<MtpDraftPolicyRegime>,
    cost_estimates: Vec<MtpDraftCostEstimateSnapshot>,
    probe_budget: Option<usize>,
    probe_origin_budget: Option<usize>,
    probe_windows_remaining: usize,
    long_context_mtp_warmup_windows_remaining: usize,
    cooldown_windows: usize,
    next_probe_cooldown_windows: usize,
}

impl QwenMtpDraftPolicyState {
    const EWMA_ALPHA: f64 = 0.35;
    const LOW_ACCEPTANCE: f64 = 0.50;
    const HIGH_ACCEPTANCE: f64 = 0.85;
    const MIN_COST_SAMPLES: usize = 2;
    const PROBE_WINDOWS: usize = 2;
    const ZERO_DRAFT_MIN_COST_SAMPLES: usize = 8;
    // The first two MTP windows after entering a 32K+ regime include the
    // post-prefill transition and deferred setup. Omit both before collecting
    // cost samples unless the second window is also partially accepted; two
    // consecutive partial windows are enough evidence to probe ordinary decode.
    const LONG_CONTEXT_MTP_WARMUP_WINDOWS: usize = 2;
    const LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES: usize = 2;
    const ZERO_DRAFT_PROBE_WINDOWS: usize = 4;
    // The first long-context d=0 window resolves deferred work from the MTP
    // window that armed the probe and is not representative of steady-state
    // ordinary decode. Keep the MTP cache synchronized, but omit that transition
    // window from the cost estimate before collecting the four measured controls.
    const LONG_CONTEXT_ZERO_DRAFT_WARMUP_WINDOWS: usize = 1;
    const INITIAL_PROBE_COOLDOWN_WINDOWS: usize = 8;
    const MAX_PROBE_COOLDOWN_WINDOWS: usize = 64;
    const COST_IMPROVEMENT_RATIO: f64 = 0.95;
    // Ordinary decode is the safe control path. Require only a small measured
    // margin before bypassing MTP so a 5-10% speculative regression is not
    // hidden by overly conservative hysteresis. The four-window probe and
    // EWMA still absorb single-window timing noise.
    const ZERO_DRAFT_COST_IMPROVEMENT_RATIO: f64 = 0.98;

    pub(crate) fn new(max_draft_tokens: usize) -> Self {
        let max_draft_tokens = max_draft_tokens.max(1);
        Self {
            max_draft_tokens,
            current_budget: max_draft_tokens,
            acceptance_ewma: None,
            active_regime: None,
            cost_estimates: Vec::new(),
            probe_budget: None,
            probe_origin_budget: None,
            probe_windows_remaining: 0,
            long_context_mtp_warmup_windows_remaining: 0,
            cooldown_windows: 0,
            next_probe_cooldown_windows: (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2)
                .min(Self::MAX_PROBE_COOLDOWN_WINDOWS),
        }
    }

    pub(crate) fn current_budget(&self) -> usize {
        self.current_budget.min(self.max_draft_tokens)
    }

    pub(crate) fn should_maintain_mtp_cache(&self) -> bool {
        self.current_budget() > 0 || self.probe_budget.is_some()
    }

    pub(crate) fn uses_ordinary_decode(&self) -> bool {
        self.current_budget() == 0 && self.probe_budget.is_none()
    }

    pub(crate) fn snapshot(&self) -> QwenMtpDraftPolicySnapshot {
        QwenMtpDraftPolicySnapshot {
            max_draft_tokens: self.max_draft_tokens,
            current_budget: self.current_budget,
            acceptance_ewma_bits: self.acceptance_ewma.map(f64::to_bits),
            active_regime: self.active_regime,
            cost_estimates: self
                .cost_estimates
                .iter()
                .map(|estimate| MtpDraftCostEstimateSnapshot {
                    regime: estimate.regime,
                    draft_tokens: estimate.draft_tokens,
                    cost_ewma_bits: estimate.cost_ewma.to_bits(),
                    acceptance_ewma_bits: estimate.acceptance_ewma.to_bits(),
                    samples: estimate.samples,
                })
                .collect(),
            probe_budget: self.probe_budget,
            probe_origin_budget: self.probe_origin_budget,
            probe_windows_remaining: self.probe_windows_remaining,
            long_context_mtp_warmup_windows_remaining: self
                .long_context_mtp_warmup_windows_remaining,
            cooldown_windows: self.cooldown_windows,
            next_probe_cooldown_windows: self.next_probe_cooldown_windows,
        }
    }

    pub(crate) fn restore_snapshot(&mut self, snapshot: QwenMtpDraftPolicySnapshot) -> Result<()> {
        anyhow::ensure!(
            snapshot.max_draft_tokens == self.max_draft_tokens,
            "MTP draft policy snapshot max {} != destination max {}",
            snapshot.max_draft_tokens,
            self.max_draft_tokens
        );
        anyhow::ensure!(
            snapshot.current_budget <= snapshot.max_draft_tokens,
            "MTP draft policy snapshot budget {} is outside [0, {}]",
            snapshot.current_budget,
            snapshot.max_draft_tokens
        );
        self.current_budget = snapshot.current_budget;
        self.acceptance_ewma = snapshot.acceptance_ewma_bits.map(f64::from_bits);
        self.active_regime = snapshot.active_regime;
        self.cost_estimates = snapshot
            .cost_estimates
            .into_iter()
            .map(|estimate| MtpDraftCostEstimate {
                regime: estimate.regime,
                draft_tokens: estimate.draft_tokens,
                cost_ewma: f64::from_bits(estimate.cost_ewma_bits),
                acceptance_ewma: f64::from_bits(estimate.acceptance_ewma_bits),
                samples: estimate.samples,
            })
            .collect();
        self.probe_budget = snapshot.probe_budget;
        self.probe_origin_budget = snapshot.probe_origin_budget;
        self.probe_windows_remaining = snapshot.probe_windows_remaining;
        self.long_context_mtp_warmup_windows_remaining =
            snapshot.long_context_mtp_warmup_windows_remaining;
        self.cooldown_windows = snapshot.cooldown_windows;
        self.next_probe_cooldown_windows = snapshot.next_probe_cooldown_windows;
        Ok(())
    }

    pub(crate) fn observe_window(&mut self, window: MtpDraftPolicyWindow) -> MtpDraftBudgetChange {
        if qwen_fixed_mtp_draft_depth_is_armed() {
            return MtpDraftBudgetChange::default();
        }
        let regime = window.regime();
        let long_context = matches!(
            regime.context_bucket,
            MtpDraftCapContextBucket::UpTo128k | MtpDraftCapContextBucket::Above128k
        );
        let regime_changed = self.active_regime != Some(regime);
        if regime_changed {
            self.active_regime = Some(regime);
            self.acceptance_ewma = None;
            self.cost_estimates.clear();
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            self.long_context_mtp_warmup_windows_remaining =
                usize::from(long_context) * Self::LONG_CONTEXT_MTP_WARMUP_WINDOWS;
            self.cooldown_windows = 0;
            self.next_probe_cooldown_windows =
                (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2).min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
        }
        let old = self.current_budget();

        let zero_draft_transition_warmup = long_context
            && window.attempted_draft_tokens == 0
            && self.probe_budget == Some(0)
            && self.probe_windows_remaining > Self::ZERO_DRAFT_PROBE_WINDOWS;
        let mtp_transition_window = long_context
            && window.attempted_draft_tokens > 0
            && window.attempted_draft_tokens == old
            && self.probe_budget.is_none()
            && self.long_context_mtp_warmup_windows_remaining > 0;
        let repeated_partial_transition = mtp_transition_window
            && self.long_context_mtp_warmup_windows_remaining == 1
            && window.accepted_draft_tokens < window.attempted_draft_tokens;
        if mtp_transition_window {
            self.long_context_mtp_warmup_windows_remaining -= 1;
        }
        let omit_mtp_transition_sample = mtp_transition_window && !repeated_partial_transition;
        let acceptance = window.acceptance_rate();
        if !zero_draft_transition_warmup && !omit_mtp_transition_sample {
            self.acceptance_ewma = Some(update_ewma(
                self.acceptance_ewma,
                acceptance,
                Self::EWMA_ALPHA,
            ));
            self.record_cost(
                regime,
                window.attempted_draft_tokens,
                window.qwen_cost_per_committed_token_us(),
                acceptance,
            );
        }
        if window.attempted_draft_tokens != old {
            return MtpDraftBudgetChange::default();
        }

        let zero_draft_probe_min_samples = if long_context {
            Self::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES
        } else {
            Self::ZERO_DRAFT_MIN_COST_SAMPLES
        };
        if long_context
            && old > 0
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_none()
            && self.cost_estimate(regime, old).is_some_and(|estimate| {
                let required_samples =
                    if window.accepted_draft_tokens < window.attempted_draft_tokens {
                        1
                    } else {
                        Self::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES
                    };
                estimate.samples >= required_samples
            })
        {
            // At 32K+, first compare the configured production depth directly
            // with ordinary decode. Traversing every adjacent depth before the
            // control path makes the exploration cost dominate short replies.
            self.current_budget = 0;
            self.probe_budget = Some(0);
            self.probe_origin_budget = Some(old);
            self.probe_windows_remaining =
                Self::ZERO_DRAFT_PROBE_WINDOWS + Self::LONG_CONTEXT_ZERO_DRAFT_WARMUP_WINDOWS;
            return budget_change(old, self.current_budget);
        }
        if long_context
            && old > 0
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_none()
        {
            // Keep the initial production depth stable until the second
            // representative MTP window is available. Otherwise the generic
            // adjacent-depth explorer would move to d-1 after the first sample
            // and reintroduce the same transition bias before the d=0 control.
            return MtpDraftBudgetChange::default();
        }
        if old == 1
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_none()
            && self
                .cost_estimate(regime, old)
                .is_some_and(|estimate| estimate.samples >= zero_draft_probe_min_samples)
        {
            self.current_budget = 0;
            self.probe_budget = Some(0);
            self.probe_origin_budget = Some(old);
            self.probe_windows_remaining = Self::ZERO_DRAFT_PROBE_WINDOWS
                + usize::from(long_context) * Self::LONG_CONTEXT_ZERO_DRAFT_WARMUP_WINDOWS;
            return budget_change(old, self.current_budget);
        }
        if old == 0 && self.probe_budget.is_none() {
            return MtpDraftBudgetChange::default();
        }
        if old == 1
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_some()
            && self.cooldown_windows == 0
            && self
                .cost_estimate(regime, 1)
                .is_some_and(|estimate| estimate.samples >= Self::ZERO_DRAFT_MIN_COST_SAMPLES)
        {
            // A previous zero-draft sample is stale once the response enters
            // a different acceptance/cost phase. Re-test ordinary decode only
            // while the MTP cache is still synchronized, and discard the old
            // control estimate so it cannot drive an uncontrolled switch.
            self.cost_estimates
                .retain(|estimate| estimate.regime != regime || estimate.draft_tokens != 0);
            self.current_budget = 0;
            self.probe_budget = Some(0);
            self.probe_origin_budget = Some(old);
            self.probe_windows_remaining = Self::ZERO_DRAFT_PROBE_WINDOWS
                + usize::from(long_context) * Self::LONG_CONTEXT_ZERO_DRAFT_WARMUP_WINDOWS;
            return budget_change(old, self.current_budget);
        }

        let mut next = old;
        let full_accept = window.accepted_draft_tokens == window.attempted_draft_tokens;
        if !full_accept {
            let rejected_probe = self.probe_budget == Some(old);
            next = if long_context && old > 1 {
                // At long context, every extra verifier position carries the
                // full attention/cache-read cost. If depth d did not fully
                // accept, probe the actually useful depth next instead of
                // spending another window at d.
                window.accepted_draft_tokens.max(1).min(old)
            } else {
                window.accepted_draft_tokens.saturating_add(1).min(old)
            };
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            if rejected_probe {
                self.back_off_next_probe();
            } else if old == 1 {
                self.cooldown_windows = self.cooldown_windows.saturating_sub(1);
            } else {
                self.arm_initial_probe_cooldown();
            }
        } else if self.probe_budget == Some(old) {
            self.probe_windows_remaining = self.probe_windows_remaining.saturating_sub(1);
            if self.probe_windows_remaining == 0 {
                let origin = self.probe_origin_budget.take();
                self.probe_budget = None;
                next = origin.map_or(old, |origin| {
                    self.preferred_probe_budget(regime, origin, old)
                });
                if origin.is_some_and(|origin| next == origin) && origin != Some(0) {
                    self.back_off_next_probe();
                } else {
                    self.arm_initial_probe_cooldown();
                }
            }
        } else {
            next = self.best_measured_budget(regime, old);
            if next > old && self.acceptance_ewma.unwrap_or(acceptance) < Self::HIGH_ACCEPTANCE {
                next = old;
            }
            let adjacent_probe_min_samples = if long_context {
                1
            } else {
                Self::MIN_COST_SAMPLES
            };
            if next == old {
                if self.cooldown_windows > 0 {
                    self.cooldown_windows -= 1;
                } else if self
                    .cost_estimate(regime, old)
                    .is_some_and(|estimate| estimate.samples >= adjacent_probe_min_samples)
                {
                    if let Some(probe) = self.next_probe_budget(regime, old) {
                        if probe < old
                            || self.acceptance_ewma.unwrap_or(acceptance) >= Self::HIGH_ACCEPTANCE
                        {
                            next = probe;
                            if long_context && probe < old {
                                // Adopt the cheaper-depth candidate directly
                                // for one window. The next d=1 observation
                                // immediately compares against ordinary decode,
                                // avoiding a multi-window nested probe at 32K+.
                                self.probe_budget = None;
                                self.probe_origin_budget = None;
                                self.probe_windows_remaining = 0;
                            } else {
                                self.probe_budget = Some(probe);
                                self.probe_origin_budget = Some(old);
                                self.probe_windows_remaining = Self::PROBE_WINDOWS;
                            }
                        }
                    }
                }
            }
        }

        let smoothed_acceptance = self.acceptance_ewma.unwrap_or(acceptance);
        if smoothed_acceptance < Self::LOW_ACCEPTANCE && old > 1 {
            next = next.min(old - 1);
        }
        if acceptance == 0.0 {
            next = next.min(1);
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            // A rejected wider probe should back off before it is retried.
            // At budget 1, however, repeatedly re-arming this cooldown makes
            // an unprofitable MTP path impossible to compare with ordinary
            // decode: every zero-acceptance window resets the countdown. Keep
            // the existing countdown moving toward a fresh zero-draft probe.
            if old > 1 {
                self.arm_initial_probe_cooldown();
            }
        }

        self.current_budget = next.min(self.max_draft_tokens);
        budget_change(old, self.current_budget)
    }

    fn arm_initial_probe_cooldown(&mut self) {
        self.cooldown_windows = Self::INITIAL_PROBE_COOLDOWN_WINDOWS;
        self.next_probe_cooldown_windows =
            (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2).min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
    }

    fn back_off_next_probe(&mut self) {
        self.cooldown_windows = self.next_probe_cooldown_windows.clamp(
            Self::INITIAL_PROBE_COOLDOWN_WINDOWS,
            Self::MAX_PROBE_COOLDOWN_WINDOWS,
        );
        self.next_probe_cooldown_windows = self
            .cooldown_windows
            .saturating_mul(2)
            .min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
    }

    fn record_cost(
        &mut self,
        regime: MtpDraftPolicyRegime,
        draft_tokens: usize,
        cost: f64,
        acceptance: f64,
    ) {
        if let Some(estimate) = self
            .cost_estimates
            .iter_mut()
            .find(|estimate| estimate.regime == regime && estimate.draft_tokens == draft_tokens)
        {
            estimate.cost_ewma = update_ewma(Some(estimate.cost_ewma), cost, Self::EWMA_ALPHA);
            estimate.acceptance_ewma =
                update_ewma(Some(estimate.acceptance_ewma), acceptance, Self::EWMA_ALPHA);
            estimate.samples = estimate.samples.saturating_add(1);
        } else {
            self.cost_estimates.push(MtpDraftCostEstimate {
                regime,
                draft_tokens,
                cost_ewma: cost,
                acceptance_ewma: acceptance,
                samples: 1,
            });
        }
    }

    fn cost_estimate(
        &self,
        regime: MtpDraftPolicyRegime,
        draft_tokens: usize,
    ) -> Option<&MtpDraftCostEstimate> {
        self.cost_estimates
            .iter()
            .find(|estimate| estimate.regime == regime && estimate.draft_tokens == draft_tokens)
    }

    fn best_measured_budget(&self, regime: MtpDraftPolicyRegime, current: usize) -> usize {
        let Some(current_estimate) = self.cost_estimate(regime, current) else {
            return current;
        };
        if current_estimate.samples < Self::MIN_COST_SAMPLES {
            return current;
        }
        self.cost_estimates
            .iter()
            .filter(|estimate| {
                estimate.regime == regime
                    && estimate.draft_tokens > 0
                    && estimate.samples >= Self::MIN_COST_SAMPLES
                    && (estimate.draft_tokens <= current
                        || estimate.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
                    && estimate.cost_ewma
                        < current_estimate.cost_ewma * Self::COST_IMPROVEMENT_RATIO
            })
            .min_by(|left, right| left.cost_ewma.total_cmp(&right.cost_ewma))
            .map_or(current, |estimate| estimate.draft_tokens)
    }

    fn next_probe_budget(&self, regime: MtpDraftPolicyRegime, current: usize) -> Option<usize> {
        let lower = current.checked_sub(1);
        let upper = current
            .checked_add(1)
            .filter(|&budget| budget <= self.max_draft_tokens);
        [lower, upper]
            .into_iter()
            .flatten()
            .filter(|&budget| {
                if budget == 0 && current > 0 {
                    // Wait for a controlled zero-draft probe. Once sampled,
                    // refreshes are scheduled explicitly after cooldown rather
                    // than by the generic adjacent-budget probe path.
                    if self.cost_estimate(regime, 0).is_some()
                        || self.cost_estimate(regime, current).is_none_or(|estimate| {
                            estimate.samples < Self::ZERO_DRAFT_MIN_COST_SAMPLES
                        })
                    {
                        return false;
                    }
                }
                budget <= current
                    || current == 0
                    || self
                        .cost_estimate(regime, budget)
                        .is_none_or(|estimate| estimate.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
            })
            .min_by_key(|&budget| {
                self.cost_estimate(regime, budget)
                    .map_or(0, |estimate| estimate.samples)
            })
    }

    fn preferred_probe_budget(
        &self,
        regime: MtpDraftPolicyRegime,
        origin: usize,
        probe: usize,
    ) -> usize {
        let Some(origin_cost) = self.cost_estimate(regime, origin) else {
            return probe;
        };
        let Some(probe_cost) = self.cost_estimate(regime, probe) else {
            return origin;
        };
        let improvement_ratio = if probe == 0 {
            Self::ZERO_DRAFT_COST_IMPROVEMENT_RATIO
        } else {
            Self::COST_IMPROVEMENT_RATIO
        };
        if probe_cost.samples >= Self::MIN_COST_SAMPLES
            && (probe <= origin
                || origin == 0
                || probe_cost.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
            && probe_cost.cost_ewma < origin_cost.cost_ewma * improvement_ratio
        {
            probe
        } else {
            origin
        }
    }
}

fn update_ewma(current: Option<f64>, sample: f64, alpha: f64) -> f64 {
    current.map_or(sample, |value| value.mul_add(1.0 - alpha, sample * alpha))
}

fn budget_change(old: usize, new: usize) -> MtpDraftBudgetChange {
    MtpDraftBudgetChange {
        reduced: new < old,
        increased: new > old,
    }
}

/// Text-only single-request stream for Qwen MTP speculative decoding.
pub struct MtpTextGenerationStream<'m, M>
where
    M: MtpSpeculativeModel,
{
    model: &'m M,
    mtp: &'m M::MtpHead,
    cache: Vec<LayerCache>,
    mtp_cache: MtpCache,
    history: Vec<u32>,
    request: GenerateRequest,
    cfg: MtpSpeculativeConfig,
    pending_tokens: VecDeque<u32>,
    detok: DecodeStream<'m>,
    /// Hidden state for the token immediately before the current pending token.
    last_hidden: Array,
    emitted_new_tokens: usize,
    finished: bool,
    dummy_position_ids: Option<Array>,
    prng_state: Array,
    adaptive_draft_tokens: usize,
    draft_policy: QwenMtpDraftPolicyState,
    stats: MtpSpeculativeStats,
    constraint: Option<ConstraintSession>,
}

impl<'m, M> MtpTextGenerationStream<'m, M>
where
    M: MtpSpeculativeModel,
{
    /// Construct a text-only MTP speculative stream.
    pub fn new_text_only(
        model: &'m M,
        mtp: &'m M::MtpHead,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Self> {
        if request.pixel_values.is_some() {
            return Err(anyhow!(
                "MtpTextGenerationStream::new_text_only called with pixel_values; MTP speculative decoding is text-only"
            ));
        }
        if request.prompt_ids.is_empty() {
            return Err(anyhow!(
                "MtpTextGenerationStream::new_text_only: prompt_ids cannot be empty"
            ));
        }
        if cfg.max_draft_tokens == 0 {
            return Err(anyhow!(
                "MtpTextGenerationStream::new_text_only: max_draft_tokens must be > 0"
            ));
        }
        let prompt_len = request.prompt_ids.len();
        let cap = ((prompt_len + request.max_new_tokens) as i32)
            .max(crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let dtype = model.cache_dtype();
        let mut cache = model.make_cache(1, cap, dtype)?;
        if let Some(bits) = request.kv_cache_turboquant_bits {
            enable_turboquant_kv_caches(&mut cache, bits)?;
        }
        let mut mtp_cache = model.make_mtp_cache(mtp, 1, cap, dtype)?;
        let dummy_position_ids = if model.requires_position_ids() {
            None
        } else {
            Some(build_position_ids(0, 1)?)
        };

        let chunk_size = request.prefill_chunk_size;
        let prompt_len_i32 = prompt_len as i32;
        let mut pos = 0_i32;
        let mut stats = MtpSpeculativeStats::default();
        let mut last_prompt_hidden = None;
        let mut mtp_prev_hidden: Option<Array> = None;
        while pos < prompt_len_i32 {
            let remaining = prompt_len_i32 - pos;
            let n = if chunk_size == 0 {
                remaining
            } else {
                remaining.min(chunk_size as i32)
            };
            let chunk_ids = &request.prompt_ids[pos as usize..(pos as usize + n as usize)];
            let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;
            let chunk_pos_ids = match dummy_position_ids.as_ref() {
                Some(dummy) => dummy.clone(),
                None => build_position_ids(pos, n)?,
            };
            let forward_start = Instant::now();
            let hidden = model.forward_text_hidden(
                &chunk_arr,
                &chunk_pos_ids,
                None,
                None,
                Some(&mut cache),
                ().into(),
            )?;
            add_elapsed_us(&mut stats.verify_forward_us, forward_start);
            let prev_hidden = match mtp_prev_hidden.as_ref() {
                Some(hidden) => hidden.clone(),
                None => zero_hidden_like_position(&hidden)?,
            };
            let commit_start = Instant::now();
            commit_mtp_cache_hidden_prefix(
                model,
                mtp,
                &mut mtp_cache,
                &prev_hidden,
                chunk_ids,
                &hidden,
                &chunk_pos_ids,
                (),
            )?;
            add_mtp_prefill_cache_commit_us(&mut stats, commit_start);
            let chunk_last_hidden = slice_hidden_position(&hidden, n - 1)?;
            mtp_prev_hidden = Some(chunk_last_hidden.clone());
            if pos + n == prompt_len_i32 {
                last_prompt_hidden = Some(chunk_last_hidden);
            }
            pos += n;
        }
        let last_prompt_hidden =
            last_prompt_hidden.ok_or_else(|| anyhow!("MTP prefill produced no prompt hidden"))?;

        let projection_start = Instant::now();
        let first_logits =
            model.project_hidden_on(&last_prompt_hidden, StreamOrDevice::default())?;
        add_elapsed_us(&mut stats.projection_us, projection_start);
        let mut constraint = request
            .constraint
            .as_ref()
            .map(|plan| plan.start_session())
            .transpose()?;
        let first_logits = constrain_speculative_logits(&mut constraint, &first_logits, &[])?;
        let mut prng_state = mlx::random::key(request.sampler.seed)?;
        let sampling_start = Instant::now();
        let first_tokens = sample_logits_positions(
            &first_logits,
            request.sampler,
            &request.prompt_ids,
            &mut prng_state,
        )?;
        add_elapsed_us(&mut stats.sampling_us, sampling_start);
        let first_token = *first_tokens
            .first()
            .ok_or_else(|| anyhow!("MTP prefill produced no first token"))?;
        commit_constraint_token(&mut constraint, first_token)?;

        let mut history = request.prompt_ids.clone();
        history.push(first_token);
        let mut pending_tokens = VecDeque::new();
        pending_tokens.push_back(first_token);

        Ok(Self {
            model,
            mtp,
            cache,
            mtp_cache,
            history,
            request,
            cfg,
            pending_tokens,
            detok: tokenizer.decode_stream(true),
            last_hidden: last_prompt_hidden,
            emitted_new_tokens: 0,
            finished: false,
            dummy_position_ids,
            prng_state,
            adaptive_draft_tokens: cfg.max_draft_tokens,
            draft_policy: QwenMtpDraftPolicyState::new(cfg.max_draft_tokens),
            stats,
            constraint,
        })
    }

    /// Return cumulative speculative-window counters for this stream.
    pub fn stats(&self) -> MtpSpeculativeStats {
        self.stats.clone()
    }

    /// Pull the next generated token event.
    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }

        let token = self
            .pending_tokens
            .pop_front()
            .ok_or_else(|| anyhow!("MTP stream invariant: pending token queue is empty"))?;
        self.emitted_new_tokens += 1;
        let text = self.detok.step(token)?.unwrap_or_default();
        let finish_reason = if self.request.stop_token_ids.contains(&token) {
            Some("stop")
        } else if self.emitted_new_tokens >= self.request.max_new_tokens {
            Some("length")
        } else {
            None
        };

        if finish_reason == Some("length") {
            if let Some(constraint) = self.constraint.as_mut() {
                if constraint.requires_accepting_state_at_length() && !constraint.is_accepting()? {
                    self.finished = true;
                    return Err(anyhow!(
                        "max_new_tokens reached before constrained output became complete"
                    ));
                }
            }
        }

        if finish_reason.is_some() {
            self.finished = true;
            return Ok(Some(GenerateEvent {
                token,
                text,
                finish_reason,
            }));
        }

        if self.pending_tokens.is_empty() {
            self.fill_window(token)?;
        }

        Ok(Some(GenerateEvent {
            token,
            text,
            finish_reason: None,
        }))
    }

    fn fill_window(&mut self, current_token: u32) -> Result<()> {
        let remaining = self
            .request
            .max_new_tokens
            .saturating_sub(self.emitted_new_tokens);
        if remaining == 0 {
            return Ok(());
        }

        let window_started = Instant::now();
        let stats_before_window = self.stats.clone();
        let timing_before = self.stats.draft_cap_timing();
        let context_tokens = self.history.len();
        let draft_budget = self
            .adaptive_draft_tokens
            .min(self.cfg.max_draft_tokens)
            .min(remaining);
        let maintain_mtp_cache = self.draft_policy.should_maintain_mtp_cache();
        let mut draft_constraint = self.constraint.as_ref().map(ConstraintSession::fork);
        let draft_result = self.draft_tokens(current_token, draft_budget, &mut draft_constraint)?;
        let draft_tokens = draft_result.tokens;
        let _draft_distributions = draft_result.distributions;
        let verify_input = verify_input(current_token, &draft_tokens);
        let verify_start_pos = (self.history.len() - 1) as i32;
        let verify_pos_ids = self.position_ids(verify_start_pos, verify_input.len() as i32)?;
        let verify_arr: Array =
            (&verify_input[..], &[1_i32, verify_input.len() as i32][..]).try_into()?;
        let pre_window_hidden = self.last_hidden.clone();

        let base_snapshot = (draft_budget > 0).then(|| {
            self.cache
                .iter()
                .map(LayerCache::snapshot)
                .collect::<Vec<_>>()
        });
        let verify_forward_start = Instant::now();
        let verified_hidden = {
            // A zero-draft window is the ordinary single-token target path.
            // Exact speculative QMM is required only when target positions
            // must be compared against drafted tokens.
            let _verify_qmm = (draft_budget > 0).then(crate::nn::verify_qmm_scope);
            self.model.forward_text_hidden(
                &verify_arr,
                &verify_pos_ids,
                None,
                None,
                Some(&mut self.cache),
                ().into(),
            )?
        };
        add_elapsed_us(&mut self.stats.verify_forward_us, verify_forward_start);
        let resolution = if self.request.sampler.is_pipelinable() && self.constraint.is_none() {
            resolve_greedy_verified_hidden_until_mismatch(
                self.model,
                &verified_hidden,
                &draft_tokens,
                &mut self.stats,
                (),
            )?
        } else {
            let projection_start = Instant::now();
            let verified_logits = self
                .model
                .project_mtp_verify_hidden_on(&verified_hidden, ())?;
            add_elapsed_us(&mut self.stats.projection_us, projection_start);
            let verified_logits = constrain_speculative_logits(
                &mut self.constraint,
                &verified_logits,
                &draft_tokens,
            )?;
            let sampling_start = Instant::now();
            let resolution = if self.request.sampler.is_pipelinable() {
                let verified_ids = mlx::ops::reduction::argmax(&verified_logits, -1, false)?;
                let verified_tokens: Vec<u32> = verified_ids.to_vec()?;
                resolve_speculative_tokens(&draft_tokens, &verified_tokens)?
            } else {
                resolve_exact_deterministic_target_logits(
                    &draft_tokens,
                    &verified_logits,
                    self.request.sampler,
                    &self.history,
                    &mut self.prng_state,
                )?
            };
            add_elapsed_us(&mut self.stats.sampling_us, sampling_start);
            resolution
        };
        self.stats.windows += 1;
        self.stats.drafted_tokens += draft_tokens.len();
        self.stats.accepted_draft_tokens += resolution.accepted_draft_len;
        self.stats
            .record_exact_sampling(resolution.exact_sampling());
        self.stats
            .record_window_acceptance(draft_tokens.len(), resolution.accepted_draft_len);
        if resolution.needs_rollback {
            self.stats.rollback_count += 1;
        }
        let accepted_len = resolution.accepted_verify_input_len;
        let (accepted_hidden, accepted_position_ids, accepted_last_hidden) = if resolution
            .needs_rollback
        {
            let accepted_position_ids = slice_position_ids_prefix(&verify_pos_ids, accepted_len)?;
            let rollback_start = Instant::now();
            let accepted_hidden = rollback_main_cache_to_accepted_prefix(
                self.model,
                &mut self.cache,
                base_snapshot
                    .as_deref()
                    .ok_or_else(|| anyhow!("MTP rollback snapshot absent"))?,
                MainCacheRollbackInput {
                    accepted_by_row: &[(0, accepted_len)],
                    verify_input: &verify_input,
                    accepted_position_ids: &accepted_position_ids,
                    verified_hidden: &verified_hidden,
                },
                (),
            )?;
            add_elapsed_us(&mut self.stats.main_rollback_us, rollback_start);
            (
                accepted_hidden.clone(),
                accepted_position_ids,
                slice_hidden_position(&accepted_hidden, accepted_len as i32 - 1)?,
            )
        } else {
            (
                verified_hidden.clone(),
                verify_pos_ids.clone(),
                slice_hidden_position(&verified_hidden, accepted_len as i32 - 1)?,
            )
        };
        let accepted_input = verify_input[..accepted_len].to_vec();

        if resolution.needs_rollback {
            let restore_start = Instant::now();
            self.mtp_cache.restore(&draft_result.cache_snapshot)?;
            add_elapsed_us(&mut self.stats.mtp_cache_restore_us, restore_start);
            let commit_start = Instant::now();
            commit_mtp_cache_hidden_prefix(
                self.model,
                self.mtp,
                &mut self.mtp_cache,
                &pre_window_hidden,
                &accepted_input,
                &accepted_hidden,
                &accepted_position_ids,
                (),
            )?;
            add_mtp_decode_cache_commit_us(&mut self.stats, commit_start);
        } else if maintain_mtp_cache {
            let commit_start = Instant::now();
            commit_mtp_cache_hidden_tail(
                self.model,
                self.mtp,
                &mut self.mtp_cache,
                &pre_window_hidden,
                &accepted_input,
                &accepted_hidden,
                &accepted_position_ids,
                (),
            )?;
            add_mtp_decode_cache_commit_us(&mut self.stats, commit_start);
            self.stats.mtp_cache_reuse_count = self.stats.mtp_cache_reuse_count.saturating_add(1);
            self.stats.mtp_cache_reused_tokens = self
                .stats
                .mtp_cache_reused_tokens
                .saturating_add(accepted_input.len().saturating_sub(1));
        }
        self.last_hidden = accepted_last_hidden;

        let mut tokens_to_append = resolution.tokens_to_append;
        if let Some(stop_idx) = tokens_to_append
            .iter()
            .position(|token| self.request.stop_token_ids.contains(token))
        {
            tokens_to_append.truncate(stop_idx + 1);
        }
        tokens_to_append.truncate(remaining);
        if let Some(constraint) = self.constraint.as_ref() {
            constraint.truncate_invalid_speculative_bonus(&mut tokens_to_append)?;
        }
        let committed_tokens = tokens_to_append.len();
        for token in tokens_to_append {
            commit_constraint_token(&mut self.constraint, token)?;
            self.history.push(token);
            self.pending_tokens.push_back(token);
        }

        let total_us = elapsed_us_since(window_started);
        let stats_delta = self.stats.saturating_delta_since(&stats_before_window);
        let change = self
            .draft_policy
            .observe_window(MtpDraftPolicyWindow::from_stats_delta(
                draft_tokens.len(),
                resolution.accepted_draft_len,
                committed_tokens,
                total_us,
                context_tokens,
                1,
                MtpDraftPolicyKvState::Contiguous,
                &stats_delta,
            ));
        if change.reduced {
            self.stats.draft_budget_reductions =
                self.stats.draft_budget_reductions.saturating_add(1);
        } else if change.increased {
            self.stats.draft_budget_increases = self.stats.draft_budget_increases.saturating_add(1);
        }
        self.adaptive_draft_tokens = self.draft_policy.current_budget();
        let timing_delta = self
            .stats
            .draft_cap_timing()
            .saturating_delta_since(timing_before);
        self.stats.record_draft_cap_observation(
            self.cfg.max_draft_tokens,
            &[draft_tokens.len()],
            &[context_tokens],
            resolution.accepted_draft_len,
            committed_tokens,
            usize::from(resolution.needs_rollback),
            total_us,
            timing_delta,
        );

        Ok(())
    }

    fn draft_tokens(
        &mut self,
        current_token: u32,
        draft_budget: usize,
        constraint: &mut Option<ConstraintSession>,
    ) -> Result<MtpDraftResult> {
        let mtp_snapshot = self.mtp_cache.snapshot();
        let mut draft_tokens = Vec::with_capacity(draft_budget);
        let mut draft_history = self.history.clone();
        let mut input_hidden = self.last_hidden.clone();
        let mut input_token = current_token;
        let start_pos = (self.history.len() - 1) as i32;
        let draft_uniforms = if self.request.sampler.is_pipelinable() {
            vec![0.0; draft_budget]
        } else {
            let mut draft_prng = split_speculative_draft_prng(&mut self.prng_state)?;
            draw_uniforms(&mut draft_prng, draft_budget)?
        };
        let mut distributions = Vec::with_capacity(draft_budget);

        for (offset, &draft_uniform) in draft_uniforms.iter().enumerate().take(draft_budget) {
            let token_arr: Array = (&[input_token][..], &[1_i32, 1_i32][..]).try_into()?;
            let position_ids = self.position_ids(start_pos + offset as i32, 1)?;
            let draft_forward_start = Instant::now();
            let output = self.model.mtp_forward_on(
                self.mtp,
                &input_hidden,
                &token_arr,
                &position_ids,
                None,
                Some(&mut self.mtp_cache),
                (),
            )?;
            add_elapsed_us(&mut self.stats.draft_forward_us, draft_forward_start);
            let draft_logits = constrain_speculative_logits(constraint, &output.logits, &[])?;
            let sampling_start = Instant::now();
            let (next_token, distribution) = if self.request.sampler.is_pipelinable() {
                sample_draft_logits_position(
                    &draft_logits,
                    self.request.sampler,
                    &draft_history,
                    None,
                )?
            } else {
                sample_draft_logits_position_with_uniform(
                    &draft_logits,
                    self.request.sampler,
                    &draft_history,
                    draft_uniform,
                )?
            };
            add_elapsed_us(&mut self.stats.sampling_us, sampling_start);
            commit_constraint_token(constraint, next_token)?;
            draft_tokens.push(next_token);
            distributions.push(distribution);
            draft_history.push(next_token);
            input_hidden = output.hidden_states;
            input_token = next_token;
        }

        Ok(MtpDraftResult {
            tokens: draft_tokens,
            distributions,
            cache_snapshot: mtp_snapshot,
        })
    }

    fn position_ids(&self, start_pos: i32, len: i32) -> Result<Array> {
        match self.dummy_position_ids.as_ref() {
            Some(dummy) => Ok(dummy.clone()),
            None => build_position_ids(start_pos, len),
        }
    }
}

fn constrain_speculative_logits(
    constraint: &mut Option<ConstraintSession>,
    logits: &Array,
    draft_tokens: &[u32],
) -> Result<Array> {
    match constraint {
        Some(session) => {
            apply_speculative_token_masks(logits, &[Some(session.speculative_masks(draft_tokens)?)])
        }
        None => Ok(logits.clone()),
    }
}

fn commit_constraint_token(constraint: &mut Option<ConstraintSession>, token: u32) -> Result<()> {
    if let Some(session) = constraint {
        session.commit_token(token)?;
    }
    Ok(())
}

pub(crate) fn resolve_greedy_verified_hidden_until_mismatch<M>(
    model: &M,
    verified_hidden: &Array,
    draft_tokens: &[u32],
    stats: &mut MtpSpeculativeStats,
    target: impl Into<StreamOrDevice>,
) -> Result<SpeculativeResolution>
where
    M: MtpSpeculativeModel,
{
    let target = target.into();
    let projection_start = Instant::now();
    let verified_logits = model.project_mtp_verify_hidden_on(verified_hidden, target)?;
    add_elapsed_us(&mut stats.projection_us, projection_start);

    let sampling_start = Instant::now();
    let verified_ids = mlx::ops::reduction::argmax(&verified_logits, -1, false)?;
    let verified_tokens: Vec<u32> = verified_ids.to_vec()?;
    add_elapsed_us(&mut stats.sampling_us, sampling_start);
    resolve_speculative_tokens(draft_tokens, &verified_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeGreedyProjectModel {
        tokens: Vec<u32>,
        project_calls: AtomicUsize,
        replay_calls: AtomicUsize,
    }

    impl FakeGreedyProjectModel {
        fn new(tokens: Vec<u32>) -> Self {
            Self {
                tokens,
                project_calls: AtomicUsize::new(0),
                replay_calls: AtomicUsize::new(0),
            }
        }

        fn project_calls(&self) -> usize {
            self.project_calls.load(Ordering::Relaxed)
        }
    }

    impl Model for FakeGreedyProjectModel {
        fn make_cache(&self, _batch: i32, _cap: i32, _dtype: Dtype) -> Result<Vec<LayerCache>> {
            Ok(Vec::new())
        }

        fn forward_on(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            Err(anyhow!("FakeGreedyProjectModel::forward_on unused"))
        }

        fn batched_prefill(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _attention_mask: &Array,
            _linear_attention_mask: &Array,
            _per_row_lens: &[i32],
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            Err(anyhow!("FakeGreedyProjectModel::batched_prefill unused"))
        }

        fn forward_text_hidden(
            &self,
            input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            self.replay_calls.fetch_add(1, Ordering::Relaxed);
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            if dims.len() != 2 {
                return Err(anyhow!(
                    "FakeGreedyProjectModel::forward_text_hidden expected [B,S], got {dims:?}"
                ));
            }
            if let Some(cache) = cache {
                for layer in cache {
                    if let LayerCache::Linear(gd) = layer {
                        let row_lens = vec![dims[1]; dims[0] as usize];
                        gd.advance(&row_lens)?;
                    }
                }
            }
            Array::zeros((dims[0], dims[1], 1_i32), Dtype::Float32).map_err(anyhow::Error::from)
        }

        fn project_hidden_on(&self, hidden: &Array, _target: StreamOrDevice) -> Result<Array> {
            self.project_calls.fetch_add(1, Ordering::Relaxed);
            let shape = hidden.shape();
            let dims = shape.as_slice();
            let seq = dims[1] as usize;
            if self.tokens.len() != seq {
                return Err(anyhow!(
                    "fake token count {} does not match hidden seq {seq}",
                    self.tokens.len()
                ));
            }
            let vocab = 128_usize;
            let mut logits = vec![0.0_f32; seq * vocab];
            for (pos, &token) in self.tokens.iter().enumerate() {
                logits[pos * vocab + token as usize] = 100.0;
            }
            (&logits[..], &[1_i32, seq as i32, vocab as i32][..])
                .try_into()
                .map_err(anyhow::Error::from)
        }

        fn model_meta(&self) -> crate::core::model::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    impl MtpSpeculativeModel for FakeGreedyProjectModel {
        type MtpHead = ();

        fn load_mtp_head(&self, _loader: &Loader) -> Result<Self::MtpHead> {
            Ok(())
        }

        fn make_mtp_cache(
            &self,
            _mtp: &Self::MtpHead,
            _batch: i32,
            _cap: i32,
            _dtype: Dtype,
        ) -> Result<MtpCache> {
            Err(anyhow!("FakeGreedyProjectModel::make_mtp_cache unused"))
        }

        fn mtp_hidden_size(&self, _mtp: &Self::MtpHead) -> i32 {
            1
        }

        fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
            Dtype::Float32
        }

        fn mtp_forward_hidden_on(
            &self,
            _mtp: &Self::MtpHead,
            _hidden_states: &Array,
            _next_token_ids: &Array,
            _position_ids: &Array,
            _mask: Option<&Array>,
            _mtp_cache: Option<&mut MtpCache>,
            _target: impl Into<StreamOrDevice>,
        ) -> Result<Array> {
            Err(anyhow!(
                "FakeGreedyProjectModel::mtp_forward_hidden_on unused"
            ))
        }

        fn mtp_forward_on(
            &self,
            _mtp: &Self::MtpHead,
            _hidden_states: &Array,
            _next_token_ids: &Array,
            _position_ids: &Array,
            _mask: Option<&Array>,
            _mtp_cache: Option<&mut MtpCache>,
            _target: impl Into<StreamOrDevice>,
        ) -> Result<MtpStepOutput> {
            Err(anyhow!("FakeGreedyProjectModel::mtp_forward_on unused"))
        }
    }

    #[test]
    fn greedy_verify_resolve_batches_projection_before_mismatch_resolution() {
        let model = FakeGreedyProjectModel::new(vec![4, 99, 6, 7]);
        let hidden = Array::zeros((1_i32, 4_i32, 1_i32), Dtype::Float32).expect("hidden");
        let mut stats = MtpSpeculativeStats::default();

        let resolution = resolve_greedy_verified_hidden_until_mismatch(
            &model,
            &hidden,
            &[4, 5, 6],
            &mut stats,
            (),
        )
        .expect("resolution");

        assert_eq!(resolution.accepted_draft_len, 1);
        assert_eq!(resolution.tokens_to_append, vec![4, 99]);
        assert_eq!(resolution.accepted_verify_input_len, 2);
        assert!(resolution.needs_rollback);
        assert_eq!(model.project_calls(), 1);
    }

    #[test]
    fn greedy_verify_resolve_projects_bonus_after_full_accept() {
        let model = FakeGreedyProjectModel::new(vec![4, 5, 6, 7]);
        let hidden = Array::zeros((1_i32, 4_i32, 1_i32), Dtype::Float32).expect("hidden");
        let mut stats = MtpSpeculativeStats::default();

        let resolution = resolve_greedy_verified_hidden_until_mismatch(
            &model,
            &hidden,
            &[4, 5, 6],
            &mut stats,
            (),
        )
        .expect("resolution");

        assert_eq!(resolution.accepted_draft_len, 3);
        assert_eq!(resolution.tokens_to_append, vec![4, 5, 6, 7]);
        assert_eq!(resolution.accepted_verify_input_len, 4);
        assert!(!resolution.needs_rollback);
        assert_eq!(model.project_calls(), 1);
    }

    #[test]
    fn mtp_policy_defaults_qwen35_dense_4b_to_d1() {
        let raw = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 2560,
                "num_hidden_layers": 32
            }
        });

        assert_eq!(default_mtp_draft_tokens_for_config(&raw), 1);
        assert_eq!(
            resolve_mtp_draft_tokens(&raw, MtpDraftTokensArg::Omitted),
            1
        );
    }

    #[test]
    fn mtp_policy_defaults_qwen36_and_qwen38_dense_27b_to_d2() {
        let raw = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 64
            }
        });

        assert_eq!(default_mtp_draft_tokens_for_config(&raw), 2);
    }

    #[test]
    fn mtp_policy_defaults_qwen36_moe_35b_a3b_to_d2() {
        let raw = serde_json::json!({
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_experts": 256,
                "num_experts_per_tok": 8
            }
        });

        assert_eq!(default_mtp_draft_tokens_for_config(&raw), 2);
    }

    #[test]
    fn mtp_policy_defaults_gemma4_to_d1() {
        for model_type in ["gemma4", "gemma4_unified"] {
            let raw = serde_json::json!({
                "model_type": model_type,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 3584,
                    "num_hidden_layers": 34
                }
            });

            assert_eq!(default_mtp_draft_tokens_for_config(&raw), 1);
            assert_eq!(
                resolve_mtp_draft_tokens(&raw, MtpDraftTokensArg::Omitted),
                1
            );
        }
    }

    #[test]
    fn mtp_policy_preserves_explicit_value() {
        let raw = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 64
            }
        });

        assert_eq!(
            resolve_mtp_draft_tokens(&raw, MtpDraftTokensArg::Explicit(1)),
            1
        );
    }

    #[test]
    fn mtp_stats_tracks_attempts_and_accepts_by_draft_position() {
        let mut stats = MtpSpeculativeStats::default();

        stats.record_window_acceptance(4, 0);
        stats.record_window_acceptance(4, 2);
        stats.record_window_acceptance(2, 2);

        assert_eq!(stats.draft_attempts_by_position, vec![3, 3, 2, 2]);
        assert_eq!(stats.draft_accepts_by_position, vec![2, 2, 0, 0]);
        assert_eq!(stats.multi_token_windows(), 3);
    }

    #[test]
    fn draft_cap_context_bucket_uses_inclusive_boundaries() {
        assert_eq!(
            MtpDraftCapContextBucket::for_tokens(2_048),
            MtpDraftCapContextBucket::UpTo2k
        );
        assert_eq!(
            MtpDraftCapContextBucket::for_tokens(2_049),
            MtpDraftCapContextBucket::UpTo8k
        );
        assert_eq!(
            MtpDraftCapContextBucket::for_tokens(131_073),
            MtpDraftCapContextBucket::Above128k
        );
    }

    #[test]
    fn draft_cap_observation_aggregates_only_matching_regimes() {
        let mut stats = MtpSpeculativeStats::default();
        let timing = MtpDraftCapTiming {
            draft_forward_us: 10,
            verify_forward_us: 20,
            projection_us: 3,
            sampling_us: 4,
            main_rollback_us: 5,
            decode_cache_commit_us: 6,
            cache_restore_us: 7,
        };

        stats.record_draft_cap_observation(2, &[2, 2], &[1_000, 2_000], 3, 5, 1, 100, timing);
        stats.record_draft_cap_observation(2, &[2, 2], &[1_500, 2_048], 2, 4, 2, 120, timing);
        stats.record_draft_cap_observation(2, &[1, 2], &[2_048, 2_049], 1, 2, 1, 80, timing);

        assert_eq!(stats.draft_cap_observations.len(), 2);
        let homogeneous = &stats.draft_cap_observations[0];
        assert_eq!(homogeneous.windows, 4);
        assert_eq!(homogeneous.accepted_draft_tokens, 5);
        assert_eq!(homogeneous.committed_tokens, 9);
        assert_eq!(homogeneous.total_us, 220);
        assert_eq!(homogeneous.draft_forward_us, 20);

        let mixed = &stats.draft_cap_observations[1];
        assert_eq!(mixed.windows, 2);
        assert!(mixed.mixed_context_buckets);
        assert_eq!(mixed.min_draft_tokens, 1);
        assert_eq!(mixed.max_draft_tokens, 2);
        assert_eq!(mixed.context_bucket, MtpDraftCapContextBucket::UpTo8k);
    }

    fn policy_window(
        draft_tokens: usize,
        committed_tokens: usize,
        total_us: u64,
    ) -> MtpDraftPolicyWindow {
        MtpDraftPolicyWindow {
            attempted_draft_tokens: draft_tokens,
            accepted_draft_tokens: draft_tokens,
            committed_tokens,
            total_us,
            context_tokens: 1_024,
            batch_width: 1,
            kv_state: MtpDraftPolicyKvState::Contiguous,
            ..MtpDraftPolicyWindow::default()
        }
    }

    #[test]
    fn mtp_cost_aware_policy_immediately_reduces_zero_acceptance() {
        let mut policy = Gemma4DrafterPolicyState::new(4);
        let mut window = policy_window(4, 1, 4_000);
        window.accepted_draft_tokens = 0;
        window.main_rollback_us = 500;
        window.mtp_cache_restore_us = 300;

        let change = policy.observe_window(window);

        assert_eq!(policy.current_budget(), 1);
        assert!(change.reduced);
    }

    #[test]
    fn mtp_cost_aware_policy_rejects_a_more_expensive_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(4);

        policy.observe_window(policy_window(4, 4, 400));
        let regime = policy_window(4, 4, 400).regime();
        policy.record_cost(regime, 0, 1_000.0, 1.0);
        assert_eq!(
            policy.observe_window(policy_window(4, 4, 400)).reduced,
            true
        );
        assert_eq!(policy.current_budget(), 3);
        policy.observe_window(policy_window(3, 3, 600));
        let change = policy.observe_window(policy_window(3, 3, 600));

        assert_eq!(policy.current_budget(), 4);
        assert!(change.increased);
    }

    #[test]
    fn mtp_cost_aware_policy_backs_off_rejected_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(4);

        policy.observe_window(policy_window(4, 4, 400));
        let regime = policy_window(4, 4, 400).regime();
        policy.record_cost(regime, 0, 1_000.0, 1.0);
        policy.observe_window(policy_window(4, 4, 400));
        policy.observe_window(policy_window(3, 3, 600));
        policy.observe_window(policy_window(3, 3, 600));
        assert_eq!(policy.current_budget(), 4);

        for _ in 0..16 {
            policy.observe_window(policy_window(4, 4, 400));
            assert_eq!(policy.current_budget(), 4);
        }
        assert!(policy.observe_window(policy_window(4, 4, 400)).reduced);
        assert_eq!(policy.current_budget(), 3);
    }

    #[test]
    fn mtp_cost_aware_policy_transitions_through_single_draft_before_ordinary_decode() {
        let mut policy = Gemma4DrafterPolicyState::new(2);
        policy.observe_window(policy_window(2, 2, 100));
        let mut rejected = policy_window(2, 1, 100);
        rejected.accepted_draft_tokens = 0;
        policy.observe_window(rejected);
        assert_eq!(policy.current_budget(), 1);

        let mut rejected_single = policy_window(1, 1, 100);
        rejected_single.accepted_draft_tokens = 0;
        policy.observe_window(rejected_single);
        policy.observe_window(rejected_single);
        assert_eq!(policy.current_budget(), 0);
        assert!(policy.uses_ordinary_decode());
    }

    #[test]
    fn mtp_cost_aware_policy_seeds_only_an_unobserved_regime() {
        let mut policy = Gemma4DrafterPolicyState::new(4);
        assert!(policy.seed_initial_budget(1));
        assert_eq!(policy.current_budget(), 1);

        policy.observe_window(policy_window(1, 2, 100));
        assert!(!policy.seed_initial_budget(3));
        assert_eq!(policy.current_budget(), 1);
    }

    #[test]
    fn mtp_cost_aware_policy_probes_ordinary_decode_early_for_low_acceptance() {
        let mut policy = Gemma4DrafterPolicyState::new(2);
        let mut rejected_wide = policy_window(2, 1, 2_000);
        rejected_wide.accepted_draft_tokens = 0;
        policy.observe_window(rejected_wide);
        assert_eq!(policy.current_budget(), 1);

        let mut rejected_single = policy_window(1, 1, 1_000);
        rejected_single.accepted_draft_tokens = 0;
        policy.observe_window(rejected_single);
        let change = policy.observe_window(rejected_single);

        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        assert!(policy.uses_ordinary_decode());
        assert!(!policy.should_maintain_mtp_cache());
    }

    #[test]
    fn mtp_cost_aware_policy_probes_ordinary_decode_early_at_long_context() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        let long_window = MtpDraftPolicyWindow {
            context_tokens: 65_536,
            ..policy_window(1, 2, 200)
        };

        policy.observe_window(long_window);
        let change = policy.observe_window(long_window);

        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        assert!(policy.should_maintain_mtp_cache());
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(MtpDraftPolicyWindow {
                context_tokens: 65_536,
                ..policy_window(0, 1, 50)
            });
        }
        assert!(policy.uses_ordinary_decode());
    }

    #[test]
    fn mtp_cost_aware_policy_does_not_collapse_wide_full_acceptance_into_ordinary_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(2);

        for _ in 0..64 {
            let budget = policy.current_budget();
            policy.observe_window(policy_window(budget, budget + 1, 200));
        }

        assert_eq!(policy.current_budget(), 2);
        assert!(!policy.uses_ordinary_decode());
    }

    #[test]
    fn mtp_cost_aware_policy_keeps_a_cheaper_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(4);

        policy.observe_window(policy_window(4, 4, 800));
        let regime = policy_window(4, 4, 800).regime();
        policy.record_cost(regime, 0, 1_000.0, 1.0);
        policy.observe_window(policy_window(4, 4, 800));
        policy.observe_window(policy_window(3, 3, 300));
        let change = policy.observe_window(policy_window(3, 3, 300));

        assert_eq!(policy.current_budget(), 3);
        assert!(!change.reduced);
        assert!(!change.increased);
    }

    #[test]
    fn mtp_cost_aware_policy_separates_context_batch_and_kv_regimes() {
        let base = policy_window(2, 2, 200);
        let long_context = MtpDraftPolicyWindow {
            context_tokens: 32_000,
            ..base
        };
        let batched = MtpDraftPolicyWindow {
            batch_width: 4,
            ..base
        };
        let paged = MtpDraftPolicyWindow {
            kv_state: MtpDraftPolicyKvState::PagedActiveKv,
            ..base
        };

        assert_ne!(base.regime(), long_context.regime());
        assert_ne!(base.regime(), batched.regime());
        assert_ne!(base.regime(), paged.regime());
    }

    #[test]
    fn mtp_cost_aware_policy_uses_actual_committed_tokens_and_cache_costs() {
        let cheap = policy_window(2, 2, 200);
        let fewer_commits = policy_window(2, 1, 200);
        let cache_restore = MtpDraftPolicyWindow {
            total_us: 0,
            verify_forward_us: 100,
            mtp_cache_restore_us: 300,
            ..policy_window(2, 2, 0)
        };

        assert_eq!(cheap.gemma4_cost_per_committed_token_us(), 100.0);
        assert_eq!(cheap.qwen_cost_per_committed_token_us(), 100.0);
        assert_eq!(fewer_commits.gemma4_cost_per_committed_token_us(), 200.0);
        assert_eq!(fewer_commits.qwen_cost_per_committed_token_us(), 200.0);
        assert_eq!(cache_restore.gemma4_cost_per_committed_token_us(), 200.0);
        assert_eq!(cache_restore.qwen_cost_per_committed_token_us(), 200.0);
    }

    #[test]
    fn mtp_cost_aware_policy_compares_zero_draft_without_speculative_overhead() {
        let control = MtpDraftPolicyWindow {
            attempted_draft_tokens: 0,
            committed_tokens: 1,
            total_us: 1_000,
            sampling_us: 100,
            ..policy_window(0, 1, 0)
        };

        assert_eq!(control.gemma4_cost_per_committed_token_us(), 100.0);
        assert_eq!(control.qwen_cost_per_committed_token_us(), 1_000.0);
    }

    #[test]
    fn mtp_cost_aware_policy_excludes_cold_ordinary_compile_cost() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        let regime = policy_window(0, 1, 1_000).regime();

        policy.record_cost(regime, 0, 1_000.0, 1.0);
        let cold = policy.cost_estimate(regime, 0).unwrap();
        assert_eq!(cold.samples, 0);

        policy.record_cost(regime, 0, 100.0, 1.0);
        let warm = policy.cost_estimate(regime, 0).unwrap();
        assert_eq!(warm.samples, 1);
        assert_eq!(warm.cost_ewma, 100.0);
    }

    #[test]
    fn qwen_mtp_policy_keeps_the_pre_gemma_control_cost_sampling() {
        let mut policy = QwenMtpDraftPolicyState::new(1);
        let regime = policy_window(0, 1, 1_000).regime();

        policy.record_cost(regime, 0, 1_000.0, 1.0);
        let first = policy.cost_estimate(regime, 0).unwrap();
        assert_eq!(first.samples, 1);
        assert_eq!(first.cost_ewma, 1_000.0);

        policy.record_cost(regime, 0, 100.0, 1.0);
        let second = policy.cost_estimate(regime, 0).unwrap();
        assert_eq!(second.samples, 2);
        assert_eq!(second.cost_ewma, 685.0);
    }

    #[test]
    fn qwen_fixed_draft_depth_scope_disables_adaptation_only_while_armed() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut rejected = policy_window(2, 1, 1_000);
        rejected.accepted_draft_tokens = 0;

        {
            let _fixed = qwen_fixed_mtp_draft_depth_scope();
            let change = policy.observe_window(rejected);
            assert_eq!(change, MtpDraftBudgetChange::default());
            assert_eq!(policy.current_budget(), 2);
        }

        let change = policy.observe_window(rejected);
        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 1);
    }

    #[test]
    fn mtp_cost_aware_policy_snapshot_restores_cost_history() {
        let mut source = Gemma4DrafterPolicyState::new(2);
        source.observe_window(policy_window(2, 2, 200));
        let snapshot = source.snapshot();
        let mut restored = Gemma4DrafterPolicyState::new(2);

        restored.restore_snapshot(snapshot.clone()).unwrap();

        assert_eq!(restored.snapshot(), snapshot);
    }

    #[test]
    fn mtp_cost_aware_policy_keeps_a_cheaper_zero_draft_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES - 1 {
            policy.observe_window(policy_window(1, 2, 200));
        }
        let change = policy.observe_window(policy_window(1, 2, 200));

        assert_eq!(policy.current_budget(), 0);
        assert!(change.reduced);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(policy_window(0, 1, 50));
        }
        assert_eq!(policy.current_budget(), 0);
        assert!(!policy.should_maintain_mtp_cache());
    }

    #[test]
    fn mtp_cost_aware_policy_rejects_a_more_expensive_zero_draft_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            policy.observe_window(policy_window(1, 2, 200));
        }
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS - 1 {
            policy.observe_window(policy_window(0, 1, 99));
        }
        let decision = policy.observe_window(policy_window(0, 1, 99));
        assert!(decision.increased);

        assert_eq!(policy.current_budget(), 1);
        assert!(policy.should_maintain_mtp_cache());
    }

    #[test]
    fn mtp_cost_aware_policy_refreshes_zero_draft_cost_before_switching() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            policy.observe_window(policy_window(1, 2, 200));
        }
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(policy_window(0, 1, 99));
        }
        assert_eq!(policy.current_budget(), 1);

        for _ in 0..16 {
            policy.observe_window(policy_window(1, 2, 300));
            assert_eq!(
                policy.current_budget(),
                1,
                "an old zero-draft sample must not trigger an uncontrolled mode switch"
            );
        }

        let change = policy.observe_window(policy_window(1, 2, 300));
        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS - 1 {
            policy.observe_window(policy_window(0, 1, 200));
        }
        let decision = policy.observe_window(policy_window(0, 1, 200));

        assert!(decision.increased);
        assert_eq!(policy.current_budget(), 1);
    }

    #[test]
    fn mtp_cost_aware_policy_reprobes_ordinary_decode_after_single_draft_rejections() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            policy.observe_window(policy_window(1, 2, 200));
        }
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(policy_window(0, 1, 99));
        }
        assert_eq!(policy.current_budget(), 1);

        let mut rejected = policy_window(1, 1, 300);
        rejected.accepted_draft_tokens = 0;
        for _ in 0..16 {
            let change = policy.observe_window(rejected);
            assert!(!change.reduced);
            assert_eq!(policy.current_budget(), 1);
        }

        let change = policy.observe_window(rejected);
        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
    }

    #[test]
    fn qwen_mtp_policy_keeps_the_pre_gemma_zero_draft_decision_flow() {
        let mut policy = QwenMtpDraftPolicyState::new(1);
        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            policy.observe_window(policy_window(1, 2, 200));
        }
        assert_eq!(policy.current_budget(), 0);
        assert!(policy.should_maintain_mtp_cache());

        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(policy_window(0, 1, 80));
        }

        assert_eq!(policy.current_budget(), 0);
        assert!(policy.uses_ordinary_decode());
        assert!(!policy.should_maintain_mtp_cache());
    }

    #[test]
    fn qwen_mtp_policy_probes_ordinary_decode_early_after_32k() {
        let mut policy = QwenMtpDraftPolicyState::new(1);
        let mut long_window = policy_window(1, 2, 200);
        long_window.context_tokens = 32_769;

        for _ in 0..QwenMtpDraftPolicyState::LONG_CONTEXT_MTP_WARMUP_WINDOWS {
            let transition = policy.observe_window(long_window);
            assert!(!transition.reduced);
            assert_eq!(policy.current_budget(), 1);
        }
        assert!(policy.cost_estimate(long_window.regime(), 1).is_none());
        for _ in 0..QwenMtpDraftPolicyState::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES - 1 {
            let change = policy.observe_window(long_window);
            assert!(!change.reduced);
            assert_eq!(policy.current_budget(), 1);
        }
        let change = policy.observe_window(long_window);

        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        assert!(policy.should_maintain_mtp_cache());
    }

    #[test]
    fn qwen_mtp_policy_requires_a_second_partial_long_context_window() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut long_window = policy_window(2, 2, 200);
        long_window.context_tokens = 32_769;
        long_window.accepted_draft_tokens = 1;

        let first = policy.observe_window(long_window);
        assert!(!first.reduced);
        assert_eq!(policy.current_budget(), 2);
        assert!(policy.cost_estimate(long_window.regime(), 2).is_none());

        let second = policy.observe_window(long_window);
        assert!(second.reduced);
        assert_eq!(policy.current_budget(), 0);
        assert_eq!(policy.probe_budget, Some(0));
        assert_eq!(policy.probe_origin_budget, Some(2));
        assert_eq!(
            policy
                .cost_estimate(long_window.regime(), 2)
                .unwrap()
                .samples,
            1
        );
    }

    #[test]
    fn qwen_mtp_policy_waits_after_partial_then_full_long_context_transition() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut transition = policy_window(2, 2, 200);
        transition.context_tokens = 32_769;
        transition.accepted_draft_tokens = 1;

        let mut steady = policy_window(2, 3, 100);
        steady.context_tokens = 32_769;

        assert!(!policy.observe_window(transition).reduced);
        for _ in 1..QwenMtpDraftPolicyState::LONG_CONTEXT_MTP_WARMUP_WINDOWS {
            assert!(!policy.observe_window(steady).reduced);
        }
        for _ in 0..QwenMtpDraftPolicyState::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES - 1 {
            assert!(!policy.observe_window(steady).reduced);
        }
        assert!(policy.observe_window(steady).reduced);
        assert_eq!(policy.current_budget(), 0);
        assert_eq!(policy.probe_budget, Some(0));
        assert_eq!(policy.probe_origin_budget, Some(2));
    }

    #[test]
    fn qwen_mtp_policy_preserves_short_context_partial_acceptance_rule() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut short_window = policy_window(2, 2, 200);
        short_window.context_tokens = 8_192;
        short_window.accepted_draft_tokens = 1;

        let change = policy.observe_window(short_window);

        assert!(!change.reduced);
        assert_eq!(policy.current_budget(), 2);
    }

    #[test]
    fn qwen_mtp_policy_compares_full_long_context_window_with_ordinary_decode() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut long_window = policy_window(2, 3, 200);
        long_window.context_tokens = 32_769;

        for _ in 0..QwenMtpDraftPolicyState::LONG_CONTEXT_MTP_WARMUP_WINDOWS {
            assert!(!policy.observe_window(long_window).reduced);
        }
        for _ in 0..QwenMtpDraftPolicyState::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES - 1 {
            assert!(!policy.observe_window(long_window).reduced);
        }
        let change = policy.observe_window(long_window);

        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        assert_eq!(policy.probe_budget, Some(0));
        assert_eq!(policy.probe_origin_budget, Some(2));
    }

    #[test]
    fn qwen_mtp_policy_omits_long_context_zero_draft_transition_from_cost() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut mtp_window = policy_window(2, 3, 60);
        mtp_window.context_tokens = 32_769;
        for _ in 0..QwenMtpDraftPolicyState::LONG_CONTEXT_MTP_WARMUP_WINDOWS {
            assert!(!policy.observe_window(mtp_window).reduced);
        }
        for _ in 0..QwenMtpDraftPolicyState::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES - 1 {
            assert!(!policy.observe_window(mtp_window).reduced);
        }
        assert!(policy.observe_window(mtp_window).reduced);

        let regime = mtp_window.regime();
        let mut transition = policy_window(0, 1, 1_000);
        transition.context_tokens = 32_769;
        assert!(!policy.observe_window(transition).reduced);
        assert_eq!(policy.probe_windows_remaining, 4);
        assert!(policy.cost_estimate(regime, 0).is_none());

        let mut steady = policy_window(0, 1, 10);
        steady.context_tokens = 32_769;
        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(steady);
        }

        let control = policy.cost_estimate(regime, 0).unwrap();
        assert_eq!(control.samples, 4);
        assert_eq!(control.cost_ewma, 10.0);
        assert!(policy.uses_ordinary_decode());
        assert!(!policy.should_maintain_mtp_cache());
    }

    #[test]
    fn qwen_mtp_policy_keeps_short_context_depth_after_one_full_window() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut short_window = policy_window(2, 3, 200);
        short_window.context_tokens = 8_192;

        let change = policy.observe_window(short_window);

        assert!(!change.reduced);
        assert_eq!(policy.current_budget(), 2);
    }

    #[test]
    fn qwen_mtp_policy_keeps_mtp_when_ordinary_decode_is_not_cheaper() {
        let mut policy = QwenMtpDraftPolicyState::new(1);
        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            policy.observe_window(policy_window(1, 2, 80));
        }
        assert_eq!(policy.current_budget(), 0);

        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(policy_window(0, 1, 200));
        }

        assert_eq!(policy.current_budget(), 1);
        assert!(!policy.uses_ordinary_decode());
    }

    #[test]
    fn qwen_mtp_policy_snapshot_preserves_long_context_mtp_warmup() {
        let mut source = QwenMtpDraftPolicyState::new(2);
        let mut transition = policy_window(2, 2, 200);
        transition.context_tokens = 32_769;
        transition.accepted_draft_tokens = 1;
        assert!(!source.observe_window(transition).reduced);
        let snapshot = source.snapshot();

        let mut restored = QwenMtpDraftPolicyState::new(2);
        restored.restore_snapshot(snapshot.clone()).unwrap();

        assert_eq!(restored.snapshot(), snapshot);
        let mut steady = policy_window(2, 3, 100);
        steady.context_tokens = 32_769;
        for _ in 1..(QwenMtpDraftPolicyState::LONG_CONTEXT_MTP_WARMUP_WINDOWS
            + QwenMtpDraftPolicyState::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES)
        {
            source.observe_window(steady);
            restored.observe_window(steady);
            assert_eq!(restored.snapshot(), source.snapshot());
        }
        assert_eq!(restored.current_budget(), 0);
    }

    #[test]
    fn qwen_mtp_policy_snapshot_preserves_zero_draft_probe() {
        let mut source = QwenMtpDraftPolicyState::new(1);
        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            source.observe_window(policy_window(1, 2, 200));
        }
        source.observe_window(policy_window(0, 1, 80));
        let snapshot = source.snapshot();

        let mut restored = QwenMtpDraftPolicyState::new(1);
        restored.restore_snapshot(snapshot.clone()).unwrap();

        assert_eq!(restored.snapshot(), snapshot);
        for _ in 1..QwenMtpDraftPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            let window = policy_window(0, 1, 80);
            source.observe_window(window);
            restored.observe_window(window);
        }
        assert_eq!(restored.snapshot(), source.snapshot());
        assert!(restored.uses_ordinary_decode());
    }
}

// Keep legacy execution-driver import paths while computations live in lm.
pub(crate) use ironmlx_lm::core::speculative_ops::{
    commit_mtp_cache_hidden_prefix, commit_mtp_cache_hidden_tail,
    layer_cache_supports_accepted_prefix_trim, resolve_exact_deterministic_target_logits,
    resolve_exact_deterministic_target_tokens, restore_layer_cache,
    rollback_main_cache_to_accepted_prefix, sample_draft_logits_position,
    sample_draft_logits_position_with_uniform, sample_logits_positions, slice_hidden_position,
    slice_position_ids_prefix, split_speculative_draft_prng,
    trim_full_layer_cache_rows_to_accepted_prefix, verify_input, zero_hidden_like_position,
    DraftTokenDistribution, ExactSamplingCounters, MainCacheRollbackInput,
};
pub use ironmlx_lm::core::speculative_ops::{resolve_speculative_tokens, SpeculativeResolution};
