//! Speculative decoding helpers shared by MTP generation paths.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};
use serde::{Deserialize, Serialize};

use crate::core::cache::{MtpCache, MtpCacheSnapshot};
use crate::core::generate::{build_position_ids, GenerateEvent, GenerateRequest};
use crate::core::tokenizer::{DecodeStream, Tokenizer};
use crate::core::{Loader, Model, Sampler};
use crate::models::{Qwen35Model, Qwen35MoeModel, Qwen35MoeMtp, Qwen36MoeModel};
use crate::nn::{enable_turboquant_kv_caches, LayerCache, LayerCacheSnapshot, Mtp, MtpStepOutput};
use crate::Result;

/// Runtime limits for a single-request MTP speculative generation stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpSpeculativeConfig {
    pub max_draft_tokens: usize,
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

pub fn default_mtp_draft_tokens_for_config(_raw_config: &serde_json::Value) -> usize {
    // Cap 2 remains available through explicit runtime configuration and
    // scheduler profiles, but is not safe as an unconditional default across
    // Gemma4 context and batch regimes.
    1
}

pub fn effective_mtp_draft_tokens_for_paged_prefix(
    draft_tokens: usize,
    paged_prefix_cache_enabled: bool,
) -> usize {
    let draft_tokens = draft_tokens.max(1);
    if paged_prefix_cache_enabled {
        1
    } else {
        draft_tokens
    }
}

impl MtpSpeculativeConfig {
    pub fn new(max_draft_tokens: usize, sampler: Sampler) -> Result<Self> {
        if max_draft_tokens == 0 {
            return Err(anyhow!(
                "MtpSpeculativeConfig::new: max_draft_tokens must be > 0"
            ));
        }
        if !sampler.is_pipelinable() {
            return Err(anyhow!(
                "MtpSpeculativeConfig::new: MTP speculative decoding currently requires greedy sampling"
            ));
        }
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

/// Narrow model capability required by single-request MTP speculative decoding.
pub trait MtpSpeculativeModel: Model {
    type MtpHead;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead>;

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache>;

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32;

    fn mtp_hidden_dtype(&self, mtp: &Self::MtpHead) -> Dtype;

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Model::project_hidden_on(self, hidden, target.into())
    }

    #[allow(clippy::too_many_arguments)]
    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array>;

    #[allow(clippy::too_many_arguments)]
    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput>;
}

impl MtpSpeculativeModel for Qwen35Model {
    type MtpHead = Mtp;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead> {
        Qwen35Model::load_mtp_head(self, loader)
    }

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache> {
        let layer_cfg = mtp.config().layer;
        MtpCache::new_with_cap(
            mtp.num_layers(),
            batch,
            layer_cfg.num_kv_heads,
            layer_cfg.head_dim,
            layer_cfg.head_dim,
            dtype,
            cap,
        )
    }

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35Model::project_mtp_verify_hidden_on(self, hidden, target)
    }

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32 {
        mtp.config().hidden_size
    }

    fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
        self.hidden_dtype()
    }

    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        Qwen35Model::mtp_forward_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35Model::mtp_forward_hidden_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }
}

impl MtpSpeculativeModel for Qwen35MoeModel {
    type MtpHead = Qwen35MoeMtp;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead> {
        Qwen35MoeModel::load_mtp_head(self, loader)
    }

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache> {
        let layer_cfg = mtp.config().layer;
        MtpCache::new_with_cap(
            mtp.num_layers(),
            batch,
            layer_cfg.num_kv_heads,
            layer_cfg.head_dim,
            layer_cfg.head_dim,
            dtype,
            cap,
        )
    }

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35MoeModel::project_mtp_verify_hidden_on(self, hidden, target)
    }

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32 {
        mtp.config().hidden_size
    }

    fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
        self.hidden_dtype()
    }

    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        Qwen35MoeModel::mtp_forward_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35MoeModel::mtp_forward_hidden_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }
}

impl MtpSpeculativeModel for Qwen36MoeModel {
    type MtpHead = Qwen35MoeMtp;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead> {
        Qwen36MoeModel::load_mtp_head(self, loader)
    }

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache> {
        let layer_cfg = mtp.config().layer;
        MtpCache::new_with_cap(
            mtp.num_layers(),
            batch,
            layer_cfg.num_kv_heads,
            layer_cfg.head_dim,
            layer_cfg.head_dim,
            dtype,
            cap,
        )
    }

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen36MoeModel::project_mtp_verify_hidden_on(self, hidden, target)
    }

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32 {
        mtp.config().hidden_size
    }

    fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
        self.hidden_dtype()
    }

    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        Qwen36MoeModel::mtp_forward_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen36MoeModel::mtp_forward_hidden_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }
}

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

/// Outcome of comparing MTP draft tokens with the main model's verified tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeResolution {
    /// Number of MTP draft tokens accepted before the first mismatch.
    pub accepted_draft_len: usize,
    /// Tokens that should be appended to generation history:
    /// accepted draft tokens plus either the corrected token or the bonus token.
    pub tokens_to_append: Vec<u32>,
    /// Number of verify input tokens that must remain in the main KV cache.
    ///
    /// The verify input is `[current_token] + draft_tokens`; keeping
    /// `accepted_draft_len + 1` positions preserves the current token and the
    /// accepted draft prefix.
    pub accepted_verify_input_len: usize,
    /// Whether the caller must rollback the main KV cache after a full-window
    /// verify pass.
    pub needs_rollback: bool,
}

pub fn resolve_speculative_tokens(
    draft_tokens: &[u32],
    verified_tokens: &[u32],
) -> Result<SpeculativeResolution> {
    if verified_tokens.len() != draft_tokens.len() + 1 {
        return Err(anyhow!(
            "resolve_speculative_tokens: verified tokens len {} != draft len {} + 1",
            verified_tokens.len(),
            draft_tokens.len()
        ));
    }

    let accepted_draft_len = draft_tokens
        .iter()
        .zip(verified_tokens.iter())
        .take_while(|(draft, verified)| draft == verified)
        .count();
    let mut tokens_to_append = Vec::with_capacity(accepted_draft_len + 1);
    tokens_to_append.extend_from_slice(&draft_tokens[..accepted_draft_len]);
    tokens_to_append.push(verified_tokens[accepted_draft_len]);
    let accepted_verify_input_len = accepted_draft_len + 1;
    let needs_rollback = accepted_draft_len < draft_tokens.len();

    Ok(SpeculativeResolution {
        accepted_draft_len,
        tokens_to_append,
        accepted_verify_input_len,
        needs_rollback,
    })
}

#[derive(Debug)]
pub(crate) struct MtpDraftResult {
    pub tokens: Vec<u32>,
    pub cache_snapshot: MtpCacheSnapshot,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MtpDraftPolicyWindow {
    pub attempted_draft_tokens: usize,
    pub accepted_draft_tokens: usize,
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

impl MtpDraftPolicyWindow {
    pub(crate) fn from_stats_delta(
        attempted_draft_tokens: usize,
        accepted_draft_tokens: usize,
        delta: &MtpSpeculativeStats,
    ) -> Self {
        Self {
            attempted_draft_tokens,
            accepted_draft_tokens,
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

    fn non_verify_overhead_us(self) -> u64 {
        self.draft_forward_us
            .saturating_add(self.projection_us)
            .saturating_add(self.sampling_us)
            .saturating_add(self.verify_accept_host_sync_us)
            .saturating_add(self.main_rollback_us)
            .saturating_add(self.mtp_decode_cache_commit_us)
            .saturating_add(self.mtp_cache_restore_us)
    }

    fn acceptance_rate(self) -> f64 {
        if self.attempted_draft_tokens == 0 {
            1.0
        } else {
            self.accepted_draft_tokens.min(self.attempted_draft_tokens) as f64
                / self.attempted_draft_tokens as f64
        }
    }

    fn overhead_ratio(self) -> f64 {
        self.non_verify_overhead_us() as f64 / self.verify_forward_us.max(1) as f64
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MtpDraftBudgetChange {
    pub reduced: bool,
    pub increased: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct MtpDraftPolicyState {
    max_draft_tokens: usize,
    current_budget: usize,
    acceptance_ewma: Option<f64>,
    overhead_ratio_ewma: Option<f64>,
    cooldown_windows: usize,
}

impl MtpDraftPolicyState {
    const EWMA_ALPHA: f64 = 0.35;
    const HIGH_OVERHEAD_RATIO: f64 = 1.50;
    const VERY_HIGH_OVERHEAD_RATIO: f64 = 3.00;
    const LOW_ACCEPTANCE: f64 = 0.50;
    const HIGH_ACCEPTANCE: f64 = 0.85;
    const MIN_POSITION_ATTEMPTS: usize = 4;
    const LOW_POSITION_ACCEPTANCE: f64 = 0.45;

    pub(crate) fn new(max_draft_tokens: usize) -> Self {
        let max_draft_tokens = max_draft_tokens.max(1);
        Self {
            max_draft_tokens,
            current_budget: max_draft_tokens,
            acceptance_ewma: None,
            overhead_ratio_ewma: None,
            cooldown_windows: 0,
        }
    }

    pub(crate) fn current_budget(&self) -> usize {
        self.current_budget.clamp(1, self.max_draft_tokens)
    }

    pub(crate) fn observe_window(
        &mut self,
        window: MtpDraftPolicyWindow,
        cumulative_stats: &MtpSpeculativeStats,
    ) -> MtpDraftBudgetChange {
        let old = self.current_budget();
        if self.max_draft_tokens <= 1 || window.attempted_draft_tokens == 0 {
            self.current_budget = self.max_draft_tokens.max(1);
            return budget_change(old, self.current_budget);
        }

        let acceptance = window.acceptance_rate();
        let overhead_ratio = window.overhead_ratio();
        self.acceptance_ewma = Some(update_ewma(
            self.acceptance_ewma,
            acceptance,
            Self::EWMA_ALPHA,
        ));
        self.overhead_ratio_ewma = Some(update_ewma(
            self.overhead_ratio_ewma,
            overhead_ratio,
            Self::EWMA_ALPHA,
        ));

        let mut next = if self.cooldown_windows > 0 {
            self.cooldown_windows -= 1;
            if window.accepted_draft_tokens == window.attempted_draft_tokens
                && overhead_ratio <= Self::HIGH_OVERHEAD_RATIO
            {
                old.saturating_add(1)
            } else {
                1
            }
        } else if window.accepted_draft_tokens == window.attempted_draft_tokens {
            if overhead_ratio > Self::HIGH_OVERHEAD_RATIO {
                old
            } else {
                old.saturating_add(1)
            }
        } else {
            window.accepted_draft_tokens.saturating_add(1)
        };

        let smoothed_acceptance = self.acceptance_ewma.unwrap_or(acceptance);
        let smoothed_overhead_ratio = self.overhead_ratio_ewma.unwrap_or(overhead_ratio);
        if smoothed_acceptance < Self::LOW_ACCEPTANCE
            && smoothed_overhead_ratio > Self::HIGH_OVERHEAD_RATIO
        {
            next = next.min(old.saturating_sub(1).max(1));
        }
        if acceptance < Self::LOW_ACCEPTANCE && overhead_ratio > Self::VERY_HIGH_OVERHEAD_RATIO {
            next = 1;
            self.cooldown_windows = self.cooldown_windows.max(1);
        }
        if smoothed_acceptance >= Self::HIGH_ACCEPTANCE
            && smoothed_overhead_ratio <= Self::HIGH_OVERHEAD_RATIO
            && window.accepted_draft_tokens == window.attempted_draft_tokens
        {
            next = next.max(old.saturating_add(1));
        }
        if let Some(position_cap) = position_acceptance_budget_cap(cumulative_stats) {
            next = next.min(position_cap);
        }

        self.current_budget = next.clamp(1, self.max_draft_tokens);
        budget_change(old, self.current_budget)
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

fn position_acceptance_budget_cap(stats: &MtpSpeculativeStats) -> Option<usize> {
    stats
        .draft_attempts_by_position
        .iter()
        .zip(stats.draft_accepts_by_position.iter())
        .enumerate()
        .find_map(|(idx, (&attempts, &accepts))| {
            if attempts < MtpDraftPolicyState::MIN_POSITION_ATTEMPTS {
                return None;
            }
            let acceptance = accepts as f64 / attempts.max(1) as f64;
            (acceptance < MtpDraftPolicyState::LOW_POSITION_ACCEPTANCE).then_some(idx.max(1))
        })
}

pub(crate) fn adjust_mtp_draft_budget(
    max_draft_tokens: usize,
    adaptive_draft_tokens: &mut usize,
    attempted_draft_tokens: usize,
    accepted_draft_tokens: usize,
    stats: &mut MtpSpeculativeStats,
) {
    if max_draft_tokens <= 1 || attempted_draft_tokens == 0 {
        *adaptive_draft_tokens = max_draft_tokens.max(1);
        return;
    }
    let old = (*adaptive_draft_tokens).clamp(1, max_draft_tokens);
    let next = if accepted_draft_tokens == attempted_draft_tokens {
        old.saturating_add(1).min(max_draft_tokens)
    } else {
        accepted_draft_tokens
            .saturating_add(1)
            .clamp(1, max_draft_tokens)
    };
    if next < old {
        stats.draft_budget_reductions = stats.draft_budget_reductions.saturating_add(1);
    } else if next > old {
        stats.draft_budget_increases = stats.draft_budget_increases.saturating_add(1);
    }
    *adaptive_draft_tokens = next;
}

pub(crate) fn zero_hidden_like_position(hidden: &Array) -> Result<Array> {
    let shape = hidden.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(anyhow!(
            "zero_hidden_like_position: expected hidden shape [1, S, H], got {:?}",
            dims
        ));
    }
    Array::zeros((1_i32, 1_i32, dims[2]), hidden.dtype()).map_err(anyhow::Error::from)
}

pub(crate) fn shift_hidden_for_mtp(
    prev_hidden: &Array,
    hidden: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let target = target.into();
    let prev_shape = prev_hidden.shape();
    let prev_dims = prev_shape.as_slice();
    let hidden_shape = hidden.shape();
    let hidden_dims = hidden_shape.as_slice();
    if prev_dims.len() != 3 || prev_dims[0] != 1 || prev_dims[1] != 1 {
        return Err(anyhow!(
            "shift_hidden_for_mtp: expected prev_hidden shape [1, 1, H], got {:?}",
            prev_dims
        ));
    }
    if hidden_dims.len() != 3 || hidden_dims[0] != 1 {
        return Err(anyhow!(
            "shift_hidden_for_mtp: expected hidden shape [1, S, H], got {:?}",
            hidden_dims
        ));
    }
    let seq = hidden_dims[1];
    let hidden_size = hidden_dims[2];
    if prev_dims[2] != hidden_size {
        return Err(anyhow!(
            "shift_hidden_for_mtp: prev hidden size {} != hidden size {}",
            prev_dims[2],
            hidden_size
        ));
    }
    if seq == 1 {
        return Ok(prev_hidden.clone());
    }
    let prefix = mlx::ops::indexing::slice_strided_on(
        hidden,
        &[0_i32, 0_i32, 0_i32][..],
        &[1_i32, seq - 1, hidden_size][..],
        &[1_i32, 1_i32, 1_i32][..],
        target,
    )?;
    mlx::ops::shape::concatenate_on(&[prev_hidden, &prefix], 1, target).map_err(anyhow::Error::from)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_mtp_cache_hidden_prefix<M>(
    model: &M,
    mtp: &M::MtpHead,
    mtp_cache: &mut MtpCache,
    prev_hidden: &Array,
    input_tokens: &[u32],
    input_hidden: &Array,
    position_ids: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<()>
where
    M: MtpSpeculativeModel,
{
    if input_tokens.is_empty() {
        return Ok(());
    }
    let target = target.into();
    let hidden_shape = input_hidden.shape();
    let hidden_dims = hidden_shape.as_slice();
    if hidden_dims.len() != 3 || hidden_dims[0] != 1 || hidden_dims[1] != input_tokens.len() as i32
    {
        return Err(anyhow!(
            "commit_mtp_cache_hidden_prefix: hidden shape {:?} does not match {} input tokens",
            hidden_dims,
            input_tokens.len()
        ));
    }
    let shifted_hidden = shift_hidden_for_mtp(prev_hidden, input_hidden, target)?;
    let token_arr: Array = (input_tokens, &[1_i32, input_tokens.len() as i32][..]).try_into()?;
    let mtp_hidden = model.mtp_forward_hidden_on(
        mtp,
        &shifted_hidden,
        &token_arr,
        position_ids,
        None,
        Some(mtp_cache),
        target,
    )?;
    mlx::transforms::eval(&[&mtp_hidden])?;
    Ok(())
}

fn slice_position_ids_position(position_ids: &Array, pos: i32) -> Result<Array> {
    let shape = position_ids.shape();
    let dims = shape.as_slice();
    match dims {
        [1, seq] => {
            if *seq == 1 {
                return Ok(position_ids.clone());
            }
            if pos < 0 || pos >= *seq {
                return Err(anyhow!(
                    "slice_position_ids_position: pos {pos} out of [0, {seq})"
                ));
            }
            mlx::ops::indexing::slice_strided(
                position_ids,
                &[0_i32, pos][..],
                &[1_i32, pos + 1][..],
                &[1_i32, 1_i32][..],
            )
            .map_err(anyhow::Error::from)
        }
        [planes, 1, seq] => {
            if *seq == 1 {
                return Ok(position_ids.clone());
            }
            if pos < 0 || pos >= *seq {
                return Err(anyhow!(
                    "slice_position_ids_position: pos {pos} out of [0, {seq})"
                ));
            }
            mlx::ops::indexing::slice_strided(
                position_ids,
                &[0_i32, 0_i32, pos][..],
                &[*planes, 1_i32, pos + 1][..],
                &[1_i32, 1_i32, 1_i32][..],
            )
            .map_err(anyhow::Error::from)
        }
        _ => Err(anyhow!(
            "slice_position_ids_position: expected position_ids shape [1, S] or [P, 1, S], got {:?}",
            dims
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_mtp_cache_hidden_tail<M>(
    model: &M,
    mtp: &M::MtpHead,
    mtp_cache: &mut MtpCache,
    prev_hidden: &Array,
    input_tokens: &[u32],
    input_hidden: &Array,
    position_ids: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<()>
where
    M: MtpSpeculativeModel,
{
    if input_tokens.is_empty() {
        return Ok(());
    }
    let tail_idx = input_tokens.len() - 1;
    let tail_prev_hidden = if tail_idx == 0 {
        prev_hidden.clone()
    } else {
        slice_hidden_position(input_hidden, tail_idx as i32 - 1)?
    };
    let tail_hidden = slice_hidden_position(input_hidden, tail_idx as i32)?;
    let tail_position_ids = slice_position_ids_position(position_ids, tail_idx as i32)?;
    commit_mtp_cache_hidden_prefix(
        model,
        mtp,
        mtp_cache,
        &tail_prev_hidden,
        &input_tokens[tail_idx..],
        &tail_hidden,
        &tail_position_ids,
        target,
    )
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
    draft_policy: MtpDraftPolicyState,
    stats: MtpSpeculativeStats,
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
        if !request.sampler.is_pipelinable() {
            return Err(anyhow!(
                "MtpTextGenerationStream::new_text_only: MTP speculative decoding currently requires greedy sampling"
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
            draft_policy: MtpDraftPolicyState::new(cfg.max_draft_tokens),
            stats,
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

        let stats_before_window = self.stats.clone();
        let draft_budget = self
            .adaptive_draft_tokens
            .clamp(1, self.cfg.max_draft_tokens)
            .min(remaining);
        let draft_result = self.draft_tokens(current_token, draft_budget)?;
        let draft_tokens = draft_result.tokens;
        let verify_input = verify_input(current_token, &draft_tokens);
        let verify_start_pos = (self.history.len() - 1) as i32;
        let verify_pos_ids = self.position_ids(verify_start_pos, verify_input.len() as i32)?;
        let verify_arr: Array =
            (&verify_input[..], &[1_i32, verify_input.len() as i32][..]).try_into()?;
        let pre_window_hidden = self.last_hidden.clone();

        let base_snapshot: Vec<LayerCacheSnapshot> =
            self.cache.iter().map(LayerCache::snapshot).collect();
        let verify_forward_start = Instant::now();
        let verified_hidden = {
            let _verify_qmm = crate::nn::verify_qmm::armed_scope();
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
        let resolution = if self.request.sampler.is_pipelinable() {
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
            let sampling_start = Instant::now();
            let verified_tokens = sample_logits_positions(
                &verified_logits,
                self.request.sampler,
                &self.history,
                &mut self.prng_state,
            )?;
            add_elapsed_us(&mut self.stats.sampling_us, sampling_start);
            resolve_speculative_tokens(&draft_tokens, &verified_tokens)?
        };
        self.stats.windows += 1;
        self.stats.drafted_tokens += draft_tokens.len();
        self.stats.accepted_draft_tokens += resolution.accepted_draft_len;
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
                &base_snapshot,
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
        } else {
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

        let stats_delta = self.stats.saturating_delta_since(&stats_before_window);
        let change = self.draft_policy.observe_window(
            MtpDraftPolicyWindow::from_stats_delta(
                draft_tokens.len(),
                resolution.accepted_draft_len,
                &stats_delta,
            ),
            &self.stats,
        );
        if change.reduced {
            self.stats.draft_budget_reductions =
                self.stats.draft_budget_reductions.saturating_add(1);
        } else if change.increased {
            self.stats.draft_budget_increases = self.stats.draft_budget_increases.saturating_add(1);
        }
        self.adaptive_draft_tokens = self.draft_policy.current_budget();

        let mut tokens_to_append = resolution.tokens_to_append;
        if let Some(stop_idx) = tokens_to_append
            .iter()
            .position(|token| self.request.stop_token_ids.contains(token))
        {
            tokens_to_append.truncate(stop_idx + 1);
        }
        tokens_to_append.truncate(remaining);
        for token in tokens_to_append {
            self.history.push(token);
            self.pending_tokens.push_back(token);
        }

        Ok(())
    }

    fn draft_tokens(&mut self, current_token: u32, draft_budget: usize) -> Result<MtpDraftResult> {
        let mtp_snapshot = self.mtp_cache.snapshot();
        let mut draft_tokens = Vec::with_capacity(draft_budget);
        let mut draft_history = self.history.clone();
        let mut input_hidden = self.last_hidden.clone();
        let mut input_token = current_token;
        let start_pos = (self.history.len() - 1) as i32;

        for offset in 0..draft_budget {
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
            let sampling_start = Instant::now();
            let sampled = sample_logits_positions(
                &output.logits,
                self.request.sampler,
                &draft_history,
                &mut self.prng_state,
            )?;
            add_elapsed_us(&mut self.stats.sampling_us, sampling_start);
            let next_token = *sampled
                .first()
                .ok_or_else(|| anyhow!("MTP draft produced no token"))?;
            draft_tokens.push(next_token);
            draft_history.push(next_token);
            input_hidden = output.hidden_states;
            input_token = next_token;
        }

        Ok(MtpDraftResult {
            tokens: draft_tokens,
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

pub(crate) fn verify_input(current_token: u32, draft_tokens: &[u32]) -> Vec<u32> {
    let mut input = Vec::with_capacity(draft_tokens.len() + 1);
    input.push(current_token);
    input.extend_from_slice(draft_tokens);
    input
}

pub(crate) fn sample_logits_positions(
    logits: &Array,
    sampler: Sampler,
    history: &[u32],
    prng_state: &mut Array,
) -> Result<Vec<u32>> {
    let shape = logits.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(anyhow!(
            "sample_logits_positions: expected logits shape [1, S, V], got {:?}",
            dims
        ));
    }
    let seq = dims[1];
    let vocab = dims[2];
    if sampler.is_pipelinable() {
        let ids = mlx::ops::reduction::argmax(logits, -1, false)?;
        let tokens: Vec<u32> = ids.to_vec()?;
        if tokens.len() != seq as usize {
            return Err(anyhow!(
                "sample_logits_positions: greedy argmax returned {} tokens, expected {}",
                tokens.len(),
                seq
            ));
        }
        return Ok(tokens);
    }
    let mut sampled = Vec::with_capacity(seq as usize);
    let mut running_history = history.to_vec();
    for pos in 0..seq {
        let row = mlx::ops::indexing::slice(
            logits,
            &[0_i32, pos, 0_i32][..],
            &[1_i32, pos + 1, vocab][..],
        )?;
        let row = row.reshape((vocab,))?;
        let token = sampler.sample(&row, &running_history, prng_state)?;
        running_history.push(token);
        sampled.push(token);
    }
    Ok(sampled)
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

pub(crate) fn slice_hidden_position(hidden: &Array, pos: i32) -> Result<Array> {
    let shape = hidden.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(anyhow!(
            "slice_hidden_position: expected hidden shape [1, S, H], got {:?}",
            dims
        ));
    }
    let seq = dims[1];
    let hidden_size = dims[2];
    if pos < 0 || pos >= seq {
        return Err(anyhow!(
            "slice_hidden_position: pos {pos} out of [0, {seq})"
        ));
    }
    mlx::ops::indexing::slice_strided(
        hidden,
        &[0_i32, pos, 0_i32][..],
        &[1_i32, pos + 1, hidden_size][..],
        &[1_i32, 1_i32, 1_i32][..],
    )
    .map_err(anyhow::Error::from)
}

pub(crate) fn slice_hidden_prefix(hidden: &Array, len: usize) -> Result<Array> {
    let shape = hidden.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(anyhow!(
            "slice_hidden_prefix: expected hidden shape [1, S, H], got {:?}",
            dims
        ));
    }
    if len == 0 || len > dims[1] as usize {
        return Err(anyhow!(
            "slice_hidden_prefix: len {len} out of [1, {}]",
            dims[1]
        ));
    }
    if len == dims[1] as usize {
        return Ok(hidden.clone());
    }
    mlx::ops::indexing::slice_strided(
        hidden,
        &[0_i32, 0_i32, 0_i32][..],
        &[1_i32, len as i32, dims[2]][..],
        &[1_i32, 1_i32, 1_i32][..],
    )
    .map_err(anyhow::Error::from)
}

pub(crate) fn slice_position_ids_prefix(position_ids: &Array, len: usize) -> Result<Array> {
    let shape = position_ids.shape();
    let dims = shape.as_slice();
    if len == 0 {
        return Err(anyhow!("slice_position_ids_prefix: len must be > 0"));
    }
    match dims {
        [1, seq] => {
            if len > *seq as usize {
                return Err(anyhow!(
                    "slice_position_ids_prefix: len {len} exceeds position_ids seq {seq}"
                ));
            }
            if len == *seq as usize {
                return Ok(position_ids.clone());
            }
            mlx::ops::indexing::slice_strided(
                position_ids,
                &[0_i32, 0_i32][..],
                &[1_i32, len as i32][..],
                &[1_i32, 1_i32][..],
            )
            .map_err(anyhow::Error::from)
        }
        [planes, 1, seq] => {
            if len > *seq as usize {
                return Err(anyhow!(
                    "slice_position_ids_prefix: len {len} exceeds position_ids seq {seq}"
                ));
            }
            if len == *seq as usize {
                return Ok(position_ids.clone());
            }
            mlx::ops::indexing::slice_strided(
                position_ids,
                &[0_i32, 0_i32, 0_i32][..],
                &[*planes, 1_i32, len as i32][..],
                &[1_i32, 1_i32, 1_i32][..],
            )
            .map_err(anyhow::Error::from)
        }
        _ => Err(anyhow!(
            "slice_position_ids_prefix: expected position_ids shape [1, S] or [P, 1, S], got {:?}",
            dims
        )),
    }
}

pub(crate) fn restore_layer_cache(
    cache: &mut [LayerCache],
    snapshots: &[LayerCacheSnapshot],
) -> Result<()> {
    if cache.len() != snapshots.len() {
        return Err(anyhow!(
            "restore_layer_cache: cache layers {} != snapshot layers {}",
            cache.len(),
            snapshots.len()
        ));
    }
    for (layer, snapshot) in cache.iter_mut().zip(snapshots.iter()) {
        layer.restore(snapshot)?;
    }
    Ok(())
}

pub(crate) fn layer_cache_supports_accepted_prefix_trim(cache: &[LayerCache]) -> bool {
    cache
        .iter()
        .all(|layer| matches!(layer, LayerCache::Full(_)))
}

pub(crate) fn trim_full_layer_cache_rows_to_accepted_prefix(
    cache: &mut [LayerCache],
    snapshots: &[LayerCacheSnapshot],
    accepted_by_row: &[(usize, usize)],
) -> Result<()> {
    if cache.len() != snapshots.len() {
        return Err(anyhow!(
            "trim_full_layer_cache_rows_to_accepted_prefix: cache layers {} != snapshot layers {}",
            cache.len(),
            snapshots.len()
        ));
    }
    if accepted_by_row.is_empty() {
        return Ok(());
    }

    for (layer_idx, (layer, snapshot)) in cache.iter_mut().zip(snapshots.iter()).enumerate() {
        let (LayerCache::Full(kv), LayerCacheSnapshot::Full(saved)) = (layer, snapshot) else {
            return Err(anyhow!(
                "trim_full_layer_cache_rows_to_accepted_prefix: accepted-prefix trim only supports Full KV layers, layer {layer_idx}"
            ));
        };
        let mut offsets = kv.offsets().to_vec();
        for &(row, accepted_len) in accepted_by_row {
            let base = *saved.offsets().get(row).ok_or_else(|| {
                anyhow!(
                    "trim_full_layer_cache_rows_to_accepted_prefix: row {row} out of snapshot offsets for layer {layer_idx}"
                )
            })?;
            let live = offsets.get_mut(row).ok_or_else(|| {
                anyhow!(
                    "trim_full_layer_cache_rows_to_accepted_prefix: row {row} out of live offsets for layer {layer_idx}"
                )
            })?;
            let accepted_len = i32::try_from(accepted_len).map_err(|_| {
                anyhow!(
                    "trim_full_layer_cache_rows_to_accepted_prefix: accepted_len {accepted_len} exceeds i32"
                )
            })?;
            let target = base.checked_add(accepted_len).ok_or_else(|| {
                anyhow!(
                    "trim_full_layer_cache_rows_to_accepted_prefix: base {base} + accepted_len {accepted_len} overflow"
                )
            })?;
            if target > *live {
                return Err(anyhow!(
                    "trim_full_layer_cache_rows_to_accepted_prefix: target offset {target} exceeds live offset {} for row {row} layer {layer_idx}",
                    *live
                ));
            }
            *live = target;
        }
        kv.restore_offsets(&offsets)?;
    }
    Ok(())
}

pub(crate) struct MainCacheRollbackInput<'a> {
    pub(crate) accepted_by_row: &'a [(usize, usize)],
    pub(crate) verify_input: &'a [u32],
    pub(crate) accepted_position_ids: &'a Array,
    pub(crate) verified_hidden: &'a Array,
}

pub(crate) fn rollback_main_cache_to_accepted_prefix<M: MtpSpeculativeModel>(
    model: &M,
    cache: &mut [LayerCache],
    snapshots: &[LayerCacheSnapshot],
    input: MainCacheRollbackInput<'_>,
    target: impl Into<mlx::StreamOrDevice>,
) -> Result<Array> {
    if input.accepted_by_row.len() != 1 || input.accepted_by_row[0].0 != 0 {
        return Err(anyhow!(
            "rollback_main_cache_to_accepted_prefix: single-row helper got accepted_by_row={:?}",
            input.accepted_by_row
        ));
    }
    let accepted_len = input.accepted_by_row[0].1;
    if accepted_len == 0 || accepted_len > input.verify_input.len() {
        return Err(anyhow!(
            "rollback_main_cache_to_accepted_prefix: accepted_len {accepted_len} outside [1, {}]",
            input.verify_input.len()
        ));
    }

    if layer_cache_supports_accepted_prefix_trim(cache) {
        trim_full_layer_cache_rows_to_accepted_prefix(cache, snapshots, input.accepted_by_row)?;
        return slice_hidden_prefix(input.verified_hidden, accepted_len);
    }

    restore_layer_cache(cache, snapshots)?;
    let accepted_arr: Array = (
        &input.verify_input[..accepted_len],
        &[1_i32, accepted_len as i32][..],
    )
        .try_into()?;
    let _verify_qmm = crate::nn::verify_qmm::armed_scope();
    model.forward_text_hidden(
        &accepted_arr,
        input.accepted_position_ids,
        None,
        None,
        Some(cache),
        target.into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cache::{KVCache, TurboQuantKVBits};
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

        fn replay_calls(&self) -> usize {
            self.replay_calls.load(Ordering::Relaxed)
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

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
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
    fn mtp_rollback_main_cache_replays_hybrid_cache_after_mismatch() {
        let model = FakeGreedyProjectModel::new(Vec::new());
        let mut cache = vec![LayerCache::Linear(
            crate::core::cache::GatedDeltaCache::new_with_cap(1, 4, 8, 1, 4, 4, Dtype::Float32, 16)
                .expect("linear cache"),
        )];
        if let LayerCache::Linear(gd) = &mut cache[0] {
            gd.advance(&[4]).expect("base prefix");
        }
        let snapshots = cache.iter().map(LayerCache::snapshot).collect::<Vec<_>>();
        if let LayerCache::Linear(gd) = &mut cache[0] {
            gd.advance(&[3]).expect("verified suffix");
        }

        let verify_input = vec![10_u32, 11, 12];
        let accepted_position_ids =
            crate::core::generate::build_position_ids(4, 2).expect("position ids");
        let verified_hidden =
            Array::zeros((1_i32, 3_i32, 1_i32), Dtype::Float32).expect("verified hidden");

        let accepted_hidden = rollback_main_cache_to_accepted_prefix(
            &model,
            &mut cache,
            &snapshots,
            MainCacheRollbackInput {
                accepted_by_row: &[(0, 2)],
                verify_input: &verify_input,
                accepted_position_ids: &accepted_position_ids,
                verified_hidden: &verified_hidden,
            },
            (),
        )
        .expect("hybrid rollback replay");

        assert_eq!(accepted_hidden.shape().as_slice(), &[1, 2, 1]);
        assert_eq!(model.replay_calls(), 1);
        let LayerCache::Linear(gd) = &cache[0] else {
            panic!("expected linear cache");
        };
        assert_eq!(gd.offsets(), &[6]);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn accepted_prefix_trim_supports_paged_kv() {
        let mut kv = KVCache::new(1, 1, 2, 2, Dtype::Float32, 8).with_step(4);
        kv.enable_paged(2, 4).expect("enable paged KV");
        let base_k: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], &[1_i32, 1, 2, 2][..])
            .try_into()
            .unwrap();
        let base_v = &base_k + 100.0_f32;
        kv.update_and_fetch(&base_k, &base_v, &[2])
            .expect("paged base prefix");
        let mut cache = vec![LayerCache::Full(kv)];
        let snapshots = cache.iter().map(LayerCache::snapshot).collect::<Vec<_>>();
        let verify_k: Array = (
            &[5.0_f32, 6.0, 7.0, 8.0, 9.0, 10.0][..],
            &[1_i32, 1, 3, 2][..],
        )
            .try_into()
            .unwrap();
        let verify_v = &verify_k + 100.0_f32;
        let LayerCache::Full(kv) = &mut cache[0] else {
            panic!("full cache");
        };
        kv.update_and_fetch(&verify_k, &verify_v, &[3])
            .expect("paged verify suffix");

        trim_full_layer_cache_rows_to_accepted_prefix(&mut cache, &snapshots, &[(0, 1)])
            .expect("trim paged accepted prefix");

        let LayerCache::Full(kv) = &cache[0] else {
            panic!("full cache");
        };
        assert_eq!(kv.offsets(), &[3]);
        let (keys, values) = kv
            .materialize_current_paged_prefix_on(())
            .expect("materialize trimmed paged prefix");
        assert_eq!(keys.shape().as_slice(), &[1, 1, 3, 2]);
        assert_eq!(values.shape().as_slice(), &[1, 1, 3, 2]);
        assert_eq!(
            keys.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn accepted_prefix_trim_supports_turboquant_kv() {
        let mut kv = KVCache::new(1, 1, 8, 8, Dtype::Float32, 8)
            .with_step(8)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable TurboQuant KV");
        let base_data = (0..16).map(|idx| idx as f32 * 0.1).collect::<Vec<_>>();
        let base_k: Array = (base_data.as_slice(), &[1_i32, 1, 2, 8][..])
            .try_into()
            .unwrap();
        let base_v = &base_k + 1.0_f32;
        kv.update_and_fetch(&base_k, &base_v, &[2])
            .expect("TurboQuant base prefix");
        let mut cache = vec![LayerCache::Full(kv)];
        let snapshots = cache.iter().map(LayerCache::snapshot).collect::<Vec<_>>();
        let verify_data = (0..24)
            .map(|idx| 2.0_f32 + idx as f32 * 0.1)
            .collect::<Vec<_>>();
        let verify_k: Array = (verify_data.as_slice(), &[1_i32, 1, 3, 8][..])
            .try_into()
            .unwrap();
        let verify_v = &verify_k + 1.0_f32;
        let LayerCache::Full(kv) = &mut cache[0] else {
            panic!("full cache");
        };
        kv.update_and_fetch(&verify_k, &verify_v, &[3])
            .expect("TurboQuant verify suffix");

        trim_full_layer_cache_rows_to_accepted_prefix(&mut cache, &snapshots, &[(0, 1)])
            .expect("trim TurboQuant accepted prefix");

        let LayerCache::Full(kv) = &cache[0] else {
            panic!("full cache");
        };
        assert_eq!(kv.offsets(), &[3]);
        let (keys, values, len) = kv
            .dense_prefix_layer_for_row_on(0, ())
            .expect("materialize trimmed TurboQuant prefix");
        assert_eq!(len, 3);
        assert_eq!(keys.shape().as_slice(), &[1, 1, 3, 8]);
        assert_eq!(values.shape().as_slice(), &[1, 1, 3, 8]);
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
    fn mtp_policy_defaults_qwen36_dense_27b_to_d1() {
        let raw = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 64
            }
        });

        assert_eq!(default_mtp_draft_tokens_for_config(&raw), 1);
    }

    #[test]
    fn mtp_policy_defaults_qwen36_moe_35b_a3b_to_d1() {
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

        assert_eq!(default_mtp_draft_tokens_for_config(&raw), 1);
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

    #[test]
    fn mtp_cost_aware_policy_reduces_high_overhead_low_acceptance_window() {
        let mut policy = MtpDraftPolicyState::new(4);
        let mut stats = MtpSpeculativeStats::default();
        stats.record_window_acceptance(4, 0);

        let change = policy.observe_window(
            MtpDraftPolicyWindow {
                attempted_draft_tokens: 4,
                accepted_draft_tokens: 0,
                draft_forward_us: 1_000,
                verify_forward_us: 500,
                projection_us: 600,
                sampling_us: 800,
                verify_accept_host_sync_us: 400,
                main_rollback_us: 200,
                mtp_cache_commit_us: 900,
                mtp_prefill_cache_commit_us: 0,
                mtp_decode_cache_commit_us: 900,
                mtp_cache_restore_us: 300,
            },
            &stats,
        );

        assert_eq!(policy.current_budget(), 1);
        assert!(change.reduced);
    }

    #[test]
    fn mtp_cost_aware_policy_accounts_for_acceptance_host_sync() {
        let mut policy = MtpDraftPolicyState::new(4);
        let mut stats = MtpSpeculativeStats::default();
        stats.record_window_acceptance(4, 0);

        let change = policy.observe_window(
            MtpDraftPolicyWindow {
                attempted_draft_tokens: 4,
                accepted_draft_tokens: 0,
                draft_forward_us: 0,
                verify_forward_us: 500,
                projection_us: 0,
                sampling_us: 0,
                verify_accept_host_sync_us: 2_000,
                main_rollback_us: 0,
                mtp_cache_commit_us: 0,
                mtp_prefill_cache_commit_us: 0,
                mtp_decode_cache_commit_us: 0,
                mtp_cache_restore_us: 0,
            },
            &stats,
        );

        assert_eq!(policy.current_budget(), 1);
        assert!(change.reduced);
    }

    #[test]
    fn mtp_cost_aware_policy_restores_after_cheap_full_accept_windows() {
        let mut policy = MtpDraftPolicyState::new(4);
        let mut stats = MtpSpeculativeStats::default();

        stats.record_window_acceptance(4, 0);
        policy.observe_window(
            MtpDraftPolicyWindow {
                attempted_draft_tokens: 4,
                accepted_draft_tokens: 0,
                draft_forward_us: 1_000,
                verify_forward_us: 500,
                projection_us: 600,
                sampling_us: 800,
                verify_accept_host_sync_us: 400,
                main_rollback_us: 200,
                mtp_cache_commit_us: 900,
                mtp_prefill_cache_commit_us: 0,
                mtp_decode_cache_commit_us: 900,
                mtp_cache_restore_us: 300,
            },
            &stats,
        );
        assert_eq!(policy.current_budget(), 1);

        for _ in 0..4 {
            stats.record_window_acceptance(policy.current_budget(), policy.current_budget());
            let change = policy.observe_window(
                MtpDraftPolicyWindow {
                    attempted_draft_tokens: policy.current_budget(),
                    accepted_draft_tokens: policy.current_budget(),
                    draft_forward_us: 50,
                    verify_forward_us: 1_000,
                    projection_us: 20,
                    sampling_us: 20,
                    verify_accept_host_sync_us: 10,
                    main_rollback_us: 0,
                    mtp_cache_commit_us: 30,
                    mtp_prefill_cache_commit_us: 0,
                    mtp_decode_cache_commit_us: 30,
                    mtp_cache_restore_us: 0,
                },
                &stats,
            );
            assert!(!change.reduced);
        }

        assert_eq!(policy.current_budget(), 4);
    }

    #[test]
    fn mtp_cost_aware_policy_caps_depth_from_position_acceptance() {
        let mut policy = MtpDraftPolicyState::new(4);
        let mut stats = MtpSpeculativeStats::default();
        for _ in 0..6 {
            stats.record_window_acceptance(4, 2);
        }

        let change = policy.observe_window(
            MtpDraftPolicyWindow {
                attempted_draft_tokens: 4,
                accepted_draft_tokens: 4,
                draft_forward_us: 50,
                verify_forward_us: 1_000,
                projection_us: 20,
                sampling_us: 20,
                verify_accept_host_sync_us: 10,
                main_rollback_us: 0,
                mtp_cache_commit_us: 30,
                mtp_prefill_cache_commit_us: 0,
                mtp_decode_cache_commit_us: 30,
                mtp_cache_restore_us: 0,
            },
            &stats,
        );

        assert_eq!(policy.current_budget(), 2);
        assert!(change.reduced);
    }

    #[test]
    fn mtp_cost_aware_policy_keeps_single_budget_for_zero_attempt_window() {
        let mut policy = MtpDraftPolicyState::new(1);
        let stats = MtpSpeculativeStats::default();

        let change = policy.observe_window(MtpDraftPolicyWindow::default(), &stats);

        assert_eq!(policy.current_budget(), 1);
        assert!(!change.reduced);
        assert!(!change.increased);
    }
}
