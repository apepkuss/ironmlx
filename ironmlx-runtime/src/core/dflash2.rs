//! Standalone DFlash2 greedy and exact sampled generation path.
//!
//! This module deliberately does not share the MTP scheduler or MTP stream
//! state machine. The only shared piece is model-agnostic token resolution.

use std::collections::VecDeque;
#[cfg(feature = "tools")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{anyhow, Context};
use mlx::{Array, StreamOrDevice};
use serde::Serialize;
use thiserror::Error;

use crate::core::generation_types::{GenerateEvent, GenerateRequest};
use crate::core::speculative::{
    resolve_exact_deterministic_target_logits, resolve_exact_deterministic_target_tokens,
    sample_logits_positions, ExactSamplingCounters, MtpDraftPolicyWindow, QwenMtpDraftPolicyState,
};
use crate::Result;
use ironmlx_lm::core::cache::prefix_payload::PagedPrefixEntry;
use ironmlx_lm::core::model_input::build_position_ids;
use ironmlx_lm::core::speculative_ops::{resolve_speculative_tokens, SpeculativeResolution};
use {
    ironmlx_core::sampler::prepare_target_tokens_with_uniforms_batch,
    ironmlx_core::sampler::prepare_uniforms, ironmlx_core::sampler::sample_position_keyed_v1_batch,
    ironmlx_core::sampler::PreparedTargetTokenSampling,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DFlash2P2Options {
    /// Zero keeps the stable linear proposal path. Values 1..=15 enable the
    /// bounded best-first tree for B1 execution.
    pub tree_max_nodes: usize,
    /// Opt into versioned position-keyed sampling. Stateful exact sampling
    /// remains the default and is not silently changed.
    pub position_keyed_sampling: bool,
}

impl DFlash2P2Options {
    pub fn validate(self) -> Result<Self> {
        anyhow::ensure!(
            self.tree_max_nodes <= ironmlx_lm::models::dflash2::DFlash2DraftTree::MAX_NODES,
            "DFlash2 tree_max_nodes must be in [0, {}]",
            ironmlx_lm::models::dflash2::DFlash2DraftTree::MAX_NODES
        );
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DFlash2ExecutionOptions {
    pub(crate) block_size: usize,
    pub(crate) p2: DFlash2P2Options,
}
use {
    ironmlx_lm::core::cache::layer::prefix_entry_for_row,
    ironmlx_lm::core::cache::layer::restore_prefix_entry_for_row,
    ironmlx_lm::core::cache::layer::LayerCache,
};
use {
    ironmlx_lm::core::constrained::apply_speculative_token_masks,
    ironmlx_lm::core::constrained::apply_token_mask,
    ironmlx_lm::core::constrained::ConstraintSession,
};
use {ironmlx_lm::core::tokenizer::DecodeStream, ironmlx_lm::core::tokenizer::Tokenizer};
use {
    ironmlx_lm::models::dflash2::DFlash2DraftCache, ironmlx_lm::models::dflash2::DFlash2DraftModel,
    ironmlx_lm::models::dflash2::DFlash2DraftTree, ironmlx_lm::models::dflash2::DFlash2Target,
    ironmlx_lm::models::dflash2::DFlash2TargetForwardMode,
    ironmlx_lm::models::dflash2::DFlash2VerifyCapabilities,
    ironmlx_lm::models::dflash2::DFlash2VerifyPlan,
    ironmlx_lm::models::dflash2::DFlash2VerifyShape,
};

#[cfg(feature = "tools")]
static BATCHED_Q16_QUALIFICATION_ENABLED: AtomicBool = AtomicBool::new(false);

/// Widest DFlash2 proposal block promoted for automatic production use.
/// Wider lanes remain an explicit opt-in until they pass the promotion gate.
pub const DEFAULT_DFLASH2_BLOCK_SIZE_CAP: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2BlockSizeResolution {
    pub checkpoint_block_size: usize,
    pub block_size: usize,
    pub explicit: bool,
}

/// Resolve the runtime proposal width from an optional operator override and
/// the draft checkpoint capability. Automatic selection never promotes a lane
/// wider than the qualified Q8 default; explicit Q16 remains available when
/// the checkpoint supports it.
pub fn resolve_dflash2_block_size(
    requested: Option<usize>,
    checkpoint_block_size: usize,
) -> Result<DFlash2BlockSizeResolution> {
    anyhow::ensure!(
        checkpoint_block_size >= 2,
        "DFlash2 checkpoint block_size must be at least 2, got {checkpoint_block_size}"
    );
    let block_size = if let Some(requested) = requested {
        anyhow::ensure!(
            (2..=16).contains(&requested),
            "DFlash2 runtime block_size {requested} must be in [2, 16]"
        );
        anyhow::ensure!(
            requested <= checkpoint_block_size,
            "DFlash2 runtime block_size {requested} exceeds checkpoint block_size {checkpoint_block_size}"
        );
        requested
    } else {
        checkpoint_block_size.min(DEFAULT_DFLASH2_BLOCK_SIZE_CAP)
    };
    Ok(DFlash2BlockSizeResolution {
        checkpoint_block_size,
        block_size,
        explicit: requested.is_some(),
    })
}

/// Process-local capability guard used only by qualification binaries.
///
/// Production builds do not expose this type because they do not enable the
/// `tools` feature. The single-owner guard also prevents unrelated engines in
/// one qualification process from silently inheriting the experimental lane.
#[cfg(feature = "tools")]
#[must_use = "the guard must be retained for the qualification engine lifetime"]
pub struct BatchedQ16QualificationGuard;

#[cfg(feature = "tools")]
impl Drop for BatchedQ16QualificationGuard {
    fn drop(&mut self) {
        BATCHED_Q16_QUALIFICATION_ENABLED.store(false, Ordering::Release);
    }
}

#[cfg(feature = "tools")]
pub fn enable_batched_q16_qualification() -> Result<BatchedQ16QualificationGuard> {
    BATCHED_Q16_QUALIFICATION_ENABLED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| anyhow!("DFlash2 B2/B4 Q16 qualification is already active"))?;
    Ok(BatchedQ16QualificationGuard)
}

fn batched_q16_qualification_enabled() -> bool {
    #[cfg(feature = "tools")]
    {
        BATCHED_Q16_QUALIFICATION_ENABLED.load(Ordering::Acquire)
    }
    #[cfg(not(feature = "tools"))]
    {
        false
    }
}

fn qualification_verify_capabilities(
    capabilities: DFlash2VerifyCapabilities,
) -> Result<DFlash2VerifyCapabilities> {
    extend_batched_q16_qualification_capabilities(capabilities, batched_q16_qualification_enabled())
}

fn extend_batched_q16_qualification_capabilities(
    mut capabilities: DFlash2VerifyCapabilities,
    enabled: bool,
) -> Result<DFlash2VerifyCapabilities> {
    if !enabled {
        return Ok(capabilities);
    }
    let lane_pack = capabilities
        .lane_kernel_pack
        .as_ref()
        .ok_or_else(|| anyhow!("DFlash2 B2/B4 Q16 qualification requires a lane kernel pack"))?;
    anyhow::ensure!(
        capabilities.profile == "qwen35-affine4" && lane_pack.quant_bits == 4,
        "DFlash2 B2/B4 Q16 qualification requires the Qwen3.8 affine-4 profile"
    );
    for batch_width in 2..=4 {
        for verify_width in 9..=16 {
            if lane_pack.supports(batch_width, verify_width) {
                let shape = DFlash2VerifyShape {
                    batch_width,
                    verify_width,
                };
                if !capabilities.supported_shapes.contains(&shape) {
                    capabilities.supported_shapes.push(shape);
                }
            }
        }
    }
    Ok(capabilities)
}

#[derive(Debug, Clone)]
struct DFlash2PrefixArtifact {
    token_ids: Vec<u32>,
    fingerprint: String,
    target_cache: PagedPrefixEntry,
    context_hidden: Array,
    last_hidden: Array,
    cached_len: i32,
    payload_bytes: usize,
    generation: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DFlash2PrefixCacheSnapshot {
    pub(crate) entries: usize,
    pub(crate) bytes: usize,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) saves: u64,
    pub(crate) evictions: u64,
}

/// Request-independent DFlash2 prefix artifacts.
///
/// DFlash2 cannot reuse the ordinary prefix entry alone: the drafter also
/// needs the target-layer hidden-state tail and the final target hidden state
/// at the exact restored position. The cache therefore owns those values as
/// one atomically inserted, in-memory-only artifact.
pub(crate) struct DFlash2PrefixCache {
    max_bytes: usize,
    total_bytes: usize,
    generation: u64,
    hits: u64,
    misses: u64,
    saves: u64,
    evictions: u64,
    entries: Vec<DFlash2PrefixArtifact>,
}

impl DFlash2PrefixCache {
    pub(crate) fn new(max_bytes: usize) -> Result<Self> {
        anyhow::ensure!(max_bytes > 0, "DFlash2 prefix cache max_bytes must be > 0");
        Ok(Self {
            max_bytes,
            total_bytes: 0,
            generation: 0,
            hits: 0,
            misses: 0,
            saves: 0,
            evictions: 0,
            entries: Vec::new(),
        })
    }

    fn load_longest(
        &mut self,
        prompt_ids: &[u32],
        fingerprint: &str,
    ) -> Option<DFlash2PrefixArtifact> {
        let hit_index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.fingerprint == fingerprint
                    && entry.token_ids.len() <= prompt_ids.len()
                    && prompt_ids.starts_with(&entry.token_ids)
            })
            .max_by_key(|(_, entry)| entry.cached_len)
            .map(|(index, _)| index);
        let Some(hit_index) = hit_index else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        self.hits = self.hits.saturating_add(1);
        let generation = self.next_generation();
        self.entries[hit_index].generation = generation;
        Some(self.entries[hit_index].clone())
    }

    fn insert(
        &mut self,
        token_ids: &[u32],
        fingerprint: &str,
        target_cache: &[LayerCache],
        context_hidden: &Array,
        last_hidden: &Array,
    ) -> Result<bool> {
        let Some((target_cache, cached_len)) = prefix_entry_for_row(target_cache, 0)? else {
            return Ok(false);
        };
        anyhow::ensure!(
            cached_len == i32::try_from(token_ids.len())?,
            "DFlash2 prefix cache target offset {cached_len} != token length {}",
            token_ids.len()
        );
        target_cache.eval()?;
        mlx::transforms::eval(&[context_hidden, last_hidden])?;
        let payload_bytes = target_cache
            .observability_stats(cached_len)
            .payload_bytes
            .saturating_add(array_payload_bytes(context_hidden))
            .saturating_add(array_payload_bytes(last_hidden))
            .saturating_add(token_ids.len().saturating_mul(std::mem::size_of::<u32>()))
            .saturating_add(fingerprint.len());
        if payload_bytes > self.max_bytes {
            return Ok(false);
        }

        if let Some(index) = self.entries.iter().position(|entry| {
            entry.fingerprint == fingerprint && entry.token_ids.as_slice() == token_ids
        }) {
            let previous = self.entries.swap_remove(index);
            self.total_bytes = self.total_bytes.saturating_sub(previous.payload_bytes);
        }
        let generation = self.next_generation();
        self.total_bytes = self.total_bytes.saturating_add(payload_bytes);
        self.entries.push(DFlash2PrefixArtifact {
            token_ids: token_ids.to_vec(),
            fingerprint: fingerprint.to_owned(),
            target_cache,
            context_hidden: context_hidden.clone(),
            last_hidden: last_hidden.clone(),
            cached_len,
            payload_bytes,
            generation,
        });
        self.saves = self.saves.saturating_add(1);
        self.shrink_to(self.max_bytes);
        Ok(true)
    }

    fn invalidate(&mut self, token_ids: &[u32], fingerprint: &str) {
        if let Some(index) = self.entries.iter().position(|entry| {
            entry.fingerprint == fingerprint && entry.token_ids.as_slice() == token_ids
        }) {
            let removed = self.entries.swap_remove(index);
            self.total_bytes = self.total_bytes.saturating_sub(removed.payload_bytes);
            self.evictions = self.evictions.saturating_add(1);
        }
    }

    pub(crate) fn shrink_to(&mut self, target_bytes: usize) -> usize {
        let before = self.total_bytes;
        while self.total_bytes > target_bytes {
            let Some((index, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.generation)
            else {
                break;
            };
            let removed = self.entries.swap_remove(index);
            self.total_bytes = self.total_bytes.saturating_sub(removed.payload_bytes);
            self.evictions = self.evictions.saturating_add(1);
        }
        before.saturating_sub(self.total_bytes)
    }

    pub(crate) fn snapshot(&self) -> DFlash2PrefixCacheSnapshot {
        DFlash2PrefixCacheSnapshot {
            entries: self.entries.len(),
            bytes: self.total_bytes,
            hits: self.hits,
            misses: self.misses,
            saves: self.saves,
            evictions: self.evictions,
        }
    }

    fn next_generation(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        self.generation
    }
}

#[derive(Debug, Error)]
pub enum DFlash2RequestError {
    #[error("DFlash2 is text-only; image inputs are unsupported")]
    VisionUnsupported,
    #[error("DFlash2 prompt_ids cannot be empty")]
    EmptyPrompt,
    #[error("DFlash2 max_new_tokens must be greater than zero")]
    ZeroMaxNewTokens,
    #[error("DFlash2 has not qualified TurboQuant KV caches")]
    TurboQuantUnsupported,
}

#[derive(Debug, Clone, Serialize)]
pub struct DFlash2Metrics {
    pub block_size: usize,
    pub tree_max_nodes: usize,
    pub experimental_tree_profile: bool,
    pub tree_proposal_width: usize,
    pub position_keyed_sampling: bool,
    pub sampled: bool,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub windows: usize,
    pub drafted_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub rollback_count: usize,
    pub ordinary_windows: usize,
    pub tree_windows: usize,
    pub tree_drafted_nodes: usize,
    pub ragged_linear_windows: usize,
    pub tree_to_ragged_switches: usize,
    pub ragged_to_tree_switches: usize,
    pub draft_budget_changes: usize,
    pub current_draft_budget: usize,
    pub adaptive_acceptance_ewma: Option<f64>,
    pub exact_sampling_windows: usize,
    pub exact_acceptance_draws: usize,
    pub exact_residual_corrections: usize,
    pub exact_bonus_samples: usize,
    pub draft_build_us: u64,
    pub draft_schedule_us: u64,
    pub verify_build_us: u64,
    pub projection_build_us: u64,
    pub sampling_us: u64,
    pub verify_schedule_us: u64,
    pub host_sync_us: u64,
    pub rollback_us: u64,
    pub window_us: u64,
    pub prefill_us: u64,
    pub generation_us: u64,
    pub prompt_tps: f64,
    pub generation_tps: f64,
    pub acceptance_rate: f64,
    pub peak_memory_bytes: usize,
}

#[derive(Debug, Clone, Default)]
struct DFlash2Counters {
    windows: usize,
    drafted_tokens: usize,
    accepted_draft_tokens: usize,
    rollback_count: usize,
    ordinary_windows: usize,
    tree_windows: usize,
    tree_drafted_nodes: usize,
    ragged_linear_windows: usize,
    /// Path of the latest tree or ragged linear window (true = ragged).
    last_window_ragged: Option<bool>,
    tree_to_ragged_switches: usize,
    ragged_to_tree_switches: usize,
    draft_budget_changes: usize,
    exact_sampling: ExactSamplingCounters,
    draft_build_us: u64,
    draft_schedule_us: u64,
    verify_build_us: u64,
    projection_build_us: u64,
    sampling_us: u64,
    verify_schedule_us: u64,
    host_sync_us: u64,
    rollback_us: u64,
    window_us: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct DFlash2TensorBatchKey {
    draft_len: usize,
    verify_start: usize,
    context_len: i32,
    draft_processed: i32,
    draft_retained: i32,
    supported_batch_widths: u64,
    sampled: bool,
}

impl DFlash2TensorBatchKey {
    pub(crate) fn is_ordinary_decode(self) -> bool {
        self.draft_len == 0
    }

    pub(crate) fn draft_len(self) -> usize {
        self.draft_len
    }

    pub(crate) fn supported_batch_widths(self) -> u64 {
        self.supported_batch_widths
    }

    pub(crate) fn supports_batch_width(self, batch_width: usize) -> bool {
        batch_width < u64::BITS as usize
            && self.supported_batch_widths & (1_u64 << batch_width) != 0
    }

    pub(crate) fn largest_supported_batch_width(self, limit: usize) -> usize {
        (1..=limit)
            .rev()
            .find(|&batch_width| self.supports_batch_width(batch_width))
            .unwrap_or(1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DFlash2PrefillExecution {
    GenerationStream,
    SchedulerB1,
}

struct DFlash2PrefillContext<'a> {
    execution: DFlash2PrefillExecution,
    /// Use single-graph prefill when the prefix cache restores nothing.
    cold_single_prefill: bool,
    prefix_cache: Option<(&'a mut DFlash2PrefixCache, &'a str)>,
    is_cancelled: Option<&'a dyn Fn() -> bool>,
}

fn experimental_fixed_budget(maximum: usize) -> Result<Option<usize>> {
    let Some(raw) =
        ironmlx_core::m5_profile::setting(ironmlx_core::m5_profile::settings::DFLASH2_FIXED_BUDGET)
    else {
        return Ok(None);
    };
    let value: usize = raw
        .parse()
        .map_err(|_| anyhow!("fixed DFlash2 budget must be a non-negative integer"))?;
    anyhow::ensure!(
        value <= maximum,
        "fixed DFlash2 budget {value} exceeds supported {maximum}"
    );
    Ok(Some(value))
}

/// Text-only, single-request DFlash2 stream with greedy and exact sampled decoding.
pub struct DFlash2TextGenerationStream<'m, M>
where
    M: DFlash2Target,
{
    model: &'m M,
    draft: &'m DFlash2DraftModel,
    target_cache: Vec<LayerCache>,
    draft_cache: DFlash2DraftCache,
    history: Vec<u32>,
    request: GenerateRequest,
    pending_tokens: VecDeque<u32>,
    detok: DecodeStream<'m>,
    pending_context_hidden: Array,
    verify_capabilities: DFlash2VerifyCapabilities,
    draft_policy: QwenMtpDraftPolicyState,
    experimental_fixed_budget: Option<usize>,
    prng_state: Array,
    block_size: usize,
    emitted_new_tokens: usize,
    finished: bool,
    prefill_us: u64,
    generation_started: Instant,
    counters: DFlash2Counters,
    constraint: Option<ConstraintSession>,
    prefix_cache_hit_tokens: usize,
    target_cache_cap: i32,
    p2_options: DFlash2P2Options,
}

/// Token capacity of a ragged batched target cache for rows needing
/// `row_tokens` (the largest row, floored at the GPU-efficient minimum). The
/// actor reserves memory for exactly this capacity before building a group.
pub(crate) fn ragged_batch_cache_cap(row_tokens: impl IntoIterator<Item = usize>) -> usize {
    row_tokens.into_iter().max().unwrap_or(1).max(
        usize::try_from(ironmlx_lm::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF)
            .unwrap_or(usize::MAX),
    )
}

/// Prototype, default off: persistent target cache of a ragged linear batch
/// (rows at different context lengths). Built by copying each row's target
/// cache into one batched cache; scattered back when membership changes.
#[doc(hidden)]
pub struct DFlash2RaggedBatchCache {
    target: Vec<LayerCache>,
    batch_size: usize,
}

/// Prototype, default off: per-window timings of a ragged linear batch
/// window. With `IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES=1` the stage
/// boundaries synchronize (draft, verify, head, commit attributable);
/// without it only `window_us` is meaningful.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize)]
pub struct DFlash2RaggedWindowTiming {
    pub rows: usize,
    pub positions: Vec<usize>,
    pub accepted: Vec<usize>,
    pub emitted: Vec<usize>,
    pub cache_build_us: u64,
    pub stages: Vec<(&'static str, u64)>,
    pub window_us: u64,
}

impl DFlash2RaggedBatchCache {
    /// Copy the batched target cache back into the rows (in row order).
    pub fn scatter_to_rows<M: DFlash2Target>(
        &self,
        rows: &mut [&mut DFlash2TextGenerationStream<'_, M>],
    ) -> Result<()> {
        anyhow::ensure!(
            rows.len() == self.batch_size,
            "ragged cache width {} cannot address {} rows",
            self.batch_size,
            rows.len()
        );
        for (batch_row, row) in rows.iter_mut().enumerate() {
            ironmlx_lm::core::cache::layer::adopt_layer_cache_rows(
                &mut row.target_cache,
                &self.target,
                0,
                batch_row,
            )?;
        }
        Ok(())
    }
}

pub(crate) struct DFlash2TensorBatchCache {
    target: Vec<LayerCache>,
    draft: DFlash2DraftCache,
    batch_size: usize,
}

impl DFlash2TensorBatchCache {
    fn ensure_row_width<M: DFlash2Target>(
        &self,
        rows: &[&mut DFlash2TextGenerationStream<'_, M>],
    ) -> Result<()> {
        anyhow::ensure!(
            rows.len() == self.batch_size,
            "DFlash2 tensor cache width {} cannot address {} rows",
            self.batch_size,
            rows.len()
        );
        Ok(())
    }

    /// Keep each stream's lightweight draft-cache position view aligned with
    /// the authoritative persistent tensor cache. Target KV remains owned by
    /// the tensor group until it is scattered, so this does not dismantle the
    /// persistent B=N execution state between windows.
    fn sync_draft_rows<M: DFlash2Target>(
        &self,
        rows: &mut [&mut DFlash2TextGenerationStream<'_, M>],
    ) -> Result<()> {
        self.ensure_row_width(rows)?;
        let target = StreamOrDevice::default();
        for (batch_row, row) in rows.iter_mut().enumerate() {
            row.draft_cache = self.draft.row_on(batch_row, target)?;
        }
        Ok(())
    }

    pub(crate) fn scatter_to_rows<M: DFlash2Target>(
        &self,
        rows: &mut [&mut DFlash2TextGenerationStream<'_, M>],
    ) -> Result<()> {
        self.ensure_row_width(rows)?;
        let target = StreamOrDevice::default();
        for (batch_row, row) in rows.iter_mut().enumerate() {
            ironmlx_lm::core::cache::layer::adopt_layer_cache_rows(
                &mut row.target_cache,
                &self.target,
                0,
                batch_row,
            )?;
            row.draft_cache = self.draft.row_on(batch_row, target)?;
        }
        Ok(())
    }
}

impl<'m, M> DFlash2TextGenerationStream<'m, M>
where
    M: DFlash2Target,
{
    pub(crate) fn validate_text_request(
        draft: &DFlash2DraftModel,
        request: &GenerateRequest,
        block_size: usize,
    ) -> Result<()> {
        if request.pixel_values.is_some() || request.image_grid_thw.is_some() {
            return Err(anyhow::Error::new(DFlash2RequestError::VisionUnsupported));
        }
        if request.prompt_ids.is_empty() {
            return Err(anyhow::Error::new(DFlash2RequestError::EmptyPrompt));
        }
        if request.max_new_tokens == 0 {
            return Err(anyhow::Error::new(DFlash2RequestError::ZeroMaxNewTokens));
        }
        if request.kv_cache_turboquant_bits.is_some() {
            return Err(anyhow::Error::new(
                DFlash2RequestError::TurboQuantUnsupported,
            ));
        }
        let checkpoint_block_size = usize::try_from(draft.config().dflash_config.block_size)?;
        if !(2..=checkpoint_block_size).contains(&block_size) {
            return Err(anyhow!(
                "DFlash2 runtime block_size {block_size} must be in [2, checkpoint block_size={checkpoint_block_size}]"
            ));
        }
        Ok(())
    }

    pub fn new_text_only(
        model: &'m M,
        draft: &'m DFlash2DraftModel,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
        block_size: usize,
    ) -> Result<Self> {
        Self::new_text_only_with_options(
            model,
            draft,
            tokenizer,
            request,
            block_size,
            DFlash2P2Options::default(),
        )
    }

    pub fn new_text_only_with_options(
        model: &'m M,
        draft: &'m DFlash2DraftModel,
        tokenizer: &'m Tokenizer,
        mut request: GenerateRequest,
        block_size: usize,
        options: DFlash2P2Options,
    ) -> Result<Self> {
        let options = options.validate()?;
        if options.position_keyed_sampling {
            request.sampler = request.sampler.with_position_keyed_v1();
        }
        Self::new_text_only_with_prefill_execution(
            model,
            draft,
            tokenizer,
            request,
            block_size,
            options,
            DFlash2PrefillContext {
                execution: DFlash2PrefillExecution::GenerationStream,
                cold_single_prefill: false,
                prefix_cache: None,
                is_cancelled: None,
            },
        )
    }

    pub(crate) fn new_scheduler_b1_text_only_with_cancellation(
        model: &'m M,
        draft: &'m DFlash2DraftModel,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
        execution: DFlash2ExecutionOptions,
        prefix_cache: Option<(&mut DFlash2PrefixCache, &str)>,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Self> {
        let options = execution.p2.validate()?;
        // Single-graph prefill is qualified only with the M5 affine4 target
        // route, and it needs the whole prompt in one pass: without a prefix
        // cache it always applies; with one it applies only to a cold miss
        // (nothing restored). A restored prefix keeps the chunked scheduler
        // prefill for the remainder.
        let single_prefill = ironmlx_core::m5_profile::flag(
            ironmlx_core::m5_profile::settings::DFLASH2_SINGLE_PREFILL,
        ) && model
            .dflash2_execution_fingerprint()
            .contains("experimental-m5-affine4");
        let cold_single_prefill = single_prefill && prefix_cache.is_some();
        let single_prefill = single_prefill && prefix_cache.is_none();
        let mut request = request;
        if options.position_keyed_sampling {
            request.sampler = request.sampler.with_position_keyed_v1();
        }
        Self::new_text_only_with_prefill_execution(
            model,
            draft,
            tokenizer,
            request,
            execution.block_size,
            options,
            DFlash2PrefillContext {
                execution: if single_prefill {
                    DFlash2PrefillExecution::GenerationStream
                } else {
                    DFlash2PrefillExecution::SchedulerB1
                },
                cold_single_prefill,
                prefix_cache,
                is_cancelled: Some(is_cancelled),
            },
        )
    }

    /// Construct an equal-length scheduler batch with one target prefill graph.
    ///
    /// The batched path deliberately preserves the scheduler B1 `[N - 1] + [1]`
    /// prefill morphology. Quantized projections are isolated per batch row by
    /// the target implementation, so each resulting stream starts from the same
    /// target hidden state, logits, and cache state as its B1 counterpart.
    pub(crate) fn new_scheduler_bn_text_only_with_cancellation(
        model: &'m M,
        draft: &'m DFlash2DraftModel,
        tokenizer: &'m Tokenizer,
        requests: Vec<GenerateRequest>,
        block_size: usize,
        options: DFlash2P2Options,
        is_cancelled: &dyn Fn(usize) -> bool,
    ) -> Result<Vec<Self>> {
        let options = options.validate()?;
        anyhow::ensure!(
            requests.len() > 1,
            "DFlash2 batched prefill requires at least two requests"
        );
        let mut requests = requests;
        if options.position_keyed_sampling {
            for request in &mut requests {
                request.sampler = request.sampler.with_position_keyed_v1();
            }
        }
        for (index, request) in requests.iter().enumerate() {
            Self::validate_text_request(draft, request, block_size)?;
            ensure_dflash2_request_not_cancelled(Some(&|| is_cancelled(index)))?;
        }

        let batch_size = requests.len();
        let batch_size_i32 = i32::try_from(batch_size)?;
        let prompt_len = requests[0].prompt_ids.len();
        anyhow::ensure!(
            requests
                .iter()
                .all(|request| request.prompt_ids.len() == prompt_len),
            "DFlash2 batched prefill requires equal prompt lengths"
        );
        let requested_chunk_size = requests[0].prefill_chunk_size;
        anyhow::ensure!(
            requests
                .iter()
                .all(|request| request.prefill_chunk_size == requested_chunk_size),
            "DFlash2 batched prefill requires equal prefill chunk sizes"
        );

        let prompt_len_i32 = i32::try_from(prompt_len).context("DFlash2 prompt is too long")?;
        let cap = requests
            .iter()
            .map(|request| {
                request
                    .prompt_ids
                    .len()
                    .saturating_add(request.max_new_tokens)
            })
            .max()
            .unwrap_or(1);
        let cap =
            i32::try_from(cap)?.max(ironmlx_lm::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let mut batched_target_cache =
            model.make_cache(batch_size_i32, cap, model.cache_dtype())?;
        let target_layer_ids = &draft.config().dflash_config.target_layer_ids;
        let context_limit = draft.config().sliding_window - 1;
        let prefill_started = Instant::now();
        let mut context_hidden: Option<Array> = None;
        let mut last_hidden: Option<Array> = None;
        let mut position = 0_i32;

        while position < prompt_len_i32 {
            for index in 0..batch_size {
                ensure_dflash2_request_not_cancelled(Some(&|| is_cancelled(index)))?;
            }
            let remaining = prompt_len_i32 - position;
            let chunk_len = dflash2_prefill_chunk_len(
                remaining,
                requested_chunk_size,
                position,
                DFlash2PrefillExecution::SchedulerB1,
            );
            let start = position as usize;
            let stop = start + chunk_len as usize;
            let mut flat = Vec::with_capacity(batch_size * chunk_len as usize);
            for request in &requests {
                flat.extend_from_slice(&request.prompt_ids[start..stop]);
            }
            let input: Array = (&flat[..], &[batch_size_i32, chunk_len][..]).try_into()?;
            let position_ids = build_position_ids(position, chunk_len)?;
            let position_ids = mlx::ops::shape::broadcast_to(
                &position_ids,
                &[3_i32, batch_size_i32, chunk_len][..],
            )?;
            let output = model.dflash2_forward_target_on(
                &input,
                &position_ids,
                Some(&mut batched_target_cache),
                target_layer_ids,
                DFlash2TargetForwardMode::Prefill,
                StreamOrDevice::default(),
            )?;
            context_hidden = Some(retain_context_tail_batched(
                context_hidden.as_ref(),
                &output.context_hidden,
                context_limit,
                StreamOrDevice::default(),
            )?);
            last_hidden = Some(slice_sequence_position_batched(
                &output.hidden,
                chunk_len - 1,
                StreamOrDevice::default(),
            )?);
            mlx::transforms::eval(&[
                context_hidden
                    .as_ref()
                    .expect("batched DFlash2 context is present"),
                last_hidden
                    .as_ref()
                    .expect("batched DFlash2 final hidden is present"),
            ])?;
            position += chunk_len;
        }

        let context_hidden =
            context_hidden.ok_or_else(|| anyhow!("DFlash2 batched prefill produced no context"))?;
        let last_hidden =
            last_hidden.ok_or_else(|| anyhow!("DFlash2 batched prefill produced no hidden"))?;
        let retained_len = sequence_len_batched(&context_hidden)?;
        let initial_offset = prompt_len_i32
            .checked_sub(retained_len)
            .ok_or_else(|| anyhow!("DFlash2 retained context exceeds prompt length"))?;
        let prefill_us = elapsed_us(prefill_started);
        let mut streams = Vec::with_capacity(batch_size);

        for (batch_row, request) in requests.into_iter().enumerate() {
            ensure_dflash2_request_not_cancelled(Some(&|| is_cancelled(batch_row)))?;
            let row_context =
                slice_batch_row(&context_hidden, batch_row, StreamOrDevice::default())?;
            let row_last_hidden =
                slice_batch_row(&last_hidden, batch_row, StreamOrDevice::default())?;
            let first_logits =
                model.dflash2_project_hidden_on(&row_last_hidden, StreamOrDevice::default())?;
            mlx::transforms::eval(&[&first_logits, &row_context])?;

            let mut target_cache = model.make_cache(1, cap, model.cache_dtype())?;
            ironmlx_lm::core::cache::layer::adopt_layer_cache_rows(
                &mut target_cache,
                &batched_target_cache,
                0,
                batch_row,
            )?;
            let mut prng_state = mlx::random::key(request.sampler.seed)?;
            let mut constraint = request
                .constraint
                .as_ref()
                .map(|plan| plan.start_session())
                .transpose()?;
            let first_token = sample_initial_token(
                &first_logits,
                request.sampler,
                &request.prompt_ids,
                &mut prng_state,
                &mut constraint,
            )?;
            commit_constraint_token(&mut constraint, first_token)?;
            let draft_cache = draft.make_cache(initial_offset)?;
            let mut history = request.prompt_ids.clone();
            history.push(first_token);
            let mut pending_tokens = VecDeque::new();
            pending_tokens.push_back(first_token);
            let verify_capabilities =
                qualification_verify_capabilities(model.dflash2_verify_capabilities())?;
            let max_draft_tokens = verify_capabilities
                .max_draft_tokens(1)
                .unwrap_or(0)
                .min(block_size - 1);
            anyhow::ensure!(
                max_draft_tokens > 0,
                "DFlash2 verify profile {} has no certified B1 speculative width",
                verify_capabilities.profile
            );
            streams.push(Self {
                model,
                draft,
                target_cache,
                draft_cache,
                history,
                request,
                pending_tokens,
                detok: tokenizer.decode_stream(true),
                pending_context_hidden: row_context,
                verify_capabilities,
                draft_policy: QwenMtpDraftPolicyState::new(max_draft_tokens),
                experimental_fixed_budget: experimental_fixed_budget(max_draft_tokens)?,
                prng_state,
                block_size,
                emitted_new_tokens: 0,
                finished: false,
                prefill_us,
                generation_started: Instant::now(),
                counters: DFlash2Counters::default(),
                constraint,
                prefix_cache_hit_tokens: 0,
                target_cache_cap: cap,
                p2_options: options,
            });
        }
        Ok(streams)
    }

    fn new_text_only_with_prefill_execution(
        model: &'m M,
        draft: &'m DFlash2DraftModel,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
        block_size: usize,
        p2_options: DFlash2P2Options,
        prefill: DFlash2PrefillContext<'_>,
    ) -> Result<Self> {
        let DFlash2PrefillContext {
            execution: prefill_execution,
            cold_single_prefill,
            prefix_cache,
            is_cancelled,
        } = prefill;
        let (mut prefix_cache, prefix_fingerprint) = match prefix_cache {
            Some((cache, fingerprint)) => (Some(cache), fingerprint),
            None => (None, "generation-stream-no-prefix-cache"),
        };
        Self::validate_text_request(draft, &request, block_size)?;
        ensure_dflash2_request_not_cancelled(is_cancelled)?;

        let prompt_len = request.prompt_ids.len();
        let cap = ((prompt_len + request.max_new_tokens) as i32)
            .max(ironmlx_lm::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let mut target_cache = model.make_cache(1, cap, model.cache_dtype())?;
        let target_layer_ids = &draft.config().dflash_config.target_layer_ids;
        let context_limit = draft.config().sliding_window - 1;
        let prefill_started = Instant::now();
        memory_phase_diagnostic("prefill_start", 0, prompt_len);
        let mut context_hidden: Option<Array> = None;
        let mut last_hidden: Option<Array> = None;
        let mut position = 0_i32;
        let mut prefix_cache_hit_tokens = 0_usize;
        let prompt_len_i32 = i32::try_from(prompt_len).context("DFlash2 prompt is too long")?;
        let mut phases = PrefillPhaseDiagnostic::start();

        if let Some(cache) = prefix_cache.as_deref_mut() {
            let artifact = cache.load_longest(&request.prompt_ids, prefix_fingerprint);
            PrefillPhaseDiagnostic::mark(&mut phases, "load", 0);
            if let Some(artifact) = artifact {
                let restore_result = restore_prefix_entry_for_row(
                    &mut target_cache,
                    &artifact.target_cache,
                    0,
                    artifact.cached_len,
                );
                PrefillPhaseDiagnostic::mark(&mut phases, "restore_graph", artifact.cached_len);
                let restore_result = restore_result.and_then(|()| {
                    materialize_dflash2_target_cache_prefix(&target_cache, artifact.cached_len)
                });
                PrefillPhaseDiagnostic::mark(&mut phases, "materialize", artifact.cached_len);
                match restore_result {
                    Ok(()) => {
                        position = artifact.cached_len;
                        prefix_cache_hit_tokens = artifact.token_ids.len();
                        context_hidden = Some(artifact.context_hidden);
                        last_hidden = Some(artifact.last_hidden);
                    }
                    Err(error) => {
                        tracing::warn!(
                            cached_len = artifact.cached_len,
                            error = %error,
                            "invalid DFlash2 prefix artifact evicted; falling back to cold prefill"
                        );
                        cache.invalidate(&artifact.token_ids, prefix_fingerprint);
                        target_cache = model.make_cache(1, cap, model.cache_dtype())?;
                    }
                }
            }
        }
        PrefillPhaseDiagnostic::mark(&mut phases, "lookup", position);
        // A cold miss computes the whole prompt like a request without a
        // prefix cache; its saved artifacts come from that same pass.
        let prefill_execution = if cold_single_prefill && position == 0 {
            DFlash2PrefillExecution::GenerationStream
        } else {
            prefill_execution
        };

        while position < prompt_len_i32 {
            // Experimental, default off: at each bulk chunk boundary release
            // cached buffers sized for the previous chunk (attention scores
            // grow with KV and are never reused), so a raised MLX cache
            // ceiling retains only the current chunk's buffers.
            if position > 0
                && prompt_len_i32 - position > 128
                && prefill_chunk_cache_reset_requested()
            {
                mlx::clear_cache();
            }
            ensure_dflash2_request_not_cancelled(is_cancelled)?;
            let remaining = prompt_len_i32 - position;
            let chunk_len = dflash2_prefill_chunk_len(
                remaining,
                request.prefill_chunk_size,
                position,
                prefill_execution,
            );
            let start = position as usize;
            let stop = start + chunk_len as usize;
            let input: Array =
                (&request.prompt_ids[start..stop], &[1_i32, chunk_len][..]).try_into()?;
            let position_ids = build_position_ids(position, chunk_len)?;
            let output = model.dflash2_forward_target_on(
                &input,
                &position_ids,
                Some(&mut target_cache),
                target_layer_ids,
                DFlash2TargetForwardMode::Prefill,
                StreamOrDevice::default(),
            )?;
            ensure_dflash2_request_not_cancelled(is_cancelled)?;
            context_hidden = Some(retain_context_tail(
                context_hidden.as_ref(),
                &output.context_hidden,
                context_limit,
                StreamOrDevice::default(),
            )?);
            last_hidden = Some(slice_sequence_position(
                &output.hidden,
                chunk_len - 1,
                StreamOrDevice::default(),
            )?);
            let retained = context_hidden
                .as_ref()
                .ok_or_else(|| anyhow!("DFlash2 prefill retained no target context"))?;
            let chunk_last_hidden = last_hidden
                .as_ref()
                .ok_or_else(|| anyhow!("DFlash2 prefill retained no final hidden"))?;
            mlx::transforms::eval(&[retained, chunk_last_hidden])?;
            PrefillPhaseDiagnostic::mark(&mut phases, "chunk", chunk_len);
            memory_phase_diagnostic("prefill_chunk_evaluated", position + chunk_len, prompt_len);
            ensure_dflash2_request_not_cancelled(is_cancelled)?;
            // Dense and linear target state is materially larger than an
            // ordinary paged-KV prefix entry. Retain the last reusable chunk
            // boundary and the exact full prompt instead of every cumulative
            // prefill chunk. This preserves exact-repeat and appended-turn
            // reuse without multiplying resident memory by the chunk count.
            let cache_this_boundary = should_cache_dflash2_prefill_boundary(
                prompt_len_i32,
                position,
                chunk_len,
                request.prefill_chunk_size,
                prefill_execution,
            );
            if let (true, Some(cache)) = (cache_this_boundary, prefix_cache.as_deref_mut()) {
                match cache.insert(
                    &request.prompt_ids[..stop],
                    prefix_fingerprint,
                    &target_cache,
                    retained,
                    chunk_last_hidden,
                ) {
                    Ok(true) => tracing::debug!(
                        cached_len = stop,
                        "DFlash2 cross-request prefix artifact saved"
                    ),
                    Ok(false) => {}
                    Err(error) => tracing::warn!(
                        cached_len = stop,
                        error = %error,
                        "DFlash2 cross-request prefix save skipped"
                    ),
                }
                PrefillPhaseDiagnostic::mark(&mut phases, "save", stop as i32);
            }
            position += chunk_len;
        }

        ensure_dflash2_request_not_cancelled(is_cancelled)?;

        let last_hidden =
            last_hidden.ok_or_else(|| anyhow!("DFlash2 prefill produced no hidden"))?;
        let first_logits =
            model.dflash2_project_hidden_on(&last_hidden, StreamOrDevice::default())?;
        let context_hidden =
            context_hidden.ok_or_else(|| anyhow!("DFlash2 prefill produced no target context"))?;
        // Keep the target's canonical logits graph identical to ordinary
        // generation. Evaluating the tap-concatenation as a co-root can alter
        // MLX fusion decisions in shared ancestors even though the values are
        // logically independent.
        mlx::transforms::eval(&[&first_logits])?;
        mlx::transforms::eval(&[&context_hidden])?;
        PrefillPhaseDiagnostic::mark(&mut phases, "first_logits", 1);
        PrefillPhaseDiagnostic::finish(
            phases,
            prefill_execution,
            prompt_len,
            prefix_cache_hit_tokens,
        );
        memory_phase_diagnostic("first_logits_evaluated", prompt_len_i32, prompt_len);
        ensure_dflash2_request_not_cancelled(is_cancelled)?;
        let prefill_us = elapsed_us(prefill_started);
        let generation_started = Instant::now();
        let mut prng_state = mlx::random::key(request.sampler.seed)?;
        let mut constraint = request
            .constraint
            .as_ref()
            .map(|plan| plan.start_session())
            .transpose()?;
        let first_token = sample_initial_token(
            &first_logits,
            request.sampler,
            &request.prompt_ids,
            &mut prng_state,
            &mut constraint,
        )?;
        commit_constraint_token(&mut constraint, first_token)?;
        let retained_len = sequence_len(&context_hidden)?;
        let initial_offset = prompt_len_i32
            .checked_sub(retained_len)
            .ok_or_else(|| anyhow!("DFlash2 retained context exceeds prompt length"))?;
        let draft_cache = draft.make_cache(initial_offset)?;

        let mut history = request.prompt_ids.clone();
        history.push(first_token);
        let mut pending_tokens = VecDeque::new();
        pending_tokens.push_back(first_token);

        let verify_capabilities =
            qualification_verify_capabilities(model.dflash2_verify_capabilities())?;
        let max_draft_tokens = verify_capabilities
            .max_draft_tokens(1)
            .unwrap_or(0)
            .min(block_size - 1);
        anyhow::ensure!(
            max_draft_tokens > 0,
            "DFlash2 verify profile {} has no certified B1 speculative width",
            verify_capabilities.profile
        );
        Ok(Self {
            model,
            draft,
            target_cache,
            draft_cache,
            history,
            request,
            pending_tokens,
            detok: tokenizer.decode_stream(true),
            pending_context_hidden: context_hidden,
            verify_capabilities,
            draft_policy: QwenMtpDraftPolicyState::new(max_draft_tokens),
            experimental_fixed_budget: experimental_fixed_budget(max_draft_tokens)?,
            prng_state,
            block_size,
            emitted_new_tokens: 0,
            finished: false,
            prefill_us,
            generation_started,
            counters: DFlash2Counters::default(),
            constraint,
            prefix_cache_hit_tokens,
            target_cache_cap: cap,
            p2_options,
        })
    }

    pub(crate) fn prefix_cache_hit_tokens(&self) -> usize {
        self.prefix_cache_hit_tokens
    }

    pub fn metrics(&self) -> DFlash2Metrics {
        let generation_us = elapsed_us(self.generation_started);
        let prompt_tokens = self.request.prompt_ids.len();
        let acceptance_rate = if self.counters.drafted_tokens == 0 {
            0.0
        } else {
            self.counters.accepted_draft_tokens as f64 / self.counters.drafted_tokens as f64
        };
        DFlash2Metrics {
            block_size: self.block_size,
            tree_max_nodes: self.p2_options.tree_max_nodes,
            experimental_tree_profile: ironmlx_lm::models::dflash2::experimental_tree_profile(),
            tree_proposal_width: if self.p2_options.tree_max_nodes > 0
                && ironmlx_lm::models::dflash2::experimental_tree_profile()
            {
                self.p2_options.tree_max_nodes.min(15) + 1
            } else {
                self.block_size
            },
            position_keyed_sampling: self.p2_options.position_keyed_sampling,
            sampled: self.request.sampler.temperature > 0.0,
            prompt_tokens,
            generated_tokens: self.emitted_new_tokens,
            windows: self.counters.windows,
            drafted_tokens: self.counters.drafted_tokens,
            accepted_draft_tokens: self.counters.accepted_draft_tokens,
            rollback_count: self.counters.rollback_count,
            ordinary_windows: self.counters.ordinary_windows,
            tree_windows: self.counters.tree_windows,
            tree_drafted_nodes: self.counters.tree_drafted_nodes,
            ragged_linear_windows: self.counters.ragged_linear_windows,
            tree_to_ragged_switches: self.counters.tree_to_ragged_switches,
            ragged_to_tree_switches: self.counters.ragged_to_tree_switches,
            draft_budget_changes: self.counters.draft_budget_changes,
            current_draft_budget: self.current_draft_budget(),
            adaptive_acceptance_ewma: self.draft_policy.acceptance_ewma(),
            exact_sampling_windows: self.counters.exact_sampling.windows,
            exact_acceptance_draws: self.counters.exact_sampling.acceptance_draws,
            exact_residual_corrections: self.counters.exact_sampling.residual_corrections,
            exact_bonus_samples: self.counters.exact_sampling.bonus_samples,
            draft_build_us: self.counters.draft_build_us,
            draft_schedule_us: self.counters.draft_schedule_us,
            verify_build_us: self.counters.verify_build_us,
            projection_build_us: self.counters.projection_build_us,
            sampling_us: self.counters.sampling_us,
            verify_schedule_us: self.counters.verify_schedule_us,
            host_sync_us: self.counters.host_sync_us,
            rollback_us: self.counters.rollback_us,
            window_us: self.counters.window_us,
            prefill_us: self.prefill_us,
            generation_us,
            prompt_tps: rate_per_second(prompt_tokens, self.prefill_us),
            generation_tps: rate_per_second(self.emitted_new_tokens, generation_us),
            acceptance_rate,
            peak_memory_bytes: mlx::memory::snapshot().peak_bytes,
        }
    }

    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        let event = self.next_token_deferred()?;
        if let Some(event) = event.as_ref() {
            if event.finish_reason.is_none() && self.pending_tokens.is_empty() {
                self.fill_next_window(event.token)?;
            }
        }
        Ok(event)
    }

    pub(crate) fn next_token_deferred(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }
        let token = self
            .pending_tokens
            .pop_front()
            .ok_or_else(|| anyhow!("DFlash2 stream invariant: pending token queue is empty"))?;
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
            memory_phase_diagnostic(
                "generation_finished",
                i32::try_from(self.history.len()).unwrap_or(i32::MAX),
                self.request.prompt_ids.len(),
            );
        }
        Ok(Some(GenerateEvent {
            token,
            text,
            finish_reason,
        }))
    }

    pub(crate) fn tensor_batch_key(&self) -> Result<Option<DFlash2TensorBatchKey>> {
        if self.finished || !self.pending_tokens.is_empty() {
            return Ok(None);
        }
        let remaining = self
            .request
            .max_new_tokens
            .saturating_sub(self.emitted_new_tokens);
        if remaining == 0 {
            return Ok(None);
        }
        let draft_len = self.current_draft_budget().min(remaining);
        let context_shape = self.pending_context_hidden.shape();
        let context_dims = context_shape.as_slice();
        anyhow::ensure!(
            context_dims.len() == 3 && context_dims[0] == 1,
            "DFlash2 pending context must be [1,S,H], got {context_dims:?}"
        );
        let (draft_processed, draft_retained) = self.draft_cache.position_signature()?;
        if draft_len > 0 {
            anyhow::ensure!(
                draft_processed.checked_add(context_dims[1])
                    == Some(i32::try_from(self.history.len() - 1)?),
                "DFlash2 draft cache/context position mismatch: processed={draft_processed} retained={draft_retained} pending={} verify_start={} draft_len={draft_len}",
                context_dims[1],
                self.history.len() - 1,
            );
        }
        DFlash2VerifyPlan::build(&self.verify_capabilities, 1, draft_len)?;
        let verify_width = draft_len + 1;
        let supported_batch_widths = if draft_len == 0 {
            1_u64 << 1
        } else {
            self.verify_capabilities
                .supported_shapes
                .iter()
                .filter(|shape| {
                    shape.verify_width == verify_width && shape.batch_width < u64::BITS as usize
                })
                .fold(0_u64, |mask, shape| mask | (1_u64 << shape.batch_width))
        };
        Ok(Some(DFlash2TensorBatchKey {
            draft_len,
            verify_start: self.history.len() - 1,
            context_len: context_dims[1],
            draft_processed,
            draft_retained,
            supported_batch_widths,
            sampled: !self.request.sampler.is_greedy(),
        }))
    }

    pub(crate) fn pending_token_count(&self) -> usize {
        self.pending_tokens.len()
    }

    pub(crate) fn prompt_token_ids(&self) -> &[u32] {
        &self.request.prompt_ids
    }

    /// Token ids already handed to the actor for publication (popped from
    /// the pending queue), in order.
    pub(crate) fn published_token_ids(&self) -> &[u32] {
        let start = self.request.prompt_ids.len().min(self.history.len());
        let end = start
            .saturating_add(self.emitted_new_tokens)
            .min(self.history.len());
        &self.history[start..end]
    }

    /// Token capacity this row needs in a ragged batched target cache.
    pub(crate) fn ragged_cache_tokens(&self) -> usize {
        self.request
            .prompt_ids
            .len()
            .saturating_add(self.request.max_new_tokens)
    }

    /// Whether the next window of this row may run as a ragged linear batch
    /// row: greedy, unconstrained, fixed positive draft budget, and enough
    /// remaining tokens for a full linear block. Other rows keep their own
    /// window path.
    pub(crate) fn ragged_linear_eligible(&self) -> bool {
        let Some(draft_len) = self.experimental_fixed_budget else {
            return false;
        };
        !self.finished
            && self.pending_tokens.is_empty()
            && draft_len > 0
            && self.request.sampler.is_greedy()
            && self.constraint.is_none()
            && self
                .request
                .max_new_tokens
                .saturating_sub(self.emitted_new_tokens)
                > draft_len
    }

    pub(crate) fn fill_deferred_window_b1(&mut self) -> Result<()> {
        let current_token = *self
            .history
            .last()
            .ok_or_else(|| anyhow!("DFlash2 stream history is empty"))?;
        self.fill_next_window(current_token)
    }

    fn fill_next_window(&mut self, current_token: u32) -> Result<()> {
        if self.current_draft_budget() == 0 {
            self.fill_ordinary_window(current_token)
        } else if self.tree_eligible() {
            self.fill_tree_window(current_token)
        } else {
            self.fill_window(current_token)
        }
    }

    fn tree_eligible(&self) -> bool {
        self.p2_options.tree_max_nodes > 0
            && self.constraint.is_none()
            && (self.request.sampler.is_greedy() || self.request.sampler.uses_position_keyed_v1())
            && self
                .verify_capabilities
                .lane_kernel_pack
                .as_ref()
                .is_some_and(|pack| pack.quant_bits == 4)
    }

    /// Prototype diagnostic: evaluate this stream's target cache buffers
    /// (used to time cache scatter, which is otherwise lazy).
    #[doc(hidden)]
    pub fn diagnostic_eval_target_cache(&self) -> Result<()> {
        let arrays: Vec<&Array> = self
            .target_cache
            .iter()
            .flat_map(LayerCache::diagnostic_buffers)
            .collect();
        mlx::transforms::eval(&arrays)?;
        Ok(())
    }

    /// Prototype diagnostic: true when the next token needs a new window.
    #[doc(hidden)]
    pub fn diagnostic_needs_window(&self) -> bool {
        !self.finished && self.pending_tokens.is_empty()
    }

    /// Prototype diagnostic: pop the next committed token without filling a
    /// window (the caller decides between B1 and a batched window).
    #[doc(hidden)]
    pub fn diagnostic_next_token_deferred(&mut self) -> Result<Option<GenerateEvent>> {
        self.next_token_deferred()
    }

    /// Prototype diagnostic: fill one ordinary single-request window.
    #[doc(hidden)]
    pub fn diagnostic_fill_b1(&mut self) -> Result<()> {
        self.fill_deferred_window_b1()
    }

    /// Prototype diagnostic: position of the next verify block.
    #[doc(hidden)]
    pub fn diagnostic_verify_start(&self) -> usize {
        self.history.len() - 1
    }

    /// Prototype diagnostic: time the draft proposal for `widths` identical
    /// copies of this stream's draft state (equal positions, so the existing
    /// batched draft path applies). Each pass stacks fresh copies, so this
    /// stream's state is not changed. Returns (width, median us, samples).
    #[doc(hidden)]
    pub fn diagnostic_draft_batch_bench(
        &self,
        widths: &[usize],
        iters: usize,
    ) -> Result<Vec<(usize, u64, Vec<u64>)>> {
        let target = StreamOrDevice::default();
        let draft_len = self.current_draft_budget();
        anyhow::ensure!(draft_len > 0, "draft bench needs a positive draft budget");
        let current = *self
            .history
            .last()
            .ok_or_else(|| anyhow!("empty history"))?;
        let mask_token = self.draft.config().dflash_config.mask_token_id;
        let mut out = Vec::new();
        for &width in widths {
            let mut samples = Vec::with_capacity(iters + 2);
            for pass in 0..iters + 2 {
                let rows = (0..width).map(|_| &self.draft_cache).collect::<Vec<_>>();
                let mut cache = DFlash2DraftCache::stack_rows_on(&rows, target)?;
                let contexts = (0..width)
                    .map(|_| &self.pending_context_hidden)
                    .collect::<Vec<_>>();
                let context = mlx::ops::shape::concatenate_on(&contexts, 0, target)?;
                let mut block = Vec::with_capacity(width * (draft_len + 1));
                for _ in 0..width {
                    block.push(current);
                    block.resize(block.len() + draft_len, mask_token);
                }
                let block: Array = (
                    &block[..],
                    &[i32::try_from(width)?, i32::try_from(draft_len + 1)?][..],
                )
                    .try_into()?;
                mlx::transforms::eval(&[&context, &block])?;
                mlx::transforms::synchronize()?;
                let started = Instant::now();
                let tokens = self
                    .draft
                    .propose_greedy_on(self.model, &block, &context, &mut cache, target)?;
                mlx::transforms::eval(&[&tokens])?;
                mlx::transforms::synchronize()?;
                if pass >= 2 {
                    samples.push(elapsed_us(started));
                }
            }
            let mut sorted = samples.clone();
            sorted.sort_unstable();
            out.push((width, sorted[sorted.len() / 2], samples));
        }
        Ok(out)
    }

    /// Prototype, default off: one linear draft/verify window for rows at
    /// different context lengths (greedy, equal draft budget). Drafts run per
    /// row (the draft cache has one shared position per batch); the target
    /// verify block runs batched with per-row positions and cache offsets,
    /// attention per row on its exact key range (`ragged_rows_scope`), and
    /// each row commits its own accepted prefix. The batched target cache is
    /// returned for reuse while membership is unchanged.
    #[doc(hidden)]
    pub fn diagnostic_ragged_window_bn(
        rows: &mut [&mut Self],
        cache: Option<DFlash2RaggedBatchCache>,
    ) -> Result<(DFlash2RaggedBatchCache, DFlash2RaggedWindowTiming)> {
        Self::fill_ragged_linear_window_bn(rows, cache)
    }

    /// One ragged linear window (see [`Self::diagnostic_ragged_window_bn`]);
    /// used by the DFlash2 actor when the experimental tree/linear switch is
    /// enabled.
    pub(crate) fn fill_ragged_linear_window_bn(
        rows: &mut [&mut Self],
        cache: Option<DFlash2RaggedBatchCache>,
    ) -> Result<(DFlash2RaggedBatchCache, DFlash2RaggedWindowTiming)> {
        anyhow::ensure!(rows.len() >= 2, "ragged batch needs at least two rows");
        let batch_size = rows.len();
        let batch_size_i32 = i32::try_from(batch_size)?;
        let target = StreamOrDevice::default();
        let window_started = Instant::now();
        let draft_len = rows[0].current_draft_budget();
        for (i, row) in rows.iter().enumerate() {
            anyhow::ensure!(
                row.ragged_linear_eligible() && row.current_draft_budget() == draft_len,
                "ragged row {i} is not an eligible greedy linear row"
            );
        }
        anyhow::ensure!(draft_len > 0, "ragged batch needs a positive draft budget");
        DFlash2VerifyPlan::build(&rows[0].verify_capabilities, batch_size, draft_len)?;
        let verify_len = draft_len + 1;
        let mask_token = rows[0].draft.config().dflash_config.mask_token_id;
        let positions_before = rows.iter().map(|r| r.history.len() - 1).collect::<Vec<_>>();

        // Persistent batched target cache.
        let cache_started = Instant::now();
        let mut batch_cache = match cache {
            Some(cache) => {
                anyhow::ensure!(cache.batch_size == batch_size, "ragged cache width changed");
                cache
            }
            None => {
                let cap = i32::try_from(ragged_batch_cache_cap(
                    rows.iter().map(|row| row.ragged_cache_tokens()),
                ))?;
                let mut target_cache =
                    rows[0]
                        .model
                        .make_cache(batch_size_i32, cap, rows[0].model.cache_dtype())?;
                for (batch_row, row) in rows.iter().enumerate() {
                    ironmlx_lm::core::cache::layer::adopt_layer_cache_rows(
                        &mut target_cache,
                        &row.target_cache,
                        batch_row,
                        0,
                    )?;
                }
                if window_stages_enabled() {
                    let arrays: Vec<&Array> = target_cache
                        .iter()
                        .flat_map(LayerCache::diagnostic_buffers)
                        .collect();
                    mlx::transforms::eval(&arrays)?;
                    mlx::transforms::synchronize()?;
                }
                DFlash2RaggedBatchCache {
                    target: target_cache,
                    batch_size,
                }
            }
        };
        let cache_build_us = elapsed_us(cache_started);
        let mut stage_clock = StageClock::start()?;

        // Drafts per row (single-request draft path, unchanged).
        let current_tokens = rows
            .iter()
            .map(|row| {
                row.history
                    .last()
                    .copied()
                    .ok_or_else(|| anyhow!("empty history"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut draft_rows = Vec::with_capacity(batch_size);
        for (row, &current_token) in rows.iter_mut().zip(&current_tokens) {
            let mut block = vec![current_token];
            block.resize(verify_len, mask_token);
            let block_arr: Array =
                (&block[..], &[1_i32, i32::try_from(verify_len)?][..]).try_into()?;
            let draft = row.draft.propose_greedy_on(
                row.model,
                &block_arr,
                &row.pending_context_hidden,
                &mut row.draft_cache,
                target,
            )?;
            anyhow::ensure!(
                draft.shape().as_slice() == [1, i32::try_from(draft_len)?],
                "ragged draft shape {:?}",
                draft.shape().as_slice()
            );
            draft_rows.push(draft);
        }
        let draft_refs = draft_rows.iter().collect::<Vec<_>>();
        let draft_tokens_arr = mlx::ops::shape::concatenate_on(&draft_refs, 0, target)?;
        StageClock::mark(&mut stage_clock, "draft", &[&draft_tokens_arr])?;

        // Batched target verify with per-row positions.
        let current_arr: Array = (&current_tokens[..], &[batch_size_i32, 1][..]).try_into()?;
        let verify_arr =
            mlx::ops::shape::concatenate_on(&[&current_arr, &draft_tokens_arr], 1, target)?;
        let position_rows = positions_before
            .iter()
            .map(|&start| build_position_ids(i32::try_from(start)?, i32::try_from(verify_len)?))
            .collect::<Result<Vec<_>>>()?;
        let position_refs = position_rows.iter().collect::<Vec<_>>();
        let verify_positions = mlx::ops::shape::concatenate_on(&position_refs, 1, target)?;
        let snapshots = batch_cache
            .target
            .iter()
            .map(LayerCache::dflash2_transaction_snapshot)
            .collect::<Result<Vec<_>>>()?;
        for layer in &mut batch_cache.target {
            layer.begin_speculative_prefix_capture()?;
        }
        let verified = {
            let _ragged = ironmlx_lm::nn::gated_attention::ragged_rows_scope();
            rows[0].model.dflash2_forward_target_on(
                &verify_arr,
                &verify_positions,
                Some(&mut batch_cache.target),
                &rows[0].draft.config().dflash_config.target_layer_ids,
                dflash2_target_forward_mode(rows[0].request.sampler),
                target,
            )?
        };
        StageClock::mark(
            &mut stage_clock,
            "verify",
            &[&verified.hidden, &verified.context_hidden],
        )?;

        let logits = rows[0]
            .model
            .dflash2_project_hidden_on(&verified.hidden, target)?;
        let verified_tokens_arr = mlx::ops::reduction::argmax(&logits, -1, false)?;
        mlx::transforms::async_eval(&[&verified_tokens_arr, &verified.context_hidden])?;
        let draft_tokens =
            materialize_dflash2_draft_tokens(&draft_tokens_arr, batch_size, draft_len, mask_token)?;
        let greedy = verified_tokens_arr.to_vec::<u32>()?;
        StageClock::mark(&mut stage_clock, "head", &[])?;

        let mut resolutions = Vec::with_capacity(batch_size);
        for batch_row in 0..batch_size {
            resolutions.push(resolve_speculative_tokens(
                &draft_tokens[batch_row],
                &greedy[batch_row * verify_len..(batch_row + 1) * verify_len],
            )?);
        }
        let accepted_lens = resolutions
            .iter()
            .map(|r| r.accepted_verify_input_len)
            .collect::<Vec<_>>();
        for (batch_row, &len) in accepted_lens.iter().enumerate() {
            anyhow::ensure!(
                len > 0 && len <= verify_len,
                "ragged row {batch_row} accepted invalid prefix {len}/{verify_len}"
            );
        }
        rows[0].model.dflash2_restore_target_prefix_rows_on(
            &mut batch_cache.target,
            &snapshots,
            &accepted_lens,
            target,
        )?;
        let mut contexts = Vec::with_capacity(batch_size);
        for (batch_row, &len) in accepted_lens.iter().enumerate() {
            let context_row = slice_batch_row(&verified.context_hidden, batch_row, target)?;
            contexts.push(slice_sequence_prefix(
                &context_row,
                i32::try_from(len)?,
                target,
            )?);
        }
        if stage_clock.is_some() {
            let mut arrays: Vec<&Array> = batch_cache
                .target
                .iter()
                .flat_map(LayerCache::diagnostic_buffers)
                .collect();
            arrays.extend(contexts.iter());
            StageClock::mark(&mut stage_clock, "commit", &arrays)?;
        }

        let mut emitted = Vec::with_capacity(batch_size);
        let mut accepted = Vec::with_capacity(batch_size);
        for ((batch_row, row), (resolution, context)) in rows
            .iter_mut()
            .enumerate()
            .zip(resolutions.into_iter().zip(contexts))
        {
            row.pending_context_hidden = context;
            row.counters.windows += 1;
            row.counters.drafted_tokens += draft_tokens[batch_row].len();
            row.counters.accepted_draft_tokens += resolution.accepted_draft_len;
            if resolution.needs_rollback {
                row.counters.rollback_count += 1;
            }
            let remaining = row
                .request
                .max_new_tokens
                .saturating_sub(row.emitted_new_tokens);
            let mut tokens_to_append = resolution.tokens_to_append;
            if let Some(stop_index) = tokens_to_append
                .iter()
                .position(|token| row.request.stop_token_ids.contains(token))
            {
                tokens_to_append.truncate(stop_index + 1);
            }
            tokens_to_append.truncate(remaining);
            anyhow::ensure!(
                !tokens_to_append.contains(&mask_token),
                "ragged verification emitted the mask token"
            );
            accepted.push(resolution.accepted_draft_len);
            emitted.push(tokens_to_append.len());
            for token in tokens_to_append {
                row.history.push(token);
                row.pending_tokens.push_back(token);
            }
        }
        let window_us = elapsed_us(window_started);
        for row in rows.iter_mut() {
            if row.counters.last_window_ragged == Some(false) {
                row.counters.tree_to_ragged_switches += 1;
            }
            row.counters.last_window_ragged = Some(true);
            row.counters.ragged_linear_windows += 1;
            row.counters.window_us = row.counters.window_us.saturating_add(window_us);
        }
        let timing = DFlash2RaggedWindowTiming {
            rows: batch_size,
            positions: positions_before,
            accepted,
            emitted,
            cache_build_us,
            stages: stage_clock.map(|c| c.stages).unwrap_or_default(),
            window_us,
        };
        Ok((batch_cache, timing))
    }

    /// Execute one equal-shape multi-row draft/verify window. Rows with a common
    /// accepted-prefix length keep their tensor cache. If the lengths diverge,
    /// restore the pre-verify target state and replay only each accepted input
    /// prefix through the exact Q=1 target path before returning ownership to
    /// the individual streams.
    pub(crate) fn fill_deferred_window_bn(
        rows: &mut [&mut Self],
        cache: Option<DFlash2TensorBatchCache>,
    ) -> Result<Option<DFlash2TensorBatchCache>> {
        anyhow::ensure!(
            rows.len() >= 2,
            "DFlash2 tensor batch requires at least two rows"
        );
        let batch_size = rows.len();
        let batch_size_i32 = i32::try_from(batch_size)?;
        let first_key = rows[0]
            .tensor_batch_key()?
            .ok_or_else(|| anyhow!("DFlash2 tensor row 0 is not ready"))?;
        for (batch_row, row) in rows.iter().enumerate().skip(1) {
            let key = row
                .tensor_batch_key()?
                .ok_or_else(|| anyhow!("DFlash2 tensor row {batch_row} is not ready"))?;
            anyhow::ensure!(
                first_key == key,
                "DFlash2 tensor rows have incompatible window shapes"
            );
        }

        let target = StreamOrDevice::default();
        let window_started = Instant::now();
        let draft_len = first_key.draft_len;
        anyhow::ensure!(
            draft_len > 0,
            "DFlash2 Q1 control windows must execute as B1 rows"
        );
        DFlash2VerifyPlan::build(&rows[0].verify_capabilities, batch_size, draft_len)?;
        let verify_len = draft_len + 1;
        let mask_token = rows[0].draft.config().dflash_config.mask_token_id;
        let sampling_prepare_started = Instant::now();
        let mut exact_uniforms = Vec::with_capacity(batch_size);
        for row in rows.iter_mut() {
            exact_uniforms.push(
                (row.request.sampler.temperature > 0.0
                    && !row.request.sampler.uses_position_keyed_v1()
                    && !row.request.sampler.requires_sampling_history())
                .then(|| prepare_uniforms(&mut row.prng_state, verify_len))
                .transpose()?,
            );
        }
        let sampling_prepare_us = elapsed_us(sampling_prepare_started);

        let current_tokens = rows
            .iter()
            .map(|row| {
                row.history
                    .last()
                    .copied()
                    .ok_or_else(|| anyhow!("DFlash2 stream history is empty"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut blocks = Vec::with_capacity(batch_size * verify_len);
        for &current_token in &current_tokens {
            blocks.push(current_token);
            blocks.resize(blocks.len() + draft_len, mask_token);
        }
        let block_arr: Array = (
            &blocks[..],
            &[batch_size_i32, i32::try_from(verify_len)?][..],
        )
            .try_into()?;
        let context_rows = rows
            .iter()
            .map(|row| &row.pending_context_hidden)
            .collect::<Vec<_>>();
        let context_hidden = mlx::ops::shape::concatenate_on(&context_rows, 0, target)?;
        let mut batch_cache = match cache {
            Some(cache) => cache,
            None => {
                let draft_cache_rows = rows.iter().map(|row| &row.draft_cache).collect::<Vec<_>>();
                let draft = DFlash2DraftCache::stack_rows_on(&draft_cache_rows, target)?;
                let cap = rows
                    .iter()
                    .map(|row| {
                        row.request
                            .prompt_ids
                            .len()
                            .saturating_add(row.request.max_new_tokens)
                    })
                    .max()
                    .unwrap_or(1);
                let cap = i32::try_from(cap)?
                    .max(ironmlx_lm::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
                let mut target_cache =
                    rows[0]
                        .model
                        .make_cache(batch_size_i32, cap, rows[0].model.cache_dtype())?;
                for (batch_row, row) in rows.iter().enumerate() {
                    ironmlx_lm::core::cache::layer::adopt_layer_cache_rows(
                        &mut target_cache,
                        &row.target_cache,
                        batch_row,
                        0,
                    )?;
                }
                DFlash2TensorBatchCache {
                    target: target_cache,
                    draft,
                    batch_size,
                }
            }
        };
        anyhow::ensure!(
            batch_cache.batch_size == batch_size,
            "DFlash2 tensor cache width {} does not match {} active rows",
            batch_cache.batch_size,
            batch_size
        );
        let draft_started = Instant::now();
        let draft_tokens_arr = rows[0].draft.propose_greedy_on(
            rows[0].model,
            &block_arr,
            &context_hidden,
            &mut batch_cache.draft,
            target,
        )?;
        let draft_build_us = elapsed_us(draft_started);
        anyhow::ensure!(
            draft_tokens_arr.shape().as_slice() == [batch_size_i32, i32::try_from(draft_len)?],
            "DFlash2 B={batch_size} draft returned shape {:?} for budget {draft_len}",
            draft_tokens_arr.shape().as_slice()
        );
        let draft_schedule_started = Instant::now();
        mlx::transforms::async_eval(&[&draft_tokens_arr])?;
        let draft_schedule_us = elapsed_us(draft_schedule_started);

        let verify_started = Instant::now();
        let current_arr: Array = (&current_tokens[..], &[batch_size_i32, 1][..]).try_into()?;
        let verify_arr =
            mlx::ops::shape::concatenate_on(&[&current_arr, &draft_tokens_arr], 1, target)?;
        let positions =
            build_position_ids(i32::try_from(first_key.verify_start)?, verify_len as i32)?;
        let position_rows = (0..batch_size).map(|_| &positions).collect::<Vec<_>>();
        let verify_positions = mlx::ops::shape::concatenate_on(&position_rows, 1, target)?;
        let snapshots = batch_cache
            .target
            .iter()
            .map(LayerCache::dflash2_transaction_snapshot)
            .collect::<Result<Vec<_>>>()?;
        for layer in &mut batch_cache.target {
            layer.begin_speculative_prefix_capture()?;
        }
        let target_mode = dflash2_target_forward_mode(rows[0].request.sampler);
        let verified = rows[0].model.dflash2_forward_target_on(
            &verify_arr,
            &verify_positions,
            Some(&mut batch_cache.target),
            &rows[0].draft.config().dflash_config.target_layer_ids,
            target_mode,
            target,
        )?;
        let verify_build_us = elapsed_us(verify_started);

        let projection_started = Instant::now();
        let projected_logits = rows[0]
            .model
            .dflash2_project_hidden_on(&verified.hidden, target)?;
        let all_unconstrained = rows.iter().all(|row| row.constraint.is_none());
        let mut host_sync_us = 0;
        let mut draft_tokens = if all_unconstrained {
            None
        } else {
            let host_sync_started = Instant::now();
            let tokens = materialize_dflash2_draft_tokens(
                &draft_tokens_arr,
                batch_size,
                draft_len,
                mask_token,
            )?;
            host_sync_us = elapsed_us(host_sync_started);
            Some(tokens)
        };
        let verified_logits = if all_unconstrained {
            projected_logits
        } else {
            let draft_tokens = draft_tokens
                .as_ref()
                .expect("constrained DFlash2 batch materializes draft tokens");
            let mut constrained_rows = Vec::with_capacity(batch_size);
            for (batch_row, row) in rows.iter().enumerate() {
                let logits = slice_batch_row(&projected_logits, batch_row, target)?;
                constrained_rows.push(constrain_dflash2_verified_logits(
                    row.constraint.as_ref(),
                    &logits,
                    &draft_tokens[batch_row],
                )?);
            }
            let verified_logits_refs = constrained_rows.iter().collect::<Vec<_>>();
            mlx::ops::shape::concatenate_on(&verified_logits_refs, 0, target)?
        };
        let verified_tokens_arr = (!first_key.sampled)
            .then(|| mlx::ops::reduction::argmax(&verified_logits, -1, false))
            .transpose()?;
        let mut prepared_sampling = Vec::with_capacity(batch_size);
        for (batch_row, row) in rows.iter().enumerate() {
            prepared_sampling.push(if first_key.sampled {
                let logits = slice_batch_row(&verified_logits, batch_row, target)?;
                prepare_dflash2_exact_sampling(
                    &logits,
                    row.request.sampler,
                    draft_len,
                    exact_uniforms[batch_row].is_some(),
                )?
            } else {
                None
            });
        }
        let projection_build_us = elapsed_us(projection_started);

        let verify_schedule_started = Instant::now();
        let mut eval_roots = vec![&verified.context_hidden];
        if let Some(tokens) = verified_tokens_arr.as_ref() {
            eval_roots.push(tokens);
        } else {
            eval_roots.push(&verified_logits);
        }
        mlx::transforms::async_eval(&eval_roots)?;
        let verify_schedule_us = elapsed_us(verify_schedule_started);
        let host_sync_started = Instant::now();
        let draft_tokens = match draft_tokens.take() {
            Some(tokens) => tokens,
            None => materialize_dflash2_draft_tokens(
                &draft_tokens_arr,
                batch_size,
                draft_len,
                mask_token,
            )?,
        };
        let greedy_tokens_flat = verified_tokens_arr
            .as_ref()
            .map(Array::to_vec::<u32>)
            .transpose()?;
        host_sync_us = host_sync_us.saturating_add(elapsed_us(host_sync_started));

        let sampling_started = Instant::now();
        let mut resolutions = Vec::with_capacity(batch_size);
        for batch_row in 0..batch_size {
            let greedy_tokens = greedy_tokens_flat
                .as_ref()
                .map(|tokens| &tokens[batch_row * verify_len..(batch_row + 1) * verify_len]);
            let resolution = if !first_key.sampled {
                let target_tokens = greedy_tokens
                    .ok_or_else(|| anyhow!("DFlash2 greedy verification tokens are absent"))?;
                resolve_speculative_tokens(&draft_tokens[batch_row], target_tokens)?
            } else if let (Some(prepared), Some(uniforms)) = (
                prepared_sampling[batch_row].take(),
                exact_uniforms[batch_row].as_ref(),
            ) {
                let target_tokens = prepared.sample(&uniforms.to_vec()?)?;
                resolve_exact_deterministic_target_tokens(&draft_tokens[batch_row], &target_tokens)?
            } else {
                let logits = slice_batch_row(&verified_logits, batch_row, target)?;
                resolve_dflash2_window(
                    &draft_tokens[batch_row],
                    greedy_tokens,
                    &logits,
                    rows[batch_row].request.sampler,
                    &rows[batch_row].history,
                    &mut rows[batch_row].prng_state,
                )?
            };
            resolutions.push(resolution);
        }
        let sampling_us = sampling_prepare_us.saturating_add(elapsed_us(sampling_started));

        let keep_batch_cache = resolutions.iter().all(|resolution| {
            resolution.accepted_verify_input_len == resolutions[0].accepted_verify_input_len
        });
        let rollback_started = Instant::now();
        let context_rows = if keep_batch_cache {
            let accepted_len = resolutions[0].accepted_verify_input_len;
            if resolutions[0].needs_rollback {
                rows[0].model.dflash2_restore_target_prefix_on(
                    &mut batch_cache.target,
                    &snapshots,
                    accepted_len,
                    target,
                )?;
            } else {
                for layer in &mut batch_cache.target {
                    layer.discard_speculative_prefix_capture();
                }
            }
            (0..batch_size)
                .map(|batch_row| {
                    let context_row = slice_batch_row(&verified.context_hidden, batch_row, target)?;
                    if resolutions[batch_row].needs_rollback {
                        slice_sequence_prefix(&context_row, i32::try_from(accepted_len)?, target)
                    } else {
                        Ok(context_row)
                    }
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            let accepted_lens = resolutions
                .iter()
                .map(|resolution| resolution.accepted_verify_input_len)
                .collect::<Vec<_>>();
            for (batch_row, &accepted_len) in accepted_lens.iter().enumerate() {
                anyhow::ensure!(
                    accepted_len > 0 && accepted_len <= verify_len,
                    "DFlash2 B={batch_size} row {batch_row} accepted invalid verify prefix {accepted_len}/{verify_len}"
                );
            }
            if first_key.sampled {
                rows[0].model.dflash2_restore_target_prefix_rows_on(
                    &mut batch_cache.target,
                    &snapshots,
                    &accepted_lens,
                    target,
                )?;
                let contexts = accepted_lens
                    .iter()
                    .enumerate()
                    .map(|(batch_row, &accepted_len)| {
                        let context_row =
                            slice_batch_row(&verified.context_hidden, batch_row, target)?;
                        slice_sequence_prefix(&context_row, i32::try_from(accepted_len)?, target)
                    })
                    .collect::<Result<Vec<_>>>()?;
                batch_cache.scatter_to_rows(rows)?;
                contexts
            } else {
                for (layer, snapshot) in batch_cache.target.iter_mut().zip(&snapshots) {
                    layer.restore(snapshot)?;
                }
                batch_cache.scatter_to_rows(rows)?;
                let mut replayed_contexts = Vec::with_capacity(batch_size);
                for (batch_row, row) in rows.iter_mut().enumerate() {
                    let accepted_len = accepted_lens[batch_row];
                    let mut context_steps = Vec::with_capacity(accepted_len);
                    for depth in 0..accepted_len {
                        let token = if depth == 0 {
                            current_tokens[batch_row]
                        } else {
                            draft_tokens[batch_row][depth - 1]
                        };
                        let input: Array = (&[token][..], &[1_i32, 1][..]).try_into()?;
                        let position = build_position_ids(
                            i32::try_from(first_key.verify_start.saturating_add(depth))?,
                            1,
                        )?;
                        let replayed = row.model.dflash2_forward_target_on(
                            &input,
                            &position,
                            Some(&mut row.target_cache),
                            &row.draft.config().dflash_config.target_layer_ids,
                            dflash2_target_forward_mode(row.request.sampler),
                            target,
                        )?;
                        mlx::transforms::eval(&[&replayed.context_hidden])?;
                        context_steps.push(replayed.context_hidden);
                    }
                    let context_refs = context_steps.iter().collect::<Vec<_>>();
                    replayed_contexts.push(mlx::ops::shape::concatenate_on(
                        &context_refs,
                        1,
                        target,
                    )?);
                }
                replayed_contexts
            }
        };
        let rollback_us = elapsed_us(rollback_started);
        let window_us = elapsed_us(window_started);

        for ((batch_row, (row, resolution)), context_row) in rows
            .iter_mut()
            .zip(resolutions)
            .enumerate()
            .zip(context_rows)
        {
            let accepted_draft_len = resolution.accepted_draft_len;
            row.pending_context_hidden = context_row;
            row.counters.windows += 1;
            row.counters.drafted_tokens += draft_tokens[batch_row].len();
            row.counters.accepted_draft_tokens += resolution.accepted_draft_len;
            row.counters.exact_sampling.windows = row
                .counters
                .exact_sampling
                .windows
                .saturating_add(resolution.exact_sampling().windows);
            row.counters.exact_sampling.acceptance_draws = row
                .counters
                .exact_sampling
                .acceptance_draws
                .saturating_add(resolution.exact_sampling().acceptance_draws);
            row.counters.exact_sampling.residual_corrections = row
                .counters
                .exact_sampling
                .residual_corrections
                .saturating_add(resolution.exact_sampling().residual_corrections);
            row.counters.exact_sampling.bonus_samples = row
                .counters
                .exact_sampling
                .bonus_samples
                .saturating_add(resolution.exact_sampling().bonus_samples);
            if resolution.needs_rollback {
                row.counters.rollback_count += 1;
            }
            row.counters.draft_build_us =
                row.counters.draft_build_us.saturating_add(draft_build_us);
            row.counters.draft_schedule_us = row
                .counters
                .draft_schedule_us
                .saturating_add(draft_schedule_us);
            row.counters.verify_build_us =
                row.counters.verify_build_us.saturating_add(verify_build_us);
            row.counters.projection_build_us = row
                .counters
                .projection_build_us
                .saturating_add(projection_build_us);
            row.counters.sampling_us = row.counters.sampling_us.saturating_add(sampling_us);
            row.counters.verify_schedule_us = row
                .counters
                .verify_schedule_us
                .saturating_add(verify_schedule_us);
            row.counters.host_sync_us = row.counters.host_sync_us.saturating_add(host_sync_us);
            row.counters.rollback_us = row.counters.rollback_us.saturating_add(rollback_us);
            row.counters.window_us = row.counters.window_us.saturating_add(window_us);

            let context_tokens = row.history.len();
            let remaining = row
                .request
                .max_new_tokens
                .saturating_sub(row.emitted_new_tokens);
            let mut tokens_to_append = resolution.tokens_to_append;
            if let Some(stop_index) = tokens_to_append
                .iter()
                .position(|token| row.request.stop_token_ids.contains(token))
            {
                tokens_to_append.truncate(stop_index + 1);
            }
            tokens_to_append.truncate(remaining);
            if let Some(constraint) = row.constraint.as_ref() {
                constraint.truncate_invalid_speculative_bonus(&mut tokens_to_append)?;
            }
            anyhow::ensure!(
                !tokens_to_append.contains(&mask_token),
                "DFlash2 verification emitted reserved mask token {mask_token}"
            );
            let committed_tokens = tokens_to_append.len();
            for token in tokens_to_append {
                commit_constraint_token(&mut row.constraint, token)?;
                row.history.push(token);
                row.pending_tokens.push_back(token);
            }
            row.observe_adaptive_window(MtpDraftPolicyWindow::from_measured_components(
                draft_tokens[batch_row].len(),
                accepted_draft_len,
                committed_tokens,
                window_us,
                context_tokens,
                batch_size,
                draft_build_us.saturating_add(draft_schedule_us),
                verify_build_us.saturating_add(verify_schedule_us),
                projection_build_us,
                sampling_us,
                host_sync_us,
                rollback_us,
            ));
        }
        if keep_batch_cache {
            batch_cache.sync_draft_rows(rows)?;
            Ok(Some(batch_cache))
        } else {
            Ok(None)
        }
    }

    fn fill_tree_window(&mut self, current_token: u32) -> Result<()> {
        let window_started = Instant::now();
        let counters_before = self.counters.clone();
        let remaining = self
            .request
            .max_new_tokens
            .saturating_sub(self.emitted_new_tokens);
        if remaining == 0 {
            return Ok(());
        }
        let mut stage_clock = StageClock::start()?;
        let tf_tree_profile = ironmlx_lm::models::dflash2::experimental_tree_profile();
        if tf_tree_profile {
            anyhow::ensure!(
                ironmlx_core::m5_profile::flag(
                    ironmlx_core::m5_profile::settings::DFLASH2_FLAT_TREE
                ),
                "experimental tf-v1 proposal requires flat-tree verification"
            );
        }
        let draft_len = if tf_tree_profile {
            self.p2_options.tree_max_nodes.min(15)
        } else {
            self.current_draft_budget()
        }
        .min(remaining);
        let mask_token = self.draft.config().dflash_config.mask_token_id;
        let mut block = Vec::with_capacity(draft_len + 1);
        block.push(current_token);
        block.resize(draft_len + 1, mask_token);
        let block_arr: Array =
            (&block[..], &[1_i32, i32::try_from(block.len())?][..]).try_into()?;

        let draft_started = Instant::now();
        let tree = self.draft.propose_tree_on(
            self.model,
            &block_arr,
            &self.pending_context_hidden,
            &mut self.draft_cache,
            ironmlx_lm::models::dflash2::DFlash2TreeSpec {
                max_nodes: self.p2_options.tree_max_nodes,
                children_per_node: if tf_tree_profile { 4 } else { 2 },
            },
            StreamOrDevice::default(),
        )?;
        self.counters.draft_build_us = self
            .counters
            .draft_build_us
            .saturating_add(elapsed_us(draft_started));
        StageClock::mark(&mut stage_clock, "draft", &[])?;
        if ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::DFLASH2_FLAT_TREE) {
            return self.fill_flat_tree(
                current_token,
                &tree,
                remaining,
                window_started,
                stage_clock,
            );
        }
        let paths = tree.leaf_paths();
        anyhow::ensure!(
            !paths.is_empty() && paths.len() <= 8,
            "DFlash2 tree produced unsupported leaf count {}",
            paths.len()
        );
        let max_depth = paths.iter().map(Vec::len).max().unwrap_or(0);
        anyhow::ensure!(
            max_depth > 0 && max_depth <= draft_len,
            "DFlash2 tree depth {max_depth} exceeds proposal depth {draft_len}"
        );
        let verify_len = max_depth + 1;
        DFlash2VerifyPlan::build(&self.verify_capabilities, paths.len(), max_depth)?;

        let mut verify_tokens = Vec::with_capacity(paths.len() * verify_len);
        for path in &paths {
            verify_tokens.push(current_token);
            verify_tokens.extend(path.iter().map(|&node| tree.tokens[node]));
            verify_tokens.resize(verify_tokens.len() + (max_depth - path.len()), mask_token);
        }
        let verify_arr: Array = (
            verify_tokens.as_slice(),
            &[i32::try_from(paths.len())?, i32::try_from(verify_len)?][..],
        )
            .try_into()?;
        let verify_start = i32::try_from(self.history.len() - 1)?;
        let verify_positions = build_position_ids(verify_start, i32::try_from(verify_len)?)?;
        let verify_positions = mlx::ops::shape::broadcast_to(
            &verify_positions,
            &[
                3_i32,
                i32::try_from(paths.len())?,
                i32::try_from(verify_len)?,
            ][..],
        )?;

        let mut tree_cache = self.model.make_cache(
            i32::try_from(paths.len())?,
            self.target_cache_cap,
            self.model.cache_dtype(),
        )?;
        for row in 0..paths.len() {
            ironmlx_lm::core::cache::layer::adopt_layer_cache_rows(
                &mut tree_cache,
                &self.target_cache,
                row,
                0,
            )?;
        }
        let snapshots = tree_cache
            .iter()
            .map(LayerCache::dflash2_transaction_snapshot)
            .collect::<Result<Vec<_>>>()?;
        for layer in &mut tree_cache {
            layer.begin_speculative_prefix_capture()?;
        }
        let verify_started = Instant::now();
        let verified = self.model.dflash2_forward_target_on(
            &verify_arr,
            &verify_positions,
            Some(&mut tree_cache),
            &self.draft.config().dflash_config.target_layer_ids,
            dflash2_target_forward_mode(self.request.sampler),
            StreamOrDevice::default(),
        )?;
        self.counters.verify_build_us = self
            .counters
            .verify_build_us
            .saturating_add(elapsed_us(verify_started));
        let projection_started = Instant::now();
        let logits = self
            .model
            .dflash2_project_hidden_on(&verified.hidden, StreamOrDevice::default())?;
        self.counters.projection_build_us = self
            .counters
            .projection_build_us
            .saturating_add(elapsed_us(projection_started));
        let sampling_started = Instant::now();
        let target_tokens_arr = if self.request.sampler.is_greedy() {
            mlx::ops::reduction::argmax(&logits, -1, false)?
        } else {
            sample_tree_position_keyed_targets(
                &logits,
                self.request.sampler,
                &self.history,
                &tree,
                &paths,
                verify_len,
            )?
        };
        self.counters.sampling_us = self
            .counters
            .sampling_us
            .saturating_add(elapsed_us(sampling_started));
        let schedule_started = Instant::now();
        mlx::transforms::async_eval(&[&target_tokens_arr, &verified.context_hidden])?;
        self.counters.verify_schedule_us = self
            .counters
            .verify_schedule_us
            .saturating_add(elapsed_us(schedule_started));
        let host_sync_started = Instant::now();
        let target_tokens = target_tokens_arr.to_vec::<u32>()?;
        self.counters.host_sync_us = self
            .counters
            .host_sync_us
            .saturating_add(elapsed_us(host_sync_started));
        let accepted = resolve_tree_tokens(&tree, &paths, verify_len, &target_tokens)?;

        let rollback_started = Instant::now();
        let mut accepted_lens = vec![0_usize; paths.len()];
        accepted_lens[accepted.row] = accepted.accepted_nodes.len() + 1;
        self.model.dflash2_restore_target_prefix_rows_on(
            &mut tree_cache,
            &snapshots,
            &accepted_lens,
            StreamOrDevice::default(),
        )?;
        ironmlx_lm::core::cache::layer::adopt_layer_cache_rows(
            &mut self.target_cache,
            &tree_cache,
            0,
            accepted.row,
        )?;
        let context_row = slice_batch_row(
            &verified.context_hidden,
            accepted.row,
            StreamOrDevice::default(),
        )?;
        self.pending_context_hidden = slice_sequence_prefix(
            &context_row,
            i32::try_from(accepted.accepted_nodes.len() + 1)?,
            StreamOrDevice::default(),
        )?;
        self.counters.rollback_us = self
            .counters
            .rollback_us
            .saturating_add(elapsed_us(rollback_started));

        let accepted_draft_len = accepted.accepted_nodes.len();
        let mut tokens_to_append = accepted
            .accepted_nodes
            .iter()
            .map(|&node| tree.tokens[node])
            .collect::<Vec<_>>();
        tokens_to_append.push(accepted.bonus_token);
        if let Some(stop_index) = tokens_to_append
            .iter()
            .position(|token| self.request.stop_token_ids.contains(token))
        {
            tokens_to_append.truncate(stop_index + 1);
        }
        tokens_to_append.truncate(remaining);
        let committed_tokens = tokens_to_append.len();
        for token in tokens_to_append {
            self.history.push(token);
            self.pending_tokens.push_back(token);
        }
        self.counters.windows += 1;
        self.counters.tree_windows = self.counters.tree_windows.saturating_add(1);
        self.counters.tree_drafted_nodes = self
            .counters
            .tree_drafted_nodes
            .saturating_add(tree.tokens.len());
        self.counters.drafted_tokens = self
            .counters
            .drafted_tokens
            .saturating_add(tree.tokens.len());
        self.counters.accepted_draft_tokens = self
            .counters
            .accepted_draft_tokens
            .saturating_add(accepted_draft_len);
        self.counters.rollback_count = self.counters.rollback_count.saturating_add(1);
        let window_us = elapsed_us(window_started);
        self.counters.window_us = self.counters.window_us.saturating_add(window_us);
        self.observe_adaptive_window(MtpDraftPolicyWindow::from_measured_components(
            max_depth,
            accepted_draft_len,
            committed_tokens,
            window_us,
            self.history.len().saturating_sub(committed_tokens),
            // The leaf-path batch is one request's internal verify shape, not
            // independent scheduler rows. Keep it in the B1 cost regime so a
            // d=0 control window remains comparable with this tree window.
            1,
            self.counters
                .draft_build_us
                .saturating_sub(counters_before.draft_build_us),
            self.counters
                .verify_build_us
                .saturating_sub(counters_before.verify_build_us)
                .saturating_add(
                    self.counters
                        .verify_schedule_us
                        .saturating_sub(counters_before.verify_schedule_us),
                ),
            self.counters
                .projection_build_us
                .saturating_sub(counters_before.projection_build_us),
            self.counters
                .sampling_us
                .saturating_sub(counters_before.sampling_us),
            self.counters
                .host_sync_us
                .saturating_sub(counters_before.host_sync_us),
            self.counters
                .rollback_us
                .saturating_sub(counters_before.rollback_us),
        ));
        Ok(())
    }

    fn fill_flat_tree(
        &mut self,
        current_token: u32,
        tree: &DFlash2DraftTree,
        remaining: usize,
        started: Instant,
        mut stage_clock: Option<StageClock>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.request.sampler.is_greedy() && self.experimental_fixed_budget.is_some(),
            "experimental flat tree requires greedy sampling and fixed draft budget"
        );
        let history_before = self.history.len();
        let mut tokens = vec![current_token];
        tokens.extend_from_slice(&tree.tokens);
        let mut parents = vec![-1];
        parents.extend(tree.parents.iter().map(|p| p + 1));
        let input: Array = (tokens.as_slice(), &[1, tokens.len() as i32][..]).try_into()?;
        let snapshots = self
            .target_cache
            .iter()
            .map(LayerCache::dflash2_transaction_snapshot)
            .collect::<Result<Vec<_>>>()?;
        for cache in &mut self.target_cache {
            cache.begin_speculative_prefix_capture()?;
        }
        let begin = Instant::now();
        let verified = self.model.dflash2_forward_tree_on(
            &input,
            &parents,
            (self.history.len() - 1) as i32,
            &mut self.target_cache,
            &self.draft.config().dflash_config.target_layer_ids,
            StreamOrDevice::default(),
        )?;
        self.counters.verify_build_us += elapsed_us(begin);
        StageClock::mark(
            &mut stage_clock,
            "verify",
            &[&verified.hidden, &verified.context_hidden],
        )?;
        let begin = Instant::now();
        let logits = self
            .model
            .dflash2_project_hidden_on(&verified.hidden, StreamOrDevice::default())?;
        let predictions = mlx::ops::reduction::argmax(&logits, -1, false)?;
        self.counters.projection_build_us += elapsed_us(begin);
        let begin = Instant::now();
        mlx::transforms::async_eval(&[&predictions, &verified.context_hidden])?;
        self.counters.verify_schedule_us += elapsed_us(begin);
        let begin = Instant::now();
        let predictions = predictions.to_vec::<u32>()?;
        self.counters.host_sync_us += elapsed_us(begin);
        StageClock::mark(&mut stage_clock, "head", &[])?;
        let mut rows = vec![0_i32];
        loop {
            let last = *rows.last().unwrap() as usize;
            let next = (last + 1..tokens.len())
                .find(|&i| parents[i] == last as i32 && tokens[i] == predictions[last]);
            match next {
                Some(i) => rows.push(i as i32),
                None => break,
            }
        }
        let accepted = rows.len() - 1;
        let bonus = predictions[*rows.last().unwrap() as usize];
        let begin = Instant::now();
        self.model.dflash2_commit_tree_on(
            &mut self.target_cache,
            &snapshots,
            &rows,
            StreamOrDevice::default(),
        )?;
        let indices: Array = (rows.as_slice(), &[rows.len() as i32][..]).try_into()?;
        self.pending_context_hidden =
            mlx::ops::indexing::take(&verified.context_hidden, &indices, 1)?;
        self.counters.rollback_us += elapsed_us(begin);
        if stage_clock.is_some() {
            let arrays = cache_barrier_arrays(&self.target_cache, &self.pending_context_hidden);
            StageClock::mark(&mut stage_clock, "commit", &arrays)?;
        }
        let mut append = rows[1..]
            .iter()
            .map(|&r| tokens[r as usize])
            .collect::<Vec<_>>();
        append.push(bonus);
        if let Some(i) = append
            .iter()
            .position(|t| self.request.stop_token_ids.contains(t))
        {
            append.truncate(i + 1);
        }
        append.truncate(remaining);
        for token in append {
            self.history.push(token);
            self.pending_tokens.push_back(token);
        }
        StageClock::finish(
            stage_clock,
            "tree",
            tree.tokens.len(),
            accepted,
            self.history.len() - history_before,
            history_before,
        );
        self.counters.windows += 1;
        self.counters.tree_windows += 1;
        if self.counters.last_window_ragged == Some(true) {
            self.counters.ragged_to_tree_switches += 1;
        }
        self.counters.last_window_ragged = Some(false);
        self.counters.tree_drafted_nodes += tree.tokens.len();
        self.counters.drafted_tokens += tree.tokens.len();
        self.counters.accepted_draft_tokens += accepted;
        self.counters.rollback_count += 1;
        self.counters.window_us += elapsed_us(started);
        Ok(())
    }

    fn fill_window(&mut self, current_token: u32) -> Result<()> {
        let window_started = Instant::now();
        let counters_before = self.counters.clone();
        let remaining = self
            .request
            .max_new_tokens
            .saturating_sub(self.emitted_new_tokens);
        if remaining == 0 {
            return Ok(());
        }
        let mut stage_clock = StageClock::start()?;
        let history_before = self.history.len();
        let draft_len = self.current_draft_budget().min(remaining);
        DFlash2VerifyPlan::build(&self.verify_capabilities, 1, draft_len)?;
        let sampling_prepare_started = Instant::now();
        let exact_sampling_uniforms = if self.request.sampler.temperature > 0.0
            && !self.request.sampler.uses_position_keyed_v1()
            && !self.request.sampler.requires_sampling_history()
        {
            Some(prepare_uniforms(&mut self.prng_state, draft_len + 1)?)
        } else {
            None
        };
        self.counters.sampling_us = self
            .counters
            .sampling_us
            .saturating_add(elapsed_us(sampling_prepare_started));
        let mask_token = self.draft.config().dflash_config.mask_token_id;
        let mut block = Vec::with_capacity(draft_len + 1);
        block.push(current_token);
        block.resize(draft_len + 1, mask_token);
        let block_arr: Array = (&block[..], &[1_i32, block.len() as i32][..]).try_into()?;
        let draft_started = Instant::now();
        let draft_tokens_arr = self.draft.propose_greedy_on(
            self.model,
            &block_arr,
            &self.pending_context_hidden,
            &mut self.draft_cache,
            StreamOrDevice::default(),
        )?;
        self.counters.draft_build_us = self
            .counters
            .draft_build_us
            .saturating_add(elapsed_us(draft_started));
        if usize::try_from(draft_tokens_arr.shape().as_slice()[1])? != draft_len {
            return Err(anyhow!(
                "DFlash2 draft returned shape {:?} for budget {draft_len}",
                draft_tokens_arr.shape().as_slice()
            ));
        }
        StageClock::mark(&mut stage_clock, "draft", &[&draft_tokens_arr])?;
        let draft_schedule_started = Instant::now();
        mlx::transforms::async_eval(&[&draft_tokens_arr])?;
        self.counters.draft_schedule_us = self
            .counters
            .draft_schedule_us
            .saturating_add(elapsed_us(draft_schedule_started));
        let verify_started = Instant::now();
        let current_arr: Array = (&[current_token][..], &[1_i32, 1][..]).try_into()?;
        let verify_arr = mlx::ops::shape::concatenate_on(
            &[&current_arr, &draft_tokens_arr],
            1,
            StreamOrDevice::default(),
        )?;
        let verify_len = draft_len + 1;
        let verify_start = i32::try_from(self.history.len() - 1)?;
        let verify_positions = build_position_ids(verify_start, verify_len as i32)?;
        let snapshots = self
            .target_cache
            .iter()
            .map(LayerCache::dflash2_transaction_snapshot)
            .collect::<Result<Vec<_>>>()?;
        for layer in &mut self.target_cache {
            layer.begin_speculative_prefix_capture()?;
        }
        let target_mode = dflash2_target_forward_mode(self.request.sampler);
        let verified = self.model.dflash2_forward_target_on(
            &verify_arr,
            &verify_positions,
            Some(&mut self.target_cache),
            &self.draft.config().dflash_config.target_layer_ids,
            target_mode,
            StreamOrDevice::default(),
        )?;
        self.counters.verify_build_us = self
            .counters
            .verify_build_us
            .saturating_add(elapsed_us(verify_started));
        StageClock::mark(
            &mut stage_clock,
            "verify",
            &[&verified.hidden, &verified.context_hidden],
        )?;
        let projection_started = Instant::now();
        // DFlash2 has its own product-stable target projection. Do not arm the
        // MTP verify-QMM candidate here: its MSG route is throughput-oriented
        // and does not preserve the ordinary Q=1 accumulation tree.
        let verified_logits = self
            .model
            .dflash2_project_hidden_on(&verified.hidden, StreamOrDevice::default())?;
        let constrained_draft_tokens = if self.constraint.is_some() {
            let host_sync_started = Instant::now();
            let tokens = draft_tokens_arr.to_vec::<u32>()?;
            self.counters.host_sync_us = self
                .counters
                .host_sync_us
                .saturating_add(elapsed_us(host_sync_started));
            Some(tokens)
        } else {
            None
        };
        let verified_logits = constrain_dflash2_verified_logits(
            self.constraint.as_ref(),
            &verified_logits,
            constrained_draft_tokens.as_deref().unwrap_or_default(),
        )?;
        let verified_tokens_arr = self
            .request
            .sampler
            .is_greedy()
            .then(|| mlx::ops::reduction::argmax(&verified_logits, -1, false))
            .transpose()?;
        self.counters.projection_build_us = self
            .counters
            .projection_build_us
            .saturating_add(elapsed_us(projection_started));
        let sampling_prepare_started = Instant::now();
        let prepared_sampling = prepare_dflash2_exact_sampling(
            &verified_logits,
            self.request.sampler,
            draft_len,
            exact_sampling_uniforms.is_some(),
        )?;
        self.counters.sampling_us = self
            .counters
            .sampling_us
            .saturating_add(elapsed_us(sampling_prepare_started));
        let verify_schedule_started = Instant::now();
        if let Some(verified_tokens_arr) = verified_tokens_arr.as_ref() {
            mlx::transforms::async_eval(&[verified_tokens_arr, &verified.context_hidden])?;
        } else if let (Some(prepared_sampling), Some(exact_sampling_uniforms)) =
            (prepared_sampling.as_ref(), exact_sampling_uniforms.as_ref())
        {
            if let (Some(compact_probabilities), Some(compact_indices)) = (
                prepared_sampling.compact_probabilities(),
                prepared_sampling.compact_indices(),
            ) {
                mlx::transforms::async_eval(&[
                    compact_probabilities,
                    compact_indices,
                    exact_sampling_uniforms,
                    &verified.context_hidden,
                ])?;
            } else {
                mlx::transforms::async_eval(&[
                    prepared_sampling.probabilities(),
                    exact_sampling_uniforms,
                    &verified.context_hidden,
                ])?;
            }
        } else {
            mlx::transforms::async_eval(&[&verified_logits, &verified.context_hidden])?;
        }
        self.counters.verify_schedule_us = self
            .counters
            .verify_schedule_us
            .saturating_add(elapsed_us(verify_schedule_started));
        let host_sync_started = Instant::now();
        let draft_tokens = match constrained_draft_tokens {
            Some(tokens) => tokens,
            None => draft_tokens_arr.to_vec()?,
        };
        let greedy_verified_tokens = verified_tokens_arr
            .as_ref()
            .map(Array::to_vec::<u32>)
            .transpose()?;
        self.counters.host_sync_us = self
            .counters
            .host_sync_us
            .saturating_add(elapsed_us(host_sync_started));
        StageClock::mark(&mut stage_clock, "head", &[])?;
        if draft_tokens.contains(&mask_token) {
            return Err(anyhow!(
                "DFlash2 draft emitted reserved mask token {mask_token}"
            ));
        }
        let sampling_started = Instant::now();
        let resolution = if let (Some(prepared_sampling), Some(exact_sampling_uniforms)) =
            (prepared_sampling, exact_sampling_uniforms)
        {
            let uniforms = exact_sampling_uniforms.to_vec()?;
            let target_tokens = prepared_sampling.sample(&uniforms)?;
            resolve_exact_deterministic_target_tokens(&draft_tokens, &target_tokens)?
        } else {
            resolve_dflash2_window(
                &draft_tokens,
                greedy_verified_tokens.as_deref(),
                &verified_logits,
                self.request.sampler,
                &self.history,
                &mut self.prng_state,
            )?
        };
        self.counters.sampling_us = self
            .counters
            .sampling_us
            .saturating_add(elapsed_us(sampling_started));

        self.counters.windows += 1;
        self.counters.drafted_tokens += draft_tokens.len();
        self.counters.accepted_draft_tokens += resolution.accepted_draft_len;
        self.counters.exact_sampling.windows = self
            .counters
            .exact_sampling
            .windows
            .saturating_add(resolution.exact_sampling().windows);
        self.counters.exact_sampling.acceptance_draws = self
            .counters
            .exact_sampling
            .acceptance_draws
            .saturating_add(resolution.exact_sampling().acceptance_draws);
        self.counters.exact_sampling.residual_corrections = self
            .counters
            .exact_sampling
            .residual_corrections
            .saturating_add(resolution.exact_sampling().residual_corrections);
        self.counters.exact_sampling.bonus_samples = self
            .counters
            .exact_sampling
            .bonus_samples
            .saturating_add(resolution.exact_sampling().bonus_samples);
        if resolution.needs_rollback {
            self.counters.rollback_count += 1;
        }

        let accepted_len = resolution.accepted_verify_input_len;
        let rollback_started = Instant::now();
        self.pending_context_hidden = if resolution.needs_rollback {
            self.model.dflash2_restore_target_prefix_on(
                &mut self.target_cache,
                &snapshots,
                accepted_len,
                StreamOrDevice::default(),
            )?;
            slice_sequence_prefix(
                &verified.context_hidden,
                i32::try_from(accepted_len)?,
                StreamOrDevice::default(),
            )?
        } else {
            for layer in &mut self.target_cache {
                layer.discard_speculative_prefix_capture();
            }
            verified.context_hidden
        };
        self.counters.rollback_us = self
            .counters
            .rollback_us
            .saturating_add(elapsed_us(rollback_started));
        if stage_clock.is_some() {
            let arrays = cache_barrier_arrays(&self.target_cache, &self.pending_context_hidden);
            StageClock::mark(&mut stage_clock, "commit", &arrays)?;
        }

        let accepted_draft_len = resolution.accepted_draft_len;
        let mut tokens_to_append = resolution.tokens_to_append;
        if let Some(stop_index) = tokens_to_append
            .iter()
            .position(|token| self.request.stop_token_ids.contains(token))
        {
            tokens_to_append.truncate(stop_index + 1);
        }
        tokens_to_append.truncate(remaining);
        if let Some(constraint) = self.constraint.as_ref() {
            constraint.truncate_invalid_speculative_bonus(&mut tokens_to_append)?;
        }
        if tokens_to_append.contains(&mask_token) {
            return Err(anyhow!(
                "DFlash2 verification emitted reserved mask token {mask_token}"
            ));
        }
        let committed_tokens = tokens_to_append.len();
        for token in tokens_to_append {
            commit_constraint_token(&mut self.constraint, token)?;
            self.history.push(token);
            self.pending_tokens.push_back(token);
        }
        StageClock::finish(
            stage_clock,
            "linear",
            draft_tokens.len(),
            accepted_draft_len,
            self.history.len() - history_before,
            history_before,
        );
        let window_us = elapsed_us(window_started);
        self.counters.window_us = self.counters.window_us.saturating_add(window_us);
        self.observe_adaptive_window(MtpDraftPolicyWindow::from_measured_components(
            draft_tokens.len(),
            accepted_draft_len,
            committed_tokens,
            window_us,
            self.history.len().saturating_sub(committed_tokens),
            1,
            self.counters
                .draft_build_us
                .saturating_sub(counters_before.draft_build_us)
                .saturating_add(
                    self.counters
                        .draft_schedule_us
                        .saturating_sub(counters_before.draft_schedule_us),
                ),
            self.counters
                .verify_build_us
                .saturating_sub(counters_before.verify_build_us)
                .saturating_add(
                    self.counters
                        .verify_schedule_us
                        .saturating_sub(counters_before.verify_schedule_us),
                ),
            self.counters
                .projection_build_us
                .saturating_sub(counters_before.projection_build_us),
            self.counters
                .sampling_us
                .saturating_sub(counters_before.sampling_us),
            self.counters
                .host_sync_us
                .saturating_sub(counters_before.host_sync_us),
            self.counters
                .rollback_us
                .saturating_sub(counters_before.rollback_us),
        ));
        Ok(())
    }

    fn current_draft_budget(&self) -> usize {
        self.experimental_fixed_budget
            .unwrap_or_else(|| self.draft_policy.current_budget())
    }

    fn observe_adaptive_window(&mut self, window: MtpDraftPolicyWindow) {
        if self.experimental_fixed_budget.is_some() {
            return;
        }
        let change = self.draft_policy.observe_external_window(window);
        if change.reduced || change.increased {
            self.counters.draft_budget_changes =
                self.counters.draft_budget_changes.saturating_add(1);
        }
    }

    fn fill_ordinary_window(&mut self, current_token: u32) -> Result<()> {
        let window_started = Instant::now();
        let remaining = self
            .request
            .max_new_tokens
            .saturating_sub(self.emitted_new_tokens);
        if remaining == 0 {
            return Ok(());
        }
        DFlash2VerifyPlan::build(&self.verify_capabilities, 1, 0)?;
        let context_tokens = self.history.len();
        let input: Array = (&[current_token][..], &[1_i32, 1][..]).try_into()?;
        let positions = build_position_ids(i32::try_from(self.history.len() - 1)?, 1)?;
        let verify_started = Instant::now();
        let output = self.model.dflash2_forward_target_on(
            &input,
            &positions,
            Some(&mut self.target_cache),
            &self.draft.config().dflash_config.target_layer_ids,
            DFlash2TargetForwardMode::OrdinaryDecode,
            StreamOrDevice::default(),
        )?;
        let verify_build_us = elapsed_us(verify_started);
        self.counters.verify_build_us = self
            .counters
            .verify_build_us
            .saturating_add(verify_build_us);

        let projection_started = Instant::now();
        let logits = self
            .model
            .dflash2_project_hidden_on(&output.hidden, StreamOrDevice::default())?;
        let row = logits.reshape((logits.shape().as_slice()[2],))?;
        let row = match self.constraint.as_mut() {
            Some(constraint) => apply_token_mask(&row, &constraint.compute_mask()?)?,
            None => row,
        };
        let projection_build_us = elapsed_us(projection_started);
        self.counters.projection_build_us = self
            .counters
            .projection_build_us
            .saturating_add(projection_build_us);
        let sampling_started = Instant::now();
        let next_token = if self.request.sampler.uses_position_keyed_v1()
            && self.request.sampler.temperature > 0.0
        {
            let rows = row.reshape(&[1_i32, row.shape().as_slice()[0]][..])?;
            sample_position_keyed_v1_batch(
                &[&self.request.sampler],
                &rows,
                &[self.history.as_slice()],
                &[u32::try_from(self.history.len())?],
            )?
            .item::<u32>()?
        } else {
            self.request
                .sampler
                .sample(&row, &self.history, &mut self.prng_state)?
        };
        mlx::transforms::eval(&[&output.context_hidden])?;
        let sampling_us = elapsed_us(sampling_started);
        self.counters.sampling_us = self.counters.sampling_us.saturating_add(sampling_us);

        let ordinary_context_hidden = output.context_hidden;
        self.pending_context_hidden = retain_context_tail(
            Some(&self.pending_context_hidden),
            &ordinary_context_hidden,
            // A d=0 probe is bounded to a handful of windows. Keep every raw
            // target context produced during the probe so its original
            // positions remain contiguous with the paused draft cache; the
            // draft cache performs its own sliding eviction when resumed.
            i32::MAX,
            StreamOrDevice::default(),
        )?;
        commit_constraint_token(&mut self.constraint, next_token)?;
        self.history.push(next_token);
        self.pending_tokens.push_back(next_token);
        self.counters.windows = self.counters.windows.saturating_add(1);
        self.counters.ordinary_windows = self.counters.ordinary_windows.saturating_add(1);
        let window_us = elapsed_us(window_started);
        self.counters.window_us = self.counters.window_us.saturating_add(window_us);
        self.observe_adaptive_window(MtpDraftPolicyWindow::from_measured_components(
            0,
            0,
            1,
            window_us,
            context_tokens,
            1,
            0,
            verify_build_us,
            projection_build_us,
            sampling_us,
            0,
            0,
        ));
        if self.draft_policy.uses_ordinary_decode() {
            // A completed d=0 probe never re-enters the speculative path. Drop
            // the accumulated raw target context; the target cache is already
            // authoritative for continued ordinary decoding.
            self.pending_context_hidden = ordinary_context_hidden;
        }
        Ok(())
    }
}

struct DFlash2TreeResolution {
    accepted_nodes: Vec<usize>,
    bonus_token: u32,
    row: usize,
}

fn resolve_tree_tokens(
    tree: &DFlash2DraftTree,
    paths: &[Vec<usize>],
    verify_len: usize,
    target_tokens: &[u32],
) -> Result<DFlash2TreeResolution> {
    anyhow::ensure!(
        target_tokens.len() == paths.len().saturating_mul(verify_len),
        "DFlash2 tree target token count {} != {} rows * {verify_len}",
        target_tokens.len(),
        paths.len()
    );
    let mut representative = vec![None; tree.tokens.len()];
    for (row, path) in paths.iter().enumerate() {
        for (position, &node) in path.iter().enumerate() {
            representative[node].get_or_insert((row, position + 1));
        }
    }
    anyhow::ensure!(
        representative.iter().all(Option::is_some),
        "DFlash2 tree has a node absent from every leaf path"
    );
    let mut accepted = Vec::new();
    let mut parent = -1_i32;
    loop {
        let (row, prediction_position) = if parent < 0 {
            (0, 0)
        } else {
            representative[parent as usize].expect("validated tree representative")
        };
        let prediction = target_tokens[row * verify_len + prediction_position];
        let child = tree
            .parents
            .iter()
            .enumerate()
            .find(|(node, candidate_parent)| {
                **candidate_parent == parent && tree.tokens[*node] == prediction
            })
            .map(|(node, _)| node);
        let Some(child) = child else {
            return Ok(DFlash2TreeResolution {
                accepted_nodes: accepted,
                bonus_token: prediction,
                row,
            });
        };
        accepted.push(child);
        parent = child as i32;
    }
}

fn sample_tree_position_keyed_targets(
    logits: &Array,
    sampler: ironmlx_core::sampler::Sampler,
    history: &[u32],
    tree: &DFlash2DraftTree,
    paths: &[Vec<usize>],
    verify_len: usize,
) -> Result<Array> {
    let shape = logits.shape();
    let dims = shape.as_slice();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] as usize == paths.len() && dims[1] as usize == verify_len,
        "DFlash2 tree logits must be [{},{verify_len},V], got {dims:?}",
        paths.len()
    );
    let rows = paths.len().saturating_mul(verify_len);
    let flat = logits.reshape(&[i32::try_from(rows)?, dims[2]][..])?;
    let mut histories = Vec::with_capacity(rows);
    let mut positions = Vec::with_capacity(rows);
    let start = u32::try_from(history.len())?;
    for path in paths {
        for position in 0..verify_len {
            let mut prefix = Vec::with_capacity(history.len() + position);
            prefix.extend_from_slice(history);
            prefix.extend(path.iter().take(position).map(|&node| tree.tokens[node]));
            histories.push(prefix);
            positions.push(
                start
                    .checked_add(u32::try_from(position)?)
                    .ok_or_else(|| anyhow!("DFlash2 tree sampling position overflow"))?,
            );
        }
    }
    let sampler_refs = vec![&sampler; rows];
    let history_refs = histories.iter().map(Vec::as_slice).collect::<Vec<_>>();
    sample_position_keyed_v1_batch(&sampler_refs, &flat, &history_refs, &positions)?
        .reshape(&[i32::try_from(paths.len())?, i32::try_from(verify_len)?][..])
        .map_err(Into::into)
}

fn ensure_dflash2_request_not_cancelled(is_cancelled: Option<&dyn Fn() -> bool>) -> Result<()> {
    if is_cancelled.is_some_and(|is_cancelled| is_cancelled()) {
        anyhow::bail!("DFlash2 request cancelled");
    }
    Ok(())
}

fn materialize_dflash2_target_cache_prefix(
    target_cache: &[LayerCache],
    expected_cached_len: i32,
) -> Result<()> {
    let Some((entry, cached_len)) = prefix_entry_for_row(target_cache, 0)? else {
        anyhow::bail!("DFlash2 restored target cache is empty");
    };
    anyhow::ensure!(
        cached_len == expected_cached_len,
        "DFlash2 restored target offset {cached_len} != artifact offset {expected_cached_len}"
    );
    entry.eval()
}

fn sample_initial_token(
    logits: &Array,
    sampler: ironmlx_core::sampler::Sampler,
    history: &[u32],
    prng_state: &mut Array,
    constraint: &mut Option<ConstraintSession>,
) -> Result<u32> {
    let shape = logits.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != 1 {
        return Err(anyhow!(
            "DFlash2 initial logits must be [1,1,V], got {dims:?}"
        ));
    }
    let row = logits.reshape((dims[2],))?;
    let row = match constraint {
        Some(session) => apply_token_mask(&row, &session.compute_mask()?)?,
        None => row,
    };
    if sampler.uses_position_keyed_v1() && sampler.temperature > 0.0 {
        let rows = row.reshape(&[1_i32, dims[2]][..])?;
        let token = sample_position_keyed_v1_batch(
            &[&sampler],
            &rows,
            &[history],
            &[u32::try_from(history.len())?],
        )?;
        return token.item::<u32>().map_err(Into::into);
    }
    sampler.sample(&row, history, prng_state)
}

fn constrain_dflash2_verified_logits(
    constraint: Option<&ConstraintSession>,
    logits: &Array,
    draft_tokens: &[u32],
) -> Result<Array> {
    match constraint {
        Some(constraint) => apply_speculative_token_masks(
            logits,
            &[Some(constraint.speculative_masks(draft_tokens)?)],
        ),
        None => Ok(logits.clone()),
    }
}

fn commit_constraint_token(constraint: &mut Option<ConstraintSession>, token: u32) -> Result<()> {
    if let Some(constraint) = constraint {
        constraint.commit_token(token)?;
    }
    Ok(())
}

fn dflash2_target_forward_mode(
    sampler: ironmlx_core::sampler::Sampler,
) -> DFlash2TargetForwardMode {
    if sampler.is_greedy() {
        DFlash2TargetForwardMode::GreedyVerify
    } else {
        DFlash2TargetForwardMode::SampledVerify
    }
}

fn prepare_dflash2_exact_sampling(
    target_logits: &Array,
    sampler: ironmlx_core::sampler::Sampler,
    draft_len: usize,
    exact_sampling: bool,
) -> Result<Option<PreparedTargetTokenSampling>> {
    if !exact_sampling {
        return Ok(None);
    }
    let shape = target_logits.shape();
    let dims = shape.as_slice();
    let positions = draft_len + 1;
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] as usize == positions,
        "prepared DFlash2 target logits must be [1, {positions}, V], got {dims:?}"
    );
    let rows = target_logits.reshape(&[i32::try_from(positions)?, dims[2]][..])?;
    let sampler_refs = vec![&sampler; positions];
    let empty_histories = vec![&[][..]; positions];
    prepare_target_tokens_with_uniforms_batch(&sampler_refs, &rows, &empty_histories).map(Some)
}

fn resolve_dflash2_window(
    draft_tokens: &[u32],
    greedy_verified_tokens: Option<&[u32]>,
    target_logits: &Array,
    sampler: ironmlx_core::sampler::Sampler,
    history: &[u32],
    prng_state: &mut Array,
) -> Result<SpeculativeResolution> {
    if sampler.is_greedy() {
        let target_tokens = greedy_verified_tokens
            .ok_or_else(|| anyhow!("DFlash2 greedy verification tokens are absent"))?;
        return resolve_speculative_tokens(draft_tokens, target_tokens);
    }
    if sampler.uses_position_keyed_v1() && sampler.temperature > 0.0 {
        let target_tokens =
            sample_dflash2_position_keyed_targets(target_logits, sampler, history, draft_tokens)?;
        return resolve_exact_deterministic_target_tokens(draft_tokens, &target_tokens);
    }
    if sampler.temperature > 0.0 {
        return resolve_exact_deterministic_target_logits(
            draft_tokens,
            target_logits,
            sampler,
            history,
            prng_state,
        );
    }
    let target_tokens = sample_logits_positions(target_logits, sampler, history, prng_state)?;
    resolve_speculative_tokens(draft_tokens, &target_tokens)
}

fn sample_dflash2_position_keyed_targets(
    target_logits: &Array,
    sampler: ironmlx_core::sampler::Sampler,
    history: &[u32],
    draft_tokens: &[u32],
) -> Result<Vec<u32>> {
    let shape = target_logits.shape();
    let dims = shape.as_slice();
    let positions = draft_tokens.len() + 1;
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] as usize == positions,
        "position-keyed DFlash2 target logits must be [1,{positions},V], got {dims:?}"
    );
    let rows = target_logits.reshape(&[i32::try_from(positions)?, dims[2]][..])?;
    let histories = (0..positions)
        .map(|position| {
            let mut prefix = Vec::with_capacity(history.len() + position);
            prefix.extend_from_slice(history);
            prefix.extend_from_slice(&draft_tokens[..position]);
            prefix
        })
        .collect::<Vec<_>>();
    let history_refs = histories.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let sampler_refs = vec![&sampler; positions];
    let start = u32::try_from(history.len())?;
    let absolute_positions = (0..positions)
        .map(|position| {
            start
                .checked_add(position as u32)
                .ok_or_else(|| anyhow!("sampling position overflow"))
        })
        .collect::<Result<Vec<_>>>()?;
    let sampled =
        sample_position_keyed_v1_batch(&sampler_refs, &rows, &history_refs, &absolute_positions)?;
    sampled.to_vec::<u32>().map_err(Into::into)
}

fn sequence_len(array: &Array) -> Result<i32> {
    let shape = array.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(anyhow!(
            "DFlash2 expected single-row [1,S,H] tensor, got {dims:?}"
        ));
    }
    Ok(dims[1])
}

fn sequence_len_batched(array: &Array) -> Result<i32> {
    let shape = array.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] <= 0 {
        return Err(anyhow!(
            "DFlash2 expected batched [B,S,H] tensor, got {dims:?}"
        ));
    }
    Ok(dims[1])
}

fn array_payload_bytes(array: &Array) -> usize {
    array.size().saturating_mul(array.dtype().byte_size())
}

fn slice_batch_row(array: &Array, row: usize, target: StreamOrDevice) -> Result<Array> {
    let shape = array.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || row >= dims[0] as usize {
        return Err(anyhow!(
            "DFlash2 cannot slice batch row {row} from shape {dims:?}"
        ));
    }
    mlx::ops::indexing::slice_strided_on(
        array,
        &[row as i32, 0, 0][..],
        &[row as i32 + 1, dims[1], dims[2]][..],
        &[1_i32, 1, 1][..],
        target,
    )
    .map_err(Into::into)
}

fn materialize_dflash2_draft_tokens(
    draft_tokens: &Array,
    batch_size: usize,
    draft_len: usize,
    mask_token: u32,
) -> Result<Vec<Vec<u32>>> {
    let flat = draft_tokens.to_vec::<u32>()?;
    let rows = flat
        .chunks_exact(draft_len)
        .map(<[u32]>::to_vec)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        rows.len() == batch_size && flat.len() == batch_size.saturating_mul(draft_len),
        "DFlash2 B={batch_size} draft host result has invalid length {}",
        flat.len()
    );
    for tokens in &rows {
        anyhow::ensure!(
            !tokens.contains(&mask_token),
            "DFlash2 draft emitted reserved mask token {mask_token}"
        );
    }
    Ok(rows)
}

fn slice_sequence_position(hidden: &Array, position: i32, target: StreamOrDevice) -> Result<Array> {
    let shape = hidden.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 || position < 0 || position >= dims[1] {
        return Err(anyhow!(
            "DFlash2 cannot slice position {position} from hidden shape {dims:?}"
        ));
    }
    mlx::ops::indexing::slice_strided_on(
        hidden,
        &[0_i32, position, 0][..],
        &[1_i32, position + 1, dims[2]][..],
        &[1_i32, 1, 1][..],
        target,
    )
    .map_err(Into::into)
}

fn slice_sequence_position_batched(
    hidden: &Array,
    position: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = hidden.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] <= 0 || position < 0 || position >= dims[1] {
        return Err(anyhow!(
            "DFlash2 cannot slice batched position {position} from hidden shape {dims:?}"
        ));
    }
    mlx::ops::indexing::slice_strided_on(
        hidden,
        &[0_i32, position, 0][..],
        &[dims[0], position + 1, dims[2]][..],
        &[1_i32, 1, 1][..],
        target,
    )
    .map_err(Into::into)
}

fn slice_sequence_prefix(hidden: &Array, length: i32, target: StreamOrDevice) -> Result<Array> {
    let shape = hidden.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 || length <= 0 || length > dims[1] {
        return Err(anyhow!(
            "DFlash2 cannot slice prefix {length} from hidden shape {dims:?}"
        ));
    }
    mlx::ops::indexing::slice_strided_on(
        hidden,
        &[0_i32, 0, 0][..],
        &[1_i32, length, dims[2]][..],
        &[1_i32, 1, 1][..],
        target,
    )
    .map_err(Into::into)
}

fn retain_context_tail(
    previous: Option<&Array>,
    next: &Array,
    limit: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let combined = match previous {
        Some(previous) => mlx::ops::shape::concatenate_on(&[previous, next], 1, target)?,
        None => next.clone(),
    };
    let len = sequence_len(&combined)?;
    if len <= limit {
        return Ok(combined);
    }
    let shape = combined.shape();
    let hidden = shape.as_slice()[2];
    mlx::ops::indexing::slice_strided_on(
        &combined,
        &[0_i32, len - limit, 0][..],
        &[1_i32, len, hidden][..],
        &[1_i32, 1, 1][..],
        target,
    )
    .map_err(Into::into)
}

fn retain_context_tail_batched(
    previous: Option<&Array>,
    next: &Array,
    limit: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let combined = match previous {
        Some(previous) => mlx::ops::shape::concatenate_on(&[previous, next], 1, target)?,
        None => next.clone(),
    };
    let len = sequence_len_batched(&combined)?;
    if len <= limit {
        return Ok(combined);
    }
    let shape = combined.shape();
    let dims = shape.as_slice();
    mlx::ops::indexing::slice_strided_on(
        &combined,
        &[0_i32, len - limit, 0][..],
        &[dims[0], len, dims[2]][..],
        &[1_i32, 1, 1][..],
        target,
    )
    .map_err(Into::into)
}

/// Diagnostic-only DFlash2 window stage accounting
/// (`IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES=1`): evaluates and synchronizes
/// at stage boundaries and records the wall time of each stage. It removes
/// CPU/GPU overlap, so stage times attribute work but are not production
/// timings. Default off; never enabled by a serving profile.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize)]
pub struct WindowStageRecord {
    pub kind: &'static str,
    pub stages: Vec<(&'static str, u64)>,
    pub drafted: usize,
    pub accepted: usize,
    pub emitted: usize,
    pub context: usize,
}

static WINDOW_STAGE_RECORDS: std::sync::Mutex<Vec<WindowStageRecord>> =
    std::sync::Mutex::new(Vec::new());

/// Diagnostic-only: drain the recorded window stages.
#[doc(hidden)]
pub fn take_window_stage_records() -> Vec<WindowStageRecord> {
    std::mem::take(&mut *WINDOW_STAGE_RECORDS.lock().expect("stage records"))
}

fn window_stages_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES").as_deref() == Ok("1")
    })
}

struct StageClock {
    last: Instant,
    stages: Vec<(&'static str, u64)>,
}

impl StageClock {
    fn start() -> Result<Option<Self>> {
        if !window_stages_enabled() {
            return Ok(None);
        }
        mlx::transforms::synchronize()?;
        Ok(Some(Self {
            last: Instant::now(),
            stages: Vec::new(),
        }))
    }

    fn mark(clock: &mut Option<Self>, stage: &'static str, arrays: &[&Array]) -> Result<()> {
        if let Some(clock) = clock.as_mut() {
            if !arrays.is_empty() {
                mlx::transforms::eval(arrays)?;
            }
            mlx::transforms::synchronize()?;
            let now = Instant::now();
            clock
                .stages
                .push((stage, (now - clock.last).as_micros() as u64));
            clock.last = now;
        }
        Ok(())
    }

    fn finish(
        clock: Option<Self>,
        kind: &'static str,
        drafted: usize,
        accepted: usize,
        emitted: usize,
        context: usize,
    ) {
        if let Some(clock) = clock {
            WINDOW_STAGE_RECORDS
                .lock()
                .expect("stage records")
                .push(WindowStageRecord {
                    kind,
                    stages: clock.stages,
                    drafted,
                    accepted,
                    emitted,
                    context,
                });
        }
    }
}

fn cache_barrier_arrays<'a>(cache: &'a [LayerCache], extra: &'a Array) -> Vec<&'a Array> {
    let mut arrays: Vec<&Array> = cache
        .iter()
        .flat_map(LayerCache::diagnostic_buffers)
        .collect();
    arrays.push(extra);
    arrays
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

/// Diagnostic-only memory phase record. Reads counters; never evaluates arrays,
/// adds barriers or changes graph roots. Enabled by
/// `IRONMLX_DIAGNOSTIC_MEMORY_PHASES=1`.
fn prefill_chunk_cache_reset_requested() -> bool {
    static REQUESTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REQUESTED.get_or_init(|| {
        let enabled = ironmlx_core::m5_profile::flag(
            ironmlx_core::m5_profile::settings::PREFILL_CHUNK_CACHE_RESET,
        );
        if enabled {
            tracing::info!("prefill chunk cache reset enabled");
        }
        enabled
    })
}

/// Diagnostic-only prefill phase timing, enabled by
/// `IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES=1`. Reads the clock only at the
/// prefill's existing synchronization points; adds no evaluation.
struct PrefillPhaseDiagnostic {
    last: Instant,
    /// (phase, size, elapsed us, MLX active MiB, MLX cache MiB, footprint MiB)
    phases: Vec<(&'static str, i32, u64, usize, usize, usize)>,
}

pub(crate) fn prefill_phase_diagnostic_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES").as_deref() == Ok("1")
    })
}

impl PrefillPhaseDiagnostic {
    fn start() -> Option<Self> {
        prefill_phase_diagnostic_enabled().then(|| Self {
            last: Instant::now(),
            phases: Vec::new(),
        })
    }

    fn mark(this: &mut Option<Self>, phase: &'static str, size: i32) {
        if let Some(this) = this {
            let now = Instant::now();
            let us = u64::try_from((now - this.last).as_micros()).unwrap_or(u64::MAX);
            // Counter reads only; no evaluation.
            let memory = mlx::memory::snapshot();
            let footprint = super::process_memory::macos_phys_footprint_bytes().unwrap_or(0);
            this.phases.push((
                phase,
                size,
                us,
                memory.active_bytes >> 20,
                memory.cache_bytes >> 20,
                footprint >> 20,
            ));
            this.last = Instant::now();
        }
    }

    fn finish(
        this: Option<Self>,
        execution: DFlash2PrefillExecution,
        prompt_len: usize,
        hit_tokens: usize,
    ) {
        if let Some(this) = this {
            let phases = this
                .phases
                .iter()
                .map(|(phase, size, us, active, cache, footprint)| {
                    format!("{phase}:{size}:{us}:{active}:{cache}:{footprint}")
                })
                .collect::<Vec<_>>()
                .join(",");
            tracing::warn!(
                diagnostic = "dflash2_prefill_phases",
                execution = ?execution,
                prompt_len,
                hit_tokens,
                phases,
                "diagnostic prefill phases; not a formal measurement"
            );
        }
    }
}

fn memory_phase_diagnostic(phase: &str, position: i32, prompt_len: usize) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED
        .get_or_init(|| std::env::var("IRONMLX_DIAGNOSTIC_MEMORY_PHASES").as_deref() == Ok("1"))
    {
        return;
    }
    let memory = mlx::memory::snapshot();
    let (prepared_count, prepared_bytes) = ironmlx_lm::nn::m5_prepared_totals();
    let (shared_stores, shared_to_tiled, shared_to_native) =
        ironmlx_lm::nn::shared_weight_layout_totals();
    tracing::warn!(
        diagnostic = "memory_phase",
        phase,
        position,
        prompt_len,
        active_bytes = memory.active_bytes,
        cache_bytes = memory.cache_bytes,
        peak_bytes = memory.peak_bytes,
        footprint_bytes = super::process_memory::macos_phys_footprint_bytes().unwrap_or(0),
        m5_prepared_count = prepared_count,
        m5_prepared_bytes = prepared_bytes,
        shared_stores,
        shared_to_tiled,
        shared_to_native,
        "diagnostic memory phase; not a formal resource measurement"
    );
}

fn dflash2_prefill_chunk_len(
    remaining: i32,
    requested_chunk_size: usize,
    position: i32,
    execution: DFlash2PrefillExecution,
) -> i32 {
    let ordinary_chunk = if requested_chunk_size == 0 {
        remaining
    } else {
        (requested_chunk_size as i32).min(remaining)
    };
    if execution == DFlash2PrefillExecution::SchedulerB1
        && position == 0
        && ordinary_chunk == remaining
        && remaining > 1
    {
        remaining - 1
    } else {
        ordinary_chunk
    }
}

fn should_cache_dflash2_prefill_boundary(
    prompt_len: i32,
    position: i32,
    chunk_len: i32,
    requested_chunk_size: usize,
    execution: DFlash2PrefillExecution,
) -> bool {
    let next_position = position + chunk_len;
    if next_position == prompt_len {
        return true;
    }
    let next_remaining = prompt_len - next_position;
    let next_chunk_len = dflash2_prefill_chunk_len(
        next_remaining,
        requested_chunk_size,
        next_position,
        execution,
    );
    next_position + next_chunk_len == prompt_len
}

fn rate_per_second(tokens: usize, elapsed_us: u64) -> f64 {
    if elapsed_us == 0 {
        0.0
    } else {
        tokens as f64 * 1_000_000.0 / elapsed_us as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn automatic_block_size_uses_checkpoint_width_up_to_q8() {
        assert_eq!(
            resolve_dflash2_block_size(None, 4).expect("Q4 auto"),
            DFlash2BlockSizeResolution {
                checkpoint_block_size: 4,
                block_size: 4,
                explicit: false,
            }
        );
        assert_eq!(
            resolve_dflash2_block_size(None, 8)
                .expect("Q8 auto")
                .block_size,
            8
        );
        assert_eq!(
            resolve_dflash2_block_size(None, 32)
                .expect("b32 remains Q8 by default")
                .block_size,
            8
        );
    }

    #[test]
    fn explicit_block_size_preserves_q16_opt_in_and_checkpoint_guard() {
        let q16 = resolve_dflash2_block_size(Some(16), 32).expect("explicit Q16");
        assert_eq!(q16.block_size, 16);
        assert!(q16.explicit);
        assert!(resolve_dflash2_block_size(Some(16), 8).is_err());
        assert!(resolve_dflash2_block_size(Some(1), 8).is_err());
    }

    fn assert_array_exact(label: &str, expected: &Array, actual: &Array) {
        assert_eq!(expected.shape(), actual.shape(), "{label} shape");
        let expected = mlx::ops::cast::astype(expected, mlx::Dtype::Float32)
            .expect("cast expected")
            .to_vec::<f32>()
            .expect("read expected");
        let actual = mlx::ops::cast::astype(actual, mlx::Dtype::Float32)
            .expect("cast actual")
            .to_vec::<f32>()
            .expect("read actual");
        if expected != actual {
            let mut mismatch_count = 0_usize;
            let mut first_mismatch = None;
            let mut max_abs_diff = 0.0_f32;
            for (index, (&left, &right)) in expected.iter().zip(&actual).enumerate() {
                if left != right {
                    mismatch_count += 1;
                    first_mismatch.get_or_insert((index, left, right));
                    max_abs_diff = max_abs_diff.max((left - right).abs());
                }
            }
            panic!(
                "{label}: mismatch_count={mismatch_count} first={first_mismatch:?} max_abs_diff={max_abs_diff}"
            );
        }
    }

    #[test]
    #[ignore = "loads the full local Qwen3.8 target and DFlash2 draft checkpoints"]
    #[serial(mlx_metal)]
    fn qwen38_position_keyed_tree_matches_linear_dflash2() {
        use ironmlx_core::sampler::Sampler;
        use ironmlx_lm::models::{dflash2::DFlash2DraftModel, Qwen35Model};
        use ironmlx_lm::{core::loader::Loader, core::tokenizer::Tokenizer};

        let target_dir = std::env::var("QWEN38_MODEL").expect("QWEN38_MODEL not set");
        let draft_dir = std::env::var("DFLASH2_MODEL").expect("DFLASH2_MODEL not set");
        let mut target_loader = Loader::open(std::path::Path::new(&target_dir))
            .expect("open Qwen3.8 target checkpoint");
        let tokenizer = Tokenizer::from_loader(&target_loader).expect("load tokenizer");
        let target =
            Qwen35Model::from_loader_dflash2(&mut target_loader).expect("load DFlash2 target");
        let draft_loader = Loader::open_dflash2(std::path::Path::new(&draft_dir))
            .expect("open DFlash2 draft checkpoint");
        let draft = DFlash2DraftModel::from_loader(&draft_loader, target.config(), Some(4))
            .expect("load runtime-quantized DFlash2 draft");
        for (label, sampler, position_keyed_sampling) in [
            ("greedy", Sampler::greedy(), false),
            (
                "position-keyed",
                Sampler::greedy()
                    .with_temperature(0.7)
                    .with_top_p(0.9)
                    .with_seed(20_260_928),
                true,
            ),
        ] {
            let request = GenerateRequest {
                priority: Default::default(),
                prompt_ids: vec![151_644, 872, 198, 3_838],
                max_new_tokens: 64,
                sampler,
                stop_token_ids: Vec::new(),
                prefill_chunk_size: 0,
                decode_cadence_mid_chunk_cap: 1,
                kv_cache_turboquant_bits: None,
                pixel_values: None,
                image_grid_thw: None,
                image_spatial_merge_size: 2,
                image_token_id: 248_056,
                constraint: None,
            };
            let mut linear = DFlash2TextGenerationStream::new_text_only_with_options(
                &target,
                &draft,
                &tokenizer,
                request.clone(),
                4,
                DFlash2P2Options {
                    tree_max_nodes: 0,
                    position_keyed_sampling,
                },
            )
            .expect("linear stream");
            let mut tree = DFlash2TextGenerationStream::new_text_only_with_options(
                &target,
                &draft,
                &tokenizer,
                request,
                4,
                DFlash2P2Options {
                    tree_max_nodes: DFlash2DraftTree::MAX_NODES,
                    position_keyed_sampling,
                },
            )
            .expect("tree stream");
            for step in 0..64 {
                let expected = linear
                    .next_token()
                    .expect("linear token")
                    .map(|event| (event.token, event.finish_reason));
                let actual = tree
                    .next_token()
                    .expect("tree token")
                    .map(|event| (event.token, event.finish_reason));
                assert_eq!(actual, expected, "{label} tree divergence at step {step}");
            }
        }
    }

    #[test]
    #[ignore = "loads the full local Qwen3.8 target and block-16-capable DFlash2 draft checkpoints"]
    #[serial(mlx_metal)]
    fn qwen38_b32_q16_linear_matches_ordinary_generation() {
        use ironmlx_core::sampler::Sampler;
        use ironmlx_lm::models::{dflash2::DFlash2DraftModel, Qwen35Model};
        use ironmlx_lm::{core::loader::Loader, core::tokenizer::Tokenizer};

        let target_dir = std::env::var("QWEN38_MODEL").expect("QWEN38_MODEL not set");
        let draft_dir = std::env::var("DFLASH2_MODEL").expect("DFLASH2_MODEL not set");
        let mut target_loader = Loader::open(std::path::Path::new(&target_dir))
            .expect("open Qwen3.8 target checkpoint");
        let tokenizer = Tokenizer::from_loader(&target_loader).expect("load tokenizer");
        let target =
            Qwen35Model::from_loader_dflash2(&mut target_loader).expect("load DFlash2 target");
        let draft_loader = Loader::open_dflash2(std::path::Path::new(&draft_dir))
            .expect("open DFlash2 draft checkpoint");
        let draft = DFlash2DraftModel::from_loader(&draft_loader, target.config(), Some(4))
            .expect("load runtime-quantized DFlash2 draft");
        assert!(
            draft.config().dflash_config.block_size >= 16,
            "Q16 qualification requires a checkpoint block size of at least 16"
        );

        let request = |sampler| GenerateRequest {
            priority: Default::default(),
            prompt_ids: vec![151_644, 872, 198, 3_838],
            max_new_tokens: 64,
            sampler,
            stop_token_ids: Vec::new(),
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 1,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248_056,
            constraint: None,
        };
        {
            let mut ordinary = crate::core::generate::GenerationStream::new_text_only(
                &target,
                &tokenizer,
                request(Sampler::greedy()),
            )
            .expect("ordinary stream");
            let mut dflash = DFlash2TextGenerationStream::new_text_only_with_options(
                &target,
                &draft,
                &tokenizer,
                request(Sampler::greedy()),
                16,
                DFlash2P2Options {
                    tree_max_nodes: 0,
                    position_keyed_sampling: false,
                },
            )
            .expect("Q16 DFlash2 stream");
            for step in 0..64 {
                let expected = ordinary
                    .next_token()
                    .expect("ordinary token")
                    .map(|event| (event.token, event.finish_reason));
                let actual = dflash
                    .next_token()
                    .expect("Q16 DFlash2 token")
                    .map(|event| (event.token, event.finish_reason));
                assert_eq!(actual, expected, "greedy Q16 divergence at step {step}");
            }
            let metrics = dflash.metrics();
            assert_eq!(metrics.block_size, 16);
            assert_eq!(metrics.generated_tokens, 64);
            assert!(metrics.windows > 0, "greedy executed no draft windows");
            assert!(
                metrics.drafted_tokens >= 15,
                "greedy did not execute an initial fifteen-token Q16 draft"
            );
        }

        let sampled = Sampler::greedy()
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_seed(20_260_928);
        {
            // Position-keyed sampling is a DFlash2 opt-in rather than the
            // ordinary stream's stateful sampling contract. Use Q8 as the
            // width-independent oracle for the same absolute positions.
            let mut q8 = DFlash2TextGenerationStream::new_text_only_with_options(
                &target,
                &draft,
                &tokenizer,
                request(sampled),
                8,
                DFlash2P2Options {
                    tree_max_nodes: 0,
                    position_keyed_sampling: true,
                },
            )
            .expect("Q8 position-keyed DFlash2 stream");
            let mut q16 = DFlash2TextGenerationStream::new_text_only_with_options(
                &target,
                &draft,
                &tokenizer,
                request(sampled),
                16,
                DFlash2P2Options {
                    tree_max_nodes: 0,
                    position_keyed_sampling: true,
                },
            )
            .expect("Q16 position-keyed DFlash2 stream");
            for step in 0..64 {
                let expected = q8
                    .next_token()
                    .expect("Q8 position-keyed token")
                    .map(|event| (event.token, event.finish_reason));
                let actual = q16
                    .next_token()
                    .expect("Q16 position-keyed token")
                    .map(|event| (event.token, event.finish_reason));
                assert_eq!(
                    actual, expected,
                    "position-keyed Q8/Q16 divergence at step {step}"
                );
            }
            let metrics = q16.metrics();
            assert!(
                metrics.drafted_tokens >= 15,
                "position-keyed sampling did not execute an initial fifteen-token Q16 draft"
            );
        }

        let mut stateful = DFlash2TextGenerationStream::new_text_only_with_options(
            &target,
            &draft,
            &tokenizer,
            request(sampled),
            16,
            DFlash2P2Options::default(),
        )
        .expect("Q16 stateful-exact DFlash2 stream");
        for _ in 0..64 {
            stateful
                .next_token()
                .expect("Q16 stateful-exact token")
                .expect("Q16 stateful-exact stream ended early");
        }
        let metrics = stateful.metrics();
        assert_eq!(metrics.generated_tokens, 64);
        assert!(metrics.exact_sampling_windows > 0);
        assert!(
            metrics.drafted_tokens >= 15,
            "stateful exact sampling did not execute an initial fifteen-token Q16 draft"
        );
    }

    #[test]
    #[ignore = "loads the full local Qwen3.8 target and DFlash2 draft checkpoints"]
    #[serial(mlx_metal)]
    fn qwen38_dflash2_batched_prefill_matches_scheduler_b1_exactly() {
        use ironmlx_core::sampler::Sampler;
        use ironmlx_lm::models::dflash2::DFlash2DraftModel;
        use ironmlx_lm::models::Qwen35Model;
        use {ironmlx_lm::core::loader::Loader, ironmlx_lm::core::tokenizer::Tokenizer};

        let target_dir = std::env::var("QWEN38_MODEL").expect("QWEN38_MODEL not set");
        let draft_dir = std::env::var("DFLASH2_MODEL").expect("DFLASH2_MODEL not set");
        let mut target_loader = Loader::open(std::path::Path::new(&target_dir))
            .expect("open Qwen3.8 target checkpoint");
        let tokenizer = Tokenizer::from_loader(&target_loader).expect("load tokenizer");
        let target =
            Qwen35Model::from_loader_dflash2(&mut target_loader).expect("load DFlash2 target");
        let draft_loader = Loader::open_dflash2(std::path::Path::new(&draft_dir))
            .expect("open DFlash2 draft checkpoint");
        let draft = DFlash2DraftModel::from_loader(&draft_loader, target.config(), Some(4))
            .expect("load runtime-quantized DFlash2 draft");
        let request = |prompt_ids: Vec<u32>| GenerateRequest {
            priority: Default::default(),
            prompt_ids,
            max_new_tokens: 256,
            sampler: Sampler::greedy(),
            stop_token_ids: Vec::new(),
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 1,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248_056,
            constraint: None,
        };
        let requests = vec![
            request(vec![151_644, 872, 198, 3_838]),
            request(vec![151_644, 872, 198, 10_264]),
        ];
        let mut references = requests
            .iter()
            .cloned()
            .map(|request| {
                DFlash2TextGenerationStream::new_scheduler_b1_text_only_with_cancellation(
                    &target,
                    &draft,
                    &tokenizer,
                    request,
                    DFlash2ExecutionOptions {
                        block_size: 4,
                        p2: DFlash2P2Options::default(),
                    },
                    None,
                    &|| false,
                )
                .expect("B1 DFlash2 prefill")
            })
            .collect::<Vec<_>>();
        let mut batched =
            DFlash2TextGenerationStream::new_scheduler_bn_text_only_with_cancellation(
                &target,
                &draft,
                &tokenizer,
                requests,
                4,
                DFlash2P2Options::default(),
                &|_| false,
            )
            .expect("batched DFlash2 prefill");
        eprintln!(
            "dflash2_prefill_timing b1_row_us={:?} batched_us={}",
            references
                .iter()
                .map(|stream| stream.prefill_us)
                .collect::<Vec<_>>(),
            batched[0].prefill_us
        );

        for row in 0..2 {
            assert_eq!(
                references[row].history, batched[row].history,
                "row {row} history"
            );
            assert_array_exact(
                &format!("row {row} retained target context"),
                &references[row].pending_context_hidden,
                &batched[row].pending_context_hidden,
            );
        }
        for step in 0..256 {
            for row in 0..2 {
                let expected = references[row]
                    .next_token()
                    .expect("B1 token")
                    .map(|event| (event.token, event.finish_reason));
                let actual = batched[row]
                    .next_token()
                    .expect("batched-prefill token")
                    .map(|event| (event.token, event.finish_reason));
                assert_eq!(expected, actual, "row {row} step {step}");
            }
        }
    }

    #[test]
    #[ignore = "loads the full local Qwen3.8 target and DFlash2 draft checkpoints"]
    #[serial(mlx_metal)]
    fn qwen38_dflash2_b4_windows_are_row_exact_and_greedy_matches_scheduler_b1() {
        use ironmlx_core::sampler::Sampler;
        use ironmlx_lm::models::dflash2::DFlash2DraftModel;
        use ironmlx_lm::models::Qwen35Model;
        use {
            ironmlx_lm::core::chat_template::Message, ironmlx_lm::core::loader::Loader,
            ironmlx_lm::core::tokenizer::Tokenizer,
        };

        let target_dir = std::env::var("QWEN38_MODEL").expect("QWEN38_MODEL not set");
        let draft_dir = std::env::var("DFLASH2_MODEL").expect("DFLASH2_MODEL not set");
        let mut target_loader = Loader::open(std::path::Path::new(&target_dir))
            .expect("open Qwen3.8 target checkpoint");
        let tokenizer = Tokenizer::from_loader(&target_loader).expect("load tokenizer");
        let target =
            Qwen35Model::from_loader_dflash2(&mut target_loader).expect("load DFlash2 target");
        let draft_loader = Loader::open_dflash2(std::path::Path::new(&draft_dir))
            .expect("open DFlash2 draft checkpoint");
        let draft = DFlash2DraftModel::from_loader(&draft_loader, target.config(), Some(4))
            .expect("load runtime-quantized DFlash2 draft");
        let prompt = tokenizer
            .apply_chat_template(
                &[Message {
                    role: "user".to_owned(),
                    content: "Use Rust language to write a function for computing the nth Fibonacci number. Explain overflow handling and include tests for n = 0, 1, 10, and 93."
                        .to_owned(),
                }],
                true,
                Some(&serde_json::json!({"enable_thinking": false})),
            )
            .expect("render benchmark chat prompt");
        let prompt_ids = tokenizer
            .encode(&prompt, false)
            .expect("encode benchmark chat prompt");
        eprintln!("dflash2_b4_exact_prompt_tokens={}", prompt_ids.len());
        for (case, max_new_tokens, sampler) in [
            ("greedy", 64, Sampler::greedy()),
            (
                "sampled",
                256,
                Sampler::greedy()
                    .with_temperature(0.7)
                    .with_top_p(0.9)
                    .with_seed(20_260_824),
            ),
        ] {
            let request = GenerateRequest {
                priority: Default::default(),
                prompt_ids: prompt_ids.clone(),
                max_new_tokens,
                sampler,
                stop_token_ids: tokenizer.eos_token_ids().to_vec(),
                prefill_chunk_size: 0,
                decode_cadence_mid_chunk_cap: 1,
                kv_cache_turboquant_bits: None,
                pixel_values: None,
                image_grid_thw: None,
                image_spatial_merge_size: 2,
                image_token_id: 248_056,
                constraint: None,
            };
            let mut reference =
                DFlash2TextGenerationStream::new_scheduler_b1_text_only_with_cancellation(
                    &target,
                    &draft,
                    &tokenizer,
                    request.clone(),
                    DFlash2ExecutionOptions {
                        block_size: 3,
                        p2: DFlash2P2Options::default(),
                    },
                    None,
                    &|| false,
                )
                .expect("B1 DFlash2 stream");
            let mut batched =
                DFlash2TextGenerationStream::new_scheduler_bn_text_only_with_cancellation(
                    &target,
                    &draft,
                    &tokenizer,
                    vec![request; 4],
                    3,
                    DFlash2P2Options::default(),
                    &|_| false,
                )
                .expect("B4 DFlash2 streams");
            let mut tensor_cache: Option<DFlash2TensorBatchCache> = None;

            for (row, stream) in batched.iter().enumerate() {
                assert_array_exact(
                    &format!("{case} B4 row {row} prefill context"),
                    &reference.pending_context_hidden,
                    &stream.pending_context_hidden,
                );
            }

            for step in 0..max_new_tokens {
                let expected = reference
                    .next_token()
                    .expect("B1 token")
                    .map(|event| (event.token, event.finish_reason));
                let mut first_b4 = None;
                for (row, stream) in batched.iter_mut().enumerate() {
                    let actual = stream
                        .next_token_deferred()
                        .expect("B4 token")
                        .map(|event| (event.token, event.finish_reason));
                    if case == "greedy" {
                        assert_eq!(expected, actual, "{case} row {row} step {step}");
                    }
                    if let Some(first_b4) = first_b4.as_ref() {
                        assert_eq!(first_b4, &actual, "{case} row {row} step {step}");
                    } else {
                        first_b4 = Some(actual);
                    }
                }
                let should_fill = if case == "greedy" {
                    expected
                        .as_ref()
                        .is_some_and(|(_, finish_reason)| finish_reason.is_none())
                } else {
                    first_b4.as_ref().is_some_and(|actual| {
                        actual
                            .as_ref()
                            .is_some_and(|(_, finish_reason)| finish_reason.is_none())
                    })
                };
                if should_fill {
                    let keys = batched
                        .iter()
                        .map(|stream| stream.tensor_batch_key().expect("batch key"))
                        .collect::<Vec<_>>();
                    if keys.iter().all(Option::is_some) {
                        assert!(keys.iter().all(|key| *key == keys[0]));
                        let mut rows = batched.iter_mut().collect::<Vec<_>>();
                        if keys[0].is_some_and(DFlash2TensorBatchKey::is_ordinary_decode) {
                            if let Some(cache) = tensor_cache.take() {
                                cache
                                    .scatter_to_rows(&mut rows)
                                    .expect("scatter B4 cache for Q1 control window");
                            }
                            for stream in rows {
                                stream
                                    .fill_deferred_window_b1()
                                    .expect("B1 ordinary control window");
                            }
                        } else {
                            tensor_cache = DFlash2TextGenerationStream::fill_deferred_window_bn(
                                &mut rows,
                                tensor_cache.take(),
                            )
                            .expect("B4 tensor window");
                        }
                    }
                    for row in 1..batched.len() {
                        assert_array_exact(
                            &format!("{case} B4 row {row} step {step} row-exact context"),
                            &batched[0].pending_context_hidden,
                            &batched[row].pending_context_hidden,
                        );
                    }
                    for (row, stream) in batched.iter().enumerate() {
                        if case == "greedy"
                            && reference.pending_context_hidden.shape()
                                == stream.pending_context_hidden.shape()
                        {
                            assert_array_exact(
                                &format!("{case} B4 row {row} step {step} aligned context"),
                                &reference.pending_context_hidden,
                                &stream.pending_context_hidden,
                            );
                        }
                        if stream.draft_policy.should_maintain_mtp_cache() {
                            let processed = stream
                                .draft_cache
                                .position_signature()
                                .expect("B4 draft position")
                                .0;
                            let pending = stream.pending_context_hidden.shape().as_slice()[1];
                            assert_eq!(
                                processed + pending,
                                i32::try_from(stream.history.len() - 1)
                                    .expect("B4 history position"),
                                "{case} B4 row {row} step {step} resumable draft position"
                            );
                        }
                    }
                }
            }
        }
    }

    fn test_prefix_artifact(
        token_ids: &[u32],
        fingerprint: &str,
        payload_bytes: usize,
        generation: u64,
    ) -> DFlash2PrefixArtifact {
        let hidden = Array::zeros((1_i32, 1_i32, 2_i32), mlx::Dtype::Float32).expect("hidden");
        DFlash2PrefixArtifact {
            token_ids: token_ids.to_vec(),
            fingerprint: fingerprint.to_owned(),
            target_cache: PagedPrefixEntry::default(),
            context_hidden: hidden.clone(),
            last_hidden: hidden,
            cached_len: i32::try_from(token_ids.len()).expect("cached len"),
            payload_bytes,
            generation,
        }
    }

    fn fixed_json_constraint() -> ConstraintSession {
        let tokenizer =
            ironmlx_lm::test_support::byte_level_constraint_tokenizer().expect("byte tokenizer");
        let plan = tokenizer
            .compile_json_output(&serde_json::json!({
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "const": "done"}
                },
                "required": ["answer"],
                "additionalProperties": false
            }))
            .expect("compile JSON constraint");
        plan.start_session().expect("start constraint")
    }

    #[test]
    fn cancellation_guard_is_fail_closed_only_when_signalled() {
        assert!(ensure_dflash2_request_not_cancelled(None).is_ok());
        assert!(ensure_dflash2_request_not_cancelled(Some(&|| false)).is_ok());
        let error = ensure_dflash2_request_not_cancelled(Some(&|| true))
            .expect_err("cancelled request must stop at the next safe boundary");
        assert_eq!(error.to_string(), "DFlash2 request cancelled");
    }

    #[test]
    fn p2_options_keep_stable_defaults_and_enforce_tree_cap() {
        let stable = DFlash2P2Options::default();
        assert_eq!(stable.tree_max_nodes, 0);
        assert!(!stable.position_keyed_sampling);
        stable.validate().expect("stable defaults");
        DFlash2P2Options {
            tree_max_nodes: DFlash2DraftTree::MAX_NODES,
            position_keyed_sampling: true,
        }
        .validate()
        .expect("maximum P2 tree");
        assert!(DFlash2P2Options {
            tree_max_nodes: DFlash2DraftTree::MAX_NODES + 1,
            position_keyed_sampling: false,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn qualification_tensor_q16_is_explicit_and_lane_bounded() {
        use ironmlx_lm::models::dflash2::DFlash2LaneKernelPack;

        let capabilities = DFlash2VerifyCapabilities {
            profile: "qwen35-affine4".to_owned(),
            row_bit_exact_qmm: true,
            row_bit_exact_attention: true,
            transactional_state_restore: true,
            supported_shapes: vec![DFlash2VerifyShape {
                batch_width: 2,
                verify_width: 8,
            }],
            lane_kernel_pack: Some(DFlash2LaneKernelPack {
                family: "qwen3.8-27b-dflash2".to_owned(),
                revision: 1,
                quant_bits: 4,
                quant_group_size: 64,
                max_lanes: 64,
                prepared_layout: "test".to_owned(),
                attention_layout: "test".to_owned(),
                state_layout: "test".to_owned(),
                layer_submit_interval: 4,
            }),
        };
        let stable = extend_batched_q16_qualification_capabilities(capabilities.clone(), false)
            .expect("stable capabilities");
        assert!(!stable.supports(2, 16));

        let qualification = extend_batched_q16_qualification_capabilities(capabilities, true)
            .expect("qualification capabilities");
        assert!(qualification.supports(2, 16));
        assert!(qualification.supports(4, 16));
        assert!(!qualification.supports(5, 16));
    }

    #[test]
    fn tree_resolution_follows_matching_branch_and_returns_leaf_bonus() {
        let tree = DFlash2DraftTree::new(vec![10, 11, 20, 21], vec![-1, -1, 0, 0]).expect("tree");
        let paths = tree.leaf_paths();
        assert_eq!(paths, vec![vec![1], vec![0, 2], vec![0, 3]]);
        let target_tokens = vec![
            10, 0, 0, // shared root prediction from row zero
            10, 21, 0, // representative for node 0 predicts node 3
            10, 21, 99, // representative for node 3 predicts the bonus
        ];
        let resolution =
            resolve_tree_tokens(&tree, &paths, 3, &target_tokens).expect("resolve accepted branch");
        assert_eq!(resolution.accepted_nodes, vec![0, 3]);
        assert_eq!(resolution.bonus_token, 99);
        assert_eq!(resolution.row, 2);

        let mut rejected = target_tokens;
        rejected[0] = 77;
        let resolution =
            resolve_tree_tokens(&tree, &paths, 3, &rejected).expect("resolve root miss");
        assert!(resolution.accepted_nodes.is_empty());
        assert_eq!(resolution.bonus_token, 77);
        assert_eq!(resolution.row, 0);
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefix_cache_restores_the_longest_matching_runtime_artifact() {
        let mut cache = DFlash2PrefixCache::new(1024).expect("cache");
        cache.entries = vec![
            test_prefix_artifact(&[1, 2], "runtime-a", 64, 1),
            test_prefix_artifact(&[1, 2, 3, 4], "runtime-a", 96, 2),
            test_prefix_artifact(&[1, 2, 3, 4, 5], "runtime-b", 128, 3),
        ];
        cache.total_bytes = 288;

        let hit = cache
            .load_longest(&[1, 2, 3, 4, 9], "runtime-a")
            .expect("longest hit");
        assert_eq!(hit.token_ids, vec![1, 2, 3, 4]);
        assert_eq!(cache.snapshot().hits, 1);

        assert!(cache.load_longest(&[1, 7], "runtime-a").is_none());
        assert_eq!(cache.snapshot().misses, 1);
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefix_cache_pressure_shrink_evicts_least_recent_artifact() {
        let mut cache = DFlash2PrefixCache::new(1024).expect("cache");
        cache.entries = vec![
            test_prefix_artifact(&[1], "runtime", 100, 1),
            test_prefix_artifact(&[2], "runtime", 100, 2),
        ];
        cache.total_bytes = 200;

        assert_eq!(cache.shrink_to(100), 100);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.entries[0].token_ids, vec![2]);
        assert_eq!(cache.snapshot().evictions, 1);
    }

    #[test]
    #[serial(mlx_metal)]
    fn initial_sampling_masks_an_invalid_highest_logit() {
        let mut values = vec![-100.0_f32; 257];
        values[usize::from(b'x')] = 100.0;
        values[usize::from(b'{')] = 50.0;
        let logits: Array = (&values[..], &[1_i32, 1, 257][..])
            .try_into()
            .expect("logits");
        let mut key = mlx::random::key(0).expect("key");
        let mut constraint = Some(fixed_json_constraint());

        let token = sample_initial_token(
            &logits,
            ironmlx_core::sampler::Sampler::greedy(),
            &[],
            &mut key,
            &mut constraint,
        )
        .expect("sample constrained token");

        assert_eq!(token, u32::from(b'{'));
    }

    #[test]
    #[serial(mlx_metal)]
    fn verified_logits_apply_every_prefix_conditioned_constraint_mask() {
        let session = fixed_json_constraint();
        let draft = [u32::from(b'x'), u32::from(b'x')];
        let masks = session
            .speculative_masks(&draft)
            .expect("speculative masks");
        let allowed = masks
            .iter()
            .map(|mask| {
                (0_u32..257)
                    .find(|token| mask.is_allowed(*token))
                    .expect("allowed token")
            })
            .collect::<Vec<_>>();
        let mut values = vec![-100.0_f32; 3 * 257];
        for (step, token) in allowed.iter().copied().enumerate() {
            values[step * 257 + usize::try_from(token).expect("token index")] = 50.0;
            values[step * 257 + usize::from(b'x')] = 100.0;
        }
        let logits: Array = (&values[..], &[1_i32, 3, 257][..])
            .try_into()
            .expect("logits");

        let constrained = constrain_dflash2_verified_logits(Some(&session), &logits, &draft)
            .expect("constrained logits");
        let tokens = mlx::ops::reduction::argmax(&constrained, -1, false)
            .expect("argmax")
            .to_vec::<u32>()
            .expect("materialize tokens");

        assert_eq!(tokens, allowed);
        assert!(tokens.iter().all(|token| *token != u32::from(b'x')));
    }

    #[test]
    #[serial(mlx_metal)]
    fn retained_context_keeps_only_the_sliding_tail() {
        let first: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], &[1_i32, 2, 2][..])
            .try_into()
            .expect("first");
        let second: Array = (&[5.0_f32, 6.0, 7.0, 8.0][..], &[1_i32, 2, 2][..])
            .try_into()
            .expect("second");
        let retained = retain_context_tail(Some(&first), &second, 3, StreamOrDevice::default())
            .expect("retain")
            .to_vec::<f32>()
            .expect("materialize");
        assert_eq!(retained, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn target_prefill_preserves_the_selected_execution_morphology() {
        assert_eq!(
            dflash2_prefill_chunk_len(33, 2048, 0, DFlash2PrefillExecution::GenerationStream,),
            33
        );
        assert_eq!(
            dflash2_prefill_chunk_len(33, 2048, 0, DFlash2PrefillExecution::SchedulerB1,),
            32
        );
        assert_eq!(
            dflash2_prefill_chunk_len(1, 2048, 32, DFlash2PrefillExecution::SchedulerB1,),
            1
        );
        assert_eq!(
            dflash2_prefill_chunk_len(4097, 2048, 0, DFlash2PrefillExecution::SchedulerB1,),
            2048
        );
        assert_eq!(
            dflash2_prefill_chunk_len(33, 0, 0, DFlash2PrefillExecution::GenerationStream,),
            33
        );
    }

    #[test]
    fn prefix_cache_retains_only_penultimate_and_full_prefill_boundaries() {
        let execution = DFlash2PrefillExecution::SchedulerB1;
        assert!(!should_cache_dflash2_prefill_boundary(
            4097, 0, 2048, 2048, execution,
        ));
        assert!(should_cache_dflash2_prefill_boundary(
            4097, 2048, 2048, 2048, execution,
        ));
        assert!(should_cache_dflash2_prefill_boundary(
            4097, 4096, 1, 2048, execution,
        ));
        assert!(should_cache_dflash2_prefill_boundary(
            33, 0, 32, 2048, execution,
        ));
        assert!(should_cache_dflash2_prefill_boundary(
            33, 32, 1, 2048, execution,
        ));
    }

    #[test]
    #[serial(mlx_metal)]
    fn exact_sampling_same_seed_replays_dflash2_window_and_prng() {
        let logits: Array = (
            &[
                0.0_f32, 1.0, 2.0, 3.0, //
                3.0, 2.0, 1.0, 0.0, //
                0.5, 1.5, 2.5, 3.5,
            ][..],
            &[1_i32, 3, 4][..],
        )
            .try_into()
            .expect("logits");
        let sampler = ironmlx_core::sampler::Sampler::greedy()
            .with_temperature(0.8)
            .with_top_p(0.95)
            .with_seed(71);
        let mut key_a = mlx::random::key(sampler.seed).expect("key a");
        let mut key_b = mlx::random::key(sampler.seed).expect("key b");

        let resolution_a =
            resolve_dflash2_window(&[3, 0], None, &logits, sampler, &[1, 2], &mut key_a)
                .expect("resolution a");
        let resolution_b =
            resolve_dflash2_window(&[3, 0], None, &logits, sampler, &[1, 2], &mut key_b)
                .expect("resolution b");

        assert_eq!(resolution_a, resolution_b);
        assert_eq!(resolution_a.exact_sampling().windows, 1);
        assert!(resolution_a.exact_sampling().acceptance_draws > 0);
        assert_eq!(
            key_a.to_vec::<u32>().expect("key a values"),
            key_b.to_vec::<u32>().expect("key b values")
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn deterministic_penalty_sampling_uses_target_tokens_without_exact_draws() {
        let logits: Array = (
            &[
                0.0_f32, 4.0, 3.0, //
                0.0, 4.0, 3.0, //
                0.0, 4.0, 3.0,
            ][..],
            &[1_i32, 3, 3][..],
        )
            .try_into()
            .expect("logits");
        let sampler = ironmlx_core::sampler::Sampler::greedy().with_repetition_penalty(2.0);
        let mut key = mlx::random::key(0).expect("key");

        let resolution = resolve_dflash2_window(&[1, 1], None, &logits, sampler, &[1], &mut key)
            .expect("resolution");

        assert_eq!(resolution.tokens_to_append[0], 2);
        assert_eq!(resolution.accepted_draft_len, 0);
        assert_eq!(
            resolution.exact_sampling(),
            ExactSamplingCounters::default()
        );
    }

    #[test]
    fn sampler_selects_greedy_or_sampled_target_mode() {
        assert_eq!(
            dflash2_target_forward_mode(ironmlx_core::sampler::Sampler::greedy()),
            DFlash2TargetForwardMode::GreedyVerify
        );
        assert_eq!(
            dflash2_target_forward_mode(
                ironmlx_core::sampler::Sampler::greedy().with_temperature(0.8)
            ),
            DFlash2TargetForwardMode::SampledVerify
        );
        assert_eq!(
            dflash2_target_forward_mode(
                ironmlx_core::sampler::Sampler::greedy().with_repetition_penalty(1.1)
            ),
            DFlash2TargetForwardMode::SampledVerify
        );
    }

    #[test]
    fn tensor_batch_key_selects_only_certified_group_widths() {
        let key = DFlash2TensorBatchKey {
            draft_len: 3,
            verify_start: 64,
            context_len: 32,
            draft_processed: 32,
            draft_retained: 32,
            supported_batch_widths: (1_u64 << 1) | (1_u64 << 2) | (1_u64 << 4),
            sampled: false,
        };

        assert!(key.supports_batch_width(4));
        assert!(!key.supports_batch_width(3));
        assert_eq!(key.largest_supported_batch_width(4), 4);
        assert_eq!(key.largest_supported_batch_width(3), 2);
        assert_eq!(key.largest_supported_batch_width(1), 1);
    }
}
