//! Standalone DFlash2 greedy and exact sampled generation path.
//!
//! This module deliberately does not share the MTP scheduler or MTP stream
//! state machine. The only shared piece is model-agnostic token resolution.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::{anyhow, Context};
use mlx::{Array, StreamOrDevice};
use serde::Serialize;
use thiserror::Error;

use crate::core::cache::PagedPrefixEntry;
use crate::core::constrained::{
    apply_speculative_token_masks, apply_token_mask, ConstraintSession,
};
use crate::core::generate::{build_position_ids, GenerateEvent, GenerateRequest};
use crate::core::sampler::{
    prepare_target_tokens_with_uniforms_batch, prepare_uniforms, PreparedTargetTokenSampling,
};
use crate::core::speculative::{
    resolve_exact_deterministic_target_logits, resolve_exact_deterministic_target_tokens,
    resolve_speculative_tokens, sample_logits_positions, ExactSamplingCounters,
    SpeculativeResolution,
};
use crate::core::tokenizer::{DecodeStream, Tokenizer};
use crate::models::dflash2::{
    DFlash2DraftCache, DFlash2DraftModel, DFlash2Target, DFlash2TargetForwardMode,
};
use crate::nn::{prefix_entry_for_row, restore_prefix_entry_for_row, LayerCache};
use crate::Result;

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
pub(crate) enum DFlash2RequestError {
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
    pub sampled: bool,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub windows: usize,
    pub drafted_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub rollback_count: usize,
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
    sampled: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DFlash2PrefillExecution {
    GenerationStream,
    SchedulerB1,
}

struct DFlash2PrefillContext<'a> {
    execution: DFlash2PrefillExecution,
    prefix_cache: Option<(&'a mut DFlash2PrefixCache, &'a str)>,
    is_cancelled: Option<&'a dyn Fn() -> bool>,
}

/// Text-only, single-request DFlash2 stream with greedy and exact sampled decoding.
pub(crate) struct DFlash2TextGenerationStream<'m, M>
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
    prng_state: Array,
    block_size: usize,
    emitted_new_tokens: usize,
    finished: bool,
    prefill_us: u64,
    generation_started: Instant,
    counters: DFlash2Counters,
    constraint: Option<ConstraintSession>,
    prefix_cache_hit_tokens: usize,
}

pub(crate) struct DFlash2TensorBatchCache {
    target: Vec<LayerCache>,
    draft: DFlash2DraftCache,
    batch_size: usize,
}

impl DFlash2TensorBatchCache {
    pub(crate) fn scatter_to_rows<M: DFlash2Target>(
        &self,
        rows: &mut [&mut DFlash2TextGenerationStream<'_, M>],
    ) -> Result<()> {
        anyhow::ensure!(
            rows.len() == self.batch_size,
            "DFlash2 tensor cache width {} cannot scatter into {} rows",
            self.batch_size,
            rows.len()
        );
        let target = StreamOrDevice::default();
        for (batch_row, row) in rows.iter_mut().enumerate() {
            crate::nn::decoder_layer::adopt_layer_cache_rows(
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
        Self::new_text_only_with_prefill_execution(
            model,
            draft,
            tokenizer,
            request,
            block_size,
            DFlash2PrefillContext {
                execution: DFlash2PrefillExecution::GenerationStream,
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
        block_size: usize,
        prefix_cache: Option<(&mut DFlash2PrefixCache, &str)>,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Self> {
        Self::new_text_only_with_prefill_execution(
            model,
            draft,
            tokenizer,
            request,
            block_size,
            DFlash2PrefillContext {
                execution: DFlash2PrefillExecution::SchedulerB1,
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
        is_cancelled: &dyn Fn(usize) -> bool,
    ) -> Result<Vec<Self>> {
        anyhow::ensure!(
            requests.len() > 1,
            "DFlash2 batched prefill requires at least two requests"
        );
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
        let cap = i32::try_from(cap)?.max(crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
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
            crate::nn::decoder_layer::adopt_layer_cache_rows(
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
                prng_state,
                block_size,
                emitted_new_tokens: 0,
                finished: false,
                prefill_us,
                generation_started: Instant::now(),
                counters: DFlash2Counters::default(),
                constraint,
                prefix_cache_hit_tokens: 0,
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
        prefill: DFlash2PrefillContext<'_>,
    ) -> Result<Self> {
        let DFlash2PrefillContext {
            execution: prefill_execution,
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
            .max(crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let mut target_cache = model.make_cache(1, cap, model.cache_dtype())?;
        let target_layer_ids = &draft.config().dflash_config.target_layer_ids;
        let context_limit = draft.config().sliding_window - 1;
        let prefill_started = Instant::now();
        let mut context_hidden: Option<Array> = None;
        let mut last_hidden: Option<Array> = None;
        let mut position = 0_i32;
        let mut prefix_cache_hit_tokens = 0_usize;
        let prompt_len_i32 = i32::try_from(prompt_len).context("DFlash2 prompt is too long")?;

        if let Some(cache) = prefix_cache.as_deref_mut() {
            if let Some(artifact) = cache.load_longest(&request.prompt_ids, prefix_fingerprint) {
                let restore_result = restore_prefix_entry_for_row(
                    &mut target_cache,
                    &artifact.target_cache,
                    0,
                    artifact.cached_len,
                )
                .and_then(|()| {
                    materialize_dflash2_target_cache_prefix(&target_cache, artifact.cached_len)
                });
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

        while position < prompt_len_i32 {
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
            prng_state,
            block_size,
            emitted_new_tokens: 0,
            finished: false,
            prefill_us,
            generation_started,
            counters: DFlash2Counters::default(),
            constraint,
            prefix_cache_hit_tokens,
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
            sampled: self.request.sampler.temperature > 0.0,
            prompt_tokens,
            generated_tokens: self.emitted_new_tokens,
            windows: self.counters.windows,
            drafted_tokens: self.counters.drafted_tokens,
            accepted_draft_tokens: self.counters.accepted_draft_tokens,
            rollback_count: self.counters.rollback_count,
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
                self.fill_window(event.token)?;
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
        let context_shape = self.pending_context_hidden.shape();
        let context_dims = context_shape.as_slice();
        anyhow::ensure!(
            context_dims.len() == 3 && context_dims[0] == 1,
            "DFlash2 pending context must be [1,S,H], got {context_dims:?}"
        );
        let (draft_processed, draft_retained) = self.draft_cache.position_signature()?;
        Ok(Some(DFlash2TensorBatchKey {
            draft_len: (self.block_size - 1).min(remaining),
            verify_start: self.history.len() - 1,
            context_len: context_dims[1],
            draft_processed,
            draft_retained,
            sampled: !self.request.sampler.is_greedy(),
        }))
    }

    pub(crate) fn fill_deferred_window_b1(&mut self) -> Result<()> {
        let current_token = *self
            .history
            .last()
            .ok_or_else(|| anyhow!("DFlash2 stream history is empty"))?;
        self.fill_window(current_token)
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
        let verify_len = draft_len + 1;
        let mask_token = rows[0].draft.config().dflash_config.mask_token_id;
        let sampling_prepare_started = Instant::now();
        let mut exact_uniforms = Vec::with_capacity(batch_size);
        for row in rows.iter_mut() {
            exact_uniforms.push(
                (row.request.sampler.temperature > 0.0
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
                let cap =
                    i32::try_from(cap)?.max(crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
                let mut target_cache =
                    rows[0]
                        .model
                        .make_cache(batch_size_i32, cap, rows[0].model.cache_dtype())?;
                for (batch_row, row) in rows.iter().enumerate() {
                    crate::nn::decoder_layer::adopt_layer_cache_rows(
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
            .map(LayerCache::snapshot)
            .collect::<Vec<_>>();
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
            row.pending_context_hidden = context_row;
            row.counters.windows += 1;
            row.counters.drafted_tokens += draft_tokens[batch_row].len();
            row.counters.accepted_draft_tokens += resolution.accepted_draft_len;
            row.counters.exact_sampling.windows = row
                .counters
                .exact_sampling
                .windows
                .saturating_add(resolution.exact_sampling.windows);
            row.counters.exact_sampling.acceptance_draws = row
                .counters
                .exact_sampling
                .acceptance_draws
                .saturating_add(resolution.exact_sampling.acceptance_draws);
            row.counters.exact_sampling.residual_corrections = row
                .counters
                .exact_sampling
                .residual_corrections
                .saturating_add(resolution.exact_sampling.residual_corrections);
            row.counters.exact_sampling.bonus_samples = row
                .counters
                .exact_sampling
                .bonus_samples
                .saturating_add(resolution.exact_sampling.bonus_samples);
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
            for token in tokens_to_append {
                commit_constraint_token(&mut row.constraint, token)?;
                row.history.push(token);
                row.pending_tokens.push_back(token);
            }
        }
        Ok(keep_batch_cache.then_some(batch_cache))
    }

    fn fill_window(&mut self, current_token: u32) -> Result<()> {
        let window_started = Instant::now();
        let remaining = self
            .request
            .max_new_tokens
            .saturating_sub(self.emitted_new_tokens);
        if remaining == 0 {
            return Ok(());
        }
        let draft_len = (self.block_size - 1).min(remaining);
        let sampling_prepare_started = Instant::now();
        let exact_sampling_uniforms = if self.request.sampler.temperature > 0.0
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
            .map(LayerCache::snapshot)
            .collect::<Vec<_>>();
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
            .saturating_add(resolution.exact_sampling.windows);
        self.counters.exact_sampling.acceptance_draws = self
            .counters
            .exact_sampling
            .acceptance_draws
            .saturating_add(resolution.exact_sampling.acceptance_draws);
        self.counters.exact_sampling.residual_corrections = self
            .counters
            .exact_sampling
            .residual_corrections
            .saturating_add(resolution.exact_sampling.residual_corrections);
        self.counters.exact_sampling.bonus_samples = self
            .counters
            .exact_sampling
            .bonus_samples
            .saturating_add(resolution.exact_sampling.bonus_samples);
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
        for token in tokens_to_append {
            commit_constraint_token(&mut self.constraint, token)?;
            self.history.push(token);
            self.pending_tokens.push_back(token);
        }
        self.counters.window_us = self
            .counters
            .window_us
            .saturating_add(elapsed_us(window_started));
        Ok(())
    }
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
    sampler: crate::core::sampler::Sampler,
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

fn dflash2_target_forward_mode(sampler: crate::core::sampler::Sampler) -> DFlash2TargetForwardMode {
    if sampler.is_greedy() {
        DFlash2TargetForwardMode::GreedyVerify
    } else {
        DFlash2TargetForwardMode::SampledVerify
    }
}

fn prepare_dflash2_exact_sampling(
    target_logits: &Array,
    sampler: crate::core::sampler::Sampler,
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
    sampler: crate::core::sampler::Sampler,
    history: &[u32],
    prng_state: &mut Array,
) -> Result<SpeculativeResolution> {
    if sampler.is_greedy() {
        let target_tokens = greedy_verified_tokens
            .ok_or_else(|| anyhow!("DFlash2 greedy verification tokens are absent"))?;
        return resolve_speculative_tokens(draft_tokens, target_tokens);
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

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
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
    use crate::core::constrained::ConstraintTokenizer;
    use serial_test::serial;

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
    fn qwen38_dflash2_batched_prefill_matches_scheduler_b1_exactly() {
        use crate::core::sampler::Sampler;
        use crate::core::{Loader, Tokenizer};
        use crate::models::dflash2::DFlash2DraftModel;
        use crate::models::Qwen35Model;

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
                    4,
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
    fn qwen38_dflash2_b4_windows_match_scheduler_b1_exactly() {
        use crate::core::sampler::Sampler;
        use crate::core::{Loader, Message, Tokenizer};
        use crate::models::dflash2::DFlash2DraftModel;
        use crate::models::Qwen35Model;

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
                    3,
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
                    &|_| false,
                )
                .expect("B4 DFlash2 streams");
            let mut tensor_cache = None;

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
                for (row, stream) in batched.iter_mut().enumerate() {
                    let actual = stream
                        .next_token_deferred()
                        .expect("B4 token")
                        .map(|event| (event.token, event.finish_reason));
                    assert_eq!(expected, actual, "{case} row {row} step {step}");
                }
                if expected
                    .as_ref()
                    .is_some_and(|(_, finish_reason)| finish_reason.is_none())
                    && batched
                        .iter()
                        .all(|stream| stream.tensor_batch_key().expect("batch key").is_some())
                {
                    let mut rows = batched.iter_mut().collect::<Vec<_>>();
                    tensor_cache = DFlash2TextGenerationStream::fill_deferred_window_bn(
                        &mut rows,
                        tensor_cache.take(),
                    )
                    .expect("B4 tensor window");
                    for (row, stream) in batched.iter().enumerate() {
                        assert_array_exact(
                            &format!("{case} B4 row {row} step {step} aligned context"),
                            &reference.pending_context_hidden,
                            &stream.pending_context_hidden,
                        );
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
        let tokenizer = ConstraintTokenizer::byte_level().expect("byte tokenizer");
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
            crate::core::sampler::Sampler::greedy(),
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
        let sampler = crate::core::sampler::Sampler::greedy()
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
        assert_eq!(resolution_a.exact_sampling.windows, 1);
        assert!(resolution_a.exact_sampling.acceptance_draws > 0);
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
        let sampler = crate::core::sampler::Sampler::greedy().with_repetition_penalty(2.0);
        let mut key = mlx::random::key(0).expect("key");

        let resolution = resolve_dflash2_window(&[1, 1], None, &logits, sampler, &[1], &mut key)
            .expect("resolution");

        assert_eq!(resolution.tokens_to_append[0], 2);
        assert_eq!(resolution.accepted_draft_len, 0);
        assert_eq!(resolution.exact_sampling, ExactSamplingCounters::default());
    }

    #[test]
    fn sampler_selects_greedy_or_sampled_target_mode() {
        assert_eq!(
            dflash2_target_forward_mode(crate::core::sampler::Sampler::greedy()),
            DFlash2TargetForwardMode::GreedyVerify
        );
        assert_eq!(
            dflash2_target_forward_mode(
                crate::core::sampler::Sampler::greedy().with_temperature(0.8)
            ),
            DFlash2TargetForwardMode::SampledVerify
        );
        assert_eq!(
            dflash2_target_forward_mode(
                crate::core::sampler::Sampler::greedy().with_repetition_penalty(1.1)
            ),
            DFlash2TargetForwardMode::SampledVerify
        );
    }
}
