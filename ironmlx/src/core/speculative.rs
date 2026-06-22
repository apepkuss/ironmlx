//! Speculative decoding helpers shared by MTP generation paths.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

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

pub fn default_mtp_draft_tokens_for_config(raw_config: &serde_json::Value) -> usize {
    let model_type = raw_config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let text = raw_config
        .get("text_config")
        .and_then(serde_json::Value::as_object);
    let hidden_size = text
        .and_then(|v| v.get("hidden_size"))
        .and_then(serde_json::Value::as_i64);
    let layers = text
        .and_then(|v| v.get("num_hidden_layers"))
        .and_then(serde_json::Value::as_i64);
    let experts = text
        .and_then(|v| v.get("num_experts"))
        .and_then(serde_json::Value::as_i64);
    let experts_per_tok = text
        .and_then(|v| v.get("num_experts_per_tok"))
        .and_then(serde_json::Value::as_i64);

    match (model_type, hidden_size, layers, experts, experts_per_tok) {
        ("qwen3_5", Some(5120), Some(64), None, None) => 2,
        ("qwen3_5_moe", Some(2048), Some(40), Some(256), Some(8)) => 2,
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
        if !sampler.is_pipelinable() {
            return Err(anyhow!(
                "MtpSpeculativeConfig::new: MTP speculative decoding currently requires greedy sampling"
            ));
        }
        Ok(Self { max_draft_tokens })
    }
}

/// Runtime counters collected by [`MtpTextGenerationStream`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MtpSpeculativeStats {
    /// Speculative windows verified by the main model.
    pub windows: usize,
    /// Draft tokens proposed by the MTP head.
    pub drafted_tokens: usize,
    /// Draft tokens accepted before mismatch.
    pub accepted_draft_tokens: usize,
    /// Windows that required main-cache rollback and accepted-prefix replay.
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
    /// Microseconds spent in main-model verify/replay hidden forward passes.
    pub verify_forward_us: u64,
    /// Microseconds spent projecting hidden states to logits.
    pub projection_us: u64,
    /// Microseconds spent sampling logits.
    pub sampling_us: u64,
    /// Microseconds spent restoring/replaying the main KV cache after mismatch.
    pub main_rollback_us: u64,
    /// Microseconds spent committing accepted tokens into the MTP KV cache.
    pub mtp_cache_commit_us: u64,
    /// Microseconds spent restoring the MTP KV cache after temporary draft.
    pub mtp_cache_restore_us: u64,
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

    fn project_hidden_on(&self, hidden: &Array, target: impl Into<StreamOrDevice>)
        -> Result<Array>;

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

    fn project_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35Model::project_hidden_on(self, hidden, target)
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

    fn project_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35MoeModel::project_hidden_on(self, hidden, target)
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

    fn project_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen36MoeModel::project_hidden_on(self, hidden, target)
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
            add_elapsed_us(&mut stats.mtp_cache_commit_us, commit_start);
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
        let first_logits = model.project_hidden_on(&last_prompt_hidden, ())?;
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
            stats,
        })
    }

    /// Return cumulative speculative-window counters for this stream.
    pub fn stats(&self) -> MtpSpeculativeStats {
        self.stats
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
        let verified_hidden = self.model.forward_text_hidden(
            &verify_arr,
            &verify_pos_ids,
            None,
            None,
            Some(&mut self.cache),
            ().into(),
        )?;
        add_elapsed_us(&mut self.stats.verify_forward_us, verify_forward_start);
        let projection_start = Instant::now();
        let verified_logits = self.model.project_hidden_on(&verified_hidden, ())?;
        add_elapsed_us(&mut self.stats.projection_us, projection_start);
        let sampling_start = Instant::now();
        let verified_tokens = sample_logits_positions(
            &verified_logits,
            self.request.sampler,
            &self.history,
            &mut self.prng_state,
        )?;
        add_elapsed_us(&mut self.stats.sampling_us, sampling_start);
        let resolution = resolve_speculative_tokens(&draft_tokens, &verified_tokens)?;
        self.stats.windows += 1;
        self.stats.drafted_tokens += draft_tokens.len();
        self.stats.accepted_draft_tokens += resolution.accepted_draft_len;
        if resolution.needs_rollback {
            self.stats.rollback_count += 1;
        }
        adjust_mtp_draft_budget(
            self.cfg.max_draft_tokens,
            &mut self.adaptive_draft_tokens,
            draft_tokens.len(),
            resolution.accepted_draft_len,
            &mut self.stats,
        );

        let (accepted_input, accepted_hidden, accepted_position_ids, accepted_last_hidden) =
            if resolution.needs_rollback {
                let rollback_start = Instant::now();
                restore_layer_cache(&mut self.cache, &base_snapshot)?;
                add_elapsed_us(&mut self.stats.main_rollback_us, rollback_start);
                let replay_len = resolution.accepted_verify_input_len;
                let replay_input = &verify_input[..replay_len];
                let replay_arr: Array =
                    (replay_input, &[1_i32, replay_len as i32][..]).try_into()?;
                let replay_pos_ids = self.position_ids(verify_start_pos, replay_len as i32)?;
                let replay_forward_start = Instant::now();
                let replay_hidden = self.model.forward_text_hidden(
                    &replay_arr,
                    &replay_pos_ids,
                    None,
                    None,
                    Some(&mut self.cache),
                    ().into(),
                )?;
                add_elapsed_us(&mut self.stats.verify_forward_us, replay_forward_start);
                let last_hidden = slice_hidden_position(&replay_hidden, replay_len as i32 - 1)?;
                (
                    replay_input.to_vec(),
                    replay_hidden,
                    replay_pos_ids,
                    last_hidden,
                )
            } else {
                (
                    verify_input[..resolution.accepted_verify_input_len].to_vec(),
                    verified_hidden.clone(),
                    verify_pos_ids.clone(),
                    slice_hidden_position(
                        &verified_hidden,
                        resolution.accepted_verify_input_len as i32 - 1,
                    )?,
                )
            };

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
            add_elapsed_us(&mut self.stats.mtp_cache_commit_us, commit_start);
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
            add_elapsed_us(&mut self.stats.mtp_cache_commit_us, commit_start);
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn mtp_policy_defaults_qwen36_dense_27b_to_d2() {
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
}
