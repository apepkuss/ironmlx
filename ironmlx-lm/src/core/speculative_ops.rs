//! Model-side speculative verification, sampling composition, and cache operations.
use crate::core::cache::layer::{LayerCache, LayerCacheSnapshot};
use crate::core::cache::MtpCache;
use crate::core::sampler::{draw_uniforms, sample_target_tokens_with_uniforms_batch, Sampler};
#[cfg(test)]
use crate::core::sampler::{SamplingDistribution, SamplingTestExt};
use crate::core::speculative_model::MtpSpeculativeModel;
#[cfg(test)]
use crate::core::{Loader, Model};
#[cfg(test)]
use crate::nn::MtpStepOutput;
use crate::Result;
use anyhow::anyhow;
#[cfg(test)]
use mlx::Dtype;
use mlx::{random, Array, StreamOrDevice};
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
    exact_sampling: ExactSamplingCounters,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExactSamplingCounters {
    pub windows: usize,
    pub acceptance_draws: usize,
    pub residual_corrections: usize,
    pub bonus_samples: usize,
}

#[derive(Debug, Clone)]
pub enum DraftTokenDistribution {
    Deterministic,
    #[cfg(test)]
    Sampled(SamplingDistribution),
}

pub fn split_speculative_draft_prng(prng_state: &mut Array) -> Result<Array> {
    anyhow::ensure!(
        prng_state.size() == 2,
        "speculative PRNG state must contain one two-word key, got shape {:?}",
        prng_state.shape().as_slice()
    );
    let original_shape = prng_state.shape().as_slice().to_vec();
    let flat = prng_state.reshape(&[2_i32][..])?;
    let (next_decision_key, draft_key) = random::split(&flat)?;
    *prng_state = next_decision_key.reshape(original_shape.as_slice())?;
    Ok(draft_key)
}

pub fn sample_draft_logits_position(
    logits: &Array,
    _sampler: Sampler,
    _history: &[u32],
    _draft_prng: Option<&mut Array>,
) -> Result<(u32, DraftTokenDistribution)> {
    let dims = logits.shape();
    let dims = dims.as_slice();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] == 1,
        "draft logits must be [1, 1, V], got {dims:?}"
    );
    let vocab = dims[2];
    let row = logits.reshape((vocab,))?;
    let token = mlx::ops::reduction::argmax(&row, -1, false)?.item::<u32>()?;
    Ok((token, DraftTokenDistribution::Deterministic))
}

pub fn sample_draft_logits_position_with_uniform(
    logits: &Array,
    _sampler: Sampler,
    _history: &[u32],
    _uniform: f32,
) -> Result<(u32, DraftTokenDistribution)> {
    let dims = logits.shape();
    let dims = dims.as_slice();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] == 1,
        "draft logits must be [1, 1, V], got {dims:?}"
    );
    let vocab = dims[2];
    let row = logits.reshape((vocab,))?;
    let token = mlx::ops::reduction::argmax(&row, -1, false)?.item::<u32>()?;
    Ok((token, DraftTokenDistribution::Deterministic))
}

#[cfg(test)]
impl DraftTokenDistribution {
    fn acceptance_probability(&self, target: &SamplingDistribution, token: u32) -> Result<f32> {
        match self {
            Self::Deterministic => Ok(target.probability(token)),
            #[cfg(test)]
            Self::Sampled(draft) => target.acceptance_probability(draft, token),
        }
    }

    fn residual(&self, target: &SamplingDistribution, token: u32) -> Result<SamplingDistribution> {
        match self {
            Self::Deterministic => target.residual_point_mass(token),
            #[cfg(test)]
            Self::Sampled(draft) => target.residual(draft),
        }
    }
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
        exact_sampling: ExactSamplingCounters::default(),
    })
}

#[cfg(test)]
pub(crate) fn resolve_exact_speculative_logits(
    draft_tokens: &[u32],
    draft_distributions: &[DraftTokenDistribution],
    target_logits: &Array,
    sampler: Sampler,
    history: &[u32],
    prng_state: &mut Array,
) -> Result<SpeculativeResolution> {
    anyhow::ensure!(
        sampler.temperature > 0.0,
        "exact speculative sampling requires temperature > 0"
    );
    anyhow::ensure!(
        draft_tokens.len() == draft_distributions.len(),
        "exact speculative draft token count {} != distribution count {}",
        draft_tokens.len(),
        draft_distributions.len()
    );
    let shape = target_logits.shape();
    let dims = shape.as_slice();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1,
        "exact speculative target logits must be [1, S, V], got {dims:?}"
    );
    anyhow::ensure!(
        dims[1] as usize == draft_tokens.len() + 1,
        "exact speculative target positions {} != draft count {} + 1",
        dims[1],
        draft_tokens.len()
    );

    let target_distributions =
        speculative_target_distributions(target_logits, sampler, history, draft_tokens)?;
    let uniforms = draw_uniforms(prng_state, draft_tokens.len() + 1)?;
    let correction_uniform = uniforms[draft_tokens.len()];
    let mut tokens_to_append = Vec::with_capacity(draft_tokens.len() + 1);
    let mut exact_sampling = ExactSamplingCounters {
        windows: 1,
        ..ExactSamplingCounters::default()
    };
    for (position, (&draft_token, draft_distribution)) in
        draft_tokens.iter().zip(draft_distributions).enumerate()
    {
        let target = &target_distributions[position];
        let accept_probability = draft_distribution.acceptance_probability(target, draft_token)?;
        exact_sampling.acceptance_draws = exact_sampling.acceptance_draws.saturating_add(1);
        if uniforms[position] < accept_probability {
            tokens_to_append.push(draft_token);
            continue;
        }

        let corrected = draft_distribution
            .residual(target, draft_token)?
            .sample_with_uniform(correction_uniform)?;
        exact_sampling.residual_corrections = exact_sampling.residual_corrections.saturating_add(1);
        tokens_to_append.push(corrected);
        return Ok(SpeculativeResolution {
            accepted_draft_len: position,
            tokens_to_append,
            accepted_verify_input_len: position + 1,
            needs_rollback: true,
            exact_sampling,
        });
    }

    tokens_to_append
        .push(target_distributions[draft_tokens.len()].sample_with_uniform(correction_uniform)?);
    exact_sampling.bonus_samples = exact_sampling.bonus_samples.saturating_add(1);
    Ok(SpeculativeResolution {
        accepted_draft_len: draft_tokens.len(),
        tokens_to_append,
        accepted_verify_input_len: draft_tokens.len() + 1,
        needs_rollback: false,
        exact_sampling,
    })
}

#[cfg(test)]
pub(crate) fn resolve_exact_deterministic_target_distributions(
    draft_tokens: &[u32],
    target_distributions: &[SamplingDistribution],
    prng_state: &mut Array,
) -> Result<SpeculativeResolution> {
    anyhow::ensure!(
        target_distributions.len() == draft_tokens.len() + 1,
        "exact deterministic target distribution count {} != draft count {} + 1",
        target_distributions.len(),
        draft_tokens.len()
    );
    let uniforms = draw_uniforms(prng_state, target_distributions.len())?;
    let mut tokens_to_append = Vec::with_capacity(target_distributions.len());
    let mut exact_sampling = ExactSamplingCounters {
        windows: 1,
        ..ExactSamplingCounters::default()
    };

    for (position, &draft_token) in draft_tokens.iter().enumerate() {
        let target_token =
            target_distributions[position].sample_with_uniform(uniforms[position])?;
        exact_sampling.acceptance_draws = exact_sampling.acceptance_draws.saturating_add(1);
        if target_token == draft_token {
            tokens_to_append.push(draft_token);
            continue;
        }

        tokens_to_append.push(target_token);
        exact_sampling.residual_corrections = exact_sampling.residual_corrections.saturating_add(1);
        return Ok(SpeculativeResolution {
            accepted_draft_len: position,
            tokens_to_append,
            accepted_verify_input_len: position + 1,
            needs_rollback: true,
            exact_sampling,
        });
    }

    tokens_to_append.push(
        target_distributions[draft_tokens.len()]
            .sample_with_uniform(uniforms[draft_tokens.len()])?,
    );
    exact_sampling.bonus_samples = exact_sampling.bonus_samples.saturating_add(1);
    Ok(SpeculativeResolution {
        accepted_draft_len: draft_tokens.len(),
        tokens_to_append,
        accepted_verify_input_len: draft_tokens.len() + 1,
        needs_rollback: false,
        exact_sampling,
    })
}

pub fn resolve_exact_deterministic_target_tokens(
    draft_tokens: &[u32],
    target_tokens: &[u32],
) -> Result<SpeculativeResolution> {
    anyhow::ensure!(
        target_tokens.len() == draft_tokens.len() + 1,
        "exact deterministic target token count {} != draft count {} + 1",
        target_tokens.len(),
        draft_tokens.len()
    );
    let accepted_draft_len = draft_tokens
        .iter()
        .zip(target_tokens)
        .take_while(|(draft, target)| draft == target)
        .count();
    let mut tokens_to_append = Vec::with_capacity(accepted_draft_len + 1);
    tokens_to_append.extend_from_slice(&draft_tokens[..accepted_draft_len]);
    tokens_to_append.push(target_tokens[accepted_draft_len]);
    let mismatch = accepted_draft_len < draft_tokens.len();

    Ok(SpeculativeResolution {
        accepted_draft_len,
        tokens_to_append,
        accepted_verify_input_len: accepted_draft_len + 1,
        needs_rollback: mismatch,
        exact_sampling: ExactSamplingCounters {
            windows: 1,
            acceptance_draws: if mismatch {
                accepted_draft_len + 1
            } else {
                draft_tokens.len()
            },
            residual_corrections: usize::from(mismatch),
            bonus_samples: usize::from(!mismatch),
        },
    })
}

/// Exact coupling for a deterministic draft distribution `q = delta(draft)`.
///
/// Sampling once from each target distribution `p` is equivalent to the
/// standard accept/reject algorithm: a sampled draft token is accepted when
/// the target sample matches it; otherwise that same target sample has the
/// conditional residual distribution over all non-draft tokens.
pub fn resolve_exact_deterministic_target_logits(
    draft_tokens: &[u32],
    target_logits: &Array,
    sampler: Sampler,
    history: &[u32],
    prng_state: &mut Array,
) -> Result<SpeculativeResolution> {
    anyhow::ensure!(
        sampler.temperature > 0.0,
        "exact deterministic target sampling requires temperature > 0"
    );
    let shape = target_logits.shape();
    let dims = shape.as_slice();
    let positions = draft_tokens.len() + 1;
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] as usize == positions,
        "exact deterministic target logits must be [1, {positions}, V], got {dims:?}"
    );
    let rows = target_logits.reshape(&[i32::try_from(positions)?, dims[2]][..])?;
    let histories = (0..positions)
        .map(|position| {
            let mut position_history = Vec::with_capacity(history.len() + position);
            position_history.extend_from_slice(history);
            position_history.extend_from_slice(&draft_tokens[..position]);
            position_history
        })
        .collect::<Vec<_>>();
    let history_refs = histories.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let sampler_refs = vec![&sampler; positions];
    let uniforms = draw_uniforms(prng_state, positions)?;
    let target_tokens =
        sample_target_tokens_with_uniforms_batch(&sampler_refs, &rows, &history_refs, &uniforms)?;
    resolve_exact_deterministic_target_tokens(draft_tokens, &target_tokens)
}

#[cfg(test)]
fn speculative_target_distributions(
    logits: &Array,
    sampler: Sampler,
    history: &[u32],
    draft_tokens: &[u32],
) -> Result<Vec<SamplingDistribution>> {
    let dims = logits.shape();
    let dims = dims.as_slice();
    let positions = draft_tokens.len() + 1;
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] as usize == positions,
        "speculative target distributions require [1, {positions}, V], got {dims:?}"
    );
    let rows = logits.reshape(&[i32::try_from(positions)?, dims[2]][..])?;
    let histories = (0..positions)
        .map(|position| {
            let mut position_history = Vec::with_capacity(history.len() + position);
            position_history.extend_from_slice(history);
            position_history.extend_from_slice(&draft_tokens[..position]);
            position_history
        })
        .collect::<Vec<_>>();
    let history_refs = histories.iter().map(Vec::as_slice).collect::<Vec<_>>();
    sampler.distributions(&rows, &history_refs)
}

pub fn zero_hidden_like_position(hidden: &Array) -> Result<Array> {
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
pub fn commit_mtp_cache_hidden_prefix<M>(
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
pub fn commit_mtp_cache_hidden_tail<M>(
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

pub fn verify_input(current_token: u32, draft_tokens: &[u32]) -> Vec<u32> {
    let mut input = Vec::with_capacity(draft_tokens.len() + 1);
    input.push(current_token);
    input.extend_from_slice(draft_tokens);
    input
}

pub fn sample_logits_positions(
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

pub fn slice_hidden_position(hidden: &Array, pos: i32) -> Result<Array> {
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

pub fn slice_position_ids_prefix(position_ids: &Array, len: usize) -> Result<Array> {
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

pub fn restore_layer_cache(
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

pub fn layer_cache_supports_accepted_prefix_trim(cache: &[LayerCache]) -> bool {
    cache
        .iter()
        .all(|layer| matches!(layer, LayerCache::Full(_)))
}

pub fn trim_full_layer_cache_rows_to_accepted_prefix(
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

pub struct MainCacheRollbackInput<'a> {
    pub accepted_by_row: &'a [(usize, usize)],
    pub verify_input: &'a [u32],
    pub accepted_position_ids: &'a Array,
    pub verified_hidden: &'a Array,
}

pub fn rollback_main_cache_to_accepted_prefix<M: MtpSpeculativeModel>(
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
    let _verify_qmm = crate::nn::verify_qmm_scope();
    model.forward_text_hidden(
        &accepted_arr,
        input.accepted_position_ids,
        None,
        None,
        Some(cache),
        target.into(),
    )
}

impl SpeculativeResolution {
    /// Construct an accepted-prefix result for a deterministic verification route.
    pub fn deterministic(
        accepted_draft_len: usize,
        tokens_to_append: Vec<u32>,
        accepted_verify_input_len: usize,
        needs_rollback: bool,
    ) -> Self {
        Self {
            accepted_draft_len,
            tokens_to_append,
            accepted_verify_input_len,
            needs_rollback,
            exact_sampling: ExactSamplingCounters::default(),
        }
    }

    /// Counters for this resolution; aggregation belongs to the caller.
    pub fn exact_sampling(&self) -> ExactSamplingCounters {
        self.exact_sampling
    }
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

        fn model_meta(&self) -> crate::core::model::ModelMeta {
            crate::core::model::ModelMeta {
                num_hidden_layers: 28,
                num_attention_heads: 32,
                num_key_value_heads: 8,
                hidden_size: 4096,
                head_dim: None,
                weight_bytes: 3 * 1024 * 1024 * 1024,
                max_position_embeddings: 32768,
                spatial_merge_size: 2,
            }
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
    fn exact_sampling_accepts_deterministic_draft_and_samples_bonus() {
        let logits: Array = (
            &[
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
            ][..],
            &[1_i32, 3_i32, 4_i32][..],
        )
            .try_into()
            .unwrap();
        let mut prng = mlx::random::key(17).unwrap();
        let resolution = resolve_exact_speculative_logits(
            &[1, 1],
            &[
                DraftTokenDistribution::Deterministic,
                DraftTokenDistribution::Deterministic,
            ],
            &logits,
            Sampler::greedy().with_temperature(1.0),
            &[9],
            &mut prng,
        )
        .unwrap();

        assert_eq!(resolution.accepted_draft_len, 2);
        assert_eq!(resolution.tokens_to_append, vec![1, 1, 2]);
        assert_eq!(resolution.accepted_verify_input_len, 3);
        assert!(!resolution.needs_rollback);
        assert_eq!(resolution.exact_sampling.windows, 1);
        assert_eq!(resolution.exact_sampling.acceptance_draws, 2);
        assert_eq!(resolution.exact_sampling.residual_corrections, 0);
        assert_eq!(resolution.exact_sampling.bonus_samples, 1);
    }

    #[test]
    fn exact_sampling_rejects_deterministic_draft_and_uses_residual() {
        let logits: Array = (
            &[
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ][..],
            &[1_i32, 2_i32, 4_i32][..],
        )
            .try_into()
            .unwrap();
        let mut prng = mlx::random::key(23).unwrap();
        let resolution = resolve_exact_speculative_logits(
            &[1],
            &[DraftTokenDistribution::Deterministic],
            &logits,
            Sampler::greedy().with_temperature(1.0),
            &[9],
            &mut prng,
        )
        .unwrap();

        assert_eq!(resolution.accepted_draft_len, 0);
        assert_eq!(resolution.tokens_to_append, vec![2]);
        assert_eq!(resolution.accepted_verify_input_len, 1);
        assert!(resolution.needs_rollback);
        assert_eq!(resolution.exact_sampling.windows, 1);
        assert_eq!(resolution.exact_sampling.acceptance_draws, 1);
        assert_eq!(resolution.exact_sampling.residual_corrections, 1);
        assert_eq!(resolution.exact_sampling.bonus_samples, 0);
    }

    #[test]
    fn exact_sampling_target_coupling_accepts_and_samples_bonus() {
        let target_distributions = [
            SamplingDistribution::new(vec![0.0, 1.0, 0.0, 0.0]).unwrap(),
            SamplingDistribution::new(vec![0.0, 0.0, 1.0, 0.0]).unwrap(),
            SamplingDistribution::new(vec![0.0, 0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut prng = mlx::random::key(29).unwrap();
        let resolution = resolve_exact_deterministic_target_distributions(
            &[1, 2],
            &target_distributions,
            &mut prng,
        )
        .unwrap();

        assert_eq!(resolution.accepted_draft_len, 2);
        assert_eq!(resolution.tokens_to_append, vec![1, 2, 3]);
        assert!(!resolution.needs_rollback);
        assert_eq!(resolution.exact_sampling.acceptance_draws, 2);
        assert_eq!(resolution.exact_sampling.residual_corrections, 0);
        assert_eq!(resolution.exact_sampling.bonus_samples, 1);
    }

    #[test]
    fn exact_sampling_target_coupling_reuses_rejected_target_as_correction() {
        let target_distributions = [
            SamplingDistribution::new(vec![0.0, 0.0, 1.0, 0.0]).unwrap(),
            SamplingDistribution::new(vec![0.0, 0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut prng = mlx::random::key(31).unwrap();
        let resolution = resolve_exact_deterministic_target_distributions(
            &[1],
            &target_distributions,
            &mut prng,
        )
        .unwrap();

        assert_eq!(resolution.accepted_draft_len, 0);
        assert_eq!(resolution.tokens_to_append, vec![2]);
        assert!(resolution.needs_rollback);
        assert_eq!(resolution.exact_sampling.acceptance_draws, 1);
        assert_eq!(resolution.exact_sampling.residual_corrections, 1);
        assert_eq!(resolution.exact_sampling.bonus_samples, 0);
    }

    #[test]
    fn exact_sampling_target_tokens_preserve_target_coupling_counters() {
        let accepted =
            resolve_exact_deterministic_target_tokens(&[1, 2], &[1, 2, 3]).expect("accepted");
        assert_eq!(accepted.accepted_draft_len, 2);
        assert_eq!(accepted.tokens_to_append, vec![1, 2, 3]);
        assert!(!accepted.needs_rollback);
        assert_eq!(accepted.exact_sampling.acceptance_draws, 2);
        assert_eq!(accepted.exact_sampling.residual_corrections, 0);
        assert_eq!(accepted.exact_sampling.bonus_samples, 1);

        let rejected =
            resolve_exact_deterministic_target_tokens(&[1, 2], &[1, 9, 3]).expect("rejected");
        assert_eq!(rejected.accepted_draft_len, 1);
        assert_eq!(rejected.tokens_to_append, vec![1, 9]);
        assert!(rejected.needs_rollback);
        assert_eq!(rejected.exact_sampling.acceptance_draws, 2);
        assert_eq!(rejected.exact_sampling.residual_corrections, 1);
        assert_eq!(rejected.exact_sampling.bonus_samples, 0);
    }

    #[test]
    fn exact_deterministic_coupling_preserves_target_distribution_on_uniform_grid() {
        let target =
            SamplingDistribution::new(vec![0.1, 0.2, 0.3, 0.4]).expect("target distribution");
        let mut counts = [0_usize; 4];
        let mut accepted = 0_usize;
        let mut corrected = 0_usize;

        for index in 0..1_000 {
            let uniform = (index as f32 + 0.5) / 1_000.0;
            let target_token = target
                .sample_with_uniform(uniform)
                .expect("sample target token");
            let resolution = resolve_exact_deterministic_target_tokens(&[3], &[target_token, 0])
                .expect("resolve deterministic draft");
            counts[resolution.tokens_to_append[0] as usize] += 1;
            accepted += resolution.accepted_draft_len;
            corrected += resolution.exact_sampling.residual_corrections;
        }

        assert_eq!(counts, [100, 200, 300, 400]);
        assert_eq!(accepted, 400);
        assert_eq!(corrected, 600);
    }

    #[test]
    fn exact_target_logits_preserve_position_histories_with_penalties() {
        let logits: Array = (
            &[
                0.0_f32, 20.0, 0.0, 0.0, //
                0.0, 0.0, 20.0, 0.0, //
                0.0, 0.0, 0.0, 20.0,
            ][..],
            &[1_i32, 3, 4][..],
        )
            .try_into()
            .unwrap();
        let sampler = Sampler::greedy()
            .with_temperature(0.8)
            .with_top_p(0.95)
            .with_repetition_penalty(1.1)
            .with_frequency_penalty(0.2)
            .with_presence_penalty(0.1);
        let mut prng = mlx::random::key(41).unwrap();
        let resolution = resolve_exact_deterministic_target_logits(
            &[1, 2],
            &logits,
            sampler,
            &[0, 0, 1],
            &mut prng,
        )
        .unwrap();

        assert_eq!(resolution.accepted_draft_len, 2);
        assert_eq!(resolution.tokens_to_append, vec![1, 2, 3]);
        assert!(!resolution.needs_rollback);
    }

    #[test]
    fn speculative_prng_split_is_reproducible_independent_and_shape_preserving() {
        for shape in [&[2_i32][..], &[1_i32, 2_i32][..]] {
            let mut decision_a = mlx::random::key(47).unwrap().reshape(shape).unwrap();
            let mut decision_b = mlx::random::key(47).unwrap().reshape(shape).unwrap();

            let draft_a = split_speculative_draft_prng(&mut decision_a).unwrap();
            let draft_b = split_speculative_draft_prng(&mut decision_b).unwrap();

            assert_eq!(decision_a.shape().as_slice(), shape);
            assert_eq!(decision_b.shape().as_slice(), shape);
            assert_eq!(
                decision_a.to_vec::<u32>().unwrap(),
                decision_b.to_vec::<u32>().unwrap()
            );
            assert_eq!(
                draft_a.to_vec::<u32>().unwrap(),
                draft_b.to_vec::<u32>().unwrap()
            );
            assert_ne!(
                decision_a.to_vec::<u32>().unwrap(),
                draft_a.to_vec::<u32>().unwrap()
            );
        }
    }

    #[test]
    fn exact_sampling_same_seed_replays_resolution_and_prng_state() {
        let logits: Array = (
            &[
                0.0_f32, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.5, 1.5, 2.5, 3.5,
            ][..],
            &[1_i32, 3_i32, 4_i32][..],
        )
            .try_into()
            .unwrap();
        let draft_distributions = vec![
            DraftTokenDistribution::Sampled(
                SamplingDistribution::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
            ),
            DraftTokenDistribution::Sampled(
                SamplingDistribution::new(vec![0.4, 0.3, 0.2, 0.1]).unwrap(),
            ),
        ];
        let sampler = Sampler::greedy()
            .with_temperature(0.8)
            .with_top_p(0.95)
            .with_seed(71);
        let mut key_a = mlx::random::key(71)
            .unwrap()
            .reshape((1_i32, 2_i32))
            .unwrap();
        let mut key_b = mlx::random::key(71)
            .unwrap()
            .reshape((1_i32, 2_i32))
            .unwrap();

        let resolution_a = resolve_exact_speculative_logits(
            &[3, 0],
            &draft_distributions,
            &logits,
            sampler,
            &[9, 8],
            &mut key_a,
        )
        .unwrap();
        let resolution_b = resolve_exact_speculative_logits(
            &[3, 0],
            &draft_distributions,
            &logits,
            sampler,
            &[9, 8],
            &mut key_b,
        )
        .unwrap();

        assert_eq!(resolution_a, resolution_b);
        assert_eq!(key_a.shape().as_slice(), &[1, 2]);
        assert_eq!(key_b.shape().as_slice(), &[1, 2]);
        assert_eq!(
            key_a.to_vec::<u32>().unwrap(),
            key_b.to_vec::<u32>().unwrap()
        );
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
            crate::core::model_input::build_position_ids(4, 2).expect("position ids");
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
}
