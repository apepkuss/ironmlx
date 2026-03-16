pub mod batch;
mod chat_template;
mod kv_cache;
mod sampler;
mod tokenizer;

pub use batch::{BatchGenerator, BatchResponse, FinishReason, SeqUid};
pub use chat_template::{ChatMessage, ChatTemplate};
pub use kv_cache::KVCache;
pub use sampler::{SamplerConfig, sample};
pub use tokenizer::Tokenizer;

use crate::array::Array;
use crate::cache::CacheManager;
use crate::device::Device;
use crate::error::Result;
use crate::model::Model;
use crate::ops;
use crate::stream::Stream;

/// Reason a generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Reached the EOS token.
    Eos,
    /// Reached the maximum number of tokens.
    MaxTokens,
}

/// Generate tokens from a prompt, calling `on_token` for each generated token.
/// Returns the stop reason.
pub fn stream_generate(
    model: &Model,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampler_config: &SamplerConfig,
    eos_token_id: i32,
    on_token: impl FnMut(i32) -> bool,
) -> Result<StopReason> {
    stream_generate_with_cache(
        model,
        prompt_tokens,
        max_tokens,
        sampler_config,
        eos_token_id,
        on_token,
        None,
    )
}

/// Generate tokens with optional prefix cache support.
pub fn stream_generate_with_cache(
    model: &Model,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampler_config: &SamplerConfig,
    eos_token_id: i32,
    mut on_token: impl FnMut(i32) -> bool,
    mut cache_manager: Option<&mut CacheManager>,
) -> Result<StopReason> {
    let stream = Stream::new(&Device::gpu());
    let num_layers = model.num_layers();

    // Try prefix cache lookup
    let (mut cache, matched_tokens) = match cache_manager {
        Some(ref mut cm) => match cm.lookup_and_load(prompt_tokens) {
            Ok((c, m)) if m > 0 => (c, m),
            _ => ((0..num_layers).map(|_| (None, None)).collect(), 0),
        },
        None => ((0..num_layers).map(|_| (None, None)).collect(), 0),
    };

    // Prefill remaining tokens
    let tokens_to_prefill = &prompt_tokens[matched_tokens..];
    let prompt_arr = Array::from_slice_i32(tokens_to_prefill);
    let prompt_arr = ops::astype(&prompt_arr, crate::dtype::Dtype::Int32, &stream)?;
    let prompt_2d = ops::reshape(&prompt_arr, &[1, tokens_to_prefill.len() as i32], &stream)?;
    let logits = model.forward(&prompt_2d, &mut cache, "causal", None)?;

    // Store in prefix cache after prefill
    if let Some(ref mut cm) = cache_manager {
        let _ = cm.store_after_prefill(prompt_tokens, &cache);
    }

    let last_idx = tokens_to_prefill.len() as i32 - 1;
    let last_logits = ops::slice(
        &logits,
        &[0, last_idx, 0],
        &[1, last_idx + 1, logits.shape()[2]],
        &[1, 1, 1],
        &stream,
    )?;
    let last_logits = ops::reshape(&last_logits, &[1, logits.shape()[2]], &stream)?;

    let mut next_token_arr = sample(&last_logits, sampler_config, &stream)?;
    next_token_arr.eval()?;
    let mut next_token = next_token_arr.item_i32()?;

    if next_token == eos_token_id {
        return Ok(StopReason::Eos);
    }
    if !on_token(next_token) {
        return Ok(StopReason::Eos);
    }

    // Decode loop
    for i in 1..max_tokens {
        let input = Array::from_int(next_token);
        let input = ops::astype(&input, crate::dtype::Dtype::Int32, &stream)?;
        let input_2d = ops::reshape(&input, &[1, 1], &stream)?;
        let logits = model.forward(&input_2d, &mut cache, "causal", None)?;
        let logits_2d = ops::reshape(&logits, &[1, logits.shape()[2]], &stream)?;

        next_token_arr = sample(&logits_2d, sampler_config, &stream)?;
        next_token_arr.eval()?;
        next_token = next_token_arr.item_i32()?;

        if next_token == eos_token_id {
            return Ok(StopReason::Eos);
        }
        if !on_token(next_token) {
            return Ok(StopReason::Eos);
        }
        if i + 1 >= max_tokens {
            return Ok(StopReason::MaxTokens);
        }
    }

    Ok(StopReason::MaxTokens)
}

/// Generate tokens from a prompt.
pub fn generate(
    model: &Model,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampler_config: &SamplerConfig,
    eos_token_id: i32,
) -> Result<Vec<i32>> {
    let stream = Stream::new(&Device::gpu());
    let num_layers = model.num_layers();

    // Initialize KV cache (empty for each layer)
    let mut cache: Vec<(Option<Array>, Option<Array>)> =
        (0..num_layers).map(|_| (None, None)).collect();

    let mut generated = Vec::new();

    // Prefill: process all prompt tokens at once — astype ensures GPU migration
    let prompt_arr = Array::from_slice_i32(prompt_tokens);
    let prompt_arr = ops::astype(&prompt_arr, crate::dtype::Dtype::Int32, &stream)?;
    let prompt_2d = ops::reshape(&prompt_arr, &[1, prompt_tokens.len() as i32], &stream)?;
    let logits = model.forward(&prompt_2d, &mut cache, "causal", None)?;

    // Take logits of last token
    let last_idx = prompt_tokens.len() as i32 - 1;
    let last_logits = ops::slice(
        &logits,
        &[0, last_idx, 0],
        &[1, last_idx + 1, logits.shape()[2]],
        &[1, 1, 1],
        &stream,
    )?;
    let last_logits = ops::reshape(&last_logits, &[1, logits.shape()[2]], &stream)?;

    // Sample first token
    let mut next_token_arr = sample(&last_logits, sampler_config, &stream)?;
    next_token_arr.eval()?;
    let mut next_token = next_token_arr.item_i32()?;

    if next_token == eos_token_id {
        return Ok(generated);
    }
    generated.push(next_token);

    // Decode loop
    for _ in 1..max_tokens {
        let input = Array::from_int(next_token);
        let input = ops::astype(&input, crate::dtype::Dtype::Int32, &stream)?;
        let input_2d = ops::reshape(&input, &[1, 1], &stream)?;
        let logits = model.forward(&input_2d, &mut cache, "causal", None)?;

        // logits shape: [1, 1, vocab_size] -> [1, vocab_size]
        let logits_2d = ops::reshape(&logits, &[1, logits.shape()[2]], &stream)?;

        next_token_arr = sample(&logits_2d, sampler_config, &stream)?;
        next_token_arr.eval()?;
        next_token = next_token_arr.item_i32()?;

        if next_token == eos_token_id {
            break;
        }
        generated.push(next_token);
    }

    Ok(generated)
}
