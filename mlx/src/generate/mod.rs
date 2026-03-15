mod kv_cache;
mod sampler;
mod tokenizer;

pub use kv_cache::KVCache;
pub use sampler::{sample, SamplerConfig};
pub use tokenizer::Tokenizer;

use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::model::LlamaModel;
use crate::ops;
use crate::stream::Stream;

/// Generate tokens from a prompt.
pub fn generate(
    model: &LlamaModel,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampler_config: &SamplerConfig,
    eos_token_id: i32,
) -> Result<Vec<i32>> {
    let stream = Stream::new(&Device::gpu());
    let num_layers = model.layers.len();

    // Initialize KV cache (empty for each layer)
    let mut cache: Vec<(Option<Array>, Option<Array>)> =
        (0..num_layers).map(|_| (None, None)).collect();

    let mut generated = Vec::new();

    // Prefill: process all prompt tokens at once
    let prompt_arr = Array::from_slice_i32(prompt_tokens);
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
        let input = Array::from_slice_i32(&[next_token]);
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
