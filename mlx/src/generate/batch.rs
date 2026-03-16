use std::collections::HashMap;

use crate::array::Array;
use crate::cache::CacheManager;
use crate::device::Device;
use crate::error::Result;
use crate::model::Model;
use crate::ops;
use crate::stream::Stream;

use super::sampler::{SamplerConfig, sample};

/// Unique identifier for a sequence in the batch.
pub type SeqUid = u32;

/// Reason a sequence finished generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    MaxTokens,
}

/// Response from one step of batch generation.
#[derive(Debug)]
pub struct BatchResponse {
    pub uid: SeqUid,
    pub token_id: i32,
    pub finish_reason: Option<FinishReason>,
}

/// Per-sequence state tracked by BatchGenerator.
struct SeqState {
    sampler: SamplerConfig,
    eos_token_id: i32,
    max_tokens: usize,
    generated_count: usize,
    cache: Vec<(Option<Array>, Option<Array>)>,
    last_token: i32,
    #[allow(dead_code)]
    prompt_tokens: Vec<i32>,
}

/// BatchGenerator manages multiple sequences sharing a single model.
///
/// Current approach (7B):
/// - Prefill: each new sequence is prefilled individually in `insert()`
/// - Decode: each active sequence is decoded one at a time in `step()`
///
/// This provides lifecycle management and the API abstraction needed for
/// the engine, while true batched forward passes are a future optimization.
pub struct BatchGenerator<'a> {
    model: &'a Model,
    num_layers: usize,
    sequences: HashMap<SeqUid, SeqState>,
    next_uid: SeqUid,
    cache_manager: Option<CacheManager>,
}

impl<'a> BatchGenerator<'a> {
    /// Create a new BatchGenerator wrapping a model reference.
    pub fn new(model: &'a Model) -> Self {
        let num_layers = model.num_layers();
        Self {
            model,
            num_layers,
            sequences: HashMap::new(),
            next_uid: 0,
            cache_manager: None,
        }
    }

    /// Create a new BatchGenerator with prefix caching enabled.
    pub fn with_cache_manager(model: &'a Model, cache_manager: CacheManager) -> Self {
        let num_layers = model.num_layers();
        Self {
            model,
            num_layers,
            sequences: HashMap::new(),
            next_uid: 0,
            cache_manager: Some(cache_manager),
        }
    }

    /// Insert a new sequence. Performs prefill immediately.
    /// Returns `(uid, first_token_response)`.
    pub fn insert(
        &mut self,
        prompt_tokens: &[i32],
        sampler: SamplerConfig,
        eos_token_id: i32,
        max_tokens: usize,
    ) -> Result<(SeqUid, BatchResponse)> {
        let uid = self.next_uid;
        self.next_uid += 1;

        let stream = Stream::new(&Device::gpu());

        // Try prefix cache lookup
        let (mut cache, matched_tokens) = if let Some(ref mut cm) = self.cache_manager {
            match cm.lookup_and_load(prompt_tokens) {
                Ok((c, m)) if m > 0 => (c, m),
                _ => ((0..self.num_layers).map(|_| (None, None)).collect(), 0),
            }
        } else {
            ((0..self.num_layers).map(|_| (None, None)).collect(), 0)
        };

        // Prefill: run model.forward() with remaining tokens (or all if no cache hit)
        let tokens_to_prefill = &prompt_tokens[matched_tokens..];
        let prompt_arr = Array::from_slice_i32(tokens_to_prefill);
        let prompt_2d = ops::reshape(&prompt_arr, &[1, tokens_to_prefill.len() as i32], &stream)?;
        let logits = self.model.forward(&prompt_2d, &mut cache, "causal", None)?;

        // Extract last token logits (from tokens_to_prefill output)
        let last_idx = tokens_to_prefill.len() as i32 - 1;
        let vocab_size = logits.shape()[2];
        let last_logits = ops::slice(
            &logits,
            &[0, last_idx, 0],
            &[1, last_idx + 1, vocab_size],
            &[1, 1, 1],
            &stream,
        )?;
        let last_logits = ops::reshape(&last_logits, &[1, vocab_size], &stream)?;

        // Sample first token
        let token_arr = sample(&last_logits, &sampler, &stream)?;
        token_arr.eval()?;
        let first_token = token_arr.item_i32()?;

        // Check if first token is EOS or max_tokens is 0
        let finish_reason = if first_token == eos_token_id || max_tokens == 0 {
            Some(FinishReason::Eos)
        } else if max_tokens == 1 {
            Some(FinishReason::MaxTokens)
        } else {
            None
        };

        let response = BatchResponse {
            uid,
            token_id: first_token,
            finish_reason,
        };

        // Store KV blocks in prefix cache for future reuse
        if let Some(ref mut cm) = self.cache_manager {
            let _ = cm.store_after_prefill(prompt_tokens, &cache);
        }

        // Only store the sequence if it's not already finished
        if finish_reason.is_none() {
            self.sequences.insert(
                uid,
                SeqState {
                    sampler,
                    eos_token_id,
                    max_tokens,
                    generated_count: 1,
                    cache,
                    last_token: first_token,
                    prompt_tokens: prompt_tokens.to_vec(),
                },
            );
        }

        Ok((uid, response))
    }

    /// Execute one decode step for ALL active sequences.
    /// Returns one `BatchResponse` per active sequence.
    /// Finished sequences are automatically removed.
    pub fn step(&mut self) -> Result<Vec<BatchResponse>> {
        let stream = Stream::new(&Device::gpu());
        let mut responses = Vec::with_capacity(self.sequences.len());
        let mut finished_uids = Vec::new();

        // Collect UIDs to iterate (avoid borrow issues)
        let uids: Vec<SeqUid> = self.sequences.keys().copied().collect();

        for uid in uids {
            let seq = self.sequences.get_mut(&uid).unwrap();

            // Forward with last_token
            let input = Array::from_slice_i32(&[seq.last_token]);
            let input_2d = ops::reshape(&input, &[1, 1], &stream)?;
            let logits = self
                .model
                .forward(&input_2d, &mut seq.cache, "causal", None)?;
            let logits_2d = ops::reshape(&logits, &[1, logits.shape()[2]], &stream)?;

            // Sample next token
            let token_arr = sample(&logits_2d, &seq.sampler, &stream)?;
            token_arr.eval()?;
            let next_token = token_arr.item_i32()?;

            seq.generated_count += 1;
            seq.last_token = next_token;

            // Check finish conditions
            let finish_reason = if next_token == seq.eos_token_id {
                Some(FinishReason::Eos)
            } else if seq.generated_count >= seq.max_tokens {
                Some(FinishReason::MaxTokens)
            } else {
                None
            };

            if finish_reason.is_some() {
                finished_uids.push(uid);
            }

            responses.push(BatchResponse {
                uid,
                token_id: next_token,
                finish_reason,
            });
        }

        // Remove finished sequences
        for uid in finished_uids {
            self.sequences.remove(&uid);
        }

        Ok(responses)
    }

    /// Remove a sequence (abort).
    pub fn remove(&mut self, uid: SeqUid) {
        self.sequences.remove(&uid);
    }

    /// Number of active sequences.
    pub fn active_count(&self) -> usize {
        self.sequences.len()
    }

    /// Check if a sequence is still active.
    pub fn is_active(&self, uid: SeqUid) -> bool {
        self.sequences.contains_key(&uid)
    }
}
