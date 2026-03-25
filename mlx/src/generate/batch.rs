use std::collections::{BTreeMap, HashMap};

use crate::array::Array;
use crate::cache::CacheManager;
use crate::device::Device;
use crate::error::Result;
use crate::media::ProcessedMedia;
use crate::model::Model;
use crate::ops;
use crate::stream::Stream;

use crate::vector::VectorArray;

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
    stream: Stream,
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
            stream: Stream::new(&Device::gpu()),
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
            stream: Stream::new(&Device::gpu()),
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

        let stream = &self.stream;

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
        let prompt_2d = ops::reshape(&prompt_arr, &[1, tokens_to_prefill.len() as i32], stream)?;
        let logits = self.model.forward(&prompt_2d, &mut cache, "causal", None)?;

        // Extract last token logits (from tokens_to_prefill output)
        let last_idx = tokens_to_prefill.len() as i32 - 1;
        let vocab_size = logits.shape()[2];
        let last_logits = ops::slice(
            &logits,
            &[0, last_idx, 0],
            &[1, last_idx + 1, vocab_size],
            &[1, 1, 1],
            stream,
        )?;
        let last_logits = ops::reshape(&last_logits, &[1, vocab_size], stream)?;

        // Sample first token
        let token_arr = sample(&last_logits, &sampler, stream)?;
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

    /// Insert a new VLM sequence with media. Performs prefill with vision encoding.
    pub fn insert_vlm(
        &mut self,
        prompt_tokens: &[i32],
        media: &[ProcessedMedia],
        sampler: SamplerConfig,
        eos_token_id: i32,
        max_tokens: usize,
    ) -> Result<(SeqUid, BatchResponse)> {
        let uid = self.next_uid;
        self.next_uid += 1;

        let stream = &self.stream;
        let mut cache: Vec<(Option<Array>, Option<Array>)> =
            (0..self.num_layers).map(|_| (None, None)).collect();

        // VLM prefill: model.forward_vlm() handles vision encoding + embedding injection
        let prompt_arr = Array::from_slice_i32(prompt_tokens);
        let prompt_2d = ops::reshape(&prompt_arr, &[1, prompt_tokens.len() as i32], stream)?;
        let logits = self
            .model
            .forward_vlm(&prompt_2d, Some(media), &mut cache)?;

        // Extract last token logits
        let last_idx = prompt_tokens.len() as i32 - 1;
        let vocab_size = logits.shape()[2];
        let last_logits = ops::slice(
            &logits,
            &[0, last_idx, 0],
            &[1, last_idx + 1, vocab_size],
            &[1, 1, 1],
            stream,
        )?;
        let last_logits = ops::reshape(&last_logits, &[1, vocab_size], stream)?;

        let token_arr = sample(&last_logits, &sampler, stream)?;
        token_arr.eval()?;
        let first_token = token_arr.item_i32()?;

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
        let stream = &self.stream;
        let mut responses = Vec::with_capacity(self.sequences.len());
        let mut finished_uids = Vec::new();

        // Collect UIDs to iterate (avoid borrow issues)
        let uids: Vec<SeqUid> = self.sequences.keys().copied().collect();

        for uid in uids {
            let seq = self.sequences.get_mut(&uid).unwrap();

            // Forward with last_token
            let input = Array::from_slice_i32(&[seq.last_token]);
            let input_2d = ops::reshape(&input, &[1, 1], stream)?;
            let logits = self
                .model
                .forward(&input_2d, &mut seq.cache, "causal", None)?;
            let logits_2d = ops::reshape(&logits, &[1, logits.shape()[2]], stream)?;

            // Sample next token
            let token_arr = sample(&logits_2d, &seq.sampler, stream)?;
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

    /// Execute one decode step with batched evaluation.
    ///
    /// Unlike `step()` which evals per-sequence, this method:
    /// 1. Runs all forward passes (MLX lazy — builds compute graph)
    /// 2. Samples all tokens (still lazy)
    /// 3. Calls eval once for the entire batch (single GPU dispatch)
    ///
    /// This leverages MLX's lazy evaluation to merge multiple per-sequence
    /// forward passes into one GPU submission, improving throughput for
    /// concurrent requests without changing the per-sequence cache structure.
    pub fn step_batched(&mut self) -> Result<Vec<BatchResponse>> {
        let stream = &self.stream;
        let uids: Vec<SeqUid> = self.sequences.keys().copied().collect();

        if uids.is_empty() {
            return Ok(vec![]);
        }

        // Phase 1: Build compute graph (all lazy, no eval yet)
        let mut pending: Vec<(SeqUid, Array)> = Vec::with_capacity(uids.len());
        for &uid in &uids {
            let seq = self.sequences.get_mut(&uid).unwrap();
            let input = Array::from_slice_i32(&[seq.last_token]);
            let input_2d = ops::reshape(&input, &[1, 1], stream)?;
            let logits = self
                .model
                .forward(&input_2d, &mut seq.cache, "causal", None)?;
            let logits_2d = ops::reshape(&logits, &[1, logits.shape()[2]], stream)?;
            let token_arr = sample(&logits_2d, &seq.sampler, stream)?;
            pending.push((uid, token_arr));
        }

        // Phase 2: Single eval — MLX merges all pending ops into one GPU dispatch
        for (_, token_arr) in &pending {
            token_arr.eval()?;
        }

        // Phase 3: Collect results
        let mut responses = Vec::with_capacity(pending.len());
        let mut finished_uids = Vec::new();

        for (uid, token_arr) in pending {
            let next_token = token_arr.item_i32()?;
            let seq = self.sequences.get_mut(&uid).unwrap();
            seq.generated_count += 1;
            seq.last_token = next_token;

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

        for uid in finished_uids {
            self.sequences.remove(&uid);
        }

        Ok(responses)
    }

    /// True batched decode: merge all sequences into a single forward pass.
    ///
    /// Steps:
    /// 1. Collect per-sequence tokens and cache lengths
    /// 2. Pad caches to uniform length, build attention mask
    /// 3. Single batched forward pass
    /// 4. Unpad caches back to per-sequence
    /// 5. Sample and collect results
    pub fn step_true_batched(&mut self) -> Result<Vec<BatchResponse>> {
        let stream = &self.stream;
        let uids: Vec<SeqUid> = self.sequences.keys().copied().collect();

        if uids.is_empty() {
            return Ok(vec![]);
        }

        // Single sequence: skip padding overhead, use per-sequence path
        if uids.len() == 1 {
            return self.step_batched();
        }

        let num_layers = self.num_layers;
        let batch_size = uids.len() as i32;

        // Collect per-sequence info
        let mut tokens_vec: Vec<i32> = Vec::with_capacity(uids.len());
        let mut offsets_vec: Vec<i32> = Vec::with_capacity(uids.len());
        let mut cache_lens: Vec<i32> = Vec::with_capacity(uids.len());

        for &uid in &uids {
            let seq = &self.sequences[&uid];
            tokens_vec.push(seq.last_token);
            let cache_len = seq.cache[0].0.as_ref().map_or(0, |k| k.shape()[2]);
            offsets_vec.push(cache_len);
            cache_lens.push(cache_len);
        }

        let max_cache_len = *cache_lens.iter().max().unwrap_or(&0);

        // Build batched input tokens [B, 1]
        let tokens_arr = Array::from_slice_i32(&tokens_vec);
        let tokens_2d = ops::reshape(&tokens_arr, &[batch_size, 1], stream)?;

        // Build offsets array [B]
        let offsets_arr = Array::from_slice_i32(&offsets_vec);

        // Build attention mask [B, 1, 1, max_cache_len + 1]
        // For each sequence: 0.0 for valid positions, -inf for pad positions
        let total_len = max_cache_len + 1; // +1 for the new token
        let mut mask_data: Vec<f32> = Vec::with_capacity(uids.len() * total_len as usize);
        for &cl in &cache_lens {
            let pad_len = (max_cache_len - cl) as usize;
            // Left-padded: first `pad_len` positions are masked, rest are valid
            mask_data.extend(std::iter::repeat_n(f32::NEG_INFINITY, pad_len));
            mask_data.extend(std::iter::repeat_n(0.0_f32, (cl + 1) as usize));
        }
        let mask_flat = Array::from_slice_f32(&mask_data);
        let mask = ops::reshape(&mask_flat, &[batch_size, 1, 1, total_len], stream)?;

        // Pad per-sequence caches to [B, n_kv_heads, max_cache_len, head_dim]
        let mut batched_cache: Vec<(Array, Array)> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let mut k_arrays: Vec<Array> = Vec::with_capacity(uids.len());
            let mut v_arrays: Vec<Array> = Vec::with_capacity(uids.len());

            for &uid in &uids {
                let seq = &self.sequences[&uid];
                let (ref ck, ref cv) = seq.cache[layer_idx];
                let cl = ck.as_ref().map_or(0, |k| k.shape()[2]);
                let pad_len = max_cache_len - cl;

                if let (Some(k), Some(v)) = (ck, cv) {
                    if pad_len > 0 {
                        // Left-pad with zeros: [1, n_kv, pad_len, head_dim]
                        let k_shape = k.shape();
                        let pad_shape = &[1, k_shape[1], pad_len, k_shape[3]];
                        let pad_k = ops::zeros(pad_shape, k.dtype(), stream)?;
                        let pad_v = ops::zeros(pad_shape, v.dtype(), stream)?;
                        let k_arr = VectorArray::from_arrays(&[&pad_k, k]);
                        let v_arr = VectorArray::from_arrays(&[&pad_v, v]);
                        k_arrays.push(ops::concatenate(&k_arr, 2, stream)?);
                        v_arrays.push(ops::concatenate(&v_arr, 2, stream)?);
                    } else {
                        k_arrays.push(k.clone());
                        v_arrays.push(v.clone());
                    }
                } else {
                    // No cache yet — shouldn't happen in decode, but handle gracefully
                    // Create zero cache of max_cache_len
                    // We need shape info from another sequence
                    return self.step_batched(); // fallback
                }
            }

            // Stack along batch dim: [B, n_kv, max_cache_len, head_dim]
            let k_refs: Vec<&Array> = k_arrays.iter().collect();
            let v_refs: Vec<&Array> = v_arrays.iter().collect();
            let k_vec = VectorArray::from_arrays(&k_refs);
            let v_vec = VectorArray::from_arrays(&v_refs);
            let batched_k = ops::concatenate(&k_vec, 0, stream)?;
            let batched_v = ops::concatenate(&v_vec, 0, stream)?;
            batched_cache.push((batched_k, batched_v));
        }

        // Single forward pass [B, 1] -> [B, 1, vocab]
        let logits =
            self.model
                .forward_batched(&tokens_2d, &mut batched_cache, &offsets_arr, &mask)?;

        // Sample per-sequence
        let mut pending: Vec<(SeqUid, Array)> = Vec::with_capacity(uids.len());
        for (i, &uid) in uids.iter().enumerate() {
            let seq = &self.sequences[&uid];
            // Slice logits for this sequence: [1, vocab]
            let seq_logits = ops::slice(
                &logits,
                &[i as i32, 0, 0],
                &[i as i32 + 1, 1, logits.shape()[2]],
                &[1, 1, 1],
                stream,
            )?;
            let seq_logits_2d = ops::reshape(&seq_logits, &[1, logits.shape()[2]], stream)?;
            let token_arr = sample(&seq_logits_2d, &seq.sampler, stream)?;
            pending.push((uid, token_arr));
        }

        // Eval all sampled tokens
        for (_, token_arr) in &pending {
            token_arr.eval()?;
        }

        // Unpad caches back to per-sequence
        for (i, &uid) in uids.iter().enumerate() {
            let seq = self.sequences.get_mut(&uid).unwrap();
            for (layer_idx, (bk, bv)) in batched_cache.iter().enumerate() {
                let bk_shape = bk.shape();
                // New cache len = old offset + 1 (new token)
                let new_len = offsets_vec[i] + 1;
                // Extract this sequence's cache: slice batch dim and remove left padding
                let pad_len = max_cache_len + 1 - new_len;
                let seq_k = ops::slice(
                    bk,
                    &[i as i32, 0, pad_len, 0],
                    &[i as i32 + 1, bk_shape[1], max_cache_len + 1, bk_shape[3]],
                    &[1, 1, 1, 1],
                    stream,
                )?;
                let seq_v = ops::slice(
                    bv,
                    &[i as i32, 0, pad_len, 0],
                    &[i as i32 + 1, bk_shape[1], max_cache_len + 1, bk_shape[3]],
                    &[1, 1, 1, 1],
                    stream,
                )?;
                seq.cache[layer_idx] = (Some(seq_k), Some(seq_v));
            }
        }

        // Collect results
        let mut responses = Vec::with_capacity(pending.len());
        let mut finished_uids = Vec::new();

        for (uid, token_arr) in pending {
            let next_token = token_arr.item_i32()?;
            let seq = self.sequences.get_mut(&uid).unwrap();
            seq.generated_count += 1;
            seq.last_token = next_token;

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

    /// Check if a set of sequences can be merged into a batch.
    /// Requirements: all must have the same cache state (all cached or all uncached).
    /// Returns true if batchable.
    pub fn can_batch_together(&self, uids: &[SeqUid]) -> bool {
        if uids.len() <= 1 {
            return true;
        }
        let first_uid = uids[0];
        let first_cache_len = match self.sequences.get(&first_uid) {
            Some(seq) => seq.cache[0].0.as_ref().map(|k| k.shape()[2]).unwrap_or(0),
            None => return false,
        };
        uids.iter().all(|&uid| {
            self.sequences.get(&uid).is_some_and(|s| {
                let cache_len = s.cache[0].0.as_ref().map(|k| k.shape()[2]).unwrap_or(0);
                cache_len == first_cache_len
            })
        })
    }

    /// Partition sequences into groups that can be batched together.
    /// Each group has sequences with the same cache length.
    pub fn partition_by_cache_state(&self, uids: &[SeqUid]) -> Vec<Vec<SeqUid>> {
        let mut groups: BTreeMap<i32, Vec<SeqUid>> = BTreeMap::new();
        for &uid in uids {
            if let Some(seq) = self.sequences.get(&uid) {
                let cache_len = seq.cache[0].0.as_ref().map(|k| k.shape()[2]).unwrap_or(0);
                groups.entry(cache_len).or_default().push(uid);
            }
        }
        groups.into_values().collect()
    }
}
