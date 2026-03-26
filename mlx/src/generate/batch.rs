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

/// State for a sequence undergoing chunked prefill.
pub struct PrefillState {
    pub uid: SeqUid,
    pub prompt_tokens: Vec<i32>,
    pub processed: usize,
    pub sampler: SamplerConfig,
    pub eos_token_id: i32,
    pub max_tokens: usize,
    cache: Vec<(Option<Array>, Option<Array>)>,
}

/// Result of one chunked prefill step.
pub enum ChunkedPrefillResult {
    /// More chunks remain to be processed.
    InProgress,
    /// Prefill complete. Contains the first generated token response.
    Done(BatchResponse),
}

/// Default chunk size for chunked prefill (in tokens).
const PREFILL_CHUNK_SIZE: usize = 512;

/// BatchGenerator manages multiple sequences sharing a single model.
///
/// Supports three modes:
/// - Full prefill: `insert()` processes entire prompt in one forward pass
/// - Chunked prefill: `begin_prefill()` + `step_prefill_chunk()` for incremental prefill
/// - Decode: `step_batched()` or `step_true_batched()` for token generation
pub struct BatchGenerator<'a> {
    model: &'a Model,
    num_layers: usize,
    sequences: HashMap<SeqUid, SeqState>,
    next_uid: SeqUid,
    cache_manager: Option<CacheManager>,
    stream: Stream,
    /// Sequences undergoing chunked prefill.
    prefilling: Vec<PrefillState>,
    /// Persistent batched cache: [B, n_kv, seq_len, head_dim] per layer.
    /// None when not in batched decode mode (0 or 1 active sequences).
    batched_cache: Option<BatchedCacheState>,
}

/// Pre-allocated capacity for batched cache (tokens beyond current length).
const BATCHED_CACHE_PREALLOC: i32 = 256;

/// Persistent state for batched decode — avoids pad/unpad every step.
struct BatchedCacheState {
    /// Ordered UIDs matching batch dimension positions.
    uids: Vec<SeqUid>,
    /// Per-layer batched KV cache: [B, n_kv, capacity, head_dim].
    /// Pre-allocated to max_cache_len + BATCHED_CACHE_PREALLOC.
    cache: Vec<(Array, Array)>,
    /// Per-sequence cache length (actual data). Updated each step.
    cache_lens: Vec<i32>,
    /// Max cache length across all sequences (actual data, not capacity).
    max_cache_len: i32,
    /// Total allocated capacity (seq_len dimension of cache arrays).
    capacity: i32,
    /// Model dtype for mask casting.
    model_dtype: crate::Dtype,
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
            prefilling: Vec::new(),
            batched_cache: None,
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
            prefilling: Vec::new(),
            batched_cache: None,
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

    /// Begin chunked prefill for a new sequence. Does NOT run any forward pass yet.
    /// Call `step_prefill_chunk()` repeatedly to process chunks.
    pub fn begin_prefill(
        &mut self,
        prompt_tokens: &[i32],
        sampler: SamplerConfig,
        eos_token_id: i32,
        max_tokens: usize,
    ) -> SeqUid {
        let uid = self.next_uid;
        self.next_uid += 1;

        // Try prefix cache lookup
        let (cache, matched_tokens) = if let Some(ref mut cm) = self.cache_manager {
            match cm.lookup_and_load(prompt_tokens) {
                Ok((c, m)) if m > 0 => (c, m),
                _ => ((0..self.num_layers).map(|_| (None, None)).collect(), 0),
            }
        } else {
            ((0..self.num_layers).map(|_| (None, None)).collect(), 0)
        };

        self.prefilling.push(PrefillState {
            uid,
            prompt_tokens: prompt_tokens.to_vec(),
            processed: matched_tokens,
            sampler,
            eos_token_id,
            max_tokens,
            cache,
        });

        uid
    }

    /// Returns the number of sequences currently undergoing chunked prefill.
    pub fn prefilling_count(&self) -> usize {
        self.prefilling.len()
    }

    /// Process one chunk of prefill for all prefilling sequences.
    ///
    /// Each sequence advances by up to `PREFILL_CHUNK_SIZE` tokens.
    /// Sequences that finish prefill are sampled and moved to active decode.
    /// Returns responses for sequences that completed prefill this step.
    pub fn step_prefill_chunk(&mut self) -> Result<Vec<(SeqUid, BatchResponse)>> {
        if self.prefilling.is_empty() {
            return Ok(vec![]);
        }

        let stream = &self.stream;
        let chunk_size = PREFILL_CHUNK_SIZE;
        let mut completed = Vec::new();

        // Process each prefilling sequence's next chunk
        for ps in &mut self.prefilling {
            let remaining = ps.prompt_tokens.len() - ps.processed;
            if remaining == 0 {
                continue;
            }

            let this_chunk = remaining.min(chunk_size);
            let chunk_tokens = &ps.prompt_tokens[ps.processed..ps.processed + this_chunk];

            let chunk_arr = Array::from_slice_i32(chunk_tokens);
            let chunk_2d = ops::reshape(&chunk_arr, &[1, this_chunk as i32], stream)?;

            let _logits = self
                .model
                .forward(&chunk_2d, &mut ps.cache, "causal", None)?;

            ps.processed += this_chunk;

            // If this was the last chunk, sample the first token
            if ps.processed >= ps.prompt_tokens.len() {
                let last_idx = this_chunk as i32 - 1;
                let vocab_size = _logits.shape()[2];
                let last_logits = ops::slice(
                    &_logits,
                    &[0, last_idx, 0],
                    &[1, last_idx + 1, vocab_size],
                    &[1, 1, 1],
                    stream,
                )?;
                let last_logits = ops::reshape(&last_logits, &[1, vocab_size], stream)?;
                let token_arr = sample(&last_logits, &ps.sampler, stream)?;
                token_arr.eval()?;
                let first_token = token_arr.item_i32()?;

                let finish_reason = if first_token == ps.eos_token_id || ps.max_tokens == 0 {
                    Some(FinishReason::Eos)
                } else if ps.max_tokens == 1 {
                    Some(FinishReason::MaxTokens)
                } else {
                    None
                };

                completed.push((ps.uid, first_token, finish_reason));
            }
        }

        // Move completed sequences from prefilling to active (or drop if finished)
        let mut results = Vec::new();
        let completed_uids: Vec<SeqUid> = completed.iter().map(|(uid, _, _)| *uid).collect();

        for (uid, first_token, finish_reason) in completed {
            // Find and remove from prefilling
            let idx = self.prefilling.iter().position(|ps| ps.uid == uid);
            if let Some(idx) = idx {
                let ps = self.prefilling.swap_remove(idx);

                // Store KV blocks in prefix cache
                if let Some(ref mut cm) = self.cache_manager {
                    let _ = cm.store_after_prefill(&ps.prompt_tokens, &ps.cache);
                }

                let response = BatchResponse {
                    uid,
                    token_id: first_token,
                    finish_reason,
                };

                if finish_reason.is_none() {
                    self.sequences.insert(
                        uid,
                        SeqState {
                            sampler: ps.sampler,
                            eos_token_id: ps.eos_token_id,
                            max_tokens: ps.max_tokens,
                            generated_count: 1,
                            cache: ps.cache,
                            last_token: first_token,
                            prompt_tokens: ps.prompt_tokens,
                        },
                    );
                }

                results.push((uid, response));
            }
        }

        // Discard logits for non-completed sequences (only last chunk logits matter)
        let _ = completed_uids;

        Ok(results)
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

    /// Build or rebuild the persistent batched cache from per-sequence caches.
    fn build_batched_cache(&mut self) -> Result<()> {
        let stream = &self.stream;
        let uids: Vec<SeqUid> = self.sequences.keys().copied().collect();
        let num_layers = self.num_layers;

        let mut cache_lens: Vec<i32> = Vec::with_capacity(uids.len());
        for &uid in &uids {
            let seq = &self.sequences[&uid];
            let cl = seq.cache[0].0.as_ref().map_or(0, |k| k.shape()[2]);
            cache_lens.push(cl);
        }
        let max_cache_len = *cache_lens.iter().max().unwrap_or(&0);

        let model_dtype = self.sequences[&uids[0]].cache[0]
            .0
            .as_ref()
            .map(|k| k.dtype())
            .unwrap_or(crate::Dtype::Float32);

        let mut batched_layers: Vec<(Array, Array)> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let mut k_arrays: Vec<Array> = Vec::with_capacity(uids.len());
            let mut v_arrays: Vec<Array> = Vec::with_capacity(uids.len());

            for (i, &uid) in uids.iter().enumerate() {
                let seq = &self.sequences[&uid];
                let (ref ck, ref cv) = seq.cache[layer_idx];
                let pad_len = max_cache_len - cache_lens[i];

                if let (Some(k), Some(v)) = (ck, cv) {
                    if pad_len > 0 {
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
                    return Err(crate::error::Error::Mlx(
                        "missing cache in build_batched_cache".to_string(),
                    ));
                }
            }

            let k_refs: Vec<&Array> = k_arrays.iter().collect();
            let v_refs: Vec<&Array> = v_arrays.iter().collect();
            let k_vec = VectorArray::from_arrays(&k_refs);
            let v_vec = VectorArray::from_arrays(&v_refs);
            batched_layers.push((
                ops::concatenate(&k_vec, 0, stream)?,
                ops::concatenate(&v_vec, 0, stream)?,
            ));
        }

        // Pre-allocate extra capacity to avoid rebuilds
        let capacity = max_cache_len + BATCHED_CACHE_PREALLOC;
        let extra = capacity - max_cache_len;
        if extra > 0 {
            for (bk, bv) in &mut batched_layers {
                let shape = bk.shape();
                let pad_shape = &[shape[0], shape[1], extra, shape[3]];
                let pad_k = ops::zeros(pad_shape, bk.dtype(), stream)?;
                let pad_v = ops::zeros(pad_shape, bv.dtype(), stream)?;
                let k_arr = VectorArray::from_arrays(&[&*bk, &pad_k]);
                let v_arr = VectorArray::from_arrays(&[&*bv, &pad_v]);
                *bk = ops::concatenate(&k_arr, 2, stream)?;
                *bv = ops::concatenate(&v_arr, 2, stream)?;
            }
        }

        self.batched_cache = Some(BatchedCacheState {
            uids,
            cache: batched_layers,
            cache_lens,
            max_cache_len,
            capacity,
            model_dtype,
        });
        Ok(())
    }

    /// Invalidate the persistent batched cache (e.g. when sequences change).
    fn invalidate_batched_cache(&mut self) {
        self.batched_cache = None;
    }

    /// True batched decode with persistent cache.
    ///
    /// First call builds the batched cache from per-sequence caches (pad once).
    /// Subsequent calls reuse the batched cache — forward_batched appends new KV
    /// entries in-place, no pad/unpad per step.
    /// When a sequence finishes, the cache is invalidated and rebuilt next step.
    pub fn step_true_batched(&mut self) -> Result<Vec<BatchResponse>> {
        let uids: Vec<SeqUid> = self.sequences.keys().copied().collect();

        if uids.is_empty() {
            return Ok(vec![]);
        }

        if uids.len() == 1 {
            self.invalidate_batched_cache();
            return self.step_batched();
        }

        let step_t0 = std::time::Instant::now();

        // Build batched cache if not present or if sequences changed
        let needs_rebuild = match &self.batched_cache {
            None => true,
            Some(bc) => bc.uids != uids,
        };
        if needs_rebuild {
            self.build_batched_cache()?;
            let bc = self.batched_cache.as_ref().unwrap();
            eprintln!(
                "[batched] built persistent cache: B={}, max_len={}, capacity={}",
                uids.len(),
                bc.max_cache_len,
                bc.capacity
            );
        }

        // Check if we need to re-allocate (exceeded pre-allocated capacity)
        let bc = self.batched_cache.as_ref().unwrap();
        if bc.max_cache_len + 1 > bc.capacity {
            // Rebuild with fresh pre-allocation
            self.invalidate_batched_cache();
            self.build_batched_cache()?;
            let bc = self.batched_cache.as_ref().unwrap();
            eprintln!(
                "[batched] capacity exceeded, rebuilt: B={}, max_len={}, capacity={}",
                uids.len(),
                bc.max_cache_len,
                bc.capacity
            );
        }

        let stream = &self.stream;
        let batch_size = uids.len() as i32;

        // Collect tokens and build offsets/write_positions from persistent state
        let bc = self.batched_cache.as_ref().unwrap();
        let mut tokens_vec: Vec<i32> = Vec::with_capacity(uids.len());
        for &uid in &bc.uids {
            tokens_vec.push(self.sequences[&uid].last_token);
        }

        let tokens_arr = Array::from_slice_i32(&tokens_vec);
        let tokens_2d = ops::reshape(&tokens_arr, &[batch_size, 1], stream)?;
        let offsets_arr = Array::from_slice_i32(&bc.cache_lens);

        // Write positions for scatter: each sequence writes at its current cache_len
        // (within the pre-allocated buffer)
        let write_pos_arr = Array::from_slice_i32(&bc.cache_lens);

        // Build attention mask [B, 1, 1, capacity]
        // Mask: -inf for positions before data start AND after data end
        let cap = bc.capacity;
        let mut mask_data: Vec<f32> = Vec::with_capacity(uids.len() * cap as usize);
        for &cl in &bc.cache_lens {
            let left_pad = (bc.max_cache_len - cl) as usize;
            let valid = (cl + 1) as usize; // current data + 1 new token
            let right_pad = (cap - bc.max_cache_len - 1).max(0) as usize;
            mask_data.extend(std::iter::repeat_n(f32::NEG_INFINITY, left_pad));
            mask_data.extend(std::iter::repeat_n(0.0_f32, valid));
            mask_data.extend(std::iter::repeat_n(f32::NEG_INFINITY, right_pad));
        }
        let mask_flat = Array::from_slice_f32(&mask_data);
        let mask = ops::reshape(&mask_flat, &[batch_size, 1, 1, cap], stream)?;
        let mask = ops::astype(&mask, bc.model_dtype, stream)?;

        let build_ms = step_t0.elapsed().as_secs_f64() * 1000.0;

        // Forward pass with scatter-based KV update (no concat, no allocation)
        let fwd_t0 = std::time::Instant::now();
        let bc_mut = self.batched_cache.as_mut().unwrap();
        let logits = self.model.forward_batched(
            &tokens_2d,
            &mut bc_mut.cache,
            &offsets_arr,
            &mask,
            Some(&write_pos_arr),
        )?;
        let fwd_ms = fwd_t0.elapsed().as_secs_f64() * 1000.0;

        // Update cache_lens and max_cache_len (each sequence grew by 1 token)
        let bc_mut = self.batched_cache.as_mut().unwrap();
        for cl in &mut bc_mut.cache_lens {
            *cl += 1;
        }
        bc_mut.max_cache_len += 1;

        // Sample per-sequence
        let bc = self.batched_cache.as_ref().unwrap();
        let mut pending: Vec<(SeqUid, Array)> = Vec::with_capacity(uids.len());
        for (i, &uid) in bc.uids.iter().enumerate() {
            let seq = &self.sequences[&uid];
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

        // Eval
        for (_, token_arr) in &pending {
            token_arr.eval()?;
        }

        let eval_ms = step_t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[batched] B={} build={:.1}ms fwd={:.1}ms eval_total={:.1}ms{}",
            batch_size,
            build_ms,
            fwd_ms,
            eval_ms,
            if needs_rebuild { " (rebuilt)" } else { "" }
        );

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

        // If any sequence finished, invalidate batched cache
        // (will be rebuilt next step with remaining sequences)
        if !finished_uids.is_empty() {
            // Extract per-sequence caches from pre-allocated batched cache
            let bc = self.batched_cache.as_ref().unwrap();
            for &uid in &uids {
                if finished_uids.contains(&uid) {
                    continue; // will be removed below
                }
                let i = bc.uids.iter().position(|&u| u == uid).unwrap();
                let seq = self.sequences.get_mut(&uid).unwrap();
                let seq_len = bc.cache_lens[i];
                // Data starts at left_pad position within the capacity
                let left_pad = bc.max_cache_len - seq_len;
                for (layer_idx, (bk, bv)) in bc.cache.iter().enumerate() {
                    let bk_shape = bk.shape();
                    let seq_k = ops::slice(
                        bk,
                        &[i as i32, 0, left_pad, 0],
                        &[i as i32 + 1, bk_shape[1], left_pad + seq_len, bk_shape[3]],
                        &[1, 1, 1, 1],
                        stream,
                    )?;
                    let seq_v = ops::slice(
                        bv,
                        &[i as i32, 0, left_pad, 0],
                        &[i as i32 + 1, bk_shape[1], left_pad + seq_len, bk_shape[3]],
                        &[1, 1, 1, 1],
                        stream,
                    )?;
                    seq.cache[layer_idx] = (Some(seq_k), Some(seq_v));
                }
            }

            self.invalidate_batched_cache();
            for uid in finished_uids {
                self.sequences.remove(&uid);
            }
        }

        Ok(responses)
    }

    /// Remove a sequence (abort).
    pub fn remove(&mut self, uid: SeqUid) {
        self.sequences.remove(&uid);
        self.invalidate_batched_cache();
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
