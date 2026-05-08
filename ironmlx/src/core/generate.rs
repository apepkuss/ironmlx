//! Single-request generation driver: prefill + decode + sampler + EOS termination.
//!
//! Borrows a [`Qwen35Model`] and [`Tokenizer`] for the lifetime of the stream;
//! owns the per-call cache vector and accumulating token history.

use anyhow::anyhow;
use mlx::{Array, Dtype};

use crate::core::sampler::Sampler;
use crate::core::tokenizer::Tokenizer;
use crate::models::Qwen35Model;
use crate::nn::LayerCache;
use crate::Result;

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Tokenized prompt (after chat template rendering, if any).
    pub prompt_ids: Vec<u32>,
    /// Hard cap on tokens generated beyond the prompt.
    pub max_new_tokens: usize,
    /// Sampling configuration. Defaults to greedy if left at `Sampler::greedy()`.
    pub sampler: Sampler,
    /// Token ids that terminate the stream when produced.
    pub stop_token_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct GenerateEvent {
    /// The token id this step produced.
    pub token: u32,
    /// Incremental decoded text since the previous event. May be empty
    /// (BPE boundary not yet reached); callers should concatenate.
    pub text: String,
    /// Some on the final event: "stop" (EOS hit) or "length" (max_new_tokens).
    pub finish_reason: Option<&'static str>,
}

/// Single-request prefill+decode driver. Owns a per-call cache vector and
/// accumulates token history; yields one [`GenerateEvent`] per decode step
/// until EOS or `max_new_tokens`.
pub struct GenerationStream<'m> {
    model: &'m Qwen35Model,
    tokenizer: &'m Tokenizer,
    cache: Vec<LayerCache>,
    /// All token ids so far: prompt ++ generated.
    history: Vec<u32>,
    /// Last full-text snapshot — diffed against the next decode to produce incremental text.
    last_decoded_text: String,
    request: GenerateRequest,
    finished: bool,
}

/// Build a position_ids Array of shape `[3, 1, len]` with values
/// `[start_pos, start_pos+1, ..., start_pos+len-1]` repeated across all 3 streams.
/// All three Mrope streams hold the same sequence for text-only single-request paths.
pub fn build_position_ids(start_pos: i32, len: i32) -> Result<Array> {
    if len <= 0 {
        return Err(anyhow!(
            "build_position_ids: len must be positive, got {len}"
        ));
    }
    let one_stream = mlx::ops::constructors::arange(
        start_pos as f64,
        (start_pos + len) as f64,
        1.0,
        Dtype::Int32,
    )?;
    let one_stream = one_stream.reshape((1, 1, len))?;
    mlx::ops::shape::broadcast_to(&one_stream, &[3_i32, 1, len][..]).map_err(anyhow::Error::from)
}

impl<'m> GenerationStream<'m> {
    pub fn new(
        model: &'m Qwen35Model,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
    ) -> Result<Self> {
        if request.prompt_ids.is_empty() {
            return Err(anyhow!("GenerationStream::new: prompt_ids cannot be empty"));
        }
        let prompt_len = request.prompt_ids.len();
        let cap = (prompt_len + request.max_new_tokens) as i32;
        let dtype = Dtype::Bfloat16;
        let mut cache = model.make_cache(/* batch */ 1, cap, dtype)?;

        // Prefill: shape [1, prompt_len] u32.
        let prompt_arr: Array = (
            request.prompt_ids.as_slice(),
            &[1_i32, prompt_len as i32][..],
        )
            .try_into()?;
        let position_ids = build_position_ids(0, prompt_len as i32)?;

        let logits = model.forward_on(&prompt_arr, &position_ids, Some(&mut cache), ())?;
        // logits shape [1, prompt_len, vocab]. Extract the last position slice.
        let vocab = logits.shape().as_slice()[2];
        let last_logits = mlx::ops::indexing::slice_strided(
            &logits,
            &[0_i32, (prompt_len as i32) - 1, 0][..],
            &[1_i32, prompt_len as i32, vocab][..],
            &[1_i32, 1, 1][..],
        )?;
        // Flatten to [vocab] for Sampler.
        let last_logits = last_logits.reshape((vocab,))?;

        let mut history = request.prompt_ids.clone();
        let first_token = request.sampler.sample(&last_logits, &history)?;
        history.push(first_token);

        // Initial decoded text = full history decoded; subsequent calls diff against this.
        let initial_text = tokenizer
            .decode(&history, /* skip_special = */ true)
            .unwrap_or_default();

        Ok(Self {
            model,
            tokenizer,
            cache,
            history,
            last_decoded_text: initial_text,
            request,
            finished: false,
        })
    }

    /// Pull the next event. Returns `Ok(None)` after the stream terminates.
    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }

        // The token to emit is the most-recent push to history.
        let token = *self.history.last().expect("history non-empty post-new");

        // Compute incremental text via cumulative-detok diff.
        let full_text = self
            .tokenizer
            .decode(&self.history, /* skip_special = */ true)
            .unwrap_or_default();
        let text = full_text
            .strip_prefix(&self.last_decoded_text)
            .unwrap_or(&full_text)
            .to_string();
        self.last_decoded_text = full_text;

        // Termination check using the just-emitted token.
        let new_count = self.history.len() - self.request.prompt_ids.len();
        let finish_reason = if self.request.stop_token_ids.contains(&token) {
            Some("stop")
        } else if new_count >= self.request.max_new_tokens {
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

        // Decode one step: feed the just-emitted token back through the model.
        let token_arr: Array = (&[token][..], &[1_i32, 1][..]).try_into()?;
        let pos = (self.history.len() - 1) as i32;
        let position_ids = build_position_ids(pos, 1)?;
        let logits = self
            .model
            .forward_on(&token_arr, &position_ids, Some(&mut self.cache), ())?;
        // Logits shape [1, 1, vocab] — flatten to [vocab].
        let vocab = logits.shape().as_slice()[2];
        let logits_flat = logits.reshape((vocab,))?;
        let next = self.request.sampler.sample(&logits_flat, &self.history)?;
        self.history.push(next);

        Ok(Some(GenerateEvent {
            token,
            text,
            finish_reason: None,
        }))
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The unit tests in this module would normally use a "mock model" that
    // returns deterministic logits. Building one in-tree is non-trivial
    // because Qwen35Model is a concrete type with no trait abstraction
    // (per Boss memory: avoid trait + dyn dispatch on hot paths).
    //
    // Instead, we exercise the structural invariants of the GenerationStream
    // API surface here. End-to-end correctness is verified by:
    //   1. Task 6's logits-alignment integration test (real 4B checkpoint).
    //   2. Task 10's HTTP smoke test.

    #[test]
    fn build_position_ids_shape_and_values() {
        let p = build_position_ids(/* start_pos */ 5, /* len */ 4).expect("build");
        assert_eq!(p.shape().as_slice(), &[3, 1, 4]);
        let v: Vec<i32> = p.to_vec().unwrap();
        // 3 streams * 1 batch * 4 positions = 12 entries.
        assert_eq!(v.len(), 12);
        // Each of the 3 streams holds [5, 6, 7, 8].
        for stream in 0..3 {
            for k in 0..4 {
                assert_eq!(v[stream * 4 + k], 5 + k as i32, "stream {stream}, k {k}");
            }
        }
    }

    #[test]
    fn build_position_ids_rejects_zero_len() {
        let r = build_position_ids(0, 0);
        assert!(r.is_err(), "len=0 must Err");
    }

    #[test]
    fn generate_event_struct_field_visibility() {
        let ev = GenerateEvent {
            token: 7,
            text: "abc".into(),
            finish_reason: Some("stop"),
        };
        assert_eq!(ev.token, 7);
        assert_eq!(ev.text, "abc");
        assert_eq!(ev.finish_reason, Some("stop"));
    }
}
