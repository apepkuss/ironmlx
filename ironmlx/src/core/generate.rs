//! Single-request generation driver: prefill + decode + sampler + EOS termination.
//!
//! Borrows a [`Qwen35Model`] and [`Tokenizer`] for the lifetime of the stream;
//! owns the per-call cache vector and accumulating token history.

use anyhow::anyhow;
use mlx::{Array, Dtype};

use crate::core::sampler::Sampler;
use crate::core::tokenizer::{DecodeStream, Tokenizer};
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
///
/// At construction the driver classifies the sampler:
/// - **Pipelined mode** (greedy + no penalties): each `next_token` call
///   pre-dispatches step N+1's forward+argmax+async_eval before
///   materialising step N's `.item()`, fully overlapping CPU and GPU work.
///   Token text is produced incrementally via [`DecodeStream`] (O(1) per
///   step instead of O(N²) full-history decode).
/// - **Synchronous mode** (temperature > 0 or any penalty configured):
///   forward → sample.item() → push history → decode full history → diff
///   loop, identical to pre-P8a behavior. The non-greedy paths already
///   call `.to_vec()` for penalty masking, defeating any pipelining
///   benefit, so they stay on the simpler path.
pub struct GenerationStream<'m> {
    model: &'m Qwen35Model,
    tokenizer: &'m Tokenizer,
    cache: Vec<LayerCache>,
    /// All token ids so far: prompt ++ generated.
    history: Vec<u32>,
    request: GenerateRequest,
    finished: bool,

    // Mode selector — set once by `new()`, read each `next_token`.
    pipelined: bool,

    // — Pipelined-mode state (Some iff pipelined=true) —
    /// Lazy `[shape]` u32 Array — the token next_token() will emit on its
    /// next non-finished call. Always pre-dispatched via async_eval so the
    /// GPU has work to do while we materialise it.
    pending_token_arr: Option<Array>,
    /// Incremental BPE detokenizer; receives one push per emitted token.
    detok: Option<DecodeStream<'m>>,

    // — Synchronous-mode state (populated iff pipelined=false) —
    /// Last full-text snapshot — diffed against the next decode to produce
    /// incremental text. Sync path only.
    last_decoded_text: String,
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
        let vocab = logits.shape().as_slice()[2];
        let last_logits = mlx::ops::indexing::slice_strided(
            &logits,
            &[0_i32, (prompt_len as i32) - 1, 0][..],
            &[1_i32, prompt_len as i32, vocab][..],
            &[1_i32, 1, 1][..],
        )?;
        let last_logits = last_logits.reshape((vocab,))?;

        let history = request.prompt_ids.clone();
        let pipelined = request.sampler.is_pipelinable();

        if pipelined {
            // Pipelined path: pending_token_arr starts as the prefill's argmax,
            // pre-dispatched via async_eval so the GPU is already working on
            // it by the time the first next_token() call materialises it.
            let pending = request.sampler.sample_async_greedy(&last_logits)?;
            mlx::transforms::async_eval(&[&pending])?;
            let detok = tokenizer.decode_stream(/* skip_special */ true);

            Ok(Self {
                model,
                tokenizer,
                cache,
                history,
                request,
                finished: false,
                pipelined: true,
                pending_token_arr: Some(pending),
                detok: Some(detok),
                last_decoded_text: String::new(),
            })
        } else {
            // Sync path: existing pre-P8a behavior. First token sampled
            // synchronously here; pushed into history; initial text snapshot
            // captured for incremental diff.
            let first_token = request.sampler.sample(&last_logits, &history)?;
            let mut history = history;
            history.push(first_token);

            let initial_text = tokenizer
                .decode(&history, /* skip_special = */ true)
                .unwrap_or_default();

            Ok(Self {
                model,
                tokenizer,
                cache,
                history,
                request,
                finished: false,
                pipelined: false,
                pending_token_arr: None,
                detok: None,
                last_decoded_text: initial_text,
            })
        }
    }

    /// Pull the next event. Returns `Ok(None)` after the stream terminates.
    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }
        if self.pipelined {
            self.next_token_pipelined()
        } else {
            self.next_token_sync()
        }
    }

    /// Pipelined hot path. Invariant: `self.pending_token_arr` is `Some` and
    /// the lazy [shape] u32 Array of the token to be returned on this call.
    fn next_token_pipelined(&mut self) -> Result<Option<GenerateEvent>> {
        // 1. Materialise the pending token. The GPU has been working on it
        //    since the previous next_token call's async_eval (or new()).
        let pending = self
            .pending_token_arr
            .as_ref()
            .expect("pipelined mode invariant: pending_token_arr is Some");
        let token: u32 = pending.item()?;

        // 2. Push to history; produce incremental text via DecodeStream.
        self.history.push(token);
        let detok = self
            .detok
            .as_mut()
            .expect("pipelined mode invariant: detok is Some");
        let text = detok.step(token)?.unwrap_or_default();

        // 3. Termination check.
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
            // Drop pending_token_arr — no further dispatch on this terminal step.
            self.pending_token_arr = None;
            return Ok(Some(GenerateEvent {
                token,
                text,
                finish_reason,
            }));
        }

        // 4. Dispatch step N+1: build forward graph using the just-materialised
        //    pending Array (still holds its value), sample greedily, async_eval
        //    so the GPU starts immediately.
        let token_arr_in = self
            .pending_token_arr
            .as_ref()
            .expect("invariant")
            .reshape((1_i32, 1_i32))?;
        let pos = (self.history.len() - 1) as i32;
        let position_ids = build_position_ids(pos, 1)?;
        let logits =
            self.model
                .forward_on(&token_arr_in, &position_ids, Some(&mut self.cache), ())?;
        let vocab = logits.shape().as_slice()[2];
        let logits_flat = logits.reshape((vocab,))?;
        let next_arr = self.request.sampler.sample_async_greedy(&logits_flat)?;
        mlx::transforms::async_eval(&[&next_arr])?;

        // 5. Replace pending and return.
        self.pending_token_arr = Some(next_arr);
        Ok(Some(GenerateEvent {
            token,
            text,
            finish_reason: None,
        }))
    }

    /// Synchronous (pre-P8a) decode path. Used when the sampler is
    /// not pipelinable (temperature > 0 or any penalty configured).
    fn next_token_sync(&mut self) -> Result<Option<GenerateEvent>> {
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

    /// Returns `true` iff this stream was constructed with a pipelinable
    /// sampler (greedy + no penalties) and will use the async-eval double-
    /// buffered decode path. Read-only after construction.
    pub fn is_pipelined(&self) -> bool {
        self.pipelined
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

    #[test]
    fn is_pipelined_true_for_greedy_sampler() {
        // GenerationStream::new requires a real Qwen35Model — covered by
        // tests/p4_qwen35_logits_match.rs. Here we verify the upstream
        // predicate (Sampler::is_pipelinable) which GenerationStream::new
        // uses to set the pipelined flag.
        assert!(Sampler::greedy().is_pipelinable());
    }

    #[test]
    fn is_pipelined_false_for_temperature_sampler() {
        assert!(!Sampler::greedy().with_temperature(0.7).is_pipelinable());
    }
}
