//! Single-request generation driver: prefill + decode + sampler + EOS termination.
//!
//! Borrows a concrete [`Model`] implementation and [`Tokenizer`] for the
//! lifetime of the stream; owns the per-call cache vector and accumulating
//! token history.

use std::{sync::OnceLock, time::Instant};

use anyhow::anyhow;
use mlx::Array;
#[cfg(test)]
use mlx::Dtype;

use crate::core::cache::layer::{enable_turboquant_kv_caches, LayerCache};
use crate::core::constrained::{apply_token_mask, ConstraintPlan, ConstraintSession};
use crate::core::model::Model;
#[cfg(test)]
use crate::core::sampler::Sampler;
use crate::core::tokenizer::{DecodeStream, Tokenizer};
use crate::core::vision::DenseVlMethods;
use crate::Result;

/// Process-lifetime gate: only the FIRST `GenerationStream` constructed in
/// the process can claim the Metal capture window. Subsequent constructions
/// see this `OnceLock` already set and skip capture (otherwise stacked
/// captures would error). The lock is sticky for process lifetime — to
/// capture another request, restart the server.
static CAPTURE_CLAIMED: OnceLock<()> = OnceLock::new();

pub use super::generation_types::{GenerateEvent, GenerateRequest};

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
pub struct GenerationStream<'m, M: Model> {
    model: &'m M,
    tokenizer: &'m Tokenizer,
    cache: Vec<LayerCache>,
    /// Pre-computed vision-tower output, populated when the request is VL.
    /// Lives for the duration of prefill; each chunk slices rows from it
    /// keyed by `image_pad_consumed`.
    #[allow(dead_code)]
    vision_embeds_full: Option<Array>,
    /// Pre-computed MRoPE 3-stream position ids `[3, 1, prompt_len]` for
    /// VL requests. Each chunk slices on axis 2 by `[pos .. pos + n]`.
    #[allow(dead_code)]
    position_ids_full: Option<Array>,
    /// Running count of `<|image_pad|>` rows already consumed from
    /// `vision_embeds_full` by previous chunks.
    #[allow(dead_code)]
    image_pad_consumed: usize,
    /// Reusable `[3, 1, 1]` placeholder for models that derive positions
    /// internally instead of consuming caller-built MRoPE position ids.
    dummy_position_ids: Option<Array>,
    /// All token ids so far: prompt ++ generated.
    history: Vec<u32>,
    request: GenerateRequest,
    finished: bool,

    // Mode selector — set once by `new()`, read each `next_token`.
    pipelined: bool,

    // — Pipelined-mode state (Some iff pipelined=true) —
    /// Lazy scalar (shape `[]` or `[1]`) u32 Array — the token next_token()
    /// will emit on its next non-finished call. Always pre-dispatched via
    /// async_eval so the GPU has work to do while we materialise it.
    pending_token_arr: Option<Array>,
    /// Incremental BPE detokenizer; receives one push per emitted token.
    detok: Option<DecodeStream<'m>>,

    // — Synchronous-mode state (populated iff pipelined=false) —
    /// Last full-text snapshot — diffed against the next decode to produce
    /// incremental text. Sync path only.
    last_decoded_text: String,
    vl_profile: bool,

    /// True iff this stream owns the in-flight Metal capture (set when env
    /// var `IRONMLX_CAPTURE_FILE=<path>` was honored at construction time).
    /// Calls `mlx::metal::stop()` in `Drop`.
    capture_active: bool,

    /// When `IRONMLX_CAPTURE_PHASE=decode` is set, capture is deferred until
    /// the first `next_token` call (skipping prefill). This field holds the
    /// path until that first call starts the capture. `None` once started or
    /// if not in decode-only mode.
    capture_pending_decode: Option<String>,

    /// Per-stream PRNG state `[2]` u32. Initialized from `request.sampler.seed`
    /// in `new()`. Advanced by each `Sampler::sample` call. (B1-p2.3e.2)
    prng_state: Array,
    /// Request-local mutable grammar state. Speculative paths clone this state
    /// for proposal verification and only commit emitted tokens.
    constraint: Option<ConstraintSession>,
}

impl<M: Model> Drop for GenerationStream<'_, M> {
    fn drop(&mut self) {
        if self.capture_active {
            // Best-effort stop. Errors are logged but not propagated (we're
            // dropping; the .gputrace file is either complete or partially
            // written — caller can inspect either way).
            if let Err(e) = mlx::metal::stop() {
                tracing::warn!("metal capture stop failed: {e}");
            }
        }
    }
}

/// Honor `IRONMLX_CAPTURE_FILE` + `IRONMLX_CAPTURE_PHASE` env vars.
///
/// - `IRONMLX_CAPTURE_PHASE` unset / "all" / empty (default): start capture
///   immediately at construction (covers prefill + decode). Returns
///   `(capture_active=true, capture_pending_decode=None)`.
/// - `IRONMLX_CAPTURE_PHASE=decode`: defer capture until the first
///   `next_token` call (skips prefill — useful at long PP where prefill GPU
///   work dominates the trace and Xcode replay struggles). Returns
///   `(capture_active=true, capture_pending_decode=Some(path))`.
///
/// Either way, `capture_active=true` means `Drop` calls `stop_capture`.
fn try_start_capture() -> (bool, Option<String>) {
    let Ok(path) = std::env::var("IRONMLX_CAPTURE_FILE") else {
        return (false, None);
    };
    if CAPTURE_CLAIMED.set(()).is_err() {
        tracing::info!(
            "IRONMLX_CAPTURE_FILE set but capture already in progress; \
             this request will not be captured"
        );
        return (false, None);
    }
    let decode_only = std::env::var("IRONMLX_CAPTURE_PHASE").ok().as_deref() == Some("decode");
    if decode_only {
        tracing::info!("metal capture deferred (phase=decode) -> {path}");
        return (true, Some(path));
    }
    match mlx::metal::start(&path) {
        Ok(()) => {
            tracing::info!("metal capture started -> {path}");
            (true, None)
        }
        Err(e) => {
            tracing::warn!(
                "metal capture failed to start ({path}): {e}; continuing without capture \
                 (set MTL_CAPTURE_ENABLED=1 before launch + ensure path is writable)"
            );
            (false, None)
        }
    }
}

pub use super::model_input::*;

fn gemma4_vl_profile_enabled() -> bool {
    std::env::var_os("IRONMLX_GEMMA4_VL_PROFILE").is_some()
}

fn gemma4_vl_pipeline_profile_enabled() -> bool {
    gemma4_vl_profile_enabled() && std::env::var_os("IRONMLX_GEMMA4_VL_PIPELINE_PROFILE").is_some()
}

fn gemma4_vl_pipeline_sync_probe_enabled() -> bool {
    gemma4_vl_pipeline_profile_enabled()
        && std::env::var_os("IRONMLX_GEMMA4_VL_PIPELINE_SYNC_PROBE").is_some()
}

fn log_gemma4_vl_profile_step_ms(label: &str, start: Option<Instant>, step: usize) {
    if let Some(start) = start {
        tracing::info!(
            "[gemma4-vl-profile] {label}_ms={:.3} decode_step={step}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

impl<'m, M: crate::core::Model + DenseVlMethods> GenerationStream<'m, M> {
    pub fn new(model: &'m M, tokenizer: &'m Tokenizer, request: GenerateRequest) -> Result<Self> {
        if request.prompt_ids.is_empty() {
            return Err(anyhow!("GenerationStream::new: prompt_ids cannot be empty"));
        }

        let prompt_len = request.prompt_ids.len();

        // P8a-stage4/6 Metal capture hook. Gated by `IRONMLX_CAPTURE_FILE`
        // env var + first-construction OnceLock. `IRONMLX_CAPTURE_PHASE=decode`
        // defers start to the first `next_token` call (skips prefill).
        let (capture_active, capture_pending_decode) = try_start_capture();

        // B1-p2.3f T4: floor the cap at MIN_KV_CACHE_CAP_FOR_GPU_PERF so
        // short-prompt single-stream decode doesn't fall off the MLX
        // Metal kernel slow path (cap < ~256 → 100-300× decode slowdown
        // on Apple Silicon). Caught by b1_p2_3b_3
        // `admission_window_concurrent_scheduler_and_gs_no_deadlock`
        // standalone regression.
        let cap = ((prompt_len + request.max_new_tokens) as i32)
            .max(crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let dtype = model.cache_dtype();
        let mut cache = model.make_cache(/* batch */ 1, cap, dtype)?;
        if let Some(bits) = request.kv_cache_turboquant_bits {
            enable_turboquant_kv_caches(&mut cache, bits)?;
        }

        let dummy_position_ids = if model.requires_position_ids() {
            None
        } else {
            Some(build_position_ids(0, 1)?)
        };

        // Prefill: chunked when `prefill_chunk_size > 0` and the prompt exceeds
        // it. Intermediate chunks call the text-only forward (cache update,
        // no lm_head); the last chunk goes through the full forward to
        // produce the [1, 1, vocab] last-position logits.
        //
        // Each intermediate chunk closes with `eval(hidden)` — a synchronous
        // wait. The original design used `async_eval` to overlap chunk N's
        // CPU graph build with chunk N-1's GPU work, but that's a trap with
        // KV cache: chunk N+1's graph reads the KV buffers that chunk N just
        // wrote, so its DFS pulls in the still-unscheduled prior writes,
        // ballooning the recorded tape with every chunk. Submission overhead
        // grows quadratically — at chunk_size=512, PP=2048 took 260 s on M1
        // Pro vs 7.3 s for the synchronous variant (35× regression). The
        // sync wait is essentially free here because the next chunk's
        // `forward_on` has nothing to do until the previous chunk's writes
        // land in the cache anyway.

        // P6.7: For VL requests, run the vision tower once before the
        // chunking loop. Models that consume MRoPE position ids also build
        // them for the full prompt so each chunk can slice its own range.
        let (vision_embeds_full, position_ids_full) = if let (Some(pv), Some(grids)) = (
            request.pixel_values.as_deref(),
            request.image_grid_thw.as_deref(),
        ) {
            let ve = model.compute_vision_embeds(pv, grids, ().into())?;
            let pos_full = if dummy_position_ids.is_some() {
                None
            } else if model.vl_positions_sequential() {
                // MiniCPM-V: flat sequential positions over the whole prompt
                // (image tokens included); all three MRoPE streams identical.
                Some(build_position_ids(0, prompt_len as i32)?)
            } else {
                let full_ids_i32: Vec<i32> = request.prompt_ids.iter().map(|&u| u as i32).collect();
                Some(build_position_ids_vl(
                    &full_ids_i32,
                    grids,
                    request.image_token_id,
                    request.image_spatial_merge_size,
                )?)
            };
            (Some(ve), pos_full)
        } else {
            (None, None)
        };

        let chunk_size = request.prefill_chunk_size;
        let prompt_len_i32 = prompt_len as i32;
        let mut pos: i32 = 0;
        let mut image_pad_consumed: usize = 0;
        let last_logits = loop {
            let remaining = prompt_len_i32 - pos;
            let mut n = if chunk_size == 0 {
                remaining
            } else {
                remaining.min(chunk_size as i32)
            };
            if chunk_size != 0 && vision_embeds_full.is_some() {
                let adjusted_end = extend_vl_chunk_end_for_image_pad(
                    &request.prompt_ids,
                    request.image_token_id,
                    pos,
                    pos + n,
                );
                n = adjusted_end - pos;
            }

            let chunk_result: Result<Option<Array>> = (|| -> Result<Option<Array>> {
                let chunk_ids = &request.prompt_ids[pos as usize..(pos as usize + n as usize)];
                let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;

                let chunk_pos_ids = if let Some(dummy) = dummy_position_ids.as_ref() {
                    dummy.clone()
                } else if let Some(pos_full) = position_ids_full.as_ref() {
                    slice_pos_ids_axis2(pos_full, pos, pos + n)?
                } else {
                    build_position_ids(pos, n)?
                };

                let is_vl = vision_embeds_full.is_some();
                let is_last = pos + n == prompt_len_i32;
                let k_i = if is_vl {
                    count_image_pad(chunk_ids, request.image_token_id)
                } else {
                    0
                };
                let image_rows_start = image_pad_consumed;
                let ve_slice = if let Some(ve_full) = vision_embeds_full.as_ref() {
                    if k_i > 0 {
                        let start = image_pad_consumed;
                        let slice = slice_vision_embeds_rows(ve_full, start, start + k_i)?;
                        image_pad_consumed += k_i;
                        Some(slice)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if is_vl {
                    log_vl_chunk_composition(
                        "generate",
                        pos..pos + n,
                        is_last,
                        chunk_ids,
                        request.image_token_id,
                        image_rows_start..image_rows_start + k_i,
                    );
                }

                let logits_or_hidden = if vision_embeds_full.is_some() {
                    if is_last {
                        Some(model.forward_vl_chunk(
                            &chunk_arr,
                            &chunk_pos_ids,
                            None, // per_row_lens
                            None, // decode_mask
                            Some(&mut cache),
                            ve_slice.as_ref(),
                            request.image_token_id,
                            ().into(),
                        )?)
                    } else {
                        let hidden = model.forward_vl_hidden(
                            &chunk_arr,
                            &chunk_pos_ids,
                            None, // per_row_lens
                            None, // decode_mask
                            Some(&mut cache),
                            ve_slice.as_ref(),
                            request.image_token_id,
                            ().into(),
                        )?;
                        mlx::transforms::eval(&[&hidden])?;
                        None
                    }
                } else if is_last {
                    Some(model.forward_on(
                        &chunk_arr,
                        &chunk_pos_ids,
                        None, // per_row_lens
                        None, // decode_mask
                        Some(&mut cache),
                        ().into(),
                    )?)
                } else {
                    let hidden = model.forward_text_hidden(
                        &chunk_arr,
                        &chunk_pos_ids,
                        None, // per_row_lens
                        None, // decode_mask
                        Some(&mut cache),
                        ().into(),
                    )?;
                    mlx::transforms::eval(&[&hidden])?;
                    None
                };
                Ok(logits_or_hidden)
            })();

            if let Some(logits) = chunk_result? {
                let vocab = logits.shape().as_slice()[2];
                break logits.reshape((vocab,))?;
            }
            pos += n;
        };

        // After the loop, every image_pad must have been consumed by some
        // chunk. If this fails, the chunked path is dropping data.
        if let Some(ve_full) = vision_embeds_full.as_ref() {
            let expected = ve_full.shape().as_slice()[0] as usize;
            if image_pad_consumed != expected {
                return Err(anyhow!(
                    "P6.7 chunked prefill: consumed {} image_pad rows, expected {}",
                    image_pad_consumed,
                    expected,
                ));
            }
        }

        let history = request.prompt_ids.clone();
        let pipelined = request.sampler.is_pipelinable();
        let vl_profile = gemma4_vl_profile_enabled();
        let mut constraint = request
            .constraint
            .as_ref()
            .map(ConstraintPlan::start_session)
            .transpose()?;

        // Initialize per-stream PRNG state from sampler seed. [2] u32.
        let mut prng_state = mlx::random::key(request.sampler.seed)?;

        if pipelined {
            // Pipelined path: pending_token_arr starts as the prefill's argmax,
            // pre-dispatched via async_eval so the GPU is already working on
            // it by the time the first next_token() call materialises it.
            let pending = {
                let constrained_logits = constrain_logits(&mut constraint, &last_logits)?;
                let pending = request.sampler.sample_async_greedy(&constrained_logits)?;
                mlx::transforms::async_eval(&[&pending])?;
                pending
            };
            let detok = tokenizer.decode_stream(/* skip_special */ true);

            Ok(Self {
                model,
                tokenizer,
                cache,
                vision_embeds_full: None,
                position_ids_full: None,
                image_pad_consumed: 0,
                dummy_position_ids,
                history,
                request,
                finished: false,
                pipelined: true,
                pending_token_arr: Some(pending),
                detok: Some(detok),
                last_decoded_text: String::new(),
                vl_profile,
                capture_active,
                capture_pending_decode,
                prng_state,
                constraint,
            })
        } else {
            // Sync path: existing pre-P8a behavior. First token sampled
            // synchronously here; pushed into history; initial text snapshot
            // captured for incremental diff.
            let constrained_logits = constrain_logits(&mut constraint, &last_logits)?;
            let first_token =
                request
                    .sampler
                    .sample(&constrained_logits, &history, &mut prng_state)?;
            commit_constraint_token(&mut constraint, first_token)?;
            let mut history = history;
            history.push(first_token);

            let initial_text = tokenizer
                .decode(&history, /* skip_special = */ true)
                .unwrap_or_default();

            Ok(Self {
                model,
                tokenizer,
                cache,
                vision_embeds_full: None,
                position_ids_full: None,
                image_pad_consumed: 0,
                dummy_position_ids,
                history,
                request,
                finished: false,
                pipelined: false,
                pending_token_arr: None,
                detok: None,
                last_decoded_text: initial_text,
                vl_profile,
                capture_active,
                capture_pending_decode,
                prng_state,
                constraint,
            })
        }
    }
}

// Non-VL methods (works for any Model) — decode path and helpers that only
// call model.forward_on / model.forward_text_hidden (both Model trait methods).
impl<'m, M: crate::core::Model> GenerationStream<'m, M> {
    /// Text-only constructor: works for any `M: Model`.
    ///
    /// Asserts `request.pixel_values.is_none()` — returns `Err` if called with
    /// image inputs. For VL requests use `GenerationStream::new` instead.
    pub fn new_text_only(
        model: &'m M,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
    ) -> Result<Self> {
        if request.pixel_values.is_some() {
            return Err(anyhow!(
                "GenerationStream::new_text_only called with pixel_values; use new() for VL requests"
            ));
        }
        if request.prompt_ids.is_empty() {
            return Err(anyhow!(
                "GenerationStream::new_text_only: prompt_ids cannot be empty"
            ));
        }

        let prompt_len = request.prompt_ids.len();
        let (capture_active, capture_pending_decode) = try_start_capture();

        let cap = ((prompt_len + request.max_new_tokens) as i32)
            .max(crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let dtype = model.cache_dtype();
        let mut cache = model.make_cache(/* batch */ 1, cap, dtype)?;
        if let Some(bits) = request.kv_cache_turboquant_bits {
            enable_turboquant_kv_caches(&mut cache, bits)?;
        }
        let dummy_position_ids = if model.requires_position_ids() {
            None
        } else {
            Some(build_position_ids(0, 1)?)
        };

        let chunk_size = request.prefill_chunk_size;
        let prompt_len_i32 = prompt_len as i32;
        let mut pos: i32 = 0;
        let last_logits = loop {
            let remaining = prompt_len_i32 - pos;
            let n = if chunk_size == 0 {
                remaining
            } else {
                remaining.min(chunk_size as i32)
            };
            let chunk_ids = &request.prompt_ids[pos as usize..(pos as usize + n as usize)];
            let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;
            let chunk_pos_ids = if let Some(dummy) = dummy_position_ids.as_ref() {
                dummy.clone()
            } else {
                build_position_ids(pos, n)?
            };

            let is_last = pos + n == prompt_len_i32;
            let logits_or_hidden = if is_last {
                Some(model.forward_on(
                    &chunk_arr,
                    &chunk_pos_ids,
                    None, // per_row_lens
                    None, // decode_mask
                    Some(&mut cache),
                    ().into(),
                )?)
            } else {
                let hidden = model.forward_text_hidden(
                    &chunk_arr,
                    &chunk_pos_ids,
                    None, // per_row_lens
                    None, // decode_mask
                    Some(&mut cache),
                    ().into(),
                )?;
                mlx::transforms::eval(&[&hidden])?;
                None
            };

            if let Some(logits) = logits_or_hidden {
                let vocab = logits.shape().as_slice()[2];
                break logits.reshape((vocab,))?;
            }
            pos += n;
        };

        let history = request.prompt_ids.clone();
        let pipelined = request.sampler.is_pipelinable();
        let vl_profile = gemma4_vl_profile_enabled();
        let mut prng_state = mlx::random::key(request.sampler.seed)?;
        let mut constraint = request
            .constraint
            .as_ref()
            .map(ConstraintPlan::start_session)
            .transpose()?;

        if pipelined {
            let constrained_logits = constrain_logits(&mut constraint, &last_logits)?;
            let pending = request.sampler.sample_async_greedy(&constrained_logits)?;
            mlx::transforms::async_eval(&[&pending])?;
            let detok = tokenizer.decode_stream(/* skip_special */ true);
            Ok(Self {
                model,
                tokenizer,
                cache,
                vision_embeds_full: None,
                position_ids_full: None,
                image_pad_consumed: 0,
                dummy_position_ids,
                history,
                request,
                finished: false,
                pipelined: true,
                pending_token_arr: Some(pending),
                detok: Some(detok),
                last_decoded_text: String::new(),
                vl_profile,
                capture_active,
                capture_pending_decode,
                prng_state,
                constraint,
            })
        } else {
            let constrained_logits = constrain_logits(&mut constraint, &last_logits)?;
            let first_token =
                request
                    .sampler
                    .sample(&constrained_logits, &history, &mut prng_state)?;
            commit_constraint_token(&mut constraint, first_token)?;
            let mut history = history;
            history.push(first_token);
            let initial_text = tokenizer
                .decode(&history, /* skip_special = */ true)
                .unwrap_or_default();
            Ok(Self {
                model,
                tokenizer,
                cache,
                vision_embeds_full: None,
                position_ids_full: None,
                image_pad_consumed: 0,
                dummy_position_ids,
                history,
                request,
                finished: false,
                pipelined: false,
                pending_token_arr: None,
                detok: None,
                last_decoded_text: initial_text,
                vl_profile,
                capture_active,
                capture_pending_decode,
                prng_state,
                constraint,
            })
        }
    }

    /// If a capture was deferred (phase=decode), start it now (lazily, on
    /// first `next_token` call). Idempotent — once started, the pending
    /// path is cleared.
    fn start_deferred_capture(&mut self) {
        if let Some(path) = self.capture_pending_decode.take() {
            match mlx::metal::start(&path) {
                Ok(()) => tracing::info!("metal capture started (decode phase) -> {path}"),
                Err(e) => tracing::warn!(
                    "metal capture failed to start ({path}): {e}; continuing without capture"
                ),
            }
        }
    }

    fn decode_position_ids(&self, pos: i32) -> Result<Array> {
        match self.dummy_position_ids.as_ref() {
            Some(dummy) => Ok(dummy.clone()),
            None => build_position_ids(pos, 1),
        }
    }

    /// Pull the next event. Returns `Ok(None)` after the stream terminates.
    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        self.next_token_inner()
    }

    fn next_token_inner(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }
        let profile_start = self.vl_profile.then(Instant::now);
        let decode_step = self
            .history
            .len()
            .saturating_sub(self.request.prompt_ids.len())
            + if self.pipelined { 1 } else { 0 };
        // If decode-phase Metal capture was deferred, start it now (right
        // before the first decode-step work hits the GPU).
        self.start_deferred_capture();
        let event = if self.pipelined {
            self.next_token_pipelined()
        } else {
            self.next_token_sync()
        };
        if matches!(&event, Ok(Some(_))) {
            log_gemma4_vl_profile_step_ms("decode_step_total", profile_start, decode_step);
        }
        event
    }

    /// Pipelined hot path. Invariant: `self.pending_token_arr` is `Some` and
    /// the lazy scalar (shape `[]` or `[1]`) u32 Array of the token to be
    /// returned on this call.
    fn next_token_pipelined(&mut self) -> Result<Option<GenerateEvent>> {
        let profile_enabled = self.vl_profile;
        let pipeline_profile = profile_enabled && gemma4_vl_pipeline_profile_enabled();
        let sync_probe = pipeline_profile && gemma4_vl_pipeline_sync_probe_enabled();
        let decode_step = self.history.len() - self.request.prompt_ids.len() + 1;

        // 1. Materialise the pending token. The GPU has been working on it
        //    since the previous next_token call's async_eval (or new()).
        let wait_start = profile_enabled.then(Instant::now);
        let pending = self
            .pending_token_arr
            .as_ref()
            .expect("pipelined mode invariant: pending_token_arr is Some");
        let token: u32 = pending.item()?;
        commit_constraint_token(&mut self.constraint, token)?;
        log_gemma4_vl_profile_step_ms("decode_pipelined_pending_item", wait_start, decode_step);

        // 2. Push to history; produce incremental text via DecodeStream.
        let detok_start = profile_enabled.then(Instant::now);
        self.history.push(token);
        let detok = self
            .detok
            .as_mut()
            .expect("pipelined mode invariant: detok is Some");
        let text = detok.step(token)?.unwrap_or_default();
        log_gemma4_vl_profile_step_ms("decode_detok", detok_start, decode_step);

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
        let dispatch_start = profile_enabled.then(Instant::now);
        let t0 = pipeline_profile.then(Instant::now);
        let token_arr_in = self
            .pending_token_arr
            .as_ref()
            .expect("pipelined mode invariant: pending_token_arr is Some")
            .reshape((1_i32, 1_i32))?;
        log_gemma4_vl_profile_step_ms("decode_pipeline_token_arr_reshape", t0, decode_step);

        let pos = (self.history.len() - 1) as i32;
        let t0 = pipeline_profile.then(Instant::now);
        let position_ids = self.decode_position_ids(pos)?;
        log_gemma4_vl_profile_step_ms("decode_pipeline_position_ids", t0, decode_step);

        let t0 = pipeline_profile.then(Instant::now);
        let logits = self.model.forward_on(
            &token_arr_in,
            &position_ids,
            None, // per_row_lens
            None, // decode_mask
            Some(&mut self.cache),
            ().into(),
        )?;
        log_gemma4_vl_profile_step_ms("decode_pipeline_forward_graph", t0, decode_step);

        let t0 = pipeline_profile.then(Instant::now);
        let vocab = logits.shape().as_slice()[2];
        let logits_flat = logits.reshape((vocab,))?;
        log_gemma4_vl_profile_step_ms("decode_pipeline_logits_reshape", t0, decode_step);

        let t0 = pipeline_profile.then(Instant::now);
        let constrained_logits = constrain_logits(&mut self.constraint, &logits_flat)?;
        let next_arr = self
            .request
            .sampler
            .sample_async_greedy(&constrained_logits)?;
        log_gemma4_vl_profile_step_ms("decode_pipeline_sample_graph", t0, decode_step);

        let t0 = pipeline_profile.then(Instant::now);
        mlx::transforms::async_eval(&[&next_arr])?;
        log_gemma4_vl_profile_step_ms("decode_pipeline_async_eval", t0, decode_step);

        if sync_probe {
            let t0 = Some(Instant::now());
            mlx::transforms::eval(&[&next_arr])?;
            log_gemma4_vl_profile_step_ms("decode_pipeline_sync_probe_eval", t0, decode_step);
        }

        log_gemma4_vl_profile_step_ms("decode_pipelined_dispatch", dispatch_start, decode_step);

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
        let profile_enabled = self.vl_profile;
        let decode_step = self.history.len() - self.request.prompt_ids.len();

        // The token to emit is the most-recent push to history.
        let token = *self.history.last().expect("history non-empty post-new");

        // Compute incremental text via cumulative-detok diff.
        let detok_start = profile_enabled.then(Instant::now);
        let full_text = self
            .tokenizer
            .decode(&self.history, /* skip_special = */ true)
            .unwrap_or_default();
        let text = full_text
            .strip_prefix(&self.last_decoded_text)
            .unwrap_or(&full_text)
            .to_string();
        self.last_decoded_text = full_text;
        log_gemma4_vl_profile_step_ms("decode_detok", detok_start, decode_step);

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
        let forward_sample_start = profile_enabled.then(Instant::now);
        let token_arr: Array = (&[token][..], &[1_i32, 1][..]).try_into()?;
        let pos = (self.history.len() - 1) as i32;
        let position_ids = self.decode_position_ids(pos)?;
        let logits = self.model.forward_on(
            &token_arr,
            &position_ids,
            None, // per_row_lens
            None, // decode_mask
            Some(&mut self.cache),
            ().into(),
        )?;
        // Logits shape [1, 1, vocab] — flatten to [vocab].
        let vocab = logits.shape().as_slice()[2];
        let logits_flat = logits.reshape((vocab,))?;
        let constrained_logits = constrain_logits(&mut self.constraint, &logits_flat)?;
        let next = self.request.sampler.sample(
            &constrained_logits,
            &self.history,
            &mut self.prng_state,
        )?;
        commit_constraint_token(&mut self.constraint, next)?;
        self.history.push(next);
        log_gemma4_vl_profile_step_ms(
            "decode_sync_forward_sample",
            forward_sample_start,
            decode_step,
        );

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

    /// Returns all token ids accumulated so far: prompt tokens plus generated
    /// tokens.
    ///
    /// **Pipelined-mode note**: between construction and the first
    /// `next_token()` call, `history` does not yet contain the first
    /// generated token (it's a lazy Array waiting for `.item()`). In sync
    /// mode the first generated token is already pushed at construction.
    /// After N successful `next_token()` calls, both modes hold exactly N
    /// generated tokens beyond the prompt — callers inspecting history
    /// only after iteration are unaffected by the asymmetry.
    pub fn history(&self) -> &[u32] {
        &self.history
    }
}

fn constrain_logits(constraint: &mut Option<ConstraintSession>, logits: &Array) -> Result<Array> {
    match constraint {
        Some(session) => apply_token_mask(logits, &session.compute_mask()?),
        None => Ok(logits.clone()),
    }
}

fn commit_constraint_token(constraint: &mut Option<ConstraintSession>, token: u32) -> Result<()> {
    if let Some(session) = constraint {
        session.commit_token(token)?;
    }
    Ok(())
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

    /// Contract test: `build_position_ids(0, S)` produces flat sequential
    /// positions `[0, 1, …, S-1]` across all 3 MRoPE streams — the invariant that
    /// `GenerationStream`'s `vl_positions_sequential` branch relies on.
    #[test]
    fn build_position_ids_is_flat_sequential_three_streams() {
        let p = build_position_ids(0, 4).unwrap();
        assert_eq!(p.shape().as_slice(), &[3, 1, 4]);
        let v: Vec<i32> = p.to_vec().unwrap();
        assert_eq!(&v[0..4], &[0, 1, 2, 3]);
        assert_eq!(&v[4..8], &[0, 1, 2, 3]);
        assert_eq!(&v[8..12], &[0, 1, 2, 3]);
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
        // tests/qwen35_logits_match.rs. Here we verify the upstream
        // predicate (Sampler::is_pipelinable) which GenerationStream::new
        // uses to set the pipelined flag.
        assert!(Sampler::greedy().is_pipelinable());
    }

    #[test]
    fn is_pipelined_false_for_temperature_sampler() {
        assert!(!Sampler::greedy().with_temperature(0.7).is_pipelinable());
    }

    /// Verify that GenerateRequest can be constructed with the new optional VL
    /// fields set to None (text-only regression — field presence check).
    #[test]
    fn generate_request_pixel_values_none_construction() {
        let req = GenerateRequest {
            prompt_ids: vec![1_u32, 2, 3],
            max_new_tokens: 10,
            sampler: Sampler::greedy(),
            stop_token_ids: vec![2_u32],
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: IMAGE_TOKEN_ID,
            constraint: None,
        };
        assert!(req.pixel_values.is_none());
        assert!(req.image_grid_thw.is_none());
        assert_eq!(req.prompt_ids.len(), 3);
    }

    #[test]
    fn position_ids_vl_single_stream_two_images_preserves_order() {
        let image_token_id = 258880_i32;
        let merge_size = 3_i32;
        let input_ids: Vec<i32> = vec![
            10,
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            11,
            image_token_id,
            12,
        ];
        let grids = vec![(1_i32, 6_i32, 6_i32), (1_i32, 3_i32, 3_i32)];

        let pos = build_position_ids_vl(&input_ids, &grids, image_token_id, merge_size)
            .expect("position ids");
        assert_eq!(pos.shape().as_slice(), &[3_i32, 1_i32, 8_i32]);
        let flat: Vec<i32> = pos.to_vec().expect("to_vec");
        assert_eq!(&flat[0..8], &[0, 1, 1, 1, 1, 3, 4, 5]);
        assert_eq!(&flat[8..16], &[0, 1, 1, 2, 2, 3, 4, 5]);
        assert_eq!(&flat[16..24], &[0, 1, 2, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn position_ids_vl_batched_b2_each_one_image_matches_per_stream() {
        // Two VL rows, different prompts, both contain exactly one
        // image. Verify [3, B, max_len] matches per-row [3, 1, L_i]
        // sliced+padded.
        let image_token_id = 248056_i32;
        let merge_size = 2_i32;

        // Row 0 prompt: 4 text + 4 image_pad + 2 text = 10 tokens, grid (1,4,4) → 4 pads after spatial_merge_size
        let row0_ids: Vec<i32> = vec![
            100,
            101,
            102,
            103,
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            200,
            201,
        ];
        let row0_grids: Vec<(i32, i32, i32)> = vec![(1, 4, 4)];

        // Row 1 prompt: 2 text + 1 image_pad + 3 text = 6 tokens, grid (1,2,2) → 1 pad
        let row1_ids: Vec<i32> = vec![300, 301, image_token_id, 400, 401, 402];
        let row1_grids: Vec<(i32, i32, i32)> = vec![(1, 2, 2)];

        let max_len = row0_ids.len().max(row1_ids.len()) as i32;

        // Per-row reference build via existing single-stream API
        let row0_pos = build_position_ids_vl(&row0_ids, &row0_grids, image_token_id, merge_size)
            .expect("row0 single-stream");
        let row1_pos = build_position_ids_vl(&row1_ids, &row1_grids, image_token_id, merge_size)
            .expect("row1 single-stream");

        // Batched build under test
        let per_row_prompt_ids: Vec<&[i32]> = vec![&row0_ids[..], &row1_ids[..]];
        let per_row_grid_thw: Vec<Option<&[(i32, i32, i32)]>> =
            vec![Some(&row0_grids[..]), Some(&row1_grids[..])];
        let batched = build_position_ids_vl_batched(
            &per_row_prompt_ids,
            &per_row_grid_thw,
            image_token_id,
            merge_size,
            max_len,
        )
        .expect("batched build");

        // Expected shape [3, 2, max_len]
        let shape = batched.shape();
        let dims = shape.as_slice();
        assert_eq!(dims, &[3_i32, 2_i32, max_len], "batched shape");

        // Read both as Vec<i32>
        let batched_flat: Vec<i32> = batched.to_vec().expect("batched to_vec");
        let row0_flat: Vec<i32> = row0_pos.to_vec().expect("row0 to_vec");
        let row1_flat: Vec<i32> = row1_pos.to_vec().expect("row1 to_vec");
        // row0_pos shape = [3, 1, L_0] flat layout [stream0 | stream1 | stream2]
        // batched_flat layout for [3, B, max_len]: stream s at offset s*B*max_len + b*max_len + col
        let b_len = max_len as usize;
        let len0 = row0_ids.len();
        let len1 = row1_ids.len();
        for s in 0..3 {
            for col in 0..len0 {
                let bat = batched_flat[s * 2 * b_len + 0 * b_len + col];
                let ref_v = row0_flat[s * len0 + col];
                assert_eq!(bat, ref_v, "row 0 stream {s} col {col}");
            }
            for col in len0..b_len {
                let bat = batched_flat[s * 2 * b_len + 0 * b_len + col];
                assert_eq!(bat, 0, "row 0 pad stream {s} col {col}");
            }
            for col in 0..len1 {
                let bat = batched_flat[s * 2 * b_len + 1 * b_len + col];
                let ref_v = row1_flat[s * len1 + col];
                assert_eq!(bat, ref_v, "row 1 stream {s} col {col}");
            }
            for col in len1..b_len {
                let bat = batched_flat[s * 2 * b_len + 1 * b_len + col];
                assert_eq!(bat, 0, "row 1 pad stream {s} col {col}");
            }
        }
    }

    #[test]
    fn position_ids_vl_batched_mixed_text_vl_matches_per_stream() {
        // B=2: row 0 text-only (no images), row 1 VL with 1 image.
        // Verify text row uses degraded MRoPE (0..L_i triple-replicated),
        // VL row matches single-stream build_position_ids_vl.
        let image_token_id = 248056_i32;
        let merge_size = 2_i32;

        // Row 0: 4 text tokens (no image)
        let row0_ids: Vec<i32> = vec![10, 11, 12, 13];
        // Row 1: 2 text + 4 image_pad = 6 tokens, grid (1, 4, 4) → 4 pads
        let row1_ids: Vec<i32> = vec![
            20,
            21,
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
        ];
        let row1_grids: Vec<(i32, i32, i32)> = vec![(1, 4, 4)];
        let max_len = row1_ids.len() as i32; // 6

        // Reference: text row degraded; VL row via build_position_ids_vl
        let row1_pos = build_position_ids_vl(&row1_ids, &row1_grids, image_token_id, merge_size)
            .expect("row1 single-stream");
        let row1_flat: Vec<i32> = row1_pos.to_vec().expect("row1 to_vec");
        let l1 = row1_ids.len();

        let per_row_prompt_ids: Vec<&[i32]> = vec![&row0_ids[..], &row1_ids[..]];
        let per_row_grid_thw: Vec<Option<&[(i32, i32, i32)]>> = vec![None, Some(&row1_grids[..])];
        let batched = build_position_ids_vl_batched(
            &per_row_prompt_ids,
            &per_row_grid_thw,
            image_token_id,
            merge_size,
            max_len,
        )
        .expect("batched build");

        let shape = batched.shape();
        assert_eq!(shape.as_slice(), &[3_i32, 2_i32, max_len]);

        let batched_flat: Vec<i32> = batched.to_vec().expect("batched to_vec");
        let m = max_len as usize;

        // Row 0 (text-only, 4 real + 2 pad): all three streams = 0..L_0 then pad zeros
        let l0 = row0_ids.len();
        for s in 0..3 {
            for col in 0..l0 {
                assert_eq!(
                    batched_flat[s * 2 * m + 0 * m + col],
                    col as i32,
                    "row 0 (text) stream {s} col {col}"
                );
            }
            for col in l0..m {
                assert_eq!(
                    batched_flat[s * 2 * m + 0 * m + col],
                    0,
                    "row 0 pad stream {s} col {col}"
                );
            }
        }

        // Row 1 (VL): matches per-row build for real columns, pad = 0
        // (here L_1 == max_len so no pad columns)
        for s in 0..3 {
            for col in 0..l1 {
                assert_eq!(
                    batched_flat[s * 2 * m + 1 * m + col],
                    row1_flat[s * l1 + col],
                    "row 1 (VL) stream {s} col {col}"
                );
            }
        }
    }
}

#[cfg(test)]
mod p6_7_helper_tests {
    use super::*;

    #[test]
    fn count_image_pad_basic() {
        let ids: Vec<u32> = vec![1, 248056, 2, 248056, 248056, 3];
        assert_eq!(count_image_pad(&ids, 248056), 3);
        assert_eq!(count_image_pad(&ids, 999), 0);
    }

    #[test]
    fn extend_vl_chunk_end_extends_inside_image_run() {
        let ids: Vec<u32> = (0..400_u32)
            .map(|i| if (250..260).contains(&i) { 42 } else { 1 })
            .collect();
        assert_eq!(extend_vl_chunk_end_for_image_pad(&ids, 42, 0, 256), 260);
    }

    #[test]
    fn extend_vl_chunk_end_absorbs_short_final_text_tail_after_image_run() {
        let ids: Vec<u32> = (0..275_u32)
            .map(|i| if (250..260).contains(&i) { 42 } else { 1 })
            .collect();

        assert_eq!(extend_vl_chunk_end_for_image_pad(&ids, 42, 0, 256), 275);
    }

    #[test]
    fn extend_vl_chunk_end_keeps_tail_with_image_tokens() {
        let ids: Vec<u32> = (0..275_u32)
            .map(|i| {
                if (250..260).contains(&i) || (270..273).contains(&i) {
                    42
                } else {
                    1
                }
            })
            .collect();

        assert_eq!(extend_vl_chunk_end_for_image_pad(&ids, 42, 0, 256), 260);
    }

    #[test]
    fn extend_vl_chunk_end_keeps_exact_run_boundary() {
        let ids: Vec<u32> = (0..400_u32)
            .map(|i| if (200..256).contains(&i) { 42 } else { 1 })
            .collect();
        assert_eq!(extend_vl_chunk_end_for_image_pad(&ids, 42, 0, 256), 256);
    }

    #[test]
    fn extend_vl_chunk_end_absorbs_short_text_tail_after_exact_boundary() {
        let ids: Vec<u32> = (0..275_u32)
            .map(|i| if (200..256).contains(&i) { 42 } else { 1 })
            .collect();

        assert_eq!(extend_vl_chunk_end_for_image_pad(&ids, 42, 0, 256), 275);
    }

    #[test]
    fn extend_vl_chunk_end_keeps_non_image_boundary() {
        let ids: Vec<u32> = (0..400_u32)
            .map(|i| if (300..340).contains(&i) { 42 } else { 1 })
            .collect();
        assert_eq!(extend_vl_chunk_end_for_image_pad(&ids, 42, 0, 256), 256);
        assert_eq!(extend_vl_chunk_end_for_image_pad(&ids, -1, 0, 256), 256);
    }

    #[test]
    fn slice_pos_ids_axis2_basic() {
        let data: Vec<i32> = (0..15).collect();
        let pos: mlx::Array = (&data[..], &[3_i32, 1, 5][..]).try_into().expect("pos arr");
        let sliced = slice_pos_ids_axis2(&pos, 1, 4).expect("slice");
        assert_eq!(sliced.shape().as_slice(), &[3, 1, 3]);
        let flat: Vec<i32> = sliced.to_vec::<i32>().expect("to_vec");
        assert_eq!(flat, vec![1, 2, 3, 6, 7, 8, 11, 12, 13]);
    }

    #[test]
    fn slice_pos_ids_axis2_rejects_bad_shape() {
        let data: Vec<i32> = vec![0; 6];
        let bad: mlx::Array = (&data[..], &[2_i32, 1, 3][..]).try_into().expect("bad");
        let err = slice_pos_ids_axis2(&bad, 0, 2).expect_err("must err on [2,1,S]");
        assert!(format!("{err}").contains("expected [3,1,S]"));
    }

    #[test]
    fn slice_vision_embeds_rows_basic() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let ve: mlx::Array = (&data[..], &[4_i32, 3][..]).try_into().expect("ve arr");
        let sliced = slice_vision_embeds_rows(&ve, 1, 3).expect("slice");
        assert_eq!(sliced.shape().as_slice(), &[2, 3]);
        let flat: Vec<f32> = sliced.to_vec::<f32>().expect("to_vec");
        assert_eq!(flat, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
}

#[cfg(test)]
mod b1_p2_1_position_id_tests {
    use super::*;

    #[test]
    fn build_position_ids_batched_same_length() {
        // B=2, both length 4, max_len=4 → no padding.
        let arr = build_position_ids_batched(&[4, 4], 4).expect("build");
        assert_eq!(arr.shape().as_slice(), &[3, 2, 4]);
        let flat: Vec<i32> = arr.to_vec::<i32>().expect("to_vec");
        // All 3 streams identical; each row is [0, 1, 2, 3].
        let expected: Vec<i32> = (0..3).flat_map(|_| (0..2).flat_map(|_| 0..4_i32)).collect();
        assert_eq!(flat, expected);
    }

    #[test]
    fn build_position_ids_batched_right_padded() {
        // B=2, lens [3, 5], max_len=5 (right-padded).
        // Row 0: real positions 0,1,2 at indices 0,1,2; pad (zero) at indices 3,4.
        // Row 1: full sequence 0..4 at indices 0..4.
        let arr = build_position_ids_batched(&[3, 5], 5).expect("build");
        assert_eq!(arr.shape().as_slice(), &[3, 2, 5]);
        let flat: Vec<i32> = arr.to_vec::<i32>().expect("to_vec");
        // Single stream: [0,1,2,0,0,  0,1,2,3,4]; replicated 3x along axis 0.
        let one_stream: Vec<i32> = vec![0, 1, 2, 0, 0, 0, 1, 2, 3, 4];
        let mut expected = Vec::with_capacity(30);
        for _ in 0..3 {
            expected.extend_from_slice(&one_stream);
        }
        assert_eq!(flat, expected);
    }
}

#[cfg(test)]
mod b1_p2_1_mask_tests {
    use super::*;

    #[test]
    fn build_batch_attention_mask_causal_no_padding() {
        // B=1, length=3, max_len=3 → standard lower-triangular causal.
        let mask = build_batch_attention_mask(&[3], 3, Dtype::Float32).expect("mask");
        assert_eq!(mask.shape().as_slice(), &[1, 1, 3, 3]);
        let flat: Vec<f32> = mask.to_vec::<f32>().expect("to_vec");
        let ni = f32::NEG_INFINITY;
        let expected = vec![0.0, ni, ni, 0.0, 0.0, ni, 0.0, 0.0, 0.0];
        assert_eq!(flat, expected);
    }

    #[test]
    fn build_batch_attention_mask_right_padded() {
        // B=2, lens [2, 3], max_len=3 (right-padded).
        let mask = build_batch_attention_mask(&[2, 3], 3, Dtype::Float32).expect("mask");
        assert_eq!(mask.shape().as_slice(), &[2, 1, 3, 3]);
        let flat: Vec<f32> = mask.to_vec::<f32>().expect("to_vec");
        let ni = f32::NEG_INFINITY;
        // Row 0 (i=0, L=2): real at columns 0,1; pad at column 2.
        //   q=0 is real → k=0 allowed.
        //   q=1 is real → k=0,1 allowed.
        //   q=2 is pad → self-attend only (mask[2,2]=0).
        // Row 1 (i=1, L=3, no pad): standard causal lower-triangle.
        let expected = vec![
            // Row 0
            0.0, ni, ni, // q=0 (real): k=0 allowed
            0.0, 0.0, ni, // q=1 (real): k=0,1 allowed
            ni, ni, 0.0, // q=2 (pad): self-attend only
            // Row 1 (standard causal)
            0.0, ni, ni, 0.0, 0.0, ni, 0.0, 0.0, 0.0,
        ];
        assert_eq!(flat, expected);
    }

    #[test]
    fn build_batched_append_mask_handles_complementary_offsets_and_lens() {
        // Both rows end at K=8, but row 0 appends one token after a longer
        // history while row 1 appends five. Treating this as an unmasked
        // uniform append would incorrectly build K=12 from max(pre)+Q.
        let mask =
            build_batched_append_attention_mask(&[7, 3], &[1, 5], 5, Dtype::Float32).expect("mask");
        assert_eq!(mask.shape().as_slice(), &[2, 1, 5, 8]);
        let flat: Vec<f32> = mask.to_vec().expect("to_vec");
        let row_stride = 5 * 8;

        // Row 0's only real query sees all eight valid keys. Its padded
        // queries remain finite without extending the cache.
        assert!(flat[..row_stride].iter().all(|&value| value == 0.0));

        // Row 1's real queries reveal one new key at a time after K=3.
        for q in 0..5 {
            let query = &flat[row_stride + q * 8..row_stride + (q + 1) * 8];
            let visible = 3 + q + 1;
            assert!(query[..visible].iter().all(|&value| value == 0.0));
            assert!(query[visible..]
                .iter()
                .all(|value| value.is_infinite() && value.is_sign_negative()));
        }
    }
}

#[cfg(test)]
mod b1_p2_2_decode_position_id_tests {
    use super::*;

    #[test]
    fn build_decode_position_ids_basic() {
        // B=2 with distinct positions.
        let arr = build_decode_position_ids(&[10, 20]).expect("build");
        assert_eq!(arr.shape().as_slice(), &[3, 2, 1]);
        let flat: Vec<i32> = arr.to_vec::<i32>().expect("to_vec");
        // All 3 streams identical: [10, 20] repeated 3 times.
        assert_eq!(flat, vec![10, 20, 10, 20, 10, 20]);
    }

    #[test]
    fn build_decode_position_ids_rejects_empty() {
        let err = build_decode_position_ids(&[]).expect_err("must err on empty");
        assert!(format!("{err}").contains("per_row_pos must be non-empty"));
    }

    #[test]
    fn build_batch_linear_mask_same_length() {
        // B=2, lens [4, 4], max_len=4 → all true (no padding).
        let mask = build_batch_linear_mask(&[4, 4], 4).expect("build");
        assert_eq!(mask.shape().as_slice(), &[2, 4]);
        let flat: Vec<bool> = mask.to_vec::<bool>().expect("to_vec");
        assert_eq!(flat, vec![true; 8]);
    }

    #[test]
    fn build_batch_linear_mask_right_padded() {
        // B=2, lens [2, 4], max_len=4 (right-padded).
        // Row 0: L=2 → [true, true, false, false]
        // Row 1: L=4 (no pad) → [true, true, true, true]
        let mask = build_batch_linear_mask(&[2, 4], 4).expect("build");
        assert_eq!(mask.shape().as_slice(), &[2, 4]);
        let flat: Vec<bool> = mask.to_vec::<bool>().expect("to_vec");
        assert_eq!(
            flat,
            vec![
                true, true, false, false, // row 0
                true, true, true, true, // row 1
            ]
        );
    }
}

#[cfg(test)]
mod per_row_decode_mask_tests {
    use super::*;
    use mlx::Dtype;

    #[test]
    fn mask_per_row_decode_uniform_lens() {
        // B=2, both rows have real_len = 4, max_len = 4.
        // Expected: all zeros (every column is valid).
        let m = build_per_row_decode_mask(&[4, 4], 4, Dtype::Float32).expect("mask");
        assert_eq!(m.shape().as_slice(), &[2, 1, 1, 4]);
        let v: Vec<f32> = m.to_vec().expect("read mask");
        for x in &v {
            assert_eq!(*x, 0.0_f32, "uniform-lens mask must be all zeros");
        }
    }

    #[test]
    fn mask_per_row_decode_ragged() {
        // B=2, real_lens = [2, 5], max_len = 5.
        // Row 0: positions 0,1 = 0; positions 2,3,4 = -inf.
        // Row 1: positions 0..5 = 0.
        let m = build_per_row_decode_mask(&[2, 5], 5, Dtype::Float32).expect("mask");
        assert_eq!(m.shape().as_slice(), &[2, 1, 1, 5]);
        let v: Vec<f32> = m.to_vec().expect("read mask");
        // Layout: [B=2][1][1][K=5] → row-major flat 10.
        // Row 0:
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert!(v[2].is_infinite() && v[2].is_sign_negative());
        assert!(v[3].is_infinite() && v[3].is_sign_negative());
        assert!(v[4].is_infinite() && v[4].is_sign_negative());
        // Row 1:
        for k in 5..10 {
            assert_eq!(v[k], 0.0, "row 1 position {} should be 0", k - 5);
        }
    }

    #[test]
    fn mask_per_row_decode_invalid_args() {
        // max_len < max(per_row_real_lens) → Err.
        let r = build_per_row_decode_mask(&[3, 5], 4, Dtype::Bfloat16);
        assert!(r.is_err());

        // empty per_row_real_lens → Err.
        let r2 = build_per_row_decode_mask(&[], 4, Dtype::Bfloat16);
        assert!(r2.is_err());

        // negative entry → Err.
        let r3 = build_per_row_decode_mask(&[-1, 4], 4, Dtype::Bfloat16);
        assert!(r3.is_err());

        // zero-length row → Err (would produce all-`-inf` mask).
        let r4 = build_per_row_decode_mask(&[0, 4], 4, Dtype::Bfloat16);
        assert!(r4.is_err());
        let msg = format!("{}", r4.unwrap_err());
        assert!(
            msg.contains("must be > 0"),
            "msg should mention > 0 contract; got: {msg}"
        );
    }

    #[test]
    fn mask_per_row_decode_bfloat16_dtype() {
        // Verify the astype cast to Bfloat16 actually produces a Bfloat16
        // array. The other tests use Float32 for direct .to_vec() access;
        // this one confirms the dtype-cast path works for the production
        // dtype.
        let m = build_per_row_decode_mask(&[3], 4, Dtype::Bfloat16).expect("mask");
        assert_eq!(m.dtype(), Dtype::Bfloat16);
        assert_eq!(m.shape().as_slice(), &[1, 1, 1, 4]);
    }
}
