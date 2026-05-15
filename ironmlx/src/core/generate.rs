//! Single-request generation driver: prefill + decode + sampler + EOS termination.
//!
//! Borrows a [`Qwen35Model`] and [`Tokenizer`] for the lifetime of the stream;
//! owns the per-call cache vector and accumulating token history.

use std::sync::OnceLock;

use anyhow::anyhow;
use mlx::{Array, Dtype};

use crate::core::sampler::Sampler;
use crate::core::tokenizer::{DecodeStream, Tokenizer};
use crate::models::Qwen35Model;
use crate::nn::LayerCache;
use crate::Result;

/// Process-lifetime gate: only the FIRST `GenerationStream` constructed in
/// the process can claim the Metal capture window. Subsequent constructions
/// see this `OnceLock` already set and skip capture (otherwise stacked
/// captures would error). The lock is sticky for process lifetime — to
/// capture another request, restart the server.
static CAPTURE_CLAIMED: OnceLock<()> = OnceLock::new();

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
    /// Max tokens per prefill forward. `0` disables chunking (entire prompt
    /// goes through a single forward). The chunked path bounds activation
    /// memory peak for long agent prompts and lets the GPU pipeline subsequent
    /// chunks; intermediate chunks update the cache only (no lm_head), the
    /// last chunk runs the full forward + lm_head.
    pub prefill_chunk_size: usize,
    /// Image patches `[N_patches, 2, 3, 16, 16]` from preprocess. `None` = text-only.
    pub pixel_values: Option<Array>,
    /// Per-image `(T, H, W)` grids — must match `pixel_values` patch count.
    pub image_grid_thw: Option<Vec<(i32, i32, i32)>>,
    /// `VisionConfig.spatial_merge_size` for this model. Used to compute the
    /// MRoPE VL position-id strides; only consulted when `image_grid_thw` is
    /// `Some`. Default `2` matches Qwen3.5-VL. Sibling VL models with a
    /// different merge factor must set this explicitly.
    pub image_spatial_merge_size: i32,
    /// Token id of `<|image_pad|>` (the per-patch image placeholder). Used to
    /// locate which input_id positions get replaced with vision embeddings and
    /// to drive MRoPE VL stride boundaries. Default [`IMAGE_TOKEN_ID`]
    /// (`248056` for Qwen3.5-VL). Sibling VL models with a different image-pad
    /// id must set this from `Tokenizer::token_to_id("<|image_pad|>")`.
    pub image_token_id: i32,
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

    /// True iff this stream owns the in-flight Metal capture (set when env
    /// var `IRONMLX_CAPTURE_FILE=<path>` was honored at construction time).
    /// Calls `mlx::metal::stop()` in `Drop`.
    capture_active: bool,

    /// When `IRONMLX_CAPTURE_PHASE=decode` is set, capture is deferred until
    /// the first `next_token` call (skipping prefill). This field holds the
    /// path until that first call starts the capture. `None` once started or
    /// if not in decode-only mode.
    capture_pending_decode: Option<String>,
}

impl Drop for GenerationStream<'_> {
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

/// Build MRoPE position ids for a batched, right-padded prefill.
/// Returns `[3, B, max_len]` int32. For batch row i with actual length
/// `prompt_lens[i] = L_i`, the leading `L_i` positions hold `0..L_i-1`;
/// the trailing `max_len - L_i` positions hold 0 (pad — masked out by
/// attention).
///
/// All three MRoPE streams hold the same per-batch-row sequence — this is
/// the text-only convention. VL B>1 (B1-p2.4) will need a multi-stream variant.
pub fn build_position_ids_batched(prompt_lens: &[i32], max_len: i32) -> Result<Array> {
    if prompt_lens.is_empty() {
        return Err(anyhow!(
            "build_position_ids_batched: prompt_lens must be non-empty"
        ));
    }
    if max_len <= 0 {
        return Err(anyhow!(
            "build_position_ids_batched: max_len must be > 0, got {max_len}"
        ));
    }
    let b = prompt_lens.len();
    for (i, &l) in prompt_lens.iter().enumerate() {
        if l <= 0 || l > max_len {
            return Err(anyhow!(
                "build_position_ids_batched: prompt_lens[{i}] = {l} out of (0, {max_len}]"
            ));
        }
    }

    // Build one stream of shape [B, max_len], then tile to [3, B, max_len].
    let s = max_len as usize;
    let mut single_stream = vec![0_i32; b * s];
    for (i, &l) in prompt_lens.iter().enumerate() {
        let l = l as usize;
        for j in 0..l {
            single_stream[i * s + j] = j as i32;
        }
        // positions [l..s] stay 0 (pad — masked out)
    }
    let mut flat = Vec::with_capacity(3 * b * s);
    for _ in 0..3 {
        flat.extend_from_slice(&single_stream);
    }
    let arr: Array = (&flat[..], &[3_i32, b as i32, max_len][..]).try_into()?;
    Ok(arr)
}

/// Build an additive attention mask `[B, 1, max_len, max_len]` for a
/// right-padded batched prefill. For batch row `i` with actual length
/// `prompt_lens[i] = L_i`:
///
///   mask[i, 0, q, k] = 0.0   iff (q < L_i) AND (k < L_i) AND (k <= q)
///                    = -inf  otherwise
///
/// Real tokens occupy columns `[0..L_i)`; the trailing `max_len - L_i`
/// columns are pad. Pad query rows (`q >= L_i`) attend only to themselves
/// (`mask[i, 0, q, q] = 0`) to prevent `softmax(all-`-inf`)` NaN.
///
/// The dtype is `dtype` (typically `Dtype::Bfloat16` to match the SDPA promoted
/// type). Returns a value broadcast-compatible with mlx fast SDPA's expected
/// `[B, N, T_q, T_kv]` shape.
pub fn build_batch_attention_mask(
    prompt_lens: &[i32],
    max_len: i32,
    dtype: Dtype,
) -> Result<Array> {
    if prompt_lens.is_empty() {
        return Err(anyhow!(
            "build_batch_attention_mask: prompt_lens must be non-empty"
        ));
    }
    if max_len <= 0 {
        return Err(anyhow!(
            "build_batch_attention_mask: max_len must be > 0, got {max_len}"
        ));
    }
    for (i, &l) in prompt_lens.iter().enumerate() {
        if l <= 0 || l > max_len {
            return Err(anyhow!(
                "build_batch_attention_mask: prompt_lens[{i}] = {l} out of (0, {max_len}]"
            ));
        }
    }

    let b = prompt_lens.len();
    let s = max_len as usize;
    let total = b * s * s;
    let neg_inf = f32::NEG_INFINITY;
    let mut flat = vec![neg_inf; total];
    for (i, &l) in prompt_lens.iter().enumerate() {
        let l = l as usize;
        // Real query rows (q < l): causal attend to real keys (k < l, k <= q).
        for q in 0..l {
            for k in 0..=q {
                flat[(i * s + q) * s + k] = 0.0;
            }
        }
        // Pad query rows (q >= l): allow self-attention only
        // (`mask[i, 0, q, q] = 0`). Without this, the row is all `-inf`
        // and `softmax(all-INF)` yields NaN, which propagates through
        // subsequent layers and contaminates real-row outputs via
        // residual connections / layer norms (NaN × any = NaN). Letting
        // pad-q attend to itself produces a benign zero output (since
        // pad-row outputs are discarded by `slice_last_and_project`'s
        // per-row slice anyway, and `kv_validity_mask` zeros V at pad
        // positions in `attention::forward_on`).
        for q in l..s {
            flat[(i * s + q) * s + q] = 0.0;
        }
    }

    let arr_f32: Array = (&flat[..], &[b as i32, 1_i32, max_len, max_len][..]).try_into()?;
    mlx::ops::cast::astype(&arr_f32, dtype).map_err(|e| anyhow!("astype mask: {e}"))
}

/// Build MRoPE position ids for one batched decode step.
/// Returns `[3, B, 1]` int32. Each batch row `i` holds the position id
/// `per_row_pos[i]` for its new token; all three MRoPE streams hold the
/// same value (text-only convention; VL B>1 in B1-p2.4 will need a
/// multi-stream variant).
pub fn build_decode_position_ids(per_row_pos: &[i32]) -> Result<Array> {
    if per_row_pos.is_empty() {
        return Err(anyhow!(
            "build_decode_position_ids: per_row_pos must be non-empty"
        ));
    }
    for (i, &p) in per_row_pos.iter().enumerate() {
        if p < 0 {
            return Err(anyhow!(
                "build_decode_position_ids: per_row_pos[{i}] = {p} must be >= 0"
            ));
        }
    }

    let b = per_row_pos.len();
    let mut flat = Vec::with_capacity(3 * b);
    for _ in 0..3 {
        flat.extend_from_slice(per_row_pos);
    }
    let arr: Array = (&flat[..], &[3_i32, b as i32, 1_i32][..]).try_into()?;
    Ok(arr)
}

/// Build a per-row decode attention mask `[B, 1, 1, max_len]`.
///
/// Each batch row `b` attends to K/V positions `0..per_row_real_lens[b]`
/// (real cache) and is `-inf`-masked at positions
/// `per_row_real_lens[b]..max_len` (stale / unused cache slots). Used by
/// the decode path when rows have ragged cache offsets — typically
/// `per_row_real_lens[b] = cache.offsets()[b] + 1` after a per-row write.
///
/// `max_len` must satisfy `max_len >= max(per_row_real_lens)` — it sets
/// the K-dimension of the returned mask and must equal the fetched K/V
/// slice's K dim. The returned mask is additive (consumed by mlx fast
/// SDPA's `mask_arr` slot with `mask_mode = ""`); 0.0 means attend, -inf
/// means mask out.
///
/// Differs in shape from [`build_batch_attention_mask`] (which is
/// prefill-only, `[B, 1, T_q, T_kv]`) because decode has `T_q = 1`.
///
/// Every entry of `per_row_real_lens` must be `> 0`: a zero-length row would
/// produce an all-`-inf` mask, and SDPA's softmax of all-`-inf` yields NaN
/// which would contaminate other rows via residual connections. Callers
/// that have inactive slots should omit them from the batch rather than
/// pass a length-0 mask row. Matches the `prompt_lens[i] > 0` contract
/// enforced by [`build_batch_attention_mask`].
///
/// **Production callers (B1-p2.3c-2):** [`Scheduler::step`](crate::core::scheduler::Scheduler::step)
/// — builds this mask from per-row cache offsets + per_row_lens before
/// each decode forward, so SDPA correctly masks out stale K/V cells for
/// rows whose offsets have diverged from `max(offsets)` (typically because
/// the row has finished and its cache no longer advances while other rows
/// continue).
pub fn build_per_row_decode_mask(
    per_row_real_lens: &[i32],
    max_len: i32,
    dtype: Dtype,
) -> Result<Array> {
    if per_row_real_lens.is_empty() {
        return Err(anyhow!(
            "build_per_row_decode_mask: per_row_real_lens must be non-empty"
        ));
    }
    if max_len <= 0 {
        return Err(anyhow!(
            "build_per_row_decode_mask: max_len must be > 0, got {max_len}"
        ));
    }
    for (i, &l) in per_row_real_lens.iter().enumerate() {
        if l <= 0 {
            return Err(anyhow!(
                "build_per_row_decode_mask: per_row_real_lens[{i}] = {l} must be > 0 \
                 (zero-length row would produce all-`-inf` mask, yielding softmax NaN)"
            ));
        }
        if l > max_len {
            return Err(anyhow!(
                "build_per_row_decode_mask: per_row_real_lens[{i}] = {l} > max_len = {max_len}"
            ));
        }
    }

    let b = per_row_real_lens.len();
    let s = max_len as usize;
    let neg_inf = f32::NEG_INFINITY;
    let mut flat = vec![neg_inf; b * s];
    for (i, &l) in per_row_real_lens.iter().enumerate() {
        let l = l as usize;
        for k in 0..l {
            flat[i * s + k] = 0.0;
        }
    }

    let arr_f32: Array = (&flat[..], &[b as i32, 1_i32, 1_i32, max_len][..]).try_into()?;
    mlx::ops::cast::astype(&arr_f32, dtype).map_err(|e| anyhow!("astype mask: {e}"))
}

/// Build a per-token validity mask `[B, max_len]` for the hybrid model's
/// **linear-attention** path (`GatedDeltaNet`). For batch row `i` with
/// actual length `prompt_lens[i] = L_i` (right-padded prefill):
///
///   linear_mask[i, t] = true   if t < L_i        (real token)
///                     = false  otherwise         (right-pad slot)
///
/// The kernel reads `mask[b_idx * T + t]` as a boolean (`if (mask[...])`)
/// — `true` → compute, `false` → emit zero for that position. This
/// differs in shape from the full-attention mask returned by
/// [`build_batch_attention_mask`] (which is `[B, 1, T_q, T_kv]` additive
/// bf16 for `scaled_dot_product_attention`). The hybrid model's
/// `DecoderLayer` routes each mask to the matching attention path.
///
/// The mask dtype is `bool` — the kernel only needs truthiness, not
/// magnitudes, and bool minimises memory.
pub fn build_batch_linear_mask(prompt_lens: &[i32], max_len: i32) -> Result<Array> {
    if prompt_lens.is_empty() {
        return Err(anyhow!(
            "build_batch_linear_mask: prompt_lens must be non-empty"
        ));
    }
    if max_len <= 0 {
        return Err(anyhow!(
            "build_batch_linear_mask: max_len must be > 0, got {max_len}"
        ));
    }
    for (i, &l) in prompt_lens.iter().enumerate() {
        if l <= 0 || l > max_len {
            return Err(anyhow!(
                "build_batch_linear_mask: prompt_lens[{i}] = {l} out of (0, {max_len}]"
            ));
        }
    }

    let b = prompt_lens.len();
    let s = max_len as usize;
    let mut flat = vec![false; b * s];
    for (i, &l) in prompt_lens.iter().enumerate() {
        let l = l as usize;
        for t in 0..l {
            flat[i * s + t] = true;
        }
        // positions [l..s] stay false (pad — kernel skips compute)
    }

    let arr: Array = (&flat[..], &[b as i32, max_len][..]).try_into()?;
    Ok(arr)
}

/// Token ID for `<|image_pad|>` in Qwen3.5-VL (from model `config.json`,
/// **not** from mlx-vlm defaults which differ).
///
/// TODO P6.5 (audit ref B5): plumb from `Tokenizer` at load time so the value
/// works for sibling VL models with different image-pad token ids.
pub const IMAGE_TOKEN_ID: i32 = 248056;

/// MRoPE 3-stream position_ids for a VL sequence (B=1, image-only, no video).
///
/// Output shape: `[3, 1, S]` (int32).
///   - Stream 0 (t): temporal positions; equals spatial stream for text tokens.
///   - Stream 1 (h): height positions; equals temporal stream for text tokens.
///   - Stream 2 (w): width positions; equals temporal stream for text tokens.
///
/// Algorithm: faithful Rust translation of `LanguageModel.get_rope_index` in
/// `mlx_vlm/models/qwen3_vl/language.py:333-486`, restricted to the B=1
/// image-only path. Video token support is intentionally omitted (P6 scope).
///
/// For each image at grid `(t, h, w)`:
///   - `llm_grid_t = t`, `llm_grid_h = h / spatial_merge_size`,
///     `llm_grid_w = w / spatial_merge_size`
///   - Image token count = `llm_grid_t * llm_grid_h * llm_grid_w`
///   - t_index: broadcasts `arange(llm_grid_t)` over `(llm_grid_t, llm_grid_h*llm_grid_w)` then flattened
///   - h_index: broadcasts `arange(llm_grid_h)` over `(llm_grid_t, llm_grid_h, llm_grid_w)` then flattened
///   - w_index: broadcasts `arange(llm_grid_w)` over `(llm_grid_t, llm_grid_h, llm_grid_w)` then flattened
///   - Image block = `stack([t_index, h_index, w_index]) + text_len + st_idx`
///
/// `grid_thw`: one entry per image, `(t, h, w)` in original pixel-patch units.
/// `image_token_id`: the sentinel value that marks each image token in `input_ids`.
/// `spatial_merge_size`: typically 2 (from `vision_config.spatial_merge_size`).
///
/// # Panics
/// Returns `Err` if `grid_thw.is_empty()` or `spatial_merge_size <= 0`. Caller
/// is responsible for going through [`build_position_ids`] in the text-only
/// case rather than passing an empty `grid_thw`.
pub fn build_position_ids_vl(
    input_ids: &[i32],
    grid_thw: &[(i32, i32, i32)],
    image_token_id: i32,
    spatial_merge_size: i32,
) -> crate::Result<Array> {
    if grid_thw.is_empty() {
        return Err(anyhow!(
            "build_position_ids_vl: grid_thw must be non-empty (use build_position_ids for text-only)"
        ));
    }
    if spatial_merge_size <= 0 {
        return Err(anyhow!(
            "build_position_ids_vl: spatial_merge_size must be > 0 (got {spatial_merge_size})"
        ));
    }

    // We build every block entirely in Rust (Vec<i32>) and assemble a single
    // Array at the end. This avoids repeatedly flushing the MLX graph for tiny
    // integer bookkeeping work.
    //
    // `result` accumulates the [3, S] position matrix row-major:
    //   result[0..S]   → stream 0 (t)
    //   result[S..2S]  → stream 1 (h)
    //   result[2S..3S] → stream 2 (w)
    // We build three parallel Vecs and interleave at the end.
    let s = input_ids.len();
    let mut stream_t: Vec<i32> = Vec::with_capacity(s);
    let mut stream_h: Vec<i32> = Vec::with_capacity(s);
    let mut stream_w: Vec<i32> = Vec::with_capacity(s);

    let mut st: usize = 0; // current scan position in input_ids
    let mut st_idx: i32 = 0; // logical position offset (max of last block + 1)

    for (img_idx, &(t, h, w)) in grid_thw.iter().enumerate() {
        let llm_grid_t = t;
        let llm_grid_h = h / spatial_merge_size;
        let llm_grid_w = w / spatial_merge_size;

        // Find the first occurrence of image_token_id at or after `st`.
        // This is `ed_image` in the Python — the start of the image token span.
        let ed_image = input_ids[st..]
            .iter()
            .position(|&tok| tok == image_token_id)
            .map(|rel| st + rel)
            .ok_or_else(|| {
                anyhow!(
                    "build_position_ids_vl: no image_token_id found for image {img_idx} \
                     (st={st}, input_ids len={})",
                    input_ids.len()
                )
            })?;

        // --- Text prefix block [st .. ed_image) ---
        let text_len = (ed_image - st) as i32;
        // All three streams hold the same values for text tokens.
        for k in 0..text_len {
            stream_t.push(st_idx + k);
            stream_h.push(st_idx + k);
            stream_w.push(st_idx + k);
        }

        // st_idx for the image block = st_idx + text_len (max of text block + 1)
        let img_st_idx = st_idx + text_len;

        // --- Image block ---
        // t_index: arange(llm_grid_t) broadcast over (llm_grid_t, llm_grid_h*llm_grid_w), flattened
        // h_index: arange(llm_grid_h) broadcast over (llm_grid_t, llm_grid_h, llm_grid_w), flattened
        // w_index: arange(llm_grid_w) broadcast over (llm_grid_t, llm_grid_h, llm_grid_w), flattened
        let n_img = llm_grid_t * llm_grid_h * llm_grid_w;
        for ti in 0..llm_grid_t {
            for hi in 0..llm_grid_h {
                for wi in 0..llm_grid_w {
                    stream_t.push(img_st_idx + ti);
                    stream_h.push(img_st_idx + hi);
                    stream_w.push(img_st_idx + wi);
                }
            }
        }

        // st_idx for the next iteration = max of this image block + 1.
        // max of image block = img_st_idx + max(llm_grid_t-1, llm_grid_h-1, llm_grid_w-1)
        // BUT: st_idx = previous_max + 1, so:
        let img_block_max = img_st_idx + (llm_grid_t - 1).max(llm_grid_h - 1).max(llm_grid_w - 1);
        st_idx = img_block_max + 1;

        // Advance input scan past the image token span.
        st = ed_image + n_img as usize;
    }

    // --- Trailing text block (after last image) ---
    if st < input_ids.len() {
        let trail_len = (input_ids.len() - st) as i32;
        for k in 0..trail_len {
            stream_t.push(st_idx + k);
            stream_h.push(st_idx + k);
            stream_w.push(st_idx + k);
        }
    }

    // Sanity: each stream must have exactly S entries. Algorithmic invariant
    // (every push to one stream pushes to the other two), so debug_assert.
    let total = stream_t.len();
    debug_assert_eq!(total, s);
    debug_assert_eq!(stream_h.len(), s);
    debug_assert_eq!(stream_w.len(), s);

    // Build [3, S] array and reshape to [3, 1, S].
    // Layout: [stream_t | stream_h | stream_w] contiguous, shape [3, S].
    let mut flat: Vec<i32> = Vec::with_capacity(3 * s);
    flat.extend_from_slice(&stream_t);
    flat.extend_from_slice(&stream_h);
    flat.extend_from_slice(&stream_w);

    let arr: Array = (&flat[..], &[3_i32, 1_i32, s as i32][..]).try_into()?;
    Ok(arr)
}

/// Count occurrences of `image_token_id` in a u32 slice of token ids.
/// Used by the chunked-prefill loop to know how many vision_embed rows
/// belong to a given chunk.
fn count_image_pad(ids: &[u32], image_token_id: i32) -> usize {
    let target = image_token_id as u32;
    ids.iter().filter(|&&t| t == target).count()
}

/// Slice a MRoPE `[3, 1, S]` position-id tensor on axis 2 by a half-open
/// range `[start, stop)`. Returns `[3, 1, stop - start]`.
fn slice_pos_ids_axis2(pos_full: &mlx::Array, start: i32, stop: i32) -> Result<mlx::Array> {
    let shape = pos_full.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 3 || dims[1] != 1 {
        return Err(anyhow!(
            "slice_pos_ids_axis2: expected [3,1,S] tensor, got {:?}",
            dims
        ));
    }
    let s_full = dims[2];
    if start < 0 || stop > s_full || start > stop {
        return Err(anyhow!(
            "slice_pos_ids_axis2: bad range [{}, {}) for S={}",
            start,
            stop,
            s_full
        ));
    }
    mlx::ops::slice(pos_full, &[0_i32, 0, start][..], &[3_i32, 1, stop][..])
        .map_err(|e| anyhow!("slice_pos_ids_axis2 mlx::ops::slice failed: {e}"))
}

/// Slice rows `[start, stop)` from a `[N, hidden]` vision_embeds tensor.
fn slice_vision_embeds_rows(ve_full: &mlx::Array, start: usize, stop: usize) -> Result<mlx::Array> {
    let shape = ve_full.shape();
    let dims = shape.as_slice();
    if dims.len() != 2 {
        return Err(anyhow!(
            "slice_vision_embeds_rows: expected [N, H] tensor, got {:?}",
            dims
        ));
    }
    let n = dims[0] as usize;
    let hidden = dims[1];
    if stop > n || start > stop {
        return Err(anyhow!(
            "slice_vision_embeds_rows: bad range [{}, {}) for N={}",
            start,
            stop,
            n
        ));
    }
    mlx::ops::slice(
        ve_full,
        &[start as i32, 0_i32][..],
        &[stop as i32, hidden][..],
    )
    .map_err(|e| anyhow!("slice_vision_embeds_rows mlx::ops::slice failed: {e}"))
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

        // P8a-stage4/6 Metal capture hook. Gated by `IRONMLX_CAPTURE_FILE`
        // env var + first-construction OnceLock. `IRONMLX_CAPTURE_PHASE=decode`
        // defers start to the first `next_token` call (skips prefill).
        let (capture_active, capture_pending_decode) = try_start_capture();

        let cap = (prompt_len + request.max_new_tokens) as i32;
        let dtype = Dtype::Bfloat16;
        let mut cache = model.make_cache(/* batch */ 1, cap, dtype)?;

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
        // chunking loop and build MRoPE position ids for the full prompt.
        // Each chunk then slices vision_embeds and position_ids by its
        // own range, ensuring the chunked path is numerically equivalent
        // to single-chunk forward_vl.
        let (vision_embeds_full, position_ids_full) = if let (Some(pv), Some(grids)) = (
            request.pixel_values.as_ref(),
            request.image_grid_thw.as_deref(),
        ) {
            let ve = model.compute_vision_embeds(pv, grids, ())?;
            let full_ids_i32: Vec<i32> = request.prompt_ids.iter().map(|&u| u as i32).collect();
            let pos_full = build_position_ids_vl(
                &full_ids_i32,
                grids,
                request.image_token_id,
                request.image_spatial_merge_size,
            )?;
            (Some(ve), Some(pos_full))
        } else {
            (None, None)
        };

        let chunk_size = request.prefill_chunk_size;
        let prompt_len_i32 = prompt_len as i32;
        let mut pos: i32 = 0;
        let mut image_pad_consumed: usize = 0;
        let last_logits = loop {
            let remaining = prompt_len_i32 - pos;
            let n = if chunk_size == 0 {
                remaining
            } else {
                remaining.min(chunk_size as i32)
            };
            let chunk_ids = &request.prompt_ids[pos as usize..(pos as usize + n as usize)];
            let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;

            // VL chunk: slice pre-computed position_ids by chunk range.
            // Text chunk: use the simpler single-stream builder.
            let chunk_pos_ids = if let Some(pos_full) = position_ids_full.as_ref() {
                slice_pos_ids_axis2(pos_full, pos, pos + n)?
            } else {
                build_position_ids(pos, n)?
            };

            // VL chunk: count image_pad tokens, slice the matching rows
            // out of vision_embeds_full, advance the consumed counter.
            let ve_slice = if let Some(ve_full) = vision_embeds_full.as_ref() {
                let k_i = count_image_pad(chunk_ids, request.image_token_id);
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

            let is_last = pos + n == prompt_len_i32;
            let logits_or_hidden = if vision_embeds_full.is_some() {
                let logits = model.forward_vl_chunk(
                    &chunk_arr,
                    &chunk_pos_ids,
                    None, // per_row_lens
                    None, // decode_mask
                    Some(&mut cache),
                    ve_slice.as_ref(),
                    request.image_token_id,
                    (),
                )?;
                if is_last {
                    Some(logits)
                } else {
                    None
                }
            } else if is_last {
                Some(model.forward_on(
                    &chunk_arr,
                    &chunk_pos_ids,
                    None, // per_row_lens
                    None, // decode_mask
                    Some(&mut cache),
                    (),
                )?)
            } else {
                let hidden = model.text().forward_on(
                    &chunk_arr,
                    &chunk_pos_ids,
                    None, // per_row_lens
                    None, // decode_mask
                    Some(&mut cache),
                    (),
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
                vision_embeds_full: None,
                position_ids_full: None,
                image_pad_consumed: 0,
                history,
                request,
                finished: false,
                pipelined: true,
                pending_token_arr: Some(pending),
                detok: Some(detok),
                last_decoded_text: String::new(),
                capture_active,
                capture_pending_decode,
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
                vision_embeds_full: None,
                position_ids_full: None,
                image_pad_consumed: 0,
                history,
                request,
                finished: false,
                pipelined: false,
                pending_token_arr: None,
                detok: None,
                last_decoded_text: initial_text,
                capture_active,
                capture_pending_decode,
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

    /// Pull the next event. Returns `Ok(None)` after the stream terminates.
    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }
        // If decode-phase Metal capture was deferred, start it now (right
        // before the first decode-step work hits the GPU).
        self.start_deferred_capture();
        if self.pipelined {
            self.next_token_pipelined()
        } else {
            self.next_token_sync()
        }
    }

    /// Pipelined hot path. Invariant: `self.pending_token_arr` is `Some` and
    /// the lazy scalar (shape `[]` or `[1]`) u32 Array of the token to be
    /// returned on this call.
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
            .expect("pipelined mode invariant: pending_token_arr is Some")
            .reshape((1_i32, 1_i32))?;
        let pos = (self.history.len() - 1) as i32;
        let position_ids = build_position_ids(pos, 1)?;
        let logits = self.model.forward_on(
            &token_arr_in,
            &position_ids,
            None, // per_row_lens
            None, // decode_mask
            Some(&mut self.cache),
            (),
        )?;
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
        let logits = self.model.forward_on(
            &token_arr,
            &position_ids,
            None, // per_row_lens
            None, // decode_mask
            Some(&mut self.cache),
            (),
        )?;
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
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: IMAGE_TOKEN_ID,
        };
        assert!(req.pixel_values.is_none());
        assert!(req.image_grid_thw.is_none());
        assert_eq!(req.prompt_ids.len(), 3);
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
