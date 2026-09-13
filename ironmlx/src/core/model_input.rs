//! Model input construction: positions, attention masks, and vision token alignment.

use crate::Result;
use anyhow::anyhow;
use mlx::{Array, Dtype};

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

/// Slice `logits[row_idx, 0, :]` and reshape to `[vocab]`.
///
/// Common pattern used by both [`Scheduler::step`](crate::core::scheduler::Scheduler::step)
/// (per-row decode sampling) and [`Scheduler::admit_mid`](crate::core::scheduler::Scheduler::admit_mid)
/// (first-token sampling after temp-cache adoption in 3c-3). Extracted so
/// the indexing math lives in one place.
///
/// Requires `logits` to be a rank-3 tensor `[B, 1, vocab]` (the shape
/// produced by both `Qwen35Model::batched_prefill` and
/// `Qwen35Model::forward_on` on the decode path).
pub fn slice_logits_row(logits: &Array, row_idx: usize) -> Result<Array> {
    let shape = logits.shape();
    let shape_slice = shape.as_slice();
    if shape_slice.len() != 3 {
        return Err(anyhow!(
            "slice_logits_row: expected logits shape [B, 1, vocab]; got rank {}",
            shape_slice.len()
        ));
    }
    let b = shape_slice[0];
    if row_idx as i32 >= b {
        return Err(anyhow!("slice_logits_row: row_idx {} >= B {}", row_idx, b));
    }
    let vocab = shape_slice[2];
    let row = mlx::ops::indexing::slice(
        logits,
        &[row_idx as i32, 0_i32, 0_i32][..],
        &[row_idx as i32 + 1, 1_i32, vocab][..],
    )
    .map_err(|e| anyhow!("slice_logits_row: slice failed: {e:?}"))?;
    row.reshape(&[vocab][..])
        .map_err(|e| anyhow!("slice_logits_row: reshape failed: {e:?}"))
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

/// Build an additive causal mask for a ragged batched cache append.
///
/// The returned shape is `[B, 1, max_new_len, max_post_len]`. Row `i`
/// appends `per_row_new_lens[i]` real tokens after `pre_offsets[i]` cached
/// tokens. Real query `q` can attend through `pre_offsets[i] + q`; padded
/// queries can attend to the row's valid post-append history so their softmax
/// remains finite, but their outputs are discarded and never written to KV.
pub fn build_batched_append_attention_mask(
    pre_offsets: &[i32],
    per_row_new_lens: &[i32],
    max_new_len: i32,
    dtype: Dtype,
) -> Result<Array> {
    if pre_offsets.is_empty() || pre_offsets.len() != per_row_new_lens.len() {
        return Err(anyhow!(
            "build_batched_append_attention_mask: pre_offsets len {} must equal non-zero per_row_new_lens len {}",
            pre_offsets.len(),
            per_row_new_lens.len()
        ));
    }
    if max_new_len <= 0 {
        return Err(anyhow!(
            "build_batched_append_attention_mask: max_new_len must be > 0, got {max_new_len}"
        ));
    }

    let mut post_lens = Vec::with_capacity(pre_offsets.len());
    for (row, (&pre, &new_len)) in pre_offsets.iter().zip(per_row_new_lens.iter()).enumerate() {
        if pre < 0 {
            return Err(anyhow!(
                "build_batched_append_attention_mask: pre_offsets[{row}] = {pre} must be >= 0"
            ));
        }
        if new_len < 0 || new_len > max_new_len {
            return Err(anyhow!(
                "build_batched_append_attention_mask: per_row_new_lens[{row}] = {new_len} out of [0, {max_new_len}]"
            ));
        }
        post_lens.push(pre + new_len);
    }
    let max_post_len = post_lens.iter().copied().max().unwrap_or(0);
    if max_post_len <= 0 {
        return Err(anyhow!(
            "build_batched_append_attention_mask: every row has zero post-append length"
        ));
    }

    let b = pre_offsets.len();
    let q_len = max_new_len as usize;
    let kv_len = max_post_len as usize;
    let neg_inf = f32::NEG_INFINITY;
    let mut flat = vec![neg_inf; b * q_len * kv_len];
    for row in 0..b {
        let pre = pre_offsets[row] as usize;
        let new_len = per_row_new_lens[row] as usize;
        let post_len = post_lens[row] as usize;
        for q in 0..q_len {
            let visible = if q < new_len {
                pre + q + 1
            } else {
                post_len.max(1)
            };
            for k in 0..visible.min(kv_len) {
                flat[(row * q_len + q) * kv_len + k] = 0.0;
            }
        }
    }

    let arr_f32: Array =
        (&flat[..], &[b as i32, 1_i32, max_new_len, max_post_len][..]).try_into()?;
    mlx::ops::cast::astype(&arr_f32, dtype).map_err(|e| anyhow!("astype mask: {e}"))
}

/// Build a cross-chunk prefill attention mask for one chunk of a chunked
/// mid-batch admit. Shape: `[1, 1, chunk_len, chunk_start + chunk_len]`.
///
/// Row `q` in the chunk queries K/V positions `0..chunk_start + q + 1`:
/// - Columns `0..chunk_start`: attend to earlier-chunk KV cells (all 0.0).
/// - Columns `chunk_start..chunk_start + q + 1`: causal within-chunk (0.0).
/// - Columns `chunk_start + q + 1..`: masked out (-inf).
///
/// This matches the KV slice returned by the model's internal `KVCache::update_and_fetch_on`
/// after `temp_cache.offsets[0] = chunk_start` at call time: the model reads back
/// `chunk_start + chunk_len` keys/values (earlier chunks' + this chunk's).
///
/// `dtype` is typically `Dtype::Bfloat16` to match the SDPA promoted type.
pub fn build_chunked_prefill_attention_mask(
    chunk_start: i32,
    chunk_len: i32,
    dtype: Dtype,
) -> Result<Array> {
    if chunk_len <= 0 {
        return Err(anyhow!(
            "build_chunked_prefill_attention_mask: chunk_len must be > 0, got {chunk_len}"
        ));
    }
    if chunk_start < 0 {
        return Err(anyhow!(
            "build_chunked_prefill_attention_mask: chunk_start must be >= 0, got {chunk_start}"
        ));
    }

    let kv_len = (chunk_start + chunk_len) as usize;
    let q_len = chunk_len as usize;
    let cs = chunk_start as usize;
    let neg_inf = f32::NEG_INFINITY;
    let mut flat = vec![neg_inf; q_len * kv_len];

    for q in 0..q_len {
        // Attend to all earlier-chunk positions [0..chunk_start].
        for k in 0..cs {
            flat[q * kv_len + k] = 0.0;
        }
        // Causal within current chunk: attend to chunk positions [0..=q].
        for k in 0..=q {
            flat[q * kv_len + cs + k] = 0.0;
        }
        // Positions cs + q + 1..kv_len stay -inf (not yet written).
    }

    let arr_f32: Array = (
        &flat[..],
        &[1_i32, 1_i32, chunk_len, chunk_start + chunk_len][..],
    )
        .try_into()
        .map_err(|e| anyhow!("build_chunked_prefill_attention_mask try_into: {e:?}"))?;
    mlx::ops::cast::astype(&arr_f32, dtype)
        .map_err(|e| anyhow!("build_chunked_prefill_attention_mask astype: {e}"))
}

/// Build a per-token validity mask `[1, chunk_start + chunk_len]` (bool) for
/// the hybrid model's linear-attention path for one chunk of a chunked
/// mid-batch admit.
///
/// All positions `0..chunk_start + chunk_len` are `true` — every position
/// from earlier chunks and the current chunk is real (no padding).
pub fn build_chunked_prefill_linear_mask(chunk_start: i32, chunk_len: i32) -> Result<Array> {
    if chunk_len <= 0 {
        return Err(anyhow!(
            "build_chunked_prefill_linear_mask: chunk_len must be > 0, got {chunk_len}"
        ));
    }
    if chunk_start < 0 {
        return Err(anyhow!(
            "build_chunked_prefill_linear_mask: chunk_start must be >= 0, got {chunk_start}"
        ));
    }

    let total = (chunk_start + chunk_len) as usize;
    let flat = vec![true; total];
    let arr: Array = (&flat[..], &[1_i32, chunk_start + chunk_len][..])
        .try_into()
        .map_err(|e| anyhow!("build_chunked_prefill_linear_mask try_into: {e:?}"))?;
    Ok(arr)
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
/// Handles B=1 image-only sequences. Video tokens are not supported.
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

/// Build MRoPE position ids `[3, B, max_len]` for mixed text+VL batch prefill
/// (right-padded). Each row independently builds its `[3, L_i]` position ids:
/// - VL row (`per_row_grid_thw[i].is_some()`): reuse [`build_position_ids_vl`].
/// - Text row (`per_row_grid_thw[i].is_none()`): triple-replicated `0..L_i`
///   (same convention as `build_position_ids_batched`'s per-row degraded MRoPE).
///
/// Pad columns `[L_i..max_len]` get position 0 on all three streams. This
/// matches `build_position_ids_batched` pad convention and is harmless under
/// the right-pad attention mask which zeroes pad K/V.
///
/// # Arguments
/// - `per_row_prompt_ids` — exactly `B` slices; each slice is row `i`'s prompt token ids.
/// - `per_row_grid_thw` — exactly `B` options; `Some(grids)` for VL row, `None` for text row.
///   `Some(&[])` (empty grids) is treated as text row (degraded MRoPE), not an error.
/// - `image_token_id` — token id of `<|image_pad|>`.
/// - `image_spatial_merge_size` — same as [`build_position_ids_vl`].
/// - `max_len` — must be `>= max(L_i)`. Pad columns beyond `L_i` get position 0.
///
/// # Errors
/// - Length of `per_row_prompt_ids` != length of `per_row_grid_thw`.
/// - Empty slice in `per_row_prompt_ids` (zero-length row).
/// - `max_len < L_i` for any row.
/// - `image_spatial_merge_size <= 0` (only checked when at least one VL row is present;
///   propagates from `build_position_ids_vl`).
/// - Any per-row VL build error propagates from `build_position_ids_vl`.
#[allow(clippy::type_complexity)]
pub fn build_position_ids_vl_batched(
    per_row_prompt_ids: &[&[i32]],
    per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
    image_token_id: i32,
    image_spatial_merge_size: i32,
    max_len: i32,
) -> Result<Array> {
    let b = per_row_prompt_ids.len();
    if b != per_row_grid_thw.len() {
        return Err(anyhow!(
            "build_position_ids_vl_batched: per_row_prompt_ids.len()={} != per_row_grid_thw.len()={}",
            b,
            per_row_grid_thw.len()
        ));
    }
    if b == 0 {
        return Err(anyhow!(
            "build_position_ids_vl_batched: batch must be non-empty"
        ));
    }
    if max_len <= 0 {
        return Err(anyhow!(
            "build_position_ids_vl_batched: max_len must be > 0 (got {max_len})"
        ));
    }

    let m = max_len as usize;
    // Flat output layout: [3, B, max_len] contiguous row-major.
    // Element (s, b, col) at index s*B*max_len + b*max_len + col.
    let mut flat: Vec<i32> = vec![0; 3 * b * m];

    for (row, (&ids, grids_opt)) in per_row_prompt_ids
        .iter()
        .zip(per_row_grid_thw.iter())
        .enumerate()
    {
        let l_i = ids.len();
        if l_i == 0 {
            return Err(anyhow!(
                "build_position_ids_vl_batched: row {row} has zero-length prompt"
            ));
        }
        if l_i > m {
            return Err(anyhow!(
                "build_position_ids_vl_batched: row {row} length {l_i} > max_len {max_len}"
            ));
        }

        // Per-row [3, L_i] flat: stream s at offset s*L_i + col within row buffer.
        let row_flat: Vec<i32> = match grids_opt {
            Some(grids) if !grids.is_empty() => {
                // VL row: reuse existing single-stream builder.
                let arr =
                    build_position_ids_vl(ids, grids, image_token_id, image_spatial_merge_size)?;
                arr.to_vec::<i32>()
                    .map_err(|e| anyhow!("row {row} to_vec: {e}"))?
            }
            _ => {
                // Text row (None) or VL row with empty grids: degrade to triple-replicated 0..L_i.
                let mut buf = vec![0_i32; 3 * l_i];
                for col in 0..l_i {
                    buf[col] = col as i32;
                    buf[l_i + col] = col as i32;
                    buf[2 * l_i + col] = col as i32;
                }
                buf
            }
        };

        // Scatter per-row [3, L_i] into [3, B, max_len] flat. Pad columns
        // [L_i..max_len] stay 0 from the initial fill.
        for s in 0..3 {
            for col in 0..l_i {
                flat[s * b * m + row * m + col] = row_flat[s * l_i + col];
            }
        }
    }

    let arr: Array = (&flat[..], &[3_i32, b as i32, max_len][..])
        .try_into()
        .map_err(|e| anyhow!("build_position_ids_vl_batched try_into: {e:?}"))?;
    Ok(arr)
}

/// Count occurrences of `image_token_id` in a u32 slice of token ids.
/// Used by the chunked-prefill loop to know how many vision_embed rows
/// belong to a given chunk.
pub fn count_image_pad(ids: &[u32], image_token_id: i32) -> usize {
    let target = image_token_id as u32;
    ids.iter().filter(|&&t| t == target).count()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VlChunkComposition {
    pub seq_len: usize,
    pub image_tokens: usize,
    pub text_tokens: usize,
    pub image_runs: usize,
    pub leading_image_tokens: usize,
    pub trailing_image_tokens: usize,
}

pub(crate) fn vl_chunk_composition(ids: &[u32], image_token_id: i32) -> VlChunkComposition {
    let target = if image_token_id >= 0 {
        Some(image_token_id as u32)
    } else {
        None
    };
    let mut image_tokens = 0usize;
    let mut image_runs = 0usize;
    let mut in_image_run = false;
    for &id in ids {
        let is_image = Some(id) == target;
        if is_image {
            image_tokens += 1;
            if !in_image_run {
                image_runs += 1;
                in_image_run = true;
            }
        } else {
            in_image_run = false;
        }
    }
    let leading_image_tokens = ids.iter().take_while(|&&id| Some(id) == target).count();
    let trailing_image_tokens = ids
        .iter()
        .rev()
        .take_while(|&&id| Some(id) == target)
        .count();
    VlChunkComposition {
        seq_len: ids.len(),
        image_tokens,
        text_tokens: ids.len().saturating_sub(image_tokens),
        image_runs,
        leading_image_tokens,
        trailing_image_tokens,
    }
}

pub(crate) fn log_vl_chunk_composition(
    path: &str,
    chunk_range: std::ops::Range<i32>,
    is_last: bool,
    ids: &[u32],
    image_token_id: i32,
    image_rows: std::ops::Range<usize>,
) {
    if std::env::var_os("IRONMLX_GEMMA4_VL_PROFILE").is_none() {
        return;
    }
    let c = vl_chunk_composition(ids, image_token_id);
    tracing::info!(
        "[gemma4-vl-profile] vl_chunk_composition path={} chunk_start={} chunk_end={} seq={} image_tokens={} text_tokens={} image_runs={} leading_image_tokens={} trailing_image_tokens={} image_rows_start={} image_rows_end={} is_last={}",
        path,
        chunk_range.start,
        chunk_range.end,
        c.seq_len,
        c.image_tokens,
        c.text_tokens,
        c.image_runs,
        c.leading_image_tokens,
        c.trailing_image_tokens,
        image_rows.start,
        image_rows.end,
        is_last
    );
}

const VL_FINAL_TEXT_TAIL_ABSORB_TOKENS: usize = 64;

/// Extend a VL chunk end when the fixed boundary would split a contiguous
/// image-token run. Keeping each image's placeholder run in one text forward
/// avoids extra cache-update chunks and reduces long-tail MLX/Metal stalls.
///
/// After choosing the boundary, absorb a short final text-only tail into the
/// current chunk. The extra text tokens are cheap compared with launching a
/// separate final forward over a tiny tail while carrying the full KV state.
pub(crate) fn extend_vl_chunk_end_for_image_pad(
    prompt_ids: &[u32],
    image_token_id: i32,
    chunk_start: i32,
    base_chunk_end: i32,
) -> i32 {
    if image_token_id < 0 || chunk_start < 0 || base_chunk_end <= chunk_start {
        return base_chunk_end;
    }

    let len = prompt_ids.len();
    let Ok(mut end) = usize::try_from(base_chunk_end) else {
        return base_chunk_end;
    };
    if end == 0 || end >= len {
        return base_chunk_end.min(len as i32);
    }

    let pad = image_token_id as u32;
    if prompt_ids[end - 1] == pad && prompt_ids[end] == pad {
        while end < len && prompt_ids[end] == pad {
            end += 1;
        }
    }
    let tail_len = len.saturating_sub(end);
    if tail_len > 0
        && tail_len <= VL_FINAL_TEXT_TAIL_ABSORB_TOKENS
        && !prompt_ids[end..].contains(&pad)
    {
        return len as i32;
    }
    end as i32
}

/// Slice a MRoPE `[3, 1, S]` position-id tensor on axis 2 by a half-open
/// range `[start, stop)`. Returns `[3, 1, stop - start]`.
pub fn slice_pos_ids_axis2(pos_full: &mlx::Array, start: i32, stop: i32) -> Result<mlx::Array> {
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
pub fn slice_vision_embeds_rows(
    ve_full: &mlx::Array,
    start: usize,
    stop: usize,
) -> Result<mlx::Array> {
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

// VL-bearing methods (gated by DenseVlMethods).
// new() calls model.compute_vision_embeds + model.forward_vl_chunk (only on VL path),
// but also calls model.forward_text_hidden (which is now Model trait — works for any M).
// The new() function STILL needs DenseVlMethods because the VL branches
// (compute_vision_embeds, forward_vl_chunk) are called conditionally in its body.
