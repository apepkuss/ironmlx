//! Fused MLX kernels for Transformer inference: rms_norm, layer_norm,
//! rope, scaled_dot_product_attention.
//!
//! These are MLX's `mlx::core::fast::*` ops — single fused Metal kernels,
//! not compositions of primitives. They are the performance-critical
//! primitives for LLM/VLM inference.
//!
//! Each op exposes both a default variant (current default stream) and a
//! `*_on` variant taking `impl Into<StreamOrDevice>` (P5.7). All five fast
//! ops carry one or more `Option<&Array>` parameters; the pointer-conversion
//! prologue plus an `unsafe` FFI call don't fit [`crate::op_with_stream!`]
//! cleanly, so the `_on` variants are written by hand and the default
//! variants delegate to `*_on(.., ())`.

pub mod metal_kernel;
mod paged_attention;
mod turboquant;

pub use metal_kernel::{DispatchBuilder, MetalKernel, MetalKernelBuilder, Set, TemplateArg, Unset};
pub use paged_attention::{
    paged_scaled_dot_product_attention_decode, paged_scaled_dot_product_attention_decode_on,
};
pub use turboquant::{
    turbo_dequantize, turbo_dequantize_on, turbo_quantize, turbo_quantize_on,
    turboquant_sdpa_decode, turboquant_sdpa_decode_on, turboquant_sdpa_decode_parallel,
    turboquant_sdpa_decode_parallel_on, turboquant_sdpa_decode_parallel_pre_rotated,
    turboquant_sdpa_decode_parallel_pre_rotated_on, turboquant_sdpa_multirow,
    turboquant_sdpa_multirow_on, TURBOQUANT_MULTIROW_MAX_QUERY_ROWS,
    TURBOQUANT_MULTIROW_MIN_SEQ_LEN, TURBOQUANT_PARALLEL_DECODE_SEQ_THRESHOLD,
};

use crate::{Array, Error, Result, StreamOrDevice};

/// Root-mean-square normalization with optional learned scale.
///
/// `weight=None` skips the scale step (pure normalization).
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
    rms_norm_on(x, weight, eps, ())
}

/// Stream-targeted variant of [`rms_norm`].
pub fn rms_norm_on(
    x: &Array,
    weight: Option<&Array>,
    eps: f32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let w = weight.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: w is null or a borrow of `weight: &Array` valid for this call.
    let inner = unsafe {
        mlx_sys::fast::ffi::fast_rms_norm(x.as_inner(), w, eps, has, dev_only, dev_t, idx)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Layer normalization with optional learned scale and bias.
pub fn layer_norm(
    x: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
) -> Result<Array> {
    layer_norm_on(x, weight, bias, eps, ())
}

/// Stream-targeted variant of [`layer_norm`].
pub fn layer_norm_on(
    x: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let w = weight.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let b = bias.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: w/b each null or borrow of an &Array valid for this call.
    let inner = unsafe {
        mlx_sys::fast::ffi::fast_layer_norm(x.as_inner(), w, b, eps, has, dev_only, dev_t, idx)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Rotary position embedding with a scalar offset (single-stream decode
/// or fixed-context prefill).
///
/// `base=None` requires `freqs=Some(_)` (precomputed frequencies);
/// `base=Some(_)` typically pairs with `freqs=None`. MLX validates the
/// combination and raises if both are missing.
pub fn rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
    freqs: Option<&Array>,
) -> Result<Array> {
    rope_on(x, dims, traditional, base, scale, offset, freqs, ())
}

/// Stream-targeted variant of [`rope`].
#[allow(clippy::too_many_arguments)]
pub fn rope_on(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
    freqs: Option<&Array>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has_base, base_val) = base.map_or((false, 0.0), |b| (true, b));
    let f = freqs.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: f is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::fast::ffi::fast_rope(
            x.as_inner(),
            dims,
            traditional,
            has_base,
            base_val,
            scale,
            offset,
            f,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// RoPE with per-batch-row offsets — for variable-length batched inference.
/// `offset` shape: `[batch]`, dtype `i32`.
pub fn rope_with_array_offset(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: &Array,
    freqs: Option<&Array>,
) -> Result<Array> {
    rope_with_array_offset_on(x, dims, traditional, base, scale, offset, freqs, ())
}

/// Stream-targeted variant of [`rope_with_array_offset`].
#[allow(clippy::too_many_arguments)]
pub fn rope_with_array_offset_on(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: &Array,
    freqs: Option<&Array>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has_base, base_val) = base.map_or((false, 0.0), |b| (true, b));
    let f = freqs.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: f is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::fast::ffi::fast_rope_with_array_offset(
            x.as_inner(),
            dims,
            traditional,
            has_base,
            base_val,
            scale,
            offset.as_inner(),
            f,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Fused scaled dot-product attention: `softmax(Q @ K.T * scale + mask) @ V`.
///
/// `mask_mode`:
/// - `""` — no implicit mask (default if `mask_arr=None`)
/// - `"causal"` — standard causal mask
/// - `"chunked_causal"` — block-causal for chunked prefill
///
/// `mask_arr=Some(_)` supplies a custom additive mask (broadcastable).
/// `sinks=Some(_)` enables attention sinks (StreamingLLM-style).
pub fn scaled_dot_product_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask_mode: &str,
    mask_arr: Option<&Array>,
    sinks: Option<&Array>,
) -> Result<Array> {
    scaled_dot_product_attention_on(queries, keys, values, scale, mask_mode, mask_arr, sinks, ())
}

/// Split-K NAX attention of a short B1 head-dim-128 query block in which
/// query row `r` sees keys `[window_start + r, keys)` (no mask array).
/// `Ok(None)` when the shape or device is not supported (or the experimental
/// attention library is not configured).
pub fn window_attention_on(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    window_start: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Option<Array>> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::fast::ffi::fast_window_attention(
        queries.as_inner(),
        keys.as_inner(),
        values.as_inner(),
        scale,
        window_start,
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok((!inner.is_null()).then(|| Array::from_inner(inner)))
}

/// Split-K D256 NAX attention of a B1 flat draft tree. `queries` are
/// `[1, Hkv, slots * gqa, 256]` node-major (two slots per 16-row part, both on
/// one root-to-leaf path); `tail_keys` / `tail_values` `[1, Hkv, parts, TL,
/// 256]` hold each part's keys from `tail_from` (a multiple of 32) in the
/// positions a linear verify of the part's path gives them; `limits` (int32
/// `[slots]`) each slot's key count. Each row equals the row of the linear
/// split-K verify of its path. `Ok(None)` when unsupported.
#[allow(clippy::too_many_arguments)]
pub fn tree_attention_on(
    queries: &Array,
    keys: &Array,
    values: &Array,
    tail_keys: &Array,
    tail_values: &Array,
    limits: &Array,
    scale: f32,
    gqa: i32,
    key_count: i32,
    tail_from: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Option<Array>> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::fast::ffi::fast_tree_attention(
        queries.as_inner(),
        keys.as_inner(),
        values.as_inner(),
        tail_keys.as_inner(),
        tail_values.as_inner(),
        limits.as_inner(),
        scale,
        gqa,
        key_count,
        tail_from,
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok((!inner.is_null()).then(|| Array::from_inner(inner)))
}

/// Keys per split-K block of a D256 decode / verify call over `keys` keys.
pub fn decode_block_keys(keys: i32) -> i32 {
    mlx_sys::fast::ffi::fast_decode_block_keys(keys)
}

/// Stream-targeted variant of [`scaled_dot_product_attention`].
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_on(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask_mode: &str,
    mask_arr: Option<&Array>,
    sinks: Option<&Array>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let m = mask_arr.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let s = sinks.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: m/s each null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::fast::ffi::fast_scaled_dot_product_attention(
            queries.as_inner(),
            keys.as_inner(),
            values.as_inner(),
            scale,
            mask_mode,
            m,
            s,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
