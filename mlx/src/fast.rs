//! Fused MLX kernels for Transformer inference: rms_norm, layer_norm,
//! rope, scaled_dot_product_attention.
//!
//! These are MLX's `mlx::core::fast::*` ops — single fused Metal kernels,
//! not compositions of primitives. They are the performance-critical
//! primitives for LLM/VLM inference.
//!
//! Like all ops in this crate, fast ops queue work on the caller thread's
//! current default stream. Use [`crate::set_default_stream`] to override.

use crate::{Array, Error, Result};

/// Root-mean-square normalization with optional learned scale.
///
/// `weight=None` skips the scale step (pure normalization).
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
    let w = weight.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: w is null or a borrow of `weight: &Array` valid for this call.
    let inner =
        unsafe { mlx_sys::fast::ffi::fast_rms_norm(x.as_inner(), w, eps) }.map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Layer normalization with optional learned scale and bias.
pub fn layer_norm(
    x: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
) -> Result<Array> {
    let w = weight.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let b = bias.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: w/b each null or borrow of an &Array valid for this call.
    let inner = unsafe { mlx_sys::fast::ffi::fast_layer_norm(x.as_inner(), w, b, eps) }
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
    let (has_base, base_val) = base.map_or((false, 0.0), |b| (true, b));
    let f = freqs.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
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
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
