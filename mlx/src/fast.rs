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
