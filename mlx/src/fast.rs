//! Wrappers for MLX fast (hardware-optimized) operations.

use mlx_sys as sys;

use crate::array::Array;
use crate::error::{Result, check};
use crate::stream::Stream;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return the raw `mlx_array` handle for an optional array reference,
/// producing a zeroed (null) handle when `None`.
fn nullable_array(a: Option<&Array>) -> sys::mlx_array {
    match a {
        Some(arr) => arr.as_raw(),
        None => unsafe { std::mem::zeroed() },
    }
}

/// Convert `Option<f32>` to the C `mlx_optional_float` struct.
fn optional_float(v: Option<f32>) -> sys::mlx_optional_float_ {
    match v {
        Some(value) => sys::mlx_optional_float_ {
            value,
            has_value: true,
        },
        None => sys::mlx_optional_float_ {
            value: 0.0,
            has_value: false,
        },
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// RMSNorm — hardware-optimized implementation.
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32, stream: &Stream) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_fast_rms_norm(
            res.as_raw_mut(),
            x.as_raw(),
            nullable_array(weight),
            eps,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// LayerNorm — hardware-optimized implementation.
pub fn layer_norm(
    x: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_fast_layer_norm(
            res.as_raw_mut(),
            x.as_raw(),
            nullable_array(weight),
            nullable_array(bias),
            eps,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Rotary Position Embedding (RoPE).
#[allow(clippy::too_many_arguments)]
pub fn rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
    freqs: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_fast_rope(
            res.as_raw_mut(),
            x.as_raw(),
            dims,
            traditional,
            optional_float(base),
            scale,
            offset,
            nullable_array(freqs),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Dynamic RoPE with array offset.
#[allow(clippy::too_many_arguments)]
pub fn rope_dynamic(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: &Array,
    freqs: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_fast_rope_dynamic(
            res.as_raw_mut(),
            x.as_raw(),
            dims,
            traditional,
            optional_float(base),
            scale,
            offset.as_raw(),
            nullable_array(freqs),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Scaled dot-product attention — hardware-optimized.
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask_mode: &str,
    mask: Option<&Array>,
    sinks: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let c_mask_mode =
        std::ffi::CString::new(mask_mode).map_err(|e| crate::error::Error::Mlx(e.to_string()))?;
    let mut res = Array::new_empty();
    check(unsafe {
        sys::mlx_fast_scaled_dot_product_attention(
            res.as_raw_mut(),
            queries.as_raw(),
            keys.as_raw(),
            values.as_raw(),
            scale,
            c_mask_mode.as_ptr(),
            nullable_array(mask),
            nullable_array(sinks),
            stream.as_raw(),
        )
    })?;
    Ok(res)
}
