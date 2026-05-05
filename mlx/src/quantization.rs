//! MLX low-precision subsystem: affine/NVFP4 quantization + FP8 conversion.
//!
//! Affine quantization (mode="affine"): pack high-precision weights into
//! lower-bit groups + per-group scale/bias. Used by mlx-lm 4-bit/8-bit
//! quantized models (e.g. .safetensors with `.scales` / `.biases` suffixed
//! tensor naming convention).
//!
//! NVFP4 mode (qqmm): both inputs may be quantized; scheme used by Nvidia
//! NVFP4 / MXFP4 hardware-accelerated formats.
//!
//! FP8 (E4M3): 8-bit floating-point format conversion. MLX represents FP8
//! data as a uint8 array with bytes interpreted per E4M3 layout.

use crate::{Array, Dtype, Error, Result};

/// Quantize a matrix along its last axis.
///
/// For `mode="affine"` (the default), the result is
/// `[packed_weights, scales, biases]` (3 arrays). Other modes may return
/// a different number of arrays.
pub fn quantize(
    w: &Array,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    global_scale: Option<&Array>,
) -> Result<Vec<Array>> {
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let gscale = global_scale.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: gscale is null or borrow of `global_scale: &Array` valid for this call.
    let mut result = unsafe {
        mlx_sys::quantization::ffi::quantize(w.as_inner(), has_gs, gs, has_b, b, mode, gscale)
    }
    .map_err(Error::from)?;
    let count = mlx_sys::quantization::ffi::quantize_result_count(&result);
    let mut output = Vec::with_capacity(count);
    for i in 0..count {
        let arr_ptr = mlx_sys::quantization::ffi::quantize_result_take_at(result.pin_mut(), i)
            .map_err(Error::from)?;
        output.push(Array::from_inner(arr_ptr));
    }
    Ok(output)
}

/// Inverse of [`quantize`]. Reconstructs the original-precision matrix.
#[allow(clippy::too_many_arguments)]
pub fn dequantize(
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    global_scale: Option<&Array>,
    dtype: Option<Dtype>,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let gscale = global_scale.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_dt, dt) = dtype.map_or((false, 0), |d| (true, d.as_u8()));
    // SAFETY: b_ptr/gscale each null or borrow of an &Array valid for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::dequantize(
            w.as_inner(),
            scales.as_inner(),
            b_ptr,
            has_gs,
            gs,
            has_b,
            b,
            mode,
            gscale,
            has_dt,
            dt,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Compute `x @ w` where `w` is a quantized matrix. The workhorse for
/// inference of quantized models.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    // SAFETY: b_ptr is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::quantized_matmul(
            x.as_inner(),
            w.as_inner(),
            scales.as_inner(),
            b_ptr,
            transpose,
            has_gs,
            gs,
            has_b,
            b,
            mode,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Quantized-quantized matmul. Both x and w may be quantized; default
/// mode is `"nvfp4"`.
#[allow(clippy::too_many_arguments)]
pub fn qqmm(
    x: &Array,
    w: &Array,
    w_scales: Option<&Array>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    global_scale_x: Option<&Array>,
    global_scale_w: Option<&Array>,
) -> Result<Array> {
    let ws = w_scales.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let gx = global_scale_x.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let gw = global_scale_w.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: ws/gx/gw each null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::qqmm(
            x.as_inner(),
            w.as_inner(),
            ws,
            has_gs,
            gs,
            has_b,
            b,
            mode,
            gx,
            gw,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Quantized matmul with matrix-level gather (MoE / expert routing).
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    sorted_indices: bool,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let li = lhs_indices.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let ri = rhs_indices.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    // SAFETY: b_ptr/li/ri each null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::gather_qmm(
            x.as_inner(),
            w.as_inner(),
            scales.as_inner(),
            b_ptr,
            li,
            ri,
            transpose,
            has_gs,
            gs,
            has_b,
            b,
            mode,
            sorted_indices,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
