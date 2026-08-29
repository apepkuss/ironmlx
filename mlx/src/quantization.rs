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
//!
//! Each op exposes both a default variant (current default stream) and a
//! `*_on` variant taking `impl Into<StreamOrDevice>` (P5.7). Every op
//! carries `Option<&Array>` parameters and/or returns a `Vec<Array>` /
//! tuple shape, so the variants are written by hand and the default
//! variants delegate to `*_on(.., ())`.

use crate::{Array, Dtype, Error, Result, StreamOrDevice};

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
    quantize_on(w, group_size, bits, mode, global_scale, ())
}

/// Stream-targeted variant of [`quantize`].
pub fn quantize_on(
    w: &Array,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    global_scale: Option<&Array>,
    target: impl Into<StreamOrDevice>,
) -> Result<Vec<Array>> {
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let gscale = global_scale.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: gscale is null or borrow of `global_scale: &Array` valid for this call.
    let mut result = unsafe {
        mlx_sys::quantization::ffi::quantize(
            w.as_inner(),
            has_gs,
            gs,
            has_b,
            b,
            mode,
            gscale,
            has,
            dev_only,
            dev_t,
            idx,
        )
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
    dequantize_on(
        w,
        scales,
        biases,
        group_size,
        bits,
        mode,
        global_scale,
        dtype,
        (),
    )
}

/// Stream-targeted variant of [`dequantize`].
#[allow(clippy::too_many_arguments)]
pub fn dequantize_on(
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    global_scale: Option<&Array>,
    dtype: Option<Dtype>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let gscale = global_scale.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_dt, dt) = dtype.map_or((false, 0), |d| (true, d.as_u8()));
    let (has, dev_only, dev_t, idx) = target.into().encode();
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
            has,
            dev_only,
            dev_t,
            idx,
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
    quantized_matmul_on(x, w, scales, biases, transpose, group_size, bits, mode, ())
}

/// Stream-targeted variant of [`quantized_matmul`].
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_on(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let (has, dev_only, dev_t, idx) = target.into().encode();
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
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Compute a quantized matmul while evaluating each leading batch matrix with
/// the same matrix shape as an independent single-batch call.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_batch_isolated(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
) -> Result<Array> {
    quantized_matmul_batch_isolated_on(x, w, scales, biases, transpose, group_size, bits, mode, ())
}

/// Stream-targeted variant of [`quantized_matmul_batch_isolated`].
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_batch_isolated_on(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: b_ptr is null or borrows a valid array for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::quantized_matmul_batch_isolated(
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
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Compute a quantized matmul with a product accumulation tree that remains
/// stable across supported small batch widths.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_product_stable(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
) -> Result<Array> {
    quantized_matmul_product_stable_on(x, w, scales, biases, transpose, group_size, bits, mode, ())
}

/// Stream-targeted variant of [`quantized_matmul_product_stable`].
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_product_stable_on(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: b_ptr is null or borrows a valid array for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::quantized_matmul_product_stable(
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
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Compute a product-stable affine8 quantized matmul using the explicit
/// wide-vector Metal tiling hint.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_product_stable_affine8_wide(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
) -> Result<Array> {
    quantized_matmul_product_stable_affine8_wide_on(
        x,
        w,
        scales,
        biases,
        transpose,
        group_size,
        bits,
        mode,
        (),
    )
}

/// Stream-targeted variant of
/// [`quantized_matmul_product_stable_affine8_wide`].
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_product_stable_affine8_wide_on(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: b_ptr is null or borrows a valid array for this call.
    let inner = unsafe {
        mlx_sys::quantization::ffi::quantized_matmul_product_stable_affine8_wide(
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
            has,
            dev_only,
            dev_t,
            idx,
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
    qqmm_on(
        x,
        w,
        w_scales,
        group_size,
        bits,
        mode,
        global_scale_x,
        global_scale_w,
        (),
    )
}

/// Stream-targeted variant of [`qqmm`].
#[allow(clippy::too_many_arguments)]
pub fn qqmm_on(
    x: &Array,
    w: &Array,
    w_scales: Option<&Array>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    global_scale_x: Option<&Array>,
    global_scale_w: Option<&Array>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let ws = w_scales.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let gx = global_scale_x.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let gw = global_scale_w.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has, dev_only, dev_t, idx) = target.into().encode();
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
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Diagnostic-only C++-side timing loop for [`quantized_matmul_on`].
///
/// This is intentionally not used by production inference. It lets benchmark
/// binaries separate Rust-side loop / return-value overhead from MLX C++ kernel
/// scheduling when investigating backend regressions.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_bench_ms(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    runs: usize,
    target: impl Into<StreamOrDevice>,
) -> Result<Vec<f64>> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: b_ptr is null or borrow of `biases: &Array` valid for this call.
    unsafe {
        mlx_sys::quantization::ffi::quantized_matmul_bench_ms(
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
            runs,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)
}

/// Quantized matmul with matrix-level gather (MoE / expert routing).
#[allow(clippy::too_many_arguments)]
pub fn gather_quantized_matmul(
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
    gather_quantized_matmul_on(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        transpose,
        group_size,
        bits,
        mode,
        sorted_indices,
        (),
    )
}

/// Stream-targeted variant of [`gather_quantized_matmul`].
#[allow(clippy::too_many_arguments)]
pub fn gather_quantized_matmul_on(
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
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let b_ptr = biases.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let li = lhs_indices.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let ri = rhs_indices.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let (has_gs, gs) = group_size.map_or((false, 0), |v| (true, v));
    let (has_b, b) = bits.map_or((false, 0), |v| (true, v));
    let (has, dev_only, dev_t, idx) = target.into().encode();
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
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Convert an E4M3 float8 array to the given floating-point dtype.
pub fn from_fp8(x: &Array, dtype: Dtype) -> Result<Array> {
    from_fp8_on(x, dtype, ())
}

/// Stream-targeted variant of [`from_fp8`].
pub fn from_fp8_on(x: &Array, dtype: Dtype, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::quantization::ffi::from_fp8(
        x.as_inner(),
        dtype.as_u8(),
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Convert a floating-point matrix to E4M3 float8.
pub fn to_fp8(x: &Array) -> Result<Array> {
    to_fp8_on(x, ())
}

/// Stream-targeted variant of [`to_fp8`].
pub fn to_fp8_on(x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::quantization::ffi::to_fp8(x.as_inner(), has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
