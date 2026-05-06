//! Matrix multiplication.
//!
//! `matmul(a, b)` covers all NumPy-/MLX-style matmul cases:
//!
//! - 2D × 2D: standard matrix product `[M, K] @ [K, N] → [M, N]`
//! - Batched: `[B..., M, K] @ [B..., K, N] → [B..., M, N]`
//! - Broadcasting on batch dims: `[B, 1, M, K] @ [1, H, K, N] → [B, H, M, N]`
//!
//! MLX handles all dispatch internally; this is a single FFI thin wrapper.
//!
//! Each op exposes both a default variant (current default stream) and a
//! `*_on` variant taking `impl Into<StreamOrDevice>` (P5.7). Simple
//! signatures (matmul/tensordot/outer/inner_product/addmm/tensordot_axes/
//! segmented_matmul) use [`op_with_stream!`]; ops with `Option<&Array>`
//! (block_masked_matmul / gather_matmul) are written by hand because the
//! pointer-conversion prologue plus an unsafe FFI call don't fit the macro.

use crate::{Array, Error, Result, StreamOrDevice};

op_with_stream! {
    /// Matrix multiplication. See module docs for shape rules.
    pub fn matmul(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_matmul(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Tensor contraction over the last `axis` dims of `a` and first `axis` dims of `b`.
    ///
    /// For 2D arrays with `axis=1`, this is equivalent to `a.matmul(b)`.
    pub fn tensordot(a: &Array, b: &Array, axis: i32) -> Result<Array>
        => mlx_sys::array::ffi::tensordot_axis(a.as_inner(), b.as_inner(), axis);
}

op_with_stream! {
    /// Tensor contraction over arbitrary axes pairs.
    ///
    /// Contracts `a` along `axes_a` and `b` along `axes_b`. The two axes lists
    /// must have the same length, and `a.shape[axes_a[i]] == b.shape[axes_b[i]]`
    /// for each i.
    pub fn tensordot_axes(a: &Array, b: &Array, axes_a: &[i32], axes_b: &[i32]) -> Result<Array>
        => mlx_sys::array::ffi::tensordot_axes(a.as_inner(), b.as_inner(), axes_a, axes_b);
}

op_with_stream! {
    /// Outer product of two 1-D vectors. For `a` of shape `[N]` and `b` of shape
    /// `[M]`, returns shape `[N, M]` with `out[i, j] = a[i] * b[j]`.
    pub fn outer(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::outer(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Inner (dot) product of two arrays. Renamed from MLX's `inner` to avoid
    /// conflicting with the project's pervasive `as_inner` / `from_inner` naming.
    pub fn inner_product(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::inner(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Compute `D = beta * C + alpha * (A @ B)` in a single fused kernel.
    pub fn addmm(c: &Array, a: &Array, b: &Array, alpha: f32, beta: f32) -> Result<Array>
        => mlx_sys::array::ffi::addmm(c.as_inner(), a.as_inner(), b.as_inner(), alpha, beta);
}

op_with_stream! {
    /// Matrix product with segmented inner dimension. `segments` is an i32
    /// array describing how the inner dimension is partitioned across batches.
    pub fn segmented_matmul(a: &Array, b: &Array, segments: &Array) -> Result<Array>
        => mlx_sys::array::ffi::segmented_mm(a.as_inner(), b.as_inner(), segments.as_inner());
}

/// Block-masked matrix product. Each of the 3 masks is optional and applies
/// at block granularity (`block_size`).
pub fn block_masked_matmul(
    a: &Array,
    b: &Array,
    block_size: i32,
    mask_out: Option<&Array>,
    mask_lhs: Option<&Array>,
    mask_rhs: Option<&Array>,
) -> Result<Array> {
    block_masked_matmul_on(a, b, block_size, mask_out, mask_lhs, mask_rhs, ())
}

/// Stream-targeted variant of [`block_masked_matmul`].
pub fn block_masked_matmul_on(
    a: &Array,
    b: &Array,
    block_size: i32,
    mask_out: Option<&Array>,
    mask_lhs: Option<&Array>,
    mask_rhs: Option<&Array>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let mo = mask_out.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let ml = mask_lhs.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let mr = mask_rhs.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: mo/ml/mr each null or borrow of an &Array valid for this call.
    let inner = unsafe {
        mlx_sys::array::ffi::block_masked_mm(
            a.as_inner(),
            b.as_inner(),
            block_size,
            mo,
            ml,
            mr,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Matrix product with row-level gather. **Non-quantized** version.
/// For the quantized counterpart, see `mlx::quantization::gather_quantized_matmul` (P3).
pub fn gather_matmul(
    a: &Array,
    b: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted_indices: bool,
) -> Result<Array> {
    gather_matmul_on(a, b, lhs_indices, rhs_indices, sorted_indices, ())
}

/// Stream-targeted variant of [`gather_matmul`].
pub fn gather_matmul_on(
    a: &Array,
    b: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted_indices: bool,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let li = lhs_indices.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let ri = rhs_indices.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: li/ri each null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::array::ffi::gather_mm(
            a.as_inner(),
            b.as_inner(),
            li,
            ri,
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
