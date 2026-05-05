//! Matrix multiplication.
//!
//! `matmul(a, b)` covers all NumPy-/MLX-style matmul cases:
//!
//! - 2D × 2D: standard matrix product `[M, K] @ [K, N] → [M, N]`
//! - Batched: `[B..., M, K] @ [B..., K, N] → [B..., M, N]`
//! - Broadcasting on batch dims: `[B, 1, M, K] @ [1, H, K, N] → [B, H, M, N]`
//!
//! MLX handles all dispatch internally; this is a single FFI thin wrapper.

use crate::{Array, Error, Result};

/// Matrix multiplication. See module docs for shape rules.
pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
    let inner =
        mlx_sys::array::ffi::array_matmul(a.as_inner(), b.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

// ===== P5 ops extensions =====

/// Tensor contraction over the last `axis` dims of `a` and first `axis` dims of `b`.
///
/// For 2D arrays with `axis=1`, this is equivalent to `a.matmul(b)`.
pub fn tensordot(a: &Array, b: &Array, axis: i32) -> Result<Array> {
    let inner = mlx_sys::array::ffi::tensordot_axis(a.as_inner(), b.as_inner(), axis)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Tensor contraction over arbitrary axes pairs.
///
/// Contracts `a` along `axes_a` and `b` along `axes_b`. The two axes lists
/// must have the same length, and `a.shape[axes_a[i]] == b.shape[axes_b[i]]`
/// for each i.
pub fn tensordot_axes(a: &Array, b: &Array, axes_a: &[i32], axes_b: &[i32]) -> Result<Array> {
    let inner = mlx_sys::array::ffi::tensordot_axes(a.as_inner(), b.as_inner(), axes_a, axes_b)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Outer product of two 1-D vectors. For `a` of shape `[N]` and `b` of shape
/// `[M]`, returns shape `[N, M]` with `out[i, j] = a[i] * b[j]`.
pub fn outer(a: &Array, b: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::outer(a.as_inner(), b.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Inner (dot) product of two arrays. Renamed from MLX's `inner` to avoid
/// conflicting with the project's pervasive `as_inner` / `from_inner` naming.
pub fn inner_product(a: &Array, b: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::inner(a.as_inner(), b.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Compute `D = beta * C + alpha * (A @ B)` in a single fused kernel.
pub fn addmm(c: &Array, a: &Array, b: &Array, alpha: f32, beta: f32) -> Result<Array> {
    let inner = mlx_sys::array::ffi::addmm(c.as_inner(), a.as_inner(), b.as_inner(), alpha, beta)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
