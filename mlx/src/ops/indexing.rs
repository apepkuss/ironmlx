//! Indexing ops: `where_`, `take`, `take_along_axis`, `slice`, `slice_strided`, `gather`.

use crate::{broadcast, Array, Error, Result};

/// Element-wise conditional select: `cond ? x : y`, with NumPy broadcasting
/// across all three operands.
///
/// `cond` is typically a `bool` array but MLX accepts any numeric dtype
/// (non-zero is treated as true).
///
/// Trailing underscore in the name avoids the Rust `where` keyword.
pub fn where_(cond: &Array, x: &Array, y: &Array) -> Result<Array> {
    // Validate broadcast compatibility in two steps: cond+x, then result+y.
    // This produces structured Error::BroadcastMismatch instead of opaque MLX strings.
    let cond_x = broadcast::broadcast_shape(&cond.shape(), &x.shape())?;
    broadcast::broadcast_shape(&cond_x, &y.shape())?;
    let inner = mlx_sys::array::ffi::array_where(cond.as_inner(), x.as_inner(), y.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Take values along `axis` according to a 1-D `indices` array.
///
/// Output shape: same as `a` but with the `axis` dim replaced by `indices.size()`.
/// Indices must be an unsigned integer dtype (u32/u64); MLX validates.
pub fn take(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_take(a.as_inner(), indices.as_inner(), axis)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Take values where `indices` has the same shape as `a` (per-axis pick).
///
/// Equivalent to PyTorch's `torch.gather`. Output shape = `indices.shape`.
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_take_along_axis(a.as_inner(), indices.as_inner(), axis)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
