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

/// Slice with stride 1 along every dimension. `start` and `stop` must each have
/// length equal to `a.ndim()`. Negative indices are supported (per MLX rules).
pub fn slice(a: &Array, start: &[i32], stop: &[i32]) -> Result<Array> {
    let strides: Vec<i32> = vec![1; a.ndim()];
    slice_strided(a, start, stop, &strides)
}

/// Slice with explicit per-dim strides. `start`, `stop`, `strides` must all
/// have length equal to `a.ndim()`. Negative indices and negative strides are
/// supported per MLX rules.
pub fn slice_strided(a: &Array, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
    let ndim = a.ndim();
    if start.len() != ndim || stop.len() != ndim || strides.len() != ndim {
        let actual = vec![start.len() as i32, stop.len() as i32, strides.len() as i32];
        let expected = vec![ndim as i32; 3];
        return Err(Error::ShapeMismatch { expected, actual });
    }
    let inner = mlx_sys::array::ffi::array_slice_strided(a.as_inner(), start, stop, strides)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// N-dimensional gather. Picks slices of `a` at the cartesian product of
/// `indices` along `axes`, with each gathered slice sized per `slice_sizes`.
///
/// Returns shape `indices_shape ++ slice_sizes` (concatenation). See MLX docs
/// for full semantics — this is the most flexible / least intuitive indexing op.
pub fn gather(
    a: &Array,
    indices: &[&Array],
    axes: &[i32],
    slice_sizes: &[i32],
) -> Result<Array> {
    // Build a slice of raw pointers to bridge to the unsafe shim. Each pointer
    // is valid for the duration of this call because `indices` (a slice of
    // &Array) outlives the FFI invocation.
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        indices.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: `raw` contains valid pointers into the borrowed `&Array`s in
    // `indices`, all live for the duration of this call. The shim copies via
    // copy ctor (refcount-shared, cheap) — no aliasing or lifetime escape.
    let inner = unsafe {
        mlx_sys::array::ffi::array_gather(a.as_inner(), &raw, axes, slice_sizes)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
