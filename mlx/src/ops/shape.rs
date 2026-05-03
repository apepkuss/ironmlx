//! Shape ops: reshape, transpose family, broadcast_to, concatenate, stack, split.

use smallvec::SmallVec;

use crate::{Array, Error, Result};

/// Reshape an array to the given shape. A single `-1` in the shape is replaced
/// by the inferred size; multiple `-1`s or a non-divisible product return
/// `Err(Error::Mlx)`.
pub fn reshape(a: &Array, shape: &[i32]) -> Result<Array> {
    let total: usize = a.size();
    let neg_count = shape.iter().filter(|&&d| d == -1).count();
    let resolved: SmallVec<[i32; 8]> = match neg_count {
        0 => shape.iter().copied().collect(),
        1 => {
            let known: usize = shape
                .iter()
                .filter(|&&d| d != -1)
                .map(|&d| d as usize)
                .product();
            if known == 0 || !total.is_multiple_of(known) {
                return Err(Error::Mlx(format!(
                    "reshape: cannot infer -1 dim — total {total} not divisible by product {known} of remaining dims {shape:?}"
                )));
            }
            let inferred = (total / known) as i32;
            shape
                .iter()
                .map(|&d| if d == -1 { inferred } else { d })
                .collect()
        }
        _ => {
            return Err(Error::Mlx(format!(
                "reshape: at most one -1 placeholder allowed, got {neg_count} in {shape:?}"
            )))
        }
    };
    let inner =
        mlx_sys::array::ffi::array_reshape(a.as_inner(), &resolved).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Reverse all axes (NumPy `arr.T` equivalent).
pub fn transpose(a: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_transpose(a.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Permute axes per the given permutation. `axes` must be a permutation of
/// `[0, a.ndim())`; MLX validates and errors otherwise.
pub fn transpose_axes(a: &Array, axes: &[i32]) -> Result<Array> {
    let inner =
        mlx_sys::array::ffi::array_transpose_axes(a.as_inner(), axes).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Broadcast `a` to the given shape, replicating dims of size 1. The target
/// shape must be broadcast-compatible per NumPy rules.
pub fn broadcast_to(a: &Array, shape: &[i32]) -> Result<Array> {
    let inner =
        mlx_sys::array::ffi::array_broadcast_to(a.as_inner(), shape).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Concatenate arrays along the given axis. All arrays must have identical
/// shape except along the concatenation axis.
pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
    // Build a slice of raw pointers to bridge to the unsafe shim. Each pointer
    // is valid for the duration of this call because `arrays` (a slice of
    // `&Array`) outlives the FFI invocation.
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: `raw` contains valid pointers into the borrowed `&Array`s in
    // `arrays`, all live for the duration of this call. The shim copies via
    // copy ctor (refcount-shared, cheap) — no aliasing or lifetime escape.
    let inner = unsafe { mlx_sys::array::ffi::array_concatenate(&raw, axis) }
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Stack arrays along a new axis. All arrays must have identical shape; the
/// result has rank `arrays[0].ndim() + 1`.
pub fn stack(arrays: &[&Array], axis: i32) -> Result<Array> {
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    // SAFETY: same as `concatenate` — pointers are bounded by call lifetime.
    let inner = unsafe { mlx_sys::array::ffi::array_stack(&raw, axis) }.map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Split `a` into `num_splits` equal-sized pieces along `axis`. Returns a
/// `Vec<Array>` of length `num_splits`. The split axis size must be evenly
/// divisible by `num_splits`; MLX validates and errors otherwise.
pub fn split_n(a: &Array, num_splits: i32, axis: i32) -> Result<Vec<Array>> {
    let v = mlx_sys::array::ffi::array_split_n(a.as_inner(), num_splits, axis)
        .map_err(Error::from)?;
    let len = mlx_sys::array::ffi::split_result_len(&v);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let inner = mlx_sys::array::ffi::split_result_at(&v, i).map_err(Error::from)?;
        out.push(Array::from_inner(inner));
    }
    Ok(out)
}

/// Split `a` at the given indices along `axis`. With `indices = [i, j, ...]`
/// and the split axis size `S`, the result has pieces with sizes
/// `[i, j-i, ..., S - last_idx]`.
pub fn split_at(a: &Array, indices: &[i32], axis: i32) -> Result<Vec<Array>> {
    let v = mlx_sys::array::ffi::array_split_at(a.as_inner(), indices, axis)
        .map_err(Error::from)?;
    let len = mlx_sys::array::ffi::split_result_len(&v);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let inner = mlx_sys::array::ffi::split_result_at(&v, i).map_err(Error::from)?;
        out.push(Array::from_inner(inner));
    }
    Ok(out)
}
