//! Shape ops: reshape, transpose family, broadcast_to, concatenate, stack, split.
//!
//! Each op exposes both a default variant (current default stream) and a
//! `*_on` variant taking `impl Into<StreamOrDevice>` (P5.7). Because most
//! shape ops are generic over `IntoShape` or take pointer-slice / `Vec<Array>`
//! return shapes that don't fit cleanly into [`op_with_stream!`], the
//! variants are written by hand and the default delegates to `*_on(.., ())`.

use smallvec::SmallVec;

use crate::ops::reduction::IntoAxes;
use crate::{Array, Error, IntoShape, Result, StreamOrDevice};

/// Reshape an array to the given shape. A single `-1` in the shape is replaced
/// by the inferred size; multiple `-1`s or a non-divisible product return
/// `Err(Error::Mlx)`.
pub fn reshape<S: IntoShape>(a: &Array, shape: S) -> Result<Array> {
    reshape_on(a, shape, ())
}

/// Stream-targeted variant of [`reshape`].
pub fn reshape_on<S: IntoShape>(
    a: &Array,
    shape: S,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let shape = shape.into_shape();
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
                    "reshape: cannot infer -1 dim — total {total} not divisible by product {known} of remaining dims {shape}"
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
                "reshape: at most one -1 placeholder allowed, got {neg_count} in {shape}"
            )))
        }
    };
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner =
        mlx_sys::array::ffi::array_reshape(a.as_inner(), &resolved, has, dev_only, dev_t, idx)
            .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Reverse all axes (NumPy `arr.T` equivalent).
pub fn transpose(a: &Array) -> Result<Array> {
    transpose_on(a, ())
}

/// Stream-targeted variant of [`transpose`].
pub fn transpose_on(a: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::array_transpose(a.as_inner(), has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Ensure `a` has contiguous storage. If `allow_col_major` is true, MLX may
/// keep an existing column-contiguous buffer instead of copying.
pub fn contiguous(a: &Array, allow_col_major: bool) -> Result<Array> {
    contiguous_on(a, allow_col_major, ())
}

/// Stream-targeted variant of [`contiguous`].
pub fn contiguous_on(
    a: &Array,
    allow_col_major: bool,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::array_contiguous(
        a.as_inner(),
        allow_col_major,
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Permute axes per the given permutation. `axes` must be a permutation of
/// `[0, a.ndim())`; MLX validates and errors otherwise.
pub fn transpose_axes<S: IntoShape>(a: &Array, axes: S) -> Result<Array> {
    transpose_axes_on(a, axes, ())
}

/// Stream-targeted variant of [`transpose_axes`].
pub fn transpose_axes_on<S: IntoShape>(
    a: &Array,
    axes: S,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let axes = axes.into_shape();
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::array_transpose_axes(
        a.as_inner(),
        axes.as_slice(),
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Broadcast `a` to the given shape, replicating dims of size 1. The target
/// shape must be broadcast-compatible per NumPy rules.
pub fn broadcast_to<S: IntoShape>(a: &Array, shape: S) -> Result<Array> {
    broadcast_to_on(a, shape, ())
}

/// Stream-targeted variant of [`broadcast_to`].
pub fn broadcast_to_on<S: IntoShape>(
    a: &Array,
    shape: S,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let shape = shape.into_shape();
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::array_broadcast_to(
        a.as_inner(),
        shape.as_slice(),
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Concatenate arrays along the given axis. All arrays must have identical
/// shape except along the concatenation axis.
pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
    concatenate_on(arrays, axis, ())
}

/// Stream-targeted variant of [`concatenate`].
pub fn concatenate_on(
    arrays: &[&Array],
    axis: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    // Build a slice of raw pointers to bridge to the unsafe shim. Each pointer
    // is valid for the duration of this call because `arrays` (a slice of
    // `&Array`) outlives the FFI invocation.
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: `raw` contains valid pointers into the borrowed `&Array`s in
    // `arrays`, all live for the duration of this call. The shim copies via
    // copy ctor (refcount-shared, cheap) — no aliasing or lifetime escape.
    let inner =
        unsafe { mlx_sys::array::ffi::array_concatenate(&raw, axis, has, dev_only, dev_t, idx) }
            .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Stack arrays along a new axis. All arrays must have identical shape; the
/// result has rank `arrays[0].ndim() + 1`.
pub fn stack(arrays: &[&Array], axis: i32) -> Result<Array> {
    stack_on(arrays, axis, ())
}

/// Stream-targeted variant of [`stack`].
pub fn stack_on(arrays: &[&Array], axis: i32, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        arrays.iter().map(|a| a.as_inner() as *const _).collect();
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: same as `concatenate_on` — pointers are bounded by call lifetime.
    let inner = unsafe { mlx_sys::array::ffi::array_stack(&raw, axis, has, dev_only, dev_t, idx) }
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Split `a` into `num_splits` equal-sized pieces along `axis`. Returns a
/// `Vec<Array>` of length `num_splits`. The split axis size must be evenly
/// divisible by `num_splits`; MLX validates and errors otherwise.
pub fn split_n(a: &Array, num_splits: i32, axis: i32) -> Result<Vec<Array>> {
    split_n_on(a, num_splits, axis, ())
}

/// Stream-targeted variant of [`split_n`].
pub fn split_n_on(
    a: &Array,
    num_splits: i32,
    axis: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Vec<Array>> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let v = mlx_sys::array::ffi::array_split_n(
        a.as_inner(),
        num_splits,
        axis,
        has,
        dev_only,
        dev_t,
        idx,
    )
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
    split_at_on(a, indices, axis, ())
}

/// Stream-targeted variant of [`split_at`].
pub fn split_at_on(
    a: &Array,
    indices: &[i32],
    axis: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Vec<Array>> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let v =
        mlx_sys::array::ffi::array_split_at(a.as_inner(), indices, axis, has, dev_only, dev_t, idx)
            .map_err(Error::from)?;
    let len = mlx_sys::array::ffi::split_result_len(&v);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let inner = mlx_sys::array::ffi::split_result_at(&v, i).map_err(Error::from)?;
        out.push(Array::from_inner(inner));
    }
    Ok(out)
}

// === P5.5 expand_dims / squeeze (axis-driven shape ops) ===
//
// Both ops route through the [`IntoAxes`] trait — same dispatch convention
// as `softmax` and the reductions. Semantics of [`crate::ops::All`]:
//
// - `squeeze(a, All)` -> drop every size-1 dim (MLX's no-axis overload).
// - `expand_dims(a, All)` is **illegal**: MLX rejects an empty axes vector
//   for expand_dims (every call must specify at least one new axis index).
//   We document the requirement and let MLX surface the error instead of
//   adding a Rust-side check, so the failure mode matches the C++ API
//   exactly (`Error::Mlx`).

/// Insert size-1 dimensions at the given axes. Indices are interpreted
/// **after** insertion: `expand_dims(a, [0, -1])` on a rank-2 array yields
/// rank-4 with size-1 dims at the front and back.
///
/// Passing [`crate::ops::All`] is illegal — surfaces as `Err(Error::Mlx)`.
pub fn expand_dims<A: IntoAxes>(a: &Array, axes: A) -> Result<Array> {
    expand_dims_on(a, axes, ())
}

/// Stream-targeted variant of [`expand_dims`].
pub fn expand_dims_on<A: IntoAxes>(
    a: &Array,
    axes: A,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // `as_axes()` returns `None` for `All` (empty slice) or `Some(slice)`
    // for specific axes. The shim forwards both cases verbatim; for the
    // empty case MLX raises an error (illegal for expand_dims).
    let slice = axes.as_axes().unwrap_or(&[]);
    let inner = mlx_sys::array::ffi::expand_dims(a.as_inner(), slice, has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Remove size-1 dimensions. With [`crate::ops::All`], drops every size-1
/// dim; with explicit axes, drops only those (MLX errors if any selected
/// axis is not size-1).
pub fn squeeze<A: IntoAxes>(a: &Array, axes: A) -> Result<Array> {
    squeeze_on(a, axes, ())
}

/// Stream-targeted variant of [`squeeze`].
pub fn squeeze_on<A: IntoAxes>(
    a: &Array,
    axes: A,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // For `All` (empty slice) the shim takes the no-axis MLX overload
    // (drop every size-1 dim); otherwise it forwards the explicit axes.
    let slice = axes.as_axes().unwrap_or(&[]);
    let inner = mlx_sys::array::ffi::squeeze(a.as_inner(), slice, has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

// === P5.6 shape 补完 (flatten/repeat) ===

op_with_stream! {
    /// Flatten dims `[start_axis, end_axis]` (inclusive, negative indices
    /// allowed) into one. Use `start_axis=0, end_axis=-1` to fully flatten
    /// to 1D.
    pub fn flatten(a: &Array, start_axis: i32, end_axis: i32) -> Result<Array>
        => mlx_sys::array::ffi::flatten(a.as_inner(), start_axis, end_axis);
}

op_with_stream! {
    /// Repeat `a` `repeats` times along `axis`. The output's `axis` size
    /// becomes `a.shape()[axis] * repeats`.
    pub fn repeat(a: &Array, repeats: i32, axis: i32) -> Result<Array>
        => mlx_sys::array::ffi::repeat(a.as_inner(), repeats, axis);
}
