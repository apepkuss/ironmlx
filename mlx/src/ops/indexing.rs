//! Indexing ops: `where_`, `take`, `take_along_axis`, `slice`, `slice_strided`, `gather`.
//!
//! Each op exposes both a default variant and a `*_on` variant taking
//! `impl Into<StreamOrDevice>` (P5.7). Generic / pointer-slice signatures
//! and the `where_` prologue don't fit [`op_with_stream!`] cleanly, so the
//! variants are written by hand and the default delegates to `*_on(.., ())`.

use crate::{broadcast, Array, Error, IntoShape, Result, StreamOrDevice};

/// Element-wise conditional select: `cond ? x : y`, with NumPy broadcasting
/// across all three operands.
///
/// `cond` is typically a `bool` array but MLX accepts any numeric dtype
/// (non-zero is treated as true).
///
/// Trailing underscore in the name avoids the Rust `where` keyword.
pub fn where_(cond: &Array, x: &Array, y: &Array) -> Result<Array> {
    where_on(cond, x, y, ())
}

/// Stream-targeted variant of [`where_`].
pub fn where_on(
    cond: &Array,
    x: &Array,
    y: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    // Validate broadcast compatibility in two steps: cond+x, then result+y.
    // This produces structured Error::BroadcastMismatch instead of opaque MLX strings.
    let cond_x = broadcast::broadcast_shape(cond.shape().as_slice(), x.shape().as_slice())?;
    broadcast::broadcast_shape(&cond_x, y.shape().as_slice())?;
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::array_where(
        cond.as_inner(),
        x.as_inner(),
        y.as_inner(),
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Take values along `axis` according to a 1-D `indices` array.
///
/// Output shape: same as `a` but with the `axis` dim replaced by `indices.size()`.
/// Indices must be an unsigned integer dtype (u32/u64); MLX validates.
pub fn take(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    take_on(a, indices, axis, ())
}

/// Stream-targeted variant of [`take`].
pub fn take_on(
    a: &Array,
    indices: &Array,
    axis: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::array_take(
        a.as_inner(),
        indices.as_inner(),
        axis,
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Take values where `indices` has the same shape as `a` (per-axis pick).
///
/// Equivalent to PyTorch's `torch.gather`. Output shape = `indices.shape`.
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    take_along_axis_on(a, indices, axis, ())
}

/// Stream-targeted variant of [`take_along_axis`].
pub fn take_along_axis_on(
    a: &Array,
    indices: &Array,
    axis: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::array_take_along_axis(
        a.as_inner(),
        indices.as_inner(),
        axis,
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Slice with stride 1 along every dimension. `start` and `stop` must each have
/// length equal to `a.ndim()`. Negative indices are supported (per MLX rules).
pub fn slice<S1: IntoShape, S2: IntoShape>(a: &Array, start: S1, stop: S2) -> Result<Array> {
    slice_on(a, start, stop, ())
}

/// Stream-targeted variant of [`slice`].
pub fn slice_on<S1: IntoShape, S2: IntoShape>(
    a: &Array,
    start: S1,
    stop: S2,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let strides: Vec<i32> = vec![1; a.ndim()];
    let start = start.into_shape();
    let stop = stop.into_shape();
    slice_strided_inner(
        a,
        start.as_slice(),
        stop.as_slice(),
        &strides,
        target.into(),
    )
}

/// Slice with explicit per-dim strides. `start`, `stop`, `strides` must all
/// have length equal to `a.ndim()`. Negative indices and negative strides are
/// supported per MLX rules.
pub fn slice_strided<S1: IntoShape, S2: IntoShape, S3: IntoShape>(
    a: &Array,
    start: S1,
    stop: S2,
    strides: S3,
) -> Result<Array> {
    slice_strided_on(a, start, stop, strides, ())
}

/// Stream-targeted variant of [`slice_strided`].
pub fn slice_strided_on<S1: IntoShape, S2: IntoShape, S3: IntoShape>(
    a: &Array,
    start: S1,
    stop: S2,
    strides: S3,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let start = start.into_shape();
    let stop = stop.into_shape();
    let strides = strides.into_shape();
    slice_strided_inner(
        a,
        start.as_slice(),
        stop.as_slice(),
        strides.as_slice(),
        target.into(),
    )
}

fn slice_strided_inner(
    a: &Array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    target: StreamOrDevice,
) -> Result<Array> {
    let ndim = a.ndim();
    if start.len() != ndim || stop.len() != ndim || strides.len() != ndim {
        return Err(Error::Mlx(format!(
            "slice: start/stop/strides length must equal ndim={ndim}, got {}/{}/{}",
            start.len(),
            stop.len(),
            strides.len()
        )));
    }
    let (has, dev_only, dev_t, idx) = target.encode();
    let inner = mlx_sys::array::ffi::array_slice_strided(
        a.as_inner(),
        start,
        stop,
        strides,
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// N-dimensional gather. Picks slices of `a` at the cartesian product of
/// `indices` along `axes`, with each gathered slice sized per `slice_sizes`.
///
/// Returns shape `indices_shape ++ slice_sizes` (concatenation). See MLX docs
/// for full semantics — this is the most flexible / least intuitive indexing op.
pub fn gather(a: &Array, indices: &[&Array], axes: &[i32], slice_sizes: &[i32]) -> Result<Array> {
    gather_on(a, indices, axes, slice_sizes, ())
}

/// Stream-targeted variant of [`gather`].
pub fn gather_on(
    a: &Array,
    indices: &[&Array],
    axes: &[i32],
    slice_sizes: &[i32],
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    // Build a slice of raw pointers to bridge to the unsafe shim. Each pointer
    // is valid for the duration of this call because `indices` (a slice of
    // &Array) outlives the FFI invocation.
    let raw: Vec<*const mlx_sys::array::ffi::MlxArray> =
        indices.iter().map(|a| a.as_inner() as *const _).collect();
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: `raw` contains valid pointers into the borrowed `&Array`s in
    // `indices`, all live for the duration of this call. The shim copies via
    // copy ctor (refcount-shared, cheap) — no aliasing or lifetime escape.
    let inner = unsafe {
        mlx_sys::array::ffi::array_gather(
            a.as_inner(),
            &raw,
            axes,
            slice_sizes,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
