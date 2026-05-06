//! Array constructor ops (P5.5 Task 4).
//!
//! These free functions cover the MLX constructor surface:
//!
//! - **Range constructors** ([`arange`], [`linspace`]) — scalar inputs
//!   describing how to fill a 1-D array.
//! - **Shape-driven fillers** ([`ones`], [`full`]) — generic over
//!   [`IntoShape`] so callers can pass tuples / slices / `Vec<i32>`.
//! - **Reference-shaped fillers** ([`ones_like`], [`zeros_like`],
//!   [`full_like`]) — take their shape (and dtype, where applicable) from an
//!   existing [`Array`].
//! - **Diagonal/triangular** ([`eye`], [`identity`], [`tri`]) — matrix
//!   builders driven by `(n, m, k)` and a [`Dtype`].
//! - **Triangular masks** ([`tril`], [`triu`]) — operators that zero out one
//!   half of an existing array.
//!
//! Each function exposes both a default variant (current default stream) and
//! a `*_on` variant taking `impl Into<StreamOrDevice>` per the P5.7
//! contract.
//!
//! `Array::zeros` is intentionally *not* re-emitted here — it already lives
//! as a `Self`-returning constructor on [`Array`] (`mlx/src/array.rs`).

use crate::{Array, Dtype, Error, IntoShape, Result, StreamOrDevice};

/// Evenly-spaced values in `[start, stop)` with the given `step`. Empty if
/// `step` does not advance from `start` toward `stop`.
pub fn arange(start: f64, stop: f64, step: f64, dtype: Dtype) -> Result<Array> {
    arange_on(start, stop, step, dtype, ())
}

/// Stream-targeted variant of [`arange`].
pub fn arange_on(
    start: f64,
    stop: f64,
    step: f64,
    dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner =
        mlx_sys::array::ffi::arange(start, stop, step, dtype.as_u8(), has, dev_only, dev_t, idx)
            .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// `num` evenly-spaced values from `start` to `stop` (both endpoints
/// included).
pub fn linspace(start: f64, stop: f64, num: i32, dtype: Dtype) -> Result<Array> {
    linspace_on(start, stop, num, dtype, ())
}

/// Stream-targeted variant of [`linspace`].
pub fn linspace_on(
    start: f64,
    stop: f64,
    num: i32,
    dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner =
        mlx_sys::array::ffi::linspace(start, stop, num, dtype.as_u8(), has, dev_only, dev_t, idx)
            .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Array of the given shape filled with ones.
pub fn ones<S: IntoShape>(shape: S, dtype: Dtype) -> Result<Array> {
    ones_on(shape, dtype, ())
}

/// Stream-targeted variant of [`ones`].
pub fn ones_on<S: IntoShape>(
    shape: S,
    dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let s = shape.into_shape();
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::ones(s.as_slice(), dtype.as_u8(), has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Array with the same shape and dtype as `a`, filled with ones.
pub fn ones_like(a: &Array) -> Result<Array> {
    ones_like_on(a, ())
}

/// Stream-targeted variant of [`ones_like`].
pub fn ones_like_on(a: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::ones_like(a.as_inner(), has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Array with the same shape and dtype as `a`, filled with zeros.
pub fn zeros_like(a: &Array) -> Result<Array> {
    zeros_like_on(a, ())
}

/// Stream-targeted variant of [`zeros_like`].
pub fn zeros_like_on(a: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::zeros_like(a.as_inner(), has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Array of the given shape filled with `vals` (broadcast as needed) and
/// converted to `dtype`.
pub fn full<S: IntoShape>(shape: S, vals: &Array, dtype: Dtype) -> Result<Array> {
    full_on(shape, vals, dtype, ())
}

/// Stream-targeted variant of [`full`].
pub fn full_on<S: IntoShape>(
    shape: S,
    vals: &Array,
    dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let s = shape.into_shape();
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::full(
        s.as_slice(),
        vals.as_inner(),
        dtype.as_u8(),
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Array with the same shape (and dtype) as `a`, filled with `vals`
/// (broadcast as needed).
pub fn full_like(a: &Array, vals: &Array) -> Result<Array> {
    full_like_on(a, vals, ())
}

/// Stream-targeted variant of [`full_like`].
pub fn full_like_on(a: &Array, vals: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner =
        mlx_sys::array::ffi::full_like(a.as_inner(), vals.as_inner(), has, dev_only, dev_t, idx)
            .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// `n × m` matrix with ones on the diagonal offset by `k` and zeros
/// everywhere else.
pub fn eye(n: i32, m: i32, k: i32, dtype: Dtype) -> Result<Array> {
    eye_on(n, m, k, dtype, ())
}

/// Stream-targeted variant of [`eye`].
pub fn eye_on(
    n: i32,
    m: i32,
    k: i32,
    dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::eye(n, m, k, dtype.as_u8(), has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Square `n × n` identity matrix.
pub fn identity(n: i32, dtype: Dtype) -> Result<Array> {
    identity_on(n, dtype, ())
}

/// Stream-targeted variant of [`identity`].
pub fn identity_on(n: i32, dtype: Dtype, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::identity(n, dtype.as_u8(), has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// `n × m` lower-triangular mask: ones on and below the diagonal offset by
/// `k`, zeros above.
pub fn tri(n: i32, m: i32, k: i32, dtype: Dtype) -> Result<Array> {
    tri_on(n, m, k, dtype, ())
}

/// Stream-targeted variant of [`tri`].
pub fn tri_on(
    n: i32,
    m: i32,
    k: i32,
    dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::tri(n, m, k, dtype.as_u8(), has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Lower triangular part of `x`: zero out elements strictly above the
/// diagonal offset by `k`.
pub fn tril(x: &Array, k: i32) -> Result<Array> {
    tril_on(x, k, ())
}

/// Stream-targeted variant of [`tril`].
pub fn tril_on(x: &Array, k: i32, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::tril(x.as_inner(), k, has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Upper triangular part of `x`: zero out elements strictly below the
/// diagonal offset by `k`.
pub fn triu(x: &Array, k: i32) -> Result<Array> {
    triu_on(x, k, ())
}

/// Stream-targeted variant of [`triu`].
pub fn triu_on(x: &Array, k: i32, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::triu(x.as_inner(), k, has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
