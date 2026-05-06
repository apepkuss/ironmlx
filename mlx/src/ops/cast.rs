//! Dtype conversion (`astype`).
//!
//! `astype` returns a new [`Array`] whose elements are cast to the requested
//! [`Dtype`]. MLX performs the conversion lazily — the actual cast kernel
//! runs at evaluation time, not at construction.
//!
//! Both the default variant ([`astype`]) and the stream-targeted variant
//! ([`astype_on`]) follow the P5.7 contract: the default delegates to the
//! `_on` form with `()` (i.e. `StreamOrDevice::Default`).

use crate::{Array, Dtype, Error, Result, StreamOrDevice};

/// Convert `a` to a new [`Array`] with the given [`Dtype`]. If `a` is already
/// of that dtype MLX still produces a fresh array node (no in-place mutation).
pub fn astype(a: &Array, dtype: Dtype) -> Result<Array> {
    astype_on(a, dtype, ())
}

/// Stream-targeted variant of [`astype`]. Pass `()` for the current default
/// stream, a `Stream`, or a `Device`.
pub fn astype_on(a: &Array, dtype: Dtype, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::astype(a.as_inner(), dtype.as_u8(), has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
