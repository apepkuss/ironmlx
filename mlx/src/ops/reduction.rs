//! Reduction ops (`sum`/`mean`/`max`/`min`/`argmax`) over array axes.
//!
//! Axes are passed via the `IntoAxes` sealed trait, accepting:
//!
//! - [`All`] — reduce over every axis (returns a scalar by default; or shape
//!   `[1, 1, ...]` if `keepdim` is true)
//! - `i32` — reduce a single axis (negative supported)
//! - `&[i32]` / `Vec<i32>` / `[i32; N]` — reduce multiple axes
//!
//! Keepdim is a positional `bool` (NumPy/PyTorch convention). When `true`,
//! reduced axes are kept as size-1 to preserve broadcast compatibility.
//!
//! Each reduction op exposes both a default variant (current default stream)
//! and a `*_on` variant taking `impl Into<StreamOrDevice>` (P5.7). Because
//! reductions dispatch to one of three FFI functions at runtime depending on
//! `axes`, the macro form does not fit cleanly — both variants are written
//! by hand and the default delegates to `*_on(.., ())`.

use crate::{Array, Error, Result, StreamOrDevice};

/// Marker for "reduce over all axes". Use as `sum(&a, All, false)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct All;

mod sealed {
    pub trait Sealed {}
}

/// Sealed trait describing how an argument is interpreted as reduction axes.
///
/// Implemented for [`All`], `i32`, `&[i32]`, `Vec<i32>`, and `[i32; N]`.
/// External crates cannot implement this trait.
pub trait IntoAxes: sealed::Sealed {
    /// Returns `None` for the all-axes case, or `Some(slice)` for specific axes.
    /// Internal — used by the reduction dispatchers to pick the matching shim.
    #[doc(hidden)]
    fn as_axes(&self) -> Option<&[i32]>;
}

impl sealed::Sealed for All {}
impl IntoAxes for All {
    fn as_axes(&self) -> Option<&[i32]> {
        None
    }
}

impl sealed::Sealed for i32 {}
impl IntoAxes for i32 {
    fn as_axes(&self) -> Option<&[i32]> {
        Some(std::slice::from_ref(self))
    }
}

impl sealed::Sealed for &[i32] {}
impl IntoAxes for &[i32] {
    fn as_axes(&self) -> Option<&[i32]> {
        Some(self)
    }
}

impl sealed::Sealed for Vec<i32> {}
impl IntoAxes for Vec<i32> {
    fn as_axes(&self) -> Option<&[i32]> {
        Some(self.as_slice())
    }
}

impl<const N: usize> sealed::Sealed for [i32; N] {}
impl<const N: usize> IntoAxes for [i32; N] {
    fn as_axes(&self) -> Option<&[i32]> {
        Some(self.as_slice())
    }
}

/// Sum over the specified axes.
///
/// Pass [`All`] to reduce over every axis (yielding a scalar by default),
/// `i32` for a single axis (negative indexing supported), or `&[i32]` /
/// `Vec<i32>` / `[i32; N]` for multiple axes. `keepdim = true` retains
/// reduced axes as size-1.
pub fn sum<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    sum_on(a, axes, keepdim, ())
}

/// Stream-targeted variant of [`sum`]. Pass `()` for the current default
/// stream, a `Stream`, or a `Device`.
pub fn sum_on<A: IntoAxes>(
    a: &Array,
    axes: A,
    keepdim: bool,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = match axes.as_axes() {
        None => {
            mlx_sys::array::ffi::array_sum_all(a.as_inner(), keepdim, has, dev_only, dev_t, idx)
        }
        Some([axis]) => mlx_sys::array::ffi::array_sum_axis(
            a.as_inner(),
            *axis,
            keepdim,
            has,
            dev_only,
            dev_t,
            idx,
        ),
        Some(axes) => mlx_sys::array::ffi::array_sum_axes(
            a.as_inner(),
            axes,
            keepdim,
            has,
            dev_only,
            dev_t,
            idx,
        ),
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Mean over the specified axes. See [`sum`] for axes semantics.
pub fn mean<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    mean_on(a, axes, keepdim, ())
}

/// Stream-targeted variant of [`mean`].
pub fn mean_on<A: IntoAxes>(
    a: &Array,
    axes: A,
    keepdim: bool,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = match axes.as_axes() {
        None => {
            mlx_sys::array::ffi::array_mean_all(a.as_inner(), keepdim, has, dev_only, dev_t, idx)
        }
        Some([axis]) => mlx_sys::array::ffi::array_mean_axis(
            a.as_inner(),
            *axis,
            keepdim,
            has,
            dev_only,
            dev_t,
            idx,
        ),
        Some(axes) => mlx_sys::array::ffi::array_mean_axes(
            a.as_inner(),
            axes,
            keepdim,
            has,
            dev_only,
            dev_t,
            idx,
        ),
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Maximum over the specified axes. See [`sum`] for axes semantics.
pub fn max<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    max_on(a, axes, keepdim, ())
}

/// Stream-targeted variant of [`max`].
pub fn max_on<A: IntoAxes>(
    a: &Array,
    axes: A,
    keepdim: bool,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = match axes.as_axes() {
        None => {
            mlx_sys::array::ffi::array_max_all(a.as_inner(), keepdim, has, dev_only, dev_t, idx)
        }
        Some([axis]) => mlx_sys::array::ffi::array_max_axis(
            a.as_inner(),
            *axis,
            keepdim,
            has,
            dev_only,
            dev_t,
            idx,
        ),
        Some(axes) => mlx_sys::array::ffi::array_max_axes(
            a.as_inner(),
            axes,
            keepdim,
            has,
            dev_only,
            dev_t,
            idx,
        ),
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Minimum over the specified axes. See [`sum`] for axes semantics.
pub fn min<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    min_on(a, axes, keepdim, ())
}

/// Stream-targeted variant of [`min`].
pub fn min_on<A: IntoAxes>(
    a: &Array,
    axes: A,
    keepdim: bool,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = match axes.as_axes() {
        None => {
            mlx_sys::array::ffi::array_min_all(a.as_inner(), keepdim, has, dev_only, dev_t, idx)
        }
        Some([axis]) => mlx_sys::array::ffi::array_min_axis(
            a.as_inner(),
            *axis,
            keepdim,
            has,
            dev_only,
            dev_t,
            idx,
        ),
        Some(axes) => mlx_sys::array::ffi::array_min_axes(
            a.as_inner(),
            axes,
            keepdim,
            has,
            dev_only,
            dev_t,
            idx,
        ),
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Indices of the maximum values along the specified axis. Returns `Uint32`
/// (MLX convention). For [`All`], reduces over the flattened array.
///
/// Multi-axis argmax is not supported by MLX; pass a single `i32` axis or [`All`].
pub fn argmax<A: IntoAxes>(a: &Array, axes: A, keepdim: bool) -> Result<Array> {
    argmax_on(a, axes, keepdim, ())
}

/// Stream-targeted variant of [`argmax`].
pub fn argmax_on<A: IntoAxes>(
    a: &Array,
    axes: A,
    keepdim: bool,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = match axes.as_axes() {
        None => {
            mlx_sys::array::ffi::array_argmax_all(a.as_inner(), keepdim, has, dev_only, dev_t, idx)
        }
        Some([axis]) => mlx_sys::array::ffi::array_argmax_axis(
            a.as_inner(),
            *axis,
            keepdim,
            has,
            dev_only,
            dev_t,
            idx,
        ),
        Some(axes) => {
            return Err(Error::Mlx(format!(
                "argmax does not support multi-axis reduction (got axes={axes:?})"
            )));
        }
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_returns_none() {
        assert_eq!(All.as_axes(), None);
    }

    #[test]
    fn i32_returns_single_element_slice() {
        let axis: i32 = -1;
        assert_eq!(axis.as_axes(), Some(&[-1][..]));
    }

    #[test]
    fn slice_returns_self() {
        let axes: &[i32] = &[0, 2];
        assert_eq!(axes.as_axes(), Some(&[0, 2][..]));
    }

    #[test]
    fn vec_returns_slice() {
        let axes = vec![0_i32, 2];
        assert_eq!(axes.as_axes(), Some(&[0, 2][..]));
    }

    #[test]
    fn array_literal_returns_slice() {
        let axes: [i32; 2] = [0, 2];
        assert_eq!(axes.as_axes(), Some(&[0, 2][..]));
    }
}
