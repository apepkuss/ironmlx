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

use crate::{Array, Error, Result};

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
    fn as_axes(&self) -> Option<&[i32]> { None }
}

impl sealed::Sealed for i32 {}
impl IntoAxes for i32 {
    fn as_axes(&self) -> Option<&[i32]> { Some(std::slice::from_ref(self)) }
}

impl sealed::Sealed for &[i32] {}
impl IntoAxes for &[i32] {
    fn as_axes(&self) -> Option<&[i32]> { Some(self) }
}

impl sealed::Sealed for Vec<i32> {}
impl IntoAxes for Vec<i32> {
    fn as_axes(&self) -> Option<&[i32]> { Some(self.as_slice()) }
}

impl<const N: usize> sealed::Sealed for [i32; N] {}
impl<const N: usize> IntoAxes for [i32; N] {
    fn as_axes(&self) -> Option<&[i32]> { Some(self.as_slice()) }
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

// Reduction functions land in Tasks 4 and 5.
#[allow(unused_imports)]
use Array as _;
#[allow(unused_imports)]
use Error as _;
#[allow(unused_imports)]
use Result as _;
