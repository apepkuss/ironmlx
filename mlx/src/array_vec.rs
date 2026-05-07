//! Thin safe wrapper around `mlx_sys::compile::ffi::ArrayVec` — outputs of
//! multi-array ops (`MetalKernel::dispatch`, compile callback) come back as
//! an `ArrayVec`. Take individual elements via `take_at(i)` in the order
//! they were produced.

use crate::{Array, Error, Result};

/// Owning wrapper around a C++ `std::vector<array>`.
pub struct ArrayVec {
    inner: cxx::UniquePtr<mlx_sys::compile::ffi::ArrayVec>,
}

impl ArrayVec {
    /// Construct from a raw cxx UniquePtr. Internal use only.
    pub(crate) fn from_inner(inner: cxx::UniquePtr<mlx_sys::compile::ffi::ArrayVec>) -> Self {
        Self { inner }
    }

    /// Number of arrays.
    pub fn len(&self) -> usize {
        mlx_sys::compile::ffi::array_vec_count(&self.inner)
    }

    /// Returns `true` if the vec contains no arrays.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Take the i-th array out of the vec.
    ///
    /// **Erase-and-shift semantics:** the slot is removed from the vec and
    /// subsequent elements shift down. After taking element `0`, the
    /// original element `1` becomes the new element `0`, and `len()`
    /// decreases by 1. To drain the vec in declared order, repeatedly call
    /// `take_at(0)` rather than `take_at(0)`, `take_at(1)`, ...
    ///
    /// Returns `Err` if `i >= len()`.
    pub fn take_at(&mut self, i: usize) -> Result<Array> {
        let raw = mlx_sys::compile::ffi::array_vec_take_at(self.inner.pin_mut(), i)
            .map_err(Error::from)?;
        Ok(Array::from_inner(raw))
    }
}

impl std::fmt::Debug for ArrayVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayVec")
            .field("len", &self.len())
            .finish()
    }
}
