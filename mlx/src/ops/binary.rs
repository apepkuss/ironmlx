//! Binary element-wise ops with NumPy broadcasting.
//!
//! Each function validates broadcast compatibility before crossing the FFI
//! boundary so we can return `Error::BroadcastMismatch` with structured
//! `lhs`/`rhs` fields, instead of relying on MLX's English exception strings.

use crate::{broadcast, Array, Error, Result};

/// Element-wise addition with NumPy broadcasting.
pub fn add(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner = mlx_sys::array::ffi::array_add(a.as_inner(), b.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise subtraction with NumPy broadcasting.
pub fn subtract(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner =
        mlx_sys::array::ffi::array_subtract(a.as_inner(), b.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise multiplication with NumPy broadcasting.
pub fn multiply(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner =
        mlx_sys::array::ffi::array_multiply(a.as_inner(), b.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise division with NumPy broadcasting.
pub fn divide(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner =
        mlx_sys::array::ffi::array_divide(a.as_inner(), b.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise negation.
///
/// On unsigned dtypes (`u8`/`u16`/etc.) MLX wraps two's-complement style
/// (e.g. `1u8 → 255u8`); it does not throw. On `bool` MLX errors at eval
/// time per its own dtype rules.
pub fn negative(a: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_negative(a.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
