//! Free-function form of MLX ops. Operator overloads (`Add`, `Sub`, etc.)
//! and `Array` methods (`a.exp()`, `a.matmul()`) all delegate here.
//!
//! Every op returns `Result<Array>` because broadcasting validation, dtype
//! mismatch, or MLX-side errors all surface as recoverable Rust errors.

use crate::{broadcast, Array, Error, Result};

/// Element-wise addition with NumPy broadcasting.
pub fn add(a: &Array, b: &Array) -> Result<Array> {
    // Validate broadcast compatibility before crossing the FFI boundary so
    // we can return Error::BroadcastMismatch with structured lhs/rhs fields.
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner = mlx_sys::array::ffi::array_add(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise subtraction with NumPy broadcasting.
pub fn subtract(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner = mlx_sys::array::ffi::array_subtract(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise multiplication with NumPy broadcasting.
pub fn multiply(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner = mlx_sys::array::ffi::array_multiply(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise division with NumPy broadcasting.
pub fn divide(a: &Array, b: &Array) -> Result<Array> {
    broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let inner = mlx_sys::array::ffi::array_divide(a.as_inner(), b.as_inner())
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Element-wise negation. May error on unsigned/bool dtypes.
///
/// Eagerly evaluates so that dtype errors (e.g. negating u8/bool) surface
/// immediately as `Err(Error::Mlx(...))` rather than at a later eval point.
pub fn negative(a: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_negative(a.as_inner()).map_err(Error::from)?;
    let result = Array::from_inner(inner);
    result.eval()?;
    Ok(result)
}
