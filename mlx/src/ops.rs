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
