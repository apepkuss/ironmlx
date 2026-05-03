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

/// Element-wise negation.
///
/// On unsigned dtypes (`u8`/`u16`/etc.) MLX wraps two's-complement style
/// (e.g. `1u8 → 255u8`); it does not throw. On `bool` MLX errors at eval
/// time per its own dtype rules.
pub fn negative(a: &Array) -> Result<Array> {
    let inner = mlx_sys::array::ffi::array_negative(a.as_inner()).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Macro to define a unary op delegating to a single shim function.
macro_rules! unary_op {
    ($name:ident, $shim:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(a: &Array) -> Result<Array> {
            let inner = mlx_sys::array::ffi::$shim(a.as_inner()).map_err(Error::from)?;
            Ok(Array::from_inner(inner))
        }
    };
}

unary_op!(exp,        array_exp,        "Element-wise natural exponential.");
unary_op!(log,        array_log,        "Element-wise natural logarithm.");
unary_op!(sqrt,       array_sqrt,       "Element-wise square root.");
unary_op!(tanh,       array_tanh,       "Element-wise hyperbolic tangent.");
unary_op!(sigmoid,    array_sigmoid,    "Element-wise sigmoid (1 / (1 + exp(-x))).");
unary_op!(square,     array_square,     "Element-wise x^2.");
unary_op!(rsqrt,      array_rsqrt,      "Element-wise 1/sqrt(x). Used in attention scaling.");
unary_op!(erf,        array_erf,        "Element-wise error function. Used in GELU.");
unary_op!(reciprocal, array_reciprocal, "Element-wise 1/x.");
