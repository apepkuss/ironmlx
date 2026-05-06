//! Element-wise unary ops.
//!
//! All return `Result<Array>` because dtype mismatches (e.g. `sqrt` on integer
//! types) raise MLX exceptions that we surface as `Error::Mlx`.
//!
//! Both the default variant (`exp(a)`) and the stream-targeted variant
//! (`exp_on(a, target)`) are generated from one declaration by the
//! [`op_with_stream!`](crate::op_with_stream) macro.

use crate::{Array, Result};

op_with_stream! {
    /// Element-wise natural exponential.
    pub fn exp(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_exp(a.as_inner());
}

op_with_stream! {
    /// Element-wise natural logarithm.
    pub fn log(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_log(a.as_inner());
}

op_with_stream! {
    /// Element-wise square root.
    pub fn sqrt(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_sqrt(a.as_inner());
}

op_with_stream! {
    /// Element-wise hyperbolic tangent.
    pub fn tanh(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_tanh(a.as_inner());
}

op_with_stream! {
    /// Element-wise sigmoid (1 / (1 + exp(-x))).
    pub fn sigmoid(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_sigmoid(a.as_inner());
}

op_with_stream! {
    /// Element-wise x^2.
    pub fn square(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_square(a.as_inner());
}

op_with_stream! {
    /// Element-wise 1/sqrt(x). Used in attention scaling.
    pub fn rsqrt(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_rsqrt(a.as_inner());
}

op_with_stream! {
    /// Element-wise error function. Used in GELU.
    pub fn erf(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_erf(a.as_inner());
}

op_with_stream! {
    /// Element-wise 1/x.
    pub fn reciprocal(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_reciprocal(a.as_inner());
}
