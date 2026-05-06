//! Binary element-wise ops with NumPy broadcasting.
//!
//! Each function validates broadcast compatibility before crossing the FFI
//! boundary so we can return `Error::BroadcastMismatch` with structured
//! `lhs`/`rhs` fields, instead of relying on MLX's English exception strings.
//!
//! Both the default variant (`add(a, b)`) and the stream-targeted
//! variant (`add_on(a, b, target)`) are generated from one declaration
//! by the [`op_with_stream!`](crate::op_with_stream) macro. The default
//! variant delegates to `_on` with `()` (i.e. `StreamOrDevice::Default`),
//! so behavior of pre-P5.7 callers is bit-identical.

use crate::{broadcast, Array, Result};

op_with_stream! {
    /// Element-wise addition with NumPy broadcasting.
    pub fn add(a: &Array, b: &Array) -> Result<Array> {
        broadcast::broadcast_shape(a.shape().as_slice(), b.shape().as_slice())?;
    } => mlx_sys::array::ffi::array_add(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Element-wise subtraction with NumPy broadcasting.
    pub fn subtract(a: &Array, b: &Array) -> Result<Array> {
        broadcast::broadcast_shape(a.shape().as_slice(), b.shape().as_slice())?;
    } => mlx_sys::array::ffi::array_subtract(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Element-wise multiplication with NumPy broadcasting.
    pub fn multiply(a: &Array, b: &Array) -> Result<Array> {
        broadcast::broadcast_shape(a.shape().as_slice(), b.shape().as_slice())?;
    } => mlx_sys::array::ffi::array_multiply(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Element-wise division with NumPy broadcasting.
    pub fn divide(a: &Array, b: &Array) -> Result<Array> {
        broadcast::broadcast_shape(a.shape().as_slice(), b.shape().as_slice())?;
    } => mlx_sys::array::ffi::array_divide(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Element-wise negation.
    ///
    /// On unsigned dtypes (`u8`/`u16`/etc.) MLX wraps two's-complement style
    /// (e.g. `1u8 → 255u8`); it does not throw. On `bool` MLX errors at eval
    /// time per its own dtype rules.
    pub fn negative(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::array_negative(a.as_inner());
}
