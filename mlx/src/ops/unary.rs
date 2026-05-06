//! Element-wise unary ops.
//!
//! All return `Result<Array>` because dtype mismatches (e.g. `sqrt` on integer
//! types) raise MLX exceptions that we surface as `Error::Mlx`.
//!
//! Both the default variant (`exp(a)`) and the stream-targeted variant
//! (`exp_on(a, target)`) are generated from one declaration by the
//! [`op_with_stream!`](crate::op_with_stream) macro.

use crate::ops::reduction::IntoAxes;
use crate::{Array, Error, Result, StreamOrDevice};

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

// === P5.5 softmax (axis-driven reduction-style op) ===
//
// MLX exposes three softmax overloads (multi-axis vector, single-axis int,
// last-axis default). We unify them via the `IntoAxes` trait — the same
// dispatch convention used for `sum`/`mean`/etc. The shim takes a slice
// and treats empty as "all axes" (matches `IntoAxes::All`).

/// Softmax along the given axes. With [`crate::ops::All`], reduces over all
/// axes (every element gets re-normalized). For inference attention you
/// usually want `softmax(&logits, -1, false)` (last axis only).
///
/// `precise = true` requests higher-precision intermediate math (slower).
pub fn softmax<A: IntoAxes>(a: &Array, axes: A, precise: bool) -> Result<Array> {
    softmax_on(a, axes, precise, ())
}

/// Stream-targeted variant of [`softmax`]. Pass `()` for the current default
/// stream, a `Stream`, or a `Device`.
pub fn softmax_on<A: IntoAxes>(
    a: &Array,
    axes: A,
    precise: bool,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // For the multi-axis form we hold the axes in a stack-borrowed slice.
    // For the All case we pass an empty slice and let the shim default to
    // "all axes" (matches MLX's vector<int>{0..ndim} construction).
    let inner = match axes.as_axes() {
        None => mlx_sys::array::ffi::softmax(a.as_inner(), &[], precise, has, dev_only, dev_t, idx),
        Some(slice) => {
            mlx_sys::array::ffi::softmax(a.as_inner(), slice, precise, has, dev_only, dev_t, idx)
        }
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
