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

// === P5.6 一元补完 ===
//
// 8 ops follow the standard `op_with_stream!` pattern; `round` carries an
// extra `decimals: i32` parameter and is hand-written.

op_with_stream! {
    /// Element-wise absolute value `|x|`.
    pub fn abs(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::abs(a.as_inner());
}

op_with_stream! {
    /// Element-wise sign: `-1` for negatives, `0` for zero, `+1` for positives.
    pub fn sign(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::sign(a.as_inner());
}

op_with_stream! {
    /// Element-wise floor (largest integer `<= x`).
    pub fn floor(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::floor(a.as_inner());
}

op_with_stream! {
    /// Element-wise ceiling (smallest integer `>= x`).
    pub fn ceil(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::ceil(a.as_inner());
}

op_with_stream! {
    /// Element-wise sine.
    pub fn sin(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::sin(a.as_inner());
}

op_with_stream! {
    /// Element-wise cosine.
    pub fn cos(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::cos(a.as_inner());
}

op_with_stream! {
    /// Element-wise tangent.
    pub fn tan(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::tan(a.as_inner());
}

op_with_stream! {
    /// Element-wise `exp(x) - 1` (numerically stable for small `x`).
    pub fn expm1(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::expm1(a.as_inner());
}

/// Round to `decimals` decimal places. Use `decimals = 0` for nearest integer.
pub fn round(a: &Array, decimals: i32) -> Result<Array> {
    round_on(a, decimals, ())
}

/// Stream-targeted variant of [`round`]. Pass `()` for the current default
/// stream, a `Stream`, or a `Device`.
pub fn round_on(a: &Array, decimals: i32, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::round(a.as_inner(), decimals, has, dev_only, dev_t, idx)
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

// === P5.6 数值卫生 + logical_not ===
//
// 4 simple unary ops follow the standard `op_with_stream!` pattern; they
// return `Bool` dtype arrays (use `.to_vec::<bool>()` on the Rust side).
// `nan_to_num` is hand-written to encode `Option<f32>` × 2 via `(bool, f32)`
// pairs, mirroring the P4 random `loc`/`scale` optional-encoding pattern.

op_with_stream! {
    /// Element-wise NaN test. Returns a `Bool` array (`true` where the input
    /// is NaN). Non-floating-point inputs raise [`Error::Mlx`].
    pub fn isnan(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::isnan(a.as_inner());
}

op_with_stream! {
    /// Element-wise infinity test. Returns a `Bool` array (`true` where the
    /// input is +inf or -inf).
    pub fn isinf(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::isinf(a.as_inner());
}

op_with_stream! {
    /// Element-wise finiteness test. Returns a `Bool` array (`true` where
    /// the input is neither NaN nor infinite).
    pub fn isfinite(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::isfinite(a.as_inner());
}

op_with_stream! {
    /// Element-wise logical negation. Inputs are interpreted as booleans
    /// (zero → false, non-zero → true) and the output is a `Bool` array.
    pub fn logical_not(a: &Array) -> Result<Array>
        => mlx_sys::array::ffi::logical_not(a.as_inner());
}

/// Replace non-finite values: NaN → `nan`, +∞ → `posinf` (or the dtype's
/// largest finite value when `None`), −∞ → `neginf` (or the smallest finite
/// value when `None`). Mirrors `numpy.nan_to_num`.
pub fn nan_to_num(a: &Array, nan: f32, posinf: Option<f32>, neginf: Option<f32>) -> Result<Array> {
    nan_to_num_on(a, nan, posinf, neginf, ())
}

/// Stream-targeted variant of [`nan_to_num`]. Pass `()` for the current
/// default stream, a `Stream`, or a `Device`.
pub fn nan_to_num_on(
    a: &Array,
    nan: f32,
    posinf: Option<f32>,
    neginf: Option<f32>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has_posinf, posinf_v) = posinf.map_or((false, 0.0), |v| (true, v));
    let (has_neginf, neginf_v) = neginf.map_or((false, 0.0), |v| (true, v));
    let (has, dev_only, dev_t, idx) = target.into().encode();
    let inner = mlx_sys::array::ffi::nan_to_num(
        a.as_inner(),
        nan,
        has_posinf,
        posinf_v,
        has_neginf,
        neginf_v,
        has,
        dev_only,
        dev_t,
        idx,
    )
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
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
