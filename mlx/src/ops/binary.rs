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

use crate::{broadcast, Array, Error, Result, StreamOrDevice};

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

// === P5.5 comparison ops ===
//
// All return a `Bool` array. Broadcasting is handled MLX-side; we keep the
// Rust-side prologue minimal (no `broadcast_shape` validation) because MLX's
// own broadcast diagnostic is sufficient and matches NumPy semantics.

op_with_stream! {
    /// Element-wise `a == b`. Returns a `Bool` array.
    pub fn equal(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::equal(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Element-wise `a != b`. Returns a `Bool` array.
    pub fn not_equal(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::not_equal(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Element-wise `a < b`. Returns a `Bool` array.
    pub fn less(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::less(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Element-wise `a <= b`. Returns a `Bool` array.
    pub fn less_equal(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::less_equal(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Element-wise `a > b`. Returns a `Bool` array.
    pub fn greater(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::greater(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Element-wise `a >= b`. Returns a `Bool` array.
    pub fn greater_equal(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::greater_equal(a.as_inner(), b.as_inner());
}

// === P5.5 element-wise max/min ===

op_with_stream! {
    /// Element-wise `max(a, b)` (broadcasted).
    pub fn maximum(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::maximum(a.as_inner(), b.as_inner());
}

op_with_stream! {
    /// Element-wise `min(a, b)` (broadcasted).
    pub fn minimum(a: &Array, b: &Array) -> Result<Array>
        => mlx_sys::array::ffi::minimum(a.as_inner(), b.as_inner());
}

// === P5.5 clip ===
//
// Hand-written because both bounds are `Option<&Array>` and bridge through
// raw pointers (cxx 1.0 doesn't bridge `Option<&T>` directly).

/// Element-wise clamp: result is `clamp(a, a_min, a_max)`. Either bound is
/// optional — pass `None` to leave that side unbounded. With both `None`
/// this is effectively a no-op (returns a copy).
pub fn clip(a: &Array, a_min: Option<&Array>, a_max: Option<&Array>) -> Result<Array> {
    clip_on(a, a_min, a_max, ())
}

/// Stream-targeted variant of [`clip`]. Pass `()` for the current default
/// stream, a `Stream`, or a `Device`.
pub fn clip_on(
    a: &Array,
    a_min: Option<&Array>,
    a_max: Option<&Array>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let lo_ptr = a_min.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let hi_ptr = a_max.map_or(std::ptr::null(), |x| x.as_inner() as *const _);
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: `lo_ptr` / `hi_ptr` are either null or borrowed from a live
    // `&Array` that outlives this call (they share `a`'s caller frame).
    let inner = unsafe {
        mlx_sys::array::ffi::clip(a.as_inner(), lo_ptr, hi_ptr, has, dev_only, dev_t, idx)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
