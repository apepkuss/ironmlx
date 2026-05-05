//! MLX random number generation (PRNG).
//!
//! Functional-style RNG: explicit `key(seed) -> Array` returns a PRNG key,
//! which is split via `split(&key)` (returns 2 sub-keys) or `split_n(&key, n)`
//! to get N sub-keys. Distribution functions (added in subsequent tasks)
//! accept `Option<&Array>` for the key — None uses the global default
//! (set via `seed(seed)`).
//!
//! For LLM token sampling, see `categorical` (P4 Task 3).

use crate::{Array, Dtype, Error, Result};

/// Get a PRNG key from a u64 seed. The returned array is a uint32 key
/// suitable for passing to distribution functions or to `split` / `split_n`.
pub fn key(seed: u64) -> Result<Array> {
    let inner = mlx_sys::random::ffi::key(seed).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Seed the global default PRNG key sequence. Distribution functions called
/// without an explicit `key` will use this default.
pub fn seed(seed: u64) {
    mlx_sys::random::ffi::seed(seed);
}

/// Split a key into 2 distinct sub-keys. Use to derive independent random
/// streams without correlation.
pub fn split(key: &Array) -> Result<(Array, Array)> {
    let mut pair = mlx_sys::random::ffi::split(key.as_inner()).map_err(Error::from)?;
    let first = mlx_sys::random::ffi::key_pair_take_first(pair.pin_mut()).map_err(Error::from)?;
    let second = mlx_sys::random::ffi::key_pair_take_second(pair.pin_mut()).map_err(Error::from)?;
    Ok((Array::from_inner(first), Array::from_inner(second)))
}

/// Split a key into `num` distinct sub-keys, returned as a single array
/// with shape `[num, ...]`.
pub fn split_n(key: &Array, num: i32) -> Result<Array> {
    let inner = mlx_sys::random::ffi::split_n(key.as_inner(), num).map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

// ===== Basic distributions =====

/// Generate an array of random uniform 32-bit integers.
pub fn bits(shape: &[i32], width: i32, key: Option<&Array>) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe { mlx_sys::random::ffi::bits(shape, width, k) }.map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate uniform random numbers in the range `[low, high)`.
pub fn uniform(
    low: &Array,
    high: &Array,
    shape: &[i32],
    dtype: Dtype,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::uniform(low.as_inner(), high.as_inner(), shape, dtype.as_u8(), k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate uniform random numbers in `[0, 1)` with the given shape and dtype.
pub fn uniform_default(shape: &[i32], dtype: Dtype, key: Option<&Array>) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe { mlx_sys::random::ffi::uniform_default(shape, dtype.as_u8(), k) }
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate samples from the normal distribution. `loc` and `scale` default
/// to 0.0 and 1.0 respectively when `None`.
pub fn normal(
    shape: &[i32],
    dtype: Dtype,
    loc: Option<&Array>,
    scale: Option<&Array>,
    key: Option<&Array>,
) -> Result<Array> {
    let l = loc.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let s = scale.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: l/s/k each null or borrow valid for this call.
    let inner = unsafe { mlx_sys::random::ffi::normal(shape, dtype.as_u8(), l, s, k) }
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Generate uniform random integers in `[low, high)`.
pub fn randint(
    low: &Array,
    high: &Array,
    shape: &[i32],
    dtype: Dtype,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe {
        mlx_sys::random::ffi::randint(low.as_inner(), high.as_inner(), shape, dtype.as_u8(), k)
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

// ===== Discrete distributions =====

/// Sample binary (0/1) values with probability `p`. Output shape must match
/// `p`'s broadcastable shape via the `shape` argument.
pub fn bernoulli(p: &Array, shape: &[i32], key: Option<&Array>) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner =
        unsafe { mlx_sys::random::ffi::bernoulli(p.as_inner(), shape, k) }.map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Sample binary values with probability `p`. Output shape inferred from `p`.
pub fn bernoulli_default(p: &Array, key: Option<&Array>) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner =
        unsafe { mlx_sys::random::ffi::bernoulli_default(p.as_inner(), k) }.map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Sample 1 index per row from `logits` along `axis`. The canonical token
/// sampling op for LLM decoding.
pub fn categorical(logits: &Array, axis: i32, key: Option<&Array>) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner = unsafe { mlx_sys::random::ffi::categorical(logits.as_inner(), axis, k) }
        .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Sample `num_samples` indices per row from `logits` along `axis`.
pub fn categorical_n(
    logits: &Array,
    axis: i32,
    num_samples: i32,
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner =
        unsafe { mlx_sys::random::ffi::categorical_n(logits.as_inner(), axis, num_samples, k) }
            .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Sample with explicit output `shape` from `logits` along `axis`.
pub fn categorical_shaped(
    logits: &Array,
    axis: i32,
    shape: &[i32],
    key: Option<&Array>,
) -> Result<Array> {
    let k = key.map_or(std::ptr::null(), |a| a.as_inner() as *const _);
    // SAFETY: k is null or borrow valid for this call.
    let inner =
        unsafe { mlx_sys::random::ffi::categorical_shaped(logits.as_inner(), axis, shape, k) }
            .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
