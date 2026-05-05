//! MLX random number generation (PRNG).
//!
//! Functional-style RNG: explicit `key(seed) -> Array` returns a PRNG key,
//! which is split via `split(&key)` (returns 2 sub-keys) or `split_n(&key, n)`
//! to get N sub-keys. Distribution functions (added in subsequent tasks)
//! accept `Option<&Array>` for the key — None uses the global default
//! (set via `seed(seed)`).
//!
//! For LLM token sampling, see `categorical` (P4 Task 3).

use crate::{Array, Error, Result};

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
