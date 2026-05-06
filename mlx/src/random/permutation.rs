//! Permutation builders: array permutation along an axis, or `arange(n)`.

use crate::{Array, Error, Result};

/// Builder for a random permutation of `x` along an axis. Defaults to `axis = 0`.
pub struct Permutation<'a, 'k> {
    x: &'a Array,
    axis: i32,
    key: Option<&'k Array>,
}

impl<'a, 'k> Permutation<'a, 'k> {
    /// Create a new builder permuting along axis `0` of `x`.
    pub fn new(x: &'a Array) -> Self {
        Self {
            x,
            axis: 0,
            key: None,
        }
    }

    /// Axis along which to permute (default `0`).
    pub fn axis(mut self, a: i32) -> Self {
        self.axis = a;
        self
    }
    /// PRNG key (default: global state via `seed()`).
    pub fn key(mut self, k: &'k Array) -> Self {
        self.key = Some(k);
        self
    }

    /// Materialize the random permutation. Returns `Err` on FFI failure or invalid params.
    pub fn sample(self) -> Result<Array> {
        let k = self
            .key
            .map_or(std::ptr::null(), |a| a.as_inner() as *const _);
        // SAFETY: k is null or borrow valid for this call.
        let inner = unsafe { mlx_sys::random::ffi::permutation(self.x.as_inner(), self.axis, k) }
            .map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
}

/// Builder for a random permutation of `arange(n)`. Output is a 1D `int32` array.
pub struct PermutationRange<'k> {
    n: i32,
    key: Option<&'k Array>,
}

impl<'k> PermutationRange<'k> {
    /// Create a new builder for a permutation of `arange(n)`.
    pub fn new(n: i32) -> Self {
        Self { n, key: None }
    }

    /// PRNG key (default: global state via `seed()`).
    pub fn key(mut self, k: &'k Array) -> Self {
        self.key = Some(k);
        self
    }

    /// Materialize the random permutation. Returns `Err` on FFI failure or invalid params.
    pub fn sample(self) -> Result<Array> {
        let k = self
            .key
            .map_or(std::ptr::null(), |a| a.as_inner() as *const _);
        // SAFETY: k is null or borrow valid for this call.
        let inner =
            unsafe { mlx_sys::random::ffi::permutation_arange(self.n, k) }.map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
}

/// Build a [`Permutation`] sampler. Same as `Permutation::new(x)`.
pub fn permutation<'a, 'k>(x: &'a Array) -> Permutation<'a, 'k> {
    Permutation::new(x)
}

/// Build a [`PermutationRange`] sampler. Same as `PermutationRange::new(n)`.
pub fn permutation_range<'k>(n: i32) -> PermutationRange<'k> {
    PermutationRange::new(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    #[test]
    fn range_default() {
        let k = random::key(7).expect("key");
        let p = permutation_range(10).key(&k).sample().expect("sample");
        assert_eq!(p.shape().as_slice(), &[10]);
    }

    #[test]
    fn array_chain() {
        let k = random::key(7).expect("key");
        let x = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[4][..])).expect("x");
        let p = permutation(&x).axis(0).key(&k).sample().expect("sample");
        assert_eq!(p.shape().as_slice(), &[4]);
    }
}
