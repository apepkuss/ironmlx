//! Bernoulli distribution builder.

use crate::{Array, Error, IntoShape, Result, Shape};

/// Builder for sampling from the Bernoulli distribution with probability `p`.
/// If `.shape(...)` is not called, output shape is inferred from `p`.
pub struct Bernoulli<'a, 'k> {
    p: &'a Array,
    shape: Option<Shape>,
    key: Option<&'k Array>,
}

impl<'a, 'k> Bernoulli<'a, 'k> {
    /// Create a new builder with the given probability tensor `p`.
    pub fn new(p: &'a Array) -> Self {
        Self {
            p,
            shape: None,
            key: None,
        }
    }

    /// Output shape (default: inferred from `p`). Last setter wins.
    pub fn shape<S: IntoShape>(mut self, s: S) -> Self {
        self.shape = Some(s.into_shape());
        self
    }
    /// PRNG key (default: global state via `seed()`).
    pub fn key(mut self, k: &'k Array) -> Self {
        self.key = Some(k);
        self
    }

    /// Materialize the random sample. Returns `Err` on FFI failure or invalid params.
    pub fn sample(self) -> Result<Array> {
        let k = self
            .key
            .map_or(std::ptr::null(), |a| a.as_inner() as *const _);
        let inner = match self.shape {
            Some(s) => {
                // SAFETY: k is null or borrow valid for this call.
                unsafe { mlx_sys::random::ffi::bernoulli(self.p.as_inner(), s.as_slice(), k) }
            }
            None => {
                // SAFETY: k is null or borrow valid for this call.
                unsafe { mlx_sys::random::ffi::bernoulli_default(self.p.as_inner(), k) }
            }
        }
        .map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
}

/// Build a [`Bernoulli`] sampler. Same as `Bernoulli::new(p)`.
pub fn bernoulli<'a, 'k>(p: &'a Array) -> Bernoulli<'a, 'k> {
    Bernoulli::new(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    #[test]
    fn default_shape_from_p_scalar() {
        let k = random::key(7).expect("key");
        let p = Array::try_from((&[0.5_f32][..], &[][..])).expect("p");
        let b = bernoulli(&p).key(&k).sample().expect("sample");
        assert_eq!(b.shape().as_slice(), &[] as &[i32]);
    }

    #[test]
    fn explicit_shape_chain() {
        let k = random::key(7).expect("key");
        let p = Array::try_from((&[0.5_f32][..], &[][..])).expect("p");
        let b = bernoulli(&p).shape(50).key(&k).sample().expect("sample");
        assert_eq!(b.shape().as_slice(), &[50]);
    }
}
