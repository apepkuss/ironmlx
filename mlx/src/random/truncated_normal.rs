//! Truncated normal distribution builder.

use crate::{Array, Dtype, Error, IntoShape, Result, Shape};

/// Builder for sampling from the truncated normal distribution restricted to
/// `[lower, upper]`. If `.shape(...)` is not called, output shape is inferred
/// from `broadcast(lower, upper)`. Defaults to `f32`.
pub struct TruncatedNormal<'a, 'k> {
    lower: &'a Array,
    upper: &'a Array,
    shape: Option<Shape>,
    dtype: Dtype,
    key: Option<&'k Array>,
}

impl<'a, 'k> TruncatedNormal<'a, 'k> {
    /// Create a new builder with the given `lower` and `upper` bound tensors.
    pub fn new(lower: &'a Array, upper: &'a Array) -> Self {
        Self {
            lower,
            upper,
            shape: None,
            dtype: Dtype::Float32,
            key: None,
        }
    }

    /// Output shape (default: inferred from `broadcast(lower, upper)`). Last setter wins.
    pub fn shape<S: IntoShape>(mut self, s: S) -> Self {
        self.shape = Some(s.into_shape());
        self
    }
    /// Output dtype (default `Float32`).
    pub fn dtype(mut self, d: Dtype) -> Self {
        self.dtype = d;
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
                unsafe {
                    mlx_sys::random::ffi::truncated_normal(
                        self.lower.as_inner(),
                        self.upper.as_inner(),
                        s.as_slice(),
                        self.dtype.as_u8(),
                        k,
                    )
                }
            }
            None => {
                // SAFETY: k is null or borrow valid for this call.
                unsafe {
                    mlx_sys::random::ffi::truncated_normal_default(
                        self.lower.as_inner(),
                        self.upper.as_inner(),
                        self.dtype.as_u8(),
                        k,
                    )
                }
            }
        }
        .map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
}

/// Build a [`TruncatedNormal`] sampler. Same as `TruncatedNormal::new(lower, upper)`.
pub fn truncated_normal<'a, 'k>(lower: &'a Array, upper: &'a Array) -> TruncatedNormal<'a, 'k> {
    TruncatedNormal::new(lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    #[test]
    fn default_broadcast_shape() {
        let k = random::key(7).expect("key");
        let lo = Array::try_from((&[-1.0_f32, -2.0][..], &[2][..])).expect("lo");
        let hi = Array::try_from((&[1.0_f32, 2.0][..], &[2][..])).expect("hi");
        let t = truncated_normal(&lo, &hi).key(&k).sample().expect("sample");
        assert_eq!(t.shape().as_slice(), &[2]);
    }

    #[test]
    fn explicit_shape_in_bounds() {
        let k = random::key(7).expect("key");
        let lo = Array::try_from((&[-1.0_f32][..], &[][..])).expect("lo");
        let hi = Array::try_from((&[1.0_f32][..], &[][..])).expect("hi");
        let t = truncated_normal(&lo, &hi)
            .shape(50)
            .key(&k)
            .sample()
            .expect("sample");
        let v: Vec<f32> = t.to_vec().expect("to_vec");
        for x in &v {
            assert!(*x >= -1.0 && *x <= 1.0);
        }
    }
}
