//! Uniform distribution builder.

use crate::{Array, Dtype, Error, IntoShape, Result, Shape, StreamOrDevice};

/// Builder for sampling from the uniform distribution on `[low, high)`.
/// Defaults to `[0, 1)` `f32` scalar.
pub struct Uniform<'k> {
    low: f64,
    high: f64,
    shape: Shape,
    dtype: Dtype,
    key: Option<&'k Array>,
    target: StreamOrDevice,
}

impl<'k> Uniform<'k> {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            low: 0.0,
            high: 1.0,
            shape: Shape::new(),
            dtype: Dtype::Float32,
            key: None,
            target: StreamOrDevice::Default,
        }
    }

    /// Lower bound (default 0.0). Cast to `f32` at sample time.
    pub fn low(mut self, v: f64) -> Self {
        self.low = v;
        self
    }
    /// Upper bound, exclusive (default 1.0). Cast to `f32` at sample time.
    pub fn high(mut self, v: f64) -> Self {
        self.high = v;
        self
    }
    /// Output shape (default scalar). Last setter wins on overlapping calls.
    pub fn shape<S: IntoShape>(mut self, s: S) -> Self {
        self.shape = s.into_shape();
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
    /// Set the target stream/device for this sample call.
    pub fn stream(mut self, target: impl Into<StreamOrDevice>) -> Self {
        self.target = target.into();
        self
    }

    /// Materialize the random sample. Returns `Err` on FFI failure or invalid params.
    pub fn sample(self) -> Result<Array> {
        let (has, dev_only, dev_t, idx) = self.target.encode();
        let low_arr = super::scalar_f32(self.low)?;
        let high_arr = super::scalar_f32(self.high)?;
        let k = self
            .key
            .map_or(std::ptr::null(), |a| a.as_inner() as *const _);
        // SAFETY: k is null or borrow valid for this call.
        let inner = unsafe {
            mlx_sys::random::ffi::uniform(
                low_arr.as_inner(),
                high_arr.as_inner(),
                self.shape.as_slice(),
                self.dtype.as_u8(),
                k,
                has,
                dev_only,
                dev_t,
                idx,
            )
        }
        .map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
}

impl Default for Uniform<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a [`Uniform`] sampler. Same as `Uniform::new()`.
pub fn uniform<'k>() -> Uniform<'k> {
    Uniform::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    #[test]
    fn defaults_zero_to_one_f32() {
        let k = random::key(7).expect("key");
        let u = uniform().shape(64).key(&k).sample().expect("sample");
        assert_eq!(u.shape().as_slice(), &[64]);
        let v: Vec<f32> = u.to_vec().expect("to_vec");
        for x in &v {
            assert!(*x >= 0.0 && *x < 1.0);
        }
    }

    #[test]
    fn full_chain() {
        let k = random::key(7).expect("key");
        let u = uniform()
            .low(-2.0)
            .high(3.0)
            .shape((4, 5))
            .dtype(Dtype::Float32)
            .key(&k)
            .sample()
            .expect("sample");
        assert_eq!(u.shape().as_slice(), &[4, 5]);
        let v: Vec<f32> = u.to_vec().expect("to_vec");
        for x in &v {
            assert!(*x >= -2.0 && *x < 3.0);
        }
    }
}
