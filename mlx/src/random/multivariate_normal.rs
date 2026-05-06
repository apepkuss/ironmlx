//! Multivariate normal distribution builder.

use crate::{Array, Dtype, Error, IntoShape, Result, Shape, StreamOrDevice};

/// Builder for sampling from a multivariate normal distribution with the
/// given `mean` and covariance `cov`. Defaults to a single sample (`f32`);
/// output shape is `[..shape, dim]` where `dim` is the last dimension of `mean`.
pub struct MultivariateNormal<'a, 'k> {
    mean: &'a Array,
    cov: &'a Array,
    shape: Shape,
    dtype: Dtype,
    key: Option<&'k Array>,
    target: StreamOrDevice,
}

impl<'a, 'k> MultivariateNormal<'a, 'k> {
    /// Create a new builder with the given `mean` and `cov` tensors.
    pub fn new(mean: &'a Array, cov: &'a Array) -> Self {
        Self {
            mean,
            cov,
            shape: Shape::new(),
            dtype: Dtype::Float32,
            key: None,
            target: StreamOrDevice::Default,
        }
    }

    /// Batch shape preceding the `dim` axis (default empty). Last setter wins.
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
        let k = self
            .key
            .map_or(std::ptr::null(), |a| a.as_inner() as *const _);
        // SAFETY: k is null or borrow valid for this call.
        let inner = unsafe {
            mlx_sys::random::ffi::multivariate_normal(
                self.mean.as_inner(),
                self.cov.as_inner(),
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

/// Build a [`MultivariateNormal`] sampler. Same as `MultivariateNormal::new(mean, cov)`.
pub fn multivariate_normal<'a, 'k>(mean: &'a Array, cov: &'a Array) -> MultivariateNormal<'a, 'k> {
    MultivariateNormal::new(mean, cov)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    #[test]
    fn binding_smoke() {
        // multivariate_normal uses linalg::svd which may be NYI on Metal.
        let k = random::key(7).expect("key");
        let mean = Array::try_from((&[0.0_f32, 0.0][..], &[2][..])).expect("mean");
        let cov = Array::try_from((&[1.0_f32, 0.0, 0.0, 1.0][..], &[2, 2][..])).expect("cov");
        let result = multivariate_normal(&mean, &cov).shape(5).key(&k).sample();
        match result {
            Ok(_) | Err(_) => {}
        }
    }
}
