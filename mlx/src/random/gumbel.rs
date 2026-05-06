//! Gumbel distribution builder.

use crate::{Array, Dtype, Error, IntoShape, Result, Shape, StreamOrDevice};

/// Builder for sampling from the standard Gumbel distribution. Defaults to
/// `f32` scalar.
pub struct Gumbel<'k> {
    shape: Shape,
    dtype: Dtype,
    key: Option<&'k Array>,
    target: StreamOrDevice,
}

impl<'k> Gumbel<'k> {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            shape: Shape::new(),
            dtype: Dtype::Float32,
            key: None,
            target: StreamOrDevice::Default,
        }
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
        let k = self
            .key
            .map_or(std::ptr::null(), |a| a.as_inner() as *const _);
        // SAFETY: k is null or borrow valid for this call.
        let inner = unsafe {
            mlx_sys::random::ffi::gumbel(
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

impl Default for Gumbel<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a [`Gumbel`] sampler. Same as `Gumbel::new()`.
pub fn gumbel<'k>() -> Gumbel<'k> {
    Gumbel::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    #[test]
    fn finite_chain() {
        let k = random::key(7).expect("key");
        let g = gumbel().shape(100).key(&k).sample().expect("sample");
        let v: Vec<f32> = g.to_vec().expect("to_vec");
        for x in &v {
            assert!(x.is_finite());
        }
    }
}
