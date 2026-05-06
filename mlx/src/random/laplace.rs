//! Laplace distribution builder.

use crate::{Array, Dtype, Error, IntoShape, Result, Shape, StreamOrDevice};

/// Builder for sampling from the Laplace distribution. Defaults to standard
/// Laplace with `loc = 0.0`, `scale = 1.0`, `f32` scalar.
pub struct Laplace<'k> {
    shape: Shape,
    dtype: Dtype,
    loc: f64,
    scale: f64,
    key: Option<&'k Array>,
    target: StreamOrDevice,
}

impl<'k> Laplace<'k> {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            shape: Shape::new(),
            dtype: Dtype::Float32,
            loc: 0.0,
            scale: 1.0,
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
    /// Location (default 0.0). Cast to `f32` at sample time.
    pub fn loc(mut self, v: f64) -> Self {
        self.loc = v;
        self
    }
    /// Scale (default 1.0). Cast to `f32` at sample time.
    pub fn scale(mut self, v: f64) -> Self {
        self.scale = v;
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
            mlx_sys::random::ffi::laplace(
                self.shape.as_slice(),
                self.dtype.as_u8(),
                self.loc as f32,
                self.scale as f32,
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

impl Default for Laplace<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a [`Laplace`] sampler. Same as `Laplace::new()`.
pub fn laplace<'k>() -> Laplace<'k> {
    Laplace::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    #[test]
    fn defaults_finite() {
        let k = random::key(7).expect("key");
        let l = laplace().shape(100).key(&k).sample().expect("sample");
        let v: Vec<f32> = l.to_vec().expect("to_vec");
        for x in &v {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn loc_scale_chain() {
        let k = random::key(7).expect("key");
        let l = laplace()
            .shape(100)
            .loc(2.0)
            .scale(0.5)
            .key(&k)
            .sample()
            .expect("sample");
        assert_eq!(l.shape().as_slice(), &[100]);
    }
}
