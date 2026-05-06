//! Normal distribution builder.

use crate::{Array, Dtype, Error, IntoShape, Result, Shape};

/// Builder for sampling from the normal distribution. Defaults to standard
/// normal `N(0, 1)` `f32` scalar; `loc` defaults to 0 and `scale` to 1 when
/// not set.
pub struct Normal<'k> {
    shape: Shape,
    dtype: Dtype,
    loc: Option<f64>,
    scale: Option<f64>,
    key: Option<&'k Array>,
}

impl<'k> Normal<'k> {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            shape: Shape::new(),
            dtype: Dtype::Float32,
            loc: None,
            scale: None,
            key: None,
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
    /// Mean (default 0.0). Cast to `f32` at sample time.
    pub fn loc(mut self, v: f64) -> Self {
        self.loc = Some(v);
        self
    }
    /// Standard deviation (default 1.0). Cast to `f32` at sample time.
    pub fn scale(mut self, v: f64) -> Self {
        self.scale = Some(v);
        self
    }
    /// PRNG key (default: global state via `seed()`).
    pub fn key(mut self, k: &'k Array) -> Self {
        self.key = Some(k);
        self
    }

    /// Materialize the random sample. Returns `Err` on FFI failure or invalid params.
    pub fn sample(self) -> Result<Array> {
        // Materialize loc/scale arrays only when set.
        let loc_arr: Option<Array> = self.loc.map(super::scalar_f32).transpose()?;
        let scale_arr: Option<Array> = self.scale.map(super::scalar_f32).transpose()?;
        let l = loc_arr
            .as_ref()
            .map_or(std::ptr::null(), |a| a.as_inner() as *const _);
        let s = scale_arr
            .as_ref()
            .map_or(std::ptr::null(), |a| a.as_inner() as *const _);
        let k = self
            .key
            .map_or(std::ptr::null(), |a| a.as_inner() as *const _);
        // SAFETY: l/s/k are null or borrows valid for this call.
        let inner = unsafe {
            mlx_sys::random::ffi::normal(self.shape.as_slice(), self.dtype.as_u8(), l, s, k)
        }
        .map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
}

impl Default for Normal<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a [`Normal`] sampler. Same as `Normal::new()`.
pub fn normal<'k>() -> Normal<'k> {
    Normal::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    #[test]
    fn default_finite() {
        let k = random::key(7).expect("key");
        let n = normal().shape(100).key(&k).sample().expect("sample");
        assert_eq!(n.shape().as_slice(), &[100]);
        let v: Vec<f32> = n.to_vec().expect("to_vec");
        for x in &v {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn loc_scale_chain() {
        let k = random::key(7).expect("key");
        let n = normal()
            .shape(500)
            .loc(10.0)
            .scale(0.1)
            .key(&k)
            .sample()
            .expect("sample");
        let v: Vec<f32> = n.to_vec().expect("to_vec");
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!((mean - 10.0).abs() < 0.5, "mean {mean} not near 10");
    }
}
