//! Categorical distribution builder for token sampling.

use crate::{Array, Error, IntoShape, Result, Shape, StreamOrDevice};

enum Output {
    Default,
    NumSamples(i32),
    Shape(Shape),
}

/// Builder for sampling from a categorical distribution over `logits` along
/// an axis. Defaults to one sample per row with `axis = -1`.
///
/// Modes (last setter wins):
/// - default: 1 sample per row, output shape is logits with `axis` removed.
/// - `.num_samples(n)`: `n` samples per row.
/// - `.shape(s)`: explicit output shape.
///
/// Note: `shape()` and `num_samples()` are mutually exclusive — last setter wins.
pub struct Categorical<'a, 'k> {
    logits: &'a Array,
    axis: i32,
    output: Output,
    key: Option<&'k Array>,
    target: StreamOrDevice,
}

impl<'a, 'k> Categorical<'a, 'k> {
    /// Create a new builder over the given `logits` tensor.
    pub fn new(logits: &'a Array) -> Self {
        Self {
            logits,
            axis: -1,
            output: Output::Default,
            key: None,
            target: StreamOrDevice::Default,
        }
    }

    /// Axis along which to sample (default `-1`, the last axis).
    pub fn axis(mut self, a: i32) -> Self {
        self.axis = a;
        self
    }
    /// Number of samples per row. Mutually exclusive with `shape()`; last setter wins.
    pub fn num_samples(mut self, n: i32) -> Self {
        self.output = Output::NumSamples(n);
        self
    }
    /// Explicit output shape. Mutually exclusive with `num_samples()`; last setter wins.
    pub fn shape<S: IntoShape>(mut self, s: S) -> Self {
        self.output = Output::Shape(s.into_shape());
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
        let inner = match self.output {
            Output::Default => {
                // SAFETY: k is null or borrow valid for this call.
                unsafe {
                    mlx_sys::random::ffi::categorical(
                        self.logits.as_inner(),
                        self.axis,
                        k,
                        has,
                        dev_only,
                        dev_t,
                        idx,
                    )
                }
            }
            Output::NumSamples(n) => {
                // SAFETY: k is null or borrow valid for this call.
                unsafe {
                    mlx_sys::random::ffi::categorical_n(
                        self.logits.as_inner(),
                        self.axis,
                        n,
                        k,
                        has,
                        dev_only,
                        dev_t,
                        idx,
                    )
                }
            }
            Output::Shape(s) => {
                // SAFETY: k is null or borrow valid for this call.
                unsafe {
                    mlx_sys::random::ffi::categorical_shaped(
                        self.logits.as_inner(),
                        self.axis,
                        s.as_slice(),
                        k,
                        has,
                        dev_only,
                        dev_t,
                        idx,
                    )
                }
            }
        }
        .map_err(Error::from)?;
        Ok(Array::from_inner(inner))
    }
}

/// Build a [`Categorical`] sampler. Same as `Categorical::new(logits)`.
pub fn categorical<'a, 'k>(logits: &'a Array) -> Categorical<'a, 'k> {
    Categorical::new(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    fn logits() -> Array {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        Array::try_from((&data[..], &[4, 8][..])).expect("logits")
    }

    #[test]
    fn default_one_sample_per_row() {
        let k = random::key(7).expect("key");
        let out = categorical(&logits()).key(&k).sample().expect("sample");
        assert_eq!(out.shape().as_slice(), &[4]);
    }

    #[test]
    fn num_samples_chain() {
        let k = random::key(7).expect("key");
        let out = categorical(&logits())
            .axis(-1)
            .num_samples(3)
            .key(&k)
            .sample()
            .expect("sample");
        assert_eq!(out.shape().as_slice(), &[4, 3]);
    }

    #[test]
    fn explicit_shape() {
        let k = random::key(7).expect("key");
        let small_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let l = Array::try_from((&small_data[..], &[2, 8][..])).expect("logits");
        let out = categorical(&l)
            .axis(-1)
            .shape((5, 2))
            .key(&k)
            .sample()
            .expect("sample");
        assert_eq!(out.shape().as_slice(), &[5, 2]);
    }
}
