//! Random integer distribution builder.

use crate::{Array, Dtype, Error, IntoShape, Result, Shape, StreamOrDevice};

/// Builder for sampling uniform random integers in `[low, high)`. Defaults to
/// `i32` scalar in `[0, 1)`.
pub struct RandInt<'k> {
    low: i64,
    high: i64,
    shape: Shape,
    dtype: Dtype,
    key: Option<&'k Array>,
    target: StreamOrDevice,
}

impl<'k> RandInt<'k> {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            low: 0,
            high: 1,
            shape: Shape::new(),
            dtype: Dtype::Int32,
            key: None,
            target: StreamOrDevice::Default,
        }
    }

    /// Lower bound (default 0). Cast to `i32` at sample time.
    pub fn low(mut self, v: i64) -> Self {
        self.low = v;
        self
    }
    /// Upper bound, exclusive (default 1). Cast to `i32` at sample time.
    pub fn high(mut self, v: i64) -> Self {
        self.high = v;
        self
    }
    /// Output shape (default scalar). Last setter wins on overlapping calls.
    pub fn shape<S: IntoShape>(mut self, s: S) -> Self {
        self.shape = s.into_shape();
        self
    }
    /// Output dtype (default `Int32`).
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
        let low_arr = super::scalar_i32(self.low)?;
        let high_arr = super::scalar_i32(self.high)?;
        let k = self
            .key
            .map_or(std::ptr::null(), |a| a.as_inner() as *const _);
        // SAFETY: k is null or borrow valid for this call.
        let inner = unsafe {
            mlx_sys::random::ffi::randint(
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

impl Default for RandInt<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a [`RandInt`] sampler. Same as `RandInt::new()`.
pub fn randint<'k>() -> RandInt<'k> {
    RandInt::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    #[test]
    fn in_range() {
        let k = random::key(7).expect("key");
        let r = randint()
            .low(0)
            .high(10)
            .shape(100)
            .key(&k)
            .sample()
            .expect("sample");
        let v: Vec<i32> = r.to_vec().expect("to_vec");
        for x in &v {
            assert!(*x >= 0 && *x < 10);
        }
    }
}
