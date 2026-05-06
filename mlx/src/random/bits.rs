//! Random bits builder.

use crate::{Array, Error, IntoShape, Result, Shape, StreamOrDevice};

/// Builder for sampling arrays of random uniform integers of a given byte width.
/// Defaults to `width = 4` (32-bit `u32`) scalar.
pub struct Bits<'k> {
    shape: Shape,
    width: i32,
    key: Option<&'k Array>,
    target: StreamOrDevice,
}

impl<'k> Bits<'k> {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            shape: Shape::new(),
            width: 4,
            key: None,
            target: StreamOrDevice::Default,
        }
    }

    /// Output shape (default scalar). Last setter wins on overlapping calls.
    pub fn shape<S: IntoShape>(mut self, s: S) -> Self {
        self.shape = s.into_shape();
        self
    }
    /// Element byte width (default 4 = `u32`). Valid values are 1, 2, or 4.
    pub fn width(mut self, w: i32) -> Self {
        self.width = w;
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
            mlx_sys::random::ffi::bits(
                self.shape.as_slice(),
                self.width,
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

impl Default for Bits<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a [`Bits`] sampler. Same as `Bits::new()`.
pub fn bits<'k>() -> Bits<'k> {
    Bits::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    #[test]
    fn default_chain() {
        let k = random::key(7).expect("key");
        let b = bits().shape(10).key(&k).sample().expect("sample");
        assert_eq!(b.shape().as_slice(), &[10]);
        let v: Vec<u32> = b.to_vec().expect("to_vec");
        assert_eq!(v.len(), 10);
    }
}
