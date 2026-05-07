//! Gated Delta SSM cache: conv_state (sliding window) + recurrent_state (SSM state).
//!
//! Used by [`crate::nn::GatedDeltaNet`]. Mirrors P2 [`crate::core::cache::KVCache`]'s
//! cap-bounded design — capacity is fixed at construction; `advance` enforces
//! offset ≤ cap.

use anyhow::anyhow;
use mlx::{Array, Dtype};

use crate::Result;

/// Per-layer cache for [`crate::nn::GatedDeltaNet`].
pub struct GatedDeltaCache {
    /// Sliding window of last `kernel_size - 1` tokens for conv1d. Shape:
    /// `[B, kernel_size - 1, conv_dim]`. Dtype matches input.
    conv_state: Array,
    /// SSM recurrent state. Shape: `[B, Hv, Dv, Dk]`. Always fp32 to avoid
    /// drift across long sequences.
    recurrent_state: Array,
    /// Number of tokens consumed so far.
    offset: i32,
    /// Maximum tokens this cache will accept (prompt + decode).
    cap: i32,
}

impl GatedDeltaCache {
    /// Allocate a fresh cache.
    ///
    /// `cap` must be ≥ 1. Both states start zero-initialized.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_cap(
        b: i32,
        kernel_size: i32,
        conv_dim: i32,
        hv: i32,
        dv: i32,
        dk: i32,
        input_dtype: Dtype,
        cap: i32,
    ) -> Result<Self> {
        if cap < 1 {
            return Err(anyhow!("GatedDeltaCache: cap={cap} must be >= 1"));
        }
        if kernel_size < 1 {
            return Err(anyhow!(
                "GatedDeltaCache: kernel_size={kernel_size} must be >= 1"
            ));
        }
        let conv_state = Array::zeros((b, kernel_size - 1, conv_dim), input_dtype)?;
        let recurrent_state = Array::zeros((b, hv, dv, dk), Dtype::Float32)?;
        Ok(Self {
            conv_state,
            recurrent_state,
            offset: 0,
            cap,
        })
    }

    pub fn conv_state(&self) -> &Array {
        &self.conv_state
    }

    pub fn recurrent_state(&self) -> &Array {
        &self.recurrent_state
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    pub fn cap(&self) -> i32 {
        self.cap
    }

    /// Replace the conv_state with a freshly-computed sliding window.
    pub fn update_conv(&mut self, new_conv_state: Array) {
        self.conv_state = new_conv_state;
    }

    /// Replace the recurrent_state with the kernel's `state_out`.
    pub fn update_recurrent(&mut self, new_state: Array) {
        self.recurrent_state = new_state;
    }

    /// Bump offset by `n` tokens. Errors if offset+n > cap.
    pub fn advance(&mut self, n: i32) -> Result<()> {
        let new_off = self.offset + n;
        if new_off > self.cap {
            return Err(anyhow!(
                "GatedDeltaCache: offset {} + {} exceeds cap {}",
                self.offset,
                n,
                self.cap
            ));
        }
        self.offset = new_off;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    #[test]
    fn cache_initial_zeros() {
        let cache = GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16)
            .expect("cache new");
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.cap(), 16);
        assert_eq!(cache.conv_state().shape().as_slice(), &[1, 3, 8]);
        assert_eq!(cache.recurrent_state().shape().as_slice(), &[1, 4, 8, 8]);
        assert_eq!(cache.recurrent_state().dtype(), Dtype::Float32);
        assert_eq!(cache.conv_state().dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn cache_advance_within_cap() {
        let mut cache =
            GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 8).expect("cache new");
        cache.advance(4).expect("advance 4");
        assert_eq!(cache.offset(), 4);
        cache.advance(4).expect("advance to cap");
        assert_eq!(cache.offset(), 8);
    }

    #[test]
    fn cache_advance_beyond_cap_errors() {
        let mut cache =
            GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 4).expect("cache new");
        cache.advance(2).unwrap();
        let r = cache.advance(3);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("exceeds cap"), "msg: {msg}");
    }

    #[test]
    fn cache_rejects_zero_cap() {
        let r = GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 0);
        assert!(r.is_err());
    }
}
