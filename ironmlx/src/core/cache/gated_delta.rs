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
    conv_state: Array,
    recurrent_state: Array,
    offsets: Vec<i32>,
    cap: i32,
    b: i32,
}

impl GatedDeltaCache {
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
            offsets: vec![0; b as usize],
            cap,
            b,
        })
    }

    pub fn conv_state(&self) -> &Array {
        &self.conv_state
    }

    pub fn recurrent_state(&self) -> &Array {
        &self.recurrent_state
    }

    /// Per-row offsets (length == B). Row `i` has consumed `offsets[i]` tokens.
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }

    pub fn cap(&self) -> i32 {
        self.cap
    }

    pub fn update_conv(&mut self, new_conv_state: Array) {
        self.conv_state = new_conv_state;
    }

    pub fn update_recurrent(&mut self, new_state: Array) {
        self.recurrent_state = new_state;
    }

    /// Per-row offset bump. `per_row_n.len() == B`; each row `i` advances
    /// by `per_row_n[i]` tokens. Errors on length mismatch, negative entry,
    /// or `offsets[i] + per_row_n[i] > cap`.
    pub fn advance(&mut self, per_row_n: &[i32]) -> Result<()> {
        if per_row_n.len() != self.b as usize {
            return Err(anyhow!(
                "GatedDeltaCache::advance: per_row_n.len()={} != B={}",
                per_row_n.len(),
                self.b
            ));
        }
        for (i, &n) in per_row_n.iter().enumerate() {
            if n < 0 {
                return Err(anyhow!(
                    "GatedDeltaCache::advance: per_row_n[{i}] = {n} must be >= 0"
                ));
            }
            let new_off = self.offsets[i] + n;
            if new_off > self.cap {
                return Err(anyhow!(
                    "GatedDeltaCache: offset {} + {} exceeds cap {} on row {i}",
                    self.offsets[i],
                    n,
                    self.cap
                ));
            }
        }
        for (o, &n) in self.offsets.iter_mut().zip(per_row_n.iter()) {
            *o += n;
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        let conv_dims = self.conv_state.shape();
        let conv_dims = conv_dims.as_slice();
        let conv_dtype = self.conv_state.dtype();
        self.conv_state = Array::zeros(conv_dims, conv_dtype)?;

        let rec_dims = self.recurrent_state.shape();
        let rec_dims = rec_dims.as_slice();
        self.recurrent_state = Array::zeros(rec_dims, Dtype::Float32)?;

        for o in &mut self.offsets {
            *o = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    fn make_cache_b(b: i32, cap: i32) -> GatedDeltaCache {
        GatedDeltaCache::new_with_cap(b, 4, 8, 4, 8, 8, Dtype::Bfloat16, cap).expect("cache new")
    }

    #[test]
    fn gdcache_per_row_offsets_initial_zero() {
        let c = make_cache_b(2, 16);
        assert_eq!(c.offsets(), &[0, 0]);
        assert_eq!(c.cap(), 16);
        assert_eq!(c.conv_state().shape().as_slice(), &[2, 3, 8]);
        assert_eq!(c.recurrent_state().shape().as_slice(), &[2, 4, 8, 8]);
    }

    #[test]
    fn gdcache_advance_uniform() {
        let mut c = make_cache_b(2, 8);
        c.advance(&[4, 4]).expect("advance 4,4");
        assert_eq!(c.offsets(), &[4, 4]);
        c.advance(&[4, 4]).expect("advance to cap");
        assert_eq!(c.offsets(), &[8, 8]);
    }

    #[test]
    fn gdcache_advance_mixed() {
        let mut c = make_cache_b(2, 16);
        c.advance(&[3, 12]).expect("advance mixed");
        assert_eq!(c.offsets(), &[3, 12]);
        c.advance(&[5, 0]).expect("advance row 0 only");
        assert_eq!(c.offsets(), &[8, 12]);
    }

    #[test]
    fn gdcache_advance_invalid_returns_err() {
        let mut c = make_cache_b(2, 4);
        // length mismatch
        let r = c.advance(&[1, 1, 1]);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("per_row_n.len()") || msg.contains("len"),
            "msg should mention len mismatch; got: {msg}"
        );

        // negative
        let r2 = c.advance(&[-1, 1]);
        assert!(r2.is_err());

        // exceed cap
        c.advance(&[2, 2]).unwrap();
        let r3 = c.advance(&[3, 1]); // 2+3=5 > cap=4 for row 0
        assert!(r3.is_err());
        let msg = format!("{}", r3.unwrap_err());
        assert!(
            msg.contains("cap") || msg.contains("exceeds"),
            "msg should mention cap; got: {msg}"
        );
    }

    #[test]
    fn gdcache_reset_clears_all_offsets() {
        let mut c = make_cache_b(2, 16);
        c.advance(&[4, 8]).unwrap();
        assert_eq!(c.offsets(), &[4, 8]);
        c.reset().expect("reset");
        assert_eq!(c.offsets(), &[0, 0]);
        // Shapes preserved.
        assert_eq!(c.conv_state().shape().as_slice(), &[2, 3, 8]);
        assert_eq!(c.recurrent_state().shape().as_slice(), &[2, 4, 8, 8]);
    }

    #[test]
    fn cache_rejects_zero_cap() {
        let r = GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 0);
        assert!(r.is_err());
    }

    #[test]
    fn cache_rejects_zero_kernel_size() {
        let r = GatedDeltaCache::new_with_cap(1, 0, 8, 4, 8, 8, Dtype::Bfloat16, 16);
        assert!(r.is_err());
    }
}
