//! KV caches for an MTP head's layers — one [`KVCache`] per layer.
//!
//! Mirrors the cap-bounded design of P2 [`crate::core::cache::KVCache`]: capacity
//! is fixed at construction; per-layer `KVCache::update_and_fetch_on` enforces
//! `offset ≤ cap` independently. `num_layers` is locked at construction and
//! validated by the consumer ([`crate::nn::Mtp::forward_on`]) at forward time.

use anyhow::anyhow;
use mlx::Dtype;

use crate::core::cache::KVCache;
use crate::Result;

/// KV caches for the layers of an MTP head.
pub struct MtpCache {
    layers: Vec<KVCache>,
}

impl MtpCache {
    /// Construct caches for `num_layers` layers, each a fresh [`KVCache`] with
    /// the same `cap`, `n_kv_heads`, `head_dim`, `v_head_dim`, and `dtype`.
    ///
    /// `num_layers` must be `> 0`. `cap` is the hard maximum sequence length
    /// (forwarded to each [`KVCache::new`] as its `cap` argument).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_cap(
        num_layers: usize,
        batch: i32,
        n_kv_heads: i32,
        head_dim: i32,
        v_head_dim: i32,
        dtype: Dtype,
        cap: i32,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(anyhow!("MtpCache::new_with_cap: num_layers must be > 0"));
        }
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(KVCache::new(
                batch, n_kv_heads, head_dim, v_head_dim, dtype, cap,
            ));
        }
        Ok(Self { layers })
    }

    /// Number of cached layers (fixed at construction).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Immutable view of one layer's cache.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.num_layers()` (Vec indexing).
    pub fn layer(&self, idx: usize) -> &KVCache {
        &self.layers[idx]
    }

    /// Mutable view of one layer's cache (used by the consumer's per-layer forward path).
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.num_layers()` (Vec indexing).
    pub fn layer_mut(&mut self, idx: usize) -> &mut KVCache {
        &mut self.layers[idx]
    }

    /// Reset every contained [`KVCache`] back to `offset = 0`. Buffers are retained for reuse.
    pub fn reset(&mut self) {
        for c in &mut self.layers {
            c.reset();
        }
    }

    /// Returns layer 0's offset.
    ///
    /// All layers are expected to advance in lock-step when driven through
    /// [`crate::nn::Mtp::forward_on`]. This is a caller-discipline contract,
    /// not a structural invariant — if a per-layer `update_and_fetch` errors
    /// mid-loop, layer 0's offset may diverge from later layers'. In any
    /// error-recovery path, call [`reset`](Self::reset) before reuse.
    pub fn offset(&self) -> i32 {
        self.layers.first().map(|c| c.offset()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(num_layers: usize) -> MtpCache {
        // Match P2 KVCache::new signature: (batch, n_kv_heads, head_dim, v_head_dim, dtype, cap)
        MtpCache::new_with_cap(num_layers, 1, 2, 8, 8, Dtype::Bfloat16, 16).expect("new_with_cap")
    }

    #[test]
    fn mtp_cache_new_with_cap_layers_and_zero_offset() {
        let cache = make_cache(3);
        assert_eq!(cache.num_layers(), 3);
        // All layer offsets start at 0; the wrapper exposes layer 0's offset by invariant.
        assert_eq!(cache.offset(), 0);
    }

    #[test]
    fn mtp_cache_reset_resets_all_layer_offsets() {
        let mut cache = make_cache(2);
        // Drive layer 0 forward by one update to advance its offset.
        let k0: mlx::Array = mlx::Array::zeros((1, 2, 4, 8), Dtype::Bfloat16).unwrap();
        let v0: mlx::Array = mlx::Array::zeros((1, 2, 4, 8), Dtype::Bfloat16).unwrap();
        cache.layer_mut(0).update_and_fetch(&k0, &v0).unwrap();
        // Drive layer 1 forward similarly.
        cache.layer_mut(1).update_and_fetch(&k0, &v0).unwrap();
        assert_eq!(cache.layer(0).offset(), 4);
        assert_eq!(cache.layer(1).offset(), 4);

        cache.reset();
        assert_eq!(cache.layer(0).offset(), 0);
        assert_eq!(cache.layer(1).offset(), 0);
        assert_eq!(cache.offset(), 0);
    }
}
