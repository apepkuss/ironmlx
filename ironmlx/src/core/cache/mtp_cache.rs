//! KV caches for an MTP head's layers — one [`KVCache`] per layer.
//!
//! Mirrors the cap-bounded design of P2 [`crate::core::cache::KVCache`]: capacity
//! is fixed at construction; per-layer `KVCache::update_and_fetch_on` enforces
//! `offset ≤ cap` independently. `num_layers` is locked at construction and
//! validated by the consumer ([`crate::nn::Mtp::forward_on`]) at forward time.

use anyhow::anyhow;
use mlx::{Dtype, StreamOrDevice};

use crate::core::cache::{KVCache, KVCacheSnapshot, PrefixMtpLayerPayload};
use crate::Result;

/// KV caches for the layers of an MTP head.
pub struct MtpCache {
    layers: Vec<KVCache>,
}

/// Lightweight checkpoint for [`MtpCache`] rollback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpCacheSnapshot {
    layers: Vec<KVCacheSnapshot>,
}

impl MtpCacheSnapshot {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, idx: usize) -> &KVCacheSnapshot {
        &self.layers[idx]
    }
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

    /// Capture every MTP layer's logical KV position. No K/V buffers are copied.
    pub fn snapshot(&self) -> MtpCacheSnapshot {
        MtpCacheSnapshot {
            layers: self.layers.iter().map(KVCache::snapshot).collect(),
        }
    }

    /// Restore every MTP layer's logical KV position from a prior checkpoint.
    pub fn restore(&mut self, snapshot: &MtpCacheSnapshot) -> Result<()> {
        if snapshot.layers.len() != self.layers.len() {
            return Err(anyhow!(
                "MtpCache::restore: snapshot layers {} != cache layers {}",
                snapshot.layers.len(),
                self.layers.len()
            ));
        }
        for (layer, layer_snapshot) in self.layers.iter_mut().zip(snapshot.layers.iter()) {
            layer.restore(layer_snapshot)?;
        }
        Ok(())
    }

    pub fn prefix_layers_for_row_on(
        &self,
        row: usize,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Vec<PrefixMtpLayerPayload>, i32)> {
        let target = target.into();
        let mut payloads = Vec::with_capacity(self.layers.len());
        let mut cached_len: Option<i32> = None;
        for (idx, layer) in self.layers.iter().enumerate() {
            let (k, v, layer_cached_len) = layer.dense_prefix_layer_for_row_on(row, target)?;
            if let Some(expected) = cached_len {
                if layer_cached_len != expected {
                    return Err(anyhow!(
                        "MtpCache::prefix_layers_for_row_on: layer {idx} cached_len {layer_cached_len} != layer0 {expected}"
                    ));
                }
            } else {
                cached_len = Some(layer_cached_len);
            }
            payloads.push(PrefixMtpLayerPayload { k, v });
        }
        Ok((payloads, cached_len.unwrap_or(0)))
    }

    pub fn restore_prefix_layers_for_row_on(
        &mut self,
        layers: &[PrefixMtpLayerPayload],
        row: usize,
        cached_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let target = target.into();
        if layers.len() != self.layers.len() {
            return Err(anyhow!(
                "MtpCache::restore_prefix_layers_for_row_on: payload layers {} != cache layers {}",
                layers.len(),
                self.layers.len()
            ));
        }
        for (idx, (cache_layer, payload)) in self.layers.iter_mut().zip(layers.iter()).enumerate() {
            cache_layer
                .restore_dense_prefix_layer_for_row_on(
                    &payload.k, &payload.v, row, cached_len, target,
                )
                .map_err(|err| {
                    anyhow!("MtpCache::restore_prefix_layers_for_row_on: layer {idx}: {err:#}")
                })?;
        }
        Ok(())
    }

    /// Returns layer 0's offset (the maximum offset across rows in the
    /// lockstep-uniform case).
    ///
    /// All layers are expected to advance in lock-step when driven through
    /// [`crate::nn::Mtp::forward_on`]. This is a caller-discipline contract,
    /// not a structural invariant — if a per-layer `update_and_fetch` errors
    /// mid-loop, layer 0's offset may diverge from later layers'. In any
    /// error-recovery path, call [`reset`](Self::reset) before reuse.
    pub fn offset(&self) -> i32 {
        self.layers
            .first()
            .and_then(|c| c.offsets().iter().copied().max())
            .unwrap_or(0)
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
        cache.layer_mut(0).update_and_fetch(&k0, &v0, &[4]).unwrap();
        // Drive layer 1 forward similarly.
        cache.layer_mut(1).update_and_fetch(&k0, &v0, &[4]).unwrap();
        assert_eq!(cache.layer(0).offsets(), &[4]);
        assert_eq!(cache.layer(1).offsets(), &[4]);

        cache.reset();
        assert_eq!(cache.layer(0).offsets(), &[0]);
        assert_eq!(cache.layer(1).offsets(), &[0]);
        assert_eq!(cache.offset(), 0);
    }

    #[test]
    fn mtp_cache_snapshot_restore_all_layers() {
        let mut cache = make_cache(2);
        let k4: mlx::Array = mlx::Array::zeros((1, 2, 4, 8), Dtype::Bfloat16).unwrap();
        let v4: mlx::Array = mlx::Array::zeros((1, 2, 4, 8), Dtype::Bfloat16).unwrap();
        cache.layer_mut(0).update_and_fetch(&k4, &v4, &[4]).unwrap();
        cache.layer_mut(1).update_and_fetch(&k4, &v4, &[4]).unwrap();
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.num_layers(), 2);
        assert_eq!(snapshot.layer(0).offsets(), &[4]);
        assert_eq!(snapshot.layer(1).offsets(), &[4]);

        cache.layer_mut(0).update_and_fetch(&k4, &v4, &[4]).unwrap();
        cache.layer_mut(1).update_and_fetch(&k4, &v4, &[4]).unwrap();
        assert_eq!(cache.layer(0).offsets(), &[8]);
        assert_eq!(cache.layer(1).offsets(), &[8]);

        cache.restore(&snapshot).expect("restore mtp snapshot");
        assert_eq!(cache.layer(0).offsets(), &[4]);
        assert_eq!(cache.layer(1).offsets(), &[4]);
    }

    #[test]
    fn mtp_cache_prefix_layers_round_trip_dense_kv() {
        let mut src = MtpCache::new_with_cap(1, 1, 1, 1, 1, Dtype::Float32, 8).expect("src");
        let k: mlx::Array = (&[1.0_f32, 2.0, 3.0][..], (1_i32, 1_i32, 3_i32, 1_i32))
            .try_into()
            .unwrap();
        let v: mlx::Array = (&[10.0_f32, 20.0, 30.0][..], (1_i32, 1_i32, 3_i32, 1_i32))
            .try_into()
            .unwrap();
        src.layer_mut(0)
            .update_and_fetch(&k, &v, &[3])
            .expect("fill src");

        let (layers, cached_len) = src.prefix_layers_for_row_on(0, ()).expect("export");

        let mut dst = MtpCache::new_with_cap(1, 1, 1, 1, 1, Dtype::Float32, 8).expect("dst");
        dst.restore_prefix_layers_for_row_on(&layers, 0, cached_len, ())
            .expect("restore");
        assert_eq!(dst.offset(), 3);

        let k_next: mlx::Array = (&[4.0_f32][..], (1_i32, 1_i32, 1_i32, 1_i32))
            .try_into()
            .unwrap();
        let v_next: mlx::Array = (&[40.0_f32][..], (1_i32, 1_i32, 1_i32, 1_i32))
            .try_into()
            .unwrap();
        let (k_full, v_full) = dst
            .layer_mut(0)
            .update_and_fetch(&k_next, &v_next, &[1])
            .expect("append after restore");
        assert_eq!(k_full.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            v_full.to_vec::<f32>().unwrap(),
            vec![10.0, 20.0, 30.0, 40.0]
        );
    }
}
