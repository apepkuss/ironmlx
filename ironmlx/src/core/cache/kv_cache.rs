//! Per-layer KV cache for full-attention layers. See P2 spec § 3 for design.
//!
//! Implementation strategy: lazy alloc + step-rounded grow via concatenate;
//! per-update writes use Strategy A: a B-loop of `slice_update_on` calls
//! (one per row with per_row_lens[i] > 0). The public API (`new`, `with_step`,
//! `update_and_fetch`, `offsets`, `cap`, `reset`) is stable across strategies.

use mlx::ops::indexing::{slice_strided_on, slice_update_on};
use mlx::ops::shape::concatenate_on;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::Result;

/// Per-layer KV cache for full-attention layers.
///
/// Holds keys + values pre-allocated up to `cap` tokens; grows in
/// `step`-size chunks via `concatenate`. `update_and_fetch` takes a
/// per-row lens slice and returns slices covering [0..max(offsets_after)].
///
/// The cache is dense `[batch, n_kv_heads, cap, head_dim]` — NOT paged.
/// Rows with smaller per-row offsets have stale data above `offsets[i]`;
/// the caller masks those via per-row decode mask helpers (Task 3+).
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offsets: Vec<i32>,
    cap: i32,
    step: i32,
    batch: i32,
    n_kv_heads: i32,
    head_dim: i32,
    v_head_dim: i32,
    dtype: Dtype,
}

impl KVCache {
    /// Construct a fresh cache. Keys/values are allocated lazily on first
    /// `update_and_fetch`.
    ///
    /// `cap` is the hard maximum sequence length; callers compute it as
    /// `prompt_tokens + max_new_tokens`.
    pub fn new(
        batch: i32,
        n_kv_heads: i32,
        head_dim: i32,
        v_head_dim: i32,
        dtype: Dtype,
        cap: i32,
    ) -> Self {
        Self {
            keys: None,
            values: None,
            offsets: vec![0; batch as usize],
            cap,
            step: 256,
            batch,
            n_kv_heads,
            head_dim,
            v_head_dim,
            dtype,
        }
    }

    /// Override grow step (default 256). `step >= cap` triggers one-shot
    /// preallocation. Panics if `step <= 0`.
    pub fn with_step(mut self, step: i32) -> Self {
        assert!(step > 0, "KVCache step must be positive (got {step})");
        self.step = step;
        self
    }

    /// Per-row write offsets (length == batch). Row `i`'s next K/V write
    /// lands at sequence position `offsets[i]`.
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }

    pub fn cap(&self) -> i32 {
        self.cap
    }

    /// Reset every row's offset to 0; retains allocated buffers for reuse.
    pub fn reset(&mut self) {
        for o in &mut self.offsets {
            *o = 0;
        }
    }

    /// Append `(k, v)` and return slices covering all cached tokens.
    pub fn update_and_fetch(
        &mut self,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
    ) -> Result<(Array, Array)> {
        self.update_and_fetch_on(k, v, per_row_lens, ())
    }

    /// Stream-targeted variant.
    pub fn update_and_fetch_on(
        &mut self,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        let target: StreamOrDevice = target.into();

        // Validate per_row_lens. (Spec §4.7 invariants 3-5.)
        if per_row_lens.len() != self.batch as usize {
            anyhow::bail!(
                "KVCache::update_and_fetch_on: per_row_lens.len()={} != batch={}",
                per_row_lens.len(),
                self.batch,
            );
        }
        let k_seq = k.shape().as_slice()[2];
        for (i, &n) in per_row_lens.iter().enumerate() {
            if n < 0 {
                anyhow::bail!("KVCache::update_and_fetch_on: per_row_lens[{i}] = {n} must be >= 0",);
            }
            if n > k_seq {
                anyhow::bail!(
                    "KVCache::update_and_fetch_on: per_row_lens[{i}] = {n} > k.shape()[2] = {k_seq}",
                );
            }
            let new_off = self.offsets[i] + n;
            if new_off > self.cap {
                anyhow::bail!(
                    "KVCache cap {} exceeded on row {i}: offset {} + new {} = {}",
                    self.cap,
                    self.offsets[i],
                    n,
                    new_off,
                );
            }
        }

        // All-zero fast path: every row skips its write. Return empty slices
        // along axis 2 without touching backing buffers (avoids a panic when
        // keys/values are not yet allocated).
        if per_row_lens.iter().all(|&n| n == 0) {
            let empty_k = Array::zeros_on(
                (self.batch, self.n_kv_heads, 0_i32, self.head_dim),
                self.dtype,
                target,
            )?;
            let empty_v = Array::zeros_on(
                (self.batch, self.n_kv_heads, 0_i32, self.v_head_dim),
                self.dtype,
                target,
            )?;
            return Ok((empty_k, empty_v));
        }

        // Compute the post-write max offset across rows (the K dim of the
        // returned fetched slice).
        let max_off_after: i32 = self
            .offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);

        // Ensure backing buffers reach max_off_after along axis 2.
        let current_capacity = self
            .keys
            .as_ref()
            .map(|a| a.shape().as_slice()[2])
            .unwrap_or(0);
        if max_off_after > current_capacity {
            let target_capacity =
                ((max_off_after + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, target)?;
        }

        // Strategy A: per-row slice_update_on loop. Each row writes its own
        // [offsets[i]..offsets[i]+per_row_lens[i]] slab. Rows with
        // per_row_lens[i] == 0 skip the write entirely.
        self.write_per_row(k, v, per_row_lens, target)?;

        // Bump per-row offsets after the writes complete.
        for (o, &n) in self.offsets.iter_mut().zip(per_row_lens.iter()) {
            *o += n;
        }

        // Return slices covering [0..max_off_after] along axis 2. Rows with
        // smaller per-row offsets have stale data at positions
        // [offsets[i]..max_off_after] — the caller is responsible for
        // masking those out (via build_per_row_decode_mask for decode, via
        // build_batch_attention_mask for prefill).
        let keys_full = self.keys.as_ref().expect("keys allocated");
        let values_full = self.values.as_ref().expect("values allocated");
        let k_slice = slice_strided_on(
            keys_full,
            [0_i32, 0, 0, 0],
            [self.batch, self.n_kv_heads, max_off_after, self.head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v_slice = slice_strided_on(
            values_full,
            [0_i32, 0, 0, 0],
            [self.batch, self.n_kv_heads, max_off_after, self.v_head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        Ok((k_slice, v_slice))
    }

    /// Grow underlying K/V buffers to `new_capacity` along axis 2 (sequence
    /// dimension). Old contents are preserved at `[..., 0..max_offset, ...]`.
    fn grow_to(&mut self, new_capacity: i32, target: StreamOrDevice) -> Result<()> {
        // Preservation watermark: keep [..max_offset] of existing data so
        // every row's previously-written content survives the grow. Rows
        // with smaller offsets still have valid data in their slab below
        // their offset; rows above their offset are zero (post-allocation)
        // or stale (post-shrink) — caller-mask handles both.
        let max_off: i32 = self.offsets.iter().copied().max().unwrap_or(0);

        let new_k = match (&self.keys, max_off) {
            (None, _) | (Some(_), 0) => Array::zeros_on(
                (self.batch, self.n_kv_heads, new_capacity, self.head_dim),
                self.dtype,
                target,
            )?,
            (Some(old), _) => {
                let old_kept = slice_strided_on(
                    old,
                    [0_i32, 0, 0, 0],
                    [self.batch, self.n_kv_heads, max_off, self.head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                let tail = Array::zeros_on(
                    (
                        self.batch,
                        self.n_kv_heads,
                        new_capacity - max_off,
                        self.head_dim,
                    ),
                    self.dtype,
                    target,
                )?;
                concatenate_on(&[&old_kept, &tail], 2, target)?
            }
        };
        let new_v = match (&self.values, max_off) {
            (None, _) | (Some(_), 0) => Array::zeros_on(
                (self.batch, self.n_kv_heads, new_capacity, self.v_head_dim),
                self.dtype,
                target,
            )?,
            (Some(old), _) => {
                let old_kept = slice_strided_on(
                    old,
                    [0_i32, 0, 0, 0],
                    [self.batch, self.n_kv_heads, max_off, self.v_head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                let tail = Array::zeros_on(
                    (
                        self.batch,
                        self.n_kv_heads,
                        new_capacity - max_off,
                        self.v_head_dim,
                    ),
                    self.dtype,
                    target,
                )?;
                concatenate_on(&[&old_kept, &tail], 2, target)?
            }
        };
        self.keys = Some(new_k);
        self.values = Some(new_v);
        Ok(())
    }

    /// Strategy A: per-row K/V write via a B-loop of `slice_update_on`
    /// calls. Each row `i` writes the leading `per_row_lens[i]` columns of
    /// `k[i, :, :, :]` to `keys[i, :, offsets[i]..offsets[i]+per_row_lens[i], :]`.
    /// Rows with `per_row_lens[i] == 0` skip the call (no GPU dispatch).
    fn write_per_row(
        &mut self,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        target: StreamOrDevice,
    ) -> Result<()> {
        for (i_usize, &n) in per_row_lens.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let i = i_usize as i32;
            let off_i = self.offsets[i_usize];
            let end_i = off_i + n;

            // Slice the row's K/V leading-n along axis 2 (the K shape is
            // [batch, n_kv_heads, S_max, head_dim], so we take rows i:i+1
            // and seq 0:n).
            let k_row = slice_strided_on(
                k,
                [i, 0, 0, 0],
                [i + 1, self.n_kv_heads, n, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let v_row = slice_strided_on(
                v,
                [i, 0, 0, 0],
                [i + 1, self.n_kv_heads, n, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;

            // Write the row's leading-n slab at [i, :, off_i..end_i, :] in
            // the full keys/values buffer.
            let keys_full = self.keys.as_ref().expect("keys allocated by grow_to");
            let values_full = self.values.as_ref().expect("values allocated by grow_to");
            let new_keys = slice_update_on(
                keys_full,
                &k_row,
                [i, 0, off_i, 0],
                [i + 1, self.n_kv_heads, end_i, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let new_values = slice_update_on(
                values_full,
                &v_row,
                [i, 0, off_i, 0],
                [i + 1, self.n_kv_heads, end_i, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            self.keys = Some(new_keys);
            self.values = Some(new_values);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache_b(batch: i32, cap: i32) -> KVCache {
        KVCache::new(batch, 4, 256, 256, Dtype::Float32, cap)
    }

    fn make_kv_b(batch: i32, seq: i32) -> (Array, Array) {
        let total = (batch * 4 * seq * 256) as usize;
        let k_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..total).map(|i| (i as f32) * 10.0).collect();
        let k: Array = (&k_data[..], (batch, 4, seq, 256)).try_into().unwrap();
        let v: Array = (&v_data[..], (batch, 4, seq, 256)).try_into().unwrap();
        (k, v)
    }

    #[test]
    fn kvcache_per_row_offsets_initial_zero() {
        let c = make_cache_b(2, 1024);
        assert_eq!(c.offsets(), &[0, 0]);
        assert_eq!(c.cap(), 1024);
    }

    #[test]
    fn kvcache_per_row_write_uniform_lens() {
        let mut c = make_cache_b(2, 1024);
        let (k, v) = make_kv_b(2, 8);
        let (kf, vf) = c.update_and_fetch(&k, &v, &[8, 8]).expect("update uniform");
        assert_eq!(c.offsets(), &[8, 8]);
        assert_eq!(kf.shape().as_slice(), &[2, 4, 8, 256]);
        assert_eq!(vf.shape().as_slice(), &[2, 4, 8, 256]);
    }

    #[test]
    fn kvcache_per_row_write_mixed_lens() {
        let mut c = make_cache_b(2, 1024);
        let (k, v) = make_kv_b(2, 8);
        // Row 0 writes 4 tokens, row 1 writes 8 tokens. Returned slices have
        // K dim = max(offsets_after) = 8; row 0 positions 4..8 are stale.
        let (kf, _vf) = c.update_and_fetch(&k, &v, &[4, 8]).expect("update mixed");
        assert_eq!(c.offsets(), &[4, 8]);
        assert_eq!(kf.shape().as_slice(), &[2, 4, 8, 256]);
    }

    #[test]
    fn kvcache_per_row_zero_len_skips_row() {
        let mut c = make_cache_b(2, 1024);
        let (k, v) = make_kv_b(2, 8);
        let (_kf, _vf) = c.update_and_fetch(&k, &v, &[0, 8]).expect("update zero");
        assert_eq!(c.offsets(), &[0, 8], "row 0 unchanged, row 1 advanced");
    }

    #[test]
    fn kvcache_all_zero_lens_on_fresh_cache_returns_empty_slices() {
        // Regression: previously panicked with "keys allocated" because the
        // grow check skipped allocation when max_off_after == 0 and the
        // post-write fetch unwrapped a None keys buffer.
        let mut c = make_cache_b(2, 1024);
        let (k, v) = make_kv_b(2, 8);
        let (kf, vf) = c
            .update_and_fetch(&k, &v, &[0, 0])
            .expect("all-zero update should not panic");
        assert_eq!(c.offsets(), &[0, 0]);
        assert_eq!(kf.shape().as_slice(), &[2, 4, 0, 256]);
        assert_eq!(vf.shape().as_slice(), &[2, 4, 0, 256]);
    }

    #[test]
    fn kvcache_reset_clears_all_offsets() {
        let mut c = make_cache_b(2, 1024);
        let (k, v) = make_kv_b(2, 8);
        c.update_and_fetch(&k, &v, &[8, 8]).unwrap();
        assert_eq!(c.offsets(), &[8, 8]);
        c.reset();
        assert_eq!(c.offsets(), &[0, 0]);
    }

    #[test]
    fn kvcache_per_row_lens_len_mismatch_returns_err() {
        let mut c = make_cache_b(2, 1024);
        let (k, v) = make_kv_b(2, 8);
        let r = c.update_and_fetch(&k, &v, &[8, 8, 8]);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("per_row_lens.len()"),
            "msg should mention per_row_lens.len(); got: {msg}"
        );
    }

    #[test]
    fn kvcache_per_row_lens_negative_returns_err() {
        let mut c = make_cache_b(2, 1024);
        let (k, v) = make_kv_b(2, 8);
        let r = c.update_and_fetch(&k, &v, &[-1, 8]);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains(">= 0"), "msg should mention >= 0; got: {msg}");
    }

    #[test]
    fn kvcache_per_row_lens_exceeds_k_returns_err() {
        let mut c = make_cache_b(2, 1024);
        let (k, v) = make_kv_b(2, 8);
        let r = c.update_and_fetch(&k, &v, &[9, 8]); // 9 > k.shape()[2] == 8
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("k.shape()") || msg.contains("seq"),
            "msg should mention k seq dim; got: {msg}"
        );
    }

    #[test]
    fn kvcache_per_row_cap_exceeded_returns_err() {
        let mut c = make_cache_b(2, 10);
        let (k1, v1) = make_kv_b(2, 8);
        c.update_and_fetch(&k1, &v1, &[8, 8]).unwrap();
        let (k2, v2) = make_kv_b(2, 5);
        let r = c.update_and_fetch(&k2, &v2, &[5, 5]); // 8+5=13 > cap=10
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("cap"), "msg should mention cap; got: {msg}");
    }

    #[test]
    fn with_step_overrides_default() {
        let _ = KVCache::new(1, 4, 256, 256, Dtype::Float32, 4096).with_step(512);
    }

    #[test]
    #[should_panic(expected = "step must be positive")]
    fn with_step_panics_on_zero() {
        let _ = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024).with_step(0);
    }

    #[test]
    fn with_step_eq_cap_preallocates() {
        let mut c = KVCache::new(1, 4, 256, 256, Dtype::Float32, 64).with_step(64);
        let (k, v) = make_kv_b(1, 8);
        let (kf, _vf) = c.update_and_fetch(&k, &v, &[8]).unwrap();
        assert_eq!(kf.shape().as_slice(), &[1, 4, 8, 256]);
    }
}
