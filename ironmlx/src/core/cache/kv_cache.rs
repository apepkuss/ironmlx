//! Per-layer KV cache for full-attention layers. See P2 spec § 3 for design.
//!
//! Implementation strategy: mlx-lm-style concatenate (slice_update is not
//! bound in cxx-mlx). Each grow concatenates `[old_keys[..offset], k_new,
//! zeros[trailing]]` along axis 2. The public API (`new`, `with_step`,
//! `update_and_fetch`, `offset`, `cap`, `reset`) is stable across
//! implementation strategies.

use mlx::ops::indexing::slice_strided_on;
use mlx::ops::shape::concatenate_on;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::Result;

/// Per-layer KV cache for full-attention layers.
///
/// Holds keys + values pre-allocated up to `cap` tokens; grows in
/// `step`-size chunks via `concatenate`. `update_and_fetch` advances an
/// offset pointer and returns slices of the occupied region.
///
/// P2 supports single-request usage (one cache instance per layer per
/// request). Multi-request paged cache is P8/P9 work.
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
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
            offset: 0,
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

    pub fn offset(&self) -> i32 {
        self.offset
    }

    pub fn cap(&self) -> i32 {
        self.cap
    }

    /// Reset offset to 0; retains allocated buffers for reuse.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Append `(k, v)` and return slices covering all cached tokens.
    pub fn update_and_fetch(&mut self, k: &Array, v: &Array) -> Result<(Array, Array)> {
        self.update_and_fetch_on(k, v, ())
    }

    /// Stream-targeted variant.
    pub fn update_and_fetch_on(
        &mut self,
        k: &Array,
        v: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        let target: StreamOrDevice = target.into();

        let n_new = k.shape().as_slice()[2];
        let new_offset = self.offset + n_new;
        if new_offset > self.cap {
            anyhow::bail!(
                "KVCache cap {} exceeded by {} tokens (offset {} + new {})",
                self.cap,
                new_offset - self.cap,
                self.offset,
                n_new,
            );
        }

        let current_capacity = self
            .keys
            .as_ref()
            .map(|a| a.shape().as_slice()[2])
            .unwrap_or(0);

        if new_offset > current_capacity {
            // Round new offset up to next step boundary, clamped at cap.
            let target_capacity =
                ((new_offset + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, target)?;
        }

        // Write K/V at [..., offset..offset+n_new, ...] via concatenate:
        //   new_keys = concat([old_keys[..offset], k_new, old_keys[offset+n_new..]])
        // To avoid materializing the trailing zero region as a separate
        // operation, we exploit the pre-allocated buffer:
        //   keys = [..before, k_new, ..after_zero] all stitched once.
        self.write_at_offset(k, v, target)?;
        self.offset = new_offset;

        // Return slices covering [0..offset] along axis 2.
        let keys_full = self.keys.as_ref().expect("keys allocated");
        let values_full = self.values.as_ref().expect("values allocated");
        let k_slice = slice_strided_on(
            keys_full,
            [0_i32, 0, 0, 0],
            [self.batch, self.n_kv_heads, self.offset, self.head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v_slice = slice_strided_on(
            values_full,
            [0_i32, 0, 0, 0],
            [self.batch, self.n_kv_heads, self.offset, self.v_head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        Ok((k_slice, v_slice))
    }

    /// Grow underlying K/V buffers to `new_capacity` along axis 2 (sequence
    /// dimension). Old contents are preserved at `[..., 0..offset, ...]`.
    fn grow_to(&mut self, new_capacity: i32, target: StreamOrDevice) -> Result<()> {
        // Two cases combine into "fresh allocation":
        //   - keys is None: first-ever grow.
        //   - offset == 0: reset()-ed, no live data to preserve.
        // Otherwise: preserve [..offset] from the old buffer, append a
        // zeros tail of exactly `new_capacity - offset` rows along axis 2
        // (avoids materializing an extra `new_capacity` zeros buffer that
        // we'd then immediately slice).
        let new_k = match (&self.keys, self.offset) {
            (None, _) | (Some(_), 0) => Array::zeros_on(
                (self.batch, self.n_kv_heads, new_capacity, self.head_dim),
                self.dtype,
                target,
            )?,
            (Some(old), _) => {
                let old_kept = slice_strided_on(
                    old,
                    [0_i32, 0, 0, 0],
                    [self.batch, self.n_kv_heads, self.offset, self.head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                let tail = Array::zeros_on(
                    (
                        self.batch,
                        self.n_kv_heads,
                        new_capacity - self.offset,
                        self.head_dim,
                    ),
                    self.dtype,
                    target,
                )?;
                concatenate_on(&[&old_kept, &tail], 2, target)?
            }
        };
        let new_v = match (&self.values, self.offset) {
            (None, _) | (Some(_), 0) => Array::zeros_on(
                (self.batch, self.n_kv_heads, new_capacity, self.v_head_dim),
                self.dtype,
                target,
            )?,
            (Some(old), _) => {
                let old_kept = slice_strided_on(
                    old,
                    [0_i32, 0, 0, 0],
                    [self.batch, self.n_kv_heads, self.offset, self.v_head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                let tail = Array::zeros_on(
                    (
                        self.batch,
                        self.n_kv_heads,
                        new_capacity - self.offset,
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

    /// Write `k` / `v` into K/V buffers at `[..., self.offset..self.offset+n_new, ...]`.
    /// Without `slice_update`, we reconstruct the buffer via concatenate of
    /// three pieces: prefix[..offset], k_new, suffix[offset+n_new..].
    fn write_at_offset(&mut self, k: &Array, v: &Array, target: StreamOrDevice) -> Result<()> {
        let n_new = k.shape().as_slice()[2];
        let end = self.offset + n_new;
        let capacity = self
            .keys
            .as_ref()
            .map(|a| a.shape().as_slice()[2])
            .expect("keys allocated by grow_to");

        // Build new keys: [keys[..offset], k, keys[end..capacity]]
        // - if offset == 0: just [k, suffix]
        // - if end == capacity: just [prefix, k]
        // - else: 3-way concat
        let keys_full = self.keys.as_ref().expect("keys allocated");
        let new_keys = if self.offset == 0 && end == capacity {
            k.clone()
        } else if self.offset == 0 {
            let suffix = slice_strided_on(
                keys_full,
                [0_i32, 0, end, 0],
                [self.batch, self.n_kv_heads, capacity, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            concatenate_on(&[k, &suffix], 2, target)?
        } else if end == capacity {
            let prefix = slice_strided_on(
                keys_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, self.offset, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            concatenate_on(&[&prefix, k], 2, target)?
        } else {
            let prefix = slice_strided_on(
                keys_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, self.offset, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let suffix = slice_strided_on(
                keys_full,
                [0_i32, 0, end, 0],
                [self.batch, self.n_kv_heads, capacity, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            concatenate_on(&[&prefix, k, &suffix], 2, target)?
        };

        let values_full = self.values.as_ref().expect("values allocated");
        let new_values = if self.offset == 0 && end == capacity {
            v.clone()
        } else if self.offset == 0 {
            let suffix = slice_strided_on(
                values_full,
                [0_i32, 0, end, 0],
                [self.batch, self.n_kv_heads, capacity, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            concatenate_on(&[v, &suffix], 2, target)?
        } else if end == capacity {
            let prefix = slice_strided_on(
                values_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, self.offset, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            concatenate_on(&[&prefix, v], 2, target)?
        } else {
            let prefix = slice_strided_on(
                values_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, self.offset, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let suffix = slice_strided_on(
                values_full,
                [0_i32, 0, end, 0],
                [self.batch, self.n_kv_heads, capacity, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            concatenate_on(&[&prefix, v, &suffix], 2, target)?
        };

        self.keys = Some(new_keys);
        self.values = Some(new_values);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(cap: i32) -> KVCache {
        KVCache::new(1, 4, 256, 256, Dtype::Float32, cap)
    }

    fn make_kv(seq: i32) -> (Array, Array) {
        let total = (1 * 4 * seq * 256) as usize;
        let k_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..total).map(|i| (i as f32) * 10.0).collect();
        let k: Array = (&k_data[..], (1, 4, seq, 256)).try_into().unwrap();
        let v: Array = (&v_data[..], (1, 4, seq, 256)).try_into().unwrap();
        (k, v)
    }

    #[test]
    fn new_lazy_allocation_and_zero_offset() {
        let c = make_cache(1024);
        assert_eq!(c.offset(), 0);
        assert_eq!(c.cap(), 1024);
    }

    #[test]
    fn update_first_call_assigns_buffer_and_advances_offset() {
        let mut c = make_cache(1024);
        let (k, v) = make_kv(8);
        let (kf, vf) = c.update_and_fetch(&k, &v).expect("update");
        assert_eq!(c.offset(), 8);
        assert_eq!(kf.shape().as_slice(), &[1, 4, 8, 256]);
        assert_eq!(vf.shape().as_slice(), &[1, 4, 8, 256]);
    }

    #[test]
    fn returned_slices_match_written_data() {
        let mut c = make_cache(1024);
        let (k, v) = make_kv(4);
        let (kf, _vf) = c.update_and_fetch(&k, &v).expect("update");
        let kf_vec: Vec<f32> = kf.to_vec().unwrap();
        let k_vec: Vec<f32> = k.to_vec().unwrap();
        assert_eq!(kf_vec[..16], k_vec[..16]);
    }

    #[test]
    fn second_update_concatenates_and_grows_capacity() {
        let mut c = make_cache(1024);
        let (k1, v1) = make_kv(8);
        c.update_and_fetch(&k1, &v1).unwrap();
        let (k2, v2) = make_kv(4);
        let (kf, vf) = c.update_and_fetch(&k2, &v2).unwrap();
        assert_eq!(c.offset(), 12);
        assert_eq!(kf.shape().as_slice(), &[1, 4, 12, 256]);
        assert_eq!(vf.shape().as_slice(), &[1, 4, 12, 256]);
    }

    #[test]
    fn cap_exceeded_returns_error() {
        let mut c = make_cache(10);
        let (k1, _v1) = make_kv(8);
        let (_, _) = c.update_and_fetch(&k1, &_v1).unwrap();
        let (k2, v2) = make_kv(5);
        let r = c.update_and_fetch(&k2, &v2);
        assert!(r.is_err(), "expected cap exceeded error");
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("cap"), "msg should mention cap; got: {msg}");
    }

    #[test]
    fn reset_clears_offset_keeping_buffers() {
        let mut c = make_cache(1024);
        let (k, v) = make_kv(8);
        c.update_and_fetch(&k, &v).unwrap();
        assert_eq!(c.offset(), 8);
        c.reset();
        assert_eq!(c.offset(), 0);
        c.update_and_fetch(&k, &v).unwrap();
        assert_eq!(c.offset(), 8);
    }

    #[test]
    fn with_step_overrides_default() {
        let c = KVCache::new(1, 4, 256, 256, Dtype::Float32, 4096).with_step(512);
        let _ = c;
    }

    #[test]
    fn with_step_eq_cap_preallocates() {
        let mut c = KVCache::new(1, 4, 256, 256, Dtype::Float32, 64).with_step(64);
        let (k, v) = make_kv(8);
        let (kf, _vf) = c.update_and_fetch(&k, &v).unwrap();
        assert_eq!(kf.shape().as_slice(), &[1, 4, 8, 256]);
    }

    #[test]
    #[should_panic(expected = "step must be positive")]
    fn with_step_panics_on_zero() {
        let _ = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024).with_step(0);
    }
}
