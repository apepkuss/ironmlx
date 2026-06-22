//! Gated Delta SSM cache: conv_state (sliding window) + recurrent_state (SSM state).
//!
//! Used by [`crate::nn::GatedDeltaNet`]. Mirrors P2 [`crate::core::cache::KVCache`]'s
//! cap-bounded design — capacity is fixed at construction; `advance` enforces
//! offset ≤ cap.

use anyhow::anyhow;
use mlx::ops::indexing::{slice_strided_on, slice_update_on};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::Result;

/// Per-layer cache for [`crate::nn::GatedDeltaNet`].
pub struct GatedDeltaCache {
    conv_state: Array,
    recurrent_state: Array,
    offsets: Vec<i32>,
    cap: i32,
}

/// Lightweight checkpoint for [`GatedDeltaCache`] rollback.
///
/// Unlike KV caches, GatedDelta state cannot be restored by offsets alone:
/// speculative suffixes update both the convolution window and recurrent
/// state. Cloning MLX arrays here preserves handles to the pre-verify state
/// without copying dense KV history.
#[derive(Clone)]
pub struct GatedDeltaCacheSnapshot {
    offsets: Vec<i32>,
    conv_state: Array,
    recurrent_state: Array,
}

impl GatedDeltaCacheSnapshot {
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }
}

impl GatedDeltaCache {
    /// Allocate a fresh cache.
    ///
    /// `cap` and `kernel_size` must each be ≥ 1. Both states start zero-initialized.
    /// `kernel_size = 1` is technically valid (guard passes) but produces a
    /// zero-width `conv_state` of shape `[B, 0, conv_dim]`, which means the
    /// conv1d sees no historical context. Typical usage has `kernel_size ≥ 2`
    /// (Qwen3.5 uses `kernel_size = 4`).
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

    /// Raise `self.cap` to `new_cap` if larger; no-op otherwise. `cap` is
    /// a purely logical bound here — `conv_state` and `recurrent_state`
    /// shapes are kernel/state-defined and do not depend on `cap`, so
    /// this is a single i32 field update with no buffer work.
    ///
    /// B1-p2.3f: paired with [`crate::core::cache::KVCache::grow_cap`],
    /// lets `Scheduler::admit_mid` extend the main cache when a new
    /// row's `prompt_len + max_new_tokens` exceeds the initial batch's
    /// cap. Shrinking is intentionally not supported.
    pub fn grow_cap(&mut self, new_cap: i32) {
        if new_cap > self.cap {
            self.cap = new_cap;
        }
    }

    /// Replace the conv_state with a freshly-computed sliding window.
    ///
    /// Caller is responsible for supplying shape `[B, kernel_size - 1, conv_dim]`
    /// matching the cache's allocation. No shape validation here — downstream
    /// conv1d dispatch surfaces shape mismatches.
    pub fn update_conv(&mut self, new_conv_state: Array) {
        self.conv_state = new_conv_state;
    }

    /// Replace the recurrent_state with the kernel's `state_out`.
    ///
    /// Caller is responsible for supplying shape `[B, Hv, Dv, Dk]` matching
    /// the cache's allocation, dtype fp32. No shape validation here — downstream
    /// kernel dispatch surfaces shape mismatches.
    pub fn update_recurrent(&mut self, new_state: Array) {
        self.recurrent_state = new_state;
    }

    /// Per-row offset bump. `per_row_n.len() == B`; each row `i` advances
    /// by `per_row_n[i]` tokens. Errors on length mismatch, negative entry,
    /// or `offsets[i] + per_row_n[i] > cap`.
    pub fn advance(&mut self, per_row_n: &[i32]) -> Result<()> {
        if per_row_n.len() != self.offsets.len() {
            return Err(anyhow!(
                "GatedDeltaCache::advance: per_row_n.len()={} != B={}",
                per_row_n.len(),
                self.offsets.len()
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

        self.offsets.fill(0);
        Ok(())
    }

    /// Capture offsets plus recurrent/conv state handles for rollback.
    pub fn snapshot(&self) -> GatedDeltaCacheSnapshot {
        GatedDeltaCacheSnapshot {
            offsets: self.offsets.clone(),
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
        }
    }

    /// Restore offsets and recurrent/conv states from a prior checkpoint.
    pub fn restore(&mut self, snapshot: &GatedDeltaCacheSnapshot) -> Result<()> {
        if snapshot.offsets.len() != self.offsets.len() {
            return Err(anyhow!(
                "GatedDeltaCache::restore: snapshot offsets len {} != B {}",
                snapshot.offsets.len(),
                self.offsets.len()
            ));
        }
        for (i, &off) in snapshot.offsets.iter().enumerate() {
            if off < 0 {
                return Err(anyhow!(
                    "GatedDeltaCache::restore: snapshot offsets[{i}] = {off} must be >= 0"
                ));
            }
            if off > self.cap {
                return Err(anyhow!(
                    "GatedDeltaCache::restore: snapshot offsets[{i}] = {off} > cap {}",
                    self.cap
                ));
            }
        }

        let self_conv = self.conv_state.shape();
        let snap_conv = snapshot.conv_state.shape();
        if self_conv.as_slice() != snap_conv.as_slice() {
            return Err(anyhow!(
                "GatedDeltaCache::restore: conv_state shape mismatch self {:?} snapshot {:?}",
                self_conv.as_slice(),
                snap_conv.as_slice()
            ));
        }
        let self_rec = self.recurrent_state.shape();
        let snap_rec = snapshot.recurrent_state.shape();
        if self_rec.as_slice() != snap_rec.as_slice() {
            return Err(anyhow!(
                "GatedDeltaCache::restore: recurrent_state shape mismatch self {:?} snapshot {:?}",
                self_rec.as_slice(),
                snap_rec.as_slice()
            ));
        }
        if self.conv_state.dtype() != snapshot.conv_state.dtype() {
            return Err(anyhow!(
                "GatedDeltaCache::restore: conv_state dtype mismatch self {:?} snapshot {:?}",
                self.conv_state.dtype(),
                snapshot.conv_state.dtype()
            ));
        }
        if self.recurrent_state.dtype() != snapshot.recurrent_state.dtype() {
            return Err(anyhow!(
                "GatedDeltaCache::restore: recurrent_state dtype mismatch self {:?} snapshot {:?}",
                self.recurrent_state.dtype(),
                snapshot.recurrent_state.dtype()
            ));
        }

        self.offsets.clone_from_slice(&snapshot.offsets);
        self.conv_state = snapshot.conv_state.clone();
        self.recurrent_state = snapshot.recurrent_state.clone();
        Ok(())
    }

    pub fn prefix_state_for_row_on(
        &self,
        row: usize,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array, i32)> {
        let target = target.into();
        let conv_dims = self.conv_state.shape();
        let conv_dims = conv_dims.as_slice();
        let rec_dims = self.recurrent_state.shape();
        let rec_dims = rec_dims.as_slice();
        if row >= self.offsets.len() {
            anyhow::bail!(
                "GatedDeltaCache::prefix_state_for_row_on: row {} >= B {}",
                row,
                self.offsets.len()
            );
        }
        let conv_state = slice_strided_on(
            &self.conv_state,
            [row as i32, 0, 0],
            [row as i32 + 1, conv_dims[1], conv_dims[2]],
            [1_i32, 1, 1],
            target,
        )?;
        let recurrent_state = slice_strided_on(
            &self.recurrent_state,
            [row as i32, 0, 0, 0],
            [row as i32 + 1, rec_dims[1], rec_dims[2], rec_dims[3]],
            [1_i32, 1, 1, 1],
            target,
        )?;
        Ok((conv_state, recurrent_state, self.offsets[row]))
    }

    pub fn restore_prefix_state_for_row_on(
        &mut self,
        conv_state: &Array,
        recurrent_state: &Array,
        row: usize,
        cached_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let target = target.into();
        if row >= self.offsets.len() {
            anyhow::bail!(
                "GatedDeltaCache::restore_prefix_state_for_row_on: row {} >= B {}",
                row,
                self.offsets.len()
            );
        }
        if cached_len < 0 || cached_len > self.cap {
            anyhow::bail!(
                "GatedDeltaCache::restore_prefix_state_for_row_on: cached_len {cached_len} outside [0, {}]",
                self.cap
            );
        }
        let self_conv = self.conv_state.shape();
        let self_conv = self_conv.as_slice();
        let conv_shape = conv_state.shape();
        let conv_shape = conv_shape.as_slice();
        if conv_shape != [1_i32, self_conv[1], self_conv[2]] {
            anyhow::bail!(
                "GatedDeltaCache::restore_prefix_state_for_row_on: conv_state shape {:?} incompatible with [1,{},{}]",
                conv_shape,
                self_conv[1],
                self_conv[2]
            );
        }
        if conv_state.dtype() != self.conv_state.dtype() {
            anyhow::bail!(
                "GatedDeltaCache::restore_prefix_state_for_row_on: conv_state dtype {} != {}",
                conv_state.dtype(),
                self.conv_state.dtype()
            );
        }
        let self_rec = self.recurrent_state.shape();
        let self_rec = self_rec.as_slice();
        let rec_shape = recurrent_state.shape();
        let rec_shape = rec_shape.as_slice();
        if rec_shape != [1_i32, self_rec[1], self_rec[2], self_rec[3]] {
            anyhow::bail!(
                "GatedDeltaCache::restore_prefix_state_for_row_on: recurrent_state shape {:?} incompatible with [1,{},{},{}]",
                rec_shape,
                self_rec[1],
                self_rec[2],
                self_rec[3]
            );
        }
        if recurrent_state.dtype() != self.recurrent_state.dtype() {
            anyhow::bail!(
                "GatedDeltaCache::restore_prefix_state_for_row_on: recurrent_state dtype {} != {}",
                recurrent_state.dtype(),
                self.recurrent_state.dtype()
            );
        }
        self.conv_state = slice_update_on(
            &self.conv_state,
            conv_state,
            [row as i32, 0, 0],
            [row as i32 + 1, self_conv[1], self_conv[2]],
            [1_i32, 1, 1],
            target,
        )?;
        self.recurrent_state = slice_update_on(
            &self.recurrent_state,
            recurrent_state,
            [row as i32, 0, 0, 0],
            [row as i32 + 1, self_rec[1], self_rec[2], self_rec[3]],
            [1_i32, 1, 1, 1],
            target,
        )?;
        self.offsets[row] = cached_len;
        Ok(())
    }

    /// Copy a single row's full SSM state from `src` into `self` at
    /// `dst_row`. The destination's `conv_state[dst_row, :, :]` and
    /// `recurrent_state[dst_row, :, :, :]` slabs are overwritten;
    /// `self.offsets[dst_row]` is set to `src.offsets[src_row]`.
    ///
    /// Unlike `KVCache::adopt_row_from`, the conv_state and recurrent_state
    /// slabs are written unconditionally (no `src_off == 0` skip). The SSM
    /// kernel reads state_in for every forward, so leaving stale state
    /// from a previous occupant would corrupt the next prefill's conv1d
    /// output. For a fresh src cache (zero-init), the adoption writes
    /// zeros into the dst slab — which is what we want.
    ///
    /// Requires matching `kernel_size - 1`, `conv_dim`, `Hv`, `Dv`, `Dk`
    /// between src and self. Batch dimensions may differ (typical usage:
    /// src.B = 1, self.B = b_max).
    ///
    /// Errors on conv_state / recurrent_state shape mismatch,
    /// dst_row >= self.B, src_row >= src.B, or src.offsets[src_row] > self.cap.
    pub fn adopt_row_from(
        &mut self,
        src: &GatedDeltaCache,
        dst_row: usize,
        src_row: usize,
    ) -> Result<()> {
        let self_conv_dims = self.conv_state.shape();
        let self_conv_dims = self_conv_dims.as_slice();
        let src_conv_dims = src.conv_state.shape();
        let src_conv_dims = src_conv_dims.as_slice();
        if self_conv_dims[1] != src_conv_dims[1] || self_conv_dims[2] != src_conv_dims[2] {
            anyhow::bail!(
                "GatedDeltaCache::adopt_row_from: conv_state shape mismatch (self [_,{},{}] src [_,{},{}])",
                self_conv_dims[1], self_conv_dims[2],
                src_conv_dims[1], src_conv_dims[2],
            );
        }
        let self_rec_dims = self.recurrent_state.shape();
        let self_rec_dims = self_rec_dims.as_slice();
        let src_rec_dims = src.recurrent_state.shape();
        let src_rec_dims = src_rec_dims.as_slice();
        if self_rec_dims[1] != src_rec_dims[1]
            || self_rec_dims[2] != src_rec_dims[2]
            || self_rec_dims[3] != src_rec_dims[3]
        {
            anyhow::bail!(
                "GatedDeltaCache::adopt_row_from: recurrent_state shape mismatch (self [_,{},{},{}] src [_,{},{},{}])",
                self_rec_dims[1], self_rec_dims[2], self_rec_dims[3],
                src_rec_dims[1], src_rec_dims[2], src_rec_dims[3],
            );
        }
        if dst_row >= self.offsets.len() {
            anyhow::bail!(
                "GatedDeltaCache::adopt_row_from: dst_row {} >= self.B {}",
                dst_row,
                self.offsets.len(),
            );
        }
        if src_row >= src.offsets.len() {
            anyhow::bail!(
                "GatedDeltaCache::adopt_row_from: src_row {} >= src.B {}",
                src_row,
                src.offsets.len(),
            );
        }
        let src_off = src.offsets[src_row];
        if src_off > self.cap {
            anyhow::bail!(
                "GatedDeltaCache::adopt_row_from: src.offsets[{}] = {} > self.cap {}",
                src_row,
                src_off,
                self.cap,
            );
        }

        let kernel_minus_one = self_conv_dims[1];
        let conv_dim = self_conv_dims[2];
        let hv = self_rec_dims[1];
        let dv = self_rec_dims[2];
        let dk = self_rec_dims[3];

        // Copy conv_state[src_row, :, :] -> self.conv_state[dst_row, :, :].
        let src_conv_slice = slice_strided_on(
            &src.conv_state,
            [src_row as i32, 0, 0],
            [src_row as i32 + 1, kernel_minus_one, conv_dim],
            [1_i32, 1, 1],
            (),
        )?;
        self.conv_state = slice_update_on(
            &self.conv_state,
            &src_conv_slice,
            [dst_row as i32, 0, 0],
            [dst_row as i32 + 1, kernel_minus_one, conv_dim],
            [1_i32, 1, 1],
            (),
        )?;

        // Copy recurrent_state[src_row, :, :, :] -> self.recurrent_state[dst_row, :, :, :].
        let src_rec_slice = slice_strided_on(
            &src.recurrent_state,
            [src_row as i32, 0, 0, 0],
            [src_row as i32 + 1, hv, dv, dk],
            [1_i32, 1, 1, 1],
            (),
        )?;
        self.recurrent_state = slice_update_on(
            &self.recurrent_state,
            &src_rec_slice,
            [dst_row as i32, 0, 0, 0],
            [dst_row as i32 + 1, hv, dv, dk],
            [1_i32, 1, 1, 1],
            (),
        )?;

        self.offsets[dst_row] = src_off;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    fn make_cache_b(b: i32, cap: i32) -> GatedDeltaCache {
        GatedDeltaCache::new_with_cap(b, 4, 8, 4, 8, 8, Dtype::Bfloat16, cap).expect("cache new")
    }

    fn filled_f32(shape: &[i32], value: f32) -> Array {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let data = vec![value; n];
        (&data[..], shape).try_into().unwrap()
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
    fn gdcache_advance_rejects_length_mismatch() {
        let mut c = make_cache_b(2, 4);
        let r = c.advance(&[1, 1, 1]);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("per_row_n.len()") || msg.contains("len"),
            "msg should mention len mismatch; got: {msg}"
        );
    }

    #[test]
    fn gdcache_advance_rejects_negative_entry() {
        let mut c = make_cache_b(2, 4);
        let r = c.advance(&[-1, 1]);
        assert!(r.is_err());
    }

    #[test]
    fn gdcache_advance_rejects_cap_exceeded() {
        let mut c = make_cache_b(2, 4);
        c.advance(&[2, 2]).unwrap(); // valid pre-state
        let r = c.advance(&[3, 1]); // 2+3=5 > cap=4 for row 0
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("cap") || msg.contains("exceeds"),
            "msg should mention cap; got: {msg}"
        );
    }

    #[test]
    fn gdcache_grow_cap_extends_and_allows_advance_beyond_initial_cap() {
        // Initial cap=4 — after advancing to 4, a further +6 would exceed
        // cap. grow_cap(32) lifts the limit so the advance succeeds.
        let mut c = make_cache_b(2, 4);
        c.advance(&[4, 4]).expect("advance to cap");
        assert_eq!(c.cap(), 4);

        c.grow_cap(32);
        assert_eq!(c.cap(), 32);

        c.advance(&[6, 6]).expect("advance past old cap");
        assert_eq!(c.offsets(), &[10, 10]);
    }

    #[test]
    fn gdcache_grow_cap_is_monotonic_noop_on_shrink() {
        let mut c = make_cache_b(1, 100);
        c.grow_cap(50);
        assert_eq!(c.cap(), 100);
        c.grow_cap(100);
        assert_eq!(c.cap(), 100);
        c.grow_cap(200);
        assert_eq!(c.cap(), 200);
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
    fn gdcache_snapshot_restore_offsets_and_states() {
        let mut c =
            GatedDeltaCache::new_with_cap(2, 4, 8, 4, 8, 8, Dtype::Float32, 16).expect("new");
        c.update_conv(filled_f32(&[2, 3, 8], 1.0));
        c.update_recurrent(filled_f32(&[2, 4, 8, 8], 2.0));
        c.advance(&[3, 5]).expect("advance snapshot state");
        let snapshot = c.snapshot();
        assert_eq!(snapshot.offsets(), &[3, 5]);

        c.update_conv(filled_f32(&[2, 3, 8], 9.0));
        c.update_recurrent(filled_f32(&[2, 4, 8, 8], 10.0));
        c.advance(&[2, 2]).expect("advance speculative suffix");
        assert_eq!(c.offsets(), &[5, 7]);

        c.restore(&snapshot).expect("restore snapshot");
        assert_eq!(c.offsets(), &[3, 5]);

        let conv: Vec<f32> = c.conv_state().to_vec().unwrap();
        assert!(conv.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        let recurrent: Vec<f32> = c.recurrent_state().to_vec().unwrap();
        assert!(recurrent.iter().all(|&v| (v - 2.0).abs() < 1e-6));
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
        let msg = format!("{}", r.err().unwrap());
        assert!(msg.contains("kernel_size"), "msg: {msg}");
    }

    #[test]
    fn gdcache_adopt_row_from_state_and_offset() {
        // src: B=1 cache. Mutate conv_state to all 1.0 (bf16) and
        // recurrent_state to all 2.0 (f32 — recurrent is always f32 per
        // new_with_cap). Advance offset to 4.
        let mut src =
            GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("src new");
        let conv_marker =
            mlx::ops::constructors::ones((1_i32, 3, 8), Dtype::Bfloat16).expect("conv_marker");
        src.update_conv(conv_marker);
        let rec_marker_f32: Array = (&vec![2.0_f32; 256][..], &[1_i32, 4, 8, 8][..])
            .try_into()
            .expect("rec_marker");
        src.update_recurrent(rec_marker_f32);
        src.advance(&[4]).expect("src advance");
        assert_eq!(src.offsets(), &[4]);

        // dst: B=2 cache, fresh (all zeros).
        let mut dst =
            GatedDeltaCache::new_with_cap(2, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("dst new");
        dst.adopt_row_from(&src, /*dst_row=*/ 1, /*src_row=*/ 0)
            .expect("adopt_row_from");

        assert_eq!(dst.offsets(), &[0, 4]);

        // Verify dst.conv_state[1, :, :] is all 1.0 (adopted from src)
        // and dst.conv_state[0, :, :] is all 0.0 (untouched).
        let conv_as_f32 =
            mlx::ops::cast::astype(dst.conv_state(), Dtype::Float32).expect("cast conv to f32");
        let conv_vec: Vec<f32> = conv_as_f32.to_vec().expect("conv to_vec");
        assert_eq!(conv_vec.len(), 2 * 3 * 8); // [B=2, k-1=3, conv_dim=8]
        let conv_stride_row = 3 * 8; // (k-1) * conv_dim
        for i in 0..conv_stride_row {
            assert_eq!(
                conv_vec[i], 0.0_f32,
                "dst.conv_state row 0 corrupted at {i}"
            );
        }
        for i in conv_stride_row..(2 * conv_stride_row) {
            assert_eq!(conv_vec[i], 1.0_f32, "dst.conv_state row 1 wrong at {i}");
        }

        // Verify dst.recurrent_state[1, :, :, :] is all 2.0 and [0, ...] is 0.0.
        let rec_vec: Vec<f32> = dst.recurrent_state().to_vec().expect("rec to_vec");
        assert_eq!(rec_vec.len(), 2 * 4 * 8 * 8); // [B=2, Hv=4, Dv=8, Dk=8]
        let rec_stride_row = 4 * 8 * 8;
        for i in 0..rec_stride_row {
            assert_eq!(rec_vec[i], 0.0_f32, "dst.rec row 0 corrupted at {i}");
        }
        for i in rec_stride_row..(2 * rec_stride_row) {
            assert_eq!(rec_vec[i], 2.0_f32, "dst.rec row 1 wrong at {i}");
        }
    }

    #[test]
    fn gdcache_adopt_row_from_out_of_bounds_err() {
        // Case 1: dst_row >= self.B
        let src =
            GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("src new");
        let mut dst =
            GatedDeltaCache::new_with_cap(2, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("dst new");
        let r = dst.adopt_row_from(&src, 2, 0);
        assert!(r.is_err(), "dst_row=2 with B=2 should Err");
        let msg = format!("{}", r.err().unwrap());
        assert!(
            msg.contains("dst_row") || msg.contains("B"),
            "msg should mention dst_row OOB; got: {msg}"
        );

        // Case 2: src_row >= src.B
        let src2 =
            GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("src2 new");
        let mut dst2 =
            GatedDeltaCache::new_with_cap(2, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("dst2 new");
        let r2 = dst2.adopt_row_from(&src2, 0, 1);
        assert!(r2.is_err(), "src_row=1 with src.B=1 should Err");
        let msg2 = format!("{}", r2.err().unwrap());
        assert!(
            msg2.contains("src_row") || msg2.contains("B"),
            "msg should mention src_row OOB; got: {msg2}"
        );

        // Case 3: src.offsets[src_row] > self.cap
        // src has cap=16, advance offset to 8. dst has cap=4 < src.offset=8 → Err.
        let mut src3 =
            GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("src3 new");
        src3.advance(&[8]).expect("src3 advance to 8");
        let mut dst3 =
            GatedDeltaCache::new_with_cap(2, 4, 8, 4, 8, 8, Dtype::Bfloat16, 4 /* cap=4 */)
                .expect("dst3 new with cap=4");
        let r3 = dst3.adopt_row_from(&src3, 0, 0);
        assert!(r3.is_err(), "src.offsets=8 > self.cap=4 should Err");
        let msg3 = format!("{}", r3.err().unwrap());
        assert!(
            msg3.contains("cap"),
            "msg should mention cap exceeded; got: {msg3}"
        );
    }

    #[test]
    fn gdcache_adopt_row_from_shape_mismatch_err() {
        // Case A: conv_state shape mismatch — different kernel_size.
        // src: kernel_size=4 → conv_state.dim[1] = 3
        // dst: kernel_size=6 → conv_state.dim[1] = 5
        let src_a = GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16)
            .expect("src_a new");
        let mut dst_a = GatedDeltaCache::new_with_cap(2, 6, 8, 4, 8, 8, Dtype::Bfloat16, 16)
            .expect("dst_a new");
        let r_a = dst_a.adopt_row_from(&src_a, 0, 0);
        assert!(r_a.is_err(), "conv_state kernel_size mismatch should Err");
        let msg_a = format!("{}", r_a.err().unwrap());
        assert!(
            msg_a.contains("conv_state") && (msg_a.contains("mismatch") || msg_a.contains("shape")),
            "msg should mention conv_state shape mismatch; got: {msg_a}"
        );

        // Case B: recurrent_state shape mismatch — different Hv.
        // src: hv=4 → recurrent_state.dim[1] = 4
        // dst: hv=8 → recurrent_state.dim[1] = 8
        let src_b = GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16)
            .expect("src_b new");
        let mut dst_b = GatedDeltaCache::new_with_cap(2, 4, 8, 8, 8, 8, Dtype::Bfloat16, 16)
            .expect("dst_b new");
        let r_b = dst_b.adopt_row_from(&src_b, 0, 0);
        assert!(r_b.is_err(), "recurrent_state hv mismatch should Err");
        let msg_b = format!("{}", r_b.err().unwrap());
        assert!(
            msg_b.contains("recurrent_state")
                && (msg_b.contains("mismatch") || msg_b.contains("shape")),
            "msg should mention recurrent_state shape mismatch; got: {msg_b}"
        );
    }

    #[test]
    fn gdcache_prefix_state_round_trips_single_row() {
        let mut src =
            GatedDeltaCache::new_with_cap(2, 4, 8, 4, 8, 8, Dtype::Float32, 16).expect("src");
        let mut conv_data = vec![0.0_f32; 2 * 3 * 8];
        for value in conv_data.iter_mut().skip(3 * 8) {
            *value = 7.0;
        }
        let conv: Array = (&conv_data[..], &[2_i32, 3, 8][..]).try_into().unwrap();
        let mut rec_data = vec![0.0_f32; 2 * 4 * 8 * 8];
        for value in rec_data.iter_mut().skip(4 * 8 * 8) {
            *value = 11.0;
        }
        let recurrent: Array = (&rec_data[..], &[2_i32, 4, 8, 8][..]).try_into().unwrap();
        src.update_conv(conv);
        src.update_recurrent(recurrent);
        src.advance(&[0, 5]).expect("advance row 1");

        let (conv_row, recurrent_row, cached_len) =
            src.prefix_state_for_row_on(1, ()).expect("export row");

        let mut dst =
            GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Float32, 16).expect("dst");
        dst.restore_prefix_state_for_row_on(&conv_row, &recurrent_row, 0, cached_len, ())
            .expect("restore row");

        assert_eq!(dst.offsets(), &[5]);
        assert!(dst
            .conv_state()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|&v| v == 7.0));
        assert!(dst
            .recurrent_state()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|&v| v == 11.0));
    }
}
