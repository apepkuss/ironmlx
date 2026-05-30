//! Latent MQA cache for GLM absorbed-MLA: normalized `c_kv[kv_lora]` + post-rope
//! `k_pe[rope]`, single kv head (`n_kv_heads = 1`).
//!
//! Mirrors [`crate::core::cache::KVCache`] EXACTLY (lazy alloc + step-grow via
//! `concatenate` + per-row `slice_update_on` loop + fetch `[0..max_offset]` +
//! per-row bounds checks + offset advance), but applied to TWO buffers of
//! differing last-dim width: `c_kv` `[B, 1, cap, kv_lora]` and `k_pe`
//! `[B, 1, cap, rope]`. The strategy is NOT reinvented — it is the verified
//! `KVCache::update_and_fetch_on` body duplicated across the two buffers.
//!
//! `step` default 256 + [`with_step`](MlaLatentCache::with_step)`(cap)` one-shot
//! prealloc avoids the cache-update Metal slow path (the KV-floor caveat is
//! about `cap`/`step`, NOT head-dim width; `c_kv` width 512 / `k_pe` width 64
//! are both fine).

use anyhow::{anyhow, Result};
use mlx::ops::indexing::{slice_strided_on, slice_update_on};
use mlx::ops::shape::concatenate_on;
use mlx::{Array, Dtype, StreamOrDevice};

/// Per-layer latent cache for GLM absorbed-MLA full-attention layers.
///
/// Holds the normalized compressed KV latent (`c_kv`) and the post-rope key
/// positional component (`k_pe`) pre-allocated up to `cap` tokens; grows in
/// `step`-size chunks via `concatenate`. Both buffers share `n_kv_heads = 1`
/// and a single per-row `offsets` vector — they advance in lockstep.
pub struct MlaLatentCache {
    c_kv: Option<Array>,
    k_pe: Option<Array>,
    offsets: Vec<i32>,
    cap: i32,
    step: i32,
    batch: i32,
    kv_lora: i32,
    rope: i32,
    dtype: Dtype,
}

impl MlaLatentCache {
    /// Construct a fresh cache. `c_kv` / `k_pe` are allocated lazily on first
    /// `update_and_fetch_on`. `cap` is the hard maximum sequence length.
    pub fn new(batch: i32, kv_lora: i32, rope: i32, dtype: Dtype, cap: i32) -> Self {
        Self {
            c_kv: None,
            k_pe: None,
            offsets: vec![0; batch as usize],
            cap,
            step: 256,
            batch,
            kv_lora,
            rope,
            dtype,
        }
    }

    /// Override grow step (default 256). `step >= cap` triggers one-shot
    /// preallocation. Panics if `step <= 0`.
    pub fn with_step(mut self, step: i32) -> Self {
        assert!(
            step > 0,
            "MlaLatentCache step must be positive (got {step})"
        );
        self.step = step;
        self
    }

    /// Per-row write offsets (length == batch). Row `i`'s next latent write
    /// lands at sequence position `offsets[i]`.
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }

    pub fn cap(&self) -> i32 {
        self.cap
    }

    /// Dtype used for both latent buffers.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Raise `self.cap` to `new_cap` if larger; no-op otherwise. Mirrors
    /// [`KVCache::grow_cap`](crate::core::cache::KVCache::grow_cap): the
    /// physical buffers are not reallocated here — they remain at their
    /// current capacity until the next `update_and_fetch_on` triggers
    /// `grow_to`. Shrinking is intentionally not supported.
    pub fn grow_cap(&mut self, new_cap: i32) {
        if new_cap > self.cap {
            self.cap = new_cap;
        }
    }

    /// Reset every row's offset to 0 and drop the backing buffers. Mirrors the
    /// plan-directed `LayerCache::Mla` reset semantics.
    pub fn reset(&mut self) -> Result<()> {
        self.c_kv = None;
        self.k_pe = None;
        for o in &mut self.offsets {
            *o = 0;
        }
        Ok(())
    }

    /// Append `(c_kv_new, k_pe_new)` and return slices covering all cached
    /// tokens. `c_kv_new` is `[B, 1, S, kv_lora]`, `k_pe_new` is `[B, 1, S, rope]`;
    /// returns full-history `[B, 1, L, kv_lora]` + `[B, 1, L, rope]` where
    /// `L = max(offsets_after)`.
    pub fn update_and_fetch_on(
        &mut self,
        c_kv_new: &Array,
        k_pe_new: &Array,
        per_row_lens: &[i32],
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        let target: StreamOrDevice = target.into();

        // Validate per_row_lens — mirrors KVCache::update_and_fetch_on.
        if per_row_lens.len() != self.batch as usize {
            return Err(anyhow!(
                "MlaLatentCache::update_and_fetch_on: per_row_lens.len()={} != batch={}",
                per_row_lens.len(),
                self.batch,
            ));
        }
        let c_kv_seq = c_kv_new.shape().as_slice()[2];
        let k_pe_seq = k_pe_new.shape().as_slice()[2];
        if c_kv_seq != k_pe_seq {
            return Err(anyhow!(
                "MlaLatentCache::update_and_fetch_on: c_kv seq {c_kv_seq} != k_pe seq {k_pe_seq}",
            ));
        }
        for (i, &n) in per_row_lens.iter().enumerate() {
            if n < 0 {
                return Err(anyhow!(
                    "MlaLatentCache::update_and_fetch_on: per_row_lens[{i}] = {n} must be >= 0",
                ));
            }
            if n > c_kv_seq {
                return Err(anyhow!(
                    "MlaLatentCache::update_and_fetch_on: per_row_lens[{i}] = {n} > seq = {c_kv_seq}",
                ));
            }
            let new_off = self.offsets[i] + n;
            if new_off > self.cap {
                return Err(anyhow!(
                    "MlaLatentCache cap {} exceeded on row {i}: offset {} + new {} = {}",
                    self.cap,
                    self.offsets[i],
                    n,
                    new_off,
                ));
            }
        }

        // All-zero fast path: every row skips its write. Return empty slices
        // along axis 2 without touching backing buffers (avoids a panic when
        // c_kv/k_pe are not yet allocated).
        if per_row_lens.iter().all(|&n| n == 0) {
            let empty_c_kv =
                Array::zeros_on((self.batch, 1_i32, 0_i32, self.kv_lora), self.dtype, target)?;
            let empty_k_pe =
                Array::zeros_on((self.batch, 1_i32, 0_i32, self.rope), self.dtype, target)?;
            return Ok((empty_c_kv, empty_k_pe));
        }

        // Compute the post-write max offset across rows (the L dim of the
        // returned fetched slices).
        let max_off_after: i32 = self
            .offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);

        // Ensure both backing buffers reach max_off_after along axis 2.
        let current_capacity = self
            .c_kv
            .as_ref()
            .map(|a| a.shape().as_slice()[2])
            .unwrap_or(0);
        if max_off_after > current_capacity {
            let target_capacity =
                ((max_off_after + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, target)?;
        }

        // Per-row slice_update_on loop. Each row writes its own
        // [offsets[i]..offsets[i]+per_row_lens[i]] slab into BOTH buffers.
        // Rows with per_row_lens[i] == 0 skip the write entirely.
        self.write_per_row(c_kv_new, k_pe_new, per_row_lens, target)?;

        // Bump per-row offsets after the writes complete.
        for (o, &n) in self.offsets.iter_mut().zip(per_row_lens.iter()) {
            *o += n;
        }

        // Return slices covering [0..max_off_after] along axis 2 for both
        // buffers. Rows with smaller per-row offsets have stale data at
        // positions [offsets[i]..max_off_after] — the caller masks those.
        let c_kv_full = self.c_kv.as_ref().expect("c_kv allocated");
        let k_pe_full = self.k_pe.as_ref().expect("k_pe allocated");
        let c_kv_slice = slice_strided_on(
            c_kv_full,
            [0_i32, 0, 0, 0],
            [self.batch, 1, max_off_after, self.kv_lora],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let k_pe_slice = slice_strided_on(
            k_pe_full,
            [0_i32, 0, 0, 0],
            [self.batch, 1, max_off_after, self.rope],
            [1_i32, 1, 1, 1],
            target,
        )?;
        Ok((c_kv_slice, k_pe_slice))
    }

    /// Grow both latent buffers to `new_capacity` along axis 2 (sequence
    /// dimension). Old contents are preserved at `[..., 0..max_offset, ...]`.
    /// Mirrors `KVCache::grow_to`, duplicated for the two differing widths.
    fn grow_to(&mut self, new_capacity: i32, target: StreamOrDevice) -> Result<()> {
        let max_off: i32 = self.offsets.iter().copied().max().unwrap_or(0);

        let new_c_kv = match (&self.c_kv, max_off) {
            (None, _) | (Some(_), 0) => Array::zeros_on(
                (self.batch, 1_i32, new_capacity, self.kv_lora),
                self.dtype,
                target,
            )?,
            (Some(old), _) => {
                let old_kept = slice_strided_on(
                    old,
                    [0_i32, 0, 0, 0],
                    [self.batch, 1, max_off, self.kv_lora],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                let tail = Array::zeros_on(
                    (self.batch, 1_i32, new_capacity - max_off, self.kv_lora),
                    self.dtype,
                    target,
                )?;
                concatenate_on(&[&old_kept, &tail], 2, target)?
            }
        };
        let new_k_pe = match (&self.k_pe, max_off) {
            (None, _) | (Some(_), 0) => Array::zeros_on(
                (self.batch, 1_i32, new_capacity, self.rope),
                self.dtype,
                target,
            )?,
            (Some(old), _) => {
                let old_kept = slice_strided_on(
                    old,
                    [0_i32, 0, 0, 0],
                    [self.batch, 1, max_off, self.rope],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                let tail = Array::zeros_on(
                    (self.batch, 1_i32, new_capacity - max_off, self.rope),
                    self.dtype,
                    target,
                )?;
                concatenate_on(&[&old_kept, &tail], 2, target)?
            }
        };
        self.c_kv = Some(new_c_kv);
        self.k_pe = Some(new_k_pe);
        Ok(())
    }

    /// Per-row latent write via a B-loop of `slice_update_on` calls. Each row
    /// `i` writes the leading `per_row_lens[i]` columns of `c_kv_new[i]` /
    /// `k_pe_new[i]` to positions `[offsets[i]..offsets[i]+per_row_lens[i]]`.
    /// Rows with `per_row_lens[i] == 0` skip the call. Mirrors
    /// `KVCache::write_per_row`, duplicated for the two buffers + the single
    /// batch-1 fast path.
    fn write_per_row(
        &mut self,
        c_kv_new: &Array,
        k_pe_new: &Array,
        per_row_lens: &[i32],
        target: StreamOrDevice,
    ) -> Result<()> {
        let seq = c_kv_new.shape().as_slice()[2];
        if self.batch == 1 && per_row_lens.len() == 1 && per_row_lens[0] == seq {
            let n = per_row_lens[0];
            if n == 0 {
                return Ok(());
            }
            let off = self.offsets[0];
            let end = off + n;
            let c_kv_full = self.c_kv.as_ref().expect("c_kv allocated by grow_to");
            let k_pe_full = self.k_pe.as_ref().expect("k_pe allocated by grow_to");
            let new_c_kv = slice_update_on(
                c_kv_full,
                c_kv_new,
                [0_i32, 0, off, 0],
                [1_i32, 1, end, self.kv_lora],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let new_k_pe = slice_update_on(
                k_pe_full,
                k_pe_new,
                [0_i32, 0, off, 0],
                [1_i32, 1, end, self.rope],
                [1_i32, 1, 1, 1],
                target,
            )?;
            self.c_kv = Some(new_c_kv);
            self.k_pe = Some(new_k_pe);
            return Ok(());
        }

        for (i_usize, &n) in per_row_lens.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let i = i_usize as i32;
            let off_i = self.offsets[i_usize];
            let end_i = off_i + n;

            // Slice the row's c_kv/k_pe leading-n along axis 2.
            let c_kv_row = slice_strided_on(
                c_kv_new,
                [i, 0, 0, 0],
                [i + 1, 1, n, self.kv_lora],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let k_pe_row = slice_strided_on(
                k_pe_new,
                [i, 0, 0, 0],
                [i + 1, 1, n, self.rope],
                [1_i32, 1, 1, 1],
                target,
            )?;

            // Write the row's leading-n slab at [i, :, off_i..end_i, :].
            let c_kv_full = self.c_kv.as_ref().expect("c_kv allocated by grow_to");
            let k_pe_full = self.k_pe.as_ref().expect("k_pe allocated by grow_to");
            let new_c_kv = slice_update_on(
                c_kv_full,
                &c_kv_row,
                [i, 0, off_i, 0],
                [i + 1, 1, end_i, self.kv_lora],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let new_k_pe = slice_update_on(
                k_pe_full,
                &k_pe_row,
                [i, 0, off_i, 0],
                [i + 1, 1, end_i, self.rope],
                [1_i32, 1, 1, 1],
                target,
            )?;
            self.c_kv = Some(new_c_kv);
            self.k_pe = Some(new_k_pe);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[batch=1, kv_head=1, seq, width]` f32 ramp.
    fn make_input(seq: i32, width: i32) -> Array {
        let total = (seq * width) as usize;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        (&data[..], (1_i32, 1_i32, seq, width)).try_into().unwrap()
    }

    #[test]
    fn append_and_fetch_two_steps() {
        let mut c = MlaLatentCache::new(1, 512, 64, Dtype::Float32, 8).with_step(8);
        assert_eq!(c.offsets(), &[0]);
        assert_eq!(c.cap(), 8);
        assert_eq!(c.dtype(), Dtype::Float32);

        // Step 1: append 2 tokens.
        let c_kv = make_input(2, 512);
        let k_pe = make_input(2, 64);
        let (kv_f, pe_f) = c.update_and_fetch_on(&c_kv, &k_pe, &[2], ()).unwrap();
        assert_eq!(kv_f.shape().as_slice(), &[1, 1, 2, 512]);
        assert_eq!(pe_f.shape().as_slice(), &[1, 1, 2, 64]);
        assert_eq!(c.offsets(), &[2]);

        // Step 2: append 1 more token.
        let c_kv2 = make_input(1, 512);
        let k_pe2 = make_input(1, 64);
        let (kv_f2, pe_f2) = c.update_and_fetch_on(&c_kv2, &k_pe2, &[1], ()).unwrap();
        assert_eq!(kv_f2.shape().as_slice(), &[1, 1, 3, 512]);
        assert_eq!(pe_f2.shape().as_slice(), &[1, 1, 3, 64]);
        assert_eq!(c.offsets(), &[3]);
    }

    #[test]
    fn rejects_wrong_per_row_lens_len() {
        let mut c = MlaLatentCache::new(2, 512, 64, Dtype::Float32, 8).with_step(8);
        let c_kv = make_input(1, 512);
        let k_pe = make_input(1, 64);
        let r = c.update_and_fetch_on(&c_kv, &k_pe, &[1], ());
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("per_row_lens.len()"),
            "msg should mention per_row_lens.len(); got: {msg}"
        );
    }

    #[test]
    fn data_preserved_across_steps() {
        // Verify step-1 data survives the step-2 grow/write and lands at the
        // correct offset, mirroring KVCache's accumulation guarantee.
        let mut c = MlaLatentCache::new(1, 4, 2, Dtype::Float32, 8).with_step(8);

        // Step 1: c_kv row of 1.0 (2 tokens × width 4), k_pe row of 7.0.
        let c_kv1: Array = (&[1.0_f32; 2 * 4][..], (1_i32, 1_i32, 2_i32, 4_i32))
            .try_into()
            .unwrap();
        let k_pe1: Array = (&[7.0_f32; 2 * 2][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        c.update_and_fetch_on(&c_kv1, &k_pe1, &[2], ()).unwrap();

        // Step 2: c_kv 3.0 (1 token), k_pe 9.0.
        let c_kv2: Array = (&[3.0_f32; 4][..], (1_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let k_pe2: Array = (&[9.0_f32; 2][..], (1_i32, 1_i32, 1_i32, 2_i32))
            .try_into()
            .unwrap();
        let (kv_f, pe_f) = c.update_and_fetch_on(&c_kv2, &k_pe2, &[1], ()).unwrap();

        // c_kv layout [1,1,3,4] row-major: tokens 0,1 = 1.0; token 2 = 3.0.
        let kv: Vec<f32> = kv_f.to_vec().unwrap();
        assert_eq!(kv.len(), 3 * 4);
        for v in kv.iter().take(2 * 4) {
            assert_eq!(*v, 1.0_f32, "c_kv step-1 corrupted");
        }
        for v in kv.iter().take(3 * 4).skip(2 * 4) {
            assert_eq!(*v, 3.0_f32, "c_kv step-2 wrong");
        }
        // k_pe layout [1,1,3,2]: tokens 0,1 = 7.0; token 2 = 9.0.
        let pe: Vec<f32> = pe_f.to_vec().unwrap();
        assert_eq!(pe.len(), 3 * 2);
        for v in pe.iter().take(2 * 2) {
            assert_eq!(*v, 7.0_f32, "k_pe step-1 corrupted");
        }
        for v in pe.iter().take(3 * 2).skip(2 * 2) {
            assert_eq!(*v, 9.0_f32, "k_pe step-2 wrong");
        }
    }

    #[test]
    fn rejects_cap_exceeded() {
        let mut c = MlaLatentCache::new(1, 512, 64, Dtype::Float32, 4).with_step(4);
        let c_kv = make_input(3, 512);
        let k_pe = make_input(3, 64);
        c.update_and_fetch_on(&c_kv, &k_pe, &[3], ()).unwrap();
        let c_kv2 = make_input(3, 512);
        let k_pe2 = make_input(3, 64);
        let r = c.update_and_fetch_on(&c_kv2, &k_pe2, &[3], ()); // 3+3 = 6 > cap 4
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("cap"), "msg should mention cap; got: {msg}");
    }
}
