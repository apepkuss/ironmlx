//! TurboQuant packed storage for full-attention KV caches.
//!
//! This module stores the packed TurboQuant representation for K/V history.
//! Prefill and unsupported reads can still materialize a packed prefix for the
//! dense SDPA path; decode reads can consume packed K/V directly.

use std::fmt;

use mlx::ops::indexing::{slice_strided_on, slice_update_on};
use mlx::ops::shape::concatenate_on;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::Result;

const TURBOQUANT_SIGN_SEED: u64 = 42;

/// TurboQuant bit-widths for the key and value sides of a full-attention KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TurboQuantKVBits {
    key_bits: u8,
    value_bits: u8,
}

impl TurboQuantKVBits {
    pub const K3V3: Self = Self {
        key_bits: 3,
        value_bits: 3,
    };
    pub const K4V4: Self = Self {
        key_bits: 4,
        value_bits: 4,
    };
    pub const K3V4: Self = Self {
        key_bits: 3,
        value_bits: 4,
    };

    pub fn new(key_bits: u8, value_bits: u8) -> Result<Self> {
        validate_bit_width("key_bits", key_bits)?;
        validate_bit_width("value_bits", value_bits)?;
        Ok(Self {
            key_bits,
            value_bits,
        })
    }

    pub fn key_bits(self) -> u8 {
        self.key_bits
    }

    pub fn value_bits(self) -> u8 {
        self.value_bits
    }

    pub fn cache_profile(self) -> String {
        format!("turboquant-k{}v{}", self.key_bits, self.value_bits)
    }
}

impl fmt::Display for TurboQuantKVBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "K{}V{}", self.key_bits, self.value_bits)
    }
}

/// TurboQuant metadata and packed K/V arrays for one full-attention layer.
pub struct TurboQuantKVCache {
    bits: TurboQuantKVBits,
    batch: i32,
    n_kv_heads: i32,
    head_dim: i32,
    v_head_dim: i32,
    packed_head_dim: i32,
    packed_v_head_dim: i32,
    cap: i32,
    step: i32,
    k_signs: Array,
    k_codebook: Array,
    v_signs: Array,
    v_codebook: Array,
    k_packed: Option<Array>,
    k_norms: Option<Array>,
    v_packed: Option<Array>,
    v_norms: Option<Array>,
}

#[derive(Debug, Clone)]
pub struct TurboQuantPrefixLayer {
    pub k_packed: Array,
    pub k_norms: Array,
    pub v_packed: Array,
    pub v_norms: Array,
}

impl TurboQuantKVCache {
    pub fn new(
        batch: i32,
        n_kv_heads: i32,
        head_dim: i32,
        v_head_dim: i32,
        cap: i32,
        step: i32,
        bits: TurboQuantKVBits,
    ) -> Result<Self> {
        validate_positive("batch", batch)?;
        validate_positive("n_kv_heads", n_kv_heads)?;
        validate_positive("cap", cap)?;
        validate_positive("step", step)?;
        validate_dim("head_dim", head_dim)?;
        validate_dim("v_head_dim", v_head_dim)?;

        let k_signs = signs_array(head_dim)?;
        let v_signs = signs_array(v_head_dim)?;
        let k_codebook = codebook_array(bits.key_bits(), head_dim)?;
        let v_codebook = codebook_array(bits.value_bits(), v_head_dim)?;

        Ok(Self {
            bits,
            batch,
            n_kv_heads,
            head_dim,
            v_head_dim,
            packed_head_dim: packed_dim(head_dim, bits.key_bits()),
            packed_v_head_dim: packed_dim(v_head_dim, bits.value_bits()),
            cap,
            step,
            k_signs,
            k_codebook,
            v_signs,
            v_codebook,
            k_packed: None,
            k_norms: None,
            v_packed: None,
            v_norms: None,
        })
    }

    pub fn bits(&self) -> TurboQuantKVBits {
        self.bits
    }

    pub fn key_bits(&self) -> u8 {
        self.bits.key_bits()
    }

    pub fn value_bits(&self) -> u8 {
        self.bits.value_bits()
    }

    pub fn head_dim(&self) -> i32 {
        self.head_dim
    }

    pub fn v_head_dim(&self) -> i32 {
        self.v_head_dim
    }

    pub fn packed_head_dim(&self) -> i32 {
        self.packed_head_dim
    }

    pub fn packed_v_head_dim(&self) -> i32 {
        self.packed_v_head_dim
    }

    pub fn k_packed(&self) -> Option<&Array> {
        self.k_packed.as_ref()
    }

    pub fn k_norms(&self) -> Option<&Array> {
        self.k_norms.as_ref()
    }

    pub fn v_packed(&self) -> Option<&Array> {
        self.v_packed.as_ref()
    }

    pub fn v_norms(&self) -> Option<&Array> {
        self.v_norms.as_ref()
    }

    pub(crate) fn key_signs(&self) -> &Array {
        &self.k_signs
    }

    pub fn clear(&mut self) {
        self.k_packed = None;
        self.k_norms = None;
        self.v_packed = None;
        self.v_norms = None;
    }

    pub fn grow_cap(&mut self, new_cap: i32) {
        if new_cap > self.cap {
            self.cap = new_cap;
        }
    }

    pub fn set_step(&mut self, step: i32) {
        assert!(
            step > 0,
            "TurboQuantKVCache step must be positive (got {step})"
        );
        self.step = step;
    }

    pub fn capacity(&self) -> i32 {
        self.k_packed
            .as_ref()
            .map(|a| a.shape().as_slice()[2])
            .unwrap_or(0)
    }

    pub fn update_from_dense_on(
        &mut self,
        k: &Array,
        v: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        validate_dense_shape("k", k, self.head_dim)?;
        validate_dense_shape("v", v, self.v_head_dim)?;
        validate_dense_batch_heads("k", k, self.batch, self.n_kv_heads)?;
        validate_dense_batch_heads("v", v, self.batch, self.n_kv_heads)?;

        let target = target.into();
        let (k_packed, k_norms) = mlx::fast::turbo_quantize_on(
            k,
            &self.k_signs,
            &self.k_codebook,
            self.bits.key_bits(),
            target,
        )?;
        let (v_packed, v_norms) = mlx::fast::turbo_quantize_on(
            v,
            &self.v_signs,
            &self.v_codebook,
            self.bits.value_bits(),
            target,
        )?;

        self.k_packed = Some(k_packed);
        self.k_norms = Some(k_norms);
        self.v_packed = Some(v_packed);
        self.v_norms = Some(v_norms);
        Ok(())
    }

    pub fn update_and_fetch_on(
        &mut self,
        k: &Array,
        v: &Array,
        offsets: &mut [i32],
        per_row_lens: &[i32],
        output_dtype: Dtype,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        validate_dense_shape("k", k, self.head_dim)?;
        validate_dense_shape("v", v, self.v_head_dim)?;
        validate_dense_batch_heads("k", k, self.batch, self.n_kv_heads)?;
        validate_dense_batch_heads("v", v, self.batch, self.n_kv_heads)?;
        if offsets.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_fetch_on: offsets.len()={} != batch={}",
                offsets.len(),
                self.batch,
            );
        }
        if per_row_lens.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_fetch_on: per_row_lens.len()={} != batch={}",
                per_row_lens.len(),
                self.batch,
            );
        }

        let k_seq = k.shape().as_slice()[2];
        for (i, &n) in per_row_lens.iter().enumerate() {
            if n < 0 {
                anyhow::bail!(
                    "TurboQuantKVCache::update_and_fetch_on: per_row_lens[{i}] = {n} must be >= 0",
                );
            }
            if n > k_seq {
                anyhow::bail!(
                    "TurboQuantKVCache::update_and_fetch_on: per_row_lens[{i}] = {n} > k.shape()[2] = {k_seq}",
                );
            }
            let new_off = offsets[i] + n;
            if new_off > self.cap {
                anyhow::bail!(
                    "TurboQuantKVCache cap {} exceeded on row {i}: offset {} + new {} = {}",
                    self.cap,
                    offsets[i],
                    n,
                    new_off,
                );
            }
        }

        let max_off_after = offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);
        if per_row_lens.iter().all(|&n| n == 0) {
            if max_off_after == 0 {
                let target = target.into();
                let empty_k = Array::zeros_on(
                    (self.batch, self.n_kv_heads, 0_i32, self.head_dim),
                    output_dtype,
                    target,
                )?;
                let empty_v = Array::zeros_on(
                    (self.batch, self.n_kv_heads, 0_i32, self.v_head_dim),
                    output_dtype,
                    target,
                )?;
                return Ok((empty_k, empty_v));
            }
            return self.materialize_prefix_on(max_off_after, output_dtype, target);
        }

        let target = target.into();
        if max_off_after > self.capacity() {
            let target_capacity =
                ((max_off_after + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, offsets, target)?;
        }
        self.write_per_row(k, v, offsets, per_row_lens, target)?;
        for (o, &n) in offsets.iter_mut().zip(per_row_lens.iter()) {
            *o += n;
        }
        self.materialize_prefix_on(max_off_after, output_dtype, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_and_attend_decode_on(
        &mut self,
        queries: &Array,
        k: &Array,
        v: &Array,
        offsets: &mut [i32],
        per_row_lens: &[i32],
        scale: f32,
        mask_arr: Option<&Array>,
        output_dtype: Dtype,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        validate_dense_shape("k", k, self.head_dim)?;
        validate_dense_shape("v", v, self.v_head_dim)?;
        validate_dense_batch_heads("k", k, self.batch, self.n_kv_heads)?;
        validate_dense_batch_heads("v", v, self.batch, self.n_kv_heads)?;
        if self.head_dim != self.v_head_dim {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_on: value head dim {} must match key/query head dim {}",
                self.v_head_dim,
                self.head_dim,
            );
        }
        if offsets.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_on: offsets.len()={} != batch={}",
                offsets.len(),
                self.batch,
            );
        }
        if per_row_lens.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_on: per_row_lens.len()={} != batch={}",
                per_row_lens.len(),
                self.batch,
            );
        }

        let k_seq = k.shape().as_slice()[2];
        if k_seq != 1 {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_on: k seq dim must be 1 for decode (got {k_seq})",
            );
        }
        for (i, &n) in per_row_lens.iter().enumerate() {
            if n != 1 {
                anyhow::bail!(
                    "TurboQuantKVCache::update_and_attend_decode_on: per_row_lens[{i}] = {n}, expected 1 for decode",
                );
            }
            let new_off = offsets[i] + n;
            if new_off > self.cap {
                anyhow::bail!(
                    "TurboQuantKVCache cap {} exceeded on row {i}: offset {} + new {} = {}",
                    self.cap,
                    offsets[i],
                    n,
                    new_off,
                );
            }
        }

        let max_off_after = offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);
        let requires_mask = offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .any(|off| off != max_off_after);
        if requires_mask && mask_arr.is_none() {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_on: ragged decode requires an explicit mask",
            );
        }

        let target = target.into();
        if max_off_after > self.capacity() {
            let target_capacity =
                ((max_off_after + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, offsets, target)?;
        }
        self.write_per_row(k, v, offsets, per_row_lens, target)?;
        for (o, &n) in offsets.iter_mut().zip(per_row_lens.iter()) {
            *o += n;
        }

        let k_packed = self.k_packed.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::update_and_attend_decode_on: missing K")
        })?;
        let k_norms = self.k_norms.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::update_and_attend_decode_on: missing K norms")
        })?;
        let v_packed = self.v_packed.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::update_and_attend_decode_on: missing V")
        })?;
        let v_norms = self.v_norms.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::update_and_attend_decode_on: missing V norms")
        })?;
        let k_packed = slice_strided_on(
            k_packed,
            [0_i32, 0, 0, 0],
            [
                self.batch,
                self.n_kv_heads,
                max_off_after,
                self.packed_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let k_norms = slice_strided_on(
            k_norms,
            [0_i32, 0, 0],
            [self.batch, self.n_kv_heads, max_off_after],
            [1_i32, 1, 1],
            target,
        )?;
        let v_packed = slice_strided_on(
            v_packed,
            [0_i32, 0, 0, 0],
            [
                self.batch,
                self.n_kv_heads,
                max_off_after,
                self.packed_v_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v_norms = slice_strided_on(
            v_norms,
            [0_i32, 0, 0],
            [self.batch, self.n_kv_heads, max_off_after],
            [1_i32, 1, 1],
            target,
        )?;

        let output = mlx::fast::turboquant_sdpa_decode_on(
            queries,
            &k_packed,
            &k_norms,
            &v_packed,
            &v_norms,
            &self.k_signs,
            &self.k_codebook,
            &self.v_signs,
            &self.v_codebook,
            scale,
            self.bits.key_bits(),
            self.bits.value_bits(),
            mask_arr,
            output_dtype,
            target,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_and_attend_multirow_on(
        &mut self,
        queries: &Array,
        k: &Array,
        v: &Array,
        offsets: &mut [i32],
        per_row_lens: &[i32],
        scale: f32,
        mask_arr: Option<&Array>,
        output_dtype: Dtype,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        validate_dense_shape("k", k, self.head_dim)?;
        validate_dense_shape("v", v, self.v_head_dim)?;
        validate_dense_batch_heads("k", k, self.batch, self.n_kv_heads)?;
        validate_dense_batch_heads("v", v, self.batch, self.n_kv_heads)?;
        if self.head_dim != self.v_head_dim {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_multirow_on: value head dim {} must match key/query head dim {}",
                self.v_head_dim,
                self.head_dim,
            );
        }
        if offsets.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_multirow_on: offsets.len()={} != batch={}",
                offsets.len(),
                self.batch,
            );
        }
        if per_row_lens.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_multirow_on: per_row_lens.len()={} != batch={}",
                per_row_lens.len(),
                self.batch,
            );
        }

        let q_shape = queries.shape();
        let q_dims = q_shape.as_slice();
        if q_dims.len() != 4
            || q_dims[0] != self.batch
            || q_dims[1] % self.n_kv_heads != 0
            || q_dims[3] != self.head_dim
        {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_multirow_on: queries must be [B,Hq,Q,D] with B={}, D={}, Hq divisible by Hkv={} (got {:?})",
                self.batch,
                self.head_dim,
                self.n_kv_heads,
                q_dims,
            );
        }
        let q_rows = q_dims[2];
        if !(2..=mlx::fast::TURBOQUANT_MULTIROW_MAX_QUERY_ROWS).contains(&q_rows) {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_multirow_on: query rows must be in [2, {}] (got {q_rows})",
                mlx::fast::TURBOQUANT_MULTIROW_MAX_QUERY_ROWS,
            );
        }
        let k_seq = k.shape().as_slice()[2];
        if k_seq != q_rows || v.shape().as_slice()[2] != q_rows {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_multirow_on: K/V seq dims must match query rows {q_rows} (got K={k_seq}, V={})",
                v.shape().as_slice()[2],
            );
        }
        for (i, &n) in per_row_lens.iter().enumerate() {
            if n <= 0 || n > q_rows {
                anyhow::bail!(
                    "TurboQuantKVCache::update_and_attend_multirow_on: per_row_lens[{i}] = {n} must be in [1, {q_rows}]",
                );
            }
            let new_off = offsets[i] + n;
            if new_off > self.cap {
                anyhow::bail!(
                    "TurboQuantKVCache cap {} exceeded on row {i}: offset {} + new {} = {}",
                    self.cap,
                    offsets[i],
                    n,
                    new_off,
                );
            }
        }

        let max_off_after = offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);
        let kv_lens = offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(offset, len)| offset + len)
            .collect::<Vec<_>>();

        let target = target.into();
        if max_off_after > self.capacity() {
            let target_capacity =
                ((max_off_after + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, offsets, target)?;
        }
        self.write_per_row(k, v, offsets, per_row_lens, target)?;
        for (o, &n) in offsets.iter_mut().zip(per_row_lens.iter()) {
            *o += n;
        }

        let k_packed = self.k_packed.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::update_and_attend_multirow_on: missing K")
        })?;
        let k_norms = self.k_norms.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::update_and_attend_multirow_on: missing K norms")
        })?;
        let v_packed = self.v_packed.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::update_and_attend_multirow_on: missing V")
        })?;
        let v_norms = self.v_norms.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::update_and_attend_multirow_on: missing V norms")
        })?;
        let k_packed = slice_strided_on(
            k_packed,
            [0_i32, 0, 0, 0],
            [
                self.batch,
                self.n_kv_heads,
                max_off_after,
                self.packed_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let k_norms = slice_strided_on(
            k_norms,
            [0_i32, 0, 0],
            [self.batch, self.n_kv_heads, max_off_after],
            [1_i32, 1, 1],
            target,
        )?;
        let v_packed = slice_strided_on(
            v_packed,
            [0_i32, 0, 0, 0],
            [
                self.batch,
                self.n_kv_heads,
                max_off_after,
                self.packed_v_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v_norms = slice_strided_on(
            v_norms,
            [0_i32, 0, 0],
            [self.batch, self.n_kv_heads, max_off_after],
            [1_i32, 1, 1],
            target,
        )?;
        let query_lens: Array = (per_row_lens, &[self.batch][..]).try_into()?;
        let kv_lens: Array = (kv_lens.as_slice(), &[self.batch][..]).try_into()?;

        Ok(mlx::fast::turboquant_sdpa_multirow_on(
            queries,
            &k_packed,
            &k_norms,
            &v_packed,
            &v_norms,
            &self.k_signs,
            &self.k_codebook,
            &self.v_signs,
            &self.v_codebook,
            scale,
            self.bits.key_bits(),
            self.bits.value_bits(),
            &query_lens,
            &kv_lens,
            mask_arr,
            output_dtype,
            target,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_and_attend_decode_pre_rotated_on(
        &mut self,
        q_rot: &Array,
        k: &Array,
        v: &Array,
        offsets: &mut [i32],
        per_row_lens: &[i32],
        scale: f32,
        mask_arr: Option<&Array>,
        output_dtype: Dtype,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        validate_dense_shape("k", k, self.head_dim)?;
        validate_dense_shape("v", v, self.v_head_dim)?;
        validate_dense_batch_heads("k", k, self.batch, self.n_kv_heads)?;
        validate_dense_batch_heads("v", v, self.batch, self.n_kv_heads)?;
        if self.head_dim != self.v_head_dim {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: value head dim {} must match key/query head dim {}",
                self.v_head_dim,
                self.head_dim,
            );
        }
        let q_shape = q_rot.shape();
        let q_dims = q_shape.as_slice();
        if q_dims.len() != 3
            || q_dims[0] != self.batch
            || q_dims[2] != self.head_dim
            || q_dims[1] % self.n_kv_heads != 0
        {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: q_rot must be [B,Hq,D] with B={}, D={}, Hq divisible by Hkv={} (got {:?})",
                self.batch,
                self.head_dim,
                self.n_kv_heads,
                q_dims
            );
        }
        if q_rot.dtype() != Dtype::Float32 {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: q_rot dtype must be Float32 (got {:?})",
                q_rot.dtype()
            );
        }
        if offsets.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: offsets.len()={} != batch={}",
                offsets.len(),
                self.batch,
            );
        }
        if per_row_lens.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: per_row_lens.len()={} != batch={}",
                per_row_lens.len(),
                self.batch,
            );
        }

        let k_seq = k.shape().as_slice()[2];
        if k_seq != 1 {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: k seq dim must be 1 for decode (got {k_seq})",
            );
        }
        for (i, &n) in per_row_lens.iter().enumerate() {
            if n != 1 {
                anyhow::bail!(
                    "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: per_row_lens[{i}] = {n}, expected 1 for decode",
                );
            }
            let new_off = offsets[i] + n;
            if new_off > self.cap {
                anyhow::bail!(
                    "TurboQuantKVCache cap {} exceeded on row {i}: offset {} + new {} = {}",
                    self.cap,
                    offsets[i],
                    n,
                    new_off,
                );
            }
        }

        let max_off_after = offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);
        if max_off_after < mlx::fast::TURBOQUANT_PARALLEL_DECODE_SEQ_THRESHOLD {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: seq_len {max_off_after} is below parallel decode threshold {}",
                mlx::fast::TURBOQUANT_PARALLEL_DECODE_SEQ_THRESHOLD
            );
        }
        let requires_mask = offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .any(|off| off != max_off_after);
        if requires_mask && mask_arr.is_none() {
            anyhow::bail!(
                "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: ragged decode requires an explicit mask",
            );
        }

        let target = target.into();
        if max_off_after > self.capacity() {
            let target_capacity =
                ((max_off_after + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, offsets, target)?;
        }
        self.write_per_row(k, v, offsets, per_row_lens, target)?;
        for (o, &n) in offsets.iter_mut().zip(per_row_lens.iter()) {
            *o += n;
        }

        let k_packed = self.k_packed.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: missing K")
        })?;
        let k_norms = self.k_norms.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: missing K norms"
            )
        })?;
        let v_packed = self.v_packed.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: missing V")
        })?;
        let v_norms = self.v_norms.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "TurboQuantKVCache::update_and_attend_decode_pre_rotated_on: missing V norms"
            )
        })?;
        let k_packed = slice_strided_on(
            k_packed,
            [0_i32, 0, 0, 0],
            [
                self.batch,
                self.n_kv_heads,
                max_off_after,
                self.packed_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let k_norms = slice_strided_on(
            k_norms,
            [0_i32, 0, 0],
            [self.batch, self.n_kv_heads, max_off_after],
            [1_i32, 1, 1],
            target,
        )?;
        let v_packed = slice_strided_on(
            v_packed,
            [0_i32, 0, 0, 0],
            [
                self.batch,
                self.n_kv_heads,
                max_off_after,
                self.packed_v_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v_norms = slice_strided_on(
            v_norms,
            [0_i32, 0, 0],
            [self.batch, self.n_kv_heads, max_off_after],
            [1_i32, 1, 1],
            target,
        )?;

        let output = mlx::fast::turboquant_sdpa_decode_parallel_pre_rotated_on(
            q_rot,
            &k_packed,
            &k_norms,
            &v_packed,
            &v_norms,
            &self.k_codebook,
            &self.v_signs,
            &self.v_codebook,
            scale,
            self.bits.key_bits(),
            self.bits.value_bits(),
            mask_arr,
            output_dtype,
            target,
        )?;
        Ok(output)
    }

    pub fn materialize_on(
        &self,
        output_dtype: Dtype,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        self.materialize_prefix_on(self.capacity(), output_dtype, target)
    }

    pub fn materialize_prefix_on(
        &self,
        len: i32,
        output_dtype: Dtype,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        if len == 0 {
            let target = target.into();
            let empty_k = Array::zeros_on(
                (self.batch, self.n_kv_heads, 0_i32, self.head_dim),
                output_dtype,
                target,
            )?;
            let empty_v = Array::zeros_on(
                (self.batch, self.n_kv_heads, 0_i32, self.v_head_dim),
                output_dtype,
                target,
            )?;
            return Ok((empty_k, empty_v));
        }
        if len > self.capacity() {
            anyhow::bail!(
                "TurboQuantKVCache::materialize_prefix_on: len {len} exceeds packed capacity {}",
                self.capacity()
            );
        }
        let k_packed = self
            .k_packed
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TurboQuantKVCache::materialize_on: missing K"))?;
        let k_norms = self
            .k_norms
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TurboQuantKVCache::materialize_on: missing K norms"))?;
        let v_packed = self
            .v_packed
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TurboQuantKVCache::materialize_on: missing V"))?;
        let v_norms = self
            .v_norms
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TurboQuantKVCache::materialize_on: missing V norms"))?;

        let target = target.into();
        let k_packed = slice_strided_on(
            k_packed,
            [0_i32, 0, 0, 0],
            [self.batch, self.n_kv_heads, len, self.packed_head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let k_norms = slice_strided_on(
            k_norms,
            [0_i32, 0, 0],
            [self.batch, self.n_kv_heads, len],
            [1_i32, 1, 1],
            target,
        )?;
        let v_packed = slice_strided_on(
            v_packed,
            [0_i32, 0, 0, 0],
            [self.batch, self.n_kv_heads, len, self.packed_v_head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v_norms = slice_strided_on(
            v_norms,
            [0_i32, 0, 0],
            [self.batch, self.n_kv_heads, len],
            [1_i32, 1, 1],
            target,
        )?;
        let k = mlx::fast::turbo_dequantize_on(
            &k_packed,
            &k_norms,
            &self.k_signs,
            &self.k_codebook,
            self.bits.key_bits(),
            self.head_dim,
            output_dtype,
            target,
        )?;
        let v = mlx::fast::turbo_dequantize_on(
            &v_packed,
            &v_norms,
            &self.v_signs,
            &self.v_codebook,
            self.bits.value_bits(),
            self.v_head_dim,
            output_dtype,
            target,
        )?;
        Ok((k, v))
    }

    pub fn adopt_row_from(
        &mut self,
        src: &TurboQuantKVCache,
        dst_offsets: &[i32],
        dst_row: usize,
        src_row: usize,
        src_off: i32,
    ) -> Result<()> {
        if self.bits != src.bits
            || self.n_kv_heads != src.n_kv_heads
            || self.head_dim != src.head_dim
            || self.v_head_dim != src.v_head_dim
        {
            anyhow::bail!(
                "TurboQuantKVCache::adopt_row_from: shape/bits mismatch (self={}/{}/{}/{}, src={}/{}/{}/{})",
                self.bits,
                self.n_kv_heads,
                self.head_dim,
                self.v_head_dim,
                src.bits,
                src.n_kv_heads,
                src.head_dim,
                src.v_head_dim,
            );
        }
        if dst_row >= self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::adopt_row_from: dst_row {} >= self.batch {}",
                dst_row,
                self.batch,
            );
        }
        if src_row >= src.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::adopt_row_from: src_row {} >= src.batch {}",
                src_row,
                src.batch,
            );
        }
        if dst_offsets.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::adopt_row_from: dst_offsets.len()={} != batch={}",
                dst_offsets.len(),
                self.batch,
            );
        }
        if src_off > self.cap {
            anyhow::bail!(
                "TurboQuantKVCache::adopt_row_from: src_off {src_off} > self.cap {}",
                self.cap,
            );
        }
        if src_off == 0 {
            return Ok(());
        }
        if src_off > src.capacity() {
            anyhow::bail!(
                "TurboQuantKVCache::adopt_row_from: src_off {src_off} exceeds src capacity {}",
                src.capacity(),
            );
        }

        if src_off > self.capacity() {
            let target_capacity = ((src_off + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, dst_offsets, ())?;
        }

        self.adopt_array_row(
            src.k_packed.as_ref().ok_or_else(|| {
                anyhow::anyhow!("TurboQuantKVCache::adopt_row_from: src missing K")
            })?,
            src_row,
            dst_row,
            src_off,
            PackedSide::Key,
        )?;
        self.adopt_array_row(
            src.k_norms.as_ref().ok_or_else(|| {
                anyhow::anyhow!("TurboQuantKVCache::adopt_row_from: src missing K norms")
            })?,
            src_row,
            dst_row,
            src_off,
            PackedSide::KeyNorm,
        )?;
        self.adopt_array_row(
            src.v_packed.as_ref().ok_or_else(|| {
                anyhow::anyhow!("TurboQuantKVCache::adopt_row_from: src missing V")
            })?,
            src_row,
            dst_row,
            src_off,
            PackedSide::Value,
        )?;
        self.adopt_array_row(
            src.v_norms.as_ref().ok_or_else(|| {
                anyhow::anyhow!("TurboQuantKVCache::adopt_row_from: src missing V norms")
            })?,
            src_row,
            dst_row,
            src_off,
            PackedSide::ValueNorm,
        )?;
        Ok(())
    }

    pub fn prefix_layer_for_row_on(
        &self,
        offsets: &[i32],
        row: usize,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(TurboQuantPrefixLayer, i32)> {
        if offsets.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::prefix_layer_for_row_on: offsets.len()={} != batch={}",
                offsets.len(),
                self.batch,
            );
        }
        if row >= self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::prefix_layer_for_row_on: row {} >= batch {}",
                row,
                self.batch,
            );
        }
        let cached_len = offsets[row];
        if cached_len < 0 || cached_len > self.capacity() {
            anyhow::bail!(
                "TurboQuantKVCache::prefix_layer_for_row_on: row {row} cached_len {cached_len} outside [0, {}]",
                self.capacity()
            );
        }
        let target = target.into();
        if cached_len == 0 {
            return Ok((
                TurboQuantPrefixLayer {
                    k_packed: Array::zeros_on(
                        (1_i32, self.n_kv_heads, 0_i32, self.packed_head_dim),
                        Dtype::Uint32,
                        target,
                    )?,
                    k_norms: Array::zeros_on(
                        (1_i32, self.n_kv_heads, 0_i32),
                        Dtype::Float32,
                        target,
                    )?,
                    v_packed: Array::zeros_on(
                        (1_i32, self.n_kv_heads, 0_i32, self.packed_v_head_dim),
                        Dtype::Uint32,
                        target,
                    )?,
                    v_norms: Array::zeros_on(
                        (1_i32, self.n_kv_heads, 0_i32),
                        Dtype::Float32,
                        target,
                    )?,
                },
                0,
            ));
        }

        let row = row as i32;
        let k_packed = self.k_packed.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::prefix_layer_for_row_on: missing K")
        })?;
        let k_norms = self.k_norms.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::prefix_layer_for_row_on: missing K norms")
        })?;
        let v_packed = self.v_packed.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::prefix_layer_for_row_on: missing V")
        })?;
        let v_norms = self.v_norms.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TurboQuantKVCache::prefix_layer_for_row_on: missing V norms")
        })?;

        Ok((
            TurboQuantPrefixLayer {
                k_packed: slice_strided_on(
                    k_packed,
                    [row, 0, 0, 0],
                    [row + 1, self.n_kv_heads, cached_len, self.packed_head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?,
                k_norms: slice_strided_on(
                    k_norms,
                    [row, 0, 0],
                    [row + 1, self.n_kv_heads, cached_len],
                    [1_i32, 1, 1],
                    target,
                )?,
                v_packed: slice_strided_on(
                    v_packed,
                    [row, 0, 0, 0],
                    [row + 1, self.n_kv_heads, cached_len, self.packed_v_head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?,
                v_norms: slice_strided_on(
                    v_norms,
                    [row, 0, 0],
                    [row + 1, self.n_kv_heads, cached_len],
                    [1_i32, 1, 1],
                    target,
                )?,
            },
            cached_len,
        ))
    }

    pub fn restore_packed_prefix_for_row_on(
        &mut self,
        layer: &TurboQuantPrefixLayer,
        offsets: &mut [i32],
        row: usize,
        cached_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        if offsets.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::restore_packed_prefix_for_row_on: offsets.len()={} != batch={}",
                offsets.len(),
                self.batch,
            );
        }
        if row >= self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::restore_packed_prefix_for_row_on: row {} >= batch {}",
                row,
                self.batch,
            );
        }
        if offsets[row] != 0 {
            anyhow::bail!(
                "TurboQuantKVCache::restore_packed_prefix_for_row_on: row {row} offset {} must be 0 before prefix restore",
                offsets[row],
            );
        }
        if cached_len < 0 || cached_len > self.cap {
            anyhow::bail!(
                "TurboQuantKVCache::restore_packed_prefix_for_row_on: cached_len {cached_len} outside [0, {}]",
                self.cap
            );
        }
        self.validate_prefix_layer(layer, cached_len)?;
        if cached_len == 0 {
            offsets[row] = 0;
            return Ok(());
        }

        let target = target.into();
        if cached_len > self.capacity() {
            let target_capacity =
                ((cached_len + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, offsets, target)?;
        }
        self.update_key_value_arrays(
            PackedWrite {
                k_packed: &layer.k_packed,
                k_norms: &layer.k_norms,
                v_packed: &layer.v_packed,
                v_norms: &layer.v_norms,
            },
            PackedWriteRange {
                row: row as i32,
                off: 0,
                end: cached_len,
            },
            target,
        )?;
        offsets[row] = cached_len;
        Ok(())
    }

    pub fn restore_dense_prefix_for_row_on(
        &mut self,
        k: &Array,
        v: &Array,
        offsets: &mut [i32],
        row: usize,
        cached_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        validate_dense_shape("k", k, self.head_dim)?;
        validate_dense_shape("v", v, self.v_head_dim)?;
        if offsets.len() != self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::restore_dense_prefix_for_row_on: offsets.len()={} != batch={}",
                offsets.len(),
                self.batch,
            );
        }
        if row >= self.batch as usize {
            anyhow::bail!(
                "TurboQuantKVCache::restore_dense_prefix_for_row_on: row {} >= batch {}",
                row,
                self.batch,
            );
        }
        if offsets[row] != 0 {
            anyhow::bail!(
                "TurboQuantKVCache::restore_dense_prefix_for_row_on: row {row} offset {} must be 0 before prefix restore",
                offsets[row],
            );
        }
        if cached_len < 0 || cached_len > self.cap {
            anyhow::bail!(
                "TurboQuantKVCache::restore_dense_prefix_for_row_on: cached_len {cached_len} outside [0, {}]",
                self.cap
            );
        }
        let k_shape = k.shape();
        let k_shape = k_shape.as_slice();
        if k_shape != [1_i32, self.n_kv_heads, cached_len, self.head_dim] {
            anyhow::bail!(
                "TurboQuantKVCache::restore_dense_prefix_for_row_on: K shape {:?} incompatible with [1,{},{cached_len},{}]",
                k_shape,
                self.n_kv_heads,
                self.head_dim
            );
        }
        let v_shape = v.shape();
        let v_shape = v_shape.as_slice();
        if v_shape != [1_i32, self.n_kv_heads, cached_len, self.v_head_dim] {
            anyhow::bail!(
                "TurboQuantKVCache::restore_dense_prefix_for_row_on: V shape {:?} incompatible with [1,{},{cached_len},{}]",
                v_shape,
                self.n_kv_heads,
                self.v_head_dim
            );
        }
        if cached_len == 0 {
            offsets[row] = 0;
            return Ok(());
        }

        let target = target.into();
        if cached_len > self.capacity() {
            let target_capacity =
                ((cached_len + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, offsets, target)?;
        }
        let (k_packed, k_norms) = mlx::fast::turbo_quantize_on(
            k,
            &self.k_signs,
            &self.k_codebook,
            self.bits.key_bits(),
            target,
        )?;
        let (v_packed, v_norms) = mlx::fast::turbo_quantize_on(
            v,
            &self.v_signs,
            &self.v_codebook,
            self.bits.value_bits(),
            target,
        )?;
        self.update_key_value_arrays(
            PackedWrite {
                k_packed: &k_packed,
                k_norms: &k_norms,
                v_packed: &v_packed,
                v_norms: &v_norms,
            },
            PackedWriteRange {
                row: row as i32,
                off: 0,
                end: cached_len,
            },
            target,
        )?;
        offsets[row] = cached_len;
        Ok(())
    }

    fn validate_prefix_layer(&self, layer: &TurboQuantPrefixLayer, cached_len: i32) -> Result<()> {
        validate_packed_rank4(
            "K",
            &layer.k_packed,
            self.n_kv_heads,
            cached_len,
            self.packed_head_dim,
        )?;
        validate_norm_rank3("K norms", &layer.k_norms, self.n_kv_heads, cached_len)?;
        validate_packed_rank4(
            "V",
            &layer.v_packed,
            self.n_kv_heads,
            cached_len,
            self.packed_v_head_dim,
        )?;
        validate_norm_rank3("V norms", &layer.v_norms, self.n_kv_heads, cached_len)?;
        Ok(())
    }

    fn write_per_row(
        &mut self,
        k: &Array,
        v: &Array,
        offsets: &[i32],
        per_row_lens: &[i32],
        target: StreamOrDevice,
    ) -> Result<()> {
        let k_seq = k.shape().as_slice()[2];
        if self.batch == 1 && per_row_lens.len() == 1 && per_row_lens[0] == k_seq {
            let n = per_row_lens[0];
            if n == 0 {
                return Ok(());
            }
            let off = offsets[0];
            let end = off + n;
            let (k_packed, k_norms) = mlx::fast::turbo_quantize_on(
                k,
                &self.k_signs,
                &self.k_codebook,
                self.bits.key_bits(),
                target,
            )?;
            let (v_packed, v_norms) = mlx::fast::turbo_quantize_on(
                v,
                &self.v_signs,
                &self.v_codebook,
                self.bits.value_bits(),
                target,
            )?;
            self.update_key_value_arrays(
                PackedWrite {
                    k_packed: &k_packed,
                    k_norms: &k_norms,
                    v_packed: &v_packed,
                    v_norms: &v_norms,
                },
                PackedWriteRange { row: 0, off, end },
                target,
            )?;
            return Ok(());
        }

        for (i_usize, &n) in per_row_lens.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let i = i_usize as i32;
            let off_i = offsets[i_usize];
            let end_i = off_i + n;
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
            let (k_packed, k_norms) = mlx::fast::turbo_quantize_on(
                &k_row,
                &self.k_signs,
                &self.k_codebook,
                self.bits.key_bits(),
                target,
            )?;
            let (v_packed, v_norms) = mlx::fast::turbo_quantize_on(
                &v_row,
                &self.v_signs,
                &self.v_codebook,
                self.bits.value_bits(),
                target,
            )?;
            self.update_key_value_arrays(
                PackedWrite {
                    k_packed: &k_packed,
                    k_norms: &k_norms,
                    v_packed: &v_packed,
                    v_norms: &v_norms,
                },
                PackedWriteRange {
                    row: i,
                    off: off_i,
                    end: end_i,
                },
                target,
            )?;
        }
        Ok(())
    }

    fn update_key_value_arrays(
        &mut self,
        packed: PackedWrite<'_>,
        range: PackedWriteRange,
        target: StreamOrDevice,
    ) -> Result<()> {
        let next_k = slice_update_on(
            self.k_packed.as_ref().expect("k_packed allocated"),
            packed.k_packed,
            [range.row, 0, range.off, 0],
            [
                range.row + 1,
                self.n_kv_heads,
                range.end,
                self.packed_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let next_k_norms = slice_update_on(
            self.k_norms.as_ref().expect("k_norms allocated"),
            packed.k_norms,
            [range.row, 0, range.off],
            [range.row + 1, self.n_kv_heads, range.end],
            [1_i32, 1, 1],
            target,
        )?;
        let next_v = slice_update_on(
            self.v_packed.as_ref().expect("v_packed allocated"),
            packed.v_packed,
            [range.row, 0, range.off, 0],
            [
                range.row + 1,
                self.n_kv_heads,
                range.end,
                self.packed_v_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let next_v_norms = slice_update_on(
            self.v_norms.as_ref().expect("v_norms allocated"),
            packed.v_norms,
            [range.row, 0, range.off],
            [range.row + 1, self.n_kv_heads, range.end],
            [1_i32, 1, 1],
            target,
        )?;
        self.k_packed = Some(next_k);
        self.k_norms = Some(next_k_norms);
        self.v_packed = Some(next_v);
        self.v_norms = Some(next_v_norms);
        Ok(())
    }

    fn grow_to(
        &mut self,
        new_capacity: i32,
        offsets: &[i32],
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let target = target.into();
        let max_off = offsets.iter().copied().max().unwrap_or(0);
        self.k_packed = Some(grow_rank4(
            self.k_packed.as_ref(),
            Rank4GrowShape {
                batch: self.batch,
                heads: self.n_kv_heads,
                width: self.packed_head_dim,
            },
            Dtype::Uint32,
            max_off,
            new_capacity,
            target,
        )?);
        self.k_norms = Some(grow_rank3(
            self.k_norms.as_ref(),
            self.batch,
            self.n_kv_heads,
            Dtype::Float32,
            max_off,
            new_capacity,
            target,
        )?);
        self.v_packed = Some(grow_rank4(
            self.v_packed.as_ref(),
            Rank4GrowShape {
                batch: self.batch,
                heads: self.n_kv_heads,
                width: self.packed_v_head_dim,
            },
            Dtype::Uint32,
            max_off,
            new_capacity,
            target,
        )?);
        self.v_norms = Some(grow_rank3(
            self.v_norms.as_ref(),
            self.batch,
            self.n_kv_heads,
            Dtype::Float32,
            max_off,
            new_capacity,
            target,
        )?);
        Ok(())
    }

    fn adopt_array_row(
        &mut self,
        src: &Array,
        src_row: usize,
        dst_row: usize,
        src_off: i32,
        side: PackedSide,
    ) -> Result<()> {
        match side {
            PackedSide::Key | PackedSide::Value => {
                let width = match side {
                    PackedSide::Key => self.packed_head_dim,
                    PackedSide::Value => self.packed_v_head_dim,
                    _ => unreachable!(),
                };
                let row = slice_strided_on(
                    src,
                    [src_row as i32, 0, 0, 0],
                    [src_row as i32 + 1, self.n_kv_heads, src_off, width],
                    [1_i32, 1, 1, 1],
                    (),
                )?;
                let dst = match side {
                    PackedSide::Key => self.k_packed.as_ref().expect("dst k allocated"),
                    PackedSide::Value => self.v_packed.as_ref().expect("dst v allocated"),
                    _ => unreachable!(),
                };
                let next = slice_update_on(
                    dst,
                    &row,
                    [dst_row as i32, 0, 0, 0],
                    [dst_row as i32 + 1, self.n_kv_heads, src_off, width],
                    [1_i32, 1, 1, 1],
                    (),
                )?;
                match side {
                    PackedSide::Key => self.k_packed = Some(next),
                    PackedSide::Value => self.v_packed = Some(next),
                    _ => unreachable!(),
                }
            }
            PackedSide::KeyNorm | PackedSide::ValueNorm => {
                let row = slice_strided_on(
                    src,
                    [src_row as i32, 0, 0],
                    [src_row as i32 + 1, self.n_kv_heads, src_off],
                    [1_i32, 1, 1],
                    (),
                )?;
                let dst = match side {
                    PackedSide::KeyNorm => self.k_norms.as_ref().expect("dst k norms allocated"),
                    PackedSide::ValueNorm => self.v_norms.as_ref().expect("dst v norms allocated"),
                    _ => unreachable!(),
                };
                let next = slice_update_on(
                    dst,
                    &row,
                    [dst_row as i32, 0, 0],
                    [dst_row as i32 + 1, self.n_kv_heads, src_off],
                    [1_i32, 1, 1],
                    (),
                )?;
                match side {
                    PackedSide::KeyNorm => self.k_norms = Some(next),
                    PackedSide::ValueNorm => self.v_norms = Some(next),
                    _ => unreachable!(),
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum PackedSide {
    Key,
    KeyNorm,
    Value,
    ValueNorm,
}

struct PackedWrite<'a> {
    k_packed: &'a Array,
    k_norms: &'a Array,
    v_packed: &'a Array,
    v_norms: &'a Array,
}

#[derive(Clone, Copy)]
struct PackedWriteRange {
    row: i32,
    off: i32,
    end: i32,
}

#[derive(Clone, Copy)]
struct Rank4GrowShape {
    batch: i32,
    heads: i32,
    width: i32,
}

fn validate_bit_width(name: &str, bits: u8) -> Result<()> {
    if matches!(bits, 3 | 4) {
        Ok(())
    } else {
        anyhow::bail!("TurboQuantKVBits: {name} must be 3 or 4, got {bits}")
    }
}

fn validate_dim(name: &str, dim: i32) -> Result<()> {
    validate_positive(name, dim)?;
    if !(dim as usize).is_power_of_two() {
        anyhow::bail!("TurboQuantKVCache: {name} {dim} must be a power of two");
    }
    Ok(())
}

fn validate_positive(name: &str, dim: i32) -> Result<()> {
    if dim <= 0 {
        anyhow::bail!("TurboQuantKVCache: {name} must be positive, got {dim}");
    }
    Ok(())
}

fn validate_dense_shape(name: &str, arr: &Array, expected_last_dim: i32) -> Result<()> {
    let shape = arr.shape();
    let dims = shape.as_slice();
    if dims.len() != 4 {
        anyhow::bail!(
            "TurboQuantKVCache::update_from_dense_on: {name} must be rank-4 [B, H, S, D], got rank {}",
            dims.len()
        );
    }
    if dims[3] != expected_last_dim {
        anyhow::bail!(
            "TurboQuantKVCache::update_from_dense_on: {name}.shape()[3] = {} but expected {}",
            dims[3],
            expected_last_dim
        );
    }
    Ok(())
}

fn validate_dense_batch_heads(name: &str, arr: &Array, batch: i32, n_kv_heads: i32) -> Result<()> {
    let dims = arr.shape();
    let dims = dims.as_slice();
    if dims[0] != batch || dims[1] != n_kv_heads {
        anyhow::bail!(
            "TurboQuantKVCache: {name} shape batch/heads [{}, {}] != expected [{batch}, {n_kv_heads}]",
            dims[0],
            dims[1]
        );
    }
    Ok(())
}

fn validate_packed_rank4(
    name: &str,
    arr: &Array,
    n_kv_heads: i32,
    cached_len: i32,
    packed_dim: i32,
) -> Result<()> {
    if arr.dtype() != Dtype::Uint32 {
        anyhow::bail!(
            "TurboQuantKVCache::restore_packed_prefix_for_row_on: {name} dtype {} != Uint32",
            arr.dtype()
        );
    }
    let shape = arr.shape();
    let dims = shape.as_slice();
    if dims != [1_i32, n_kv_heads, cached_len, packed_dim] {
        anyhow::bail!(
            "TurboQuantKVCache::restore_packed_prefix_for_row_on: {name} shape {:?} incompatible with [1,{n_kv_heads},{cached_len},{packed_dim}]",
            dims
        );
    }
    Ok(())
}

fn validate_norm_rank3(name: &str, arr: &Array, n_kv_heads: i32, cached_len: i32) -> Result<()> {
    if arr.dtype() != Dtype::Float32 {
        anyhow::bail!(
            "TurboQuantKVCache::restore_packed_prefix_for_row_on: {name} dtype {} != Float32",
            arr.dtype()
        );
    }
    let shape = arr.shape();
    let dims = shape.as_slice();
    if dims != [1_i32, n_kv_heads, cached_len] {
        anyhow::bail!(
            "TurboQuantKVCache::restore_packed_prefix_for_row_on: {name} shape {:?} incompatible with [1,{n_kv_heads},{cached_len}]",
            dims
        );
    }
    Ok(())
}

fn grow_rank4(
    current: Option<&Array>,
    shape: Rank4GrowShape,
    dtype: Dtype,
    max_off: i32,
    new_capacity: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    match (current, max_off) {
        (None, _) | (Some(_), 0) => Ok(Array::zeros_on(
            (shape.batch, shape.heads, new_capacity, shape.width),
            dtype,
            target,
        )?),
        (Some(old), _) => {
            let old_kept = slice_strided_on(
                old,
                [0_i32, 0, 0, 0],
                [shape.batch, shape.heads, max_off, shape.width],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let tail = Array::zeros_on(
                (
                    shape.batch,
                    shape.heads,
                    new_capacity - max_off,
                    shape.width,
                ),
                dtype,
                target,
            )?;
            let grown = concatenate_on(&[&old_kept, &tail], 2, target)?;
            Ok(grown)
        }
    }
}

fn grow_rank3(
    current: Option<&Array>,
    batch: i32,
    heads: i32,
    dtype: Dtype,
    max_off: i32,
    new_capacity: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    match (current, max_off) {
        (None, _) | (Some(_), 0) => Ok(Array::zeros_on(
            (batch, heads, new_capacity),
            dtype,
            target,
        )?),
        (Some(old), _) => {
            let old_kept = slice_strided_on(
                old,
                [0_i32, 0, 0],
                [batch, heads, max_off],
                [1_i32, 1, 1],
                target,
            )?;
            let tail = Array::zeros_on((batch, heads, new_capacity - max_off), dtype, target)?;
            let grown = concatenate_on(&[&old_kept, &tail], 2, target)?;
            Ok(grown)
        }
    }
}

fn signs_array(dim: i32) -> Result<Array> {
    let signs = turboquant::wht::generate_signs(dim as usize, TURBOQUANT_SIGN_SEED);
    let signs: Array = (signs.as_slice(), (dim,)).try_into()?;
    Ok(signs)
}

fn codebook_array(bits: u8, dim: i32) -> Result<Array> {
    let codebook = turboquant::codebook::Codebook::new(bits, dim as usize);
    let centroids = codebook.centroids;
    let codebook: Array = (centroids.as_slice(), (centroids.len() as i32,)).try_into()?;
    Ok(codebook)
}

fn packed_dim(dim: i32, bits: u8) -> i32 {
    let values_per_word = match bits {
        3 => 10,
        4 => 8,
        _ => unreachable!("validated bits"),
    };
    (dim + values_per_word - 1) / values_per_word
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    fn test_values(len: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(len);
        let mut state = 0x1234_5678_9abc_def0_u64;
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push(((state & 0xffff) as f32 / 65_535.0 - 0.5) * 1.7);
        }
        values
    }

    #[test]
    fn packed_dim_matches_word_layout() {
        let tq3 = TurboQuantKVCache::new(1, 1, 128, 64, 16, 16, TurboQuantKVBits::K3V3).unwrap();
        assert_eq!(tq3.packed_head_dim(), 13);
        assert_eq!(tq3.packed_v_head_dim(), 7);

        let tq4 = TurboQuantKVCache::new(1, 1, 128, 64, 16, 16, TurboQuantKVBits::K4V4).unwrap();
        assert_eq!(tq4.packed_head_dim(), 16);
        assert_eq!(tq4.packed_v_head_dim(), 8);
    }

    #[test]
    fn mixed_bits_use_distinct_kv_layouts() {
        let tq = TurboQuantKVCache::new(1, 1, 128, 64, 16, 16, TurboQuantKVBits::K3V4).unwrap();

        assert_eq!(tq.bits(), TurboQuantKVBits::K3V4);
        assert_eq!(tq.key_bits(), 3);
        assert_eq!(tq.value_bits(), 4);
        assert_eq!(tq.packed_head_dim(), 13);
        assert_eq!(tq.packed_v_head_dim(), 8);
        assert_eq!(tq.k_codebook.shape().as_slice(), &[8]);
        assert_eq!(tq.v_codebook.shape().as_slice(), &[16]);
    }

    #[test]
    fn constructed_constants_have_expected_shapes_and_dtypes() {
        let tq = TurboQuantKVCache::new(1, 1, 8, 16, 16, 16, TurboQuantKVBits::K4V4).unwrap();
        assert_eq!(tq.k_signs.shape().as_slice(), &[8]);
        assert_eq!(tq.v_signs.shape().as_slice(), &[16]);
        assert_eq!(tq.k_codebook.shape().as_slice(), &[16]);
        assert_eq!(tq.v_codebook.shape().as_slice(), &[16]);
        assert_eq!(tq.k_signs.dtype(), Dtype::Float32);
        assert_eq!(tq.k_codebook.dtype(), Dtype::Float32);
    }

    #[test]
    fn k3v4_per_row_append_preserves_identical_d512_batch_rows() {
        let batch = 4_i32;
        let heads = 2_i32;
        let seq = 121_i32;
        let head_dim = 512_i32;
        let row = test_values((heads * seq * head_dim) as usize);
        let data = row.repeat(batch as usize);
        let k: Array = (data.as_slice(), &[batch, heads, seq, head_dim][..])
            .try_into()
            .unwrap();
        let v: Array = (data.as_slice(), &[batch, heads, seq, head_dim][..])
            .try_into()
            .unwrap();
        let mut cache = TurboQuantKVCache::new(
            batch,
            heads,
            head_dim,
            head_dim,
            256,
            256,
            TurboQuantKVBits::K3V4,
        )
        .unwrap();
        let mut offsets = vec![0_i32; batch as usize];
        let row_lens = vec![seq; batch as usize];

        let _ = cache
            .update_and_fetch_on(&k, &v, &mut offsets, &row_lens, Dtype::Float32, ())
            .unwrap();
        let decode_row = test_values((heads * head_dim) as usize)
            .into_iter()
            .map(|value| value * 0.7 + 0.2)
            .collect::<Vec<_>>();
        let decode_data = decode_row.repeat(batch as usize);
        let decode_k: Array = (decode_data.as_slice(), &[batch, heads, 1_i32, head_dim][..])
            .try_into()
            .unwrap();
        let decode_v = decode_k.clone();
        let decode_lens = vec![1_i32; batch as usize];
        let (dense_k, dense_v) = cache
            .update_and_fetch_on(
                &decode_k,
                &decode_v,
                &mut offsets,
                &decode_lens,
                Dtype::Float32,
                (),
            )
            .unwrap();

        let k_packed = cache.k_packed.as_ref().unwrap().to_vec::<u32>().unwrap();
        let k_norms = cache.k_norms.as_ref().unwrap().to_vec::<f32>().unwrap();
        let v_packed = cache.v_packed.as_ref().unwrap().to_vec::<u32>().unwrap();
        let v_norms = cache.v_norms.as_ref().unwrap().to_vec::<f32>().unwrap();
        for values in [&k_packed, &v_packed] {
            let row_size = values.len() / batch as usize;
            for row in 1..batch as usize {
                assert_eq!(
                    &values[row * row_size..(row + 1) * row_size],
                    &values[..row_size],
                    "packed cache row {row} differs"
                );
            }
        }
        for values in [&k_norms, &v_norms] {
            let row_size = values.len() / batch as usize;
            for row in 1..batch as usize {
                assert_eq!(
                    &values[row * row_size..(row + 1) * row_size],
                    &values[..row_size],
                    "norm cache row {row} differs"
                );
            }
        }
        for values in [
            dense_k.to_vec::<f32>().unwrap(),
            dense_v.to_vec::<f32>().unwrap(),
        ] {
            let row_size = values.len() / batch as usize;
            for row in 1..batch as usize {
                assert_eq!(
                    &values[row * row_size..(row + 1) * row_size],
                    &values[..row_size],
                    "materialized cache row {row} differs"
                );
            }
        }
    }
}
