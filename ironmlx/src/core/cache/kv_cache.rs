//! Per-layer KV cache for full-attention layers. See P2 spec § 3 for design.
//!
//! Implementation strategy: lazy alloc + step-rounded grow via concatenate;
//! per-update writes use Strategy A: a B-loop of `slice_update_on` calls
//! (one per row with per_row_lens[i] > 0). The public API (`new`, `with_step`,
//! `update_and_fetch`, `offsets`, `cap`, `reset`) is stable across strategies.

use mlx::ops::indexing::{slice_strided_on, slice_update_on};
use mlx::ops::shape::concatenate_on;
use mlx::{Array, Dtype, StreamOrDevice};

use super::{
    PagedKVCache, PagedKvHotColdConfig, PagedKvHotColdSummary, PagedPrefixLayer, TurboQuantKVBits,
    TurboQuantKVCache, TurboQuantPrefixLayer,
};
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
    turboquant: Option<Box<TurboQuantKVCache>>,
    paged: Option<Box<PagedKVCache>>,
}

/// Lightweight checkpoint for [`KVCache`] rollback.
///
/// This intentionally stores only logical offsets. The dense K/V buffers may
/// retain stale data past those offsets; callers already mask positions above
/// each row's logical length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KVCacheSnapshot {
    offsets: Vec<i32>,
}

impl KVCacheSnapshot {
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }
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
            turboquant: None,
            paged: None,
        }
    }

    /// Override grow step (default 256). `step >= cap` triggers one-shot
    /// preallocation. Panics if `step <= 0`.
    pub fn with_step(mut self, step: i32) -> Self {
        assert!(step > 0, "KVCache step must be positive (got {step})");
        self.step = step;
        if let Some(tq) = &mut self.turboquant {
            tq.set_step(step);
        }
        self
    }

    /// Enable TurboQuant packed storage for this cache.
    ///
    /// Once enabled, long-lived full-attention K/V history is stored in the
    /// packed TurboQuant buffers. Dense K/V buffers are released after any
    /// existing prefix has been copied into the packed representation.
    pub fn with_turboquant(mut self, bits: TurboQuantKVBits) -> Result<Self> {
        self.enable_turboquant(bits)?;
        Ok(self)
    }

    pub fn enable_turboquant(&mut self, bits: TurboQuantKVBits) -> Result<()> {
        if self.paged.is_some() {
            anyhow::bail!("KVCache::enable_turboquant: paged KV cache is already enabled");
        }
        let mut turboquant = TurboQuantKVCache::new(
            self.batch,
            self.n_kv_heads,
            self.head_dim,
            self.v_head_dim,
            self.cap,
            self.step,
            bits,
        )?;
        let max_off = self.offsets.iter().copied().max().unwrap_or(0);
        if max_off > 0 {
            let keys_full = self.keys.as_ref().ok_or_else(|| {
                anyhow::anyhow!("KVCache::enable_turboquant: keys are unallocated")
            })?;
            let values_full = self.values.as_ref().ok_or_else(|| {
                anyhow::anyhow!("KVCache::enable_turboquant: values are unallocated")
            })?;
            let k_slice = slice_strided_on(
                keys_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, max_off, self.head_dim],
                [1_i32, 1, 1, 1],
                (),
            )?;
            let v_slice = slice_strided_on(
                values_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, max_off, self.v_head_dim],
                [1_i32, 1, 1, 1],
                (),
            )?;
            turboquant.update_from_dense_on(&k_slice, &v_slice, ())?;
        }
        self.turboquant = Some(Box::new(turboquant));
        self.keys = None;
        self.values = None;
        Ok(())
    }

    /// Enable paged K/V storage for full-attention layers.
    ///
    /// Paged mode is mutually exclusive with TurboQuant. Existing dense
    /// prefix data, if any, is copied into pages and dense buffers are then
    /// released; future prefill reads materialize dense K/V on demand while
    /// decode uses the paged attention kernel directly.
    pub fn with_paged(mut self, block_size: i32, max_pages: i32) -> Result<Self> {
        self.enable_paged(block_size, max_pages)?;
        Ok(self)
    }

    pub fn enable_paged(&mut self, block_size: i32, max_pages: i32) -> Result<()> {
        if self.turboquant.is_some() {
            anyhow::bail!("KVCache::enable_paged: TurboQuant KV cache is already enabled");
        }
        let mut paged = PagedKVCache::new(
            self.batch,
            self.n_kv_heads,
            self.head_dim,
            self.v_head_dim,
            self.dtype,
            self.cap,
            block_size,
            max_pages,
        )?;
        let max_off = self.offsets.iter().copied().max().unwrap_or(0);
        if max_off > 0 {
            let keys_full = self
                .keys
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("KVCache::enable_paged: keys are unallocated"))?;
            let values_full = self
                .values
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("KVCache::enable_paged: values are unallocated"))?;
            let k_slice = slice_strided_on(
                keys_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, max_off, self.head_dim],
                [1_i32, 1, 1, 1],
                (),
            )?;
            let v_slice = slice_strided_on(
                values_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, max_off, self.v_head_dim],
                [1_i32, 1, 1, 1],
                (),
            )?;
            let mut paged_offsets = vec![0_i32; self.batch as usize];
            paged.update_and_fetch_on(&k_slice, &v_slice, &mut paged_offsets, &self.offsets, ())?;
        }
        self.paged = Some(Box::new(paged));
        self.keys = None;
        self.values = None;
        Ok(())
    }

    pub fn enable_paged_hot_cold_tiering(&mut self, config: PagedKvHotColdConfig) -> Result<()> {
        let paged = self.paged.as_mut().ok_or_else(|| {
            anyhow::anyhow!("KVCache::enable_paged_hot_cold_tiering: paged KV is not enabled")
        })?;
        paged.enable_hot_cold_tiering(config)
    }

    pub fn turboquant(&self) -> Option<&TurboQuantKVCache> {
        self.turboquant.as_deref()
    }

    pub fn prefix_cache_profile(&self) -> Option<String> {
        self.turboquant.as_ref().map(|tq| tq.bits().cache_profile())
    }

    pub fn paged(&self) -> Option<&PagedKVCache> {
        self.paged.as_deref()
    }

    pub fn paged_hot_cold_summary(&self) -> Option<PagedKvHotColdSummary> {
        self.paged
            .as_ref()
            .and_then(|paged| paged.hot_cold_summary())
    }

    pub fn shrink_paged_hot_window(&mut self, hot_window_pages: i32) -> Result<usize> {
        let Some(paged) = self.paged.as_mut() else {
            return Ok(0);
        };
        paged.shrink_hot_window_on(&self.offsets, hot_window_pages, ())
    }

    pub fn restore_configured_paged_hot_window(&mut self) -> bool {
        self.paged
            .as_mut()
            .is_some_and(|paged| paged.restore_configured_hot_window())
    }

    pub fn batch(&self) -> i32 {
        self.batch
    }

    pub fn n_kv_heads(&self) -> i32 {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> i32 {
        self.head_dim
    }

    pub fn v_head_dim(&self) -> i32 {
        self.v_head_dim
    }

    pub fn paged_prefix_layer_for_row_on(
        &self,
        row: usize,
        target: impl Into<StreamOrDevice>,
    ) -> Result<PagedPrefixLayer> {
        let paged = self.paged.as_ref().ok_or_else(|| {
            anyhow::anyhow!("KVCache::paged_prefix_layer_for_row_on: paged KV is not enabled")
        })?;
        let (k_pages, v_pages) = paged.prefix_pages_for_row_on(&self.offsets, row, target)?;
        Ok(PagedPrefixLayer { k_pages, v_pages })
    }

    pub fn materialize_current_paged_prefix_on(
        &self,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        let paged = self.paged.as_ref().ok_or_else(|| {
            anyhow::anyhow!("KVCache::materialize_current_paged_prefix_on: paged KV is not enabled")
        })?;
        let max_off = self.offsets.iter().copied().max().unwrap_or(0);
        paged.materialize_prefix_on(&self.offsets, max_off, target)
    }

    pub fn restore_paged_prefix_layer_for_row_on(
        &mut self,
        layer: &PagedPrefixLayer,
        row: usize,
        prefix_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let paged = self.paged.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "KVCache::restore_paged_prefix_layer_for_row_on: paged KV is not enabled"
            )
        })?;
        paged.restore_prefix_pages_for_row_on(
            &layer.k_pages,
            &layer.v_pages,
            &mut self.offsets,
            row,
            prefix_len,
            target,
        )
    }

    pub fn restore_paged_prefix_layer_for_rows_on(
        &mut self,
        layer: &PagedPrefixLayer,
        rows: &[usize],
        prefix_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let paged = self.paged.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "KVCache::restore_paged_prefix_layer_for_rows_on: paged KV is not enabled"
            )
        })?;
        paged.restore_prefix_pages_for_rows_on(
            &layer.k_pages,
            &layer.v_pages,
            &mut self.offsets,
            rows,
            prefix_len,
            target,
        )
    }

    pub fn dense_prefix_layer_for_row_on(
        &self,
        row: usize,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array, i32)> {
        let target = target.into();
        if row >= self.batch as usize {
            anyhow::bail!(
                "KVCache::dense_prefix_layer_for_row_on: row {} >= batch {}",
                row,
                self.batch
            );
        }
        if self.paged.is_some() {
            anyhow::bail!(
                "KVCache::dense_prefix_layer_for_row_on: dense prefix export requires dense KV storage"
            );
        }
        let cached_len = self.offsets[row];
        if cached_len == 0 {
            let k = Array::zeros_on(
                (1_i32, self.n_kv_heads, 0_i32, self.head_dim),
                self.dtype,
                target,
            )?;
            let v = Array::zeros_on(
                (1_i32, self.n_kv_heads, 0_i32, self.v_head_dim),
                self.dtype,
                target,
            )?;
            return Ok((k, v, 0));
        }
        if let Some(tq) = &self.turboquant {
            let (keys_full, values_full) =
                tq.materialize_prefix_on(cached_len, self.dtype, target)?;
            let k = slice_strided_on(
                &keys_full,
                [row as i32, 0, 0, 0],
                [row as i32 + 1, self.n_kv_heads, cached_len, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let v = slice_strided_on(
                &values_full,
                [row as i32, 0, 0, 0],
                [row as i32 + 1, self.n_kv_heads, cached_len, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            return Ok((k, v, cached_len));
        }
        let keys_full = self.keys.as_ref().ok_or_else(|| {
            anyhow::anyhow!("KVCache::dense_prefix_layer_for_row_on: keys are unallocated")
        })?;
        let values_full = self.values.as_ref().ok_or_else(|| {
            anyhow::anyhow!("KVCache::dense_prefix_layer_for_row_on: values are unallocated")
        })?;
        let k = slice_strided_on(
            keys_full,
            [row as i32, 0, 0, 0],
            [row as i32 + 1, self.n_kv_heads, cached_len, self.head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v = slice_strided_on(
            values_full,
            [row as i32, 0, 0, 0],
            [row as i32 + 1, self.n_kv_heads, cached_len, self.v_head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        Ok((k, v, cached_len))
    }

    pub fn turboquant_prefix_layer_for_row_on(
        &self,
        row: usize,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(TurboQuantPrefixLayer, i32)> {
        let tq = self.turboquant.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "KVCache::turboquant_prefix_layer_for_row_on: TurboQuant is not enabled"
            )
        })?;
        if self.paged.is_some() {
            anyhow::bail!(
                "KVCache::turboquant_prefix_layer_for_row_on: paged KV storage is not compatible with TurboQuant packed prefixes"
            );
        }
        tq.prefix_layer_for_row_on(&self.offsets, row, target)
    }

    pub fn restore_turboquant_prefix_layer_for_row_on(
        &mut self,
        layer: &TurboQuantPrefixLayer,
        row: usize,
        cached_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let tq = self.turboquant.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "KVCache::restore_turboquant_prefix_layer_for_row_on: TurboQuant is not enabled"
            )
        })?;
        if self.paged.is_some() {
            anyhow::bail!(
                "KVCache::restore_turboquant_prefix_layer_for_row_on: paged KV storage is not compatible with TurboQuant packed prefixes"
            );
        }
        tq.restore_packed_prefix_for_row_on(layer, &mut self.offsets, row, cached_len, target)
    }

    pub fn restore_dense_prefix_layer_for_row_on(
        &mut self,
        k: &Array,
        v: &Array,
        row: usize,
        cached_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let target = target.into();
        if row >= self.batch as usize {
            anyhow::bail!(
                "KVCache::restore_dense_prefix_layer_for_row_on: row {} >= batch {}",
                row,
                self.batch
            );
        }
        if self.paged.is_some() {
            anyhow::bail!(
                "KVCache::restore_dense_prefix_layer_for_row_on: dense prefix restore requires dense KV storage"
            );
        }
        if cached_len < 0 || cached_len > self.cap {
            anyhow::bail!(
                "KVCache::restore_dense_prefix_layer_for_row_on: cached_len {cached_len} outside [0, {}]",
                self.cap
            );
        }
        let k_shape = k.shape();
        let k_shape = k_shape.as_slice();
        if k_shape != [1_i32, self.n_kv_heads, cached_len, self.head_dim] {
            anyhow::bail!(
                "KVCache::restore_dense_prefix_layer_for_row_on: K shape {:?} incompatible with [1,{},{cached_len},{}]",
                k_shape,
                self.n_kv_heads,
                self.head_dim
            );
        }
        let v_shape = v.shape();
        let v_shape = v_shape.as_slice();
        if v_shape != [1_i32, self.n_kv_heads, cached_len, self.v_head_dim] {
            anyhow::bail!(
                "KVCache::restore_dense_prefix_layer_for_row_on: V shape {:?} incompatible with [1,{},{cached_len},{}]",
                v_shape,
                self.n_kv_heads,
                self.v_head_dim
            );
        }
        if k.dtype() != self.dtype || v.dtype() != self.dtype {
            anyhow::bail!(
                "KVCache::restore_dense_prefix_layer_for_row_on: dtype mismatch K={} V={} expected {}",
                k.dtype(),
                v.dtype(),
                self.dtype
            );
        }
        if let Some(tq) = &mut self.turboquant {
            tq.restore_dense_prefix_for_row_on(k, v, &mut self.offsets, row, cached_len, target)?;
            return Ok(());
        }
        if cached_len > 0 {
            let current_capacity = self
                .keys
                .as_ref()
                .map(|a| a.shape().as_slice()[2])
                .unwrap_or(0);
            if cached_len > current_capacity {
                let target_capacity =
                    ((cached_len + self.step - 1) / self.step * self.step).min(self.cap);
                self.grow_to(target_capacity, target)?;
            }
            let keys_full = self.keys.as_ref().expect("grow_to allocated keys");
            let values_full = self.values.as_ref().expect("grow_to allocated values");
            self.keys = Some(slice_update_on(
                keys_full,
                k,
                [row as i32, 0, 0, 0],
                [row as i32 + 1, self.n_kv_heads, cached_len, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?);
            self.values = Some(slice_update_on(
                values_full,
                v,
                [row as i32, 0, 0, 0],
                [row as i32 + 1, self.n_kv_heads, cached_len, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?);
        }
        self.offsets[row] = cached_len;
        Ok(())
    }

    pub(crate) fn turboquant_pre_rotated_decode_query_signs(
        &self,
        queries: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        mask_arr: Option<&Array>,
    ) -> Option<&Array> {
        let tq = self.turboquant.as_ref()?;
        if self.supports_turboquant_pre_rotated_decode_attention(
            queries,
            k,
            v,
            per_row_lens,
            mask_arr,
        ) {
            Some(tq.key_signs())
        } else {
            None
        }
    }

    /// Per-row write offsets (length == batch). Row `i`'s next K/V write
    /// lands at sequence position `offsets[i]`.
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }

    pub fn cap(&self) -> i32 {
        self.cap
    }

    /// Raise `self.cap` to `new_cap` if larger; no-op otherwise. The
    /// physical K/V buffers are not reallocated here — they remain at
    /// their current capacity until the next `update_and_fetch` or
    /// `adopt_row_from` triggers `grow_to` against the new cap.
    ///
    /// B1-p2.3f: enables `Scheduler::admit_mid` to extend the main
    /// cache when a new row's `prompt_len + max_new_tokens` exceeds
    /// the initial batch's cap. Shrinking is intentionally not
    /// supported — existing rows may rely on the current cap.
    pub fn grow_cap(&mut self, new_cap: i32) {
        if new_cap > self.cap {
            self.cap = new_cap;
            if let Some(tq) = &mut self.turboquant {
                tq.grow_cap(new_cap);
            }
            if let Some(paged) = &mut self.paged {
                paged.grow_cap(new_cap);
            }
        }
    }

    /// Dtype used for the K/V buffer. Exposed so `adopt_row_from` can
    /// validate that `src` and `self` agree before slicing.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Reset every row's offset to 0; retains allocated buffers for reuse.
    pub fn reset(&mut self) {
        for o in &mut self.offsets {
            *o = 0;
        }
        if let Some(tq) = &mut self.turboquant {
            tq.clear();
        }
        if let Some(paged) = &mut self.paged {
            paged.clear();
        }
    }

    /// Capture the current logical cache position. No K/V buffers are copied.
    pub fn snapshot(&self) -> KVCacheSnapshot {
        KVCacheSnapshot {
            offsets: self.offsets.clone(),
        }
    }

    /// Restore logical offsets from a prior checkpoint.
    ///
    /// This is the cheap rollback path used by speculative decoding: stale
    /// K/V data beyond restored offsets is left in-place and ignored by masks.
    pub fn restore(&mut self, snapshot: &KVCacheSnapshot) -> Result<()> {
        self.restore_offsets(snapshot.offsets())
    }

    /// Set logical offsets directly. Intended for rollback/truncation to an
    /// accepted prefix. Does not clear or copy K/V buffers.
    pub fn restore_offsets(&mut self, offsets: &[i32]) -> Result<()> {
        if offsets.len() != self.batch as usize {
            anyhow::bail!(
                "KVCache::restore_offsets: offsets.len()={} != batch={}",
                offsets.len(),
                self.batch,
            );
        }
        for (i, &off) in offsets.iter().enumerate() {
            if off < 0 {
                anyhow::bail!("KVCache::restore_offsets: offsets[{i}] = {off} must be >= 0");
            }
            if off > self.cap {
                anyhow::bail!(
                    "KVCache::restore_offsets: offsets[{i}] = {off} > cap {}",
                    self.cap,
                );
            }
        }
        if let Some(paged) = &mut self.paged {
            paged.restore_offsets(&mut self.offsets, offsets)?;
            return Ok(());
        }
        let max_off = offsets.iter().copied().max().unwrap_or(0);
        if max_off > 0 {
            let (key_cap, value_cap) = if let Some(tq) = &self.turboquant {
                let cap = tq.capacity();
                (cap, cap)
            } else {
                let key_cap = self
                    .keys
                    .as_ref()
                    .map(|a| a.shape().as_slice()[2])
                    .unwrap_or(0);
                let value_cap = self
                    .values
                    .as_ref()
                    .map(|a| a.shape().as_slice()[2])
                    .unwrap_or(0);
                (key_cap, value_cap)
            };
            if max_off > key_cap || max_off > value_cap {
                anyhow::bail!(
                    "KVCache::restore_offsets: max offset {max_off} exceeds allocated key/value capacity {key_cap}/{value_cap}",
                );
            }
        }
        self.offsets.clone_from_slice(offsets);
        Ok(())
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

    pub fn update_and_fetch_for_attention(
        &mut self,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
    ) -> Result<(Array, Array)> {
        self.update_and_fetch_for_attention_on(k, v, per_row_lens, ())
    }

    pub fn update_and_fetch_for_attention_on(
        &mut self,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        let target: StreamOrDevice = target.into();
        self.update_and_fetch_on(k, v, per_row_lens, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_update_and_attend_on(
        &mut self,
        queries: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        scale: f32,
        mask_arr: Option<&Array>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Option<Array>> {
        let target = target.into();
        if let Some(out) = self.try_update_and_attend_decode_on(
            queries,
            k,
            v,
            per_row_lens,
            scale,
            mask_arr,
            target,
        )? {
            return Ok(Some(out));
        }
        if let Some(out) = self.try_update_and_attend_multirow_on(
            queries,
            k,
            v,
            per_row_lens,
            scale,
            mask_arr,
            target,
        )? {
            return Ok(Some(out));
        }
        if self.paged.is_some()
            && self.supports_paged_prefill_attention(queries, k, v, per_row_lens, mask_arr)
        {
            let paged = match self.paged.as_mut() {
                Some(paged) => paged,
                None => unreachable!("paged cache presence checked before prefill dispatch"),
            };
            return Ok(Some(paged.update_and_attend_prefill_on(
                queries,
                k,
                v,
                &mut self.offsets,
                per_row_lens,
                scale,
                mask_arr,
                target,
            )?));
        }
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_update_and_attend_decode(
        &mut self,
        queries: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        scale: f32,
        mask_arr: Option<&Array>,
    ) -> Result<Option<Array>> {
        self.try_update_and_attend_decode_on(queries, k, v, per_row_lens, scale, mask_arr, ())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_update_and_attend_decode_on(
        &mut self,
        queries: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        scale: f32,
        mask_arr: Option<&Array>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Option<Array>> {
        if self.paged.is_some()
            && self.supports_paged_decode_attention(queries, k, v, per_row_lens, mask_arr)
        {
            let paged = match self.paged.as_mut() {
                Some(paged) => paged,
                None => unreachable!("paged cache presence checked before decode dispatch"),
            };
            let target = target.into();
            return Ok(Some(paged.update_and_attend_decode_on(
                queries,
                k,
                v,
                &mut self.offsets,
                per_row_lens,
                scale,
                target,
            )?));
        }

        if self.turboquant.is_none()
            || !self.supports_turboquant_decode_attention(queries, k, v, per_row_lens, mask_arr)
        {
            return Ok(None);
        }

        let target = target.into();
        let output_dtype = queries.dtype();
        let tq = self
            .turboquant
            .as_mut()
            .expect("checked turboquant is some");
        Ok(Some(tq.update_and_attend_decode_on(
            queries,
            k,
            v,
            &mut self.offsets,
            per_row_lens,
            scale,
            mask_arr,
            output_dtype,
            target,
        )?))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_update_and_attend_multirow_on(
        &mut self,
        queries: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        scale: f32,
        mask_arr: Option<&Array>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Option<Array>> {
        if self.turboquant.is_none()
            || !self.supports_turboquant_multirow_attention(queries, k, v, per_row_lens, mask_arr)
        {
            return Ok(None);
        }

        let target = target.into();
        let output_dtype = queries.dtype();
        let tq = self
            .turboquant
            .as_mut()
            .expect("checked turboquant is some");
        Ok(Some(tq.update_and_attend_multirow_on(
            queries,
            k,
            v,
            &mut self.offsets,
            per_row_lens,
            scale,
            mask_arr,
            output_dtype,
            target,
        )?))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_update_and_attend_decode_pre_rotated_on(
        &mut self,
        q_rot: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        scale: f32,
        mask_arr: Option<&Array>,
        output_dtype: Dtype,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Option<Array>> {
        if self.turboquant.is_none()
            || !self.supports_turboquant_pre_rotated_decode_attention_rotated(
                q_rot,
                k,
                v,
                per_row_lens,
                mask_arr,
                output_dtype,
            )
        {
            return Ok(None);
        }

        let target = target.into();
        let tq = self
            .turboquant
            .as_mut()
            .expect("checked turboquant is some");
        Ok(Some(tq.update_and_attend_decode_pre_rotated_on(
            q_rot,
            k,
            v,
            &mut self.offsets,
            per_row_lens,
            scale,
            mask_arr,
            output_dtype,
            target,
        )?))
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
        if let Some(paged) = &mut self.paged {
            return paged.update_and_fetch_on(k, v, &mut self.offsets, per_row_lens, target);
        }
        if let Some(tq) = &mut self.turboquant {
            return tq.update_and_fetch_on(
                k,
                v,
                &mut self.offsets,
                per_row_lens,
                self.dtype,
                target,
            );
        }

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

    fn supports_paged_decode_attention(
        &self,
        queries: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        _mask_arr: Option<&Array>,
    ) -> bool {
        if self.v_head_dim != self.head_dim
            || per_row_lens.len() != self.batch as usize
            || !matches!(
                queries.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
        {
            return false;
        }
        let q_shape = queries.shape();
        let q_dims = q_shape.as_slice();
        let k_shape = k.shape();
        let k_dims = k_shape.as_slice();
        let v_shape = v.shape();
        let v_dims = v_shape.as_slice();
        if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
            return false;
        }
        if q_dims[0] != self.batch
            || q_dims[2] != 1
            || q_dims[3] != self.head_dim
            || q_dims[1] % self.n_kv_heads != 0
        {
            return false;
        }
        if k_dims != [self.batch, self.n_kv_heads, 1, self.head_dim]
            || v_dims != [self.batch, self.n_kv_heads, 1, self.v_head_dim]
        {
            return false;
        }
        if per_row_lens.iter().any(|&n| n != 1) {
            return false;
        }
        let max_off_after = self
            .offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);
        if max_off_after > self.cap {
            return false;
        }
        true
    }

    fn supports_paged_prefill_attention(
        &self,
        queries: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        mask_arr: Option<&Array>,
    ) -> bool {
        if self.v_head_dim != self.head_dim
            || per_row_lens.len() != self.batch as usize
            || !matches!(
                queries.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
        {
            return false;
        }
        let Some(paged) = self.paged.as_ref() else {
            return false;
        };
        let q_shape = queries.shape();
        let q_dims = q_shape.as_slice();
        let k_shape = k.shape();
        let k_dims = k_shape.as_slice();
        let v_shape = v.shape();
        let v_dims = v_shape.as_slice();
        if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
            return false;
        }
        if q_dims[0] != self.batch
            || q_dims[2] <= 1
            || q_dims[2] != k_dims[2]
            || q_dims[3] != self.head_dim
            || q_dims[1] % self.n_kv_heads != 0
        {
            return false;
        }
        if k_dims != [self.batch, self.n_kv_heads, q_dims[2], self.head_dim]
            || v_dims != [self.batch, self.n_kv_heads, q_dims[2], self.v_head_dim]
        {
            return false;
        }
        if per_row_lens.iter().any(|&n| n <= 0 || n > q_dims[2]) {
            return false;
        }
        let max_off_after = self
            .offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);
        if max_off_after > self.cap
            || !paged.hot_cold_needs_streaming_after(&self.offsets, per_row_lens)
        {
            return false;
        }
        match mask_arr {
            Some(mask) => {
                let mask_shape = mask.shape();
                let mask_dims = mask_shape.as_slice();
                if mask_dims.len() != 4
                    || !(mask_dims[0] == 1 || mask_dims[0] == self.batch)
                    || !(mask_dims[1] == 1 || mask_dims[1] == q_dims[1])
                    || mask_dims[2] != q_dims[2]
                    || mask_dims[3] != max_off_after
                {
                    return false;
                }
            }
            None => {
                if per_row_lens.iter().any(|&n| n != q_dims[2]) {
                    return false;
                }
            }
        }
        true
    }

    fn supports_turboquant_decode_attention(
        &self,
        queries: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        mask_arr: Option<&Array>,
    ) -> bool {
        if self.v_head_dim != self.head_dim
            || per_row_lens.len() != self.batch as usize
            || !matches!(
                queries.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
        {
            return false;
        }
        let q_shape = queries.shape();
        let q_dims = q_shape.as_slice();
        let k_shape = k.shape();
        let k_dims = k_shape.as_slice();
        let v_shape = v.shape();
        let v_dims = v_shape.as_slice();
        if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
            return false;
        }
        if q_dims[0] != self.batch
            || q_dims[2] != 1
            || q_dims[3] != self.head_dim
            || q_dims[1] % self.n_kv_heads != 0
        {
            return false;
        }
        if k_dims != [self.batch, self.n_kv_heads, 1, self.head_dim]
            || v_dims != [self.batch, self.n_kv_heads, 1, self.v_head_dim]
        {
            return false;
        }
        if per_row_lens.iter().any(|&n| n != 1) {
            return false;
        }
        let max_off_after = self
            .offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);
        if max_off_after > self.cap {
            return false;
        }
        let ragged = self
            .offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .any(|off| off != max_off_after);
        if ragged && mask_arr.is_none() {
            return false;
        }
        if let Some(mask) = mask_arr {
            let mask_shape = mask.shape();
            let mask_dims = mask_shape.as_slice();
            if mask_dims.len() != 4
                || mask_dims[0] != self.batch
                || mask_dims[2] != 1
                || mask_dims[3] != max_off_after
                || !(mask_dims[1] == 1 || mask_dims[1] == q_dims[1])
            {
                return false;
            }
        }
        true
    }

    fn supports_turboquant_multirow_attention(
        &self,
        queries: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        mask_arr: Option<&Array>,
    ) -> bool {
        if self.v_head_dim != self.head_dim
            || self.head_dim < 32
            || self.head_dim % 32 != 0
            || per_row_lens.len() != self.batch as usize
            || !matches!(
                queries.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
        {
            return false;
        }
        let q_shape = queries.shape();
        let q_dims = q_shape.as_slice();
        let k_shape = k.shape();
        let k_dims = k_shape.as_slice();
        let v_shape = v.shape();
        let v_dims = v_shape.as_slice();
        if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
            return false;
        }
        let q_rows = q_dims[2];
        if q_dims[0] != self.batch
            || !(2..=mlx::fast::TURBOQUANT_MULTIROW_MAX_QUERY_ROWS).contains(&q_rows)
            || q_dims[3] != self.head_dim
            || q_dims[1] % self.n_kv_heads != 0
            || q_dims[1] / self.n_kv_heads > 32
        {
            return false;
        }
        if k_dims != [self.batch, self.n_kv_heads, q_rows, self.head_dim]
            || v_dims != [self.batch, self.n_kv_heads, q_rows, self.v_head_dim]
            || per_row_lens.iter().any(|&n| n <= 0 || n > q_rows)
        {
            return false;
        }
        let max_off_after = self
            .offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);
        if max_off_after > self.cap || max_off_after <= mlx::fast::TURBOQUANT_MULTIROW_MIN_SEQ_LEN {
            return false;
        }
        if let Some(mask) = mask_arr {
            let mask_shape = mask.shape();
            let mask_dims = mask_shape.as_slice();
            if mask_dims.len() != 4
                || mask_dims[0] != self.batch
                || !(mask_dims[2] == 1 || mask_dims[2] == q_rows)
                || mask_dims[3] != max_off_after
                || !(mask_dims[1] == 1 || mask_dims[1] == q_dims[1])
            {
                return false;
            }
        }
        true
    }

    fn supports_turboquant_pre_rotated_decode_attention(
        &self,
        queries: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        mask_arr: Option<&Array>,
    ) -> bool {
        if self.v_head_dim != self.head_dim
            || per_row_lens.len() != self.batch as usize
            || !matches!(
                queries.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
        {
            return false;
        }
        let q_shape = queries.shape();
        let q_dims = q_shape.as_slice();
        let k_shape = k.shape();
        let k_dims = k_shape.as_slice();
        let v_shape = v.shape();
        let v_dims = v_shape.as_slice();
        if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
            return false;
        }
        if q_dims[0] != self.batch
            || q_dims[2] != 1
            || q_dims[3] != self.head_dim
            || q_dims[1] % self.n_kv_heads != 0
        {
            return false;
        }
        self.supports_turboquant_pre_rotated_decode_common(
            q_dims[1],
            k_dims,
            v_dims,
            per_row_lens,
            mask_arr,
        )
    }

    fn supports_turboquant_pre_rotated_decode_attention_rotated(
        &self,
        q_rot: &Array,
        k: &Array,
        v: &Array,
        per_row_lens: &[i32],
        mask_arr: Option<&Array>,
        output_dtype: Dtype,
    ) -> bool {
        if self.v_head_dim != self.head_dim
            || per_row_lens.len() != self.batch as usize
            || q_rot.dtype() != Dtype::Float32
            || !matches!(
                output_dtype,
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
        {
            return false;
        }
        let q_shape = q_rot.shape();
        let q_dims = q_shape.as_slice();
        let k_shape = k.shape();
        let k_dims = k_shape.as_slice();
        let v_shape = v.shape();
        let v_dims = v_shape.as_slice();
        if q_dims.len() != 3 || k_dims.len() != 4 || v_dims.len() != 4 {
            return false;
        }
        if q_dims[0] != self.batch || q_dims[2] != self.head_dim || q_dims[1] % self.n_kv_heads != 0
        {
            return false;
        }
        self.supports_turboquant_pre_rotated_decode_common(
            q_dims[1],
            k_dims,
            v_dims,
            per_row_lens,
            mask_arr,
        )
    }

    fn supports_turboquant_pre_rotated_decode_common(
        &self,
        q_heads: i32,
        k_dims: &[i32],
        v_dims: &[i32],
        per_row_lens: &[i32],
        mask_arr: Option<&Array>,
    ) -> bool {
        if k_dims != [self.batch, self.n_kv_heads, 1, self.head_dim]
            || v_dims != [self.batch, self.n_kv_heads, 1, self.v_head_dim]
        {
            return false;
        }
        if per_row_lens.iter().any(|&n| n != 1) {
            return false;
        }
        let max_off_after = self
            .offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);
        if max_off_after > self.cap
            || max_off_after < mlx::fast::TURBOQUANT_PARALLEL_DECODE_SEQ_THRESHOLD
        {
            return false;
        }
        let ragged = self
            .offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .any(|off| off != max_off_after);
        if ragged && mask_arr.is_none() {
            return false;
        }
        if let Some(mask) = mask_arr {
            let mask_shape = mask.shape();
            let mask_dims = mask_shape.as_slice();
            if mask_dims.len() != 4
                || mask_dims[0] != self.batch
                || mask_dims[2] != 1
                || mask_dims[3] != max_off_after
                || !(mask_dims[1] == 1 || mask_dims[1] == q_heads)
            {
                return false;
            }
        }
        true
    }

    /// Copy a single row's cache state from `src` into `self` at
    /// `dst_row`. The destination slot's K/V at positions
    /// `[0..src.offsets[src_row]]` is overwritten; positions beyond
    /// (stale or unallocated) are not touched. `self.offsets[dst_row]`
    /// is set to `src.offsets[src_row]`.
    ///
    /// Requires matching n_kv_heads / head_dim / v_head_dim / dtype.
    /// src and self may have different batch sizes (typical usage:
    /// src.batch = 1, self.batch = b_max).
    ///
    /// Errors on shape/dtype mismatch, dst_row >= self.batch,
    /// src_row >= src.batch, or src.offsets[src_row] > self.cap.
    pub fn adopt_row_from(&mut self, src: &KVCache, dst_row: usize, src_row: usize) -> Result<()> {
        if self.n_kv_heads != src.n_kv_heads
            || self.head_dim != src.head_dim
            || self.v_head_dim != src.v_head_dim
            || self.dtype != src.dtype
        {
            anyhow::bail!(
                "KVCache::adopt_row_from: shape/dtype mismatch (self={}/{}/{}/{:?}, src={}/{}/{}/{:?})",
                self.n_kv_heads, self.head_dim, self.v_head_dim, self.dtype,
                src.n_kv_heads, src.head_dim, src.v_head_dim, src.dtype,
            );
        }
        if dst_row >= self.batch as usize {
            anyhow::bail!(
                "KVCache::adopt_row_from: dst_row {} >= self.batch {}",
                dst_row,
                self.batch,
            );
        }
        if src_row >= src.batch as usize {
            anyhow::bail!(
                "KVCache::adopt_row_from: src_row {} >= src.batch {}",
                src_row,
                src.batch,
            );
        }
        let src_off = src.offsets[src_row];
        if src_off > self.cap {
            anyhow::bail!(
                "KVCache::adopt_row_from: src.offsets[{}] = {} > self.cap {}",
                src_row,
                src_off,
                self.cap,
            );
        }

        match (&mut self.paged, &src.paged) {
            (Some(dst_paged), Some(src_paged)) => {
                dst_paged.adopt_row_from_on(
                    src_paged,
                    &mut self.offsets,
                    &src.offsets,
                    dst_row,
                    src_row,
                    (),
                )?;
                return Ok(());
            }
            (Some(_), None) | (None, Some(_)) => {
                anyhow::bail!("KVCache::adopt_row_from: paged cache kind mismatch");
            }
            (None, None) => {}
        }

        let dst_offsets = self.offsets.clone();
        match (&mut self.turboquant, &src.turboquant) {
            (Some(dst_tq), Some(src_tq)) => {
                dst_tq.adopt_row_from(src_tq, &dst_offsets, dst_row, src_row, src_off)?;
                self.offsets[dst_row] = src_off;
                return Ok(());
            }
            (Some(_), None) | (None, Some(_)) => {
                anyhow::bail!("KVCache::adopt_row_from: TurboQuant cache kind mismatch");
            }
            (None, None) => {}
        }

        if src_off > 0 {
            // Ensure self.keys / values are allocated up to src_off.
            let current_capacity = self
                .keys
                .as_ref()
                .map(|a| a.shape().as_slice()[2])
                .unwrap_or(0);
            if src_off > current_capacity {
                let target_capacity =
                    ((src_off + self.step - 1) / self.step * self.step).min(self.cap);
                self.grow_to(target_capacity, ().into())?;
            }

            let src_keys = src.keys.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "KVCache::adopt_row_from: src has offset {} but keys are unallocated",
                    src_off
                )
            })?;
            let src_values = src.values.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "KVCache::adopt_row_from: src has offset {} but values are unallocated",
                    src_off
                )
            })?;

            // Slice src[src_row, :, 0..src_off, :].
            let k_slice = slice_strided_on(
                src_keys,
                [src_row as i32, 0, 0, 0],
                [src_row as i32 + 1, self.n_kv_heads, src_off, self.head_dim],
                [1_i32, 1, 1, 1],
                (),
            )?;
            let v_slice = slice_strided_on(
                src_values,
                [src_row as i32, 0, 0, 0],
                [
                    src_row as i32 + 1,
                    self.n_kv_heads,
                    src_off,
                    self.v_head_dim,
                ],
                [1_i32, 1, 1, 1],
                (),
            )?;

            // Write into self[dst_row, :, 0..src_off, :].
            let keys_full = self.keys.as_ref().expect("grow_to allocated keys");
            let values_full = self.values.as_ref().expect("grow_to allocated values");
            let new_keys = slice_update_on(
                keys_full,
                &k_slice,
                [dst_row as i32, 0, 0, 0],
                [dst_row as i32 + 1, self.n_kv_heads, src_off, self.head_dim],
                [1_i32, 1, 1, 1],
                (),
            )?;
            let new_values = slice_update_on(
                values_full,
                &v_slice,
                [dst_row as i32, 0, 0, 0],
                [
                    dst_row as i32 + 1,
                    self.n_kv_heads,
                    src_off,
                    self.v_head_dim,
                ],
                [1_i32, 1, 1, 1],
                (),
            )?;
            self.keys = Some(new_keys);
            self.values = Some(new_values);
        }

        self.offsets[dst_row] = src_off;
        Ok(())
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
        let k_seq = k.shape().as_slice()[2];
        if self.batch == 1 && per_row_lens.len() == 1 && per_row_lens[0] == k_seq {
            let n = per_row_lens[0];
            if n == 0 {
                return Ok(());
            }
            let off = self.offsets[0];
            let end = off + n;
            let keys_full = self.keys.as_ref().expect("keys allocated by grow_to");
            let values_full = self.values.as_ref().expect("values allocated by grow_to");
            let new_keys = slice_update_on(
                keys_full,
                k,
                [0_i32, 0, off, 0],
                [1_i32, self.n_kv_heads, end, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let new_values = slice_update_on(
                values_full,
                v,
                [0_i32, 0, off, 0],
                [1_i32, self.n_kv_heads, end, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            self.keys = Some(new_keys);
            self.values = Some(new_values);
            return Ok(());
        }

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

    fn wht_inplace(x: &mut [f32]) {
        let n = x.len();
        let mut h = 1;
        while h < n {
            let mut i = 0;
            while i < n {
                for j in i..i + h {
                    let a = x[j];
                    let b = x[j + h];
                    x[j] = a + b;
                    x[j + h] = a - b;
                }
                i += h * 2;
            }
            h *= 2;
        }

        let scale = 1.0 / (n as f32).sqrt();
        for value in x {
            *value *= scale;
        }
    }

    fn reference_query_turbo_rotate(input: &[f32], head_dim: usize, signs: &[f32]) -> Vec<f32> {
        let vector_count = input.len() / head_dim;
        let mut out = vec![0.0_f32; input.len()];
        for vec_idx in 0..vector_count {
            let start = vec_idx * head_dim;
            let mut values: Vec<f32> = input[start..start + head_dim]
                .iter()
                .zip(signs.iter())
                .map(|(&x, &sign)| x * sign)
                .collect();
            wht_inplace(&mut values);
            out[start..start + head_dim].copy_from_slice(&values);
        }
        out
    }

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
    fn kvcache_snapshot_restore_offsets_only() {
        let mut c = make_cache_b(2, 1024);
        let (k1, v1) = make_kv_b(2, 8);
        c.update_and_fetch(&k1, &v1, &[4, 8]).unwrap();
        let snapshot = c.snapshot();
        assert_eq!(snapshot.offsets(), &[4, 8]);

        let (k2, v2) = make_kv_b(2, 4);
        c.update_and_fetch(&k2, &v2, &[4, 4]).unwrap();
        assert_eq!(c.offsets(), &[8, 12]);

        c.restore(&snapshot).expect("restore snapshot");
        assert_eq!(c.offsets(), &[4, 8]);

        c.restore_offsets(&[5, 9]).expect("restore accepted prefix");
        assert_eq!(c.offsets(), &[5, 9]);
    }

    #[test]
    fn kvcache_restore_offsets_validates_shape_and_capacity() {
        let mut c = make_cache_b(2, 16);
        let (k, v) = make_kv_b(2, 4);
        c.update_and_fetch(&k, &v, &[4, 4]).unwrap();

        let bad_len = c.restore_offsets(&[1, 1, 1]);
        assert!(bad_len.is_err());

        let beyond_cap = c.restore_offsets(&[17, 0]);
        assert!(beyond_cap.is_err());

        let mut c_large = make_cache_b(2, 1024);
        let (k2, v2) = make_kv_b(2, 4);
        c_large.update_and_fetch(&k2, &v2, &[4, 4]).unwrap();
        let beyond_allocated = c_large.restore_offsets(&[300, 0]);
        assert!(beyond_allocated.is_err());
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
    fn kvcache_grow_cap_extends_and_allows_writes_beyond_initial_cap() {
        // Initial cap=8 — after writing 4 tokens, a second 6-token write
        // would exceed cap. grow_cap(64) lifts the limit so the second
        // write succeeds + post-grow capacity reflects the new cap.
        let mut c = make_cache_b(2, 8);
        let (k1, v1) = make_kv_b(2, 4);
        c.update_and_fetch(&k1, &v1, &[4, 4]).expect("write 1");
        assert_eq!(c.cap(), 8);

        c.grow_cap(64);
        assert_eq!(c.cap(), 64);

        let (k2, v2) = make_kv_b(2, 6);
        // 4+6=10 > old cap=8; passes against new cap=64.
        let (kf, _vf) = c.update_and_fetch(&k2, &v2, &[6, 6]).expect("write 2");
        assert_eq!(c.offsets(), &[10, 10]);
        assert_eq!(kf.shape().as_slice(), &[2, 4, 10, 256]);
    }

    #[test]
    fn kvcache_grow_cap_is_monotonic_noop_on_shrink() {
        let mut c = make_cache_b(2, 100);
        c.grow_cap(50); // smaller than current — no-op
        assert_eq!(c.cap(), 100);
        c.grow_cap(100); // equal — no-op
        assert_eq!(c.cap(), 100);
        c.grow_cap(200); // larger — grows
        assert_eq!(c.cap(), 200);
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

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_turboquant_disabled_by_default() {
        let c = KVCache::new(1, 2, 8, 8, Dtype::Float32, 16);
        assert!(c.turboquant().is_none());
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_turboquant_write_records_packed_shapes() {
        let mut c = KVCache::new(1, 2, 8, 8, Dtype::Float32, 16)
            .with_step(16)
            .with_turboquant(TurboQuantKVBits::K3V3)
            .expect("enable turboquant");
        let k_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| ((i as f32) * 0.031).sin())
            .collect();
        let v_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| ((i as f32) * 0.047).cos())
            .collect();
        let k: Array = (k_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();

        let (kf, vf) = c.update_and_fetch(&k, &v, &[4]).expect("update");

        assert_eq!(c.offsets(), &[4]);
        assert_eq!(kf.shape().as_slice(), &[1, 2, 4, 8]);
        assert_eq!(vf.shape().as_slice(), &[1, 2, 4, 8]);
        let tq = c.turboquant().expect("turboquant cache");
        assert_eq!(tq.bits(), TurboQuantKVBits::K3V3);
        assert_eq!(tq.head_dim(), 8);
        assert_eq!(tq.v_head_dim(), 8);
        assert_eq!(
            tq.k_packed().expect("k packed").shape().as_slice(),
            &[1, 2, 16, 1]
        );
        assert_eq!(
            tq.v_packed().expect("v packed").shape().as_slice(),
            &[1, 2, 16, 1]
        );
        assert_eq!(
            tq.k_norms().expect("k norms").shape().as_slice(),
            &[1, 2, 16]
        );
        assert_eq!(
            tq.v_norms().expect("v norms").shape().as_slice(),
            &[1, 2, 16]
        );
        assert_eq!(tq.k_packed().expect("k packed").dtype(), Dtype::Uint32);
        assert_eq!(tq.k_norms().expect("k norms").dtype(), Dtype::Float32);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_turboquant_update_does_not_retain_dense_buffers() {
        let mut c = KVCache::new(1, 2, 8, 8, Dtype::Float32, 16)
            .with_step(16)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable turboquant");
        let k_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| ((i as f32) * 0.071).sin())
            .collect();
        let v_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| ((i as f32) * 0.083).cos())
            .collect();
        let k: Array = (k_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();

        let (kf, vf) = c
            .update_and_fetch_for_attention(&k, &v, &[4])
            .expect("turboquant attention update");

        assert_eq!(c.offsets(), &[4]);
        assert_eq!(kf.shape().as_slice(), &[1, 2, 4, 8]);
        assert_eq!(vf.shape().as_slice(), &[1, 2, 4, 8]);
        assert!(c.keys.is_none(), "TurboQuant cache must not retain dense K");
        assert!(
            c.values.is_none(),
            "TurboQuant cache must not retain dense V"
        );
        let tq = c.turboquant().expect("turboquant cache");
        assert_eq!(
            tq.k_packed().expect("k packed").shape().as_slice(),
            &[1, 2, 16, 1]
        );
        assert_eq!(
            tq.v_packed().expect("v packed").shape().as_slice(),
            &[1, 2, 16, 1]
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_turboquant_exports_and_restores_packed_prefix() {
        let mut source = KVCache::new(1, 2, 8, 8, Dtype::Float32, 16)
            .with_step(16)
            .with_turboquant(TurboQuantKVBits::K3V4)
            .expect("enable turboquant");
        let k_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| ((i as f32) * 0.037).sin())
            .collect();
        let v_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| ((i as f32) * 0.041).cos())
            .collect();
        let k: Array = (k_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();

        source
            .update_and_fetch(&k, &v, &[4])
            .expect("turboquant source update");
        let (prefix, cached_len) = source
            .turboquant_prefix_layer_for_row_on(0, ())
            .expect("export packed prefix from TurboQuant");

        assert_eq!(cached_len, 4);
        assert_eq!(prefix.k_packed.shape().as_slice(), &[1, 2, 4, 1]);
        assert_eq!(prefix.k_norms.shape().as_slice(), &[1, 2, 4]);
        assert_eq!(prefix.v_packed.shape().as_slice(), &[1, 2, 4, 1]);
        assert_eq!(prefix.v_norms.shape().as_slice(), &[1, 2, 4]);
        assert_eq!(prefix.k_packed.dtype(), Dtype::Uint32);
        assert_eq!(prefix.k_norms.dtype(), Dtype::Float32);

        let mut restored = KVCache::new(1, 2, 8, 8, Dtype::Float32, 16)
            .with_step(16)
            .with_turboquant(TurboQuantKVBits::K3V4)
            .expect("enable restore turboquant");
        restored
            .restore_turboquant_prefix_layer_for_row_on(&prefix, 0, cached_len, ())
            .expect("restore packed prefix into TurboQuant");

        assert_eq!(restored.offsets(), &[4]);
        assert!(restored.keys.is_none());
        assert!(restored.values.is_none());
        let tq = restored.turboquant().expect("restored turboquant");
        assert_eq!(tq.bits(), TurboQuantKVBits::K3V4);
        assert_eq!(
            tq.k_packed().expect("restored K packed").shape().as_slice(),
            &[1, 2, 16, 1]
        );
        let (roundtrip, roundtrip_len) = restored
            .turboquant_prefix_layer_for_row_on(0, ())
            .expect("re-export restored prefix");
        assert_eq!(roundtrip_len, 4);
        assert_eq!(roundtrip.k_packed.shape().as_slice(), &[1, 2, 4, 1]);
        assert_eq!(roundtrip.v_packed.shape().as_slice(), &[1, 2, 4, 1]);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_turboquant_restore_offsets_uses_packed_capacity() {
        let mut c = KVCache::new(1, 2, 8, 8, Dtype::Float32, 16)
            .with_step(16)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable turboquant");
        let k_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| ((i as f32) * 0.091).sin())
            .collect();
        let v_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| ((i as f32) * 0.103).cos())
            .collect();
        let k: Array = (k_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();

        c.update_and_fetch(&k, &v, &[4]).expect("turboquant update");
        c.restore_offsets(&[2]).expect("restore packed prefix");

        assert_eq!(c.offsets(), &[2]);
        assert!(
            c.keys.is_none(),
            "TurboQuant restore must not allocate dense K"
        );
        assert!(
            c.values.is_none(),
            "TurboQuant restore must not allocate dense V"
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_turboquant_pre_rotated_decode_matches_regular_decode() {
        let mut regular = KVCache::new(1, 1, 8, 8, Dtype::Float32, 256)
            .with_step(128)
            .with_turboquant(TurboQuantKVBits::K3V4)
            .expect("enable regular turboquant");
        let mut pre_rotated = KVCache::new(1, 1, 8, 8, Dtype::Float32, 256)
            .with_step(128)
            .with_turboquant(TurboQuantKVBits::K3V4)
            .expect("enable pre-rotated turboquant");

        let prefix_k_data: Vec<f32> = (0..(127 * 8)).map(|i| ((i as f32) * 0.017).sin()).collect();
        let prefix_v_data: Vec<f32> = (0..(127 * 8)).map(|i| ((i as f32) * 0.019).cos()).collect();
        let prefix_k: Array = (prefix_k_data.as_slice(), (1_i32, 1_i32, 127_i32, 8_i32))
            .try_into()
            .unwrap();
        let prefix_v: Array = (prefix_v_data.as_slice(), (1_i32, 1_i32, 127_i32, 8_i32))
            .try_into()
            .unwrap();
        regular
            .update_and_fetch(&prefix_k, &prefix_v, &[127])
            .expect("regular prefix");
        pre_rotated
            .update_and_fetch(&prefix_k, &prefix_v, &[127])
            .expect("pre-rotated prefix");

        let q_data: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.029).sin()).collect();
        let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 1_i32, 8_i32))
            .try_into()
            .unwrap();
        let k_step_data: Vec<f32> = (0..8).map(|i| ((i as f32) * 0.031).cos()).collect();
        let v_step_data: Vec<f32> = (0..8).map(|i| ((i as f32) * 0.037).sin()).collect();
        let k_step: Array = (k_step_data.as_slice(), (1_i32, 1_i32, 1_i32, 8_i32))
            .try_into()
            .unwrap();
        let v_step: Array = (v_step_data.as_slice(), (1_i32, 1_i32, 1_i32, 8_i32))
            .try_into()
            .unwrap();

        let signs = pre_rotated
            .turboquant_pre_rotated_decode_query_signs(&q, &k_step, &v_step, &[1], None)
            .expect("threshold reached at 128");
        let signs_data = signs.to_vec::<f32>().expect("signs to_vec");
        let q_rot_data = reference_query_turbo_rotate(&q_data, 8, &signs_data);
        let q_rot: Array = (q_rot_data.as_slice(), (1_i32, 2_i32, 8_i32))
            .try_into()
            .unwrap();

        let expected = regular
            .try_update_and_attend_decode_on(&q, &k_step, &v_step, &[1], 0.25, None, ())
            .expect("regular decode")
            .expect("regular turboquant decode");
        let actual = pre_rotated
            .try_update_and_attend_decode_pre_rotated_on(
                &q_rot,
                &k_step,
                &v_step,
                &[1],
                0.25,
                None,
                Dtype::Float32,
                (),
            )
            .expect("pre-rotated decode")
            .expect("pre-rotated turboquant decode");

        assert_eq!(regular.offsets(), &[128]);
        assert_eq!(pre_rotated.offsets(), &[128]);
        assert_eq!(actual.shape().as_slice(), expected.shape().as_slice());
        let expected = expected.to_vec::<f32>().expect("expected to_vec");
        let actual = actual.to_vec::<f32>().expect("actual to_vec");
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-5,
                "attn[{idx}] mismatch: actual={a} expected={e}"
            );
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_turboquant_adopt_row_from_preserves_existing_packed_rows_when_growing() {
        let mut src_short = KVCache::new(1, 2, 8, 8, Dtype::Float32, 16)
            .with_step(4)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable short src turboquant");
        let short_k_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| 0.5 + ((i as f32) * 0.071).sin())
            .collect();
        let short_v_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| 0.25 + ((i as f32) * 0.083).cos())
            .collect();
        let short_k: Array = (short_k_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();
        let short_v: Array = (short_v_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();
        src_short
            .update_and_fetch(&short_k, &short_v, &[4])
            .expect("short src update");

        let mut src_long = KVCache::new(1, 2, 8, 8, Dtype::Float32, 16)
            .with_step(4)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable long src turboquant");
        let long_k_data: Vec<f32> = (0..(1 * 2 * 8 * 8))
            .map(|i| 1.0 + ((i as f32) * 0.097).sin())
            .collect();
        let long_v_data: Vec<f32> = (0..(1 * 2 * 8 * 8))
            .map(|i| 1.0 + ((i as f32) * 0.109).cos())
            .collect();
        let long_k: Array = (long_k_data.as_slice(), (1_i32, 2_i32, 8_i32, 8_i32))
            .try_into()
            .unwrap();
        let long_v: Array = (long_v_data.as_slice(), (1_i32, 2_i32, 8_i32, 8_i32))
            .try_into()
            .unwrap();
        src_long
            .update_and_fetch(&long_k, &long_v, &[8])
            .expect("long src update");

        let mut dst = KVCache::new(2, 2, 8, 8, Dtype::Float32, 16)
            .with_step(4)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable dst turboquant");
        dst.adopt_row_from(&src_short, 0, 0)
            .expect("adopt short row");
        let (row0_before, _) = dst
            .turboquant()
            .expect("dst turboquant")
            .materialize_prefix_on(4, Dtype::Float32, ())
            .expect("materialize row0 before grow");

        dst.adopt_row_from(&src_long, 1, 0).expect("adopt long row");

        assert_eq!(dst.offsets(), &[4, 8]);
        assert!(
            dst.keys.is_none(),
            "TurboQuant adopt must not allocate dense K"
        );
        assert!(
            dst.values.is_none(),
            "TurboQuant adopt must not allocate dense V"
        );
        let (rows_after, _) = dst
            .turboquant()
            .expect("dst turboquant")
            .materialize_prefix_on(8, Dtype::Float32, ())
            .expect("materialize rows after grow");
        let before = row0_before.to_vec::<f32>().expect("row0 before to_vec");
        let after = rows_after.to_vec::<f32>().expect("rows after to_vec");
        let before_row_stride = 2 * 4 * 8;
        let after_row_stride = 2 * 8 * 8;
        for head in 0..2 {
            for seq in 0..4 {
                for dim in 0..8 {
                    let before_idx = head * 4 * 8 + seq * 8 + dim;
                    let after_idx = head * 8 * 8 + seq * 8 + dim;
                    assert_eq!(
                        before[before_idx], after[after_idx],
                        "row 0 packed payload changed at h={head} seq={seq} dim={dim}"
                    );
                }
            }
        }
        assert_eq!(before.len(), 2 * before_row_stride);
        assert_eq!(after.len(), 2 * after_row_stride);
    }

    #[test]
    fn kvcache_turboquant_rejects_invalid_configuration() {
        let bad_bits = TurboQuantKVBits::new(2, 4);
        assert!(bad_bits.is_err());

        let bad_k_dim =
            KVCache::new(1, 2, 7, 8, Dtype::Float32, 16).with_turboquant(TurboQuantKVBits::K3V3);
        assert!(bad_k_dim.is_err());

        let bad_v_dim =
            KVCache::new(1, 2, 8, 7, Dtype::Float32, 16).with_turboquant(TurboQuantKVBits::K3V3);
        assert!(bad_v_dim.is_err());
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_paged_rejects_turboquant_coexistence() {
        let mut turbo = KVCache::new(1, 1, 8, 8, Dtype::Float32, 16)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable turboquant");
        assert!(turbo.enable_paged(4, 4).is_err());

        let mut paged = KVCache::new(1, 1, 8, 8, Dtype::Float32, 16);
        paged.enable_paged(4, 4).expect("enable paged");
        assert!(paged.enable_turboquant(TurboQuantKVBits::K4V4).is_err());
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_paged_prefill_materializes_dense_without_retaining_dense_buffers() {
        let mut c = KVCache::new(1, 1, 2, 2, Dtype::Float32, 8);
        c.enable_paged(2, 4).expect("enable paged");
        let k_data: Vec<f32> = (0..10).map(|i| i as f32 + 1.0).collect();
        let v_data: Vec<f32> = k_data.iter().map(|x| x * 10.0).collect();
        let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 5_i32, 2_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 5_i32, 2_i32))
            .try_into()
            .unwrap();

        let (k_read, v_read) = c
            .update_and_fetch_for_attention(&k, &v, &[5])
            .expect("paged prefill");

        assert_eq!(c.offsets(), &[5]);
        assert!(c.keys.is_none());
        assert!(c.values.is_none());
        assert_eq!(c.paged().expect("paged").allocated_pages(), 3);
        assert_eq!(k_read.to_vec::<f32>().unwrap(), k_data);
        assert_eq!(v_read.to_vec::<f32>().unwrap(), v_data);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_paged_decode_attention_matches_dense_sdpa() {
        let mut dense = KVCache::new(2, 1, 4, 4, Dtype::Float32, 8).with_step(8);
        let mut paged = KVCache::new(2, 1, 4, 4, Dtype::Float32, 8).with_step(8);
        paged.enable_paged(2, 8).expect("enable paged");

        let prefix_k_data: Vec<f32> = (0..(2 * 1 * 4 * 4))
            .map(|i| ((i % 19) as f32 - 9.0) * 0.03)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(2 * 1 * 4 * 4))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.025)
            .collect();
        let prefix_k: Array = (prefix_k_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
            .try_into()
            .unwrap();
        let prefix_v: Array = (prefix_v_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
            .try_into()
            .unwrap();
        dense
            .update_and_fetch_for_attention(&prefix_k, &prefix_v, &[4, 2])
            .expect("dense prefix");
        paged
            .update_and_fetch_for_attention(&prefix_k, &prefix_v, &[4, 2])
            .expect("paged prefix");

        let q_data: Vec<f32> = (0..(2 * 2 * 4))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
            .collect();
        let step_k_data: Vec<f32> = (0..(2 * 1 * 1 * 4))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.04)
            .collect();
        let step_v_data: Vec<f32> = (0..(2 * 1 * 1 * 4))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();
        let q: Array = (q_data.as_slice(), (2_i32, 2_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_k: Array = (step_k_data.as_slice(), (2_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_v: Array = (step_v_data.as_slice(), (2_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let scale = 0.5_f32;

        let actual = paged
            .try_update_and_attend_decode(&q, &step_k, &step_v, &[1, 1], scale, None)
            .expect("paged decode")
            .expect("paged path");
        let (k_ref, v_ref) = dense
            .update_and_fetch_for_attention(&step_k, &step_v, &[1, 1])
            .expect("dense decode write");
        let mask =
            crate::core::generate::build_per_row_decode_mask(dense.offsets(), 5, Dtype::Float32)
                .expect("mask");
        let expected = mlx::fast::scaled_dot_product_attention(
            &q,
            &k_ref,
            &v_ref,
            scale,
            "",
            Some(&mask),
            None,
        )
        .expect("dense sdpa");

        assert_eq!(paged.offsets(), dense.offsets());
        assert!(paged.keys.is_none());
        assert!(paged.values.is_none());
        let actual = actual.to_vec::<f32>().unwrap();
        let expected = expected.to_vec::<f32>().unwrap();
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= 1.0e-4, "idx={idx} actual={a} expected={e}");
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_paged_decode_attention_accepts_scheduler_decode_mask() {
        let mut dense = KVCache::new(2, 1, 4, 4, Dtype::Float32, 8).with_step(8);
        let mut paged = KVCache::new(2, 1, 4, 4, Dtype::Float32, 8).with_step(8);
        paged.enable_paged(2, 8).expect("enable paged");

        let prefix_k_data: Vec<f32> = (0..(2 * 1 * 4 * 4))
            .map(|i| ((i % 19) as f32 - 9.0) * 0.03)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(2 * 1 * 4 * 4))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.025)
            .collect();
        let prefix_k: Array = (prefix_k_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
            .try_into()
            .unwrap();
        let prefix_v: Array = (prefix_v_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
            .try_into()
            .unwrap();
        dense
            .update_and_fetch_for_attention(&prefix_k, &prefix_v, &[4, 2])
            .expect("dense prefix");
        paged
            .update_and_fetch_for_attention(&prefix_k, &prefix_v, &[4, 2])
            .expect("paged prefix");

        let q_data: Vec<f32> = (0..(2 * 2 * 4))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
            .collect();
        let step_k_data: Vec<f32> = (0..(2 * 1 * 1 * 4))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.04)
            .collect();
        let step_v_data: Vec<f32> = (0..(2 * 1 * 1 * 4))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();
        let q: Array = (q_data.as_slice(), (2_i32, 2_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_k: Array = (step_k_data.as_slice(), (2_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_v: Array = (step_v_data.as_slice(), (2_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let scale = 0.5_f32;
        let mask = crate::core::generate::build_per_row_decode_mask(&[5, 3], 5, Dtype::Float32)
            .expect("scheduler decode mask");

        let actual = paged
            .try_update_and_attend_decode(&q, &step_k, &step_v, &[1, 1], scale, Some(&mask))
            .expect("masked paged decode dispatch")
            .expect("paged path with scheduler decode mask");
        let (k_ref, v_ref) = dense
            .update_and_fetch_for_attention(&step_k, &step_v, &[1, 1])
            .expect("dense decode write");
        let expected = mlx::fast::scaled_dot_product_attention(
            &q,
            &k_ref,
            &v_ref,
            scale,
            "",
            Some(&mask),
            None,
        )
        .expect("dense sdpa");

        assert_eq!(paged.offsets(), dense.offsets());
        assert!(
            paged.keys.is_none(),
            "paged decode must not allocate dense K"
        );
        assert!(
            paged.values.is_none(),
            "paged decode must not allocate dense V"
        );
        let actual = actual.to_vec::<f32>().unwrap();
        let expected = expected.to_vec::<f32>().unwrap();
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= 1.0e-4, "idx={idx} actual={a} expected={e}");
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_attention_read_returns_dense_when_turboquant_disabled() {
        let mut c = KVCache::new(1, 1, 8, 8, Dtype::Float32, 16).with_step(16);
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.125 - 1.0).collect();
        let v_data: Vec<f32> = (0..16).map(|i| 1.0 - (i as f32) * 0.0625).collect();
        let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 2_i32, 8_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 2_i32, 8_i32))
            .try_into()
            .unwrap();

        let (k_read, v_read) = c
            .update_and_fetch_for_attention(&k, &v, &[2])
            .expect("attention read");

        assert_eq!(k_read.to_vec::<f32>().unwrap(), k_data);
        assert_eq!(v_read.to_vec::<f32>().unwrap(), v_data);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_attention_read_materializes_turboquant_packed_cache_when_enabled() {
        let mut c = KVCache::new(1, 1, 8, 8, Dtype::Float32, 16)
            .with_step(16)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable turboquant");
        let k_data: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.173).sin() * 1.7).collect();
        let v_data: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.219).cos() * 1.3).collect();
        let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 2_i32, 8_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 2_i32, 8_i32))
            .try_into()
            .unwrap();

        let (k_read, v_read) = c
            .update_and_fetch_for_attention(&k, &v, &[2])
            .expect("attention read");
        let (k_mat, v_mat) = c
            .turboquant()
            .expect("turboquant cache")
            .materialize_prefix_on(2, Dtype::Float32, ())
            .expect("materialize");

        assert_eq!(k_read.shape().as_slice(), &[1, 1, 2, 8]);
        assert_eq!(v_read.shape().as_slice(), &[1, 1, 2, 8]);
        assert_eq!(k_read.dtype(), Dtype::Float32);
        assert_eq!(v_read.dtype(), Dtype::Float32);
        assert_eq!(
            k_read.to_vec::<f32>().unwrap(),
            k_mat.to_vec::<f32>().unwrap()
        );
        assert_eq!(
            v_read.to_vec::<f32>().unwrap(),
            v_mat.to_vec::<f32>().unwrap()
        );

        let max_dense_diff = k_read
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(k_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_dense_diff > 1.0e-4,
            "TurboQuant read should consume quantized packed cache, not exact dense K"
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_turboquant_decode_attention_uses_packed_path() {
        let mut c = KVCache::new(1, 1, 64, 64, Dtype::Float32, 8)
            .with_step(8)
            .with_turboquant(TurboQuantKVBits::K3V4)
            .expect("enable turboquant");
        let prefix_k_data: Vec<f32> = (0..(4 * 64))
            .map(|i| ((i as f32) * 0.031).sin() * 0.9)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(4 * 64))
            .map(|i| ((i as f32) * 0.047).cos() * 1.1)
            .collect();
        let prefix_k: Array = (prefix_k_data.as_slice(), (1_i32, 1_i32, 4_i32, 64_i32))
            .try_into()
            .unwrap();
        let prefix_v: Array = (prefix_v_data.as_slice(), (1_i32, 1_i32, 4_i32, 64_i32))
            .try_into()
            .unwrap();
        c.update_and_fetch(&prefix_k, &prefix_v, &[4])
            .expect("prefix update");

        let q_data: Vec<f32> = (0..(2 * 64))
            .map(|i| ((i as f32) * 0.053).sin() * 0.7)
            .collect();
        let decode_k_data: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.071).cos() * 0.8).collect();
        let decode_v_data: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.083).sin() * 1.2).collect();
        let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 1_i32, 64_i32))
            .try_into()
            .unwrap();
        let decode_k: Array = (decode_k_data.as_slice(), (1_i32, 1_i32, 1_i32, 64_i32))
            .try_into()
            .unwrap();
        let decode_v: Array = (decode_v_data.as_slice(), (1_i32, 1_i32, 1_i32, 64_i32))
            .try_into()
            .unwrap();
        let scale = (64_f32).sqrt().recip();

        let actual = c
            .try_update_and_attend_decode(&q, &decode_k, &decode_v, &[1], scale, None)
            .expect("decode attention")
            .expect("turboquant packed path");

        assert_eq!(c.offsets(), &[5]);
        assert!(c.keys.is_none(), "packed decode must not allocate dense K");
        assert!(
            c.values.is_none(),
            "packed decode must not allocate dense V"
        );

        let (k_ref, v_ref) = c
            .turboquant()
            .expect("turboquant cache")
            .materialize_prefix_on(5, Dtype::Float32, ())
            .expect("materialize reference");
        let expected =
            mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, scale, "", None, None)
                .expect("dense reference");

        assert_eq!(actual.shape().as_slice(), &[1, 2, 1, 64]);
        let actual = actual.to_vec::<f32>().unwrap();
        let expected = expected.to_vec::<f32>().unwrap();
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-3,
                "idx={idx} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_turboquant_multirow_attention_uses_packed_path() {
        let prefix_len = mlx::fast::TURBOQUANT_MULTIROW_MIN_SEQ_LEN;
        let q_rows = 3_i32;
        let total_len = prefix_len + q_rows;
        let head_dim = 64_i32;
        let mut cache = KVCache::new(1, 1, head_dim, head_dim, Dtype::Float32, total_len)
            .with_step(total_len)
            .with_turboquant(TurboQuantKVBits::K3V4)
            .expect("enable turboquant");
        let prefix_k_data: Vec<f32> = (0..(prefix_len * head_dim))
            .map(|i| ((i as f32) * 0.0031).sin() * 0.9)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(prefix_len * head_dim))
            .map(|i| ((i as f32) * 0.0047).cos() * 1.1)
            .collect();
        let prefix_k: Array = (
            prefix_k_data.as_slice(),
            (1_i32, 1_i32, prefix_len, head_dim),
        )
            .try_into()
            .unwrap();
        let prefix_v: Array = (
            prefix_v_data.as_slice(),
            (1_i32, 1_i32, prefix_len, head_dim),
        )
            .try_into()
            .unwrap();
        cache
            .update_and_fetch(&prefix_k, &prefix_v, &[prefix_len])
            .expect("prefix update");

        let q_data: Vec<f32> = (0..(4 * q_rows * head_dim))
            .map(|i| ((i as f32) * 0.0053).sin() * 0.7)
            .collect();
        let step_k_data: Vec<f32> = (0..(q_rows * head_dim))
            .map(|i| ((i as f32) * 0.0071).cos() * 0.8)
            .collect();
        let step_v_data: Vec<f32> = (0..(q_rows * head_dim))
            .map(|i| ((i as f32) * 0.0083).sin() * 1.2)
            .collect();
        let q: Array = (q_data.as_slice(), (1_i32, 4_i32, q_rows, head_dim))
            .try_into()
            .unwrap();
        let step_k: Array = (step_k_data.as_slice(), (1_i32, 1_i32, q_rows, head_dim))
            .try_into()
            .unwrap();
        let step_v: Array = (step_v_data.as_slice(), (1_i32, 1_i32, q_rows, head_dim))
            .try_into()
            .unwrap();
        let scale = (head_dim as f32).sqrt().recip();

        let actual = cache
            .try_update_and_attend_on(&q, &step_k, &step_v, &[q_rows], scale, None, ())
            .expect("multi-row attention")
            .expect("turboquant multi-row packed path");

        assert_eq!(cache.offsets(), &[total_len]);
        assert!(
            cache.keys.is_none(),
            "packed path must not allocate dense K"
        );
        assert!(
            cache.values.is_none(),
            "packed path must not allocate dense V"
        );
        let (k_ref, v_ref) = cache
            .turboquant()
            .expect("turboquant cache")
            .materialize_prefix_on(total_len, Dtype::Float32, ())
            .expect("materialize reference");
        let expected = mlx::fast::scaled_dot_product_attention(
            &q, &k_ref, &v_ref, scale, "causal", None, None,
        )
        .expect("dense causal reference");

        assert_eq!(actual.shape().as_slice(), &[1, 4, q_rows, head_dim]);
        let actual = actual.to_vec::<f32>().unwrap();
        let expected = expected.to_vec::<f32>().unwrap();
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff < 2.5e-2,
                "idx={idx} actual={actual} expected={expected} diff={diff}"
            );
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn kvcache_turboquant_multirow_attention_handles_ragged_batch() {
        let prefix_rows = [2048_i32, 2040_i32];
        let query_rows = 4_i32;
        let query_lens = [4_i32, 2_i32];
        let kv_lens = [2052_i32, 2042_i32];
        let max_len = kv_lens[0];
        let head_dim = 64_i32;
        let mut cache = KVCache::new(2, 1, head_dim, head_dim, Dtype::Float32, max_len)
            .with_step(max_len)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable turboquant");

        let prefix_values = (2 * prefix_rows[0] * head_dim) as usize;
        let prefix_k_data = (0..prefix_values)
            .map(|i| ((i as f32) * 0.0029).sin() * 0.8)
            .collect::<Vec<_>>();
        let prefix_v_data = (0..prefix_values)
            .map(|i| ((i as f32) * 0.0041).cos())
            .collect::<Vec<_>>();
        let prefix_k: Array = (
            prefix_k_data.as_slice(),
            (2_i32, 1_i32, prefix_rows[0], head_dim),
        )
            .try_into()
            .unwrap();
        let prefix_v: Array = (
            prefix_v_data.as_slice(),
            (2_i32, 1_i32, prefix_rows[0], head_dim),
        )
            .try_into()
            .unwrap();
        cache
            .update_and_fetch(&prefix_k, &prefix_v, &prefix_rows)
            .expect("ragged prefix update");

        let q_values = (2 * 4 * query_rows * head_dim) as usize;
        let kv_values = (2 * query_rows * head_dim) as usize;
        let q_data = (0..q_values)
            .map(|i| ((i as f32) * 0.0059).sin() * 0.7)
            .collect::<Vec<_>>();
        let step_k_data = (0..kv_values)
            .map(|i| ((i as f32) * 0.0073).cos() * 0.9)
            .collect::<Vec<_>>();
        let step_v_data = (0..kv_values)
            .map(|i| ((i as f32) * 0.0089).sin() * 1.1)
            .collect::<Vec<_>>();
        let q: Array = (q_data.as_slice(), (2_i32, 4_i32, query_rows, head_dim))
            .try_into()
            .unwrap();
        let step_k: Array = (step_k_data.as_slice(), (2_i32, 1_i32, query_rows, head_dim))
            .try_into()
            .unwrap();
        let step_v: Array = (step_v_data.as_slice(), (2_i32, 1_i32, query_rows, head_dim))
            .try_into()
            .unwrap();
        let scale = (head_dim as f32).sqrt().recip();
        let actual = cache
            .try_update_and_attend_on(&q, &step_k, &step_v, &query_lens, scale, None, ())
            .expect("ragged multi-row attention")
            .expect("ragged batch uses packed path");

        assert_eq!(cache.offsets(), &kv_lens);
        assert!(cache.keys.is_none());
        assert!(cache.values.is_none());
        let (k_ref, v_ref) = cache
            .turboquant()
            .expect("turboquant cache")
            .materialize_prefix_on(max_len, Dtype::Float32, ())
            .expect("materialize ragged reference");
        let mut mask_data = vec![f32::NEG_INFINITY; (2 * query_rows * max_len) as usize];
        for batch in 0..2_usize {
            for row in 0..query_lens[batch] as usize {
                let visible = (kv_lens[batch] - query_lens[batch] + row as i32 + 1) as usize;
                let start = (batch * query_rows as usize + row) * max_len as usize;
                mask_data[start..start + visible].fill(0.0);
            }
        }
        let mask: Array = (mask_data.as_slice(), (2_i32, 1_i32, query_rows, max_len))
            .try_into()
            .unwrap();
        let expected = mlx::fast::scaled_dot_product_attention(
            &q,
            &k_ref,
            &v_ref,
            scale,
            "",
            Some(&mask),
            None,
        )
        .expect("dense ragged reference");

        let actual = actual.to_vec::<f32>().unwrap();
        let expected = expected.to_vec::<f32>().unwrap();
        let row_size = head_dim as usize;
        for batch in 0..2_usize {
            for head in 0..4_usize {
                for row in 0..query_rows as usize {
                    let start = ((batch * 4 + head) * query_rows as usize + row) * row_size;
                    if row >= query_lens[batch] as usize {
                        assert!(
                            actual[start..start + row_size]
                                .iter()
                                .all(|value| value.abs() < 1.0e-6),
                            "padded output must be zero: batch={batch} head={head} row={row}"
                        );
                        continue;
                    }
                    for dim in 0..row_size {
                        let diff = (actual[start + dim] - expected[start + dim]).abs();
                        assert!(
                            diff < 2.5e-2,
                            "batch={batch} head={head} row={row} dim={dim} diff={diff}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn kvcache_multi_step_accumulation() {
        // Verify two successive update_and_fetch calls accumulate per-row
        // offsets correctly, returned slice grows along axis 2, and the
        // K values written in step 1 stay intact at positions [0..4]
        // after step 2 writes positions [4..8].
        let mut c = make_cache_b(2, 1024);

        // Step 1: write 4 K/V tokens per row with marker values.
        // K shape [2, 4, 4, 256]; row 0 filled with 1.0, row 1 with 2.0.
        let n_kv_heads = 4;
        let head_dim = 256;
        let n_per_row_step1 = (n_kv_heads * 4 * head_dim) as usize;
        let mut k1_data: Vec<f32> = Vec::with_capacity(2 * n_per_row_step1);
        k1_data.extend(std::iter::repeat(1.0_f32).take(n_per_row_step1));
        k1_data.extend(std::iter::repeat(2.0_f32).take(n_per_row_step1));
        let v1_data: Vec<f32> = k1_data.iter().map(|x| x * 10.0).collect();
        let k1: Array = (&k1_data[..], (2_i32, 4_i32, 4_i32, 256_i32))
            .try_into()
            .unwrap();
        let v1: Array = (&v1_data[..], (2_i32, 4_i32, 4_i32, 256_i32))
            .try_into()
            .unwrap();

        let (kf1, _vf1) = c.update_and_fetch(&k1, &v1, &[4, 4]).expect("step 1");
        assert_eq!(c.offsets(), &[4, 4]);
        assert_eq!(kf1.shape().as_slice(), &[2, 4, 4, 256]);

        // Step 2: write 4 more K/V tokens per row with different marker
        // values (row 0 = 3.0, row 1 = 4.0).
        let mut k2_data: Vec<f32> = Vec::with_capacity(2 * n_per_row_step1);
        k2_data.extend(std::iter::repeat(3.0_f32).take(n_per_row_step1));
        k2_data.extend(std::iter::repeat(4.0_f32).take(n_per_row_step1));
        let v2_data: Vec<f32> = k2_data.iter().map(|x| x * 10.0).collect();
        let k2: Array = (&k2_data[..], (2_i32, 4_i32, 4_i32, 256_i32))
            .try_into()
            .unwrap();
        let v2: Array = (&v2_data[..], (2_i32, 4_i32, 4_i32, 256_i32))
            .try_into()
            .unwrap();

        let (kf2, _vf2) = c.update_and_fetch(&k2, &v2, &[4, 4]).expect("step 2");
        assert_eq!(c.offsets(), &[8, 8]);
        assert_eq!(kf2.shape().as_slice(), &[2, 4, 8, 256]);

        // Verify accumulated K exhaustively. Row-major [B=2, n_kv_heads=4, S=8, head_dim=256]:
        //   row 0 cols [0..4] = 1.0 (step 1), cols [4..8] = 3.0 (step 2)
        //   row 1 cols [0..4] = 2.0 (step 1), cols [4..8] = 4.0 (step 2)
        let kf2_vec: Vec<f32> = kf2.to_vec().expect("to_vec K");
        let total_k = 2 * 4 * 8 * 256;
        assert_eq!(kf2_vec.len(), total_k);
        let stride_batch = 4 * 8 * 256; // n_kv_heads * S * head_dim
        let stride_head = 8 * 256; // S * head_dim
        let stride_seq = 256; // head_dim
        for b_idx in 0..2 {
            let row_marker_step1 = if b_idx == 0 { 1.0_f32 } else { 2.0_f32 };
            let row_marker_step2 = if b_idx == 0 { 3.0_f32 } else { 4.0_f32 };
            for h in 0..4 {
                for col in 0..8 {
                    for d in 0..256 {
                        let idx = b_idx * stride_batch + h * stride_head + col * stride_seq + d;
                        let expected = if col < 4 {
                            row_marker_step1
                        } else {
                            row_marker_step2
                        };
                        assert_eq!(
                            kf2_vec[idx],
                            expected,
                            "K mismatch at b={b_idx} h={h} col={col} d={d}: expected {expected}, got {}",
                            kf2_vec[idx]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn kvcache_per_row_data_isolation() {
        // Verify that a single update_and_fetch with row-distinct K values
        // produces cache contents where row 0's slab contains only row 0
        // data (no cross-row contamination from Strategy A's B-loop writes).
        let mut c = make_cache_b(2, 1024);

        let n_kv_heads = 4;
        let head_dim = 256;
        let n_per_row = (n_kv_heads * 4 * head_dim) as usize;
        // Row 0 K = all 1.0; row 1 K = all 2.0.
        let mut k_data: Vec<f32> = Vec::with_capacity(2 * n_per_row);
        k_data.extend(std::iter::repeat(1.0_f32).take(n_per_row));
        k_data.extend(std::iter::repeat(2.0_f32).take(n_per_row));
        let v_data: Vec<f32> = k_data.iter().map(|x| x * 10.0).collect();
        let k: Array = (&k_data[..], (2_i32, 4_i32, 4_i32, 256_i32))
            .try_into()
            .unwrap();
        let v: Array = (&v_data[..], (2_i32, 4_i32, 4_i32, 256_i32))
            .try_into()
            .unwrap();

        let (kf, vf) = c.update_and_fetch(&k, &v, &[4, 4]).expect("update");
        assert_eq!(c.offsets(), &[4, 4]);

        let kf_vec: Vec<f32> = kf.to_vec().expect("to_vec K");
        let total = (2 * 4 * 4 * 256) as usize;
        assert_eq!(kf_vec.len(), total);
        let row_stride = 4 * 4 * 256; // n_kv_heads * cap_so_far * head_dim
                                      // K row 0: every element in slab [0 * row_stride .. 1 * row_stride] must be 1.0
        for i in 0..row_stride {
            assert_eq!(kf_vec[i], 1.0_f32, "K row 0 slab corrupted at index {i}");
        }
        // K row 1: every element in slab [1 * row_stride .. 2 * row_stride] must be 2.0
        for i in row_stride..(2 * row_stride) {
            assert_eq!(kf_vec[i], 2.0_f32, "K row 1 slab corrupted at index {i}");
        }

        // V verification: same exhaustive loop pattern as K, but V markers are
        // 10.0 (row 0) / 20.0 (row 1) because v_data = k_data * 10.0.
        let vf_vec: Vec<f32> = vf.to_vec().expect("to_vec V");
        assert_eq!(vf_vec.len(), total);
        for i in 0..row_stride {
            assert_eq!(vf_vec[i], 10.0_f32, "V row 0 slab corrupted at index {i}");
        }
        for i in row_stride..(2 * row_stride) {
            assert_eq!(vf_vec[i], 20.0_f32, "V row 1 slab corrupted at index {i}");
        }
    }

    #[test]
    fn kvcache_adopt_row_from_basic() {
        // src: B=1, write 4 K/V tokens with marker values:
        // K = 7.0, V = 70.0. Shape: [B=1, n_kv_heads=4, seq=4, head_dim=256].
        let mut src = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024);
        let n_per_row = (4 * 4 * 256) as usize;
        let k_data: Vec<f32> = std::iter::repeat(7.0_f32).take(n_per_row).collect();
        let v_data: Vec<f32> = std::iter::repeat(70.0_f32).take(n_per_row).collect();
        let k: Array = (&k_data[..], (1_i32, 4_i32, 4_i32, 256_i32))
            .try_into()
            .unwrap();
        let v: Array = (&v_data[..], (1_i32, 4_i32, 4_i32, 256_i32))
            .try_into()
            .unwrap();
        src.update_and_fetch(&k, &v, &[4]).expect("src write");
        assert_eq!(src.offsets(), &[4]);

        // dst: B=2, fresh.
        let mut dst = KVCache::new(2, 4, 256, 256, Dtype::Float32, 1024);
        dst.adopt_row_from(&src, /*dst_row=*/ 1, /*src_row=*/ 0)
            .expect("adopt_row_from basic");

        assert_eq!(dst.offsets(), &[0, 4]);
        assert_eq!(dst.cap(), 1024);

        // Read back dst K/V by doing a 1-token write to row 1 with
        // per_row_lens=[0, 1] (row 0 skipped, row 1 writes 1 token at
        // offset 4 → post-write offsets=[0, 5]). The returned slice
        // is [B=2, n_kv_heads=4, max_off=5, head_dim=256]. Row 1 cols
        // [0..4] are the adopted values from src.
        let probe_k_data: Vec<f32> = vec![0.0_f32; (2 * 4 * 1 * 256) as usize];
        let probe_v_data: Vec<f32> = vec![0.0_f32; (2 * 4 * 1 * 256) as usize];
        let probe_k: Array = (&probe_k_data[..], (2_i32, 4_i32, 1_i32, 256_i32))
            .try_into()
            .unwrap();
        let probe_v: Array = (&probe_v_data[..], (2_i32, 4_i32, 1_i32, 256_i32))
            .try_into()
            .unwrap();
        let (kf, vf) = dst
            .update_and_fetch(&probe_k, &probe_v, &[0, 1])
            .expect("probe write to row 1");
        assert_eq!(kf.shape().as_slice(), &[2, 4, 5, 256]);
        assert_eq!(vf.shape().as_slice(), &[2, 4, 5, 256]);
        let kf_vec: Vec<f32> = kf.to_vec().expect("kf to_vec");
        let vf_vec: Vec<f32> = vf.to_vec().expect("vf to_vec");

        // Layout: [B=2, n_kv_heads=4, S=5, head_dim=256] row-major.
        let stride_batch = 4 * 5 * 256;
        let stride_head = 5 * 256;
        let stride_seq = 256;

        // Row 1 (adopted), col 0, head 0, dim 0 → K=7.0 / V=70.0
        assert_eq!(
            kf_vec[stride_batch + 0 * stride_head + 0 * stride_seq + 0],
            7.0_f32,
            "dst row 1 col 0 head 0 dim 0 should be 7.0 (adopted K)"
        );
        assert_eq!(
            vf_vec[stride_batch + 0 * stride_head + 0 * stride_seq + 0],
            70.0_f32,
            "dst row 1 col 0 head 0 dim 0 should be 70.0 (adopted V)"
        );
        // Row 1, col 3 (last adopted), head 3, dim 255 → K=7.0 / V=70.0
        assert_eq!(
            kf_vec[stride_batch + 3 * stride_head + 3 * stride_seq + 255],
            7.0_f32,
            "dst row 1 col 3 head 3 dim 255 should be 7.0 (adopted K, last cell)"
        );
        assert_eq!(
            vf_vec[stride_batch + 3 * stride_head + 3 * stride_seq + 255],
            70.0_f32,
            "dst row 1 col 3 head 3 dim 255 should be 70.0 (adopted V, last cell)"
        );
        // Row 0 (un-adopted), any cell → 0.0 (untouched zero buffer)
        assert_eq!(
            kf_vec[0 * stride_batch + 0 * stride_head + 0 * stride_seq + 0],
            0.0_f32,
            "dst row 0 should be untouched (zero buffer)"
        );
        assert_eq!(
            vf_vec[0 * stride_batch + 0 * stride_head + 0 * stride_seq + 0],
            0.0_f32,
            "dst row 0 V should be untouched (zero buffer)"
        );
    }

    #[test]
    fn kvcache_adopt_row_from_shape_mismatch_err() {
        // src has different n_kv_heads → adopt_row_from must Err.
        let src = KVCache::new(
            1,
            8, /* different n_kv_heads */
            256,
            256,
            Dtype::Float32,
            1024,
        );
        let mut dst = KVCache::new(2, 4, 256, 256, Dtype::Float32, 1024);
        let r = dst.adopt_row_from(&src, 1, 0);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("mismatch") || msg.contains("shape"),
            "msg should mention shape mismatch; got: {msg}"
        );
    }

    #[test]
    fn kvcache_adopt_row_from_out_of_bounds_err() {
        // Case 1: dst_row >= self.batch
        let src = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024);
        let mut dst = KVCache::new(2, 4, 256, 256, Dtype::Float32, 1024);
        let r = dst.adopt_row_from(&src, 2, 0);
        assert!(r.is_err(), "dst_row=2 with batch=2 should Err");
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("dst_row") || msg.contains("batch"),
            "msg should mention dst_row OOB; got: {msg}"
        );

        // Case 2: src_row >= src.batch
        let src2 = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024);
        let mut dst2 = KVCache::new(2, 4, 256, 256, Dtype::Float32, 1024);
        let r2 = dst2.adopt_row_from(&src2, 0, 1);
        assert!(r2.is_err(), "src_row=1 with src.batch=1 should Err");
        let msg2 = format!("{}", r2.unwrap_err());
        assert!(
            msg2.contains("src_row") || msg2.contains("batch"),
            "msg should mention src_row OOB; got: {msg2}"
        );

        // Case 3: src.offsets[src_row] > self.cap
        // src writes 8 tokens (offset=8); dst has cap=4. adopt_row_from must Err.
        let mut src3 = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024);
        let n_per_row = (4 * 8 * 256) as usize;
        let k_data: Vec<f32> = std::iter::repeat(1.0_f32).take(n_per_row).collect();
        let v_data: Vec<f32> = std::iter::repeat(1.0_f32).take(n_per_row).collect();
        let k: Array = (&k_data[..], (1_i32, 4_i32, 8_i32, 256_i32))
            .try_into()
            .unwrap();
        let v: Array = (&v_data[..], (1_i32, 4_i32, 8_i32, 256_i32))
            .try_into()
            .unwrap();
        src3.update_and_fetch(&k, &v, &[8]).expect("src3 write");
        // dst with cap=4 (< src's offset=8) → adopt_row_from should Err.
        let mut dst3 = KVCache::new(2, 4, 256, 256, Dtype::Float32, 4 /* cap=4 */);
        let r3 = dst3.adopt_row_from(&src3, 0, 0);
        assert!(r3.is_err(), "src.offsets=8 > self.cap=4 should Err");
        let msg3 = format!("{}", r3.unwrap_err());
        assert!(
            msg3.contains("cap"),
            "msg should mention cap exceeded; got: {msg3}"
        );
    }

    #[test]
    fn kvcache_adopt_row_from_dtype_mismatch_err() {
        // src: Bfloat16; dst: Float32 → adopt_row_from must Err.
        let src = KVCache::new(1, 4, 256, 256, Dtype::Bfloat16, 1024);
        let mut dst = KVCache::new(2, 4, 256, 256, Dtype::Float32, 1024);
        let r = dst.adopt_row_from(&src, 1, 0);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("mismatch") || msg.contains("dtype"),
            "msg should mention dtype mismatch; got: {msg}"
        );
    }
}
