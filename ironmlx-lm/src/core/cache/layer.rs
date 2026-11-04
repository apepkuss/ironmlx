//! Model-side cache variants, snapshots and prefix conversion operations.

use crate::core::cache::prefix_payload::{
    PagedPrefixEntry, PagedPrefixKeySpec, PagedPrefixLayer, PrefixLayerKind, PrefixLayerPayload,
    PrefixLayerSpec, PrefixTensorSpec,
};
use crate::core::cache::prefix_segments::{
    PrefixKvSegment, PrefixLineage, PrefixSegmentId, PrefixSegmentLayer, PrefixSegmentLayout,
};
use crate::core::cache::{
    GatedDeltaCache, GatedDeltaCacheSnapshot, KVCache, KVCacheSnapshot, PagedKvHotColdConfig,
    TurboQuantKVBits,
};
use crate::models::glm4_moe_lite::mla_cache::MlaLatentCacheSnapshot;
use mlx::Dtype;

/// Model-side per-layer cache; separate from decoder layer computation.
#[doc(hidden)]
pub enum LayerCache {
    Full(KVCache),
    Linear(GatedDeltaCache),
    Mla(crate::models::glm4_moe_lite::mla_cache::MlaLatentCache),
    #[cfg(unix)]
    Flash(crate::models::qwen4_exp::cache::FlashLayerCache),
}

/// Per-layer cache checkpoint used by speculative decoding rollback.
#[doc(hidden)]
pub enum LayerCacheSnapshot {
    Full(KVCacheSnapshot),
    Linear(GatedDeltaCacheSnapshot),
    Mla(MlaLatentCacheSnapshot),
    #[cfg(unix)]
    Flash(crate::models::qwen4_exp::cache::FlashLayerSnapshot),
}

impl LayerCache {
    /// Arrays holding this layer's cache state (KV buffers, or the GatedDelta
    /// conv and recurrent states), for scheduling their pending updates.
    pub fn state_arrays(&self) -> Vec<&mlx::Array> {
        match self {
            LayerCache::Full(kv) => kv.diagnostic_buffers(),
            LayerCache::Linear(gd) => vec![gd.conv_state(), gd.recurrent_state()],
            LayerCache::Mla(_) => Vec::new(),
            #[cfg(unix)]
            LayerCache::Flash(cache) => cache.state_arrays(),
        }
    }

    /// Diagnostic-only: arrays holding this layer's cache state, used as
    /// evaluation barriers by `IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES`.
    #[doc(hidden)]
    pub fn diagnostic_buffers(&self) -> Vec<&mlx::Array> {
        self.state_arrays()
    }

    pub fn begin_speculative_prefix_capture(&mut self) -> anyhow::Result<()> {
        match self {
            LayerCache::Full(_) => Ok(()),
            LayerCache::Linear(cache) => cache.begin_speculative_prefix_capture(),
            #[cfg(unix)]
            LayerCache::Flash(cache) => cache.begin_speculative_prefix_capture(),
            LayerCache::Mla(_) => {
                anyhow::bail!("speculative prefix capture does not support MLA cache")
            }
        }
    }

    pub fn discard_speculative_prefix_capture(&mut self) {
        #[cfg(unix)]
        if let LayerCache::Flash(cache) = self {
            cache.discard_speculative_prefix_capture();
        }
        if let LayerCache::Linear(cache) = self {
            cache.discard_speculative_prefix_capture();
        }
    }

    pub fn enable_turboquant(&mut self, bits: TurboQuantKVBits) -> anyhow::Result<()> {
        #[cfg(unix)]
        anyhow::ensure!(
            !matches!(self, LayerCache::Flash(_)),
            "Flash text requires BF16 KV; TurboQuant is not qualified"
        );
        if let LayerCache::Full(kv) = self {
            kv.enable_turboquant(bits)?;
        }
        Ok(())
    }

    pub fn enable_paged_kv(&mut self, block_size: i32, max_pages: i32) -> anyhow::Result<()> {
        if let LayerCache::Full(kv) = self {
            kv.enable_paged(block_size, max_pages)?;
        }
        Ok(())
    }

    pub fn enable_paged_hot_cold_tiering(
        &mut self,
        config: PagedKvHotColdConfig,
    ) -> anyhow::Result<()> {
        if let LayerCache::Full(kv) = self {
            kv.enable_paged_hot_cold_tiering(config)?;
        }
        Ok(())
    }

    pub fn shrink_paged_hot_window(&mut self, hot_window_pages: i32) -> anyhow::Result<usize> {
        match self {
            LayerCache::Full(kv) => kv.shrink_paged_hot_window(hot_window_pages),
            LayerCache::Linear(_) | LayerCache::Mla(_) => Ok(0),
            #[cfg(unix)]
            LayerCache::Flash(_) => Ok(0),
        }
    }

    pub fn restore_configured_paged_hot_window(&mut self) -> bool {
        match self {
            LayerCache::Full(kv) => kv.restore_configured_paged_hot_window(),
            LayerCache::Linear(_) | LayerCache::Mla(_) => false,
            #[cfg(unix)]
            LayerCache::Flash(_) => false,
        }
    }

    /// Reset to empty state (offset → 0; recurrent state cleared). Preserves
    /// any underlying Array allocations so the next batch can reuse them.
    pub fn reset(&mut self) -> anyhow::Result<()> {
        match self {
            LayerCache::Full(kv) => {
                kv.reset();
                Ok(())
            }
            LayerCache::Linear(gd) => gd.reset(),
            LayerCache::Mla(c) => c.reset(),
            #[cfg(unix)]
            LayerCache::Flash(c) => c.reset(),
        }
    }

    /// Capture a lightweight rollback checkpoint for this layer cache.
    pub fn snapshot(&self) -> LayerCacheSnapshot {
        match self {
            LayerCache::Full(kv) => LayerCacheSnapshot::Full(kv.snapshot()),
            LayerCache::Linear(gd) => LayerCacheSnapshot::Linear(gd.snapshot()),
            LayerCache::Mla(c) => LayerCacheSnapshot::Mla(c.snapshot()),
            #[cfg(unix)]
            LayerCache::Flash(c) => LayerCacheSnapshot::Flash(c.snapshot()),
        }
    }

    /// Capture an offsets-only checkpoint for append-only Full-KV verify.
    pub fn append_snapshot(&self) -> anyhow::Result<LayerCacheSnapshot> {
        match self {
            LayerCache::Full(kv) => Ok(LayerCacheSnapshot::Full(kv.append_snapshot())),
            #[cfg(unix)]
            LayerCache::Flash(c) => Ok(LayerCacheSnapshot::Flash(c.snapshot())),
            LayerCache::Linear(_) => {
                anyhow::bail!("LayerCache::append_snapshot does not support Linear cache")
            }
            LayerCache::Mla(_) => {
                anyhow::bail!("LayerCache::append_snapshot does not support MLA cache")
            }
        }
    }

    /// Capture the transaction boundary used by DFlash2 verification.
    ///
    /// Full-attention caches are append-only during a verify window, so an
    /// offsets-only checkpoint is sufficient and avoids retaining dense K/V
    /// Array handles that would force copy-on-write of the whole prefix.
    /// Recurrent caches keep their fixed-size state checkpoint; the accepted
    /// state is committed from the separately recorded per-position replay.
    pub fn dflash2_transaction_snapshot(&self) -> anyhow::Result<LayerCacheSnapshot> {
        match self {
            LayerCache::Full(kv) => Ok(LayerCacheSnapshot::Full(kv.append_snapshot())),
            LayerCache::Linear(gd) => Ok(LayerCacheSnapshot::Linear(gd.snapshot())),
            #[cfg(unix)]
            LayerCache::Flash(c) => Ok(LayerCacheSnapshot::Flash(c.snapshot())),
            LayerCache::Mla(_) => {
                anyhow::bail!("DFlash2 transaction snapshots do not support MLA cache")
            }
        }
    }

    /// Restore this layer cache from a matching checkpoint.
    pub fn restore(&mut self, snapshot: &LayerCacheSnapshot) -> anyhow::Result<()> {
        match (self, snapshot) {
            (LayerCache::Full(kv), LayerCacheSnapshot::Full(s)) => kv.restore(s),
            (LayerCache::Linear(gd), LayerCacheSnapshot::Linear(s)) => gd.restore(s),
            (LayerCache::Mla(c), LayerCacheSnapshot::Mla(s)) => c.restore(s),
            #[cfg(unix)]
            (LayerCache::Flash(c), LayerCacheSnapshot::Flash(s)) => c.restore(s),
            #[cfg(unix)]
            (LayerCache::Flash(_), _) => {
                anyhow::bail!("Flash cache received another architecture's snapshot")
            }
            (LayerCache::Full(_), _) => {
                anyhow::bail!("LayerCache::restore: Full cache received non-Full snapshot")
            }
            (LayerCache::Linear(_), _) => {
                anyhow::bail!("LayerCache::restore: Linear cache received non-Linear snapshot")
            }
            (LayerCache::Mla(_), _) => {
                anyhow::bail!("LayerCache::restore: Mla cache received non-Mla snapshot")
            }
        }
    }
}

/// A full-attention layer whose K/V can live in shared prefix segments
/// (paged or TurboQuant storage).
fn shareable_full_kv(cache: &LayerCache) -> Option<&KVCache> {
    match cache {
        LayerCache::Full(kv) if kv.paged().is_some() || kv.turboquant().is_some() => Some(kv),
        _ => None,
    }
}

/// Row `row`'s prefix segment record, read from the first shareable
/// full-attention layer (every such layer carries the same record).
pub fn row_prefix_lineage(caches: &[LayerCache], row: usize) -> Option<PrefixLineage> {
    caches
        .iter()
        .find_map(shareable_full_kv)
        .and_then(|kv| kv.prefix_lineage(row).cloned())
}

/// Write row `row`'s prefix segment record on every shareable full-attention
/// layer. Call after a prefix restore or save has fully succeeded.
pub fn set_row_prefix_lineage(
    caches: &mut [LayerCache],
    row: usize,
    lineage: Option<PrefixLineage>,
) {
    for cache in caches {
        if shareable_full_kv(cache).is_none() {
            continue;
        }
        if let LayerCache::Full(kv) = cache {
            kv.set_prefix_lineage(row, lineage.clone());
        }
    }
}

pub fn adopt_layer_cache_rows(
    dst: &mut [LayerCache],
    src: &[LayerCache],
    dst_row: usize,
    src_row: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        dst.len() == src.len(),
        "cache layer count mismatch: destination={} source={}",
        dst.len(),
        src.len()
    );
    for (dst_layer, src_layer) in dst.iter_mut().zip(src) {
        match (dst_layer, src_layer) {
            (LayerCache::Full(dst), LayerCache::Full(src)) => {
                dst.adopt_row_from(src, dst_row, src_row)?;
            }
            (LayerCache::Linear(dst), LayerCache::Linear(src)) => {
                dst.adopt_row_from(src, dst_row, src_row)?;
            }
            (LayerCache::Mla(dst), LayerCache::Mla(src)) => {
                dst.adopt_row_from(src, dst_row, src_row)?;
            }
            #[cfg(unix)]
            (LayerCache::Flash(dst), LayerCache::Flash(src)) => {
                dst.adopt_row_from(src, dst_row, src_row)?
            }
            _ => anyhow::bail!("cache layer kind mismatch"),
        }
    }
    Ok(())
}

pub fn enable_turboquant_kv_caches(
    caches: &mut [LayerCache],
    bits: TurboQuantKVBits,
) -> anyhow::Result<()> {
    for cache in caches {
        cache.enable_turboquant(bits)?;
    }
    Ok(())
}

pub fn enable_paged_kv_caches(
    caches: &mut [LayerCache],
    block_size: i32,
    max_pages: i32,
) -> anyhow::Result<()> {
    for cache in caches {
        cache.enable_paged_kv(block_size, max_pages)?;
    }
    Ok(())
}

pub fn enable_paged_hot_cold_tiering_caches(
    caches: &mut [LayerCache],
    config: PagedKvHotColdConfig,
) -> anyhow::Result<()> {
    for cache in caches {
        cache.enable_paged_hot_cold_tiering(config.clone())?;
    }
    Ok(())
}

pub fn prefix_key_spec_for_caches(
    model_id: &str,
    token_ids: &[u32],
    cached_len: i32,
    fingerprint: Option<&str>,
    block_size: i32,
    caches: &[LayerCache],
) -> anyhow::Result<Option<PagedPrefixKeySpec>> {
    if caches.is_empty() {
        return Ok(None);
    }
    if cached_len <= 0 {
        return Ok(None);
    }
    let token_len = i32::try_from(token_ids.len())
        .map_err(|_| anyhow::anyhow!("paged prefix token length exceeds i32"))?;
    if token_len != cached_len {
        anyhow::bail!(
            "prefix_key_spec_for_caches: token length {token_len} != cached_len {cached_len}"
        );
    }
    if block_size <= 0 {
        anyhow::bail!("prefix_key_spec_for_caches: block_size must be > 0");
    }

    let mut main_layers = Vec::with_capacity(caches.len());
    let mut kv_cache_profile: Option<String> = None;
    for cache in caches {
        match cache {
            #[cfg(unix)]
            LayerCache::Flash(cache) => {
                remember_kv_cache_profile(
                    &mut kv_cache_profile,
                    Some(cache.durable_prefix_contract()?),
                )?;
                main_layers.push(cache.durable_prefix_spec(cached_len)?);
            }
            LayerCache::Full(kv) => {
                if let Some(paged) = kv.paged() {
                    remember_kv_cache_profile(&mut kv_cache_profile, kv.prefix_cache_profile())?;
                    if paged.block_size() != block_size {
                        anyhow::bail!(
                            "prefix_key_spec_for_caches: full-attention block_size {} != configured {}",
                            paged.block_size(),
                            block_size
                        );
                    }
                    let page_count = (cached_len + block_size - 1) / block_size;
                    main_layers.push(PrefixLayerSpec {
                        kind: PrefixLayerKind::FullPaged,
                        tensors: vec![
                            PrefixTensorSpec {
                                dtype: kv.dtype(),
                                shape: vec![page_count, kv.n_kv_heads(), block_size, kv.head_dim()],
                            },
                            PrefixTensorSpec {
                                dtype: kv.dtype(),
                                shape: vec![
                                    page_count,
                                    kv.n_kv_heads(),
                                    block_size,
                                    kv.v_head_dim(),
                                ],
                            },
                        ],
                    });
                } else if let Some(tq) = kv.turboquant() {
                    let profile = kv.prefix_cache_profile().ok_or_else(|| {
                        anyhow::anyhow!(
                            "prefix_key_spec_for_caches: TurboQuant cache missing prefix profile"
                        )
                    })?;
                    remember_kv_cache_profile(&mut kv_cache_profile, Some(profile))?;
                    main_layers.push(PrefixLayerSpec {
                        kind: PrefixLayerKind::FullTurboQuantPacked,
                        tensors: vec![
                            PrefixTensorSpec {
                                dtype: Dtype::Uint32,
                                shape: vec![
                                    1_i32,
                                    kv.n_kv_heads(),
                                    cached_len,
                                    tq.packed_head_dim(),
                                ],
                            },
                            PrefixTensorSpec {
                                dtype: Dtype::Float32,
                                shape: vec![1_i32, kv.n_kv_heads(), cached_len],
                            },
                            PrefixTensorSpec {
                                dtype: Dtype::Uint32,
                                shape: vec![
                                    1_i32,
                                    kv.n_kv_heads(),
                                    cached_len,
                                    tq.packed_v_head_dim(),
                                ],
                            },
                            PrefixTensorSpec {
                                dtype: Dtype::Float32,
                                shape: vec![1_i32, kv.n_kv_heads(), cached_len],
                            },
                        ],
                    });
                } else {
                    return Ok(None);
                }
            }
            LayerCache::Linear(gd) => {
                let conv_shape = gd.conv_state().shape();
                let conv_shape = conv_shape.as_slice();
                let rec_shape = gd.recurrent_state().shape();
                let rec_shape = rec_shape.as_slice();
                main_layers.push(PrefixLayerSpec {
                    kind: PrefixLayerKind::Linear,
                    tensors: vec![
                        PrefixTensorSpec {
                            dtype: gd.conv_state().dtype(),
                            shape: vec![1_i32, conv_shape[1], conv_shape[2]],
                        },
                        PrefixTensorSpec {
                            dtype: gd.recurrent_state().dtype(),
                            shape: vec![1_i32, rec_shape[1], rec_shape[2], rec_shape[3]],
                        },
                    ],
                });
            }
            LayerCache::Mla(mla) => {
                main_layers.push(PrefixLayerSpec {
                    kind: PrefixLayerKind::Mla,
                    tensors: vec![
                        PrefixTensorSpec {
                            dtype: mla.dtype(),
                            shape: vec![1_i32, 1, cached_len, mla.kv_lora()],
                        },
                        PrefixTensorSpec {
                            dtype: mla.dtype(),
                            shape: vec![1_i32, 1, cached_len, mla.rope()],
                        },
                    ],
                });
            }
        }
    }

    Ok(Some(PagedPrefixKeySpec {
        entry_kind: crate::core::cache::prefix_payload::PrefixEntryKind::WholePrefix,
        model_id: model_id.to_owned(),
        token_ids: token_ids.iter().map(|&id| id as i32).collect(),
        cached_len,
        fingerprint: fingerprint.map(str::to_owned),
        block_size,
        kv_cache_profile,
        main_layers,
        mtp_layers: vec![],
        mtp_last_hidden: None,
        gemma4_drafter_last_hidden: None,
    }))
}

fn remember_kv_cache_profile(
    current: &mut Option<String>,
    next: Option<String>,
) -> anyhow::Result<()> {
    match (current.as_ref(), next) {
        (None, Some(profile)) => {
            *current = Some(profile);
        }
        (Some(current), Some(next)) if current != &next => {
            anyhow::bail!(
                "prefix_key_spec_for_caches: mixed KV cache profiles {current} and {next}"
            );
        }
        _ => {}
    }
    Ok(())
}

pub fn prefix_entry_for_row(
    caches: &[LayerCache],
    row: usize,
) -> anyhow::Result<Option<(PagedPrefixEntry, i32)>> {
    if caches.is_empty() {
        return Ok(None);
    }
    let mut layers = Vec::with_capacity(caches.len());
    let mut cached_len = LayerLen::default();
    for (idx, cache) in caches.iter().enumerate() {
        let (payload, layer_cached_len) = match cache {
            LayerCache::Full(kv) if kv.paged().is_some() => {
                let layer = kv.paged_prefix_layer_for_row_on(row, ())?;
                (
                    PrefixLayerPayload::FullPaged {
                        k_pages: layer.k_pages,
                        v_pages: layer.v_pages,
                    },
                    full_row_len(kv, row, idx)?,
                )
            }
            LayerCache::Full(kv) if kv.turboquant().is_some() => {
                let (layer, packed_cached_len) = kv.turboquant_prefix_layer_for_row_on(row, ())?;
                (
                    PrefixLayerPayload::FullTurboQuantPacked {
                        k_packed: layer.k_packed,
                        k_norms: layer.k_norms,
                        v_packed: layer.v_packed,
                        v_norms: layer.v_norms,
                    },
                    packed_cached_len,
                )
            }
            #[cfg(unix)]
            LayerCache::Flash(cache) => {
                let (tensors, len) = cache.durable_prefix_for_row(row)?;
                (PrefixLayerPayload::Flash { tensors }, len)
            }
            other => unshared_layer_payload(other, row)?,
        };
        cached_len.check(idx, layer_cached_len)?;
        layers.push(payload);
    }
    Ok(Some((entry_from_layers(layers), cached_len.0.unwrap_or(0))))
}

/// The one cached length every layer of a row must agree on.
#[derive(Default)]
struct LayerLen(Option<i32>);

impl LayerLen {
    fn check(&mut self, idx: usize, len: i32) -> anyhow::Result<()> {
        match self.0 {
            Some(expected) if expected != len => anyhow::bail!(
                "prefix_entry_for_row: layer {idx} cached_len {len} != layer0 {expected}"
            ),
            Some(_) => Ok(()),
            None => {
                self.0 = Some(len);
                Ok(())
            }
        }
    }
}

fn full_row_len(kv: &KVCache, row: usize, idx: usize) -> anyhow::Result<i32> {
    kv.offsets().get(row).copied().ok_or_else(|| {
        anyhow::anyhow!("prefix_entry_for_row: full cache row {row} out of range for layer {idx}")
    })
}

fn entry_from_layers(main_layers: Vec<PrefixLayerPayload>) -> PagedPrefixEntry {
    PagedPrefixEntry {
        main_layers,
        mtp_layers: vec![],
        mtp_last_hidden: None,
        gemma4_drafter_last_hidden: None,
    }
}

/// Dense full-attention, linear and MLA layers: never shared, exported whole.
fn unshared_layer_payload(
    cache: &LayerCache,
    row: usize,
) -> anyhow::Result<(PrefixLayerPayload, i32)> {
    Ok(match cache {
        LayerCache::Full(kv) => {
            let (k, v, len) = kv.dense_prefix_layer_for_row_on(row, ())?;
            (PrefixLayerPayload::FullDense { k, v }, len)
        }
        LayerCache::Linear(gd) => {
            let (conv_state, recurrent_state, len) = gd.prefix_state_for_row_on(row, ())?;
            (
                PrefixLayerPayload::Linear {
                    conv_state,
                    recurrent_state,
                },
                len,
            )
        }
        LayerCache::Mla(mla) => {
            let (c_kv, k_pe, len) = mla.prefix_latent_for_row_on(row, ())?;
            (PrefixLayerPayload::Mla { c_kv, k_pe }, len)
        }
        #[cfg(unix)]
        LayerCache::Flash(_) => {
            anyhow::bail!("Flash auxiliary state is not a legacy persistent prefix payload")
        }
    })
}

/// Export `row` for a save that already references segments covering
/// `[0, from_len)`: `[from_len, aligned_len)` of every paged and TurboQuant
/// layer becomes a new segment (`None` when empty), and the private entry
/// holds those layers' tails `[aligned_len, cached_len)` (paged: the partial
/// last page; empty when aligned) plus every other layer whole. Segment and
/// tail arrays are evaluated and each owns its buffer: neither is a view of
/// the live cache or of the other. `aligned_len = floor(cached_len /
/// block_size) * block_size`; `from_len` must be block aligned and at most
/// `aligned_len`; paged layers must use `block_size` pages. Returns
/// `(segment, private, cached_len)`.
pub fn export_prefix_segment_for_row(
    caches: &[LayerCache],
    row: usize,
    from_len: i32,
    block_size: i32,
    layout: PrefixSegmentLayout,
) -> anyhow::Result<Option<(Option<PrefixKvSegment>, PagedPrefixEntry, i32)>> {
    if caches.is_empty() {
        return Ok(None);
    }
    if block_size <= 0 || from_len < 0 || from_len % block_size != 0 {
        anyhow::bail!(
            "export_prefix_segment_for_row: from_len {from_len} must be a non-negative multiple of block_size {block_size}"
        );
    }
    let mut private = Vec::with_capacity(caches.len());
    let mut segment_layers = Vec::new();
    let mut cached_len = LayerLen::default();
    let mut aligned_len = from_len;
    for (idx, cache) in caches.iter().enumerate() {
        let Some(kv) = shareable_full_kv(cache) else {
            let (payload, len) = unshared_layer_payload(cache, row)?;
            cached_len.check(idx, len)?;
            private.push(payload);
            continue;
        };
        let len = full_row_len(kv, row, idx)?;
        cached_len.check(idx, len)?;
        aligned_len = len / block_size * block_size;
        if from_len > aligned_len {
            anyhow::bail!(
                "export_prefix_segment_for_row: from_len {from_len} > aligned_len {aligned_len} (layer {idx})"
            );
        }
        if let Some(paged) = kv.paged() {
            if paged.block_size() != block_size {
                anyhow::bail!(
                    "export_prefix_segment_for_row: layer {idx} page size {} != block_size {block_size}",
                    paged.block_size()
                );
            }
            let (first, aligned_blocks) = (from_len / block_size, aligned_len / block_size);
            let row_blocks = (len + block_size - 1) / block_size;
            if aligned_blocks > first {
                let (k, v) =
                    paged.owned_pages_for_row_on(kv.offsets(), row, first, aligned_blocks, ())?;
                segment_layers.push((idx, PrefixSegmentLayer::Paged { k, v }));
            }
            let (k_pages, v_pages) =
                paged.owned_pages_for_row_on(kv.offsets(), row, aligned_blocks, row_blocks, ())?;
            private.push(PrefixLayerPayload::FullPaged { k_pages, v_pages });
        } else if let Some(tq) = kv.turboquant() {
            if aligned_len > from_len {
                let seg =
                    tq.owned_packed_range_for_row_on(kv.offsets(), row, from_len, aligned_len, ())?;
                segment_layers.push((
                    idx,
                    PrefixSegmentLayer::TurboQuant {
                        k_packed: seg.k_packed,
                        k_norms: seg.k_norms,
                        v_packed: seg.v_packed,
                        v_norms: seg.v_norms,
                    },
                ));
            }
            let tail = tq.owned_packed_range_for_row_on(kv.offsets(), row, aligned_len, len, ())?;
            private.push(PrefixLayerPayload::FullTurboQuantPacked {
                k_packed: tail.k_packed,
                k_norms: tail.k_norms,
                v_packed: tail.v_packed,
                v_norms: tail.v_norms,
            });
        }
    }
    let segment = (!segment_layers.is_empty()).then(|| PrefixKvSegment {
        id: PrefixSegmentId::new(),
        layout,
        start: from_len,
        stop: aligned_len,
        layers: segment_layers,
    });
    Ok(Some((
        segment,
        entry_from_layers(private),
        cached_len.0.unwrap_or(0),
    )))
}

/// Copy `[start, stop)` of `row`'s shareable layers into a segment named
/// `id`. Rebuilds a segment the row's lineage names when the cached copy is
/// gone: the row's K/V over that range came from that segment, so the copy
/// is the same segment. `None` when no layer is shareable.
pub fn export_prefix_segment_range_for_row(
    caches: &[LayerCache],
    row: usize,
    id: PrefixSegmentId,
    start: i32,
    stop: i32,
    block_size: i32,
    layout: PrefixSegmentLayout,
) -> anyhow::Result<Option<PrefixKvSegment>> {
    if block_size <= 0
        || start < 0
        || stop <= start
        || start % block_size != 0
        || stop % block_size != 0
    {
        anyhow::bail!(
            "export_prefix_segment_range_for_row: range [{start}, {stop}) is not block aligned (block_size {block_size})"
        );
    }
    let mut layers = Vec::new();
    for (idx, cache) in caches.iter().enumerate() {
        let Some(kv) = shareable_full_kv(cache) else {
            continue;
        };
        let len = full_row_len(kv, row, idx)?;
        if stop > len {
            anyhow::bail!(
                "export_prefix_segment_range_for_row: stop {stop} > row length {len} (layer {idx})"
            );
        }
        if let Some(paged) = kv.paged() {
            if paged.block_size() != block_size {
                anyhow::bail!(
                    "export_prefix_segment_range_for_row: layer {idx} page size {} != block_size {block_size}",
                    paged.block_size()
                );
            }
            let (k, v) = paged.owned_pages_for_row_on(
                kv.offsets(),
                row,
                start / block_size,
                stop / block_size,
                (),
            )?;
            layers.push((idx, PrefixSegmentLayer::Paged { k, v }));
        } else if let Some(tq) = kv.turboquant() {
            let seg = tq.owned_packed_range_for_row_on(kv.offsets(), row, start, stop, ())?;
            layers.push((
                idx,
                PrefixSegmentLayer::TurboQuant {
                    k_packed: seg.k_packed,
                    k_norms: seg.k_norms,
                    v_packed: seg.v_packed,
                    v_norms: seg.v_norms,
                },
            ));
        }
    }
    Ok((!layers.is_empty()).then(|| PrefixKvSegment {
        id,
        layout,
        start,
        stop,
        layers,
    }))
}

pub fn restore_prefix_entry_for_row(
    caches: &mut [LayerCache],
    entry: &PagedPrefixEntry,
    row: usize,
    cached_len: i32,
) -> anyhow::Result<()> {
    check_prefix_restore_target(caches, entry, &[row], "restore_prefix_entry_for_row")?;
    validate_flash_prefix_payloads(caches, entry, &[row], cached_len)?;
    for (idx, (cache, layer)) in caches.iter_mut().zip(entry.main_layers.iter()).enumerate() {
        restore_prefix_layer_for_rows(
            cache,
            layer,
            &[row],
            cached_len,
            idx,
            "restore_prefix_entry_for_row",
        )?;
    }
    Ok(())
}

pub fn restore_prefix_entry_for_rows(
    caches: &mut [LayerCache],
    entry: &PagedPrefixEntry,
    rows: &[usize],
    cached_len: i32,
) -> anyhow::Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    if rows.len() == 1 {
        return restore_prefix_entry_for_row(caches, entry, rows[0], cached_len);
    }
    check_prefix_restore_target(caches, entry, rows, "restore_prefix_entry_for_rows")?;
    validate_flash_prefix_payloads(caches, entry, rows, cached_len)?;
    for (idx, (cache, layer)) in caches.iter_mut().zip(entry.main_layers.iter()).enumerate() {
        restore_prefix_layer_for_rows(
            cache,
            layer,
            rows,
            cached_len,
            idx,
            "restore_prefix_entry_for_rows",
        )?;
    }
    Ok(())
}

/// A Flash layer restores only from a complete Flash payload that matches it;
/// checked for every Flash layer before any layer is restored.
fn validate_flash_prefix_payloads(
    caches: &[LayerCache],
    entry: &PagedPrefixEntry,
    rows: &[usize],
    cached_len: i32,
) -> anyhow::Result<()> {
    #[cfg(unix)]
    for (cache, layer) in caches.iter().zip(&entry.main_layers) {
        if let LayerCache::Flash(cache) = cache {
            let PrefixLayerPayload::Flash { tensors } = layer else {
                anyhow::bail!("expected complete Flash prefix payload")
            };
            for &row in rows {
                cache.validate_durable_prefix(tensors, row, cached_len)?;
            }
        }
    }
    #[cfg(not(unix))]
    let _ = (caches, entry, rows, cached_len);
    Ok(())
}

/// One stored layer per cache layer, and no row twice.
pub(crate) fn check_prefix_restore_target(
    caches: &[LayerCache],
    entry: &PagedPrefixEntry,
    rows: &[usize],
    what: &str,
) -> anyhow::Result<()> {
    if caches.len() != entry.main_layers.len() {
        anyhow::bail!(
            "{what}: cache layer count {} != stored layers {}",
            caches.len(),
            entry.main_layers.len()
        );
    }
    for (idx, &row) in rows.iter().enumerate() {
        if rows[..idx].contains(&row) {
            anyhow::bail!("{what}: duplicate row {row}");
        }
    }
    Ok(())
}

/// Restore stored layer `idx` into `rows` of its cache layer.
pub(crate) fn restore_prefix_layer_for_rows(
    cache: &mut LayerCache,
    layer: &PrefixLayerPayload,
    rows: &[usize],
    cached_len: i32,
    idx: usize,
    what: &str,
) -> anyhow::Result<()> {
    match (cache, layer) {
        (LayerCache::Full(kv), PrefixLayerPayload::FullDense { k, v }) => {
            for &row in rows {
                kv.restore_dense_prefix_layer_for_row_on(k, v, row, cached_len, ())?;
            }
        }
        (LayerCache::Full(kv), PrefixLayerPayload::FullPaged { k_pages, v_pages }) => {
            let layer = PagedPrefixLayer {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            };
            if let [row] = rows {
                kv.restore_paged_prefix_layer_for_row_on(&layer, *row, cached_len, ())?;
            } else {
                kv.restore_paged_prefix_layer_for_rows_on(&layer, rows, cached_len, ())?;
            }
        }
        (
            LayerCache::Full(kv),
            PrefixLayerPayload::FullTurboQuantPacked {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            },
        ) => {
            let layer = crate::core::cache::TurboQuantPrefixLayer {
                k_packed: k_packed.clone(),
                k_norms: k_norms.clone(),
                v_packed: v_packed.clone(),
                v_norms: v_norms.clone(),
            };
            for &row in rows {
                kv.restore_turboquant_prefix_layer_for_row_on(&layer, row, cached_len, ())?;
            }
        }
        (
            LayerCache::Linear(gd),
            PrefixLayerPayload::Linear {
                conv_state,
                recurrent_state,
            },
        ) => {
            for &row in rows {
                gd.restore_prefix_state_for_row_on(
                    conv_state,
                    recurrent_state,
                    row,
                    cached_len,
                    (),
                )?;
            }
        }
        (LayerCache::Mla(mla), PrefixLayerPayload::Mla { c_kv, k_pe }) => {
            for &row in rows {
                mla.restore_prefix_latent_for_row_on(c_kv, k_pe, row, cached_len, ())?;
            }
        }
        #[cfg(unix)]
        (LayerCache::Flash(cache), PrefixLayerPayload::Flash { tensors }) => {
            for &row in rows {
                cache.restore_durable_prefix(tensors, row, cached_len)?;
            }
        }
        #[cfg(unix)]
        (LayerCache::Flash(_), _) => {
            anyhow::bail!("{what}: layer {idx} expected complete Flash prefix payload")
        }
        (LayerCache::Full(_), _) => {
            anyhow::bail!(
                "{what}: layer {idx} expected FullDense, FullPaged, or FullTurboQuantPacked payload"
            )
        }
        (LayerCache::Linear(_), _) => {
            anyhow::bail!("{what}: layer {idx} expected Linear payload")
        }
        (LayerCache::Mla(_), _) => {
            anyhow::bail!("{what}: layer {idx} expected Mla payload")
        }
    }
    Ok(())
}

/// Key spec of a FullPaged-only cache's first `token_ids.len()` tokens. Each
/// layer keeps its own KV heads, head dims and dtype (Gemma4 mixes sliding
/// and global attention shapes); the layers share one page size, since the
/// prefix is stored as whole pages.
pub fn paged_prefix_key_spec_for_full_caches(
    model_id: &str,
    token_ids: &[u32],
    caches: &[LayerCache],
) -> anyhow::Result<Option<PagedPrefixKeySpec>> {
    if caches.is_empty() {
        return Ok(None);
    }
    let cached_len = i32::try_from(token_ids.len())
        .map_err(|_| anyhow::anyhow!("paged prefix token length exceeds i32"))?;
    let mut block_size: Option<i32> = None;
    let mut main_layers = Vec::with_capacity(caches.len());
    for cache in caches {
        let LayerCache::Full(kv) = cache else {
            return Ok(None);
        };
        let Some(paged) = kv.paged() else {
            return Ok(None);
        };
        let layer_block_size = paged.block_size();
        if *block_size.get_or_insert(layer_block_size) != layer_block_size {
            anyhow::bail!("paged prefix cache requires one page size across full-attention layers");
        }
        let page_count = (cached_len + layer_block_size - 1) / layer_block_size;
        main_layers.push(PrefixLayerSpec {
            kind: PrefixLayerKind::FullPaged,
            tensors: vec![
                PrefixTensorSpec {
                    dtype: kv.dtype(),
                    shape: vec![page_count, kv.n_kv_heads(), layer_block_size, kv.head_dim()],
                },
                PrefixTensorSpec {
                    dtype: kv.dtype(),
                    shape: vec![
                        page_count,
                        kv.n_kv_heads(),
                        layer_block_size,
                        kv.v_head_dim(),
                    ],
                },
            ],
        });
    }
    let Some(block_size) = block_size else {
        return Ok(None);
    };
    Ok(Some(PagedPrefixKeySpec {
        entry_kind: crate::core::cache::prefix_payload::PrefixEntryKind::WholePrefix,
        model_id: model_id.to_owned(),
        token_ids: token_ids.iter().map(|&id| id as i32).collect(),
        cached_len,
        fingerprint: None,
        block_size,
        kv_cache_profile: None,
        main_layers,
        mtp_layers: vec![],
        mtp_last_hidden: None,
        gemma4_drafter_last_hidden: None,
    }))
}

pub fn paged_prefix_layers_for_row(
    caches: &[LayerCache],
    row: usize,
) -> anyhow::Result<Option<PagedPrefixEntry>> {
    if caches.is_empty() {
        return Ok(None);
    }
    let mut layers = Vec::with_capacity(caches.len());
    for cache in caches {
        let LayerCache::Full(kv) = cache else {
            return Ok(None);
        };
        if kv.paged().is_none() {
            return Ok(None);
        }
        let layer = kv.paged_prefix_layer_for_row_on(row, ())?;
        layers.push(PrefixLayerPayload::FullPaged {
            k_pages: layer.k_pages,
            v_pages: layer.v_pages,
        });
    }
    Ok(Some(PagedPrefixEntry {
        main_layers: layers,
        mtp_layers: vec![],
        mtp_last_hidden: None,
        gemma4_drafter_last_hidden: None,
    }))
}

pub fn restore_paged_prefix_layers_for_row(
    caches: &mut [LayerCache],
    entry: &PagedPrefixEntry,
    row: usize,
    prefix_len: i32,
) -> anyhow::Result<()> {
    if caches.len() != entry.main_layers.len() {
        anyhow::bail!(
            "restore_paged_prefix_layers_for_row: cache layer count {} != stored layers {}",
            caches.len(),
            entry.main_layers.len()
        );
    }
    if !entry.mtp_layers.is_empty()
        || entry.mtp_last_hidden.is_some()
        || entry.gemma4_drafter_last_hidden.is_some()
    {
        anyhow::bail!("restore_paged_prefix_layers_for_row: unexpected auxiliary payload");
    }
    for (cache, layer) in caches.iter_mut().zip(entry.main_layers.iter()) {
        let LayerCache::Full(kv) = cache else {
            anyhow::bail!("restore_paged_prefix_layers_for_row: non-Full cache layer");
        };
        let PrefixLayerPayload::FullPaged { k_pages, v_pages } = layer else {
            anyhow::bail!("restore_paged_prefix_layers_for_row: non-Full payload");
        };
        let layer = PagedPrefixLayer {
            k_pages: k_pages.clone(),
            v_pages: v_pages.clone(),
        };
        kv.restore_paged_prefix_layer_for_row_on(&layer, row, prefix_len, ())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{paged_prefix_key_spec_for_full_caches, LayerCache, LayerCacheSnapshot};
    use crate::core::cache::KVCache;
    use mlx::Dtype;

    /// Layers with different KV heads and head dims (Gemma4's sliding and
    /// global attention) get a spec each; differing page sizes are rejected.
    #[test]
    fn full_paged_key_spec_keeps_each_layers_kv_shape() {
        let paged = |heads, head_dim, block_size| {
            LayerCache::Full(
                KVCache::new(1, heads, head_dim, head_dim, Dtype::Bfloat16, 64)
                    .with_paged(block_size, 16)
                    .expect("paged KV"),
            )
        };
        let caches = [paged(8, 256, 16), paged(2, 512, 16)];
        let spec = paged_prefix_key_spec_for_full_caches("gemma4", &[1; 20], &caches)
            .expect("spec")
            .expect("FullPaged-only cache");
        let shapes: Vec<_> = spec
            .main_layers
            .iter()
            .map(|layer| layer.tensors[0].shape.clone())
            .collect();
        assert_eq!(shapes, vec![vec![2, 8, 16, 256], vec![2, 2, 16, 512]]);
        assert_eq!(spec.block_size, 16);

        let mixed_pages = [paged(8, 256, 16), paged(2, 512, 32)];
        assert!(paged_prefix_key_spec_for_full_caches("gemma4", &[1; 20], &mixed_pages).is_err());
    }

    #[test]
    fn dflash2_full_kv_transaction_snapshot_is_offsets_only() {
        let cache = LayerCache::Full(KVCache::new(2, 1, 8, 8, Dtype::Bfloat16, 64));
        let snapshot = cache
            .dflash2_transaction_snapshot()
            .expect("DFlash2 transaction snapshot");
        let LayerCacheSnapshot::Full(snapshot) = snapshot else {
            panic!("expected full-KV snapshot");
        };
        assert_eq!(snapshot.offsets(), &[0, 0]);
        assert!(!snapshot.restores_dense_storage());
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn row_lineage_round_trips_across_full_layers_only() {
        use super::{row_prefix_lineage, set_row_prefix_lineage};
        use crate::core::cache::prefix_segments::{
            PrefixLineage, PrefixSegmentId, PrefixSegmentLayout,
        };
        use crate::core::cache::GatedDeltaCache;
        use mlx::Array;

        let paged = || {
            let mut kv = KVCache::new(2, 1, 2, 2, Dtype::Float32, 64)
                .with_paged(4, 32)
                .expect("paged KV");
            let data: Vec<f32> = (0..2 * 8 * 2).map(|i| i as f32).collect();
            let k: Array = (data.as_slice(), (2_i32, 1_i32, 8_i32, 2_i32))
                .try_into()
                .unwrap();
            kv.update_and_fetch(&k, &k, &[8, 8]).expect("fill");
            LayerCache::Full(kv)
        };
        let mut turboquant = KVCache::new(2, 2, 8, 8, Dtype::Float32, 16)
            .with_step(16)
            .with_turboquant(crate::core::cache::TurboQuantKVBits::K4V4)
            .expect("TurboQuant KV");
        let data: Vec<f32> = (0..2 * 2 * 8 * 8).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Array = (data.as_slice(), (2_i32, 2_i32, 8_i32, 8_i32))
            .try_into()
            .unwrap();
        turboquant
            .update_and_fetch(&k, &k, &[8, 8])
            .expect("fill TQ");
        let mut caches = vec![
            LayerCache::Linear(
                GatedDeltaCache::new_with_cap(2, 4, 8, 1, 4, 4, Dtype::Float32, 16)
                    .expect("linear cache"),
            ),
            LayerCache::Full(KVCache::new(2, 1, 2, 2, Dtype::Float32, 64)),
            paged(),
            paged(),
            LayerCache::Full(turboquant),
        ];
        assert_eq!(row_prefix_lineage(&caches, 1), None);

        let lineage = PrefixLineage {
            layout: PrefixSegmentLayout("test-layout".to_owned()),
            refs: vec![crate::core::cache::prefix_segments::PrefixLineageRef {
                id: PrefixSegmentId::new(),
                start: 0,
                stop: 8,
                segment_stop: 8,
            }],
        };
        set_row_prefix_lineage(&mut caches, 1, Some(lineage.clone()));
        assert_eq!(row_prefix_lineage(&caches, 1), Some(lineage.clone()));
        assert_eq!(row_prefix_lineage(&caches, 0), None);
        let per_layer: Vec<_> = caches
            .iter()
            .map(|layer| match layer {
                LayerCache::Full(kv) => Some(kv.prefix_lineage(1).cloned()),
                _ => None,
            })
            .collect();
        assert_eq!(
            per_layer,
            vec![
                None,
                Some(None),
                Some(Some(lineage.clone())),
                Some(Some(lineage.clone())),
                Some(Some(lineage))
            ],
            "only shareable (paged / TurboQuant) full-attention layers carry the record"
        );

        set_row_prefix_lineage(&mut caches, 1, None);
        assert_eq!(row_prefix_lineage(&caches, 1), None);
    }
}
