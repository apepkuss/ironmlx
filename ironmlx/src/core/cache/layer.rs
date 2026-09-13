//! Model-side cache variants, snapshots and prefix conversion operations.

use crate::core::cache::prefix_payload::{
    PagedPrefixEntry, PagedPrefixKeySpec, PagedPrefixLayer, PrefixLayerKind, PrefixLayerPayload,
    PrefixLayerSpec, PrefixTensorSpec,
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
}

/// Per-layer cache checkpoint used by speculative decoding rollback.
#[doc(hidden)]
pub enum LayerCacheSnapshot {
    Full(KVCacheSnapshot),
    Linear(GatedDeltaCacheSnapshot),
    Mla(MlaLatentCacheSnapshot),
}

impl LayerCache {
    pub(crate) fn begin_speculative_prefix_capture(&mut self) -> anyhow::Result<()> {
        match self {
            LayerCache::Full(_) => Ok(()),
            LayerCache::Linear(cache) => cache.begin_speculative_prefix_capture(),
            LayerCache::Mla(_) => {
                anyhow::bail!("speculative prefix capture does not support MLA cache")
            }
        }
    }

    pub(crate) fn discard_speculative_prefix_capture(&mut self) {
        if let LayerCache::Linear(cache) = self {
            cache.discard_speculative_prefix_capture();
        }
    }

    pub fn enable_turboquant(&mut self, bits: TurboQuantKVBits) -> anyhow::Result<()> {
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
        }
    }

    pub fn restore_configured_paged_hot_window(&mut self) -> bool {
        match self {
            LayerCache::Full(kv) => kv.restore_configured_paged_hot_window(),
            LayerCache::Linear(_) | LayerCache::Mla(_) => false,
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
        }
    }

    /// Capture a lightweight rollback checkpoint for this layer cache.
    pub fn snapshot(&self) -> LayerCacheSnapshot {
        match self {
            LayerCache::Full(kv) => LayerCacheSnapshot::Full(kv.snapshot()),
            LayerCache::Linear(gd) => LayerCacheSnapshot::Linear(gd.snapshot()),
            LayerCache::Mla(c) => LayerCacheSnapshot::Mla(c.snapshot()),
        }
    }

    /// Capture an offsets-only checkpoint for append-only Full-KV verify.
    pub fn append_snapshot(&self) -> anyhow::Result<LayerCacheSnapshot> {
        match self {
            LayerCache::Full(kv) => Ok(LayerCacheSnapshot::Full(kv.append_snapshot())),
            LayerCache::Linear(_) => {
                anyhow::bail!("LayerCache::append_snapshot does not support Linear cache")
            }
            LayerCache::Mla(_) => {
                anyhow::bail!("LayerCache::append_snapshot does not support MLA cache")
            }
        }
    }

    /// Restore this layer cache from a matching checkpoint.
    pub fn restore(&mut self, snapshot: &LayerCacheSnapshot) -> anyhow::Result<()> {
        match (self, snapshot) {
            (LayerCache::Full(kv), LayerCacheSnapshot::Full(s)) => kv.restore(s),
            (LayerCache::Linear(gd), LayerCacheSnapshot::Linear(s)) => gd.restore(s),
            (LayerCache::Mla(c), LayerCacheSnapshot::Mla(s)) => c.restore(s),
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

pub(crate) fn adopt_layer_cache_rows(
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
    let mut cached_len: Option<i32> = None;
    for (idx, cache) in caches.iter().enumerate() {
        let (payload, layer_cached_len) = match cache {
            LayerCache::Full(kv) => {
                let layer_cached_len = *kv.offsets().get(row).ok_or_else(|| {
                    anyhow::anyhow!(
                        "prefix_entry_for_row: full cache row {} out of range for layer {}",
                        row,
                        idx
                    )
                })?;
                if kv.paged().is_some() {
                    let layer = kv.paged_prefix_layer_for_row_on(row, ())?;
                    (
                        PrefixLayerPayload::FullPaged {
                            k_pages: layer.k_pages,
                            v_pages: layer.v_pages,
                        },
                        layer_cached_len,
                    )
                } else if kv.turboquant().is_some() {
                    let (layer, packed_cached_len) =
                        kv.turboquant_prefix_layer_for_row_on(row, ())?;
                    (
                        PrefixLayerPayload::FullTurboQuantPacked {
                            k_packed: layer.k_packed,
                            k_norms: layer.k_norms,
                            v_packed: layer.v_packed,
                            v_norms: layer.v_norms,
                        },
                        packed_cached_len,
                    )
                } else {
                    let (k, v, layer_cached_len) = kv.dense_prefix_layer_for_row_on(row, ())?;
                    (PrefixLayerPayload::FullDense { k, v }, layer_cached_len)
                }
            }
            LayerCache::Linear(gd) => {
                let (conv_state, recurrent_state, layer_cached_len) =
                    gd.prefix_state_for_row_on(row, ())?;
                (
                    PrefixLayerPayload::Linear {
                        conv_state,
                        recurrent_state,
                    },
                    layer_cached_len,
                )
            }
            LayerCache::Mla(mla) => {
                let (c_kv, k_pe, layer_cached_len) = mla.prefix_latent_for_row_on(row, ())?;
                (PrefixLayerPayload::Mla { c_kv, k_pe }, layer_cached_len)
            }
        };
        if let Some(expected) = cached_len {
            if layer_cached_len != expected {
                anyhow::bail!(
                    "prefix_entry_for_row: layer {idx} cached_len {layer_cached_len} != layer0 {expected}"
                );
            }
        } else {
            cached_len = Some(layer_cached_len);
        }
        layers.push(payload);
    }

    Ok(Some((
        PagedPrefixEntry {
            main_layers: layers,
            mtp_layers: vec![],
            mtp_last_hidden: None,
            gemma4_drafter_last_hidden: None,
        },
        cached_len.unwrap_or(0),
    )))
}

pub fn restore_prefix_entry_for_row(
    caches: &mut [LayerCache],
    entry: &PagedPrefixEntry,
    row: usize,
    cached_len: i32,
) -> anyhow::Result<()> {
    if caches.len() != entry.main_layers.len() {
        anyhow::bail!(
            "restore_prefix_entry_for_row: cache layer count {} != stored layers {}",
            caches.len(),
            entry.main_layers.len()
        );
    }
    for (idx, (cache, layer)) in caches.iter_mut().zip(entry.main_layers.iter()).enumerate() {
        match (cache, layer) {
            (LayerCache::Full(kv), PrefixLayerPayload::FullDense { k, v }) => {
                kv.restore_dense_prefix_layer_for_row_on(k, v, row, cached_len, ())?;
            }
            (LayerCache::Full(kv), PrefixLayerPayload::FullPaged { k_pages, v_pages }) => {
                let layer = PagedPrefixLayer {
                    k_pages: k_pages.clone(),
                    v_pages: v_pages.clone(),
                };
                kv.restore_paged_prefix_layer_for_row_on(&layer, row, cached_len, ())?;
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
                kv.restore_turboquant_prefix_layer_for_row_on(&layer, row, cached_len, ())?;
            }
            (
                LayerCache::Linear(gd),
                PrefixLayerPayload::Linear {
                    conv_state,
                    recurrent_state,
                },
            ) => {
                gd.restore_prefix_state_for_row_on(
                    conv_state,
                    recurrent_state,
                    row,
                    cached_len,
                    (),
                )?;
            }
            (LayerCache::Mla(mla), PrefixLayerPayload::Mla { c_kv, k_pe }) => {
                mla.restore_prefix_latent_for_row_on(c_kv, k_pe, row, cached_len, ())?;
            }
            (LayerCache::Full(_), _) => {
                anyhow::bail!(
                    "restore_prefix_entry_for_row: layer {idx} expected FullDense, FullPaged, or FullTurboQuantPacked payload"
                )
            }
            (LayerCache::Linear(_), _) => {
                anyhow::bail!("restore_prefix_entry_for_row: layer {idx} expected Linear payload")
            }
            (LayerCache::Mla(_), _) => {
                anyhow::bail!("restore_prefix_entry_for_row: layer {idx} expected Mla payload")
            }
        }
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
    if caches.len() != entry.main_layers.len() {
        anyhow::bail!(
            "restore_prefix_entry_for_rows: cache layer count {} != stored layers {}",
            caches.len(),
            entry.main_layers.len()
        );
    }
    for (idx, &row) in rows.iter().enumerate() {
        if rows[..idx].contains(&row) {
            anyhow::bail!("restore_prefix_entry_for_rows: duplicate row {row}");
        }
    }
    for (idx, (cache, layer)) in caches.iter_mut().zip(entry.main_layers.iter()).enumerate() {
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
                kv.restore_paged_prefix_layer_for_rows_on(&layer, rows, cached_len, ())?;
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
            (LayerCache::Full(_), _) => {
                anyhow::bail!(
                    "restore_prefix_entry_for_rows: layer {idx} expected FullDense, FullPaged, or FullTurboQuantPacked payload"
                )
            }
            (LayerCache::Linear(_), _) => {
                anyhow::bail!("restore_prefix_entry_for_rows: layer {idx} expected Linear payload")
            }
            (LayerCache::Mla(_), _) => {
                anyhow::bail!("restore_prefix_entry_for_rows: layer {idx} expected Mla payload")
            }
        }
    }
    Ok(())
}

pub fn paged_prefix_key_spec_for_full_caches(
    model_id: &str,
    token_ids: &[u32],
    caches: &[LayerCache],
) -> anyhow::Result<Option<PagedPrefixKeySpec>> {
    if caches.is_empty() {
        return Ok(None);
    }
    let mut first: Option<&KVCache> = None;
    for cache in caches {
        let LayerCache::Full(kv) = cache else {
            return Ok(None);
        };
        if kv.paged().is_none() {
            return Ok(None);
        }
        if let Some(base) = first {
            if kv.n_kv_heads() != base.n_kv_heads()
                || kv.head_dim() != base.head_dim()
                || kv.v_head_dim() != base.v_head_dim()
                || kv.dtype() != base.dtype()
                || kv.paged().map(|p| p.block_size()) != base.paged().map(|p| p.block_size())
            {
                anyhow::bail!("paged prefix cache requires uniform full-attention KV layout");
            }
        } else {
            first = Some(kv);
        }
    }
    let Some(base) = first else {
        return Ok(None);
    };
    let paged = base
        .paged()
        .expect("paged checked above for every full-attention cache");
    let cached_len = i32::try_from(token_ids.len())
        .map_err(|_| anyhow::anyhow!("paged prefix token length exceeds i32"))?;
    let page_count = (cached_len + paged.block_size() - 1) / paged.block_size();
    let main_layers = (0..caches.len())
        .map(|_| PrefixLayerSpec {
            kind: PrefixLayerKind::FullPaged,
            tensors: vec![
                PrefixTensorSpec {
                    dtype: base.dtype(),
                    shape: vec![
                        page_count,
                        base.n_kv_heads(),
                        paged.block_size(),
                        base.head_dim(),
                    ],
                },
                PrefixTensorSpec {
                    dtype: base.dtype(),
                    shape: vec![
                        page_count,
                        base.n_kv_heads(),
                        paged.block_size(),
                        base.v_head_dim(),
                    ],
                },
            ],
        })
        .collect();
    Ok(Some(PagedPrefixKeySpec {
        entry_kind: crate::core::cache::prefix_payload::PrefixEntryKind::WholePrefix,
        model_id: model_id.to_owned(),
        token_ids: token_ids.iter().map(|&id| id as i32).collect(),
        cached_len,
        fingerprint: None,
        block_size: paged.block_size(),
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
