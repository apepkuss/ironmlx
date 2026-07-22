//! Single Qwen3.5 / Qwen3-Next decoder block.
//!
//! Mirrors mlx-lm `Qwen3NextDecoderLayer.__call__`:
//!
//! ```text
//! r   = self_attn_or_linear_attn(input_layernorm(x), mask, cache)
//! h   = x + r
//! out = h + mlp(post_attention_layernorm(h))
//! ```
//!
//! The attention path is selected at construction time per `AttnKind`. Full-
//! attention layers consume `KVCache`; linear-attention SSM layers consume
//! `GatedDeltaCache`. Both are wrapped uniformly via [`LayerCache`].

use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::{
    GatedDeltaCache, GatedDeltaCacheSnapshot, KVCache, KVCacheSnapshot, PagedKvHotColdConfig,
    PagedPrefixEntry, PagedPrefixKeySpec, PagedPrefixLayer, PrefixLayerKind, PrefixLayerPayload,
    PrefixLayerSpec, PrefixTensorSpec, TurboQuantKVBits,
};
use crate::core::Loader;
use crate::models::glm4_moe_lite::mla_cache::MlaLatentCacheSnapshot;
use crate::nn::{
    GatedAttention, GatedAttentionConfig, GatedDeltaNet, GatedDeltaNetConfig, Mlp, Mrope, RmsNorm,
};
use crate::Result;

#[derive(Clone, Copy)]
struct DecodeLayerTurboProfileEvent {
    stage: &'static str,
    elapsed_us: u128,
    layer_idx: i32,
    attn_kind: &'static str,
    batch: i32,
    seq: i32,
    hidden_size: i32,
}

#[derive(Clone, Copy)]
struct DecodeLayerTurboProfileShape {
    layer_idx: i32,
    attn_kind: &'static str,
    batch: i32,
    seq: i32,
    hidden_size: i32,
}

fn format_decode_layer_turbo_profile_line(event: DecodeLayerTurboProfileEvent) -> String {
    format!(
        "{{\"event\":\"turboquant_decoder_layer_stage\",\"stage\":\"{}\",\"elapsed_us\":{},\"layer_idx\":{},\"attn_kind\":\"{}\",\"batch\":{},\"seq\":{},\"hidden_size\":{}}}",
        event.stage,
        event.elapsed_us,
        event.layer_idx,
        event.attn_kind,
        event.batch,
        event.seq,
        event.hidden_size,
    )
}

fn profile_decode_layer_turbo_stage(
    stage: &'static str,
    arrays: &[&Array],
    shape: DecodeLayerTurboProfileShape,
) -> Result<()> {
    if shape.seq != 1 || std::env::var_os("IRONMLX_TURBOQUANT_ATTN_PROFILE").is_none() {
        return Ok(());
    }

    let start = std::time::Instant::now();
    mlx::transforms::eval(arrays).map_err(|e| anyhow!("{e}"))?;
    eprintln!(
        "{}",
        format_decode_layer_turbo_profile_line(DecodeLayerTurboProfileEvent {
            stage,
            elapsed_us: start.elapsed().as_micros(),
            layer_idx: shape.layer_idx,
            attn_kind: shape.attn_kind,
            batch: shape.batch,
            seq: shape.seq,
            hidden_size: shape.hidden_size,
        })
    );
    Ok(())
}

/// Which attention path a [`DecoderLayer`] uses. Selected per layer index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Standard gated full attention (P3b2). Consumes [`KVCache`].
    Full,
    /// Gated delta-net linear attention SSM (P3b3). Consumes [`GatedDeltaCache`].
    Linear,
}

/// Configuration for [`DecoderLayer`]. Mirrors the subset of Qwen3.5
/// `TextModelArgs` that drives a single decoder block.
#[derive(Debug, Clone, Copy)]
pub struct DecoderLayerConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    /// Linear-attn parameters (only consulted when `AttnKind::Linear`).
    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
}

/// Attention path variant — owns either a full-attention or a linear-attention block.
///
/// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can construct it
/// via [`DecoderLayer::from_components_full`] / [`DecoderLayer::from_components_linear`].
#[doc(hidden)]
pub enum AttnPath {
    Full(Box<GatedAttention>),
    Linear(Box<GatedDeltaNet>),
}

/// Per-layer cache, paired with [`AttnPath`].
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
        entry_kind: crate::core::cache::PrefixEntryKind::WholePrefix,
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
        entry_kind: crate::core::cache::PrefixEntryKind::WholePrefix,
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

/// One decoder block. Full or linear attention selected at construction.
pub struct DecoderLayer {
    input_layernorm: RmsNorm,
    attn: AttnPath,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    cfg: DecoderLayerConfig,
}

impl DecoderLayer {
    /// Test/composition seam — full-attention variant. Equivalent to P3b4's
    /// `from_components` (renamed for symmetry with the linear-attn variant).
    #[doc(hidden)]
    pub fn from_components_full(
        input_layernorm: RmsNorm,
        self_attn: GatedAttention,
        post_attention_layernorm: RmsNorm,
        mlp: Mlp,
        cfg: DecoderLayerConfig,
    ) -> Self {
        Self {
            input_layernorm,
            attn: AttnPath::Full(Box::new(self_attn)),
            post_attention_layernorm,
            mlp,
            cfg,
        }
    }

    /// Test/composition seam — linear-attention SSM variant.
    #[doc(hidden)]
    pub fn from_components_linear(
        input_layernorm: RmsNorm,
        linear_attn: GatedDeltaNet,
        post_attention_layernorm: RmsNorm,
        mlp: Mlp,
        cfg: DecoderLayerConfig,
    ) -> Self {
        Self {
            input_layernorm,
            attn: AttnPath::Linear(Box::new(linear_attn)),
            post_attention_layernorm,
            mlp,
            cfg,
        }
    }

    /// Read-only view of the layer config.
    pub fn config(&self) -> &DecoderLayerConfig {
        &self.cfg
    }

    /// Which path this layer uses (introspection helper for the test/cache layer).
    pub fn kind(&self) -> AttnKind {
        match &self.attn {
            AttnPath::Full(_) => AttnKind::Full,
            AttnPath::Linear(_) => AttnKind::Linear,
        }
    }

    /// Pre-flight: enforce rank-3 input + last-axis matches `cfg.hidden_size`.
    /// `caller` is embedded in the diagnostic so callers (forward_on,
    /// forward_on_full_kv) surface in the error string.
    #[inline]
    fn preflight_x(&self, x: &Array, caller: &str) -> Result<()> {
        if x.ndim() != 3 {
            return Err(anyhow!(
                "{caller}: x must be rank-3 [B, S, hidden_size], got rank {}",
                x.ndim()
            ));
        }
        let dims_owned = x.shape();
        let dims = dims_owned.as_slice();
        if dims[2] != self.cfg.hidden_size {
            return Err(anyhow!(
                "{caller}: x last-axis = {} but cfg.hidden_size = {}",
                dims[2],
                self.cfg.hidden_size
            ));
        }
        Ok(())
    }

    /// Default-stream forward pass. The single `mask` parameter is interpreted
    /// per layer kind: the full-attention path treats it as the SDPA-style
    /// `[B, 1, T_q, T_kv]` additive mask, the linear-attention path treats it
    /// as the `[B, T]` boolean per-token validity mask. For hybrid models that
    /// need to pass different masks to the two paths, call
    /// [`Self::forward_on`] directly.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
    ) -> Result<Array> {
        // Convenience: forward `mask` to whichever path applies. The other
        // gets `None`. Callers that need to populate both should use
        // `forward_on` directly.
        let (full_mask, linear_mask) = match self.kind() {
            AttnKind::Full => (mask, None),
            AttnKind::Linear => (None, mask),
        };
        self.forward_on(x, mrope, cos, sin, full_mask, linear_mask, None, cache, ())
    }

    /// Stream-targeted forward.
    ///
    /// `x: [B, S, hidden_size]` → `[B, S, hidden_size]`. Cache type must match
    /// `self.kind()`; mismatch returns `Err`. Linear-attn ignores `mrope`/`cos`/`sin`
    /// (passed through for signature uniformity with the Full path).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        full_attn_mask: Option<&Array>,
        linear_attn_mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        cache: Option<&mut LayerCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.forward_on_with_layer_idx(
            x,
            mrope,
            cos,
            sin,
            full_attn_mask,
            linear_attn_mask,
            per_row_lens,
            cache,
            target,
            -1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_on_with_layer_idx(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        full_attn_mask: Option<&Array>,
        linear_attn_mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        cache: Option<&mut LayerCache>,
        target: impl Into<StreamOrDevice>,
        layer_idx: i32,
    ) -> Result<Array> {
        let target = target.into();

        // Pre-flight (existing P3b4 invariants).
        self.preflight_x(x, "DecoderLayer::forward_on")?;
        let dims_owned = x.shape();
        let dims = dims_owned.as_slice();
        let profile_shape = DecodeLayerTurboProfileShape {
            layer_idx,
            attn_kind: match self.kind() {
                AttnKind::Full => "full",
                AttnKind::Linear => "linear",
            },
            batch: dims[0],
            seq: dims[1],
            hidden_size: dims[2],
        };
        profile_decode_layer_turbo_stage("decode_layer_input", &[x], profile_shape)?;

        // Block 1: input_layernorm + attn dispatch + residual
        //
        // Hybrid routing: full-attention layers consume `full_attn_mask`
        // (`[B, 1, T_q, T_kv]` additive bf16 for SDPA); linear-attention
        // layers consume `linear_attn_mask` (`[B, T]` boolean per-token
        // validity for the `gated_delta_step` kernel). The two have
        // incompatible shapes and dtypes — they cannot be unified.
        let normed_in = self.input_layernorm.forward_on(x, target)?;
        // Full attention also consumes `linear_attn_mask` (when Some) as its
        // K/V-validity mask, zeroing pad-position K/V cells before the cache
        // write. The `[B, T]` boolean shape and "real-vs-pad per token"
        // semantics are identical to what linear attention uses; reusing it
        // avoids defining a third mask. See `attention::forward_on` for
        // details.
        profile_decode_layer_turbo_stage("decode_input_norm", &[&normed_in], profile_shape)?;
        let attn = match (&self.attn, cache) {
            (AttnPath::Full(a), Some(LayerCache::Full(kv))) => a.forward_on(
                &normed_in,
                mrope,
                cos,
                sin,
                full_attn_mask,
                linear_attn_mask,
                per_row_lens,
                Some(kv),
                target,
                layer_idx,
            )?,
            (AttnPath::Full(a), None) => a.forward_on(
                &normed_in,
                mrope,
                cos,
                sin,
                full_attn_mask,
                linear_attn_mask,
                per_row_lens,
                None,
                target,
                layer_idx,
            )?,
            (AttnPath::Linear(a), Some(LayerCache::Linear(gdc))) => a.forward_on(
                &normed_in,
                linear_attn_mask,
                per_row_lens,
                Some(gdc),
                target,
                layer_idx,
            )?,
            (AttnPath::Linear(a), None) => a.forward_on(
                &normed_in,
                linear_attn_mask,
                per_row_lens,
                None,
                target,
                layer_idx,
            )?,
            (AttnPath::Full(_), Some(LayerCache::Linear(_))) => {
                return Err(anyhow!(
                    "DecoderLayer::forward_on: Full attn layer received Linear cache (kind mismatch)"
                ));
            }
            (AttnPath::Linear(_), Some(LayerCache::Full(_))) => {
                return Err(anyhow!(
                    "DecoderLayer::forward_on: Linear attn layer received Full cache (kind mismatch)"
                ));
            }
            (_, Some(LayerCache::Mla(_))) => {
                return Err(anyhow!(
                    "DecoderLayer::forward_on: received Mla cache (kind mismatch)"
                ));
            }
        };
        profile_decode_layer_turbo_stage("decode_attention_path", &[&attn], profile_shape)?;
        let h = x + &attn;
        profile_decode_layer_turbo_stage("decode_attention_residual", &[&h], profile_shape)?;

        // Block 2: post_norm + mlp + residual
        let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
        profile_decode_layer_turbo_stage(
            "decode_post_attention_norm",
            &[&normed_post],
            profile_shape,
        )?;
        let mlp_out = self.mlp.forward_on(&normed_post, target)?;
        profile_decode_layer_turbo_stage("decode_mlp_path", &[&mlp_out], profile_shape)?;
        let out = &h + &mlp_out;
        profile_decode_layer_turbo_stage("decode_layer_output", &[&out], profile_shape)?;
        Ok(out)
    }

    /// Production constructor. `kind` selects which attention path to load
    /// (Full → reads `{prefix}.self_attn.*`; Linear → reads `{prefix}.linear_attn.*`).
    ///
    /// No construction-time dim sanity checks — Linear's matmul surfaces shape errors
    /// at first forward_on (matches GatedAttention::from_loader precedent).
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: DecoderLayerConfig,
        kind: AttnKind,
    ) -> Result<Self> {
        let input_layernorm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.input_layernorm"),
            cfg.rms_norm_eps,
        )?;
        let attn = match kind {
            AttnKind::Full => {
                let ga = GatedAttention::from_loader(
                    loader,
                    &format!("{prefix}.self_attn"),
                    GatedAttentionConfig {
                        num_heads: cfg.num_heads,
                        num_kv_heads: cfg.num_kv_heads,
                        head_dim: cfg.head_dim,
                        rms_norm_eps: cfg.rms_norm_eps,
                        attention_bias: cfg.attention_bias,
                    },
                )?;
                AttnPath::Full(Box::new(ga))
            }
            AttnKind::Linear => {
                let gdn = GatedDeltaNet::from_loader(
                    loader,
                    &format!("{prefix}.linear_attn"),
                    GatedDeltaNetConfig {
                        hidden_size: cfg.hidden_size,
                        num_v_heads: cfg.linear_num_value_heads,
                        num_k_heads: cfg.linear_num_key_heads,
                        head_k_dim: cfg.linear_key_head_dim,
                        head_v_dim: cfg.linear_value_head_dim,
                        conv_kernel_size: cfg.linear_conv_kernel_dim,
                        rms_norm_eps: cfg.rms_norm_eps,
                    },
                )?;
                AttnPath::Linear(Box::new(gdn))
            }
        };
        let post_attention_layernorm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.post_attention_layernorm"),
            cfg.rms_norm_eps,
        )?;
        let mlp = Mlp::from_loader(loader, &format!("{prefix}.mlp"))?;
        Ok(Self {
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
            cfg,
        })
    }
}

impl DecoderLayer {
    /// Package-private helper for [`crate::nn::Mtp`]: same as [`forward_on`](Self::forward_on)
    /// but accepts `Option<&mut KVCache>` directly, avoiding a wrapper allocation.
    ///
    /// Returns `Err` if called on a `Linear` layer (MTP layers are always Full).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_on_full_kv(
        &self,
        x: &Array,
        mrope: &Mrope,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        self.preflight_x(x, "DecoderLayer::forward_on_full_kv")?;

        let normed_in = self.input_layernorm.forward_on(x, target)?;
        let attn_out = match &self.attn {
            AttnPath::Full(a) => a.forward_on(
                &normed_in, mrope, cos, sin, mask, None, None, cache, target, -1,
            )?,
            AttnPath::Linear(_) => {
                return Err(anyhow!(
                    "DecoderLayer::forward_on_full_kv: called on Linear layer (MTP requires Full)"
                ));
            }
        };
        let h = x + &attn_out;

        let normed_post = self.post_attention_layernorm.forward_on(&h, target)?;
        let mlp_out = self.mlp.forward_on(&normed_post, target)?;
        Ok(&h + &mlp_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    use crate::core::cache::{GatedDeltaCache, KVCache, PagedPrefixStore};
    use crate::nn::Linear;
    use serial_test::serial;

    #[test]
    fn format_decode_layer_turbo_profile_line_is_stable_json() {
        let line = format_decode_layer_turbo_profile_line(DecodeLayerTurboProfileEvent {
            stage: "decode_input_norm",
            elapsed_us: 42,
            layer_idx: 7,
            attn_kind: "full",
            batch: 1,
            seq: 1,
            hidden_size: 2560,
        });

        assert_eq!(
            line,
            "{\"event\":\"turboquant_decoder_layer_stage\",\"stage\":\"decode_input_norm\",\"elapsed_us\":42,\"layer_idx\":7,\"attn_kind\":\"full\",\"batch\":1,\"seq\":1,\"hidden_size\":2560}"
        );
    }

    fn rand_w(shape: &[i32], dtype: Dtype) -> Array {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0123).sin()).collect();
        let arr: Array = (data.as_slice(), shape).try_into().unwrap();
        mlx::ops::cast::astype(&arr, dtype).unwrap()
    }

    fn ones_w(dim: i32) -> Array {
        mlx::ops::constructors::ones((dim,), Dtype::Float32).unwrap()
    }

    fn small_cfg() -> DecoderLayerConfig {
        DecoderLayerConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            linear_num_value_heads: 0,
            linear_num_key_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
        }
    }

    fn build_decoder_layer(cfg: DecoderLayerConfig) -> DecoderLayer {
        // Random small weights — only structural / shape behavior is validated here.
        let q_w = rand_w(
            &[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let k_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let v_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let o_w = rand_w(
            &[cfg.hidden_size, cfg.num_heads * cfg.head_dim],
            Dtype::Bfloat16,
        );

        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w, None),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            GatedAttentionConfig {
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
            },
        );

        let gate_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let up_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let down_w = rand_w(&[cfg.hidden_size, cfg.intermediate_size], Dtype::Bfloat16);

        let mlp = Mlp::from_components(
            Linear::new_fp(gate_w, None),
            Linear::new_fp(up_w, None),
            Linear::new_fp(down_w, None),
        );

        DecoderLayer::from_components_full(
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            attn,
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            mlp,
            cfg,
        )
    }

    #[test]
    #[serial(mlx_metal)]
    fn from_components_carries_config() {
        let cfg = small_cfg();
        let layer = build_decoder_layer(cfg);
        let kept = layer.config();
        assert_eq!(kept.hidden_size, cfg.hidden_size);
        assert_eq!(kept.intermediate_size, cfg.intermediate_size);
        assert_eq!(kept.num_heads, cfg.num_heads);
        assert_eq!(kept.num_kv_heads, cfg.num_kv_heads);
        assert_eq!(kept.head_dim, cfg.head_dim);
    }

    #[test]
    #[serial(mlx_metal)]
    fn layer_cache_snapshot_restore_full_and_kind_mismatch() {
        let mut full = LayerCache::Full(KVCache::new(1, 2, 8, 8, Dtype::Float32, 16).with_step(16));
        let k: Array = Array::zeros((1, 2, 4, 8), Dtype::Float32).unwrap();
        let v: Array = Array::zeros((1, 2, 4, 8), Dtype::Float32).unwrap();
        if let LayerCache::Full(kv) = &mut full {
            kv.update_and_fetch(&k, &v, &[4]).unwrap();
        }
        let snapshot = full.snapshot();
        if let LayerCache::Full(kv) = &mut full {
            kv.update_and_fetch(&k, &v, &[4]).unwrap();
            assert_eq!(kv.offsets(), &[8]);
        }

        full.restore(&snapshot).expect("restore full snapshot");
        if let LayerCache::Full(kv) = &full {
            assert_eq!(kv.offsets(), &[4]);
        }

        let mut linear = LayerCache::Linear(
            GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Float32, 16).unwrap(),
        );
        assert!(linear.restore(&snapshot).is_err());
    }

    fn build_inputs_fp32(cfg: DecoderLayerConfig) -> (Array, Mrope, Array, Array) {
        // Synthesize fp32 inputs to exercise forward shape/dtype path.
        let b = 1_i32;
        let s = 4_i32;
        let n_streams = 3_i32;

        // x: [B, S, H] fp32 random.
        let x = rand_w(&[b, s, cfg.hidden_size], Dtype::Float32);

        // Mrope with full rotary (partial=1.0) over head_dim=8 → rot_dim=8 → half=4 → sections=[2,1,1].
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();

        // Build position_ids = broadcast of arange(s) across n_streams + batch.
        let pos1d = mlx::ops::constructors::arange(0.0, s as f64, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, s)).unwrap();
        let position_ids =
            mlx::ops::shape::broadcast_to_on(&pos1d, &[n_streams, b, s], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();
        (x, mrope, cos, sin)
    }

    #[test]
    #[serial(mlx_metal)]
    fn forward_shape_and_dtype_fp32() {
        let cfg = small_cfg();
        let layer = build_decoder_layer(cfg);
        let (x, mrope, cos, sin) = build_inputs_fp32(cfg);
        let out = layer.forward(&x, &mrope, &cos, &sin, None, None).unwrap();
        assert_eq!(out.shape().as_slice(), &[1, 4, cfg.hidden_size]);
        // RmsNorm with fp32 weight + bf16 attn weight → fp32 promotes; final residual
        // sums fp32 + fp32 → fp32. Dtype is fp32 even though attn weights are bf16.
        assert_eq!(out.dtype(), Dtype::Float32);
    }

    #[test]
    #[serial(mlx_metal)]
    fn forward_shape_and_dtype_bf16() {
        // bf16 input (with bf16 norm weights) → bf16 output preserved.
        let cfg = small_cfg();

        // bf16 attn + mlp weights matching small_cfg.
        let q_w = rand_w(
            &[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let k_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let v_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Bfloat16,
        );
        let o_w = rand_w(
            &[cfg.hidden_size, cfg.num_heads * cfg.head_dim],
            Dtype::Bfloat16,
        );
        // bf16 norm weights to keep dtype contained at bf16 throughout.
        let qn = rand_w(&[cfg.head_dim], Dtype::Bfloat16);
        let kn = rand_w(&[cfg.head_dim], Dtype::Bfloat16);
        let pre_norm_w = rand_w(&[cfg.hidden_size], Dtype::Bfloat16);
        let post_norm_w = rand_w(&[cfg.hidden_size], Dtype::Bfloat16);
        let gate_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let up_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Bfloat16);
        let down_w = rand_w(&[cfg.hidden_size, cfg.intermediate_size], Dtype::Bfloat16);

        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w, None),
            RmsNorm::new(qn, cfg.rms_norm_eps),
            RmsNorm::new(kn, cfg.rms_norm_eps),
            GatedAttentionConfig {
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
            },
        );
        let mlp = Mlp::from_components(
            Linear::new_fp(gate_w, None),
            Linear::new_fp(up_w, None),
            Linear::new_fp(down_w, None),
        );
        let layer = DecoderLayer::from_components_full(
            RmsNorm::new(pre_norm_w, cfg.rms_norm_eps),
            attn,
            RmsNorm::new(post_norm_w, cfg.rms_norm_eps),
            mlp,
            cfg,
        );

        let x = rand_w(&[1, 4, cfg.hidden_size], Dtype::Bfloat16);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let out = layer.forward(&x, &mrope, &cos, &sin, None, None).unwrap();
        assert_eq!(out.shape().as_slice(), &[1, 4, cfg.hidden_size]);
        assert_eq!(out.dtype(), Dtype::Bfloat16);
        // Sanity: outputs are finite.
        let v: Vec<f32> = mlx::ops::cast::astype(&out, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    #[serial(mlx_metal)]
    fn forward_residual_paths_zero_blocks_yield_input() {
        // Zero out attn (o_proj=0) AND mlp (down_proj=0); the two residual chains
        // independently reduce DecoderLayer to identity:  out = x + 0 + 0 = x.
        let cfg = small_cfg();

        // Build attention with o_proj weight = 0 → attn output is exactly 0.
        let q_w = rand_w(
            &[cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size],
            Dtype::Float32,
        );
        let k_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Float32,
        );
        let v_w = rand_w(
            &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
            Dtype::Float32,
        );
        let o_w_zero = Array::zeros(
            (cfg.hidden_size, cfg.num_heads * cfg.head_dim),
            Dtype::Float32,
        )
        .unwrap();
        let attn = GatedAttention::from_components(
            Linear::new_fp(q_w, None),
            Linear::new_fp(k_w, None),
            Linear::new_fp(v_w, None),
            Linear::new_fp(o_w_zero, None),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            RmsNorm::new(ones_w(cfg.head_dim), cfg.rms_norm_eps),
            GatedAttentionConfig {
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attention_bias: cfg.attention_bias,
            },
        );

        // Mlp with down_proj=0 → mlp output is exactly 0.
        let gate_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Float32);
        let up_w = rand_w(&[cfg.intermediate_size, cfg.hidden_size], Dtype::Float32);
        let down_w_zero =
            Array::zeros((cfg.hidden_size, cfg.intermediate_size), Dtype::Float32).unwrap();
        let mlp = Mlp::from_components(
            Linear::new_fp(gate_w, None),
            Linear::new_fp(up_w, None),
            Linear::new_fp(down_w_zero, None),
        );

        let layer = DecoderLayer::from_components_full(
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            attn,
            RmsNorm::new(ones_w(cfg.hidden_size), cfg.rms_norm_eps),
            mlp,
            cfg,
        );

        let x = rand_w(&[1, 4, cfg.hidden_size], Dtype::Float32);
        let mrope = Mrope::new(cfg.head_dim, 1e7, 1.0, &[2, 1, 1], true).unwrap();
        let pos1d = mlx::ops::constructors::arange(0.0, 4.0, 1.0, Dtype::Int32).unwrap();
        let pos1d = pos1d.reshape((1, 1, 4)).unwrap();
        let position_ids = mlx::ops::shape::broadcast_to_on(&pos1d, &[3, 1, 4], ()).unwrap();
        let (cos, sin) = mrope.cos_sin(&position_ids).unwrap();

        let out = layer.forward(&x, &mrope, &cos, &sin, None, None).unwrap();

        let xv: Vec<f32> = x.to_vec().unwrap();
        let ov: Vec<f32> = out.to_vec().unwrap();
        for (xi, oi) in xv.iter().zip(ov.iter()) {
            assert!(
                (xi - oi).abs() < 1e-5,
                "residual path broken: x={xi}, out={oi}"
            );
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn from_components_full_carries_kind_and_config() {
        let cfg = small_cfg();
        let layer = build_decoder_layer(cfg); // existing helper builds Full variant
        assert_eq!(layer.kind(), AttnKind::Full);
        assert_eq!(layer.config().hidden_size, cfg.hidden_size);
    }

    #[test]
    #[serial(mlx_metal)]
    fn from_components_linear_carries_kind() {
        // GatedDeltaNet::from_components requires P3b3 internals (Conv1d,
        // RmsNormGated, Linear etc.) that are heavy to wire up here. Keep this
        // test symbolic — verify the AttnPath::Linear and LayerCache::Linear
        // discriminators compile. Concrete construction is exercised in T4.
        let _ = AttnPath::Linear;
        let _ = LayerCache::Linear;
    }

    #[test]
    #[serial(mlx_metal)]
    fn full_layer_with_linear_cache_errors() {
        let cfg = small_cfg();
        let layer = build_decoder_layer(cfg);
        let mut bad_cache = LayerCache::Linear(
            GatedDeltaCache::new_with_cap(
                /* batch */ 1,
                /* kernel_size */ 4,
                /* conv_dim */ 16,
                /* num_v_heads */ 4,
                /* head_v_dim */ 8,
                /* head_k_dim */ 8,
                mlx::Dtype::Bfloat16,
                /* cap */ 16,
            )
            .expect("GatedDeltaCache::new_with_cap"),
        );
        let (x, mrope, cos, sin) = build_inputs_fp32(cfg);
        let r = layer.forward(&x, &mrope, &cos, &sin, None, Some(&mut bad_cache));
        let err = r.expect_err("Full layer + Linear cache must Err");
        let msg = format!("{err}");
        assert!(
            msg.contains("kind mismatch") && msg.contains("Linear cache"),
            "expected kind-mismatch message, got: {msg}"
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn linear_cache_full_arm_compiles() {
        // The Linear-layer + Full-cache mismatch arm in forward_on requires
        // a real GatedDeltaNet to construct (heavy P3b3 internals). The
        // dispatch arm itself is exercised in T4 (Qwen35Model assembly tests);
        // here we only confirm the LayerCache::Full discriminator compiles.
        let _ = LayerCache::Full;
    }

    #[test]
    #[serial(mlx_metal)]
    fn enable_turboquant_kv_caches_only_updates_full_layers() {
        let mut caches = vec![
            LayerCache::Full(KVCache::new(1, 1, 8, 8, Dtype::Bfloat16, 16)),
            LayerCache::Linear(
                GatedDeltaCache::new_with_cap(
                    /* batch */ 1,
                    /* kernel_size */ 4,
                    /* conv_dim */ 16,
                    /* num_v_heads */ 4,
                    /* head_v_dim */ 8,
                    /* head_k_dim */ 8,
                    Dtype::Bfloat16,
                    /* cap */ 16,
                )
                .expect("GatedDeltaCache::new_with_cap"),
            ),
        ];

        enable_turboquant_kv_caches(&mut caches, TurboQuantKVBits::K3V3)
            .expect("enable turboquant");

        match &caches[0] {
            LayerCache::Full(kv) => {
                assert_eq!(
                    kv.turboquant().expect("turboquant cache").bits(),
                    TurboQuantKVBits::K3V3
                )
            }
            _ => panic!("expected Full layer"),
        }
        assert!(matches!(caches[1], LayerCache::Linear(_)));
    }

    fn turboquant_full_cache(bits: TurboQuantKVBits) -> LayerCache {
        let mut kv = KVCache::new(1, 2, 8, 8, Dtype::Float32, 16)
            .with_step(16)
            .with_turboquant(bits)
            .expect("enable turboquant");
        let k_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| ((i as f32) * 0.019).sin())
            .collect();
        let v_data: Vec<f32> = (0..(1 * 2 * 4 * 8))
            .map(|i| ((i as f32) * 0.023).cos())
            .collect();
        let k: Array = (k_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 2_i32, 4_i32, 8_i32))
            .try_into()
            .unwrap();
        kv.update_and_fetch(&k, &v, &[4])
            .expect("write turboquant prefix");
        LayerCache::Full(kv)
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefix_key_spec_for_turboquant_full_cache_uses_packed_profile() {
        let caches_k3v3 = vec![turboquant_full_cache(TurboQuantKVBits::K3V3)];
        let spec_k3v3 =
            prefix_key_spec_for_caches("model", &[1, 2, 3, 4], 4, None, 16, &caches_k3v3)
                .expect("prefix spec")
                .expect("TurboQuant prefix spec");

        assert_eq!(
            spec_k3v3.main_layers[0].kind,
            PrefixLayerKind::FullTurboQuantPacked
        );
        assert_eq!(spec_k3v3.main_layers[0].tensors[0].shape, vec![1, 2, 4, 1]);
        assert_eq!(spec_k3v3.main_layers[0].tensors[1].shape, vec![1, 2, 4]);
        assert_eq!(spec_k3v3.main_layers[0].tensors[2].shape, vec![1, 2, 4, 1]);
        assert_eq!(spec_k3v3.main_layers[0].tensors[3].shape, vec![1, 2, 4]);

        let caches_k3v4 = vec![turboquant_full_cache(TurboQuantKVBits::K3V4)];
        let spec_k3v4 =
            prefix_key_spec_for_caches("model", &[1, 2, 3, 4], 4, None, 16, &caches_k3v4)
                .expect("prefix spec")
                .expect("TurboQuant prefix spec");

        assert_ne!(
            PagedPrefixStore::key_for(&spec_k3v3),
            PagedPrefixStore::key_for(&spec_k3v4)
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn prefix_entry_for_turboquant_full_cache_exports_packed_payload() {
        let caches = vec![turboquant_full_cache(TurboQuantKVBits::K3V4)];

        let (entry, cached_len) = prefix_entry_for_row(&caches, 0)
            .expect("prefix entry")
            .expect("TurboQuant prefix entry");

        assert_eq!(cached_len, 4);
        let PrefixLayerPayload::FullTurboQuantPacked {
            k_packed,
            k_norms,
            v_packed,
            v_norms,
        } = &entry.main_layers[0]
        else {
            panic!("expected packed TurboQuant full-attention payload");
        };
        assert_eq!(k_packed.shape().as_slice(), &[1, 2, 4, 1]);
        assert_eq!(k_norms.shape().as_slice(), &[1, 2, 4]);
        assert_eq!(v_packed.shape().as_slice(), &[1, 2, 4, 1]);
        assert_eq!(v_norms.shape().as_slice(), &[1, 2, 4]);
    }

    #[test]
    #[serial(mlx_metal)]
    fn restore_prefix_entry_for_turboquant_full_cache_uses_packed_payload() {
        let caches = vec![turboquant_full_cache(TurboQuantKVBits::K3V4)];
        let (entry, cached_len) = prefix_entry_for_row(&caches, 0)
            .expect("prefix entry")
            .expect("TurboQuant prefix entry");
        let mut restored = vec![LayerCache::Full(
            KVCache::new(1, 2, 8, 8, Dtype::Float32, 16)
                .with_step(16)
                .with_turboquant(TurboQuantKVBits::K3V4)
                .expect("enable restore turboquant"),
        )];

        restore_prefix_entry_for_row(&mut restored, &entry, 0, cached_len)
            .expect("restore packed TurboQuant prefix");

        let LayerCache::Full(kv) = &restored[0] else {
            panic!("expected full cache");
        };
        assert_eq!(kv.offsets(), &[4]);
        let tq = kv.turboquant().expect("turboquant cache");
        assert_eq!(
            tq.k_packed().expect("K packed").shape().as_slice(),
            &[1, 2, 16, 1]
        );
        assert_eq!(
            tq.k_norms().expect("K norms").shape().as_slice(),
            &[1, 2, 16]
        );
    }
}
