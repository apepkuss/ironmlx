use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::{
    ActiveKvOffloadConfig, ActiveKvOffloadSharedStats, ActiveKvResidencySummary,
    PagedKvHotColdConfig, PagedPrefixCacheConfig, PagedPrefixEntry, PagedPrefixEntryStats,
    PagedPrefixLoadStatus, PagedPrefixStore, PrefixLruCache, PrefixLruCacheConfig,
    PrefixLruInsertStatus, PrefixTensorSpec, TurboQuantKVBits,
};
use crate::core::generate::{
    build_position_ids, build_position_ids_vl, count_image_pad, extend_vl_chunk_end_for_image_pad,
    slice_pos_ids_axis2, slice_vision_embeds_rows, GenerateEvent, GenerateRequest,
};
use crate::core::scheduler::{paged_prefix_fingerprint_for_request, DenseVlMethods};
use crate::core::speculative::{
    add_elapsed_us, adjust_mtp_draft_budget, resolve_speculative_tokens, restore_layer_cache,
    sample_logits_positions, slice_hidden_position, verify_input, MtpSpeculativeConfig,
    MtpSpeculativeStats,
};
use crate::core::tokenizer::{DecodeStream, Tokenizer};
use crate::core::{Loader, Model};
use crate::nn::{
    enable_paged_hot_cold_tiering_caches, enable_paged_kv_caches, enable_turboquant_kv_caches,
    prefix_entry_for_row, prefix_key_spec_for_caches, restore_prefix_entry_for_row, LayerCache,
    LayerCacheSnapshot, Linear,
};
use crate::Result;

use super::attention::SharedKv;
use super::config::{Gemma4AssistantConfig, Gemma4LayerKind, Gemma4TextConfig};
use super::model::Gemma4Model;
use super::text_model::{Gemma4SharedKvStates, Gemma4TextModel};

pub struct Gemma4DrafterMasks {
    sliding: Option<Array>,
    full: Option<Array>,
}

impl Gemma4DrafterMasks {
    pub fn get(&self, kind: Gemma4LayerKind) -> Option<&Array> {
        match kind {
            Gemma4LayerKind::Sliding => self.sliding.as_ref(),
            Gemma4LayerKind::Full => self.full.as_ref(),
        }
    }
}

pub struct Gemma4DrafterStepOutput {
    pub hidden_states: Array,
    pub logits: Array,
}

pub struct Gemma4AssistantModel {
    cfg: Gemma4AssistantConfig,
    text: Gemma4TextModel,
    pre_projection: Linear,
    post_projection: Linear,
    masked_embedding: Option<MaskedEmbedder>,
}

impl Gemma4AssistantModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = Gemma4AssistantConfig::from_loader(loader)?;
        let text =
            Gemma4TextModel::from_loader_external_shared_kv(loader, cfg.text_config.clone())?;
        let pre_projection = Linear::from_loader(loader, "pre_projection")?;
        let post_projection = Linear::from_loader(loader, "post_projection")?;
        let masked_embedding = if cfg.use_ordered_embeddings {
            Some(MaskedEmbedder::from_loader(loader, &cfg)?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            text,
            pre_projection,
            post_projection,
            masked_embedding,
        })
    }

    pub fn config(&self) -> &Gemma4AssistantConfig {
        &self.cfg
    }

    pub fn forward_on(
        &self,
        inputs_embeds: &Array,
        shared_kv: &Gemma4SharedKvStates,
        position: i32,
        kv_valid_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Gemma4DrafterStepOutput> {
        let target = target.into();
        let h = self.pre_projection.forward_on(inputs_embeds, target)?;
        let shape = h.shape();
        let dims = shape.as_slice();
        if dims.len() != 3 {
            return Err(anyhow!(
                "Gemma4AssistantModel::forward_on: expected hidden [B,S,H], got {dims:?}"
            ));
        }
        let masks = make_drafter_masks(
            shared_kv,
            dims[1],
            position,
            self.cfg.text_config.sliding_window,
            h.dtype(),
            kv_valid_len,
            target,
        )?;
        let h = self
            .text
            .forward_external_shared_kv_on(&h, shared_kv, &masks, position, target)?;
        let hidden_states = self.post_projection.forward_on(&h, target)?;
        let logits = match self.masked_embedding.as_ref() {
            Some(masked) => {
                let weight = self.text_embedding_dense_weight_on(target)?;
                masked.forward_on(&h, &weight, target)?
            }
            None => self.text.as_output_on(&h, target)?,
        };
        Ok(Gemma4DrafterStepOutput {
            hidden_states,
            logits,
        })
    }

    fn text_embedding_dense_weight_on(&self, target: StreamOrDevice) -> Result<Array> {
        self.text.dense_embedding_weight_on(target)
    }
}

type Gemma4DrafterPrefixLruHandle = Arc<Mutex<PrefixLruCache>>;

#[derive(Clone)]
pub struct Gemma4DrafterActiveKvRuntime {
    config: ActiveKvOffloadConfig,
    stats: ActiveKvOffloadSharedStats,
    soft_limit_bytes: usize,
    bytes_per_token: usize,
}

impl Gemma4DrafterActiveKvRuntime {
    pub fn new(
        config: ActiveKvOffloadConfig,
        stats: ActiveKvOffloadSharedStats,
        soft_limit_bytes: usize,
        bytes_per_token: usize,
    ) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        Some(Self {
            config,
            stats,
            soft_limit_bytes,
            bytes_per_token,
        })
    }
}

#[derive(Clone, Default)]
pub struct Gemma4DrafterPrefixCache {
    config: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<Gemma4DrafterPrefixLruHandle>,
    active_kv: Option<Gemma4DrafterActiveKvRuntime>,
}

impl Gemma4DrafterPrefixCache {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn new(
        config: Option<PagedPrefixCacheConfig>,
        prefix_lru_cache: Option<PrefixLruCacheConfig>,
        active_kv: Option<Gemma4DrafterActiveKvRuntime>,
    ) -> Result<Self> {
        if prefix_lru_cache.is_some() && config.is_none() {
            return Err(anyhow!(
                "Gemma4 drafter prefix LRU cache requires paged prefix cache"
            ));
        }
        let prefix_lru_cache = prefix_lru_cache
            .map(PrefixLruCache::new)
            .transpose()?
            .map(|cache| Arc::new(Mutex::new(cache)));
        Ok(Self {
            config,
            prefix_lru_cache,
            active_kv,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.config.is_some()
    }

    fn config(&self) -> Option<&PagedPrefixCacheConfig> {
        self.config.as_ref()
    }

    fn prefix_lru_cache(&self) -> Option<&Gemma4DrafterPrefixLruHandle> {
        self.prefix_lru_cache.as_ref()
    }

    fn enable_runtime_cache_storage(
        &self,
        cache: &mut [LayerCache],
        turboquant_bits: Option<TurboQuantKVBits>,
        cache_cap: i32,
        batch: i32,
    ) -> Result<()> {
        if let Some(bits) = turboquant_bits {
            enable_turboquant_kv_caches(cache, bits)?;
        } else if let Some(config) = self.config() {
            enable_paged_kv_caches(cache, config.block_size, config.max_pages)?;
            self.enable_active_kv_hot_cold_tiering(cache, config, cache_cap, batch)?;
        }
        self.refresh_active_kv_residency_stats(cache);
        Ok(())
    }

    fn enable_active_kv_hot_cold_tiering(
        &self,
        cache: &mut [LayerCache],
        config: &PagedPrefixCacheConfig,
        cache_cap: i32,
        batch: i32,
    ) -> Result<()> {
        let Some(active_kv) = self.active_kv.as_ref() else {
            return Ok(());
        };
        let hot_window_pages = active_kv
            .config
            .hot_window_pages_override
            .unwrap_or_else(|| {
                crate::core::scheduler::active_kv_hot_window_pages_for_budget(
                    config.block_size,
                    cache_cap,
                    batch,
                    active_kv.soft_limit_bytes,
                    active_kv.bytes_per_token,
                )
            });
        let chunk_pages = active_kv.config.chunk_pages_override.unwrap_or_else(|| {
            crate::core::scheduler::active_kv_chunk_pages_for_budget(
                config.block_size,
                cache_cap,
                batch,
                active_kv.soft_limit_bytes,
                active_kv.bytes_per_token,
                hot_window_pages,
            )
        });
        let hot_cold = PagedKvHotColdConfig::new(
            active_kv.config.root.clone(),
            hot_window_pages,
            chunk_pages,
        )?;
        enable_paged_hot_cold_tiering_caches(cache, hot_cold)
    }

    fn refresh_active_kv_residency_stats(&self, cache: &[LayerCache]) {
        let Some(active_kv) = self.active_kv.as_ref() else {
            return;
        };
        let mut summary = ActiveKvResidencySummary::default();
        for layer in cache {
            let LayerCache::Full(kv) = layer else {
                continue;
            };
            let Some(layer_summary) = kv.paged_hot_cold_summary() else {
                continue;
            };
            summary.resident_pages += layer_summary.resident_pages;
            summary.offloaded_pages += layer_summary.offloaded_pages;
            summary.loading_pages += layer_summary.loading_pages;
            summary.dirty_pages += layer_summary.dirty_pages;
            summary.offloaded_bytes = summary
                .offloaded_bytes
                .saturating_add(layer_summary.offloaded_bytes);
            summary.swap_out_count = summary
                .swap_out_count
                .saturating_add(layer_summary.swap_out_count);
            summary.swap_in_count = summary
                .swap_in_count
                .saturating_add(layer_summary.swap_in_count);
            summary.stream_read_count = summary
                .stream_read_count
                .saturating_add(layer_summary.stream_read_count);
        }
        active_kv.stats.set_residency_summary(summary);
    }

    fn try_restore(
        &self,
        model: &Gemma4Model,
        cache: &mut [LayerCache],
        prompt_ids: &[u32],
        fingerprint: Option<&str>,
    ) -> Result<Option<Gemma4DrafterPrefixRestore>> {
        let Some(config) = self.config() else {
            return Ok(None);
        };
        if prompt_ids.is_empty() {
            return Ok(None);
        }

        let store = config.store();
        for (restore_len, cached_len) in gemma4_drafter_prefix_restore_candidates(
            &store,
            self.prefix_lru_cache(),
            prompt_ids.len(),
        )? {
            let Some(mut spec) = prefix_key_spec_for_caches(
                &config.model_id,
                &prompt_ids[..restore_len],
                cached_len,
                fingerprint,
                config.block_size,
                cache,
            )?
            else {
                return Ok(None);
            };
            spec.gemma4_drafter_last_hidden = Some(gemma4_drafter_last_hidden_spec(
                model.hidden_dtype(),
                model.config().hidden_size,
            ));

            if let Some((key, entry, stats, load_us)) =
                gemma4_drafter_try_load_prefix_lru_entry(self.prefix_lru_cache(), &spec)?
            {
                restore_prefix_entry_for_row(cache, &entry, 0, cached_len)?;
                let last_hidden = entry
                    .gemma4_drafter_last_hidden
                    .ok_or_else(|| anyhow!("Gemma4 drafter prefix LRU hit missing last_hidden"))?;
                let shared_kv = gemma4_shared_kv_from_cache_on(model.config(), cache, ())?;
                log_gemma4_drafter_prefix_hit("prefix LRU hit", &key, restore_len, stats, load_us);
                return Ok(Some(Gemma4DrafterPrefixRestore {
                    cached_len,
                    last_hidden,
                    shared_kv,
                }));
            }

            let load_start = Instant::now();
            let observed = store.load_observed(&spec)?;
            let load_us = load_start.elapsed().as_micros();
            if observed.status != PagedPrefixLoadStatus::Hit {
                tracing::trace!(
                    "paged SSD prefix cache Gemma4 drafter miss: tokens={} key={} status={:?} load_us={}",
                    restore_len,
                    observed.key,
                    observed.status,
                    load_us
                );
                continue;
            }
            let key = observed.key;
            let stats = observed
                .stats
                .unwrap_or_else(|| gemma4_empty_prefix_stats(cached_len));
            let entry = observed
                .entry
                .ok_or_else(|| anyhow!("paged prefix Gemma4 drafter hit without entry"))?;
            gemma4_drafter_try_insert_prefix_lru_entry(
                self.prefix_lru_cache(),
                spec,
                entry.clone(),
            )?;
            restore_prefix_entry_for_row(cache, &entry, 0, cached_len)?;
            let last_hidden = entry
                .gemma4_drafter_last_hidden
                .ok_or_else(|| anyhow!("paged prefix Gemma4 drafter hit missing last_hidden"))?;
            let shared_kv = gemma4_shared_kv_from_cache_on(model.config(), cache, ())?;
            log_gemma4_drafter_prefix_hit("paged SSD hit", &key, restore_len, stats, load_us);
            return Ok(Some(Gemma4DrafterPrefixRestore {
                cached_len,
                last_hidden,
                shared_kv,
            }));
        }

        Ok(None)
    }

    fn try_save(
        &self,
        model: &Gemma4Model,
        cache: &[LayerCache],
        last_hidden: &Array,
        prompt_ids: &[u32],
        fingerprint: Option<&str>,
    ) -> Result<Option<String>> {
        let Some(config) = self.config() else {
            return Ok(None);
        };
        if prompt_ids.is_empty() {
            return Ok(None);
        }
        let Some(cached_len) = gemma4_cache_row_cached_len(cache, 0)? else {
            return Ok(None);
        };
        if cached_len == 0 {
            return Ok(None);
        }
        if cached_len != prompt_ids.len() as i32 {
            return Err(anyhow!(
                "Gemma4 drafter prefix save: cache cached_len {cached_len} != token length {}",
                prompt_ids.len()
            ));
        }
        let Some(mut spec) = prefix_key_spec_for_caches(
            &config.model_id,
            prompt_ids,
            cached_len,
            fingerprint,
            config.block_size,
            cache,
        )?
        else {
            return Ok(None);
        };
        let actual_last_hidden_spec = PrefixTensorSpec::from_array(last_hidden);
        let expected_last_hidden_spec =
            gemma4_drafter_last_hidden_spec(model.hidden_dtype(), model.config().hidden_size);
        spec.gemma4_drafter_last_hidden = Some(actual_last_hidden_spec.clone());
        if actual_last_hidden_spec != expected_last_hidden_spec {
            return Err(anyhow!(
                "Gemma4 drafter prefix save: last_hidden spec {actual_last_hidden_spec:?} does not match expected {expected_last_hidden_spec:?}"
            ));
        }

        let Some((mut entry, entry_cached_len)) = prefix_entry_for_row(cache, 0)? else {
            return Ok(None);
        };
        if entry_cached_len != cached_len {
            return Err(anyhow!(
                "Gemma4 drafter prefix save: entry cached_len {entry_cached_len} != cache {cached_len}"
            ));
        }
        entry.gemma4_drafter_last_hidden = Some(last_hidden.clone());
        spec.gemma4_drafter_last_hidden = entry.gemma4_drafter_last_hidden_spec();
        let stats = entry.observability_stats(cached_len);

        gemma4_drafter_try_insert_prefix_lru_entry(
            self.prefix_lru_cache(),
            spec.clone(),
            entry.clone(),
        )?;
        let store = config.store();
        if let Some(key) = store.matching_entry_key(&spec)? {
            tracing::trace!(
                "paged SSD prefix cache Gemma4 drafter save skipped: tokens={} key={} status=already_present",
                prompt_ids.len(),
                key
            );
            return Ok(None);
        }
        let save_start = Instant::now();
        let (key, saved) = store.save_if_absent(&spec, &entry)?;
        let save_us = save_start.elapsed().as_micros();
        if !saved {
            tracing::trace!(
                "paged SSD prefix cache Gemma4 drafter save skipped: tokens={} key={} status=already_present",
                prompt_ids.len(),
                key
            );
            return Ok(None);
        }
        tracing::debug!(
            "paged SSD prefix cache Gemma4 drafter saved: key={} tokens={} cached_len={} payload_bytes={} tensors={} save_us={}",
            key,
            prompt_ids.len(),
            stats.cached_len,
            stats.payload_bytes,
            stats.tensor_count,
            save_us
        );
        Ok(Some(key))
    }
}

struct Gemma4DrafterPrefixRestore {
    cached_len: i32,
    last_hidden: Array,
    shared_kv: Gemma4SharedKvStates,
}

fn gemma4_drafter_last_hidden_spec(dtype: Dtype, hidden_size: i32) -> PrefixTensorSpec {
    PrefixTensorSpec {
        dtype,
        shape: vec![1_i32, 1_i32, hidden_size],
    }
}

fn gemma4_empty_prefix_stats(cached_len: i32) -> PagedPrefixEntryStats {
    PagedPrefixEntryStats {
        cached_len,
        ..PagedPrefixEntryStats::default()
    }
}

fn gemma4_drafter_prefix_restore_candidates(
    store: &PagedPrefixStore,
    prefix_lru_cache: Option<&Gemma4DrafterPrefixLruHandle>,
    prompt_len: usize,
) -> Result<Vec<(usize, i32)>> {
    if prompt_len == 0 {
        return Ok(Vec::new());
    }
    let max_cached_len = i32::try_from(prompt_len)
        .map_err(|_| anyhow!("Gemma4 drafter prefix restore length exceeds i32"))?;
    let mut cached_lengths = Vec::new();
    if let Some(prefix_lru_cache) = prefix_lru_cache {
        cached_lengths.extend(
            gemma4_lock_prefix_lru_cache(prefix_lru_cache)?
                .cached_lengths_descending(max_cached_len as usize),
        );
    }
    cached_lengths.extend(store.cached_lengths_descending(max_cached_len)?);
    cached_lengths.sort_unstable_by(|a, b| b.cmp(a));
    cached_lengths.dedup();

    let mut candidates = Vec::with_capacity(cached_lengths.len());
    for cached_len in cached_lengths {
        if cached_len <= 0 {
            continue;
        }
        let restore_len = usize::try_from(cached_len)
            .map_err(|_| anyhow!("Gemma4 drafter cached length must be positive"))?;
        candidates.push((restore_len, cached_len));
    }
    Ok(candidates)
}

fn gemma4_lock_prefix_lru_cache(
    prefix_lru_cache: &Gemma4DrafterPrefixLruHandle,
) -> Result<MutexGuard<'_, PrefixLruCache>> {
    prefix_lru_cache
        .lock()
        .map_err(|_| anyhow!("Gemma4 drafter prefix LRU cache lock poisoned"))
}

fn gemma4_drafter_try_load_prefix_lru_entry(
    prefix_lru_cache: Option<&Gemma4DrafterPrefixLruHandle>,
    spec: &crate::core::cache::PagedPrefixKeySpec,
) -> Result<Option<(String, PagedPrefixEntry, PagedPrefixEntryStats, u128)>> {
    let Some(prefix_lru_cache) = prefix_lru_cache else {
        return Ok(None);
    };
    let load_start = Instant::now();
    let observed = gemma4_lock_prefix_lru_cache(prefix_lru_cache)?.load_observed(spec)?;
    let load_us = load_start.elapsed().as_micros();
    if observed.status != PagedPrefixLoadStatus::Hit {
        tracing::trace!(
            "Gemma4 drafter prefix LRU miss: key={} status={:?} load_us={}",
            observed.key,
            observed.status,
            load_us
        );
        return Ok(None);
    }
    let key = observed.key;
    let stats = observed
        .stats
        .unwrap_or_else(|| gemma4_empty_prefix_stats(spec.cached_len));
    let entry = observed
        .entry
        .ok_or_else(|| anyhow!("Gemma4 drafter prefix LRU observed hit without entry"))?;
    Ok(Some((key, entry, stats, load_us)))
}

fn gemma4_drafter_try_insert_prefix_lru_entry(
    prefix_lru_cache: Option<&Gemma4DrafterPrefixLruHandle>,
    spec: crate::core::cache::PagedPrefixKeySpec,
    entry: PagedPrefixEntry,
) -> Result<Option<String>> {
    let Some(prefix_lru_cache) = prefix_lru_cache else {
        return Ok(None);
    };
    let save_start = Instant::now();
    let result = gemma4_lock_prefix_lru_cache(prefix_lru_cache)?.insert(spec, entry)?;
    let save_us = save_start.elapsed().as_micros();
    match result.status {
        PrefixLruInsertStatus::Stored | PrefixLruInsertStatus::Replaced => {
            tracing::debug!(
                "Gemma4 drafter prefix LRU {}: key={} cached_len={} payload_bytes={} tensors={} save_us={}",
                match result.status {
                    PrefixLruInsertStatus::Stored => "saved",
                    PrefixLruInsertStatus::Replaced => "updated",
                    PrefixLruInsertStatus::SkippedOversized => unreachable!(),
                },
                result.key,
                result.stats.cached_len,
                result.stats.payload_bytes,
                result.stats.tensor_count,
                save_us
            );
            Ok(Some(result.key))
        }
        PrefixLruInsertStatus::SkippedOversized => {
            tracing::trace!(
                "Gemma4 drafter prefix LRU save skipped: key={} status=oversized payload_bytes={} max_bytes={}",
                result.key,
                result.stats.payload_bytes,
                gemma4_lock_prefix_lru_cache(prefix_lru_cache)?.max_bytes()
            );
            Ok(None)
        }
    }
}

fn log_gemma4_drafter_prefix_hit(
    source: &'static str,
    key: &str,
    tokens: usize,
    stats: PagedPrefixEntryStats,
    load_us: u128,
) {
    tracing::debug!(
        "Gemma4 drafter prefix cache {}: key={} tokens={} cached_len={} payload_bytes={} tensors={} load_us={}",
        source,
        key,
        tokens,
        stats.cached_len,
        stats.payload_bytes,
        stats.tensor_count,
        load_us
    );
}

fn gemma4_cache_row_cached_len(cache: &[LayerCache], row: usize) -> Result<Option<i32>> {
    for layer in cache {
        if let LayerCache::Full(kv) = layer {
            let cached_len = *kv.offsets().get(row).ok_or_else(|| {
                anyhow!(
                    "Gemma4 drafter prefix cache row {row} out of range for batch {}",
                    kv.offsets().len()
                )
            })?;
            return Ok(Some(cached_len));
        }
    }
    Ok(None)
}

fn gemma4_shared_kv_from_cache_on(
    cfg: &Gemma4TextConfig,
    cache: &[LayerCache],
    target: impl Into<StreamOrDevice>,
) -> Result<Gemma4SharedKvStates> {
    let target = target.into();
    let first_cache_layer = cfg.first_kv_shared_layer_idx();
    if cache.len() != first_cache_layer {
        return Err(anyhow!(
            "Gemma4 drafter shared KV restore: cache.len()={} != cache-bearing layers {}",
            cache.len(),
            first_cache_layer
        ));
    }
    if first_cache_layer == 0 {
        return Err(anyhow!(
            "Gemma4 drafter shared KV restore: target model has no cache-bearing layers"
        ));
    }

    let mut states = Gemma4SharedKvStates::default();
    let mut restored_len: Option<i32> = None;
    for (idx, layer) in cache.iter().enumerate() {
        let LayerCache::Full(kv) = layer else {
            return Err(anyhow!(
                "Gemma4 drafter shared KV restore: layer {idx} is not a Full KV cache"
            ));
        };
        let layer_len = *kv.offsets().first().ok_or_else(|| {
            anyhow!("Gemma4 drafter shared KV restore: layer {idx} has empty offsets")
        })?;
        if layer_len <= 0 {
            return Err(anyhow!(
                "Gemma4 drafter shared KV restore: layer {idx} cached_len must be > 0"
            ));
        }
        if let Some(expected) = restored_len {
            if layer_len != expected {
                return Err(anyhow!(
                    "Gemma4 drafter shared KV restore: layer {idx} cached_len {layer_len} != layer0 {expected}"
                ));
            }
        } else {
            restored_len = Some(layer_len);
        }

        let (keys, values) = if kv.paged().is_some() {
            kv.materialize_current_paged_prefix_on(target)?
        } else {
            let (keys, values, dense_len) = kv.dense_prefix_layer_for_row_on(0, target)?;
            if dense_len != layer_len {
                return Err(anyhow!(
                    "Gemma4 drafter shared KV restore: layer {idx} dense_len {dense_len} != offset {layer_len}"
                ));
            }
            (keys, values)
        };
        if keys.shape().as_slice().first().copied() != Some(1)
            || values.shape().as_slice().first().copied() != Some(1)
        {
            return Err(anyhow!(
                "Gemma4 drafter shared KV restore: layer {idx} restored batch must be 1"
            ));
        }
        states.insert(cfg.layer_kind(idx), SharedKv { keys, values });
    }

    for idx in 0..cfg.num_hidden_layers as usize {
        let kind = cfg.layer_kind(idx);
        if states.get(kind).is_none() {
            return Err(anyhow!(
                "Gemma4 drafter shared KV restore: missing {:?} shared KV state",
                kind
            ));
        }
    }

    Ok(states)
}

pub struct Gemma4DrafterGenerationStream<'m> {
    model: &'m Gemma4Model,
    drafter: &'m Gemma4AssistantModel,
    prefix_cache: Gemma4DrafterPrefixCache,
    cache: Vec<LayerCache>,
    history: Vec<u32>,
    request: GenerateRequest,
    cfg: MtpSpeculativeConfig,
    pending_tokens: VecDeque<u32>,
    detok: DecodeStream<'m>,
    /// Hidden state for the token immediately before the current pending token.
    last_hidden: Array,
    shared_kv: Gemma4SharedKvStates,
    emitted_new_tokens: usize,
    finished: bool,
    dummy_position_ids: Option<Array>,
    prng_state: Array,
    adaptive_draft_tokens: usize,
    stats: MtpSpeculativeStats,
    trace_window_limit: usize,
    trace_windows: Vec<Gemma4DrafterTraceWindow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4DrafterTraceWindow {
    pub history_len: usize,
    pub verify_start_pos: i32,
    pub draft_tokens: Vec<u32>,
    pub verified_tokens: Vec<u32>,
    pub accepted_draft_len: usize,
}

impl<'m> Gemma4DrafterGenerationStream<'m> {
    pub fn new(
        model: &'m Gemma4Model,
        drafter: &'m Gemma4AssistantModel,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Self> {
        Self::new_with_prefix_cache(
            model,
            drafter,
            tokenizer,
            request,
            cfg,
            Gemma4DrafterPrefixCache::disabled(),
        )
    }

    pub fn new_with_prefix_cache(
        model: &'m Gemma4Model,
        drafter: &'m Gemma4AssistantModel,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
        cfg: MtpSpeculativeConfig,
        prefix_cache: Gemma4DrafterPrefixCache,
    ) -> Result<Self> {
        if request.prompt_ids.is_empty() {
            return Err(anyhow!(
                "Gemma4DrafterGenerationStream::new: prompt_ids cannot be empty"
            ));
        }
        if cfg.max_draft_tokens == 0 {
            return Err(anyhow!(
                "Gemma4DrafterGenerationStream::new: max_draft_tokens must be > 0"
            ));
        }
        if !request.sampler.is_pipelinable() {
            return Err(anyhow!(
                "Gemma4DrafterGenerationStream::new: Gemma4 drafter decoding currently requires greedy sampling"
            ));
        }
        if request.pixel_values.is_none() && request.image_grid_thw.is_some() {
            return Err(anyhow!(
                "Gemma4DrafterGenerationStream::new: image_grid_thw present but pixel_values is None"
            ));
        }

        let prompt_len = request.prompt_ids.len();
        let cap = ((prompt_len + request.max_new_tokens) as i32)
            .max(crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let dtype = model.cache_dtype();
        let mut cache = model.make_cache(1, cap, dtype)?;
        prefix_cache.enable_runtime_cache_storage(
            &mut cache,
            request.kv_cache_turboquant_bits,
            cap,
            1,
        )?;
        let dummy_position_ids = if model.requires_position_ids() {
            None
        } else {
            Some(build_position_ids(0, 1)?)
        };

        let mut stats = MtpSpeculativeStats::default();
        let mut pos = 0_i32;
        let prompt_len_i32 = prompt_len as i32;
        let mut image_pad_consumed = 0usize;
        let is_vl = request.pixel_values.is_some();
        let prefix_fingerprint = if prefix_cache.is_enabled() {
            paged_prefix_fingerprint_for_request(
                request.pixel_values.as_deref(),
                request.image_grid_thw.as_deref(),
                request.image_token_id,
                request.image_spatial_merge_size,
            )?
        } else {
            None
        };
        let mut vision_embeds_full = None;
        let position_ids_full = if is_vl && dummy_position_ids.is_none() {
            let grids = request.image_grid_thw.as_deref().ok_or_else(|| {
                anyhow!("Gemma4DrafterGenerationStream::new: pixel_values present but image_grid_thw is None")
            })?;
            if model.vl_positions_sequential() {
                Some(build_position_ids(0, prompt_len_i32)?)
            } else {
                let prompt_ids_i32: Vec<i32> =
                    request.prompt_ids.iter().map(|&id| id as i32).collect();
                Some(build_position_ids_vl(
                    &prompt_ids_i32,
                    grids,
                    request.image_token_id,
                    request.image_spatial_merge_size,
                )?)
            }
        } else {
            None
        };
        let mut last_prompt_hidden = None;
        let mut last_shared_kv = None;

        if let Some(restored) = prefix_cache.try_restore(
            model,
            &mut cache,
            &request.prompt_ids,
            prefix_fingerprint.as_deref(),
        )? {
            pos = restored.cached_len;
            image_pad_consumed =
                count_image_pad(&request.prompt_ids[..pos as usize], request.image_token_id);
            last_prompt_hidden = Some(restored.last_hidden);
            last_shared_kv = Some(restored.shared_kv);
            prefix_cache.refresh_active_kv_residency_stats(&cache);
        }

        while pos < prompt_len_i32 {
            let remaining = prompt_len_i32 - pos;
            let mut n = if request.prefill_chunk_size == 0 {
                remaining
            } else {
                remaining.min(request.prefill_chunk_size as i32)
            };
            if n <= 0 {
                return Err(anyhow!(
                    "Gemma4DrafterGenerationStream::new: invalid prefill chunk length {n}"
                ));
            }
            if is_vl && request.prefill_chunk_size != 0 {
                let adjusted_end = extend_vl_chunk_end_for_image_pad(
                    &request.prompt_ids,
                    request.image_token_id,
                    pos,
                    pos + n,
                );
                n = adjusted_end - pos;
            }

            let chunk_ids = &request.prompt_ids[pos as usize..(pos as usize + n as usize)];
            let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;
            let chunk_pos_ids = match (dummy_position_ids.as_ref(), position_ids_full.as_ref()) {
                (Some(dummy), _) => dummy.clone(),
                (None, Some(full)) => slice_pos_ids_axis2(full, pos, pos + n)?,
                (None, None) => build_position_ids(pos, n)?,
            };

            let forward_start = Instant::now();
            let out = if is_vl {
                let image_tokens = count_image_pad(chunk_ids, request.image_token_id);
                if image_tokens > 0 && vision_embeds_full.is_none() {
                    let pixel_values = request.pixel_values.as_deref().ok_or_else(|| {
                        anyhow!(
                            "Gemma4DrafterGenerationStream::new: image tokens without pixel_values"
                        )
                    })?;
                    let grid_thw = request.image_grid_thw.as_deref().ok_or_else(|| {
                        anyhow!("Gemma4DrafterGenerationStream::new: image tokens without image_grid_thw")
                    })?;
                    vision_embeds_full =
                        Some(model.compute_vision_embeds(pixel_values, grid_thw, ().into())?);
                }
                let vision_slice = match vision_embeds_full.as_ref() {
                    Some(ve) if image_tokens > 0 => Some(slice_vision_embeds_rows(
                        ve,
                        image_pad_consumed,
                        image_pad_consumed + image_tokens,
                    )?),
                    _ => None,
                };
                image_pad_consumed += image_tokens;
                model.forward_vl_hidden_with_shared_kv_on(
                    &chunk_arr,
                    &chunk_pos_ids,
                    None,
                    None,
                    Some(&mut cache),
                    vision_slice.as_ref(),
                    request.image_token_id,
                    ().into(),
                )?
            } else {
                model.forward_text_hidden_with_shared_kv_on(
                    &chunk_arr,
                    &chunk_pos_ids,
                    None,
                    None,
                    Some(&mut cache),
                    (),
                )?
            };
            add_elapsed_us(&mut stats.verify_forward_us, forward_start);
            let chunk_last_hidden = slice_hidden_position(&out.hidden, n - 1)?;
            let new_pos = pos + n;
            match prefix_cache.try_save(
                model,
                &cache,
                &chunk_last_hidden,
                &request.prompt_ids[..new_pos as usize],
                prefix_fingerprint.as_deref(),
            ) {
                Ok(Some(key)) => {
                    tracing::debug!("paged SSD prefix cache Gemma4 drafter saved: key={key}");
                }
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!("paged SSD prefix cache Gemma4 drafter save skipped: {err:#}");
                }
            }
            prefix_cache.refresh_active_kv_residency_stats(&cache);
            if new_pos == prompt_len_i32 {
                last_prompt_hidden = Some(chunk_last_hidden);
                last_shared_kv = Some(out.shared_kv);
            } else {
                mlx::transforms::eval(&[&out.hidden])?;
            }
            pos = new_pos;
        }

        let last_prompt_hidden = last_prompt_hidden
            .ok_or_else(|| anyhow!("Gemma4 drafter prefill produced no prompt hidden"))?;
        let shared_kv = last_shared_kv
            .ok_or_else(|| anyhow!("Gemma4 drafter prefill produced no shared KV"))?;
        let projection_start = Instant::now();
        let first_logits = model.project_hidden_on(&last_prompt_hidden, ())?;
        add_elapsed_us(&mut stats.projection_us, projection_start);
        let mut prng_state = mlx::random::key(request.sampler.seed)?;
        let sampling_start = Instant::now();
        let first_tokens = sample_logits_positions(
            &first_logits,
            request.sampler,
            &request.prompt_ids,
            &mut prng_state,
        )?;
        add_elapsed_us(&mut stats.sampling_us, sampling_start);
        let first_token = *first_tokens
            .first()
            .ok_or_else(|| anyhow!("Gemma4 drafter prefill produced no first token"))?;

        let mut history = request.prompt_ids.clone();
        history.push(first_token);
        let mut pending_tokens = VecDeque::new();
        pending_tokens.push_back(first_token);

        Ok(Self {
            model,
            drafter,
            prefix_cache,
            cache,
            history,
            request,
            cfg,
            pending_tokens,
            detok: tokenizer.decode_stream(true),
            last_hidden: last_prompt_hidden,
            shared_kv,
            emitted_new_tokens: 0,
            finished: false,
            dummy_position_ids,
            prng_state,
            adaptive_draft_tokens: cfg.max_draft_tokens,
            stats,
            trace_window_limit: 0,
            trace_windows: Vec::new(),
        })
    }

    pub fn stats(&self) -> MtpSpeculativeStats {
        self.stats.clone()
    }

    pub fn set_trace_window_limit(&mut self, limit: usize) {
        self.trace_window_limit = limit;
        self.trace_windows.truncate(limit);
    }

    pub fn trace_windows(&self) -> &[Gemma4DrafterTraceWindow] {
        &self.trace_windows
    }

    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }

        let token = self
            .pending_tokens
            .pop_front()
            .ok_or_else(|| anyhow!("Gemma4 drafter stream invariant: pending queue is empty"))?;
        self.emitted_new_tokens += 1;
        let text = self.detok.step(token)?.unwrap_or_default();
        let finish_reason = if self.request.stop_token_ids.contains(&token) {
            Some("stop")
        } else if self.emitted_new_tokens >= self.request.max_new_tokens {
            Some("length")
        } else {
            None
        };

        if finish_reason.is_some() {
            self.finished = true;
            return Ok(Some(GenerateEvent {
                token,
                text,
                finish_reason,
            }));
        }

        if self.pending_tokens.is_empty() {
            self.fill_window(token)?;
        }

        Ok(Some(GenerateEvent {
            token,
            text,
            finish_reason: None,
        }))
    }

    fn fill_window(&mut self, current_token: u32) -> Result<()> {
        let remaining = self
            .request
            .max_new_tokens
            .saturating_sub(self.emitted_new_tokens);
        if remaining == 0 {
            return Ok(());
        }

        let draft_budget = self
            .adaptive_draft_tokens
            .clamp(1, self.cfg.max_draft_tokens)
            .min(remaining);
        let draft_tokens = self.draft_tokens(current_token, draft_budget)?;
        let verify_input = verify_input(current_token, &draft_tokens);
        let verify_start_pos = (self.history.len() - 1) as i32;
        let verify_pos_ids = self.position_ids(verify_start_pos, verify_input.len() as i32)?;
        let verify_arr: Array =
            (&verify_input[..], &[1_i32, verify_input.len() as i32][..]).try_into()?;

        let base_snapshot: Vec<LayerCacheSnapshot> =
            self.cache.iter().map(LayerCache::snapshot).collect();
        let verify_forward_start = Instant::now();
        let verified = self.model.forward_text_hidden_with_shared_kv_on(
            &verify_arr,
            &verify_pos_ids,
            None,
            None,
            Some(&mut self.cache),
            (),
        )?;
        self.prefix_cache
            .refresh_active_kv_residency_stats(&self.cache);
        add_elapsed_us(&mut self.stats.verify_forward_us, verify_forward_start);
        let projection_start = Instant::now();
        let verified_logits = self.model.project_hidden_on(&verified.hidden, ())?;
        add_elapsed_us(&mut self.stats.projection_us, projection_start);
        let sampling_start = Instant::now();
        let verified_tokens = sample_logits_positions(
            &verified_logits,
            self.request.sampler,
            &self.history,
            &mut self.prng_state,
        )?;
        add_elapsed_us(&mut self.stats.sampling_us, sampling_start);

        let resolution = resolve_speculative_tokens(&draft_tokens, &verified_tokens)?;
        if self.trace_windows.len() < self.trace_window_limit {
            self.trace_windows.push(Gemma4DrafterTraceWindow {
                history_len: self.history.len(),
                verify_start_pos,
                draft_tokens: draft_tokens.clone(),
                verified_tokens: verified_tokens.clone(),
                accepted_draft_len: resolution.accepted_draft_len,
            });
        }
        self.stats.windows += 1;
        self.stats.drafted_tokens += draft_tokens.len();
        self.stats.accepted_draft_tokens += resolution.accepted_draft_len;
        self.stats
            .record_window_acceptance(draft_tokens.len(), resolution.accepted_draft_len);
        if resolution.needs_rollback {
            self.stats.rollback_count += 1;
        }
        adjust_mtp_draft_budget(
            self.cfg.max_draft_tokens,
            &mut self.adaptive_draft_tokens,
            draft_tokens.len(),
            resolution.accepted_draft_len,
            &mut self.stats,
        );

        let (accepted_last_hidden, accepted_shared_kv) = if resolution.needs_rollback {
            let rollback_start = Instant::now();
            restore_layer_cache(&mut self.cache, &base_snapshot)?;
            self.prefix_cache
                .refresh_active_kv_residency_stats(&self.cache);
            add_elapsed_us(&mut self.stats.main_rollback_us, rollback_start);
            let replay_len = resolution.accepted_verify_input_len;
            let replay_input = &verify_input[..replay_len];
            let replay_arr: Array = (replay_input, &[1_i32, replay_len as i32][..]).try_into()?;
            let replay_pos_ids = self.position_ids(verify_start_pos, replay_len as i32)?;
            let replay_forward_start = Instant::now();
            let replay = self.model.forward_text_hidden_with_shared_kv_on(
                &replay_arr,
                &replay_pos_ids,
                None,
                None,
                Some(&mut self.cache),
                (),
            )?;
            self.prefix_cache
                .refresh_active_kv_residency_stats(&self.cache);
            add_elapsed_us(&mut self.stats.verify_forward_us, replay_forward_start);
            (
                slice_hidden_position(&replay.hidden, replay_len as i32 - 1)?,
                replay.shared_kv,
            )
        } else {
            (
                slice_hidden_position(
                    &verified.hidden,
                    resolution.accepted_verify_input_len as i32 - 1,
                )?,
                verified.shared_kv,
            )
        };
        self.last_hidden = accepted_last_hidden;
        self.shared_kv = accepted_shared_kv;

        let mut tokens_to_append = resolution.tokens_to_append;
        if let Some(stop_idx) = tokens_to_append
            .iter()
            .position(|token| self.request.stop_token_ids.contains(token))
        {
            tokens_to_append.truncate(stop_idx + 1);
        }
        tokens_to_append.truncate(remaining);
        for token in tokens_to_append {
            self.history.push(token);
            self.pending_tokens.push_back(token);
        }
        self.prefix_cache
            .refresh_active_kv_residency_stats(&self.cache);
        Ok(())
    }

    fn draft_tokens(&mut self, current_token: u32, draft_budget: usize) -> Result<Vec<u32>> {
        let mut draft_tokens = Vec::with_capacity(draft_budget);
        let mut draft_history = self.history.clone();
        let mut input_hidden = self.last_hidden.clone();
        let mut input_token = current_token;
        let kv_valid_len = (self.history.len() - 1) as i32;
        let draft_position = draft_position_for_shared_kv(kv_valid_len);

        for _ in 0..draft_budget {
            let token_arr: Array = (&[input_token][..], &[1_i32, 1_i32][..]).try_into()?;
            let token_embed = self.model.embed_on(&token_arr, ())?;
            let inputs_embeds =
                mlx::ops::shape::concatenate_on(&[&token_embed, &input_hidden], 2, ())?;
            let draft_forward_start = Instant::now();
            let output = self.drafter.forward_on(
                &inputs_embeds,
                &self.shared_kv,
                draft_position,
                kv_valid_len,
                (),
            )?;
            add_elapsed_us(&mut self.stats.draft_forward_us, draft_forward_start);
            let sampling_start = Instant::now();
            let sampled = sample_logits_positions(
                &output.logits,
                self.request.sampler,
                &draft_history,
                &mut self.prng_state,
            )?;
            add_elapsed_us(&mut self.stats.sampling_us, sampling_start);
            let next_token = *sampled
                .first()
                .ok_or_else(|| anyhow!("Gemma4 drafter produced no token"))?;
            draft_tokens.push(next_token);
            draft_history.push(next_token);
            input_hidden = output.hidden_states;
            input_token = next_token;
        }

        Ok(draft_tokens)
    }

    fn position_ids(&self, start_pos: i32, len: i32) -> Result<Array> {
        match self.dummy_position_ids.as_ref() {
            Some(dummy) => Ok(dummy.clone()),
            None => build_position_ids(start_pos, len),
        }
    }
}

struct MaskedEmbedder {
    centroids: Linear,
    token_ordering: Array,
    hidden_size: i32,
    vocab_size: i32,
    num_centroids: i32,
    top_k: i32,
    vocab_size_per_centroid: i32,
}

impl MaskedEmbedder {
    fn from_loader(loader: &Loader, cfg: &Gemma4AssistantConfig) -> Result<Self> {
        let num_centroids = cfg
            .num_centroids
            .ok_or_else(|| anyhow!("Gemma4 MaskedEmbedder: num_centroids missing"))?;
        let top_k = cfg
            .centroid_intermediate_top_k
            .ok_or_else(|| anyhow!("Gemma4 MaskedEmbedder: centroid_intermediate_top_k missing"))?;
        let vocab_size = cfg.text_config.vocab_size;
        if vocab_size % num_centroids != 0 {
            return Err(anyhow!(
                "Gemma4 MaskedEmbedder: vocab_size {vocab_size} not divisible by num_centroids {num_centroids}"
            ));
        }
        Ok(Self {
            centroids: Linear::from_loader(loader, "masked_embedding.centroids")?,
            token_ordering: loader.tensor("masked_embedding.token_ordering")?.clone(),
            hidden_size: cfg.text_config.hidden_size,
            vocab_size,
            num_centroids,
            top_k,
            vocab_size_per_centroid: vocab_size / num_centroids,
        })
    }

    fn forward_on(
        &self,
        hidden_states: &Array,
        lm_head_weight: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = hidden_states.shape();
        let dims = shape.as_slice();
        if dims.len() != 3 {
            return Err(anyhow!(
                "Gemma4 MaskedEmbedder: expected hidden [B,S,H], got {dims:?}"
            ));
        }
        let (b, seq, h) = (dims[0], dims[1], dims[2]);
        if h != self.hidden_size {
            return Err(anyhow!(
                "Gemma4 MaskedEmbedder: hidden size {h} != {}",
                self.hidden_size
            ));
        }
        let centroid_logits = self.centroids.forward_on(hidden_states, target)?;
        let partition = mlx::ops::sort::argpartition_on(&centroid_logits, -self.top_k, -1, target)?;
        let c = centroid_logits.shape_at(2);
        let topk_idx = mlx::ops::indexing::slice_strided_on(
            &partition,
            &[0_i32, 0, c - self.top_k][..],
            &[b, seq, c][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        let ordering = self
            .token_ordering
            .reshape_on((self.num_centroids, self.vocab_size_per_centroid), target)?;
        let selected_canonical = ordering.take_on(&topk_idx, 0, target)?;
        let selected = self.top_k * self.vocab_size_per_centroid;
        let flat_idx = selected_canonical.reshape_on((b * seq * selected,), target)?;
        let selected_emb = lm_head_weight
            .take_on(&flat_idx, 0, target)?
            .reshape_on((b, seq, selected, self.hidden_size), target)?;
        let hidden4 = hidden_states.reshape_on((b, seq, 1_i32, self.hidden_size), target)?;
        let selected_t = selected_emb.transpose_axes_on(&[0_i32, 1, 3, 2][..], target)?;
        let selected_logits = hidden4
            .matmul_on(&selected_t, target)?
            .reshape_on((b, seq, selected), target)?;
        let min = mlx::ops::reduction::min_on(&selected_logits, mlx::ops::All, false, target)?;
        let mask_value = &min - 1.0_f32;
        let full = &Array::zeros_on((b, seq, self.vocab_size), hidden_states.dtype(), target)?
            + &mask_value;
        mlx::ops::indexing::put_along_axis_on(
            &full,
            &selected_canonical.reshape_on((b, seq, selected), target)?,
            &selected_logits,
            -1,
            target,
        )
        .map_err(anyhow::Error::from)
    }
}

fn make_drafter_masks(
    shared_kv: &Gemma4SharedKvStates,
    query_len: i32,
    query_offset: i32,
    sliding_window: i32,
    dtype: Dtype,
    kv_valid_len: i32,
    target: StreamOrDevice,
) -> Result<Gemma4DrafterMasks> {
    let sliding = match shared_kv.get(Gemma4LayerKind::Sliding) {
        Some(kv) => {
            let len = kv_len(kv)?;
            bidirectional_swa_mask_on(
                query_len,
                query_offset.min(len),
                len,
                sliding_window,
                Some(kv_valid_len.min(len)),
                0,
                dtype,
                target,
            )?
        }
        None => None,
    };
    let full = match shared_kv.get(Gemma4LayerKind::Full) {
        Some(kv) => {
            let len = kv_len(kv)?;
            let key_offset = (kv_valid_len - len).max(0);
            bidirectional_full_mask_on(
                query_len,
                len,
                Some(kv_valid_len),
                key_offset,
                dtype,
                target,
            )?
        }
        None => None,
    };
    Ok(Gemma4DrafterMasks { sliding, full })
}

fn kv_len(kv: &super::attention::SharedKv) -> Result<i32> {
    let shape = kv.keys.shape();
    let dims = shape.as_slice();
    if dims.len() != 4 {
        return Err(anyhow!("Gemma4 drafter expected K/V rank 4, got {dims:?}"));
    }
    Ok(dims[2])
}

fn draft_position_for_shared_kv(kv_valid_len: i32) -> i32 {
    (kv_valid_len - 1).max(0)
}

#[cfg(test)]
pub(crate) fn build_bidirectional_swa_mask_for_test(
    query_len: i32,
    query_offset: i32,
    kv_len: i32,
    window: i32,
    kv_valid_len: Option<i32>,
    key_offset: i32,
    dtype: Dtype,
) -> Result<Option<Array>> {
    bidirectional_swa_mask_on(
        query_len,
        query_offset,
        kv_len,
        window,
        kv_valid_len,
        key_offset,
        dtype,
        ().into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cache::KVCache;

    #[test]
    fn draft_position_uses_previous_target_hidden_position() {
        assert_eq!(draft_position_for_shared_kv(0), 0);
        assert_eq!(draft_position_for_shared_kv(1), 0);
        assert_eq!(draft_position_for_shared_kv(20_400), 20_399);
    }

    fn shared_kv_restore_config() -> Gemma4TextConfig {
        let mut cfg: crate::models::gemma4::Gemma4Config =
            serde_json::from_value(serde_json::json!({
                    "model_type": "gemma4",
                    "text_config": {
                        "hidden_size": 16,
                        "num_hidden_layers": 4,
                        "intermediate_size": 32,
                        "num_attention_heads": 4,
                        "head_dim": 4,
                        "vocab_size": 128,
                        "num_key_value_heads": 2,
                        "num_kv_shared_layers": 2,
                        "hidden_size_per_layer_input": 0,
                        "layer_types": [
                            "sliding_attention",
                            "full_attention",
                            "sliding_attention",
                            "full_attention"
                        ],
                        "tie_word_embeddings": true
                    }
            }))
            .unwrap();
        cfg.validate_and_finalize().unwrap();
        cfg.text_config
    }

    fn paged_layer_cache(seed: f32) -> LayerCache {
        let mut kv = KVCache::new(1, 2, 4, 4, Dtype::Float32, 8).with_step(4);
        kv.enable_paged(2, 16).unwrap();
        let k_values: Vec<f32> = (0..24).map(|idx| seed + idx as f32).collect();
        let v_values: Vec<f32> = (0..24).map(|idx| seed + 100.0 + idx as f32).collect();
        let k: Array = (&k_values[..], &[1_i32, 2, 3, 4][..]).try_into().unwrap();
        let v: Array = (&v_values[..], &[1_i32, 2, 3, 4][..]).try_into().unwrap();
        kv.update_and_fetch(&k, &v, &[3]).unwrap();
        LayerCache::Full(kv)
    }

    #[test]
    fn shared_kv_restore_materializes_gemma4_paged_cache_by_layer_kind() {
        let cfg = shared_kv_restore_config();
        let cache = vec![paged_layer_cache(1.0), paged_layer_cache(1000.0)];

        let restored = gemma4_shared_kv_from_cache_on(&cfg, &cache, ()).unwrap();
        let sliding = restored.require(Gemma4LayerKind::Sliding).unwrap();
        let full = restored.require(Gemma4LayerKind::Full).unwrap();

        assert_eq!(sliding.keys.shape().as_slice(), &[1_i32, 2, 3, 4]);
        assert_eq!(full.keys.shape().as_slice(), &[1_i32, 2, 3, 4]);
        assert_eq!(sliding.keys.to_vec::<f32>().unwrap()[0], 1.0);
        assert_eq!(sliding.values.to_vec::<f32>().unwrap()[0], 101.0);
        assert_eq!(full.keys.to_vec::<f32>().unwrap()[0], 1000.0);
        assert_eq!(full.values.to_vec::<f32>().unwrap()[0], 1100.0);
    }

    #[test]
    fn shared_kv_restore_rejects_wrong_cache_layer_count() {
        let cfg = shared_kv_restore_config();
        let cache = vec![paged_layer_cache(1.0)];

        let err = match gemma4_shared_kv_from_cache_on(&cfg, &cache, ()) {
            Ok(_) => panic!("layer count mismatch should fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("cache.len()=1"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn prefix_cache_rejects_lru_without_paged_prefix_config() {
        let err = match Gemma4DrafterPrefixCache::new(
            None,
            Some(PrefixLruCacheConfig::new(1024).unwrap()),
            None,
        ) {
            Ok(_) => panic!("LRU without prefix store should fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("requires paged prefix cache"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn prefix_cache_accepts_paged_prefix_and_lru_configs() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-gemma4-drafter-prefix-cache-test-{}",
            std::process::id()
        ));
        let config = PagedPrefixCacheConfig::new(root, "gemma4-test", 2, 16).unwrap();
        let cache = Gemma4DrafterPrefixCache::new(
            Some(config),
            Some(PrefixLruCacheConfig::new(1024 * 1024).unwrap()),
            None,
        )
        .unwrap();

        assert!(cache.is_enabled());
        assert!(cache.prefix_lru_cache().is_some());
    }

    #[test]
    fn prefix_cache_active_kv_enables_paged_hot_cold_runtime_storage() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-gemma4-drafter-active-kv-test-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let config =
            PagedPrefixCacheConfig::new(root.join("prefix"), "gemma4-test", 2, 16).unwrap();
        let active_config = ActiveKvOffloadConfig::enabled(root.join("active"))
            .with_hot_window_pages_override(Some(1))
            .with_chunk_pages_override(Some(1));
        let stats = ActiveKvOffloadSharedStats::new(&active_config);
        let active = Gemma4DrafterActiveKvRuntime::new(active_config, stats.clone(), 1024, 1);
        let prefix_cache = Gemma4DrafterPrefixCache::new(Some(config), None, active).unwrap();

        let mut cache = vec![LayerCache::Full(
            KVCache::new(1, 2, 4, 4, Dtype::Float32, 8).with_step(4),
        )];
        prefix_cache
            .enable_runtime_cache_storage(&mut cache, None, 8, 1)
            .unwrap();

        let LayerCache::Full(kv) = &cache[0] else {
            panic!("expected full cache")
        };
        assert!(kv.paged().is_some());
        assert!(kv.paged_hot_cold_summary().is_some());
        assert!(stats.snapshot().enabled);
    }
}

fn bidirectional_full_mask_on(
    query_len: i32,
    kv_len: i32,
    kv_valid_len: Option<i32>,
    key_offset: i32,
    dtype: Dtype,
    target: StreamOrDevice,
) -> Result<Option<Array>> {
    let Some(valid_len) = kv_valid_len else {
        return Ok(None);
    };
    if key_offset + kv_len <= valid_len {
        return Ok(None);
    }
    let mut flat = vec![f32::NEG_INFINITY; query_len as usize * kv_len as usize];
    for q in 0..query_len {
        let base = q as usize * kv_len as usize;
        for k in 0..kv_len {
            if key_offset + k < valid_len {
                flat[base + k as usize] = 0.0;
            }
        }
    }
    let arr: Array = (&flat[..], &[1_i32, 1, query_len, kv_len][..]).try_into()?;
    Ok(Some(mlx::ops::cast::astype_on(&arr, dtype, target)?))
}

#[allow(clippy::too_many_arguments)]
fn bidirectional_swa_mask_on(
    query_len: i32,
    query_offset: i32,
    kv_len: i32,
    window: i32,
    kv_valid_len: Option<i32>,
    key_offset: i32,
    dtype: Dtype,
    target: StreamOrDevice,
) -> Result<Option<Array>> {
    if kv_len <= 0 || query_len <= 0 || window <= 0 {
        return Err(anyhow!(
            "Gemma4 drafter mask: query_len={query_len} kv_len={kv_len} window={window}"
        ));
    }
    if kv_len <= window
        && query_offset - key_offset < window
        && key_offset + kv_len - (query_offset + query_len) < window
        && kv_valid_len.is_none_or(|valid| key_offset + kv_len <= valid)
    {
        return Ok(None);
    }

    let valid_len = kv_valid_len.unwrap_or(i32::MAX);
    let mut flat = vec![f32::NEG_INFINITY; query_len as usize * kv_len as usize];
    for q in 0..query_len {
        let q_abs = query_offset + q;
        let base = q as usize * kv_len as usize;
        for k in 0..kv_len {
            let k_abs = key_offset + k;
            let dist = q_abs - k_abs;
            if dist > -window && dist < window && k_abs < valid_len {
                flat[base + k as usize] = 0.0;
            }
        }
    }
    let arr: Array = (&flat[..], &[1_i32, 1, query_len, kv_len][..]).try_into()?;
    Ok(Some(mlx::ops::cast::astype_on(&arr, dtype, target)?))
}
