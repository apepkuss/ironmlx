//! Runtime Gemma4 drafter generation and prefix residency adapters.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use anyhow::anyhow;
use mlx::{Array, Dtype};

use crate::core::generation_types::{GenerateEvent, GenerateRequest};
use crate::core::scheduler::paged_prefix_fingerprint_for_request;
use crate::core::speculative::{
    add_elapsed_us, elapsed_us_since, resolve_exact_deterministic_target_logits,
    resolve_speculative_tokens, sample_draft_logits_position,
    sample_draft_logits_position_with_uniform, sample_logits_positions, slice_hidden_position,
    split_speculative_draft_prng, trim_full_layer_cache_rows_to_accepted_prefix, verify_input,
    DraftTokenDistribution, Gemma4DrafterPolicyState, MtpDraftPolicyKvState, MtpDraftPolicyWindow,
    MtpSpeculativeConfig, MtpSpeculativeStats,
};
use crate::Result;
use ironmlx_core::sampler::draw_uniforms;
use ironmlx_lm::core::model::Model;
use ironmlx_lm::core::vision::DenseVlMethods;
use {
    crate::core::cache::ActiveKvOffloadConfig, crate::core::cache::ActiveKvOffloadSharedStats,
    crate::core::cache::ActiveKvResidencySummary, crate::core::cache::AsyncPrefixStoreAdmission,
    crate::core::cache::AsyncPrefixStoreCancellation, crate::core::cache::AsyncPrefixStoreSubmit,
    crate::core::cache::PagedPrefixCacheConfig, crate::core::cache::PagedPrefixLoadStatus,
    crate::core::cache::PagedPrefixStore, crate::core::cache::PrefixLruCache,
    crate::core::cache::PrefixLruCacheConfig, crate::core::cache::PrefixLruInsertStatus,
    ironmlx_lm::core::cache::turboquant_kv::TurboQuantKVBits,
};
use {
    ironmlx_lm::core::cache::layer::enable_paged_hot_cold_tiering_caches,
    ironmlx_lm::core::cache::layer::enable_paged_kv_caches,
    ironmlx_lm::core::cache::layer::enable_turboquant_kv_caches,
    ironmlx_lm::core::cache::layer::prefix_entry_for_row,
    ironmlx_lm::core::cache::layer::prefix_key_spec_for_caches,
    ironmlx_lm::core::cache::layer::restore_prefix_entry_for_row,
    ironmlx_lm::core::cache::layer::LayerCache, ironmlx_lm::core::cache::layer::LayerCacheSnapshot,
};
use {
    ironmlx_lm::core::cache::prefix_payload::PagedPrefixEntry,
    ironmlx_lm::core::cache::prefix_payload::PagedPrefixEntryStats,
    ironmlx_lm::core::cache::prefix_payload::PrefixTensorSpec,
};
use {
    ironmlx_lm::core::model_input::build_position_ids,
    ironmlx_lm::core::model_input::build_position_ids_vl,
    ironmlx_lm::core::model_input::count_image_pad,
    ironmlx_lm::core::model_input::extend_vl_chunk_end_for_image_pad,
    ironmlx_lm::core::model_input::slice_pos_ids_axis2,
    ironmlx_lm::core::model_input::slice_vision_embeds_rows,
};
use {ironmlx_lm::core::tokenizer::DecodeStream, ironmlx_lm::core::tokenizer::Tokenizer};

use ironmlx_lm::models::gemma4::Gemma4Model;
use ironmlx_lm::models::gemma4::Gemma4SharedKvStates;

use {
    ironmlx_lm::models::gemma4::draft_position_for_shared_kv,
    ironmlx_lm::models::gemma4::gemma4_shared_kv_from_cache_on,
    ironmlx_lm::models::gemma4::shared_kv_row_trim_suffix_on,
    ironmlx_lm::models::gemma4::Gemma4AssistantModel,
};

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

    pub(crate) fn new_with_shared_prefix_lru(
        config: Option<PagedPrefixCacheConfig>,
        prefix_lru_cache: Option<Gemma4DrafterPrefixLruHandle>,
        active_kv: Option<Gemma4DrafterActiveKvRuntime>,
    ) -> Self {
        Self {
            config,
            prefix_lru_cache,
            active_kv,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.is_some()
    }

    fn active_kv_enabled(&self) -> bool {
        self.active_kv.is_some()
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
        let hot_cold = crate::core::page_storage::file_paged_kv_config(
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

    pub(crate) fn try_restore(
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

    pub(crate) fn try_save(
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

        let store = config.store();
        let key = PagedPrefixStore::key_for(&spec);
        let permit = match crate::core::cache::process_async_prefix_store_queue().try_admit(
            store,
            spec.clone(),
            AsyncPrefixStoreCancellation::default(),
        ) {
            AsyncPrefixStoreAdmission::Admitted(permit) => *permit,
            AsyncPrefixStoreAdmission::Coalesced => return Ok(Some(key)),
            AsyncPrefixStoreAdmission::Backpressured => {
                tracing::warn!(
                    key,
                    "paged SSD prefix cache Gemma4 drafter extraction skipped by async-store backpressure"
                );
                return Ok(None);
            }
            AsyncPrefixStoreAdmission::Closed => {
                return Err(anyhow!("paged SSD prefix cache async store is closed"));
            }
        };

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
        match permit.submit(entry) {
            AsyncPrefixStoreSubmit::Queued => {
                tracing::debug!(
                    "paged SSD prefix cache Gemma4 drafter queued: key={} tokens={} cached_len={} payload_bytes={} tensors={}",
                    key,
                    prompt_ids.len(),
                    stats.cached_len,
                    stats.payload_bytes,
                    stats.tensor_count,
                );
                Ok(Some(key))
            }
            AsyncPrefixStoreSubmit::Coalesced => Ok(Some(key)),
            AsyncPrefixStoreSubmit::Cancelled => Ok(None),
            AsyncPrefixStoreSubmit::Backpressured => {
                tracing::warn!(
                    key,
                    "paged SSD prefix cache Gemma4 drafter skipped by async-store backpressure"
                );
                Ok(None)
            }
            AsyncPrefixStoreSubmit::Closed => {
                Err(anyhow!("paged SSD prefix cache async store is closed"))
            }
        }
    }
}

pub(crate) struct Gemma4DrafterPrefixRestore {
    pub(crate) cached_len: i32,
    pub(crate) last_hidden: Array,
    pub(crate) shared_kv: Gemma4SharedKvStates,
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
    let mut cached_lengths = vec![max_cached_len];
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
    spec: &ironmlx_lm::core::cache::prefix_payload::PagedPrefixKeySpec,
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
    spec: ironmlx_lm::core::cache::prefix_payload::PagedPrefixKeySpec,
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

pub(crate) fn gemma4_cache_row_cached_len(cache: &[LayerCache], row: usize) -> Result<Option<i32>> {
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
    draft_policy: Gemma4DrafterPolicyState,
    stats: MtpSpeculativeStats,
    trace_window_limit: usize,
    trace_windows: Vec<Gemma4DrafterTraceWindow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4DrafterTraceWindow {
    pub history_len: usize,
    pub verify_start_pos: i32,
    pub draft_tokens: Vec<u32>,
    /// Greedy target tokens at every verify position. Empty for sampled exact
    /// windows because the target at each position is a distribution.
    pub verified_tokens: Vec<u32>,
    /// Tokens selected by acceptance, correction, or full-accept bonus.
    pub resolved_tokens: Vec<u32>,
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
        if request.pixel_values.is_none() && request.image_grid_thw.is_some() {
            return Err(anyhow!(
                "Gemma4DrafterGenerationStream::new: image_grid_thw present but pixel_values is None"
            ));
        }

        let prompt_len = request.prompt_ids.len();
        let cap = ((prompt_len + request.max_new_tokens) as i32)
            .max(ironmlx_lm::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
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
            draft_policy: Gemma4DrafterPolicyState::new(cfg.max_draft_tokens),
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

        let window_started = Instant::now();
        let stats_before_window = self.stats.clone();
        let timing_before = self.stats.draft_cap_timing();
        let context_tokens = self.history.len();

        let draft_budget = effective_draft_budget(
            self.adaptive_draft_tokens,
            self.cfg.max_draft_tokens,
            remaining,
        );
        let (draft_tokens, _draft_distributions) =
            self.draft_tokens(current_token, draft_budget)?;
        let verify_input = verify_input(current_token, &draft_tokens);
        let verify_start_pos = (self.history.len() - 1) as i32;
        let verify_pos_ids = self.position_ids(verify_start_pos, verify_input.len() as i32)?;
        let verify_arr: Array =
            (&verify_input[..], &[1_i32, verify_input.len() as i32][..]).try_into()?;

        let base_snapshot: Vec<LayerCacheSnapshot> = self
            .cache
            .iter()
            .map(LayerCache::append_snapshot)
            .collect::<Result<_>>()?;
        let verify_forward_start = Instant::now();
        let position_stable_verify = verify_input.len() > 1;
        let _position_stable_linear =
            position_stable_verify.then(ironmlx_lm::nn::position_stable_linear_scope);
        let _position_stable_qmm =
            position_stable_verify.then(ironmlx_lm::nn::position_stable_qmm_scope);
        let stable_attention = position_stable_verify
            && context_tokens > self.model.config().sliding_window.max(0) as usize;
        let _stable_attention =
            stable_attention.then(ironmlx_lm::nn::gemma4_verify_attention_scope);
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
        let (resolution, verified_tokens) = if self.request.sampler.is_pipelinable() {
            let verified_tokens = sample_logits_positions(
                &verified_logits,
                self.request.sampler,
                &self.history,
                &mut self.prng_state,
            )?;
            (
                resolve_speculative_tokens(&draft_tokens, &verified_tokens)?,
                verified_tokens,
            )
        } else {
            (
                resolve_exact_deterministic_target_logits(
                    &draft_tokens,
                    &verified_logits,
                    self.request.sampler,
                    &self.history,
                    &mut self.prng_state,
                )?,
                Vec::new(),
            )
        };
        add_elapsed_us(&mut self.stats.sampling_us, sampling_start);

        let accepted_draft_len = resolution.accepted_draft_len;
        let rollback_count = usize::from(resolution.needs_rollback);
        if self.trace_windows.len() < self.trace_window_limit {
            self.trace_windows.push(Gemma4DrafterTraceWindow {
                history_len: self.history.len(),
                verify_start_pos,
                draft_tokens: draft_tokens.clone(),
                verified_tokens,
                resolved_tokens: resolution.tokens_to_append.clone(),
                accepted_draft_len: resolution.accepted_draft_len,
            });
        }
        self.stats.windows += 1;
        self.stats.drafted_tokens += draft_tokens.len();
        self.stats.accepted_draft_tokens += resolution.accepted_draft_len;
        self.stats
            .record_exact_sampling(resolution.exact_sampling());
        self.stats
            .record_window_acceptance(draft_tokens.len(), resolution.accepted_draft_len);
        if resolution.needs_rollback {
            self.stats.rollback_count += 1;
        }
        let accepted_len = resolution.accepted_verify_input_len;
        let accepted_last_hidden =
            slice_hidden_position(&verified.hidden, i32::try_from(accepted_len)? - 1)?;
        let accepted_shared_kv = if resolution.needs_rollback {
            let rollback_start = Instant::now();
            trim_full_layer_cache_rows_to_accepted_prefix(
                &mut self.cache,
                &base_snapshot,
                &[(0, accepted_len)],
            )?;
            self.prefix_cache
                .refresh_active_kv_residency_stats(&self.cache);
            add_elapsed_us(&mut self.stats.main_rollback_us, rollback_start);
            let rejected_len = verify_input
                .len()
                .checked_sub(accepted_len)
                .ok_or_else(|| {
                    anyhow!(
                    "Gemma4 drafter accepted verify length {accepted_len} exceeds input length {}",
                    verify_input.len()
                )
                })?;
            shared_kv_row_trim_suffix_on(&verified.shared_kv, 0, rejected_len, ())?
        } else {
            verified.shared_kv
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
        let committed_tokens = tokens_to_append.len();
        for token in tokens_to_append {
            self.history.push(token);
            self.pending_tokens.push_back(token);
        }
        self.prefix_cache
            .refresh_active_kv_residency_stats(&self.cache);
        let total_us = elapsed_us_since(window_started);
        let stats_delta = self.stats.saturating_delta_since(&stats_before_window);
        let change = self
            .draft_policy
            .observe_window(MtpDraftPolicyWindow::from_stats_delta(
                draft_tokens.len(),
                accepted_draft_len,
                committed_tokens,
                total_us,
                context_tokens,
                1,
                MtpDraftPolicyKvState::from_runtime(
                    self.prefix_cache.is_enabled(),
                    self.prefix_cache.active_kv_enabled(),
                ),
                &stats_delta,
            ));
        if change.reduced {
            self.stats.draft_budget_reductions =
                self.stats.draft_budget_reductions.saturating_add(1);
        } else if change.increased {
            self.stats.draft_budget_increases = self.stats.draft_budget_increases.saturating_add(1);
        }
        self.adaptive_draft_tokens = self.draft_policy.current_budget();
        let timing_delta = self
            .stats
            .draft_cap_timing()
            .saturating_delta_since(timing_before);
        self.stats.record_draft_cap_observation(
            self.cfg.max_draft_tokens,
            &[draft_tokens.len()],
            &[context_tokens],
            accepted_draft_len,
            committed_tokens,
            rollback_count,
            total_us,
            timing_delta,
        );
        Ok(())
    }

    fn draft_tokens(
        &mut self,
        current_token: u32,
        draft_budget: usize,
    ) -> Result<(Vec<u32>, Vec<DraftTokenDistribution>)> {
        let mut draft_tokens = Vec::with_capacity(draft_budget);
        let mut draft_distributions = Vec::with_capacity(draft_budget);
        let mut draft_history = self.history.clone();
        let mut input_hidden = self.last_hidden.clone();
        let mut input_token = current_token;
        let kv_valid_len = (self.history.len() - 1) as i32;
        let draft_position = draft_position_for_shared_kv(kv_valid_len);
        let draft_uniforms = if self.request.sampler.is_pipelinable() {
            vec![0.0; draft_budget]
        } else {
            let mut draft_prng = split_speculative_draft_prng(&mut self.prng_state)?;
            draw_uniforms(&mut draft_prng, draft_budget)?
        };

        for &draft_uniform in draft_uniforms.iter().take(draft_budget) {
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
            let (next_token, distribution) = if self.request.sampler.is_pipelinable() {
                sample_draft_logits_position(
                    &output.logits,
                    self.request.sampler,
                    &draft_history,
                    None,
                )?
            } else {
                sample_draft_logits_position_with_uniform(
                    &output.logits,
                    self.request.sampler,
                    &draft_history,
                    draft_uniform,
                )?
            };
            add_elapsed_us(&mut self.stats.sampling_us, sampling_start);
            draft_tokens.push(next_token);
            draft_distributions.push(distribution);
            draft_history.push(next_token);
            input_hidden = output.hidden_states;
            input_token = next_token;
        }

        Ok((draft_tokens, draft_distributions))
    }

    fn position_ids(&self, start_pos: i32, len: i32) -> Result<Array> {
        match self.dummy_position_ids.as_ref() {
            Some(dummy) => Ok(dummy.clone()),
            None => build_position_ids(start_pos, len),
        }
    }
}

pub(crate) fn effective_draft_budget(
    adaptive: usize,
    configured: usize,
    remaining: usize,
) -> usize {
    adaptive.min(configured).min(remaining)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironmlx_lm::core::cache::kv_cache::KVCache;
    #[test]
    fn standalone_drafter_can_fall_back_to_ordinary_decode() {
        assert_eq!(effective_draft_budget(0, 4, 32), 0);
        assert_eq!(effective_draft_budget(3, 4, 2), 2);
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
