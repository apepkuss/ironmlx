use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::{
    PagedPrefixEntry, PagedPrefixEntryStats, PagedPrefixKeySpec, PagedPrefixStore,
    PrefixLayerPayload, PrefixMtpLayerPayload,
};

const ACTIVE_KV_PROFILE: &str = "active_kv_offload_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveKvResidencyState {
    Resident,
    Offloaded,
    Loading,
    Dirty,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveKvPageResidency {
    pub request_id: u64,
    pub page_index: usize,
    pub state: ActiveKvResidencyState,
    pub path: Option<PathBuf>,
    pub offloaded_bytes: usize,
}

impl ActiveKvPageResidency {
    pub fn resident(request_id: u64, page_index: usize) -> Self {
        Self {
            request_id,
            page_index,
            state: ActiveKvResidencyState::Resident,
            path: None,
            offloaded_bytes: 0,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.state = ActiveKvResidencyState::Dirty;
    }

    pub fn mark_loading(&mut self) {
        self.state = ActiveKvResidencyState::Loading;
    }

    pub fn mark_offloaded(&mut self, path: PathBuf, offloaded_bytes: usize) {
        self.state = ActiveKvResidencyState::Offloaded;
        self.path = Some(path);
        self.offloaded_bytes = offloaded_bytes;
    }

    pub fn mark_resident(&mut self) {
        self.state = ActiveKvResidencyState::Resident;
        self.path = None;
        self.offloaded_bytes = 0;
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveKvResidencySummary {
    pub resident_pages: usize,
    pub offloaded_pages: usize,
    pub loading_pages: usize,
    pub dirty_pages: usize,
    pub offloaded_bytes: usize,
    pub swap_out_count: u64,
    pub swap_in_count: u64,
    pub stream_read_count: u64,
}

#[derive(Debug, Default)]
pub struct ActiveKvResidencyTracker {
    pages: HashMap<(u64, usize), ActiveKvPageResidency>,
}

impl ActiveKvResidencyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, page: ActiveKvPageResidency) {
        self.pages.insert((page.request_id, page.page_index), page);
    }

    pub fn remove_request(&mut self, request_id: u64) {
        self.pages.retain(|(id, _), _| *id != request_id);
    }

    pub fn summary(&self) -> ActiveKvResidencySummary {
        let mut summary = ActiveKvResidencySummary::default();
        for page in self.pages.values() {
            match page.state {
                ActiveKvResidencyState::Resident => summary.resident_pages += 1,
                ActiveKvResidencyState::Offloaded => {
                    summary.offloaded_pages += 1;
                    summary.offloaded_bytes =
                        summary.offloaded_bytes.saturating_add(page.offloaded_bytes);
                }
                ActiveKvResidencyState::Loading => summary.loading_pages += 1,
                ActiveKvResidencyState::Dirty => summary.dirty_pages += 1,
            }
        }
        summary
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveKvLayerChunkKind {
    FullDense,
    FullPaged,
    FullTurboQuantPacked,
    Mla,
    GatedDeltaLinear,
    MtpSpeculativeSideCache,
}

impl ActiveKvLayerChunkKind {
    pub fn is_supported_for_active_offload(self) -> bool {
        matches!(
            self,
            Self::FullDense
                | Self::FullPaged
                | Self::FullTurboQuantPacked
                | Self::Mla
                | Self::GatedDeltaLinear
                | Self::MtpSpeculativeSideCache
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ActiveKvLayerChunkPayload<'a> {
    FullDense {
        k: &'a mlx::Array,
        v: &'a mlx::Array,
    },
    FullPaged {
        k_pages: &'a mlx::Array,
        v_pages: &'a mlx::Array,
    },
    FullTurboQuantPacked {
        k_packed: &'a mlx::Array,
        k_norms: &'a mlx::Array,
        v_packed: &'a mlx::Array,
        v_norms: &'a mlx::Array,
    },
    GatedDeltaLinear {
        conv_state: &'a mlx::Array,
        recurrent_state: &'a mlx::Array,
    },
    Mla {
        c_kv: &'a mlx::Array,
        k_pe: &'a mlx::Array,
    },
    MtpSpeculativeSideCache {
        k: &'a mlx::Array,
        v: &'a mlx::Array,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct ActiveKvLayerChunk<'a> {
    pub layer_index: usize,
    pub is_main_layer: bool,
    pub kind: ActiveKvLayerChunkKind,
    pub payload: ActiveKvLayerChunkPayload<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct ActiveKvEntryChunkReader<'a> {
    entry: &'a PagedPrefixEntry,
}

impl<'a> ActiveKvEntryChunkReader<'a> {
    pub fn new(entry: &'a PagedPrefixEntry) -> Self {
        Self { entry }
    }

    pub fn chunks(&self) -> impl Iterator<Item = ActiveKvLayerChunk<'a>> + '_ {
        self.entry
            .main_layers
            .iter()
            .enumerate()
            .map(main_layer_chunk)
            .chain(
                self.entry
                    .mtp_layers
                    .iter()
                    .enumerate()
                    .map(mtp_layer_chunk),
            )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveKvOffloadConfig {
    pub enabled: bool,
    pub root: PathBuf,
    pub hot_window_pages_override: Option<i32>,
    pub chunk_pages_override: Option<i32>,
}

impl ActiveKvOffloadConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            root: default_active_kv_offload_dir(),
            hot_window_pages_override: None,
            chunk_pages_override: None,
        }
    }

    pub fn enabled(root: impl Into<PathBuf>) -> Self {
        Self {
            enabled: true,
            root: root.into(),
            hot_window_pages_override: None,
            chunk_pages_override: None,
        }
    }

    pub fn with_hot_window_pages_override(mut self, hot_window_pages: Option<i32>) -> Self {
        self.hot_window_pages_override = hot_window_pages;
        self
    }

    pub fn with_chunk_pages_override(mut self, chunk_pages: Option<i32>) -> Self {
        self.chunk_pages_override = chunk_pages;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ActiveKvStoredPayload {
    pub request_id: u64,
    pub cached_token_ids: Vec<u32>,
    pub cached_len: i32,
    pub key: String,
    pub path: PathBuf,
    pub spec: PagedPrefixKeySpec,
    pub stats: PagedPrefixEntryStats,
}

#[derive(Debug, Clone)]
pub struct ActiveKvOffloadStore {
    root: PathBuf,
    store: PagedPrefixStore,
}

impl ActiveKvOffloadStore {
    pub fn new(config: ActiveKvOffloadConfig) -> Result<Self> {
        anyhow::ensure!(
            config.enabled,
            "ActiveKvOffloadStore requires enabled config"
        );
        Ok(Self {
            store: PagedPrefixStore::new(&config.root),
            root: config.root,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn save(
        &self,
        request_id: u64,
        cached_token_ids: &[u32],
        cached_len: i32,
        entry: &PagedPrefixEntry,
    ) -> Result<ActiveKvStoredPayload> {
        anyhow::ensure!(cached_len > 0, "active KV cached_len must be positive");
        anyhow::ensure!(
            cached_token_ids.len() == cached_len as usize,
            "active KV token length {} does not match cached_len {cached_len}",
            cached_token_ids.len()
        );
        anyhow::ensure!(
            !entry.main_layers.is_empty() || !entry.mtp_layers.is_empty(),
            "active KV payload must contain at least one cache layer"
        );

        let token_ids = cached_token_ids
            .iter()
            .map(|&id| {
                i32::try_from(id)
                    .with_context(|| format!("active KV token id {id} exceeds i32::MAX"))
            })
            .collect::<Result<Vec<_>>>()?;
        let block_size = infer_block_size(entry).unwrap_or(1);
        let spec = PagedPrefixKeySpec {
            entry_kind: super::PrefixEntryKind::WholePrefix,
            model_id: format!("active-kv-request-{request_id}"),
            token_ids,
            cached_len,
            fingerprint: Some(uuid::Uuid::new_v4().simple().to_string()),
            block_size,
            kv_cache_profile: Some(ACTIVE_KV_PROFILE.to_owned()),
            main_layers: entry.main_layer_specs(),
            mtp_layers: entry.mtp_layer_specs(),
            mtp_last_hidden: entry.mtp_last_hidden_spec(),
            gemma4_drafter_last_hidden: entry.gemma4_drafter_last_hidden_spec(),
        };
        let key = self.store.save(&spec, entry)?;
        let path = self.root.join(&key);
        let stats = entry.observability_stats(cached_len);
        Ok(ActiveKvStoredPayload {
            request_id,
            cached_token_ids: cached_token_ids.to_vec(),
            cached_len,
            key,
            path,
            spec,
            stats,
        })
    }

    pub fn load(&self, payload: &ActiveKvStoredPayload) -> Result<PagedPrefixEntry> {
        self.store
            .load(&payload.spec)?
            .ok_or_else(|| anyhow::anyhow!("active KV payload {} is missing", payload.key))
    }

    pub fn remove(&self, payload: &ActiveKvStoredPayload) -> Result<()> {
        match fs::remove_dir_all(&payload.path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err)
                .with_context(|| format!("remove active KV payload {}", payload.path.display())),
        }
    }

    pub fn cleanup_all(&self) -> Result<()> {
        match fs::remove_dir_all(&self.root) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err)
                .with_context(|| format!("cleanup active KV offload root {}", self.root.display())),
        }
    }
}

#[derive(Debug, Default)]
struct ActiveKvOffloadStatsInner {
    enabled: AtomicBool,
    parked_requests: AtomicUsize,
    resident_pages: AtomicUsize,
    loading_pages: AtomicUsize,
    dirty_pages: AtomicUsize,
    parked_offloaded_pages: AtomicUsize,
    parked_offloaded_bytes: AtomicUsize,
    residency_offloaded_pages: AtomicUsize,
    residency_offloaded_bytes: AtomicUsize,
    residency_swap_out_count: AtomicU64,
    residency_swap_in_count: AtomicU64,
    residency_stream_read_count: AtomicU64,
    swap_out_count: AtomicU64,
    swap_in_count: AtomicU64,
    swap_error_count: AtomicU64,
    last_swap_out_us: AtomicU64,
    last_swap_in_us: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveKvOffloadStatus {
    Disabled,
    Idle,
    Active,
    Degraded,
}

#[derive(Debug, Clone)]
pub struct ActiveKvOffloadSharedStats {
    inner: Arc<ActiveKvOffloadStatsInner>,
    storage_dir: Arc<PathBuf>,
}

impl ActiveKvOffloadSharedStats {
    pub fn new(config: &ActiveKvOffloadConfig) -> Self {
        let inner = ActiveKvOffloadStatsInner::default();
        inner.enabled.store(config.enabled, Ordering::Relaxed);
        Self {
            inner: Arc::new(inner),
            storage_dir: Arc::new(config.root.clone()),
        }
    }

    pub fn record_swap_out(&self, stats: PagedPrefixEntryStats, elapsed_us: u64) {
        self.inner.swap_out_count.fetch_add(1, Ordering::Relaxed);
        self.inner
            .parked_offloaded_bytes
            .fetch_add(stats.payload_bytes, Ordering::Relaxed);
        self.inner
            .parked_offloaded_pages
            .fetch_add(stats.full_paged_pages, Ordering::Relaxed);
        self.inner
            .last_swap_out_us
            .store(elapsed_us, Ordering::Relaxed);
    }

    pub fn record_swap_in(&self, stats: PagedPrefixEntryStats, elapsed_us: u64) {
        self.inner.swap_in_count.fetch_add(1, Ordering::Relaxed);
        subtract_saturating(&self.inner.parked_offloaded_bytes, stats.payload_bytes);
        subtract_saturating(&self.inner.parked_offloaded_pages, stats.full_paged_pages);
        self.inner
            .last_swap_in_us
            .store(elapsed_us, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.inner.swap_error_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_residency_summary(&self, summary: ActiveKvResidencySummary) {
        self.inner
            .resident_pages
            .store(summary.resident_pages, Ordering::Relaxed);
        self.inner
            .residency_offloaded_pages
            .store(summary.offloaded_pages, Ordering::Relaxed);
        self.inner
            .loading_pages
            .store(summary.loading_pages, Ordering::Relaxed);
        self.inner
            .dirty_pages
            .store(summary.dirty_pages, Ordering::Relaxed);
        self.inner
            .residency_offloaded_bytes
            .store(summary.offloaded_bytes, Ordering::Relaxed);
        self.inner
            .residency_swap_out_count
            .store(summary.swap_out_count, Ordering::Relaxed);
        self.inner
            .residency_swap_in_count
            .store(summary.swap_in_count, Ordering::Relaxed);
        self.inner
            .residency_stream_read_count
            .store(summary.stream_read_count, Ordering::Relaxed);
    }

    pub fn set_parked_requests(&self, count: usize) {
        self.inner.parked_requests.store(count, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> ActiveKvOffloadHealth {
        let parked_offloaded_pages = self.inner.parked_offloaded_pages.load(Ordering::Relaxed);
        let residency_offloaded_pages =
            self.inner.residency_offloaded_pages.load(Ordering::Relaxed);
        let parked_offloaded_bytes = self.inner.parked_offloaded_bytes.load(Ordering::Relaxed);
        let residency_offloaded_bytes =
            self.inner.residency_offloaded_bytes.load(Ordering::Relaxed);
        let parked_swap_out_count = self.inner.swap_out_count.load(Ordering::Relaxed);
        let parked_swap_in_count = self.inner.swap_in_count.load(Ordering::Relaxed);
        let residency_swap_out_count = self.inner.residency_swap_out_count.load(Ordering::Relaxed);
        let residency_swap_in_count = self.inner.residency_swap_in_count.load(Ordering::Relaxed);
        let residency_stream_read_count = self
            .inner
            .residency_stream_read_count
            .load(Ordering::Relaxed);
        let mut health = ActiveKvOffloadHealth {
            enabled: self.inner.enabled.load(Ordering::Relaxed),
            status: ActiveKvOffloadStatus::Disabled,
            active: false,
            degraded: false,
            mode: "request_preemption_hot_cold_tiering",
            storage_dir: self.storage_dir.as_ref().clone(),
            resident_pages: self.inner.resident_pages.load(Ordering::Relaxed),
            offloaded_pages: parked_offloaded_pages.saturating_add(residency_offloaded_pages),
            loading_pages: self.inner.loading_pages.load(Ordering::Relaxed),
            dirty_pages: self.inner.dirty_pages.load(Ordering::Relaxed),
            parked_requests: self.inner.parked_requests.load(Ordering::Relaxed),
            offloaded_bytes: parked_offloaded_bytes.saturating_add(residency_offloaded_bytes),
            swap_out_count: parked_swap_out_count.saturating_add(residency_swap_out_count),
            swap_in_count: parked_swap_in_count.saturating_add(residency_swap_in_count),
            stream_read_count: residency_stream_read_count,
            swap_error_count: self.inner.swap_error_count.load(Ordering::Relaxed),
            last_swap_out_us: self.inner.last_swap_out_us.load(Ordering::Relaxed),
            last_swap_in_us: self.inner.last_swap_in_us.load(Ordering::Relaxed),
            supported_cache_kinds: vec![
                "full_attention_dense",
                "full_attention_paged",
                "turboquant_full_attention_packed",
                "mla",
                "gated_delta_linear",
                "mtp_speculative_side_cache",
            ],
            not_applicable_cache_kinds: Vec::new(),
        };
        health.refresh_status();
        health
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveKvOffloadHealth {
    pub enabled: bool,
    pub status: ActiveKvOffloadStatus,
    pub active: bool,
    pub degraded: bool,
    pub mode: &'static str,
    pub storage_dir: PathBuf,
    pub resident_pages: usize,
    pub offloaded_pages: usize,
    pub loading_pages: usize,
    pub dirty_pages: usize,
    pub parked_requests: usize,
    pub offloaded_bytes: usize,
    pub swap_out_count: u64,
    pub swap_in_count: u64,
    pub stream_read_count: u64,
    pub swap_error_count: u64,
    pub last_swap_out_us: u64,
    pub last_swap_in_us: u64,
    pub supported_cache_kinds: Vec<&'static str>,
    pub not_applicable_cache_kinds: Vec<&'static str>,
}

impl ActiveKvOffloadHealth {
    pub fn disabled() -> Self {
        ActiveKvOffloadSharedStats::new(&ActiveKvOffloadConfig::disabled()).snapshot()
    }

    pub fn aggregate(snapshots: impl IntoIterator<Item = Self>) -> Self {
        let mut aggregate = Self::disabled();
        for snapshot in snapshots {
            aggregate.enabled |= snapshot.enabled;
            if snapshot.enabled {
                aggregate.storage_dir = snapshot.storage_dir;
            }
            aggregate.resident_pages += snapshot.resident_pages;
            aggregate.offloaded_pages += snapshot.offloaded_pages;
            aggregate.loading_pages += snapshot.loading_pages;
            aggregate.dirty_pages += snapshot.dirty_pages;
            aggregate.parked_requests += snapshot.parked_requests;
            aggregate.offloaded_bytes += snapshot.offloaded_bytes;
            aggregate.swap_out_count += snapshot.swap_out_count;
            aggregate.swap_in_count += snapshot.swap_in_count;
            aggregate.stream_read_count += snapshot.stream_read_count;
            aggregate.swap_error_count += snapshot.swap_error_count;
            aggregate.last_swap_out_us = aggregate.last_swap_out_us.max(snapshot.last_swap_out_us);
            aggregate.last_swap_in_us = aggregate.last_swap_in_us.max(snapshot.last_swap_in_us);
        }
        aggregate.refresh_status();
        aggregate
    }

    fn refresh_status(&mut self) {
        self.degraded = self.enabled && self.swap_error_count > 0;
        self.active = self.enabled
            && (self.offloaded_pages > 0
                || self.loading_pages > 0
                || self.parked_requests > 0
                || self.swap_out_count > 0
                || self.swap_in_count > 0
                || self.stream_read_count > 0);
        self.status = if !self.enabled {
            ActiveKvOffloadStatus::Disabled
        } else if self.degraded {
            ActiveKvOffloadStatus::Degraded
        } else if self.active {
            ActiveKvOffloadStatus::Active
        } else {
            ActiveKvOffloadStatus::Idle
        };
    }
}

pub fn default_active_kv_offload_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ironmlx")
        .join("cache")
        .join("active_kv_offload")
}

pub fn timed<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, u64)> {
    let start = Instant::now();
    let value = f()?;
    let elapsed_us = start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
    Ok((value, elapsed_us))
}

fn infer_block_size(entry: &PagedPrefixEntry) -> Option<i32> {
    entry.main_layers.iter().find_map(|layer| match layer {
        PrefixLayerPayload::FullPaged { k_pages, .. } => {
            let shape = k_pages.shape();
            shape.as_slice().get(2).copied().filter(|value| *value > 0)
        }
        _ => None,
    })
}

fn main_layer_chunk<'a>(
    (layer_index, layer): (usize, &'a PrefixLayerPayload),
) -> ActiveKvLayerChunk<'a> {
    match layer {
        PrefixLayerPayload::FullDense { k, v } => ActiveKvLayerChunk {
            layer_index,
            is_main_layer: true,
            kind: ActiveKvLayerChunkKind::FullDense,
            payload: ActiveKvLayerChunkPayload::FullDense { k, v },
        },
        PrefixLayerPayload::FullPaged { k_pages, v_pages } => ActiveKvLayerChunk {
            layer_index,
            is_main_layer: true,
            kind: ActiveKvLayerChunkKind::FullPaged,
            payload: ActiveKvLayerChunkPayload::FullPaged { k_pages, v_pages },
        },
        PrefixLayerPayload::FullTurboQuantPacked {
            k_packed,
            k_norms,
            v_packed,
            v_norms,
        } => ActiveKvLayerChunk {
            layer_index,
            is_main_layer: true,
            kind: ActiveKvLayerChunkKind::FullTurboQuantPacked,
            payload: ActiveKvLayerChunkPayload::FullTurboQuantPacked {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            },
        },
        PrefixLayerPayload::Linear {
            conv_state,
            recurrent_state,
        } => ActiveKvLayerChunk {
            layer_index,
            is_main_layer: true,
            kind: ActiveKvLayerChunkKind::GatedDeltaLinear,
            payload: ActiveKvLayerChunkPayload::GatedDeltaLinear {
                conv_state,
                recurrent_state,
            },
        },
        PrefixLayerPayload::Mla { c_kv, k_pe } => ActiveKvLayerChunk {
            layer_index,
            is_main_layer: true,
            kind: ActiveKvLayerChunkKind::Mla,
            payload: ActiveKvLayerChunkPayload::Mla { c_kv, k_pe },
        },
    }
}

fn mtp_layer_chunk<'a>(
    (layer_index, layer): (usize, &'a PrefixMtpLayerPayload),
) -> ActiveKvLayerChunk<'a> {
    ActiveKvLayerChunk {
        layer_index,
        is_main_layer: false,
        kind: ActiveKvLayerChunkKind::MtpSpeculativeSideCache,
        payload: ActiveKvLayerChunkPayload::MtpSpeculativeSideCache {
            k: &layer.k,
            v: &layer.v,
        },
    }
}

fn subtract_saturating(value: &AtomicUsize, amount: usize) {
    let mut current = value.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_sub(amount);
        match value.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => current = actual,
        }
    }
}
