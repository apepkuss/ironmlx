use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use mlx::{Array, Dtype};
use serde::{Deserialize, Serialize};

use crate::core::chat_prefix::ChatTurnBoundary;
use crate::Result;

const SCHEMA_VERSION: u32 = 7;
const META_FILE: &str = "meta.json";
const PAYLOAD_FILE: &str = "payload.safetensors";

use ironmlx_lm::core::cache::layer::{
    export_prefix_segment_for_row, export_prefix_segment_range_for_row, row_prefix_lineage,
    LayerCache,
};
use ironmlx_lm::core::cache::prefix_payload::*;
use ironmlx_lm::core::cache::prefix_segments::{
    materialize_prefix_entry, prefix_segment_layout, PrefixKvSegment, PrefixLineage,
    PrefixLineageRef, PrefixSegmentId, PrefixSegmentLayer, PrefixSegmentLayout, PrefixSegmentRef,
    SegmentedPrefixEntry,
};

mod wp8;
#[cfg(test)]
mod wp8_tests;
use wp8::*;
pub use wp8::{PagedPrefixSegmentedLoadResult, PrefixSegmentLookup};

#[derive(Debug, Clone)]
pub struct PagedPrefixStore {
    root: PathBuf,
    max_bytes: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedPrefixLoadStatus {
    Hit,
    MissingEntry,
    InvalidMetadata,
    MetadataMismatch,
    PayloadReadFailed,
    PayloadInvalid,
    EntryInvalid,
}

#[derive(Debug, Clone)]
pub struct PagedPrefixLoadResult {
    pub key: String,
    pub status: PagedPrefixLoadStatus,
    pub entry: Option<PagedPrefixEntry>,
    pub stats: Option<PagedPrefixEntryStats>,
}

pub const DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE: i32 = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedPrefixCacheConfig {
    /// Directory of the SSD store. `None` keeps prefixes in memory only (the
    /// prefix LRU cache); nothing is written to or read from disk.
    pub root: Option<PathBuf>,
    pub model_id: String,
    pub block_size: i32,
    pub max_pages: i32,
    pub max_disk_bytes: Option<usize>,
    /// Where a chat prompt's next turn diverges: a prompt ending in its
    /// template's generation suffix is saved without it (see
    /// [`ChatTurnBoundary`]).
    turn_boundary: Option<ChatTurnBoundary>,
}

impl PagedPrefixCacheConfig {
    pub fn new(
        root: impl AsRef<Path>,
        model_id: impl Into<String>,
        block_size: i32,
        max_pages: i32,
    ) -> Result<Self> {
        Self::new_with_max_disk_bytes(root, model_id, block_size, max_pages, None)
    }

    pub fn new_with_max_disk_bytes(
        root: impl AsRef<Path>,
        model_id: impl Into<String>,
        block_size: i32,
        max_pages: i32,
        max_disk_bytes: Option<usize>,
    ) -> Result<Self> {
        let config = Self {
            root: Some(root.as_ref().to_path_buf()),
            model_id: model_id.into(),
            block_size,
            max_pages,
            max_disk_bytes,
            turn_boundary: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// Paged K/V without an SSD store: prefixes are reused only through the
    /// in-memory prefix LRU cache.
    pub fn memory_only(
        model_id: impl Into<String>,
        block_size: i32,
        max_pages: i32,
    ) -> Result<Self> {
        let config = Self {
            root: None,
            model_id: model_id.into(),
            block_size,
            max_pages,
            max_disk_bytes: None,
            turn_boundary: None,
        };
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn with_turn_boundary(mut self, turn_boundary: Option<ChatTurnBoundary>) -> Self {
        self.turn_boundary = turn_boundary;
        self
    }

    /// The length of `prompt_ids` a follow-up chat turn shares, when the
    /// prompt ends in its template's generation suffix.
    pub(crate) fn turn_reusable_len(&self, prompt_ids: &[u32]) -> Option<usize> {
        self.turn_boundary
            .as_ref()
            .and_then(|boundary| boundary.reusable_len(prompt_ids))
    }

    /// The SSD store, when this cache has one.
    pub fn store(&self) -> Option<PagedPrefixStore> {
        self.root
            .as_ref()
            .map(|root| PagedPrefixStore::new(root).with_optional_max_bytes(self.max_disk_bytes))
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_id.is_empty() {
            anyhow::bail!("PagedPrefixCacheConfig: model_id must not be empty");
        }
        if self.block_size <= 0 {
            anyhow::bail!(
                "PagedPrefixCacheConfig: block_size must be > 0, got {}",
                self.block_size
            );
        }
        if self.max_pages <= 0 {
            anyhow::bail!(
                "PagedPrefixCacheConfig: max_pages must be > 0, got {}",
                self.max_pages
            );
        }
        if self.max_disk_bytes == Some(0) {
            anyhow::bail!("PagedPrefixCacheConfig: max_disk_bytes must be > 0");
        }
        Ok(())
    }
}

/// Where a DFlash2 engine also keeps its prefix cache on disk (under the
/// `dflash2` subdirectory of `dir`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlash2PrefixDiskConfig {
    pub dir: PathBuf,
    /// Bytes the store may hold; unlimited when `None`.
    pub max_bytes: Option<u64>,
    /// The target model's directory: stored state is reused only for the
    /// same weights.
    pub model_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixLruCacheConfig {
    pub max_bytes: usize,
}

impl PrefixLruCacheConfig {
    pub fn new(max_bytes: usize) -> Result<Self> {
        let config = Self { max_bytes };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_bytes == 0 {
            anyhow::bail!("PrefixLruCacheConfig: max_bytes must be > 0");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixLruInsertStatus {
    Stored,
    Replaced,
    SkippedOversized,
}

#[derive(Debug, Clone)]
pub struct PrefixLruInsertResult {
    pub key: String,
    pub status: PrefixLruInsertStatus,
    /// The entry's logical size (`payload_bytes` counts every token it
    /// restores, shared or not).
    pub stats: PagedPrefixEntryStats,
    /// Bytes this insert added to the cache: segments it brought in plus its
    /// private data. Zero when skipped.
    pub charged_bytes: usize,
}

#[derive(Debug, Clone)]
struct PrefixLruEntry {
    spec: PagedPrefixKeySpec,
    segmented: SegmentedPrefixEntry,
    private_bytes: usize,
    stats: PagedPrefixEntryStats,
    generation: u64,
}

/// A segment held by the cache, counted once however many entries use it.
#[derive(Debug)]
struct SegmentSlot {
    segment: Arc<PrefixKvSegment>,
    entry_refs: usize,
    bytes: usize,
}

#[derive(Debug)]
pub struct PrefixLruCache {
    max_bytes: usize,
    /// Each held segment once plus each entry's private bytes.
    total_bytes: usize,
    generation: u64,
    entries: HashMap<String, PrefixLruEntry>,
    segments: HashMap<PrefixSegmentId, SegmentSlot>,
    recency: VecDeque<(String, u64)>,
}

/// Distinct segment ids an entry uses, in order.
fn entry_segment_ids(entry: &SegmentedPrefixEntry) -> Vec<PrefixSegmentId> {
    let mut ids = Vec::with_capacity(entry.segments.len());
    for r in &entry.segments {
        if !ids.contains(&r.segment.id) {
            ids.push(r.segment.id);
        }
    }
    ids
}

pub type SharedPrefixLruCache = Arc<Mutex<PrefixLruCache>>;

/// What the hot prefix cache holds, as `/healthz` reports it.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct PrefixLruHealth {
    pub entry_count: usize,
    pub segment_count: usize,
    /// Every held segment's bytes, each counted once.
    pub segment_bytes: usize,
    /// The entries' logical payload bytes summed, i.e. what charging each
    /// entry a whole copy would cost.
    pub logical_bytes: usize,
}

/// [`PrefixLruCache::health`] of the process-wide hot prefix cache; all zero
/// when no runtime created one.
pub fn process_prefix_lru_health() -> PrefixLruHealth {
    process_prefix_lru_cache()
        .ok()
        .flatten()
        .and_then(|cache| cache.lock().ok().map(|cache| cache.health()))
        .unwrap_or_default()
}

/// Most segments one cached entry references. A save that would inherit
/// more copies the row's whole aligned prefix into one new segment instead,
/// so restoring never concatenates an unbounded number of pieces.
pub const MAX_PREFIX_ENTRY_SEGMENTS: usize = 16;

/// The record of which segments `entry` references, in order.
pub fn segmented_entry_lineage(
    entry: &SegmentedPrefixEntry,
    layout: PrefixSegmentLayout,
) -> PrefixLineage {
    PrefixLineage {
        layout,
        refs: entry
            .segments
            .iter()
            .map(|r| PrefixLineageRef {
                id: r.segment.id,
                start: r.segment.start,
                stop: r.stop,
                segment_stop: r.segment.stop,
            })
            .collect(),
    }
}

fn lock_shared_prefix_lru_cache(
    prefix_lru_cache: &SharedPrefixLruCache,
) -> Result<std::sync::MutexGuard<'_, PrefixLruCache>> {
    prefix_lru_cache
        .lock()
        .map_err(|_| anyhow::anyhow!("prefix LRU cache lock poisoned"))
}

/// Export row `row` of `caches` as an entry for `spec`. The leading segments
/// the row's K/V came from (its lineage) that `prefix_lru_cache` still holds
/// are referenced, not copied; only the rest of the block-aligned prefix
/// becomes one new segment, and the tail and unshareable layers go to the
/// private entry. Every exported array is owned and evaluated. Returns the
/// entry and the row's cached length.
pub fn export_segmented_prefix_for_row(
    prefix_lru_cache: Option<&SharedPrefixLruCache>,
    caches: &[LayerCache],
    row: usize,
    spec: &PagedPrefixKeySpec,
) -> Result<Option<(SegmentedPrefixEntry, i32)>> {
    let layout = prefix_segment_layout(spec);
    let block_size = spec.block_size;
    let mut segments = match row_prefix_lineage(caches, row) {
        Some(lineage) if lineage.layout == layout => lineage_segments_for_row(
            prefix_lru_cache,
            caches,
            row,
            &lineage.truncated(spec.cached_len, block_size),
            block_size,
        )?,
        _ => Vec::new(),
    };
    if segments.len() >= MAX_PREFIX_ENTRY_SEGMENTS {
        segments.clear();
    }
    let from_len = segments.last().map_or(0, |r| r.stop);
    let Some((segment, private, cached_len)) =
        export_prefix_segment_for_row(caches, row, from_len, block_size, layout)?
    else {
        return Ok(None);
    };
    if let Some(segment) = segment {
        let stop = segment.stop;
        segments.push(PrefixSegmentRef {
            segment: Arc::new(segment),
            stop,
        });
    }
    let aligned_len = segments.last().map_or(0, |r| r.stop);
    Ok(Some((
        SegmentedPrefixEntry {
            segments,
            aligned_len,
            private,
        },
        cached_len,
    )))
}

/// The segments `lineage` names, in order, as far as they continue each
/// other. A segment the hot cache holds is reused. One it no longer holds is
/// copied again from the row under the same id when the row holds its whole
/// range (the row's K/V there came from it), so the SSD store finds its file
/// and does not write it again. Stops at the first segment it can do neither
/// for.
fn lineage_segments_for_row(
    prefix_lru_cache: Option<&SharedPrefixLruCache>,
    caches: &[LayerCache],
    row: usize,
    lineage: &PrefixLineage,
    block_size: i32,
) -> Result<Vec<PrefixSegmentRef>> {
    // Short lock: only looks segments up; copies run outside it.
    let held = match prefix_lru_cache {
        Some(prefix_lru_cache) => {
            let cache = lock_shared_prefix_lru_cache(prefix_lru_cache)?;
            lineage
                .refs
                .iter()
                .map(|r| cache.segment(r.id))
                .collect::<Vec<_>>()
        }
        None => vec![None; lineage.refs.len()],
    };
    let mut refs = Vec::with_capacity(lineage.refs.len());
    let mut pos = 0;
    for (r, held) in lineage.refs.iter().zip(held) {
        if r.start != pos || r.stop <= r.start {
            break;
        }
        let segment = match held {
            Some(segment) => {
                if segment.start != r.start
                    || r.stop > segment.stop
                    || segment.layout != lineage.layout
                {
                    break;
                }
                segment
            }
            None => {
                if !r.is_whole() {
                    break;
                }
                let Some(segment) = export_prefix_segment_range_for_row(
                    caches,
                    row,
                    r.id,
                    r.start,
                    r.stop,
                    block_size,
                    lineage.layout.clone(),
                )?
                else {
                    break;
                };
                Arc::new(segment)
            }
        };
        refs.push(PrefixSegmentRef {
            segment,
            stop: r.stop,
        });
        pos = r.stop;
    }
    Ok(refs)
}

/// The lineage to record on a row restored from `entry` (`None` when the
/// entry references no segment).
pub fn restored_row_lineage(
    lineage: &PrefixLineage,
    cached_len: i32,
    block_size: i32,
) -> Option<PrefixLineage> {
    let lineage = lineage.truncated(cached_len, block_size);
    (!lineage.refs.is_empty()).then_some(lineage)
}

/// A hot-cache hit. Restore it with `restore_segmented_prefix_entry_for_rows`,
/// which copies the shared segments without joining them first.
#[derive(Debug)]
pub struct PrefixLruLoaded {
    pub key: String,
    pub entry: SegmentedPrefixEntry,
    /// The segments a row restored from `entry` holds (`None`: none).
    pub lineage: Option<PrefixLineage>,
    pub stats: PagedPrefixEntryStats,
}

/// A hot-cache hit for exactly `spec`: the entry, the lineage a row restored
/// from it gets, and its stats.
pub fn load_segmented_prefix_entry(
    prefix_lru_cache: &SharedPrefixLruCache,
    spec: &PagedPrefixKeySpec,
) -> Result<Option<PrefixLruLoaded>> {
    let Some((segmented, lineage)) =
        lock_shared_prefix_lru_cache(prefix_lru_cache)?.load_segmented(spec)?
    else {
        return Ok(None);
    };
    // Lazy: only reads shapes.
    let stats = materialize_prefix_entry(&segmented, spec.cached_len, spec.block_size)?
        .observability_stats(spec.cached_len);
    Ok(Some(PrefixLruLoaded {
        key: PagedPrefixStore::key_for(spec),
        entry: segmented,
        lineage: restored_row_lineage(&lineage, spec.cached_len, spec.block_size),
        stats,
    }))
}

/// An SSD hit for exactly `spec`.
#[derive(Debug)]
pub struct PrefixStoreHit {
    /// The entry, for restoring a row and for the prefix LRU cache.
    pub segmented: SegmentedPrefixEntry,
    /// The segments a row restored from it holds (`None`: none).
    pub lineage: Option<PrefixLineage>,
    pub stats: PagedPrefixEntryStats,
}

/// What an SSD lookup found.
#[derive(Debug)]
pub struct PrefixStoreLoaded {
    pub key: String,
    pub status: PagedPrefixLoadStatus,
    pub hit: Option<PrefixStoreHit>,
}

/// Read `spec` from the SSD store. Segments the prefix LRU cache still
/// holds are reused instead of read again.
pub fn load_store_prefix_entry(
    store: &PagedPrefixStore,
    prefix_lru_cache: Option<&SharedPrefixLruCache>,
    spec: &PagedPrefixKeySpec,
) -> Result<PrefixStoreLoaded> {
    let lookup = |id: PrefixSegmentId| {
        prefix_lru_cache
            .and_then(|cache| cache.lock().ok())
            .and_then(|cache| cache.segment(id))
    };
    let observed = store.load_segmented_observed(spec, Some(&lookup))?;
    let hit = match (observed.status, observed.entry) {
        (PagedPrefixLoadStatus::Hit, Some(segmented)) => {
            // Read from disk here, outside any lock: inserting the hit into
            // the hot cache evaluates it under the process-wide cache lock,
            // which is then a no-op. Segments reused from the cache are
            // evaluated already.
            for r in &segmented.segments {
                r.segment.eval()?;
            }
            segmented.private.eval()?;
            let stats = match observed.stats {
                Some(stats) => stats,
                // Lazy: only reads shapes.
                None => materialize_prefix_entry(&segmented, spec.cached_len, spec.block_size)?
                    .observability_stats(spec.cached_len),
            };
            let lineage = restored_row_lineage(
                &segmented_entry_lineage(&segmented, prefix_segment_layout(spec)),
                spec.cached_len,
                spec.block_size,
            );
            Some(PrefixStoreHit {
                segmented,
                lineage,
                stats,
            })
        }
        _ => None,
    };
    Ok(PrefixStoreLoaded {
        key: observed.key,
        status: if hit.is_some() {
            PagedPrefixLoadStatus::Hit
        } else if observed.status == PagedPrefixLoadStatus::Hit {
            PagedPrefixLoadStatus::EntryInvalid
        } else {
            observed.status
        },
        hit,
    })
}

static PROCESS_PREFIX_LRU_CACHE: OnceLock<Mutex<Weak<Mutex<PrefixLruCache>>>> = OnceLock::new();

/// Return the process-wide hot prefix cache for a configured global budget.
/// Every engine and scheduler charges and evicts against the same byte
/// counter. If independently-created runtimes request different limits, the
/// process adopts the smaller limit immediately; a later caller cannot grow a
/// budget that is already in use.
pub fn process_shared_prefix_lru_cache(
    config: PrefixLruCacheConfig,
) -> Result<SharedPrefixLruCache> {
    config.validate()?;
    let registry = PROCESS_PREFIX_LRU_CACHE.get_or_init(|| Mutex::new(Weak::new()));
    let mut registry = registry
        .lock()
        .map_err(|_| anyhow::anyhow!("process prefix LRU registry lock poisoned"))?;
    if let Some(cache) = registry.upgrade() {
        let mut cache_guard = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("process prefix LRU cache lock poisoned"))?;
        if config.max_bytes < cache_guard.max_bytes {
            cache_guard.max_bytes = config.max_bytes;
            cache_guard.shrink_to(config.max_bytes);
        }
        drop(cache_guard);
        return Ok(cache);
    }
    let cache = Arc::new(Mutex::new(PrefixLruCache::new(config)?));
    *registry = Arc::downgrade(&cache);
    Ok(cache)
}

/// What a reclaim took out of the hot prefix cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixLruReclaim {
    pub bytes: usize,
    /// Entries a longer cached entry still covers.
    pub covered_entries: usize,
    pub other_entries: usize,
}

/// The process-wide hot prefix cache, if any runtime has created it.
pub fn process_prefix_lru_cache() -> Result<Option<SharedPrefixLruCache>> {
    let Some(registry) = PROCESS_PREFIX_LRU_CACHE.get() else {
        return Ok(None);
    };
    Ok(registry
        .lock()
        .map_err(|_| anyhow::anyhow!("process prefix LRU registry lock poisoned"))?
        .upgrade())
}

/// [`PrefixLruCache::reclaim`] on the process-wide hot prefix cache.
pub fn reclaim_process_prefix_lru_bytes(
    needed_bytes: usize,
    spare_model: Option<&str>,
) -> Result<PrefixLruReclaim> {
    let Some(cache) = process_prefix_lru_cache()? else {
        return Ok(PrefixLruReclaim::default());
    };
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("process prefix LRU cache lock poisoned"))?;
    Ok(cache.reclaim(needed_bytes, spare_model))
}

const PROCESS_PREFIX_STORE_QUEUE_CAPACITY: usize = 4;
const PROCESS_PREFIX_STORE_PENDING_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct AsyncPrefixStoreStats {
    pub pending_jobs: usize,
    pub pending_bytes: usize,
    pub queued_total: u64,
    pub completed_total: u64,
    pub failed_total: u64,
    pub cancelled_total: u64,
    pub backpressured_total: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncPrefixStoreSubmit {
    Queued,
    Coalesced,
    Cancelled,
    Backpressured,
    Closed,
}

pub enum AsyncPrefixStoreAdmission {
    Admitted(Box<AsyncPrefixStorePermit>),
    Coalesced,
    Backpressured,
    Closed,
}

#[derive(Debug, Clone, Default)]
pub struct AsyncPrefixStoreCancellation(Arc<AtomicBool>);

impl AsyncPrefixStoreCancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug, Default)]
struct AsyncPrefixStoreCounters {
    pending_jobs: AtomicUsize,
    pending_bytes: AtomicUsize,
    queued_total: AtomicU64,
    completed_total: AtomicU64,
    failed_total: AtomicU64,
    cancelled_total: AtomicU64,
    backpressured_total: AtomicU64,
}

/// What a queued save writes: an immutable block (or a whole prefix kept as
/// one private part), or a whole prefix in its segmented form.
#[derive(Clone)]
enum AsyncPrefixStorePayload {
    Whole(PagedPrefixEntry),
    Segmented(SegmentedPrefixEntry),
}

impl AsyncPrefixStorePayload {
    /// The entry's logical bytes: every token it restores, shared or not.
    fn logical_bytes(&self, spec: &PagedPrefixKeySpec) -> Result<usize> {
        Ok(match self {
            Self::Whole(entry) => entry.observability_stats(spec.cached_len).payload_bytes,
            // Lazy: no array is evaluated.
            Self::Segmented(entry) => {
                materialize_prefix_entry(entry, spec.cached_len, spec.block_size)?
                    .observability_stats(spec.cached_len)
                    .payload_bytes
            }
        })
    }

    fn eval(&self) -> Result<()> {
        match self {
            Self::Whole(entry) => entry.eval()?,
            Self::Segmented(entry) => {
                for r in &entry.segments {
                    r.segment.eval()?;
                }
                entry.private.eval()?;
            }
        }
        Ok(())
    }

    fn segmented(&self) -> SegmentedPrefixEntry {
        match self {
            Self::Whole(entry) => SegmentedPrefixEntry {
                segments: Vec::new(),
                aligned_len: 0,
                private: entry.clone(),
            },
            Self::Segmented(entry) => entry.clone(),
        }
    }

    fn save_if_absent(&self, store: &PagedPrefixStore, spec: &PagedPrefixKeySpec) -> Result<()> {
        match self {
            Self::Whole(entry) => store.save_if_absent(spec, entry).map(drop),
            Self::Segmented(entry) => store.save_segmented_if_absent(spec, entry).map(drop),
        }
    }
}

struct AsyncPrefixStoreJob {
    id: u64,
    pending_key: (PathBuf, String),
    store: PagedPrefixStore,
    spec: PagedPrefixKeySpec,
    entry: AsyncPrefixStorePayload,
    payload_bytes: usize,
    cancellation: AsyncPrefixStoreCancellation,
}

#[derive(Clone)]
struct AsyncPrefixPendingEntry {
    id: u64,
    spec: PagedPrefixKeySpec,
    entry: Option<AsyncPrefixStorePayload>,
    cancellation: AsyncPrefixStoreCancellation,
}

struct AsyncPrefixStoreInner {
    capacity: usize,
    max_pending_bytes: usize,
    sender: Mutex<Option<SyncSender<Box<AsyncPrefixStoreJob>>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    counters: Arc<AsyncPrefixStoreCounters>,
    pending: Arc<Mutex<HashMap<(PathBuf, String), AsyncPrefixPendingEntry>>>,
    next_job_id: AtomicU64,
    idle: Arc<(Mutex<()>, Condvar)>,
}

impl Drop for AsyncPrefixStoreInner {
    fn drop(&mut self) {
        self.sender.get_mut().ok().and_then(Option::take);
        if let Some(worker) = self.worker.get_mut().ok().and_then(Option::take) {
            let _ = worker.join();
        }
    }
}

#[derive(Clone)]
pub struct AsyncPrefixStoreQueue(Arc<AsyncPrefixStoreInner>);

pub struct AsyncPrefixStorePermit {
    queue: AsyncPrefixStoreQueue,
    id: u64,
    pending_key: (PathBuf, String),
    store: Option<PagedPrefixStore>,
    spec: Option<PagedPrefixKeySpec>,
    payload_bytes: usize,
    cancellation: AsyncPrefixStoreCancellation,
    active: bool,
}

impl std::fmt::Debug for AsyncPrefixStorePermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AsyncPrefixStorePermit")
            .field("id", &self.id)
            .field("pending_key", &self.pending_key)
            .field("payload_bytes", &self.payload_bytes)
            .field("active", &self.active)
            .finish()
    }
}

impl Drop for AsyncPrefixStorePermit {
    fn drop(&mut self) {
        if self.active {
            self.queue
                .remove_pending_if_current(&self.pending_key, self.id);
            self.queue.release_pending(self.payload_bytes);
        }
    }
}

impl AsyncPrefixStorePermit {
    pub fn submit(mut self, entry: PagedPrefixEntry) -> AsyncPrefixStoreSubmit {
        let queue = self.queue.clone();
        queue.submit_permit(&mut self, AsyncPrefixStorePayload::Whole(entry))
    }
}

impl std::fmt::Debug for AsyncPrefixStoreQueue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AsyncPrefixStoreQueue")
            .field("capacity", &self.0.capacity)
            .field("max_pending_bytes", &self.0.max_pending_bytes)
            .field("stats", &self.stats())
            .finish()
    }
}

impl AsyncPrefixStoreQueue {
    pub fn new(capacity: usize, max_pending_bytes: usize) -> Result<Self> {
        anyhow::ensure!(
            capacity > 0,
            "async prefix store queue capacity must be > 0"
        );
        anyhow::ensure!(
            max_pending_bytes > 0,
            "async prefix store pending-byte limit must be > 0"
        );
        let (sender, receiver) = mpsc::sync_channel::<Box<AsyncPrefixStoreJob>>(capacity);
        let counters = Arc::new(AsyncPrefixStoreCounters::default());
        let worker_counters = Arc::clone(&counters);
        let pending = Arc::new(Mutex::new(HashMap::<
            (PathBuf, String),
            AsyncPrefixPendingEntry,
        >::new()));
        let worker_pending = Arc::clone(&pending);
        let idle = Arc::new((Mutex::new(()), Condvar::new()));
        let worker_idle = Arc::clone(&idle);
        let worker = std::thread::Builder::new()
            .name("ironmlx-prefix-store".to_owned())
            .spawn(move || {
                // MLX stream registries are thread-local. The async writer
                // receives GPU-backed arrays created by scheduler threads, so
                // it must own a real Metal stream before safetensors can
                // evaluate or copy those arrays.
                let worker_device = mlx::Device::gpu(0);
                mlx::set_default_device(worker_device);
                let worker_stream = mlx::default_stream(worker_device);
                mlx::set_default_stream(worker_stream);
                while let Ok(job) = receiver.recv() {
                    if job.cancellation.is_cancelled() {
                        worker_counters
                            .cancelled_total
                            .fetch_add(1, Ordering::Relaxed);
                    } else if let Err(error) = job.entry.save_if_absent(&job.store, &job.spec) {
                        worker_counters.failed_total.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(%error, "async paged prefix store write failed");
                    } else {
                        worker_counters
                            .completed_total
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    if let Ok(mut pending) = worker_pending.lock() {
                        if pending
                            .get(&job.pending_key)
                            .is_some_and(|entry| entry.id == job.id)
                        {
                            pending.remove(&job.pending_key);
                        }
                    }
                    if let Ok(_idle_guard) = worker_idle.0.lock() {
                        worker_counters.pending_jobs.fetch_sub(1, Ordering::AcqRel);
                        worker_counters
                            .pending_bytes
                            .fetch_sub(job.payload_bytes, Ordering::AcqRel);
                        worker_idle.1.notify_all();
                    } else {
                        worker_counters.pending_jobs.fetch_sub(1, Ordering::AcqRel);
                        worker_counters
                            .pending_bytes
                            .fetch_sub(job.payload_bytes, Ordering::AcqRel);
                    }
                }
                mlx::clear_streams();
            })
            .context("spawn async prefix store worker")?;
        Ok(Self(Arc::new(AsyncPrefixStoreInner {
            capacity,
            max_pending_bytes,
            sender: Mutex::new(Some(sender)),
            worker: Mutex::new(Some(worker)),
            counters,
            pending,
            next_job_id: AtomicU64::new(1),
            idle,
        })))
    }

    pub fn try_enqueue(
        &self,
        store: PagedPrefixStore,
        spec: PagedPrefixKeySpec,
        entry: PagedPrefixEntry,
    ) -> AsyncPrefixStoreSubmit {
        self.try_enqueue_cancellable(store, spec, entry, AsyncPrefixStoreCancellation::default())
    }

    pub fn try_enqueue_cancellable(
        &self,
        store: PagedPrefixStore,
        spec: PagedPrefixKeySpec,
        entry: PagedPrefixEntry,
        cancellation: AsyncPrefixStoreCancellation,
    ) -> AsyncPrefixStoreSubmit {
        match self.try_admit(store, spec, cancellation) {
            AsyncPrefixStoreAdmission::Admitted(permit) => (*permit).submit(entry),
            AsyncPrefixStoreAdmission::Coalesced => AsyncPrefixStoreSubmit::Coalesced,
            AsyncPrefixStoreAdmission::Backpressured => AsyncPrefixStoreSubmit::Backpressured,
            AsyncPrefixStoreAdmission::Closed => AsyncPrefixStoreSubmit::Closed,
        }
    }

    pub fn try_admit(
        &self,
        store: PagedPrefixStore,
        spec: PagedPrefixKeySpec,
        cancellation: AsyncPrefixStoreCancellation,
    ) -> AsyncPrefixStoreAdmission {
        let sender_open = self
            .0
            .sender
            .lock()
            .ok()
            .is_some_and(|sender| sender.is_some());
        if !sender_open {
            return AsyncPrefixStoreAdmission::Closed;
        }
        let key = PagedPrefixStore::key_for(&spec);
        let pending_key = (store.root.clone(), key);
        match self.0.pending.lock() {
            Ok(pending)
                if pending
                    .get(&pending_key)
                    .is_some_and(|entry| !entry.cancellation.is_cancelled()) =>
            {
                return AsyncPrefixStoreAdmission::Coalesced;
            }
            Ok(_) => {}
            Err(_) => return AsyncPrefixStoreAdmission::Closed,
        }
        let payload_bytes = spec.payload_bytes();
        if !self.reserve_pending(payload_bytes) {
            self.0
                .counters
                .backpressured_total
                .fetch_add(1, Ordering::Relaxed);
            return AsyncPrefixStoreAdmission::Backpressured;
        }
        let id = self.0.next_job_id.fetch_add(1, Ordering::Relaxed);
        let pending_result = self.0.pending.lock().map(|mut pending| {
            if pending
                .get(&pending_key)
                .is_some_and(|entry| !entry.cancellation.is_cancelled())
            {
                return false;
            }
            pending.insert(
                pending_key.clone(),
                AsyncPrefixPendingEntry {
                    id,
                    spec: spec.clone(),
                    entry: None,
                    cancellation: cancellation.clone(),
                },
            );
            true
        });
        match pending_result {
            Ok(true) => AsyncPrefixStoreAdmission::Admitted(Box::new(AsyncPrefixStorePermit {
                queue: self.clone(),
                id,
                pending_key,
                store: Some(store),
                spec: Some(spec),
                payload_bytes,
                cancellation,
                active: true,
            })),
            Ok(false) => {
                self.release_pending(payload_bytes);
                AsyncPrefixStoreAdmission::Coalesced
            }
            Err(_) => {
                self.release_pending(payload_bytes);
                AsyncPrefixStoreAdmission::Closed
            }
        }
    }

    fn submit_permit(
        &self,
        permit: &mut AsyncPrefixStorePermit,
        entry: AsyncPrefixStorePayload,
    ) -> AsyncPrefixStoreSubmit {
        let spec = permit.spec.take().expect("active permit owns spec");
        // The reservation is the entry's logical size, an upper bound on
        // what the writer puts on disk (segments already there are skipped).
        let actual_payload_bytes = match entry.logical_bytes(&spec) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.0.counters.failed_total.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(%error, "async paged prefix store payload does not fit its spec");
                return AsyncPrefixStoreSubmit::Closed;
            }
        };
        if actual_payload_bytes != permit.payload_bytes {
            self.0.counters.failed_total.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                reserved_bytes = permit.payload_bytes,
                actual_bytes = actual_payload_bytes,
                "async paged prefix store payload size changed after admission"
            );
            return AsyncPrefixStoreSubmit::Closed;
        }
        if let Err(error) = entry.eval() {
            self.0.counters.failed_total.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(%error, "async paged prefix store payload evaluation failed");
            return AsyncPrefixStoreSubmit::Closed;
        }
        if permit.cancellation.is_cancelled() {
            self.0
                .counters
                .cancelled_total
                .fetch_add(1, Ordering::Relaxed);
            return AsyncPrefixStoreSubmit::Cancelled;
        }
        let pending_updated = self.0.pending.lock().is_ok_and(|mut pending| {
            let Some(current) = pending.get_mut(&permit.pending_key) else {
                return false;
            };
            if current.id != permit.id || current.cancellation.is_cancelled() {
                return false;
            }
            current.entry = Some(entry.clone());
            true
        });
        if !pending_updated {
            self.0
                .counters
                .cancelled_total
                .fetch_add(1, Ordering::Relaxed);
            return AsyncPrefixStoreSubmit::Cancelled;
        }
        let job = AsyncPrefixStoreJob {
            id: permit.id,
            pending_key: permit.pending_key.clone(),
            store: permit.store.take().expect("active permit owns store"),
            spec,
            entry,
            payload_bytes: permit.payload_bytes,
            cancellation: permit.cancellation.clone(),
        };
        let send_result = self
            .0
            .sender
            .lock()
            .ok()
            .and_then(|sender| sender.as_ref().map(|sender| sender.try_send(Box::new(job))));
        match send_result {
            Some(Ok(())) => {
                permit.active = false;
                self.0.counters.queued_total.fetch_add(1, Ordering::Relaxed);
                AsyncPrefixStoreSubmit::Queued
            }
            Some(Err(TrySendError::Full(job))) => {
                self.remove_pending_if_current(&job.pending_key, job.id);
                self.release_pending(job.payload_bytes);
                permit.active = false;
                self.0
                    .counters
                    .backpressured_total
                    .fetch_add(1, Ordering::Relaxed);
                AsyncPrefixStoreSubmit::Backpressured
            }
            Some(Err(TrySendError::Disconnected(job))) => {
                self.remove_pending_if_current(&job.pending_key, job.id);
                self.release_pending(job.payload_bytes);
                permit.active = false;
                AsyncPrefixStoreSubmit::Closed
            }
            None => {
                self.remove_pending_if_current(&permit.pending_key, permit.id);
                self.release_pending(permit.payload_bytes);
                permit.active = false;
                AsyncPrefixStoreSubmit::Closed
            }
        }
    }

    fn remove_pending_if_current(&self, key: &(PathBuf, String), id: u64) {
        if let Ok(mut pending) = self.0.pending.lock() {
            if pending.get(key).is_some_and(|entry| entry.id == id) {
                pending.remove(key);
            }
        }
    }

    fn load_pending_observed(
        &self,
        store: &PagedPrefixStore,
        spec: &PagedPrefixKeySpec,
    ) -> Option<PagedPrefixLoadResult> {
        let key = PagedPrefixStore::key_for(spec);
        let pending_key = (store.root.clone(), key.clone());
        let payload = self.pending_payload(&pending_key, spec)?;
        let cached_entry = match payload {
            AsyncPrefixStorePayload::Whole(entry) => entry,
            AsyncPrefixStorePayload::Segmented(entry) => {
                materialize_prefix_entry(&entry, spec.cached_len, spec.block_size).ok()?
            }
        };
        Some(PagedPrefixLoadResult {
            key,
            status: PagedPrefixLoadStatus::Hit,
            stats: Some(cached_entry.observability_stats(spec.cached_len)),
            entry: Some(cached_entry),
        })
    }

    /// A whole prefix waiting to be written, in segmented form.
    fn load_pending_segmented(
        &self,
        store: &PagedPrefixStore,
        spec: &PagedPrefixKeySpec,
    ) -> Option<PagedPrefixSegmentedLoadResult> {
        let key = PagedPrefixStore::key_for(spec);
        let pending_key = (store.root.clone(), key.clone());
        let entry = self.pending_payload(&pending_key, spec)?.segmented();
        let stats = materialize_prefix_entry(&entry, spec.cached_len, spec.block_size)
            .ok()?
            .observability_stats(spec.cached_len);
        Some(PagedPrefixSegmentedLoadResult {
            key,
            status: PagedPrefixLoadStatus::Hit,
            entry: Some(entry),
            stats: Some(stats),
        })
    }

    fn pending_payload(
        &self,
        pending_key: &(PathBuf, String),
        spec: &PagedPrefixKeySpec,
    ) -> Option<AsyncPrefixStorePayload> {
        let pending = self.0.pending.lock().ok()?;
        let entry = pending.get(pending_key)?;
        if entry.cancellation.is_cancelled() || entry.spec != *spec {
            return None;
        }
        entry.entry.clone()
    }

    fn contains_pending(&self, store: &PagedPrefixStore, spec: &PagedPrefixKeySpec) -> bool {
        let key = PagedPrefixStore::key_for(spec);
        let pending_key = (store.root.clone(), key);
        self.0.pending.lock().is_ok_and(|pending| {
            pending
                .get(&pending_key)
                .is_some_and(|entry| !entry.cancellation.is_cancelled() && entry.spec == *spec)
        })
    }

    fn pending_cached_lengths(&self, store: &PagedPrefixStore, max_cached_len: i32) -> Vec<i32> {
        let Ok(pending) = self.0.pending.lock() else {
            return Vec::new();
        };
        pending
            .iter()
            .filter_map(|((root, _), entry)| {
                (root == &store.root
                    && !entry.cancellation.is_cancelled()
                    && entry.entry.is_some()
                    && entry.spec.entry_kind == PrefixEntryKind::WholePrefix
                    && entry.spec.cached_len > 0
                    && entry.spec.cached_len <= max_cached_len)
                    .then_some(entry.spec.cached_len)
            })
            .collect()
    }

    fn reserve_pending(&self, bytes: usize) -> bool {
        if self.0.counters.pending_jobs.load(Ordering::Acquire) >= self.0.capacity {
            return false;
        }
        // One save larger than the byte limit is still accepted when nothing
        // else is waiting: otherwise a long prompt could never reach disk.
        let reserved = self.0.counters.pending_bytes.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| {
                current
                    .checked_add(bytes)
                    .filter(|projected| *projected <= self.0.max_pending_bytes || current == 0)
            },
        );
        if reserved.is_err() {
            return false;
        }
        let previous_jobs = self.0.counters.pending_jobs.fetch_add(1, Ordering::AcqRel);
        if previous_jobs >= self.0.capacity {
            self.0.counters.pending_jobs.fetch_sub(1, Ordering::AcqRel);
            self.0
                .counters
                .pending_bytes
                .fetch_sub(bytes, Ordering::AcqRel);
            return false;
        }
        true
    }

    fn release_pending(&self, bytes: usize) {
        self.0.counters.pending_jobs.fetch_sub(1, Ordering::AcqRel);
        self.0
            .counters
            .pending_bytes
            .fetch_sub(bytes, Ordering::AcqRel);
    }

    pub fn is_backpressured(&self) -> bool {
        self.0.counters.pending_jobs.load(Ordering::Acquire) >= self.0.capacity
            || self.0.counters.pending_bytes.load(Ordering::Acquire) >= self.0.max_pending_bytes
    }

    /// Cancel queued store work owned by a model that is being unloaded.
    ///
    /// Cancellation is cooperative: jobs that have not started are skipped;
    /// an already-running atomic filesystem save is allowed to finish. Pending
    /// read-through stops exposing cancelled entries immediately.
    pub fn cancel_model(&self, model_id: &str) -> usize {
        let Ok(pending) = self.0.pending.lock() else {
            return 0;
        };
        let mut cancelled = 0usize;
        for entry in pending.values() {
            if entry.spec.model_id == model_id && !entry.cancellation.is_cancelled() {
                entry.cancellation.cancel();
                cancelled = cancelled.saturating_add(1);
            }
        }
        cancelled
    }

    pub fn stats(&self) -> AsyncPrefixStoreStats {
        AsyncPrefixStoreStats {
            pending_jobs: self.0.counters.pending_jobs.load(Ordering::Acquire),
            pending_bytes: self.0.counters.pending_bytes.load(Ordering::Acquire),
            queued_total: self.0.counters.queued_total.load(Ordering::Relaxed),
            completed_total: self.0.counters.completed_total.load(Ordering::Relaxed),
            failed_total: self.0.counters.failed_total.load(Ordering::Relaxed),
            cancelled_total: self.0.counters.cancelled_total.load(Ordering::Relaxed),
            backpressured_total: self.0.counters.backpressured_total.load(Ordering::Relaxed),
        }
    }

    pub fn shutdown(&self) {
        if let Ok(mut sender) = self.0.sender.lock() {
            sender.take();
        }
        if let Ok(mut worker) = self.0.worker.lock() {
            if let Some(worker) = worker.take() {
                let _ = worker.join();
            }
        }
    }

    pub fn wait_idle(&self) {
        let (lock, idle) = &*self.0.idle;
        let mut guard = lock.lock().expect("async prefix store idle lock poisoned");
        while self.0.counters.pending_jobs.load(Ordering::Acquire) != 0 {
            guard = idle
                .wait(guard)
                .expect("async prefix store idle lock poisoned");
        }
    }
}

static PROCESS_PREFIX_STORE_QUEUE: OnceLock<AsyncPrefixStoreQueue> = OnceLock::new();

pub fn process_async_prefix_store_queue() -> &'static AsyncPrefixStoreQueue {
    PROCESS_PREFIX_STORE_QUEUE.get_or_init(|| {
        AsyncPrefixStoreQueue::new(
            PROCESS_PREFIX_STORE_QUEUE_CAPACITY,
            PROCESS_PREFIX_STORE_PENDING_BYTES,
        )
        .expect("valid process prefix store queue")
    })
}

pub fn cancel_process_async_prefix_store_model(model_id: &str) -> usize {
    PROCESS_PREFIX_STORE_QUEUE
        .get()
        .map_or(0, |queue| queue.cancel_model(model_id))
}

pub fn shutdown_process_async_prefix_store_queue() {
    if let Some(queue) = PROCESS_PREFIX_STORE_QUEUE.get() {
        queue.shutdown();
    }
}

impl PrefixLruCache {
    pub fn new(config: PrefixLruCacheConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            max_bytes: config.max_bytes,
            total_bytes: 0,
            generation: 0,
            entries: HashMap::new(),
            segments: HashMap::new(),
            recency: VecDeque::new(),
        })
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn health(&self) -> PrefixLruHealth {
        PrefixLruHealth {
            entry_count: self.entries.len(),
            segment_count: self.segments.len(),
            segment_bytes: self
                .segments
                .values()
                .fold(0usize, |sum, slot| sum.saturating_add(slot.bytes)),
            logical_bytes: self.entries.values().fold(0usize, |sum, entry| {
                sum.saturating_add(entry.stats.payload_bytes)
            }),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn cached_lengths_descending(&self, max_cached_len: usize) -> Vec<i32> {
        let mut lengths = self
            .entries
            .values()
            .filter_map(|entry| {
                let cached_len = entry.spec.cached_len;
                if cached_len > 0 && cached_len as usize <= max_cached_len {
                    Some(cached_len)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        lengths.dedup();
        lengths
    }

    /// Whether exactly this prefix is cached. Leaves recency untouched.
    pub fn contains(&self, spec: &PagedPrefixKeySpec) -> bool {
        self.entries
            .get(&PagedPrefixStore::key_for(spec))
            .is_some_and(|entry| entry.spec == *spec)
    }

    pub fn load_observed(&mut self, spec: &PagedPrefixKeySpec) -> Result<PagedPrefixLoadResult> {
        let key = PagedPrefixStore::key_for(spec);
        let Some(entry) = self.entries.get(&key) else {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::MissingEntry,
                entry: None,
                stats: None,
            });
        };
        if entry.spec != *spec {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::MetadataMismatch,
                entry: None,
                stats: Some(entry.stats),
            });
        }
        let cached_entry =
            materialize_prefix_entry(&entry.segmented, spec.cached_len, spec.block_size)?;
        let stats = entry.stats;
        self.touch(&key);
        Ok(PagedPrefixLoadResult {
            key,
            status: PagedPrefixLoadStatus::Hit,
            entry: Some(cached_entry),
            stats: Some(stats),
        })
    }

    /// The cached entry for exactly `spec` and the segments it came from.
    /// Counts as a use.
    pub fn load_segmented(
        &mut self,
        spec: &PagedPrefixKeySpec,
    ) -> Result<Option<(SegmentedPrefixEntry, PrefixLineage)>> {
        let key = PagedPrefixStore::key_for(spec);
        let Some(entry) = self.entries.get(&key).filter(|entry| entry.spec == *spec) else {
            return Ok(None);
        };
        let segmented = entry.segmented.clone();
        let lineage = segmented_entry_lineage(&segmented, prefix_segment_layout(spec));
        self.touch(&key);
        Ok(Some((segmented, lineage)))
    }

    /// A segment some cached entry uses.
    pub fn segment(&self, id: PrefixSegmentId) -> Option<Arc<PrefixKvSegment>> {
        self.segments.get(&id).map(|slot| slot.segment.clone())
    }

    /// The head of `lineage` this cache still holds, as segment references:
    /// stops at the first segment that is gone or does not continue the
    /// previous one.
    pub fn lineage_segments(&self, lineage: &PrefixLineage) -> Vec<PrefixSegmentRef> {
        let mut refs = Vec::with_capacity(lineage.refs.len());
        let mut pos = 0;
        for &PrefixLineageRef {
            id, start, stop, ..
        } in &lineage.refs
        {
            let Some(segment) = self.segment(id) else {
                break;
            };
            if start != pos
                || segment.start != start
                || stop <= start
                || stop > segment.stop
                || segment.layout != lineage.layout
            {
                break;
            }
            refs.push(PrefixSegmentRef { segment, stop });
            pos = stop;
        }
        refs
    }

    /// Insert an unshared entry. It is kept whole as private data: it
    /// references no segment and nothing shares its buffers, so its bytes are
    /// exactly what removing it gives back.
    pub fn insert(
        &mut self,
        spec: PagedPrefixKeySpec,
        entry: PagedPrefixEntry,
    ) -> Result<PrefixLruInsertResult> {
        self.insert_segmented(
            spec,
            SegmentedPrefixEntry {
                segments: vec![],
                aligned_len: 0,
                private: entry,
            },
        )
    }

    /// Insert an entry whose head lives in shared segments. Segments already
    /// cached are not charged again.
    pub fn insert_segmented(
        &mut self,
        spec: PagedPrefixKeySpec,
        mut entry: SegmentedPrefixEntry,
    ) -> Result<PrefixLruInsertResult> {
        let validator = PagedPrefixStore::new(Path::new(""));
        validator.validate_spec(&spec)?;
        let layout = prefix_segment_layout(&spec);
        if let Some(r) = entry.segments.iter().find(|r| r.segment.layout != layout) {
            anyhow::bail!(
                "PrefixLruCache: segment {} layout does not match the entry",
                r.segment.id
            );
        }
        // One copy per segment id: an entry that brings its own copy of a
        // segment this cache (or an earlier ref of the entry) already holds
        // uses that one instead.
        let mut seen = HashMap::<PrefixSegmentId, Arc<PrefixKvSegment>>::new();
        for r in &mut entry.segments {
            let held = self
                .segments
                .get(&r.segment.id)
                .map(|slot| slot.segment.clone())
                .or_else(|| seen.get(&r.segment.id).cloned());
            let Some(held) = held else {
                seen.insert(r.segment.id, r.segment.clone());
                continue;
            };
            if held.start != r.segment.start
                || held.stop != r.segment.stop
                || held.layout != r.segment.layout
            {
                anyhow::bail!(
                    "PrefixLruCache: segment {} does not match the copy already held",
                    r.segment.id
                );
            }
            r.segment = held;
        }
        // Lazy: validates shapes without evaluating anything.
        let materialized = materialize_prefix_entry(&entry, spec.cached_len, spec.block_size)?;
        validator.validate_entry(&spec, &materialized)?;
        let stats = materialized.observability_stats(spec.cached_len);
        drop(materialized);

        let key = PagedPrefixStore::key_for(&spec);
        let private_bytes = entry
            .private
            .observability_stats(spec.cached_len)
            .payload_bytes;
        let ids = entry_segment_ids(&entry);
        let new_segment_bytes = entry
            .segments
            .iter()
            .filter(|r| !self.segments.contains_key(&r.segment.id))
            .map(|r| (r.segment.id, r.segment.payload_bytes()))
            .collect::<HashMap<_, _>>()
            .into_values()
            .fold(0usize, usize::saturating_add);
        let charged_bytes = new_segment_bytes.saturating_add(private_bytes);
        if charged_bytes > self.max_bytes {
            return Ok(PrefixLruInsertResult {
                key,
                status: PrefixLruInsertStatus::SkippedOversized,
                stats,
                charged_bytes: 0,
            });
        }

        // Evaluate before the entry becomes visible: cached segments are
        // shared across threads, which `PrefixKvSegment: Sync` allows only
        // for evaluated arrays. Callers evaluate first (an SSD hit comes
        // back evaluated), so this guard normally does no work under the
        // cache lock.
        for r in &entry.segments {
            if !self.segments.contains_key(&r.segment.id) {
                r.segment.eval()?;
            }
        }
        entry.private.eval()?;

        // Take the new references before dropping a replaced entry's, so a
        // segment both use is kept.
        for id in &ids {
            let Some(r) = entry.segments.iter().find(|r| r.segment.id == *id) else {
                continue;
            };
            let slot = self
                .segments
                .entry(r.segment.id)
                .or_insert_with(|| SegmentSlot {
                    bytes: r.segment.payload_bytes(),
                    segment: r.segment.clone(),
                    entry_refs: 0,
                });
            if slot.entry_refs == 0 {
                self.total_bytes = self.total_bytes.saturating_add(slot.bytes);
            }
            slot.entry_refs += 1;
        }
        self.total_bytes = self.total_bytes.saturating_add(private_bytes);
        let status = if self.remove_entry(&key).is_some() {
            PrefixLruInsertStatus::Replaced
        } else {
            PrefixLruInsertStatus::Stored
        };
        let generation = self.next_generation();
        self.entries.insert(
            key.clone(),
            PrefixLruEntry {
                spec,
                segmented: entry,
                private_bytes,
                stats,
                generation,
            },
        );
        self.recency.push_back((key.clone(), generation));
        self.shrink_to(self.max_bytes);

        Ok(PrefixLruInsertResult {
            key,
            status,
            stats,
            charged_bytes,
        })
    }

    fn next_generation(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        self.generation
    }

    fn touch(&mut self, key: &str) {
        let generation = self.next_generation();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.generation = generation;
            self.recency.push_back((key.to_owned(), generation));
        }
    }

    /// Count the entries a prompt of `model_id` can restore as used now, so a
    /// reclaim made to admit that prompt takes them last.
    pub fn touch_prefixes_of(&mut self, model_id: &str, prompt_ids: &[u32]) {
        let keys = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                let tokens = &entry.spec.token_ids;
                entry.spec.model_id == model_id
                    && tokens.len() <= prompt_ids.len()
                    && tokens
                        .iter()
                        .zip(prompt_ids)
                        .all(|(&cached, &prompt)| i64::from(cached) == i64::from(prompt))
            })
            .map(|(key, entry)| (entry.spec.token_ids.len(), key.clone()))
            .collect::<std::collections::BTreeSet<_>>();
        // Shortest first: the longest ends up the most recent.
        for (_, key) in keys {
            self.touch(&key);
        }
    }

    /// Shrink to an absolute byte target; the order is [`Self::reclaim`]'s
    /// with no model spared.
    pub fn shrink_to(&mut self, target_bytes: usize) -> usize {
        self.evict_to(target_bytes, None).bytes
    }

    /// Evict until at least `needed_bytes` are gone or the cache is empty.
    ///
    /// Covered entries go first, least recently used first: an entry is
    /// covered when a longer cached entry of the same model continues its
    /// tokens, so a later prompt that would restore it restores the longer
    /// one instead (a chunk checkpoint under its turn boundary, an earlier
    /// turn under a later one). Then, as before, the model furthest above
    /// its equal share loses its least recently used entry; `spare_model`,
    /// the model the memory is wanted for, loses its own last.
    pub fn reclaim(&mut self, needed_bytes: usize, spare_model: Option<&str>) -> PrefixLruReclaim {
        self.evict_to(self.total_bytes.saturating_sub(needed_bytes), spare_model)
    }

    fn covered_keys(&self) -> Vec<(String, u64)> {
        let mut covered = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                self.entries
                    .values()
                    .any(|longer| longer.spec.continues(&entry.spec))
            })
            .map(|(key, entry)| (key.clone(), entry.generation))
            .collect::<Vec<_>>();
        covered.sort_unstable_by_key(|(_, generation)| *generation);
        covered
    }

    /// Drop exactly this entry (tests evict one entry without the LRU order).
    #[cfg(test)]
    pub(crate) fn remove(&mut self, spec: &PagedPrefixKeySpec) -> bool {
        self.remove_entry(&PagedPrefixStore::key_for(spec))
            .is_some()
    }

    /// Remove `key`; returns the bytes this gave back (its private data and
    /// the segments no other entry uses), `None` if it was not cached.
    fn remove_entry(&mut self, key: &str) -> Option<usize> {
        let entry = self.entries.remove(key)?;
        let mut freed = entry.private_bytes;
        for id in entry_segment_ids(&entry.segmented) {
            let Some(slot) = self.segments.get_mut(&id) else {
                continue;
            };
            slot.entry_refs = slot.entry_refs.saturating_sub(1);
            if slot.entry_refs == 0 {
                freed = freed.saturating_add(slot.bytes);
                self.segments.remove(&id);
            }
        }
        self.total_bytes = self.total_bytes.saturating_sub(freed);
        Some(freed)
    }

    /// Bytes per model: private data plus each segment its entries use, once.
    fn owner_bytes(&self, skip: impl Fn(&PrefixLruEntry) -> bool) -> HashMap<&str, usize> {
        let mut owner_bytes = HashMap::<&str, usize>::new();
        let mut counted = std::collections::HashSet::<(&str, PrefixSegmentId)>::new();
        for entry in self.entries.values() {
            if skip(entry) {
                continue;
            }
            let owner = entry.spec.model_id.as_str();
            let mut bytes = entry.private_bytes;
            for id in entry_segment_ids(&entry.segmented) {
                if counted.insert((owner, id)) {
                    if let Some(slot) = self.segments.get(&id) {
                        bytes = bytes.saturating_add(slot.bytes);
                    }
                }
            }
            let total = owner_bytes.entry(owner).or_default();
            *total = total.saturating_add(bytes);
        }
        owner_bytes
    }

    fn evict_to(&mut self, target_bytes: usize, spare_model: Option<&str>) -> PrefixLruReclaim {
        let before = self.total_bytes;
        let mut reclaim = PrefixLruReclaim::default();
        if self.total_bytes > target_bytes {
            for (key, _) in self.covered_keys() {
                if self.total_bytes <= target_bytes {
                    break;
                }
                self.remove_entry(&key);
                reclaim.covered_entries += 1;
            }
        }
        while self.total_bytes > target_bytes {
            let spared = |entry: &PrefixLruEntry| spare_model == Some(entry.spec.model_id.as_str());
            let others_remain = self.entries.values().any(|entry| !spared(entry));
            let owner_bytes = self.owner_bytes(|entry| others_remain && spared(entry));
            let fair_share = if owner_bytes.is_empty() {
                target_bytes
            } else {
                target_bytes / owner_bytes.len()
            };
            let Some(selected_owner) = owner_bytes
                .iter()
                .max_by_key(|(_, bytes)| bytes.saturating_sub(fair_share))
                .map(|(owner, _)| *owner)
            else {
                break;
            };
            let Some(key) = self
                .entries
                .iter()
                .filter(|(_, entry)| entry.spec.model_id == selected_owner)
                .min_by_key(|(_, entry)| entry.generation)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            self.remove_entry(&key);
            reclaim.other_entries += 1;
        }
        self.recency.retain(|(key, generation)| {
            self.entries
                .get(key)
                .is_some_and(|entry| entry.generation == *generation)
        });
        reclaim.bytes = before.saturating_sub(self.total_bytes);
        reclaim
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TensorSpecMetadata {
    dtype: String,
    shape: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct LayerSpecMetadata {
    kind: PrefixLayerKind,
    tensors: Vec<TensorSpecMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct MtpLayerSpecMetadata {
    k: TensorSpecMetadata,
    v: TensorSpecMetadata,
}

#[derive(Debug, Serialize)]
struct KeyMaterial<'a> {
    schema_version: u32,
    entry_kind: PrefixEntryKind,
    model_id: &'a str,
    token_ids: &'a [i32],
    cached_len: i32,
    fingerprint: Option<&'a str>,
    block_size: i32,
    kv_cache_profile: Option<&'a str>,
    main_layers: Vec<LayerSpecMetadata>,
    mtp_layers: Vec<MtpLayerSpecMetadata>,
    mtp_last_hidden: Option<TensorSpecMetadata>,
    gemma4_drafter_last_hidden: Option<TensorSpecMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PrefixMetadata {
    schema_version: u32,
    entry_kind: PrefixEntryKind,
    key: String,
    model_id: String,
    token_hash: String,
    token_count: usize,
    cached_len: i32,
    fingerprint_hash: Option<String>,
    block_size: i32,
    kv_cache_profile: Option<String>,
    main_layers: Vec<LayerSpecMetadata>,
    mtp_layers: Vec<MtpLayerSpecMetadata>,
    mtp_last_hidden: Option<TensorSpecMetadata>,
    gemma4_drafter_last_hidden: Option<TensorSpecMetadata>,
}

impl PagedPrefixStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            max_bytes: None,
        }
    }

    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = Some(max_bytes);
        self
    }

    fn with_optional_max_bytes(mut self, max_bytes: Option<usize>) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    pub fn key_for(spec: &PagedPrefixKeySpec) -> String {
        let material = KeyMaterial {
            schema_version: SCHEMA_VERSION,
            entry_kind: spec.entry_kind,
            model_id: &spec.model_id,
            token_ids: &spec.token_ids,
            cached_len: spec.cached_len,
            fingerprint: spec.fingerprint.as_deref(),
            block_size: spec.block_size,
            kv_cache_profile: spec.kv_cache_profile.as_deref(),
            main_layers: layer_metadata(&spec.main_layers),
            mtp_layers: mtp_layer_metadata(&spec.mtp_layers),
            mtp_last_hidden: spec.mtp_last_hidden.as_ref().map(tensor_metadata),
            gemma4_drafter_last_hidden: spec
                .gemma4_drafter_last_hidden
                .as_ref()
                .map(tensor_metadata),
        };
        let json = serde_json::to_string(&material)
            .expect("PagedPrefixStore::key_for serializes fixed metadata");
        stable_hex_hash(&json)
    }

    /// Save `entry`. A whole prefix is written in the `wp8` format as one
    /// private part; an immutable block keeps its `<root>/<key>/` directory.
    pub fn save(&self, spec: &PagedPrefixKeySpec, entry: &PagedPrefixEntry) -> Result<String> {
        if spec.entry_kind == PrefixEntryKind::WholePrefix {
            return self.save_segmented(
                spec,
                &SegmentedPrefixEntry {
                    segments: Vec::new(),
                    aligned_len: 0,
                    private: entry.clone(),
                },
            );
        }
        self.save_root_entry(spec, entry)
    }

    /// Write `entry` as `<root>/<key>/`: immutable blocks, and parked
    /// requests (the active-KV store), which are not cache entries.
    pub(crate) fn save_root_entry(
        &self,
        spec: &PagedPrefixKeySpec,
        entry: &PagedPrefixEntry,
    ) -> Result<String> {
        self.validate_spec(spec)?;
        self.validate_entry(spec, entry)?;

        if let Some(max_bytes) = self.max_bytes {
            let payload_bytes = entry.observability_stats(spec.cached_len).payload_bytes;
            if payload_bytes > max_bytes {
                anyhow::bail!(
                    "PagedPrefixStore: entry payload_bytes {payload_bytes} exceeds max_bytes {max_bytes}"
                );
            }
        }

        let key = Self::key_for(spec);
        let metadata = metadata_for(spec, &key)?;
        let final_dir = self.root.join(&key);
        let tmp_dir = self
            .root
            .join(format!(".tmp-{key}-{}", uuid::Uuid::new_v4().simple()));
        fs::create_dir_all(&self.root)
            .with_context(|| format!("create prefix cache root {}", self.root.display()))?;
        // Immutable blocks come from the single store writer, like `wp8`
        // entries; the active-KV store writes from scheduler threads and
        // cleans its own root.
        if spec.entry_kind == PrefixEntryKind::ImmutableBlock {
            self.ensure_wp8_startup_cleanup();
        }
        fs::create_dir_all(&tmp_dir)
            .with_context(|| format!("create temporary prefix cache dir {}", tmp_dir.display()))?;

        let save_result = (|| -> Result<()> {
            let meta_path = tmp_dir.join(META_FILE);
            let payload_path = tmp_dir.join(PAYLOAD_FILE);
            let meta_bytes =
                serde_json::to_vec_pretty(&metadata).context("serialize paged prefix metadata")?;
            fs::write(&meta_path, meta_bytes)
                .with_context(|| format!("write prefix cache metadata {}", meta_path.display()))?;

            let mut tensors = HashMap::new();
            insert_entry_tensors(entry, &mut tensors);
            let mut safetensors_meta = HashMap::new();
            safetensors_meta.insert("ironmlx.prefix_cache.key".to_owned(), key.clone());
            safetensors_meta.insert(
                "ironmlx.prefix_cache.schema_version".to_owned(),
                SCHEMA_VERSION.to_string(),
            );
            let payload_path = payload_path.to_string_lossy().into_owned();
            mlx::io::save_safetensors(&payload_path, &tensors, &safetensors_meta)
                .with_context(|| format!("save prefix cache payload {payload_path}"))?;
            Ok(())
        })();

        if let Err(err) = save_result {
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(err);
        }

        if final_dir.exists() {
            fs::remove_dir_all(&final_dir)
                .with_context(|| format!("replace prefix cache dir {}", final_dir.display()))?;
        }
        fs::rename(&tmp_dir, &final_dir).with_context(|| {
            format!(
                "install prefix cache entry {} -> {}",
                tmp_dir.display(),
                final_dir.display()
            )
        })?;
        if let Some(max_bytes) = self.max_bytes {
            self.evict_to_disk_capacity(&key, max_bytes)?;
        }
        Ok(key)
    }

    pub fn save_if_absent(
        &self,
        spec: &PagedPrefixKeySpec,
        entry: &PagedPrefixEntry,
    ) -> Result<(String, bool)> {
        self.validate_spec(spec)?;
        self.validate_entry(spec, entry)?;

        let key = Self::key_for(spec);
        if self.entry_metadata_matches(spec, &key)? {
            return Ok((key, false));
        }
        self.save(spec, entry).map(|key| (key, true))
    }

    pub fn matching_entry_key(&self, spec: &PagedPrefixKeySpec) -> Result<Option<String>> {
        self.validate_spec(spec)?;
        let key = Self::key_for(spec);
        if self.entry_metadata_matches(spec, &key)? {
            Ok(Some(key))
        } else {
            Ok(None)
        }
    }

    pub fn cached_lengths_descending(&self, max_cached_len: i32) -> Result<Vec<i32>> {
        if max_cached_len <= 0 {
            return Ok(Vec::new());
        }
        let mut lengths = PROCESS_PREFIX_STORE_QUEUE
            .get()
            .map_or_else(Vec::new, |queue| {
                queue.pending_cached_lengths(self, max_cached_len)
            });
        // Whole prefixes live under `wp8/`; `<root>/<key>/` holds immutable
        // blocks and legacy whole-prefix entries, which are not read.
        lengths.extend(self.wp8_cached_lengths(max_cached_len)?);
        lengths.sort_unstable_by(|left, right| right.cmp(left));
        lengths.dedup();
        Ok(lengths)
    }

    pub fn load(&self, spec: &PagedPrefixKeySpec) -> Result<Option<PagedPrefixEntry>> {
        Ok(self.load_observed(spec)?.entry)
    }

    pub fn contains(&self, spec: &PagedPrefixKeySpec) -> Result<bool> {
        self.validate_spec(spec)?;
        if PROCESS_PREFIX_STORE_QUEUE
            .get()
            .is_some_and(|queue| queue.contains_pending(self, spec))
        {
            return Ok(true);
        }
        self.contains_persisted(spec)
    }

    pub fn contains_persisted(&self, spec: &PagedPrefixKeySpec) -> Result<bool> {
        self.validate_spec(spec)?;
        let key = Self::key_for(spec);
        self.entry_metadata_matches(spec, &key)
    }

    pub fn load_observed(&self, spec: &PagedPrefixKeySpec) -> Result<PagedPrefixLoadResult> {
        self.validate_spec(spec)?;
        if spec.entry_kind == PrefixEntryKind::WholePrefix {
            let observed = self.load_segmented_observed(spec, None)?;
            let entry = observed
                .entry
                .map(|entry| materialize_prefix_entry(&entry, spec.cached_len, spec.block_size))
                .transpose()?;
            return Ok(PagedPrefixLoadResult {
                key: observed.key,
                status: observed.status,
                entry,
                stats: observed.stats,
            });
        }
        self.load_root_observed(spec)
    }

    /// Read `<root>/<key>/` (see [`Self::save_root_entry`]).
    pub(crate) fn load_root_observed(
        &self,
        spec: &PagedPrefixKeySpec,
    ) -> Result<PagedPrefixLoadResult> {
        self.validate_spec(spec)?;
        if let Some(observed) = PROCESS_PREFIX_STORE_QUEUE
            .get()
            .and_then(|queue| queue.load_pending_observed(self, spec))
        {
            return Ok(observed);
        }
        let key = Self::key_for(spec);
        let entry_dir = self.root.join(&key);
        if !entry_dir.is_dir() {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::MissingEntry,
                entry: None,
                stats: None,
            });
        }

        let metadata = metadata_for(spec, &key)?;
        let meta_path = entry_dir.join(META_FILE);
        let actual_metadata = match fs::read(&meta_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PrefixMetadata>(&bytes).ok())
        {
            Some(meta) => meta,
            None => {
                return Ok(PagedPrefixLoadResult {
                    key,
                    status: PagedPrefixLoadStatus::InvalidMetadata,
                    entry: None,
                    stats: None,
                });
            }
        };
        if actual_metadata != metadata {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::MetadataMismatch,
                entry: None,
                stats: None,
            });
        }

        let payload_path = entry_dir.join(PAYLOAD_FILE);
        let payload_path_string = payload_path.to_string_lossy().into_owned();
        let (mut tensors, _metadata) = match mlx::io::load_safetensors(&payload_path_string) {
            Ok(loaded) => loaded,
            Err(_) => {
                return Ok(PagedPrefixLoadResult {
                    key,
                    status: PagedPrefixLoadStatus::PayloadReadFailed,
                    entry: None,
                    stats: None,
                });
            }
        };
        let entry = match entry_from_tensors(spec, &mut tensors) {
            Ok(entry) => entry,
            Err(_) => {
                return Ok(PagedPrefixLoadResult {
                    key,
                    status: PagedPrefixLoadStatus::PayloadInvalid,
                    entry: None,
                    stats: None,
                });
            }
        };
        if !tensors.is_empty() {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::PayloadInvalid,
                entry: None,
                stats: None,
            });
        }
        if self.validate_entry(spec, &entry).is_err() {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::EntryInvalid,
                entry: None,
                stats: None,
            });
        }
        let stats = entry.observability_stats(spec.cached_len);
        Ok(PagedPrefixLoadResult {
            key,
            status: PagedPrefixLoadStatus::Hit,
            entry: Some(entry),
            stats: Some(stats),
        })
    }

    fn entry_metadata_matches(&self, spec: &PagedPrefixKeySpec, key: &str) -> Result<bool> {
        if spec.entry_kind == PrefixEntryKind::WholePrefix {
            return Ok(self.wp8_entry_matches(spec, key));
        }
        let entry_dir = self.root.join(key);
        if !entry_dir.is_dir() {
            return Ok(false);
        }
        if !entry_dir.join(PAYLOAD_FILE).is_file() {
            return Ok(false);
        }
        let meta_path = entry_dir.join(META_FILE);
        let Some(actual_metadata) = fs::read(&meta_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PrefixMetadata>(&bytes).ok())
        else {
            return Ok(false);
        };
        Ok(actual_metadata == metadata_for(spec, key)?)
    }

    /// Delete entries until the store holds at most `max_bytes`, sparing
    /// `keep_key`. Block-pool entries, legacy `<root>/<key>/` directories,
    /// `wp8` entries and their segment files share the budget. `wp8` entries
    /// a longer entry covers go first; then the oldest of everything. A
    /// segment file goes when no `wp8` entry references it any more.
    fn evict_to_disk_capacity(&self, keep_key: &str, max_bytes: usize) -> Result<()> {
        let max_bytes = max_bytes as u64;
        let root_entries = self.disk_entries()?;
        let wp8_entries = self.wp8_disk_entries()?;
        let mut segment_files = self.wp8_segment_files()?;
        let total = |root: &[PrefixDiskEntry],
                     wp8: &[(PrefixDiskEntry, Wp8DiskEntry)],
                     segments: &HashMap<PathBuf, HashMap<String, u64>>| {
            root.iter()
                .chain(wp8.iter().map(|(entry, _)| entry))
                .map(|entry| entry.bytes)
                .chain(segments.values().flat_map(|files| files.values().copied()))
                .fold(0_u64, u64::saturating_add)
        };
        let mut total_bytes = total(&root_entries, &wp8_entries, &segment_files);
        if total_bytes <= max_bytes {
            return Ok(());
        }

        // Segment references per layout directory.
        let mut refs = HashMap::<(PathBuf, String), usize>::new();
        for (_, wp8) in &wp8_entries {
            for name in wp8_entry_segment_files(wp8) {
                *refs.entry((wp8.layout_dir.clone(), name)).or_default() += 1;
            }
        }
        // Files no entry references (an entry lost to a failed read).
        for (layout_dir, files) in &mut segment_files {
            files.retain(|name, bytes| {
                let referenced = refs.contains_key(&(layout_dir.clone(), name.clone()));
                if !referenced {
                    remove_path(&layout_dir.join(WP8_SEGMENTS_DIR).join(name.as_str()));
                    total_bytes = total_bytes.saturating_sub(*bytes);
                }
                referenced
            });
        }

        let covered = covered_wp8_entries(&wp8_entries);
        enum Victim<'a> {
            Root(&'a PrefixDiskEntry),
            Wp8(&'a PrefixDiskEntry, &'a Wp8DiskEntry),
        }
        let mut covered_victims = Vec::new();
        let mut others = Vec::new();
        for (idx, (entry, wp8)) in wp8_entries.iter().enumerate() {
            if covered.contains(&idx) {
                covered_victims.push(Victim::Wp8(entry, wp8));
            } else {
                others.push(Victim::Wp8(entry, wp8));
            }
        }
        others.extend(root_entries.iter().map(Victim::Root));
        let entry_of = |victim: &Victim<'_>| match victim {
            Victim::Root(entry) | Victim::Wp8(entry, _) => (entry.modified, entry.key.clone()),
        };
        covered_victims.sort_by_key(|victim| entry_of(victim));
        others.sort_by_key(|victim| entry_of(victim));

        for victim in covered_victims.iter().chain(&others) {
            if total_bytes <= max_bytes {
                break;
            }
            let (entry, wp8) = match victim {
                Victim::Root(entry) => (*entry, None),
                Victim::Wp8(entry, wp8) => (*entry, Some(*wp8)),
            };
            if entry.key == keep_key {
                continue;
            }
            remove_entry_dir(&entry.path)?;
            total_bytes = total_bytes.saturating_sub(entry.bytes);
            let Some(wp8) = wp8 else {
                continue;
            };
            for name in wp8_entry_segment_files(wp8) {
                let key = (wp8.layout_dir.clone(), name);
                let Some(count) = refs.get_mut(&key) else {
                    continue;
                };
                *count -= 1;
                if *count > 0 {
                    continue;
                }
                refs.remove(&key);
                let (layout_dir, name) = key;
                if let Some(bytes) = segment_files
                    .get_mut(&layout_dir)
                    .and_then(|files| files.remove(&name))
                {
                    remove_path(&layout_dir.join(WP8_SEGMENTS_DIR).join(name));
                    total_bytes = total_bytes.saturating_sub(bytes);
                }
            }
        }

        if total_bytes > max_bytes {
            anyhow::bail!(
                "PagedPrefixStore: retained entry exceeds max_bytes {max_bytes}; total_bytes={total_bytes}"
            );
        }
        Ok(())
    }

    fn disk_entries(&self) -> Result<Vec<PrefixDiskEntry>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(err) if err.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("read prefix cache root {}", self.root.display()));
            }
        };
        let mut result = Vec::new();
        for entry in entries {
            let entry =
                entry.with_context(|| format!("read prefix cache root {}", self.root.display()))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("read prefix cache entry type {}", path.display()))?;
            if !file_type.is_dir() {
                continue;
            }
            let key = entry.file_name().to_string_lossy().into_owned();
            // The DFlash2 prefix store shares this directory and keeps its
            // own limit.
            if key.starts_with(".tmp-")
                || key == crate::core::dflash2_prefix_store::DFLASH2_STORE_DIR
                || key == WP8_DIR
            {
                continue;
            }
            let metadata = match entry.metadata() {
                Ok(metadata) => metadata,
                // A reader removed it meanwhile.
                Err(err) if err.kind() == ErrorKind::NotFound => continue,
                Err(err) => {
                    return Err(err)
                        .with_context(|| format!("read prefix cache metadata {}", path.display()));
                }
            };
            let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
            let bytes = directory_disk_bytes(&path)?;
            result.push(PrefixDiskEntry {
                key,
                path,
                bytes,
                modified,
            });
        }
        Ok(result)
    }

    fn validate_spec(&self, spec: &PagedPrefixKeySpec) -> Result<()> {
        if spec.model_id.is_empty() {
            anyhow::bail!("PagedPrefixStore: model_id must not be empty");
        }
        if spec.block_size <= 0 {
            anyhow::bail!(
                "PagedPrefixStore: block_size must be > 0, got {}",
                spec.block_size
            );
        }
        if spec.cached_len <= 0 {
            anyhow::bail!(
                "PagedPrefixStore: cached_len must be > 0, got {}",
                spec.cached_len
            );
        }
        if spec.token_ids.len() != spec.cached_len as usize {
            anyhow::bail!(
                "PagedPrefixStore: token_ids.len()={} != cached_len {}",
                spec.token_ids.len(),
                spec.cached_len
            );
        }
        if spec
            .fingerprint
            .as_ref()
            .is_some_and(|fingerprint| fingerprint.is_empty())
        {
            anyhow::bail!("PagedPrefixStore: fingerprint must not be empty when present");
        }
        if spec
            .kv_cache_profile
            .as_ref()
            .is_some_and(|profile| profile.is_empty())
        {
            anyhow::bail!("PagedPrefixStore: kv_cache_profile must not be empty when present");
        }
        if spec.main_layers.is_empty() && spec.mtp_layers.is_empty() {
            anyhow::bail!("PagedPrefixStore: entry must contain at least one cache layer");
        }
        if spec.entry_kind == PrefixEntryKind::ImmutableBlock {
            if spec.cached_len != spec.block_size {
                anyhow::bail!(
                    "PagedPrefixStore: immutable block cached_len {} != block_size {}",
                    spec.cached_len,
                    spec.block_size
                );
            }
            if !spec.mtp_layers.is_empty()
                || spec.mtp_last_hidden.is_some()
                || spec.gemma4_drafter_last_hidden.is_some()
                || spec
                    .main_layers
                    .iter()
                    .any(|layer| layer.kind != PrefixLayerKind::FullPaged)
            {
                anyhow::bail!(
                    "PagedPrefixStore: immutable blocks support FullPaged main layers only"
                );
            }
            if spec.fingerprint.as_deref().is_none_or(|fingerprint| {
                fingerprint != "immutable-root" && !fingerprint.starts_with("immutable-parent:")
            }) {
                anyhow::bail!(
                    "PagedPrefixStore: immutable block requires an explicit parent fingerprint"
                );
            }
        }
        if spec.mtp_layers.is_empty() && spec.mtp_last_hidden.is_some() {
            anyhow::bail!("PagedPrefixStore: mtp_last_hidden cannot be present without mtp_layers");
        }
        if !spec.mtp_layers.is_empty() && spec.mtp_last_hidden.is_none() {
            anyhow::bail!("PagedPrefixStore: mtp_layers require mtp_last_hidden");
        }
        for (idx, layer) in spec.main_layers.iter().enumerate() {
            validate_layer_spec(idx, layer, spec.block_size, spec.cached_len)?;
        }
        for (idx, layer) in spec.mtp_layers.iter().enumerate() {
            validate_tensor_spec(&format!("mtp layer {idx} K"), &layer.k)?;
            validate_tensor_spec(&format!("mtp layer {idx} V"), &layer.v)?;
        }
        if let Some(last_hidden) = &spec.mtp_last_hidden {
            validate_tensor_spec("mtp last_hidden", last_hidden)?;
        }
        if let Some(last_hidden) = &spec.gemma4_drafter_last_hidden {
            validate_tensor_spec("Gemma4 drafter last_hidden", last_hidden)?;
        }
        Ok(())
    }

    fn validate_entry(&self, spec: &PagedPrefixKeySpec, entry: &PagedPrefixEntry) -> Result<()> {
        if entry.main_layers.len() != spec.main_layers.len() {
            anyhow::bail!(
                "PagedPrefixStore: main layer count {} != spec {}",
                entry.main_layers.len(),
                spec.main_layers.len()
            );
        }
        if entry.mtp_layers.len() != spec.mtp_layers.len() {
            anyhow::bail!(
                "PagedPrefixStore: MTP layer count {} != spec {}",
                entry.mtp_layers.len(),
                spec.mtp_layers.len()
            );
        }
        for (idx, (layer_spec, payload)) in spec
            .main_layers
            .iter()
            .zip(entry.main_layers.iter())
            .enumerate()
        {
            validate_layer_payload(idx, layer_spec, payload)?;
        }
        for (idx, (layer_spec, payload)) in spec
            .mtp_layers
            .iter()
            .zip(entry.mtp_layers.iter())
            .enumerate()
        {
            validate_tensor_payload(&format!("mtp layer {idx} K"), &layer_spec.k, &payload.k)?;
            validate_tensor_payload(&format!("mtp layer {idx} V"), &layer_spec.v, &payload.v)?;
        }
        match (&spec.mtp_last_hidden, &entry.mtp_last_hidden) {
            (Some(tensor_spec), Some(payload)) => {
                validate_tensor_payload("mtp last_hidden", tensor_spec, payload)?;
            }
            (None, None) => {}
            (Some(_), None) => anyhow::bail!("PagedPrefixStore: missing mtp_last_hidden payload"),
            (None, Some(_)) => {
                anyhow::bail!("PagedPrefixStore: unexpected mtp_last_hidden payload")
            }
        }
        match (
            &spec.gemma4_drafter_last_hidden,
            &entry.gemma4_drafter_last_hidden,
        ) {
            (Some(tensor_spec), Some(payload)) => {
                validate_tensor_payload("Gemma4 drafter last_hidden", tensor_spec, payload)?;
            }
            (None, None) => {}
            (Some(_), None) => {
                anyhow::bail!("PagedPrefixStore: missing Gemma4 drafter last_hidden payload")
            }
            (None, Some(_)) => {
                anyhow::bail!("PagedPrefixStore: unexpected Gemma4 drafter last_hidden payload")
            }
        }
        Ok(())
    }
}

/// Distinct segment files a `wp8` entry references.
fn wp8_entry_segment_files(entry: &Wp8DiskEntry) -> Vec<String> {
    let mut names = entry
        .meta
        .iter()
        .flat_map(|meta| meta.segments.iter().map(|segment| segment.file_name()))
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

/// Indices of the `wp8` entries a longer entry continues: same layout and
/// fingerprint, and its tokens start with theirs.
fn covered_wp8_entries(
    entries: &[(PrefixDiskEntry, Wp8DiskEntry)],
) -> std::collections::HashSet<usize> {
    let mut groups = HashMap::<(&Path, Option<&str>), Vec<usize>>::new();
    for (idx, (_, wp8)) in entries.iter().enumerate() {
        if let Some(meta) = &wp8.meta {
            groups
                .entry((
                    wp8.layout_dir.as_path(),
                    meta.prefix.fingerprint_hash.as_deref(),
                ))
                .or_default()
                .push(idx);
        }
    }
    let mut covered = std::collections::HashSet::new();
    for members in groups.values().filter(|members| members.len() > 1) {
        let tokens = members
            .iter()
            .filter_map(|&idx| read_wp8_tokens(&entries[idx].0.path).map(|tokens| (idx, tokens)))
            .collect::<Vec<_>>();
        for (idx, shorter) in &tokens {
            if tokens
                .iter()
                .any(|(_, longer)| longer.len() > shorter.len() && longer.starts_with(shorter))
            {
                covered.insert(*idx);
            }
        }
    }
    covered
}

#[derive(Debug)]
struct PrefixDiskEntry {
    key: String,
    path: PathBuf,
    bytes: u64,
    modified: SystemTime,
}

/// Bytes under `path`. Whatever a concurrent reader removed meanwhile (an
/// invalid entry it deleted) counts as zero.
fn directory_disk_bytes(path: &Path) -> Result<u64> {
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(0),
        Err(err) => {
            return Err(err).with_context(|| format!("read prefix cache entry {}", path.display()));
        }
    };
    let mut total = 0_u64;
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("read prefix cache entry {}", path.display()));
            }
        };
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "read prefix cache entry metadata {}",
                        entry.path().display()
                    )
                });
            }
        };
        if metadata.is_dir() {
            total = total.saturating_add(directory_disk_bytes(&entry.path())?);
        } else {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

/// Remove an evicted entry directory; one a concurrent reader already
/// removed counts as removed.
fn remove_entry_dir(path: &Path) -> Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => {
            Err(err).with_context(|| format!("evict prefix cache entry {}", path.display()))
        }
    }
}

fn validate_layer_spec(
    idx: usize,
    spec: &PrefixLayerSpec,
    block_size: i32,
    cached_len: i32,
) -> Result<()> {
    for (tensor_idx, tensor) in spec.tensors.iter().enumerate() {
        validate_tensor_spec(&format!("layer {idx} tensor {tensor_idx}"), tensor)?;
    }
    match spec.kind {
        PrefixLayerKind::FullDense => {
            require_tensor_count(idx, spec, 2)?;
            for (name, tensor) in ["K", "V"].iter().zip(spec.tensors.iter()) {
                let dims = tensor.shape.as_slice();
                if dims.len() != 4
                    || dims[0] != 1
                    || dims[1] <= 0
                    || dims[2] != cached_len
                    || dims[3] <= 0
                {
                    anyhow::bail!(
                        "PagedPrefixStore: FullDense layer {idx} {name} shape {:?} incompatible with [1,heads,{cached_len},dim]",
                        dims
                    );
                }
            }
        }
        PrefixLayerKind::FullPaged => {
            require_tensor_count(idx, spec, 2)?;
            for (name, tensor) in ["K", "V"].iter().zip(spec.tensors.iter()) {
                let dims = tensor.shape.as_slice();
                if dims.len() != 4
                    || dims[0] < 0
                    || dims[1] <= 0
                    || dims[2] != block_size
                    || dims[3] <= 0
                {
                    anyhow::bail!(
                        "PagedPrefixStore: FullPaged layer {idx} {name} shape {:?} incompatible with [pages,heads,{block_size},dim]",
                        dims
                    );
                }
            }
        }
        PrefixLayerKind::FullTurboQuantPacked => {
            require_tensor_count(idx, spec, 4)?;
            validate_turboquant_packed_spec(idx, "K packed", &spec.tensors[0], cached_len)?;
            validate_turboquant_norm_spec(idx, "K norms", &spec.tensors[1], cached_len)?;
            validate_turboquant_packed_spec(idx, "V packed", &spec.tensors[2], cached_len)?;
            validate_turboquant_norm_spec(idx, "V norms", &spec.tensors[3], cached_len)?;
        }
        PrefixLayerKind::Linear | PrefixLayerKind::Mla => {
            require_tensor_count(idx, spec, 2)?;
        }
    }
    Ok(())
}

fn require_tensor_count(idx: usize, spec: &PrefixLayerSpec, expected: usize) -> Result<()> {
    if spec.tensors.len() != expected {
        anyhow::bail!(
            "PagedPrefixStore: layer {idx} kind {:?} must describe exactly {expected} tensors, got {}",
            spec.kind,
            spec.tensors.len()
        );
    }
    Ok(())
}

fn validate_turboquant_packed_spec(
    idx: usize,
    name: &str,
    tensor: &PrefixTensorSpec,
    cached_len: i32,
) -> Result<()> {
    if tensor.dtype != Dtype::Uint32 {
        anyhow::bail!(
            "PagedPrefixStore: FullTurboQuantPacked layer {idx} {name} dtype {} != Uint32",
            tensor.dtype
        );
    }
    let dims = tensor.shape.as_slice();
    if dims.len() != 4 || dims[0] != 1 || dims[1] <= 0 || dims[2] != cached_len || dims[3] <= 0 {
        anyhow::bail!(
            "PagedPrefixStore: FullTurboQuantPacked layer {idx} {name} shape {:?} incompatible with [1,heads,{cached_len},packed_dim]",
            dims
        );
    }
    Ok(())
}

fn validate_turboquant_norm_spec(
    idx: usize,
    name: &str,
    tensor: &PrefixTensorSpec,
    cached_len: i32,
) -> Result<()> {
    if tensor.dtype != Dtype::Float32 {
        anyhow::bail!(
            "PagedPrefixStore: FullTurboQuantPacked layer {idx} {name} dtype {} != Float32",
            tensor.dtype
        );
    }
    let dims = tensor.shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] != cached_len {
        anyhow::bail!(
            "PagedPrefixStore: FullTurboQuantPacked layer {idx} {name} shape {:?} incompatible with [1,heads,{cached_len}]",
            dims
        );
    }
    Ok(())
}

fn validate_tensor_spec(name: &str, spec: &PrefixTensorSpec) -> Result<()> {
    if spec.shape.iter().any(|&dim| dim < 0) {
        anyhow::bail!(
            "PagedPrefixStore: {name} shape {:?} contains a negative dimension",
            spec.shape
        );
    }
    Ok(())
}

fn validate_layer_payload(
    idx: usize,
    spec: &PrefixLayerSpec,
    payload: &PrefixLayerPayload,
) -> Result<()> {
    match (spec.kind, payload) {
        (PrefixLayerKind::FullDense, PrefixLayerPayload::FullDense { k, v }) => {
            validate_tensor_payload(&format!("layer {idx} dense K"), &spec.tensors[0], k)?;
            validate_tensor_payload(&format!("layer {idx} dense V"), &spec.tensors[1], v)?;
        }
        (PrefixLayerKind::FullPaged, PrefixLayerPayload::FullPaged { k_pages, v_pages }) => {
            validate_tensor_payload(&format!("layer {idx} full K"), &spec.tensors[0], k_pages)?;
            validate_tensor_payload(&format!("layer {idx} full V"), &spec.tensors[1], v_pages)?;
        }
        (
            PrefixLayerKind::FullTurboQuantPacked,
            PrefixLayerPayload::FullTurboQuantPacked {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            },
        ) => {
            validate_tensor_payload(
                &format!("layer {idx} TurboQuant K packed"),
                &spec.tensors[0],
                k_packed,
            )?;
            validate_tensor_payload(
                &format!("layer {idx} TurboQuant K norms"),
                &spec.tensors[1],
                k_norms,
            )?;
            validate_tensor_payload(
                &format!("layer {idx} TurboQuant V packed"),
                &spec.tensors[2],
                v_packed,
            )?;
            validate_tensor_payload(
                &format!("layer {idx} TurboQuant V norms"),
                &spec.tensors[3],
                v_norms,
            )?;
        }
        (
            PrefixLayerKind::Linear,
            PrefixLayerPayload::Linear {
                conv_state,
                recurrent_state,
            },
        ) => {
            validate_tensor_payload(
                &format!("layer {idx} linear conv_state"),
                &spec.tensors[0],
                conv_state,
            )?;
            validate_tensor_payload(
                &format!("layer {idx} linear recurrent_state"),
                &spec.tensors[1],
                recurrent_state,
            )?;
        }
        (PrefixLayerKind::Mla, PrefixLayerPayload::Mla { c_kv, k_pe }) => {
            validate_tensor_payload(&format!("layer {idx} MLA c_kv"), &spec.tensors[0], c_kv)?;
            validate_tensor_payload(&format!("layer {idx} MLA k_pe"), &spec.tensors[1], k_pe)?;
        }
        _ => anyhow::bail!(
            "PagedPrefixStore: layer {idx} payload kind does not match {:?}",
            spec.kind
        ),
    }
    Ok(())
}

fn validate_tensor_payload(name: &str, spec: &PrefixTensorSpec, tensor: &Array) -> Result<()> {
    if tensor.dtype() != spec.dtype {
        anyhow::bail!(
            "PagedPrefixStore: {name} dtype {} != expected {}",
            tensor.dtype(),
            spec.dtype
        );
    }
    let shape = tensor.shape();
    if shape.as_slice() != spec.shape.as_slice() {
        anyhow::bail!(
            "PagedPrefixStore: {name} shape {:?} != expected {:?}",
            shape.as_slice(),
            spec.shape
        );
    }
    Ok(())
}

fn metadata_for(spec: &PagedPrefixKeySpec, key: &str) -> Result<PrefixMetadata> {
    Ok(PrefixMetadata {
        schema_version: SCHEMA_VERSION,
        entry_kind: spec.entry_kind,
        key: key.to_owned(),
        model_id: spec.model_id.clone(),
        token_hash: token_hash(&spec.token_ids)?,
        token_count: spec.token_ids.len(),
        cached_len: spec.cached_len,
        fingerprint_hash: spec
            .fingerprint
            .as_ref()
            .map(|fingerprint| stable_hex_hash(fingerprint)),
        block_size: spec.block_size,
        kv_cache_profile: spec.kv_cache_profile.clone(),
        main_layers: layer_metadata(&spec.main_layers),
        mtp_layers: mtp_layer_metadata(&spec.mtp_layers),
        mtp_last_hidden: spec.mtp_last_hidden.as_ref().map(tensor_metadata),
        gemma4_drafter_last_hidden: spec
            .gemma4_drafter_last_hidden
            .as_ref()
            .map(tensor_metadata),
    })
}

fn token_hash(token_ids: &[i32]) -> Result<String> {
    let json = serde_json::to_string(token_ids).context("serialize prefix token ids")?;
    Ok(stable_hex_hash(&json))
}

fn layer_metadata(layers: &[PrefixLayerSpec]) -> Vec<LayerSpecMetadata> {
    layers
        .iter()
        .map(|layer| LayerSpecMetadata {
            kind: layer.kind,
            tensors: layer.tensors.iter().map(tensor_metadata).collect(),
        })
        .collect()
}

fn mtp_layer_metadata(layers: &[PrefixMtpLayerSpec]) -> Vec<MtpLayerSpecMetadata> {
    layers
        .iter()
        .map(|layer| MtpLayerSpecMetadata {
            k: tensor_metadata(&layer.k),
            v: tensor_metadata(&layer.v),
        })
        .collect()
}

fn tensor_metadata(spec: &PrefixTensorSpec) -> TensorSpecMetadata {
    TensorSpecMetadata {
        dtype: spec.dtype.to_string(),
        shape: spec.shape.clone(),
    }
}

fn insert_entry_tensors(entry: &PagedPrefixEntry, tensors: &mut HashMap<String, Array>) {
    for (idx, layer) in entry.main_layers.iter().enumerate() {
        match layer {
            PrefixLayerPayload::FullDense { k, v } => {
                tensors.insert(main_dense_k_name(idx), k.clone());
                tensors.insert(main_dense_v_name(idx), v.clone());
            }
            PrefixLayerPayload::FullPaged { k_pages, v_pages } => {
                tensors.insert(main_full_k_name(idx), k_pages.clone());
                tensors.insert(main_full_v_name(idx), v_pages.clone());
            }
            PrefixLayerPayload::FullTurboQuantPacked {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            } => {
                tensors.insert(main_turboquant_k_packed_name(idx), k_packed.clone());
                tensors.insert(main_turboquant_k_norms_name(idx), k_norms.clone());
                tensors.insert(main_turboquant_v_packed_name(idx), v_packed.clone());
                tensors.insert(main_turboquant_v_norms_name(idx), v_norms.clone());
            }
            PrefixLayerPayload::Linear {
                conv_state,
                recurrent_state,
            } => {
                tensors.insert(main_linear_conv_name(idx), conv_state.clone());
                tensors.insert(main_linear_recurrent_name(idx), recurrent_state.clone());
            }
            PrefixLayerPayload::Mla { c_kv, k_pe } => {
                tensors.insert(main_mla_c_kv_name(idx), c_kv.clone());
                tensors.insert(main_mla_k_pe_name(idx), k_pe.clone());
            }
        }
    }
    for (idx, layer) in entry.mtp_layers.iter().enumerate() {
        tensors.insert(mtp_k_name(idx), layer.k.clone());
        tensors.insert(mtp_v_name(idx), layer.v.clone());
    }
    if let Some(last_hidden) = &entry.mtp_last_hidden {
        tensors.insert(mtp_last_hidden_name(), last_hidden.clone());
    }
    if let Some(last_hidden) = &entry.gemma4_drafter_last_hidden {
        tensors.insert(gemma4_drafter_last_hidden_name(), last_hidden.clone());
    }
}

fn entry_from_tensors(
    spec: &PagedPrefixKeySpec,
    tensors: &mut HashMap<String, Array>,
) -> Result<PagedPrefixEntry> {
    let mut main_layers = Vec::with_capacity(spec.main_layers.len());
    for (idx, layer_spec) in spec.main_layers.iter().enumerate() {
        let layer = match layer_spec.kind {
            PrefixLayerKind::FullDense => PrefixLayerPayload::FullDense {
                k: take_tensor(tensors, main_dense_k_name(idx))?,
                v: take_tensor(tensors, main_dense_v_name(idx))?,
            },
            PrefixLayerKind::FullPaged => PrefixLayerPayload::FullPaged {
                k_pages: take_tensor(tensors, main_full_k_name(idx))?,
                v_pages: take_tensor(tensors, main_full_v_name(idx))?,
            },
            PrefixLayerKind::FullTurboQuantPacked => PrefixLayerPayload::FullTurboQuantPacked {
                k_packed: take_tensor(tensors, main_turboquant_k_packed_name(idx))?,
                k_norms: take_tensor(tensors, main_turboquant_k_norms_name(idx))?,
                v_packed: take_tensor(tensors, main_turboquant_v_packed_name(idx))?,
                v_norms: take_tensor(tensors, main_turboquant_v_norms_name(idx))?,
            },
            PrefixLayerKind::Linear => PrefixLayerPayload::Linear {
                conv_state: take_tensor(tensors, main_linear_conv_name(idx))?,
                recurrent_state: take_tensor(tensors, main_linear_recurrent_name(idx))?,
            },
            PrefixLayerKind::Mla => PrefixLayerPayload::Mla {
                c_kv: take_tensor(tensors, main_mla_c_kv_name(idx))?,
                k_pe: take_tensor(tensors, main_mla_k_pe_name(idx))?,
            },
        };
        main_layers.push(layer);
    }

    let mut mtp_layers = Vec::with_capacity(spec.mtp_layers.len());
    for idx in 0..spec.mtp_layers.len() {
        mtp_layers.push(PrefixMtpLayerPayload {
            k: take_tensor(tensors, mtp_k_name(idx))?,
            v: take_tensor(tensors, mtp_v_name(idx))?,
        });
    }

    let mtp_last_hidden = if spec.mtp_last_hidden.is_some() {
        Some(take_tensor(tensors, mtp_last_hidden_name())?)
    } else {
        None
    };
    let gemma4_drafter_last_hidden = if spec.gemma4_drafter_last_hidden.is_some() {
        Some(take_tensor(tensors, gemma4_drafter_last_hidden_name())?)
    } else {
        None
    };

    Ok(PagedPrefixEntry {
        main_layers,
        mtp_layers,
        mtp_last_hidden,
        gemma4_drafter_last_hidden,
    })
}

fn take_tensor(tensors: &mut HashMap<String, Array>, name: String) -> Result<Array> {
    tensors
        .remove(&name)
        .ok_or_else(|| anyhow::anyhow!("PagedPrefixStore: missing tensor {name}"))
}

fn main_dense_k_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_dense_k")
}

fn main_dense_v_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_dense_v")
}

fn main_full_k_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_full_k_pages")
}

fn main_full_v_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_full_v_pages")
}

fn main_turboquant_k_packed_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_turboquant_k_packed")
}

fn main_turboquant_k_norms_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_turboquant_k_norms")
}

fn main_turboquant_v_packed_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_turboquant_v_packed")
}

fn main_turboquant_v_norms_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_turboquant_v_norms")
}

fn main_linear_conv_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_linear_conv_state")
}

fn main_linear_recurrent_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_linear_recurrent_state")
}

fn main_mla_c_kv_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_mla_c_kv")
}

fn main_mla_k_pe_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_mla_k_pe")
}

fn mtp_k_name(layer_idx: usize) -> String {
    format!("mtp_{layer_idx:04}_k")
}

fn mtp_v_name(layer_idx: usize) -> String {
    format!("mtp_{layer_idx:04}_v")
}

fn mtp_last_hidden_name() -> String {
    "mtp_last_hidden".to_owned()
}

fn gemma4_drafter_last_hidden_name() -> String {
    "gemma4_drafter_last_hidden".to_owned()
}

fn stable_hex_hash(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    fn temp_root(name: &str) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("ironmlx-{name}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn tensor_spec(array: &Array) -> PrefixTensorSpec {
        PrefixTensorSpec::from_array(array)
    }

    fn spec(
        tokens: Vec<i32>,
        cached_len: i32,
        fingerprint: Option<&str>,
        main_layers: Vec<PrefixLayerSpec>,
        mtp_layers: Vec<PrefixMtpLayerSpec>,
        mtp_last_hidden: Option<PrefixTensorSpec>,
    ) -> PagedPrefixKeySpec {
        PagedPrefixKeySpec {
            entry_kind: PrefixEntryKind::WholePrefix,
            model_id: "qwen3-test".to_owned(),
            token_ids: tokens,
            cached_len,
            fingerprint: fingerprint.map(ToOwned::to_owned),
            block_size: 2,
            kv_cache_profile: None,
            main_layers,
            mtp_layers,
            mtp_last_hidden,
            gemma4_drafter_last_hidden: None,
        }
    }

    #[test]
    fn prefix_key_includes_cached_len_and_fingerprint() {
        let layer = PrefixLayerSpec {
            kind: PrefixLayerKind::FullPaged,
            tensors: vec![
                PrefixTensorSpec {
                    dtype: Dtype::Float32,
                    shape: vec![2, 1, 2, 2],
                },
                PrefixTensorSpec {
                    dtype: Dtype::Float32,
                    shape: vec![2, 1, 2, 2],
                },
            ],
        };
        let a = spec(vec![1, 2, 3], 3, None, vec![layer.clone()], vec![], None);
        let b = spec(vec![1, 2, 3], 3, None, vec![layer.clone()], vec![], None);
        let shorter = spec(vec![1, 2], 2, None, vec![layer.clone()], vec![], None);
        let vl_a = spec(
            vec![1, 2, 3],
            3,
            Some("vl:image-a"),
            vec![layer.clone()],
            vec![],
            None,
        );
        let vl_b = spec(
            vec![1, 2, 3],
            3,
            Some("vl:image-b"),
            vec![layer],
            vec![],
            None,
        );
        let mut immutable = shorter.clone();
        immutable.entry_kind = PrefixEntryKind::ImmutableBlock;
        immutable.fingerprint = Some("immutable-root".to_owned());
        assert_eq!(PagedPrefixStore::key_for(&a), PagedPrefixStore::key_for(&b));
        assert_ne!(
            PagedPrefixStore::key_for(&a),
            PagedPrefixStore::key_for(&shorter)
        );
        assert_ne!(
            PagedPrefixStore::key_for(&a),
            PagedPrefixStore::key_for(&vl_a)
        );
        assert_ne!(
            PagedPrefixStore::key_for(&vl_a),
            PagedPrefixStore::key_for(&vl_b)
        );
        assert_ne!(
            PagedPrefixStore::key_for(&shorter),
            PagedPrefixStore::key_for(&immutable)
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_entry_observability_stats_count_layers_pages_tensors_and_bytes() {
        let k_pages: Array = (&[1.0_f32; 8][..], (2_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (&[2.0_f32; 8][..], (2_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let conv_state: Array = (&[3.0_f32; 4][..], (1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let recurrent_state: Array = (&[4.0_f32; 4][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let mtp_k: Array = (&[5.0_f32; 3][..], (1_i32, 1_i32, 3_i32, 1_i32))
            .try_into()
            .unwrap();
        let mtp_v: Array = (&[6.0_f32; 3][..], (1_i32, 1_i32, 3_i32, 1_i32))
            .try_into()
            .unwrap();
        let mtp_last_hidden: Array = (&[7.0_f32; 2][..], (1_i32, 1_i32, 2_i32))
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![
                PrefixLayerPayload::FullPaged { k_pages, v_pages },
                PrefixLayerPayload::Linear {
                    conv_state,
                    recurrent_state,
                },
            ],
            mtp_layers: vec![PrefixMtpLayerPayload { k: mtp_k, v: mtp_v }],
            mtp_last_hidden: Some(mtp_last_hidden),
            gemma4_drafter_last_hidden: None,
        };

        let stats = entry.observability_stats(3);

        assert_eq!(stats.cached_len, 3);
        assert_eq!(stats.main_layers, 2);
        assert_eq!(stats.full_paged_layers, 1);
        assert_eq!(stats.linear_layers, 1);
        assert_eq!(stats.mla_layers, 0);
        assert_eq!(stats.mtp_layers, 1);
        assert_eq!(stats.full_paged_pages, 2);
        assert_eq!(stats.tensor_count, 7);
        assert_eq!(stats.payload_bytes, 128);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_load_observed_reports_hit_stats_and_miss_reason() {
        let root = temp_root("prefix-store-observed");
        let store = PagedPrefixStore::new(&root);
        let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
            gemma4_drafter_last_hidden: None,
        };
        let wanted = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);
        let missed = store.load_observed(&wanted).expect("miss load");
        assert_eq!(missed.status, PagedPrefixLoadStatus::MissingEntry);
        assert!(missed.entry.is_none());
        assert!(missed.stats.is_none());

        let key = store.save(&wanted, &entry).expect("save");
        let hit = store.load_observed(&wanted).expect("hit load");

        assert_eq!(hit.key, key);
        assert_eq!(hit.status, PagedPrefixLoadStatus::Hit);
        assert!(hit.entry.is_some());
        let stats = hit.stats.expect("hit stats");
        assert_eq!(stats.cached_len, 2);
        assert_eq!(stats.full_paged_layers, 1);
        assert_eq!(stats.full_paged_pages, 1);
        assert_eq!(stats.payload_bytes, 32);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_roundtrips_turboquant_packed_payload() {
        let root = temp_root("prefix-store-turboquant-packed");
        let store = PagedPrefixStore::new(&root);
        let k_packed: Array = (
            &[1_u32, 2, 3, 4, 5, 6, 7, 8][..],
            (1_i32, 2_i32, 4_i32, 1_i32),
        )
            .try_into()
            .unwrap();
        let k_norms: Array = (
            &[0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][..],
            (1_i32, 2_i32, 4_i32),
        )
            .try_into()
            .unwrap();
        let v_packed: Array = (
            &[11_u32, 12, 13, 14, 15, 16, 17, 18][..],
            (1_i32, 2_i32, 4_i32, 1_i32),
        )
            .try_into()
            .unwrap();
        let v_norms: Array = (
            &[1.1_f32, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8][..],
            (1_i32, 2_i32, 4_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullTurboQuantPacked {
                k_packed: k_packed.clone(),
                k_norms: k_norms.clone(),
                v_packed: v_packed.clone(),
                v_norms: v_norms.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
            gemma4_drafter_last_hidden: None,
        };
        let mut wanted = spec(
            vec![1, 2, 3, 4],
            4,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        wanted.kv_cache_profile = Some("turboquant-k3v4".to_owned());

        let key = store.save(&wanted, &entry).expect("save packed payload");
        let hit = store.load_observed(&wanted).expect("hit load");

        assert_eq!(hit.key, key);
        assert_eq!(hit.status, PagedPrefixLoadStatus::Hit);
        let stats = hit.stats.expect("hit stats");
        assert_eq!(stats.tensor_count, 4);
        assert_eq!(stats.payload_bytes, 128);
        let loaded = hit.entry.expect("hit entry");
        let PrefixLayerPayload::FullTurboQuantPacked {
            k_packed,
            k_norms,
            v_packed,
            v_norms,
        } = &loaded.main_layers[0]
        else {
            panic!("expected TurboQuant packed payload");
        };
        assert_eq!(k_packed.shape().as_slice(), &[1, 2, 4, 1]);
        assert_eq!(k_norms.shape().as_slice(), &[1, 2, 4]);
        assert_eq!(v_packed.dtype(), Dtype::Uint32);
        assert_eq!(v_norms.dtype(), Dtype::Float32);

        std::fs::remove_dir_all(root).unwrap();
    }

    fn single_full_paged_entry(seed: f32) -> PagedPrefixEntry {
        let k_pages: Array = (
            &[seed, seed + 1.0, seed + 2.0, seed + 3.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[seed + 10.0, seed + 11.0, seed + 12.0, seed + 13.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged { k_pages, v_pages }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
            gemma4_drafter_last_hidden: None,
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_round_trips_gemma4_drafter_last_hidden_without_mtp_layers() {
        let root = temp_root("prefix-store-gemma4-drafter-hidden");
        let store = PagedPrefixStore::new(&root);
        let mut entry = single_full_paged_entry(1.0);
        let hidden: Array = (&[0.25_f32, 0.5, 0.75, 1.0][..], (1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        entry.gemma4_drafter_last_hidden = Some(hidden);
        let mut wanted = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);
        wanted.gemma4_drafter_last_hidden = entry.gemma4_drafter_last_hidden_spec();

        let key = store.save(&wanted, &entry).expect("save drafter prefix");
        let loaded = store
            .load(&wanted)
            .expect("load drafter prefix")
            .expect("prefix hit");

        assert_eq!(key, PagedPrefixStore::key_for(&wanted));
        assert_eq!(
            loaded
                .gemma4_drafter_last_hidden
                .expect("loaded hidden")
                .shape()
                .as_slice(),
            &[1, 1, 4]
        );
        assert!(loaded.mtp_layers.is_empty());
        assert!(loaded.mtp_last_hidden.is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_key_distinguishes_gemma4_drafter_hidden_from_plain_prefix() {
        let entry = single_full_paged_entry(2.0);
        let plain = spec(vec![9, 8], 2, None, entry.main_layer_specs(), vec![], None);
        let mut drafter = plain.clone();
        drafter.gemma4_drafter_last_hidden = Some(PrefixTensorSpec {
            dtype: Dtype::Float32,
            shape: vec![1, 1, 4],
        });

        assert_ne!(
            PagedPrefixStore::key_for(&plain),
            PagedPrefixStore::key_for(&drafter)
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_lru_cache_contains_leaves_recency_alone() {
        let mut cache =
            PrefixLruCache::new(PrefixLruCacheConfig::new(64).expect("config")).expect("cache");
        let entry1 = single_full_paged_entry(1.0);
        let entry2 = single_full_paged_entry(2.0);
        let entry3 = single_full_paged_entry(3.0);
        let spec1 = spec(vec![1, 2], 2, None, entry1.main_layer_specs(), vec![], None);
        let spec2 = spec(vec![3, 4], 2, None, entry2.main_layer_specs(), vec![], None);
        let spec3 = spec(vec![5, 6], 2, None, entry3.main_layer_specs(), vec![], None);

        assert!(!cache.contains(&spec1));
        cache.insert(spec1.clone(), entry1).expect("insert spec1");
        cache.insert(spec2.clone(), entry2).expect("insert spec2");
        assert!(cache.contains(&spec1));

        // A load would have made spec1 the most recent; `contains` does not.
        cache.insert(spec3.clone(), entry3).expect("insert spec3");
        assert!(!cache.contains(&spec1));
        assert!(cache.contains(&spec2) && cache.contains(&spec3));
    }

    #[test]
    fn memory_only_paged_prefix_cache_has_no_store() {
        let config = PagedPrefixCacheConfig::memory_only("model", 128, 16).expect("config");
        assert_eq!(config.root, None);
        assert!(config.store().is_none());
        assert!(PagedPrefixCacheConfig::memory_only("model", 0, 16).is_err());
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_lru_cache_hits_and_evicts_least_recent_entry() {
        let mut cache =
            PrefixLruCache::new(PrefixLruCacheConfig::new(64).expect("config")).expect("cache");
        let entry1 = single_full_paged_entry(1.0);
        let entry2 = single_full_paged_entry(2.0);
        let entry3 = single_full_paged_entry(3.0);
        let spec1 = spec(vec![1, 2], 2, None, entry1.main_layer_specs(), vec![], None);
        let spec2 = spec(vec![3, 4], 2, None, entry2.main_layer_specs(), vec![], None);
        let spec3 = spec(vec![5, 6], 2, None, entry3.main_layer_specs(), vec![], None);

        cache.insert(spec1.clone(), entry1).expect("insert spec1");
        cache.insert(spec2.clone(), entry2).expect("insert spec2");
        assert_eq!(cache.total_bytes(), 64);
        assert_eq!(
            cache.load_observed(&spec1).expect("load spec1").status,
            PagedPrefixLoadStatus::Hit
        );

        cache.insert(spec3.clone(), entry3).expect("insert spec3");

        assert_eq!(
            cache.load_observed(&spec1).expect("reload spec1").status,
            PagedPrefixLoadStatus::Hit
        );
        assert_eq!(
            cache.load_observed(&spec2).expect("reload spec2").status,
            PagedPrefixLoadStatus::MissingEntry
        );
        assert_eq!(
            cache.load_observed(&spec3).expect("reload spec3").status,
            PagedPrefixLoadStatus::Hit
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_lru_cache_cached_lengths_descending_filters_to_usable_lengths() {
        let mut cache =
            PrefixLruCache::new(PrefixLruCacheConfig::new(96).expect("config")).expect("cache");
        let entry2 = single_full_paged_entry(2.0);
        let entry4 = single_full_paged_entry(4.0);
        let entry6 = single_full_paged_entry(6.0);
        cache
            .insert(
                spec(vec![1, 2], 2, None, entry2.main_layer_specs(), vec![], None),
                entry2,
            )
            .expect("insert len2");
        cache
            .insert(
                spec(
                    vec![1, 2, 3, 4],
                    4,
                    None,
                    entry4.main_layer_specs(),
                    vec![],
                    None,
                ),
                entry4,
            )
            .expect("insert len4");
        cache
            .insert(
                spec(
                    vec![1, 2, 3, 4, 5, 6],
                    6,
                    None,
                    entry6.main_layer_specs(),
                    vec![],
                    None,
                ),
                entry6,
            )
            .expect("insert len6");

        assert_eq!(cache.cached_lengths_descending(10), vec![6, 4, 2]);
        assert_eq!(cache.cached_lengths_descending(5), vec![4, 2]);
        assert_eq!(cache.cached_lengths_descending(1), Vec::<i32>::new());
    }

    fn lru_spec(model: &str, tokens: Vec<i32>, entry: &PagedPrefixEntry) -> PagedPrefixKeySpec {
        let cached_len = tokens.len() as i32;
        let mut spec = spec(
            tokens,
            cached_len,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        spec.model_id = model.to_owned();
        spec
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_lru_reclaim_takes_covered_entries_before_recent_use_decides() {
        let mut cache =
            PrefixLruCache::new(PrefixLruCacheConfig::new(160).expect("config")).expect("cache");
        let entry = single_full_paged_entry(1.0);
        // Oldest first: another conversation, then one conversation's chunk
        // checkpoint, first turn and second turn.
        let other = lru_spec("m", vec![9, 9], &entry);
        let checkpoint = lru_spec("m", vec![1, 2], &entry);
        let turn1 = lru_spec("m", vec![1, 2, 3], &entry);
        let turn2 = lru_spec("m", vec![1, 2, 3, 4], &entry);
        for spec in [&other, &checkpoint, &turn1, &turn2] {
            cache.insert(spec.clone(), entry.clone()).expect("insert");
        }
        // The checkpoint was used last; it is still covered.
        cache.load_observed(&checkpoint).expect("touch");

        let reclaim = cache.reclaim(33, None);

        assert_eq!(
            reclaim,
            PrefixLruReclaim {
                bytes: 64,
                covered_entries: 2,
                other_entries: 0
            }
        );
        assert!(cache.contains(&other) && cache.contains(&turn2));
        assert!(!cache.contains(&checkpoint) && !cache.contains(&turn1));

        // Nothing covered is left: least recently used goes.
        assert_eq!(cache.reclaim(1, None).other_entries, 1);
        assert!(!cache.contains(&other) && cache.contains(&turn2));
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_lru_reclaim_spares_the_requesting_model_until_others_are_gone() {
        let mut cache =
            PrefixLruCache::new(PrefixLruCacheConfig::new(160).expect("config")).expect("cache");
        let entry = single_full_paged_entry(1.0);
        let mine_old = lru_spec("mine", vec![1, 2], &entry);
        let theirs = lru_spec("theirs", vec![3, 4], &entry);
        let mine_new = lru_spec("mine", vec![5, 6], &entry);
        for spec in [&mine_old, &theirs, &mine_new] {
            cache.insert(spec.clone(), entry.clone()).expect("insert");
        }

        assert_eq!(cache.reclaim(1, Some("mine")).bytes, 32);
        assert!(!cache.contains(&theirs));
        assert!(cache.contains(&mine_old) && cache.contains(&mine_new));

        assert_eq!(cache.reclaim(1, Some("mine")).bytes, 32);
        assert!(!cache.contains(&mine_old) && cache.contains(&mine_new));
        assert_eq!(cache.reclaim(usize::MAX, Some("mine")).bytes, 32);
        assert_eq!(cache.reclaim(1, Some("mine")), PrefixLruReclaim::default());
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_lru_reclaim_for_a_prompt_takes_what_it_can_restore_last() {
        let mut cache =
            PrefixLruCache::new(PrefixLruCacheConfig::new(160).expect("config")).expect("cache");
        let entry = single_full_paged_entry(1.0);
        // The arriving prompt's conversation is the least recently used.
        let mine = lru_spec("m", vec![1, 2, 3], &entry);
        let other = lru_spec("m", vec![7, 8], &entry);
        cache.insert(mine.clone(), entry.clone()).expect("insert");
        cache.insert(other.clone(), entry.clone()).expect("insert");

        cache.touch_prefixes_of("m", &[1, 2, 3, 4, 5]);
        cache.reclaim(1, Some("m"));

        assert!(cache.contains(&mine) && !cache.contains(&other));
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_lru_entry_of_another_model_or_kind_covers_nothing() {
        let entry = single_full_paged_entry(1.0);
        let short = lru_spec("m", vec![1, 2], &entry);
        let long = lru_spec("m", vec![1, 2, 3], &entry);
        assert!(long.continues(&short));
        assert!(!short.continues(&long) && !long.continues(&long));
        assert!(!lru_spec("n", vec![1, 2, 3], &entry).continues(&short));
        assert!(!lru_spec("m", vec![1, 7, 3], &entry).continues(&short));
        let mut image = long.clone();
        image.fingerprint = Some("image".to_owned());
        assert!(!image.continues(&short));
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_lru_shrink_reclaims_from_owner_above_fair_share() {
        let mut cache =
            PrefixLruCache::new(PrefixLruCacheConfig::new(96).expect("config")).expect("cache");
        let entry_b = single_full_paged_entry(10.0);
        let entry_a1 = single_full_paged_entry(20.0);
        let entry_a2 = single_full_paged_entry(30.0);
        let mut spec_b = spec(
            vec![10, 11],
            2,
            None,
            entry_b.main_layer_specs(),
            vec![],
            None,
        );
        spec_b.model_id = "model-b".to_owned();
        let mut spec_a1 = spec(
            vec![20, 21],
            2,
            None,
            entry_a1.main_layer_specs(),
            vec![],
            None,
        );
        spec_a1.model_id = "model-a".to_owned();
        let mut spec_a2 = spec(
            vec![30, 31],
            2,
            None,
            entry_a2.main_layer_specs(),
            vec![],
            None,
        );
        spec_a2.model_id = "model-a".to_owned();

        cache.insert(spec_b.clone(), entry_b).expect("insert b");
        cache.insert(spec_a1.clone(), entry_a1).expect("insert a1");
        cache.insert(spec_a2.clone(), entry_a2).expect("insert a2");
        assert_eq!(cache.shrink_to(64), 32);
        assert_eq!(
            cache.load_observed(&spec_b).expect("load b").status,
            PagedPrefixLoadStatus::Hit
        );
        assert_eq!(
            cache.load_observed(&spec_a1).expect("load a1").status,
            PagedPrefixLoadStatus::MissingEntry
        );
        assert_eq!(
            cache.load_observed(&spec_a2).expect("load a2").status,
            PagedPrefixLoadStatus::Hit
        );
    }

    #[test]
    fn process_prefix_lru_budget_is_singleton() {
        let config = PrefixLruCacheConfig::new(123_457).expect("config");
        let first = process_shared_prefix_lru_cache(config).expect("first");
        let second = process_shared_prefix_lru_cache(config).expect("second");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn async_prefix_store_completes_and_releases_pending_budget() {
        let root = temp_root("async-prefix-store-complete");
        let entry = single_full_paged_entry(40.0);
        let wanted = spec(
            vec![40, 41],
            2,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        let entry_dir = PagedPrefixStore::new(&root).wp8_entry_dir(&wanted);
        let queue = AsyncPrefixStoreQueue::new(2, 1024).expect("queue");
        assert_eq!(
            queue.try_enqueue(PagedPrefixStore::new(&root), wanted, entry),
            AsyncPrefixStoreSubmit::Queued
        );
        queue.shutdown();
        let stats = queue.stats();
        assert_eq!(stats.pending_jobs, 0);
        assert_eq!(stats.pending_bytes, 0);
        assert_eq!(stats.completed_total, 1);
        assert!(entry_dir.is_dir());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn async_prefix_store_enforces_byte_backpressure_and_cancellation() {
        let entry = single_full_paged_entry(50.0);
        let wanted = spec(
            vec![50, 51],
            2,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        let payload_bytes = entry.observability_stats(2).payload_bytes;
        let queue = AsyncPrefixStoreQueue::new(2, payload_bytes - 1).expect("queue");
        // Larger than the byte limit, but nothing else is waiting.
        let oversize = match queue.try_admit(
            PagedPrefixStore::new(temp_root("unused")),
            wanted,
            AsyncPrefixStoreCancellation::default(),
        ) {
            AsyncPrefixStoreAdmission::Admitted(permit) => permit,
            _ => panic!("an oversize save is admitted into an empty queue"),
        };
        let second_entry = single_full_paged_entry(51.0);
        let second = spec(
            vec![52, 53],
            2,
            None,
            second_entry.main_layer_specs(),
            vec![],
            None,
        );
        assert_eq!(
            queue.try_enqueue(
                PagedPrefixStore::new(temp_root("unused")),
                second,
                second_entry
            ),
            AsyncPrefixStoreSubmit::Backpressured
        );
        assert_eq!(queue.stats().backpressured_total, 1);
        drop(oversize);
        assert_eq!(queue.stats().pending_bytes, 0);
        queue.shutdown();

        let root = temp_root("async-prefix-store-cancel");
        let entry = single_full_paged_entry(60.0);
        let wanted = spec(
            vec![60, 61],
            2,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        let cancellation = AsyncPrefixStoreCancellation::default();
        cancellation.cancel();
        let queue = AsyncPrefixStoreQueue::new(1, 1024).expect("queue");
        assert_eq!(
            queue.try_enqueue_cancellable(
                PagedPrefixStore::new(&root),
                wanted,
                entry,
                cancellation,
            ),
            AsyncPrefixStoreSubmit::Cancelled
        );
        assert_eq!(queue.stats().cancelled_total, 1);
        assert_eq!(queue.stats().pending_jobs, 0);
        queue.shutdown();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn async_prefix_store_admits_before_payload_extraction_and_coalesces_duplicate_key() {
        let root = temp_root("async-prefix-store-early-admission");
        let store = PagedPrefixStore::new(&root);
        let entry = single_full_paged_entry(61.0);
        let wanted = spec(
            vec![61, 62],
            2,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        assert_eq!(
            wanted.payload_bytes(),
            entry.observability_stats(wanted.cached_len).payload_bytes
        );
        let queue = AsyncPrefixStoreQueue::new(1, 1024).expect("queue");
        let cancellation = AsyncPrefixStoreCancellation::default();
        let permit = match queue.try_admit(store.clone(), wanted.clone(), cancellation.clone()) {
            AsyncPrefixStoreAdmission::Admitted(permit) => *permit,
            _ => panic!("first admission must reserve before extraction"),
        };
        assert_eq!(queue.stats().pending_jobs, 1);
        assert!(matches!(
            queue.try_admit(
                store,
                wanted.clone(),
                AsyncPrefixStoreCancellation::default(),
            ),
            AsyncPrefixStoreAdmission::Coalesced
        ));
        assert_eq!(queue.stats().pending_jobs, 1);
        assert_eq!(queue.cancel_model(&wanted.model_id), 1);
        assert!(cancellation.is_cancelled());
        drop(permit);
        assert_eq!(queue.stats().pending_jobs, 0);
        assert_eq!(queue.stats().pending_bytes, 0);
        queue.shutdown();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn async_prefix_store_model_unload_cancels_pending_read_through() {
        let root = temp_root("async-prefix-store-model-cancel");
        let store = PagedPrefixStore::new(&root);
        let entry = single_full_paged_entry(65.0);
        let wanted = spec(
            vec![65, 66],
            2,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        let key = PagedPrefixStore::key_for(&wanted);
        let pending_key = (root.clone(), key);
        let cancellation = AsyncPrefixStoreCancellation::default();
        let queue = AsyncPrefixStoreQueue::new(1, 1024).expect("queue");
        queue.0.pending.lock().unwrap().insert(
            pending_key.clone(),
            AsyncPrefixPendingEntry {
                id: 1,
                spec: wanted.clone(),
                entry: Some(AsyncPrefixStorePayload::Whole(entry)),
                cancellation: cancellation.clone(),
            },
        );

        assert_eq!(queue.cancel_model("other-model"), 0);
        assert_eq!(queue.cancel_model(&wanted.model_id), 1);
        assert!(cancellation.is_cancelled());
        assert!(queue.load_pending_observed(&store, &wanted).is_none());

        queue.0.pending.lock().unwrap().remove(&pending_key);
        queue.shutdown();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn async_prefix_store_counts_failures_and_shutdown_closes_queue() {
        let parent = temp_root("async-prefix-store-failure");
        let invalid_root = parent.join("not-a-directory");
        std::fs::write(&invalid_root, b"file").unwrap();
        let entry = single_full_paged_entry(70.0);
        let wanted = spec(
            vec![70, 71],
            2,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        let queue = AsyncPrefixStoreQueue::new(1, 1024).expect("queue");
        assert_eq!(
            queue.try_enqueue(PagedPrefixStore::new(&invalid_root), wanted, entry),
            AsyncPrefixStoreSubmit::Queued
        );
        queue.shutdown();
        assert_eq!(queue.stats().failed_total, 1);

        let entry = single_full_paged_entry(80.0);
        let wanted = spec(
            vec![80, 81],
            2,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        assert_eq!(
            queue.try_enqueue(PagedPrefixStore::new(&invalid_root), wanted, entry),
            AsyncPrefixStoreSubmit::Closed
        );
        std::fs::remove_dir_all(parent).unwrap();
    }

    #[test]
    fn prefix_lru_cache_config_rejects_zero_capacity() {
        let err = PrefixLruCacheConfig::new(0).expect_err("zero capacity");
        assert!(err.to_string().contains("max_bytes must be > 0"));
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_cached_lengths_descending_filters_to_usable_lengths() {
        let root = temp_root("prefix-store-cached-lengths");
        let store = PagedPrefixStore::new(&root);
        let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
            gemma4_drafter_last_hidden: None,
        };

        let len2 = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);
        let len4 = spec(
            vec![1, 2, 3, 4],
            4,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        let mut immutable = len2.clone();
        immutable.entry_kind = PrefixEntryKind::ImmutableBlock;
        immutable.fingerprint = Some("immutable-root".to_owned());
        store.save(&len2, &entry).expect("save len2");
        store.save(&len4, &entry).expect("save len4");
        store
            .save(&immutable, &entry)
            .expect("save immutable block");

        assert_eq!(
            store.cached_lengths_descending(10).expect("cached lengths"),
            vec![4, 2]
        );
        assert_eq!(
            store
                .cached_lengths_descending(3)
                .expect("filtered cached lengths"),
            vec![2]
        );
        assert_eq!(
            store
                .cached_lengths_descending(1)
                .expect("no usable cached lengths"),
            Vec::<i32>::new()
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    fn directory_size(path: &std::path::Path) -> u64 {
        let mut total = 0_u64;
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let metadata = entry.metadata().unwrap();
            if metadata.is_dir() {
                total += directory_size(&entry.path());
            } else {
                total += metadata.len();
            }
        }
        total
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_evicts_old_entries_to_ssd_capacity() {
        let root = temp_root("prefix-store-ssd-capacity");
        let entry1 = single_full_paged_entry(1.0);
        let entry2 = single_full_paged_entry(2.0);
        let spec1 = spec(vec![1, 2], 2, None, entry1.main_layer_specs(), vec![], None);
        let spec2 = spec(vec![3, 4], 2, None, entry2.main_layer_specs(), vec![], None);

        let unlimited = PagedPrefixStore::new(&root);
        unlimited.save(&spec1, &entry1).expect("save first entry");
        let dir1 = unlimited.wp8_entry_dir(&spec1);
        let dir2 = unlimited.wp8_entry_dir(&spec2);
        let first_entry_bytes = directory_size(&dir1);
        let limited = PagedPrefixStore::new(&root).with_max_bytes(first_entry_bytes as usize + 1);
        limited.save(&spec2, &entry2).expect("save second entry");

        assert!(
            !dir1.exists(),
            "oldest entry should be evicted when the SSD cache exceeds its limit"
        );
        assert!(dir2.exists(), "newly saved entry should remain");
        assert!(
            directory_size(&root) <= first_entry_bytes + 1,
            "SSD cache directory should stay within the configured capacity"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_eviction_skips_the_dflash2_directory() {
        let root = temp_root("prefix-store-skips-dflash2");
        let dflash2 = root.join(crate::core::dflash2_prefix_store::DFLASH2_STORE_DIR);
        std::fs::create_dir_all(dflash2.join("identity")).unwrap();
        // Older than every entry and larger than the limit.
        std::fs::write(dflash2.join("identity").join("range"), vec![0_u8; 1 << 20]).unwrap();
        let entry1 = single_full_paged_entry(1.0);
        let entry2 = single_full_paged_entry(2.0);
        let spec1 = spec(vec![1, 2], 2, None, entry1.main_layer_specs(), vec![], None);
        let spec2 = spec(vec![3, 4], 2, None, entry2.main_layer_specs(), vec![], None);

        let unlimited = PagedPrefixStore::new(&root);
        unlimited.save(&spec1, &entry1).expect("save first entry");
        let dir1 = unlimited.wp8_entry_dir(&spec1);
        let dir2 = unlimited.wp8_entry_dir(&spec2);
        let first_entry_bytes = directory_size(&dir1);
        let limited = PagedPrefixStore::new(&root).with_max_bytes(first_entry_bytes as usize + 1);
        limited.save(&spec2, &entry2).expect("save second entry");

        assert!(!dir1.exists(), "the oldest entry is evicted");
        assert!(dir2.exists(), "the new entry stays");
        assert!(
            dflash2.join("identity").join("range").exists(),
            "the DFlash2 store is neither evicted nor counted"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_save_if_absent_skips_existing_matching_entry() {
        let root = temp_root("prefix-store-save-if-absent");
        let store = PagedPrefixStore::new(&root);
        let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
            gemma4_drafter_last_hidden: None,
        };
        let wanted = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);

        let (first_key, first_saved) = store.save_if_absent(&wanted, &entry).expect("first save");
        let payload_path = store.wp8_entry_dir(&wanted).join("private.safetensors");
        let payload_before = std::fs::read(&payload_path).expect("payload before");
        let (second_key, second_saved) =
            store.save_if_absent(&wanted, &entry).expect("second save");

        assert_eq!(second_key, first_key);
        assert!(first_saved);
        assert!(!second_saved);
        assert_eq!(
            std::fs::read(&payload_path).expect("payload after"),
            payload_before
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_save_if_absent_rewrites_metadata_mismatch() {
        let root = temp_root("prefix-store-save-if-absent-mismatch");
        let store = PagedPrefixStore::new(&root);
        let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
            gemma4_drafter_last_hidden: None,
        };
        let wanted = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);
        let key = store.save(&wanted, &entry).expect("save");
        let meta_path = store.wp8_entry_dir(&wanted).join(META_FILE);
        let mut meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        meta["block_size"] = serde_json::json!(4);
        std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

        let (rewritten_key, saved) = store.save_if_absent(&wanted, &entry).expect("rewrite");
        let loaded = store.load(&wanted).expect("load").expect("cache hit");

        assert_eq!(rewritten_key, key);
        assert!(saved);
        assert_eq!(loaded.main_layers.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_round_trips_mixed_payloads() {
        let root = temp_root("prefix-store-roundtrip");
        let store = PagedPrefixStore::new(&root);
        let k_data: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let v_data: Vec<f32> = k_data.iter().map(|v| v * 10.0).collect();
        let k_pages: Array = (k_data.as_slice(), (2_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (v_data.as_slice(), (2_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let conv_state: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let recurrent_state: Array = (&[5.0_f32, 6.0, 7.0, 8.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let c_kv: Array = (
            &[9.0_f32, 10.0, 11.0, 12.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let k_pe: Array = (&[13.0_f32, 14.0][..], (1_i32, 1_i32, 2_i32, 1_i32))
            .try_into()
            .unwrap();
        let mtp_k: Array = (&[15.0_f32, 16.0, 17.0][..], (1_i32, 1_i32, 3_i32, 1_i32))
            .try_into()
            .unwrap();
        let mtp_v: Array = (&[18.0_f32, 19.0, 20.0][..], (1_i32, 1_i32, 3_i32, 1_i32))
            .try_into()
            .unwrap();
        let mtp_last_hidden: Array = (&[21.0_f32, 22.0][..], (1_i32, 1_i32, 2_i32))
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![
                PrefixLayerPayload::FullPaged {
                    k_pages: k_pages.clone(),
                    v_pages: v_pages.clone(),
                },
                PrefixLayerPayload::Linear {
                    conv_state: conv_state.clone(),
                    recurrent_state: recurrent_state.clone(),
                },
                PrefixLayerPayload::Mla {
                    c_kv: c_kv.clone(),
                    k_pe: k_pe.clone(),
                },
            ],
            mtp_layers: vec![PrefixMtpLayerPayload {
                k: mtp_k.clone(),
                v: mtp_v.clone(),
            }],
            mtp_last_hidden: Some(mtp_last_hidden.clone()),
            gemma4_drafter_last_hidden: None,
        };
        let spec = spec(
            vec![1, 2, 3],
            3,
            Some("vl:fingerprint"),
            entry.main_layer_specs(),
            entry.mtp_layer_specs(),
            Some(tensor_spec(&mtp_last_hidden)),
        );
        let key = store.save(&spec, &entry).expect("save");

        let loaded = store.load(&spec).expect("load").expect("cache hit");

        assert_eq!(key, PagedPrefixStore::key_for(&spec));
        assert_eq!(loaded.main_layers.len(), 3);
        match &loaded.main_layers[0] {
            PrefixLayerPayload::FullPaged { k_pages, v_pages } => {
                assert_eq!(k_pages.to_vec::<f32>().unwrap(), k_data);
                assert_eq!(v_pages.to_vec::<f32>().unwrap(), v_data);
            }
            other => panic!("expected full paged layer, got {other:?}"),
        }
        match &loaded.main_layers[1] {
            PrefixLayerPayload::Linear {
                conv_state,
                recurrent_state,
            } => {
                assert_eq!(
                    conv_state.to_vec::<f32>().unwrap(),
                    vec![1.0, 2.0, 3.0, 4.0]
                );
                assert_eq!(
                    recurrent_state.to_vec::<f32>().unwrap(),
                    vec![5.0, 6.0, 7.0, 8.0]
                );
            }
            other => panic!("expected linear layer, got {other:?}"),
        }
        match &loaded.main_layers[2] {
            PrefixLayerPayload::Mla { c_kv, k_pe } => {
                assert_eq!(c_kv.to_vec::<f32>().unwrap(), vec![9.0, 10.0, 11.0, 12.0]);
                assert_eq!(k_pe.to_vec::<f32>().unwrap(), vec![13.0, 14.0]);
            }
            other => panic!("expected MLA layer, got {other:?}"),
        }
        assert_eq!(loaded.mtp_layers.len(), 1);
        assert_eq!(
            loaded.mtp_layers[0].k.to_vec::<f32>().unwrap(),
            vec![15.0, 16.0, 17.0]
        );
        assert_eq!(
            loaded.mtp_layers[0].v.to_vec::<f32>().unwrap(),
            vec![18.0, 19.0, 20.0]
        );
        assert_eq!(
            loaded
                .mtp_last_hidden
                .as_ref()
                .expect("last hidden")
                .to_vec::<f32>()
                .unwrap(),
            vec![21.0, 22.0]
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_rejects_metadata_mismatch() {
        let root = temp_root("prefix-store-mismatch");
        let store = PagedPrefixStore::new(&root);
        let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
            gemma4_drafter_last_hidden: None,
        };
        let wanted = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);
        store.save(&wanted, &entry).expect("save");
        let meta_path = store.wp8_entry_dir(&wanted).join(META_FILE);
        let mut meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        meta["block_size"] = serde_json::json!(4);
        std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

        let loaded = store.load(&wanted).expect("load");

        assert!(loaded.is_none(), "tampered metadata must miss");
        std::fs::remove_dir_all(root).unwrap();
    }

    use ironmlx_lm::core::cache::prefix_segments::{
        materialize_prefix_entry, prefix_segment_layout, PrefixKvSegment, PrefixSegmentId,
        PrefixSegmentLayer, PrefixSegmentLayout, PrefixSegmentRef, SegmentedPrefixEntry,
    };

    const SEG_BLOCK: i32 = 128;
    const SEG_HEADS: i32 = 8;
    const SEG_DIM: i32 = 128;
    const MIB: usize = 1 << 20;

    /// `[pages, H, block, D]` f32: 512 KiB per page.
    fn seg_pages(pages: i32, evaluated: bool) -> Array {
        let pages = Array::zeros(&[pages, SEG_HEADS, SEG_BLOCK, SEG_DIM], Dtype::Float32).unwrap();
        if evaluated {
            pages.eval().unwrap();
        }
        pages
    }

    /// One paged layer's tail pages plus a linear layer holding
    /// `private_mib` MiB of recurrent state.
    fn private_entry(tail_pages: i32, private_mib: i32, evaluated: bool) -> PagedPrefixEntry {
        let conv_state = Array::zeros(&[1, 4, 8], Dtype::Float32).unwrap();
        let recurrent_state = Array::zeros(&[1, private_mib, 512, 512], Dtype::Float32).unwrap();
        if evaluated {
            conv_state.eval().unwrap();
            recurrent_state.eval().unwrap();
        }
        PagedPrefixEntry {
            main_layers: vec![
                PrefixLayerPayload::FullPaged {
                    k_pages: seg_pages(tail_pages, evaluated),
                    v_pages: seg_pages(tail_pages, evaluated),
                },
                PrefixLayerPayload::Linear {
                    conv_state,
                    recurrent_state,
                },
            ],
            mtp_layers: vec![],
            mtp_last_hidden: None,
            gemma4_drafter_last_hidden: None,
        }
    }

    fn seg_spec(model: &str, tokens: Vec<i32>, entry: &PagedPrefixEntry) -> PagedPrefixKeySpec {
        let cached_len = tokens.len() as i32;
        let mut spec = spec(
            tokens,
            cached_len,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        spec.model_id = model.to_owned();
        spec.block_size = SEG_BLOCK;
        spec
    }

    fn seg_layout(model: &str, private_mib: i32) -> PrefixSegmentLayout {
        prefix_segment_layout(&seg_spec(
            model,
            vec![0],
            &private_entry(1, private_mib, false),
        ))
    }

    fn shared_segment(
        model: &str,
        tokens: i32,
        private_mib: i32,
        evaluated: bool,
    ) -> Arc<PrefixKvSegment> {
        let pages = tokens / SEG_BLOCK;
        Arc::new(PrefixKvSegment {
            id: PrefixSegmentId::new(),
            layout: seg_layout(model, private_mib),
            start: 0,
            stop: tokens,
            layers: vec![(
                0,
                PrefixSegmentLayer::Paged {
                    k: seg_pages(pages, evaluated),
                    v: seg_pages(pages, evaluated),
                },
            )],
        })
    }

    /// An entry using `[0, stop)` of `segment` plus 5 tail tokens; `salt`
    /// tells sharers' tails apart.
    fn sharer(
        model: &str,
        segment: &Arc<PrefixKvSegment>,
        stop: i32,
        salt: i32,
        private_mib: i32,
        evaluated: bool,
    ) -> (PagedPrefixKeySpec, SegmentedPrefixEntry) {
        let entry = SegmentedPrefixEntry {
            segments: vec![PrefixSegmentRef {
                segment: segment.clone(),
                stop,
            }],
            aligned_len: stop,
            private: private_entry(1, private_mib, evaluated),
        };
        let mut tokens = (0..stop).collect::<Vec<_>>();
        tokens.extend((0..5).map(|i| -(salt * 100 + i)));
        let materialized =
            materialize_prefix_entry(&entry, tokens.len() as i32, SEG_BLOCK).unwrap();
        (seg_spec(model, tokens, &materialized), entry)
    }

    fn private_bytes(entry: &SegmentedPrefixEntry) -> usize {
        entry.private.observability_stats(0).payload_bytes
    }

    fn big_cache() -> PrefixLruCache {
        PrefixLruCache::new(PrefixLruCacheConfig::new(1 << 40).unwrap()).unwrap()
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn shared_segment_is_charged_once() {
        let mut cache = big_cache();
        let s = shared_segment("m", 1024, 64, false);
        let (spec_a, a) = sharer("m", &s, 1024, 1, 64, false);
        let (spec_b, b) = sharer("m", &s, 1024, 2, 64, false);
        let private = private_bytes(&a);
        assert!(private >= 64 * MIB);

        let inserted_a = cache.insert_segmented(spec_a, a).unwrap();
        let inserted_b = cache.insert_segmented(spec_b, b).unwrap();

        assert_eq!(cache.total_bytes(), s.payload_bytes() + 2 * private);
        assert_eq!(inserted_a.charged_bytes, s.payload_bytes() + private);
        assert_eq!(inserted_b.charged_bytes, private);
        // The logical size is the whole entry, as before.
        assert_eq!(inserted_b.stats.payload_bytes, s.payload_bytes() + private);
        assert_eq!(
            cache.segment(s.id).map(|seg| seg.id),
            Some(s.id),
            "the cached segment is reachable by id"
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn health_reports_prefix_segment_sharing() {
        let mut cache = big_cache();
        assert_eq!(cache.health(), PrefixLruHealth::default());
        let s = shared_segment("m", 1024, 64, false);
        let (spec_a, a) = sharer("m", &s, 1024, 1, 64, false);
        let (spec_b, b) = sharer("m", &s, 1024, 2, 64, false);
        let private = private_bytes(&a);
        cache.insert_segmented(spec_a, a).unwrap();
        cache.insert_segmented(spec_b, b).unwrap();

        let health = cache.health();
        assert_eq!(health.entry_count, 2);
        assert_eq!(health.segment_count, 1);
        assert_eq!(health.segment_bytes, s.payload_bytes());
        assert_eq!(health.logical_bytes, 2 * (s.payload_bytes() + private));
        // The shared segment is held once, so the whole-copy cost is larger.
        assert!(health.logical_bytes > health.segment_bytes + 2 * private);
        assert_eq!(cache.total_bytes(), health.segment_bytes + 2 * private);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn evicting_one_sharer_keeps_the_segment() {
        let mut cache = big_cache();
        let s = shared_segment("m", 1024, 64, false);
        let (spec_a, a) = sharer("m", &s, 1024, 1, 64, false);
        let (spec_b, b) = sharer("m", &s, 1024, 2, 64, false);
        let private = private_bytes(&a);
        let key_a = cache.insert_segmented(spec_a, a).unwrap().key;
        let key_b = cache.insert_segmented(spec_b, b).unwrap().key;

        cache.remove_entry(&key_a);
        assert_eq!(cache.total_bytes(), s.payload_bytes() + private);
        assert!(cache.segment(s.id).is_some());

        cache.remove_entry(&key_b);
        assert_eq!(cache.total_bytes(), 0);
        assert!(cache.segments.is_empty());
        assert!(cache.segment(s.id).is_none());
    }

    /// `segment` again under the same id, with fresh arrays of equal data.
    fn segment_copy(segment: &PrefixKvSegment, stop: i32) -> Arc<PrefixKvSegment> {
        let pages = (stop - segment.start) / SEG_BLOCK;
        Arc::new(PrefixKvSegment {
            id: segment.id,
            layout: segment.layout.clone(),
            start: segment.start,
            stop,
            layers: vec![(
                0,
                PrefixSegmentLayer::Paged {
                    k: seg_pages(pages, false),
                    v: seg_pages(pages, false),
                },
            )],
        })
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn insert_reuses_the_cached_copy_of_a_segment() {
        let mut cache = big_cache();
        let s = shared_segment("m", 1024, 64, true);
        let (spec_a, a) = sharer("m", &s, 1024, 1, 64, true);
        let copy = segment_copy(&s, s.stop);
        let (spec_b, b) = sharer("m", &copy, 1024, 2, 64, false);
        assert!(!Arc::ptr_eq(&copy, &s));
        let private = private_bytes(&a);
        cache.insert_segmented(spec_a, a).unwrap();
        let key_b = cache.insert_segmented(spec_b, b).unwrap().key;

        let stored = &cache.entries.get(&key_b).unwrap().segmented.segments[0];
        assert!(Arc::ptr_eq(&stored.segment, &s));
        assert_eq!(cache.total_bytes(), s.payload_bytes() + 2 * private);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn insert_rejects_a_segment_that_contradicts_the_cached_one() {
        let mut cache = big_cache();
        let s = shared_segment("m", 1024, 64, true);
        let (spec_a, a) = sharer("m", &s, 1024, 1, 64, true);
        let shorter = segment_copy(&s, 512);
        let (spec_b, b) = sharer("m", &shorter, 512, 2, 64, false);
        cache.insert_segmented(spec_a, a).unwrap();
        let before = cache.total_bytes();

        assert!(cache.insert_segmented(spec_b.clone(), b).is_err());
        assert!(!cache.contains(&spec_b));
        assert_eq!(cache.total_bytes(), before);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn reclaim_counts_only_bytes_actually_freed() {
        let mut cache = big_cache();
        let s = shared_segment("m", 1024, 64, false);
        let (spec_a, a) = sharer("m", &s, 1024, 1, 64, false);
        let (spec_b, b) = sharer("m", &s, 1024, 2, 64, false);
        let private = private_bytes(&a);
        cache.insert_segmented(spec_a.clone(), a).unwrap();
        cache.insert_segmented(spec_b.clone(), b).unwrap();

        // Room for one private payload: the oldest sharer goes first and
        // frees only its own bytes.
        let before = cache.total_bytes();
        let reclaim = cache.reclaim(private, None);
        assert_eq!(reclaim.bytes, before - cache.total_bytes());
        assert_eq!(reclaim.bytes, private);
        assert!(!cache.contains(&spec_a) && cache.contains(&spec_b));

        // The last sharer takes the segment with it.
        let before = cache.total_bytes();
        let reclaim = cache.reclaim(private, None);
        assert_eq!(reclaim.bytes, before - cache.total_bytes());
        assert_eq!(reclaim.bytes, s.payload_bytes() + private);
        assert!(cache.is_empty());
    }

    fn settled_active_bytes() -> usize {
        mlx::transforms::synchronize().unwrap();
        mlx::transforms::clear_cache();
        mlx::memory::snapshot().active_bytes
    }

    fn assert_close(mlx_freed: isize, counted_freed: isize, what: &str) {
        assert!(
            (mlx_freed - counted_freed).unsigned_abs() <= MIB,
            "{what}: MLX freed {mlx_freed} bytes, the cache counted {counted_freed}"
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn memory_given_back_matches_mlx() {
        let mut cache = big_cache();
        // 32 pages: 32 MiB of K/V. `half` uses its first 16 pages only.
        let s = shared_segment("m", 4096, 4, true);
        let segment_bytes = s.payload_bytes();
        let (spec_half, half) = sharer("m", &s, 2048, 1, 4, true);
        let (spec_whole, whole) = sharer("m", &s, 4096, 2, 4, true);
        let key_half = cache.insert_segmented(spec_half, half).unwrap().key;
        let key_whole = cache.insert_segmented(spec_whole, whole).unwrap().key;
        drop(s);

        let active = settled_active_bytes();
        let counted = cache.total_bytes();
        cache.remove_entry(&key_whole);
        let mlx_freed = active as isize - settled_active_bytes() as isize;
        let counted_freed = counted as isize - cache.total_bytes() as isize;
        assert_close(mlx_freed, counted_freed, "removing the whole-segment entry");
        // `half` still holds the segment: neither MLX nor the count lets it go.
        assert!(mlx_freed < segment_bytes as isize, "{mlx_freed}");
        assert!(counted_freed < segment_bytes as isize, "{counted_freed}");

        let active = settled_active_bytes();
        let counted = cache.total_bytes();
        cache.remove_entry(&key_half);
        let mlx_freed = active as isize - settled_active_bytes() as isize;
        let counted_freed = counted as isize - cache.total_bytes() as isize;
        assert_close(mlx_freed, counted_freed, "removing the last sharer");
        assert!(counted_freed >= segment_bytes as isize);
        assert_eq!(cache.total_bytes(), 0);
    }

    /// Two paged layers of `[1, SEG_HEADS, ·, SEG_DIM]` f32 K/V.
    fn live_paged_caches(tokens: i32) -> Vec<ironmlx_lm::core::cache::LayerCache> {
        use ironmlx_lm::core::cache::{KVCache, LayerCache};
        (0..2)
            .map(|_| {
                let mut kv = KVCache::new(1, SEG_HEADS, SEG_DIM, SEG_DIM, Dtype::Float32, 4096)
                    .with_step(SEG_BLOCK)
                    .with_paged(SEG_BLOCK, 32)
                    .expect("paged");
                let kv_tokens =
                    mlx::ops::ones(&[1, SEG_HEADS, tokens, SEG_DIM], Dtype::Float32).unwrap();
                kv.update_and_fetch(&kv_tokens, &kv_tokens, &[tokens])
                    .expect("fill");
                LayerCache::Full(kv)
            })
            .collect()
    }

    fn extend_live(caches: &mut [ironmlx_lm::core::cache::LayerCache], tokens: i32) {
        for cache in caches {
            let ironmlx_lm::core::cache::LayerCache::Full(kv) = cache else {
                unreachable!()
            };
            let kv_tokens =
                mlx::ops::ones(&[1, SEG_HEADS, tokens, SEG_DIM], Dtype::Float32).unwrap();
            kv.update_and_fetch(&kv_tokens, &kv_tokens, &[tokens])
                .expect("extend");
        }
    }

    /// Export `[from, aligned)` as a new segment, then key it with the spec
    /// of `refs` + that segment + the private tail.
    fn exported_entry(
        caches: &[ironmlx_lm::core::cache::LayerCache],
        refs: Vec<PrefixSegmentRef>,
        from: i32,
    ) -> (
        PagedPrefixKeySpec,
        SegmentedPrefixEntry,
        Arc<PrefixKvSegment>,
    ) {
        let (segment, private, cached_len) =
            ironmlx_lm::core::cache::export_prefix_segment_for_row(
                caches,
                0,
                from,
                SEG_BLOCK,
                PrefixSegmentLayout("placeholder".into()),
            )
            .unwrap()
            .unwrap();
        let mut segments = refs;
        let segment = segment.expect("new segment");
        let stop = segment.stop;
        segments.push(PrefixSegmentRef {
            segment: Arc::new(segment),
            stop,
        });
        let mut entry = SegmentedPrefixEntry {
            segments,
            aligned_len: stop,
            private,
        };
        let materialized = materialize_prefix_entry(&entry, cached_len, SEG_BLOCK).unwrap();
        let spec = seg_spec("m", (0..cached_len).collect(), &materialized);
        drop(materialized);
        // The layout ignores lengths, so it is known only once a spec exists.
        let layout = prefix_segment_layout(&spec);
        let last = entry.segments.pop().unwrap();
        let fresh = Arc::new(PrefixKvSegment {
            layout,
            ..Arc::try_unwrap(last.segment).unwrap()
        });
        entry.segments.push(PrefixSegmentRef {
            segment: fresh.clone(),
            stop,
        });
        (spec, entry, fresh)
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn deleting_the_entry_that_created_a_shared_segment_frees_what_it_counts() {
        let mut cache = big_cache();
        let mut caches = live_paged_caches(1024 + 5);
        // `first` creates S = [0, 1024); `second` references S and adds
        // [1024, 2048) plus its own tail.
        let (spec_first, first, s) = exported_entry(&caches, vec![], 0);
        extend_live(&mut caches, 1024 + 2);
        let (spec_second, second, _) = exported_entry(
            &caches,
            vec![PrefixSegmentRef {
                segment: s.clone(),
                stop: 1024,
            }],
            1024,
        );
        drop(caches);
        let segment_bytes = s.payload_bytes();
        drop(s);
        let key_first = cache.insert_segmented(spec_first, first).unwrap().key;
        let key_second = cache.insert_segmented(spec_second, second).unwrap().key;

        let active = settled_active_bytes();
        let counted = cache.total_bytes();
        cache.remove_entry(&key_first);
        let mlx_freed = active as isize - settled_active_bytes() as isize;
        let counted_freed = counted as isize - cache.total_bytes() as isize;
        assert_close(mlx_freed, counted_freed, "removing the segment's creator");
        assert!(counted_freed > 0 && counted_freed < segment_bytes as isize);

        let active = settled_active_bytes();
        let counted = cache.total_bytes();
        cache.remove_entry(&key_second);
        let mlx_freed = active as isize - settled_active_bytes() as isize;
        let counted_freed = counted as isize - cache.total_bytes() as isize;
        assert_close(mlx_freed, counted_freed, "removing the last sharer");
        assert!(counted_freed > segment_bytes as isize);
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn plain_insert_keeps_the_entry_private() {
        let mut cache = big_cache();
        let entry = single_full_paged_entry(1.0);
        let spec = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);
        let inserted = cache.insert(spec.clone(), entry).unwrap();
        assert_eq!(inserted.charged_bytes, inserted.stats.payload_bytes);
        assert!(cache.segments.is_empty());
        let (_, lineage) = cache.load_segmented(&spec).unwrap().expect("hit");
        assert!(lineage.refs.is_empty());
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_exports_own_their_buffers() {
        use ironmlx_lm::core::cache::{KVCache, LayerCache, TurboQuantKVBits};
        for kind in ["paged", "turboquant"] {
            let caches: Vec<LayerCache> = (0..2)
                .map(|_| {
                    let base = KVCache::new(1, 32, SEG_DIM, SEG_DIM, Dtype::Float32, 8192)
                        .with_step(SEG_BLOCK);
                    let mut kv = match kind {
                        "paged" => base.with_paged(SEG_BLOCK, 40).expect("paged"),
                        _ => base
                            .with_turboquant(TurboQuantKVBits::K3V4)
                            .expect("turboquant"),
                    };
                    let x = mlx::ops::ones(&[1, 32, 4096 + 5, SEG_DIM], Dtype::Float32).unwrap();
                    kv.update_and_fetch(&x, &x, &[4096 + 5]).expect("fill");
                    LayerCache::Full(kv)
                })
                .collect();
            let (segment, private, _) = ironmlx_lm::core::cache::export_prefix_segment_for_row(
                &caches,
                0,
                0,
                SEG_BLOCK,
                PrefixSegmentLayout("x".into()),
            )
            .unwrap()
            .unwrap();
            let segment = segment.expect("segment");
            let private_bytes = private.observability_stats(0).payload_bytes;
            let segment_bytes = segment.payload_bytes();
            drop(caches);
            let active = settled_active_bytes();
            drop(private);
            let after_private = settled_active_bytes();
            drop(segment);
            let after_segment = settled_active_bytes();
            assert_close(
                active as isize - after_private as isize,
                private_bytes as isize,
                &format!("{kind} private tail"),
            );
            assert_close(
                after_private as isize - after_segment as isize,
                segment_bytes as isize,
                &format!("{kind} segment"),
            );
        }
    }
}
