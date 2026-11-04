//! Per-layer cache types for inference. See P2 spec § 1 for scope.

pub mod active_kv;
pub mod prefix_save;
pub mod prefix_store;
pub(crate) mod prefix_stream;

pub use active_kv::{
    default_active_kv_offload_dir, timed, ActiveKvOffloadConfig, ActiveKvOffloadHealth,
    ActiveKvOffloadSharedStats, ActiveKvOffloadStatus, ActiveKvOffloadStore, ActiveKvPageResidency,
    ActiveKvResidencyState, ActiveKvResidencySummary, ActiveKvResidencyTracker,
    ActiveKvStoredPayload,
};
pub use prefix_store::{
    cancel_process_async_prefix_store_model, export_segmented_prefix_for_row,
    load_segmented_prefix_entry, load_store_prefix_entry, process_async_prefix_store_queue,
    process_prefix_lru_cache, process_shared_prefix_lru_cache, reclaim_process_prefix_lru_bytes,
    segmented_entry_lineage, shutdown_process_async_prefix_store_queue, AsyncPrefixStoreAdmission,
    AsyncPrefixStoreCancellation, AsyncPrefixStorePermit, AsyncPrefixStoreQueue,
    AsyncPrefixStoreStats, AsyncPrefixStoreSubmit, DFlash2PrefixDiskConfig, PagedPrefixCacheConfig,
    PagedPrefixLoadStatus, PagedPrefixSegmentedLoadResult, PagedPrefixStore, PrefixLruCache,
    PrefixLruCacheConfig, PrefixLruHealth, PrefixLruInsertResult, PrefixLruInsertStatus,
    PrefixLruLoaded, PrefixLruReclaim, PrefixSegmentLookup, PrefixStoreHit, PrefixStoreLoaded,
    SharedPrefixLruCache, DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE,
};
