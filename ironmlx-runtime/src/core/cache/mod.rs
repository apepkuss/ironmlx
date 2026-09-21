//! Per-layer cache types for inference. See P2 spec § 1 for scope.

pub mod active_kv;
pub mod prefix_store;

pub use active_kv::{
    default_active_kv_offload_dir, timed, ActiveKvOffloadConfig, ActiveKvOffloadHealth,
    ActiveKvOffloadSharedStats, ActiveKvOffloadStatus, ActiveKvOffloadStore, ActiveKvPageResidency,
    ActiveKvResidencyState, ActiveKvResidencySummary, ActiveKvResidencyTracker,
    ActiveKvStoredPayload,
};
pub use prefix_store::{
    cancel_process_async_prefix_store_model, process_async_prefix_store_queue,
    process_shared_prefix_lru_cache, shrink_process_prefix_lru_caches,
    shutdown_process_async_prefix_store_queue, AsyncPrefixStoreAdmission,
    AsyncPrefixStoreCancellation, AsyncPrefixStorePermit, AsyncPrefixStoreQueue,
    AsyncPrefixStoreStats, AsyncPrefixStoreSubmit, PagedPrefixCacheConfig, PagedPrefixLoadStatus,
    PagedPrefixStore, PrefixLruCache, PrefixLruCacheConfig, PrefixLruInsertResult,
    PrefixLruInsertStatus, SharedPrefixLruCache, DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE,
};
