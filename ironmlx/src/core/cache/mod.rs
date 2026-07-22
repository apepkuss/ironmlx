//! Per-layer cache types for inference. See P2 spec § 1 for scope.

pub mod active_kv;
pub mod gated_delta;
pub mod kv_cache;
pub mod mtp_cache;
pub mod paged_kv;
pub mod prefix_store;
pub mod turboquant_kv;

pub use active_kv::{
    default_active_kv_offload_dir, timed, ActiveKvEntryChunkReader, ActiveKvLayerChunk,
    ActiveKvLayerChunkKind, ActiveKvLayerChunkPayload, ActiveKvOffloadConfig,
    ActiveKvOffloadHealth, ActiveKvOffloadSharedStats, ActiveKvOffloadStatus, ActiveKvOffloadStore,
    ActiveKvPageResidency, ActiveKvResidencyState, ActiveKvResidencySummary,
    ActiveKvResidencyTracker, ActiveKvStoredPayload,
};
pub use gated_delta::{GatedDeltaCache, GatedDeltaCacheSnapshot};
pub use kv_cache::{KVCache, KVCacheSnapshot};
pub use mtp_cache::{MtpCache, MtpCacheSnapshot};
pub use paged_kv::{
    PagedKVCache, PagedKvBlockOwner, PagedKvHotColdConfig, PagedKvHotColdSummary,
    PagedKvImmutableBlockHandle, PagedKvPhysicalStats,
};
pub use prefix_store::{
    cancel_process_async_prefix_store_model, process_async_prefix_store_queue,
    process_shared_prefix_lru_cache, shrink_process_prefix_lru_caches,
    shutdown_process_async_prefix_store_queue, AsyncPrefixStoreAdmission,
    AsyncPrefixStoreCancellation, AsyncPrefixStorePermit, AsyncPrefixStoreQueue,
    AsyncPrefixStoreStats, AsyncPrefixStoreSubmit, PagedPrefixCacheConfig, PagedPrefixEntry,
    PagedPrefixEntryStats, PagedPrefixKeySpec, PagedPrefixLayer, PagedPrefixLoadStatus,
    PagedPrefixStore, PrefixLayerKind, PrefixLayerPayload, PrefixLayerSpec, PrefixLruCache,
    PrefixLruCacheConfig, PrefixLruInsertResult, PrefixLruInsertStatus, PrefixMtpLayerPayload,
    PrefixMtpLayerSpec, PrefixTensorSpec, SharedPrefixLruCache,
    DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE,
};
pub use turboquant_kv::{TurboQuantKVBits, TurboQuantKVCache, TurboQuantPrefixLayer};
