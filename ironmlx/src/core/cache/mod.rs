//! Per-layer cache types for inference. See P2 spec § 1 for scope.

pub mod gated_delta;
pub mod kv_cache;
pub mod mtp_cache;
pub mod paged_kv;
pub mod prefix_store;
pub mod turboquant_kv;

pub use gated_delta::{GatedDeltaCache, GatedDeltaCacheSnapshot};
pub use kv_cache::{KVCache, KVCacheSnapshot};
pub use mtp_cache::{MtpCache, MtpCacheSnapshot};
pub use paged_kv::PagedKVCache;
pub use prefix_store::{
    PagedPrefixCacheConfig, PagedPrefixEntry, PagedPrefixEntryStats, PagedPrefixKeySpec,
    PagedPrefixLayer, PagedPrefixLoadStatus, PagedPrefixStore, PrefixLayerKind, PrefixLayerPayload,
    PrefixLayerSpec, PrefixLruCache, PrefixLruCacheConfig, PrefixLruInsertResult,
    PrefixLruInsertStatus, PrefixMtpLayerPayload, PrefixMtpLayerSpec, PrefixTensorSpec,
};
pub use turboquant_kv::{TurboQuantKVBits, TurboQuantKVCache};
