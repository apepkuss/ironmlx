//! Per-layer cache types for inference. See P2 spec § 1 for scope.

pub mod kv_cache;

pub use kv_cache::KVCache;
