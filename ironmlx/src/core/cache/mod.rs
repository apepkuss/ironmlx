//! Per-layer cache types for inference. See P2 spec § 1 for scope.

pub mod gated_delta;
pub mod kv_cache;

pub use gated_delta::GatedDeltaCache;
pub use kv_cache::KVCache;
