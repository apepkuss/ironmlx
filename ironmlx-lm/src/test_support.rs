//! Explicit helpers for tests of downstream execution drivers.
use crate::core::constrained::ConstraintTokenizer;
/// Build the same byte-level vocabulary used by the model constraint tests.
pub fn byte_level_constraint_tokenizer() -> crate::Result<ConstraintTokenizer> {
    ConstraintTokenizer::byte_level()
}

/// Inspect stream-cache residency without exposing cache internals to production.
pub fn paged_stream_cache_lengths(cache: &crate::core::cache::PagedKVCache) -> (usize, usize) {
    crate::core::cache::paged_kv::stream_cache_lengths(cache)
}
/// Query direct installation eligibility for file-backend qualification.
pub fn paged_can_direct_install_prefix_pages(
    cache: &crate::core::cache::PagedKVCache,
    pages: i32,
) -> bool {
    crate::core::cache::paged_kv::can_direct_install_prefix_pages(cache, pages)
}

/// Inspect offloaded references for the runtime file-backend tests.
pub fn paged_has_nonresident_referenced_pages(
    cache: &crate::core::cache::PagedKVCache,
    offsets: &[i32],
) -> bool {
    crate::core::cache::paged_kv::has_nonresident_referenced_pages(cache, offsets)
}
