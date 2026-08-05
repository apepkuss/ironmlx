use std::sync::{Arc, Mutex, MutexGuard};

use sha2::{Digest, Sha256};

const MAX_ENTRIES: usize = 64;
const MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolPromptCacheStats {
    pub exact_hits: u64,
    pub prefix_hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub reused_tokens: u64,
    pub entries: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SafeBoundary {
    pub byte_offset: usize,
    pub token_count: usize,
}

pub(crate) struct PrefixMatch {
    pub byte_offset: usize,
    pub token_ids: Vec<u32>,
    pub boundaries: Vec<SafeBoundary>,
}

pub(crate) enum CacheMatch {
    Exact(Vec<u32>),
    Prefix(PrefixMatch),
    Miss,
}

pub(crate) struct CacheLookupKey<'a> {
    namespace: &'a [u8; 32],
    messages: &'a [u8],
    kwargs: &'a [u8],
    message_digest: [u8; 32],
    kwargs_digest: [u8; 32],
}

impl<'a> CacheLookupKey<'a> {
    pub(crate) fn new(namespace: &'a [u8; 32], messages: &'a [u8], kwargs: &'a [u8]) -> Self {
        Self {
            namespace,
            messages,
            kwargs,
            message_digest: Sha256::digest(messages).into(),
            kwargs_digest: Sha256::digest(kwargs).into(),
        }
    }
}

pub(crate) struct CacheInsertKey {
    namespace: [u8; 32],
    messages: Vec<u8>,
    kwargs: Vec<u8>,
    message_digest: [u8; 32],
    kwargs_digest: [u8; 32],
}

impl CacheInsertKey {
    pub(crate) fn new(namespace: [u8; 32], messages: Vec<u8>, kwargs: Vec<u8>) -> Self {
        let message_digest = Sha256::digest(&messages).into();
        let kwargs_digest = Sha256::digest(&kwargs).into();
        Self {
            namespace,
            messages,
            kwargs,
            message_digest,
            kwargs_digest,
        }
    }
}

struct Entry {
    id: u64,
    namespace: [u8; 32],
    messages: Vec<u8>,
    kwargs: Vec<u8>,
    message_digest: [u8; 32],
    kwargs_digest: [u8; 32],
    prompt: Arc<str>,
    token_ids: Arc<[u32]>,
    boundaries: Arc<[SafeBoundary]>,
    last_used: u64,
    charge: usize,
}

impl Entry {
    fn exact_key_matches(&self, key: &CacheLookupKey<'_>) -> bool {
        self.namespace == *key.namespace
            && self.message_digest == key.message_digest
            && self.kwargs_digest == key.kwargs_digest
            && self.messages == key.messages
            && self.kwargs == key.kwargs
    }

    fn bucket_matches(&self, key: &CacheLookupKey<'_>) -> bool {
        self.namespace == *key.namespace
            && self.kwargs_digest == key.kwargs_digest
            && self.kwargs == key.kwargs
    }

    fn owned_key_matches(&self, key: &CacheInsertKey) -> bool {
        self.namespace == key.namespace
            && self.message_digest == key.message_digest
            && self.kwargs_digest == key.kwargs_digest
            && self.messages == key.messages
            && self.kwargs == key.kwargs
    }
}

#[derive(Default)]
struct State {
    entries: Vec<Entry>,
    clock: u64,
    next_id: u64,
    bytes: usize,
    stats: ToolPromptCacheStats,
}

pub(crate) struct ToolPromptCache {
    state: Mutex<State>,
    max_entries: usize,
    max_bytes: usize,
}

impl Default for ToolPromptCache {
    fn default() -> Self {
        Self::new(MAX_ENTRIES, MAX_BYTES)
    }
}

impl ToolPromptCache {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            state: Mutex::new(State::default()),
            max_entries,
            max_bytes,
        }
    }

    pub(crate) fn lookup_exact(&self, key: &CacheLookupKey<'_>) -> Option<Vec<u32>> {
        let mut state = self.lock();
        let index = state
            .entries
            .iter()
            .position(|entry| entry.exact_key_matches(key))?;
        let token_ids = Arc::clone(&state.entries[index].token_ids);
        state.clock = state.clock.wrapping_add(1);
        state.entries[index].last_used = state.clock;
        state.stats.exact_hits = state.stats.exact_hits.saturating_add(1);
        drop(state);
        Some(token_ids.to_vec())
    }

    pub(crate) fn lookup_after_render(&self, key: &CacheLookupKey<'_>, prompt: &str) -> CacheMatch {
        let candidates = {
            let mut state = self.lock();
            if let Some(index) = state
                .entries
                .iter()
                .position(|entry| entry.exact_key_matches(key))
            {
                let token_ids = Arc::clone(&state.entries[index].token_ids);
                state.clock = state.clock.wrapping_add(1);
                state.entries[index].last_used = state.clock;
                state.stats.exact_hits = state.stats.exact_hits.saturating_add(1);
                drop(state);
                return CacheMatch::Exact(token_ids.to_vec());
            }
            state
                .entries
                .iter()
                .filter(|entry| entry.bucket_matches(key))
                .map(|entry| PrefixCandidate {
                    id: entry.id,
                    prompt: Arc::clone(&entry.prompt),
                    token_ids: Arc::clone(&entry.token_ids),
                    boundaries: Arc::clone(&entry.boundaries),
                })
                .collect::<Vec<_>>()
        };

        let best = candidates
            .iter()
            .filter_map(|candidate| {
                let common = common_prefix_len(&candidate.prompt, prompt);
                let boundary = candidate
                    .boundaries
                    .iter()
                    .rev()
                    .find(|boundary| boundary.byte_offset <= common)?;
                Some((candidate, *boundary))
            })
            .max_by_key(|(_, boundary)| boundary.byte_offset);

        if let Some((candidate, boundary)) = best {
            let token_ids = candidate.token_ids[..boundary.token_count].to_vec();
            let boundaries = candidate
                .boundaries
                .iter()
                .copied()
                .take_while(|item| item.byte_offset <= boundary.byte_offset)
                .collect();
            let mut state = self.lock();
            state.clock = state.clock.wrapping_add(1);
            let clock = state.clock;
            if let Some(entry) = state
                .entries
                .iter_mut()
                .find(|entry| entry.id == candidate.id)
            {
                entry.last_used = clock;
            }
            state.stats.prefix_hits = state.stats.prefix_hits.saturating_add(1);
            state.stats.reused_tokens = state
                .stats
                .reused_tokens
                .saturating_add(boundary.token_count as u64);
            CacheMatch::Prefix(PrefixMatch {
                byte_offset: boundary.byte_offset,
                token_ids,
                boundaries,
            })
        } else {
            let mut state = self.lock();
            state.stats.misses = state.stats.misses.saturating_add(1);
            CacheMatch::Miss
        }
    }

    pub(crate) fn insert(
        &self,
        key: CacheInsertKey,
        prompt: String,
        token_ids: Vec<u32>,
        boundaries: Vec<SafeBoundary>,
    ) {
        let mut state = self.lock();
        if state
            .entries
            .iter()
            .any(|entry| entry.owned_key_matches(&key))
        {
            return;
        }
        state.clock = state.clock.wrapping_add(1);
        state.next_id = state.next_id.wrapping_add(1);
        let charge = key
            .namespace
            .len()
            .saturating_add(key.messages.len())
            .saturating_add(key.kwargs.len())
            .saturating_add(prompt.len())
            .saturating_add(token_ids.len().saturating_mul(size_of::<u32>()))
            .saturating_add(boundaries.len().saturating_mul(size_of::<SafeBoundary>()));
        let entry = Entry {
            id: state.next_id,
            namespace: key.namespace,
            messages: key.messages,
            kwargs: key.kwargs,
            message_digest: key.message_digest,
            kwargs_digest: key.kwargs_digest,
            prompt: prompt.into(),
            token_ids: token_ids.into(),
            boundaries: boundaries.into(),
            last_used: state.clock,
            charge,
        };
        if self.max_entries == 0 || entry.charge > self.max_bytes {
            return;
        }
        while state.entries.len() >= self.max_entries
            || state.bytes.saturating_add(entry.charge) > self.max_bytes
        {
            let Some((index, _)) = state
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, candidate)| candidate.last_used)
            else {
                break;
            };
            let evicted = state.entries.swap_remove(index);
            state.bytes = state.bytes.saturating_sub(evicted.charge);
            state.stats.evictions = state.stats.evictions.saturating_add(1);
        }
        state.bytes = state.bytes.saturating_add(entry.charge);
        state.entries.push(entry);
    }

    pub(crate) fn stats(&self) -> ToolPromptCacheStats {
        let state = self.lock();
        ToolPromptCacheStats {
            entries: state.entries.len(),
            bytes: state.bytes,
            ..state.stats
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

struct PrefixCandidate {
    id: u64,
    prompt: Arc<str>,
    token_ids: Arc<[u32]>,
    boundaries: Arc<[SafeBoundary]>,
}

fn common_prefix_len(left: &str, right: &str) -> usize {
    let max = left.len().min(right.len());
    let mut length = 0;
    for ((offset, left_char), right_char) in left.char_indices().zip(right.chars()) {
        if offset >= max || left_char != right_char {
            break;
        }
        length = offset + left_char.len_utf8();
    }
    length
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert(cache: &ToolPromptCache, key: &[u8], prompt: &str, token_ids: &[u32]) {
        cache.insert(
            CacheInsertKey::new([7; 32], key.to_vec(), b"schema".to_vec()),
            prompt.to_owned(),
            token_ids.to_vec(),
            vec![SafeBoundary {
                byte_offset: 5,
                token_count: 1,
            }],
        );
    }

    #[test]
    fn exact_and_prefix_hits_are_distinct() {
        let cache = ToolPromptCache::new(4, 1024);
        insert(&cache, b"a", "turn>alpha", &[10, 11]);
        assert_eq!(
            cache.lookup_exact(&CacheLookupKey::new(&[7; 32], b"a", b"schema")),
            Some(vec![10, 11])
        );
        match cache
            .lookup_after_render(&CacheLookupKey::new(&[7; 32], b"b", b"schema"), "turn>beta")
        {
            CacheMatch::Prefix(hit) => {
                assert_eq!(hit.byte_offset, 5);
                assert_eq!(hit.token_ids, vec![10]);
            }
            _ => panic!("expected prefix hit"),
        }
        let stats = cache.stats();
        assert_eq!(stats.exact_hits, 1);
        assert_eq!(stats.prefix_hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn schema_change_cannot_reuse_a_prefix() {
        let cache = ToolPromptCache::new(4, 1024);
        insert(&cache, b"a", "turn>alpha", &[10, 11]);
        assert!(matches!(
            cache.lookup_after_render(&CacheLookupKey::new(&[7; 32], b"b", b"other"), "turn>beta"),
            CacheMatch::Miss
        ));
    }

    #[test]
    fn tokenizer_or_template_identity_change_cannot_reuse_an_entry() {
        let cache = ToolPromptCache::new(4, 1024);
        insert(&cache, b"a", "turn>alpha", &[10, 11]);
        assert!(cache
            .lookup_exact(&CacheLookupKey::new(&[8; 32], b"a", b"schema"))
            .is_none());
        assert!(matches!(
            cache.lookup_after_render(&CacheLookupKey::new(&[8; 32], b"b", b"schema"), "turn>beta"),
            CacheMatch::Miss
        ));
    }

    #[test]
    fn capacity_uses_lru_eviction() {
        let cache = ToolPromptCache::new(2, 1024);
        insert(&cache, b"a", "turn>alpha", &[10, 11]);
        insert(&cache, b"b", "turn>beta", &[10, 12]);
        let _ = cache.lookup_exact(&CacheLookupKey::new(&[7; 32], b"a", b"schema"));
        insert(&cache, b"c", "turn>gamma", &[10, 13]);
        assert!(cache
            .lookup_exact(&CacheLookupKey::new(&[7; 32], b"a", b"schema"))
            .is_some());
        assert!(cache
            .lookup_exact(&CacheLookupKey::new(&[7; 32], b"b", b"schema"))
            .is_none());
        assert!(cache
            .lookup_exact(&CacheLookupKey::new(&[7; 32], b"c", b"schema"))
            .is_some());
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn entry_larger_than_byte_budget_is_not_retained() {
        let cache = ToolPromptCache::new(4, 32);
        insert(&cache, b"a", "turn>alpha", &[10, 11]);
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn common_prefix_never_splits_utf8() {
        assert_eq!(common_prefix_len("工具甲", "工具乙"), "工具".len());
    }
}
