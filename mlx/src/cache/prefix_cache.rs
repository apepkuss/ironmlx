use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::block_store::{BLOCK_SIZE, BlockId, BlockStore};
use super::ssd_store::SSDStore;
use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::stream::Stream;

/// Hash of a block in the prefix chain.
/// Each block's hash = hash(prev_block_hash, block_tokens).
pub type BlockHash = u64;

/// PrefixCache enables cross-request KV cache reuse by matching token prefixes.
///
/// Token sequences are split into blocks of `BLOCK_SIZE` tokens. Each block is
/// identified by a chain hash that depends on both the block's tokens and all
/// preceding blocks. This ensures that identical token prefixes map to the same
/// cached KV data.
pub struct PrefixCache {
    /// Maps chain hash → BlockId for prefix lookup.
    hash_to_block: HashMap<BlockHash, BlockId>,
    /// Maps BlockId → chain hash for reverse lookup (cleanup).
    block_to_hash: HashMap<BlockId, BlockHash>,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self {
            hash_to_block: HashMap::new(),
            block_to_hash: HashMap::new(),
        }
    }

    /// Compute chain hashes for a token sequence split into blocks.
    ///
    /// Returns a vector of `(block_hash, token_slice_start, token_slice_end)` for
    /// each complete block. Partial trailing blocks are not included.
    pub fn compute_block_hashes(tokens: &[i32]) -> Vec<(BlockHash, usize, usize)> {
        let mut result = Vec::new();
        let mut prev_hash: BlockHash = 0;

        let num_complete_blocks = tokens.len() / BLOCK_SIZE;
        for i in 0..num_complete_blocks {
            let start = i * BLOCK_SIZE;
            let end = start + BLOCK_SIZE;
            let block_tokens = &tokens[start..end];

            let hash = chain_hash(prev_hash, block_tokens);
            result.push((hash, start, end));
            prev_hash = hash;
        }

        result
    }

    /// Look up the longest cached prefix for a token sequence.
    ///
    /// Returns `(matched_block_ids, num_matched_tokens)`. The caller can load
    /// the KV data for these blocks from BlockStore/SSDStore and skip prefill
    /// for the first `num_matched_tokens` tokens.
    pub fn lookup_prefix(&self, tokens: &[i32]) -> (Vec<BlockId>, usize) {
        let block_hashes = Self::compute_block_hashes(tokens);
        let mut matched_blocks = Vec::new();

        for (hash, _start, _end) in &block_hashes {
            if let Some(&block_id) = self.hash_to_block.get(hash) {
                matched_blocks.push(block_id);
            } else {
                break; // prefix chain broken
            }
        }

        let num_matched_tokens = matched_blocks.len() * BLOCK_SIZE;
        (matched_blocks, num_matched_tokens)
    }

    /// Register blocks in the prefix index after a new prefill.
    ///
    /// `tokens` is the full prompt token sequence. `block_ids` are the BlockStore
    /// ids for the KV data of each complete block, in order.
    pub fn insert_blocks(&mut self, tokens: &[i32], block_ids: &[BlockId]) {
        let block_hashes = Self::compute_block_hashes(tokens);

        for (i, (hash, _start, _end)) in block_hashes.iter().enumerate() {
            if i >= block_ids.len() {
                break;
            }
            let block_id = block_ids[i];
            self.hash_to_block.insert(*hash, block_id);
            self.block_to_hash.insert(block_id, *hash);
        }
    }

    /// Remove a block from the prefix index.
    pub fn remove_block(&mut self, block_id: BlockId) {
        if let Some(hash) = self.block_to_hash.remove(&block_id) {
            self.hash_to_block.remove(&hash);
        }
    }

    /// Number of blocks in the prefix index.
    pub fn len(&self) -> usize {
        self.hash_to_block.len()
    }

    /// Returns true if the prefix index is empty.
    pub fn is_empty(&self) -> bool {
        self.hash_to_block.is_empty()
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a chain hash: hash(prev_hash, block_tokens).
fn chain_hash(prev_hash: BlockHash, block_tokens: &[i32]) -> BlockHash {
    let mut hasher = DefaultHasher::new();
    prev_hash.hash(&mut hasher);
    block_tokens.hash(&mut hasher);
    hasher.finish()
}

/// High-level cache manager that coordinates BlockStore, SSDStore, and PrefixCache.
pub struct CacheManager {
    pub block_store: BlockStore,
    pub ssd_store: SSDStore,
    pub prefix_cache: PrefixCache,
    num_layers: usize,
}

impl CacheManager {
    /// Create a new cache manager.
    pub fn new(ssd_store: SSDStore, num_layers: usize) -> Self {
        Self {
            block_store: BlockStore::new(),
            ssd_store,
            prefix_cache: PrefixCache::new(),
            num_layers,
        }
    }

    /// Look up prefix cache and load matched KV blocks.
    ///
    /// Returns `(kv_cache_per_layer, num_matched_tokens)` where `kv_cache_per_layer`
    /// contains the concatenated KV data from matched blocks for each layer.
    /// If no prefix matches, returns empty vecs and 0.
    #[allow(clippy::type_complexity)]
    pub fn lookup_and_load(
        &mut self,
        tokens: &[i32],
    ) -> Result<(Vec<(Option<Array>, Option<Array>)>, usize)> {
        let (matched_block_ids, _) = self.prefix_cache.lookup_prefix(tokens);

        if matched_block_ids.is_empty() {
            let empty_cache: Vec<(Option<Array>, Option<Array>)> =
                (0..self.num_layers).map(|_| (None, None)).collect();
            return Ok((empty_cache, 0));
        }

        let stream = Stream::default_stream(&Device::cpu());

        // Load all matched blocks (from memory or SSD)
        let mut all_blocks_kv: Vec<Vec<(Array, Array)>> = Vec::new();
        for &block_id in &matched_block_ids {
            // Try memory first
            let in_memory = self.block_store.get_block(block_id).map(|block| {
                block
                    .kv_data
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>()
            });
            if let Some(kv_data) = in_memory {
                self.block_store.touch(block_id);
                all_blocks_kv.push(kv_data);
            } else if self.ssd_store.has_block(block_id) {
                // Load from SSD
                let kv_data = self.ssd_store.load_block(block_id, &stream)?;
                all_blocks_kv.push(kv_data);
            } else {
                // Block missing, prefix chain broken
                break;
            }
        }

        if all_blocks_kv.is_empty() {
            let empty_cache: Vec<(Option<Array>, Option<Array>)> =
                (0..self.num_layers).map(|_| (None, None)).collect();
            return Ok((empty_cache, 0));
        }

        let actual_matched = all_blocks_kv.len();
        let actual_tokens = actual_matched * BLOCK_SIZE;

        // Concatenate blocks per layer
        let mut cache: Vec<(Option<Array>, Option<Array>)> = Vec::with_capacity(self.num_layers);
        let gpu_stream = Stream::new(&Device::gpu());

        for layer_idx in 0..self.num_layers {
            let k_arrays: Vec<&Array> = all_blocks_kv
                .iter()
                .map(|block_kv| &block_kv[layer_idx].0)
                .collect();
            let v_arrays: Vec<&Array> = all_blocks_kv
                .iter()
                .map(|block_kv| &block_kv[layer_idx].1)
                .collect();

            let k_vec = crate::vector::VectorArray::from_arrays(&k_arrays);
            let v_vec = crate::vector::VectorArray::from_arrays(&v_arrays);

            let concat_k = crate::ops::concatenate(&k_vec, 2, &gpu_stream)?;
            let concat_v = crate::ops::concatenate(&v_vec, 2, &gpu_stream)?;

            cache.push((Some(concat_k), Some(concat_v)));
        }

        Ok((cache, actual_tokens))
    }

    /// Store KV cache blocks after a prefill and register them in the prefix index.
    ///
    /// `tokens` is the full prompt. `kv_cache` is the per-layer KV cache after
    /// forward pass. Only complete blocks (multiples of BLOCK_SIZE) are stored.
    pub fn store_after_prefill(
        &mut self,
        tokens: &[i32],
        kv_cache: &[(Option<Array>, Option<Array>)],
    ) -> Result<()> {
        let num_complete_blocks = tokens.len() / BLOCK_SIZE;
        if num_complete_blocks == 0 {
            return Ok(());
        }

        let stream = Stream::new(&Device::gpu());
        let mut block_ids = Vec::with_capacity(num_complete_blocks);

        for block_idx in 0..num_complete_blocks {
            let start_token = (block_idx * BLOCK_SIZE) as i32;
            let end_token = start_token + BLOCK_SIZE as i32;

            // Slice each layer's KV to extract this block's range
            let mut block_kv: Vec<(Array, Array)> = Vec::with_capacity(self.num_layers);
            for (keys_opt, values_opt) in kv_cache {
                if let (Some(keys), Some(values)) = (keys_opt, values_opt) {
                    let shape_k = keys.shape();
                    let shape_v = values.shape();

                    let k_block = crate::ops::slice(
                        keys,
                        &[0, 0, start_token, 0],
                        &[shape_k[0], shape_k[1], end_token, shape_k[3]],
                        &[1, 1, 1, 1],
                        &stream,
                    )?;
                    let v_block = crate::ops::slice(
                        values,
                        &[0, 0, start_token, 0],
                        &[shape_v[0], shape_v[1], end_token, shape_v[3]],
                        &[1, 1, 1, 1],
                        &stream,
                    )?;
                    block_kv.push((k_block, v_block));
                }
            }

            if block_kv.len() != self.num_layers {
                continue; // skip incomplete layers
            }

            // Allocate in BlockStore
            let block_id = self.block_store.alloc_block(block_kv.clone(), BLOCK_SIZE);

            // Persist to SSD (sync for now, async in future)
            let _ = self.ssd_store.store_block(block_id, &block_kv, &stream);

            block_ids.push(block_id);
        }

        // Register in prefix index
        self.prefix_cache.insert_blocks(tokens, &block_ids);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_hash_deterministic() {
        let tokens: Vec<i32> = (0..256).collect();
        let h1 = chain_hash(0, &tokens);
        let h2 = chain_hash(0, &tokens);
        assert_eq!(h1, h2);
    }

    #[test]
    fn chain_hash_differs_by_prev() {
        let tokens: Vec<i32> = (0..256).collect();
        let h1 = chain_hash(0, &tokens);
        let h2 = chain_hash(1, &tokens);
        assert_ne!(h1, h2);
    }

    #[test]
    fn compute_block_hashes_partial_ignored() {
        // 300 tokens = 1 complete block + 44 remainder
        let tokens: Vec<i32> = (0..300).collect();
        let hashes = PrefixCache::compute_block_hashes(&tokens);
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes[0].1, 0);
        assert_eq!(hashes[0].2, 256);
    }

    #[test]
    fn compute_block_hashes_two_blocks() {
        let tokens: Vec<i32> = (0..512).collect();
        let hashes = PrefixCache::compute_block_hashes(&tokens);
        assert_eq!(hashes.len(), 2);
        // Second hash depends on first
        assert_ne!(hashes[0].0, hashes[1].0);
    }

    #[test]
    fn lookup_insert_roundtrip() {
        let mut pc = PrefixCache::new();
        let tokens: Vec<i32> = (0..512).collect();

        // Initially no match
        let (blocks, matched) = pc.lookup_prefix(&tokens);
        assert!(blocks.is_empty());
        assert_eq!(matched, 0);

        // Insert blocks
        pc.insert_blocks(&tokens, &[100, 200]);

        // Now should match
        let (blocks, matched) = pc.lookup_prefix(&tokens);
        assert_eq!(blocks, vec![100, 200]);
        assert_eq!(matched, 512);
    }

    #[test]
    fn lookup_partial_prefix() {
        let mut pc = PrefixCache::new();

        // Insert prefix for first 256 tokens
        let prefix: Vec<i32> = (0..256).collect();
        pc.insert_blocks(&prefix, &[42]);

        // Query with 512 tokens that share the same first 256
        let mut longer: Vec<i32> = (0..256).collect();
        longer.extend(1000..1256);

        let (blocks, matched) = pc.lookup_prefix(&longer);
        assert_eq!(blocks, vec![42]);
        assert_eq!(matched, 256);
    }

    #[test]
    fn remove_block_cleans_index() {
        let mut pc = PrefixCache::new();
        let tokens: Vec<i32> = (0..256).collect();
        pc.insert_blocks(&tokens, &[10]);
        assert_eq!(pc.len(), 1);

        pc.remove_block(10);
        assert_eq!(pc.len(), 0);

        let (blocks, _) = pc.lookup_prefix(&tokens);
        assert!(blocks.is_empty());
    }
}
