use std::collections::{HashMap, VecDeque};

use crate::array::Array;

use super::block_pool::BlockPool;

/// Number of tokens each block can hold.
pub const BLOCK_SIZE: usize = 256;

/// Unique identifier for a block in the store.
pub type BlockId = u64;

/// Storage strategy for a block's KV data.
pub enum BlockStorage {
    /// Block owns its KV data directly.
    Owned(Vec<(Array, Array)>),
    /// Block's KV data lives in a pre-allocated pool slot.
    Pooled { slot_idx: usize },
}

/// A single block of paged KV cache data.
///
/// Each block stores one `(keys, values)` pair per model layer and tracks how
/// many sequences currently reference it via `ref_count`.
pub struct Block {
    pub id: BlockId,
    pub ref_count: u32,
    pub storage: BlockStorage,
    /// Number of tokens currently stored in this block (up to [`BLOCK_SIZE`]).
    pub token_count: usize,
}

impl Block {
    /// Access KV data, resolving pooled storage if needed.
    pub fn kv_data<'a>(&'a self, pool: Option<&'a BlockPool>) -> &'a Vec<(Array, Array)> {
        match &self.storage {
            BlockStorage::Owned(data) => data,
            BlockStorage::Pooled { slot_idx } => pool
                .expect("pooled block requires pool reference")
                .get_slot(*slot_idx),
        }
    }
}

/// An LRU-managed store of paged KV cache blocks.
///
/// Blocks are reference-counted.  When a block's reference count drops to zero
/// it becomes eligible for eviction.  The LRU order tracks *usage* recency so
/// that the least-recently-used unreferenced block is evicted first.
///
/// Optionally backed by a `BlockPool` for pre-allocated slot reuse.
pub struct BlockStore {
    blocks: HashMap<BlockId, Block>,
    /// Front = most recently used, back = least recently used.
    lru_order: VecDeque<BlockId>,
    next_id: BlockId,
    /// Maximum bytes allowed (0 = unlimited).
    max_bytes: u64,
    /// Current estimated byte usage.
    current_bytes: u64,
    /// Optional pre-allocated block pool.
    pool: Option<BlockPool>,
}

impl BlockStore {
    /// Create an empty block store with no size limit and no pool.
    pub fn new() -> Self {
        Self::with_limit(0)
    }

    /// Create an empty block store with a byte limit (0 = unlimited) and no pool.
    pub fn with_limit(max_bytes: u64) -> Self {
        Self {
            blocks: HashMap::new(),
            lru_order: VecDeque::new(),
            next_id: 0,
            max_bytes,
            current_bytes: 0,
            pool: None,
        }
    }

    /// Create a block store with a byte limit and a pre-allocated pool.
    pub fn with_pool(max_bytes: u64, pool: BlockPool) -> Self {
        Self {
            blocks: HashMap::new(),
            lru_order: VecDeque::new(),
            next_id: 0,
            max_bytes,
            current_bytes: 0,
            pool: Some(pool),
        }
    }

    /// Allocate a new block with the given KV data and token count.
    ///
    /// If a pool is available and has free slots, the data is written into a
    /// pooled slot. Otherwise, the block owns its data directly.
    ///
    /// The block is created with `ref_count = 1` and placed at the front
    /// (most-recently-used position) of the LRU list.
    pub fn alloc_block(&mut self, kv_data: Vec<(Array, Array)>, token_count: usize) -> BlockId {
        let id = self.next_id;
        self.next_id += 1;

        let (storage, size) = if let Some(ref mut pool) = self.pool {
            if let Some(slot_idx) = pool.acquire() {
                let size = pool.slot_bytes;
                pool.write_slot(slot_idx, kv_data);
                (BlockStorage::Pooled { slot_idx }, size)
            } else {
                let size = estimate_block_bytes(&kv_data);
                (BlockStorage::Owned(kv_data), size)
            }
        } else {
            let size = estimate_block_bytes(&kv_data);
            (BlockStorage::Owned(kv_data), size)
        };

        let block = Block {
            id,
            ref_count: 1,
            storage,
            token_count,
        };

        self.blocks.insert(id, block);
        self.lru_order.push_front(id);
        self.current_bytes += size;

        // Auto-evict if over limit
        if self.max_bytes > 0 {
            self.evict_until_under_limit();
        }

        id
    }

    /// Return a reference to the block with the given id, if it exists.
    pub fn get_block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(&id)
    }

    /// Get a reference to the pool (for resolving pooled block data).
    pub fn pool(&self) -> Option<&BlockPool> {
        self.pool.as_ref()
    }

    /// Move the given block to the front (most-recently-used) of the LRU list.
    pub fn touch(&mut self, id: BlockId) {
        if let Some(pos) = self.lru_order.iter().position(|&bid| bid == id) {
            self.lru_order.remove(pos);
            self.lru_order.push_front(id);
        }
    }

    /// Increment the reference count of the given block.
    pub fn inc_ref(&mut self, id: BlockId) {
        if let Some(block) = self.blocks.get_mut(&id) {
            block.ref_count += 1;
        }
    }

    /// Decrement the reference count of the given block.
    ///
    /// If the count reaches zero the block is removed from the store and the
    /// LRU list. Pooled slots are released back to the pool.
    pub fn dec_ref(&mut self, id: BlockId) {
        let should_free = self
            .blocks
            .get_mut(&id)
            .map(|b| {
                b.ref_count = b.ref_count.saturating_sub(1);
                b.ref_count == 0
            })
            .unwrap_or(false);

        if should_free {
            if let Some(block) = self.blocks.remove(&id) {
                let size = self.block_size_bytes(&block);
                self.current_bytes = self.current_bytes.saturating_sub(size);
                self.release_storage(block.storage);
            }
            if let Some(pos) = self.lru_order.iter().position(|&bid| bid == id) {
                self.lru_order.remove(pos);
            }
        }
    }

    /// Copy-on-write: if the block has `ref_count > 1`, clone it into a new
    /// block with `ref_count = 1` (and decrement the original's ref count).
    /// If `ref_count == 1`, simply return the same id.
    pub fn cow_copy(&mut self, id: BlockId) -> BlockId {
        let rc = match self.blocks.get(&id) {
            Some(b) => b.ref_count,
            None => return id,
        };

        if rc <= 1 {
            return id;
        }

        // Clone the KV data from the original block.
        let block = &self.blocks[&id];
        let cloned_kv: Vec<(Array, Array)> = block
            .kv_data(self.pool.as_ref())
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let token_count = block.token_count;

        // Decrement the original's ref_count (won't free because rc > 1).
        self.blocks.get_mut(&id).unwrap().ref_count -= 1;

        // Allocate a fresh block with the cloned data.
        self.alloc_block(cloned_kv, token_count)
    }

    /// Evict the least-recently-used block whose `ref_count == 0`.
    ///
    /// Returns the removed block's id and owned KV data, or `None` if no
    /// evictable block exists. Pooled blocks have their data extracted from
    /// the pool before the slot is released.
    pub fn evict_lru(&mut self) -> Option<(BlockId, Vec<(Array, Array)>)> {
        // Scan from the back (least recently used) to find an evictable block.
        let pos = self
            .lru_order
            .iter()
            .rev()
            .position(|&bid| self.blocks.get(&bid).is_some_and(|b| b.ref_count <= 1));

        let pos = pos?;
        // `pos` is the offset from the back, convert to front-based index.
        let front_idx = self.lru_order.len() - 1 - pos;
        let bid = self.lru_order.remove(front_idx).unwrap();
        let block = self.blocks.remove(&bid).unwrap();
        let size = self.block_size_bytes(&block);
        self.current_bytes = self.current_bytes.saturating_sub(size);

        // Extract KV data and release pool slot if needed
        let kv_data = match block.storage {
            BlockStorage::Owned(data) => data,
            BlockStorage::Pooled { slot_idx } => {
                let data = self
                    .pool
                    .as_ref()
                    .expect("pooled block requires pool")
                    .get_slot(slot_idx)
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                if let Some(ref mut pool) = self.pool {
                    pool.release(slot_idx);
                }
                data
            }
        };

        Some((bid, kv_data))
    }

    /// Number of blocks currently in the store.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Returns true if the store contains no blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Total number of blocks ever allocated (equivalent to `next_id`).
    pub fn total_blocks(&self) -> usize {
        self.next_id as usize
    }

    /// Generate the next block ID without allocating a block.
    /// Used when hot cache is disabled but an ID is needed for SSD/prefix index.
    pub fn next_block_id(&mut self) -> BlockId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Evict LRU blocks until `current_bytes <= max_bytes`.
    fn evict_until_under_limit(&mut self) {
        while self.current_bytes > self.max_bytes {
            if self.evict_lru().is_none() {
                break; // no more evictable blocks
            }
        }
    }

    /// Get byte size of a block.
    fn block_size_bytes(&self, block: &Block) -> u64 {
        match &block.storage {
            BlockStorage::Owned(data) => estimate_block_bytes(data),
            BlockStorage::Pooled { .. } => self.pool.as_ref().map(|p| p.slot_bytes).unwrap_or(0),
        }
    }

    /// Release storage back to pool if pooled.
    fn release_storage(&mut self, storage: BlockStorage) {
        if let BlockStorage::Pooled { slot_idx } = storage
            && let Some(ref mut pool) = self.pool
        {
            pool.release(slot_idx);
        }
    }
}

/// Estimate the byte size of a block's KV data.
fn estimate_block_bytes(kv_data: &[(Array, Array)]) -> u64 {
    kv_data
        .iter()
        .map(|(k, v)| (k.nbytes() + v.nbytes()) as u64)
        .sum()
}

impl Default for BlockStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create dummy KV data (empty arrays) for `n_layers` layers.
    fn dummy_kv(n_layers: usize) -> Vec<(Array, Array)> {
        (0..n_layers)
            .map(|_| (Array::from_float(0.0), Array::from_float(0.0)))
            .collect()
    }

    #[test]
    fn alloc_and_get() {
        let mut store = BlockStore::new();
        let id = store.alloc_block(dummy_kv(2), 64);
        assert_eq!(store.len(), 1);
        let pool = store.pool();
        let block = store.get_block(id).unwrap();
        assert_eq!(block.token_count, 64);
        assert_eq!(block.ref_count, 1);
        assert_eq!(block.kv_data(pool).len(), 2);
    }

    #[test]
    fn inc_dec_ref() {
        let mut store = BlockStore::new();
        let id = store.alloc_block(dummy_kv(1), 10);
        store.inc_ref(id);
        assert_eq!(store.get_block(id).unwrap().ref_count, 2);
        store.dec_ref(id);
        assert_eq!(store.get_block(id).unwrap().ref_count, 1);
        store.dec_ref(id);
        // Block should be freed now.
        assert!(store.get_block(id).is_none());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn cow_copy_shared() {
        let mut store = BlockStore::new();
        let id = store.alloc_block(dummy_kv(1), 100);
        store.inc_ref(id);
        assert_eq!(store.get_block(id).unwrap().ref_count, 2);

        let new_id = store.cow_copy(id);
        assert_ne!(new_id, id);
        assert_eq!(store.get_block(id).unwrap().ref_count, 1);
        assert_eq!(store.get_block(new_id).unwrap().ref_count, 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn cow_copy_unique() {
        let mut store = BlockStore::new();
        let id = store.alloc_block(dummy_kv(1), 50);
        let same = store.cow_copy(id);
        assert_eq!(same, id);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn evict_lru_order() {
        let mut store = BlockStore::new();
        let a = store.alloc_block(dummy_kv(1), 10);
        let b = store.alloc_block(dummy_kv(1), 20);

        // Both have ref_count=1, set to 0 so they are evictable.
        store.blocks.get_mut(&a).unwrap().ref_count = 0;
        store.blocks.get_mut(&b).unwrap().ref_count = 0;

        // `a` was allocated first and not touched since, so it is the LRU.
        let evicted = store.evict_lru().unwrap();
        assert_eq!(evicted.0, a);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn touch_updates_lru() {
        let mut store = BlockStore::new();
        let a = store.alloc_block(dummy_kv(1), 10);
        let b = store.alloc_block(dummy_kv(1), 20);

        store.blocks.get_mut(&a).unwrap().ref_count = 0;
        store.blocks.get_mut(&b).unwrap().ref_count = 0;

        // Touch `a` so it becomes most-recently-used.
        store.touch(a);

        // Now `b` should be evicted first.
        let evicted = store.evict_lru().unwrap();
        assert_eq!(evicted.0, b);
    }

    #[test]
    fn evict_skips_referenced() {
        let mut store = BlockStore::new();
        let _a = store.alloc_block(dummy_kv(1), 10);
        let b = store.alloc_block(dummy_kv(1), 20);

        // Only `b` is evictable (ref_count == 0).
        store.blocks.get_mut(&b).unwrap().ref_count = 0;

        let evicted = store.evict_lru().unwrap();
        assert_eq!(evicted.0, b);

        // `a` still has ref_count == 1, so nothing to evict.
        assert!(store.evict_lru().is_none());
    }

    #[test]
    fn total_blocks_monotonic() {
        let mut store = BlockStore::new();
        store.alloc_block(dummy_kv(1), 10);
        store.alloc_block(dummy_kv(1), 20);
        assert_eq!(store.total_blocks(), 2);

        // Even after freeing, total_blocks keeps counting.
        let id = store.alloc_block(dummy_kv(1), 30);
        store.dec_ref(id);
        assert_eq!(store.total_blocks(), 3);
        assert_eq!(store.len(), 2);
    }
}
