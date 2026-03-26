use crate::array::Array;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::stream::Stream;

/// Configuration for creating a BlockPool.
pub struct BlockPoolConfig {
    /// Number of pre-allocated slots.
    pub num_slots: usize,
    /// Number of model layers.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub n_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Data type for KV arrays.
    pub dtype: Dtype,
    /// Tokens per block.
    pub block_size: usize,
}

/// Pre-allocated pool of KV cache block slots.
///
/// Each slot holds `num_layers` pairs of (keys, values) arrays with shape
/// `[1, n_kv_heads, block_size, head_dim]`. Slots are reused via a free list
/// to avoid repeated GPU memory allocation.
pub struct BlockPool {
    /// All pre-allocated KV data: `slots[slot_idx][layer_idx] = (keys, values)`.
    slots: Vec<Vec<(Array, Array)>>,
    /// Stack of free slot indices.
    free_slots: Vec<usize>,
    /// Per-slot byte size (for accounting).
    pub slot_bytes: u64,
}

impl BlockPool {
    /// Create a new pool with `config.num_slots` pre-allocated slots.
    pub fn new(config: &BlockPoolConfig) -> Self {
        let stream = Stream::new(&Device::gpu());
        let shape = &[
            1,
            config.n_kv_heads as i32,
            config.block_size as i32,
            config.head_dim as i32,
        ];
        let mut slots = Vec::with_capacity(config.num_slots);
        let mut free_slots = Vec::with_capacity(config.num_slots);

        for i in 0..config.num_slots {
            let mut layer_kv = Vec::with_capacity(config.num_layers);
            for _ in 0..config.num_layers {
                let k = crate::ops::zeros(shape, config.dtype, &stream)
                    .expect("failed to allocate pool slot");
                let v = crate::ops::zeros(shape, config.dtype, &stream)
                    .expect("failed to allocate pool slot");
                layer_kv.push((k, v));
            }
            slots.push(layer_kv);
            free_slots.push(i);
        }

        let slot_bytes = config.num_layers as u64
            * 2
            * config.n_kv_heads as u64
            * config.block_size as u64
            * config.head_dim as u64
            * config.dtype.size_of() as u64;

        Self {
            slots,
            free_slots,
            slot_bytes,
        }
    }

    /// Acquire a free slot. Returns `None` if pool is exhausted.
    pub fn acquire(&mut self) -> Option<usize> {
        self.free_slots.pop()
    }

    /// Release a slot back to the pool.
    pub fn release(&mut self, slot_idx: usize) {
        debug_assert!(slot_idx < self.slots.len());
        self.free_slots.push(slot_idx);
    }

    /// Get immutable reference to a slot's KV data.
    pub fn get_slot(&self, slot_idx: usize) -> &Vec<(Array, Array)> {
        &self.slots[slot_idx]
    }

    /// Get mutable reference to a slot's KV data.
    pub fn get_slot_mut(&mut self, slot_idx: usize) -> &mut Vec<(Array, Array)> {
        &mut self.slots[slot_idx]
    }

    /// Write KV data into a slot (overwrites existing data).
    pub fn write_slot(&mut self, slot_idx: usize, kv_data: Vec<(Array, Array)>) {
        self.slots[slot_idx] = kv_data;
    }

    /// Number of free slots available.
    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Total number of slots in the pool.
    pub fn total_slots(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_acquire_release() {
        let config = BlockPoolConfig {
            num_slots: 4,
            num_layers: 2,
            n_kv_heads: 8,
            head_dim: 64,
            dtype: Dtype::Float16,
            block_size: 256,
        };
        let mut pool = BlockPool::new(&config);
        assert_eq!(pool.total_slots(), 4);
        assert_eq!(pool.free_count(), 4);

        let s0 = pool.acquire().unwrap();
        let _s1 = pool.acquire().unwrap();
        assert_eq!(pool.free_count(), 2);

        pool.release(s0);
        assert_eq!(pool.free_count(), 3);

        // Acquire all remaining
        pool.acquire().unwrap();
        pool.acquire().unwrap();
        pool.acquire().unwrap();
        assert!(pool.acquire().is_none());
    }

    #[test]
    fn pool_slot_data_shape() {
        let config = BlockPoolConfig {
            num_slots: 1,
            num_layers: 3,
            n_kv_heads: 4,
            head_dim: 32,
            dtype: Dtype::Float16,
            block_size: 256,
        };
        let pool = BlockPool::new(&config);
        let slot = pool.get_slot(0);
        assert_eq!(slot.len(), 3); // num_layers
        for (k, v) in slot {
            assert_eq!(k.shape(), &[1, 4, 256, 32]);
            assert_eq!(v.shape(), &[1, 4, 256, 32]);
        }
    }

    #[test]
    fn pool_slot_bytes_calculation() {
        let config = BlockPoolConfig {
            num_slots: 1,
            num_layers: 32,
            n_kv_heads: 8,
            head_dim: 128,
            dtype: Dtype::Float16,
            block_size: 256,
        };
        let pool = BlockPool::new(&config);
        // 32 layers * 2 (k+v) * 8 heads * 256 tokens * 128 dim * 2 bytes = 33,554,432
        assert_eq!(pool.slot_bytes, 32 * 2 * 8 * 256 * 128 * 2);
    }
}
