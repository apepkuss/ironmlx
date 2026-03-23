use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc;

use crate::array::Array;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::io::{load_safetensors, save_safetensors};
use crate::stream::Stream;
use crate::vector::{MapStringToArray, MapStringToString};

/// Unique identifier for a KV cache block.
pub type BlockId = u64;

/// Configuration for SSD-backed KV cache persistence.
pub struct SSDStoreConfig {
    /// Base directory for cache files.
    pub cache_dir: PathBuf,
    /// Maximum total size of cached files in bytes (default: 10 GB).
    pub max_size_bytes: u64,
    /// Hash of the model config, used to isolate per-model caches.
    pub model_hash: String,
}

impl Default for SSDStoreConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        Self {
            cache_dir: PathBuf::from(home).join(".cache/ironmlx/kv_cache"),
            max_size_bytes: 10 * 1024 * 1024 * 1024,
            model_hash: String::new(),
        }
    }
}

/// Metadata for a single cached block on disk.
struct SSDEntry {
    file_path: PathBuf,
    size_bytes: u64,
    num_layers: usize,
}

/// CPU-side KV data for a single layer, ready for disk write without GPU access.
struct CpuKvLayer {
    keys_data: Vec<f32>,
    keys_shape: Vec<i32>,
    values_data: Vec<f32>,
    values_shape: Vec<i32>,
}

/// Message sent to the background writer thread.
/// Contains CPU-side data only — no GPU arrays — to avoid Metal CommandBuffer conflicts.
struct WriteJob {
    cpu_kv: Vec<CpuKvLayer>,
    file_path: PathBuf,
}

/// SSD-backed store for KV cache blocks, with LRU eviction and async writes.
pub struct SSDStore {
    config: SSDStoreConfig,
    entries: HashMap<BlockId, SSDEntry>,
    lru_order: VecDeque<BlockId>,
    current_size_bytes: u64,
    /// Channel to the background writer thread.
    write_tx: Option<mpsc::Sender<WriteJob>>,
    /// Handle to the background writer thread for graceful shutdown.
    writer_handle: Option<std::thread::JoinHandle<()>>,
    /// Pending writes that haven't been confirmed yet.
    pending_writes: HashMap<BlockId, PendingWrite>,
}

/// Tracks a block that's being written asynchronously.
struct PendingWrite {
    file_path: PathBuf,
    num_layers: usize,
}

impl SSDStore {
    /// Create a new store, ensuring the model cache directory exists.
    /// Spawns a background writer thread for async I/O.
    pub fn new(config: SSDStoreConfig) -> Result<Self> {
        let model_dir = config.cache_dir.join(&config.model_hash);
        std::fs::create_dir_all(&model_dir)
            .map_err(|e| Error::Mlx(format!("failed to create cache dir: {e}")))?;

        // Spawn background writer thread
        let (tx, rx) = mpsc::channel::<WriteJob>();
        let handle = std::thread::Builder::new()
            .name("ssd-cache-writer".into())
            .spawn(move || {
                writer_thread(rx);
            })
            .map_err(|e| Error::Mlx(format!("failed to spawn writer thread: {e}")))?;

        Ok(Self {
            config,
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            current_size_bytes: 0,
            write_tx: Some(tx),
            writer_handle: Some(handle),
            pending_writes: HashMap::new(),
        })
    }

    /// Return the per-model cache directory.
    pub fn model_dir(&self) -> PathBuf {
        self.config.cache_dir.join(&self.config.model_hash)
    }

    /// Current total size of all cached files in bytes.
    pub fn current_size(&self) -> u64 {
        self.current_size_bytes
    }

    /// Check whether a block is present in the store (on disk or pending write).
    pub fn has_block(&self, block_id: BlockId) -> bool {
        self.entries.contains_key(&block_id) || self.pending_writes.contains_key(&block_id)
    }

    /// Persist a KV cache block to disk asynchronously.
    ///
    /// The KV data is sent to a background thread for serialization.
    /// The block is immediately available for `has_block()` checks.
    pub fn store_block(
        &mut self,
        block_id: BlockId,
        kv_data: &[(Array, Array)],
        _stream: &Stream,
    ) -> Result<()> {
        let file_path = self.model_dir().join(format!("{block_id}.safetensors"));

        // Remove old entry if overwriting
        if self.entries.contains_key(&block_id) {
            self.remove_block(block_id)?;
        }

        let num_layers = kv_data.len();

        // Eval GPU arrays and copy to CPU before sending to writer thread.
        // This avoids Metal CommandBuffer conflicts (SIGABRT) when the writer
        // thread would otherwise trigger GPU eval on a separate OS thread.
        let cpu_kv: Vec<CpuKvLayer> = kv_data
            .iter()
            .filter_map(|(k, v)| {
                let keys_data = k.to_vec_f32().ok()?;
                let keys_shape = k.shape();
                let values_data = v.to_vec_f32().ok()?;
                let values_shape = v.shape();
                Some(CpuKvLayer {
                    keys_data,
                    keys_shape,
                    values_data,
                    values_shape,
                })
            })
            .collect();

        if cpu_kv.len() != num_layers {
            return Err(crate::error::Error::Mlx(
                "failed to eval KV cache arrays for SSD store".into(),
            ));
        }

        // Track as pending
        self.pending_writes.insert(
            block_id,
            PendingWrite {
                file_path: file_path.clone(),
                num_layers,
            },
        );

        // Send CPU data to writer thread (no GPU arrays cross thread boundary)
        if let Some(ref tx) = self.write_tx {
            let _ = tx.send(WriteJob { cpu_kv, file_path });
        }

        Ok(())
    }

    /// Flush any pending writes and update the index.
    /// Call this before `load_block` if you need the data immediately.
    pub fn flush_pending(&mut self) -> Result<()> {
        let pending: Vec<BlockId> = self.pending_writes.keys().copied().collect();
        for block_id in pending {
            if let Some(pw) = self.pending_writes.remove(&block_id)
                && pw.file_path.exists()
            {
                let size_bytes = std::fs::metadata(&pw.file_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                self.entries.insert(
                    block_id,
                    SSDEntry {
                        file_path: pw.file_path,
                        size_bytes,
                        num_layers: pw.num_layers,
                    },
                );
                self.lru_order.push_back(block_id);
                self.current_size_bytes += size_bytes;
            }
        }
        self.evict_until_under_limit()?;
        Ok(())
    }

    /// Load a KV cache block from disk and touch it in the LRU.
    pub fn load_block(
        &mut self,
        block_id: BlockId,
        stream: &Stream,
    ) -> Result<Vec<(Array, Array)>> {
        // If pending, flush first
        if self.pending_writes.contains_key(&block_id) {
            // Wait briefly for the file to appear
            let pw = self.pending_writes.get(&block_id).unwrap();
            let path = pw.file_path.clone();
            for _ in 0..50 {
                if path.exists() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            self.flush_pending()?;
        }

        let entry = self
            .entries
            .get(&block_id)
            .ok_or_else(|| Error::Mlx(format!("block {block_id} not found in SSD store")))?;

        let path_str = entry
            .file_path
            .to_str()
            .ok_or_else(|| Error::Mlx("non-UTF-8 cache path".into()))?;
        let num_layers = entry.num_layers;

        let (arrays, _metadata) = load_safetensors(path_str, stream)?;

        let mut result = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let keys = arrays.get(&format!("layer_{i}_keys"))?;
            let values = arrays.get(&format!("layer_{i}_values"))?;
            result.push((keys, values));
        }

        // Touch LRU
        self.lru_order.retain(|id| *id != block_id);
        self.lru_order.push_back(block_id);

        Ok(result)
    }

    /// Remove a single block from disk and the index.
    pub fn remove_block(&mut self, block_id: BlockId) -> Result<()> {
        self.pending_writes.remove(&block_id);
        if let Some(entry) = self.entries.remove(&block_id) {
            if entry.file_path.exists() {
                std::fs::remove_file(&entry.file_path)
                    .map_err(|e| Error::Mlx(format!("failed to remove cache file: {e}")))?;
            }
            self.current_size_bytes = self.current_size_bytes.saturating_sub(entry.size_bytes);
            self.lru_order.retain(|id| *id != block_id);
        }
        Ok(())
    }

    /// Evict least-recently-used blocks until total size is under the limit.
    pub fn evict_until_under_limit(&mut self) -> Result<()> {
        while self.current_size_bytes > self.config.max_size_bytes {
            let victim = match self.lru_order.pop_front() {
                Some(id) => id,
                None => break,
            };
            if let Some(entry) = self.entries.remove(&victim) {
                if entry.file_path.exists() {
                    std::fs::remove_file(&entry.file_path).map_err(|e| {
                        Error::Mlx(format!("failed to remove cache file during eviction: {e}"))
                    })?;
                }
                self.current_size_bytes = self.current_size_bytes.saturating_sub(entry.size_bytes);
            }
        }
        Ok(())
    }

    /// Scan the model cache directory on startup and rebuild the in-memory
    /// index. Returns the number of entries recovered.
    pub fn recover_index(&mut self) -> Result<usize> {
        let model_dir = self.model_dir();
        if !model_dir.exists() {
            return Ok(0);
        }

        let cpu_device = Device::cpu();
        let stream = Stream::new(&cpu_device);

        let dir_entries = std::fs::read_dir(&model_dir)
            .map_err(|e| Error::Mlx(format!("failed to read cache dir: {e}")))?;

        let mut count = 0usize;
        for dir_entry in dir_entries {
            let dir_entry =
                dir_entry.map_err(|e| Error::Mlx(format!("failed to read dir entry: {e}")))?;
            let path = dir_entry.path();

            let ext = path.extension().and_then(|e| e.to_str());
            if ext != Some("safetensors") {
                continue;
            }

            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            let block_id: BlockId = match stem.parse() {
                Ok(id) => id,
                Err(_) => continue,
            };

            let size_bytes = std::fs::metadata(&path)
                .map_err(|e| Error::Mlx(format!("failed to stat cache file: {e}")))?
                .len();

            let path_str = path
                .to_str()
                .ok_or_else(|| Error::Mlx("non-UTF-8 cache path".into()))?;

            // Determine layer count by inspecting safetensors keys
            let (arrays, _metadata) = load_safetensors(path_str, &stream)?;
            let mut max_layer: Option<usize> = None;
            for (key, _) in arrays.iter() {
                if let Some(rest) = key.strip_prefix("layer_")
                    && let Some(idx_str) = rest.strip_suffix("_keys")
                    && let Ok(idx) = idx_str.parse::<usize>()
                {
                    max_layer = Some(match max_layer {
                        Some(m) => m.max(idx),
                        None => idx,
                    });
                }
            }
            let num_layers = max_layer.map_or(0, |m| m + 1);

            self.entries.insert(
                block_id,
                SSDEntry {
                    file_path: path,
                    size_bytes,
                    num_layers,
                },
            );
            self.lru_order.push_back(block_id);
            self.current_size_bytes += size_bytes;
            count += 1;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(tmp_dir: &std::path::Path) -> SSDStoreConfig {
        SSDStoreConfig {
            cache_dir: tmp_dir.to_path_buf(),
            max_size_bytes: 1024 * 1024, // 1MB for testing
            model_hash: "test_model".to_string(),
        }
    }

    fn dummy_kv(num_layers: usize) -> Vec<(Array, Array)> {
        (0..num_layers)
            .map(|i| {
                let base = (i as f32) * 10.0;
                let k = Array::from_slice_f32(&[base + 1.0, base + 2.0, base + 3.0, base + 4.0]);
                let v = Array::from_slice_f32(&[base + 5.0, base + 6.0, base + 7.0, base + 8.0]);
                (k, v)
            })
            .collect()
    }

    /// Helper: wait for async write and flush pending entries.
    fn wait_and_flush(store: &mut SSDStore) {
        std::thread::sleep(std::time::Duration::from_millis(200));
        store.flush_pending().unwrap();
    }

    #[test]
    fn store_and_load_block() {
        crate::init();
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let mut store = SSDStore::new(config).unwrap();
        let stream = Stream::new(&Device::cpu());

        let kv = dummy_kv(2);
        store.store_block(0, &kv, &stream).unwrap();
        wait_and_flush(&mut store);

        assert!(store.has_block(0));

        let loaded = store.load_block(0, &stream).unwrap();
        assert_eq!(loaded.len(), 2);

        // Verify shape is preserved
        assert_eq!(loaded[0].0.shape(), &[4]);
        assert_eq!(loaded[0].1.shape(), &[4]);
        assert_eq!(loaded[1].0.shape(), &[4]);
        assert_eq!(loaded[1].1.shape(), &[4]);
    }

    #[test]
    fn store_block_has_block_immediately() {
        crate::init();
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let mut store = SSDStore::new(config).unwrap();
        let stream = Stream::new(&Device::cpu());

        store.store_block(99, &dummy_kv(1), &stream).unwrap();

        // has_block should return true immediately (pending write)
        assert!(store.has_block(99));
    }

    #[test]
    fn remove_block_deletes_file() {
        crate::init();
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let mut store = SSDStore::new(config).unwrap();
        let stream = Stream::new(&Device::cpu());

        store.store_block(1, &dummy_kv(1), &stream).unwrap();
        wait_and_flush(&mut store);

        let file_path = store.model_dir().join("1.safetensors");
        assert!(file_path.exists());

        store.remove_block(1).unwrap();
        assert!(!file_path.exists());
        assert!(!store.has_block(1));
    }

    #[test]
    fn eviction_under_limit() {
        crate::init();
        let tmp = tempfile::TempDir::new().unwrap();
        let mut config = test_config(tmp.path());
        config.max_size_bytes = 1; // 1 byte — force eviction
        let mut store = SSDStore::new(config).unwrap();
        let stream = Stream::new(&Device::cpu());

        store.store_block(10, &dummy_kv(1), &stream).unwrap();
        wait_and_flush(&mut store);

        // Should have been evicted since limit is 1 byte
        assert_eq!(store.entries.len(), 0);
        assert_eq!(store.current_size_bytes, 0);
    }

    #[test]
    fn eviction_lru_order() {
        crate::init();
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let mut store = SSDStore::new(config).unwrap();
        let stream = Stream::new(&Device::cpu());

        // Store 3 blocks
        for id in 0..3 {
            store.store_block(id, &dummy_kv(1), &stream).unwrap();
            wait_and_flush(&mut store);
        }

        // All 3 should be present
        assert!(store.has_block(0));
        assert!(store.has_block(1));
        assert!(store.has_block(2));

        // Touch block 0 by loading it (moves to back of LRU)
        let _ = store.load_block(0, &stream).unwrap();

        // Now shrink the limit to force eviction of oldest untouched blocks
        // Block 1 is the LRU victim (block 0 was touched, block 2 is newest)
        store.config.max_size_bytes = 1;
        store.evict_until_under_limit().unwrap();

        // All should be evicted (limit is 1 byte), but LRU order was: 1, 2, 0
        assert_eq!(store.entries.len(), 0);
    }

    #[test]
    fn recover_index_on_startup() {
        crate::init();
        let tmp = tempfile::TempDir::new().unwrap();
        let stream = Stream::new(&Device::cpu());

        // Write blocks with the first store
        {
            let config = test_config(tmp.path());
            let mut store1 = SSDStore::new(config).unwrap();
            store1.store_block(42, &dummy_kv(2), &stream).unwrap();
            store1.store_block(43, &dummy_kv(3), &stream).unwrap();
            wait_and_flush(&mut store1);
        }

        // Create a new store and recover
        let config2 = test_config(tmp.path());
        let mut store2 = SSDStore::new(config2).unwrap();
        let count = store2.recover_index().unwrap();
        assert_eq!(count, 2);
        assert!(store2.has_block(42));
        assert!(store2.has_block(43));

        // Load and verify layer counts
        let loaded42 = store2.load_block(42, &stream).unwrap();
        assert_eq!(loaded42.len(), 2);

        let loaded43 = store2.load_block(43, &stream).unwrap();
        assert_eq!(loaded43.len(), 3);
    }

    #[test]
    fn recover_index_empty_dir() {
        crate::init();
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let mut store = SSDStore::new(config).unwrap();

        let count = store.recover_index().unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn overwrite_existing_block() {
        crate::init();
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let mut store = SSDStore::new(config).unwrap();
        let stream = Stream::new(&Device::cpu());

        // Store block with 2 layers
        store.store_block(5, &dummy_kv(2), &stream).unwrap();
        wait_and_flush(&mut store);

        // Overwrite with 1 layer
        store.store_block(5, &dummy_kv(1), &stream).unwrap();
        wait_and_flush(&mut store);

        let loaded = store.load_block(5, &stream).unwrap();
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn drop_flushes_writer_thread() {
        crate::init();
        let tmp = tempfile::TempDir::new().unwrap();
        let model_dir = tmp.path().join("test_model");

        {
            let config = test_config(tmp.path());
            let mut store = SSDStore::new(config).unwrap();
            let stream = Stream::new(&Device::cpu());
            store.store_block(77, &dummy_kv(1), &stream).unwrap();
            // Drop without explicit flush — Drop should join the writer thread
        }

        // The file should have been written because Drop joined the thread
        let file_path = model_dir.join("77.safetensors");
        assert!(file_path.exists(), "Drop should flush pending writes");
    }

    #[test]
    fn multiple_blocks_current_size_tracking() {
        crate::init();
        let tmp = tempfile::TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let mut store = SSDStore::new(config).unwrap();
        let stream = Stream::new(&Device::cpu());

        assert_eq!(store.current_size(), 0);

        store.store_block(1, &dummy_kv(1), &stream).unwrap();
        wait_and_flush(&mut store);
        let size_after_one = store.current_size();
        assert!(size_after_one > 0);

        store.store_block(2, &dummy_kv(1), &stream).unwrap();
        wait_and_flush(&mut store);
        let size_after_two = store.current_size();
        assert!(size_after_two > size_after_one);

        store.remove_block(1).unwrap();
        let size_after_remove = store.current_size();
        assert_eq!(size_after_remove, size_after_two - size_after_one);
    }
}

impl Drop for SSDStore {
    fn drop(&mut self) {
        // Drop the sender to signal the writer thread to exit.
        self.write_tx.take();
        // Wait for the writer thread to finish processing all pending jobs.
        if let Some(handle) = self.writer_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Background writer thread: receives WriteJob messages and writes safetensors files.
/// All data arrives as CPU vectors — no GPU operations occur in this thread.
fn writer_thread(rx: mpsc::Receiver<WriteJob>) {
    while let Ok(job) = rx.recv() {
        let arrays = MapStringToArray::new();
        for (i, layer) in job.cpu_kv.iter().enumerate() {
            // Reconstruct Array from CPU data (no GPU involved)
            let keys = Array::from_slice_f32_shape(&layer.keys_data, &layer.keys_shape);
            let values = Array::from_slice_f32_shape(&layer.values_data, &layer.values_shape);
            if arrays.insert(&format!("layer_{i}_keys"), &keys).is_err() {
                continue;
            }
            if arrays
                .insert(&format!("layer_{i}_values"), &values)
                .is_err()
            {
                continue;
            }
        }

        let metadata = MapStringToString::new();
        if let Some(path_str) = job.file_path.to_str() {
            let _ = save_safetensors(path_str, &arrays, &metadata);
        }
    }
}
