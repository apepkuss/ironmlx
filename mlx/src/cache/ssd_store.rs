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

/// Message sent to the background writer thread.
struct WriteJob {
    kv_data: Vec<(Array, Array)>,
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
        std::thread::spawn(move || {
            writer_thread(rx);
        });

        Ok(Self {
            config,
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            current_size_bytes: 0,
            write_tx: Some(tx),
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

        // Clone arrays for the background thread
        let cloned_kv: Vec<(Array, Array)> = kv_data
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Track as pending
        self.pending_writes.insert(
            block_id,
            PendingWrite {
                file_path: file_path.clone(),
                num_layers,
            },
        );

        // Send to writer thread
        if let Some(ref tx) = self.write_tx {
            let _ = tx.send(WriteJob {
                kv_data: cloned_kv,
                file_path,
            });
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

/// Background writer thread: receives WriteJob messages and writes safetensors files.
fn writer_thread(rx: mpsc::Receiver<WriteJob>) {
    while let Ok(job) = rx.recv() {
        let arrays = MapStringToArray::new();
        for (i, (keys, values)) in job.kv_data.iter().enumerate() {
            if arrays.insert(&format!("layer_{i}_keys"), keys).is_err() {
                continue;
            }
            if arrays.insert(&format!("layer_{i}_values"), values).is_err() {
                continue;
            }
        }

        let metadata = MapStringToString::new();
        if let Some(path_str) = job.file_path.to_str() {
            let _ = save_safetensors(path_str, &arrays, &metadata);
        }
    }
}
