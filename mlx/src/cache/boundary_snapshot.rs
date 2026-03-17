use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::array::Array;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::io::{load_safetensors, save_safetensors};
use crate::stream::Stream;
use crate::vector::{MapStringToArray, MapStringToString};

/// Key for identifying a snapshot: (request_id, token_count).
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct SnapshotKey {
    pub request_id: String,
    pub token_count: usize,
}

/// Stores boundary snapshots of non-sliceable cache states.
///
/// Used for cache types that cannot be split into blocks (e.g., RotatingKVCache,
/// ArraysCache). Instead of paged block storage, the entire cache state is
/// serialized as a safetensors file at token boundaries.
pub struct BoundarySnapshotStore {
    cache_dir: PathBuf,
    /// In-memory buffer for recently saved snapshots (avoid re-reading from SSD).
    buffer: HashMap<SnapshotKey, Vec<(Array, Array)>>,
    max_buffer_entries: usize,
}

impl BoundarySnapshotStore {
    /// Create a new store backed by the given directory.
    pub fn new(cache_dir: &Path, max_buffer_entries: usize) -> Self {
        std::fs::create_dir_all(cache_dir).ok();
        Self {
            cache_dir: cache_dir.to_path_buf(),
            buffer: HashMap::new(),
            max_buffer_entries,
        }
    }

    /// Save a snapshot of the cache state at a token boundary.
    pub fn save(
        &mut self,
        key: &SnapshotKey,
        cache_state: &[(Option<Array>, Option<Array>)],
    ) -> Result<()> {
        // Serialize to safetensors format
        let path = self.snapshot_path(key);
        let path_str = path
            .to_str()
            .ok_or_else(|| Error::Mlx("non-UTF-8 snapshot path".into()))?;

        let arrays = MapStringToArray::new();
        for (i, (k, v)) in cache_state.iter().enumerate() {
            if let Some(k) = k {
                arrays
                    .insert(&format!("layer_{i}_keys"), k)
                    .map_err(|e| Error::Mlx(format!("failed to insert snapshot key: {e}")))?;
            }
            if let Some(v) = v {
                arrays
                    .insert(&format!("layer_{i}_values"), v)
                    .map_err(|e| Error::Mlx(format!("failed to insert snapshot value: {e}")))?;
            }
        }

        let metadata = MapStringToString::new();
        save_safetensors(path_str, &arrays, &metadata)?;

        // Buffer in memory
        let buffered: Vec<(Array, Array)> = cache_state
            .iter()
            .filter_map(|(k, v)| match (k, v) {
                (Some(k), Some(v)) => Some((k.clone(), v.clone())),
                _ => None,
            })
            .collect();

        // Evict oldest if buffer full
        if self.buffer.len() >= self.max_buffer_entries
            && let Some(oldest_key) = self.buffer.keys().next().cloned()
        {
            self.buffer.remove(&oldest_key);
        }
        self.buffer.insert(key.clone(), buffered);

        Ok(())
    }

    /// Load a snapshot, first checking in-memory buffer, then SSD.
    pub fn load(
        &mut self,
        key: &SnapshotKey,
        num_layers: usize,
    ) -> Result<Vec<(Option<Array>, Option<Array>)>> {
        // Check buffer first
        if let Some(cached) = self.buffer.get(key) {
            let mut result = Vec::with_capacity(num_layers);
            for pair in cached {
                result.push((Some(pair.0.clone()), Some(pair.1.clone())));
            }
            // Pad if fewer layers in cache
            while result.len() < num_layers {
                result.push((None, None));
            }
            return Ok(result);
        }

        // Load from SSD
        let path = self.snapshot_path(key);
        if !path.exists() {
            return Err(Error::Mlx(format!("snapshot not found: {:?}", key)));
        }

        let path_str = path
            .to_str()
            .ok_or_else(|| Error::Mlx("non-UTF-8 snapshot path".into()))?;

        let stream = Stream::default_stream(&Device::cpu());
        let (arrays, _metadata) = load_safetensors(path_str, &stream)?;

        let mut result = vec![(None, None); num_layers];
        for (name, arr) in arrays.iter() {
            if let Some(rest) = name.strip_prefix("layer_")
                && let [idx_str, kind] = rest.splitn(2, '_').collect::<Vec<&str>>()[..]
                && let Ok(layer_idx) = idx_str.parse::<usize>()
                && layer_idx < num_layers
            {
                match kind {
                    "keys" => result[layer_idx].0 = Some(arr),
                    "values" => result[layer_idx].1 = Some(arr),
                    _ => {}
                }
            }
        }

        Ok(result)
    }

    /// Check if a snapshot exists (in buffer or on disk).
    pub fn has(&self, key: &SnapshotKey) -> bool {
        self.buffer.contains_key(key) || self.snapshot_path(key).exists()
    }

    /// Clean up all snapshots for a given request.
    pub fn cleanup_request(&mut self, request_id: &str) {
        self.buffer.retain(|k, _| k.request_id != request_id);
        // Also clean files
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&format!("{request_id}_")) {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }

    /// Number of snapshots currently in the in-memory buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    fn snapshot_path(&self, key: &SnapshotKey) -> PathBuf {
        self.cache_dir.join(format!(
            "{}_{}.safetensors",
            key.request_id, key.token_count
        ))
    }
}
