use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use mlx::{Array, Dtype};
use serde::{Deserialize, Serialize};

use crate::Result;

const SCHEMA_VERSION: u32 = 5;
const META_FILE: &str = "meta.json";
const PAYLOAD_FILE: &str = "payload.safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixLayerKind {
    FullDense,
    FullPaged,
    FullTurboQuantPacked,
    Linear,
    Mla,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixTensorSpec {
    pub dtype: Dtype,
    pub shape: Vec<i32>,
}

impl PrefixTensorSpec {
    pub fn from_array(array: &Array) -> Self {
        Self {
            dtype: array.dtype(),
            shape: array.shape().as_slice().to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixLayerSpec {
    pub kind: PrefixLayerKind,
    pub tensors: Vec<PrefixTensorSpec>,
}

impl PrefixLayerSpec {
    pub fn from_payload(payload: &PrefixLayerPayload) -> Self {
        match payload {
            PrefixLayerPayload::FullDense { k, v } => Self {
                kind: PrefixLayerKind::FullDense,
                tensors: vec![
                    PrefixTensorSpec::from_array(k),
                    PrefixTensorSpec::from_array(v),
                ],
            },
            PrefixLayerPayload::FullPaged { k_pages, v_pages } => Self {
                kind: PrefixLayerKind::FullPaged,
                tensors: vec![
                    PrefixTensorSpec::from_array(k_pages),
                    PrefixTensorSpec::from_array(v_pages),
                ],
            },
            PrefixLayerPayload::FullTurboQuantPacked {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            } => Self {
                kind: PrefixLayerKind::FullTurboQuantPacked,
                tensors: vec![
                    PrefixTensorSpec::from_array(k_packed),
                    PrefixTensorSpec::from_array(k_norms),
                    PrefixTensorSpec::from_array(v_packed),
                    PrefixTensorSpec::from_array(v_norms),
                ],
            },
            PrefixLayerPayload::Linear {
                conv_state,
                recurrent_state,
            } => Self {
                kind: PrefixLayerKind::Linear,
                tensors: vec![
                    PrefixTensorSpec::from_array(conv_state),
                    PrefixTensorSpec::from_array(recurrent_state),
                ],
            },
            PrefixLayerPayload::Mla { c_kv, k_pe } => Self {
                kind: PrefixLayerKind::Mla,
                tensors: vec![
                    PrefixTensorSpec::from_array(c_kv),
                    PrefixTensorSpec::from_array(k_pe),
                ],
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMtpLayerSpec {
    pub k: PrefixTensorSpec,
    pub v: PrefixTensorSpec,
}

impl PrefixMtpLayerSpec {
    pub fn from_payload(payload: &PrefixMtpLayerPayload) -> Self {
        Self {
            k: PrefixTensorSpec::from_array(&payload.k),
            v: PrefixTensorSpec::from_array(&payload.v),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedPrefixKeySpec {
    pub model_id: String,
    pub token_ids: Vec<i32>,
    pub cached_len: i32,
    pub fingerprint: Option<String>,
    pub block_size: i32,
    pub kv_cache_profile: Option<String>,
    pub main_layers: Vec<PrefixLayerSpec>,
    pub mtp_layers: Vec<PrefixMtpLayerSpec>,
    pub mtp_last_hidden: Option<PrefixTensorSpec>,
}

#[derive(Debug, Clone)]
pub struct PagedPrefixLayer {
    pub k_pages: Array,
    pub v_pages: Array,
}

#[derive(Debug, Clone)]
pub enum PrefixLayerPayload {
    FullDense {
        k: Array,
        v: Array,
    },
    FullPaged {
        k_pages: Array,
        v_pages: Array,
    },
    FullTurboQuantPacked {
        k_packed: Array,
        k_norms: Array,
        v_packed: Array,
        v_norms: Array,
    },
    Linear {
        conv_state: Array,
        recurrent_state: Array,
    },
    Mla {
        c_kv: Array,
        k_pe: Array,
    },
}

#[derive(Debug, Clone)]
pub struct PrefixMtpLayerPayload {
    pub k: Array,
    pub v: Array,
}

#[derive(Debug, Clone, Default)]
pub struct PagedPrefixEntry {
    pub main_layers: Vec<PrefixLayerPayload>,
    pub mtp_layers: Vec<PrefixMtpLayerPayload>,
    pub mtp_last_hidden: Option<Array>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagedPrefixEntryStats {
    pub cached_len: i32,
    pub main_layers: usize,
    pub full_dense_layers: usize,
    pub full_paged_layers: usize,
    pub linear_layers: usize,
    pub mla_layers: usize,
    pub mtp_layers: usize,
    pub full_paged_pages: usize,
    pub tensor_count: usize,
    pub payload_bytes: usize,
}

impl PagedPrefixEntry {
    pub fn main_layer_specs(&self) -> Vec<PrefixLayerSpec> {
        self.main_layers
            .iter()
            .map(PrefixLayerSpec::from_payload)
            .collect()
    }

    pub fn mtp_layer_specs(&self) -> Vec<PrefixMtpLayerSpec> {
        self.mtp_layers
            .iter()
            .map(PrefixMtpLayerSpec::from_payload)
            .collect()
    }

    pub fn mtp_last_hidden_spec(&self) -> Option<PrefixTensorSpec> {
        self.mtp_last_hidden
            .as_ref()
            .map(PrefixTensorSpec::from_array)
    }

    pub fn observability_stats(&self, cached_len: i32) -> PagedPrefixEntryStats {
        let mut stats = PagedPrefixEntryStats {
            cached_len,
            main_layers: self.main_layers.len(),
            mtp_layers: self.mtp_layers.len(),
            ..PagedPrefixEntryStats::default()
        };
        for layer in &self.main_layers {
            match layer {
                PrefixLayerPayload::FullDense { k, v } => {
                    stats.full_dense_layers += 1;
                    stats.tensor_count += 2;
                    stats.payload_bytes = stats
                        .payload_bytes
                        .saturating_add(tensor_payload_bytes(k))
                        .saturating_add(tensor_payload_bytes(v));
                }
                PrefixLayerPayload::FullPaged { k_pages, v_pages } => {
                    stats.full_paged_layers += 1;
                    stats.full_paged_pages += first_dim_usize(k_pages);
                    stats.tensor_count += 2;
                    stats.payload_bytes = stats
                        .payload_bytes
                        .saturating_add(tensor_payload_bytes(k_pages))
                        .saturating_add(tensor_payload_bytes(v_pages));
                }
                PrefixLayerPayload::FullTurboQuantPacked {
                    k_packed,
                    k_norms,
                    v_packed,
                    v_norms,
                } => {
                    stats.tensor_count += 4;
                    stats.payload_bytes = stats
                        .payload_bytes
                        .saturating_add(tensor_payload_bytes(k_packed))
                        .saturating_add(tensor_payload_bytes(k_norms))
                        .saturating_add(tensor_payload_bytes(v_packed))
                        .saturating_add(tensor_payload_bytes(v_norms));
                }
                PrefixLayerPayload::Linear {
                    conv_state,
                    recurrent_state,
                } => {
                    stats.linear_layers += 1;
                    stats.tensor_count += 2;
                    stats.payload_bytes = stats
                        .payload_bytes
                        .saturating_add(tensor_payload_bytes(conv_state))
                        .saturating_add(tensor_payload_bytes(recurrent_state));
                }
                PrefixLayerPayload::Mla { c_kv, k_pe } => {
                    stats.mla_layers += 1;
                    stats.tensor_count += 2;
                    stats.payload_bytes = stats
                        .payload_bytes
                        .saturating_add(tensor_payload_bytes(c_kv))
                        .saturating_add(tensor_payload_bytes(k_pe));
                }
            }
        }
        for layer in &self.mtp_layers {
            stats.tensor_count += 2;
            stats.payload_bytes = stats
                .payload_bytes
                .saturating_add(tensor_payload_bytes(&layer.k))
                .saturating_add(tensor_payload_bytes(&layer.v));
        }
        if let Some(last_hidden) = &self.mtp_last_hidden {
            stats.tensor_count += 1;
            stats.payload_bytes = stats
                .payload_bytes
                .saturating_add(tensor_payload_bytes(last_hidden));
        }
        stats
    }
}

#[derive(Debug, Clone)]
pub struct PagedPrefixStore {
    root: PathBuf,
    max_bytes: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedPrefixLoadStatus {
    Hit,
    MissingEntry,
    InvalidMetadata,
    MetadataMismatch,
    PayloadReadFailed,
    PayloadInvalid,
    EntryInvalid,
}

#[derive(Debug, Clone)]
pub struct PagedPrefixLoadResult {
    pub key: String,
    pub status: PagedPrefixLoadStatus,
    pub entry: Option<PagedPrefixEntry>,
    pub stats: Option<PagedPrefixEntryStats>,
}

pub const DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE: i32 = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedPrefixCacheConfig {
    pub root: PathBuf,
    pub model_id: String,
    pub block_size: i32,
    pub max_pages: i32,
    pub max_disk_bytes: Option<usize>,
}

impl PagedPrefixCacheConfig {
    pub fn new(
        root: impl AsRef<Path>,
        model_id: impl Into<String>,
        block_size: i32,
        max_pages: i32,
    ) -> Result<Self> {
        Self::new_with_max_disk_bytes(root, model_id, block_size, max_pages, None)
    }

    pub fn new_with_max_disk_bytes(
        root: impl AsRef<Path>,
        model_id: impl Into<String>,
        block_size: i32,
        max_pages: i32,
        max_disk_bytes: Option<usize>,
    ) -> Result<Self> {
        let config = Self {
            root: root.as_ref().to_path_buf(),
            model_id: model_id.into(),
            block_size,
            max_pages,
            max_disk_bytes,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn store(&self) -> PagedPrefixStore {
        PagedPrefixStore::new(&self.root).with_optional_max_bytes(self.max_disk_bytes)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_id.is_empty() {
            anyhow::bail!("PagedPrefixCacheConfig: model_id must not be empty");
        }
        if self.block_size <= 0 {
            anyhow::bail!(
                "PagedPrefixCacheConfig: block_size must be > 0, got {}",
                self.block_size
            );
        }
        if self.max_pages <= 0 {
            anyhow::bail!(
                "PagedPrefixCacheConfig: max_pages must be > 0, got {}",
                self.max_pages
            );
        }
        if self.max_disk_bytes == Some(0) {
            anyhow::bail!("PagedPrefixCacheConfig: max_disk_bytes must be > 0");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixLruCacheConfig {
    pub max_bytes: usize,
}

impl PrefixLruCacheConfig {
    pub fn new(max_bytes: usize) -> Result<Self> {
        let config = Self { max_bytes };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_bytes == 0 {
            anyhow::bail!("PrefixLruCacheConfig: max_bytes must be > 0");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixLruInsertStatus {
    Stored,
    Replaced,
    SkippedOversized,
}

#[derive(Debug, Clone)]
pub struct PrefixLruInsertResult {
    pub key: String,
    pub status: PrefixLruInsertStatus,
    pub stats: PagedPrefixEntryStats,
}

#[derive(Debug, Clone)]
struct PrefixLruEntry {
    spec: PagedPrefixKeySpec,
    entry: PagedPrefixEntry,
    stats: PagedPrefixEntryStats,
    generation: u64,
}

#[derive(Debug)]
pub struct PrefixLruCache {
    max_bytes: usize,
    total_bytes: usize,
    generation: u64,
    entries: HashMap<String, PrefixLruEntry>,
    recency: VecDeque<(String, u64)>,
}

impl PrefixLruCache {
    pub fn new(config: PrefixLruCacheConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            max_bytes: config.max_bytes,
            total_bytes: 0,
            generation: 0,
            entries: HashMap::new(),
            recency: VecDeque::new(),
        })
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn cached_lengths_descending(&self, max_cached_len: usize) -> Vec<i32> {
        let mut lengths = self
            .entries
            .values()
            .filter_map(|entry| {
                let cached_len = entry.spec.cached_len;
                if cached_len > 0 && cached_len as usize <= max_cached_len {
                    Some(cached_len)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        lengths.dedup();
        lengths
    }

    pub fn load_observed(&mut self, spec: &PagedPrefixKeySpec) -> Result<PagedPrefixLoadResult> {
        let key = PagedPrefixStore::key_for(spec);
        let Some(entry) = self.entries.get(&key) else {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::MissingEntry,
                entry: None,
                stats: None,
            });
        };
        if entry.spec != *spec {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::MetadataMismatch,
                entry: None,
                stats: Some(entry.stats),
            });
        }
        let cached_entry = entry.entry.clone();
        let stats = entry.stats;
        self.touch(&key);
        Ok(PagedPrefixLoadResult {
            key,
            status: PagedPrefixLoadStatus::Hit,
            entry: Some(cached_entry),
            stats: Some(stats),
        })
    }

    pub fn insert(
        &mut self,
        spec: PagedPrefixKeySpec,
        entry: PagedPrefixEntry,
    ) -> Result<PrefixLruInsertResult> {
        let validator = PagedPrefixStore::new(Path::new(""));
        validator.validate_spec(&spec)?;
        validator.validate_entry(&spec, &entry)?;

        let key = PagedPrefixStore::key_for(&spec);
        let stats = entry.observability_stats(spec.cached_len);
        if stats.payload_bytes > self.max_bytes {
            return Ok(PrefixLruInsertResult {
                key,
                status: PrefixLruInsertStatus::SkippedOversized,
                stats,
            });
        }

        let status = if let Some(previous) = self.entries.remove(&key) {
            self.total_bytes = self
                .total_bytes
                .saturating_sub(previous.stats.payload_bytes);
            PrefixLruInsertStatus::Replaced
        } else {
            PrefixLruInsertStatus::Stored
        };
        let generation = self.next_generation();
        self.total_bytes = self.total_bytes.saturating_add(stats.payload_bytes);
        self.entries.insert(
            key.clone(),
            PrefixLruEntry {
                spec,
                entry,
                stats,
                generation,
            },
        );
        self.recency.push_back((key.clone(), generation));
        self.evict_to_capacity();

        Ok(PrefixLruInsertResult { key, status, stats })
    }

    fn next_generation(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        self.generation
    }

    fn touch(&mut self, key: &str) {
        let generation = self.next_generation();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.generation = generation;
            self.recency.push_back((key.to_owned(), generation));
        }
    }

    fn evict_to_capacity(&mut self) {
        while self.total_bytes > self.max_bytes {
            let Some((key, generation)) = self.recency.pop_front() else {
                break;
            };
            let is_current = self
                .entries
                .get(&key)
                .is_some_and(|entry| entry.generation == generation);
            if !is_current {
                continue;
            }
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(entry.stats.payload_bytes);
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TensorSpecMetadata {
    dtype: String,
    shape: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct LayerSpecMetadata {
    kind: PrefixLayerKind,
    tensors: Vec<TensorSpecMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct MtpLayerSpecMetadata {
    k: TensorSpecMetadata,
    v: TensorSpecMetadata,
}

#[derive(Debug, Serialize)]
struct KeyMaterial<'a> {
    schema_version: u32,
    model_id: &'a str,
    token_ids: &'a [i32],
    cached_len: i32,
    fingerprint: Option<&'a str>,
    block_size: i32,
    kv_cache_profile: Option<&'a str>,
    main_layers: Vec<LayerSpecMetadata>,
    mtp_layers: Vec<MtpLayerSpecMetadata>,
    mtp_last_hidden: Option<TensorSpecMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PrefixMetadata {
    schema_version: u32,
    key: String,
    model_id: String,
    token_hash: String,
    token_count: usize,
    cached_len: i32,
    fingerprint_hash: Option<String>,
    block_size: i32,
    kv_cache_profile: Option<String>,
    main_layers: Vec<LayerSpecMetadata>,
    mtp_layers: Vec<MtpLayerSpecMetadata>,
    mtp_last_hidden: Option<TensorSpecMetadata>,
}

impl PagedPrefixStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            max_bytes: None,
        }
    }

    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = Some(max_bytes);
        self
    }

    fn with_optional_max_bytes(mut self, max_bytes: Option<usize>) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    pub fn key_for(spec: &PagedPrefixKeySpec) -> String {
        let material = KeyMaterial {
            schema_version: SCHEMA_VERSION,
            model_id: &spec.model_id,
            token_ids: &spec.token_ids,
            cached_len: spec.cached_len,
            fingerprint: spec.fingerprint.as_deref(),
            block_size: spec.block_size,
            kv_cache_profile: spec.kv_cache_profile.as_deref(),
            main_layers: layer_metadata(&spec.main_layers),
            mtp_layers: mtp_layer_metadata(&spec.mtp_layers),
            mtp_last_hidden: spec.mtp_last_hidden.as_ref().map(tensor_metadata),
        };
        let json = serde_json::to_string(&material)
            .expect("PagedPrefixStore::key_for serializes fixed metadata");
        stable_hex_hash(&json)
    }

    pub fn save(&self, spec: &PagedPrefixKeySpec, entry: &PagedPrefixEntry) -> Result<String> {
        self.validate_spec(spec)?;
        self.validate_entry(spec, entry)?;

        if let Some(max_bytes) = self.max_bytes {
            let payload_bytes = entry.observability_stats(spec.cached_len).payload_bytes;
            if payload_bytes > max_bytes {
                anyhow::bail!(
                    "PagedPrefixStore: entry payload_bytes {payload_bytes} exceeds max_bytes {max_bytes}"
                );
            }
        }

        let key = Self::key_for(spec);
        let metadata = metadata_for(spec, &key)?;
        let final_dir = self.root.join(&key);
        let tmp_dir = self
            .root
            .join(format!(".tmp-{key}-{}", uuid::Uuid::new_v4().simple()));
        fs::create_dir_all(&self.root)
            .with_context(|| format!("create prefix cache root {}", self.root.display()))?;
        fs::create_dir_all(&tmp_dir)
            .with_context(|| format!("create temporary prefix cache dir {}", tmp_dir.display()))?;

        let save_result = (|| -> Result<()> {
            let meta_path = tmp_dir.join(META_FILE);
            let payload_path = tmp_dir.join(PAYLOAD_FILE);
            let meta_bytes =
                serde_json::to_vec_pretty(&metadata).context("serialize paged prefix metadata")?;
            fs::write(&meta_path, meta_bytes)
                .with_context(|| format!("write prefix cache metadata {}", meta_path.display()))?;

            let mut tensors = HashMap::new();
            insert_entry_tensors(entry, &mut tensors);
            let mut safetensors_meta = HashMap::new();
            safetensors_meta.insert("ironmlx.prefix_cache.key".to_owned(), key.clone());
            safetensors_meta.insert(
                "ironmlx.prefix_cache.schema_version".to_owned(),
                SCHEMA_VERSION.to_string(),
            );
            let payload_path = payload_path.to_string_lossy().into_owned();
            mlx::io::save_safetensors(&payload_path, &tensors, &safetensors_meta)
                .with_context(|| format!("save prefix cache payload {payload_path}"))?;
            Ok(())
        })();

        if let Err(err) = save_result {
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(err);
        }

        if final_dir.exists() {
            fs::remove_dir_all(&final_dir)
                .with_context(|| format!("replace prefix cache dir {}", final_dir.display()))?;
        }
        fs::rename(&tmp_dir, &final_dir).with_context(|| {
            format!(
                "install prefix cache entry {} -> {}",
                tmp_dir.display(),
                final_dir.display()
            )
        })?;
        if let Some(max_bytes) = self.max_bytes {
            self.evict_to_disk_capacity(&key, max_bytes)?;
        }
        Ok(key)
    }

    pub fn save_if_absent(
        &self,
        spec: &PagedPrefixKeySpec,
        entry: &PagedPrefixEntry,
    ) -> Result<(String, bool)> {
        self.validate_spec(spec)?;
        self.validate_entry(spec, entry)?;

        let key = Self::key_for(spec);
        if self.entry_metadata_matches(spec, &key)? {
            return Ok((key, false));
        }
        self.save(spec, entry).map(|key| (key, true))
    }

    pub fn matching_entry_key(&self, spec: &PagedPrefixKeySpec) -> Result<Option<String>> {
        self.validate_spec(spec)?;
        let key = Self::key_for(spec);
        if self.entry_metadata_matches(spec, &key)? {
            Ok(Some(key))
        } else {
            Ok(None)
        }
    }

    pub fn cached_lengths_descending(&self, max_cached_len: i32) -> Result<Vec<i32>> {
        if max_cached_len <= 0 {
            return Ok(Vec::new());
        }
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(err) if err.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("read prefix cache root {}", self.root.display()));
            }
        };
        let mut lengths = Vec::new();
        for entry in entries {
            let entry =
                entry.with_context(|| format!("read prefix cache root {}", self.root.display()))?;
            let entry_path = entry.path();
            let file_type = entry.file_type().with_context(|| {
                format!("read prefix cache entry type {}", entry_path.display())
            })?;
            if !file_type.is_dir() {
                continue;
            }
            let meta_path = entry_path.join(META_FILE);
            let Some(metadata) = fs::read(&meta_path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<PrefixMetadata>(&bytes).ok())
            else {
                continue;
            };
            if metadata.schema_version == SCHEMA_VERSION
                && metadata.cached_len > 0
                && metadata.cached_len <= max_cached_len
            {
                lengths.push(metadata.cached_len);
            }
        }
        lengths.sort_unstable_by(|left, right| right.cmp(left));
        lengths.dedup();
        Ok(lengths)
    }

    pub fn load(&self, spec: &PagedPrefixKeySpec) -> Result<Option<PagedPrefixEntry>> {
        Ok(self.load_observed(spec)?.entry)
    }

    pub fn load_observed(&self, spec: &PagedPrefixKeySpec) -> Result<PagedPrefixLoadResult> {
        self.validate_spec(spec)?;
        let key = Self::key_for(spec);
        let entry_dir = self.root.join(&key);
        if !entry_dir.is_dir() {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::MissingEntry,
                entry: None,
                stats: None,
            });
        }

        let metadata = metadata_for(spec, &key)?;
        let meta_path = entry_dir.join(META_FILE);
        let actual_metadata = match fs::read(&meta_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PrefixMetadata>(&bytes).ok())
        {
            Some(meta) => meta,
            None => {
                return Ok(PagedPrefixLoadResult {
                    key,
                    status: PagedPrefixLoadStatus::InvalidMetadata,
                    entry: None,
                    stats: None,
                });
            }
        };
        if actual_metadata != metadata {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::MetadataMismatch,
                entry: None,
                stats: None,
            });
        }

        let payload_path = entry_dir.join(PAYLOAD_FILE);
        let payload_path_string = payload_path.to_string_lossy().into_owned();
        let (mut tensors, _metadata) = match mlx::io::load_safetensors(&payload_path_string) {
            Ok(loaded) => loaded,
            Err(_) => {
                return Ok(PagedPrefixLoadResult {
                    key,
                    status: PagedPrefixLoadStatus::PayloadReadFailed,
                    entry: None,
                    stats: None,
                });
            }
        };
        let entry = match entry_from_tensors(spec, &mut tensors) {
            Ok(entry) => entry,
            Err(_) => {
                return Ok(PagedPrefixLoadResult {
                    key,
                    status: PagedPrefixLoadStatus::PayloadInvalid,
                    entry: None,
                    stats: None,
                });
            }
        };
        if !tensors.is_empty() {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::PayloadInvalid,
                entry: None,
                stats: None,
            });
        }
        if self.validate_entry(spec, &entry).is_err() {
            return Ok(PagedPrefixLoadResult {
                key,
                status: PagedPrefixLoadStatus::EntryInvalid,
                entry: None,
                stats: None,
            });
        }
        let stats = entry.observability_stats(spec.cached_len);
        Ok(PagedPrefixLoadResult {
            key,
            status: PagedPrefixLoadStatus::Hit,
            entry: Some(entry),
            stats: Some(stats),
        })
    }

    fn entry_metadata_matches(&self, spec: &PagedPrefixKeySpec, key: &str) -> Result<bool> {
        let entry_dir = self.root.join(key);
        if !entry_dir.is_dir() {
            return Ok(false);
        }
        if !entry_dir.join(PAYLOAD_FILE).is_file() {
            return Ok(false);
        }
        let meta_path = entry_dir.join(META_FILE);
        let Some(actual_metadata) = fs::read(&meta_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PrefixMetadata>(&bytes).ok())
        else {
            return Ok(false);
        };
        Ok(actual_metadata == metadata_for(spec, key)?)
    }

    fn evict_to_disk_capacity(&self, keep_key: &str, max_bytes: usize) -> Result<()> {
        let mut entries = self.disk_entries()?;
        let mut total_bytes = entries
            .iter()
            .fold(0_u64, |total, entry| total.saturating_add(entry.bytes));
        if total_bytes <= max_bytes as u64 {
            return Ok(());
        }

        entries.sort_by(|left, right| {
            left.modified
                .cmp(&right.modified)
                .then_with(|| left.key.cmp(&right.key))
        });
        for entry in entries {
            if entry.key == keep_key {
                continue;
            }
            fs::remove_dir_all(&entry.path)
                .with_context(|| format!("evict prefix cache entry {}", entry.path.display()))?;
            total_bytes = total_bytes.saturating_sub(entry.bytes);
            if total_bytes <= max_bytes as u64 {
                break;
            }
        }

        if total_bytes > max_bytes as u64 {
            anyhow::bail!(
                "PagedPrefixStore: retained entry exceeds max_bytes {max_bytes}; total_bytes={total_bytes}"
            );
        }
        Ok(())
    }

    fn disk_entries(&self) -> Result<Vec<PrefixDiskEntry>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(err) if err.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("read prefix cache root {}", self.root.display()));
            }
        };
        let mut result = Vec::new();
        for entry in entries {
            let entry =
                entry.with_context(|| format!("read prefix cache root {}", self.root.display()))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("read prefix cache entry type {}", path.display()))?;
            if !file_type.is_dir() {
                continue;
            }
            let key = entry.file_name().to_string_lossy().into_owned();
            if key.starts_with(".tmp-") {
                continue;
            }
            let metadata = entry
                .metadata()
                .with_context(|| format!("read prefix cache metadata {}", path.display()))?;
            let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
            let bytes = directory_disk_bytes(&path)?;
            result.push(PrefixDiskEntry {
                key,
                path,
                bytes,
                modified,
            });
        }
        Ok(result)
    }

    fn validate_spec(&self, spec: &PagedPrefixKeySpec) -> Result<()> {
        if spec.model_id.is_empty() {
            anyhow::bail!("PagedPrefixStore: model_id must not be empty");
        }
        if spec.block_size <= 0 {
            anyhow::bail!(
                "PagedPrefixStore: block_size must be > 0, got {}",
                spec.block_size
            );
        }
        if spec.cached_len <= 0 {
            anyhow::bail!(
                "PagedPrefixStore: cached_len must be > 0, got {}",
                spec.cached_len
            );
        }
        if spec.token_ids.len() != spec.cached_len as usize {
            anyhow::bail!(
                "PagedPrefixStore: token_ids.len()={} != cached_len {}",
                spec.token_ids.len(),
                spec.cached_len
            );
        }
        if spec
            .fingerprint
            .as_ref()
            .is_some_and(|fingerprint| fingerprint.is_empty())
        {
            anyhow::bail!("PagedPrefixStore: fingerprint must not be empty when present");
        }
        if spec
            .kv_cache_profile
            .as_ref()
            .is_some_and(|profile| profile.is_empty())
        {
            anyhow::bail!("PagedPrefixStore: kv_cache_profile must not be empty when present");
        }
        if spec.main_layers.is_empty() && spec.mtp_layers.is_empty() {
            anyhow::bail!("PagedPrefixStore: entry must contain at least one cache layer");
        }
        if spec.mtp_layers.is_empty() && spec.mtp_last_hidden.is_some() {
            anyhow::bail!("PagedPrefixStore: mtp_last_hidden cannot be present without mtp_layers");
        }
        if !spec.mtp_layers.is_empty() && spec.mtp_last_hidden.is_none() {
            anyhow::bail!("PagedPrefixStore: mtp_layers require mtp_last_hidden");
        }
        for (idx, layer) in spec.main_layers.iter().enumerate() {
            validate_layer_spec(idx, layer, spec.block_size, spec.cached_len)?;
        }
        for (idx, layer) in spec.mtp_layers.iter().enumerate() {
            validate_tensor_spec(&format!("mtp layer {idx} K"), &layer.k)?;
            validate_tensor_spec(&format!("mtp layer {idx} V"), &layer.v)?;
        }
        if let Some(last_hidden) = &spec.mtp_last_hidden {
            validate_tensor_spec("mtp last_hidden", last_hidden)?;
        }
        Ok(())
    }

    fn validate_entry(&self, spec: &PagedPrefixKeySpec, entry: &PagedPrefixEntry) -> Result<()> {
        if entry.main_layers.len() != spec.main_layers.len() {
            anyhow::bail!(
                "PagedPrefixStore: main layer count {} != spec {}",
                entry.main_layers.len(),
                spec.main_layers.len()
            );
        }
        if entry.mtp_layers.len() != spec.mtp_layers.len() {
            anyhow::bail!(
                "PagedPrefixStore: MTP layer count {} != spec {}",
                entry.mtp_layers.len(),
                spec.mtp_layers.len()
            );
        }
        for (idx, (layer_spec, payload)) in spec
            .main_layers
            .iter()
            .zip(entry.main_layers.iter())
            .enumerate()
        {
            validate_layer_payload(idx, layer_spec, payload)?;
        }
        for (idx, (layer_spec, payload)) in spec
            .mtp_layers
            .iter()
            .zip(entry.mtp_layers.iter())
            .enumerate()
        {
            validate_tensor_payload(&format!("mtp layer {idx} K"), &layer_spec.k, &payload.k)?;
            validate_tensor_payload(&format!("mtp layer {idx} V"), &layer_spec.v, &payload.v)?;
        }
        match (&spec.mtp_last_hidden, &entry.mtp_last_hidden) {
            (Some(tensor_spec), Some(payload)) => {
                validate_tensor_payload("mtp last_hidden", tensor_spec, payload)?;
            }
            (None, None) => {}
            (Some(_), None) => anyhow::bail!("PagedPrefixStore: missing mtp_last_hidden payload"),
            (None, Some(_)) => {
                anyhow::bail!("PagedPrefixStore: unexpected mtp_last_hidden payload")
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct PrefixDiskEntry {
    key: String,
    path: PathBuf,
    bytes: u64,
    modified: SystemTime,
}

fn directory_disk_bytes(path: &Path) -> Result<u64> {
    let mut total = 0_u64;
    for entry in
        fs::read_dir(path).with_context(|| format!("read prefix cache entry {}", path.display()))?
    {
        let entry = entry.with_context(|| format!("read prefix cache entry {}", path.display()))?;
        let metadata = entry.metadata().with_context(|| {
            format!(
                "read prefix cache entry metadata {}",
                entry.path().display()
            )
        })?;
        if metadata.is_dir() {
            total = total.saturating_add(directory_disk_bytes(&entry.path())?);
        } else {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

fn validate_layer_spec(
    idx: usize,
    spec: &PrefixLayerSpec,
    block_size: i32,
    cached_len: i32,
) -> Result<()> {
    for (tensor_idx, tensor) in spec.tensors.iter().enumerate() {
        validate_tensor_spec(&format!("layer {idx} tensor {tensor_idx}"), tensor)?;
    }
    match spec.kind {
        PrefixLayerKind::FullDense => {
            require_tensor_count(idx, spec, 2)?;
            for (name, tensor) in ["K", "V"].iter().zip(spec.tensors.iter()) {
                let dims = tensor.shape.as_slice();
                if dims.len() != 4
                    || dims[0] != 1
                    || dims[1] <= 0
                    || dims[2] != cached_len
                    || dims[3] <= 0
                {
                    anyhow::bail!(
                        "PagedPrefixStore: FullDense layer {idx} {name} shape {:?} incompatible with [1,heads,{cached_len},dim]",
                        dims
                    );
                }
            }
        }
        PrefixLayerKind::FullPaged => {
            require_tensor_count(idx, spec, 2)?;
            for (name, tensor) in ["K", "V"].iter().zip(spec.tensors.iter()) {
                let dims = tensor.shape.as_slice();
                if dims.len() != 4
                    || dims[0] < 0
                    || dims[1] <= 0
                    || dims[2] != block_size
                    || dims[3] <= 0
                {
                    anyhow::bail!(
                        "PagedPrefixStore: FullPaged layer {idx} {name} shape {:?} incompatible with [pages,heads,{block_size},dim]",
                        dims
                    );
                }
            }
        }
        PrefixLayerKind::FullTurboQuantPacked => {
            require_tensor_count(idx, spec, 4)?;
            validate_turboquant_packed_spec(idx, "K packed", &spec.tensors[0], cached_len)?;
            validate_turboquant_norm_spec(idx, "K norms", &spec.tensors[1], cached_len)?;
            validate_turboquant_packed_spec(idx, "V packed", &spec.tensors[2], cached_len)?;
            validate_turboquant_norm_spec(idx, "V norms", &spec.tensors[3], cached_len)?;
        }
        PrefixLayerKind::Linear | PrefixLayerKind::Mla => {
            require_tensor_count(idx, spec, 2)?;
        }
    }
    Ok(())
}

fn require_tensor_count(idx: usize, spec: &PrefixLayerSpec, expected: usize) -> Result<()> {
    if spec.tensors.len() != expected {
        anyhow::bail!(
            "PagedPrefixStore: layer {idx} kind {:?} must describe exactly {expected} tensors, got {}",
            spec.kind,
            spec.tensors.len()
        );
    }
    Ok(())
}

fn validate_turboquant_packed_spec(
    idx: usize,
    name: &str,
    tensor: &PrefixTensorSpec,
    cached_len: i32,
) -> Result<()> {
    if tensor.dtype != Dtype::Uint32 {
        anyhow::bail!(
            "PagedPrefixStore: FullTurboQuantPacked layer {idx} {name} dtype {} != Uint32",
            tensor.dtype
        );
    }
    let dims = tensor.shape.as_slice();
    if dims.len() != 4 || dims[0] != 1 || dims[1] <= 0 || dims[2] != cached_len || dims[3] <= 0 {
        anyhow::bail!(
            "PagedPrefixStore: FullTurboQuantPacked layer {idx} {name} shape {:?} incompatible with [1,heads,{cached_len},packed_dim]",
            dims
        );
    }
    Ok(())
}

fn validate_turboquant_norm_spec(
    idx: usize,
    name: &str,
    tensor: &PrefixTensorSpec,
    cached_len: i32,
) -> Result<()> {
    if tensor.dtype != Dtype::Float32 {
        anyhow::bail!(
            "PagedPrefixStore: FullTurboQuantPacked layer {idx} {name} dtype {} != Float32",
            tensor.dtype
        );
    }
    let dims = tensor.shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] != cached_len {
        anyhow::bail!(
            "PagedPrefixStore: FullTurboQuantPacked layer {idx} {name} shape {:?} incompatible with [1,heads,{cached_len}]",
            dims
        );
    }
    Ok(())
}

fn validate_tensor_spec(name: &str, spec: &PrefixTensorSpec) -> Result<()> {
    if spec.shape.iter().any(|&dim| dim < 0) {
        anyhow::bail!(
            "PagedPrefixStore: {name} shape {:?} contains a negative dimension",
            spec.shape
        );
    }
    Ok(())
}

fn validate_layer_payload(
    idx: usize,
    spec: &PrefixLayerSpec,
    payload: &PrefixLayerPayload,
) -> Result<()> {
    match (spec.kind, payload) {
        (PrefixLayerKind::FullDense, PrefixLayerPayload::FullDense { k, v }) => {
            validate_tensor_payload(&format!("layer {idx} dense K"), &spec.tensors[0], k)?;
            validate_tensor_payload(&format!("layer {idx} dense V"), &spec.tensors[1], v)?;
        }
        (PrefixLayerKind::FullPaged, PrefixLayerPayload::FullPaged { k_pages, v_pages }) => {
            validate_tensor_payload(&format!("layer {idx} full K"), &spec.tensors[0], k_pages)?;
            validate_tensor_payload(&format!("layer {idx} full V"), &spec.tensors[1], v_pages)?;
        }
        (
            PrefixLayerKind::FullTurboQuantPacked,
            PrefixLayerPayload::FullTurboQuantPacked {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            },
        ) => {
            validate_tensor_payload(
                &format!("layer {idx} TurboQuant K packed"),
                &spec.tensors[0],
                k_packed,
            )?;
            validate_tensor_payload(
                &format!("layer {idx} TurboQuant K norms"),
                &spec.tensors[1],
                k_norms,
            )?;
            validate_tensor_payload(
                &format!("layer {idx} TurboQuant V packed"),
                &spec.tensors[2],
                v_packed,
            )?;
            validate_tensor_payload(
                &format!("layer {idx} TurboQuant V norms"),
                &spec.tensors[3],
                v_norms,
            )?;
        }
        (
            PrefixLayerKind::Linear,
            PrefixLayerPayload::Linear {
                conv_state,
                recurrent_state,
            },
        ) => {
            validate_tensor_payload(
                &format!("layer {idx} linear conv_state"),
                &spec.tensors[0],
                conv_state,
            )?;
            validate_tensor_payload(
                &format!("layer {idx} linear recurrent_state"),
                &spec.tensors[1],
                recurrent_state,
            )?;
        }
        (PrefixLayerKind::Mla, PrefixLayerPayload::Mla { c_kv, k_pe }) => {
            validate_tensor_payload(&format!("layer {idx} MLA c_kv"), &spec.tensors[0], c_kv)?;
            validate_tensor_payload(&format!("layer {idx} MLA k_pe"), &spec.tensors[1], k_pe)?;
        }
        _ => anyhow::bail!(
            "PagedPrefixStore: layer {idx} payload kind does not match {:?}",
            spec.kind
        ),
    }
    Ok(())
}

fn validate_tensor_payload(name: &str, spec: &PrefixTensorSpec, tensor: &Array) -> Result<()> {
    if tensor.dtype() != spec.dtype {
        anyhow::bail!(
            "PagedPrefixStore: {name} dtype {} != expected {}",
            tensor.dtype(),
            spec.dtype
        );
    }
    let shape = tensor.shape();
    if shape.as_slice() != spec.shape.as_slice() {
        anyhow::bail!(
            "PagedPrefixStore: {name} shape {:?} != expected {:?}",
            shape.as_slice(),
            spec.shape
        );
    }
    Ok(())
}

fn metadata_for(spec: &PagedPrefixKeySpec, key: &str) -> Result<PrefixMetadata> {
    Ok(PrefixMetadata {
        schema_version: SCHEMA_VERSION,
        key: key.to_owned(),
        model_id: spec.model_id.clone(),
        token_hash: token_hash(&spec.token_ids)?,
        token_count: spec.token_ids.len(),
        cached_len: spec.cached_len,
        fingerprint_hash: spec
            .fingerprint
            .as_ref()
            .map(|fingerprint| stable_hex_hash(fingerprint)),
        block_size: spec.block_size,
        kv_cache_profile: spec.kv_cache_profile.clone(),
        main_layers: layer_metadata(&spec.main_layers),
        mtp_layers: mtp_layer_metadata(&spec.mtp_layers),
        mtp_last_hidden: spec.mtp_last_hidden.as_ref().map(tensor_metadata),
    })
}

fn token_hash(token_ids: &[i32]) -> Result<String> {
    let json = serde_json::to_string(token_ids).context("serialize prefix token ids")?;
    Ok(stable_hex_hash(&json))
}

fn layer_metadata(layers: &[PrefixLayerSpec]) -> Vec<LayerSpecMetadata> {
    layers
        .iter()
        .map(|layer| LayerSpecMetadata {
            kind: layer.kind,
            tensors: layer.tensors.iter().map(tensor_metadata).collect(),
        })
        .collect()
}

fn mtp_layer_metadata(layers: &[PrefixMtpLayerSpec]) -> Vec<MtpLayerSpecMetadata> {
    layers
        .iter()
        .map(|layer| MtpLayerSpecMetadata {
            k: tensor_metadata(&layer.k),
            v: tensor_metadata(&layer.v),
        })
        .collect()
}

fn tensor_metadata(spec: &PrefixTensorSpec) -> TensorSpecMetadata {
    TensorSpecMetadata {
        dtype: spec.dtype.to_string(),
        shape: spec.shape.clone(),
    }
}

fn insert_entry_tensors(entry: &PagedPrefixEntry, tensors: &mut HashMap<String, Array>) {
    for (idx, layer) in entry.main_layers.iter().enumerate() {
        match layer {
            PrefixLayerPayload::FullDense { k, v } => {
                tensors.insert(main_dense_k_name(idx), k.clone());
                tensors.insert(main_dense_v_name(idx), v.clone());
            }
            PrefixLayerPayload::FullPaged { k_pages, v_pages } => {
                tensors.insert(main_full_k_name(idx), k_pages.clone());
                tensors.insert(main_full_v_name(idx), v_pages.clone());
            }
            PrefixLayerPayload::FullTurboQuantPacked {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            } => {
                tensors.insert(main_turboquant_k_packed_name(idx), k_packed.clone());
                tensors.insert(main_turboquant_k_norms_name(idx), k_norms.clone());
                tensors.insert(main_turboquant_v_packed_name(idx), v_packed.clone());
                tensors.insert(main_turboquant_v_norms_name(idx), v_norms.clone());
            }
            PrefixLayerPayload::Linear {
                conv_state,
                recurrent_state,
            } => {
                tensors.insert(main_linear_conv_name(idx), conv_state.clone());
                tensors.insert(main_linear_recurrent_name(idx), recurrent_state.clone());
            }
            PrefixLayerPayload::Mla { c_kv, k_pe } => {
                tensors.insert(main_mla_c_kv_name(idx), c_kv.clone());
                tensors.insert(main_mla_k_pe_name(idx), k_pe.clone());
            }
        }
    }
    for (idx, layer) in entry.mtp_layers.iter().enumerate() {
        tensors.insert(mtp_k_name(idx), layer.k.clone());
        tensors.insert(mtp_v_name(idx), layer.v.clone());
    }
    if let Some(last_hidden) = &entry.mtp_last_hidden {
        tensors.insert(mtp_last_hidden_name(), last_hidden.clone());
    }
}

fn entry_from_tensors(
    spec: &PagedPrefixKeySpec,
    tensors: &mut HashMap<String, Array>,
) -> Result<PagedPrefixEntry> {
    let mut main_layers = Vec::with_capacity(spec.main_layers.len());
    for (idx, layer_spec) in spec.main_layers.iter().enumerate() {
        let layer = match layer_spec.kind {
            PrefixLayerKind::FullDense => PrefixLayerPayload::FullDense {
                k: take_tensor(tensors, main_dense_k_name(idx))?,
                v: take_tensor(tensors, main_dense_v_name(idx))?,
            },
            PrefixLayerKind::FullPaged => PrefixLayerPayload::FullPaged {
                k_pages: take_tensor(tensors, main_full_k_name(idx))?,
                v_pages: take_tensor(tensors, main_full_v_name(idx))?,
            },
            PrefixLayerKind::FullTurboQuantPacked => PrefixLayerPayload::FullTurboQuantPacked {
                k_packed: take_tensor(tensors, main_turboquant_k_packed_name(idx))?,
                k_norms: take_tensor(tensors, main_turboquant_k_norms_name(idx))?,
                v_packed: take_tensor(tensors, main_turboquant_v_packed_name(idx))?,
                v_norms: take_tensor(tensors, main_turboquant_v_norms_name(idx))?,
            },
            PrefixLayerKind::Linear => PrefixLayerPayload::Linear {
                conv_state: take_tensor(tensors, main_linear_conv_name(idx))?,
                recurrent_state: take_tensor(tensors, main_linear_recurrent_name(idx))?,
            },
            PrefixLayerKind::Mla => PrefixLayerPayload::Mla {
                c_kv: take_tensor(tensors, main_mla_c_kv_name(idx))?,
                k_pe: take_tensor(tensors, main_mla_k_pe_name(idx))?,
            },
        };
        main_layers.push(layer);
    }

    let mut mtp_layers = Vec::with_capacity(spec.mtp_layers.len());
    for idx in 0..spec.mtp_layers.len() {
        mtp_layers.push(PrefixMtpLayerPayload {
            k: take_tensor(tensors, mtp_k_name(idx))?,
            v: take_tensor(tensors, mtp_v_name(idx))?,
        });
    }

    let mtp_last_hidden = if spec.mtp_last_hidden.is_some() {
        Some(take_tensor(tensors, mtp_last_hidden_name())?)
    } else {
        None
    };

    Ok(PagedPrefixEntry {
        main_layers,
        mtp_layers,
        mtp_last_hidden,
    })
}

fn take_tensor(tensors: &mut HashMap<String, Array>, name: String) -> Result<Array> {
    tensors
        .remove(&name)
        .ok_or_else(|| anyhow::anyhow!("PagedPrefixStore: missing tensor {name}"))
}

fn first_dim_usize(array: &Array) -> usize {
    array
        .shape()
        .as_slice()
        .first()
        .copied()
        .and_then(|dim| usize::try_from(dim).ok())
        .unwrap_or(0)
}

fn tensor_payload_bytes(array: &Array) -> usize {
    tensor_element_count(array).saturating_mul(dtype_size_bytes(array.dtype()))
}

fn tensor_element_count(array: &Array) -> usize {
    let mut elements = 1_usize;
    for &dim in array.shape().as_slice() {
        let Ok(dim) = usize::try_from(dim) else {
            return 0;
        };
        elements = elements.saturating_mul(dim);
    }
    elements
}

fn dtype_size_bytes(dtype: Dtype) -> usize {
    match dtype {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
        Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
        _ => 0,
    }
}

fn main_dense_k_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_dense_k")
}

fn main_dense_v_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_dense_v")
}

fn main_full_k_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_full_k_pages")
}

fn main_full_v_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_full_v_pages")
}

fn main_turboquant_k_packed_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_turboquant_k_packed")
}

fn main_turboquant_k_norms_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_turboquant_k_norms")
}

fn main_turboquant_v_packed_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_turboquant_v_packed")
}

fn main_turboquant_v_norms_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_turboquant_v_norms")
}

fn main_linear_conv_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_linear_conv_state")
}

fn main_linear_recurrent_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_linear_recurrent_state")
}

fn main_mla_c_kv_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_mla_c_kv")
}

fn main_mla_k_pe_name(layer_idx: usize) -> String {
    format!("main_{layer_idx:04}_mla_k_pe")
}

fn mtp_k_name(layer_idx: usize) -> String {
    format!("mtp_{layer_idx:04}_k")
}

fn mtp_v_name(layer_idx: usize) -> String {
    format!("mtp_{layer_idx:04}_v")
}

fn mtp_last_hidden_name() -> String {
    "mtp_last_hidden".to_owned()
}

fn stable_hex_hash(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype};

    fn temp_root(name: &str) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("ironmlx-{name}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn tensor_spec(array: &Array) -> PrefixTensorSpec {
        PrefixTensorSpec::from_array(array)
    }

    fn spec(
        tokens: Vec<i32>,
        cached_len: i32,
        fingerprint: Option<&str>,
        main_layers: Vec<PrefixLayerSpec>,
        mtp_layers: Vec<PrefixMtpLayerSpec>,
        mtp_last_hidden: Option<PrefixTensorSpec>,
    ) -> PagedPrefixKeySpec {
        PagedPrefixKeySpec {
            model_id: "qwen3-test".to_owned(),
            token_ids: tokens,
            cached_len,
            fingerprint: fingerprint.map(ToOwned::to_owned),
            block_size: 2,
            kv_cache_profile: None,
            main_layers,
            mtp_layers,
            mtp_last_hidden,
        }
    }

    #[test]
    fn prefix_key_includes_cached_len_and_fingerprint() {
        let layer = PrefixLayerSpec {
            kind: PrefixLayerKind::FullPaged,
            tensors: vec![
                PrefixTensorSpec {
                    dtype: Dtype::Float32,
                    shape: vec![2, 1, 2, 2],
                },
                PrefixTensorSpec {
                    dtype: Dtype::Float32,
                    shape: vec![2, 1, 2, 2],
                },
            ],
        };
        let a = spec(vec![1, 2, 3], 3, None, vec![layer.clone()], vec![], None);
        let b = spec(vec![1, 2, 3], 3, None, vec![layer.clone()], vec![], None);
        let shorter = spec(vec![1, 2], 2, None, vec![layer.clone()], vec![], None);
        let vl_a = spec(
            vec![1, 2, 3],
            3,
            Some("vl:image-a"),
            vec![layer.clone()],
            vec![],
            None,
        );
        let vl_b = spec(
            vec![1, 2, 3],
            3,
            Some("vl:image-b"),
            vec![layer],
            vec![],
            None,
        );
        assert_eq!(PagedPrefixStore::key_for(&a), PagedPrefixStore::key_for(&b));
        assert_ne!(
            PagedPrefixStore::key_for(&a),
            PagedPrefixStore::key_for(&shorter)
        );
        assert_ne!(
            PagedPrefixStore::key_for(&a),
            PagedPrefixStore::key_for(&vl_a)
        );
        assert_ne!(
            PagedPrefixStore::key_for(&vl_a),
            PagedPrefixStore::key_for(&vl_b)
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_entry_observability_stats_count_layers_pages_tensors_and_bytes() {
        let k_pages: Array = (&[1.0_f32; 8][..], (2_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (&[2.0_f32; 8][..], (2_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let conv_state: Array = (&[3.0_f32; 4][..], (1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let recurrent_state: Array = (&[4.0_f32; 4][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let mtp_k: Array = (&[5.0_f32; 3][..], (1_i32, 1_i32, 3_i32, 1_i32))
            .try_into()
            .unwrap();
        let mtp_v: Array = (&[6.0_f32; 3][..], (1_i32, 1_i32, 3_i32, 1_i32))
            .try_into()
            .unwrap();
        let mtp_last_hidden: Array = (&[7.0_f32; 2][..], (1_i32, 1_i32, 2_i32))
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![
                PrefixLayerPayload::FullPaged { k_pages, v_pages },
                PrefixLayerPayload::Linear {
                    conv_state,
                    recurrent_state,
                },
            ],
            mtp_layers: vec![PrefixMtpLayerPayload { k: mtp_k, v: mtp_v }],
            mtp_last_hidden: Some(mtp_last_hidden),
        };

        let stats = entry.observability_stats(3);

        assert_eq!(stats.cached_len, 3);
        assert_eq!(stats.main_layers, 2);
        assert_eq!(stats.full_paged_layers, 1);
        assert_eq!(stats.linear_layers, 1);
        assert_eq!(stats.mla_layers, 0);
        assert_eq!(stats.mtp_layers, 1);
        assert_eq!(stats.full_paged_pages, 2);
        assert_eq!(stats.tensor_count, 7);
        assert_eq!(stats.payload_bytes, 128);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_load_observed_reports_hit_stats_and_miss_reason() {
        let root = temp_root("prefix-store-observed");
        let store = PagedPrefixStore::new(&root);
        let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
        };
        let wanted = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);
        let missed = store.load_observed(&wanted).expect("miss load");
        assert_eq!(missed.status, PagedPrefixLoadStatus::MissingEntry);
        assert!(missed.entry.is_none());
        assert!(missed.stats.is_none());

        let key = store.save(&wanted, &entry).expect("save");
        let hit = store.load_observed(&wanted).expect("hit load");

        assert_eq!(hit.key, key);
        assert_eq!(hit.status, PagedPrefixLoadStatus::Hit);
        assert!(hit.entry.is_some());
        let stats = hit.stats.expect("hit stats");
        assert_eq!(stats.cached_len, 2);
        assert_eq!(stats.full_paged_layers, 1);
        assert_eq!(stats.full_paged_pages, 1);
        assert_eq!(stats.payload_bytes, 32);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_roundtrips_turboquant_packed_payload() {
        let root = temp_root("prefix-store-turboquant-packed");
        let store = PagedPrefixStore::new(&root);
        let k_packed: Array = (
            &[1_u32, 2, 3, 4, 5, 6, 7, 8][..],
            (1_i32, 2_i32, 4_i32, 1_i32),
        )
            .try_into()
            .unwrap();
        let k_norms: Array = (
            &[0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][..],
            (1_i32, 2_i32, 4_i32),
        )
            .try_into()
            .unwrap();
        let v_packed: Array = (
            &[11_u32, 12, 13, 14, 15, 16, 17, 18][..],
            (1_i32, 2_i32, 4_i32, 1_i32),
        )
            .try_into()
            .unwrap();
        let v_norms: Array = (
            &[1.1_f32, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8][..],
            (1_i32, 2_i32, 4_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullTurboQuantPacked {
                k_packed: k_packed.clone(),
                k_norms: k_norms.clone(),
                v_packed: v_packed.clone(),
                v_norms: v_norms.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
        };
        let mut wanted = spec(
            vec![1, 2, 3, 4],
            4,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        wanted.kv_cache_profile = Some("turboquant-k3v4".to_owned());

        let key = store.save(&wanted, &entry).expect("save packed payload");
        let hit = store.load_observed(&wanted).expect("hit load");

        assert_eq!(hit.key, key);
        assert_eq!(hit.status, PagedPrefixLoadStatus::Hit);
        let stats = hit.stats.expect("hit stats");
        assert_eq!(stats.tensor_count, 4);
        assert_eq!(stats.payload_bytes, 128);
        let loaded = hit.entry.expect("hit entry");
        let PrefixLayerPayload::FullTurboQuantPacked {
            k_packed,
            k_norms,
            v_packed,
            v_norms,
        } = &loaded.main_layers[0]
        else {
            panic!("expected TurboQuant packed payload");
        };
        assert_eq!(k_packed.shape().as_slice(), &[1, 2, 4, 1]);
        assert_eq!(k_norms.shape().as_slice(), &[1, 2, 4]);
        assert_eq!(v_packed.dtype(), Dtype::Uint32);
        assert_eq!(v_norms.dtype(), Dtype::Float32);

        std::fs::remove_dir_all(root).unwrap();
    }

    fn single_full_paged_entry(seed: f32) -> PagedPrefixEntry {
        let k_pages: Array = (
            &[seed, seed + 1.0, seed + 2.0, seed + 3.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[seed + 10.0, seed + 11.0, seed + 12.0, seed + 13.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged { k_pages, v_pages }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_lru_cache_hits_and_evicts_least_recent_entry() {
        let mut cache =
            PrefixLruCache::new(PrefixLruCacheConfig::new(64).expect("config")).expect("cache");
        let entry1 = single_full_paged_entry(1.0);
        let entry2 = single_full_paged_entry(2.0);
        let entry3 = single_full_paged_entry(3.0);
        let spec1 = spec(vec![1, 2], 2, None, entry1.main_layer_specs(), vec![], None);
        let spec2 = spec(vec![3, 4], 2, None, entry2.main_layer_specs(), vec![], None);
        let spec3 = spec(vec![5, 6], 2, None, entry3.main_layer_specs(), vec![], None);

        cache.insert(spec1.clone(), entry1).expect("insert spec1");
        cache.insert(spec2.clone(), entry2).expect("insert spec2");
        assert_eq!(cache.total_bytes(), 64);
        assert_eq!(
            cache.load_observed(&spec1).expect("load spec1").status,
            PagedPrefixLoadStatus::Hit
        );

        cache.insert(spec3.clone(), entry3).expect("insert spec3");

        assert_eq!(
            cache.load_observed(&spec1).expect("reload spec1").status,
            PagedPrefixLoadStatus::Hit
        );
        assert_eq!(
            cache.load_observed(&spec2).expect("reload spec2").status,
            PagedPrefixLoadStatus::MissingEntry
        );
        assert_eq!(
            cache.load_observed(&spec3).expect("reload spec3").status,
            PagedPrefixLoadStatus::Hit
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_lru_cache_cached_lengths_descending_filters_to_usable_lengths() {
        let mut cache =
            PrefixLruCache::new(PrefixLruCacheConfig::new(96).expect("config")).expect("cache");
        let entry2 = single_full_paged_entry(2.0);
        let entry4 = single_full_paged_entry(4.0);
        let entry6 = single_full_paged_entry(6.0);
        cache
            .insert(
                spec(vec![1, 2], 2, None, entry2.main_layer_specs(), vec![], None),
                entry2,
            )
            .expect("insert len2");
        cache
            .insert(
                spec(
                    vec![1, 2, 3, 4],
                    4,
                    None,
                    entry4.main_layer_specs(),
                    vec![],
                    None,
                ),
                entry4,
            )
            .expect("insert len4");
        cache
            .insert(
                spec(
                    vec![1, 2, 3, 4, 5, 6],
                    6,
                    None,
                    entry6.main_layer_specs(),
                    vec![],
                    None,
                ),
                entry6,
            )
            .expect("insert len6");

        assert_eq!(cache.cached_lengths_descending(10), vec![6, 4, 2]);
        assert_eq!(cache.cached_lengths_descending(5), vec![4, 2]);
        assert_eq!(cache.cached_lengths_descending(1), Vec::<i32>::new());
    }

    #[test]
    fn prefix_lru_cache_config_rejects_zero_capacity() {
        let err = PrefixLruCacheConfig::new(0).expect_err("zero capacity");
        assert!(err.to_string().contains("max_bytes must be > 0"));
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_cached_lengths_descending_filters_to_usable_lengths() {
        let root = temp_root("prefix-store-cached-lengths");
        let store = PagedPrefixStore::new(&root);
        let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
        };

        let len2 = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);
        let len4 = spec(
            vec![1, 2, 3, 4],
            4,
            None,
            entry.main_layer_specs(),
            vec![],
            None,
        );
        store.save(&len2, &entry).expect("save len2");
        store.save(&len4, &entry).expect("save len4");

        assert_eq!(
            store.cached_lengths_descending(10).expect("cached lengths"),
            vec![4, 2]
        );
        assert_eq!(
            store
                .cached_lengths_descending(3)
                .expect("filtered cached lengths"),
            vec![2]
        );
        assert_eq!(
            store
                .cached_lengths_descending(1)
                .expect("no usable cached lengths"),
            Vec::<i32>::new()
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    fn directory_size(path: &std::path::Path) -> u64 {
        let mut total = 0_u64;
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let metadata = entry.metadata().unwrap();
            if metadata.is_dir() {
                total += directory_size(&entry.path());
            } else {
                total += metadata.len();
            }
        }
        total
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_evicts_old_entries_to_ssd_capacity() {
        let root = temp_root("prefix-store-ssd-capacity");
        let entry1 = single_full_paged_entry(1.0);
        let entry2 = single_full_paged_entry(2.0);
        let spec1 = spec(vec![1, 2], 2, None, entry1.main_layer_specs(), vec![], None);
        let spec2 = spec(vec![3, 4], 2, None, entry2.main_layer_specs(), vec![], None);

        let unlimited = PagedPrefixStore::new(&root);
        let key1 = unlimited.save(&spec1, &entry1).expect("save first entry");
        let first_entry_bytes = directory_size(&root.join(&key1));
        let limited = PagedPrefixStore::new(&root).with_max_bytes(first_entry_bytes as usize + 1);
        let key2 = limited.save(&spec2, &entry2).expect("save second entry");

        assert!(
            !root.join(&key1).exists(),
            "oldest entry should be evicted when the SSD cache exceeds its limit"
        );
        assert!(root.join(&key2).exists(), "newly saved entry should remain");
        assert!(
            directory_size(&root) <= first_entry_bytes + 1,
            "SSD cache directory should stay within the configured capacity"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_save_if_absent_skips_existing_matching_entry() {
        let root = temp_root("prefix-store-save-if-absent");
        let store = PagedPrefixStore::new(&root);
        let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
        };
        let wanted = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);

        let (first_key, first_saved) = store.save_if_absent(&wanted, &entry).expect("first save");
        let payload_path = root.join(&first_key).join(PAYLOAD_FILE);
        let payload_before = std::fs::read(&payload_path).expect("payload before");
        let (second_key, second_saved) =
            store.save_if_absent(&wanted, &entry).expect("second save");

        assert_eq!(second_key, first_key);
        assert!(first_saved);
        assert!(!second_saved);
        assert_eq!(
            std::fs::read(&payload_path).expect("payload after"),
            payload_before
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_save_if_absent_rewrites_metadata_mismatch() {
        let root = temp_root("prefix-store-save-if-absent-mismatch");
        let store = PagedPrefixStore::new(&root);
        let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
        };
        let wanted = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);
        let key = store.save(&wanted, &entry).expect("save");
        let meta_path = root.join(&key).join(META_FILE);
        let mut meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        meta["block_size"] = serde_json::json!(4);
        std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

        let (rewritten_key, saved) = store.save_if_absent(&wanted, &entry).expect("rewrite");
        let loaded = store.load(&wanted).expect("load").expect("cache hit");

        assert_eq!(rewritten_key, key);
        assert!(saved);
        assert_eq!(loaded.main_layers.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_round_trips_mixed_payloads() {
        let root = temp_root("prefix-store-roundtrip");
        let store = PagedPrefixStore::new(&root);
        let k_data: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let v_data: Vec<f32> = k_data.iter().map(|v| v * 10.0).collect();
        let k_pages: Array = (k_data.as_slice(), (2_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (v_data.as_slice(), (2_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let conv_state: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let recurrent_state: Array = (&[5.0_f32, 6.0, 7.0, 8.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let c_kv: Array = (
            &[9.0_f32, 10.0, 11.0, 12.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let k_pe: Array = (&[13.0_f32, 14.0][..], (1_i32, 1_i32, 2_i32, 1_i32))
            .try_into()
            .unwrap();
        let mtp_k: Array = (&[15.0_f32, 16.0, 17.0][..], (1_i32, 1_i32, 3_i32, 1_i32))
            .try_into()
            .unwrap();
        let mtp_v: Array = (&[18.0_f32, 19.0, 20.0][..], (1_i32, 1_i32, 3_i32, 1_i32))
            .try_into()
            .unwrap();
        let mtp_last_hidden: Array = (&[21.0_f32, 22.0][..], (1_i32, 1_i32, 2_i32))
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![
                PrefixLayerPayload::FullPaged {
                    k_pages: k_pages.clone(),
                    v_pages: v_pages.clone(),
                },
                PrefixLayerPayload::Linear {
                    conv_state: conv_state.clone(),
                    recurrent_state: recurrent_state.clone(),
                },
                PrefixLayerPayload::Mla {
                    c_kv: c_kv.clone(),
                    k_pe: k_pe.clone(),
                },
            ],
            mtp_layers: vec![PrefixMtpLayerPayload {
                k: mtp_k.clone(),
                v: mtp_v.clone(),
            }],
            mtp_last_hidden: Some(mtp_last_hidden.clone()),
        };
        let spec = spec(
            vec![1, 2, 3],
            3,
            Some("vl:fingerprint"),
            entry.main_layer_specs(),
            entry.mtp_layer_specs(),
            Some(tensor_spec(&mtp_last_hidden)),
        );
        let key = store.save(&spec, &entry).expect("save");

        let loaded = store.load(&spec).expect("load").expect("cache hit");

        assert_eq!(key, PagedPrefixStore::key_for(&spec));
        assert_eq!(loaded.main_layers.len(), 3);
        match &loaded.main_layers[0] {
            PrefixLayerPayload::FullPaged { k_pages, v_pages } => {
                assert_eq!(k_pages.to_vec::<f32>().unwrap(), k_data);
                assert_eq!(v_pages.to_vec::<f32>().unwrap(), v_data);
            }
            other => panic!("expected full paged layer, got {other:?}"),
        }
        match &loaded.main_layers[1] {
            PrefixLayerPayload::Linear {
                conv_state,
                recurrent_state,
            } => {
                assert_eq!(
                    conv_state.to_vec::<f32>().unwrap(),
                    vec![1.0, 2.0, 3.0, 4.0]
                );
                assert_eq!(
                    recurrent_state.to_vec::<f32>().unwrap(),
                    vec![5.0, 6.0, 7.0, 8.0]
                );
            }
            other => panic!("expected linear layer, got {other:?}"),
        }
        match &loaded.main_layers[2] {
            PrefixLayerPayload::Mla { c_kv, k_pe } => {
                assert_eq!(c_kv.to_vec::<f32>().unwrap(), vec![9.0, 10.0, 11.0, 12.0]);
                assert_eq!(k_pe.to_vec::<f32>().unwrap(), vec![13.0, 14.0]);
            }
            other => panic!("expected MLA layer, got {other:?}"),
        }
        assert_eq!(loaded.mtp_layers.len(), 1);
        assert_eq!(
            loaded.mtp_layers[0].k.to_vec::<f32>().unwrap(),
            vec![15.0, 16.0, 17.0]
        );
        assert_eq!(
            loaded.mtp_layers[0].v.to_vec::<f32>().unwrap(),
            vec![18.0, 19.0, 20.0]
        );
        assert_eq!(
            loaded
                .mtp_last_hidden
                .as_ref()
                .expect("last hidden")
                .to_vec::<f32>()
                .unwrap(),
            vec![21.0, 22.0]
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn prefix_store_rejects_metadata_mismatch() {
        let root = temp_root("prefix-store-mismatch");
        let store = PagedPrefixStore::new(&root);
        let k_pages: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let entry = PagedPrefixEntry {
            main_layers: vec![PrefixLayerPayload::FullPaged {
                k_pages: k_pages.clone(),
                v_pages: v_pages.clone(),
            }],
            mtp_layers: vec![],
            mtp_last_hidden: None,
        };
        let wanted = spec(vec![1, 2], 2, None, entry.main_layer_specs(), vec![], None);
        let key = store.save(&wanted, &entry).expect("save");
        let meta_path = root.join(key).join("meta.json");
        let mut meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        meta["block_size"] = serde_json::json!(4);
        std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

        let loaded = store.load(&wanted).expect("load");

        assert!(loaded.is_none(), "tampered metadata must miss");
        std::fs::remove_dir_all(root).unwrap();
    }
}
