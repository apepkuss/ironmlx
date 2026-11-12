//! On-disk store for DFlash2 prefix artifacts.
//!
//! The artifacts of one conversation are prefixes of each other, so the store
//! keeps target K/V and target context as range files (consecutive positions
//! `start..stop`) that every artifact covering them references, and an
//! artifact directory holds only what is its own:
//!
//! ```text
//! <dir>/dflash2/<identity>/identity.json
//! <dir>/dflash2/<identity>/segments/<start>-<stop>-<hash>.safetensors
//! <dir>/dflash2/<identity>/context/<start>-<stop>-<hash>.safetensors
//! <dir>/dflash2/<identity>/artifacts/<key>/meta.json
//! <dir>/dflash2/<identity>/artifacts/<key>/state.safetensors
//! ```
//!
//! A save writes temporary files, syncs them, and installs them by rename
//! with `meta.json` last, so an artifact the index can see is complete.
//! Anything a restart cannot prove usable is deleted and counts as a miss.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{anyhow, Context};
use ironmlx_lm::core::cache::layer::{
    prefix_entry_for_row, restore_prefix_entry_for_row, LayerCache,
};
use ironmlx_lm::core::cache::prefix_payload::{PagedPrefixEntry, PrefixLayerPayload};
use mlx::{Array, StreamOrDevice};
use serde::{Deserialize, Serialize};

use crate::Result;

/// The subdirectory of a prefix cache directory this store owns.
pub(crate) const DFLASH2_STORE_DIR: &str = "dflash2";

const FORMAT_VERSION: u32 = 1;
/// An artifact referencing more range files than this is rewritten to one
/// K/V file and one context file.
const MAX_RANGE_FILES: usize = 256;
/// Newest boundaries a conversation always keeps.
pub(crate) const RECENT_BOUNDARIES: usize = 3;
/// Another identity's store (another model, build or numeric configuration)
/// is removed when it has not been opened for this long.
const STALE_IDENTITY_AFTER: std::time::Duration = std::time::Duration::from_secs(30 * 24 * 3600);

const IDENTITY_FILE: &str = "identity.json";
const META_FILE: &str = "meta.json";
const STATE_FILE: &str = "state.safetensors";
const SEGMENTS_DIR: &str = "segments";
const CONTEXT_DIR: &str = "context";
const ARTIFACTS_DIR: &str = "artifacts";
const TMP_MARK: &str = ".tmp-";
const LAST_HIDDEN: &str = "last_hidden";
const CONTEXT_TENSOR: &str = "hidden";

/// Everything that decides whether stored state is the state this process
/// would compute. Each identity has its own directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DFlash2StoreIdentity {
    pub(crate) format_version: u32,
    pub(crate) target_model_digest: String,
    pub(crate) target_execution_fingerprint: String,
    /// The build and every resolved switch that can change target arithmetic
    /// (see `store_execution_profile` in the DFlash2 actor).
    pub(crate) execution_profile: String,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) target_layer_ids: Vec<usize>,
    pub(crate) context_limit: usize,
}

impl DFlash2StoreIdentity {
    pub(crate) fn for_model_dir(
        model_dir: &Path,
        target_execution_fingerprint: &str,
        execution_profile: &str,
        prefill_chunk_size: usize,
        target_layer_ids: &[usize],
        context_limit: usize,
    ) -> Result<Self> {
        Ok(Self {
            format_version: FORMAT_VERSION,
            target_model_digest: model_dir_digest(model_dir)?,
            target_execution_fingerprint: target_execution_fingerprint.to_owned(),
            execution_profile: execution_profile.to_owned(),
            prefill_chunk_size,
            target_layer_ids: target_layer_ids.to_vec(),
            context_limit,
        })
    }

    pub(crate) fn key(&self) -> String {
        let material = serde_json::to_string(self).unwrap_or_default();
        format!("{:016x}", fnv1a(material.as_bytes()))
    }
}

/// `config.json` and the name and size of every weight file: changes when
/// the weights do, stays put across restarts, and reads no weight data.
fn model_dir_digest(model_dir: &Path) -> Result<String> {
    let config = fs::read(model_dir.join("config.json"))
        .with_context(|| format!("read {}/config.json", model_dir.display()))?;
    let mut weights = Vec::new();
    for entry in fs::read_dir(model_dir)
        .with_context(|| format!("read model directory {}", model_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".safetensors") {
            // Follows the symlinks of a Hugging Face snapshot.
            weights.push((name, fs::metadata(entry.path())?.len()));
        }
    }
    weights.sort();
    let mut hash = fnv1a(&config);
    for (name, len) in weights {
        hash = fnv1a_extend(hash, name.as_bytes());
        hash = fnv1a_extend(hash, &len.to_le_bytes());
    }
    Ok(format!("{hash:016x}"))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_extend(0xcbf2_9ce4_8422_2325, bytes)
}

fn fnv1a_extend(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn token_hash(token_ids: &[u32]) -> u64 {
    token_ids.iter().fold(fnv1a(&[]), |hash, token| {
        fnv1a_extend(hash, &token.to_le_bytes())
    })
}

fn artifact_key(token_ids: &[u32], fingerprint: &str) -> String {
    let hash = fnv1a_extend(token_hash(token_ids), fingerprint.as_bytes());
    format!("{hash:016x}")
}

fn range_file_name(token_ids: &[u32], fingerprint: &str, start: i32, stop: i32) -> String {
    let covered = &token_ids[..usize::try_from(stop).unwrap_or(0).min(token_ids.len())];
    let hash = fnv1a_extend(token_hash(covered), fingerprint.as_bytes());
    format!("{start:08}-{stop:08}-{hash:016x}.safetensors")
}

/// Of one conversation's boundaries `(tokens, id)`, the ones it no longer
/// needs: it keeps its [`RECENT_BOUNDARIES`] newest, and of the older ones
/// the shortest in each power-of-two length range (a prompt that rewinds
/// further recomputes at most half of what it keeps).
pub(crate) fn superseded_boundaries<T: Copy>(boundaries: &mut [(usize, T)]) -> Vec<T> {
    boundaries.sort_unstable_by_key(|(len, _)| *len);
    let older = boundaries.len().saturating_sub(RECENT_BOUNDARIES);
    let mut dropped = Vec::new();
    // Shortest first: the boundary kept for a length range never changes as
    // the conversation grows.
    let mut kept_range = None;
    for &(len, id) in &boundaries[..older] {
        let range = len.checked_ilog2();
        if kept_range == Some(range) {
            dropped.push(id);
        } else {
            kept_range = Some(range);
        }
    }
    dropped
}

#[derive(Debug, Clone)]
pub(crate) enum DFlash2DiskLayer {
    /// A full-attention layer; its K/V is in the artifact's range files.
    Dense,
    /// Fixed-size per-artifact state.
    State(PrefixLayerPayload),
}

/// K and V of every full-attention layer at positions `start..stop`, each
/// `[1, heads, stop - start, dim]`.
#[derive(Debug)]
pub(crate) struct DFlash2DiskRange {
    pub(crate) start: i32,
    pub(crate) stop: i32,
    pub(crate) layers: Vec<(Array, Array)>,
}

/// One boundary artifact to store. Every array is evaluated and compact (not
/// a view): the writer copies bytes and builds no graph.
#[derive(Debug)]
pub(crate) struct DFlash2DiskSave {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) fingerprint: String,
    pub(crate) layers: Vec<DFlash2DiskLayer>,
    /// The stored artifact this one extends (from
    /// [`DFlash2DiskStore::lineage`]); its K/V and context are referenced,
    /// not rewritten.
    pub(crate) parent: Option<String>,
    /// Shorter stored artifacts of the same conversation, the parent among
    /// them.
    pub(crate) ancestors: Vec<String>,
    /// Consecutive ranges from the parent's length to this artifact's.
    pub(crate) kv: Vec<DFlash2DiskRange>,
    /// `[1, cached_len - context_start, hidden]`: the context positions the
    /// parent does not hold.
    pub(crate) context_start: i32,
    pub(crate) context: Array,
    pub(crate) context_tail_len: i32,
    pub(crate) last_hidden: Array,
}

impl DFlash2DiskSave {
    /// Bytes the writer holds until the save is on disk.
    pub(crate) fn payload_bytes(&self) -> usize {
        let mut bytes = array_bytes(&self.context) + array_bytes(&self.last_hidden);
        for range in &self.kv {
            for (k, v) in &range.layers {
                bytes += array_bytes(k) + array_bytes(v);
            }
        }
        for layer in &self.layers {
            if let DFlash2DiskLayer::State(payload) = layer {
                if let Ok((_, arrays)) = state_arrays(payload) {
                    bytes += arrays.iter().map(|array| array_bytes(array)).sum::<usize>();
                }
            }
        }
        bytes
    }
}

fn array_bytes(array: &Array) -> usize {
    array.size().saturating_mul(array.dtype().byte_size())
}

/// What a stored artifact covers, for choosing between it and a cached one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DFlash2DiskEntry {
    pub(crate) key: String,
    pub(crate) cached_len: i32,
    pub(crate) bytes: u64,
}

/// The stored conversation a new artifact continues.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DFlash2DiskLineage {
    /// The artifact is already stored.
    pub(crate) stored: bool,
    pub(crate) parent: Option<String>,
    pub(crate) parent_len: i32,
    pub(crate) ancestors: Vec<String>,
}

/// An artifact read back, not yet written into a cache.
#[derive(Debug)]
pub(crate) struct DFlash2DiskLoaded {
    pub(crate) cached_len: i32,
    pub(crate) layers: Vec<DFlash2DiskLayer>,
    /// Per range file, K and V of every full-attention layer.
    pub(crate) kv: Vec<Vec<(Array, Array)>>,
    pub(crate) context_tail: Array,
    pub(crate) last_hidden: Array,
}

#[derive(Debug)]
pub(crate) struct DFlash2DiskRestored {
    pub(crate) cached_len: i32,
    pub(crate) tokens: usize,
    pub(crate) context_tail: Array,
    pub(crate) last_hidden: Array,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DFlash2DiskSnapshot {
    pub(crate) entries: usize,
    pub(crate) bytes: u64,
    pub(crate) hits: u64,
    pub(crate) saves: u64,
    /// Saves given up before writing (queue full, no memory for the copy).
    pub(crate) dropped: u64,
    /// Saves that failed while writing.
    pub(crate) failed: u64,
    /// Stored artifacts discarded as unusable.
    pub(crate) invalid: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RangeRef {
    file: String,
    start: i32,
    stop: i32,
    bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TensorMeta {
    name: String,
    dtype: String,
    shape: Vec<i32>,
}

impl TensorMeta {
    fn of(name: &str, array: &Array) -> Self {
        Self {
            name: name.to_owned(),
            dtype: format!("{:?}", array.dtype()),
            shape: array.shape().as_slice().to_vec(),
        }
    }

    fn matches(&self, array: &Array) -> bool {
        self.dtype == format!("{:?}", array.dtype()) && self.shape == array.shape().as_slice()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArtifactMeta {
    format_version: u32,
    token_ids: Vec<u32>,
    fingerprint: String,
    cached_len: i32,
    /// `dense`, or the kind of a state layer.
    layers: Vec<String>,
    kv_dtype: String,
    kv: Vec<RangeRef>,
    context_tail_len: i32,
    context: Vec<RangeRef>,
    state_bytes: u64,
    state_tensors: Vec<TensorMeta>,
}

impl ArtifactMeta {
    fn dense_layers(&self) -> usize {
        self.layers.iter().filter(|kind| *kind == "dense").count()
    }
}

#[derive(Debug)]
struct DiskArtifact {
    key: String,
    meta: ArtifactMeta,
    /// The artifact directory's files.
    own_bytes: u64,
    last_used: SystemTime,
    /// Restores reading it; eviction leaves it alone meanwhile.
    pins: usize,
}

/// A save whose files are written under temporary names.
#[derive(Debug)]
pub(crate) struct DFlash2WrittenSave {
    key: String,
    meta: ArtifactMeta,
    own_bytes: u64,
    ancestors: Vec<String>,
    /// `(temporary path, final path)`, the artifact directory last.
    installs: Vec<(PathBuf, PathBuf)>,
}

impl DFlash2WrittenSave {
    fn discard(&self) {
        for (tmp, _) in &self.installs {
            remove_path(tmp);
        }
    }
}

/// What a save needs from the index before it writes.
#[derive(Debug)]
pub(crate) struct DFlash2SavePlan {
    root: PathBuf,
    key: String,
    parent_kv: Vec<RangeRef>,
    parent_context: Vec<RangeRef>,
}

/// What a restore needs from the index before it reads.
#[derive(Debug)]
pub(crate) struct DFlash2RestorePlan {
    root: PathBuf,
    key: String,
    meta: ArtifactMeta,
}

impl DFlash2SavePlan {
    pub(crate) fn key(&self) -> &str {
        &self.key
    }
}

impl DFlash2RestorePlan {
    /// Every file the restore reads is present and whole.
    pub(crate) fn check(&self) -> Result<()> {
        check_files(&self.root, &self.meta)
    }

    /// Bytes the restore reads.
    pub(crate) fn payload_bytes(&self) -> u64 {
        self.meta.state_bytes
            + self
                .meta
                .kv
                .iter()
                .chain(&self.meta.context)
                .map(|range| range.bytes)
                .sum::<u64>()
    }

    pub(crate) fn tokens(&self) -> usize {
        self.meta.token_ids.len()
    }
}

/// The store of another identity in the same directory: it shares the byte
/// limit and goes first, whole, least recently opened first.
#[derive(Debug)]
struct OtherIdentity {
    path: PathBuf,
    bytes: u64,
    last_opened: SystemTime,
}

pub(crate) struct DFlash2DiskStore {
    root: PathBuf,
    max_bytes: Option<u64>,
    entries: Vec<DiskArtifact>,
    others: Vec<OtherIdentity>,
    hits: u64,
    saves: u64,
    dropped: u64,
    failed: u64,
    invalid: u64,
}

impl DFlash2DiskStore {
    /// Open `dir`'s store for `identity`: remove what an interrupted run left
    /// behind and index what is complete.
    pub(crate) fn open(
        dir: &Path,
        identity: &DFlash2StoreIdentity,
        max_bytes: Option<u64>,
    ) -> Result<Self> {
        let root = dir.join(DFLASH2_STORE_DIR).join(identity.key());
        for sub in [SEGMENTS_DIR, CONTEXT_DIR, ARTIFACTS_DIR] {
            fs::create_dir_all(root.join(sub))
                .with_context(|| format!("create DFlash2 prefix store {}", root.display()))?;
        }
        let identity_path = root.join(IDENTITY_FILE);
        if !identity_path.exists() {
            write_synced(&identity_path, &serde_json::to_vec_pretty(identity)?)?;
        }
        // When this identity was last opened.
        if let Ok(file) = fs::File::options().write(true).open(&identity_path) {
            let _ = file.set_modified(SystemTime::now());
        }
        let mut store = Self {
            others: other_identities(&dir.join(DFLASH2_STORE_DIR), &root),
            root,
            max_bytes,
            entries: Vec::new(),
            hits: 0,
            saves: 0,
            dropped: 0,
            failed: 0,
            invalid: 0,
        };
        store.scan()?;
        store.evict_to_limit(None);
        store.collect_ranges();
        Ok(store)
    }

    fn scan(&mut self) -> Result<()> {
        for sub in [SEGMENTS_DIR, CONTEXT_DIR, ARTIFACTS_DIR] {
            for entry in fs::read_dir(self.root.join(sub))? {
                let entry = entry?;
                if entry.file_name().to_string_lossy().contains(TMP_MARK) {
                    remove_path(&entry.path());
                }
            }
        }
        for entry in fs::read_dir(self.root.join(ARTIFACTS_DIR))? {
            let entry = entry?;
            let key = entry.file_name().to_string_lossy().into_owned();
            match self.read_artifact(&key) {
                Ok(artifact) => self.entries.push(artifact),
                Err(error) => {
                    tracing::warn!(key, error = %error, "unusable DFlash2 prefix artifact removed from disk");
                    remove_path(&entry.path());
                    self.invalid = self.invalid.saturating_add(1);
                }
            }
        }
        self.collect_ranges();
        Ok(())
    }

    fn read_artifact(&self, key: &str) -> Result<DiskArtifact> {
        let dir = self.root.join(ARTIFACTS_DIR).join(key);
        let meta_path = dir.join(META_FILE);
        let meta_bytes = fs::read(&meta_path)?;
        let meta: ArtifactMeta = serde_json::from_slice(&meta_bytes)?;
        anyhow::ensure!(
            meta.format_version == FORMAT_VERSION,
            "format version {}",
            meta.format_version
        );
        anyhow::ensure!(
            artifact_key(&meta.token_ids, &meta.fingerprint) == key,
            "key does not match its tokens"
        );
        check_files(&self.root, &meta)?;
        Ok(DiskArtifact {
            key: key.to_owned(),
            own_bytes: meta.state_bytes + meta_bytes.len() as u64,
            last_used: fs::metadata(&meta_path)?
                .modified()
                .unwrap_or(SystemTime::UNIX_EPOCH),
            meta,
            pins: 0,
        })
    }

    fn matching<'a>(
        &'a self,
        prompt_ids: &'a [u32],
        fingerprint: &'a dyn Fn(usize) -> String,
    ) -> impl Iterator<Item = &'a DiskArtifact> {
        self.entries.iter().filter(move |entry| {
            let tokens = &entry.meta.token_ids;
            tokens.len() <= prompt_ids.len()
                && prompt_ids.starts_with(tokens)
                && entry.meta.fingerprint == fingerprint(tokens.len())
        })
    }

    /// The longest stored artifact whose tokens prefix `prompt_ids` and whose
    /// fingerprint is `fingerprint(its token count)`.
    pub(crate) fn longest(
        &self,
        prompt_ids: &[u32],
        fingerprint: &dyn Fn(usize) -> String,
    ) -> Option<DFlash2DiskEntry> {
        self.matching(prompt_ids, fingerprint)
            .max_by_key(|entry| entry.meta.cached_len)
            .map(|entry| DFlash2DiskEntry {
                key: entry.key.clone(),
                cached_len: entry.meta.cached_len,
                bytes: self.artifact_bytes(entry),
            })
    }

    /// The stored artifacts an artifact at `token_ids` would continue.
    pub(crate) fn lineage(
        &self,
        token_ids: &[u32],
        fingerprint: &dyn Fn(usize) -> String,
    ) -> DFlash2DiskLineage {
        let mut lineage = DFlash2DiskLineage::default();
        for entry in self.matching(token_ids, fingerprint) {
            if entry.meta.token_ids.len() == token_ids.len() {
                lineage.stored = true;
                continue;
            }
            lineage.ancestors.push(entry.key.clone());
            if entry.meta.cached_len > lineage.parent_len {
                lineage.parent_len = entry.meta.cached_len;
                lineage.parent = Some(entry.key.clone());
            }
        }
        lineage
    }

    fn artifact_bytes(&self, entry: &DiskArtifact) -> u64 {
        entry.own_bytes
            + entry
                .meta
                .kv
                .iter()
                .chain(&entry.meta.context)
                .map(|range| range.bytes)
                .sum::<u64>()
    }

    /// Bytes on disk: every range file once, plus each artifact's own files.
    pub(crate) fn total_bytes(&self) -> u64 {
        let mut seen = HashSet::new();
        let mut bytes = 0_u64;
        for entry in &self.entries {
            bytes += entry.own_bytes;
            for (dir, ranges) in [
                (SEGMENTS_DIR, &entry.meta.kv),
                (CONTEXT_DIR, &entry.meta.context),
            ] {
                for range in ranges {
                    if seen.insert((dir, range.file.as_str())) {
                        bytes += range.bytes;
                    }
                }
            }
        }
        bytes
    }

    pub(crate) fn snapshot(&self) -> DFlash2DiskSnapshot {
        DFlash2DiskSnapshot {
            entries: self.entries.len(),
            bytes: self.total_bytes(),
            hits: self.hits,
            saves: self.saves,
            dropped: self.dropped,
            failed: self.failed,
            invalid: self.invalid,
        }
    }

    #[cfg(test)]
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn note_dropped(&mut self) {
        self.dropped = self.dropped.saturating_add(1);
    }

    pub(crate) fn note_failed(&mut self) {
        self.failed = self.failed.saturating_add(1);
    }

    /// Store one artifact. `Ok(false)` when it is already stored.
    #[cfg(test)]
    pub(crate) fn save(&mut self, save: &DFlash2DiskSave) -> Result<bool> {
        let Some(plan) = self.plan_save(save)? else {
            return Ok(false);
        };
        let written = write_save(&plan, save)?;
        self.commit_save(written)?;
        if let Some(compaction) = self.begin_compaction(plan.key()) {
            let written = write_compaction(&compaction);
            self.finish_compaction(plan.key(), written.ok())?;
        }
        Ok(true)
    }

    /// The index's part of a save, before its files are written. `None` when
    /// the artifact is already stored.
    pub(crate) fn plan_save(&self, save: &DFlash2DiskSave) -> Result<Option<DFlash2SavePlan>> {
        let key = artifact_key(&save.token_ids, &save.fingerprint);
        if self.entries.iter().any(|entry| entry.key == key) {
            return Ok(None);
        }
        let (parent_kv, parent_context) = match &save.parent {
            Some(parent) => {
                let parent = self
                    .entries
                    .iter()
                    .find(|entry| &entry.key == parent)
                    .ok_or_else(|| anyhow!("DFlash2 prefix store no longer holds the parent"))?;
                (parent.meta.kv.clone(), parent.meta.context.clone())
            }
            None => (Vec::new(), Vec::new()),
        };
        Ok(Some(DFlash2SavePlan {
            root: self.root.clone(),
            key,
            parent_kv,
            parent_context,
        }))
    }

    /// Install a written save and bring the store back within its limits.
    pub(crate) fn commit_save(&mut self, written: DFlash2WrittenSave) -> Result<()> {
        if self.entries.iter().any(|entry| entry.key == written.key) {
            written.discard();
            return Ok(());
        }
        if let Err(error) = self.install(&written) {
            self.collect_ranges();
            return Err(error);
        }
        let key = written.key.clone();
        self.entries.push(DiskArtifact {
            key: key.clone(),
            meta: written.meta,
            own_bytes: written.own_bytes,
            last_used: SystemTime::now(),
            pins: 0,
        });
        self.saves = self.saves.saturating_add(1);
        self.enforce_limits(&key, &written.ancestors)
    }

    /// Rename a written save's files into place. The parent's range files
    /// may have been removed while the save was written (its artifact was
    /// evicted, or a restore found it unusable), so every file the artifact
    /// references is checked before the index may point at it.
    fn install(&self, written: &DFlash2WrittenSave) -> Result<()> {
        let mut installed = Ok(());
        for (tmp, path) in &written.installs {
            if path.is_dir() {
                remove_path(path);
            }
            if let Err(error) = fs::rename(tmp, path) {
                installed = Err(error)
                    .with_context(|| format!("install DFlash2 prefix file {}", path.display()));
                break;
            }
        }
        let installed = installed.and_then(|()| check_files(&self.root, &written.meta));
        if installed.is_err() {
            written.discard();
            remove_path(&self.root.join(ARTIFACTS_DIR).join(&written.key));
        }
        installed
    }

    /// Drop what the saved artifact's conversation no longer needs, then
    /// whatever is beyond the byte limit.
    fn enforce_limits(&mut self, saved: &str, ancestors: &[String]) -> Result<()> {
        // An artifact a restore is reading stays until the restore ends.
        let mut boundaries: Vec<(usize, &str)> = self
            .entries
            .iter()
            .filter(|entry| {
                entry.key == saved || (ancestors.contains(&entry.key) && entry.pins == 0)
            })
            .map(|entry| (entry.meta.token_ids.len(), entry.key.as_str()))
            .collect();
        let superseded: Vec<String> = superseded_boundaries(&mut boundaries)
            .into_iter()
            .map(str::to_owned)
            .collect();
        for key in &superseded {
            self.remove_entry(key);
        }
        let fits = self.evict_to_limit(Some(saved));
        if !fits {
            self.remove_entry(saved);
        }
        self.collect_ranges();
        anyhow::ensure!(
            fits,
            "DFlash2 prefix artifact does not fit the disk limit of {:?} bytes",
            self.max_bytes
        );
        Ok(())
    }

    /// Bring the directory within the byte limit: other identities' stores
    /// first (least recently opened first), then this one's least recently
    /// used artifacts, never `keep` or one a restore is reading. `false` when
    /// that is not enough.
    fn evict_to_limit(&mut self, keep: Option<&str>) -> bool {
        let Some(max_bytes) = self.max_bytes else {
            return true;
        };
        loop {
            let others: u64 = self.others.iter().map(|other| other.bytes).sum();
            if self.total_bytes().saturating_add(others) <= max_bytes {
                return true;
            }
            if let Some(index) =
                (0..self.others.len()).min_by_key(|index| self.others[*index].last_opened)
            {
                let other = self.others.swap_remove(index);
                tracing::info!(
                    path = %other.path.display(),
                    bytes = other.bytes,
                    "DFlash2 prefix store of another configuration removed for the disk limit"
                );
                remove_path(&other.path);
                continue;
            }
            let victim = self
                .entries
                .iter()
                .filter(|entry| Some(entry.key.as_str()) != keep && entry.pins == 0)
                .min_by_key(|entry| entry.last_used)
                .map(|entry| entry.key.clone());
            match victim {
                Some(key) => self.remove_entry(&key),
                None => return false,
            }
        }
    }

    fn remove_entry(&mut self, key: &str) {
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            self.entries.swap_remove(index);
            remove_path(&self.root.join(ARTIFACTS_DIR).join(key));
        }
    }

    /// Remove a stored artifact and the range files only it referenced.
    pub(crate) fn remove(&mut self, key: &str) {
        self.remove_entry(key);
        self.collect_ranges();
    }

    /// Delete range files no artifact references. Files being written have
    /// temporary names and are left alone.
    fn collect_ranges(&mut self) {
        for (dir, select) in [
            (
                SEGMENTS_DIR,
                (|meta: &ArtifactMeta| &meta.kv) as fn(&ArtifactMeta) -> &Vec<RangeRef>,
            ),
            (CONTEXT_DIR, |meta: &ArtifactMeta| &meta.context),
        ] {
            let referenced: HashSet<&str> = self
                .entries
                .iter()
                .flat_map(|entry| select(&entry.meta))
                .map(|range| range.file.as_str())
                .collect();
            let Ok(files) = fs::read_dir(self.root.join(dir)) else {
                continue;
            };
            for file in files.flatten() {
                let name = file.file_name().to_string_lossy().into_owned();
                if !name.contains(TMP_MARK) && !referenced.contains(name.as_str()) {
                    remove_path(&file.path());
                }
            }
        }
    }

    /// The index's part of a compaction: `Some` when the artifact references
    /// more range files than a restore should open. The artifact stays on
    /// disk until [`Self::finish_compaction`].
    pub(crate) fn begin_compaction(&mut self, key: &str) -> Option<DFlash2RestorePlan> {
        let entry = self.entries.iter_mut().find(|entry| entry.key == key)?;
        if entry.meta.kv.len() + entry.meta.context.len() <= MAX_RANGE_FILES {
            return None;
        }
        entry.pins += 1;
        Some(DFlash2RestorePlan {
            root: self.root.clone(),
            key: key.to_owned(),
            meta: entry.meta.clone(),
        })
    }

    /// End a compaction: point the artifact at the files `written` holds, or
    /// leave it as it was when there are none.
    pub(crate) fn finish_compaction(
        &mut self,
        key: &str,
        written: Option<DFlash2WrittenSave>,
    ) -> Result<()> {
        let index = self.entries.iter().position(|entry| entry.key == key);
        if let Some(index) = index {
            self.entries[index].pins = self.entries[index].pins.saturating_sub(1);
        }
        let Some(written) = written else {
            return Ok(());
        };
        let Some(index) = index else {
            written.discard();
            return Ok(());
        };
        if let Err(error) = self.install(&written) {
            // The artifact directory was replaced and then removed.
            self.entries.swap_remove(index);
            self.collect_ranges();
            return Err(error);
        }
        let entry = &mut self.entries[index];
        entry.meta = written.meta;
        entry.own_bytes = written.own_bytes;
        self.collect_ranges();
        Ok(())
    }

    /// The index's part of a restore: the artifact stays on disk until
    /// [`Self::finish_restore`].
    pub(crate) fn begin_restore(&mut self, key: &str) -> Option<DFlash2RestorePlan> {
        let entry = self.entries.iter_mut().find(|entry| entry.key == key)?;
        entry.pins += 1;
        Some(DFlash2RestorePlan {
            root: self.root.clone(),
            key: key.to_owned(),
            meta: entry.meta.clone(),
        })
    }

    /// End a restore: a hit refreshes the artifact's last use, a failure
    /// removes it.
    pub(crate) fn finish_restore(&mut self, key: &str, restored: bool) {
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.key == key) {
            entry.pins = entry.pins.saturating_sub(1);
        }
        if restored {
            self.hits = self.hits.saturating_add(1);
            if let Some(entry) = self.entries.iter_mut().find(|entry| entry.key == key) {
                entry.last_used = SystemTime::now();
                let meta_path = self.root.join(ARTIFACTS_DIR).join(key).join(META_FILE);
                if let Ok(file) = fs::File::options().write(true).open(meta_path) {
                    let _ = file.set_modified(entry.last_used);
                }
            }
        } else {
            self.invalid = self.invalid.saturating_add(1);
            self.remove(key);
        }
    }

    /// Read a stored artifact into row 0 of a fresh target cache.
    /// `make_room` is told how many bytes the read brings into memory. A
    /// failure removes the artifact; the caller then needs a fresh cache.
    #[cfg(test)]
    pub(crate) fn restore_into(
        &mut self,
        key: &str,
        target_cache: &mut [LayerCache],
        make_room: &mut dyn FnMut(usize),
    ) -> Result<DFlash2DiskRestored> {
        let plan = self
            .begin_restore(key)
            .ok_or_else(|| anyhow!("DFlash2 prefix store does not hold artifact {key}"))?;
        make_room(usize::try_from(plan.payload_bytes()).unwrap_or(usize::MAX));
        let restored = restore_planned(&plan, target_cache);
        self.finish_restore(key, restored.is_ok());
        restored
    }
}

/// Rewrite `plan`'s artifact as one K/V range file and one context range
/// file, under temporary names. Reads and evaluates the whole artifact, so
/// it runs without the store's lock.
pub(crate) fn write_compaction(plan: &DFlash2RestorePlan) -> Result<DFlash2WrittenSave> {
    let meta = &plan.meta;
    let loaded = load_artifact(plan)?;
    let mut layers = Vec::with_capacity(meta.dense_layers());
    for dense in 0..meta.dense_layers() {
        let keys: Vec<&Array> = loaded.kv.iter().map(|range| &range[dense].0).collect();
        let values: Vec<&Array> = loaded.kv.iter().map(|range| &range[dense].1).collect();
        layers.push((
            mlx::ops::shape::concatenate(&keys, 2)?,
            mlx::ops::shape::concatenate(&values, 2)?,
        ));
    }
    let save = DFlash2DiskSave {
        token_ids: meta.token_ids.clone(),
        fingerprint: meta.fingerprint.clone(),
        layers: loaded.layers,
        parent: None,
        ancestors: Vec::new(),
        kv: vec![DFlash2DiskRange {
            start: 0,
            stop: meta.cached_len,
            layers,
        }],
        context_start: meta.cached_len - meta.context_tail_len,
        context: loaded.context_tail,
        context_tail_len: meta.context_tail_len,
        last_hidden: loaded.last_hidden,
    };
    let save_plan = DFlash2SavePlan {
        root: plan.root.clone(),
        key: plan.key.clone(),
        parent_kv: Vec::new(),
        parent_context: Vec::new(),
    };
    write_save(&save_plan, &save)
}

/// Read `plan`'s artifact and write it into row 0 of a fresh target cache.
pub(crate) fn restore_planned(
    plan: &DFlash2RestorePlan,
    target_cache: &mut [LayerCache],
) -> Result<DFlash2DiskRestored> {
    let loaded = load_artifact(plan)?;
    anyhow::ensure!(
        target_cache.len() == loaded.layers.len(),
        "stored DFlash2 prefix artifact has {} layers, target cache has {}",
        loaded.layers.len(),
        target_cache.len()
    );
    let mut dense = 0_usize;
    for (index, (cache, layer)) in target_cache.iter_mut().zip(&loaded.layers).enumerate() {
        match layer {
            DFlash2DiskLayer::Dense => {
                let LayerCache::Full(kv) = cache else {
                    anyhow::bail!("stored DFlash2 prefix layer {index} expects dense K/V");
                };
                let segments: Vec<(&Array, &Array)> = loaded
                    .kv
                    .iter()
                    .map(|range| (&range[dense].0, &range[dense].1))
                    .collect();
                kv.restore_dense_prefix_segments_for_row_on(&segments, 0, loaded.cached_len, ())?;
                dense += 1;
            }
            DFlash2DiskLayer::State(payload) => {
                let entry = PagedPrefixEntry {
                    main_layers: vec![payload.clone()],
                    ..PagedPrefixEntry::default()
                };
                restore_prefix_entry_for_row(
                    std::slice::from_mut(cache),
                    &entry,
                    0,
                    loaded.cached_len,
                )?;
            }
        }
    }
    // Reads the files: a load only maps them.
    let (entry, cached_len) = prefix_entry_for_row(target_cache, 0)?
        .ok_or_else(|| anyhow!("restored DFlash2 target cache is empty"))?;
    anyhow::ensure!(
        cached_len == loaded.cached_len,
        "restored DFlash2 target offset {cached_len} != stored offset {}",
        loaded.cached_len
    );
    entry.eval()?;
    mlx::transforms::eval(&[&loaded.context_tail, &loaded.last_hidden])?;
    Ok(DFlash2DiskRestored {
        cached_len: loaded.cached_len,
        tokens: plan.tokens(),
        context_tail: loaded.context_tail,
        last_hidden: loaded.last_hidden,
    })
}

/// Every file `meta` references exists with the size recorded for it.
fn check_files(root: &Path, meta: &ArtifactMeta) -> Result<()> {
    let key = artifact_key(&meta.token_ids, &meta.fingerprint);
    let state = root.join(ARTIFACTS_DIR).join(key).join(STATE_FILE);
    let mut files = vec![(state, meta.state_bytes)];
    for (dir, ranges) in [(SEGMENTS_DIR, &meta.kv), (CONTEXT_DIR, &meta.context)] {
        for range in ranges {
            files.push((root.join(dir).join(&range.file), range.bytes));
        }
    }
    for (path, bytes) in files {
        let len = fs::metadata(&path)
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        anyhow::ensure!(
            len == bytes,
            "{} holds {len} bytes, expected {bytes}",
            path.display()
        );
    }
    Ok(())
}

/// The stores of other identities under `store_dir`. One not opened for
/// [`STALE_IDENTITY_AFTER`] is removed here.
fn other_identities(store_dir: &Path, own: &Path) -> Vec<OtherIdentity> {
    let mut others = Vec::new();
    let Ok(entries) = fs::read_dir(store_dir) else {
        return others;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == own || !path.is_dir() {
            continue;
        }
        let last_opened = fs::metadata(path.join(IDENTITY_FILE))
            .and_then(|meta| meta.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let stale = last_opened
            .elapsed()
            .is_ok_and(|unused| unused > STALE_IDENTITY_AFTER);
        if stale {
            tracing::info!(path = %path.display(), "unused DFlash2 prefix store removed");
            remove_path(&path);
            continue;
        }
        others.push(OtherIdentity {
            bytes: directory_bytes(&path),
            path,
            last_opened,
        });
    }
    others
}

fn directory_bytes(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.metadata() {
            Ok(meta) if meta.is_dir() => directory_bytes(&entry.path()),
            Ok(meta) => meta.len(),
            Err(_) => 0,
        })
        .sum()
}

fn load_tensors(path: &Path) -> Result<HashMap<String, Array>> {
    let (tensors, _) = mlx::io::load_safetensors(&path.to_string_lossy())
        .with_context(|| format!("load {}", path.display()))?;
    Ok(tensors)
}

fn take(tensors: &mut HashMap<String, Array>, name: &str, path: &Path) -> Result<Array> {
    tensors
        .remove(name)
        .ok_or_else(|| anyhow!("{} lacks tensor {name}", path.display()))
}

fn load_artifact(plan: &DFlash2RestorePlan) -> Result<DFlash2DiskLoaded> {
    let meta = &plan.meta;
    let root = &plan.root;
    check_files(root, meta)?;
    let dense_layers = meta.dense_layers();

    let mut kv = Vec::with_capacity(meta.kv.len());
    let mut position = 0_i32;
    for range in &meta.kv {
        anyhow::ensure!(
            range.start == position && range.stop > range.start,
            "stored K/V ranges are not consecutive at {position}"
        );
        let path = root.join(SEGMENTS_DIR).join(&range.file);
        let mut tensors = load_tensors(&path)?;
        let mut layers = Vec::with_capacity(dense_layers);
        for dense in 0..dense_layers {
            let k = take(&mut tensors, &kv_name('k', dense), &path)?;
            let v = take(&mut tensors, &kv_name('v', dense), &path)?;
            for array in [&k, &v] {
                let shape = array.shape();
                let dims = shape.as_slice();
                anyhow::ensure!(
                    dims.len() == 4
                        && dims[2] == range.stop - range.start
                        && format!("{:?}", array.dtype()) == meta.kv_dtype,
                    "{} holds K/V of shape {dims:?}",
                    path.display()
                );
            }
            layers.push((k, v));
        }
        anyhow::ensure!(
            tensors.is_empty(),
            "{} holds unexpected tensors",
            path.display()
        );
        kv.push(layers);
        position = range.stop;
    }
    anyhow::ensure!(
        position == meta.cached_len,
        "stored K/V covers {position} of {} positions",
        meta.cached_len
    );

    let tail_start = meta.cached_len - meta.context_tail_len;
    let mut parts = Vec::new();
    let mut position = tail_start;
    for range in &meta.context {
        if range.stop <= tail_start {
            continue;
        }
        anyhow::ensure!(
            range.start <= position && position < range.stop,
            "stored context ranges are not consecutive at {position}"
        );
        let path = root.join(CONTEXT_DIR).join(&range.file);
        let mut tensors = load_tensors(&path)?;
        let hidden = take(&mut tensors, CONTEXT_TENSOR, &path)?;
        let shape = hidden.shape();
        let dims = shape.as_slice();
        anyhow::ensure!(
            dims.len() == 3 && dims[1] == range.stop - range.start,
            "{} holds context of shape {dims:?}",
            path.display()
        );
        parts.push(if position == range.start {
            hidden
        } else {
            mlx::ops::indexing::slice_strided_on(
                &hidden,
                &[0_i32, position - range.start, 0][..],
                &[1_i32, dims[1], dims[2]][..],
                &[1_i32, 1, 1][..],
                StreamOrDevice::default(),
            )?
        });
        position = range.stop;
    }
    anyhow::ensure!(
        position == meta.cached_len && !parts.is_empty(),
        "stored context covers up to {position} of {} positions",
        meta.cached_len
    );
    let context_tail = match parts.as_slice() {
        [only] => only.clone(),
        _ => mlx::ops::shape::concatenate(&parts.iter().collect::<Vec<_>>(), 1)?,
    };

    let state_path = root.join(ARTIFACTS_DIR).join(&plan.key).join(STATE_FILE);
    let mut tensors = load_tensors(&state_path)?;
    for tensor in &meta.state_tensors {
        let array = tensors
            .get(&tensor.name)
            .ok_or_else(|| anyhow!("{} lacks tensor {}", state_path.display(), tensor.name))?;
        anyhow::ensure!(
            tensor.matches(array),
            "{} holds {} with another type or shape",
            state_path.display(),
            tensor.name
        );
    }
    anyhow::ensure!(
        tensors.len() == meta.state_tensors.len(),
        "{} holds unexpected tensors",
        state_path.display()
    );
    let last_hidden = take(&mut tensors, LAST_HIDDEN, &state_path)?;
    let mut layers = Vec::with_capacity(meta.layers.len());
    for (index, kind) in meta.layers.iter().enumerate() {
        layers.push(if kind == "dense" {
            DFlash2DiskLayer::Dense
        } else {
            let mut arrays = Vec::new();
            for slot in 0..state_slots(kind)? {
                arrays.push(take(&mut tensors, &state_name(index, slot), &state_path)?);
            }
            DFlash2DiskLayer::State(state_payload(kind, arrays)?)
        });
    }
    Ok(DFlash2DiskLoaded {
        cached_len: meta.cached_len,
        layers,
        kv,
        context_tail,
        last_hidden,
    })
}

fn kv_name(side: char, dense: usize) -> String {
    format!("{side}{dense:04}")
}

fn state_name(layer: usize, slot: usize) -> String {
    format!("state_{layer:04}_{slot}")
}

/// The kind and arrays of a state layer the store can hold.
fn state_arrays(payload: &PrefixLayerPayload) -> Result<(&'static str, Vec<&Array>)> {
    match payload {
        PrefixLayerPayload::Linear {
            conv_state,
            recurrent_state,
        } => Ok(("linear", vec![conv_state, recurrent_state])),
        PrefixLayerPayload::Mla { c_kv, k_pe } => Ok(("mla", vec![c_kv, k_pe])),
        _ => Err(anyhow!(
            "DFlash2 prefix store cannot hold this layer's state"
        )),
    }
}

fn state_slots(kind: &str) -> Result<usize> {
    match kind {
        "linear" | "mla" => Ok(2),
        _ => Err(anyhow!("unknown stored state layer kind {kind}")),
    }
}

fn state_payload(kind: &str, mut arrays: Vec<Array>) -> Result<PrefixLayerPayload> {
    let second = arrays.pop();
    let first = arrays.pop();
    match (kind, first, second) {
        ("linear", Some(conv_state), Some(recurrent_state)) => Ok(PrefixLayerPayload::Linear {
            conv_state,
            recurrent_state,
        }),
        ("mla", Some(c_kv), Some(k_pe)) => Ok(PrefixLayerPayload::Mla { c_kv, k_pe }),
        _ => Err(anyhow!("unknown stored state layer kind {kind}")),
    }
}

fn remove_path(path: &Path) {
    let result = if path.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    if let Err(error) = result {
        if error.kind() != ErrorKind::NotFound {
            tracing::warn!(path = %path.display(), error = %error, "DFlash2 prefix store could not remove a file");
        }
    }
}

fn tmp_name(name: &str) -> String {
    format!("{TMP_MARK}{}-{name}", uuid::Uuid::new_v4().simple())
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    fs::File::open(path)?.sync_all()?;
    Ok(())
}

/// Write `tensors` to `path` (which ends in `.safetensors`) and sync it;
/// returns the file's size.
fn write_tensors(path: &Path, tensors: &HashMap<String, Array>) -> Result<u64> {
    mlx::io::save_safetensors(&path.to_string_lossy(), tensors, &HashMap::new())
        .with_context(|| format!("write {}", path.display()))?;
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    file.sync_all()?;
    Ok(file.metadata()?.len())
}

/// Write a save's files under temporary names. Touches no index state, so it
/// runs without the store's lock.
pub(crate) fn write_save(
    plan: &DFlash2SavePlan,
    save: &DFlash2DiskSave,
) -> Result<DFlash2WrittenSave> {
    let mut installs = Vec::new();
    let result = write_save_files(plan, save, &mut installs);
    if result.is_err() {
        for (tmp, _) in &installs {
            remove_path(tmp);
        }
    }
    let (meta, own_bytes) = result?;
    Ok(DFlash2WrittenSave {
        key: plan.key.clone(),
        meta,
        own_bytes,
        ancestors: save.ancestors.clone(),
        installs,
    })
}

fn write_save_files(
    plan: &DFlash2SavePlan,
    save: &DFlash2DiskSave,
    installs: &mut Vec<(PathBuf, PathBuf)>,
) -> Result<(ArtifactMeta, u64)> {
    let root = &plan.root;
    let cached_len = i32::try_from(save.token_ids.len())?;
    let dense_layers = save
        .layers
        .iter()
        .filter(|layer| matches!(layer, DFlash2DiskLayer::Dense))
        .count();

    let mut kv = plan.parent_kv.clone();
    let mut position = kv.last().map_or(0, |range| range.stop);
    let mut kv_dtype = None;
    for range in &save.kv {
        anyhow::ensure!(
            range.start == position
                && range.stop > range.start
                && range.layers.len() == dense_layers,
            "DFlash2 prefix save K/V range {}..{} does not continue {position}",
            range.start,
            range.stop
        );
        let mut tensors = HashMap::new();
        for (dense, (k, v)) in range.layers.iter().enumerate() {
            kv_dtype.get_or_insert_with(|| format!("{:?}", k.dtype()));
            tensors.insert(kv_name('k', dense), k.clone());
            tensors.insert(kv_name('v', dense), v.clone());
        }
        let file = range_file_name(&save.token_ids, &save.fingerprint, range.start, range.stop);
        let dir = root.join(SEGMENTS_DIR);
        let tmp = dir.join(tmp_name(&file));
        installs.push((tmp.clone(), dir.join(&file)));
        let bytes = write_tensors(&tmp, &tensors)?;
        kv.push(RangeRef {
            file,
            start: range.start,
            stop: range.stop,
            bytes,
        });
        position = range.stop;
    }
    anyhow::ensure!(
        position == cached_len,
        "DFlash2 prefix save K/V covers {position} of {cached_len} positions"
    );

    let tail_start = cached_len - save.context_tail_len;
    let mut context: Vec<RangeRef> = plan
        .parent_context
        .iter()
        .filter(|range| range.stop > tail_start && range.start < save.context_start)
        .cloned()
        .collect();
    let covered = context.last().map_or(tail_start, |range| range.stop);
    let context_len = save.context.shape().as_slice().get(1).copied().unwrap_or(0);
    anyhow::ensure!(
        0 < save.context_tail_len
            && save.context_tail_len <= cached_len
            && save.context_start >= tail_start
            && covered >= save.context_start
            && save.context_start + context_len == cached_len
            && context
                .first()
                .is_none_or(|range| range.start <= tail_start),
        "DFlash2 prefix save context {}+{context_len} does not complete the tail from {tail_start}",
        save.context_start
    );
    if context_len > 0 {
        let file = range_file_name(
            &save.token_ids,
            &save.fingerprint,
            save.context_start,
            cached_len,
        );
        let dir = root.join(CONTEXT_DIR);
        let tmp = dir.join(tmp_name(&file));
        installs.push((tmp.clone(), dir.join(&file)));
        let tensors = HashMap::from([(CONTEXT_TENSOR.to_owned(), save.context.clone())]);
        let bytes = write_tensors(&tmp, &tensors)?;
        context.push(RangeRef {
            file,
            start: save.context_start,
            stop: cached_len,
            bytes,
        });
    }

    let artifacts = root.join(ARTIFACTS_DIR);
    let tmp_dir = artifacts.join(tmp_name(&plan.key));
    fs::create_dir_all(&tmp_dir).with_context(|| format!("create {}", tmp_dir.display()))?;
    installs.push((tmp_dir.clone(), artifacts.join(&plan.key)));
    let mut tensors = HashMap::from([(LAST_HIDDEN.to_owned(), save.last_hidden.clone())]);
    let mut state_tensors = vec![TensorMeta::of(LAST_HIDDEN, &save.last_hidden)];
    let mut layers = Vec::with_capacity(save.layers.len());
    for (index, layer) in save.layers.iter().enumerate() {
        match layer {
            DFlash2DiskLayer::Dense => layers.push("dense".to_owned()),
            DFlash2DiskLayer::State(payload) => {
                let (kind, arrays) = state_arrays(payload)?;
                layers.push(kind.to_owned());
                for (slot, array) in arrays.into_iter().enumerate() {
                    let name = state_name(index, slot);
                    state_tensors.push(TensorMeta::of(&name, array));
                    tensors.insert(name, array.clone());
                }
            }
        }
    }
    let state_bytes = write_tensors(&tmp_dir.join(STATE_FILE), &tensors)?;
    let meta = ArtifactMeta {
        format_version: FORMAT_VERSION,
        token_ids: save.token_ids.clone(),
        fingerprint: save.fingerprint.clone(),
        cached_len,
        layers,
        kv_dtype: kv_dtype.unwrap_or_default(),
        kv,
        context_tail_len: save.context_tail_len,
        context,
        state_bytes,
        state_tensors,
    };
    let meta_bytes = serde_json::to_vec(&meta)?;
    write_synced(&tmp_dir.join(META_FILE), &meta_bytes)?;
    Ok((meta, state_bytes + meta_bytes.len() as u64))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use ironmlx_lm::core::cache::kv_cache::KVCache;

    pub(crate) fn temp_root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("ironmlx-{name}-{}", uuid::Uuid::new_v4().simple()));
        fs::create_dir_all(&root).expect("temp root");
        root
    }

    pub(crate) fn test_identity() -> DFlash2StoreIdentity {
        DFlash2StoreIdentity {
            format_version: FORMAT_VERSION,
            target_model_digest: "model".to_owned(),
            target_execution_fingerprint: "runtime".to_owned(),
            execution_profile: "build".to_owned(),
            prefill_chunk_size: 2048,
            target_layer_ids: vec![1, 5],
            context_limit: 24,
        }
    }

    const TAIL: i32 = 24;

    fn positions(start: i32, stop: i32, heads: i32, width: i32) -> Array {
        let mut data = Vec::new();
        for _head in 0..heads {
            for position in start..stop {
                data.extend(std::iter::repeat_n(position as f32, width as usize));
            }
        }
        (data.as_slice(), (1_i32, heads, stop - start, width))
            .try_into()
            .expect("kv")
    }

    fn context_positions(start: i32, stop: i32) -> Array {
        let data: Vec<f32> = (start..stop).flat_map(|p| [p as f32; 2]).collect();
        (data.as_slice(), (1_i32, stop - start, 2_i32))
            .try_into()
            .expect("context")
    }

    fn state_layer(seed: f32) -> PrefixLayerPayload {
        let conv: Array = ([seed, seed + 1.0].as_slice(), (1_i32, 2_i32))
            .try_into()
            .expect("conv");
        let recurrent: Array = ([seed + 2.0; 3].as_slice(), (1_i32, 3_i32))
            .try_into()
            .expect("recurrent");
        PrefixLayerPayload::Linear {
            conv_state: conv,
            recurrent_state: recurrent,
        }
    }

    /// A boundary at `tokens`, extending the store's longest stored prefix.
    /// K/V and context position `p` hold the value `p`.
    pub(crate) fn test_save(
        store: &DFlash2DiskStore,
        tokens: &[u32],
        with_state: bool,
    ) -> DFlash2DiskSave {
        let len = i32::try_from(tokens.len()).expect("len");
        let lineage = store.lineage(tokens, &|_| "runtime".to_owned());
        let tail = TAIL.min(len);
        let context_start = lineage.parent_len.max(len - tail);
        let mut layers = vec![DFlash2DiskLayer::Dense];
        if with_state {
            layers.push(DFlash2DiskLayer::State(state_layer(len as f32)));
        }
        layers.push(DFlash2DiskLayer::Dense);
        DFlash2DiskSave {
            token_ids: tokens.to_vec(),
            fingerprint: "runtime".to_owned(),
            layers,
            parent: lineage.parent,
            ancestors: lineage.ancestors,
            kv: vec![DFlash2DiskRange {
                start: lineage.parent_len,
                stop: len,
                layers: (0..2)
                    .map(|_| {
                        (
                            positions(lineage.parent_len, len, 2, 4),
                            positions(lineage.parent_len, len, 2, 4),
                        )
                    })
                    .collect(),
            }],
            context_start,
            context: context_positions(context_start, len),
            context_tail_len: tail,
            last_hidden: context_positions(len - 1, len),
        }
    }

    fn save_prefix(store: &mut DFlash2DiskStore, first: u32, len: u32) -> String {
        let tokens: Vec<u32> = (first..first + len).collect();
        let save = test_save(store, &tokens, true);
        store.save(&save).expect("save");
        artifact_key(&tokens, "runtime")
    }

    fn load(store: &mut DFlash2DiskStore, key: &str) -> DFlash2DiskLoaded {
        let plan = store.begin_restore(key).expect("stored");
        let loaded = load_artifact(&plan);
        store.finish_restore(key, loaded.is_ok());
        loaded.expect("load")
    }

    fn values(array: &Array) -> Vec<f32> {
        array.to_vec::<f32>().expect("values")
    }

    /// The first head's key value at every stored position of dense layer 0.
    fn kv_positions(loaded: &DFlash2DiskLoaded) -> Vec<f32> {
        loaded
            .kv
            .iter()
            .flat_map(|range| {
                let len = range[0].0.shape().as_slice()[2] as usize;
                values(&range[0].0)
                    .chunks(4)
                    .take(len)
                    .map(|position| position[0])
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn context_values(loaded: &DFlash2DiskLoaded) -> Vec<f32> {
        values(&loaded.context_tail)
            .chunks(2)
            .map(|position| position[0])
            .collect()
    }

    fn range(start: i32, stop: i32) -> Vec<f32> {
        (start..stop).map(|position| position as f32).collect()
    }

    fn dir_bytes(path: &Path) -> u64 {
        let mut bytes = 0;
        for entry in fs::read_dir(path).expect("dir").flatten() {
            let meta = entry.metadata().expect("meta");
            bytes += if meta.is_dir() {
                dir_bytes(&entry.path())
            } else {
                meta.len()
            };
        }
        bytes
    }

    pub(crate) fn file_names(path: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(path)
            .expect("dir")
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_round_trips_an_artifact_bit_for_bit() {
        let dir = temp_root("dflash2-store-round-trip");
        let key = {
            let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
            save_prefix(&mut store, 0, 40)
        };
        let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("reopen");
        assert_eq!(store.snapshot().entries, 1);
        let loaded = load(&mut store, &key);
        assert_eq!(loaded.cached_len, 40);
        assert_eq!(kv_positions(&loaded), range(0, 40));
        assert_eq!(values(&loaded.kv[0][1].1), values(&positions(0, 40, 2, 4)));
        assert_eq!(context_values(&loaded), range(16, 40));
        assert_eq!(values(&loaded.last_hidden), vec![39.0, 39.0]);
        let DFlash2DiskLayer::State(PrefixLayerPayload::Linear {
            conv_state,
            recurrent_state,
        }) = &loaded.layers[1]
        else {
            panic!("state layer");
        };
        assert_eq!(values(conv_state), vec![40.0, 41.0]);
        assert_eq!(values(recurrent_state), vec![42.0; 3]);
        assert_eq!(store.snapshot().hits, 1);
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_restores_dense_layers_into_a_target_cache() {
        let dir = temp_root("dflash2-store-restore");
        let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
        let tokens: Vec<u32> = (0..40).collect();
        store
            .save(&test_save(&store, &tokens, false))
            .expect("save");
        let tokens: Vec<u32> = (0..50).collect();
        store
            .save(&test_save(&store, &tokens, false))
            .expect("save");
        let key = artifact_key(&tokens, "runtime");
        let mut target_cache: Vec<LayerCache> = (0..2)
            .map(|_| {
                LayerCache::Full(
                    KVCache::new(1, 2, 4, 4, mlx::Dtype::Float32, 1 << 20).with_step(256),
                )
            })
            .collect();
        let mut asked = 0_usize;
        let restored = store
            .restore_into(&key, &mut target_cache, &mut |bytes| asked = bytes)
            .expect("restore");
        assert!(asked > 0);
        assert_eq!((restored.cached_len, restored.tokens), (50, 50));
        let (entry, cached_len) = prefix_entry_for_row(&target_cache, 0)
            .expect("entry")
            .expect("state");
        assert_eq!(cached_len, 50);
        let PrefixLayerPayload::FullDense { k, .. } = &entry.main_layers[1] else {
            panic!("dense layer");
        };
        let restored_positions: Vec<f32> = values(k)
            .chunks(4)
            .take(50)
            .map(|position| position[0])
            .collect();
        assert_eq!(restored_positions, range(0, 50));
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_shares_ranges_between_artifacts_of_a_conversation() {
        let dir = temp_root("dflash2-store-share");
        let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
        save_prefix(&mut store, 0, 40);
        let segments = store.root.join(SEGMENTS_DIR);
        let first = dir_bytes(&segments);
        let key = save_prefix(&mut store, 0, 50);
        assert_eq!(file_names(&segments).len(), 2);
        // The second file holds 10 of the 50 positions, not all of them.
        assert!(dir_bytes(&segments) < first * 2);
        let loaded = load(&mut store, &key);
        assert_eq!(kv_positions(&loaded), range(0, 50));
        assert_eq!(context_values(&loaded), range(26, 50));
        // Saving it again writes nothing.
        let tokens: Vec<u32> = (0..50).collect();
        assert!(!store.save(&test_save(&store, &tokens, true)).expect("save"));
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_longest_matches_prefix_and_fingerprint() {
        let dir = temp_root("dflash2-store-longest");
        let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
        save_prefix(&mut store, 0, 30);
        save_prefix(&mut store, 100, 60);
        let prompt: Vec<u32> = (0..45).collect();
        let hit = store
            .longest(&prompt, &|_| "runtime".to_owned())
            .expect("prefix");
        assert_eq!(hit.cached_len, 30);
        assert!(store.longest(&prompt, &|_| "other".to_owned()).is_none());
        assert!(store
            .longest(&prompt[..20], &|_| "runtime".to_owned())
            .is_none());
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_open_discards_truncated_and_orphaned_files() {
        let dir = temp_root("dflash2-store-scan");
        let (root, broken, intact) = {
            let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
            let broken = save_prefix(&mut store, 0, 40);
            let intact = save_prefix(&mut store, 100, 40);
            (store.root.clone(), broken, intact)
        };
        let segments = root.join(SEGMENTS_DIR);
        let truncated = file_names(&segments)
            .into_iter()
            .find(|name| name.starts_with("00000000-00000040"))
            .expect("segment");
        // Both conversations start at position 0; the broken one is whichever
        // artifact references the file truncated here.
        let truncated_path = segments.join(&truncated);
        let bytes = fs::read(&truncated_path).expect("read");
        fs::write(&truncated_path, &bytes[..bytes.len() / 2]).expect("truncate");
        fs::write(segments.join(format!("{TMP_MARK}x-left.safetensors")), b"x").expect("tmp");
        fs::write(segments.join("00000000-00000009-orphan.safetensors"), b"x").expect("orphan");

        let store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("reopen");
        let snapshot = store.snapshot();
        assert_eq!((snapshot.entries, snapshot.invalid), (1, 1));
        assert_eq!(file_names(&segments).len(), 1);
        assert!(!truncated_path.exists());
        assert_eq!(file_names(&root.join(ARTIFACTS_DIR)).len(), 1);
        let _ = (broken, intact);
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_ignores_another_identity() {
        let dir = temp_root("dflash2-store-identity");
        let root = {
            let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
            save_prefix(&mut store, 0, 40);
            store.root.clone()
        };
        let other = DFlash2StoreIdentity {
            prefill_chunk_size: 4096,
            ..test_identity()
        };
        let store = DFlash2DiskStore::open(&dir, &other, None).expect("open other");
        assert_eq!(store.snapshot().entries, 0);
        assert_ne!(store.root, root);
        assert_eq!(file_names(&root.join(ARTIFACTS_DIR)).len(), 1);
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_restore_failure_removes_the_entry() {
        let dir = temp_root("dflash2-store-bad-restore");
        let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
        let tokens: Vec<u32> = (0..40).collect();
        store
            .save(&test_save(&store, &tokens, false))
            .expect("save");
        let key = artifact_key(&tokens, "runtime");
        let segments = store.root.join(SEGMENTS_DIR);
        for name in file_names(&segments) {
            fs::remove_file(segments.join(name)).expect("remove");
        }
        let mut target_cache: Vec<LayerCache> = (0..2)
            .map(|_| LayerCache::Full(KVCache::new(1, 2, 4, 4, mlx::Dtype::Float32, 1 << 20)))
            .collect();
        assert!(store
            .restore_into(&key, &mut target_cache, &mut |_| {})
            .is_err());
        assert!(store.longest(&tokens, &|_| "runtime".to_owned()).is_none());
        assert_eq!(store.snapshot().invalid, 1);
        assert!(file_names(&store.root.join(ARTIFACTS_DIR)).is_empty());
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn store_identity_changes_with_weights_and_numeric_switches() {
        let model = temp_root("dflash2-store-model");
        fs::write(model.join("config.json"), b"{}").expect("config");
        fs::write(model.join("model.safetensors"), b"12345678").expect("weights");
        let identity = |fingerprint: &str, chunk: usize| {
            DFlash2StoreIdentity::for_model_dir(&model, fingerprint, "build", chunk, &[1, 5], 24)
                .expect("identity")
                .key()
        };
        let base = identity("runtime", 2048);
        assert_eq!(base, identity("runtime", 2048));
        assert_ne!(base, identity("runtime-m5", 2048));
        assert_ne!(base, identity("runtime", 4096));
        let other_build =
            DFlash2StoreIdentity::for_model_dir(&model, "runtime", "build-2", 2048, &[1, 5], 24)
                .expect("identity")
                .key();
        assert_ne!(base, other_build);
        fs::write(model.join("model.safetensors"), b"123456789").expect("weights");
        assert_ne!(base, identity("runtime", 2048));
        fs::remove_dir_all(model).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_limit_covers_every_identity() {
        let dir = temp_root("dflash2-store-shared-limit");
        let (other_root, one) = {
            let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
            save_prefix(&mut store, 0, 40);
            (store.root.clone(), store.total_bytes())
        };
        std::thread::sleep(std::time::Duration::from_millis(20));
        let identity = DFlash2StoreIdentity {
            prefill_chunk_size: 4096,
            ..test_identity()
        };
        // Room for one artifact and a half: the other identity's goes first.
        let mut store =
            DFlash2DiskStore::open(&dir, &identity, Some(one + one / 2)).expect("open other");
        assert!(other_root.exists(), "within the limit nothing is removed");
        save_prefix(&mut store, 100, 40);
        assert!(
            !other_root.exists(),
            "the least recently opened identity is removed"
        );
        assert_eq!(store.snapshot().entries, 1);
        assert!(dir_bytes(&dir) <= one + one / 2 + 4096);
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_keeps_an_artifact_a_restore_is_reading() {
        let dir = temp_root("dflash2-store-pinned");
        let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
        // 120 is superseded once three newer boundaries exist.
        save_prefix(&mut store, 0, 100);
        let reading = save_prefix(&mut store, 0, 120);
        let plan = store.begin_restore(&reading).expect("stored");
        for len in [300, 310, 320] {
            save_prefix(&mut store, 0, len);
        }
        let loaded = load_artifact(&plan).expect("still readable");
        assert_eq!(kv_positions(&loaded), range(0, 120));
        store.finish_restore(&reading, true);
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_rejects_a_save_whose_parent_was_removed_while_it_was_written() {
        let dir = temp_root("dflash2-store-parent-gone");
        let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
        let parent = save_prefix(&mut store, 0, 40);
        let tokens: Vec<u32> = (0..50).collect();
        let save = test_save(&store, &tokens, true);
        let plan = store.plan_save(&save).expect("plan").expect("new");
        let written = write_save(&plan, &save).expect("write");
        store.remove(&parent);
        assert!(store.commit_save(written).is_err());
        assert_eq!(store.snapshot().entries, 0);
        for sub in [SEGMENTS_DIR, CONTEXT_DIR, ARTIFACTS_DIR] {
            assert!(file_names(&store.root.join(sub)).is_empty(), "{sub}");
        }
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn superseded_boundaries_keeps_three_newest_and_one_per_power_of_two() {
        let mut boundaries: Vec<(usize, usize)> = [100, 120, 300, 310, 320, 330, 340]
            .into_iter()
            .map(|len| (len, len))
            .collect();
        assert_eq!(superseded_boundaries(&mut boundaries), vec![120, 310]);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_drops_boundaries_a_conversation_supersedes() {
        let dir = temp_root("dflash2-store-retain");
        let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
        for len in [100, 120, 300, 310, 320, 330, 340] {
            save_prefix(&mut store, 0, len);
        }
        let mut lens: Vec<i32> = store
            .entries
            .iter()
            .map(|entry| entry.meta.cached_len)
            .collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![100, 300, 320, 330, 340]);
        // What the dropped boundaries added is still part of the survivors.
        let key = artifact_key(&(0..340).collect::<Vec<u32>>(), "runtime");
        let loaded = load(&mut store, &key);
        assert_eq!(kv_positions(&loaded), range(0, 340));
        assert_eq!(context_values(&loaded), range(316, 340));
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_evicts_least_recently_used_to_max_bytes() {
        let dir = temp_root("dflash2-store-lru");
        let one = {
            let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
            save_prefix(&mut store, 0, 40);
            store.total_bytes()
        };
        let max_bytes = one * 2 + one / 2;
        let mut store =
            DFlash2DiskStore::open(&dir, &test_identity(), Some(max_bytes)).expect("reopen");
        let first = artifact_key(&(0..40).collect::<Vec<u32>>(), "runtime");
        std::thread::sleep(std::time::Duration::from_millis(20));
        let second = save_prefix(&mut store, 100, 40);
        std::thread::sleep(std::time::Duration::from_millis(20));
        load(&mut store, &first);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let third = save_prefix(&mut store, 200, 40);
        let keys: HashSet<String> = store.entries.iter().map(|e| e.key.clone()).collect();
        assert_eq!(keys, HashSet::from([first, third]));
        assert!(!keys.contains(&second));
        // The evicted conversation's range files went with it.
        assert_eq!(file_names(&store.root.join(SEGMENTS_DIR)).len(), 2);
        assert_eq!(file_names(&store.root.join(CONTEXT_DIR)).len(), 2);
        assert!(dir_bytes(&store.root) <= max_bytes + 4096);
        assert!(store.total_bytes() <= max_bytes);
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_rejects_an_artifact_larger_than_max_bytes() {
        let dir = temp_root("dflash2-store-too-large");
        let mut store = DFlash2DiskStore::open(&dir, &test_identity(), Some(64)).expect("open");
        let tokens: Vec<u32> = (0..40).collect();
        assert!(store.save(&test_save(&store, &tokens, true)).is_err());
        assert_eq!(store.snapshot().entries, 0);
        for sub in [SEGMENTS_DIR, CONTEXT_DIR, ARTIFACTS_DIR] {
            assert!(file_names(&store.root.join(sub)).is_empty(), "{sub}");
        }
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn disk_store_compacts_a_long_range_chain() {
        let dir = temp_root("dflash2-store-compact");
        let mut store = DFlash2DiskStore::open(&dir, &test_identity(), None).expect("open");
        let mut key = String::new();
        for len in 1..=300 {
            key = save_prefix(&mut store, 0, len);
        }
        let entry = store
            .entries
            .iter()
            .find(|entry| entry.key == key)
            .expect("newest");
        assert!(entry.meta.kv.len() + entry.meta.context.len() <= MAX_RANGE_FILES);
        assert!(entry.meta.kv.len() < 100);
        let loaded = load(&mut store, &key);
        assert_eq!(kv_positions(&loaded), range(0, 300));
        assert_eq!(context_values(&loaded), range(276, 300));
        let DFlash2DiskLayer::State(PrefixLayerPayload::Linear { conv_state, .. }) =
            &loaded.layers[1]
        else {
            panic!("state layer");
        };
        assert_eq!(values(conv_state), vec![300.0, 301.0]);
        fs::remove_dir_all(dir).expect("cleanup");
    }
}
