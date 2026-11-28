//! `wp8`: whole-prefix entries on disk as shared segment files plus a
//! private part.
//!
//! ```text
//! <root>/wp8/<layout>/segments/<start>-<stop>-<segment id>.safetensors
//! <root>/wp8/<layout>/entries/<key>/meta.json
//! <root>/wp8/<layout>/entries/<key>/private.safetensors
//! <root>/wp8/<layout>/entries/<key>/tokens.bin
//! ```
//!
//! A segment file holds one segment's whole token range; entries reference
//! it by id, possibly only its head. Writing an entry writes the segment
//! files that are not on disk yet, then the entry directory (temporary
//! directory + rename). The immutable block pool keeps `<root>/<key>/`.

use std::collections::HashSet;

use super::*;

pub(super) const WP8_DIR: &str = "wp8";
pub(super) const WP8_SEGMENTS_DIR: &str = "segments";
pub(super) const WP8_ENTRIES_DIR: &str = "entries";
const WP8_FORMAT: &str = "wp8-v1";
const PRIVATE_FILE: &str = "private.safetensors";
/// The entry's token ids (little-endian `i32`), read only to find entries a
/// longer one covers.
const TOKENS_FILE: &str = "tokens.bin";
const TMP_PREFIX: &str = ".tmp-";
const SAFETENSORS_EXT: &str = ".safetensors";

pub(super) fn wp8_layout_dir(root: &Path, layout: &PrefixSegmentLayout) -> PathBuf {
    root.join(WP8_DIR).join(&layout.0)
}

/// File name of a segment covering `[start, stop)`.
pub(super) fn wp8_segment_file_name(id: PrefixSegmentId, start: i32, stop: i32) -> String {
    format!("{start}-{stop}-{id}{SAFETENSORS_EXT}")
}

/// A disk read of a whole-prefix entry in its segmented form.
#[derive(Debug)]
pub struct PagedPrefixSegmentedLoadResult {
    pub key: String,
    pub status: PagedPrefixLoadStatus,
    pub entry: Option<SegmentedPrefixEntry>,
    pub stats: Option<PagedPrefixEntryStats>,
}

impl PagedPrefixSegmentedLoadResult {
    fn miss(key: String, status: PagedPrefixLoadStatus) -> Self {
        Self {
            key,
            status,
            entry: None,
            stats: None,
        }
    }
}

/// Finds a segment already in memory (the hot prefix cache) by id.
pub type PrefixSegmentLookup<'a> = &'a dyn Fn(PrefixSegmentId) -> Option<Arc<PrefixKvSegment>>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Wp8SegmentMetadata {
    pub(super) id: String,
    pub(super) start: i32,
    /// End of the range this entry uses.
    pub(super) stop: i32,
    /// End of the segment (and its file).
    pub(super) segment_stop: i32,
    pub(super) bytes: u64,
}

impl Wp8SegmentMetadata {
    pub(super) fn file_name(&self) -> String {
        format!(
            "{}-{}-{}{SAFETENSORS_EXT}",
            self.start, self.segment_stop, self.id
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Wp8EntryMetadata {
    pub(super) format: String,
    #[serde(flatten)]
    pub(super) prefix: PrefixMetadata,
    pub(super) aligned_len: i32,
    pub(super) segments: Vec<Wp8SegmentMetadata>,
    /// How many trailing `segments` hold the sliding-window layers, and the
    /// first position those layers store. Absent when there are none.
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub(super) window_refs: usize,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub(super) window_start: i32,
}

fn is_zero_usize(value: &usize) -> bool {
    *value == 0
}

fn is_zero_i32(value: &i32) -> bool {
    *value == 0
}

fn read_wp8_metadata(entry_dir: &Path) -> Option<Wp8EntryMetadata> {
    fs::read(entry_dir.join(META_FILE))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Wp8EntryMetadata>(&bytes).ok())
        .filter(|meta| meta.format == WP8_FORMAT)
}

fn is_tmp(name: &str) -> bool {
    name.starts_with(TMP_PREFIX)
}

fn list_dir(path: &Path) -> Result<Vec<fs::DirEntry>> {
    match fs::read_dir(path) {
        Ok(entries) => entries
            .collect::<std::io::Result<Vec<_>>>()
            .with_context(|| format!("read prefix cache dir {}", path.display())),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err).with_context(|| format!("read prefix cache dir {}", path.display())),
    }
}

pub(super) fn remove_path(path: &Path) {
    let result = if path.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    if let Err(error) = result {
        if error.kind() != ErrorKind::NotFound {
            tracing::warn!(%error, path = %path.display(), "prefix cache cleanup failed");
        }
    }
}

fn segment_tensor_name(layer_idx: usize, part: &str) -> String {
    format!("main_{layer_idx:04}_{part}")
}

const PAGED_PARTS: [&str; 2] = ["k_pages", "v_pages"];
const TURBOQUANT_PARTS: [&str; 4] = ["k_packed", "k_norms", "v_packed", "v_norms"];
const DENSE_PARTS: [&str; 2] = ["k", "v"];

/// Roots this process has cleaned of `wp8` leftovers.
static WP8_CLEANED_ROOTS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();

/// A `wp8` entry directory as eviction sees it.
#[derive(Debug)]
pub(super) struct Wp8DiskEntry {
    pub(super) layout_dir: PathBuf,
    pub(super) meta: Option<Wp8EntryMetadata>,
}

impl PagedPrefixStore {
    #[cfg(test)]
    pub(crate) fn wp8_entry_dir(&self, spec: &PagedPrefixKeySpec) -> PathBuf {
        wp8_layout_dir(&self.root, &prefix_segment_layout(spec))
            .join(WP8_ENTRIES_DIR)
            .join(Self::key_for(spec))
    }

    /// Write `entry` for `spec`: the segment files not on disk yet, then the
    /// entry. Returns its key.
    pub fn save_segmented(
        &self,
        spec: &PagedPrefixKeySpec,
        entry: &SegmentedPrefixEntry,
    ) -> Result<String> {
        self.save_segmented_counted(spec, entry).map(|(key, _)| key)
    }

    /// Like [`Self::save_segmented`] unless a matching entry is on disk.
    /// `true` when it wrote.
    pub fn save_segmented_if_absent(
        &self,
        spec: &PagedPrefixKeySpec,
        entry: &SegmentedPrefixEntry,
    ) -> Result<(String, bool)> {
        self.validate_spec(spec)?;
        let key = Self::key_for(spec);
        if self.wp8_entry_matches(spec, &key) {
            return Ok((key, false));
        }
        self.save_segmented(spec, entry).map(|key| (key, true))
    }

    /// [`Self::save_segmented`], also returning how many segment files it
    /// wrote.
    pub(crate) fn save_segmented_counted(
        &self,
        spec: &PagedPrefixKeySpec,
        entry: &SegmentedPrefixEntry,
    ) -> Result<(String, usize)> {
        if spec.entry_kind != PrefixEntryKind::WholePrefix {
            anyhow::bail!("PagedPrefixStore: segmented entries are whole prefixes");
        }
        self.validate_spec(spec)?;
        let layout = prefix_segment_layout(spec);
        if let Some(r) = entry.segments.iter().find(|r| r.segment.layout != layout) {
            anyhow::bail!(
                "PagedPrefixStore: segment {} layout does not match the entry",
                r.segment.id
            );
        }
        // Lazy: checks shapes without evaluating.
        let materialized = materialize_prefix_entry(entry, spec.cached_len, spec.block_size)?;
        self.validate_entry(spec, &materialized)?;
        let payload_bytes = materialized
            .observability_stats(spec.cached_len)
            .payload_bytes;
        drop(materialized);
        if let Some(max_bytes) = self.max_bytes {
            if payload_bytes > max_bytes {
                anyhow::bail!(
                    "PagedPrefixStore: entry payload_bytes {payload_bytes} exceeds max_bytes {max_bytes}"
                );
            }
        }
        self.ensure_wp8_startup_cleanup();

        let key = Self::key_for(spec);
        let layout_dir = wp8_layout_dir(&self.root, &layout);
        let segments_dir = layout_dir.join(WP8_SEGMENTS_DIR);
        let entries_dir = layout_dir.join(WP8_ENTRIES_DIR);
        for dir in [&segments_dir, &entries_dir] {
            fs::create_dir_all(dir)
                .with_context(|| format!("create prefix cache dir {}", dir.display()))?;
        }

        // What is on disk decides: a segment that only lived in memory
        // (backpressure, cancelled, no disk then) is written now.
        let mut written = 0_usize;
        let mut seen = HashSet::new();
        let mut segments = Vec::with_capacity(entry.segments.len());
        for r in &entry.segments {
            let segment = &r.segment;
            let name = wp8_segment_file_name(segment.id, segment.start, segment.stop);
            if seen.insert(segment.id) && !segments_dir.join(&name).is_file() {
                write_segment_file(&segments_dir, &name, segment)?;
                written += 1;
            }
            segments.push(Wp8SegmentMetadata {
                id: segment.id.to_string(),
                start: segment.start,
                stop: r.stop,
                segment_stop: segment.stop,
                bytes: segment.payload_bytes() as u64,
            });
        }

        let metadata = Wp8EntryMetadata {
            format: WP8_FORMAT.to_owned(),
            prefix: metadata_for(spec, &key)?,
            aligned_len: entry.aligned_len,
            segments,
            window_refs: entry.window_refs,
            window_start: entry.window_start,
        };
        let final_dir = entries_dir.join(&key);
        let tmp_dir = entries_dir.join(format!(
            "{TMP_PREFIX}{key}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        fs::create_dir_all(&tmp_dir)
            .with_context(|| format!("create temporary prefix cache dir {}", tmp_dir.display()))?;
        let write_result = (|| -> Result<()> {
            let meta_path = tmp_dir.join(META_FILE);
            let meta_bytes =
                serde_json::to_vec_pretty(&metadata).context("serialize wp8 prefix metadata")?;
            fs::write(&meta_path, meta_bytes)
                .with_context(|| format!("write prefix cache metadata {}", meta_path.display()))?;
            let tokens = spec
                .token_ids
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect::<Vec<_>>();
            fs::write(tmp_dir.join(TOKENS_FILE), tokens).context("write prefix cache tokens")?;
            let mut tensors = HashMap::new();
            insert_entry_tensors(&entry.private, &mut tensors);
            let mut file_meta = HashMap::new();
            file_meta.insert("ironmlx.prefix_cache.key".to_owned(), key.clone());
            file_meta.insert(
                "ironmlx.prefix_cache.format".to_owned(),
                WP8_FORMAT.to_owned(),
            );
            let private_path = tmp_dir.join(PRIVATE_FILE).to_string_lossy().into_owned();
            mlx::io::save_safetensors(&private_path, &tensors, &file_meta)
                .with_context(|| format!("save prefix cache private payload {private_path}"))?;
            Ok(())
        })();
        if let Err(err) = write_result {
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
        if entry.window_refs > 0 {
            self.drop_superseded_window_checkpoints(spec, &key)?;
        }
        if let Some(max_bytes) = self.max_bytes {
            self.evict_to_disk_capacity(&key, max_bytes)?;
        }
        Ok((key, written))
    }

    /// Thin out the stored shorter prefixes of the entry just written at
    /// `key`, which stores sliding-window layers, by the rule the hot cache
    /// applies (`PrefixLruCache::drop_superseded_window_checkpoints`): each
    /// carries its own window files, so keeping every prefill checkpoint
    /// costs a window apiece. Segment files left without an entry go too.
    fn drop_superseded_window_checkpoints(
        &self,
        spec: &PagedPrefixKeySpec,
        key: &str,
    ) -> Result<()> {
        let layout_dir = wp8_layout_dir(&self.root, &prefix_segment_layout(spec));
        let entries_dir = layout_dir.join(WP8_ENTRIES_DIR);
        let Some(own) = read_wp8_metadata(&entries_dir.join(key)) else {
            return Ok(());
        };
        let mut paths = Vec::new();
        let mut line = Vec::new();
        for item in list_dir(&entries_dir)? {
            let name = item.file_name().to_string_lossy().into_owned();
            let path = item.path();
            if is_tmp(&name) || !path.is_dir() {
                continue;
            }
            let Some(meta) = read_wp8_metadata(&path) else {
                continue;
            };
            if meta.prefix.fingerprint_hash != own.prefix.fingerprint_hash {
                continue;
            }
            let Some(tokens) = read_wp8_tokens(&path) else {
                continue;
            };
            let continued = name == key
                || (tokens.len() < spec.token_ids.len()
                    && tokens
                        .iter()
                        .zip(&spec.token_ids)
                        .all(|(stored, token)| i64::from(*stored) == i64::from(*token)));
            if continued {
                line.push((tokens.len(), paths.len()));
                paths.push(path);
            }
        }
        let dropped = crate::core::dflash2_prefix_store::superseded_boundaries(&mut line);
        if dropped.is_empty() {
            return Ok(());
        }
        for index in dropped {
            remove_path(&paths[index]);
        }
        let referenced = referenced_segment_files(&layout_dir)?;
        for item in list_dir(&layout_dir.join(WP8_SEGMENTS_DIR))? {
            let name = item.file_name().to_string_lossy().into_owned();
            if !is_tmp(&name) && !referenced.contains(&name) {
                remove_path(&item.path());
            }
        }
        Ok(())
    }

    /// Bytes writing `entry` for `spec` puts on disk: its private part and
    /// every segment whose file is not there yet, except those in
    /// `queued`, which a save ahead of it in the queue writes first.
    pub(super) fn segmented_write_bytes(
        &self,
        spec: &PagedPrefixKeySpec,
        entry: &SegmentedPrefixEntry,
        queued: &HashSet<PrefixSegmentId>,
    ) -> usize {
        let segments_dir =
            wp8_layout_dir(&self.root, &prefix_segment_layout(spec)).join(WP8_SEGMENTS_DIR);
        let mut seen = HashSet::new();
        let mut bytes = entry
            .private
            .observability_stats(spec.cached_len)
            .payload_bytes;
        for r in &entry.segments {
            let segment = &r.segment;
            let name = wp8_segment_file_name(segment.id, segment.start, segment.stop);
            if seen.insert(segment.id)
                && !queued.contains(&segment.id)
                && !segments_dir.join(&name).is_file()
            {
                bytes = bytes.saturating_add(segment.payload_bytes());
            }
        }
        bytes
    }

    /// Whether a readable `wp8` entry for exactly `spec` is on disk (its
    /// segments are checked when it is loaded).
    pub(super) fn wp8_entry_matches(&self, spec: &PagedPrefixKeySpec, key: &str) -> bool {
        let entry_dir = wp8_layout_dir(&self.root, &prefix_segment_layout(spec))
            .join(WP8_ENTRIES_DIR)
            .join(key);
        entry_dir.join(PRIVATE_FILE).is_file()
            && read_wp8_metadata(&entry_dir)
                .is_some_and(|meta| metadata_for(spec, key).is_ok_and(|want| meta.prefix == want))
    }

    /// Read the whole-prefix entry for `spec`. Segments `lookup` finds in
    /// memory are reused; the rest are read from their files. A missing or
    /// corrupt segment is a miss and removes the entry.
    pub fn load_segmented_observed(
        &self,
        spec: &PagedPrefixKeySpec,
        lookup: Option<PrefixSegmentLookup<'_>>,
    ) -> Result<PagedPrefixSegmentedLoadResult> {
        self.validate_spec(spec)?;
        if spec.entry_kind != PrefixEntryKind::WholePrefix {
            anyhow::bail!("PagedPrefixStore: segmented entries are whole prefixes");
        }
        if let Some(observed) = PROCESS_PREFIX_STORE_QUEUE
            .get()
            .and_then(|queue| queue.load_pending_segmented(self, spec))
        {
            return Ok(observed);
        }
        let key = Self::key_for(spec);
        let layout = prefix_segment_layout(spec);
        let layout_dir = wp8_layout_dir(&self.root, &layout);
        let entry_dir = layout_dir.join(WP8_ENTRIES_DIR).join(&key);
        if !entry_dir.is_dir() {
            return Ok(PagedPrefixSegmentedLoadResult::miss(
                key,
                PagedPrefixLoadStatus::MissingEntry,
            ));
        }
        let Some(metadata) = read_wp8_metadata(&entry_dir) else {
            return Ok(PagedPrefixSegmentedLoadResult::miss(
                key,
                PagedPrefixLoadStatus::InvalidMetadata,
            ));
        };
        if metadata.prefix != metadata_for(spec, &key)? {
            return Ok(PagedPrefixSegmentedLoadResult::miss(
                key,
                PagedPrefixLoadStatus::MetadataMismatch,
            ));
        }

        let private_path = entry_dir.join(PRIVATE_FILE).to_string_lossy().into_owned();
        let Ok((mut tensors, _)) = mlx::io::load_safetensors(&private_path) else {
            return Ok(PagedPrefixSegmentedLoadResult::miss(
                key,
                PagedPrefixLoadStatus::PayloadReadFailed,
            ));
        };
        let private = match entry_from_tensors(spec, &mut tensors) {
            Ok(private) if tensors.is_empty() => private,
            _ => {
                return Ok(PagedPrefixSegmentedLoadResult::miss(
                    key,
                    PagedPrefixLoadStatus::PayloadInvalid,
                ));
            }
        };

        let segments_dir = layout_dir.join(WP8_SEGMENTS_DIR);
        let mut loaded = HashMap::<PrefixSegmentId, Arc<PrefixKvSegment>>::new();
        let mut segments = Vec::with_capacity(metadata.segments.len());
        for meta in &metadata.segments {
            let segment = u128::from_str_radix(&meta.id, 16)
                .ok()
                .map(PrefixSegmentId)
                .and_then(|id| {
                    if let Some(segment) = loaded.get(&id) {
                        return Some(segment.clone());
                    }
                    let in_memory = lookup.and_then(|lookup| lookup(id)).filter(|segment| {
                        segment.layout == layout
                            && segment.start == meta.start
                            && segment.stop == meta.segment_stop
                    });
                    let segment = match in_memory {
                        Some(segment) => segment,
                        None => Arc::new(
                            read_segment_file(
                                &segments_dir.join(meta.file_name()),
                                spec,
                                &layout,
                                id,
                                meta,
                            )
                            .ok()?,
                        ),
                    };
                    loaded.insert(id, segment.clone());
                    Some(segment)
                });
            let Some(segment) = segment else {
                tracing::debug!(
                    key,
                    segment = meta.id,
                    "paged SSD prefix entry lost a segment; removing it"
                );
                remove_path(&entry_dir);
                return Ok(PagedPrefixSegmentedLoadResult::miss(
                    key,
                    PagedPrefixLoadStatus::MissingEntry,
                ));
            };
            segments.push(PrefixSegmentRef {
                segment,
                stop: meta.stop,
            });
        }
        let entry = SegmentedPrefixEntry {
            window_refs: metadata.window_refs,
            window_start: metadata.window_start,
            segments,
            aligned_len: metadata.aligned_len,
            private,
        };
        let materialized = materialize_prefix_entry(&entry, spec.cached_len, spec.block_size)
            .ok()
            .filter(|materialized| self.validate_entry(spec, materialized).is_ok());
        let Some(materialized) = materialized else {
            remove_path(&entry_dir);
            return Ok(PagedPrefixSegmentedLoadResult::miss(
                key,
                PagedPrefixLoadStatus::EntryInvalid,
            ));
        };
        let stats = materialized.observability_stats(spec.cached_len);
        Ok(PagedPrefixSegmentedLoadResult {
            key,
            status: PagedPrefixLoadStatus::Hit,
            entry: Some(entry),
            stats: Some(stats),
        })
    }

    /// Lengths of the `wp8` entries on disk in `(0, max_cached_len]`.
    pub(super) fn wp8_cached_lengths(&self, max_cached_len: i32) -> Result<Vec<i32>> {
        let mut lengths = Vec::new();
        for layout in list_dir(&self.root.join(WP8_DIR))? {
            for entry in list_dir(&layout.path().join(WP8_ENTRIES_DIR))? {
                if is_tmp(&entry.file_name().to_string_lossy()) {
                    continue;
                }
                let Some(meta) = read_wp8_metadata(&entry.path()) else {
                    continue;
                };
                let cached_len = meta.prefix.cached_len;
                if meta.prefix.schema_version == SCHEMA_VERSION
                    && meta.prefix.entry_kind == PrefixEntryKind::WholePrefix
                    && cached_len > 0
                    && cached_len <= max_cached_len
                {
                    lengths.push(cached_len);
                }
            }
        }
        Ok(lengths)
    }

    /// The first write to a root in this process first removes what an
    /// earlier run left: `.tmp-*` directories and files, and segment files
    /// no entry references. The DFlash2 directory is never touched.
    pub(super) fn ensure_wp8_startup_cleanup(&self) {
        let first = WP8_CLEANED_ROOTS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .is_ok_and(|mut cleaned| cleaned.insert(self.root.clone()));
        if first {
            if let Err(error) = self.clean_wp8_residue() {
                tracing::warn!(
                    %error,
                    root = %self.root.display(),
                    "paged SSD prefix cache startup cleanup failed"
                );
            }
        }
    }

    fn clean_wp8_residue(&self) -> Result<()> {
        for item in list_dir(&self.root)? {
            if is_tmp(&item.file_name().to_string_lossy()) {
                remove_path(&item.path());
            }
        }
        for layout in list_dir(&self.root.join(WP8_DIR))? {
            let layout_dir = layout.path();
            for item in list_dir(&layout_dir.join(WP8_ENTRIES_DIR))? {
                if is_tmp(&item.file_name().to_string_lossy()) {
                    remove_path(&item.path());
                }
            }
            let referenced = referenced_segment_files(&layout_dir)?;
            for item in list_dir(&layout_dir.join(WP8_SEGMENTS_DIR))? {
                let name = item.file_name().to_string_lossy().into_owned();
                if is_tmp(&name) || !referenced.contains(&name) {
                    remove_path(&item.path());
                }
            }
        }
        Ok(())
    }

    /// Every `wp8` entry directory (temporary ones excluded).
    pub(super) fn wp8_disk_entries(&self) -> Result<Vec<(PrefixDiskEntry, Wp8DiskEntry)>> {
        let mut result = Vec::new();
        for layout in list_dir(&self.root.join(WP8_DIR))? {
            let layout_dir = layout.path();
            for item in list_dir(&layout_dir.join(WP8_ENTRIES_DIR))? {
                let key = item.file_name().to_string_lossy().into_owned();
                let path = item.path();
                if is_tmp(&key) || !path.is_dir() {
                    continue;
                }
                let modified = item
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(UNIX_EPOCH);
                result.push((
                    PrefixDiskEntry {
                        key,
                        bytes: directory_disk_bytes(&path)?,
                        path: path.clone(),
                        modified,
                    },
                    Wp8DiskEntry {
                        layout_dir: layout_dir.clone(),
                        meta: read_wp8_metadata(&path),
                    },
                ));
            }
        }
        Ok(result)
    }

    /// Segment files per layout directory with their sizes.
    pub(super) fn wp8_segment_files(&self) -> Result<HashMap<PathBuf, HashMap<String, u64>>> {
        let mut result = HashMap::new();
        for layout in list_dir(&self.root.join(WP8_DIR))? {
            let layout_dir = layout.path();
            let mut files = HashMap::new();
            for item in list_dir(&layout_dir.join(WP8_SEGMENTS_DIR))? {
                let bytes = item.metadata().map(|m| m.len()).unwrap_or(0);
                files.insert(item.file_name().to_string_lossy().into_owned(), bytes);
            }
            result.insert(layout_dir, files);
        }
        Ok(result)
    }
}

/// Segment files the entries of `layout_dir` reference.
fn referenced_segment_files(layout_dir: &Path) -> Result<HashSet<String>> {
    let mut referenced = HashSet::new();
    for item in list_dir(&layout_dir.join(WP8_ENTRIES_DIR))? {
        if is_tmp(&item.file_name().to_string_lossy()) {
            continue;
        }
        if let Some(meta) = read_wp8_metadata(&item.path()) {
            referenced.extend(meta.segments.iter().map(Wp8SegmentMetadata::file_name));
        }
    }
    Ok(referenced)
}

/// Read an entry's token ids from its directory.
pub(super) fn read_wp8_tokens(entry_dir: &Path) -> Option<Vec<i32>> {
    let bytes = fs::read(entry_dir.join(TOKENS_FILE)).ok()?;
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

/// Write `segment` as `segments_dir/name` (temporary file + rename).
fn write_segment_file(segments_dir: &Path, name: &str, segment: &PrefixKvSegment) -> Result<()> {
    let mut tensors = HashMap::new();
    for (idx, layer) in &segment.layers {
        match layer {
            PrefixSegmentLayer::Paged { k, v } => {
                for (part, array) in PAGED_PARTS.iter().zip([k, v]) {
                    tensors.insert(segment_tensor_name(*idx, part), array.clone());
                }
            }
            PrefixSegmentLayer::Dense { k, v } => {
                for (part, array) in DENSE_PARTS.iter().zip([k, v]) {
                    tensors.insert(segment_tensor_name(*idx, part), array.clone());
                }
            }
            PrefixSegmentLayer::TurboQuant {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            } => {
                for (part, array) in TURBOQUANT_PARTS
                    .iter()
                    .zip([k_packed, k_norms, v_packed, v_norms])
                {
                    tensors.insert(segment_tensor_name(*idx, part), array.clone());
                }
            }
        }
    }
    let mut file_meta = HashMap::new();
    file_meta.insert(
        "ironmlx.prefix_segment.id".to_owned(),
        segment.id.to_string(),
    );
    file_meta.insert(
        "ironmlx.prefix_segment.layout".to_owned(),
        segment.layout.0.clone(),
    );
    file_meta.insert(
        "ironmlx.prefix_cache.format".to_owned(),
        WP8_FORMAT.to_owned(),
    );
    // MLX appends the extension to a path without it.
    let tmp_path = segments_dir.join(format!(
        "{TMP_PREFIX}{}{SAFETENSORS_EXT}",
        uuid::Uuid::new_v4().simple()
    ));
    let tmp = tmp_path.to_string_lossy().into_owned();
    if let Err(err) = mlx::io::save_safetensors(&tmp, &tensors, &file_meta) {
        let _ = fs::remove_file(&tmp_path);
        return Err(anyhow::Error::from(err).context(format!("save prefix segment {tmp}")));
    }
    let final_path = segments_dir.join(name);
    fs::rename(&tmp_path, &final_path).with_context(|| {
        format!(
            "install prefix segment {} -> {}",
            tmp_path.display(),
            final_path.display()
        )
    })
}

/// Read and check one segment file of an entry for `spec`.
fn read_segment_file(
    path: &Path,
    spec: &PagedPrefixKeySpec,
    layout: &PrefixSegmentLayout,
    id: PrefixSegmentId,
    meta: &Wp8SegmentMetadata,
) -> Result<PrefixKvSegment> {
    let len = meta.segment_stop - meta.start;
    if meta.start < 0 || len <= 0 || len % spec.block_size != 0 {
        anyhow::bail!("wp8 segment {} has an invalid range", meta.id);
    }
    let (mut tensors, file_meta) =
        mlx::io::load_safetensors(&path.to_string_lossy()).map_err(anyhow::Error::from)?;
    if file_meta.get("ironmlx.prefix_segment.id") != Some(&meta.id) {
        anyhow::bail!("wp8 segment file {} names another segment", path.display());
    }
    let mut layers = Vec::new();
    for (idx, layer_spec) in spec.main_layers.iter().enumerate() {
        let (parts, length_dim, length): (&[&str], usize, i32) = match layer_spec.kind {
            PrefixLayerKind::FullPaged => (&PAGED_PARTS, 0, len / spec.block_size),
            PrefixLayerKind::FullTurboQuantPacked => (&TURBOQUANT_PARTS, 2, len),
            PrefixLayerKind::FullDense => (&DENSE_PARTS, 2, len),
            _ => continue,
        };
        let names = parts
            .iter()
            .map(|part| segment_tensor_name(idx, part))
            .collect::<Vec<_>>();
        if !names.iter().any(|name| tensors.contains_key(name)) {
            continue;
        }
        let mut arrays = Vec::with_capacity(names.len());
        for (name, tensor_spec) in names.into_iter().zip(&layer_spec.tensors) {
            let array = take_tensor(&mut tensors, name.clone())?;
            let mut want = tensor_spec.shape.clone();
            want[length_dim] = length;
            if array.dtype() != tensor_spec.dtype || array.shape().as_slice() != want.as_slice() {
                anyhow::bail!("wp8 segment tensor {name} does not match the entry");
            }
            arrays.push(array);
        }
        let layer = match arrays.as_slice() {
            [k, v] if layer_spec.kind == PrefixLayerKind::FullDense => PrefixSegmentLayer::Dense {
                k: k.clone(),
                v: v.clone(),
            },
            [k, v] => PrefixSegmentLayer::Paged {
                k: k.clone(),
                v: v.clone(),
            },
            [k_packed, k_norms, v_packed, v_norms] => PrefixSegmentLayer::TurboQuant {
                k_packed: k_packed.clone(),
                k_norms: k_norms.clone(),
                v_packed: v_packed.clone(),
                v_norms: v_norms.clone(),
            },
            _ => anyhow::bail!("wp8 segment layer {idx} is incomplete"),
        };
        layers.push((idx, layer));
    }
    if !tensors.is_empty() || layers.is_empty() {
        anyhow::bail!("wp8 segment file {} has unexpected tensors", path.display());
    }
    Ok(PrefixKvSegment {
        id,
        layout: layout.clone(),
        start: meta.start,
        stop: meta.segment_stop,
        layers,
    })
}

impl AsyncPrefixStorePermit {
    /// Submit a whole-prefix entry in its segmented form. The writer writes
    /// the segments it references that are not on disk yet.
    pub fn submit_segmented(mut self, entry: SegmentedPrefixEntry) -> AsyncPrefixStoreSubmit {
        let queue = self.queue.clone();
        queue.submit_permit(&mut self, AsyncPrefixStorePayload::Segmented(entry))
    }
}
