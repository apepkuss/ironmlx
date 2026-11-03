//! Read-only safetensors ranges for bounded PLE and expert residency.
//!
//! Opening a file reads its header only. Reads use positional I/O and an
//! explicit byte limit; no MLX array or whole-model evaluation is involved.
//! Callers must hold a verified immutable snapshot for the file's lifetime.
use std::collections::BTreeMap;
use std::fs::{File, Metadata};
use std::io::Read;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Component, Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, OnceLock};

use anyhow::{anyhow, ensure, Context, Result};
use serde::Deserialize;

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileStamp {
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl From<Metadata> for FileStamp {
    fn from(m: Metadata) -> Self {
        Self {
            device: m.dev(),
            inode: m.ino(),
            size: m.len(),
            modified: (m.mtime(), m.mtime_nsec()),
            changed: (m.ctime(), m.ctime_nsec()),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl TensorInfo {
    pub fn bytes(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }

    fn item_bytes(&self) -> Result<u64> {
        Ok(match self.dtype.as_str() {
            "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
            "F16" | "BF16" | "I16" | "U16" => 2,
            "F32" | "I32" | "U32" => 4,
            "F64" | "I64" | "U64" => 8,
            other => return Err(anyhow!("unsupported tensor storage dtype {other}")),
        })
    }
}

pub struct TensorFile {
    file: File,
    stamp: FileStamp,
    payload: u64,
    tensors: BTreeMap<String, TensorInfo>,
}

impl TensorFile {
    pub fn validate_source(&self) -> Result<()> {
        ensure!(
            FileStamp::from(self.file.metadata()?) == self.stamp,
            "tensor source changed after inspection"
        );
        Ok(())
    }
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        match std::env::var("IRONMLX_FLASH_IO_POLICY")
            .unwrap_or_else(|_| "cached".into())
            .as_str()
        {
            "cached" => {}
            "no_cache" => {
                #[cfg(target_os = "macos")]
                {
                    use std::os::fd::AsRawFd;
                    // This is an explicit A/B policy; it does not promise a cold SSD/controller.
                    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
                        return Err(std::io::Error::last_os_error().into());
                    }
                }
                #[cfg(not(target_os = "macos"))]
                anyhow::bail!("no_cache paging policy requires macOS");
            }
            _ => anyhow::bail!("IRONMLX_FLASH_IO_POLICY must be cached or no_cache"),
        }
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "tensor source must be a regular file");
        let stamp = FileStamp::from(metadata);
        let mut length = [0u8; 8];
        file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        ensure!(
            (2..=MAX_HEADER_BYTES).contains(&length),
            "invalid safetensors header length"
        );
        let payload = length
            .checked_add(8)
            .ok_or_else(|| anyhow!("header offset overflow"))?;
        ensure!(payload <= stamp.size, "truncated safetensors header");
        let mut header = vec![0; usize::try_from(length)?];
        file.read_exact(&mut header)?;
        let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)?;
        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let info: TensorInfo =
                serde_json::from_value(value).with_context(|| format!("tensor {name}"))?;
            let [start, end] = info.data_offsets;
            ensure!(
                start <= end && end <= stamp.size - payload,
                "tensor {name} is outside the payload"
            );
            let expected = info
                .shape
                .iter()
                .try_fold(info.item_bytes()?, |n, &dim| n.checked_mul(dim))
                .ok_or_else(|| anyhow!("tensor {name} size overflow"))?;
            ensure!(
                expected == end - start,
                "tensor {name} shape does not match storage length"
            );
            ranges.push((start, end));
            tensors.insert(name, info);
        }
        ranges.sort_unstable();
        let mut cursor = 0;
        for (start, end) in ranges {
            ensure!(
                start == cursor,
                "tensor payload has overlapping ranges or holes"
            );
            cursor = end;
        }
        ensure!(
            cursor == stamp.size - payload,
            "unindexed safetensors payload bytes"
        );
        ensure!(
            FileStamp::from(file.metadata()?) == stamp,
            "tensor source changed during inspection"
        );
        Ok(Self {
            file,
            stamp,
            payload,
            tensors,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    pub fn read_tensor(&self, name: &str, max_bytes: usize) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?;
        self.read(info.data_offsets[0], info.bytes(), max_bytes)
    }

    /// Read consecutive dimension-zero rows, preserving their packed bytes.
    /// Experts are rows of [E,N,K/pack]; PLE entries are rows of [V,D/pack].
    /// Quantization scales and biases must be read with the same row range.
    pub fn read_rows(
        &self,
        name: &str,
        start: u64,
        count: u64,
        max_bytes: usize,
    ) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let rows = *info
            .shape
            .first()
            .ok_or_else(|| anyhow!("scalar tensor has no row axis"))?;
        ensure!(
            start.checked_add(count).is_some_and(|end| end <= rows),
            "tensor row range is out of bounds"
        );
        let row_bytes = info.shape[1..]
            .iter()
            .try_fold(info.item_bytes()?, |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| anyhow!("tensor row size overflow"))?;
        let offset = start
            .checked_mul(row_bytes)
            .and_then(|n| n.checked_add(info.data_offsets[0]))
            .ok_or_else(|| anyhow!("tensor row offset overflow"))?;
        let bytes = count
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow!("tensor read size overflow"))?;
        self.read(offset, bytes, max_bytes)
    }

    /// Materialize selected rows through MLX's safetensors decoder without
    /// reinterpreting unaligned Rust bytes. The envelope, decoder and final
    /// array each fit the staging reservation. Eager evaluation on this
    /// thread prevents lazy loading primitives crossing actor threads.
    pub fn read_rows_array(
        &self,
        name: &str,
        start: u64,
        count: u64,
        staging_budget_bytes: usize,
    ) -> Result<mlx::Array> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?;
        ensure!(!info.shape.is_empty(), "scalar tensor has no row axis");
        let mut shape = info.shape.clone();
        shape[0] = count;
        let payload_bytes = shape
            .iter()
            .try_fold(info.item_bytes()?, |n, &d| n.checked_mul(d))
            .ok_or_else(|| anyhow!("tensor row size overflow"))?;
        let mut header = serde_json::to_vec(&serde_json::json!({
            "rows": {"dtype":info.dtype, "shape":shape, "data_offsets":[0,payload_bytes]}
        }))?;
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let payload_bytes = usize::try_from(payload_bytes)?;
        let peak = payload_bytes
            .checked_add(header.len() + 8)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| anyhow!("tensor staging size overflow"))?;
        ensure!(
            peak <= staging_budget_bytes,
            "tensor array staging requires {peak} bytes, budget is {staging_budget_bytes}"
        );
        let mut raw = self.read_rows(name, start, count, payload_bytes)?;
        let mut envelope = Vec::with_capacity(header.len() + 8 + raw.len());
        envelope.extend_from_slice(&(header.len() as u64).to_le_bytes());
        envelope.extend_from_slice(&header);
        envelope.append(&mut raw);
        drop(raw);
        let mut reader = mlx::io::Reader::from_bytes(&envelope);
        let (mut tensors, _) = mlx::io::load_safetensors_from_reader(&mut reader)?;
        let array = tensors
            .remove("rows")
            .ok_or_else(|| anyhow!("tensor decoder returned no rows"))?;
        array.eval()?;
        Ok(array)
    }

    fn read(&self, offset: u64, bytes: u64, max_bytes: usize) -> Result<Vec<u8>> {
        let bytes = usize::try_from(bytes)?;
        ensure!(
            bytes <= max_bytes,
            "tensor read exceeds the staging budget ({bytes} > {max_bytes})"
        );
        ensure!(
            FileStamp::from(self.file.metadata()?) == self.stamp,
            "tensor source changed after inspection"
        );
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)?;
        data.resize(bytes, 0);
        self.file.read_exact_at(
            &mut data,
            self.payload
                .checked_add(offset)
                .ok_or_else(|| anyhow!("file offset overflow"))?,
        )?;
        ensure!(
            FileStamp::from(self.file.metadata()?) == self.stamp,
            "tensor source changed during read"
        );
        Ok(data)
    }
}

/// Metadata index with one open descriptor per checkpoint shard. Tensor
/// names retain their checkpoint spelling; model-specific sanitization is
/// performed by the model adapter, without joining full expert/PLE tensors.
pub struct ModelTensorFiles {
    sources: Vec<Arc<TensorFile>>,
    tensors: BTreeMap<String, Arc<TensorFile>>,
}

impl ModelTensorFiles {
    pub fn validate_sources(&self) -> Result<()> {
        for file in &self.sources {
            file.validate_source()?;
        }
        Ok(())
    }
    pub fn open(directory: &Path) -> Result<Self> {
        let root = directory.canonicalize()?;
        let index = root.join("model.safetensors.index.json");
        let mut tensors = BTreeMap::new();
        if !index.exists() {
            let file = Arc::new(TensorFile::open(&snapshot_file(
                &root,
                "model.safetensors",
            )?)?);
            for name in file.tensors.keys() {
                tensors.insert(name.clone(), Arc::clone(&file));
            }
            return Ok(Self {
                sources: vec![file],
                tensors,
            });
        }
        let index_path = snapshot_file(&root, "model.safetensors.index.json")?;
        let index_file = File::open(index_path)?;
        ensure!(
            index_file.metadata()?.len() <= MAX_HEADER_BYTES,
            "safetensors index is too large"
        );
        #[derive(Deserialize)]
        struct Index {
            weight_map: BTreeMap<String, String>,
        }
        let mut index_bytes = Vec::new();
        index_file
            .take(MAX_HEADER_BYTES + 1)
            .read_to_end(&mut index_bytes)?;
        ensure!(
            index_bytes.len() as u64 <= MAX_HEADER_BYTES,
            "safetensors index grew beyond the header budget"
        );
        let index: Index = serde_json::from_slice(&index_bytes)?;
        ensure!(!index.weight_map.is_empty(), "empty safetensors weight map");
        let mut shards = BTreeMap::<String, Arc<TensorFile>>::new();
        for (name, shard) in &index.weight_map {
            if !shards.contains_key(shard) {
                let path = snapshot_file(&root, shard)?;
                shards.insert(shard.clone(), Arc::new(TensorFile::open(&path)?));
            }
            let file = &shards[shard];
            ensure!(
                file.tensors.contains_key(name),
                "indexed tensor {name} is missing from shard {shard}"
            );
            tensors.insert(name.clone(), Arc::clone(file));
        }
        for (shard, file) in &shards {
            for name in file.tensors.keys() {
                ensure!(
                    index.weight_map.get(name) == Some(shard),
                    "tensor {name} disagrees with the shard index"
                );
            }
        }
        Ok(Self {
            sources: shards.into_values().collect(),
            tensors,
        })
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)?.tensors.get(name)
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.tensors
            .iter()
            .map(|(name, file)| (name.as_str(), &file.tensors[name]))
    }

    pub fn read_rows(
        &self,
        name: &str,
        start: u64,
        count: u64,
        max_bytes: usize,
    ) -> Result<Vec<u8>> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?
            .read_rows(name, start, count, max_bytes)
    }

    pub fn read_raw_rows(
        &self,
        name: &str,
        start: u64,
        count: u64,
        budget: usize,
    ) -> Result<RawRows> {
        let file = self.tensors.get(name).context("missing tensor")?;
        let mut info = file.tensors[name].clone();
        let data = file.read_rows(name, start, count, budget)?;
        info.shape[0] = count;
        Ok(RawRows { info, data })
    }
    /// Bounded CPU-only parallel reads. No MLX value crosses a worker boundary.
    pub fn read_rows_batch(
        &self,
        requests: &[(String, u64, u64)],
        max_bytes: usize,
    ) -> Result<Vec<RawRows>> {
        let mut total = 0usize;
        let mut pending = Vec::new();
        for (name, start, count) in requests {
            let file = Arc::clone(self.tensors.get(name).context("missing tensor")?);
            let info = file.tensors[name].clone();
            let rows = *info.shape.first().context("scalar tensor has no rows")?;
            ensure!(
                rows > 0 && start.checked_add(*count).is_some_and(|end| end <= rows),
                "invalid raw row range"
            );
            let bytes = usize::try_from(info.bytes() / rows)?
                .checked_mul(usize::try_from(*count)?)
                .context("raw row size overflow")?;
            total = total.checked_add(bytes).context("raw batch overflow")?;
            ensure!(total <= max_bytes, "raw batch exceeds staging budget");
            pending.push((file, info, name.clone(), *start, *count, bytes));
        }
        let receivers: Vec<_> = pending
            .into_iter()
            .map(|(file, mut info, name, start, count, bytes)| {
                io_task(move || {
                    let data = file.read_rows(&name, start, count, bytes)?;
                    info.shape[0] = count;
                    Ok(RawRows { info, data })
                })
            })
            .collect();
        let mut output = Vec::new();
        let mut failure = None;
        // Drain every submitted operation even on failure (unload/cancel never
        // leaves unaccounted I/O or buffers running in the background).
        for receiver in receivers {
            match receiver.recv().context("paging reader stopped")? {
                Ok(rows) => output.push(rows),
                Err(error) => {
                    failure = Some(error);
                }
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(output)
    }

    pub fn read_rows_array(
        &self,
        name: &str,
        start: u64,
        count: u64,
        staging_budget_bytes: usize,
    ) -> Result<mlx::Array> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?
            .read_rows_array(name, start, count, staging_budget_bytes)
    }
}

fn snapshot_file(root: &Path, relative: &str) -> Result<PathBuf> {
    ensure!(
        !relative.is_empty()
            && Path::new(relative)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "unsafe safetensors path"
    );
    let path = root.join(relative).canonicalize()?;
    ensure!(
        path.starts_with(root),
        "safetensors file escapes the snapshot"
    );
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("ironmlx-tensor-file-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn file(&self, header: serde_json::Value, data: &[u8]) -> PathBuf {
            let path = self.0.join("model.safetensors");
            let mut file = File::create(&path).unwrap();
            let mut header = serde_json::to_vec(&header).unwrap();
            while header.len() % 8 != 0 {
                header.push(b' ');
            }
            file.write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(&header).unwrap();
            file.write_all(data).unwrap();
            path
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn row_reads_preserve_storage_and_enforce_bounds_and_budget() {
        let d = Directory::new();
        let path = d.file(
            serde_json::json!({"w":{"dtype":"U32","shape":[3,2],"data_offsets":[0,24]}}),
            &(0u8..24).collect::<Vec<_>>(),
        );
        let f = TensorFile::open(&path).unwrap();
        assert_eq!(
            f.read_rows("w", 1, 1, 8).unwrap(),
            (8u8..16).collect::<Vec<_>>()
        );
        assert!(f.read_rows("w", 1, 2, 8).is_err());
        assert!(f.read_rows("w", 3, 1, 8).is_err());
        assert!(f.read_rows("w", u64::MAX, 2, 8).is_err());
        assert!(f.read_tensor("w", 23).is_err());
        assert!(f.read_rows("w", 3, 0, 0).unwrap().is_empty());
    }

    #[test]
    fn parallel_raw_reads_preserve_order_and_reject_oversized_batches() {
        let d = Directory::new();
        d.file(
            serde_json::json!({"w":{"dtype":"U32","shape":[3,2],"data_offsets":[0,24]}}),
            &(0u8..24).collect::<Vec<_>>(),
        );
        let files = ModelTensorFiles::open(&d.0).unwrap();
        let requests = vec![("w".into(), 2, 1), ("w".into(), 0, 1)];
        let rows = files.read_rows_batch(&requests, 16).unwrap();
        assert_eq!(rows[0].data, (16u8..24).collect::<Vec<_>>());
        assert_eq!(rows[1].data, (0u8..8).collect::<Vec<_>>());
        assert!(files.read_rows_batch(&requests, 15).is_err());
        std::fs::write(d.0.join("model.safetensors"), [0u8; 32]).unwrap();
        assert!(files.read_rows_batch(&requests, 16).is_err());
    }

    #[test]
    fn rejects_invalid_ranges_shapes_dtypes_and_trailing_payload() {
        let d = Directory::new();
        for header in [
            serde_json::json!({"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}),
            serde_json::json!({"w":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}),
            serde_json::json!({"w":{"dtype":"BAD","shape":[4],"data_offsets":[0,4]}}),
            serde_json::json!({"w":{"dtype":"F32","shape":[u64::MAX,8],"data_offsets":[0,4]}}),
            serde_json::json!({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
        ] {
            assert!(TensorFile::open(&d.file(header, &[0; 4])).is_err());
        }
        let path = d.file(
            serde_json::json!({"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}),
            &[0; 8],
        );
        assert!(TensorFile::open(&path).is_err());
    }

    #[test]
    fn source_changes_are_rejected_before_returning_rows() {
        let d = Directory::new();
        let path = d.file(
            serde_json::json!({"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}),
            &[1, 2, 3, 4],
        );
        let f = TensorFile::open(&path).unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .unwrap()
            .write_all(&[5])
            .unwrap();
        assert!(f.read_rows("w", 0, 4, 4).is_err());
    }

    #[test]
    fn concurrent_reads_do_not_share_a_seek_cursor() {
        let d = Directory::new();
        let path = d.file(
            serde_json::json!({"w":{"dtype":"U8","shape":[8,4],"data_offsets":[0,32]}}),
            &(0u8..32).collect::<Vec<_>>(),
        );
        let file = Arc::new(TensorFile::open(&path).unwrap());
        std::thread::scope(|scope| {
            for row in 0..8 {
                let file = Arc::clone(&file);
                scope.spawn(move || {
                    for _ in 0..100 {
                        assert_eq!(
                            file.read_rows("w", row, 1, 4).unwrap(),
                            ((row * 4) as u8..(row * 4 + 4) as u8).collect::<Vec<_>>()
                        );
                    }
                });
            }
        });
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn materialized_rows_preserve_bits_dtype_and_cross_thread_lifetime() {
        let d = Directory::new();
        let bits = [0x3f80u16, 0xc000, 0x3f00, 0x4080];
        let bytes: Vec<u8> = bits.into_iter().flat_map(u16::to_le_bytes).collect();
        let path = d.file(
            serde_json::json!({"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}),
            &bytes,
        );
        let source = Arc::new(TensorFile::open(&path).unwrap());
        assert!(source.read_rows_array("w", 0, 2, 0).is_err());
        let array = std::thread::spawn(move || source.read_rows_array("w", 1, 1, 4096).unwrap())
            .join()
            .unwrap();
        assert_eq!(array.dtype(), mlx::Dtype::Bfloat16);
        assert_eq!(array.shape().as_slice(), &[1, 2]);
        assert_eq!(
            array
                .astype(mlx::Dtype::Float32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![0.5, 4.0]
        );
        let word = 0xfedcba98u32;
        let path = d.file(
            serde_json::json!({"w":{"dtype":"U32","shape":[1,1],"data_offsets":[0,4]}}),
            &word.to_le_bytes(),
        );
        let source = TensorFile::open(&path).unwrap();
        assert_eq!(
            source
                .read_rows_array("w", 0, 1, 4096)
                .unwrap()
                .to_vec::<u32>()
                .unwrap(),
            vec![word]
        );
    }

    #[test]
    fn checkpoint_index_rejects_missing_entries_and_escaped_paths() {
        let d = Directory::new();
        d.file(
            serde_json::json!({"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}),
            &[1, 2, 3, 4],
        );
        let index = d.0.join("model.safetensors.index.json");
        for map in [
            serde_json::json!({"w":"../outside.safetensors"}),
            serde_json::json!({"missing":"model.safetensors"}),
        ] {
            std::fs::write(
                &index,
                serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap(),
            )
            .unwrap();
            assert!(ModelTensorFiles::open(&d.0).is_err());
        }
        std::fs::write(&index, br#"{"weight_map":{"w":"model.safetensors"}}"#).unwrap();
        let model = ModelTensorFiles::open(&d.0).unwrap();
        assert_eq!(model.read_rows("w", 1, 2, 2).unwrap(), vec![2, 3]);
        assert_eq!(model.tensors().count(), 1);
    }
}

/// CPU bytes plus validated metadata; conversion happens on the GPU owner thread.
pub struct RawRows {
    info: TensorInfo,
    pub data: Vec<u8>,
}
impl RawRows {
    pub fn into_array(self, staging_budget: usize) -> Result<mlx::Array> {
        let mut header = serde_json::to_vec(
            &serde_json::json!({"rows":{"dtype":self.info.dtype,"shape":self.info.shape,"data_offsets":[0,self.data.len()]}}),
        )?;
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        ensure!(
            (self.data.len() + header.len() + 8)
                .checked_mul(3)
                .is_some_and(|n| n <= staging_budget),
            "raw decoder exceeds staging budget"
        );
        let mut envelope = Vec::with_capacity(8 + header.len() + self.data.len());
        envelope.extend_from_slice(&(header.len() as u64).to_le_bytes());
        envelope.extend(header);
        envelope.extend(self.data);
        let mut reader = mlx::io::Reader::from_bytes(&envelope);
        let (mut arrays, _) = mlx::io::load_safetensors_from_reader(&mut reader)?;
        let array = arrays.remove("rows").context("decoder returned no rows")?;
        array.eval()?;
        Ok(array)
    }
}
type IoJob = Box<dyn FnOnce() + Send + 'static>;
fn reader_count(requested: Option<&str>) -> Result<usize> {
    match requested {
        None | Some("2") => Ok(2),
        Some("1") => Ok(1),
        _ => anyhow::bail!("IRONMLX_FLASH_IO_WORKERS must be 1 or 2"),
    }
}

pub fn io_task<T: Send + 'static>(
    task: impl FnOnce() -> Result<T> + Send + 'static,
) -> mpsc::Receiver<Result<T>> {
    static POOL: OnceLock<std::result::Result<mpsc::SyncSender<IoJob>, String>> = OnceLock::new();
    let pool = POOL.get_or_init(|| {
        let workers = reader_count(std::env::var("IRONMLX_FLASH_IO_WORKERS").ok().as_deref())
            .map_err(|e| e.to_string())?;
        let (sender, receiver) = mpsc::sync_channel::<IoJob>(8);
        let receiver = Arc::new(Mutex::new(receiver));
        for index in 0..workers {
            let receiver = Arc::clone(&receiver);
            std::thread::Builder::new()
                .name(format!("paging-reader-{index}"))
                .spawn(move || loop {
                    let job = receiver.lock().expect("paging reader lock").recv();
                    match job {
                        Ok(job) => job(),
                        Err(_) => break,
                    }
                })
                .map_err(|e| format!("start bounded paging reader: {e}"))?;
        }
        Ok(sender)
    });
    let (sender, receiver) = mpsc::sync_channel(1);
    match pool {
        Ok(pool) => {
            let _ = pool.send(Box::new(move || {
                let _ = sender.send(task());
            }));
        }
        Err(error) => {
            let _ = sender.send(Err(anyhow!(error.clone())));
        }
    }
    receiver
}

#[cfg(test)]
mod reader_config_tests {
    use super::reader_count;

    #[test]
    fn diagnostic_reader_count_is_bounded() {
        assert_eq!(reader_count(None).unwrap(), 2);
        assert_eq!(reader_count(Some("1")).unwrap(), 1);
        assert_eq!(reader_count(Some("2")).unwrap(), 2);
        assert!(reader_count(Some("0")).is_err());
        assert!(reader_count(Some("3")).is_err());
    }
}
