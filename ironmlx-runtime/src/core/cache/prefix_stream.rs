//! Byte-owned snapshots and bounded safetensors writes. The disk worker never
//! owns MLX arrays, lazy graphs, model instances, or borrowed request state.
use anyhow::{ensure, Context, Result};
use ironmlx_lm::core::cache::prefix_payload::{
    PagedPrefixEntry, PagedPrefixKeySpec, PrefixTensorSpec,
};
use mlx::{Array, Dtype};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs::File, io::Write, path::Path, sync::Arc};

pub(crate) struct FrozenTensor {
    name: String,
    spec: PrefixTensorSpec,
    bytes: Vec<u8>,
}
pub(crate) struct FrozenPrefix {
    pub spec: PagedPrefixKeySpec,
    tensors: Vec<FrozenTensor>,
}
/// Called between bounded writes, before sync and before publication. A final
/// OS fsync itself is not preemptible.
pub(crate) struct WriteControl<'a> {
    pub chunk_bytes: usize,
    pub checkpoint: &'a dyn Fn(usize) -> Result<()>,
}
impl WriteControl<'_> {
    pub fn sync() -> Self {
        Self {
            chunk_bytes: 8 << 20,
            checkpoint: &|_| Ok(()),
        }
    }
    pub fn check(&self, bytes: usize) -> Result<()> {
        (self.checkpoint)(bytes)
    }
}
impl FrozenPrefix {
    pub fn capture(
        spec: PagedPrefixKeySpec,
        entry: &PagedPrefixEntry,
        control: &WriteControl<'_>,
    ) -> Result<Arc<Self>> {
        let chunk_bytes = control.chunk_bytes;
        control.check(0)?;
        let arrays = super::prefix_store::entry_tensors(entry);
        let mut tensors = Vec::with_capacity(arrays.len());
        for (name, array) in sorted(arrays) {
            control.check(0)?;
            let compact = array.contiguous(false)?;
            compact.eval()?;
            let tensor_spec = PrefixTensorSpec::from_array(&compact);
            let len = compact
                .size()
                .checked_mul(compact.dtype().byte_size())
                .context("snapshot size overflow")?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(len)?;
            bytes.resize(len, 0);
            for (i, chunk) in bytes.chunks_mut(chunk_bytes.max(1)).enumerate() {
                control.check(chunk.len())?;
                compact.read_bytes_into(i * chunk_bytes.max(1), chunk)?;
            }
            tensors.push(FrozenTensor {
                name,
                spec: tensor_spec,
                bytes,
            });
        }
        ensure!(
            tensors.iter().map(|t| t.bytes.len()).sum::<usize>() == spec.payload_bytes(),
            "frozen prefix payload mismatch"
        );
        Ok(Arc::new(Self { spec, tensors }))
    }
    pub fn thaw(&self) -> Result<PagedPrefixEntry> {
        let mut arrays = HashMap::new();
        for t in &self.tensors {
            arrays.insert(
                t.name.clone(),
                Array::from_bytes(&t.bytes, &t.spec.shape, t.spec.dtype)?,
            );
        }
        super::prefix_store::entry_from_tensors(&self.spec, &mut arrays)
    }
    pub fn write(&self, path: &Path, control: &WriteControl<'_>) -> Result<String> {
        let views: Vec<_> = self
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), &t.spec, Source::Bytes(t.bytes.as_slice())))
            .collect();
        write_tensors(path, &self.spec, &views, control)
    }
}
fn sorted(arrays: HashMap<String, Array>) -> Vec<(String, Array)> {
    let mut tensors: Vec<_> = arrays.into_iter().collect();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    tensors
}
enum Source<'a> {
    Array(&'a Array),
    Bytes(&'a [u8]),
}
pub(crate) fn write_entry(
    path: &Path,
    spec: &PagedPrefixKeySpec,
    entry: &PagedPrefixEntry,
    control: &WriteControl<'_>,
) -> Result<String> {
    let tensors = sorted(super::prefix_store::entry_tensors(entry));
    let specs: Vec<_> = tensors
        .iter()
        .map(|(_, a)| PrefixTensorSpec::from_array(a))
        .collect();
    let views: Vec<_> = tensors
        .iter()
        .zip(&specs)
        .map(|((n, a), s)| (n.as_str(), s, Source::Array(a)))
        .collect();
    write_tensors(path, spec, &views, control)
}
fn dtype_name(dtype: Dtype) -> Result<&'static str> {
    Ok(match dtype {
        Dtype::Bool => "BOOL",
        Dtype::Uint8 => "U8",
        Dtype::Uint16 => "U16",
        Dtype::Uint32 => "U32",
        Dtype::Uint64 => "U64",
        Dtype::Int8 => "I8",
        Dtype::Int16 => "I16",
        Dtype::Int32 => "I32",
        Dtype::Int64 => "I64",
        Dtype::Float16 => "F16",
        Dtype::Bfloat16 => "BF16",
        Dtype::Float32 => "F32",
        Dtype::Float64 => "F64",
        _ => anyhow::bail!("unsupported streamed prefix dtype {dtype}"),
    })
}
fn write_tensors(
    path: &Path,
    spec: &PagedPrefixKeySpec,
    tensors: &[(&str, &PrefixTensorSpec, Source<'_>)],
    control: &WriteControl<'_>,
) -> Result<String> {
    ensure!(
        cfg!(target_endian = "little"),
        "safetensors streaming requires little-endian host"
    );
    ensure!(control.chunk_bytes > 0, "empty prefix write buffer");
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), serde_json::json!({"ironmlx.prefix_cache.key":super::prefix_store::PagedPrefixStore::key_for(spec),"ironmlx.prefix_cache.schema_version":super::prefix_store::SCHEMA_VERSION.to_string()}));
    let mut offset = 0usize;
    for (name, tensor, _) in tensors {
        let size = tensor
            .shape
            .iter()
            .try_fold(tensor.dtype.byte_size(), |n, &dim| {
                n.checked_mul(usize::try_from(dim).ok()?)
            })
            .context("prefix tensor size overflow")?;
        let end = offset.checked_add(size).context("prefix offset overflow")?;
        header.insert((*name).into(), serde_json::json!({"dtype":dtype_name(tensor.dtype)?,"shape":tensor.shape,"data_offsets":[offset,end]}));
        offset = end;
    }
    let mut header = serde_json::to_vec(&header)?;
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    control.check(0)?;
    let mut file = File::create(path)?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        ensure!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } != -1,
            "prefix writer F_NOCACHE: {}",
            std::io::Error::last_os_error()
        );
    }
    let mut hash = Sha256::new();
    let mut write = |bytes: &[u8]| -> Result<()> {
        control.check(bytes.len())?;
        file.write_all(bytes)?;
        hash.update(bytes);
        Ok(())
    };
    write(&(header.len() as u64).to_le_bytes())?;
    write(&header)?;
    // One reusable IO buffer, never a full-record CPU serialization buffer.
    let mut scratch = if tensors
        .iter()
        .any(|(_, _, s)| matches!(s, Source::Array(_)))
    {
        vec![0u8; control.chunk_bytes.min(offset.max(1))]
    } else {
        Vec::new()
    };
    for (_, _, source) in tensors {
        match source {
            Source::Bytes(bytes) => {
                for chunk in bytes.chunks(control.chunk_bytes) {
                    write(chunk)?;
                }
            }
            Source::Array(array) => {
                control.check(0)?;
                let compact = array.contiguous(false)?;
                compact.eval()?;
                let len = compact
                    .size()
                    .checked_mul(compact.dtype().byte_size())
                    .context("prefix tensor bytes overflow")?;
                for start in (0..len).step_by(scratch.len()) {
                    let n = (len - start).min(scratch.len());
                    compact.read_bytes_into(start, &mut scratch[..n])?;
                    write(&scratch[..n])?;
                }
            }
        }
    }
    control.check(0)?;
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::super::prefix_store::{flash_durable_tests::fixture, PagedPrefixLoadStatus};
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn frozen_snapshot_is_independent_streamed_and_loadable_by_existing_codec() {
        let root = std::env::temp_dir().join(format!("prefix-frozen-{}", uuid::Uuid::new_v4()));
        let (store, spec, entry) = fixture(&root);
        let control = WriteControl {
            chunk_bytes: 13,
            checkpoint: &|_| Ok(()),
        };
        let frozen = FrozenPrefix::capture(spec.clone(), &entry, &control).unwrap();
        drop(entry);
        let thawed = frozen.thaw().unwrap();
        assert_eq!(thawed.main_layer_specs(), spec.main_layers);
        let key = store.save_frozen(&frozen, &control).unwrap();
        assert_eq!(
            store.load_observed(&spec).unwrap().status,
            PagedPrefixLoadStatus::Hit
        );
        let (raw, _) =
            mlx::io::load_safetensors(root.join(key).join("payload.safetensors").to_str().unwrap())
                .unwrap();
        let expected = super::super::prefix_store::entry_tensors(&thawed);
        for (name, a) in raw {
            let b = &expected[&name];
            assert_eq!(a.dtype(), b.dtype());
            assert_eq!(a.shape(), b.shape());
            let len = a.size() * a.dtype().byte_size();
            let mut left = vec![0; len];
            let mut right = vec![0; len];
            a.eval().unwrap();
            b.eval().unwrap();
            a.read_bytes_into(0, &mut left).unwrap();
            b.read_bytes_into(0, &mut right).unwrap();
            assert_eq!(left, right);
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn cancellation_cleans_staging_and_never_publishes_a_partial_prefix() {
        let root = std::env::temp_dir().join(format!("prefix-cancel-{}", uuid::Uuid::new_v4()));
        let (store, spec, entry) = fixture(&root);
        let frozen = FrozenPrefix::capture(spec.clone(), &entry, &WriteControl::sync()).unwrap();
        let writes = AtomicUsize::new(0);
        let check = |_: usize| -> Result<()> {
            anyhow::ensure!(writes.fetch_add(1, Ordering::Relaxed) < 6, "test_cancel");
            Ok(())
        };
        assert!(store
            .save_frozen(
                &frozen,
                &WriteControl {
                    chunk_bytes: 13,
                    checkpoint: &check
                }
            )
            .is_err());
        assert!(!store.contains_persisted(&spec).unwrap());
        assert!(!std::fs::read_dir(&root).unwrap().any(|e| e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".tmp-")));
        store.save_frozen(&frozen, &WriteControl::sync()).unwrap();
        assert!(store.contains_persisted(&spec).unwrap());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn capture_cancellation_stops_before_copying_the_next_tensor() {
        let root = std::env::temp_dir().join("unused-prefix-capture");
        let (_, spec, entry) = fixture(&root);
        assert!(FrozenPrefix::capture(
            spec,
            &entry,
            &WriteControl {
                chunk_bytes: 8,
                checkpoint: &|_| anyhow::bail!("cancel")
            }
        )
        .is_err());
    }
}
