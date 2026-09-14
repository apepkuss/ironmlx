//! Filesystem backend for active paged KV segments. Runtime construction injects
//! this backend into the model cache; schema and atomic install remain unchanged.

use crate::Result;
use anyhow::Context;
use ironmlx_lm::core::cache::paged_kv::PagedKvHotColdConfig;
use mlx::Array;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use {
    ironmlx_lm::core::cache::page_storage::KvPageStore,
    ironmlx_lm::core::cache::page_storage::KvPageStoreFactory,
};

#[derive(Debug)]
struct FilePageStoreFactory {
    root: PathBuf,
}
#[derive(Debug)]
struct FilePageStore {
    cache_dir: PathBuf,
}

pub fn file_paged_kv_config(
    root: impl Into<PathBuf>,
    hot_window_pages: i32,
    chunk_pages: i32,
) -> Result<PagedKvHotColdConfig> {
    PagedKvHotColdConfig::new(
        Arc::new(FilePageStoreFactory { root: root.into() }),
        hot_window_pages,
        chunk_pages,
    )
}

impl KvPageStoreFactory for FilePageStoreFactory {
    fn open(&self) -> Result<Box<dyn KvPageStore>> {
        let cache_dir = self
            .root
            .join(format!("cache-{}", uuid::Uuid::new_v4().simple()));
        fs::create_dir_all(&cache_dir)
            .with_context(|| format!("create active KV page cache dir {}", cache_dir.display()))?;
        Ok(Box::new(FilePageStore { cache_dir }))
    }
}
impl KvPageStore for FilePageStore {
    fn location(&self) -> PathBuf {
        self.cache_dir.clone()
    }
    fn segment_path(&self, start_page: i32, page_count: i32) -> PathBuf {
        self.cache_dir.join(format!(
            "pages-{start_page}-{page_count}-{}.safetensors",
            uuid::Uuid::new_v4().simple()
        ))
    }
    fn save(&self, path: &Path, k_pages: &Array, v_pages: &Array) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create active KV page dir {}", parent.display()))?;
        }
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("page");
        let tmp_path = path.with_file_name(format!(
            "{stem}.tmp-{}.safetensors",
            uuid::Uuid::new_v4().simple()
        ));
        let mut tensors = HashMap::new();
        tensors.insert("k".to_owned(), k_pages.clone());
        tensors.insert("v".to_owned(), v_pages.clone());
        let mut metadata = HashMap::new();
        metadata.insert(
            "ironmlx.active_kv.schema".to_owned(),
            "hot_cold_segment_v1".to_owned(),
        );
        let tmp = tmp_path.to_string_lossy().into_owned();
        mlx::io::save_safetensors(&tmp, &tensors, &metadata)
            .with_context(|| format!("save active KV page {}", tmp_path.display()))?;
        fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "install active KV page {} -> {}",
                tmp_path.display(),
                path.display()
            )
        })?;
        Ok(())
    }
    fn load(&self, path: &Path) -> Result<(Array, Array)> {
        let path_str = path.to_string_lossy().into_owned();
        let (mut tensors, _) = mlx::io::load_safetensors(&path_str)
            .with_context(|| format!("load active KV page {path_str}"))?;
        let k = tensors
            .remove("k")
            .ok_or_else(|| anyhow::anyhow!("active KV page {} missing tensor k", path.display()))?;
        let v = tensors
            .remove("v")
            .ok_or_else(|| anyhow::anyhow!("active KV page {} missing tensor v", path.display()))?;
        Ok((k, v))
    }
    fn size(&self, path: &Path) -> usize {
        fs::metadata(path)
            .map(|meta| meta.len().min(usize::MAX as u64) as usize)
            .unwrap_or(0)
    }
    fn remove(&self, path: &Path) -> Result<()> {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("remove offloaded active KV page {}", path.display())
                });
            }
        }
        Ok(())
    }
    fn reset(&self) -> Result<()> {
        let _ = fs::remove_dir_all(&self.cache_dir);
        fs::create_dir_all(&self.cache_dir).map_err(Into::into)
    }
}
impl Drop for FilePageStore {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.cache_dir);
    }
}
