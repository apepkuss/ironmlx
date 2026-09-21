//! Storage operations required by model-side paged KV state.
//! Keys are opaque locations supplied by the store. The model cache owns page
//! references and tensor computation; the injected backend owns IO and cleanup.

use crate::Result;
use mlx::Array;
use std::path::{Path, PathBuf};

pub trait KvPageStoreFactory: std::fmt::Debug + Send + Sync {
    /// Create a private store for one cache, including when configuration is cloned.
    fn open(&self) -> Result<Box<dyn KvPageStore>>;
}

pub trait KvPageStore: std::fmt::Debug + Send {
    fn location(&self) -> PathBuf;
    fn segment_path(&self, start_page: i32, page_count: i32) -> PathBuf;
    fn save(&self, path: &Path, k_pages: &Array, v_pages: &Array) -> Result<()>;
    /// Preserve lazy MLX loading; callers retain their existing evaluation points.
    fn load(&self, path: &Path) -> Result<(Array, Array)>;
    fn size(&self, path: &Path) -> usize;
    fn remove(&self, path: &Path) -> Result<()>;
    fn reset(&self) -> Result<()>;
}
