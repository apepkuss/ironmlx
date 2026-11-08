//! Process-wide IronMLX data root.
//!
//! Every persistent backend path (scheduler profiles, speculative
//! qualification profiles, voices, prefix and offload caches) lives under one
//! root, `~/.ironmlx` by default. `ironmlx --data-root <DIR>` replaces it for
//! the whole process, so the App can run the backend against an isolated
//! directory. The backend instance lock is deliberately not derived from it.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};

/// Directory name of the default data root under the home directory.
pub const DEFAULT_DATA_ROOT_DIR_NAME: &str = ".ironmlx";

static CONFIGURED_DATA_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// Sets the data root for this process. Call once at startup, before any path
/// is resolved. Setting the same root twice is allowed; a different one is an
/// error.
pub fn configure_data_root(root: PathBuf) -> Result<()> {
    if !root.is_absolute() {
        bail!("data root must be an absolute path: {}", root.display());
    }
    let stored = CONFIGURED_DATA_ROOT.get_or_init(|| root.clone());
    if *stored != root {
        bail!(
            "data root already configured as {}; refusing {}",
            stored.display(),
            root.display()
        );
    }
    Ok(())
}

/// The root set by [`configure_data_root`], if any.
pub fn configured_data_root() -> Option<&'static Path> {
    CONFIGURED_DATA_ROOT.get().map(PathBuf::as_path)
}

/// `configured` when set, otherwise `<home>/.ironmlx`.
pub fn resolve_data_root(configured: Option<&Path>, home: Option<&Path>) -> Option<PathBuf> {
    match configured {
        Some(root) => Some(root.to_path_buf()),
        None => home.map(|home| home.join(DEFAULT_DATA_ROOT_DIR_NAME)),
    }
}

/// The data root of this process.
pub fn data_root() -> Result<PathBuf> {
    resolve_data_root(configured_data_root(), dirs::home_dir().as_deref())
        .context("locating home directory for the IronMLX data root")
}

/// The data root of this process, or `./.ironmlx` when the home directory is
/// unknown.
pub fn data_root_or_current_dir() -> PathBuf {
    resolve_data_root(
        configured_data_root(),
        Some(dirs::home_dir().as_deref().unwrap_or(Path::new("."))),
    )
    .unwrap_or_else(|| PathBuf::from(".").join(DEFAULT_DATA_ROOT_DIR_NAME))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_data_root_is_dot_ironmlx_under_home() {
        assert_eq!(
            resolve_data_root(None, Some(Path::new("/Users/tester"))),
            Some(PathBuf::from("/Users/tester/.ironmlx"))
        );
        assert_eq!(resolve_data_root(None, None), None);
    }

    #[test]
    fn configured_data_root_replaces_home_default() {
        assert_eq!(
            resolve_data_root(
                Some(Path::new("/tmp/ironmlx-data")),
                Some(Path::new("/Users/tester"))
            ),
            Some(PathBuf::from("/tmp/ironmlx-data"))
        );
        assert_eq!(
            resolve_data_root(Some(Path::new("/tmp/ironmlx-data")), None),
            Some(PathBuf::from("/tmp/ironmlx-data"))
        );
    }

    #[test]
    fn configure_rejects_relative_roots() {
        assert!(configure_data_root(PathBuf::from("relative/root")).is_err());
    }
}
