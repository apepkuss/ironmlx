use anyhow::{Context, Result};

/// Conservative local identity: config/index contents and checkpoint file
/// identities. Replacing or moving weights invalidates reuse even if the HF
/// name stays unchanged, without hashing an entire 112 GB checkpoint at load.
pub fn artifact_identity(dir: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    let mut hash = Sha256::new();
    hash.update(b"ironmlx-artifact-v1");
    hash.update(env!("CARGO_PKG_VERSION").as_bytes());
    // Rebuilding after changing the pinned backend must not reuse a prefix
    // computed under an older numerical contract.
    hash.update(include_bytes!("../../../scripts/release-config.sh"));
    let mut paths = std::fs::read_dir(dir)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    paths.sort();
    for path in paths {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("non UTF-8 model filename")?;
        if name == "config.json" || name.ends_with(".index.json") {
            hash.update(name.as_bytes());
            hash.update(std::fs::read(&path)?);
        } else if path.extension().is_some_and(|e| e == "safetensors") {
            let meta = path.metadata()?;
            hash.update(path.canonicalize()?.as_os_str().as_encoded_bytes());
            hash.update(meta.len().to_le_bytes());
            #[cfg(unix)]
            for value in [
                meta.dev(),
                meta.ino(),
                meta.mtime() as u64,
                meta.mtime_nsec() as u64,
                meta.ctime() as u64,
                meta.ctime_nsec() as u64,
            ] {
                hash.update(value.to_le_bytes());
            }
            #[cfg(not(unix))]
            hash.update(format!("{:?}", meta.modified()?).as_bytes());
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn changing_config_or_replacing_checkpoint_invalidates_identity() {
        let dir = std::env::temp_dir().join(format!(
            "ironmlx-artifact-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{\"bits\":4}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"fixture-a").unwrap();
        let first = artifact_identity(&dir).unwrap();
        assert_eq!(first, artifact_identity(&dir).unwrap());
        std::fs::write(dir.join("config.json"), b"{\"bits\":8}").unwrap();
        let second = artifact_identity(&dir).unwrap();
        assert_ne!(first, second);
        std::fs::write(dir.join("replacement"), b"fixture-b").unwrap();
        std::fs::rename(dir.join("replacement"), dir.join("model.safetensors")).unwrap();
        assert_ne!(second, artifact_identity(&dir).unwrap());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
