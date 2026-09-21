//! Language-aware normalization and token preparation.
mod tokenizer;
pub use tokenizer::IndexTts25Tokenizer;
mod fst;
mod japanese;
use crate::{AudioError, Result};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};
pub(crate) fn verify_resource(path: &Path, package: &str, member: &str) -> Result<()> {
    let mismatch = |reason: String| AudioError::ResourceMismatch {
        component: member.into(),
        reason,
    };
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../../resources/indextts25/sources.json"))
            .map_err(|e| mismatch(e.to_string()))?;
    let entry = manifest["text_resources"]
        .as_array()
        .and_then(|v| v.iter().find(|r| r["name"] == package))
        .and_then(|r| r["members"].as_array())
        .and_then(|v| v.iter().find(|r| r["path"] == member))
        .ok_or_else(|| mismatch("missing pin".into()))?;
    let mut file = std::fs::File::open(path)?;
    if Some(file.metadata()?.len()) != entry["bytes"].as_u64() {
        return Err(mismatch("size mismatch".into()));
    }
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    if Some(format!("{:x}", hash.finalize()).as_str()) != entry["sha256"].as_str() {
        return Err(mismatch("SHA-256 mismatch".into()));
    }
    Ok(())
}
mod frontend;
mod normalize;
pub use frontend::{resolve_language, IndexTts25TextFrontend, PreparedText, TextLimits};
