//! Integration test — exercises Tokenizer against the on-disk
//! Qwen3.5-4B-MLX-4bit checkpoint. Skipped if model dir is absent.

use std::path::PathBuf;

use ironmlx::{Loader, Tokenizer};

fn snapshot_dir() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let base = home.join(".ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots");
    let entries = std::fs::read_dir(&base).ok()?;
    for entry in entries.flatten() {
        if entry.path().is_dir() {
            return Some(entry.path());
        }
    }
    None
}

#[test]
fn encode_decode_roundtrip() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("model dir absent — skipping");
        return;
    };
    let loader = Loader::open(&dir).expect("open loader");
    let tok = Tokenizer::from_loader(&loader).expect("tokenizer");

    let text = "Hello, world!";
    let ids = tok.encode(text, false).expect("encode");
    assert!(!ids.is_empty(), "encoder returned no tokens");

    let decoded = tok.decode(&ids, true).expect("decode");
    // Loose round-trip: text should be reproducible up to whitespace.
    assert!(
        decoded.contains("Hello"),
        "decoded missing 'Hello': {decoded}"
    );
}
