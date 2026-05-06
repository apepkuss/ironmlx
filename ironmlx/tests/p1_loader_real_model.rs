//! Integration test — exercises the Loader against the on-disk
//! Qwen3.5-4B-MLX-4bit checkpoint. Skipped if the model directory is
//! absent (e.g. in CI without the cache).

use std::path::PathBuf;

use ironmlx::Loader;

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
fn load_qwen35_4b_mlx_4bit() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("model dir absent — skipping");
        return;
    };

    let loader = Loader::open(&dir).expect("open loader");

    // Quantization metadata
    let q = loader.quant_meta().expect("quant present");
    assert_eq!(q.bits, 4);
    assert_eq!(q.group_size, 64);

    // Spot-check key presence
    assert!(loader.contains("language_model.model.embed_tokens.weight"));
    assert!(loader.contains("language_model.model.embed_tokens.scales"));
    assert!(loader.contains("language_model.model.layers.3.self_attn.q_proj.weight"));
    assert!(loader.contains("language_model.model.layers.3.self_attn.q_proj.scales"));

    // Linear attention layer 0 keys
    assert!(loader.contains("language_model.model.layers.0.linear_attn.A_log"));
    assert!(loader.contains("language_model.model.layers.0.linear_attn.conv1d.weight"));

    // Norm weights are not quantized
    assert!(loader.contains("language_model.model.layers.3.input_layernorm.weight"));
    assert!(!loader.contains("language_model.model.layers.3.input_layernorm.scales"));

    // Final norm
    assert!(loader.contains("language_model.model.norm.weight"));

    // No standalone lm_head — tied embedding
    assert!(!loader.contains("language_model.lm_head.weight"));
}
