//! Real-checkpoint smoke for mlx-community/gemma-4-e4b-it-OptiQ-4bit.
//!
//! Skipped unless IRONMLX_TEST_REAL_OPTIQ=1 because the checkpoint is large.

use std::path::PathBuf;

use ironmlx::core::QuantMode;
use ironmlx::models::Gemma4Model;
use ironmlx::Loader;

fn snapshot_dir() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let base =
        home.join(".ironmlx/models/models--mlx-community--gemma-4-e4b-it-OptiQ-4bit/snapshots");
    let entries = std::fs::read_dir(&base).ok()?;
    for entry in entries.flatten() {
        if entry.path().is_dir() {
            return Some(entry.path());
        }
    }
    None
}

#[test]
fn load_gemma4_optiq_real_model_when_requested() {
    if std::env::var_os("IRONMLX_TEST_REAL_OPTIQ").as_deref() != Some("1".as_ref()) {
        eprintln!("IRONMLX_TEST_REAL_OPTIQ=1 not set; skipping real OptiQ checkpoint smoke");
        return;
    }
    let Some(dir) = snapshot_dir() else {
        eprintln!("gemma-4-e4b-it-OptiQ-4bit cache absent; skipping");
        return;
    };

    let loader = Loader::open(&dir).expect("Loader::open OptiQ");
    let global = loader.quant_meta().expect("global OptiQ quant meta");
    assert_eq!(global.mode, QuantMode::OptiQ);
    assert_eq!(global.bits, 4);
    assert_eq!(global.group_size, 64);

    let q = loader
        .quant_meta_for("model.layers.19.self_attn.q_proj")
        .expect("layer 19 q_proj OptiQ meta");
    let k = loader
        .quant_meta_for("model.layers.19.self_attn.k_proj")
        .expect("layer 19 k_proj OptiQ meta");
    let v = loader
        .quant_meta_for("model.layers.19.self_attn.v_proj")
        .expect("layer 19 v_proj OptiQ meta");
    assert_eq!(q.mode, QuantMode::OptiQ);
    assert_eq!(k.mode, QuantMode::OptiQ);
    assert_eq!(v.mode, QuantMode::OptiQ);
    assert_eq!(q.bits, 4);
    assert_eq!(k.bits, 8);
    assert_eq!(v.bits, 8);

    let gate = loader
        .quant_meta_for("model.layers.1.mlp.gate_proj")
        .expect("layer 1 gate_proj OptiQ meta");
    let up = loader
        .quant_meta_for("model.layers.1.mlp.up_proj")
        .expect("layer 1 up_proj OptiQ meta");
    assert_eq!(gate.mode, QuantMode::OptiQ);
    assert_eq!(up.mode, QuantMode::OptiQ);
    assert_eq!(gate.bits, 8);
    assert_eq!(up.bits, 4);

    let _model = Gemma4Model::from_loader(&loader).expect("Gemma4Model::from_loader OptiQ");
}
