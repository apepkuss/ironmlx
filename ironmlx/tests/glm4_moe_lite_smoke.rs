//! GLM-4.7-Flash (`glm4_moe_lite`) integration smoke test.
//!
//! Loads the REAL mlx-community 4-bit checkpoint and verifies the full forward
//! pipeline (absorbed-MLA attention + noaux_tc MoE + decoder layers + lm_head)
//! produces finite logits of the expected shape — for BOTH the prefill regime
//! (`forward_on` with `mask = None`, exercising the model's internal causal
//! mask) and a follow-on decode step (`forward_on` on a single token,
//! exercising the absorbed-decode regime + the per-row cache offset path the
//! scheduler relies on).
//!
//! Env-gated: skips (with an eprintln) when no GLM checkpoint is present. Run
//! with the model present:
//!   GLM47_MODEL_DIR=$(echo ~/.ironmlx/models/models--mlx-community--GLM-4.7-Flash-4bit/snapshots/*) \
//!     MLX_DIR=/tmp/ironmlx-perf-mlx-install-3f6c3113f734 \
//!     cargo test -p ironmlx --release --test glm4_moe_lite_smoke -- --nocapture --test-threads=1

use mlx::{Array, Dtype, StreamOrDevice};

use ironmlx::core::{Loader, Model};
use ironmlx::models::glm4_moe_lite::Glm4MoeLiteModel;

const VOCAB: i32 = 154_880;

/// Resolve the GLM-4.7-Flash snapshot directory. Honors `GLM47_MODEL_DIR`
/// first, then falls back to the default HF cache layout. Returns `None`
/// (caller skips) when nothing is found.
fn glm_snapshot_dir() -> Option<String> {
    if let Ok(p) = std::env::var("GLM47_MODEL_DIR") {
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
        eprintln!("GLM47_MODEL_DIR={p} does not exist");
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        format!("{home}/.ironmlx/models/models--mlx-community--GLM-4.7-Flash-4bit/snapshots");
    let entries = std::fs::read_dir(&snapshots).ok()?;
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .map(|p| p.to_string_lossy().into_owned())
}

fn ids(d: &[i32], s: &[i32]) -> Array {
    (d, s).try_into().unwrap()
}

fn all_finite(logits: &Array) -> bool {
    let v: Vec<f32> = mlx::ops::cast::astype(logits, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    v.iter().all(|x| x.is_finite())
}

#[test]
fn glm_loads_and_prefill_decode_are_finite() {
    let Some(dir) = glm_snapshot_dir() else {
        eprintln!("skip: no GLM-4.7-Flash weights (set GLM47_MODEL_DIR)");
        return;
    };
    eprintln!("loading GLM-4.7-Flash from {dir}");

    let loader = Loader::open(std::path::Path::new(&dir)).expect("Loader::open");
    let model = Glm4MoeLiteModel::from_loader(&loader).expect("Glm4MoeLiteModel::from_loader");
    let cfg = model.config();
    assert_eq!(cfg.vocab_size, VOCAB, "vocab_size from config");
    eprintln!(
        "loaded: {} layers, first_k_dense_replace={}",
        cfg.num_hidden_layers, cfg.first_k_dense_replace
    );

    // Cap big enough to prefill 4 + decode a few; one-shot preallocated.
    let cap = 16;
    let mut cache = Model::make_cache(&model, 1, cap, Dtype::Bfloat16).expect("make_cache");

    // --- Prefill: 4 tokens, mask=None → model builds its own causal mask. ---
    let input = ids(&[1, 2, 3, 4], &[1, 4]);
    let pos = ids(&[0], &[1]); // dummy (requires_position_ids = false)
    let prefill_logits = Model::forward_on(
        &model,
        &input,
        &pos,
        None,
        None,
        Some(&mut cache),
        StreamOrDevice::default(),
    )
    .expect("prefill forward_on");

    let pshape = prefill_logits.shape();
    let ps = pshape.as_slice();
    eprintln!("prefill logits shape: {ps:?}");
    assert_eq!(ps.len(), 3, "prefill logits rank-3 [B,1,vocab]");
    assert_eq!(ps[0], 1, "B");
    assert_eq!(ps[1], 1, "S (last-position sliced)");
    assert_eq!(ps[2], VOCAB, "vocab dim");
    assert!(
        all_finite(&prefill_logits),
        "prefill logits contain non-finite values"
    );

    // argmax (informational — confirms the distribution is meaningful).
    let pv: Vec<f32> = mlx::ops::cast::astype(&prefill_logits, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    let (argmax, val) = pv
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!("prefill argmax token id: {argmax} (logit={val:.4})");

    // --- Decode: feed one token, exercising the absorbed-decode regime and
    //     the per-row cache offset (cache now holds 4 tokens). ---
    let decode_in = ids(&[argmax as i32], &[1, 1]);
    let decode_logits = Model::forward_on(
        &model,
        &decode_in,
        &pos,
        None,
        None,
        Some(&mut cache),
        StreamOrDevice::default(),
    )
    .expect("decode forward_on");

    let dshape = decode_logits.shape();
    let ds = dshape.as_slice();
    eprintln!("decode logits shape: {ds:?}");
    assert_eq!(ds, &[1, 1, VOCAB], "decode logits [1,1,vocab]");
    assert!(
        all_finite(&decode_logits),
        "decode logits contain non-finite values"
    );

    eprintln!("GLM-4.7-Flash smoke OK: prefill + decode both finite, shape [1,1,{VOCAB}]");
}
