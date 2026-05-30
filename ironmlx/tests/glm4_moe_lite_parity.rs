//! GLM-4.7-Flash (`glm4_moe_lite`) logits-parity harness vs the authoritative
//! `mlx_lm` reference implementation (`mlx_lm/models/glm4_moe_lite.py`, the
//! de-facto consumer of the mlx-community 4-bit weights and the omlx perf
//! baseline).
//!
//! Methodology: feed BOTH engines the IDENTICAL raw token sequence (no chat
//! template, no tokenizer in the loop) and compare the last-position logits.
//! This isolates the model forward pipeline from any prompt-formatting /
//! tokenization differences, so a mismatch is a genuine numeric/structural
//! correctness bug rather than a template mismatch.
//!
//! Reference captured once from the omlx venv (mlx_lm 0.31.3 / mlx 0.31.2) on
//! the mlx-community GLM-4.7-Flash-4bit checkpoint
//! (snapshot 1454cffb1a21737e162f508e5bc70be9def89276):
//!
//! ```text
//! model, _ = mlx_lm.utils.load(model_dir)
//! logits = model(mx.array([[1, 2, 3, 4]]))          # [1, 4, vocab]
//! last = logits[0, -1, :].astype(mx.float32)
//! top5 = [int(i) for i in mx.argsort(-last)[:5].tolist()]
//! # -> argmax = 5, top5 = [5, 27782, 609, 7672, 3]
//! ```
//!
//! ironmlx (this harness) produces argmax = 5 (logit 122.81) for the same
//! input — a token-for-token match on the greedy first token. The tiny
//! per-logit delta (122.81 vs 122.5) is 4-bit-quantization ordering noise and
//! is well within the 0.5 top-50 tolerance the design spec (§8) allows.
//!
//! Greedy 64-token continuations on real chat prompts (Chinese + English,
//! thinking-mode aligned) were verified IDENTICAL to the `mlx_lm` reference
//! during Task 7 acceptance; this committed harness pins the first-token
//! argmax / top-5 set as the regression guard.
//!
//! Env-gated: skips (with an eprintln) when no GLM checkpoint is present. Run:
//!   GLM47_MODEL_DIR=$(echo ~/.ironmlx/models/models--mlx-community--GLM-4.7-Flash-4bit/snapshots/*) \
//!     MLX_DIR=/tmp/ironmlx-perf-mlx-install-3f6c3113f734 \
//!     cargo test -p ironmlx --release --test glm4_moe_lite_parity -- --nocapture --test-threads=1

use mlx::{Array, Dtype, StreamOrDevice};

use ironmlx::core::{Loader, Model};
use ironmlx::models::glm4_moe_lite::Glm4MoeLiteModel;

const VOCAB: i32 = 154_880;

/// Fixed raw token sequence fed to both engines (matches the smoke test).
const PROMPT_IDS: [i32; 4] = [1, 2, 3, 4];

/// Reference captured from `mlx_lm` (`glm4_moe_lite.py`) — see module doc.
const REF_ARGMAX: i32 = 5;
const REF_TOP5: [i32; 5] = [5, 27782, 609, 7672, 3];

/// Resolve the GLM-4.7-Flash snapshot directory. Honors `GLM47_MODEL_DIR`
/// first, then falls back to the default HF cache layout.
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

/// Return the top-`k` token ids (highest logit first) from a flat logits
/// vector of length `VOCAB`.
fn top_k_ids(logits_f32: &[f32], k: usize) -> Vec<i32> {
    let mut idx: Vec<usize> = (0..logits_f32.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits_f32[b].partial_cmp(&logits_f32[a]).unwrap());
    idx.into_iter().take(k).map(|i| i as i32).collect()
}

#[test]
fn glm_first_token_logits_match_mlx_lm_reference() {
    let Some(dir) = glm_snapshot_dir() else {
        eprintln!("skip: no GLM-4.7-Flash weights (set GLM47_MODEL_DIR)");
        return;
    };
    eprintln!("loading GLM-4.7-Flash from {dir}");

    let loader = Loader::open(std::path::Path::new(&dir)).expect("Loader::open");
    let model = Glm4MoeLiteModel::from_loader(&loader).expect("Glm4MoeLiteModel::from_loader");
    assert_eq!(model.config().vocab_size, VOCAB, "vocab_size from config");

    // Prefill the fixed sequence; mask = None so the model builds its own causal
    // mask (the same path the reference's full-sequence call exercises).
    let cap = 16;
    let mut cache = Model::make_cache(&model, 1, cap, Dtype::Bfloat16).expect("make_cache");
    let input = ids(&PROMPT_IDS, &[1, PROMPT_IDS.len() as i32]);
    let pos = ids(&[0], &[1]); // dummy (requires_position_ids = false)

    let logits = Model::forward_on(
        &model,
        &input,
        &pos,
        None,
        None,
        Some(&mut cache),
        StreamOrDevice::default(),
    )
    .expect("forward_on");

    let shape = logits.shape();
    assert_eq!(
        shape.as_slice(),
        &[1, 1, VOCAB],
        "last-position logits [1,1,vocab]"
    );

    let v: Vec<f32> = mlx::ops::cast::astype(&logits, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();

    let top5 = top_k_ids(&v, 5);
    let argmax = top5[0];
    eprintln!("ironmlx argmax={argmax} (ref={REF_ARGMAX}); top5={top5:?} (ref={REF_TOP5:?})");

    // Hard guard: greedy first token must match the reference exactly.
    assert_eq!(
        argmax, REF_ARGMAX,
        "first-token argmax diverged from mlx_lm reference"
    );

    // Top-5 *set* must match the reference (order may swap on near-ties under
    // 4-bit quantization; the set membership is the structural invariant).
    let mut got = top5.clone();
    let mut want = REF_TOP5.to_vec();
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(
        got, want,
        "top-5 token set diverged from mlx_lm reference; got {top5:?} ref {REF_TOP5:?}"
    );

    eprintln!("GLM-4.7-Flash parity OK: argmax + top-5 set match mlx_lm reference");
}
