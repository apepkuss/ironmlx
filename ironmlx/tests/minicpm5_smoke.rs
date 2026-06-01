//! MiniCPM5-1B (standard `llama` GQA dense) numeric correctness smoke test.
//!
//! Gold standard: feed the EXACT same prompt token ids that mlx_lm's default
//! (no-BOS) `tokenizer.encode` produces, then assert ironmlx's next-token
//! argmax + top-1 logit magnitude agree with mlx_lm's reference forward on the
//! identical ids. This is an apples-to-apples logits comparison (same input,
//! same model weights) — the only sound correctness gate, since greedy *text*
//! comparison is confounded by BOS-prepend protocol differences between
//! engines.
//!
//! Reference (mlx_lm 0.31.x, MiniCPM5-1B-8bit, prompt "The capital of France
//! is", default no-BOS encode → ids [608, 4894, 304, 6918, 357]):
//!   next-token top-5 ids   = [8181, 8266, 285, 280, 9260]
//!   next-token top-5 logits= [11.81, 10.44, 10.44, 10.44, 9.63]
//!   ⇒ argmax id 8181 (" Paris"), clear 1.37-logit margin over #2.
//!
//! Env-gated on `IRONMLX_MINICPM5_MODEL`. Run with:
//! ```bash
//! IRONMLX_MINICPM5_MODEL=$(echo ~/.ironmlx/models/models--mlx-community--MiniCPM5-1B-8bit/snapshots/*) \
//!   MLX_DIR=/tmp/ironmlx-perf-mlx-install-3f6c3113f734 \
//!   cargo test -p ironmlx --release --test minicpm5_smoke -- --nocapture --test-threads=1
//! ```

use mlx::{Array, Dtype, StreamOrDevice};

use ironmlx::core::{Loader, Model};
use ironmlx::models::LlamaModel;
use ironmlx::Tokenizer;

const ENV_MODEL: &str = "IRONMLX_MINICPM5_MODEL";
const PROMPT: &str = "The capital of France is";

// mlx_lm default (no-BOS) tokenization of PROMPT.
const EXPECTED_PROMPT_IDS: &[i32] = &[608, 4894, 304, 6918, 357];
// mlx_lm next-token argmax on those ids (" Paris").
const EXPECTED_ARGMAX_ID: usize = 8181;
// mlx_lm reference top-1 logit; allow a wide bf16/fused-vs-eager tolerance.
const REFERENCE_TOP1_LOGIT: f32 = 11.8125;
const LOGIT_TOLERANCE: f32 = 1.0;

fn snapshot_dir() -> Option<String> {
    if let Ok(p) = std::env::var(ENV_MODEL) {
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
        eprintln!("{ENV_MODEL}={p} does not exist");
        return None;
    }
    None
}

#[test]
fn minicpm5_first_token_matches_mlx_lm_reference() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skip: no MiniCPM5 weights (set {ENV_MODEL})");
        return;
    };
    eprintln!("loading MiniCPM5-1B from {dir}");

    let loader = Loader::open(std::path::Path::new(&dir)).expect("Loader::open");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = LlamaModel::from_loader(&loader).expect("LlamaModel::from_loader");

    let cfg = model.config();
    assert_eq!(cfg.num_hidden_layers, 24, "num_hidden_layers");
    assert_eq!(cfg.num_attention_heads, 16, "num_attention_heads");
    assert_eq!(cfg.num_key_value_heads, 2, "num_key_value_heads");
    assert_eq!(
        cfg.effective_head_dim(),
        128,
        "head_dim (explicit, ≠ 1536/16)"
    );
    assert_eq!(cfg.vocab_size, 130_560, "vocab_size");

    // Match mlx_lm's default (no-BOS) encode for an apples-to-apples compare.
    let prompt_ids = tokenizer.encode(PROMPT, false).expect("encode");
    let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
    eprintln!("ironmlx prompt ids: {prompt_i32:?}");
    assert_eq!(
        prompt_i32, EXPECTED_PROMPT_IDS,
        "tokenization mismatch vs mlx_lm no-BOS reference"
    );

    let n = prompt_i32.len() as i32;
    let input: Array = (prompt_i32.as_slice(), &[1, n][..]).try_into().unwrap();
    let pos: Array = (&[0_i32][..], &[1][..]).try_into().unwrap(); // dummy: requires_position_ids=false

    let mut cache = Model::make_cache(&model, 1, 64, Dtype::Bfloat16).expect("make_cache");
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
    assert_eq!(shape.as_slice(), &[1, 1, 130_560], "logits [1,1,vocab]");

    let v: Vec<f32> = mlx::ops::cast::astype(&logits, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "logits must be finite");

    // Top-5 for the record.
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    eprintln!("ironmlx next-token top-5 (id, logit):");
    for &i in idx.iter().take(5) {
        eprintln!("  id={i} logit={:.4}", v[i]);
    }

    let argmax = idx[0];
    assert_eq!(
        argmax, EXPECTED_ARGMAX_ID,
        "first-token argmax id mismatch vs mlx_lm: got {argmax}, expected {EXPECTED_ARGMAX_ID}"
    );
    let top1 = v[argmax];
    assert!(
        (top1 - REFERENCE_TOP1_LOGIT).abs() <= LOGIT_TOLERANCE,
        "top-1 logit {top1:.4} deviates from mlx_lm reference {REFERENCE_TOP1_LOGIT:.4} by > {LOGIT_TOLERANCE}"
    );

    eprintln!("MiniCPM5 smoke OK: argmax={argmax} (id 8181 = ' Paris'), top1 logit={top1:.4}");
}
