//! P5c batched_prefill MoE equivalence: B=2 batched output should match
//! per-row B=1 single-stream output with argmax bit-identical and
//! max_abs_diff within the bf16 hybrid-linear-attention drift envelope.
//!
//! NOTE on tolerances: Qwen3.5-MoE uses the same hybrid linear-attention
//! path as dense Qwen3.5 (see batched_prefill.rs for dense baseline).
//! That path exhibits ~0.1-1.0 bf16 numerical drift under batch due to GPU
//! kernel reduction scheduling on GatedDeltaNet linear layers.  The logits
//! tolerance is therefore LOGITS_TOL = 1.0 (matching the dense test), NOT
//! 1e-3 (which would be pure bf16 round-off on deterministic fp32 paths).
//! The primary correctness signal is argmax bit-identical, confirming that
//! MoE expert routing + attention masks are batch-position-independent.
//!
//! Run with:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --test qwen35_moe_batched \
//!       -- --ignored --nocapture --test-threads=1

use mlx::Dtype;

/// Logits tolerance matching the dense Qwen3.5 hybrid-linear-attention drift
/// envelope (see batched_prefill.rs: LOGITS_TOL = 1.0).
const LOGITS_TOL: f32 = 1.0;

use ironmlx::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_position_ids,
    build_position_ids_batched,
};
use ironmlx::core::{Loader, Model};
use ironmlx::models::qwen3_5_moe::MIN_KV_CACHE_CAP_FOR_GPU_PERF;
use ironmlx::models::Qwen35MoeModel;

fn locate_snapshot() -> String {
    if let Ok(p) = std::env::var("IRONMLX_MOE_MODEL_DIR") {
        return p;
    }
    let home = std::env::var("HOME").expect("HOME env");
    let glob =
        format!("{home}/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots");
    let entries = std::fs::read_dir(&glob).expect("snapshots dir");
    let first = entries
        .filter_map(|e| e.ok())
        .next()
        .expect("at least one snapshot");
    first.path().to_string_lossy().into_owned()
}

#[test]
#[ignore]
fn p5c_batched_prefill_b2_equals_b1_per_row() {
    let dir = locate_snapshot();
    let loader = Loader::open(std::path::Path::new(&dir)).expect("Loader::open");
    let model = Qwen35MoeModel::from_loader(&loader).expect("Qwen35MoeModel::from_loader");

    let prompt_a: Vec<i32> = vec![100, 200, 300, 400, 500]; // len 5
    let prompt_b: Vec<i32> = vec![600, 700, 800]; // len 3
    let max_len: i32 = 5;
    let prompt_lens: Vec<i32> = vec![prompt_a.len() as i32, prompt_b.len() as i32];

    // === B=1 baseline: prompt_a ===
    let s_a = prompt_a.len() as i32;
    let inp_a: mlx::Array = (&prompt_a[..], &[1_i32, s_a][..]).try_into().unwrap();
    let pos_a = build_position_ids(0, s_a).expect("build_position_ids a");
    let cap_a = (s_a + 4).max(MIN_KV_CACHE_CAP_FOR_GPU_PERF);
    let mut cache_a = Model::make_cache(&model, 1, cap_a, Dtype::Bfloat16).expect("make_cache_a");
    let logits_a = Model::forward_on(
        &model,
        &inp_a,
        &pos_a,
        None,
        None,
        Some(&mut cache_a),
        mlx::StreamOrDevice::default(),
    )
    .expect("forward_on_a");
    // logits_a shape: [1, 1, vocab] — flatten to [vocab]
    let vocab = logits_a.shape().as_slice()[2] as usize;
    let logits_a = logits_a.reshape(&[vocab as i32][..]).expect("reshape_a");

    // === B=1 baseline: prompt_b ===
    let s_b = prompt_b.len() as i32;
    let inp_b: mlx::Array = (&prompt_b[..], &[1_i32, s_b][..]).try_into().unwrap();
    let pos_b = build_position_ids(0, s_b).expect("build_position_ids b");
    let cap_b = (s_b + 4).max(MIN_KV_CACHE_CAP_FOR_GPU_PERF);
    let mut cache_b = Model::make_cache(&model, 1, cap_b, Dtype::Bfloat16).expect("make_cache_b");
    let logits_b = Model::forward_on(
        &model,
        &inp_b,
        &pos_b,
        None,
        None,
        Some(&mut cache_b),
        mlx::StreamOrDevice::default(),
    )
    .expect("forward_on_b");
    let vocab_b = logits_b.shape().as_slice()[2] as usize;
    assert_eq!(vocab_b, vocab, "vocab size must match between prompts");
    let logits_b = logits_b.reshape(&[vocab as i32][..]).expect("reshape_b");

    // === B=2 batched ===
    // Build right-padded [2, max_len] input, padding with token id 0.
    let mut flat: Vec<i32> = vec![0_i32; 2 * max_len as usize];
    flat[..prompt_a.len()].copy_from_slice(&prompt_a);
    let row_b_start = max_len as usize;
    flat[row_b_start..row_b_start + prompt_b.len()].copy_from_slice(&prompt_b);
    let inp_batch: mlx::Array = (&flat[..], &[2_i32, max_len][..]).try_into().unwrap();
    let pos_batch = build_position_ids_batched(&prompt_lens, max_len).unwrap();
    let attn_mask = build_batch_attention_mask(&prompt_lens, max_len, Dtype::Bfloat16).unwrap();
    let lin_mask = build_batch_linear_mask(&prompt_lens, max_len).unwrap();
    let cap_batch = (max_len + 4).max(MIN_KV_CACHE_CAP_FOR_GPU_PERF);
    let mut cache_batch =
        Model::make_cache(&model, 2, cap_batch, Dtype::Bfloat16).expect("make_cache_batch");
    let logits_batch = Model::batched_prefill(
        &model,
        &inp_batch,
        &pos_batch,
        &attn_mask,
        &lin_mask,
        &prompt_lens,
        Some(&mut cache_batch),
        mlx::StreamOrDevice::default(),
    )
    .expect("batched_prefill");

    // logits_batch shape: [2, 1, vocab]
    let batch_dims = logits_batch.shape();
    let batch_dims = batch_dims.as_slice();
    eprintln!("[p5c] batched logits shape: {:?}", batch_dims);
    assert_eq!(batch_dims.len(), 3, "expected [B, 1, vocab]");
    assert_eq!(batch_dims[0], 2, "B=2");
    assert_eq!(batch_dims[1], 1, "S=1");
    assert_eq!(batch_dims[2] as usize, vocab, "vocab dim match");

    // Slice each row to [vocab] and convert to f32
    let row0 = mlx::ops::indexing::slice(
        &logits_batch,
        &[0_i32, 0_i32, 0_i32][..],
        &[1_i32, 1_i32, vocab as i32][..],
    )
    .expect("slice row 0")
    .reshape(&[vocab as i32][..])
    .expect("reshape row 0");
    let row1 = mlx::ops::indexing::slice(
        &logits_batch,
        &[1_i32, 0_i32, 0_i32][..],
        &[2_i32, 1_i32, vocab as i32][..],
    )
    .expect("slice row 1")
    .reshape(&[vocab as i32][..])
    .expect("reshape row 1");

    let la: Vec<f32> = mlx::ops::cast::astype(&logits_a, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    let lb: Vec<f32> = mlx::ops::cast::astype(&logits_b, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    let lr0: Vec<f32> = mlx::ops::cast::astype(&row0, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    let lr1: Vec<f32> = mlx::ops::cast::astype(&row1, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();

    assert_eq!(la.len(), vocab, "la length");
    assert_eq!(lb.len(), vocab, "lb length");
    assert_eq!(lr0.len(), vocab, "lr0 length");
    assert_eq!(lr1.len(), vocab, "lr1 length");

    let argmax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("non-empty")
    };

    let arg_a = argmax(&la);
    let arg_b = argmax(&lb);
    let arg_r0 = argmax(&lr0);
    let arg_r1 = argmax(&lr1);

    eprintln!(
        "[p5c] B=1 prompt_a argmax: {} (logit={:.4})",
        arg_a, la[arg_a]
    );
    eprintln!(
        "[p5c] B=2 row 0 argmax:    {} (logit={:.4})",
        arg_r0, lr0[arg_r0]
    );
    eprintln!(
        "[p5c] B=1 prompt_b argmax: {} (logit={:.4})",
        arg_b, lb[arg_b]
    );
    eprintln!(
        "[p5c] B=2 row 1 argmax:    {} (logit={:.4})",
        arg_r1, lr1[arg_r1]
    );

    // max_abs_diff per row
    let mut max_diff_a = 0.0_f32;
    let mut max_diff_b = 0.0_f32;
    for i in 0..vocab {
        max_diff_a = max_diff_a.max((la[i] - lr0[i]).abs());
        max_diff_b = max_diff_b.max((lb[i] - lr1[i]).abs());
    }
    eprintln!("[p5c] row 0 max_abs_diff: {}", max_diff_a);
    eprintln!("[p5c] row 1 max_abs_diff: {}", max_diff_b);

    // argmax must be bit-identical
    assert_eq!(
        arg_a, arg_r0,
        "row 0 (prompt_a) argmax mismatch: B=1 says {arg_a}, B=2 batched says {arg_r0}"
    );
    assert_eq!(
        arg_b, arg_r1,
        "row 1 (prompt_b) argmax mismatch: B=1 says {arg_b}, B=2 batched says {arg_r1}"
    );

    // logits max_abs_diff per row within hybrid linear-attention drift envelope.
    // Threshold matches the dense Qwen3.5 batched_prefill test (LOGITS_TOL = 1.0).
    // GatedDeltaNet linear layers accumulate ~0.1-1.0 bf16 drift under batch.
    assert!(
        max_diff_a < LOGITS_TOL,
        "row 0 max_abs_diff {max_diff_a} >= {LOGITS_TOL} — batched path has unexpected divergence"
    );
    assert!(
        max_diff_b < LOGITS_TOL,
        "row 1 max_abs_diff {max_diff_b} >= {LOGITS_TOL} — batched path has unexpected divergence"
    );

    eprintln!(
        "[p5c] PASS — argmax bit-identical, max_abs_diff < {LOGITS_TOL} for both rows \
         (hybrid linear-attention bf16 drift envelope)"
    );
}
