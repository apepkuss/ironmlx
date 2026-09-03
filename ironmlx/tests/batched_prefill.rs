//! B1-p2.1 static batched prefill — 4-point numerical equivalence test.
//!
//! For each (B, prompt_lens) configuration:
//!   1. Per-stream reference: for each prompt i, run Qwen35Model::forward_on
//!      with a fresh batch=1 cache; record last-position logits.
//!   2. Batched call: build right-padded input_ids[B, S_max], position_ids[3,B,S_max],
//!      attention_mask[B,1,S_max,S_max], linear_attention_mask[B,S_max],
//!      cache(batch=B); call batched_prefill.
//!   3. Per batch row i, assert max_abs_diff < `LOGITS_TOL` (1.0). Argmax
//!      bit-identical is tracked as a statistic but NOT a hard assertion —
//!      Qwen3.5's hybrid linear-attention path has small (~0.1-0.6) bf16
//!      numerical drift between B>1 and B=1 due to GPU kernel reduction
//!      scheduling, and near-tied logits can flip argmax. The test still
//!      requires the majority of rows to be argmax bit-identical (≥ 75%).
//!
//! Run with:
//!   QWEN35_MODEL=/path/to/model \
//!   MLX_DIR=$HOME/.local/mlx \
//!   cargo test -p ironmlx --release --test batched_prefill -- --ignored --nocapture

use std::path::Path;

use mlx::Array;
use mlx::Dtype;

use ironmlx::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_position_ids,
    build_position_ids_batched,
};
use ironmlx::core::Loader;
use ironmlx::models::qwen3_5::Qwen35Model;

const LOGITS_TOL: f32 = 1.0;
const ARGMAX_BIT_ID_FLOOR: f32 = 0.75; // ≥ 75% of rows must be argmax bit-identical

/// Pad-token id used to fill the trailing slots of each batch row.
/// Any in-vocab id works; the attention mask discards these positions.
const PAD_TOKEN_ID: u32 = 0;

/// Pick a deterministic synthetic prompt of length `n` using a u64 seed.
/// Returns u32 token ids within [1, max_vocab_id - 1] (avoids 0 since
/// we reserve 0 as pad).
fn synth_prompt(seed: u64, n: usize, max_vocab_id: u32) -> Vec<u32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let token = 1 + ((s >> 33) as u32 % (max_vocab_id - 2));
        out.push(token);
    }
    out
}

fn max_abs_diff_f32(a: &Array, b: &Array) -> f32 {
    let a32 = mlx::ops::cast::astype(a, Dtype::Float32).expect("af32");
    let b32 = mlx::ops::cast::astype(b, Dtype::Float32).expect("bf32");
    let av: Vec<f32> = a32.to_vec::<f32>().expect("av");
    let bv: Vec<f32> = b32.to_vec::<f32>().expect("bv");
    av.iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn argmax(arr: &Array) -> i32 {
    let f32_arr = mlx::ops::cast::astype(arr, Dtype::Float32).expect("astype f32");
    let v: Vec<f32> = f32_arr.to_vec::<f32>().expect("to_vec");
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .expect("non-empty")
}

/// Per-stream reference: run forward_on for one prompt with a fresh batch=1
/// cache. Returns last_logits [vocab].
fn per_stream_reference(model: &Qwen35Model, prompt: &[u32]) -> Array {
    let s = prompt.len() as i32;
    let input_ids: Array = (&prompt[..], &[1_i32, s][..])
        .try_into()
        .expect("input_ids");
    let pos_ids = build_position_ids(0, s).expect("build_position_ids");
    let mut cache = model
        .make_cache(/* batch */ 1, s + 1, Dtype::Bfloat16)
        .expect("make_cache");
    let logits = model
        .forward_on(&input_ids, &pos_ids, Some(&[s]), None, Some(&mut cache), ())
        .expect("forward_on");
    // forward_on returns [B, 1, vocab]; reshape to [vocab].
    let vocab = logits.shape().as_slice()[2];
    logits.reshape(&[vocab][..]).expect("reshape")
}

/// Statistics aggregated across all rows of all points.
struct MatrixStats {
    total_rows: usize,
    argmax_bit_id_rows: usize,
}

impl MatrixStats {
    fn new() -> Self {
        Self {
            total_rows: 0,
            argmax_bit_id_rows: 0,
        }
    }
}

/// Run one (B, prompt_lens, seed_base) point and assert all checks.
fn run_point(model: &Qwen35Model, prompt_lens: &[i32], seed_base: u64, stats: &mut MatrixStats) {
    let b = prompt_lens.len();
    let max_len = *prompt_lens.iter().max().expect("at least one") as usize;
    let max_vocab_id: u32 = 32_000;

    let prompts: Vec<Vec<u32>> = (0..b)
        .map(|i| synth_prompt(seed_base + i as u64, prompt_lens[i] as usize, max_vocab_id))
        .collect();

    eprintln!(
        "[b1_p2_1] point B={}, lens={:?}, max_len={}",
        b, prompt_lens, max_len
    );

    // Per-stream references.
    let refs: Vec<Array> = prompts
        .iter()
        .map(|p| per_stream_reference(model, p))
        .collect();

    // Build batched inputs (right-padded).
    let mut packed: Vec<u32> = Vec::with_capacity(b * max_len);
    for p in &prompts {
        packed.extend_from_slice(p);
        let pad_n = max_len - p.len();
        for _ in 0..pad_n {
            packed.push(PAD_TOKEN_ID);
        }
    }
    let input_ids: Array = (&packed[..], &[b as i32, max_len as i32][..])
        .try_into()
        .expect("packed input_ids");

    let pos_ids = build_position_ids_batched(prompt_lens, max_len as i32)
        .expect("build_position_ids_batched");
    let attn_mask = build_batch_attention_mask(prompt_lens, max_len as i32, Dtype::Bfloat16)
        .expect("build_batch_attention_mask");
    let linear_mask =
        build_batch_linear_mask(prompt_lens, max_len as i32).expect("build_batch_linear_mask");

    let mut cache = model
        .make_cache(b as i32, max_len as i32 + 1, Dtype::Bfloat16)
        .expect("make_cache batch=B");

    let batched_logits = model
        .batched_prefill(
            &input_ids,
            &pos_ids,
            &attn_mask,
            &linear_mask,
            prompt_lens,
            Some(&mut cache),
            (),
        )
        .expect("batched_prefill");
    eprintln!(
        "[b1_p2_1] batched logits shape: {:?}",
        batched_logits.shape().as_slice()
    );

    // batched_prefill returns [B, 1, vocab]; per row slice + compare.
    let dims = batched_logits.shape();
    let batched_dims = dims.as_slice();
    assert_eq!(batched_dims.len(), 3, "expected [B, 1, vocab]");
    let vocab = batched_dims[2];

    for i in 0..b {
        let row = mlx::ops::indexing::slice(
            &batched_logits,
            &[i as i32, 0_i32, 0_i32][..],
            &[i as i32 + 1, 1_i32, vocab][..],
        )
        .expect("slice row");
        let row_flat = row.reshape(&[vocab][..]).expect("reshape row");

        let ref_logits = &refs[i];
        let d = max_abs_diff_f32(&row_flat, ref_logits);
        let our_arg = argmax(&row_flat);
        let ref_arg = argmax(ref_logits);
        let argmax_match = our_arg == ref_arg;
        eprintln!(
            "[b1_p2_1] row {i}: max_abs_diff={:.6}, argmax_batched={}, argmax_ref={} ({})",
            d,
            our_arg,
            ref_arg,
            if argmax_match { "bit-id" } else { "FLIP" }
        );
        assert!(d < LOGITS_TOL, "row {i}: max_abs_diff={d} >= {LOGITS_TOL}");
        stats.total_rows += 1;
        if argmax_match {
            stats.argmax_bit_id_rows += 1;
        }
    }

    eprintln!(
        "[b1_p2_1] point B={} lens={:?} PASS (max_abs_diff gate)",
        b, prompt_lens
    );
}

#[test]
#[ignore = "requires QWEN35_MODEL env"]
fn b1_p2_1_batched_prefill_matrix() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL");
    let loader = Loader::open_multimodal(Path::new(&model_dir)).expect("loader");
    let model = Qwen35Model::from_loader(&loader).expect("model");

    let mut stats = MatrixStats::new();

    // Point 1: B=2 same length.
    run_point(&model, &[128, 128], 0x1111, &mut stats);
    // Point 2: B=2 mixed length (right-padded).
    run_point(&model, &[128, 96], 0x2222, &mut stats);
    // Point 3: B=4 same length.
    run_point(&model, &[128, 128, 128, 128], 0x3333, &mut stats);
    // Point 4: B=4 mixed length.
    run_point(&model, &[128, 96, 64, 128], 0x4444, &mut stats);

    let bit_id_frac = stats.argmax_bit_id_rows as f32 / stats.total_rows as f32;
    eprintln!(
        "[b1_p2_1] argmax bit-id summary: {}/{} rows ({:.1}%)",
        stats.argmax_bit_id_rows,
        stats.total_rows,
        bit_id_frac * 100.0
    );
    assert!(
        bit_id_frac >= ARGMAX_BIT_ID_FLOOR,
        "argmax bit-id rate {bit_id_frac:.2} below floor {ARGMAX_BIT_ID_FLOOR:.2} — \
         hybrid linear-attention path may have regressed beyond expected bf16 drift"
    );
    eprintln!(
        "[b1_p2_1] PASS — all 4 points (max_abs_diff < {LOGITS_TOL}, argmax bit-id ≥ {:.0}%)",
        ARGMAX_BIT_ID_FLOOR * 100.0
    );
}
