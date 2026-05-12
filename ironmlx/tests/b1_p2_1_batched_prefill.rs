//! B1-p2.1 static batched prefill — 4-point numerical equivalence test.
//!
//! For each (B, prompt_lens) configuration:
//!   1. Per-stream reference: for each prompt i, run Qwen35Model::forward_on
//!      with a fresh batch=1 cache; record last-position logits.
//!   2. Batched call: build left-padded input_ids[B, S_max], position_ids[3,B,S_max],
//!      attention_mask[B,1,S_max,S_max], cache(batch=B); call batched_prefill.
//!   3. Verify per batch row i: max_abs(batched[i, :] - per_stream[i].last_logits) < 1e-3
//!      AND argmax(batched[i, :]) == argmax(per_stream[i].last_logits)
//!
//! Run with:
//!   QWEN35_MODEL=/path/to/model \
//!   MLX_DIR=$HOME/.local/mlx \
//!   cargo test -p ironmlx --release --test b1_p2_1_batched_prefill -- --ignored --nocapture

use std::path::Path;

use mlx::Array;
use mlx::Dtype;

use ironmlx::core::generate::{
    build_batch_attention_mask, build_position_ids, build_position_ids_batched,
};
use ironmlx::core::Loader;
use ironmlx::models::qwen3_5::Qwen35Model;

const LOGITS_TOL: f32 = 1e-3;

/// Pad-token id used to fill the left side of each batch row.
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
        .forward_on(&input_ids, &pos_ids, Some(&mut cache), ())
        .expect("forward_on");
    // forward_on returns [B, 1, vocab]; reshape to [vocab].
    let vocab = logits.shape().as_slice()[2];
    logits.reshape(&[vocab][..]).expect("reshape")
}

/// Run one (B, prompt_lens, seed_base) point and assert all checks.
fn run_point(model: &Qwen35Model, prompt_lens: &[i32], seed_base: u64) {
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

    // Build batched inputs (left-padded).
    let mut packed: Vec<u32> = Vec::with_capacity(b * max_len);
    for p in &prompts {
        let pad_n = max_len - p.len();
        for _ in 0..pad_n {
            packed.push(PAD_TOKEN_ID);
        }
        packed.extend_from_slice(p);
    }
    let input_ids: Array = (&packed[..], &[b as i32, max_len as i32][..])
        .try_into()
        .expect("packed input_ids");

    let pos_ids = build_position_ids_batched(prompt_lens, max_len as i32)
        .expect("build_position_ids_batched");
    let attn_mask = build_batch_attention_mask(prompt_lens, max_len as i32, Dtype::Bfloat16)
        .expect("build_batch_attention_mask");

    let mut cache = model
        .make_cache(b as i32, max_len as i32 + 1, Dtype::Bfloat16)
        .expect("make_cache batch=B");

    let batched_logits = model
        .batched_prefill(&input_ids, &pos_ids, &attn_mask, Some(&mut cache), ())
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
        eprintln!(
            "[b1_p2_1] row {i}: max_abs_diff={:.6}, argmax_batched={}, argmax_ref={}",
            d, our_arg, ref_arg
        );
        assert!(d < LOGITS_TOL, "row {i}: max_abs_diff={d} >= {LOGITS_TOL}");
        assert_eq!(
            our_arg, ref_arg,
            "row {i}: argmax mismatch (batched={our_arg}, ref={ref_arg})"
        );
    }

    eprintln!("[b1_p2_1] point B={} lens={:?} PASS", b, prompt_lens);
}

#[test]
#[ignore = "requires QWEN35_MODEL env"]
fn b1_p2_1_batched_prefill_matrix() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL");
    let loader = Loader::open_multimodal(Path::new(&model_dir)).expect("loader");
    let model = Qwen35Model::from_loader(&loader).expect("model");

    // Point 1: B=2 same length.
    run_point(&model, &[128, 128], 0x1111);
    // Point 2: B=2 mixed length (left-padded).
    run_point(&model, &[128, 96], 0x2222);
    // Point 3: B=4 same length.
    run_point(&model, &[128, 128, 128, 128], 0x3333);
    // Point 4: B=4 mixed length.
    run_point(&model, &[128, 96, 64, 128], 0x4444);

    eprintln!("[b1_p2_1] PASS — all 4 points");
}
