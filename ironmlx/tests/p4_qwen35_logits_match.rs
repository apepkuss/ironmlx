//! P4 Qwen3.5 logits-alignment integration test.
//!
//! Loads `mlx-community/Qwen3.5-4B-MLX-4bit` from `$QWEN35_MODEL`,
//! tokenizes a fixed prompt, runs `Qwen35Model::forward_on`, and compares
//! the last-position logits to an mlx-lm reference saved as `.npy`.
//!
//! Run with:
//! ```text
//! MLX_DIR=$HOME/.local/mlx \
//!   QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \
//!   cargo test --release --ignored -p ironmlx -- p4_qwen35_logits_match -- --test-threads=1
//! ```

use std::path::PathBuf;

use mlx::{Array, Dtype};

use ironmlx::core::{generate::build_position_ids, Loader, Tokenizer};
use ironmlx::models::Qwen35Model;

const FIXTURE_DIR: &str = "tests/fixtures/p4_qwen35";

fn load_expected_logits() -> Array {
    let p = format!("{FIXTURE_DIR}/expected_last_logits.npy");
    mlx::io::load_npy(&p).unwrap_or_else(|e| {
        panic!("failed to load {p} — run gen_logits.py first (see README): {e}")
    })
}

fn checkpoint_dir() -> PathBuf {
    let env = std::env::var("QWEN35_MODEL")
        .expect("QWEN35_MODEL env var must be set to the Qwen3.5-4B-MLX-4bit dir (#[ignore] test)");
    PathBuf::from(env)
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let a32 = mlx::ops::cast::astype(a, Dtype::Float32).unwrap();
    let b32 = mlx::ops::cast::astype(b, Dtype::Float32).unwrap();
    let av: Vec<f32> = a32.to_vec().unwrap();
    let bv: Vec<f32> = b32.to_vec().unwrap();
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
#[ignore = "requires QWEN35_MODEL env var pointing to a real 4-bit checkpoint"]
fn p4_qwen35_logits_match() {
    let model_dir = checkpoint_dir();
    let loader = Loader::open(&model_dir).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");

    // Match the Python fixture exactly: same prompt, no chat template, no special tokens.
    let prompt = "What is 2+2?";
    let ids = tokenizer
        .encode(prompt, /* add_special_tokens = */ false)
        .expect("tokenizer.encode");

    let s = ids.len() as i32;
    let input_ids: Array = (ids.as_slice(), &[1_i32, s][..])
        .try_into()
        .expect("input_ids");
    let position_ids = build_position_ids(0, s).expect("position_ids");

    let mut cache = model
        .make_cache(/* batch */ 1, s + 1, Dtype::Bfloat16)
        .expect("make_cache");
    let logits = model
        .forward_on(
            &input_ids,
            &position_ids,
            Some(&[s]),
            None,
            Some(&mut cache),
            (),
        )
        .expect("forward_on");
    // logits: [1, 1, vocab] — last-position only (Qwen35Model::forward_on slices
    // the last hidden state before the lm_head projection).
    let vocab = logits.shape().as_slice()[2];
    let last_flat = logits.reshape((vocab,)).expect("reshape");

    let expected = load_expected_logits();
    assert_eq!(
        last_flat.shape().as_slice().last().copied(),
        expected.shape().as_slice().last().copied(),
        "vocab dim must match"
    );

    // Strong correctness: greedy argmax token MUST match exactly. This is what
    // determines actual inference output, regardless of per-element noise.
    let argmax_rust = greedy_argmax(&last_flat);
    let argmax_py = greedy_argmax(&expected);
    assert_eq!(
        argmax_rust, argmax_py,
        "greedy argmax token mismatch — Rust picked {argmax_rust}, mlx-lm picked {argmax_py}",
    );

    // Loose structural sanity: max-abs-diff < 0.5. Tighter than this is impossible
    // on 4-bit BF16 over 32 layers — physical accumulation noise is ~0.2-0.3 per
    // non-top logit (BF16 ULP at logit magnitude ~2 is 0.015625; 32 layers × 4-bit
    // quant across an attention chain stacks to ~17 ULPs ≈ 0.27). 0.5 catches
    // structural bugs (wrong layer count, missing residual, wrong norm) without
    // false-positiving on quantization noise.
    let err = max_abs_diff(&last_flat, &expected);
    assert!(
        err < 0.5,
        "Qwen35 last-position logits max abs diff = {err} > 0.5 (structural bug suspected)",
    );
}

/// Greedy argmax over a 1-D logits Array. Casts to fp32 first so bf16 input is supported.
fn greedy_argmax(arr: &Array) -> usize {
    let f = mlx::ops::cast::astype(arr, Dtype::Float32).unwrap();
    let v: Vec<f32> = f.to_vec().unwrap();
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

/// Verify that `Qwen35Model::forward_on` works correctly when called from a
/// tokio blocking-pool thread (i.e. the thread that handles HTTP requests).
///
/// The bug this guards against: lazy arrays produced by `concatenate` during
/// `GatedDeltaNet::from_loader` carry the main thread's MLX stream
/// (Stream(gpu, 0)).  When a tokio `spawn_blocking` thread initialises its own
/// stream (Stream(gpu, 1)) and then evaluates those arrays, `gpu::eval` looks
/// up Stream(gpu, 0) in the current thread's thread_local encoder map and
/// throws "There is no Stream(gpu, 0) in current thread."
///
/// After the fix (eager eval in `GatedDeltaNet::from_loader`) the concatenated
/// weight tensors are plain data buffers before they enter any `Linear` field,
/// so no stream mismatch can occur.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires QWEN35_MODEL env var pointing to a real 4-bit checkpoint"]
async fn p4_model_forward_from_blocking_thread() {
    use ironmlx::core::generate::build_position_ids;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let model_dir = checkpoint_dir();
    // Load model on the main thread (simulates `ironmlx serve` startup).
    let loader = Loader::open(&model_dir).expect("Loader::open");
    let model = Arc::new(Mutex::new(
        Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader"),
    ));

    let model_for_task = model.clone();
    // Spawn the forward pass on a blocking thread — this is exactly what the
    // HTTP server does inside `tokio::task::spawn_blocking`.
    let result = tokio::task::spawn_blocking(move || {
        let model_guard = model_for_task.blocking_lock();
        let prompt_ids = [9454u32, 374, 220, 17, 10, 17, 30]; // "What is 2+2?"
        let s = prompt_ids.len() as i32;
        let input_ids: Array = (prompt_ids.as_slice(), &[1_i32, s][..])
            .try_into()
            .expect("input_ids");
        let position_ids = build_position_ids(0, s).expect("position_ids");
        let mut cache = model_guard
            .make_cache(1, s + 1, Dtype::Bfloat16)
            .expect("make_cache");
        model_guard
            .forward_on(
                &input_ids,
                &position_ids,
                Some(&[s]),
                None,
                Some(&mut cache),
                (),
            )
            .expect("forward_on from blocking thread")
    })
    .await
    .expect("spawn_blocking join");

    // Verify the result has the expected shape: [1, S, vocab_size].
    let shape = result.shape();
    assert_eq!(shape.as_slice().len(), 3, "logits must be rank 3");
    assert_eq!(shape.as_slice()[0], 1, "batch must be 1");
    assert!(shape.as_slice()[2] > 0, "vocab must be positive");
}
