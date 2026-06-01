//! MiniCPM-V-4.6 text-only logits-alignment integration test.
//!
//! MiniCPM-V-4.6's language backbone is Qwen3.5-text verbatim. This loads
//! `mlx-community/MiniCPM-V-4.6-4bit` from `$MINICPMV46_MODEL` via the
//! `minicpmv4_6` text-only facade, feeds the exact token ids captured from
//! mlx-vlm, runs `MiniCpmV46Model::forward_on`, and compares the last-position
//! logits to the mlx-vlm reference. Feeding the saved ids (rather than
//! re-tokenizing) isolates LM-forward correctness from tokenizer parity.
//!
//! Regenerate the fixtures with `tests/fixtures/minicpmv46/gen_logits.py`.
//!
//! Run with:
//! ```text
//! MLX_DIR=$HOME/.local/mlx \
//!   MINICPMV46_MODEL=/path/to/MiniCPM-V-4.6-4bit/snapshots/<sha> \
//!   cargo test --release -p ironmlx --test minicpmv46_text_logits_match -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx::{Array, Dtype};

use ironmlx::core::{generate::build_position_ids, Loader, Model};
use ironmlx::models::minicpmv4_6;

const FIXTURE_DIR: &str = "tests/fixtures/minicpmv46";

fn load_npy(name: &str) -> Array {
    let p = format!("{FIXTURE_DIR}/{name}");
    mlx::io::load_npy(&p)
        .unwrap_or_else(|e| panic!("failed to load {p} — run gen_logits.py first: {e}"))
}

fn checkpoint_dir() -> PathBuf {
    let env = std::env::var("MINICPMV46_MODEL").expect(
        "MINICPMV46_MODEL env var must point to the MiniCPM-V-4.6-4bit snapshot dir (#[ignore] test)",
    );
    PathBuf::from(env)
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let av: Vec<f32> = mlx::ops::cast::astype(a, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    let bv: Vec<f32> = mlx::ops::cast::astype(b, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Greedy argmax over a 1-D logits Array (cast to fp32 first).
fn greedy_argmax(arr: &Array) -> usize {
    let v: Vec<f32> = mlx::ops::cast::astype(arr, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

/// Top-k token ids by logit value (descending), for distribution-shape comparison.
fn top_k(arr: &Array, k: usize) -> Vec<usize> {
    let v: Vec<f32> = mlx::ops::cast::astype(arr, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

/// Absolute logit difference at a specific token index.
fn diff_at(a: &Array, b: &Array, i: usize) -> f32 {
    let av: Vec<f32> = mlx::ops::cast::astype(a, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    let bv: Vec<f32> = mlx::ops::cast::astype(b, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    (av[i] - bv[i]).abs()
}

/// Number of prompt fixtures emitted by `gen_logits.py` (`PROMPTS` length).
const NUM_PROMPTS: usize = 4;

#[test]
#[ignore = "requires MINICPMV46_MODEL env var pointing to a real 4-bit checkpoint"]
fn minicpmv46_text_logits_match() {
    let model_dir = checkpoint_dir();
    // Loader::open drops vision_tower.* keys — text-only path.
    let loader = Loader::open(&model_dir).expect("Loader::open");
    let model = minicpmv4_6::model_from_loader(&loader).expect("minicpmv4_6::model_from_loader");

    for p in 0..NUM_PROMPTS {
        // Use the exact token ids captured from mlx-vlm (isolates LM forward
        // from any tokenizer-encoding discrepancy).
        let ids: Vec<i32> = load_npy(&format!("expected_input_ids_p{p}.npy"))
            .to_vec()
            .expect("input ids to_vec");
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
                mlx::StreamOrDevice::default(),
            )
            .expect("forward_on");
        // logits: [1, 1, vocab] — last position only.
        let vocab = logits.shape().as_slice()[2];
        let last_flat = logits.reshape((vocab,)).expect("reshape");

        let expected = load_npy(&format!("expected_last_logits_p{p}.npy"));
        assert_eq!(
            last_flat.shape().as_slice().last().copied(),
            expected.shape().as_slice().last().copied(),
            "prompt {p}: vocab dim must match"
        );

        let argmax_rust = greedy_argmax(&last_flat);
        let argmax_ref = greedy_argmax(&expected);
        let top5_rust = top_k(&last_flat, 5);
        let top5_ref = top_k(&expected, 5);
        let err = max_abs_diff(&last_flat, &expected);
        let diff_at_argmax = diff_at(&last_flat, &expected, argmax_ref);
        println!(
            "prompt {p}: tokens={s} argmax rust={argmax_rust} ref={argmax_ref} \
             max_abs={err:.4} diff@argmax={diff_at_argmax:.4} \
             top5_rust={top5_rust:?} top5_ref={top5_ref:?}"
        );

        // Primary correctness gate — what actually determines generated output:
        //   1. greedy argmax token matches exactly, and
        //   2. the whole top-5 head of the distribution matches as a set.
        // Together these are strictly stronger than a single-argmax check; a
        // structural bug (wrong norm offset / rope / gate / layer count) would
        // corrupt the head and flip these.
        assert_eq!(
            argmax_rust, argmax_ref,
            "prompt {p}: greedy argmax mismatch — ironmlx={argmax_rust}, mlx-vlm={argmax_ref}",
        );
        let set_rust: std::collections::BTreeSet<usize> = top5_rust.iter().copied().collect();
        let set_ref: std::collections::BTreeSet<usize> = top5_ref.iter().copied().collect();
        assert_eq!(
            set_rust, set_ref,
            "prompt {p}: top-5 token set mismatch — ironmlx={top5_rust:?}, mlx-vlm={top5_ref:?}",
        );

        // Structural-sanity guard on the full-vocab worst-element deviation.
        // 4-bit BF16 across 24 hybrid layers + an independent quantized-matmul
        // accumulation order (ironmlx self_qmm/gather vs mlx quantized_matmul)
        // produces a far-tail noise floor of ~0.53 here (observed across these
        // 4 prompts; the winning-token logit itself stays within ~0.125, and is
        // bit-identical on prompt 2). 1.0 is ~1.9x that floor — the same
        // headroom ratio the Qwen3.5-4B p4 test uses (0.5 over a ~0.27 floor) —
        // and still far below the multi-unit deviation a real structural bug
        // would yield.
        assert!(
            err < 1.0,
            "prompt {p}: max abs logits diff = {err} > 1.0 (structural bug suspected)",
        );
    }
}
