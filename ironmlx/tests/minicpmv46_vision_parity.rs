//! MiniCPM-V-4.6 P1 vision-embeds parity test vs mlx-vlm.
//!
//! This is the P1 ACCEPTANCE GATE for the vision stack. It isolates the VISION
//! STACK (SigLIP embeddings → 27 encoder layers with VitMerger after layer 6 →
//! post_layernorm → Merger) from the LLaVA-UHD preprocessing/slicing pipeline
//! (P2/P3). The fixture captures the packed `pixel_values` EXACTLY as mlx-vlm's
//! `Model.get_vision_embedding` vision tower consumes them (post CHW→HWC
//! transpose + `expand_dims(0)` → `[1, 14, n*14, 3]`), feeds those identical
//! bytes to the Rust `MiniCpmV46Vision::compute_vision_embeds`, and compares the
//! final merged embeds `[N, 1024]`.
//!
//! Regenerate the fixtures with
//! `tests/fixtures/minicpmv46_vl/gen_vision_embeds.py`.
//!
//! Run with:
//! ```text
//! MLX_DIR=$HOME/.local/mlx \
//!   MINICPMV46_MODEL=/path/to/MiniCPM-V-4.6-4bit/snapshots/<sha> \
//!   cargo test --release -p ironmlx --test minicpmv46_vision_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx::{ops, Array, Dtype};

use ironmlx::core::Loader;
use ironmlx::models::minicpmv4_6::config::MiniCpmV46VisionConfig;
use ironmlx::models::minicpmv4_6::vision::MiniCpmV46Vision;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/minicpmv46_vl");

fn load_npy(name: &str) -> Array {
    let p = format!("{FIXTURE_DIR}/{name}");
    mlx::io::load_npy(&p)
        .unwrap_or_else(|e| panic!("failed to load {p} — run gen_vision_embeds.py first: {e}"))
}

fn checkpoint_dir() -> PathBuf {
    let env = std::env::var("MINICPMV46_MODEL").expect(
        "MINICPMV46_MODEL env var must point to the MiniCPM-V-4.6-4bit snapshot dir (#[ignore] test)",
    );
    PathBuf::from(env)
}

fn to_f32_vec(a: &Array) -> Vec<f32> {
    ops::cast::astype(a, Dtype::Float32)
        .expect("astype f32")
        .to_vec()
        .expect("to_vec")
}

/// Worst-element absolute deviation between two arrays (cast to f32).
fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let av = to_f32_vec(a);
    let bv = to_f32_vec(b);
    assert_eq!(
        av.len(),
        bv.len(),
        "size mismatch: {} vs {}",
        av.len(),
        bv.len()
    );
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Flattened cosine similarity between two arrays (cast to f32).
fn cosine_sim(a: &Array, b: &Array) -> f32 {
    let av = to_f32_vec(a);
    let bv = to_f32_vec(b);
    assert_eq!(
        av.len(),
        bv.len(),
        "size mismatch: {} vs {}",
        av.len(),
        bv.len()
    );
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

#[test]
#[ignore = "requires MINICPMV46_MODEL env var pointing to a real 4-bit checkpoint"]
fn minicpmv46_vision_parity() {
    let model_dir = checkpoint_dir();
    // open_multimodal retains vision_tower.* / vit_merger.* / merger.* keys.
    let loader = Loader::open_multimodal(&model_dir).expect("Loader::open_multimodal");
    let cfg = MiniCpmV46VisionConfig::from_loader(&loader).expect("MiniCpmV46VisionConfig");
    let vision = MiniCpmV46Vision::from_loader(&loader, &cfg).expect("MiniCpmV46Vision");

    // input_grid.npy = [grid_h, grid_w] int32.
    let grid: Vec<i32> = load_npy("input_grid.npy").to_vec().expect("grid to_vec");
    assert_eq!(grid.len(), 2, "input_grid.npy must be [grid_h, grid_w]");
    let (gh, gw) = (grid[0], grid[1]);

    // input_pixel_values.npy = [1, 14, n*14, 3] f32 (post CHW→HWC + expand_dims).
    // compute_vision_embeds expects exactly this layout; cast to bf16 to match
    // the checkpoint's vision-tower precision.
    let pix_f32 = load_npy("input_pixel_values.npy");
    let dims = pix_f32.shape();
    let d = dims.as_slice();
    assert_eq!(
        d.len(),
        4,
        "input_pixel_values must be 4-D [1, 14, n*14, 3], got {d:?}"
    );
    assert_eq!(d[0], 1, "batch dim must be 1, got {d:?}");
    assert_eq!(d[1], 14, "packed height must be patch=14, got {d:?}");
    assert_eq!(d[3], 3, "channel dim must be 3, got {d:?}");
    let n = d[2] / 14;
    assert_eq!(
        n,
        gh * gw,
        "packed patch count n={n} must equal grid_h*grid_w={}",
        gh * gw
    );
    let pix = ops::cast::astype(&pix_f32, Dtype::Bfloat16).expect("cast pixels to bf16");

    let got = vision
        .compute_vision_embeds(&pix, gh, gw, ())
        .expect("compute_vision_embeds");

    let expected = load_npy("expected_vision_embeds.npy");
    assert_eq!(
        got.shape().as_slice(),
        expected.shape().as_slice(),
        "merged embeds shape mismatch: ironmlx={:?} mlx-vlm={:?}",
        got.shape().as_slice(),
        expected.shape().as_slice()
    );

    let cos = cosine_sim(&got, &expected);
    let max_abs = max_abs_diff(&got, &expected);
    println!(
        "minicpmv46_vision_parity: grid=({gh},{gw}) n={n} shape={:?} cos={cos:.6} max_abs={max_abs:.4}",
        got.shape().as_slice()
    );

    // Primary correctness gate: cosine similarity over the full merged-embeds
    // tensor. A structural divergence anywhere in the stack (position buckets,
    // SDPA scale, CHW/HWC layout, gelu variant, VitMerger/Merger reshape) would
    // tank the cosine far below this floor.
    assert!(
        cos > 0.999,
        "vision-embeds cosine similarity = {cos} <= 0.999 (structural divergence suspected)",
    );

    // Structural-sanity guard on the worst-element deviation. This checkpoint's
    // SigLIP vision tower is NOT quantized (only the LM is 4-bit), so it runs as
    // dense bf16 — and ironmlx executes the identical op sequence as mlx-vlm
    // (same packed-patch matmul, same SDPA, same gelu_tanh, same LayerNorm).
    // The observed deviation is therefore exactly 0.0 (bit-identical at every
    // stage, confirmed by the stage-by-stage localizer during P1 debugging).
    // We keep a small non-zero structural bound (0.05) rather than asserting
    // literal zero: it absorbs any incidental future bf16 op-reordering jitter
    // while still tripping hard on a real structural bug — the position-bucket
    // tie-rounding bug that was fixed here produced max_abs ≈ 0.39, ~8× this
    // bound. Provisional upper guard well under 2.0.
    assert!(
        max_abs < 0.05,
        "vision-embeds max abs diff = {max_abs} >= 0.05 (structural bug suspected)",
    );
}
