//! Shared parity-test helpers for MiniCPM-V-4.6 integration tests.
//!
//! Used by all `minicpmv46_*_parity`, `minicpmv46_*_match`, and
//! `minicpmv46_*_e2e` integration tests.
//!
//! Each test loads fixtures from its own directory; `load_npy_in` is the
//! general entry-point that accepts the directory explicitly.  Tests that all
//! share the VL fixture dir can use the `FIXTURE_DIR_VL` const directly.

use std::path::PathBuf;

use mlx::{ops, Array, Dtype};

/// Absolute path to the shared VL fixture directory.
/// Used by the four vision / VL parity tests.
pub const FIXTURE_DIR_VL: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/minicpmv46_vl");

// ---------------------------------------------------------------------------
// I/O helper
// ---------------------------------------------------------------------------

/// Load a `.npy` file from `dir/name`.  Panics with a descriptive message if
/// the file is missing (to guide the user to re-run the generator script).
pub fn load_npy_in(dir: &str, name: &str) -> Array {
    let p = format!("{dir}/{name}");
    mlx::io::load_npy(&p)
        .unwrap_or_else(|e| panic!("failed to load {p} — regenerate fixtures: {e}"))
}

// ---------------------------------------------------------------------------
// Env / model-path helper
// ---------------------------------------------------------------------------

/// Read `MINICPMV46_MODEL` and return the checkpoint directory.
///
/// The env var must point to the unpacked `MiniCPM-V-4.6-4bit` snapshot
/// directory.  This is required by every `#[ignore]` parity test that loads
/// real model weights.
pub fn checkpoint_dir() -> PathBuf {
    let env = std::env::var("MINICPMV46_MODEL").expect(
        "MINICPMV46_MODEL env var must point to the MiniCPM-V-4.6-4bit snapshot dir (#[ignore] test)",
    );
    PathBuf::from(env)
}

// ---------------------------------------------------------------------------
// Array conversion
// ---------------------------------------------------------------------------

/// Cast `a` to `f32` and flatten into a `Vec<f32>`.
pub fn to_f32_vec(a: &Array) -> Vec<f32> {
    ops::cast::astype(a, Dtype::Float32)
        .expect("astype f32")
        .to_vec()
        .expect("to_vec")
}

// ---------------------------------------------------------------------------
// Numeric comparison helpers
// ---------------------------------------------------------------------------

/// Greedy argmax over a 1-D logits `Array` (cast to fp32 first).
pub fn greedy_argmax(arr: &Array) -> usize {
    let v = to_f32_vec(arr);
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

/// Top-k token ids by logit value (descending).
pub fn top_k(arr: &Array, k: usize) -> Vec<usize> {
    let v = to_f32_vec(arr);
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

/// Worst-element absolute deviation between two arrays (cast to f32).
pub fn max_abs_diff(a: &Array, b: &Array) -> f32 {
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

/// Absolute logit difference at a specific token index.
pub fn diff_at(a: &Array, b: &Array, i: usize) -> f32 {
    let av = to_f32_vec(a);
    let bv = to_f32_vec(b);
    (av[i] - bv[i]).abs()
}
