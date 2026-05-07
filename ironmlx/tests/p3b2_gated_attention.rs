//! P3b2 GatedAttention numerical-correctness integration test.
//!
//! Loads .npy fixtures from `tests/fixtures/p3b2_gated_attention/` (generated
//! by `gen_fixture.py` against an independent Python reference of Qwen3-Next
//! gated full attention) and verifies that `nn::GatedAttention::forward`
//! produces numerically equivalent output.
//!
//! Tolerance: bf16 atol = 1e-3 (limited by bf16 rounding).
//!
//! Regenerate fixtures via:
//!
//! ```text
//! cd ironmlx/tests/fixtures/p3b2_gated_attention && python gen_fixture.py
//! ```

use mlx::{Array, Dtype};

use ironmlx::nn::{GatedAttention, GatedAttentionConfig, Linear, Mrope, RmsNorm};

const FIXTURE_DIR: &str = "tests/fixtures/p3b2_gated_attention";

/// Pinned by the fixture's small-scale config.
const HEAD_DIM: i32 = 8;
const NUM_HEADS: i32 = 4;
const NUM_KV_HEADS: i32 = 2;

fn load(name: &str) -> Array {
    let path = format!("{FIXTURE_DIR}/{name}.npy");
    mlx::io::load_npy(&path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"))
}

/// max(|a - b|) for arrays cast to fp32.
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
fn gated_attention_matches_python_fixture() {
    let q_w = load("q_proj_weight");
    let k_w = load("k_proj_weight");
    let v_w = load("v_proj_weight");
    let o_w = load("o_proj_weight");
    let q_norm_w = load("q_norm_weight");
    let k_norm_w = load("k_norm_weight");

    let cfg = GatedAttentionConfig {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        rms_norm_eps: 1e-6,
        attention_bias: false,
    };

    let attn = GatedAttention::from_components(
        Linear::new_fp(q_w, None),
        Linear::new_fp(k_w, None),
        Linear::new_fp(v_w, None),
        Linear::new_fp(o_w, None),
        RmsNorm::new(q_norm_w, cfg.rms_norm_eps),
        RmsNorm::new(k_norm_w, cfg.rms_norm_eps),
        cfg,
    );

    let mrope = Mrope::new(HEAD_DIM, 1e7, 1.0, &[2, 1, 1], true).unwrap();

    let x = load("input_x");
    let cos = load("expected_cos");
    let sin = load("expected_sin");
    let expected = load("expected_gated_attn_out");

    let out = attn
        .forward(&x, &mrope, &cos, &sin, None, None)
        .expect("forward");

    assert_eq!(out.shape().as_slice(), expected.shape().as_slice());

    let err = max_abs_diff(&out, &expected);
    assert!(
        err < 1e-3,
        "GatedAttention output max abs diff = {err} > 1e-3"
    );
}
