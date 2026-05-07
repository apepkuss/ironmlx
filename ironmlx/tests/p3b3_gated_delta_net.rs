//! P3b3 GatedDeltaNet numerical-correctness integration test.
//!
//! Loads .npy fixtures from `tests/fixtures/p3b3_gated_delta_net/` (generated
//! by `gen_fixture.py` against an independent Python ops-based reference,
//! NOT mlx-lm's Metal kernel — avoids circular validation).
//!
//! Tolerance: atol = 1e-3 (bf16/fp32 mixed; rms_norm with fp32 weight promotes
//! the chain to fp32 internally, but the final output is cast back to bf16).
//!
//! Regenerate fixtures via:
//!
//! ```text
//! cd ironmlx/tests/fixtures/p3b3_gated_delta_net && python gen_fixture.py
//! ```

use mlx::{Array, Dtype};

use ironmlx::nn::{Conv1d, Conv1dConfig, GatedDeltaNet, GatedDeltaNetConfig, Linear, RmsNormGated};

const FIXTURE_DIR: &str = "tests/fixtures/p3b3_gated_delta_net";

const HV: i32 = 4;
const HK: i32 = 2;
const DK: i32 = 32;
const DV: i32 = 32;
const HIDDEN: i32 = HV * DV; // 128
const CONV_DIM: i32 = HK * DK * 2 + HV * DV; // 64+64+128 = 256
const CONV_KERNEL: i32 = 4;

fn load(name: &str) -> Array {
    let path = format!("{FIXTURE_DIR}/{name}.npy");
    mlx::io::load_npy(&path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"))
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
fn gated_delta_net_matches_python_fixture() {
    let qkv_w = load("qkv_proj_weight");
    let z_w = load("z_proj_weight");
    let a_w = load("a_proj_weight");
    let b_w = load("b_proj_weight");
    let conv_w = load("conv1d_weight");
    let norm_w = load("norm_weight");
    let out_w = load("out_proj_weight");
    let a_log = load("A_log");
    let dt_bias = load("dt_bias");

    let cfg = GatedDeltaNetConfig {
        hidden_size: HIDDEN,
        num_v_heads: HV,
        num_k_heads: HK,
        head_k_dim: DK,
        head_v_dim: DV,
        conv_kernel_size: CONV_KERNEL,
        rms_norm_eps: 1e-6,
    };

    let conv1d = Conv1d::new(
        conv_w,
        None,
        Conv1dConfig {
            in_channels: CONV_DIM,
            out_channels: CONV_DIM,
            kernel_size: CONV_KERNEL,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: CONV_DIM,
        },
    );

    let gdn = GatedDeltaNet::from_components(
        Linear::new_fp(qkv_w, None),
        Linear::new_fp(z_w, None),
        Linear::new_fp(b_w, None),
        Linear::new_fp(a_w, None),
        conv1d,
        RmsNormGated::new(norm_w, cfg.rms_norm_eps),
        Linear::new_fp(out_w, None),
        a_log,
        dt_bias,
        cfg,
    );

    let x = load("input_x");
    let expected = load("expected_output");

    let out = gdn.forward(&x, None, None).expect("forward");

    assert_eq!(out.shape().as_slice(), expected.shape().as_slice());
    let err = max_abs_diff(&out, &expected);
    assert!(
        err < 1e-3,
        "GatedDeltaNet output max abs diff = {err} > 1e-3"
    );
}
