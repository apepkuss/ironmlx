//! SDPA (scaled dot-product attention) integration tests.
//!
//! Implements the canonical attention algorithm using the cxx-mlx ops from
//! P0/P1a/P1b1/P1b2a/P1b2b. P2 will add `fast::scaled_dot_product_attention`
//! which should match these results numerically.

use mlx::{ops, Array, Result};

/// SDPA: out = softmax((Q @ K.T) * scale + mask) @ V
///
/// Q/K/V: [B, H, S, D]
/// mask: [S, S] additive (-inf in masked positions, 0 elsewhere) — broadcasts on B, H
/// Returns: [B, H, S, D]
fn sdpa(
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    scale: f32,
) -> Result<Array> {
    // K.transpose(-1, -2): [B, H, S, D] → [B, H, D, S]
    let kt = k.transpose_axes(&[0, 1, 3, 2])?;
    let scores = q.matmul(&kt)?;
    let scaled = (&scores * scale)?;
    let masked = match mask {
        Some(m) => (&scaled + m)?,
        None => scaled,
    };
    // Softmax along last axis
    let m = ops::max(&masked, -1, true)?;
    let shifted = (&masked - &m)?;
    let e = shifted.exp()?;
    let s = ops::sum(&e, -1, true)?;
    let weights = (&e / &s)?;
    weights.matmul(v)
}

/// Build a causal mask of shape [S, S]: 0 on/below diagonal, -inf above.
fn causal_mask(s: usize) -> Result<Array> {
    let mut data = Vec::with_capacity(s * s);
    for i in 0..s {
        for j in 0..s {
            data.push(if j <= i { 0.0_f32 } else { f32::NEG_INFINITY });
        }
    }
    Array::from_slice(&data, &[s as i32, s as i32])
}

#[test]
fn sdpa_no_mask_shape_finite() {
    // [B=1, H=2, S=4, D=8]
    let total: usize = 64; // 1 * 2 * 4 * 8
    let q_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let k_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.02).collect();
    let v_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.03).collect();
    let q = Array::from_slice(&q_data, &[1, 2, 4, 8]).expect("q");
    let k = Array::from_slice(&k_data, &[1, 2, 4, 8]).expect("k");
    let v = Array::from_slice(&v_data, &[1, 2, 4, 8]).expect("v");

    let scale = 1.0 / (8.0_f32).sqrt();
    let out = sdpa(&q, &k, &v, None, scale).expect("sdpa");
    assert_eq!(out.shape().as_slice(), &[1, 2, 4, 8]);
    let v_out = out.to_vec::<f32>().expect("to_vec");
    for x in &v_out {
        assert!(x.is_finite(), "non-finite value in SDPA output: {x}");
    }
}

#[test]
fn sdpa_softmax_rows_sum_to_one() {
    // Verify the softmax-of-scores property: the attention weight rows sum to 1.
    // We compute weights directly (without the final V matmul) to inspect.
    let total: usize = 12; // 1 * 1 * 3 * 4
    let q_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
    let q = Array::from_slice(&q_data, &[1, 1, 3, 4]).expect("q");
    let k = Array::from_slice(&k_data, &[1, 1, 3, 4]).expect("k");

    let kt = k.transpose_axes(&[0, 1, 3, 2]).expect("kt");
    let scores = q.matmul(&kt).expect("matmul");
    let scaled = (&scores * (1.0 / 2.0_f32)).expect("scale");
    let m = ops::max(&scaled, -1, true).expect("max");
    let shifted = (&scaled - &m).expect("sub");
    let e = shifted.exp().expect("exp");
    let s = ops::sum(&e, -1, true).expect("sum");
    let weights = (&e / &s).expect("div");

    // Row sums of weights: sum over last axis with keepdim=false → [1, 1, 3]
    let row_sums = ops::sum(&weights, -1, false).expect("row_sums");
    let v = row_sums.to_vec::<f32>().expect("to_vec");
    for sum in &v {
        assert!((sum - 1.0).abs() < 1e-5, "row sum should be ~1.0, got {sum}");
    }
}

#[test]
fn sdpa_causal_mask_zeros_future() {
    // With a causal mask, attention weights for j > i (future positions) should be 0.
    // We compute weights manually (without V matmul) to inspect.
    let s = 4;
    let total: usize = s * 4; // 1 * 1 * s * 4
    let q_data: Vec<f32> = (0..total).map(|i| 0.1 * (i as f32)).collect();
    let k_data: Vec<f32> = (0..total).map(|i| 0.1 * (i as f32)).collect();
    let q = Array::from_slice(&q_data, &[1, 1, s as i32, 4]).expect("q");
    let k = Array::from_slice(&k_data, &[1, 1, s as i32, 4]).expect("k");
    let mask = causal_mask(s).expect("mask");

    let kt = k.transpose_axes(&[0, 1, 3, 2]).expect("kt");
    let scores = q.matmul(&kt).expect("matmul");
    let scaled = (&scores * 0.5_f32).expect("scale");
    let masked = (&scaled + &mask).expect("add mask");
    let m = ops::max(&masked, -1, true).expect("max");
    let shifted = (&masked - &m).expect("sub");
    let e = shifted.exp().expect("exp");
    let sum_e = ops::sum(&e, -1, true).expect("sum");
    let weights = (&e / &sum_e).expect("div");

    // Reshape to [S, S] for inspection
    let w_2d = weights.reshape(&[s as i32, s as i32]).expect("reshape");
    let v = w_2d.to_vec::<f32>().expect("to_vec");
    // For each row i, positions j > i should be ~0
    for i in 0..s {
        for j in 0..s {
            let val = v[i * s + j];
            if j > i {
                assert!(val.abs() < 1e-6, "w[{i},{j}] should be 0 (causal), got {val}");
            }
        }
    }
}

#[test]
fn sdpa_numerical_match_reference() {
    // Deterministic small input. Q=K=V=I (4x4, embedded in [1, 1, 4, 4]), scale=1, no mask.
    // Q @ K.T = I @ I = I (identity)
    // softmax of identity row [1, 0, 0, 0] = [exp(1), exp(0), exp(0), exp(0)] / sum
    //                                       = [e, 1, 1, 1] / (e + 3) ≈ [0.4754, 0.1749, 0.1749, 0.1749]
    // Then weights @ V where V=I → just weights themselves.
    let n = 4;
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    let identity_2d = Array::from_slice(&data, &[n as i32, n as i32]).expect("identity");
    let q = identity_2d.reshape(&[1, 1, n as i32, n as i32]).expect("reshape");
    let k = q.clone();
    let v = q.clone();

    let out = sdpa(&q, &k, &v, None, 1.0).expect("sdpa");
    let result = out.to_vec::<f32>().expect("to_vec");

    let e = std::f32::consts::E;
    let norm = e + 3.0;
    let expected_diag = e / norm;
    let expected_off = 1.0 / norm;

    // Check diagonal of each row equals expected_diag, off-diagonal equals expected_off
    for i in 0..n {
        for j in 0..n {
            let actual = result[i * n + j];
            let expected = if i == j { expected_diag } else { expected_off };
            assert!(
                (actual - expected).abs() < 1e-3,
                "out[{i},{j}] expected {expected}, got {actual}"
            );
        }
    }
}
