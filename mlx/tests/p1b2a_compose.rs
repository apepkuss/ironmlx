use mlx::{ops, Array, Result};

/// Softmax using max-subtraction trick for numerical stability.
fn softmax(x: &Array, axis: i32) -> Result<Array> {
    let m = ops::max(x, axis, true)?;
    let shifted = (x - &m)?;
    let e = shifted.exp()?;
    let s = ops::sum(&e, axis, true)?;
    &e / &s
}

/// Exact GELU using erf: 0.5 * x * (1 + erf(x / sqrt(2)))
fn gelu(x: &Array) -> Result<Array> {
    let sqrt_2 = std::f32::consts::SQRT_2;
    let half = (x * 0.5_f32)?;
    let inner = (x / sqrt_2)?.erf()?;
    let one_plus = (&inner + 1.0_f32)?;
    &half * &one_plus
}

/// SiLU (a.k.a. Swish): x * sigmoid(x)
fn silu(x: &Array) -> Result<Array> {
    let s = x.sigmoid()?;
    x * &s
}

#[test]
fn softmax_along_last_axis_sums_to_one() {
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]).expect("from_slice");
    let s = softmax(&x, -1).expect("softmax");
    let v = s.to_vec::<f32>().expect("to_vec");
    let total: f32 = v.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "sum should be ~1.0, got {total}");
    // Each value must be positive.
    for val in &v {
        assert!(*val > 0.0, "softmax value should be positive: {val}");
    }
    // Largest input → largest softmax value.
    assert!(v[2] > v[1] && v[1] > v[0]);
}

#[test]
fn softmax_2d_per_row() {
    // [[1, 2, 3], [4, 5, 6]] softmax along axis -1: each row sums to 1
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("from_slice");
    let s = softmax(&x, -1).expect("softmax");
    let v = s.to_vec::<f32>().expect("to_vec");
    assert!((v[0] + v[1] + v[2] - 1.0).abs() < 1e-6);
    assert!((v[3] + v[4] + v[5] - 1.0).abs() < 1e-6);
}

#[test]
fn gelu_at_known_points() {
    // gelu(0) ≈ 0
    let zero = Array::from_slice(&[0.0_f32], &[]).expect("from_slice");
    assert!((gelu(&zero).expect("gelu").item::<f32>().expect("item") - 0.0).abs() < 1e-6);

    // gelu(1) = 0.5 * 1 * (1 + erf(1/sqrt(2))) ≈ 0.8413
    let one = Array::from_slice(&[1.0_f32], &[]).expect("from_slice");
    let g = gelu(&one).expect("gelu").item::<f32>().expect("item");
    assert!((g - 0.8413).abs() < 1e-3, "gelu(1) ≈ 0.8413, got {g}");

    // gelu(-1) ≈ -0.1587
    let neg_one = Array::from_slice(&[-1.0_f32], &[]).expect("from_slice");
    let g_neg = gelu(&neg_one).expect("gelu").item::<f32>().expect("item");
    assert!((g_neg - (-0.1587)).abs() < 1e-3, "gelu(-1) ≈ -0.1587, got {g_neg}");
}

#[test]
fn silu_at_known_points() {
    // silu(0) = 0
    let zero = Array::from_slice(&[0.0_f32], &[]).expect("from_slice");
    assert!((silu(&zero).expect("silu").item::<f32>().expect("item") - 0.0).abs() < 1e-6);

    // silu(1) = 1 * sigmoid(1) ≈ 0.7311
    let one = Array::from_slice(&[1.0_f32], &[]).expect("from_slice");
    let s = silu(&one).expect("silu").item::<f32>().expect("item");
    assert!((s - 0.7311).abs() < 1e-3, "silu(1) ≈ 0.7311, got {s}");

    // silu(-2) = -2 * sigmoid(-2) ≈ -2 * 0.1192 ≈ -0.2384
    let neg_two = Array::from_slice(&[-2.0_f32], &[]).expect("from_slice");
    let s_neg = silu(&neg_two).expect("silu").item::<f32>().expect("item");
    assert!((s_neg - (-0.2384)).abs() < 1e-3, "silu(-2) ≈ -0.2384, got {s_neg}");
}
