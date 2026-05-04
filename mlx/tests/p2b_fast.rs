//! Integration tests for mlx::fast — fused MLX kernels for Transformer inference.

use mlx::{fast, Array};

#[test]
fn rms_norm_no_weight_known_values() {
    // x = [[1.0, 2.0, 3.0, 4.0]], shape [1, 4]
    // mean(x^2) = (1+4+9+16)/4 = 7.5
    // sqrt(7.5 + 1e-5) ≈ 2.7386140
    // Expected output ≈ x / 2.7386 = [0.36514, 0.73029, 1.09543, 1.46059]
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]).expect("x");
    let out = fast::rms_norm(&x, None, 1e-5).expect("rms_norm");
    assert_eq!(out.shape().as_slice(), &[1, 4]);

    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let expected = [0.36514_f32, 0.73029, 1.09543, 1.46059];
    for (i, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "rms_norm[{i}] = {got}, want {want}"
        );
    }
}

#[test]
fn rms_norm_with_weight_scales_output() {
    // Same x as above; weight = [2.0, 2.0, 2.0, 2.0] → output = 2 × no-weight result.
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]).expect("x");
    let w = Array::from_slice(&[2.0_f32, 2.0, 2.0, 2.0], &[4]).expect("w");
    let out = fast::rms_norm(&x, Some(&w), 1e-5).expect("rms_norm");
    let v: Vec<f32> = out.to_vec().expect("to_vec");
    let expected = [0.73029_f32, 1.46058, 2.19087, 2.92117];
    for (i, (got, want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "rms_norm_w[{i}] = {got}, want {want}"
        );
    }
}
