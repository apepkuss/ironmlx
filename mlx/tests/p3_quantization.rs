//! Integration tests for mlx::quantization — low-precision subsystem.

use mlx::quantization::{dequantize, quantize};
use mlx::Array;

/// 构造 [N=4, K=64] f32 测试权重矩阵（K=64 = 默认 group_size）。
fn make_test_weight() -> Array {
    let total: usize = 256; // 4 * 64
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 1.0).collect();
    Array::from_slice(&data, &[4, 64]).expect("weight")
}

#[test]
fn quantize_affine_4bit_returns_three_arrays() {
    let w = make_test_weight();
    let result = quantize(&w, Some(64), Some(4), "affine", None).expect("quantize");
    // affine 模式下应返回 [packed_weights, scales, biases]
    assert_eq!(result.len(), 3, "affine quantize should return 3 arrays");
}

#[test]
fn quantize_dequantize_round_trip_4bit() {
    let w = make_test_weight();
    let v_in: Vec<f32> = w.to_vec().expect("w to_vec");

    let parts = quantize(&w, Some(64), Some(4), "affine", None).expect("quantize");
    assert_eq!(parts.len(), 3);

    let dequantized = dequantize(
        &parts[0],       // packed
        &parts[1],       // scales
        Some(&parts[2]), // biases
        Some(64),
        Some(4),
        "affine",
        None,
        None,
    )
    .expect("dequantize");

    let v_out: Vec<f32> = dequantized.to_vec().expect("dequantized to_vec");
    assert_eq!(v_in.len(), v_out.len());

    // 4-bit 量化误差容差较宽（典型 SQNR ~25 dB，相对误差几个百分点）
    let mut max_err = 0.0_f32;
    for (a, b) in v_in.iter().zip(&v_out) {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(max_err < 5e-2, "4-bit round-trip max err {max_err}");
}

#[test]
fn quantize_dequantize_round_trip_8bit() {
    let w = make_test_weight();
    let v_in: Vec<f32> = w.to_vec().expect("w to_vec");

    let parts = quantize(&w, Some(64), Some(8), "affine", None).expect("quantize");
    let dequantized = dequantize(
        &parts[0],
        &parts[1],
        Some(&parts[2]),
        Some(64),
        Some(8),
        "affine",
        None,
        None,
    )
    .expect("dequantize");

    let v_out: Vec<f32> = dequantized.to_vec().expect("to_vec");
    let mut max_err = 0.0_f32;
    for (a, b) in v_in.iter().zip(&v_out) {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(max_err < 5e-3, "8-bit round-trip max err {max_err}");
}
