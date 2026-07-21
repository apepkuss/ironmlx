//! Integration tests for mlx::quantization — low-precision subsystem.

use mlx::quantization::{dequantize, quantize, quantized_matmul, quantized_matmul_batch_isolated};
use mlx::Array;

/// 构造 [N=4, K=64] f32 测试权重矩阵（K=64 = 默认 group_size）。
fn make_test_weight() -> Array {
    let total: usize = 256; // 4 * 64
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 1.0).collect();
    Array::try_from((&data[..], &[4, 64][..])).expect("weight")
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

#[test]
fn quantized_matmul_matches_dequantize_matmul() {
    // W: [N=4, K=64], x: [B=2, K=64]
    // y_qmm = quantized_matmul(x, packed_W, scales, biases, transpose=true)
    // y_ref = x @ dequantize(packed_W).T
    // 两者应在 4-bit 量化容差内一致
    let w = make_test_weight(); // [4, 64]
    let x_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.005).collect();
    let x = Array::try_from((&x_data[..], &[2, 64][..])).expect("x");

    let parts = quantize(&w, Some(64), Some(4), "affine", None).expect("quantize");

    // y_qmm = x @ W.T (transpose=true)，输出 [2, 4]
    let y_qmm = quantized_matmul(
        &x,
        &parts[0],
        &parts[1],
        Some(&parts[2]),
        true,
        Some(64),
        Some(4),
        "affine",
    )
    .expect("qmm");
    assert_eq!(y_qmm.shape().as_slice(), &[2, 4]);

    // 参考路径: y_ref = x @ dequantize(W).T
    let dq = dequantize(
        &parts[0],
        &parts[1],
        Some(&parts[2]),
        Some(64),
        Some(4),
        "affine",
        None,
        None,
    )
    .expect("dq");
    let dq_t = dq.transpose_axes(&[1, 0]).expect("transpose");
    let y_ref = x.matmul(&dq_t).expect("ref matmul");

    let v_qmm: Vec<f32> = y_qmm.to_vec().expect("qmm to_vec");
    let v_ref: Vec<f32> = y_ref.to_vec().expect("ref to_vec");
    assert_eq!(v_qmm.len(), v_ref.len());

    // MLX qmm 走专用 4-bit kernel；相对 dense matmul 允许小幅累积路径差异。
    const MAX_QMM_ABS_ERR: f32 = 3e-2;
    const MAX_QMM_PEAK_REL_ERR: f32 = 1e-3;
    let mut max_err = 0.0_f32;
    let mut max_ref = 0.0_f32;
    for (a, b) in v_qmm.iter().zip(&v_ref) {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
        }
        if b.abs() > max_ref {
            max_ref = b.abs();
        }
    }
    let peak_rel_err = max_err / max_ref.max(1.0);
    assert!(
        max_err < MAX_QMM_ABS_ERR && peak_rel_err < MAX_QMM_PEAK_REL_ERR,
        "qmm vs ref max err {max_err}, peak relative err {peak_rel_err}"
    );
}

#[test]
fn batch_isolated_quantized_matmul_matches_rowwise_calls_exactly() {
    let (batch, rows, out_dim, in_dim) = (4_i32, 3_i32, 128_i32, 256_i32);
    let weight_data = (0..out_dim * in_dim)
        .map(|idx| ((idx % 41) as f32 - 20.0) * 0.0125)
        .collect::<Vec<_>>();
    let input_data = (0..batch * rows * in_dim)
        .map(|idx| ((idx % 37) as f32 - 18.0) * 0.0175)
        .collect::<Vec<_>>();
    let weight = Array::try_from((weight_data.as_slice(), &[out_dim, in_dim][..]))
        .expect("weight")
        .astype(mlx::Dtype::Bfloat16)
        .expect("bf16 weight");
    let input = Array::try_from((input_data.as_slice(), &[batch, rows, in_dim][..]))
        .expect("input")
        .astype(mlx::Dtype::Bfloat16)
        .expect("bf16 input");
    let q = quantize(&weight, Some(64), Some(4), "affine", None).expect("quantize");

    let actual = quantized_matmul_batch_isolated(
        &input,
        &q[0],
        &q[1],
        Some(&q[2]),
        true,
        Some(64),
        Some(4),
        "affine",
    )
    .expect("batch-isolated qmm");
    let mut expected_rows = Vec::with_capacity(batch as usize);
    for row in 0..batch {
        let input_row = mlx::ops::indexing::slice_strided(
            &input,
            &[row, 0_i32, 0],
            &[row + 1, rows, in_dim],
            &[1_i32, 1, 1],
        )
        .expect("slice input row");
        expected_rows.push(
            quantized_matmul(
                &input_row,
                &q[0],
                &q[1],
                Some(&q[2]),
                true,
                Some(64),
                Some(4),
                "affine",
            )
            .expect("rowwise qmm"),
        );
    }
    let refs = expected_rows.iter().collect::<Vec<_>>();
    let expected = mlx::ops::shape::concatenate(&refs, 0).expect("concatenate rows");

    assert_eq!(actual.shape().as_slice(), &[batch, rows, out_dim]);
    let actual = actual
        .astype(mlx::Dtype::Float32)
        .expect("cast actual to f32");
    let expected = expected
        .astype(mlx::Dtype::Float32)
        .expect("cast expected to f32");
    assert_eq!(
        actual.to_vec::<f32>().expect("actual values"),
        expected.to_vec::<f32>().expect("expected values")
    );
}

fn assert_affine_non_power_of_two_qmm_matches_reference(bits: i32, rows: i32) {
    let n = 16_i32;
    let k = 64_i32;
    let weight_data: Vec<f32> = (0..n * k)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.015)
        .collect();
    let input_data: Vec<f32> = (0..rows * k)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.02)
        .collect();
    let weight_f32 = Array::try_from((weight_data.as_slice(), &[n, k][..])).expect("weight");
    let input_f32 = Array::try_from((input_data.as_slice(), &[rows, k][..])).expect("input");
    let weight = weight_f32.astype(Dtype::Bfloat16).expect("bf16 weight");
    let input = input_f32.astype(Dtype::Bfloat16).expect("bf16 input");
    let parts = quantize(&weight, Some(64), Some(bits), "affine", None).expect("quantize");
    assert_eq!(parts[0].shape().as_slice(), &[n, k * bits / 32]);

    let got = quantized_matmul(
        &input,
        &parts[0],
        &parts[1],
        Some(&parts[2]),
        true,
        Some(64),
        Some(bits),
        "affine",
    )
    .expect("qmm");
    let dequantized = dequantize(
        &parts[0],
        &parts[1],
        Some(&parts[2]),
        Some(64),
        Some(bits),
        "affine",
        None,
        None,
    )
    .expect("dequantize");
    let expected = input
        .matmul(&dequantized.transpose().expect("transpose"))
        .expect("dense reference");

    let got = got
        .astype(Dtype::Float32)
        .expect("cast qmm")
        .to_vec::<f32>()
        .expect("qmm values");
    let expected = expected
        .astype(Dtype::Float32)
        .expect("cast reference")
        .to_vec::<f32>()
        .expect("reference values");
    let mut max_abs = 0.0_f32;
    let mut max_ref = 0.0_f32;
    for (actual, reference) in got.iter().zip(&expected) {
        assert!(actual.is_finite() && reference.is_finite());
        max_abs = max_abs.max((actual - reference).abs());
        max_ref = max_ref.max(reference.abs());
    }
    let peak_relative = max_abs / max_ref.max(1.0);
    assert!(
        max_abs <= 0.25 && peak_relative <= 0.01,
        "bits={bits} rows={rows} max_abs={max_abs} peak_relative={peak_relative}"
    );
}

#[test]
fn affine_non_power_of_two_qmm_matches_dequantized_reference() {
    for bits in [5, 6] {
        for rows in [1, 64] {
            assert_affine_non_power_of_two_qmm_matches_reference(bits, rows);
        }
    }
}

use mlx::quantization::qqmm;

#[test]
fn qqmm_binding_smoke() {
    // Smoke test: validates binding wiring only (shim → bridge → safe API).
    // Does NOT validate NVFP4 kernel correctness — at the time of writing,
    // MLX's QQMatmul kernel returns "[QQMatmul] NYI for the general case"
    // on the macOS Metal backend.
    //
    // The test tolerates Err only if the message contains "NYI"; any other
    // failure mode (real regression, invalid bindings) panics.
    //
    // TODO: when MLX lands NVFP4 Metal kernel, replace this with a real
    // round-trip test (similar to `quantized_matmul_matches_dequantize_matmul`).
    let w = make_test_weight();
    let parts = quantize(&w, Some(64), Some(4), "affine", None).expect("quantize");
    let x_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.005).collect();
    let x = Array::try_from((&x_data[..], &[2, 64][..])).expect("x");

    let result = qqmm(
        &x,
        &parts[0],
        Some(&parts[1]),
        Some(64),
        Some(4),
        "nvfp4",
        None,
        None,
    );

    match result {
        Ok(y) => match y.to_vec::<f32>() {
            Ok(v) => {
                for x in &v {
                    assert!(x.is_finite(), "non-finite value: {x}");
                }
            }
            Err(e) => {
                let msg = format!("{e:?}");
                assert!(
                    msg.contains("NYI"),
                    "qqmm eval failed with non-NYI error (real regression?): {msg}"
                );
            }
        },
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("NYI"),
                "qqmm construction failed with non-NYI error (real regression?): {msg}"
            );
        }
    }
}

use mlx::quantization::gather_quantized_matmul;

#[test]
fn gather_quantized_matmul_no_indices_binding_smoke() {
    // gather_quantized_matmul 不传 lhs/rhs indices 时退化为常规 quantized_matmul。
    // 本测试验证 binding wiring：仅容忍 NYI 错误（与 qqmm_binding_smoke 对齐）。
    //
    // TODO: 当 MLX 在 Metal 后端完整支持 gather_quantized_matmul（含 indices 路径）时，
    // 加一个 indices=Some 的 round-trip 测试。
    let w = make_test_weight();
    let parts = quantize(&w, Some(64), Some(4), "affine", None).expect("quantize");
    let x_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.005).collect();
    let x = Array::try_from((&x_data[..], &[2, 64][..])).expect("x");

    let result = gather_quantized_matmul(
        &x,
        &parts[0],
        &parts[1],
        Some(&parts[2]),
        None, // lhs_indices
        None, // rhs_indices
        true, // transpose
        Some(64),
        Some(4),
        "affine",
        false, // sorted_indices
    );

    match result {
        Ok(y) => match y.to_vec::<f32>() {
            Ok(v) => {
                for x in &v {
                    assert!(x.is_finite(), "non-finite value: {x}");
                }
            }
            Err(e) => {
                let msg = format!("{e:?}");
                assert!(
                    msg.contains("NYI"),
                    "gather_quantized_matmul eval failed with non-NYI error (real regression?): {msg}"
                );
            }
        },
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("NYI"),
                "gather_quantized_matmul construction failed with non-NYI error (real regression?): {msg}"
            );
        }
    }
}

#[test]
fn gather_quantized_matmul_affine_non_power_of_two_bits_matches_reference() {
    let experts = 2_i32;
    let n = 4_i32;
    let k = 64_i32;
    let batch = 2_i32;
    let weight_data: Vec<f32> = (0..experts * n * k)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.01)
        .collect();
    let input_data: Vec<f32> = (0..batch * k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.015)
        .collect();
    let weight = Array::try_from((weight_data.as_slice(), &[experts, n, k][..])).expect("weight");
    let input = Array::try_from((input_data.as_slice(), &[batch, 1, 1, k][..])).expect("input");
    let expert_indices = Array::try_from((&[0_u32, 1][..], &[batch, 1][..])).expect("indices");

    for bits in [5, 6] {
        let parts = quantize(&weight, Some(64), Some(bits), "affine", None).expect("quantize");
        let got = gather_quantized_matmul(
            &input,
            &parts[0],
            &parts[1],
            Some(&parts[2]),
            None,
            Some(&expert_indices),
            true,
            Some(64),
            Some(bits),
            "affine",
            false,
        )
        .expect("gather qmm")
        .to_vec::<f32>()
        .expect("gather qmm values");
        let dequantized = dequantize(
            &parts[0],
            &parts[1],
            Some(&parts[2]),
            Some(64),
            Some(bits),
            "affine",
            None,
            None,
        )
        .expect("dequantize")
        .to_vec::<f32>()
        .expect("dequantized values");

        let mut expected = Vec::with_capacity((batch * n) as usize);
        for batch_idx in 0..batch as usize {
            let expert = batch_idx;
            for output_idx in 0..n as usize {
                let mut sum = 0.0_f32;
                for input_idx in 0..k as usize {
                    let x = input_data[batch_idx * k as usize + input_idx];
                    let w =
                        dequantized[(expert * n as usize + output_idx) * k as usize + input_idx];
                    sum += x * w;
                }
                expected.push(sum);
            }
        }

        assert_eq!(got.len(), expected.len());
        for (idx, (actual, reference)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (actual - reference).abs() <= 0.03,
                "bits={bits} idx={idx} actual={actual} reference={reference}"
            );
        }
    }
}

use mlx::quantization::{from_fp8, to_fp8};
use mlx::Dtype;

#[test]
fn fp8_round_trip_f32_small_integers() {
    // 小整数 1.0/2.0/3.0/4.0 在 E4M3 (4-exp 3-mantissa) 范围内可精确或近似表达。
    // E4M3 mantissa 仅 3-bit，相对误差典型 ~6-12%；容差 0.5 安全（绝对误差对小值）。
    let x = Array::try_from((&[1.0_f32, 2.0, 3.0, 4.0][..], &[4][..])).expect("x");
    let fp8 = to_fp8(&x).expect("to_fp8");

    let back = from_fp8(&fp8, Dtype::Float32).expect("from_fp8");
    assert_eq!(back.shape().as_slice(), &[4]);

    let v_back: Vec<f32> = back.to_vec().expect("to_vec");
    let expected = [1.0_f32, 2.0, 3.0, 4.0];
    for (i, (got, want)) in v_back.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 0.5,
            "fp8 round-trip[{i}] = {got}, want {want}"
        );
    }
}

#[test]
fn submodule_path_works() {
    // 验证通过 mlx::quantization::* 子模块路径访问 P3 公开 API
    let w = make_test_weight();
    let parts =
        mlx::quantization::quantize(&w, Some(64), Some(4), "affine", None).expect("submodule");
    assert_eq!(parts.len(), 3);
}
