//! GPU qualification across affine group sizes and routed expert shapes.
//! References consume the same dequantized weights, so quantization error is
//! excluded from the comparison. Run serially against the pinned MLX build.
use mlx::quantization::{dequantize, gather_quantized_matmul_on, quantize_on, quantized_matmul_on};
use mlx::{Array, Device, Dtype};

fn data(len: i32, period: i32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 17 % period) as f32 - (period / 2) as f32) * scale)
        .collect()
}

fn values(x: &Array) -> Vec<f32> {
    x.astype(Dtype::Float32).unwrap().to_vec().unwrap()
}

fn check(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len());
    let mut error = 0.0f64;
    let mut energy = 0.0f64;
    let mut peak = 0.0f32;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite(),
            "{context}: nonfinite result"
        );
        let e = (a - b) as f64;
        error += e * e;
        energy += (b as f64).powi(2);
        peak = peak.max((a - b).abs());
    }
    let relative = (error / energy.max(1e-12)).sqrt();
    eprintln!("{context}: relative_l2={relative:.6}, max_abs={peak:.6}");
    assert!(
        relative < 0.015 && peak < 0.125,
        "{context}: relative_l2={relative}, max_abs={peak}"
    );
}

#[test]
fn gpu_affine_qmm_flashnext_dimensions_match_dequantized_reference() {
    for (n, k) in [(640, 2560), (2560, 640)] {
        let w = Array::try_from((data(n * k, 127, 0.003).as_slice(), &[n, k][..]))
            .unwrap()
            .astype(Dtype::Bfloat16)
            .unwrap();
        for group in [32, 64] {
            let q = quantize_on(&w, Some(group), Some(4), "affine", None, Device::gpu(0)).unwrap();
            let reference_w = dequantize(
                &q[0],
                &q[1],
                Some(&q[2]),
                Some(group),
                Some(4),
                "affine",
                None,
                Some(Dtype::Float32),
            )
            .unwrap()
            .transpose_axes(&[1, 0])
            .unwrap();
            for rows in [1, 8, 32, 129] {
                let x = Array::try_from((data(rows * k, 73, 0.004).as_slice(), &[rows, k][..]))
                    .unwrap()
                    .astype(Dtype::Bfloat16)
                    .unwrap();
                let actual = quantized_matmul_on(
                    &x,
                    &q[0],
                    &q[1],
                    Some(&q[2]),
                    true,
                    Some(group),
                    Some(4),
                    "affine",
                    Device::gpu(0),
                )
                .unwrap();
                let reference = x
                    .astype(Dtype::Float32)
                    .unwrap()
                    .matmul_on(&reference_w, Device::gpu(0))
                    .unwrap();
                check(
                    &values(&actual),
                    &values(&reference),
                    &format!("qmm gs={group} M={rows} N={n} K={k}"),
                );
            }
        }
    }
}

#[test]
fn gpu_affine_expert_gather_handles_duplicate_and_unused_experts() {
    let (experts, n, k, tokens, topk) = (8, 128, 640, 17, 4);
    let w = Array::try_from((
        data(experts * n * k, 113, 0.003).as_slice(),
        &[experts, n, k][..],
    ))
    .unwrap()
    .astype(Dtype::Bfloat16)
    .unwrap();
    let x = Array::try_from((
        data(tokens * k, 71, 0.005).as_slice(),
        &[tokens, 1, 1, k][..],
    ))
    .unwrap()
    .astype(Dtype::Bfloat16)
    .unwrap();
    // Repeated experts, uneven occupancy, and unused experts 1, 3, 5, 7.
    let routes: Vec<u32> = (0..tokens * topk)
        .map(|i| [6, 0, 6, 2, 4][i as usize % 5])
        .collect();
    let indices = Array::try_from((routes.as_slice(), &[tokens, topk][..])).unwrap();
    let input = values(&x);
    for group in [32, 64] {
        let q = quantize_on(&w, Some(group), Some(4), "affine", None, Device::gpu(0)).unwrap();
        let actual = gather_quantized_matmul_on(
            &x,
            &q[0],
            &q[1],
            Some(&q[2]),
            None,
            Some(&indices),
            true,
            Some(group),
            Some(4),
            "affine",
            false,
            Device::gpu(0),
        )
        .unwrap();
        let weights = values(
            &dequantize(
                &q[0],
                &q[1],
                Some(&q[2]),
                Some(group),
                Some(4),
                "affine",
                None,
                Some(Dtype::Float32),
            )
            .unwrap(),
        );
        let mut expected = Vec::new();
        for token in 0..tokens as usize {
            for route in 0..topk as usize {
                let expert = routes[token * topk as usize + route] as usize;
                for output in 0..n as usize {
                    let mut acc = 0f64;
                    for column in 0..k as usize {
                        acc += input[token * k as usize + column] as f64
                            * weights[(expert * n as usize + output) * k as usize + column] as f64;
                    }
                    expected.push(acc as f32);
                }
            }
        }
        check(&values(&actual), &expected, &format!("gather gs={group}"));
    }
}
