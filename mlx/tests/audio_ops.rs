use mlx::ops::fft::{irfft, rfft, FftNorm};
use mlx::{ops, Array};
fn tensor(data: &[f32], shape: &[i32]) -> Array {
    Array::try_from((data, shape)).unwrap()
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() < 2e-4, "{i}: {a} != {b}");
    }
}
#[test]
fn convolution_numeric_and_real_fft_roundtrip() {
    // Scalar reference convolution, non-square input/stride and two output channels.
    let x: Vec<f32> = (0..40).map(|i| (i as f32 - 17.) / 7.).collect();
    let w: Vec<f32> = (0..24).map(|i| (i as f32 - 11.) / 9.).collect();
    let actual = ops::conv2d(
        &tensor(&x, &[1, 4, 5, 2]),
        &tensor(&w, &[2, 2, 3, 2]),
        (2, 1),
        (0, 1),
        (1, 1),
        1,
    )
    .unwrap();
    assert_eq!(actual.shape().as_slice(), &[1, 2, 5, 2]);
    let mut expected = Vec::new();
    for h in 0..2 {
        for j in 0..5 {
            for c in 0..2 {
                let mut sum = 0.;
                for kh in 0..2 {
                    for kw in 0..3 {
                        for ci in 0..2 {
                            let col = j as isize + kw as isize - 1;
                            if (0..5).contains(&col) {
                                sum += x[((h * 2 + kh) * 5 + col as usize) * 2 + ci]
                                    * w[((c * 2 + kh) * 3 + kw) * 2 + ci];
                            }
                        }
                    }
                }
                expected.push(sum);
            }
        }
    }
    close(&actual.to_vec::<f32>().unwrap(), &expected);
    // Transpose convolution includes overlap, dilation, asymmetric right output padding.
    let y = ops::conv_transpose1d(
        &tensor(&[1., 2., 3.], &[1, 3, 1]),
        &tensor(&[2., -1., 3.], &[1, 3, 1]),
        2,
        1,
        2,
        1,
        1,
    )
    .unwrap();
    let mut expected = vec![0.; 8];
    for (i, v) in [1., 2., 3.].iter().enumerate() {
        for (k, w) in [2., -1., 3.].iter().enumerate() {
            let pos = (i * 2 + k * 2) as isize - 1;
            if (0..8).contains(&pos) {
                expected[pos as usize] += v * w;
            }
        }
    }
    close(&y.to_vec::<f32>().unwrap(), &expected);
    // Distinct input/output channels expose transposed weight-layout mistakes.
    let x = [1., -2., 3., 4., -5., 6.];
    // Dyadic values isolate channel/layout correctness from backend precision.
    let w: Vec<f32> = (0..18).map(|i| (i as f32 - 7.) / 4.).collect();
    let y = ops::conv_transpose1d(
        &tensor(&x, &[1, 3, 2]),
        &tensor(&w, &[3, 3, 2]),
        2,
        1,
        1,
        1,
        1,
    )
    .unwrap();
    assert_eq!(y.shape().as_slice(), &[1, 6, 3]);
    let mut expected = vec![0.; 18];
    for i in 0..3 {
        for k in 0..3 {
            let pos = (i * 2 + k) as isize - 1;
            if (0..6).contains(&pos) {
                for co in 0..3 {
                    for ci in 0..2 {
                        expected[pos as usize * 3 + co] += x[i * 2 + ci] * w[(co * 3 + k) * 2 + ci];
                    }
                }
            }
        }
    }
    close(&y.to_vec::<f32>().unwrap(), &expected);
    // DFT magnitudes are independently evaluated for odd and even lengths, not just shape tested.
    for n in [5, 8, 17] {
        for norm in [FftNorm::Backward, FftNorm::Ortho, FftNorm::Forward] {
            let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let a = tensor(&signal, &[n]);
            let f = rfft(&a, n, -1, norm).unwrap();
            let scale = match norm {
                FftNorm::Backward => 1.,
                FftNorm::Ortho => 1. / (n as f32).sqrt(),
                FftNorm::Forward => 1. / n as f32,
            };
            let magnitude: Vec<f32> = (0..n / 2 + 1)
                .map(|k| {
                    let (mut re, mut im) = (0., 0.);
                    for (j, x) in signal.iter().enumerate() {
                        let angle = -2. * std::f32::consts::PI * k as f32 * j as f32 / n as f32;
                        re += x * angle.cos();
                        im += x * angle.sin();
                    }
                    (re * re + im * im).sqrt() * scale
                })
                .collect();
            close(&ops::abs(&f).unwrap().to_vec::<f32>().unwrap(), &magnitude);
            close(
                &irfft(&f, n, -1, norm).unwrap().to_vec::<f32>().unwrap(),
                &signal,
            );
        }
    }
    assert!(rfft(&tensor(&[1.], &[1]), 0, -1, FftNorm::Backward).is_err());
    assert!(ops::conv2d(
        &tensor(&[1.], &[1]),
        &tensor(&[1.], &[1]),
        (1, 1),
        (0, 0),
        (1, 1),
        1
    )
    .is_err());
}
