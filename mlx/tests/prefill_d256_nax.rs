//! Opt-in production-FFI test with strided Q and sliced/pitched KV storage.
//! Analytical causal mean, not bit-exact native attention or model qualification.
use mlx::{fast, ops, Array, Device, Dtype};

#[test]
#[ignore = "requires separately pinned D256 NAX metallib and M5+ GPU"]
fn strided_prefill_matches_analytical_causal_mean() {
    assert!(std::env::var("IRONMLX_EXPERIMENTAL_PREFILL_D256_NAX_METALLIB").is_ok());
    if let Ok(path) = std::env::var("IRONMLX_TEST_METALLIB") {
        mlx::metal::set_metallib_path(&path).unwrap();
    }
    mlx::set_default_device(Device::gpu(0));
    let qlen = 2048_i32;
    for klen in [2048_i32, 32768, 32780] {
        let q = Array::zeros((1_i32, qlen, 24_i32, 256_i32), Dtype::Bfloat16)
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3][..])
            .unwrap();
        let backing_len = klen + 32;
        let k = Array::zeros((1_i32, backing_len, 4_i32, 256_i32), Dtype::Bfloat16)
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3][..])
            .unwrap();
        let mut values = Vec::with_capacity((backing_len * 4 * 256) as usize);
        for row in 0..backing_len {
            for head in 0..4 {
                for dim in 0..256 {
                    values.push(((row % 17 - 8) + head * 2 + dim % 5) as f32 / 16.0);
                }
            }
        }
        let v = Array::try_from((values.as_slice(), &[1, backing_len, 4, 256][..]))
            .unwrap()
            .astype(Dtype::Bfloat16)
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3][..])
            .unwrap();
        // Offset three rows into a wider underlying interleaved cache tensor.
        let k = ops::indexing::slice(&k, (0, 0, 3, 0), (1, 4, 3 + klen, 256)).unwrap();
        let v = ops::indexing::slice(&v, (0, 0, 3, 0), (1, 4, 3 + klen, 256)).unwrap();
        mlx::transforms::eval(&[&q, &k, &v]).unwrap();
        let out = fast::scaled_dot_product_attention(&q, &k, &v, 0.0625, "causal", None, None)
            .unwrap()
            .astype(Dtype::Float32)
            .unwrap();
        let result = out.to_vec::<f32>().unwrap();
        let mut prefix = vec![0_i64; (klen + 1) as usize];
        for row in 0..klen {
            prefix[(row + 1) as usize] = prefix[row as usize] + ((row + 3) % 17 - 8) as i64;
        }
        let mut max_abs = 0.0_f32;
        for head in 0..24_i32 {
            for row in 0..qlen {
                let count = klen - qlen + row + 1;
                let mean = prefix[count as usize] as f32 / count as f32;
                for dim in 0..256_i32 {
                    let expected = (mean + (head / 6 * 2 + dim % 5) as f32) / 16.0;
                    let actual = result[((head * qlen + row) * 256 + dim) as usize];
                    assert!(actual.is_finite());
                    max_abs = max_abs.max((actual - expected).abs());
                }
            }
        }
        eprintln!("strided NAX klen={klen}, max_abs={max_abs}");
        assert!(
            max_abs <= 1.0 / 256.0,
            "causal mean/offset/head-stride mismatch"
        );
    }
}
