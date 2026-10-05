//! Default-off masked causal softmax SDPA candidate: bit parity with MLX's
//! op-level D256 fallback (where + precise softmax) on prefill geometries.
use mlx::{fast, ops, random, Array, Dtype};

fn replica(q: &Array, k: &Array, v: &Array) -> Array {
    let (hq, ql, d) = (
        q.shape().as_slice()[1],
        q.shape().as_slice()[2],
        q.shape().as_slice()[3],
    );
    let (hk, kv) = (k.shape().as_slice()[1], k.shape().as_slice()[2]);
    let scale = ops::astype(
        &Array::try_from((&[1.0_f32 / 16.0][..], &[1][..])).unwrap(),
        Dtype::Bfloat16,
    )
    .unwrap();
    let qs = ops::reshape(
        &ops::multiply(&scale, q).unwrap(),
        &[1, hk, hq / hk, ql, d][..],
    )
    .unwrap();
    let k5 = ops::expand_dims(k, 2).unwrap();
    let v5 = ops::expand_dims(v, 2).unwrap();
    let scores = ops::matmul(
        &qs,
        &ops::transpose_axes(&k5, &[0, 1, 2, 4, 3][..]).unwrap(),
    )
    .unwrap();
    let offset = kv - ql;
    let qi = ops::arange(offset as f64, (offset + ql) as f64, 1.0, Dtype::Int32).unwrap();
    let ki = ops::arange(0.0, kv as f64, 1.0, Dtype::Int32).unwrap();
    let mask = ops::greater_equal(
        &ops::expand_dims(&qi, 1).unwrap(),
        &ops::expand_dims(&ki, 0).unwrap(),
    )
    .unwrap();
    let min = ops::astype(
        &Array::try_from((&[-3.389_531_4e38_f32][..], &[1][..])).unwrap(),
        Dtype::Bfloat16,
    )
    .unwrap();
    let probs = ops::softmax(&ops::where_(&mask, &scores, &min).unwrap(), &[-1][..], true).unwrap();
    ops::flatten(&ops::matmul(&probs, &v5).unwrap(), 1, 2).unwrap()
}

#[test]
#[ignore]
fn masked_causal_sdpa_matches_mlx_fallback_bitwise() {
    std::env::set_var("IRONMLX_EXPERIMENTAL_PREFILL_MASKED_SOFTMAX", "1");
    if let Ok(root) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
    }
    let key = random::key(20261006).unwrap();
    let (k0, k1) = random::split(&key).unwrap();
    let (k1, k2) = random::split(&k1).unwrap();
    let normal = |shape: &[i32], key: &Array| {
        let a = random::normal()
            .shape(shape)
            .dtype(Dtype::Float32)
            .key(key)
            .sample()
            .unwrap();
        ops::astype(&a, Dtype::Bfloat16).unwrap()
    };
    let k_all = normal(&[1, 4, 36876, 256], &k1);
    let v_all = normal(&[1, 4, 36876, 256], &k2);
    for (ql, kv) in [
        (2048, 2048),
        (2048, 4096),
        (2048, 6144),
        (1000, 7146),
        (2048, 32768),
        (2048, 32780),
    ] {
        // Token-major projection layout, transposed: non-contiguous Q as in the model.
        let q = ops::transpose_axes(&normal(&[1, ql, 24, 256], &k0), &[0, 2, 1, 3][..]).unwrap();
        let k = ops::slice(&k_all, &[0, 0, 0, 0][..], &[1, 4, kv, 256][..]).unwrap();
        let v = ops::slice(&v_all, &[0, 0, 0, 0][..], &[1, 4, kv, 256][..]).unwrap();
        let expected = replica(&q, &k, &v);
        let actual =
            fast::scaled_dot_product_attention(&q, &k, &v, 1.0 / 16.0, "causal", None, None)
                .unwrap();
        let a = ops::astype(&expected, Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let b = ops::astype(&actual, Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let mismatches = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        eprintln!(
            "masked sdpa L={ql} S={kv} elements={} mismatches={mismatches}",
            a.len()
        );
        assert_eq!(mismatches, 0, "L={ql} S={kv}");
    }
}
