//! Diagnostic probe: fresh-allocation cost of growing-KV native D256 prefill
//! attention versus same-size buffer reuse. Not a correctness or API test.
use mlx::{fast, memory, ops, random, Array, Dtype};
use std::time::Instant;

fn sdpa(q: &Array, k_all: &Array, v_all: &Array, kv: i32) -> Array {
    let k = ops::slice(k_all, &[0, 0, 0, 0][..], &[1, 4, kv, 256][..]).unwrap();
    let v = ops::slice(v_all, &[0, 0, 0, 0][..], &[1, 4, kv, 256][..]).unwrap();
    fast::scaled_dot_product_attention(q, &k, &v, 1.0 / 16.0, "causal", None, None).unwrap()
}

#[test]
#[ignore]
fn growing_kv_attention_allocation_cost() {
    if let Ok(root) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
    }
    let key = random::key(20261002).unwrap();
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
    let q = normal(&[1, 24, 2048, 256], &k0);
    let k_all = normal(&[1, 4, 36876, 256], &k1);
    let v_all = normal(&[1, 4, 36876, 256], &k2);
    mlx::eval(&[&q, &k_all, &v_all]).unwrap();
    let layers = 4;
    for round in 0..2 {
        mlx::clear_cache();
        for chunk in 1..=16 {
            let kv = 2048 * chunk;
            let before = memory::snapshot();
            let mut walls = Vec::new();
            for _ in 0..layers {
                let started = Instant::now();
                let out = sdpa(&q, &k_all, &v_all, kv);
                mlx::eval(&[&out]).unwrap();
                walls.push(started.elapsed().as_micros());
            }
            let after = memory::snapshot();
            eprintln!(
                "round={round} kv={kv} first_us={} rest_us={:?} cache_before={} cache_after={}",
                walls[0],
                &walls[1..],
                before.cache_bytes,
                after.cache_bytes
            );
        }
    }
}

/// Diagnostic: stage costs of the native D256 causal fallback, using an
/// op-for-op replica that must match `fast::scaled_dot_product_attention`
/// bit-for-bit at the probed shape.
#[test]
#[ignore]
fn fallback_stage_costs_and_replica_parity() {
    if let Ok(root) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
    }
    let key = random::key(20261003).unwrap();
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
    let q = normal(&[1, 24, 2048, 256], &k0);
    let k_all = normal(&[1, 4, 36876, 256], &k1);
    let v_all = normal(&[1, 4, 36876, 256], &k2);
    mlx::eval(&[&q, &k_all, &v_all]).unwrap();
    let timed = |label: &str, f: &dyn Fn() -> Array| -> Array {
        let started = Instant::now();
        let out = f();
        mlx::eval(&[&out]).unwrap();
        eprintln!("stage {label} us={}", started.elapsed().as_micros());
        out
    };
    for kv in [8192, 32768] {
        for round in 0..3 {
            let k = ops::slice(&k_all, &[0, 0, 0, 0][..], &[1, 4, kv, 256][..]).unwrap();
            let v = ops::slice(&v_all, &[0, 0, 0, 0][..], &[1, 4, kv, 256][..]).unwrap();
            let scale = ops::astype(
                &Array::try_from((&[1.0_f32 / 16.0][..], &[1][..])).unwrap(),
                Dtype::Bfloat16,
            )
            .unwrap();
            eprintln!("kv={kv} round={round}");
            let qs = timed("q_scale", &|| {
                let qs = ops::multiply(&scale, &q).unwrap();
                ops::reshape(&qs, &[1, 4, 6, 2048, 256][..]).unwrap()
            });
            let k5 = ops::expand_dims(&k, 2).unwrap();
            let v5 = ops::expand_dims(&v, 2).unwrap();
            let scores = timed("qk_matmul", &|| {
                ops::matmul(
                    &qs,
                    &ops::transpose_axes(&k5, &[0, 1, 2, 4, 3][..]).unwrap(),
                )
                .unwrap()
            });
            let masked = timed("mask_where", &|| {
                let offset = kv - 2048;
                let qi =
                    ops::arange(offset as f64, (offset + 2048) as f64, 1.0, Dtype::Int32).unwrap();
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
                ops::where_(&mask, &scores, &min).unwrap()
            });
            let probs = timed("softmax", &|| {
                ops::softmax(&masked, &[-1][..], true).unwrap()
            });
            let out = timed("pv_matmul", &|| {
                ops::flatten(&ops::matmul(&probs, &v5).unwrap(), 1, 2).unwrap()
            });
            let fused = timed("fast_sdpa_total", &|| {
                fast::scaled_dot_product_attention(&q, &k, &v, 1.0 / 16.0, "causal", None, None)
                    .unwrap()
            });
            let a = ops::astype(&out, Dtype::Float32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let b = ops::astype(&fused, Dtype::Float32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let mismatches = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            eprintln!(
                "replica kv={kv} elements={} mismatches={mismatches}",
                a.len()
            );
            assert_eq!(mismatches, 0);
        }
    }
}
