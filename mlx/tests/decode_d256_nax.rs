//! Opt-in test of the split-K BF16 D256 NAX attention for short query blocks
//! over long caches: each row's output is bit-identical whatever block it is
//! computed in (rows 1..=8, any position), and its error against an f64
//! reference is no worse than MLX's own vector attention.
use mlx::{fast, ops, Array, Device, Dtype};

const HQ: i32 = 16;
const HKV: i32 = 2;
const D: i32 = 256;
const MIN_KEYS: i32 = 8193;

fn uniform(seed: &mut u64, scale: f32) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
}

fn bf16(values: &[f32], shape: &[i32]) -> Array {
    Array::try_from((values, shape))
        .unwrap()
        .astype(Dtype::Bfloat16)
        .unwrap()
}

fn f32s(a: &Array) -> Vec<f32> {
    a.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap()
}

/// Causal attention of query rows [first, first + rows) whose last row sees
/// `keys` keys; `mask` routes it to MLX's own kernel instead.
fn attend(q: &Array, k: &Array, v: &Array, first: i32, rows: i32, keys: i32, mask: bool) -> Array {
    let q = ops::indexing::slice(q, (0, 0, first, 0), (1, HQ, first + rows, D)).unwrap();
    let k = ops::indexing::slice(k, (0, 0, 0, 0), (1, HKV, keys, D)).unwrap();
    let v = ops::indexing::slice(v, (0, 0, 0, 0), (1, HKV, keys, D)).unwrap();
    if !mask {
        return fast::scaled_dot_product_attention(&q, &k, &v, 0.0625, "causal", None, None)
            .unwrap();
    }
    let mut additive = Vec::with_capacity((rows * keys) as usize);
    for row in 0..rows {
        for key in 0..keys {
            additive.push(if key <= keys - rows + row {
                0.0
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    let additive = bf16(&additive, &[1, 1, rows, keys]);
    fast::scaled_dot_product_attention(&q, &k, &v, 0.0625, "", Some(&additive), None).unwrap()
}

#[test]
#[ignore = "requires an M5+ GPU (NAX)"]
fn split_k_rows_are_position_independent_and_accurate() {
    if let Ok(path) = std::env::var("IRONMLX_TEST_METALLIB") {
        mlx::metal::set_metallib_path(&path).unwrap();
    }
    let library =
        std::env::temp_dir().join(format!("ironmlx-d256-nax-{}.metallib", std::process::id()));
    std::fs::write(
        &library,
        mlx::metal::prefill_shaders::PREFILL_D256_NAX_METALLIB,
    )
    .unwrap();
    assert!(mlx::metal::set_prefill_d256_nax_library(
        library.to_str().unwrap()
    ));
    assert!(mlx::metal::set_decode_d256_nax_min_keys(MIN_KEYS));
    mlx::set_default_device(Device::gpu(0));

    let cap = 40_200_i32;
    let mut seed = 7_u64;
    let kv_len = (HKV * cap * D) as usize;
    let k_host: Vec<f32> = (0..kv_len).map(|_| uniform(&mut seed, 2.0)).collect();
    let v_host: Vec<f32> = (0..kv_len).map(|_| uniform(&mut seed, 1.0)).collect();
    let q_host: Vec<f32> = (0..(HQ * 8 * D) as usize)
        .map(|_| uniform(&mut seed, 2.0))
        .collect();
    let k = bf16(&k_host, &[1, HKV, cap, D]);
    let v = bf16(&v_host, &[1, HKV, cap, D]);
    let q = bf16(&q_host, &[1, HQ, 8, D]);
    mlx::transforms::eval(&[&q, &k, &v]).unwrap();

    for base in [MIN_KEYS - 1, 31_030, 40_183] {
        // Row p of the 8 sees base + p + 1 keys.
        let single: Vec<Vec<f32>> = (0..8)
            .map(|p| f32s(&attend(&q, &k, &v, p, 1, base + p + 1, false)))
            .collect();
        for rows in 2..=8 {
            for first in 0..=8 - rows {
                let out = f32s(&attend(&q, &k, &v, first, rows, base + first + rows, false));
                for head in 0..HQ {
                    for row in 0..rows {
                        let got = &out[((head * rows + row) * D) as usize..][..D as usize];
                        let want =
                            &single[(first + row) as usize][(head * D) as usize..][..D as usize];
                        assert!(
                            got.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()),
                            "base {base} rows {rows} first {first} head {head} row {row} differs from its single-row result"
                        );
                    }
                }
            }
        }
        eprintln!("base {base}: rows 1..=8 at every position bit-identical");
    }

    // Accuracy against an f64 reference over the bf16-rounded inputs.
    let k_r = f32s(&k);
    let v_r = f32s(&v);
    let q_r = f32s(&q);
    for (base, rows) in [(MIN_KEYS - 1, 4), (31_030, 4), (40_183, 8)] {
        let keys = base + rows;
        let ours = f32s(&attend(&q, &k, &v, 0, rows, keys, false));
        let mlx = f32s(&attend(&q, &k, &v, 0, rows, keys, true));
        let (mut ours_max, mut mlx_max, mut ours_sum, mut mlx_sum) = (0.0_f64, 0.0_f64, 0.0, 0.0);
        for head in 0..HQ {
            let kv = head / (HQ / HKV);
            for row in 0..rows {
                let qrow = &q_r[((head * 8 + row) * D) as usize..][..D as usize];
                let visible = (keys - rows + row + 1) as usize;
                let mut scores = vec![0.0_f64; visible];
                for (key, score) in scores.iter_mut().enumerate() {
                    let krow = &k_r[((kv * cap) as usize + key) * D as usize..][..D as usize];
                    *score = qrow
                        .iter()
                        .zip(krow)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>()
                        * 0.0625;
                }
                let top = scores.iter().cloned().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let total: f64 = weights.iter().sum();
                for dim in 0..D as usize {
                    let mut expected = 0.0;
                    for (key, w) in weights.iter().enumerate() {
                        expected += w * v_r[((kv * cap) as usize + key) * D as usize + dim] as f64;
                    }
                    expected /= total;
                    let at = ((head * rows + row) * D) as usize + dim;
                    let (a, b) = (
                        (ours[at] as f64 - expected).abs(),
                        (mlx[at] as f64 - expected).abs(),
                    );
                    ours_max = ours_max.max(a);
                    mlx_max = mlx_max.max(b);
                    ours_sum += a;
                    mlx_sum += b;
                }
            }
        }
        let n = (HQ * rows * D) as f64;
        eprintln!(
            "keys {keys} rows {rows}: max abs err ours {ours_max:.3e} mlx {mlx_max:.3e}, mean ours {:.3e} mlx {:.3e}",
            ours_sum / n,
            mlx_sum / n
        );
        assert!(
            ours_sum <= mlx_sum * 1.05,
            "mean error worse than MLX's attention"
        );
        assert!(
            ours_max <= mlx_max * 1.5 + 1e-4,
            "max error worse than MLX's attention"
        );
    }
    let _ = std::fs::remove_file(&library);
}
