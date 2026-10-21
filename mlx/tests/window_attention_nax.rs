//! Opt-in test of the split-K BF16 NAX sliding-window attention of short
//! head-dim-128 query blocks (DFlash2 drafter shape): its error against an
//! f64 reference matches MLX's own masked attention (up to bf16 output
//! rounding), for every window start, row count and key count it serves.
use mlx::{fast, Array, Device, Dtype};

const HQ: i32 = 32;
const HKV: i32 = 8;
const D: i32 = 128;

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

#[test]
#[ignore = "requires an M5+ GPU (NAX)"]
fn window_attention_matches_masked_attention() {
    if let Ok(path) = std::env::var("IRONMLX_TEST_METALLIB") {
        mlx::metal::set_metallib_path(&path).unwrap();
    }
    let library = std::env::temp_dir().join(format!(
        "ironmlx-window-nax-{}.metallib",
        std::process::id()
    ));
    std::fs::write(
        &library,
        mlx::metal::prefill_shaders::PREFILL_D256_NAX_METALLIB,
    )
    .unwrap();
    assert!(mlx::metal::set_prefill_d256_nax_library(
        library.to_str().unwrap()
    ));
    mlx::set_default_device(Device::gpu(0));
    let scale = 1.0 / (D as f32).sqrt();
    let mut seed = 11_u64;
    let mut worst_ratio = 0.0_f64;
    for (keys, rows, window_start) in [
        (2055, 8, 0),
        (2055, 8, -40),
        (1500, 8, -547),
        (2051, 4, 0),
        (2049, 2, 0),
        (2063, 16, 0),
        (300, 8, -1747),
        (9, 8, -2038),
        (2100, 8, 45),
    ] {
        let k_host: Vec<f32> = (0..(HKV * keys * D) as usize)
            .map(|_| uniform(&mut seed, 2.0))
            .collect();
        let v_host: Vec<f32> = (0..(HKV * keys * D) as usize)
            .map(|_| uniform(&mut seed, 1.0))
            .collect();
        let q_host: Vec<f32> = (0..(HQ * rows * D) as usize)
            .map(|_| uniform(&mut seed, 2.0))
            .collect();
        let q = bf16(&q_host, &[1, HQ, rows, D]);
        let k = bf16(&k_host, &[1, HKV, keys, D]);
        let v = bf16(&v_host, &[1, HKV, keys, D]);
        let window = fast::window_attention_on(&q, &k, &v, scale, window_start, ())
            .unwrap()
            .expect("window attention applies");
        let mut additive = Vec::with_capacity((rows * keys) as usize);
        for row in 0..rows {
            for key in 0..keys {
                additive.push(if key >= window_start + row {
                    0.0
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
        let additive = bf16(&additive, &[1, 1, rows, keys]);
        let mlx_out =
            fast::scaled_dot_product_attention(&q, &k, &v, scale, "", Some(&additive), None)
                .unwrap();
        let (got, base) = (f32s(&window), f32s(&mlx_out));
        // f64 reference from the bf16-rounded inputs.
        let qr = f32s(&q);
        let kr = f32s(&k);
        let vr = f32s(&v);
        let gqa = HQ / HKV;
        let (mut err_got, mut err_base) = (0.0_f64, 0.0_f64);
        for h in 0..HQ {
            let kvh = h / gqa;
            for r in 0..rows {
                let qo = ((h * rows + r) * D) as usize;
                let lo = (window_start + r).max(0);
                let scores: Vec<f64> = (lo..keys)
                    .map(|key| {
                        let ko = ((kvh * keys + key) * D) as usize;
                        (0..D as usize)
                            .map(|d| qr[qo + d] as f64 * kr[ko + d] as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f64 = w.iter().sum();
                for d in 0..D as usize {
                    let reference: f64 = (lo..keys)
                        .zip(&w)
                        .map(|(key, w)| w * vr[((kvh * keys + key) * D) as usize + d] as f64)
                        .sum::<f64>()
                        / sum;
                    err_got = err_got.max((got[qo + d] as f64 - reference).abs());
                    err_base = err_base.max((base[qo + d] as f64 - reference).abs());
                }
            }
        }
        eprintln!(
            "keys {keys} rows {rows} window_start {window_start}: max err window {err_got:.3e} mlx {err_base:.3e}"
        );
        // Both round the output to bf16: allow half a unit-scale bf16 ulp.
        assert!(
            err_got <= err_base * 1.1 + 2f64.powi(-9),
            "window attention error {err_got} exceeds MLX {err_base}"
        );
        worst_ratio = worst_ratio.max(err_got / err_base.max(1e-12));
    }
    eprintln!("worst error ratio window/mlx {worst_ratio:.3}");
}
