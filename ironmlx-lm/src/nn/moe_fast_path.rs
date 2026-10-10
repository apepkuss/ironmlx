//! Kernel routes of the qualified Qwen3.6 MoE affine4 target.
//!
//! The target's forwards and logit projections arm this scope. Every route
//! keeps the bits of the op-by-op path it replaces, so outputs are those of the
//! ordinary MLX graph; only the number of dispatches changes. Under the scope:
//! - GatedDeltaNet computes its decay/beta gates and its q/k RMS norms in one
//!   kernel each, replaying MLX's own operators and reductions;
//! - the MoE router computes MLX's softmax, argpartition top-k and the
//!   renormalised scores in one kernel, and the expert gathers reuse their
//!   default left indices;
//! - verify attention applies MRoPE once to every position and groups the
//!   exact causal SDPA of neighbouring positions;
//! - product-stable affine4 projections of 3..=5 rows run MLX's kernel in
//!   one threadgroup per row group instead of padding to pairs
//!   (`product_stable_qmm`).
//!
//! Other models and targets never arm it.

use std::cell::Cell;

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

pub(crate) struct Scope;

impl Drop for Scope {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn scope() -> Scope {
    DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    Scope
}

pub(crate) fn is_armed() -> bool {
    DEPTH.with(|depth| depth.get() > 0)
}

#[cfg(test)]
mod tests {
    use mlx::{Array, Dtype, StreamOrDevice};
    use serial_test::serial;

    fn setup() {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
        }
    }

    fn normal(shape: &[i32], scale: f64, seed: u64) -> Array {
        let key = mlx::random::key(seed).unwrap();
        let a = mlx::random::normal()
            .shape(shape)
            .dtype(Dtype::Float32)
            .scale(scale)
            .key(&key)
            .sample()
            .unwrap();
        mlx::ops::cast::astype(&a, Dtype::Bfloat16).unwrap()
    }

    fn bits(a: &Array) -> Vec<u32> {
        mlx::ops::cast::astype(a, Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn mismatches(a: &Array, b: &Array) -> usize {
        assert_eq!(a.dtype(), b.dtype());
        bits(a)
            .iter()
            .zip(bits(b))
            .filter(|(x, y)| **x != *y)
            .count()
    }

    #[test]
    #[serial(mlx_metal)]
    fn fused_gdn_gates_keep_the_op_chain_bits() {
        setup();
        let t = StreamOrDevice::default();
        for seed in 0..4 {
            let a = normal(&[64, 8, 32], 4.0, seed);
            let b = normal(&[64, 8, 32], 4.0, seed + 100);
            let a_log = normal(&[32], 1.0, seed + 200);
            let dt_bias = normal(&[32], 2.0, seed + 300);
            let (g, beta) =
                mlx::quantization::gdn_gates_fused_on(&a, &b, &a_log, &dt_bias, t).unwrap();
            let x_sp = &a + &dt_bias;
            let twenty: Array = (&[20.0_f32][..], ()).try_into().unwrap();
            let safe = a.zeros_like().unwrap().logaddexp(&x_sp).unwrap();
            let sp = x_sp.greater(&twenty).unwrap().where_(&x_sp, &safe).unwrap();
            let neg = mlx::ops::binary::negative(
                &mlx::ops::cast::astype(&a_log, Dtype::Float32)
                    .unwrap()
                    .exp()
                    .unwrap(),
            )
            .unwrap();
            let g_ref = (&neg * &sp).exp().unwrap();
            let beta_ref = b.sigmoid_on(t).unwrap();
            assert_eq!(mismatches(&g, &g_ref), 0, "g seed {seed}");
            assert_eq!(mismatches(&beta, &beta_ref), 0, "beta seed {seed}");
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn fused_gdn_gates_keep_the_op_chain_bits_for_every_bf16_input() {
        setup();
        let t = StreamOrDevice::default();
        // Every BF16 bit pattern (NaNs included) as `a` and as `b`, 32 heads.
        let all: Vec<u32> = (0..=u16::MAX).map(|v| u32::from(v) << 16).collect();
        let floats: Vec<f32> = all.iter().map(|&v| f32::from_bits(v)).collect();
        let a: Array = (&floats[..], &[2048_i32, 32][..]).try_into().unwrap();
        let a = mlx::ops::cast::astype(&a, Dtype::Bfloat16).unwrap();
        let b = a.clone();
        for seed in 0..3 {
            let a_log = normal(&[32], 1.0, seed + 500);
            let dt_bias = normal(&[32], 2.0, seed + 600);
            let (g, beta) =
                mlx::quantization::gdn_gates_fused_on(&a, &b, &a_log, &dt_bias, t).unwrap();
            let x_sp = &a + &dt_bias;
            let twenty: Array = (&[20.0_f32][..], ()).try_into().unwrap();
            let safe = a.zeros_like().unwrap().logaddexp(&x_sp).unwrap();
            let sp = x_sp.greater(&twenty).unwrap().where_(&x_sp, &safe).unwrap();
            let neg = mlx::ops::binary::negative(
                &mlx::ops::cast::astype(&a_log, Dtype::Float32)
                    .unwrap()
                    .exp()
                    .unwrap(),
            )
            .unwrap();
            let g_ref = (&neg * &sp).exp().unwrap();
            let beta_ref = b.sigmoid_on(t).unwrap();
            assert_eq!(mismatches(&g, &g_ref), 0, "g seed {seed}");
            assert_eq!(mismatches(&beta, &beta_ref), 0, "beta seed {seed}");
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn fused_qk_norm_keeps_the_op_chain_bits() {
        setup();
        let t = StreamOrDevice::default();
        let (heads, d) = (16, 128);
        let key_dim = heads * d;
        for seed in 0..4 {
            let src = normal(&[3, 17, 2 * key_dim + 4096], 1.5, seed);
            let inv = 1.0_f32 / (d as f32).sqrt();
            let (q, k) = mlx::quantization::qk_norm_fused_on(
                &src,
                heads,
                d,
                0,
                key_dim,
                inv * inv,
                inv,
                1e-6,
                t,
            )
            .unwrap();
            let parts = mlx::ops::shape::split_at_on(&src, &[key_dim, 2 * key_dim], -1, t).unwrap();
            let q_in = parts[0].reshape_on((3, 17, heads, d), t).unwrap();
            let k_in = parts[1].reshape_on((3, 17, heads, d), t).unwrap();
            let q_ref = &mlx::fast::rms_norm_on(&q_in, None, 1e-6, t).unwrap() * (inv * inv);
            let k_ref = &mlx::fast::rms_norm_on(&k_in, None, 1e-6, t).unwrap() * inv;
            let q = q.reshape_on((3, 17, heads, d), t).unwrap();
            let k = k.reshape_on((3, 17, heads, d), t).unwrap();
            assert_eq!(mismatches(&q, &q_ref), 0, "q seed {seed}");
            assert_eq!(mismatches(&k, &k_ref), 0, "k seed {seed}");
        }
    }

    fn zeros(shape: &[i32], dtype: Dtype) -> Array {
        Array::zeros(shape, dtype).unwrap()
    }

    /// Malformed operands of the native entry points return `Err` through the
    /// public API (the process survives); none of these graphs is evaluated.
    #[test]
    #[serial(mlx_metal)]
    fn native_entry_points_reject_malformed_operands() {
        use mlx::quantization as q;
        setup();
        let t = StreamOrDevice::default();
        let bf = Dtype::Bfloat16;
        let scalar: Array = (&[0.0_f32][..], ()).try_into().unwrap();
        let scalar = mlx::ops::cast::astype(&scalar, bf).unwrap();
        let x = zeros(&[3, 512], bf);
        let w = zeros(&[32, 64], Dtype::Uint32);
        let s = zeros(&[32, 8], bf);
        let tiny = zeros(&[1], bf);

        // GatedDeltaNet gates.
        assert!(q::gdn_gates_fused_on(&scalar, &scalar, &scalar, &scalar, t).is_err());
        let a = zeros(&[2, 4], bf);
        let h3 = zeros(&[3], bf);
        assert!(q::gdn_gates_fused_on(&a, &a, &h3, &h3, t).is_err());
        let empty = zeros(&[0, 4], bf);
        let h4 = zeros(&[4], bf);
        assert!(q::gdn_gates_fused_on(&empty, &empty, &h4, &h4, t).is_err());

        // Product-stable wide qmv.
        assert!(q::qmv_fast_wide_on(&x, &w, &tiny, &tiny, 5, t).is_err());
        assert!(q::qmv_fast_wide_on(&scalar, &w, &s, &s, 5, t).is_err());
        let x0 = zeros(&[3, 0], bf);
        let w0 = zeros(&[32, 0], Dtype::Uint32);
        assert!(q::qmv_fast_wide_on(&x0, &w0, &s, &s, 5, t).is_err());
        assert!(q::qmv_fast_wide_on(&zeros(&[1, 512], bf), &w, &s, &s, 5, t).is_err());
        assert!(q::qmv_fast_wide_on(&x, &w, &s, &s, 6, t).is_err());

        // q/k norm.
        let src = zeros(&[1, 512], bf);
        assert!(q::qk_norm_fused_on(&src, 1, 128, -1, 128, 1.0, 1.0, 1e-6, t).is_err());
        assert!(q::qk_norm_fused_on(&scalar, 1, 128, 0, 128, 1.0, 1.0, 1e-6, t).is_err());
        assert!(q::qk_norm_fused_on(&src, 0, 128, 0, 128, 1.0, 1.0, 1e-6, t).is_err());
        assert!(q::qk_norm_fused_on(&src, 1, 128, 0, 400, 1.0, 1.0, 1e-6, t).is_err());
        assert!(q::qk_norm_fused_on(&src, i32::MAX, 128, 0, 0, 1.0, 1.0, 1e-6, t).is_err());
        assert!(q::qk_norm_fused_on(&zeros(&[1, 0], bf), 1, 128, 0, 0, 1.0, 1.0, 1e-6, t).is_err());

        // Row-stable affine4.
        let x1 = zeros(&[1, 512], bf);
        assert!(q::row_stable_affine4_matmul_on(&x1, &w, &s, &s, 0, 0, 1, 1, t).is_err());
        // One-row tile of 16 * (32 / 8) * 1 = 64 outputs cannot cover N = 32.
        assert!(q::row_stable_affine4_matmul_on(&x1, &w, &s, &s, 0, 8, 1, 16, t).is_err());
        assert!(q::row_stable_affine4_matmul_on(&x1, &w, &s, &s, 0, 8, 0, 1, t).is_err());
        assert!(q::row_stable_affine4_matmul_on(&x1, &w, &s, &s, 0, 8, 1, 0, t).is_err());
        assert!(q::row_stable_affine4_matmul_on(&x1, &w, &s, &s, 2, 8, 2, 1, t).is_err());
        assert!(q::row_stable_affine4_matmul_on(&x1, &w, &tiny, &tiny, 0, 8, 1, 1, t).is_err());
        assert!(q::row_stable_affine4_matmul_on(&x0, &w0, &s, &s, 2, 8, 1, 1, t).is_err());
        assert!(q::row_stable_affine4_matmul_on(&tiny, &w, &s, &s, 0, 8, 1, 1, t).is_err());
        assert!(q::row_stable_affine4_matmul_on(&x1, &w, &s, &s, 7, 8, 1, 1, t).is_err());

        // MoE router.
        assert!(q::router_topk_fused_on(&zeros(&[2, 0], bf), 1, true, t).is_err());
        assert!(q::router_topk_fused_on(&scalar, 1, true, t).is_err());
        assert!(q::router_topk_fused_on(&zeros(&[0, 256], bf), 8, true, t).is_err());

        // Selector walk.
        let u32 = Dtype::Uint32;
        let cand = zeros(&[1, 2, 4], u32);
        let unary = zeros(&[1, 2, 4], bf);
        let hidden = zeros(&[1, 2, 8], bf);
        let anchor = zeros(&[1], u32);
        let book = zeros(&[16, 8], bf);
        assert!(
            q::dflash2_selector_walk_on(&scalar, &unary, &hidden, &anchor, &book, &book, t)
                .is_err()
        );
        let no_rows = zeros(&[0, 8], bf);
        assert!(q::dflash2_selector_walk_on(
            &cand, &unary, &hidden, &anchor, &no_rows, &no_rows, t
        )
        .is_err());
        assert!(q::dflash2_selector_walk_on(
            &zeros(&[1, 2, 33], u32),
            &zeros(&[1, 2, 33], bf),
            &hidden,
            &anchor,
            &book,
            &book,
            t
        )
        .is_err());

        // The legal operands of the same shapes still build.
        assert!(q::qmv_fast_wide_on(&x, &w, &s, &s, 5, t).is_ok());
        assert!(q::row_stable_affine4_matmul_on(&x1, &w, &s, &s, 0, 8, 1, 2, t).is_ok());
        assert!(q::qk_norm_fused_on(&src, 1, 128, 0, 128, 1.0, 1.0, 1e-6, t).is_ok());
        assert!(
            q::dflash2_selector_walk_on(&cand, &unary, &hidden, &anchor, &book, &book, t).is_ok()
        );
    }

    /// Token ids are data the shape checks cannot see: ids outside the
    /// codebooks read nothing, and in-range rows keep their path.
    #[test]
    #[serial(mlx_metal)]
    fn selector_walk_ignores_token_ids_outside_the_codebooks() {
        setup();
        let t = StreamOrDevice::default();
        let book = normal(&[16, 64], 1.0, 7);
        let hidden = normal(&[2, 3, 64], 1.0, 8);
        let unary = normal(&[2, 3, 4], 1.0, 9);
        let ids: Vec<u32> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        ];
        let cand: Array = (&ids[..], &[2_i32, 3, 4][..]).try_into().unwrap();
        let anchors: Array = (&[0_u32, 0][..], &[2_i32][..]).try_into().unwrap();
        let reference = mlx::quantization::dflash2_selector_walk_on(
            &cand, &unary, &hidden, &anchors, &book, &book, t,
        )
        .unwrap()
        .to_vec::<u32>()
        .unwrap();
        let mut bad = ids.clone();
        bad[12..].iter_mut().for_each(|id| *id += 1 << 30);
        let cand_bad: Array = (&bad[..], &[2_i32, 3, 4][..]).try_into().unwrap();
        let anchors_bad: Array = (&[0_u32, u32::MAX][..], &[2_i32][..]).try_into().unwrap();
        let out = mlx::quantization::dflash2_selector_walk_on(
            &cand_bad,
            &unary,
            &hidden,
            &anchors_bad,
            &book,
            &book,
            t,
        )
        .unwrap()
        .to_vec::<u32>()
        .unwrap();
        assert_eq!(out[..3], reference[..3]);
        assert!(out[3..].iter().all(|&id| id >= 1 << 30));
    }

    #[test]
    #[serial(mlx_metal)]
    fn fused_router_keeps_the_op_chain_bits() {
        setup();
        let t = StreamOrDevice::default();
        let (rows, experts, k) = (513, 256, 8);
        for seed in 0..4 {
            // Coarse logits make equal probabilities (ties) common.
            let fine = normal(&[rows, experts], 3.0, seed);
            let logits = if seed % 2 == 0 {
                fine
            } else {
                let q = mlx::ops::unary::round(&(&fine * 2.0_f32), 0).unwrap();
                mlx::ops::cast::astype(&(&q * 0.5_f32), Dtype::Bfloat16).unwrap()
            };
            for norm in [true, false] {
                let (scores, inds) =
                    mlx::quantization::router_topk_fused_on(&logits, k, norm, t).unwrap();
                let probs = mlx::ops::softmax_on(&logits, -1_i32, true, t).unwrap();
                let part = mlx::ops::sort::argpartition_on(&probs, -k, -1, t).unwrap();
                let inds_ref = mlx::ops::slice_strided_on(
                    &part,
                    [0_i32, experts - k],
                    [rows, experts],
                    [1_i32, 1_i32],
                    t,
                )
                .unwrap();
                let raw = mlx::ops::indexing::take_along_axis_on(&probs, &inds_ref, -1, t).unwrap();
                let scores_ref = if norm {
                    let sum = mlx::ops::sum_on(&raw, -1_i32, true, t).unwrap();
                    &raw / &sum
                } else {
                    raw
                };
                let inds_ref = mlx::ops::cast::astype(&inds_ref, Dtype::Uint32).unwrap();
                assert_eq!(
                    inds.to_vec::<u32>().unwrap(),
                    inds_ref.to_vec::<u32>().unwrap(),
                    "indices seed {seed} norm {norm}"
                );
                assert_eq!(
                    mismatches(&scores, &scores_ref),
                    0,
                    "scores seed {seed} norm {norm}"
                );
            }
        }
    }
}
