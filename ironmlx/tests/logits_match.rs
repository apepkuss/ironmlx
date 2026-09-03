//! P6.6 multi-image e2e logits-match (Gate 3).
//!
//! Drives Qwen35Model::forward_vl on a 2-image input + the mlx-vlm-generated
//! input_ids, compares last-position logits + greedy first token against
//! the mlx-vlm reference fixture written by run_p6_6_dump.py.
//!
//! Run with:
//!   MLX_DIR=$HOME/.local/mlx \
//!   QWEN35_MODEL=/path/to/model \
//!   cargo test -p ironmlx --test logits_match --release -- --ignored

use std::path::Path;

use mlx::Dtype;

use ironmlx::core::generate::{build_position_ids_vl, IMAGE_TOKEN_ID};
use ironmlx::core::Loader;
use ironmlx::models::qwen3_5::Qwen35Config;
use ironmlx::models::qwen3_5::Qwen35Model;

const FIXTURE_DIR: &str = "tests/fixtures/qwen35_vl/multi_image";

fn load_npy_int32(path: &str) -> mlx::Array {
    mlx::io::load_npy(path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"))
}

fn load_npy_f32(path: &str) -> mlx::Array {
    let arr = mlx::io::load_npy(path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"));
    mlx::ops::cast::astype(&arr, Dtype::Float32).expect("cast f32")
}

fn max_abs_diff(a: &mlx::Array, b: &mlx::Array) -> f32 {
    let a32 = mlx::ops::cast::astype(a, Dtype::Float32).expect("a32");
    let b32 = mlx::ops::cast::astype(b, Dtype::Float32).expect("b32");
    let av: Vec<f32> = a32.to_vec::<f32>().expect("av");
    let bv: Vec<f32> = b32.to_vec::<f32>().expect("bv");
    av.iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn greedy_argmax(arr: &mlx::Array) -> i32 {
    let f32_arr = mlx::ops::cast::astype(arr, Dtype::Float32).expect("astype f32");
    let v: Vec<f32> = f32_arr.to_vec::<f32>().expect("to_vec");
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .expect("empty array")
}

#[test]
#[ignore = "requires QWEN35_MODEL env var + fixture files from run_p6_6_dump.py"]
fn p6_6_logits_match() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env required");
    let loader = Loader::open_multimodal(Path::new(&model_dir)).expect("loader");
    let cfg = Qwen35Config::from_loader(&loader).expect("config");
    let model = Qwen35Model::from_loader(&loader).expect("model");

    // Load mlx-vlm fixture artifacts.
    let input_ids = load_npy_int32(&format!("{FIXTURE_DIR}/expected_input_ids.npy"));
    let grid_arr = load_npy_int32(&format!("{FIXTURE_DIR}/expected_image_grid_thw.npy"));
    let (mut loaded, _meta) =
        mlx::io::load_safetensors(&format!("{FIXTURE_DIR}/expected_pixel_values.safetensors"))
            .expect("load pv safetensors");
    let pv_flat = loaded.remove("tensor").expect("tensor key in pv");
    let expected_logits = load_npy_f32(&format!("{FIXTURE_DIR}/expected_last_logits.npy"));
    let expected_first_token: i32 =
        std::fs::read_to_string(format!("{FIXTURE_DIR}/expected_first_token.txt"))
            .expect("read expected_first_token.txt")
            .trim()
            .parse()
            .expect("parse first_token");

    eprintln!(
        "[p6_6_logits_match] input_ids shape: {:?}",
        input_ids.shape().as_slice()
    );
    eprintln!(
        "[p6_6_logits_match] pv_flat shape: {:?}",
        pv_flat.shape().as_slice()
    );
    eprintln!(
        "[p6_6_logits_match] grid_arr shape: {:?}",
        grid_arr.shape().as_slice()
    );
    eprintln!(
        "[p6_6_logits_match] expected_logits shape: {:?}",
        expected_logits.shape().as_slice()
    );
    eprintln!("[p6_6_logits_match] expected_first_token: {expected_first_token}");

    // Convert mlx-vlm [N_total, 1536] C-major to ironmlx [N_total, T, C, H, W].
    // reshape([N, 1536]) -> [N, 3, 2, 16, 16] (N, C, T, H, W)
    // transpose([0, 2, 1, 3, 4]) -> [N, 2, 3, 16, 16] (N, T, C, H, W)
    let n_total = pv_flat.shape().as_slice()[0];
    let pv_5d = pv_flat
        .reshape(&[n_total, 3, 2, 16, 16][..])
        .expect("reshape pv to [N, C, T, H, W]");
    let pv = mlx::ops::shape::transpose_axes(&pv_5d, &[0_i32, 2, 1, 3, 4][..])
        .expect("transpose pv to [N, T, C, H, W]");
    let pv = mlx::ops::cast::astype(&pv, Dtype::Bfloat16).expect("cast pv bf16");

    eprintln!(
        "[p6_6_logits_match] pixel_values after reshape+transpose: {:?}",
        pv.shape().as_slice()
    );

    // grid_thw: [2, 3] -> Vec<(t, h, w)>
    let grids_flat: Vec<i32> = mlx::ops::cast::astype(&grid_arr, Dtype::Int32)
        .expect("grid cast i32")
        .to_vec::<i32>()
        .expect("grid to_vec");
    let grids: Vec<(i32, i32, i32)> = grids_flat
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect();
    assert!(
        grids.len() >= 1,
        "expected at least 1 image in grid_thw, got {}",
        grids.len()
    );

    eprintln!("[p6_6_logits_match] grids: {grids:?}");

    // Build MRoPE VL position ids.
    let spatial_merge_size = cfg
        .vision_config
        .as_ref()
        .map(|vc| vc.spatial_merge_size)
        .unwrap_or(2);
    let ids_i32: Vec<i32> = mlx::ops::cast::astype(&input_ids, Dtype::Int32)
        .expect("input_ids cast")
        .to_vec::<i32>()
        .expect("input_ids to_vec");
    let pos_ids = build_position_ids_vl(&ids_i32, &grids, IMAGE_TOKEN_ID, spatial_merge_size)
        .expect("build_position_ids_vl");

    eprintln!(
        "[p6_6_logits_match] pos_ids shape: {:?}",
        pos_ids.shape().as_slice()
    );

    // Forward pass.
    let logits = model
        .forward_vl(
            &input_ids,
            &pos_ids,
            None, // per_row_lens
            None, // decode_mask
            None, // cache
            Some(std::slice::from_ref(&pv)),
            Some(&grids),
            IMAGE_TOKEN_ID,
            (),
        )
        .expect("forward_vl");

    // Last-position diff: logits shape [1, 1, vocab] or [1, vocab] -> flatten to [vocab].
    let logits_shape = logits.shape();
    let vocab = *logits_shape.as_slice().last().expect("logits non-empty");
    let our_flat = logits
        .reshape(&[vocab][..])
        .expect("reshape logits to [vocab]");
    let expected_flat = expected_logits
        .reshape(&[vocab][..])
        .expect("reshape expected_logits to [vocab]");

    let max_diff = max_abs_diff(&our_flat, &expected_flat);
    eprintln!("[p6_6_logits_match] max_abs_diff = {max_diff:.4}");

    // Signed-diff distribution diagnostic (helps distinguish systematic offset
    // vs structural outliers vs uniform noise).
    {
        let af = mlx::ops::cast::astype(&our_flat, Dtype::Float32).expect("af");
        let bf = mlx::ops::cast::astype(&expected_flat, Dtype::Float32).expect("bf");
        let av: Vec<f32> = af.to_vec::<f32>().expect("av");
        let bv: Vec<f32> = bf.to_vec::<f32>().expect("bv");
        let signed: Vec<f32> = av.iter().zip(&bv).map(|(a, b)| a - b).collect();
        let mean: f64 = signed.iter().map(|&x| x as f64).sum::<f64>() / signed.len() as f64;
        let mut sorted = signed.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let above_05 = signed.iter().filter(|&&x| x.abs() > 0.5).count();
        let above_10 = signed.iter().filter(|&&x| x.abs() > 1.0).count();
        eprintln!(
            "[p6_6_logits_match] signed diff: mean={mean:.6} median={median:.6} \
             |diff|>0.5 count={above_05}/{} |diff|>1.0 count={above_10}",
            signed.len()
        );
        // Residual after subtracting mean — indicates whether the elevation
        // is a uniform offset or true scatter.
        let residual_max: f32 = signed
            .iter()
            .map(|&x| (x - mean as f32).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "[p6_6_logits_match] residual max_abs_diff (after mean subtraction): {residual_max:.4}"
        );
        // Top-5 absolute outliers — token id + ours / expected / diff.
        let mut idxd: Vec<(usize, f32)> = signed
            .iter()
            .enumerate()
            .map(|(i, &d)| (i, d.abs()))
            .collect();
        idxd.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("[p6_6_logits_match] top-5 outliers:");
        for &(idx, _) in idxd.iter().take(5) {
            eprintln!(
                "  logit[{idx}]: ours={:.4} expected={:.4} diff={:.4}",
                av[idx],
                bv[idx],
                av[idx] - bv[idx]
            );
        }
        // Histogram in 0.25-wide bins covering [-2, 2].
        eprintln!("[p6_6_logits_match] Signed diff histogram (ours - expected):");
        let mut lo = -2.0_f32;
        while lo < 2.0 {
            let hi = lo + 0.25;
            let cnt = signed.iter().filter(|&&x| x >= lo && x < hi).count();
            if cnt > 0 {
                eprintln!("  [{lo:.2},{hi:.2}): {cnt}");
            }
            lo = hi;
        }
    }

    // Gate 3B: greedy first token must match (hard gate).
    let our_first = greedy_argmax(&our_flat);
    eprintln!("[p6_6_logits_match] our_first_token={our_first} expected={expected_first_token}");

    assert_eq!(
        our_first, expected_first_token,
        "Gate 3B (greedy first token) failed: ours={our_first}, expected={expected_first_token}"
    );

    // Gate 3A threshold will be set after Task 7 baseline run.
    // For now, only Gate 3B is asserted.
    eprintln!("[p6_6_logits_match] PASS — max_diff={max_diff:.4}, first_token={our_first}");
}
