//! Pure-Rust port of HF `Qwen2VLImageProcessorFast`. See spec §4.1.
//!
//! Pipeline: decode → smart_resize → normalize → patchify.

use anyhow::{anyhow, Result};
use mlx::ops::shape::{broadcast_to, expand_dims, transpose_axes};
use mlx::Array;

// 默认值跟 HF qwen2_vl image_processor 一致
const FACTOR: i32 = 32; // patch_size * spatial_merge_size = 16 * 2
const MIN_PIXELS: i32 = 56 * 56; // 3136
const MAX_PIXELS: i32 = 14 * 14 * 4 * 1280; // 1003520

/// Port of mlx-vlm `_smart_resize_image` — 保 aspect ratio + 满足 patch
/// 对齐 + 总像素在 [MIN_PIXELS, MAX_PIXELS]。
///
/// Returns `Err` if absolute aspect ratio > 200 (mlx-vlm parity — bound is
/// from `_smart_resize_image`).
pub fn smart_resize(height: i32, width: i32) -> Result<(i32, i32)> {
    let max_dim = height.max(width) as f64;
    let min_dim = height.min(width) as f64;
    if max_dim / min_dim > 200.0 {
        return Err(anyhow!(
            "absolute aspect ratio must be smaller than 200 (got {}x{})",
            height,
            width
        ));
    }
    let f = FACTOR as f64;
    let mut h_bar = ((height as f64 / f).round() * f) as i32;
    let mut w_bar = ((width as f64 / f).round() * f) as i32;
    if h_bar * w_bar > MAX_PIXELS {
        let beta = ((height as f64 * width as f64) / MAX_PIXELS as f64).sqrt();
        h_bar = (((height as f64 / beta) / f).floor() * f).max(f) as i32;
        w_bar = (((width as f64 / beta) / f).floor() * f).max(f) as i32;
    } else if h_bar * w_bar < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (height as f64 * width as f64)).sqrt();
        h_bar = (((height as f64 * beta) / f).ceil() * f) as i32;
        w_bar = (((width as f64 * beta) / f).ceil() * f) as i32;
    }
    Ok((h_bar, w_bar))
}

const IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// `(px/255 - mean) / std` for a single RGB pixel.
pub fn normalize_pixel(rgb: [u8; 3]) -> [f32; 3] {
    [
        (rgb[0] as f32 / 255.0 - IMAGE_MEAN[0]) / IMAGE_STD[0],
        (rgb[1] as f32 / 255.0 - IMAGE_MEAN[1]) / IMAGE_STD[1],
        (rgb[2] as f32 / 255.0 - IMAGE_MEAN[2]) / IMAGE_STD[2],
    ]
}

const PATCH: i32 = 16;
const TEMPORAL_PATCH: i32 = 2;
const MERGE_SIZE: i32 = 2;

/// Reshape `[3, H, W]` f32 raw pixels into Qwen3.5 vision patches.
///
/// Matches mlx-vlm's `_process_one` merge_size grouping exactly:
/// patches are ordered as `(grid_h/ms, grid_w/ms, ms, ms)` merge tiles,
/// NOT simple row-major `(grid_h, grid_w)`.
///
/// Output shape: `[grid_h * grid_w, TEMPORAL_PATCH (=2), 3, PATCH (=16), PATCH]`,
/// where the temporal axis duplicates the single image frame to match
/// `temporal_patch_size=2`.
///
/// Returns `(pixel_values_array, grid_h, grid_w)`. Errors when `h`/`w` are not
/// multiples of `PATCH * MERGE_SIZE` or when `raw.len()` doesn't match
/// `3 * h * w` — invariants the production caller (`preprocess` → after
/// `smart_resize`) always satisfies, but enforced as errors here so a future
/// caller that feeds a hand-built buffer can't silently corrupt outputs.
pub fn patchify(raw: &[f32], h: i32, w: i32) -> Result<(Array, i32, i32)> {
    if raw.len() != (3 * h * w) as usize {
        return Err(anyhow!(
            "patchify: raw.len() {} != 3*{}*{} = {}",
            raw.len(),
            h,
            w,
            3 * h * w
        ));
    }
    if h % PATCH != 0 || w % PATCH != 0 {
        return Err(anyhow!(
            "patchify: h={} w={} must be multiples of PATCH={}",
            h,
            w,
            PATCH
        ));
    }
    let grid_h = h / PATCH;
    let grid_w = w / PATCH;
    if grid_h % MERGE_SIZE != 0 || grid_w % MERGE_SIZE != 0 {
        return Err(anyhow!(
            "patchify: grid {}x{} must be divisible by MERGE_SIZE={}",
            grid_h,
            grid_w,
            MERGE_SIZE
        ));
    }
    let mgh = grid_h / MERGE_SIZE; // merge-tile rows
    let mgw = grid_w / MERGE_SIZE; // merge-tile cols
    let ms = MERGE_SIZE;
    // [3, H, W]
    //   → reshape [3, mgh, ms, PATCH, mgw, ms, PATCH]
    //   → permute (1, 4, 2, 5, 0, 3, 6) → [mgh, mgw, ms, ms, 3, PATCH, PATCH]
    //   → reshape → [grid_h * grid_w, 3, PATCH, PATCH]
    let arr: Array = (raw, &[3, h, w][..])
        .try_into()
        .map_err(|e| anyhow!("patchify: array construction: {e}"))?;
    let arr = arr.reshape(&[3, mgh, ms, PATCH, mgw, ms, PATCH][..])?;
    let arr = transpose_axes(&arr, &[1_i32, 4, 2, 5, 0, 3, 6][..])?;
    let arr = arr.reshape(&[grid_h * grid_w, 3, PATCH, PATCH][..])?;
    // expand temporal: [N, 3, P, P] → [N, 1, 3, P, P] → broadcast to [N, 2, 3, P, P]
    let arr = expand_dims(&arr, &[1_i32][..])?;
    let arr = broadcast_to(
        &arr,
        &[grid_h * grid_w, TEMPORAL_PATCH, 3, PATCH, PATCH][..],
    )?;
    Ok((arr, grid_h, grid_w))
}

/// Pipeline: decode → smart_resize → Lanczos resize → normalize → patchify.
/// Returns `(pixel_values, grid_h, grid_w)`.
pub fn preprocess(img_bytes: &[u8]) -> Result<(Array, i32, i32)> {
    // 1. decode
    let img = crate::core::image_input::load_from_memory_bounded(img_bytes)
        .map_err(|e| anyhow!("decode image: {e}"))?
        .to_rgb8();
    let (orig_w, orig_h) = (img.width() as i32, img.height() as i32);

    // 2. smart resize target size
    let (h2, w2) = smart_resize(orig_h, orig_w)?;

    // 3. Lanczos3 resize (HF default).
    // Skip resize when target equals source — PIL LANCZOS is a no-op in this
    // case (identity); resampling through a Lanczos kernel would introduce
    // rounding error without changing meaning.
    let n_pix = (3 * h2 * w2) as usize;
    let mut chw = vec![0.0_f32; n_pix];
    let plane = (h2 * w2) as usize;
    if orig_h == h2 && orig_w == w2 {
        // 4a. Already correct size — normalize directly.
        for (i, p) in img.pixels().enumerate() {
            let n = normalize_pixel([p.0[0], p.0[1], p.0[2]]);
            chw[i] = n[0];
            chw[plane + i] = n[1];
            chw[2 * plane + i] = n[2];
        }
    } else {
        let resized = image::imageops::resize(
            &img,
            w2 as u32,
            h2 as u32,
            image::imageops::FilterType::Lanczos3,
        );
        // 4b. normalize: [(H, W, 3) u8] → [(3, H, W) f32], (px/255 - 0.5)/0.5
        for (i, p) in resized.pixels().enumerate() {
            let n = normalize_pixel([p.0[0], p.0[1], p.0[2]]);
            chw[i] = n[0];
            chw[plane + i] = n[1];
            chw[2 * plane + i] = n[2];
        }
    }

    // 5. patchify
    let (pixel_values, gh, gw) = patchify(&chw, h2, w2)?;
    Ok((pixel_values, gh, gw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_centers_to_minus1_to_1() {
        // Pure black [0,0,0] → -1; pure white [255,255,255] → 1
        let black = [0u8, 0, 0];
        let white = [255u8, 255, 255];
        let nb = normalize_pixel(black);
        let nw = normalize_pixel(white);
        assert!((nb[0] - (-1.0)).abs() < 1e-5);
        assert!((nw[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mlxvlm_c_major_reshape_to_ironmlx_layout() {
        // The mlx-vlm processor packs pixel_values into [N, 1536] where the
        // 1536 inner dim is C-major (C, T, H, W) row-major flatten — see
        // /Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py:114-120
        // ("reshape(-1, C, T, H, W).moveaxis(1, 4)"). To consume that input
        // through ironmlx's VisionTower (which expects [N, T, C, H, W]), the
        // test driver must reshape [N, 1536] → [N, 3, 2, 16, 16] (C-major)
        // and transpose [0, 2, 1, 3, 4]. This test pins that contract.
        //
        // P6.2 regression marker: this is the byte-layout reshape that was
        // mis-coded in vision_dump.rs (P6.1 Task 4).
        let flat: Vec<f32> = (0..1536).map(|i| i as f32).collect();
        let pv: mlx::Array = (flat.as_slice(), &[1_i32, 1536][..]).try_into().unwrap();
        let pv_5d = pv.reshape(&[1_i32, 3, 2, 16, 16][..]).unwrap();
        let pv_out = mlx::ops::shape::transpose_axes(&pv_5d, &[0_i32, 2, 1, 3, 4][..]).unwrap();
        assert_eq!(pv_out.shape().as_slice(), &[1, 2, 3, 16, 16]);

        let v: Vec<f32> = pv_out.to_vec().unwrap();
        // pv_out[0, t, c, h, w] flat index in [2, 3, 16, 16] layout:
        let dst =
            |t: usize, c: usize, h: usize, w: usize| -> usize { ((t * 3 + c) * 16 + h) * 16 + w };
        // pv_in C-major formula: source byte at (c, t, h, w) is c*2*16*16 + t*16*16 + h*16 + w
        let src = |c: usize, t: usize, h: usize, w: usize| -> f32 {
            (c * 2 * 16 * 16 + t * 16 * 16 + h * 16 + w) as f32
        };
        // Spot-check several positions
        assert_eq!(v[dst(0, 0, 0, 0)], src(0, 0, 0, 0)); // byte 0
        assert_eq!(v[dst(0, 0, 0, 1)], src(0, 0, 0, 1)); // byte 1
        assert_eq!(v[dst(1, 0, 0, 0)], src(0, 1, 0, 0)); // (t=1) byte 256
        assert_eq!(v[dst(0, 1, 0, 0)], src(1, 0, 0, 0)); // (c=1) byte 512
        assert_eq!(v[dst(1, 2, 15, 15)], src(2, 1, 15, 15)); // last byte 1535
    }

    #[test]
    fn patchify_shape_correct() {
        // Synthetic [3, 32, 32] → grid 2×2 patches of 16×16, temporal=2
        // Output shape: [grid_h*grid_w=4, 2 (temporal), 3 (channels), 16, 16]
        let raw_pixels = vec![0.0_f32; 3 * 32 * 32];
        let (out, grid_h, grid_w) = patchify(&raw_pixels, 32, 32).expect("patchify");
        assert_eq!(out.shape().as_slice(), &[4, 2, 3, 16, 16]);
        assert_eq!((grid_h, grid_w), (2, 2));
    }

    #[test]
    fn patchify_rejects_mismatched_buffer() {
        // raw.len() != 3*h*w → Err
        let raw_pixels = vec![0.0_f32; 100];
        let r = patchify(&raw_pixels, 32, 32);
        assert!(r.is_err());
    }

    /// Golden values 从 mlx-vlm `_smart_resize_image` 直接取
    /// (FACTOR=32, MIN=56*56, MAX=14*14*4*1280 跟 spec 配置一致)。
    #[test]
    fn smart_resize_typical_image() {
        // 768×1024 输入 → 应该 round 到 32 倍数，没超 max
        let (h, w) = smart_resize(768, 1024).expect("smart_resize");
        assert_eq!(h, 768);
        assert_eq!(w, 1024);
    }

    #[test]
    fn smart_resize_too_large_downscaled() {
        // 4096×4096 = 16M px > max=14*14*4*1280 = 1003520; 应缩小
        let (h, w) = smart_resize(4096, 4096).expect("smart_resize");
        assert!(h * w <= 14 * 14 * 4 * 1280);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn smart_resize_too_small_upscaled() {
        // 32×32 = 1024 px < min=56*56=3136; 应放大
        let (h, w) = smart_resize(32, 32).expect("smart_resize");
        assert!(h * w >= 56 * 56);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn smart_resize_extreme_aspect_ratio_rejected() {
        // 长宽比 > 200 → Err (no panic — hostile request safety)
        let r = smart_resize(1, 250);
        assert!(r.is_err(), "expected Err on extreme aspect ratio");
    }

    #[test]
    fn preprocess_coco_sample_matches_hf() {
        use std::path::Path;
        let path = Path::new("tests/fixtures/qwen35_vl/coco_sample.jpg");
        let bytes = std::fs::read(path).expect("read coco sample");
        let (pixel_values, grid_h, grid_w) = preprocess(&bytes).expect("preprocess");
        assert!(grid_h * grid_w >= 4); // at least 4 patches

        // Check normalized values match HF
        let hf_path = Path::new("tests/fixtures/qwen35_vl/coco_sample_normalized.bin");
        let hf_bytes = std::fs::read(hf_path).expect("read hf normalized");
        let hf_floats: Vec<f32> = hf_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // pixel_values has shape [N=grid_h*grid_w, 2 (T), 3, 16, 16]
        // We're checking only first temporal frame (T=0; T=1 is identical via broadcast)
        use mlx::ops;
        let sliced = ops::slice(
            &pixel_values,
            &[0_i32, 0, 0, 0, 0][..],
            &[grid_h * grid_w, 1, 3, 16, 16][..],
        )
        .expect("slice");
        // sliced shape: [N, 1, 3, 16, 16]
        // Reorder to match HF normalized layout: [3, h2, w2] = [3, grid_h*16, grid_w*16]
        // Our pixel_values has been patched (each patch in row-major flatten);
        // To compare with HF's [3, h2, w2] image, we need to "depatchify" — invert the
        // patchify operation: [N=grid_h*grid_w, 1, 3, 16, 16]
        //   → squeeze T dim → [grid_h*grid_w, 3, 16, 16]
        //   → reshape → [grid_h, grid_w, 3, 16, 16]
        //   → permute (2, 0, 3, 1, 4) → [3, grid_h, 16, grid_w, 16]
        //   → reshape → [3, grid_h*16, grid_w*16]
        let arr = ops::squeeze(&sliced, &[1_i32][..]).expect("squeeze");
        // Patches are in merge-size order: [N=mgh*mgw*ms*ms, C, ps, ps]
        // where N is ordered as (mgh, mgw, ms, ms) NOT (grid_h, grid_w).
        // Depatchify: reverse the merge-size interleaving.
        let ms = MERGE_SIZE;
        let mgh = grid_h / ms;
        let mgw = grid_w / ms;
        let arr = arr
            .reshape(&[mgh, mgw, ms, ms, 3_i32, 16, 16][..])
            .expect("reshape to merge grid");
        // permute (4, 0, 2, 5, 1, 3, 6) -> [C, mgh, ms, ps, mgw, ms, ps]
        let arr = ops::shape::transpose_axes(&arr, &[4_i32, 0, 2, 5, 1, 3, 6]).expect("permute");
        let arr = arr
            .reshape(&[3_i32, grid_h * 16, grid_w * 16][..])
            .expect("reshape back to image");

        let our_floats: Vec<f32> = arr.to_vec().expect("vec");

        assert_eq!(
            our_floats.len(),
            hf_floats.len(),
            "length mismatch: ours={} hf={}",
            our_floats.len(),
            hf_floats.len()
        );
        let max_diff = our_floats
            .iter()
            .zip(&hf_floats)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "max_diff = {} (HF Lanczos vs Rust Lanczos may differ slightly)",
            max_diff
        );
    }

    #[test]
    fn smart_resize_matches_hf_100_random() {
        let golden = include_str!("../../../tests/fixtures/qwen35_vl/smart_resize_golden.txt");
        for line in golden.lines() {
            let parts: Vec<i32> = line.split(',').map(|s| s.parse().unwrap()).collect();
            let (h, w, py_h, py_w) = (parts[0], parts[1], parts[2], parts[3]);
            let (rs_h, rs_w) = smart_resize(h, w).expect("smart_resize");
            assert_eq!(
                (rs_h, rs_w),
                (py_h, py_w),
                "mismatch on {h}x{w}: rust=({rs_h},{rs_w}) py=({py_h},{py_w})"
            );
        }
    }
}
