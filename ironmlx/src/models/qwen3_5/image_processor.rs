//! Pure-Rust port of HF `Qwen2VLImageProcessorFast`. See spec §4.1.
//!
//! Pipeline: decode → smart_resize → normalize → patchify.

use mlx::ops::shape::{broadcast_to, expand_dims, transpose_axes};
use mlx::Array;

// 默认值跟 HF qwen2_vl image_processor 一致
const FACTOR: i32 = 32; // patch_size * spatial_merge_size = 16 * 2
const MIN_PIXELS: i32 = 56 * 56; // 3136
const MAX_PIXELS: i32 = 14 * 14 * 4 * 1280; // 1003520

/// Port of mlx-vlm `_smart_resize_image` — 保 aspect ratio + 满足 patch
/// 对齐 + 总像素在 [MIN_PIXELS, MAX_PIXELS]。
///
/// Panics if absolute aspect ratio > 200.
pub fn smart_resize(height: i32, width: i32) -> (i32, i32) {
    let max_dim = height.max(width) as f64;
    let min_dim = height.min(width) as f64;
    if max_dim / min_dim > 200.0 {
        panic!("absolute aspect ratio must be smaller than 200");
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
    (h_bar, w_bar)
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

/// Reshape `[3, H, W]` f32 raw pixels into Qwen3.5 vision patches.
///
/// Output shape: `[grid_h * grid_w, TEMPORAL_PATCH (=2), 3, PATCH (=16), PATCH]`,
/// where the temporal axis duplicates the single image frame to match
/// `temporal_patch_size=2`.
///
/// Returns `(pixel_values_array, grid_h, grid_w)`.
pub fn patchify(raw: &[f32], h: i32, w: i32) -> (Array, i32, i32) {
    assert_eq!(raw.len(), (3 * h * w) as usize);
    assert_eq!(h % PATCH, 0);
    assert_eq!(w % PATCH, 0);
    let grid_h = h / PATCH;
    let grid_w = w / PATCH;
    // [3, H, W] → [3, grid_h, PATCH, grid_w, PATCH]
    //           → permute (1, 3, 0, 2, 4) → [grid_h, grid_w, 3, PATCH, PATCH]
    //           → reshape → [grid_h * grid_w, 3, PATCH, PATCH]
    let arr: Array = (raw, &[3, h, w][..]).try_into().expect("array");
    let arr = arr
        .reshape(&[3, grid_h, PATCH, grid_w, PATCH][..])
        .expect("reshape");
    let arr = transpose_axes(&arr, &[1_i32, 3, 0, 2, 4][..]).expect("transpose");
    let arr = arr
        .reshape(&[grid_h * grid_w, 3, PATCH, PATCH][..])
        .expect("reshape2");
    // expand temporal: [N, 3, P, P] → [N, 1, 3, P, P] → broadcast to [N, 2, 3, P, P]
    let arr = expand_dims(&arr, &[1_i32][..]).expect("expand");
    let arr = broadcast_to(
        &arr,
        &[grid_h * grid_w, TEMPORAL_PATCH, 3, PATCH, PATCH][..],
    )
    .expect("broadcast");
    (arr, grid_h, grid_w)
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
    fn patchify_shape_correct() {
        // Synthetic [3, 32, 32] → grid 2×2 patches of 16×16, temporal=2
        // Output shape: [grid_h*grid_w=4, 2 (temporal), 3 (channels), 16, 16]
        let raw_pixels = vec![0.0_f32; 3 * 32 * 32];
        let (out, grid_h, grid_w) = patchify(&raw_pixels, 32, 32);
        assert_eq!(out.shape().as_slice(), &[4, 2, 3, 16, 16]);
        assert_eq!((grid_h, grid_w), (2, 2));
    }

    /// Golden values 从 mlx-vlm `_smart_resize_image` 直接取
    /// (FACTOR=32, MIN=56*56, MAX=14*14*4*1280 跟 spec 配置一致)。
    #[test]
    fn smart_resize_typical_image() {
        // 768×1024 输入 → 应该 round 到 32 倍数，没超 max
        let (h, w) = smart_resize(768, 1024);
        assert_eq!(h, 768);
        assert_eq!(w, 1024);
    }

    #[test]
    fn smart_resize_too_large_downscaled() {
        // 4096×4096 = 16M px > max=14*14*4*1280 = 1003520; 应缩小
        let (h, w) = smart_resize(4096, 4096);
        assert!(h * w <= 14 * 14 * 4 * 1280);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn smart_resize_too_small_upscaled() {
        // 32×32 = 1024 px < min=56*56=3136; 应放大
        let (h, w) = smart_resize(32, 32);
        assert!(h * w >= 56 * 56);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn smart_resize_extreme_aspect_ratio_rejected() {
        // 长宽比 > 200 应 panic 或 error
        let r = std::panic::catch_unwind(|| smart_resize(1, 250));
        assert!(r.is_err(), "expected panic on extreme aspect ratio");
    }

    #[test]
    fn smart_resize_matches_hf_100_random() {
        let golden = include_str!("../../../tests/fixtures/p6_qwen35_vl/smart_resize_golden.txt");
        for line in golden.lines() {
            let parts: Vec<i32> = line.split(',').map(|s| s.parse().unwrap()).collect();
            let (h, w, py_h, py_w) = (parts[0], parts[1], parts[2], parts[3]);
            let (rs_h, rs_w) = smart_resize(h, w);
            assert_eq!(
                (rs_h, rs_w),
                (py_h, py_w),
                "mismatch on {h}x{w}: rust=({rs_h},{rs_w}) py=({py_h},{py_w})"
            );
        }
    }
}
