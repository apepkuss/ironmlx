//! Pure-Rust port of HF `Qwen2VLImageProcessorFast`. See spec §4.1.
//!
//! Pipeline: decode → smart_resize → normalize → patchify.

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

#[cfg(test)]
mod tests {
    use super::*;

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
