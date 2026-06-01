//! Pure-Rust port of mlx-vlm `MiniCPMVImageProcessor` — single-image
//! (no-slice, `slice_mode=False`) path. See P2a Task 1.
//!
//! Pipeline: decode → `_find_best_resize` → BICUBIC resize → normalize →
//! `_reshape_by_patch` → CHW→HWC transpose + expand_dims, yielding the packed
//! `[1, 14, n*14, 3]` (HWC) tensor that `SiglipEmbeddings::forward_on` consumes
//! (n = grid_h*grid_w). The constants below are the model-fixed MiniCPM-V-4.6
//! values from the checkpoint's `preprocessor_config.json`
//! (`scale_resolution=448`, `patch_size=14`, `image_mean`/`image_std` = 0.5).

use anyhow::{anyhow, Result};
use mlx::ops::shape::expand_dims;
use mlx::Array;

/// `patch_size` from preprocessor_config.json.
const PATCH: i32 = 14;
/// `scale_resolution` from preprocessor_config.json.
const SCALE_RESOLUTION: i32 = 448;
/// `image_mean` from preprocessor_config.json (all channels 0.5).
const IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
/// `image_std` from preprocessor_config.json (all channels 0.5).
const IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Python 3 `round()` — round-half-to-even (banker's rounding). mlx-vlm's
/// `_ensure_divide`/`_find_best_resize` rely on this exact tie behaviour, so we
/// reproduce it rather than using Rust's round-half-away-from-zero.
fn py_round(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff > 0.5 {
        floor + 1.0
    } else if diff < 0.5 {
        floor
    } else {
        // Exactly .5 — round to even.
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

/// Port of mlx-vlm `_ensure_divide`: `max(round(length/patch)*patch, patch)`.
fn ensure_divide(length: f64, patch: i32) -> i32 {
    let p = patch as f64;
    (py_round(length / p) * p).max(p) as i32
}

/// Port of mlx-vlm `_find_best_resize` for the single-image path
/// (`allow_upscale=True`, so the rescale branch always runs).
///
/// Input/output are `(width, height)` to match the PIL `image.size` ordering
/// mlx-vlm uses. Returns `(best_width, best_height)`.
fn find_best_resize(width: i32, height: i32) -> (i32, i32) {
    // allow_upscale=True → always enter the rescale branch.
    let ratio = width as f64 / (height.max(1) as f64);
    let height = (SCALE_RESOLUTION as f64 / ratio.max(1e-6).sqrt()) as i32;
    let width = (height as f64 * ratio) as i32;

    let merge_factor = PATCH * 4;
    let best_width = ensure_divide(width as f64, merge_factor);
    let best_height = ensure_divide(height as f64, merge_factor);
    (best_width, best_height)
}

// --- PIL-exact BICUBIC resampler ---------------------------------------------
//
// mlx-vlm resizes with `PIL.Image.resize(size, BICUBIC)`. PIL's bicubic is a
// separable cubic-convolution filter (Keys, a = -0.5) applied in two passes
// (horizontal then vertical) over the u8 image, with fixed-point integer
// coefficients (PRECISION_BITS) and per-pass clamp-to-[0,255]. No `image`-crate
// FilterType reproduces it bit-exactly (CatmullRom — same kernel family — is the
// closest at ~7/255 worst-pixel error). We port PIL's algorithm directly so the
// resize matches mlx-vlm's bicubic output to ≤3/255 (the only remaining gap is
// the JPEG decoder: the `image` crate's libjpeg vs PIL's libjpeg already differ
// by ≤3/255 on the DECODED pixels before any resize — that decode floor, not
// the resize, is the residual the parity test observes). Normalize + pack are
// exact. Ref: Pillow `src/libImaging/Resample.c`.

const PRECISION_BITS: i32 = 32 - 8 - 2; // 22, as in Pillow Resample.c
const BICUBIC_SUPPORT: f64 = 2.0;

/// Keys cubic convolution kernel with `a = -0.5` (Pillow `bicubic_filter`).
fn bicubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Pillow's `(int)` truncation-toward-zero of a coefficient sum offset.
fn clip8(acc: i64) -> u8 {
    // acc is in fixed point with PRECISION_BITS fractional bits.
    let v = acc >> PRECISION_BITS;
    if v <= 0 {
        0
    } else if v >= 255 {
        255
    } else {
        v as u8
    }
}

/// Pillow `_round`: round half away from zero (used by `normalize_coeffs_8bpc`).
fn pil_round(x: f64) -> i64 {
    if x < 0.0 {
        (x - 0.5).ceil() as i64 // ceil(x-0.5) == round-half-away for negatives
    } else {
        (x + 0.5).floor() as i64
    }
}

/// Pillow `precompute_coeffs` for one axis. Returns fixed-point integer
/// coefficients `kk` (row-major `[out_size * ksize]`), `bounds` (`xmin,xsize`
/// pairs) and `ksize`.
fn precompute_coeffs(in_size: i32, out_size: i32) -> (Vec<i32>, Vec<i32>, usize) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = BICUBIC_SUPPORT * filterscale;
    let ksize = (support.ceil() as usize) * 2 + 1;

    let mut kk = vec![0_i32; out_size as usize * ksize];
    let mut bounds = vec![0_i32; out_size as usize * 2];
    let mut prekk = vec![0.0_f64; ksize];

    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // Pillow: xmin = (int)(center - support + 0.5)
        let mut xmin = (center - support + 0.5) as i32;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i32;
        if xmax > in_size {
            xmax = in_size;
        }
        xmax -= xmin;

        let mut ww = 0.0_f64;
        for x in 0..xmax {
            let w = bicubic_filter((x as f64 + xmin as f64 - center + 0.5) * ss);
            prekk[x as usize] = w;
            ww += w;
        }
        if ww != 0.0 {
            for w in prekk.iter_mut().take(xmax as usize) {
                *w /= ww;
            }
        }
        // normalize_coeffs_8bpc: fixed-point round
        let base = xx as usize * ksize;
        for (x, &w) in prekk.iter().take(xmax as usize).enumerate() {
            kk[base + x] = pil_round(w * (1_i64 << PRECISION_BITS) as f64) as i32;
        }
        // remaining ksize entries stay 0
        bounds[xx as usize * 2] = xmin;
        bounds[xx as usize * 2 + 1] = xmax;
    }
    (kk, bounds, ksize)
}

/// Horizontal resample pass (Pillow `ImagingResampleHorizontal_8bpc`).
/// `src`: HWC u8, `in_w`×`h`. Returns HWC u8, `out_w`×`h`.
fn resample_horizontal(src: &[u8], in_w: usize, h: usize, out_w: i32) -> Vec<u8> {
    let (kk, bounds, ksize) = precompute_coeffs(in_w as i32, out_w);
    let out_w_us = out_w as usize;
    let mut dst = vec![0_u8; out_w_us * h * 3];
    let init = 1_i64 << (PRECISION_BITS - 1); // rounding offset
    for yy in 0..h {
        let src_row = yy * in_w * 3;
        let dst_row = yy * out_w_us * 3;
        for xx in 0..out_w_us {
            let xmin = bounds[xx * 2] as usize;
            let xsize = bounds[xx * 2 + 1] as usize;
            let kbase = xx * ksize;
            let mut acc = [init, init, init];
            for x in 0..xsize {
                let k = kk[kbase + x] as i64;
                let sp = src_row + (xmin + x) * 3;
                acc[0] += src[sp] as i64 * k;
                acc[1] += src[sp + 1] as i64 * k;
                acc[2] += src[sp + 2] as i64 * k;
            }
            let dp = dst_row + xx * 3;
            dst[dp] = clip8(acc[0]);
            dst[dp + 1] = clip8(acc[1]);
            dst[dp + 2] = clip8(acc[2]);
        }
    }
    dst
}

/// Vertical resample pass (Pillow `ImagingResampleVertical_8bpc`).
/// `src`: HWC u8, `w`×`in_h`. Returns HWC u8, `w`×`out_h`.
fn resample_vertical(src: &[u8], w: usize, in_h: usize, out_h: i32) -> Vec<u8> {
    let (kk, bounds, ksize) = precompute_coeffs(in_h as i32, out_h);
    let out_h_us = out_h as usize;
    let mut dst = vec![0_u8; w * out_h_us * 3];
    let init = 1_i64 << (PRECISION_BITS - 1);
    for yy in 0..out_h_us {
        let ymin = bounds[yy * 2] as usize;
        let ysize = bounds[yy * 2 + 1] as usize;
        let kbase = yy * ksize;
        let dst_row = yy * w * 3;
        for xx in 0..w {
            let mut acc = [init, init, init];
            for y in 0..ysize {
                let k = kk[kbase + y] as i64;
                let sp = (ymin + y) * w * 3 + xx * 3;
                acc[0] += src[sp] as i64 * k;
                acc[1] += src[sp + 1] as i64 * k;
                acc[2] += src[sp + 2] as i64 * k;
            }
            let dp = dst_row + xx * 3;
            dst[dp] = clip8(acc[0]);
            dst[dp + 1] = clip8(acc[1]);
            dst[dp + 2] = clip8(acc[2]);
        }
    }
    dst
}

/// PIL-exact BICUBIC resize of an HWC u8 RGB buffer `in_w`×`in_h` → `out_w`×`out_h`.
/// Two-pass (horizontal then vertical), matching `PIL.Image.resize(BICUBIC)`.
fn pil_bicubic_resize(src: &[u8], in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    // Pillow resizes horizontally first, then vertically.
    let horiz = resample_horizontal(src, in_w, in_h, out_w as i32);
    resample_vertical(&horiz, out_w, in_h, out_h as i32)
}

/// `(px/255 - mean) / std` for a single RGB pixel.
fn normalize_pixel(rgb: [u8; 3]) -> [f32; 3] {
    [
        (rgb[0] as f32 / 255.0 - IMAGE_MEAN[0]) / IMAGE_STD[0],
        (rgb[1] as f32 / 255.0 - IMAGE_MEAN[1]) / IMAGE_STD[1],
        (rgb[2] as f32 / 255.0 - IMAGE_MEAN[2]) / IMAGE_STD[2],
    ]
}

/// Single-image (no-slice) MiniCPM-V-4.6 preprocess.
///
/// Returns `(pixel_values [1, 14, n*14, 3] f32 (HWC), grid_h, grid_w)` where
/// `n = grid_h*grid_w`. The layout matches mlx-vlm's `_reshape_by_patch`
/// followed by `get_vision_embedding`'s CHW→HWC transpose + `expand_dims(0)`,
/// i.e. the exact bytes `SiglipEmbeddings::forward_on` consumes.
pub fn preprocess(img_bytes: &[u8]) -> Result<(Array, i32, i32)> {
    // 1. Decode → RGB8 (matches PIL `.convert("RGB")`).
    let img = image::load_from_memory(img_bytes)
        .map_err(|e| anyhow!("decode image: {e}"))?
        .to_rgb8();
    let (orig_w, orig_h) = (img.width() as i32, img.height() as i32);

    // 2. _find_best_resize on (width, height).
    let (best_w, best_h) = find_best_resize(orig_w, orig_h);

    // 3. PIL-exact BICUBIC resize over the HWC u8 buffer (see `pil_bicubic_resize`).
    let src_hwc: &[u8] = img.as_raw(); // HWC u8, orig_w × orig_h
    let resized: Vec<u8> = if orig_w == best_w && orig_h == best_h {
        src_hwc.to_vec()
    } else {
        pil_bicubic_resize(
            src_hwc,
            orig_w as usize,
            orig_h as usize,
            best_w as usize,
            best_h as usize,
        )
    };

    let h = best_h;
    let w = best_w;
    let grid_h = h / PATCH;
    let grid_w = w / PATCH;
    let n = grid_h * grid_w;

    // 4 + 5. Normalize and pack directly into the final HWC packed layout
    //        `[1, 14, n*14, 3]`.
    //
    // mlx-vlm builds CHW `(3, H, W)` normalized pixels, runs `_reshape_by_patch`
    // → CHW-packed `(3, 14, n*14)` whose last axis flattens `(gh, gw, pw)`
    // row-major, then `get_vision_embedding` transposes `(1,2,0)` → HWC
    // `(14, n*14, 3)` and `expand_dims(0)` → `(1, 14, n*14, 3)`.
    //
    // Net mapping: out[0, ph, j, c] = normalized_pixel(c, gh*14 + ph, gw*14 + pw)
    // where j = ((gh * grid_w) + gw) * 14 + pw. We write that mapping directly.
    let total_w = (n * PATCH) as usize; // n*14
    let mut packed = vec![0.0_f32; PATCH as usize * total_w * 3];
    let row_w = w as usize; // resized image row width in pixels (HWC)

    for gh in 0..grid_h {
        for ph in 0..PATCH {
            let src_y = (gh * PATCH + ph) as usize;
            for gw in 0..grid_w {
                let j_base = (((gh * grid_w) + gw) * PATCH) as usize;
                for pw in 0..PATCH {
                    let src_x = (gw * PATCH + pw) as usize;
                    let sp = (src_y * row_w + src_x) * 3;
                    let norm = normalize_pixel([resized[sp], resized[sp + 1], resized[sp + 2]]);
                    // out index for [1, 14, total_w, 3] HWC, batch 0:
                    //   ((ph * total_w) + j) * 3 + c
                    let j = j_base + pw as usize;
                    let base = ((ph as usize) * total_w + j) * 3;
                    packed[base] = norm[0];
                    packed[base + 1] = norm[1];
                    packed[base + 2] = norm[2];
                }
            }
        }
    }

    // Build [14, n*14, 3] then expand_dims(0) → [1, 14, n*14, 3], matching the
    // qwen3_5 image_processor idiom for Array construction + shape ops.
    let arr: Array = (packed.as_slice(), &[PATCH, total_w as i32, 3][..])
        .try_into()
        .map_err(|e| anyhow!("preprocess: array construction: {e}"))?;
    let arr = expand_dims(&arr, &[0_i32][..])?;
    Ok((arr, grid_h, grid_w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_centers_to_minus1_to_1() {
        assert!((normalize_pixel([0, 0, 0])[0] - (-1.0)).abs() < 1e-6);
        assert!((normalize_pixel([255, 255, 255])[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn py_round_half_to_even() {
        assert_eq!(py_round(0.5), 0.0);
        assert_eq!(py_round(1.5), 2.0);
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(9.214), 9.0);
        assert_eq!(py_round(6.911), 7.0);
    }

    #[test]
    fn ensure_divide_matches_python() {
        // _ensure_divide(516, 56) = round(9.214)*56 = 9*56 = 504
        assert_eq!(ensure_divide(516.0, 56), 504);
        // _ensure_divide(387, 56) = round(6.911)*56 = 7*56 = 392
        assert_eq!(ensure_divide(387.0, 56), 392);
        // floor at patch
        assert_eq!(ensure_divide(1.0, 56), 56);
    }

    #[test]
    fn find_best_resize_coco_640x480() {
        // PIL image.size = (width=640, height=480).
        // ratio = 640/480 = 1.3333; height = int(448/sqrt(1.3333)) = int(387.97)=387
        // width = int(387*1.3333) = int(516.0) = 516
        // best = (ensure_divide(516,56), ensure_divide(387,56)) = (504, 392)
        let (bw, bh) = find_best_resize(640, 480);
        assert_eq!((bw, bh), (504, 392));
        // grid_h = 392/14 = 28, grid_w = 504/14 = 36
        assert_eq!((bh / PATCH, bw / PATCH), (28, 36));
    }

    #[test]
    fn bicubic_filter_keys_a_minus_half() {
        // Keys cubic (a=-0.5): f(0)=1, f(1)=0, f(2)=0, even symmetry.
        assert!((bicubic_filter(0.0) - 1.0).abs() < 1e-12);
        assert!(bicubic_filter(1.0).abs() < 1e-12);
        assert!(bicubic_filter(2.0).abs() < 1e-12);
        assert!((bicubic_filter(0.5) - bicubic_filter(-0.5)).abs() < 1e-12);
        assert_eq!(bicubic_filter(3.0), 0.0);
    }

    #[test]
    fn pil_bicubic_resize_identity_when_same_size() {
        // 2×2 RGB → 2×2 must be a no-op (coeffs reduce to a unit kernel).
        let src: Vec<u8> = vec![
            10, 20, 30, 40, 50, 60, // row 0: px(0,0), px(1,0)
            70, 80, 90, 100, 110, 120, // row 1: px(0,1), px(1,1)
        ];
        let out = pil_bicubic_resize(&src, 2, 2, 2, 2);
        assert_eq!(out, src);
    }

    #[test]
    fn pil_bicubic_resize_output_dims() {
        // 4×4 → 2×3 (out_w=2, out_h=3): output length = out_w*out_h*3.
        let src: Vec<u8> = (0..(4 * 4 * 3)).map(|i| (i % 256) as u8).collect();
        let out = pil_bicubic_resize(&src, 4, 4, 2, 3);
        assert_eq!(out.len(), 2 * 3 * 3);
    }
}
