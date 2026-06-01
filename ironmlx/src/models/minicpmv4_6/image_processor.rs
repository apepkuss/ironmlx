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

/// Ordered list of preprocessed image slices returned by the LLaVA-UHD pipeline:
/// `(pixel_values [1, 14, n*14, 3], grid_h, grid_w)` per slice. Source first,
/// then refine patches in row-major order.
pub type SlicedImages = Vec<(Array, i32, i32)>;

/// `patch_size` from preprocessor_config.json.
const PATCH: i32 = 14;
/// `scale_resolution` from preprocessor_config.json.
const SCALE_RESOLUTION: i32 = 448;
/// `max_slice_nums` from preprocessor_config.json — the LLaVA-UHD slice cap and
/// the default `max_slice_nums` argument for [`preprocess_sliced`].
pub const MAX_SLICE_NUMS: i32 = 9;
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

/// Port of mlx-vlm `_find_best_resize`.
///
/// Input is `(width, height)` to match the PIL `image.size` ordering mlx-vlm
/// uses; the values are `f64` because `_get_refine_size` passes fractional
/// `grid_width`/`grid_height`. Returns `(best_width, best_height)` integers.
///
/// The rescale branch runs iff `width*height > scale_resolution²` OR
/// `allow_upscale`. `int(...)` casts truncate toward zero (Python `int()`).
fn find_best_resize(width: f64, height: f64, allow_upscale: bool) -> (i32, i32) {
    let (mut w, mut h) = (width, height);
    let scale = SCALE_RESOLUTION as f64;
    if (w * h > scale * scale) || allow_upscale {
        let ratio = w / h.max(1.0);
        // Python `int()` truncates toward zero; values here are positive.
        let new_h = (scale / ratio.max(1e-6).sqrt()).trunc();
        let new_w = (new_h * ratio).trunc();
        w = new_w;
        h = new_h;
    }
    let merge_factor = PATCH * 4; // PATCH * 4: mlx-vlm dims are kept divisible by patch_size * merge_size(=4).
    let best_width = ensure_divide(w, merge_factor);
    let best_height = ensure_divide(h, merge_factor);
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
        // xmax is now the tap count (not a right-boundary index); stored below as the per-row 'xsize'.
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

/// Normalize a resized HWC u8 RGB buffer and pack it into the
/// `[14, n*14, 3]` flat layout used by `SiglipEmbeddings`.
///
/// # Net mapping
/// `out[ph, j, c] = normalize_pixel(c, gh*14 + ph, gw*14 + pw)`
/// where `j = ((gh * grid_w) + gw) * 14 + pw`.
///
/// Returns `(packed, grid_h, grid_w)`.
fn pack_patches(resized: &[u8], h: i32, w: i32) -> (Vec<f32>, i32, i32) {
    let grid_h = h / PATCH;
    let grid_w = w / PATCH;
    let n = grid_h * grid_w;
    let total_w = (n * PATCH) as usize; // n*14
    let mut packed = vec![0.0_f32; PATCH as usize * total_w * 3];
    let row_w = w as usize; // resized image row width in pixels (HWC)

    // mlx-vlm builds CHW `(3, H, W)` normalized pixels, runs `_reshape_by_patch`
    // → CHW-packed `(3, 14, n*14)` whose last axis flattens `(gh, gw, pw)`
    // row-major, then `get_vision_embedding` transposes `(1,2,0)` → HWC
    // `(14, n*14, 3)` and `expand_dims(0)` → `(1, 14, n*14, 3)`.
    for gh in 0..grid_h {
        for ph in 0..PATCH {
            let src_y = (gh * PATCH + ph) as usize;
            for gw in 0..grid_w {
                let j_base = (((gh * grid_w) + gw) * PATCH) as usize;
                for pw in 0..PATCH {
                    let src_x = (gw * PATCH + pw) as usize;
                    let sp = (src_y * row_w + src_x) * 3;
                    let norm = normalize_pixel([resized[sp], resized[sp + 1], resized[sp + 2]]);
                    // out index for [14, total_w, 3] HWC:
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
    (packed, grid_h, grid_w)
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

    // 2. _find_best_resize on (width, height) — single-image path is allow_upscale=True.
    let (best_w, best_h) = find_best_resize(orig_w as f64, orig_h as f64, true);

    // 3. PIL-exact BICUBIC resize over the HWC u8 buffer (see `pil_bicubic_resize`).
    let src_hwc: &[u8] = img.as_raw(); // HWC u8, orig_w × orig_h
    let resized: Vec<u8> = resize_rgb(src_hwc, orig_w, orig_h, best_w, best_h);

    // 4 + 5. Normalize, pack into [14, n*14, 3], expand_dims(0) → [1, 14, n*14, 3].
    slice_to_array(&resized, best_h, best_w)
}

/// Resize an HWC u8 RGB buffer `orig_w`×`orig_h` → `dst_w`×`dst_h` with PIL-exact
/// BICUBIC, short-circuiting a no-op resize (identical dims) to a copy.
fn resize_rgb(src_hwc: &[u8], orig_w: i32, orig_h: i32, dst_w: i32, dst_h: i32) -> Vec<u8> {
    if orig_w == dst_w && orig_h == dst_h {
        src_hwc.to_vec()
    } else {
        pil_bicubic_resize(
            src_hwc,
            orig_w as usize,
            orig_h as usize,
            dst_w as usize,
            dst_h as usize,
        )
    }
}

/// Normalize + pack a resized HWC u8 RGB buffer (`w`×`h`) into the
/// `[1, 14, n*14, 3]` (HWC) tensor `SiglipEmbeddings::forward_on` consumes, plus
/// `(grid_h = h/14, grid_w = w/14)`. Shared by `preprocess` and
/// `preprocess_sliced` so every slice goes through the identical code path.
fn slice_to_array(resized: &[u8], h: i32, w: i32) -> Result<(Array, i32, i32)> {
    let (packed, grid_h, grid_w) = pack_patches(resized, h, w);
    let n = grid_h * grid_w;
    let total_w = (n * PATCH) as usize;

    // Build [14, n*14, 3] then expand_dims(0) → [1, 14, n*14, 3], matching the
    // qwen3_5 image_processor idiom for Array construction + shape ops.
    let arr: Array = (packed.as_slice(), &[PATCH, total_w as i32, 3][..])
        .try_into()
        .map_err(|e| anyhow!("slice_to_array: array construction: {e}"))?;
    let arr = expand_dims(&arr, &[0_i32][..])?;
    Ok((arr, grid_h, grid_w))
}

// --- LLaVA-UHD adaptive multi-slice preprocessing (slice_mode=True) ----------
//
// Port of mlx-vlm's `slice_image` / `get_sliced_grid` / `_get_refine_size` /
// `_split_to_patches`. Operates on the decoded HWC u8 RGB buffer (carried as
// `(Vec<u8>, width, height)`); crops + resizes reuse `resize_rgb` /
// `pil_bicubic_resize` so the per-slice pixels go through the same resampler as
// the single-image path. Grid tuples are `(grid_x, grid_y)` matching PIL/mlx-vlm
// where `grid_x` divides WIDTH and `grid_y` divides HEIGHT.

/// Port of mlx-vlm `get_sliced_grid`. Returns `Some((grid_x, grid_y))` when the
/// image should be sliced, or `None` (`multiple <= 1`) for the no-slice path.
///
/// Input `(width, height)`; `grid_x` divides width, `grid_y` divides height.
fn get_sliced_grid(width: i32, height: i32, max_slice_nums: i32) -> Option<(i32, i32)> {
    let scale = SCALE_RESOLUTION as f64;
    let ratio = (width as f64 * height as f64) / (scale * scale);
    // multiple = min(ceil(ratio), max_slice_nums)
    let multiple = (ratio.ceil() as i32).min(max_slice_nums);
    if multiple <= 1 {
        return None;
    }

    // Candidate grid_nums: {multiple-1, multiple, multiple+1} filtered to
    // `gn != 1 && gn <= max_slice_nums`.
    let mut candidate_grids: Vec<(i32, i32)> = Vec::new();
    for grid_num in [multiple - 1, multiple, multiple + 1] {
        if grid_num == 1 || grid_num > max_slice_nums {
            continue;
        }
        // All (factor, grid_num/factor) for factor dividing grid_num, ascending.
        let mut factor = 1;
        while factor <= grid_num {
            if grid_num % factor == 0 {
                candidate_grids.push((factor, grid_num / factor));
            }
            factor += 1;
        }
    }

    // No valid factor-pair candidates (degenerate max_slice_nums) → treat as no-slice.
    if candidate_grids.is_empty() {
        return None;
    }

    // Pick the grid minimizing |log(w/h) - log(gx/gy)|. First-seen wins on ties
    // (strict `<`), matching Python's iteration order.
    let log_ratio = (width as f64 / (height.max(1) as f64)).ln();
    let mut best_grid = (1, 1);
    let mut min_error = f64::INFINITY;
    for grid in candidate_grids {
        let error = (log_ratio - (grid.0 as f64 / grid.1 as f64).ln()).abs();
        if error < min_error {
            best_grid = grid;
            min_error = error;
        }
    }
    Some(best_grid)
}

/// Port of mlx-vlm `_get_refine_size`. Input `(width, height)` and the chosen
/// `grid = (grid_x, grid_y)`; returns the refine-image `(width, height)` (each
/// dimension a clean multiple of the corresponding grid dimension).
fn get_refine_size(width: i32, height: i32, grid: (i32, i32)) -> (i32, i32) {
    let (gx, gy) = grid;
    let refine_w = ensure_divide(width as f64, gx);
    let refine_h = ensure_divide(height as f64, gy);
    // grid_width / grid_height are fractional; _find_best_resize takes floats.
    let grid_width = refine_w as f64 / gx as f64;
    let grid_height = refine_h as f64 / gy as f64;
    // mlx-vlm calls with allow_upscale=True here.
    let (best_w, best_h) = find_best_resize(grid_width, grid_height, true);
    (best_w * gx, best_h * gy)
}

/// Port of mlx-vlm `_split_to_patches`. Crops the resized refine image
/// (`(buf, width, height)`) into a row-major grid of cells.
///
/// `cell_w = width / grid_x`, `cell_h = height / grid_y` (integer division).
/// Iterates `top` (rows) outer, `left` (cols) inner — row-major. Returns the
/// flattened cells as `(cell_buf, cell_w, cell_h)`.
fn split_to_patches(
    buf: &[u8],
    width: i32,
    height: i32,
    grid: (i32, i32),
) -> Vec<(Vec<u8>, i32, i32)> {
    let (gx, gy) = grid;
    let cell_w = width / gx;
    let cell_h = height / gy;
    let row_stride = width as usize * 3;
    let mut out = Vec::with_capacity((gx * gy) as usize);

    let mut top = 0;
    while top < height {
        let mut left = 0;
        while left < width {
            // Crop (left, top, left+cell_w, top+cell_h) → HWC u8.
            let mut cell = vec![0_u8; (cell_w * cell_h * 3) as usize];
            for row in 0..cell_h as usize {
                let src_off = (top as usize + row) * row_stride + left as usize * 3;
                let dst_off = row * cell_w as usize * 3;
                let span = cell_w as usize * 3;
                cell[dst_off..dst_off + span].copy_from_slice(&buf[src_off..src_off + span]);
            }
            out.push((cell, cell_w, cell_h));
            left += cell_w;
        }
        top += cell_h;
    }
    out
}

/// Port of mlx-vlm `slice_image`. Operates on the decoded HWC u8 RGB buffer
/// `(src, orig_w, orig_h)` and returns the ordered slice list (source first,
/// then patches row-major), each `(resized_buf, width, height)`, plus the
/// `best_grid = Some((grid_x, grid_y))` that drove the slicing (`None` when
/// no slicing was applied).
///
/// - No-slice (`get_sliced_grid` → None): one resized source with
///   `find_best_resize(allow_upscale=true)`.
/// - Slice: source uses `find_best_resize(allow_upscale=false)`; the refine
///   image uses `get_refine_size` and is split into `grid_x*grid_y` patches.
#[allow(clippy::type_complexity)]
fn slice_image(
    src: &[u8],
    orig_w: i32,
    orig_h: i32,
    max_slice_nums: i32,
) -> (Vec<(Vec<u8>, i32, i32)>, Option<(i32, i32)>) {
    match get_sliced_grid(orig_w, orig_h, max_slice_nums) {
        None => {
            let (best_w, best_h) = find_best_resize(orig_w as f64, orig_h as f64, true);
            let source = resize_rgb(src, orig_w, orig_h, best_w, best_h);
            (vec![(source, best_w, best_h)], None)
        }
        Some(grid) => {
            // Source image: allow_upscale=false.
            let (src_w, src_h) = find_best_resize(orig_w as f64, orig_h as f64, false);
            let source = resize_rgb(src, orig_w, orig_h, src_w, src_h);

            // Refine image: allow_upscale=true, then split into patches.
            let (refine_w, refine_h) = get_refine_size(orig_w, orig_h, grid);
            let refine = resize_rgb(src, orig_w, orig_h, refine_w, refine_h);
            let patches = split_to_patches(&refine, refine_w, refine_h, grid);

            let mut slices = Vec::with_capacity(1 + patches.len());
            slices.push((source, src_w, src_h));
            slices.extend(patches);
            (slices, Some(grid))
        }
    }
}

/// LLaVA-UHD adaptive multi-slice MiniCPM-V-4.6 preprocess (`slice_mode=True`).
///
/// Returns the ordered slice list — the source (overview) image first, then the
/// refine-image patches in row-major order — each as
/// `(pixel_values [1, 14, n*14, 3] f32 (HWC), grid_h, grid_w)` where
/// `n = grid_h*grid_w`. When the image is too small to slice (`get_sliced_grid`
/// → None) the list has a single element identical to `preprocess`'s output.
///
/// `max_slice_nums` caps the LLaVA-UHD slice count; the checkpoint default is 9
/// (see [`MAX_SLICE_NUMS`]).
pub fn preprocess_sliced(img_bytes: &[u8], max_slice_nums: i32) -> Result<SlicedImages> {
    let (slices, _best_grid) = preprocess_sliced_inner(img_bytes, max_slice_nums)?;
    Ok(slices)
}

/// LLaVA-UHD adaptive multi-slice preprocess that also surfaces the slice grid.
///
/// Like [`preprocess_sliced`] but additionally returns `best_grid =
/// Some((grid_x, grid_y))` (the LLaVA-UHD grid that drove the slicing), or
/// `None` when the image was too small to slice.  The slice grid is required by
/// the prompt-placeholder builder ([`crate::models::minicpmv4_6::sliced_image_placeholder_string`]).
///
/// Order guarantees are identical to [`preprocess_sliced`]: source first, then
/// refine patches in row-major order matching `grid_x` (columns) × `grid_y`
/// (rows).
pub fn preprocess_sliced_with_grid(
    img_bytes: &[u8],
    max_slice_nums: i32,
) -> Result<(SlicedImages, Option<(i32, i32)>)> {
    preprocess_sliced_inner(img_bytes, max_slice_nums)
}

/// Shared implementation for [`preprocess_sliced`] and
/// [`preprocess_sliced_with_grid`].
fn preprocess_sliced_inner(
    img_bytes: &[u8],
    max_slice_nums: i32,
) -> Result<(SlicedImages, Option<(i32, i32)>)> {
    // 1. Decode → RGB8 (matches PIL `.convert("RGB")`).
    let img = image::load_from_memory(img_bytes)
        .map_err(|e| anyhow!("decode image: {e}"))?
        .to_rgb8();
    let (orig_w, orig_h) = (img.width() as i32, img.height() as i32);
    let src_hwc: &[u8] = img.as_raw();

    // 2. LLaVA-UHD slice (source + row-major refine patches).
    let (raw_slices, best_grid) = slice_image(src_hwc, orig_w, orig_h, max_slice_nums);

    // 3. Each slice → normalize + pack + Array (shared with `preprocess`).
    let slices: Result<Vec<(Array, i32, i32)>> = raw_slices
        .into_iter()
        .map(|(buf, w, h)| slice_to_array(&buf, h, w))
        .collect();
    Ok((slices?, best_grid))
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
        let (bw, bh) = find_best_resize(640.0, 480.0, true);
        assert_eq!((bw, bh), (504, 392));
        // grid_h = 392/14 = 28, grid_w = 504/14 = 36
        assert_eq!((bh / PATCH, bw / PATCH), (28, 36));
    }

    #[test]
    fn get_sliced_grid_cases() {
        // Small image: ratio = 300*300/448² = 0.4484 → ceil = 1 → multiple=1 → None.
        assert_eq!(get_sliced_grid(300, 300, 9), None);
        // Exactly scale²: ratio = 1.0 → ceil = 1 → multiple=1 → None.
        assert_eq!(get_sliced_grid(448, 448, 9), None);

        // coco_sample (640×480): ratio = 1.5306 → ceil=2 → multiple=2.
        // candidates {1(skip),2,3}: gn=2 → (1,2),(2,1); gn=3 → (1,3),(3,1).
        // log(640/480)=0.2877; errors: (1,2)=|0.2877-(-0.693)|=0.981,
        // (2,1)=|0.2877-0.693|=0.405 (best so far), (1,3)=1.386, (3,1)=0.811.
        // → best (2,1): grid_x=2 (width split), grid_y=1.
        assert_eq!(get_sliced_grid(640, 480, 9), Some((2, 1)));

        // Wide landscape (1600×600): ratio = 960000/200704 = 4.7832 → ceil=5 → multiple=5.
        // candidates {4,5,6}: gn=4→(1,4),(2,2),(4,1); gn=5→(1,5),(5,1);
        // gn=6→(1,6),(2,3),(3,2),(6,1).
        // log(1600/600)=ln(2.6667)=0.9808; errors: (4,1)=|0.9808-ln(4)|=0.405 (min),
        // next-best (3,2)=|0.9808-ln(1.5)|=0.576, (5,1)=0.628 → best (4,1).
        assert_eq!(get_sliced_grid(1600, 600, 9), Some((4, 1)));

        // 1280×960: ratio = 1228800/200704 = 6.1224 → ceil=7 → multiple=7.
        // candidates {6,7,8}: gn=6→(1,6),(2,3),(3,2),(6,1); gn=7→(1,7),(7,1);
        // gn=8→(1,8),(2,4),(4,2),(8,1).
        // log(1280/960)=ln(1.3333)=0.2877; errors: (3,2)=|0.2877-ln(1.5)|=0.118 (min),
        // next-best (4,2)=|0.2877-ln(2)|=0.405, (2,3)=0.693 → best (3,2).
        assert_eq!(get_sliced_grid(1280, 960, 9), Some((3, 2)));

        // 2000×1000: ratio = 9.96 capped at max=9 → best (4,2).
        assert_eq!(get_sliced_grid(2000, 1000, 9), Some((4, 2)));
    }

    #[test]
    fn ensure_divide_and_refine() {
        // ensure_divide(100, 56) = round(1.7857)*56 = 2*56 = 112.
        assert_eq!(ensure_divide(100.0, 56), 112);

        // get_refine_size for coco_sample (640×480) with grid (gx=2, gy=1):
        //   refine_w = ensure_divide(640, 2) = round(320)*2 = 640
        //   refine_h = ensure_divide(480, 1) = round(480)*1 = 480
        //   grid_width = 640/2 = 320, grid_height = 480/1 = 480
        //   find_best_resize((320,480), allow_upscale=true):
        //     ratio = 320/480 = 0.6667; h = int(448/sqrt(0.6667)) = int(548.6) = 548
        //     w = int(548*0.6667) = int(365.3) = 365
        //     best = (ensure_divide(365,56), ensure_divide(548,56)) = (392, 560)
        //   refine_size = (392*2, 560*1) = (784, 560)
        assert_eq!(get_refine_size(640, 480, (2, 1)), (784, 560));
        // refine dims must be clean multiples of (patch * grid) per axis:
        // 784 / (14*2) = 28 patch-cols/grid_x, 560 / (14*1) = 40 patch-rows.
        assert_eq!(784 % (PATCH * 2), 0);
        assert_eq!(560 % (PATCH * 1), 0);
    }

    #[test]
    fn split_to_patches_row_major() {
        // 4×2 image (W=4,H=2), grid (gx=2, gy=1): cell_w=2, cell_h=2, 2 cells.
        // Build HWC u8 where px(x,y,c) = x*10 + y*100 + c so we can identify cells.
        let (w, h) = (4_i32, 2_i32);
        let mut buf = vec![0_u8; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                for c in 0..3usize {
                    buf[(y * w as usize + x) * 3 + c] = (x * 10 + y * 100 + c) as u8;
                }
            }
        }
        let cells = split_to_patches(&buf, w, h, (2, 1));
        assert_eq!(cells.len(), 2);
        // Cell 0: left=0 → columns x in {0,1}. Cell 1: left=2 → x in {2,3}.
        assert_eq!((cells[0].1, cells[0].2), (2, 2));
        // top-left pixel of cell 0 = px(0,0): [0,1,2]
        assert_eq!(&cells[0].0[0..3], &[0, 1, 2]);
        // top-left pixel of cell 1 = px(2,0): [20,21,22]
        assert_eq!(&cells[1].0[0..3], &[20, 21, 22]);
        // row 1 of cell 1, col 0 = px(2,1): [120,121,122]
        let row1 = cells[1].0[(2 * 3)..(2 * 3 + 3)].to_vec();
        assert_eq!(row1, vec![120, 121, 122]);
    }

    #[test]
    fn slice_image_count_and_grids_coco() {
        // Synthetic 640×480 solid buffer → grid (2,1) → 3 slices.
        let (w, h) = (640_i32, 480_i32);
        let buf = vec![128_u8; (w * h * 3) as usize];
        let (slices, best_grid) = slice_image(&buf, w, h, 9);
        assert_eq!(slices.len(), 3, "1 source + 2 refine patches");
        assert_eq!(best_grid, Some((2, 1)), "best_grid should be (gx=2, gy=1)");
        // Source: find_best_resize(allow_upscale=false) → area 307200 > 448²=200704 so rescale fires anyway → (504, 392).
        assert_eq!((slices[0].1, slices[0].2), (504, 392));
        // Each refine patch: refine (784×560) split (2,1) → cell (392, 560).
        assert_eq!((slices[1].1, slices[1].2), (392, 560));
        assert_eq!((slices[2].1, slices[2].2), (392, 560));
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
        // same-size resize: the 3-tap bicubic kernel over edge-clamped boundaries reproduces the original pixels
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

    #[test]
    fn precompute_coeffs_downscale_produces_normalized_taps() {
        // 8 → 4: filterscale = 2.0 > 1.0, so ksize is larger than in the upscale
        // (same-size) case where filterscale == 1.0.
        let (kk_down, _bounds_down, ksize_down) = precompute_coeffs(8, 4);

        // ksize for downscale (filterscale=2): support = 2*2 = 4 → ceil(4)*2+1 = 9
        // ksize for identity (filterscale=1): support = 2*1 = 2 → ceil(2)*2+1 = 5
        let (_kk_id, _bounds_id, ksize_id) = precompute_coeffs(4, 4);
        assert!(
            ksize_down > ksize_id,
            "downscale ksize ({ksize_down}) should exceed identity ksize ({ksize_id})"
        );

        // Each output position's coefficients must sum to 1<<PRECISION_BITS in
        // fixed-point (the normalize_coeffs_8bpc convention Pillow uses).
        let expected_sum = 1_i64 << PRECISION_BITS;
        let tolerance: i64 = 4; // rounding headroom: at most 1 ULP per tap

        for out_idx in 0..4_usize {
            let base = out_idx * ksize_down;
            let sum: i64 = kk_down[base..base + ksize_down]
                .iter()
                .map(|&v| v as i64)
                .sum();
            assert!(
                (sum - expected_sum).abs() <= tolerance,
                "output pixel {out_idx}: coeff sum {sum} deviates from {expected_sum} by more than {tolerance}"
            );
        }
    }

    #[test]
    fn pack_loop_maps_patch_grid_row_major() {
        // Build a 28×28 synthetic HWC u8 image (2×2 patch grid, PATCH=14).
        // Each pixel value encodes its position: pixel[row, col, ch] is set to
        // a distinguishable u8 derived from (row, col, ch).  We then call
        // pack_patches and verify the pack-loop index mapping for one known cell.
        //
        // pack_patches: out[ph, j, c] = normalize_pixel(c, gh*14+ph, gw*14+pw)
        // where j = ((gh*grid_w + gw)*14 + pw).
        //
        // For gh=1, gw=0, ph=3, pw=5, c=2:
        //   src_y = 1*14+3 = 17, src_x = 0*14+5 = 5
        //   j     = (1*2 + 0)*14 + 5 = 33
        //   base  = (3 * (2*2*14) + 33) * 3 + 2 = (3*56 + 33)*3 + 2 = 201*3 + 2 = 605

        const H: i32 = 28;
        const W: i32 = 28;
        let mut src = vec![0_u8; (H * W * 3) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                for ch in 0..3_usize {
                    // encode: (row * W * 3 + col * 3 + ch) % 251  (251 is prime, fits u8)
                    src[(row * W as usize + col) * 3 + ch] =
                        ((row * W as usize * 3 + col * 3 + ch) % 251) as u8;
                }
            }
        }

        let (packed, grid_h, grid_w) = pack_patches(&src, H, W);
        assert_eq!((grid_h, grid_w), (2, 2));

        // Spot-check: gh=1, gw=0, ph=3, pw=5, c=2
        let gh = 1_i32;
        let gw = 0_i32;
        let ph = 3_i32;
        let pw = 5_i32;
        let c = 2_usize;

        let src_y = (gh * PATCH + ph) as usize;
        let src_x = (gw * PATCH + pw) as usize;
        let sp = (src_y * W as usize + src_x) * 3 + c;
        let expected_norm = normalize_pixel([src[sp - c], src[sp - c + 1], src[sp - c + 2]])[c];

        let total_w = (grid_h * grid_w * PATCH) as usize; // 2*2*14=56
        let j = (((gh * grid_w) + gw) * PATCH + pw) as usize;
        let base = (ph as usize * total_w + j) * 3 + c;

        assert!(
            (packed[base] - expected_norm).abs() < 1e-6,
            "packed[ph={ph}, j={j}, c={c}] = {} but expected normalize_pixel(...)[{c}] = {}",
            packed[base],
            expected_norm
        );
    }
}
