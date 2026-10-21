//! PIL-exact BICUBIC resize of RGB images, shared by the vision
//! preprocessors (Qwen, MiniCPM-V, Gemma4) whose references resize with
//! `PIL.Image.resize(size, BICUBIC)`.
//!
//! mlx-vlm resizes with `PIL.Image.resize(size, BICUBIC)`. PIL's bicubic is a
//! separable cubic-convolution filter (Keys, a = -0.5) applied in two passes
//! (horizontal then vertical) over the u8 image, with fixed-point integer
//! coefficients (PRECISION_BITS) and per-pass clamp-to-[0,255]. No `image`-crate
//! FilterType reproduces it bit-exactly (CatmullRom — same kernel family — is the
//! closest at ~7/255 worst-pixel error). We port PIL's algorithm directly so the
//! resize matches mlx-vlm's bicubic output to ≤3/255 (the only remaining gap is
//! the JPEG decoder: the `image` crate's libjpeg vs PIL's libjpeg already differ
//! by ≤3/255 on the DECODED pixels before any resize — that decode floor, not
//! the resize, is the residual the parity test observes). Ref: Pillow
//! `src/libImaging/Resample.c`.

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

/// `src_hwc` (HWC u8 RGB, `orig_w`×`orig_h`) resized to `dst_w`×`dst_h`
/// with PIL's BICUBIC; a same-size request returns the pixels unchanged.
pub(crate) fn resize_rgb(
    src_hwc: &[u8],
    orig_w: i32,
    orig_h: i32,
    dst_w: i32,
    dst_h: i32,
) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Synthetic images (no JPEG decoder involved) resized up and down
    /// match Pillow's BICUBIC byte for byte (`gen_pil_bicubic_synthetic.py`).
    #[test]
    fn resize_matches_pillow_bicubic_exactly() {
        let expected =
            include_bytes!("../../../ironmlx/tests/fixtures/qwen35_vl/pil_bicubic_synthetic.bin");
        let synthetic = |w: i32, h: i32| -> Vec<u8> {
            let mut pixels = Vec::with_capacity((w * h * 3) as usize);
            for y in 0..h {
                for x in 0..w {
                    pixels.push(((x * 37 + y * 11) % 256) as u8);
                    pixels.push(((x * 5 + y * 29 + 7) % 256) as u8);
                    pixels.push(((x * x + 3 * y) % 256) as u8);
                }
            }
            pixels
        };
        let mut offset = 0;
        for ((w, h), (dw, dh)) in [((37, 23), (64, 48)), ((97, 61), (32, 24))] {
            let len = (dw * dh * 3) as usize;
            let resized = resize_rgb(&synthetic(w, h), w, h, dw, dh);
            assert_eq!(
                resized,
                &expected[offset..offset + len],
                "{w}x{h} -> {dw}x{dh}"
            );
            offset += len;
        }
        assert_eq!(offset, expected.len());
    }
}
