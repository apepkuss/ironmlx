//! Gemma4 image preprocessing.
//!
//! Tower-based Gemma4 consumes regular channel-first pixels `[B, 3, H, W]`;
//! patchify happens inside `vision_tower.patch_embedder`. Gemma4 Unified is
//! encoder-free and consumes padded merged image patch rows directly.

use anyhow::{anyhow, Result};
use mlx::Array;
use std::time::Instant;

use super::config::Gemma4VisionConfig;

#[derive(Debug)]
pub struct ProcessedImage {
    pub pixel_values: Array,
    pub grid_h: i32,
    pub grid_w: i32,
    pub soft_tokens: usize,
}

pub fn resize_target(height: i32, width: i32, cfg: &Gemma4VisionConfig) -> Result<(i32, i32)> {
    if height <= 0 || width <= 0 {
        return Err(anyhow!(
            "Gemma4 image resize: height and width must be positive, got {height}x{width}"
        ));
    }
    let max_patches = cfg.max_patches();
    let target_px = max_patches * cfg.patch_size * cfg.patch_size;
    let factor = (target_px as f64 / (height as f64 * width as f64)).sqrt();
    let side_mult = cfg.pooling_kernel_size * cfg.patch_size;

    let mut target_h = ((factor * height as f64 / side_mult as f64).floor() as i32) * side_mult;
    let mut target_w = ((factor * width as f64 / side_mult as f64).floor() as i32) * side_mult;

    if target_h == 0 && target_w == 0 {
        return Err(anyhow!("Gemma4 image resize would produce 0x0 output"));
    }

    let max_side = (max_patches / (cfg.pooling_kernel_size * cfg.pooling_kernel_size)) * side_mult;
    if target_h == 0 {
        target_h = side_mult;
        target_w = (((width as f64 / height as f64).floor() as i32) * side_mult)
            .max(side_mult)
            .min(max_side);
    } else if target_w == 0 {
        target_w = side_mult;
        target_h = (((height as f64 / width as f64).floor() as i32) * side_mult)
            .max(side_mult)
            .min(max_side);
    }

    Ok((target_h, target_w))
}

pub fn preprocess(img_bytes: &[u8], cfg: &Gemma4VisionConfig) -> Result<ProcessedImage> {
    let profile = std::env::var_os("IRONMLX_GEMMA4_VL_PROFILE").is_some();
    let total_t0 = Instant::now();
    let t0 = Instant::now();
    let img = image::load_from_memory(img_bytes)
        .map_err(|e| anyhow!("decode image: {e}"))?
        .to_rgb8();
    if profile {
        tracing::info!(
            "[gemma4-vl-profile] image_decode_ms={:.3}",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    let (orig_w, orig_h) = (img.width() as i32, img.height() as i32);
    let (h2, w2) = resize_target(orig_h, orig_w, cfg)?;

    let t0 = Instant::now();
    let resized = resize_rgb(img.as_raw(), orig_w, orig_h, w2, h2);
    let plane = (h2 * w2) as usize;
    let mut chw = vec![0.0_f32; 3 * plane];
    for i in 0..plane {
        let sp = i * 3;
        chw[i] = resized[sp] as f32 / 255.0;
        chw[plane + i] = resized[sp + 1] as f32 / 255.0;
        chw[2 * plane + i] = resized[sp + 2] as f32 / 255.0;
    }
    if profile {
        tracing::info!(
            "[gemma4-vl-profile] image_resize_chw_ms={:.3} orig={}x{} resized={}x{}",
            t0.elapsed().as_secs_f64() * 1000.0,
            orig_h,
            orig_w,
            h2,
            w2
        );
    }

    let grid_h = h2 / cfg.patch_size;
    let grid_w = w2 / cfg.patch_size;
    if grid_h % cfg.pooling_kernel_size != 0 || grid_w % cfg.pooling_kernel_size != 0 {
        return Err(anyhow!(
            "Gemma4 image preprocess: resized patch grid {grid_h}x{grid_w} is not divisible by pooling kernel {}",
            cfg.pooling_kernel_size
        ));
    }
    let soft_tokens =
        (grid_h * grid_w / (cfg.pooling_kernel_size * cfg.pooling_kernel_size)) as usize;
    if soft_tokens > cfg.default_output_length as usize {
        return Err(anyhow!(
            "Gemma4 image preprocess: soft token count {soft_tokens} exceeds max {}",
            cfg.default_output_length
        ));
    }

    let t0 = Instant::now();
    let pixel_values: Array = if cfg.is_unified() {
        unified_pixel_values(&chw, h2, w2, cfg)?
    } else {
        (chw.as_slice(), &[1_i32, 3, h2, w2][..])
            .try_into()
            .map_err(|e| anyhow!("Gemma4 image pixel_values array construction: {e}"))?
    };
    if profile {
        tracing::info!(
            "[gemma4-vl-profile] image_array_ms={:.3} image_preprocess_total_ms={:.3} soft_tokens={}",
            t0.elapsed().as_secs_f64() * 1000.0,
            total_t0.elapsed().as_secs_f64() * 1000.0,
            soft_tokens
        );
    }
    Ok(ProcessedImage {
        pixel_values,
        grid_h,
        grid_w,
        soft_tokens,
    })
}

const PRECISION_BITS: i32 = 32 - 8 - 2;
const BICUBIC_SUPPORT: f64 = 2.0;

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

fn clip8(acc: i64) -> u8 {
    let v = acc >> PRECISION_BITS;
    if v <= 0 {
        0
    } else if v >= 255 {
        255
    } else {
        v as u8
    }
}

fn pil_round(x: f64) -> i64 {
    if x < 0.0 {
        (x - 0.5).ceil() as i64
    } else {
        (x + 0.5).floor() as i64
    }
}

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
        let base = xx as usize * ksize;
        for (x, &w) in prekk.iter().take(xmax as usize).enumerate() {
            kk[base + x] = pil_round(w * (1_i64 << PRECISION_BITS) as f64) as i32;
        }
        bounds[xx as usize * 2] = xmin;
        bounds[xx as usize * 2 + 1] = xmax;
    }
    (kk, bounds, ksize)
}

fn resample_horizontal(src: &[u8], in_w: usize, h: usize, out_w: i32) -> Vec<u8> {
    let (kk, bounds, ksize) = precompute_coeffs(in_w as i32, out_w);
    let out_w_usize = out_w as usize;
    let mut dst = vec![0_u8; out_w_usize * h * 3];
    let init = 1_i64 << (PRECISION_BITS - 1);
    for yy in 0..h {
        let src_row = yy * in_w * 3;
        let dst_row = yy * out_w_usize * 3;
        for xx in 0..out_w_usize {
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

fn resample_vertical(src: &[u8], w: usize, in_h: usize, out_h: i32) -> Vec<u8> {
    let (kk, bounds, ksize) = precompute_coeffs(in_h as i32, out_h);
    let out_h_usize = out_h as usize;
    let mut dst = vec![0_u8; w * out_h_usize * 3];
    let init = 1_i64 << (PRECISION_BITS - 1);
    for yy in 0..out_h_usize {
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

fn pil_bicubic_resize(src: &[u8], in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let horiz = resample_horizontal(src, in_w, in_h, out_w as i32);
    resample_vertical(&horiz, out_w, in_h, out_h as i32)
}

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

fn unified_pixel_values(
    chw: &[f32],
    height: i32,
    width: i32,
    cfg: &Gemma4VisionConfig,
) -> Result<Array> {
    let p = cfg.patch_size;
    let k = cfg.pooling_kernel_size;
    let model_patch = cfg.model_patch_size();
    let patch_dim = model_patch
        .checked_mul(model_patch)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| anyhow!("Gemma4 unified image patch dimension overflow"))?;
    let grid_h = height / p;
    let grid_w = width / p;
    let soft_h = grid_h / k;
    let soft_w = grid_w / k;
    let soft_tokens = soft_h * soft_w;
    let max_soft_tokens = cfg.default_output_length;
    if soft_tokens <= 0 || soft_tokens > max_soft_tokens {
        return Err(anyhow!(
            "Gemma4 unified image preprocess: soft token count {soft_tokens} out of range 1..={max_soft_tokens}"
        ));
    }

    let total = max_soft_tokens
        .checked_mul(patch_dim)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| anyhow!("Gemma4 unified image padded tensor size overflow"))?;
    let mut merged = vec![0.0_f32; total];
    let plane = usize::try_from(height * width)
        .map_err(|_| anyhow!("Gemma4 unified image plane size overflow"))?;
    let width_usize =
        usize::try_from(width).map_err(|_| anyhow!("Gemma4 unified image width overflow"))?;
    let p_usize =
        usize::try_from(p).map_err(|_| anyhow!("Gemma4 unified image patch_size overflow"))?;
    let k_usize = usize::try_from(k)
        .map_err(|_| anyhow!("Gemma4 unified image pooling_kernel_size overflow"))?;
    let soft_h_usize =
        usize::try_from(soft_h).map_err(|_| anyhow!("Gemma4 unified soft_h overflow"))?;
    let soft_w_usize =
        usize::try_from(soft_w).map_err(|_| anyhow!("Gemma4 unified soft_w overflow"))?;
    let patch_dim_usize =
        usize::try_from(patch_dim).map_err(|_| anyhow!("Gemma4 unified patch_dim overflow"))?;

    for by in 0..soft_h_usize {
        for bx in 0..soft_w_usize {
            let token = by * soft_w_usize + bx;
            let base = token * patch_dim_usize;
            let mut out = base;
            for ky in 0..k_usize {
                for py in 0..p_usize {
                    let y = (by * k_usize + ky) * p_usize + py;
                    for kx in 0..k_usize {
                        for px in 0..p_usize {
                            let x = (bx * k_usize + kx) * p_usize + px;
                            let spatial = y * width_usize + x;
                            for c in 0..3_usize {
                                merged[out] = chw[c * plane + spatial];
                                out += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    (merged.as_slice(), &[1_i32, max_soft_tokens, patch_dim][..])
        .try_into()
        .map_err(|e| anyhow!("Gemma4 unified image pixel_values array construction: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Gemma4VisionConfig {
        serde_json::from_value(serde_json::json!({
            "model_type": "gemma4_vision",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 16,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "head_dim": 64,
            "patch_size": 16,
            "default_output_length": 280,
            "pooling_kernel_size": 3,
            "position_embedding_size": 10240
        }))
        .unwrap()
    }

    fn unified_cfg() -> Gemma4VisionConfig {
        unified_cfg_with_soft_tokens(280)
    }

    fn unified_cfg_with_soft_tokens(num_soft_tokens: i32) -> Gemma4VisionConfig {
        serde_json::from_value(serde_json::json!({
            "model_type": "gemma4_unified_vision",
            "hidden_size": 3840,
            "intermediate_size": 1,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "mm_embed_dim": 3840,
            "mm_posemb_size": 1120,
            "model_patch_size": 48,
            "num_soft_tokens": num_soft_tokens,
            "output_proj_dims": 3840,
            "patch_size": 16,
            "pooling_kernel_size": 3
        }))
        .unwrap()
    }

    fn png_bytes(width: u32, height: u32) -> Vec<u8> {
        let mut img = image::RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
        }
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        bytes
    }

    #[test]
    fn resize_target_is_patch_pool_aligned() {
        let cfg = cfg();
        let (h, w) = resize_target(224, 224, &cfg).expect("resize");
        let side_mult = cfg.patch_size * cfg.pooling_kernel_size;
        assert_eq!(h % side_mult, 0);
        assert_eq!(w % side_mult, 0);
        let soft = (h / cfg.patch_size) * (w / cfg.patch_size)
            / (cfg.pooling_kernel_size * cfg.pooling_kernel_size);
        assert!(soft <= cfg.default_output_length);
    }

    #[test]
    fn unified_preprocess_returns_padded_merged_patch_rows() {
        let cfg = unified_cfg();
        let image = png_bytes(96, 48);
        let processed = preprocess(&image, &cfg).expect("preprocess unified image");

        assert_eq!(processed.pixel_values.dtype(), mlx::Dtype::Float32);
        assert_eq!(
            processed.pixel_values.shape().as_slice(),
            &[1, cfg.default_output_length, 48 * 48 * 3]
        );
        assert!(processed.soft_tokens > 0);
        assert!(processed.soft_tokens <= cfg.default_output_length as usize);
        assert_eq!(processed.grid_h % cfg.pooling_kernel_size, 0);
        assert_eq!(processed.grid_w % cfg.pooling_kernel_size, 0);
    }

    #[test]
    fn unified_preprocess_matches_model_patch_order_without_resize() {
        let cfg = unified_cfg_with_soft_tokens(2);
        let image = png_bytes(96, 48);
        let processed = preprocess(&image, &cfg).expect("preprocess unified image");

        assert_eq!(processed.soft_tokens, 2);
        assert_eq!((processed.grid_h, processed.grid_w), (3, 6));
        assert_eq!(
            processed.pixel_values.shape().as_slice(),
            &[1, 2, 48 * 48 * 3]
        );

        let values: Vec<f32> = processed.pixel_values.to_vec().unwrap();
        let patch_dim = (48 * 48 * 3) as usize;
        let close = |actual: f32, expected_u8: u8| {
            let expected = expected_u8 as f32 / 255.0;
            assert!(
                (actual - expected).abs() < 1e-6,
                "actual={actual} expected={expected}"
            );
        };

        close(values[0], 0);
        close(values[1], 0);
        close(values[2], 0);
        close(values[3], 1);
        close(values[4], 0);
        close(values[5], 1);
        close(values[48], 16);
        close(values[49], 0);
        close(values[50], 16);
        close(values[2304], 0);
        close(values[2305], 16);
        close(values[2306], 16);
        close(values[patch_dim], 48);
        close(values[patch_dim + 1], 0);
        close(values[patch_dim + 2], 48);
    }

    #[test]
    fn pil_bicubic_resize_identity_when_same_size() {
        let src = vec![0_u8, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32];
        let out = resize_rgb(&src, 2, 2, 2, 2);
        assert_eq!(out, src);
    }
}
