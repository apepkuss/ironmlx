//! Gemma4 image preprocessing.
//!
//! The model consumes regular channel-first pixels `[B, 3, H, W]`; patchify
//! happens inside `vision_tower.patch_embedder`.

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

    let plane = (h2 * w2) as usize;
    let mut chw = vec![0.0_f32; 3 * plane];

    let t0 = Instant::now();
    if orig_h == h2 && orig_w == w2 {
        for (i, p) in img.pixels().enumerate() {
            chw[i] = p.0[0] as f32 / 255.0;
            chw[plane + i] = p.0[1] as f32 / 255.0;
            chw[2 * plane + i] = p.0[2] as f32 / 255.0;
        }
    } else {
        let resized = image::imageops::resize(
            &img,
            w2 as u32,
            h2 as u32,
            image::imageops::FilterType::CatmullRom,
        );
        for (i, p) in resized.pixels().enumerate() {
            chw[i] = p.0[0] as f32 / 255.0;
            chw[plane + i] = p.0[1] as f32 / 255.0;
            chw[2 * plane + i] = p.0[2] as f32 / 255.0;
        }
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
    let pixel_values: Array = (chw.as_slice(), &[1_i32, 3, h2, w2][..])
        .try_into()
        .map_err(|e| anyhow!("Gemma4 image pixel_values array construction: {e}"))?;
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
}
