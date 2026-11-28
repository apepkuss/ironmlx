//! Inkling preserves image resolution, pads to a patch/merge block, and composites alpha.
use super::config::VisionConfig;
use anyhow::{ensure, Result};
use image::GenericImageView;
use mlx::Array;

pub const IMAGE_TOKEN: &str = "<|unused_200054|>";

pub fn preprocess(
    bytes: &[u8],
    config: &VisionConfig,
    max_pixels: usize,
) -> Result<(Array, i32, i32)> {
    let image = image::load_from_memory(bytes)?;
    let (w, h) = image.dimensions();
    ensure!(
        w > 0 && h > 0 && (w as usize).saturating_mul(h as usize) <= max_pixels,
        "Inkling 2 image exceeds the configured pixel limit ({max_pixels})"
    );
    let block = (config.native_patch_size * config.adapter_merge_size) as u32;
    ensure!(block > 0, "invalid Inkling 2 image block size");
    let width = w.div_ceil(block) * block;
    let height = h.div_ceil(block) * block;
    let mut values = vec![(-1.0_f32 / 255.0 - 0.5) / 0.5; (3 * width * height) as usize];
    let rgba = image.to_rgba8();
    for y in 0..h {
        for x in 0..w {
            let pixel = rgba.get_pixel(x, y);
            let alpha = pixel[3] as u32;
            let background = if (x / 8 + y / 8) % 2 == 0 { 204 } else { 153 };
            for c in 0..3 {
                let value = (pixel[c] as u32 * alpha + background * (255 - alpha) + 127) / 255;
                values[c * (width * height) as usize + (y * width + x) as usize] =
                    (value as f32 * (1.0 / 255.0) - 0.5) / 0.5;
            }
        }
    }
    let pixels: Array = (values.as_slice(), &[3, height as i32, width as i32][..]).try_into()?;
    Ok((
        pixels,
        height as i32 / config.native_patch_size,
        width as i32 / config.native_patch_size,
    ))
}
