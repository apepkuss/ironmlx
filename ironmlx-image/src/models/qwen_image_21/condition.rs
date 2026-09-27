use anyhow::{anyhow, Context};
use image::imageops::FilterType;
use mlx::Array;

use ironmlx_lm::models::qwen3_5::image_processor::patchify;

use crate::Result;

const CONDITION_RESOLUTION: u32 = 1024;

/// Model-side representations of one Qwen Image 2.1 condition image.
///
/// The same aspect-preserving resize feeds both encoders. The Qwen3-VL copy
/// is alpha-composited over white and patchified as RGB, while the VAE copy
/// preserves and normalizes all four RGBA channels.
pub struct QwenImage21Condition {
    pub(crate) vision_pixels: Array,
    pub(crate) vae_pixels: Array,
    pub(crate) grid: (i32, i32, i32),
    pub(crate) latent_height: i32,
    pub(crate) latent_width: i32,
}

impl QwenImage21Condition {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let image = ironmlx_lm::core::image_input::load_from_memory_bounded(bytes)
            .context("decoding Qwen Image 2.1 condition image")?
            .to_rgba8();
        let (source_width, source_height) = image.dimensions();
        if source_width == 0 || source_height == 0 {
            return Err(anyhow!("Qwen Image 2.1 condition image is empty"));
        }
        let (width, height) = condition_dimensions(source_width, source_height)?;
        let resized = if (width, height) == (source_width, source_height) {
            image
        } else {
            image::imageops::resize(&image, width, height, FilterType::Lanczos3)
        };

        let (rgb_chw, rgba_chw) = normalized_channels(&resized)?;

        let height_i32 = i32::try_from(height)?;
        let width_i32 = i32::try_from(width)?;
        let (vision_pixels, grid_height, grid_width) = patchify(&rgb_chw, height_i32, width_i32)?;
        let vae_pixels: Array = (
            rgba_chw.as_slice(),
            &[1_i32, 4_i32, height_i32, width_i32][..],
        )
            .try_into()?;

        Ok(Self {
            vision_pixels,
            vae_pixels,
            grid: (1, grid_height, grid_width),
            latent_height: height_i32 / 16,
            latent_width: width_i32 / 16,
        })
    }

    pub fn resized_dimensions(&self) -> (u32, u32) {
        (
            (self.latent_width * 16) as u32,
            (self.latent_height * 16) as u32,
        )
    }
}

fn normalized_channels(image: &image::RgbaImage) -> Result<(Vec<f32>, Vec<f32>)> {
    let pixels = usize::try_from(u64::from(image.width()) * u64::from(image.height()))?;
    let mut rgba_chw = vec![0.0_f32; 4 * pixels];
    let mut rgb_chw = vec![0.0_f32; 3 * pixels];
    for (index, pixel) in image.pixels().enumerate() {
        let [red, green, blue, alpha] = pixel.0;
        let alpha_f = alpha as f32 / 255.0;
        let composite = [
            red as f32 * alpha_f + 255.0 * (1.0 - alpha_f),
            green as f32 * alpha_f + 255.0 * (1.0 - alpha_f),
            blue as f32 * alpha_f + 255.0 * (1.0 - alpha_f),
        ];
        for channel in 0..3 {
            rgb_chw[channel * pixels + index] = composite[channel] / 127.5 - 1.0;
        }
        for (channel, value) in [red, green, blue, alpha].into_iter().enumerate() {
            rgba_chw[channel * pixels + index] = value as f32 / 127.5 - 1.0;
        }
    }
    Ok((rgb_chw, rgba_chw))
}

fn condition_dimensions(width: u32, height: u32) -> Result<(u32, u32)> {
    let ratio = width as f64 / height as f64;
    if !ratio.is_finite() || !(1.0 / 200.0..=200.0).contains(&ratio) {
        return Err(anyhow!(
            "Qwen Image 2.1 condition aspect ratio must be within 1:200..200:1"
        ));
    }
    let target_area = f64::from(CONDITION_RESOLUTION * CONDITION_RESOLUTION);
    let resized_width = ((target_area * ratio).sqrt() / 32.0).round().max(1.0) * 32.0;
    let resized_height = ((target_area / ratio).sqrt() / 32.0).round().max(1.0) * 32.0;
    Ok((resized_width as u32, resized_height as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_dimensions_preserve_ratio_and_patch_alignment() {
        assert_eq!(condition_dimensions(1600, 900).unwrap(), (1376, 768));
        let (width, height) = condition_dimensions(900, 1600).unwrap();
        assert_eq!((width % 32, height % 32), (0, 0));
        assert!(((width as f64 / height as f64) - 9.0 / 16.0).abs() < 0.03);
    }

    #[test]
    fn condition_dimensions_reject_extreme_aspect_ratio() {
        assert!(condition_dimensions(1000, 1).is_err());
    }

    #[test]
    fn transparent_pixels_are_white_for_vision_but_preserved_for_vae() {
        let image = image::RgbaImage::from_raw(1, 1, vec![10, 20, 30, 0]).unwrap();
        let (vision, vae) = normalized_channels(&image).unwrap();
        assert_eq!(vision, vec![1.0, 1.0, 1.0]);
        assert!((vae[0] - (10.0 / 127.5 - 1.0)).abs() < 1e-6);
        assert!((vae[1] - (20.0 / 127.5 - 1.0)).abs() < 1e-6);
        assert!((vae[2] - (30.0 / 127.5 - 1.0)).abs() < 1e-6);
        assert_eq!(vae[3], -1.0);
    }
}
