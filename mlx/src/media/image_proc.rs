use crate::array::Array;
use crate::error::{Error, Result};

/// ImageNet normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Decode raw bytes into an image::DynamicImage
pub fn decode_image(data: &[u8]) -> Result<image::DynamicImage> {
    image::load_from_memory(data).map_err(|e| Error::Mlx(format!("image decode failed: {}", e)))
}

/// Resize image so both dimensions are multiples of `align` (typically 32 = patch_size * spatial_merge_size).
/// Preserves aspect ratio by rounding to nearest multiple.
pub fn resize_to_aligned(
    img: &image::DynamicImage,
    align: usize,
    min_size: usize,
    max_size: usize,
) -> image::DynamicImage {
    use image::imageops::FilterType;
    let (w, h) = (img.width() as usize, img.height() as usize);

    // Scale to fit within [min_size, max_size] range
    let scale = if w < min_size || h < min_size {
        min_size as f64 / (w.min(h) as f64)
    } else if w > max_size || h > max_size {
        max_size as f64 / (w.max(h) as f64)
    } else {
        1.0
    };

    let new_w = ((w as f64 * scale) as usize).max(align);
    let new_h = ((h as f64 * scale) as usize).max(align);

    // Align to multiples of `align`
    let aligned_w = (new_w / align) * align;
    let aligned_h = (new_h / align) * align;

    img.resize_exact(aligned_w as u32, aligned_h as u32, FilterType::Lanczos3)
}

/// Convert image to normalized float32 MLX array [1, C, H, W].
/// Applies ImageNet mean/std normalization.
pub fn image_to_array(img: &image::DynamicImage, mean: &[f32; 3], std: &[f32; 3]) -> Result<Array> {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.as_raw();

    // Convert to float32 and normalize: (pixel/255 - mean) / std
    // Layout: [H, W, 3] -> need [1, 3, H, W]
    let mut data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            for c in 0..3 {
                let pixel = raw[idx + c] as f32 / 255.0;
                let normalized = (pixel - mean[c]) / std[c];
                // CHW layout: data[c * h * w + y * w + x]
                data[c * h * w + y * w + x] = normalized;
            }
        }
    }

    // Create MLX array [1, 3, H, W]
    let flat = Array::from_slice_f32_shape(&data, &[1, 3, h as i32, w as i32]);
    Ok(flat)
}

/// Full pipeline: decode + resize + normalize
pub fn process_image_bytes(
    data: &[u8],
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<(Array, usize, usize)> {
    let img = decode_image(data)?;
    let align = patch_size * spatial_merge_size;
    let resized = resize_to_aligned(&img, align, align, 1024);
    let (w, h) = (resized.width() as usize, resized.height() as usize);
    let arr = image_to_array(&resized, &IMAGENET_MEAN, &IMAGENET_STD)?;
    Ok((arr, h, w))
}
