pub mod image_proc;
pub mod loader;
pub mod video;

use crate::array::Array;
use crate::error::Result;

/// Processed media ready for vision encoder
pub struct ProcessedMedia {
    /// Pixel values [B, C, H, W] float32
    pub pixel_values: Array,
    /// Grid dimensions [(T, H_patches, W_patches)] per image/video
    pub grid_thw: Vec<(usize, usize, usize)>,
}

/// Media item extracted from API request
pub enum MediaItem {
    Image(Vec<u8>),    // raw image bytes
    VideoPath(String), // path to video file
}

/// Media processor trait
pub trait MediaProcessor: Send + Sync {
    fn process_image(&self, data: &[u8]) -> Result<ProcessedMedia>;
    fn process_video(&self, path: &str) -> Result<ProcessedMedia>;
}
