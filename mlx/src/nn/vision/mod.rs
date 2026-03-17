pub mod patch_embed;
pub mod qwen35_vision;
pub mod spatial_merger;
pub mod vision_block;

use crate::array::Array;
use crate::error::Result;
use crate::stream::Stream;

/// Vision encoder trait — encodes pixel tensors to embeddings
pub trait VisionEncoder: Send + Sync {
    /// Encode pixel values to vision embeddings.
    /// pixel_values: preprocessed image/video tensor
    /// grid_thw: [(temporal, height_patches, width_patches)] per image
    fn encode(
        &self,
        pixel_values: &Array,
        grid_thw: &[(usize, usize, usize)],
        stream: &Stream,
    ) -> Result<Array>;

    /// Output embedding dimension
    fn output_dim(&self) -> usize;
}
