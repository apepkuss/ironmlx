use crate::array::Array;
use crate::error::Result;
use crate::nn::activations::gelu;
use crate::nn::linear::Linear;
use crate::nn::norm::LayerNorm;
use crate::ops;
use crate::stream::Stream;

/// SpatialMerger: merges 2x2 adjacent patches and projects to LM hidden dim.
pub struct SpatialMerger {
    pub norm: LayerNorm,
    pub fc1: Linear, // [intermediate, merge_size^2 * hidden] with bias
    pub fc2: Linear, // [out_hidden, intermediate] with bias
    pub spatial_merge_size: usize,
}

impl SpatialMerger {
    /// Forward: merge 2x2 patches then project.
    /// x: [B, H*W, hidden_size]
    /// grid_thw: [(t, h, w)] — h, w are BEFORE merge (in patches)
    pub fn forward(
        &self,
        x: &Array,
        grid_thw: &[(usize, usize, usize)],
        stream: &Stream,
    ) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0] as usize;
        let hidden = shape[2] as usize;
        let ms = self.spatial_merge_size;

        // Norm BEFORE merge (norm weight dim = hidden_size = 1024)
        let x = self.norm.forward_with_stream(x, stream)?;

        // grid_thw gives us the patch grid dimensions
        let (_, h, w) = grid_thw[0]; // patches_h, patches_w (before merge)

        // x: [B, H*W, hidden] → [B, H/ms, ms, W/ms, ms, hidden]
        let h_merged = h / ms;
        let w_merged = w / ms;

        let x = ops::reshape(
            &x,
            &[
                b as i32,
                h_merged as i32,
                ms as i32,
                w_merged as i32,
                ms as i32,
                hidden as i32,
            ],
            stream,
        )?;

        // Transpose to [B, H/ms, W/ms, ms, ms, hidden]
        let x = ops::transpose_axes(&x, &[0, 1, 3, 2, 4, 5], stream)?;

        // Reshape to [B, H/ms*W/ms, ms*ms*hidden]
        let merged_patches = h_merged * w_merged;
        let merged_dim = ms * ms * hidden;
        let x = ops::reshape(
            &x,
            &[b as i32, merged_patches as i32, merged_dim as i32],
            stream,
        )?;

        // fc1 → GELU → fc2
        let x = self.fc1.forward_with_stream(&x, stream)?;
        let x = gelu(&x, stream)?;
        self.fc2.forward_with_stream(&x, stream)
    }
}
