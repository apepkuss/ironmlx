use crate::array::Array;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;

/// PatchEmbed converts images to patch tokens via Conv3D (im2col + matmul).
/// Weight shape: [out_channels, temporal_patch_size, patch_size, patch_size, in_channels]
pub struct PatchEmbed {
    pub weight: Array,
    pub bias: Array,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
}

impl PatchEmbed {
    pub fn new(weight: Array, bias: Array, patch_size: usize, temporal_patch_size: usize) -> Self {
        Self {
            weight,
            bias,
            patch_size,
            temporal_patch_size,
        }
    }

    /// Forward pass: pixel_values [B*T, C, H, W] -> [B, num_patches, hidden_size]
    /// For images: T=temporal_patch_size (frames are stacked), B=num_images
    pub fn forward(&self, pixel_values: &Array, stream: &Stream) -> Result<Array> {
        let shape = pixel_values.shape();
        // pixel_values: [B*T, C, H, W] where T=temporal_patch_size
        let bt = shape[0] as usize;
        let c = shape[1] as usize; // 3
        let h = shape[2] as usize;
        let w = shape[3] as usize;
        let t = self.temporal_patch_size;
        let b = bt / t;
        let ps = self.patch_size;
        let out_ch = self.weight.shape()[0] as usize;

        let patches_h = h / ps;
        let patches_w = w / ps;
        let num_patches = patches_h * patches_w;
        let patch_dim = ps * ps * c; // 16*16*3 = 768

        // Weight: [out_ch, t, ps, ps, c] → reshape to [out_ch, t * patch_dim]
        let w_reshaped = ops::reshape(
            &self.weight,
            &[out_ch as i32, (t * patch_dim) as i32],
            stream,
        )?;

        // Reshape pixel_values from [B*T, C, H, W] to [B, T, C, H, W]
        let pv = ops::reshape(
            pixel_values,
            &[b as i32, t as i32, c as i32, h as i32, w as i32],
            stream,
        )?;

        // Transpose to [B, T, H, W, C] for easier patch extraction
        let pv = ops::transpose_axes(&pv, &[0, 1, 3, 4, 2], stream)?;

        // Reshape to [B, T, patches_h, ps, patches_w, ps, C]
        let pv = ops::reshape(
            &pv,
            &[
                b as i32,
                t as i32,
                patches_h as i32,
                ps as i32,
                patches_w as i32,
                ps as i32,
                c as i32,
            ],
            stream,
        )?;

        // Transpose to [B, patches_h, patches_w, T, ps, ps, C]
        let pv = ops::transpose_axes(&pv, &[0, 2, 4, 1, 3, 5, 6], stream)?;

        // Reshape to [B, num_patches, T*ps*ps*C]
        let pv = ops::reshape(
            &pv,
            &[b as i32, num_patches as i32, (t * patch_dim) as i32],
            stream,
        )?;

        // Matmul: [B, num_patches, T*patch_dim] @ [T*patch_dim, out_ch] → [B, num_patches, out_ch]
        let w_t = ops::transpose_axes(&w_reshaped, &[1, 0], stream)?;
        let output = ops::matmul(&pv, &w_t, stream)?;

        // Add bias [out_ch] broadcast to [B, num_patches, out_ch]
        let bias_reshaped = ops::reshape(&self.bias, &[1, 1, out_ch as i32], stream)?;
        ops::add(&output, &bias_reshaped, stream)
    }
}
