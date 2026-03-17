use std::collections::HashMap;

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::nn::linear::Linear;
use crate::nn::norm::LayerNorm;
use crate::ops;
use crate::stream::Stream;

use super::VisionEncoder;
use super::patch_embed::PatchEmbed;
use super::spatial_merger::SpatialMerger;
use super::vision_block::*;

pub struct Qwen35VisionEncoder {
    pub patch_embed: PatchEmbed,
    pub pos_embed: Array,         // [num_position_embeddings, hidden_size]
    pub blocks: Vec<VisionBlock>, // depth layers (24 for Qwen3.5)
    pub merger: SpatialMerger,
    pub hidden_size: usize,
    pub num_heads: usize,
}

impl VisionEncoder for Qwen35VisionEncoder {
    fn encode(
        &self,
        pixel_values: &Array,
        grid_thw: &[(usize, usize, usize)],
        stream: &Stream,
    ) -> Result<Array> {
        // Ensure pixel_values has temporal_patch_size frames for PatchEmbed.
        // For images (1 frame), repeat to match temporal_patch_size.
        let t = self.patch_embed.temporal_patch_size;
        let pv_shape = pixel_values.shape();
        let pixel_values = if pv_shape[0] as usize == 1 && t > 1 {
            // [1, C, H, W] → repeat along axis 0 to [T, C, H, W]
            let refs: Vec<&Array> = (0..t).map(|_| pixel_values).collect();
            let va = crate::vector::VectorArray::from_arrays(&refs);
            ops::concatenate(&va, 0, stream)?
        } else {
            pixel_values.clone()
        };

        // 1. Patch embedding: pixel_values → [B, num_patches, hidden_size]
        let mut h = self.patch_embed.forward(&pixel_values, stream)?;

        // 2. Add position embeddings via gather
        let pos = self.compute_position_embeddings(grid_thw, stream)?;
        h = ops::add(&h, &pos, stream)?;

        // 3. Vision transformer blocks (global, non-causal attention)
        for block in &self.blocks {
            h = block.forward(&h, None, stream)?;
        }

        // 4. Spatial merger: 2x2 patch merge + projection to LM hidden dim
        self.merger.forward(&h, grid_thw, stream)
    }

    fn output_dim(&self) -> usize {
        // fc2 weight shape is [out_hidden_size, intermediate_size],
        // so shape()[0] gives out_hidden_size (2560 for Qwen3.5)
        self.merger.fc2.weight.shape()[0] as usize
    }
}

impl Qwen35VisionEncoder {
    /// Compute position embeddings by gathering from `pos_embed` using sequential
    /// patch indices derived from `grid_thw`.
    fn compute_position_embeddings(
        &self,
        grid_thw: &[(usize, usize, usize)],
        stream: &Stream,
    ) -> Result<Array> {
        // Build sequential indices for all patches across the batch
        let mut total_patches: usize = 0;
        for &(t, h, w) in grid_thw {
            total_patches += t * h * w;
        }

        // arange(0, total_patches) as i32 indices
        let indices = ops::arange(0.0, total_patches as f64, 1.0, Dtype::Int32, stream)?;

        // Gather: pos_embed[indices] → [total_patches, hidden_size]
        let pos = ops::take_axis(&self.pos_embed, &indices, 0, stream)?;

        // Reshape to [1, total_patches, hidden_size] for broadcasting with h
        let hidden = self.hidden_size as i32;
        let pos = ops::reshape(&pos, &[1, total_patches as i32, hidden], stream)?;

        Ok(pos)
    }
}

/// Build a `Qwen35VisionEncoder` from a flat weights map and config values.
#[allow(clippy::too_many_arguments)]
pub fn build_vision_encoder(
    weights: &HashMap<String, Array>,
    depth: usize,               // 24
    hidden_size: usize,         // 1024
    num_heads: usize,           // 16
    _intermediate_size: usize,  // 4096
    patch_size: usize,          // 16
    temporal_patch_size: usize, // 2
    spatial_merge_size: usize,  // 2
    _out_hidden_size: usize,    // 2560
) -> Result<Qwen35VisionEncoder> {
    let prefix = "vision_tower";

    // Helper: fetch a single weight tensor by key
    let w = |name: &str| -> Result<Array> {
        weights
            .get(&format!("{}.{}", prefix, name))
            .cloned()
            .ok_or_else(|| Error::Mlx(format!("missing vision weight: {}.{}", prefix, name)))
    };

    // Helper: build a Linear layer from weight + optional bias
    let linear = |name: &str| -> Result<Linear> {
        let weight = w(&format!("{}.weight", name))?;
        let bias_key = format!("{}.{}.bias", prefix, name);
        let bias = weights.get(&bias_key).cloned();
        Ok(Linear::new(weight, bias))
    };

    // Helper: build a LayerNorm from weight + optional bias
    let layer_norm = |name: &str| -> Result<LayerNorm> {
        let weight_key = format!("{}.{}.weight", prefix, name);
        let bias_key = format!("{}.{}.bias", prefix, name);
        let weight = weights.get(&weight_key).cloned();
        let bias = weights.get(&bias_key).cloned();
        Ok(LayerNorm::new(weight, bias, 1e-6))
    };

    // --- PatchEmbed ---
    let patch_embed = PatchEmbed::new(
        w("patch_embed.proj.weight")?,
        w("patch_embed.proj.bias")?,
        patch_size,
        temporal_patch_size,
    );

    // --- Position embeddings ---
    let pos_embed = w("pos_embed.weight")?;

    // --- Vision transformer blocks ---
    let mut blocks = Vec::with_capacity(depth);
    for i in 0..depth {
        let bp = format!("blocks.{}", i);

        let attn = VisionAttention {
            qkv: linear(&format!("{}.attn.qkv", bp))?,
            proj: linear(&format!("{}.attn.proj", bp))?,
            num_heads,
            head_dim: hidden_size / num_heads,
        };

        let mlp = VisionMLP {
            fc1: linear(&format!("{}.mlp.linear_fc1", bp))?,
            fc2: linear(&format!("{}.mlp.linear_fc2", bp))?,
        };

        blocks.push(VisionBlock {
            norm1: layer_norm(&format!("{}.norm1", bp))?,
            attn,
            norm2: layer_norm(&format!("{}.norm2", bp))?,
            mlp,
        });
    }

    // --- Spatial merger ---
    let merger = SpatialMerger {
        norm: layer_norm("merger.norm")?,
        fc1: linear("merger.linear_fc1")?,
        fc2: linear("merger.linear_fc2")?,
        spatial_merge_size,
    };

    Ok(Qwen35VisionEncoder {
        patch_embed,
        pos_embed,
        blocks,
        merger,
        hidden_size,
        num_heads,
    })
}
