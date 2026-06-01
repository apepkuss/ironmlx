//! MiniCPM-V-4.6 vision stack — SigLIP encoder + VitMerger + Merger.
//!
//! [`MiniCpmV46Vision`] orchestrates the full single-image pipeline:
//! embeddings → encoder (with mid-encoder VitMerger insertion after
//! `insert_layer_id`) → post_layernorm → final Merger.

pub mod embeddings;
pub mod encoder;
pub mod merger;

use anyhow::Result;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::models::minicpmv4_6::config::MiniCpmV46VisionConfig;
use crate::nn::LayerNorm;

use embeddings::SiglipEmbeddings;
use encoder::SiglipEncoder;
use merger::{Merger, VitMerger};

/// Full vision pipeline for MiniCPM-V-4.6.
///
/// Encapsulates:
/// - SigLIP patch embeddings
/// - SigLIP encoder (27 layers, with mid-encoder VitMerger after layer 6)
/// - `vision_tower.post_layernorm` applied after all encoder layers
/// - Final `Merger` projecting to LM hidden size
///
/// Semantics match mlx-vlm `Model.get_vision_embedding` exactly:
///   1. Run encoder layer `i`.
///   2. After layer whose index == `insert_layer_id`, apply VitMerger.
///   3. After all 27 layers, apply post_layernorm.
///   4. Apply final Merger → `[N, lm_hidden]` output.
pub struct MiniCpmV46Vision {
    embeddings: SiglipEmbeddings,
    encoder: SiglipEncoder,
    vit_merger: VitMerger,
    post_ln: LayerNorm,
    merger: Merger,
    insert_layer_id: i32,
}

impl MiniCpmV46Vision {
    /// Load all vision sub-modules from a checkpoint opened via
    /// [`Loader::open_multimodal`].
    pub fn from_loader(loader: &Loader, cfg: &MiniCpmV46VisionConfig) -> Result<Self> {
        Ok(Self {
            embeddings: SiglipEmbeddings::from_loader(loader, cfg)?,
            encoder: SiglipEncoder::from_loader(loader, cfg)?,
            vit_merger: VitMerger::from_loader(loader, cfg)?,
            post_ln: LayerNorm::from_loader(
                loader,
                "vision_tower.post_layernorm",
                cfg.layer_norm_eps,
            )?,
            merger: Merger::from_loader(loader, cfg)?,
            insert_layer_id: cfg.insert_layer_id,
        })
    }

    /// Single image: `pixel_values` patch-packed, `(grid_h, grid_w)`.
    ///
    /// Returns merged vision embeddings `[N, lm_hidden=1024]`.
    ///
    /// Loop semantics (matching mlx-vlm `get_vision_embedding`):
    ///   - Layer `i` runs, then the VitMerger fires immediately AFTER the
    ///     layer whose index equals `insert_layer_id` (default: 6).
    ///   - Remaining encoder layers continue on the spatially-downsampled sequence.
    ///   - post_layernorm is applied after ALL encoder layers complete.
    ///   - The final Merger projects from vision-hidden to LM-hidden.
    pub fn compute_vision_embeds(
        &self,
        pixel_values: &Array,
        grid_h: i32,
        grid_w: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let t = target.into();

        // Patch embedding: [1, grid_h*grid_w, 1152]
        let mut h = self
            .embeddings
            .forward_on(pixel_values, grid_h, grid_w, t)?;
        let (mut gh, mut gw) = (grid_h, grid_w);

        for (i, layer) in self.encoder.layers.iter().enumerate() {
            h = layer.forward_on(&h, t)?;
            if i as i32 == self.insert_layer_id {
                // VitMerger expects [grid_h*grid_w, hidden]; squeeze the batch dim.
                let hidden_dim = h.shape().as_slice()[2];
                let row = h.reshape_on(&[gh * gw, hidden_dim][..], t)?;
                let (merged, nh, nw) = self.vit_merger.forward_on(&row, gh, gw, t)?;
                gh = nh;
                gw = nw;
                // Restore batch dim: [1, merged_h*merged_w, hidden]
                let merged_hidden = merged.shape().as_slice()[1];
                h = merged.reshape_on(&[1, gh * gw, merged_hidden][..], t)?;
            }
        }

        // post_layernorm after all encoder layers.
        let h = self.post_ln.forward_on(&h, t)?;

        // Final Merger: squeeze batch, apply Merger, return [N, lm_hidden].
        let hidden_dim = h.shape().as_slice()[2];
        let row = h.reshape_on(&[gh * gw, hidden_dim][..], t)?;
        let (merged, _, _) = self.merger.forward_on(&row, gh, gw, t)?;
        Ok(merged)
    }
}
