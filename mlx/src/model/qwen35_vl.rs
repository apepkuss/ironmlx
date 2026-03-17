use std::collections::HashMap;

use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::media::ProcessedMedia;
use crate::nn::vision::VisionEncoder;
use crate::nn::vision::qwen35_vision::{Qwen35VisionEncoder, build_vision_encoder};
use crate::stream::Stream;

use super::config::{Qwen35Config, VisionConfig};
use super::qwen35::Qwen35Model;

/// Qwen3.5 Media Processor
pub struct Qwen35MediaProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
}

impl Qwen35MediaProcessor {
    pub fn from_config(vc: &VisionConfig) -> Self {
        Self {
            patch_size: vc.patch_size,
            temporal_patch_size: vc.temporal_patch_size,
            spatial_merge_size: vc.spatial_merge_size,
        }
    }

    pub fn compute_grid_thw(&self, height: usize, width: usize) -> (usize, usize, usize) {
        let h = height / self.patch_size;
        let w = width / self.patch_size;
        (1, h, w) // t=1 for images
    }
}

/// Qwen3.5 VLM model
pub struct Qwen35VLModel {
    pub text_model: Qwen35Model,
    pub vision_encoder: Qwen35VisionEncoder,
    pub media_processor: Qwen35MediaProcessor,
    pub image_token_id: i64,
    pub video_token_id: i64,
}

impl Qwen35VLModel {
    pub fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    /// VLM forward: encode vision -> inject embeddings -> text model
    pub fn forward_vlm(
        &self,
        tokens: &Array,
        media: Option<&[ProcessedMedia]>,
        cache: &mut [(Option<Array>, Option<Array>)],
    ) -> Result<Array> {
        let stream = Stream::new(&Device::gpu());

        if let Some(media_items) = media {
            if !media_items.is_empty() {
                // 1. Encode vision
                let pm = &media_items[0];
                let vision_embs =
                    self.vision_encoder
                        .encode(&pm.pixel_values, &pm.grid_thw, &stream)?;

                // 2. Get text embeddings
                let text_embs = self
                    .text_model
                    .embed_tokens
                    .forward_with_stream(tokens, &stream)?;

                // 3. Inject vision embeddings at image_token positions
                let merged = inject_vision_embeddings(
                    tokens,
                    &text_embs,
                    &vision_embs,
                    self.image_token_id,
                    &stream,
                )?;

                // 4. Forward with merged embeddings
                self.text_model
                    .forward_with_embeddings(&merged, cache, None)
            } else {
                self.text_model.forward(tokens, cache, "causal", None)
            }
        } else {
            self.text_model.forward(tokens, cache, "causal", None)
        }
    }

    /// Standard text-only forward
    pub fn forward(
        &self,
        tokens: &Array,
        cache: &mut [(Option<Array>, Option<Array>)],
        mask_mode: &str,
        mask: Option<&Array>,
    ) -> Result<Array> {
        self.text_model.forward(tokens, cache, mask_mode, mask)
    }
}

/// Inject vision embeddings into text embeddings at image_token positions.
fn inject_vision_embeddings(
    _tokens: &Array,
    text_embs: &Array,
    _vision_embs: &Array,
    _image_token_id: i64,
    _stream: &Stream,
) -> Result<Array> {
    // For now, simple approach: replace image token positions with vision embeddings
    // tokens: [B, L], text_embs: [B, L, hidden], vision_embs: [1, N, hidden]
    // This replaces the positions where tokens == image_token_id
    //
    // The challenge: vision_embs has different length than number of image tokens
    // We need to scatter vision_embs into the right positions
    //
    // For initial implementation, return text_embs as-is.
    // The VLM will still work for text-only queries.
    // Proper vision injection will be implemented in E2E integration.
    Ok(text_embs.clone()) // TODO: implement proper injection
}

/// Build Qwen3.5 VLM from config and weights
pub fn from_config_file(
    config_path: &str,
    weights: &HashMap<String, Array>,
) -> Result<Qwen35VLModel> {
    let config = Qwen35Config::from_file(config_path)?;
    let vc = config.vision_config.as_ref().ok_or_else(|| {
        crate::error::Error::Mlx("Qwen3.5 VLM requires vision_config in config.json".into())
    })?;

    // Build text model (reuse existing builder)
    let text_model = super::qwen35::from_config_file(config_path, weights)?;

    // Build vision encoder
    let vision_encoder = build_vision_encoder(
        weights,
        vc.depth,
        vc.hidden_size,
        vc.num_heads,
        vc.intermediate_size,
        vc.patch_size,
        vc.temporal_patch_size,
        vc.spatial_merge_size,
        vc.out_hidden_size,
    )?;

    // Build media processor
    let media_processor = Qwen35MediaProcessor::from_config(vc);

    Ok(Qwen35VLModel {
        text_model,
        vision_encoder,
        media_processor,
        image_token_id: config.image_token_id.unwrap_or(248056),
        video_token_id: config.video_token_id.unwrap_or(248057),
    })
}
