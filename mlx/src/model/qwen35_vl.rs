use std::collections::HashMap;

use crate::array::Array;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::media::ProcessedMedia;
use crate::nn::vision::VisionEncoder;
use crate::nn::vision::qwen35_vision::{Qwen35VisionEncoder, build_vision_encoder};
use crate::ops;
use crate::stream::Stream;
use crate::vector::VectorArray;

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
///
/// tokens: [1, L] — token IDs with image placeholders
/// text_embs: [1, L, hidden] — text embeddings from embed_tokens
/// vision_embs: [1, N, hidden] — vision embeddings from vision encoder
///
/// Strategy: read tokens to CPU, find image_token positions, then build
/// merged embedding by slicing text segments and vision segments alternately.
fn inject_vision_embeddings(
    tokens: &Array,
    text_embs: &Array,
    vision_embs: &Array,
    image_token_id: i64,
    stream: &Stream,
) -> Result<Array> {
    // Read tokens to CPU to find image positions
    tokens.eval()?;
    let token_vec = tokens.to_vec_i32()?;
    let image_id = image_token_id as i32;

    // Find contiguous runs of image tokens
    // Example: [BOS, t1, t2, IMG, IMG, IMG, t3, t4] → text[0:3], vision[0:3], text[6:8]
    let seq_len = token_vec.len();
    let mut segments: Vec<Segment> = Vec::new();
    let mut i = 0;
    let mut vision_offset = 0usize;

    while i < seq_len {
        if token_vec[i] == image_id {
            // Count consecutive image tokens
            let start = i;
            while i < seq_len && token_vec[i] == image_id {
                i += 1;
            }
            let count = i - start;
            segments.push(Segment::Vision {
                offset: vision_offset,
                count,
            });
            vision_offset += count;
        } else {
            let start = i;
            while i < seq_len && token_vec[i] != image_id {
                i += 1;
            }
            segments.push(Segment::Text { start, end: i });
        }
    }

    // Check vision embedding count matches
    let vision_embs_shape = vision_embs.shape();
    let total_vision = vision_embs_shape[1] as usize; // [1, N, hidden]
    if vision_offset != total_vision {
        return Err(Error::Mlx(format!(
            "vision embedding count mismatch: {} image tokens but {} vision patches",
            vision_offset, total_vision
        )));
    }

    // Build merged embedding by slicing and concatenating
    let hidden = text_embs.shape()[2];
    let mut parts: Vec<Array> = Vec::new();

    for seg in &segments {
        match seg {
            Segment::Text { start, end } => {
                // Slice text_embs[0, start:end, :]
                let part = ops::slice(
                    text_embs,
                    &[0, *start as i32, 0],
                    &[1, *end as i32, hidden],
                    &[1, 1, 1],
                    stream,
                )?;
                parts.push(part);
            }
            Segment::Vision { offset, count } => {
                // Slice vision_embs[0, offset:offset+count, :]
                let part = ops::slice(
                    vision_embs,
                    &[0, *offset as i32, 0],
                    &[1, (*offset + *count) as i32, hidden],
                    &[1, 1, 1],
                    stream,
                )?;
                parts.push(part);
            }
        }
    }

    // Concatenate along seq dim (axis=1)
    if parts.len() == 1 {
        return Ok(parts.into_iter().next().unwrap());
    }
    let refs: Vec<&Array> = parts.iter().collect();
    let va = VectorArray::from_arrays(&refs);
    ops::concatenate(&va, 1, stream)
}

enum Segment {
    Text { start: usize, end: usize },
    Vision { offset: usize, count: usize },
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
