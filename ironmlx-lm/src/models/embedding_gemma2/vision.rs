//! Checkpoint-specific image preprocessing and Gemma 4 vision encoder.
use std::path::Path;

use anyhow::{ensure, Context};
use mlx::Array;

use crate::{core::image_input, models::gemma4, Loader, Result};
use gemma4::vision::{MultimodalEmbedder, VisionModel};

pub(super) struct ImageEncoder {
    config: gemma4::Gemma4VisionConfig,
    tower: VisionModel,
    projection: MultimodalEmbedder,
}

impl ImageEncoder {
    pub fn load(loader: &Loader, path: &Path) -> Result<Option<Self>> {
        let raw = loader.config_raw_value();
        let Some(config) = validate_config(raw)? else {
            return Ok(None);
        };
        let processor: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("processor_config.json"))
                .context("reading embedding image processor")?,
        )?;
        let image = &processor["image_processor"];
        ensure!(
            image["image_processor_type"] == "Gemma4ImageProcessor"
                && image["do_convert_rgb"] == true
                && image["do_normalize"] == false
                && image["do_rescale"] == true
                && image["do_resize"] == true
                && image["resample"] == 3
                && image["patch_size"] == 16
                && image["pooling_kernel_size"] == 3
                && image["max_soft_tokens"] == 280
                && image["rescale_factor"]
                    .as_f64()
                    .is_some_and(|factor| (factor - 1.0 / 255.0).abs() < 1e-12),
            "unsupported EmbeddingGemma 2 image preprocessing configuration"
        );
        Ok(Some(Self {
            tower: VisionModel::from_loader_with_prefix(loader, config.clone(), "vision_tower")?
                .with_embedding_numerics(),
            projection: MultimodalEmbedder::from_loader(
                loader,
                "embed_vision",
                config.rms_norm_eps,
            )?,
            config,
        }))
    }

    pub fn token_count(&self, bytes: &[u8]) -> Result<usize> {
        ensure!(
            bytes.len() <= image_input::MAX_IMAGE_BYTES,
            "image input exceeds 10 MiB"
        );
        let (width, height, _) =
            image_input::inspect_image(bytes).context("invalid image input")?;
        ensure!(
            width <= image_input::MAX_IMAGE_SIDE
                && height <= image_input::MAX_IMAGE_SIDE
                && u64::from(width) * u64::from(height) <= image_input::MAX_IMAGE_PIXELS,
            "image input dimensions exceed supported bounds"
        );
        let (h, w) =
            gemma4::image_processor::resize_target(height as i32, width as i32, &self.config)?;
        Ok((h / 48 * (w / 48)) as usize)
    }

    pub fn encode(&self, bytes: &[u8]) -> Result<Array> {
        let processed = gemma4::image_processor::preprocess(bytes, &self.config)
            .context("invalid embedding image input")?;
        let features = self.tower.forward_on(&processed.pixel_values, ())?;
        self.projection.forward_on(&features, ())
    }
}

/// Validate only metadata; this does not allocate arrays or read weight files.
pub(super) fn validate_config(
    raw: &serde_json::Value,
) -> Result<Option<gemma4::Gemma4VisionConfig>> {
    let Some(value) = raw.get("vision_config").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let config: gemma4::Gemma4VisionConfig = serde_json::from_value(value.clone())?;
    ensure!(
        config
            .rope_parameters
            .as_ref()
            .is_some_and(|rope| rope.rope_type == "axial"),
        "unsupported embedding vision RoPE"
    );
    config.validate()?;
    ensure!(
        config.model_type == "gemma4_vision"
            && config.hidden_size == 768
            && config.intermediate_size == 3072
            && config.num_hidden_layers == 16
            && config.num_attention_heads == 12
            && config.num_key_value_heads == 12
            && config.head_dim == 64
            && config.patch_size == 16
            && config.pooling_kernel_size == 3
            && config.default_output_length == 280
            && config.rope_theta() == 100.0
            && !config.standardize
            && !config.use_clipped_linears,
        "unsupported EmbeddingGemma 2 vision configuration"
    );
    for (name, expected) in [
        ("image_token_id", 258880),
        ("boi_token_id", 255999),
        ("eoi_token_id", 258882),
    ] {
        ensure!(
            raw[name].as_u64() == Some(expected),
            "unsupported embedding image token {name}"
        );
    }
    ensure!(
        raw["text_config"]["hidden_size"] == 512,
        "unsupported embedding image projection dimension"
    );
    Ok(Some(config))
}
