use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::loader::ComponentLoader;
use crate::Result;
use ironmlx_lm::models::qwen3_5::VisionConfig;

#[derive(Debug, Clone, Deserialize)]
pub struct QwenImage21TextConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: QwenImage21RopeScaling,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QwenImage21RopeScaling {
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<i32>,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TextEncoderEnvelope {
    model_type: String,
    text_config: QwenImage21TextConfig,
    vision_config: VisionConfig,
}

impl QwenImage21TextConfig {
    pub fn from_loader(loader: &ComponentLoader) -> Result<Self> {
        let envelope: TextEncoderEnvelope = loader
            .config()
            .context("parsing Qwen Image 2.1 text encoder config")?;
        if envelope.model_type != "qwen3_vl" {
            return Err(anyhow!(
                "Qwen Image 2.1 text encoder expected model_type qwen3_vl, got {}",
                envelope.model_type
            ));
        }
        envelope.text_config.validate()?;
        Ok(envelope.text_config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size != 4096
            || self.intermediate_size != 12288
            || self.num_hidden_layers != 36
            || self.num_attention_heads != 32
            || self.num_key_value_heads != 8
            || self.head_dim != 128
            || self.vocab_size != 151936
        {
            return Err(anyhow!(
                "unsupported Qwen Image 2.1 text encoder dimensions"
            ));
        }
        if self.rope_scaling.rope_type != "default"
            || !self.rope_scaling.mrope_interleaved
            || self.rope_scaling.mrope_section != [24, 20, 20]
        {
            return Err(anyhow!(
                "unsupported Qwen Image 2.1 text encoder MRoPE configuration"
            ));
        }
        Ok(())
    }
}

pub(crate) fn vision_config_from_loader(loader: &ComponentLoader) -> Result<VisionConfig> {
    let envelope: TextEncoderEnvelope = loader
        .config()
        .context("parsing Qwen Image 2.1 vision encoder config")?;
    if envelope.model_type != "qwen3_vl" {
        return Err(anyhow!(
            "Qwen Image 2.1 vision encoder expected model_type qwen3_vl, got {}",
            envelope.model_type
        ));
    }
    let config = envelope.vision_config;
    if config.depth != 27
        || config.hidden_size != 1152
        || config.num_heads != 16
        || config.intermediate_size != 4304
        || config.out_hidden_size != 4096
        || config.patch_size != 16
        || config.spatial_merge_size != 2
        || config.temporal_patch_size != 2
        || config.in_channels != 3
        || config.num_position_embeddings != 2304
        || config.deepstack_visual_indexes != [8, 16, 24]
    {
        return Err(anyhow!(
            "unsupported Qwen Image 2.1 vision encoder dimensions"
        ));
    }
    Ok(config)
}
