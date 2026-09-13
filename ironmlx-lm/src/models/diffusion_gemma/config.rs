use std::collections::HashMap;

use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::core::Loader;
use crate::models::gemma4::Gemma4VisionConfig;
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiffusionGemmaLayerKind {
    Sliding,
    Full,
}

impl DiffusionGemmaLayerKind {
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "sliding_attention" => Ok(Self::Sliding),
            "full_attention" => Ok(Self::Full),
            other => Err(anyhow!("DiffusionGemma: unsupported layer type `{other}`")),
        }
    }

    pub fn as_key(self) -> &'static str {
        match self {
            Self::Sliding => "sliding_attention",
            Self::Full => "full_attention",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionGemmaRopeParams {
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default = "default_factor")]
    pub factor: f32,
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_rope_type() -> String {
    "default".to_owned()
}

fn default_factor() -> f32 {
    1.0
}

fn default_text_model_type() -> String {
    "diffusion_gemma_text".to_owned()
}

fn default_model_type() -> String {
    "diffusion_gemma".to_owned()
}

fn default_true() -> bool {
    true
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_sliding_window() -> i32 {
    1024
}

fn default_max_position_embeddings() -> i32 {
    131_072
}

fn default_pad_token_id() -> i32 {
    0
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionGemmaTextConfig {
    #[serde(default = "default_text_model_type")]
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub moe_intermediate_size: i32,
    pub num_attention_heads: i32,
    pub head_dim: i32,
    #[serde(default)]
    pub global_head_dim: Option<i32>,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub num_global_key_value_heads: Option<i32>,
    pub num_experts: i32,
    pub top_k_experts: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub rope_parameters: HashMap<String, DiffusionGemmaRopeParams>,
    pub layer_types: Vec<String>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: i32,
    #[serde(default)]
    pub use_bidirectional_attention: Option<String>,

    #[serde(skip)]
    layer_kinds: Vec<DiffusionGemmaLayerKind>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionGemmaConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub canvas_length: i32,
    pub text_config: DiffusionGemmaTextConfig,
    #[serde(default)]
    pub vision_config: Option<Gemma4VisionConfig>,
    #[serde(default)]
    pub image_token_id: Option<i32>,
    #[serde(default)]
    pub boi_token_id: Option<i32>,
    #[serde(default)]
    pub eoi_token_id: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionGemmaSamplerConfig {
    #[serde(rename = "_cls_name", default = "default_sampler_name")]
    pub class_name: String,
    #[serde(default = "default_entropy_bound")]
    pub entropy_bound: f32,
}

fn default_sampler_name() -> String {
    "EntropyBoundSamplerConfig".to_owned()
}

fn default_entropy_bound() -> f32 {
    0.1
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionGemmaGenerationConfig {
    #[serde(default = "default_max_denoising_steps")]
    pub max_denoising_steps: i32,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_t_min")]
    pub t_min: f32,
    #[serde(default = "default_t_max")]
    pub t_max: f32,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default = "default_stability_threshold")]
    pub stability_threshold: usize,
    #[serde(default)]
    pub eos_token_id: Option<crate::core::loader::EosTokenId>,
    #[serde(default)]
    pub sampler_config: Option<DiffusionGemmaSamplerConfig>,
}

fn default_max_denoising_steps() -> i32 {
    48
}

fn default_max_new_tokens() -> usize {
    256
}

fn default_t_min() -> f32 {
    0.4
}

fn default_t_max() -> f32 {
    0.8
}

fn default_confidence_threshold() -> f32 {
    0.005
}

fn default_stability_threshold() -> usize {
    1
}

impl DiffusionGemmaConfig {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let mut cfg: DiffusionGemmaConfig =
            loader.config().context("parsing DiffusionGemmaConfig")?;
        cfg.validate_and_finalize()?;
        Ok(cfg)
    }

    fn validate_and_finalize(&mut self) -> Result<()> {
        if self.model_type != "diffusion_gemma" {
            return Err(anyhow!(
                "DiffusionGemmaConfig: expected model_type=diffusion_gemma, got `{}`",
                self.model_type
            ));
        }
        if self.canvas_length <= 0 {
            return Err(anyhow!(
                "DiffusionGemmaConfig: canvas_length must be > 0, got {}",
                self.canvas_length
            ));
        }
        self.text_config.validate_and_finalize()?;
        if let Some(vc) = &self.vision_config {
            vc.validate()?;
        }
        Ok(())
    }
}

impl DiffusionGemmaTextConfig {
    fn validate_and_finalize(&mut self) -> Result<()> {
        if self.model_type != "diffusion_gemma_text" {
            return Err(anyhow!(
                "DiffusionGemmaTextConfig: expected text_config.model_type=diffusion_gemma_text, got `{}`",
                self.model_type
            ));
        }
        if self.hidden_size <= 0
            || self.num_hidden_layers <= 0
            || self.intermediate_size <= 0
            || self.moe_intermediate_size <= 0
            || self.num_attention_heads <= 0
            || self.num_key_value_heads <= 0
            || self.head_dim <= 0
            || self.vocab_size <= 0
            || self.num_experts <= 0
            || self.top_k_experts <= 0
        {
            return Err(anyhow!(
                "DiffusionGemmaTextConfig: hidden/layer/head/expert dimensions must be positive"
            ));
        }
        if self.top_k_experts > self.num_experts {
            return Err(anyhow!(
                "DiffusionGemmaTextConfig: top_k_experts={} > num_experts={}",
                self.top_k_experts,
                self.num_experts
            ));
        }
        if self.pad_token_id < 0 {
            return Err(anyhow!(
                "DiffusionGemmaTextConfig: pad_token_id must be non-negative"
            ));
        }
        if let Some(mode) = &self.use_bidirectional_attention {
            if mode != "vision" {
                return Err(anyhow!(
                    "DiffusionGemmaTextConfig: unsupported use_bidirectional_attention `{mode}`"
                ));
            }
        }
        if self.layer_types.len() != self.num_hidden_layers as usize {
            return Err(anyhow!(
                "DiffusionGemmaTextConfig: layer_types.len()={} != num_hidden_layers={}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if !self.tie_word_embeddings {
            return Err(anyhow!(
                "DiffusionGemmaTextConfig: text-only path requires tie_word_embeddings=true"
            ));
        }
        if !self.rope_parameters.contains_key("sliding_attention") {
            self.rope_parameters.insert(
                "sliding_attention".to_owned(),
                DiffusionGemmaRopeParams {
                    partial_rotary_factor: 1.0,
                    rope_theta: 10_000.0,
                    rope_type: "default".to_owned(),
                    factor: 1.0,
                },
            );
        }
        if !self.rope_parameters.contains_key("full_attention") {
            self.rope_parameters.insert(
                "full_attention".to_owned(),
                DiffusionGemmaRopeParams {
                    partial_rotary_factor: 0.25,
                    rope_theta: 1_000_000.0,
                    rope_type: "proportional".to_owned(),
                    factor: 1.0,
                },
            );
        }
        self.layer_kinds = self
            .layer_types
            .iter()
            .map(|s| DiffusionGemmaLayerKind::from_str(s))
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    pub fn layer_kind(&self, layer_idx: usize) -> DiffusionGemmaLayerKind {
        self.layer_kinds[layer_idx]
    }

    pub fn rope_params_for(&self, kind: DiffusionGemmaLayerKind) -> &DiffusionGemmaRopeParams {
        self.rope_parameters
            .get(kind.as_key())
            .expect("validate_and_finalize inserted rope params")
    }

    pub fn head_dim_for_layer(&self, layer_idx: usize) -> i32 {
        match self.layer_kind(layer_idx) {
            DiffusionGemmaLayerKind::Sliding => self.head_dim,
            DiffusionGemmaLayerKind::Full => self.global_head_dim.unwrap_or(self.head_dim),
        }
    }

    pub fn kv_heads_for_layer(&self, layer_idx: usize) -> i32 {
        match (self.layer_kind(layer_idx), self.num_global_key_value_heads) {
            (DiffusionGemmaLayerKind::Full, Some(global_heads)) => global_heads,
            _ => self.num_key_value_heads,
        }
    }
}

impl DiffusionGemmaGenerationConfig {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let path = loader.model_dir().join("generation_config.json");
        if !path.exists() {
            return Ok(Self {
                max_denoising_steps: default_max_denoising_steps(),
                max_new_tokens: default_max_new_tokens(),
                t_min: default_t_min(),
                t_max: default_t_max(),
                confidence_threshold: default_confidence_threshold(),
                stability_threshold: default_stability_threshold(),
                eos_token_id: None,
                sampler_config: Some(DiffusionGemmaSamplerConfig {
                    class_name: default_sampler_name(),
                    entropy_bound: default_entropy_bound(),
                }),
            });
        }
        let mut cfg: Self = serde_json::from_reader(
            std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))?;
        if cfg.max_denoising_steps <= 0 {
            return Err(anyhow!(
                "DiffusionGemmaGenerationConfig: max_denoising_steps must be > 0"
            ));
        }
        if cfg.sampler_config.is_none() {
            cfg.sampler_config = Some(DiffusionGemmaSamplerConfig {
                class_name: default_sampler_name(),
                entropy_bound: default_entropy_bound(),
            });
        }
        Ok(cfg)
    }

    pub fn entropy_bound(&self) -> Result<f32> {
        let sampler = self
            .sampler_config
            .as_ref()
            .expect("from_loader fills sampler_config");
        if sampler.class_name != "EntropyBoundSamplerConfig" {
            return Err(anyhow!(
                "DiffusionGemma only supports EntropyBoundSamplerConfig, got `{}`",
                sampler.class_name
            ));
        }
        Ok(sampler.entropy_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tiny_diffusion_gemma_text_config() {
        let mut cfg: DiffusionGemmaConfig = serde_json::from_value(serde_json::json!({
            "model_type": "diffusion_gemma",
            "canvas_length": 3,
            "vision_config": {
                "model_type": "gemma4_vision",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "default_output_length": 280
            },
            "text_config": {
                "model_type": "diffusion_gemma_text",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 24,
                "moe_intermediate_size": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_global_key_value_heads": 1,
                "head_dim": 4,
                "global_head_dim": 4,
                "vocab_size": 64,
                "num_experts": 4,
                "top_k_experts": 2,
                "sliding_window": 8,
                "pad_token_id": 0,
                "use_bidirectional_attention": "vision",
                "layer_types": ["sliding_attention", "full_attention"]
            }
        }))
        .unwrap();

        cfg.validate_and_finalize().unwrap();

        assert_eq!(cfg.model_type, "diffusion_gemma");
        assert_eq!(
            cfg.text_config.layer_kind(0),
            DiffusionGemmaLayerKind::Sliding
        );
        assert_eq!(cfg.text_config.layer_kind(1), DiffusionGemmaLayerKind::Full);
        assert_eq!(cfg.text_config.head_dim_for_layer(1), 4);
        assert_eq!(cfg.text_config.kv_heads_for_layer(1), 1);
        assert_eq!(cfg.text_config.pad_token_id, 0);
        assert_eq!(
            cfg.text_config.use_bidirectional_attention.as_deref(),
            Some("vision")
        );
        assert_eq!(
            cfg.vision_config
                .as_ref()
                .expect("vision_config")
                .pooling_kernel_size,
            3
        );
    }
}
