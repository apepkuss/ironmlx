use std::collections::HashMap;

use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::core::Loader;
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4LayerKind {
    Sliding,
    Full,
}

impl Gemma4LayerKind {
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "sliding_attention" => Ok(Self::Sliding),
            "full_attention" => Ok(Self::Full),
            other => Err(anyhow!("Gemma4: unsupported layer type `{other}`")),
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
pub struct Gemma4RopeParams {
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

fn default_true() -> bool {
    true
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_max_position_embeddings() -> i32 {
    131_072
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionConfig {
    #[serde(default = "default_vision_model_type")]
    pub model_type: String,
    #[serde(default)]
    pub hidden_size: i32,
    #[serde(default)]
    pub intermediate_size: i32,
    #[serde(default)]
    pub num_hidden_layers: i32,
    #[serde(default)]
    pub num_attention_heads: i32,
    #[serde(default)]
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub head_dim: i32,
    #[serde(default)]
    pub global_head_dim: Option<i32>,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: String,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub rope_parameters: Option<Gemma4VisionRopeParams>,
    #[serde(default = "default_vision_output_length", alias = "num_soft_tokens")]
    pub default_output_length: i32,
    #[serde(default = "default_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_position_embedding_size")]
    pub position_embedding_size: i32,
    #[serde(default = "default_pooling_kernel_size")]
    pub pooling_kernel_size: i32,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default)]
    pub mm_embed_dim: i32,
    #[serde(default)]
    pub mm_posemb_size: i32,
    #[serde(default)]
    pub model_patch_size: i32,
    #[serde(default)]
    pub output_proj_dims: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionRopeParams {
    #[serde(default = "default_vision_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

fn default_vision_model_type() -> String {
    "gemma4_vision".to_owned()
}

fn default_hidden_activation() -> String {
    "gelu_pytorch_tanh".to_owned()
}

fn default_vision_output_length() -> i32 {
    280
}

fn default_patch_size() -> i32 {
    16
}

fn default_position_embedding_size() -> i32 {
    10_240
}

fn default_pooling_kernel_size() -> i32 {
    3
}

fn default_vision_rope_theta() -> f32 {
    100.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    #[serde(default = "default_text_model_type")]
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub head_dim: i32,
    #[serde(default)]
    pub global_head_dim: Option<i32>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<i32>,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub num_global_key_value_heads: Option<i32>,
    #[serde(default)]
    pub num_kv_shared_layers: i32,
    #[serde(default)]
    pub hidden_size_per_layer_input: i32,
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default)]
    pub rope_parameters: HashMap<String, Gemma4RopeParams>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<i32>,
    #[serde(default)]
    pub top_k_experts: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    pub layer_types: Vec<String>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,

    #[serde(skip)]
    layer_kinds: Vec<Gemma4LayerKind>,
    #[serde(skip)]
    previous_kvs: Vec<usize>,
}

fn default_text_model_type() -> String {
    "gemma4_text".to_owned()
}

fn default_sliding_window() -> i32 {
    512
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub text_config: Gemma4TextConfig,
    #[serde(default)]
    pub vision_config: Option<Gemma4VisionConfig>,
    #[serde(default)]
    pub image_token_id: Option<i32>,
    #[serde(default)]
    pub audio_token_id: Option<i32>,
    #[serde(default)]
    pub boi_token_id: Option<i32>,
    #[serde(default)]
    pub eoi_token_id: Option<i32>,
    #[serde(default = "default_vision_output_length")]
    pub vision_soft_tokens_per_image: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4AssistantConfig {
    #[serde(default = "default_assistant_model_type")]
    pub model_type: String,
    pub text_config: Gemma4TextConfig,
    pub backbone_hidden_size: i32,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub use_ordered_embeddings: bool,
    #[serde(default)]
    pub num_centroids: Option<i32>,
    #[serde(default)]
    pub centroid_intermediate_top_k: Option<i32>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
}

fn default_model_type() -> String {
    "gemma4".to_owned()
}

fn default_assistant_model_type() -> String {
    "gemma4_assistant".to_owned()
}

impl Gemma4Config {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let mut cfg: Gemma4Config = loader.config().context("parsing Gemma4Config")?;
        cfg.validate_and_finalize()?;
        Ok(cfg)
    }

    pub(crate) fn validate_and_finalize(&mut self) -> Result<()> {
        if !matches!(self.model_type.as_str(), "gemma4" | "gemma4_unified") {
            return Err(anyhow!(
                "Gemma4Config: expected model_type=gemma4 or gemma4_unified, got `{}`",
                self.model_type
            ));
        }
        match self.model_type.as_str() {
            "gemma4" if self.text_config.model_type != "gemma4_text" => {
                return Err(anyhow!(
                    "Gemma4Config: model_type=gemma4 requires text_config.model_type=gemma4_text, got `{}`",
                    self.text_config.model_type
                ));
            }
            "gemma4_unified" if self.text_config.model_type != "gemma4_unified_text" => {
                return Err(anyhow!(
                    "Gemma4Config: model_type=gemma4_unified requires text_config.model_type=gemma4_unified_text, got `{}`",
                    self.text_config.model_type
                ));
            }
            _ => {}
        }
        self.text_config.validate_and_finalize()?;
        if let Some(vc) = &self.vision_config {
            match self.model_type.as_str() {
                "gemma4" if vc.model_type != "gemma4_vision" => {
                    return Err(anyhow!(
                        "Gemma4Config: model_type=gemma4 requires vision_config.model_type=gemma4_vision, got `{}`",
                        vc.model_type
                    ));
                }
                "gemma4_unified" if vc.model_type != "gemma4_unified_vision" => {
                    return Err(anyhow!(
                        "Gemma4Config: model_type=gemma4_unified requires vision_config.model_type=gemma4_unified_vision, got `{}`",
                        vc.model_type
                    ));
                }
                _ => {}
            }
            vc.validate()?;
            self.vision_soft_tokens_per_image = vc.default_output_length;
        }
        Ok(())
    }
}

impl Gemma4AssistantConfig {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let mut cfg: Gemma4AssistantConfig =
            loader.config().context("parsing Gemma4AssistantConfig")?;
        cfg.validate_and_finalize()?;
        Ok(cfg)
    }

    pub(crate) fn validate_and_finalize(&mut self) -> Result<()> {
        if !matches!(
            self.model_type.as_str(),
            "gemma4_assistant" | "gemma4_unified_assistant"
        ) {
            return Err(anyhow!(
                "Gemma4AssistantConfig: expected model_type=gemma4_assistant or gemma4_unified_assistant, got `{}`",
                self.model_type
            ));
        }
        match self.model_type.as_str() {
            "gemma4_assistant" if self.text_config.model_type != "gemma4_text" => {
                return Err(anyhow!(
                    "Gemma4AssistantConfig: model_type=gemma4_assistant requires text_config.model_type=gemma4_text, got `{}`",
                    self.text_config.model_type
                ));
            }
            "gemma4_unified_assistant" if self.text_config.model_type != "gemma4_unified_text" => {
                return Err(anyhow!(
                    "Gemma4AssistantConfig: model_type=gemma4_unified_assistant requires text_config.model_type=gemma4_unified_text, got `{}`",
                    self.text_config.model_type
                ));
            }
            _ => {}
        }
        if self.backbone_hidden_size <= 0 {
            return Err(anyhow!(
                "Gemma4AssistantConfig: backbone_hidden_size must be > 0, got {}",
                self.backbone_hidden_size
            ));
        }
        if !self.tie_word_embeddings || !self.text_config.tie_word_embeddings {
            return Err(anyhow!(
                "Gemma4AssistantConfig: assistant checkpoints must use tied word embeddings"
            ));
        }
        if self.text_config.hidden_size_per_layer_input != 0 {
            return Err(anyhow!(
                "Gemma4AssistantConfig: assistant hidden_size_per_layer_input must be 0"
            ));
        }
        if self.text_config.num_kv_shared_layers != self.text_config.num_hidden_layers {
            return Err(anyhow!(
                "Gemma4AssistantConfig: assistant num_kv_shared_layers must equal num_hidden_layers"
            ));
        }
        if self.use_ordered_embeddings {
            if self.num_centroids.unwrap_or(0) <= 0 {
                return Err(anyhow!(
                    "Gemma4AssistantConfig: ordered embeddings require num_centroids > 0"
                ));
            }
            if self.centroid_intermediate_top_k.unwrap_or(0) <= 0 {
                return Err(anyhow!(
                    "Gemma4AssistantConfig: ordered embeddings require centroid_intermediate_top_k > 0"
                ));
            }
        }
        self.text_config.validate_and_finalize_for_assistant()?;
        Ok(())
    }
}

impl Gemma4VisionConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.is_unified() {
            return self.validate_unified();
        }
        if self.model_type != "gemma4_vision" {
            return Err(anyhow!(
                "Gemma4VisionConfig: expected model_type=gemma4_vision or gemma4_unified_vision, got `{}`",
                self.model_type
            ));
        }
        if self.hidden_size <= 0
            || self.intermediate_size <= 0
            || self.num_hidden_layers <= 0
            || self.num_attention_heads <= 0
            || self.num_key_value_heads <= 0
            || self.head_dim <= 0
        {
            return Err(anyhow!(
                "Gemma4VisionConfig: hidden/intermediate/layer/head dimensions must be positive"
            ));
        }
        if self.patch_size <= 0 || self.pooling_kernel_size <= 0 {
            return Err(anyhow!(
                "Gemma4VisionConfig: patch_size and pooling_kernel_size must be positive"
            ));
        }
        if self.default_output_length <= 0 || self.position_embedding_size <= 0 {
            return Err(anyhow!(
                "Gemma4VisionConfig: output length and position embedding size must be positive"
            ));
        }
        if self.hidden_activation != "gelu_pytorch_tanh" {
            return Err(anyhow!(
                "Gemma4VisionConfig: unsupported hidden_activation `{}`",
                self.hidden_activation
            ));
        }
        if let Some(params) = &self.rope_parameters {
            if params.rope_type != "default" {
                return Err(anyhow!(
                    "Gemma4VisionConfig: unsupported rope_type `{}`",
                    params.rope_type
                ));
            }
        }
        Ok(())
    }

    fn validate_unified(&self) -> Result<()> {
        if self.patch_size <= 0 || self.pooling_kernel_size <= 0 {
            return Err(anyhow!(
                "Gemma4UnifiedVisionConfig: patch_size and pooling_kernel_size must be positive"
            ));
        }
        if self.default_output_length <= 0 {
            return Err(anyhow!(
                "Gemma4UnifiedVisionConfig: num_soft_tokens/default_output_length must be positive"
            ));
        }
        if self.mm_embed_dim <= 0 || self.mm_posemb_size <= 0 || self.output_proj_dims <= 0 {
            return Err(anyhow!(
                "Gemma4UnifiedVisionConfig: mm_embed_dim/mm_posemb_size/output_proj_dims must be positive"
            ));
        }
        let derived = self.patch_size * self.pooling_kernel_size;
        let configured = self.model_patch_size();
        if configured != derived {
            return Err(anyhow!(
                "Gemma4UnifiedVisionConfig: model_patch_size={} must equal patch_size*pooling_kernel_size={derived}",
                configured
            ));
        }
        Ok(())
    }

    pub fn is_unified(&self) -> bool {
        self.model_type == "gemma4_unified_vision"
    }

    pub fn model_patch_size(&self) -> i32 {
        if self.model_patch_size > 0 {
            self.model_patch_size
        } else {
            self.patch_size * self.pooling_kernel_size
        }
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.rope_theta)
            .unwrap_or_else(default_vision_rope_theta)
    }

    pub fn max_patches(&self) -> i32 {
        self.default_output_length * self.pooling_kernel_size * self.pooling_kernel_size
    }
}

impl Gemma4TextConfig {
    fn validate_and_finalize(&mut self) -> Result<()> {
        self.validate_and_finalize_impl(false)
    }

    fn validate_and_finalize_for_assistant(&mut self) -> Result<()> {
        self.validate_and_finalize_impl(true)
    }

    fn validate_and_finalize_impl(&mut self, assistant: bool) -> Result<()> {
        if !matches!(
            self.model_type.as_str(),
            "gemma4_text" | "gemma4_unified_text"
        ) {
            return Err(anyhow!(
                "Gemma4TextConfig: expected text_config.model_type=gemma4_text or gemma4_unified_text, got `{}`",
                self.model_type
            ));
        }
        if assistant && self.enable_moe_block {
            return Err(anyhow!(
                "Gemma4AssistantConfig: assistant MoE text_config is not supported"
            ));
        }
        self.validate_moe_config()?;
        if self.num_hidden_layers <= 0 {
            return Err(anyhow!(
                "Gemma4TextConfig: num_hidden_layers must be > 0, got {}",
                self.num_hidden_layers
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers as usize {
            return Err(anyhow!(
                "Gemma4TextConfig: layer_types.len()={} != num_hidden_layers={}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        let invalid_kv_shared = if assistant {
            self.num_kv_shared_layers < 0 || self.num_kv_shared_layers > self.num_hidden_layers
        } else {
            self.num_kv_shared_layers < 0 || self.num_kv_shared_layers >= self.num_hidden_layers
        };
        if invalid_kv_shared {
            return Err(anyhow!(
                "Gemma4TextConfig: num_kv_shared_layers={} out of supported range for num_hidden_layers={}",
                self.num_kv_shared_layers,
                self.num_hidden_layers
            ));
        }
        if self.hidden_size_per_layer_input > 0 && self.vocab_size_per_layer_input.is_none() {
            return Err(anyhow!(
                "Gemma4TextConfig: vocab_size_per_layer_input required when hidden_size_per_layer_input > 0"
            ));
        }
        if !self.tie_word_embeddings {
            return Err(anyhow!("Gemma4 text path expects tie_word_embeddings=true"));
        }
        if !self.rope_parameters.contains_key("sliding_attention") {
            self.rope_parameters.insert(
                "sliding_attention".to_owned(),
                Gemma4RopeParams {
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
                Gemma4RopeParams {
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
            .map(|s| Gemma4LayerKind::from_str(s))
            .collect::<Result<Vec<_>>>()?;
        self.previous_kvs = self.build_previous_kvs()?;
        Ok(())
    }

    fn validate_moe_config(&self) -> Result<()> {
        if !self.enable_moe_block {
            if self.num_experts.is_some()
                || self.top_k_experts.is_some()
                || self.moe_intermediate_size.is_some()
            {
                return Err(anyhow!(
                    "Gemma4TextConfig: num_experts/top_k_experts/moe_intermediate_size require enable_moe_block=true"
                ));
            }
            return Ok(());
        }

        let num_experts = self.num_experts.ok_or_else(|| {
            anyhow!("Gemma4TextConfig: enable_moe_block=true requires num_experts")
        })?;
        if num_experts <= 0 {
            return Err(anyhow!(
                "Gemma4TextConfig: num_experts must be > 0, got {num_experts}"
            ));
        }
        let top_k = self.top_k_experts.ok_or_else(|| {
            anyhow!("Gemma4TextConfig: enable_moe_block=true requires top_k_experts")
        })?;
        if top_k <= 0 || top_k > num_experts {
            return Err(anyhow!(
                "Gemma4TextConfig: top_k_experts must be in 1..=num_experts, got top_k_experts={top_k}, num_experts={num_experts}"
            ));
        }
        let moe_intermediate = self.moe_intermediate_size.ok_or_else(|| {
            anyhow!("Gemma4TextConfig: enable_moe_block=true requires moe_intermediate_size")
        })?;
        if moe_intermediate <= 0 {
            return Err(anyhow!(
                "Gemma4TextConfig: moe_intermediate_size must be > 0, got {moe_intermediate}"
            ));
        }
        Ok(())
    }

    fn build_previous_kvs(&self) -> Result<Vec<usize>> {
        let n = self.num_hidden_layers as usize;
        let first_shared = self.first_kv_shared_layer_idx();
        let mut previous: Vec<usize> = (0..n).collect();
        if first_shared == 0 || first_shared >= n {
            return Ok(previous);
        }

        let mut last_by_type: HashMap<&'static str, usize> = HashMap::new();
        for i in 0..first_shared {
            last_by_type.insert(self.layer_kind(i).as_key(), i);
        }
        for (i, slot) in previous.iter_mut().enumerate().skip(first_shared) {
            let key = self.layer_kind(i).as_key();
            let prev = last_by_type.get(key).copied().ok_or_else(|| {
                anyhow!("Gemma4TextConfig: no pre-shared K/V source for layer type `{key}`")
            })?;
            *slot = prev;
        }
        Ok(previous)
    }

    pub fn layer_kind(&self, layer_idx: usize) -> Gemma4LayerKind {
        self.layer_kinds[layer_idx]
    }

    pub fn first_kv_shared_layer_idx(&self) -> usize {
        (self.num_hidden_layers - self.num_kv_shared_layers) as usize
    }

    pub fn previous_kv_layer(&self, layer_idx: usize) -> usize {
        self.previous_kvs[layer_idx]
    }

    pub fn all_layers_share_external_kv(&self) -> bool {
        self.num_kv_shared_layers == self.num_hidden_layers
    }

    pub(crate) fn num_experts_value(&self) -> i32 {
        self.num_experts
            .expect("Gemma4TextConfig validation requires num_experts for MoE")
    }

    pub(crate) fn top_k_experts_value(&self) -> i32 {
        self.top_k_experts
            .expect("Gemma4TextConfig validation requires top_k_experts for MoE")
    }

    pub fn rope_params_for(&self, kind: Gemma4LayerKind) -> &Gemma4RopeParams {
        self.rope_parameters
            .get(kind.as_key())
            .expect("validate_and_finalize inserted rope params")
    }

    pub fn head_dim_for_layer(&self, layer_idx: usize) -> i32 {
        match self.layer_kind(layer_idx) {
            Gemma4LayerKind::Sliding => self.head_dim,
            Gemma4LayerKind::Full => self.global_head_dim.unwrap_or(self.head_dim),
        }
    }

    pub fn kv_heads_for_layer(&self, layer_idx: usize) -> i32 {
        match (
            self.layer_kind(layer_idx),
            self.attention_k_eq_v,
            self.num_global_key_value_heads,
        ) {
            (Gemma4LayerKind::Full, true, Some(global_heads)) => global_heads,
            _ => self.num_key_value_heads,
        }
    }

    pub fn vocab_size_per_layer_input(&self) -> i32 {
        self.vocab_size_per_layer_input.unwrap_or(self.vocab_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e4b_like() -> Gemma4TextConfig {
        let mut cfg: Gemma4TextConfig = serde_json::from_value(serde_json::json!({
            "model_type": "gemma4_text",
            "hidden_size": 2560,
            "num_hidden_layers": 42,
            "intermediate_size": 10240,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "vocab_size_per_layer_input": 262144,
            "num_key_value_heads": 2,
            "num_kv_shared_layers": 18,
            "hidden_size_per_layer_input": 256,
            "layer_types": (0..42).map(|i| if (i + 1) % 6 == 0 { "full_attention" } else { "sliding_attention" }).collect::<Vec<_>>(),
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }))
        .unwrap();
        cfg.validate_and_finalize().unwrap();
        cfg
    }

    fn moe_26b_like() -> Gemma4TextConfig {
        let mut cfg: Gemma4TextConfig = serde_json::from_value(serde_json::json!({
            "model_type": "gemma4_text",
            "hidden_size": 2816,
            "num_hidden_layers": 30,
            "intermediate_size": 2112,
            "moe_intermediate_size": 704,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 2,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "attention_k_eq_v": true,
            "enable_moe_block": true,
            "num_experts": 128,
            "top_k_experts": 8,
            "layer_types": (0..30).map(|i| if (i + 1) % 6 == 0 { "full_attention" } else { "sliding_attention" }).collect::<Vec<_>>(),
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }))
        .unwrap();
        cfg.validate_and_finalize().unwrap();
        cfg
    }

    fn unified_12b_like() -> Gemma4Config {
        let layer_types = (0..48)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect::<Vec<_>>();
        let mut cfg: Gemma4Config = serde_json::from_value(serde_json::json!({
            "model_type": "gemma4_unified",
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "boi_token_id": 255999,
            "eoi_token_id": 258882,
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 3840,
                "num_hidden_layers": 48,
                "intermediate_size": 15360,
                "num_attention_heads": 16,
                "head_dim": 256,
                "global_head_dim": 512,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "attention_k_eq_v": true,
                "enable_moe_block": false,
                "num_experts": null,
                "top_k_experts": null,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "final_logit_softcapping": 30.0,
                "sliding_window": 1024,
                "max_position_embeddings": 262144,
                "layer_types": layer_types,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            },
            "vision_config": {
                "model_type": "gemma4_unified_vision",
                "mm_embed_dim": 3840,
                "mm_posemb_size": 1120,
                "model_patch_size": 48,
                "num_soft_tokens": 280,
                "output_proj_dims": 3840,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "rms_norm_eps": 0.000001
            }
        }))
        .unwrap();
        cfg.validate_and_finalize().unwrap();
        cfg
    }

    #[test]
    fn gemma4_moe_26b_like_config_validates() {
        let cfg = moe_26b_like();

        assert!(cfg.enable_moe_block);
        assert_eq!(cfg.num_experts_value(), 128);
        assert_eq!(cfg.top_k_experts_value(), 8);
        assert_eq!(cfg.moe_intermediate_size, Some(704));
        assert_eq!(cfg.first_kv_shared_layer_idx(), 30);
        assert_eq!(cfg.kv_heads_for_layer(5), 2);
    }

    #[test]
    fn dense_config_rejects_moe_fields_without_moe_enabled() {
        let mut value = serde_json::json!({
            "model_type": "gemma4_text",
            "hidden_size": 2816,
            "num_hidden_layers": 1,
            "intermediate_size": 2112,
            "num_attention_heads": 8,
            "head_dim": 256,
            "vocab_size": 262144,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "layer_types": ["full_attention"],
            "num_experts": 128
        });
        let mut cfg: Gemma4TextConfig = serde_json::from_value(value.clone()).unwrap();
        let err = cfg
            .validate_and_finalize()
            .expect_err("dense config must reject dangling MoE fields");
        assert!(err.to_string().contains("require enable_moe_block=true"));

        value["enable_moe_block"] = serde_json::Value::Bool(true);
        let mut cfg: Gemma4TextConfig = serde_json::from_value(value).unwrap();
        let err = cfg
            .validate_and_finalize()
            .expect_err("MoE config must require top_k_experts and moe_intermediate_size");
        assert!(err.to_string().contains("requires top_k_experts"));
    }

    fn e4b_assistant_like() -> Gemma4AssistantConfig {
        let mut cfg: Gemma4AssistantConfig = serde_json::from_value(serde_json::json!({
            "model_type": "gemma4_assistant",
            "architectures": ["Gemma4AssistantForCausalLM"],
            "backbone_hidden_size": 2560,
            "use_ordered_embeddings": true,
            "num_centroids": 2048,
            "centroid_intermediate_top_k": 32,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 2048,
                "num_attention_heads": 4,
                "head_dim": 256,
                "global_head_dim": 512,
                "vocab_size": 262144,
                "num_key_value_heads": 2,
                "num_kv_shared_layers": 4,
                "hidden_size_per_layer_input": 0,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "tie_word_embeddings": true
            }
        }))
        .unwrap();
        cfg.validate_and_finalize().unwrap();
        cfg
    }

    #[test]
    fn e4b_assistant_config_accepts_all_external_shared_kv_layers() {
        let cfg = e4b_assistant_like();

        assert_eq!(cfg.model_type, "gemma4_assistant");
        assert_eq!(cfg.backbone_hidden_size, 2560);
        assert_eq!(cfg.text_config.first_kv_shared_layer_idx(), 0);
        assert!(cfg.text_config.all_layers_share_external_kv());
        assert_eq!(cfg.text_config.previous_kv_layer(0), 0);
        assert_eq!(cfg.text_config.layer_kind(3), Gemma4LayerKind::Full);
    }

    #[test]
    fn unified_assistant_config_accepts_gemma4_unified_assistant_type() {
        let mut cfg: Gemma4AssistantConfig = serde_json::from_value(serde_json::json!({
            "model_type": "gemma4_unified_assistant",
            "backbone_hidden_size": 3840,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "intermediate_size": 8192,
                "num_attention_heads": 16,
                "head_dim": 256,
                "global_head_dim": 512,
                "vocab_size": 262144,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 1,
                "num_kv_shared_layers": 4,
                "hidden_size_per_layer_input": 0,
                "attention_k_eq_v": true,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "tie_word_embeddings": true
            }
        }))
        .unwrap();

        cfg.validate_and_finalize().unwrap();

        assert_eq!(cfg.model_type, "gemma4_unified_assistant");
        assert!(cfg.text_config.all_layers_share_external_kv());
    }

    #[test]
    fn e4b_previous_kv_mapping_by_layer_type() {
        let cfg = e4b_like();
        assert_eq!(cfg.first_kv_shared_layer_idx(), 24);
        assert_eq!(cfg.previous_kv_layer(24), 22);
        assert_eq!(cfg.previous_kv_layer(29), 23);
        assert_eq!(cfg.previous_kv_layer(41), 23);
        assert_eq!(cfg.head_dim_for_layer(0), 256);
        assert_eq!(cfg.head_dim_for_layer(5), 512);
    }

    #[test]
    fn unified_12b_dense_config_parses_text_and_vision() {
        let cfg = unified_12b_like();

        assert_eq!(cfg.model_type, "gemma4_unified");
        assert_eq!(cfg.vision_soft_tokens_per_image, 280);
        assert_eq!(cfg.text_config.first_kv_shared_layer_idx(), 48);
        assert_eq!(cfg.text_config.previous_kv_layer(47), 47);
        assert_eq!(cfg.text_config.head_dim_for_layer(5), 512);
        assert_eq!(cfg.text_config.kv_heads_for_layer(5), 1);
        assert_eq!(cfg.text_config.kv_heads_for_layer(0), 8);

        let vision = cfg.vision_config.as_ref().expect("vision config");
        assert!(vision.is_unified());
        assert_eq!(vision.default_output_length, 280);
        assert_eq!(vision.max_patches(), 2520);
        assert_eq!(vision.model_patch_size(), 48);
        assert_eq!(vision.mm_embed_dim, 3840);
        assert_eq!(vision.mm_posemb_size, 1120);
        assert_eq!(vision.output_proj_dims, 3840);
    }
}
