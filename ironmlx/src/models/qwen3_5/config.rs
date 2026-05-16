//! Qwen3.5 text-config parsing.

use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::core::Loader;
use crate::nn::AttnKind;
use crate::Result;

/// RoPE-related fields parsed out of `text_config.rope_parameters`.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Per-stream length list (sum = rot_dim/2). Qwen3.5 default `[11, 11, 10]`.
    #[serde(default)]
    pub mrope_section: Vec<i32>,
}

fn default_partial_rotary_factor() -> f32 {
    0.25
}

fn default_max_position_embeddings() -> i32 {
    // Conservative fallback for older / non-Qwen3 configs that omit the field.
    // Production Qwen3.5 configs always declare it (262144 for 4B variant).
    32768
}

fn default_rope_theta() -> f32 {
    100_000.0
}

/// Vision encoder config from Qwen3.5 `config["vision_config"]`.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: i32,
    pub hidden_size: i32,
    pub num_heads: i32,
    pub intermediate_size: i32,
    pub out_hidden_size: i32,
    pub patch_size: i32,
    pub spatial_merge_size: i32,
    pub temporal_patch_size: i32,
    pub in_channels: i32,
    pub num_position_embeddings: i32,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<i32>,
}

/// Subset of `config.json["text_config"]` that drives Qwen3.5 inference.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    /// `None` in raw config → derived from `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<i32>,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub full_attention_interval: i32,
    // Linear-attn fields. Default to 0 if absent (non-hybrid Qwen3 variants).
    #[serde(default)]
    pub linear_num_value_heads: i32,
    #[serde(default)]
    pub linear_num_key_heads: i32,
    #[serde(default)]
    pub linear_key_head_dim: i32,
    #[serde(default)]
    pub linear_value_head_dim: i32,
    #[serde(default)]
    pub linear_conv_kernel_dim: i32,
    #[serde(default)]
    pub rope_parameters: RopeParams,
    /// Present in multimodal variants (VL models); `None` for text-only.
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    /// Maximum sequence length the model supports (= `text_config.max_position_embeddings`
    /// from config.json). Qwen3.5-4B: 262144. Used as a hard upper bound on
    /// per-request `prompt_len + max_new_tokens` to prevent MRoPE
    /// out-of-distribution garbage. B1-p2.3f.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self {
            partial_rotary_factor: default_partial_rotary_factor(),
            rope_theta: default_rope_theta(),
            mrope_section: Vec::new(),
        }
    }
}

impl Qwen35Config {
    /// Parse from a [`Loader`]'s `config.json`. Reads `config["text_config"]`.
    /// For multimodal models, also reads top-level `config["vision_config"]`.
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let raw = loader.config_raw_value();
        let text_config = raw
            .get("text_config")
            .ok_or_else(|| anyhow!("config.json missing text_config field"))?;
        let mut cfg: Qwen35Config = serde_json::from_value(text_config.clone())
            .context("failed to deserialize Qwen35Config from text_config")?;
        // 顶层也可能有 vision_config（multimodal model）
        if let Some(vc) = raw.get("vision_config") {
            cfg.vision_config = Some(
                serde_json::from_value(vc.clone()).context("failed to deserialize VisionConfig")?,
            );
        }
        Ok(cfg)
    }

    /// Effective per-head dim: `head_dim` if specified, else `hidden_size / num_attention_heads`.
    pub fn effective_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Returns the attention path for `layer_idx` (0-based).
    /// Layer i is Full when `(i + 1) % full_attention_interval == 0`, else Linear.
    pub fn layer_kind(&self, layer_idx: i32) -> AttnKind {
        if (layer_idx + 1) % self.full_attention_interval == 0 {
            AttnKind::Full
        } else {
            AttnKind::Linear
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real text_config from mlx-community/Qwen3.5-4B-MLX-4bit (subset).
    fn realistic_text_config_json() -> serde_json::Value {
        serde_json::json!({
            "attention_bias": false,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 192,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 64,
            "linear_value_head_dim": 128,
            "num_attention_heads": 20,
            "num_hidden_layers": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 100000.0,
                "type": "default"
            },
            "tie_word_embeddings": true,
            "vocab_size": 248064
        })
    }

    #[test]
    fn parses_real_text_config_subset() {
        let v = realistic_text_config_json();
        let cfg: Qwen35Config = serde_json::from_value(v).expect("parse");
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.head_dim, Some(256));
        assert_eq!(cfg.linear_num_value_heads, 64);
        assert_eq!(cfg.linear_key_head_dim, 192);
        assert_eq!(cfg.tie_word_embeddings, true);
        assert_eq!(cfg.rope_parameters.mrope_section, vec![11, 11, 10]);
        assert!((cfg.rope_parameters.rope_theta - 100_000.0).abs() < 1e-3);
        assert!((cfg.rope_parameters.partial_rotary_factor - 0.25).abs() < 1e-6);
    }

    #[test]
    fn effective_head_dim_default_path() {
        let mut cfg: Qwen35Config = serde_json::from_value(realistic_text_config_json()).unwrap();
        cfg.head_dim = None;
        // hidden_size=2560, num_attention_heads=20 → 128
        assert_eq!(cfg.effective_head_dim(), 128);
    }

    #[test]
    fn effective_head_dim_explicit_path() {
        let cfg: Qwen35Config = serde_json::from_value(realistic_text_config_json()).unwrap();
        // explicit head_dim=256 wins over hidden/heads = 128
        assert_eq!(cfg.effective_head_dim(), 256);
    }

    #[test]
    fn vision_config_parsed_from_qwen35_4b() {
        use crate::core::Loader;
        let env = std::env::var("QWEN35_MODEL");
        let dir = match env {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: QWEN35_MODEL not set");
                return;
            }
        };
        let loader = Loader::open(std::path::Path::new(&dir)).expect("load");
        let cfg = Qwen35Config::from_loader(&loader).expect("parse");
        let vc = cfg.vision_config.as_ref().expect("vision_config present");
        assert_eq!(vc.depth, 24);
        assert_eq!(vc.hidden_size, 1024);
        assert_eq!(vc.num_heads, 16);
        assert_eq!(vc.intermediate_size, 4096);
        assert_eq!(vc.out_hidden_size, 2560);
        assert_eq!(vc.patch_size, 16);
        assert_eq!(vc.spatial_merge_size, 2);
        assert_eq!(vc.temporal_patch_size, 2);
        assert_eq!(vc.in_channels, 3);
        assert_eq!(vc.num_position_embeddings, 2304);
    }

    #[test]
    fn layer_kind_partition_full_attention_interval_4() {
        let cfg: Qwen35Config = serde_json::from_value(realistic_text_config_json()).unwrap();
        // With full_attention_interval=4, num_hidden_layers=32:
        //   Full layers at idx ∈ {3, 7, 11, 15, 19, 23, 27, 31} (8 of them)
        //   Linear elsewhere (24 of them)
        let mut full_indices: Vec<i32> = (0..cfg.num_hidden_layers)
            .filter(|i| matches!(cfg.layer_kind(*i), AttnKind::Full))
            .collect();
        full_indices.sort();
        assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23, 27, 31]);
        // And exactly 24 linear:
        let linear_count = (0..cfg.num_hidden_layers)
            .filter(|i| matches!(cfg.layer_kind(*i), AttnKind::Linear))
            .count();
        assert_eq!(linear_count, 24);
    }
}
