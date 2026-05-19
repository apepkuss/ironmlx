//! Qwen3.5 MoE text-config parsing.

use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::core::Loader;
use crate::nn::AttnKind;
use crate::Result;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub mrope_section: Vec<i32>,
}

fn default_partial_rotary_factor() -> f32 {
    0.25
}
fn default_rope_theta() -> f32 {
    10_000_000.0
}
fn default_max_position_embeddings() -> i32 {
    32768
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

/// Subset of `config.json["text_config"]` for Qwen3.5 MoE inference.
///
/// Note: `norm_topk_prob` is NOT included — mlx-vlm reference always
/// renormalizes top-k probabilities regardless of this flag (T0 research
/// confirmed). `router_aux_loss_coef` is also omitted (inference-time
/// ignored).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35MoeConfig {
    // ─ Dense-shared fields (same names as Qwen35Config) ─
    pub hidden_size: i32,
    /// Dense MLP intermediate size. Present in some config.json variants but
    /// absent from the real Qwen3.5-35B-A3B-4bit snapshot (which uses
    /// `moe_intermediate_size` / `shared_expert_intermediate_size` instead).
    /// Unused at inference time; kept for forward-compatibility.
    #[serde(default)]
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub full_attention_interval: i32,
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
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,

    // ─ MoE-specific fields ─
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub moe_intermediate_size: i32,
    pub shared_expert_intermediate_size: i32,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
}

impl Qwen35MoeConfig {
    /// Parse from a [`Loader`]'s `config.json`. Reads `config["text_config"]`.
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let raw = loader.config_raw_value();
        let text_config = raw
            .get("text_config")
            .ok_or_else(|| anyhow!("config.json missing text_config field"))?;
        let cfg: Qwen35MoeConfig = serde_json::from_value(text_config.clone())
            .context("failed to deserialize Qwen35MoeConfig from text_config")?;
        Ok(cfg)
    }

    /// Effective per-head dim: `head_dim` if specified, else `hidden_size / num_attention_heads`.
    pub fn effective_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Returns attention path for `layer_idx` (0-based).
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

    /// Subset of mlx-community/Qwen3.5-35B-A3B-4bit text_config (verified via T0 snapshot).
    fn realistic_text_config_json() -> serde_json::Value {
        serde_json::json!({
            "attention_bias": false,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 2048,
            "intermediate_size": 512,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0
            },
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320
        })
    }

    #[test]
    fn parses_35b_a3b_text_config() {
        let v = realistic_text_config_json();
        let cfg: Qwen35MoeConfig = serde_json::from_value(v).expect("parse");
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.moe_intermediate_size, 512);
        assert_eq!(cfg.shared_expert_intermediate_size, 512);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert!(cfg.mlp_only_layers.is_empty());
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.head_dim, Some(256));
    }

    #[test]
    fn layer_kind_partition_full_attention_interval_4() {
        let cfg: Qwen35MoeConfig = serde_json::from_value(realistic_text_config_json()).unwrap();
        // 40 layers, interval=4 -> Full at idx {3,7,11,15,19,23,27,31,35,39} (10 of them)
        let full_count = (0..cfg.num_hidden_layers)
            .filter(|i| matches!(cfg.layer_kind(*i), AttnKind::Full))
            .count();
        assert_eq!(full_count, 10);
        let linear_count = (0..cfg.num_hidden_layers)
            .filter(|i| matches!(cfg.layer_kind(*i), AttnKind::Linear))
            .count();
        assert_eq!(linear_count, 30);
    }
}
