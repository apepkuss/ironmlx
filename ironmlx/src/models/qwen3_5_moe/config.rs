//! Qwen3.5 MoE text-config parsing.

use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::core::Loader;
use crate::models::vision::VisionConfig;
use crate::nn::AttnKind;
use crate::Result;

use super::decoder_layer::DecoderLayerMoeConfig;
use super::mtp::Qwen35MoeMtpConfig;

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
fn default_norm_topk_prob() -> bool {
    true
}
fn default_mtp_num_hidden_layers() -> i32 {
    0
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
/// Note: `router_aux_loss_coef` is deliberately excluded from this struct.
/// It is a training-time field with no effect at inference. serde silently
/// ignores unknown fields by default, so extra fields in the snapshot
/// `text_config` are harmlessly skipped.
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
    #[serde(default = "default_mtp_num_hidden_layers")]
    pub mtp_num_hidden_layers: i32,
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
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    pub moe_intermediate_size: i32,
    pub shared_expert_intermediate_size: i32,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    /// Present in multimodal MoE variants; `None` for text-only.
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
}

impl Qwen35MoeConfig {
    /// Parse from a [`Loader`]'s `config.json`. Reads `config["text_config"]`.
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        Self::from_raw_config_value(loader.config_raw_value())
    }

    pub(crate) fn from_raw_config_value(raw: &serde_json::Value) -> Result<Self> {
        let text_config = raw
            .get("text_config")
            .ok_or_else(|| anyhow!("config.json missing text_config field"))?;
        let mut cfg: Qwen35MoeConfig = serde_json::from_value(text_config.clone())
            .context("failed to deserialize Qwen35MoeConfig from text_config")?;
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

    /// Returns attention path for `layer_idx` (0-based).
    /// Layer i is Full when `(i + 1) % full_attention_interval == 0`, else Linear.
    pub fn layer_kind(&self, layer_idx: i32) -> AttnKind {
        if (layer_idx + 1) % self.full_attention_interval == 0 {
            AttnKind::Full
        } else {
            AttnKind::Linear
        }
    }

    pub fn mtp_config(&self) -> Result<Qwen35MoeMtpConfig> {
        if self.mtp_num_hidden_layers <= 0 {
            return Err(anyhow!(
                "Qwen35MoeConfig::mtp_config: mtp_num_hidden_layers must be > 0, got {}",
                self.mtp_num_hidden_layers
            ));
        }
        let head_dim = self.effective_head_dim();
        Ok(Qwen35MoeMtpConfig {
            hidden_size: self.hidden_size,
            num_mtp_layers: self.mtp_num_hidden_layers,
            layer: DecoderLayerMoeConfig {
                hidden_size: self.hidden_size,
                num_heads: self.num_attention_heads,
                num_kv_heads: self.num_key_value_heads,
                head_dim,
                rms_norm_eps: self.rms_norm_eps,
                attention_bias: self.attention_bias,
                linear_num_value_heads: self.linear_num_value_heads,
                linear_num_key_heads: self.linear_num_key_heads,
                linear_key_head_dim: self.linear_key_head_dim,
                linear_value_head_dim: self.linear_value_head_dim,
                linear_conv_kernel_dim: self.linear_conv_kernel_dim,
                num_experts: self.num_experts,
                num_experts_per_tok: self.num_experts_per_tok,
                norm_topk_prob: self.norm_topk_prob,
            },
        })
    }

    pub fn ensure_mtp_compatible(&self, mtp: &Qwen35MoeConfig) -> Result<()> {
        macro_rules! check_eq {
            ($field:ident) => {
                if self.$field != mtp.$field {
                    return Err(anyhow!(
                        "Qwen35MoeConfig::ensure_mtp_compatible: {} mismatch target={} mtp={}",
                        stringify!($field),
                        self.$field,
                        mtp.$field
                    ));
                }
            };
        }

        check_eq!(hidden_size);
        check_eq!(num_attention_heads);
        check_eq!(num_key_value_heads);
        check_eq!(vocab_size);
        check_eq!(attention_bias);
        check_eq!(tie_word_embeddings);
        check_eq!(full_attention_interval);
        check_eq!(linear_num_value_heads);
        check_eq!(linear_num_key_heads);
        check_eq!(linear_key_head_dim);
        check_eq!(linear_value_head_dim);
        check_eq!(linear_conv_kernel_dim);
        check_eq!(num_experts);
        check_eq!(num_experts_per_tok);
        check_eq!(norm_topk_prob);
        check_eq!(moe_intermediate_size);
        check_eq!(shared_expert_intermediate_size);

        if self.effective_head_dim() != mtp.effective_head_dim() {
            return Err(anyhow!(
                "Qwen35MoeConfig::ensure_mtp_compatible: head_dim mismatch target={} mtp={}",
                self.effective_head_dim(),
                mtp.effective_head_dim()
            ));
        }
        if (self.rms_norm_eps - mtp.rms_norm_eps).abs() > f32::EPSILON {
            return Err(anyhow!(
                "Qwen35MoeConfig::ensure_mtp_compatible: rms_norm_eps mismatch target={} mtp={}",
                self.rms_norm_eps,
                mtp.rms_norm_eps
            ));
        }
        Ok(())
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

    fn realistic_vision_config_json() -> serde_json::Value {
        serde_json::json!({
            "depth": 27,
            "hidden_size": 1152,
            "num_heads": 16,
            "intermediate_size": 4304,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "num_position_embeddings": 2304
        })
    }

    #[test]
    fn parses_top_level_vision_config_from_raw_config() {
        let raw = serde_json::json!({
            "text_config": realistic_text_config_json(),
            "vision_config": realistic_vision_config_json()
        });

        let cfg = Qwen35MoeConfig::from_raw_config_value(&raw).expect("parse");
        let vc = cfg.vision_config.as_ref().expect("vision_config present");

        assert_eq!(vc.depth, 27);
        assert_eq!(vc.hidden_size, 1152);
        assert_eq!(vc.num_heads, 16);
        assert_eq!(vc.intermediate_size, 4304);
        assert_eq!(vc.out_hidden_size, 2048);
        assert_eq!(vc.patch_size, 16);
        assert_eq!(vc.spatial_merge_size, 2);
        assert_eq!(vc.temporal_patch_size, 2);
        assert_eq!(vc.in_channels, 3);
        assert_eq!(vc.num_position_embeddings, 2304);
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
    fn norm_topk_prob_defaults_true_when_absent() {
        let cfg: Qwen35MoeConfig =
            serde_json::from_value(realistic_text_config_json()).expect("parse");
        assert!(cfg.norm_topk_prob);
    }

    #[test]
    fn norm_topk_prob_parses_explicit_true() {
        let mut v = realistic_text_config_json();
        v["norm_topk_prob"] = serde_json::Value::Bool(true);
        let cfg: Qwen35MoeConfig = serde_json::from_value(v).expect("parse");
        assert!(cfg.norm_topk_prob);
    }

    #[test]
    fn norm_topk_prob_parses_explicit_false() {
        let mut v = realistic_text_config_json();
        v["norm_topk_prob"] = serde_json::Value::Bool(false);
        let cfg: Qwen35MoeConfig = serde_json::from_value(v).expect("parse");
        assert!(!cfg.norm_topk_prob);
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

    #[test]
    fn mtp_config_preserves_moe_layer_dimensions() {
        let mut v = realistic_text_config_json();
        v["mtp_num_hidden_layers"] = serde_json::json!(2);
        let cfg: Qwen35MoeConfig = serde_json::from_value(v).expect("parse");

        let mtp_cfg = cfg.mtp_config().expect("mtp config");

        assert_eq!(mtp_cfg.hidden_size, cfg.hidden_size);
        assert_eq!(mtp_cfg.num_mtp_layers, 2);
        assert_eq!(mtp_cfg.layer.hidden_size, cfg.hidden_size);
        assert_eq!(mtp_cfg.layer.num_heads, cfg.num_attention_heads);
        assert_eq!(mtp_cfg.layer.num_kv_heads, cfg.num_key_value_heads);
        assert_eq!(mtp_cfg.layer.num_experts, cfg.num_experts);
        assert_eq!(mtp_cfg.layer.num_experts_per_tok, cfg.num_experts_per_tok);
        assert_eq!(mtp_cfg.layer.norm_topk_prob, cfg.norm_topk_prob);
    }
}
