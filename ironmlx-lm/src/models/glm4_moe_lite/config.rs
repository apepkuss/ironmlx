//! GLM-4.7-Flash (`glm4_moe_lite`) config parsing.
//!
//! GLM uses a FLAT top-level config (unlike Qwen's `config["text_config"]`),
//! so the whole `config.json` value is deserialized directly.

use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::core::Loader;
use crate::Result;

fn default_norm_topk_prob() -> bool {
    true
}
fn default_topk_method() -> String {
    "noaux_tc".to_string()
}
fn default_n_group() -> i32 {
    1
}
fn default_topk_group() -> i32 {
    1
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_partial_rotary_factor() -> f32 {
    1.0
}
fn default_max_position_embeddings() -> i32 {
    202_752
}

/// Subset of `config.json` for GLM-4.7-Flash (`glm4_moe_lite`) inference.
///
/// DeepSeek-style absorbed-MLA attention + noaux_tc sigmoid router + ungated
/// shared expert; layer 0 dense, layers `>= first_k_dense_replace` MoE.
///
/// serde silently ignores unknown fields, so training-time / unused keys in the
/// snapshot are harmlessly skipped.
#[derive(Debug, Clone, Deserialize)]
pub struct Glm4MoeLiteConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub first_k_dense_replace: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,

    // ─ MLA latent / per-head dims ─
    pub q_lora_rank: i32,
    pub kv_lora_rank: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,

    // ─ MoE ─
    pub n_routed_experts: i32,
    pub num_experts_per_tok: i32,
    pub n_shared_experts: i32,
    pub moe_intermediate_size: i32,
    /// Dense (layer-0) MLP intermediate size.
    pub intermediate_size: i32,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    #[serde(default = "default_topk_method")]
    pub topk_method: String,
    #[serde(default = "default_n_group")]
    pub n_group: i32,
    #[serde(default = "default_topk_group")]
    pub topk_group: i32,

    // ─ RoPE / misc ─
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// Long-context RoPE scaling block. GLM-4.7-Flash ships `null`; a non-null
    /// value would change the `1/sqrt(q_head_dim)` softmax scale (mscale) and is
    /// rejected by [`Glm4MoeLiteConfig::validate`].
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

impl Glm4MoeLiteConfig {
    /// Parse + validate from a raw `config.json` string.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let cfg: Glm4MoeLiteConfig =
            serde_json::from_str(s).context("failed to deserialize Glm4MoeLiteConfig")?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse + validate from a [`Loader`]'s `config.json`.
    ///
    /// GLM uses a FLAT top-level config (unlike Qwen's `config["text_config"]`)
    /// — deserialize the whole value.
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg: Glm4MoeLiteConfig = serde_json::from_value(loader.config_raw_value().clone())
            .context("failed to deserialize Glm4MoeLiteConfig from config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject configurations this implementation does not support.
    pub fn validate(&self) -> Result<()> {
        if self.topk_method != "noaux_tc" {
            return Err(anyhow!(
                "glm4_moe_lite: only topk_method=\"noaux_tc\" supported; got {:?}",
                self.topk_method
            ));
        }
        if self.n_group != 1 || self.topk_group != 1 {
            return Err(anyhow!(
                "glm4_moe_lite: only n_group=1 and topk_group=1 supported; got n_group={}, topk_group={}",
                self.n_group,
                self.topk_group
            ));
        }
        if self.rope_scaling.is_some() {
            return Err(anyhow!(
                "glm4_moe_lite: rope_scaling must be null (YaRN/mscale not supported); got {:?}",
                self.rope_scaling
            ));
        }
        Ok(())
    }

    /// Per-head query/key dim = nope + rope (= 192 + 64 = 256).
    pub fn q_head_dim(&self) -> i32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Attention softmax scale = `1 / sqrt(q_head_dim)` (DeepSeek-V2 Eq 18;
    /// `rope_scaling=null` ⇒ no YaRN/mscale factor).
    pub fn softmax_scale(&self) -> f32 {
        1.0 / (self.q_head_dim() as f32).sqrt()
    }

    /// Layer `i` is MoE when `i >= first_k_dense_replace`, else dense.
    pub fn is_moe_layer(&self, i: i32) -> bool {
        i >= self.first_k_dense_replace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const RAW: &str = r#"{ "model_type":"glm4_moe_lite","hidden_size":2048,"num_hidden_layers":47,
      "first_k_dense_replace":1,"num_attention_heads":20,"num_key_value_heads":20,"q_lora_rank":768,
      "kv_lora_rank":512,"qk_nope_head_dim":192,"qk_rope_head_dim":64,"v_head_dim":256,
      "n_routed_experts":64,"num_experts_per_tok":4,"n_shared_experts":1,"moe_intermediate_size":1536,
      "intermediate_size":10240,"norm_topk_prob":true,"routed_scaling_factor":1.8,"topk_method":"noaux_tc",
      "n_group":1,"topk_group":1,"rope_theta":1000000,"partial_rotary_factor":1.0,"rms_norm_eps":1e-5,
      "vocab_size":154880,"tie_word_embeddings":false,"max_position_embeddings":202752,"rope_scaling":null }"#;
    #[test]
    fn parses_and_validates() {
        let c = Glm4MoeLiteConfig::from_json_str(RAW).unwrap();
        assert_eq!(
            (c.hidden_size, c.num_hidden_layers, c.first_k_dense_replace),
            (2048, 47, 1)
        );
        assert_eq!(
            (
                c.q_lora_rank,
                c.kv_lora_rank,
                c.qk_nope_head_dim,
                c.qk_rope_head_dim,
                c.v_head_dim
            ),
            (768, 512, 192, 64, 256)
        );
        assert_eq!(c.q_head_dim(), 256);
        assert_eq!(c.softmax_scale(), 1.0 / 16.0);
        assert_eq!((c.n_routed_experts, c.num_experts_per_tok), (64, 4));
        assert!(c.norm_topk_prob);
        assert_eq!(c.routed_scaling_factor, 1.8);
        assert!(c.is_moe_layer(1) && !c.is_moe_layer(0));
    }
    #[test]
    fn rejects_grouped_routing() {
        assert!(
            Glm4MoeLiteConfig::from_json_str(&RAW.replace("\"n_group\":1", "\"n_group\":8"))
                .is_err()
        );
    }
    #[test]
    fn rejects_non_noaux_tc() {
        assert!(Glm4MoeLiteConfig::from_json_str(&RAW.replace("noaux_tc", "greedy")).is_err());
    }
    #[test]
    fn rejects_rope_scaling() {
        assert!(Glm4MoeLiteConfig::from_json_str(
            &RAW.replace("\"rope_scaling\":null", "\"rope_scaling\":{\"factor\":2.0}")
        )
        .is_err());
    }
}
