//! Standard Llama-family config parsing.
//!
//! Targets `model_type = "llama"` GQA dense checkpoints (MiniCPM5-1B and other
//! `LlamaForCausalLM` exports). FLAT top-level config (no `text_config`
//! nesting); serde silently ignores unknown keys, so quantization / tokenizer
//! keys in the snapshot are harmlessly skipped.

use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::core::Loader;
use crate::Result;

fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_max_position_embeddings() -> i32 {
    4096
}

/// Subset of a standard Llama `config.json` that drives GQA dense inference.
///
/// Plain GQA + RoPE + SwiGLU + RMSNorm with a separate `lm_head`
/// (`tie_word_embeddings = false` for MiniCPM5-1B). No MoE / MLA /
/// sliding-window / Q-K norm / attention bias.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    /// `None` in raw config → derived from `hidden_size / num_attention_heads`.
    /// MiniCPM5-1B sets it explicitly to 128 (≠ 1536/16 = 96), so the declared
    /// value MUST be honored rather than recomputed.
    #[serde(default)]
    pub head_dim: Option<i32>,
    pub vocab_size: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// Llama 3.1/3.2's exact frequency scaling contract. MiniCPM5-1B ships
    /// `null`; other RoPE scaling families remain unsupported.
    #[serde(default)]
    pub rope_scaling: Option<Llama3RopeScaling>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Llama3RopeScaling {
    pub factor: f32,
    pub high_freq_factor: f32,
    pub low_freq_factor: f32,
    pub original_max_position_embeddings: i32,
    pub rope_type: String,
}

impl LlamaConfig {
    /// Parse + validate from a raw `config.json` string.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let cfg: LlamaConfig =
            serde_json::from_str(s).context("failed to deserialize LlamaConfig")?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse + validate from a [`Loader`]'s `config.json` (flat top-level).
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg: LlamaConfig = serde_json::from_value(loader.config_raw_value().clone())
            .context("failed to deserialize LlamaConfig from config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Effective per-head dim: `head_dim` if declared, else
    /// `hidden_size / num_attention_heads`.
    pub fn effective_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Reject configurations this implementation does not support.
    pub fn validate(&self) -> Result<()> {
        if let Some(scaling) = &self.rope_scaling {
            if scaling.rope_type != "llama3" {
                return Err(anyhow!(
                    "llama: unsupported rope_scaling.rope_type `{}`; only `llama3` is supported",
                    scaling.rope_type
                ));
            }
            if !scaling.factor.is_finite()
                || scaling.factor < 1.0
                || !scaling.low_freq_factor.is_finite()
                || scaling.low_freq_factor <= 0.0
                || !scaling.high_freq_factor.is_finite()
                || scaling.high_freq_factor <= scaling.low_freq_factor
                || scaling.original_max_position_embeddings <= 0
            {
                return Err(anyhow!("llama: invalid llama3 rope_scaling parameters"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MiniCPM5-1B-8bit config.json (structural fields only).
    const MINICPM5: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 1536,
        "num_hidden_layers": 24,
        "intermediate_size": 4608,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 130560,
        "rms_norm_eps": 1e-06,
        "rope_theta": 5000000,
        "rope_scaling": null,
        "tie_word_embeddings": false,
        "max_position_embeddings": 131072,
        "quantization": { "group_size": 64, "bits": 8, "mode": "affine" }
    }"#;

    #[test]
    fn parses_minicpm5_structural_fields() {
        let c = LlamaConfig::from_json_str(MINICPM5).unwrap();
        assert_eq!(c.hidden_size, 1536);
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.intermediate_size, 4608);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, 2);
        assert_eq!(c.vocab_size, 130560);
        assert_eq!(c.rope_theta, 5_000_000.0);
        assert!(!c.tie_word_embeddings);
        assert_eq!(c.max_position_embeddings, 131072);
    }

    #[test]
    fn honors_explicit_head_dim_not_derived() {
        // head_dim=128 is declared and MUST win over hidden/heads = 1536/16 = 96.
        let c = LlamaConfig::from_json_str(MINICPM5).unwrap();
        assert_eq!(c.head_dim, Some(128));
        assert_eq!(c.effective_head_dim(), 128);
        // q_proj out_features = num_heads * head_dim = 16 * 128 = 2048.
        assert_eq!(c.num_attention_heads * c.effective_head_dim(), 2048);
    }

    #[test]
    fn derives_head_dim_when_absent() {
        let raw = MINICPM5.replace("\"head_dim\": 128,", "");
        let c = LlamaConfig::from_json_str(&raw).unwrap();
        assert_eq!(c.head_dim, None);
        assert_eq!(c.effective_head_dim(), 1536 / 16); // 96
    }

    #[test]
    fn accepts_exact_llama3_rope_scaling_and_rejects_other_families() {
        let llama3 = MINICPM5.replace(
            "\"rope_scaling\": null,",
            concat!(
                "\"rope_scaling\": {",
                "\"factor\": 32.0,",
                "\"high_freq_factor\": 4.0,",
                "\"low_freq_factor\": 1.0,",
                "\"original_max_position_embeddings\": 8192,",
                "\"rope_type\": \"llama3\"},"
            ),
        );
        let parsed = LlamaConfig::from_json_str(&llama3).unwrap();
        assert_eq!(parsed.rope_scaling.unwrap().factor, 32.0);

        let raw = MINICPM5.replace(
            "\"rope_scaling\": null,",
            concat!(
                "\"rope_scaling\": {",
                "\"factor\": 2.0,",
                "\"high_freq_factor\": 4.0,",
                "\"low_freq_factor\": 1.0,",
                "\"original_max_position_embeddings\": 8192,",
                "\"rope_type\": \"longrope\"},"
            ),
        );
        assert!(LlamaConfig::from_json_str(&raw).is_err());
    }
}
