//! MiniCPM-V-4.6 `text_config` → [`Qwen35Config`] adapter (text-only backbone).
//!
//! MiniCPM-V-4.6 declares `model_type = "minicpmv4_6"` and nests a
//! `text_config` with `model_type = "qwen3_5_text"`. That backbone is
//! Qwen3.5-text verbatim — same hybrid gated-delta + gated-full attention,
//! same `language_model.*` tensor layout (stripped to `model.*` by
//! [`crate::core::Loader`] sanitize), same offset-gamma RMSNorm convention
//! (detected via `text_config.model_type`). The only two text-path
//! divergences from a native Qwen3.5 checkpoint are handled here:
//!
//!   1. `text_config.rope_parameters` omits `mrope_section`. For text-only
//!      inference the three MRoPE position streams carry identical sequential
//!      positions, so any partition summing to `rot_dim / 2` is numerically a
//!      no-op (equivalent to 1-D RoPE). We default to the Qwen3.5 family value
//!      `[11, 11, 10]`, matching the mlx-vlm `TextConfig` default.
//!   2. The top-level `vision_config` is a SigLIP encoder (`minicpmv4_6_vision`)
//!      whose schema is incompatible with the Qwen3.5-VL NaViT
//!      [`VisionConfig`](crate::models::qwen3_5::VisionConfig). The SigLIP
//!      tower + resampler are not yet implemented, so it is skipped
//!      (`vision_config = None`) — image inputs are out of scope.

use anyhow::{anyhow, Context};
use serde_json::Value;

use crate::core::Loader;
use crate::models::Qwen35Config;
use crate::Result;

/// Qwen3.5 family default MRoPE section (sums to 32 = `rot_dim / 2` for
/// `head_dim = 256`, `partial_rotary_factor = 0.25`). MiniCPM-V-4.6 omits
/// `mrope_section`; for the text-only path this value is numerically
/// irrelevant (identical position streams), but matching the family default
/// keeps the rotary tables identical to a native Qwen3.5 run.
const DEFAULT_MROPE_SECTION: [i32; 3] = [11, 11, 10];

/// Parse a MiniCPM-V-4.6 checkpoint's `text_config` into a [`Qwen35Config`]
/// suitable for the text-only Qwen3.5 dense execution graph.
pub fn text_config_from_loader(loader: &Loader) -> Result<Qwen35Config> {
    text_config_from_raw(loader.config_raw_value())
}

/// Core adapter over a raw `config.json` value (test seam).
pub(crate) fn text_config_from_raw(raw: &Value) -> Result<Qwen35Config> {
    let model_type = raw
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("MiniCPM-V-4.6 config missing model_type"))?;
    if model_type != "minicpmv4_6" {
        return Err(anyhow!(
            "MiniCPM-V-4.6 config expected model_type=minicpmv4_6, got {model_type}"
        ));
    }

    let text_config = raw
        .get("text_config")
        .ok_or_else(|| anyhow!("MiniCPM-V-4.6 config missing text_config field"))?;
    let mut cfg: Qwen35Config = serde_json::from_value(text_config.clone())
        .context("failed to deserialize Qwen35Config from MiniCPM-V-4.6 text_config")?;

    // Text-only feasibility: the SigLIP vision tower is not implemented. Drop
    // any vision_config so the shared Qwen3.5 builder takes the text-only path.
    cfg.vision_config = None;

    // MiniCPM-V-4.6 omits mrope_section — default to the Qwen3.5 family value.
    if cfg.rope_parameters.mrope_section.is_empty() {
        cfg.rope_parameters.mrope_section = DEFAULT_MROPE_SECTION.to_vec();
    }

    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real `text_config` subset from mlx-community/MiniCPM-V-4.6-4bit, with the
    /// SigLIP `vision_config` it actually ships at the top level.
    fn raw_minicpmv46_config() -> Value {
        serde_json::json!({
            "model_type": "minicpmv4_6",
            "architectures": ["MiniCPMV4_6ForConditionalGeneration"],
            "image_token_id": 248056,
            "text_config": {
                "model_type": "qwen3_5_text",
                "attention_bias": false,
                "attn_output_gate": true,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_size": 1024,
                "intermediate_size": 3584,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 128,
                "max_position_embeddings": 262144,
                "mtp_num_hidden_layers": 1,
                "num_attention_heads": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default"
                },
                "tie_word_embeddings": true,
                "vocab_size": 248094
            },
            "vision_config": {
                "model_type": "minicpmv4_6_vision",
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "patch_size": 14,
                "image_size": 980
            }
        })
    }

    #[test]
    fn parses_text_config_into_qwen35_config() {
        let cfg = text_config_from_raw(&raw_minicpmv46_config()).expect("parse");
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, Some(256));
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.linear_num_value_heads, 16);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.vocab_size, 248094);
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert!((cfg.rope_parameters.rope_theta - 10_000_000.0).abs() < 1.0);
        assert!((cfg.rope_parameters.partial_rotary_factor - 0.25).abs() < 1e-6);
    }

    #[test]
    fn defaults_empty_mrope_section_to_qwen35_family_value() {
        let cfg = text_config_from_raw(&raw_minicpmv46_config()).expect("parse");
        assert_eq!(cfg.rope_parameters.mrope_section, vec![11, 11, 10]);
    }

    #[test]
    fn skips_incompatible_siglip_vision_config() {
        // The raw config carries a SigLIP vision_config that would fail the
        // Qwen3.5-VL NaViT VisionConfig schema; the adapter must drop it rather
        // than choke on it.
        let cfg = text_config_from_raw(&raw_minicpmv46_config()).expect("parse");
        assert!(cfg.vision_config.is_none());
    }

    #[test]
    fn preserves_explicit_mrope_section_when_present() {
        let mut raw = raw_minicpmv46_config();
        raw["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([8, 8, 16]);
        let cfg = text_config_from_raw(&raw).expect("parse");
        assert_eq!(cfg.rope_parameters.mrope_section, vec![8, 8, 16]);
    }

    #[test]
    fn rejects_non_minicpmv46_model_type() {
        let mut raw = raw_minicpmv46_config();
        raw["model_type"] = serde_json::json!("qwen3_5");
        let err = text_config_from_raw(&raw).unwrap_err();
        assert!(
            err.to_string().contains("expected model_type=minicpmv4_6"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn errors_when_text_config_missing() {
        let raw = serde_json::json!({ "model_type": "minicpmv4_6" });
        let err = text_config_from_raw(&raw).unwrap_err();
        assert!(
            err.to_string().contains("missing text_config"),
            "unexpected error: {err}"
        );
    }
}
