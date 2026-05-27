//! Qwen3.6 MoE config detection and validation.

use std::ops::Deref;

use anyhow::{anyhow, Context};
use serde_json::Value;

use crate::core::Loader;
use crate::models::qwen3_5_moe::Qwen35MoeConfig;
use crate::Result;

#[derive(Debug, Clone)]
pub struct Qwen36MoeConfig {
    inner: Qwen35MoeConfig,
}

impl Qwen36MoeConfig {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        Self::from_raw_config_value(loader.config_raw_value())
    }

    pub(crate) fn from_raw_config_value(raw: &Value) -> Result<Self> {
        validate_qwen36_moe_config(raw)?;
        let inner = Qwen35MoeConfig::from_raw_config_value(raw)
            .context("failed to parse Qwen3.6 MoE text/vision config")?;
        Ok(Self { inner })
    }

    pub fn as_qwen35_moe_config(&self) -> &Qwen35MoeConfig {
        &self.inner
    }

    pub fn into_qwen35_moe_config(self) -> Qwen35MoeConfig {
        self.inner
    }

    pub fn is_qwen36_moe_config(raw: &Value) -> bool {
        is_qwen36_moe_config(raw)
    }

    #[cfg(test)]
    pub(crate) fn from_inner_for_test(inner: Qwen35MoeConfig) -> Self {
        Self { inner }
    }
}

impl Deref for Qwen36MoeConfig {
    type Target = Qwen35MoeConfig;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub fn is_qwen36_moe_config(raw: &Value) -> bool {
    validate_qwen36_moe_config(raw).is_ok()
}

fn validate_qwen36_moe_config(raw: &Value) -> Result<()> {
    let model_type = raw
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("Qwen3.6 MoE config missing model_type"))?;
    if model_type != "qwen3_5_moe" {
        return Err(anyhow!(
            "Qwen3.6 MoE config expected model_type=qwen3_5_moe, got {model_type}"
        ));
    }

    let arch_ok = raw
        .get("architectures")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .any(|v| v.as_str() == Some("Qwen3_5MoeForConditionalGeneration"))
        })
        .unwrap_or(false);
    if !arch_ok {
        return Err(anyhow!(
            "Qwen3.6 MoE config missing Qwen3_5MoeForConditionalGeneration architecture marker"
        ));
    }

    let text = raw
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("Qwen3.6 MoE config missing text_config object"))?;
    let num_layers = text
        .get("num_hidden_layers")
        .and_then(Value::as_i64)
        .ok_or_else(|| anyhow!("Qwen3.6 MoE config missing text_config.num_hidden_layers"))?;
    let num_experts = text
        .get("num_experts")
        .and_then(Value::as_i64)
        .ok_or_else(|| anyhow!("Qwen3.6 MoE config missing text_config.num_experts"))?;
    let experts_per_tok = text
        .get("num_experts_per_tok")
        .and_then(Value::as_i64)
        .ok_or_else(|| anyhow!("Qwen3.6 MoE config missing text_config.num_experts_per_tok"))?;
    if num_layers <= 0 || num_experts <= 1 || experts_per_tok <= 0 {
        return Err(anyhow!(
            "Qwen3.6 MoE config has invalid MoE dimensions: layers={num_layers}, \
             num_experts={num_experts}, num_experts_per_tok={experts_per_tok}"
        ));
    }

    raw.get("vision_config")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("Qwen3.6 MoE config missing top-level vision_config object"))?;
    raw.get("image_token_id")
        .and_then(Value::as_i64)
        .ok_or_else(|| anyhow!("Qwen3.6 MoE config missing image_token_id"))?;

    let expected_override_count = (num_layers as usize) * 2;
    let override_count = qwen36_gate_quant_override_count(raw)?;
    if override_count != expected_override_count {
        return Err(anyhow!(
            "Qwen3.6 MoE config expected {expected_override_count} gate quant overrides, \
             found {override_count}"
        ));
    }

    Ok(())
}

fn qwen36_gate_quant_override_count(raw: &Value) -> Result<usize> {
    let quant = raw
        .get("quantization")
        .or_else(|| raw.get("quantization_config"))
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("Qwen3.6 MoE config missing quantization object"))?;

    Ok(quant
        .iter()
        .filter(|(key, value)| is_qwen36_gate_override(key, value))
        .count())
}

fn is_qwen36_gate_override(key: &str, value: &Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    let bits_ok = obj.get("bits").and_then(Value::as_i64) == Some(8);
    let group_ok = obj.get("group_size").and_then(Value::as_i64) == Some(64);
    let key_ok = (key.starts_with("language_model.model.layers.")
        || key.starts_with("model.layers."))
        && (key.ends_with(".mlp.gate") || key.ends_with(".mlp.shared_expert_gate"));
    bits_ok && group_ok && key_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_config_json(num_hidden_layers: i32) -> Value {
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
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0
            },
            "shared_expert_intermediate_size": 512,
            "tie_word_embeddings": false,
            "vocab_size": 248320
        })
    }

    fn vision_config_json() -> Value {
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

    fn qwen36_quant_json(num_hidden_layers: i32) -> Value {
        let mut quant = serde_json::Map::new();
        quant.insert("bits".to_owned(), serde_json::json!(4));
        quant.insert("group_size".to_owned(), serde_json::json!(64));
        quant.insert("mode".to_owned(), serde_json::json!("affine"));
        for layer in 0..num_hidden_layers {
            quant.insert(
                format!("language_model.model.layers.{layer}.mlp.gate"),
                serde_json::json!({"bits": 8, "group_size": 64}),
            );
            quant.insert(
                format!("language_model.model.layers.{layer}.mlp.shared_expert_gate"),
                serde_json::json!({"bits": 8, "group_size": 64}),
            );
        }
        Value::Object(quant)
    }

    fn raw_qwen36_config(num_hidden_layers: i32) -> Value {
        serde_json::json!({
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "image_token_id": 248056,
            "text_config": text_config_json(num_hidden_layers),
            "vision_config": vision_config_json(),
            "quantization": qwen36_quant_json(num_hidden_layers)
        })
    }

    #[test]
    fn detects_structural_qwen36_moe_config() {
        let raw = raw_qwen36_config(2);
        assert!(is_qwen36_moe_config(&raw));
    }

    #[test]
    fn rejects_moe_config_without_qwen36_gate_quant_overrides() {
        let mut raw = raw_qwen36_config(2);
        raw["quantization"] = serde_json::json!({"bits": 4, "group_size": 64, "mode": "affine"});
        assert!(!is_qwen36_moe_config(&raw));
    }

    #[test]
    fn parses_inner_qwen35_moe_config_after_qwen36_validation() {
        let raw = raw_qwen36_config(2);
        let cfg = Qwen36MoeConfig::from_raw_config_value(&raw).expect("parse");
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert!(cfg.vision_config.is_some());
    }
}
