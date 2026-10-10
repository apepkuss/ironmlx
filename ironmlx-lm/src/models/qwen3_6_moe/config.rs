//! Qwen3.6 MoE checkpoint-identity detection and validation.
//!
//! This is stricter than execution-architecture detection: Qwen3.6 MoE uses
//! the Qwen3.5 MoE graph, but its released checkpoint has distinctive vision
//! metadata and 8-bit router-gate quantization overrides that are useful for
//! validation and regression tests.

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

/// Target bit widths whose DFlash2 verification has a dedicated qualification.
pub const QWEN36_MOE_DFLASH2_TARGET_BITS: [i64; 4] = [4, 5, 6, 8];

/// Return the default affine bit width of a Qwen3.6-35B-A3B checkpoint whose
/// architecture and per-module quantization coverage match a DFlash2-qualified
/// target exactly, or explain why it does not.
///
/// Qualification is keyed by the whole execution recipe, not by the default
/// bit width alone: every projection uses that width with affine group 64,
/// except the per-layer router `mlp.gate` and `mlp.shared_expert_gate`, which
/// are affine 8-bit group 64. Checkpoints with any other override (for
/// example mixed-precision OptiQ variants) execute different kernels and are
/// rejected even when their default width matches.
pub fn qwen36_moe_dflash2_target_bits(raw: &Value) -> Result<i32> {
    validate_qwen36_moe_config(raw)?;
    let text = raw
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("Qwen3.6 MoE config missing text_config object"))?;
    let int = |key: &str| text.get(key).and_then(Value::as_i64);
    for (key, expected) in [
        ("hidden_size", 2048),
        ("num_hidden_layers", 40),
        ("num_experts", 256),
        ("num_experts_per_tok", 8),
        ("moe_intermediate_size", 512),
        ("shared_expert_intermediate_size", 512),
        ("full_attention_interval", 4),
        ("head_dim", 256),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 2),
        ("linear_num_key_heads", 16),
        ("linear_num_value_heads", 32),
        ("linear_key_head_dim", 128),
        ("linear_value_head_dim", 128),
        ("linear_conv_kernel_dim", 4),
        ("vocab_size", 248320),
    ] {
        let actual = int(key);
        if actual != Some(expected) {
            return Err(anyhow!(
                "Qwen3.6 MoE DFlash2 target requires text_config.{key}={expected}, got {actual:?}"
            ));
        }
    }
    if text.get("dtype").and_then(Value::as_str) != Some("bfloat16") {
        return Err(anyhow!(
            "Qwen3.6 MoE DFlash2 target requires a bfloat16 checkpoint"
        ));
    }
    if text.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true) {
        return Err(anyhow!(
            "Qwen3.6 MoE DFlash2 target requires untied word embeddings"
        ));
    }

    let quant = raw
        .get("quantization")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("Qwen3.6 MoE DFlash2 target requires a quantization object"))?;
    if let Some(mirror) = raw.get("quantization_config") {
        if mirror.as_object() != Some(quant) {
            return Err(anyhow!(
                "Qwen3.6 MoE DFlash2 target quantization and quantization_config differ"
            ));
        }
    }
    let mode = quant
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("affine");
    if mode != "affine" {
        return Err(anyhow!(
            "Qwen3.6 MoE DFlash2 target requires affine quantization, got {mode}"
        ));
    }
    if quant.get("group_size").and_then(Value::as_i64) != Some(64) {
        return Err(anyhow!(
            "Qwen3.6 MoE DFlash2 target requires quantization group_size 64"
        ));
    }
    let bits = quant
        .get("bits")
        .and_then(Value::as_i64)
        .filter(|bits| QWEN36_MOE_DFLASH2_TARGET_BITS.contains(bits))
        .ok_or_else(|| {
            anyhow!(
                "Qwen3.6 MoE DFlash2 target requires affine bits in {QWEN36_MOE_DFLASH2_TARGET_BITS:?}, got {:?}",
                quant.get("bits")
            )
        })?;
    // Every router gate of every layer must carry exactly one override,
    // identified as the loader identifies it, so an alias of another layer's
    // gate or an out-of-range layer cannot stand in for a missing one.
    let mut gates = std::collections::HashSet::new();
    for (key, value) in quant {
        let Some(module) = value.as_object() else {
            if !matches!(key.as_str(), "bits" | "group_size" | "mode") {
                return Err(anyhow!(
                    "Qwen3.6 MoE DFlash2 target has unsupported quantization field {key}"
                ));
            }
            continue;
        };
        let mode_ok = module
            .get("mode")
            .and_then(Value::as_str)
            .is_none_or(|mode| mode == "affine");
        let extra_keys = module
            .keys()
            .any(|field| !matches!(field.as_str(), "bits" | "group_size" | "mode"));
        let identity = qwen36_gate_override_identity(key, QWEN36_MOE_DFLASH2_LAYERS);
        let Some(identity) =
            identity.filter(|_| is_qwen36_gate_override(key, value) && mode_ok && !extra_keys)
        else {
            return Err(anyhow!(
                "Qwen3.6 MoE DFlash2 target has an unqualified quantization override for {key}"
            ));
        };
        if !gates.insert(identity) {
            return Err(anyhow!(
                "Qwen3.6 MoE DFlash2 target has a duplicate router gate override for {key} (module {})",
                crate::core::loader::normalize_quant_prefix(key)
            ));
        }
    }
    let missing: Vec<String> = (0..QWEN36_MOE_DFLASH2_LAYERS)
        .flat_map(|layer| [(layer, false), (layer, true)])
        .filter(|identity| !gates.contains(identity))
        .map(|(layer, shared)| {
            format!(
                "model.layers.{layer}.mlp.{}",
                if shared { "shared_expert_gate" } else { "gate" }
            )
        })
        .collect();
    if !missing.is_empty() {
        return Err(anyhow!(
            "Qwen3.6 MoE DFlash2 target is missing router gate overrides for {}",
            missing.join(", ")
        ));
    }
    i32::try_from(bits).map_err(Into::into)
}

/// Decoder layers of a DFlash2-qualified Qwen3.6 MoE target.
const QWEN36_MOE_DFLASH2_LAYERS: usize = 40;

/// `(layer, shared_expert_gate)` of a router gate override, identified after
/// the loader's key normalization: exactly `model.layers.{layer}.mlp.gate`
/// or `model.layers.{layer}.mlp.shared_expert_gate` with a canonical decimal
/// layer below `layers`. Anything else is not a qualified gate override.
fn qwen36_gate_override_identity(key: &str, layers: usize) -> Option<(usize, bool)> {
    let module = crate::core::loader::normalize_quant_prefix(key);
    let (layer, path) = module.strip_prefix("model.layers.")?.split_once('.')?;
    let shared = match path {
        "mlp.gate" => false,
        "mlp.shared_expert_gate" => true,
        _ => return None,
    };
    let canonical = !layer.is_empty()
        && layer.bytes().all(|byte| byte.is_ascii_digit())
        && (layer == "0" || !layer.starts_with('0'));
    let layer: usize = layer.parse().ok().filter(|_| canonical)?;
    (layer < layers).then_some((layer, shared))
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

    fn dflash2_target_json(bits: i64) -> Value {
        let mut raw = raw_qwen36_config(40);
        raw["text_config"]["dtype"] = serde_json::json!("bfloat16");
        raw["text_config"]["head_dim"] = serde_json::json!(256);
        raw["quantization"]["bits"] = serde_json::json!(bits);
        raw["quantization"]["mode"] = serde_json::json!("affine");
        raw
    }

    #[test]
    fn dflash2_target_bits_require_exact_quantization_coverage() {
        for bits in [4, 5, 6, 8] {
            assert_eq!(
                qwen36_moe_dflash2_target_bits(&dflash2_target_json(bits)).unwrap(),
                bits as i32
            );
        }
        assert!(qwen36_moe_dflash2_target_bits(&dflash2_target_json(3)).is_err());

        let mut mixed = dflash2_target_json(4);
        mixed["quantization"]["language_model.model.layers.0.linear_attn.in_proj_qkv"] =
            serde_json::json!({"bits": 8, "group_size": 64});
        let error = qwen36_moe_dflash2_target_bits(&mixed).unwrap_err();
        assert!(error
            .to_string()
            .contains("unqualified quantization override"));

        let mut group32 = dflash2_target_json(4);
        group32["quantization"]["group_size"] = serde_json::json!(32);
        assert!(qwen36_moe_dflash2_target_bits(&group32).is_err());

        let mut mxfp4 = dflash2_target_json(4);
        mxfp4["quantization"]["mode"] = serde_json::json!("mxfp4");
        assert!(qwen36_moe_dflash2_target_bits(&mxfp4).is_err());

        let mut wider = dflash2_target_json(4);
        wider["text_config"]["hidden_size"] = serde_json::json!(4096);
        assert!(qwen36_moe_dflash2_target_bits(&wider).is_err());
    }

    /// Router gate coverage is checked per module identity under the
    /// loader's key normalization, not by key shape and count.
    #[test]
    fn dflash2_target_bits_require_every_layer_gate_by_module_identity() {
        let gate = |layer: &str| format!("language_model.model.layers.{layer}.mlp.gate");
        let replace_layer0_gate = |key: &str| {
            let mut raw = dflash2_target_json(4);
            let quant = raw["quantization"].as_object_mut().expect("quantization");
            quant.remove(&gate("0")).expect("layer 0 gate");
            quant.insert(
                key.to_owned(),
                serde_json::json!({"bits": 8, "group_size": 64}),
            );
            raw
        };
        let rejected = |raw: &Value, needle: &str| {
            let error = qwen36_moe_dflash2_target_bits(raw).expect_err("must be rejected");
            assert!(error.to_string().contains(needle), "{error}");
        };

        // Both spellings the loader accepts qualify.
        let mut plain = dflash2_target_json(6);
        let quant = plain["quantization"].as_object_mut().expect("quantization");
        for layer in 0..40 {
            for module in ["gate", "shared_expert_gate"] {
                let value = quant
                    .remove(&format!("language_model.model.layers.{layer}.mlp.{module}"))
                    .expect("override");
                quant.insert(format!("model.layers.{layer}.mlp.{module}"), value);
            }
        }
        assert_eq!(qwen36_moe_dflash2_target_bits(&plain).unwrap(), 6);

        // Missing coverage (rejected already by the structural gate count).
        let mut missing = dflash2_target_json(4);
        missing["quantization"]
            .as_object_mut()
            .expect("quantization")
            .remove("language_model.model.layers.17.mlp.shared_expert_gate");
        rejected(&missing, "expected 80 gate quant overrides, found 79");

        // Aliases of an existing module (the loader strips `language_model.`).
        rejected(
            &replace_layer0_gate("model.layers.1.mlp.gate"),
            "duplicate router gate override",
        );
        rejected(
            &replace_layer0_gate("model.layers.1.mlp.shared_expert_gate"),
            "duplicate router gate override",
        );
        // The same alias with the canonical key removed is the plain-form
        // spelling of that module and qualifies.
        let mut respelled = replace_layer0_gate("model.layers.0.mlp.gate");
        assert_eq!(qwen36_moe_dflash2_target_bits(&respelled).unwrap(), 4);
        respelled["quantization"]
            .as_object_mut()
            .expect("quantization")
            .insert(
                "language_model.model.layers.0.mlp.gate".to_owned(),
                serde_json::json!({"bits": 8, "group_size": 64}),
            );
        rejected(&respelled, "expected 80 gate quant overrides, found 81");

        // Out-of-range, non-canonical and unsupported module paths.
        for key in [
            gate("999"),
            gate("40"),
            gate("00"),
            gate("01"),
            gate("+0"),
            gate("-1"),
            gate("x"),
            gate(""),
            "language_model.language_model.model.layers.0.mlp.gate".to_owned(),
            "model.language_model.layers.0.mlp.gate".to_owned(),
            "language_model.model.layers.0.mlp.gate.weight".to_owned(),
            "language_model.model.layers.0.mlp.experts.gate".to_owned(),
            "language_model.model.layers.0.mlp.shared_expert.gate".to_owned(),
            "language_model.model.layers.0.self_attn.gate".to_owned(),
        ] {
            // Keys that do not even look like a gate already fail the
            // structural gate count; the rest fail module identification.
            let error = qwen36_moe_dflash2_target_bits(&replace_layer0_gate(&key))
                .expect_err("must be rejected")
                .to_string();
            assert!(
                error.contains("unqualified quantization override")
                    || error.contains("expected 80 gate quant overrides, found 79"),
                "{key}: {error}"
            );
        }

        // The existing value checks still apply to a correctly named gate.
        let mut wrong_bits = dflash2_target_json(4);
        wrong_bits["quantization"][gate("3")] = serde_json::json!({"bits": 4, "group_size": 64});
        rejected(&wrong_bits, "expected 80 gate quant overrides, found 79");
        let mut extra_field = dflash2_target_json(4);
        extra_field["quantization"][gate("3")] =
            serde_json::json!({"bits": 8, "group_size": 64, "scale": 1});
        rejected(&extra_field, "unqualified quantization override");
    }

    /// The real mlx-community checkpoints still qualify (CPU only).
    /// `QWEN36_MOE_DFLASH2_CONFIGS` lists `bits=path/to/config.json` pairs.
    #[test]
    #[ignore = "reads local Qwen3.6 MoE checkpoint configs"]
    fn dflash2_target_bits_accept_real_checkpoint_configs() {
        let Ok(list) = std::env::var("QWEN36_MOE_DFLASH2_CONFIGS") else {
            eprintln!("skip: set QWEN36_MOE_DFLASH2_CONFIGS");
            return;
        };
        for entry in list.split(',') {
            let (bits, path) = entry.split_once('=').expect("bits=path");
            let raw: Value =
                serde_json::from_str(&std::fs::read_to_string(path).expect("read config"))
                    .expect("parse config");
            assert_eq!(
                qwen36_moe_dflash2_target_bits(&raw).expect(path),
                bits.parse::<i32>().expect("bits"),
                "{path}"
            );
            eprintln!("affine{bits} {path}: qualified");
        }
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
