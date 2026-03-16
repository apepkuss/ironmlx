mod config;
mod llama;
mod loader;
mod qwen3;
mod qwen35;

pub use config::{ModelConfig, Qwen35Config};
pub use llama::{LlamaModel, TransformerBlock};
pub use loader::load_model_weights;
pub use qwen35::Qwen35Model;

use crate::array::Array;
use crate::error::Result;
use std::collections::HashMap;

/// Unified model type that can hold any supported architecture.
pub enum Model {
    /// Llama, Qwen3, and similar decoder-only models.
    Standard(LlamaModel),
    /// Qwen3.5 with mixed GatedDeltaNet + FullAttention layers.
    Qwen35(Qwen35Model),
}

impl Model {
    /// Number of layers in the model.
    pub fn num_layers(&self) -> usize {
        match self {
            Model::Standard(m) => m.layers.len(),
            Model::Qwen35(m) => m.num_layers(),
        }
    }

    /// Forward pass. Returns logits [batch, seq_len, vocab_size].
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        tokens: &Array,
        cache: &mut [(Option<Array>, Option<Array>)],
        mask_mode: &str,
        mask: Option<&Array>,
    ) -> Result<Array> {
        match self {
            Model::Standard(m) => m.forward(tokens, cache, mask_mode, mask),
            Model::Qwen35(m) => m.forward(tokens, cache, mask_mode, mask),
        }
    }
}

/// Build a model from config, dispatching by `model_type`.
///
/// Supported model types: `llama`, `qwen3`, `qwen3_5`
pub fn build_model(config: &ModelConfig, weights: &HashMap<String, Array>) -> Result<Model> {
    match config.model_type.as_str() {
        "llama" => Ok(Model::Standard(LlamaModel::from_config(config, weights)?)),
        "qwen3" => Ok(Model::Standard(qwen3::from_config(config, weights)?)),
        other => Err(crate::error::Error::Mlx(format!(
            "unsupported model_type: '{}' (supported: llama, qwen3, qwen3_5)",
            other
        ))),
    }
}

/// Build a model from a config file path, auto-detecting model type.
///
/// For Qwen3.5 (nested config), uses `Qwen35Config`. For others, uses `ModelConfig`.
pub fn build_model_from_file(config_path: &str, weights: &HashMap<String, Array>) -> Result<Model> {
    // Try standard config first
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| crate::error::Error::Mlx(format!("failed to read config: {}", e)))?;

    // Quick check for model_type
    let raw: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| crate::error::Error::Mlx(format!("failed to parse config: {}", e)))?;

    let model_type = raw
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    match model_type {
        "qwen3_5" => {
            let model = qwen35::from_config_file(config_path, weights)?;
            Ok(Model::Qwen35(model))
        }
        _ => {
            let config: ModelConfig = serde_json::from_str(&content)
                .map_err(|e| crate::error::Error::Mlx(format!("failed to parse config: {}", e)))?;
            build_model(&config, weights)
        }
    }
}
