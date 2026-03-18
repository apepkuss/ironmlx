pub mod bert;
mod config;
pub mod embedding_model;
mod llama;
mod loader;
mod qwen3;
mod qwen35;
mod qwen35_vl;

pub use bert::{BertConfig, BertModel};
pub use config::{ModelConfig, Qwen35Config, VisionConfig};
pub use embedding_model::{EmbeddingModel, PoolingStrategy};
pub use llama::{LlamaModel, TransformerBlock};
pub use loader::load_model_weights;
pub use qwen35::Qwen35Model;
pub use qwen35_vl::Qwen35VLModel;

use crate::array::Array;
use crate::error::Result;
use crate::media::ProcessedMedia;
use std::collections::HashMap;

/// Unified model type that can hold any supported architecture.
pub enum Model {
    /// Llama, Qwen3, and similar decoder-only models.
    Standard(LlamaModel),
    /// Qwen3.5 with mixed GatedDeltaNet + FullAttention layers.
    Qwen35(Qwen35Model),
    /// Qwen3.5 VLM with vision encoder.
    Qwen35VL(Box<Qwen35VLModel>),
    /// BERT encoder-only model for embeddings.
    Bert(BertModel),
}

impl Model {
    /// Number of layers in the model.
    pub fn num_layers(&self) -> usize {
        match self {
            Model::Standard(m) => m.layers.len(),
            Model::Qwen35(m) => m.num_layers(),
            Model::Qwen35VL(m) => m.num_layers(),
            Model::Bert(m) => m.num_layers(),
        }
    }

    /// Forward pass. Returns logits [batch, seq_len, vocab_size].
    /// For BERT models, returns hidden states [batch, seq_len, hidden_size] (no KV cache).
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
            Model::Qwen35VL(m) => m.forward(tokens, cache, mask_mode, mask),
            Model::Bert(m) => {
                let stream = crate::stream::Stream::new(&crate::device::Device::gpu());
                m.forward(tokens, &stream)
            }
        }
    }

    /// VLM forward pass with optional media. Returns logits.
    /// Falls back to text-only forward for non-VLM models.
    pub fn forward_vlm(
        &self,
        tokens: &Array,
        media: Option<&[ProcessedMedia]>,
        cache: &mut [(Option<Array>, Option<Array>)],
    ) -> Result<Array> {
        match self {
            Model::Qwen35VL(m) => m.forward_vlm(tokens, media, cache),
            _ => self.forward(tokens, cache, "causal", None),
        }
    }

    /// Returns true if this is an encoder-only model (e.g., BERT).
    pub fn is_encoder(&self) -> bool {
        matches!(self, Model::Bert(_))
    }
}

/// Build a model from config, dispatching by `model_type`.
///
/// Supported model types: `llama`, `qwen3`, `qwen3_5`, `bert`
pub fn build_model(config: &ModelConfig, weights: &HashMap<String, Array>) -> Result<Model> {
    match config.model_type.as_str() {
        "llama" => Ok(Model::Standard(LlamaModel::from_config(config, weights)?)),
        "qwen3" => Ok(Model::Standard(qwen3::from_config(config, weights)?)),
        other => Err(crate::error::Error::Mlx(format!(
            "unsupported model_type: '{}' (supported: llama, qwen3, qwen3_5, bert)",
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
        "bert" | "xlm-roberta" => {
            let model = bert::from_config_file(config_path, weights)?;
            Ok(Model::Bert(model))
        }
        "qwen3_5" => {
            // Check if vision_config is present to decide VLM vs text-only
            let has_vision = raw.get("vision_config").is_some();
            if has_vision {
                let model = qwen35_vl::from_config_file(config_path, weights)?;
                Ok(Model::Qwen35VL(Box::new(model)))
            } else {
                let model = qwen35::from_config_file(config_path, weights)?;
                Ok(Model::Qwen35(model))
            }
        }
        _ => {
            let config: ModelConfig = serde_json::from_str(&content)
                .map_err(|e| crate::error::Error::Mlx(format!("failed to parse config: {}", e)))?;
            build_model(&config, weights)
        }
    }
}
