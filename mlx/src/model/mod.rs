mod config;
mod llama;
mod loader;
mod qwen3;

pub use config::ModelConfig;
pub use llama::{LlamaModel, TransformerBlock};
pub use loader::load_model_weights;

use crate::array::Array;
use crate::error::Result;
use std::collections::HashMap;

/// Build a model from config, dispatching by `model_type`.
///
/// Supported model types: `llama`, `qwen3`
pub fn build_model(config: &ModelConfig, weights: &HashMap<String, Array>) -> Result<LlamaModel> {
    match config.model_type.as_str() {
        "llama" => LlamaModel::from_config(config, weights),
        "qwen3" => qwen3::from_config(config, weights),
        other => Err(crate::error::Error::Mlx(format!(
            "unsupported model_type: '{}' (supported: llama, qwen3)",
            other
        ))),
    }
}
