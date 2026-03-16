use std::path::Path;
use std::sync::Arc;

use ironmlx_core::device::Device;
use ironmlx_core::generate::Tokenizer;
use ironmlx_core::model::{LlamaModel, ModelConfig, load_model_weights};
use ironmlx_core::stream::Stream;

use crate::engine_handle::EngineHandle;

pub struct AppState {
    pub engine: EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    #[allow(dead_code)]
    pub config: ModelConfig,
    pub model_id: String,
}

/// Load model artifacts from a directory. Returns (model, tokenizer, config, model_id).
pub fn load_model(model_dir: &str) -> Result<(LlamaModel, Tokenizer, ModelConfig, String), String> {
    let dir = Path::new(model_dir);

    // Load config
    let config_path = dir.join("config.json");
    let config = ModelConfig::from_file(config_path.to_str().unwrap())
        .map_err(|e| format!("config error: {}", e))?;

    println!(
        "  Model type: {}, Layers: {}, Heads: {}, Hidden: {}",
        config.model_type, config.num_hidden_layers, config.num_attention_heads, config.hidden_size
    );

    // Load weights
    let stream = Stream::new(&Device::gpu());
    let weights =
        load_model_weights(dir, &stream).map_err(|e| format!("weight loading error: {}", e))?;
    println!("  Loaded {} weight tensors", weights.len());

    // Build model
    let llama = LlamaModel::from_config(&config, &weights)
        .map_err(|e| format!("model build error: {}", e))?;

    // Load tokenizer
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
        .map_err(|e| format!("tokenizer error: {}", e))?;

    // Derive model_id from directory name
    let model_id = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    Ok((llama, tokenizer, config, model_id))
}
