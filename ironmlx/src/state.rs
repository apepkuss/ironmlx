use std::path::Path;
use std::sync::Arc;

use ironmlx_core::device::Device;
use ironmlx_core::generate::{ChatTemplate, Tokenizer};
use ironmlx_core::model::{LlamaModel, ModelConfig, build_model, load_model_weights};
use ironmlx_core::stream::Stream;

use crate::engine_handle::EngineHandle;

pub struct AppState {
    pub engine: EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Option<ChatTemplate>,
    pub config: ModelConfig,
    pub model_id: String,
}

/// Load model artifacts from a directory.
pub fn load_model(
    model_dir: &str,
) -> Result<
    (
        LlamaModel,
        Tokenizer,
        Option<ChatTemplate>,
        ModelConfig,
        String,
    ),
    String,
> {
    let dir = Path::new(model_dir);

    // Load config
    let config_path = dir.join("config.json");
    let config = ModelConfig::from_file(config_path.to_str().unwrap())
        .map_err(|e| format!("config error: {}", e))?;

    println!(
        "  Model type: {}, Layers: {}, Heads: {}, Hidden: {}",
        config.model_type, config.num_hidden_layers, config.num_attention_heads, config.hidden_size
    );
    println!("  EOS token ID: {}", config.eos_token_id);

    // Load weights
    let stream = Stream::new(&Device::gpu());
    let weights =
        load_model_weights(dir, &stream).map_err(|e| format!("weight loading error: {}", e))?;
    println!("  Loaded {} weight tensors", weights.len());

    // Build model (auto-dispatch by model_type)
    let model = build_model(&config, &weights).map_err(|e| format!("model build error: {}", e))?;

    // Load tokenizer
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
        .map_err(|e| format!("tokenizer error: {}", e))?;

    // Load chat template (optional)
    let tc_path = dir.join("tokenizer_config.json");
    let chat_template = if tc_path.exists() {
        match ChatTemplate::from_file(&tc_path.to_string_lossy()) {
            Ok(ct) => {
                println!("  Chat template: loaded");
                Some(ct)
            }
            Err(e) => {
                println!("  Chat template: not available ({})", e);
                None
            }
        }
    } else {
        println!("  Chat template: not found");
        None
    };

    // Derive model_id from directory name
    let model_id = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    Ok((model, tokenizer, chat_template, config, model_id))
}
