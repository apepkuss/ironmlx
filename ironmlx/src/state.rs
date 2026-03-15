use std::path::Path;
use std::sync::Mutex;

use ironmlx_core::device::Device;
use ironmlx_core::generate::Tokenizer;
use ironmlx_core::model::{LlamaModel, ModelConfig, load_model_weights};
use ironmlx_core::stream::Stream;

pub struct AppState {
    pub model: Mutex<ModelState>,
    pub tokenizer: Tokenizer,
    #[allow(dead_code)]
    pub config: ModelConfig,
    pub model_id: String,
}

pub struct ModelState {
    pub llama: LlamaModel,
}

// SAFETY: MLX C handles are reference-counted and internally synchronized.
// All access to `LlamaModel` is guarded by `Mutex<ModelState>`.
unsafe impl Send for ModelState {}

// SAFETY: `AppState` is safe to share across threads because:
// - `ModelState` is behind a `Mutex` (exclusive access)
// - `Tokenizer` wraps HuggingFace `tokenizers::Tokenizer` (Send+Sync)
// - `ModelConfig` and `String` are Send+Sync
unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

impl AppState {
    pub fn load(model_dir: &str) -> Result<Self, String> {
        let dir = Path::new(model_dir);

        // Load config
        let config_path = dir.join("config.json");
        let config = ModelConfig::from_file(config_path.to_str().unwrap())
            .map_err(|e| format!("config error: {}", e))?;

        println!(
            "  Model type: {}, Layers: {}, Heads: {}, Hidden: {}",
            config.model_type,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size
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

        Ok(Self {
            model: Mutex::new(ModelState { llama }),
            tokenizer,
            config,
            model_id,
        })
    }
}
