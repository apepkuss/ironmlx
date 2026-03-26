use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

use ironmlx_core::device::Device;
use ironmlx_core::generate::{ChatTemplate, Tokenizer};
use ironmlx_core::model::{Model, build_model_from_file, load_model_weights};
use ironmlx_core::stream::Stream;
use serde::Serialize;

use crate::config::ServerConfig;
use crate::engine_pool::EnginePool;

/// A single log entry stored in the in-memory buffer.
#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub timestamp: i64,
    pub level: String,
    pub message: String,
}

/// Fixed-capacity in-memory log buffer (FIFO).
pub struct LogBuffer {
    entries: std::sync::Mutex<VecDeque<LogEntry>>,
    capacity: std::sync::atomic::AtomicUsize,
}

impl LogBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: std::sync::Mutex::new(VecDeque::with_capacity(capacity)),
            capacity: std::sync::atomic::AtomicUsize::new(capacity),
        }
    }

    /// Push a log entry, evicting the oldest if at capacity.
    pub fn push(&self, level: &str, message: &str) {
        let entry = LogEntry {
            timestamp: chrono::Utc::now().timestamp(),
            level: level.to_string(),
            message: message.to_string(),
        };
        let cap = self.capacity.load(std::sync::atomic::Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap();
        while entries.len() >= cap {
            entries.pop_front();
        }
        entries.push_back(entry);
    }

    /// Return a snapshot of all entries.
    pub fn snapshot(&self) -> Vec<LogEntry> {
        self.entries.lock().unwrap().iter().cloned().collect()
    }

    /// Clear all entries.
    #[allow(dead_code)]
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }

    /// Update the buffer capacity. If shrinking, drop the oldest entries.
    pub fn set_capacity(&self, new_cap: usize) {
        self.capacity
            .store(new_cap, std::sync::atomic::Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap();
        while entries.len() > new_cap {
            entries.pop_front();
        }
    }

    /// Return the current capacity.
    #[allow(dead_code)]
    pub fn get_capacity(&self) -> usize {
        self.capacity.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// A stored benchmark result for historical tracking.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    pub timestamp: i64,
    pub model: String,
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub batch_size: usize,
    pub ttft_ms: f64,
    pub tpot_ms: f64,
    pub tg_tps: f64,
    pub pp_tps: f64,
    pub total_ms: f64,
    pub total_throughput: f64,
}

/// Status of a HuggingFace model download.
#[derive(Debug, Clone, Serialize)]
pub struct DownloadStatus {
    pub repo_id: String,
    pub status: String,
    pub progress_pct: f32,
    pub error: Option<String>,
}

pub struct AppState {
    pub pool: EnginePool,
    pub started_at: i64,
    pub config: Arc<std::sync::RwLock<ServerConfig>>,
    pub log_buffer: LogBuffer,
    pub downloads: Mutex<HashMap<String, DownloadStatus>>,
    pub benchmark_history: Mutex<Vec<BenchmarkResult>>,
    /// Total tokens processed (prompt + generated) across all requests.
    pub total_tokens: std::sync::atomic::AtomicU64,
}

impl AppState {
    /// Sync the default model to the ironmlx-app config file
    /// (~/.ironmlx/config/app_config.json)
    /// so the menubar app shows the correct model name.
    pub fn sync_default_model(&self, model_id: &str) {
        let config_path = crate::config::ironmlx_root()
            .join("config")
            .join("app_config.json");

        if let Ok(data) = std::fs::read_to_string(&config_path)
            && let Ok(mut json) = serde_json::from_str::<serde_json::Value>(&data)
        {
            json["last_model"] = serde_json::Value::String(model_id.to_string());
            let _ = std::fs::write(
                &config_path,
                serde_json::to_string_pretty(&json).unwrap_or_default(),
            );
        }
    }
}

/// Load model artifacts from a directory.
/// Returns (Model, Tokenizer, ChatTemplate, eos_token_id, model_id, patch_size, spatial_merge_size)
#[allow(clippy::type_complexity)]
/// Resolve a model path: if it looks like a HuggingFace repo ID (contains '/'
/// but doesn't exist as a local path), download it first using hf-hub.
fn resolve_model_path(model_dir: &str) -> Result<String, String> {
    let path = Path::new(model_dir);
    if path.exists() {
        return Ok(model_dir.to_string());
    }
    // Looks like a HF repo ID (e.g., "mlx-community/Qwen3-0.6B-4bit")
    if model_dir.contains('/') {
        println!("  Downloading model: {}", model_dir);
        let models_dir = crate::config::ironmlx_root().join("models");
        let _ = std::fs::create_dir_all(&models_dir);
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(models_dir)
            .build()
            .map_err(|e| format!("HF API init failed: {}", e))?;
        let repo = api.model(model_dir.to_string());
        // Download essential files
        for filename in &[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
        ] {
            if let Ok(p) = repo.get(filename) {
                println!("    {} -> {}", filename, p.display());
            }
        }
        // Return the snapshot directory
        let config = repo
            .get("config.json")
            .map_err(|e| format!("failed to download config.json: {}", e))?;
        let snapshot_dir = config
            .parent()
            .ok_or("cannot determine snapshot dir")?
            .to_string_lossy()
            .to_string();
        println!("  Model directory: {}", snapshot_dir);
        Ok(snapshot_dir)
    } else {
        Err(format!("model path not found: {}", model_dir))
    }
}

/// (Model, Tokenizer, ChatTemplate, eos_token_id, model_id, vocab_size, num_layers)
pub type LoadModelResult = (
    Model,
    Tokenizer,
    Option<ChatTemplate>,
    i32,
    String,
    usize,
    usize,
);

pub fn load_model(model_dir: &str) -> Result<LoadModelResult, String> {
    let resolved = resolve_model_path(model_dir)?;
    let dir = Path::new(&resolved);

    // Load config to extract metadata
    let config_path = dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("failed to read config: {}", e))?;
    let raw: serde_json::Value =
        serde_json::from_str(&config_str).map_err(|e| format!("failed to parse config: {}", e))?;

    let model_type = raw
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");
    println!("  Model type: {}", model_type);

    // Extract EOS token ID from config
    let eos_token_id = if model_type == "qwen3_5" {
        raw.get("text_config")
            .and_then(|tc| tc.get("eos_token_id"))
            .and_then(|v| v.as_i64())
            .unwrap_or(151645) as i32
    } else {
        raw.get("eos_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(2) as i32
    };
    println!("  EOS token ID: {}", eos_token_id);

    // Load weights
    let stream = Stream::new(&Device::gpu());
    let weights =
        load_model_weights(dir, &stream).map_err(|e| format!("weight loading error: {}", e))?;
    println!("  Loaded {} weight tensors", weights.len());

    // Build model (auto-dispatch by model_type, handles nested config)
    let model = build_model_from_file(config_path.to_str().unwrap(), &weights)
        .map_err(|e| format!("model build error: {}", e))?;
    println!("  Model layers: {}", model.num_layers());

    // Load tokenizer
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
        .map_err(|e| format!("tokenizer error: {}", e))?;

    // Load chat template (optional)
    let tc_path = dir.join("tokenizer_config.json");
    let mut chat_template = if tc_path.exists() {
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

    // Fallback: use built-in Qwen3 chat template for qwen3_5 models without one
    if chat_template.is_none() && (model_type == "qwen3_5" || model_type == "qwen3") {
        println!("  Chat template: using built-in Qwen3 template");
        chat_template = Some(ChatTemplate::new(
            include_str!("qwen3_chat_template.txt").to_string(),
            "<|im_end|>".to_string(),
            String::new(),
        ));
    }

    // Extract vision config for media processing
    let (patch_size, spatial_merge_size) = if let Some(vc) = raw.get("vision_config") {
        let ps = vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
        let sms = vc
            .get("spatial_merge_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        println!("  Vision: patch_size={}, spatial_merge_size={}", ps, sms);
        (ps, sms)
    } else {
        (16, 2) // defaults
    };

    // Derive model_id
    let model_id = extract_model_id(dir);

    Ok((
        model,
        tokenizer,
        chat_template,
        eos_token_id,
        model_id,
        patch_size,
        spatial_merge_size,
    ))
}

/// Extract a human-readable model ID from a path.
fn extract_model_id(dir: &Path) -> String {
    for ancestor in dir.ancestors() {
        if let Some(name) = ancestor.file_name().and_then(|n| n.to_str())
            && let Some(rest) = name.strip_prefix("models--")
        {
            return rest.replacen("--", "/", 1);
        }
    }
    dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
