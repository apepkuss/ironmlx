use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use tokio::sync::mpsc;

use ironmlx_core::generate::{ChatTemplate, Tokenizer};

use crate::engine::EngineCore;
use crate::engine_handle::EngineHandle;

/// Metadata for a loaded model engine
pub struct EngineEntry {
    pub engine: EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Option<ChatTemplate>,
    pub eos_token_id: i32,
    pub model_id: String,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
}

/// Pool of model engines — supports multiple models loaded simultaneously.
pub struct EnginePool {
    engines: RwLock<HashMap<String, Arc<EngineEntry>>>,
    default_model: RwLock<Option<String>>,
}

impl EnginePool {
    pub fn new() -> Self {
        Self {
            engines: RwLock::new(HashMap::new()),
            default_model: RwLock::new(None),
        }
    }

    /// Load a model and register its engine.
    /// Spawns a dedicated OS thread for the engine.
    pub fn load_model(
        &self,
        model_dir: &str,
        hot_cache_bytes: u64,
        cold_cache_bytes: u64,
        cache_dir_override: Option<&str>,
        max_num_seqs: usize,
    ) -> Result<String, String> {
        let (
            model,
            tokenizer,
            chat_template,
            eos_token_id,
            model_id,
            patch_size,
            spatial_merge_size,
        ) = crate::state::load_model(model_dir)?;

        // Check if already loaded
        {
            let engines = self.engines.read().unwrap();
            if engines.contains_key(&model_id) {
                return Ok(model_id); // already loaded
            }
        }

        // Create engine
        let (cmd_tx, cmd_rx) = mpsc::channel(256);
        let engine_handle = EngineHandle::new(cmd_tx);
        let engine_tokenizer = tokenizer.clone();

        // Create CacheManager with SSD-backed store
        // Note: store_block() now eval+copies arrays to CPU before sending to writer thread,
        // avoiding Metal CommandBuffer conflicts (SIGABRT) between inference and cache I/O.
        let num_layers = model.num_layers();
        let cache_dir = if let Some(dir) = cache_dir_override {
            // Expand ~ to home directory
            if let Some(stripped) = dir.strip_prefix("~/") {
                dirs::home_dir().unwrap_or_default().join(stripped)
            } else {
                std::path::PathBuf::from(dir)
            }
        } else {
            crate::config::ironmlx_root().join("cache").join("kv_cache")
        };
        let model_hash = model_id.replace('/', "_");
        let ssd_config = ironmlx_core::cache::SSDStoreConfig {
            cache_dir,
            max_size_bytes: cold_cache_bytes,
            model_hash,
        };
        let ssd_store = ironmlx_core::cache::SSDStore::new(ssd_config)
            .map_err(|e| format!("cache error: {}", e))?;
        let cache_manager =
            ironmlx_core::cache::CacheManager::new(ssd_store, num_layers, hot_cache_bytes);

        let mut engine =
            EngineCore::with_cache_manager(cmd_rx, model, engine_tokenizer, cache_manager, max_num_seqs);
        std::thread::spawn(move || {
            engine.run();
        });

        let entry = Arc::new(EngineEntry {
            engine: engine_handle,
            tokenizer: Arc::new(tokenizer),
            chat_template,
            eos_token_id,
            model_id: model_id.clone(),
            patch_size,
            spatial_merge_size,
        });

        // Register
        {
            let mut engines = self.engines.write().unwrap();
            engines.insert(model_id.clone(), entry);
        }

        // Set as default if first model
        {
            let mut default = self.default_model.write().unwrap();
            if default.is_none() {
                *default = Some(model_id.clone());
            }
        }

        Ok(model_id)
    }

    /// Unload a model and shutdown its engine.
    pub fn unload_model(&self, model_id: &str) -> Result<(), String> {
        let entry = {
            let mut engines = self.engines.write().unwrap();
            engines.remove(model_id)
        };

        if let Some(entry) = entry {
            // Send shutdown command (fire and forget)
            let engine = entry.engine.clone();
            tokio::spawn(async move {
                engine.shutdown().await;
            });

            // Clear default if this was the default
            let mut default = self.default_model.write().unwrap();
            if default.as_deref() == Some(model_id) {
                *default = self.engines.read().unwrap().keys().next().cloned();
            }

            Ok(())
        } else {
            Err(format!("model not loaded: {}", model_id))
        }
    }

    /// Get engine entry for a model. If model is None, use default.
    /// If `model_id` is not found directly, tries to resolve it via the provided alias map.
    pub fn get(&self, model_id: Option<&str>) -> Result<Arc<EngineEntry>, String> {
        let engines = self.engines.read().unwrap();

        if let Some(id) = model_id {
            engines
                .get(id)
                .cloned()
                .ok_or_else(|| format!("model not loaded: {}", id))
        } else {
            let default = self.default_model.read().unwrap();
            let default_id = default.as_deref().ok_or("no model loaded")?;
            engines
                .get(default_id)
                .cloned()
                .ok_or_else(|| "default model not found".to_string())
        }
    }

    /// Set default model.
    pub fn set_default(&self, model_id: &str) -> Result<(), String> {
        let engines = self.engines.read().unwrap();
        if !engines.contains_key(model_id) {
            return Err(format!("model not loaded: {}", model_id));
        }
        drop(engines);
        let mut default = self.default_model.write().unwrap();
        *default = Some(model_id.to_string());
        Ok(())
    }

    /// Get engine entry, resolving model aliases from the provided map.
    pub fn get_with_aliases(
        &self,
        model_id: Option<&str>,
        aliases: &HashMap<String, String>,
    ) -> Result<Arc<EngineEntry>, String> {
        if let Some(id) = model_id {
            // Try direct lookup first
            let engines = self.engines.read().unwrap();
            if let Some(entry) = engines.get(id) {
                return Ok(entry.clone());
            }
            // Try alias resolution
            if let Some(real_id) = aliases.get(id) {
                return engines
                    .get(real_id.as_str())
                    .cloned()
                    .ok_or_else(|| format!("model not loaded: {} (alias for {})", id, real_id));
            }
            Err(format!("model not loaded: {}", id))
        } else {
            self.get(None)
        }
    }

    /// List all loaded models.
    pub fn list_models(&self) -> Vec<String> {
        self.engines.read().unwrap().keys().cloned().collect()
    }

    /// Get the default model ID.
    #[allow(dead_code)]
    pub fn default_model_id(&self) -> Option<String> {
        self.default_model.read().unwrap().clone()
    }
}
