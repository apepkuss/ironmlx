use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub model_dir: Option<String>,
    #[serde(default = "default_host")]
    pub host: String,
    pub port: u16,
    pub auto_start: bool,
    pub last_model: Option<String>,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default)]
    pub theme: Option<String>, // None=System, Some("light"), Some("dark")
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Total memory limit in GB (0.0 = auto-detect)
    #[serde(default)]
    pub memory_limit_total: f64,
    /// Model-only memory limit in GB (0.0 = no limit)
    #[serde(default)]
    pub memory_limit_model: f64,
    /// Hot cache (in-memory KV) limit in GB (0.0 = disabled)
    #[serde(default)]
    pub hot_cache_gb: f64,
    /// Cold cache (SSD KV) limit in GB (default 10.0, 0.0 = disabled)
    #[serde(default = "default_cold_cache_gb")]
    pub cold_cache_gb: f64,
    /// Max concurrent sequences
    #[serde(default = "default_max_sequences")]
    pub max_sequences: usize,
    /// Initial cache blocks (0 = auto-calculate from hot cache limit)
    #[serde(default)]
    pub init_cache_blocks: usize,
    /// Master cache toggle (false = disable all caching)
    #[serde(default = "default_true")]
    pub cache_enabled: bool,
    /// SSD cache directory
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_cold_cache_gb() -> f64 {
    10.0
}

fn default_max_sequences() -> usize {
    16
}

fn default_true() -> bool {
    true
}

fn default_cache_dir() -> String {
    "~/.ironmlx/cache/kv_cache".to_string()
}

fn default_log_level() -> String {
    "ALL".to_string()
}

fn default_language() -> String {
    "en".to_string()
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            host: "127.0.0.1".to_string(),
            port: 9068,
            auto_start: true,
            last_model: None,
            language: "en".to_string(),
            theme: None,
            log_level: "ALL".to_string(),
            memory_limit_total: 0.0,
            memory_limit_model: 0.0,
            hot_cache_gb: 0.0,
            cold_cache_gb: 10.0,
            max_sequences: 16,
            init_cache_blocks: 0,
            cache_enabled: true,
            cache_dir: "~/.ironmlx/cache/kv_cache".to_string(),
        }
    }
}

/// Root directory for all ironmlx data: ~/.ironmlx/
pub fn ironmlx_root() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ironmlx")
}

impl AppConfig {
    pub fn config_path() -> PathBuf {
        ironmlx_root().join("config").join("app_config.json")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        if path.exists() {
            let data = std::fs::read_to_string(&path).unwrap_or_default();
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    pub fn save(&self) {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(data) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(&path, data);
        }
    }
}
