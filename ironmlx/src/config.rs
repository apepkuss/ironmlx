use std::path::PathBuf;

use serde::Deserialize;

/// Server configuration with layered loading: CLI args > env vars > config file > defaults.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model: Option<String>,
    /// Memory limit in GB (0 = auto-detect from hardware)
    pub memory_limit_gb: f64,
    /// Memory warn threshold (0.0-1.0)
    pub memory_warn_threshold: f64,
    /// KV cache directory
    pub cache_dir: Option<String>,
    /// KV cache max size in GB
    pub cache_max_size_gb: f64,
    /// Maximum concurrent sequences
    pub max_num_seqs: usize,
    /// Log level
    pub log_level: String,
    /// Sampling temperature
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// API key for authentication (None = no auth)
    pub api_key: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            model: None,
            memory_limit_gb: 0.0,
            memory_warn_threshold: 0.9,
            cache_dir: None,
            cache_max_size_gb: 10.0,
            max_num_seqs: 256,
            log_level: "info".to_string(),
            temperature: 1.0,
            top_p: 1.0,
            api_key: None,
        }
    }
}

impl ServerConfig {
    /// Load config: CLI args > env vars > config file > defaults.
    #[allow(dead_code)]
    pub fn load(cli_model: &str, cli_host: &str, cli_port: u16) -> Self {
        let mut config = Self::default();

        // Try config file
        if let Some(path) = Self::find_config_file()
            && let Ok(content) = std::fs::read_to_string(&path)
            && let Ok(file_config) = serde_json::from_str::<ServerConfig>(&content)
        {
            config = file_config;
            println!("  Config loaded from: {}", path.display());
        }

        // Environment variables override
        if let Ok(v) = std::env::var("IRONMLX_HOST") {
            config.host = v;
        }
        if let Ok(v) = std::env::var("IRONMLX_PORT")
            && let Ok(p) = v.parse()
        {
            config.port = p;
        }
        if let Ok(v) = std::env::var("IRONMLX_MODEL") {
            config.model = Some(v);
        }
        if let Ok(v) = std::env::var("IRONMLX_MEMORY_LIMIT_GB")
            && let Ok(f) = v.parse()
        {
            config.memory_limit_gb = f;
        }
        if let Ok(v) = std::env::var("IRONMLX_CACHE_DIR") {
            config.cache_dir = Some(v);
        }
        if let Ok(v) = std::env::var("IRONMLX_CACHE_MAX_SIZE_GB")
            && let Ok(f) = v.parse()
        {
            config.cache_max_size_gb = f;
        }
        if let Ok(v) = std::env::var("IRONMLX_LOG_LEVEL") {
            config.log_level = v;
        }
        if let Ok(v) = std::env::var("IRONMLX_TEMPERATURE")
            && let Ok(f) = v.parse()
        {
            config.temperature = f;
        }
        if let Ok(v) = std::env::var("IRONMLX_TOP_P")
            && let Ok(f) = v.parse()
        {
            config.top_p = f;
        }
        if let Ok(v) = std::env::var("IRONMLX_API_KEY") {
            config.api_key = if v.is_empty() { None } else { Some(v) };
        }

        // CLI args (highest priority)
        config.model = Some(cli_model.to_string());
        config.host = cli_host.to_string();
        config.port = cli_port;

        config
    }

    #[allow(dead_code)]
    fn find_config_file() -> Option<PathBuf> {
        let candidates = [
            PathBuf::from("ironmlx.json"),
            dirs::config_dir()
                .unwrap_or_default()
                .join("ironmlx")
                .join("config.json"),
        ];
        candidates.into_iter().find(|p| p.exists())
    }

    /// Get cache directory.
    #[allow(dead_code)]
    pub fn cache_directory(&self) -> PathBuf {
        if let Some(ref dir) = self.cache_dir {
            PathBuf::from(dir)
        } else {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("ironmlx")
                .join("kv_cache")
        }
    }

    /// Get cache max size in bytes.
    #[allow(dead_code)]
    pub fn cache_max_size_bytes(&self) -> u64 {
        (self.cache_max_size_gb * 1024.0 * 1024.0 * 1024.0) as u64
    }
}
