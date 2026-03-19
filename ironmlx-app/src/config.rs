use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub model_dir: Option<String>,
    pub port: u16,
    pub auto_start: bool,
    pub last_model: Option<String>,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default)]
    pub theme: Option<String>, // None=System, Some("light"), Some("dark")
}

fn default_language() -> String {
    "en".to_string()
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            port: 8080,
            auto_start: true,
            last_model: None,
            language: "en".to_string(),
            theme: None,
        }
    }
}

impl AppConfig {
    pub fn config_path() -> PathBuf {
        let home = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
        home.join("ironmlx").join("app_config.json")
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
