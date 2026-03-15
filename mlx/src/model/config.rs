use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_model_type() -> String {
    "llama".to_string()
}
fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_max_pos() -> usize {
    4096
}

impl ModelConfig {
    pub fn from_file(path: &str) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::error::Error::Mlx(format!("failed to read config: {}", e)))?;
        serde_json::from_str(&content)
            .map_err(|e| crate::error::Error::Mlx(format!("failed to parse config: {}", e)))
    }

    /// Number of KV heads (defaults to num_attention_heads if not specified -- MHA).
    pub fn n_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Head dimension (defaults to hidden_size / num_attention_heads).
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}
