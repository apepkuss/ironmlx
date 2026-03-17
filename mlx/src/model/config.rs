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
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default = "default_eos")]
    pub eos_token_id: i64,
    #[serde(default)]
    pub bos_token_id: Option<i64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
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
fn default_eos() -> i64 {
    2
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

// ---------------------------------------------------------------------------
// Qwen3.5 config — nested `text_config` structure
// ---------------------------------------------------------------------------

fn default_rms_norm_eps_6() -> f64 {
    1e-6
}
fn default_full_attn_interval() -> usize {
    4
}
fn default_linear_num_key_heads() -> usize {
    16
}
fn default_linear_num_value_heads() -> usize {
    32
}
fn default_linear_head_dim() -> usize {
    128
}
fn default_conv_kernel() -> usize {
    4
}
fn default_rope_theta_qwen35() -> f64 {
    10_000_000.0
}
fn default_partial_rotary() -> f64 {
    0.25
}

#[derive(Debug, Deserialize, Clone)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta_qwen35")]
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct Qwen35TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps_6")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_linear_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub eos_token_id: Option<i64>,
}

impl Qwen35TextConfig {
    /// Number of KV heads (defaults to num_attention_heads if not specified).
    pub fn n_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Head dimension (defaults to hidden_size / num_attention_heads).
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Rope theta from nested rope_parameters (defaults to 10_000_000).
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map_or(default_rope_theta_qwen35(), |rp| rp.rope_theta)
    }

    /// Partial rotary factor from nested rope_parameters (defaults to 0.25).
    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map_or(default_partial_rotary(), |rp| rp.partial_rotary_factor)
    }

    /// EOS token id (defaults to 151645).
    pub fn eos_token_id(&self) -> i64 {
        self.eos_token_id.unwrap_or(151645)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub patch_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    #[serde(default = "default_num_position_embeddings")]
    pub num_position_embeddings: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
}

fn default_temporal_patch_size() -> usize {
    2
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_in_channels() -> usize {
    3
}
fn default_num_position_embeddings() -> usize {
    2304
}
fn default_hidden_act() -> String {
    "gelu_pytorch_tanh".to_string()
}

#[derive(Debug, Deserialize)]
pub struct Qwen35Config {
    pub model_type: String,
    pub text_config: Qwen35TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    #[serde(default)]
    pub image_token_id: Option<i64>,
    #[serde(default)]
    pub video_token_id: Option<i64>,
    #[serde(default)]
    pub vision_start_token_id: Option<i64>,
    #[serde(default)]
    pub vision_end_token_id: Option<i64>,
}

impl Qwen35Config {
    pub fn from_file(path: &str) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::error::Error::Mlx(format!("failed to read config: {}", e)))?;
        serde_json::from_str(&content)
            .map_err(|e| crate::error::Error::Mlx(format!("failed to parse config: {}", e)))
    }
}
