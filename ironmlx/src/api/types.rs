use serde::{Deserialize, Serialize};

// -- Chat Completions --------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// When set to false, disables thinking/reasoning mode for models that support it (e.g. Qwen3).
    /// Default: None (use model's default behavior).
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

/// Multimodal content: either a plain text string or a list of content parts.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content (backward compatible)
    Text(String),
    /// Multimodal content parts (text + images/videos)
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract text content for backward compatibility.
    /// For plain text, returns the string directly.
    /// For parts, concatenates all text parts.
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// A single content part in a multimodal message.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },
    /// Image via URL or base64
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlContent },
    /// Video via URL or path
    #[serde(rename = "video_url")]
    VideoUrl { video_url: VideoUrlContent },
}

/// Image URL content (supports URL or base64 data URI).
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageUrlContent {
    pub url: String,
}

/// Video URL content.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VideoUrlContent {
    pub url: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// -- Completions -------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

// -- Models ------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

// -- Common ------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryInfo>,
    pub started_at: i64,
    pub total_tokens: u64,
    pub cached_tokens: u64,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Serialize)]
pub struct MemoryInfo {
    pub active_mb: f64,
    pub cache_mb: f64,
    pub peak_mb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_mb: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
}

// -- SSE Streaming -----------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

// -- Tool Calling ------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Serialize, Clone)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// -- Anthropic Messages API --------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

/// Anthropic content: either a plain string or an array of content blocks.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

impl AnthropicContent {
    pub fn as_text(&self) -> String {
        match self {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .map(|b| match b {
                    AnthropicContentBlock::Text { text } => text.as_str(),
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

// Anthropic streaming event types

#[derive(Debug, Serialize)]
pub struct AnthropicMessageStart {
    pub r#type: String,
    pub message: AnthropicMessageStartPayload,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessageStartPayload {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicContentBlockStart {
    pub r#type: String,
    pub index: usize,
    pub content_block: AnthropicContentBlock,
}

#[derive(Debug, Serialize)]
pub struct AnthropicContentBlockDelta {
    pub r#type: String,
    pub index: usize,
    pub delta: AnthropicTextDelta,
}

#[derive(Debug, Serialize)]
pub struct AnthropicTextDelta {
    pub r#type: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicContentBlockStop {
    pub r#type: String,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessageDelta {
    pub r#type: String,
    pub delta: AnthropicMessageDeltaPayload,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessageDeltaPayload {
    pub stop_reason: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessageStop {
    pub r#type: String,
}

// -- Engine Pool Management --------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model_dir: String,
}

#[derive(Debug, Deserialize)]
pub struct UnloadModelRequest {
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub struct SetDefaultRequest {
    pub model: String,
}

// -- Embeddings --------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

fn default_encoding_format() -> String {
    "float".to_string()
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// -- Rerank ------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<RerankDocument>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub top_n: Option<usize>,
    #[serde(default)]
    pub return_documents: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum RerankDocument {
    Text(String),
    Object { text: String },
}

impl RerankDocument {
    pub fn text(&self) -> &str {
        match self {
            RerankDocument::Text(s) => s,
            RerankDocument::Object { text } => text,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
    pub model: String,
    pub usage: RerankUsage,
}

#[derive(Debug, Serialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankResultDocument>,
}

#[derive(Debug, Serialize)]
pub struct RerankResultDocument {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct RerankUsage {
    pub total_tokens: usize,
}

// -- Defaults ----------------------------------------------------------------

pub fn default_max_tokens() -> usize {
    256
}
pub fn default_temperature() -> f32 {
    1.0
}
pub fn default_top_p() -> f32 {
    1.0
}
pub fn default_top_k() -> i32 {
    -1
}
pub fn default_repetition_penalty() -> f32 {
    1.0
}
