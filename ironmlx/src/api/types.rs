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

// -- Defaults ----------------------------------------------------------------

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
