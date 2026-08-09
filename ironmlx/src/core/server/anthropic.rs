//! Anthropic-compatible Messages API: /v1/messages.
//!
//! Streaming uses the native SSE lifecycle:
//!   message_start → one or more content_block_start/delta/stop groups
//!     → message_delta → message_stop
//!
//! Each event is framed as `event: <type>\ndata: <json>\n\n`.

use axum::{
    body::{Body, Bytes},
    extract::{rejection::JsonRejection, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::core::constrained::ToolConstraintOptions;
use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::generated_output::{
    GeneratedFinishReason, GeneratedOutputDecoder, GeneratedOutputEvent, ToolOutputDecoderConfig,
};
#[cfg(test)]
use crate::core::image_input::{ImageInputError, ImageRequestBudget};
use crate::core::model::Model;
use crate::core::native_output::NativeOutputDecoderConfig;
use crate::core::sampler::Sampler;
use crate::core::scheduler::DenseVlMethods;
use crate::core::server::chat_format::{
    render_and_encode, ChatFunctionCall, ChatMessage, ChatToolCall, Content, ContentPart, ImageUrl,
};
use crate::core::server::scheduler_actor::{AdmitReply, SchedulerCommand};
use crate::core::server::structured_output::StructuredOutputFormat;
#[cfg(test)]
use crate::core::server::vision::{DecodedMessage, DecodedPart};
use crate::core::speculative::MtpSpeculativeConfig;
use crate::core::tool_calling::{ToolCall, ToolDefinition, ToolDialect};

use super::SamplingDefaults;
use super::{request_token_capacity_error_response, AppState, Gemma4DrafterAppState};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a SchedulerActor admit Err into an HTTP response. Spec §4.7 + §2 G7:
/// - `SchedulerError::QueueFull` → 503 Service Unavailable + Retry-After: 5
/// - `SchedulerError::RequestTooLarge` → 413 Payload Too Large (no Retry-After)
/// - Other anyhow Errs (prompt parsing, OOM, etc.) → 400 Bad Request
///
/// Pre-3e.3 used `err.to_string().contains("admission queue full")` string
/// match (spec §9 R3 acknowledged-fragile). 3e.3 replaces with typed
/// `anyhow::Error::downcast_ref::<SchedulerError>()`. 3f adds RequestTooLarge arm.
fn admit_err_to_response(err: anyhow::Error) -> Response {
    use crate::core::SchedulerError;
    use axum::http::HeaderValue;
    let msg = format!("{err:#}");
    match err.downcast_ref::<SchedulerError>() {
        Some(SchedulerError::QueueFull { .. }) => {
            // 503 Service Unavailable + Retry-After
            let mut resp = (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
            resp
        }
        Some(error @ SchedulerError::RequestTooLarge { .. }) => {
            request_token_capacity_error_response(error)
        }
        Some(
            SchedulerError::MemoryBudgetExceeded { .. }
            | SchedulerError::MemoryPressure { .. }
            | SchedulerError::PrefillPeakUnsafe { .. }
            | SchedulerError::VisionPrefillPeakUnsafe { .. }
            | SchedulerError::ColdMaterializationUnsafe { .. }
            | SchedulerError::StoreBackpressure { .. },
        ) => {
            // 503 Service Unavailable — runtime KV budget soft-limit hit.
            // Retry-After: 5s (fixed conservative backoff). B1-p2.5 §4.1.4.
            let mut resp = (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
            resp
        }
        None => {
            // Other anyhow Errs (prompt parsing, OOM, etc.) → 400 Bad Request.
            (StatusCode::BAD_REQUEST, msg).into_response()
        }
    }
}

fn generation_err_to_response(err: anyhow::Error) -> Response {
    if err.downcast_ref::<crate::core::SchedulerError>().is_some() {
        admit_err_to_response(err)
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")).into_response()
    }
}

pub(crate) fn anthropic_error_response(status: StatusCode, message: String) -> Response {
    let error_type = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "api_error"
    };
    (
        status,
        Json(serde_json::json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        })),
    )
        .into_response()
}

fn format_stream_error(error: &anyhow::Error) -> Bytes {
    format_event(
        "error",
        &serde_json::json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": format!("{error:#}")
            }
        }),
    )
}

/// Anthropic native image source — base64 only (URL source is out of scope).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicImageSource {
    Base64 { media_type: String, data: String },
}

/// Anthropic native content block.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentPart {
    Text {
        text: String,
    },
    Image {
        source: AnthropicImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<AnthropicToolResultContent>,
        #[serde(default)]
        is_error: bool,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicToolResultContent {
    Text(String),
    Parts(Vec<AnthropicToolResultPart>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicToolResultPart {
    Text { text: String },
    Image { source: AnthropicImageSource },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicSystemPrompt {
    Text(String),
    Parts(Vec<AnthropicSystemPart>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicSystemPart {
    Text { text: String },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AnthropicToolDefinition {
    name: String,
    #[serde(default)]
    description: Option<String>,
    input_schema: serde_json::Value,
    #[serde(default)]
    strict: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum AnthropicToolChoice {
    Auto {
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    Any {
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    Tool {
        name: String,
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    None,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnthropicOutputConfig {
    #[serde(default)]
    format: Option<AnthropicOutputFormat>,
    #[serde(default)]
    effort: Option<AnthropicEffort>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum AnthropicOutputFormat {
    JsonSchema { schema: serde_json::Value },
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AnthropicEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AnthropicThinkingDisplay {
    Summarized,
    Omitted,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum AnthropicThinkingConfig {
    Disabled,
    Enabled {
        budget_tokens: usize,
        #[serde(default)]
        display: Option<AnthropicThinkingDisplay>,
    },
    Adaptive {
        #[serde(default)]
        display: Option<AnthropicThinkingDisplay>,
    },
}

impl AnthropicThinkingConfig {
    fn enabled(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    fn template_kwargs(&self) -> serde_json::Value {
        serde_json::json!({"enable_thinking": self.enabled()})
    }

    fn validate(&self, max_tokens: usize, effort: Option<AnthropicEffort>) -> anyhow::Result<()> {
        match self {
            Self::Disabled => anyhow::ensure!(
                effort.is_none(),
                "output_config.effort requires thinking.type=`enabled` or `adaptive`"
            ),
            Self::Enabled {
                budget_tokens,
                display,
            } => {
                anyhow::ensure!(
                    *budget_tokens >= 1024,
                    "thinking.budget_tokens must be at least 1024"
                );
                anyhow::ensure!(
                    *budget_tokens < max_tokens,
                    "thinking.budget_tokens must be less than max_tokens"
                );
                validate_thinking_display(*display)?;
            }
            Self::Adaptive { display } => validate_thinking_display(*display)?,
        }
        Ok(())
    }
}

fn validate_thinking_display(display: Option<AnthropicThinkingDisplay>) -> anyhow::Result<()> {
    anyhow::ensure!(
        !matches!(display, Some(AnthropicThinkingDisplay::Omitted)),
        "thinking.display=`omitted` is not supported because local models do not provide an encrypted hidden-thinking channel"
    );
    Ok(())
}

/// Anthropic message content: plain string or an array of content blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Parts(Vec<AnthropicContentPart>),
}

/// Anthropic message (private wire type; not shared with the OpenAI endpoint).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub(crate) messages: Vec<AnthropicMessage>,
    #[serde(default)]
    system: Option<AnthropicSystemPrompt>,
    #[serde(default)]
    tools: Option<Vec<AnthropicToolDefinition>>,
    #[serde(default)]
    tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    output_config: Option<AnthropicOutputConfig>,
    #[serde(default)]
    thinking: Option<AnthropicThinkingConfig>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
}

impl MessagesRequest {
    fn validate_sampling(&self) -> anyhow::Result<()> {
        if let Some(temperature) = self.temperature {
            anyhow::ensure!(
                temperature.is_finite() && (0.0..=1.0).contains(&temperature),
                "temperature must be finite and between 0 and 1"
            );
        }
        if let Some(top_p) = self.top_p {
            anyhow::ensure!(
                top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
                "top_p must be finite and in (0, 1]"
            );
        }
        if let Some(top_k) = self.top_k {
            anyhow::ensure!(top_k > 0, "top_k must be greater than zero");
        }
        Ok(())
    }
}

fn default_max_tokens() -> usize {
    256
}

#[derive(Debug, Serialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Serialize)]
struct OutputTokensDetails {
    thinking_tokens: u32,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MessageContentBlock {
    Thinking {
        thinking: String,
        signature: String,
    },
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Serialize)]
struct MessageEnvelope {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    content: Vec<MessageContentBlock>,
    model: String,
    stop_reason: Option<&'static str>,
    stop_sequence: Option<String>,
    usage: Usage,
}

fn gen_msg_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}

fn thinking_signature(thinking: &str) -> String {
    format!("ironmlx-v1:{:x}", Sha256::digest(thinking.as_bytes()))
}

fn validate_thinking_signature(thinking: &str, signature: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        signature == thinking_signature(thinking),
        "thinking block signature does not match its content"
    );
    Ok(())
}

fn build_sampler(req: &MessagesRequest, defaults: SamplingDefaults) -> Sampler {
    let mut s = Sampler::greedy();
    if let Some(t) = req.temperature.or(defaults.temperature) {
        if t > 0.0 {
            s = s.with_temperature(t);
        }
    }
    if let Some(p) = req.top_p.or(defaults.top_p) {
        if p > 0.0 && p <= 1.0 {
            s = s.with_top_p(p);
        }
    }
    if let Some(k) = req.top_k.or(defaults.top_k) {
        if k > 0 {
            s = s.with_top_k(k);
        }
    }
    if let Some(penalty) = defaults.repetition_penalty {
        if penalty > 0.0 && penalty != 1.0 {
            s = s.with_repetition_penalty(penalty);
        }
    }
    s
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessagesRoute {
    SchedulerStream,
    GenerationStreamStream,
    SchedulerUnary,
    GenerationStreamUnary,
}

struct ToolResponseContext {
    dialect: ToolDialect,
    definitions: Vec<ToolDefinition>,
    constraint_options: ToolConstraintOptions,
    output_schema: Option<serde_json::Value>,
    output_format: StructuredOutputFormat,
    native_output: Option<NativeOutputDecoderConfig>,
}

impl ToolResponseContext {
    fn decoder_config(&self) -> ToolOutputDecoderConfig {
        ToolOutputDecoderConfig {
            dialect: self.dialect,
            response_id: uuid::Uuid::new_v4().simple().to_string(),
            definitions: self.definitions.clone(),
            output_schema: self.output_schema.clone(),
        }
    }
}

fn messages_route(stream: bool, use_scheduler: bool) -> MessagesRoute {
    match (stream, use_scheduler) {
        (true, true) => MessagesRoute::SchedulerStream,
        (true, false) => MessagesRoute::GenerationStreamStream,
        (false, true) => MessagesRoute::SchedulerUnary,
        (false, false) => MessagesRoute::GenerationStreamUnary,
    }
}

fn format_event(event_type: &str, payload: &serde_json::Value) -> Bytes {
    let mut buf = String::new();
    buf.push_str("event: ");
    buf.push_str(event_type);
    buf.push('\n');
    buf.push_str("data: ");
    buf.push_str(&serde_json::to_string(payload).unwrap_or_else(|_| "{}".into()));
    buf.push_str("\n\n");
    Bytes::from(buf)
}

fn image_source_to_openai(source: AnthropicImageSource) -> ImageUrl {
    let AnthropicImageSource::Base64 { media_type, data } = source;
    ImageUrl {
        url: format!("data:{media_type};base64,{data}"),
    }
}

fn content_from_parts(parts: Vec<ContentPart>) -> Content {
    if parts
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }))
    {
        return Content::Text(
            parts
                .into_iter()
                .map(|part| match part {
                    ContentPart::Text { text } => text,
                    ContentPart::ImageUrl { .. } => unreachable!("checked text-only parts"),
                })
                .collect(),
        );
    }
    Content::Parts(parts)
}

fn tool_result_content(content: Option<AnthropicToolResultContent>, is_error: bool) -> Content {
    let mut parts = match content {
        None => Vec::new(),
        Some(AnthropicToolResultContent::Text(text)) => vec![ContentPart::Text { text }],
        Some(AnthropicToolResultContent::Parts(parts)) => parts
            .into_iter()
            .map(|part| match part {
                AnthropicToolResultPart::Text { text } => ContentPart::Text { text },
                AnthropicToolResultPart::Image { source } => ContentPart::ImageUrl {
                    image_url: image_source_to_openai(source),
                },
            })
            .collect(),
    };
    if is_error {
        parts.insert(
            0,
            ContentPart::Text {
                text: "{\"is_error\":true}\n".to_owned(),
            },
        );
    }
    content_from_parts(parts)
}

fn system_message(system: AnthropicSystemPrompt) -> ChatMessage {
    let text = match system {
        AnthropicSystemPrompt::Text(text) => text,
        AnthropicSystemPrompt::Parts(parts) => parts
            .into_iter()
            .map(|part| match part {
                AnthropicSystemPart::Text { text } => text,
            })
            .collect(),
    };
    ChatMessage::text("system", text)
}

fn normalize_user_message(content: AnthropicContent) -> anyhow::Result<Vec<ChatMessage>> {
    let AnthropicContent::Parts(parts) = content else {
        let AnthropicContent::Text(text) = content else {
            unreachable!()
        };
        return Ok(vec![ChatMessage::text("user", text)]);
    };

    let mut output = Vec::new();
    let mut ordinary = Vec::new();
    for part in parts {
        match part {
            AnthropicContentPart::Text { text } => ordinary.push(ContentPart::Text { text }),
            AnthropicContentPart::Image { source } => ordinary.push(ContentPart::ImageUrl {
                image_url: image_source_to_openai(source),
            }),
            AnthropicContentPart::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                anyhow::ensure!(
                    ordinary.is_empty(),
                    "tool_result blocks must precede text or image blocks in a user message"
                );
                output.push(ChatMessage {
                    role: "tool".to_owned(),
                    content: tool_result_content(content, is_error),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    tool_call_id: Some(tool_use_id),
                });
            }
            AnthropicContentPart::ToolUse { .. } => {
                anyhow::bail!("tool_use blocks are only valid in assistant messages")
            }
            AnthropicContentPart::Thinking { .. }
            | AnthropicContentPart::RedactedThinking { .. } => {
                anyhow::bail!("thinking blocks are only valid in assistant messages")
            }
        }
    }
    if !ordinary.is_empty() || output.is_empty() {
        output.push(ChatMessage {
            role: "user".to_owned(),
            content: content_from_parts(ordinary),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
    }
    Ok(output)
}

fn normalize_assistant_message(content: AnthropicContent) -> anyhow::Result<ChatMessage> {
    let AnthropicContent::Parts(parts) = content else {
        let AnthropicContent::Text(text) = content else {
            unreachable!()
        };
        return Ok(ChatMessage::text("assistant", text));
    };

    let mut text = String::new();
    let mut reasoning_content = None;
    let mut tool_calls = Vec::new();
    let mut saw_tool_use = false;
    let mut saw_visible_content = false;
    for part in parts {
        match part {
            AnthropicContentPart::Text { text: delta } => {
                anyhow::ensure!(
                    !saw_tool_use,
                    "assistant text blocks cannot follow a tool_use block"
                );
                saw_visible_content = true;
                text.push_str(&delta);
            }
            AnthropicContentPart::ToolUse { id, name, input } => {
                anyhow::ensure!(input.is_object(), "tool_use.input must be a JSON object");
                saw_visible_content = true;
                saw_tool_use = true;
                tool_calls.push(ChatToolCall {
                    id,
                    kind: "function".to_owned(),
                    function: ChatFunctionCall {
                        name,
                        arguments: serde_json::to_string(&input)?,
                    },
                });
            }
            AnthropicContentPart::Image { .. } => {
                anyhow::bail!("image blocks are only valid in user messages")
            }
            AnthropicContentPart::ToolResult { .. } => {
                anyhow::bail!("tool_result blocks are only valid in user messages")
            }
            AnthropicContentPart::Thinking {
                thinking,
                signature,
            } => {
                anyhow::ensure!(
                    !saw_visible_content,
                    "thinking blocks must precede assistant text and tool_use blocks"
                );
                anyhow::ensure!(
                    reasoning_content.is_none(),
                    "multiple thinking blocks in one assistant message are not supported"
                );
                validate_thinking_signature(&thinking, &signature)?;
                reasoning_content = Some(thinking);
            }
            AnthropicContentPart::RedactedThinking { data } => {
                anyhow::bail!(
                    "redacted_thinking is not supported because local models cannot decrypt opaque thinking data ({} bytes)",
                    data.len()
                )
            }
        }
    }
    Ok(ChatMessage {
        role: "assistant".to_owned(),
        content: Content::Text(text),
        reasoning_content,
        tool_calls,
        tool_call_id: None,
    })
}

fn normalize_messages(
    system: Option<AnthropicSystemPrompt>,
    messages: Vec<AnthropicMessage>,
) -> anyhow::Result<Vec<ChatMessage>> {
    let mut output = Vec::new();
    if let Some(system) = system {
        output.push(system_message(system));
    }
    for message in messages {
        match message.role.as_str() {
            "user" => output.extend(normalize_user_message(message.content)?),
            "assistant" => output.push(normalize_assistant_message(message.content)?),
            role => anyhow::bail!("unsupported Anthropic message role `{role}`"),
        }
    }
    Ok(output)
}

fn normalize_tool_choice(
    choice: Option<AnthropicToolChoice>,
) -> (Option<serde_json::Value>, Option<bool>) {
    match choice {
        None => (None, None),
        Some(AnthropicToolChoice::Auto {
            disable_parallel_tool_use,
        }) => (
            Some(serde_json::json!("auto")),
            Some(!disable_parallel_tool_use),
        ),
        Some(AnthropicToolChoice::Any {
            disable_parallel_tool_use,
        }) => (
            Some(serde_json::json!("required")),
            Some(!disable_parallel_tool_use),
        ),
        Some(AnthropicToolChoice::Tool {
            name,
            disable_parallel_tool_use,
        }) => (
            Some(serde_json::json!({
                "type": "function",
                "function": {"name": name}
            })),
            Some(!disable_parallel_tool_use),
        ),
        Some(AnthropicToolChoice::None) => (Some(serde_json::json!("none")), None),
    }
}

impl MessagesRequest {
    pub(crate) fn into_chat_request(self) -> anyhow::Result<super::openai::ChatRequest> {
        let has_output_format = self
            .output_config
            .as_ref()
            .and_then(|config| config.format.as_ref())
            .is_some();
        let effort = self.output_config.as_ref().and_then(|config| config.effort);
        if let Some(config) = self.output_config.as_ref() {
            anyhow::ensure!(
                config.format.is_some() || config.effort.is_some(),
                "output_config must include `format` or `effort`"
            );
        }
        match self.thinking.as_ref() {
            Some(thinking) => thinking.validate(self.max_tokens, effort)?,
            None => anyhow::ensure!(
                effort.is_none(),
                "output_config.effort requires thinking.type=`enabled` or `adaptive`"
            ),
        }
        if has_output_format
            && self
                .messages
                .last()
                .is_some_and(|message| message.role == "assistant")
        {
            anyhow::bail!("output_config.format is incompatible with assistant message prefilling");
        }
        let allows_final_output = self.tool_choice.as_ref().is_none_or(|choice| {
            matches!(
                choice,
                AnthropicToolChoice::Auto { .. } | AnthropicToolChoice::None
            )
        });
        let mut messages = normalize_messages(self.system, self.messages)?;
        let response_format = self
            .output_config
            .and_then(|config| config.format)
            .map(|format| match format {
                AnthropicOutputFormat::JsonSchema { schema } => {
                    super::openai::ChatResponseFormat::JsonSchema {
                        json_schema: super::openai::ChatJsonSchema {
                            name: "anthropic_output".to_owned(),
                            description: None,
                            schema,
                            strict: Some(false),
                        },
                    }
                }
            });
        if allows_final_output {
            let output_format = match response_format.as_ref() {
                Some(super::openai::ChatResponseFormat::JsonSchema { json_schema }) => {
                    StructuredOutputFormat::JsonSchema {
                        name: json_schema.name.clone(),
                        description: json_schema.description.clone(),
                        schema: json_schema.schema.clone(),
                        strict: json_schema.strict,
                    }
                }
                _ => StructuredOutputFormat::Text,
            };
            output_format.apply_prompt_instruction(&mut messages);
        }
        let tools = self.tools.map(|tools| {
            tools
                .into_iter()
                .map(|tool| super::openai::OpenAiTool {
                    kind: "function".to_owned(),
                    function: ToolDefinition {
                        name: tool.name,
                        description: tool.description,
                        parameters: tool.input_schema,
                        strict: tool.strict,
                    },
                })
                .collect()
        });
        let (tool_choice, parallel_tool_calls) = normalize_tool_choice(self.tool_choice);
        Ok(super::openai::ChatRequest {
            model: self.model,
            messages,
            tools,
            tool_choice,
            parallel_tool_calls,
            function_call: None,
            functions: None,
            response_format,
            stream: self.stream,
            stream_options: None,
            ignore_eos: false,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            seed: None,
            chat_template_kwargs: self
                .thinking
                .as_ref()
                .map(AnthropicThinkingConfig::template_kwargs),
        })
    }
}

pub(crate) fn messages_native_output_config(
    tokenizer: &crate::core::Tokenizer,
    req: &super::openai::ChatRequest,
) -> anyhow::Result<Option<NativeOutputDecoderConfig>> {
    let config = tokenizer.native_output_decoder_config(req.chat_template_kwargs.as_ref())?;
    let explicitly_enabled = req
        .chat_template_kwargs
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .and_then(|kwargs| kwargs.get("enable_thinking"))
        .and_then(serde_json::Value::as_bool)
        == Some(true);
    anyhow::ensure!(
        !explicitly_enabled || config.is_some(),
        "the loaded model does not expose a supported native thinking channel"
    );
    Ok(config)
}

/// Decode Anthropic native content blocks into the wire-agnostic
/// `DecodedMessage` list. base64 source is decoded in-process (no network);
/// `media_type` is informational and not validated.
#[cfg(test)]
pub(crate) fn decode_anthropic_messages(
    messages: Vec<AnthropicMessage>,
) -> Result<Vec<DecodedMessage>, ImageInputError> {
    let mut budget = ImageRequestBudget::default();
    let mut out: Vec<DecodedMessage> = Vec::with_capacity(messages.len());
    for m in messages {
        let mut parts: Vec<DecodedPart> = Vec::new();
        match m.content {
            AnthropicContent::Text(t) => {
                budget.add_text(&t)?;
                parts.push(DecodedPart::Text(t));
            }
            AnthropicContent::Parts(ps) => {
                for p in ps {
                    match p {
                        AnthropicContentPart::Text { text } => {
                            budget.add_text(&text)?;
                            parts.push(DecodedPart::Text(text));
                        }
                        AnthropicContentPart::Image { source } => {
                            let AnthropicImageSource::Base64 { media_type, data } = source;
                            let bytes = budget.decode_base64(&media_type, &data)?;
                            parts.push(DecodedPart::Image(bytes));
                        }
                        AnthropicContentPart::ToolUse { .. }
                        | AnthropicContentPart::ToolResult { .. }
                        | AnthropicContentPart::Thinking { .. }
                        | AnthropicContentPart::RedactedThinking { .. } => {
                            return Err(ImageInputError::DecodeFailed);
                        }
                    }
                }
            }
        }
        out.push(DecodedMessage {
            role: m.role,
            parts,
            reasoning_content: None,
        });
    }
    Ok(out)
}

pub async fn messages<M>(
    State(state): State<AppState<M>>,
    payload: std::result::Result<Json<MessagesRequest>, JsonRejection>,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let req = match payload {
        Ok(Json(req)) => req,
        Err(error) => {
            return anthropic_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid Messages request: {}", error.body_text()),
            );
        }
    };
    messages_with_state(state, req).await
}

pub(crate) async fn messages_with_state<M>(state: AppState<M>, req: MessagesRequest) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if let Err(error) = req.validate_sampling() {
        return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
    }
    let sampler = build_sampler(&req, state.sampling_defaults);
    let req = match req.into_chat_request() {
        Ok(req) => req,
        Err(error) => {
            return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
        }
    };
    let native_output = match messages_native_output_config(&state.tokenizer, &req) {
        Ok(config) => config,
        Err(error) => {
            return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
        }
    };
    let output_format = match req.structured_output_format() {
        Ok(format) => format,
        Err(error) => {
            return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
        }
    };
    let output_schema = output_format.constraint_schema();
    let max_tokens = req.max_tokens;
    let stream = req.stream;
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    if let Err(error) = super::validate_prompt_lookup_sampler(state.prompt_lookup_enabled, sampler)
    {
        return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
    }
    let prepared_tools =
        match super::openai::prepare_tool_request(&req, state.tokenizer.tool_dialect()) {
            Ok(prepared) => prepared,
            Err(error) => {
                return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
            }
        };
    let chat_template_kwargs = req.chat_template_kwargs.clone();
    let original_messages = prepared_tools.as_ref().map(|_| req.messages.clone());

    let (flat_messages, pixel_values, image_grid_thw) =
        match super::openai::expand_image_parts_in_messages(req.messages, &state.vision_input).await
        {
            Ok(t) => t,
            Err(e) => return super::security::image_error_response(e),
        };
    let image_grid_thw_opt = if image_grid_thw.is_empty() {
        None
    } else {
        Some(image_grid_thw)
    };
    let (image_token_id, image_spatial_merge_size) =
        crate::core::server::vision::derive_image_token_and_merge(
            &state.vision_input,
            &state.tokenizer,
        );

    let prompt_ids_result = if let Some(prepared) = &prepared_tools {
        super::openai::build_agent_messages(
            original_messages
                .as_deref()
                .expect("captured for tool request"),
            &flat_messages,
        )
        .and_then(|messages| {
            let kwargs =
                super::openai::tool_template_kwargs(chat_template_kwargs.clone(), prepared)?;
            super::openai::render_tool_prompt(&state.tokenizer, &messages, &kwargs)
        })
    } else {
        render_and_encode(
            &state.tokenizer,
            &flat_messages,
            chat_template_kwargs.as_ref(),
        )
    };
    let prompt_ids = match prompt_ids_result {
        Ok(ids) => ids,
        Err(e) => {
            return anthropic_error_response(
                StatusCode::BAD_REQUEST,
                format!("chat template / tokenize: {e}"),
            );
        }
    };
    let input_tokens = prompt_ids.len() as u32;
    let prompt_len = prompt_ids.len();
    let scheduler_config = state.scheduler_request_config(prompt_len, max_tokens);
    let stop_token_ids = state.tokenizer.eos_token_ids().to_vec();
    let constraint = match super::openai::compile_output_constraint_with_native(
        &state.tokenizer,
        prepared_tools.as_ref(),
        output_schema.as_ref(),
        native_output,
    ) {
        Ok(constraint) => constraint,
        Err(error) => {
            return anthropic_error_response(
                StatusCode::BAD_REQUEST,
                format!("compile output decoding constraint: {error:#}"),
            );
        }
    };
    let request = GenerateRequest {
        prompt_ids,
        max_new_tokens: max_tokens,
        sampler,
        stop_token_ids,
        prefill_chunk_size: scheduler_config.prefill_chunk_size,
        decode_cadence_mid_chunk_cap: scheduler_config.decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits: state.kv_cache_turboquant_bits,
        pixel_values,
        image_grid_thw: image_grid_thw_opt,
        image_spatial_merge_size,
        image_token_id,
        constraint,
    };

    let use_scheduler = super::should_route_to_scheduler::<M>(
        prompt_len,
        scheduler_config.prefill_chunk_size,
        state.b_max,
        state.paged_prefix_cache_enabled,
        state.force_scheduler_for_greedy && sampler.is_pipelinable(),
    );

    if let Some(prepared) = prepared_tools.filter(|prepared| prepared.constraint_options.is_some())
    {
        let constraint_options = prepared
            .constraint_options
            .expect("filtered constrained tool request");
        let tool_context = ToolResponseContext {
            dialect: prepared.dialect,
            definitions: prepared.definitions,
            output_schema: matches!(
                &constraint_options.choice,
                crate::core::constrained::ToolChoiceConstraint::Auto
            )
            .then(|| output_schema.clone())
            .flatten(),
            output_format: output_format.clone(),
            constraint_options,
            native_output,
        };
        return match messages_route(stream, use_scheduler) {
            MessagesRoute::SchedulerStream | MessagesRoute::SchedulerUnary => {
                serve_via_scheduler_tools(
                    state,
                    request,
                    model_label,
                    input_tokens,
                    stream,
                    tool_context,
                )
                .await
            }
            MessagesRoute::GenerationStreamStream | MessagesRoute::GenerationStreamUnary => {
                serve_via_gs_tools(
                    state,
                    request,
                    model_label,
                    input_tokens,
                    stream,
                    tool_context,
                )
                .await
            }
        };
    }

    match messages_route(stream, use_scheduler) {
        MessagesRoute::SchedulerStream => {
            serve_via_scheduler_stream_with_output_format(
                state,
                request,
                model_label,
                input_tokens,
                output_format,
                native_output,
            )
            .await
        }
        MessagesRoute::GenerationStreamStream => {
            serve_via_gs_stream(
                state,
                request,
                model_label,
                input_tokens,
                output_format,
                native_output,
            )
            .await
        }
        MessagesRoute::SchedulerUnary => {
            serve_via_scheduler_unary_with_output_format(
                state,
                request,
                model_label,
                input_tokens,
                output_format,
                native_output,
            )
            .await
        }
        MessagesRoute::GenerationStreamUnary => {
            serve_via_gs_unary(
                state,
                request,
                model_label,
                input_tokens,
                output_format,
                native_output,
            )
            .await
        }
    }
}

pub(crate) async fn gemma4_drafter_messages(
    State(state): State<Gemma4DrafterAppState>,
    payload: std::result::Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let req = match payload {
        Ok(Json(req)) => req,
        Err(error) => {
            return anthropic_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid Messages request: {}", error.body_text()),
            );
        }
    };
    messages_with_gemma4_drafter_state(state, req).await
}

pub(crate) async fn messages_with_gemma4_drafter_state(
    state: Gemma4DrafterAppState,
    req: MessagesRequest,
) -> Response {
    if let Err(error) = req.validate_sampling() {
        return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
    }
    let sampler = build_sampler(&req, state.base.sampling_defaults);
    let req = match req.into_chat_request() {
        Ok(req) => req,
        Err(error) => {
            return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
        }
    };
    let native_output = match messages_native_output_config(&state.base.tokenizer, &req) {
        Ok(config) => config,
        Err(error) => {
            return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
        }
    };
    let output_format = match req.structured_output_format() {
        Ok(format) => format,
        Err(error) => {
            return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
        }
    };
    let output_schema = output_format.constraint_schema();
    let max_tokens = req.max_tokens;
    let stream = req.stream;
    let model_label = req
        .model
        .clone()
        .unwrap_or_else(|| state.base.model_id.clone());
    let _cfg = match MtpSpeculativeConfig::new(state.mtp_draft_tokens, sampler) {
        Ok(cfg) => cfg,
        Err(e) => return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let prepared_tools =
        match super::openai::prepare_tool_request(&req, state.base.tokenizer.tool_dialect()) {
            Ok(prepared) => prepared,
            Err(error) => {
                return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
            }
        };
    let chat_template_kwargs = req.chat_template_kwargs.clone();
    let original_messages = prepared_tools.as_ref().map(|_| req.messages.clone());
    let (flat_messages, pixel_values, image_grid_thw) =
        match super::openai::expand_image_parts_in_messages(req.messages, &state.base.vision_input)
            .await
        {
            Ok(t) => t,
            Err(e) => return super::security::image_error_response(e),
        };
    let image_grid_thw_opt = if image_grid_thw.is_empty() {
        None
    } else {
        Some(image_grid_thw)
    };
    let (image_token_id, image_spatial_merge_size) =
        crate::core::server::vision::derive_image_token_and_merge(
            &state.base.vision_input,
            &state.base.tokenizer,
        );
    let prompt_ids_result = if let Some(prepared) = &prepared_tools {
        super::openai::build_agent_messages(
            original_messages
                .as_deref()
                .expect("captured for tool request"),
            &flat_messages,
        )
        .and_then(|messages| {
            let kwargs =
                super::openai::tool_template_kwargs(chat_template_kwargs.clone(), prepared)?;
            super::openai::render_tool_prompt(&state.base.tokenizer, &messages, &kwargs)
        })
    } else {
        render_and_encode(
            &state.base.tokenizer,
            &flat_messages,
            chat_template_kwargs.as_ref(),
        )
    };
    let prompt_ids = match prompt_ids_result {
        Ok(ids) => ids,
        Err(e) => {
            return anthropic_error_response(
                StatusCode::BAD_REQUEST,
                format!("chat template / tokenize: {e}"),
            );
        }
    };
    let input_tokens = prompt_ids.len() as u32;
    let prompt_len = prompt_ids.len();
    let total_tokens = prompt_len.saturating_add(max_tokens);
    if total_tokens > state.base.effective_cap_max {
        return anthropic_error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "request too large: prompt_len + max_tokens = {total_tokens}, max = {}",
                state.base.effective_cap_max
            ),
        );
    }
    let scheduler_config = state.base.scheduler_request_config(prompt_len, max_tokens);
    let stop_token_ids = state.base.tokenizer.eos_token_ids().to_vec();
    let constraint = match super::openai::compile_output_constraint_with_native(
        &state.base.tokenizer,
        prepared_tools.as_ref(),
        output_schema.as_ref(),
        native_output,
    ) {
        Ok(constraint) => constraint,
        Err(error) => {
            return anthropic_error_response(
                StatusCode::BAD_REQUEST,
                format!("compile output decoding constraint: {error:#}"),
            );
        }
    };
    let request = GenerateRequest {
        prompt_ids,
        max_new_tokens: max_tokens,
        sampler,
        stop_token_ids,
        prefill_chunk_size: scheduler_config.prefill_chunk_size,
        decode_cadence_mid_chunk_cap: scheduler_config.decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits: state.base.kv_cache_turboquant_bits,
        pixel_values,
        image_grid_thw: image_grid_thw_opt,
        image_spatial_merge_size,
        image_token_id,
        constraint,
    };

    if let Some(prepared) = prepared_tools.filter(|prepared| prepared.constraint_options.is_some())
    {
        let constraint_options = prepared
            .constraint_options
            .expect("filtered constrained tool request");
        let tool_context = ToolResponseContext {
            dialect: prepared.dialect,
            definitions: prepared.definitions,
            output_schema: matches!(
                &constraint_options.choice,
                crate::core::constrained::ToolChoiceConstraint::Auto
            )
            .then(|| output_schema.clone())
            .flatten(),
            output_format: output_format.clone(),
            constraint_options,
            native_output,
        };
        return serve_via_scheduler_tools(
            state.base,
            request,
            model_label,
            input_tokens,
            stream,
            tool_context,
        )
        .await;
    }

    if stream {
        serve_via_scheduler_stream_with_output_format(
            state.base,
            request,
            model_label,
            input_tokens,
            output_format,
            native_output,
        )
        .await
    } else {
        serve_via_scheduler_unary_with_output_format(
            state.base,
            request,
            model_label,
            input_tokens,
            output_format,
            native_output,
        )
        .await
    }
}

#[derive(Debug)]
struct ParsedToolOutput {
    content: String,
    reasoning: String,
    tool_calls: Vec<ToolCall>,
    finish_reason: &'static str,
    completion_tokens: u32,
    thinking_tokens: u32,
}

fn anthropic_finish_reason(reason: GeneratedFinishReason) -> &'static str {
    match reason {
        GeneratedFinishReason::Stop => "end_turn",
        GeneratedFinishReason::Length => "max_tokens",
        GeneratedFinishReason::ToolCalls => "tool_use",
    }
}

fn collect_tool_events(
    output: &mut ParsedToolOutput,
    events: Vec<GeneratedOutputEvent>,
) -> anyhow::Result<()> {
    for event in events {
        match event {
            GeneratedOutputEvent::TextDelta(text) => output.content.push_str(&text),
            GeneratedOutputEvent::ReasoningDelta(text) => output.reasoning.push_str(&text),
            GeneratedOutputEvent::ToolCall(call) => output.tool_calls.push(call),
            GeneratedOutputEvent::Finished(reason) => {
                output.finish_reason = anthropic_finish_reason(reason);
            }
            other => anyhow::bail!(
                "Anthropic Messages cannot represent generated {} output",
                other.kind()
            ),
        }
    }
    Ok(())
}

fn validate_tool_output(options: &ToolConstraintOptions, calls: &[ToolCall]) -> anyhow::Result<()> {
    let names = calls
        .iter()
        .map(|call| call.name.clone())
        .collect::<Vec<_>>();
    super::openai::validate_tool_choice_output(options, &names)
}

fn tool_unary_response(
    id: String,
    model_id: String,
    input_tokens: u32,
    output: ParsedToolOutput,
    output_format: StructuredOutputFormat,
) -> Response {
    let has_tool_calls = !output.tool_calls.is_empty();
    if let Err(error) =
        output_format.validate_completion(&output.content, has_tool_calls, output.finish_reason)
    {
        return anthropic_error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"));
    }
    let mut content = Vec::new();
    if !output.reasoning.is_empty() {
        content.push(MessageContentBlock::Thinking {
            signature: thinking_signature(&output.reasoning),
            thinking: output.reasoning,
        });
    }
    if !output.content.is_empty() {
        content.push(MessageContentBlock::Text {
            text: output.content,
        });
    }
    content.extend(
        output
            .tool_calls
            .into_iter()
            .map(|call| MessageContentBlock::ToolUse {
                id: call.id,
                name: call.name,
                input: call.arguments,
            }),
    );
    if content.is_empty() {
        content.push(MessageContentBlock::Text {
            text: String::new(),
        });
    }
    Json(MessageEnvelope {
        id,
        kind: "message",
        role: "assistant",
        content,
        model: model_id,
        stop_reason: Some(if has_tool_calls {
            "tool_use"
        } else {
            output.finish_reason
        }),
        stop_sequence: None,
        usage: Usage {
            input_tokens,
            output_tokens: output.completion_tokens,
            output_tokens_details: (output.thinking_tokens > 0).then_some(OutputTokensDetails {
                thinking_tokens: output.thinking_tokens,
            }),
        },
    })
    .into_response()
}

pub(crate) struct CollectedOutput {
    pub(crate) content: Option<String>,
    pub(crate) reasoning: String,
    pub(crate) tool_calls: Vec<ToolCall>,
    pub(crate) finish_reason: &'static str,
    pub(crate) completion_tokens: u32,
    pub(crate) thinking_tokens: u32,
}

pub(crate) fn collected_response(
    model_id: String,
    input_tokens: u32,
    output: CollectedOutput,
    output_format: StructuredOutputFormat,
) -> Response {
    tool_unary_response(
        gen_msg_id(),
        model_id,
        input_tokens,
        ParsedToolOutput {
            content: output.content.unwrap_or_default(),
            reasoning: output.reasoning,
            tool_calls: output.tool_calls,
            finish_reason: output.finish_reason,
            completion_tokens: output.completion_tokens,
            thinking_tokens: output.thinking_tokens,
        },
        output_format,
    )
}

fn utf8_fragments(value: &str, max_bytes: usize) -> Vec<&str> {
    if value.is_empty() {
        return vec![""];
    }
    let mut fragments = Vec::new();
    let mut start = 0;
    while start < value.len() {
        let mut end = (start + max_bytes).min(value.len());
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        fragments.push(&value[start..end]);
        start = end;
    }
    fragments
}

pub(crate) struct ToolStreamEncoder {
    message_id: String,
    model_id: String,
    input_tokens: u32,
    next_index: usize,
    open_thinking: Option<(usize, String)>,
    open_text_index: Option<usize>,
    call_names: Vec<String>,
    content: String,
    output_format: StructuredOutputFormat,
}

impl ToolStreamEncoder {
    pub(crate) fn new(
        message_id: String,
        model_id: String,
        input_tokens: u32,
        output_format: StructuredOutputFormat,
    ) -> Self {
        Self {
            message_id,
            model_id,
            input_tokens,
            next_index: 0,
            open_thinking: None,
            open_text_index: None,
            call_names: Vec::new(),
            content: String::new(),
            output_format,
        }
    }

    pub(crate) fn message_start(&self) -> Bytes {
        format_event(
            "message_start",
            &serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model_id,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0}
                }
            }),
        )
    }

    fn close_text(&mut self, frames: &mut Vec<Bytes>) {
        if let Some(index) = self.open_text_index.take() {
            frames.push(format_event(
                "content_block_stop",
                &serde_json::json!({"type": "content_block_stop", "index": index}),
            ));
        }
    }

    fn close_thinking(&mut self, frames: &mut Vec<Bytes>) {
        if let Some((index, thinking)) = self.open_thinking.take() {
            frames.push(format_event(
                "content_block_delta",
                &serde_json::json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": thinking_signature(&thinking)
                    }
                }),
            ));
            frames.push(format_event(
                "content_block_stop",
                &serde_json::json!({"type": "content_block_stop", "index": index}),
            ));
        }
    }

    pub(crate) fn push_events(
        &mut self,
        events: Vec<GeneratedOutputEvent>,
    ) -> anyhow::Result<Vec<Bytes>> {
        let mut frames = Vec::new();
        for event in events {
            match event {
                GeneratedOutputEvent::ReasoningDelta(thinking) if !thinking.is_empty() => {
                    self.close_text(&mut frames);
                    let index = match self.open_thinking.as_mut() {
                        Some((index, accumulated)) => {
                            accumulated.push_str(&thinking);
                            *index
                        }
                        None => {
                            let index = self.next_index;
                            self.next_index += 1;
                            self.open_thinking = Some((index, thinking.clone()));
                            frames.push(format_event(
                                "content_block_start",
                                &serde_json::json!({
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {
                                        "type": "thinking",
                                        "thinking": "",
                                        "signature": ""
                                    }
                                }),
                            ));
                            index
                        }
                    };
                    frames.push(format_event(
                        "content_block_delta",
                        &serde_json::json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "thinking_delta", "thinking": thinking}
                        }),
                    ));
                }
                GeneratedOutputEvent::ReasoningDelta(_) => {}
                GeneratedOutputEvent::TextDelta(text) if !text.is_empty() => {
                    self.close_thinking(&mut frames);
                    self.content.push_str(&text);
                    let index = match self.open_text_index {
                        Some(index) => index,
                        None => {
                            let index = self.next_index;
                            self.next_index += 1;
                            self.open_text_index = Some(index);
                            frames.push(format_event(
                                "content_block_start",
                                &serde_json::json!({
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {"type": "text", "text": ""}
                                }),
                            ));
                            index
                        }
                    };
                    frames.push(format_event(
                        "content_block_delta",
                        &serde_json::json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": text}
                        }),
                    ));
                }
                GeneratedOutputEvent::TextDelta(_) => {}
                GeneratedOutputEvent::ToolCall(call) => {
                    self.close_thinking(&mut frames);
                    self.close_text(&mut frames);
                    let index = self.next_index;
                    self.next_index += 1;
                    self.call_names.push(call.name.clone());
                    frames.push(format_event(
                        "content_block_start",
                        &serde_json::json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.name,
                                "input": {}
                            }
                        }),
                    ));
                    let arguments = serde_json::to_string(&call.arguments)
                        .expect("tool arguments are JSON values");
                    for fragment in utf8_fragments(&arguments, 64) {
                        frames.push(format_event(
                            "content_block_delta",
                            &serde_json::json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": fragment
                                }
                            }),
                        ));
                    }
                    frames.push(format_event(
                        "content_block_stop",
                        &serde_json::json!({"type": "content_block_stop", "index": index}),
                    ));
                }
                GeneratedOutputEvent::Finished(_) => {}
                other => anyhow::bail!(
                    "Anthropic Messages cannot represent generated {} output",
                    other.kind()
                ),
            }
        }
        Ok(frames)
    }

    pub(crate) fn finish(
        mut self,
        options: &ToolConstraintOptions,
        model_finish: &'static str,
        output_tokens: u32,
        thinking_tokens: u32,
    ) -> anyhow::Result<Vec<Bytes>> {
        super::openai::validate_tool_choice_output(options, &self.call_names)?;
        let mut frames = Vec::new();
        self.close_thinking(&mut frames);
        self.close_text(&mut frames);
        if self.next_index == 0 {
            frames.push(format_event(
                "content_block_start",
                &serde_json::json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                }),
            ));
            frames.push(format_event(
                "content_block_stop",
                &serde_json::json!({"type": "content_block_stop", "index": 0}),
            ));
        }
        let stop_reason = if self.call_names.is_empty() {
            model_finish
        } else {
            "tool_use"
        };
        self.output_format.validate_completion(
            &self.content,
            !self.call_names.is_empty(),
            stop_reason,
        )?;
        let usage = if thinking_tokens > 0 {
            serde_json::json!({
                "output_tokens": output_tokens,
                "output_tokens_details": {"thinking_tokens": thinking_tokens}
            })
        } else {
            serde_json::json!({"output_tokens": output_tokens})
        };
        frames.push(format_event(
            "message_delta",
            &serde_json::json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": usage
            }),
        ));
        frames.push(format_event(
            "message_stop",
            &serde_json::json!({"type": "message_stop"}),
        ));
        Ok(frames)
    }
}

async fn serve_via_gs_tools<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
    stream: bool,
    context: ToolResponseContext,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if stream {
        serve_via_gs_tools_stream(state, request, model_id, input_tokens, context).await
    } else {
        serve_via_gs_tools_unary(state, request, model_id, input_tokens, context).await
    }
}

async fn serve_via_scheduler_tools<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
    stream: bool,
    context: ToolResponseContext,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if stream {
        serve_via_scheduler_tools_stream(state, request, model_id, input_tokens, context).await
    } else {
        serve_via_scheduler_tools_unary(state, request, model_id, input_tokens, context).await
    }
}

async fn serve_via_gs_tools_unary<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
    context: ToolResponseContext,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let id = gen_msg_id();
    let decoder_config = context.decoder_config();
    let native_output = context.native_output;
    let output_format = context.output_format.clone();
    let constraint_options = context.constraint_options;
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<ParsedToolOutput> {
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let memory = super::begin_direct_request_memory(&state, &*model_guard, &request)?;
        let mut generation = GenerationStream::new(&*model_guard, tokenizer, request)?;
        let mut decoder = GeneratedOutputDecoder::new_with_native(
            tokenizer,
            Some(decoder_config),
            native_output,
        )?;
        state.record_request_started(input_tokens);
        let mut output = ParsedToolOutput {
            content: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            finish_reason: "end_turn",
            completion_tokens: 0,
            thinking_tokens: 0,
        };
        let mut memory = Some(memory);
        let mut finished = false;
        let mut model_finish = "stop";
        while let Some(event) = generation.next_token()? {
            if let Some(memory) = memory.take() {
                memory.commit();
            }
            output.completion_tokens += 1;
            state.runtime_usage.record_output_tokens(1);
            let events = if event.finish_reason == Some("stop") {
                Vec::new()
            } else {
                decoder.push_token(event.token)?
            };
            if event.finish_reason != Some("stop") && decoder.last_token_was_reasoning() {
                output.thinking_tokens += 1;
            }
            collect_tool_events(&mut output, events)?;
            if let Some(reason) = event.finish_reason {
                model_finish = reason;
                output.finish_reason = match reason {
                    "stop" => "end_turn",
                    "length" => "max_tokens",
                    other => other,
                };
                finished = true;
                break;
            }
        }
        anyhow::ensure!(finished, "generation ended before a terminal event");
        let events = decoder.finish(model_finish)?;
        collect_tool_events(&mut output, events)?;
        validate_tool_output(&constraint_options, &output.tool_calls)?;
        Ok(output)
    })
    .await;

    match result {
        Ok(Ok(output)) => tool_unary_response(id, model_id, input_tokens, output, output_format),
        Ok(Err(error)) => generation_err_to_response(error),
        Err(error) => anthropic_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("generation task failed: {error}"),
        ),
    }
}

async fn admit_tool_request<M>(
    state: &AppState<M>,
    request: GenerateRequest,
) -> std::result::Result<mpsc::UnboundedReceiver<crate::core::scheduler::StepEvent>, Response>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .scheduler_handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .is_err()
    {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler actor unavailable",
        )
            .into_response());
    }
    match reply_rx.await {
        Ok(Ok(AdmitReply { event_rx, .. })) => Ok(event_rx),
        Ok(Err(error)) => Err(admit_err_to_response(error)),
        Err(_) => Err((StatusCode::SERVICE_UNAVAILABLE, "scheduler reply lost").into_response()),
    }
}

async fn serve_via_scheduler_tools_unary<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
    context: ToolResponseContext,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let id = gen_msg_id();
    let mut event_rx = match admit_tool_request(&state, request).await {
        Ok(event_rx) => event_rx,
        Err(response) => return response,
    };
    let decoder_config = context.decoder_config();
    let native_output = context.native_output;
    let output_format = context.output_format.clone();
    let constraint_options = context.constraint_options;
    let mut decoder = match GeneratedOutputDecoder::new_with_native(
        &state.tokenizer,
        Some(decoder_config),
        native_output,
    ) {
        Ok(decoder) => decoder,
        Err(error) => {
            return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"))
        }
    };
    state.record_request_started(input_tokens);
    let mut output = ParsedToolOutput {
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        finish_reason: "end_turn",
        completion_tokens: 0,
        thinking_tokens: 0,
    };
    let mut finished = false;
    let mut model_finish = "stop";
    while let Some(event) = event_rx.recv().await {
        output.completion_tokens += 1;
        state.runtime_usage.record_output_tokens(1);
        let events = if event.finish_reason == Some("stop") {
            Ok(Vec::new())
        } else {
            decoder.push_token(event.token)
        };
        if event.finish_reason != Some("stop") && decoder.last_token_was_reasoning() {
            output.thinking_tokens += 1;
        }
        let events = match events {
            Ok(events) => events,
            Err(error) => {
                return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"))
            }
        };
        if let Err(error) = collect_tool_events(&mut output, events) {
            return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
        }
        if let Some(reason) = event.finish_reason {
            model_finish = reason;
            output.finish_reason = match reason {
                "stop" => "end_turn",
                "length" => "max_tokens",
                other => other,
            };
            finished = true;
            break;
        }
    }
    if !finished {
        return anthropic_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler stream ended before a terminal event".to_owned(),
        );
    }
    let events = match decoder.finish(model_finish) {
        Ok(events) => events,
        Err(error) => {
            return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"))
        }
    };
    if let Err(error) = collect_tool_events(&mut output, events) {
        return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
    }
    if let Err(error) = validate_tool_output(&constraint_options, &output.tool_calls) {
        return anthropic_error_response(StatusCode::BAD_REQUEST, format!("{error:#}"));
    }
    tool_unary_response(id, model_id, input_tokens, output, output_format)
}

fn tool_sse_response(rx: mpsc::Receiver<std::result::Result<Bytes, std::io::Error>>) -> Response {
    let body = Body::from_stream(ReceiverStream::new(rx));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap()
}

async fn serve_via_gs_tools_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
    context: ToolResponseContext,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let (init_tx, init_rx) = oneshot::channel::<anyhow::Result<()>>();
    let message_id = gen_msg_id();
    tokio::task::spawn_blocking(move || {
        let decoder_config = context.decoder_config();
        let native_output = context.native_output;
        let output_format = context.output_format.clone();
        let constraint_options = context.constraint_options;
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let memory = match super::begin_direct_request_memory(&state, &*model_guard, &request) {
            Ok(memory) => memory,
            Err(error) => {
                let _ = init_tx.send(Err(error));
                return;
            }
        };
        let mut generation = match GenerationStream::new(&*model_guard, tokenizer, request) {
            Ok(generation) => generation,
            Err(error) => {
                let _ = init_tx.send(Err(error));
                return;
            }
        };
        let first_event = match generation.next_token() {
            Ok(event) => event,
            Err(error) => {
                let _ = init_tx.send(Err(error));
                return;
            }
        };
        let mut decoder = match GeneratedOutputDecoder::new_with_native(
            tokenizer,
            Some(decoder_config),
            native_output,
        ) {
            Ok(decoder) => decoder,
            Err(error) => {
                let _ = init_tx.send(Err(error));
                return;
            }
        };
        memory.commit();
        state.record_request_started(input_tokens);
        if init_tx.send(Ok(())).is_err() {
            return;
        }
        let mut encoder = ToolStreamEncoder::new(message_id, model_id, input_tokens, output_format);
        if tx.blocking_send(Ok(encoder.message_start())).is_err() {
            return;
        }
        let mut output_tokens = 0_u32;
        let mut thinking_tokens = 0_u32;
        let mut model_finish = "stop";
        let mut finished = false;
        let mut first_event = first_event;
        loop {
            let event = match first_event.take() {
                Some(event) => Some(event),
                None => match generation.next_token() {
                    Ok(event) => event,
                    Err(error) => {
                        let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                        return;
                    }
                },
            };
            let Some(event) = event else {
                break;
            };
            output_tokens += 1;
            state.runtime_usage.record_output_tokens(1);
            let events = if event.finish_reason == Some("stop") {
                Ok(Vec::new())
            } else {
                decoder.push_token(event.token)
            };
            if event.finish_reason != Some("stop") && decoder.last_token_was_reasoning() {
                thinking_tokens += 1;
            }
            let events = match events {
                Ok(events) => events,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                    return;
                }
            };
            let frames = match encoder.push_events(events) {
                Ok(frames) => frames,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                    return;
                }
            };
            for frame in frames {
                if tx.blocking_send(Ok(frame)).is_err() {
                    return;
                }
            }
            if let Some(reason) = event.finish_reason {
                model_finish = reason;
                finished = true;
                break;
            }
        }
        if !finished {
            let error = anyhow::anyhow!("generation ended before a terminal event");
            let _ = tx.blocking_send(Ok(format_stream_error(&error)));
            return;
        }
        let events = match decoder.finish(model_finish) {
            Ok(events) => events,
            Err(error) => {
                let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                return;
            }
        };
        let frames = match encoder.push_events(events) {
            Ok(frames) => frames,
            Err(error) => {
                let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                return;
            }
        };
        for frame in frames {
            if tx.blocking_send(Ok(frame)).is_err() {
                return;
            }
        }
        let stop_reason = anthropic_finish_reason(
            GeneratedFinishReason::from_generation(model_finish, !encoder.call_names.is_empty())
                .expect("generation finish reason already validated"),
        );
        let frames = match encoder.finish(
            &constraint_options,
            stop_reason,
            output_tokens,
            thinking_tokens,
        ) {
            Ok(frames) => frames,
            Err(error) => {
                let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                return;
            }
        };
        for frame in frames {
            if tx.blocking_send(Ok(frame)).is_err() {
                return;
            }
        }
    });

    match init_rx.await {
        Ok(Ok(())) => tool_sse_response(rx),
        Ok(Err(error)) => generation_err_to_response(error),
        Err(error) => anthropic_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("generation initialization channel closed: {error}"),
        ),
    }
}

async fn serve_via_scheduler_tools_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
    context: ToolResponseContext,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let mut event_rx = match admit_tool_request(&state, request).await {
        Ok(event_rx) => event_rx,
        Err(response) => return response,
    };
    let decoder_config = context.decoder_config();
    let native_output = context.native_output;
    let output_format = context.output_format.clone();
    let constraint_options = context.constraint_options;
    state.record_request_started(input_tokens);
    let tokenizer = state.tokenizer.clone();
    let runtime_usage = state.runtime_usage.clone();
    let message_id = gen_msg_id();
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    tokio::spawn(async move {
        let mut decoder = match GeneratedOutputDecoder::new_with_native(
            &tokenizer,
            Some(decoder_config),
            native_output,
        ) {
            Ok(decoder) => decoder,
            Err(error) => {
                let _ = tx.send(Ok(format_stream_error(&error))).await;
                return;
            }
        };
        let mut encoder = ToolStreamEncoder::new(message_id, model_id, input_tokens, output_format);
        if tx.send(Ok(encoder.message_start())).await.is_err() {
            return;
        }
        let mut output_tokens = 0_u32;
        let mut thinking_tokens = 0_u32;
        let mut model_finish = "stop";
        let mut finished = false;
        while let Some(event) = event_rx.recv().await {
            output_tokens += 1;
            runtime_usage.record_output_tokens(1);
            let events = if event.finish_reason == Some("stop") {
                Ok(Vec::new())
            } else {
                decoder.push_token(event.token)
            };
            if event.finish_reason != Some("stop") && decoder.last_token_was_reasoning() {
                thinking_tokens += 1;
            }
            let events = match events {
                Ok(events) => events,
                Err(error) => {
                    let _ = tx.send(Ok(format_stream_error(&error))).await;
                    return;
                }
            };
            let frames = match encoder.push_events(events) {
                Ok(frames) => frames,
                Err(error) => {
                    let _ = tx.send(Ok(format_stream_error(&error))).await;
                    return;
                }
            };
            for frame in frames {
                if tx.send(Ok(frame)).await.is_err() {
                    return;
                }
            }
            if let Some(reason) = event.finish_reason {
                model_finish = reason;
                finished = true;
                break;
            }
        }
        if !finished {
            let error = anyhow::anyhow!("scheduler stream ended before a terminal event");
            let _ = tx.send(Ok(format_stream_error(&error))).await;
            return;
        }
        let events = match decoder.finish(model_finish) {
            Ok(events) => events,
            Err(error) => {
                let _ = tx.send(Ok(format_stream_error(&error))).await;
                return;
            }
        };
        let frames = match encoder.push_events(events) {
            Ok(frames) => frames,
            Err(error) => {
                let _ = tx.send(Ok(format_stream_error(&error))).await;
                return;
            }
        };
        for frame in frames {
            if tx.send(Ok(frame)).await.is_err() {
                return;
            }
        }
        let stop_reason = anthropic_finish_reason(
            GeneratedFinishReason::from_generation(model_finish, !encoder.call_names.is_empty())
                .expect("generation finish reason already validated"),
        );
        let frames = match encoder.finish(
            &constraint_options,
            stop_reason,
            output_tokens,
            thinking_tokens,
        ) {
            Ok(frames) => frames,
            Err(error) => {
                let _ = tx.send(Ok(format_stream_error(&error))).await;
                return;
            }
        };
        for frame in frames {
            if tx.send(Ok(frame)).await.is_err() {
                return;
            }
        }
    });
    tool_sse_response(rx)
}

async fn serve_via_gs_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
    output_format: StructuredOutputFormat,
    native_output: Option<NativeOutputDecoderConfig>,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let (init_tx, init_rx) = oneshot::channel::<anyhow::Result<()>>();
    let id = gen_msg_id();
    let id_for_task = id.clone();
    let model_id_for_task = model_id.clone();

    tokio::task::spawn_blocking(move || {
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let memory = match super::begin_direct_request_memory(&state, &*model_guard, &request) {
            Ok(memory) => memory,
            Err(error) => {
                let _ = init_tx.send(Err(error));
                return;
            }
        };
        let mut stream = match GenerationStream::new(&*model_guard, tokenizer, request) {
            Ok(stream) => stream,
            Err(error) => {
                let _ = init_tx.send(Err(error));
                return;
            }
        };
        let first_event = match stream.next_token() {
            Ok(event) => event,
            Err(error) => {
                let _ = init_tx.send(Err(error));
                return;
            }
        };
        memory.commit();
        state.record_request_started(input_tokens);
        if init_tx.send(Ok(())).is_err() {
            return;
        }

        // 1. message_start
        let start_payload = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": id_for_task,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model_id_for_task,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        });
        if tx
            .blocking_send(Ok(format_event("message_start", &start_payload)))
            .is_err()
        {
            return;
        }
        let mut decoder =
            match GeneratedOutputDecoder::new_with_native(tokenizer, None, native_output) {
                Ok(decoder) => decoder,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                    return;
                }
            };
        let mut encoder =
            ToolStreamEncoder::new(id_for_task, model_id_for_task, input_tokens, output_format);

        // 2..N. Protocol-neutral events become Anthropic content blocks.
        let mut output_tokens: u32 = 0;
        let mut thinking_tokens: u32 = 0;
        let mut model_finish: &'static str = "stop";
        let mut finished = false;
        let mut first_event = Some(first_event);
        loop {
            let event = match first_event.take() {
                Some(event) => Ok(event),
                None => stream.next_token(),
            };
            match event {
                Ok(Some(ev)) => {
                    let mut events = if ev.finish_reason == Some("stop") {
                        Vec::new()
                    } else {
                        match decoder.push_token(ev.token) {
                            Ok(events) => events,
                            Err(error) => {
                                let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                                return;
                            }
                        }
                    };
                    if ev.finish_reason != Some("stop") && decoder.last_token_was_reasoning() {
                        thinking_tokens += 1;
                    }
                    if let Some(reason) = ev.finish_reason {
                        model_finish = reason;
                        match decoder.finish(reason) {
                            Ok(tail) => events.extend(tail),
                            Err(error) => {
                                let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                                return;
                            }
                        }
                        finished = true;
                    }
                    let frames = match encoder.push_events(events) {
                        Ok(frames) => frames,
                        Err(error) => {
                            let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                            return;
                        }
                    };
                    for frame in frames {
                        if tx.blocking_send(Ok(frame)).is_err() {
                            return;
                        }
                    }
                    output_tokens += 1;
                    state.runtime_usage.record_output_tokens(1);
                    if ev.finish_reason.is_some() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    let payload = serde_json::json!({
                        "type": "error",
                        "error": {"message": e.to_string()}
                    });
                    let _ = tx.blocking_send(Ok(format_event("error", &payload)));
                    return;
                }
            }
        }
        if !finished {
            let error = anyhow::anyhow!("generation ended before a terminal event");
            let _ = tx.blocking_send(Ok(format_stream_error(&error)));
            return;
        }
        let stop_reason = anthropic_finish_reason(
            GeneratedFinishReason::from_generation(model_finish, false)
                .expect("generation finish reason already validated"),
        );
        let frames = match encoder.finish(
            &ToolConstraintOptions::default(),
            stop_reason,
            output_tokens,
            thinking_tokens,
        ) {
            Ok(frames) => frames,
            Err(error) => {
                let _ = tx.blocking_send(Ok(format_stream_error(&error)));
                return;
            }
        };
        for frame in frames {
            if tx.blocking_send(Ok(frame)).is_err() {
                return;
            }
        }
    });

    match init_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return generation_err_to_response(error),
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("generation initialization channel closed: {error}"),
            )
                .into_response();
        }
    }

    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap()
}

/// Text-only short-prompt streaming path via SchedulerActor (3b-4 swap-in).
/// Emits the same 6-event SSE sequence as `serve_via_gs_stream`:
///   message_start → content_block_start → N × content_block_delta →
///   content_block_stop → message_delta → message_stop.
pub async fn serve_via_scheduler_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    serve_via_scheduler_stream_with_output_format(
        state,
        request,
        model_id,
        input_tokens,
        StructuredOutputFormat::Text,
        None,
    )
    .await
}

async fn serve_via_scheduler_stream_with_output_format<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
    output_format: StructuredOutputFormat,
    native_output: Option<NativeOutputDecoderConfig>,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let msg_id = gen_msg_id();

    // 1. Admit request to the actor.
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .scheduler_handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler actor unavailable",
        )
            .into_response();
    }
    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match reply_rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return admit_err_to_response(e);
        }
        Err(_) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "scheduler reply lost").into_response();
        }
    };
    state.record_request_started(input_tokens);

    // 2. Spawn forwarder that emits the 6-event SSE sequence.
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let msg_id_for_task = msg_id.clone();
    let model_id_for_task = model_id.clone();
    let tokenizer = state.tokenizer.clone();
    let runtime_usage = state.runtime_usage.clone();

    tokio::spawn(async move {
        let mut decoder =
            match GeneratedOutputDecoder::new_with_native(&tokenizer, None, native_output) {
                Ok(decoder) => decoder,
                Err(error) => {
                    let _ = tx.send(Ok(format_stream_error(&error))).await;
                    return;
                }
            };
        let mut encoder = ToolStreamEncoder::new(
            msg_id_for_task,
            model_id_for_task,
            input_tokens,
            output_format,
        );
        if tx.send(Ok(encoder.message_start())).await.is_err() {
            return;
        }
        let mut output_tokens: u32 = 0;
        let mut thinking_tokens: u32 = 0;
        let mut model_finish: &'static str = "stop";
        let mut finished = false;
        while let Some(ev) = event_rx.recv().await {
            let mut events = if ev.finish_reason == Some("stop") {
                Vec::new()
            } else {
                match decoder.push_token(ev.token) {
                    Ok(events) => events,
                    Err(error) => {
                        let _ = tx.send(Ok(format_stream_error(&error))).await;
                        return;
                    }
                }
            };
            if ev.finish_reason != Some("stop") && decoder.last_token_was_reasoning() {
                thinking_tokens += 1;
            }
            if let Some(reason) = ev.finish_reason {
                model_finish = reason;
                match decoder.finish(reason) {
                    Ok(tail) => events.extend(tail),
                    Err(error) => {
                        let _ = tx.send(Ok(format_stream_error(&error))).await;
                        return;
                    }
                }
                finished = true;
            }
            let frames = match encoder.push_events(events) {
                Ok(frames) => frames,
                Err(error) => {
                    let _ = tx.send(Ok(format_stream_error(&error))).await;
                    return;
                }
            };
            for frame in frames {
                if tx.send(Ok(frame)).await.is_err() {
                    return;
                }
            }
            output_tokens += 1;
            runtime_usage.record_output_tokens(1);
            if ev.finish_reason.is_some() {
                break;
            }
        }
        if !finished {
            let error = anyhow::anyhow!("scheduler stream ended before a terminal event");
            let _ = tx.send(Ok(format_stream_error(&error))).await;
            return;
        }
        let stop_reason = anthropic_finish_reason(
            GeneratedFinishReason::from_generation(model_finish, false)
                .expect("generation finish reason already validated"),
        );
        let frames = match encoder.finish(
            &ToolConstraintOptions::default(),
            stop_reason,
            output_tokens,
            thinking_tokens,
        ) {
            Ok(frames) => frames,
            Err(error) => {
                let _ = tx.send(Ok(format_stream_error(&error))).await;
                return;
            }
        };
        for frame in frames {
            if tx.send(Ok(frame)).await.is_err() {
                return;
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap()
}

async fn serve_via_gs_unary<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
    output_format: StructuredOutputFormat,
    native_output: Option<NativeOutputDecoderConfig>,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let id = gen_msg_id();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<ParsedToolOutput> {
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let memory = super::begin_direct_request_memory(&state, &*model_guard, &request)?;
        let mut stream = GenerationStream::new(&*model_guard, tokenizer, request)?;
        let mut decoder = GeneratedOutputDecoder::new_with_native(tokenizer, None, native_output)?;
        state.record_request_started(input_tokens);
        let mut output = ParsedToolOutput {
            content: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            finish_reason: "end_turn",
            completion_tokens: 0,
            thinking_tokens: 0,
        };
        let mut memory = Some(memory);
        let mut finished = false;
        loop {
            let next = stream.next_token()?;
            if let Some(memory) = memory.take() {
                memory.commit();
            }
            let Some(ev) = next else {
                break;
            };
            let events = if ev.finish_reason == Some("stop") {
                Vec::new()
            } else {
                decoder.push_token(ev.token)?
            };
            if ev.finish_reason != Some("stop") && decoder.last_token_was_reasoning() {
                output.thinking_tokens += 1;
            }
            collect_tool_events(&mut output, events)?;
            output.completion_tokens += 1;
            if let Some(reason) = ev.finish_reason {
                collect_tool_events(&mut output, decoder.finish(reason)?)?;
                finished = true;
                break;
            }
        }
        anyhow::ensure!(finished, "generation ended before a terminal event");
        state
            .runtime_usage
            .record_output_tokens(u64::from(output.completion_tokens));
        Ok(output)
    })
    .await;

    let output = match result {
        Ok(Ok(output)) => output,
        Ok(Err(err)) => return generation_err_to_response(err),
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")).into_response();
        }
    };

    tool_unary_response(id, model_id, input_tokens, output, output_format)
}

/// Text-only short-prompt unary path via SchedulerActor (3b-4 swap-in).
pub async fn serve_via_scheduler_unary<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    serve_via_scheduler_unary_with_output_format(
        state,
        request,
        model_id,
        input_tokens,
        StructuredOutputFormat::Text,
        None,
    )
    .await
}

async fn serve_via_scheduler_unary_with_output_format<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
    output_format: StructuredOutputFormat,
    native_output: Option<NativeOutputDecoderConfig>,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let id = gen_msg_id();

    // 1. Admit.
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .scheduler_handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler actor unavailable",
        )
            .into_response();
    }
    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match reply_rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return admit_err_to_response(e);
        }
        Err(_) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "scheduler reply lost").into_response();
        }
    };
    state.record_request_started(input_tokens);

    // 2. Drain committed tokens through the protocol-neutral decoder.
    let mut decoder =
        match GeneratedOutputDecoder::new_with_native(&state.tokenizer, None, native_output) {
            Ok(decoder) => decoder,
            Err(error) => return generation_err_to_response(error),
        };
    let mut output = ParsedToolOutput {
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        finish_reason: "end_turn",
        completion_tokens: 0,
        thinking_tokens: 0,
    };
    let mut finished = false;
    while let Some(ev) = event_rx.recv().await {
        output.completion_tokens += 1;
        let events = if ev.finish_reason == Some("stop") {
            Ok(Vec::new())
        } else {
            decoder.push_token(ev.token)
        };
        if ev.finish_reason != Some("stop") && decoder.last_token_was_reasoning() {
            output.thinking_tokens += 1;
        }
        if let Err(error) = events.and_then(|events| collect_tool_events(&mut output, events)) {
            return generation_err_to_response(error);
        }
        if let Some(reason) = ev.finish_reason {
            if let Err(error) = decoder
                .finish(reason)
                .and_then(|events| collect_tool_events(&mut output, events))
            {
                return generation_err_to_response(error);
            }
            finished = true;
            break;
        }
    }
    if !finished {
        return generation_err_to_response(anyhow::anyhow!(
            "scheduler stream ended before a terminal event"
        ));
    }
    state
        .runtime_usage
        .record_output_tokens(u64::from(output.completion_tokens));

    tool_unary_response(id, model_id, input_tokens, output, output_format)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_adapter_rejects_unmapped_typed_output() {
        let mut output = ParsedToolOutput {
            content: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            finish_reason: "end_turn",
            completion_tokens: 0,
            thinking_tokens: 0,
        };
        let error = collect_tool_events(
            &mut output,
            vec![GeneratedOutputEvent::AudioDelta(
                crate::core::generated_output::AudioChunk {
                    data: vec![1],
                    mime_type: "audio/pcm".to_owned(),
                    sample_rate_hz: None,
                    channels: None,
                },
            )],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("cannot represent generated audio"));
        assert!(output.content.is_empty());
    }

    #[test]
    fn event_format_uses_event_line_prefix_and_double_newline() {
        let payload = serde_json::json!({"type": "message_stop"});
        let bytes = format_event("message_stop", &payload);
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(s.starts_with("event: message_stop\ndata: "));
        assert!(s.ends_with("\n\n"));
        assert!(s.contains("\"type\":\"message_stop\""));
    }

    #[test]
    fn messages_routes_streaming_and_unary_scheduler_requests() {
        assert_eq!(messages_route(true, true), MessagesRoute::SchedulerStream);
        assert_eq!(messages_route(false, true), MessagesRoute::SchedulerUnary);
        assert_eq!(
            messages_route(true, false),
            MessagesRoute::GenerationStreamStream
        );
        assert_eq!(
            messages_route(false, false),
            MessagesRoute::GenerationStreamUnary
        );
    }

    #[test]
    fn six_event_sequence_kinds_match_anthropic_protocol() {
        let kinds = [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ];
        for k in kinds {
            assert!(!k.is_empty());
            assert!(k.chars().all(|c| c.is_ascii_lowercase() || c == '_'));
        }
    }

    #[test]
    fn message_envelope_serializes_with_anthropic_fields() {
        let env = MessageEnvelope {
            id: "msg_1".into(),
            kind: "message",
            role: "assistant",
            content: vec![MessageContentBlock::Text { text: "hi".into() }],
            model: "qwen3.5-4b".into(),
            stop_reason: Some("end_turn"),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 3,
                output_tokens: 1,
                output_tokens_details: None,
            },
        };
        let s = serde_json::to_string(&env).unwrap();
        assert!(s.contains("\"type\":\"message\""));
        assert!(s.contains("\"role\":\"assistant\""));
        assert!(s.contains("\"text\":\"hi\""));
        assert!(s.contains("\"stop_reason\":\"end_turn\""));
        assert!(s.contains("\"input_tokens\":3"));
        assert!(s.contains("\"output_tokens\":1"));
    }

    #[test]
    fn build_sampler_uses_model_defaults_when_request_omits_sampling_params() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "messages": [],
            "max_tokens": 8
        }))
        .expect("messages request");
        let defaults = super::super::SamplingDefaults {
            temperature: Some(0.7),
            top_p: Some(0.8),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
        };

        let sampler = build_sampler(&req, defaults);

        assert_eq!(sampler.temperature, 0.7);
        assert_eq!(sampler.top_p, Some(0.8));
        assert_eq!(sampler.top_k, Some(40));
        assert_eq!(sampler.repetition_penalty, Some(1.1));
    }

    #[test]
    fn build_sampler_prefers_native_request_values_and_keeps_internal_penalty_default() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "messages": [],
            "max_tokens": 8,
            "temperature": 0.2,
            "top_p": 0.6,
            "top_k": 16
        }))
        .expect("messages request");
        let defaults = super::super::SamplingDefaults {
            temperature: Some(0.7),
            top_p: Some(0.8),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
        };

        let sampler = build_sampler(&req, defaults);

        assert_eq!(sampler.temperature, 0.2);
        assert_eq!(sampler.top_p, Some(0.6));
        assert_eq!(sampler.top_k, Some(16));
        assert_eq!(sampler.repetition_penalty, Some(1.1));
    }

    #[test]
    fn messages_request_rejects_nonstandard_repetition_penalty() {
        let error = serde_json::from_value::<MessagesRequest>(serde_json::json!({
            "messages": [],
            "repetition_penalty": 1.1
        }))
        .unwrap_err();
        assert!(error.to_string().contains("repetition_penalty"), "{error}");
    }

    #[test]
    fn messages_sampling_contract_accepts_boundaries_and_rejects_invalid_values() {
        for (temperature, top_p, top_k) in [(0.0, 0.01, 1), (1.0, 1.0, 128)] {
            let req: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }))
            .unwrap();
            req.validate_sampling().unwrap();
        }

        for body in [
            serde_json::json!({"messages": [], "temperature": -0.1}),
            serde_json::json!({"messages": [], "temperature": 1.1}),
            serde_json::json!({"messages": [], "top_p": 0.0}),
            serde_json::json!({"messages": [], "top_p": 1.1}),
            serde_json::json!({"messages": [], "top_k": 0}),
            serde_json::json!({"messages": [], "top_k": -1}),
        ] {
            let req: MessagesRequest = serde_json::from_value(body).unwrap();
            assert!(req.validate_sampling().is_err());
        }

        let mut non_finite: MessagesRequest =
            serde_json::from_value(serde_json::json!({"messages": []})).unwrap();
        non_finite.temperature = Some(f32::NAN);
        assert!(non_finite.validate_sampling().is_err());
        non_finite.temperature = None;
        non_finite.top_p = Some(f32::INFINITY);
        assert!(non_finite.validate_sampling().is_err());
    }

    #[tokio::test]
    async fn messages_http_contract_rejects_unknown_and_invalid_sampling_fields_with_400() {
        use axum::body::Body;
        use axum::extract::rejection::JsonRejection;
        use axum::http::Request;
        use axum::routing::post;
        use axum::Router;
        use tower::ServiceExt;

        async fn validate(
            payload: std::result::Result<Json<MessagesRequest>, JsonRejection>,
        ) -> Response {
            let req = match payload {
                Ok(Json(req)) => req,
                Err(error) => {
                    return anthropic_error_response(StatusCode::BAD_REQUEST, error.body_text());
                }
            };
            match req.validate_sampling() {
                Ok(()) => StatusCode::NO_CONTENT.into_response(),
                Err(error) => anthropic_error_response(StatusCode::BAD_REQUEST, error.to_string()),
            }
        }

        let app = Router::new().route("/v1/messages", post(validate));
        for body in [
            serde_json::json!({"messages": [], "repetition_penalty": 1.1}),
            serde_json::json!({"messages": [], "unsupported": true}),
            serde_json::json!({"messages": [], "temperature": 1.1}),
            serde_json::json!({"messages": [], "top_p": 0.0}),
            serde_json::json!({"messages": [], "top_k": 0}),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::post("/v1/messages")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
        }

        let response = app
            .oneshot(
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "messages": [],
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": 1
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn request_too_large_returns_actionable_json_413() {
        let error = crate::core::SchedulerError::RequestTooLarge {
            required_total_tokens: 273,
            input_tokens: 17,
            requested_max_output_tokens: 256,
            server_max_context_tokens: 128,
            max_allowed_output_tokens: 111,
        };

        let response = admit_err_to_response(error.into());
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
        assert!(response.headers().get(header::RETRY_AFTER).is_none());

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["code"], "request_token_capacity_exceeded");
        assert_eq!(body["required_total_tokens"], 273);
        assert_eq!(body["input_tokens"], 17);
        assert_eq!(body["requested_max_output_tokens"], 256);
        assert_eq!(body["server_max_context_tokens"], 128);
        assert_eq!(body["max_allowed_output_tokens"], 111);
        assert!(body["message"]
            .as_str()
            .unwrap()
            .contains("Dashboard → MAX CONTEXT TOKENS"));
    }

    #[test]
    fn cold_materialization_rejection_is_retryable_http_503() {
        let error = crate::core::SchedulerError::ColdMaterializationUnsafe {
            requested_bytes: 3 * 1024 * 1024 * 1024,
            current_bytes: 20 * 1024 * 1024 * 1024,
            target_bytes: 22 * 1024 * 1024 * 1024,
        };

        let response = admit_err_to_response(error.into());

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .expect("Retry-After header"),
            "5"
        );
    }

    mod wire_tests {
        use super::*;

        fn tool_schema() -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            })
        }

        fn request_with_choice(choice: serde_json::Value) -> MessagesRequest {
            serde_json::from_value(serde_json::json!({
                "model": "local-model",
                "system": [{"type": "text", "text": "Be exact."}],
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": tool_schema(),
                    "strict": true
                }],
                "tool_choice": choice,
                "max_tokens": 32
            }))
            .unwrap()
        }

        fn event_payload(frame: &Bytes) -> serde_json::Value {
            let text = std::str::from_utf8(frame).unwrap();
            let data = text.split_once("\ndata: ").unwrap().1.trim();
            serde_json::from_str(data).unwrap()
        }

        fn content_text(content: &Content) -> Option<&str> {
            match content {
                Content::Text(text) => Some(text),
                Content::Parts(_) => None,
            }
        }

        #[test]
        fn maps_adaptive_and_manual_thinking_to_native_template_kwargs() {
            let adaptive: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "user", "content": "Solve it."}],
                "max_tokens": 4096,
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "medium"}
            }))
            .unwrap();
            let adaptive = adaptive.into_chat_request().unwrap();
            assert_eq!(
                adaptive.chat_template_kwargs,
                Some(serde_json::json!({"enable_thinking": true}))
            );

            let manual: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "user", "content": "Solve it."}],
                "max_tokens": 2048,
                "thinking": {"type": "enabled", "budget_tokens": 1024}
            }))
            .unwrap();
            assert_eq!(
                manual.into_chat_request().unwrap().chat_template_kwargs,
                Some(serde_json::json!({"enable_thinking": true}))
            );

            let disabled: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "user", "content": "Answer directly."}],
                "thinking": {"type": "disabled"}
            }))
            .unwrap();
            assert_eq!(
                disabled.into_chat_request().unwrap().chat_template_kwargs,
                Some(serde_json::json!({"enable_thinking": false}))
            );
        }

        #[test]
        fn validates_thinking_budget_effort_and_display_contracts() {
            for invalid in [
                serde_json::json!({
                    "messages": [],
                    "max_tokens": 2048,
                    "thinking": {"type": "enabled", "budget_tokens": 1023}
                }),
                serde_json::json!({
                    "messages": [],
                    "max_tokens": 1024,
                    "thinking": {"type": "enabled", "budget_tokens": 1024}
                }),
                serde_json::json!({
                    "messages": [],
                    "thinking": {"type": "adaptive", "display": "omitted"}
                }),
                serde_json::json!({
                    "messages": [],
                    "output_config": {"effort": "high"}
                }),
                serde_json::json!({
                    "messages": [],
                    "thinking": {"type": "disabled"},
                    "output_config": {"effort": "low"}
                }),
            ] {
                let request: MessagesRequest = serde_json::from_value(invalid).unwrap();
                assert!(request.into_chat_request().is_err());
            }

            let combined: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [],
                "thinking": {"type": "adaptive"},
                "tools": [{
                    "name": "get_weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }],
                "tool_choice": {"type": "auto"},
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": false
                        }
                    }
                }
            }))
            .unwrap();
            let chat = combined.into_chat_request().unwrap();
            assert_eq!(
                chat.chat_template_kwargs,
                Some(serde_json::json!({"enable_thinking": true}))
            );
            assert!(matches!(
                chat.response_format,
                Some(crate::core::server::openai::ChatResponseFormat::JsonSchema { .. })
            ));
            assert_eq!(chat.tools.as_ref().map(Vec::len), Some(1));

            assert!(
                serde_json::from_value::<MessagesRequest>(serde_json::json!({
                    "messages": [],
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "ultra"}
                }))
                .is_err()
            );
        }

        #[test]
        fn round_trips_signed_leading_thinking_history() {
            let reasoning = "inspect the inputs first";
            let signature = thinking_signature(reasoning);
            let request: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": reasoning, "signature": signature},
                        {"type": "text", "text": "answer"}
                    ]
                }]
            }))
            .unwrap();
            let chat = request.into_chat_request().unwrap();
            assert_eq!(
                chat.messages[0].reasoning_content.as_deref(),
                Some(reasoning)
            );

            let tampered: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{
                    "role": "assistant",
                    "content": [{
                        "type": "thinking",
                        "thinking": "modified",
                        "signature": thinking_signature(reasoning)
                    }]
                }]
            }))
            .unwrap();
            assert!(tampered.into_chat_request().is_err());

            let interleaved: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "visible"},
                        {"type": "thinking", "thinking": reasoning, "signature": thinking_signature(reasoning)}
                    ]
                }]
            }))
            .unwrap();
            assert!(interleaved.into_chat_request().is_err());

            let redacted: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": "opaque"}]
                }]
            }))
            .unwrap();
            assert!(redacted.into_chat_request().is_err());
        }

        #[test]
        fn parses_native_base64_image_block() {
            let body = r#"
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}}
                ]
            }"#;
            let m: AnthropicMessage = serde_json::from_str(body).unwrap();
            assert_eq!(m.role, "user");
            let parts = match m.content {
                AnthropicContent::Parts(p) => p,
                _ => panic!("expected Parts"),
            };
            assert_eq!(parts.len(), 2);
            assert!(matches!(parts[0], AnthropicContentPart::Text { .. }));
            match &parts[1] {
                AnthropicContentPart::Image { source } => {
                    let AnthropicImageSource::Base64 { media_type, data } = source;
                    assert_eq!(media_type, "image/png");
                    assert_eq!(data, "aGVsbG8=");
                }
                _ => panic!("expected Image"),
            }
        }

        #[test]
        fn parses_plain_string_content() {
            let m: AnthropicMessage =
                serde_json::from_str(r#"{"role":"user","content":"hi"}"#).unwrap();
            assert!(matches!(m.content, AnthropicContent::Text(ref t) if t == "hi"));
        }

        #[test]
        fn maps_official_output_config_format_and_injects_final_answer_instruction() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "notes": {"type": "string"}
                },
                "required": ["name"],
                "additionalProperties": false
            });
            let request: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "user", "content": "Extract the name."}],
                "output_config": {
                    "format": {"type": "json_schema", "schema": schema}
                }
            }))
            .unwrap();

            let chat = request.into_chat_request().unwrap();
            let format = chat.structured_output_format().unwrap();

            assert_eq!(format.constraint_schema(), Some(schema));
            assert_eq!(chat.messages[0].role, "system");
            assert!(content_text(&chat.messages[0].content)
                .unwrap()
                .contains("return only one JSON object"));
        }

        #[test]
        fn tool_only_choice_does_not_inject_structured_final_answer_instruction() {
            let request: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "user", "content": "Call a tool."}],
                "tool_choice": {"type": "any"},
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": false
                        }
                    }
                }
            }))
            .unwrap();

            let chat = request.into_chat_request().unwrap();

            assert!(chat.structured_output_format().is_ok());
            assert_eq!(chat.messages.len(), 1);
            assert_eq!(chat.messages[0].role, "user");
        }

        #[test]
        fn structured_outputs_reject_prefill_deprecated_and_unknown_shapes() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": false
            });
            let prefill: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "assistant", "content": "{"}],
                "output_config": {
                    "format": {"type": "json_schema", "schema": schema}
                }
            }))
            .unwrap();
            assert!(prefill.into_chat_request().is_err());

            for invalid in [
                serde_json::json!({
                    "messages": [],
                    "output_format": {"type": "json_schema", "schema": schema}
                }),
                serde_json::json!({
                    "messages": [],
                    "output_config": {
                        "format": {"type": "json_schema", "schema": schema, "extra": true}
                    }
                }),
                serde_json::json!({
                    "messages": [],
                    "output_config": {
                        "format": {"type": "json_object"}
                    }
                }),
            ] {
                assert!(serde_json::from_value::<MessagesRequest>(invalid).is_err());
            }
        }

        #[test]
        fn rejects_openai_image_url_shape() {
            let body = r#"
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
                ]
            }"#;
            assert!(serde_json::from_str::<AnthropicMessage>(body).is_err());
        }

        #[test]
        fn normalizes_anthropic_tools_system_and_forced_choice() {
            let chat = request_with_choice(serde_json::json!({
                "type": "tool",
                "name": "get_weather",
                "disable_parallel_tool_use": true
            }))
            .into_chat_request()
            .unwrap();

            assert_eq!(chat.messages[0].role, "system");
            assert_eq!(content_text(&chat.messages[0].content), Some("Be exact."));
            let tools = chat.tools.as_ref().unwrap();
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].function.name, "get_weather");
            assert_eq!(tools[0].function.parameters, tool_schema());
            assert_eq!(tools[0].function.strict, Some(true));
            assert_eq!(
                chat.tool_choice,
                Some(serde_json::json!({
                    "type": "function",
                    "function": {"name": "get_weather"}
                }))
            );
            assert_eq!(chat.parallel_tool_calls, Some(false));
        }

        #[test]
        fn normalizes_tool_use_and_tool_result_lifecycle() {
            let request: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Tokyo"}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                        {"type": "text", "text": "Summarize it."}
                    ]}
                ],
                "tools": [{"name": "get_weather", "input_schema": tool_schema()}]
            }))
            .unwrap();
            let chat = request.into_chat_request().unwrap();

            assert_eq!(
                chat.messages
                    .iter()
                    .map(|message| message.role.as_str())
                    .collect::<Vec<_>>(),
                vec!["user", "assistant", "tool", "user"]
            );
            assert_eq!(chat.messages[1].tool_calls[0].id, "toolu_1");
            assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("toolu_1"));
            let agent =
                super::super::super::openai::build_agent_messages(&chat.messages, &chat.messages)
                    .unwrap();
            assert_eq!(agent[1].tool_calls[0].id, "toolu_1");
            assert_eq!(agent[2].tool_call_id.as_deref(), Some("toolu_1"));
            assert_eq!(agent[2].content.as_deref(), Some("sunny"));
        }

        #[test]
        fn preserves_tool_result_error_signal_for_native_templates() {
            let request: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true, "content": "timeout"}]}
                ],
                "tools": [{"name": "lookup", "input_schema": {"type": "object", "properties": {}, "additionalProperties": false}}]
            }))
            .unwrap();
            let chat = request.into_chat_request().unwrap();
            assert_eq!(
                content_text(&chat.messages[1].content),
                Some("{\"is_error\":true}\ntimeout")
            );
        }

        #[test]
        fn rejects_unknown_fields_and_invalid_block_roles() {
            assert!(
                serde_json::from_value::<MessagesRequest>(serde_json::json!({
                    "messages": [],
                    "unsupported": true
                }))
                .is_err()
            );

            let user_tool_use: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "user", "content": [{"type": "tool_use", "id": "x", "name": "f", "input": {}}]}]
            }))
            .unwrap();
            assert!(user_tool_use.into_chat_request().is_err());

            let scalar_input: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "f", "input": 1}]}]
            }))
            .unwrap();
            assert!(scalar_input.into_chat_request().is_err());

            let reordered_content: MessagesRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "assistant", "content": [
                    {"type": "tool_use", "id": "x", "name": "f", "input": {}},
                    {"type": "text", "text": "after"}
                ]}]
            }))
            .unwrap();
            assert!(reordered_content.into_chat_request().is_err());
        }

        #[test]
        fn maps_all_anthropic_tool_choice_modes() {
            let auto = request_with_choice(serde_json::json!({"type": "auto"}))
                .into_chat_request()
                .unwrap();
            assert_eq!(auto.tool_choice, Some(serde_json::json!("auto")));
            assert_eq!(auto.parallel_tool_calls, Some(true));

            let any = request_with_choice(serde_json::json!({"type": "any"}))
                .into_chat_request()
                .unwrap();
            assert_eq!(any.tool_choice, Some(serde_json::json!("required")));

            let none = request_with_choice(serde_json::json!({"type": "none"}))
                .into_chat_request()
                .unwrap();
            let prepared =
                super::super::super::openai::prepare_tool_request(&none, Some(ToolDialect::Qwen35))
                    .unwrap()
                    .unwrap();
            assert!(prepared.constraint_options.is_none());
        }

        #[tokio::test]
        async fn unary_response_emits_native_tool_use_blocks() {
            let response = collected_response(
                "local-model".to_owned(),
                12,
                CollectedOutput {
                    content: Some("I will check.".to_owned()),
                    reasoning: String::new(),
                    tool_calls: vec![ToolCall {
                        id: "toolu_1".to_owned(),
                        name: "get_weather".to_owned(),
                        arguments: serde_json::json!({"city": "Tokyo"}),
                    }],
                    finish_reason: "end_turn",
                    completion_tokens: 7,
                    thinking_tokens: 0,
                },
                StructuredOutputFormat::Text,
            );
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["stop_reason"], "tool_use");
            assert_eq!(value["content"][0]["type"], "text");
            assert_eq!(value["content"][1]["type"], "tool_use");
            assert_eq!(value["content"][1]["id"], "toolu_1");
            assert_eq!(value["content"][1]["input"]["city"], "Tokyo");
        }

        #[tokio::test]
        async fn unary_response_emits_signed_thinking_before_text_and_usage_details() {
            let reasoning = "check the evidence".to_owned();
            let response = collected_response(
                "local-model".to_owned(),
                12,
                CollectedOutput {
                    content: Some("final answer".to_owned()),
                    reasoning: reasoning.clone(),
                    tool_calls: Vec::new(),
                    finish_reason: "end_turn",
                    completion_tokens: 7,
                    thinking_tokens: 4,
                },
                StructuredOutputFormat::Text,
            );
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["content"][0]["type"], "thinking");
            assert_eq!(value["content"][0]["thinking"], reasoning);
            assert_eq!(
                value["content"][0]["signature"],
                thinking_signature("check the evidence")
            );
            assert_eq!(value["content"][1]["type"], "text");
            assert_eq!(
                value["usage"]["output_tokens_details"]["thinking_tokens"],
                4
            );
        }

        #[test]
        fn stream_encoder_emits_thinking_signature_then_text() {
            let mut encoder = ToolStreamEncoder::new(
                "msg_thinking".to_owned(),
                "local-model".to_owned(),
                4,
                StructuredOutputFormat::Text,
            );
            let mut frames = vec![encoder.message_start()];
            frames.extend(
                encoder
                    .push_events(vec![
                        GeneratedOutputEvent::ReasoningDelta("check ".to_owned()),
                        GeneratedOutputEvent::ReasoningDelta("evidence".to_owned()),
                        GeneratedOutputEvent::TextDelta("answer".to_owned()),
                    ])
                    .unwrap(),
            );
            frames.extend(
                encoder
                    .finish(&ToolConstraintOptions::default(), "end_turn", 3, 2)
                    .unwrap(),
            );
            let payloads = frames.iter().map(event_payload).collect::<Vec<_>>();
            let kinds = payloads
                .iter()
                .filter_map(|payload| payload["delta"]["type"].as_str().map(ToOwned::to_owned))
                .collect::<Vec<_>>();
            assert_eq!(
                kinds,
                vec![
                    "thinking_delta",
                    "thinking_delta",
                    "signature_delta",
                    "text_delta"
                ]
            );
            let starts = payloads
                .iter()
                .filter(|payload| payload["type"] == "content_block_start")
                .collect::<Vec<_>>();
            assert_eq!(starts[0]["content_block"]["type"], "thinking");
            assert_eq!(starts[1]["content_block"]["type"], "text");
            let signature = payloads
                .iter()
                .find(|payload| payload["delta"]["type"] == "signature_delta")
                .unwrap();
            assert_eq!(
                signature["delta"]["signature"],
                thinking_signature("check evidence")
            );
            let message_delta = payloads
                .iter()
                .find(|payload| payload["type"] == "message_delta")
                .unwrap();
            assert_eq!(
                message_delta["usage"]["output_tokens_details"]["thinking_tokens"],
                2
            );
        }

        #[test]
        fn stream_encoder_emits_parallel_tool_blocks_and_json_deltas() {
            let mut encoder = ToolStreamEncoder::new(
                "msg_1".to_owned(),
                "local-model".to_owned(),
                4,
                StructuredOutputFormat::Text,
            );
            let mut frames = vec![encoder.message_start()];
            frames.extend(
                encoder
                    .push_events(vec![
                        GeneratedOutputEvent::TextDelta("Checking".to_owned()),
                        GeneratedOutputEvent::ToolCall(ToolCall {
                            id: "toolu_1".to_owned(),
                            name: "weather".to_owned(),
                            arguments: serde_json::json!({"city": "東京"}),
                        }),
                        GeneratedOutputEvent::ToolCall(ToolCall {
                            id: "toolu_2".to_owned(),
                            name: "time".to_owned(),
                            arguments: serde_json::json!({"zone": "Asia/Tokyo"}),
                        }),
                    ])
                    .unwrap(),
            );
            frames.extend(
                encoder
                    .finish(&ToolConstraintOptions::default(), "end_turn", 9, 0)
                    .unwrap(),
            );
            let payloads = frames.iter().map(event_payload).collect::<Vec<_>>();
            let starts = payloads
                .iter()
                .filter(|payload| payload["type"] == "content_block_start")
                .collect::<Vec<_>>();
            assert_eq!(starts.len(), 3);
            assert_eq!(starts[0]["content_block"]["type"], "text");
            assert_eq!(starts[1]["content_block"]["type"], "tool_use");
            assert_eq!(starts[1]["index"], 1);
            assert_eq!(starts[2]["index"], 2);

            let partial_json = payloads
                .iter()
                .filter(|payload| payload["type"] == "content_block_delta" && payload["index"] == 1)
                .filter_map(|payload| payload["delta"]["partial_json"].as_str())
                .collect::<String>();
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&partial_json).unwrap(),
                serde_json::json!({"city": "東京"})
            );
            assert_eq!(
                payloads
                    .iter()
                    .find(|payload| payload["type"] == "message_delta")
                    .unwrap()["delta"]["stop_reason"],
                "tool_use"
            );
        }

        #[test]
        fn stream_encoder_enforces_required_and_serial_choices() {
            let required = ToolConstraintOptions {
                choice: crate::core::constrained::ToolChoiceConstraint::Required,
                allow_parallel_calls: true,
            };
            let encoder = ToolStreamEncoder::new(
                "msg_1".to_owned(),
                "local-model".to_owned(),
                1,
                StructuredOutputFormat::Text,
            );
            assert!(encoder.finish(&required, "end_turn", 1, 0).is_err());

            let serial = ToolConstraintOptions {
                choice: crate::core::constrained::ToolChoiceConstraint::Auto,
                allow_parallel_calls: false,
            };
            let mut encoder = ToolStreamEncoder::new(
                "msg_2".to_owned(),
                "local-model".to_owned(),
                1,
                StructuredOutputFormat::Text,
            );
            encoder
                .push_events(vec![
                    GeneratedOutputEvent::ToolCall(ToolCall {
                        id: "a".to_owned(),
                        name: "one".to_owned(),
                        arguments: serde_json::json!({}),
                    }),
                    GeneratedOutputEvent::ToolCall(ToolCall {
                        id: "b".to_owned(),
                        name: "two".to_owned(),
                        arguments: serde_json::json!({}),
                    }),
                ])
                .unwrap();
            assert!(encoder.finish(&serial, "end_turn", 1, 0).is_err());
        }
    }
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::core::server::chat_format::{ChatMessage, Content, ContentPart, ImageUrl};
    use crate::core::server::openai::decode_openai_messages;
    use crate::core::server::vision::expand_decoded_messages;
    use crate::core::server::VisionInputConfig;
    use base64::Engine;

    /// Base64 of the real coco test image (shared by both endpoint paths so the
    /// decoded bytes are identical by construction; the test proves the two
    /// decode paths + shared core agree).
    fn coco_b64() -> String {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/p6_qwen35_vl/coco_sample.jpg"
        ))
        .expect("read coco_sample.jpg fixture");
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn anthropic_one_image(b64: &str) -> Vec<AnthropicMessage> {
        vec![AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicContent::Parts(vec![
                AnthropicContentPart::Text {
                    text: "what is this?".to_string(),
                },
                AnthropicContentPart::Image {
                    source: AnthropicImageSource::Base64 {
                        media_type: "image/jpeg".to_string(),
                        data: b64.to_string(),
                    },
                },
            ]),
        }]
    }

    fn openai_one_image(b64: &str) -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".to_string(),
            content: Content::Parts(vec![
                ContentPart::Text {
                    text: "what is this?".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: format!("data:image/jpeg;base64,{b64}"),
                    },
                },
            ]),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }]
    }

    async fn run_parity(vision_input: VisionInputConfig) {
        let b64 = coco_b64();
        // OpenAI path: data: URL → bytes → shared core.
        let openai_decoded = decode_openai_messages(openai_one_image(&b64)).unwrap();
        let (o_flat, o_pv, o_grid) =
            expand_decoded_messages(openai_decoded, &vision_input).unwrap();
        // Anthropic path: raw base64 → bytes → shared core.
        let anthropic_decoded = decode_anthropic_messages(anthropic_one_image(&b64)).unwrap();
        let (a_flat, a_pv, a_grid) =
            expand_decoded_messages(anthropic_decoded, &vision_input).unwrap();

        // flat text identical
        assert_eq!(o_flat.len(), a_flat.len());
        for (o, a) in o_flat.iter().zip(a_flat.iter()) {
            let (ot, at) = match (&o.content, &a.content) {
                (Content::Text(o), Content::Text(a)) => (o, a),
                _ => panic!("expected flat Content::Text"),
            };
            assert_eq!(ot, at, "flat text mismatch");
        }
        // grid identical
        assert_eq!(o_grid, a_grid, "grid_thw mismatch");
        // pixel_values byte-identical
        let o_pv = o_pv.expect("openai pixels");
        let a_pv = a_pv.expect("anthropic pixels");
        assert_eq!(o_pv.len(), a_pv.len(), "pixel tensor count");
        for (o, a) in o_pv.iter().zip(a_pv.iter()) {
            let od: Vec<f32> = mlx::ops::cast::astype(o, mlx::Dtype::Float32)
                .unwrap()
                .to_vec()
                .unwrap();
            let ad: Vec<f32> = mlx::ops::cast::astype(a, mlx::Dtype::Float32)
                .unwrap()
                .to_vec()
                .unwrap();
            assert_eq!(od, ad, "pixel_values byte mismatch");
        }
    }

    #[tokio::test]
    async fn byte_parity_qwen() {
        run_parity(VisionInputConfig::Qwen {
            spatial_merge_size: 2,
        })
        .await;
    }

    #[tokio::test]
    async fn byte_parity_minicpmv46() {
        run_parity(VisionInputConfig::MiniCpmV46 {
            spatial_merge_size: 4,
        })
        .await;
    }

    #[tokio::test]
    #[ignore = "requires GEMMA4_MODEL pointing at a gemma4 checkpoint with vision_config"]
    async fn byte_parity_gemma4() {
        let dir = std::path::PathBuf::from(
            std::env::var("GEMMA4_MODEL").expect("GEMMA4_MODEL must be set"),
        );
        let loader = crate::core::Loader::open_multimodal(&dir).expect("open_multimodal");
        let vc = crate::models::gemma4::Gemma4Config::from_loader(&loader)
            .expect("Gemma4Config::from_loader")
            .vision_config
            .expect("gemma4 vision_config present");
        run_parity(VisionInputConfig::Gemma4 { vision_config: vc }).await;
    }
}
