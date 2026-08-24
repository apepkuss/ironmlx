//! OpenAI-compatible Chat Completions API: /v1/chat/completions.
//!
//! Supports both streaming (`stream: true` → SSE) and non-streaming
//! (`stream: false` → JSON).

use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(test)]
use axum::{
    body::Body,
    http::{header, StatusCode},
};
use axum::{
    body::Bytes,
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use mlx::Array;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::core::constrained::{ToolChoiceConstraint, ToolConstraintOptions};
use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::generated_output::{
    GeneratedOutputDecoder, GeneratedOutputEvent, ToolOutputDecoderConfig,
};
use crate::core::image_input::{ImageInputError, ImageRequestBudget};
use crate::core::model::Model;
use crate::core::sampler::Sampler;
use crate::core::scheduler::DenseVlMethods;
use crate::core::server::chat_format::render_and_encode;
use crate::core::server::chat_format::{ChatMessage, Content, ContentPart};
use crate::core::server::scheduler_actor::AdmitReply;
use crate::core::server::structured_output::StructuredOutputFormat;
use crate::core::server::vision::{expand_decoded_messages_bounded, DecodedMessage, DecodedPart};
use crate::core::server::VisionInputConfig;
use crate::core::speculative::MtpSpeculativeConfig;
use crate::core::tool_calling::{
    lower_gemma_tool_arguments, lower_gemma_tool_definitions, validate_function_name,
    validate_tool_definitions, AgentMessage, TemplateToolCall, ToolCall, ToolDefinition,
    ToolDialect,
};

use super::api_transport::ApiJson;
use super::{AppState, Gemma4DrafterAppState, RequestAdmissionError, SamplingDefaults};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a SchedulerActor admit Err into an HTTP response. Spec §4.7 + §2 G7:
/// - `SchedulerError::QueueFull` → 503 Service Unavailable + Retry-After: 5
/// - `SchedulerError::RequestTooLarge` → 413 Payload Too Large (no Retry-After)
/// - Memory governor and prefix-store backpressure errors → retryable 503
/// - Non-scheduler admission errors → 400 Bad Request
pub(crate) fn admit_err_to_response(err: anyhow::Error) -> Response {
    super::api_error::ApiError::scheduler_admission(err)
        .into_response(super::api_error::ApiProtocol::OpenAi)
}

pub(crate) fn generation_err_to_response(err: anyhow::Error) -> Response {
    super::api_error::ApiError::generation(err).into_response(super::api_error::ApiProtocol::OpenAi)
}

fn bad_request_response(code: &'static str, message: impl Into<String>) -> Response {
    super::api_error::ApiError::invalid_request(code, message)
        .into_response(super::api_error::ApiProtocol::OpenAi)
}

fn internal_error_response(code: &'static str, message: impl Into<String>) -> Response {
    super::api_error::ApiError::internal(code, message)
        .into_response(super::api_error::ApiProtocol::OpenAi)
}

fn service_unavailable_response(code: &'static str, message: impl Into<String>) -> Response {
    super::api_error::ApiError::service_unavailable(code, message)
        .into_response(super::api_error::ApiProtocol::OpenAi)
}

async fn admit_request<M>(
    state: &AppState<M>,
    request: GenerateRequest,
) -> std::result::Result<AdmitReply, Response>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    match state.request_execution.admit(request).await {
        Ok(reply) => Ok(reply),
        Err(RequestAdmissionError::Rejected(error)) => Err(admit_err_to_response(error)),
        Err(RequestAdmissionError::Unavailable) => Err(service_unavailable_response(
            "execution_unavailable",
            "request execution actor unavailable",
        )),
        Err(RequestAdmissionError::ReplyLost) => Err(service_unavailable_response(
            "execution_reply_lost",
            "request execution actor reply lost",
        )),
    }
}

// ---------------------------------------------------------------------------
// Request / Response shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Option<Vec<OpenAiTool>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Deprecated OpenAI function-calling field. It is surfaced only so API-1
    /// can reject it explicitly instead of silently ignoring it.
    #[serde(default)]
    pub function_call: Option<serde_json::Value>,
    #[serde(default)]
    pub functions: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) response_format: Option<ChatResponseFormat>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Generate until `max_tokens` even when the model emits an EOS token.
    /// Intended for controlled full-length performance measurements.
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Qwen3.8 native reasoning depth. The model's official template accepts
    /// exactly `low`, `medium`, and `xhigh`.
    #[serde(default)]
    pub reasoning_effort: Option<QwenReasoningEffort>,
    /// HuggingFace `apply_chat_template` extra kwargs — passed through as
    /// top-level template render-context variables. Honors Qwen3+'s
    /// `enable_thinking` toggle, vLLM's `tools` / `documents`, etc.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QwenReasoningEffort {
    Low,
    Medium,
    Xhigh,
}

impl QwenReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::Xhigh => "xhigh",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ChatResponseFormat {
    Text {},
    JsonObject {},
    JsonSchema { json_schema: ChatJsonSchema },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatJsonSchema {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) description: Option<String>,
    pub(crate) schema: serde_json::Value,
    #[serde(default)]
    pub(crate) strict: Option<bool>,
}

impl ChatRequest {
    fn resolved_chat_template_kwargs(&self) -> anyhow::Result<Option<serde_json::Value>> {
        let Some(reasoning_effort) = self.reasoning_effort else {
            return Ok(self.chat_template_kwargs.clone());
        };
        let mut kwargs = match self.chat_template_kwargs.clone() {
            None => serde_json::Map::new(),
            Some(serde_json::Value::Object(kwargs)) => kwargs,
            Some(_) => anyhow::bail!("chat_template_kwargs must be a JSON object"),
        };
        let reasoning_effort = serde_json::Value::String(reasoning_effort.as_str().to_owned());
        if let Some(existing) = kwargs.get("reasoning_effort") {
            anyhow::ensure!(
                existing == &reasoning_effort,
                "reasoning_effort conflicts with chat_template_kwargs.reasoning_effort"
            );
        }
        kwargs.insert("reasoning_effort".to_owned(), reasoning_effort);
        Ok(Some(serde_json::Value::Object(kwargs)))
    }

    pub(crate) fn validate_sampling(&self) -> anyhow::Result<()> {
        if let Some(temperature) = self.temperature {
            anyhow::ensure!(
                temperature.is_finite() && (0.0..=2.0).contains(&temperature),
                "temperature must be finite and between 0 and 2"
            );
        }
        if let Some(top_p) = self.top_p {
            anyhow::ensure!(
                top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
                "top_p must be finite and in (0, 1]"
            );
        }
        Ok(())
    }

    pub(crate) fn structured_output_format(&self) -> anyhow::Result<StructuredOutputFormat> {
        let format = match self.response_format.clone() {
            None | Some(ChatResponseFormat::Text {}) => StructuredOutputFormat::Text,
            Some(ChatResponseFormat::JsonObject {}) => StructuredOutputFormat::JsonObject,
            Some(ChatResponseFormat::JsonSchema { json_schema }) => {
                StructuredOutputFormat::JsonSchema {
                    name: json_schema.name,
                    description: json_schema.description,
                    schema: json_schema.schema,
                    strict: json_schema.strict,
                }
            }
        };
        format.validate_contract("response_format")?;
        Ok(format)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolDefinition,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedToolRequest {
    pub(crate) dialect: ToolDialect,
    pub(crate) wire_tools: Vec<OpenAiTool>,
    pub(crate) definitions: Vec<ToolDefinition>,
    pub(crate) model_definitions: Vec<ToolDefinition>,
    pub(crate) constraint_options: Option<ToolConstraintOptions>,
}

#[derive(Debug)]
struct ToolResponseContext {
    dialect: ToolDialect,
    definitions: Vec<ToolDefinition>,
    constraint_options: ToolConstraintOptions,
    output_schema: Option<serde_json::Value>,
    output_format: StructuredOutputFormat,
}

impl ToolResponseContext {
    fn decoder_config(
        self,
    ) -> (
        ToolOutputDecoderConfig,
        ToolConstraintOptions,
        StructuredOutputFormat,
    ) {
        (
            ToolOutputDecoderConfig {
                dialect: self.dialect,
                response_id: uuid::Uuid::new_v4().simple().to_string(),
                definitions: self.definitions,
                output_schema: self.output_schema,
            },
            self.constraint_options,
            self.output_format,
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedToolChoice {
    #[serde(rename = "type")]
    kind: String,
    function: NamedFunctionChoice,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedFunctionChoice {
    name: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

fn default_max_tokens() -> usize {
    256
}

#[derive(Debug, Serialize)]
struct Choice<T> {
    index: u32,
    delta: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct DeltaRole {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct DeltaContent<'a> {
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct DeltaToolCalls {
    tool_calls: Vec<DeltaToolCall>,
}

#[derive(Debug, Serialize)]
struct DeltaToolCall {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    function: DeltaFunctionCall,
}

#[derive(Debug, Serialize)]
struct DeltaFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    arguments: String,
}

#[cfg(test)]
#[derive(Debug, Serialize)]
struct DeltaEmpty {}

#[derive(Debug, Serialize)]
struct ChunkResponse<T> {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice<T>>,
}

#[derive(Debug, Serialize)]
struct StreamUsageChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<serde_json::Value>,
    usage: Usage,
}

impl StreamUsageChunk {
    fn new(id: impl Into<String>, model: impl Into<String>, prompt: u32, completion: u32) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk",
            created: now_unix(),
            model: model.into(),
            choices: Vec::new(),
            usage: Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct CompletionMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<CompletionToolCall>,
}

#[derive(Debug, Serialize)]
struct CompletionToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: CompletionFunctionCall,
}

#[derive(Debug, Serialize)]
struct CompletionFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    index: u32,
    message: CompletionMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

// ---------------------------------------------------------------------------
// Image URL decoding (Step 19.1 + 19.2)
// ---------------------------------------------------------------------------

/// Decode an in-request image data URL. Server-side URL fetching is forbidden.
pub fn decode_image_url(
    url: &str,
    budget: &mut ImageRequestBudget,
) -> Result<Vec<u8>, ImageInputError> {
    budget.decode_data_url(url)
}

// ---------------------------------------------------------------------------
// Multimodal message expansion (Step 18.4 + 19.3 helper)
// ---------------------------------------------------------------------------

/// Decode every OpenAI `image_url` content part into raw bytes, producing the
/// wire-agnostic `DecodedMessage` list consumed by the shared vision core.
pub fn decode_openai_messages(
    messages: Vec<ChatMessage>,
) -> Result<Vec<DecodedMessage>, ImageInputError> {
    let mut budget = ImageRequestBudget::default();
    let mut out: Vec<DecodedMessage> = Vec::with_capacity(messages.len());
    for msg in messages {
        let mut parts: Vec<DecodedPart> = Vec::new();
        match msg.content {
            Content::Text(t) => {
                budget.add_text(&t)?;
                parts.push(DecodedPart::Text(t));
            }
            Content::Parts(ps) => {
                for p in ps {
                    match p {
                        ContentPart::Text { text } => {
                            budget.add_text(&text)?;
                            parts.push(DecodedPart::Text(text));
                        }
                        ContentPart::ImageUrl { image_url } => {
                            let bytes = decode_image_url(&image_url.url, &mut budget)?;
                            parts.push(DecodedPart::Image(bytes));
                        }
                    }
                }
            }
        }
        out.push(DecodedMessage {
            role: msg.role,
            parts,
            reasoning_content: msg.reasoning_content,
        });
    }
    Ok(out)
}

/// Walk `messages`, decode + preprocess every `image_url`, and rewrite to
/// text-with-placeholder. Signature unchanged so the handler call site is
/// untouched; the body now delegates to the shared `vision` core.
pub async fn expand_image_parts_in_messages(
    messages: Vec<ChatMessage>,
    vision_input: &VisionInputConfig,
) -> anyhow::Result<(Vec<ChatMessage>, Option<Vec<Array>>, Vec<(i32, i32, i32)>)> {
    let decoded = decode_openai_messages(messages)?;
    expand_decoded_messages_bounded(decoded, vision_input.clone()).await
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn gen_id() -> String {
    format!("chatcmpl-{}", now_unix())
}

pub(crate) fn prepare_tool_request(
    req: &ChatRequest,
    dialect: Option<ToolDialect>,
) -> anyhow::Result<Option<PreparedToolRequest>> {
    if req.function_call.is_some() || req.functions.is_some() {
        anyhow::bail!(
            "deprecated `functions` / `function_call` fields are not supported; use `tools`"
        );
    }
    let Some(wire_tools) = req.tools.as_ref() else {
        if req
            .messages
            .iter()
            .any(|message| !message.tool_calls.is_empty() || message.tool_call_id.is_some())
        {
            anyhow::bail!("tool-call history requires a non-empty `tools` array");
        }
        if req.parallel_tool_calls.is_some() {
            anyhow::bail!("parallel_tool_calls requires a non-empty `tools` array");
        }
        if let Some(choice) = &req.tool_choice {
            if choice.as_str() != Some("none") {
                anyhow::bail!("tool_choice requires a non-empty `tools` array");
            }
        }
        return Ok(None);
    };

    let dialect = dialect.ok_or_else(|| {
        anyhow::anyhow!(
            "this model chat template does not support API-1 tools; supported native dialects: Qwen3.5/Qwen3.6/Qwen3.8, Gemma, GLM, Llama, and MiniCPM"
        )
    })?;
    if wire_tools.len() > 128 {
        anyhow::bail!("tools exceeds the supported maximum of 128 functions");
    }
    if req
        .chat_template_kwargs
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .is_some_and(|kwargs| kwargs.contains_key("tools"))
    {
        anyhow::bail!("chat_template_kwargs.tools conflicts with the top-level `tools` field");
    }
    let mut definitions = Vec::with_capacity(wire_tools.len());
    for tool in wire_tools {
        if tool.kind != "function" {
            anyhow::bail!("only tools[].type=`function` is supported");
        }
        definitions.push(tool.function.clone());
    }
    validate_tool_definitions(&definitions)?;
    let constraint_options = resolve_tool_constraint_options(req, &definitions)?;
    let model_definitions = if dialect == ToolDialect::Gemma {
        lower_gemma_tool_definitions(&definitions)?
    } else {
        definitions.clone()
    };
    let model_wire_tools = wire_tools
        .iter()
        .zip(&model_definitions)
        .map(|(wire, definition)| OpenAiTool {
            kind: wire.kind.clone(),
            function: definition.clone(),
        })
        .collect();
    Ok(Some(PreparedToolRequest {
        dialect,
        wire_tools: model_wire_tools,
        definitions,
        model_definitions,
        constraint_options,
    }))
}

fn resolve_tool_constraint_options(
    req: &ChatRequest,
    definitions: &[ToolDefinition],
) -> anyhow::Result<Option<ToolConstraintOptions>> {
    let choice = match req.tool_choice.as_ref() {
        None => ToolChoiceConstraint::Auto,
        Some(choice) if choice.as_str() == Some("auto") => ToolChoiceConstraint::Auto,
        Some(choice) if choice.as_str() == Some("none") => return Ok(None),
        Some(choice) if choice.as_str() == Some("required") => ToolChoiceConstraint::Required,
        Some(choice) if choice.is_object() => {
            let selected: NamedToolChoice = serde_json::from_value(choice.clone()).map_err(|_| {
                anyhow::anyhow!(
                    "tool_choice object must be {{\"type\":\"function\",\"function\":{{\"name\":\"...\"}}}}"
                )
            })?;
            anyhow::ensure!(
                selected.kind == "function",
                "tool_choice.type must be `function`"
            );
            validate_function_name(&selected.function.name)?;
            anyhow::ensure!(
                definitions
                    .iter()
                    .any(|definition| definition.name == selected.function.name),
                "tool_choice references unknown function `{}`",
                selected.function.name
            );
            ToolChoiceConstraint::Function(selected.function.name)
        }
        Some(_) => {
            anyhow::bail!("tool_choice must be `auto`, `none`, `required`, or a specified function")
        }
    };
    Ok(Some(ToolConstraintOptions {
        choice,
        allow_parallel_calls: req.parallel_tool_calls.unwrap_or(true),
    }))
}

pub(crate) fn compile_output_constraint(
    tokenizer: &crate::core::Tokenizer,
    prepared_tools: Option<&PreparedToolRequest>,
    output_schema: Option<&serde_json::Value>,
) -> anyhow::Result<Option<crate::core::constrained::ConstraintPlan>> {
    compile_output_constraint_with_native(tokenizer, prepared_tools, output_schema, None)
}

pub(crate) fn compile_output_constraint_with_native(
    tokenizer: &crate::core::Tokenizer,
    prepared_tools: Option<&PreparedToolRequest>,
    output_schema: Option<&serde_json::Value>,
    native_output: Option<crate::core::native_output::NativeOutputDecoderConfig>,
) -> anyhow::Result<Option<crate::core::constrained::ConstraintPlan>> {
    let enabled_reasoning = native_output.filter(|config| config.reasoning_enabled);
    prepared_tools
        .and_then(|prepared| {
            prepared
                .constraint_options
                .as_ref()
                .map(|options| (prepared, options))
        })
        .map(|(prepared, options)| {
            if matches!(&options.choice, ToolChoiceConstraint::Auto) {
                if let Some(schema) = output_schema {
                    if let Some(reasoning) = enabled_reasoning {
                        return tokenizer.compile_tool_or_json_constraint_with_reasoning(
                            &prepared.model_definitions,
                            options,
                            schema,
                            reasoning,
                        );
                    }
                    return tokenizer.compile_tool_or_json_constraint(
                        &prepared.model_definitions,
                        options,
                        schema,
                    );
                }
            }
            tokenizer.compile_tool_constraint(&prepared.model_definitions, options)
        })
        .transpose()
        .and_then(|tool_constraint| match (tool_constraint, output_schema) {
            (Some(constraint), _) => Ok(Some(constraint)),
            (None, Some(schema)) => match enabled_reasoning {
                Some(reasoning) => tokenizer
                    .compile_json_output_constraint_with_reasoning(schema, reasoning)
                    .map(Some),
                None => tokenizer.compile_json_output_constraint(schema).map(Some),
            },
            (None, None) => Ok(None),
        })
}

pub(crate) fn allows_structured_final_output(prepared_tools: Option<&PreparedToolRequest>) -> bool {
    prepared_tools
        .and_then(|prepared| prepared.constraint_options.as_ref())
        .is_none_or(|options| matches!(&options.choice, ToolChoiceConstraint::Auto))
}

pub(crate) fn tool_template_kwargs(
    base: Option<serde_json::Value>,
    prepared: &PreparedToolRequest,
) -> anyhow::Result<serde_json::Value> {
    let mut object = match base {
        Some(serde_json::Value::Object(object)) => object,
        Some(_) => anyhow::bail!("chat_template_kwargs must be a JSON object"),
        None => serde_json::Map::new(),
    };
    if prepared.constraint_options.is_some() {
        object.insert(
            "tools".to_owned(),
            serde_json::to_value(&prepared.wire_tools)?,
        );
    }
    Ok(serde_json::Value::Object(object))
}

pub(crate) fn build_agent_messages(
    original: &[ChatMessage],
    flattened: &[ChatMessage],
) -> anyhow::Result<Vec<AgentMessage>> {
    if original.len() != flattened.len() {
        anyhow::bail!("internal message expansion changed conversation length");
    }
    let mut output = Vec::with_capacity(original.len());
    let mut unresolved = std::collections::HashSet::<String>::new();
    let mut seen_ids = std::collections::HashSet::<String>::new();

    for (wire, flat) in original.iter().zip(flattened) {
        let content = match &flat.content {
            Content::Text(text) => text.clone(),
            Content::Parts(_) => anyhow::bail!("message image parts were not flattened"),
        };
        match wire.role.as_str() {
            "system" | "user" => {
                if !unresolved.is_empty() {
                    anyhow::bail!("all assistant tool calls must receive tool results before the next conversation turn");
                }
                if !wire.tool_calls.is_empty() || wire.tool_call_id.is_some() {
                    anyhow::bail!("role `{}` cannot contain tool call fields", wire.role);
                }
                output.push(AgentMessage {
                    role: wire.role.clone(),
                    content: Some(content),
                    reasoning_content: wire.reasoning_content.clone(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                });
            }
            "assistant" => {
                if !unresolved.is_empty() {
                    anyhow::bail!("previous assistant tool calls are missing tool results");
                }
                if wire.tool_call_id.is_some() {
                    anyhow::bail!("assistant messages cannot contain tool_call_id");
                }
                let mut calls = Vec::with_capacity(wire.tool_calls.len());
                for call in &wire.tool_calls {
                    if call.kind != "function" {
                        anyhow::bail!("assistant tool_calls only support type=`function`");
                    }
                    validate_function_name(&call.function.name)?;
                    if call.id.is_empty() || !seen_ids.insert(call.id.clone()) {
                        anyhow::bail!("assistant tool call IDs must be non-empty and unique");
                    }
                    let arguments: serde_json::Value =
                        serde_json::from_str(&call.function.arguments).map_err(|error| {
                            anyhow::anyhow!(
                                "assistant tool call `{}` arguments are not valid JSON: {error}",
                                call.id
                            )
                        })?;
                    if !arguments.is_object() {
                        anyhow::bail!(
                            "assistant tool call `{}` arguments must encode a JSON object",
                            call.id
                        );
                    }
                    unresolved.insert(call.id.clone());
                    calls.push(TemplateToolCall::from(ToolCall {
                        id: call.id.clone(),
                        name: call.function.name.clone(),
                        arguments,
                    }));
                }
                output.push(AgentMessage {
                    role: wire.role.clone(),
                    content: (!content.is_empty()).then_some(content),
                    reasoning_content: wire.reasoning_content.clone(),
                    tool_calls: calls,
                    tool_call_id: None,
                });
            }
            "tool" => {
                if !wire.tool_calls.is_empty() {
                    anyhow::bail!("tool result messages cannot contain tool_calls");
                }
                let call_id = wire
                    .tool_call_id
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("tool result message requires tool_call_id"))?;
                if !unresolved.remove(call_id) {
                    anyhow::bail!("orphan or duplicate tool result for tool_call_id `{call_id}`");
                }
                output.push(AgentMessage {
                    role: wire.role.clone(),
                    content: Some(content),
                    reasoning_content: wire.reasoning_content.clone(),
                    tool_calls: Vec::new(),
                    tool_call_id: Some(call_id.clone()),
                });
            }
            role => anyhow::bail!("unsupported message role `{role}` for tool calling"),
        }
    }
    if !unresolved.is_empty() {
        anyhow::bail!("assistant tool calls are missing corresponding tool result messages");
    }
    Ok(output)
}

pub(crate) fn render_tool_prompt(
    tokenizer: &crate::core::Tokenizer,
    messages: &[AgentMessage],
    kwargs: &serde_json::Value,
    prepared: &PreparedToolRequest,
) -> anyhow::Result<Vec<u32>> {
    if prepared.dialect != ToolDialect::Gemma {
        return tokenizer.render_and_encode_tool_prompt(messages, kwargs);
    }
    let mut lowered = messages.to_vec();
    for message in &mut lowered {
        for call in &mut message.tool_calls {
            call.function.arguments = lower_gemma_tool_arguments(
                &prepared.definitions,
                &call.function.name,
                &call.function.arguments,
            )?;
        }
    }
    tokenizer.render_and_encode_tool_prompt(&lowered, kwargs)
}

pub(crate) fn build_sampler(req: &ChatRequest, defaults: SamplingDefaults) -> Sampler {
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
    if let Some(k) = defaults.top_k {
        if k > 0 {
            s = s.with_top_k(k);
        }
    }
    if let Some(penalty) = defaults.repetition_penalty {
        if penalty > 0.0 && penalty != 1.0 {
            s = s.with_repetition_penalty(penalty);
        }
    }
    if let Some(seed) = req.seed {
        s = s.with_seed(seed);
    }
    s
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatCompletionsRoute {
    SchedulerStream,
    GenerationStreamStream,
    SchedulerUnary,
    GenerationStreamUnary,
}

fn chat_completions_route(stream: bool, use_scheduler: bool) -> ChatCompletionsRoute {
    match (stream, use_scheduler) {
        (true, true) => ChatCompletionsRoute::SchedulerStream,
        (true, false) => ChatCompletionsRoute::GenerationStreamStream,
        (false, true) => ChatCompletionsRoute::SchedulerUnary,
        (false, false) => ChatCompletionsRoute::GenerationStreamUnary,
    }
}

pub(crate) fn stop_token_ids_for_request(eos_token_ids: &[u32], ignore_eos: bool) -> Vec<u32> {
    if ignore_eos {
        Vec::new()
    } else {
        eos_token_ids.to_vec()
    }
}

// ---------------------------------------------------------------------------
// Handler (Step 19.3)
// ---------------------------------------------------------------------------

pub(crate) async fn chat_completions<M>(
    State(state): State<AppState<M>>,
    ApiJson(req): ApiJson<ChatRequest>,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    chat_completions_with_state(state, req).await
}

pub(crate) async fn chat_completions_with_state<M>(
    state: AppState<M>,
    mut req: ChatRequest,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if let Err(error) = req.validate_sampling() {
        return bad_request_response("invalid_sampling_parameters", format!("{error:#}"));
    }
    // Extract fields we need after consuming req.messages.
    let stream = req.stream;
    let include_usage = req
        .stream_options
        .map(|options| options.include_usage)
        .unwrap_or(false);
    let ignore_eos = req.ignore_eos;

    let max_tokens = req.max_tokens;
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let sampler = build_sampler(&req, state.sampling_defaults);
    if let Err(error) = super::validate_prompt_lookup_sampler(state.prompt_lookup_enabled, sampler)
    {
        return bad_request_response("invalid_sampling_parameters", format!("{error:#}"));
    }
    let output_format = match req.structured_output_format() {
        Ok(format) => format,
        Err(error) => return bad_request_response("invalid_response_format", format!("{error:#}")),
    };
    let output_schema = output_format.constraint_schema();
    let prepared_tools = match prepare_tool_request(&req, state.tokenizer.tool_dialect()) {
        Ok(prepared) => prepared,
        Err(error) => return bad_request_response("invalid_tools", format!("{error:#}")),
    };
    if allows_structured_final_output(prepared_tools.as_ref()) {
        output_format.apply_prompt_instruction(&mut req.messages);
    }
    let chat_template_kwargs = match req.resolved_chat_template_kwargs() {
        Ok(kwargs) => kwargs,
        Err(error) => {
            return bad_request_response("invalid_reasoning_effort", format!("{error:#}"))
        }
    };
    let original_messages = prepared_tools.as_ref().map(|_| req.messages.clone());

    let (image_token_id, spatial_merge_size) =
        crate::core::server::vision::derive_image_token_and_merge(
            &state.vision_input,
            &state.tokenizer,
        );

    // Expand multimodal content parts: decode images, build pixel_values,
    // rewrite messages to text-with-placeholder.
    let (flat_messages, pixel_values, image_grid_thw) =
        match expand_image_parts_in_messages(req.messages, &state.vision_input).await {
            Ok(t) => t,
            Err(e) => {
                return super::security::image_error_response(
                    e,
                    super::api_error::ApiProtocol::OpenAi,
                )
            }
        };

    let image_grid_thw_opt = if image_grid_thw.is_empty() {
        None
    } else {
        Some(image_grid_thw)
    };

    let prompt_ids_result = if let Some(prepared) = &prepared_tools {
        let agent_messages = build_agent_messages(
            original_messages
                .as_deref()
                .expect("captured for tool request"),
            &flat_messages,
        );
        agent_messages.and_then(|messages| {
            let kwargs = tool_template_kwargs(chat_template_kwargs, prepared)?;
            render_tool_prompt(&state.tokenizer, &messages, &kwargs, prepared)
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
            return bad_request_response(
                "invalid_prompt",
                format!("chat template / tokenize: {e}"),
            );
        }
    };

    let prompt_len = prompt_ids.len();
    let scheduler_config = state.scheduler_request_config(prompt_len, max_tokens);

    // Routing: short-prompt, paged-prefix-cache, and model-limited chunked
    // long-prompt requests use SchedulerActor; other chunked long prompts keep
    // using GenerationStream.
    // B1-p2.4: VL fallback removed — VL requests now route through Scheduler
    // via Scheduler::admit/admit_mid + batched_prefill_vl.
    let use_scheduler = state.request_execution.is_dflash2()
        || super::should_route_to_scheduler::<M>(
            prompt_len,
            scheduler_config.prefill_chunk_size,
            state.b_max,
            state.paged_prefix_cache_enabled,
            state.force_scheduler_for_greedy && sampler.is_pipelinable(),
        );

    let stop_token_ids = stop_token_ids_for_request(state.tokenizer.eos_token_ids(), ignore_eos);
    let constraint = match compile_output_constraint(
        &state.tokenizer,
        prepared_tools.as_ref(),
        output_schema.as_ref(),
    ) {
        Ok(constraint) => constraint,
        Err(error) => {
            return bad_request_response(
                "invalid_output_schema",
                format!("compile output decoding constraint: {error:#}"),
            );
        }
    };
    let prompt_tokens = prompt_len as u32;
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
        image_spatial_merge_size: spatial_merge_size,
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
            output_schema: matches!(&constraint_options.choice, ToolChoiceConstraint::Auto)
                .then(|| output_schema.clone())
                .flatten(),
            output_format: output_format.clone(),
            constraint_options,
        };
        return match chat_completions_route(stream, use_scheduler) {
            ChatCompletionsRoute::SchedulerStream | ChatCompletionsRoute::SchedulerUnary => {
                serve_via_scheduler_tools(
                    state,
                    request,
                    model_label,
                    prompt_tokens,
                    include_usage,
                    stream,
                    tool_context,
                )
                .await
            }
            ChatCompletionsRoute::GenerationStreamStream
            | ChatCompletionsRoute::GenerationStreamUnary => {
                serve_via_gs_tools(
                    state,
                    request,
                    model_label,
                    prompt_tokens,
                    include_usage,
                    stream,
                    tool_context,
                )
                .await
            }
        };
    }

    match chat_completions_route(stream, use_scheduler) {
        ChatCompletionsRoute::SchedulerStream => {
            serve_via_scheduler_stream(
                state,
                request,
                model_label,
                prompt_tokens,
                include_usage,
                output_format,
            )
            .await
        }
        ChatCompletionsRoute::GenerationStreamStream => {
            serve_via_gs_stream(
                state,
                request,
                model_label,
                prompt_tokens,
                include_usage,
                output_format,
            )
            .await
        }
        ChatCompletionsRoute::SchedulerUnary => {
            serve_via_scheduler_unary(state, request, model_label, prompt_tokens, output_format)
                .await
        }
        ChatCompletionsRoute::GenerationStreamUnary => {
            serve_via_gs_unary(state, request, model_label, prompt_tokens, output_format).await
        }
    }
}

pub(crate) async fn gemma4_drafter_chat_completions(
    State(state): State<Gemma4DrafterAppState>,
    ApiJson(req): ApiJson<ChatRequest>,
) -> Response {
    chat_completions_with_gemma4_drafter_state(state, req).await
}

pub(crate) async fn chat_completions_with_gemma4_drafter_state(
    state: Gemma4DrafterAppState,
    mut req: ChatRequest,
) -> Response {
    if let Err(error) = req.validate_sampling() {
        return bad_request_response("invalid_sampling_parameters", format!("{error:#}"));
    }
    let output_format = match req.structured_output_format() {
        Ok(format) => format,
        Err(error) => return bad_request_response("invalid_response_format", format!("{error:#}")),
    };
    let output_schema = output_format.constraint_schema();
    let prepared_tools = match prepare_tool_request(&req, state.base.tokenizer.tool_dialect()) {
        Ok(prepared) => prepared,
        Err(error) => return bad_request_response("invalid_tools", format!("{error:#}")),
    };
    if allows_structured_final_output(prepared_tools.as_ref()) {
        output_format.apply_prompt_instruction(&mut req.messages);
    }
    let original_messages = prepared_tools.as_ref().map(|_| req.messages.clone());
    let stream = req.stream;
    let include_usage = req
        .stream_options
        .map(|options| options.include_usage)
        .unwrap_or(false);
    let ignore_eos = req.ignore_eos;
    let max_tokens = req.max_tokens;
    let model_label = req
        .model
        .clone()
        .unwrap_or_else(|| state.base.model_id.clone());
    let sampler = build_sampler(&req, state.base.sampling_defaults);
    let _cfg = match MtpSpeculativeConfig::new(state.mtp_draft_tokens, sampler) {
        Ok(cfg) => cfg,
        Err(e) => return bad_request_response("invalid_sampling_parameters", format!("{e:#}")),
    };
    let chat_template_kwargs = match req.resolved_chat_template_kwargs() {
        Ok(kwargs) => kwargs,
        Err(error) => {
            return bad_request_response("invalid_reasoning_effort", format!("{error:#}"))
        }
    };

    let (image_token_id, spatial_merge_size) =
        crate::core::server::vision::derive_image_token_and_merge(
            &state.base.vision_input,
            &state.base.tokenizer,
        );
    let (flat_messages, pixel_values, image_grid_thw) =
        match expand_image_parts_in_messages(req.messages, &state.base.vision_input).await {
            Ok(t) => t,
            Err(e) => {
                return super::security::image_error_response(
                    e,
                    super::api_error::ApiProtocol::OpenAi,
                )
            }
        };
    let image_grid_thw_opt = if image_grid_thw.is_empty() {
        None
    } else {
        Some(image_grid_thw)
    };
    let prompt_ids_result = if let Some(prepared) = &prepared_tools {
        build_agent_messages(
            original_messages
                .as_deref()
                .expect("captured for tool request"),
            &flat_messages,
        )
        .and_then(|messages| {
            let kwargs = tool_template_kwargs(chat_template_kwargs, prepared)?;
            render_tool_prompt(&state.base.tokenizer, &messages, &kwargs, prepared)
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
            return bad_request_response(
                "invalid_prompt",
                format!("chat template / tokenize: {e}"),
            );
        }
    };
    let prompt_len = prompt_ids.len();
    let total_tokens = prompt_len.saturating_add(max_tokens);
    if total_tokens > state.base.effective_cap_max {
        let error = crate::core::SchedulerError::RequestTooLarge {
            required_total_tokens: total_tokens,
            input_tokens: prompt_len,
            requested_max_output_tokens: max_tokens,
            server_max_context_tokens: state.base.effective_cap_max,
            max_allowed_output_tokens: state.base.effective_cap_max.saturating_sub(prompt_len),
        };
        return super::api_error::ApiError::request_token_capacity(&error)
            .into_response(super::api_error::ApiProtocol::OpenAi);
    }
    let scheduler_config = state.base.scheduler_request_config(prompt_len, max_tokens);
    let stop_token_ids =
        stop_token_ids_for_request(state.base.tokenizer.eos_token_ids(), ignore_eos);
    let constraint = match compile_output_constraint(
        &state.base.tokenizer,
        prepared_tools.as_ref(),
        output_schema.as_ref(),
    ) {
        Ok(constraint) => constraint,
        Err(error) => {
            return bad_request_response(
                "invalid_output_schema",
                format!("compile output decoding constraint: {error:#}"),
            );
        }
    };
    let prompt_tokens = prompt_len as u32;
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
        image_spatial_merge_size: spatial_merge_size,
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
            output_schema: matches!(&constraint_options.choice, ToolChoiceConstraint::Auto)
                .then(|| output_schema.clone())
                .flatten(),
            output_format: output_format.clone(),
            constraint_options,
        };
        return serve_via_scheduler_tools(
            state.base,
            request,
            model_label,
            prompt_tokens,
            include_usage,
            stream,
            tool_context,
        )
        .await;
    }

    if stream {
        serve_via_scheduler_stream(
            state.base,
            request,
            model_label,
            prompt_tokens,
            include_usage,
            output_format,
        )
        .await
    } else {
        serve_via_scheduler_unary(
            state.base,
            request,
            model_label,
            prompt_tokens,
            output_format,
        )
        .await
    }
}

#[derive(Debug)]
struct ParsedAssistantOutput {
    content: String,
    tool_calls: Vec<ToolCall>,
    finish_reason: &'static str,
    completion_tokens: u32,
}

fn collect_generated_events(
    output: &mut ParsedAssistantOutput,
    events: Vec<GeneratedOutputEvent>,
) -> anyhow::Result<()> {
    for event in events {
        match event {
            GeneratedOutputEvent::TextDelta(text) => output.content.push_str(&text),
            GeneratedOutputEvent::ToolCall(call) => output.tool_calls.push(call),
            GeneratedOutputEvent::Finished(reason) => output.finish_reason = reason.as_str(),
            other => anyhow::bail!(
                "Chat Completions cannot represent generated {} output",
                other.kind()
            ),
        }
    }
    Ok(())
}

fn finish_output_decoder(
    decoder: &mut GeneratedOutputDecoder<'_>,
    output: &mut ParsedAssistantOutput,
    model_finish: &'static str,
) -> anyhow::Result<()> {
    collect_generated_events(output, decoder.finish(model_finish)?)
}

pub(crate) fn validate_tool_choice_output(
    options: &ToolConstraintOptions,
    call_names: &[String],
) -> anyhow::Result<()> {
    if !options.allow_parallel_calls {
        anyhow::ensure!(
            call_names.len() <= 1,
            "parallel_tool_calls=false produced more than one tool call"
        );
    }
    match &options.choice {
        ToolChoiceConstraint::Auto => Ok(()),
        ToolChoiceConstraint::Required => {
            anyhow::ensure!(
                !call_names.is_empty(),
                "tool_choice=required completed without a tool call"
            );
            Ok(())
        }
        ToolChoiceConstraint::Function(expected) => {
            anyhow::ensure!(
                call_names.len() == 1 && call_names[0] == *expected,
                "specified function `{expected}` must be called exactly once"
            );
            Ok(())
        }
    }
}

fn tool_completion_response(
    id: String,
    model_id: String,
    prompt_tokens: u32,
    output: ParsedAssistantOutput,
) -> Response {
    let message = CompletionMessage {
        role: "assistant",
        content: (!output.content.is_empty()).then_some(output.content),
        tool_calls: output
            .tool_calls
            .into_iter()
            .map(|call| CompletionToolCall {
                id: call.id,
                kind: "function",
                function: CompletionFunctionCall {
                    name: call.name,
                    arguments: serde_json::to_string(&call.arguments)
                        .expect("tool arguments are JSON values"),
                },
            })
            .collect(),
    };
    let response = CompletionResponse {
        id,
        object: "chat.completion",
        created: now_unix(),
        model: model_id,
        choices: vec![CompletionChoice {
            index: 0,
            message,
            finish_reason: output.finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: prompt_tokens + output.completion_tokens,
        },
    };
    Json(response).into_response()
}

#[cfg(test)]
fn tool_stream_response(
    id: String,
    model_id: String,
    prompt_tokens: u32,
    include_usage: bool,
    output: ParsedAssistantOutput,
) -> Response {
    let mut frames = Vec::<std::result::Result<Bytes, std::io::Error>>::new();
    frames.push(Ok(format_sse_data(&ChunkResponse {
        id: id.clone(),
        object: "chat.completion.chunk",
        created: now_unix(),
        model: model_id.clone(),
        choices: vec![Choice {
            index: 0,
            delta: DeltaRole {
                role: "assistant",
                content: String::new(),
            },
            finish_reason: None,
        }],
    })));
    if !output.content.is_empty() {
        frames.push(Ok(format_sse_data(&ChunkResponse {
            id: id.clone(),
            object: "chat.completion.chunk",
            created: now_unix(),
            model: model_id.clone(),
            choices: vec![Choice {
                index: 0,
                delta: DeltaContent {
                    content: &output.content,
                },
                finish_reason: None,
            }],
        })));
    }
    for (index, call) in output.tool_calls.iter().enumerate() {
        frames.push(Ok(format_sse_data(&ChunkResponse {
            id: id.clone(),
            object: "chat.completion.chunk",
            created: now_unix(),
            model: model_id.clone(),
            choices: vec![Choice {
                index: 0,
                delta: DeltaToolCalls {
                    tool_calls: vec![DeltaToolCall {
                        index,
                        id: Some(call.id.clone()),
                        kind: Some("function"),
                        function: DeltaFunctionCall {
                            name: Some(call.name.clone()),
                            arguments: String::new(),
                        },
                    }],
                },
                finish_reason: None,
            }],
        })));
        let arguments =
            serde_json::to_string(&call.arguments).expect("tool arguments are JSON values");
        for fragment in utf8_fragments(&arguments, 64) {
            frames.push(Ok(format_sse_data(&ChunkResponse {
                id: id.clone(),
                object: "chat.completion.chunk",
                created: now_unix(),
                model: model_id.clone(),
                choices: vec![Choice {
                    index: 0,
                    delta: DeltaToolCalls {
                        tool_calls: vec![DeltaToolCall {
                            index,
                            id: None,
                            kind: None,
                            function: DeltaFunctionCall {
                                name: None,
                                arguments: fragment.to_owned(),
                            },
                        }],
                    },
                    finish_reason: None,
                }],
            })));
        }
    }
    frames.push(Ok(format_sse_data(&ChunkResponse {
        id: id.clone(),
        object: "chat.completion.chunk",
        created: now_unix(),
        model: model_id.clone(),
        choices: vec![Choice {
            index: 0,
            delta: serde_json::json!({}),
            finish_reason: Some(output.finish_reason),
        }],
    })));
    if include_usage {
        frames.push(Ok(format_sse_data(&StreamUsageChunk::new(
            id,
            model_id,
            prompt_tokens,
            output.completion_tokens,
        ))));
    }
    frames.push(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
    let body = Body::from_stream(tokio_stream::iter(frames));
    super::api_transport::sse_response(body)
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

fn format_tool_output_events(
    id: &str,
    model_id: &str,
    events: Vec<GeneratedOutputEvent>,
    next_call_index: &mut usize,
    call_names: &mut Vec<String>,
    finish_reason: &mut Option<&'static str>,
    content: &mut String,
) -> anyhow::Result<Vec<Bytes>> {
    let mut frames = Vec::new();
    for event in events {
        match event {
            GeneratedOutputEvent::TextDelta(text) => {
                content.push_str(&text);
                frames.push(format_sse_data(&ChunkResponse {
                    id: id.to_owned(),
                    object: "chat.completion.chunk",
                    created: now_unix(),
                    model: model_id.to_owned(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaContent { content: &text },
                        finish_reason: None,
                    }],
                }));
            }
            GeneratedOutputEvent::ToolCall(call) => {
                call_names.push(call.name.clone());
                let index = *next_call_index;
                *next_call_index += 1;
                frames.push(format_sse_data(&ChunkResponse {
                    id: id.to_owned(),
                    object: "chat.completion.chunk",
                    created: now_unix(),
                    model: model_id.to_owned(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaToolCalls {
                            tool_calls: vec![DeltaToolCall {
                                index,
                                id: Some(call.id),
                                kind: Some("function"),
                                function: DeltaFunctionCall {
                                    name: Some(call.name),
                                    arguments: String::new(),
                                },
                            }],
                        },
                        finish_reason: None,
                    }],
                }));
                let arguments =
                    serde_json::to_string(&call.arguments).expect("tool arguments are JSON values");
                for fragment in utf8_fragments(&arguments, 64) {
                    frames.push(format_sse_data(&ChunkResponse {
                        id: id.to_owned(),
                        object: "chat.completion.chunk",
                        created: now_unix(),
                        model: model_id.to_owned(),
                        choices: vec![Choice {
                            index: 0,
                            delta: DeltaToolCalls {
                                tool_calls: vec![DeltaToolCall {
                                    index,
                                    id: None,
                                    kind: None,
                                    function: DeltaFunctionCall {
                                        name: None,
                                        arguments: fragment.to_owned(),
                                    },
                                }],
                            },
                            finish_reason: None,
                        }],
                    }));
                }
            }
            GeneratedOutputEvent::Finished(reason) => {
                anyhow::ensure!(
                    finish_reason.replace(reason.as_str()).is_none(),
                    "generated output emitted more than one terminal event"
                );
            }
            other => anyhow::bail!(
                "Chat Completions cannot represent generated {} output",
                other.kind()
            ),
        }
    }
    Ok(frames)
}

fn tool_role_chunk(id: &str, model_id: &str) -> Bytes {
    format_sse_data(&ChunkResponse {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created: now_unix(),
        model: model_id.to_owned(),
        choices: vec![Choice {
            index: 0,
            delta: DeltaRole {
                role: "assistant",
                content: String::new(),
            },
            finish_reason: None,
        }],
    })
}

fn tool_finish_chunk(id: &str, model_id: &str, finish_reason: &'static str) -> Bytes {
    format_sse_data(&ChunkResponse {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created: now_unix(),
        model: model_id.to_owned(),
        choices: vec![Choice {
            index: 0,
            delta: serde_json::json!({}),
            finish_reason: Some(finish_reason),
        }],
    })
}

async fn serve_via_gs_tools_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
    include_usage: bool,
    tool_context: ToolResponseContext,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let started_at = Instant::now();
    let (decoder_config, constraint_options, output_format) = tool_context.decoder_config();
    let id = gen_id();
    let id_for_task = id.clone();
    let model_for_task = model_id.clone();
    let (tx, rx, disconnect) = super::api_transport::disconnect_aware_sse_channel(8);
    let (init_tx, init_rx) = oneshot::channel::<anyhow::Result<()>>();
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
        let mut decoder = match GeneratedOutputDecoder::new(tokenizer, Some(decoder_config)) {
            Ok(decoder) => decoder,
            Err(error) => {
                let _ = init_tx.send(Err(error));
                return;
            }
        };
        memory.commit();
        let mut performance = state.record_request_started(prompt_tokens, started_at);
        if init_tx.send(Ok(())).is_err()
            || tx
                .blocking_send(Ok(tool_role_chunk(&id_for_task, &model_for_task)))
                .is_err()
        {
            return;
        }

        let mut completion_tokens = 0_u32;
        let mut next_call_index = 0_usize;
        let mut call_names = Vec::new();
        let mut first_event = Some(first_event);
        let mut model_finish = "stop";
        let mut typed_finish = None;
        let mut content = String::new();
        loop {
            if disconnect.is_cancelled() {
                return;
            }
            let event_result = match first_event.take() {
                Some(event) => Ok(event),
                None => generation.next_token(),
            };
            let event = match event_result {
                Ok(Some(event)) => event,
                Ok(None) => break,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(format_sse_error(&error)));
                    return;
                }
            };
            completion_tokens += 1;
            performance.record_output_tokens(1);
            let events = if event.finish_reason == Some("stop") {
                Ok(Vec::new())
            } else {
                decoder.push_token(event.token)
            };
            let events = match events {
                Ok(events) => events,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(format_sse_error(&error)));
                    return;
                }
            };
            for frame in match format_tool_output_events(
                &id_for_task,
                &model_for_task,
                events,
                &mut next_call_index,
                &mut call_names,
                &mut typed_finish,
                &mut content,
            ) {
                Ok(frames) => frames,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(format_sse_error(&error)));
                    return;
                }
            } {
                if tx.blocking_send(Ok(frame)).is_err() {
                    return;
                }
            }
            if let Some(reason) = event.finish_reason {
                model_finish = reason;
                break;
            }
        }
        if disconnect.is_cancelled() {
            return;
        }
        let events = match decoder.finish(model_finish) {
            Ok(events) => events,
            Err(error) => {
                let _ = tx.blocking_send(Ok(format_sse_error(&error)));
                return;
            }
        };
        for frame in match format_tool_output_events(
            &id_for_task,
            &model_for_task,
            events,
            &mut next_call_index,
            &mut call_names,
            &mut typed_finish,
            &mut content,
        ) {
            Ok(frames) => frames,
            Err(error) => {
                let _ = tx.blocking_send(Ok(format_sse_error(&error)));
                return;
            }
        } {
            if tx.blocking_send(Ok(frame)).is_err() {
                return;
            }
        }
        if let Err(error) = validate_tool_choice_output(&constraint_options, &call_names) {
            let _ = tx.blocking_send(Ok(format_sse_error(&error)));
            return;
        }
        let finish = typed_finish.unwrap_or(model_finish);
        if let Err(error) =
            output_format.validate_completion(&content, !call_names.is_empty(), finish)
        {
            let _ = tx.blocking_send(Ok(format_sse_error(&error)));
            return;
        }
        performance.complete();
        if tx
            .blocking_send(Ok(tool_finish_chunk(&id_for_task, &model_for_task, finish)))
            .is_err()
        {
            return;
        }
        if include_usage {
            let usage = StreamUsageChunk::new(
                id_for_task,
                model_for_task,
                prompt_tokens,
                completion_tokens,
            );
            if tx.blocking_send(Ok(format_sse_data(&usage))).is_err() {
                return;
            }
        }
        let _ = tx.blocking_send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
    });

    match init_rx.await {
        Ok(Ok(())) => super::api_transport::disconnect_aware_sse_response(rx),
        Ok(Err(error)) => generation_err_to_response(error),
        Err(error) => internal_error_response(
            "generation_initialization_channel_closed",
            format!("generation initialization channel closed: {error}"),
        ),
    }
}

async fn serve_via_scheduler_tools_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
    include_usage: bool,
    tool_context: ToolResponseContext,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let started_at = Instant::now();
    let (decoder_config, constraint_options, output_format) = tool_context.decoder_config();
    let id = gen_id();
    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match admit_request(&state, request).await {
        Ok(reply) => reply,
        Err(response) => return response,
    };
    let mut performance = state.record_request_started(prompt_tokens, started_at);
    let tokenizer = state.tokenizer.clone();
    let (tx, rx, disconnect) = super::api_transport::disconnect_aware_sse_channel(8);
    tokio::spawn(async move {
        let mut decoder = match GeneratedOutputDecoder::new(&tokenizer, Some(decoder_config)) {
            Ok(decoder) => decoder,
            Err(error) => {
                let _ = tx.send(Ok(format_sse_error(&error))).await;
                return;
            }
        };
        if tx.send(Ok(tool_role_chunk(&id, &model_id))).await.is_err() {
            return;
        }
        let mut completion_tokens = 0_u32;
        let mut next_call_index = 0_usize;
        let mut call_names = Vec::new();
        let mut model_finish = "stop";
        let mut typed_finish = None;
        let mut content = String::new();
        while let Some(event) =
            super::api_transport::recv_or_disconnect(&disconnect, &mut event_rx).await
        {
            completion_tokens += 1;
            performance.record_output_tokens(1);
            let events = if event.finish_reason == Some("stop") {
                Ok(Vec::new())
            } else {
                decoder.push_token(event.token)
            };
            let events = match events {
                Ok(events) => events,
                Err(error) => {
                    let _ = tx.send(Ok(format_sse_error(&error))).await;
                    return;
                }
            };
            for frame in match format_tool_output_events(
                &id,
                &model_id,
                events,
                &mut next_call_index,
                &mut call_names,
                &mut typed_finish,
                &mut content,
            ) {
                Ok(frames) => frames,
                Err(error) => {
                    let _ = tx.send(Ok(format_sse_error(&error))).await;
                    return;
                }
            } {
                if tx.send(Ok(frame)).await.is_err() {
                    return;
                }
            }
            if let Some(reason) = event.finish_reason {
                model_finish = reason;
                break;
            }
        }
        let events = match decoder.finish(model_finish) {
            Ok(events) => events,
            Err(error) => {
                let _ = tx.send(Ok(format_sse_error(&error))).await;
                return;
            }
        };
        for frame in match format_tool_output_events(
            &id,
            &model_id,
            events,
            &mut next_call_index,
            &mut call_names,
            &mut typed_finish,
            &mut content,
        ) {
            Ok(frames) => frames,
            Err(error) => {
                let _ = tx.send(Ok(format_sse_error(&error))).await;
                return;
            }
        } {
            if tx.send(Ok(frame)).await.is_err() {
                return;
            }
        }
        if let Err(error) = validate_tool_choice_output(&constraint_options, &call_names) {
            let _ = tx.send(Ok(format_sse_error(&error))).await;
            return;
        }
        let finish = typed_finish.unwrap_or(model_finish);
        if let Err(error) =
            output_format.validate_completion(&content, !call_names.is_empty(), finish)
        {
            let _ = tx.send(Ok(format_sse_error(&error))).await;
            return;
        }
        performance.complete();
        if tx
            .send(Ok(tool_finish_chunk(&id, &model_id, finish)))
            .await
            .is_err()
        {
            return;
        }
        if include_usage {
            let usage = StreamUsageChunk::new(
                id.clone(),
                model_id.clone(),
                prompt_tokens,
                completion_tokens,
            );
            if tx.send(Ok(format_sse_data(&usage))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
    });
    super::api_transport::disconnect_aware_sse_response(rx)
}

async fn serve_via_gs_tools<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
    include_usage: bool,
    stream_response: bool,
    tool_context: ToolResponseContext,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if stream_response {
        return serve_via_gs_tools_stream(
            state,
            request,
            model_id,
            prompt_tokens,
            include_usage,
            tool_context,
        )
        .await;
    }
    let started_at = Instant::now();
    let (decoder_config, constraint_options, output_format) = tool_context.decoder_config();
    let id = gen_id();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<ParsedAssistantOutput> {
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let memory = super::begin_direct_request_memory(&state, &*model_guard, &request)?;
        let mut generation = GenerationStream::new(&*model_guard, tokenizer, request)?;
        let mut decoder = GeneratedOutputDecoder::new(tokenizer, Some(decoder_config))?;
        let mut output = ParsedAssistantOutput {
            content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: "stop",
            completion_tokens: 0,
        };
        let mut memory = Some(memory);
        let mut performance = None;
        loop {
            let Some(event) = generation.next_token()? else {
                break;
            };
            if let Some(memory) = memory.take() {
                memory.commit();
                performance = Some(state.record_request_started(prompt_tokens, started_at));
            }
            output.completion_tokens += 1;
            performance
                .as_mut()
                .expect("performance tracker starts with the first generated token")
                .record_output_tokens(1);
            let events = if event.finish_reason == Some("stop") {
                Vec::new()
            } else {
                decoder.push_token(event.token)?
            };
            collect_generated_events(&mut output, events)?;
            if let Some(reason) = event.finish_reason {
                output.finish_reason = reason;
                break;
            }
        }
        let model_finish = output.finish_reason;
        finish_output_decoder(&mut decoder, &mut output, model_finish)?;
        let call_names = output
            .tool_calls
            .iter()
            .map(|call| call.name.clone())
            .collect::<Vec<_>>();
        validate_tool_choice_output(&constraint_options, &call_names)?;
        output_format.validate_completion(
            &output.content,
            !output.tool_calls.is_empty(),
            output.finish_reason,
        )?;
        performance
            .ok_or_else(|| anyhow::anyhow!("generation ended before producing a token"))?
            .complete();
        Ok(output)
    })
    .await;
    let output = match result {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => return generation_err_to_response(error),
        Err(error) => {
            return internal_error_response(
                "generation_task_failed",
                format!("join error: {error}"),
            );
        }
    };
    tool_completion_response(id, model_id, prompt_tokens, output)
}

async fn serve_via_scheduler_tools<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
    include_usage: bool,
    stream_response: bool,
    tool_context: ToolResponseContext,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if stream_response {
        return serve_via_scheduler_tools_stream(
            state,
            request,
            model_id,
            prompt_tokens,
            include_usage,
            tool_context,
        )
        .await;
    }
    let started_at = Instant::now();
    let (decoder_config, constraint_options, output_format) = tool_context.decoder_config();
    let id = gen_id();
    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match admit_request(&state, request).await {
        Ok(reply) => reply,
        Err(response) => return response,
    };
    let mut performance = state.record_request_started(prompt_tokens, started_at);
    let mut decoder = match GeneratedOutputDecoder::new(&state.tokenizer, Some(decoder_config)) {
        Ok(decoder) => decoder,
        Err(error) => return bad_request_response("invalid_tool_parser", format!("{error:#}")),
    };
    let mut output = ParsedAssistantOutput {
        content: String::new(),
        tool_calls: Vec::new(),
        finish_reason: "stop",
        completion_tokens: 0,
    };
    while let Some(event) = event_rx.recv().await {
        output.completion_tokens += 1;
        performance.record_output_tokens(1);
        let events = if event.finish_reason == Some("stop") {
            Ok(Vec::new())
        } else {
            decoder.push_token(event.token)
        };
        match events {
            Ok(events) => {
                if let Err(error) = collect_generated_events(&mut output, events) {
                    return generation_err_to_response(error);
                }
            }
            Err(error) => {
                return generation_err_to_response(error);
            }
        }
        if let Some(reason) = event.finish_reason {
            output.finish_reason = reason;
            break;
        }
    }
    let model_finish = output.finish_reason;
    if let Err(error) = finish_output_decoder(&mut decoder, &mut output, model_finish) {
        return generation_err_to_response(error);
    }
    let call_names = output
        .tool_calls
        .iter()
        .map(|call| call.name.clone())
        .collect::<Vec<_>>();
    if let Err(error) = validate_tool_choice_output(&constraint_options, &call_names) {
        return generation_err_to_response(error);
    }
    if let Err(error) = output_format.validate_completion(
        &output.content,
        !output.tool_calls.is_empty(),
        output.finish_reason,
    ) {
        return generation_err_to_response(error);
    }
    performance.complete();
    tool_completion_response(id, model_id, prompt_tokens, output)
}

async fn serve_via_gs_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
    include_usage: bool,
    output_format: StructuredOutputFormat,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let started_at = Instant::now();
    let (tx, rx, disconnect) = super::api_transport::disconnect_aware_sse_channel(8);
    let (init_tx, init_rx) = oneshot::channel::<anyhow::Result<()>>();
    let id = gen_id();
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

        // T0a.8 Step 2: wrap GenerationStream::new in gs_stream_init_and_chunk_loop
        // so deep spans inside (gs_kv_cache_alloc / gs_chunk_N /
        // gs_first_token_sample_dispatch) chain under it via the trace guard.

        let stream_result = GenerationStream::new(&*model_guard, tokenizer, request);

        let mut stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                let _ = init_tx.send(Err(e));
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
        let mut performance = state.record_request_started(prompt_tokens, started_at);
        if init_tx.send(Ok(())).is_err() {
            return;
        }

        // First chunk: emit role.
        let role_chunk = ChunkResponse {
            id: id_for_task.clone(),
            object: "chat.completion.chunk",
            created: now_unix(),
            model: model_id_for_task.clone(),
            choices: vec![Choice {
                index: 0,
                delta: DeltaRole {
                    role: "assistant",
                    content: String::new(),
                },
                finish_reason: None,
            }],
        };

        // T0a.8 Step 5 (role): wrap the role-chunk send in sse_write_role_chunk.

        let role_send_result = tx.blocking_send(Ok(format_sse_data(&role_chunk)));

        if role_send_result.is_err() {
            return;
        }

        let mut decoder = match GeneratedOutputDecoder::new(tokenizer, None) {
            Ok(decoder) => decoder,
            Err(error) => {
                let _ = tx.blocking_send(Ok(format_sse_error(&error)));
                return;
            }
        };
        let mut completion_tokens = 0_u32;
        let mut first_event = Some(first_event);
        let mut finish_reason = None;
        let mut next_call_index = 0_usize;
        let mut call_names = Vec::new();
        let mut content = String::new();
        loop {
            if disconnect.is_cancelled() {
                return;
            }
            let ev_result = match first_event.take() {
                Some(event) => Ok(event),
                None => stream.next_token(),
            };

            match ev_result {
                Ok(Some(ev)) => {
                    completion_tokens += 1;
                    let mut events = if ev.finish_reason == Some("stop") {
                        Vec::new()
                    } else {
                        match decoder.push_token(ev.token) {
                            Ok(events) => events,
                            Err(error) => {
                                let _ = tx.blocking_send(Ok(format_sse_error(&error)));
                                return;
                            }
                        }
                    };
                    if let Some(reason) = ev.finish_reason {
                        match decoder.finish(reason) {
                            Ok(tail) => events.extend(tail),
                            Err(error) => {
                                let _ = tx.blocking_send(Ok(format_sse_error(&error)));
                                return;
                            }
                        }
                    }
                    let frames = match format_tool_output_events(
                        &id_for_task,
                        &model_id_for_task,
                        events,
                        &mut next_call_index,
                        &mut call_names,
                        &mut finish_reason,
                        &mut content,
                    ) {
                        Ok(frames) => frames,
                        Err(error) => {
                            let _ = tx.blocking_send(Ok(format_sse_error(&error)));
                            return;
                        }
                    };
                    for frame in frames {
                        if tx.blocking_send(Ok(frame)).is_err() {
                            return;
                        }
                    }
                    performance.record_output_tokens(1);
                    if ev.finish_reason.is_some() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    let _ = tx.blocking_send(Ok(format_sse_error(&e)));
                    break;
                }
            }
        }
        if disconnect.is_cancelled() {
            return;
        }
        if let Some(reason) = finish_reason {
            if let Err(error) = output_format.validate_completion(&content, false, reason) {
                let _ = tx.blocking_send(Ok(format_sse_error(&error)));
                return;
            }
            performance.complete();
            let _ = tx.blocking_send(Ok(tool_finish_chunk(
                &id_for_task,
                &model_id_for_task,
                reason,
            )));
        }
        if include_usage {
            let usage = StreamUsageChunk::new(
                id_for_task.clone(),
                model_id_for_task.clone(),
                prompt_tokens,
                completion_tokens,
            );
            let _ = tx.blocking_send(Ok(format_sse_data(&usage)));
        }
        let _ = tx.blocking_send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
    });

    match init_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return generation_err_to_response(error),
        Err(error) => {
            return internal_error_response(
                "generation_initialization_channel_closed",
                format!("generation initialization channel closed: {error}"),
            );
        }
    }

    super::api_transport::disconnect_aware_sse_response(rx)
}

/// Text-only short-prompt SSE path via SchedulerActor (3b-2 swap-in).
async fn serve_via_scheduler_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
    include_usage: bool,
    output_format: StructuredOutputFormat,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let started_at = Instant::now();
    let id = gen_id();

    // 1. Admit request to the actor.
    let admission_result = admit_request(&state, request).await;

    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match admission_result {
        Ok(reply) => reply,
        Err(resp) => {
            return resp;
        }
    };
    let mut performance = state.record_request_started(prompt_tokens, started_at);

    // Successful admission — proceed to spawn the forwarder using `event_rx`.

    // 2. Stream events as SSE. Spawn a forwarder task that detokenizes
    // per-event and pushes formatted SSE chunks to a bounded channel.
    let (tx, rx, disconnect) = super::api_transport::disconnect_aware_sse_channel(8);
    let id_for_task = id.clone();
    let model_id_for_task = model_id.clone();
    let tokenizer = state.tokenizer.clone();

    tokio::spawn(async move {
        // First chunk: role.
        let role_chunk = ChunkResponse {
            id: id_for_task.clone(),
            object: "chat.completion.chunk",
            created: now_unix(),
            model: model_id_for_task.clone(),
            choices: vec![Choice {
                index: 0,
                delta: DeltaRole {
                    role: "assistant",
                    content: String::new(),
                },
                finish_reason: None,
            }],
        };

        let role_send_result = tx.send(Ok(format_sse_data(&role_chunk))).await;

        if role_send_result.is_err() {
            return;
        }

        let mut decoder = match GeneratedOutputDecoder::new(&tokenizer, None) {
            Ok(decoder) => decoder,
            Err(error) => {
                let _ = tx.send(Ok(format_sse_error(&error))).await;
                return;
            }
        };
        let mut completion_tokens = 0_u32;
        let mut finish_reason = None;
        let mut next_call_index = 0_usize;
        let mut call_names = Vec::new();
        let mut content = String::new();
        while let Some(ev) =
            super::api_transport::recv_or_disconnect(&disconnect, &mut event_rx).await
        {
            completion_tokens += 1;
            let mut events = if ev.finish_reason == Some("stop") {
                Vec::new()
            } else {
                match decoder.push_token(ev.token) {
                    Ok(events) => events,
                    Err(error) => {
                        let _ = tx.send(Ok(format_sse_error(&error))).await;
                        return;
                    }
                }
            };
            if let Some(reason) = ev.finish_reason {
                match decoder.finish(reason) {
                    Ok(tail) => events.extend(tail),
                    Err(error) => {
                        let _ = tx.send(Ok(format_sse_error(&error))).await;
                        return;
                    }
                }
            }
            let frames = match format_tool_output_events(
                &id_for_task,
                &model_id_for_task,
                events,
                &mut next_call_index,
                &mut call_names,
                &mut finish_reason,
                &mut content,
            ) {
                Ok(frames) => frames,
                Err(error) => {
                    let _ = tx.send(Ok(format_sse_error(&error))).await;
                    return;
                }
            };
            for frame in frames {
                if tx.send(Ok(frame)).await.is_err() {
                    return;
                }
            }
            performance.record_output_tokens(1);
            if ev.finish_reason.is_some() {
                break;
            }
        }
        if let Some(reason) = finish_reason {
            if let Err(error) = output_format.validate_completion(&content, false, reason) {
                let _ = tx.send(Ok(format_sse_error(&error))).await;
                return;
            }
            performance.complete();
            let _ = tx
                .send(Ok(tool_finish_chunk(
                    &id_for_task,
                    &model_id_for_task,
                    reason,
                )))
                .await;
        }
        if include_usage {
            let usage = StreamUsageChunk::new(
                id_for_task.clone(),
                model_id_for_task.clone(),
                prompt_tokens,
                completion_tokens,
            );
            let _ = tx.send(Ok(format_sse_data(&usage))).await;
        }
        let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
    });

    super::api_transport::disconnect_aware_sse_response(rx)
}

async fn serve_via_gs_unary<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
    output_format: StructuredOutputFormat,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let started_at = Instant::now();
    let id = gen_id();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<ParsedAssistantOutput> {
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let memory = super::begin_direct_request_memory(&state, &*model_guard, &request)?;
        let mut stream = GenerationStream::new(&*model_guard, tokenizer, request)?;
        let mut decoder = GeneratedOutputDecoder::new(tokenizer, None)?;
        let mut performance = state.record_request_started(prompt_tokens, started_at);
        let mut output = ParsedAssistantOutput {
            content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: "stop",
            completion_tokens: 0,
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
            performance.record_output_tokens(1);
            let events = if ev.finish_reason == Some("stop") {
                Vec::new()
            } else {
                decoder.push_token(ev.token)?
            };
            collect_generated_events(&mut output, events)?;
            output.completion_tokens += 1;
            if let Some(reason) = ev.finish_reason {
                collect_generated_events(&mut output, decoder.finish(reason)?)?;
                finished = true;
                break;
            }
        }
        anyhow::ensure!(finished, "generation ended before a terminal event");
        output_format.validate_completion(&output.content, false, output.finish_reason)?;
        performance.complete();
        Ok(output)
    })
    .await;

    let output = match result {
        Ok(Ok(output)) => output,
        Ok(Err(err)) => return generation_err_to_response(err),
        Err(e) => {
            return internal_error_response("generation_task_failed", format!("join error: {e}"));
        }
    };

    let resp = CompletionResponse {
        id,
        object: "chat.completion",
        created: now_unix(),
        model: model_id,
        choices: vec![CompletionChoice {
            index: 0,
            message: CompletionMessage {
                role: "assistant",
                content: Some(output.content),
                tool_calls: Vec::new(),
            },
            finish_reason: output.finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: prompt_tokens + output.completion_tokens,
        },
    };
    Json(resp).into_response()
}

/// Text-only short-prompt unary path via SchedulerActor (3b-2 swap-in).
async fn serve_via_scheduler_unary<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
    output_format: StructuredOutputFormat,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let started_at = Instant::now();
    let id = gen_id();

    // 1. Admit.
    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match admit_request(&state, request).await {
        Ok(reply) => reply,
        Err(response) => return response,
    };
    let mut performance = state.record_request_started(prompt_tokens, started_at);

    // 2. Collect all committed tokens through the protocol-neutral decoder.
    let mut decoder = match GeneratedOutputDecoder::new(&state.tokenizer, None) {
        Ok(decoder) => decoder,
        Err(error) => return generation_err_to_response(error),
    };
    let mut output = ParsedAssistantOutput {
        content: String::new(),
        tool_calls: Vec::new(),
        finish_reason: "stop",
        completion_tokens: 0,
    };
    let mut finished = false;
    while let Some(ev) = event_rx.recv().await {
        output.completion_tokens += 1;
        performance.record_output_tokens(1);
        let events = if ev.finish_reason == Some("stop") {
            Ok(Vec::new())
        } else {
            decoder.push_token(ev.token)
        };
        if let Err(error) = events.and_then(|events| collect_generated_events(&mut output, events))
        {
            return generation_err_to_response(error);
        }
        if let Some(reason) = ev.finish_reason {
            if let Err(error) = decoder
                .finish(reason)
                .and_then(|events| collect_generated_events(&mut output, events))
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
    if let Err(error) =
        output_format.validate_completion(&output.content, false, output.finish_reason)
    {
        return generation_err_to_response(error);
    }
    performance.complete();

    let resp = CompletionResponse {
        id,
        object: "chat.completion",
        created: now_unix(),
        model: model_id,
        choices: vec![CompletionChoice {
            index: 0,
            message: CompletionMessage {
                role: "assistant",
                content: Some(output.content),
                tool_calls: Vec::new(),
            },
            finish_reason: output.finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: prompt_tokens + output.completion_tokens,
        },
    };
    Json(resp).into_response()
}

fn format_sse_data<T: Serialize>(payload: &T) -> Bytes {
    let s = serde_json::to_string(payload).unwrap_or_else(|_| "{}".into());
    let mut buf = String::with_capacity(s.len() + 8);
    buf.push_str("data: ");
    buf.push_str(&s);
    buf.push_str("\n\n");
    Bytes::from(buf)
}

fn format_sse_error(e: &anyhow::Error) -> Bytes {
    let payload =
        serde_json::json!({"error": {"message": e.to_string(), "type": "internal_error"}});
    format_sse_data(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_adapter_rejects_unmapped_typed_output() {
        let mut output = ParsedAssistantOutput {
            content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: "stop",
            completion_tokens: 0,
        };
        let error = collect_generated_events(
            &mut output,
            vec![GeneratedOutputEvent::ReasoningDelta("hidden".to_owned())],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("cannot represent generated reasoning"));
        assert!(output.content.is_empty());
    }

    #[test]
    fn sse_data_format_includes_prefix_and_double_newline() {
        let payload = serde_json::json!({"a": 1});
        let bytes = format_sse_data(&payload);
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(s.starts_with("data: "), "missing prefix: {s:?}");
        assert!(s.ends_with("\n\n"), "missing terminator: {s:?}");
        assert!(s.contains("\"a\":1"), "payload not embedded: {s:?}");
    }

    #[test]
    fn chat_completions_routes_streaming_and_unary_scheduler_requests() {
        assert_eq!(
            chat_completions_route(true, true),
            ChatCompletionsRoute::SchedulerStream
        );
        assert_eq!(
            chat_completions_route(false, true),
            ChatCompletionsRoute::SchedulerUnary
        );
        assert_eq!(
            chat_completions_route(true, false),
            ChatCompletionsRoute::GenerationStreamStream
        );
        assert_eq!(
            chat_completions_route(false, false),
            ChatCompletionsRoute::GenerationStreamUnary
        );
    }

    #[test]
    fn chat_request_supports_full_length_stream_controls() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [],
            "stream": true,
            "ignore_eos": true,
            "stream_options": {"include_usage": true}
        }))
        .expect("chat request");

        assert!(req.ignore_eos);
        assert!(req.stream_options.expect("stream options").include_usage);
        assert!(stop_token_ids_for_request(&[1, 2], true).is_empty());
        assert_eq!(stop_token_ids_for_request(&[1, 2], false), vec![1, 2]);
    }

    #[test]
    fn chat_request_parses_official_structured_output_wire_shapes() {
        let json_object: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"}
        }))
        .unwrap();
        assert!(matches!(
            json_object.structured_output_format().unwrap(),
            StructuredOutputFormat::JsonObject
        ));

        let json_schema: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "Return weather"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "description": "A forecast",
                    "schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    },
                    "strict": true
                }
            }
        }))
        .unwrap();
        assert!(matches!(
            json_schema.structured_output_format().unwrap(),
            StructuredOutputFormat::JsonSchema {
                name,
                strict: Some(true),
                ..
            } if name == "weather"
        ));
    }

    #[test]
    fn chat_request_rejects_responses_wire_shape_and_invalid_strict_schema() {
        let wrong_shape = serde_json::from_value::<ChatRequest>(serde_json::json!({
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object", "properties": {}}
            }
        }));
        assert!(wrong_shape.is_err());

        let invalid: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": [],
                        "additionalProperties": false
                    },
                    "strict": true
                }
            }
        }))
        .unwrap();
        assert!(invalid
            .structured_output_format()
            .unwrap_err()
            .to_string()
            .contains("must be listed in required"));
    }

    #[test]
    fn chat_request_rejects_unknown_response_format_fields() {
        let json_object = serde_json::from_value::<ChatRequest>(serde_json::json!({
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object", "extra": true}
        }));
        assert!(json_object.is_err());

        let text = serde_json::from_value::<ChatRequest>(serde_json::json!({
            "messages": [{"role": "user", "content": "Return text"}],
            "response_format": {"type": "text", "extra": true}
        }));
        assert!(text.is_err());
    }

    #[test]
    fn qwen38_reasoning_effort_merges_into_template_kwargs() {
        let request: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "Solve it"}],
            "reasoning_effort": "medium",
            "chat_template_kwargs": {"preserve_thinking": false}
        }))
        .unwrap();
        assert_eq!(
            request.resolved_chat_template_kwargs().unwrap(),
            Some(serde_json::json!({
                "preserve_thinking": false,
                "reasoning_effort": "medium"
            }))
        );

        let unsupported = serde_json::from_value::<ChatRequest>(serde_json::json!({
            "messages": [{"role": "user", "content": "Solve it"}],
            "reasoning_effort": "high"
        }));
        assert!(unsupported.is_err());
    }

    #[test]
    fn qwen38_reasoning_effort_rejects_conflicting_template_kwarg() {
        let request: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "Solve it"}],
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "xhigh"}
        }))
        .unwrap();
        assert!(request
            .resolved_chat_template_kwargs()
            .unwrap_err()
            .to_string()
            .contains("conflicts"));
    }

    #[test]
    fn structured_final_answer_is_available_only_for_auto_or_disabled_tools() {
        let auto = tool_request(serde_json::json!({"tool_choice": "auto"}));
        let prepared = prepare_tool_request(&auto, Some(ToolDialect::Qwen35)).unwrap();
        assert!(allows_structured_final_output(prepared.as_ref()));

        let none = tool_request(serde_json::json!({"tool_choice": "none"}));
        let prepared = prepare_tool_request(&none, Some(ToolDialect::Qwen35)).unwrap();
        assert!(allows_structured_final_output(prepared.as_ref()));

        let required = tool_request(serde_json::json!({"tool_choice": "required"}));
        let prepared = prepare_tool_request(&required, Some(ToolDialect::Qwen35)).unwrap();
        assert!(!allows_structured_final_output(prepared.as_ref()));
    }

    #[test]
    fn stream_usage_chunk_reports_authoritative_token_counts() {
        let chunk = StreamUsageChunk::new("x", "m", 50, 512);
        let value = serde_json::to_value(chunk).expect("serialize usage chunk");

        assert_eq!(value["object"], "chat.completion.chunk");
        assert_eq!(value["choices"], serde_json::json!([]));
        assert_eq!(value["usage"]["prompt_tokens"], 50);
        assert_eq!(value["usage"]["completion_tokens"], 512);
        assert_eq!(value["usage"]["total_tokens"], 562);
    }

    #[test]
    fn role_chunk_serializes_with_assistant_role() {
        let chunk = ChunkResponse {
            id: "chatcmpl-x".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "qwen3.5-4b".into(),
            choices: vec![Choice {
                index: 0,
                delta: DeltaRole {
                    role: "assistant",
                    content: String::new(),
                },
                finish_reason: None,
            }],
        };
        let s = serde_json::to_string(&chunk).unwrap();
        assert!(s.contains("\"role\":\"assistant\""));
        assert!(s.contains("\"object\":\"chat.completion.chunk\""));
        assert!(
            !s.contains("finish_reason"),
            "finish_reason None should be skipped"
        );
    }

    #[test]
    fn delta_chunk_with_finish_reason_includes_reason() {
        let chunk = ChunkResponse::<DeltaEmpty> {
            id: "x".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![Choice {
                index: 0,
                delta: DeltaEmpty {},
                finish_reason: Some("stop"),
            }],
        };
        let s = serde_json::to_string(&chunk).unwrap();
        assert!(s.contains("\"finish_reason\":\"stop\""));
    }

    #[test]
    fn completion_response_has_choices_and_message() {
        let r = CompletionResponse {
            id: "x".into(),
            object: "chat.completion",
            created: 0,
            model: "m".into(),
            choices: vec![CompletionChoice {
                index: 0,
                message: CompletionMessage {
                    role: "assistant",
                    content: Some("hi".into()),
                    tool_calls: Vec::new(),
                },
                finish_reason: "stop",
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
            },
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"object\":\"chat.completion\""));
        assert!(s.contains("\"role\":\"assistant\""));
        assert!(s.contains("\"content\":\"hi\""));
        assert!(s.contains("\"finish_reason\":\"stop\""));
        assert!(s.contains("\"prompt_tokens\":5"));
        assert!(s.contains("\"completion_tokens\":1"));
        assert!(s.contains("\"total_tokens\":6"));
    }

    fn tool_request(extra: serde_json::Value) -> ChatRequest {
        let mut request = serde_json::json!({
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }]
        });
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(request).unwrap()
    }

    #[test]
    fn tool_request_validation_maps_advanced_choice_semantics() {
        let auto = tool_request(serde_json::json!({
            "tool_choice": "auto",
            "parallel_tool_calls": true
        }));
        let prepared = prepare_tool_request(&auto, Some(ToolDialect::Qwen35))
            .unwrap()
            .unwrap();
        assert_eq!(
            prepared.constraint_options,
            Some(ToolConstraintOptions::default())
        );
        assert_eq!(prepared.dialect, ToolDialect::Qwen35);

        let prepared = prepare_tool_request(&auto, Some(ToolDialect::Llama))
            .unwrap()
            .unwrap();
        assert_eq!(prepared.dialect, ToolDialect::Llama);
        assert_eq!(
            prepared.constraint_options,
            Some(ToolConstraintOptions::default())
        );

        for dialect in [ToolDialect::MiniCpmV46, ToolDialect::MiniCpm5] {
            let prepared = prepare_tool_request(&auto, Some(dialect)).unwrap().unwrap();
            assert_eq!(prepared.dialect, dialect);
            assert_eq!(
                prepared.constraint_options,
                Some(ToolConstraintOptions::default())
            );
        }

        let prepared = prepare_tool_request(&auto, Some(ToolDialect::Gemma))
            .unwrap()
            .unwrap();
        assert_eq!(prepared.dialect, ToolDialect::Gemma);
        assert_eq!(
            prepared.constraint_options,
            Some(ToolConstraintOptions::default())
        );

        let prepared = prepare_tool_request(&auto, Some(ToolDialect::Glm))
            .unwrap()
            .unwrap();
        assert_eq!(prepared.dialect, ToolDialect::Glm);
        assert_eq!(
            prepared.constraint_options,
            Some(ToolConstraintOptions::default())
        );

        let required = tool_request(serde_json::json!({"tool_choice": "required"}));
        let prepared = prepare_tool_request(&required, Some(ToolDialect::Qwen35))
            .unwrap()
            .unwrap();
        assert_eq!(
            prepared.constraint_options.unwrap().choice,
            ToolChoiceConstraint::Required
        );

        let forced = tool_request(serde_json::json!({
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }));
        let prepared = prepare_tool_request(&forced, Some(ToolDialect::Qwen35))
            .unwrap()
            .unwrap();
        assert_eq!(
            prepared.constraint_options.unwrap().choice,
            ToolChoiceConstraint::Function("get_weather".into())
        );

        let serial = tool_request(serde_json::json!({"parallel_tool_calls": false}));
        let prepared = prepare_tool_request(&serial, Some(ToolDialect::Qwen35))
            .unwrap()
            .unwrap();
        assert!(!prepared.constraint_options.unwrap().allow_parallel_calls);

        let unsupported = tool_request(serde_json::json!({}));
        assert!(prepare_tool_request(&unsupported, None)
            .unwrap_err()
            .to_string()
            .contains("does not support"));
    }

    #[test]
    fn gemma_preparation_keeps_original_tools_and_projects_dynamic_objects() {
        let request: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "run pwd"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "bash",
                    "strict": false,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "env": {
                                "type": "object",
                                "additionalProperties": {"type": "string"}
                            }
                        },
                        "required": ["command"],
                        "additionalProperties": false
                    }
                }
            }]
        }))
        .unwrap();
        let prepared = prepare_tool_request(&request, Some(ToolDialect::Gemma))
            .unwrap()
            .unwrap();
        assert_eq!(
            prepared.definitions[0].parameters["properties"]["env"]["type"],
            "object"
        );
        assert_eq!(
            prepared.model_definitions[0].parameters["properties"]["env"]["type"],
            "array"
        );
        assert_eq!(
            prepared.wire_tools[0].function.parameters["properties"]["env"]["type"],
            "array"
        );
        assert!(
            !prepared.wire_tools[0].function.parameters["properties"]["env"]["items"]["properties"]
                .as_object()
                .unwrap()
                .contains_key("additionalProperties")
        );
    }

    #[test]
    fn tool_request_rejects_invalid_forced_function_choices() {
        for (extra, expected) in [
            (
                serde_json::json!({"tool_choice": {"type": "function", "function": {"name": "missing"}}}),
                "unknown function",
            ),
            (
                serde_json::json!({"tool_choice": {"type": "function", "function": {"name": "get_weather", "extra": true}}}),
                "tool_choice object",
            ),
            (
                serde_json::json!({"tool_choice": {"type": "custom", "function": {"name": "get_weather"}}}),
                "tool_choice.type",
            ),
        ] {
            let request = tool_request(extra);
            let error = prepare_tool_request(&request, Some(ToolDialect::Qwen35)).unwrap_err();
            assert!(error.to_string().contains(expected), "{error:#}");
        }
    }

    #[test]
    fn strict_tool_request_accepts_only_official_strict_schema_shape() {
        let mut valid = tool_request(serde_json::json!({}));
        let function = &mut valid.tools.as_mut().unwrap()[0].function;
        function.strict = Some(true);
        function.parameters["additionalProperties"] = serde_json::Value::Bool(false);
        prepare_tool_request(&valid, Some(ToolDialect::Qwen35)).unwrap();

        let mut invalid = tool_request(serde_json::json!({}));
        invalid.tools.as_mut().unwrap()[0].function.strict = Some(true);
        let error = prepare_tool_request(&invalid, Some(ToolDialect::Qwen35)).unwrap_err();
        assert!(error.to_string().contains("additionalProperties=false"));
    }

    #[test]
    fn generated_calls_are_postvalidated_against_request_semantics() {
        let auto = ToolConstraintOptions::default();
        validate_tool_choice_output(&auto, &[]).unwrap();
        validate_tool_choice_output(&auto, &["a".into(), "b".into()]).unwrap();

        let required = ToolConstraintOptions {
            choice: ToolChoiceConstraint::Required,
            allow_parallel_calls: true,
        };
        assert!(validate_tool_choice_output(&required, &[]).is_err());
        validate_tool_choice_output(&required, &["a".into()]).unwrap();

        let serial = ToolConstraintOptions {
            choice: ToolChoiceConstraint::Auto,
            allow_parallel_calls: false,
        };
        assert!(validate_tool_choice_output(&serial, &["a".into(), "b".into()]).is_err());

        let forced = ToolConstraintOptions {
            choice: ToolChoiceConstraint::Function("a".into()),
            allow_parallel_calls: true,
        };
        validate_tool_choice_output(&forced, &["a".into()]).unwrap();
        assert!(validate_tool_choice_output(&forced, &[]).is_err());
        assert!(validate_tool_choice_output(&forced, &["b".into()]).is_err());
        assert!(validate_tool_choice_output(&forced, &["a".into(), "a".into()]).is_err());
    }

    #[tokio::test]
    async fn fake_http_endpoint_rejects_invalid_tool_requests_with_400() {
        use axum::http::Request;
        use axum::routing::post;
        use axum::Router;
        use tower::ServiceExt;

        async fn validate(ApiJson(req): ApiJson<ChatRequest>) -> Response {
            match prepare_tool_request(&req, Some(ToolDialect::Qwen35)) {
                Ok(_) => StatusCode::NO_CONTENT.into_response(),
                Err(error) => (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
            }
        }

        let app = Router::new().route("/v1/chat/completions", post(validate));
        for body in [
            serde_json::json!({
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "retrieval", "function": {
                    "name": "x", "parameters": {"type": "object", "properties": {}}
                }}]
            }),
            serde_json::json!({
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "function", "function": {
                    "name": "x", "parameters": []
                }}]
            }),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::post("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn chat_http_contract_rejects_unknown_and_invalid_sampling_fields_with_400() {
        use axum::http::Request;
        use axum::routing::post;
        use axum::Router;
        use tower::ServiceExt;

        async fn validate(ApiJson(_req): ApiJson<ChatRequest>) -> Response {
            StatusCode::NO_CONTENT.into_response()
        }

        let app = Router::new().route("/v1/chat/completions", post(validate));
        for body in [
            serde_json::json!({"messages": [], "top_k": 16}),
            serde_json::json!({"messages": [], "repetition_penalty": 1.1}),
            serde_json::json!({"messages": [], "max_completion_tokens": 8}),
            serde_json::json!({"messages": [{"role": "user", "content": "hi", "name": "alice"}]}),
            serde_json::json!({"messages": [], "temperature": -0.1}),
            serde_json::json!({"messages": [], "top_p": 0.0}),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::post("/v1/chat/completions")
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
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "messages": [],
                            "temperature": 2.0,
                            "top_p": 1.0
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
    }

    #[test]
    fn tool_choice_none_validates_tools_but_disables_output_parser() {
        let request = tool_request(serde_json::json!({"tool_choice": "none"}));
        let prepared = prepare_tool_request(&request, Some(ToolDialect::Qwen35))
            .unwrap()
            .unwrap();
        assert!(prepared.constraint_options.is_none());
        let kwargs = tool_template_kwargs(None, &prepared).unwrap();
        assert!(kwargs.get("tools").is_none());
    }

    #[test]
    fn tool_history_requires_matching_results_and_json_arguments() {
        let request: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [
                {"role": "user", "content": "weather"},
                {"role": "assistant", "content": "Need weather data.\n</think>\n\n", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                {"role": "user", "content": "thanks"}
            ],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }}]
        }))
        .unwrap();
        let flattened = request.messages.clone();
        let messages = build_agent_messages(&request.messages, &flattened).unwrap();
        assert_eq!(messages[1].tool_calls.len(), 1);
        assert_eq!(
            messages[1].content.as_deref(),
            Some("Need weather data.\n</think>\n\n")
        );
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));

        let mut orphan = request.messages.clone();
        orphan[2].tool_call_id = Some("call_missing".into());
        assert!(build_agent_messages(&orphan, &orphan)
            .unwrap_err()
            .to_string()
            .contains("orphan"));
    }

    #[tokio::test]
    async fn tool_sse_reconstructs_parallel_calls_by_stable_index() {
        let output = ParsedAssistantOutput {
            content: String::new(),
            tool_calls: vec![
                ToolCall {
                    id: "call_a".into(),
                    name: "first".into(),
                    arguments: serde_json::json!({"city": "东京"}),
                },
                ToolCall {
                    id: "call_b".into(),
                    name: "second".into(),
                    arguments: serde_json::json!({"days": 3}),
                },
            ],
            finish_reason: "tool_calls",
            completion_tokens: 12,
        };
        let response = tool_stream_response("chatcmpl-x".into(), "qwen".into(), 8, true, output);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("\"index\":0"));
        assert!(body.contains("\"index\":1"));
        assert!(body.contains("\"id\":\"call_a\""));
        assert!(body.contains("\"finish_reason\":\"tool_calls\""));
        assert!(body.contains("data: [DONE]\n\n"));
    }

    #[test]
    fn utf8_argument_fragmentation_preserves_exact_json() {
        let input = "{\"city\":\"东京大阪札幌\"}";
        let fragments = utf8_fragments(input, 7);
        assert_eq!(fragments.concat(), input);
        assert!(fragments.iter().all(|fragment| fragment.len() <= 7));
    }

    #[test]
    fn build_sampler_uses_model_defaults_when_request_omits_sampling_params() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [],
            "max_tokens": 8
        }))
        .expect("chat request");
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
    fn build_sampler_prefers_public_request_values_and_keeps_internal_defaults() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [],
            "max_tokens": 8,
            "temperature": 0.2,
            "top_p": 0.6
        }))
        .expect("chat request");
        let defaults = super::super::SamplingDefaults {
            temperature: Some(0.7),
            top_p: Some(0.8),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
        };

        let sampler = build_sampler(&req, defaults);

        assert_eq!(sampler.temperature, 0.2);
        assert_eq!(sampler.top_p, Some(0.6));
        assert_eq!(sampler.top_k, Some(40));
        assert_eq!(sampler.repetition_penalty, Some(1.1));
    }

    #[test]
    fn chat_request_rejects_unknown_fields_at_every_public_level() {
        for body in [
            serde_json::json!({
                "messages": [],
                "max_completion_tokens": 8
            }),
            serde_json::json!({
                "messages": [{"role": "user", "content": "hi", "name": "alice"}]
            }),
            serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": "hi", "extra": true}]
                }]
            }),
            serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aA==", "detail": "low"}
                    }]
                }]
            }),
            serde_json::json!({
                "messages": [],
                "stream_options": {"include_usage": true, "extra": true}
            }),
        ] {
            assert!(
                serde_json::from_value::<ChatRequest>(body).is_err(),
                "unknown field must be rejected"
            );
        }
    }

    #[test]
    fn chat_request_rejects_nonstandard_sampling_fields() {
        for field in ["top_k", "repetition_penalty"] {
            let mut body = serde_json::json!({"messages": []});
            body.as_object_mut()
                .unwrap()
                .insert(field.to_owned(), serde_json::json!(1));
            let error = serde_json::from_value::<ChatRequest>(body).unwrap_err();
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn chat_sampling_contract_accepts_boundaries_and_rejects_invalid_values() {
        for (temperature, top_p) in [(0.0, 0.01), (2.0, 1.0)] {
            let req: ChatRequest = serde_json::from_value(serde_json::json!({
                "messages": [],
                "temperature": temperature,
                "top_p": top_p
            }))
            .unwrap();
            req.validate_sampling().unwrap();
        }

        for body in [
            serde_json::json!({"messages": [], "temperature": -0.1}),
            serde_json::json!({"messages": [], "temperature": 2.1}),
            serde_json::json!({"messages": [], "top_p": 0.0}),
            serde_json::json!({"messages": [], "top_p": 1.1}),
        ] {
            let req: ChatRequest = serde_json::from_value(body).unwrap();
            assert!(req.validate_sampling().is_err());
        }

        let mut non_finite: ChatRequest =
            serde_json::from_value(serde_json::json!({"messages": []})).unwrap();
        non_finite.temperature = Some(f32::NAN);
        assert!(non_finite.validate_sampling().is_err());
        non_finite.temperature = None;
        non_finite.top_p = Some(f32::INFINITY);
        assert!(non_finite.validate_sampling().is_err());
    }

    #[test]
    fn data_url_decoded_to_bytes() {
        let url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
        let mut budget = ImageRequestBudget::default();
        let bytes = decode_image_url(url, &mut budget).unwrap();
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    }

    #[tokio::test]
    async fn admit_err_503_for_queue_full() {
        // 3e.3: typed SchedulerError::QueueFull → 503 via downcast.
        let err = anyhow::Error::new(crate::core::SchedulerError::QueueFull { capacity: 32 });
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let retry = resp
            .headers()
            .get("retry-after")
            .expect("retry-after header");
        assert_eq!(retry.to_str().unwrap(), "5");
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "scheduler_queue_full");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("admission queue full"));
    }

    #[tokio::test]
    async fn admit_err_400_for_untyped_anyhow() {
        // Anyhow Err WITHOUT SchedulerError::QueueFull → 400 (even if message
        // mentions "admission queue full" — string match is gone in 3e.3).
        let err = anyhow::anyhow!("admission queue full (untyped message, not the typed Err)");
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(resp.headers().get("retry-after").is_none());
    }

    #[tokio::test]
    async fn admit_err_400_for_other() {
        let err = anyhow::anyhow!("prompt too long: 999999 tokens exceeds limit");
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(resp.headers().get("retry-after").is_none());
    }

    #[tokio::test]
    async fn admit_err_413_for_request_too_large() {
        use axum::body::to_bytes;

        // 3f: typed SchedulerError::RequestTooLarge → 413 Payload Too Large.
        let err = anyhow::Error::new(crate::core::SchedulerError::RequestTooLarge {
            required_total_tokens: 273,
            input_tokens: 17,
            requested_max_output_tokens: 256,
            server_max_context_tokens: 128,
            max_allowed_output_tokens: 111,
        });
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );

        // No Retry-After header for 413 (client error, not transient).
        assert!(resp.headers().get("retry-after").is_none());

        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "request_token_capacity_exceeded");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["details"]["required_total_tokens"], 273);
        assert_eq!(body["error"]["details"]["input_tokens"], 17);
        assert_eq!(body["error"]["details"]["requested_max_output_tokens"], 256);
        assert_eq!(body["error"]["details"]["server_max_context_tokens"], 128);
        assert_eq!(body["error"]["details"]["max_allowed_output_tokens"], 111);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Dashboard → MAX CONTEXT TOKENS"));
    }

    #[test]
    fn admit_err_503_for_memory_budget_exceeded() {
        // B1-p2.5 §4.1.4: MemoryBudgetExceeded → 503 + Retry-After: 5.
        let err: anyhow::Error = crate::core::SchedulerError::MemoryBudgetExceeded {
            active_bytes: 500_000_000,
            requested_bytes: 200_000_000,
            soft_limit_bytes: 600_000_000,
        }
        .into();
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let retry = resp
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .expect("Retry-After header should be set");
        assert_eq!(retry, "5");
    }

    #[test]
    fn memory_governor_rejections_are_retryable_http_503() {
        let errors = [
            crate::core::SchedulerError::MemoryPressure {
                level: crate::core::process_memory::PressureLevel::Hard,
                current_bytes: 95,
                ceiling_bytes: 100,
            },
            crate::core::SchedulerError::PrefillPeakUnsafe {
                requested_tokens: 4096,
                selected_tokens: 0,
                target_bytes: 100,
            },
            crate::core::SchedulerError::ColdMaterializationUnsafe {
                requested_bytes: 3 * 1024 * 1024 * 1024,
                current_bytes: 20 * 1024 * 1024 * 1024,
                target_bytes: 22 * 1024 * 1024 * 1024,
            },
            crate::core::SchedulerError::StoreBackpressure {
                pending_jobs: 4,
                pending_bytes: 512 * 1024 * 1024,
            },
        ];
        for error in errors {
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
    }

    #[tokio::test]
    async fn admit_err_400_for_unrelated_typed_error() {
        // A typed Err that is NOT SchedulerError → falls through to 400.
        #[derive(Debug, thiserror::Error)]
        #[error("test error: {msg}")]
        struct OtherError {
            msg: String,
        }
        let err = anyhow::Error::new(OtherError {
            msg: "unrelated".to_string(),
        });
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(resp.headers().get("retry-after").is_none());
    }

    /// MiniCPM-V-4.6 placeholder string: <image> + <|image_pad|>×N + </image>.
    #[test]
    fn minicpmv46_placeholder_format() {
        // N=3 → [248078, 248056, 248056, 248056, 248079] when tokenised.
        assert_eq!(
            crate::models::minicpmv4_6::image_placeholder_string(3),
            "<image><|image_pad|><|image_pad|><|image_pad|></image>"
        );
    }
}
