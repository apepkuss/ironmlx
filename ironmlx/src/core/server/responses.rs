//! OpenAI Responses API adapter (`POST /v1/responses`).
//!
//! IronMLX deliberately implements the stateless local-inference surface:
//! callers send the complete typed input history and set `store: false`.
//! Response persistence, conversations, hosted tools, and background jobs are
//! not inference responsibilities and are rejected explicitly.

use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(test)]
use axum::http::header;
use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::generated_output::{
    GeneratedOutputDecoder, GeneratedOutputEvent, ToolOutputDecoderConfig,
};
use crate::core::model::Model;
use crate::core::native_output::NativeOutputDecoderConfig;
use crate::core::scheduler::DenseVlMethods;
use crate::core::server::chat_format::{
    ChatFunctionCall, ChatMessage, ChatToolCall, Content, ContentPart, ImageUrl,
};
use crate::core::server::scheduler_actor::{AdmitReply, SchedulerCommand};
use crate::core::tool_calling::{ToolCall, ToolDefinition};

use super::api_transport::ApiJson;
use super::structured_output::{coalesce_system_messages, StructuredOutputFormat};
use super::{openai, AppState, Gemma4DrafterAppState};

const DEFAULT_MAX_OUTPUT_TOKENS: usize = 256;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    pub input: ResponsesInput,
    #[serde(default)]
    pub tools: Vec<ResponseTool>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub conversation: Option<serde_json::Value>,
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub reasoning: Option<ReasoningRequest>,
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub client_metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub text: Option<serde_json::Value>,
    #[serde(default)]
    pub service_tier: Option<String>,
    #[serde(default)]
    pub truncation: Option<String>,
}

type ResponseTextFormat = StructuredOutputFormat;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(#[serde(deserialize_with = "deserialize_response_input_items")] Vec<ResponseInputItem>),
}

fn deserialize_response_input_items<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<ResponseInputItem>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut items = Vec::<serde_json::Value>::deserialize(deserializer)?;
    for item in &mut items {
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        if !object.contains_key("type")
            && object.contains_key("role")
            && object.contains_key("content")
        {
            object.insert(
                "type".to_owned(),
                serde_json::Value::String("message".to_owned()),
            );
        }
    }
    items
        .into_iter()
        .map(|item| serde_json::from_value(item).map_err(serde::de::Error::custom))
        .collect()
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputItem {
    Message {
        role: String,
        content: ResponseMessageContent,
    },
    FunctionCall {
        #[serde(default)]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        namespace: Option<String>,
    },
    FunctionCallOutput {
        call_id: String,
        output: FunctionCallOutput,
        #[serde(default)]
        status: Option<String>,
    },
    Reasoning {
        #[serde(default)]
        id: Option<String>,
        // Summary metadata is accepted in replay history but the full
        // reasoning content remains the authoritative local prompt input.
        #[allow(dead_code)]
        #[serde(default)]
        summary: Vec<ReasoningSummaryPart>,
        #[serde(default)]
        content: Vec<ReasoningContentPart>,
        #[serde(default)]
        encrypted_content: Option<String>,
        #[serde(default)]
        status: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ReasoningEffort {
    fn enables_native_reasoning(self) -> bool {
        self != Self::None
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummaryMode {
    Auto,
    None,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningRequest {
    #[serde(default)]
    pub effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub summary: Option<ReasoningSummaryMode>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningSummaryPart {
    SummaryText { text: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningContentPart {
    ReasoningText { text: String },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponseMessageContent {
    Text(String),
    Parts(Vec<ResponseContentPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentPart {
    InputText {
        text: String,
    },
    OutputText {
        text: String,
    },
    InputImage {
        #[serde(default)]
        image_url: Option<String>,
        #[serde(default)]
        file_id: Option<String>,
        #[serde(default)]
        detail: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum FunctionCallOutput {
    Text(String),
    Parts(Vec<FunctionCallOutputPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FunctionCallOutputPart {
    InputText { text: String },
    OutputText { text: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseTool {
    Function {
        name: String,
        #[serde(default)]
        description: Option<String>,
        parameters: serde_json::Value,
        #[serde(default)]
        strict: Option<bool>,
        #[serde(default)]
        defer_loading: Option<bool>,
    },
    Namespace {
        name: String,
        description: String,
        tools: Vec<ResponseNamespaceTool>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseNamespaceTool {
    Function {
        name: String,
        #[serde(default)]
        description: Option<String>,
        parameters: serde_json::Value,
        #[serde(default)]
        strict: Option<bool>,
        #[serde(default)]
        defer_loading: Option<bool>,
    },
}

#[derive(Debug, Clone)]
struct NamespaceDispatcher {
    namespace: String,
    children: HashMap<String, NamespaceArgumentEncoding>,
}

#[derive(Debug, Clone, Copy)]
enum NamespaceArgumentEncoding {
    Structured,
    JsonString,
}

#[derive(Debug, Clone, Default)]
struct ToolAliases {
    namespace_by_internal: HashMap<String, NamespaceDispatcher>,
    internal_by_namespace: HashMap<String, String>,
}

impl ToolAliases {
    fn insert_namespace(
        &mut self,
        internal: String,
        namespace: String,
        children: HashMap<String, NamespaceArgumentEncoding>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.namespace_by_internal
                .insert(
                    internal.clone(),
                    NamespaceDispatcher {
                        namespace: namespace.clone(),
                        children,
                    },
                )
                .is_none(),
            "duplicate internal namespace tool alias `{internal}`"
        );
        anyhow::ensure!(
            self.internal_by_namespace
                .insert(namespace.clone(), internal)
                .is_none(),
            "duplicate namespace `{namespace}`"
        );
        Ok(())
    }

    fn wrap_public_call(
        &self,
        namespace: &str,
        name: &str,
        arguments: &str,
    ) -> anyhow::Result<(String, String)> {
        let internal = self
            .internal_by_namespace
            .get(namespace)
            .ok_or_else(|| anyhow::anyhow!("unknown tool namespace `{namespace}`"))?;
        let dispatcher = self
            .namespace_by_internal
            .get(internal)
            .expect("namespace maps are consistent");
        let encoding = dispatcher
            .children
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("unknown namespace tool `{namespace}.{name}`"))?;
        let arguments = serde_json::from_str::<serde_json::Value>(arguments)
            .map_err(|error| anyhow::anyhow!("function_call arguments must be JSON: {error}"))?;
        let envelope = match encoding {
            NamespaceArgumentEncoding::Structured => serde_json::json!({
                "name": name,
                "arguments": arguments,
            }),
            NamespaceArgumentEncoding::JsonString => serde_json::json!({
                "name": name,
                "arguments_json": serde_json::to_string(&arguments)?,
            }),
        };
        Ok((
            internal.clone(),
            serde_json::to_string(&envelope).expect("namespace function call serializes"),
        ))
    }

    fn resolve_call(&self, call: ToolCall) -> anyhow::Result<(Option<String>, ToolCall)> {
        let Some(dispatcher) = self.namespace_by_internal.get(&call.name) else {
            return Ok((None, call));
        };
        let arguments = call
            .arguments
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("namespace dispatcher arguments must be an object"))?;
        let name = arguments
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("namespace dispatcher is missing its child name"))?;
        let encoding = dispatcher.children.get(name).ok_or_else(|| {
            anyhow::anyhow!(
                "namespace dispatcher selected unknown tool `{}.{name}`",
                dispatcher.namespace
            )
        })?;
        let child_arguments = match encoding {
            NamespaceArgumentEncoding::Structured => arguments
                .get("arguments")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("namespace dispatcher is missing arguments"))?,
            NamespaceArgumentEncoding::JsonString => {
                let encoded = arguments
                    .get("arguments_json")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        anyhow::anyhow!("namespace dispatcher is missing JSON arguments")
                    })?;
                serde_json::from_str(encoded).map_err(|error| {
                    anyhow::anyhow!("namespace dispatcher produced invalid JSON arguments: {error}")
                })?
            }
        };
        Ok((
            Some(dispatcher.namespace.clone()),
            ToolCall {
                id: call.id,
                name: name.to_owned(),
                arguments: child_arguments,
            },
        ))
    }
}

#[derive(Debug)]
pub(crate) struct NormalizedRequest {
    pub(crate) chat: openai::ChatRequest,
    pub(crate) instructions: Option<String>,
    text_format: ResponseTextFormat,
    pub(crate) response_tools: Vec<ResponseTool>,
    pub(crate) response_tool_choice: serde_json::Value,
    reasoning: ReasoningRequest,
    tool_aliases: ToolAliases,
}

impl NormalizedRequest {
    pub(crate) fn output_schema(&self) -> Option<serde_json::Value> {
        self.text_format.constraint_schema()
    }

    pub(crate) fn native_output_config(
        &self,
        tokenizer: &crate::core::Tokenizer,
    ) -> anyhow::Result<Option<NativeOutputDecoderConfig>> {
        let config =
            tokenizer.native_output_decoder_config(self.chat.chat_template_kwargs.as_ref())?;
        anyhow::ensure!(
            !self.reasoning.explicitly_enables_reasoning() || config.is_some(),
            "the loaded model does not expose a supported native reasoning channel"
        );
        Ok(config)
    }
}

impl ReasoningRequest {
    fn template_kwargs(&self) -> Option<serde_json::Value> {
        self.effort.map(|effort| {
            serde_json::json!({
                "enable_thinking": effort.enables_native_reasoning()
            })
        })
    }

    fn explicitly_enables_reasoning(&self) -> bool {
        self.effort
            .is_some_and(ReasoningEffort::enables_native_reasoning)
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn response_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

fn message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}

fn function_item_id() -> String {
    format!("fc_{}", uuid::Uuid::new_v4().simple())
}

fn parse_response_text_format(
    text: Option<&serde_json::Value>,
) -> anyhow::Result<ResponseTextFormat> {
    let Some(text) = text else {
        return Ok(ResponseTextFormat::Text);
    };
    let object = text
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("text must be a JSON object"))?;
    let Some(format) = object.get("format") else {
        return Ok(ResponseTextFormat::Text);
    };
    let format: ResponseTextFormat = serde_json::from_value(format.clone()).map_err(|error| {
        anyhow::anyhow!(
            "text.format must be `text`, `json_object`, or a valid `json_schema` object: {error}"
        )
    })?;
    format.validate_contract("text.format")?;
    Ok(format)
}

pub(crate) fn error_response(
    status: StatusCode,
    code: &'static str,
    message: impl Into<String>,
) -> Response {
    super::api_error::ApiError::from_status(status, code, message)
        .into_response(super::api_error::ApiProtocol::OpenAi)
}

impl ResponsesRequest {
    pub(crate) fn validate_topology_contract(&self) -> anyhow::Result<()> {
        validate_advisory_fields(self)
    }
}

fn validate_advisory_fields(req: &ResponsesRequest) -> anyhow::Result<()> {
    if let Some(model) = req.model.as_deref() {
        anyhow::ensure!(!model.is_empty(), "model must not be empty");
    }
    if let Some(max_output_tokens) = req.max_output_tokens {
        anyhow::ensure!(
            max_output_tokens > 0,
            "max_output_tokens must be greater than zero"
        );
    }
    if let Some(temperature) = req.temperature {
        anyhow::ensure!(
            temperature.is_finite() && (0.0..=2.0).contains(&temperature),
            "temperature must be finite and between 0 and 2"
        );
    }
    if let Some(top_p) = req.top_p {
        anyhow::ensure!(
            top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
            "top_p must be finite and in (0, 1]"
        );
    }
    anyhow::ensure!(
        req.store != Some(true),
        "store=true is not supported by the stateless local Responses API; send store=false and the complete input history"
    );
    anyhow::ensure!(
        req.previous_response_id.is_none(),
        "previous_response_id requires server-side response storage and is not supported"
    );
    anyhow::ensure!(
        req.conversation.is_none(),
        "conversation state is not supported by the local inference server"
    );
    anyhow::ensure!(
        req.background != Some(true),
        "background responses are not supported"
    );
    if let Some(tier) = req.service_tier.as_deref() {
        anyhow::ensure!(
            matches!(tier, "auto" | "default"),
            "service_tier must be `auto` or `default`"
        );
    }
    if let Some(truncation) = req.truncation.as_deref() {
        anyhow::ensure!(
            truncation == "disabled",
            "only truncation=`disabled` is supported"
        );
    }
    for include in &req.include {
        anyhow::ensure!(
            include == "reasoning.encrypted_content",
            "unsupported include value `{include}`"
        );
    }
    parse_response_text_format(req.text.as_ref())?;
    if let Some(stream_options) = &req.stream_options {
        anyhow::ensure!(
            stream_options.is_object(),
            "stream_options must be a JSON object"
        );
    }
    if let Some(key) = req.prompt_cache_key.as_deref() {
        anyhow::ensure!(
            !key.is_empty() && key.len() <= 256,
            "prompt_cache_key must contain 1 to 256 bytes"
        );
    }
    if let Some(client_metadata) = &req.client_metadata {
        anyhow::ensure!(
            client_metadata.is_object(),
            "client_metadata must be a JSON object"
        );
    }
    if let Some(metadata) = &req.metadata {
        anyhow::ensure!(metadata.is_object(), "metadata must be a JSON object");
    }
    Ok(())
}

fn response_content_to_chat(
    role: &str,
    content: ResponseMessageContent,
) -> anyhow::Result<Content> {
    match content {
        ResponseMessageContent::Text(text) => Ok(Content::Text(text)),
        ResponseMessageContent::Parts(parts) => {
            let mut output = Vec::with_capacity(parts.len());
            for part in parts {
                match part {
                    ResponseContentPart::InputText { text }
                    | ResponseContentPart::OutputText { text } => {
                        output.push(ContentPart::Text { text });
                    }
                    ResponseContentPart::InputImage {
                        image_url,
                        file_id,
                        detail,
                    } => {
                        anyhow::ensure!(
                            matches!(role, "user" | "system" | "developer"),
                            "input_image is not valid for role `{role}`"
                        );
                        anyhow::ensure!(
                            file_id.is_none(),
                            "file_id image inputs are not supported"
                        );
                        if let Some(detail) = detail.as_deref() {
                            anyhow::ensure!(
                                matches!(detail, "auto" | "low" | "high"),
                                "input_image.detail must be `auto`, `low`, or `high`"
                            );
                        }
                        let url = image_url.ok_or_else(|| {
                            anyhow::anyhow!(
                                "input_image requires image_url; file_id is unsupported"
                            )
                        })?;
                        output.push(ContentPart::ImageUrl {
                            image_url: ImageUrl { url },
                        });
                    }
                }
            }
            Ok(Content::Parts(output))
        }
    }
}

fn function_output_text(output: FunctionCallOutput) -> String {
    match output {
        FunctionCallOutput::Text(text) => text,
        FunctionCallOutput::Parts(parts) => parts
            .into_iter()
            .map(|part| match part {
                FunctionCallOutputPart::InputText { text }
                | FunctionCallOutputPart::OutputText { text } => text,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

fn append_function_call(
    messages: &mut Vec<ChatMessage>,
    call_id: String,
    name: String,
    arguments: String,
    reasoning_content: Option<String>,
) {
    let call = ChatToolCall {
        id: call_id,
        kind: "function".to_owned(),
        function: ChatFunctionCall { name, arguments },
    };
    if let Some(last) = messages.last_mut() {
        if last.role == "assistant" && last.tool_call_id.is_none() {
            if last.reasoning_content.is_none() {
                last.reasoning_content = reasoning_content;
            }
            last.tool_calls.push(call);
            return;
        }
    }
    messages.push(ChatMessage {
        role: "assistant".to_owned(),
        content: Content::Text(String::new()),
        reasoning_content,
        tool_calls: vec![call],
        tool_call_id: None,
    });
}

fn convert_input(
    instructions: Option<&str>,
    input: ResponsesInput,
    tool_aliases: &ToolAliases,
) -> anyhow::Result<Vec<ChatMessage>> {
    let mut messages = Vec::new();
    let mut pending_reasoning = None::<String>;
    if let Some(instructions) = instructions.filter(|value| !value.is_empty()) {
        messages.push(ChatMessage::text("system", instructions));
    }
    match input {
        ResponsesInput::Text(text) => messages.push(ChatMessage::text("user", text)),
        ResponsesInput::Items(items) => {
            for item in items {
                match item {
                    ResponseInputItem::Message { role, content } => {
                        anyhow::ensure!(
                            matches!(role.as_str(), "user" | "system" | "developer" | "assistant"),
                            "unsupported Responses message role `{role}`"
                        );
                        let chat_role = if role == "developer" { "system" } else { &role };
                        let content = response_content_to_chat(&role, content)?;
                        let reasoning_content = if role == "assistant" {
                            pending_reasoning.take()
                        } else {
                            anyhow::ensure!(
                                pending_reasoning.is_none(),
                                "reasoning input must be followed by an assistant message or function_call"
                            );
                            None
                        };
                        messages.push(ChatMessage {
                            role: chat_role.to_owned(),
                            content,
                            reasoning_content,
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    ResponseInputItem::FunctionCall {
                        id,
                        call_id,
                        name,
                        arguments,
                        status,
                        namespace,
                    } => {
                        if let Some(id) = id.as_deref() {
                            anyhow::ensure!(!id.is_empty(), "function_call.id must not be empty");
                        }
                        if let Some(status) = status.as_deref() {
                            anyhow::ensure!(
                                matches!(status, "in_progress" | "completed" | "incomplete"),
                                "unsupported function_call status `{status}`"
                            );
                        }
                        let (name, arguments) = match namespace.as_deref() {
                            Some(namespace) => {
                                tool_aliases.wrap_public_call(namespace, &name, &arguments)?
                            }
                            None => (name, arguments),
                        };
                        append_function_call(
                            &mut messages,
                            call_id,
                            name,
                            arguments,
                            pending_reasoning.take(),
                        );
                    }
                    ResponseInputItem::FunctionCallOutput {
                        call_id,
                        output,
                        status,
                    } => {
                        anyhow::ensure!(
                            pending_reasoning.is_none(),
                            "reasoning input must be followed by an assistant message or function_call"
                        );
                        if let Some(status) = status.as_deref() {
                            anyhow::ensure!(
                                matches!(status, "in_progress" | "completed" | "incomplete"),
                                "unsupported function_call_output status `{status}`"
                            );
                        }
                        messages.push(ChatMessage {
                            role: "tool".to_owned(),
                            content: Content::Text(function_output_text(output)),
                            reasoning_content: None,
                            tool_calls: Vec::new(),
                            tool_call_id: Some(call_id),
                        });
                    }
                    ResponseInputItem::Reasoning {
                        id,
                        summary: _,
                        content,
                        encrypted_content,
                        status,
                    } => {
                        if let Some(id) = id.as_deref() {
                            anyhow::ensure!(!id.is_empty(), "reasoning.id must not be empty");
                        }
                        if let Some(status) = status.as_deref() {
                            anyhow::ensure!(
                                matches!(status, "in_progress" | "completed" | "incomplete"),
                                "unsupported reasoning status `{status}`"
                            );
                        }
                        anyhow::ensure!(
                            pending_reasoning.is_none(),
                            "consecutive reasoning input items are not supported"
                        );
                        anyhow::ensure!(
                            encrypted_content.is_none() || !content.is_empty(),
                            "encrypted reasoning cannot be replayed without reasoning_text content"
                        );
                        let text = content
                            .into_iter()
                            .map(|part| match part {
                                ReasoningContentPart::ReasoningText { text } => text,
                            })
                            .collect::<String>();
                        pending_reasoning = (!text.is_empty()).then_some(text);
                    }
                }
            }
        }
    }
    anyhow::ensure!(
        pending_reasoning.is_none(),
        "reasoning input must be followed by an assistant message or function_call"
    );
    coalesce_system_messages(&mut messages);
    anyhow::ensure!(!messages.is_empty(), "input must not be empty");
    Ok(messages)
}

fn namespace_alias(namespace_index: usize, name: &str) -> String {
    let prefix = format!("ns{namespace_index}_");
    let available = 64_usize.saturating_sub(prefix.len());
    let suffix = &name[..name.len().min(available)];
    format!("{prefix}{suffix}")
}

fn flatten_response_tools(
    tools: &[ResponseTool],
) -> anyhow::Result<(Vec<openai::OpenAiTool>, ToolAliases)> {
    let mut flattened = Vec::new();
    let mut aliases = ToolAliases::default();
    for (namespace_index, tool) in tools.iter().enumerate() {
        match tool {
            ResponseTool::Function {
                name,
                description,
                parameters,
                strict,
                defer_loading: _,
            } => flattened.push(openai::OpenAiTool {
                kind: "function".to_owned(),
                function: ToolDefinition {
                    name: name.clone(),
                    description: description.clone(),
                    parameters: parameters.clone(),
                    strict: *strict,
                },
            }),
            ResponseTool::Namespace {
                name: namespace,
                description: namespace_description,
                tools,
            } => {
                crate::core::tool_calling::validate_function_name(namespace)?;
                anyhow::ensure!(!tools.is_empty(), "namespace `{namespace}` has no tools");
                let alias = namespace_alias(namespace_index, namespace);
                let mut children = HashMap::new();
                let mut branches = Vec::with_capacity(tools.len());
                for tool in tools {
                    match tool {
                        ResponseNamespaceTool::Function {
                            name,
                            description,
                            parameters,
                            strict,
                            defer_loading: _,
                        } => {
                            crate::core::tool_calling::validate_function_name(name)?;
                            let definition = ToolDefinition {
                                name: name.clone(),
                                description: description.clone(),
                                parameters: parameters.clone(),
                                strict: *strict,
                            };
                            let mut encoding = match crate::core::constrained::validate_tool_schemas(
                                std::slice::from_ref(&definition),
                            ) {
                                Ok(()) => {
                                    if *strict == Some(true) {
                                        crate::core::constrained::validate_strict_tool_schema(
                                            &definition,
                                        )?;
                                    }
                                    NamespaceArgumentEncoding::Structured
                                }
                                Err(_) if *strict != Some(true) => {
                                    NamespaceArgumentEncoding::JsonString
                                }
                                Err(error) => {
                                    return Err(anyhow::anyhow!(
                                        "unsupported strict schema for namespace tool `{namespace}.{name}`: {error:#}"
                                    ));
                                }
                            };
                            let structured_branch = serde_json::json!({
                                "type":"object",
                                "description": description,
                                "properties": {
                                    "name": {"type":"string", "const":name},
                                    "arguments": parameters,
                                },
                                "required":["name", "arguments"],
                                "additionalProperties":false,
                            });
                            if matches!(encoding, NamespaceArgumentEncoding::Structured) {
                                let wrapped = ToolDefinition {
                                    name: alias.clone(),
                                    description: None,
                                    parameters: serde_json::json!({
                                        "type":"object",
                                        "properties":{},
                                        "anyOf":[structured_branch.clone()],
                                    }),
                                    strict: Some(false),
                                };
                                if let Err(error) = crate::core::constrained::validate_tool_schemas(
                                    std::slice::from_ref(&wrapped),
                                ) {
                                    if *strict == Some(true) {
                                        return Err(anyhow::anyhow!(
                                            "namespace wrapper cannot preserve strict schema for `{namespace}.{name}`: {error:#}"
                                        ));
                                    }
                                    encoding = NamespaceArgumentEncoding::JsonString;
                                }
                            }
                            anyhow::ensure!(
                                children.insert(name.clone(), encoding).is_none(),
                                "duplicate namespace tool `{namespace}.{name}`"
                            );
                            match encoding {
                                NamespaceArgumentEncoding::Structured => {
                                    branches.push(structured_branch);
                                }
                                NamespaceArgumentEncoding::JsonString => {
                                    let schema = serde_json::to_string(parameters)?;
                                    branches.push(serde_json::json!({
                                        "type":"object",
                                        "description": description,
                                        "properties": {
                                            "name": {"type":"string", "const":name},
                                            "arguments_json": {
                                                "type":"string",
                                                "description": format!(
                                                    "JSON-encoded arguments for `{namespace}.{name}` matching this dynamic schema: {schema}"
                                                ),
                                            },
                                        },
                                        "required":["name", "arguments_json"],
                                        "additionalProperties":false,
                                    }));
                                }
                            }
                        }
                    }
                }
                let mut dispatcher_description = format!(
                    "Dispatch one function in namespace `{namespace}`. {namespace_description}"
                );
                let mut dispatcher_parameters = serde_json::json!({
                    "type":"object",
                    "properties":{},
                    "anyOf":branches,
                });
                let aggregate = ToolDefinition {
                    name: alias.clone(),
                    description: None,
                    parameters: dispatcher_parameters.clone(),
                    strict: Some(false),
                };
                if let Err(error) = crate::core::constrained::validate_tool_schemas(
                    std::slice::from_ref(&aggregate),
                ) {
                    let mut names = Vec::with_capacity(tools.len());
                    let mut catalog = Vec::with_capacity(tools.len());
                    for tool in tools {
                        match tool {
                            ResponseNamespaceTool::Function {
                                name,
                                description,
                                parameters,
                                strict,
                                defer_loading: _,
                            } => {
                                anyhow::ensure!(
                                    *strict != Some(true),
                                    "namespace `{namespace}` is too complex for one dispatcher and cannot collapse strict tool `{name}`: {error:#}"
                                );
                                children
                                    .insert(name.clone(), NamespaceArgumentEncoding::JsonString);
                                names.push(name.clone());
                                catalog.push(format!(
                                    "`{name}`: {}; JSON schema: {}",
                                    description.as_deref().unwrap_or("No description."),
                                    serde_json::to_string(parameters)?
                                ));
                            }
                        }
                    }
                    dispatcher_description.push_str(
                        " Select `name` from the enum and put the selected function's JSON object arguments in `arguments_json`. Available functions: ",
                    );
                    dispatcher_description.push_str(&catalog.join("\n"));
                    dispatcher_parameters = serde_json::json!({
                        "type":"object",
                        "properties":{
                            "name":{"type":"string","enum":names},
                            "arguments_json":{
                                "type":"string",
                                "description":"JSON-encoded object arguments for the selected namespace function."
                            }
                        },
                        "required":["name","arguments_json"],
                        "additionalProperties":false,
                    });
                    let collapsed = ToolDefinition {
                        name: alias.clone(),
                        description: None,
                        parameters: dispatcher_parameters.clone(),
                        strict: Some(false),
                    };
                    crate::core::constrained::validate_tool_schemas(std::slice::from_ref(
                        &collapsed,
                    ))?;
                }
                aliases.insert_namespace(alias.clone(), namespace.clone(), children)?;
                flattened.push(openai::OpenAiTool {
                    kind: "function".to_owned(),
                    function: ToolDefinition {
                        name: alias,
                        description: Some(dispatcher_description),
                        parameters: dispatcher_parameters,
                        strict: Some(false),
                    },
                });
            }
        }
    }
    Ok((flattened, aliases))
}

fn normalize_tool_choice(choice: Option<serde_json::Value>) -> anyhow::Result<serde_json::Value> {
    let Some(choice) = choice else {
        return Ok(serde_json::Value::String("auto".to_owned()));
    };
    if choice.is_string() {
        return Ok(choice);
    }
    let object = choice
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("tool_choice must be a string or function object"))?;
    anyhow::ensure!(
        object.get("type").and_then(serde_json::Value::as_str) == Some("function"),
        "tool_choice.type must be `function`"
    );
    let name = object
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("function tool_choice requires name"))?;
    anyhow::ensure!(
        object.get("namespace").is_none(),
        "named tool_choice for namespace children is not supported; use tool_choice=`auto` or `required`"
    );
    anyhow::ensure!(
        object
            .keys()
            .all(|key| matches!(key.as_str(), "type" | "name" | "namespace")),
        "function tool_choice contains unsupported fields"
    );
    Ok(serde_json::json!({
        "type": "function",
        "function": {"name": name}
    }))
}

impl ResponsesRequest {
    pub(crate) fn normalize(self) -> anyhow::Result<NormalizedRequest> {
        validate_advisory_fields(&self)?;
        let reasoning = self.reasoning.clone().unwrap_or_default();
        let text_format = parse_response_text_format(self.text.as_ref())?;
        let response_tools = self.tools.clone();
        let (tools, tool_aliases) = flatten_response_tools(&self.tools)?;
        let response_tool_choice = self
            .tool_choice
            .clone()
            .unwrap_or_else(|| serde_json::Value::String("auto".to_owned()));
        let chat_tool_choice = normalize_tool_choice(self.tool_choice)?;
        let allows_final_output = matches!(chat_tool_choice.as_str(), Some("auto" | "none"));
        let mut messages = convert_input(self.instructions.as_deref(), self.input, &tool_aliases)?;
        if allows_final_output {
            text_format.apply_prompt_instruction(&mut messages);
        }
        let tool_choice = if tools.is_empty() && chat_tool_choice.as_str() == Some("auto") {
            None
        } else {
            Some(chat_tool_choice)
        };
        Ok(NormalizedRequest {
            instructions: self.instructions,
            text_format,
            response_tools,
            response_tool_choice,
            reasoning: reasoning.clone(),
            tool_aliases,
            chat: openai::ChatRequest {
                model: self.model,
                messages,
                tools: (!tools.is_empty()).then_some(tools),
                tool_choice,
                parallel_tool_calls: self.parallel_tool_calls,
                function_call: None,
                functions: None,
                response_format: None,
                stream: self.stream,
                stream_options: None,
                ignore_eos: false,
                max_tokens: self.max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
                temperature: self.temperature,
                top_p: self.top_p,
                seed: None,
                chat_template_kwargs: reasoning.template_kwargs(),
            },
        })
    }
}

#[derive(Debug)]
struct ToolContext {
    dialect: crate::core::tool_calling::ToolDialect,
    definitions: Vec<ToolDefinition>,
    constraint_options: crate::core::constrained::ToolConstraintOptions,
    output_schema: Option<serde_json::Value>,
}

impl ToolContext {
    fn decoder_config(&self) -> ToolOutputDecoderConfig {
        ToolOutputDecoderConfig {
            dialect: self.dialect,
            response_id: uuid::Uuid::new_v4().simple().to_string(),
            definitions: self.definitions.clone(),
            output_schema: self.output_schema.clone(),
        }
    }
}

#[derive(Debug)]
struct PreparedResponse {
    request: GenerateRequest,
    model: String,
    prompt_tokens: u32,
    use_scheduler: bool,
    stream: bool,
    max_output_tokens: usize,
    instructions: Option<String>,
    text_format: ResponseTextFormat,
    tools: Vec<ResponseTool>,
    tool_choice: serde_json::Value,
    parallel_tool_calls: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    tool_context: Option<ToolContext>,
    native_output: Option<NativeOutputDecoderConfig>,
    reasoning: ReasoningRequest,
    tool_aliases: ToolAliases,
}

async fn prepare_response<M>(
    state: &AppState<M>,
    normalized: NormalizedRequest,
    force_scheduler: bool,
) -> std::result::Result<PreparedResponse, Response>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let native_output = match normalized.native_output_config(&state.tokenizer) {
        Ok(config) => config,
        Err(error) => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "unsupported_reasoning",
                format!("resolve native reasoning mode: {error:#}"),
            ));
        }
    };
    let NormalizedRequest {
        mut chat,
        instructions,
        text_format,
        response_tools,
        response_tool_choice,
        reasoning,
        tool_aliases,
    } = normalized;
    let stream = chat.stream;
    let max_output_tokens = chat.max_tokens;
    let model = chat.model.clone().unwrap_or_else(|| state.model_id.clone());
    let temperature = chat.temperature;
    let top_p = chat.top_p;
    let parallel_tool_calls = chat.parallel_tool_calls.unwrap_or(true);
    let sampler = openai::build_sampler(&chat, state.sampling_defaults);
    if let Err(error) = super::validate_prompt_lookup_sampler(state.prompt_lookup_enabled, sampler)
    {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_sampling_parameters",
            format!("{error:#}"),
        ));
    }
    let prepared_tools = match openai::prepare_tool_request(&chat, state.tokenizer.tool_dialect()) {
        Ok(prepared) => prepared,
        Err(error) => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_tools",
                format!("{error:#}"),
            ));
        }
    };
    let original_messages = prepared_tools.as_ref().map(|_| chat.messages.clone());
    let chat_template_kwargs = chat.chat_template_kwargs.take();
    let (image_token_id, spatial_merge_size) =
        super::vision::derive_image_token_and_merge(&state.vision_input, &state.tokenizer);
    let (flat_messages, pixel_values, image_grid_thw) =
        match openai::expand_image_parts_in_messages(chat.messages, &state.vision_input).await {
            Ok(result) => result,
            Err(error) => {
                return Err(super::security::image_error_response(
                    error,
                    super::api_error::ApiProtocol::OpenAi,
                ))
            }
        };
    let prompt_ids = match if let Some(prepared) = &prepared_tools {
        openai::build_agent_messages(
            original_messages
                .as_deref()
                .expect("captured for Responses tool request"),
            &flat_messages,
        )
        .and_then(|messages| {
            let kwargs = openai::tool_template_kwargs(chat_template_kwargs, prepared)?;
            openai::render_tool_prompt(&state.tokenizer, &messages, &kwargs, prepared)
        })
    } else {
        super::chat_format::render_and_encode(
            &state.tokenizer,
            &flat_messages,
            chat_template_kwargs.as_ref(),
        )
    } {
        Ok(ids) => ids,
        Err(error) => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_prompt",
                format!("chat template / tokenize: {error}"),
            ));
        }
    };
    let prompt_len = prompt_ids.len();
    let scheduler_config = state.scheduler_request_config(prompt_len, max_output_tokens);
    let use_scheduler = force_scheduler
        || super::should_route_to_scheduler::<M>(
            prompt_len,
            scheduler_config.prefill_chunk_size,
            state.b_max,
            state.paged_prefix_cache_enabled,
            state.force_scheduler_for_greedy && sampler.is_pipelinable(),
        );
    let output_schema = text_format.constraint_schema();
    let constraint = match openai::compile_output_constraint(
        &state.tokenizer,
        prepared_tools.as_ref(),
        output_schema.as_ref(),
    ) {
        Ok(value) => value,
        Err(error) => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_output_schema",
                format!("compile response decoding constraint: {error:#}"),
            ));
        }
    };
    let tool_context = prepared_tools.and_then(|prepared| {
        prepared
            .constraint_options
            .map(|constraint_options| ToolContext {
                dialect: prepared.dialect,
                definitions: prepared.definitions,
                output_schema: matches!(
                    &constraint_options.choice,
                    crate::core::constrained::ToolChoiceConstraint::Auto
                )
                .then(|| output_schema.clone())
                .flatten(),
                constraint_options,
            })
    });
    Ok(PreparedResponse {
        request: GenerateRequest {
            prompt_ids,
            max_new_tokens: max_output_tokens,
            sampler,
            stop_token_ids: openai::stop_token_ids_for_request(
                state.tokenizer.eos_token_ids(),
                false,
            ),
            prefill_chunk_size: scheduler_config.prefill_chunk_size,
            decode_cadence_mid_chunk_cap: scheduler_config.decode_cadence_mid_chunk_cap,
            kv_cache_turboquant_bits: state.kv_cache_turboquant_bits,
            pixel_values,
            image_grid_thw: (!image_grid_thw.is_empty()).then_some(image_grid_thw),
            image_spatial_merge_size: spatial_merge_size,
            image_token_id,
            constraint,
        },
        model,
        prompt_tokens: prompt_len as u32,
        use_scheduler,
        stream,
        max_output_tokens,
        instructions,
        text_format,
        tools: response_tools,
        tool_choice: response_tool_choice,
        parallel_tool_calls,
        temperature,
        top_p,
        tool_context,
        native_output,
        reasoning,
        tool_aliases,
    })
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct Usage {
    input_tokens: u32,
    input_tokens_details: InputTokenDetails,
    output_tokens: u32,
    output_tokens_details: OutputTokenDetails,
    total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
struct InputTokenDetails {
    cached_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
struct OutputTokenDetails {
    reasoning_tokens: u32,
}

impl Usage {
    pub(crate) fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            input_tokens_details: InputTokenDetails { cached_tokens: 0 },
            output_tokens,
            output_tokens_details: OutputTokenDetails {
                reasoning_tokens: 0,
            },
            total_tokens: input_tokens + output_tokens,
        }
    }

    pub(crate) fn with_reasoning_tokens(mut self, reasoning_tokens: u32) -> Self {
        self.output_tokens_details.reasoning_tokens = reasoning_tokens;
        self
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputItem {
    Reasoning {
        id: String,
        summary: Vec<ReasoningSummaryOutput>,
        content: Vec<ReasoningContentOutput>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
    Message {
        id: String,
        status: &'static str,
        role: &'static str,
        content: Vec<OutputContent>,
    },
    FunctionCall {
        id: String,
        status: &'static str,
        arguments: String,
        call_id: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputContent {
    OutputText {
        annotations: Vec<serde_json::Value>,
        logprobs: Vec<serde_json::Value>,
        text: String,
    },
    ReasoningText {
        text: String,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReasoningSummaryOutput {
    SummaryText { text: String },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReasoningContentOutput {
    ReasoningText { text: String },
}

fn reasoning_item(id: String, reasoning: String, summary: String) -> OutputItem {
    OutputItem::Reasoning {
        id,
        summary: (!summary.is_empty())
            .then_some(ReasoningSummaryOutput::SummaryText { text: summary })
            .into_iter()
            .collect(),
        content: (!reasoning.is_empty())
            .then_some(ReasoningContentOutput::ReasoningText { text: reasoning })
            .into_iter()
            .collect(),
        encrypted_content: None,
    }
}

fn message_item(id: String, text: String) -> OutputItem {
    OutputItem::Message {
        id,
        status: "completed",
        role: "assistant",
        content: vec![OutputContent::OutputText {
            annotations: Vec::new(),
            logprobs: Vec::new(),
            text,
        }],
    }
}

fn function_item(
    id: String,
    call: ToolCall,
    tool_aliases: &ToolAliases,
) -> anyhow::Result<OutputItem> {
    let (namespace, call) = tool_aliases.resolve_call(call)?;
    Ok(OutputItem::FunctionCall {
        id,
        status: "completed",
        arguments: serde_json::to_string(&call.arguments).expect("tool arguments are JSON values"),
        call_id: call.id,
        name: call.name,
        namespace,
    })
}

#[derive(Debug, Clone, Serialize)]
struct ResponseObject {
    id: String,
    object: &'static str,
    created_at: u64,
    status: &'static str,
    background: bool,
    error: Option<serde_json::Value>,
    incomplete_details: Option<IncompleteDetails>,
    instructions: Option<String>,
    max_output_tokens: usize,
    model: String,
    output: Vec<OutputItem>,
    parallel_tool_calls: bool,
    previous_response_id: Option<String>,
    reasoning: ReasoningInfo,
    service_tier: &'static str,
    store: bool,
    temperature: Option<f32>,
    text: TextConfig,
    tool_choice: serde_json::Value,
    tools: Vec<ResponseTool>,
    top_p: Option<f32>,
    truncation: &'static str,
    usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
struct IncompleteDetails {
    reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct ReasoningInfo {
    effort: Option<ReasoningEffort>,
    summary: Option<ReasoningSummaryMode>,
}

#[derive(Debug, Clone, Serialize)]
struct TextConfig {
    format: ResponseTextFormat,
}

#[derive(Debug)]
pub(crate) struct ResponseMeta {
    id: String,
    created_at: u64,
    instructions: Option<String>,
    text_format: ResponseTextFormat,
    max_output_tokens: usize,
    model: String,
    parallel_tool_calls: bool,
    temperature: Option<f32>,
    tool_choice: serde_json::Value,
    tools: Vec<ResponseTool>,
    top_p: Option<f32>,
    reasoning: ReasoningRequest,
    tool_aliases: ToolAliases,
}

impl PreparedResponse {
    fn meta(&self) -> ResponseMeta {
        ResponseMeta {
            id: response_id(),
            created_at: now_unix(),
            instructions: self.instructions.clone(),
            text_format: self.text_format.clone(),
            max_output_tokens: self.max_output_tokens,
            model: self.model.clone(),
            parallel_tool_calls: self.parallel_tool_calls,
            temperature: self.temperature,
            tool_choice: self.tool_choice.clone(),
            tools: self.tools.clone(),
            top_p: self.top_p,
            reasoning: self.reasoning.clone(),
            tool_aliases: self.tool_aliases.clone(),
        }
    }
}

impl ResponseMeta {
    pub(crate) fn from_normalized(normalized: &NormalizedRequest, model: String) -> Self {
        Self {
            id: response_id(),
            created_at: now_unix(),
            instructions: normalized.instructions.clone(),
            text_format: normalized.text_format.clone(),
            max_output_tokens: normalized.chat.max_tokens,
            model,
            parallel_tool_calls: normalized.chat.parallel_tool_calls.unwrap_or(true),
            temperature: normalized.chat.temperature,
            tool_choice: normalized.response_tool_choice.clone(),
            tools: normalized.response_tools.clone(),
            top_p: normalized.chat.top_p,
            reasoning: normalized.reasoning.clone(),
            tool_aliases: normalized.tool_aliases.clone(),
        }
    }

    fn object(
        &self,
        status: &'static str,
        output: Vec<OutputItem>,
        usage: Option<Usage>,
    ) -> ResponseObject {
        ResponseObject {
            id: self.id.clone(),
            object: "response",
            created_at: self.created_at,
            status,
            background: false,
            error: None,
            incomplete_details: (status == "incomplete").then_some(IncompleteDetails {
                reason: "max_output_tokens",
            }),
            instructions: self.instructions.clone(),
            max_output_tokens: self.max_output_tokens,
            model: self.model.clone(),
            output,
            parallel_tool_calls: self.parallel_tool_calls,
            previous_response_id: None,
            reasoning: ReasoningInfo {
                effort: self.reasoning.effort,
                summary: self.reasoning.summary,
            },
            service_tier: "default",
            store: false,
            temperature: self.temperature,
            text: TextConfig {
                format: self.text_format.clone(),
            },
            tool_choice: self.tool_choice.clone(),
            tools: self.tools.clone(),
            top_p: self.top_p,
            truncation: "disabled",
            usage,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CollectedOutput {
    pub(crate) content: String,
    pub(crate) reasoning: String,
    pub(crate) reasoning_summary: String,
    pub(crate) tool_calls: Vec<ToolCall>,
    pub(crate) finish_reason: &'static str,
    pub(crate) completion_tokens: u32,
    pub(crate) reasoning_tokens: u32,
}

impl CollectedOutput {
    pub(crate) fn new() -> Self {
        Self {
            content: String::new(),
            reasoning: String::new(),
            reasoning_summary: String::new(),
            tool_calls: Vec::new(),
            finish_reason: "stop",
            completion_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    fn collect(&mut self, events: Vec<GeneratedOutputEvent>) -> anyhow::Result<()> {
        for event in events {
            match event {
                GeneratedOutputEvent::TextDelta(text) => self.content.push_str(&text),
                GeneratedOutputEvent::ReasoningDelta(text) => self.reasoning.push_str(&text),
                GeneratedOutputEvent::ReasoningSummaryDelta(text) => {
                    self.reasoning_summary.push_str(&text);
                }
                GeneratedOutputEvent::ToolCall(call) => self.tool_calls.push(call),
                GeneratedOutputEvent::Finished(reason) => self.finish_reason = reason.as_str(),
                other => anyhow::bail!(
                    "Responses adapter has no enabled producer mapping for generated {} output",
                    other.kind()
                ),
            }
        }
        Ok(())
    }
}

fn validate_collected_output(
    output: &CollectedOutput,
    context: &ToolContext,
) -> anyhow::Result<()> {
    let call_names = output
        .tool_calls
        .iter()
        .map(|call| call.name.clone())
        .collect::<Vec<_>>();
    openai::validate_tool_choice_output(&context.constraint_options, &call_names)
}

pub(crate) fn unary_response(
    meta: ResponseMeta,
    input_tokens: u32,
    output: CollectedOutput,
) -> Response {
    if let Err(error) = meta.text_format.validate_completion(
        &output.content,
        !output.tool_calls.is_empty(),
        output.finish_reason,
    ) {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "structured_output_error",
            format!("{error:#}"),
        );
    }
    let mut items = Vec::new();
    if !output.reasoning.is_empty() || !output.reasoning_summary.is_empty() {
        items.push(reasoning_item(
            format!("rs_{}", uuid::Uuid::new_v4().simple()),
            output.reasoning,
            output.reasoning_summary,
        ));
    }
    if !output.content.is_empty() {
        items.push(message_item(message_id(), output.content));
    }
    for call in output.tool_calls {
        match function_item(function_item_id(), call, &meta.tool_aliases) {
            Ok(item) => items.push(item),
            Err(error) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "tool_parse_error",
                    format!("{error:#}"),
                );
            }
        }
    }
    let status = if output.finish_reason == "length" {
        "incomplete"
    } else {
        "completed"
    };
    Json(
        meta.object(
            status,
            items,
            Some(
                Usage::new(input_tokens, output.completion_tokens)
                    .with_reasoning_tokens(output.reasoning_tokens),
            ),
        ),
    )
    .into_response()
}

#[derive(Debug, Serialize)]
struct StreamEvent<T> {
    #[serde(rename = "type")]
    kind: &'static str,
    sequence_number: u64,
    #[serde(flatten)]
    payload: T,
}

#[derive(Debug, Serialize)]
struct ResponsePayload {
    response: ResponseObject,
}

#[derive(Debug, Serialize)]
struct ItemPayload {
    item: OutputItem,
    output_index: usize,
}

#[derive(Debug, Serialize)]
struct ContentPartPayload {
    content_index: usize,
    item_id: String,
    output_index: usize,
    part: OutputContent,
}

#[derive(Debug, Serialize)]
struct TextDeltaPayload {
    content_index: usize,
    delta: String,
    item_id: String,
    logprobs: Vec<serde_json::Value>,
    output_index: usize,
}

#[derive(Debug, Serialize)]
struct TextDonePayload {
    content_index: usize,
    item_id: String,
    logprobs: Vec<serde_json::Value>,
    output_index: usize,
    text: String,
}

#[derive(Debug, Serialize)]
struct ReasoningDeltaPayload {
    content_index: usize,
    delta: String,
    item_id: String,
    output_index: usize,
}

#[derive(Debug, Serialize)]
struct ReasoningDonePayload {
    content_index: usize,
    item_id: String,
    output_index: usize,
    text: String,
}

#[derive(Debug, Serialize)]
struct ArgumentsDeltaPayload {
    delta: String,
    item_id: String,
    output_index: usize,
}

#[derive(Debug, Serialize)]
struct ArgumentsDonePayload {
    arguments: String,
    item_id: String,
    name: String,
    output_index: usize,
}

#[derive(Debug, Serialize)]
struct FailedPayload {
    response: FailedResponse,
}

#[derive(Debug, Serialize)]
struct FailedResponse {
    id: String,
    object: &'static str,
    created_at: u64,
    status: &'static str,
    error: FailedError,
}

#[derive(Debug, Serialize)]
struct FailedError {
    code: &'static str,
    message: String,
}

fn sse_event<T: Serialize>(kind: &'static str, sequence_number: u64, payload: T) -> Bytes {
    let event = StreamEvent {
        kind,
        sequence_number,
        payload,
    };
    let json = serde_json::to_string(&event).expect("Responses SSE event serializes");
    Bytes::from(format!("event: {kind}\ndata: {json}\n\n"))
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

pub(crate) struct ResponsesStream {
    meta: ResponseMeta,
    sequence: u64,
    output: Vec<OutputItem>,
    active_reasoning: Option<(usize, String, String)>,
    active_text: Option<(usize, String, String)>,
}

impl ResponsesStream {
    pub(crate) fn new(meta: ResponseMeta) -> Self {
        Self {
            meta,
            sequence: 0,
            output: Vec::new(),
            active_reasoning: None,
            active_text: None,
        }
    }

    fn event<T: Serialize>(&mut self, kind: &'static str, payload: T) -> Bytes {
        let frame = sse_event(kind, self.sequence, payload);
        self.sequence += 1;
        frame
    }

    pub(crate) fn created(&mut self) -> Bytes {
        self.event(
            "response.created",
            ResponsePayload {
                response: self.meta.object("in_progress", Vec::new(), None),
            },
        )
    }

    pub(crate) fn text_delta(&mut self, delta: String) -> Vec<Bytes> {
        let mut frames = self.finish_reasoning();
        if self.active_text.is_none() {
            let output_index = self.output.len();
            let id = message_id();
            frames.push(self.event(
                "response.output_item.added",
                ItemPayload {
                    item: OutputItem::Message {
                        id: id.clone(),
                        status: "in_progress",
                        role: "assistant",
                        content: Vec::new(),
                    },
                    output_index,
                },
            ));
            frames.push(self.event(
                "response.content_part.added",
                ContentPartPayload {
                    content_index: 0,
                    item_id: id.clone(),
                    output_index,
                    part: OutputContent::OutputText {
                        annotations: Vec::new(),
                        logprobs: Vec::new(),
                        text: String::new(),
                    },
                },
            ));
            self.active_text = Some((output_index, id, String::new()));
        }
        let (output_index, id) = {
            let (index, id, text) = self.active_text.as_mut().expect("created text item");
            text.push_str(&delta);
            (*index, id.clone())
        };
        frames.push(self.event(
            "response.output_text.delta",
            TextDeltaPayload {
                content_index: 0,
                delta,
                item_id: id,
                logprobs: Vec::new(),
                output_index,
            },
        ));
        frames
    }

    pub(crate) fn reasoning_delta(&mut self, delta: String) -> Vec<Bytes> {
        let mut frames = self.finish_text();
        if self.active_reasoning.is_none() {
            let output_index = self.output.len();
            let id = format!("rs_{}", uuid::Uuid::new_v4().simple());
            frames.push(self.event(
                "response.output_item.added",
                ItemPayload {
                    item: OutputItem::Reasoning {
                        id: id.clone(),
                        summary: Vec::new(),
                        content: Vec::new(),
                        encrypted_content: None,
                    },
                    output_index,
                },
            ));
            frames.push(self.event(
                "response.content_part.added",
                ContentPartPayload {
                    content_index: 0,
                    item_id: id.clone(),
                    output_index,
                    part: OutputContent::ReasoningText {
                        text: String::new(),
                    },
                },
            ));
            self.active_reasoning = Some((output_index, id, String::new()));
        }
        let (output_index, id) = {
            let (index, id, text) = self
                .active_reasoning
                .as_mut()
                .expect("created reasoning item");
            text.push_str(&delta);
            (*index, id.clone())
        };
        frames.push(self.event(
            "response.reasoning_text.delta",
            ReasoningDeltaPayload {
                content_index: 0,
                delta,
                item_id: id,
                output_index,
            },
        ));
        frames
    }

    fn finish_reasoning(&mut self) -> Vec<Bytes> {
        let Some((output_index, id, text)) = self.active_reasoning.take() else {
            return Vec::new();
        };
        let mut frames = Vec::new();
        frames.push(self.event(
            "response.reasoning_text.done",
            ReasoningDonePayload {
                content_index: 0,
                item_id: id.clone(),
                output_index,
                text: text.clone(),
            },
        ));
        frames.push(self.event(
            "response.content_part.done",
            ContentPartPayload {
                content_index: 0,
                item_id: id.clone(),
                output_index,
                part: OutputContent::ReasoningText { text: text.clone() },
            },
        ));
        let item = reasoning_item(id, text, String::new());
        frames.push(self.event(
            "response.output_item.done",
            ItemPayload {
                item: item.clone(),
                output_index,
            },
        ));
        self.output.push(item);
        frames
    }

    fn finish_text(&mut self) -> Vec<Bytes> {
        let Some((output_index, id, text)) = self.active_text.take() else {
            return Vec::new();
        };
        let mut frames = Vec::new();
        frames.push(self.event(
            "response.output_text.done",
            TextDonePayload {
                content_index: 0,
                item_id: id.clone(),
                logprobs: Vec::new(),
                output_index,
                text: text.clone(),
            },
        ));
        frames.push(self.event(
            "response.content_part.done",
            ContentPartPayload {
                content_index: 0,
                item_id: id.clone(),
                output_index,
                part: OutputContent::OutputText {
                    annotations: Vec::new(),
                    logprobs: Vec::new(),
                    text: text.clone(),
                },
            },
        ));
        let item = message_item(id, text);
        frames.push(self.event(
            "response.output_item.done",
            ItemPayload {
                item: item.clone(),
                output_index,
            },
        ));
        self.output.push(item);
        frames
    }

    pub(crate) fn tool_call(&mut self, call: ToolCall) -> anyhow::Result<Vec<Bytes>> {
        let mut frames = self.finish_reasoning();
        frames.extend(self.finish_text());
        let output_index = self.output.len();
        let item_id = function_item_id();
        let (namespace, call) = self.meta.tool_aliases.resolve_call(call)?;
        let public_name = call.name.clone();
        let arguments =
            serde_json::to_string(&call.arguments).expect("tool arguments are JSON values");
        frames.push(self.event(
            "response.output_item.added",
            ItemPayload {
                item: OutputItem::FunctionCall {
                    id: item_id.clone(),
                    status: "in_progress",
                    arguments: String::new(),
                    call_id: call.id.clone(),
                    name: public_name.clone(),
                    namespace: namespace.clone(),
                },
                output_index,
            },
        ));
        for fragment in utf8_fragments(&arguments, 64) {
            frames.push(self.event(
                "response.function_call_arguments.delta",
                ArgumentsDeltaPayload {
                    delta: fragment.to_owned(),
                    item_id: item_id.clone(),
                    output_index,
                },
            ));
        }
        frames.push(self.event(
            "response.function_call_arguments.done",
            ArgumentsDonePayload {
                arguments: arguments.clone(),
                item_id: item_id.clone(),
                name: public_name,
                output_index,
            },
        ));
        let item = OutputItem::FunctionCall {
            id: item_id,
            status: "completed",
            arguments,
            call_id: call.id,
            name: call.name,
            namespace,
        };
        frames.push(self.event(
            "response.output_item.done",
            ItemPayload {
                item: item.clone(),
                output_index,
            },
        ));
        self.output.push(item);
        Ok(frames)
    }

    pub(crate) fn completed(&mut self, finish_reason: &'static str, usage: Usage) -> Vec<Bytes> {
        let has_tool_calls = self
            .output
            .iter()
            .any(|item| matches!(item, OutputItem::FunctionCall { .. }));
        let active_text = self
            .active_text
            .as_ref()
            .map(|(_, _, text)| text.as_str())
            .unwrap_or_default();
        if let Err(error) =
            self.meta
                .text_format
                .validate_completion(active_text, has_tool_calls, finish_reason)
        {
            return vec![self.failed(format!("{error:#}"))];
        }
        let mut frames = self.finish_reasoning();
        frames.extend(self.finish_text());
        let (kind, status) = if finish_reason == "length" {
            ("response.incomplete", "incomplete")
        } else {
            ("response.completed", "completed")
        };
        frames.push(self.event(
            kind,
            ResponsePayload {
                response: self.meta.object(status, self.output.clone(), Some(usage)),
            },
        ));
        frames
    }

    pub(crate) fn failed(&mut self, message: String) -> Bytes {
        self.event(
            "response.failed",
            FailedPayload {
                response: FailedResponse {
                    id: self.meta.id.clone(),
                    object: "response",
                    created_at: self.meta.created_at,
                    status: "failed",
                    error: FailedError {
                        code: "generation_error",
                        message,
                    },
                },
            },
        )
    }
}

fn finish_decoder(
    output: &mut CollectedOutput,
    decoder: &mut GeneratedOutputDecoder<'_>,
    context: &ToolContext,
    model_finish: &'static str,
) -> anyhow::Result<()> {
    output.collect(decoder.finish(model_finish)?)?;
    validate_collected_output(output, context)
}

async fn serve_unary_gs<M>(state: AppState<M>, prepared: PreparedResponse) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let meta = prepared.meta();
    let input_tokens = prepared.prompt_tokens;
    let tool_context = prepared.tool_context;
    let native_output = prepared.native_output;
    let request = prepared.request;
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<CollectedOutput> {
        let model = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let memory = super::begin_direct_request_memory(&state, &*model, &request)?;
        let mut generation = GenerationStream::new(&*model, tokenizer, request)?;
        let mut output = CollectedOutput::new();
        let mut decoder = GeneratedOutputDecoder::new_with_native(
            tokenizer,
            tool_context.as_ref().map(ToolContext::decoder_config),
            native_output,
        )?;
        let mut memory = Some(memory);
        loop {
            let Some(event) = generation.next_token()? else {
                break;
            };
            if let Some(memory) = memory.take() {
                memory.commit();
                state.record_request_started(input_tokens);
            }
            output.completion_tokens += 1;
            let events = if event.finish_reason == Some("stop") {
                Vec::new()
            } else {
                decoder.push_token(event.token)?
            };
            if decoder.last_token_was_reasoning() {
                output.reasoning_tokens = output.reasoning_tokens.saturating_add(1);
            }
            output.collect(events)?;
            if let Some(reason) = event.finish_reason {
                output.finish_reason = reason;
                break;
            }
        }
        let model_finish = output.finish_reason;
        if let Some(context) = tool_context.as_ref() {
            finish_decoder(&mut output, &mut decoder, context, model_finish)?;
        } else {
            output.collect(decoder.finish(model_finish)?)?;
        }
        state
            .runtime_usage
            .record_output_tokens(u64::from(output.completion_tokens));
        Ok(output)
    })
    .await;
    match result {
        Ok(Ok(output)) => unary_response(meta, input_tokens, output),
        Ok(Err(error)) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "generation_error",
            format!("{error:#}"),
        ),
        Err(error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "generation_task_failed",
            format!("generation task failed: {error}"),
        ),
    }
}

async fn admit_request<M>(
    state: &AppState<M>,
    request: GenerateRequest,
) -> std::result::Result<AdmitReply, Response>
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
        return Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler_unavailable",
            "scheduler actor unavailable",
        ));
    }
    match reply_rx.await {
        Ok(Ok(reply)) => Ok(reply),
        Ok(Err(error)) => Err(super::api_error::ApiError::scheduler_admission(error)
            .into_response(super::api_error::ApiProtocol::OpenAi)),
        Err(_) => Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler_reply_lost",
            "scheduler reply lost",
        )),
    }
}

async fn serve_unary_scheduler<M>(state: AppState<M>, prepared: PreparedResponse) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let meta = prepared.meta();
    let input_tokens = prepared.prompt_tokens;
    let tool_context = prepared.tool_context;
    let native_output = prepared.native_output;
    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match admit_request(&state, prepared.request).await {
        Ok(reply) => reply,
        Err(response) => return response,
    };
    state.record_request_started(input_tokens);
    let mut output = CollectedOutput::new();
    let mut decoder = match GeneratedOutputDecoder::new_with_native(
        &state.tokenizer,
        tool_context.as_ref().map(ToolContext::decoder_config),
        native_output,
    ) {
        Ok(decoder) => decoder,
        Err(error) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_tool_parser",
                format!("{error:#}"),
            );
        }
    };
    while let Some(event) = event_rx.recv().await {
        output.completion_tokens += 1;
        let events = if event.finish_reason == Some("stop") {
            Ok(Vec::new())
        } else {
            decoder.push_token(event.token)
        };
        let events = match events {
            Ok(events) => events,
            Err(error) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "generated_output_decode_error",
                    format!("{error:#}"),
                );
            }
        };
        if decoder.last_token_was_reasoning() {
            output.reasoning_tokens = output.reasoning_tokens.saturating_add(1);
        }
        if let Err(error) = output.collect(events) {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generated_output_mapping_error",
                format!("{error:#}"),
            );
        }
        if let Some(reason) = event.finish_reason {
            output.finish_reason = reason;
            break;
        }
    }
    let model_finish = output.finish_reason;
    if let Some(context) = tool_context.as_ref() {
        if let Err(error) = finish_decoder(&mut output, &mut decoder, context, model_finish) {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generated_output_decode_error",
                format!("{error:#}"),
            );
        }
    } else if let Err(error) = decoder
        .finish(model_finish)
        .and_then(|events| output.collect(events))
    {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "generated_output_decode_error",
            format!("{error:#}"),
        );
    }
    state
        .runtime_usage
        .record_output_tokens(u64::from(output.completion_tokens));
    unary_response(meta, input_tokens, output)
}

pub(crate) fn stream_generated_events(
    formatter: &mut ResponsesStream,
    events: Vec<GeneratedOutputEvent>,
    call_names: &mut Vec<String>,
    finish_reason: &mut Option<&'static str>,
) -> anyhow::Result<Vec<Bytes>> {
    let mut frames = Vec::new();
    for event in events {
        match event {
            GeneratedOutputEvent::TextDelta(text) => frames.extend(formatter.text_delta(text)),
            GeneratedOutputEvent::ReasoningDelta(text) => {
                frames.extend(formatter.reasoning_delta(text));
            }
            GeneratedOutputEvent::ToolCall(call) => {
                call_names.push(call.name.clone());
                frames.extend(formatter.tool_call(call)?);
            }
            GeneratedOutputEvent::Finished(reason) => {
                anyhow::ensure!(
                    finish_reason.replace(reason.as_str()).is_none(),
                    "generated output emitted more than one terminal event"
                );
            }
            other => anyhow::bail!(
                "Responses adapter has no enabled producer mapping for generated {} output",
                other.kind()
            ),
        }
    }
    Ok(frames)
}

async fn serve_stream_scheduler<M>(state: AppState<M>, prepared: PreparedResponse) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let meta = prepared.meta();
    let input_tokens = prepared.prompt_tokens;
    let tool_context = prepared.tool_context;
    let native_output = prepared.native_output;
    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match admit_request(&state, prepared.request).await {
        Ok(reply) => reply,
        Err(response) => return response,
    };
    state.record_request_started(input_tokens);
    let tokenizer = state.tokenizer.clone();
    let runtime_usage = state.runtime_usage.clone();
    let (tx, rx, disconnect) = super::api_transport::disconnect_aware_sse_channel(8);
    tokio::spawn(async move {
        let mut formatter = ResponsesStream::new(meta);
        if tx.send(Ok(formatter.created())).await.is_err() {
            return;
        }
        let mut decoder = match GeneratedOutputDecoder::new_with_native(
            &tokenizer,
            tool_context.as_ref().map(ToolContext::decoder_config),
            native_output,
        ) {
            Ok(decoder) => decoder,
            Err(error) => {
                let frame = formatter.failed(format!("{error:#}"));
                let _ = tx.send(Ok(frame)).await;
                return;
            }
        };
        let mut output = CollectedOutput::new();
        let mut call_names = Vec::new();
        let mut typed_finish = None;
        while let Some(event) =
            super::api_transport::recv_or_disconnect(&disconnect, &mut event_rx).await
        {
            output.completion_tokens += 1;
            runtime_usage.record_output_tokens(1);
            let events = if event.finish_reason == Some("stop") {
                Ok(Vec::new())
            } else {
                decoder.push_token(event.token)
            };
            let events = match events {
                Ok(events) => events,
                Err(error) => {
                    let frame = formatter.failed(format!("{error:#}"));
                    let _ = tx.send(Ok(frame)).await;
                    return;
                }
            };
            if decoder.last_token_was_reasoning() {
                output.reasoning_tokens = output.reasoning_tokens.saturating_add(1);
            }
            let frames = match stream_generated_events(
                &mut formatter,
                events,
                &mut call_names,
                &mut typed_finish,
            ) {
                Ok(frames) => frames,
                Err(error) => {
                    let frame = formatter.failed(format!("{error:#}"));
                    let _ = tx.send(Ok(frame)).await;
                    return;
                }
            };
            for frame in frames {
                if tx.send(Ok(frame)).await.is_err() {
                    return;
                }
            }
            if let Some(reason) = event.finish_reason {
                output.finish_reason = reason;
                break;
            }
        }
        let events = match decoder.finish(output.finish_reason) {
            Ok(events) => events,
            Err(error) => {
                let frame = formatter.failed(format!("{error:#}"));
                let _ = tx.send(Ok(frame)).await;
                return;
            }
        };
        let frames = match stream_generated_events(
            &mut formatter,
            events,
            &mut call_names,
            &mut typed_finish,
        ) {
            Ok(frames) => frames,
            Err(error) => {
                let frame = formatter.failed(format!("{error:#}"));
                let _ = tx.send(Ok(frame)).await;
                return;
            }
        };
        for frame in frames {
            if tx.send(Ok(frame)).await.is_err() {
                return;
            }
        }
        if let Some(context) = tool_context.as_ref() {
            if let Err(error) =
                openai::validate_tool_choice_output(&context.constraint_options, &call_names)
            {
                let frame = formatter.failed(format!("{error:#}"));
                let _ = tx.send(Ok(frame)).await;
                return;
            }
        }
        output.finish_reason = typed_finish.unwrap_or(output.finish_reason);
        for frame in formatter.completed(
            output.finish_reason,
            Usage::new(input_tokens, output.completion_tokens)
                .with_reasoning_tokens(output.reasoning_tokens),
        ) {
            if tx.send(Ok(frame)).await.is_err() {
                return;
            }
        }
    });
    super::api_transport::disconnect_aware_sse_response(rx)
}

async fn serve_stream_gs<M>(state: AppState<M>, prepared: PreparedResponse) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let meta = prepared.meta();
    let input_tokens = prepared.prompt_tokens;
    let tool_context = prepared.tool_context;
    let native_output = prepared.native_output;
    let request = prepared.request;
    let (tx, rx, disconnect) = super::api_transport::disconnect_aware_sse_channel(8);
    let (init_tx, init_rx) = oneshot::channel::<anyhow::Result<()>>();
    tokio::task::spawn_blocking(move || {
        let model = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let memory = match super::begin_direct_request_memory(&state, &*model, &request) {
            Ok(memory) => memory,
            Err(error) => {
                let _ = init_tx.send(Err(error));
                return;
            }
        };
        let mut generation = match GenerationStream::new(&*model, tokenizer, request) {
            Ok(generation) => generation,
            Err(error) => {
                let _ = init_tx.send(Err(error));
                return;
            }
        };
        let first = match generation.next_token() {
            Ok(first) => first,
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
        let mut formatter = ResponsesStream::new(meta);
        if tx.blocking_send(Ok(formatter.created())).is_err() {
            return;
        }
        let mut decoder = match GeneratedOutputDecoder::new_with_native(
            tokenizer,
            tool_context.as_ref().map(ToolContext::decoder_config),
            native_output,
        ) {
            Ok(decoder) => decoder,
            Err(error) => {
                let _ = tx.blocking_send(Ok(formatter.failed(format!("{error:#}"))));
                return;
            }
        };
        let mut completion_tokens = 0_u32;
        let mut reasoning_tokens = 0_u32;
        let mut finish_reason = "stop";
        let mut call_names = Vec::new();
        let mut typed_finish = None;
        let mut first = Some(first);
        loop {
            if disconnect.is_cancelled() {
                return;
            }
            let event = match first.take() {
                Some(event) => Ok(event),
                None => generation.next_token(),
            };
            let event = match event {
                Ok(Some(event)) => event,
                Ok(None) => break,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(formatter.failed(format!("{error:#}"))));
                    return;
                }
            };
            completion_tokens += 1;
            state.runtime_usage.record_output_tokens(1);
            let events = if event.finish_reason == Some("stop") {
                Ok(Vec::new())
            } else {
                decoder.push_token(event.token)
            };
            let events = match events {
                Ok(events) => events,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(formatter.failed(format!("{error:#}"))));
                    return;
                }
            };
            if decoder.last_token_was_reasoning() {
                reasoning_tokens = reasoning_tokens.saturating_add(1);
            }
            let frames = match stream_generated_events(
                &mut formatter,
                events,
                &mut call_names,
                &mut typed_finish,
            ) {
                Ok(frames) => frames,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(formatter.failed(format!("{error:#}"))));
                    return;
                }
            };
            for frame in frames {
                if tx.blocking_send(Ok(frame)).is_err() {
                    return;
                }
            }
            if let Some(reason) = event.finish_reason {
                finish_reason = reason;
                break;
            }
        }
        if disconnect.is_cancelled() {
            return;
        }
        let events = match decoder.finish(finish_reason) {
            Ok(events) => events,
            Err(error) => {
                let _ = tx.blocking_send(Ok(formatter.failed(format!("{error:#}"))));
                return;
            }
        };
        let frames = match stream_generated_events(
            &mut formatter,
            events,
            &mut call_names,
            &mut typed_finish,
        ) {
            Ok(frames) => frames,
            Err(error) => {
                let _ = tx.blocking_send(Ok(formatter.failed(format!("{error:#}"))));
                return;
            }
        };
        for frame in frames {
            if tx.blocking_send(Ok(frame)).is_err() {
                return;
            }
        }
        if let Some(context) = tool_context.as_ref() {
            if let Err(error) =
                openai::validate_tool_choice_output(&context.constraint_options, &call_names)
            {
                let _ = tx.blocking_send(Ok(formatter.failed(format!("{error:#}"))));
                return;
            }
        }
        finish_reason = typed_finish.unwrap_or(finish_reason);
        for frame in formatter.completed(
            finish_reason,
            Usage::new(input_tokens, completion_tokens).with_reasoning_tokens(reasoning_tokens),
        ) {
            if tx.blocking_send(Ok(frame)).is_err() {
                return;
            }
        }
    });
    match init_rx.await {
        Ok(Ok(())) => super::api_transport::disconnect_aware_sse_response(rx),
        Ok(Err(error)) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "generation_initialization_error",
            format!("{error:#}"),
        ),
        Err(error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "generation_initialization_channel_closed",
            format!("generation initialization channel closed: {error}"),
        ),
    }
}

pub(crate) async fn responses<M>(
    State(state): State<AppState<M>>,
    ApiJson(request): ApiJson<ResponsesRequest>,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    responses_with_state(state, request, false).await
}

pub(crate) async fn responses_with_state<M>(
    state: AppState<M>,
    request: ResponsesRequest,
    force_scheduler: bool,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let normalized = match request.normalize() {
        Ok(normalized) => normalized,
        Err(error) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("{error:#}"),
            );
        }
    };
    let prepared = match prepare_response(&state, normalized, force_scheduler).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    match (prepared.stream, prepared.use_scheduler) {
        (true, true) => serve_stream_scheduler(state, prepared).await,
        (true, false) => serve_stream_gs(state, prepared).await,
        (false, true) => serve_unary_scheduler(state, prepared).await,
        (false, false) => serve_unary_gs(state, prepared).await,
    }
}

pub(crate) async fn gemma4_drafter_responses(
    State(state): State<Gemma4DrafterAppState>,
    ApiJson(request): ApiJson<ResponsesRequest>,
) -> Response {
    responses_with_state(state.base, request, true).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn responses_errors_use_openai_json_and_retry_contracts() {
        let overloaded = crate::core::server::api_error::ApiError::scheduler_admission(
            crate::core::SchedulerError::QueueFull { capacity: 8 }.into(),
        )
        .into_response(crate::core::server::api_error::ApiProtocol::OpenAi);
        assert_eq!(overloaded.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(overloaded.headers()[header::RETRY_AFTER], "5");
        let bytes = axum::body::to_bytes(overloaded.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "scheduler_queue_full");

        let too_large = crate::core::server::api_error::ApiError::scheduler_admission(
            crate::core::SchedulerError::RequestTooLarge {
                required_total_tokens: 273,
                input_tokens: 17,
                requested_max_output_tokens: 256,
                server_max_context_tokens: 128,
                max_allowed_output_tokens: 111,
            }
            .into(),
        )
        .into_response(crate::core::server::api_error::ApiProtocol::OpenAi);
        assert_eq!(too_large.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(too_large.headers().get(header::RETRY_AFTER).is_none());
        let bytes = axum::body::to_bytes(too_large.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "request_token_capacity_exceeded");
        assert_eq!(body["error"]["details"]["input_tokens"], 17);
    }

    #[test]
    fn responses_adapter_rejects_typed_output_without_an_enabled_producer_mapping() {
        let mut output = CollectedOutput::new();
        let error = output
            .collect(vec![GeneratedOutputEvent::RefusalDelta(
                "cannot comply".to_owned(),
            )])
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("no enabled producer mapping for generated refusal"));
        assert!(output.content.is_empty());
    }

    fn request(value: serde_json::Value) -> ResponsesRequest {
        serde_json::from_value(value).expect("valid fixture")
    }

    #[test]
    fn responses_sampling_contract_rejects_nonstandard_and_invalid_fields() {
        for field in ["top_k", "repetition_penalty"] {
            let mut body = serde_json::json!({"model": "local", "input": "hi"});
            body.as_object_mut()
                .unwrap()
                .insert(field.to_owned(), serde_json::json!(1));
            let error = serde_json::from_value::<ResponsesRequest>(body).unwrap_err();
            assert!(error.to_string().contains(field), "{error}");
        }

        for body in [
            serde_json::json!({"model": "local", "input": "hi", "temperature": -0.1}),
            serde_json::json!({"model": "local", "input": "hi", "temperature": 2.1}),
            serde_json::json!({"model": "local", "input": "hi", "top_p": 0.0}),
            serde_json::json!({"model": "local", "input": "hi", "top_p": 1.1}),
        ] {
            assert!(request(body).normalize().is_err());
        }

        for (temperature, top_p) in [(0.0, 0.01), (2.0, 1.0)] {
            request(serde_json::json!({
                "model": "local",
                "input": "hi",
                "temperature": temperature,
                "top_p": top_p
            }))
            .normalize()
            .unwrap();
        }
    }

    #[test]
    fn converts_codex_function_round_trip() {
        let normalized = request(serde_json::json!({
            "model": "local",
            "instructions": "Be concise",
            "input": [
                {"type":"message","role":"user","content":[{"type":"input_text","text":"weather"}]},
                {"type":"function_call","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Tokyo\"}"},
                {"type":"function_call_output","call_id":"call_1","output":"22 C"}
            ],
            "tools": [{
                "type":"function",
                "name":"weather",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false},
                "strict":true
            }],
            "tool_choice":"auto",
            "parallel_tool_calls":false,
            "store":false,
            "stream":true,
            "reasoning":{"effort":null,"summary":"auto"},
            "include":["reasoning.encrypted_content"],
            "prompt_cache_key":"session",
            "client_metadata":{}
        }))
        .normalize()
        .expect("Codex request normalizes");
        assert_eq!(normalized.chat.messages.len(), 4);
        assert_eq!(normalized.chat.messages[0].role, "system");
        assert_eq!(normalized.chat.messages[2].tool_calls.len(), 1);
        assert_eq!(
            normalized.chat.messages[3].tool_call_id.as_deref(),
            Some("call_1")
        );
    }

    #[test]
    fn accepts_oh_my_pi_easy_input_message_without_type() {
        let normalized = request(serde_json::json!({
            "model": "local",
            "instructions": "Use tools only when required",
            "input": [{
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "请只回复：IronMLX connection OK"
                }]
            }],
            "stream": true,
            "prompt_cache_key": "omp-session",
            "store": false,
            "max_output_tokens": 1024,
            "tools": [{
                "type": "function",
                "name": "bash",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "env": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": {"type": "string"}
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": false
                }
            }]
        }))
        .normalize()
        .expect("OMP easy input message normalizes");

        assert_eq!(normalized.chat.messages.len(), 2);
        assert_eq!(normalized.chat.messages[0].role, "system");
        assert_eq!(normalized.chat.messages[1].role, "user");
        assert_eq!(
            normalized.chat.messages[1]
                .content
                .to_flat_string(&mut std::collections::VecDeque::new()),
            "请只回复：IronMLX connection OK"
        );
        let tools = normalized
            .chat
            .tools
            .expect("OMP tools survive normalization");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "bash");
    }

    #[test]
    fn rejects_untyped_responses_item_without_easy_message_shape() {
        let result = serde_json::from_value::<ResponsesRequest>(serde_json::json!({
            "model": "local",
            "input": [{"role": "user"}]
        }));

        assert!(result.is_err());
    }

    #[test]
    fn accepts_oh_my_pi_blocking_tool_schemas_through_responses_normalization() {
        let normalized = request(serde_json::json!({
            "model": "local",
            "input": "Use the agent tools",
            "tools": [
                {
                    "type": "function",
                    "name": "bash",
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
                },
                {
                    "type": "function",
                    "name": "hub",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string"},
                            "name": {"type": "string", "maxLength": 48},
                            "application": {"type": "string", "minLength": 1},
                            "timeout": {"type": "number", "exclusiveMinimum": 0}
                        },
                        "required": ["op"],
                        "additionalProperties": false
                    }
                },
                {
                    "type": "function",
                    "name": "ask",
                    "strict": true,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "options": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "label": {"type": "string"},
                                                    "description": {
                                                        "anyOf": [
                                                            {"type": "string"},
                                                            {"type": "null"}
                                                        ]
                                                    }
                                                },
                                                "required": ["label", "description"],
                                                "additionalProperties": false
                                            }
                                        }
                                    },
                                    "required": ["options"],
                                    "additionalProperties": false
                                }
                            }
                        },
                        "required": ["questions"],
                        "additionalProperties": false
                    }
                }
            ],
            "store": false
        }))
        .normalize()
        .unwrap();

        let prepared = openai::prepare_tool_request(
            &normalized.chat,
            Some(crate::core::tool_calling::ToolDialect::Qwen35),
        )
        .unwrap()
        .expect("tools prepared");
        assert_eq!(prepared.definitions.len(), 3);
    }

    #[test]
    fn reasoning_effort_controls_native_template_and_round_trip_history() {
        let normalized = request(serde_json::json!({
            "model":"local",
            "input":[
                {
                    "type":"reasoning",
                    "id":"rs_1",
                    "summary":[],
                    "content":[{"type":"reasoning_text","text":"inspect the weather first"}],
                    "status":"completed"
                },
                {
                    "type":"function_call",
                    "call_id":"call_1",
                    "name":"weather",
                    "arguments":"{\"city\":\"Tokyo\"}"
                },
                {"type":"function_call_output","call_id":"call_1","output":"22 C"}
            ],
            "tools":[{
                "type":"function",
                "name":"weather",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
            }],
            "reasoning":{"effort":"high","summary":"none"}
        }))
        .normalize()
        .expect("reasoning request normalizes");
        assert_eq!(
            normalized.chat.chat_template_kwargs,
            Some(serde_json::json!({"enable_thinking":true}))
        );
        assert_eq!(
            normalized.chat.messages[0].reasoning_content.as_deref(),
            Some("inspect the weather first")
        );
        assert_eq!(normalized.reasoning.effort, Some(ReasoningEffort::High));
        assert_eq!(
            normalized.reasoning.summary,
            Some(ReasoningSummaryMode::None)
        );
    }

    #[test]
    fn encrypted_reasoning_without_plaintext_cannot_be_replayed() {
        let error = request(serde_json::json!({
            "model":"local",
            "input":[
                {
                    "type":"reasoning",
                    "encrypted_content":"opaque",
                    "summary":[],
                    "content":[]
                },
                {"type":"message","role":"assistant","content":"done"}
            ]
        }))
        .normalize()
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("encrypted reasoning cannot be replayed"));
    }

    #[test]
    fn coalesces_responses_instructions_and_developer_items() {
        let normalized = request(serde_json::json!({
            "model":"local",
            "instructions":"Base instructions",
            "input":[
                {"type":"message","role":"developer","content":"Repository instructions"},
                {"type":"message","role":"user","content":"First user message"},
                {"type":"message","role":"system","content":"Later system policy"},
                {"type":"message","role":"user","content":"Second user message"}
            ]
        }))
        .normalize()
        .expect("instructions normalize");
        assert_eq!(normalized.chat.messages.len(), 3);
        assert_eq!(normalized.chat.messages[0].role, "system");
        assert!(matches!(
            &normalized.chat.messages[0].content,
            Content::Text(text)
                if text == "Base instructions\n\nRepository instructions\n\nLater system policy"
        ));
        assert!(normalized.chat.messages[1..]
            .iter()
            .all(|message| message.role == "user"));
    }

    #[test]
    fn converts_codex_namespace_tools_and_restores_wire_identity() {
        let normalized = request(serde_json::json!({
            "model":"local",
            "input":[
                {"type":"message","role":"user","content":"check time"},
                {"type":"function_call","call_id":"call_1","namespace":"clock","name":"current_time","arguments":"{}"},
                {"type":"function_call_output","call_id":"call_1","output":"12:00"}
            ],
            "tools":[{
                "type":"namespace",
                "name":"clock",
                "description":"Local clock tools.",
                "tools":[{
                    "type":"function",
                    "name":"current_time",
                    "description":"Read local time.",
                    "strict":false,
                    "defer_loading":false,
                    "parameters":{"type":"object","properties":{},"additionalProperties":false}
                }]
            }],
            "tool_choice":"auto"
        }))
        .normalize()
        .expect("namespace request normalizes");
        let internal = normalized
            .tool_aliases
            .internal_by_namespace
            .get("clock")
            .expect("namespace alias")
            .to_owned();
        assert_eq!(
            normalized.chat.messages[1].tool_calls[0].function.name,
            internal
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                &normalized.chat.messages[1].tool_calls[0].function.arguments
            )
            .expect("dispatcher arguments are JSON"),
            serde_json::json!({"name":"current_time","arguments":{}})
        );

        let meta = ResponseMeta::from_normalized(&normalized, "local".into());
        let response = function_item(
            "fc_1".into(),
            ToolCall {
                id: "call_2".into(),
                name: internal,
                arguments: serde_json::json!({
                    "name":"current_time",
                    "arguments":{}
                }),
            },
            &meta.tool_aliases,
        )
        .expect("namespace call resolves");
        let value = serde_json::to_value(response).expect("response item serializes");
        assert_eq!(value["namespace"], "clock");
        assert_eq!(value["name"], "current_time");
    }

    #[test]
    fn wraps_dynamic_non_strict_namespace_arguments_as_json() {
        let normalized = request(serde_json::json!({
            "model":"local",
            "input":[
                {"type":"message","role":"user","content":"update document"},
                {
                    "type":"function_call",
                    "call_id":"call_1",
                    "namespace":"documents",
                    "name":"execute",
                    "arguments":"{\"args\":{\"cell\":\"A1\",\"value\":7}}"
                },
                {"type":"function_call_output","call_id":"call_1","output":"ok"}
            ],
            "tools":[{
                "type":"namespace",
                "name":"documents",
                "description":"Dynamic document tools.",
                "tools":[{
                    "type":"function",
                    "name":"execute",
                    "strict":false,
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "args":{
                                "type":"object",
                                "properties":{},
                                "additionalProperties":{}
                            }
                        },
                        "required":["args"],
                        "additionalProperties":false
                    }
                }]
            }]
        }))
        .normalize()
        .expect("dynamic namespace request normalizes");
        let internal = normalized
            .tool_aliases
            .internal_by_namespace
            .get("documents")
            .expect("namespace alias")
            .to_owned();
        let envelope = serde_json::from_str::<serde_json::Value>(
            &normalized.chat.messages[1].tool_calls[0].function.arguments,
        )
        .expect("dispatcher arguments are JSON");
        assert_eq!(envelope["name"], "execute");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                envelope["arguments_json"].as_str().expect("JSON envelope")
            )
            .expect("child arguments decode"),
            serde_json::json!({"args":{"cell":"A1","value":7}})
        );

        let item = function_item(
            "fc_1".into(),
            ToolCall {
                id: "call_2".into(),
                name: internal,
                arguments: serde_json::json!({
                    "name":"execute",
                    "arguments_json":"{\"args\":{\"cell\":\"B2\"}}"
                }),
            },
            &normalized.tool_aliases,
        )
        .expect("dynamic namespace call resolves");
        let value = serde_json::to_value(item).expect("response item serializes");
        assert_eq!(value["namespace"], "documents");
        assert_eq!(value["name"], "execute");
        assert_eq!(value["arguments"], "{\"args\":{\"cell\":\"B2\"}}");
    }

    #[test]
    fn collapses_large_non_strict_namespace_to_one_bounded_dispatcher() {
        let children = (0..100)
            .map(|index| ResponseNamespaceTool::Function {
                name: format!("tool_{index}"),
                description: Some(format!("Tool {index}")),
                parameters: serde_json::json!({
                    "type":"object",
                    "properties":{"value":{"type":"string"}},
                    "required":["value"],
                    "additionalProperties":false
                }),
                strict: Some(false),
                defer_loading: None,
            })
            .collect();
        let (flattened, aliases) = flatten_response_tools(&[ResponseTool::Namespace {
            name: "large".into(),
            description: "Large tool catalog.".into(),
            tools: children,
        }])
        .expect("large namespace collapses");
        assert_eq!(flattened.len(), 1);
        assert!(flattened[0].function.parameters.get("anyOf").is_none());
        assert_eq!(
            flattened[0].function.parameters["properties"]["name"]["enum"]
                .as_array()
                .expect("name enum")
                .len(),
            100
        );
        let dispatcher = aliases
            .namespace_by_internal
            .values()
            .next()
            .expect("namespace dispatcher");
        assert!(dispatcher
            .children
            .values()
            .all(|encoding| matches!(encoding, NamespaceArgumentEncoding::JsonString)));
    }

    #[test]
    fn rejects_stateful_and_hosted_semantics() {
        let error = request(serde_json::json!({
            "model":"local",
            "input":"hello",
            "store":true
        }))
        .normalize()
        .unwrap_err();
        assert!(error.to_string().contains("store=true"));
    }

    #[test]
    fn responses_stream_has_no_chat_done_sentinel() {
        let meta = ResponseMeta {
            id: "resp_test".into(),
            created_at: 1,
            instructions: None,
            text_format: ResponseTextFormat::Text,
            max_output_tokens: 10,
            model: "local".into(),
            parallel_tool_calls: true,
            temperature: None,
            tool_choice: serde_json::json!("auto"),
            tools: Vec::new(),
            top_p: None,
            reasoning: ReasoningRequest::default(),
            tool_aliases: ToolAliases::default(),
        };
        let mut stream = ResponsesStream::new(meta);
        let mut frames = vec![stream.created()];
        frames.extend(stream.text_delta("hello".into()));
        frames.extend(stream.completed("stop", Usage::new(2, 1)));
        let wire = frames
            .into_iter()
            .map(|frame| String::from_utf8(frame.to_vec()).unwrap())
            .collect::<String>();
        assert!(wire.contains("response.completed"));
        assert!(!wire.contains("[DONE]"));
    }

    #[test]
    fn reasoning_stream_uses_native_responses_lifecycle() {
        let meta = ResponseMeta {
            id: "resp_reasoning".into(),
            created_at: 1,
            instructions: None,
            text_format: ResponseTextFormat::Text,
            max_output_tokens: 32,
            model: "local".into(),
            parallel_tool_calls: true,
            temperature: None,
            tool_choice: serde_json::json!("auto"),
            tools: Vec::new(),
            top_p: None,
            reasoning: ReasoningRequest {
                effort: Some(ReasoningEffort::Medium),
                summary: Some(ReasoningSummaryMode::None),
            },
            tool_aliases: ToolAliases::default(),
        };
        let mut stream = ResponsesStream::new(meta);
        let mut frames = vec![stream.created()];
        frames.extend(stream.reasoning_delta("check".into()));
        frames.extend(stream.text_delta("answer".into()));
        frames.extend(stream.completed("stop", Usage::new(4, 3).with_reasoning_tokens(2)));
        let events = frames
            .into_iter()
            .map(|frame| {
                let wire = String::from_utf8(frame.to_vec()).unwrap();
                let data = wire
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap();
                serde_json::from_str::<serde_json::Value>(data).unwrap()
            })
            .collect::<Vec<_>>();
        let kinds = events
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let completed = events.last().unwrap();
        assert_eq!(
            completed["response"]["usage"]["output_tokens_details"]["reasoning_tokens"],
            2
        );
        assert_eq!(completed["response"]["reasoning"]["effort"], "medium");
    }

    #[test]
    fn reasoning_item_uses_native_responses_wire_shape() {
        let value = serde_json::to_value(reasoning_item(
            "rs_1".into(),
            "inspect inputs".into(),
            String::new(),
        ))
        .expect("reasoning item serializes");
        assert_eq!(
            value,
            serde_json::json!({
                "type":"reasoning",
                "id":"rs_1",
                "summary":[],
                "content":[{"type":"reasoning_text","text":"inspect inputs"}]
            })
        );
    }

    #[test]
    fn reasoning_stream_closes_before_function_call() {
        let meta = ResponseMeta {
            id: "resp_reasoning_tool".into(),
            created_at: 1,
            instructions: None,
            text_format: ResponseTextFormat::Text,
            max_output_tokens: 32,
            model: "local".into(),
            parallel_tool_calls: false,
            temperature: None,
            tool_choice: serde_json::json!({"type":"function","name":"weather"}),
            tools: Vec::new(),
            top_p: None,
            reasoning: ReasoningRequest {
                effort: Some(ReasoningEffort::High),
                summary: Some(ReasoningSummaryMode::None),
            },
            tool_aliases: ToolAliases::default(),
        };
        let mut stream = ResponsesStream::new(meta);
        let mut frames = stream.reasoning_delta("inspect weather".into());
        frames.extend(
            stream
                .tool_call(ToolCall {
                    id: "call_1".into(),
                    name: "weather".into(),
                    arguments: serde_json::json!({"city":"Tokyo"}),
                })
                .expect("function call formats"),
        );
        let kinds = frames
            .into_iter()
            .map(|frame| {
                let wire = String::from_utf8(frame.to_vec()).unwrap();
                let data = wire
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap();
                serde_json::from_str::<serde_json::Value>(data).unwrap()["type"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect::<Vec<_>>();
        let reasoning_done = kinds
            .iter()
            .position(|kind| kind == "response.output_item.done")
            .expect("reasoning item closes");
        assert_eq!(
            kinds[reasoning_done + 1],
            "response.output_item.added",
            "function item must start only after reasoning item is complete"
        );
    }

    #[test]
    fn function_stream_uses_native_responses_lifecycle() {
        let meta = ResponseMeta {
            id: "resp_test".into(),
            created_at: 1,
            instructions: None,
            text_format: ResponseTextFormat::Text,
            max_output_tokens: 32,
            model: "local".into(),
            parallel_tool_calls: false,
            temperature: None,
            tool_choice: serde_json::json!({"type":"function","name":"weather"}),
            tools: Vec::new(),
            top_p: None,
            reasoning: ReasoningRequest::default(),
            tool_aliases: ToolAliases::default(),
        };
        let mut stream = ResponsesStream::new(meta);
        let mut frames = vec![stream.created()];
        frames.extend(
            stream
                .tool_call(ToolCall {
                    id: "call_1".into(),
                    name: "weather".into(),
                    arguments: serde_json::json!({"city":"東京"}),
                })
                .expect("function call formats"),
        );
        frames.extend(stream.completed("tool_calls", Usage::new(10, 5)));
        let events = frames
            .into_iter()
            .map(|frame| {
                let wire = String::from_utf8(frame.to_vec()).unwrap();
                let data = wire
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap();
                serde_json::from_str::<serde_json::Value>(data).unwrap()
            })
            .collect::<Vec<_>>();
        let kinds = events
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                "response.created",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (sequence, event) in events.iter().enumerate() {
            assert_eq!(event["sequence_number"], sequence as u64);
        }
        assert_eq!(events[4]["item"]["call_id"], "call_1");
        assert_eq!(events[4]["item"]["arguments"], "{\"city\":\"東京\"}");
    }

    #[test]
    fn normalizes_native_named_tool_choice() {
        let normalized = request(serde_json::json!({
            "model":"local",
            "input":"weather",
            "tools":[{
                "type":"function",
                "name":"weather",
                "parameters":{"type":"object","properties":{},"additionalProperties":false}
            }],
            "tool_choice":{"type":"function","name":"weather"},
            "store":false
        }))
        .normalize()
        .unwrap();
        assert_eq!(
            normalized.chat.tool_choice,
            Some(serde_json::json!({
                "type":"function",
                "function":{"name":"weather"}
            }))
        );
    }

    #[test]
    fn rejects_file_ids_and_invalid_structured_text_format() {
        let image_error = request(serde_json::json!({
            "model":"local",
            "input":[{"type":"message","role":"user","content":[{
                "type":"input_image","file_id":"file_1"
            }]}],
            "store":false
        }))
        .normalize()
        .unwrap_err();
        assert!(image_error.to_string().contains("file_id"));

        let format_error = request(serde_json::json!({
            "model":"local",
            "input":"hello",
            "text":{"format":{"type":"json_schema","name":"answer","schema":{}}},
            "store":false
        }))
        .normalize()
        .unwrap_err();
        assert!(format!("{format_error:#}").contains("root type must be `object`"));
    }

    #[test]
    fn normalizes_json_object_and_strict_json_schema_formats() {
        let json_object = request(serde_json::json!({
            "model":"local",
            "input":"Return an object",
            "text":{"format":{"type":"json_object"}},
            "store":false
        }))
        .normalize()
        .unwrap();
        assert!(matches!(
            json_object.text_format,
            ResponseTextFormat::JsonObject
        ));
        assert!(matches!(
            &json_object.chat.messages[0].content,
            Content::Text(text) if text.contains("return only one valid JSON object")
        ));

        let json_schema = request(serde_json::json!({
            "model":"local",
            "input":"Return weather",
            "text":{"format":{
                "type":"json_schema",
                "name":"weather_answer",
                "description":"A compact forecast.",
                "schema":{
                    "type":"object",
                    "properties":{
                        "city":{"type":"string"},
                        "days":{"type":"integer"}
                    },
                    "required":["city","days"],
                    "additionalProperties":false
                },
                "strict":true
            }},
            "store":false
        }))
        .normalize()
        .unwrap();
        assert!(matches!(
            &json_schema.text_format,
            ResponseTextFormat::JsonSchema { name, strict, .. }
                if name == "weather_answer" && *strict == Some(true)
        ));
        assert!(matches!(
            &json_schema.chat.messages[0].content,
            Content::Text(text)
                if text.contains("weather_answer")
                    && text.contains("A compact forecast.")
                    && text.find("\"city\"") < text.find("\"days\"")
        ));
    }

    #[test]
    fn strict_structured_format_rejects_non_strict_or_unsupported_schemas() {
        let missing_required = request(serde_json::json!({
            "model":"local",
            "input":"hello",
            "text":{"format":{
                "type":"json_schema",
                "name":"answer",
                "schema":{
                    "type":"object",
                    "properties":{"answer":{"type":"string"}},
                    "required":[],
                    "additionalProperties":false
                },
                "strict":true
            }},
            "store":false
        }))
        .normalize()
        .unwrap_err();
        assert!(missing_required
            .to_string()
            .contains("must be listed in required"));

        let unsupported = request(serde_json::json!({
            "model":"local",
            "input":"hello",
            "text":{"format":{
                "type":"json_schema",
                "name":"answer",
                "schema":{
                    "type":"object",
                    "properties":{"answer":{"type":"string","pattern":"x"}},
                    "required":["answer"],
                    "additionalProperties":false
                }
            }},
            "store":false
        }))
        .normalize()
        .unwrap_err();
        assert!(format!("{unsupported:#}").contains("unsupported JSON Schema keyword `pattern`"));
    }

    #[test]
    fn required_tools_do_not_inject_structured_final_answer_instructions() {
        let normalized = request(serde_json::json!({
            "model":"local",
            "input":"Use the tool",
            "tools":[{
                "type":"function",
                "name":"weather",
                "parameters":{
                    "type":"object",
                    "properties":{},
                    "required":[],
                    "additionalProperties":false
                }
            }],
            "tool_choice":"required",
            "text":{"format":{"type":"json_object"}},
            "store":false
        }))
        .normalize()
        .unwrap();
        assert!(normalized.chat.messages.iter().all(|message| !matches!(
            &message.content,
            Content::Text(text) if text.contains("return only one valid JSON object")
        )));
    }

    #[test]
    fn structured_stream_fails_without_emitting_completed_for_invalid_json() {
        let meta = ResponseMeta {
            id: "resp_structured".into(),
            created_at: 1,
            instructions: None,
            text_format: ResponseTextFormat::JsonObject,
            max_output_tokens: 32,
            model: "local".into(),
            parallel_tool_calls: true,
            temperature: None,
            tool_choice: serde_json::json!("none"),
            tools: Vec::new(),
            top_p: None,
            reasoning: ReasoningRequest::default(),
            tool_aliases: ToolAliases::default(),
        };
        let mut stream = ResponsesStream::new(meta);
        stream.text_delta("not json".into());
        let frames = stream.completed("stop", Usage::new(2, 2));
        let wire = frames
            .iter()
            .map(|frame| String::from_utf8(frame.to_vec()).unwrap())
            .collect::<String>();
        assert!(wire.contains("response.failed"));
        assert!(!wire.contains("response.completed"));
    }
}
