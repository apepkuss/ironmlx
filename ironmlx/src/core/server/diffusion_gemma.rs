//! DiffusionGemma HTTP server lane.
//!
//! This module is intentionally separate from the causal `AppState<M>` /
//! `SchedulerActor` server path. DiffusionGemma is a block-diffusion model, so
//! requests are admitted through a bounded serial lane and completed as either
//! unary responses or SSE streams of committed block-diffusion output.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use mlx::Array;
use serde::Serialize;
use tokio::sync::{mpsc, Mutex, OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio_stream::wrappers::ReceiverStream;

use crate::core::constrained::{ConstraintPlan, ToolConstraintOptions};
use crate::core::server::chat_format::render_and_encode;
use crate::core::server::vision::expand_decoded_messages;
use crate::core::server::VisionInputConfig;
use crate::core::tokenizer::Tokenizer;
use crate::core::tool_calling::{
    AssistantOutputEvent, ToolCall, ToolCallParser, ToolDefinition, ToolDialect,
};
use crate::models::{
    DiffusionGemmaGenerateEvent, DiffusionGemmaGenerationConfig, DiffusionGemmaModel,
};
use crate::Result;

const DEFAULT_DIFFUSION_GEMMA_QUEUE_CAPACITY: usize = 8;

#[derive(Clone)]
pub struct DiffusionGemmaAppState {
    pub model: Arc<Mutex<DiffusionGemmaModel>>,
    pub tokenizer: Arc<Tokenizer>,
    pub generation_config: DiffusionGemmaGenerationConfig,
    pub model_id: String,
    pub model_weight_bytes: usize,
    pub vision_input: VisionInputConfig,
    pub lane: Arc<DiffusionGemmaLane>,
    pub runtime_usage: Arc<crate::core::runtime_usage::ModelRuntimeUsageCounters>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffusionGemmaLaneStats {
    pub active_requests: usize,
    pub queued_requests: usize,
    pub queue_capacity: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionGemmaLaneError {
    Overloaded,
    Closed,
}

pub struct DiffusionGemmaLane {
    queue_slots: Arc<Semaphore>,
    execution_slot: Arc<Semaphore>,
    active_requests: AtomicUsize,
    queued_requests: AtomicUsize,
    queue_capacity: usize,
}

impl DiffusionGemmaLane {
    pub fn new(queue_capacity: usize) -> Self {
        Self {
            queue_slots: Arc::new(Semaphore::new(queue_capacity + 1)),
            execution_slot: Arc::new(Semaphore::new(1)),
            active_requests: AtomicUsize::new(0),
            queued_requests: AtomicUsize::new(0),
            queue_capacity,
        }
    }

    pub fn stats(&self) -> DiffusionGemmaLaneStats {
        DiffusionGemmaLaneStats {
            active_requests: self.active_requests.load(Ordering::SeqCst),
            queued_requests: self.queued_requests.load(Ordering::SeqCst),
            queue_capacity: self.queue_capacity,
        }
    }

    pub async fn enter(
        self: Arc<Self>,
    ) -> std::result::Result<DiffusionGemmaLaneGuard, DiffusionGemmaLaneError> {
        let queue_permit = match self.queue_slots.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::NoPermits) => return Err(DiffusionGemmaLaneError::Overloaded),
            Err(TryAcquireError::Closed) => return Err(DiffusionGemmaLaneError::Closed),
        };
        self.queued_requests.fetch_add(1, Ordering::SeqCst);
        let queued = DiffusionGemmaQueuedSlot {
            lane: Arc::clone(&self),
            queue_permit: Some(queue_permit),
        };
        let execution_permit = self
            .execution_slot
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| DiffusionGemmaLaneError::Closed)?;
        Ok(queued.promote(execution_permit))
    }
}

struct DiffusionGemmaQueuedSlot {
    lane: Arc<DiffusionGemmaLane>,
    queue_permit: Option<OwnedSemaphorePermit>,
}

impl DiffusionGemmaQueuedSlot {
    fn promote(mut self, execution_permit: OwnedSemaphorePermit) -> DiffusionGemmaLaneGuard {
        let queue_permit = self
            .queue_permit
            .take()
            .expect("queued slot must own its queue permit");
        self.lane.queued_requests.fetch_sub(1, Ordering::SeqCst);
        self.lane.active_requests.fetch_add(1, Ordering::SeqCst);
        DiffusionGemmaLaneGuard {
            lane: Arc::clone(&self.lane),
            _queue_permit: queue_permit,
            _execution_permit: execution_permit,
        }
    }
}

impl Drop for DiffusionGemmaQueuedSlot {
    fn drop(&mut self) {
        if self.queue_permit.is_some() {
            self.lane.queued_requests.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

pub struct DiffusionGemmaLaneGuard {
    lane: Arc<DiffusionGemmaLane>,
    _queue_permit: OwnedSemaphorePermit,
    _execution_permit: OwnedSemaphorePermit,
}

impl Drop for DiffusionGemmaLaneGuard {
    fn drop(&mut self) {
        self.lane.active_requests.fetch_sub(1, Ordering::SeqCst);
    }
}

struct PreparedRequest {
    prompt_ids: Vec<u32>,
    pixel_values: Option<Vec<Array>>,
    image_grid_thw: Option<Vec<(i32, i32, i32)>>,
    image_token_id: i32,
    constraint: Option<ConstraintPlan>,
    skip_special_tokens: bool,
}

struct PreparedOpenAiRequest {
    generation: PreparedRequest,
    tool_context: Option<ToolResponseContext>,
}

struct OpenAiStreamRequest {
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    model_label: String,
    prompt_tokens: u32,
    include_usage: bool,
    tool_context: Option<ToolResponseContext>,
}

struct ResponsesStreamRequest {
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    prompt_tokens: u32,
    tool_context: Option<ToolResponseContext>,
    meta: super::responses::ResponseMeta,
}

struct ToolResponseContext {
    dialect: ToolDialect,
    definitions: Vec<ToolDefinition>,
    constraint_options: ToolConstraintOptions,
    output_schema: Option<serde_json::Value>,
}

struct CompletionParts {
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
    finish_reason: &'static str,
    completion_tokens: u32,
}

enum CompletionError {
    Overloaded,
    Internal(String),
}

impl From<DiffusionGemmaLaneError> for CompletionError {
    fn from(value: DiffusionGemmaLaneError) -> Self {
        match value {
            DiffusionGemmaLaneError::Overloaded => Self::Overloaded,
            DiffusionGemmaLaneError::Closed => {
                Self::Internal("DiffusionGemma serial lane is closed".to_string())
            }
        }
    }
}

impl CompletionError {
    fn into_response(self) -> Response {
        match self {
            Self::Overloaded => overloaded_response(),
            Self::Internal(message) => internal_error_response(message),
        }
    }
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    message: &'static str,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAiCompletionToolCall>,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiCompletionFunctionCall,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionChoice {
    index: u32,
    message: OpenAiCompletionMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiCompletionChoice>,
    usage: OpenAiUsage,
}

#[derive(Debug, Serialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Serialize)]
struct AnthropicContentBlockText {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
}

#[derive(Debug, Serialize)]
struct AnthropicMessageEnvelope {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    content: Vec<AnthropicContentBlockText>,
    model: String,
    stop_reason: Option<&'static str>,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
struct DiffusionGemmaHealth {
    status: &'static str,
    scheduler: &'static str,
    active_requests: usize,
    queued_requests: usize,
    queue_capacity: usize,
}

#[derive(Debug, Serialize)]
struct OpenAiStreamChoice<T> {
    index: u32,
    delta: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct OpenAiStreamChunk<T> {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiStreamChoice<T>>,
}

#[derive(Debug, Serialize)]
struct OpenAiStreamUsageChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<serde_json::Value>,
    usage: OpenAiUsage,
}

#[derive(Debug, Serialize)]
struct OpenAiDeltaRole {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAiDeltaContent<'a> {
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct OpenAiDeltaToolCalls {
    tool_calls: Vec<OpenAiDeltaToolCall>,
}

#[derive(Debug, Serialize)]
struct OpenAiDeltaToolCall {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    function: OpenAiDeltaFunctionCall,
}

#[derive(Debug, Serialize)]
struct OpenAiDeltaFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    arguments: String,
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn gen_openai_id() -> String {
    format!("chatcmpl-{}", now_unix())
}

fn gen_anthropic_id() -> String {
    format!("msg_{}", now_unix())
}

fn overloaded_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorEnvelope {
            error: ErrorBody {
                message: "DiffusionGemma serial lane is overloaded; retry later",
                kind: "overloaded",
            },
        }),
    )
        .into_response()
}

fn internal_error_response(message: String) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, message).into_response()
}

fn diffusion_event_is_length_sentinel(event: &DiffusionGemmaGenerateEvent) -> bool {
    event.finish_reason == Some("length") && event.token == 0 && event.text.is_empty()
}

fn collect_events(
    events: Vec<DiffusionGemmaGenerateEvent>,
    default_finish: &'static str,
) -> CompletionParts {
    let mut content = String::new();
    let mut finish_reason = default_finish;
    let mut completion_tokens = 0_u32;
    for event in events {
        if !diffusion_event_is_length_sentinel(&event) {
            content.push_str(&event.text);
            completion_tokens += 1;
        }
        if let Some(reason) = event.finish_reason {
            finish_reason = reason;
            break;
        }
    }
    CompletionParts {
        content: Some(content),
        tool_calls: Vec::new(),
        finish_reason,
        completion_tokens,
    }
}

fn collect_tool_events(
    events: Vec<DiffusionGemmaGenerateEvent>,
    context: ToolResponseContext,
    default_finish: &'static str,
) -> Result<CompletionParts> {
    let mut parser = ToolCallParser::new_with_output_schema(
        context.dialect,
        uuid::Uuid::new_v4().simple().to_string(),
        &context.definitions,
        context.output_schema,
    )?;
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut finish_reason = default_finish;
    let mut completion_tokens = 0_u32;
    for event in events {
        if !diffusion_event_is_length_sentinel(&event) {
            completion_tokens = completion_tokens.saturating_add(1);
            collect_tool_parser_events(&mut content, &mut tool_calls, parser.push(&event.text)?);
        }
        if let Some(reason) = event.finish_reason {
            finish_reason = reason;
            break;
        }
    }
    let (tail, saw_tool_call) = parser.finish()?;
    collect_tool_parser_events(&mut content, &mut tool_calls, tail);
    let call_names = tool_calls
        .iter()
        .map(|call| call.name.clone())
        .collect::<Vec<_>>();
    super::openai::validate_tool_choice_output(&context.constraint_options, &call_names)?;
    if saw_tool_call {
        finish_reason = "tool_calls";
    }
    Ok(CompletionParts {
        content: (!content.is_empty()).then_some(content),
        tool_calls,
        finish_reason,
        completion_tokens,
    })
}

fn collect_tool_parser_events(
    content: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    events: Vec<AssistantOutputEvent>,
) {
    for event in events {
        match event {
            AssistantOutputEvent::TextDelta(text) => content.push_str(&text),
            AssistantOutputEvent::ToolCall(call) => tool_calls.push(call),
        }
    }
}

fn anthropic_stop_reason(reason: &'static str) -> &'static str {
    match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        other => other,
    }
}

fn format_openai_sse_data<T: Serialize>(payload: &T) -> Bytes {
    let s = serde_json::to_string(payload).unwrap_or_else(|_| "{}".into());
    let mut buf = String::with_capacity(s.len() + 8);
    buf.push_str("data: ");
    buf.push_str(&s);
    buf.push_str("\n\n");
    Bytes::from(buf)
}

fn openai_role_frame(id: &str, model: &str, created: u64) -> Bytes {
    let role_chunk = OpenAiStreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![OpenAiStreamChoice {
            index: 0,
            delta: OpenAiDeltaRole {
                role: "assistant",
                content: String::new(),
            },
            finish_reason: None,
        }],
    };
    format_openai_sse_data(&role_chunk)
}

fn openai_event_frame(
    id: &str,
    model: &str,
    created: u64,
    event: &DiffusionGemmaGenerateEvent,
) -> Bytes {
    let chunk = OpenAiStreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![OpenAiStreamChoice {
            index: 0,
            delta: OpenAiDeltaContent {
                content: &event.text,
            },
            finish_reason: event.finish_reason,
        }],
    };
    format_openai_sse_data(&chunk)
}

fn openai_finish_frame(id: &str, model: &str, created: u64, finish_reason: &'static str) -> Bytes {
    let event = DiffusionGemmaGenerateEvent {
        token: 0,
        text: String::new(),
        finish_reason: Some(finish_reason),
    };
    openai_event_frame(id, model, created, &event)
}

fn openai_done_frame() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}

fn openai_tool_event_frames(
    id: &str,
    model: &str,
    created: u64,
    events: Vec<AssistantOutputEvent>,
    next_call_index: &mut usize,
    call_names: &mut Vec<String>,
) -> Vec<Bytes> {
    let mut frames = Vec::new();
    for event in events {
        match event {
            AssistantOutputEvent::TextDelta(text) => {
                let event = DiffusionGemmaGenerateEvent {
                    token: 0,
                    text,
                    finish_reason: None,
                };
                frames.push(openai_event_frame(id, model, created, &event));
            }
            AssistantOutputEvent::ToolCall(call) => {
                call_names.push(call.name.clone());
                let index = *next_call_index;
                *next_call_index += 1;
                let chunk = OpenAiStreamChunk {
                    id: id.to_owned(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.to_owned(),
                    choices: vec![OpenAiStreamChoice {
                        index: 0,
                        delta: OpenAiDeltaToolCalls {
                            tool_calls: vec![OpenAiDeltaToolCall {
                                index,
                                id: Some(call.id),
                                kind: Some("function"),
                                function: OpenAiDeltaFunctionCall {
                                    name: Some(call.name),
                                    arguments: String::new(),
                                },
                            }],
                        },
                        finish_reason: None,
                    }],
                };
                frames.push(format_openai_sse_data(&chunk));
                let arguments =
                    serde_json::to_string(&call.arguments).expect("tool arguments are JSON values");
                for fragment in utf8_fragments(&arguments, 64) {
                    let chunk = OpenAiStreamChunk {
                        id: id.to_owned(),
                        object: "chat.completion.chunk",
                        created,
                        model: model.to_owned(),
                        choices: vec![OpenAiStreamChoice {
                            index: 0,
                            delta: OpenAiDeltaToolCalls {
                                tool_calls: vec![OpenAiDeltaToolCall {
                                    index,
                                    id: None,
                                    kind: None,
                                    function: OpenAiDeltaFunctionCall {
                                        name: None,
                                        arguments: fragment.to_owned(),
                                    },
                                }],
                            },
                            finish_reason: None,
                        }],
                    };
                    frames.push(format_openai_sse_data(&chunk));
                }
            }
        }
    }
    frames
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

fn openai_error_frame(message: &str) -> Bytes {
    let payload = serde_json::json!({"error": {"message": message, "type": "internal_error"}});
    format_openai_sse_data(&payload)
}

#[cfg(test)]
fn openai_stream_frames(
    id: &str,
    model: &str,
    created: u64,
    events: Vec<DiffusionGemmaGenerateEvent>,
) -> Vec<Bytes> {
    let mut frames = Vec::with_capacity(events.len() + 2);
    frames.push(openai_role_frame(id, model, created));
    for event in events {
        frames.push(openai_event_frame(id, model, created, &event));
        if event.finish_reason.is_some() {
            break;
        }
    }
    frames.push(openai_done_frame());
    frames
}

fn format_anthropic_event(event_type: &str, payload: &serde_json::Value) -> Bytes {
    let mut buf = String::new();
    buf.push_str("event: ");
    buf.push_str(event_type);
    buf.push('\n');
    buf.push_str("data: ");
    buf.push_str(&serde_json::to_string(payload).unwrap_or_else(|_| "{}".into()));
    buf.push_str("\n\n");
    Bytes::from(buf)
}

fn anthropic_error_event(message: &str) -> Bytes {
    let payload = serde_json::json!({
        "type": "error",
        "error": {"message": message}
    });
    format_anthropic_event("error", &payload)
}

#[cfg(test)]
fn anthropic_stream_frames(
    id: &str,
    model: &str,
    input_tokens: u32,
    events: Vec<DiffusionGemmaGenerateEvent>,
) -> Vec<Bytes> {
    let mut frames = Vec::with_capacity(events.len() + 5);
    let start_payload = serde_json::json!({
        "type": "message_start",
        "message": {
            "id": id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}
        }
    });
    frames.push(format_anthropic_event("message_start", &start_payload));
    let block_start = serde_json::json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""}
    });
    frames.push(format_anthropic_event("content_block_start", &block_start));

    let mut output_tokens = 0_u32;
    let mut stop_reason = "end_turn";
    for event in events {
        if !diffusion_event_is_length_sentinel(&event) {
            output_tokens += 1;
            if !event.text.is_empty() {
                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": event.text}
                });
                frames.push(format_anthropic_event("content_block_delta", &delta));
            }
        }
        if let Some(reason) = event.finish_reason {
            stop_reason = anthropic_stop_reason(reason);
            break;
        }
    }

    let block_stop = serde_json::json!({"type": "content_block_stop", "index": 0});
    frames.push(format_anthropic_event("content_block_stop", &block_stop));
    let msg_delta = serde_json::json!({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
        "usage": {"output_tokens": output_tokens}
    });
    frames.push(format_anthropic_event("message_delta", &msg_delta));
    let msg_stop = serde_json::json!({"type": "message_stop"});
    frames.push(format_anthropic_event("message_stop", &msg_stop));
    frames
}

async fn prepare_openai_request(
    state: &DiffusionGemmaAppState,
    req: super::openai::ChatRequest,
    output_schema: Option<serde_json::Value>,
) -> std::result::Result<(PreparedOpenAiRequest, u32), Response> {
    let prepared_tools =
        match super::openai::prepare_tool_request(&req, state.tokenizer.tool_dialect()) {
            Ok(prepared) => prepared,
            Err(error) => {
                return Err((StatusCode::BAD_REQUEST, format!("{error:#}")).into_response());
            }
        };
    let original_messages = prepared_tools.as_ref().map(|_| req.messages.clone());
    let chat_template_kwargs = req.chat_template_kwargs;
    let (flat_messages, pixel_values, image_grid_thw) =
        match super::openai::expand_image_parts_in_messages(req.messages, &state.vision_input).await
        {
            Ok(t) => t,
            Err(e) => return Err(super::security::image_error_response(e)),
        };
    let prompt_ids_result = if let Some(prepared) = &prepared_tools {
        super::openai::build_agent_messages(
            original_messages
                .as_deref()
                .expect("captured for DiffusionGemma tool request"),
            &flat_messages,
        )
        .and_then(|messages| {
            let kwargs = super::openai::tool_template_kwargs(chat_template_kwargs, prepared)?;
            state
                .tokenizer
                .render_and_encode_tool_prompt(&messages, &kwargs)
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
            return Err((
                StatusCode::BAD_REQUEST,
                format!("chat template / tokenize: {e}"),
            )
                .into_response());
        }
    };
    let input_tokens = prompt_ids.len() as u32;
    let (image_token_id, _) = crate::core::server::vision::derive_image_token_and_merge(
        &state.vision_input,
        &state.tokenizer,
    );
    let constraint = match prepared_tools
        .as_ref()
        .and_then(|prepared| {
            prepared
                .constraint_options
                .as_ref()
                .map(|options| (prepared, options))
        })
        .map(|(prepared, options)| {
            if matches!(
                &options.choice,
                crate::core::constrained::ToolChoiceConstraint::Auto
            ) {
                if let Some(schema) = output_schema.as_ref() {
                    return state.tokenizer.compile_tool_or_json_constraint(
                        &prepared.definitions,
                        options,
                        schema,
                    );
                }
            }
            state
                .tokenizer
                .compile_tool_constraint(&prepared.definitions, options)
        })
        .transpose()
        .and_then(
            |tool_constraint| match (tool_constraint, output_schema.as_ref()) {
                (Some(constraint), _) => Ok(Some(constraint)),
                (None, Some(schema)) => state
                    .tokenizer
                    .compile_json_output_constraint(schema)
                    .map(Some),
                (None, None) => Ok(None),
            },
        ) {
        Ok(constraint) => constraint,
        Err(error) => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("compile DiffusionGemma output constraint: {error:#}"),
            )
                .into_response());
        }
    };
    let tool_context = prepared_tools.and_then(|prepared| {
        prepared
            .constraint_options
            .map(|constraint_options| ToolResponseContext {
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
    let skip_special_tokens = tool_context
        .as_ref()
        .map(|context| context.dialect.skip_special_tokens())
        .unwrap_or(true);
    Ok((
        PreparedOpenAiRequest {
            generation: PreparedRequest {
                prompt_ids,
                pixel_values,
                image_grid_thw: if image_grid_thw.is_empty() {
                    None
                } else {
                    Some(image_grid_thw)
                },
                image_token_id,
                constraint,
                skip_special_tokens,
            },
            tool_context,
        },
        input_tokens,
    ))
}

async fn prepare_anthropic_request(
    state: &DiffusionGemmaAppState,
    req: super::anthropic::MessagesRequest,
) -> std::result::Result<(PreparedRequest, u32), Response> {
    let decoded = match super::anthropic::decode_anthropic_messages(req.messages) {
        Ok(d) => d,
        Err(e) => {
            return Err((StatusCode::BAD_REQUEST, format!("image decode: {e}")).into_response());
        }
    };
    let (flat_messages, pixel_values, image_grid_thw) =
        match expand_decoded_messages(decoded, &state.vision_input) {
            Ok(t) => t,
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("image decode/preprocess: {e}"),
                )
                    .into_response());
            }
        };
    let prompt_ids = match render_and_encode(&state.tokenizer, &flat_messages, None) {
        Ok(ids) => ids,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("chat template / tokenize: {e}"),
            )
                .into_response());
        }
    };
    let input_tokens = prompt_ids.len() as u32;
    let (image_token_id, _) = crate::core::server::vision::derive_image_token_and_merge(
        &state.vision_input,
        &state.tokenizer,
    );
    Ok((
        PreparedRequest {
            prompt_ids,
            pixel_values,
            image_grid_thw: if image_grid_thw.is_empty() {
                None
            } else {
                Some(image_grid_thw)
            },
            image_token_id,
            constraint: None,
            skip_special_tokens: true,
        },
        input_tokens,
    ))
}

async fn generate_completion(
    state: DiffusionGemmaAppState,
    request: PreparedRequest,
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    default_finish: &'static str,
    tool_context: Option<ToolResponseContext>,
) -> std::result::Result<CompletionParts, CompletionError> {
    let lane_guard = state.lane.clone().enter().await?;
    state
        .runtime_usage
        .record_input_tokens(request.prompt_ids.len() as u64);
    tokio::task::spawn_blocking(move || -> std::result::Result<CompletionParts, String> {
        let _lane_guard = lane_guard;
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let mut events = Vec::new();
        run_generation_with_events(
            &model_guard,
            tokenizer,
            &state.generation_config,
            request,
            max_tokens,
            temperature,
            seed,
            &mut |event| {
                events.push(event);
                Ok(true)
            },
        )?;
        mlx::transforms::clear_cache();
        let completion = match tool_context {
            Some(context) => collect_tool_events(events, context, default_finish)
                .map_err(|error| format!("parse DiffusionGemma tool output: {error:#}"))?,
            None => collect_events(events, default_finish),
        };
        state
            .runtime_usage
            .record_output_tokens(u64::from(completion.completion_tokens));
        Ok(completion)
    })
    .await
    .map_err(|e| CompletionError::Internal(format!("join error: {e}")))?
    .map_err(CompletionError::Internal)
}

#[allow(clippy::too_many_arguments)]
fn run_generation_with_events(
    model: &DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    generation_config: &DiffusionGemmaGenerationConfig,
    request: PreparedRequest,
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    emit: crate::models::diffusion_gemma::DiffusionGemmaEventSink<'_>,
) -> std::result::Result<(), String> {
    let PreparedRequest {
        prompt_ids,
        pixel_values,
        image_grid_thw,
        image_token_id,
        constraint,
        skip_special_tokens,
    } = request;
    match (pixel_values.as_deref(), image_grid_thw.as_deref()) {
        (Some(pixel_values), Some(image_grid_thw)) => match constraint.as_ref() {
            Some(constraint) => {
                crate::models::diffusion_gemma::generate_image_text_with_events_constrained(
                    model,
                    tokenizer,
                    &prompt_ids,
                    pixel_values,
                    image_grid_thw,
                    image_token_id,
                    generation_config,
                    max_tokens,
                    temperature,
                    seed,
                    constraint,
                    skip_special_tokens,
                    emit,
                )
                .map_err(|e| e.to_string())
            }
            None => crate::models::diffusion_gemma::generate_image_text_with_events(
                model,
                tokenizer,
                &prompt_ids,
                pixel_values,
                image_grid_thw,
                image_token_id,
                generation_config,
                max_tokens,
                temperature,
                seed,
                emit,
            )
            .map_err(|e| e.to_string()),
        },
        (None, None) => match constraint.as_ref() {
            Some(constraint) => {
                crate::models::diffusion_gemma::generate_text_with_events_constrained(
                    model,
                    tokenizer,
                    &prompt_ids,
                    generation_config,
                    max_tokens,
                    temperature,
                    seed,
                    constraint,
                    skip_special_tokens,
                    emit,
                )
                .map_err(|e| e.to_string())
            }
            None => crate::models::diffusion_gemma::generate_text_with_events(
                model,
                tokenizer,
                &prompt_ids,
                generation_config,
                max_tokens,
                temperature,
                seed,
                emit,
            )
            .map_err(|e| e.to_string()),
        },
        (Some(_), None) | (None, Some(_)) => {
            Err("DiffusionGemma image request missing image tensors or grids".to_string())
        }
    }
}

async fn openai_stream_completion(
    state: DiffusionGemmaAppState,
    request: PreparedRequest,
    stream_request: OpenAiStreamRequest,
) -> Response {
    let OpenAiStreamRequest {
        max_tokens,
        temperature,
        seed,
        model_label,
        prompt_tokens,
        include_usage,
        tool_context,
    } = stream_request;
    let lane_guard = match state.lane.clone().enter().await {
        Ok(guard) => guard,
        Err(err) => return CompletionError::from(err).into_response(),
    };
    state
        .runtime_usage
        .record_input_tokens(request.prompt_ids.len() as u64);
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let id = gen_openai_id();
    let created = now_unix();

    tokio::task::spawn_blocking(move || {
        let _lane_guard = lane_guard;
        if tx
            .blocking_send(Ok(openai_role_frame(&id, &model_label, created)))
            .is_err()
        {
            return;
        }

        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        if let Some(context) = tool_context {
            let mut parser = match ToolCallParser::new_with_output_schema(
                context.dialect,
                uuid::Uuid::new_v4().simple().to_string(),
                &context.definitions,
                context.output_schema,
            ) {
                Ok(parser) => parser,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(openai_error_frame(&format!("{error:#}"))));
                    let _ = tx.blocking_send(Ok(openai_done_frame()));
                    return;
                }
            };
            let mut connected = true;
            let mut completion_tokens = 0_u32;
            let mut model_finish = "stop";
            let mut next_call_index = 0_usize;
            let mut call_names = Vec::new();
            let generation_result = {
                let mut emit = |event: DiffusionGemmaGenerateEvent| -> Result<bool> {
                    if !diffusion_event_is_length_sentinel(&event) {
                        completion_tokens = completion_tokens.saturating_add(1);
                        state.runtime_usage.record_output_tokens(1);
                        let events = parser.push(&event.text)?;
                        for frame in openai_tool_event_frames(
                            &id,
                            &model_label,
                            created,
                            events,
                            &mut next_call_index,
                            &mut call_names,
                        ) {
                            if tx.blocking_send(Ok(frame)).is_err() {
                                connected = false;
                                return Ok(false);
                            }
                        }
                    }
                    if let Some(reason) = event.finish_reason {
                        model_finish = reason;
                    }
                    Ok(true)
                };
                run_generation_with_events(
                    &model_guard,
                    tokenizer,
                    &state.generation_config,
                    request,
                    max_tokens,
                    temperature,
                    seed,
                    &mut emit,
                )
            };
            if !connected {
                return;
            }
            if let Err(message) = generation_result {
                let _ = tx.blocking_send(Ok(openai_error_frame(&message)));
                let _ = tx.blocking_send(Ok(openai_done_frame()));
                return;
            }
            let (tail, saw_tool_call) = match parser.finish() {
                Ok(result) => result,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(openai_error_frame(&format!("{error:#}"))));
                    let _ = tx.blocking_send(Ok(openai_done_frame()));
                    return;
                }
            };
            for frame in openai_tool_event_frames(
                &id,
                &model_label,
                created,
                tail,
                &mut next_call_index,
                &mut call_names,
            ) {
                if tx.blocking_send(Ok(frame)).is_err() {
                    return;
                }
            }
            if let Err(error) =
                super::openai::validate_tool_choice_output(&context.constraint_options, &call_names)
            {
                let _ = tx.blocking_send(Ok(openai_error_frame(&format!("{error:#}"))));
                let _ = tx.blocking_send(Ok(openai_done_frame()));
                return;
            }
            let finish_reason = if saw_tool_call {
                "tool_calls"
            } else {
                model_finish
            };
            if tx
                .blocking_send(Ok(openai_finish_frame(
                    &id,
                    &model_label,
                    created,
                    finish_reason,
                )))
                .is_err()
            {
                return;
            }
            if include_usage {
                let usage = OpenAiStreamUsageChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_label.clone(),
                    choices: Vec::new(),
                    usage: OpenAiUsage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens.saturating_add(completion_tokens),
                    },
                };
                if tx
                    .blocking_send(Ok(format_openai_sse_data(&usage)))
                    .is_err()
                {
                    return;
                }
            }
            let _ = tx.blocking_send(Ok(openai_done_frame()));
            return;
        }
        let mut connected = true;
        let mut saw_finish = false;
        let generation_result = {
            let mut emit = |event: DiffusionGemmaGenerateEvent| -> Result<bool> {
                if event.finish_reason.is_some() {
                    saw_finish = true;
                }
                if tx
                    .blocking_send(Ok(openai_event_frame(&id, &model_label, created, &event)))
                    .is_err()
                {
                    connected = false;
                    return Ok(false);
                }
                if !diffusion_event_is_length_sentinel(&event) {
                    state.runtime_usage.record_output_tokens(1);
                }
                Ok(true)
            };
            run_generation_with_events(
                &model_guard,
                tokenizer,
                &state.generation_config,
                request,
                max_tokens,
                temperature,
                seed,
                &mut emit,
            )
        };

        if !connected {
            return;
        }
        if let Err(message) = generation_result {
            let _ = tx.blocking_send(Ok(openai_error_frame(&message)));
            let _ = tx.blocking_send(Ok(openai_done_frame()));
            return;
        }
        if !saw_finish
            && tx
                .blocking_send(Ok(openai_finish_frame(&id, &model_label, created, "stop")))
                .is_err()
        {
            return;
        }
        let _ = tx.blocking_send(Ok(openai_done_frame()));
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

async fn responses_stream_completion(
    state: DiffusionGemmaAppState,
    request: PreparedRequest,
    stream_request: ResponsesStreamRequest,
) -> Response {
    let ResponsesStreamRequest {
        max_tokens,
        temperature,
        seed,
        prompt_tokens,
        tool_context,
        meta,
    } = stream_request;
    let lane_guard = match state.lane.clone().enter().await {
        Ok(guard) => guard,
        Err(error) => {
            return super::responses::error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "diffusion_lane_unavailable",
                match error {
                    DiffusionGemmaLaneError::Overloaded => {
                        "DiffusionGemma serial lane is overloaded".to_owned()
                    }
                    DiffusionGemmaLaneError::Closed => {
                        "DiffusionGemma serial lane is closed".to_owned()
                    }
                },
            );
        }
    };
    state
        .runtime_usage
        .record_input_tokens(request.prompt_ids.len() as u64);
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);

    tokio::task::spawn_blocking(move || {
        let _lane_guard = lane_guard;
        let mut formatter = super::responses::ResponsesStream::new(meta);
        if tx.blocking_send(Ok(formatter.created())).is_err() {
            return;
        }
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let mut parser = match tool_context
            .as_ref()
            .map(|context| {
                ToolCallParser::new_with_output_schema(
                    context.dialect,
                    uuid::Uuid::new_v4().simple().to_string(),
                    &context.definitions,
                    context.output_schema.clone(),
                )
            })
            .transpose()
        {
            Ok(parser) => parser,
            Err(error) => {
                let _ = tx.blocking_send(Ok(formatter.failed(format!("{error:#}"))));
                return;
            }
        };
        let mut call_names = Vec::new();
        let mut completion_tokens = 0_u32;
        let mut finish_reason = "stop";
        let mut connected = true;
        let generation_result = {
            let mut emit = |event: DiffusionGemmaGenerateEvent| -> Result<bool> {
                if !diffusion_event_is_length_sentinel(&event) {
                    completion_tokens = completion_tokens.saturating_add(1);
                    state.runtime_usage.record_output_tokens(1);
                    let frames = if let Some(parser) = parser.as_mut() {
                        let events = parser.push(&event.text)?;
                        call_names.extend(events.iter().filter_map(|event| match event {
                            AssistantOutputEvent::ToolCall(call) => Some(call.name.clone()),
                            AssistantOutputEvent::TextDelta(_) => None,
                        }));
                        let mut frames = Vec::new();
                        for event in events {
                            match event {
                                AssistantOutputEvent::TextDelta(text) => {
                                    frames.extend(formatter.text_delta(text));
                                }
                                AssistantOutputEvent::ToolCall(call) => {
                                    frames.extend(formatter.tool_call(call)?);
                                }
                            }
                        }
                        frames
                    } else {
                        formatter.text_delta(event.text.clone())
                    };
                    for frame in frames {
                        if tx.blocking_send(Ok(frame)).is_err() {
                            connected = false;
                            return Ok(false);
                        }
                    }
                }
                if let Some(reason) = event.finish_reason {
                    finish_reason = reason;
                }
                Ok(true)
            };
            run_generation_with_events(
                &model_guard,
                tokenizer,
                &state.generation_config,
                request,
                max_tokens,
                temperature,
                seed,
                &mut emit,
            )
        };
        if !connected {
            return;
        }
        if let Err(message) = generation_result {
            let _ = tx.blocking_send(Ok(formatter.failed(message)));
            return;
        }
        if let (Some(parser), Some(context)) = (parser, tool_context.as_ref()) {
            let (events, saw_tool_call) = match parser.finish() {
                Ok(result) => result,
                Err(error) => {
                    let _ = tx.blocking_send(Ok(formatter.failed(format!("{error:#}"))));
                    return;
                }
            };
            call_names.extend(events.iter().filter_map(|event| match event {
                AssistantOutputEvent::ToolCall(call) => Some(call.name.clone()),
                AssistantOutputEvent::TextDelta(_) => None,
            }));
            if let Err(error) =
                super::openai::validate_tool_choice_output(&context.constraint_options, &call_names)
            {
                let _ = tx.blocking_send(Ok(formatter.failed(format!("{error:#}"))));
                return;
            }
            for event in events {
                let frames = match event {
                    AssistantOutputEvent::TextDelta(text) => Ok(formatter.text_delta(text)),
                    AssistantOutputEvent::ToolCall(call) => formatter.tool_call(call),
                };
                let frames = match frames {
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
            }
            if saw_tool_call {
                finish_reason = "tool_calls";
            }
        }
        for frame in formatter.completed(
            finish_reason,
            super::responses::Usage::new(prompt_tokens, completion_tokens),
        ) {
            if tx.blocking_send(Ok(frame)).is_err() {
                return;
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .expect("valid Responses SSE response")
}

async fn anthropic_stream_completion(
    state: DiffusionGemmaAppState,
    request: PreparedRequest,
    max_tokens: usize,
    temperature: f32,
    model_label: String,
    input_tokens: u32,
) -> Response {
    let lane_guard = match state.lane.clone().enter().await {
        Ok(guard) => guard,
        Err(err) => return CompletionError::from(err).into_response(),
    };
    state
        .runtime_usage
        .record_input_tokens(u64::from(input_tokens));
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let id = gen_anthropic_id();

    tokio::task::spawn_blocking(move || {
        let _lane_guard = lane_guard;
        let start_payload = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model_label,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        });
        if tx
            .blocking_send(Ok(format_anthropic_event("message_start", &start_payload)))
            .is_err()
        {
            return;
        }
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        });
        if tx
            .blocking_send(Ok(format_anthropic_event(
                "content_block_start",
                &block_start,
            )))
            .is_err()
        {
            return;
        }

        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let mut connected = true;
        let mut output_tokens = 0_u32;
        let mut stop_reason = "end_turn";
        let generation_result = {
            let mut emit = |event: DiffusionGemmaGenerateEvent| -> Result<bool> {
                if !diffusion_event_is_length_sentinel(&event) {
                    output_tokens += 1;
                    if !event.text.is_empty() {
                        let delta = serde_json::json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": event.text}
                        });
                        if tx
                            .blocking_send(Ok(format_anthropic_event(
                                "content_block_delta",
                                &delta,
                            )))
                            .is_err()
                        {
                            connected = false;
                            return Ok(false);
                        }
                    }
                    state.runtime_usage.record_output_tokens(1);
                }
                if let Some(reason) = event.finish_reason {
                    stop_reason = anthropic_stop_reason(reason);
                }
                Ok(true)
            };
            run_generation_with_events(
                &model_guard,
                tokenizer,
                &state.generation_config,
                request,
                max_tokens,
                temperature,
                None,
                &mut emit,
            )
        };

        if !connected {
            return;
        }
        if let Err(message) = generation_result {
            let _ = tx.blocking_send(Ok(anthropic_error_event(&message)));
            return;
        }
        let block_stop = serde_json::json!({"type": "content_block_stop", "index": 0});
        if tx
            .blocking_send(Ok(format_anthropic_event(
                "content_block_stop",
                &block_stop,
            )))
            .is_err()
        {
            return;
        }
        let msg_delta = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        });
        if tx
            .blocking_send(Ok(format_anthropic_event("message_delta", &msg_delta)))
            .is_err()
        {
            return;
        }
        let msg_stop = serde_json::json!({"type": "message_stop"});
        let _ = tx.blocking_send(Ok(format_anthropic_event("message_stop", &msg_stop)));
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

pub async fn openai_chat_completions(
    State(state): State<DiffusionGemmaAppState>,
    payload: std::result::Result<
        Json<super::openai::ChatRequest>,
        axum::extract::rejection::JsonRejection,
    >,
) -> Response {
    let req = match payload {
        Ok(Json(req)) => req,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("invalid Chat Completions request: {}", error.body_text()),
            )
                .into_response();
        }
    };
    let stream = req.stream;
    let include_usage = req
        .stream_options
        .map(|options| options.include_usage)
        .unwrap_or(false);
    let max_tokens = req.max_tokens;
    let temperature = req.temperature.unwrap_or(0.0);
    let seed = req.seed;
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let (prepared, prompt_tokens) = match prepare_openai_request(&state, req, None).await {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let PreparedOpenAiRequest {
        generation,
        tool_context,
    } = prepared;
    if stream {
        return openai_stream_completion(
            state,
            generation,
            OpenAiStreamRequest {
                max_tokens,
                temperature,
                seed,
                model_label,
                prompt_tokens,
                include_usage,
                tool_context,
            },
        )
        .await;
    }

    let completion = match generate_completion(
        state,
        generation,
        max_tokens,
        temperature,
        seed,
        "stop",
        tool_context,
    )
    .await
    {
        Ok(c) => c,
        Err(err) => return err.into_response(),
    };
    let resp = OpenAiCompletionResponse {
        id: gen_openai_id(),
        object: "chat.completion",
        created: now_unix(),
        model: model_label,
        choices: vec![OpenAiCompletionChoice {
            index: 0,
            message: OpenAiCompletionMessage {
                role: "assistant",
                content: completion.content,
                tool_calls: completion
                    .tool_calls
                    .into_iter()
                    .map(|call| OpenAiCompletionToolCall {
                        id: call.id,
                        kind: "function",
                        function: OpenAiCompletionFunctionCall {
                            name: call.name,
                            arguments: serde_json::to_string(&call.arguments)
                                .expect("tool arguments are JSON values"),
                        },
                    })
                    .collect(),
            },
            finish_reason: completion.finish_reason,
        }],
        usage: OpenAiUsage {
            prompt_tokens,
            completion_tokens: completion.completion_tokens,
            total_tokens: prompt_tokens + completion.completion_tokens,
        },
    };
    Json(resp).into_response()
}

pub async fn openai_responses(
    State(state): State<DiffusionGemmaAppState>,
    payload: std::result::Result<
        Json<super::responses::ResponsesRequest>,
        axum::extract::rejection::JsonRejection,
    >,
) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return super::responses::error_response(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                format!("invalid Responses request: {}", error.body_text()),
            );
        }
    };
    let normalized = match request.normalize() {
        Ok(normalized) => normalized,
        Err(error) => {
            return super::responses::error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("{error:#}"),
            );
        }
    };
    let model_label = normalized
        .chat
        .model
        .clone()
        .unwrap_or_else(|| state.model_id.clone());
    let meta = super::responses::ResponseMeta::from_normalized(&normalized, model_label);
    let stream = normalized.chat.stream;
    let max_tokens = normalized.chat.max_tokens;
    let temperature = normalized.chat.temperature.unwrap_or(0.0);
    let seed = normalized.chat.seed;
    let output_schema = normalized.output_schema();
    let (prepared, prompt_tokens) =
        match prepare_openai_request(&state, normalized.chat, output_schema).await {
            Ok(result) => result,
            Err(response) => return response,
        };
    let PreparedOpenAiRequest {
        generation,
        tool_context,
    } = prepared;
    if stream {
        return responses_stream_completion(
            state,
            generation,
            ResponsesStreamRequest {
                max_tokens,
                temperature,
                seed,
                prompt_tokens,
                tool_context,
                meta,
            },
        )
        .await;
    }
    let completion = match generate_completion(
        state,
        generation,
        max_tokens,
        temperature,
        seed,
        "stop",
        tool_context,
    )
    .await
    {
        Ok(completion) => completion,
        Err(error) => {
            return super::responses::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                match error {
                    CompletionError::Overloaded => "DiffusionGemma lane overloaded".to_owned(),
                    CompletionError::Internal(message) => message,
                },
            );
        }
    };
    super::responses::unary_response(
        meta,
        prompt_tokens,
        super::responses::CollectedOutput {
            content: completion.content.unwrap_or_default(),
            tool_calls: completion.tool_calls,
            finish_reason: completion.finish_reason,
            completion_tokens: completion.completion_tokens,
        },
    )
}

pub async fn anthropic_messages(
    State(state): State<DiffusionGemmaAppState>,
    Json(req): Json<super::anthropic::MessagesRequest>,
) -> Response {
    let stream = req.stream;
    let max_tokens = req.max_tokens;
    let temperature = req.temperature.unwrap_or(0.0);
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let (prepared, input_tokens) = match prepare_anthropic_request(&state, req).await {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    if stream {
        return anthropic_stream_completion(
            state,
            prepared,
            max_tokens,
            temperature,
            model_label,
            input_tokens,
        )
        .await;
    }

    let completion =
        match generate_completion(state, prepared, max_tokens, temperature, None, "stop", None)
            .await
        {
            Ok(c) => c,
            Err(err) => return err.into_response(),
        };
    let stop_reason = anthropic_stop_reason(completion.finish_reason);
    let envelope = AnthropicMessageEnvelope {
        id: gen_anthropic_id(),
        kind: "message",
        role: "assistant",
        content: vec![AnthropicContentBlockText {
            kind: "text",
            text: completion.content.unwrap_or_default(),
        }],
        model: model_label,
        stop_reason: Some(stop_reason),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens,
            output_tokens: completion.completion_tokens,
        },
    };
    Json(envelope).into_response()
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_diffusion_gemma(
    model: DiffusionGemmaModel,
    tokenizer: Tokenizer,
    generation_config: DiffusionGemmaGenerationConfig,
    model_id: String,
    model_weight_bytes: usize,
    network_config: super::security::ServerNetworkConfig,
    vision_input: VisionInputConfig,
) -> Result<()> {
    let state = build_diffusion_gemma_app_state(
        model,
        tokenizer,
        generation_config,
        model_id,
        model_weight_bytes,
        vision_input,
    );
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(diffusion_gemma_healthz))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/responses", post(openai_responses))
        .route("/v1/messages", post(anthropic_messages))
        .with_state(state);

    super::security::serve_router(app, network_config, "ironmlx DiffusionGemma server").await
}

pub(crate) fn build_diffusion_gemma_app_state(
    model: DiffusionGemmaModel,
    tokenizer: Tokenizer,
    generation_config: DiffusionGemmaGenerationConfig,
    model_id: String,
    model_weight_bytes: usize,
    vision_input: VisionInputConfig,
) -> DiffusionGemmaAppState {
    DiffusionGemmaAppState {
        model: Arc::new(Mutex::new(model)),
        tokenizer: Arc::new(tokenizer),
        generation_config,
        model_id,
        model_weight_bytes,
        vision_input,
        lane: Arc::new(DiffusionGemmaLane::new(
            DEFAULT_DIFFUSION_GEMMA_QUEUE_CAPACITY,
        )),
        runtime_usage: Arc::new(crate::core::runtime_usage::ModelRuntimeUsageCounters::default()),
    }
}

async fn diffusion_gemma_healthz(State(state): State<DiffusionGemmaAppState>) -> Response {
    let stats = state.lane.stats();
    Json(DiffusionGemmaHealth {
        status: "ok",
        scheduler: "serial_block_diffusion",
        active_requests: stats.active_requests,
        queued_requests: stats.queued_requests,
        queue_capacity: stats.queue_capacity,
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tool() -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_owned(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                },
                "required": ["city"],
                "additionalProperties": false
            }),
            strict: None,
        }
    }

    #[tokio::test]
    async fn lane_tracks_active_queued_and_rejects_when_full() {
        let lane = Arc::new(DiffusionGemmaLane::new(1));
        let first = lane.clone().enter().await.expect("first request admitted");
        assert_eq!(
            lane.stats(),
            DiffusionGemmaLaneStats {
                active_requests: 1,
                queued_requests: 0,
                queue_capacity: 1,
            }
        );

        let second = {
            let lane = lane.clone();
            tokio::spawn(async move { lane.enter().await.expect("queued request admitted") })
        };

        for _ in 0..10 {
            if lane.stats().queued_requests == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(lane.stats().queued_requests, 1);
        assert!(matches!(
            lane.clone().enter().await,
            Err(DiffusionGemmaLaneError::Overloaded)
        ));

        drop(first);
        let second = second.await.expect("queued task joined");
        assert_eq!(lane.stats().active_requests, 1);
        drop(second);
        assert_eq!(
            lane.stats(),
            DiffusionGemmaLaneStats {
                active_requests: 0,
                queued_requests: 0,
                queue_capacity: 1,
            }
        );
    }

    #[tokio::test]
    async fn lane_releases_queued_count_when_waiter_is_cancelled() {
        let lane = Arc::new(DiffusionGemmaLane::new(1));
        let first = lane.clone().enter().await.expect("first request admitted");
        let waiter = {
            let lane = lane.clone();
            tokio::spawn(async move { lane.enter().await })
        };

        for _ in 0..10 {
            if lane.stats().queued_requests == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(lane.stats().queued_requests, 1);
        waiter.abort();
        let _ = waiter.await;
        assert_eq!(lane.stats().queued_requests, 0);
        drop(first);
        assert_eq!(lane.stats().active_requests, 0);
    }

    #[test]
    fn openai_stream_frames_emit_role_content_finish_and_done() {
        let events = vec![
            DiffusionGemmaGenerateEvent {
                token: 1,
                text: "hel".to_string(),
                finish_reason: None,
            },
            DiffusionGemmaGenerateEvent {
                token: 2,
                text: "lo".to_string(),
                finish_reason: Some("stop"),
            },
        ];

        let frames = openai_stream_frames("chatcmpl-test", "dg-test", 0, events);
        let rendered: Vec<String> = frames
            .into_iter()
            .map(|b| String::from_utf8(b.to_vec()).unwrap())
            .collect();

        assert!(rendered[0].contains("\"role\":\"assistant\""));
        assert!(rendered[1].contains("\"content\":\"hel\""));
        assert!(rendered[2].contains("\"content\":\"lo\""));
        assert!(rendered[2].contains("\"finish_reason\":\"stop\""));
        assert_eq!(rendered.last().unwrap(), "data: [DONE]\n\n");
    }

    #[test]
    fn anthropic_stream_frames_emit_protocol_sequence_and_usage() {
        let events = vec![
            DiffusionGemmaGenerateEvent {
                token: 1,
                text: "A".to_string(),
                finish_reason: None,
            },
            DiffusionGemmaGenerateEvent {
                token: 0,
                text: String::new(),
                finish_reason: Some("length"),
            },
        ];

        let frames = anthropic_stream_frames("msg_test", "dg-test", 3, events);
        let rendered: Vec<String> = frames
            .into_iter()
            .map(|b| String::from_utf8(b.to_vec()).unwrap())
            .collect();

        assert!(rendered[0].starts_with("event: message_start\ndata: "));
        assert!(rendered[1].starts_with("event: content_block_start\ndata: "));
        assert!(rendered[2].contains("\"type\":\"content_block_delta\""));
        assert!(rendered[2].contains("\"text\":\"A\""));
        assert!(rendered[3].starts_with("event: content_block_stop\ndata: "));
        assert!(rendered[4].contains("\"stop_reason\":\"max_tokens\""));
        assert!(rendered[4].contains("\"output_tokens\":1"));
        assert!(rendered[5].starts_with("event: message_stop\ndata: "));
    }

    #[test]
    fn collect_events_joins_text_and_uses_finish_reason() {
        let events = vec![
            DiffusionGemmaGenerateEvent {
                token: 1,
                text: "hel".to_string(),
                finish_reason: None,
            },
            DiffusionGemmaGenerateEvent {
                token: 2,
                text: "lo".to_string(),
                finish_reason: None,
            },
            DiffusionGemmaGenerateEvent {
                token: 0,
                text: String::new(),
                finish_reason: Some("length"),
            },
        ];

        let completion = collect_events(events, "stop");
        assert_eq!(completion.content.as_deref(), Some("hello"));
        assert_eq!(completion.finish_reason, "length");
        assert_eq!(completion.completion_tokens, 2);
    }

    #[test]
    fn collect_tool_events_validates_gemma_call_and_sets_tool_finish_reason() {
        let events = vec![
            DiffusionGemmaGenerateEvent {
                token: 256,
                text: "<|tool_call>call:get_weather{city:<|\"|>Tokyo".to_owned(),
                finish_reason: None,
            },
            DiffusionGemmaGenerateEvent {
                token: 257,
                text: "<|\"|>,days:2}<tool_call|>".to_owned(),
                finish_reason: None,
            },
            DiffusionGemmaGenerateEvent {
                token: 261,
                text: String::new(),
                finish_reason: Some("stop"),
            },
        ];
        let completion = collect_tool_events(
            events,
            ToolResponseContext {
                dialect: ToolDialect::Gemma,
                definitions: vec![weather_tool()],
                output_schema: None,
                constraint_options: ToolConstraintOptions {
                    choice: crate::core::constrained::ToolChoiceConstraint::Required,
                    allow_parallel_calls: false,
                },
            },
            "stop",
        )
        .unwrap();
        assert_eq!(completion.finish_reason, "tool_calls");
        assert_eq!(completion.content, None);
        assert_eq!(completion.completion_tokens, 3);
        assert_eq!(completion.tool_calls.len(), 1);
        assert_eq!(completion.tool_calls[0].name, "get_weather");
        assert_eq!(
            completion.tool_calls[0].arguments,
            serde_json::json!({"city": "Tokyo", "days": 2})
        );
    }

    #[test]
    fn openai_tool_frames_emit_name_then_json_arguments() {
        let mut next = 0;
        let mut names = Vec::new();
        let frames = openai_tool_event_frames(
            "chatcmpl-test",
            "diffusion-gemma",
            0,
            vec![AssistantOutputEvent::ToolCall(ToolCall {
                id: "call_test_0".to_owned(),
                name: "get_weather".to_owned(),
                arguments: serde_json::json!({"city": "东京", "days": 2}),
            })],
            &mut next,
            &mut names,
        );
        let rendered = frames
            .iter()
            .map(|frame| String::from_utf8(frame.to_vec()).unwrap())
            .collect::<String>();
        assert!(rendered.contains("\"name\":\"get_weather\""));
        assert!(rendered.contains("\\\"city\\\":\\\"东京\\\""));
        assert_eq!(next, 1);
        assert_eq!(names, vec!["get_weather"]);
    }

    #[test]
    fn anthropic_stop_reason_maps_openai_reasons() {
        assert_eq!(anthropic_stop_reason("stop"), "end_turn");
        assert_eq!(anthropic_stop_reason("length"), "max_tokens");
        assert_eq!(anthropic_stop_reason("other"), "other");
    }
}
