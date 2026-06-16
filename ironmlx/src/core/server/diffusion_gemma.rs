//! DiffusionGemma HTTP server lane.
//!
//! This module is intentionally separate from the causal `AppState<M>` /
//! `SchedulerActor` server path. DiffusionGemma is a block-diffusion model, so
//! requests are admitted through a bounded serial lane and completed as either
//! unary responses or SSE streams of committed block-diffusion output.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
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

use crate::core::server::chat_format::render_and_encode;
use crate::core::server::vision::expand_decoded_messages;
use crate::core::server::VisionInputConfig;
use crate::core::tokenizer::Tokenizer;
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
    pub vision_input: VisionInputConfig,
    pub lane: Arc<DiffusionGemmaLane>,
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
}

struct CompletionParts {
    content: String,
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
    content: String,
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
struct OpenAiDeltaRole {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAiDeltaContent<'a> {
    content: &'a str,
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
        content,
        finish_reason,
        completion_tokens,
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
) -> std::result::Result<(PreparedRequest, u32), Response> {
    let http_client = reqwest::Client::new();
    let (flat_messages, pixel_values, image_grid_thw) =
        match super::openai::expand_image_parts_in_messages(
            req.messages,
            &http_client,
            &state.vision_input,
        )
        .await
        {
            Ok(t) => t,
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("image decode/preprocess: {e}"),
                )
                    .into_response());
            }
        };
    let prompt_ids = match render_and_encode(
        &state.tokenizer,
        &flat_messages,
        req.chat_template_kwargs.as_ref(),
    ) {
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
        },
        input_tokens,
    ))
}

async fn generate_completion(
    state: DiffusionGemmaAppState,
    request: PreparedRequest,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
    default_finish: &'static str,
) -> std::result::Result<CompletionParts, CompletionError> {
    let lane_guard = state.lane.clone().enter().await?;
    tokio::task::spawn_blocking(move || -> std::result::Result<CompletionParts, String> {
        let _lane_guard = lane_guard;
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let events = match (
            request.pixel_values.as_deref(),
            request.image_grid_thw.as_deref(),
        ) {
            (Some(pixel_values), Some(image_grid_thw)) => {
                crate::models::diffusion_gemma::generate_image_text(
                    &model_guard,
                    tokenizer,
                    &request.prompt_ids,
                    pixel_values,
                    image_grid_thw,
                    request.image_token_id,
                    &state.generation_config,
                    max_tokens,
                    temperature,
                    seed,
                )
                .map_err(|e| e.to_string())?
            }
            (None, None) => crate::models::diffusion_gemma::generate_text(
                &model_guard,
                tokenizer,
                &request.prompt_ids,
                &state.generation_config,
                max_tokens,
                temperature,
                seed,
            )
            .map_err(|e| e.to_string())?,
            (Some(_), None) | (None, Some(_)) => {
                return Err(
                    "DiffusionGemma image request missing image tensors or grids".to_string(),
                );
            }
        };
        Ok(collect_events(events, default_finish))
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
    seed: u64,
    emit: crate::models::diffusion_gemma::DiffusionGemmaEventSink<'_>,
) -> std::result::Result<(), String> {
    match (
        request.pixel_values.as_deref(),
        request.image_grid_thw.as_deref(),
    ) {
        (Some(pixel_values), Some(image_grid_thw)) => {
            crate::models::diffusion_gemma::generate_image_text_with_events(
                model,
                tokenizer,
                &request.prompt_ids,
                pixel_values,
                image_grid_thw,
                request.image_token_id,
                generation_config,
                max_tokens,
                temperature,
                seed,
                emit,
            )
            .map_err(|e| e.to_string())
        }
        (None, None) => crate::models::diffusion_gemma::generate_text_with_events(
            model,
            tokenizer,
            &request.prompt_ids,
            generation_config,
            max_tokens,
            temperature,
            seed,
            emit,
        )
        .map_err(|e| e.to_string()),
        (Some(_), None) | (None, Some(_)) => {
            Err("DiffusionGemma image request missing image tensors or grids".to_string())
        }
    }
}

async fn openai_stream_completion(
    state: DiffusionGemmaAppState,
    request: PreparedRequest,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
    model_label: String,
) -> Response {
    let lane_guard = match state.lane.clone().enter().await {
        Ok(guard) => guard,
        Err(err) => return CompletionError::from(err).into_response(),
    };
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
                0,
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
    Json(req): Json<super::openai::ChatRequest>,
) -> Response {
    let stream = req.stream;
    let max_tokens = req.max_tokens;
    let temperature = req.temperature.unwrap_or(0.0);
    let seed = req.seed.unwrap_or(0);
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let (prepared, prompt_tokens) = match prepare_openai_request(&state, req).await {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    if stream {
        return openai_stream_completion(
            state,
            prepared,
            max_tokens,
            temperature,
            seed,
            model_label,
        )
        .await;
    }

    let completion =
        match generate_completion(state, prepared, max_tokens, temperature, seed, "stop").await {
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
        match generate_completion(state, prepared, max_tokens, temperature, 0, "stop").await {
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
            text: completion.content,
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

pub async fn serve_diffusion_gemma(
    model: DiffusionGemmaModel,
    tokenizer: Tokenizer,
    generation_config: DiffusionGemmaGenerationConfig,
    model_id: String,
    host: &str,
    port: u16,
    vision_input: VisionInputConfig,
) -> Result<()> {
    let state = DiffusionGemmaAppState {
        model: Arc::new(Mutex::new(model)),
        tokenizer: Arc::new(tokenizer),
        generation_config,
        model_id,
        vision_input,
        lane: Arc::new(DiffusionGemmaLane::new(
            DEFAULT_DIFFUSION_GEMMA_QUEUE_CAPACITY,
        )),
    };
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(diffusion_gemma_healthz))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("parsing socket addr {host}:{port}"))?;
    tracing::info!("ironmlx DiffusionGemma server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    axum::serve(listener, app).await?;
    Ok(())
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
        assert_eq!(completion.content, "hello");
        assert_eq!(completion.finish_reason, "length");
        assert_eq!(completion.completion_tokens, 2);
    }

    #[test]
    fn anthropic_stop_reason_maps_openai_reasons() {
        assert_eq!(anthropic_stop_reason("stop"), "end_turn");
        assert_eq!(anthropic_stop_reason("length"), "max_tokens");
        assert_eq!(anthropic_stop_reason("other"), "other");
    }
}
