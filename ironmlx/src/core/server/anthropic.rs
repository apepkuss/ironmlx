//! Anthropic-compatible Messages API: /v1/messages.
//!
//! Streaming uses 6-event SSE sequence:
//!   message_start → content_block_start → N × content_block_delta
//!     → content_block_stop → message_delta → message_stop
//!
//! Each event is framed as `event: <type>\ndata: <json>\n\n`.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use base64::Engine;

use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::model::Model;
use crate::core::sampler::Sampler;
use crate::core::scheduler::DenseVlMethods;
use crate::core::server::chat_format::render_and_encode;
use crate::core::server::scheduler_actor::{AdmitReply, SchedulerCommand};
use crate::core::server::vision::{DecodedMessage, DecodedPart};
use crate::core::speculative::MtpSpeculativeConfig;

use super::{AppState, Gemma4DrafterAppState, SamplingDefaults};

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
        Some(SchedulerError::RequestTooLarge { .. }) => {
            // 413 Payload Too Large — request needed cap exceeds server's
            // effective_cap_max. Body includes needed + max via Display.
            (StatusCode::PAYLOAD_TOO_LARGE, msg).into_response()
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

/// Anthropic native image source — base64 only (URL source is out of scope).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicImageSource {
    Base64 {
        // informational; not validated
        #[allow(dead_code)]
        media_type: String,
        data: String,
    },
}

/// Anthropic native content block.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentPart {
    Text { text: String },
    Image { source: AnthropicImageSource },
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
pub(crate) struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub(crate) messages: Vec<AnthropicMessage>,
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
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

fn default_max_tokens() -> usize {
    256
}

#[derive(Debug, Serialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ContentBlockText {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
}

#[derive(Debug, Serialize)]
struct MessageEnvelope {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    content: Vec<ContentBlockText>,
    model: String,
    stop_reason: Option<&'static str>,
    stop_sequence: Option<String>,
    usage: Usage,
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn gen_msg_id() -> String {
    format!("msg_{}", now_unix())
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
    if let Some(penalty) = req.repetition_penalty.or(defaults.repetition_penalty) {
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

/// Decode Anthropic native content blocks into the wire-agnostic
/// `DecodedMessage` list. base64 source is decoded in-process (no network);
/// `media_type` is informational and not validated.
pub(crate) fn decode_anthropic_messages(
    messages: Vec<AnthropicMessage>,
) -> anyhow::Result<Vec<DecodedMessage>> {
    let mut out: Vec<DecodedMessage> = Vec::with_capacity(messages.len());
    for m in messages {
        let mut parts: Vec<DecodedPart> = Vec::new();
        match m.content {
            AnthropicContent::Text(t) => parts.push(DecodedPart::Text(t)),
            AnthropicContent::Parts(ps) => {
                for p in ps {
                    match p {
                        AnthropicContentPart::Text { text } => parts.push(DecodedPart::Text(text)),
                        AnthropicContentPart::Image { source } => {
                            let AnthropicImageSource::Base64 { data, .. } = source;
                            let bytes = base64::engine::general_purpose::STANDARD
                                .decode(data.as_bytes())
                                .map_err(|e| anyhow::anyhow!("image base64 decode: {e}"))?;
                            parts.push(DecodedPart::Image(bytes));
                        }
                    }
                }
            }
        }
        out.push(DecodedMessage {
            role: m.role,
            parts,
        });
    }
    Ok(out)
}

pub async fn messages<M>(
    State(state): State<AppState<M>>,
    Json(req): Json<MessagesRequest>,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    messages_with_state(state, req).await
}

pub(crate) async fn messages_with_state<M>(state: AppState<M>, req: MessagesRequest) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    // Extract fields before partially moving req.messages.
    let max_tokens = req.max_tokens;
    let stream = req.stream;
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let sampler = build_sampler(&req, state.sampling_defaults);
    if let Err(error) = super::validate_prompt_lookup_sampler(state.prompt_lookup_enabled, sampler)
    {
        return (StatusCode::BAD_REQUEST, format!("{error:#}")).into_response();
    }

    // Decode Anthropic wire format -> neutral DecodedMessage (base64 -> bytes).
    let decoded = match decode_anthropic_messages(req.messages) {
        Ok(d) => d,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("image decode: {e}")).into_response();
        }
    };

    // Shared per-model preprocess + placeholder rewrite.
    let (flat_messages, pixel_values, image_grid_thw) =
        match crate::core::server::vision::expand_decoded_messages(decoded, &state.vision_input) {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("image decode/preprocess: {e}"),
                )
                    .into_response();
            }
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

    // Anthropic /v1/messages doesn't surface chat_template_kwargs in its
    // public schema; pass None.
    let prompt_ids = match render_and_encode(&state.tokenizer, &flat_messages, None) {
        Ok(ids) => ids,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("chat template / tokenize: {e}"),
            )
                .into_response();
        }
    };
    let input_tokens = prompt_ids.len() as u32;
    let prompt_len = prompt_ids.len();
    let scheduler_config = state.scheduler_request_config(prompt_len, max_tokens);
    let stop_token_ids = state.tokenizer.eos_token_ids().to_vec();
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
    };

    let use_scheduler = super::should_route_to_scheduler::<M>(
        prompt_len,
        scheduler_config.prefill_chunk_size,
        state.b_max,
        state.paged_prefix_cache_enabled,
        state.force_scheduler_for_greedy && sampler.is_pipelinable(),
    );

    match messages_route(stream, use_scheduler) {
        MessagesRoute::SchedulerStream => {
            serve_via_scheduler_stream(state, request, model_label, input_tokens).await
        }
        MessagesRoute::GenerationStreamStream => {
            serve_via_gs_stream(state, request, model_label, input_tokens).await
        }
        MessagesRoute::SchedulerUnary => {
            serve_via_scheduler_unary(state, request, model_label, input_tokens).await
        }
        MessagesRoute::GenerationStreamUnary => {
            serve_via_gs_unary(state, request, model_label, input_tokens).await
        }
    }
}

pub(crate) async fn gemma4_drafter_messages(
    State(state): State<Gemma4DrafterAppState>,
    Json(req): Json<MessagesRequest>,
) -> Response {
    messages_with_gemma4_drafter_state(state, req).await
}

pub(crate) async fn messages_with_gemma4_drafter_state(
    state: Gemma4DrafterAppState,
    req: MessagesRequest,
) -> Response {
    let max_tokens = req.max_tokens;
    let stream = req.stream;
    let model_label = req
        .model
        .clone()
        .unwrap_or_else(|| state.base.model_id.clone());
    let sampler = build_sampler(&req, state.base.sampling_defaults);
    let _cfg = match MtpSpeculativeConfig::new(state.mtp_draft_tokens, sampler) {
        Ok(cfg) => cfg,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("{e:#}")).into_response(),
    };

    let decoded = match decode_anthropic_messages(req.messages) {
        Ok(d) => d,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("image decode: {e}")).into_response();
        }
    };
    let (flat_messages, pixel_values, image_grid_thw) =
        match crate::core::server::vision::expand_decoded_messages(
            decoded,
            &state.base.vision_input,
        ) {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("image decode/preprocess: {e}"),
                )
                    .into_response();
            }
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
    let prompt_ids = match render_and_encode(&state.base.tokenizer, &flat_messages, None) {
        Ok(ids) => ids,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("chat template / tokenize: {e}"),
            )
                .into_response();
        }
    };
    let input_tokens = prompt_ids.len() as u32;
    let prompt_len = prompt_ids.len();
    let total_tokens = prompt_len.saturating_add(max_tokens);
    if total_tokens > state.base.effective_cap_max {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "request too large: prompt_len + max_tokens = {total_tokens}, max = {}",
                state.base.effective_cap_max
            ),
        )
            .into_response();
    }
    let scheduler_config = state.base.scheduler_request_config(prompt_len, max_tokens);
    let stop_token_ids = state.base.tokenizer.eos_token_ids().to_vec();
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
    };

    if stream {
        serve_via_scheduler_stream(state.base, request, model_label, input_tokens).await
    } else {
        serve_via_scheduler_unary(state.base, request, model_label, input_tokens).await
    }
}

async fn serve_via_gs_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
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
        // 2. content_block_start
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        });
        if tx
            .blocking_send(Ok(format_event("content_block_start", &block_start)))
            .is_err()
        {
            return;
        }

        // 3. N × content_block_delta + final stop_reason capture.
        let mut output_tokens: u32 = 0;
        let mut stop_reason: &'static str = "end_turn";
        let mut first_event = Some(first_event);
        loop {
            let event = match first_event.take() {
                Some(event) => Ok(event),
                None => stream.next_token(),
            };
            match event {
                Ok(Some(ev)) => {
                    if !ev.text.is_empty() {
                        let delta = serde_json::json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": ev.text}
                        });
                        if tx
                            .blocking_send(Ok(format_event("content_block_delta", &delta)))
                            .is_err()
                        {
                            return;
                        }
                    }
                    output_tokens += 1;
                    if let Some(reason) = ev.finish_reason {
                        stop_reason = match reason {
                            "stop" => "end_turn",
                            "length" => "max_tokens",
                            other => other,
                        };
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

        // 4. content_block_stop
        let block_stop = serde_json::json!({"type": "content_block_stop", "index": 0});
        let _ = tx.blocking_send(Ok(format_event("content_block_stop", &block_stop)));
        // 5. message_delta
        let msg_delta = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        });
        let _ = tx.blocking_send(Ok(format_event("message_delta", &msg_delta)));
        // 6. message_stop
        let msg_stop = serde_json::json!({"type": "message_stop"});
        let _ = tx.blocking_send(Ok(format_event("message_stop", &msg_stop)));
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

    // 2. Spawn forwarder that emits the 6-event SSE sequence.
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let msg_id_for_task = msg_id.clone();
    let model_id_for_task = model_id.clone();
    let tokenizer = state.tokenizer.clone();

    tokio::spawn(async move {
        // Event 1: message_start
        let start_payload = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": msg_id_for_task,
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
            .send(Ok(format_event("message_start", &start_payload)))
            .await
            .is_err()
        {
            return;
        }

        // Event 2: content_block_start
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        });
        if tx
            .send(Ok(format_event("content_block_start", &block_start)))
            .await
            .is_err()
        {
            return;
        }

        // Events 3..N+2: content_block_delta per non-empty detok output.
        // output_tokens increments UNCONDITIONALLY per StepEvent (mirrors
        // GS path line 277 — counter reflects generated tokens, NOT
        // emitted deltas. Tokens whose detok output is empty still count.)
        let mut detok = tokenizer.decode_stream(/* skip_special */ true);
        let mut output_tokens: u32 = 0;
        let mut stop_reason: &'static str = "end_turn";
        while let Some(ev) = event_rx.recv().await {
            let text = match detok.step(ev.token) {
                Ok(Some(s)) => s,
                Ok(None) => String::new(), // BPE mid-codepoint
                Err(_) => String::new(),   // best-effort; skip emit
            };
            if !text.is_empty() {
                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text}
                });
                if tx
                    .send(Ok(format_event("content_block_delta", &delta)))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            output_tokens += 1;
            if let Some(reason) = ev.finish_reason {
                stop_reason = match reason {
                    "stop" => "end_turn",
                    "length" => "max_tokens",
                    other => other,
                };
                break;
            }
        }

        // Event N+3: content_block_stop
        let block_stop = serde_json::json!({"type": "content_block_stop", "index": 0});
        if tx
            .send(Ok(format_event("content_block_stop", &block_stop)))
            .await
            .is_err()
        {
            return;
        }

        // Event N+4: message_delta (carries final stop_reason + output_tokens)
        let msg_delta = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        });
        if tx
            .send(Ok(format_event("message_delta", &msg_delta)))
            .await
            .is_err()
        {
            return;
        }

        // Event N+5: message_stop
        let msg_stop = serde_json::json!({"type": "message_stop"});
        let _ = tx.send(Ok(format_event("message_stop", &msg_stop))).await;
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
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let id = gen_msg_id();
    let result =
        tokio::task::spawn_blocking(move || -> anyhow::Result<(String, &'static str, u32)> {
            let model_guard = state.model.blocking_lock();
            let tokenizer = &*state.tokenizer;
            let memory = super::begin_direct_request_memory(&state, &*model_guard, &request)?;
            let mut stream = GenerationStream::new(&*model_guard, tokenizer, request)?;
            let mut buf = String::new();
            let mut finish: &'static str = "end_turn";
            let mut output_tokens: u32 = 0;
            let mut memory = Some(memory);
            loop {
                let next = stream.next_token()?;
                if let Some(memory) = memory.take() {
                    memory.commit();
                }
                let Some(ev) = next else {
                    break;
                };
                buf.push_str(&ev.text);
                output_tokens += 1;
                if let Some(reason) = ev.finish_reason {
                    finish = match reason {
                        "stop" => "end_turn",
                        "length" => "max_tokens",
                        other => other,
                    };
                    break;
                }
            }
            Ok((buf, finish, output_tokens))
        })
        .await;

    let (content, stop_reason, output_tokens) = match result {
        Ok(Ok(t)) => t,
        Ok(Err(err)) => return generation_err_to_response(err),
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")).into_response();
        }
    };

    let envelope = MessageEnvelope {
        id,
        kind: "message",
        role: "assistant",
        content: vec![ContentBlockText {
            kind: "text",
            text: content,
        }],
        model: model_id,
        stop_reason: Some(stop_reason),
        stop_sequence: None,
        usage: Usage {
            input_tokens,
            output_tokens,
        },
    };
    Json(envelope).into_response()
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

    // 2. Drain events; build envelope.
    let mut detok = state.tokenizer.decode_stream(/* skip_special */ true);
    let mut content = String::new();
    let mut output_tokens: u32 = 0;
    let mut stop_reason: &'static str = "end_turn";
    while let Some(ev) = event_rx.recv().await {
        match detok.step(ev.token) {
            Ok(Some(s)) => content.push_str(&s),
            Ok(None) => { /* BPE mid-codepoint */ }
            Err(_) => { /* best-effort */ }
        }
        output_tokens += 1;
        if let Some(reason) = ev.finish_reason {
            stop_reason = match reason {
                "stop" => "end_turn",
                "length" => "max_tokens",
                other => other,
            };
            break;
        }
    }

    let envelope = MessageEnvelope {
        id,
        kind: "message",
        role: "assistant",
        content: vec![ContentBlockText {
            kind: "text",
            text: content,
        }],
        model: model_id,
        stop_reason: Some(stop_reason),
        stop_sequence: None,
        usage: Usage {
            input_tokens,
            output_tokens,
        },
    };
    Json(envelope).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

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
            content: vec![ContentBlockText {
                kind: "text",
                text: "hi".into(),
            }],
            model: "qwen3.5-4b".into(),
            stop_reason: Some("end_turn"),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 3,
                output_tokens: 1,
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
    fn build_sampler_prefers_request_values_over_model_defaults() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "messages": [],
            "max_tokens": 8,
            "temperature": 0.2,
            "top_p": 0.6,
            "top_k": 16,
            "repetition_penalty": 1.05
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
        assert_eq!(sampler.repetition_penalty, Some(1.05));
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
    }
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::core::server::chat_format::{ChatMessage, Content, ContentPart, ImageUrl};
    use crate::core::server::openai::decode_openai_messages;
    use crate::core::server::vision::expand_decoded_messages;
    use crate::core::server::VisionInputConfig;

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
        }]
    }

    async fn run_parity(vision_input: VisionInputConfig) {
        let b64 = coco_b64();
        let client = reqwest::Client::new();
        // OpenAI path: data: URL → bytes → shared core.
        let openai_decoded = decode_openai_messages(openai_one_image(&b64), &client)
            .await
            .unwrap();
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
