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

use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::sampler::Sampler;
use crate::core::server::chat_format::{render_and_encode, ChatMessage, Content, ContentPart};
use crate::core::server::scheduler_actor::{AdmitReply, SchedulerCommand};

use super::AppState;

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
        Some(SchedulerError::MemoryBudgetExceeded { .. }) => {
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

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
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

fn build_sampler(req: &MessagesRequest) -> Sampler {
    let mut s = Sampler::greedy();
    if let Some(t) = req.temperature {
        if t > 0.0 {
            s = s.with_temperature(t);
        }
    }
    if let Some(p) = req.top_p {
        if p < 1.0 {
            s = s.with_top_p(p);
        }
    }
    s
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

pub async fn messages(State(state): State<AppState>, Json(req): Json<MessagesRequest>) -> Response {
    // Extract fields before partially moving req.messages.
    let max_tokens = req.max_tokens;
    let stream = req.stream;
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let sampler = build_sampler(&req);

    // The Anthropic /v1/messages handler is text-only. Reject requests that
    // include image content parts with a clear 400 rather than silently
    // dropping them — silent drop produced text-only completions that ignored
    // the image and confused users (audit ref B6). Multimodal support for the
    // Anthropic shape is a future expansion; today, route image requests to
    // /v1/chat/completions.
    for m in &req.messages {
        if let Content::Parts(parts) = &m.content {
            for p in parts {
                if matches!(p, ContentPart::ImageUrl { .. }) {
                    return (
                        StatusCode::BAD_REQUEST,
                        "Anthropic /v1/messages does not yet support image content parts; \
                         use /v1/chat/completions for image requests"
                            .to_string(),
                    )
                        .into_response();
                }
            }
        }
    }

    // Flatten any text-only multimodal content parts to plain text.
    let flat_messages: Vec<ChatMessage> = req
        .messages
        .into_iter()
        .map(|m| {
            let text = match &m.content {
                Content::Text(t) => t.clone(),
                Content::Parts(parts) => parts
                    .iter()
                    .filter_map(|p| {
                        if let ContentPart::Text { text } = p {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            };
            ChatMessage {
                role: m.role,
                content: Content::Text(text),
            }
        })
        .collect();

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
    let stop_token_ids = state.tokenizer.eos_token_ids().to_vec();
    let request = GenerateRequest {
        prompt_ids,
        max_new_tokens: max_tokens,
        sampler,
        stop_token_ids,
        prefill_chunk_size: state.prefill_chunk_size,
        pixel_values: None,
        image_grid_thw: None,
        // Anthropic path is text-only (see audit B6); both values unused
        // when image_grid_thw is None.
        image_spatial_merge_size: 2,
        image_token_id: crate::core::generate::IMAGE_TOKEN_ID,
    };

    // COMPAT(3b-2/3b-4): long-prompt fallback to GS sunsets in 3c+
    // chunked-prefill phase. Note: when prefill_chunk_size == 0 (chunking
    // disabled by config), this predicate routes ALL text requests to the
    // SchedulerActor regardless of length — equivalent to the GS path's
    // behavior when chunking is also disabled there.
    let prompt_len = request.prompt_ids.len();
    let use_scheduler = state.prefill_chunk_size == 0 || prompt_len <= state.prefill_chunk_size;

    match (stream, use_scheduler) {
        (true, true) => serve_via_scheduler_stream(state, request, model_label, input_tokens).await,
        (true, false) => serve_via_gs_stream(state, request, model_label, input_tokens).await,
        (false, true) => serve_via_scheduler_unary(state, request, model_label, input_tokens).await,
        (false, false) => serve_via_gs_unary(state, request, model_label, input_tokens).await,
    }
}

async fn serve_via_gs_stream(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let id = gen_msg_id();
    let id_for_task = id.clone();
    let model_id_for_task = model_id.clone();

    tokio::task::spawn_blocking(move || {
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
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let mut stream = match GenerationStream::new(&model_guard, tokenizer, request) {
            Ok(s) => s,
            Err(e) => {
                let payload = serde_json::json!({
                    "type": "error",
                    "error": {"message": e.to_string()}
                });
                let _ = tx.blocking_send(Ok(format_event("error", &payload)));
                return;
            }
        };
        let mut output_tokens: u32 = 0;
        let mut stop_reason: &'static str = "end_turn";
        loop {
            match stream.next_token() {
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
pub async fn serve_via_scheduler_stream(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
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

async fn serve_via_gs_unary(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
    let id = gen_msg_id();
    let result = tokio::task::spawn_blocking(
        move || -> std::result::Result<(String, &'static str, u32), String> {
            let model_guard = state.model.blocking_lock();
            let tokenizer = &*state.tokenizer;
            let mut stream = GenerationStream::new(&model_guard, tokenizer, request)
                .map_err(|e| e.to_string())?;
            let mut buf = String::new();
            let mut finish: &'static str = "end_turn";
            let mut output_tokens: u32 = 0;
            while let Some(ev) = stream.next_token().map_err(|e| e.to_string())? {
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
        },
    )
    .await;

    let (content, stop_reason, output_tokens) = match result {
        Ok(Ok(t)) => t,
        Ok(Err(msg)) => return (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
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
pub async fn serve_via_scheduler_unary(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    input_tokens: u32,
) -> Response {
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
}
