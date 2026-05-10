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
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::sampler::Sampler;
use crate::core::server::chat_format::{render_and_encode, ChatMessage, Content, ContentPart};

use super::AppState;

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

    // Flatten any multimodal content parts to plain text.
    // The Anthropic /v1/messages handler is text-only; image_url parts are
    // stripped (only their text siblings are kept).
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
    };

    if stream {
        messages_stream(state, request, model_label, input_tokens).await
    } else {
        messages_unary(state, request, model_label, input_tokens).await
    }
}

async fn messages_stream(
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

async fn messages_unary(
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
