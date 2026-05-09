//! OpenAI-compatible Chat Completions API: /v1/chat/completions.
//!
//! Supports both streaming (`stream: true` → SSE) and non-streaming
//! (`stream: false` → JSON).

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
use crate::core::server::chat_format::{render_and_encode, ChatMessage};

use super::AppState;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// HuggingFace `apply_chat_template` extra kwargs — passed through as
    /// top-level template render-context variables. Honors Qwen3+'s
    /// `enable_thinking` toggle, vLLM's `tools` / `documents`, etc.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
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
struct CompletionMessage {
    role: &'static str,
    content: String,
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

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn gen_id() -> String {
    format!("chatcmpl-{}", now_unix())
}

fn build_sampler(req: &ChatRequest) -> Sampler {
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
    if let Some(seed) = req.seed {
        s = s.with_seed(seed);
    }
    s
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let prompt_ids = match render_and_encode(
        &state.tokenizer,
        &req.messages,
        req.chat_template_kwargs.as_ref(),
    ) {
        Ok(ids) => ids,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("chat template / tokenize: {e}"),
            )
                .into_response();
        }
    };
    let sampler = build_sampler(&req);
    let stop_token_ids = state.tokenizer.eos_token_ids().to_vec();
    let request = GenerateRequest {
        prompt_ids,
        max_new_tokens: req.max_tokens,
        sampler,
        stop_token_ids,
        prefill_chunk_size: state.prefill_chunk_size,
    };

    let prompt_tokens = request.prompt_ids.len() as u32;
    let model_id = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    if req.stream {
        chat_completions_stream(state, request, model_id).await
    } else {
        chat_completions_unary(state, request, model_id, prompt_tokens).await
    }
}

async fn chat_completions_stream(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
) -> Response {
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let id = gen_id();
    let id_for_task = id.clone();
    let model_id_for_task = model_id.clone();

    tokio::task::spawn_blocking(move || {
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;
        let mut stream = match GenerationStream::new(&model_guard, tokenizer, request) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.blocking_send(Ok(format_sse_error(&e)));
                return;
            }
        };

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
        if tx.blocking_send(Ok(format_sse_data(&role_chunk))).is_err() {
            return;
        }

        loop {
            match stream.next_token() {
                Ok(Some(ev)) => {
                    let chunk = ChunkResponse {
                        id: id_for_task.clone(),
                        object: "chat.completion.chunk",
                        created: now_unix(),
                        model: model_id_for_task.clone(),
                        choices: vec![Choice {
                            index: 0,
                            delta: DeltaContent { content: &ev.text },
                            finish_reason: ev.finish_reason,
                        }],
                    };
                    if tx.blocking_send(Ok(format_sse_data(&chunk))).is_err() {
                        break;
                    }
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
        let _ = tx.blocking_send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
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

async fn chat_completions_unary(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
) -> Response {
    let id = gen_id();
    let result = tokio::task::spawn_blocking(
        move || -> std::result::Result<(String, &'static str, u32), String> {
            let model_guard = state.model.blocking_lock();
            let tokenizer = &*state.tokenizer;
            let mut stream = GenerationStream::new(&model_guard, tokenizer, request)
                .map_err(|e| e.to_string())?;
            let mut buf = String::new();
            let mut finish: &'static str = "stop";
            let mut completion_tokens: u32 = 0;
            while let Some(ev) = stream.next_token().map_err(|e| e.to_string())? {
                buf.push_str(&ev.text);
                completion_tokens += 1;
                if let Some(reason) = ev.finish_reason {
                    finish = reason;
                    break;
                }
            }
            Ok((buf, finish, completion_tokens))
        },
    )
    .await;

    let (content, finish, completion_tokens) = match result {
        Ok(Ok(t)) => t,
        Ok(Err(msg)) => return (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("join error: {e}"),
            )
                .into_response();
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
                content,
            },
            finish_reason: finish,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
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
    fn sse_data_format_includes_prefix_and_double_newline() {
        let payload = serde_json::json!({"a": 1});
        let bytes = format_sse_data(&payload);
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(s.starts_with("data: "), "missing prefix: {s:?}");
        assert!(s.ends_with("\n\n"), "missing terminator: {s:?}");
        assert!(s.contains("\"a\":1"), "payload not embedded: {s:?}");
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
                    content: "hi".into(),
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
}
