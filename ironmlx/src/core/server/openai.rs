//! OpenAI-compatible Chat Completions API: /v1/chat/completions.
//!
//! Supports both streaming (`stream: true` → SSE) and non-streaming
//! (`stream: false` → JSON).

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::Engine;
use mlx::ops::shape::concatenate;
use mlx::Array;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use tokio::sync::oneshot;

use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::sampler::Sampler;
use crate::core::server::chat_format::{render_and_encode, ChatMessage, Content, ContentPart};
use crate::core::server::scheduler_actor::{AdmitReply, SchedulerCommand};
use crate::models::qwen3_5::image_processor;

use super::AppState;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a SchedulerActor admit Err into an HTTP response. Spec §4.7:
/// "admission queue full" → 503 + Retry-After: 5; everything else → 400.
fn admit_err_to_response(err: anyhow::Error) -> Response {
    use axum::http::HeaderValue;
    let msg = format!("{err:#}");
    if msg.contains("admission queue full") {
        let mut resp = (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
        resp.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
        resp
    } else {
        (StatusCode::BAD_REQUEST, msg).into_response()
    }
}

// ---------------------------------------------------------------------------
// Request / Response shapes
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Image URL decoding (Step 19.1 + 19.2)
// ---------------------------------------------------------------------------

/// Decode an image URL to raw bytes.
///
/// Supports:
/// - `data:<mime>;base64,<b64>` — decoded in-process (no network)
/// - `http://` / `https://` — fetched via the provided async `reqwest::Client`
pub async fn decode_image_url(url: &str, client: &reqwest::Client) -> anyhow::Result<Vec<u8>> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (_meta, b64) = rest
            .split_once(',')
            .ok_or_else(|| anyhow::anyhow!("malformed data URL — missing ','"))?;
        let bytes = base64::engine::general_purpose::STANDARD.decode(b64)?;
        Ok(bytes)
    } else if url.starts_with("http://") || url.starts_with("https://") {
        let resp = client.get(url).send().await?;
        let bytes = resp.bytes().await?.to_vec();
        Ok(bytes)
    } else {
        anyhow::bail!("unsupported image_url scheme: {url}")
    }
}

// ---------------------------------------------------------------------------
// Multimodal message expansion (Step 18.4 + 19.3 helper)
// ---------------------------------------------------------------------------

/// Walk `messages`, decode + preprocess every `image_url` content part, and
/// rewrite the messages so all `Content::Parts` are converted to
/// `Content::Text` with vision token placeholder strings inserted.
///
/// Returns:
/// - rewritten text-only messages (ready for `render_and_encode`)
/// - concatenated pixel_values Array (None when no images present)
/// - image_grid_thw list (one entry per image)
pub async fn expand_image_parts_in_messages(
    messages: Vec<ChatMessage>,
    client: &reqwest::Client,
    spatial_merge_size: i32,
) -> anyhow::Result<(Vec<ChatMessage>, Option<Array>, Vec<(i32, i32, i32)>)> {
    if spatial_merge_size <= 0 {
        return Err(anyhow::anyhow!(
            "expand_image_parts_in_messages: spatial_merge_size must be > 0 (got {spatial_merge_size})"
        ));
    }
    let mut all_pixel_values: Vec<Array> = Vec::new();
    let mut grid_thw: Vec<(i32, i32, i32)> = Vec::new();

    // First pass: collect pixel_values + grid info for every image_url part
    // across all messages, in order.
    for msg in &messages {
        if let Content::Parts(parts) = &msg.content {
            for part in parts {
                if let ContentPart::ImageUrl { image_url } = part {
                    let img_bytes = decode_image_url(&image_url.url, client).await?;
                    let (pv, gh, gw) = image_processor::preprocess(&img_bytes)?;
                    all_pixel_values.push(pv);
                    grid_thw.push((1, gh, gw));
                }
            }
        }
    }

    // Build per-message image token counts ((gh/m) * (gw/m)) in the same
    // order images were collected, where m = spatial_merge_size.
    let token_counts: Vec<usize> = grid_thw
        .iter()
        .map(|&(_, gh, gw)| ((gh / spatial_merge_size) * (gw / spatial_merge_size)) as usize)
        .collect();
    let mut counts_deque: VecDeque<usize> = VecDeque::from(token_counts);

    // Second pass: rewrite messages to plain-text with placeholder tokens.
    let flat_messages: Vec<ChatMessage> = messages
        .into_iter()
        .map(|msg| {
            let flat = msg.content.to_flat_string(&mut counts_deque);
            ChatMessage {
                role: msg.role,
                content: Content::Text(flat),
            }
        })
        .collect();

    // Concatenate pixel_values along axis 0.
    let pixel_values = if all_pixel_values.is_empty() {
        None
    } else {
        let refs: Vec<&Array> = all_pixel_values.iter().collect();
        let concat = concatenate(&refs, 0)?;
        // Eagerly materialize on this (async tokio worker) thread before the
        // tensor crosses into spawn_blocking, where a different worker thread's
        // default MLX stream would not be able to evaluate this thread's lazy
        // graph (errors with "There is no Stream(gpu, N) in current thread").
        mlx::transforms::eval(&[&concat])?;
        Some(concat)
    };

    Ok((flat_messages, pixel_values, grid_thw))
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

// ---------------------------------------------------------------------------
// Handler (Step 19.3)
// ---------------------------------------------------------------------------

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    // Extract fields we need after consuming req.messages.
    let stream = req.stream;
    let max_tokens = req.max_tokens;
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let sampler = build_sampler(&req);
    let chat_template_kwargs = req.chat_template_kwargs;

    // Build a per-request reqwest client for image fetching.
    // For text-only requests this is a cheap no-op (no images to fetch).
    let http_client = reqwest::Client::new();

    // Read VisionConfig.spatial_merge_size from the model so multi-image
    // token-count math + MRoPE VL position-id stride pick up whatever the
    // loaded checkpoint actually uses. Default `2` for text-only models or
    // VL models without an explicit vision_config (matches Qwen3.5-VL).
    let spatial_merge_size: i32 = state
        .model
        .lock()
        .await
        .config()
        .vision_config
        .as_ref()
        .map(|vc| vc.spatial_merge_size)
        .unwrap_or(2);

    // Resolve `<|image_pad|>` to its tokenizer id, so VL routing works for
    // sibling Qwen-family models with different special-token ids. Falls back
    // to the Qwen3.5-VL default constant if the token is absent (text-only
    // tokenizer or otherwise non-VL model — in that case image_grid_thw will
    // also be empty so the id is unused).
    let image_token_id: i32 = state
        .tokenizer
        .token_to_id("<|image_pad|>")
        .map(|id| id as i32)
        .unwrap_or(crate::core::generate::IMAGE_TOKEN_ID);

    // Expand multimodal content parts: decode images, build pixel_values,
    // rewrite messages to text-with-placeholder.
    let (flat_messages, pixel_values, image_grid_thw) = match expand_image_parts_in_messages(
        req.messages,
        &http_client,
        spatial_merge_size,
    )
    .await
    {
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

    let prompt_ids = match render_and_encode(
        &state.tokenizer,
        &flat_messages,
        chat_template_kwargs.as_ref(),
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
    let stop_token_ids = state.tokenizer.eos_token_ids().to_vec();
    let request = GenerateRequest {
        prompt_ids,
        max_new_tokens: max_tokens,
        sampler,
        stop_token_ids,
        prefill_chunk_size: state.prefill_chunk_size,
        pixel_values,
        image_grid_thw: image_grid_thw_opt,
        image_spatial_merge_size: spatial_merge_size,
        image_token_id,
    };

    let prompt_tokens = request.prompt_ids.len() as u32;

    // Routing: short-prompt → SchedulerActor; long-prompt → GenerationStream.
    // B1-p2.4: VL fallback removed — VL requests now route through Scheduler
    // via Scheduler::admit/admit_mid + batched_prefill_vl.
    // COMPAT(3b-2): long-prompt fallback to GS sunsets in 3c+ chunked-prefill phase.
    let prompt_len = request.prompt_ids.len();
    let use_scheduler = state.prefill_chunk_size == 0 || prompt_len <= state.prefill_chunk_size;

    match (stream, use_scheduler) {
        (true, true) => serve_via_scheduler_stream(state, request, model_label).await,
        (true, false) => serve_via_gs_stream(state, request, model_label).await,
        (false, true) => {
            serve_via_scheduler_unary(state, request, model_label, prompt_tokens).await
        }
        (false, false) => serve_via_gs_unary(state, request, model_label, prompt_tokens).await,
    }
}

async fn serve_via_gs_stream(
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

/// Text-only short-prompt SSE path via SchedulerActor (3b-2 swap-in).
async fn serve_via_scheduler_stream(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
) -> Response {
    let id = gen_id();

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

    // 2. Stream events as SSE. Spawn a forwarder task that detokenizes
    // per-event and pushes formatted SSE chunks to a bounded channel.
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
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
        if tx.send(Ok(format_sse_data(&role_chunk))).await.is_err() {
            return;
        }

        let mut detok = tokenizer.decode_stream(/* skip_special */ true);
        while let Some(ev) = event_rx.recv().await {
            let text = match detok.step(ev.token) {
                Ok(Some(s)) => s,
                Ok(None) => String::new(),
                Err(e) => {
                    let _ = tx
                        .send(Ok(format_sse_error(&anyhow::anyhow!("detok: {e}"))))
                        .await;
                    break;
                }
            };
            let chunk = ChunkResponse {
                id: id_for_task.clone(),
                object: "chat.completion.chunk",
                created: now_unix(),
                model: model_id_for_task.clone(),
                choices: vec![Choice {
                    index: 0,
                    delta: DeltaContent { content: &text },
                    finish_reason: ev.finish_reason,
                }],
            };
            if tx.send(Ok(format_sse_data(&chunk))).await.is_err() {
                break;
            }
            if ev.finish_reason.is_some() {
                break;
            }
        }
        let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
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

/// Text-only short-prompt unary path via SchedulerActor (3b-2 swap-in).
async fn serve_via_scheduler_unary(
    state: AppState,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
) -> Response {
    let id = gen_id();

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

    // 2. Collect all events, detokenize, build CompletionResponse.
    let mut detok = state.tokenizer.decode_stream(/* skip_special */ true);
    let mut content = String::new();
    let mut finish: &'static str = "stop";
    let mut completion_tokens: u32 = 0;
    while let Some(ev) = event_rx.recv().await {
        completion_tokens += 1;
        match detok.step(ev.token) {
            Ok(Some(s)) => content.push_str(&s),
            Ok(None) => { /* BPE mid-codepoint */ }
            Err(e) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, format!("detok: {e}")).into_response();
            }
        }
        if let Some(reason) = ev.finish_reason {
            finish = reason;
            break;
        }
    }

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

    #[tokio::test]
    async fn data_url_decoded_to_bytes() {
        // "/9j/4AAQABAA" is a truncated JPEG header (base64):
        // 0xff 0xd8 0xff 0xe0 0x00 0x10 0x00 0x10 0x00
        let url = "data:image/jpeg;base64,/9j/4AAQABAA";
        let bytes = decode_image_url(url, &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(
            bytes,
            vec![0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x00, 0x10, 0x00]
        );
    }

    #[tokio::test]
    async fn admit_err_503_for_queue_full() {
        let err = anyhow::anyhow!("admission queue full: capacity=32 reached");
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
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("admission queue full"));
    }

    #[tokio::test]
    async fn admit_err_400_for_other() {
        let err = anyhow::anyhow!("prompt too long: 999999 tokens exceeds limit");
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(resp.headers().get("retry-after").is_none());
    }
}
