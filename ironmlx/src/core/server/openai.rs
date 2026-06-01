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
use base64::Engine;
use mlx::Array;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use tokio::sync::oneshot;

use crate::core::generate::{GenerateRequest, GenerationStream};
use crate::core::model::Model;
use crate::core::sampler::Sampler;
use crate::core::scheduler::DenseVlMethods;
#[cfg(not(feature = "p5h-profile"))]
use crate::core::server::chat_format::render_and_encode;
use crate::core::server::chat_format::{ChatMessage, Content, ContentPart};
use crate::core::server::scheduler_actor::{AdmitReply, SchedulerCommand};
use crate::core::server::VisionInputConfig;
use crate::models::{gemma4, qwen3_5};

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
/// - per-image pixel_values tensors (None when no images present)
/// - image_grid_thw list (one entry per image)
pub async fn expand_image_parts_in_messages(
    messages: Vec<ChatMessage>,
    client: &reqwest::Client,
    vision_input: &VisionInputConfig,
) -> anyhow::Result<(Vec<ChatMessage>, Option<Vec<Array>>, Vec<(i32, i32, i32)>)> {
    let spatial_merge_size = match vision_input {
        VisionInputConfig::Qwen { spatial_merge_size } => *spatial_merge_size,
        VisionInputConfig::Gemma4 { vision_config } => vision_config.pooling_kernel_size,
    };
    if spatial_merge_size <= 0 {
        return Err(anyhow::anyhow!(
            "expand_image_parts_in_messages: spatial_merge_size must be > 0 (got {spatial_merge_size})"
        ));
    }
    let mut all_pixel_values: Vec<Array> = Vec::new();
    let mut grid_thw: Vec<(i32, i32, i32)> = Vec::new();
    let mut placeholders: Vec<String> = Vec::new();

    // First pass: collect pixel_values + grid info for every image_url part
    // across all messages, in order.
    for msg in &messages {
        if let Content::Parts(parts) = &msg.content {
            for part in parts {
                if let ContentPart::ImageUrl { image_url } = part {
                    let img_bytes = decode_image_url(&image_url.url, client).await?;
                    match vision_input {
                        VisionInputConfig::Qwen { .. } => {
                            let (pv, gh, gw) = qwen3_5::image_processor::preprocess(&img_bytes)?;
                            let n =
                                ((gh / spatial_merge_size) * (gw / spatial_merge_size)) as usize;
                            placeholders.push(qwen_placeholder(n));
                            all_pixel_values.push(pv);
                            grid_thw.push((1, gh, gw));
                        }
                        VisionInputConfig::Gemma4 { vision_config } => {
                            let processed =
                                gemma4::image_processor::preprocess(&img_bytes, vision_config)?;
                            placeholders.push(gemma4_placeholder(processed.soft_tokens));
                            grid_thw.push((1, processed.grid_h, processed.grid_w));
                            all_pixel_values.push(processed.pixel_values);
                        }
                    }
                }
            }
        }
    }

    // Second pass: rewrite messages to plain-text with placeholder tokens.
    let mut placeholders = placeholders.into_iter();
    let flat_messages: Vec<ChatMessage> = messages
        .into_iter()
        .map(|msg| {
            let flat = flatten_content_with_placeholders(msg.content, &mut placeholders);
            ChatMessage {
                role: msg.role,
                content: Content::Text(flat),
            }
        })
        .collect();

    let pixel_values = if all_pixel_values.is_empty() {
        None
    } else {
        // Eagerly materialize on this (async tokio worker) thread before the
        // tensor crosses into spawn_blocking, where a different worker thread's
        // default MLX stream would not be able to evaluate this thread's lazy
        // graph (errors with "There is no Stream(gpu, N) in current thread").
        for pv in &all_pixel_values {
            mlx::transforms::eval(&[pv])?;
        }
        Some(all_pixel_values)
    };

    Ok((flat_messages, pixel_values, grid_thw))
}

fn qwen_placeholder(n: usize) -> String {
    let mut s = String::from("<|vision_start|>");
    for _ in 0..n {
        s.push_str("<|image_pad|>");
    }
    s.push_str("<|vision_end|>");
    s
}

fn gemma4_placeholder(n: usize) -> String {
    let mut s = String::from("<|image>");
    for _ in 0..n {
        s.push_str("<|image|>");
    }
    s.push_str("<image|>");
    s
}

fn flatten_content_with_placeholders(
    content: Content,
    placeholders: &mut impl Iterator<Item = String>,
) -> String {
    match content {
        Content::Text(t) => t,
        Content::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => out.push_str(&text),
                    ContentPart::ImageUrl { .. } => {
                        out.push_str(&placeholders.next().unwrap_or_default());
                    }
                }
            }
            out
        }
    }
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

pub async fn chat_completions<M>(
    State(state): State<AppState<M>>,
    Json(req): Json<ChatRequest>,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    // Extract fields we need after consuming req.messages.
    let stream = req.stream;

    // P5h root + http_parse_render_tokenize start capture (per spec § 2.5a step 1).
    // Both timestamps captured at handler entry BEFORE any parse/tokenize work,
    // because the http_parse_render_tokenize span's true start is the entry point,
    // and the root span needs the same anchor.
    // Per Codex plan review v16 P1 #2 + v17 P1 #1: only capture timestamps if the
    // request will be served by a streaming path — non-streaming has no root
    // terminal. Reuse the existing `let stream = req.stream;` local; do NOT
    // introduce a parallel `p5h_stream_enabled` derivation.
    #[cfg(feature = "p5h-profile")]
    let (p5h_request_id, p5h_root_start_ns, p5h_http_start_ns) = if stream {
        (
            uuid::Uuid::new_v4().to_string(),
            crate::core::p5h::monotonic_ns_public(),
            crate::core::p5h::monotonic_ns_public(),
        )
    } else {
        // Sentinel: empty request_id signals "no P5h state for this request".
        // Step 3 + Step 4 below conditionally skip when this is empty.
        (String::new(), 0, 0)
    };

    let max_tokens = req.max_tokens;
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let sampler = build_sampler(&req);
    let chat_template_kwargs = req.chat_template_kwargs;

    // Build a per-request reqwest client for image fetching.
    // For text-only requests this is a cheap no-op (no images to fetch).
    let http_client = reqwest::Client::new();

    let (image_token_id, spatial_merge_size) = match &state.vision_input {
        VisionInputConfig::Qwen { spatial_merge_size } => (
            state
                .tokenizer
                .token_to_id("<|image_pad|>")
                .map(|id| id as i32)
                .unwrap_or(crate::core::generate::IMAGE_TOKEN_ID),
            *spatial_merge_size,
        ),
        VisionInputConfig::Gemma4 { vision_config } => (
            state
                .tokenizer
                .token_to_id("<|image|>")
                .map(|id| id as i32)
                .unwrap_or(258_880),
            vision_config.pooling_kernel_size,
        ),
    };

    // Expand multimodal content parts: decode images, build pixel_values,
    // rewrite messages to text-with-placeholder.
    let (flat_messages, pixel_values, image_grid_thw) =
        match expand_image_parts_in_messages(req.messages, &http_client, &state.vision_input).await
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

    // T4.4: under p5h-profile, capture encode start/end timestamps so the
    // openai handler can retroactively open a `tokenizer_encode` child span
    // under `http_parse_render_tokenize` (which itself was opened at the
    // handler-entry timestamp captured before the ctx existed). The non-
    // profile path uses the original `render_and_encode` signature.
    #[cfg(feature = "p5h-profile")]
    let (prompt_ids, p5h_encode_start_ns, p5h_encode_end_ns) =
        match crate::core::server::chat_format::render_and_encode_with_encode_timing(
            &state.tokenizer,
            &flat_messages,
            chat_template_kwargs.as_ref(),
        ) {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("chat template / tokenize: {e}"),
                )
                    .into_response();
            }
        };
    #[cfg(not(feature = "p5h-profile"))]
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

    let prompt_len = prompt_ids.len();

    // Routing: short-prompt and model-limited chunked long-prompt requests
    // use SchedulerActor; other chunked long prompts keep using GenerationStream.
    // B1-p2.4: VL fallback removed — VL requests now route through Scheduler
    // via Scheduler::admit/admit_mid + batched_prefill_vl.
    let use_scheduler =
        super::should_route_to_scheduler::<M>(prompt_len, state.prefill_chunk_size, state.b_max);

    // Per Codex plan review v16 P1 #2 + v17 P1 #1 + v18 P1 #1: p5h state ONLY
    // for streaming requests. Reuse the existing `stream` local from Step 2
    // (which comes from the handler's `let stream = req.stream;` extraction
    // around `openai.rs:318`) — do NOT introduce a parallel
    // `p5h_stream_enabled`. Wrap the entire state-building block in
    // `Option<(P5hTraceContext, SpanHandle)>` so the p5h_trace/p5h_root_span
    // fields of GenerateRequest in Step 4 can be populated unconditionally
    // (Some(...) for streaming, None for unary), and the
    // http_parse_render_tokenize emission only fires on the streaming branch.
    #[cfg(feature = "p5h-profile")]
    let p5h_state: Option<(
        crate::core::p5h::P5hTraceContext,
        crate::core::p5h::SpanHandle,
    )> = if stream {
        let p5h_routing_path: &'static str = if use_scheduler {
            "scheduler"
        } else {
            "gs_chunked"
        };

        let p5h_ctx = crate::core::p5h::P5hTraceContext {
            request_id: p5h_request_id.clone(),
            prompt_tokens: prompt_len as u32,
            routing_path: p5h_routing_path,
        };

        let p5h_root_span = crate::core::p5h::open_p5h_span_at(
            &p5h_ctx,
            None,
            "server_request_recv_to_first_content_sse_write",
            p5h_root_start_ns,
        );

        // Per Codex plan review v10 P1 #3: `RootSpanHandle::new(...)` was here
        // in earlier drafts but `chat_completions` itself never used it — each
        // `serve_via_*` constructs its own RootSpanHandle from
        // `request.p5h_root_span` after dispatch (T0a.6 Step 4.5 pre-move clone).
        // Constructing a handle here would be unused → `clippy -D warnings`
        // rejects. Just keep `p5h_ctx` + `p5h_root_span` as plain values for use
        // by the http_parse_render_tokenize emission below + the GenerateRequest
        // population in Step 4.

        let http_span = crate::core::p5h::open_p5h_span_at(
            &p5h_ctx,
            Some(&p5h_root_span),
            "http_parse_render_tokenize",
            p5h_http_start_ns,
        );

        // T4.4: retroactively open + close a `tokenizer_encode` child span
        // under `http_parse_render_tokenize`. Encode start/end timestamps
        // were captured inside `render_and_encode_with_encode_timing` above
        // — at that point the `P5hTraceContext` did not yet exist (it's
        // built from `prompt_len`, which is the encode result), so the
        // child span has to be opened/closed retroactively here using the
        // captured timestamps. Pattern matches the
        // `detok_format_first_content_chunk` retroactive open at the
        // streaming SSE first-content site (openai.rs:968).
        let encode_span = crate::core::p5h::open_p5h_span_at(
            &p5h_ctx,
            Some(&http_span),
            "tokenizer_encode",
            p5h_encode_start_ns,
        );
        crate::core::p5h::close_p5h_span(
            &p5h_ctx,
            encode_span,
            p5h_encode_end_ns,
            crate::core::p5h::SpanFields::default(),
        );

        crate::core::p5h::close_p5h_span(
            &p5h_ctx,
            http_span,
            crate::core::p5h::monotonic_ns_public(),
            crate::core::p5h::SpanFields::default(),
        );

        Some((p5h_ctx, p5h_root_span))
    } else {
        None
    };

    let stop_token_ids = state.tokenizer.eos_token_ids().to_vec();
    let prompt_tokens = prompt_len as u32;
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
        #[cfg(feature = "p5h-profile")]
        p5h_trace: p5h_state.as_ref().map(|(ctx, _)| ctx.clone()),
        #[cfg(feature = "p5h-profile")]
        p5h_root_span: p5h_state.as_ref().map(|(_, span)| span.clone()),
    };

    match (stream, use_scheduler) {
        (true, true) => serve_via_scheduler_stream(state, request, model_label).await,
        (true, false) => serve_via_gs_stream(state, request, model_label).await,
        (false, true) => {
            serve_via_scheduler_unary(state, request, model_label, prompt_tokens).await
        }
        (false, false) => serve_via_gs_unary(state, request, model_label, prompt_tokens).await,
    }
}

async fn serve_via_gs_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    // Pre-move clones (per Codex plan review v8 P1 #2). After this block we
    // can read p5h state via these locals even after `request` is moved into
    // spawn_blocking.
    #[cfg(feature = "p5h-profile")]
    let p5h_ctx_for_closure = request
        .p5h_trace
        .clone()
        .expect("p5h-profile: GenerateRequest.p5h_trace not populated by handler");
    #[cfg(feature = "p5h-profile")]
    let p5h_root_handle_for_closure = crate::core::p5h::RootSpanHandle::new(
        p5h_ctx_for_closure.clone(),
        request
            .p5h_root_span
            .clone()
            .expect("p5h-profile: GenerateRequest.p5h_root_span not populated by handler"),
    );
    #[cfg(feature = "p5h-profile")]
    let p5h_response_request_id = p5h_ctx_for_closure.request_id.clone();

    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let id = gen_id();
    let id_for_task = id.clone();
    let model_id_for_task = model_id.clone();

    // T0a.8 Step 1: aliases consumed by the spawn_blocking closure. These are
    // clones of the pre-move locals above; the originals stay on the async
    // task so the response header builder can still read p5h_response_request_id.
    #[cfg(feature = "p5h-profile")]
    let p5h_ctx: crate::core::p5h::P5hTraceContext = p5h_ctx_for_closure.clone();
    #[cfg(feature = "p5h-profile")]
    let p5h_root_handle_gs: crate::core::p5h::RootSpanHandle = p5h_root_handle_for_closure.clone();

    tokio::task::spawn_blocking(move || {
        let model_guard = state.model.blocking_lock();
        let tokenizer = &*state.tokenizer;

        #[cfg(feature = "p5h-profile")]
        let mut root_guard = crate::core::p5h::P5hRootCloseGuard::new(p5h_root_handle_gs);

        // T0a.8 Step 2: wrap GenerationStream::new in gs_stream_init_and_chunk_loop
        // so deep spans inside (gs_kv_cache_alloc / gs_chunk_N /
        // gs_first_token_sample_dispatch) chain under it via the trace guard.
        #[cfg(feature = "p5h-profile")]
        let gs_top_span = crate::core::p5h::open_p5h_span(
            &p5h_ctx,
            Some(root_guard.span()),
            "gs_stream_init_and_chunk_loop",
        );

        #[cfg(feature = "p5h-profile")]
        let _gs_guard =
            crate::core::p5h::P5hTraceGuard::enter(p5h_ctx.clone(), gs_top_span.clone());

        let stream_result = GenerationStream::new(&*model_guard, tokenizer, request);

        #[cfg(feature = "p5h-profile")]
        drop(_gs_guard);
        #[cfg(feature = "p5h-profile")]
        let gs_close_end_ns = crate::core::p5h::monotonic_ns_public();
        #[cfg(feature = "p5h-profile")]
        crate::core::p5h::close_p5h_span(
            &p5h_ctx,
            gs_top_span,
            gs_close_end_ns,
            crate::core::p5h::SpanFields::default(),
        );

        let mut stream = match stream_result {
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

        // T0a.8 Step 5 (role): wrap the role-chunk send in sse_write_role_chunk.
        #[cfg(feature = "p5h-profile")]
        let role_span = crate::core::p5h::open_p5h_span(
            &p5h_ctx,
            Some(root_guard.span()),
            "sse_write_role_chunk",
        );

        let role_send_result = tx.blocking_send(Ok(format_sse_data(&role_chunk)));

        #[cfg(feature = "p5h-profile")]
        let role_close_end_ns = crate::core::p5h::monotonic_ns_public();
        #[cfg(feature = "p5h-profile")]
        crate::core::p5h::close_p5h_span(
            &p5h_ctx,
            role_span,
            role_close_end_ns,
            crate::core::p5h::SpanFields::default(),
        );

        if role_send_result.is_err() {
            return;
        }

        // T0a.8 Step 4: per-iteration span. First iteration emits
        // `gs_first_token_materialize_and_predispatch`; subsequent (while
        // root is still open) emit `pre_content_decode_steps`. Once the
        // first non-empty content is sent and the root closes, the
        // remainder of the loop runs with no P5h emission.
        #[cfg(feature = "p5h-profile")]
        let mut p5h_first_iter = true;
        loop {
            #[cfg(feature = "p5h-profile")]
            let iter_top_span = if root_guard.is_open() {
                let name: &'static str = if p5h_first_iter {
                    "gs_first_token_materialize_and_predispatch"
                } else {
                    "pre_content_decode_steps"
                };
                Some(crate::core::p5h::open_p5h_span(
                    &p5h_ctx,
                    Some(root_guard.span()),
                    name,
                ))
            } else {
                None
            };

            #[cfg(feature = "p5h-profile")]
            let _iter_guard = iter_top_span
                .as_ref()
                .map(|s| crate::core::p5h::P5hTraceGuard::enter(p5h_ctx.clone(), s.clone()));

            let ev_result = stream.next_token();

            #[cfg(feature = "p5h-profile")]
            drop(_iter_guard);
            #[cfg(feature = "p5h-profile")]
            if let Some(span) = iter_top_span {
                crate::core::p5h::close_p5h_span(
                    &p5h_ctx,
                    span,
                    crate::core::p5h::monotonic_ns_public(),
                    crate::core::p5h::SpanFields::default(),
                );
            }

            #[cfg(feature = "p5h-profile")]
            {
                p5h_first_iter = false;
            }

            match ev_result {
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

                    // T0a.8 Step 5 (content): wrap first non-empty content send
                    // in detok_format_first_content_chunk + close root after.
                    #[cfg(feature = "p5h-profile")]
                    let is_first_non_empty_content = !ev.text.is_empty() && root_guard.is_open();
                    #[cfg(feature = "p5h-profile")]
                    let content_span = if is_first_non_empty_content {
                        Some(crate::core::p5h::open_p5h_span(
                            &p5h_ctx,
                            Some(root_guard.span()),
                            "detok_format_first_content_chunk",
                        ))
                    } else {
                        None
                    };

                    let content_send_result = tx.blocking_send(Ok(format_sse_data(&chunk)));
                    #[cfg(feature = "p5h-profile")]
                    let content_send_end_ns = crate::core::p5h::monotonic_ns_public();

                    #[cfg(feature = "p5h-profile")]
                    if let Some(handle) = content_span {
                        // Close the content_span on BOTH send success and error
                        // paths (prevents OPEN_SPAN_REGISTRY leak — per Codex
                        // plan review v10 P2 #4).
                        crate::core::p5h::close_p5h_span(
                            &p5h_ctx,
                            handle,
                            content_send_end_ns,
                            crate::core::p5h::SpanFields::default(),
                        );
                        // Per Codex T0a.14 review: root closes as success ONLY
                        // when first content was actually delivered. When
                        // tx.send fails (receiver/client disconnected before
                        // first content arrived), leave root_guard open so
                        // P5hRootCloseGuard::Drop runs close_at_aborted with
                        // mode="aborted" — required by spec § 2.5a (design.md
                        // line 576) for T0a/T5 structural validation +
                        // coverage gate correctness.
                        if content_send_result.is_ok() {
                            root_guard.close_success(content_send_end_ns);
                        }
                    }

                    if content_send_result.is_err() {
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
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap();
    #[cfg(feature = "p5h-profile")]
    let response = {
        let mut resp = response;
        resp.headers_mut().insert(
            "X-Ironmlx-Request-Id",
            p5h_response_request_id
                .parse()
                .expect("p5h request_id is a valid HTTP header value (UUID)"),
        );
        resp
    };
    response
}

/// Text-only short-prompt SSE path via SchedulerActor (3b-2 swap-in).
async fn serve_via_scheduler_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    // Pre-move clones (per Codex plan review v8 P1 #2). After this block we
    // can read p5h state via these locals even after `request` is moved into
    // SchedulerCommand::Admit.
    #[cfg(feature = "p5h-profile")]
    let p5h_ctx_for_admission = request
        .p5h_trace
        .clone()
        .expect("p5h-profile: GenerateRequest.p5h_trace not populated by handler");
    #[cfg(feature = "p5h-profile")]
    let p5h_root_span_for_admission = request
        .p5h_root_span
        .clone()
        .expect("p5h-profile: GenerateRequest.p5h_root_span not populated by handler");
    #[cfg(feature = "p5h-profile")]
    let p5h_response_request_id = p5h_ctx_for_admission.request_id.clone();
    #[cfg(feature = "p5h-profile")]
    let p5h_root_handle_for_forwarder = crate::core::p5h::RootSpanHandle::new(
        p5h_ctx_for_admission.clone(),
        p5h_root_span_for_admission.clone(),
    );

    let id = gen_id();

    // 1. Admit request to the actor.
    let (reply_tx, reply_rx) = oneshot::channel();

    #[cfg(feature = "p5h-profile")]
    let admission_span_handle = crate::core::p5h::open_p5h_span(
        &p5h_ctx_for_admission,
        Some(&p5h_root_span_for_admission),
        "scheduler_admission",
    );

    // Capture-result: collect send + reply_rx.await + inner-Result match into
    // Result<AdmitReply, Response>. On success we preserve AdmitReply so the
    // forwarder can recover event_rx; on error we already have the Response
    // shape the function returns. Per Codex v17 P1 #2.
    let admission_result: std::result::Result<AdmitReply, Response> = async {
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
            Ok(Ok(r)) => Ok(r),
            Ok(Err(e)) => Err(admit_err_to_response(e)),
            Err(_) => {
                Err((StatusCode::SERVICE_UNAVAILABLE, "scheduler reply lost").into_response())
            }
        }
    }
    .await;

    #[cfg(feature = "p5h-profile")]
    let admission_close_end_ns = crate::core::p5h::monotonic_ns_public();
    #[cfg(feature = "p5h-profile")]
    crate::core::p5h::close_p5h_span(
        &p5h_ctx_for_admission,
        admission_span_handle,
        admission_close_end_ns,
        crate::core::p5h::SpanFields::default(),
    );

    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match admission_result {
        Ok(reply) => reply,
        Err(resp) => {
            // Per Codex plan review v16 P1 #2 + v17 P1 #2: admission failed →
            // forwarder never spawned → no `P5hRootCloseGuard` exists to
            // abort-close root on drop. Close root explicitly via
            // `close_at_aborted` so OPEN_SPAN_REGISTRY does not leak the root
            // span_id. Reconstruct `RootSpanHandle` from the pre-move locals
            // (Step 4.5 already cloned ctx + root span).
            #[cfg(feature = "p5h-profile")]
            crate::core::p5h::RootSpanHandle::new(
                p5h_ctx_for_admission.clone(),
                p5h_root_span_for_admission.clone(),
            )
            .close_at_aborted(admission_close_end_ns);

            return resp;
        }
    };

    // Successful admission — proceed to spawn the forwarder using `event_rx`.
    // The forwarder's own `P5hRootCloseGuard::new(p5h_root_handle_for_forwarder)`
    // (T0a.7 Step 2) takes over once-close + abort-cleanup ownership from here.

    // 2. Stream events as SSE. Spawn a forwarder task that detokenizes
    // per-event and pushes formatted SSE chunks to a bounded channel.
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    let id_for_task = id.clone();
    let model_id_for_task = model_id.clone();
    let tokenizer = state.tokenizer.clone();

    // p5h_ctx_for_admission + p5h_root_handle_for_forwarder are already in
    // scope from T0a.6 Step 4.5. For the forwarder we want self-documenting
    // aliases at the spawn site; .clone() is cheap (P5hTraceContext +
    // RootSpanHandle both derive Clone). Both types are PLAIN (not Option)
    // per Codex plan review v11 P1 #1.
    #[cfg(feature = "p5h-profile")]
    let p5h_ctx: crate::core::p5h::P5hTraceContext = p5h_ctx_for_admission.clone();
    #[cfg(feature = "p5h-profile")]
    let p5h_root_handle_forwarder: crate::core::p5h::RootSpanHandle =
        p5h_root_handle_for_forwarder.clone();

    tokio::spawn(async move {
        #[cfg(feature = "p5h-profile")]
        let mut root_guard = crate::core::p5h::P5hRootCloseGuard::new(p5h_root_handle_forwarder);

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

        #[cfg(feature = "p5h-profile")]
        let role_span = crate::core::p5h::open_p5h_span(
            &p5h_ctx,
            Some(root_guard.span()),
            "sse_write_role_chunk_diagnostic",
        );

        let role_send_result = tx.send(Ok(format_sse_data(&role_chunk))).await;

        // Close diagnostic span on BOTH success and error paths (per Codex
        // plan review v10 P2 #4) — if the receiver dropped, we still need
        // to close the open span before the closure returns, otherwise
        // OPEN_SPAN_REGISTRY leaks the span_id and the next close with
        // that id panics "duplicate".
        #[cfg(feature = "p5h-profile")]
        let role_close_end_ns = crate::core::p5h::monotonic_ns_public();
        #[cfg(feature = "p5h-profile")]
        crate::core::p5h::close_p5h_span_diagnostic(
            &p5h_ctx,
            role_span,
            role_close_end_ns,
            crate::core::p5h::SpanFields::default(),
        );

        if role_send_result.is_err() {
            // Per Codex plan review v12 P2 #6 + v13 P1 #1 + v14 P1 #1: the
            // `P5hRootCloseGuard` declared at the top of the forwarder
            // closure fires on drop when the root is still open, so no
            // explicit `close_at_aborted` is needed here. Just `return;`.
            return;
        }

        let mut detok = tokenizer.decode_stream(/* skip_special */ true);
        while let Some(ev) = event_rx.recv().await {
            #[cfg(feature = "p5h-profile")]
            let detok_start_ns = crate::core::p5h::monotonic_ns_public();

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

            #[cfg(feature = "p5h-profile")]
            let is_first_non_empty_content = !text.is_empty() && root_guard.is_open();

            #[cfg(feature = "p5h-profile")]
            let content_span = if is_first_non_empty_content {
                Some(crate::core::p5h::open_p5h_span_at(
                    &p5h_ctx,
                    Some(root_guard.span()),
                    "detok_format_first_content_chunk",
                    detok_start_ns,
                ))
            } else {
                None
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
            let content_send_result = tx.send(Ok(format_sse_data(&chunk))).await;
            #[cfg(feature = "p5h-profile")]
            let content_send_end_ns = crate::core::p5h::monotonic_ns_public();

            // Close the content_span on BOTH send success and error
            // paths (prevents OPEN_SPAN_REGISTRY leak — per Codex plan
            // review v10 P2 #4).
            #[cfg(feature = "p5h-profile")]
            if let Some(handle) = content_span {
                crate::core::p5h::close_p5h_span(
                    &p5h_ctx,
                    handle,
                    content_send_end_ns,
                    crate::core::p5h::SpanFields::default(),
                );
                // Per Codex T0a.14 review: root closes as success ONLY
                // when first content was actually delivered. When
                // tx.send fails (receiver/client disconnected before
                // first content arrived), leave root_guard open so
                // P5hRootCloseGuard::Drop runs close_at_aborted with
                // mode="aborted" — required by spec § 2.5a (design.md
                // line 576) for T0a/T5 structural validation + coverage
                // gate correctness. close_success enforces once-close
                // discipline; panics if called twice (state-machine bug
                // — is_first_non_empty_content stayed true across
                // iterations).
                if content_send_result.is_ok() {
                    root_guard.close_success(content_send_end_ns);
                }
            }

            if content_send_result.is_err() {
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
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap();
    #[cfg(feature = "p5h-profile")]
    let response = {
        let mut resp = response;
        resp.headers_mut().insert(
            "X-Ironmlx-Request-Id",
            p5h_response_request_id
                .parse()
                .expect("p5h request_id is a valid HTTP header value (UUID)"),
        );
        resp
    };
    response
}

async fn serve_via_gs_unary<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let id = gen_id();
    let result = tokio::task::spawn_blocking(
        move || -> std::result::Result<(String, &'static str, u32), String> {
            let model_guard = state.model.blocking_lock();
            let tokenizer = &*state.tokenizer;
            let mut stream = GenerationStream::new(&*model_guard, tokenizer, request)
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
async fn serve_via_scheduler_unary<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
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

    #[test]
    fn qwen_placeholder_uses_existing_vl_tokens() {
        assert_eq!(
            qwen_placeholder(2),
            "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
        );
    }

    #[test]
    fn gemma4_placeholder_uses_boundary_and_soft_tokens() {
        assert_eq!(gemma4_placeholder(2), "<|image><|image|><|image|><image|>");
    }

    #[test]
    fn flatten_content_inserts_placeholders_in_part_order() {
        let content = Content::Parts(vec![
            ContentPart::Text {
                text: "before ".into(),
            },
            ContentPart::ImageUrl {
                image_url: crate::core::server::chat_format::ImageUrl {
                    url: "data:image/jpeg;base64,".into(),
                },
            },
            ContentPart::Text {
                text: " after".into(),
            },
        ]);
        let placeholders = vec!["<image-placeholder>".to_string()];
        let mut iter = placeholders.into_iter();
        assert_eq!(
            flatten_content_with_placeholders(content, &mut iter),
            "before <image-placeholder> after"
        );
    }

    #[test]
    fn flatten_content_inserts_multiple_placeholders_in_part_order() {
        let content = Content::Parts(vec![
            ContentPart::Text { text: "a ".into() },
            ContentPart::ImageUrl {
                image_url: crate::core::server::chat_format::ImageUrl {
                    url: "data:image/jpeg;base64,".into(),
                },
            },
            ContentPart::Text { text: " b ".into() },
            ContentPart::ImageUrl {
                image_url: crate::core::server::chat_format::ImageUrl {
                    url: "data:image/jpeg;base64,".into(),
                },
            },
            ContentPart::Text { text: " c".into() },
        ]);
        let placeholders = vec!["<img-0>".to_string(), "<img-1>".to_string()];
        let mut iter = placeholders.into_iter();
        assert_eq!(
            flatten_content_with_placeholders(content, &mut iter),
            "a <img-0> b <img-1> c"
        );
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
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("admission queue full"));
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
            needed: 50000,
            max: 32768,
        });
        let resp = admit_err_to_response(err);
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

        // No Retry-After header for 413 (client error, not transient).
        assert!(resp.headers().get("retry-after").is_none());

        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            body_str.contains("50000"),
            "body should mention needed=50000, got: {body_str}"
        );
        assert!(
            body_str.contains("32768"),
            "body should mention max=32768, got: {body_str}"
        );
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
}
