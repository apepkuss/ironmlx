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
use crate::core::server::chat_format::render_and_encode;
use crate::core::server::chat_format::{ChatMessage, Content, ContentPart};
use crate::core::server::scheduler_actor::{AdmitReply, SchedulerCommand};
use crate::core::server::vision::{expand_decoded_messages, DecodedMessage, DecodedPart};
use crate::core::server::VisionInputConfig;
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
    pub top_k: Option<i32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// HuggingFace `apply_chat_template` extra kwargs — passed through as
    /// top-level template render-context variables. Honors Qwen3+'s
    /// `enable_thinking` toggle, vLLM's `tools` / `documents`, etc.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
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

/// Decode every OpenAI `image_url` content part into raw bytes, producing the
/// wire-agnostic `DecodedMessage` list consumed by the shared vision core.
pub async fn decode_openai_messages(
    messages: Vec<ChatMessage>,
    client: &reqwest::Client,
) -> anyhow::Result<Vec<DecodedMessage>> {
    let mut out: Vec<DecodedMessage> = Vec::with_capacity(messages.len());
    for msg in messages {
        let mut parts: Vec<DecodedPart> = Vec::new();
        match msg.content {
            Content::Text(t) => parts.push(DecodedPart::Text(t)),
            Content::Parts(ps) => {
                for p in ps {
                    match p {
                        ContentPart::Text { text } => parts.push(DecodedPart::Text(text)),
                        ContentPart::ImageUrl { image_url } => {
                            let bytes = decode_image_url(&image_url.url, client).await?;
                            parts.push(DecodedPart::Image(bytes));
                        }
                    }
                }
            }
        }
        out.push(DecodedMessage {
            role: msg.role,
            parts,
        });
    }
    Ok(out)
}

/// Walk `messages`, decode + preprocess every `image_url`, and rewrite to
/// text-with-placeholder. Signature unchanged so the handler call site is
/// untouched; the body now delegates to the shared `vision` core.
pub async fn expand_image_parts_in_messages(
    messages: Vec<ChatMessage>,
    client: &reqwest::Client,
    vision_input: &VisionInputConfig,
) -> anyhow::Result<(Vec<ChatMessage>, Option<Vec<Array>>, Vec<(i32, i32, i32)>)> {
    let decoded = decode_openai_messages(messages, client).await?;
    expand_decoded_messages(decoded, vision_input)
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

fn build_sampler(req: &ChatRequest, defaults: SamplingDefaults) -> Sampler {
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

fn stop_token_ids_for_request(eos_token_ids: &[u32], ignore_eos: bool) -> Vec<u32> {
    if ignore_eos {
        Vec::new()
    } else {
        eos_token_ids.to_vec()
    }
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
    chat_completions_with_state(state, req).await
}

pub(crate) async fn chat_completions_with_state<M>(state: AppState<M>, req: ChatRequest) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
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
        return (StatusCode::BAD_REQUEST, format!("{error:#}")).into_response();
    }
    let chat_template_kwargs = req.chat_template_kwargs;

    // Build a per-request reqwest client for image fetching.
    // For text-only requests this is a cheap no-op (no images to fetch).
    let http_client = reqwest::Client::new();

    let (image_token_id, spatial_merge_size) =
        crate::core::server::vision::derive_image_token_and_merge(
            &state.vision_input,
            &state.tokenizer,
        );

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
    let scheduler_config = state.scheduler_request_config(prompt_len, max_tokens);

    // Routing: short-prompt, paged-prefix-cache, and model-limited chunked
    // long-prompt requests use SchedulerActor; other chunked long prompts keep
    // using GenerationStream.
    // B1-p2.4: VL fallback removed — VL requests now route through Scheduler
    // via Scheduler::admit/admit_mid + batched_prefill_vl.
    let use_scheduler = super::should_route_to_scheduler::<M>(
        prompt_len,
        scheduler_config.prefill_chunk_size,
        state.b_max,
        state.paged_prefix_cache_enabled,
        state.force_scheduler_for_greedy && sampler.is_pipelinable(),
    );

    let stop_token_ids = stop_token_ids_for_request(state.tokenizer.eos_token_ids(), ignore_eos);
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
    };

    match chat_completions_route(stream, use_scheduler) {
        ChatCompletionsRoute::SchedulerStream => {
            serve_via_scheduler_stream(state, request, model_label, prompt_tokens, include_usage)
                .await
        }
        ChatCompletionsRoute::GenerationStreamStream => {
            serve_via_gs_stream(state, request, model_label, prompt_tokens, include_usage).await
        }
        ChatCompletionsRoute::SchedulerUnary => {
            serve_via_scheduler_unary(state, request, model_label, prompt_tokens).await
        }
        ChatCompletionsRoute::GenerationStreamUnary => {
            serve_via_gs_unary(state, request, model_label, prompt_tokens).await
        }
    }
}

pub(crate) async fn gemma4_drafter_chat_completions(
    State(state): State<Gemma4DrafterAppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    chat_completions_with_gemma4_drafter_state(state, req).await
}

pub(crate) async fn chat_completions_with_gemma4_drafter_state(
    state: Gemma4DrafterAppState,
    req: ChatRequest,
) -> Response {
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
        Err(e) => return (StatusCode::BAD_REQUEST, format!("{e:#}")).into_response(),
    };
    let chat_template_kwargs = req.chat_template_kwargs;
    let http_client = reqwest::Client::new();

    let (image_token_id, spatial_merge_size) =
        crate::core::server::vision::derive_image_token_and_merge(
            &state.base.vision_input,
            &state.base.tokenizer,
        );
    let (flat_messages, pixel_values, image_grid_thw) =
        match expand_image_parts_in_messages(req.messages, &http_client, &state.base.vision_input)
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
        &state.base.tokenizer,
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
    let stop_token_ids =
        stop_token_ids_for_request(state.base.tokenizer.eos_token_ids(), ignore_eos);
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
    };

    if stream {
        serve_via_scheduler_stream(
            state.base,
            request,
            model_label,
            prompt_tokens,
            include_usage,
        )
        .await
    } else {
        serve_via_scheduler_unary(state.base, request, model_label, prompt_tokens).await
    }
}

async fn serve_via_gs_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
    include_usage: bool,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
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

        let mut completion_tokens = 0_u32;
        let mut first_event = Some(first_event);
        loop {
            let ev_result = match first_event.take() {
                Some(event) => Ok(event),
                None => stream.next_token(),
            };

            match ev_result {
                Ok(Some(ev)) => {
                    completion_tokens += 1;
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

                    let content_send_result = tx.blocking_send(Ok(format_sse_data(&chunk)));

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

/// Text-only short-prompt SSE path via SchedulerActor (3b-2 swap-in).
async fn serve_via_scheduler_stream<M>(
    state: AppState<M>,
    request: GenerateRequest,
    model_id: String,
    prompt_tokens: u32,
    include_usage: bool,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let id = gen_id();

    // 1. Admit request to the actor.
    let (reply_tx, reply_rx) = oneshot::channel();

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

    let AdmitReply {
        request_id: _,
        mut event_rx,
    } = match admission_result {
        Ok(reply) => reply,
        Err(resp) => {
            return resp;
        }
    };

    // Successful admission — proceed to spawn the forwarder using `event_rx`.

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

        let role_send_result = tx.send(Ok(format_sse_data(&role_chunk))).await;

        if role_send_result.is_err() {
            return;
        }

        let mut detok = tokenizer.decode_stream(/* skip_special */ true);
        let mut completion_tokens = 0_u32;
        while let Some(ev) = event_rx.recv().await {
            completion_tokens += 1;
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
            let content_send_result = tx.send(Ok(format_sse_data(&chunk))).await;

            // Close the content_span on BOTH send success and error
            // paths (prevents OPEN_SPAN_REGISTRY leak — per Codex plan
            // review v10 P2 #4).

            if content_send_result.is_err() {
                break;
            }
            if ev.finish_reason.is_some() {
                break;
            }
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
    prompt_tokens: u32,
) -> Response
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let id = gen_id();
    let result =
        tokio::task::spawn_blocking(move || -> anyhow::Result<(String, &'static str, u32)> {
            let model_guard = state.model.blocking_lock();
            let tokenizer = &*state.tokenizer;
            let memory = super::begin_direct_request_memory(&state, &*model_guard, &request)?;
            let mut stream = GenerationStream::new(&*model_guard, tokenizer, request)?;
            let mut buf = String::new();
            let mut finish: &'static str = "stop";
            let mut completion_tokens: u32 = 0;
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
                completion_tokens += 1;
                if let Some(reason) = ev.finish_reason {
                    finish = reason;
                    break;
                }
            }
            Ok((buf, finish, completion_tokens))
        })
        .await;

    let (content, finish, completion_tokens) = match result {
        Ok(Ok(t)) => t,
        Ok(Err(err)) => return generation_err_to_response(err),
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
    fn build_sampler_prefers_request_values_over_model_defaults() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [],
            "max_tokens": 8,
            "temperature": 0.2,
            "top_p": 0.6,
            "top_k": 16,
            "repetition_penalty": 1.05
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
        assert_eq!(sampler.top_k, Some(16));
        assert_eq!(sampler.repetition_penalty, Some(1.05));
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
