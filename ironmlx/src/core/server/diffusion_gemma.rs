//! DiffusionGemma HTTP server lane.
//!
//! This module is intentionally separate from the causal `AppState<M>` /
//! `SchedulerActor` server path. DiffusionGemma is a block-diffusion model, so
//! requests are admitted through a serial lane backed by the model mutex and
//! completed as non-streaming OpenAI/Anthropic responses.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use mlx::Array;
use serde::Serialize;
use tokio::sync::Mutex;

use crate::core::server::chat_format::render_and_encode;
use crate::core::server::vision::expand_decoded_messages;
use crate::core::server::VisionInputConfig;
use crate::core::tokenizer::Tokenizer;
use crate::models::{
    DiffusionGemmaGenerateEvent, DiffusionGemmaGenerationConfig, DiffusionGemmaModel,
};
use crate::Result;

#[derive(Clone)]
pub struct DiffusionGemmaAppState {
    pub model: Arc<Mutex<DiffusionGemmaModel>>,
    pub tokenizer: Arc<Tokenizer>,
    pub generation_config: DiffusionGemmaGenerationConfig,
    pub model_id: String,
    pub vision_input: VisionInputConfig,
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

fn unsupported_streaming_response() -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorEnvelope {
            error: ErrorBody {
                message: "DiffusionGemma server lane does not support streaming; send stream=false",
                kind: "unsupported_feature",
            },
        }),
    )
        .into_response()
}

fn collect_events(
    events: Vec<DiffusionGemmaGenerateEvent>,
    default_finish: &'static str,
) -> CompletionParts {
    let mut content = String::new();
    let mut finish_reason = default_finish;
    let mut completion_tokens = 0_u32;
    for event in events {
        let length_sentinel =
            event.finish_reason == Some("length") && event.token == 0 && event.text.is_empty();
        if !length_sentinel {
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
) -> std::result::Result<CompletionParts, String> {
    tokio::task::spawn_blocking(move || -> std::result::Result<CompletionParts, String> {
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
    .map_err(|e| format!("join error: {e}"))?
}

pub async fn openai_chat_completions(
    State(state): State<DiffusionGemmaAppState>,
    Json(req): Json<super::openai::ChatRequest>,
) -> Response {
    if req.stream {
        return unsupported_streaming_response();
    }

    let max_tokens = req.max_tokens;
    let temperature = req.temperature.unwrap_or(0.0);
    let seed = req.seed.unwrap_or(0);
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let (prepared, prompt_tokens) = match prepare_openai_request(&state, req).await {
        Ok(t) => t,
        Err(resp) => return resp,
    };

    let completion =
        match generate_completion(state, prepared, max_tokens, temperature, seed, "stop").await {
            Ok(c) => c,
            Err(msg) => return (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
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
    if req.stream {
        return unsupported_streaming_response();
    }

    let max_tokens = req.max_tokens;
    let temperature = req.temperature.unwrap_or(0.0);
    let model_label = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let (prepared, input_tokens) = match prepare_anthropic_request(&state, req).await {
        Ok(t) => t,
        Err(resp) => return resp,
    };

    let completion =
        match generate_completion(state, prepared, max_tokens, temperature, 0, "stop").await {
            Ok(c) => c,
            Err(msg) => return (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
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
    };
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route(
            "/healthz",
            get(|| async {
                Json(DiffusionGemmaHealth {
                    status: "ok",
                    scheduler: "serial_block_diffusion",
                })
            }),
        )
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

#[cfg(test)]
mod tests {
    use super::*;

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
