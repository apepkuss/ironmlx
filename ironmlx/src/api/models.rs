use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::types::*;
use crate::state::AppState;
use ironmlx_core::generate::{SamplerConfig, StopReason, generate, stream_generate};

/// POST /v1/chat/completions — supports both regular and streaming responses
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    if req.stream {
        chat_completions_stream(state, req).await
    } else {
        chat_completions_sync(state, req)
            .await
            .map(|j| j.into_response())
    }
}

async fn chat_completions_sync(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let prompt = build_chat_prompt(&req.messages);
    let prompt_tokens = encode_or_error(&state, &prompt)?;
    let prompt_len = prompt_tokens.len();
    let sampler = build_sampler(&req);
    let max_tokens = req.max_tokens;

    let generated_ids = {
        let model_state = state.model.lock().unwrap();
        generate(&model_state.llama, &prompt_tokens, max_tokens, &sampler, 2)
            .map_err(|e| internal_error(&format!("generate error: {}", e)))?
    };

    let completion_len = generated_ids.len();
    let response_text = state
        .tokenizer
        .decode(&generated_ids)
        .map_err(|e| internal_error(&format!("decode error: {}", e)))?;

    let finish_reason = if completion_len >= max_tokens {
        "length"
    } else {
        "stop"
    };

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: state.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_text,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
            total_tokens: prompt_len + completion_len,
        },
    }))
}

async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let prompt = build_chat_prompt(&req.messages);
    let prompt_tokens = encode_or_error(&state, &prompt)?;
    let sampler = build_sampler(&req);
    let max_tokens = req.max_tokens;
    let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model_id = state.model_id.clone();

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);

    // Spawn blocking generation in a separate thread
    let tx_gen = tx.clone();
    let chunk_id_gen = chunk_id.clone();
    let model_id_gen = model_id.clone();
    tokio::task::spawn_blocking(move || {
        // Send initial chunk with role
        let role_chunk = ChatCompletionChunk {
            id: chunk_id_gen.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id_gen.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let _ = tx_gen.blocking_send(Ok(
            Event::default().data(serde_json::to_string(&role_chunk).unwrap())
        ));

        let model_state = state.model.lock().unwrap();
        let stop_reason = stream_generate(
            &model_state.llama,
            &prompt_tokens,
            max_tokens,
            &sampler,
            2,
            |token_id| {
                let text = state.tokenizer.decode(&[token_id]).unwrap_or_default();
                let chunk = ChatCompletionChunk {
                    id: chunk_id_gen.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id_gen.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                };
                let data = serde_json::to_string(&chunk).unwrap();
                tx_gen
                    .blocking_send(Ok(Event::default().data(data)))
                    .is_ok()
            },
        );

        // Send final chunk with finish_reason
        let finish_reason = match stop_reason {
            Ok(StopReason::Eos) => "stop",
            Ok(StopReason::MaxTokens) => "length",
            Err(_) => "stop",
        };
        let final_chunk = ChatCompletionChunk {
            id: chunk_id_gen,
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id_gen,
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
        };
        let _ = tx_gen.blocking_send(Ok(
            Event::default().data(serde_json::to_string(&final_chunk).unwrap())
        ));

        // Send [DONE] marker
        let _ = tx_gen.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream).into_response())
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let prompt_tokens = encode_or_error(&state, &req.prompt)?;
    let prompt_len = prompt_tokens.len();
    let sampler = SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        ..SamplerConfig::default()
    };
    let max_tokens = req.max_tokens;

    let generated_ids = {
        let model_state = state.model.lock().unwrap();
        generate(&model_state.llama, &prompt_tokens, max_tokens, &sampler, 2)
            .map_err(|e| internal_error(&format!("generate error: {}", e)))?
    };

    let completion_len = generated_ids.len();
    let text = state
        .tokenizer
        .decode(&generated_ids)
        .map_err(|e| internal_error(&format!("decode error: {}", e)))?;

    let finish_reason = if completion_len >= max_tokens {
        "length"
    } else {
        "stop"
    };

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: state.model_id.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
            total_tokens: prompt_len + completion_len,
        },
    }))
}

/// GET /v1/models
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "local".to_string(),
        }],
    })
}

/// GET /health
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.model_id.clone(),
    })
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn build_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", msg.content)),
            "user" => prompt.push_str(&format!("[INST] {} [/INST]\n", msg.content)),
            "assistant" => prompt.push_str(&format!("{}\n", msg.content)),
            _ => prompt.push_str(&format!("{}\n", msg.content)),
        }
    }
    prompt
}

fn build_sampler(req: &ChatCompletionRequest) -> SamplerConfig {
    SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        ..SamplerConfig::default()
    }
}

fn encode_or_error(
    state: &AppState,
    text: &str,
) -> Result<Vec<i32>, (StatusCode, Json<ErrorResponse>)> {
    state
        .tokenizer
        .encode(text)
        .map_err(|e| internal_error(&format!("tokenize error: {}", e)))
}

fn internal_error(message: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                r#type: "server_error".to_string(),
            },
        }),
    )
}
