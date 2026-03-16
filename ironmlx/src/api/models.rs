use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use tokio_stream::wrappers::ReceiverStream;

use super::types::*;
use crate::state::AppState;
use ironmlx_core::generate::{ChatMessage as CoreChatMessage, SamplerConfig};

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
    let prompt = build_chat_prompt(&state, &req.messages)?;
    let prompt_tokens = encode_or_error(&state, &prompt)?;
    let prompt_len = prompt_tokens.len();
    let sampler = build_sampler(&req);
    let max_tokens = req.max_tokens;
    let eos = state.config.eos_token_id as i32;
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let mut token_rx = state
        .engine
        .add_request(request_id.clone(), prompt_tokens, sampler, max_tokens, eos)
        .await;

    let mut generated_text = String::new();
    let mut completion_len = 0usize;
    let mut finish_reason = "stop".to_string();

    while let Some(output) = token_rx.recv().await {
        if let Some(ref reason) = output.finish_reason {
            finish_reason = reason.clone();
            break;
        }
        generated_text.push_str(&output.token_text);
        completion_len += 1;
    }

    Ok(Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: state.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: generated_text,
            },
            finish_reason,
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
    let prompt = build_chat_prompt(&state, &req.messages)?;
    let prompt_tokens = encode_or_error(&state, &prompt)?;
    let sampler = build_sampler(&req);
    let max_tokens = req.max_tokens;
    let eos = state.config.eos_token_id as i32;
    let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model_id = state.model_id.clone();

    let mut token_rx = state
        .engine
        .add_request(chunk_id.clone(), prompt_tokens, sampler, max_tokens, eos)
        .await;

    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    let chunk_id_clone = chunk_id.clone();
    let model_id_clone = model_id.clone();
    tokio::spawn(async move {
        // Send initial chunk with role
        let role_chunk = ChatCompletionChunk {
            id: chunk_id_clone.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id_clone.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let _ = sse_tx
            .send(Ok(
                Event::default().data(serde_json::to_string(&role_chunk).unwrap())
            ))
            .await;

        // Stream tokens
        while let Some(output) = token_rx.recv().await {
            if let Some(ref reason) = output.finish_reason {
                let final_chunk = ChatCompletionChunk {
                    id: chunk_id_clone.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id_clone.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some(reason.clone()),
                    }],
                };
                let _ = sse_tx
                    .send(Ok(
                        Event::default().data(serde_json::to_string(&final_chunk).unwrap())
                    ))
                    .await;
                break;
            }

            let chunk = ChatCompletionChunk {
                id: chunk_id_clone.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id_clone.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: Some(output.token_text),
                    },
                    finish_reason: None,
                }],
            };
            if sse_tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                ))
                .await
                .is_err()
            {
                break; // Client disconnected
            }
        }

        // Send [DONE] marker
        let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    let stream = ReceiverStream::new(sse_rx);
    Ok(Sse::new(stream).into_response())
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let prompt_tokens = encode_or_error(&state, &req.prompt)?;
    let prompt_len = prompt_tokens.len();
    let eos = state.config.eos_token_id as i32;
    let sampler = SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        ..SamplerConfig::default()
    };
    let max_tokens = req.max_tokens;
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

    let mut token_rx = state
        .engine
        .add_request(request_id.clone(), prompt_tokens, sampler, max_tokens, eos)
        .await;

    let mut text = String::new();
    let mut completion_len = 0usize;
    let mut finish_reason = "stop".to_string();

    while let Some(output) = token_rx.recv().await {
        if let Some(ref reason) = output.finish_reason {
            finish_reason = reason.clone();
            break;
        }
        text.push_str(&output.token_text);
        completion_len += 1;
    }

    Ok(Json(CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: state.model_id.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason,
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

fn build_chat_prompt(
    state: &AppState,
    messages: &[ChatMessage],
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    if let Some(ref ct) = state.chat_template {
        // Convert API ChatMessage to core ChatMessage for template rendering
        let core_messages: Vec<CoreChatMessage> = messages
            .iter()
            .map(|m| CoreChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();
        ct.apply(&core_messages, true)
            .map_err(|e| internal_error(&format!("chat template error: {}", e)))
    } else {
        // Fallback: simple concatenation
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
        }
        prompt.push_str("assistant: ");
        Ok(prompt)
    }
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
