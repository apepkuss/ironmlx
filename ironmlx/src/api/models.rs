use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use super::types::*;
use crate::state::AppState;
use ironmlx_core::generate::{SamplerConfig, generate};

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let prompt = build_chat_prompt(&req.messages);

    let prompt_tokens = state.tokenizer.encode(&prompt).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_response(&format!("tokenize error: {}", e))),
        )
    })?;
    let prompt_len = prompt_tokens.len();

    let sampler = SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        ..SamplerConfig::default()
    };

    let max_tokens = req.max_tokens;

    let generated_ids = {
        let model_state = state.model.lock().unwrap();
        generate(&model_state.llama, &prompt_tokens, max_tokens, &sampler, 2).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(error_response(&format!("generate error: {}", e))),
            )
        })?
    };

    let completion_len = generated_ids.len();
    let response_text = state.tokenizer.decode(&generated_ids).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_response(&format!("decode error: {}", e))),
        )
    })?;

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

/// POST /v1/completions
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let prompt_tokens = state.tokenizer.encode(&req.prompt).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_response(&format!("tokenize error: {}", e))),
        )
    })?;
    let prompt_len = prompt_tokens.len();

    let sampler = SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        ..SamplerConfig::default()
    };

    let max_tokens = req.max_tokens;

    let generated_ids = {
        let model_state = state.model.lock().unwrap();
        generate(&model_state.llama, &prompt_tokens, max_tokens, &sampler, 2).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(error_response(&format!("generate error: {}", e))),
            )
        })?
    };

    let completion_len = generated_ids.len();
    let text = state.tokenizer.decode(&generated_ids).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_response(&format!("decode error: {}", e))),
        )
    })?;

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

fn build_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("[INST] {} [/INST]\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("{}\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("{}\n", msg.content));
            }
        }
    }
    prompt
}

fn error_response(message: &str) -> ErrorResponse {
    ErrorResponse {
        error: ErrorDetail {
            message: message.to_string(),
            r#type: "server_error".to_string(),
        },
    }
}
