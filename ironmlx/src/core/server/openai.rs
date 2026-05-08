//! OpenAI-compatible /v1/chat/completions handler. Real implementation lands in T8.

use axum::{extract::State, http::StatusCode, response::IntoResponse};

use super::AppState;

pub async fn chat_completions(State(_state): State<AppState>, body: String) -> impl IntoResponse {
    let _ = body;
    (
        StatusCode::NOT_IMPLEMENTED,
        "openai handler implemented in T8",
    )
}
