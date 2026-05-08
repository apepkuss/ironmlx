//! Anthropic-compatible /v1/messages handler. Real implementation lands in T9.

use axum::{extract::State, http::StatusCode, response::IntoResponse};

use super::AppState;

pub async fn messages(State(_state): State<AppState>, body: String) -> impl IntoResponse {
    let _ = body;
    (
        StatusCode::NOT_IMPLEMENTED,
        "anthropic handler implemented in T9",
    )
}
