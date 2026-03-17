mod models;
mod types;

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};

use crate::state::AppState;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(models::chat_completions))
        .route("/v1/completions", post(models::completions))
        .route("/v1/messages", post(models::anthropic_messages))
        .route("/v1/models", get(models::list_models))
        .route("/health", get(models::health))
        .with_state(state)
}
