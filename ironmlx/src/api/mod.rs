mod models;
mod types;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};

use crate::state::AppState;

pub fn router(state: Arc<AppState>) -> Router {
    let admin_routes = crate::admin::router(state.clone());

    Router::new()
        .route("/v1/chat/completions", post(models::chat_completions))
        .route("/v1/completions", post(models::completions))
        .route("/v1/messages", post(models::anthropic_messages))
        .route("/v1/embeddings", post(models::embeddings))
        .route("/v1/rerank", post(models::rerank))
        .route("/v1/models", get(models::list_models))
        .route("/v1/models/load", post(models::load_model_endpoint))
        .route("/v1/models/unload", post(models::unload_model_endpoint))
        .route("/v1/models/default", post(models::set_default_model))
        .route("/health", get(models::health))
        .merge(admin_routes)
        // Limit request body to 2MB to prevent OOM from oversized prompts
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(state)
}
