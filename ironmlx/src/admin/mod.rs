mod api;

use std::sync::Arc;

use axum::Router;
use axum::response::Html;
use axum::routing::{delete, get, post};

use crate::state::AppState;

// Embed static files at compile time
const INDEX_HTML: &str = include_str!("static/index.html");
const APP_JS: &str = include_str!("static/app.js");
const I18N_JS: &str = include_str!("static/i18n.js");
const STYLE_CSS: &str = include_str!("static/style.css");

/// Admin panel routes (state applied by parent router).
pub fn router(_state: Arc<AppState>) -> Router<Arc<AppState>> {
    Router::new()
        .route("/admin", get(admin_index))
        .route("/admin/static/app.js", get(serve_app_js))
        .route("/admin/static/i18n.js", get(serve_i18n_js))
        .route("/admin/static/style.css", get(serve_style_css))
        .route(
            "/admin/api/settings",
            get(api::get_settings).post(api::update_settings),
        )
        .route("/admin/api/auth", post(api::auth))
        .route("/admin/api/logs", get(api::get_logs))
        .route("/admin/api/benchmark", post(api::run_benchmark))
        .route(
            "/admin/api/benchmark/history",
            get(api::get_benchmark_history).delete(api::clear_benchmark_history),
        )
        .route("/admin/api/models/search", post(api::hf_search))
        .route("/admin/api/models/download", post(api::hf_download))
        .route("/admin/api/models/downloads", get(api::hf_downloads))
}

async fn admin_index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn serve_app_js() -> (
    [(axum::http::header::HeaderName, &'static str); 1],
    &'static str,
) {
    (
        [(axum::http::header::CONTENT_TYPE, "application/javascript")],
        APP_JS,
    )
}

async fn serve_i18n_js() -> (
    [(axum::http::header::HeaderName, &'static str); 1],
    &'static str,
) {
    (
        [(axum::http::header::CONTENT_TYPE, "application/javascript")],
        I18N_JS,
    )
}

async fn serve_style_css() -> (
    [(axum::http::header::HeaderName, &'static str); 1],
    &'static str,
) {
    ([(axum::http::header::CONTENT_TYPE, "text/css")], STYLE_CSS)
}
