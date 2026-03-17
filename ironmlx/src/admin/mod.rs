use std::sync::Arc;

use axum::Router;
use axum::response::Html;
use axum::routing::get;

use crate::state::AppState;

// Embed static files at compile time
const INDEX_HTML: &str = include_str!("static/index.html");
const APP_JS: &str = include_str!("static/app.js");
const STYLE_CSS: &str = include_str!("static/style.css");

/// Admin panel routes (state applied by parent router).
pub fn router(_state: Arc<AppState>) -> Router<Arc<AppState>> {
    Router::new()
        .route("/admin", get(admin_index))
        .route("/admin/static/app.js", get(serve_app_js))
        .route("/admin/static/style.css", get(serve_style_css))
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

async fn serve_style_css() -> (
    [(axum::http::header::HeaderName, &'static str); 1],
    &'static str,
) {
    ([(axum::http::header::CONTENT_TYPE, "text/css")], STYLE_CSS)
}
