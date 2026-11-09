//! Routes served while the process loads its startup model, and the switch
//! to the full router once the model is ready.
//!
//! The listeners are bound before the load starts so clients can watch it:
//! `/healthz` answers `503 {"status":"starting"}` (any non-2xx means "not
//! ready"), `/admin/api/models/loads` reports the load, and every other path
//! answers `503` with code `backend_starting`.

use std::convert::Infallible;
use std::sync::{Arc, PoisonError, RwLock};
use std::task::{Context, Poll};

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::future::RouteFuture;
use axum::routing::get;
use axum::{Json, Router};
use ironmlx_runtime::core::load_registry::LoadRegistry;
use tower::Service;

use super::api_error::{ApiError, ApiProtocol};

/// Router that serves the startup routes until [`SwitchableRouter::activate`]
/// replaces them with the full router. Clones share the active router.
#[derive(Clone)]
pub struct SwitchableRouter {
    current: Arc<RwLock<Router>>,
}

impl SwitchableRouter {
    pub fn starting(registry: Arc<LoadRegistry>) -> Self {
        Self {
            current: Arc::new(RwLock::new(startup_router(registry))),
        }
    }

    /// Serve `router` for every request from now on.
    pub fn activate(&self, router: Router) {
        *self.current.write().unwrap_or_else(PoisonError::into_inner) = router;
    }

    fn current(&self) -> Router {
        self.current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Service<Request> for SwitchableRouter {
    type Response = Response;
    type Error = Infallible;
    type Future = RouteFuture<Infallible>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: Request) -> Self::Future {
        self.current().call(request)
    }
}

fn startup_router(registry: Arc<LoadRegistry>) -> Router {
    Router::new()
        .route("/healthz", get(starting_healthz))
        .route("/admin/api/models/loads", get(registry_loads_handler))
        .fallback(backend_starting)
        .with_state(registry)
}

async fn starting_healthz() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({"status": "starting"})),
    )
        .into_response()
}

/// `GET /admin/api/models/loads` served from a standalone registry.
pub(crate) async fn registry_loads_handler(
    State(registry): State<Arc<LoadRegistry>>,
) -> Json<serde_json::Value> {
    super::model_manager::loads_response(registry.list())
}

async fn backend_starting(request: Request) -> Response {
    ApiError::from_status(
        StatusCode::SERVICE_UNAVAILABLE,
        "backend_starting",
        "The server is still loading its model.",
    )
    .into_response(ApiProtocol::from_path(request.uri().path()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::{to_bytes, Body};
    use axum::extract::Request;
    use axum::http::StatusCode;
    use axum::routing::{get, post};
    use axum::Router;
    use ironmlx_runtime::core::load_registry::{
        LoadBegin, LoadKind, LoadRegistry, LoadStage, LoadTriggerKind, LoadVariant,
    };
    use tower::ServiceExt;

    use super::SwitchableRouter;
    use crate::server::security::{serve_bound, BoundListeners, ServerNetworkConfig};

    fn startup_load(registry: &LoadRegistry) -> ironmlx_runtime::core::load_registry::LoadHandle {
        registry.begin(LoadBegin {
            load_id: None,
            model: "mlx-community/Qwen3.6-35B-A3B-4bit".into(),
            variant: LoadVariant::Dflash2,
            kind: LoadKind::Load,
            trigger: LoadTriggerKind::Startup,
            config_fingerprint: "startup".into(),
            warning: None,
        })
    }

    async fn get_json(router: &Router, path: &str) -> (StatusCode, serde_json::Value) {
        let response = router
            .clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or_default())
    }

    #[tokio::test]
    async fn startup_status_router_reports_503_until_ready() {
        let listeners =
            BoundListeners::bind(&ServerNetworkConfig::local("127.0.0.1", 0).unwrap()).unwrap();
        let base = format!("http://{}", listeners.local_addr());
        let registry = LoadRegistry::new();
        let router = SwitchableRouter::starting(registry.clone());
        let server = tokio::spawn(serve_bound(listeners, router.clone(), "startup test"));
        let client = reqwest::Client::new();

        let health = client.get(format!("{base}/healthz")).send().await.unwrap();
        assert_eq!(health.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            health.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!({"status": "starting"})
        );
        let chat = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "m", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(chat.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = chat.json().await.unwrap();
        assert_eq!(body["error"]["code"], "backend_starting");
        // Local management routes keep their wrapper while starting.
        let log_level = client
            .get(format!("{base}/admin/api/log-level"))
            .send()
            .await
            .unwrap();
        assert_eq!(log_level.text().await.unwrap(), "log_control_unavailable");

        router.activate(Router::new().route("/healthz", get(|| async { "ready" })));
        let health = client.get(format!("{base}/healthz")).send().await.unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        assert_eq!(health.text().await.unwrap(), "ready");
        server.abort();
    }

    #[tokio::test]
    async fn loads_visible_during_startup() {
        let registry = LoadRegistry::new();
        let load = startup_load(&registry);
        load.enter_stage(LoadStage::ReadingWeights);
        let router = Router::new().fallback_service(SwitchableRouter::starting(registry.clone()));

        let (status, body) = get_json(&router, "/admin/api/models/loads").await;
        assert_eq!(status, StatusCode::OK);
        let loads = body["loads"].as_array().unwrap();
        assert_eq!(loads.len(), 1);
        assert_eq!(loads[0]["load_id"], load.load_id());
        assert_eq!(loads[0]["trigger"], "startup");
        assert_eq!(loads[0]["variant"], "dflash2");
        assert_eq!(loads[0]["state"], "loading");
        assert_eq!(loads[0]["stage"], "reading_weights");
        // Same shape the engine-pool route returns.
        assert_eq!(
            serde_json::to_value(registry.list()).unwrap(),
            body["loads"].clone()
        );
        load.finish_loaded();
    }

    #[tokio::test]
    async fn activate_switches_to_full_router() {
        let registry = LoadRegistry::new();
        let switchable = SwitchableRouter::starting(registry);
        let router = Router::new().fallback_service(switchable.clone());

        let (status, body) = get_json(&router, "/v1/models").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["code"], "backend_starting");

        switchable.activate(
            Router::new()
                .route(
                    "/healthz",
                    get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }),
                )
                .route("/v1/chat/completions", post(|| async { "done" })),
        );
        let (status, body) = get_json(&router, "/healthz").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // The startup fallback is gone: unknown paths are the full router's 404.
        let (status, _) = get_json(&router, "/v1/unknown").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn dflash2_full_router_serves_loads() {
        let registry = LoadRegistry::new();
        let load = startup_load(&registry);
        let load_id = load.load_id().to_owned();
        load.enter_stage(LoadStage::WarmingUp);
        load.finish_loaded();
        let state = crate::server::responses::eof_tests::state().await;
        let router = crate::server::dflash2_router(state, Arc::clone(&registry));

        let (status, body) = get_json(&router, "/admin/api/models/loads").await;
        assert_eq!(status, StatusCode::OK);
        let loads = body["loads"].as_array().unwrap();
        assert_eq!(loads.len(), 1);
        assert_eq!(loads[0]["load_id"], load_id);
        assert_eq!(loads[0]["state"], "loaded");
        assert_eq!(
            serde_json::to_value(registry.list()).unwrap(),
            body["loads"].clone()
        );
        let (status, _) = get_json(&router, "/healthz").await;
        assert_eq!(status, StatusCode::OK);
    }
}
