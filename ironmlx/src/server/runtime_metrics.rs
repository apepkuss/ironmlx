//! `GET /admin/api/metrics` (one JSON snapshot) and
//! `GET /admin/api/metrics/stream` (SSE) over a [`RuntimeMetricsHub`].

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::get;
use axum::{Json, Router};
use ironmlx_runtime::core::runtime_metrics::{RuntimeMetricsHub, RuntimeMetricsSnapshot};
use tokio_stream::wrappers::WatchStream;
use tokio_stream::{Stream, StreamExt};

const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(15);

/// Both metrics endpoints, with their state already applied: merge into a
/// router after its own `.with_state(...)`.
pub(crate) fn routes(hub: Arc<RuntimeMetricsHub>) -> Router {
    Router::new()
        .route("/admin/api/metrics", get(metrics_handler))
        .route("/admin/api/metrics/stream", get(metrics_stream_handler))
        .with_state(hub)
}

async fn metrics_handler(
    State(hub): State<Arc<RuntimeMetricsHub>>,
) -> Json<RuntimeMetricsSnapshot> {
    Json(hub.snapshot_now())
}

/// The current snapshot first, then every published change. Ends if the
/// sampling task dies, so the client reconnects instead of showing a frozen
/// frame; its new `subscribe` restarts the task.
async fn metrics_stream_handler(
    State(hub): State<Arc<RuntimeMetricsHub>>,
) -> Sse<impl Stream<Item = Result<Event, axum::Error>>> {
    let stopped = hub.sampling_stopped();
    let snapshots = WatchStream::new(hub.subscribe());
    let events = snapshots.map(|snapshot| Event::default().event("metrics").json_data(&*snapshot));
    let events = futures::StreamExt::take_until(events, stopped);
    Sse::new(events).keep_alive(KeepAlive::new().interval(KEEP_ALIVE_INTERVAL))
}

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::http::{header, Request, StatusCode};
    use ironmlx_runtime::core::runtime_metrics::{MemorySampling, RuntimeMetricsHub};
    use tokio_stream::StreamExt;
    use tower::ServiceExt;

    use super::routes;

    #[tokio::test]
    async fn metrics_route_returns_schema_v1() {
        let app = routes(RuntimeMetricsHub::new(MemorySampling::Cached));
        let response = app
            .oneshot(
                Request::get("/admin/api/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(body["schema_version"], 1);
        assert!(body["models"].is_array(), "{body}");
    }

    #[tokio::test]
    async fn metrics_stream_sends_snapshot_first() {
        let app = routes(RuntimeMetricsHub::new(MemorySampling::Cached));
        let response = app
            .oneshot(
                Request::get("/admin/api/metrics/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let mut body = response.into_body().into_data_stream();
        let first = tokio::time::timeout(std::time::Duration::from_secs(5), body.next())
            .await
            .expect("first event arrives without waiting for a tick")
            .expect("stream open")
            .expect("chunk");
        let first = String::from_utf8(first.to_vec()).unwrap();
        assert!(first.contains("event: metrics"), "{first}");
        assert!(first.contains("\"schema_version\":1"), "{first}");
    }
}
