//! `GET /admin/api/metrics` (one JSON snapshot) and
//! `GET /admin/api/metrics/stream` (SSE) over a [`RuntimeMetricsHub`].

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::get;
use axum::{Json, Router};
use ironmlx_runtime::core::runtime_activity::{
    ActivityEvent, HistoryResponse, RecentRequestsResponse,
};
use ironmlx_runtime::core::runtime_metrics::{RuntimeMetricsHub, RuntimeMetricsSnapshot};
use tokio_stream::wrappers::{BroadcastStream, WatchStream};
use tokio_stream::{Stream, StreamExt};

const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(15);

/// Both metrics endpoints, with their state already applied: merge into a
/// router after its own `.with_state(...)`.
pub(crate) fn routes(hub: Arc<RuntimeMetricsHub>) -> Router {
    Router::new()
        .route("/admin/api/metrics", get(metrics_handler))
        .route("/admin/api/metrics/stream", get(metrics_stream_handler))
        .route("/admin/api/metrics/history", get(history_handler))
        .route("/admin/api/requests/recent", get(recent_handler))
        .with_state(hub)
}

async fn metrics_handler(
    State(hub): State<Arc<RuntimeMetricsHub>>,
) -> Json<RuntimeMetricsSnapshot> {
    Json(hub.snapshot_now())
}

async fn history_handler(State(hub): State<Arc<RuntimeMetricsHub>>) -> Json<HistoryResponse> {
    Json(hub.activity().history())
}

async fn recent_handler(State(hub): State<Arc<RuntimeMetricsHub>>) -> Json<RecentRequestsResponse> {
    Json(hub.activity().recent())
}

/// The current snapshot first, then every published change, interleaved with
/// `history` / `request` activity events (lagged receivers skip ahead
/// silently; the client notices the gap and refetches). Ends if the
/// sampling task dies, so the client reconnects instead of showing a frozen
/// frame; its new `subscribe` restarts the task.
async fn metrics_stream_handler(
    State(hub): State<Arc<RuntimeMetricsHub>>,
) -> Sse<impl Stream<Item = Result<Event, axum::Error>>> {
    let stopped = hub.sampling_stopped();
    // Subscribe before the first snapshot so nothing between open and the
    // first poll is lost.
    let activity = BroadcastStream::new(hub.activity().subscribe()).filter_map(|event| {
        let event = event.ok()?;
        match event {
            ActivityEvent::History(point) => Event::default().event("history").json_data(&point),
            ActivityEvent::Request(request) => {
                Event::default().event("request").json_data(&request)
            }
        }
        .ok()
        .map(Ok)
    });
    let snapshots = WatchStream::new(hub.subscribe());
    let events = snapshots.map(|snapshot| Event::default().event("metrics").json_data(&*snapshot));
    let events = futures::StreamExt::take_until(futures::stream::select(events, activity), stopped);
    Sse::new(events).keep_alive(KeepAlive::new().interval(KEEP_ALIVE_INTERVAL))
}

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::http::{header, Request, StatusCode};
    use ironmlx_runtime::core::runtime_activity::{
        ActivitySample, FinishedRequest, RequestOutcome, RuntimeActivityRecorder,
    };
    use ironmlx_runtime::core::runtime_metrics::{MemorySampling, RuntimeMetricsHub};
    use tokio_stream::StreamExt;
    use tower::ServiceExt;

    use super::routes;

    fn finished(model: &'static str) -> FinishedRequest<'static> {
        FinishedRequest {
            model,
            endpoint: Some("/v1/chat/completions"),
            input_tokens: 10,
            output_tokens: 20,
            ttft: Some(std::time::Duration::from_millis(100)),
            decode_duration: Some(std::time::Duration::from_secs(1)),
            duration: std::time::Duration::from_secs(2),
            outcome: RequestOutcome::Completed,
        }
    }

    async fn get_json(hub: std::sync::Arc<RuntimeMetricsHub>, uri: &str) -> serde_json::Value {
        let response = routes(hub)
            .oneshot(Request::get(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        serde_json::from_slice(
            &to_bytes(response.into_body(), 4 * 1024 * 1024)
                .await
                .unwrap(),
        )
        .unwrap()
    }

    /// Reads chunks until `needle` shows up in the accumulated text.
    async fn read_until(
        body: &mut (impl tokio_stream::Stream<Item = Result<axum::body::Bytes, axum::Error>> + Unpin),
        needle: &str,
    ) -> String {
        let mut text = String::new();
        while !text.contains(needle) {
            let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), body.next())
                .await
                .unwrap_or_else(|_| panic!("timed out waiting for {needle}; got {text}"))
                .expect("stream open")
                .expect("chunk");
            text.push_str(&String::from_utf8(chunk.to_vec()).unwrap());
        }
        text
    }

    fn close(recorder: &RuntimeActivityRecorder, t: u64) {
        recorder.close_bucket(t, ActivitySample::default());
    }

    #[tokio::test]
    async fn history_route_returns_tiers() {
        let hub = RuntimeMetricsHub::new(MemorySampling::Cached);
        close(hub.activity(), 5_000);
        let body = get_json(hub, "/admin/api/metrics/history").await;
        assert_eq!(body["schema_version"], 1);
        assert_eq!(body["bucket_ms"]["5s"], 5_000);
        for tier in ["5s", "30s", "5m"] {
            assert!(body["points"][tier].is_array(), "{body}");
        }
        assert_eq!(body["points"]["5s"][0]["t"], 5_000);
    }

    #[tokio::test]
    async fn recent_route_returns_newest_first() {
        let hub = RuntimeMetricsHub::new(MemorySampling::Cached);
        hub.activity().record_finished(finished("first"), 1_000);
        hub.activity().record_finished(finished("second"), 2_000);
        let body = get_json(hub, "/admin/api/requests/recent").await;
        assert_eq!(body["schema_version"], 1);
        assert_eq!(body["requests"][0]["model"], "second");
        assert_eq!(body["requests"][1]["model"], "first");
    }

    #[tokio::test]
    async fn stream_sends_history_and_request_events() {
        let hub = RuntimeMetricsHub::new(MemorySampling::Cached);
        let response = routes(hub.clone())
            .oneshot(
                Request::get("/admin/api/metrics/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = response.into_body().into_data_stream();
        read_until(&mut body, "event: metrics").await;
        close(hub.activity(), 5_000);
        hub.activity().record_finished(finished("m"), 6_000);
        let text = read_until(&mut body, "event: request").await;
        let text = if text.contains("event: history") {
            text
        } else {
            text + &read_until(&mut body, "event: history").await
        };
        assert!(text.contains("\"t\":5000"), "{text}");
        assert!(text.contains("\"model\":\"m\""), "{text}");
    }

    #[tokio::test]
    async fn lagged_activity_events_are_skipped() {
        let hub = RuntimeMetricsHub::new(MemorySampling::Cached);
        let response = routes(hub.clone())
            .oneshot(
                Request::get("/admin/api/metrics/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = response.into_body().into_data_stream();
        for i in 1..=300u64 {
            close(hub.activity(), i * 5_000);
        }
        // Overflowed receiver: the stream keeps going and the tail is delivered.
        read_until(&mut body, "event: metrics").await;
        close(hub.activity(), 999_000_000);
        read_until(&mut body, "\"t\":999000000").await;
    }

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
