use std::{
    pin::Pin,
    task::{Context, Poll},
};

use axum::{
    async_trait,
    body::{Body, Bytes},
    extract::{FromRequest, Request},
    http::header,
    response::Response,
    Json,
};
use serde::de::DeserializeOwned;
use tokio::sync::{mpsc, watch};
use tokio_stream::{wrappers::ReceiverStream, Stream};

use super::api_error::{ApiError, ApiProtocol};

pub(crate) trait PublicApiRequest: DeserializeOwned {
    const API_NAME: &'static str;
    const PROTOCOL: ApiProtocol;

    fn validate_public_contract(&self) -> Result<(), RequestContractError>;
}

pub(crate) struct RequestContractError {
    code: &'static str,
    message: String,
}

impl RequestContractError {
    fn new(code: &'static str, error: anyhow::Error) -> Self {
        Self {
            code,
            message: format!("{error:#}"),
        }
    }
}

pub(crate) struct ApiJson<T>(pub(crate) T);

#[async_trait]
impl<T, S> FromRequest<S> for ApiJson<T>
where
    T: PublicApiRequest,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(request, state)
            .await
            .map_err(|error| {
                ApiError::invalid_request(
                    "invalid_json",
                    format!("invalid {} request: {}", T::API_NAME, error.body_text()),
                )
                .into_response(T::PROTOCOL)
            })?;
        value.validate_public_contract().map_err(|error| {
            ApiError::invalid_request(error.code, error.message).into_response(T::PROTOCOL)
        })?;
        Ok(Self(value))
    }
}

impl PublicApiRequest for super::openai::ChatRequest {
    const API_NAME: &'static str = "Chat Completions";
    const PROTOCOL: ApiProtocol = ApiProtocol::OpenAi;

    fn validate_public_contract(&self) -> Result<(), RequestContractError> {
        self.validate_sampling()
            .map_err(|error| RequestContractError::new("invalid_sampling_parameters", error))?;
        self.structured_output_format()
            .map_err(|error| RequestContractError::new("invalid_response_format", error))?;
        Ok(())
    }
}

impl PublicApiRequest for super::responses::ResponsesRequest {
    const API_NAME: &'static str = "Responses";
    const PROTOCOL: ApiProtocol = ApiProtocol::OpenAi;

    fn validate_public_contract(&self) -> Result<(), RequestContractError> {
        self.validate_topology_contract()
            .map_err(|error| RequestContractError::new("invalid_request", error))
    }
}

impl PublicApiRequest for super::anthropic::MessagesRequest {
    const API_NAME: &'static str = "Messages";
    const PROTOCOL: ApiProtocol = ApiProtocol::Anthropic;

    fn validate_public_contract(&self) -> Result<(), RequestContractError> {
        self.validate_topology_contract()
            .map_err(|error| RequestContractError::new("invalid_request", error))
    }
}

pub(crate) fn sse_response(body: Body) -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .expect("fixed SSE response headers are valid")
}

pub(crate) type SseItem = std::result::Result<Bytes, std::io::Error>;

#[derive(Clone)]
pub(crate) struct SseDisconnect {
    receiver: watch::Receiver<bool>,
}

impl SseDisconnect {
    pub(crate) fn is_cancelled(&self) -> bool {
        *self.receiver.borrow()
    }

    pub(crate) async fn cancelled(&self) {
        let mut receiver = self.receiver.clone();
        if *receiver.borrow() {
            return;
        }
        while receiver.changed().await.is_ok() {
            if *receiver.borrow_and_update() {
                return;
            }
        }
    }
}

pub(crate) struct DisconnectAwareSseReceiver {
    inner: ReceiverStream<SseItem>,
    disconnect_tx: watch::Sender<bool>,
}

impl Stream for DisconnectAwareSseReceiver {
    type Item = SseItem;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(context)
    }
}

impl Drop for DisconnectAwareSseReceiver {
    fn drop(&mut self) {
        let _ = self.disconnect_tx.send(true);
    }
}

pub(crate) fn disconnect_aware_sse_channel(
    capacity: usize,
) -> (
    mpsc::Sender<SseItem>,
    DisconnectAwareSseReceiver,
    SseDisconnect,
) {
    let (tx, rx) = mpsc::channel(capacity);
    let (disconnect_tx, disconnect_rx) = watch::channel(false);
    (
        tx,
        DisconnectAwareSseReceiver {
            inner: ReceiverStream::new(rx),
            disconnect_tx,
        },
        SseDisconnect {
            receiver: disconnect_rx,
        },
    )
}

pub(crate) fn disconnect_aware_sse_response(rx: DisconnectAwareSseReceiver) -> Response {
    sse_response(Body::from_stream(rx))
}

pub(crate) async fn recv_or_disconnect<T>(
    disconnect: &SseDisconnect,
    receiver: &mut mpsc::UnboundedReceiver<T>,
) -> Option<T> {
    tokio::select! {
        biased;
        _ = disconnect.cancelled() => None,
        event = receiver.recv() => event,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use axum::{
        body::{to_bytes, Body},
        extract::State,
        http::{header, Request, StatusCode},
        routing::post,
        Router,
    };
    use serde_json::Value;
    use tower::ServiceExt;

    use super::*;

    async fn chat(
        State(dispatches): State<Arc<AtomicUsize>>,
        ApiJson(_request): ApiJson<super::super::openai::ChatRequest>,
    ) -> StatusCode {
        dispatches.fetch_add(1, Ordering::Relaxed);
        StatusCode::NO_CONTENT
    }

    async fn responses(
        State(dispatches): State<Arc<AtomicUsize>>,
        ApiJson(_request): ApiJson<super::super::responses::ResponsesRequest>,
    ) -> StatusCode {
        dispatches.fetch_add(1, Ordering::Relaxed);
        StatusCode::NO_CONTENT
    }

    async fn messages(
        State(dispatches): State<Arc<AtomicUsize>>,
        ApiJson(_request): ApiJson<super::super::anthropic::MessagesRequest>,
    ) -> StatusCode {
        dispatches.fetch_add(1, Ordering::Relaxed);
        StatusCode::NO_CONTENT
    }

    async fn raw_request(router: Router, path: &str, body: impl Into<Body>) -> Response {
        router
            .oneshot(
                Request::post(path)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(body.into())
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn request(router: Router, path: &str, body: Value) -> Response {
        raw_request(router, path, body.to_string()).await
    }

    #[tokio::test]
    async fn public_request_contract_is_validated_before_topology_dispatch() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let router = Router::new()
            .route("/chat", post(chat))
            .route("/responses", post(responses))
            .route("/messages", post(messages))
            .with_state(dispatches.clone());

        let chat = request(
            router.clone(),
            "/chat",
            serde_json::json!({"messages": [], "temperature": 3.0}),
        )
        .await;
        assert_eq!(chat.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(chat.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "invalid_sampling_parameters");

        let responses = request(
            router.clone(),
            "/responses",
            serde_json::json!({"input": "hello", "store": true}),
        )
        .await;
        assert_eq!(responses.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(responses.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "invalid_request");

        let messages = request(
            router,
            "/messages",
            serde_json::json!({"messages": [], "top_k": 0}),
        )
        .await;
        assert_eq!(messages.status(), StatusCode::BAD_REQUEST);
        let request_id = messages.headers()["request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let body = to_bytes(messages.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["request_id"], request_id);
        assert_eq!(dispatches.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn malformed_json_is_rejected_before_topology_dispatch() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let router = Router::new()
            .route("/chat", post(chat))
            .route("/messages", post(messages))
            .with_state(dispatches.clone());

        let chat = raw_request(router.clone(), "/chat", "{").await;
        assert_eq!(chat.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(chat.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "invalid_json");

        let messages = raw_request(router, "/messages", "{").await;
        assert_eq!(messages.status(), StatusCode::BAD_REQUEST);
        let request_id = messages.headers()["request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let body = to_bytes(messages.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "invalid_json");
        assert_eq!(body["request_id"], request_id);
        assert_eq!(dispatches.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn all_topologies_share_the_same_sse_headers() {
        let response = sse_response(Body::empty());
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-cache");
    }

    #[tokio::test]
    async fn dropping_sse_body_publishes_disconnect() {
        let (_tx, rx, disconnect) = disconnect_aware_sse_channel(1);
        let response = disconnect_aware_sse_response(rx);
        assert!(!disconnect.is_cancelled());

        drop(response);

        tokio::time::timeout(std::time::Duration::from_secs(1), disconnect.cancelled())
            .await
            .expect("SSE body drop must publish disconnect");
        assert!(disconnect.is_cancelled());
    }

    #[tokio::test]
    async fn disconnect_wins_over_an_already_buffered_terminal_event() {
        let (_body_tx, body_rx, disconnect) = disconnect_aware_sse_channel(1);
        let response = disconnect_aware_sse_response(body_rx);
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        event_tx.send("terminal").unwrap();

        drop(response);

        assert_eq!(recv_or_disconnect(&disconnect, &mut event_rx).await, None);
    }
}
