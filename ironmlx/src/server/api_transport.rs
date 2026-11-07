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
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};

use super::api_error::{ApiError, ApiProtocol};
use super::request_diagnostics::FieldShape;

pub(crate) trait PublicApiRequest: DeserializeOwned {
    const API_NAME: &'static str;
    const PROTOCOL: ApiProtocol;

    fn validate_public_contract(&self) -> Result<(), RequestContractError>;

    fn diagnostic_shape(&self) -> Result<FieldShape, serde_json::Error> {
        Ok(FieldShape::Opaque)
    }
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
        let endpoint = request.uri().path().to_owned();
        // Tee the existing body only at DEBUG. Json<T> still performs the same
        // extraction, content-type/body-limit checks and DTO deserialization at
        // every log level, including duplicate-field validation.
        let captured = tracing::enabled!(tracing::Level::DEBUG)
            .then(|| std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
        let request = if let Some(buffer) = captured.clone() {
            let (parts, body) = request.into_parts();
            let body = Body::from_stream(body.into_data_stream().map(move |chunk| {
                if let Ok(bytes) = &chunk {
                    buffer
                        .lock()
                        .expect("diagnostic body buffer")
                        .extend_from_slice(bytes);
                }
                chunk
            }));
            Request::from_parts(parts, body)
        } else {
            request
        };
        let Json(value) = Json::<T>::from_request(request, state)
            .await
            .map_err(|error| {
                ApiError::invalid_request(
                    "invalid_json",
                    format!("invalid {} request: {}", T::API_NAME, error.body_text()),
                )
                .into_response(T::PROTOCOL)
            })?;
        if let Some(buffer) = captured {
            let raw = std::mem::take(&mut *buffer.lock().expect("diagnostic body buffer"));
            match value
                .diagnostic_shape()
                .and_then(|shape| shape.ignored_fields(&raw))
            {
                Ok(mut fields) => {
                    fields.sort();
                    fields.dedup();
                    if !fields.is_empty() {
                        tracing::debug!(api = T::API_NAME, %endpoint, fields = ?fields, "ignored unsupported request fields");
                    }
                }
                // A diagnostic failure must never prevent an otherwise valid
                // inference. Never print errors that could contain payloads.
                Err(_) => tracing::debug!("unable to inspect request field structure"),
            }
        }
        value.validate_public_contract().map_err(|error| {
            ApiError::invalid_request(error.code, error.message).into_response(T::PROTOCOL)
        })?;
        Ok(Self(value))
    }
}

impl PublicApiRequest for super::openai::ChatRequest {
    const API_NAME: &'static str = "Chat Completions";
    const PROTOCOL: ApiProtocol = ApiProtocol::OpenAi;

    fn diagnostic_shape(&self) -> Result<FieldShape, serde_json::Error> {
        super::openai::ChatRequest::diagnostic_shape(self)
    }

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

    fn diagnostic_shape(&self) -> Result<FieldShape, serde_json::Error> {
        super::responses::ResponsesRequest::diagnostic_shape(self)
    }

    fn validate_public_contract(&self) -> Result<(), RequestContractError> {
        self.validate_topology_contract()
            .map_err(|error| RequestContractError::new("invalid_request", error))
    }
}

impl PublicApiRequest for super::anthropic::MessagesRequest {
    const API_NAME: &'static str = "Messages";
    const PROTOCOL: ApiProtocol = ApiProtocol::Anthropic;

    fn diagnostic_shape(&self) -> Result<FieldShape, serde_json::Error> {
        super::anthropic::MessagesRequest::diagnostic_shape(self)
    }

    fn validate_public_contract(&self) -> Result<(), RequestContractError> {
        self.validate_topology_contract()
            .map_err(|error| RequestContractError::new("invalid_request", error))
    }
}

impl PublicApiRequest for super::images::ImagesGenerationRequest {
    const API_NAME: &'static str = "Images Generations";
    const PROTOCOL: ApiProtocol = ApiProtocol::OpenAi;

    fn validate_public_contract(&self) -> Result<(), RequestContractError> {
        Ok(())
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

    use std::io::{self, Write};
    use std::sync::Mutex;
    #[derive(Clone)]
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
        type Writer = Self;
        fn make_writer(&'a self) -> Self {
            self.clone()
        }
    }

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

    fn sdk_request_fixture(contents: &str) -> Value {
        serde_json::from_str(contents).expect("official SDK request fixture is valid JSON")
    }

    #[tokio::test]
    async fn pinned_official_sdk_requests_pass_the_shared_public_contract() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let router = Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/v1/responses", post(responses))
            .route("/v1/messages", post(messages))
            .with_state(dispatches.clone());

        for (path, fixture) in [
            (
                "/v1/chat/completions",
                include_str!("../../tests/fixtures/api_contract_sdk/chat_request.json"),
            ),
            (
                "/v1/responses",
                include_str!("../../tests/fixtures/api_contract_sdk/responses_request.json"),
            ),
            (
                "/v1/messages",
                include_str!("../../tests/fixtures/api_contract_sdk/messages_request.json"),
            ),
        ] {
            let response = request(router.clone(), path, sdk_request_fixture(fixture)).await;
            assert_eq!(response.status(), StatusCode::NO_CONTENT, "{path}");
        }

        assert_eq!(dispatches.load(Ordering::Relaxed), 3);
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
            serde_json::json!({"input": "hello", "max_output_tokens": 0}),
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
    async fn unknown_fields_do_not_block_any_text_protocol_or_stream_mode() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let router = Router::new()
            .route("/chat", post(chat))
            .route("/responses", post(responses))
            .route("/messages", post(messages))
            .with_state(dispatches.clone());
        let bodies = [
            (
                "/chat",
                serde_json::json!({"messages":[{"role":"user","content":[{"type":"text","text":"hi","content_extension":1}],"message_extension":true}],"stream_options":{"include_usage":true,"stream_extension":[]},"response_format":{"type":"text","format_extension":{}},"tools":[{"type":"function","tool_extension":true,"function":{"name":"f","parameters":{"type":"object","properties":{}},"function_extension":1}}]}),
            ),
            (
                "/responses",
                serde_json::json!({"input":[{"role":"user","content":[{"type":"input_text","text":"hi","content_extension":false}],"message_extension":null}],"reasoning":{"effort":"none","reasoning_extension":[]}}),
            ),
            (
                "/messages",
                serde_json::json!({"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}],"message_extension":true}],"thinking":{"type":"disabled","thinking_extension":true}}),
            ),
        ];
        for stream in [false, true] {
            for (path, mut body) in bodies.clone() {
                body["stream"] = stream.into();
                body["enable_thinking"] = false.into();
                body["store"] = true.into();
                body["functions"] = serde_json::json!({"unimplemented":"ignored"});
                body["function_call"] = serde_json::json!("auto");
                for extra in [
                    Value::Null,
                    serde_json::json!(true),
                    serde_json::json!(123),
                    serde_json::json!([1,{"secret":"unused"}]),
                    serde_json::json!({"nested":"unused"}),
                ] {
                    body["unknown_client_option"] = extra;
                    let response = request(router.clone(), path, body.clone()).await;
                    let status = response.status();
                    let data = to_bytes(response.into_body(), usize::MAX).await.unwrap();
                    assert_eq!(
                        status,
                        StatusCode::NO_CONTENT,
                        "{path}: {}",
                        String::from_utf8_lossy(&data)
                    );
                }
            }
        }
        assert_eq!(dispatches.load(Ordering::Relaxed), 30);
    }

    #[tokio::test]
    async fn ignored_fields_are_debug_only_and_never_log_request_values() {
        use tracing::instrument::WithSubscriber;
        let cases = [
            (
                "/chat",
                serde_json::json!({"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"secret":"DO_NOT_LOG_VALUE"}}]}]}),
                "/messages/0/content/0/cache_control",
            ),
            (
                "/responses",
                serde_json::json!({"input":[{"role":"user","content":[{"type":"input_text","text":"hi","cache_control":{"secret":"DO_NOT_LOG_VALUE"}}]}]}),
                "/input/0/content/0/cache_control",
            ),
            (
                "/messages",
                serde_json::json!({"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"secret":"DO_NOT_LOG_VALUE"}}]}]}),
                "/messages/0/content/0/cache_control",
            ),
        ];
        for level in [tracing::Level::INFO, tracing::Level::DEBUG] {
            for (path, mut body, nested) in cases.clone() {
                body["enable_thinking"] = false.into();
                body["client_extension"] = serde_json::json!({"secret":"DO_NOT_LOG_VALUE"});
                let capture = Capture(Arc::new(Mutex::new(Vec::new())));
                let subscriber = tracing_subscriber::fmt()
                    .without_time()
                    .with_ansi(false)
                    .with_max_level(level)
                    .with_writer(capture.clone())
                    .finish();
                let router = Router::new()
                    .route("/chat", post(chat))
                    .route("/responses", post(responses))
                    .route("/messages", post(messages))
                    .with_state(Arc::new(AtomicUsize::new(0)));
                let response = request(router, path, body)
                    .with_subscriber(subscriber)
                    .await;
                assert_eq!(response.status(), StatusCode::NO_CONTENT);
                let output = String::from_utf8(capture.0.lock().unwrap().clone()).unwrap();
                assert!(!output.contains("DO_NOT_LOG_VALUE"));
                assert!(!output.contains("unable to inspect"), "{output}");
                if level == tracing::Level::DEBUG {
                    assert_eq!(
                        output.matches("ignored unsupported request fields").count(),
                        1
                    );
                    assert!(
                        output.contains("/enable_thinking")
                            && output.contains("/client_extension")
                            && output.contains(nested)
                            && output.contains(path),
                        "{output}"
                    );
                } else {
                    assert!(!output.contains("ignored unsupported request fields"));
                }
            }
        }
    }

    #[tokio::test]
    async fn duplicate_deferred_fields_preserve_http_validation_and_debug_paths() {
        use tracing::instrument::WithSubscriber;
        let cases = [
            (
                "/responses",
                r#"{"input":"hi","aaa_extension":{"secret":"DO_NOT_LOG_VALUE"},"text":{"format":"DO_NOT_LOG_VALUE","format":{"type":"text","nested_unknown":true}},"zzz_extension":true}"#,
                "/text/format/nested_unknown",
                true,
            ),
            (
                "/responses",
                r#"{"input":[{"role":"user","content":[],"content":[{"type":"input_text","text":"hi","nested_unknown":true}]}],"aaa_extension":true,"zzz_extension":true}"#,
                "/input/0/content/0/nested_unknown",
                true,
            ),
            (
                "/chat",
                r#"{"messages":[],"tool_choice":{"type":"function","function":null,"function":{"name":"f","nested_unknown":true}},"aaa_extension":true,"zzz_extension":true}"#,
                "/tool_choice/function/nested_unknown",
                true,
            ),
            // Ordinary DTO duplicate fields must retain their original rejection.
            (
                "/responses",
                r#"{"input":"hi","max_output_tokens":8,"max_output_tokens":16,"aaa_extension":true}"#,
                "",
                false,
            ),
            (
                "/messages",
                r#"{"messages":[],"thinking":{"type":"disabled"},"thinking":{"type":"disabled"},"aaa_extension":true}"#,
                "",
                false,
            ),
        ];
        for level in [tracing::Level::INFO, tracing::Level::DEBUG] {
            for (path, body, nested, accepted) in cases {
                let capture = Capture(Arc::new(Mutex::new(Vec::new())));
                let subscriber = tracing_subscriber::fmt()
                    .without_time()
                    .with_ansi(false)
                    .with_max_level(level)
                    .with_writer(capture.clone())
                    .finish();
                let router = Router::new()
                    .route("/chat", post(chat))
                    .route("/responses", post(responses))
                    .route("/messages", post(messages))
                    .with_state(Arc::new(AtomicUsize::new(0)));
                let response = raw_request(router, path, body)
                    .with_subscriber(subscriber)
                    .await;
                assert_eq!(
                    response.status(),
                    if accepted {
                        StatusCode::NO_CONTENT
                    } else {
                        StatusCode::BAD_REQUEST
                    },
                    "{path}: {body}"
                );
                let output = String::from_utf8(capture.0.lock().unwrap().clone()).unwrap();
                assert!(!output.contains("DO_NOT_LOG_VALUE"));
                assert!(!output.contains("unable to inspect"), "{output}");
                if accepted && level == tracing::Level::DEBUG {
                    assert_eq!(
                        output.matches("ignored unsupported request fields").count(),
                        1
                    );
                    for field in ["/aaa_extension", nested, "/zzz_extension"] {
                        assert!(output.contains(field), "{output}");
                    }
                } else {
                    assert!(!output.contains("ignored unsupported request fields"));
                }
            }
        }
    }

    #[tokio::test]
    async fn debug_diagnostics_preserve_validation_and_skip_deep_unknown_values() {
        use tracing::instrument::WithSubscriber;
        for level in [tracing::Level::INFO, tracing::Level::DEBUG] {
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(level)
                .with_test_writer()
                .finish();
            async {
                let router = Router::new().route("/chat", post(chat)).with_state(Arc::new(AtomicUsize::new(0)));
                for body in [
                    r#"{"messages":[],"max_tokens":1,"max_tokens":2,"extra":true}"#,
                    r#"{"messages":[{"role":"user","role":"assistant","content":"hi"}],"extra":true}"#,
                    r#"{"messages":[],"temperature":"wrong","extra":true}"#,
                    r#"{"messages":[],"max_tokens":0,"extra":true}"#,
                    r#"{"messages":[],"extra":}"#,
                ] {
                    assert_eq!(raw_request(router.clone(), "/chat", body).await.status(), StatusCode::BAD_REQUEST, "{body}");
                }
                let repeated = r#"{"messages":[],"extra":1,"extra":2}"#;
                assert_eq!(raw_request(router.clone(), "/chat", repeated).await.status(), StatusCode::NO_CONTENT);
                for number in ["1e9999", "999999999999999999999999999999999999999999999"] {
                    let body = format!(r#"{{"messages":[],"extra":{number}}}"#);
                    assert_eq!(raw_request(router.clone(), "/chat", body).await.status(), StatusCode::NO_CONTENT);
                }
                let private_key = r#"{"messages":[],"extra":{"$serde_json::private::RawValue":{"secret":"ignored"}}}"#;
                assert_eq!(raw_request(router.clone(), "/chat", private_key).await.status(), StatusCode::NO_CONTENT);
                let deep = format!(r#"{{"messages":[],"extra":{}0{}}}"#, "[".repeat(200), "]".repeat(200));
                assert_eq!(raw_request(router, "/chat", deep).await.status(), StatusCode::NO_CONTENT);
            }.with_subscriber(subscriber).await;
        }
    }

    #[tokio::test]
    async fn debug_diagnostics_preserve_content_type_and_body_limits() {
        use tracing::instrument::WithSubscriber;
        for level in [tracing::Level::INFO, tracing::Level::DEBUG] {
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(level)
                .with_test_writer()
                .finish();
            async {
                let router = Router::new()
                    .route("/chat", post(chat))
                    .layer(axum::extract::DefaultBodyLimit::max(64))
                    .with_state(Arc::new(AtomicUsize::new(0)));
                for (content_type, body) in [
                    ("text/plain", r#"{"messages":[]}"#.to_owned()),
                    (
                        "application/json",
                        format!(r#"{{"messages":[],"extra":"{}"}}"#, "x".repeat(128)),
                    ),
                ] {
                    let response = router
                        .clone()
                        .oneshot(
                            Request::builder()
                                .method("POST")
                                .uri("/chat")
                                .header("content-type", content_type)
                                .body(Body::from(body))
                                .unwrap(),
                        )
                        .await
                        .unwrap();
                    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                    let data = to_bytes(response.into_body(), usize::MAX).await.unwrap();
                    let error: Value = serde_json::from_slice(&data).unwrap();
                    assert_eq!(error["error"]["code"], "invalid_json");
                }
            }
            .with_subscriber(subscriber)
            .await;
        }
    }

    #[tokio::test]
    async fn unknown_fields_never_mask_invalid_supported_parameters() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let router = Router::new()
            .route("/chat", post(chat))
            .with_state(dispatches.clone());
        for bad in [
            serde_json::json!({"temperature": "wrong"}),
            serde_json::json!({"temperature": 3}),
            serde_json::json!({"max_completion_tokens":0}),
            serde_json::json!({"frequency_penalty":3}),
            serde_json::json!({"response_format":{"type":"not_a_format"}}),
        ] {
            let mut body =
                serde_json::json!({"messages":[],"unknown_client_option":{"nested":"ignored"}});
            body.as_object_mut()
                .unwrap()
                .extend(bad.as_object().unwrap().clone());
            assert_eq!(
                request(router.clone(), "/chat", body).await.status(),
                StatusCode::BAD_REQUEST
            );
        }
        assert_eq!(dispatches.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn ignored_field_observation_never_keeps_values() {
        let body = serde_json::json!({"messages":[{"role":"user","content":"hi","client_metadata":{"secret":"DO_NOT_LOG_VALUE"}}],"enable_thinking":false,"another_option":[1,2]}).to_string();
        let request: super::super::openai::ChatRequest = serde_json::from_str(&body).unwrap();
        let fields = request
            .diagnostic_shape()
            .unwrap()
            .ignored_fields(body.as_bytes())
            .unwrap();
        assert!(fields.iter().any(|p| p == "/enable_thinking"));
        assert!(fields.iter().any(|p| p.contains("client_metadata")));
        assert!(!format!("{fields:?}").contains("DO_NOT_LOG_VALUE"));
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
