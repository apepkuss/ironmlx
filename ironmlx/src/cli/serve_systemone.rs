use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use axum::{
    extract::{rejection::JsonRejection, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Args;
use ironmlx_decision::{
    openai::parse_request, system_one_response, DecisionRequest, DecisionResponse, Laya, Scores,
    ScoringRequest, TokenBudgetExceeded,
};
use serde_json::{json, Value};

use crate::server::api_error::{ApiError, ApiProtocol};
use tokio::sync::{mpsc, oneshot};

const MODEL_ID: &str = "aac6fef/laya-multilingual-mlx";

#[derive(Args, Debug)]
pub struct ServeSystemoneArgs {
    /// Local directory containing the Laya multilingual checkpoint.
    #[arg(long)]
    model_dir: PathBuf,
    /// Bind address. Defaults to loopback; an API key is always required.
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    bind: IpAddr,
    /// Dedicated System One API port.
    #[arg(long, default_value_t = 8767)]
    port: u16,
}

struct Work {
    request: ScoringRequest,
    reply: oneshot::Sender<anyhow::Result<Scores>>,
}

#[derive(Clone)]
struct ApiState {
    key: Arc<str>,
    worker: mpsc::Sender<Work>,
}

pub fn run(args: ServeSystemoneArgs) -> Result<()> {
    let key = std::env::var("IRONMLX_SYSTEMONE_API_KEY")
        .context("set IRONMLX_SYSTEMONE_API_KEY for the System One service")?;
    if key.is_empty() || key.trim() != key {
        bail!("IRONMLX_SYSTEMONE_API_KEY must be nonempty without surrounding whitespace");
    }
    let (worker, mut receiver) = mpsc::channel::<Work>(16);
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
    let model_dir = args.model_dir;
    std::thread::Builder::new()
        .name("laya-systemone".into())
        .spawn(move || {
            let model = match Laya::load(&model_dir) {
                Ok(model) => {
                    let _ = started_tx.send(Ok(()));
                    model
                }
                Err(error) => {
                    let _ = started_tx.send(Err(format!("{error:#}")));
                    return;
                }
            };
            while let Some(work) = receiver.blocking_recv() {
                let _ = work.reply.send(model.score(&work.request));
            }
        })
        .context("starting Laya inference worker")?;
    started_rx
        .recv()
        .context("waiting for Laya model load")?
        .map_err(anyhow::Error::msg)?;

    let state = ApiState {
        key: Arc::from(key),
        worker,
    };
    let address = SocketAddr::new(args.bind, args.port);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("creating System One runtime")?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(address)
            .await
            .with_context(|| format!("binding System One at {address}"))?;
        tracing::info!(address = %listener.local_addr()?, "System One service ready");
        axum::serve(listener, router(state))
            .await
            .context("serving System One")
    })
}

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/healthz", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(system_one))
        .route("/v1/decisions", post(decisions))
        .with_state(state)
}

fn authorized(headers: &HeaderMap, key: &str) -> bool {
    use subtle::ConstantTimeEq;
    let Some(candidate) = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
    else {
        return false;
    };
    candidate.as_bytes().ct_eq(key.as_bytes()).into()
}

fn api_error(status: StatusCode, message: &str) -> (StatusCode, Json<Value>) {
    (status, Json(json!({"detail": message})))
}

async fn models(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if !authorized(&headers, &state.key) {
        return Err(api_error(StatusCode::UNAUTHORIZED, "invalid API key"));
    }
    Ok(Json(json!({"models":[{
        "name": MODEL_ID,
        "description": "Local multilingual Laya decision model (choice, score, noul)",
        "release_date": "2026-09-19"
    }]})))
}

async fn system_one(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Result<Json<DecisionRequest>, JsonRejection>,
) -> Result<Json<DecisionResponse>, (StatusCode, Json<Value>)> {
    if !authorized(&headers, &state.key) {
        return Err(api_error(StatusCode::UNAUTHORIZED, "invalid API key"));
    }
    let Json(request) =
        body.map_err(|failure| api_error(StatusCode::UNPROCESSABLE_ENTITY, &failure.body_text()))?;
    let questions = request
        .validate()
        .map_err(|failure| api_error(StatusCode::UNPROCESSABLE_ENTITY, &failure.to_string()))?;
    let scoring = ScoringRequest {
        state: request.state,
        questions: questions.iter().map(|(_, q)| q.clone()).collect(),
    };
    match score(&state, scoring).await {
        Ok(scores) => system_one_response(&questions, &scores)
            .map(Json)
            .map_err(|_| {
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "decision inference failed",
                )
            }),
        Err(WorkFailure::TokenBudget(message)) => {
            Err(api_error(StatusCode::UNPROCESSABLE_ENTITY, &message))
        }
        Err(WorkFailure::Inference) => Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "decision inference failed",
        )),
        Err(WorkFailure::Unavailable) => Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "decision worker unavailable",
        )),
    }
}

enum WorkFailure {
    TokenBudget(String),
    Inference,
    Unavailable,
}

async fn score(state: &ApiState, request: ScoringRequest) -> Result<Scores, WorkFailure> {
    let (reply, result) = oneshot::channel();
    state
        .worker
        .send(Work { request, reply })
        .await
        .map_err(|_| WorkFailure::Unavailable)?;
    match result.await {
        Ok(Ok(scores)) => Ok(scores),
        Ok(Err(error)) if error.is::<TokenBudgetExceeded>() => {
            Err(WorkFailure::TokenBudget(error.to_string()))
        }
        Ok(Err(error)) => {
            tracing::warn!(error = %format!("{error:#}"), "decision inference failed");
            Err(WorkFailure::Inference)
        }
        Err(_) => Err(WorkFailure::Unavailable),
    }
}

/// OpenAI Decisions API with the same key and worker as System One.
async fn decisions(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let result = async {
        if !authorized(&headers, &state.key) {
            return Err(ApiError::from_status(
                StatusCode::UNAUTHORIZED,
                "auth_invalid",
                "A valid Bearer API key is required.",
            ));
        }
        let Json(body) = body.map_err(crate::server::decisions::json_rejection)?;
        let prepared = parse_request(&body).map_err(crate::server::decisions::input_error)?;
        if prepared.model != MODEL_ID {
            return Err(ApiError::from_status(
                StatusCode::NOT_FOUND,
                "model_not_found",
                format!(
                    "model `{}` is not served here; use `{MODEL_ID}`",
                    prepared.model
                ),
            )
            .with_param(Some("model".to_owned())));
        }
        let scores =
            score(&state, prepared.scoring.clone())
                .await
                .map_err(|failure| match failure {
                    WorkFailure::TokenBudget(message) => {
                        crate::server::decisions::token_budget_error(message)
                    }
                    WorkFailure::Inference => {
                        ApiError::internal("decision_inference_failed", "decision inference failed")
                    }
                    WorkFailure::Unavailable => ApiError::service_unavailable(
                        "decision_worker_unavailable",
                        "decision worker unavailable",
                    ),
                })?;
        prepared.response(&scores).map_err(|_| {
            ApiError::internal("decision_inference_failed", "decision inference failed")
        })
    }
    .await;
    match result {
        Ok(decision) => Json(decision).into_response(),
        Err(error) => error.into_response(ApiProtocol::OpenAi),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use tower::ServiceExt;

    fn test_state() -> (ApiState, mpsc::Receiver<Work>) {
        let (worker, receiver) = mpsc::channel(1);
        (
            ApiState {
                key: Arc::from("test-key"),
                worker,
            },
            receiver,
        )
    }

    #[tokio::test]
    async fn models_require_auth_and_use_typesafe_list_shape() {
        let (state, _) = test_state();
        let app = router(state);
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        let allowed = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header(header::AUTHORIZATION, "Bearer test-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(allowed.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(allowed.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["models"][0]["name"], MODEL_ID);
        assert!(body["models"][0]["release_date"].is_string());
    }

    #[tokio::test]
    async fn system_one_validates_before_dispatch_and_preserves_response() {
        let (state, mut receiver) = test_state();
        let app = router(state);
        let invalid = app.clone().oneshot(Request::builder().method("POST")
            .uri("/v1/systemone")
            .header(header::AUTHORIZATION, "Bearer test-key")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"model":"jev-latest","state":"x","questions":{"q":{"type":"noul","instructions":"Yes?"}}}"#))
            .unwrap()).await.unwrap();
        assert_eq!(invalid.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(receiver.try_recv().is_err());

        let worker = tokio::spawn(async move {
            let work = receiver.recv().await.unwrap();
            assert_eq!(work.request.state, json!("x"));
            work.reply
                .send(Ok(Scores {
                    probabilities: vec![vec![0.25, 0.75]],
                    input_tokens: 9,
                }))
                .unwrap();
        });
        let valid = app.oneshot(Request::builder().method("POST")
            .uri("/v1/systemone")
            .header(header::AUTHORIZATION, "Bearer test-key")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(format!(r#"{{"model":"{MODEL_ID}","state":"x","questions":{{"q":{{"type":"noul","instructions":"Yes?"}}}}}}"#)))
            .unwrap()).await.unwrap();
        worker.await.unwrap();
        assert_eq!(valid.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(valid.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["answers"]["q"]["noul"], 0.75);
        assert_eq!(body["usage"]["input_tokens"], 9);
    }

    async fn post_decisions(app: &Router, key: Option<&str>, body: Value) -> (StatusCode, Value) {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/decisions")
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(key) = key {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        let response = app
            .clone()
            .oneshot(builder.body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn decisions_authenticate_validate_and_share_the_worker() {
        let (state, mut receiver) = test_state();
        let app = router(state);
        let request = json!({"model": MODEL_ID, "input": "x", "questions": [
            {"type": "predicate", "instructions": "Yes?"},
            {"type": "choice", "name": "flag", "instructions": "Which?",
             "choices": [{"value": true}, {"value": "true"}]}
        ]});
        for key in [None, Some("wrong")] {
            let (status, body) = post_decisions(&app, key, request.clone()).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
            assert_eq!(body["error"]["type"], "authentication_error");
            assert_eq!(body["error"]["code"], "auth_invalid");
        }
        let mut wrong_model = request.clone();
        wrong_model["model"] = json!("gpt-6-luna");
        let (status, body) = post_decisions(&app, Some("test-key"), wrong_model).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
        let (status, body) = post_decisions(
            &app,
            Some("test-key"),
            json!({"model": MODEL_ID, "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}],
                "questions": [{"type": "predicate", "instructions": "?"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "unsupported_value");
        assert_eq!(body["error"]["param"], "input[0].content[0].type");
        assert!(receiver.try_recv().is_err());

        let worker = tokio::spawn(async move {
            let work = receiver.recv().await.unwrap();
            assert_eq!(work.request.state, json!("x"));
            assert_eq!(work.request.questions.len(), 2);
            work.reply
                .send(Ok(Scores {
                    probabilities: vec![vec![0.4, 0.6], vec![0.7, 0.3]],
                    input_tokens: 21,
                }))
                .unwrap();
            let work = receiver.recv().await.unwrap();
            work.reply.send(Err(TokenBudgetExceeded.into())).unwrap();
        });
        let (status, body) = post_decisions(&app, Some("test-key"), request.clone()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["model"], MODEL_ID);
        assert_eq!(
            body["answers"][0],
            json!({"type": "predicate", "name": null, "probability": 0.6})
        );
        assert_eq!(body["answers"][1]["choice"], json!(true));
        assert_eq!(body["answers"][1]["name"], "flag");
        assert_eq!(body["usage"]["total_tokens"], 21);
        let (status, body) = post_decisions(&app, Some("test-key"), request).await;
        worker.await.unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "question_token_budget_exceeded");
    }

    async fn post_raw(
        app: &Router,
        path: &str,
        key: Option<&str>,
        body: String,
    ) -> (StatusCode, Value) {
        let mut builder = Request::builder()
            .method("POST")
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(key) = key {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        let response = app
            .clone()
            .oneshot(builder.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn decisions_distinguish_oversized_bodies_from_invalid_json() {
        let (state, mut receiver) = test_state();
        let app = router(state);
        let oversized = json!({"model": MODEL_ID, "input": "x".repeat(2 * 1024 * 1024 + 1),
            "questions": [{"type": "predicate", "instructions": "?"}]})
        .to_string();
        let (status, body) =
            post_raw(&app, "/v1/decisions", Some("test-key"), oversized.clone()).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "request_body_too_large");
        let (status, body) = post_raw(&app, "/v1/decisions", None, oversized).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
        let (status, body) = post_raw(
            &app,
            "/v1/decisions",
            Some("test-key"),
            "{\"model\":".into(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "invalid_json");
        // System One keeps its existing 422 contract for both cases.
        let oversized = json!({"model": MODEL_ID, "state": "x".repeat(2 * 1024 * 1024 + 1),
            "questions": {"q": {"type": "noul", "instructions": "?"}}})
        .to_string();
        let (status, body) = post_raw(&app, "/v1/systemone", Some("test-key"), oversized).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body["detail"].is_string());
        assert!(receiver.try_recv().is_err());
    }
}
