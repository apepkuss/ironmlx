use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use axum::{
    extract::{rejection::JsonRejection, State},
    http::{header, HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use clap::Args;
use ironmlx_decision::{DecisionRequest, DecisionResponse, Laya};
use serde_json::{json, Value};
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
    request: DecisionRequest,
    reply: oneshot::Sender<Result<DecisionResponse, String>>,
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
                let result = model
                    .predict(&work.request)
                    .map_err(|error| format!("{error:#}"));
                let _ = work.reply.send(result);
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
    request
        .validate()
        .map_err(|failure| api_error(StatusCode::UNPROCESSABLE_ENTITY, &failure.to_string()))?;
    let (reply, result) = oneshot::channel();
    state
        .worker
        .send(Work { request, reply })
        .await
        .map_err(|_| {
            api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "decision worker unavailable",
            )
        })?;
    match result.await {
        Ok(Ok(response)) => Ok(Json(response)),
        Ok(Err(message)) if message.contains("question options exceed") => {
            Err(api_error(StatusCode::UNPROCESSABLE_ENTITY, &message))
        }
        Ok(Err(_)) => Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "decision inference failed",
        )),
        Err(_) => Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "decision worker unavailable",
        )),
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
            assert_eq!(work.request.model, MODEL_ID);
            let mut answers = serde_json::Map::new();
            answers.insert("q".into(), json!({"type":"noul","noul":0.75}));
            work.reply
                .send(Ok(DecisionResponse {
                    model: MODEL_ID.into(),
                    answers,
                    usage: ironmlx_decision::contract::Usage {
                        input_tokens: 9,
                        output_tokens: 0,
                    },
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
}
