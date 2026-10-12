//! OpenAI Decisions API on the shared model pool, Laya worker and listeners.
use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use ironmlx_decision::{
    contract::MULTILINGUAL_MODEL_ID,
    openai::{parse_request, DecisionInputError},
};
use ironmlx_runtime::core::{
    decision_execution::DecisionExecutionError,
    engine_pool::{EnginePoolState, EngineRegistryError, EngineVariant},
};
use serde_json::Value;

use super::api_error::{ApiError, ApiProtocol};

pub(super) async fn decisions(
    State(pool): State<EnginePoolState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    decisions_with_pool(pool, body).await
}

pub(super) async fn decisions_with_pool(
    pool: EnginePoolState,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    match run(pool, body).await {
        Ok(response) => response,
        Err(error) => error.into_response(ApiProtocol::OpenAi),
    }
}

async fn run(
    pool: EnginePoolState,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(json_rejection)?;
    let prepared = parse_request(&body).map_err(input_error)?;
    let is_decision_model = pool
        .is_decision_model(Some(&prepared.model))
        .await
        .map_err(model_error)?;
    if is_decision_model && prepared.model != MULTILINGUAL_MODEL_ID {
        return Err(unserved_model(&prepared.model));
    }
    if !is_decision_model {
        return Err(task_mismatch(&prepared.model));
    }
    let (_, lease) = pool
        .resolve_engine(Some(&prepared.model))
        .await
        .map_err(load_error)?;
    let EngineVariant::Decision(runtime) = lease.engine() else {
        return Err(task_mismatch(&prepared.model));
    };
    let runtime = runtime.clone();
    let scores = runtime
        .score(prepared.scoring.clone(), lease)
        .await
        .map_err(execution_error)?;
    let decision = prepared
        .response(&scores)
        .map_err(|error| ApiError::internal("decision_inference_failed", format!("{error:#}")))?;
    Ok(Json(decision).into_response())
}

/// Oversized bodies keep the shared 413 code; everything else is a JSON error.
pub(crate) fn json_rejection(rejection: JsonRejection) -> ApiError {
    if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return ApiError::from_status(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_body_too_large",
            "The request body exceeds the server's size limit.",
        );
    }
    ApiError::invalid_request(
        "invalid_json",
        format!("invalid Decisions request: {}", rejection.body_text()),
    )
}

pub(crate) fn input_error(error: DecisionInputError) -> ApiError {
    ApiError::invalid_request(error.code, error.message).with_param(error.param)
}

pub(crate) fn token_budget_error(message: impl Into<String>) -> ApiError {
    ApiError::invalid_request("question_token_budget_exceeded", message)
        .with_param(Some("questions".to_owned()))
}

/// Decisions serves only the Laya checkpoint ID, matching System One.
pub(crate) fn unserved_model(model: &str) -> ApiError {
    ApiError::from_status(
        StatusCode::NOT_FOUND,
        "model_not_found",
        format!(
            "model `{model}` is not served by the Decisions API; use `{MULTILINGUAL_MODEL_ID}`"
        ),
    )
    .with_param(Some("model".to_owned()))
}

fn task_mismatch(model: &str) -> ApiError {
    ApiError::invalid_request(
        "model_task_mismatch",
        format!("model `{model}` does not support the Decisions API"),
    )
    .with_param(Some("model".to_owned()))
}

fn model_error(error: anyhow::Error) -> ApiError {
    ApiError::engine_resolution(error).with_param(Some("model".to_owned()))
}

fn load_error(error: anyhow::Error) -> ApiError {
    if error.downcast_ref::<EngineRegistryError>().is_some() {
        model_error(error)
    } else {
        ApiError::service_unavailable("model_load_failed", format!("{error:#}"))
    }
}

fn execution_error(error: DecisionExecutionError) -> ApiError {
    let message = error.to_string();
    match error {
        DecisionExecutionError::InvalidInput(_) => {
            ApiError::invalid_request("invalid_value", message)
        }
        DecisionExecutionError::TokenBudgetExceeded(_) => token_budget_error(message),
        DecisionExecutionError::QueueFull => {
            ApiError::service_unavailable("decision_queue_full", message)
        }
        DecisionExecutionError::WorkerStopped => {
            ApiError::service_unavailable("decision_worker_unavailable", message)
        }
        DecisionExecutionError::Timeout => {
            ApiError::from_status(StatusCode::GATEWAY_TIMEOUT, "decision_timeout", message)
        }
        DecisionExecutionError::WrongEngine => {
            ApiError::internal("decision_engine_mismatch", message)
        }
        DecisionExecutionError::MemoryUnavailable(_) => {
            ApiError::service_unavailable("memory_budget_exceeded", message)
        }
        DecisionExecutionError::Inference(_) => {
            ApiError::internal("decision_inference_failed", message)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn body(error: ApiError) -> (StatusCode, Option<String>, Value) {
        let response = error.into_response(ApiProtocol::OpenAi);
        let status = response.status();
        let retry = response
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .map(|value| value.to_str().unwrap().to_owned());
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, retry, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn execution_errors_have_distinct_openai_codes() {
        for (error, status, kind, code) in [
            (
                DecisionExecutionError::TokenBudgetExceeded("budget".into()),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "question_token_budget_exceeded",
            ),
            (
                DecisionExecutionError::QueueFull,
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "decision_queue_full",
            ),
            (
                DecisionExecutionError::WorkerStopped,
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "decision_worker_unavailable",
            ),
            (
                DecisionExecutionError::MemoryUnavailable("busy".into()),
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "memory_budget_exceeded",
            ),
            (
                DecisionExecutionError::Timeout,
                StatusCode::GATEWAY_TIMEOUT,
                "server_error",
                "decision_timeout",
            ),
            (
                DecisionExecutionError::Inference("boom".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "decision_inference_failed",
            ),
        ] {
            let (actual, retry, json) = body(execution_error(error)).await;
            assert_eq!(actual, status);
            assert_eq!(json["error"]["type"], kind);
            assert_eq!(json["error"]["code"], code);
            assert_eq!(
                retry.is_some(),
                status == StatusCode::SERVICE_UNAVAILABLE,
                "{code}"
            );
        }
        let (_, _, json) = body(execution_error(
            DecisionExecutionError::TokenBudgetExceeded("budget".into()),
        ))
        .await;
        assert_eq!(json["error"]["param"], "questions");
    }

    #[tokio::test]
    async fn load_and_model_errors_are_distinguished() {
        let (status, _, json) = body(load_error(
            EngineRegistryError::UnknownModel { id: "x".into() }.into(),
        ))
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json["error"]["code"], "model_not_found");
        assert_eq!(json["error"]["param"], "model");
        let (status, retry, json) = body(load_error(anyhow::anyhow!("weights missing"))).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(retry.as_deref(), Some("5"));
        assert_eq!(json["error"]["code"], "model_load_failed");
        let (status, _, json) = body(unserved_model("my-laya")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json["error"]["code"], "model_not_found");
        assert_eq!(json["error"]["param"], "model");
        let (status, _, json) = body(task_mismatch("qwen")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "model_task_mismatch");
        assert_eq!(json["error"]["param"], "model");
    }
}
