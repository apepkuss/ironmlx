//! System One transport on the App's managed model pool and network listener.
use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use ironmlx_decision::{DecisionRequest, DecisionResponse};
use ironmlx_runtime::core::{
    decision_execution::DecisionExecutionError,
    engine_pool::{EnginePoolState, EngineVariant},
};
use serde::Serialize;
use serde_json::json;

#[derive(Debug, Serialize)]
pub(super) struct SystemOneModelInfo {
    name: String,
    description: &'static str,
    release_date: &'static str,
}
pub(super) async fn model_list(pool: &EnginePoolState) -> Vec<SystemOneModelInfo> {
    pool.decision_model_ids()
        .await
        .into_iter()
        .map(|name| SystemOneModelInfo {
            name,
            description: "Local multilingual Laya decision model (choice, score, noul)",
            release_date: "2026-09-19",
        })
        .collect()
}
fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"detail": message.to_string()}))).into_response()
}
pub(super) async fn system_one(
    State(pool): State<EnginePoolState>,
    body: Result<Json<DecisionRequest>, JsonRejection>,
) -> Response {
    system_one_with_pool(pool, body).await
}
pub(super) async fn system_one_with_pool(
    pool: EnginePoolState,
    body: Result<Json<DecisionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(e) => return error(StatusCode::UNPROCESSABLE_ENTITY, e.body_text()),
    };
    if let Err(e) = request.validate() {
        return error(StatusCode::UNPROCESSABLE_ENTITY, e);
    }
    match pool.is_decision_model(Some(&request.model)).await {
        Ok(true) => {}
        _ => {
            return error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "decision model is not registered; download and load it in IronMLX App",
            )
        }
    }
    let (_, lease) = match pool.resolve_engine(Some(&request.model)).await {
        Ok(resolved) => resolved,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
    };
    let EngineVariant::Decision(runtime) = lease.engine() else {
        return error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "model does not support System One",
        );
    };
    let runtime = runtime.clone();
    let response: Result<DecisionResponse, _> = runtime.predict(request, lease).await;
    match response {
        Ok(response) => Json(response).into_response(),
        Err(e) => {
            let status = match &e {
                DecisionExecutionError::InvalidInput(_) | DecisionExecutionError::WrongEngine => {
                    StatusCode::UNPROCESSABLE_ENTITY
                }
                DecisionExecutionError::QueueFull | DecisionExecutionError::WorkerStopped => {
                    StatusCode::SERVICE_UNAVAILABLE
                }
                DecisionExecutionError::Timeout => StatusCode::GATEWAY_TIMEOUT,
                DecisionExecutionError::Inference(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            error(status, e)
        }
    }
}
