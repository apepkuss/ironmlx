//! HTTP adapters over the native model pool.
use super::{anthropic, api_transport::ApiJson, diffusion_gemma, openai, responses};
use ironmlx_runtime::core::engine_pool::*;

use crate::Result;
use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;

/// Server composition of network settings and transport-independent runtime options.
#[derive(Debug, Clone)]
pub struct EnginePoolRuntimeConfig {
    pub network: super::security::ServerNetworkConfig,
    pub options: EngineRuntimeOptions,
}

pub(crate) trait EngineRoutedRequest {
    fn model(&self) -> Option<&str>;
    fn set_model(&mut self, model: String);
}

impl EngineRoutedRequest for openai::ChatRequest {
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn set_model(&mut self, model: String) {
        self.model = Some(model);
    }
}

impl EngineRoutedRequest for responses::ResponsesRequest {
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn set_model(&mut self, model: String) {
        self.model = Some(model);
    }
}

impl EngineRoutedRequest for anthropic::MessagesRequest {
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn set_model(&mut self, model: String) {
        self.model = Some(model);
    }
}

pub(crate) trait EngineLeaseHttpAdapter {
    async fn openai_chat_completions(&self, req: openai::ChatRequest) -> Response;
    async fn openai_responses(&self, req: responses::ResponsesRequest) -> Response;
    async fn anthropic_messages(&self, req: anthropic::MessagesRequest) -> Response;
}
impl EngineLeaseHttpAdapter for EngineLease {
    async fn openai_chat_completions(&self, req: openai::ChatRequest) -> Response {
        self.engine().openai_chat_completions(req).await
    }

    async fn openai_responses(&self, req: responses::ResponsesRequest) -> Response {
        self.engine().openai_responses(req).await
    }

    async fn anthropic_messages(&self, req: anthropic::MessagesRequest) -> Response {
        self.engine().anthropic_messages(req).await
    }
}
pub(crate) trait EngineVariantHttpAdapter {
    async fn openai_chat_completions(&self, req: openai::ChatRequest) -> Response;
    async fn openai_responses(&self, req: responses::ResponsesRequest) -> Response;
    async fn anthropic_messages(&self, req: anthropic::MessagesRequest) -> Response;
}
impl EngineVariantHttpAdapter for EngineVariant {
    async fn openai_chat_completions(&self, req: openai::ChatRequest) -> Response {
        match self {
            Self::Audio(_) => super::audio::task_mismatch(super::api_error::ApiProtocol::OpenAi),
            Self::Qwen35(state) => openai::chat_completions_with_state(state.clone(), req).await,
            Self::Qwen35Moe(state) => openai::chat_completions_with_state(state.clone(), req).await,
            Self::Qwen36Moe(state) => openai::chat_completions_with_state(state.clone(), req).await,
            Self::Gemma4(state) => openai::chat_completions_with_state(state.clone(), req).await,
            Self::Gemma4Drafter(state) => {
                openai::chat_completions_with_gemma4_drafter_state(state.as_ref().clone(), req)
                    .await
            }
            Self::Glm4MoeLite(state) => {
                openai::chat_completions_with_state(state.clone(), req).await
            }
            Self::Llama(state) => openai::chat_completions_with_state(state.clone(), req).await,
            Self::MiniCpmV46(state) => {
                openai::chat_completions_with_state(state.clone(), req).await
            }
            Self::DiffusionGemma(state) => {
                diffusion_gemma::openai_chat_completions_with_state(state.clone(), req).await
            }
        }
    }

    async fn openai_responses(&self, req: responses::ResponsesRequest) -> Response {
        match self {
            Self::Audio(_) => super::audio::task_mismatch(super::api_error::ApiProtocol::OpenAi),
            Self::Qwen35(state) => responses::responses_with_state(state.clone(), req, false).await,
            Self::Qwen35Moe(state) => {
                responses::responses_with_state(state.clone(), req, false).await
            }
            Self::Qwen36Moe(state) => {
                responses::responses_with_state(state.clone(), req, false).await
            }
            Self::Gemma4(state) => responses::responses_with_state(state.clone(), req, false).await,
            Self::Gemma4Drafter(state) => {
                responses::responses_with_state(state.base.clone(), req, true).await
            }
            Self::Glm4MoeLite(state) => {
                responses::responses_with_state(state.clone(), req, false).await
            }
            Self::Llama(state) => responses::responses_with_state(state.clone(), req, false).await,
            Self::MiniCpmV46(state) => {
                responses::responses_with_state(state.clone(), req, false).await
            }
            Self::DiffusionGemma(state) => {
                diffusion_gemma::openai_responses_with_state(state.clone(), req).await
            }
        }
    }

    async fn anthropic_messages(&self, req: anthropic::MessagesRequest) -> Response {
        match self {
            Self::Audio(_) => super::audio::task_mismatch(super::api_error::ApiProtocol::Anthropic),
            Self::Qwen35(state) => anthropic::messages_with_state(state.clone(), req).await,
            Self::Qwen35Moe(state) => anthropic::messages_with_state(state.clone(), req).await,
            Self::Qwen36Moe(state) => anthropic::messages_with_state(state.clone(), req).await,
            Self::Gemma4(state) => anthropic::messages_with_state(state.clone(), req).await,
            Self::Gemma4Drafter(state) => {
                anthropic::messages_with_gemma4_drafter_state(state.as_ref().clone(), req).await
            }
            Self::Glm4MoeLite(state) => anthropic::messages_with_state(state.clone(), req).await,
            Self::Llama(state) => anthropic::messages_with_state(state.clone(), req).await,
            Self::MiniCpmV46(state) => anthropic::messages_with_state(state.clone(), req).await,
            Self::DiffusionGemma(state) => {
                diffusion_gemma::anthropic_messages_with_state(state.clone(), req).await
            }
        }
    }
}
pub(crate) trait EnginePoolHttpAdapter {
    async fn resolve_request_engine<R>(&self, request: &mut R) -> Result<EngineLease>
    where
        R: EngineRoutedRequest;
    async fn app_openai_chat_completions(&self, req: openai::ChatRequest) -> Result<Response>;
    async fn app_openai_responses(&self, req: responses::ResponsesRequest) -> Result<Response>;
    async fn app_anthropic_messages(&self, req: anthropic::MessagesRequest) -> Result<Response>;
    async fn model_list(&self) -> OpenAiModelList;
}
impl EnginePoolHttpAdapter for EnginePoolState {
    async fn resolve_request_engine<R>(&self, request: &mut R) -> Result<EngineLease>
    where
        R: EngineRoutedRequest,
    {
        let requested = request
            .model()
            .filter(|model| !model.is_empty())
            .map(str::to_owned);
        if self.is_audio_model(requested.as_deref()).await? {
            return Err(super::audio::ModelTaskMismatch.into());
        }
        let (model_id, engine) = self.resolve_engine(requested.as_deref()).await?;
        if requested.is_none() {
            request.set_model(model_id);
        }
        Ok(engine)
    }

    async fn app_openai_chat_completions(&self, mut req: openai::ChatRequest) -> Result<Response> {
        let engine = self.resolve_request_engine(&mut req).await?;
        Ok(engine.openai_chat_completions(req).await)
    }

    async fn app_openai_responses(&self, mut req: responses::ResponsesRequest) -> Result<Response> {
        let engine = self.resolve_request_engine(&mut req).await?;
        Ok(engine.openai_responses(req).await)
    }

    async fn app_anthropic_messages(
        &self,
        mut req: anthropic::MessagesRequest,
    ) -> Result<Response> {
        let engine = self.resolve_request_engine(&mut req).await?;
        Ok(engine.anthropic_messages(req).await)
    }

    async fn model_list(&self) -> OpenAiModelList {
        let data = self
            .model_snapshots()
            .await
            .into_iter()
            .map(|model| OpenAiModelInfo {
                object: "model",
                created: 0,
                owned_by: "ironmlx",
                id: model.id,
                load_policy: model.load_policy,
                state: model.state,
                unload_reason: model.unload_reason,
                last_error: model.last_error,
                changed_unix_ms: model.changed_unix_ms,
                load_started_unix_ms: model.load_started_unix_ms,
                loaded_unix_ms: model.loaded_unix_ms,
                last_used_unix_ms: model.last_used_unix_ms,
                failed_unix_ms: model.failed_unix_ms,
                load_attempts: model.load_attempts,
                request_count: model.request_count,
            })
            .collect();
        OpenAiModelList {
            object: "list",
            data,
        }
    }
}
pub async fn serve_engine_pool(
    config: EnginePoolConfig,
    runtime: EnginePoolRuntimeConfig,
) -> Result<()> {
    let network = runtime.network.clone();
    let state = EnginePoolState::new(config, runtime.options).await?;
    state.start_model_ttl_sweeper();
    state.start_memory_governor_monitor();
    let app = engine_pool_router().with_state(state);

    let serve_result =
        super::security::serve_router(app, network, "ironmlx EnginePool server").await;
    ironmlx_runtime::core::cache::prefix_store::shutdown_process_async_prefix_store_queue();
    serve_result
}

pub(crate) fn engine_pool_router() -> Router<EnginePoolState> {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(healthz_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/models/:model_id/load", post(load_model_handler))
        .route("/v1/models/:model_id/unload", post(unload_model_handler))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/responses", post(openai_responses))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/audio/speech", post(super::audio::speech))
}

async fn openai_chat_completions(
    State(pool): State<EnginePoolState>,
    ApiJson(mut req): ApiJson<openai::ChatRequest>,
) -> Response {
    let engine = match pool.resolve_request_engine(&mut req).await {
        Ok(engine) => engine,
        Err(error) => {
            return super::api_error::ApiError::engine_resolution(error)
                .into_response(super::api_error::ApiProtocol::OpenAi)
        }
    };
    engine.openai_chat_completions(req).await
}

async fn openai_responses(
    State(pool): State<EnginePoolState>,
    ApiJson(mut req): ApiJson<responses::ResponsesRequest>,
) -> Response {
    let engine = match pool.resolve_request_engine(&mut req).await {
        Ok(engine) => engine,
        Err(error) => {
            return super::api_error::ApiError::engine_resolution(error)
                .into_response(super::api_error::ApiProtocol::OpenAi)
        }
    };
    engine.openai_responses(req).await
}

async fn anthropic_messages(
    State(pool): State<EnginePoolState>,
    ApiJson(mut req): ApiJson<anthropic::MessagesRequest>,
) -> Response {
    let engine = match pool.resolve_request_engine(&mut req).await {
        Ok(engine) => engine,
        Err(error) => {
            return super::api_error::ApiError::engine_resolution(error)
                .into_response(super::api_error::ApiProtocol::Anthropic)
        }
    };
    engine.anthropic_messages(req).await
}

async fn models_handler(State(pool): State<EnginePoolState>) -> Json<OpenAiModelList> {
    Json(pool.model_list().await)
}

async fn load_model_handler(
    State(pool): State<EnginePoolState>,
    Path(model_id): Path<String>,
) -> Response {
    match pool.load_model(&model_id).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => super::api_error::ApiError::engine_resolution(error)
            .into_response(super::api_error::ApiProtocol::OpenAi),
    }
}

async fn unload_model_handler(
    State(pool): State<EnginePoolState>,
    Path(model_id): Path<String>,
) -> Response {
    match pool.unload_model(&model_id).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => super::api_error::ApiError::engine_resolution(error)
            .into_response(super::api_error::ApiProtocol::OpenAi),
    }
}

async fn healthz_handler(State(pool): State<EnginePoolState>) -> Json<EnginePoolHealth> {
    Json(pool.health_snapshot().await)
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiModelList {
    object: &'static str,
    data: Vec<OpenAiModelInfo>,
}

#[derive(Debug, Serialize)]
struct OpenAiModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
    load_policy: EngineLoadPolicy,
    state: EngineRuntimeState,
    #[serde(skip_serializing_if = "Option::is_none")]
    unload_reason: Option<EngineUnloadReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    changed_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    load_started_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    loaded_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_used_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failed_unix_ms: Option<u64>,
    load_attempts: u64,
    request_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn engine_pool_errors_render_for_each_public_protocol() {
        let openai = crate::server::api_error::ApiError::engine_resolution(
            EngineRegistryError::UnknownModel {
                id: "missing".to_owned(),
            }
            .into(),
        )
        .into_response(crate::server::api_error::ApiProtocol::OpenAi);
        assert_eq!(openai.status(), axum::http::StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(openai.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "model_not_found");

        let anthropic =
            crate::server::api_error::ApiError::engine_resolution(anyhow::anyhow!("engine failed"))
                .into_response(crate::server::api_error::ApiProtocol::Anthropic);
        assert_eq!(
            anthropic.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(anthropic.headers()[axum::http::header::RETRY_AFTER], "5");
        let request_id = anthropic.headers()["request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let body = axum::body::to_bytes(anthropic.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "overloaded_error");
        assert_eq!(body["error"]["code"], "engine_unavailable");
        assert_eq!(body["request_id"], request_id);
    }
    #[test]
    fn engine_pool_router_builds_control_routes() {
        let _router = super::engine_pool_router();
    }
}
