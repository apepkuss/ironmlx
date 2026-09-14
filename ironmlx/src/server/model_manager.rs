#[cfg(test)]
use ironmlx_runtime::core::model_management::{
    apply_load_request_scheduler_overrides, build_engine_model_config, engine_model_capabilities,
    read_generation_sampling_defaults, EngineModelBuildRequest, DEFAULT_PROFILE_WARNING,
    DEFAULT_PROFILE_WARNING_CODE, DIFFUSION_GEMMA_KV_CACHE_UNSUPPORTED_CODE,
};
use ironmlx_runtime::core::model_management::{
    ModelCapabilityError, ModelLoadRequest, ModelLoadWarning as AdminWarning, ModelManagement,
    ModelManagementError, ModelManagementOutcome, ModelManagementStatus,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use ironmlx_runtime::core::scheduler_resolution::SchedulerResolutionOptions;

use crate::Result;
use ironmlx_runtime::core::prompt_lookup::PromptLookupConfig;

use {
    super::anthropic, super::api_transport::ApiJson, super::openai, super::responses,
    ironmlx_runtime::core::runtime_config::SamplingDefaults,
};
use {
    super::engine::EnginePoolRuntimeConfig,
    ironmlx_runtime::core::engine_pool::EngineLoadedModelInfo,
    ironmlx_runtime::core::engine_pool::EnginePoolState,
    ironmlx_runtime::core::engine_pool::EngineRegistryError,
    ironmlx_runtime::core::engine_pool::EngineRuntimeState,
};
use {
    ironmlx_runtime::core::runtime_health::classify_status,
    ironmlx_runtime::core::runtime_health::system_free_ram_bytes,
    ironmlx_runtime::core::runtime_health::HealthSnapshot,
    ironmlx_runtime::core::runtime_health::MemoryInfo,
    ironmlx_runtime::core::runtime_health::ModelInfo,
    ironmlx_runtime::core::runtime_health::MtpHealthInfo,
    ironmlx_runtime::core::runtime_health::NeuralExactQualificationHealth,
    ironmlx_runtime::core::runtime_health::PromptLookupHealthInfo,
    ironmlx_runtime::core::runtime_health::SchedulerInfo,
};

const MODEL_REQUIRED_CODE: &str = "model_required";
const MODEL_REQUIRED_MESSAGE: &str = "Model is required.";
const MODEL_DIRECTORY_NOT_FOUND_CODE: &str = "model_directory_not_found";
const INVALID_MAX_CACHE_CAP_CODE: &str = "invalid_max_cache_cap";
const INVALID_MAX_CACHE_CAP_MESSAGE: &str =
    "MAX CONTEXT TOKENS must be greater than or equal to 1.";
const MODEL_NOT_LOADED_CODE: &str = "model_not_loaded";
const MODEL_NOT_REGISTERED_CODE: &str = "model_not_registered";
const BACKEND_UNLOAD_ERROR_CODE: &str = "backend_unload_error";
const GPU_MEMORY_INSUFFICIENT_CODE: &str = "gpu_memory_insufficient";
const GPU_MEMORY_INSUFFICIENT_MESSAGE: &str =
    "Not enough available GPU memory to safely load this model. Unload one or more unused loaded models to free GPU memory, then try again.";
const MAX_LOADED_MODELS_REACHED_CODE: &str = "max_loaded_models_reached";
const MAX_LOADED_MODELS_REACHED_MESSAGE: &str =
    "Maximum concurrent loaded models reached. Unload an unused loaded model before loading another model.";
const MODEL_MEMORY_LIMIT_EXCEEDED_CODE: &str = "model_memory_limit_exceeded";
const MODEL_MEMORY_LIMIT_EXCEEDED_MESSAGE: &str =
    "Model memory limit reached. Unload one or more loaded models, raise Memory Limit (Model Only), or set it to Auto, then try again.";
const TOTAL_MEMORY_LIMIT_EXCEEDED_CODE: &str = "total_memory_limit_exceeded";
const TOTAL_MEMORY_LIMIT_EXCEEDED_MESSAGE: &str =
    "Total memory limit reached. Unload one or more loaded models, raise Memory Limit (Total), or set it to Auto, then try again.";
const KV_MEMORY_BUDGET_EXCEEDED_CODE: &str = "kv_memory_budget_exceeded";
const KV_MEMORY_BUDGET_EXCEEDED_MESSAGE: &str =
    "KV cache memory budget exceeded. Enable Active KV offload with paged prefix cache for long context, or lower MAX CONTEXT TOKENS or Max Sequences in the Dashboard, then try again.";
const MODEL_RELOAD_BUSY_CODE: &str = "model_reload_busy";
const MODEL_RELOAD_BUSY_MESSAGE: &str =
    "The model is processing requests. Wait for it to become idle before switching versions.";
const MTP_MODEL_DIR_REQUIRED_CODE: &str = "mtp_model_dir_required";
const MTP_MODEL_DIR_REQUIRED_MESSAGE: &str = "MTP draft tokens require an MTP model directory.";
const MTP_INCOMPATIBLE_MESSAGE: &str =
    "MTP weights are not compatible with this model. Load the model without MTP or choose matching MTP weights.";

#[derive(Clone)]
pub struct ModelManager {
    management: ModelManagement,
    pool: EnginePoolState,
    start_time: Instant,
}

impl ModelManager {
    pub(crate) fn new(
        runtime: EnginePoolRuntimeConfig,
        max_loaded_models: Option<usize>,
        scheduler_options: SchedulerResolutionOptions,
    ) -> Result<Self> {
        let pool = EnginePoolState::new_dynamic(runtime.options, max_loaded_models)?;
        Ok(Self {
            management: ModelManagement::new(pool.clone(), scheduler_options),
            pool,
            start_time: Instant::now(),
        })
    }

    fn start_model_ttl_sweeper(&self) {
        self.pool.start_model_ttl_sweeper();
        self.pool.start_memory_governor_monitor();
    }

    async fn load_model(
        &self,
        request: LoadModelRequest,
    ) -> std::result::Result<AdminModelResponse, AdminError> {
        let parsed = parse_load_model_request(request)?;
        self.management
            .load_model(parsed)
            .await
            .map(AdminModelResponse::from)
            .map_err(AdminError::from_management_error)
    }

    async fn register_model(
        &self,
        request: LoadModelRequest,
    ) -> std::result::Result<AdminModelResponse, AdminError> {
        let pinned = request.pinned;
        let parsed = parse_load_model_request(request)?;
        self.management
            .register_model(parsed, pinned)
            .await
            .map(AdminModelResponse::from)
            .map_err(AdminError::from_management_error)
    }

    async fn unload_model(&self, request: UnloadModelRequest) -> AdminModelResponse {
        let model = request
            .model
            .as_deref()
            .or(request.model_dir.as_deref())
            .or(request.repo_id.as_deref())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let status = match model.as_deref() {
            Some(model) => match self.pool.unload_dynamic_model(model).await {
                Ok(result) => match result.state {
                    EngineRuntimeState::Draining => "unload_deferred",
                    _ => "unloaded",
                },
                Err(error) => {
                    let error = AdminError::from_control_error(error);
                    return AdminModelResponse::from_error(error.message, error.code);
                }
            },
            None => "not_loaded",
        };
        AdminModelResponse::ok(status, model, self.list_loaded().await, None)
    }

    async fn clear_shared_prompt_lookup(
        &self,
        request: ClearPromptLookupRequest,
    ) -> std::result::Result<ClearPromptLookupResponse, AdminError> {
        let model = request
            .model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let (cleared_models, cleared_entries) = self
            .pool
            .clear_shared_prompt_lookup(model)
            .await
            .map_err(AdminError::from_control_error)?;
        Ok(ClearPromptLookupResponse {
            success: true,
            status: "cleared",
            model: model.map(str::to_string),
            cleared_models,
            cleared_entries,
        })
    }

    async fn set_model_pinned(
        &self,
        request: PinModelRequest,
        pinned: bool,
    ) -> std::result::Result<AdminModelResponse, AdminError> {
        let model = request.model.trim();
        if model.is_empty() {
            return Err(AdminError::model_required());
        }
        if !self.pool.is_model_loaded(model).await {
            return Err(AdminError::model_not_loaded(model));
        }
        self.pool
            .set_model_pinned(model, pinned)
            .await
            .map_err(AdminError::from_control_error)?;
        Ok(AdminModelResponse::ok(
            if pinned { "pinned" } else { "unpinned" },
            Some(model.to_string()),
            self.list_loaded().await,
            None,
        ))
    }

    async fn set_default_model(
        &self,
        request: SetDefaultModelRequest,
    ) -> std::result::Result<AdminModelResponse, AdminError> {
        let model = request.model.trim();
        if model.is_empty() {
            return Err(AdminError::model_required());
        }
        if !self.pool.is_model_registered(model).await {
            return Err(AdminError::model_not_registered(model));
        }
        self.pool
            .set_default_model(model)
            .await
            .map_err(|_| AdminError::model_not_registered(model))?;
        Ok(AdminModelResponse::ok(
            "default_set",
            Some(model.to_string()),
            self.list_loaded().await,
            None,
        ))
    }

    async fn list_loaded(&self) -> Vec<LoadedModelInfo> {
        self.pool
            .loaded_model_infos()
            .await
            .into_iter()
            .map(LoadedModelInfo::from)
            .collect()
    }

    async fn health_snapshot(&self) -> AppHealthSnapshot {
        let snapshots = self.pool.loaded_causal_health_snapshots().await;
        AppHealthSnapshot {
            aggregate: aggregate_health(self.start_time, snapshots),
            mode: "model_manager",
            models: self.list_loaded().await,
        }
    }

    async fn openai(&self, req: openai::ChatRequest) -> Response {
        match self.pool.app_openai_chat_completions(req).await {
            Ok(response) => response,
            Err(error) => super::api_error::ApiError::engine_resolution(error)
                .into_response(super::api_error::ApiProtocol::OpenAi),
        }
    }

    async fn responses(&self, req: responses::ResponsesRequest) -> Response {
        match self.pool.app_openai_responses(req).await {
            Ok(response) => response,
            Err(error) => super::api_error::ApiError::engine_resolution(error)
                .into_response(super::api_error::ApiProtocol::OpenAi),
        }
    }

    async fn anthropic(&self, req: anthropic::MessagesRequest) -> Response {
        match self.pool.app_anthropic_messages(req).await {
            Ok(response) => response,
            Err(error) => super::api_error::ApiError::engine_resolution(error)
                .into_response(super::api_error::ApiProtocol::Anthropic),
        }
    }
}

fn parse_load_model_request(
    request: LoadModelRequest,
) -> std::result::Result<ModelLoadRequest, AdminError> {
    let model_reference = request
        .model
        .as_deref()
        .or(request.model_dir.as_deref())
        .or(request.repo_id.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(AdminError::model_required)?
        .to_string();
    let model_dir_value = request
        .model_dir
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| model_reference.clone());
    let model_dir = PathBuf::from(&model_dir_value);
    if !model_dir.exists() {
        return Err(AdminError::model_directory_not_found(&model_dir));
    }

    let max_cache_cap_override = match request.max_cache_cap {
        Some(0) => return Err(AdminError::invalid_max_cache_cap()),
        value => value,
    };
    let mtp = match (
        request
            .mtp_model_dir
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty()),
        request.mtp_draft_tokens,
    ) {
        (Some(model_dir), draft_tokens) => {
            Some(ironmlx_runtime::core::engine_pool::EngineMtpSettings {
                model_dir: PathBuf::from(model_dir),
                draft_tokens,
            })
        }
        (None, Some(_)) => {
            return Err(AdminError::bad_request_with_code(
                MTP_MODEL_DIR_REQUIRED_MESSAGE,
                Some(MTP_MODEL_DIR_REQUIRED_CODE),
            ));
        }
        (None, None) => None,
    };
    let prompt_lookup = request
        .prompt_lookup
        .map(PromptLookupConfig::validate)
        .transpose()
        .map_err(AdminError::from_load_error)?;
    Ok(ModelLoadRequest {
        model_reference,
        model_dir,
        max_cache_cap_override,
        sampling_defaults: request.sampling_defaults,
        mtp,
        prompt_lookup,
        pinned: request.pinned.unwrap_or(false),
        set_default: request.set_default.unwrap_or(false),
        reload_when_idle: request.reload_when_idle.unwrap_or(false),
        defer_when_busy: request.defer_when_busy.unwrap_or(true),
    })
}

pub(crate) fn validate_mtp_pair(
    model_dir: &Path,
    mtp_model_dir: &Path,
    draft_tokens: Option<usize>,
) -> Result<MtpValidationResponse> {
    ironmlx_runtime::core::model_validation::validate_mtp_pair(
        model_dir,
        mtp_model_dir,
        draft_tokens,
    )
    .map(Into::into)
}

pub async fn serve_app_daemon(
    manager: ModelManager,
    network: super::security::ServerNetworkConfig,
) -> Result<()> {
    manager.start_model_ttl_sweeper();
    let app = app_router(manager);

    let serve_result = super::security::serve_router(app, network, "ironmlx app daemon").await;
    ironmlx_runtime::core::cache::prefix_store::shutdown_process_async_prefix_store_queue();
    serve_result
}

fn app_router(manager: ModelManager) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(app_healthz_handler))
        .route("/v1/models", get(app_models_handler))
        .route("/v1/chat/completions", post(app_openai_handler))
        .route("/v1/responses", post(app_responses_handler))
        .route("/v1/messages", post(app_anthropic_handler))
        .route("/admin/api/models/loaded", get(list_loaded_handler))
        .route("/admin/api/models/register", post(register_model_handler))
        .route("/admin/api/models/load", post(load_model_handler))
        .route("/admin/api/models/mtp/validate", post(validate_mtp_handler))
        .route("/admin/api/models/unload", post(unload_model_handler))
        .route(
            "/admin/api/prompt-lookup/clear",
            post(clear_prompt_lookup_handler),
        )
        .route("/admin/api/models/pin", post(pin_model_handler))
        .route("/admin/api/models/unpin", post(unpin_model_handler))
        .route("/admin/api/models/default", post(set_default_model_handler))
        .with_state(manager)
}

async fn app_openai_handler(
    State(manager): State<ModelManager>,
    ApiJson(req): ApiJson<openai::ChatRequest>,
) -> Response {
    manager.openai(req).await
}

async fn app_responses_handler(
    State(manager): State<ModelManager>,
    ApiJson(req): ApiJson<responses::ResponsesRequest>,
) -> Response {
    manager.responses(req).await
}

async fn app_anthropic_handler(
    State(manager): State<ModelManager>,
    ApiJson(req): ApiJson<anthropic::MessagesRequest>,
) -> Response {
    manager.anthropic(req).await
}

async fn app_healthz_handler(State(manager): State<ModelManager>) -> Json<AppHealthSnapshot> {
    Json(manager.health_snapshot().await)
}

async fn app_models_handler(
    State(manager): State<ModelManager>,
) -> Json<super::engine::OpenAiModelList> {
    Json(manager.pool.model_list().await)
}

async fn list_loaded_handler(State(manager): State<ModelManager>) -> Json<Vec<LoadedModelInfo>> {
    Json(manager.list_loaded().await)
}

async fn load_model_handler(
    State(manager): State<ModelManager>,
    Json(request): Json<LoadModelRequest>,
) -> Response {
    match manager.load_model(request).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn register_model_handler(
    State(manager): State<ModelManager>,
    Json(request): Json<LoadModelRequest>,
) -> Response {
    match manager.register_model(request).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn validate_mtp_handler(Json(request): Json<MtpValidationRequest>) -> Response {
    let base = request
        .model_dir
        .or(request.model_path)
        .map(PathBuf::from)
        .unwrap_or_default();
    let mtp = request.mtp_model_dir.map(PathBuf::from).unwrap_or_default();
    match validate_mtp_pair(&base, &mtp, request.mtp_draft_tokens) {
        Ok(response) => Json(response).into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(MtpValidationResponse::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            )),
        )
            .into_response(),
    }
}

async fn unload_model_handler(
    State(manager): State<ModelManager>,
    Json(request): Json<UnloadModelRequest>,
) -> Json<AdminModelResponse> {
    Json(manager.unload_model(request).await)
}

async fn clear_prompt_lookup_handler(
    State(manager): State<ModelManager>,
    Json(request): Json<ClearPromptLookupRequest>,
) -> Response {
    match manager.clear_shared_prompt_lookup(request).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn pin_model_handler(
    State(manager): State<ModelManager>,
    Json(request): Json<PinModelRequest>,
) -> Response {
    match manager.set_model_pinned(request, true).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn unpin_model_handler(
    State(manager): State<ModelManager>,
    Json(request): Json<PinModelRequest>,
) -> Response {
    match manager.set_model_pinned(request, false).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn set_default_model_handler(
    State(manager): State<ModelManager>,
    Json(request): Json<SetDefaultModelRequest>,
) -> Response {
    match manager.set_default_model(request).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct LoadModelRequest {
    model: Option<String>,
    model_dir: Option<String>,
    repo_id: Option<String>,
    set_default: Option<bool>,
    max_cache_cap: Option<usize>,
    pinned: Option<bool>,
    mtp_model_dir: Option<String>,
    mtp_draft_tokens: Option<usize>,
    prompt_lookup: Option<PromptLookupConfig>,
    reload_when_idle: Option<bool>,
    defer_when_busy: Option<bool>,
    #[serde(flatten)]
    sampling_defaults: SamplingDefaults,
}

#[derive(Debug, Deserialize)]
struct MtpValidationRequest {
    model_dir: Option<String>,
    model_path: Option<String>,
    mtp_model_dir: Option<String>,
    mtp_draft_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub(crate) struct MtpValidationResponse {
    pub(crate) compatible: bool,
    pub(crate) reason_code: &'static str,
    pub(crate) message: String,
    draft_tokens: Option<usize>,
}

impl MtpValidationResponse {
    fn not_compatible(
        reason_code: &'static str,
        message: impl Into<String>,
        draft_tokens: Option<usize>,
    ) -> Self {
        Self {
            compatible: false,
            reason_code,
            message: message.into(),
            draft_tokens,
        }
    }
}

#[derive(Debug, Deserialize)]
struct UnloadModelRequest {
    model: Option<String>,
    model_dir: Option<String>,
    repo_id: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ClearPromptLookupRequest {
    model: Option<String>,
}

#[derive(Debug, Serialize)]
struct ClearPromptLookupResponse {
    success: bool,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    cleared_models: usize,
    cleared_entries: usize,
}

#[derive(Debug, Deserialize)]
struct PinModelRequest {
    model: String,
}

#[derive(Debug, Deserialize)]
struct SetDefaultModelRequest {
    model: String,
}

#[derive(Debug, Serialize)]
struct AdminModelResponse {
    success: bool,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    loaded_models: Vec<LoadedModelInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warning_code: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl AdminModelResponse {
    fn ok(
        status: &'static str,
        model: Option<String>,
        loaded_models: Vec<LoadedModelInfo>,
        warning: Option<AdminWarning>,
    ) -> Self {
        Self {
            success: true,
            status,
            code: None,
            model,
            loaded_models,
            warning_code: warning.as_ref().map(|warning| warning.code),
            warning: warning.map(|warning| warning.message),
            error: None,
        }
    }

    fn from_error(message: String, code: Option<&'static str>) -> Self {
        Self {
            success: false,
            status: "error",
            code,
            model: None,
            loaded_models: Vec::new(),
            warning_code: None,
            warning: None,
            error: Some(message),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct LoadedModelInfo {
    id: String,
    model: String,
    path: String,
    architecture: String,
    runtime_kind: &'static str,
    supports_streaming: bool,
    supports_vision: bool,
    supports_mtp: bool,
    supports_prompt_lookup: bool,
    supports_speculative_decoding: bool,
    supports_kv_cache: bool,
    supported_sampling_parameters: &'static [&'static str],
    runtime_state: EngineRuntimeState,
    #[serde(skip_serializing_if = "Option::is_none")]
    scheduler: Option<&'static str>,
    active_requests: usize,
    queued_requests: usize,
    queue_capacity: usize,
    usage: ironmlx_runtime::core::runtime_usage::ModelRuntimeUsageSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_kv_offload: Option<ironmlx_runtime::core::cache::active_kv::ActiveKvOffloadHealth>,
    #[serde(rename = "default")]
    is_default: bool,
    pinned: bool,
    max_position_embeddings: i32,
    mtp_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    mtp_model_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mtp_draft_tokens: Option<usize>,
    prompt_lookup_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_lookup: Option<PromptLookupConfig>,
}

#[derive(Debug, Serialize)]
struct AppHealthSnapshot {
    #[serde(flatten)]
    aggregate: HealthSnapshot,
    mode: &'static str,
    models: Vec<LoadedModelInfo>,
}

impl From<EngineLoadedModelInfo> for LoadedModelInfo {
    fn from(info: EngineLoadedModelInfo) -> Self {
        Self {
            id: info.id.clone(),
            model: info.id,
            path: info.path,
            architecture: info.architecture,
            runtime_kind: info.capabilities.runtime_kind,
            supports_streaming: info.capabilities.supports_streaming,
            supports_vision: info.capabilities.supports_vision,
            supports_mtp: info.capabilities.supports_mtp,
            supports_prompt_lookup: info.capabilities.supports_prompt_lookup,
            supports_speculative_decoding: info.capabilities.supports_speculative_decoding,
            supports_kv_cache: info.capabilities.supports_kv_cache,
            supported_sampling_parameters: info.capabilities.supported_sampling_parameters,
            runtime_state: info.runtime_state,
            scheduler: info.scheduler,
            active_requests: info.active_requests,
            queued_requests: info.queued_requests,
            queue_capacity: info.queue_capacity,
            usage: info.usage,
            active_kv_offload: info.active_kv_offload,
            is_default: info.is_default,
            pinned: info.pinned,
            max_position_embeddings: info.max_position_embeddings,
            mtp_enabled: info.mtp_model_dir.is_some(),
            mtp_model_dir: info.mtp_model_dir,
            mtp_draft_tokens: info.mtp_draft_tokens,
            prompt_lookup_enabled: info.prompt_lookup.is_some(),
            prompt_lookup: info.prompt_lookup,
        }
    }
}

struct AdminError {
    status: StatusCode,
    message: String,
    code: Option<&'static str>,
}

impl AdminError {
    fn from_management_error(error: ModelManagementError) -> Self {
        match error {
            ModelManagementError::Busy => {
                Self::conflict_with_code(MODEL_RELOAD_BUSY_MESSAGE, Some(MODEL_RELOAD_BUSY_CODE))
            }
            ModelManagementError::NotLoaded(model) => Self::model_not_loaded(&model),
            ModelManagementError::GpuMemoryLow => Self::service_unavailable_with_code(
                GPU_MEMORY_INSUFFICIENT_MESSAGE,
                Some(GPU_MEMORY_INSUFFICIENT_CODE),
            ),
            ModelManagementError::Load(error) => Self::from_load_error(error),
        }
    }

    fn model_required() -> Self {
        Self::bad_request_with_code(MODEL_REQUIRED_MESSAGE, Some(MODEL_REQUIRED_CODE))
    }

    fn model_directory_not_found(path: &Path) -> Self {
        Self::bad_request_with_code(
            format!("Model directory does not exist: {}", path.display()),
            Some(MODEL_DIRECTORY_NOT_FOUND_CODE),
        )
    }

    fn invalid_max_cache_cap() -> Self {
        Self::bad_request_with_code(
            INVALID_MAX_CACHE_CAP_MESSAGE,
            Some(INVALID_MAX_CACHE_CAP_CODE),
        )
    }

    fn model_not_loaded(model: &str) -> Self {
        Self::not_found_with_code(
            format!("Model is not loaded: {model}"),
            Some(MODEL_NOT_LOADED_CODE),
        )
    }

    fn model_not_registered(model: &str) -> Self {
        Self::not_found_with_code(
            format!("Model is not registered: {model}"),
            Some(MODEL_NOT_REGISTERED_CODE),
        )
    }

    fn bad_request_with_code(message: impl Into<String>, code: Option<&'static str>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            code,
        }
    }

    fn not_found_with_code(message: impl Into<String>, code: Option<&'static str>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
            code,
        }
    }

    fn conflict_with_code(message: impl Into<String>, code: Option<&'static str>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: message.into(),
            code,
        }
    }

    fn service_unavailable_with_code(
        message: impl Into<String>,
        code: Option<&'static str>,
    ) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            code,
        }
    }

    fn from_control_error(error: anyhow::Error) -> Self {
        if let Some(
            EngineRegistryError::UnknownModel { id } | EngineRegistryError::ModelDisabled { id },
        ) = error.downcast_ref::<EngineRegistryError>()
        {
            return Self::model_not_loaded(id);
        }
        Self::bad_request_with_code(format!("{error:#}"), Some(BACKEND_UNLOAD_ERROR_CODE))
    }

    fn from_load_error(error: anyhow::Error) -> Self {
        if let Some(error) = error.downcast_ref::<ModelCapabilityError>() {
            return Self::bad_request_with_code(error.message, Some(error.code));
        }
        let message = format!("{error:#}");
        if let Some(code) = mtp_error_code_from_message(&message) {
            let user_message = if code == MTP_MODEL_DIR_REQUIRED_CODE {
                MTP_MODEL_DIR_REQUIRED_MESSAGE
            } else {
                MTP_INCOMPATIBLE_MESSAGE
            };
            return Self::bad_request_with_code(user_message, Some(code));
        }
        if likely_engine_pool_model_memory_limit_error(&message) {
            return Self::service_unavailable_with_code(
                MODEL_MEMORY_LIMIT_EXCEEDED_MESSAGE,
                Some(MODEL_MEMORY_LIMIT_EXCEEDED_CODE),
            );
        }
        if likely_engine_pool_total_memory_limit_error(&message) {
            return Self::service_unavailable_with_code(
                TOTAL_MEMORY_LIMIT_EXCEEDED_MESSAGE,
                Some(TOTAL_MEMORY_LIMIT_EXCEEDED_CODE),
            );
        }
        if likely_memory_budget_error(&message) {
            return Self::service_unavailable_with_code(
                KV_MEMORY_BUDGET_EXCEEDED_MESSAGE,
                Some(KV_MEMORY_BUDGET_EXCEEDED_CODE),
            );
        }
        if likely_gpu_memory_error(&message) {
            return Self::service_unavailable_with_code(
                GPU_MEMORY_INSUFFICIENT_MESSAGE,
                Some(GPU_MEMORY_INSUFFICIENT_CODE),
            );
        }
        if likely_engine_pool_capacity_error(&message) {
            return Self::service_unavailable_with_code(
                MAX_LOADED_MODELS_REACHED_MESSAGE,
                Some(MAX_LOADED_MODELS_REACHED_CODE),
            );
        }
        Self {
            status: StatusCode::BAD_REQUEST,
            message,
            code: None,
        }
    }
}

fn mtp_error_code_from_message(message: &str) -> Option<&'static str> {
    [
        MTP_MODEL_DIR_REQUIRED_CODE,
        MTP_BASE_MODEL_NOT_FOUND_CODE,
        MTP_MODEL_NOT_FOUND_CODE,
        MTP_UNSUPPORTED_ARCHITECTURE_CODE,
        MTP_INVALID_MODEL_TYPE_CODE,
        MTP_INVALID_CONFIG_CODE,
        MTP_INVALID_DRAFT_TOKENS_CODE,
        MTP_INCOMPATIBLE_CODE,
    ]
    .into_iter()
    .find(|code| message.contains(code))
}

impl IntoResponse for AdminError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(AdminModelResponse::from_error(self.message, self.code)),
        )
            .into_response()
    }
}

fn likely_gpu_memory_error(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("out of memory")
        || lower.contains("oom")
        || lower.contains("failed to allocate")
        || lower.contains("memory allocation")
}

fn likely_memory_budget_error(message: &str) -> bool {
    message.to_ascii_lowercase().contains("memory budget")
}

fn likely_engine_pool_capacity_error(message: &str) -> bool {
    message
        .to_ascii_lowercase()
        .contains("engine pool capacity reached")
}

fn likely_engine_pool_model_memory_limit_error(message: &str) -> bool {
    message
        .to_ascii_lowercase()
        .contains("engine pool model memory limit exceeded")
}

fn likely_engine_pool_total_memory_limit_error(message: &str) -> bool {
    message
        .to_ascii_lowercase()
        .contains("engine pool total memory limit exceeded")
}

fn aggregate_health(start_time: Instant, snapshots: Vec<HealthSnapshot>) -> HealthSnapshot {
    let mlx_memory = mlx::memory::snapshot();
    let process_governor =
        ironmlx_runtime::core::process_memory::global_process_memory_governor().sample_process();
    let prefix_store =
        ironmlx_runtime::core::cache::prefix_store::process_async_prefix_store_queue().stats();
    let total_ram_bytes = ironmlx_runtime::core::memory_budget::system_total_ram_bytes();
    let free_ram_bytes = system_free_ram_bytes();
    let mut names = Vec::new();
    let mut max_position_embeddings = 0;
    let mut b_max = 0;
    let mut b_active = 0;
    let mut b_queued = 0;
    let mut queue_max = 0;
    let mut admit_count = 0;
    let mut batch_count = 0;
    let mut admission_queue_full_count = 0;
    let mut memory_budget_exceeded_count = 0;
    let mut kv_cache_active_bytes = 0;
    let mut kv_cache_soft_limit_bytes = 0;
    let mut kv_cache_logical_cap_tokens = 0;
    let mut kv_cache_resident_cap_tokens = 0;
    let mut kv_cache_budget_policies: Vec<String> = Vec::new();
    let mut mtp_enabled = false;
    let mut mtp_requested_draft_token_values: Vec<usize> = Vec::new();
    let mut mtp_draft_token_values: Vec<usize> = Vec::new();
    let mut mtp_prefill_count = 0;
    let mut mtp_step_count = 0;
    let mut mtp_fallback_prefill_count = 0;
    let mut mtp_drafted_tokens = 0;
    let mut mtp_accepted_draft_tokens = 0;
    let mut mtp_windows = 0;
    let mut mtp_multi_token_windows = 0;
    let mut mtp_exact_sampling_windows = 0;
    let mut mtp_exact_acceptance_draws = 0;
    let mut mtp_exact_residual_corrections = 0;
    let mut mtp_exact_bonus_samples = 0;
    let mut mtp_draft_forward_us = 0;
    let mut mtp_verify_forward_us = 0;
    let mut mtp_projection_us = 0;
    let mut mtp_sampling_us = 0;
    let mut mtp_draft_host_sync_count = 0;
    let mut mtp_draft_host_sync_us = 0;
    let mut mtp_verify_accept_host_sync_count = 0;
    let mut mtp_verify_accept_host_sync_us = 0;
    let mut mtp_main_rollback_us = 0;
    let mut mtp_cache_commit_us = 0;
    let mut mtp_prefill_cache_commit_us = 0;
    let mut mtp_decode_cache_commit_us = 0;
    let mut mtp_cache_restore_us = 0;
    let mut neural_exact_qualification = NeuralExactQualificationHealth::default();
    let mut active_kv_snapshots = Vec::new();
    let mut prompt_lookup_snapshots = Vec::new();
    let mut immutable_prefix_blocks =
        ironmlx_runtime::core::scheduler_actor::ImmutablePrefixBlockHealth::default();

    for snapshot in snapshots {
        if !snapshot.model.name.is_empty() {
            names.push(snapshot.model.name);
        }
        max_position_embeddings =
            max_position_embeddings.max(snapshot.model.max_position_embeddings);
        b_max += snapshot.scheduler.b_max;
        b_active += snapshot.scheduler.b_active;
        b_queued += snapshot.scheduler.b_queued;
        queue_max += snapshot.scheduler.queue_max;
        admit_count += snapshot.scheduler.admit_count;
        batch_count += snapshot.scheduler.batch_count;
        admission_queue_full_count += snapshot.scheduler.admission_queue_full_count;
        memory_budget_exceeded_count += snapshot.scheduler.memory_budget_exceeded_count;
        kv_cache_active_bytes += snapshot.memory.kv_cache_active_bytes;
        kv_cache_soft_limit_bytes += snapshot.memory.kv_cache_soft_limit_bytes;
        kv_cache_logical_cap_tokens =
            kv_cache_logical_cap_tokens.max(snapshot.memory.kv_cache_logical_cap_tokens);
        kv_cache_resident_cap_tokens =
            kv_cache_resident_cap_tokens.max(snapshot.memory.kv_cache_resident_cap_tokens);
        if !snapshot.memory.kv_cache_budget_policy.is_empty()
            && !kv_cache_budget_policies.contains(&snapshot.memory.kv_cache_budget_policy)
        {
            kv_cache_budget_policies.push(snapshot.memory.kv_cache_budget_policy);
        }
        if snapshot.mtp.enabled {
            mtp_enabled = true;
            if let Some(draft_tokens) = snapshot.mtp.requested_draft_tokens {
                if !mtp_requested_draft_token_values.contains(&draft_tokens) {
                    mtp_requested_draft_token_values.push(draft_tokens);
                }
            }
            if let Some(draft_tokens) = snapshot.mtp.draft_tokens {
                if !mtp_draft_token_values.contains(&draft_tokens) {
                    mtp_draft_token_values.push(draft_tokens);
                }
            }
        }
        mtp_prefill_count += snapshot.mtp.prefill_count;
        mtp_step_count += snapshot.mtp.step_count;
        mtp_fallback_prefill_count += snapshot.mtp.fallback_prefill_count;
        mtp_drafted_tokens += snapshot.mtp.drafted_tokens;
        mtp_accepted_draft_tokens += snapshot.mtp.accepted_draft_tokens;
        mtp_windows += snapshot.mtp.windows;
        mtp_multi_token_windows += snapshot.mtp.multi_token_windows;
        mtp_exact_sampling_windows += snapshot.mtp.exact_sampling_windows;
        mtp_exact_acceptance_draws += snapshot.mtp.exact_acceptance_draws;
        mtp_exact_residual_corrections += snapshot.mtp.exact_residual_corrections;
        mtp_exact_bonus_samples += snapshot.mtp.exact_bonus_samples;
        mtp_draft_forward_us += snapshot.mtp.draft_forward_us;
        mtp_verify_forward_us += snapshot.mtp.verify_forward_us;
        mtp_projection_us += snapshot.mtp.projection_us;
        mtp_sampling_us += snapshot.mtp.sampling_us;
        mtp_draft_host_sync_count += snapshot.mtp.draft_host_sync_count;
        mtp_draft_host_sync_us += snapshot.mtp.draft_host_sync_us;
        mtp_verify_accept_host_sync_count += snapshot.mtp.verify_accept_host_sync_count;
        mtp_verify_accept_host_sync_us += snapshot.mtp.verify_accept_host_sync_us;
        mtp_main_rollback_us += snapshot.mtp.main_rollback_us;
        mtp_cache_commit_us += snapshot.mtp.cache_commit_us;
        mtp_prefill_cache_commit_us += snapshot.mtp.prefill_cache_commit_us;
        mtp_decode_cache_commit_us += snapshot.mtp.decode_cache_commit_us;
        mtp_cache_restore_us += snapshot.mtp.cache_restore_us;
        neural_exact_qualification.ordinary_cost_samples += snapshot
            .mtp
            .sampled_exact_qualification
            .ordinary_cost_samples;
        neural_exact_qualification.exact_cost_samples +=
            snapshot.mtp.sampled_exact_qualification.exact_cost_samples;
        neural_exact_qualification.ordinary_cost_us +=
            snapshot.mtp.sampled_exact_qualification.ordinary_cost_us;
        neural_exact_qualification.exact_cost_us +=
            snapshot.mtp.sampled_exact_qualification.exact_cost_us;
        neural_exact_qualification.qualified_regimes_current += snapshot
            .mtp
            .sampled_exact_qualification
            .qualified_regimes_current;
        neural_exact_qualification.rejected_regimes_current += snapshot
            .mtp
            .sampled_exact_qualification
            .rejected_regimes_current;
        neural_exact_qualification.qualification_changes += snapshot
            .mtp
            .sampled_exact_qualification
            .qualification_changes;
        neural_exact_qualification.profile_loads +=
            snapshot.mtp.sampled_exact_qualification.profile_loads;
        neural_exact_qualification.profile_write_requests += snapshot
            .mtp
            .sampled_exact_qualification
            .profile_write_requests;
        neural_exact_qualification.profile_writes +=
            snapshot.mtp.sampled_exact_qualification.profile_writes;
        neural_exact_qualification.profile_write_failures += snapshot
            .mtp
            .sampled_exact_qualification
            .profile_write_failures;
        neural_exact_qualification.profile_write_coalesces += snapshot
            .mtp
            .sampled_exact_qualification
            .profile_write_coalesces;
        immutable_prefix_blocks.enabled |= snapshot.memory.immutable_prefix_blocks.enabled;
        immutable_prefix_blocks.blocks += snapshot.memory.immutable_prefix_blocks.blocks;
        immutable_prefix_blocks.published_blocks +=
            snapshot.memory.immutable_prefix_blocks.published_blocks;
        immutable_prefix_blocks.restored_blocks +=
            snapshot.memory.immutable_prefix_blocks.restored_blocks;
        immutable_prefix_blocks.active_block_hits +=
            snapshot.memory.immutable_prefix_blocks.active_block_hits;
        immutable_prefix_blocks.idle_block_hits +=
            snapshot.memory.immutable_prefix_blocks.idle_block_hits;
        immutable_prefix_blocks.lookup_misses +=
            snapshot.memory.immutable_prefix_blocks.lookup_misses;
        immutable_prefix_blocks.evicted_blocks +=
            snapshot.memory.immutable_prefix_blocks.evicted_blocks;
        immutable_prefix_blocks.blocked_evictions +=
            snapshot.memory.immutable_prefix_blocks.blocked_evictions;
        immutable_prefix_blocks.pressure_evicted_blocks += snapshot
            .memory
            .immutable_prefix_blocks
            .pressure_evicted_blocks;
        immutable_prefix_blocks.ssd_block_hits +=
            snapshot.memory.immutable_prefix_blocks.ssd_block_hits;
        immutable_prefix_blocks.ssd_blocks_loaded +=
            snapshot.memory.immutable_prefix_blocks.ssd_blocks_loaded;
        immutable_prefix_blocks.ssd_blocks_queued +=
            snapshot.memory.immutable_prefix_blocks.ssd_blocks_queued;
        immutable_prefix_blocks.ssd_blocks_pending +=
            snapshot.memory.immutable_prefix_blocks.ssd_blocks_pending;
        immutable_prefix_blocks.ssd_store_backpressure += snapshot
            .memory
            .immutable_prefix_blocks
            .ssd_store_backpressure;
        immutable_prefix_blocks.ssd_load_pressure_skips += snapshot
            .memory
            .immutable_prefix_blocks
            .ssd_load_pressure_skips;
        immutable_prefix_blocks.dedup_saved_bytes +=
            snapshot.memory.immutable_prefix_blocks.dedup_saved_bytes;
        prompt_lookup_snapshots.push(snapshot.prompt_lookup);
        active_kv_snapshots.push(snapshot.active_kv_offload);
    }

    let active_kv_offload =
        ironmlx_runtime::core::cache::active_kv::ActiveKvOffloadHealth::aggregate(
            active_kv_snapshots,
        );
    let mtp_draft_tokens = if mtp_enabled && mtp_draft_token_values.len() == 1 {
        mtp_draft_token_values.first().copied()
    } else {
        None
    };
    let mtp_requested_draft_tokens = if mtp_enabled && mtp_requested_draft_token_values.len() == 1 {
        mtp_requested_draft_token_values.first().copied()
    } else {
        None
    };

    let prefix_store_backpressured =
        ironmlx_runtime::core::cache::prefix_store::process_async_prefix_store_queue()
            .is_backpressured();
    let (status, degraded_reasons) = classify_status(
        b_queued,
        queue_max,
        kv_cache_active_bytes,
        kv_cache_soft_limit_bytes,
        active_kv_offload.degraded,
        prefix_store_backpressured,
        &process_governor,
    );

    HealthSnapshot {
        status,
        degraded_reasons,
        uptime_secs: start_time.elapsed().as_secs(),
        model: ModelInfo {
            name: names.join(","),
            max_position_embeddings,
        },
        scheduler: SchedulerInfo {
            b_max,
            b_active,
            b_queued,
            queue_max,
            admit_count,
            batch_count,
            admission_queue_full_count,
            memory_budget_exceeded_count,
        },
        memory: MemoryInfo {
            total_ram_bytes,
            free_ram_bytes,
            available_ram_bytes: process_governor.available_ram_bytes,
            kv_cache_active_bytes,
            kv_cache_soft_limit_bytes,
            kv_cache_logical_cap_tokens,
            kv_cache_resident_cap_tokens,
            kv_cache_budget_policy: kv_cache_budget_policies.join(","),
            mlx_total_bytes: mlx_memory.total_bytes,
            mlx_max_recommended_bytes: mlx_memory.max_recommended_bytes,
            mlx_active_bytes: mlx_memory.active_bytes,
            mlx_cache_bytes: mlx_memory.cache_bytes,
            mlx_peak_bytes: mlx_memory.peak_bytes,
            mlx_memory_limit_bytes: mlx_memory.memory_limit_bytes,
            process_governor,
            prefix_store,
            immutable_prefix_blocks,
        },
        mtp: MtpHealthInfo {
            enabled: mtp_enabled,
            requested_draft_tokens: mtp_requested_draft_tokens,
            draft_tokens: mtp_draft_tokens,
            prefill_count: mtp_prefill_count,
            step_count: mtp_step_count,
            fallback_prefill_count: mtp_fallback_prefill_count,
            drafted_tokens: mtp_drafted_tokens,
            accepted_draft_tokens: mtp_accepted_draft_tokens,
            windows: mtp_windows,
            multi_token_windows: mtp_multi_token_windows,
            exact_sampling_windows: mtp_exact_sampling_windows,
            exact_acceptance_draws: mtp_exact_acceptance_draws,
            exact_residual_corrections: mtp_exact_residual_corrections,
            exact_bonus_samples: mtp_exact_bonus_samples,
            draft_forward_us: mtp_draft_forward_us,
            verify_forward_us: mtp_verify_forward_us,
            projection_us: mtp_projection_us,
            sampling_us: mtp_sampling_us,
            draft_host_sync_count: mtp_draft_host_sync_count,
            draft_host_sync_us: mtp_draft_host_sync_us,
            verify_accept_host_sync_count: mtp_verify_accept_host_sync_count,
            verify_accept_host_sync_us: mtp_verify_accept_host_sync_us,
            main_rollback_us: mtp_main_rollback_us,
            cache_commit_us: mtp_cache_commit_us,
            prefill_cache_commit_us: mtp_prefill_cache_commit_us,
            decode_cache_commit_us: mtp_decode_cache_commit_us,
            cache_restore_us: mtp_cache_restore_us,
            sampled_exact_qualification: neural_exact_qualification,
        },
        dflash2: ironmlx_runtime::core::runtime_health::DFlash2HealthConfig::disabled().snapshot(),
        prompt_lookup: PromptLookupHealthInfo::aggregate(prompt_lookup_snapshots),
        active_kv_offload,
        device_name: mlx_memory.device_name,
        version: env!("CARGO_PKG_VERSION"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::serve::ServeArgs;
    use ironmlx_lm::models::ModelArchitecture;
    use ironmlx_runtime::core::engine_pool::{
        EngineLoadPolicy, EngineModelCapabilities, EngineModelConfig,
    };
    use ironmlx_runtime::core::scheduler_resolution::{
        ResolvedSchedulerRuntime, SchedulerProfileSource,
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn app_daemon_models_route_exposes_registered_models() {
        let model_dir = unique_temp_dir("app-models-route");
        write_config(&model_dir, r#"{"model_type":"qwen3_5"}"#);
        let args = serve_args();
        let manager = ModelManager::new(
            crate::cli::serve::engine_runtime_config(&args).unwrap(),
            args.max_loaded_models,
            SchedulerResolutionOptions::from(&args),
        )
        .expect("model manager");
        manager
            .pool
            .register_dynamic_model(
                EngineModelConfig {
                    id: "mlx-community/Qwen3.6-27B-4bit".to_string(),
                    path: model_dir.clone(),
                    load_policy: EngineLoadPolicy::Lazy,
                    default: false,
                    pinned: false,
                    scheduler_runtime_profile: None,
                    mtp: None,
                    prompt_lookup: None,
                    sampling_defaults: SamplingDefaults::default(),
                    capabilities: EngineModelCapabilities::for_architecture(
                        ModelArchitecture::Qwen35Dense,
                        false,
                    ),
                },
                true,
                None,
            )
            .await
            .expect("register model");

        let response = app_router(manager)
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/models")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("models response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("models body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("models json");
        std::fs::remove_dir_all(model_dir).expect("remove temp model dir");

        assert_eq!(body["object"], "list");
        assert_eq!(body["data"].as_array().map(Vec::len), Some(1));
        assert_eq!(body["data"][0]["id"], "mlx-community/Qwen3.6-27B-4bit");
        assert_eq!(body["data"][0]["object"], "model");
        assert_eq!(body["data"][0]["created"], 0);
        assert_eq!(body["data"][0]["owned_by"], "ironmlx");
        assert_eq!(body["data"][0]["state"], "unloaded");
    }

    #[tokio::test]
    async fn app_daemon_resolve_errors_follow_protocol_contract() {
        let openai = crate::server::api_error::ApiError::engine_resolution(
            EngineRegistryError::AmbiguousDefault.into(),
        )
        .into_response(crate::server::api_error::ApiProtocol::OpenAi);
        assert_eq!(openai.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(openai.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "model_required");

        let anthropic = crate::server::api_error::ApiError::engine_resolution(
            EngineRegistryError::UnknownModel {
                id: "missing".to_owned(),
            }
            .into(),
        )
        .into_response(crate::server::api_error::ApiProtocol::Anthropic);
        assert_eq!(anthropic.status(), StatusCode::NOT_FOUND);
        let request_id = anthropic.headers()["request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let body = axum::body::to_bytes(anthropic.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "not_found_error");
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["request_id"], request_id);
    }

    #[test]
    fn loaded_model_info_serializes_only_enabled_per_model_active_kv_health() {
        let active_config = ironmlx_runtime::core::cache::active_kv::ActiveKvOffloadConfig::enabled(
            "/tmp/model-active-kv",
        );
        let active_stats = ironmlx_runtime::core::cache::active_kv::ActiveKvOffloadSharedStats::new(
            &active_config,
        );
        active_stats.set_parked_requests(2);
        active_stats.record_error();
        let mut info = EngineLoadedModelInfo {
            id: "model-a".to_string(),
            path: "/models/model-a".to_string(),
            architecture: "llama".to_string(),
            capabilities: EngineModelCapabilities::for_architecture(
                ModelArchitecture::Llama,
                false,
            ),
            runtime_state: EngineRuntimeState::Loaded,
            scheduler: Some("continuous_batching"),
            active_requests: 0,
            queued_requests: 0,
            queue_capacity: 8,
            usage: ironmlx_runtime::core::runtime_usage::ModelRuntimeUsageSnapshot::default(),
            active_kv_offload: Some(active_stats.snapshot()),
            is_default: true,
            pinned: false,
            max_position_embeddings: 4096,
            mtp_model_dir: None,
            mtp_draft_tokens: None,
            prompt_lookup: None,
        };

        let enabled = serde_json::to_value(LoadedModelInfo::from(info.clone()))
            .expect("serialize enabled Active KV health");
        assert_eq!(
            enabled["active_kv_offload"]["status"],
            serde_json::json!("degraded")
        );
        assert_eq!(
            enabled["active_kv_offload"]["parked_requests"],
            serde_json::json!(2)
        );
        assert_eq!(
            enabled["active_kv_offload"]["swap_error_count"],
            serde_json::json!(1)
        );

        info.active_kv_offload = None;
        let hidden = serde_json::to_value(LoadedModelInfo::from(info))
            .expect("serialize model without Active KV health");
        assert!(hidden.get("active_kv_offload").is_none());
    }

    #[test]
    fn load_model_request_accepts_per_model_max_cache_cap() {
        let request: LoadModelRequest = serde_json::from_value(serde_json::json!({
            "model": "mlx-community/LongContext-4bit",
            "model_dir": "/models/long",
            "set_default": true,
            "max_cache_cap": 65536,
            "pinned": true
        }))
        .expect("load request");

        assert_eq!(request.max_cache_cap, Some(65536));
        assert_eq!(request.pinned, Some(true));
    }

    #[test]
    fn load_model_request_accepts_sampling_defaults_and_idle_reload() {
        let request: LoadModelRequest = serde_json::from_value(serde_json::json!({
            "model": "mlx-community/LongContext-4bit",
            "model_dir": "/models/long",
            "reload_when_idle": true,
            "defer_when_busy": false,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.1
        }))
        .expect("load request");

        assert_eq!(request.reload_when_idle, Some(true));
        assert_eq!(request.defer_when_busy, Some(false));
        assert_eq!(request.sampling_defaults.temperature, Some(0.7));
        assert_eq!(request.sampling_defaults.top_p, Some(0.8));
        assert_eq!(request.sampling_defaults.top_k, Some(40));
        assert_eq!(request.sampling_defaults.repetition_penalty, Some(1.1));
    }

    #[test]
    fn load_model_request_accepts_mtp_settings() {
        let request: LoadModelRequest = serde_json::from_value(serde_json::json!({
            "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
            "model_dir": "/models/qwen",
            "mtp_model_dir": "/models/qwen-mtp",
            "mtp_draft_tokens": 2
        }))
        .expect("load request");

        assert_eq!(request.mtp_model_dir.as_deref(), Some("/models/qwen-mtp"));
        assert_eq!(request.mtp_draft_tokens, Some(2));
    }

    #[test]
    fn load_model_request_accepts_prompt_lookup_settings() {
        let request: LoadModelRequest = serde_json::from_value(serde_json::json!({
            "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
            "model_dir": "/models/qwen",
            "prompt_lookup": {
                "min_ngram": 2,
                "max_ngram": 5,
                "max_draft_tokens": 3,
                "history_window_tokens": 4096,
                "max_index_entries": 8192,
                "cross_request": true
            }
        }))
        .expect("load request");

        assert_eq!(
            request.prompt_lookup,
            Some(PromptLookupConfig {
                min_ngram: 2,
                max_ngram: 5,
                max_draft_tokens: 3,
                history_window_tokens: 4096,
                max_index_entries: 8192,
                cross_request: true,
            })
        );
    }

    #[test]
    fn clear_prompt_lookup_request_accepts_targeted_and_global_scope() {
        let targeted: ClearPromptLookupRequest = serde_json::from_value(serde_json::json!({
            "model": "mlx-community/Qwen3.5-4B-MLX-4bit"
        }))
        .expect("targeted clear request");
        assert_eq!(
            targeted.model.as_deref(),
            Some("mlx-community/Qwen3.5-4B-MLX-4bit")
        );

        let global: ClearPromptLookupRequest =
            serde_json::from_value(serde_json::json!({})).expect("global clear request");
        assert_eq!(global.model, None);

        let response = serde_json::to_value(ClearPromptLookupResponse {
            success: true,
            status: "cleared",
            model: None,
            cleared_models: 2,
            cleared_entries: 17,
        })
        .expect("clear response");
        assert_eq!(response["cleared_models"], 2);
        assert_eq!(response["cleared_entries"], 17);
        assert!(response.get("model").is_none());
    }

    #[test]
    fn parsed_load_model_request_accepts_hybrid_sources() {
        let root = unique_temp_dir("hybrid-load-request");
        let base = root.join("base");
        let mtp = root.join("mtp");
        std::fs::create_dir_all(&base).expect("base model dir");
        let request: LoadModelRequest = serde_json::from_value(serde_json::json!({
            "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
            "model_dir": base,
            "mtp_model_dir": mtp,
            "mtp_draft_tokens": 2,
            "prompt_lookup": {
                "min_ngram": 2,
                "max_ngram": 5,
                "max_draft_tokens": 3,
                "history_window_tokens": 4096,
                "max_index_entries": 8192,
                "cross_request": true
            }
        }))
        .expect("load request");

        let parsed = match parse_load_model_request(request) {
            Ok(parsed) => parsed,
            Err(_) => panic!("hybrid load request should parse"),
        };
        assert_eq!(
            parsed
                .mtp
                .as_ref()
                .map(|settings| settings.model_dir.as_path()),
            Some(mtp.as_path())
        );
        assert!(parsed.prompt_lookup.is_some());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn app_dynamic_gemma4_drafter_default_scheduler_uses_bmax_four() {
        let root = unique_temp_dir("app-gemma4-drafter-default-bmax");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(&base, &gemma4_base_config("gemma4", "gemma4_text", 2560));
        write_config(
            &mtp,
            &gemma4_assistant_config("gemma4_assistant", "gemma4_text", 2560),
        );
        let args = serve_args();

        let load = build_engine_model_config(
            &SchedulerResolutionOptions::from(&args),
            EngineModelBuildRequest {
                model_id: "gemma4-test".to_string(),
                model_dir: &base,
                max_cache_cap_override: None,
                sampling_defaults_override: SamplingDefaults::default(),
                mtp: Some(ironmlx_runtime::core::engine_pool::EngineMtpSettings {
                    model_dir: mtp,
                    draft_tokens: Some(2),
                }),
                prompt_lookup: None,
                pinned: false,
            },
        )
        .expect("build app dynamic model config");

        assert_eq!(
            load.config
                .scheduler_runtime_profile
                .as_ref()
                .expect("causal scheduler profile")
                .config
                .b_max,
            4
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn app_dynamic_diffusion_gemma_uses_block_diffusion_runtime_without_causal_scheduler() {
        let root = unique_temp_dir("app-diffusion-gemma-runtime");
        let model = root.join("model");
        write_config(
            &model,
            r#"{
                "model_type": "diffusion_gemma",
                "vision_config": {"hidden_size": 1152}
            }"#,
        );

        let load = build_engine_model_config(
            &SchedulerResolutionOptions::from(&serve_args()),
            EngineModelBuildRequest {
                model_id: "diffusion-gemma-test".to_string(),
                model_dir: &model,
                max_cache_cap_override: None,
                sampling_defaults_override: SamplingDefaults {
                    temperature: Some(0.7),
                    ..SamplingDefaults::default()
                },
                mtp: None,
                prompt_lookup: None,
                pinned: true,
            },
        )
        .expect("build DiffusionGemma app model config");

        assert!(load.config.scheduler_runtime_profile.is_none());
        assert!(load.config.mtp.is_none());
        assert!(load.config.prompt_lookup.is_none());
        assert_eq!(load.config.capabilities.runtime_kind, "block_diffusion");
        assert!(load.config.capabilities.supports_streaming);
        assert!(load.config.capabilities.supports_vision);
        assert!(!load.config.capabilities.supports_kv_cache);
        assert_eq!(
            load.config.capabilities.supported_sampling_parameters,
            &["max_tokens", "temperature", "seed"]
        );
        assert!(load.config.pinned);
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn app_dynamic_minicpm_reports_vision_capability_without_nested_vision_config() {
        let root = unique_temp_dir("app-minicpm-vision-capability");
        let model = root.join("model");
        write_config(&model, r#"{"model_type": "minicpmv4_6"}"#);

        let capabilities = engine_model_capabilities(ModelArchitecture::MiniCpmV46, &model)
            .expect("read MiniCPM capabilities");

        assert!(capabilities.supports_vision);
        assert_eq!(capabilities.runtime_kind, "causal");
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn app_dynamic_diffusion_gemma_rejects_causal_only_configuration_with_stable_code() {
        let root = unique_temp_dir("app-diffusion-gemma-capability-error");
        let model = root.join("model");
        write_config(
            &model,
            r#"{"model_type": "diffusion_gemma", "vision_config": {"hidden_size": 1152}}"#,
        );

        let error = match build_engine_model_config(
            &SchedulerResolutionOptions::from(&serve_args()),
            EngineModelBuildRequest {
                model_id: "diffusion-gemma-test".to_string(),
                model_dir: &model,
                max_cache_cap_override: Some(4096),
                sampling_defaults_override: SamplingDefaults::default(),
                mtp: None,
                prompt_lookup: None,
                pinned: false,
            },
        ) {
            Ok(_) => panic!("DiffusionGemma must reject KV-cache settings"),
            Err(error) => error,
        };
        let capability_error = error
            .downcast_ref::<ModelCapabilityError>()
            .expect("stable capability error");
        assert_eq!(
            capability_error.code,
            DIFFUSION_GEMMA_KV_CACHE_UNSUPPORTED_CODE
        );

        let admin = AdminError::from_load_error(error);
        assert_eq!(admin.code, Some(DIFFUSION_GEMMA_KV_CACHE_UNSUPPORTED_CODE));
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn aggregate_health_reports_loaded_mtp_state() {
        let snapshot = HealthSnapshot {
            status: ironmlx_runtime::core::runtime_health::HealthStatus::Healthy,
            degraded_reasons: Vec::new(),
            uptime_secs: 0,
            model: ModelInfo {
                name: "gemma4".to_string(),
                max_position_embeddings: 262_144,
            },
            scheduler: SchedulerInfo {
                b_max: 1,
                b_active: 0,
                b_queued: 0,
                queue_max: 32,
                admit_count: 0,
                batch_count: 0,
                admission_queue_full_count: 0,
                memory_budget_exceeded_count: 0,
            },
            memory: MemoryInfo {
                total_ram_bytes: 0,
                free_ram_bytes: 0,
                available_ram_bytes: None,
                kv_cache_active_bytes: 0,
                kv_cache_soft_limit_bytes: 1024,
                kv_cache_logical_cap_tokens: 262_144,
                kv_cache_resident_cap_tokens: 1024,
                kv_cache_budget_policy: "active_kv_offload".to_string(),
                mlx_total_bytes: None,
                mlx_max_recommended_bytes: None,
                mlx_active_bytes: 0,
                mlx_cache_bytes: 0,
                mlx_peak_bytes: 0,
                mlx_memory_limit_bytes: 0,
                process_governor:
                    ironmlx_runtime::core::process_memory::MemoryGovernorSnapshot::default(),
                prefix_store:
                    ironmlx_runtime::core::cache::prefix_store::AsyncPrefixStoreStats::default(),
                immutable_prefix_blocks:
                    ironmlx_runtime::core::scheduler_actor::ImmutablePrefixBlockHealth::default(),
            },
            mtp: MtpHealthInfo {
                enabled: true,
                requested_draft_tokens: Some(2),
                draft_tokens: Some(2),
                prefill_count: 3,
                step_count: 5,
                fallback_prefill_count: 7,
                drafted_tokens: 11,
                accepted_draft_tokens: 13,
                windows: 17,
                multi_token_windows: 11,
                exact_sampling_windows: 0,
                exact_acceptance_draws: 0,
                exact_residual_corrections: 0,
                exact_bonus_samples: 0,
                draft_forward_us: 19,
                verify_forward_us: 23,
                projection_us: 29,
                sampling_us: 31,
                draft_host_sync_count: 0,
                draft_host_sync_us: 0,
                verify_accept_host_sync_count: 17,
                verify_accept_host_sync_us: 53,
                main_rollback_us: 37,
                cache_commit_us: 41,
                prefill_cache_commit_us: 17,
                decode_cache_commit_us: 24,
                cache_restore_us: 43,
                sampled_exact_qualification: NeuralExactQualificationHealth::default(),
            },
            dflash2: ironmlx_runtime::core::runtime_health::DFlash2HealthConfig::disabled()
                .snapshot(),
            prompt_lookup: PromptLookupHealthInfo::default(),
            active_kv_offload:
                ironmlx_runtime::core::cache::active_kv::ActiveKvOffloadHealth::disabled(),
            device_name: None,
            version: "test",
        };

        let aggregated = aggregate_health(Instant::now(), vec![snapshot]);

        assert!(aggregated.mtp.enabled);
        assert_eq!(aggregated.mtp.requested_draft_tokens, Some(2));
        assert_eq!(aggregated.mtp.draft_tokens, Some(2));
        assert_eq!(aggregated.mtp.prefill_count, 3);
        assert_eq!(aggregated.mtp.step_count, 5);
        assert_eq!(aggregated.mtp.fallback_prefill_count, 7);
        assert_eq!(aggregated.mtp.drafted_tokens, 11);
        assert_eq!(aggregated.mtp.accepted_draft_tokens, 13);
        assert_eq!(aggregated.mtp.windows, 17);
        assert_eq!(aggregated.mtp.multi_token_windows, 11);
        assert_eq!(aggregated.mtp.draft_forward_us, 19);
        assert_eq!(aggregated.mtp.verify_forward_us, 23);
        assert_eq!(aggregated.mtp.projection_us, 29);
        assert_eq!(aggregated.mtp.sampling_us, 31);
        assert_eq!(aggregated.mtp.main_rollback_us, 37);
        assert_eq!(aggregated.mtp.cache_commit_us, 41);
        assert_eq!(aggregated.mtp.prefill_cache_commit_us, 17);
        assert_eq!(aggregated.mtp.decode_cache_commit_us, 24);
        assert_eq!(aggregated.mtp.cache_restore_us, 43);
    }

    #[test]
    fn mtp_validation_accepts_compatible_qwen_pair_without_loading_weights() {
        let root = unique_temp_dir("mtp-validation-compatible");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(&base, &qwen35_config("qwen3_5", 0, 2560));
        write_config(&mtp, &qwen35_config("qwen3_5_mtp", 1, 2560));

        let response = validate_mtp_pair(&base, &mtp, Some(2)).expect("validate");

        assert!(response.compatible);
        assert_eq!(response.reason_code, "ok");
        assert_eq!(response.draft_tokens, Some(2));

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mtp_validation_accepts_same_qwen_lineage_from_snapshot_metadata() {
        let root = unique_temp_dir("mtp-validation-qwen38-lineage");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(
            &base,
            &without_model_identity(qwen35_config("qwen3_5", 0, 5120)),
        );
        write_config(
            &mtp,
            &without_model_identity(qwen35_config("qwen3_5_mtp", 1, 5120)),
        );
        write_snapshot_identity(&base, "mlx-community/Qwen3.8-27B-8bit");
        write_snapshot_identity(&mtp, "mlx-community/Qwen3.8-27B-MTP-8bit");

        let response = validate_mtp_pair(&base, &mtp, Some(2)).expect("validate");

        assert!(response.compatible, "response={response:?}");
        assert_eq!(response.reason_code, MTP_OK_CODE);

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mtp_validation_rejects_cross_generation_qwen_with_identical_structure() {
        let root = unique_temp_dir("mtp-validation-qwen-cross-generation");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(
            &base,
            &without_model_identity(qwen35_config("qwen3_5", 0, 5120)),
        );
        write_config(
            &mtp,
            &without_model_identity(qwen35_config("qwen3_5_mtp", 1, 5120)),
        );
        write_snapshot_identity(&base, "mlx-community/Qwen3.8-27B-4bit");
        write_snapshot_identity(&mtp, "mlx-community/Qwen3.6-27B-MTP-4bit");

        let response = validate_mtp_pair(&base, &mtp, Some(2)).expect("validate");

        assert!(!response.compatible);
        assert_eq!(response.reason_code, MTP_INCOMPATIBLE_CODE);
        assert!(response.message.contains("lineage mismatch"));

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mtp_validation_rejects_pair_without_lineage_metadata() {
        let root = unique_temp_dir("mtp-validation-missing-lineage");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(
            &base,
            &without_model_identity(qwen35_config("qwen3_5", 0, 2560)),
        );
        write_config(
            &mtp,
            &without_model_identity(qwen35_config("qwen3_5_mtp", 1, 2560)),
        );

        let response = validate_mtp_pair(&base, &mtp, Some(2)).expect("validate");

        assert!(!response.compatible);
        assert_eq!(response.reason_code, MTP_INCOMPATIBLE_CODE);
        assert!(response.message.contains("could not be established"));

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mtp_validation_ignores_empty_vision_config_on_qwen_mtp_artifact() {
        let root = unique_temp_dir("mtp-validation-empty-vision-config");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(&base, &qwen35_config("qwen3_5", 0, 2560));
        let mut mtp_config: serde_json::Value =
            serde_json::from_str(&qwen35_config("qwen3_5_mtp", 1, 2560))
                .expect("parse test MTP config");
        mtp_config["vision_config"] = serde_json::json!({});
        write_config(&mtp, &mtp_config.to_string());

        let response = validate_mtp_pair(&base, &mtp, Some(2)).expect("validate");

        assert!(response.compatible, "response={response:?}");
        assert_eq!(response.reason_code, MTP_OK_CODE);
        assert_eq!(response.draft_tokens, Some(2));

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mtp_validation_returns_stable_code_for_invalid_qwen_mtp_config() {
        let root = unique_temp_dir("mtp-validation-invalid-config");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(&base, &qwen35_config("qwen3_5", 0, 2560));
        write_config(
            &mtp,
            r#"{
                "model_type": "qwen3_5_mtp",
                "text_config": {"mtp_num_hidden_layers": 1},
                "vision_config": {}
            }"#,
        );

        let response = validate_mtp_pair(&base, &mtp, Some(2)).expect("validate");

        assert!(!response.compatible);
        assert_eq!(response.reason_code, MTP_INVALID_CONFIG_CODE);

        let error = AdminError::from_load_error(anyhow::anyhow!(
            "MTP validation failed: {}: {}",
            response.reason_code,
            response.message
        ));
        let value = serde_json::to_value(AdminModelResponse::from_error(error.message, error.code))
            .expect("response json");
        assert_eq!(value["code"], MTP_INVALID_CONFIG_CODE);
        assert_ne!(value["error"], response.message);

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mtp_validation_rejects_mismatched_qwen_pair_with_stable_reason_code() {
        let root = unique_temp_dir("mtp-validation-mismatch");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(&base, &qwen35_config("qwen3_5", 0, 2560));
        write_config(&mtp, &qwen35_config("qwen3_5_mtp", 1, 4096));

        let response = validate_mtp_pair(&base, &mtp, None).expect("validate");

        assert!(!response.compatible);
        assert_eq!(response.reason_code, MTP_INCOMPATIBLE_CODE);

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mtp_validation_accepts_compatible_qwen_moe_pair_without_dense_intermediate_size() {
        let root = unique_temp_dir("mtp-validation-moe-compatible");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(&base, &qwen35_moe_config("qwen3_5_moe", 0, 2048, 512));
        write_config(&mtp, &qwen35_moe_config("qwen3_5_mtp", 1, 2048, 512));

        let response = validate_mtp_pair(&base, &mtp, Some(2)).expect("validate");

        assert!(response.compatible, "response={response:?}");
        assert_eq!(response.reason_code, "ok");
        assert_eq!(response.draft_tokens, Some(2));

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mtp_validation_accepts_compatible_gemma4_assistant_pair() {
        let root = unique_temp_dir("mtp-validation-gemma4-compatible");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(&base, &gemma4_base_config("gemma4", "gemma4_text", 2560));
        write_config(
            &mtp,
            &gemma4_assistant_config("gemma4_assistant", "gemma4_text", 2560),
        );

        let response = validate_mtp_pair(&base, &mtp, Some(3)).expect("validate");

        assert!(response.compatible, "response={response:?}");
        assert_eq!(response.reason_code, "ok");
        assert_eq!(response.draft_tokens, Some(3));

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mtp_validation_rejects_gemma4_assistant_backbone_mismatch() {
        let root = unique_temp_dir("mtp-validation-gemma4-mismatch");
        let base = root.join("base");
        let mtp = root.join("mtp");
        write_config(&base, &gemma4_base_config("gemma4", "gemma4_text", 2560));
        write_config(
            &mtp,
            &gemma4_assistant_config("gemma4_assistant", "gemma4_text", 3840),
        );

        let response = validate_mtp_pair(&base, &mtp, None).expect("validate");

        assert!(!response.compatible);
        assert_eq!(response.reason_code, MTP_INCOMPATIBLE_CODE);

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn load_model_request_does_not_default_to_default_takeover() {
        let request: LoadModelRequest = serde_json::from_value(serde_json::json!({
            "model": "mlx-community/New-4bit",
            "model_dir": "/models/new"
        }))
        .expect("load request");

        assert!(!request.set_default.unwrap_or(false));
    }

    #[test]
    fn load_error_maps_engine_pool_capacity_reached() {
        let error = AdminError::from_load_error(anyhow::anyhow!(
            "engine pool capacity reached: max_loaded_models=3, unload an existing model before loading `delta`"
        ));

        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.message, MAX_LOADED_MODELS_REACHED_MESSAGE);
        assert_eq!(error.code, Some(MAX_LOADED_MODELS_REACHED_CODE));
        assert!(error
            .message
            .contains("Maximum concurrent loaded models reached"));
    }

    #[test]
    fn load_error_maps_model_memory_limit_exceeded() {
        let error = AdminError::from_load_error(anyhow::anyhow!(
            "engine pool model memory limit exceeded: loaded_model_bytes=42949672960 > limit=32212254720"
        ));

        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.message, MODEL_MEMORY_LIMIT_EXCEEDED_MESSAGE);
        assert_eq!(error.code, Some(MODEL_MEMORY_LIMIT_EXCEEDED_CODE));
    }

    #[test]
    fn load_error_maps_total_memory_limit_exceeded() {
        let error = AdminError::from_load_error(anyhow::anyhow!(
            "engine pool total memory limit exceeded: mlx_active_bytes=68719476736 > limit=64424509440"
        ));

        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.message, TOTAL_MEMORY_LIMIT_EXCEEDED_MESSAGE);
        assert_eq!(error.code, Some(TOTAL_MEMORY_LIMIT_EXCEEDED_CODE));
    }

    #[test]
    fn load_error_maps_memory_budget_to_actionable_code() {
        let error = AdminError::from_load_error(anyhow::anyhow!(
            "memory budget exceeded: b_max=1 × resident_cap=262144 × 786432 bytes/token = 206158430208 bytes > available 126701535232 (logical cap 262144, policy full_resident). Lower --b-max or --max-cache-cap."
        ));

        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.code, Some("kv_memory_budget_exceeded"));
        assert!(error.message.contains("Active KV offload"));
        assert!(error.message.contains("MAX CONTEXT TOKENS"));
    }

    #[test]
    fn load_error_serializes_engine_pool_capacity_code() {
        let error = AdminError::from_load_error(anyhow::anyhow!(
            "engine pool capacity reached: max_loaded_models=3, unload an existing model before loading `delta`"
        ));
        let response = AdminModelResponse::from_error(error.message, error.code);
        let value = serde_json::to_value(response).expect("response json");

        assert_eq!(value["success"], false);
        assert_eq!(value["code"], "max_loaded_models_reached");
        assert_eq!(value["error"], MAX_LOADED_MODELS_REACHED_MESSAGE);
    }

    #[test]
    fn admin_response_serializes_warning_code() {
        let response = AdminModelResponse::ok(
            "loaded",
            Some("mlx-community/Tiny-4bit".to_string()),
            Vec::new(),
            Some(AdminWarning::new(
                DEFAULT_PROFILE_WARNING_CODE,
                DEFAULT_PROFILE_WARNING,
            )),
        );
        let value = serde_json::to_value(response).expect("response json");

        assert_eq!(value["success"], true);
        assert_eq!(value["warning_code"], "default_scheduler_profile_used");
        assert_eq!(value["warning"], DEFAULT_PROFILE_WARNING);
    }

    #[test]
    fn admin_model_required_error_serializes_code() {
        let error = AdminError::model_required();
        let response = AdminModelResponse::from_error(error.message, error.code);
        let value = serde_json::to_value(response).expect("response json");

        assert_eq!(value["success"], false);
        assert_eq!(value["code"], "model_required");
        assert_eq!(value["error"], MODEL_REQUIRED_MESSAGE);
    }

    #[test]
    fn unload_unknown_model_error_serializes_model_not_loaded_code() {
        let error = AdminError::from_control_error(
            EngineRegistryError::UnknownModel {
                id: "missing".to_string(),
            }
            .into(),
        );
        let response = AdminModelResponse::from_error(error.message, error.code);
        let value = serde_json::to_value(response).expect("response json");

        assert_eq!(value["success"], false);
        assert_eq!(value["code"], "model_not_loaded");
        assert_eq!(value["error"], "Model is not loaded: missing");
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ironmlx-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ))
    }

    fn serve_args() -> ServeArgs {
        ServeArgs {
            model: None,
            model_id: None,
            model_manifest: None,
            max_loaded_models: None,
            memory_limit_total_gb: None,
            memory_limit_model_gb: None,
            port: 8080,
            host: "127.0.0.1".to_string(),
            network_mode: crate::server::security::NetworkMode::Local,
            lan_host: None,
            security_bootstrap_stdin: false,
            network_config: Some(
                crate::server::security::ServerNetworkConfig::local("127.0.0.1", 8080).unwrap(),
            ),
            prefill_chunk_size: None,
            force_scheduler: false,
            b_max: None,
            admission_deadline_ms: None,
            admission_queue_max: None,
            max_cache_cap: None,
            decode_cadence_mid_chunk_cap: None,
            scheduler_profile: None,
            scheduler_autotune_report: false,
            mtp_model_dir: None,
            mtp_draft_tokens: None,
            dflash2_model_dir: None,
            dflash2_block_size: 4,
            dflash2_draft_bits: 4,
            dflash2_tensor_batch_max_width: None,
            prompt_lookup: false,
            prompt_lookup_min_ngram: None,
            prompt_lookup_max_ngram: None,
            prompt_lookup_max_draft_tokens: None,
            prompt_lookup_history_window_tokens: None,
            prompt_lookup_max_index_entries: None,
            prompt_lookup_cross_request: false,
            kv_quant: crate::cli::KvQuantArg::None,
            paged_prefix_cache_dir: None,
            paged_prefix_cache_block_size:
                ironmlx_runtime::core::cache::prefix_store::DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE,
            paged_prefix_cache_max_pages: None,
            ssd_prefix_cache_max_gb: None,
            prefix_lru_cache_max_bytes: None,
            model_ttl_minutes: None,
            active_kv_offload: false,
            active_kv_offload_dir: None,
        }
    }

    fn write_config(dir: &Path, config: &str) {
        std::fs::create_dir_all(dir).expect("temp model dir");
        std::fs::write(dir.join("config.json"), config).expect("write config");
    }

    fn write_snapshot_identity(dir: &Path, repo_id: &str) {
        std::fs::write(
            dir.join(".ironmlx-snapshot.json"),
            serde_json::json!({"repo_id": repo_id}).to_string(),
        )
        .expect("write snapshot identity");
    }

    fn without_model_identity(config: String) -> String {
        let mut raw: serde_json::Value = serde_json::from_str(&config).expect("parse config");
        raw.as_object_mut()
            .expect("config object")
            .remove("_name_or_path");
        raw.to_string()
    }

    fn qwen35_config(model_type: &str, mtp_layers: i32, hidden_size: i32) -> String {
        let model_identity = if model_type == "qwen3_5_mtp" {
            "mlx-community/Qwen3.5-4B-MTP-4bit"
        } else {
            "mlx-community/Qwen3.5-4B-MLX-4bit"
        };
        format!(
            r#"{{
                "model_type": "{model_type}",
                "_name_or_path": "{model_identity}",
                "text_config": {{
                    "hidden_size": {hidden_size},
                    "intermediate_size": 9728,
                    "num_hidden_layers": 36,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 256,
                    "vocab_size": 151936,
                    "rms_norm_eps": 0.000001,
                    "attention_bias": false,
                    "tie_word_embeddings": false,
                    "full_attention_interval": 4,
                    "linear_num_value_heads": 16,
                    "linear_num_key_heads": 16,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "mtp_num_hidden_layers": {mtp_layers},
                    "max_position_embeddings": 262144
                }}
            }}"#
        )
    }

    fn gemma4_base_config(model_type: &str, text_model_type: &str, hidden_size: i32) -> String {
        format!(
            r#"{{
                "model_type": "{model_type}",
                "_name_or_path": "mlx-community/gemma-4-e4b-it-4bit",
                "text_config": {{
                    "model_type": "{text_model_type}",
                    "hidden_size": {hidden_size},
                    "num_hidden_layers": 42,
                    "intermediate_size": 10240,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "vocab_size": 262144,
                    "vocab_size_per_layer_input": 262144,
                    "num_key_value_heads": 2,
                    "num_kv_shared_layers": 18,
                    "hidden_size_per_layer_input": 256,
                    "layer_types": [
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention"
                    ],
                    "tie_word_embeddings": true
                }}
            }}"#
        )
    }

    fn gemma4_assistant_config(
        model_type: &str,
        text_model_type: &str,
        backbone_hidden_size: i32,
    ) -> String {
        format!(
            r#"{{
                "model_type": "{model_type}",
                "base_model_name_or_path": "google/gemma-4-e4b-it",
                "backbone_hidden_size": {backbone_hidden_size},
                "use_ordered_embeddings": true,
                "num_centroids": 2048,
                "centroid_intermediate_top_k": 32,
                "tie_word_embeddings": true,
                "text_config": {{
                    "model_type": "{text_model_type}",
                    "hidden_size": 256,
                    "num_hidden_layers": 4,
                    "intermediate_size": 2048,
                    "num_attention_heads": 4,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "vocab_size": 262144,
                    "num_key_value_heads": 2,
                    "num_kv_shared_layers": 4,
                    "hidden_size_per_layer_input": 0,
                    "layer_types": [
                        "sliding_attention", "sliding_attention",
                        "sliding_attention", "full_attention"
                    ],
                    "tie_word_embeddings": true
                }}
            }}"#
        )
    }

    fn qwen35_moe_config(
        model_type: &str,
        mtp_layers: i32,
        hidden_size: i32,
        moe_intermediate_size: i32,
    ) -> String {
        let model_identity = if model_type == "qwen3_5_mtp" {
            "mlx-community/Qwen3.6-35B-A3B-MTP-4bit"
        } else {
            "mlx-community/Qwen3.6-35B-A3B-4bit"
        };
        format!(
            r#"{{
                "model_type": "{model_type}",
                "_name_or_path": "{model_identity}",
                "text_config": {{
                    "model_type": "qwen3_5_moe_text",
                    "hidden_size": {hidden_size},
                    "num_hidden_layers": 40,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "vocab_size": 248320,
                    "rms_norm_eps": 0.000001,
                    "attention_bias": false,
                    "tie_word_embeddings": false,
                    "full_attention_interval": 4,
                    "linear_num_value_heads": 32,
                    "linear_num_key_heads": 16,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "moe_intermediate_size": {moe_intermediate_size},
                    "shared_expert_intermediate_size": {moe_intermediate_size},
                    "mtp_num_hidden_layers": {mtp_layers},
                    "max_position_embeddings": 262144
                }}
            }}"#
        )
    }

    #[test]
    fn generation_config_file_supplies_sampling_defaults() {
        let root = std::env::temp_dir().join(format!(
            "ironmlx-generation-config-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("temp model dir");
        std::fs::write(
            root.join("generation_config.json"),
            r#"{
              "temperature": 0.65,
              "top_p": 0.9,
              "top_k": 32,
              "repetition_penalty": 1.08
            }"#,
        )
        .expect("generation config");

        let defaults = read_generation_sampling_defaults(&root).expect("defaults");

        assert_eq!(defaults.temperature, Some(0.65));
        assert_eq!(defaults.top_p, Some(0.9));
        assert_eq!(defaults.top_k, Some(32));
        assert_eq!(defaults.repetition_penalty, Some(1.08));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn per_model_max_cache_cap_overrides_scheduler_config_and_rules() {
        let resolved = ResolvedSchedulerRuntime {
            scheduler_config: ironmlx_runtime::core::scheduler_resolution::SchedulerServeConfig {
                max_cache_cap: 32768,
                ..Default::default()
            },
            scheduler_runtime_profile: ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile {
                schema_version: ironmlx_runtime::core::scheduler_autotune::SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
                model_name: "test-model".to_string(),
                hardware_label: "test-hardware".to_string(),
                runtime_context: ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeContext::local_default(32768),
                config: ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneProfileConfig {
                    b_max: 1,
                    prefill_chunk_size: 1024,
                    admission_deadline_ms: 5,
                    admission_queue_max: 32,
                    max_cache_cap: 32768,
                    decode_cadence_mid_chunk_cap: 256,
                },
                rules: vec![ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeRule {
                    when: ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeRuleCondition {
                        prompt_len_gte: 8192,
                        max_new_tokens_gte: 128,
                        effective_concurrency_gte: 1,
                    },
                    config: ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneProfileConfig {
                        b_max: 1,
                        prefill_chunk_size: 2048,
                        admission_deadline_ms: 5,
                        admission_queue_max: 32,
                        max_cache_cap: 32768,
                        decode_cadence_mid_chunk_cap: 512,
                    },
                }],
                metadata: ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata::synthetic(0),
            },
            profile_source: Some(SchedulerProfileSource::Store),
        };

        let overridden = apply_load_request_scheduler_overrides(resolved, Some(65536));

        assert_eq!(overridden.scheduler_config.max_cache_cap, 65536);
        assert_eq!(
            overridden.scheduler_runtime_profile.config.max_cache_cap,
            65536
        );
        assert_eq!(
            overridden.scheduler_runtime_profile.rules[0]
                .config
                .max_cache_cap,
            65536
        );
    }
}

use {
    ironmlx_runtime::core::model_validation::MTP_BASE_MODEL_NOT_FOUND_CODE,
    ironmlx_runtime::core::model_validation::MTP_INCOMPATIBLE_CODE,
    ironmlx_runtime::core::model_validation::MTP_INVALID_CONFIG_CODE,
    ironmlx_runtime::core::model_validation::MTP_INVALID_DRAFT_TOKENS_CODE,
    ironmlx_runtime::core::model_validation::MTP_INVALID_MODEL_TYPE_CODE,
    ironmlx_runtime::core::model_validation::MTP_MODEL_NOT_FOUND_CODE,
    ironmlx_runtime::core::model_validation::MTP_UNSUPPORTED_ARCHITECTURE_CODE,
};

impl From<ironmlx_runtime::core::model_validation::MtpCompatibility> for MtpValidationResponse {
    fn from(value: ironmlx_runtime::core::model_validation::MtpCompatibility) -> Self {
        Self {
            compatible: value.compatible,
            reason_code: value.reason_code,
            message: value.message,
            draft_tokens: value.draft_tokens,
        }
    }
}

use super::engine::EnginePoolHttpAdapter;

#[cfg(test)]
use ironmlx_runtime::core::model_validation::MTP_OK_CODE;

impl From<ModelManagementOutcome> for AdminModelResponse {
    fn from(outcome: ModelManagementOutcome) -> Self {
        let status = match outcome.status {
            ModelManagementStatus::AlreadyLoaded => "already_loaded",
            ModelManagementStatus::ReloadDeferred => "reload_deferred",
            ModelManagementStatus::Loaded => "loaded",
            ModelManagementStatus::Registered => "registered",
            ModelManagementStatus::Reloaded => "reloaded",
        };
        Self::ok(
            status,
            outcome.model,
            outcome
                .loaded_models
                .into_iter()
                .map(LoadedModelInfo::from)
                .collect(),
            outcome.warning,
        )
    }
}
