use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::cli::serve::{
    read_model_type, resolve_active_kv_offload_config, resolve_engine_paged_prefix_cache_settings,
    resolve_model_ttl, resolve_scheduler_for_model, ResolvedSchedulerRuntime,
    SchedulerProfileSource, ServeArgs,
};
use crate::models::ModelArchitecture;
use crate::Result;

use super::engine::{
    EngineLoadPolicy, EngineLoadedModelInfo, EngineModelConfig, EnginePoolRuntimeConfig,
    EnginePoolState, EngineRegistryError, EngineRuntimeState,
};
use super::health::{
    classify_status, system_free_ram_bytes, HealthSnapshot, HealthStatus, MemoryInfo, ModelInfo,
    MtpHealthInfo, SchedulerInfo,
};
use super::{anthropic, openai, SamplingDefaults};

const MODEL_REQUIRED_CODE: &str = "model_required";
const MODEL_REQUIRED_MESSAGE: &str = "Model is required.";
const MODEL_DIRECTORY_NOT_FOUND_CODE: &str = "model_directory_not_found";
const INVALID_MAX_CACHE_CAP_CODE: &str = "invalid_max_cache_cap";
const INVALID_MAX_CACHE_CAP_MESSAGE: &str = "MAX TOKENS must be greater than or equal to 1.";
const MODEL_NOT_LOADED_CODE: &str = "model_not_loaded";
const MODEL_NOT_REGISTERED_CODE: &str = "model_not_registered";
const BACKEND_UNLOAD_ERROR_CODE: &str = "backend_unload_error";
const GPU_MEMORY_INSUFFICIENT_CODE: &str = "gpu_memory_insufficient";
const GPU_MEMORY_INSUFFICIENT_MESSAGE: &str =
    "Not enough available GPU memory to safely load this model. Unload one or more unused loaded models to free GPU memory, then try again.";
const MAX_LOADED_MODELS_REACHED_CODE: &str = "max_loaded_models_reached";
const MAX_LOADED_MODELS_REACHED_MESSAGE: &str =
    "Maximum concurrent loaded models reached. Unload an unused loaded model before loading another model.";
const DEFAULT_PROFILE_WARNING_CODE: &str = "default_scheduler_profile_used";
const DEFAULT_PROFILE_WARNING: &str =
    "No matching scheduler profile was found for this model. The model is running with the default scheduler configuration. Generate a dedicated profile with scheduler-autotune for better model-specific scheduling.";
const MODEL_RELOAD_DEFERRED_WARNING_CODE: &str = "model_reload_deferred";
const MODEL_RELOAD_DEFERRED_WARNING: &str =
    "The model is processing requests. New parameters will be applied automatically after the model becomes idle.";

#[derive(Clone)]
pub struct ModelManager {
    pending_reloads: Arc<RwLock<HashMap<String, PendingModelReload>>>,
    pool: EnginePoolState,
    serve_args: ServeArgs,
    start_time: Instant,
}

impl ModelManager {
    pub fn new(serve_args: ServeArgs) -> Result<Self> {
        let runtime = engine_runtime_config(&serve_args)?;
        let pool = EnginePoolState::new_dynamic(runtime, serve_args.max_loaded_models)?;
        Ok(Self {
            pending_reloads: Arc::new(RwLock::new(HashMap::new())),
            pool,
            serve_args,
            start_time: Instant::now(),
        })
    }

    fn start_model_ttl_sweeper(&self) {
        self.pool.start_model_ttl_sweeper();
    }

    async fn load_model(
        &self,
        request: LoadModelRequest,
    ) -> std::result::Result<AdminModelResponse, AdminError> {
        let parsed = ParsedLoadModelRequest::new(request)?;
        let reload = PendingModelReload {
            model_reference: parsed.model_reference.clone(),
            model_dir: parsed.model_dir.clone(),
            max_cache_cap_override: parsed.max_cache_cap_override,
            sampling_defaults_override: parsed.sampling_defaults,
            set_default: parsed.set_default,
        };

        let already_loaded = self.pool.is_model_loaded(&parsed.model_reference).await;
        if already_loaded && !parsed.reload_when_idle {
            return Ok(AdminModelResponse::ok(
                "already_loaded",
                Some(parsed.model_reference),
                self.list_loaded().await,
                None,
            ));
        }
        if already_loaded
            && self
                .pool
                .pending_requests(&parsed.model_reference)
                .await
                .is_some_and(|requests| requests > 0)
        {
            self.schedule_reload_when_idle(reload).await;
            return Ok(AdminModelResponse::ok(
                "reload_deferred",
                Some(parsed.model_reference),
                self.list_loaded().await,
                Some(AdminWarning::new(
                    MODEL_RELOAD_DEFERRED_WARNING_CODE,
                    MODEL_RELOAD_DEFERRED_WARNING,
                )),
            ));
        }

        if already_loaded && parsed.reload_when_idle {
            return self.reload_model_now(reload).await;
        }

        ensure_gpu_memory_headroom()?;
        let load = build_engine_model_config(
            &self.serve_args,
            parsed.model_reference.clone(),
            &parsed.model_dir,
            parsed.max_cache_cap_override,
            parsed.sampling_defaults,
        )
        .map_err(AdminError::from_load_error)?;
        self.pool
            .reload_dynamic_model(load.config, parsed.set_default)
            .await
            .map_err(AdminError::from_load_error)?;
        let loaded_models = self.list_loaded().await;
        Ok(AdminModelResponse::ok(
            "loaded",
            Some(parsed.model_reference),
            loaded_models,
            load.warning,
        ))
    }

    async fn register_model(
        &self,
        request: LoadModelRequest,
    ) -> std::result::Result<AdminModelResponse, AdminError> {
        let parsed = ParsedLoadModelRequest::new(request)?;
        let load = build_engine_model_config(
            &self.serve_args,
            parsed.model_reference.clone(),
            &parsed.model_dir,
            parsed.max_cache_cap_override,
            parsed.sampling_defaults,
        )
        .map_err(AdminError::from_load_error)?;
        self.pool
            .register_dynamic_model(load.config, parsed.set_default)
            .await
            .map_err(AdminError::from_load_error)?;
        Ok(AdminModelResponse::ok(
            "registered",
            Some(parsed.model_reference),
            self.list_loaded().await,
            load.warning,
        ))
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

    async fn health_snapshot(&self) -> HealthSnapshot {
        let snapshots = self.pool.loaded_causal_health_snapshots().await;
        aggregate_health(self.start_time, snapshots)
    }

    async fn openai(&self, req: openai::ChatRequest) -> Response {
        match self.pool.app_openai_chat_completions(req).await {
            Ok(response) => response,
            Err(error) => resolve_error_response(error),
        }
    }

    async fn anthropic(&self, req: anthropic::MessagesRequest) -> Response {
        match self.pool.app_anthropic_messages(req).await {
            Ok(response) => response,
            Err(error) => resolve_error_response(error),
        }
    }

    async fn reload_model_now(
        &self,
        reload: PendingModelReload,
    ) -> std::result::Result<AdminModelResponse, AdminError> {
        if self
            .pool
            .pending_requests(&reload.model_reference)
            .await
            .is_some_and(|requests| requests > 0)
        {
            self.schedule_reload_when_idle(reload.clone()).await;
            return Ok(AdminModelResponse::ok(
                "reload_deferred",
                Some(reload.model_reference),
                self.list_loaded().await,
                Some(AdminWarning::new(
                    MODEL_RELOAD_DEFERRED_WARNING_CODE,
                    MODEL_RELOAD_DEFERRED_WARNING,
                )),
            ));
        }
        if !self.pool.is_model_loaded(&reload.model_reference).await {
            return Err(AdminError::model_not_loaded(&reload.model_reference));
        }

        let load = build_engine_model_config(
            &self.serve_args,
            reload.model_reference.clone(),
            &reload.model_dir,
            reload.max_cache_cap_override,
            reload.sampling_defaults_override,
        )
        .map_err(AdminError::from_load_error)?;
        self.pool
            .reload_dynamic_model(load.config, reload.set_default)
            .await
            .map_err(AdminError::from_load_error)?;
        let loaded_models = self.list_loaded().await;
        Ok(AdminModelResponse::ok(
            "reloaded",
            Some(reload.model_reference),
            loaded_models,
            load.warning,
        ))
    }

    async fn schedule_reload_when_idle(&self, reload: PendingModelReload) {
        let model_reference = reload.model_reference.clone();
        self.pending_reloads
            .write()
            .await
            .insert(model_reference.clone(), reload);
        let pending_reloads = self.pending_reloads.clone();
        let pool = self.pool.clone();
        let serve_args = self.serve_args.clone();

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(500)).await;
                let still_pending = pending_reloads.read().await.contains_key(&model_reference);
                if !still_pending {
                    return;
                }
                let busy = pool
                    .pending_requests(&model_reference)
                    .await
                    .is_some_and(|requests| requests > 0);
                if busy {
                    continue;
                }
                let reload = pending_reloads.write().await.remove(&model_reference);
                let Some(reload) = reload else {
                    return;
                };
                if pool
                    .pending_requests(&model_reference)
                    .await
                    .is_some_and(|requests| requests > 0)
                {
                    pending_reloads
                        .write()
                        .await
                        .insert(model_reference.clone(), reload);
                    continue;
                }
                let load = build_engine_model_config(
                    &serve_args,
                    reload.model_reference.clone(),
                    &reload.model_dir,
                    reload.max_cache_cap_override,
                    reload.sampling_defaults_override,
                );
                match load {
                    Ok(load) => {
                        if let Err(error) = pool
                            .reload_dynamic_model(load.config, reload.set_default)
                            .await
                        {
                            tracing::error!(
                                "failed to reload model {} after idle: {error:#}",
                                reload.model_reference
                            );
                        }
                    }
                    Err(error) => tracing::error!(
                        "failed to build reload config for model {} after idle: {error:#}",
                        reload.model_reference
                    ),
                }
                return;
            }
        });
    }
}

#[derive(Clone)]
struct PendingModelReload {
    model_reference: String,
    model_dir: PathBuf,
    max_cache_cap_override: Option<usize>,
    sampling_defaults_override: SamplingDefaults,
    set_default: bool,
}

struct EngineModelLoad {
    config: EngineModelConfig,
    warning: Option<AdminWarning>,
}

struct ParsedLoadModelRequest {
    model_reference: String,
    model_dir: PathBuf,
    max_cache_cap_override: Option<usize>,
    sampling_defaults: SamplingDefaults,
    set_default: bool,
    reload_when_idle: bool,
}

impl ParsedLoadModelRequest {
    fn new(request: LoadModelRequest) -> std::result::Result<Self, AdminError> {
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
        Ok(Self {
            model_reference,
            model_dir,
            max_cache_cap_override,
            sampling_defaults: request.sampling_defaults,
            set_default: request.set_default.unwrap_or(false),
            reload_when_idle: request.reload_when_idle.unwrap_or(false),
        })
    }
}

fn engine_runtime_config(args: &ServeArgs) -> Result<EnginePoolRuntimeConfig> {
    Ok(EnginePoolRuntimeConfig {
        host: args.host.clone(),
        port: args.port,
        kv_cache_turboquant_bits: args.kv_quant.turboquant_bits(),
        scheduler_autotune_report: args.scheduler_autotune_report,
        paged_prefix_cache: resolve_engine_paged_prefix_cache_settings(args)?,
        prefix_lru_cache_max_bytes: args.prefix_lru_cache_max_bytes,
        model_ttl: resolve_model_ttl(args)?,
        active_kv_offload: resolve_active_kv_offload_config(args)?,
    })
}

fn build_engine_model_config(
    args: &ServeArgs,
    model_id: String,
    model_dir: &Path,
    max_cache_cap_override: Option<usize>,
    sampling_defaults_override: SamplingDefaults,
) -> Result<EngineModelLoad> {
    let resolved = apply_load_request_scheduler_overrides(
        resolve_scheduler_for_model(args, model_dir)?,
        max_cache_cap_override,
    );
    let sampling_defaults = read_generation_sampling_defaults(model_dir)?
        .merge_with_override(sampling_defaults_override);
    let warning = match resolved.profile_source {
        None if args.scheduler_profile.is_none() => Some(AdminWarning::new(
            DEFAULT_PROFILE_WARNING_CODE,
            DEFAULT_PROFILE_WARNING,
        )),
        Some(SchedulerProfileSource::Explicit | SchedulerProfileSource::Store) | None => None,
    };
    let model_type = read_model_type(model_dir)?;
    let architecture = ModelArchitecture::from_model_type(&model_type)?;
    if architecture == ModelArchitecture::DiffusionGemma {
        anyhow::bail!(
            "DiffusionGemma is not supported by app model manager hot-load mode; \
             use `ironmlx serve --model <path>` for the dedicated DiffusionGemma server lane"
        );
    }
    Ok(EngineModelLoad {
        config: EngineModelConfig {
            id: model_id,
            path: model_dir.to_path_buf(),
            load_policy: EngineLoadPolicy::Lazy,
            default: false,
            scheduler_runtime_profile: resolved.scheduler_runtime_profile,
            mtp: None,
            sampling_defaults,
        },
        warning,
    })
}

fn read_generation_sampling_defaults(model_dir: &Path) -> Result<SamplingDefaults> {
    let path = model_dir.join("generation_config.json");
    if !path.is_file() {
        return Ok(SamplingDefaults::default());
    }
    let data = std::fs::read(&path)
        .with_context(|| format!("reading generation config {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_slice(&data)
        .with_context(|| format!("parsing generation config {}", path.display()))?;
    Ok(SamplingDefaults {
        temperature: json_number_as_f32(json.get("temperature")),
        top_p: json_number_as_f32(json.get("top_p")),
        top_k: json_number_as_i32(json.get("top_k")),
        repetition_penalty: json_number_as_f32(json.get("repetition_penalty")),
    })
}

fn json_number_as_f32(value: Option<&serde_json::Value>) -> Option<f32> {
    match value {
        Some(serde_json::Value::Number(number)) => number.as_f64().map(|value| value as f32),
        Some(serde_json::Value::String(value)) => value.trim().parse::<f32>().ok(),
        _ => None,
    }
}

fn json_number_as_i32(value: Option<&serde_json::Value>) -> Option<i32> {
    match value {
        Some(serde_json::Value::Number(number)) => number
            .as_i64()
            .and_then(|value| i32::try_from(value).ok())
            .or_else(|| number.as_f64().map(|value| value as i32)),
        Some(serde_json::Value::String(value)) => value.trim().parse::<i32>().ok(),
        _ => None,
    }
}

fn apply_load_request_scheduler_overrides(
    mut resolved: ResolvedSchedulerRuntime,
    max_cache_cap_override: Option<usize>,
) -> ResolvedSchedulerRuntime {
    if let Some(max_cache_cap) = max_cache_cap_override {
        resolved.scheduler_config.max_cache_cap = max_cache_cap;
        resolved.scheduler_runtime_profile.config.max_cache_cap = max_cache_cap;
        for rule in &mut resolved.scheduler_runtime_profile.rules {
            rule.config.max_cache_cap = max_cache_cap;
        }
    }
    resolved
}

pub async fn serve_app_daemon(args: ServeArgs) -> Result<()> {
    let host = args.host.clone();
    let port = args.port;
    let manager = ModelManager::new(args)?;
    manager.start_model_ttl_sweeper();
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(app_healthz_handler))
        .route("/v1/chat/completions", post(app_openai_handler))
        .route("/v1/messages", post(app_anthropic_handler))
        .route("/admin/api/models/loaded", get(list_loaded_handler))
        .route("/admin/api/models/register", post(register_model_handler))
        .route("/admin/api/models/load", post(load_model_handler))
        .route("/admin/api/models/unload", post(unload_model_handler))
        .route("/admin/api/models/default", post(set_default_model_handler))
        .with_state(manager);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("parsing socket addr {host}:{port}"))?;
    tracing::info!("ironmlx app daemon listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn app_openai_handler(
    State(manager): State<ModelManager>,
    Json(req): Json<openai::ChatRequest>,
) -> Response {
    manager.openai(req).await
}

async fn app_anthropic_handler(
    State(manager): State<ModelManager>,
    Json(req): Json<anthropic::MessagesRequest>,
) -> Response {
    manager.anthropic(req).await
}

async fn app_healthz_handler(State(manager): State<ModelManager>) -> Json<HealthSnapshot> {
    Json(manager.health_snapshot().await)
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

async fn unload_model_handler(
    State(manager): State<ModelManager>,
    Json(request): Json<UnloadModelRequest>,
) -> Json<AdminModelResponse> {
    Json(manager.unload_model(request).await)
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

fn resolve_error_response(error: anyhow::Error) -> Response {
    if let Some(registry) = error.downcast_ref::<EngineRegistryError>() {
        return match registry {
            EngineRegistryError::UnknownModel { id } => {
                (StatusCode::NOT_FOUND, format!("model is not registered: {id}")).into_response()
            }
            EngineRegistryError::AmbiguousDefault => (
                StatusCode::SERVICE_UNAVAILABLE,
                "No default model is loaded. Load a model or specify an already loaded model in the request.",
            )
                .into_response(),
            _ => (StatusCode::BAD_REQUEST, format!("{error:#}")).into_response(),
        };
    }
    (StatusCode::SERVICE_UNAVAILABLE, format!("{error:#}")).into_response()
}

#[derive(Debug, Deserialize)]
struct LoadModelRequest {
    model: Option<String>,
    model_dir: Option<String>,
    repo_id: Option<String>,
    set_default: Option<bool>,
    max_cache_cap: Option<usize>,
    reload_when_idle: Option<bool>,
    #[serde(flatten)]
    sampling_defaults: SamplingDefaults,
}

#[derive(Debug, Deserialize)]
struct UnloadModelRequest {
    model: Option<String>,
    model_dir: Option<String>,
    repo_id: Option<String>,
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

#[derive(Debug, Clone)]
struct AdminWarning {
    code: &'static str,
    message: String,
}

impl AdminWarning {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct LoadedModelInfo {
    id: String,
    model: String,
    path: String,
    architecture: String,
    #[serde(rename = "default")]
    is_default: bool,
    max_position_embeddings: i32,
}

impl From<EngineLoadedModelInfo> for LoadedModelInfo {
    fn from(info: EngineLoadedModelInfo) -> Self {
        Self {
            id: info.id.clone(),
            model: info.id,
            path: info.path,
            architecture: info.architecture,
            is_default: info.is_default,
            max_position_embeddings: info.max_position_embeddings,
        }
    }
}

struct AdminError {
    status: StatusCode,
    message: String,
    code: Option<&'static str>,
}

impl AdminError {
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
        let message = format!("{error:#}");
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

impl IntoResponse for AdminError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(AdminModelResponse::from_error(self.message, self.code)),
        )
            .into_response()
    }
}

fn ensure_gpu_memory_headroom() -> std::result::Result<(), AdminError> {
    let memory = mlx::memory::snapshot();
    if let Some(max_recommended) = memory.max_recommended_bytes {
        if max_recommended > 0 && memory.active_bytes >= max_recommended {
            return Err(AdminError::service_unavailable_with_code(
                GPU_MEMORY_INSUFFICIENT_MESSAGE,
                Some(GPU_MEMORY_INSUFFICIENT_CODE),
            ));
        }
    }
    Ok(())
}

fn likely_gpu_memory_error(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("out of memory")
        || lower.contains("oom")
        || lower.contains("failed to allocate")
        || lower.contains("memory allocation")
        || lower.contains("memory budget")
}

fn likely_engine_pool_capacity_error(message: &str) -> bool {
    message
        .to_ascii_lowercase()
        .contains("engine pool capacity reached")
}

fn aggregate_health(start_time: Instant, snapshots: Vec<HealthSnapshot>) -> HealthSnapshot {
    let mlx_memory = mlx::memory::snapshot();
    let total_ram_bytes = crate::core::memory_budget::system_total_ram_bytes();
    let free_ram_bytes = system_free_ram_bytes();
    let mut names = Vec::new();
    let mut max_position_embeddings = 0;
    let mut b_max = 0;
    let mut b_active = 0;
    let mut b_queued = 0;
    let mut queue_max = 0;
    let mut admission_queue_full_count = 0;
    let mut memory_budget_exceeded_count = 0;
    let mut kv_cache_active_bytes = 0;
    let mut kv_cache_soft_limit_bytes = 0;
    let mut active_kv_snapshots = Vec::new();

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
        admission_queue_full_count += snapshot.scheduler.admission_queue_full_count;
        memory_budget_exceeded_count += snapshot.scheduler.memory_budget_exceeded_count;
        kv_cache_active_bytes += snapshot.memory.kv_cache_active_bytes;
        kv_cache_soft_limit_bytes += snapshot.memory.kv_cache_soft_limit_bytes;
        active_kv_snapshots.push(snapshot.active_kv_offload);
    }

    let active_kv_offload =
        crate::core::cache::ActiveKvOffloadHealth::aggregate(active_kv_snapshots);

    let mut status = match classify_status(
        b_queued,
        queue_max,
        free_ram_bytes,
        kv_cache_active_bytes,
        kv_cache_soft_limit_bytes,
    ) {
        HealthStatus::Healthy => HealthStatus::Healthy,
        HealthStatus::Degraded | HealthStatus::Down => HealthStatus::Degraded,
    };
    if active_kv_offload.degraded {
        status = HealthStatus::Degraded;
    }

    HealthSnapshot {
        status,
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
            admission_queue_full_count,
            memory_budget_exceeded_count,
        },
        memory: MemoryInfo {
            total_ram_bytes,
            free_ram_bytes,
            kv_cache_active_bytes,
            kv_cache_soft_limit_bytes,
            mlx_total_bytes: mlx_memory.total_bytes,
            mlx_max_recommended_bytes: mlx_memory.max_recommended_bytes,
            mlx_active_bytes: mlx_memory.active_bytes,
            mlx_cache_bytes: mlx_memory.cache_bytes,
            mlx_peak_bytes: mlx_memory.peak_bytes,
            mlx_memory_limit_bytes: mlx_memory.memory_limit_bytes,
        },
        mtp: MtpHealthInfo {
            enabled: false,
            draft_tokens: None,
            prefill_count: 0,
            step_count: 0,
            fallback_prefill_count: 0,
            drafted_tokens: 0,
            accepted_draft_tokens: 0,
        },
        active_kv_offload,
        device_name: mlx_memory.device_name,
        version: env!("CARGO_PKG_VERSION"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_model_request_accepts_per_model_max_cache_cap() {
        let request: LoadModelRequest = serde_json::from_value(serde_json::json!({
            "model": "mlx-community/LongContext-4bit",
            "model_dir": "/models/long",
            "set_default": true,
            "max_cache_cap": 65536
        }))
        .expect("load request");

        assert_eq!(request.max_cache_cap, Some(65536));
    }

    #[test]
    fn load_model_request_accepts_sampling_defaults_and_idle_reload() {
        let request: LoadModelRequest = serde_json::from_value(serde_json::json!({
            "model": "mlx-community/LongContext-4bit",
            "model_dir": "/models/long",
            "reload_when_idle": true,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.1
        }))
        .expect("load request");

        assert_eq!(request.reload_when_idle, Some(true));
        assert_eq!(request.sampling_defaults.temperature, Some(0.7));
        assert_eq!(request.sampling_defaults.top_p, Some(0.8));
        assert_eq!(request.sampling_defaults.top_k, Some(40));
        assert_eq!(request.sampling_defaults.repetition_penalty, Some(1.1));
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
            scheduler_config: crate::cli::serve::SchedulerServeConfig {
                max_cache_cap: 32768,
                ..Default::default()
            },
            scheduler_runtime_profile: crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile {
                schema_version: crate::core::scheduler_autotune::SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
                model_name: "test-model".to_string(),
                hardware_label: "test-hardware".to_string(),
                config: crate::core::scheduler_autotune::SchedulerAutotuneProfileConfig {
                    b_max: 1,
                    prefill_chunk_size: 1024,
                    admission_deadline_ms: 5,
                    admission_queue_max: 32,
                    max_cache_cap: 32768,
                    decode_cadence_mid_chunk_cap: 256,
                },
                rules: vec![crate::core::scheduler_autotune::SchedulerAutotuneRuntimeRule {
                    when: crate::core::scheduler_autotune::SchedulerAutotuneRuntimeRuleCondition {
                        prompt_len_gte: 8192,
                        max_new_tokens_gte: 128,
                        effective_concurrency_gte: 1,
                    },
                    config: crate::core::scheduler_autotune::SchedulerAutotuneProfileConfig {
                        b_max: 1,
                        prefill_chunk_size: 2048,
                        admission_deadline_ms: 5,
                        admission_queue_max: 32,
                        max_cache_cap: 32768,
                        decode_cadence_mid_chunk_cap: 512,
                    },
                }],
                metadata: crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata::synthetic(0),
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
