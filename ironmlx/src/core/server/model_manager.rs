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
    read_model_type, resolve_paged_prefix_cache_config, resolve_prefix_lru_cache_config,
    resolve_scheduler_for_model, ResolvedSchedulerRuntime, SchedulerProfileSource, ServeArgs,
};
use crate::core::{Loader, Tokenizer};
use crate::models::ModelArchitecture;
use crate::Result;

use super::health::{
    classify_status, system_free_ram_bytes, HealthSnapshot, HealthStatus, MemoryInfo, ModelInfo,
    MtpHealthInfo, SchedulerInfo,
};
use super::{
    anthropic, build_plain_app_state, openai, AppState, SamplingDefaults, VisionInputConfig,
};

const GPU_MEMORY_INSUFFICIENT_MESSAGE: &str =
    "当前可用 GPU 内存不足，无法安全加载该模型。请先卸载暂不使用的已加载模型，释放显存后再重试。";
const DEFAULT_PROFILE_WARNING: &str =
    "未找到该模型匹配的 scheduler profile，已使用默认调度配置运行。后续可通过 scheduler-autotune 为该模型生成专用 profile。";

#[derive(Clone)]
pub struct ModelManager {
    registry: Arc<RwLock<ModelRegistry<LoadedModelRuntime>>>,
    pending_reloads: Arc<RwLock<HashMap<String, PendingModelReload>>>,
    serve_args: ServeArgs,
    start_time: Instant,
}

impl ModelManager {
    pub fn new(serve_args: ServeArgs) -> Self {
        Self {
            registry: Arc::new(RwLock::new(ModelRegistry::default())),
            pending_reloads: Arc::new(RwLock::new(HashMap::new())),
            serve_args,
            start_time: Instant::now(),
        }
    }

    async fn resolve_runtime(
        &self,
        requested_model: Option<&str>,
    ) -> std::result::Result<Arc<LoadedModelRuntime>, ModelResolveError> {
        self.registry.read().await.resolve(requested_model)
    }

    async fn load_model(
        &self,
        request: LoadModelRequest,
    ) -> std::result::Result<AdminModelResponse, AdminError> {
        let model_reference = request
            .model
            .as_deref()
            .or(request.model_dir.as_deref())
            .or(request.repo_id.as_deref())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| AdminError::bad_request("model is required"))?
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
            return Err(AdminError::bad_request(format!(
                "model directory does not exist: {}",
                model_dir.display()
            )));
        }

        let max_cache_cap_override = match request.max_cache_cap {
            Some(0) => return Err(AdminError::bad_request("max_cache_cap must be >= 1")),
            value => value,
        };
        let set_default = request.set_default.unwrap_or(true);
        let reload = PendingModelReload {
            model_reference: model_reference.clone(),
            model_dir: model_dir.clone(),
            max_cache_cap_override,
            sampling_defaults_override: request.sampling_defaults,
            set_default,
        };
        let reload_when_idle = request.reload_when_idle.unwrap_or(false);

        let already_loaded = {
            let registry = self.registry.read().await;
            let already_loaded = registry.contains(&model_reference);
            if already_loaded && !reload_when_idle {
                let loaded_models = registry.list();
                return Ok(AdminModelResponse::ok(
                    "already_loaded",
                    Some(model_reference),
                    loaded_models,
                    None,
                ));
            }
            if already_loaded
                && registry
                    .pending_requests(&model_reference)
                    .is_some_and(|requests| requests > 0)
            {
                let loaded_models = registry.list();
                drop(registry);
                self.schedule_reload_when_idle(reload).await;
                return Ok(AdminModelResponse::ok(
                    "reload_deferred",
                    Some(model_reference),
                    loaded_models,
                    Some("模型正在处理请求，新参数将在该模型空闲后自动重新加载。".to_string()),
                ));
            }
            already_loaded
        };

        if already_loaded && reload_when_idle {
            return self.reload_model_now(reload).await;
        }

        ensure_gpu_memory_headroom()?;
        let load = load_runtime(
            &self.serve_args,
            model_reference.clone(),
            &model_dir,
            max_cache_cap_override,
            request.sampling_defaults,
        )
        .await
        .map_err(AdminError::from_load_error)?;
        let mut registry = self.registry.write().await;
        registry.insert(model_reference.clone(), load.runtime, set_default);
        let loaded_models = registry.list();
        Ok(AdminModelResponse::ok(
            "loaded",
            Some(model_reference),
            loaded_models,
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
        let mut registry = self.registry.write().await;
        let removed = model.as_deref().and_then(|model| registry.remove(model));
        let status = if removed.is_some() {
            "unloaded"
        } else {
            "not_loaded"
        };
        AdminModelResponse::ok(status, model, registry.list(), None)
    }

    async fn set_default_model(
        &self,
        request: SetDefaultModelRequest,
    ) -> std::result::Result<AdminModelResponse, AdminError> {
        let model = request.model.trim();
        if model.is_empty() {
            return Err(AdminError::bad_request("model is required"));
        }
        let mut registry = self.registry.write().await;
        registry
            .set_default(model)
            .map_err(|_| AdminError::not_found(format!("model is not loaded: {model}")))?;
        Ok(AdminModelResponse::ok(
            "default_set",
            Some(model.to_string()),
            registry.list(),
            None,
        ))
    }

    async fn list_loaded(&self) -> Vec<LoadedModelInfo> {
        self.registry.read().await.list()
    }

    async fn health_snapshot(&self) -> HealthSnapshot {
        let snapshots = self
            .registry
            .read()
            .await
            .models
            .values()
            .map(|runtime| runtime.health_snapshot())
            .collect::<Vec<_>>();
        aggregate_health(self.start_time, snapshots)
    }

    async fn reload_model_now(
        &self,
        reload: PendingModelReload,
    ) -> std::result::Result<AdminModelResponse, AdminError> {
        let removed = {
            let mut registry = self.registry.write().await;
            if registry
                .pending_requests(&reload.model_reference)
                .is_some_and(|requests| requests > 0)
            {
                drop(registry);
                self.schedule_reload_when_idle(reload.clone()).await;
                let loaded_models = self.registry.read().await.list();
                return Ok(AdminModelResponse::ok(
                    "reload_deferred",
                    Some(reload.model_reference),
                    loaded_models,
                    Some("模型正在处理请求，新参数将在该模型空闲后自动重新加载。".to_string()),
                ));
            }
            registry.remove(&reload.model_reference)
        };
        if removed.is_none() {
            return Err(AdminError::not_found(format!(
                "model is not loaded: {}",
                reload.model_reference
            )));
        }

        let removed = removed.expect("removed runtime checked above");
        let load = match load_runtime(
            &self.serve_args,
            reload.model_reference.clone(),
            &reload.model_dir,
            reload.max_cache_cap_override,
            reload.sampling_defaults_override,
        )
        .await
        {
            Ok(load) => load,
            Err(error) => {
                self.registry.write().await.insert_existing(
                    reload.model_reference.clone(),
                    removed,
                    reload.set_default,
                );
                return Err(AdminError::from_load_error(error));
            }
        };
        let mut registry = self.registry.write().await;
        registry.insert(
            reload.model_reference.clone(),
            load.runtime,
            reload.set_default,
        );
        let loaded_models = registry.list();
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
        let registry = self.registry.clone();
        let pending_reloads = self.pending_reloads.clone();
        let serve_args = self.serve_args.clone();

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(500)).await;
                let still_pending = pending_reloads.read().await.contains_key(&model_reference);
                if !still_pending {
                    return;
                }
                let busy = registry
                    .read()
                    .await
                    .pending_requests(&model_reference)
                    .is_some_and(|requests| requests > 0);
                if busy {
                    continue;
                }
                let reload = pending_reloads.write().await.remove(&model_reference);
                let Some(reload) = reload else {
                    return;
                };
                let removed = {
                    let mut registry = registry.write().await;
                    if registry
                        .pending_requests(&model_reference)
                        .is_some_and(|requests| requests > 0)
                    {
                        pending_reloads
                            .write()
                            .await
                            .insert(model_reference.clone(), reload);
                        continue;
                    }
                    registry.remove(&model_reference)
                };
                if removed.is_none() {
                    return;
                }
                let removed = removed.expect("removed runtime checked above");
                match load_runtime(
                    &serve_args,
                    reload.model_reference.clone(),
                    &reload.model_dir,
                    reload.max_cache_cap_override,
                    reload.sampling_defaults_override,
                )
                .await
                {
                    Ok(load) => {
                        registry.write().await.insert(
                            reload.model_reference,
                            load.runtime,
                            reload.set_default,
                        );
                    }
                    Err(error) => {
                        registry.write().await.insert_existing(
                            reload.model_reference.clone(),
                            removed,
                            reload.set_default,
                        );
                        tracing::error!(
                            "failed to reload model {} after idle: {error:#}",
                            reload.model_reference
                        );
                    }
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

struct ModelRegistry<T> {
    models: HashMap<String, Arc<T>>,
    default_model: Option<String>,
}

impl<T> Default for ModelRegistry<T> {
    fn default() -> Self {
        Self {
            models: HashMap::new(),
            default_model: None,
        }
    }
}

impl<T> ModelRegistry<T>
where
    T: RuntimeInfo,
{
    fn insert(&mut self, id: String, runtime: T, set_default: bool) -> Arc<T> {
        let runtime = Arc::new(runtime);
        self.models.insert(id.clone(), runtime.clone());
        if set_default || self.default_model.is_none() {
            self.default_model = Some(id);
        }
        runtime
    }

    fn insert_existing(&mut self, id: String, runtime: Arc<T>, set_default: bool) {
        self.models.insert(id.clone(), runtime);
        if set_default || self.default_model.is_none() {
            self.default_model = Some(id);
        }
    }

    fn contains(&self, id: &str) -> bool {
        self.models.contains_key(id)
    }

    fn remove(&mut self, id: &str) -> Option<Arc<T>> {
        let removed = self.models.remove(id);
        if self.default_model.as_deref() == Some(id) {
            self.default_model = None;
        }
        removed
    }

    fn set_default(&mut self, id: &str) -> std::result::Result<(), ()> {
        if !self.models.contains_key(id) {
            return Err(());
        }
        self.default_model = Some(id.to_string());
        Ok(())
    }

    fn resolve(
        &self,
        requested_model: Option<&str>,
    ) -> std::result::Result<Arc<T>, ModelResolveError> {
        if let Some(requested) = requested_model
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return self
                .models
                .get(requested)
                .cloned()
                .ok_or_else(|| ModelResolveError::NotLoaded(requested.to_string()));
        }
        let default_model = self
            .default_model
            .as_deref()
            .ok_or(ModelResolveError::NoDefault)?;
        self.models
            .get(default_model)
            .cloned()
            .ok_or_else(|| ModelResolveError::NotLoaded(default_model.to_string()))
    }

    fn list(&self) -> Vec<LoadedModelInfo> {
        let mut models = self
            .models
            .iter()
            .map(|(id, runtime)| LoadedModelInfo {
                id: id.clone(),
                model: id.clone(),
                path: runtime.model_path().to_string(),
                architecture: runtime.architecture().to_string(),
                is_default: self.default_model.as_deref() == Some(id.as_str()),
                max_position_embeddings: runtime.max_position_embeddings(),
            })
            .collect::<Vec<_>>();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    fn pending_requests(&self, id: &str) -> Option<usize> {
        self.models
            .get(id)
            .map(|runtime| runtime.pending_requests())
    }
}

trait RuntimeInfo {
    fn model_path(&self) -> &str;
    fn architecture(&self) -> &'static str;
    fn max_position_embeddings(&self) -> i32;
    fn pending_requests(&self) -> usize {
        0
    }
}

#[derive(Debug)]
enum ModelResolveError {
    NoDefault,
    NotLoaded(String),
}

#[allow(clippy::large_enum_variant)]
enum LoadedModelRuntime {
    Qwen35Dense {
        path: String,
        state: AppState<crate::models::Qwen35Model>,
    },
    Qwen35Moe {
        path: String,
        state: AppState<crate::models::Qwen35MoeModel>,
    },
    Gemma4 {
        path: String,
        state: AppState<crate::models::Gemma4Model>,
    },
    Glm4MoeLite {
        path: String,
        state: AppState<crate::models::Glm4MoeLiteModel>,
    },
    Llama {
        path: String,
        state: AppState<crate::models::LlamaModel>,
    },
    MiniCpmV46 {
        path: String,
        state: AppState<crate::models::MiniCpmV46Model>,
    },
}

impl LoadedModelRuntime {
    async fn openai(&self, req: openai::ChatRequest) -> Response {
        match self {
            Self::Qwen35Dense { state, .. } => {
                openai::chat_completions_with_state(state.clone(), req).await
            }
            Self::Qwen35Moe { state, .. } => {
                openai::chat_completions_with_state(state.clone(), req).await
            }
            Self::Gemma4 { state, .. } => {
                openai::chat_completions_with_state(state.clone(), req).await
            }
            Self::Glm4MoeLite { state, .. } => {
                openai::chat_completions_with_state(state.clone(), req).await
            }
            Self::Llama { state, .. } => {
                openai::chat_completions_with_state(state.clone(), req).await
            }
            Self::MiniCpmV46 { state, .. } => {
                openai::chat_completions_with_state(state.clone(), req).await
            }
        }
    }

    async fn anthropic(&self, req: anthropic::MessagesRequest) -> Response {
        match self {
            Self::Qwen35Dense { state, .. } => {
                anthropic::messages_with_state(state.clone(), req).await
            }
            Self::Qwen35Moe { state, .. } => {
                anthropic::messages_with_state(state.clone(), req).await
            }
            Self::Gemma4 { state, .. } => anthropic::messages_with_state(state.clone(), req).await,
            Self::Glm4MoeLite { state, .. } => {
                anthropic::messages_with_state(state.clone(), req).await
            }
            Self::Llama { state, .. } => anthropic::messages_with_state(state.clone(), req).await,
            Self::MiniCpmV46 { state, .. } => {
                anthropic::messages_with_state(state.clone(), req).await
            }
        }
    }

    fn health_snapshot(&self) -> HealthSnapshot {
        match self {
            Self::Qwen35Dense { state, .. } => state.health_collector.snapshot(),
            Self::Qwen35Moe { state, .. } => state.health_collector.snapshot(),
            Self::Gemma4 { state, .. } => state.health_collector.snapshot(),
            Self::Glm4MoeLite { state, .. } => state.health_collector.snapshot(),
            Self::Llama { state, .. } => state.health_collector.snapshot(),
            Self::MiniCpmV46 { state, .. } => state.health_collector.snapshot(),
        }
    }

    fn pending_requests(&self) -> usize {
        match self {
            Self::Qwen35Dense { state, .. } => runtime_pending_requests(state),
            Self::Qwen35Moe { state, .. } => runtime_pending_requests(state),
            Self::Gemma4 { state, .. } => runtime_pending_requests(state),
            Self::Glm4MoeLite { state, .. } => runtime_pending_requests(state),
            Self::Llama { state, .. } => runtime_pending_requests(state),
            Self::MiniCpmV46 { state, .. } => runtime_pending_requests(state),
        }
    }
}

fn runtime_pending_requests<M>(state: &AppState<M>) -> usize
where
    M: crate::core::model::Model + crate::core::scheduler::DenseVlMethods + Send + 'static,
{
    let snapshot = state.health_collector.snapshot();
    let model_locked = usize::from(state.model.try_lock().is_err());
    snapshot
        .scheduler
        .b_active
        .saturating_add(snapshot.scheduler.b_queued)
        .saturating_add(model_locked)
}

impl RuntimeInfo for LoadedModelRuntime {
    fn model_path(&self) -> &str {
        match self {
            Self::Qwen35Dense { path, .. }
            | Self::Qwen35Moe { path, .. }
            | Self::Gemma4 { path, .. }
            | Self::Glm4MoeLite { path, .. }
            | Self::Llama { path, .. }
            | Self::MiniCpmV46 { path, .. } => path,
        }
    }

    fn architecture(&self) -> &'static str {
        match self {
            Self::Qwen35Dense { .. } => "qwen3_5",
            Self::Qwen35Moe { .. } => "qwen3_5_moe",
            Self::Gemma4 { .. } => "gemma4",
            Self::Glm4MoeLite { .. } => "glm4_moe_lite",
            Self::Llama { .. } => "llama",
            Self::MiniCpmV46 { .. } => "minicpmv4_6",
        }
    }

    fn max_position_embeddings(&self) -> i32 {
        self.health_snapshot().model.max_position_embeddings
    }

    fn pending_requests(&self) -> usize {
        self.pending_requests()
    }
}

struct RuntimeLoad {
    runtime: LoadedModelRuntime,
    warning: Option<String>,
}

async fn load_runtime(
    args: &ServeArgs,
    model_id: String,
    model_dir: &Path,
    max_cache_cap_override: Option<usize>,
    sampling_defaults_override: SamplingDefaults,
) -> Result<RuntimeLoad> {
    let resolved = apply_load_request_scheduler_overrides(
        resolve_scheduler_for_model(args, model_dir)?,
        max_cache_cap_override,
    );
    let sampling_defaults = read_generation_sampling_defaults(model_dir)?
        .merge_with_override(sampling_defaults_override);
    let warning = match resolved.profile_source {
        None if args.scheduler_profile.is_none() => Some(DEFAULT_PROFILE_WARNING.to_string()),
        Some(SchedulerProfileSource::Explicit | SchedulerProfileSource::Store) | None => None,
    };
    let model_type = read_model_type(model_dir)?;
    let architecture = ModelArchitecture::from_model_type(&model_type)?;
    let loader = Loader::open_multimodal(model_dir).context("Loader::open_multimodal")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let vision_input = match architecture {
        ModelArchitecture::Gemma4 => {
            let cfg = crate::models::gemma4::Gemma4Config::from_loader(&loader)
                .context("Gemma4Config::from_loader")?;
            cfg.vision_config
                .map(|vision_config| VisionInputConfig::Gemma4 { vision_config })
        }
        ModelArchitecture::MiniCpmV46 => Some(VisionInputConfig::MiniCpmV46 {
            spatial_merge_size: 4,
        }),
        _ => None,
    };
    let path = model_dir.to_string_lossy().into_owned();
    let config = resolved.scheduler_config;
    let profile = resolved.scheduler_runtime_profile;
    let report = args.scheduler_autotune_report;
    let kv_cache_turboquant_bits = args.kv_quant.turboquant_bits();
    let paged_prefix_cache = resolve_paged_prefix_cache_config(args, config, &model_id)?;
    let prefix_lru_cache = resolve_prefix_lru_cache_config(args, paged_prefix_cache.as_ref())?;
    let runtime = match architecture {
        ModelArchitecture::Qwen35Dense => {
            let model = crate::models::Qwen35Model::from_loader(&loader)
                .context("Qwen35Model::from_loader")?;
            let state = build_plain_app_state(
                model,
                tokenizer,
                model_id,
                config.prefill_chunk_size,
                config.b_max,
                config.admission_deadline_ms,
                config.admission_queue_max,
                config.max_cache_cap,
                config.decode_cadence_mid_chunk_cap,
                kv_cache_turboquant_bits,
                profile,
                report,
                vision_input,
                paged_prefix_cache.clone(),
                prefix_lru_cache,
            )
            .await?
            .with_sampling_defaults(sampling_defaults);
            LoadedModelRuntime::Qwen35Dense { path, state }
        }
        ModelArchitecture::Qwen35Moe => {
            let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                .context("Qwen35MoeModel::from_loader")?;
            let state = build_plain_app_state(
                model,
                tokenizer,
                model_id,
                config.prefill_chunk_size,
                config.b_max,
                config.admission_deadline_ms,
                config.admission_queue_max,
                config.max_cache_cap,
                config.decode_cadence_mid_chunk_cap,
                kv_cache_turboquant_bits,
                profile,
                report,
                vision_input,
                paged_prefix_cache.clone(),
                prefix_lru_cache,
            )
            .await?
            .with_sampling_defaults(sampling_defaults);
            LoadedModelRuntime::Qwen35Moe { path, state }
        }
        ModelArchitecture::Gemma4 => {
            let model = crate::models::Gemma4Model::from_loader(&loader)
                .context("Gemma4Model::from_loader")?;
            let state = build_plain_app_state(
                model,
                tokenizer,
                model_id,
                config.prefill_chunk_size,
                config.b_max,
                config.admission_deadline_ms,
                config.admission_queue_max,
                config.max_cache_cap,
                config.decode_cadence_mid_chunk_cap,
                kv_cache_turboquant_bits,
                profile,
                report,
                vision_input,
                paged_prefix_cache.clone(),
                prefix_lru_cache,
            )
            .await?
            .with_sampling_defaults(sampling_defaults);
            LoadedModelRuntime::Gemma4 { path, state }
        }
        ModelArchitecture::Glm4MoeLite => {
            let model = crate::models::Glm4MoeLiteModel::from_loader(&loader)
                .context("Glm4MoeLiteModel::from_loader")?;
            let state = build_plain_app_state(
                model,
                tokenizer,
                model_id,
                config.prefill_chunk_size,
                config.b_max,
                config.admission_deadline_ms,
                config.admission_queue_max,
                config.max_cache_cap,
                config.decode_cadence_mid_chunk_cap,
                kv_cache_turboquant_bits,
                profile,
                report,
                None,
                paged_prefix_cache.clone(),
                prefix_lru_cache,
            )
            .await?
            .with_sampling_defaults(sampling_defaults);
            LoadedModelRuntime::Glm4MoeLite { path, state }
        }
        ModelArchitecture::Llama => {
            let model = crate::models::LlamaModel::from_loader(&loader)
                .context("LlamaModel::from_loader")?;
            let state = build_plain_app_state(
                model,
                tokenizer,
                model_id,
                config.prefill_chunk_size,
                config.b_max,
                config.admission_deadline_ms,
                config.admission_queue_max,
                config.max_cache_cap,
                config.decode_cadence_mid_chunk_cap,
                kv_cache_turboquant_bits,
                profile,
                report,
                None,
                paged_prefix_cache.clone(),
                prefix_lru_cache,
            )
            .await?
            .with_sampling_defaults(sampling_defaults);
            LoadedModelRuntime::Llama { path, state }
        }
        ModelArchitecture::MiniCpmV46 => {
            let model = crate::models::minicpmv4_6::model_from_loader(&loader)
                .context("minicpmv4_6::model_from_loader")?;
            let state = build_plain_app_state(
                model,
                tokenizer,
                model_id,
                config.prefill_chunk_size,
                config.b_max,
                config.admission_deadline_ms,
                config.admission_queue_max,
                config.max_cache_cap,
                config.decode_cadence_mid_chunk_cap,
                kv_cache_turboquant_bits,
                profile,
                report,
                vision_input,
                paged_prefix_cache,
                prefix_lru_cache,
            )
            .await?
            .with_sampling_defaults(sampling_defaults);
            LoadedModelRuntime::MiniCpmV46 { path, state }
        }
        ModelArchitecture::DiffusionGemma => {
            anyhow::bail!(
                "DiffusionGemma is not supported by app model manager hot-load mode; \
                 use `ironmlx serve --model <path>` for the dedicated DiffusionGemma server lane"
            );
        }
    };
    Ok(RuntimeLoad { runtime, warning })
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
    let manager = ModelManager::new(args);
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(app_healthz_handler))
        .route("/v1/chat/completions", post(app_openai_handler))
        .route("/v1/messages", post(app_anthropic_handler))
        .route("/admin/api/models/loaded", get(list_loaded_handler))
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
    let runtime = match manager.resolve_runtime(req.model.as_deref()).await {
        Ok(runtime) => runtime,
        Err(error) => return resolve_error_response(error),
    };
    runtime.openai(req).await
}

async fn app_anthropic_handler(
    State(manager): State<ModelManager>,
    Json(req): Json<anthropic::MessagesRequest>,
) -> Response {
    let runtime = match manager.resolve_runtime(req.model.as_deref()).await {
        Ok(runtime) => runtime,
        Err(error) => return resolve_error_response(error),
    };
    runtime.anthropic(req).await
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

fn resolve_error_response(error: ModelResolveError) -> Response {
    match error {
        ModelResolveError::NoDefault => (
            StatusCode::SERVICE_UNAVAILABLE,
            "No default model is loaded. Load a model or specify an already loaded model in the request.",
        )
            .into_response(),
        ModelResolveError::NotLoaded(model) => (
            StatusCode::NOT_FOUND,
            format!("model is not loaded: {model}"),
        )
            .into_response(),
    }
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
    model: Option<String>,
    loaded_models: Vec<LoadedModelInfo>,
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
        warning: Option<String>,
    ) -> Self {
        Self {
            success: true,
            status,
            model,
            loaded_models,
            warning,
            error: None,
        }
    }

    fn error(message: String) -> Self {
        Self {
            success: false,
            status: "error",
            model: None,
            loaded_models: Vec::new(),
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
    #[serde(rename = "default")]
    is_default: bool,
    max_position_embeddings: i32,
}

struct AdminError {
    status: StatusCode,
    message: String,
}

impl AdminError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }

    fn from_load_error(error: anyhow::Error) -> Self {
        let message = format!("{error:#}");
        if likely_gpu_memory_error(&message) {
            return Self::service_unavailable(GPU_MEMORY_INSUFFICIENT_MESSAGE);
        }
        Self {
            status: StatusCode::BAD_REQUEST,
            message,
        }
    }
}

impl IntoResponse for AdminError {
    fn into_response(self) -> Response {
        (self.status, Json(AdminModelResponse::error(self.message))).into_response()
    }
}

fn ensure_gpu_memory_headroom() -> std::result::Result<(), AdminError> {
    let memory = mlx::memory::snapshot();
    if let Some(max_recommended) = memory.max_recommended_bytes {
        if max_recommended > 0 && memory.active_bytes >= max_recommended {
            return Err(AdminError::service_unavailable(
                GPU_MEMORY_INSUFFICIENT_MESSAGE,
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
    }

    let status = match classify_status(
        b_queued,
        queue_max,
        free_ram_bytes,
        kv_cache_active_bytes,
        kv_cache_soft_limit_bytes,
    ) {
        HealthStatus::Healthy => HealthStatus::Healthy,
        HealthStatus::Degraded | HealthStatus::Down => HealthStatus::Degraded,
    };

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
        device_name: mlx_memory.device_name,
        version: env!("CARGO_PKG_VERSION"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestRuntime {
        path: String,
    }

    impl RuntimeInfo for TestRuntime {
        fn model_path(&self) -> &str {
            &self.path
        }

        fn architecture(&self) -> &'static str {
            "test"
        }

        fn max_position_embeddings(&self) -> i32 {
            4096
        }
    }

    #[test]
    fn registry_switches_default_for_new_resolves_without_dropping_old_arc() {
        let mut registry = ModelRegistry::default();
        let old = registry.insert(
            "old".to_string(),
            TestRuntime {
                path: "/models/old".to_string(),
            },
            true,
        );
        registry.insert(
            "new".to_string(),
            TestRuntime {
                path: "/models/new".to_string(),
            },
            true,
        );

        let resolved = registry.resolve(None).expect("default runtime");

        assert_eq!(resolved.model_path(), "/models/new");
        assert_eq!(old.model_path(), "/models/old");
    }

    #[test]
    fn registry_unload_removes_future_resolves_but_existing_arc_survives() {
        let mut registry = ModelRegistry::default();
        let existing = registry.insert(
            "model".to_string(),
            TestRuntime {
                path: "/models/model".to_string(),
            },
            true,
        );

        let removed = registry.remove("model").expect("removed runtime");

        assert!(matches!(
            registry.resolve(Some("model")),
            Err(ModelResolveError::NotLoaded(_))
        ));
        assert_eq!(existing.model_path(), "/models/model");
        assert_eq!(removed.model_path(), "/models/model");
    }

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
