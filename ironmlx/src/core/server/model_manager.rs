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
    apply_gemma4_drafter_adaptive_scheduler_defaults, read_model_type,
    resolve_active_kv_offload_config, resolve_engine_paged_prefix_cache_settings,
    resolve_memory_limit_bytes, resolve_model_ttl, resolve_scheduler_for_model,
    ResolvedSchedulerRuntime, SchedulerProfileSource, ServeArgs,
};
use crate::core::sampler::Sampler;
use crate::core::speculative::{MtpDraftTokensArg, MtpSpeculativeConfig};
use crate::models::{
    Gemma4AssistantConfig, Gemma4Config, ModelArchitecture, Qwen35Config, Qwen35MoeConfig,
};
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
const MODEL_MEMORY_LIMIT_EXCEEDED_CODE: &str = "model_memory_limit_exceeded";
const MODEL_MEMORY_LIMIT_EXCEEDED_MESSAGE: &str =
    "Model memory limit reached. Unload one or more loaded models, raise Memory Limit (Model Only), or set it to Auto, then try again.";
const TOTAL_MEMORY_LIMIT_EXCEEDED_CODE: &str = "total_memory_limit_exceeded";
const TOTAL_MEMORY_LIMIT_EXCEEDED_MESSAGE: &str =
    "Total memory limit reached. Unload one or more loaded models, raise Memory Limit (Total), or set it to Auto, then try again.";
const KV_MEMORY_BUDGET_EXCEEDED_CODE: &str = "kv_memory_budget_exceeded";
const KV_MEMORY_BUDGET_EXCEEDED_MESSAGE: &str =
    "KV cache memory budget exceeded. Enable Active KV offload with paged prefix cache for long context, or lower MAX TOKENS, max_cache_cap, or b_max, then try again.";
const DEFAULT_PROFILE_WARNING_CODE: &str = "default_scheduler_profile_used";
const DEFAULT_PROFILE_WARNING: &str =
    "No matching scheduler profile was found for this model. The model is running with the default scheduler configuration. Generate a dedicated profile with scheduler-autotune for better model-specific scheduling.";
const MODEL_RELOAD_DEFERRED_WARNING_CODE: &str = "model_reload_deferred";
const MODEL_RELOAD_DEFERRED_WARNING: &str =
    "The model is processing requests. New parameters will be applied automatically after the model becomes idle.";
const MTP_MODEL_DIR_REQUIRED_CODE: &str = "mtp_model_dir_required";
const MTP_MODEL_DIR_REQUIRED_MESSAGE: &str = "MTP draft tokens require an MTP model directory.";
const MTP_OK_CODE: &str = "ok";
const MTP_BASE_MODEL_NOT_FOUND_CODE: &str = "mtp_base_model_not_found";
const MTP_MODEL_NOT_FOUND_CODE: &str = "mtp_model_not_found";
const MTP_UNSUPPORTED_ARCHITECTURE_CODE: &str = "mtp_unsupported_architecture";
const MTP_INVALID_MODEL_TYPE_CODE: &str = "mtp_invalid_model_type";
const MTP_INVALID_CONFIG_CODE: &str = "mtp_invalid_config";
const MTP_INVALID_DRAFT_TOKENS_CODE: &str = "mtp_invalid_draft_tokens";
const MTP_INCOMPATIBLE_CODE: &str = "mtp_incompatible";
const MTP_INCOMPATIBLE_MESSAGE: &str =
    "MTP weights are not compatible with this model. Load the model without MTP or choose matching MTP weights.";

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
            mtp: parsed.mtp.clone(),
            pinned: parsed.pinned,
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
            parsed.mtp,
            parsed.pinned,
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
            parsed.mtp,
            parsed.pinned,
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
            reload.mtp.clone(),
            reload.pinned,
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
                    reload.mtp.clone(),
                    reload.pinned,
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
    mtp: Option<super::engine::EngineMtpSettings>,
    pinned: bool,
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
    mtp: Option<super::engine::EngineMtpSettings>,
    pinned: bool,
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
        let mtp = match (
            request
                .mtp_model_dir
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty()),
            request.mtp_draft_tokens,
        ) {
            (Some(model_dir), draft_tokens) => Some(super::engine::EngineMtpSettings {
                model_dir: PathBuf::from(model_dir),
                draft_tokens,
            }),
            (None, Some(_)) => {
                return Err(AdminError::bad_request_with_code(
                    MTP_MODEL_DIR_REQUIRED_MESSAGE,
                    Some(MTP_MODEL_DIR_REQUIRED_CODE),
                ));
            }
            (None, None) => None,
        };
        Ok(Self {
            model_reference,
            model_dir,
            max_cache_cap_override,
            sampling_defaults: request.sampling_defaults,
            mtp,
            pinned: request.pinned.unwrap_or(false),
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
        memory_limits: super::engine::EnginePoolMemoryLimits {
            total_memory_limit_bytes: resolve_memory_limit_bytes(
                args.memory_limit_total_gb,
                "--memory-limit-total-gb",
            )?,
            model_memory_limit_bytes: resolve_memory_limit_bytes(
                args.memory_limit_model_gb,
                "--memory-limit-model-gb",
            )?,
        },
        active_kv_offload: resolve_active_kv_offload_config(args)?,
    })
}

fn build_engine_model_config(
    args: &ServeArgs,
    model_id: String,
    model_dir: &Path,
    max_cache_cap_override: Option<usize>,
    sampling_defaults_override: SamplingDefaults,
    mtp: Option<super::engine::EngineMtpSettings>,
    pinned: bool,
) -> Result<EngineModelLoad> {
    let mut resolved = apply_load_request_scheduler_overrides(
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
    if let Some(settings) = mtp.as_ref() {
        let validation = validate_mtp_pair(model_dir, &settings.model_dir, settings.draft_tokens)?;
        if !validation.compatible {
            anyhow::bail!(
                "MTP validation failed: {}: {}",
                validation.reason_code,
                validation.message
            );
        }
    }
    if apply_gemma4_drafter_adaptive_scheduler_defaults(
        args,
        architecture,
        mtp.is_some(),
        &mut resolved,
    ) {
        tracing::info!(
            "ironmlx app: Gemma4 drafter adaptive scheduler default applied model_id={} b_max={}",
            model_id,
            resolved.scheduler_config.b_max
        );
    }
    Ok(EngineModelLoad {
        config: EngineModelConfig {
            id: model_id,
            path: model_dir.to_path_buf(),
            load_policy: EngineLoadPolicy::Lazy,
            default: false,
            pinned,
            scheduler_runtime_profile: resolved.scheduler_runtime_profile,
            mtp,
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

pub(crate) fn validate_mtp_pair(
    model_dir: &Path,
    mtp_model_dir: &Path,
    mtp_draft_tokens: Option<usize>,
) -> Result<MtpValidationResponse> {
    if !model_dir.is_dir() {
        return Ok(MtpValidationResponse::not_compatible(
            MTP_BASE_MODEL_NOT_FOUND_CODE,
            format!(
                "Base model directory does not exist: {}",
                model_dir.display()
            ),
            None,
        ));
    }
    if !mtp_model_dir.is_dir() {
        return Ok(MtpValidationResponse::not_compatible(
            MTP_MODEL_NOT_FOUND_CODE,
            format!(
                "MTP model directory does not exist: {}",
                mtp_model_dir.display()
            ),
            None,
        ));
    }

    let base_raw = read_config_json(model_dir)?;
    let model_type = base_raw
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let architecture = match ModelArchitecture::from_model_type(model_type) {
        Ok(architecture) => architecture,
        Err(error) => {
            return Ok(MtpValidationResponse::not_compatible(
                MTP_UNSUPPORTED_ARCHITECTURE_CODE,
                format!("{error:#}"),
                None,
            ));
        }
    };
    match architecture {
        ModelArchitecture::Qwen35Dense
        | ModelArchitecture::Qwen35Moe
        | ModelArchitecture::Gemma4 => {}
        _ => {
            return Ok(MtpValidationResponse::not_compatible(
                MTP_UNSUPPORTED_ARCHITECTURE_CODE,
                "MTP currently supports Qwen/Gemma4 models only.",
                None,
            ));
        }
    }

    let mtp_raw = read_config_json(mtp_model_dir)?;
    let mtp_model_type = mtp_raw
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let valid_mtp_type = match architecture {
        ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe => {
            mtp_model_type == "qwen3_5_mtp"
        }
        ModelArchitecture::Gemma4 => {
            matches!(
                mtp_model_type,
                "gemma4_assistant" | "gemma4_unified_assistant"
            )
        }
        _ => unreachable!("MTP architecture was filtered above"),
    };
    if !valid_mtp_type {
        let expected = match architecture {
            ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe => "qwen3_5_mtp",
            ModelArchitecture::Gemma4 => "gemma4_assistant or gemma4_unified_assistant",
            _ => unreachable!("MTP architecture was filtered above"),
        };
        return Ok(MtpValidationResponse::not_compatible(
            MTP_INVALID_MODEL_TYPE_CODE,
            format!("Expected MTP model_type={expected}, got {mtp_model_type}"),
            None,
        ));
    }

    match architecture {
        ModelArchitecture::Qwen35Dense => {
            if let Some(response) = validate_qwen35_dense_mtp_config(&base_raw, &mtp_raw)? {
                return Ok(response);
            }
        }
        ModelArchitecture::Qwen35Moe => {
            if let Some(response) = validate_qwen35_moe_mtp_config(&base_raw, &mtp_raw)? {
                return Ok(response);
            }
        }
        ModelArchitecture::Gemma4 => {
            if let Some(response) = validate_gemma4_assistant_mtp_config(&base_raw, &mtp_raw)? {
                return Ok(response);
            }
        }
        _ => unreachable!("MTP architecture was filtered above"),
    }

    let draft_tokens = crate::core::speculative::resolve_mtp_draft_tokens(
        &base_raw,
        mtp_draft_tokens
            .map(MtpDraftTokensArg::Explicit)
            .unwrap_or(MtpDraftTokensArg::Omitted),
    );
    if let Err(error) = MtpSpeculativeConfig::new(draft_tokens, Sampler::greedy()) {
        return Ok(MtpValidationResponse::not_compatible(
            MTP_INVALID_DRAFT_TOKENS_CODE,
            format!("{error:#}"),
            Some(draft_tokens),
        ));
    }
    Ok(MtpValidationResponse::compatible(draft_tokens))
}

fn validate_qwen35_dense_mtp_config(
    base_raw: &serde_json::Value,
    mtp_raw: &serde_json::Value,
) -> Result<Option<MtpValidationResponse>> {
    let base_cfg = qwen35_config_from_raw(base_raw)?;
    let mtp_cfg = qwen35_config_from_raw(mtp_raw)?;
    if let Err(error) = mtp_cfg.mtp_config() {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INVALID_CONFIG_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    if let Err(error) = base_cfg.ensure_mtp_compatible(&mtp_cfg) {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    Ok(None)
}

fn validate_qwen35_moe_mtp_config(
    base_raw: &serde_json::Value,
    mtp_raw: &serde_json::Value,
) -> Result<Option<MtpValidationResponse>> {
    let base_cfg = qwen35_moe_config_from_raw(base_raw)?;
    let mtp_cfg = qwen35_moe_config_from_raw(mtp_raw)?;
    if let Err(error) = mtp_cfg.mtp_config() {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INVALID_CONFIG_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    if let Err(error) = base_cfg.ensure_mtp_compatible(&mtp_cfg) {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    Ok(None)
}

fn validate_gemma4_assistant_mtp_config(
    base_raw: &serde_json::Value,
    mtp_raw: &serde_json::Value,
) -> Result<Option<MtpValidationResponse>> {
    let mut base_cfg: Gemma4Config =
        serde_json::from_value(base_raw.clone()).context("failed to deserialize Gemma4Config")?;
    if let Err(error) = base_cfg.validate_and_finalize() {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INVALID_CONFIG_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    let mut mtp_cfg: Gemma4AssistantConfig = serde_json::from_value(mtp_raw.clone())
        .context("failed to deserialize Gemma4AssistantConfig")?;
    if let Err(error) = mtp_cfg.validate_and_finalize() {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INVALID_CONFIG_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    let expected_assistant_type = match base_cfg.model_type.as_str() {
        "gemma4" => "gemma4_assistant",
        "gemma4_unified" => "gemma4_unified_assistant",
        _ => unreachable!("Gemma4Config validation filtered model type"),
    };
    if mtp_cfg.model_type != expected_assistant_type {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            format!(
                "Gemma4 base model_type={} requires assistant model_type={expected_assistant_type}, got {}",
                base_cfg.model_type, mtp_cfg.model_type
            ),
            None,
        )));
    }
    if mtp_cfg.backbone_hidden_size != base_cfg.text_config.hidden_size {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            format!(
                "Gemma4 assistant backbone_hidden_size={} must match base hidden_size={}",
                mtp_cfg.backbone_hidden_size, base_cfg.text_config.hidden_size
            ),
            None,
        )));
    }
    if mtp_cfg.text_config.vocab_size != base_cfg.text_config.vocab_size {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            format!(
                "Gemma4 assistant vocab_size={} must match base vocab_size={}",
                mtp_cfg.text_config.vocab_size, base_cfg.text_config.vocab_size
            ),
            None,
        )));
    }
    if !mtp_cfg.text_config.all_layers_share_external_kv() {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            "Gemma4 assistant must share K/V for every drafter layer".to_owned(),
            None,
        )));
    }

    let mut base_has_sliding = false;
    let mut base_has_full = false;
    for idx in 0..base_cfg.text_config.num_hidden_layers as usize {
        match base_cfg.text_config.layer_kind(idx) {
            crate::models::gemma4::Gemma4LayerKind::Sliding => base_has_sliding = true,
            crate::models::gemma4::Gemma4LayerKind::Full => base_has_full = true,
        }
    }
    let mut assistant_has_sliding = false;
    let mut assistant_has_full = false;
    for idx in 0..mtp_cfg.text_config.num_hidden_layers as usize {
        match mtp_cfg.text_config.layer_kind(idx) {
            crate::models::gemma4::Gemma4LayerKind::Sliding => assistant_has_sliding = true,
            crate::models::gemma4::Gemma4LayerKind::Full => assistant_has_full = true,
        }
    }
    if (base_has_sliding && !assistant_has_sliding) || (base_has_full && !assistant_has_full) {
        return Ok(Some(MtpValidationResponse::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            "Gemma4 assistant layer_types must cover every layer type used by the base model"
                .to_owned(),
            None,
        )));
    }
    Ok(None)
}

fn read_config_json(model_dir: &Path) -> Result<serde_json::Value> {
    let path = model_dir.join("config.json");
    let data = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&data).with_context(|| format!("parsing {}", path.display()))
}

fn qwen35_config_from_raw(raw: &serde_json::Value) -> Result<Qwen35Config> {
    let text_config = raw
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("config.json missing text_config field"))?;
    let mut cfg: Qwen35Config = serde_json::from_value(text_config.clone())
        .context("failed to deserialize Qwen35Config from text_config")?;
    if let Some(vision_config) = raw.get("vision_config") {
        cfg.vision_config = Some(
            serde_json::from_value(vision_config.clone())
                .context("failed to deserialize VisionConfig")?,
        );
    }
    Ok(cfg)
}

fn qwen35_moe_config_from_raw(raw: &serde_json::Value) -> Result<Qwen35MoeConfig> {
    Qwen35MoeConfig::from_raw_config_value(raw)
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
        .route("/admin/api/models/mtp/validate", post(validate_mtp_handler))
        .route("/admin/api/models/unload", post(unload_model_handler))
        .route("/admin/api/models/pin", post(pin_model_handler))
        .route("/admin/api/models/unpin", post(unpin_model_handler))
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
    pinned: Option<bool>,
    mtp_model_dir: Option<String>,
    mtp_draft_tokens: Option<usize>,
    reload_when_idle: Option<bool>,
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
    fn compatible(draft_tokens: usize) -> Self {
        Self {
            compatible: true,
            reason_code: MTP_OK_CODE,
            message: "MTP weights are compatible with this model.".to_string(),
            draft_tokens: Some(draft_tokens),
        }
    }

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
    pinned: bool,
    max_position_embeddings: i32,
    mtp_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    mtp_model_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mtp_draft_tokens: Option<usize>,
}

impl From<EngineLoadedModelInfo> for LoadedModelInfo {
    fn from(info: EngineLoadedModelInfo) -> Self {
        Self {
            id: info.id.clone(),
            model: info.id,
            path: info.path,
            architecture: info.architecture,
            is_default: info.is_default,
            pinned: info.pinned,
            max_position_embeddings: info.max_position_embeddings,
            mtp_enabled: info.mtp_model_dir.is_some(),
            mtp_model_dir: info.mtp_model_dir,
            mtp_draft_tokens: info.mtp_draft_tokens,
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
    let mut kv_cache_logical_cap_tokens = 0;
    let mut kv_cache_resident_cap_tokens = 0;
    let mut kv_cache_budget_policies: Vec<String> = Vec::new();
    let mut mtp_enabled = false;
    let mut mtp_draft_token_values: Vec<usize> = Vec::new();
    let mut mtp_prefill_count = 0;
    let mut mtp_step_count = 0;
    let mut mtp_fallback_prefill_count = 0;
    let mut mtp_drafted_tokens = 0;
    let mut mtp_accepted_draft_tokens = 0;
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
        active_kv_snapshots.push(snapshot.active_kv_offload);
    }

    let active_kv_offload =
        crate::core::cache::ActiveKvOffloadHealth::aggregate(active_kv_snapshots);
    let mtp_draft_tokens = if mtp_enabled && mtp_draft_token_values.len() == 1 {
        mtp_draft_token_values.first().copied()
    } else {
        None
    };

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
            kv_cache_logical_cap_tokens,
            kv_cache_resident_cap_tokens,
            kv_cache_budget_policy: kv_cache_budget_policies.join(","),
            mlx_total_bytes: mlx_memory.total_bytes,
            mlx_max_recommended_bytes: mlx_memory.max_recommended_bytes,
            mlx_active_bytes: mlx_memory.active_bytes,
            mlx_cache_bytes: mlx_memory.cache_bytes,
            mlx_peak_bytes: mlx_memory.peak_bytes,
            mlx_memory_limit_bytes: mlx_memory.memory_limit_bytes,
        },
        mtp: MtpHealthInfo {
            enabled: mtp_enabled,
            draft_tokens: mtp_draft_tokens,
            prefill_count: mtp_prefill_count,
            step_count: mtp_step_count,
            fallback_prefill_count: mtp_fallback_prefill_count,
            drafted_tokens: mtp_drafted_tokens,
            accepted_draft_tokens: mtp_accepted_draft_tokens,
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
            &args,
            "gemma4-test".to_string(),
            &base,
            None,
            SamplingDefaults::default(),
            Some(crate::core::server::engine::EngineMtpSettings {
                model_dir: mtp,
                draft_tokens: Some(2),
            }),
            false,
        )
        .expect("build app dynamic model config");

        assert_eq!(load.config.scheduler_runtime_profile.config.b_max, 4);
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn aggregate_health_reports_loaded_mtp_state() {
        let snapshot = HealthSnapshot {
            status: HealthStatus::Healthy,
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
                admission_queue_full_count: 0,
                memory_budget_exceeded_count: 0,
            },
            memory: MemoryInfo {
                total_ram_bytes: 0,
                free_ram_bytes: 0,
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
            },
            mtp: MtpHealthInfo {
                enabled: true,
                draft_tokens: Some(2),
                prefill_count: 3,
                step_count: 5,
                fallback_prefill_count: 7,
                drafted_tokens: 11,
                accepted_draft_tokens: 13,
            },
            active_kv_offload: crate::core::cache::ActiveKvOffloadHealth::disabled(),
            device_name: None,
            version: "test",
        };

        let aggregated = aggregate_health(Instant::now(), vec![snapshot]);

        assert!(aggregated.mtp.enabled);
        assert_eq!(aggregated.mtp.draft_tokens, Some(2));
        assert_eq!(aggregated.mtp.prefill_count, 3);
        assert_eq!(aggregated.mtp.step_count, 5);
        assert_eq!(aggregated.mtp.fallback_prefill_count, 7);
        assert_eq!(aggregated.mtp.drafted_tokens, 11);
        assert_eq!(aggregated.mtp.accepted_draft_tokens, 13);
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
        assert!(error.message.contains("MAX TOKENS"));
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
            model_manifest: None,
            max_loaded_models: None,
            memory_limit_total_gb: None,
            memory_limit_model_gb: None,
            port: 8080,
            host: "127.0.0.1".to_string(),
            prefill_chunk_size: None,
            b_max: None,
            admission_deadline_ms: None,
            admission_queue_max: None,
            max_cache_cap: None,
            decode_cadence_mid_chunk_cap: None,
            scheduler_profile: None,
            scheduler_autotune_report: false,
            mtp_model_dir: None,
            mtp_draft_tokens: None,
            kv_quant: crate::cli::KvQuantArg::None,
            paged_prefix_cache_dir: None,
            paged_prefix_cache_block_size:
                crate::core::cache::DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE,
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

    fn qwen35_config(model_type: &str, mtp_layers: i32, hidden_size: i32) -> String {
        format!(
            r#"{{
                "model_type": "{model_type}",
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
        format!(
            r#"{{
                "model_type": "{model_type}",
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
