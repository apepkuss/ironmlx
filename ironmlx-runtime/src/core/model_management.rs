//! Native dynamic model loading, capability validation and deferred reload ownership.
use super::audio_execution::AudioModelResources;
use super::engine_pool::{
    EngineLoadPolicy, EngineLoadedModelInfo, EngineModelCapabilities, EngineModelConfig,
    EnginePoolState,
};
use super::model_validation::validate_mtp_pair;
use super::prompt_lookup::PromptLookupConfig;
use super::runtime_config::SamplingDefaults;
use super::scheduler_autotune::SchedulerAutotuneRuntimeContext;
use super::scheduler_profile_store::SchedulerProfileStore;
use super::scheduler_resolution::{
    apply_adaptive_mtp_scheduler_defaults, read_model_type,
    resolve_scheduler_for_model_with_speculative, ResolvedSchedulerRuntime, SchedulerProfileSource,
    SchedulerResolutionOptions,
};
use super::scheduler_resolution::{
    default_scheduler_runtime_profile, resolve_engine_pool_scheduler_profile,
    EnginePoolSchedulerProfileRequest, DEFAULT_MAX_CACHE_CAP,
};
use crate::Result;
use anyhow::bail;
use anyhow::Context;
use ironmlx_lm::models::ModelArchitecture;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct ModelManagement {
    pending_reloads: Arc<RwLock<HashMap<String, PendingModelReload>>>,
    pool: EnginePoolState,
    scheduler_options: SchedulerResolutionOptions,
}
impl ModelManagement {
    pub fn new(pool: EnginePoolState, scheduler_options: SchedulerResolutionOptions) -> Self {
        Self {
            pool,
            scheduler_options,
            pending_reloads: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelManagementError {
    #[error("model is busy")]
    Busy,
    #[error("model is not loaded: {0}")]
    NotLoaded(String),
    #[error("GPU memory headroom is insufficient")]
    GpuMemoryLow,
    #[error(transparent)]
    Load(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelManagementStatus {
    AlreadyLoaded,
    ReloadDeferred,
    Loaded,
    Registered,
    Reloaded,
}

pub struct ModelManagementOutcome {
    pub status: ModelManagementStatus,
    pub model: Option<String>,
    pub loaded_models: Vec<EngineLoadedModelInfo>,
    pub warning: Option<ModelLoadWarning>,
}
impl ModelManagementOutcome {
    fn new(
        status: ModelManagementStatus,
        model: Option<String>,
        loaded_models: Vec<EngineLoadedModelInfo>,
        warning: Option<ModelLoadWarning>,
    ) -> Self {
        Self {
            status,
            model,
            loaded_models,
            warning,
        }
    }
}
pub const DEFAULT_PROFILE_WARNING_CODE: &str = "default_scheduler_profile_used";

pub const DEFAULT_PROFILE_WARNING: &str =
    "No matching scheduler profile was found for this model. The model is running with the default scheduler configuration. Generate a dedicated profile with scheduler-autotune for better model-specific scheduling.";

pub const MODEL_RELOAD_DEFERRED_WARNING_CODE: &str = "model_reload_deferred";

pub const MODEL_RELOAD_DEFERRED_WARNING: &str =
    "The model is processing requests. New parameters will be applied automatically after the model becomes idle.";

pub const DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE: &str = "diffusion_gemma_mtp_unsupported";

pub const DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE: &str =
    "DiffusionGemma uses block diffusion and does not support MTP or speculative decoding. Disable MTP and try again.";

pub const DIFFUSION_GEMMA_PROMPT_LOOKUP_UNSUPPORTED_CODE: &str =
    "diffusion_gemma_prompt_lookup_unsupported";

pub const DIFFUSION_GEMMA_PROMPT_LOOKUP_UNSUPPORTED_MESSAGE: &str =
    "DiffusionGemma uses block diffusion and does not support PromptLookup. Disable PromptLookup and try again.";

pub const DIFFUSION_GEMMA_KV_CACHE_UNSUPPORTED_CODE: &str = "diffusion_gemma_kv_cache_unsupported";

pub const DIFFUSION_GEMMA_KV_CACHE_UNSUPPORTED_MESSAGE: &str =
    "DiffusionGemma does not use the causal KV cache. Remove the per-model MAX CONTEXT TOKENS cache override and try again.";

pub const DIFFUSION_GEMMA_SAMPLING_UNSUPPORTED_CODE: &str =
    "diffusion_gemma_sampling_parameter_unsupported";

pub const DIFFUSION_GEMMA_SAMPLING_UNSUPPORTED_MESSAGE: &str =
    "DiffusionGemma supports max_tokens, temperature, and seed. Remove top_p, top_k, and repetition_penalty overrides and try again.";

#[derive(Debug)]
pub struct ModelCapabilityError {
    pub code: &'static str,
    pub message: &'static str,
}

impl std::fmt::Display for ModelCapabilityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.message)
    }
}

impl std::error::Error for ModelCapabilityError {}

#[derive(Clone)]
struct PendingModelReload {
    pub audio: Option<AudioModelResources>,
    model_reference: String,
    model_dir: PathBuf,
    max_cache_cap_override: Option<usize>,
    sampling_defaults_override: SamplingDefaults,
    mtp: Option<crate::core::engine_pool::EngineMtpSettings>,
    prompt_lookup: Option<PromptLookupConfig>,
    pinned: bool,
    set_default: bool,
    defer_when_busy: bool,
}

pub struct EngineModelLoad {
    pub config: EngineModelConfig,
    pub warning: Option<ModelLoadWarning>,
}

pub struct ModelLoadRequest {
    pub audio: Option<AudioModelResources>,
    pub model_reference: String,
    pub model_dir: PathBuf,
    pub max_cache_cap_override: Option<usize>,
    pub sampling_defaults: SamplingDefaults,
    pub mtp: Option<crate::core::engine_pool::EngineMtpSettings>,
    pub prompt_lookup: Option<PromptLookupConfig>,
    pub pinned: bool,
    pub set_default: bool,
    pub reload_when_idle: bool,
    pub defer_when_busy: bool,
}

pub struct EngineModelBuildRequest<'a> {
    pub audio: Option<AudioModelResources>,
    pub model_id: String,
    pub model_dir: &'a Path,
    pub max_cache_cap_override: Option<usize>,
    pub sampling_defaults_override: SamplingDefaults,
    pub mtp: Option<crate::core::engine_pool::EngineMtpSettings>,
    pub prompt_lookup: Option<PromptLookupConfig>,
    pub pinned: bool,
}

pub fn build_engine_model_config(
    args: &SchedulerResolutionOptions,
    request: EngineModelBuildRequest<'_>,
) -> Result<EngineModelLoad> {
    let EngineModelBuildRequest {
        audio,
        model_id,
        model_dir,
        max_cache_cap_override,
        sampling_defaults_override,
        mtp,
        prompt_lookup,
        pinned,
    } = request;
    if let Some(audio) = audio {
        if max_cache_cap_override.is_some()
            || mtp.is_some()
            || prompt_lookup.is_some()
            || sampling_defaults_override != SamplingDefaults::default()
        {
            bail!("audio models do not accept causal scheduler or sampling overrides");
        }
        return Ok(EngineModelLoad {
            config: EngineModelConfig {
                audio: Some(audio),
                id: model_id,
                path: model_dir.to_path_buf(),
                load_policy: EngineLoadPolicy::Lazy,
                default: false,
                pinned,
                scheduler_runtime_profile: None,
                mtp: None,
                prompt_lookup: None,
                sampling_defaults: SamplingDefaults::default(),
                capabilities: EngineModelCapabilities::audio(),
            },
            warning: None,
        });
    }
    let model_type = read_model_type(model_dir)?;
    let architecture = ModelArchitecture::from_model_type(&model_type)?;
    let capabilities = engine_model_capabilities(architecture, model_dir)?;
    if architecture == ModelArchitecture::DiffusionGemma {
        validate_diffusion_gemma_model_request(
            max_cache_cap_override,
            sampling_defaults_override,
            mtp.as_ref(),
            prompt_lookup.as_ref(),
        )?;
        return Ok(EngineModelLoad {
            config: EngineModelConfig {
                audio: None,
                id: model_id,
                path: model_dir.to_path_buf(),
                load_policy: EngineLoadPolicy::Lazy,
                default: false,
                pinned,
                scheduler_runtime_profile: None,
                mtp: None,
                prompt_lookup: None,
                sampling_defaults: sampling_defaults_override,
                capabilities,
            },
            warning: None,
        });
    }
    let mut resolved = apply_load_request_scheduler_overrides(
        resolve_scheduler_for_model_with_speculative(
            args,
            model_dir,
            mtp.as_ref().map(|settings| settings.model_dir.as_path()),
            mtp.as_ref().and_then(|settings| settings.draft_tokens),
            prompt_lookup,
            max_cache_cap_override,
        )?,
        max_cache_cap_override,
    );
    let sampling_defaults = read_generation_sampling_defaults(model_dir)?
        .merge_with_override(sampling_defaults_override);
    let warning = match resolved.profile_source {
        None if args.scheduler_profile.is_none() => Some(ModelLoadWarning::new(
            DEFAULT_PROFILE_WARNING_CODE,
            DEFAULT_PROFILE_WARNING,
        )),
        Some(SchedulerProfileSource::Explicit | SchedulerProfileSource::Store) | None => None,
    };
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
    if apply_adaptive_mtp_scheduler_defaults(args, architecture, mtp.is_some(), &mut resolved) {
        tracing::info!(
            "ironmlx app: adaptive MTP scheduler default applied model_id={} b_max={}",
            model_id,
            resolved.scheduler_config.b_max
        );
    }
    Ok(EngineModelLoad {
        config: EngineModelConfig {
            audio: None,
            id: model_id,
            path: model_dir.to_path_buf(),
            load_policy: EngineLoadPolicy::Lazy,
            default: false,
            pinned,
            scheduler_runtime_profile: Some(resolved.scheduler_runtime_profile),
            mtp,
            prompt_lookup,
            sampling_defaults,
            capabilities,
        },
        warning,
    })
}

fn validate_diffusion_gemma_model_request(
    max_cache_cap_override: Option<usize>,
    sampling_defaults: SamplingDefaults,
    mtp: Option<&crate::core::engine_pool::EngineMtpSettings>,
    prompt_lookup: Option<&PromptLookupConfig>,
) -> Result<()> {
    let error = if mtp.is_some() {
        Some(ModelCapabilityError {
            code: DIFFUSION_GEMMA_MTP_UNSUPPORTED_CODE,
            message: DIFFUSION_GEMMA_MTP_UNSUPPORTED_MESSAGE,
        })
    } else if prompt_lookup.is_some() {
        Some(ModelCapabilityError {
            code: DIFFUSION_GEMMA_PROMPT_LOOKUP_UNSUPPORTED_CODE,
            message: DIFFUSION_GEMMA_PROMPT_LOOKUP_UNSUPPORTED_MESSAGE,
        })
    } else if max_cache_cap_override.is_some() {
        Some(ModelCapabilityError {
            code: DIFFUSION_GEMMA_KV_CACHE_UNSUPPORTED_CODE,
            message: DIFFUSION_GEMMA_KV_CACHE_UNSUPPORTED_MESSAGE,
        })
    } else if sampling_defaults.top_p.is_some()
        || sampling_defaults.top_k.is_some()
        || sampling_defaults.repetition_penalty.is_some()
    {
        Some(ModelCapabilityError {
            code: DIFFUSION_GEMMA_SAMPLING_UNSUPPORTED_CODE,
            message: DIFFUSION_GEMMA_SAMPLING_UNSUPPORTED_MESSAGE,
        })
    } else {
        None
    };
    match error {
        Some(error) => Err(error.into()),
        None => Ok(()),
    }
}

pub fn engine_model_capabilities(
    architecture: ModelArchitecture,
    model_dir: &Path,
) -> Result<EngineModelCapabilities> {
    let config_path = model_dir.join("config.json");
    let config_data = std::fs::read(&config_path)
        .with_context(|| format!("reading model capabilities {}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_slice(&config_data)
        .with_context(|| format!("parsing model capabilities {}", config_path.display()))?;
    let has_vision_config = config
        .get("vision_config")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|vision| !vision.is_empty());
    Ok(EngineModelCapabilities::for_architecture(
        architecture,
        matches!(
            architecture,
            ModelArchitecture::DiffusionGemma | ModelArchitecture::MiniCpmV46
        ) || has_vision_config,
    ))
}

pub fn read_generation_sampling_defaults(model_dir: &Path) -> Result<SamplingDefaults> {
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

pub fn apply_load_request_scheduler_overrides(
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

#[derive(Debug, Clone)]
pub struct ModelLoadWarning {
    pub code: &'static str,
    pub message: String,
}

impl ModelLoadWarning {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

fn ensure_gpu_memory_headroom() -> std::result::Result<(), ModelManagementError> {
    let memory = mlx::memory::snapshot();
    if let Some(max_recommended) = memory.max_recommended_bytes {
        if max_recommended > 0 && memory.active_bytes >= max_recommended {
            return Err(ModelManagementError::GpuMemoryLow);
        }
    }
    Ok(())
}

impl ModelManagement {
    pub async fn load_model(
        &self,
        request: ModelLoadRequest,
    ) -> std::result::Result<ModelManagementOutcome, ModelManagementError> {
        let parsed = request;
        let reload = PendingModelReload {
            audio: parsed.audio.clone(),
            model_reference: parsed.model_reference.clone(),
            model_dir: parsed.model_dir.clone(),
            max_cache_cap_override: parsed.max_cache_cap_override,
            sampling_defaults_override: parsed.sampling_defaults,
            mtp: parsed.mtp.clone(),
            prompt_lookup: parsed.prompt_lookup,
            pinned: parsed.pinned,
            set_default: parsed.set_default,
            defer_when_busy: parsed.defer_when_busy,
        };

        let already_loaded = self.pool.is_model_loaded(&parsed.model_reference).await;
        if already_loaded && !parsed.reload_when_idle {
            return Ok(ModelManagementOutcome::new(
                ModelManagementStatus::AlreadyLoaded,
                Some(parsed.model_reference),
                self.pool.loaded_model_infos().await,
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
            if !parsed.defer_when_busy {
                return Err(ModelManagementError::Busy);
            }
            self.schedule_reload_when_idle(reload).await;
            return Ok(ModelManagementOutcome::new(
                ModelManagementStatus::ReloadDeferred,
                Some(parsed.model_reference),
                self.pool.loaded_model_infos().await,
                Some(ModelLoadWarning::new(
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
            &self.scheduler_options,
            EngineModelBuildRequest {
                audio: parsed.audio,
                model_id: parsed.model_reference.clone(),
                model_dir: &parsed.model_dir,
                max_cache_cap_override: parsed.max_cache_cap_override,
                sampling_defaults_override: parsed.sampling_defaults,
                mtp: parsed.mtp,
                prompt_lookup: parsed.prompt_lookup,
                pinned: parsed.pinned,
            },
        )
        .map_err(ModelManagementError::Load)?;
        self.pool
            .reload_dynamic_model(load.config, parsed.set_default)
            .await
            .map_err(ModelManagementError::Load)?;
        let loaded_models = self.pool.loaded_model_infos().await;
        Ok(ModelManagementOutcome::new(
            ModelManagementStatus::Loaded,
            Some(parsed.model_reference),
            loaded_models,
            load.warning,
        ))
    }

    pub async fn register_model(
        &self,
        request: ModelLoadRequest,
        pinned: Option<bool>,
    ) -> std::result::Result<ModelManagementOutcome, ModelManagementError> {
        let parsed = request;
        let load = build_engine_model_config(
            &self.scheduler_options,
            EngineModelBuildRequest {
                audio: parsed.audio,
                model_id: parsed.model_reference.clone(),
                model_dir: &parsed.model_dir,
                max_cache_cap_override: parsed.max_cache_cap_override,
                sampling_defaults_override: parsed.sampling_defaults,
                mtp: parsed.mtp,
                prompt_lookup: parsed.prompt_lookup,
                pinned: parsed.pinned,
            },
        )
        .map_err(ModelManagementError::Load)?;
        self.pool
            .register_dynamic_model(load.config, parsed.set_default, pinned)
            .await
            .map_err(ModelManagementError::Load)?;
        Ok(ModelManagementOutcome::new(
            ModelManagementStatus::Registered,
            Some(parsed.model_reference),
            self.pool.loaded_model_infos().await,
            load.warning,
        ))
    }

    async fn reload_model_now(
        &self,
        reload: PendingModelReload,
    ) -> std::result::Result<ModelManagementOutcome, ModelManagementError> {
        if self
            .pool
            .pending_requests(&reload.model_reference)
            .await
            .is_some_and(|requests| requests > 0)
        {
            if !reload.defer_when_busy {
                return Err(ModelManagementError::Busy);
            }
            self.schedule_reload_when_idle(reload.clone()).await;
            return Ok(ModelManagementOutcome::new(
                ModelManagementStatus::ReloadDeferred,
                Some(reload.model_reference),
                self.pool.loaded_model_infos().await,
                Some(ModelLoadWarning::new(
                    MODEL_RELOAD_DEFERRED_WARNING_CODE,
                    MODEL_RELOAD_DEFERRED_WARNING,
                )),
            ));
        }
        if !self.pool.is_model_loaded(&reload.model_reference).await {
            return Err(ModelManagementError::NotLoaded(
                reload.model_reference.clone(),
            ));
        }

        let load = build_engine_model_config(
            &self.scheduler_options,
            EngineModelBuildRequest {
                audio: reload.audio,
                model_id: reload.model_reference.clone(),
                model_dir: &reload.model_dir,
                max_cache_cap_override: reload.max_cache_cap_override,
                sampling_defaults_override: reload.sampling_defaults_override,
                mtp: reload.mtp.clone(),
                prompt_lookup: reload.prompt_lookup,
                pinned: reload.pinned,
            },
        )
        .map_err(ModelManagementError::Load)?;
        self.pool
            .reload_dynamic_model(load.config, reload.set_default)
            .await
            .map_err(ModelManagementError::Load)?;
        let loaded_models = self.pool.loaded_model_infos().await;
        Ok(ModelManagementOutcome::new(
            ModelManagementStatus::Reloaded,
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
        let scheduler_options = self.scheduler_options.clone();

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
                    &scheduler_options,
                    EngineModelBuildRequest {
                        audio: reload.audio,
                        model_id: reload.model_reference.clone(),
                        model_dir: &reload.model_dir,
                        max_cache_cap_override: reload.max_cache_cap_override,
                        sampling_defaults_override: reload.sampling_defaults_override,
                        mtp: reload.mtp.clone(),
                        prompt_lookup: reload.prompt_lookup,
                        pinned: reload.pinned,
                    },
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

pub fn build_engine_model_config_for_pool(
    args: &SchedulerResolutionOptions,
    model: crate::core::engine_pool::EngineModelManifest,
    scheduler_profile_store: Option<&SchedulerProfileStore>,
    hardware_label: &str,
) -> Result<crate::core::engine_pool::EngineModelConfig> {
    if let Some(audio) = model.audio.clone() {
        if model.scheduler_profile.is_some()
            || model.mtp_model_dir.is_some()
            || model.mtp_draft_tokens.is_some()
            || model.prompt_lookup.is_some()
        {
            bail!("audio model does not accept causal execution settings");
        }
        let config = EngineModelConfig {
            audio: Some(audio),
            id: model.id,
            path: model.path,
            load_policy: model.load_policy,
            default: model.default,
            pinned: false,
            scheduler_runtime_profile: None,
            mtp: None,
            prompt_lookup: None,
            sampling_defaults: SamplingDefaults::default(),
            capabilities: EngineModelCapabilities::audio(),
        };
        super::runtime_config::validate_engine_model_config(&config)?;
        return Ok(config);
    }
    let prompt_lookup = model
        .prompt_lookup
        .map(crate::core::prompt_lookup::PromptLookupConfig::validate)
        .transpose()?;
    let mtp = model
        .mtp_model_dir
        .map(|model_dir| crate::core::engine_pool::EngineMtpSettings {
            model_dir,
            draft_tokens: model.mtp_draft_tokens,
        });
    if model.load_policy == crate::core::engine_pool::EngineLoadPolicy::Disabled {
        return Ok(crate::core::engine_pool::EngineModelConfig {
            audio: None,
            id: model.id,
            path: model.path,
            load_policy: model.load_policy,
            default: model.default,
            pinned: false,
            scheduler_runtime_profile: Some(default_scheduler_runtime_profile(
                SchedulerAutotuneRuntimeContext::local_default(DEFAULT_MAX_CACHE_CAP),
            )),
            mtp,
            prompt_lookup,
            sampling_defaults: crate::core::runtime_config::SamplingDefaults::default(),
            capabilities: crate::core::engine_pool::EngineModelCapabilities::for_architecture(
                ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
                false,
            ),
        });
    }
    if !model.path.exists() {
        bail!(
            "engine model `{}` path must point to a local directory (got '{}')",
            model.id,
            model.path.display()
        );
    }
    if model.mtp_draft_tokens.is_some() && mtp.is_none() {
        bail!(
            "engine model `{}` sets mtp_draft_tokens without mtp_model_dir",
            model.id
        );
    }
    let model_type = read_model_type(&model.path)?;
    let architecture = ironmlx_lm::models::ModelArchitecture::from_model_type(&model_type)?;
    let config_data = std::fs::read(model.path.join("config.json"))?;
    let config: serde_json::Value = serde_json::from_slice(&config_data)?;
    let supports_vision = matches!(
        architecture,
        ironmlx_lm::models::ModelArchitecture::DiffusionGemma
            | ironmlx_lm::models::ModelArchitecture::MiniCpmV46
    ) || config
        .get("vision_config")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|vision| !vision.is_empty());
    let capabilities = crate::core::engine_pool::EngineModelCapabilities::for_architecture(
        architecture,
        supports_vision,
    );
    if architecture == ironmlx_lm::models::ModelArchitecture::DiffusionGemma {
        if mtp.is_some() {
            bail!(
                "engine model `{}` configures MTP for DiffusionGemma",
                model.id
            );
        }
        if prompt_lookup.is_some() {
            bail!(
                "engine model `{}` configures PromptLookup for DiffusionGemma",
                model.id
            );
        }
        if model.scheduler_profile.is_some() {
            bail!(
                "engine model `{}` configures a causal scheduler profile for DiffusionGemma",
                model.id
            );
        }
        return Ok(crate::core::engine_pool::EngineModelConfig {
            audio: None,
            id: model.id,
            path: model.path,
            load_policy: model.load_policy,
            default: model.default,
            pinned: false,
            scheduler_runtime_profile: None,
            mtp: None,
            prompt_lookup: None,
            sampling_defaults: crate::core::runtime_config::SamplingDefaults::default(),
            capabilities,
        });
    }
    let mut resolved = resolve_engine_pool_scheduler_profile(
        args,
        &model.path,
        EnginePoolSchedulerProfileRequest {
            manifest_profile: model.scheduler_profile.as_deref(),
            store: scheduler_profile_store,
            hardware_label,
            mtp_model_dir: mtp.as_ref().map(|settings| settings.model_dir.as_path()),
            mtp_draft_tokens: mtp.as_ref().and_then(|settings| settings.draft_tokens),
            prompt_lookup,
        },
    )?;
    if apply_adaptive_mtp_scheduler_defaults(args, architecture, mtp.is_some(), &mut resolved) {
        tracing::info!(
            "ironmlx serve: adaptive MTP scheduler default applied manifest_model={} b_max={}",
            model.id,
            resolved.scheduler_config.b_max
        );
    }
    Ok(crate::core::engine_pool::EngineModelConfig {
        audio: None,
        id: model.id,
        path: model.path,
        load_policy: model.load_policy,
        default: model.default,
        pinned: false,
        scheduler_runtime_profile: Some(resolved.scheduler_runtime_profile),
        mtp,
        prompt_lookup,
        sampling_defaults: crate::core::runtime_config::SamplingDefaults::default(),
        capabilities,
    })
}
