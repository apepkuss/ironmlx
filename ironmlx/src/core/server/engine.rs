use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Notify};

use crate::core::cache::{
    ActiveKvOffloadConfig, PagedPrefixCacheConfig, PrefixLruCacheConfig, TurboQuantKVBits,
};
use crate::core::model::Model;
use crate::core::sampler::Sampler;
use crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile;
use crate::core::speculative::{MtpDraftTokensArg, MtpSpeculativeConfig, MtpSpeculativeModel};
use crate::core::{Loader, Tokenizer};
use crate::models::{
    DiffusionGemmaConfig, DiffusionGemmaGenerationConfig, DiffusionGemmaModel, Gemma4Config,
    Gemma4Model, Glm4MoeLiteModel, LlamaModel, MiniCpmV46Model, ModelArchitecture, Qwen35Model,
    Qwen35MoeModel, Qwen36MoeModel,
};
use crate::Result;

use super::{
    anthropic, diffusion_gemma, health, openai, AppState, SamplingDefaults, VisionInputConfig,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineLoadPolicy {
    Preload,
    #[default]
    Lazy,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineModelManifest {
    pub id: String,
    pub path: PathBuf,
    #[serde(default)]
    pub load_policy: EngineLoadPolicy,
    #[serde(default)]
    pub default: bool,
    #[serde(default)]
    pub scheduler_profile: Option<PathBuf>,
    #[serde(default)]
    pub mtp_model_dir: Option<PathBuf>,
    #[serde(default)]
    pub mtp_draft_tokens: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnginePoolManifest {
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub max_loaded_models: Option<usize>,
    pub models: Vec<EngineModelManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineRegistryError {
    EmptyModelId,
    DuplicateModelId {
        id: String,
    },
    EmptyManifest,
    NoEnabledModels,
    InvalidMaxLoadedModels,
    PreloadCapacityExceeded {
        preload_count: usize,
        max_loaded_models: usize,
    },
    UnknownModel {
        id: String,
    },
    ModelDisabled {
        id: String,
    },
    AmbiguousDefault,
    DuplicateDefaultModels {
        first: String,
        second: String,
    },
    ConflictingDefaultModels {
        top_level: String,
        model: String,
    },
}

impl std::fmt::Display for EngineRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModelId => write!(f, "engine model id must not be empty"),
            Self::DuplicateModelId { id } => write!(f, "duplicate engine model id `{id}`"),
            Self::EmptyManifest => write!(f, "engine pool manifest must contain at least one model"),
            Self::NoEnabledModels => write!(
                f,
                "engine pool manifest must contain at least one enabled model"
            ),
            Self::InvalidMaxLoadedModels => {
                write!(f, "engine pool max_loaded_models must be >= 1")
            }
            Self::PreloadCapacityExceeded {
                preload_count,
                max_loaded_models,
            } => write!(
                f,
                "engine pool preload model count ({preload_count}) exceeds max_loaded_models ({max_loaded_models})"
            ),
            Self::UnknownModel { id } => write!(f, "unknown engine model `{id}`"),
            Self::ModelDisabled { id } => write!(f, "engine model `{id}` is disabled"),
            Self::AmbiguousDefault => {
                write!(f, "request model is required when multiple models are enabled")
            }
            Self::DuplicateDefaultModels { first, second } => write!(
                f,
                "engine pool manifest declares multiple default models: `{first}` and `{second}`"
            ),
            Self::ConflictingDefaultModels { top_level, model } => write!(
                f,
                "engine pool manifest default_model `{top_level}` conflicts with model default `{model}`"
            ),
        }
    }
}

impl std::error::Error for EngineRegistryError {}

#[derive(Debug, Clone)]
pub struct EngineRegistry {
    models: Vec<EngineModelManifest>,
    index: HashMap<String, usize>,
    default_model: Option<String>,
    max_loaded_models: Option<usize>,
}

impl EngineRegistry {
    pub fn new(manifest: EnginePoolManifest) -> Result<Self, EngineRegistryError> {
        if manifest.models.is_empty() {
            return Err(EngineRegistryError::EmptyManifest);
        }
        if manifest.max_loaded_models == Some(0) {
            return Err(EngineRegistryError::InvalidMaxLoadedModels);
        }

        let mut index = HashMap::with_capacity(manifest.models.len());
        let mut seen = HashSet::with_capacity(manifest.models.len());
        let mut model_default: Option<String> = None;
        for (idx, model) in manifest.models.iter().enumerate() {
            if model.id.is_empty() {
                return Err(EngineRegistryError::EmptyModelId);
            }
            if !seen.insert(model.id.clone()) {
                return Err(EngineRegistryError::DuplicateModelId {
                    id: model.id.clone(),
                });
            }
            index.insert(model.id.clone(), idx);
            if model.default {
                match model_default.as_ref() {
                    Some(first) => {
                        return Err(EngineRegistryError::DuplicateDefaultModels {
                            first: first.clone(),
                            second: model.id.clone(),
                        });
                    }
                    None => model_default = Some(model.id.clone()),
                }
            }
        }

        let enabled_count = manifest
            .models
            .iter()
            .filter(|model| model.load_policy != EngineLoadPolicy::Disabled)
            .count();
        if enabled_count == 0 {
            return Err(EngineRegistryError::NoEnabledModels);
        }
        if let Some(max_loaded_models) = manifest.max_loaded_models {
            let preload_count = manifest
                .models
                .iter()
                .filter(|model| model.load_policy == EngineLoadPolicy::Preload)
                .count();
            if preload_count > max_loaded_models {
                return Err(EngineRegistryError::PreloadCapacityExceeded {
                    preload_count,
                    max_loaded_models,
                });
            }
        }

        if let Some(default_model) = manifest.default_model.as_ref() {
            match index.get(default_model) {
                Some(idx) if manifest.models[*idx].load_policy == EngineLoadPolicy::Disabled => {
                    return Err(EngineRegistryError::ModelDisabled {
                        id: default_model.clone(),
                    });
                }
                Some(_) => {}
                None => {
                    return Err(EngineRegistryError::UnknownModel {
                        id: default_model.clone(),
                    });
                }
            }
        }

        let default_model = match (manifest.default_model, model_default) {
            (Some(top_level), Some(model)) if top_level != model => {
                return Err(EngineRegistryError::ConflictingDefaultModels { top_level, model });
            }
            (Some(top_level), _) => Some(top_level),
            (None, model_default) => model_default,
        };

        Ok(Self {
            models: manifest.models,
            index,
            default_model,
            max_loaded_models: manifest.max_loaded_models,
        })
    }

    pub fn empty(max_loaded_models: Option<usize>) -> Result<Self, EngineRegistryError> {
        if max_loaded_models == Some(0) {
            return Err(EngineRegistryError::InvalidMaxLoadedModels);
        }
        Ok(Self {
            models: Vec::new(),
            index: HashMap::new(),
            default_model: None,
            max_loaded_models,
        })
    }

    pub fn upsert_model(
        &mut self,
        mut model: EngineModelManifest,
        set_default: bool,
    ) -> Result<(), EngineRegistryError> {
        if model.id.is_empty() {
            return Err(EngineRegistryError::EmptyModelId);
        }
        let was_current_default = self.default_model.as_deref() == Some(model.id.as_str());
        let becomes_default = set_default || self.default_model.is_none() || was_current_default;
        model.default = becomes_default;
        if becomes_default {
            for existing in &mut self.models {
                existing.default = false;
            }
        }
        match self.index.get(&model.id).copied() {
            Some(idx) => {
                self.models[idx] = model.clone();
            }
            None => {
                let idx = self.models.len();
                self.index.insert(model.id.clone(), idx);
                self.models.push(model.clone());
            }
        }
        if becomes_default {
            self.default_model = Some(model.id);
        }
        Ok(())
    }

    pub fn remove_model(&mut self, id: &str) -> bool {
        let Some(idx) = self.index.remove(id) else {
            return false;
        };
        self.models.remove(idx);
        self.index.clear();
        for (idx, model) in self.models.iter().enumerate() {
            self.index.insert(model.id.clone(), idx);
        }
        if self.default_model.as_deref() == Some(id) {
            self.default_model = self.servable_models().first().map(|model| model.id.clone());
        }
        true
    }

    pub fn set_default_model(&mut self, id: &str) -> Result<(), EngineRegistryError> {
        let Some(model) = self.model(id) else {
            return Err(EngineRegistryError::UnknownModel { id: id.to_string() });
        };
        if model.load_policy == EngineLoadPolicy::Disabled {
            return Err(EngineRegistryError::ModelDisabled { id: id.to_string() });
        }
        for model in &mut self.models {
            model.default = model.id == id;
        }
        self.default_model = Some(id.to_string());
        Ok(())
    }

    fn restore_default_model(
        &mut self,
        default_model: Option<String>,
    ) -> Result<(), EngineRegistryError> {
        match default_model {
            Some(id) if self.model(&id).is_some() => self.set_default_model(&id),
            Some(_) | None => {
                for model in &mut self.models {
                    model.default = false;
                }
                self.default_model = None;
                Ok(())
            }
        }
    }

    pub fn resolve_model_id(&self, requested: Option<&str>) -> Result<&str, EngineRegistryError> {
        if let Some(requested) = requested.filter(|value| !value.is_empty()) {
            let Some(model) = self.model(requested) else {
                return Err(EngineRegistryError::UnknownModel {
                    id: requested.to_string(),
                });
            };
            if model.load_policy == EngineLoadPolicy::Disabled {
                return Err(EngineRegistryError::ModelDisabled {
                    id: requested.to_string(),
                });
            }
            return Ok(model.id.as_str());
        }

        if let Some(default_model) = self.default_model.as_deref() {
            return Ok(default_model);
        }

        let enabled = self.servable_models();
        let mut enabled = enabled.iter();
        let Some(model) = enabled.next() else {
            return Err(EngineRegistryError::AmbiguousDefault);
        };
        if enabled.next().is_none() {
            Ok(model.id.as_str())
        } else {
            Err(EngineRegistryError::AmbiguousDefault)
        }
    }

    pub fn servable_models(&self) -> Vec<&EngineModelManifest> {
        self.models
            .iter()
            .filter(|model| model.load_policy != EngineLoadPolicy::Disabled)
            .collect()
    }

    pub fn servable_models_owned(&self) -> Vec<EngineModelManifest> {
        self.models
            .iter()
            .filter(|model| model.load_policy != EngineLoadPolicy::Disabled)
            .cloned()
            .collect()
    }

    pub fn model(&self, id: &str) -> Option<&EngineModelManifest> {
        self.index.get(id).map(|idx| &self.models[*idx])
    }

    pub fn models(&self) -> &[EngineModelManifest] {
        &self.models
    }

    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    pub fn max_loaded_models(&self) -> Option<usize> {
        self.max_loaded_models
    }
}

#[derive(Debug, Clone)]
pub struct EngineMtpSettings {
    pub model_dir: PathBuf,
    pub draft_tokens: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct EngineModelConfig {
    pub id: String,
    pub path: PathBuf,
    pub load_policy: EngineLoadPolicy,
    pub default: bool,
    pub scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    pub mtp: Option<EngineMtpSettings>,
    pub sampling_defaults: SamplingDefaults,
}

impl EngineModelConfig {
    fn manifest_view(&self) -> EngineModelManifest {
        EngineModelManifest {
            id: self.id.clone(),
            path: self.path.clone(),
            load_policy: self.load_policy,
            default: self.default,
            scheduler_profile: None,
            mtp_model_dir: self.mtp.as_ref().map(|mtp| mtp.model_dir.clone()),
            mtp_draft_tokens: self.mtp.as_ref().and_then(|mtp| mtp.draft_tokens),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnginePoolConfig {
    pub default_model: Option<String>,
    pub max_loaded_models: Option<usize>,
    pub models: Vec<EngineModelConfig>,
}

impl EnginePoolConfig {
    fn registry_manifest(&self) -> EnginePoolManifest {
        EnginePoolManifest {
            default_model: self.default_model.clone(),
            max_loaded_models: self.max_loaded_models,
            models: self
                .models
                .iter()
                .map(EngineModelConfig::manifest_view)
                .collect(),
        }
    }

    fn validate_enabled_model_architectures(&self) -> Result<()> {
        for model in &self.models {
            validate_engine_model_config(model)?;
        }
        Ok(())
    }
}

fn validate_engine_model_config(model: &EngineModelConfig) -> Result<()> {
    if model.load_policy == EngineLoadPolicy::Disabled {
        return Ok(());
    }
    let config_path = model.path.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", config_path.display()))?;
    ModelArchitecture::from_config_value(&config).with_context(|| {
        format!(
            "engine model `{}` has unsupported architecture in {}",
            model.id,
            config_path.display()
        )
    })?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct EnginePagedPrefixCacheSettings {
    pub root: PathBuf,
    pub block_size: i32,
    pub max_pages: Option<i32>,
    pub max_disk_bytes: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct EnginePoolRuntimeConfig {
    pub host: String,
    pub port: u16,
    pub kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    pub scheduler_autotune_report: bool,
    pub paged_prefix_cache: Option<EnginePagedPrefixCacheSettings>,
    pub prefix_lru_cache_max_bytes: Option<usize>,
    pub model_ttl: Option<Duration>,
    pub memory_limits: EnginePoolMemoryLimits,
    pub active_kv_offload: ActiveKvOffloadConfig,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnginePoolMemoryLimits {
    pub total_memory_limit_bytes: Option<usize>,
    pub model_memory_limit_bytes: Option<usize>,
}

impl EnginePoolMemoryLimits {
    pub fn check_model_memory_limit(
        &self,
        model_id: &str,
        loaded_model_bytes: usize,
    ) -> Result<()> {
        if let Some(limit) = self.model_memory_limit_bytes {
            if loaded_model_bytes > limit {
                bail!(
                    "engine pool model memory limit exceeded: model={model_id} loaded_model_bytes={loaded_model_bytes} > limit={limit}"
                );
            }
        }
        Ok(())
    }

    pub fn check_total_memory_limit(&self, model_id: &str, mlx_active_bytes: usize) -> Result<()> {
        if let Some(limit) = self.total_memory_limit_bytes {
            if mlx_active_bytes > limit {
                bail!(
                    "engine pool total memory limit exceeded: model={model_id} mlx_active_bytes={mlx_active_bytes} > limit={limit}"
                );
            }
        }
        Ok(())
    }
}

impl EnginePoolRuntimeConfig {
    fn paged_prefix_cache_config(
        &self,
        model_id: &str,
        scheduler_config: crate::core::scheduler_autotune::SchedulerAutotuneProfileConfig,
    ) -> Result<Option<PagedPrefixCacheConfig>> {
        let Some(settings) = self.paged_prefix_cache.as_ref() else {
            return Ok(None);
        };
        let max_pages = match settings.max_pages {
            Some(max_pages) => max_pages,
            None => {
                let tokens = scheduler_config
                    .max_cache_cap
                    .saturating_mul(scheduler_config.b_max);
                let pages = tokens.div_ceil(settings.block_size as usize).max(1);
                i32::try_from(pages).context("derived paged prefix cache max_pages exceeds i32")?
            }
        };
        PagedPrefixCacheConfig::new_with_max_disk_bytes(
            &settings.root,
            model_id.to_string(),
            settings.block_size,
            max_pages,
            settings.max_disk_bytes,
        )
        .map(Some)
    }

    fn prefix_lru_cache_config(
        &self,
        paged_prefix_cache: Option<&PagedPrefixCacheConfig>,
    ) -> Result<Option<PrefixLruCacheConfig>> {
        let Some(max_bytes) = self.prefix_lru_cache_max_bytes else {
            return Ok(None);
        };
        if paged_prefix_cache.is_none() {
            bail!("prefix LRU cache requires paged prefix cache");
        }
        PrefixLruCacheConfig::new(max_bytes).map(Some)
    }
}

#[derive(Clone)]
pub struct EnginePoolState {
    inner: Arc<EnginePoolInner>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EngineLoadedModelInfo {
    pub id: String,
    pub path: String,
    pub architecture: String,
    pub is_default: bool,
    pub max_position_embeddings: i32,
}

struct EnginePoolInner {
    registry: Mutex<EngineRegistry>,
    slots: Mutex<HashMap<String, Arc<EngineSlot>>>,
    runtime: EnginePoolRuntimeConfig,
    capacity_policy: EnginePoolCapacityPolicy,
    load_gate: Mutex<()>,
}

struct EngineSlot {
    model: EngineModelConfig,
    runtime: EnginePoolRuntimeConfig,
    active_requests: Arc<AtomicUsize>,
    state: Mutex<EngineSlotState>,
    notify: Notify,
}

enum EngineSlotState {
    Unloaded {
        reason: EngineUnloadReason,
        last_error: Option<String>,
        changed_unix_ms: u64,
        load_attempts: u64,
    },
    Loading {
        started_unix_ms: u64,
        load_attempts: u64,
    },
    Loaded {
        engine: Arc<EngineVariant>,
        loaded_unix_ms: u64,
        last_used_unix_ms: u64,
        load_attempts: u64,
        request_count: u64,
    },
    Draining {
        started_unix_ms: u64,
        load_attempts: u64,
        request_count: u64,
    },
    Failed {
        last_error: String,
        failed_unix_ms: u64,
        load_attempts: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum EngineUnloadReason {
    Startup,
    Evicted,
    Manual,
    Ttl,
    CapacityRejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EngineRuntimeState {
    Disabled,
    Unloaded,
    Loading,
    Loaded,
    Draining,
    Failed,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineLoadTrigger {
    Request,
    Control,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnginePoolCapacityPolicy {
    EvictLruIdle,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnginePoolCapacityDecision {
    Continue,
    Reject,
    TryEvictLruIdle,
}

fn decide_engine_pool_capacity(
    policy: EnginePoolCapacityPolicy,
    max_loaded_models: usize,
    loaded_count: usize,
) -> EnginePoolCapacityDecision {
    if loaded_count < max_loaded_models {
        return EnginePoolCapacityDecision::Continue;
    }
    match policy {
        EnginePoolCapacityPolicy::EvictLruIdle => EnginePoolCapacityDecision::TryEvictLruIdle,
        EnginePoolCapacityPolicy::Reject => EnginePoolCapacityDecision::Reject,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EvictionCandidate {
    id: String,
    last_used_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelTtlCandidate {
    id: String,
    state: EngineRuntimeState,
    active_requests: usize,
    last_used_unix_ms: Option<u64>,
}

fn select_lru_eviction_candidate(mut candidates: Vec<EvictionCandidate>) -> Option<String> {
    candidates.sort_by(|left, right| {
        left.last_used_unix_ms
            .cmp(&right.last_used_unix_ms)
            .then_with(|| left.id.cmp(&right.id))
    });
    candidates.into_iter().next().map(|candidate| candidate.id)
}

fn select_model_ttl_unload_candidates(
    mut candidates: Vec<ModelTtlCandidate>,
    now_unix_ms: u64,
    ttl: Duration,
) -> Vec<String> {
    let ttl_ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX);
    candidates.sort_by(|left, right| left.id.cmp(&right.id));
    candidates
        .into_iter()
        .filter(|candidate| {
            candidate.state == EngineRuntimeState::Loaded
                && candidate.active_requests == 0
                && candidate
                    .last_used_unix_ms
                    .is_some_and(|last_used| now_unix_ms.saturating_sub(last_used) >= ttl_ms)
        })
        .map(|candidate| candidate.id)
        .collect()
}

fn model_ttl_sweep_interval(ttl: Duration) -> Duration {
    let seconds = ttl.as_secs().saturating_div(4).clamp(5, 60);
    Duration::from_secs(seconds)
}

fn unix_time_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_millis();
    millis.min(u128::from(u64::MAX)) as u64
}

impl EngineSlotState {
    fn load_attempts(&self) -> u64 {
        match self {
            Self::Unloaded { load_attempts, .. }
            | Self::Loading { load_attempts, .. }
            | Self::Loaded { load_attempts, .. }
            | Self::Draining { load_attempts, .. }
            | Self::Failed { load_attempts, .. } => *load_attempts,
        }
    }

    fn runtime_snapshot(&self) -> EngineSlotRuntimeSnapshot {
        match self {
            Self::Unloaded {
                reason,
                last_error,
                changed_unix_ms,
                load_attempts,
            } => EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Unloaded,
                unload_reason: Some(*reason),
                last_error: last_error.clone(),
                changed_unix_ms: Some(*changed_unix_ms),
                load_started_unix_ms: None,
                loaded_unix_ms: None,
                last_used_unix_ms: None,
                failed_unix_ms: None,
                load_attempts: *load_attempts,
                request_count: 0,
            },
            Self::Loading {
                started_unix_ms,
                load_attempts,
            } => EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Loading,
                unload_reason: None,
                last_error: None,
                changed_unix_ms: None,
                load_started_unix_ms: Some(*started_unix_ms),
                loaded_unix_ms: None,
                last_used_unix_ms: None,
                failed_unix_ms: None,
                load_attempts: *load_attempts,
                request_count: 0,
            },
            Self::Loaded {
                loaded_unix_ms,
                last_used_unix_ms,
                load_attempts,
                request_count,
                ..
            } => EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Loaded,
                unload_reason: None,
                last_error: None,
                changed_unix_ms: None,
                load_started_unix_ms: None,
                loaded_unix_ms: Some(*loaded_unix_ms),
                last_used_unix_ms: Some(*last_used_unix_ms),
                failed_unix_ms: None,
                load_attempts: *load_attempts,
                request_count: *request_count,
            },
            Self::Draining {
                started_unix_ms,
                load_attempts,
                request_count,
            } => EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Draining,
                unload_reason: Some(EngineUnloadReason::Manual),
                last_error: None,
                changed_unix_ms: Some(*started_unix_ms),
                load_started_unix_ms: None,
                loaded_unix_ms: None,
                last_used_unix_ms: None,
                failed_unix_ms: None,
                load_attempts: *load_attempts,
                request_count: *request_count,
            },
            Self::Failed {
                last_error,
                failed_unix_ms,
                load_attempts,
            } => EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Failed,
                unload_reason: None,
                last_error: Some(last_error.clone()),
                changed_unix_ms: None,
                load_started_unix_ms: None,
                loaded_unix_ms: None,
                last_used_unix_ms: None,
                failed_unix_ms: Some(*failed_unix_ms),
                load_attempts: *load_attempts,
                request_count: 0,
            },
        }
    }
}

#[derive(Debug, Clone)]
struct EngineSlotRuntimeSnapshot {
    state: EngineRuntimeState,
    unload_reason: Option<EngineUnloadReason>,
    last_error: Option<String>,
    changed_unix_ms: Option<u64>,
    load_started_unix_ms: Option<u64>,
    loaded_unix_ms: Option<u64>,
    last_used_unix_ms: Option<u64>,
    failed_unix_ms: Option<u64>,
    load_attempts: u64,
    request_count: u64,
}

#[derive(Clone)]
enum EngineVariant {
    Qwen35(AppState<Qwen35Model>),
    Qwen35Moe(AppState<Qwen35MoeModel>),
    Qwen36Moe(AppState<Qwen36MoeModel>),
    Gemma4(AppState<Gemma4Model>),
    Glm4MoeLite(AppState<Glm4MoeLiteModel>),
    Llama(AppState<LlamaModel>),
    MiniCpmV46(AppState<MiniCpmV46Model>),
    DiffusionGemma(diffusion_gemma::DiffusionGemmaAppState),
}

struct EngineLease {
    engine: Arc<EngineVariant>,
    active_requests: Option<Arc<AtomicUsize>>,
}

impl std::fmt::Debug for EngineLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineLease").finish_non_exhaustive()
    }
}

impl EngineLease {
    fn new(engine: Arc<EngineVariant>, active_requests: Option<Arc<AtomicUsize>>) -> Self {
        Self {
            engine,
            active_requests,
        }
    }

    async fn openai_chat_completions(&self, req: openai::ChatRequest) -> Response {
        self.engine.openai_chat_completions(req).await
    }

    async fn anthropic_messages(&self, req: anthropic::MessagesRequest) -> Response {
        self.engine.anthropic_messages(req).await
    }
}

impl Drop for EngineLease {
    fn drop(&mut self) {
        if let Some(active_requests) = &self.active_requests {
            active_requests.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

impl EnginePoolState {
    pub async fn new(config: EnginePoolConfig, runtime: EnginePoolRuntimeConfig) -> Result<Self> {
        let registry = EngineRegistry::new(config.registry_manifest())?;
        config.validate_enabled_model_architectures()?;
        let mut slots = HashMap::with_capacity(config.models.len());
        for model in config.models {
            let slot = Arc::new(EngineSlot {
                model: model.clone(),
                runtime: runtime.clone(),
                active_requests: Arc::new(AtomicUsize::new(0)),
                state: Mutex::new(EngineSlotState::Unloaded {
                    reason: EngineUnloadReason::Startup,
                    last_error: None,
                    changed_unix_ms: unix_time_ms(),
                    load_attempts: 0,
                }),
                notify: Notify::new(),
            });
            slots.insert(model.id.clone(), slot);
        }
        let state = Self {
            inner: Arc::new(EnginePoolInner {
                registry: Mutex::new(registry),
                slots: Mutex::new(slots),
                runtime,
                capacity_policy: EnginePoolCapacityPolicy::EvictLruIdle,
                load_gate: Mutex::new(()),
            }),
        };
        state.preload().await?;
        Ok(state)
    }

    pub fn new_dynamic(
        runtime: EnginePoolRuntimeConfig,
        max_loaded_models: Option<usize>,
    ) -> Result<Self> {
        let registry = EngineRegistry::empty(max_loaded_models)?;
        Ok(Self {
            inner: Arc::new(EnginePoolInner {
                registry: Mutex::new(registry),
                slots: Mutex::new(HashMap::new()),
                runtime,
                capacity_policy: EnginePoolCapacityPolicy::Reject,
                load_gate: Mutex::new(()),
            }),
        })
    }

    pub(crate) fn start_model_ttl_sweeper(&self) {
        let Some(ttl) = self.inner.runtime.model_ttl else {
            return;
        };
        let pool = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(model_ttl_sweep_interval(ttl));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                let unloaded = pool.unload_expired_model_ttl_once(unix_time_ms()).await;
                if !unloaded.is_empty() {
                    tracing::info!(
                        "ironmlx EnginePool TTL unloaded idle models count={} ids={}",
                        unloaded.len(),
                        unloaded.join(",")
                    );
                }
            }
        });
    }

    pub(crate) async fn unload_expired_model_ttl_once(&self, now_unix_ms: u64) -> Vec<String> {
        let Some(ttl) = self.inner.runtime.model_ttl else {
            return Vec::new();
        };
        let slots_snapshot = self.inner.slots.lock().await.clone();
        let mut candidates = Vec::with_capacity(slots_snapshot.len());
        for (id, slot) in &slots_snapshot {
            if slot.model.load_policy == EngineLoadPolicy::Preload {
                continue;
            }
            candidates.push(slot.model_ttl_candidate(id).await);
        }
        let ids = select_model_ttl_unload_candidates(candidates, now_unix_ms, ttl);
        let mut unloaded = Vec::new();
        for id in ids {
            let Some(slot) = slots_snapshot.get(&id) else {
                continue;
            };
            if slot.unload_if_ttl_expired(now_unix_ms, ttl).await {
                unloaded.push(id);
            }
        }
        unloaded
    }

    async fn preload(&self) -> Result<()> {
        let models = self.inner.registry.lock().await.models().to_vec();
        for model in models {
            if model.load_policy != EngineLoadPolicy::Preload {
                continue;
            }
            let slot = {
                let slots = self.inner.slots.lock().await;
                slots
                    .get(&model.id)
                    .cloned()
                    .with_context(|| format!("missing engine slot `{}`", model.id))?
            };
            slot.ensure_loaded(&self.inner, EngineLoadTrigger::Control)
                .await
                .with_context(|| {
                    format!(
                        "preloading engine model `{}` from {}",
                        model.id,
                        model.path.display()
                    )
                })?;
        }
        Ok(())
    }

    async fn resolve_engine(&self, requested: Option<&str>) -> Result<(String, EngineLease)> {
        let model_id = self
            .inner
            .registry
            .lock()
            .await
            .resolve_model_id(requested)?
            .to_string();
        let slot = {
            let slots = self.inner.slots.lock().await;
            slots
                .get(&model_id)
                .cloned()
                .with_context(|| format!("missing engine slot `{model_id}`"))?
        };
        let engine = slot
            .ensure_loaded(&self.inner, EngineLoadTrigger::Request)
            .await
            .with_context(|| format!("loading engine model `{model_id}`"))?;
        Ok((model_id, engine))
    }

    pub(crate) async fn app_openai_chat_completions(
        &self,
        mut req: openai::ChatRequest,
    ) -> Result<Response> {
        let requested = req.model.as_deref().filter(|model| !model.is_empty());
        let (model_id, engine) = self.resolve_engine(requested).await?;
        if req.model.is_none() {
            req.model = Some(model_id);
        }
        Ok(engine.openai_chat_completions(req).await)
    }

    pub(crate) async fn app_anthropic_messages(
        &self,
        mut req: anthropic::MessagesRequest,
    ) -> Result<Response> {
        let requested = req.model.as_deref().filter(|model| !model.is_empty());
        let (model_id, engine) = self.resolve_engine(requested).await?;
        if req.model.is_none() {
            req.model = Some(model_id);
        }
        Ok(engine.anthropic_messages(req).await)
    }

    pub(crate) async fn load_model(&self, requested: &str) -> Result<EngineModelControlResult> {
        let model_id = self
            .inner
            .registry
            .lock()
            .await
            .resolve_model_id(Some(requested))?
            .to_string();
        let slot = {
            let slots = self.inner.slots.lock().await;
            slots
                .get(&model_id)
                .cloned()
                .with_context(|| format!("missing engine slot `{model_id}`"))?
        };
        slot.ensure_loaded(&self.inner, EngineLoadTrigger::Control)
            .await
            .with_context(|| format!("loading engine model `{model_id}`"))?;
        Ok(slot.control_result().await)
    }

    async fn unload_model(&self, requested: &str) -> Result<EngineModelControlResult> {
        let model_id = self
            .inner
            .registry
            .lock()
            .await
            .resolve_model_id(Some(requested))?
            .to_string();
        let slot = {
            let slots = self.inner.slots.lock().await;
            slots
                .get(&model_id)
                .cloned()
                .with_context(|| format!("missing engine slot `{model_id}`"))?
        };
        slot.unload().await?;
        Ok(slot.control_result().await)
    }

    pub(crate) async fn reload_dynamic_model(
        &self,
        model: EngineModelConfig,
        set_default: bool,
    ) -> Result<EngineModelControlResult> {
        validate_engine_model_config(&model)?;
        let model_id = model.id.clone();
        let new_slot = Arc::new(EngineSlot {
            model: model.clone(),
            runtime: self.inner.runtime.clone(),
            active_requests: Arc::new(AtomicUsize::new(0)),
            state: Mutex::new(EngineSlotState::Unloaded {
                reason: EngineUnloadReason::Startup,
                last_error: None,
                changed_unix_ms: unix_time_ms(),
                load_attempts: 0,
            }),
            notify: Notify::new(),
        });

        let old_slot = self.inner.slots.lock().await.remove(&model_id);
        let previous_default = self
            .inner
            .registry
            .lock()
            .await
            .default_model()
            .map(str::to_string);
        let was_previous_default = previous_default.as_deref() == Some(model_id.as_str());
        {
            let mut registry = self.inner.registry.lock().await;
            registry.remove_model(&model_id);
            registry.upsert_model(model.manifest_view(), set_default || was_previous_default)?;
        }
        self.inner
            .slots
            .lock()
            .await
            .insert(model_id.clone(), new_slot.clone());

        match new_slot
            .ensure_loaded(&self.inner, EngineLoadTrigger::Control)
            .await
        {
            Ok(_) => Ok(new_slot.control_result().await),
            Err(error) => {
                self.inner.slots.lock().await.remove(&model_id);
                {
                    let mut registry = self.inner.registry.lock().await;
                    registry.remove_model(&model_id);
                    if let Some(old_slot) = old_slot.as_ref() {
                        registry
                            .upsert_model(old_slot.model.manifest_view(), was_previous_default)?;
                    }
                    registry.restore_default_model(previous_default)?;
                }
                if let Some(old_slot) = old_slot {
                    self.inner.slots.lock().await.insert(model_id, old_slot);
                }
                Err(error)
            }
        }
    }

    pub(crate) async fn register_dynamic_model(
        &self,
        model: EngineModelConfig,
        set_default: bool,
    ) -> Result<EngineModelControlResult> {
        validate_engine_model_config(&model)?;
        let model_id = model.id.clone();
        let previous_default = self
            .inner
            .registry
            .lock()
            .await
            .default_model()
            .map(str::to_string);
        let existing_slot = self.inner.slots.lock().await.get(&model_id).cloned();
        let slot = match existing_slot {
            Some(existing) if existing.is_loaded_or_loading().await => existing,
            _ => {
                let slot = Arc::new(EngineSlot {
                    model: model.clone(),
                    runtime: self.inner.runtime.clone(),
                    active_requests: Arc::new(AtomicUsize::new(0)),
                    state: Mutex::new(EngineSlotState::Unloaded {
                        reason: EngineUnloadReason::Startup,
                        last_error: None,
                        changed_unix_ms: unix_time_ms(),
                        load_attempts: 0,
                    }),
                    notify: Notify::new(),
                });
                self.inner
                    .slots
                    .lock()
                    .await
                    .insert(model_id.clone(), slot.clone());
                slot
            }
        };
        {
            let mut registry = self.inner.registry.lock().await;
            registry.upsert_model(model.manifest_view(), set_default)?;
            if previous_default.is_none() && !set_default {
                registry.restore_default_model(None)?;
            }
        }
        Ok(slot.control_result().await)
    }

    pub(crate) async fn unload_dynamic_model(
        &self,
        requested: &str,
    ) -> Result<EngineModelControlResult> {
        let model_id = self
            .inner
            .registry
            .lock()
            .await
            .resolve_model_id(Some(requested))?
            .to_string();
        let slot = {
            let slots = self.inner.slots.lock().await;
            slots
                .get(&model_id)
                .cloned()
                .with_context(|| format!("missing engine slot `{model_id}`"))?
        };
        slot.unload().await?;
        Ok(slot.control_result().await)
    }

    pub(crate) async fn set_default_model(&self, requested: &str) -> Result<()> {
        self.inner
            .registry
            .lock()
            .await
            .set_default_model(requested)?;
        Ok(())
    }

    pub(crate) async fn is_model_registered(&self, id: &str) -> bool {
        self.inner.registry.lock().await.model(id).is_some()
    }

    pub(crate) async fn is_model_loaded(&self, id: &str) -> bool {
        let slot = self.inner.slots.lock().await.get(id).cloned();
        match slot {
            Some(slot) => matches!(&*slot.state.lock().await, EngineSlotState::Loaded { .. }),
            None => false,
        }
    }

    pub(crate) async fn pending_requests(&self, id: &str) -> Option<usize> {
        let slot = self.inner.slots.lock().await.get(id).cloned();
        match slot {
            Some(slot) => slot.pending_requests().await,
            None => None,
        }
    }

    pub(crate) async fn loaded_model_infos(&self) -> Vec<EngineLoadedModelInfo> {
        let default_model = self
            .inner
            .registry
            .lock()
            .await
            .default_model()
            .map(str::to_string);
        let slots = self.inner.slots.lock().await.clone();
        let mut models = Vec::new();
        for (id, slot) in slots {
            let loaded = {
                let state = slot.state.lock().await;
                match &*state {
                    EngineSlotState::Loaded { engine, .. } => Some(engine.clone()),
                    _ => None,
                }
            };
            let Some(engine) = loaded else {
                continue;
            };
            let health = engine.loaded_health();
            models.push(EngineLoadedModelInfo {
                id: id.clone(),
                path: slot.model.path.to_string_lossy().into_owned(),
                architecture: engine.architecture().to_string(),
                is_default: default_model.as_deref() == Some(id.as_str()),
                max_position_embeddings: health.max_position_embeddings(),
            });
        }
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    pub(crate) async fn loaded_causal_health_snapshots(&self) -> Vec<health::HealthSnapshot> {
        let slots = self.inner.slots.lock().await.clone();
        let mut snapshots = Vec::new();
        for slot in slots.values() {
            let loaded = {
                let state = slot.state.lock().await;
                match &*state {
                    EngineSlotState::Loaded { engine, .. } => Some(engine.clone()),
                    _ => None,
                }
            };
            let Some(engine) = loaded else {
                continue;
            };
            if let LoadedEngineHealth::Causal(snapshot) = engine.loaded_health() {
                snapshots.push(*snapshot);
            }
        }
        snapshots
    }

    async fn model_list(&self) -> OpenAiModelList {
        let mut data = Vec::new();
        let models = self.inner.registry.lock().await.servable_models_owned();
        for model in models {
            let slot = {
                let slots = self.inner.slots.lock().await;
                slots.get(&model.id).cloned()
            };
            let snapshot = match slot {
                Some(slot) => slot.runtime_snapshot().await,
                None => EngineSlotRuntimeSnapshot {
                    state: EngineRuntimeState::Missing,
                    unload_reason: None,
                    last_error: Some("engine slot missing".to_string()),
                    changed_unix_ms: None,
                    load_started_unix_ms: None,
                    loaded_unix_ms: None,
                    last_used_unix_ms: None,
                    failed_unix_ms: None,
                    load_attempts: 0,
                    request_count: 0,
                },
            };
            data.push(OpenAiModelInfo {
                id: model.id.clone(),
                object: "model",
                created: 0,
                owned_by: "ironmlx",
                load_policy: model.load_policy,
                state: snapshot.state,
                unload_reason: snapshot.unload_reason,
                last_error: snapshot.last_error,
                changed_unix_ms: snapshot.changed_unix_ms,
                load_started_unix_ms: snapshot.load_started_unix_ms,
                loaded_unix_ms: snapshot.loaded_unix_ms,
                last_used_unix_ms: snapshot.last_used_unix_ms,
                failed_unix_ms: snapshot.failed_unix_ms,
                load_attempts: snapshot.load_attempts,
                request_count: snapshot.request_count,
            });
        }
        OpenAiModelList {
            object: "list",
            data,
        }
    }

    async fn health_snapshot(&self) -> EnginePoolHealth {
        let mut models = Vec::new();
        let registry = self.inner.registry.lock().await;
        let default_model = registry.default_model().map(str::to_string);
        let max_loaded_models = registry.max_loaded_models();
        let registry_models = registry.models().to_vec();
        drop(registry);
        for model in registry_models {
            let slot = {
                let slots = self.inner.slots.lock().await;
                slots.get(&model.id).cloned()
            };
            let Some(slot) = slot else {
                models.push(EngineModelHealth {
                    id: model.id.clone(),
                    load_policy: model.load_policy,
                    state: EngineRuntimeState::Missing,
                    unload_reason: None,
                    last_error: Some("engine slot missing".to_string()),
                    loaded_unix_ms: None,
                    last_used_unix_ms: None,
                    load_started_unix_ms: None,
                    failed_unix_ms: None,
                    load_attempts: 0,
                    request_count: 0,
                    loaded: None,
                });
                continue;
            };
            models.push(slot.health_snapshot().await);
        }
        let loaded_models = models
            .iter()
            .filter(|model| model.state == EngineRuntimeState::Loaded)
            .count();
        let failed_models = models
            .iter()
            .filter(|model| model.state == EngineRuntimeState::Failed)
            .count();
        let status = if failed_models > 0 {
            "degraded"
        } else {
            "healthy"
        };
        EnginePoolHealth {
            status,
            mode: "engine_pool",
            default_model,
            max_loaded_models,
            loaded_models,
            models,
            version: env!("CARGO_PKG_VERSION"),
        }
    }
}

impl EnginePoolInner {
    async fn ensure_capacity_for(&self, target_id: &str) -> Result<()> {
        let Some(max_loaded_models) = self.registry.lock().await.max_loaded_models() else {
            return Ok(());
        };
        let loaded_count = self.loaded_count().await;
        match decide_engine_pool_capacity(self.capacity_policy, max_loaded_models, loaded_count) {
            EnginePoolCapacityDecision::Continue => return Ok(()),
            EnginePoolCapacityDecision::Reject => {
                bail!(
                    "engine pool capacity reached: max_loaded_models={max_loaded_models}, unload an existing model before loading `{target_id}`"
                );
            }
            EnginePoolCapacityDecision::TryEvictLruIdle => {}
        }
        if self.evict_lru_idle_engine(target_id).await? {
            return Ok(());
        }
        bail!(
            "engine pool capacity reached: max_loaded_models={max_loaded_models}, no idle lazy engine can be evicted"
        );
    }

    async fn loaded_count(&self) -> usize {
        let mut loaded = 0;
        let slots = self.slots.lock().await;
        for slot in slots.values() {
            if matches!(
                &*slot.state.lock().await,
                EngineSlotState::Loaded { .. } | EngineSlotState::Draining { .. }
            ) {
                loaded += 1;
            }
        }
        loaded
    }

    async fn check_memory_limits_after_load(
        &self,
        target_id: &str,
        engine: &EngineVariant,
    ) -> Result<()> {
        let loaded_model_bytes = self
            .loaded_model_weight_bytes_excluding(target_id)
            .await
            .saturating_add(engine.model_weight_bytes());
        self.runtime
            .memory_limits
            .check_model_memory_limit(target_id, loaded_model_bytes)?;

        let mlx_active_bytes = mlx::memory::snapshot().active_bytes;
        self.runtime
            .memory_limits
            .check_total_memory_limit(target_id, mlx_active_bytes)?;
        Ok(())
    }

    async fn loaded_model_weight_bytes_excluding(&self, exclude_id: &str) -> usize {
        let slots_snapshot = self.slots.lock().await.clone();
        let mut total = 0usize;
        for (id, slot) in slots_snapshot {
            if id == exclude_id {
                continue;
            }
            let engine = {
                let state = slot.state.lock().await;
                match &*state {
                    EngineSlotState::Loaded { engine, .. } => Some(engine.clone()),
                    EngineSlotState::Draining { .. } => None,
                    _ => None,
                }
            };
            if let Some(engine) = engine {
                total = total.saturating_add(engine.model_weight_bytes());
            }
        }
        total
    }

    async fn evict_lru_idle_engine(&self, target_id: &str) -> Result<bool> {
        let mut candidates = Vec::new();
        let slots_snapshot = self.slots.lock().await.clone();
        for (id, slot) in &slots_snapshot {
            if id == target_id || slot.model.load_policy == EngineLoadPolicy::Preload {
                continue;
            }
            let guard = slot.state.lock().await;
            let EngineSlotState::Loaded {
                engine,
                last_used_unix_ms,
                ..
            } = &*guard
            else {
                continue;
            };
            if slot.active_requests.load(Ordering::SeqCst) == 0 && Arc::strong_count(engine) == 1 {
                candidates.push(EvictionCandidate {
                    id: id.clone(),
                    last_used_unix_ms: *last_used_unix_ms,
                });
            }
        }

        let Some(evict_id) = select_lru_eviction_candidate(candidates) else {
            return Ok(false);
        };
        let Some(slot) = self.slots.lock().await.get(&evict_id).cloned() else {
            return Ok(false);
        };
        let mut guard = slot.state.lock().await;
        let load_attempts = guard.load_attempts();
        let EngineSlotState::Loaded { engine, .. } = &*guard else {
            return Ok(false);
        };
        if slot.active_requests.load(Ordering::SeqCst) != 0 || Arc::strong_count(engine) != 1 {
            return Ok(false);
        }
        *guard = EngineSlotState::Unloaded {
            reason: EngineUnloadReason::Evicted,
            last_error: None,
            changed_unix_ms: unix_time_ms(),
            load_attempts,
        };
        tracing::info!(
            "ironmlx EnginePool evicted idle model id={} for target={}",
            evict_id,
            target_id
        );
        slot.notify.notify_waiters();
        Ok(true)
    }
}

impl EngineSlot {
    async fn is_loaded_or_loading(&self) -> bool {
        matches!(
            &*self.state.lock().await,
            EngineSlotState::Loaded { .. }
                | EngineSlotState::Loading { .. }
                | EngineSlotState::Draining { .. }
        )
    }

    async fn ensure_loaded(
        &self,
        pool: &EnginePoolInner,
        trigger: EngineLoadTrigger,
    ) -> Result<EngineLease> {
        if self.model.load_policy == EngineLoadPolicy::Disabled {
            bail!("engine model `{}` is disabled", self.model.id);
        }

        loop {
            let wait_for_load = {
                let mut state = self.state.lock().await;
                match &mut *state {
                    EngineSlotState::Loaded {
                        engine,
                        last_used_unix_ms,
                        request_count,
                        ..
                    } => {
                        let active_requests = if trigger == EngineLoadTrigger::Request {
                            *last_used_unix_ms = unix_time_ms();
                            *request_count = request_count.saturating_add(1);
                            self.active_requests.fetch_add(1, Ordering::SeqCst);
                            Some(self.active_requests.clone())
                        } else {
                            None
                        };
                        return Ok(EngineLease::new(engine.clone(), active_requests));
                    }
                    EngineSlotState::Loading { .. } => Some(self.notify.notified()),
                    EngineSlotState::Draining { .. } => {
                        if self.active_requests.load(Ordering::SeqCst) == 0 {
                            let load_attempts = state.load_attempts();
                            *state = EngineSlotState::Unloaded {
                                reason: EngineUnloadReason::Manual,
                                last_error: None,
                                changed_unix_ms: unix_time_ms(),
                                load_attempts,
                            };
                            None
                        } else {
                            bail!("engine model `{}` is currently unloading", self.model.id);
                        }
                    }
                    EngineSlotState::Failed { last_error, .. }
                        if trigger == EngineLoadTrigger::Request =>
                    {
                        bail!(
                            "engine model `{}` is in failed state: {last_error}",
                            self.model.id
                        );
                    }
                    EngineSlotState::Unloaded { .. } | EngineSlotState::Failed { .. } => None,
                }
            };
            if let Some(wait_for_load) = wait_for_load {
                wait_for_load.await;
                continue;
            }

            let _load_gate = pool.load_gate.lock().await;
            {
                let mut state = self.state.lock().await;
                match &mut *state {
                    EngineSlotState::Loaded {
                        engine,
                        last_used_unix_ms,
                        request_count,
                        ..
                    } => {
                        let active_requests = if trigger == EngineLoadTrigger::Request {
                            *last_used_unix_ms = unix_time_ms();
                            *request_count = request_count.saturating_add(1);
                            self.active_requests.fetch_add(1, Ordering::SeqCst);
                            Some(self.active_requests.clone())
                        } else {
                            None
                        };
                        return Ok(EngineLease::new(engine.clone(), active_requests));
                    }
                    EngineSlotState::Loading { .. } => {
                        continue;
                    }
                    EngineSlotState::Draining { .. } => {
                        if self.active_requests.load(Ordering::SeqCst) == 0 {
                            let load_attempts = state.load_attempts();
                            *state = EngineSlotState::Unloaded {
                                reason: EngineUnloadReason::Manual,
                                last_error: None,
                                changed_unix_ms: unix_time_ms(),
                                load_attempts,
                            };
                        } else {
                            bail!("engine model `{}` is currently unloading", self.model.id);
                        }
                    }
                    EngineSlotState::Failed { last_error, .. }
                        if trigger == EngineLoadTrigger::Request =>
                    {
                        bail!(
                            "engine model `{}` is in failed state: {last_error}",
                            self.model.id
                        );
                    }
                    EngineSlotState::Unloaded { .. } | EngineSlotState::Failed { .. } => {}
                }
            }

            if let Err(error) = pool.ensure_capacity_for(&self.model.id).await {
                let mut state = self.state.lock().await;
                let load_attempts = state.load_attempts();
                *state = EngineSlotState::Unloaded {
                    reason: EngineUnloadReason::CapacityRejected,
                    last_error: Some(format!("{error:#}")),
                    changed_unix_ms: unix_time_ms(),
                    load_attempts,
                };
                self.notify.notify_waiters();
                return Err(error);
            }

            let load_attempts = {
                let mut state = self.state.lock().await;
                let load_attempts = state.load_attempts().saturating_add(1);
                *state = EngineSlotState::Loading {
                    started_unix_ms: unix_time_ms(),
                    load_attempts,
                };
                self.notify.notify_waiters();
                load_attempts
            };

            let result = load_engine_variant(&self.model, &self.runtime).await;
            match result {
                Ok(engine) => {
                    if let Err(error) = pool
                        .check_memory_limits_after_load(&self.model.id, &engine)
                        .await
                    {
                        let message = format!("{error:#}");
                        let mut state = self.state.lock().await;
                        *state = EngineSlotState::Failed {
                            last_error: message,
                            failed_unix_ms: unix_time_ms(),
                            load_attempts,
                        };
                        self.notify.notify_waiters();
                        return Err(error);
                    }
                    let engine = Arc::new(engine);
                    let now = unix_time_ms();
                    let mut state = self.state.lock().await;
                    let active_requests = if trigger == EngineLoadTrigger::Request {
                        self.active_requests.fetch_add(1, Ordering::SeqCst);
                        Some(self.active_requests.clone())
                    } else {
                        None
                    };
                    *state = EngineSlotState::Loaded {
                        engine: engine.clone(),
                        loaded_unix_ms: now,
                        last_used_unix_ms: now,
                        load_attempts,
                        request_count: if trigger == EngineLoadTrigger::Request {
                            1
                        } else {
                            0
                        },
                    };
                    self.notify.notify_waiters();
                    return Ok(EngineLease::new(engine, active_requests));
                }
                Err(error) => {
                    let message = format!("{error:#}");
                    let mut state = self.state.lock().await;
                    *state = EngineSlotState::Failed {
                        last_error: message,
                        failed_unix_ms: unix_time_ms(),
                        load_attempts,
                    };
                    self.notify.notify_waiters();
                    return Err(error);
                }
            }
        }
    }

    async fn model_ttl_candidate(&self, id: &str) -> ModelTtlCandidate {
        let snapshot = self.runtime_snapshot().await;
        ModelTtlCandidate {
            id: id.to_string(),
            state: snapshot.state,
            active_requests: self.active_requests.load(Ordering::SeqCst),
            last_used_unix_ms: snapshot.last_used_unix_ms,
        }
    }

    async fn unload_if_ttl_expired(&self, now_unix_ms: u64, ttl: Duration) -> bool {
        if self.model.load_policy == EngineLoadPolicy::Preload
            || self.active_requests.load(Ordering::SeqCst) > 0
        {
            return false;
        }
        let ttl_ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX);
        let mut state = self.state.lock().await;
        let load_attempts = state.load_attempts();
        let EngineSlotState::Loaded {
            last_used_unix_ms, ..
        } = &*state
        else {
            return false;
        };
        if now_unix_ms.saturating_sub(*last_used_unix_ms) < ttl_ms
            || self.active_requests.load(Ordering::SeqCst) > 0
        {
            return false;
        }
        *state = EngineSlotState::Unloaded {
            reason: EngineUnloadReason::Ttl,
            last_error: None,
            changed_unix_ms: now_unix_ms,
            load_attempts,
        };
        self.notify.notify_waiters();
        true
    }

    async fn unload(self: &Arc<Self>) -> Result<()> {
        if self.model.load_policy == EngineLoadPolicy::Preload {
            bail!(
                "engine model `{}` uses preload policy and cannot be unloaded",
                self.model.id
            );
        }
        if self.model.load_policy == EngineLoadPolicy::Disabled {
            bail!("engine model `{}` is disabled", self.model.id);
        }

        let mut state = self.state.lock().await;
        let load_attempts = state.load_attempts();
        match &*state {
            EngineSlotState::Loaded { request_count, .. } => {
                if self.active_requests.load(Ordering::SeqCst) > 0 {
                    *state = EngineSlotState::Draining {
                        started_unix_ms: unix_time_ms(),
                        load_attempts,
                        request_count: *request_count,
                    };
                    self.notify.notify_waiters();
                    self.schedule_finish_draining();
                    return Ok(());
                }
            }
            EngineSlotState::Loading { .. } => {
                bail!("engine model `{}` is currently loading", self.model.id);
            }
            EngineSlotState::Draining { .. } => {
                if self.active_requests.load(Ordering::SeqCst) > 0 {
                    self.schedule_finish_draining();
                    return Ok(());
                }
            }
            EngineSlotState::Unloaded { .. } | EngineSlotState::Failed { .. } => {}
        }
        *state = EngineSlotState::Unloaded {
            reason: EngineUnloadReason::Manual,
            last_error: None,
            changed_unix_ms: unix_time_ms(),
            load_attempts,
        };
        self.notify.notify_waiters();
        Ok(())
    }

    fn schedule_finish_draining(self: &Arc<Self>) {
        let slot = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                if slot.active_requests.load(Ordering::SeqCst) == 0 {
                    let mut state = slot.state.lock().await;
                    if matches!(&*state, EngineSlotState::Draining { .. }) {
                        let load_attempts = state.load_attempts();
                        *state = EngineSlotState::Unloaded {
                            reason: EngineUnloadReason::Manual,
                            last_error: None,
                            changed_unix_ms: unix_time_ms(),
                            load_attempts,
                        };
                        slot.notify.notify_waiters();
                    }
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });
    }

    async fn runtime_snapshot(&self) -> EngineSlotRuntimeSnapshot {
        if self.model.load_policy == EngineLoadPolicy::Disabled {
            return EngineSlotRuntimeSnapshot {
                state: EngineRuntimeState::Disabled,
                unload_reason: None,
                last_error: None,
                changed_unix_ms: None,
                load_started_unix_ms: None,
                loaded_unix_ms: None,
                last_used_unix_ms: None,
                failed_unix_ms: None,
                load_attempts: 0,
                request_count: 0,
            };
        }
        self.state.lock().await.runtime_snapshot()
    }

    async fn pending_requests(&self) -> Option<usize> {
        let engine = {
            let state = self.state.lock().await;
            match &*state {
                EngineSlotState::Loaded { engine, .. } => Some(engine.clone()),
                _ => None,
            }
        };
        engine.map(|engine| engine.pending_requests())
    }

    async fn control_result(&self) -> EngineModelControlResult {
        let snapshot = self.runtime_snapshot().await;
        EngineModelControlResult {
            id: self.model.id.clone(),
            load_policy: self.model.load_policy,
            state: snapshot.state,
            unload_reason: snapshot.unload_reason,
            last_error: snapshot.last_error,
            load_attempts: snapshot.load_attempts,
            request_count: snapshot.request_count,
        }
    }

    async fn health_snapshot(&self) -> EngineModelHealth {
        let snapshot = self.runtime_snapshot().await;
        let loaded = if snapshot.state == EngineRuntimeState::Loaded {
            let guard = self.state.lock().await;
            match &*guard {
                EngineSlotState::Loaded { engine, .. } => Some(engine.loaded_health()),
                _ => None,
            }
        } else {
            None
        };
        EngineModelHealth {
            id: self.model.id.clone(),
            load_policy: self.model.load_policy,
            state: snapshot.state,
            unload_reason: snapshot.unload_reason,
            last_error: snapshot.last_error,
            loaded_unix_ms: snapshot.loaded_unix_ms,
            last_used_unix_ms: snapshot.last_used_unix_ms,
            load_started_unix_ms: snapshot.load_started_unix_ms,
            failed_unix_ms: snapshot.failed_unix_ms,
            load_attempts: snapshot.load_attempts,
            request_count: snapshot.request_count,
            loaded,
        }
    }
}

impl EngineVariant {
    async fn openai_chat_completions(&self, req: openai::ChatRequest) -> Response {
        match self {
            Self::Qwen35(state) => openai::chat_completions(State(state.clone()), Json(req)).await,
            Self::Qwen35Moe(state) => {
                openai::chat_completions(State(state.clone()), Json(req)).await
            }
            Self::Qwen36Moe(state) => {
                openai::chat_completions(State(state.clone()), Json(req)).await
            }
            Self::Gemma4(state) => openai::chat_completions(State(state.clone()), Json(req)).await,
            Self::Glm4MoeLite(state) => {
                openai::chat_completions(State(state.clone()), Json(req)).await
            }
            Self::Llama(state) => openai::chat_completions(State(state.clone()), Json(req)).await,
            Self::MiniCpmV46(state) => {
                openai::chat_completions(State(state.clone()), Json(req)).await
            }
            Self::DiffusionGemma(state) => {
                diffusion_gemma::openai_chat_completions(State(state.clone()), Json(req)).await
            }
        }
    }

    async fn anthropic_messages(&self, req: anthropic::MessagesRequest) -> Response {
        match self {
            Self::Qwen35(state) => anthropic::messages(State(state.clone()), Json(req)).await,
            Self::Qwen35Moe(state) => anthropic::messages(State(state.clone()), Json(req)).await,
            Self::Qwen36Moe(state) => anthropic::messages(State(state.clone()), Json(req)).await,
            Self::Gemma4(state) => anthropic::messages(State(state.clone()), Json(req)).await,
            Self::Glm4MoeLite(state) => anthropic::messages(State(state.clone()), Json(req)).await,
            Self::Llama(state) => anthropic::messages(State(state.clone()), Json(req)).await,
            Self::MiniCpmV46(state) => anthropic::messages(State(state.clone()), Json(req)).await,
            Self::DiffusionGemma(state) => {
                diffusion_gemma::anthropic_messages(State(state.clone()), Json(req)).await
            }
        }
    }

    fn loaded_health(&self) -> LoadedEngineHealth {
        match self {
            Self::Qwen35(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::Qwen35Moe(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::Qwen36Moe(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::Gemma4(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::Glm4MoeLite(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::Llama(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::MiniCpmV46(state) => {
                LoadedEngineHealth::Causal(Box::new(state.health_collector.snapshot()))
            }
            Self::DiffusionGemma(state) => {
                let stats = state.lane.stats();
                LoadedEngineHealth::DiffusionGemma {
                    scheduler: "serial_block_diffusion",
                    active_requests: stats.active_requests,
                    queued_requests: stats.queued_requests,
                    queue_capacity: stats.queue_capacity,
                }
            }
        }
    }

    fn architecture(&self) -> &'static str {
        match self {
            Self::Qwen35(_) => "qwen3_5",
            Self::Qwen35Moe(_) => "qwen3_5_moe",
            Self::Qwen36Moe(_) => "qwen3_6_moe",
            Self::Gemma4(_) => "gemma4",
            Self::Glm4MoeLite(_) => "glm4_moe_lite",
            Self::Llama(_) => "llama",
            Self::MiniCpmV46(_) => "minicpmv4_6",
            Self::DiffusionGemma(_) => "diffusion_gemma",
        }
    }

    fn model_weight_bytes(&self) -> usize {
        match self {
            Self::Qwen35(state) => state.model_weight_bytes,
            Self::Qwen35Moe(state) => state.model_weight_bytes,
            Self::Qwen36Moe(state) => state.model_weight_bytes,
            Self::Gemma4(state) => state.model_weight_bytes,
            Self::Glm4MoeLite(state) => state.model_weight_bytes,
            Self::Llama(state) => state.model_weight_bytes,
            Self::MiniCpmV46(state) => state.model_weight_bytes,
            Self::DiffusionGemma(state) => state.model_weight_bytes,
        }
    }

    fn pending_requests(&self) -> usize {
        match self {
            Self::Qwen35(state) => causal_pending_requests(state),
            Self::Qwen35Moe(state) => causal_pending_requests(state),
            Self::Qwen36Moe(state) => causal_pending_requests(state),
            Self::Gemma4(state) => causal_pending_requests(state),
            Self::Glm4MoeLite(state) => causal_pending_requests(state),
            Self::Llama(state) => causal_pending_requests(state),
            Self::MiniCpmV46(state) => causal_pending_requests(state),
            Self::DiffusionGemma(state) => {
                let stats = state.lane.stats();
                stats.active_requests.saturating_add(stats.queued_requests)
            }
        }
    }
}

fn causal_pending_requests<M>(state: &AppState<M>) -> usize
where
    M: Model + crate::core::scheduler::DenseVlMethods + Send + 'static,
{
    let snapshot = state.health_collector.snapshot();
    let model_locked = usize::from(state.model.try_lock().is_err());
    snapshot
        .scheduler
        .b_active
        .saturating_add(snapshot.scheduler.b_queued)
        .saturating_add(model_locked)
}

#[derive(Debug)]
struct ResolvedEngineMtpConfig {
    model_dir: PathBuf,
    draft_tokens: usize,
}

fn resolve_engine_mtp_config(
    model: &EngineModelConfig,
    architecture: ModelArchitecture,
    raw_config: &serde_json::Value,
) -> Result<Option<ResolvedEngineMtpConfig>> {
    let Some(settings) = model.mtp.as_ref() else {
        return Ok(None);
    };
    match architecture {
        ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe => {}
        _ => bail!(
            "engine model `{}` configures MTP for a non-Qwen architecture",
            model.id
        ),
    }
    if !settings.model_dir.exists() {
        bail!(
            "engine model `{}` mtp_model_dir must point to a local directory (got '{}')",
            model.id,
            settings.model_dir.display()
        );
    }
    let draft_tokens = crate::core::speculative::resolve_mtp_draft_tokens(
        raw_config,
        settings
            .draft_tokens
            .map(MtpDraftTokensArg::Explicit)
            .unwrap_or(MtpDraftTokensArg::Omitted),
    );
    MtpSpeculativeConfig::new(draft_tokens, Sampler::greedy())?;
    Ok(Some(ResolvedEngineMtpConfig {
        model_dir: settings.model_dir.clone(),
        draft_tokens,
    }))
}

async fn load_engine_variant(
    model: &EngineModelConfig,
    runtime: &EnginePoolRuntimeConfig,
) -> Result<EngineVariant> {
    let loader = Loader::open_multimodal(&model.path)
        .with_context(|| format!("Loader::open_multimodal {}", model.path.display()))?;
    let base_model_weight_bytes = loader.loaded_tensor_bytes();
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let model_type = loader
        .config_raw_value()
        .get("model_type")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("config.json missing model_type"))?;
    let architecture = ModelArchitecture::from_model_type(model_type)?;
    let mtp_config = resolve_engine_mtp_config(model, architecture, loader.config_raw_value())?;
    let scheduler_config = model.scheduler_runtime_profile.config;
    let paged_prefix_cache = runtime.paged_prefix_cache_config(&model.id, scheduler_config)?;
    let prefix_lru_cache = runtime.prefix_lru_cache_config(paged_prefix_cache.as_ref())?;

    match architecture {
        ModelArchitecture::Qwen35Dense => {
            let model_impl =
                Qwen35Model::from_loader(&loader).context("Qwen35Model::from_loader")?;
            build_qwen35_engine(
                model_impl,
                tokenizer,
                model,
                runtime,
                base_model_weight_bytes,
                paged_prefix_cache,
                prefix_lru_cache,
                mtp_config,
                None,
            )
            .await
        }
        ModelArchitecture::Qwen35Moe => {
            if crate::models::is_qwen36_moe_config(loader.config_raw_value()) {
                let model_impl =
                    Qwen36MoeModel::from_loader(&loader).context("Qwen36MoeModel::from_loader")?;
                build_qwen36_moe_engine(
                    model_impl,
                    tokenizer,
                    model,
                    runtime,
                    base_model_weight_bytes,
                    paged_prefix_cache,
                    prefix_lru_cache,
                    mtp_config,
                    None,
                )
                .await
            } else {
                let model_impl =
                    Qwen35MoeModel::from_loader(&loader).context("Qwen35MoeModel::from_loader")?;
                build_qwen35_moe_engine(
                    model_impl,
                    tokenizer,
                    model,
                    runtime,
                    base_model_weight_bytes,
                    paged_prefix_cache,
                    prefix_lru_cache,
                    mtp_config,
                    None,
                )
                .await
            }
        }
        ModelArchitecture::Gemma4 => {
            let cfg = Gemma4Config::from_loader(&loader).context("Gemma4Config::from_loader")?;
            let vision_input = cfg
                .vision_config
                .map(|vision_config| VisionInputConfig::Gemma4 { vision_config });
            let model_impl =
                Gemma4Model::from_loader(&loader).context("Gemma4Model::from_loader")?;
            let state = build_plain_causal_state(
                model_impl,
                tokenizer,
                model,
                runtime,
                base_model_weight_bytes,
                paged_prefix_cache,
                prefix_lru_cache,
                vision_input,
            )
            .await?;
            Ok(EngineVariant::Gemma4(state))
        }
        ModelArchitecture::Glm4MoeLite => {
            let model_impl =
                Glm4MoeLiteModel::from_loader(&loader).context("Glm4MoeLiteModel::from_loader")?;
            let state = build_plain_causal_state(
                model_impl,
                tokenizer,
                model,
                runtime,
                base_model_weight_bytes,
                paged_prefix_cache,
                prefix_lru_cache,
                None,
            )
            .await?;
            Ok(EngineVariant::Glm4MoeLite(state))
        }
        ModelArchitecture::Llama => {
            let model_impl = LlamaModel::from_loader(&loader).context("LlamaModel::from_loader")?;
            let state = build_plain_causal_state(
                model_impl,
                tokenizer,
                model,
                runtime,
                base_model_weight_bytes,
                paged_prefix_cache,
                prefix_lru_cache,
                None,
            )
            .await?;
            Ok(EngineVariant::Llama(state))
        }
        ModelArchitecture::MiniCpmV46 => {
            let model_impl = crate::models::minicpmv4_6::model_from_loader(&loader)
                .context("minicpmv4_6::model_from_loader")?;
            let state = build_plain_causal_state(
                model_impl,
                tokenizer,
                model,
                runtime,
                base_model_weight_bytes,
                paged_prefix_cache,
                prefix_lru_cache,
                Some(VisionInputConfig::MiniCpmV46 {
                    spatial_merge_size: 4,
                }),
            )
            .await?;
            Ok(EngineVariant::MiniCpmV46(state))
        }
        ModelArchitecture::DiffusionGemma => {
            if model.mtp.is_some() {
                bail!(
                    "engine model `{}` configures MTP for DiffusionGemma",
                    model.id
                );
            }
            let cfg = DiffusionGemmaConfig::from_loader(&loader)
                .context("DiffusionGemmaConfig::from_loader")?;
            let vision_config = cfg
                .vision_config
                .clone()
                .ok_or_else(|| anyhow::anyhow!("DiffusionGemma config has no vision_config"))?;
            let image_token_id = cfg.image_token_id;
            let generation_config = DiffusionGemmaGenerationConfig::from_loader(&loader)
                .context("DiffusionGemmaGenerationConfig::from_loader")?;
            let model_impl = DiffusionGemmaModel::from_loader(&loader)
                .context("DiffusionGemmaModel::from_loader")?;
            let state = diffusion_gemma::build_diffusion_gemma_app_state(
                model_impl,
                tokenizer,
                generation_config,
                model.id.clone(),
                base_model_weight_bytes,
                VisionInputConfig::DiffusionGemma {
                    vision_config,
                    image_token_id,
                },
            );
            Ok(EngineVariant::DiffusionGemma(state))
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_qwen35_engine(
    model_impl: Qwen35Model,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EnginePoolRuntimeConfig,
    loaded_model_weight_bytes: usize,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    mtp_config: Option<ResolvedEngineMtpConfig>,
    vision_input: Option<VisionInputConfig>,
) -> Result<EngineVariant> {
    if let Some(mtp_config) = mtp_config {
        let state = build_mtp_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            loaded_model_weight_bytes,
            paged_prefix_cache,
            prefix_lru_cache,
            mtp_config,
            vision_input,
        )
        .await?;
        Ok(EngineVariant::Qwen35(state))
    } else {
        let state = build_plain_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            loaded_model_weight_bytes,
            paged_prefix_cache,
            prefix_lru_cache,
            vision_input,
        )
        .await?;
        Ok(EngineVariant::Qwen35(state))
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_qwen35_moe_engine(
    model_impl: Qwen35MoeModel,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EnginePoolRuntimeConfig,
    loaded_model_weight_bytes: usize,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    mtp_config: Option<ResolvedEngineMtpConfig>,
    vision_input: Option<VisionInputConfig>,
) -> Result<EngineVariant> {
    if let Some(mtp_config) = mtp_config {
        let state = build_mtp_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            loaded_model_weight_bytes,
            paged_prefix_cache,
            prefix_lru_cache,
            mtp_config,
            vision_input,
        )
        .await?;
        Ok(EngineVariant::Qwen35Moe(state))
    } else {
        let state = build_plain_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            loaded_model_weight_bytes,
            paged_prefix_cache,
            prefix_lru_cache,
            vision_input,
        )
        .await?;
        Ok(EngineVariant::Qwen35Moe(state))
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_qwen36_moe_engine(
    model_impl: Qwen36MoeModel,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EnginePoolRuntimeConfig,
    loaded_model_weight_bytes: usize,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    mtp_config: Option<ResolvedEngineMtpConfig>,
    vision_input: Option<VisionInputConfig>,
) -> Result<EngineVariant> {
    if let Some(mtp_config) = mtp_config {
        let state = build_mtp_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            loaded_model_weight_bytes,
            paged_prefix_cache,
            prefix_lru_cache,
            mtp_config,
            vision_input,
        )
        .await?;
        Ok(EngineVariant::Qwen36Moe(state))
    } else {
        let state = build_plain_causal_state(
            model_impl,
            tokenizer,
            model,
            runtime,
            loaded_model_weight_bytes,
            paged_prefix_cache,
            prefix_lru_cache,
            vision_input,
        )
        .await?;
        Ok(EngineVariant::Qwen36Moe(state))
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_plain_causal_state<M>(
    model_impl: M,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EnginePoolRuntimeConfig,
    loaded_model_weight_bytes: usize,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    vision_input: Option<VisionInputConfig>,
) -> Result<AppState<M>>
where
    M: Model + crate::core::scheduler::DenseVlMethods + Send + 'static,
{
    let profile = model.scheduler_runtime_profile.clone();
    let state = super::build_plain_app_state(
        model_impl,
        tokenizer,
        model.id.clone(),
        profile.config.prefill_chunk_size,
        profile.config.b_max,
        profile.config.admission_deadline_ms,
        profile.config.admission_queue_max,
        profile.config.max_cache_cap,
        profile.config.decode_cadence_mid_chunk_cap,
        runtime.kv_cache_turboquant_bits,
        profile,
        runtime.scheduler_autotune_report,
        vision_input,
        paged_prefix_cache,
        prefix_lru_cache,
        Some(loaded_model_weight_bytes),
        runtime.active_kv_offload.clone(),
    )
    .await?;
    Ok(state.with_sampling_defaults(model.sampling_defaults))
}

#[allow(clippy::too_many_arguments)]
async fn build_mtp_causal_state<M>(
    model_impl: M,
    tokenizer: Tokenizer,
    model: &EngineModelConfig,
    runtime: &EnginePoolRuntimeConfig,
    loaded_model_weight_bytes: usize,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    mtp_config: ResolvedEngineMtpConfig,
    vision_input: Option<VisionInputConfig>,
) -> Result<AppState<M>>
where
    M: Model + crate::core::scheduler::DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    let mtp_loader = Loader::open_mtp(&mtp_config.model_dir)
        .with_context(|| format!("Loader::open_mtp {}", mtp_config.model_dir.display()))?;
    let total_model_weight_bytes =
        loaded_model_weight_bytes.saturating_add(mtp_loader.loaded_tensor_bytes());
    let mtp = model_impl
        .load_mtp_head(&mtp_loader)
        .with_context(|| format!("loading MTP head from {}", mtp_config.model_dir.display()))?;
    let profile = model.scheduler_runtime_profile.clone();
    let state = super::build_mtp_app_state(
        model_impl,
        mtp,
        mtp_config.draft_tokens,
        tokenizer,
        model.id.clone(),
        profile.config.prefill_chunk_size,
        profile.config.b_max,
        profile.config.admission_deadline_ms,
        profile.config.admission_queue_max,
        profile.config.max_cache_cap,
        profile.config.decode_cadence_mid_chunk_cap,
        runtime.kv_cache_turboquant_bits,
        profile,
        runtime.scheduler_autotune_report,
        vision_input,
        paged_prefix_cache,
        prefix_lru_cache,
        Some(total_model_weight_bytes),
        runtime.active_kv_offload.clone(),
    )
    .await?;
    Ok(state.with_sampling_defaults(model.sampling_defaults))
}

pub async fn serve_engine_pool(
    config: EnginePoolConfig,
    runtime: EnginePoolRuntimeConfig,
) -> Result<()> {
    let host = runtime.host.clone();
    let port = runtime.port;
    let state = EnginePoolState::new(config, runtime).await?;
    state.start_model_ttl_sweeper();
    let app = engine_pool_router().with_state(state);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("parsing socket addr {host}:{port}"))?;
    tracing::info!("ironmlx EnginePool server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn engine_pool_router() -> Router<EnginePoolState> {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(healthz_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/models/:model_id/load", post(load_model_handler))
        .route("/v1/models/:model_id/unload", post(unload_model_handler))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
}

async fn openai_chat_completions(
    State(pool): State<EnginePoolState>,
    Json(mut req): Json<openai::ChatRequest>,
) -> Response {
    let requested = req.model.as_deref().filter(|model| !model.is_empty());
    let (model_id, engine) = match pool.resolve_engine(requested).await {
        Ok(resolved) => resolved,
        Err(error) => return engine_error_response(error),
    };
    if req.model.is_none() {
        req.model = Some(model_id.to_string());
    }
    engine.openai_chat_completions(req).await
}

async fn anthropic_messages(
    State(pool): State<EnginePoolState>,
    Json(mut req): Json<anthropic::MessagesRequest>,
) -> Response {
    let requested = req.model.as_deref().filter(|model| !model.is_empty());
    let (model_id, engine) = match pool.resolve_engine(requested).await {
        Ok(resolved) => resolved,
        Err(error) => return engine_error_response(error),
    };
    if req.model.is_none() {
        req.model = Some(model_id.to_string());
    }
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
        Err(error) => engine_error_response(error),
    }
}

async fn unload_model_handler(
    State(pool): State<EnginePoolState>,
    Path(model_id): Path<String>,
) -> Response {
    match pool.unload_model(&model_id).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => engine_error_response(error),
    }
}

async fn healthz_handler(State(pool): State<EnginePoolState>) -> Json<EnginePoolHealth> {
    Json(pool.health_snapshot().await)
}

fn engine_error_response(error: anyhow::Error) -> Response {
    let status = if let Some(registry) = error.downcast_ref::<EngineRegistryError>() {
        match registry {
            EngineRegistryError::UnknownModel { .. }
            | EngineRegistryError::ModelDisabled { .. } => StatusCode::NOT_FOUND,
            EngineRegistryError::AmbiguousDefault => StatusCode::BAD_REQUEST,
            _ => StatusCode::BAD_REQUEST,
        }
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(EngineErrorEnvelope {
            error: EngineErrorBody {
                message: format!("{error:#}"),
                kind: "engine_pool_error",
            },
        }),
    )
        .into_response()
}

#[derive(Debug, Serialize)]
struct EngineErrorEnvelope {
    error: EngineErrorBody,
}

#[derive(Debug, Serialize)]
struct EngineErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Debug, Serialize)]
struct OpenAiModelList {
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

#[derive(Debug, Serialize)]
pub(crate) struct EngineModelControlResult {
    pub(crate) id: String,
    load_policy: EngineLoadPolicy,
    pub(crate) state: EngineRuntimeState,
    #[serde(skip_serializing_if = "Option::is_none")]
    unload_reason: Option<EngineUnloadReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    load_attempts: u64,
    request_count: u64,
}

#[derive(Debug, Serialize)]
struct EnginePoolHealth {
    status: &'static str,
    mode: &'static str,
    default_model: Option<String>,
    max_loaded_models: Option<usize>,
    loaded_models: usize,
    models: Vec<EngineModelHealth>,
    version: &'static str,
}

#[derive(Debug, Serialize)]
struct EngineModelHealth {
    id: String,
    load_policy: EngineLoadPolicy,
    state: EngineRuntimeState,
    #[serde(skip_serializing_if = "Option::is_none")]
    unload_reason: Option<EngineUnloadReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    loaded_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_used_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    load_started_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failed_unix_ms: Option<u64>,
    load_attempts: u64,
    request_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    loaded: Option<LoadedEngineHealth>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LoadedEngineHealth {
    Causal(Box<health::HealthSnapshot>),
    DiffusionGemma {
        scheduler: &'static str,
        active_requests: usize,
        queued_requests: usize,
        queue_capacity: usize,
    },
}

impl LoadedEngineHealth {
    fn max_position_embeddings(&self) -> i32 {
        match self {
            Self::Causal(snapshot) => snapshot.model.max_position_embeddings,
            Self::DiffusionGemma { .. } => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use crate::core::scheduler_autotune::{
        SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
        SchedulerAutotuneRuntimeProfileMetadata, SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
    };
    use crate::core::server::SamplingDefaults;

    use super::{
        decide_engine_pool_capacity, unix_time_ms, EngineLoadPolicy, EngineLoadTrigger,
        EngineModelConfig, EngineModelManifest, EnginePoolCapacityDecision,
        EnginePoolCapacityPolicy, EnginePoolConfig, EnginePoolManifest, EnginePoolMemoryLimits,
        EnginePoolRuntimeConfig, EnginePoolState, EngineRegistry, EngineRegistryError,
        EngineRuntimeState, EngineSlot, EngineSlotState, ModelTtlCandidate,
    };
    use tokio::sync::{Mutex, Notify};

    fn model(id: &str) -> EngineModelManifest {
        EngineModelManifest {
            id: id.to_string(),
            path: PathBuf::from(format!("/models/{id}")),
            load_policy: EngineLoadPolicy::Lazy,
            default: false,
            scheduler_profile: None,
            mtp_model_dir: None,
            mtp_draft_tokens: None,
        }
    }

    fn profile_config() -> SchedulerAutotuneProfileConfig {
        SchedulerAutotuneProfileConfig {
            b_max: 1,
            prefill_chunk_size: 512,
            admission_deadline_ms: 5,
            admission_queue_max: 4,
            max_cache_cap: 1024,
            decode_cadence_mid_chunk_cap: 128,
        }
    }

    fn runtime_profile() -> SchedulerAutotuneRuntimeProfile {
        SchedulerAutotuneRuntimeProfile {
            schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
            model_name: "test-model".to_string(),
            hardware_label: "test-host".to_string(),
            config: profile_config(),
            rules: Vec::new(),
            metadata: SchedulerAutotuneRuntimeProfileMetadata::synthetic(0),
        }
    }

    fn runtime_config() -> EnginePoolRuntimeConfig {
        EnginePoolRuntimeConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            kv_cache_turboquant_bits: None,
            scheduler_autotune_report: false,
            paged_prefix_cache: None,
            prefix_lru_cache_max_bytes: None,
            model_ttl: None,
            memory_limits: EnginePoolMemoryLimits::default(),
            active_kv_offload: crate::core::cache::ActiveKvOffloadConfig::disabled(),
        }
    }

    fn model_config(id: &str, path: &Path, load_policy: EngineLoadPolicy) -> EngineModelConfig {
        EngineModelConfig {
            id: id.to_string(),
            path: path.to_path_buf(),
            load_policy,
            default: false,
            scheduler_runtime_profile: runtime_profile(),
            mtp: None,
            sampling_defaults: SamplingDefaults::default(),
        }
    }

    fn write_minimal_model_config(model_type: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ironmlx-engine-pool-test-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp model dir");
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"model_type":"{model_type}"}}"#),
        )
        .expect("write config");
        dir
    }

    #[test]
    fn registry_rejects_duplicate_model_ids() {
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: None,
            models: vec![model("alpha"), model("alpha")],
        };

        let err = EngineRegistry::new(manifest).expect_err("duplicate ids must be rejected");

        assert!(matches!(
            err,
            EngineRegistryError::DuplicateModelId { id } if id == "alpha"
        ));
    }

    #[test]
    fn registry_uses_single_enabled_model_as_implicit_default() {
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: None,
            models: vec![model("alpha")],
        };
        let registry = EngineRegistry::new(manifest).expect("registry");

        assert_eq!(registry.resolve_model_id(None).expect("default"), "alpha");
        assert_eq!(
            registry.resolve_model_id(Some("alpha")).expect("explicit"),
            "alpha"
        );
    }

    #[test]
    fn registry_requires_default_when_multiple_models_are_enabled() {
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: None,
            models: vec![model("alpha"), model("beta")],
        };
        let registry = EngineRegistry::new(manifest).expect("registry");

        let err = registry
            .resolve_model_id(None)
            .expect_err("ambiguous default must fail");

        assert!(matches!(err, EngineRegistryError::AmbiguousDefault));
    }

    #[test]
    fn registry_routes_missing_request_to_declared_default_model() {
        let mut beta = model("beta");
        beta.default = true;
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: None,
            models: vec![model("alpha"), beta],
        };
        let registry = EngineRegistry::new(manifest).expect("registry");

        assert_eq!(registry.resolve_model_id(None).expect("default"), "beta");
    }

    #[test]
    fn registry_rejects_unknown_and_disabled_models() {
        let mut beta = model("beta");
        beta.load_policy = EngineLoadPolicy::Disabled;
        let manifest = EnginePoolManifest {
            default_model: Some("alpha".to_string()),
            max_loaded_models: None,
            models: vec![model("alpha"), beta],
        };
        let registry = EngineRegistry::new(manifest).expect("registry");

        let missing = registry
            .resolve_model_id(Some("missing"))
            .expect_err("unknown model");
        let disabled = registry
            .resolve_model_id(Some("beta"))
            .expect_err("disabled model");

        assert!(matches!(
            missing,
            EngineRegistryError::UnknownModel { id } if id == "missing"
        ));
        assert!(matches!(
            disabled,
            EngineRegistryError::ModelDisabled { id } if id == "beta"
        ));
    }

    #[test]
    fn registry_openai_model_list_excludes_disabled_models() {
        let mut beta = model("beta");
        beta.load_policy = EngineLoadPolicy::Disabled;
        let manifest = EnginePoolManifest {
            default_model: Some("alpha".to_string()),
            max_loaded_models: None,
            models: vec![model("alpha"), beta],
        };
        let registry = EngineRegistry::new(manifest).expect("registry");

        let ids: Vec<_> = registry
            .servable_models()
            .iter()
            .map(|model| model.id.as_str())
            .collect();

        assert_eq!(ids, vec!["alpha"]);
    }

    #[test]
    fn registry_rejects_manifest_with_no_enabled_models() {
        let mut alpha = model("alpha");
        alpha.load_policy = EngineLoadPolicy::Disabled;
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: None,
            models: vec![alpha],
        };

        let err = EngineRegistry::new(manifest).expect_err("no enabled models must fail");

        assert!(matches!(err, EngineRegistryError::NoEnabledModels));
    }

    #[test]
    fn registry_rejects_zero_engine_pool_capacity() {
        let manifest = EnginePoolManifest {
            default_model: None,
            max_loaded_models: Some(0),
            models: vec![model("alpha")],
        };

        let err = EngineRegistry::new(manifest).expect_err("zero capacity must fail");

        assert!(matches!(err, EngineRegistryError::InvalidMaxLoadedModels));
    }

    #[test]
    fn registry_rejects_preload_count_above_capacity() {
        let mut alpha = model("alpha");
        alpha.load_policy = EngineLoadPolicy::Preload;
        let mut beta = model("beta");
        beta.load_policy = EngineLoadPolicy::Preload;
        let manifest = EnginePoolManifest {
            default_model: Some("alpha".to_string()),
            max_loaded_models: Some(1),
            models: vec![alpha, beta],
        };

        let err = EngineRegistry::new(manifest).expect_err("preload capacity must fail");

        assert!(matches!(
            err,
            EngineRegistryError::PreloadCapacityExceeded {
                preload_count: 2,
                max_loaded_models: 1
            }
        ));
    }

    #[test]
    fn registry_dynamic_upsert_does_not_replace_existing_default_without_explicit_request() {
        let mut registry = EngineRegistry::empty(None).expect("registry");

        registry.upsert_model(model("alpha"), false).expect("alpha");
        registry.upsert_model(model("beta"), false).expect("beta");

        assert_eq!(registry.default_model(), Some("alpha"));
        assert!(registry.model("alpha").expect("alpha").default);
        assert!(!registry.model("beta").expect("beta").default);
    }

    #[test]
    fn registry_dynamic_upsert_preserves_current_default_when_replacing_same_model() {
        let mut registry = EngineRegistry::empty(None).expect("registry");

        registry.upsert_model(model("alpha"), false).expect("alpha");
        registry
            .upsert_model(model("alpha"), false)
            .expect("replace alpha");

        assert_eq!(registry.default_model(), Some("alpha"));
        assert!(registry.model("alpha").expect("alpha").default);
    }

    #[test]
    fn dynamic_engine_pool_rejects_capacity_instead_of_evicting_models() {
        let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");

        assert_eq!(pool.inner.capacity_policy, EnginePoolCapacityPolicy::Reject);
    }

    #[test]
    fn dynamic_engine_pool_capacity_decision_rejects_when_full() {
        assert_eq!(
            decide_engine_pool_capacity(EnginePoolCapacityPolicy::Reject, 3, 3),
            EnginePoolCapacityDecision::Reject
        );
    }

    #[test]
    fn manifest_engine_pool_capacity_decision_keeps_lru_eviction_when_full() {
        assert_eq!(
            decide_engine_pool_capacity(EnginePoolCapacityPolicy::EvictLruIdle, 3, 3),
            EnginePoolCapacityDecision::TryEvictLruIdle
        );
    }

    #[test]
    fn engine_pool_capacity_decision_continues_when_below_capacity() {
        assert_eq!(
            decide_engine_pool_capacity(EnginePoolCapacityPolicy::Reject, 3, 2),
            EnginePoolCapacityDecision::Continue
        );
    }

    #[test]
    fn engine_pool_model_memory_limit_rejects_weight_sum_above_limit() {
        let limits = super::EnginePoolMemoryLimits {
            total_memory_limit_bytes: None,
            model_memory_limit_bytes: Some(10),
        };

        let error = limits
            .check_model_memory_limit("alpha", 11)
            .expect_err("model bytes above limit must fail");

        assert!(format!("{error:#}").contains("engine pool model memory limit exceeded"));
    }

    #[test]
    fn engine_pool_total_memory_limit_rejects_active_bytes_above_limit() {
        let limits = super::EnginePoolMemoryLimits {
            total_memory_limit_bytes: Some(10),
            model_memory_limit_bytes: None,
        };

        let error = limits
            .check_total_memory_limit("alpha", 11)
            .expect_err("active bytes above limit must fail");

        assert!(format!("{error:#}").contains("engine pool total memory limit exceeded"));
    }

    #[tokio::test]
    async fn dynamic_register_adds_lazy_unloaded_model_without_loading() {
        let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");
        let model_dir = write_minimal_model_config("qwen3_5");
        let config = model_config("alpha", &model_dir, EngineLoadPolicy::Lazy);

        let result = pool
            .register_dynamic_model(config, false)
            .await
            .expect("register dynamic model");

        assert_eq!(result.id, "alpha");
        assert_eq!(result.state, EngineRuntimeState::Unloaded);
        assert_eq!(result.load_attempts, 0);
        assert!(pool.loaded_model_infos().await.is_empty());
        assert_eq!(pool.inner.registry.lock().await.default_model(), None);
        assert!(
            !pool
                .inner
                .registry
                .lock()
                .await
                .model("alpha")
                .expect("registered model")
                .default
        );

        let list = pool.model_list().await;
        assert_eq!(list.data.len(), 1);
        assert_eq!(list.data[0].id, "alpha");
        assert_eq!(list.data[0].state, EngineRuntimeState::Unloaded);
    }

    #[tokio::test]
    async fn dynamic_unload_keeps_model_registered_for_lazy_reload() {
        let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");
        let model_dir = write_minimal_model_config("qwen3_5");
        let config = model_config("alpha", &model_dir, EngineLoadPolicy::Lazy);
        pool.register_dynamic_model(config, true)
            .await
            .expect("register dynamic model");

        let result = pool
            .unload_dynamic_model("alpha")
            .await
            .expect("unload dynamic model");

        assert_eq!(result.id, "alpha");
        assert_eq!(result.state, EngineRuntimeState::Unloaded);
        assert_eq!(
            pool.inner
                .registry
                .lock()
                .await
                .resolve_model_id(Some("alpha"))
                .expect("registered model"),
            "alpha"
        );
    }

    #[tokio::test]
    async fn draining_model_rejects_new_request_until_unload_finishes() {
        let pool = EnginePoolState::new_dynamic(runtime_config(), Some(3)).expect("pool");
        let model_dir = write_minimal_model_config("qwen3_5");
        let slot = EngineSlot {
            model: model_config("alpha", &model_dir, EngineLoadPolicy::Lazy),
            runtime: runtime_config(),
            active_requests: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(1)),
            state: Mutex::new(EngineSlotState::Draining {
                started_unix_ms: unix_time_ms(),
                load_attempts: 1,
                request_count: 1,
            }),
            notify: Notify::new(),
        };

        let error = slot
            .ensure_loaded(&pool.inner, EngineLoadTrigger::Request)
            .await
            .expect_err("draining model must reject new requests");

        assert!(format!("{error:#}").contains("currently unloading"));
    }

    #[test]
    fn registry_restore_default_model_preserves_previous_default_after_failed_dynamic_load() {
        let mut registry = EngineRegistry::empty(None).expect("registry");
        registry.upsert_model(model("alpha"), true).expect("alpha");
        let previous_default = registry.default_model().map(str::to_string);

        registry.upsert_model(model("beta"), true).expect("beta");
        assert_eq!(registry.default_model(), Some("beta"));

        registry.remove_model("beta");
        registry
            .restore_default_model(previous_default)
            .expect("restore default");

        assert_eq!(registry.default_model(), Some("alpha"));
    }

    #[test]
    fn registry_restore_default_model_can_restore_ambiguous_no_default_state() {
        let mut registry = EngineRegistry::empty(None).expect("registry");
        registry.upsert_model(model("alpha"), false).expect("alpha");
        registry.restore_default_model(None).expect("clear default");
        registry.upsert_model(model("beta"), true).expect("beta");
        assert_eq!(registry.default_model(), Some("beta"));

        registry.remove_model("beta");
        registry
            .restore_default_model(None)
            .expect("restore no default");

        assert_eq!(registry.default_model(), None);
        assert_eq!(
            registry.resolve_model_id(None).expect("single model"),
            "alpha"
        );
    }

    #[test]
    fn eviction_candidate_selection_uses_lru_order() {
        let chosen = super::select_lru_eviction_candidate(vec![
            super::EvictionCandidate {
                id: "recent".to_string(),
                last_used_unix_ms: 30,
            },
            super::EvictionCandidate {
                id: "oldest".to_string(),
                last_used_unix_ms: 10,
            },
            super::EvictionCandidate {
                id: "middle".to_string(),
                last_used_unix_ms: 20,
            },
        ]);

        assert_eq!(chosen.as_deref(), Some("oldest"));
    }

    #[test]
    fn model_ttl_candidates_require_loaded_idle_expired_models() {
        let candidates = super::select_model_ttl_unload_candidates(
            vec![
                ModelTtlCandidate {
                    id: "expired".to_string(),
                    state: EngineRuntimeState::Loaded,
                    active_requests: 0,
                    last_used_unix_ms: Some(1_000),
                },
                ModelTtlCandidate {
                    id: "active".to_string(),
                    state: EngineRuntimeState::Loaded,
                    active_requests: 1,
                    last_used_unix_ms: Some(1_000),
                },
                ModelTtlCandidate {
                    id: "fresh".to_string(),
                    state: EngineRuntimeState::Loaded,
                    active_requests: 0,
                    last_used_unix_ms: Some(9_000),
                },
                ModelTtlCandidate {
                    id: "unloaded".to_string(),
                    state: EngineRuntimeState::Unloaded,
                    active_requests: 0,
                    last_used_unix_ms: Some(1_000),
                },
            ],
            10_000,
            Duration::from_secs(5),
        );

        assert_eq!(candidates, vec!["expired"]);
    }

    #[test]
    fn model_ttl_unload_reason_serializes_as_ttl() {
        let value = serde_json::to_value(super::EngineUnloadReason::Ttl).expect("ttl reason");

        assert_eq!(value, serde_json::json!("ttl"));
    }

    #[test]
    fn engine_pool_router_builds_control_routes() {
        let _router = super::engine_pool_router();
    }

    #[tokio::test]
    async fn engine_pool_state_accepts_gemma4_unified_model_type_before_lazy_load() {
        let model_dir = write_minimal_model_config("gemma4_unified");
        let config = EnginePoolConfig {
            default_model: Some("gemma4-unified".to_string()),
            max_loaded_models: Some(1),
            models: vec![model_config(
                "gemma4-unified",
                &model_dir,
                EngineLoadPolicy::Lazy,
            )],
        };

        EnginePoolState::new(config, runtime_config())
            .await
            .expect("gemma4_unified lazy model type should be accepted");
    }
}
