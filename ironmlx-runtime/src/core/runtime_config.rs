//! Model registry, resolved execution configuration and resource policies.
//! These types have no HTTP, network listener or command-line dependency.

use crate::core::prompt_lookup::PromptLookupConfig;
use crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile;
use anyhow::{bail, Context, Result};
use ironmlx_lm::models::ModelArchitecture;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Duration;
use {
    crate::core::cache::ActiveKvOffloadConfig, crate::core::cache::PagedPrefixCacheConfig,
    crate::core::cache::PrefixLruCacheConfig,
    ironmlx_lm::core::cache::turboquant_kv::TurboQuantKVBits,
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub repetition_penalty: Option<f32>,
}

impl SamplingDefaults {
    pub fn merge_with_override(self, override_defaults: Self) -> Self {
        Self {
            temperature: override_defaults.temperature.or(self.temperature),
            top_p: override_defaults.top_p.or(self.top_p),
            top_k: override_defaults.top_k.or(self.top_k),
            repetition_penalty: override_defaults
                .repetition_penalty
                .or(self.repetition_penalty),
        }
    }
}

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
    #[serde(default)]
    pub prompt_lookup: Option<PromptLookupConfig>,
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

    pub(crate) fn restore_default_model(
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
    pub pinned: bool,
    pub scheduler_runtime_profile: Option<SchedulerAutotuneRuntimeProfile>,
    pub mtp: Option<EngineMtpSettings>,
    pub prompt_lookup: Option<PromptLookupConfig>,
    pub sampling_defaults: SamplingDefaults,
    pub capabilities: EngineModelCapabilities,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineModelCapabilities {
    pub runtime_kind: &'static str,
    pub supports_streaming: bool,
    pub supports_vision: bool,
    pub supports_mtp: bool,
    pub supports_prompt_lookup: bool,
    pub supports_speculative_decoding: bool,
    pub supports_kv_cache: bool,
    pub supported_sampling_parameters: &'static [&'static str],
}

impl EngineModelCapabilities {
    pub fn for_architecture(
        architecture: ModelArchitecture,
        supports_vision: bool,
    ) -> EngineModelCapabilities {
        let is_diffusion = architecture == ModelArchitecture::DiffusionGemma;
        let supports_mtp = matches!(
            architecture,
            ModelArchitecture::Qwen35Dense
                | ModelArchitecture::Qwen35Moe
                | ModelArchitecture::Gemma4
        );
        EngineModelCapabilities {
            runtime_kind: if is_diffusion {
                "block_diffusion"
            } else {
                "causal"
            },
            supports_streaming: true,
            supports_vision,
            supports_mtp,
            supports_prompt_lookup: architecture.supports_prompt_lookup(),
            supports_speculative_decoding: supports_mtp,
            supports_kv_cache: !is_diffusion,
            supported_sampling_parameters: if is_diffusion {
                &["max_tokens", "temperature", "seed"]
            } else {
                &[
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "repetition_penalty",
                    "seed",
                ]
            },
        }
    }
}

impl EngineModelConfig {
    pub(crate) fn manifest_view(&self) -> EngineModelManifest {
        EngineModelManifest {
            id: self.id.clone(),
            path: self.path.clone(),
            load_policy: self.load_policy,
            default: self.default,
            scheduler_profile: None,
            mtp_model_dir: self.mtp.as_ref().map(|mtp| mtp.model_dir.clone()),
            mtp_draft_tokens: self.mtp.as_ref().and_then(|mtp| mtp.draft_tokens),
            prompt_lookup: self.prompt_lookup,
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
    pub(crate) fn registry_manifest(&self) -> EnginePoolManifest {
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

    pub(crate) fn validate_enabled_model_architectures(&self) -> Result<()> {
        for model in &self.models {
            validate_engine_model_config(model)?;
        }
        Ok(())
    }
}

pub(crate) fn validate_engine_model_config(model: &EngineModelConfig) -> Result<()> {
    if model.load_policy == EngineLoadPolicy::Disabled {
        return Ok(());
    }
    let config_path = model.path.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", config_path.display()))?;
    let architecture = ModelArchitecture::from_config_value(&config).with_context(|| {
        format!(
            "engine model `{}` has unsupported architecture in {}",
            model.id,
            config_path.display()
        )
    })?;
    if let Some(prompt_lookup) = model.prompt_lookup {
        prompt_lookup.validate()?;
        if !architecture.supports_prompt_lookup() {
            bail!(
                "engine model `{}` configures PromptLookup for DiffusionGemma",
                model.id
            );
        }
    }
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
pub struct EngineRuntimeOptions {
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

impl EngineRuntimeOptions {
    pub(crate) fn paged_prefix_cache_config(
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

    pub(crate) fn prefix_lru_cache_config(
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
