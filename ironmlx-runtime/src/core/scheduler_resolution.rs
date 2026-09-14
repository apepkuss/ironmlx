//! Native scheduler defaults, profile selection and per-model resolution.
//! The CLI translates its arguments into these options before calling this module.

use crate::core::adaptive_admission::{
    GEMMA4_DRAFTER_ADAPTIVE_PHYSICAL_B_MAX, QWEN_MTP_ADAPTIVE_PHYSICAL_B_MAX,
};
use crate::core::scheduler_autotune::{
    evaluate_scheduler_autotune_profile_health, SchedulerAutotuneProfileConfig,
    SchedulerAutotuneProfileHealthInput, SchedulerAutotuneProfileHealthReport,
    SchedulerAutotuneProfileHealthStatus, SchedulerAutotuneRuntimeContext,
    SchedulerAutotuneRuntimeProfile, SchedulerAutotuneRuntimeProfileMetadata,
    SchedulerKvQuantization, SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
};
use crate::core::scheduler_profile_context::{
    build_scheduler_runtime_context, SchedulerProfileContextOptions,
};
use crate::core::scheduler_profile_store::{
    detect_scheduler_profile_hardware_label, SchedulerProfileStore,
};
use crate::Result;
use anyhow::{bail, Context};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const DEFAULT_PREFILL_CHUNK_SIZE: usize = 2048;
pub(crate) const DEFAULT_B_MAX: usize = 1;
pub(crate) const DEFAULT_ADMISSION_DEADLINE_MS: u64 = 5;
pub(crate) const DEFAULT_ADMISSION_QUEUE_MAX: usize = 32;
pub const DEFAULT_MAX_CACHE_CAP: usize = 32768;
pub(crate) const DEFAULT_DECODE_CADENCE_MID_CHUNK_CAP: usize = 256;
pub const BYTES_PER_GIB: usize = 1024 * 1024 * 1024;
#[derive(Debug, Clone)]
pub struct SchedulerResolutionOptions {
    pub prefill_chunk_size: Option<usize>,
    pub b_max: Option<usize>,
    pub admission_deadline_ms: Option<u64>,
    pub admission_queue_max: Option<usize>,
    pub max_cache_cap: Option<usize>,
    pub decode_cadence_mid_chunk_cap: Option<usize>,
    pub scheduler_profile: Option<PathBuf>,
    pub kv_quantization: SchedulerKvQuantization,
    pub paged_prefix_cache_dir: Option<PathBuf>,
    pub paged_prefix_cache_block_size: i32,
    pub paged_prefix_cache_max_pages: Option<i32>,
    pub prefix_lru_cache_max_bytes: Option<usize>,
    pub ssd_prefix_cache_max_gb: Option<usize>,
    pub active_kv_offload: bool,
    pub memory_limit_total_gb: Option<usize>,
    pub memory_limit_model_gb: Option<usize>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerServeConfig {
    pub prefill_chunk_size: usize,
    pub b_max: usize,
    pub admission_deadline_ms: u64,
    pub admission_queue_max: usize,
    pub max_cache_cap: usize,
    pub decode_cadence_mid_chunk_cap: usize,
}

impl Default for SchedulerServeConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
            b_max: DEFAULT_B_MAX,
            admission_deadline_ms: DEFAULT_ADMISSION_DEADLINE_MS,
            admission_queue_max: DEFAULT_ADMISSION_QUEUE_MAX,
            max_cache_cap: DEFAULT_MAX_CACHE_CAP,
            decode_cadence_mid_chunk_cap: DEFAULT_DECODE_CADENCE_MID_CHUNK_CAP,
        }
    }
}

#[derive(Debug)]
pub struct SchedulerProfileLoad {
    pub path: PathBuf,
    pub profile: SchedulerAutotuneRuntimeProfile,
    pub auto_loaded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerProfileSource {
    Explicit,
    Store,
}

#[derive(Debug)]
pub struct ResolvedSchedulerRuntime {
    pub scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    pub scheduler_config: SchedulerServeConfig,
    pub profile_source: Option<SchedulerProfileSource>,
}

pub(crate) fn default_scheduler_profile_config() -> SchedulerAutotuneProfileConfig {
    SchedulerAutotuneProfileConfig {
        b_max: DEFAULT_B_MAX,
        prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
        admission_deadline_ms: DEFAULT_ADMISSION_DEADLINE_MS,
        admission_queue_max: DEFAULT_ADMISSION_QUEUE_MAX,
        max_cache_cap: DEFAULT_MAX_CACHE_CAP,
        decode_cadence_mid_chunk_cap: DEFAULT_DECODE_CADENCE_MID_CHUNK_CAP,
    }
}
pub fn default_scheduler_runtime_profile(
    runtime_context: SchedulerAutotuneRuntimeContext,
) -> SchedulerAutotuneRuntimeProfile {
    SchedulerAutotuneRuntimeProfile {
        schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
        model_name: "default".to_string(),
        hardware_label: "local".to_string(),
        runtime_context,
        config: default_scheduler_profile_config(),
        rules: Vec::new(),
        metadata: SchedulerAutotuneRuntimeProfileMetadata::synthetic(0),
    }
}
pub(crate) fn read_scheduler_runtime_profile(
    path: &Path,
) -> Result<SchedulerAutotuneRuntimeProfile> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}
pub(crate) fn adaptive_mtp_physical_b_max(
    architecture: crate::models::ModelArchitecture,
    mtp_enabled: bool,
) -> Option<usize> {
    if !mtp_enabled {
        return None;
    }
    match architecture {
        crate::models::ModelArchitecture::Gemma4 => Some(GEMMA4_DRAFTER_ADAPTIVE_PHYSICAL_B_MAX),
        crate::models::ModelArchitecture::Qwen35Dense
        | crate::models::ModelArchitecture::Qwen35Moe => Some(QWEN_MTP_ADAPTIVE_PHYSICAL_B_MAX),
        _ => None,
    }
}
pub fn apply_adaptive_mtp_scheduler_defaults(
    args: &SchedulerResolutionOptions,
    architecture: crate::models::ModelArchitecture,
    mtp_enabled: bool,
    resolved: &mut ResolvedSchedulerRuntime,
) -> bool {
    let explicit_scheduler_profile = args.scheduler_profile.is_some()
        || resolved.profile_source == Some(SchedulerProfileSource::Explicit);
    let Some(target) = adaptive_mtp_physical_b_max(architecture, mtp_enabled) else {
        return false;
    };
    if args.b_max.is_some() || explicit_scheduler_profile {
        return false;
    }

    let mut changed = false;
    if resolved.scheduler_config.b_max < target {
        resolved.scheduler_config.b_max = target;
        changed = true;
    }
    if resolved.scheduler_runtime_profile.config.b_max < target {
        resolved.scheduler_runtime_profile.config.b_max = target;
        changed = true;
    }
    for rule in &mut resolved.scheduler_runtime_profile.rules {
        if rule.config.b_max < target {
            rule.config.b_max = target;
            changed = true;
        }
    }
    changed
}
pub fn load_scheduler_profile_for_model(
    args: &SchedulerResolutionOptions,
    model_dir: &Path,
    store: Option<&SchedulerProfileStore>,
    hardware_label: &str,
    runtime_context_fingerprint: &str,
) -> Result<Option<SchedulerProfileLoad>> {
    load_scheduler_profile_for_model_with_explicit(
        args.scheduler_profile.as_deref(),
        model_dir,
        store,
        hardware_label,
        runtime_context_fingerprint,
    )
}
pub fn load_scheduler_profile_for_model_with_explicit(
    explicit_profile: Option<&Path>,
    model_dir: &Path,
    store: Option<&SchedulerProfileStore>,
    hardware_label: &str,
    runtime_context_fingerprint: &str,
) -> Result<Option<SchedulerProfileLoad>> {
    if let Some(path) = explicit_profile {
        return Ok(Some(SchedulerProfileLoad {
            path: path.to_path_buf(),
            profile: read_scheduler_runtime_profile(path)?,
            auto_loaded: false,
        }));
    }

    let Some(store) = store else {
        return Ok(None);
    };
    let model_name = scheduler_profile_model_name(model_dir)?;
    let Some(path) = (match store.find_profile(
        model_dir,
        hardware_label,
        runtime_context_fingerprint,
    ) {
        Ok(path) => path,
        Err(error) => {
            tracing::warn!(
                "ironmlx serve: scheduler profile store unavailable path={} model_name={} hardware_label={} error={:#}; using CLI/default scheduler config",
                store.root().display(),
                model_name,
                hardware_label,
                error
            );
            None
        }
    }) else {
        return Ok(None);
    };

    let profile = match read_scheduler_runtime_profile(&path) {
        Ok(profile) => profile,
        Err(error) => {
            tracing::warn!(
                "ironmlx serve: scheduler profile ignored path={} model_name={} hardware_label={} error={:#}; using CLI/default scheduler config",
                path.display(),
                model_name,
                hardware_label,
                error
            );
            return Ok(None);
        }
    };

    Ok(Some(SchedulerProfileLoad {
        profile,
        path,
        auto_loaded: true,
    }))
}
pub fn check_loaded_scheduler_profile_health(
    profile: &SchedulerAutotuneRuntimeProfile,
    expected_model_name: &str,
    expected_hardware_label: &str,
    expected_runtime_context: &SchedulerAutotuneRuntimeContext,
    now_unix_ms: u64,
) -> Result<SchedulerAutotuneProfileHealthReport> {
    let report = evaluate_scheduler_autotune_profile_health(SchedulerAutotuneProfileHealthInput {
        profile,
        expected_model_name,
        expected_hardware_label,
        expected_runtime_context,
        current_ironmlx_version: env!("CARGO_PKG_VERSION"),
        now_unix_ms,
        max_age_days: 30,
    });
    if report.status == SchedulerAutotuneProfileHealthStatus::Invalid {
        bail!("invalid scheduler profile:\n{}", report.render_text());
    }
    Ok(report)
}
pub fn log_scheduler_profile_health(
    profile_path: &Path,
    report: &SchedulerAutotuneProfileHealthReport,
) {
    let note_codes = report
        .notes
        .iter()
        .map(|note| note.code.as_str())
        .collect::<Vec<_>>()
        .join(",");
    match report.status {
        SchedulerAutotuneProfileHealthStatus::Healthy => {
            tracing::info!(
                "ironmlx serve: scheduler profile health status={} path={} notes={}",
                report.status.as_str(),
                profile_path.display(),
                note_codes
            );
        }
        SchedulerAutotuneProfileHealthStatus::Warning => {
            tracing::warn!(
                "ironmlx serve: scheduler profile health status={} path={} notes={} recommendation=\"rerun scheduler-autotune calibrate for this model\"",
                report.status.as_str(),
                profile_path.display(),
                note_codes
            );
        }
        SchedulerAutotuneProfileHealthStatus::Invalid => {
            tracing::warn!(
                "ironmlx serve: scheduler profile health status={} path={} notes={}",
                report.status.as_str(),
                profile_path.display(),
                note_codes
            );
        }
    }
}
pub fn scheduler_profile_model_name(model_dir: &Path) -> Result<String> {
    model_dir
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            anyhow::anyhow!("--model has no directory name for scheduler profile lookup")
        })
}
pub fn unix_time_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_millis();
    millis.min(u128::from(u64::MAX)) as u64
}
pub fn scheduler_runtime_context_for_model(
    args: &SchedulerResolutionOptions,
    model_dir: &Path,
    mtp_model_dir: Option<&Path>,
    mtp_draft_tokens: Option<usize>,
    prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
    logical_kv_cap_tokens: Option<usize>,
) -> Result<SchedulerAutotuneRuntimeContext> {
    let logical_kv_cap_tokens = logical_kv_cap_tokens
        .or(args.max_cache_cap)
        .unwrap_or(DEFAULT_MAX_CACHE_CAP);
    build_scheduler_runtime_context(
        model_dir,
        SchedulerProfileContextOptions {
            mtp_model_dir,
            mtp_draft_tokens,
            prompt_lookup,
            kv_quantization: args.kv_quantization,
            paged_prefix_cache_enabled: args.paged_prefix_cache_dir.is_some(),
            paged_prefix_cache_block_size: args.paged_prefix_cache_block_size,
            paged_prefix_cache_max_pages: args.paged_prefix_cache_max_pages,
            prefix_lru_cache_max_bytes: args.prefix_lru_cache_max_bytes,
            ssd_prefix_cache_max_bytes: resolve_ssd_prefix_cache_max_bytes(args)?,
            active_kv_offload: args.active_kv_offload,
            logical_kv_cap_tokens,
            memory_limit_total_bytes: resolve_memory_limit_bytes(
                args.memory_limit_total_gb,
                "--memory-limit-total-gb",
            )?,
            memory_limit_model_bytes: resolve_memory_limit_bytes(
                args.memory_limit_model_gb,
                "--memory-limit-model-gb",
            )?,
        },
    )
}
pub(crate) fn apply_scheduler_overrides(
    args: &SchedulerResolutionOptions,
    base: SchedulerAutotuneProfileConfig,
) -> SchedulerAutotuneProfileConfig {
    SchedulerAutotuneProfileConfig {
        prefill_chunk_size: args.prefill_chunk_size.unwrap_or(base.prefill_chunk_size),
        b_max: args.b_max.unwrap_or(base.b_max),
        admission_deadline_ms: args
            .admission_deadline_ms
            .unwrap_or(base.admission_deadline_ms),
        admission_queue_max: args.admission_queue_max.unwrap_or(base.admission_queue_max),
        max_cache_cap: args.max_cache_cap.unwrap_or(base.max_cache_cap),
        decode_cadence_mid_chunk_cap: args
            .decode_cadence_mid_chunk_cap
            .unwrap_or(base.decode_cadence_mid_chunk_cap),
    }
}
pub(crate) fn validate_scheduler_serve_config(
    config: SchedulerAutotuneProfileConfig,
) -> Result<()> {
    if config.b_max == 0 {
        bail!("scheduler b_max must be >= 1");
    }
    if config.decode_cadence_mid_chunk_cap == 0 {
        bail!("scheduler decode_cadence_mid_chunk_cap must be >= 1");
    }
    Ok(())
}
pub(crate) fn validate_dynamic_rules(profile: &SchedulerAutotuneRuntimeProfile) -> Result<()> {
    for rule in &profile.rules {
        if rule.config.b_max != profile.config.b_max
            || rule.config.admission_deadline_ms != profile.config.admission_deadline_ms
            || rule.config.admission_queue_max != profile.config.admission_queue_max
            || rule.config.max_cache_cap != profile.config.max_cache_cap
        {
            bail!(
                "scheduler profile dynamic rules may only vary prefill_chunk_size and decode_cadence_mid_chunk_cap"
            );
        }
        validate_scheduler_serve_config(rule.config)?;
    }
    Ok(())
}
pub fn resolve_scheduler_runtime_profile(
    args: &SchedulerResolutionOptions,
    profile: Option<&SchedulerAutotuneRuntimeProfile>,
    runtime_context: &SchedulerAutotuneRuntimeContext,
) -> Result<SchedulerAutotuneRuntimeProfile> {
    if let Some(profile) = profile {
        if profile.schema_version != SCHEDULER_AUTOTUNE_SCHEMA_VERSION {
            bail!(
                "scheduler profile schema_version mismatch: expected {}, got {}",
                SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
                profile.schema_version
            );
        }
    }

    let mut resolved = profile
        .cloned()
        .unwrap_or_else(|| default_scheduler_runtime_profile(runtime_context.clone()));
    resolved.config = apply_scheduler_overrides(args, resolved.config);
    for rule in &mut resolved.rules {
        rule.config = apply_scheduler_overrides(args, rule.config);
    }
    validate_scheduler_serve_config(resolved.config)?;
    validate_dynamic_rules(&resolved)?;
    Ok(resolved)
}
pub fn resolve_scheduler_for_model_with_speculative(
    args: &SchedulerResolutionOptions,
    model_dir: &Path,
    mtp_model_dir: Option<&Path>,
    mtp_draft_tokens: Option<usize>,
    prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
    max_cache_cap_override: Option<usize>,
) -> Result<ResolvedSchedulerRuntime> {
    let runtime_context = scheduler_runtime_context_for_model(
        args,
        model_dir,
        mtp_model_dir,
        mtp_draft_tokens,
        prompt_lookup,
        max_cache_cap_override,
    )?;
    let runtime_context_fingerprint = runtime_context.fingerprint();
    let scheduler_profile_store = if args.scheduler_profile.is_none() {
        match SchedulerProfileStore::open_default() {
            Ok(store) => Some(store),
            Err(error) => {
                tracing::warn!(
                    "ironmlx serve: scheduler profile store disabled error={:#}; using CLI/default scheduler config",
                    error
                );
                None
            }
        }
    } else {
        None
    };
    let scheduler_profile_hardware_label = detect_scheduler_profile_hardware_label();
    let mut scheduler_profile_load = load_scheduler_profile_for_model(
        args,
        model_dir,
        scheduler_profile_store.as_ref(),
        &scheduler_profile_hardware_label,
        &runtime_context_fingerprint,
    )?;
    let scheduler_profile_model_name = scheduler_profile_model_name(model_dir)?;
    let mut discard_auto_profile = false;
    if let Some(load) = scheduler_profile_load.as_ref() {
        match check_loaded_scheduler_profile_health(
            &load.profile,
            &scheduler_profile_model_name,
            &scheduler_profile_hardware_label,
            &runtime_context,
            unix_time_ms(),
        ) {
            Ok(report) => log_scheduler_profile_health(&load.path, &report),
            Err(error) if load.auto_loaded => {
                tracing::warn!(
                    "ironmlx serve: scheduler profile ignored path={} model_name={} hardware_label={} error={:#}; using CLI/default scheduler config",
                    load.path.display(),
                    scheduler_profile_model_name,
                    scheduler_profile_hardware_label,
                    error
                );
                discard_auto_profile = true;
            }
            Err(error) => return Err(error),
        }
    }
    if discard_auto_profile {
        scheduler_profile_load = None;
    }
    let scheduler_runtime_profile = resolve_scheduler_runtime_profile(
        args,
        scheduler_profile_load.as_ref().map(|load| &load.profile),
        &runtime_context,
    )?;
    if scheduler_profile_load.is_none() && args.scheduler_profile.is_none() {
        match scheduler_profile_store.as_ref() {
            Some(store) => tracing::info!(
                "ironmlx serve: no matching scheduler profile found store={} model={} hardware_label={}; using CLI/default scheduler config",
                store.root().display(),
                model_dir.display(),
                scheduler_profile_hardware_label
            ),
            None => tracing::info!(
                "ironmlx serve: no scheduler profile store available model={} hardware_label={}; using CLI/default scheduler config",
                model_dir.display(),
                scheduler_profile_hardware_label
            ),
        }
    }
    let scheduler_config = SchedulerServeConfig {
        prefill_chunk_size: scheduler_runtime_profile.config.prefill_chunk_size,
        b_max: scheduler_runtime_profile.config.b_max,
        admission_deadline_ms: scheduler_runtime_profile.config.admission_deadline_ms,
        admission_queue_max: scheduler_runtime_profile.config.admission_queue_max,
        max_cache_cap: scheduler_runtime_profile.config.max_cache_cap,
        decode_cadence_mid_chunk_cap: scheduler_runtime_profile
            .config
            .decode_cadence_mid_chunk_cap,
    };
    if let Some(load) = &scheduler_profile_load {
        let source = if load.auto_loaded {
            "store"
        } else {
            "explicit"
        };
        tracing::info!(
            "ironmlx serve: scheduler profile applied source={} path={} model_name={} hardware_label={} rules={}",
            source,
            load.path.display(),
            load.profile.model_name,
            load.profile.hardware_label,
            scheduler_runtime_profile.rules.len()
        );
    }
    let profile_source = scheduler_profile_load.as_ref().map(|load| {
        if load.auto_loaded {
            SchedulerProfileSource::Store
        } else {
            SchedulerProfileSource::Explicit
        }
    });

    Ok(ResolvedSchedulerRuntime {
        scheduler_runtime_profile,
        scheduler_config,
        profile_source,
    })
}
pub fn resolve_ssd_prefix_cache_max_bytes(
    args: &SchedulerResolutionOptions,
) -> Result<Option<usize>> {
    let Some(max_gb) = args.ssd_prefix_cache_max_gb else {
        return Ok(None);
    };
    if max_gb == 0 {
        bail!("--ssd-prefix-cache-max-gb must be > 0");
    }
    max_gb
        .checked_mul(BYTES_PER_GIB)
        .context("--ssd-prefix-cache-max-gb exceeds usize bytes")
        .map(Some)
}
pub fn resolve_memory_limit_bytes(
    limit_gb: Option<usize>,
    flag_name: &str,
) -> Result<Option<usize>> {
    let Some(limit_gb) = limit_gb else {
        return Ok(None);
    };
    limit_gb
        .checked_mul(BYTES_PER_GIB)
        .with_context(|| format!("{flag_name} exceeds usize bytes"))
        .map(Some)
}
pub fn read_model_type(model_dir: &std::path::Path) -> Result<String> {
    let config_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", config_path.display()))?;
    config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("config.json missing model_type"))
}
