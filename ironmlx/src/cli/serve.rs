//! `ironmlx serve` — boot HTTP server with OpenAI + Anthropic compatibility.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context};
use clap::Args;

use super::scheduler_profile_context::{
    build_scheduler_runtime_context, SchedulerProfileContextOptions,
};
use super::scheduler_profile_store::{
    detect_scheduler_profile_hardware_label, SchedulerProfileStore,
};
use super::KvQuantArg;
use crate::core::cache::DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE;
use crate::core::process_memory::StaticMemoryEstimate;
use crate::core::scheduler::DenseVlMethods;
use crate::core::scheduler_autotune::{
    evaluate_scheduler_autotune_profile_health, SchedulerAutotuneProfileConfig,
    SchedulerAutotuneProfileHealthInput, SchedulerAutotuneProfileHealthReport,
    SchedulerAutotuneProfileHealthStatus, SchedulerAutotuneRuntimeContext,
    SchedulerAutotuneRuntimeProfile, SchedulerAutotuneRuntimeProfileMetadata,
    SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
};
use crate::core::server::adaptive_admission::{
    GEMMA4_DRAFTER_ADAPTIVE_PHYSICAL_B_MAX, QWEN_MTP_ADAPTIVE_PHYSICAL_B_MAX,
};
use crate::core::speculative::MtpSpeculativeModel;
use crate::core::{server, Loader, Model, Tokenizer};
use crate::Result;

const DEFAULT_PREFILL_CHUNK_SIZE: usize = 2048;
const DEFAULT_B_MAX: usize = 1;
const DEFAULT_ADMISSION_DEADLINE_MS: u64 = 5;
const DEFAULT_ADMISSION_QUEUE_MAX: usize = 32;
const DEFAULT_MAX_CACHE_CAP: usize = 32768;
const DEFAULT_DECODE_CADENCE_MID_CHUNK_CAP: usize = 256;
const DEFAULT_PAGED_PREFIX_CACHE_DIR: &str = "~/.ironmlx/cache/paged_prefix_cache";
const BYTES_PER_GIB: usize = 1024 * 1024 * 1024;

#[derive(Args, Clone, Debug)]
pub struct ServeArgs {
    /// Local directory containing config.json + model.safetensors + tokenizer.json.
    /// HF repo-id resolution is deferred to a future phase; pass a local path for now.
    #[arg(long, conflicts_with = "model_manifest")]
    pub model: Option<String>,

    /// JSON manifest describing one or more model engines for runtime routing.
    #[arg(long = "model-manifest", conflicts_with = "model")]
    pub model_manifest: Option<PathBuf>,

    /// Maximum number of models that may stay loaded in App / dynamic EnginePool mode.
    #[arg(
        long = "max-loaded-models",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub max_loaded_models: Option<usize>,

    /// Automatically unload loaded lazy models after this many idle minutes.
    /// Pass `0` to disable automatic unloading.
    #[arg(long = "model-ttl-minutes")]
    pub model_ttl_minutes: Option<u64>,

    /// Total MLX active memory guardrail in GiB for App / EnginePool mode.
    /// If omitted, no explicit total memory limit is applied.
    #[arg(
        long = "memory-limit-total-gb",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub memory_limit_total_gb: Option<usize>,

    /// Loaded model weight guardrail in GiB for App / EnginePool mode.
    /// If omitted, no explicit model-only memory limit is applied.
    #[arg(
        long = "memory-limit-model-gb",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub memory_limit_model_gb: Option<usize>,

    /// Bind port.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Prefill chunk size — max tokens per prefill forward call. `0`
    /// disables chunking (single-shot forward over the whole prompt).
    /// Intermediate chunks update the cache only; the last chunk runs
    /// the full forward + lm_head. Defaults to `2048` unless supplied by
    /// `--scheduler-profile`.
    #[arg(long)]
    pub prefill_chunk_size: Option<usize>,

    /// Maximum concurrent in-flight requests (Scheduler slot count).
    /// Requests beyond this limit go to the admission queue. Default `1`
    /// optimizes single-request prefill / decode by avoiding [B,T_max]-padded
    /// MoE compute when only one slot is occupied; pass `--max-sequences N > 1` to
    /// enable concurrent multi-request batching. `0` rejected at startup
    /// because Scheduler with zero slots cannot admit any request.
    #[arg(
        long = "max-sequences",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub b_max: Option<usize>,

    /// Admission-window deadline in milliseconds. After the first
    /// admit in a batch arrives, additional admits are absorbed until
    /// this deadline expires or the batch saturates at b_max.
    /// Defaults to `5` unless supplied by `--scheduler-profile`.
    #[arg(long)]
    pub admission_deadline_ms: Option<u64>,

    /// Capacity of the FIFO admission queue. Requests received while
    /// the scheduler is saturated are parked here. `0` disables queueing
    /// (immediate Err on saturation — mirrors pre-3d behavior).
    /// Defaults to `32` unless supplied by `--scheduler-profile`.
    #[arg(long)]
    pub admission_queue_max: Option<usize>,

    /// Maximum allowed `prompt_len + max_new_tokens` per request. Capped
    /// further at the model's `max_position_embeddings` (Qwen3.5-4B: 262144).
    /// Requests beyond this return HTTP 413 Payload Too Large. B1-p2.3f.
    /// Defaults to `32768` unless supplied by `--scheduler-profile`.
    #[arg(long)]
    pub max_cache_cap: Option<usize>,

    /// Maximum chunk size used by rolling mid-admit while decode rows are active.
    /// Smaller values protect decode cadence under concurrent long-prompt admission;
    /// larger values can reduce queued-request TTFT at the cost of longer decode gaps.
    /// Defaults to `256` unless supplied by `--scheduler-profile`.
    #[arg(long, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub decode_cadence_mid_chunk_cap: Option<usize>,

    /// Runtime scheduler profile exported by `scheduler-autotune select --write-profile`.
    #[arg(long)]
    pub scheduler_profile: Option<PathBuf>,

    /// Print scheduler/autotune diagnostics and recommendations at startup.
    /// Diagnose-only: this does not change any runtime parameter.
    #[arg(long, default_value_t = false)]
    pub scheduler_autotune_report: bool,

    /// Optional local MTP/drafter model directory. When set, greedy speculative
    /// decoding is enabled for supported Qwen and Gemma4 models.
    #[arg(long = "mtp-model-dir")]
    pub mtp_model_dir: Option<PathBuf>,

    /// Maximum MTP draft tokens per speculative window. If omitted, ironmlx
    /// picks a model-aware default from local benchmark policy.
    #[arg(long = "mtp-draft-tokens")]
    pub mtp_draft_tokens: Option<usize>,

    /// Enable request-local greedy prompt lookup speculative decoding.
    #[arg(long = "prompt-lookup", default_value_t = false)]
    pub prompt_lookup: bool,

    /// Minimum n-gram length considered by --prompt-lookup.
    #[arg(long = "prompt-lookup-min-ngram")]
    pub prompt_lookup_min_ngram: Option<usize>,

    /// Maximum n-gram length considered by --prompt-lookup.
    #[arg(long = "prompt-lookup-max-ngram")]
    pub prompt_lookup_max_ngram: Option<usize>,

    /// Maximum copied draft tokens per PromptLookup verification window.
    #[arg(long = "prompt-lookup-max-draft-tokens")]
    pub prompt_lookup_max_draft_tokens: Option<usize>,

    /// Per-request committed-token history retained by PromptLookup.
    #[arg(long = "prompt-lookup-history-window-tokens")]
    pub prompt_lookup_history_window_tokens: Option<usize>,

    /// Maximum distinct n-gram keys retained per request.
    #[arg(long = "prompt-lookup-max-index-entries")]
    pub prompt_lookup_max_index_entries: Option<usize>,

    /// KV cache quantization used by attention reads: none, turbo3, turbo4, or k3v4.
    #[arg(long = "kv-quant", value_enum, default_value = "none")]
    pub(crate) kv_quant: KvQuantArg,

    /// Enable paged SSD prefix cache under this directory. Without --kv-quant,
    /// this also switches full-attention KV caches to paged storage and decode
    /// to the paged attention kernel when supported. With TurboQuant, runtime
    /// K/V stays quantized while prefix cache entries are persisted as packed
    /// TurboQuant tensors. When passed without a value, defaults to
    /// ~/.ironmlx/cache/paged_prefix_cache.
    #[arg(
        long = "paged-prefix-cache-dir",
        num_args = 0..=1,
        default_missing_value = DEFAULT_PAGED_PREFIX_CACHE_DIR
    )]
    pub paged_prefix_cache_dir: Option<PathBuf>,

    /// Tokens per physical K/V page for --paged-prefix-cache-dir.
    #[arg(long = "paged-prefix-cache-block-size", default_value_t = DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE)]
    pub paged_prefix_cache_block_size: i32,

    /// Maximum physical pages per full-attention layer cache. If omitted,
    /// defaults to ceil(b_max * max_cache_cap / block_size).
    #[arg(long = "paged-prefix-cache-max-pages")]
    pub paged_prefix_cache_max_pages: Option<i32>,

    /// Maximum SSD prefix cache directory size in GiB. If omitted, the SSD
    /// directory is not pruned by size.
    #[arg(long = "ssd-prefix-cache-max-gb")]
    pub ssd_prefix_cache_max_gb: Option<usize>,

    /// Maximum bytes for the in-process cross-request prefix LRU cache. Disabled
    /// by default; initial support requires --paged-prefix-cache-dir so L1 can
    /// share the same paged prefix cache key and restore semantics.
    #[arg(long = "prefix-lru-cache-max-bytes")]
    pub prefix_lru_cache_max_bytes: Option<usize>,

    /// Enable experimental Active KV Cache offload. Eligible decode requests may
    /// still be parked to SSD when the scheduler is full; paged full-attention
    /// KV caches also use transparent hot/cold page residency so older decode
    /// pages can be offloaded and streamed back in chunks during attention.
    #[arg(long = "active-kv-offload", default_value_t = false)]
    pub active_kv_offload: bool,

    /// Directory used by --active-kv-offload for temporary request KV payloads.
    /// Defaults to ~/.ironmlx/cache/active_kv_offload when offload is enabled.
    #[arg(long = "active-kv-offload-dir")]
    pub active_kv_offload_dir: Option<PathBuf>,
}

pub(crate) fn resolve_paged_prefix_cache_config(
    args: &ServeArgs,
    scheduler_config: SchedulerServeConfig,
    model_id: &str,
) -> Result<Option<crate::core::cache::PagedPrefixCacheConfig>> {
    let Some(root) = args.paged_prefix_cache_dir.as_ref() else {
        return Ok(None);
    };
    let root = expand_home_path(root)?;
    let block_size = args.paged_prefix_cache_block_size;
    if block_size <= 0 {
        bail!("--paged-prefix-cache-block-size must be > 0");
    }
    let max_pages = match args.paged_prefix_cache_max_pages {
        Some(max_pages) => {
            if max_pages <= 0 {
                bail!("--paged-prefix-cache-max-pages must be > 0");
            }
            max_pages
        }
        None => {
            let tokens = scheduler_config
                .max_cache_cap
                .saturating_mul(scheduler_config.b_max);
            let pages = tokens.div_ceil(block_size as usize).max(1);
            i32::try_from(pages).context("derived paged prefix cache max_pages exceeds i32")?
        }
    };
    let max_disk_bytes = resolve_ssd_prefix_cache_max_bytes(args)?;
    crate::core::cache::PagedPrefixCacheConfig::new_with_max_disk_bytes(
        root,
        model_id.to_string(),
        block_size,
        max_pages,
        max_disk_bytes,
    )
    .map(Some)
}

fn resolve_ssd_prefix_cache_max_bytes(args: &ServeArgs) -> Result<Option<usize>> {
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

pub(crate) fn resolve_memory_limit_bytes(
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

pub(crate) fn resolve_engine_paged_prefix_cache_settings(
    args: &ServeArgs,
) -> Result<Option<server::engine::EnginePagedPrefixCacheSettings>> {
    let Some(root) = args.paged_prefix_cache_dir.as_ref() else {
        return Ok(None);
    };
    let root = expand_home_path(root)?;
    let block_size = args.paged_prefix_cache_block_size;
    if block_size <= 0 {
        bail!("--paged-prefix-cache-block-size must be > 0");
    }
    if let Some(max_pages) = args.paged_prefix_cache_max_pages {
        if max_pages <= 0 {
            bail!("--paged-prefix-cache-max-pages must be > 0");
        }
    }
    let max_disk_bytes = resolve_ssd_prefix_cache_max_bytes(args)?;
    Ok(Some(server::engine::EnginePagedPrefixCacheSettings {
        root,
        block_size,
        max_pages: args.paged_prefix_cache_max_pages,
        max_disk_bytes,
    }))
}

pub(crate) fn resolve_prefix_lru_cache_config(
    args: &ServeArgs,
    paged_prefix_cache: Option<&crate::core::cache::PagedPrefixCacheConfig>,
) -> Result<Option<crate::core::cache::PrefixLruCacheConfig>> {
    let Some(max_bytes) = args.prefix_lru_cache_max_bytes else {
        return Ok(None);
    };
    if paged_prefix_cache.is_none() {
        bail!("--prefix-lru-cache-max-bytes requires --paged-prefix-cache-dir");
    }
    crate::core::cache::PrefixLruCacheConfig::new(max_bytes).map(Some)
}

pub(crate) fn resolve_model_ttl(args: &ServeArgs) -> Result<Option<Duration>> {
    let Some(minutes) = args.model_ttl_minutes else {
        return Ok(None);
    };
    if minutes == 0 {
        return Ok(None);
    }
    minutes
        .checked_mul(60)
        .map(Duration::from_secs)
        .context("--model-ttl-minutes exceeds supported duration")
        .map(Some)
}

pub(crate) fn resolve_active_kv_offload_config(
    args: &ServeArgs,
) -> Result<crate::core::cache::ActiveKvOffloadConfig> {
    if !args.active_kv_offload {
        return Ok(crate::core::cache::ActiveKvOffloadConfig::disabled());
    }
    let root = match args.active_kv_offload_dir.as_ref() {
        Some(root) => expand_home_path(root)?,
        None => crate::core::cache::default_active_kv_offload_dir(),
    };
    Ok(crate::core::cache::ActiveKvOffloadConfig::enabled(root))
}

fn expand_home_path(path: &Path) -> Result<PathBuf> {
    let Some(raw) = path.to_str() else {
        return Ok(path.to_path_buf());
    };
    let Some(rest) = raw.strip_prefix('~') else {
        return Ok(path.to_path_buf());
    };
    if !rest.is_empty() && !rest.starts_with('/') {
        return Ok(path.to_path_buf());
    }
    let home = dirs::home_dir().context("locating home directory for ~")?;
    Ok(home.join(rest.strip_prefix('/').unwrap_or(rest)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SchedulerServeConfig {
    pub(crate) prefill_chunk_size: usize,
    pub(crate) b_max: usize,
    pub(crate) admission_deadline_ms: u64,
    pub(crate) admission_queue_max: usize,
    pub(crate) max_cache_cap: usize,
    pub(crate) decode_cadence_mid_chunk_cap: usize,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ServeMtpConfig {
    model_dir: PathBuf,
    draft_tokens: usize,
}

fn resolve_prompt_lookup_config(
    args: &ServeArgs,
) -> Result<Option<crate::core::prompt_lookup::PromptLookupConfig>> {
    let has_source_params = args.prompt_lookup_min_ngram.is_some()
        || args.prompt_lookup_max_ngram.is_some()
        || args.prompt_lookup_max_draft_tokens.is_some()
        || args.prompt_lookup_history_window_tokens.is_some()
        || args.prompt_lookup_max_index_entries.is_some();
    if !args.prompt_lookup {
        if has_source_params {
            bail!("prompt lookup source parameters require --prompt-lookup");
        }
        return Ok(None);
    }
    if args.mtp_model_dir.is_some() || args.mtp_draft_tokens.is_some() {
        bail!("--prompt-lookup is mutually exclusive with neural MTP/drafter options");
    }
    let defaults = crate::core::prompt_lookup::PromptLookupConfig::default();
    Ok(Some(
        crate::core::prompt_lookup::PromptLookupConfig {
            min_ngram: args.prompt_lookup_min_ngram.unwrap_or(defaults.min_ngram),
            max_ngram: args.prompt_lookup_max_ngram.unwrap_or(defaults.max_ngram),
            max_draft_tokens: args
                .prompt_lookup_max_draft_tokens
                .unwrap_or(defaults.max_draft_tokens),
            history_window_tokens: args
                .prompt_lookup_history_window_tokens
                .unwrap_or(defaults.history_window_tokens),
            max_index_entries: args
                .prompt_lookup_max_index_entries
                .unwrap_or(defaults.max_index_entries),
        }
        .validate()?,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QwenMoeServeModel {
    Qwen35,
    Qwen36,
}

fn qwen_moe_serve_model(raw_config: &serde_json::Value) -> QwenMoeServeModel {
    if crate::models::is_qwen36_moe_config(raw_config) {
        QwenMoeServeModel::Qwen36
    } else {
        QwenMoeServeModel::Qwen35
    }
}

fn default_scheduler_profile_config() -> SchedulerAutotuneProfileConfig {
    SchedulerAutotuneProfileConfig {
        b_max: DEFAULT_B_MAX,
        prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
        admission_deadline_ms: DEFAULT_ADMISSION_DEADLINE_MS,
        admission_queue_max: DEFAULT_ADMISSION_QUEUE_MAX,
        max_cache_cap: DEFAULT_MAX_CACHE_CAP,
        decode_cadence_mid_chunk_cap: DEFAULT_DECODE_CADENCE_MID_CHUNK_CAP,
    }
}

fn default_scheduler_runtime_profile(
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

fn read_scheduler_runtime_profile(path: &Path) -> Result<SchedulerAutotuneRuntimeProfile> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

#[derive(Debug)]
struct SchedulerProfileLoad {
    path: PathBuf,
    profile: SchedulerAutotuneRuntimeProfile,
    auto_loaded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SchedulerProfileSource {
    Explicit,
    Store,
}

#[derive(Debug)]
pub(crate) struct ResolvedSchedulerRuntime {
    pub(crate) scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    pub(crate) scheduler_config: SchedulerServeConfig,
    pub(crate) profile_source: Option<SchedulerProfileSource>,
}

fn adaptive_mtp_physical_b_max(
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

pub(crate) fn apply_adaptive_mtp_scheduler_defaults(
    args: &ServeArgs,
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

fn load_scheduler_profile_for_model(
    args: &ServeArgs,
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

fn load_scheduler_profile_for_model_with_explicit(
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

fn check_loaded_scheduler_profile_health(
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

fn log_scheduler_profile_health(
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

fn scheduler_profile_model_name(model_dir: &Path) -> Result<String> {
    model_dir
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            anyhow::anyhow!("--model has no directory name for scheduler profile lookup")
        })
}

fn unix_time_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_millis();
    millis.min(u128::from(u64::MAX)) as u64
}

fn scheduler_runtime_context_for_model(
    args: &ServeArgs,
    model_dir: &Path,
    mtp_model_dir: Option<&Path>,
    mtp_draft_tokens: Option<usize>,
    prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
    logical_kv_cap_tokens: Option<usize>,
) -> Result<SchedulerAutotuneRuntimeContext> {
    if prompt_lookup.is_some() && mtp_model_dir.is_some() {
        bail!("PromptLookup is mutually exclusive with neural MTP/drafter options");
    }
    let logical_kv_cap_tokens = logical_kv_cap_tokens
        .or(args.max_cache_cap)
        .unwrap_or(DEFAULT_MAX_CACHE_CAP);
    build_scheduler_runtime_context(
        model_dir,
        SchedulerProfileContextOptions {
            mtp_model_dir,
            mtp_draft_tokens,
            prompt_lookup,
            kv_quantization: args.kv_quant.profile_context(),
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

fn apply_scheduler_cli_overrides(
    args: &ServeArgs,
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

fn validate_scheduler_serve_config(config: SchedulerAutotuneProfileConfig) -> Result<()> {
    if config.b_max == 0 {
        bail!("scheduler b_max must be >= 1");
    }
    if config.decode_cadence_mid_chunk_cap == 0 {
        bail!("scheduler decode_cadence_mid_chunk_cap must be >= 1");
    }
    Ok(())
}

fn validate_dynamic_rules(profile: &SchedulerAutotuneRuntimeProfile) -> Result<()> {
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

fn resolve_serve_mtp_config(
    args: &ServeArgs,
    architecture: crate::models::ModelArchitecture,
    raw_config: &serde_json::Value,
    _scheduler_config: SchedulerServeConfig,
) -> Result<Option<ServeMtpConfig>> {
    let Some(model_dir) = args.mtp_model_dir.as_ref() else {
        return Ok(None);
    };
    match architecture {
        crate::models::ModelArchitecture::Qwen35Dense
        | crate::models::ModelArchitecture::Qwen35Moe
        | crate::models::ModelArchitecture::Gemma4 => {}
        _ => bail!("ironmlx serve --mtp-model-dir currently supports Qwen/Gemma4 models only"),
    }
    if !model_dir.exists() {
        bail!(
            "--mtp-model-dir must point to a local directory (got '{}')",
            model_dir.display()
        );
    }
    let draft_tokens = crate::core::speculative::resolve_mtp_draft_tokens(
        raw_config,
        args.mtp_draft_tokens
            .map(crate::core::speculative::MtpDraftTokensArg::Explicit)
            .unwrap_or(crate::core::speculative::MtpDraftTokensArg::Omitted),
    );
    crate::core::speculative::MtpSpeculativeConfig::new(
        draft_tokens,
        crate::core::sampler::Sampler::greedy(),
    )?;
    Ok(Some(ServeMtpConfig {
        model_dir: model_dir.clone(),
        draft_tokens,
    }))
}

fn resolve_scheduler_runtime_profile(
    args: &ServeArgs,
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
    resolved.config = apply_scheduler_cli_overrides(args, resolved.config);
    for rule in &mut resolved.rules {
        rule.config = apply_scheduler_cli_overrides(args, rule.config);
    }
    validate_scheduler_serve_config(resolved.config)?;
    validate_dynamic_rules(&resolved)?;
    Ok(resolved)
}
#[cfg(test)]
fn resolve_scheduler_serve_config(
    args: &ServeArgs,
    profile: Option<&SchedulerAutotuneRuntimeProfile>,
) -> Result<SchedulerServeConfig> {
    let profile = resolve_scheduler_runtime_profile(
        args,
        profile,
        &SchedulerAutotuneRuntimeContext::local_default(DEFAULT_MAX_CACHE_CAP),
    )?;
    Ok(SchedulerServeConfig {
        prefill_chunk_size: profile.config.prefill_chunk_size,
        b_max: profile.config.b_max,
        admission_deadline_ms: profile.config.admission_deadline_ms,
        admission_queue_max: profile.config.admission_queue_max,
        max_cache_cap: profile.config.max_cache_cap,
        decode_cadence_mid_chunk_cap: profile.config.decode_cadence_mid_chunk_cap,
    })
}

pub(crate) fn resolve_scheduler_for_model(
    args: &ServeArgs,
    model_dir: &Path,
) -> Result<ResolvedSchedulerRuntime> {
    resolve_scheduler_for_model_with_speculative(
        args,
        model_dir,
        args.mtp_model_dir.as_deref(),
        args.mtp_draft_tokens,
        resolve_prompt_lookup_config(args)?,
        None,
    )
}

pub(crate) fn resolve_scheduler_for_model_with_speculative(
    args: &ServeArgs,
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
        match SchedulerProfileStore::default() {
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

/// Generic serve helper — shared by all model types that satisfy the
/// `SchedulerActor<M>` / `AppState<M>` bounds.
///
/// The `DenseVlMethods` bound is required by `server::serve<M>` /
/// `SchedulerActor<M>`. The trait name is historical; both dense and MoE
/// Qwen3.5 variants implement it so the same OpenAI VL route can serve either
/// checkpoint family.
fn log_scheduler_mode(scheduler_config: SchedulerServeConfig) {
    // Surface b_max at boot so operators can confirm whether single-request
    // optimized mode (default) or multi-request batching is active without
    // having to inspect process args.
    if scheduler_config.b_max == 1 {
        tracing::info!(
            "ironmlx serve: b_max=1 (single-request optimized mode; \
             pass --max-sequences N > 1 to enable concurrent multi-request batching)"
        );
    } else {
        tracing::info!(
            "ironmlx serve: b_max={} (multi-request batching enabled; \
             pass --max-sequences 1 to switch to single-request optimized mode)",
            scheduler_config.b_max,
        );
    }
}

fn serve_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio::Runtime::new")
}

fn single_model_id(args: &ServeArgs) -> Result<String> {
    args.model
        .clone()
        .context("--model is required when --model-manifest is not set")
}

fn serve_with_model<M>(
    model: M,
    tokenizer: Tokenizer,
    args: &ServeArgs,
    scheduler_config: SchedulerServeConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    vision_input: Option<server::VisionInputConfig>,
    static_memory_estimate: StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    log_scheduler_mode(scheduler_config);

    let model_id = single_model_id(args)?;
    let paged_prefix_cache = resolve_paged_prefix_cache_config(args, scheduler_config, &model_id)?;
    let prefix_lru_cache = resolve_prefix_lru_cache_config(args, paged_prefix_cache.as_ref())?;
    let active_kv_offload = resolve_active_kv_offload_config(args)?;
    let prompt_lookup = resolve_prompt_lookup_config(args)?;
    if let Some(config) = &paged_prefix_cache {
        tracing::info!(
            "ironmlx serve: paged SSD prefix cache enabled dir={} block_size={} max_pages={}",
            config.root.display(),
            config.block_size,
            config.max_pages
        );
    }
    if let Some(config) = &prefix_lru_cache {
        tracing::info!(
            "ironmlx serve: prefix LRU cache enabled max_bytes={}",
            config.max_bytes
        );
    }
    let runtime = serve_runtime()?;
    if let Some(cfg) = prompt_lookup {
        tracing::info!(
            min_ngram = cfg.min_ngram,
            max_ngram = cfg.max_ngram,
            max_draft_tokens = cfg.max_draft_tokens,
            history_window_tokens = cfg.history_window_tokens,
            max_index_entries = cfg.max_index_entries,
            "ironmlx serve: request-local PromptLookup enabled"
        );
        runtime.block_on(server::serve_with_prompt_lookup(
            model,
            cfg,
            tokenizer,
            model_id,
            &args.host,
            args.port,
            scheduler_config.prefill_chunk_size,
            scheduler_config.b_max,
            scheduler_config.admission_deadline_ms,
            scheduler_config.admission_queue_max,
            scheduler_config.max_cache_cap,
            scheduler_config.decode_cadence_mid_chunk_cap,
            args.kv_quant.turboquant_bits(),
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
            scheduler_runtime_profile,
            args.scheduler_autotune_report,
            vision_input,
            static_memory_estimate,
        ))
    } else {
        runtime.block_on(server::serve(
            model,
            tokenizer,
            model_id,
            &args.host,
            args.port,
            scheduler_config.prefill_chunk_size,
            scheduler_config.b_max,
            scheduler_config.admission_deadline_ms,
            scheduler_config.admission_queue_max,
            scheduler_config.max_cache_cap,
            scheduler_config.decode_cadence_mid_chunk_cap,
            args.kv_quant.turboquant_bits(),
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
            scheduler_runtime_profile,
            args.scheduler_autotune_report,
            vision_input,
            static_memory_estimate,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn serve_with_mtp_model<M>(
    model: M,
    tokenizer: Tokenizer,
    mtp_config: ServeMtpConfig,
    args: &ServeArgs,
    scheduler_config: SchedulerServeConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    vision_input: Option<server::VisionInputConfig>,
    mut static_memory_estimate: StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    log_scheduler_mode(scheduler_config);

    let mtp_loader = Loader::open_mtp(&mtp_config.model_dir)
        .with_context(|| format!("Loader::open_mtp {}", mtp_config.model_dir.display()))?;
    let mtp = model
        .load_mtp_head(&mtp_loader)
        .with_context(|| format!("loading MTP head from {}", mtp_config.model_dir.display()))?;
    static_memory_estimate.speculative_cold_bytes = mtp_loader.loaded_tensor_bytes();

    let model_id = single_model_id(args)?;
    let paged_prefix_cache = resolve_paged_prefix_cache_config(args, scheduler_config, &model_id)?;
    let effective_draft_tokens =
        crate::core::speculative::effective_mtp_draft_tokens_for_paged_prefix(
            mtp_config.draft_tokens,
            paged_prefix_cache.is_some(),
        );
    tracing::info!(
        "ironmlx serve: MTP enabled model_dir={} draft_tokens={} requested_draft_tokens={}",
        mtp_config.model_dir.display(),
        effective_draft_tokens,
        mtp_config.draft_tokens
    );
    let prefix_lru_cache = resolve_prefix_lru_cache_config(args, paged_prefix_cache.as_ref())?;
    let active_kv_offload = resolve_active_kv_offload_config(args)?;
    if let Some(config) = &prefix_lru_cache {
        tracing::info!(
            "ironmlx serve: prefix LRU cache enabled max_bytes={}",
            config.max_bytes
        );
    }
    let runtime = serve_runtime()?;
    runtime.block_on(server::serve_with_mtp(
        model,
        mtp,
        mtp_config.draft_tokens,
        tokenizer,
        model_id,
        &args.host,
        args.port,
        scheduler_config.prefill_chunk_size,
        scheduler_config.b_max,
        scheduler_config.admission_deadline_ms,
        scheduler_config.admission_queue_max,
        scheduler_config.max_cache_cap,
        scheduler_config.decode_cadence_mid_chunk_cap,
        args.kv_quant.turboquant_bits(),
        paged_prefix_cache,
        prefix_lru_cache,
        active_kv_offload,
        scheduler_runtime_profile,
        args.scheduler_autotune_report,
        vision_input,
        static_memory_estimate,
    ))
}

#[allow(clippy::too_many_arguments)]
fn serve_with_gemma4_drafter_model(
    model: crate::models::Gemma4Model,
    tokenizer: Tokenizer,
    model_dir: &Path,
    mut static_memory_estimate: StaticMemoryEstimate,
    mtp_config: ServeMtpConfig,
    args: &ServeArgs,
    scheduler_config: SchedulerServeConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    vision_input: Option<server::VisionInputConfig>,
) -> Result<()> {
    log_scheduler_mode(scheduler_config);
    tracing::info!(
        "ironmlx serve: Gemma4 drafter enabled model_dir={} requested_draft_tokens={}",
        mtp_config.model_dir.display(),
        mtp_config.draft_tokens
    );

    let validation = server::model_manager::validate_mtp_pair(
        model_dir,
        &mtp_config.model_dir,
        Some(mtp_config.draft_tokens),
    )?;
    if !validation.compatible {
        bail!(
            "Gemma4 drafter validation failed: {}: {}",
            validation.reason_code,
            validation.message
        );
    }

    let drafter_loader = Loader::open_gemma4_drafter(&mtp_config.model_dir).with_context(|| {
        format!(
            "Loader::open_gemma4_drafter {}",
            mtp_config.model_dir.display()
        )
    })?;
    static_memory_estimate.speculative_cold_bytes = drafter_loader.loaded_tensor_bytes();
    let drafter = crate::models::gemma4::Gemma4AssistantModel::from_loader(&drafter_loader)
        .with_context(|| {
            format!(
                "loading Gemma4 assistant drafter from {}",
                mtp_config.model_dir.display()
            )
        })?;

    let model_id = single_model_id(args)?;
    let paged_prefix_cache = resolve_paged_prefix_cache_config(args, scheduler_config, &model_id)?;
    let prefix_lru_cache = resolve_prefix_lru_cache_config(args, paged_prefix_cache.as_ref())?;
    let active_kv_offload = resolve_active_kv_offload_config(args)?;
    if let Some(config) = &prefix_lru_cache {
        tracing::info!(
            "ironmlx serve: prefix LRU cache enabled max_bytes={}",
            config.max_bytes
        );
    }
    let runtime = serve_runtime()?;
    runtime.block_on(server::serve_with_gemma4_drafter(
        model,
        drafter,
        mtp_config.draft_tokens,
        tokenizer,
        model_id,
        &args.host,
        args.port,
        scheduler_config.prefill_chunk_size,
        scheduler_config.b_max,
        scheduler_config.admission_deadline_ms,
        scheduler_config.admission_queue_max,
        scheduler_config.max_cache_cap,
        scheduler_config.decode_cadence_mid_chunk_cap,
        args.kv_quant.turboquant_bits(),
        paged_prefix_cache,
        prefix_lru_cache,
        active_kv_offload,
        scheduler_runtime_profile,
        args.scheduler_autotune_report,
        vision_input,
        static_memory_estimate,
    ))
}

fn serve_with_diffusion_gemma_model(
    model: crate::models::DiffusionGemmaModel,
    tokenizer: Tokenizer,
    generation_config: crate::models::DiffusionGemmaGenerationConfig,
    model_weight_bytes: usize,
    args: &ServeArgs,
    vision_input: server::VisionInputConfig,
) -> Result<()> {
    let model_id = single_model_id(args)?;
    let runtime = serve_runtime()?;
    runtime.block_on(server::diffusion_gemma::serve_diffusion_gemma(
        model,
        tokenizer,
        generation_config,
        model_id,
        model_weight_bytes,
        &args.host,
        args.port,
        vision_input,
    ))
}

pub(crate) fn read_model_type(model_dir: &std::path::Path) -> Result<String> {
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

fn read_engine_pool_manifest(path: &Path) -> Result<server::engine::EnginePoolManifest> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

struct EnginePoolSchedulerProfileRequest<'a> {
    manifest_profile: Option<&'a Path>,
    store: Option<&'a SchedulerProfileStore>,
    hardware_label: &'a str,
    mtp_model_dir: Option<&'a Path>,
    mtp_draft_tokens: Option<usize>,
    prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
}

fn resolve_engine_pool_scheduler_profile(
    args: &ServeArgs,
    model_dir: &Path,
    request: EnginePoolSchedulerProfileRequest<'_>,
) -> Result<ResolvedSchedulerRuntime> {
    let EnginePoolSchedulerProfileRequest {
        manifest_profile,
        store,
        hardware_label,
        mtp_model_dir,
        mtp_draft_tokens,
        prompt_lookup,
    } = request;
    let runtime_context = scheduler_runtime_context_for_model(
        args,
        model_dir,
        mtp_model_dir,
        mtp_draft_tokens,
        prompt_lookup,
        None,
    )?;
    let runtime_context_fingerprint = runtime_context.fingerprint();
    let explicit_profile = manifest_profile.or(args.scheduler_profile.as_deref());
    let mut scheduler_profile_load = load_scheduler_profile_for_model_with_explicit(
        explicit_profile,
        model_dir,
        store,
        hardware_label,
        &runtime_context_fingerprint,
    )?;
    let scheduler_profile_model_name = scheduler_profile_model_name(model_dir)?;
    let mut discard_auto_profile = false;
    if let Some(load) = scheduler_profile_load.as_ref() {
        match check_loaded_scheduler_profile_health(
            &load.profile,
            &scheduler_profile_model_name,
            hardware_label,
            &runtime_context,
            unix_time_ms(),
        ) {
            Ok(report) => log_scheduler_profile_health(&load.path, &report),
            Err(error) if load.auto_loaded => {
                tracing::warn!(
                    "ironmlx serve: scheduler profile ignored path={} model_name={} hardware_label={} error={:#}; using CLI/default scheduler config",
                    load.path.display(),
                    scheduler_profile_model_name,
                    hardware_label,
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

fn build_engine_model_config_for_pool(
    args: &ServeArgs,
    model: server::engine::EngineModelManifest,
    scheduler_profile_store: Option<&SchedulerProfileStore>,
    hardware_label: &str,
) -> Result<server::engine::EngineModelConfig> {
    let prompt_lookup = model
        .prompt_lookup
        .map(crate::core::prompt_lookup::PromptLookupConfig::validate)
        .transpose()?;
    let mtp = model
        .mtp_model_dir
        .map(|model_dir| server::engine::EngineMtpSettings {
            model_dir,
            draft_tokens: model.mtp_draft_tokens,
        });
    if model.load_policy == server::engine::EngineLoadPolicy::Disabled {
        return Ok(server::engine::EngineModelConfig {
            id: model.id,
            path: model.path,
            load_policy: model.load_policy,
            default: model.default,
            pinned: false,
            scheduler_runtime_profile: default_scheduler_runtime_profile(
                SchedulerAutotuneRuntimeContext::local_default(DEFAULT_MAX_CACHE_CAP),
            ),
            mtp,
            prompt_lookup,
            sampling_defaults: server::SamplingDefaults::default(),
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
    if mtp.is_some() && prompt_lookup.is_some() {
        bail!(
            "engine model `{}` configures both neural MTP/drafter and PromptLookup",
            model.id
        );
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
    let model_type = read_model_type(&model.path)?;
    let architecture = crate::models::ModelArchitecture::from_model_type(&model_type)?;
    if apply_adaptive_mtp_scheduler_defaults(args, architecture, mtp.is_some(), &mut resolved) {
        tracing::info!(
            "ironmlx serve: adaptive MTP scheduler default applied manifest_model={} b_max={}",
            model.id,
            resolved.scheduler_config.b_max
        );
    }
    Ok(server::engine::EngineModelConfig {
        id: model.id,
        path: model.path,
        load_policy: model.load_policy,
        default: model.default,
        pinned: false,
        scheduler_runtime_profile: resolved.scheduler_runtime_profile,
        mtp,
        prompt_lookup,
        sampling_defaults: server::SamplingDefaults::default(),
    })
}

fn run_engine_pool(args: ServeArgs, manifest_path: &Path) -> Result<()> {
    if args.mtp_model_dir.is_some() || args.mtp_draft_tokens.is_some() {
        bail!("--model-manifest uses per-model mtp_model_dir / mtp_draft_tokens entries; do not pass global MTP flags");
    }
    if resolve_prompt_lookup_config(&args)?.is_some() {
        bail!("--model-manifest uses per-model prompt_lookup entries; do not pass global PromptLookup flags");
    }
    if args.prefix_lru_cache_max_bytes.is_some() && args.paged_prefix_cache_dir.is_none() {
        bail!("--prefix-lru-cache-max-bytes requires --paged-prefix-cache-dir");
    }

    let manifest = read_engine_pool_manifest(manifest_path)?;
    let _registry = server::engine::EngineRegistry::new(manifest.clone())?;
    let scheduler_profile_store = if args.scheduler_profile.is_none() {
        match SchedulerProfileStore::default() {
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
    let hardware_label = detect_scheduler_profile_hardware_label();
    let mut models = Vec::with_capacity(manifest.models.len());
    for model in manifest.models {
        models.push(build_engine_model_config_for_pool(
            &args,
            model,
            scheduler_profile_store.as_ref(),
            &hardware_label,
        )?);
    }
    let paged_prefix_cache = resolve_engine_paged_prefix_cache_settings(&args)?;
    let active_kv_offload = resolve_active_kv_offload_config(&args)?;
    let model_ttl = resolve_model_ttl(&args)?;
    let runtime_config = server::engine::EnginePoolRuntimeConfig {
        host: args.host,
        port: args.port,
        kv_cache_turboquant_bits: args.kv_quant.turboquant_bits(),
        scheduler_autotune_report: args.scheduler_autotune_report,
        paged_prefix_cache,
        prefix_lru_cache_max_bytes: args.prefix_lru_cache_max_bytes,
        model_ttl,
        memory_limits: server::engine::EnginePoolMemoryLimits {
            total_memory_limit_bytes: resolve_memory_limit_bytes(
                args.memory_limit_total_gb,
                "--memory-limit-total-gb",
            )?,
            model_memory_limit_bytes: resolve_memory_limit_bytes(
                args.memory_limit_model_gb,
                "--memory-limit-model-gb",
            )?,
        },
        active_kv_offload,
    };
    let config = server::engine::EnginePoolConfig {
        default_model: manifest.default_model,
        max_loaded_models: manifest.max_loaded_models,
        models,
    };
    let runtime = serve_runtime()?;
    runtime.block_on(server::engine::serve_engine_pool(config, runtime_config))
}

fn run_app_daemon(args: ServeArgs) -> Result<()> {
    tracing::info!(
        "ironmlx serve: starting app daemon mode on {}:{}",
        args.host,
        args.port
    );
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio::Runtime::new")?;
    runtime.block_on(server::model_manager::serve_app_daemon(args))
}

pub fn run(args: ServeArgs) -> Result<()> {
    if let Some(manifest_path) = args.model_manifest.clone() {
        return run_engine_pool(args, &manifest_path);
    }

    let Some(model_arg) = args.model.as_deref() else {
        return run_app_daemon(args);
    };

    let model_dir = PathBuf::from(model_arg);
    if !model_dir.exists() {
        return Err(anyhow::anyhow!(
            "--model must point to a local directory (got '{}'); HF hub auto-download is deferred",
            model_arg
        ));
    }

    let mut resolved_scheduler = resolve_scheduler_for_model(&args, &model_dir)?;

    let model_type = read_model_type(&model_dir)?;
    let architecture = crate::models::ModelArchitecture::from_model_type(&model_type)?;
    // open_multimodal so Qwen VL checkpoints retain vision_tower.* keys.
    let loader = Loader::open_multimodal(&model_dir).context("Loader::open_multimodal")?;
    let component_bytes = loader.loaded_tensor_component_bytes();
    let static_memory_estimate = StaticMemoryEstimate {
        text_cold_bytes: component_bytes.text,
        vision_cold_bytes: component_bytes.vision,
        speculative_cold_bytes: 0,
    };
    let mtp_config = resolve_serve_mtp_config(
        &args,
        architecture,
        loader.config_raw_value(),
        resolved_scheduler.scheduler_config,
    )?;
    if apply_adaptive_mtp_scheduler_defaults(
        &args,
        architecture,
        mtp_config.is_some(),
        &mut resolved_scheduler,
    ) {
        tracing::info!(
            "ironmlx serve: adaptive MTP scheduler default applied b_max={}",
            resolved_scheduler.scheduler_config.b_max
        );
    }
    let ResolvedSchedulerRuntime {
        scheduler_runtime_profile,
        scheduler_config,
        ..
    } = resolved_scheduler;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let vision_input = match architecture {
        crate::models::ModelArchitecture::Gemma4 => {
            let cfg = crate::models::gemma4::Gemma4Config::from_loader(&loader)
                .context("Gemma4Config::from_loader")?;
            cfg.vision_config
                .map(|vision_config| server::VisionInputConfig::Gemma4 { vision_config })
        }
        crate::models::ModelArchitecture::MiniCpmV46 => {
            Some(server::VisionInputConfig::MiniCpmV46 {
                spatial_merge_size: 4,
            })
        }
        _ => None,
    };

    match architecture {
        crate::models::ModelArchitecture::Qwen35Dense => {
            let model = crate::models::Qwen35Model::from_loader(&loader)
                .context("Qwen35Model::from_loader")?;
            if let Some(mtp_config) = mtp_config.clone() {
                serve_with_mtp_model(
                    model,
                    tokenizer,
                    mtp_config,
                    &args,
                    scheduler_config,
                    scheduler_runtime_profile,
                    vision_input,
                    static_memory_estimate,
                )
            } else {
                serve_with_model(
                    model,
                    tokenizer,
                    &args,
                    scheduler_config,
                    scheduler_runtime_profile,
                    vision_input,
                    static_memory_estimate,
                )
            }
        }
        crate::models::ModelArchitecture::Qwen35Moe => {
            match qwen_moe_serve_model(loader.config_raw_value()) {
                QwenMoeServeModel::Qwen35 => {
                    let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                        .context("Qwen35MoeModel::from_loader")?;
                    if let Some(mtp_config) = mtp_config.clone() {
                        serve_with_mtp_model(
                            model,
                            tokenizer,
                            mtp_config,
                            &args,
                            scheduler_config,
                            scheduler_runtime_profile,
                            vision_input,
                            static_memory_estimate,
                        )
                    } else {
                        serve_with_model(
                            model,
                            tokenizer,
                            &args,
                            scheduler_config,
                            scheduler_runtime_profile,
                            vision_input,
                            static_memory_estimate,
                        )
                    }
                }
                QwenMoeServeModel::Qwen36 => {
                    let model = crate::models::Qwen36MoeModel::from_loader(&loader)
                        .context("Qwen36MoeModel::from_loader")?;
                    if let Some(mtp_config) = mtp_config.clone() {
                        serve_with_mtp_model(
                            model,
                            tokenizer,
                            mtp_config,
                            &args,
                            scheduler_config,
                            scheduler_runtime_profile,
                            vision_input,
                            static_memory_estimate,
                        )
                    } else {
                        serve_with_model(
                            model,
                            tokenizer,
                            &args,
                            scheduler_config,
                            scheduler_runtime_profile,
                            vision_input,
                            static_memory_estimate,
                        )
                    }
                }
            }
        }
        crate::models::ModelArchitecture::Gemma4 => {
            let model = crate::models::Gemma4Model::from_loader(&loader)
                .context("Gemma4Model::from_loader")?;
            if let Some(mtp_config) = mtp_config.clone() {
                serve_with_gemma4_drafter_model(
                    model,
                    tokenizer,
                    &model_dir,
                    static_memory_estimate,
                    mtp_config,
                    &args,
                    scheduler_config,
                    scheduler_runtime_profile,
                    vision_input,
                )
            } else {
                serve_with_model(
                    model,
                    tokenizer,
                    &args,
                    scheduler_config,
                    scheduler_runtime_profile,
                    vision_input,
                    static_memory_estimate,
                )
            }
        }
        crate::models::ModelArchitecture::Glm4MoeLite => {
            let model = crate::models::Glm4MoeLiteModel::from_loader(&loader)
                .context("Glm4MoeLiteModel::from_loader")?;
            serve_with_model(
                model,
                tokenizer,
                &args,
                scheduler_config,
                scheduler_runtime_profile,
                None,
                static_memory_estimate,
            )
        }
        crate::models::ModelArchitecture::Llama => {
            let model = crate::models::LlamaModel::from_loader(&loader)
                .context("LlamaModel::from_loader")?;
            serve_with_model(
                model,
                tokenizer,
                &args,
                scheduler_config,
                scheduler_runtime_profile,
                None,
                static_memory_estimate,
            )
        }
        crate::models::ModelArchitecture::MiniCpmV46 => {
            // MiniCpmV46Model serves text + single-image VL (vision_input set above).
            let model = crate::models::minicpmv4_6::model_from_loader(&loader)
                .context("minicpmv4_6::model_from_loader")?;
            serve_with_model(
                model,
                tokenizer,
                &args,
                scheduler_config,
                scheduler_runtime_profile,
                vision_input,
                static_memory_estimate,
            )
        }
        crate::models::ModelArchitecture::DiffusionGemma => {
            let cfg = crate::models::DiffusionGemmaConfig::from_loader(&loader)
                .context("DiffusionGemmaConfig::from_loader")?;
            let vision_config = cfg
                .vision_config
                .clone()
                .ok_or_else(|| anyhow::anyhow!("DiffusionGemma config has no vision_config"))?;
            let image_token_id = cfg.image_token_id;
            let generation_config =
                crate::models::DiffusionGemmaGenerationConfig::from_loader(&loader)
                    .context("DiffusionGemmaGenerationConfig::from_loader")?;
            let model = crate::models::DiffusionGemmaModel::from_loader(&loader)
                .context("DiffusionGemmaModel::from_loader")?;
            let model_weight_bytes = loader.loaded_tensor_bytes();
            serve_with_diffusion_gemma_model(
                model,
                tokenizer,
                generation_config,
                model_weight_bytes,
                &args,
                server::VisionInputConfig::DiffusionGemma {
                    vision_config,
                    image_token_id,
                },
            )
        }
    }
}

#[cfg(test)]
mod scheduler_profile_tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use clap::Parser;

    use crate::cli::scheduler_profile_store::SchedulerProfileStore;
    use crate::cli::Command;
    use crate::core::scheduler_autotune::{
        SchedulerAutotuneCacheState, SchedulerAutotuneProfileConfig,
        SchedulerAutotuneProfileHealthStatus, SchedulerAutotuneRuntimeContext,
        SchedulerAutotuneRuntimeProfile, SchedulerAutotuneRuntimeRule,
        SchedulerAutotuneRuntimeRuleCondition, SchedulerAutotuneScenario,
        SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
    };
    use crate::core::server::engine::{EngineLoadPolicy, EngineModelManifest};

    use super::{
        apply_adaptive_mtp_scheduler_defaults, build_engine_model_config_for_pool,
        check_loaded_scheduler_profile_health, default_scheduler_runtime_profile,
        load_scheduler_profile_for_model, qwen_moe_serve_model, read_engine_pool_manifest,
        resolve_active_kv_offload_config, resolve_memory_limit_bytes, resolve_model_ttl,
        resolve_paged_prefix_cache_config, resolve_prefix_lru_cache_config,
        resolve_prompt_lookup_config, resolve_scheduler_runtime_profile,
        resolve_scheduler_serve_config, resolve_serve_mtp_config, KvQuantArg, QwenMoeServeModel,
        ResolvedSchedulerRuntime, SchedulerProfileSource, SchedulerServeConfig, ServeArgs,
        BYTES_PER_GIB,
    };

    fn profile_config() -> SchedulerAutotuneProfileConfig {
        SchedulerAutotuneProfileConfig {
            b_max: 2,
            prefill_chunk_size: 1024,
            admission_deadline_ms: 7,
            admission_queue_max: 16,
            max_cache_cap: 8192,
            decode_cadence_mid_chunk_cap: 384,
        }
    }

    fn runtime_profile() -> SchedulerAutotuneRuntimeProfile {
        SchedulerAutotuneRuntimeProfile {
            schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
            model_name: "test-model".to_string(),
            hardware_label: "test-host".to_string(),
            runtime_context: runtime_context(),
            config: profile_config(),
            rules: vec![SchedulerAutotuneRuntimeRule {
                when: SchedulerAutotuneRuntimeRuleCondition {
                    prompt_len_gte: 8192,
                    max_new_tokens_gte: 512,
                    effective_concurrency_gte: 2,
                },
                config: SchedulerAutotuneProfileConfig {
                    prefill_chunk_size: 2048,
                    decode_cadence_mid_chunk_cap: 512,
                    ..profile_config()
                },
            }],
            metadata:
                crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata::synthetic(
                    1811606400000,
                ),
        }
    }

    fn runtime_context() -> SchedulerAutotuneRuntimeContext {
        SchedulerAutotuneRuntimeContext::local_default(8192)
    }

    fn base_args() -> ServeArgs {
        ServeArgs {
            model: Some("/tmp/model".to_string()),
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
            prompt_lookup: false,
            prompt_lookup_min_ngram: None,
            prompt_lookup_max_ngram: None,
            prompt_lookup_max_draft_tokens: None,
            prompt_lookup_history_window_tokens: None,
            prompt_lookup_max_index_entries: None,
            kv_quant: KvQuantArg::None,
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

    #[test]
    fn memory_limit_gigabytes_resolve_to_bytes() {
        assert_eq!(
            resolve_memory_limit_bytes(Some(2), "--memory-limit-total-gb").unwrap(),
            Some(2 * BYTES_PER_GIB)
        );
        assert_eq!(
            resolve_memory_limit_bytes(None, "--memory-limit-total-gb").unwrap(),
            None
        );
    }

    #[test]
    fn serve_engine_pool_manifest_parses_load_policies() {
        let temp_dir = unique_temp_dir("serve-engine-pool-manifest");
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let manifest_path = temp_dir.join("models.json");
        std::fs::write(
            &manifest_path,
            r#"{
                "default_model": "alpha",
                "max_loaded_models": 1,
                "models": [
                    {"id": "alpha", "path": "/models/alpha"},
                    {"id": "beta", "path": "/models/beta", "load_policy": "preload"}
                ]
            }"#,
        )
        .expect("write manifest");

        let manifest = read_engine_pool_manifest(&manifest_path).expect("manifest");

        assert_eq!(manifest.default_model.as_deref(), Some("alpha"));
        assert_eq!(manifest.max_loaded_models, Some(1));
        assert_eq!(manifest.models[0].id, "alpha");
        assert_eq!(manifest.models[0].load_policy, EngineLoadPolicy::Lazy);
        assert_eq!(manifest.models[1].load_policy, EngineLoadPolicy::Preload);
        std::fs::remove_dir_all(temp_dir).expect("cleanup");
    }

    #[test]
    fn serve_paged_prefix_cache_uses_engine_model_id_namespace() {
        let prefix_dir = unique_temp_dir("serve-prefix-engine-id");
        let mut args = base_args();
        args.paged_prefix_cache_dir = Some(prefix_dir.clone());

        let cfg = resolve_paged_prefix_cache_config(
            &args,
            SchedulerServeConfig {
                b_max: 2,
                max_cache_cap: 128,
                ..SchedulerServeConfig::default()
            },
            "manifest-alpha",
        )
        .expect("prefix config")
        .expect("enabled");

        assert_eq!(cfg.model_id, "manifest-alpha");
        std::fs::remove_dir_all(prefix_dir).ok();
    }

    #[test]
    fn serve_active_kv_offload_disabled_by_default() {
        let args = base_args();

        let cfg = resolve_active_kv_offload_config(&args).expect("active kv config");

        assert!(!cfg.enabled);
        assert_eq!(
            cfg.root,
            crate::core::cache::default_active_kv_offload_dir()
        );
    }

    #[test]
    fn serve_active_kv_offload_uses_default_dir_when_enabled() {
        let mut args = base_args();
        args.active_kv_offload = true;

        let cfg = resolve_active_kv_offload_config(&args).expect("active kv config");

        assert!(cfg.enabled);
        assert_eq!(
            cfg.root,
            crate::core::cache::default_active_kv_offload_dir()
        );
    }

    #[test]
    fn serve_active_kv_offload_uses_custom_dir() {
        let root = unique_temp_dir("serve-active-kv-root");
        let mut args = base_args();
        args.active_kv_offload = true;
        args.active_kv_offload_dir = Some(root.clone());

        let cfg = resolve_active_kv_offload_config(&args).expect("active kv config");

        assert!(cfg.enabled);
        assert_eq!(cfg.root, root);
    }

    #[test]
    fn serve_model_ttl_minutes_resolves_positive_duration() {
        let mut args = base_args();
        args.model_ttl_minutes = Some(30);

        let ttl = resolve_model_ttl(&args).expect("model ttl");

        assert_eq!(ttl, Some(std::time::Duration::from_secs(30 * 60)));
    }

    #[test]
    fn serve_model_ttl_minutes_zero_disables_ttl() {
        let mut args = base_args();
        args.model_ttl_minutes = Some(0);

        let ttl = resolve_model_ttl(&args).expect("model ttl");

        assert_eq!(ttl, None);
    }

    #[test]
    fn serve_engine_pool_skips_disabled_model_scheduler_profile_resolution() {
        let args = base_args();
        let manifest_model = EngineModelManifest {
            id: "disabled-exp".to_string(),
            path: PathBuf::from("/tmp/ironmlx-disabled-model-does-not-exist"),
            load_policy: EngineLoadPolicy::Disabled,
            default: false,
            scheduler_profile: Some(PathBuf::from(
                "/tmp/ironmlx-disabled-profile-does-not-exist.json",
            )),
            mtp_model_dir: None,
            mtp_draft_tokens: None,
            prompt_lookup: None,
        };

        let config = build_engine_model_config_for_pool(&args, manifest_model, None, "test-host")
            .expect("disabled models must not resolve scheduler profiles");

        assert_eq!(config.id, "disabled-exp");
        assert_eq!(config.load_policy, EngineLoadPolicy::Disabled);
    }

    #[test]
    fn serve_engine_pool_gemma4_drafter_default_scheduler_uses_bmax_four() {
        let temp_dir = unique_temp_dir("serve-engine-pool-gemma4-drafter");
        let model_dir = temp_dir.join("gemma4-base");
        let mtp_dir = temp_dir.join("gemma4-assistant");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        std::fs::create_dir_all(&mtp_dir).expect("create mtp dir");
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text"}}"#,
        )
        .expect("write config");
        std::fs::write(
            mtp_dir.join("config.json"),
            r#"{"model_type":"gemma4_assistant"}"#,
        )
        .expect("write assistant config");
        let args = base_args();
        let manifest_model = EngineModelManifest {
            id: "gemma4-manifest".to_string(),
            path: model_dir,
            load_policy: EngineLoadPolicy::Lazy,
            default: true,
            scheduler_profile: None,
            mtp_model_dir: Some(mtp_dir),
            mtp_draft_tokens: Some(2),
            prompt_lookup: None,
        };

        let config = build_engine_model_config_for_pool(&args, manifest_model, None, "test-host")
            .expect("build manifest model config");

        assert_eq!(config.scheduler_runtime_profile.config.b_max, 4);
        std::fs::remove_dir_all(temp_dir).expect("cleanup");
    }

    #[test]
    fn serve_engine_pool_preserves_prompt_lookup_source_config() {
        let temp_dir = unique_temp_dir("serve-engine-pool-prompt-lookup");
        let model_dir = temp_dir.join("qwen-base");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        std::fs::write(model_dir.join("config.json"), r#"{"model_type":"qwen3_5"}"#)
            .expect("write config");
        let args = base_args();
        let prompt_lookup = crate::core::prompt_lookup::PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 5,
            max_draft_tokens: 3,
            history_window_tokens: 4096,
            max_index_entries: 8192,
        };
        let manifest_model = EngineModelManifest {
            id: "qwen-prompt-lookup".to_string(),
            path: model_dir,
            load_policy: EngineLoadPolicy::Lazy,
            default: true,
            scheduler_profile: None,
            mtp_model_dir: None,
            mtp_draft_tokens: None,
            prompt_lookup: Some(prompt_lookup),
        };

        let config = build_engine_model_config_for_pool(&args, manifest_model, None, "test-host")
            .expect("build prompt lookup manifest config");

        assert_eq!(config.prompt_lookup, Some(prompt_lookup));
        assert_eq!(
            config
                .scheduler_runtime_profile
                .runtime_context
                .speculative
                .mode,
            crate::core::scheduler_autotune::SchedulerSpeculativeMode::PromptLookup
        );
        std::fs::remove_dir_all(temp_dir).expect("cleanup");
    }

    fn qwen36_dense_27b_raw_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 64
            }
        })
    }

    fn qwen36_moe_raw_config() -> serde_json::Value {
        let num_hidden_layers = 2;
        let mut quant = serde_json::Map::new();
        quant.insert("bits".to_owned(), serde_json::json!(4));
        quant.insert("group_size".to_owned(), serde_json::json!(64));
        quant.insert("mode".to_owned(), serde_json::json!("affine"));
        for layer in 0..num_hidden_layers {
            quant.insert(
                format!("language_model.model.layers.{layer}.mlp.gate"),
                serde_json::json!({"bits": 8, "group_size": 64}),
            );
            quant.insert(
                format!("language_model.model.layers.{layer}.mlp.shared_expert_gate"),
                serde_json::json!({"bits": 8, "group_size": 64}),
            );
        }
        serde_json::json!({
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "image_token_id": 248056,
            "vision_config": {},
            "text_config": {
                "num_hidden_layers": num_hidden_layers,
                "num_experts": 256,
                "num_experts_per_tok": 8
            },
            "quantization": serde_json::Value::Object(quant),
        })
    }

    #[test]
    fn serve_qwen_moe_dispatch_preserves_qwen36_checkpoint_identity() {
        assert_eq!(
            qwen_moe_serve_model(&qwen36_moe_raw_config()),
            QwenMoeServeModel::Qwen36
        );
        assert_eq!(
            qwen_moe_serve_model(&serde_json::json!({
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "num_hidden_layers": 2,
                    "num_experts": 64,
                    "num_experts_per_tok": 4
                },
                "quantization": {"bits": 4, "group_size": 64}
            })),
            QwenMoeServeModel::Qwen35
        );
    }

    #[test]
    fn scheduler_profile_supplies_missing_scheduler_values() {
        let args = base_args();

        let config =
            resolve_scheduler_serve_config(&args, Some(&runtime_profile())).expect("resolved");

        assert_eq!(
            config,
            SchedulerServeConfig {
                prefill_chunk_size: 1024,
                b_max: 2,
                admission_deadline_ms: 7,
                admission_queue_max: 16,
                max_cache_cap: 8192,
                decode_cadence_mid_chunk_cap: 384,
            }
        );
    }

    #[test]
    fn scheduler_profile_cli_values_override_profile_values() {
        let args = ServeArgs {
            prefill_chunk_size: Some(256),
            b_max: Some(1),
            admission_deadline_ms: Some(9),
            admission_queue_max: Some(7),
            max_cache_cap: Some(4096),
            decode_cadence_mid_chunk_cap: Some(512),
            ..base_args()
        };

        let config =
            resolve_scheduler_serve_config(&args, Some(&runtime_profile())).expect("resolved");

        assert_eq!(
            config,
            SchedulerServeConfig {
                prefill_chunk_size: 256,
                b_max: 1,
                admission_deadline_ms: 9,
                admission_queue_max: 7,
                max_cache_cap: 4096,
                decode_cadence_mid_chunk_cap: 512,
            }
        );
    }

    #[test]
    fn scheduler_profile_cli_values_override_dynamic_rule_values() {
        let args = ServeArgs {
            prefill_chunk_size: Some(256),
            decode_cadence_mid_chunk_cap: Some(64),
            ..base_args()
        };

        let profile =
            resolve_scheduler_runtime_profile(&args, Some(&runtime_profile()), &runtime_context())
                .expect("resolved");

        assert_eq!(profile.config.prefill_chunk_size, 256);
        assert_eq!(profile.config.decode_cadence_mid_chunk_cap, 64);
        assert_eq!(profile.rules.len(), 1);
        assert_eq!(profile.rules[0].config.prefill_chunk_size, 256);
        assert_eq!(profile.rules[0].config.decode_cadence_mid_chunk_cap, 64);
    }

    #[test]
    fn gemma4_drafter_default_scheduler_uses_bmax_four_when_unconfigured() {
        let args = base_args();
        let mut resolved = ResolvedSchedulerRuntime {
            scheduler_runtime_profile: default_scheduler_runtime_profile(runtime_context()),
            scheduler_config: SchedulerServeConfig::default(),
            profile_source: None,
        };

        let changed = apply_adaptive_mtp_scheduler_defaults(
            &args,
            crate::models::ModelArchitecture::Gemma4,
            true,
            &mut resolved,
        );

        assert!(changed, "Gemma4 drafter defaults should enable b_max=4");
        assert_eq!(resolved.scheduler_config.b_max, 4);
        assert_eq!(resolved.scheduler_runtime_profile.config.b_max, 4);
    }

    #[test]
    fn gemma4_drafter_default_scheduler_preserves_explicit_max_sequences() {
        let args = ServeArgs {
            b_max: Some(1),
            ..base_args()
        };
        let mut resolved = ResolvedSchedulerRuntime {
            scheduler_runtime_profile: resolve_scheduler_runtime_profile(
                &args,
                None,
                &runtime_context(),
            )
            .expect("resolved profile"),
            scheduler_config: SchedulerServeConfig {
                b_max: 1,
                ..SchedulerServeConfig::default()
            },
            profile_source: None,
        };

        let changed = apply_adaptive_mtp_scheduler_defaults(
            &args,
            crate::models::ModelArchitecture::Gemma4,
            true,
            &mut resolved,
        );

        assert!(!changed, "explicit --max-sequences must be respected");
        assert_eq!(resolved.scheduler_config.b_max, 1);
        assert_eq!(resolved.scheduler_runtime_profile.config.b_max, 1);
    }

    #[test]
    fn gemma4_drafter_default_scheduler_uplifts_loaded_store_profile() {
        let args = base_args();
        let mut resolved = ResolvedSchedulerRuntime {
            scheduler_runtime_profile: runtime_profile(),
            scheduler_config: SchedulerServeConfig {
                b_max: 2,
                ..SchedulerServeConfig::default()
            },
            profile_source: Some(SchedulerProfileSource::Store),
        };

        let changed = apply_adaptive_mtp_scheduler_defaults(
            &args,
            crate::models::ModelArchitecture::Gemma4,
            true,
            &mut resolved,
        );

        assert!(
            changed,
            "auto-loaded store profiles should use the adaptive default"
        );
        assert_eq!(resolved.scheduler_config.b_max, 4);
        assert_eq!(resolved.scheduler_runtime_profile.config.b_max, 4);
        assert_eq!(resolved.scheduler_runtime_profile.rules[0].config.b_max, 4);
    }

    #[test]
    fn gemma4_drafter_default_scheduler_preserves_explicit_profile() {
        let args = ServeArgs {
            scheduler_profile: Some(PathBuf::from("/tmp/profile.json")),
            ..base_args()
        };
        let mut resolved = ResolvedSchedulerRuntime {
            scheduler_runtime_profile: runtime_profile(),
            scheduler_config: SchedulerServeConfig {
                b_max: 2,
                ..SchedulerServeConfig::default()
            },
            profile_source: Some(SchedulerProfileSource::Explicit),
        };

        let changed = apply_adaptive_mtp_scheduler_defaults(
            &args,
            crate::models::ModelArchitecture::Gemma4,
            true,
            &mut resolved,
        );

        assert!(!changed, "explicit scheduler profiles must be respected");
        assert_eq!(resolved.scheduler_config.b_max, 2);
        assert_eq!(resolved.scheduler_runtime_profile.config.b_max, 2);
        assert_eq!(resolved.scheduler_runtime_profile.rules[0].config.b_max, 2);
    }

    #[test]
    fn qwen_mtp_default_scheduler_uses_latency_first_bmax_two_when_unconfigured() {
        let args = base_args();
        let mut resolved = ResolvedSchedulerRuntime {
            scheduler_runtime_profile: default_scheduler_runtime_profile(runtime_context()),
            scheduler_config: SchedulerServeConfig::default(),
            profile_source: None,
        };

        let changed = apply_adaptive_mtp_scheduler_defaults(
            &args,
            crate::models::ModelArchitecture::Qwen35Dense,
            true,
            &mut resolved,
        );

        assert!(changed, "Qwen MTP defaults should enable b_max=2");
        assert_eq!(resolved.scheduler_config.b_max, 2);
        assert_eq!(resolved.scheduler_runtime_profile.config.b_max, 2);
    }

    #[test]
    fn serve_mtp_args_default_off() {
        let args = base_args();

        assert!(args.mtp_model_dir.is_none());
        assert_eq!(args.mtp_draft_tokens, None);
        assert!(!args.prompt_lookup);
        assert!(resolve_prompt_lookup_config(&args)
            .expect("resolve prompt lookup")
            .is_none());
    }

    #[test]
    fn serve_prompt_lookup_requires_explicit_source_enablement() {
        let mut args = base_args();
        args.prompt_lookup_max_draft_tokens = Some(3);

        let error = resolve_prompt_lookup_config(&args).expect_err("source must be enabled");
        assert!(error
            .to_string()
            .contains("source parameters require --prompt-lookup"));
    }

    #[test]
    fn serve_prompt_lookup_rejects_neural_source_configuration() {
        let mut args = base_args();
        args.prompt_lookup = true;
        args.mtp_model_dir = Some(PathBuf::from("/tmp/mtp"));

        let error = resolve_prompt_lookup_config(&args).expect_err("sources are exclusive");
        assert!(error.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn serve_accepts_max_sequences_cli_arg() {
        let cli = crate::cli::Cli::parse_from([
            "ironmlx",
            "serve",
            "--model",
            "/tmp/model",
            "--max-sequences",
            "4",
        ]);
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };

        assert_eq!(args.b_max, Some(4));
    }

    #[test]
    fn serve_paged_prefix_block_size_defaults_to_capacity_friendly_page_size() {
        let cli = crate::cli::Cli::parse_from(["ironmlx", "serve", "--model", "/tmp/model"]);
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };

        assert_eq!(
            args.paged_prefix_cache_block_size,
            crate::core::cache::DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE
        );
    }

    #[test]
    fn serve_rejects_internal_b_max_cli_arg() {
        let err = crate::cli::Cli::try_parse_from([
            "ironmlx",
            "serve",
            "--model",
            "/tmp/model",
            "--b-max",
            "4",
        ])
        .expect_err("--b-max must not be accepted as a public CLI flag");

        assert!(err.to_string().contains("unexpected argument '--b-max'"));
    }

    #[test]
    fn serve_mtp_config_accepts_qwen_single_request_window() {
        let temp_dir = unique_temp_dir("serve-mtp-ok");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        let mut args = base_args();
        args.mtp_model_dir = Some(temp_dir.clone());
        args.mtp_draft_tokens = Some(2);

        let cfg = resolve_serve_mtp_config(
            &args,
            crate::models::ModelArchitecture::Qwen35Dense,
            &serde_json::json!({"model_type": "qwen3_5", "text_config": {}}),
            SchedulerServeConfig {
                b_max: 1,
                ..SchedulerServeConfig::default()
            },
        )
        .expect("resolve")
        .expect("enabled");

        assert_eq!(cfg.model_dir, temp_dir);
        assert_eq!(cfg.draft_tokens, 2);
        std::fs::remove_dir_all(cfg.model_dir).expect("cleanup");
    }

    #[test]
    fn serve_mtp_config_accepts_qwen36_latency_first_default_draft_tokens() {
        let temp_dir = unique_temp_dir("serve-mtp-qwen36-default");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        let mut args = base_args();
        args.mtp_model_dir = Some(temp_dir.clone());

        let cfg = resolve_serve_mtp_config(
            &args,
            crate::models::ModelArchitecture::Qwen35Dense,
            &qwen36_dense_27b_raw_config(),
            SchedulerServeConfig {
                b_max: 1,
                ..SchedulerServeConfig::default()
            },
        )
        .expect("resolve")
        .expect("enabled");

        assert_eq!(cfg.model_dir, temp_dir);
        assert_eq!(cfg.draft_tokens, 1);
        std::fs::remove_dir_all(cfg.model_dir).expect("cleanup");
    }

    #[test]
    fn serve_mtp_config_accepts_batched_scheduler() {
        let temp_dir = unique_temp_dir("serve-mtp-bmax");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        let mut args = base_args();
        args.mtp_model_dir = Some(temp_dir.clone());

        let cfg = resolve_serve_mtp_config(
            &args,
            crate::models::ModelArchitecture::Qwen35Dense,
            &serde_json::json!({"model_type": "qwen3_5", "text_config": {}}),
            SchedulerServeConfig {
                b_max: 2,
                ..SchedulerServeConfig::default()
            },
        )
        .expect("resolve")
        .expect("enabled");

        assert_eq!(cfg.model_dir, temp_dir);
        std::fs::remove_dir_all(cfg.model_dir).expect("cleanup");
    }

    #[test]
    fn serve_mtp_config_accepts_gemma4_drafter() {
        let temp_dir = unique_temp_dir("serve-mtp-gemma4");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        let mut args = base_args();
        args.mtp_model_dir = Some(temp_dir.clone());
        args.mtp_draft_tokens = Some(3);

        let cfg = resolve_serve_mtp_config(
            &args,
            crate::models::ModelArchitecture::Gemma4,
            &serde_json::json!({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}),
            SchedulerServeConfig {
                b_max: 1,
                ..SchedulerServeConfig::default()
            },
        )
        .expect("resolve")
        .expect("enabled");

        assert_eq!(cfg.model_dir, temp_dir);
        assert_eq!(cfg.draft_tokens, 3);
        std::fs::remove_dir_all(cfg.model_dir).expect("cleanup");
    }

    #[test]
    fn serve_mtp_config_accepts_gemma4_batched_scheduler() {
        let temp_dir = unique_temp_dir("serve-mtp-gemma4-batched");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        let mut args = base_args();
        args.mtp_model_dir = Some(temp_dir.clone());

        let cfg = resolve_serve_mtp_config(
            &args,
            crate::models::ModelArchitecture::Gemma4,
            &serde_json::json!({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}),
            SchedulerServeConfig {
                b_max: 2,
                ..SchedulerServeConfig::default()
            },
        )
        .expect("resolve")
        .expect("enabled");

        assert_eq!(cfg.model_dir, temp_dir);
        std::fs::remove_dir_all(cfg.model_dir).expect("cleanup");
    }

    #[test]
    fn serve_paged_prefix_cache_accepts_mtp_config() {
        let temp_dir = unique_temp_dir("serve-mtp-prefix");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        let prefix_dir = unique_temp_dir("serve-prefix-mtp");
        let mut args = base_args();
        args.mtp_model_dir = Some(temp_dir.clone());
        args.paged_prefix_cache_dir = Some(prefix_dir.clone());

        let cfg = resolve_paged_prefix_cache_config(
            &args,
            SchedulerServeConfig {
                b_max: 2,
                max_cache_cap: 128,
                ..SchedulerServeConfig::default()
            },
            "/tmp/model",
        )
        .expect("prefix config")
        .expect("enabled");

        assert_eq!(cfg.root, prefix_dir);
        std::fs::remove_dir_all(temp_dir).expect("cleanup mtp dir");
        std::fs::remove_dir_all(prefix_dir).ok();
    }

    #[test]
    fn serve_paged_prefix_cache_expands_default_home_dir() {
        let mut args = base_args();
        args.paged_prefix_cache_dir = Some(PathBuf::from("~/.ironmlx/cache/paged_prefix_cache"));

        let cfg = resolve_paged_prefix_cache_config(
            &args,
            SchedulerServeConfig {
                b_max: 2,
                max_cache_cap: 128,
                ..SchedulerServeConfig::default()
            },
            "/tmp/model",
        )
        .expect("prefix config")
        .expect("enabled");

        let expected = dirs::home_dir()
            .expect("home dir")
            .join(".ironmlx")
            .join("cache")
            .join("paged_prefix_cache");
        assert_eq!(cfg.root, expected);
    }

    #[test]
    fn serve_paged_prefix_cache_accepts_ssd_max_gb() {
        let prefix_dir = unique_temp_dir("serve-prefix-ssd-max-gb");
        let mut args = base_args();
        args.paged_prefix_cache_dir = Some(prefix_dir.clone());
        args.ssd_prefix_cache_max_gb = Some(10);

        let cfg = resolve_paged_prefix_cache_config(
            &args,
            SchedulerServeConfig {
                b_max: 2,
                max_cache_cap: 128,
                ..SchedulerServeConfig::default()
            },
            "/tmp/model",
        )
        .expect("prefix config")
        .expect("enabled");

        assert_eq!(cfg.max_disk_bytes, Some(10 * 1024 * 1024 * 1024));
        std::fs::remove_dir_all(prefix_dir).ok();
    }

    #[test]
    fn serve_paged_prefix_cache_accepts_turboquant() {
        let prefix_dir = unique_temp_dir("serve-prefix-kv-quant");
        let mut args = base_args();
        args.paged_prefix_cache_dir = Some(prefix_dir.clone());
        args.kv_quant = KvQuantArg::K3V4;

        let cfg = resolve_paged_prefix_cache_config(
            &args,
            SchedulerServeConfig {
                b_max: 2,
                max_cache_cap: 128,
                ..SchedulerServeConfig::default()
            },
            "/tmp/model",
        )
        .expect("paged prefix cache should allow TurboQuant")
        .expect("paged prefix cache enabled");

        assert_eq!(cfg.model_id, "/tmp/model");
        std::fs::remove_dir_all(prefix_dir).ok();
    }

    #[test]
    fn serve_paged_prefix_cache_rejects_zero_ssd_max_gb() {
        let prefix_dir = unique_temp_dir("serve-prefix-ssd-max-gb-zero");
        let mut args = base_args();
        args.paged_prefix_cache_dir = Some(prefix_dir.clone());
        args.ssd_prefix_cache_max_gb = Some(0);

        let err = resolve_paged_prefix_cache_config(
            &args,
            SchedulerServeConfig {
                b_max: 2,
                max_cache_cap: 128,
                ..SchedulerServeConfig::default()
            },
            "/tmp/model",
        )
        .expect_err("zero SSD prefix cache limit");

        assert!(err
            .to_string()
            .contains("--ssd-prefix-cache-max-gb must be > 0"));
        std::fs::remove_dir_all(prefix_dir).ok();
    }

    #[test]
    fn serve_prefix_lru_cache_requires_paged_prefix_cache() {
        let mut args = base_args();
        args.prefix_lru_cache_max_bytes = Some(1024);

        let err =
            resolve_prefix_lru_cache_config(&args, None).expect_err("L1 requires paged SSD cache");

        assert!(err
            .to_string()
            .contains("--prefix-lru-cache-max-bytes requires --paged-prefix-cache-dir"));
    }

    #[test]
    fn serve_prefix_lru_cache_rejects_zero_capacity() {
        let prefix_dir = unique_temp_dir("serve-prefix-lru-zero");
        let mut args = base_args();
        args.paged_prefix_cache_dir = Some(prefix_dir.clone());
        args.prefix_lru_cache_max_bytes = Some(0);
        let paged_prefix_cache = resolve_paged_prefix_cache_config(
            &args,
            SchedulerServeConfig {
                b_max: 2,
                max_cache_cap: 128,
                ..SchedulerServeConfig::default()
            },
            "/tmp/model",
        )
        .expect("prefix config");

        let err = resolve_prefix_lru_cache_config(&args, paged_prefix_cache.as_ref())
            .expect_err("zero capacity");

        assert!(err.to_string().contains("max_bytes must be > 0"));
        std::fs::remove_dir_all(prefix_dir).ok();
    }

    #[test]
    fn serve_prefix_lru_cache_accepts_paged_prefix_cache() {
        let prefix_dir = unique_temp_dir("serve-prefix-lru");
        let mut args = base_args();
        args.paged_prefix_cache_dir = Some(prefix_dir.clone());
        args.prefix_lru_cache_max_bytes = Some(4096);
        let paged_prefix_cache = resolve_paged_prefix_cache_config(
            &args,
            SchedulerServeConfig {
                b_max: 2,
                max_cache_cap: 128,
                ..SchedulerServeConfig::default()
            },
            "/tmp/model",
        )
        .expect("prefix config");

        let cfg = resolve_prefix_lru_cache_config(&args, paged_prefix_cache.as_ref())
            .expect("L1 config")
            .expect("enabled");

        assert_eq!(cfg.max_bytes, 4096);
        std::fs::remove_dir_all(prefix_dir).ok();
    }

    #[test]
    fn serve_mtp_config_rejects_non_qwen_or_gemma4_architecture() {
        let temp_dir = unique_temp_dir("serve-mtp-non-qwen-gemma4");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        let mut args = base_args();
        args.mtp_model_dir = Some(temp_dir.clone());

        let err = resolve_serve_mtp_config(
            &args,
            crate::models::ModelArchitecture::Llama,
            &serde_json::json!({"model_type": "llama"}),
            SchedulerServeConfig {
                b_max: 1,
                ..SchedulerServeConfig::default()
            },
        )
        .expect_err("non-Qwen/Gemma4 must be rejected");

        assert!(err.to_string().contains("Qwen/Gemma4"));
        std::fs::remove_dir_all(temp_dir).expect("cleanup");
    }

    #[test]
    fn serve_mtp_config_rejects_missing_dir_and_zero_draft_tokens() {
        let mut args = base_args();
        args.mtp_model_dir = Some(PathBuf::from("/tmp/ironmlx-missing-mtp-dir"));
        let missing = resolve_serve_mtp_config(
            &args,
            crate::models::ModelArchitecture::Qwen35Dense,
            &serde_json::json!({"model_type": "qwen3_5", "text_config": {}}),
            SchedulerServeConfig {
                b_max: 1,
                ..SchedulerServeConfig::default()
            },
        )
        .expect_err("missing dir");
        assert!(missing.to_string().contains("local directory"));

        let temp_dir = unique_temp_dir("serve-mtp-zero-draft");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        args.mtp_model_dir = Some(temp_dir.clone());
        args.mtp_draft_tokens = Some(0);
        let zero = resolve_serve_mtp_config(
            &args,
            crate::models::ModelArchitecture::Qwen35Dense,
            &serde_json::json!({"model_type": "qwen3_5", "text_config": {}}),
            SchedulerServeConfig {
                b_max: 1,
                ..SchedulerServeConfig::default()
            },
        )
        .expect_err("zero draft tokens");
        assert!(zero.to_string().contains("max_draft_tokens must be > 0"));
        std::fs::remove_dir_all(temp_dir).expect("cleanup");
    }

    #[test]
    fn scheduler_profile_health_warning_does_not_prevent_profile_resolution() {
        let mut profile = runtime_profile();
        profile.metadata.created_at_unix_ms = 1811606400000;
        profile.metadata.scenario_coverage = vec![SchedulerAutotuneScenario {
            prompt_len: 1024,
            max_new_tokens: 128,
            concurrency: 1,
            cache_state: SchedulerAutotuneCacheState::Cold,
        }];
        let args = base_args();

        let checked = check_loaded_scheduler_profile_health(
            &profile,
            "different-model-name",
            "test-host",
            &profile.runtime_context,
            1811606400000 + 31 * 24 * 60 * 60 * 1000,
        )
        .expect("warning health should not fail");

        assert_eq!(
            checked.status,
            SchedulerAutotuneProfileHealthStatus::Warning
        );
        assert!(
            resolve_scheduler_runtime_profile(&args, Some(&profile), &profile.runtime_context)
                .is_ok()
        );
    }

    #[test]
    fn scheduler_profile_health_invalid_returns_error() {
        let mut profile = runtime_profile();
        profile.hardware_label = "other-host".to_string();

        let error = check_loaded_scheduler_profile_health(
            &profile,
            "test-model",
            "test-host",
            &profile.runtime_context,
            1811606400000,
        )
        .expect_err("invalid health should fail");

        assert!(format!("{error:#}").contains("invalid scheduler profile"));
    }

    #[test]
    fn serve_auto_loads_matching_profile_from_store_when_cli_profile_absent() {
        let temp_dir = unique_temp_dir("scheduler-profile-store-serve");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));
        store
            .persist_profile(&model_dir, &runtime_profile())
            .expect("persist profile");
        let args = ServeArgs {
            model: Some(model_dir.to_string_lossy().into_owned()),
            ..base_args()
        };

        let fingerprint = runtime_context().fingerprint();
        let loaded = load_scheduler_profile_for_model(
            &args,
            &model_dir,
            Some(&store),
            "test-host",
            &fingerprint,
        )
        .expect("load profile")
        .expect("stored profile should match");

        assert_eq!(loaded.profile.config, profile_config());
        assert_eq!(
            loaded.path,
            store.profile_path(
                "test-model",
                "test-host",
                runtime_profile().metadata.selection_profile,
                &model_dir,
                &fingerprint,
            )
        );

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn explicit_scheduler_profile_overrides_profile_store() {
        let temp_dir = unique_temp_dir("scheduler-profile-explicit");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));
        store
            .persist_profile(&model_dir, &runtime_profile())
            .expect("persist stored profile");
        let explicit_profile = SchedulerAutotuneRuntimeProfile {
            model_name: "explicit-model".to_string(),
            config: SchedulerAutotuneProfileConfig {
                prefill_chunk_size: 4096,
                ..profile_config()
            },
            ..runtime_profile()
        };
        let explicit_path = temp_dir.join("explicit-profile.json");
        let output = serde_json::to_string_pretty(&explicit_profile).expect("serialize profile");
        std::fs::write(&explicit_path, format!("{output}\n")).expect("write explicit profile");
        let args = ServeArgs {
            model: Some(model_dir.to_string_lossy().into_owned()),
            scheduler_profile: Some(explicit_path.clone()),
            ..base_args()
        };

        let loaded = load_scheduler_profile_for_model(
            &args,
            &model_dir,
            Some(&store),
            "test-host",
            &runtime_context().fingerprint(),
        )
        .expect("load profile")
        .expect("explicit profile should load");

        assert_eq!(loaded.profile.model_name, "explicit-model");
        assert_eq!(loaded.profile.config.prefill_chunk_size, 4096);
        assert_eq!(loaded.path, explicit_path);

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn serve_ignores_corrupt_profile_store_index_when_cli_profile_absent() {
        let temp_dir = unique_temp_dir("scheduler-profile-corrupt-index");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let store_root = temp_dir.join("store");
        std::fs::create_dir_all(&store_root).expect("create store dir");
        std::fs::write(store_root.join("index-v5.json"), "not json").expect("write corrupt index");
        let store = SchedulerProfileStore::from_root(store_root);
        let args = ServeArgs {
            model: Some(model_dir.to_string_lossy().into_owned()),
            ..base_args()
        };

        let loaded = load_scheduler_profile_for_model(
            &args,
            &model_dir,
            Some(&store),
            "test-host",
            &runtime_context().fingerprint(),
        )
        .expect("corrupt store should fall back");

        assert!(loaded.is_none());

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn serve_ignores_corrupt_auto_loaded_profile_when_cli_profile_absent() {
        let temp_dir = unique_temp_dir("scheduler-profile-corrupt-profile");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));
        let stored_path = store
            .persist_profile(&model_dir, &runtime_profile())
            .expect("persist profile");
        std::fs::write(&stored_path, "not json").expect("corrupt stored profile");
        let args = ServeArgs {
            model: Some(model_dir.to_string_lossy().into_owned()),
            ..base_args()
        };

        let loaded = load_scheduler_profile_for_model(
            &args,
            &model_dir,
            Some(&store),
            "test-host",
            &runtime_context().fingerprint(),
        )
        .expect("corrupt auto profile should fall back");

        assert!(loaded.is_none());

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }
}
