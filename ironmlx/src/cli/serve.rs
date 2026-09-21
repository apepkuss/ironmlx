//! `ironmlx serve` — boot HTTP server with OpenAI + Anthropic compatibility.

use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context};
use clap::Args;

use super::KvQuantArg;
use crate::Result;
use ironmlx_lm::core::speculative_model::MtpSpeculativeModel;
use ironmlx_lm::core::vision::DenseVlMethods;
use ironmlx_runtime::core::cache::prefix_store::DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE;
use ironmlx_runtime::core::process_memory::StaticMemoryEstimate;
use {
    crate::server, ironmlx_lm::core::loader::Loader, ironmlx_lm::core::model::Model,
    ironmlx_lm::core::tokenizer::Tokenizer,
};
use {
    ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeContext,
    ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile,
};
use {
    ironmlx_runtime::core::scheduler_profile_store::detect_scheduler_profile_hardware_label,
    ironmlx_runtime::core::scheduler_profile_store::SchedulerProfileStore,
};

const DEFAULT_DFLASH2_TENSOR_BATCH_MAX_WIDTH: usize = 4;
const DEFAULT_PAGED_PREFIX_CACHE_DIR: &str = "~/.ironmlx/cache/paged_prefix_cache";

#[derive(Args, Clone, Debug)]
pub struct ServeArgs {
    /// Local directory containing config.json + model.safetensors + tokenizer.json.
    /// HF repo-id resolution is deferred to a future phase; pass a local path for now.
    #[arg(long, conflicts_with = "model_manifest")]
    pub model: Option<String>,

    /// Stable public identifier for single-model serving. When omitted, the
    /// value passed to --model remains the public identifier.
    #[arg(
        long = "model-id",
        requires = "model",
        conflicts_with = "model_manifest"
    )]
    pub model_id: Option<String>,

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

    /// Network exposure mode. Local mode only accepts a loopback --host.
    #[arg(long, value_enum, default_value = "local")]
    pub network_mode: server::security::NetworkMode,

    /// Concrete LAN interface address used by LAN mode. Wildcard addresses are rejected.
    #[arg(long, requires = "security_bootstrap_stdin")]
    pub lan_host: Option<IpAddr>,

    /// Read the API-key digest and TLS material as one JSON object from stdin.
    #[arg(long, default_value_t = false)]
    pub security_bootstrap_stdin: bool,

    #[arg(skip)]
    pub(crate) network_config: Option<server::security::ServerNetworkConfig>,

    /// Prefill chunk size — max tokens per prefill forward call. `0`
    /// disables chunking (single-shot forward over the whole prompt).
    /// Intermediate chunks update the cache only; the last chunk runs
    /// the full forward + lm_head. Defaults to `2048` unless supplied by
    /// `--scheduler-profile`.
    #[arg(long)]
    pub prefill_chunk_size: Option<usize>,

    /// Route greedy HTTP generation through SchedulerActor even when the
    /// ordinary GenerationStream path would otherwise be eligible.
    #[arg(long)]
    pub force_scheduler: bool,

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

    /// Optional local MTP/drafter model directory. When set, greedy or exact
    /// sampled speculative decoding is enabled for supported Qwen and Gemma4 models.
    #[arg(long = "mtp-model-dir")]
    pub mtp_model_dir: Option<PathBuf>,

    /// Maximum MTP draft tokens per speculative window. If omitted, ironmlx
    /// picks a model-aware default from local benchmark policy.
    #[arg(long = "mtp-draft-tokens")]
    pub mtp_draft_tokens: Option<usize>,

    /// Official DFlash2 draft checkpoint directory. Selects the isolated,
    /// text-only DFlash2 request actor for this server.
    #[arg(long = "dflash2-model-dir")]
    pub dflash2_model_dir: Option<PathBuf>,

    /// DFlash2 proposal block width.
    #[arg(long = "dflash2-block-size", default_value_t = 4)]
    pub dflash2_block_size: usize,

    /// Runtime affine quantization for the official BF16 DFlash2 draft.
    /// Pass 0 to keep the draft in BF16.
    #[arg(long = "dflash2-draft-bits", default_value_t = 4)]
    pub dflash2_draft_bits: i32,

    /// Maximum number of compatible DFlash2 requests combined into one tensor
    /// verification group. The effective width is also capped by
    /// --max-sequences. If omitted, the certified default is 4; pass 1 to
    /// disable cross-request tensor batching while retaining actor concurrency.
    #[arg(
        long = "dflash2-tensor-batch-max-width",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub dflash2_tensor_batch_max_width: Option<usize>,

    /// Enable request-local greedy or exact sampled prompt lookup speculative decoding.
    #[arg(long = "prompt-lookup", default_value_t = false)]
    pub prompt_lookup: bool,

    /// Reuse immutable histories from normally completed requests in this
    /// model-engine trust domain. The server must not mix untrusted tenants.
    #[arg(long = "prompt-lookup-cross-request", default_value_t = false)]
    pub prompt_lookup_cross_request: bool,

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
    /// by default. Ordinary/MTP serving requires --paged-prefix-cache-dir;
    /// DFlash2 uses a dedicated in-memory artifact and does not enable SSD.
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

impl ServeArgs {
    pub(crate) fn resolved_network_config(&self) -> Result<server::security::ServerNetworkConfig> {
        self.network_config
            .clone()
            .context("server network configuration was not initialized")
    }
}

pub(crate) fn resolve_paged_prefix_cache_config(
    args: &ServeArgs,
    scheduler_config: SchedulerServeConfig,
    model_id: &str,
) -> Result<Option<ironmlx_runtime::core::cache::prefix_store::PagedPrefixCacheConfig>> {
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
    ironmlx_runtime::core::cache::prefix_store::PagedPrefixCacheConfig::new_with_max_disk_bytes(
        root,
        model_id.to_string(),
        block_size,
        max_pages,
        max_disk_bytes,
    )
    .map(Some)
}

fn resolve_ssd_prefix_cache_max_bytes(args: &ServeArgs) -> Result<Option<usize>> {
    ironmlx_runtime::core::scheduler_resolution::resolve_ssd_prefix_cache_max_bytes(
        &SchedulerResolutionOptions::from(args),
    )
}

pub(crate) fn resolve_engine_paged_prefix_cache_settings(
    args: &ServeArgs,
) -> Result<Option<ironmlx_runtime::core::engine_pool::EnginePagedPrefixCacheSettings>> {
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
    Ok(Some(
        ironmlx_runtime::core::engine_pool::EnginePagedPrefixCacheSettings {
            root,
            block_size,
            max_pages: args.paged_prefix_cache_max_pages,
            max_disk_bytes,
        },
    ))
}

pub(crate) fn resolve_prefix_lru_cache_config(
    args: &ServeArgs,
    paged_prefix_cache: Option<&ironmlx_runtime::core::cache::prefix_store::PagedPrefixCacheConfig>,
) -> Result<Option<ironmlx_runtime::core::cache::prefix_store::PrefixLruCacheConfig>> {
    let Some(max_bytes) = args.prefix_lru_cache_max_bytes else {
        return Ok(None);
    };
    if paged_prefix_cache.is_none() {
        bail!("--prefix-lru-cache-max-bytes requires --paged-prefix-cache-dir");
    }
    ironmlx_runtime::core::cache::prefix_store::PrefixLruCacheConfig::new(max_bytes).map(Some)
}

fn resolve_dflash2_prefix_lru_cache_config(
    args: &ServeArgs,
) -> Result<Option<ironmlx_runtime::core::cache::prefix_store::PrefixLruCacheConfig>> {
    args.prefix_lru_cache_max_bytes
        .map(ironmlx_runtime::core::cache::prefix_store::PrefixLruCacheConfig::new)
        .transpose()
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
) -> Result<ironmlx_runtime::core::cache::active_kv::ActiveKvOffloadConfig> {
    if !args.active_kv_offload {
        return Ok(ironmlx_runtime::core::cache::active_kv::ActiveKvOffloadConfig::disabled());
    }
    let root = match args.active_kv_offload_dir.as_ref() {
        Some(root) => expand_home_path(root)?,
        None => ironmlx_runtime::core::cache::active_kv::default_active_kv_offload_dir(),
    };
    Ok(ironmlx_runtime::core::cache::active_kv::ActiveKvOffloadConfig::enabled(root))
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ServeMtpConfig {
    model_dir: PathBuf,
    draft_tokens: usize,
}

fn resolve_prompt_lookup_config(
    args: &ServeArgs,
) -> Result<Option<ironmlx_runtime::core::prompt_lookup::PromptLookupConfig>> {
    let has_source_params = args.prompt_lookup_min_ngram.is_some()
        || args.prompt_lookup_max_ngram.is_some()
        || args.prompt_lookup_max_draft_tokens.is_some()
        || args.prompt_lookup_history_window_tokens.is_some()
        || args.prompt_lookup_max_index_entries.is_some()
        || args.prompt_lookup_cross_request;
    if !args.prompt_lookup {
        if has_source_params {
            bail!("prompt lookup source parameters require --prompt-lookup");
        }
        return Ok(None);
    }
    let defaults = ironmlx_runtime::core::prompt_lookup::PromptLookupConfig::default();
    Ok(Some(
        ironmlx_runtime::core::prompt_lookup::PromptLookupConfig {
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
            cross_request: args.prompt_lookup_cross_request,
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
    if ironmlx_lm::models::is_qwen36_moe_config(raw_config) {
        QwenMoeServeModel::Qwen36
    } else {
        QwenMoeServeModel::Qwen35
    }
}

pub(crate) fn apply_adaptive_mtp_scheduler_defaults(
    args: &ServeArgs,
    architecture: ironmlx_lm::models::ModelArchitecture,
    mtp_enabled: bool,
    resolved: &mut ResolvedSchedulerRuntime,
) -> bool {
    ironmlx_runtime::core::scheduler_resolution::apply_adaptive_mtp_scheduler_defaults(
        &SchedulerResolutionOptions::from(args),
        architecture,
        mtp_enabled,
        resolved,
    )
}

#[cfg(test)]
fn load_scheduler_profile_for_model(
    args: &ServeArgs,
    model_dir: &Path,
    store: Option<&SchedulerProfileStore>,
    hardware_label: &str,
    runtime_context_fingerprint: &str,
) -> Result<Option<SchedulerProfileLoad>> {
    ironmlx_runtime::core::scheduler_resolution::load_scheduler_profile_for_model(
        &SchedulerResolutionOptions::from(args),
        model_dir,
        store,
        hardware_label,
        runtime_context_fingerprint,
    )
}

fn resolve_serve_mtp_config(
    args: &ServeArgs,
    architecture: ironmlx_lm::models::ModelArchitecture,
    raw_config: &serde_json::Value,
    _scheduler_config: SchedulerServeConfig,
) -> Result<Option<ServeMtpConfig>> {
    let Some(model_dir) = args.mtp_model_dir.as_ref() else {
        return Ok(None);
    };
    match architecture {
        ironmlx_lm::models::ModelArchitecture::Qwen35Dense
        | ironmlx_lm::models::ModelArchitecture::Qwen35Moe
        | ironmlx_lm::models::ModelArchitecture::Gemma4 => {}
        _ => bail!("ironmlx serve --mtp-model-dir currently supports Qwen/Gemma4 models only"),
    }
    if !model_dir.exists() {
        bail!(
            "--mtp-model-dir must point to a local directory (got '{}')",
            model_dir.display()
        );
    }
    let draft_tokens = ironmlx_runtime::core::speculative::resolve_mtp_draft_tokens(
        raw_config,
        args.mtp_draft_tokens
            .map(ironmlx_runtime::core::speculative::MtpDraftTokensArg::Explicit)
            .unwrap_or(ironmlx_runtime::core::speculative::MtpDraftTokensArg::Omitted),
    );
    ironmlx_runtime::core::speculative::MtpSpeculativeConfig::new(
        draft_tokens,
        ironmlx_core::sampler::Sampler::greedy(),
    )?;
    Ok(Some(ServeMtpConfig {
        model_dir: model_dir.clone(),
        draft_tokens,
    }))
}

fn ensure_dflash2_serve_supported(
    args: &ServeArgs,
    architecture: ironmlx_lm::models::ModelArchitecture,
    scheduler_config: SchedulerServeConfig,
) -> Result<()> {
    let Some(draft_dir) = args.dflash2_model_dir.as_ref() else {
        return Ok(());
    };
    if architecture != ironmlx_lm::models::ModelArchitecture::Qwen35Dense {
        bail!("--dflash2-model-dir currently supports dense Qwen3.5 targets only");
    }
    if !draft_dir.is_dir() {
        bail!(
            "--dflash2-model-dir must point to a local directory (got '{}')",
            draft_dir.display()
        );
    }
    if args.mtp_model_dir.is_some() || args.mtp_draft_tokens.is_some() {
        bail!("--dflash2-model-dir cannot be combined with MTP arguments");
    }
    if resolve_prompt_lookup_config(args)?.is_some() {
        bail!("--dflash2-model-dir cannot be combined with PromptLookup");
    }
    if args.kv_quant.turboquant_bits().is_some()
        || args.paged_prefix_cache_dir.is_some()
        || args.active_kv_offload
    {
        bail!(
            "--dflash2-model-dir has not qualified KV quantization, paged/SSD prefix cache, or active KV offload"
        );
    }
    if scheduler_config.b_max == 0 {
        bail!("--dflash2-model-dir requires --max-sequences greater than zero");
    }
    if args.scheduler_profile.is_some() || args.scheduler_autotune_report {
        bail!("--dflash2-model-dir does not use scheduler profiles or scheduler autotune reports");
    }
    if !(2..=8).contains(&args.dflash2_block_size) {
        bail!("--dflash2-block-size must be in [2, 8] for the official Qwen3.8 draft");
    }
    if !matches!(args.dflash2_draft_bits, 0 | 4 | 8) {
        bail!("--dflash2-draft-bits must be one of 0, 4, or 8");
    }
    Ok(())
}

fn resolve_dflash2_tensor_batch_width(args: &ServeArgs, b_max: usize) -> (usize, usize) {
    let requested = args
        .dflash2_tensor_batch_max_width
        .unwrap_or(DEFAULT_DFLASH2_TENSOR_BATCH_MAX_WIDTH);
    (requested, requested.min(b_max))
}

fn resolve_scheduler_runtime_profile(
    args: &ServeArgs,
    profile: Option<&SchedulerAutotuneRuntimeProfile>,
    runtime_context: &SchedulerAutotuneRuntimeContext,
) -> Result<SchedulerAutotuneRuntimeProfile> {
    ironmlx_runtime::core::scheduler_resolution::resolve_scheduler_runtime_profile(
        &SchedulerResolutionOptions::from(args),
        profile,
        runtime_context,
    )
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
    prompt_lookup: Option<ironmlx_runtime::core::prompt_lookup::PromptLookupConfig>,
    max_cache_cap_override: Option<usize>,
) -> Result<ResolvedSchedulerRuntime> {
    ironmlx_runtime::core::scheduler_resolution::resolve_scheduler_for_model_with_speculative(
        &SchedulerResolutionOptions::from(args),
        model_dir,
        mtp_model_dir,
        mtp_draft_tokens,
        prompt_lookup,
        max_cache_cap_override,
    )
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
    if let Some(model_id) = args.model_id.as_deref() {
        let model_id = model_id.trim();
        if model_id.is_empty() {
            bail!("--model-id cannot be empty");
        }
        return Ok(model_id.to_owned());
    }
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
    vision_input: Option<ironmlx_lm::core::vision_input::VisionInputConfig>,
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
            cross_request = cfg.cross_request,
            "ironmlx serve: PromptLookup enabled"
        );
        runtime.block_on(server::serve_with_prompt_lookup(
            model,
            cfg,
            tokenizer,
            model_id,
            args.resolved_network_config()?,
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
            args.resolved_network_config()?,
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
            args.force_scheduler,
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
    vision_input: Option<ironmlx_lm::core::vision_input::VisionInputConfig>,
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
    tracing::info!(
        "ironmlx serve: MTP enabled model_dir={} draft_tokens={}",
        mtp_config.model_dir.display(),
        mtp_config.draft_tokens
    );
    let prefix_lru_cache = resolve_prefix_lru_cache_config(args, paged_prefix_cache.as_ref())?;
    let active_kv_offload = resolve_active_kv_offload_config(args)?;
    let prompt_lookup = resolve_prompt_lookup_config(args)?;
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
        prompt_lookup,
        tokenizer,
        model_id,
        args.resolved_network_config()?,
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
fn serve_with_dflash2_model(
    model: ironmlx_lm::models::Qwen35Model,
    tokenizer: Tokenizer,
    args: &ServeArgs,
    scheduler_config: SchedulerServeConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    mut static_memory_estimate: StaticMemoryEstimate,
) -> Result<()> {
    let draft_dir = args
        .dflash2_model_dir
        .as_ref()
        .context("DFlash2 server started without --dflash2-model-dir")?;
    let draft_loader = Loader::open_dflash2(draft_dir)
        .with_context(|| format!("Loader::open_dflash2 {}", draft_dir.display()))?;
    let draft_bits = (args.dflash2_draft_bits != 0).then_some(args.dflash2_draft_bits);
    let draft = ironmlx_lm::models::DFlash2DraftModel::from_loader(
        &draft_loader,
        model.config(),
        draft_bits,
    )
    .context("DFlash2DraftModel::from_loader")?;
    static_memory_estimate.speculative_cold_bytes = draft_loader.loaded_tensor_bytes();
    drop(draft_loader);
    mlx::clear_cache();

    let model_id = single_model_id(args)?;
    let prefix_cache = resolve_dflash2_prefix_lru_cache_config(args)?;
    let (tensor_batch_requested_max_width, tensor_batch_max_width) =
        resolve_dflash2_tensor_batch_width(args, scheduler_config.b_max);
    tracing::info!(
        "ironmlx serve: DFlash2 enabled model_dir={} block_size={} draft_bits={} max_sequences={} tensor_batch_requested_max_width={} tensor_batch_effective_max_width={} prefix_cache_max_bytes={:?}",
        draft_dir.display(),
        args.dflash2_block_size,
        args.dflash2_draft_bits,
        scheduler_config.b_max,
        tensor_batch_requested_max_width,
        tensor_batch_max_width,
        prefix_cache.map(|config| config.max_bytes),
    );
    let runtime = serve_runtime()?;
    runtime.block_on(server::serve_with_dflash2(
        model,
        draft,
        tokenizer,
        model_id,
        args.resolved_network_config()?,
        scheduler_config.prefill_chunk_size,
        scheduler_config.b_max,
        scheduler_config.admission_deadline_ms,
        tensor_batch_max_width,
        scheduler_config.admission_queue_max,
        scheduler_config.max_cache_cap,
        args.dflash2_block_size,
        draft_bits,
        prefix_cache,
        scheduler_runtime_profile,
        static_memory_estimate,
    ))
}

#[allow(clippy::too_many_arguments)]
fn serve_with_gemma4_drafter_model(
    model: ironmlx_lm::models::Gemma4Model,
    tokenizer: Tokenizer,
    model_dir: &Path,
    mut static_memory_estimate: StaticMemoryEstimate,
    mtp_config: ServeMtpConfig,
    args: &ServeArgs,
    scheduler_config: SchedulerServeConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    vision_input: Option<ironmlx_lm::core::vision_input::VisionInputConfig>,
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
    let drafter = ironmlx_lm::models::gemma4::Gemma4AssistantModel::from_loader(&drafter_loader)
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
    let prompt_lookup = resolve_prompt_lookup_config(args)?;
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
        prompt_lookup,
        tokenizer,
        model_id,
        args.resolved_network_config()?,
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
    model: ironmlx_lm::models::DiffusionGemmaModel,
    tokenizer: Tokenizer,
    generation_config: ironmlx_lm::models::DiffusionGemmaGenerationConfig,
    model_weight_bytes: usize,
    args: &ServeArgs,
    vision_input: ironmlx_lm::core::vision_input::VisionInputConfig,
) -> Result<()> {
    let model_id = single_model_id(args)?;
    let runtime = serve_runtime()?;
    runtime.block_on(server::diffusion_gemma::serve_diffusion_gemma(
        model,
        tokenizer,
        generation_config,
        model_id,
        model_weight_bytes,
        args.resolved_network_config()?,
        vision_input,
    ))
}

fn read_engine_pool_manifest(
    path: &Path,
) -> Result<ironmlx_runtime::core::engine_pool::EnginePoolManifest> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn build_engine_model_config_for_pool(
    args: &ServeArgs,
    model: ironmlx_runtime::core::engine_pool::EngineModelManifest,
    scheduler_profile_store: Option<&SchedulerProfileStore>,
    hardware_label: &str,
) -> Result<ironmlx_runtime::core::engine_pool::EngineModelConfig> {
    ironmlx_runtime::core::model_management::build_engine_model_config_for_pool(
        &SchedulerResolutionOptions::from(args),
        model,
        scheduler_profile_store,
        hardware_label,
    )
}

fn run_engine_pool(args: ServeArgs, manifest_path: &Path) -> Result<()> {
    if args.dflash2_model_dir.is_some() {
        bail!("--model-manifest does not accept the single-model --dflash2-model-dir flag");
    }
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
    let _registry = ironmlx_runtime::core::engine_pool::EngineRegistry::new(manifest.clone())?;
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
        network: args.resolved_network_config()?,
        options: ironmlx_runtime::core::runtime_config::EngineRuntimeOptions {
            kv_cache_turboquant_bits: args.kv_quant.turboquant_bits(),
            scheduler_autotune_report: args.scheduler_autotune_report,
            paged_prefix_cache,
            prefix_lru_cache_max_bytes: args.prefix_lru_cache_max_bytes,
            model_ttl,
            memory_limits: ironmlx_runtime::core::engine_pool::EnginePoolMemoryLimits {
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
        },
    };
    let config = ironmlx_runtime::core::engine_pool::EnginePoolConfig {
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
    let network = args.resolved_network_config()?;
    runtime.block_on(async move {
        let manager = server::model_manager::ModelManager::new(
            engine_runtime_config(&args)?,
            args.max_loaded_models,
            SchedulerResolutionOptions::from(&args),
        )?;
        server::model_manager::serve_app_daemon(manager, network).await
    })
}

pub fn run(mut args: ServeArgs) -> Result<()> {
    args.network_config = Some(server::security::ServerNetworkConfig::resolve(
        args.network_mode,
        &args.host,
        args.port,
        args.lan_host,
        args.security_bootstrap_stdin,
    )?);
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

    let mut resolved_scheduler = if args.dflash2_model_dir.is_some() {
        let runtime_context = SchedulerAutotuneRuntimeContext::local_default(DEFAULT_MAX_CACHE_CAP);
        let scheduler_runtime_profile =
            resolve_scheduler_runtime_profile(&args, None, &runtime_context)?;
        ResolvedSchedulerRuntime {
            scheduler_config: SchedulerServeConfig {
                prefill_chunk_size: scheduler_runtime_profile.config.prefill_chunk_size,
                b_max: scheduler_runtime_profile.config.b_max,
                admission_deadline_ms: scheduler_runtime_profile.config.admission_deadline_ms,
                admission_queue_max: scheduler_runtime_profile.config.admission_queue_max,
                max_cache_cap: scheduler_runtime_profile.config.max_cache_cap,
                decode_cadence_mid_chunk_cap: scheduler_runtime_profile
                    .config
                    .decode_cadence_mid_chunk_cap,
            },
            scheduler_runtime_profile,
            profile_source: None,
        }
    } else {
        resolve_scheduler_for_model(&args, &model_dir)?
    };

    let model_type = read_model_type(&model_dir)?;
    let architecture = ironmlx_lm::models::ModelArchitecture::from_model_type(&model_type)?;
    ensure_dflash2_serve_supported(&args, architecture, resolved_scheduler.scheduler_config)?;
    if resolve_prompt_lookup_config(&args)?.is_some() && !architecture.supports_prompt_lookup() {
        bail!(
            "ironmlx serve --prompt-lookup requires a causal scheduler model; `{model_type}` is not supported"
        );
    }
    // open_multimodal so Qwen VL checkpoints retain vision_tower.* keys.
    let mut loader = Loader::open_multimodal(&model_dir).context("Loader::open_multimodal")?;
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
    let vision_input = if architecture == ironmlx_lm::models::ModelArchitecture::DiffusionGemma {
        None
    } else {
        Some(
            ironmlx_lm::core::vision_input::VisionInputConfig::from_causal_loader(
                architecture,
                &loader,
            )?,
        )
    };

    match architecture {
        ironmlx_lm::models::ModelArchitecture::Qwen35Dense => {
            let model = if args.dflash2_model_dir.is_some() {
                ironmlx_lm::models::Qwen35Model::from_loader_dflash2(&mut loader)
                    .context("Qwen35Model::from_loader_dflash2")?
            } else {
                ironmlx_lm::models::Qwen35Model::from_loader(&loader)
                    .context("Qwen35Model::from_loader")?
            };
            if args.dflash2_model_dir.is_some() {
                serve_with_dflash2_model(
                    model,
                    tokenizer,
                    &args,
                    scheduler_config,
                    scheduler_runtime_profile,
                    static_memory_estimate,
                )
            } else if let Some(mtp_config) = mtp_config.clone() {
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
        ironmlx_lm::models::ModelArchitecture::Qwen35Moe => {
            match qwen_moe_serve_model(loader.config_raw_value()) {
                QwenMoeServeModel::Qwen35 => {
                    let model = ironmlx_lm::models::Qwen35MoeModel::from_loader(&loader)
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
                    let model = ironmlx_lm::models::Qwen36MoeModel::from_loader(&loader)
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
        ironmlx_lm::models::ModelArchitecture::Gemma4 => {
            let model = ironmlx_lm::models::Gemma4Model::from_loader(&loader)
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
        ironmlx_lm::models::ModelArchitecture::Glm4MoeLite => {
            let model = ironmlx_lm::models::Glm4MoeLiteModel::from_loader(&loader)
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
        ironmlx_lm::models::ModelArchitecture::Llama => {
            let model = ironmlx_lm::models::LlamaModel::from_loader(&loader)
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
        ironmlx_lm::models::ModelArchitecture::MiniCpmV46 => {
            // MiniCpmV46Model serves text + single-image VL (vision_input set above).
            let model = ironmlx_lm::models::minicpmv4_6::model_from_loader(&loader)
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
        ironmlx_lm::models::ModelArchitecture::DiffusionGemma => {
            let cfg = ironmlx_lm::models::DiffusionGemmaConfig::from_loader(&loader)
                .context("DiffusionGemmaConfig::from_loader")?;
            let vision_config = cfg
                .vision_config
                .clone()
                .ok_or_else(|| anyhow::anyhow!("DiffusionGemma config has no vision_config"))?;
            let image_token_id = cfg.image_token_id;
            let generation_config =
                ironmlx_lm::models::DiffusionGemmaGenerationConfig::from_loader(&loader)
                    .context("DiffusionGemmaGenerationConfig::from_loader")?;
            let model = ironmlx_lm::models::DiffusionGemmaModel::from_loader(&loader)
                .context("DiffusionGemmaModel::from_loader")?;
            let model_weight_bytes = loader.loaded_tensor_bytes();
            serve_with_diffusion_gemma_model(
                model,
                tokenizer,
                generation_config,
                model_weight_bytes,
                &args,
                ironmlx_lm::core::vision_input::VisionInputConfig::DiffusionGemma {
                    vision_config,
                    image_token_id,
                },
            )
        }
    }
}

#[cfg(test)]
mod scheduler_profile_tests {
    use ironmlx_runtime::core::scheduler_resolution::{
        check_loaded_scheduler_profile_health, default_scheduler_runtime_profile,
        SchedulerProfileSource,
    };
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use clap::Parser;

    use crate::cli::Command;
    use ironmlx_runtime::core::scheduler_profile_store::SchedulerProfileStore;
    use {
        ironmlx_runtime::core::engine_pool::EngineLoadPolicy,
        ironmlx_runtime::core::engine_pool::EngineModelManifest,
    };
    use {
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneCacheState,
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneProfileConfig,
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneProfileHealthStatus,
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeContext,
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile,
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeRule,
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeRuleCondition,
        ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneScenario,
        ironmlx_runtime::core::scheduler_autotune::SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
    };

    use super::{
        apply_adaptive_mtp_scheduler_defaults, build_engine_model_config_for_pool,
        ensure_dflash2_serve_supported, load_scheduler_profile_for_model, qwen_moe_serve_model,
        read_engine_pool_manifest, resolve_active_kv_offload_config,
        resolve_dflash2_prefix_lru_cache_config, resolve_dflash2_tensor_batch_width,
        resolve_memory_limit_bytes, resolve_model_ttl, resolve_paged_prefix_cache_config,
        resolve_prefix_lru_cache_config, resolve_prompt_lookup_config,
        resolve_scheduler_runtime_profile, resolve_scheduler_serve_config,
        resolve_serve_mtp_config, single_model_id, KvQuantArg, QwenMoeServeModel,
        ResolvedSchedulerRuntime, SchedulerServeConfig, ServeArgs,
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
                ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata::synthetic(
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
            kv_quant: KvQuantArg::None,
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

    #[test]
    fn single_model_id_prefers_explicit_public_identifier() {
        let mut args = base_args();
        args.model_id = Some("mlx-community/Qwen3.8-27B-4bit".to_string());

        assert_eq!(
            single_model_id(&args).expect("model id"),
            "mlx-community/Qwen3.8-27B-4bit"
        );
    }

    #[test]
    fn memory_limit_gigabytes_resolve_to_bytes() {
        assert_eq!(
            resolve_memory_limit_bytes(Some(2), "--memory-limit-total-gb").unwrap(),
            Some(2 * ironmlx_runtime::core::scheduler_resolution::BYTES_PER_GIB)
        );
        assert_eq!(
            resolve_memory_limit_bytes(None, "--memory-limit-total-gb").unwrap(),
            None
        );
    }

    #[test]
    fn dflash2_serve_policy_is_strictly_isolated_and_supports_multiple_sequences() {
        let draft_dir = unique_temp_dir("dflash2-serve-policy");
        std::fs::create_dir_all(&draft_dir).expect("create draft dir");
        let mut args = base_args();
        args.dflash2_model_dir = Some(draft_dir.clone());
        let config = SchedulerServeConfig {
            b_max: 1,
            ..SchedulerServeConfig::default()
        };
        ensure_dflash2_serve_supported(
            &args,
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            config,
        )
        .expect("valid isolated DFlash2 policy");

        let mut concurrent = config;
        concurrent.b_max = 2;
        ensure_dflash2_serve_supported(
            &args,
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            concurrent,
        )
        .expect("DFlash2 must accept multi-sequence mode");

        let mut empty = config;
        empty.b_max = 0;
        let error = ensure_dflash2_serve_supported(
            &args,
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            empty,
        )
        .expect_err("DFlash2 must reject zero max sequences");
        assert!(format!("{error:#}").contains("greater than zero"));

        args.mtp_model_dir = Some(draft_dir.clone());
        let error = ensure_dflash2_serve_supported(
            &args,
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            config,
        )
        .expect_err("DFlash2 must reject MTP mixing");
        assert!(format!("{error:#}").contains("cannot be combined with MTP"));
        std::fs::remove_dir_all(draft_dir).expect("remove draft dir");
    }

    #[test]
    fn dflash2_tensor_batch_width_uses_certified_default_and_max_sequence_cap() {
        let mut args = base_args();
        assert_eq!(resolve_dflash2_tensor_batch_width(&args, 8), (4, 4));
        assert_eq!(resolve_dflash2_tensor_batch_width(&args, 2), (4, 2));

        args.dflash2_tensor_batch_max_width = Some(6);
        assert_eq!(resolve_dflash2_tensor_batch_width(&args, 3), (6, 3));

        args.dflash2_tensor_batch_max_width = Some(1);
        assert_eq!(resolve_dflash2_tensor_batch_width(&args, 8), (1, 1));
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
            ironmlx_runtime::core::cache::active_kv::default_active_kv_offload_dir()
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
            ironmlx_runtime::core::cache::active_kv::default_active_kv_offload_dir()
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
            audio: None,
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
            audio: None,
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

        assert_eq!(
            config
                .scheduler_runtime_profile
                .as_ref()
                .expect("causal scheduler profile")
                .config
                .b_max,
            4
        );
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
        let prompt_lookup = ironmlx_runtime::core::prompt_lookup::PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 5,
            max_draft_tokens: 3,
            history_window_tokens: 4096,
            max_index_entries: 8192,
            cross_request: true,
        };
        let manifest_model = EngineModelManifest {
            audio: None,
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
                .as_ref()
                .expect("causal scheduler profile")
                .runtime_context
                .speculative
                .mode,
            ironmlx_runtime::core::scheduler_autotune::SchedulerSpeculativeMode::PromptLookup
        );
        std::fs::remove_dir_all(temp_dir).expect("cleanup");
    }

    #[test]
    fn serve_engine_pool_preserves_hybrid_source_config() {
        let temp_dir = unique_temp_dir("serve-engine-pool-hybrid");
        let model_dir = temp_dir.join("qwen-base");
        let mtp_dir = temp_dir.join("qwen-mtp");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        std::fs::create_dir_all(&mtp_dir).expect("create MTP dir");
        std::fs::write(model_dir.join("config.json"), r#"{"model_type":"qwen3_5"}"#)
            .expect("write model config");
        std::fs::write(
            mtp_dir.join("config.json"),
            r#"{"model_type":"qwen3_5_mtp"}"#,
        )
        .expect("write MTP config");
        let args = base_args();
        let prompt_lookup = ironmlx_runtime::core::prompt_lookup::PromptLookupConfig::default();
        let manifest_model = EngineModelManifest {
            audio: None,
            id: "qwen-hybrid".to_string(),
            path: model_dir,
            load_policy: EngineLoadPolicy::Lazy,
            default: true,
            scheduler_profile: None,
            mtp_model_dir: Some(mtp_dir),
            mtp_draft_tokens: Some(2),
            prompt_lookup: Some(prompt_lookup),
        };

        let config = build_engine_model_config_for_pool(&args, manifest_model, None, "test-host")
            .expect("build hybrid manifest config");

        assert!(config.mtp.is_some());
        assert_eq!(config.prompt_lookup, Some(prompt_lookup));
        assert_eq!(
            config
                .scheduler_runtime_profile
                .as_ref()
                .expect("causal scheduler profile")
                .runtime_context
                .speculative
                .mode,
            ironmlx_runtime::core::scheduler_autotune::SchedulerSpeculativeMode::QwenMtpPromptLookup
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
            ironmlx_lm::models::ModelArchitecture::Gemma4,
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
            ironmlx_lm::models::ModelArchitecture::Gemma4,
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
            ironmlx_lm::models::ModelArchitecture::Gemma4,
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
            ironmlx_lm::models::ModelArchitecture::Gemma4,
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
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
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
        args.prompt_lookup_cross_request = true;

        let error = resolve_prompt_lookup_config(&args).expect_err("source must be enabled");
        assert!(error
            .to_string()
            .contains("source parameters require --prompt-lookup"));
    }

    #[test]
    fn serve_prompt_lookup_cross_request_is_explicit_opt_in() {
        let mut args = base_args();
        args.prompt_lookup = true;

        let local = resolve_prompt_lookup_config(&args)
            .expect("resolve local PromptLookup")
            .expect("PromptLookup config");
        assert!(!local.cross_request);

        args.prompt_lookup_cross_request = true;
        let shared = resolve_prompt_lookup_config(&args)
            .expect("resolve cross-request PromptLookup")
            .expect("PromptLookup config");
        assert!(shared.cross_request);
    }

    #[test]
    fn serve_prompt_lookup_accepts_neural_source_configuration() {
        let mut args = base_args();
        args.prompt_lookup = true;
        args.mtp_model_dir = Some(PathBuf::from("/tmp/mtp"));

        assert!(resolve_prompt_lookup_config(&args)
            .expect("hybrid sources are valid")
            .is_some());
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
            ironmlx_runtime::core::cache::prefix_store::DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE
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
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
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
    fn serve_mtp_config_accepts_qwen36_architecture_default_draft_tokens() {
        let temp_dir = unique_temp_dir("serve-mtp-qwen36-default");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        let mut args = base_args();
        args.mtp_model_dir = Some(temp_dir.clone());

        let cfg = resolve_serve_mtp_config(
            &args,
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
            &qwen36_dense_27b_raw_config(),
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
    fn serve_mtp_config_accepts_batched_scheduler() {
        let temp_dir = unique_temp_dir("serve-mtp-bmax");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        let mut args = base_args();
        args.mtp_model_dir = Some(temp_dir.clone());

        let cfg = resolve_serve_mtp_config(
            &args,
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
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
            ironmlx_lm::models::ModelArchitecture::Gemma4,
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
            ironmlx_lm::models::ModelArchitecture::Gemma4,
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
    fn dflash2_prefix_lru_cache_accepts_standalone_memory_budget() {
        let mut args = base_args();
        args.prefix_lru_cache_max_bytes = Some(4096);

        let config = resolve_dflash2_prefix_lru_cache_config(&args)
            .expect("DFlash2 prefix config")
            .expect("enabled");

        assert_eq!(config.max_bytes, 4096);
    }

    #[test]
    fn dflash2_prefix_lru_cache_rejects_zero_capacity() {
        let mut args = base_args();
        args.prefix_lru_cache_max_bytes = Some(0);

        let error = resolve_dflash2_prefix_lru_cache_config(&args)
            .expect_err("zero DFlash2 prefix capacity");

        assert!(error.to_string().contains("max_bytes must be > 0"));
    }

    #[test]
    fn serve_mtp_config_rejects_non_qwen_or_gemma4_architecture() {
        let temp_dir = unique_temp_dir("serve-mtp-non-qwen-gemma4");
        std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
        let mut args = base_args();
        args.mtp_model_dir = Some(temp_dir.clone());

        let err = resolve_serve_mtp_config(
            &args,
            ironmlx_lm::models::ModelArchitecture::Llama,
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
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
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
            ironmlx_lm::models::ModelArchitecture::Qwen35Dense,
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

use {
    ironmlx_runtime::core::scheduler_resolution::read_model_type,
    ironmlx_runtime::core::scheduler_resolution::resolve_memory_limit_bytes,
    ironmlx_runtime::core::scheduler_resolution::ResolvedSchedulerRuntime,
    ironmlx_runtime::core::scheduler_resolution::SchedulerResolutionOptions,
    ironmlx_runtime::core::scheduler_resolution::SchedulerServeConfig,
    ironmlx_runtime::core::scheduler_resolution::DEFAULT_MAX_CACHE_CAP,
};
impl From<&ServeArgs> for SchedulerResolutionOptions {
    fn from(args: &ServeArgs) -> Self {
        Self {
            prefill_chunk_size: args.prefill_chunk_size,
            b_max: args.b_max,
            admission_deadline_ms: args.admission_deadline_ms,
            admission_queue_max: args.admission_queue_max,
            max_cache_cap: args.max_cache_cap,
            decode_cadence_mid_chunk_cap: args.decode_cadence_mid_chunk_cap,
            scheduler_profile: args.scheduler_profile.clone(),
            kv_quantization: args.kv_quant.profile_context(),
            paged_prefix_cache_dir: args.paged_prefix_cache_dir.clone(),
            paged_prefix_cache_block_size: args.paged_prefix_cache_block_size,
            paged_prefix_cache_max_pages: args.paged_prefix_cache_max_pages,
            prefix_lru_cache_max_bytes: args.prefix_lru_cache_max_bytes,
            ssd_prefix_cache_max_gb: args.ssd_prefix_cache_max_gb,
            active_kv_offload: args.active_kv_offload,
            memory_limit_total_gb: args.memory_limit_total_gb,
            memory_limit_model_gb: args.memory_limit_model_gb,
        }
    }
}

pub(crate) fn engine_runtime_config(
    args: &ServeArgs,
) -> Result<server::engine::EnginePoolRuntimeConfig> {
    Ok(server::engine::EnginePoolRuntimeConfig {
        network: args.resolved_network_config()?,
        options: ironmlx_runtime::core::runtime_config::EngineRuntimeOptions {
            kv_cache_turboquant_bits: args.kv_quant.turboquant_bits(),
            scheduler_autotune_report: args.scheduler_autotune_report,
            paged_prefix_cache: resolve_engine_paged_prefix_cache_settings(args)?,
            prefix_lru_cache_max_bytes: args.prefix_lru_cache_max_bytes,
            model_ttl: resolve_model_ttl(args)?,
            memory_limits: ironmlx_runtime::core::runtime_config::EnginePoolMemoryLimits {
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
        },
    })
}

#[cfg(test)]
use ironmlx_runtime::core::scheduler_resolution::SchedulerProfileLoad;
