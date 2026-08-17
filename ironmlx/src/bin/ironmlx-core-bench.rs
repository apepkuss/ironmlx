//! Minimal in-process benchmark for ironmlx model-core text generation paths.
//!
//! This binary intentionally bypasses HTTP so scheduler/model gaps can be
//! measured without server parsing, SSE, or client timing noise.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::cache::{
    ActiveKvOffloadConfig, ActiveKvOffloadHealth, ActiveKvOffloadSharedStats,
    ActiveKvOffloadStatus, PagedPrefixCacheConfig, TurboQuantKVBits,
    DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE,
};
use ironmlx::core::scheduler::DenseVlMethods;
use ironmlx::core::speculative::{
    resolve_mtp_draft_tokens, MtpDraftCapObservation, MtpDraftTokensArg, MtpSpeculativeConfig,
    MtpSpeculativeModel, MtpSpeculativeStats, MtpTextGenerationStream,
};
use ironmlx::core::{GenerateRequest, GenerationStream, Loader, Model, Sampler, Scheduler};
use ironmlx::models::gemma4::{
    Gemma4AssistantModel, Gemma4DrafterGenerationStream, Gemma4DrafterTraceWindow,
};
use ironmlx::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF;
use ironmlx::models::{
    Gemma4Model, Glm4MoeLiteModel, LlamaModel, ModelArchitecture, Qwen35Model, Qwen35MoeModel,
    Qwen36MoeModel,
};
use ironmlx::Tokenizer;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-core-bench",
    about = "In-process model-core benchmark for ironmlx text generation paths",
    version
)]
struct Args {
    /// Local model directory.
    #[arg(long)]
    model: PathBuf,

    /// File containing an already-rendered raw model prompt. Repeat this flag
    /// to admit a concurrent Gemma4 drafter batch in scheduler-text mode.
    #[arg(long, required = true)]
    prompt_file: Vec<PathBuf>,

    /// Core path to measure.
    #[arg(long, value_enum)]
    mode: BenchMode,

    /// MTP model directory, required by mtp-text mode and optional for
    /// scheduler-text mode.
    #[arg(long = "mtp-model-dir")]
    mtp_model_dir: Option<PathBuf>,

    /// Maximum MTP draft tokens per speculative window. If omitted, ironmlx
    /// picks a model-aware default from local benchmark policy.
    #[arg(long)]
    mtp_draft_tokens: Option<usize>,

    /// Record the first N Gemma4 drafter speculative windows into JSON.
    #[arg(long, default_value_t = 0)]
    mtp_trace_windows: usize,

    /// Number of generated tokens per request.
    #[arg(long, default_value_t = 16)]
    max_tokens: usize,

    /// Timed runs.
    #[arg(long, default_value_t = 7)]
    runs: usize,

    /// For Qwen scheduler MTP benchmarks, interleave ordinary scheduler runs
    /// in the same loaded-model process and write them to this JSON path.
    #[arg(long = "scheduler-baseline-out")]
    scheduler_baseline_out: Option<PathBuf>,

    /// Warmup runs, excluded from summary.
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,

    /// Prefill chunk size passed to GenerateRequest.
    #[arg(long, default_value_t = 2048)]
    prefill_chunk_size: usize,

    /// KV cache quantization used by attention reads.
    #[arg(long = "kv-quant", value_enum, default_value_t = KvQuantBenchArg::None)]
    kv_quant: KvQuantBenchArg,

    /// Enable paged full-attention KV storage and paged SSD prefix cache under
    /// this directory. Required when benchmarking --active-kv-offload hot/cold
    /// page tiering.
    #[arg(long = "paged-prefix-cache-dir")]
    paged_prefix_cache_dir: Option<PathBuf>,

    /// Tokens per physical K/V page for --paged-prefix-cache-dir.
    #[arg(long = "paged-prefix-cache-block-size", default_value_t = DEFAULT_PAGED_PREFIX_CACHE_BLOCK_SIZE)]
    paged_prefix_cache_block_size: i32,

    /// Maximum physical pages per full-attention layer cache. If omitted,
    /// defaults to ceil(b_max * effective_cap_max / block_size).
    #[arg(long = "paged-prefix-cache-max-pages")]
    paged_prefix_cache_max_pages: Option<i32>,

    /// Enable experimental Active KV Cache offload for scheduler-text
    /// benchmarks. This benchmark requires paged KV and kv-quant=none so the
    /// measured path is transparent hot/cold page residency.
    #[arg(long = "active-kv-offload", default_value_t = false)]
    active_kv_offload: bool,

    /// Directory used by --active-kv-offload for temporary benchmark payloads.
    /// If omitted, a unique directory under std::env::temp_dir() is used.
    #[arg(long = "active-kv-offload-dir")]
    active_kv_offload_dir: Option<PathBuf>,

    /// Force Active KV hot/cold resident window size in physical pages.
    /// Intended for benchmark stress runs; omitted means budget-based sizing.
    #[arg(long = "active-kv-hot-window-pages")]
    active_kv_hot_window_pages: Option<i32>,

    /// Force Active KV streaming chunk size in physical pages.
    /// Intended for benchmark stress runs; omitted means the production default.
    #[arg(long = "active-kv-chunk-pages")]
    active_kv_chunk_pages: Option<i32>,

    /// Scheduler batch capacity, only used by scheduler-text mode. The default
    /// leaves enough admission headroom for a full-cap single benchmark request
    /// under the scheduler's 85% soft memory limit.
    #[arg(long, default_value_t = 2)]
    b_max: usize,

    /// Scheduler effective cap max. Defaults to prompt_len + max_tokens,
    /// floored at ironmlx's minimum KV cache cap.
    #[arg(long)]
    effective_cap_max: Option<usize>,

    /// JSON output path.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
enum BenchMode {
    #[value(name = "gs-text")]
    #[serde(rename = "gs-text")]
    Gs,
    #[value(name = "mtp-text")]
    #[serde(rename = "mtp-text")]
    Mtp,
    #[value(name = "scheduler-text")]
    #[serde(rename = "scheduler-text")]
    Scheduler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
enum KvQuantBenchArg {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "turbo3")]
    Turbo3,
    #[serde(rename = "turbo4")]
    Turbo4,
    #[value(name = "k3v4")]
    #[serde(rename = "k3v4")]
    K3V4,
}

impl KvQuantBenchArg {
    fn turboquant_bits(self) -> Option<TurboQuantKVBits> {
        match self {
            Self::None => None,
            Self::Turbo3 => Some(TurboQuantKVBits::K3V3),
            Self::Turbo4 => Some(TurboQuantKVBits::K4V4),
            Self::K3V4 => Some(TurboQuantKVBits::K3V4),
        }
    }
}

#[derive(Serialize)]
struct BenchOutput {
    meta: Meta,
    summary: Summary,
    warmups: Vec<Record>,
    records: Vec<Record>,
}

#[derive(Clone, Serialize)]
struct Meta {
    backend: &'static str,
    mode: BenchMode,
    speculative_source: Option<&'static str>,
    model_dir: String,
    mtp_model_dir: Option<String>,
    mtp_draft_tokens: Option<usize>,
    mtp_trace_windows: usize,
    prompt_file: String,
    prompt_tokens: usize,
    scheduler_prompt_files: Vec<String>,
    scheduler_prompt_tokens: Vec<usize>,
    scheduler_batch_width: usize,
    max_tokens: usize,
    prefill_chunk_size: usize,
    kv_quant: KvQuantBenchArg,
    paged_prefix_cache_dir: Option<String>,
    paged_prefix_cache_block_size: i32,
    paged_prefix_cache_max_pages: Option<i32>,
    active_kv_offload: bool,
    active_kv_offload_dir: Option<String>,
    active_kv_hot_window_pages: Option<i32>,
    active_kv_chunk_pages: Option<i32>,
    b_max: usize,
    effective_cap_max: usize,
    warmup_runs: usize,
    measured_runs: usize,
    load_ms: f64,
    device_name: Option<String>,
    ironmlx_version: &'static str,
}

#[derive(Serialize)]
struct Summary {
    runs: usize,
    valid_runs: usize,
    ttft_ms: Stats,
    e2e_ms: Stats,
    decode_time_ms: Stats,
    generation_tps: Stats,
    aggregate_generation_tps: Stats,
}

#[derive(Serialize)]
struct Stats {
    p50: Option<f64>,
    p95: Option<f64>,
    mean: Option<f64>,
}

#[derive(Serialize)]
struct Record {
    mode: BenchMode,
    ttft_ms: f64,
    e2e_ms: f64,
    decode_time_ms: f64,
    generated_tokens: usize,
    generated_token_ids: Vec<u32>,
    generated_text: String,
    generation_tps: f64,
    aggregate_generation_tps: f64,
    finish_reason: Option<&'static str>,
    valid: bool,
    mtp_stats: Option<MtpRecordStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mtp_trace: Option<Vec<MtpTraceWindowRecord>>,
    active_kv_stats: Option<ActiveKvRecordStats>,
    scheduler_requests: Vec<SchedulerRequestRecord>,
}

#[derive(Serialize)]
struct SchedulerRequestRecord {
    request_index: usize,
    prompt_file: String,
    ttft_ms: f64,
    e2e_ms: f64,
    generated_tokens: usize,
    generated_token_ids: Vec<u32>,
    generated_text: String,
    finish_reason: Option<&'static str>,
    valid: bool,
}

struct SchedulerRequestState {
    request_index: usize,
    request_id: ironmlx::core::RequestId,
    prompt_file: String,
    ttft_ms: Option<f64>,
    e2e_ms: Option<f64>,
    generated_token_ids: Vec<u32>,
    finish_reason: Option<&'static str>,
}

struct GeneratedOutput {
    token_ids: Vec<u32>,
    text: String,
}

#[derive(Serialize)]
struct MtpRecordStats {
    windows: usize,
    drafted_tokens: usize,
    accepted_draft_tokens: usize,
    draft_attempts_by_position: Vec<usize>,
    draft_accepts_by_position: Vec<usize>,
    rollback_count: usize,
    mtp_cache_reuse_count: usize,
    mtp_cache_reused_tokens: usize,
    draft_budget_reductions: usize,
    draft_budget_increases: usize,
    acceptance_rate: Option<f64>,
    draft_forward_us: u64,
    verify_forward_us: u64,
    projection_us: u64,
    sampling_us: u64,
    draft_host_sync_count: usize,
    draft_host_sync_us: u64,
    verify_accept_host_sync_count: usize,
    verify_accept_host_sync_us: u64,
    main_rollback_us: u64,
    mtp_cache_commit_us: u64,
    mtp_prefill_cache_commit_us: u64,
    mtp_decode_cache_commit_us: u64,
    mtp_cache_restore_us: u64,
    draft_cap_observations: Vec<MtpDraftCapObservation>,
    draft_cap_observation_dropped_windows: usize,
}

#[derive(Serialize)]
struct MtpTraceWindowRecord {
    history_len: usize,
    verify_start_pos: i32,
    draft_tokens: Vec<u32>,
    verified_tokens: Vec<u32>,
    resolved_tokens: Vec<u32>,
    accepted_draft_len: usize,
}

impl From<Gemma4DrafterTraceWindow> for MtpTraceWindowRecord {
    fn from(window: Gemma4DrafterTraceWindow) -> Self {
        Self {
            history_len: window.history_len,
            verify_start_pos: window.verify_start_pos,
            draft_tokens: window.draft_tokens,
            verified_tokens: window.verified_tokens,
            resolved_tokens: window.resolved_tokens,
            accepted_draft_len: window.accepted_draft_len,
        }
    }
}

#[derive(Serialize)]
struct ActiveKvRecordStats {
    enabled: bool,
    status: ActiveKvOffloadStatus,
    active: bool,
    degraded: bool,
    mode: &'static str,
    storage_dir: String,
    resident_pages: usize,
    offloaded_pages: usize,
    loading_pages: usize,
    dirty_pages: usize,
    parked_requests: usize,
    offloaded_bytes: usize,
    swap_out_count: u64,
    swap_in_count: u64,
    stream_read_count: u64,
    swap_error_count: u64,
    last_swap_out_us: u64,
    last_swap_in_us: u64,
}

impl From<ActiveKvOffloadHealth> for ActiveKvRecordStats {
    fn from(health: ActiveKvOffloadHealth) -> Self {
        Self {
            enabled: health.enabled,
            status: health.status,
            active: health.active,
            degraded: health.degraded,
            mode: health.mode,
            storage_dir: health.storage_dir.display().to_string(),
            resident_pages: health.resident_pages,
            offloaded_pages: health.offloaded_pages,
            loading_pages: health.loading_pages,
            dirty_pages: health.dirty_pages,
            parked_requests: health.parked_requests,
            offloaded_bytes: health.offloaded_bytes,
            swap_out_count: health.swap_out_count,
            swap_in_count: health.swap_in_count,
            stream_read_count: health.stream_read_count,
            swap_error_count: health.swap_error_count,
            last_swap_out_us: health.last_swap_out_us,
            last_swap_in_us: health.last_swap_in_us,
        }
    }
}

#[derive(Clone)]
struct BenchSchedulerFeatures {
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
}

#[derive(Clone, Copy)]
struct SchedulerBenchConfig<'a> {
    features: &'a BenchSchedulerFeatures,
    effective_cap_max: usize,
}

struct RecordInput {
    mode: BenchMode,
    ttft_ms: f64,
    e2e_ms: f64,
    generated: GeneratedOutput,
    finish_reason: Option<&'static str>,
    max_tokens: usize,
    mtp_stats: Option<MtpRecordStats>,
    mtp_trace: Option<Vec<MtpTraceWindowRecord>>,
    active_kv_stats: Option<ActiveKvRecordStats>,
}

impl From<MtpSpeculativeStats> for MtpRecordStats {
    fn from(stats: MtpSpeculativeStats) -> Self {
        Self {
            windows: stats.windows,
            drafted_tokens: stats.drafted_tokens,
            accepted_draft_tokens: stats.accepted_draft_tokens,
            draft_attempts_by_position: stats.draft_attempts_by_position,
            draft_accepts_by_position: stats.draft_accepts_by_position,
            rollback_count: stats.rollback_count,
            mtp_cache_reuse_count: stats.mtp_cache_reuse_count,
            mtp_cache_reused_tokens: stats.mtp_cache_reused_tokens,
            draft_budget_reductions: stats.draft_budget_reductions,
            draft_budget_increases: stats.draft_budget_increases,
            acceptance_rate: (stats.drafted_tokens > 0)
                .then(|| stats.accepted_draft_tokens as f64 / stats.drafted_tokens as f64),
            draft_forward_us: stats.draft_forward_us,
            verify_forward_us: stats.verify_forward_us,
            projection_us: stats.projection_us,
            sampling_us: stats.sampling_us,
            draft_host_sync_count: stats.draft_host_sync_count,
            draft_host_sync_us: stats.draft_host_sync_us,
            verify_accept_host_sync_count: stats.verify_accept_host_sync_count,
            verify_accept_host_sync_us: stats.verify_accept_host_sync_us,
            main_rollback_us: stats.main_rollback_us,
            mtp_cache_commit_us: stats.mtp_cache_commit_us,
            mtp_prefill_cache_commit_us: stats.mtp_prefill_cache_commit_us,
            mtp_decode_cache_commit_us: stats.mtp_decode_cache_commit_us,
            mtp_cache_restore_us: stats.mtp_cache_restore_us,
            draft_cap_observations: stats.draft_cap_observations,
            draft_cap_observation_dropped_windows: stats.draft_cap_observation_dropped_windows,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let load_started = Instant::now();
    let loader = Loader::open(&args.model).context("Loader::open")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let architecture = ModelArchitecture::from_config_value(loader.config_raw_value())?;

    match architecture {
        ModelArchitecture::Qwen35Dense => {
            let model = Qwen35Model::from_loader(&loader).context("Qwen35Model::from_loader")?;
            let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
            run_for_qwen_model(
                &model,
                &tokenizer,
                &args,
                loader.config_raw_value(),
                load_ms,
            )
        }
        ModelArchitecture::Qwen35Moe => {
            if args.mode == BenchMode::Mtp
                && ironmlx::models::is_qwen36_moe_config(loader.config_raw_value())
            {
                let model =
                    Qwen36MoeModel::from_loader(&loader).context("Qwen36MoeModel::from_loader")?;
                let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
                run_for_qwen_model(
                    &model,
                    &tokenizer,
                    &args,
                    loader.config_raw_value(),
                    load_ms,
                )
            } else {
                let model =
                    Qwen35MoeModel::from_loader(&loader).context("Qwen35MoeModel::from_loader")?;
                let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
                run_for_qwen_model(
                    &model,
                    &tokenizer,
                    &args,
                    loader.config_raw_value(),
                    load_ms,
                )
            }
        }
        ModelArchitecture::Glm4MoeLite => {
            let model =
                Glm4MoeLiteModel::from_loader(&loader).context("Glm4MoeLiteModel::from_loader")?;
            let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
            run_for_model(&model, &tokenizer, &args, load_ms)
        }
        ModelArchitecture::Gemma4 => {
            let model = Gemma4Model::from_loader(&loader).context("Gemma4Model::from_loader")?;
            let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
            run_for_gemma4_model(
                &model,
                &tokenizer,
                &args,
                loader.config_raw_value(),
                load_ms,
            )
        }
        ModelArchitecture::Llama => {
            let model = LlamaModel::from_loader(&loader).context("LlamaModel::from_loader")?;
            let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
            run_for_model(&model, &tokenizer, &args, load_ms)
        }
        ModelArchitecture::MiniCpmV46 => {
            // Full MiniCpmV46Model (text + optional SigLIP vision tower).
            let model = ironmlx::models::minicpmv4_6::model_from_loader(&loader)
                .context("minicpmv4_6::model_from_loader")?;
            let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
            run_for_model(&model, &tokenizer, &args, load_ms)
        }
        ModelArchitecture::DiffusionGemma => Err(anyhow!(
            "ironmlx-core-bench measures causal prefill/decode models; DiffusionGemma uses block diffusion"
        )),
    }
}

fn validate_args(args: &Args) -> Result<()> {
    if args.prompt_file.is_empty() {
        return Err(anyhow!("at least one --prompt-file is required"));
    }
    if args.prompt_file.len() > 1 && args.mode != BenchMode::Scheduler {
        return Err(anyhow!(
            "multiple --prompt-file values require --mode scheduler-text"
        ));
    }
    if args.prompt_file.len() > args.b_max {
        return Err(anyhow!(
            "scheduler prompt count {} exceeds --b-max {}",
            args.prompt_file.len(),
            args.b_max
        ));
    }
    if args.scheduler_baseline_out.is_some()
        && (args.mode != BenchMode::Scheduler
            || args.mtp_model_dir.is_none()
            || args.prompt_file.len() != 1)
    {
        return Err(anyhow!(
            "--scheduler-baseline-out requires single-prompt --mode scheduler-text with --mtp-model-dir"
        ));
    }
    if args.paged_prefix_cache_block_size <= 0 {
        return Err(anyhow!(
            "--paged-prefix-cache-block-size must be > 0, got {}",
            args.paged_prefix_cache_block_size
        ));
    }
    if let Some(max_pages) = args.paged_prefix_cache_max_pages {
        if max_pages <= 0 {
            return Err(anyhow!(
                "--paged-prefix-cache-max-pages must be > 0, got {max_pages}"
            ));
        }
    }
    if args.active_kv_offload_dir.is_some() && !args.active_kv_offload {
        return Err(anyhow!(
            "--active-kv-offload-dir requires --active-kv-offload"
        ));
    }
    if args.active_kv_hot_window_pages.is_some() && !args.active_kv_offload {
        return Err(anyhow!(
            "--active-kv-hot-window-pages requires --active-kv-offload"
        ));
    }
    if args.active_kv_chunk_pages.is_some() && !args.active_kv_offload {
        return Err(anyhow!(
            "--active-kv-chunk-pages requires --active-kv-offload"
        ));
    }
    if let Some(hot_window_pages) = args.active_kv_hot_window_pages {
        if hot_window_pages <= 0 {
            return Err(anyhow!(
                "--active-kv-hot-window-pages must be > 0, got {hot_window_pages}"
            ));
        }
    }
    if let Some(chunk_pages) = args.active_kv_chunk_pages {
        if chunk_pages <= 0 {
            return Err(anyhow!(
                "--active-kv-chunk-pages must be > 0, got {chunk_pages}"
            ));
        }
    }
    if args.active_kv_offload {
        if args.mode != BenchMode::Scheduler {
            return Err(anyhow!(
                "--active-kv-offload is only supported with --mode scheduler-text"
            ));
        }
        if args.paged_prefix_cache_dir.is_none() {
            return Err(anyhow!(
                "--active-kv-offload hot/cold benchmark requires --paged-prefix-cache-dir"
            ));
        }
        if args.kv_quant != KvQuantBenchArg::None {
            return Err(anyhow!(
                "--active-kv-offload hot/cold benchmark requires --kv-quant none"
            ));
        }
    }
    match args.mode {
        BenchMode::Mtp => {
            let mtp_dir = args
                .mtp_model_dir
                .as_ref()
                .ok_or_else(|| anyhow!("--mode mtp-text requires --mtp-model-dir"))?;
            validate_mtp_dir(mtp_dir)?;
            if let Some(draft_tokens) = args.mtp_draft_tokens {
                MtpSpeculativeConfig::new(draft_tokens, Sampler::greedy())?;
            }
        }
        BenchMode::Scheduler => {
            if let Some(mtp_dir) = args.mtp_model_dir.as_ref() {
                validate_mtp_dir(mtp_dir)?;
                if let Some(draft_tokens) = args.mtp_draft_tokens {
                    MtpSpeculativeConfig::new(draft_tokens, Sampler::greedy())?;
                }
            }
        }
        BenchMode::Gs => {
            if args.mtp_model_dir.is_some() {
                return Err(anyhow!(
                    "--mtp-model-dir is only supported with mtp-text or scheduler-text"
                ));
            }
        }
    }
    Ok(())
}

fn resolve_scheduler_features(
    args: &Args,
    effective_cap_max: usize,
) -> Result<BenchSchedulerFeatures> {
    let paged_prefix_cache = resolve_paged_prefix_cache(args, effective_cap_max)?;
    let active_kv_offload = resolve_active_kv_offload(args)?;
    Ok(BenchSchedulerFeatures {
        paged_prefix_cache,
        active_kv_offload,
    })
}

fn resolve_paged_prefix_cache(
    args: &Args,
    effective_cap_max: usize,
) -> Result<Option<PagedPrefixCacheConfig>> {
    let Some(root) = args.paged_prefix_cache_dir.as_ref() else {
        return Ok(None);
    };
    let root = expand_home_path(root)?;
    let block_size = args.paged_prefix_cache_block_size;
    if block_size <= 0 {
        return Err(anyhow!(
            "--paged-prefix-cache-block-size must be > 0, got {block_size}"
        ));
    }
    let max_pages = match args.paged_prefix_cache_max_pages {
        Some(max_pages) if max_pages > 0 => max_pages,
        Some(max_pages) => {
            return Err(anyhow!(
                "--paged-prefix-cache-max-pages must be > 0, got {max_pages}"
            ));
        }
        None => default_paged_prefix_cache_max_pages(args, effective_cap_max)?,
    };
    PagedPrefixCacheConfig::new(&root, bench_model_id(args), block_size, max_pages).map(Some)
}

fn resolve_active_kv_offload(args: &Args) -> Result<ActiveKvOffloadConfig> {
    if !args.active_kv_offload {
        return Ok(ActiveKvOffloadConfig::disabled());
    }
    let root = match args.active_kv_offload_dir.as_ref() {
        Some(root) => expand_home_path(root)?,
        None => default_bench_active_kv_offload_dir(),
    };
    Ok(ActiveKvOffloadConfig::enabled(root)
        .with_hot_window_pages_override(args.active_kv_hot_window_pages)
        .with_chunk_pages_override(args.active_kv_chunk_pages))
}

fn default_paged_prefix_cache_max_pages(args: &Args, effective_cap_max: usize) -> Result<i32> {
    let block_size = usize::try_from(args.paged_prefix_cache_block_size)
        .context("--paged-prefix-cache-block-size must fit usize")?;
    let total_tokens = args
        .b_max
        .checked_mul(effective_cap_max)
        .context("b_max * effective_cap_max overflowed while resolving paged prefix max pages")?;
    let pages = ceil_div_usize(total_tokens, block_size).max(1);
    i32::try_from(pages).context("--paged-prefix-cache-max-pages exceeds i32::MAX")
}

fn ceil_div_usize(lhs: usize, rhs: usize) -> usize {
    debug_assert!(rhs > 0);
    lhs / rhs + usize::from(!lhs.is_multiple_of(rhs))
}

fn bench_model_id(args: &Args) -> String {
    args.model
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| args.model.display().to_string())
}

fn default_bench_active_kv_offload_dir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "ironmlx-core-bench-active-kv-{}-{nanos}",
        std::process::id()
    ))
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

fn validate_mtp_dir(mtp_dir: &Path) -> Result<()> {
    if !mtp_dir.exists() {
        return Err(anyhow!(
            "--mtp-model-dir must point to a local directory (got '{}')",
            mtp_dir.display()
        ));
    }
    Ok(())
}

fn primary_prompt_file(args: &Args) -> &Path {
    args.prompt_file
        .first()
        .expect("clap and validate_args require at least one prompt file")
}

fn read_prompt_ids(tokenizer: &Tokenizer, prompt_file: &Path) -> Result<Vec<u32>> {
    let rendered_prompt = std::fs::read_to_string(prompt_file)
        .with_context(|| format!("reading {}", prompt_file.display()))?;
    let prompt_ids = tokenizer.encode(&rendered_prompt, false)?;
    if prompt_ids.is_empty() {
        return Err(anyhow!(
            "prompt_file {} encoded to zero tokens",
            prompt_file.display()
        ));
    }
    Ok(prompt_ids)
}

fn read_scheduler_prompt_ids(tokenizer: &Tokenizer, args: &Args) -> Result<Vec<Vec<u32>>> {
    args.prompt_file
        .iter()
        .map(|prompt_file| read_prompt_ids(tokenizer, prompt_file))
        .collect()
}

fn scheduler_prompt_files(args: &Args) -> Vec<String> {
    args.prompt_file
        .iter()
        .map(|path| path.display().to_string())
        .collect()
}

fn run_for_model<M>(model: &M, tokenizer: &Tokenizer, args: &Args, load_ms: f64) -> Result<()>
where
    M: Model + DenseVlMethods,
{
    if args.prompt_file.len() != 1 {
        return Err(anyhow!(
            "multiple --prompt-file values are only supported for Gemma4 scheduler drafter benchmarks"
        ));
    }
    if args.mtp_model_dir.is_some() {
        return Err(anyhow!(
            "--mtp-model-dir is only supported for Qwen dense/MoE text models"
        ));
    }
    let prompt_file = primary_prompt_file(args);
    let prompt_ids = read_prompt_ids(tokenizer, prompt_file)?;

    let effective_cap_max = args.effective_cap_max.unwrap_or_else(|| {
        prompt_ids
            .len()
            .saturating_add(args.max_tokens)
            .max(MIN_KV_CACHE_CAP_FOR_GPU_PERF as usize)
    });
    let scheduler_features = resolve_scheduler_features(args, effective_cap_max)?;
    let scheduler_config = SchedulerBenchConfig {
        features: &scheduler_features,
        effective_cap_max,
    };

    let mut warmups = Vec::with_capacity(args.warmup_runs);
    for _ in 0..args.warmup_runs {
        warmups.push(run_once(
            model,
            tokenizer,
            &prompt_ids,
            args,
            scheduler_config,
        )?);
    }

    let mut records = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        records.push(run_once(
            model,
            tokenizer,
            &prompt_ids,
            args,
            scheduler_config,
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-core",
            mode: args.mode,
            speculative_source: None,
            model_dir: args.model.display().to_string(),
            mtp_model_dir: None,
            mtp_draft_tokens: None,
            mtp_trace_windows: args.mtp_trace_windows,
            prompt_file: prompt_file.display().to_string(),
            prompt_tokens: prompt_ids.len(),
            scheduler_prompt_files: scheduler_prompt_files(args),
            scheduler_prompt_tokens: vec![prompt_ids.len()],
            scheduler_batch_width: 1,
            max_tokens: args.max_tokens,
            prefill_chunk_size: args.prefill_chunk_size,
            kv_quant: args.kv_quant,
            paged_prefix_cache_dir: scheduler_features
                .paged_prefix_cache
                .as_ref()
                .map(|config| config.root.display().to_string()),
            paged_prefix_cache_block_size: args.paged_prefix_cache_block_size,
            paged_prefix_cache_max_pages: scheduler_features
                .paged_prefix_cache
                .as_ref()
                .map(|config| config.max_pages),
            active_kv_offload: scheduler_features.active_kv_offload.enabled,
            active_kv_offload_dir: scheduler_features.active_kv_offload.enabled.then(|| {
                scheduler_features
                    .active_kv_offload
                    .root
                    .display()
                    .to_string()
            }),
            active_kv_hot_window_pages: scheduler_features
                .active_kv_offload
                .hot_window_pages_override,
            active_kv_chunk_pages: scheduler_features.active_kv_offload.chunk_pages_override,
            b_max: args.b_max,
            effective_cap_max,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            load_ms,
            device_name: mlx::memory::snapshot().device_name,
            ironmlx_version: env!("CARGO_PKG_VERSION"),
        },
        summary: summarize(&records),
        warmups,
        records,
    };

    std::fs::write(&args.out, serde_json::to_string_pretty(&output)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;
    Ok(())
}

fn run_for_qwen_model<M>(
    model: &M,
    tokenizer: &Tokenizer,
    args: &Args,
    raw_config: &serde_json::Value,
    load_ms: f64,
) -> Result<()>
where
    M: MtpSpeculativeModel + DenseVlMethods,
{
    let prompt_ids = read_scheduler_prompt_ids(tokenizer, args)?;
    let primary_prompt_ids = prompt_ids
        .first()
        .expect("validated prompt list must be non-empty");

    let effective_cap_max = args.effective_cap_max.unwrap_or_else(|| {
        prompt_ids
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .saturating_add(args.max_tokens)
            .max(MIN_KV_CACHE_CAP_FOR_GPU_PERF as usize)
    });
    let scheduler_features = resolve_scheduler_features(args, effective_cap_max)?;
    let scheduler_config = SchedulerBenchConfig {
        features: &scheduler_features,
        effective_cap_max,
    };

    let mtp_draft_tokens = args.mtp_model_dir.as_ref().map(|_| {
        resolve_mtp_draft_tokens(
            raw_config,
            args.mtp_draft_tokens
                .map(MtpDraftTokensArg::Explicit)
                .unwrap_or(MtpDraftTokensArg::Omitted),
        )
    });

    let mtp = if args.mtp_model_dir.is_some() {
        let mtp_dir = args
            .mtp_model_dir
            .as_ref()
            .ok_or_else(|| anyhow!("--mtp-model-dir is required for MTP benchmarks"))?;
        let mtp_loader = Loader::open_mtp(mtp_dir).context("Loader::open_mtp")?;
        Some(
            model
                .load_mtp_head(&mtp_loader)
                .context("loading MTP draft head")?,
        )
    } else {
        None
    };

    let mut baseline_warmups = Vec::with_capacity(args.warmup_runs);
    let mut warmups = Vec::with_capacity(args.warmup_runs);
    for index in 0..args.warmup_runs {
        if args.scheduler_baseline_out.is_some() && index % 2 == 0 {
            baseline_warmups.push(run_once_qwen(
                model,
                None,
                tokenizer,
                &prompt_ids,
                args,
                scheduler_config,
                None,
            )?);
        }
        warmups.push(run_once_qwen(
            model,
            mtp.as_ref(),
            tokenizer,
            &prompt_ids,
            args,
            scheduler_config,
            mtp_draft_tokens,
        )?);
        if args.scheduler_baseline_out.is_some() && index % 2 == 1 {
            baseline_warmups.push(run_once_qwen(
                model,
                None,
                tokenizer,
                &prompt_ids,
                args,
                scheduler_config,
                None,
            )?);
        }
    }

    let mut baseline_records = Vec::with_capacity(args.runs);
    let mut records = Vec::with_capacity(args.runs);
    for index in 0..args.runs {
        if args.scheduler_baseline_out.is_some() && index % 2 == 0 {
            baseline_records.push(run_once_qwen(
                model,
                None,
                tokenizer,
                &prompt_ids,
                args,
                scheduler_config,
                None,
            )?);
        }
        records.push(run_once_qwen(
            model,
            mtp.as_ref(),
            tokenizer,
            &prompt_ids,
            args,
            scheduler_config,
            mtp_draft_tokens,
        )?);
        if args.scheduler_baseline_out.is_some() && index % 2 == 1 {
            baseline_records.push(run_once_qwen(
                model,
                None,
                tokenizer,
                &prompt_ids,
                args,
                scheduler_config,
                None,
            )?);
        }
    }

    let meta = Meta {
        backend: "ironmlx-core",
        mode: args.mode,
        speculative_source: mtp.as_ref().map(|_| "qwen-mtp"),
        model_dir: args.model.display().to_string(),
        mtp_model_dir: args
            .mtp_model_dir
            .as_ref()
            .map(|dir| dir.display().to_string()),
        mtp_draft_tokens,
        mtp_trace_windows: args.mtp_trace_windows,
        prompt_file: primary_prompt_file(args).display().to_string(),
        prompt_tokens: primary_prompt_ids.len(),
        scheduler_prompt_files: scheduler_prompt_files(args),
        scheduler_prompt_tokens: prompt_ids.iter().map(Vec::len).collect(),
        scheduler_batch_width: prompt_ids.len(),
        max_tokens: args.max_tokens,
        prefill_chunk_size: args.prefill_chunk_size,
        kv_quant: args.kv_quant,
        paged_prefix_cache_dir: scheduler_features
            .paged_prefix_cache
            .as_ref()
            .map(|config| config.root.display().to_string()),
        paged_prefix_cache_block_size: args.paged_prefix_cache_block_size,
        paged_prefix_cache_max_pages: scheduler_features
            .paged_prefix_cache
            .as_ref()
            .map(|config| config.max_pages),
        active_kv_offload: scheduler_features.active_kv_offload.enabled,
        active_kv_offload_dir: scheduler_features.active_kv_offload.enabled.then(|| {
            scheduler_features
                .active_kv_offload
                .root
                .display()
                .to_string()
        }),
        active_kv_hot_window_pages: scheduler_features
            .active_kv_offload
            .hot_window_pages_override,
        active_kv_chunk_pages: scheduler_features.active_kv_offload.chunk_pages_override,
        b_max: args.b_max,
        effective_cap_max,
        warmup_runs: args.warmup_runs,
        measured_runs: args.runs,
        load_ms,
        device_name: mlx::memory::snapshot().device_name,
        ironmlx_version: env!("CARGO_PKG_VERSION"),
    };
    if let Some(baseline_out) = args.scheduler_baseline_out.as_ref() {
        let mut baseline_meta = meta.clone();
        baseline_meta.speculative_source = None;
        baseline_meta.mtp_model_dir = None;
        baseline_meta.mtp_draft_tokens = None;
        let baseline_output = BenchOutput {
            meta: baseline_meta,
            summary: summarize(&baseline_records),
            warmups: baseline_warmups,
            records: baseline_records,
        };
        std::fs::write(
            baseline_out,
            serde_json::to_string_pretty(&baseline_output)? + "\n",
        )
        .with_context(|| format!("writing {}", baseline_out.display()))?;
    }

    let output = BenchOutput {
        meta,
        summary: summarize(&records),
        warmups,
        records,
    };

    std::fs::write(&args.out, serde_json::to_string_pretty(&output)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;
    Ok(())
}

fn run_for_gemma4_model(
    model: &Gemma4Model,
    tokenizer: &Tokenizer,
    args: &Args,
    raw_config: &serde_json::Value,
    load_ms: f64,
) -> Result<()> {
    let prompt_ids = read_scheduler_prompt_ids(tokenizer, args)?;
    let primary_prompt_ids = prompt_ids
        .first()
        .expect("validated prompt list must be non-empty");

    let effective_cap_max = args.effective_cap_max.unwrap_or_else(|| {
        prompt_ids
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .saturating_add(args.max_tokens)
            .max(MIN_KV_CACHE_CAP_FOR_GPU_PERF as usize)
    });
    let scheduler_features = resolve_scheduler_features(args, effective_cap_max)?;
    let scheduler_config = SchedulerBenchConfig {
        features: &scheduler_features,
        effective_cap_max,
    };

    let mtp_draft_tokens = args.mtp_model_dir.as_ref().map(|_| {
        resolve_mtp_draft_tokens(
            raw_config,
            args.mtp_draft_tokens
                .map(MtpDraftTokensArg::Explicit)
                .unwrap_or(MtpDraftTokensArg::Omitted),
        )
    });

    let drafter = if args.mtp_model_dir.is_some() {
        let mtp_dir = args
            .mtp_model_dir
            .as_ref()
            .ok_or_else(|| anyhow!("--mtp-model-dir is required for Gemma4 drafter benchmarks"))?;
        let drafter_loader =
            Loader::open_gemma4_drafter(mtp_dir).context("Loader::open_gemma4_drafter")?;
        Some(
            Gemma4AssistantModel::from_loader(&drafter_loader)
                .context("Gemma4AssistantModel::from_loader")?,
        )
    } else {
        None
    };

    let mut warmups = Vec::with_capacity(args.warmup_runs);
    for _ in 0..args.warmup_runs {
        warmups.push(run_once_gemma4(
            model,
            drafter.as_ref(),
            tokenizer,
            &prompt_ids,
            args,
            scheduler_config,
            mtp_draft_tokens,
        )?);
    }

    let mut records = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        records.push(run_once_gemma4(
            model,
            drafter.as_ref(),
            tokenizer,
            &prompt_ids,
            args,
            scheduler_config,
            mtp_draft_tokens,
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-core",
            mode: args.mode,
            speculative_source: drafter.as_ref().map(|_| "gemma4-drafter"),
            model_dir: args.model.display().to_string(),
            mtp_model_dir: args
                .mtp_model_dir
                .as_ref()
                .map(|dir| dir.display().to_string()),
            mtp_draft_tokens,
            mtp_trace_windows: args.mtp_trace_windows,
            prompt_file: primary_prompt_file(args).display().to_string(),
            prompt_tokens: primary_prompt_ids.len(),
            scheduler_prompt_files: scheduler_prompt_files(args),
            scheduler_prompt_tokens: prompt_ids.iter().map(Vec::len).collect(),
            scheduler_batch_width: prompt_ids.len(),
            max_tokens: args.max_tokens,
            prefill_chunk_size: args.prefill_chunk_size,
            kv_quant: args.kv_quant,
            paged_prefix_cache_dir: scheduler_features
                .paged_prefix_cache
                .as_ref()
                .map(|config| config.root.display().to_string()),
            paged_prefix_cache_block_size: args.paged_prefix_cache_block_size,
            paged_prefix_cache_max_pages: scheduler_features
                .paged_prefix_cache
                .as_ref()
                .map(|config| config.max_pages),
            active_kv_offload: scheduler_features.active_kv_offload.enabled,
            active_kv_offload_dir: scheduler_features.active_kv_offload.enabled.then(|| {
                scheduler_features
                    .active_kv_offload
                    .root
                    .display()
                    .to_string()
            }),
            active_kv_hot_window_pages: scheduler_features
                .active_kv_offload
                .hot_window_pages_override,
            active_kv_chunk_pages: scheduler_features.active_kv_offload.chunk_pages_override,
            b_max: args.b_max,
            effective_cap_max,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            load_ms,
            device_name: mlx::memory::snapshot().device_name,
            ironmlx_version: env!("CARGO_PKG_VERSION"),
        },
        summary: summarize(&records),
        warmups,
        records,
    };

    std::fs::write(&args.out, serde_json::to_string_pretty(&output)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;
    Ok(())
}

fn run_once<M>(
    model: &M,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    args: &Args,
    scheduler_config: SchedulerBenchConfig<'_>,
) -> Result<Record>
where
    M: Model + DenseVlMethods,
{
    match args.mode {
        BenchMode::Gs => run_generation_stream(model, tokenizer, prompt_ids, args),
        BenchMode::Mtp => Err(anyhow!(
            "mtp-text mode is only supported for Qwen dense/MoE text models"
        )),
        BenchMode::Scheduler => run_scheduler(model, tokenizer, prompt_ids, args, scheduler_config),
    }
}

fn run_once_qwen<M>(
    model: &M,
    mtp: Option<&M::MtpHead>,
    tokenizer: &Tokenizer,
    prompt_ids: &[Vec<u32>],
    args: &Args,
    scheduler_config: SchedulerBenchConfig<'_>,
    mtp_draft_tokens: Option<usize>,
) -> Result<Record>
where
    M: MtpSpeculativeModel + DenseVlMethods,
{
    let primary_prompt_ids = prompt_ids
        .first()
        .ok_or_else(|| anyhow!("Qwen benchmark requires at least one prompt"))?;
    match args.mode {
        BenchMode::Gs => {
            require_single_prompt(prompt_ids, args.mode)?;
            run_generation_stream(model, tokenizer, primary_prompt_ids, args)
        }
        BenchMode::Mtp => {
            require_single_prompt(prompt_ids, args.mode)?;
            let mtp = mtp.ok_or_else(|| anyhow!("mtp-text mode requires a loaded MTP head"))?;
            let mtp_draft_tokens =
                mtp_draft_tokens.ok_or_else(|| anyhow!("MTP run missing resolved draft tokens"))?;
            run_mtp_generation_stream(
                model,
                mtp,
                tokenizer,
                primary_prompt_ids,
                args,
                mtp_draft_tokens,
            )
        }
        BenchMode::Scheduler => {
            if let Some(mtp) = mtp {
                let mtp_draft_tokens = mtp_draft_tokens
                    .ok_or_else(|| anyhow!("scheduler MTP run missing resolved draft tokens"))?;
                run_scheduler_mtp(
                    model,
                    mtp,
                    tokenizer,
                    prompt_ids,
                    args,
                    scheduler_config,
                    mtp_draft_tokens,
                )
            } else {
                require_single_prompt(prompt_ids, args.mode)?;
                run_scheduler(model, tokenizer, primary_prompt_ids, args, scheduler_config)
            }
        }
    }
}

fn run_once_gemma4(
    model: &Gemma4Model,
    drafter: Option<&Gemma4AssistantModel>,
    tokenizer: &Tokenizer,
    prompt_ids: &[Vec<u32>],
    args: &Args,
    scheduler_config: SchedulerBenchConfig<'_>,
    mtp_draft_tokens: Option<usize>,
) -> Result<Record> {
    let primary_prompt_ids = prompt_ids
        .first()
        .ok_or_else(|| anyhow!("Gemma4 benchmark requires at least one prompt"))?;
    match args.mode {
        BenchMode::Gs => {
            require_single_prompt(prompt_ids, args.mode)?;
            run_generation_stream(model, tokenizer, primary_prompt_ids, args)
        }
        BenchMode::Mtp => {
            require_single_prompt(prompt_ids, args.mode)?;
            let drafter =
                drafter.ok_or_else(|| anyhow!("mtp-text mode requires a loaded Gemma4 drafter"))?;
            let mtp_draft_tokens = mtp_draft_tokens
                .ok_or_else(|| anyhow!("Gemma4 drafter run missing resolved draft tokens"))?;
            run_gemma4_drafter_generation_stream(
                model,
                drafter,
                tokenizer,
                primary_prompt_ids,
                args,
                mtp_draft_tokens,
            )
        }
        BenchMode::Scheduler => {
            if let Some(drafter) = drafter {
                let mtp_draft_tokens = mtp_draft_tokens.ok_or_else(|| {
                    anyhow!("Gemma4 scheduler drafter run missing resolved draft tokens")
                })?;
                return run_scheduler_gemma4_drafter(
                    model,
                    drafter,
                    tokenizer,
                    prompt_ids,
                    args,
                    scheduler_config,
                    mtp_draft_tokens,
                );
            }
            require_single_prompt(prompt_ids, args.mode)?;
            run_scheduler(model, tokenizer, primary_prompt_ids, args, scheduler_config)
        }
    }
}

fn require_single_prompt(prompt_ids: &[Vec<u32>], mode: BenchMode) -> Result<()> {
    if prompt_ids.len() != 1 {
        return Err(anyhow!(
            "{} prompts are not supported for {mode:?}; concurrent prompts require Gemma4 scheduler-text with --mtp-model-dir",
            prompt_ids.len()
        ));
    }
    Ok(())
}

fn run_generation_stream<M>(
    model: &M,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    args: &Args,
) -> Result<Record>
where
    M: Model,
{
    let request = make_request(model, tokenizer, prompt_ids, args);
    let started = Instant::now();
    let mut stream = GenerationStream::new_text_only(model, tokenizer, request)?;
    let mut first_ms = None;
    let mut generated_token_ids = Vec::with_capacity(args.max_tokens);
    let mut generated_text = String::new();
    let mut finish_reason = None;

    while let Some(event) = stream.next_token()? {
        if first_ms.is_none() {
            first_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
        }
        generated_token_ids.push(event.token);
        generated_text.push_str(&event.text);
        finish_reason = event.finish_reason;
        if finish_reason.is_some() {
            break;
        }
    }
    mlx::transforms::synchronize()?;
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = first_ms.ok_or_else(|| anyhow!("generation stream produced no tokens"))?;
    Ok(make_record(RecordInput {
        mode: args.mode,
        ttft_ms,
        e2e_ms,
        generated: GeneratedOutput {
            token_ids: generated_token_ids,
            text: generated_text,
        },
        finish_reason,
        max_tokens: args.max_tokens,
        mtp_stats: None,
        mtp_trace: None,
        active_kv_stats: None,
    }))
}

fn run_scheduler<M>(
    model: &M,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    args: &Args,
    scheduler_config: SchedulerBenchConfig<'_>,
) -> Result<Record>
where
    M: Model + DenseVlMethods,
{
    let mut scheduler = Scheduler::<M>::new(
        args.b_max,
        scheduler_config.effective_cap_max,
        model.model_meta(),
    )
    .context("Scheduler::new")?;
    let active_kv_stats = configure_scheduler_features(&mut scheduler, scheduler_config.features)?;
    let request = make_request(model, tokenizer, prompt_ids, args);
    let started = Instant::now();
    let _request_id = scheduler.admit(request)?;
    let first_events = scheduler.prefill_admitted(model)?;
    refresh_active_kv_stats(&scheduler, active_kv_stats.as_ref());
    let mut generated_token_ids: Vec<u32> = first_events.iter().map(|event| event.token).collect();
    let mut finish_reason = first_events.first().and_then(|event| event.finish_reason);
    let ttft_ms = started.elapsed().as_secs_f64() * 1000.0;

    while finish_reason.is_none() && generated_token_ids.len() < args.max_tokens {
        let events = scheduler.step(model)?;
        if events.is_empty() {
            break;
        }
        refresh_active_kv_stats(&scheduler, active_kv_stats.as_ref());
        generated_token_ids.extend(events.iter().map(|event| event.token));
        finish_reason = events.first().and_then(|event| event.finish_reason);
    }
    mlx::transforms::synchronize()?;
    refresh_active_kv_stats(&scheduler, active_kv_stats.as_ref());
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    let generated_text = tokenizer
        .decode(&generated_token_ids, true)
        .unwrap_or_default();
    Ok(make_record(RecordInput {
        mode: args.mode,
        ttft_ms,
        e2e_ms,
        generated: GeneratedOutput {
            token_ids: generated_token_ids,
            text: generated_text,
        },
        finish_reason,
        max_tokens: args.max_tokens,
        mtp_stats: None,
        mtp_trace: None,
        active_kv_stats: active_kv_stats
            .as_ref()
            .map(|stats| ActiveKvRecordStats::from(stats.snapshot())),
    }))
}

fn run_mtp_generation_stream<M>(
    model: &M,
    mtp: &M::MtpHead,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    args: &Args,
    mtp_draft_tokens: usize,
) -> Result<Record>
where
    M: MtpSpeculativeModel,
{
    let request = make_request(model, tokenizer, prompt_ids, args);
    let cfg = MtpSpeculativeConfig::new(mtp_draft_tokens, request.sampler)?;
    let started = Instant::now();
    let mut stream = MtpTextGenerationStream::new_text_only(model, mtp, tokenizer, request, cfg)?;
    let mut first_ms = None;
    let mut generated_token_ids = Vec::with_capacity(args.max_tokens);
    let mut generated_text = String::new();
    let mut finish_reason = None;

    while let Some(event) = stream.next_token()? {
        if first_ms.is_none() {
            first_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
        }
        generated_token_ids.push(event.token);
        generated_text.push_str(&event.text);
        finish_reason = event.finish_reason;
        if finish_reason.is_some() {
            break;
        }
    }
    mlx::transforms::synchronize()?;
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = first_ms.ok_or_else(|| anyhow!("MTP generation stream produced no tokens"))?;
    let mtp_stats = stream.stats().into();
    Ok(make_record(RecordInput {
        mode: args.mode,
        ttft_ms,
        e2e_ms,
        generated: GeneratedOutput {
            token_ids: generated_token_ids,
            text: generated_text,
        },
        finish_reason,
        max_tokens: args.max_tokens,
        mtp_stats: Some(mtp_stats),
        mtp_trace: None,
        active_kv_stats: None,
    }))
}

fn run_gemma4_drafter_generation_stream(
    model: &Gemma4Model,
    drafter: &Gemma4AssistantModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    args: &Args,
    mtp_draft_tokens: usize,
) -> Result<Record> {
    let request = make_request(model, tokenizer, prompt_ids, args);
    let cfg = MtpSpeculativeConfig::new(mtp_draft_tokens, request.sampler)?;
    let started = Instant::now();
    let mut stream = Gemma4DrafterGenerationStream::new(model, drafter, tokenizer, request, cfg)?;
    stream.set_trace_window_limit(args.mtp_trace_windows);
    let mut first_ms = None;
    let mut generated_token_ids = Vec::with_capacity(args.max_tokens);
    let mut generated_text = String::new();
    let mut finish_reason = None;

    while let Some(event) = stream.next_token()? {
        if first_ms.is_none() {
            first_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
        }
        generated_token_ids.push(event.token);
        generated_text.push_str(&event.text);
        finish_reason = event.finish_reason;
        if finish_reason.is_some() {
            break;
        }
    }
    mlx::transforms::synchronize()?;
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms =
        first_ms.ok_or_else(|| anyhow!("Gemma4 drafter generation stream produced no tokens"))?;
    let mtp_stats = stream.stats().into();
    let mtp_trace = (!stream.trace_windows().is_empty()).then(|| {
        stream
            .trace_windows()
            .iter()
            .cloned()
            .map(MtpTraceWindowRecord::from)
            .collect()
    });
    Ok(make_record(RecordInput {
        mode: args.mode,
        ttft_ms,
        e2e_ms,
        generated: GeneratedOutput {
            token_ids: generated_token_ids,
            text: generated_text,
        },
        finish_reason,
        max_tokens: args.max_tokens,
        mtp_stats: Some(mtp_stats),
        mtp_trace,
        active_kv_stats: None,
    }))
}

fn run_scheduler_mtp<M>(
    model: &M,
    mtp: &M::MtpHead,
    tokenizer: &Tokenizer,
    prompt_ids: &[Vec<u32>],
    args: &Args,
    scheduler_config: SchedulerBenchConfig<'_>,
    mtp_draft_tokens: usize,
) -> Result<Record>
where
    M: MtpSpeculativeModel + DenseVlMethods,
{
    if prompt_ids.is_empty() || prompt_ids.len() > args.b_max {
        return Err(anyhow!(
            "Qwen MTP scheduler batch width {} must be within 1..={}",
            prompt_ids.len(),
            args.b_max
        ));
    }

    let mut scheduler = Scheduler::<M>::new(
        args.b_max,
        scheduler_config.effective_cap_max,
        model.model_meta(),
    )
    .context("Scheduler::new")?;
    let active_kv_stats = configure_scheduler_features(&mut scheduler, scheduler_config.features)?;
    let cfg = MtpSpeculativeConfig::new(mtp_draft_tokens, Sampler::greedy())?;
    let started = Instant::now();
    let mut requests = Vec::with_capacity(prompt_ids.len());
    let mut request_rows = HashMap::with_capacity(prompt_ids.len());

    for (request_index, ids) in prompt_ids.iter().enumerate() {
        let request_id = scheduler.admit(make_request(model, tokenizer, ids, args))?;
        request_rows.insert(request_id, request_index);
        requests.push(SchedulerRequestState {
            request_index,
            request_id,
            prompt_file: args.prompt_file[request_index].display().to_string(),
            ttft_ms: None,
            e2e_ms: None,
            generated_token_ids: Vec::with_capacity(args.max_tokens),
            finish_reason: None,
        });
    }

    let first_events = scheduler.prefill_admitted_mtp_batch(model, mtp, cfg)?;
    refresh_active_kv_stats(&scheduler, active_kv_stats.as_ref());
    let batch_ttft_ms = started.elapsed().as_secs_f64() * 1000.0;
    record_scheduler_events(
        &mut requests,
        &request_rows,
        &first_events,
        batch_ttft_ms,
        true,
    )?;
    if requests.iter().any(|request| request.ttft_ms.is_none()) {
        return Err(anyhow!(
            "Qwen MTP scheduler prefill did not emit one first event per admitted request"
        ));
    }

    while requests
        .iter()
        .any(|request| request.finish_reason.is_none())
    {
        let events = scheduler.step_mtp_batch(model, mtp)?;
        if events.is_empty() {
            return Err(anyhow!(
                "Qwen MTP scheduler stopped before all benchmark requests finished"
            ));
        }
        refresh_active_kv_stats(&scheduler, active_kv_stats.as_ref());
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        record_scheduler_events(&mut requests, &request_rows, &events, elapsed_ms, false)?;
    }
    mlx::transforms::synchronize()?;
    refresh_active_kv_stats(&scheduler, active_kv_stats.as_ref());
    let batch_e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mtp_stats = scheduler
        .mtp_stats()
        .ok_or_else(|| anyhow!("scheduler MTP run produced no MTP stats"))?
        .into();

    let scheduler_requests = requests
        .into_iter()
        .map(|request| {
            let finish_reason = request.finish_reason;
            let generated_tokens = request.generated_token_ids.len();
            let generated_text = tokenizer
                .decode(&request.generated_token_ids, true)
                .unwrap_or_default();
            SchedulerRequestRecord {
                request_index: request.request_index,
                prompt_file: request.prompt_file,
                ttft_ms: request.ttft_ms.unwrap_or(batch_ttft_ms),
                e2e_ms: request.e2e_ms.unwrap_or(batch_e2e_ms),
                generated_tokens,
                generated_token_ids: request.generated_token_ids,
                generated_text,
                finish_reason,
                valid: finish_reason == Some("length") && generated_tokens >= args.max_tokens,
            }
        })
        .collect::<Vec<_>>();
    let representative = scheduler_requests
        .first()
        .ok_or_else(|| anyhow!("Qwen MTP scheduler batch produced no request records"))?;
    let mut record = make_record(RecordInput {
        mode: args.mode,
        ttft_ms: batch_ttft_ms,
        e2e_ms: batch_e2e_ms,
        generated: GeneratedOutput {
            token_ids: representative.generated_token_ids.clone(),
            text: representative.generated_text.clone(),
        },
        finish_reason: representative.finish_reason,
        max_tokens: args.max_tokens,
        mtp_stats: Some(mtp_stats),
        mtp_trace: None,
        active_kv_stats: active_kv_stats
            .as_ref()
            .map(|stats| ActiveKvRecordStats::from(stats.snapshot())),
    });
    let batch_decode_time_ms = (batch_e2e_ms - batch_ttft_ms).max(0.0);
    let decoded_tokens = scheduler_requests
        .iter()
        .map(|request| request.generated_tokens.saturating_sub(1))
        .sum::<usize>();
    record.aggregate_generation_tps = if batch_decode_time_ms > 0.0 {
        decoded_tokens as f64 / (batch_decode_time_ms / 1000.0)
    } else {
        0.0
    };
    record.valid = scheduler_requests.iter().all(|request| request.valid);
    record.scheduler_requests = scheduler_requests;
    Ok(record)
}

fn run_scheduler_gemma4_drafter(
    model: &Gemma4Model,
    drafter: &Gemma4AssistantModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[Vec<u32>],
    args: &Args,
    scheduler_config: SchedulerBenchConfig<'_>,
    mtp_draft_tokens: usize,
) -> Result<Record> {
    if prompt_ids.is_empty() || prompt_ids.len() > args.b_max {
        return Err(anyhow!(
            "Gemma4 scheduler batch width {} must be within 1..={}",
            prompt_ids.len(),
            args.b_max
        ));
    }

    let mut scheduler = Scheduler::<Gemma4Model>::new(
        args.b_max,
        scheduler_config.effective_cap_max,
        model.model_meta(),
    )
    .context("Scheduler::new")?;
    let active_kv_stats = configure_scheduler_features(&mut scheduler, scheduler_config.features)?;
    let cfg = MtpSpeculativeConfig::new(mtp_draft_tokens, Sampler::greedy())?;
    let started = Instant::now();
    let mut requests = Vec::with_capacity(prompt_ids.len());
    let mut request_rows = HashMap::with_capacity(prompt_ids.len());

    for (request_index, ids) in prompt_ids.iter().enumerate() {
        let request_id = scheduler.admit(make_request(model, tokenizer, ids, args))?;
        request_rows.insert(request_id, request_index);
        requests.push(SchedulerRequestState {
            request_index,
            request_id,
            prompt_file: args.prompt_file[request_index].display().to_string(),
            ttft_ms: None,
            e2e_ms: None,
            generated_token_ids: Vec::with_capacity(args.max_tokens),
            finish_reason: None,
        });
    }

    let first_events = scheduler.prefill_admitted_gemma4_drafter_batch(model, drafter, cfg)?;
    refresh_active_kv_stats(&scheduler, active_kv_stats.as_ref());
    let batch_ttft_ms = started.elapsed().as_secs_f64() * 1000.0;
    record_scheduler_events(
        &mut requests,
        &request_rows,
        &first_events,
        batch_ttft_ms,
        true,
    )?;
    if requests.iter().any(|request| request.ttft_ms.is_none()) {
        return Err(anyhow!(
            "Gemma4 scheduler prefill did not emit one first event per admitted request"
        ));
    }

    while requests
        .iter()
        .any(|request| request.finish_reason.is_none())
    {
        let events = scheduler.step_gemma4_drafter_batch(model, drafter)?;
        if events.is_empty() {
            return Err(anyhow!(
                "Gemma4 scheduler drafter stopped before all benchmark requests finished"
            ));
        }
        refresh_active_kv_stats(&scheduler, active_kv_stats.as_ref());
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        record_scheduler_events(&mut requests, &request_rows, &events, elapsed_ms, false)?;
    }

    mlx::transforms::synchronize()?;
    refresh_active_kv_stats(&scheduler, active_kv_stats.as_ref());
    let batch_e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mtp_stats = scheduler
        .gemma4_drafter_stats()
        .ok_or_else(|| anyhow!("Gemma4 scheduler drafter run produced no MTP stats"))?
        .into();

    let scheduler_requests = requests
        .into_iter()
        .map(|request| {
            let finish_reason = request.finish_reason;
            let generated_tokens = request.generated_token_ids.len();
            let generated_text = tokenizer
                .decode(&request.generated_token_ids, true)
                .unwrap_or_default();
            SchedulerRequestRecord {
                request_index: request.request_index,
                prompt_file: request.prompt_file,
                ttft_ms: request.ttft_ms.unwrap_or(batch_ttft_ms),
                e2e_ms: request.e2e_ms.unwrap_or(batch_e2e_ms),
                generated_tokens,
                generated_token_ids: request.generated_token_ids,
                generated_text,
                finish_reason,
                valid: finish_reason == Some("length") && generated_tokens >= args.max_tokens,
            }
        })
        .collect::<Vec<_>>();
    let representative = scheduler_requests
        .first()
        .ok_or_else(|| anyhow!("Gemma4 scheduler batch produced no request records"))?;
    let mut record = make_record(RecordInput {
        mode: args.mode,
        ttft_ms: batch_ttft_ms,
        e2e_ms: batch_e2e_ms,
        generated: GeneratedOutput {
            token_ids: representative.generated_token_ids.clone(),
            text: representative.generated_text.clone(),
        },
        finish_reason: representative.finish_reason,
        max_tokens: args.max_tokens,
        mtp_stats: Some(mtp_stats),
        mtp_trace: None,
        active_kv_stats: active_kv_stats
            .as_ref()
            .map(|stats| ActiveKvRecordStats::from(stats.snapshot())),
    });
    let batch_decode_time_ms = (batch_e2e_ms - batch_ttft_ms).max(0.0);
    let decoded_tokens = scheduler_requests
        .iter()
        .map(|request| request.generated_tokens.saturating_sub(1))
        .sum::<usize>();
    record.aggregate_generation_tps = if batch_decode_time_ms > 0.0 {
        decoded_tokens as f64 / (batch_decode_time_ms / 1000.0)
    } else {
        0.0
    };
    record.valid = scheduler_requests.iter().all(|request| request.valid);
    record.scheduler_requests = scheduler_requests;
    Ok(record)
}

fn record_scheduler_events(
    requests: &mut [SchedulerRequestState],
    request_rows: &HashMap<ironmlx::core::RequestId, usize>,
    events: &[ironmlx::core::StepEvent],
    elapsed_ms: f64,
    first_events: bool,
) -> Result<()> {
    let mut seen = HashSet::with_capacity(events.len());
    for event in events {
        if !seen.insert(event.id) {
            return Err(anyhow!(
                "scheduler emitted duplicate event for request {} in one step",
                event.id.0
            ));
        }
        let request_index = request_rows.get(&event.id).copied().ok_or_else(|| {
            anyhow!(
                "scheduler emitted unknown benchmark request id {}",
                event.id.0
            )
        })?;
        let request = requests
            .get_mut(request_index)
            .ok_or_else(|| anyhow!("scheduler request index {request_index} is out of range"))?;
        if request.request_id != event.id {
            return Err(anyhow!(
                "scheduler request id mismatch at index {request_index}: expected {}, got {}",
                request.request_id.0,
                event.id.0
            ));
        }
        if request.finish_reason.is_some() {
            return Err(anyhow!(
                "scheduler emitted an event after request {} finished",
                event.id.0
            ));
        }
        if first_events {
            if request.ttft_ms.replace(elapsed_ms).is_some() {
                return Err(anyhow!(
                    "scheduler emitted duplicate first event for request {}",
                    event.id.0
                ));
            }
        } else if request.ttft_ms.is_none() {
            return Err(anyhow!(
                "scheduler emitted decode event before first event for request {}",
                event.id.0
            ));
        }
        request.generated_token_ids.push(event.token);
        request.finish_reason = event.finish_reason;
        if event.finish_reason.is_some() {
            request.e2e_ms = Some(elapsed_ms);
        }
    }
    Ok(())
}

fn configure_scheduler_features<M: Model>(
    scheduler: &mut Scheduler<M>,
    features: &BenchSchedulerFeatures,
) -> Result<Option<ActiveKvOffloadSharedStats>> {
    if let Some(config) = features.paged_prefix_cache.as_ref() {
        scheduler
            .enable_paged_prefix_cache(config.clone())
            .context("enabling benchmark paged prefix cache")?;
    }
    if !features.active_kv_offload.enabled {
        return Ok(None);
    }
    let stats = ActiveKvOffloadSharedStats::new(&features.active_kv_offload);
    scheduler
        .enable_active_kv_offload(features.active_kv_offload.clone(), stats.clone())
        .context("enabling benchmark active KV offload")?;
    Ok(Some(stats))
}

fn refresh_active_kv_stats<M: Model>(
    scheduler: &Scheduler<M>,
    stats: Option<&ActiveKvOffloadSharedStats>,
) {
    if stats.is_some() {
        scheduler.refresh_active_kv_residency_stats();
    }
}

fn make_request<M: Model>(
    model: &M,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    args: &Args,
) -> GenerateRequest {
    GenerateRequest {
        prompt_ids: prompt_ids.to_vec(),
        max_new_tokens: args.max_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: args.prefill_chunk_size,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: args.kv_quant.turboquant_bits(),
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: model.model_meta().spatial_merge_size,
        image_token_id: tokenizer
            .token_to_id("<|image_pad|>")
            .map(|id| id as i32)
            .unwrap_or(248_056),
        constraint: None,
    }
}

fn make_record(input: RecordInput) -> Record {
    let RecordInput {
        mode,
        ttft_ms,
        e2e_ms,
        generated,
        finish_reason,
        max_tokens,
        mtp_stats,
        mtp_trace,
        active_kv_stats,
    } = input;
    let decode_time_ms = (e2e_ms - ttft_ms).max(0.0);
    let generated_tokens = generated.token_ids.len();
    let generation_tps = if decode_time_ms > 0.0 {
        generated_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
    } else {
        0.0
    };
    Record {
        mode,
        ttft_ms,
        e2e_ms,
        decode_time_ms,
        generated_tokens,
        generated_token_ids: generated.token_ids,
        generated_text: generated.text,
        generation_tps,
        aggregate_generation_tps: generation_tps,
        finish_reason,
        valid: finish_reason == Some("length") && generated_tokens >= max_tokens,
        mtp_stats,
        mtp_trace,
        active_kv_stats,
        scheduler_requests: Vec::new(),
    }
}

fn summarize(records: &[Record]) -> Summary {
    let valid: Vec<&Record> = records.iter().filter(|record| record.valid).collect();
    Summary {
        runs: records.len(),
        valid_runs: valid.len(),
        ttft_ms: stats(valid.iter().map(|record| record.ttft_ms).collect()),
        e2e_ms: stats(valid.iter().map(|record| record.e2e_ms).collect()),
        decode_time_ms: stats(valid.iter().map(|record| record.decode_time_ms).collect()),
        generation_tps: stats(valid.iter().map(|record| record.generation_tps).collect()),
        aggregate_generation_tps: stats(
            valid
                .iter()
                .map(|record| record.aggregate_generation_tps)
                .collect(),
        ),
    }
}

fn stats(mut values: Vec<f64>) -> Stats {
    if values.is_empty() {
        return Stats {
            p50: None,
            p95: None,
            mean: None,
        };
    }
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Stats {
        p50: Some(percentile_sorted(&values, 50.0)),
        p95: Some(percentile_sorted(&values, 95.0)),
        mean: Some(mean),
    }
}

fn percentile_sorted(values: &[f64], pct: f64) -> f64 {
    if values.len() == 1 {
        return values[0];
    }
    let rank = (pct / 100.0) * (values.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(values.len() - 1);
    let weight = rank - lo as f64;
    values[lo] * (1.0 - weight) + values[hi] * weight
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse_args(argv: &[&str]) -> Args {
        Args::parse_from(argv)
    }

    fn temp_mtp_dir(test_name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ironmlx-core-bench-{test_name}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp mtp dir");
        dir
    }

    #[test]
    fn active_kv_record_stats_preserve_health_status_flags() {
        let stats = ActiveKvOffloadSharedStats::new(&ActiveKvOffloadConfig::enabled(
            std::env::temp_dir().join("ironmlx-core-bench-active-kv-status"),
        ));
        stats.record_error();

        let record = ActiveKvRecordStats::from(stats.snapshot());

        assert!(record.enabled);
        assert_eq!(record.status, ActiveKvOffloadStatus::Degraded);
        assert!(record.degraded);
    }

    #[test]
    fn clap_lists_mtp_text_mode() {
        let mut command = Args::command();
        let help = command.render_long_help().to_string();
        assert!(help.contains("mtp-text"));
    }

    #[test]
    fn mtp_draft_tokens_default_policy_is_omitted() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "gs-text",
            "--out",
            "/tmp/out.json",
        ]);
        assert_eq!(args.mtp_draft_tokens, None);
    }

    #[test]
    fn mtp_draft_tokens_parse_explicit_value() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "gs-text",
            "--mtp-draft-tokens",
            "6",
            "--out",
            "/tmp/out.json",
        ]);
        assert_eq!(args.mtp_draft_tokens, Some(6));
    }

    #[test]
    fn mtp_trace_windows_parse_explicit_value() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "mtp-text",
            "--mtp-model-dir",
            "/tmp/mtp",
            "--mtp-trace-windows",
            "4",
            "--out",
            "/tmp/out.json",
        ]);
        assert_eq!(args.mtp_trace_windows, 4);
    }

    #[test]
    fn kv_quant_parses_k3v4_for_core_bench() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--kv-quant",
            "k3v4",
            "--out",
            "/tmp/out.json",
        ]);
        assert_eq!(
            args.kv_quant.turboquant_bits(),
            Some(TurboQuantKVBits::K3V4)
        );
    }

    #[test]
    fn mtp_record_stats_preserves_position_buckets() {
        let mut stats = MtpSpeculativeStats::default();
        stats.record_window_acceptance(3, 1);
        stats.record_window_acceptance(2, 2);

        let record = MtpRecordStats::from(stats);

        assert_eq!(record.draft_attempts_by_position, vec![2, 2, 1]);
        assert_eq!(record.draft_accepts_by_position, vec![2, 1, 0]);
    }

    #[test]
    fn clap_lists_active_kv_and_paged_prefix_options() {
        let mut command = Args::command();
        let help = command.render_long_help().to_string();
        assert!(help.contains("--active-kv-offload"));
        assert!(help.contains("--active-kv-hot-window-pages"));
        assert!(help.contains("--active-kv-chunk-pages"));
        assert!(help.contains("--paged-prefix-cache-dir"));
    }

    #[test]
    fn active_kv_offload_parses_with_paged_prefix_cache() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--paged-prefix-cache-dir",
            "/tmp/prefix-cache",
            "--active-kv-offload",
            "--active-kv-offload-dir",
            "/tmp/active-kv",
            "--active-kv-hot-window-pages",
            "4",
            "--active-kv-chunk-pages",
            "8",
            "--out",
            "/tmp/out.json",
        ]);
        assert_eq!(
            args.paged_prefix_cache_dir,
            Some(PathBuf::from("/tmp/prefix-cache"))
        );
        assert!(args.active_kv_offload);
        assert_eq!(
            args.active_kv_offload_dir,
            Some(PathBuf::from("/tmp/active-kv"))
        );
        assert_eq!(args.active_kv_hot_window_pages, Some(4));
        assert_eq!(args.active_kv_chunk_pages, Some(8));
    }

    #[test]
    fn scheduler_b_max_defaults_to_two_for_admission_headroom() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--out",
            "/tmp/out.json",
        ]);
        assert_eq!(args.b_max, 2);
        assert_eq!(args.paged_prefix_cache_block_size, 128);
    }

    #[test]
    fn scheduler_baseline_out_requires_scheduler_mtp() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--scheduler-baseline-out",
            "/tmp/baseline.json",
            "--out",
            "/tmp/out.json",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(err.to_string().contains("--mtp-model-dir"));
    }

    #[test]
    fn scheduler_baseline_out_accepts_single_prompt_scheduler_mtp() {
        let mtp_dir = temp_mtp_dir("scheduler-baseline-out");
        let mtp_dir_arg = mtp_dir.to_string_lossy().into_owned();
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--mtp-model-dir",
            &mtp_dir_arg,
            "--scheduler-baseline-out",
            "/tmp/baseline.json",
            "--out",
            "/tmp/out.json",
        ]);

        validate_args(&args).unwrap();
        assert_eq!(
            args.scheduler_baseline_out,
            Some(PathBuf::from("/tmp/baseline.json"))
        );
        std::fs::remove_dir_all(mtp_dir).expect("remove temp mtp dir");
    }

    #[test]
    fn active_kv_offload_requires_scheduler_text_mode() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "gs-text",
            "--paged-prefix-cache-dir",
            "/tmp/prefix-cache",
            "--active-kv-offload",
            "--out",
            "/tmp/out.json",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("scheduler-text"));
    }

    #[test]
    fn active_kv_offload_requires_paged_prefix_cache() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--active-kv-offload",
            "--out",
            "/tmp/out.json",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("--paged-prefix-cache-dir"));
    }

    #[test]
    fn active_kv_hot_window_pages_requires_active_kv_offload() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--paged-prefix-cache-dir",
            "/tmp/prefix-cache",
            "--active-kv-hot-window-pages",
            "4",
            "--out",
            "/tmp/out.json",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("--active-kv-offload"));
    }

    #[test]
    fn active_kv_hot_window_pages_must_be_positive() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--paged-prefix-cache-dir",
            "/tmp/prefix-cache",
            "--active-kv-offload",
            "--active-kv-hot-window-pages",
            "0",
            "--out",
            "/tmp/out.json",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }

    #[test]
    fn active_kv_chunk_pages_requires_active_kv_offload() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--paged-prefix-cache-dir",
            "/tmp/prefix-cache",
            "--active-kv-chunk-pages",
            "8",
            "--out",
            "/tmp/out.json",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("--active-kv-offload"));
    }

    #[test]
    fn active_kv_chunk_pages_must_be_positive() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--paged-prefix-cache-dir",
            "/tmp/prefix-cache",
            "--active-kv-offload",
            "--active-kv-chunk-pages",
            "0",
            "--out",
            "/tmp/out.json",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }

    #[test]
    fn active_kv_offload_rejects_turboquant_because_hot_cold_uses_paged_kv() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--paged-prefix-cache-dir",
            "/tmp/prefix-cache",
            "--active-kv-offload",
            "--kv-quant",
            "k3v4",
            "--out",
            "/tmp/out.json",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("--kv-quant none"));
    }

    #[test]
    fn mtp_text_requires_mtp_model_dir() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "mtp-text",
            "--out",
            "/tmp/out.json",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("--mtp-model-dir"));
    }

    #[test]
    fn mtp_model_dir_is_rejected_outside_mtp_text_mode() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "gs-text",
            "--mtp-model-dir",
            "/tmp/mtp",
            "--out",
            "/tmp/out.json",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("mtp-text"));
    }

    #[test]
    fn scheduler_text_allows_mtp_model_dir_for_single_request_window() {
        let mtp_dir = temp_mtp_dir("scheduler-single");
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--b-max",
            "1",
            "--mtp-model-dir",
            mtp_dir.to_str().expect("utf8 temp path"),
            "--out",
            "/tmp/out.json",
        ]);
        let result = validate_args(&args);
        std::fs::remove_dir_all(mtp_dir).ok();
        assert!(result.is_ok(), "unexpected validate error: {result:?}");
    }

    #[test]
    fn scheduler_text_allows_mtp_model_dir_for_batched_scheduler() {
        let mtp_dir = temp_mtp_dir("scheduler-batch");
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--mode",
            "scheduler-text",
            "--b-max",
            "2",
            "--mtp-model-dir",
            mtp_dir.to_str().expect("utf8 temp path"),
            "--out",
            "/tmp/out.json",
        ]);
        let result = validate_args(&args);
        std::fs::remove_dir_all(mtp_dir).ok();
        assert!(result.is_ok(), "unexpected validate error: {result:?}");
    }

    #[test]
    fn scheduler_text_accepts_repeated_prompt_files_within_b_max() {
        let mtp_dir = temp_mtp_dir("scheduler-concurrent-prompts");
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/short.txt",
            "--prompt-file",
            "/tmp/long.txt",
            "--mode",
            "scheduler-text",
            "--b-max",
            "2",
            "--mtp-model-dir",
            mtp_dir.to_str().expect("utf8 temp path"),
            "--out",
            "/tmp/out.json",
        ]);

        let result = validate_args(&args);
        std::fs::remove_dir_all(mtp_dir).ok();

        assert!(result.is_ok(), "unexpected validate error: {result:?}");
        assert_eq!(
            args.prompt_file,
            vec![
                PathBuf::from("/tmp/short.txt"),
                PathBuf::from("/tmp/long.txt")
            ]
        );
    }

    #[test]
    fn scheduler_text_rejects_prompt_count_above_b_max() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/short.txt",
            "--prompt-file",
            "/tmp/long.txt",
            "--mode",
            "scheduler-text",
            "--b-max",
            "1",
            "--out",
            "/tmp/out.json",
        ]);

        let error = validate_args(&args).expect_err("prompt count must fit scheduler capacity");
        assert!(error.to_string().contains("exceeds --b-max"));
    }

    #[test]
    fn non_scheduler_mode_rejects_repeated_prompt_files() {
        let args = parse_args(&[
            "ironmlx-core-bench",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/short.txt",
            "--prompt-file",
            "/tmp/long.txt",
            "--mode",
            "gs-text",
            "--b-max",
            "2",
            "--out",
            "/tmp/out.json",
        ]);

        let error = validate_args(&args).expect_err("only scheduler mode accepts prompt batches");
        assert!(error.to_string().contains("require --mode scheduler-text"));
    }

    #[test]
    fn scheduler_events_are_routed_by_request_id() {
        let mut requests = vec![
            SchedulerRequestState {
                request_index: 0,
                request_id: ironmlx::core::RequestId(10),
                prompt_file: "/tmp/short.txt".to_string(),
                ttft_ms: None,
                e2e_ms: None,
                generated_token_ids: Vec::new(),
                finish_reason: None,
            },
            SchedulerRequestState {
                request_index: 1,
                request_id: ironmlx::core::RequestId(11),
                prompt_file: "/tmp/long.txt".to_string(),
                ttft_ms: None,
                e2e_ms: None,
                generated_token_ids: Vec::new(),
                finish_reason: None,
            },
        ];
        let rows = HashMap::from([
            (ironmlx::core::RequestId(10), 0),
            (ironmlx::core::RequestId(11), 1),
        ]);
        let events = [
            ironmlx::core::StepEvent {
                id: ironmlx::core::RequestId(11),
                token: 201,
                finish_reason: None,
            },
            ironmlx::core::StepEvent {
                id: ironmlx::core::RequestId(10),
                token: 101,
                finish_reason: None,
            },
        ];

        record_scheduler_events(&mut requests, &rows, &events, 5.0, true)
            .expect("route first events");

        assert_eq!(requests[0].generated_token_ids, vec![101]);
        assert_eq!(requests[1].generated_token_ids, vec![201]);
        assert_eq!(requests[0].ttft_ms, Some(5.0));
        assert_eq!(requests[1].ttft_ms, Some(5.0));
    }

    #[test]
    fn scheduler_text_mtp_keeps_scheduler_mode_stats_contract() {
        let record = make_record(RecordInput {
            mode: BenchMode::Scheduler,
            ttft_ms: 1.0,
            e2e_ms: 3.0,
            generated: GeneratedOutput {
                token_ids: vec![101, 102, 103, 104],
                text: "bench text".to_string(),
            },
            finish_reason: Some("length"),
            max_tokens: 4,
            mtp_stats: Some(MtpRecordStats {
                windows: 1,
                drafted_tokens: 2,
                accepted_draft_tokens: 1,
                draft_attempts_by_position: vec![1, 1],
                draft_accepts_by_position: vec![1, 0],
                rollback_count: 1,
                mtp_cache_reuse_count: 0,
                mtp_cache_reused_tokens: 0,
                draft_budget_reductions: 0,
                draft_budget_increases: 0,
                acceptance_rate: Some(0.5),
                draft_forward_us: 0,
                verify_forward_us: 0,
                projection_us: 0,
                sampling_us: 0,
                draft_host_sync_count: 0,
                draft_host_sync_us: 0,
                verify_accept_host_sync_count: 1,
                verify_accept_host_sync_us: 0,
                main_rollback_us: 0,
                mtp_cache_commit_us: 0,
                mtp_prefill_cache_commit_us: 0,
                mtp_decode_cache_commit_us: 0,
                mtp_cache_restore_us: 0,
                draft_cap_observations: Vec::new(),
                draft_cap_observation_dropped_windows: 0,
            }),
            mtp_trace: None,
            active_kv_stats: None,
        });

        assert_eq!(record.mode, BenchMode::Scheduler);
        assert!(record.valid);
        assert_eq!(record.generated_token_ids, vec![101, 102, 103, 104]);
        assert_eq!(record.generated_text, "bench text");
        let serialized = serde_json::to_value(&record).expect("serialize benchmark record");
        assert_eq!(serialized["scheduler_requests"], serde_json::json!([]));
        let stats = record.mtp_stats.expect("scheduler MTP stats");
        assert_eq!(stats.windows, 1);
        assert_eq!(stats.rollback_count, 1);
        assert_eq!(stats.acceptance_rate, Some(0.5));
    }
}
