//! Minimal in-process benchmark for ironmlx model-core text generation paths.
//!
//! This binary intentionally bypasses HTTP so scheduler/model gaps can be
//! measured without server parsing, SSE, or client timing noise.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::cache::TurboQuantKVBits;
use ironmlx::core::scheduler::DenseVlMethods;
use ironmlx::core::speculative::{
    resolve_mtp_draft_tokens, MtpDraftTokensArg, MtpSpeculativeConfig, MtpSpeculativeModel,
    MtpSpeculativeStats, MtpTextGenerationStream,
};
use ironmlx::core::{GenerateRequest, GenerationStream, Loader, Model, Sampler, Scheduler};
use ironmlx::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF;
use ironmlx::models::{
    Glm4MoeLiteModel, LlamaModel, ModelArchitecture, Qwen35Model, Qwen35MoeModel, Qwen36MoeModel,
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

    /// File containing the already-rendered raw model prompt.
    #[arg(long)]
    prompt_file: PathBuf,

    /// Core path to measure.
    #[arg(long, value_enum)]
    mode: BenchMode,

    /// MTP model directory, required by mtp-text mode and optional for
    /// scheduler-text single-request windows.
    #[arg(long = "mtp-model-dir")]
    mtp_model_dir: Option<PathBuf>,

    /// Maximum MTP draft tokens per speculative window. If omitted, ironmlx
    /// picks a model-aware default from local benchmark policy.
    #[arg(long)]
    mtp_draft_tokens: Option<usize>,

    /// Number of generated tokens per request.
    #[arg(long, default_value_t = 16)]
    max_tokens: usize,

    /// Timed runs.
    #[arg(long, default_value_t = 7)]
    runs: usize,

    /// Warmup runs, excluded from summary.
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,

    /// Prefill chunk size passed to GenerateRequest.
    #[arg(long, default_value_t = 2048)]
    prefill_chunk_size: usize,

    /// KV cache quantization used by attention reads.
    #[arg(long = "kv-quant", value_enum, default_value_t = KvQuantBenchArg::None)]
    kv_quant: KvQuantBenchArg,

    /// Scheduler batch capacity, only used by scheduler-text mode.
    #[arg(long, default_value_t = 1)]
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

#[derive(Serialize)]
struct Meta {
    backend: &'static str,
    mode: BenchMode,
    model_dir: String,
    mtp_model_dir: Option<String>,
    mtp_draft_tokens: Option<usize>,
    prompt_file: String,
    prompt_tokens: usize,
    max_tokens: usize,
    prefill_chunk_size: usize,
    kv_quant: KvQuantBenchArg,
    b_max: usize,
    effective_cap_max: usize,
    warmup_runs: usize,
    measured_runs: usize,
    load_ms: f64,
}

#[derive(Serialize)]
struct Summary {
    runs: usize,
    valid_runs: usize,
    ttft_ms: Stats,
    e2e_ms: Stats,
    decode_time_ms: Stats,
    generation_tps: Stats,
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
    finish_reason: Option<&'static str>,
    valid: bool,
    mtp_stats: Option<MtpRecordStats>,
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
    main_rollback_us: u64,
    mtp_cache_commit_us: u64,
    mtp_cache_restore_us: u64,
}

impl From<MtpSpeculativeStats> for MtpRecordStats {
    fn from(stats: MtpSpeculativeStats) -> Self {
        Self {
            windows: stats.windows,
            drafted_tokens: stats.drafted_tokens,
            accepted_draft_tokens: stats.accepted_draft_tokens,
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
            main_rollback_us: stats.main_rollback_us,
            mtp_cache_commit_us: stats.mtp_cache_commit_us,
            mtp_cache_restore_us: stats.mtp_cache_restore_us,
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
        ModelArchitecture::Gemma4 => Err(anyhow!(
            "unsupported model_type: gemma4 (expected 'qwen3_5', 'qwen3_5_moe', or 'glm4_moe_lite')"
        )),
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
    }
}

fn validate_args(args: &Args) -> Result<()> {
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
                if args.b_max != 1 {
                    return Err(anyhow!(
                        "--mode scheduler-text with --mtp-model-dir currently requires --b-max 1"
                    ));
                }
                validate_mtp_dir(mtp_dir)?;
                if let Some(draft_tokens) = args.mtp_draft_tokens {
                    MtpSpeculativeConfig::new(draft_tokens, Sampler::greedy())?;
                }
            }
        }
        BenchMode::Gs => {
            if args.mtp_model_dir.is_some() {
                return Err(anyhow!(
                    "--mtp-model-dir is only supported with mtp-text or scheduler-text --b-max 1"
                ));
            }
        }
    }
    Ok(())
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

fn run_for_model<M>(model: &M, tokenizer: &Tokenizer, args: &Args, load_ms: f64) -> Result<()>
where
    M: Model + DenseVlMethods,
{
    if args.mtp_model_dir.is_some() {
        return Err(anyhow!(
            "--mtp-model-dir is only supported for Qwen dense/MoE text models"
        ));
    }
    let rendered_prompt = std::fs::read_to_string(&args.prompt_file)
        .with_context(|| format!("reading {}", args.prompt_file.display()))?;
    let prompt_ids = tokenizer.encode(&rendered_prompt, false)?;
    if prompt_ids.is_empty() {
        return Err(anyhow!("prompt_file encoded to zero tokens"));
    }

    let effective_cap_max = args.effective_cap_max.unwrap_or_else(|| {
        prompt_ids
            .len()
            .saturating_add(args.max_tokens)
            .max(MIN_KV_CACHE_CAP_FOR_GPU_PERF as usize)
    });

    let mut warmups = Vec::with_capacity(args.warmup_runs);
    for _ in 0..args.warmup_runs {
        warmups.push(run_once(
            model,
            tokenizer,
            &prompt_ids,
            args,
            effective_cap_max,
        )?);
    }

    let mut records = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        records.push(run_once(
            model,
            tokenizer,
            &prompt_ids,
            args,
            effective_cap_max,
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-core",
            mode: args.mode,
            model_dir: args.model.display().to_string(),
            mtp_model_dir: None,
            mtp_draft_tokens: None,
            prompt_file: args.prompt_file.display().to_string(),
            prompt_tokens: prompt_ids.len(),
            max_tokens: args.max_tokens,
            prefill_chunk_size: args.prefill_chunk_size,
            kv_quant: args.kv_quant,
            b_max: args.b_max,
            effective_cap_max,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            load_ms,
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
    let rendered_prompt = std::fs::read_to_string(&args.prompt_file)
        .with_context(|| format!("reading {}", args.prompt_file.display()))?;
    let prompt_ids = tokenizer.encode(&rendered_prompt, false)?;
    if prompt_ids.is_empty() {
        return Err(anyhow!("prompt_file encoded to zero tokens"));
    }

    let effective_cap_max = args.effective_cap_max.unwrap_or_else(|| {
        prompt_ids
            .len()
            .saturating_add(args.max_tokens)
            .max(MIN_KV_CACHE_CAP_FOR_GPU_PERF as usize)
    });

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

    let mut warmups = Vec::with_capacity(args.warmup_runs);
    for _ in 0..args.warmup_runs {
        warmups.push(run_once_qwen(
            model,
            mtp.as_ref(),
            tokenizer,
            &prompt_ids,
            args,
            mtp_draft_tokens,
            effective_cap_max,
        )?);
    }

    let mut records = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        records.push(run_once_qwen(
            model,
            mtp.as_ref(),
            tokenizer,
            &prompt_ids,
            args,
            mtp_draft_tokens,
            effective_cap_max,
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-core",
            mode: args.mode,
            model_dir: args.model.display().to_string(),
            mtp_model_dir: args
                .mtp_model_dir
                .as_ref()
                .map(|dir| dir.display().to_string()),
            mtp_draft_tokens,
            prompt_file: args.prompt_file.display().to_string(),
            prompt_tokens: prompt_ids.len(),
            max_tokens: args.max_tokens,
            prefill_chunk_size: args.prefill_chunk_size,
            kv_quant: args.kv_quant,
            b_max: args.b_max,
            effective_cap_max,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            load_ms,
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
    effective_cap_max: usize,
) -> Result<Record>
where
    M: Model + DenseVlMethods,
{
    match args.mode {
        BenchMode::Gs => run_generation_stream(model, tokenizer, prompt_ids, args),
        BenchMode::Mtp => Err(anyhow!(
            "mtp-text mode is only supported for Qwen dense/MoE text models"
        )),
        BenchMode::Scheduler => {
            run_scheduler(model, tokenizer, prompt_ids, args, effective_cap_max)
        }
    }
}

fn run_once_qwen<M>(
    model: &M,
    mtp: Option<&M::MtpHead>,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    args: &Args,
    mtp_draft_tokens: Option<usize>,
    effective_cap_max: usize,
) -> Result<Record>
where
    M: MtpSpeculativeModel + DenseVlMethods,
{
    match args.mode {
        BenchMode::Gs => run_generation_stream(model, tokenizer, prompt_ids, args),
        BenchMode::Mtp => {
            let mtp = mtp.ok_or_else(|| anyhow!("mtp-text mode requires a loaded MTP head"))?;
            let mtp_draft_tokens =
                mtp_draft_tokens.ok_or_else(|| anyhow!("MTP run missing resolved draft tokens"))?;
            run_mtp_generation_stream(model, mtp, tokenizer, prompt_ids, args, mtp_draft_tokens)
        }
        BenchMode::Scheduler => {
            if let Some(mtp) = mtp {
                let mtp_draft_tokens = mtp_draft_tokens
                    .ok_or_else(|| anyhow!("scheduler MTP run missing resolved draft tokens"))?;
                run_scheduler_mtp_single_request(
                    model,
                    mtp,
                    tokenizer,
                    prompt_ids,
                    args,
                    mtp_draft_tokens,
                    effective_cap_max,
                )
            } else {
                run_scheduler(model, tokenizer, prompt_ids, args, effective_cap_max)
            }
        }
    }
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
    Ok(make_record(
        args.mode,
        ttft_ms,
        e2e_ms,
        GeneratedOutput {
            token_ids: generated_token_ids,
            text: generated_text,
        },
        finish_reason,
        args.max_tokens,
        None,
    ))
}

fn run_scheduler<M>(
    model: &M,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    args: &Args,
    effective_cap_max: usize,
) -> Result<Record>
where
    M: Model + DenseVlMethods,
{
    let mut scheduler = Scheduler::<M>::new(args.b_max, effective_cap_max, model.model_meta())
        .context("Scheduler::new")?;
    let request = make_request(model, tokenizer, prompt_ids, args);
    let started = Instant::now();
    let _request_id = scheduler.admit(request)?;
    let first_events = scheduler.prefill_admitted(model)?;
    let mut generated_token_ids: Vec<u32> = first_events.iter().map(|event| event.token).collect();
    let mut finish_reason = first_events.first().and_then(|event| event.finish_reason);
    let ttft_ms = started.elapsed().as_secs_f64() * 1000.0;

    while finish_reason.is_none() && generated_token_ids.len() < args.max_tokens {
        let events = scheduler.step(model)?;
        if events.is_empty() {
            break;
        }
        generated_token_ids.extend(events.iter().map(|event| event.token));
        finish_reason = events.first().and_then(|event| event.finish_reason);
    }
    mlx::transforms::synchronize()?;
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    let generated_text = tokenizer
        .decode(&generated_token_ids, true)
        .unwrap_or_default();
    Ok(make_record(
        args.mode,
        ttft_ms,
        e2e_ms,
        GeneratedOutput {
            token_ids: generated_token_ids,
            text: generated_text,
        },
        finish_reason,
        args.max_tokens,
        None,
    ))
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
    Ok(make_record(
        args.mode,
        ttft_ms,
        e2e_ms,
        GeneratedOutput {
            token_ids: generated_token_ids,
            text: generated_text,
        },
        finish_reason,
        args.max_tokens,
        Some(mtp_stats),
    ))
}

fn run_scheduler_mtp_single_request<M>(
    model: &M,
    mtp: &M::MtpHead,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    args: &Args,
    mtp_draft_tokens: usize,
    effective_cap_max: usize,
) -> Result<Record>
where
    M: MtpSpeculativeModel + DenseVlMethods,
{
    let mut scheduler =
        Scheduler::<M>::new(1, effective_cap_max, model.model_meta()).context("Scheduler::new")?;
    let request = make_request(model, tokenizer, prompt_ids, args);
    let cfg = MtpSpeculativeConfig::new(mtp_draft_tokens, request.sampler)?;
    let started = Instant::now();
    let _request_id = scheduler.admit(request)?;
    let first_events = scheduler.prefill_admitted_mtp_single(model, mtp, cfg)?;
    let mut generated_token_ids: Vec<u32> = first_events.iter().map(|event| event.token).collect();
    let mut finish_reason = first_events.first().and_then(|event| event.finish_reason);
    let ttft_ms = started.elapsed().as_secs_f64() * 1000.0;

    while finish_reason.is_none() && generated_token_ids.len() < args.max_tokens {
        let events = scheduler.step_mtp_single(model, mtp)?;
        if events.is_empty() {
            break;
        }
        generated_token_ids.extend(events.iter().map(|event| event.token));
        finish_reason = events.first().and_then(|event| event.finish_reason);
    }
    mlx::transforms::synchronize()?;
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    let generated_text = tokenizer
        .decode(&generated_token_ids, true)
        .unwrap_or_default();
    let mtp_stats = scheduler
        .mtp_stats()
        .ok_or_else(|| anyhow!("scheduler MTP run produced no MTP stats"))?
        .into();
    Ok(make_record(
        args.mode,
        ttft_ms,
        e2e_ms,
        GeneratedOutput {
            token_ids: generated_token_ids,
            text: generated_text,
        },
        finish_reason,
        args.max_tokens,
        Some(mtp_stats),
    ))
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
        #[cfg(feature = "p5h-profile")]
        p5h_trace: None,
        #[cfg(feature = "p5h-profile")]
        p5h_root_span: None,
    }
}

fn make_record(
    mode: BenchMode,
    ttft_ms: f64,
    e2e_ms: f64,
    generated: GeneratedOutput,
    finish_reason: Option<&'static str>,
    max_tokens: usize,
    mtp_stats: Option<MtpRecordStats>,
) -> Record {
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
        finish_reason,
        valid: finish_reason == Some("length") && generated_tokens >= max_tokens,
        mtp_stats,
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
    fn scheduler_text_rejects_mtp_model_dir_for_batched_scheduler() {
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
        let err = validate_args(&args).unwrap_err();
        std::fs::remove_dir_all(mtp_dir).ok();
        assert!(err.to_string().contains("--b-max 1"));
    }

    #[test]
    fn scheduler_text_mtp_keeps_scheduler_mode_stats_contract() {
        let record = make_record(
            BenchMode::Scheduler,
            1.0,
            3.0,
            GeneratedOutput {
                token_ids: vec![101, 102, 103, 104],
                text: "bench text".to_string(),
            },
            Some("length"),
            4,
            Some(MtpRecordStats {
                windows: 1,
                drafted_tokens: 2,
                accepted_draft_tokens: 1,
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
                main_rollback_us: 0,
                mtp_cache_commit_us: 0,
                mtp_cache_restore_us: 0,
            }),
        );

        assert_eq!(record.mode, BenchMode::Scheduler);
        assert!(record.valid);
        assert_eq!(record.generated_token_ids, vec![101, 102, 103, 104]);
        assert_eq!(record.generated_text, "bench text");
        let stats = record.mtp_stats.expect("scheduler MTP stats");
        assert_eq!(stats.windows, 1);
        assert_eq!(stats.rollback_count, 1);
        assert_eq!(stats.acceptance_rate, Some(0.5));
    }
}
