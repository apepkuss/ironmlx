//! Minimal in-process benchmark for ironmlx model-core text generation paths.
//!
//! This binary intentionally bypasses HTTP so scheduler/model gaps can be
//! measured without server parsing, SSE, or client timing noise.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::scheduler::DenseVlMethods;
use ironmlx::core::{GenerateRequest, GenerationStream, Loader, Model, Sampler, Scheduler};
use ironmlx::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF;
use ironmlx::models::{
    Glm4MoeLiteModel, LlamaModel, ModelArchitecture, Qwen35Model, Qwen35MoeModel,
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

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum BenchMode {
    GsText,
    SchedulerText,
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
    prompt_file: String,
    prompt_tokens: usize,
    max_tokens: usize,
    prefill_chunk_size: usize,
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
    generation_tps: f64,
    finish_reason: Option<&'static str>,
    valid: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let load_started = Instant::now();
    let loader = Loader::open(&args.model).context("Loader::open")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let architecture = ModelArchitecture::from_config_value(loader.config_raw_value())?;

    match architecture {
        ModelArchitecture::Qwen35Dense => {
            let model = Qwen35Model::from_loader(&loader).context("Qwen35Model::from_loader")?;
            let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
            run_for_model(&model, &tokenizer, &args, load_ms)
        }
        ModelArchitecture::Qwen35Moe => {
            let model =
                Qwen35MoeModel::from_loader(&loader).context("Qwen35MoeModel::from_loader")?;
            let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
            run_for_model(&model, &tokenizer, &args, load_ms)
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
            // Text-only Qwen3.5 dense backbone (SigLIP vision tower not yet implemented).
            let model = ironmlx::models::minicpmv4_6::model_from_loader(&loader)
                .context("minicpmv4_6::model_from_loader")?;
            let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
            run_for_model(&model, &tokenizer, &args, load_ms)
        }
    }
}

fn run_for_model<M>(model: &M, tokenizer: &Tokenizer, args: &Args, load_ms: f64) -> Result<()>
where
    M: Model + DenseVlMethods,
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
            prompt_file: args.prompt_file.display().to_string(),
            prompt_tokens: prompt_ids.len(),
            max_tokens: args.max_tokens,
            prefill_chunk_size: args.prefill_chunk_size,
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
        BenchMode::GsText => run_generation_stream(model, tokenizer, prompt_ids, args),
        BenchMode::SchedulerText => {
            run_scheduler(model, tokenizer, prompt_ids, args, effective_cap_max)
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
    let mut generated = 0_usize;
    let mut finish_reason = None;

    while let Some(event) = stream.next_token()? {
        if first_ms.is_none() {
            first_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
        }
        generated += 1;
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
        generated,
        finish_reason,
        args.max_tokens,
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
    let mut generated = first_events.len();
    let mut finish_reason = first_events.first().and_then(|event| event.finish_reason);
    let ttft_ms = started.elapsed().as_secs_f64() * 1000.0;

    while finish_reason.is_none() && generated < args.max_tokens {
        let events = scheduler.step(model)?;
        if events.is_empty() {
            break;
        }
        generated += events.len();
        finish_reason = events.first().and_then(|event| event.finish_reason);
    }
    mlx::transforms::synchronize()?;
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(make_record(
        args.mode,
        ttft_ms,
        e2e_ms,
        generated,
        finish_reason,
        args.max_tokens,
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
    generated_tokens: usize,
    finish_reason: Option<&'static str>,
    max_tokens: usize,
) -> Record {
    let decode_time_ms = (e2e_ms - ttft_ms).max(0.0);
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
        generation_tps,
        finish_reason,
        valid: finish_reason == Some("length") && generated_tokens >= max_tokens,
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
