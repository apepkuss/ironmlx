//! Controlled B=1 comparison of direct `GenerationStream` and `SchedulerActor`.
//!
//! The paired mode loads one model, alternates path order, and sends identical
//! requests through both paths. Direct-only and actor-only modes exist so MLX
//! process peak memory can be measured without contamination from the other
//! path.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::loader::Loader;
use ironmlx_lm::models::Qwen35Model;
use ironmlx_lm::Tokenizer;
use ironmlx_runtime::core::generate::{GenerateRequest, GenerationStream};
use ironmlx_runtime::core::scheduler_actor::{
    spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand,
};
use serde::Serialize;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(
    name = "b1-path-qualification",
    about = "Paired B=1 GenerationStream/SchedulerActor qualification",
    version
)]
struct Args {
    /// Local Qwen3.5 dense model directory.
    #[arg(long)]
    model: PathBuf,

    /// Exact mlx.metallib to load. Defaults to $MLX_DIR/lib/mlx.metallib.
    #[arg(long)]
    mlx_metallib: Option<PathBuf>,

    /// File containing a raw prompt seed.
    #[arg(long)]
    prompt_file: PathBuf,

    /// Repeat the encoded prompt seed to exactly this many tokens.
    #[arg(long)]
    prompt_target_tokens: usize,

    /// Generated tokens per request.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Qualification path. `paired` alternates direct and actor ordering.
    #[arg(long, value_enum, default_value_t = QualificationPath::Paired)]
    path: QualificationPath,

    /// Sampling mode.
    #[arg(long, value_enum, default_value_t = SamplingMode::Greedy)]
    sampling: SamplingMode,

    /// Temperature for sampled mode.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Nucleus sampling threshold for sampled mode.
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Stable request seed used by both paths.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Warmup pairs/runs excluded from the measured records.
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,

    /// Measured pairs/runs.
    #[arg(long, default_value_t = 9)]
    runs: usize,

    /// Idle interval after each path run, excluded from timing.
    #[arg(long, default_value_t = 1000)]
    cooldown_ms: u64,

    /// Prefill chunk size used by both paths.
    #[arg(long, default_value_t = 2048)]
    prefill_chunk_size: usize,

    /// SchedulerActor B capacity. Qualification remains B=1 active traffic.
    #[arg(long, default_value_t = 1)]
    b_max: usize,

    /// SchedulerActor admission window.
    #[arg(long, default_value_t = 5)]
    admission_deadline_ms: u64,

    /// Scheduler request queue capacity.
    #[arg(long, default_value_t = 32)]
    admission_queue_max: usize,

    /// Scheduler cache cap. Defaults to prompt + generated tokens.
    #[arg(long)]
    effective_cap_max: Option<usize>,

    /// JSON output path.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum QualificationPath {
    Paired,
    Direct,
    Actor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum SamplingMode {
    Greedy,
    Sampled,
}

#[derive(Clone, Serialize)]
struct MemoryStats {
    active_bytes: usize,
    cache_bytes: usize,
    peak_bytes: usize,
}

impl MemoryStats {
    fn capture() -> Self {
        let snapshot = mlx::memory::snapshot();
        Self {
            active_bytes: snapshot.active_bytes,
            cache_bytes: snapshot.cache_bytes,
            peak_bytes: snapshot.peak_bytes,
        }
    }
}

#[derive(Serialize)]
struct Meta {
    model_dir: String,
    mlx_metallib: String,
    prompt_file: String,
    prompt_tokens: usize,
    max_tokens: usize,
    path: QualificationPath,
    sampling: SamplingMode,
    temperature: f32,
    top_p: f32,
    seed: u64,
    warmup_runs: usize,
    measured_runs: usize,
    cooldown_ms: u64,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    effective_cap_max: usize,
    model_load_ms: f64,
    device_name: Option<String>,
    ironmlx_version: &'static str,
}

#[derive(Clone, Serialize)]
struct PathRecord {
    ttft_ms: f64,
    e2e_ms: f64,
    decode_ms: f64,
    generation_tps: f64,
    generated_token_ids: Vec<u32>,
    finish_reason: Option<&'static str>,
    valid: bool,
    memory: MemoryStats,
}

#[derive(Serialize)]
struct RunRecord {
    index: usize,
    order: &'static str,
    direct: Option<PathRecord>,
    actor: Option<PathRecord>,
    token_exact: Option<bool>,
}

#[derive(Serialize)]
struct Output {
    meta: Meta,
    records: Vec<RunRecord>,
    final_memory: MemoryStats,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let metallib = resolve_metallib(&args)?;
    mlx::metal::set_metallib_path(
        metallib
            .to_str()
            .ok_or_else(|| anyhow!("mlx.metallib path is not UTF-8"))?,
    )
    .context("loading mlx.metallib")?;

    let load_started = Instant::now();
    let loader = Loader::open(&args.model).context("Loader::open")?;
    let tokenizer = Arc::new(Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?);
    let model = Qwen35Model::from_loader(&loader).context("Qwen35Model::from_loader")?;
    let model_load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let prompt_seed = std::fs::read_to_string(&args.prompt_file)
        .with_context(|| format!("reading {}", args.prompt_file.display()))?;
    let encoded = tokenizer.encode(&prompt_seed, false)?;
    if encoded.is_empty() {
        return Err(anyhow!("prompt seed encoded to zero tokens"));
    }
    let prompt_ids = encoded
        .iter()
        .copied()
        .cycle()
        .take(args.prompt_target_tokens)
        .collect::<Vec<_>>();
    let effective_cap_max = args
        .effective_cap_max
        .unwrap_or_else(|| prompt_ids.len().saturating_add(args.max_tokens));
    if effective_cap_max < prompt_ids.len().saturating_add(args.max_tokens) {
        return Err(anyhow!(
            "--effective-cap-max {effective_cap_max} is smaller than prompt + output {}",
            prompt_ids.len().saturating_add(args.max_tokens)
        ));
    }

    let model = Arc::new(Mutex::new(model));
    let actor = if matches!(
        args.path,
        QualificationPath::Paired | QualificationPath::Actor
    ) {
        let meta = model.lock().await.model_meta();
        Some(
            spawn_scheduler_actor(
                Arc::clone(&model),
                args.b_max,
                Duration::from_millis(args.admission_deadline_ms),
                args.admission_queue_max,
                effective_cap_max,
                256,
                meta,
            )
            .map_err(|err| anyhow!("spawn_scheduler_actor: {err}"))?,
        )
    } else {
        None
    };

    for index in 0..args.warmup_runs {
        let record = run_iteration(
            index,
            &args,
            &model,
            actor.as_ref(),
            &tokenizer,
            &prompt_ids,
        )
        .await?;
        log_record("warmup", &record);
    }

    let mut records = Vec::with_capacity(args.runs);
    for index in 0..args.runs {
        let record = run_iteration(
            index,
            &args,
            &model,
            actor.as_ref(),
            &tokenizer,
            &prompt_ids,
        )
        .await?;
        log_record("measured", &record);
        records.push(record);
    }

    let output = Output {
        meta: Meta {
            model_dir: args.model.display().to_string(),
            mlx_metallib: metallib.display().to_string(),
            prompt_file: args.prompt_file.display().to_string(),
            prompt_tokens: prompt_ids.len(),
            max_tokens: args.max_tokens,
            path: args.path,
            sampling: args.sampling,
            temperature: args.temperature,
            top_p: args.top_p,
            seed: args.seed,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            cooldown_ms: args.cooldown_ms,
            prefill_chunk_size: args.prefill_chunk_size,
            b_max: args.b_max,
            admission_deadline_ms: args.admission_deadline_ms,
            admission_queue_max: args.admission_queue_max,
            effective_cap_max,
            model_load_ms,
            device_name: mlx::memory::snapshot().device_name,
            ironmlx_version: env!("CARGO_PKG_VERSION"),
        },
        records,
        final_memory: MemoryStats::capture(),
    };
    std::fs::write(&args.out, serde_json::to_string_pretty(&output)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;
    Ok(())
}

fn log_record(stage: &str, record: &RunRecord) {
    let describe = |value: &Option<PathRecord>| match value {
        Some(value) => format!(
            "ttft={:.2}ms e2e={:.2}ms tps={:.2}",
            value.ttft_ms, value.e2e_ms, value.generation_tps
        ),
        None => "n/a".to_string(),
    };
    eprintln!(
        "{stage} {} {}: direct [{}] actor [{}] exact={:?}",
        record.index,
        record.order,
        describe(&record.direct),
        describe(&record.actor),
        record.token_exact
    );
}

fn resolve_metallib(args: &Args) -> Result<PathBuf> {
    let path = match args.mlx_metallib.as_ref() {
        Some(path) => path.clone(),
        None => PathBuf::from(
            std::env::var_os("MLX_DIR")
                .ok_or_else(|| anyhow!("--mlx-metallib or MLX_DIR is required"))?,
        )
        .join("lib/mlx.metallib"),
    };
    anyhow::ensure!(
        path.is_file(),
        "mlx.metallib does not exist: {}",
        path.display()
    );
    Ok(path)
}

fn validate_args(args: &Args) -> Result<()> {
    anyhow::ensure!(
        args.prompt_target_tokens > 0,
        "--prompt-target-tokens must be > 0"
    );
    anyhow::ensure!(args.max_tokens > 1, "--max-tokens must be > 1");
    anyhow::ensure!(args.runs > 0, "--runs must be > 0");
    anyhow::ensure!(args.b_max > 0, "--b-max must be > 0");
    if args.sampling == SamplingMode::Sampled {
        anyhow::ensure!(
            args.temperature > 0.0,
            "sampled mode requires --temperature > 0"
        );
        anyhow::ensure!(
            args.top_p > 0.0 && args.top_p <= 1.0,
            "--top-p must be in (0, 1]"
        );
    }
    Ok(())
}

async fn run_iteration(
    index: usize,
    args: &Args,
    model: &Arc<Mutex<Qwen35Model>>,
    actor: Option<&SchedulerActorHandle>,
    tokenizer: &Arc<Tokenizer>,
    prompt_ids: &[u32],
) -> Result<RunRecord> {
    let actor_first = index % 2 == 1;
    let mut direct = None;
    let mut actor_record = None;

    match args.path {
        QualificationPath::Paired if actor_first => {
            actor_record = Some(run_actor(actor.expect("paired actor"), args, prompt_ids).await?);
            cooldown(args).await;
            direct = Some(run_direct(model, tokenizer, args, prompt_ids).await?);
            cooldown(args).await;
        }
        QualificationPath::Paired => {
            direct = Some(run_direct(model, tokenizer, args, prompt_ids).await?);
            cooldown(args).await;
            actor_record = Some(run_actor(actor.expect("paired actor"), args, prompt_ids).await?);
            cooldown(args).await;
        }
        QualificationPath::Direct => {
            direct = Some(run_direct(model, tokenizer, args, prompt_ids).await?);
            cooldown(args).await;
        }
        QualificationPath::Actor => {
            actor_record = Some(run_actor(actor.expect("actor path"), args, prompt_ids).await?);
            cooldown(args).await;
        }
    }

    let token_exact = direct
        .as_ref()
        .zip(actor_record.as_ref())
        .map(|(left, right)| left.generated_token_ids == right.generated_token_ids);
    Ok(RunRecord {
        index,
        order: match args.path {
            QualificationPath::Paired if actor_first => "actor-direct",
            QualificationPath::Paired => "direct-actor",
            QualificationPath::Direct => "direct",
            QualificationPath::Actor => "actor",
        },
        direct,
        actor: actor_record,
        token_exact,
    })
}

async fn run_direct(
    model: &Arc<Mutex<Qwen35Model>>,
    tokenizer: &Tokenizer,
    args: &Args,
    prompt_ids: &[u32],
) -> Result<PathRecord> {
    let started = Instant::now();
    let model = model.lock().await;
    let request = make_request(&model, tokenizer, args, prompt_ids);
    let mut stream = GenerationStream::new_text_only(&*model, tokenizer, request)?;
    let mut first_ms = None;
    let mut tokens = Vec::with_capacity(args.max_tokens);
    let mut finish_reason = None;
    while let Some(event) = stream.next_token()? {
        first_ms.get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.0);
        tokens.push(event.token);
        finish_reason = event.finish_reason;
        if finish_reason.is_some() {
            break;
        }
    }
    mlx::transforms::synchronize()?;
    make_record(started, first_ms, tokens, finish_reason, args.max_tokens)
}

async fn run_actor(
    actor: &SchedulerActorHandle,
    args: &Args,
    prompt_ids: &[u32],
) -> Result<PathRecord> {
    let started = Instant::now();
    let request = make_request_without_model(args, prompt_ids);
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    actor
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .map_err(|_| anyhow!("sending actor admit: actor command channel closed"))?;
    let reply = reply_rx.await.context("receiving actor admit reply")??;
    let mut event_rx = reply.event_rx;
    let mut first_ms = None;
    let mut tokens = Vec::with_capacity(args.max_tokens);
    let mut finish_reason = None;
    while let Some(event) = event_rx.recv().await {
        first_ms.get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.0);
        tokens.push(event.token);
        finish_reason = event.finish_reason;
        if finish_reason.is_some() {
            break;
        }
    }
    mlx::transforms::synchronize()?;
    make_record(started, first_ms, tokens, finish_reason, args.max_tokens)
}

fn make_record(
    started: Instant,
    first_ms: Option<f64>,
    tokens: Vec<u32>,
    finish_reason: Option<&'static str>,
    max_tokens: usize,
) -> Result<PathRecord> {
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = first_ms.ok_or_else(|| anyhow!("path produced no token events"))?;
    let decode_ms = (e2e_ms - ttft_ms).max(0.0);
    let generation_tps = if decode_ms > 0.0 {
        tokens.len().saturating_sub(1) as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let valid = finish_reason == Some("length") && tokens.len() >= max_tokens;
    Ok(PathRecord {
        ttft_ms,
        e2e_ms,
        decode_ms,
        generation_tps,
        generated_token_ids: tokens,
        finish_reason,
        valid,
        memory: MemoryStats::capture(),
    })
}

fn make_request(
    model: &Qwen35Model,
    tokenizer: &Tokenizer,
    args: &Args,
    prompt_ids: &[u32],
) -> GenerateRequest {
    let mut request = make_request_without_model(args, prompt_ids);
    request.stop_token_ids = Vec::new();
    request.image_spatial_merge_size = model.model_meta().spatial_merge_size;
    request.image_token_id = tokenizer
        .token_to_id("<|image_pad|>")
        .map(|id| id as i32)
        .unwrap_or(248_056);
    request
}

fn make_request_without_model(args: &Args, prompt_ids: &[u32]) -> GenerateRequest {
    let sampler = match args.sampling {
        SamplingMode::Greedy => Sampler::greedy(),
        SamplingMode::Sampled => Sampler::greedy()
            .with_temperature(args.temperature)
            .with_top_p(args.top_p)
            .with_seed(args.seed),
    };
    GenerateRequest {
        priority: Default::default(),
        prompt_ids: prompt_ids.to_vec(),
        max_new_tokens: args.max_tokens,
        sampler,
        stop_token_ids: Vec::new(),
        prefill_chunk_size: args.prefill_chunk_size,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248_056,
        constraint: None,
    }
}

async fn cooldown(args: &Args) {
    if args.cooldown_ms > 0 {
        tokio::time::sleep(Duration::from_millis(args.cooldown_ms)).await;
    }
}
