//! Limited local-only B2/B4 Q8/Q16 feasibility gate for Qwen3.8 DFlash2.
//!
//! The runner never resolves Hub ids or downloads artifacts. Q16 tensor
//! shapes are enabled only while a tools-only qualification guard is held.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::models::{DFlash2DraftModel, Qwen35Model};
use ironmlx_lm::{Loader, Tokenizer};
use ironmlx_runtime::core::dflash2::{enable_batched_q16_qualification, DFlash2P2Options};
use ironmlx_runtime::core::engine_state::{build_dflash2_engine_with_options, CausalEngine};
use ironmlx_runtime::core::generation_types::GenerateRequest;
use ironmlx_runtime::core::process_memory::StaticMemoryEstimate;
use ironmlx_runtime::core::scheduler_autotune::SchedulerAutotuneRuntimeContext;
use ironmlx_runtime::core::scheduler_resolution::default_scheduler_runtime_profile;
use serde::Serialize;
use tokio::task::JoinSet;

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Greedy,
    Sampled,
}

#[derive(Debug, Parser)]
#[command(about = "Run a local-only DFlash2 B2/B4 Q8 or Q16 feasibility gate")]
struct Args {
    #[arg(long)]
    target_dir: PathBuf,
    #[arg(long)]
    draft_dir: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_parser = clap::value_parser!(usize), default_value_t = 8)]
    block_size: usize,
    #[arg(long, default_value = "2,4", value_delimiter = ',')]
    batch_widths: Vec<usize>,
    #[arg(long, default_value = "2048,8192", value_delimiter = ',')]
    context_lengths: Vec<usize>,
    #[arg(long, default_value = "greedy", value_delimiter = ',')]
    modes: Vec<Mode>,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 1)]
    warmup_batches: usize,
    #[arg(long, default_value_t = 5)]
    measured_batches: usize,
    #[arg(long, default_value_t = 20_260_928)]
    seed: u64,
    #[arg(long, default_value = "ff2fcea50a49ed5c9213c70ddcdbc1decc605655")]
    frozen_p2_sha: String,
}

#[derive(Debug, Serialize)]
struct BatchRecord {
    mode: Mode,
    context_tokens: usize,
    batch_width: usize,
    replicate: usize,
    block_size: usize,
    full_wall_us: u64,
    generation_wall_us: u64,
    token_ids: Vec<Vec<u32>>,
    tensor_batch_windows: u64,
    tensor_batch_groups_created: u64,
    tensor_batch_divergent_splits: u64,
    observed_tensor_batch_max_width: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    frozen_p2_sha: String,
    execution_scope: &'static str,
    target_dir: String,
    draft_dir: String,
    draft_checkpoint_block_size: usize,
    block_size: usize,
    max_new_tokens: usize,
    warmup_batches: usize,
    measured_batches: usize,
    seed: u64,
    records: Vec<BatchRecord>,
}

fn local_model_root() -> Result<PathBuf> {
    let home = dirs::home_dir().context("resolve home directory")?;
    Ok(home.join(".ironmlx/models/huggingface"))
}

fn validate_local_checkpoint(path: &Path, label: &str) -> Result<PathBuf> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("resolve {label} checkpoint {}", path.display()))?;
    let root = local_model_root()?
        .canonicalize()
        .context("resolve ~/.ironmlx/models/huggingface")?;
    anyhow::ensure!(
        canonical.starts_with(&root),
        "{label} checkpoint must be below {}; got {}",
        root.display(),
        canonical.display()
    );
    anyhow::ensure!(
        canonical.join("config.json").is_file(),
        "{label} checkpoint has no config.json: {}",
        canonical.display()
    );
    Ok(canonical)
}

fn prompt_ids(context_tokens: usize) -> Result<Vec<u32>> {
    anyhow::ensure!(context_tokens >= 4, "context length must be at least four");
    let mut tokens = Vec::with_capacity(context_tokens);
    tokens.push(151_644);
    tokens.resize(context_tokens - 2, 872);
    tokens.extend([198, 3_838]);
    Ok(tokens)
}

fn sampler(mode: Mode, seed: u64) -> Sampler {
    match mode {
        Mode::Greedy => Sampler::greedy(),
        Mode::Sampled => Sampler::greedy()
            .with_temperature(0.7)
            .with_top_p(1.0)
            .with_seed(seed),
    }
}

fn admission_error(label: &str) -> anyhow::Error {
    anyhow!("DFlash2 {label} admission failed")
}

#[allow(clippy::too_many_arguments)]
async fn run_batch(
    engine: &CausalEngine<Qwen35Model>,
    mode: Mode,
    context_tokens: usize,
    batch_width: usize,
    max_new_tokens: usize,
    seed: u64,
    block_size: usize,
    replicate: usize,
) -> Result<BatchRecord> {
    let before = engine.health_collector.snapshot().dflash2;
    let prompt = prompt_ids(context_tokens)?;
    let started = Instant::now();
    let mut admissions = JoinSet::new();
    for row in 0..batch_width {
        let execution = engine.request_execution.clone();
        let request = GenerateRequest {
            priority: Default::default(),
            prompt_ids: prompt.clone(),
            max_new_tokens,
            sampler: sampler(mode, seed ^ row as u64),
            stop_token_ids: Vec::new(),
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 1,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: 248_056,
            constraint: None,
        };
        admissions.spawn(async move {
            let admitted = execution
                .admit(request)
                .await
                .map_err(|_| admission_error("request"))?;
            Ok::<_, anyhow::Error>((row, Instant::now(), admitted.event_rx))
        });
    }

    let mut admitted = Vec::with_capacity(batch_width);
    while let Some(result) = admissions.join_next().await {
        admitted.push(result.context("join DFlash2 admission task")??);
    }
    anyhow::ensure!(admitted.len() == batch_width, "not all rows were admitted");
    let generation_started = admitted
        .iter()
        .map(|(_, admitted_at, _)| *admitted_at)
        .min()
        .context("no admission timestamp")?;

    let mut drains = JoinSet::new();
    for (row, _, mut event_rx) in admitted {
        drains.spawn(async move {
            let mut tokens = Vec::with_capacity(max_new_tokens);
            let mut finished = false;
            while let Some(event) = event_rx.recv().await {
                tokens.push(event.token);
                if event.finish_reason.is_some() {
                    finished = true;
                    break;
                }
            }
            anyhow::ensure!(
                finished,
                "DFlash2 row {row} closed without a terminal event"
            );
            anyhow::ensure!(
                tokens.len() == max_new_tokens,
                "DFlash2 row {row} emitted {} of {max_new_tokens} tokens",
                tokens.len()
            );
            Ok::<_, anyhow::Error>((row, tokens))
        });
    }
    let mut rows = Vec::with_capacity(batch_width);
    while let Some(result) = drains.join_next().await {
        rows.push(result.context("join DFlash2 drain task")??);
    }
    rows.sort_by_key(|(row, _)| *row);
    let completed = Instant::now();
    let after = engine.health_collector.snapshot().dflash2;
    let tensor_batch_windows = after
        .tensor_batch_windows
        .saturating_sub(before.tensor_batch_windows);
    anyhow::ensure!(
        tensor_batch_windows > 0,
        "B{batch_width}/Q{block_size} executed no tensor-batched windows"
    );
    anyhow::ensure!(
        after.tensor_batch_max_width >= batch_width,
        "B{batch_width}/Q{block_size} observed tensor width {}",
        after.tensor_batch_max_width
    );
    Ok(BatchRecord {
        mode,
        context_tokens,
        batch_width,
        replicate,
        block_size,
        full_wall_us: completed.duration_since(started).as_micros() as u64,
        generation_wall_us: completed.duration_since(generation_started).as_micros() as u64,
        token_ids: rows.into_iter().map(|(_, tokens)| tokens).collect(),
        tensor_batch_windows,
        tensor_batch_groups_created: after
            .tensor_batch_groups_created
            .saturating_sub(before.tensor_batch_groups_created),
        tensor_batch_divergent_splits: after
            .tensor_batch_divergent_splits
            .saturating_sub(before.tensor_batch_divergent_splits),
        observed_tensor_batch_max_width: after.tensor_batch_max_width,
    })
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        matches!(args.block_size, 8 | 16),
        "block size must be 8 or 16"
    );
    anyhow::ensure!(args.max_new_tokens > 0, "max new tokens must be positive");
    anyhow::ensure!(
        args.measured_batches >= 5,
        "gate requires at least five batches"
    );
    anyhow::ensure!(
        !args.batch_widths.is_empty()
            && args.batch_widths.iter().all(|width| matches!(width, 2 | 4)),
        "batch widths must contain only 2 or 4"
    );
    let _qualification_guard = (args.block_size == 16)
        .then(enable_batched_q16_qualification)
        .transpose()?;
    let target_dir = validate_local_checkpoint(&args.target_dir, "target")?;
    let draft_dir = validate_local_checkpoint(&args.draft_dir, "draft")?;
    let mut target_loader = Loader::open(&target_dir).context("open target checkpoint")?;
    let tokenizer = Tokenizer::from_loader(&target_loader).context("load tokenizer")?;
    let model = Qwen35Model::from_loader_dflash2(&mut target_loader)
        .context("load Qwen3.8 DFlash2 target")?;
    let draft_loader = Loader::open_dflash2(&draft_dir).context("open draft checkpoint")?;
    let draft = DFlash2DraftModel::from_loader(&draft_loader, model.config(), Some(4))
        .context("load runtime-quantized DFlash2 draft")?;
    let checkpoint_block_size = usize::try_from(draft.config().dflash_config.block_size)?;
    anyhow::ensure!(
        checkpoint_block_size >= args.block_size,
        "draft checkpoint block_size={checkpoint_block_size} cannot run Q{}",
        args.block_size
    );
    let static_memory_estimate = StaticMemoryEstimate {
        text_cold_bytes: target_loader.loaded_tensor_bytes(),
        speculative_cold_bytes: draft_loader.loaded_tensor_bytes(),
        ..Default::default()
    };
    drop(target_loader);
    drop(draft_loader);
    let max_context = args
        .context_lengths
        .iter()
        .copied()
        .max()
        .context("no context lengths selected")?;
    let effective_cap = max_context
        .checked_add(args.max_new_tokens)
        .and_then(|value| value.checked_add(16))
        .context("effective cache cap overflow")?;
    let engine = build_dflash2_engine_with_options(
        model,
        draft,
        tokenizer,
        format!("dflash2-batched-q{}-gate", args.block_size),
        0,
        4,
        50,
        4,
        8,
        effective_cap,
        args.block_size,
        DFlash2P2Options {
            tree_max_nodes: 0,
            position_keyed_sampling: args.modes.iter().any(|mode| matches!(mode, Mode::Sampled)),
        },
        Some(4),
        None,
        default_scheduler_runtime_profile(SchedulerAutotuneRuntimeContext::local_default(
            effective_cap,
        )),
        static_memory_estimate,
    )
    .await
    .context("build DFlash2 qualification engine")?;

    let mut records = Vec::new();
    for &mode in &args.modes {
        for &context_tokens in &args.context_lengths {
            for &batch_width in &args.batch_widths {
                for batch in 0..args.warmup_batches + args.measured_batches {
                    let replicate = batch.saturating_sub(args.warmup_batches);
                    let batch_seed = args.seed
                        ^ ((context_tokens as u64) << 16)
                        ^ ((batch_width as u64) << 8)
                        ^ replicate as u64;
                    let record = run_batch(
                        &engine,
                        mode,
                        context_tokens,
                        batch_width,
                        args.max_new_tokens,
                        batch_seed,
                        args.block_size,
                        replicate,
                    )
                    .await?;
                    eprintln!(
                        "{:?} C{} B{} batch {}/{} Q{}: generation={:.3}s aggregate={:.3} tok/s tensor_windows={}",
                        mode,
                        context_tokens,
                        batch_width,
                        batch + 1,
                        args.warmup_batches + args.measured_batches,
                        args.block_size,
                        record.generation_wall_us as f64 / 1_000_000.0,
                        batch_width as f64 * args.max_new_tokens as f64 * 1_000_000.0
                            / record.generation_wall_us as f64,
                        record.tensor_batch_windows,
                    );
                    if batch >= args.warmup_batches {
                        records.push(record);
                    }
                }
                mlx::clear_cache();
            }
        }
    }
    let report = Report {
        schema_version: SCHEMA_VERSION,
        frozen_p2_sha: args.frozen_p2_sha,
        execution_scope: "limited B2/B4 linear tensor-batch feasibility gate",
        target_dir: target_dir.display().to_string(),
        draft_dir: draft_dir.display().to_string(),
        draft_checkpoint_block_size: checkpoint_block_size,
        block_size: args.block_size,
        max_new_tokens: args.max_new_tokens,
        warmup_batches: args.warmup_batches,
        measured_batches: args.measured_batches,
        seed: args.seed,
        records,
    };
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", args.output.display()))?;
    Ok(())
}
