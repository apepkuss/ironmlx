//! Interleaved B1 Q8/Q16 qualification for a local Qwen3.8 DFlash2 pair.
//!
//! The runner never resolves Hub ids and never downloads artifacts. Both
//! checkpoint arguments must name existing directories below the local
//! IronMLX Hugging Face model root.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::loader::Loader;
use ironmlx_lm::core::tokenizer::Tokenizer;
use ironmlx_lm::models::{dflash2::DFlash2DraftModel, Qwen35Model};
use ironmlx_runtime::core::dflash2::{
    DFlash2Metrics, DFlash2P2Options, DFlash2TextGenerationStream,
};
use ironmlx_runtime::core::generation_types::GenerateRequest;
use serde::Serialize;

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Greedy,
    Sampled,
}

#[derive(Debug, Parser)]
#[command(about = "Run an interleaved local-only DFlash2 Q8/Q16 B1 qualification")]
struct Args {
    #[arg(long)]
    target_dir: PathBuf,
    #[arg(long)]
    draft_dir: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "2048,8192,32768", value_delimiter = ',')]
    context_lengths: Vec<usize>,
    #[arg(long, default_value = "greedy,sampled", value_delimiter = ',')]
    modes: Vec<Mode>,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 1)]
    warmup_pairs: usize,
    #[arg(long, default_value_t = 7)]
    measured_pairs: usize,
    #[arg(long, default_value_t = 20_260_928)]
    seed: u64,
    #[arg(long, default_value = "ff2fcea50a49ed5c9213c70ddcdbc1decc605655")]
    frozen_p2_sha: String,
}

#[derive(Debug, Serialize)]
struct RunRecord {
    mode: Mode,
    context_tokens: usize,
    replicate: usize,
    block_size: usize,
    order_in_pair: usize,
    wall_total_us: u64,
    token_ids: Vec<u32>,
    metrics: DFlash2MetricsRecord,
}

#[derive(Debug, Serialize)]
struct DFlash2MetricsRecord {
    generated_tokens: usize,
    windows: usize,
    drafted_tokens: usize,
    accepted_draft_tokens: usize,
    rollback_count: usize,
    current_draft_budget: usize,
    exact_sampling_windows: usize,
    exact_acceptance_draws: usize,
    exact_residual_corrections: usize,
    exact_bonus_samples: usize,
    draft_build_us: u64,
    draft_schedule_us: u64,
    verify_build_us: u64,
    projection_build_us: u64,
    sampling_us: u64,
    verify_schedule_us: u64,
    host_sync_us: u64,
    rollback_us: u64,
    window_us: u64,
    prefill_us: u64,
    generation_us: u64,
    prompt_tps: f64,
    generation_tps: f64,
    acceptance_rate: f64,
    peak_memory_bytes: usize,
}

impl From<DFlash2Metrics> for DFlash2MetricsRecord {
    fn from(value: DFlash2Metrics) -> Self {
        Self {
            generated_tokens: value.generated_tokens,
            windows: value.windows,
            drafted_tokens: value.drafted_tokens,
            accepted_draft_tokens: value.accepted_draft_tokens,
            rollback_count: value.rollback_count,
            current_draft_budget: value.current_draft_budget,
            exact_sampling_windows: value.exact_sampling_windows,
            exact_acceptance_draws: value.exact_acceptance_draws,
            exact_residual_corrections: value.exact_residual_corrections,
            exact_bonus_samples: value.exact_bonus_samples,
            draft_build_us: value.draft_build_us,
            draft_schedule_us: value.draft_schedule_us,
            verify_build_us: value.verify_build_us,
            projection_build_us: value.projection_build_us,
            sampling_us: value.sampling_us,
            verify_schedule_us: value.verify_schedule_us,
            host_sync_us: value.host_sync_us,
            rollback_us: value.rollback_us,
            window_us: value.window_us,
            prefill_us: value.prefill_us,
            generation_us: value.generation_us,
            prompt_tps: value.prompt_tps,
            generation_tps: value.generation_tps,
            acceptance_rate: value.acceptance_rate,
            peak_memory_bytes: value.peak_memory_bytes,
        }
    }
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    frozen_p2_sha: String,
    execution_scope: &'static str,
    target_dir: String,
    draft_dir: String,
    draft_checkpoint_block_size: usize,
    max_new_tokens: usize,
    warmup_pairs: usize,
    measured_pairs: usize,
    seed: u64,
    records: Vec<RunRecord>,
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
    debug_assert_eq!(tokens.len(), context_tokens);
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

#[allow(clippy::too_many_arguments)]
fn run_once(
    target: &Qwen35Model,
    draft: &DFlash2DraftModel,
    tokenizer: &Tokenizer,
    mode: Mode,
    context_tokens: usize,
    max_new_tokens: usize,
    seed: u64,
    block_size: usize,
    replicate: usize,
    order_in_pair: usize,
) -> Result<RunRecord> {
    let request = GenerateRequest {
        priority: Default::default(),
        prompt_ids: prompt_ids(context_tokens)?,
        max_new_tokens,
        sampler: sampler(mode, seed),
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
    let started = Instant::now();
    let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
        target,
        draft,
        tokenizer,
        request,
        block_size,
        DFlash2P2Options {
            tree_max_nodes: 0,
            position_keyed_sampling: matches!(mode, Mode::Sampled),
        },
    )?;
    let mut token_ids = Vec::with_capacity(max_new_tokens);
    while let Some(event) = stream.next_token()? {
        token_ids.push(event.token);
    }
    let wall_total_us = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
    let metrics = stream.metrics();
    anyhow::ensure!(
        token_ids.len() == max_new_tokens,
        "Q{block_size} {mode:?} context {context_tokens} emitted {} of {max_new_tokens} tokens",
        token_ids.len()
    );
    Ok(RunRecord {
        mode,
        context_tokens,
        replicate,
        block_size,
        order_in_pair,
        wall_total_us,
        token_ids,
        metrics: metrics.into(),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.max_new_tokens > 0, "--max-new-tokens must be positive");
    anyhow::ensure!(
        args.measured_pairs >= 5,
        "CI95 qualification needs at least five pairs"
    );
    anyhow::ensure!(
        !args.context_lengths.is_empty(),
        "no context lengths selected"
    );
    anyhow::ensure!(!args.modes.is_empty(), "no modes selected");
    let target_dir = validate_local_checkpoint(&args.target_dir, "target")?;
    let draft_dir = validate_local_checkpoint(&args.draft_dir, "draft")?;

    let mut target_loader = Loader::open(&target_dir).context("open target checkpoint")?;
    let tokenizer = Tokenizer::from_loader(&target_loader).context("load tokenizer")?;
    let target = Qwen35Model::from_loader_dflash2(&mut target_loader)
        .context("load Qwen3.8 DFlash2 target")?;
    let draft_loader = Loader::open_dflash2(&draft_dir).context("open draft checkpoint")?;
    let draft = DFlash2DraftModel::from_loader(&draft_loader, target.config(), Some(4))
        .context("load runtime-quantized DFlash2 draft")?;
    let checkpoint_block_size = usize::try_from(draft.config().dflash_config.block_size)
        .context("draft checkpoint block_size must be non-negative")?;
    anyhow::ensure!(
        checkpoint_block_size >= 16,
        "draft checkpoint block_size={checkpoint_block_size}; Q16 requires at least 16"
    );

    let mut records = Vec::new();
    for &mode in &args.modes {
        for &context_tokens in &args.context_lengths {
            let total_pairs = args.warmup_pairs + args.measured_pairs;
            for pair in 0..total_pairs {
                let order = if pair % 2 == 0 { [8, 16] } else { [16, 8] };
                let measured = pair >= args.warmup_pairs;
                let replicate = pair.saturating_sub(args.warmup_pairs);
                let pair_seed = args.seed ^ ((context_tokens as u64) << 16) ^ replicate as u64;
                let mut pair_records = Vec::with_capacity(2);
                for (order_in_pair, block_size) in order.into_iter().enumerate() {
                    let record = run_once(
                        &target,
                        &draft,
                        &tokenizer,
                        mode,
                        context_tokens,
                        args.max_new_tokens,
                        pair_seed,
                        block_size,
                        replicate,
                        order_in_pair,
                    )?;
                    eprintln!(
                        "{:?} C{} pair {}/{} Q{}: wall={:.3}s decode={:.3}s tps={:.3} accept={:.3}",
                        mode,
                        context_tokens,
                        pair + 1,
                        total_pairs,
                        block_size,
                        record.wall_total_us as f64 / 1_000_000.0,
                        record.metrics.generation_us as f64 / 1_000_000.0,
                        record.metrics.generation_tps,
                        record.metrics.acceptance_rate,
                    );
                    pair_records.push(record);
                }
                anyhow::ensure!(
                    pair_records[0].token_ids == pair_records[1].token_ids,
                    "Q8/Q16 token mismatch for {mode:?} context {context_tokens} pair {pair}"
                );
                if measured {
                    records.extend(pair_records);
                }
                mlx::clear_cache();
            }
        }
    }

    let report = Report {
        schema_version: SCHEMA_VERSION,
        frozen_p2_sha: args.frozen_p2_sha,
        execution_scope: "B1 linear DFlash2; frozen P2 direct stream path; Q8/Q16 interleaved",
        target_dir: target_dir.display().to_string(),
        draft_dir: draft_dir.display().to_string(),
        draft_checkpoint_block_size: checkpoint_block_size,
        max_new_tokens: args.max_new_tokens,
        warmup_pairs: args.warmup_pairs,
        measured_pairs: args.measured_pairs,
        seed: args.seed,
        records,
    };
    let encoded = serde_json::to_vec_pretty(&report).context("serialize qualification report")?;
    std::fs::write(&args.output, encoded)
        .with_context(|| format!("write {}", args.output.display()))?;
    Ok(())
}
