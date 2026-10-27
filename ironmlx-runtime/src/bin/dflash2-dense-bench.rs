//! Paired in-process benchmark of DFlash2 modes on one loaded Qwen3.8 Dense
//! target (the dense counterpart of `dflash2-moe-bench`). Every run emits one
//! JSON line with the generated token ids, timings, the stream metrics and,
//! with `IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES=1`, the per-window stage
//! records; analysis is done offline. Process-wide settings (the M5 profile
//! and its experimental overrides) are compared across invocations.
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::chat_template::Message;
use ironmlx_lm::models::dflash2::{DFlash2DraftModel, DFlash2Target};
use ironmlx_lm::models::Qwen35Model;
use ironmlx_lm::{Loader, Tokenizer};
use ironmlx_runtime::core::dflash2::{DFlash2P2Options, DFlash2TextGenerationStream};
use ironmlx_runtime::core::generate::GenerationStream;
use ironmlx_runtime::core::generation_types::{GenerateEvent, GenerateRequest};
use serde_json::json;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    target: PathBuf,
    #[arg(long)]
    draft: PathBuf,
    /// JSON list of {"id", "prompt"} objects.
    #[arg(long)]
    prompts: PathBuf,
    /// Modes: ordinary, linear, tree<N> (for example tree15, the served Dense
    /// default). A final `@b<N>` suffix runs that mode with a fixed draft
    /// budget N.
    #[arg(long, value_delimiter = ',', default_value = "tree15")]
    modes: Vec<String>,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 8)]
    block_size: usize,
    /// Label written into every record (for example the process profile).
    #[arg(long, default_value = "dense")]
    profile_label: String,
    #[arg(long)]
    output: PathBuf,
    /// Install the M5 Dense profile (as `serve` does on M5 hardware).
    #[arg(long)]
    m5_profile: bool,
    /// Wire the loaded weights and refresh their residency (as `serve` does
    /// when the profile sets WIRED_WEIGHTS).
    #[arg(long)]
    wired: bool,
    /// Drafter runtime quantization bits (0 = BF16 checkpoint). The served
    /// Dense profile loads the drafter at 4 bits.
    #[arg(long, default_value_t = 4)]
    draft_bits: i32,
}

#[derive(serde::Deserialize)]
struct PromptCase {
    id: String,
    prompt: String,
}

fn run_stream(
    mut next: impl FnMut() -> Result<Option<GenerateEvent>>,
    started: Instant,
) -> Result<(Vec<u32>, f64, f64, Option<String>)> {
    let mut tokens = Vec::new();
    let mut first = None;
    let mut finish = None;
    while let Some(event) = next()? {
        if first.is_none() {
            first = Some(started.elapsed().as_secs_f64());
        }
        tokens.push(event.token);
        if let Some(reason) = event.finish_reason {
            finish = Some(format!("{reason:?}"));
            break;
        }
    }
    Ok((
        tokens,
        first.unwrap_or(0.0),
        started.elapsed().as_secs_f64(),
        finish,
    ))
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let args = Args::parse();
    let profile = ironmlx_core::m5_profile::install_for_target(
        if args.m5_profile {
            ironmlx_core::m5_profile::M5ProfileMode::Auto
        } else {
            ironmlx_core::m5_profile::M5ProfileMode::Off
        },
        true,
        ironmlx_core::m5_profile::M5ProfileTarget::Qwen35Dense,
    );
    let load_started = Instant::now();
    let mut loader = Loader::open(&args.target).context("open target")?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let target = Qwen35Model::from_loader_dflash2(&mut loader).context("load Dense target")?;
    drop(loader);
    let target_load_s = load_started.elapsed().as_secs_f64();
    let draft_started = Instant::now();
    let draft_loader = Loader::open_dflash2(&args.draft)?;
    let mut draft = DFlash2DraftModel::from_loader(
        &draft_loader,
        target.dflash2_target_spec(),
        (args.draft_bits != 0).then_some(args.draft_bits),
    )?;
    drop(draft_loader);
    draft.share_weight_layout_if_requested()?;
    target.dflash2_prewarm()?;
    let draft_load_s = draft_started.elapsed().as_secs_f64();
    mlx::clear_cache();
    if args.wired {
        let limit = mlx::memory::snapshot().active_bytes;
        mlx::memory::set_wired_limit(limit)?;
        mlx::memory::start_residency_refresh(1000)?;
    }
    let capabilities = target.dflash2_verify_capabilities();
    let prompts: Vec<PromptCase> = serde_json::from_reader(std::fs::File::open(&args.prompts)?)?;
    let encoded = prompts
        .iter()
        .map(|case| {
            let messages = vec![Message {
                role: "user".into(),
                content: case.prompt.clone(),
            }];
            let text = tokenizer.apply_chat_template(
                &messages,
                true,
                Some(&json!({"enable_thinking": false})),
            )?;
            tokenizer.encode(&text, false)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut out = std::fs::File::create(&args.output)?;
    writeln!(
        out,
        "{}",
        json!({
            "kind": "setup",
            "profile_label": args.profile_label,
            "m5_profile": profile.status.as_str(),
            "target": args.target,
            "capabilities": capabilities.stable_fingerprint(),
            "fingerprint": target.dflash2_execution_fingerprint(),
            "target_load_s": target_load_s,
            "draft_load_s": draft_load_s,
            "footprint_after_load": ironmlx_runtime::core::process_memory::macos_phys_footprint_bytes(),
            "mlx_after_load": format!("{:?}", mlx::memory::snapshot()),
        })
    )?;

    let request = |prompt_ids: Vec<u32>| GenerateRequest {
        priority: Default::default(),
        prompt_ids,
        max_new_tokens: args.max_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: Vec::new(),
        prefill_chunk_size: 2048,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248_056,
        constraint: None,
    };
    let eos = tokenizer.eos_token_ids().to_vec();
    let budget_setting =
        std::env::var(ironmlx_core::m5_profile::settings::DFLASH2_FIXED_BUDGET).ok();
    let mut order_seed = 0x9e37_79b9_7f4a_7c15_u64;
    for round in 0..args.warmup + args.rounds {
        for (case, prompt_ids) in prompts.iter().zip(&encoded) {
            // Shuffle the mode order per (round, prompt) so modes are paired
            // without a fixed position bias.
            let mut modes = args.modes.clone();
            for index in (1..modes.len()).rev() {
                order_seed ^= order_seed << 13;
                order_seed ^= order_seed >> 7;
                order_seed ^= order_seed << 17;
                modes.swap(index, (order_seed % (index as u64 + 1)) as usize);
            }
            for mode in modes {
                mlx::clear_cache();
                let (base_mode, fixed_budget) = match mode.split_once("@b") {
                    Some((base, budget)) => (base, Some(budget.to_string())),
                    None => (mode.as_str(), None),
                };
                match fixed_budget.or_else(|| budget_setting.clone()) {
                    Some(budget) => std::env::set_var(
                        ironmlx_core::m5_profile::settings::DFLASH2_FIXED_BUDGET,
                        budget,
                    ),
                    None => std::env::remove_var(
                        ironmlx_core::m5_profile::settings::DFLASH2_FIXED_BUDGET,
                    ),
                }
                let started = Instant::now();
                let (tokens, ttft, e2e, finish, metrics) = if base_mode == "ordinary" {
                    let mut stream = GenerationStream::new_text_only(
                        &target,
                        &tokenizer,
                        request(prompt_ids.clone()),
                    )?;
                    let (tokens, ttft, e2e, finish) = run_stream(|| stream.next_token(), started)?;
                    (tokens, ttft, e2e, finish, None)
                } else {
                    let tree_max_nodes = base_mode
                        .strip_prefix("tree")
                        .map(str::parse::<usize>)
                        .transpose()?
                        .unwrap_or(0);
                    let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
                        &target,
                        &draft,
                        &tokenizer,
                        request(prompt_ids.clone()),
                        args.block_size,
                        DFlash2P2Options {
                            tree_max_nodes,
                            position_keyed_sampling: false,
                        },
                    )?;
                    let (tokens, ttft, e2e, finish) = run_stream(|| stream.next_token(), started)?;
                    (
                        tokens,
                        ttft,
                        e2e,
                        finish,
                        Some(serde_json::to_value(stream.metrics())?),
                    )
                };
                let generated = tokens.len();
                let decode_tps = if generated > 1 && e2e > ttft {
                    (generated - 1) as f64 / (e2e - ttft)
                } else {
                    0.0
                };
                let natural_eos = tokens.iter().position(|token| eos.contains(token));
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(&tokens, &mut hasher);
                writeln!(
                    out,
                    "{}",
                    json!({
                        "kind": "run",
                        "profile_label": args.profile_label,
                        "warmup": round < args.warmup,
                        "round": round,
                        "prompt": case.id,
                        "prompt_tokens": prompt_ids.len(),
                        "mode": mode,
                        "generated_tokens": generated,
                        "finish": finish,
                        "natural_eos_index": natural_eos,
                        "token_hash": std::hash::Hasher::finish(&hasher),
                        "tokens": tokens,
                        "ttft_s": ttft,
                        "e2e_s": e2e,
                        "decode_tps": decode_tps,
                        "footprint": ironmlx_runtime::core::process_memory::macos_phys_footprint_bytes(),
                        "mlx_peak": mlx::memory::snapshot().peak_bytes,
                        "metrics": metrics,
                        "window_stages": ironmlx_runtime::core::dflash2::take_window_stage_records(),
                    })
                )?;
                out.flush()?;
            }
        }
    }
    Ok(())
}
