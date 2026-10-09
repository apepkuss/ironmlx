//! Paired in-process benchmark of ordinary decoding and DFlash2 modes on one
//! loaded Qwen3.6 MoE target. Every run emits one JSON line; analysis is done
//! offline. Execution profiles that are process-wide (the M5 profile) are
//! compared across separate invocations.
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::chat_template::Message;
use ironmlx_lm::models::dflash2::{DFlash2DraftModel, DFlash2Target};
use ironmlx_lm::models::Qwen35MoeModel;
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
    /// Modes: ordinary, linear, tree<N> (for example tree7). A `-generic`
    /// suffix (for example linear-generic) runs that mode with the M5
    /// expert-grouped route explicitly off, and a final `@b<N>` suffix (for
    /// example linear@b7 or linear-generic@b7) with a fixed draft budget N
    /// instead of the adaptive policy, for same-process A/B comparisons.
    #[arg(long, value_delimiter = ',', default_value = "ordinary,linear")]
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
    #[arg(long, default_value = "generic")]
    profile_label: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    m5_profile: bool,
    /// Instead of generation runs, time single target verifies per shape
    /// (linear Q1..Q8, tree chain/binary W2..W16) and draft proposals, and
    /// record routed-expert coverage.
    #[arg(long)]
    verify_curve: bool,
    #[arg(long, default_value_t = 12)]
    curve_reps: usize,
    /// Restrict the verify curve to these shape names.
    #[arg(long, value_delimiter = ',')]
    curve_shapes: Vec<String>,
    /// Time the routed-expert stage of one MoE layer in isolation.
    #[arg(long)]
    moe_microbench: bool,
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

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[allow(clippy::too_many_lines)]
fn verify_curve(
    target: &Qwen35MoeModel,
    draft: &DFlash2DraftModel,
    prompt_ids: &[u32],
    case: &str,
    args: &Args,
    out: &mut std::fs::File,
) -> Result<()> {
    use ironmlx_lm::core::cache::layer::LayerCache;
    use ironmlx_lm::core::model_input::build_position_ids;
    use ironmlx_lm::models::dflash2::DFlash2TargetForwardMode as Mode;
    use ironmlx_lm::models::qwen3_5_moe::sparse_moe::route_diagnostic;
    use ironmlx_lm::Model;
    use mlx::Array;

    let taps = draft.config().dflash_config.target_layer_ids.clone();
    let cap = (prompt_ids.len() + 64) as i32;
    let mut cache = target.make_cache(1, cap, target.cache_dtype())?;
    let ids = |tokens: &[u32]| -> Result<Array> {
        Ok((tokens, &[1_i32, tokens.len() as i32][..]).try_into()?)
    };
    let prefill = target.dflash2_forward_target_on(
        &ids(prompt_ids)?,
        &build_position_ids(0, prompt_ids.len() as i32)?,
        Some(&mut cache),
        &taps,
        Mode::Prefill,
        ().into(),
    )?;
    mlx::transforms::eval(&[&prefill.hidden, &prefill.context_hidden])?;
    let base: Vec<_> = cache.iter().map(LayerCache::snapshot).collect();
    // Realistic verify tokens: the greedy continuation of the prompt.
    let mut continuation = Vec::new();
    let mut last = {
        let dims = prefill.hidden.shape();
        let last = mlx::ops::indexing::slice_strided(
            &prefill.hidden,
            &[0_i32, dims.as_slice()[1] - 1, 0][..],
            &[1_i32, dims.as_slice()[1], dims.as_slice()[2]][..],
            &[1_i32, 1, 1][..],
        )?;
        target.dflash2_project_hidden_on(&last, ().into())?
    };
    for step in 0..17 {
        let token = mlx::ops::reduction::argmax(&last, -1, false)?
            .reshape((-1,))?
            .to_vec::<u32>()?[0];
        continuation.push(token);
        let output = target.dflash2_forward_target_on(
            &ids(&[token])?,
            &build_position_ids((prompt_ids.len() + step) as i32, 1)?,
            Some(&mut cache),
            &taps,
            Mode::OrdinaryDecode,
            ().into(),
        )?;
        last = target.dflash2_project_hidden_on(&output.hidden, ().into())?;
    }
    let start = prompt_ids.len() as i32;
    let restore = |cache: &mut Vec<LayerCache>| -> Result<()> {
        for (layer, snapshot) in cache.iter_mut().zip(&base) {
            layer.discard_speculative_prefix_capture();
            layer.restore(snapshot)?;
        }
        Ok(())
    };

    let mut shapes: Vec<(String, Vec<i32>)> = Vec::new();
    for width in 1..=8 {
        shapes.push((format!("linear-q{width}"), Vec::new()));
    }
    // Timing reference only: the ordinary multi-token graph without the
    // row-exact verify routes (not used for DFlash2 output).
    for width in [2_usize, 4, 8, 16] {
        shapes.push((format!("plain-q{width}"), vec![-2]));
    }
    // Timing attribution only: the same verifies with one component's
    // output replaced by zeros (see `timing_ablation`).
    for width in [1_usize, 8] {
        for ablation in ["noexperts", "noshared", "noattn", "nogdn", "noall"] {
            shapes.push((format!("{ablation}-q{width}"), vec![-3]));
        }
    }
    for ablation in ["noexperts", "noshared", "noattn", "nogdn", "noall"] {
        shapes.push((format!("{ablation}-tree-binary-w16"), vec![-4]));
    }
    for width in 2_i32..=16 {
        shapes.push((
            format!("tree-chain-w{width}"),
            (0..width).map(|node| node - 1).collect(),
        ));
        shapes.push((
            format!("tree-binary-w{width}"),
            (0..width)
                .map(|node| if node == 0 { -1 } else { (node - 1) / 2 })
                .collect(),
        ));
    }
    for (name, parents) in shapes {
        if !args.curve_shapes.is_empty() && !args.curve_shapes.contains(&name) {
            continue;
        }
        use ironmlx_lm::models::qwen3_5_moe::timing_ablation as ablation;
        let plain = parents == [-2];
        let ablation_flags = if parents == [-3] || parents == [-4] {
            match name.split('-').next().unwrap_or("") {
                "noexperts" => ablation::ROUTED_EXPERTS,
                "noshared" => ablation::SHARED_EXPERT,
                "noattn" => ablation::FULL_ATTENTION,
                "nogdn" => ablation::LINEAR_ATTENTION,
                "noall" => {
                    ablation::ROUTED_EXPERTS
                        | ablation::SHARED_EXPERT
                        | ablation::FULL_ATTENTION
                        | ablation::LINEAR_ATTENTION
                }
                _ => 0,
            }
        } else {
            0
        };
        let parents = if parents == [-4] {
            (0..16)
                .map(|node: i32| if node == 0 { -1 } else { (node - 1) / 2 })
                .collect()
        } else if parents == [-3] {
            Vec::new()
        } else {
            parents
        };
        let width = if parents.is_empty() && !plain {
            name.rsplit("-q").next().unwrap_or("").parse::<usize>()?
        } else if plain {
            name.trim_start_matches("plain-q").parse::<usize>()?
        } else {
            parents.len()
        };
        ablation::set(ablation_flags);
        let tokens = &continuation[..width];
        let mut timings = Vec::new();
        let mut coverage = None;
        for rep in 0..args.curve_reps {
            restore(&mut cache)?;
            for layer in cache.iter_mut() {
                layer.begin_speculative_prefix_capture()?;
            }
            if rep == 1 {
                route_diagnostic::start();
            }
            mlx::transforms::synchronize()?;
            let started = Instant::now();
            let output = if plain {
                let hidden = Model::forward_text_hidden(
                    target,
                    &ids(tokens)?,
                    &build_position_ids(start, width as i32)?,
                    None,
                    None,
                    Some(&mut cache),
                    ().into(),
                )?;
                ironmlx_lm::models::dflash2::DFlash2TargetOutput {
                    context_hidden: hidden.clone(),
                    hidden,
                }
            } else if parents.is_empty() {
                target.dflash2_forward_target_on(
                    &ids(tokens)?,
                    &build_position_ids(start, width as i32)?,
                    Some(&mut cache),
                    &taps,
                    if width == 1 {
                        Mode::OrdinaryDecode
                    } else {
                        Mode::GreedyVerify
                    },
                    ().into(),
                )?
            } else {
                target.dflash2_forward_tree_on(
                    &ids(tokens)?,
                    &parents,
                    start,
                    &mut cache,
                    &taps,
                    ().into(),
                )?
            };
            let logits = target.dflash2_project_hidden_on(&output.hidden, ().into())?;
            mlx::transforms::eval(&[&logits, &output.context_hidden])?;
            mlx::transforms::synchronize()?;
            let elapsed = started.elapsed().as_secs_f64() * 1e3;
            if rep == 1 {
                let routes = route_diagnostic::take()?;
                let per_layer = routes
                    .iter()
                    .map(|(_, experts)| {
                        experts
                            .iter()
                            .collect::<std::collections::BTreeSet<_>>()
                            .len()
                    })
                    .collect::<Vec<_>>();
                let mean = per_layer.iter().sum::<usize>() as f64 / per_layer.len().max(1) as f64;
                coverage = Some(json!({
                    "layers": per_layer.len(),
                    "slots_per_layer": width * 8,
                    "unique_experts_mean": mean,
                    "unique_experts_max": per_layer.iter().max(),
                    "unique_experts_min": per_layer.iter().min(),
                }));
            }
            if rep >= 2 {
                timings.push(elapsed);
            }
        }
        ablation::set(0);
        restore(&mut cache)?;
        writeln!(
            out,
            "{}",
            json!({
                "kind": "verify_curve",
                "profile_label": args.profile_label,
                "prompt": case,
                "context": prompt_ids.len(),
                "shape": name,
                "width": width,
                "median_ms": median(timings.clone()),
                "min_ms": timings.iter().copied().fold(f64::INFINITY, f64::min),
                "samples": timings.len(),
                "coverage": coverage,
            })
        )?;
    }

    if !args.curve_shapes.is_empty() {
        return Ok(());
    }
    // Draft proposal cost per block length and per tree budget.
    let mask = draft.config().dflash_config.mask_token_id;
    for length in 2..=args.block_size {
        let mut timings = Vec::new();
        for rep in 0..args.curve_reps {
            let mut draft_cache = draft.make_cache(0)?;
            let mut block = vec![continuation[0]];
            block.resize(length, mask);
            mlx::transforms::synchronize()?;
            let started = Instant::now();
            let proposal = draft.propose_greedy_on(
                target,
                &ids(&block)?,
                &prefill.context_hidden,
                &mut draft_cache,
                (),
            )?;
            mlx::transforms::eval(&[&proposal])?;
            if rep >= 2 {
                timings.push(started.elapsed().as_secs_f64() * 1e3);
            }
        }
        writeln!(
            out,
            "{}",
            json!({"kind": "draft_curve", "profile_label": args.profile_label, "prompt": case,
                   "shape": format!("linear-l{length}"), "median_ms": median(timings)})
        )?;
    }
    for nodes in [7_usize, 15] {
        let mut timings = Vec::new();
        for rep in 0..args.curve_reps {
            let mut draft_cache = draft.make_cache(0)?;
            let mut block = vec![continuation[0]];
            block.resize(args.block_size, mask);
            mlx::transforms::synchronize()?;
            let started = Instant::now();
            let tree = draft.propose_tree_on(
                target,
                &ids(&block)?,
                &prefill.context_hidden,
                &mut draft_cache,
                ironmlx_lm::models::dflash2::DFlash2TreeSpec {
                    max_nodes: nodes,
                    children_per_node: 2,
                },
                (),
            )?;
            if rep >= 2 {
                timings.push(started.elapsed().as_secs_f64() * 1e3);
            }
            drop(tree);
        }
        writeln!(
            out,
            "{}",
            json!({"kind": "draft_curve", "profile_label": args.profile_label, "prompt": case,
                   "shape": format!("tree-n{nodes}"), "median_ms": median(timings)})
        )?;
    }
    Ok(())
}

/// Routed-expert stage of layer 10 timed through the production
/// `RoutedExperts::apply_experts`, with real routing captured from a real
/// verify of the same width, and with synthetic all-distinct and fully shared
/// routing to measure the sensitivity to expert reuse.
fn moe_microbench(
    target: &Qwen35MoeModel,
    draft: &DFlash2DraftModel,
    prompt_ids: &[u32],
    args: &Args,
    out: &mut std::fs::File,
) -> Result<()> {
    use ironmlx_lm::core::model_input::build_position_ids;
    use ironmlx_lm::models::dflash2::DFlash2TargetForwardMode as Mode;
    use ironmlx_lm::models::qwen3_5_moe::sparse_moe::route_diagnostic;
    use ironmlx_lm::models::qwen3_5_moe::SparseMoeBlock;
    use ironmlx_lm::Model;
    use mlx::{Array, Dtype};

    const LAYER: i32 = 10;
    let loader = Loader::open(&args.target)?;
    let block = SparseMoeBlock::from_loader(
        &loader,
        &format!("model.layers.{LAYER}.mlp"),
        target.config().num_experts_per_tok,
        target.config().norm_topk_prob,
    )?;
    let routed = block.routed();
    let taps = draft.config().dflash_config.target_layer_ids.clone();
    let hidden = target.config().hidden_size;
    // Widths the target certifies for a B1 linear verify (Qwen3.6 MoE: up to
    // Q8); wider shapes need the tree interface and are not measured here.
    let capabilities = target.dflash2_verify_capabilities();
    let widths: Vec<i32> = [1_usize, 2, 4, 8, 16]
        .into_iter()
        .filter(|&width| capabilities.supports(1, width) && width <= prompt_ids.len())
        .map(|width| width as i32)
        .collect();
    eprintln!("moe microbench widths {widths:?} (B1 linear verify shapes of this target)");
    for width in widths {
        // Target routing for `width` tokens fed after the prompt. The tokens
        // are the prompt's first `width` tokens (a teacher-forced input), not
        // a generated continuation.
        let mut cache = target.make_cache(1, prompt_ids.len() as i32 + 32, target.cache_dtype())?;
        let ids: Array = (prompt_ids, &[1_i32, prompt_ids.len() as i32][..]).try_into()?;
        target.dflash2_forward_target_on(
            &ids,
            &build_position_ids(0, prompt_ids.len() as i32)?,
            Some(&mut cache),
            &taps,
            Mode::Prefill,
            ().into(),
        )?;
        let tokens = prompt_ids[..width as usize].to_vec();
        route_diagnostic::start();
        let verify: Array = (&tokens[..], &[1_i32, width][..]).try_into()?;
        let output = target.dflash2_forward_target_on(
            &verify,
            &build_position_ids(prompt_ids.len() as i32, width)?,
            Some(&mut cache),
            &taps,
            if width == 1 {
                Mode::OrdinaryDecode
            } else {
                Mode::GreedyVerify
            },
            ().into(),
        )?;
        mlx::transforms::eval(&[&output.hidden])?;
        let routes = route_diagnostic::take()?;
        let real = routes
            .iter()
            .find(|(layer, _)| *layer == LAYER)
            .map(|(_, experts)| experts.clone())
            .ok_or_else(|| anyhow::anyhow!("no routes for layer {LAYER}"))?;
        let slots = (width * 8) as usize;
        let distinct = (0..slots as u32).map(|slot| slot % 256).collect::<Vec<_>>();
        let shared = (0..slots as u32).map(|slot| slot % 8).collect::<Vec<_>>();
        let key = mlx::random::key(7)?;
        let x = mlx::random::normal()
            .shape((width, hidden))
            .dtype(Dtype::Bfloat16)
            .key(&key)
            .sample()?;
        let scores = mlx::random::uniform()
            .shape((width, 8_i32))
            .dtype(Dtype::Float32)
            .key(&key)
            .sample()?;
        for (label, experts) in [("real", real), ("distinct", distinct), ("shared8", shared)] {
            let unique = experts
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let inds: Array = (&experts[..], &[width, 8_i32][..]).try_into()?;
            let mut timings = Vec::new();
            for rep in 0..args.curve_reps {
                mlx::transforms::synchronize()?;
                let started = Instant::now();
                let y = routed.apply_experts(&x, &inds, &scores, ().into(), LAYER)?;
                mlx::transforms::eval(&[&y])?;
                mlx::transforms::synchronize()?;
                if rep >= 2 {
                    timings.push(started.elapsed().as_secs_f64() * 1e3);
                }
            }
            let line = json!({
                "kind": "moe_microbench",
                "profile_label": args.profile_label,
                "layer": LAYER,
                "width": width,
                "routing": label,
                "real_routing_tokens": "prompt_prefix_after_prompt",
                "unique_experts": unique,
                "median_ms": median(timings.clone()),
                "min_ms": timings.iter().copied().fold(f64::INFINITY, f64::min),
            });
            writeln!(out, "{line}")?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let profile = ironmlx_core::m5_profile::install_for_target(
        if args.m5_profile {
            ironmlx_core::m5_profile::M5ProfileMode::Auto
        } else {
            ironmlx_core::m5_profile::M5ProfileMode::Off
        },
        true,
        ironmlx_core::m5_profile::M5ProfileTarget::Qwen36Moe,
    );
    let load_started = Instant::now();
    let loader = Loader::open(&args.target).context("open target")?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let target = Qwen35MoeModel::from_loader(&loader).context("load MoE target")?;
    drop(loader);
    let target_load_s = load_started.elapsed().as_secs_f64();
    let draft_started = Instant::now();
    let draft_loader = Loader::open_dflash2(&args.draft)?;
    let draft = DFlash2DraftModel::from_loader(&draft_loader, target.dflash2_target_spec(), None)?;
    drop(draft_loader);
    let draft_load_s = draft_started.elapsed().as_secs_f64();
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
    if args.moe_microbench {
        return moe_microbench(&target, &draft, &encoded[0], &args, &mut out);
    }
    if args.verify_curve {
        let setup = json!({
            "kind": "setup",
            "profile_label": args.profile_label,
            "m5_profile": profile.status.as_str(),
            "target_bits": target.dflash2_target_bits(),
            "fingerprint": target.dflash2_execution_fingerprint(),
        });
        writeln!(out, "{setup}")?;
        let cases = if args.curve_shapes.is_empty() { 2 } else { 1 };
        for (case, prompt_ids) in prompts.iter().zip(&encoded).take(cases) {
            verify_curve(&target, &draft, prompt_ids, &case.id, &args, &mut out)?;
        }
        return Ok(());
    }
    writeln!(
        out,
        "{}",
        json!({
            "kind": "setup",
            "profile_label": args.profile_label,
            "m5_profile": profile.status.as_str(),
            "target": args.target,
            "target_bits": target.dflash2_target_bits(),
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
    // Process-level values of the settings a mode suffix may override.
    let grouped_setting =
        std::env::var(ironmlx_core::m5_profile::settings::M5_MOE_GROUPED_QMV).ok();
    let budget_setting =
        std::env::var(ironmlx_core::m5_profile::settings::DFLASH2_FIXED_BUDGET).ok();
    let restore = |name: &str, value: &Option<String>| match value {
        Some(value) => std::env::set_var(name, value),
        None => std::env::remove_var(name),
    };
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
                // An explicit setting overrides the profile value and is read
                // at every target forward; the bench is single-request.
                let (mode_name, fixed_budget) = match mode.split_once("@b") {
                    Some((base, budget)) => (base, Some(budget)),
                    None => (mode.as_str(), None),
                };
                match fixed_budget {
                    Some(budget) => std::env::set_var(
                        ironmlx_core::m5_profile::settings::DFLASH2_FIXED_BUDGET,
                        budget,
                    ),
                    None => restore(
                        ironmlx_core::m5_profile::settings::DFLASH2_FIXED_BUDGET,
                        &budget_setting,
                    ),
                }
                let (base_mode, generic) = match mode_name.strip_suffix("-generic") {
                    Some(base) => (base, true),
                    None => (mode_name, false),
                };
                if generic {
                    std::env::set_var(ironmlx_core::m5_profile::settings::M5_MOE_GROUPED_QMV, "0");
                } else {
                    restore(
                        ironmlx_core::m5_profile::settings::M5_MOE_GROUPED_QMV,
                        &grouped_setting,
                    );
                }
                let grouped_before = ironmlx_lm::nn::moe_grouped_qmv::dispatch_count();
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
                        "ttft_s": ttft,
                        "e2e_s": e2e,
                        "decode_tps": decode_tps,
                        "footprint": ironmlx_runtime::core::process_memory::macos_phys_footprint_bytes(),
                        "mlx_peak": mlx::memory::snapshot().peak_bytes,
                        "metrics": metrics,
                        "grouped_expert_dispatches":
                            ironmlx_lm::nn::moe_grouped_qmv::dispatch_count() - grouped_before,
                    })
                )?;
                out.flush()?;
            }
        }
    }
    Ok(())
}
