//! Diagnostic driver: DFlash2 ragged linear batching of several greedy
//! Qwen3.6 MoE requests at different context lengths, against the same
//! requests decoded one at a time. Copy of `dflash2-ragged-batch-prototype`
//! for the MoE target (M5 MoE profile installed as in serving). Modes:
//! - baseline: each request alone with the given tree setting (frozen B uses 15);
//! - ragged: requests decoded together with batched linear windows (width 8),
//!   optional staggered joins and a cancellation;
//! - draft-bench: batched draft cost for identical rows (equal positions).
use anyhow::{ensure, Result};
use clap::Parser;
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::{
    models::{
        dflash2::{DFlash2DraftModel, DFlash2Target},
        Qwen35MoeModel,
    },
    Loader, Tokenizer,
};
use ironmlx_runtime::core::{
    dflash2::{
        take_window_stage_records, DFlash2P2Options, DFlash2RaggedBatchCache,
        DFlash2TextGenerationStream,
    },
    generation_types::GenerateRequest,
};
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    target: PathBuf,
    #[arg(long)]
    draft: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    prompts: PathBuf,
    /// baseline | ragged | draft-bench
    #[arg(long)]
    mode: String,
    #[arg(long, value_delimiter = ',')]
    ids: Vec<String>,
    #[arg(long, default_value_t = 0)]
    tree_nodes: usize,
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,
    /// Per row: number of single-request windows before joining the batch.
    #[arg(long, value_delimiter = ',')]
    stagger: Vec<usize>,
    /// Cancel row index R after W batched windows: "R:W".
    #[arg(long)]
    cancel: Option<String>,
    #[arg(long, default_value = "1,2,4", value_delimiter = ',')]
    widths: Vec<usize>,
    #[arg(long, default_value_t = 20)]
    iters: usize,
    /// Parity mode: from this many ready rows on, decode one token per row
    /// per ragged window (0 = never).
    #[arg(long, default_value_t = 0)]
    ordinary_min_width: usize,
    /// Parity mode with relaxed batched arithmetic: instead of bit equality,
    /// teacher-force every ragged row through exact B1 decode and report each
    /// position where the emitted token is not the reference argmax.
    #[arg(long)]
    relaxed_check: bool,
}

fn request(tokenizer: &Tokenizer, prompt: &Value, max_tokens: usize) -> Result<GenerateRequest> {
    let text = tokenizer.apply_chat_template(
        &[json!({"role":"user","content":prompt["prompt"]})],
        true,
        Some(&json!({"enable_thinking":false})),
    )?;
    Ok(GenerateRequest {
        priority: Default::default(),
        prompt_ids: tokenizer.encode(&text, false)?,
        max_new_tokens: max_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: 0,
        decode_cadence_mid_chunk_cap: 1,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248_056,
        constraint: None,
    })
}

/// MLX memory after the target load and now, the M5 tiled copies prepared,
/// and the shared-layout conversion counters.
fn memory_report(after_target_load: &mlx::memory::MemorySnapshot) -> Value {
    let now = mlx::memory::snapshot();
    let (prepared_count, prepared_bytes) = ironmlx_lm::nn::m5_prepared_totals();
    let (stores, to_tiled, to_native) = ironmlx_lm::nn::shared_weight_layout_totals();
    json!({"after_target_load_active": after_target_load.active_bytes,
        "end_active": now.active_bytes, "peak": now.peak_bytes, "cache": now.cache_bytes,
        "m5_prepared_count": prepared_count, "m5_prepared_bytes": prepared_bytes,
        "shared_stores": stores, "shared_to_tiled": to_tiled, "shared_to_native": to_native})
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Timing attribution only: zero the selected components (outputs invalid).
    if let Some(flags) = std::env::var("IRONMLX_DIAGNOSTIC_TIMING_ABLATION")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    {
        ironmlx_lm::models::qwen3_5_moe::timing_ablation::set(flags);
    }
    ensure!(
        !args.output.exists(),
        "preserve existing report; choose a new output"
    );
    if let Ok(path) = std::env::var("IRONMLX_DIAGNOSTIC_METALLIB") {
        mlx::metal::set_metallib_path(&path)?;
    }
    let profile = ironmlx_core::m5_profile::install_for_target(
        ironmlx_core::m5_profile::M5ProfileMode::Auto,
        true,
        ironmlx_core::m5_profile::M5ProfileTarget::Qwen36Moe,
    );
    let loader = Loader::open(&args.target)?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let target = Qwen35MoeModel::from_loader(&loader)?;
    let after_target_load = mlx::memory::snapshot();
    drop(loader);
    let loader = Loader::open_dflash2(&args.draft)?;
    let draft = DFlash2DraftModel::from_loader(&loader, target.dflash2_target_spec(), Some(4))?;
    drop(loader);
    target.dflash2_prewarm()?;
    let profile_status = profile.status.as_str().to_owned();
    let fixture: Vec<Value> = serde_json::from_slice(&std::fs::read(&args.prompts)?)?;
    let prompts = args
        .ids
        .iter()
        .map(|id| {
            fixture
                .iter()
                .find(|p| p["id"] == id.as_str())
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("unknown prompt id {id}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let options = DFlash2P2Options {
        tree_max_nodes: args.tree_nodes,
        ..Default::default()
    };
    let env: serde_json::Map<String, Value> = std::env::vars()
        .filter(|(k, _)| k.starts_with("IRONMLX_"))
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    let write = |body: Value| -> Result<()> {
        let mut body = body;
        body["env"] = Value::Object(env.clone());
        body["mode"] = json!(args.mode);
        body["tree_nodes"] = json!(args.tree_nodes);
        body["m5_profile"] = json!(profile_status);
        body["fingerprint"] = json!(target.dflash2_execution_fingerprint());
        std::fs::write(&args.output, serde_json::to_vec_pretty(&body)?)?;
        Ok(())
    };

    match args.mode.as_str() {
        "baseline" => {
            let mut records = Vec::new();
            for prompt in &prompts {
                let _ = take_window_stage_records();
                let started = Instant::now();
                let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
                    &target,
                    &draft,
                    &tokenizer,
                    request(&tokenizer, prompt, args.max_tokens)?,
                    8,
                    options,
                )?;
                let mut tokens = Vec::new();
                let mut stop = None;
                while let Some(event) = stream.next_token()? {
                    tokens.push(event.token);
                    stop = event.finish_reason.or(stop);
                }
                let wall_s = started.elapsed().as_secs_f64();
                println!("{} tokens={} wall={wall_s:.2}", prompt["id"], tokens.len());
                records.push(json!({"id": prompt["id"], "tokens": tokens, "text": tokenizer.decode(&tokens, true)?,
                    "stop": format!("{stop:?}"), "wall_s": wall_s, "metrics": stream.metrics(),
                    "windows": take_window_stage_records()}));
                write(json!({"records": records}))?;
            }
        }
        "draft-bench" => {
            let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
                &target,
                &draft,
                &tokenizer,
                request(&tokenizer, &prompts[0], args.max_tokens)?,
                8,
                options,
            )?;
            // Advance a few windows so the draft state is representative.
            for _ in 0..40 {
                if stream
                    .next_token()?
                    .is_none_or(|e| e.finish_reason.is_some())
                {
                    break;
                }
            }
            while !stream.diagnostic_needs_window() {
                stream.diagnostic_next_token_deferred()?;
            }
            let bench = stream.diagnostic_draft_batch_bench(&args.widths, args.iters)?;
            write(
                json!({"prompt": prompts[0]["id"], "position": stream.diagnostic_verify_start(),
                "draft_bench": bench.iter().map(|(w, m, s)| json!({"width": w, "median_us": m, "samples": s})).collect::<Vec<_>>()}),
            )?;
        }
        "ragged" => {
            ensure!(
                args.tree_nodes == 0,
                "ragged batching uses linear windows (--tree-nodes 0)"
            );
            let n = prompts.len();
            ensure!(n >= 2, "ragged mode needs at least two prompts");
            let stagger = if args.stagger.is_empty() {
                vec![0; n]
            } else {
                args.stagger.clone()
            };
            ensure!(stagger.len() == n, "--stagger needs one value per prompt");
            let cancel = args
                .cancel
                .as_deref()
                .map(|s| -> Result<(usize, usize)> {
                    let (r, w) = s
                        .split_once(':')
                        .ok_or_else(|| anyhow::anyhow!("--cancel R:W"))?;
                    Ok((r.parse()?, w.parse()?))
                })
                .transpose()?;
            let started = Instant::now();
            let mut streams = Vec::with_capacity(n);
            for prompt in &prompts {
                streams.push(DFlash2TextGenerationStream::new_text_only_with_options(
                    &target,
                    &draft,
                    &tokenizer,
                    request(&tokenizer, prompt, args.max_tokens)?,
                    8,
                    options,
                )?);
            }
            let prefill_s = started.elapsed().as_secs_f64();
            let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
            let mut stops: Vec<Option<&'static str>> = vec![None; n];
            let mut done = vec![false; n];
            let mut cancelled = vec![false; n];
            let mut b1_windows = vec![0_usize; n];
            let mut cache: Option<(Vec<usize>, DFlash2RaggedBatchCache)> = None;
            let mut windows = Vec::new();
            let mut regroups = Vec::new();
            let mut batched_windows = 0_usize;
            let ordinary_mode = args.ordinary_min_width >= 2;
            let loop_started = Instant::now();
            loop {
                for i in 0..n {
                    if done[i] {
                        continue;
                    }
                    while !streams[i].diagnostic_needs_window() {
                        match streams[i].diagnostic_next_token_deferred()? {
                            Some(event) => {
                                tokens[i].push(event.token);
                                if event.finish_reason.is_some() {
                                    stops[i] = event.finish_reason;
                                    done[i] = true;
                                    break;
                                }
                            }
                            None => {
                                done[i] = true;
                                break;
                            }
                        }
                    }
                }
                if let Some((row, after)) = cancel {
                    if batched_windows >= after && !done[row] {
                        done[row] = true;
                        cancelled[row] = true;
                    }
                }
                // Rows still in their staggered single-request phase.
                let mut solo = Vec::new();
                let mut ready = Vec::new();
                for i in 0..n {
                    if done[i] {
                        continue;
                    }
                    let eligible = if ordinary_mode {
                        streams[i].diagnostic_ragged_ordinary_eligible()
                    } else {
                        streams[i].diagnostic_ragged_eligible()
                    };
                    if b1_windows[i] < stagger[i] || !eligible {
                        solo.push(i);
                    } else {
                        ready.push(i);
                    }
                }
                if solo.is_empty() && ready.is_empty() {
                    break;
                }
                let min_rows = if ordinary_mode {
                    args.ordinary_min_width
                } else {
                    2
                };
                let wanted: Vec<usize> = if ready.len() >= min_rows {
                    ready.clone()
                } else {
                    Vec::new()
                };
                // Membership change: return the batched cache to its rows first.
                if let Some((members, _)) = cache.as_ref() {
                    if *members != wanted {
                        let (members, c) = cache.take().expect("cache present");
                        let scatter_started = Instant::now();
                        let mut picked = pick_rows(&mut streams, &members);
                        c.scatter_to_rows(&mut picked)?;
                        let stages = std::env::var("IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES")
                            .as_deref()
                            == Ok("1");
                        if stages {
                            for row in picked.iter() {
                                row.diagnostic_eval_target_cache()?;
                            }
                        }
                        mlx::transforms::synchronize()?;
                        regroups.push(json!({"from": members, "to": wanted,
                            "scatter_us": scatter_started.elapsed().as_micros() as u64,
                            "scatter_evaluated": stages,
                            "after_batched_windows": batched_windows}));
                    }
                }
                for &i in solo.iter().chain(if wanted.is_empty() {
                    ready.iter()
                } else {
                    [].iter()
                }) {
                    streams[i].diagnostic_fill_b1()?;
                    b1_windows[i] += 1;
                }
                if !wanted.is_empty() {
                    let previous = cache.take().map(|(_, c)| c);
                    let mut picked = pick_rows(&mut streams, &wanted);
                    let (c, timing) = if ordinary_mode {
                        DFlash2TextGenerationStream::diagnostic_ragged_ordinary_window_bn(
                            &mut picked,
                            previous,
                        )?
                    } else {
                        DFlash2TextGenerationStream::diagnostic_ragged_window_bn(
                            &mut picked,
                            previous,
                        )?
                    };
                    cache = Some((wanted.clone(), c));
                    batched_windows += 1;
                    windows.push(json!({"members": wanted, "timing": timing}));
                }
            }
            if let Some((members, c)) = cache.take() {
                let mut picked = pick_rows(&mut streams, &members);
                c.scatter_to_rows(&mut picked)?;
            }
            let wall_s = loop_started.elapsed().as_secs_f64();
            let rows = (0..n)
                .map(|i| -> Result<Value> {
                    let prompt_ids = request(&tokenizer, &prompts[i], args.max_tokens)?.prompt_ids;
                    Ok(json!({"id": prompts[i]["id"], "prompt_ids": prompt_ids, "tokens": tokens[i],
                        "text": tokenizer.decode(&tokens[i], true)?,
                        "stop": format!("{:?}", stops[i]), "cancelled": cancelled[i], "solo_windows": b1_windows[i]}))
                })
                .collect::<Result<Vec<_>>>()?;
            println!(
                "ragged rows={n} batched_windows={batched_windows} regroups={} wall={wall_s:.2}s tokens={}",
                regroups.len(),
                tokens.iter().map(Vec::len).sum::<usize>()
            );
            write(
                json!({"rows": rows, "windows": windows, "regroups": regroups, "prefill_s": prefill_s,
                "wall_s": wall_s, "stagger": stagger, "cancel": args.cancel,
                "ragged_row_attention_calls": ironmlx_lm::nn::gated_attention::ragged_row_attention_calls(),
                "memory": memory_report(&after_target_load)}),
            )?;
        }
        "parity" => {
            // Same requests decoded one at a time, then together with ragged
            // windows; token ids and the final logical target cache of every
            // row must match bit for bit (run with one fixed budget so both
            // paths take the same windows).
            let n = prompts.len();
            ensure!(n >= 2, "parity mode needs at least two prompts");
            let mut serial = Vec::with_capacity(n);
            let serial_started = Instant::now();
            for prompt in &prompts {
                let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
                    &target,
                    &draft,
                    &tokenizer,
                    request(&tokenizer, prompt, args.max_tokens)?,
                    8,
                    options,
                )?;
                let mut tokens = Vec::new();
                while let Some(event) = stream.next_token()? {
                    tokens.push(event.token);
                }
                serial.push((tokens, stream.diagnostic_target_cache_digest()?));
            }
            let serial_s = serial_started.elapsed().as_secs_f64();
            let mut streams = Vec::with_capacity(n);
            let ragged_started = Instant::now();
            for prompt in &prompts {
                streams.push(DFlash2TextGenerationStream::new_text_only_with_options(
                    &target,
                    &draft,
                    &tokenizer,
                    request(&tokenizer, prompt, args.max_tokens)?,
                    8,
                    options,
                )?);
            }
            let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
            let mut done = vec![false; n];
            let mut cache: Option<(Vec<usize>, DFlash2RaggedBatchCache)> = None;
            let mut batched_windows = 0_usize;
            let mut stops: Vec<Option<&'static str>> = vec![None; n];
            let decode_started = Instant::now();
            loop {
                for i in 0..n {
                    if done[i] {
                        continue;
                    }
                    while !streams[i].diagnostic_needs_window() {
                        match streams[i].diagnostic_next_token_deferred()? {
                            Some(event) => {
                                tokens[i].push(event.token);
                                if event.finish_reason.is_some() {
                                    stops[i] = event.finish_reason;
                                    done[i] = true;
                                    break;
                                }
                            }
                            None => {
                                done[i] = true;
                                break;
                            }
                        }
                    }
                }
                let ordinary_ready: Vec<usize> = (0..n)
                    .filter(|&i| !done[i] && streams[i].diagnostic_ragged_ordinary_eligible())
                    .collect();
                let ordinary =
                    args.ordinary_min_width >= 2 && ordinary_ready.len() >= args.ordinary_min_width;
                let ready: Vec<usize> = if ordinary {
                    ordinary_ready
                } else {
                    (0..n)
                        .filter(|&i| !done[i] && streams[i].diagnostic_ragged_eligible())
                        .collect()
                };
                let solo: Vec<usize> = (0..n)
                    .filter(|&i| !done[i] && !ready.contains(&i))
                    .collect();
                if ready.is_empty() && solo.is_empty() {
                    break;
                }
                let wanted = if ready.len() >= 2 {
                    ready.clone()
                } else {
                    Vec::new()
                };
                if let Some((members, _)) = cache.as_ref() {
                    if *members != wanted {
                        let (members, c) = cache.take().expect("cache present");
                        let mut picked = pick_rows(&mut streams, &members);
                        c.scatter_to_rows(&mut picked)?;
                    }
                }
                for &i in solo.iter().chain(if wanted.is_empty() {
                    ready.iter()
                } else {
                    [].iter()
                }) {
                    streams[i].diagnostic_fill_b1()?;
                }
                if !wanted.is_empty() {
                    let previous = cache.take().map(|(_, c)| c);
                    let mut picked = pick_rows(&mut streams, &wanted);
                    let (c, _timing) = if ordinary {
                        DFlash2TextGenerationStream::diagnostic_ragged_ordinary_window_bn(
                            &mut picked,
                            previous,
                        )?
                    } else {
                        DFlash2TextGenerationStream::diagnostic_ragged_window_bn(
                            &mut picked,
                            previous,
                        )?
                    };
                    cache = Some((wanted.clone(), c));
                    batched_windows += 1;
                }
            }
            if let Some((members, c)) = cache.take() {
                let mut picked = pick_rows(&mut streams, &members);
                c.scatter_to_rows(&mut picked)?;
            }
            let decode_s = decode_started.elapsed().as_secs_f64();
            let ragged_s = ragged_started.elapsed().as_secs_f64();
            if args.relaxed_check {
                let mut rows = Vec::new();
                for i in 0..n {
                    let nonfinite = streams[i].diagnostic_target_cache_nonfinite()?;
                    let mut reference = DFlash2TextGenerationStream::new_text_only_with_options(
                        &target,
                        &draft,
                        &tokenizer,
                        request(&tokenizer, &prompts[i], args.max_tokens)?,
                        8,
                        options,
                    )?;
                    let margins = reference.diagnostic_teacher_forced_margins(&tokens[i])?;
                    let flips: Vec<Value> = margins
                        .iter()
                        .enumerate()
                        .filter(|(k, m)| m.0 != tokens[i][k + 1])
                        .map(|(k, m)| {
                            json!({"position": k + 1, "emitted": tokens[i][k + 1], "reference": m.0,
                                "margin": m.1 - m.2, "reference_gap": m.1 - m.3})
                        })
                        .collect();
                    let first_divergence = tokens[i]
                        .iter()
                        .zip(&serial[i].0)
                        .position(|(a, b)| a != b)
                        .or((tokens[i].len() != serial[i].0.len())
                            .then(|| tokens[i].len().min(serial[i].0.len())));
                    let gaps: Vec<f32> = margins.iter().map(|m| m.1 - m.3).collect();
                    rows.push(json!({"id": prompts[i]["id"], "tokens": tokens[i].len(),
                        "stop": format!("{:?}", stops[i]), "cache_nonfinite": nonfinite,
                        "first_divergence_vs_serial": first_divergence, "flips": flips,
                        "reference_gaps": gaps, "serial_tokens": serial[i].0, "ragged_tokens": tokens[i],
                        "text": tokenizer.decode(&tokens[i], true)?}));
                }
                let total_tokens: usize = tokens.iter().map(Vec::len).sum();
                println!(
                    "relaxed-check rows={n} batched_windows={batched_windows} decode={decode_s:.2}s tokens={total_tokens} flips={}",
                    rows.iter().map(|r| r["flips"].as_array().map_or(0, Vec::len)).sum::<usize>()
                );
                write(
                    json!({"rows": rows, "batched_windows": batched_windows, "serial_s": serial_s,
                    "ragged_s": ragged_s, "decode_s": decode_s, "total_tokens": total_tokens}),
                )?;
                return Ok(());
            }
            let mut rows = Vec::new();
            let mut all_equal = true;
            for i in 0..n {
                let digest = streams[i].diagnostic_target_cache_digest()?;
                let tokens_equal = tokens[i] == serial[i].0;
                let cache_equal = digest == serial[i].1;
                all_equal &= tokens_equal && cache_equal;
                rows.push(json!({"id": prompts[i]["id"], "tokens": tokens[i].len(),
                    "tokens_equal": tokens_equal, "cache_equal": cache_equal,
                    "cache_layers_differing": digest.iter().zip(&serial[i].1).filter(|(a, b)| a != b).count(),
                    "serial_tokens": serial[i].0, "ragged_tokens": tokens[i]}));
            }
            println!("parity rows={n} all_equal={all_equal} batched_windows={batched_windows} serial={serial_s:.2}s ragged={ragged_s:.2}s");
            write(
                json!({"all_equal": all_equal, "rows": rows, "batched_windows": batched_windows,
                "serial_s": serial_s, "ragged_s": ragged_s}),
            )?;
            ensure!(all_equal, "ragged decoding differs from serial decoding");
        }
        other => anyhow::bail!("unknown mode {other}"),
    }
    Ok(())
}

fn pick_rows<'a, 'm>(
    streams: &'a mut [DFlash2TextGenerationStream<'m, Qwen35MoeModel>],
    indices: &[usize],
) -> Vec<&'a mut DFlash2TextGenerationStream<'m, Qwen35MoeModel>> {
    let mut out: Vec<Option<&mut DFlash2TextGenerationStream<'m, Qwen35MoeModel>>> =
        streams.iter_mut().map(Some).collect();
    indices
        .iter()
        .map(|&i| out[i].take().expect("distinct row indices"))
        .collect()
}
