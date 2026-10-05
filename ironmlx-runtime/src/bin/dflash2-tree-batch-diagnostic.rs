//! Diagnostic-only feasibility probe for DFlash2 cross-request batching: per
//! window stage accounting of single-request tree or linear windows
//! (`IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES=1` adds synchronizing stage
//! boundaries; without it only end-to-end metrics are recorded), and a
//! projection bench at several row counts. Not a serving path.
use anyhow::{ensure, Result};
use clap::Parser;
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::{
    models::{dflash2::DFlash2DraftModel, Qwen35Model},
    Loader, Tokenizer,
};
use ironmlx_runtime::core::{
    dflash2::{take_window_stage_records, DFlash2P2Options, DFlash2TextGenerationStream},
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
    /// windows | projections
    #[arg(long)]
    mode: String,
    #[arg(long)]
    prompts: Option<PathBuf>,
    #[arg(long, value_delimiter = ',')]
    ids: Vec<String>,
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,
    #[arg(long, default_value_t = 15)]
    tree_nodes: usize,
    #[arg(long, default_value_t = 8)]
    block_size: usize,
    #[arg(long, default_value = "16,32,64", value_delimiter = ',')]
    rows: Vec<i32>,
    #[arg(long, default_value_t = 20)]
    iters: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "preserve existing report; choose a new output"
    );
    let root = std::env::var("MLX_DIR")?;
    mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
    let started = Instant::now();
    let mut loader = Loader::open(&args.target)?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let target = Qwen35Model::from_loader_dflash2(&mut loader)?;
    drop(loader);
    let load_s = started.elapsed().as_secs_f64();
    let env: Value = std::env::vars()
        .filter(|(k, _)| k.starts_with("IRONMLX_"))
        .map(|(k, v)| (k, Value::String(v)))
        .collect::<serde_json::Map<String, Value>>()
        .into();
    if args.mode == "projections" {
        let bench = target.diagnostic_projection_bench(&args.rows, args.iters)?;
        std::fs::write(
            &args.output,
            serde_json::to_vec_pretty(
                &json!({"mode": "projections", "env": env, "load_s": load_s,
                "rows": args.rows, "iters": args.iters, "result": bench}),
            )?,
        )?;
        return Ok(());
    }
    ensure!(
        args.mode == "windows",
        "mode must be windows or projections"
    );
    let loader = Loader::open_dflash2(&args.draft)?;
    let draft = DFlash2DraftModel::from_loader(&loader, target.config(), Some(4))?;
    drop(loader);
    let prompts: Vec<Value> =
        serde_json::from_slice(&std::fs::read(args.prompts.as_ref().expect("--prompts"))?)?;
    let mut records = Vec::new();
    for prompt in prompts
        .iter()
        .filter(|p| args.ids.is_empty() || args.ids.iter().any(|id| p["id"] == id.as_str()))
    {
        let text = tokenizer.apply_chat_template(
            &[json!({"role":"user","content":prompt["prompt"]})],
            true,
            Some(&json!({"enable_thinking":false})),
        )?;
        let prompt_ids = tokenizer.encode(&text, false)?;
        let request = GenerateRequest {
            priority: Default::default(),
            prompt_ids: prompt_ids.clone(),
            max_new_tokens: args.max_tokens,
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
        };
        let _ = take_window_stage_records();
        let started = Instant::now();
        let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
            &target,
            &draft,
            &tokenizer,
            request,
            args.block_size,
            DFlash2P2Options {
                tree_max_nodes: args.tree_nodes,
                ..Default::default()
            },
        )?;
        let mut tokens = Vec::new();
        let mut first_token_s = None;
        let mut stop = None;
        while let Some(event) = stream.next_token()? {
            first_token_s.get_or_insert_with(|| started.elapsed().as_secs_f64());
            tokens.push(event.token);
            if event.finish_reason.is_some() {
                stop = event.finish_reason;
            }
        }
        let wall_s = started.elapsed().as_secs_f64();
        let windows = take_window_stage_records();
        let decoded = tokenizer.decode(&tokens, true)?;
        println!(
            "{} tokens={} windows={} wall={:.2}s first={:.3}s",
            prompt["id"],
            tokens.len(),
            windows.len(),
            wall_s,
            first_token_s.unwrap_or(0.0)
        );
        records.push(json!({"id": prompt["id"], "prompt_tokens": prompt_ids.len(), "tokens": tokens,
            "text": decoded, "stop": format!("{stop:?}"), "wall_s": wall_s, "first_token_s": first_token_s,
            "metrics": stream.metrics(), "windows": windows}));
        std::fs::write(
            &args.output,
            serde_json::to_vec_pretty(&json!({"mode": "windows", "env": env, "load_s": load_s,
                "tree_nodes": args.tree_nodes, "block_size": args.block_size, "records": records}))?,
        )?;
    }
    Ok(())
}
