//! iron-bench — head-to-head HTTP benchmark harness for OpenAI-compatible LLM endpoints.
//!
//! Drives multiple `--target name=URL` endpoints with the same synthetic-prompt matrix
//! and reports TTFT / TG decode / TPOT / PP prefill / E2E across N timed runs (median +
//! p95). Engine-neutral; no dependency on ironmlx/mlx crates.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

mod client;
mod prompt;
mod report;
mod runner;

#[derive(Parser, Debug)]
#[command(
    name = "iron-bench",
    about = "Head-to-head HTTP benchmark for OpenAI-compatible LLM endpoints",
    version
)]
struct Args {
    /// Target endpoints. Repeat for multiple targets.
    /// Format: `name=URL` (e.g., `--target ironmlx=http://localhost:8080`).
    #[arg(long, value_parser = parse_target, required = true, num_args = 1..)]
    target: Vec<(String, String)>,

    /// Path to model dir containing `tokenizer.json` (used for prompt synthesis only).
    #[arg(long)]
    model_dir: PathBuf,

    /// Model name to send in the `model` field of each JSON request.
    #[arg(long, default_value = "qwen3.5-4b")]
    model: String,

    /// Prompt token lengths to test (comma-separated).
    #[arg(long, value_delimiter = ',', default_values_t = vec![128_usize, 512, 2048])]
    prompt_len: Vec<usize>,

    /// Number of generated tokens per request.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Timed runs per cell (after warmup).
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Warmup runs per cell (excluded from stats).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Markdown)]
    format: OutputFormat,

    /// HTTP request timeout (seconds).
    #[arg(long, default_value_t = 300)]
    timeout: u64,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum OutputFormat {
    Markdown,
    Csv,
    Json,
}

fn parse_target(s: &str) -> std::result::Result<(String, String), String> {
    s.split_once('=')
        .map(|(name, url)| (name.into(), url.trim_end_matches('/').into()))
        .ok_or_else(|| format!("expected name=URL, got '{s}'"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!(
        "iron-bench: {} target(s), prompt_len={:?}, max_tokens={}, runs={}, warmup={}",
        args.target.len(),
        args.prompt_len,
        args.max_tokens,
        args.runs,
        args.warmup,
    );

    // Load tokenizer.json from --model-dir for synthetic prompt construction.
    let tokenizer_path = args.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        anyhow::anyhow!(
            "failed to load tokenizer at {}: {e}",
            tokenizer_path.display()
        )
    })?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout))
        .build()
        .context("reqwest::Client::build")?;

    let mut cells: Vec<runner::CellResult> = Vec::new();
    for pp in &args.prompt_len {
        for (target_name, target_url) in &args.target {
            let cell = runner::run_cell(
                &client,
                target_name,
                target_url,
                &args.model,
                *pp,
                args.max_tokens,
                args.warmup,
                args.runs,
                &tokenizer,
            )
            .await?;
            cells.push(cell);
        }
    }

    // Render output via report module (T4 fills in the formatters).
    let out = match args.format {
        OutputFormat::Markdown => report::render_markdown(&cells, &args.target, args.warmup),
        OutputFormat::Csv => report::render_csv(&cells),
        OutputFormat::Json => report::render_json(&cells, &args.target, args.warmup),
    };
    println!("{out}");

    Ok(())
}
