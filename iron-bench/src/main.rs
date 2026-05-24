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

    /// (v1 sequential mode) Timed runs per cell. Mutually exclusive with `--concurrent`.
    #[arg(long, default_value_t = 5, conflicts_with = "concurrent")]
    runs: usize,

    /// (v1 sequential mode) Warmup runs per cell (excluded from stats).
    /// Mutually exclusive with `--concurrent`.
    #[arg(long, default_value_t = 1, conflicts_with = "concurrent")]
    warmup: usize,

    /// (v2 concurrent mode) Number of concurrent workers per cell. Each worker
    /// fires request -> awaits response -> repeats until `--duration` deadline.
    /// When absent, runs in v1 sequential mode.
    #[arg(long)]
    concurrent: Option<usize>,

    /// (v2 concurrent mode) Wall-clock duration per cell (seconds).
    /// Only meaningful when `--concurrent` is set; ignored otherwise.
    #[arg(long, default_value_t = 30)]
    duration: u64,

    /// (v2 concurrent mode) Wall-clock warmup duration per cell (seconds).
    /// Only meaningful when `--concurrent` is set; ignored otherwise.
    #[arg(long, default_value_t = 5)]
    warmup_duration: u64,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Markdown)]
    format: OutputFormat,

    /// HTTP request timeout (seconds).
    #[arg(long, default_value_t = 300)]
    timeout: u64,

    /// Capture X-Ironmlx-Request-Id response header from each request and
    /// add a request_id column to CSV output. Default off — flag-off state
    /// is byte-identical to non-P5h iron-bench output (per P5h spec § 2.5a
    /// Join key). Markdown + JSON outputs are unaffected by this flag.
    #[arg(long, default_value_t = false)]
    pub capture_server_request_id: bool,

    /// Append `run_start_unix_ns` and `run_end_unix_ns` columns to CSV
    /// output. When off, CSV is byte-identical to current output. When
    /// combined with `--capture-server-request-id`, both column families
    /// appear; downstream parsers MUST use header names (csv::DictReader),
    /// not fixed positions. Per P5h+2.b spec § 6.
    #[arg(long, default_value_t = false)]
    pub capture_run_timestamps: bool,
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

    if args.capture_run_timestamps && args.concurrent.is_some() {
        // Concurrent CSV (render_csv_concurrent) has a different header schema
        // with no run_start/run_end columns. P5h+2.b timestamp capture targets
        // sequential mode only (per spec § 6 + memory [feedback_serial_perf_experiments]).
        anyhow::bail!(
            "--capture-run-timestamps is incompatible with --concurrent: \
             run_start_unix_ns/run_end_unix_ns are defined only for v1 sequential CSV rows."
        );
    }

    if args.capture_server_request_id {
        // Per Codex plan review v21 P2 #2: reject concurrent mode entirely.
        // The concurrent CSV path (render_csv_concurrent) has a different
        // header schema with no request_id column, and P5h sweeps are
        // serial-only per memory [feedback_serial_perf_experiments].
        if args.concurrent.is_some() {
            anyhow::bail!(
                "--capture-server-request-id is incompatible with --concurrent \
                 (per P5h plan v21 P2 #2): concurrent CSV has a different header \
                 schema with no request_id column, and P5h sweeps are serial-only. \
                 Drop --concurrent for P5h sweeps."
            );
        }
        // Per Codex plan review v20 P1 #2: reject nonzero sequential warmup.
        if args.warmup != 0 {
            anyhow::bail!(
                "--capture-server-request-id is incompatible with --warmup > 0 \
                 (per P5h plan v20 P1 #2): warmup RequestResults are discarded \
                 by runner.rs, but the server still emits [p5h-profile] log \
                 lines + X-Ironmlx-Request-Id headers for warmup requests, so \
                 warmup request_ids will be server-side orphans and the \
                 aggregator's 100% join gate will hard-fail. Use --warmup 0."
            );
        }
        // Defense-in-depth — redundant given the concurrent gate above, but
        // kept in case the concurrent gate is ever relaxed.
        if args.concurrent.is_some() && args.warmup_duration != 0 {
            anyhow::bail!(
                "--capture-server-request-id is incompatible with --warmup-duration > 0 \
                 (per P5h plan v20 P1 #2)."
            );
        }
    }

    match args.concurrent {
        None => eprintln!(
            "iron-bench v1 (sequential): {} target(s), prompt_len={:?}, max_tokens={}, runs={}, warmup={}",
            args.target.len(),
            args.prompt_len,
            args.max_tokens,
            args.runs,
            args.warmup,
        ),
        Some(n) => eprintln!(
            "iron-bench v2 (concurrent): {} target(s), prompt_len={:?}, max_tokens={}, concurrent={}, duration={}s, warmup_duration={}s",
            args.target.len(),
            args.prompt_len,
            args.max_tokens,
            n,
            args.duration,
            args.warmup_duration,
        ),
    }

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

    // Cells are heterogeneous between v1 (Sequential) and v2 (Concurrent) modes.
    // Use the unified enum so the existing `for cell in cells { render }` loop
    // in main.rs stays clean.
    enum AnyCell {
        Sequential(runner::CellResult),
        Concurrent(runner::ConcurrentCellResult),
    }

    let mut cells: Vec<AnyCell> = Vec::new();

    match args.concurrent {
        None => {
            // v1 sequential path
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
                        args.capture_server_request_id,
                        args.capture_run_timestamps,
                        &tokenizer,
                    )
                    .await?;
                    cells.push(AnyCell::Sequential(cell));
                }
            }
        }
        Some(concurrent) => {
            // v2 concurrent path: share Client + Tokenizer via Arc.
            let client_arc = std::sync::Arc::new(client);
            let tokenizer_arc = std::sync::Arc::new(tokenizer);
            for pp in &args.prompt_len {
                for (target_name, target_url) in &args.target {
                    let cell = runner::run_cell_concurrent(
                        client_arc.clone(),
                        target_name,
                        target_url,
                        &args.model,
                        *pp,
                        args.max_tokens,
                        std::time::Duration::from_secs(args.warmup_duration),
                        std::time::Duration::from_secs(args.duration),
                        concurrent,
                        args.capture_server_request_id,
                        tokenizer_arc.clone(),
                    )
                    .await?;
                    cells.push(AnyCell::Concurrent(cell));
                }
            }
        }
    }

    // Split cells back into sequential vs concurrent slices for the existing
    // (v1) renderers + the new (v2) renderers. Per-cell mode mixing is
    // impossible (CLI dispatches uniformly), so all cells share one mode.
    let out = match args.concurrent {
        None => {
            let seq_cells: Vec<runner::CellResult> = cells
                .into_iter()
                .filter_map(|c| match c {
                    AnyCell::Sequential(s) => Some(s),
                    AnyCell::Concurrent(_) => None,
                })
                .collect();
            match args.format {
                OutputFormat::Markdown => {
                    report::render_markdown(&seq_cells, &args.target, args.warmup)
                }
                OutputFormat::Csv => report::render_csv(
                    &seq_cells,
                    args.capture_server_request_id,
                    args.capture_run_timestamps,
                ),
                OutputFormat::Json => report::render_json(&seq_cells, &args.target, args.warmup),
            }
        }
        Some(concurrent) => {
            let conc_cells: Vec<runner::ConcurrentCellResult> = cells
                .into_iter()
                .filter_map(|c| match c {
                    AnyCell::Sequential(_) => None,
                    AnyCell::Concurrent(c) => Some(c),
                })
                .collect();
            match args.format {
                OutputFormat::Markdown => report::render_markdown_concurrent(
                    &conc_cells,
                    &args.target,
                    concurrent,
                    args.duration,
                    args.warmup_duration,
                ),
                OutputFormat::Csv => report::render_csv_concurrent(&conc_cells),
                OutputFormat::Json => report::render_json_concurrent(
                    &conc_cells,
                    &args.target,
                    concurrent,
                    args.duration,
                    args.warmup_duration,
                ),
            }
        }
    };
    println!("{out}");

    Ok(())
}
