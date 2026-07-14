//! iron-bench — head-to-head HTTP benchmark harness for OpenAI-compatible LLM endpoints.
//!
//! Drives multiple `--target name=URL` endpoints with the same prompt matrix
//! and reports TTFT / TG decode / TPOT / PP prefill / E2E across N timed runs (median +
//! p95). Engine-neutral; no dependency on ironmlx/mlx crates.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};
use clap::Parser;

mod client;
mod prompt;
mod report;
mod runner;

use prompt::build_prompt_sources;

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

    /// Path to model dir containing `tokenizer.json` (used for prompt construction and PP labels).
    #[arg(long)]
    model_dir: PathBuf,

    /// Model name to send in the `model` field of each JSON request.
    #[arg(long, default_value = "qwen3.5-4b")]
    model: String,

    /// Prompt token lengths to test (comma-separated).
    #[arg(long, value_delimiter = ',')]
    prompt_len: Option<Vec<usize>>,

    /// Use the exact prompt text from this file for one benchmark cell.
    /// The reported PP target is the local tokenizer count of the file
    /// contents. Mutually exclusive with `--prompt-len`.
    #[arg(long)]
    fixed_prompt_file: Option<PathBuf>,

    /// Number of generated tokens per request.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Continue generation through EOS until `--max-tokens` is reached.
    /// Use only for controlled full-length decode measurements.
    #[arg(long, default_value_t = false)]
    ignore_eos: bool,

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

    /// Scheduler b_max value used by the benchmarked server.
    #[arg(long)]
    pub autotune_b_max: Option<usize>,

    /// Scheduler prefill_chunk_size value used by the benchmarked server.
    #[arg(long)]
    pub autotune_prefill_chunk_size: Option<usize>,

    /// Scheduler admission_deadline_ms value used by the benchmarked server.
    #[arg(long)]
    pub autotune_admission_deadline_ms: Option<u64>,

    /// Scheduler admission_queue_max value used by the benchmarked server.
    #[arg(long)]
    pub autotune_admission_queue_max: Option<usize>,

    /// Scheduler max_cache_cap value used by the benchmarked server.
    #[arg(long)]
    pub autotune_max_cache_cap: Option<usize>,

    /// Scheduler decode-cadence mid-admit chunk cap value used by the benchmarked server.
    #[arg(long)]
    pub autotune_decode_cadence_mid_chunk_cap: Option<usize>,

    /// Mark exported autotune measurements as memory-budget unsafe.
    #[arg(long, default_value_t = false)]
    pub autotune_memory_budget_unsafe: bool,

    /// HTTP request timeout (seconds).
    #[arg(long, default_value_t = 300)]
    timeout: u64,

    /// Capture X-Ironmlx-Request-Id response header from each request and
    /// add a request_id column to CSV output. Default off — flag-off state
    /// is byte-identical to the base iron-bench output. Markdown + JSON
    /// outputs are unaffected by this flag.
    #[arg(long, default_value_t = false)]
    pub capture_server_request_id: bool,

    /// Append `run_start_unix_ns` and `run_end_unix_ns` columns to CSV
    /// output. When off, CSV is byte-identical to current output. When
    /// combined with `--capture-server-request-id`, both column families
    /// appear; downstream parsers MUST use header names (csv::DictReader),
    /// not fixed positions.
    #[arg(long, default_value_t = false)]
    pub capture_run_timestamps: bool,

    /// Sleep N seconds between measured runs in sequential (v1) mode.
    /// Does NOT sleep during preheat or warmup; does NOT sleep after the
    /// final measured run. Default 0 (no behavior change). Use this for
    /// sweeps where thermal isolation between measured runs matters.
    #[arg(long, default_value_t = 0u64)]
    pub inter_run_cooldown_secs: u64,

    /// Override the synthetic-prompt nonce seed in sequential mode. When
    /// unset, nonce generation remains time-based. Makes measured prompt
    /// sequences reproducible across repeats while still varying nonce by
    /// warmup/run index.
    #[arg(long)]
    pub nonce_seed: Option<u64>,

    /// Reuse the same prompt text for all requests in each cell so server-side
    /// prefix caches can be measured. Default off: synthetic prompts vary by
    /// run to avoid accidental cache hits in normal PP benchmarks.
    #[arg(long, default_value_t = false)]
    pub prefix_cache_probe: bool,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Markdown,
    Csv,
    Json,
    AutotuneJson,
}

fn parse_target(s: &str) -> std::result::Result<(String, String), String> {
    s.split_once('=')
        .map(|(name, url)| (name.into(), url.trim_end_matches('/').into()))
        .ok_or_else(|| format!("expected name=URL, got '{s}'"))
}

impl Args {
    fn autotune_export_options(&self) -> Result<Option<report::AutotuneExportOptions>> {
        if self.format != OutputFormat::AutotuneJson {
            return Ok(None);
        }

        if self.target.len() != 1 {
            anyhow::bail!(
                "--format autotune-json requires exactly one --target because the calibration schema has no target field"
            );
        }

        Ok(Some(report::AutotuneExportOptions {
            model_name: self.model.clone(),
            hardware_label: detect_hardware_label(),
            config: report::AutotuneProfileConfig {
                b_max: self
                    .autotune_b_max
                    .context("--autotune-b-max is required with --format autotune-json")?,
                prefill_chunk_size: self.autotune_prefill_chunk_size.context(
                    "--autotune-prefill-chunk-size is required with --format autotune-json",
                )?,
                admission_deadline_ms: self.autotune_admission_deadline_ms.context(
                    "--autotune-admission-deadline-ms is required with --format autotune-json",
                )?,
                admission_queue_max: self.autotune_admission_queue_max.context(
                    "--autotune-admission-queue-max is required with --format autotune-json",
                )?,
                max_cache_cap: self
                    .autotune_max_cache_cap
                    .context("--autotune-max-cache-cap is required with --format autotune-json")?,
                decode_cadence_mid_chunk_cap: self
                    .autotune_decode_cadence_mid_chunk_cap
                    .context(
                        "--autotune-decode-cadence-mid-chunk-cap is required with --format autotune-json",
                    )?,
            },
            memory_budget_ok: !self.autotune_memory_budget_unsafe,
        }))
    }
}

fn detect_hardware_label() -> String {
    hardware_label_from_parts(detect_cpu_label().as_deref(), detect_total_ram_bytes())
}

fn hardware_label_from_parts(cpu_label: Option<&str>, total_ram_bytes: Option<u64>) -> String {
    let cpu = cpu_label
        .map(slugify_hardware_component)
        .filter(|label| !label.is_empty())
        .unwrap_or_else(|| slugify_hardware_component(std::env::consts::ARCH));

    match total_ram_bytes {
        Some(bytes) => format!("{cpu}-{}gb", rounded_gib(bytes)),
        None => cpu,
    }
}

fn slugify_hardware_component(value: &str) -> String {
    let mut slug = String::new();
    let mut last_was_separator = false;

    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_was_separator = false;
        } else if !last_was_separator && !slug.is_empty() {
            slug.push('-');
            last_was_separator = true;
        }
    }

    while slug.ends_with('-') {
        slug.pop();
    }

    if slug.is_empty() {
        "unknown".to_string()
    } else {
        slug
    }
}

fn rounded_gib(bytes: u64) -> u64 {
    let gib = 1024_u64.pow(3);
    ((bytes + gib / 2) / gib).max(1)
}

#[cfg(target_os = "macos")]
fn detect_cpu_label() -> Option<String> {
    command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(|| command_output("sysctl", &["-n", "hw.model"]))
}

#[cfg(target_os = "linux")]
fn detect_cpu_label() -> Option<String> {
    let raw = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    raw.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name")
            .then(|| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_cpu_label() -> Option<String> {
    None
}

#[cfg(target_os = "macos")]
fn detect_total_ram_bytes() -> Option<u64> {
    command_output("sysctl", &["-n", "hw.memsize"])?
        .parse::<u64>()
        .ok()
}

#[cfg(target_os = "linux")]
fn detect_total_ram_bytes() -> Option<u64> {
    let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        let Some(rest) = line.strip_prefix("MemTotal:") else {
            continue;
        };
        let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        return kb.checked_mul(1024);
    }
    None
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_total_ram_bytes() -> Option<u64> {
    None
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.capture_run_timestamps && args.concurrent.is_some() {
        // Concurrent CSV (render_csv_concurrent) has a different header schema
        // with no run_start/run_end columns, so timestamp capture targets
        // sequential mode only.
        anyhow::bail!(
            "--capture-run-timestamps is incompatible with --concurrent: \
             run_start_unix_ns/run_end_unix_ns are defined only for v1 sequential CSV rows."
        );
    }

    if args.capture_server_request_id {
        // The concurrent CSV path (render_csv_concurrent) has a different
        // header schema with no request_id column.
        if args.concurrent.is_some() {
            anyhow::bail!(
                "--capture-server-request-id is incompatible with --concurrent \
                 because concurrent CSV has no request_id column."
            );
        }
        // Keep request-id captures joinable by ensuring every server-side
        // request id also appears in the emitted sequential CSV.
        if args.warmup != 0 {
            anyhow::bail!(
                "--capture-server-request-id is incompatible with --warmup > 0 \
                 because warmup RequestResults are discarded by runner.rs while \
                 the server still emits X-Ironmlx-Request-Id headers for warmup \
                 requests. Use --warmup 0."
            );
        }
        // Defense-in-depth — redundant given the concurrent gate above, but
        // kept in case the concurrent gate is ever relaxed.
        if args.concurrent.is_some() && args.warmup_duration != 0 {
            anyhow::bail!(
                "--capture-server-request-id is incompatible with --warmup-duration > 0 \
                 when concurrent mode is enabled."
            );
        }
    }

    if args.concurrent.is_some() && args.inter_run_cooldown_secs != 0 {
        anyhow::bail!(
            "--inter-run-cooldown-secs is incompatible with concurrent (v2) mode \
             because cooldown semantics are defined only for the sequential \
             measured-run loop. Set --inter-run-cooldown-secs 0 when using \
             --concurrent."
        );
    }

    if args.concurrent.is_some() && args.nonce_seed.is_some() {
        anyhow::bail!(
            "--nonce-seed is incompatible with concurrent (v2) mode: fixed nonce \
             sequence semantics are defined only for v1 sequential warmup/measured runs."
        );
    }

    if args.fixed_prompt_file.is_some() && args.nonce_seed.is_some() {
        anyhow::bail!(
            "--nonce-seed is incompatible with --fixed-prompt-file: fixed prompts do not use nonces."
        );
    }

    if args.prefix_cache_probe && args.format == OutputFormat::AutotuneJson {
        anyhow::bail!("--prefix-cache-probe is incompatible with --format autotune-json");
    }

    let autotune_options = args.autotune_export_options()?;
    let prefix_cache_probe = runner::PrefixCacheProbe::from_enabled(args.prefix_cache_probe);

    // Load tokenizer.json before building prompt sources so fixed prompts get
    // an accurate local PP label.
    let tokenizer_path = args.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        anyhow::anyhow!(
            "failed to load tokenizer at {}: {e}",
            tokenizer_path.display()
        )
    })?;

    let fixed_prompt_text =
        match &args.fixed_prompt_file {
            Some(path) => Some(std::fs::read_to_string(path).with_context(|| {
                format!("failed to read --fixed-prompt-file {}", path.display())
            })?),
            None => None,
        };
    let prompt_sources = build_prompt_sources(
        &tokenizer,
        args.prompt_len.as_deref(),
        fixed_prompt_text.as_deref(),
    )?;
    let prompt_targets: Vec<usize> = prompt_sources
        .iter()
        .map(|source| source.target_tokens())
        .collect();

    match args.concurrent {
        None => eprintln!(
            "iron-bench v1 (sequential): {} target(s), prompt_len={:?}, max_tokens={}, runs={}, warmup={}",
            args.target.len(),
            prompt_targets,
            args.max_tokens,
            args.runs,
            args.warmup,
        ),
        Some(n) => eprintln!(
            "iron-bench v2 (concurrent): {} target(s), prompt_len={:?}, max_tokens={}, concurrent={}, duration={}s, warmup_duration={}s",
            args.target.len(),
            prompt_targets,
            args.max_tokens,
            n,
            args.duration,
            args.warmup_duration,
        ),
    }

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
            for prompt_source in &prompt_sources {
                for (target_name, target_url) in &args.target {
                    let cell = runner::run_cell(
                        &client,
                        target_name,
                        target_url,
                        &args.model,
                        prompt_source,
                        args.max_tokens,
                        args.warmup,
                        args.runs,
                        args.capture_server_request_id,
                        args.ignore_eos,
                        args.capture_run_timestamps,
                        args.inter_run_cooldown_secs,
                        args.nonce_seed,
                        prefix_cache_probe,
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
            for prompt_source in &prompt_sources {
                let prompt_source_arc = std::sync::Arc::new(prompt_source.clone());
                for (target_name, target_url) in &args.target {
                    let cell = runner::run_cell_concurrent(
                        client_arc.clone(),
                        target_name,
                        target_url,
                        &args.model,
                        prompt_source_arc.clone(),
                        args.max_tokens,
                        std::time::Duration::from_secs(args.warmup_duration),
                        std::time::Duration::from_secs(args.duration),
                        concurrent,
                        args.capture_server_request_id,
                        args.ignore_eos,
                        prefix_cache_probe,
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
                    if args.prefix_cache_probe {
                        report::render_markdown_with_prefix_cache_probe(
                            &seq_cells,
                            &args.target,
                            args.warmup,
                        )
                    } else {
                        report::render_markdown(&seq_cells, &args.target, args.warmup)
                    }
                }
                OutputFormat::Csv => {
                    if args.prefix_cache_probe {
                        report::render_csv_with_prefix_cache_probe(
                            &seq_cells,
                            args.capture_server_request_id,
                            args.capture_run_timestamps,
                            args.warmup,
                        )
                    } else {
                        report::render_csv(
                            &seq_cells,
                            args.capture_server_request_id,
                            args.capture_run_timestamps,
                        )
                    }
                }
                OutputFormat::Json => {
                    if args.prefix_cache_probe {
                        report::render_json_with_prefix_cache_probe(
                            &seq_cells,
                            &args.target,
                            args.warmup,
                        )
                    } else {
                        report::render_json(&seq_cells, &args.target, args.warmup)
                    }
                }
                OutputFormat::AutotuneJson => report::render_autotune_json_sequential(
                    &seq_cells,
                    autotune_options
                        .as_ref()
                        .expect("autotune options are validated before benchmark execution"),
                ),
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
                OutputFormat::Markdown => {
                    if args.prefix_cache_probe {
                        report::render_markdown_concurrent_with_prefix_cache_probe(
                            &conc_cells,
                            &args.target,
                            concurrent,
                            args.duration,
                            args.warmup_duration,
                        )
                    } else {
                        report::render_markdown_concurrent(
                            &conc_cells,
                            &args.target,
                            concurrent,
                            args.duration,
                            args.warmup_duration,
                        )
                    }
                }
                OutputFormat::Csv => {
                    if args.prefix_cache_probe {
                        report::render_csv_concurrent_with_prefix_cache_probe(&conc_cells)
                    } else {
                        report::render_csv_concurrent(&conc_cells)
                    }
                }
                OutputFormat::Json => {
                    if args.prefix_cache_probe {
                        report::render_json_concurrent_with_prefix_cache_probe(
                            &conc_cells,
                            &args.target,
                            concurrent,
                            args.duration,
                            args.warmup_duration,
                        )
                    } else {
                        report::render_json_concurrent(
                            &conc_cells,
                            &args.target,
                            concurrent,
                            args.duration,
                            args.warmup_duration,
                        )
                    }
                }
                OutputFormat::AutotuneJson => report::render_autotune_json_concurrent(
                    &conc_cells,
                    autotune_options
                        .as_ref()
                        .expect("autotune options are validated before benchmark execution"),
                ),
            }
        }
    };
    println!("{out}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{hardware_label_from_parts, Args, OutputFormat};

    #[test]
    fn autotune_cli_parses_output_format_and_scheduler_config() {
        let args = Args::parse_from([
            "iron-bench",
            "--target",
            "ironmlx=http://localhost:8080",
            "--model-dir",
            "/tmp/model",
            "--format",
            "autotune-json",
            "--autotune-b-max",
            "2",
            "--autotune-prefill-chunk-size",
            "1024",
            "--autotune-admission-deadline-ms",
            "5",
            "--autotune-admission-queue-max",
            "32",
            "--autotune-max-cache-cap",
            "32768",
            "--autotune-decode-cadence-mid-chunk-cap",
            "256",
        ]);

        assert!(matches!(args.format, OutputFormat::AutotuneJson));
        assert_eq!(args.autotune_b_max, Some(2));
        assert_eq!(args.autotune_prefill_chunk_size, Some(1024));
        assert_eq!(args.autotune_admission_deadline_ms, Some(5));
        assert_eq!(args.autotune_admission_queue_max, Some(32));
        assert_eq!(args.autotune_max_cache_cap, Some(32768));
        assert_eq!(args.autotune_decode_cadence_mid_chunk_cap, Some(256));
        assert!(!args.autotune_memory_budget_unsafe);
    }

    #[test]
    fn autotune_cli_rejects_manual_hardware_label() {
        let err = Args::try_parse_from([
            "iron-bench",
            "--target",
            "ironmlx=http://localhost:8080",
            "--model-dir",
            "/tmp/model",
            "--format",
            "autotune-json",
            "--autotune-hardware-label",
            "m3-max",
            "--autotune-b-max",
            "2",
            "--autotune-prefill-chunk-size",
            "1024",
            "--autotune-admission-deadline-ms",
            "5",
            "--autotune-admission-queue-max",
            "32",
            "--autotune-max-cache-cap",
            "32768",
            "--autotune-decode-cadence-mid-chunk-cap",
            "256",
        ])
        .expect_err("manual hardware labels should be rejected");

        assert!(err.to_string().contains("autotune-hardware-label"));
    }

    #[test]
    fn autotune_export_options_generates_hardware_label_when_omitted() {
        let args = Args::parse_from([
            "iron-bench",
            "--target",
            "ironmlx=http://localhost:8080",
            "--model-dir",
            "/tmp/model",
            "--format",
            "autotune-json",
            "--autotune-b-max",
            "2",
            "--autotune-prefill-chunk-size",
            "1024",
            "--autotune-admission-deadline-ms",
            "5",
            "--autotune-admission-queue-max",
            "32",
            "--autotune-max-cache-cap",
            "32768",
            "--autotune-decode-cadence-mid-chunk-cap",
            "256",
        ]);

        let options = args
            .autotune_export_options()
            .expect("autotune options should parse")
            .expect("autotune-json should produce export options");

        assert!(!options.hardware_label.trim().is_empty());
    }

    #[test]
    fn cli_parses_prefix_cache_probe_flag() {
        let args = Args::parse_from([
            "iron-bench",
            "--target",
            "ironmlx=http://localhost:8080",
            "--model-dir",
            "/tmp/model",
            "--prefix-cache-probe",
        ]);

        assert!(args.prefix_cache_probe);
    }

    #[test]
    fn hardware_label_from_parts_slugifies_cpu_and_memory() {
        let label = hardware_label_from_parts(Some("Apple M5 Max"), Some(128 * 1024_u64.pow(3)));

        assert_eq!(label, "apple-m5-max-128gb");
    }
}
