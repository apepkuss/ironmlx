use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Args;
use serde::Serialize;

use super::scheduler_profile_store::SchedulerProfileStore;
use crate::core::scheduler_autotune::{
    build_scheduler_autotune_runtime_profile, merge_scheduler_autotune_calibrations,
    select_scheduler_autotune_profile_with_options, SchedulerAutotuneCalibrationInput,
    SchedulerAutotuneMergeOptions, SchedulerAutotuneProfileConfig,
    SchedulerAutotuneSelectionOptions, SchedulerAutotuneSelectionProfile,
};
use crate::Result;

const DEFAULT_PORT: u16 = 18080;
const DEFAULT_STARTUP_TIMEOUT_SEC: u64 = 300;
const DEFAULT_OUTPUT_DIR: &str = "reports/scheduler-autotune";
const DEFAULT_RUNTIME_PROFILE_FILE: &str = "scheduler-profile.json";
const DEFAULT_PROMPT_LEN: &[usize] = &[1024, 4096];
const DEFAULT_CONCURRENCY: &[usize] = &[1, 2];
const RUN_ORDER_MANIFEST_FILE: &str = "run-order.json";
const RUN_ORDER_STRATEGY: &str = "concurrency-major-mirrored-candidate-order";

#[derive(Args, Debug)]
pub struct SchedulerAutotuneCalibrateArgs {
    /// Local directory containing config.json + model.safetensors + tokenizer.json.
    #[arg(long)]
    pub model: PathBuf,

    /// Model name to pass to iron-bench request payloads and calibration JSON.
    /// Defaults to the model directory name.
    #[arg(long)]
    pub model_name: Option<String>,

    /// Path to the iron-bench binary. Defaults to `iron-bench` next to the
    /// running `ironmlx` executable.
    #[arg(long)]
    pub iron_bench_bin: Option<PathBuf>,

    /// Directory for candidate JSON files, logs, and final outputs.
    /// Defaults to `reports/scheduler-autotune`.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// Scheduler candidate config, repeated once per candidate. When omitted,
    /// a conservative built-in agent-oriented matrix is used.
    #[arg(long = "candidate", value_parser = parse_candidate_config)]
    pub candidates: Vec<SchedulerAutotuneProfileConfig>,

    /// Prompt token lengths to test. Defaults to `1024,4096`.
    #[arg(long, value_delimiter = ',')]
    pub prompt_len: Vec<usize>,

    /// Number of generated tokens per request.
    #[arg(long, default_value_t = 128)]
    pub max_tokens: usize,

    /// Concurrency levels to test. `1` uses sequential iron-bench mode.
    /// Defaults to `1,2`.
    #[arg(long, value_delimiter = ',')]
    pub concurrency: Vec<usize>,

    /// Profile used to weight calibration scenarios during final selection.
    #[arg(long, value_enum, default_value_t = super::scheduler_autotune::SchedulerAutotuneSelectionProfileArg::AgentLongPrompt)]
    pub selection_profile: super::scheduler_autotune::SchedulerAutotuneSelectionProfileArg,

    /// Sequential measured runs per cell.
    #[arg(long, default_value_t = 5)]
    pub runs: usize,

    /// Sequential warmup runs per cell.
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,

    /// Concurrent measured duration in seconds.
    #[arg(long, default_value_t = 30)]
    pub duration: u64,

    /// Concurrent warmup duration in seconds.
    #[arg(long, default_value_t = 5)]
    pub warmup_duration: u64,

    /// Local port used by the candidate serve subprocess.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,

    /// Seconds to wait for the candidate serve subprocess to become healthy.
    #[arg(long, default_value_t = DEFAULT_STARTUP_TIMEOUT_SEC)]
    pub startup_timeout_sec: u64,

    /// Runtime scheduler profile output path. Defaults to
    /// `<output-dir>/scheduler-profile.json`.
    #[arg(long)]
    pub write_profile: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedRunConfig {
    model: PathBuf,
    model_name: String,
    iron_bench_bin: PathBuf,
    output_dir: PathBuf,
    candidates: Vec<SchedulerAutotuneProfileConfig>,
    prompt_len: Vec<usize>,
    max_tokens: usize,
    concurrency: Vec<usize>,
    selection_profile: SchedulerAutotuneSelectionProfile,
    runs: usize,
    warmup: usize,
    duration: u64,
    warmup_duration: u64,
    port: u16,
    startup_timeout_sec: u64,
    write_profile: PathBuf,
}

pub fn parse_candidate_config(
    raw: &str,
) -> std::result::Result<SchedulerAutotuneProfileConfig, String> {
    let mut b_max = None;
    let mut prefill_chunk_size = None;
    let mut admission_deadline_ms = None;
    let mut admission_queue_max = None;
    let mut max_cache_cap = None;
    let mut decode_cadence_mid_chunk_cap = None;

    for part in raw.split(',') {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| format!("candidate item must be key=value: {part}"))?;
        match key {
            "b_max" => b_max = Some(parse_usize_value(key, value)?),
            "prefill_chunk_size" => prefill_chunk_size = Some(parse_usize_value(key, value)?),
            "admission_deadline_ms" => {
                admission_deadline_ms = Some(parse_u64_value(key, value)?);
            }
            "admission_queue_max" => admission_queue_max = Some(parse_usize_value(key, value)?),
            "max_cache_cap" => max_cache_cap = Some(parse_usize_value(key, value)?),
            "decode_cadence_mid_chunk_cap" => {
                decode_cadence_mid_chunk_cap = Some(parse_usize_value(key, value)?);
            }
            other => return Err(format!("unknown candidate key: {other}")),
        }
    }

    Ok(SchedulerAutotuneProfileConfig {
        b_max: b_max.ok_or_else(|| "missing b_max".to_string())?,
        prefill_chunk_size: prefill_chunk_size
            .ok_or_else(|| "missing prefill_chunk_size".to_string())?,
        admission_deadline_ms: admission_deadline_ms
            .ok_or_else(|| "missing admission_deadline_ms".to_string())?,
        admission_queue_max: admission_queue_max
            .ok_or_else(|| "missing admission_queue_max".to_string())?,
        max_cache_cap: max_cache_cap.ok_or_else(|| "missing max_cache_cap".to_string())?,
        decode_cadence_mid_chunk_cap: decode_cadence_mid_chunk_cap
            .ok_or_else(|| "missing decode_cadence_mid_chunk_cap".to_string())?,
    })
}

fn parse_usize_value(key: &str, value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("invalid {key}: {err}"))
}

fn parse_u64_value(key: &str, value: &str) -> std::result::Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|err| format!("invalid {key}: {err}"))
}

fn default_candidate_configs() -> Vec<SchedulerAutotuneProfileConfig> {
    vec![
        SchedulerAutotuneProfileConfig {
            b_max: 1,
            prefill_chunk_size: 2048,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap: 256,
        },
        SchedulerAutotuneProfileConfig {
            b_max: 2,
            prefill_chunk_size: 1024,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap: 256,
        },
        SchedulerAutotuneProfileConfig {
            b_max: 2,
            prefill_chunk_size: 2048,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap: 256,
        },
        SchedulerAutotuneProfileConfig {
            b_max: 2,
            prefill_chunk_size: 1024,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap: 128,
        },
    ]
}

fn default_model_name(model: &Path) -> Result<String> {
    let name = model
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("--model-name is required when --model has no directory name")
        })?;
    Ok(name.to_string())
}

fn default_iron_bench_bin(ironmlx_bin: &Path) -> PathBuf {
    ironmlx_bin.with_file_name("iron-bench")
}

fn resolve_run_config(
    args: &SchedulerAutotuneCalibrateArgs,
    ironmlx_bin: &Path,
) -> Result<ResolvedRunConfig> {
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_DIR));
    let write_profile = args
        .write_profile
        .clone()
        .unwrap_or_else(|| output_dir.join(DEFAULT_RUNTIME_PROFILE_FILE));

    Ok(ResolvedRunConfig {
        model: args.model.clone(),
        model_name: match &args.model_name {
            Some(model_name) => model_name.clone(),
            None => default_model_name(&args.model)?,
        },
        iron_bench_bin: args
            .iron_bench_bin
            .clone()
            .unwrap_or_else(|| default_iron_bench_bin(ironmlx_bin)),
        output_dir,
        candidates: if args.candidates.is_empty() {
            default_candidate_configs()
        } else {
            args.candidates.clone()
        },
        prompt_len: if args.prompt_len.is_empty() {
            DEFAULT_PROMPT_LEN.to_vec()
        } else {
            args.prompt_len.clone()
        },
        max_tokens: args.max_tokens,
        concurrency: if args.concurrency.is_empty() {
            DEFAULT_CONCURRENCY.to_vec()
        } else {
            args.concurrency.clone()
        },
        selection_profile: args.selection_profile.into(),
        runs: args.runs,
        warmup: args.warmup,
        duration: args.duration,
        warmup_duration: args.warmup_duration,
        port: args.port,
        startup_timeout_sec: args.startup_timeout_sec,
        write_profile,
    })
}

pub fn run(args: SchedulerAutotuneCalibrateArgs) -> Result<()> {
    let ironmlx_bin = std::env::current_exe().context("locating current ironmlx executable")?;
    let resolved = resolve_run_config(&args, &ironmlx_bin)?;
    validate_matrix(&resolved)?;
    std::fs::create_dir_all(&resolved.output_dir)
        .with_context(|| format!("creating {}", resolved.output_dir.display()))?;

    let target_url = format!("http://127.0.0.1:{}", resolved.port);
    let health = health_url(resolved.port);
    let mut candidate_outputs = Vec::new();
    let benchmark_plan = build_candidate_benchmark_plan(&resolved);
    write_run_order_manifest(&resolved.output_dir, &benchmark_plan)?;

    for job in benchmark_plan {
        let serve_log = serve_log_path(&resolved.output_dir, job.candidate_idx, job.concurrency);
        let serve_invocation =
            build_serve_invocation(&ironmlx_bin, &resolved, job.config, resolved.port);
        let _serve = spawn_serve(&serve_invocation, &serve_log)?;

        wait_for_health(&health, Duration::from_secs(resolved.startup_timeout_sec))
            .with_context(|| format!("serve log: {}", serve_log.display()))?;

        let output_json =
            candidate_artifact_path(&resolved.output_dir, job.candidate_idx, job.concurrency);
        let stderr_log =
            candidate_stderr_log_path(&resolved.output_dir, job.candidate_idx, job.concurrency);
        let bench_invocation =
            build_iron_bench_invocation(&resolved, job.config, job.concurrency, &target_url);
        run_iron_bench(&bench_invocation, &output_json, &stderr_log)?;
        candidate_outputs.push(output_json);
    }

    let mut inputs = Vec::with_capacity(candidate_outputs.len());
    for path in &candidate_outputs {
        inputs.push(read_calibration(path)?);
    }
    let artifacts = FinalArtifactPaths::new(&resolved.output_dir, Some(resolved.write_profile));
    write_final_outputs(inputs, &artifacts, resolved.selection_profile)?;
    let stored_runtime_profile = artifacts.runtime_profile.as_ref().and_then(|path| {
        let stored = SchedulerProfileStore::default()
            .and_then(|store| persist_runtime_profile_from_artifact(&store, &resolved.model, path));
        match stored {
            Ok(path) => Some(path),
            Err(error) => {
                eprintln!(
                    "warning: failed to store runtime scheduler profile in ~/.ironmlx: {error:#}"
                );
                None
            }
        }
    });

    println!("calibration: {}", artifacts.calibration.display());
    println!("selection_json: {}", artifacts.selection_json.display());
    println!("selection_text: {}", artifacts.selection_text.display());
    if let Some(path) = &artifacts.runtime_profile {
        println!("runtime_profile: {}", path.display());
    }
    if let Some(path) = &stored_runtime_profile {
        println!("stored_runtime_profile: {}", path.display());
    }

    Ok(())
}

fn validate_matrix(args: &ResolvedRunConfig) -> Result<()> {
    if args.prompt_len.contains(&0) {
        anyhow::bail!("--prompt-len values must be > 0");
    }
    if args.concurrency.contains(&0) {
        anyhow::bail!("--concurrency values must be > 0");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CandidateBenchmarkJob {
    ordinal: usize,
    candidate_idx: usize,
    config: SchedulerAutotuneProfileConfig,
    concurrency: usize,
}

fn build_candidate_benchmark_plan(args: &ResolvedRunConfig) -> Vec<CandidateBenchmarkJob> {
    let mut jobs = Vec::with_capacity(args.candidates.len() * args.concurrency.len());
    for (concurrency_idx, &concurrency) in args.concurrency.iter().enumerate() {
        let mut candidate_indices = (0..args.candidates.len()).collect::<Vec<_>>();
        if concurrency_idx % 2 == 1 {
            candidate_indices.reverse();
        }
        for candidate_idx in candidate_indices {
            jobs.push(CandidateBenchmarkJob {
                ordinal: jobs.len(),
                candidate_idx,
                config: args.candidates[candidate_idx],
                concurrency,
            });
        }
    }
    jobs
}

#[derive(Debug, Serialize)]
struct RunOrderManifest {
    schema_version: u32,
    strategy: &'static str,
    jobs: Vec<RunOrderManifestJob>,
}

#[derive(Debug, Serialize)]
struct RunOrderManifestJob {
    ordinal: usize,
    candidate_idx: usize,
    concurrency: usize,
    config: SchedulerAutotuneProfileConfig,
    output_json: String,
    stderr_log: String,
    serve_log: String,
}

fn write_run_order_manifest(output_dir: &Path, jobs: &[CandidateBenchmarkJob]) -> Result<()> {
    let manifest = RunOrderManifest {
        schema_version: 1,
        strategy: RUN_ORDER_STRATEGY,
        jobs: jobs
            .iter()
            .map(|job| RunOrderManifestJob {
                ordinal: job.ordinal,
                candidate_idx: job.candidate_idx,
                concurrency: job.concurrency,
                config: job.config,
                output_json: artifact_file_name(candidate_artifact_path(
                    output_dir,
                    job.candidate_idx,
                    job.concurrency,
                )),
                stderr_log: artifact_file_name(candidate_stderr_log_path(
                    output_dir,
                    job.candidate_idx,
                    job.concurrency,
                )),
                serve_log: artifact_file_name(serve_log_path(
                    output_dir,
                    job.candidate_idx,
                    job.concurrency,
                )),
            })
            .collect(),
    };
    let output = serde_json::to_string_pretty(&manifest)?;
    let path = output_dir.join(RUN_ORDER_MANIFEST_FILE);
    std::fs::write(&path, format!("{output}\n"))
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn artifact_file_name(path: PathBuf) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .expect("artifact path should have a UTF-8 file name")
        .to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcessInvocation {
    program: PathBuf,
    args: Vec<String>,
}

fn candidate_artifact_path(output_dir: &Path, candidate_idx: usize, concurrency: usize) -> PathBuf {
    output_dir.join(format!("candidate-{candidate_idx:03}-c{concurrency}.json"))
}

fn candidate_stderr_log_path(
    output_dir: &Path,
    candidate_idx: usize,
    concurrency: usize,
) -> PathBuf {
    output_dir.join(format!(
        "candidate-{candidate_idx:03}-c{concurrency}.stderr.log"
    ))
}

fn serve_log_path(output_dir: &Path, candidate_idx: usize, concurrency: usize) -> PathBuf {
    output_dir.join(format!(
        "serve-candidate-{candidate_idx:03}-c{concurrency}.log"
    ))
}

fn build_serve_invocation(
    ironmlx_bin: &Path,
    args: &ResolvedRunConfig,
    config: SchedulerAutotuneProfileConfig,
    port: u16,
) -> ProcessInvocation {
    ProcessInvocation {
        program: ironmlx_bin.to_path_buf(),
        args: vec![
            "serve".to_string(),
            "--model".to_string(),
            args.model.to_string_lossy().into_owned(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            port.to_string(),
            "--prefill-chunk-size".to_string(),
            config.prefill_chunk_size.to_string(),
            "--b-max".to_string(),
            config.b_max.to_string(),
            "--admission-deadline-ms".to_string(),
            config.admission_deadline_ms.to_string(),
            "--admission-queue-max".to_string(),
            config.admission_queue_max.to_string(),
            "--max-cache-cap".to_string(),
            config.max_cache_cap.to_string(),
            "--decode-cadence-mid-chunk-cap".to_string(),
            config.decode_cadence_mid_chunk_cap.to_string(),
        ],
    }
}

fn build_iron_bench_invocation(
    args: &ResolvedRunConfig,
    config: SchedulerAutotuneProfileConfig,
    concurrency: usize,
    target_url: &str,
) -> ProcessInvocation {
    let mut invocation_args = vec![
        "--target".to_string(),
        format!("ironmlx={target_url}"),
        "--model-dir".to_string(),
        args.model.to_string_lossy().into_owned(),
        "--model".to_string(),
        args.model_name.clone(),
        "--prompt-len".to_string(),
        join_usize_csv(&args.prompt_len),
        "--max-tokens".to_string(),
        args.max_tokens.to_string(),
        "--format".to_string(),
        "autotune-json".to_string(),
        "--autotune-b-max".to_string(),
        config.b_max.to_string(),
        "--autotune-prefill-chunk-size".to_string(),
        config.prefill_chunk_size.to_string(),
        "--autotune-admission-deadline-ms".to_string(),
        config.admission_deadline_ms.to_string(),
        "--autotune-admission-queue-max".to_string(),
        config.admission_queue_max.to_string(),
        "--autotune-max-cache-cap".to_string(),
        config.max_cache_cap.to_string(),
        "--autotune-decode-cadence-mid-chunk-cap".to_string(),
        config.decode_cadence_mid_chunk_cap.to_string(),
    ];

    if concurrency == 1 {
        invocation_args.extend([
            "--runs".to_string(),
            args.runs.to_string(),
            "--warmup".to_string(),
            args.warmup.to_string(),
        ]);
    } else {
        invocation_args.extend([
            "--concurrent".to_string(),
            concurrency.to_string(),
            "--duration".to_string(),
            args.duration.to_string(),
            "--warmup-duration".to_string(),
            args.warmup_duration.to_string(),
        ]);
    }

    ProcessInvocation {
        program: args.iron_bench_bin.clone(),
        args: invocation_args,
    }
}

fn join_usize_csv(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FinalArtifactPaths {
    calibration: PathBuf,
    selection_json: PathBuf,
    selection_text: PathBuf,
    runtime_profile: Option<PathBuf>,
}

impl FinalArtifactPaths {
    fn new(output_dir: &Path, runtime_profile: Option<PathBuf>) -> Self {
        Self {
            calibration: output_dir.join("calibration.json"),
            selection_json: output_dir.join("selection.json"),
            selection_text: output_dir.join("selection.txt"),
            runtime_profile: Some(
                runtime_profile.unwrap_or_else(|| output_dir.join(DEFAULT_RUNTIME_PROFILE_FILE)),
            ),
        }
    }
}

fn health_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}/health")
}

fn wait_for_health(url: &str, timeout: Duration) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building scheduler-autotune health wait runtime")?;
    runtime.block_on(async move {
        let client = reqwest::Client::new();
        let deadline = Instant::now() + timeout;
        loop {
            match client.get(url).send().await {
                Ok(response) if response.status().is_success() => return Ok(()),
                _ if Instant::now() >= deadline => {
                    anyhow::bail!("timed out waiting for {url}");
                }
                _ => {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }
    })
}

struct ServeChild {
    child: Option<Child>,
}

impl Drop for ServeChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            if matches!(child.try_wait(), Ok(None)) {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

fn spawn_serve(invocation: &ProcessInvocation, log_path: &Path) -> Result<ServeChild> {
    let stdout = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(log_path)
        .with_context(|| format!("opening serve log {}", log_path.display()))?;
    let stderr = stdout
        .try_clone()
        .with_context(|| format!("cloning serve log {}", log_path.display()))?;
    let child = Command::new(&invocation.program)
        .args(&invocation.args)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .with_context(|| format!("spawning {}", invocation.program.display()))?;
    Ok(ServeChild { child: Some(child) })
}

fn run_iron_bench(
    invocation: &ProcessInvocation,
    output_json: &Path,
    stderr_log: &Path,
) -> Result<()> {
    let output = Command::new(&invocation.program)
        .args(&invocation.args)
        .output()
        .with_context(|| format!("running {}", invocation.program.display()))?;
    std::fs::write(output_json, &output.stdout)
        .with_context(|| format!("writing {}", output_json.display()))?;
    std::fs::write(stderr_log, &output.stderr)
        .with_context(|| format!("writing {}", stderr_log.display()))?;
    if !output.status.success() {
        anyhow::bail!(
            "iron-bench failed with status {}; stderr log: {}",
            output.status,
            stderr_log.display()
        );
    }
    Ok(())
}

fn read_calibration(path: &Path) -> Result<SchedulerAutotuneCalibrationInput> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn persist_runtime_profile_from_artifact(
    store: &SchedulerProfileStore,
    model_path: &Path,
    runtime_profile_path: &Path,
) -> Result<PathBuf> {
    let raw = std::fs::read_to_string(runtime_profile_path)
        .with_context(|| format!("reading {}", runtime_profile_path.display()))?;
    let profile = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {}", runtime_profile_path.display()))?;
    store.persist_profile(model_path, &profile)
}

fn write_final_outputs(
    inputs: Vec<SchedulerAutotuneCalibrationInput>,
    artifacts: &FinalArtifactPaths,
    selection_profile: SchedulerAutotuneSelectionProfile,
) -> Result<()> {
    let merged = merge_scheduler_autotune_calibrations(
        inputs,
        SchedulerAutotuneMergeOptions {
            require_complete_coverage: true,
        },
    )?;
    let calibration = serde_json::to_string_pretty(&merged)?;
    std::fs::write(&artifacts.calibration, format!("{calibration}\n"))
        .with_context(|| format!("writing {}", artifacts.calibration.display()))?;

    let selection = select_scheduler_autotune_profile_with_options(
        merged,
        SchedulerAutotuneSelectionOptions {
            profile: selection_profile,
        },
    );
    let selection_json = serde_json::to_string_pretty(&selection)?;
    std::fs::write(&artifacts.selection_json, format!("{selection_json}\n"))
        .with_context(|| format!("writing {}", artifacts.selection_json.display()))?;
    std::fs::write(&artifacts.selection_text, selection.render_text())
        .with_context(|| format!("writing {}", artifacts.selection_text.display()))?;

    if let Some(path) = &artifacts.runtime_profile {
        let profile = build_scheduler_autotune_runtime_profile(&selection)?;
        let output = serde_json::to_string_pretty(&profile)?;
        std::fs::write(path, format!("{output}\n"))
            .with_context(|| format!("writing {}", path.display()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        build_candidate_benchmark_plan, build_iron_bench_invocation, build_serve_invocation,
        candidate_artifact_path, health_url, parse_candidate_config,
        persist_runtime_profile_from_artifact, resolve_run_config, write_final_outputs,
        write_run_order_manifest, FinalArtifactPaths, SchedulerAutotuneCalibrateArgs,
    };
    use crate::cli::scheduler_autotune::SchedulerAutotuneSelectionProfileArg;
    use crate::cli::scheduler_profile_store::SchedulerProfileStore;
    use crate::core::scheduler_autotune::{
        SchedulerAutotuneCalibrationInput, SchedulerAutotuneMeasurement,
        SchedulerAutotuneObjective, SchedulerAutotuneProfileConfig,
        SchedulerAutotuneSelectionProfile, SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
    };

    #[test]
    fn parse_candidate_config_accepts_all_required_fields() {
        let config = parse_candidate_config(
            "b_max=2,prefill_chunk_size=1024,admission_deadline_ms=5,admission_queue_max=32,max_cache_cap=32768,decode_cadence_mid_chunk_cap=256",
        )
        .expect("candidate should parse");

        assert_eq!(config.b_max, 2);
        assert_eq!(config.prefill_chunk_size, 1024);
        assert_eq!(config.admission_deadline_ms, 5);
        assert_eq!(config.admission_queue_max, 32);
        assert_eq!(config.max_cache_cap, 32768);
        assert_eq!(config.decode_cadence_mid_chunk_cap, 256);
    }

    #[test]
    fn parse_candidate_config_rejects_missing_field() {
        let err = parse_candidate_config(
            "b_max=2,prefill_chunk_size=1024,admission_deadline_ms=5,admission_queue_max=32,max_cache_cap=32768",
        )
        .expect_err("missing decode_cadence_mid_chunk_cap should fail");

        assert!(
            err.contains("decode_cadence_mid_chunk_cap"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn candidate_artifact_path_includes_candidate_and_concurrency() {
        let path = candidate_artifact_path(Path::new("/tmp/out"), 3, 2);
        assert_eq!(path.to_string_lossy(), "/tmp/out/candidate-003-c2.json");
    }

    #[test]
    fn build_serve_invocation_includes_scheduler_config() {
        let args = sample_resolved_config();
        let command =
            build_serve_invocation(Path::new("/tmp/ironmlx"), &args, profile_config(), 19000);

        assert_eq!(command.program.to_string_lossy(), "/tmp/ironmlx");
        assert!(command.args.contains(&"serve".to_string()));
        assert!(command.args.contains(&"--b-max".to_string()));
        assert!(command.args.contains(&"2".to_string()));
        assert!(command.args.contains(&"--prefill-chunk-size".to_string()));
        assert!(command.args.contains(&"1024".to_string()));
        assert!(command.args.contains(&"--port".to_string()));
        assert!(command.args.contains(&"19000".to_string()));
        assert!(command
            .args
            .contains(&"--decode-cadence-mid-chunk-cap".to_string()));
        assert!(command.args.contains(&"256".to_string()));
    }

    #[test]
    fn build_iron_bench_invocation_uses_sequential_mode_for_concurrency_one() {
        let args = sample_resolved_config();
        let command =
            build_iron_bench_invocation(&args, profile_config(), 1, "http://127.0.0.1:18080");

        assert!(!command.args.contains(&"--concurrent".to_string()));
        assert!(command.args.contains(&"--runs".to_string()));
        assert!(command.args.contains(&"5".to_string()));
        assert!(command.args.contains(&"--warmup".to_string()));
        assert!(command.args.contains(&"1".to_string()));
    }

    #[test]
    fn build_iron_bench_invocation_uses_concurrent_mode_for_concurrency_above_one() {
        let args = sample_resolved_config();
        let command =
            build_iron_bench_invocation(&args, profile_config(), 2, "http://127.0.0.1:18080");

        assert!(command.args.contains(&"--concurrent".to_string()));
        assert!(command.args.contains(&"2".to_string()));
        assert!(command.args.contains(&"--duration".to_string()));
        assert!(command.args.contains(&"30".to_string()));
        assert!(command.args.contains(&"--warmup-duration".to_string()));
        assert!(command.args.contains(&"5".to_string()));
        assert!(command
            .args
            .contains(&"--autotune-decode-cadence-mid-chunk-cap".to_string()));
        assert!(command.args.contains(&"256".to_string()));
    }

    #[test]
    fn candidate_benchmark_plan_mirrors_candidate_order_across_concurrency_levels() {
        let mut args = sample_resolved_config();
        args.candidates = vec![
            profile_config_with_chunk_and_cap(1024, 128),
            profile_config_with_chunk_and_cap(2048, 256),
            profile_config_with_chunk_and_cap(4096, 512),
        ];
        args.concurrency = vec![1, 2];

        let plan = build_candidate_benchmark_plan(&args);
        let observed = plan
            .iter()
            .map(|job| (job.ordinal, job.candidate_idx, job.concurrency))
            .collect::<Vec<_>>();

        assert_eq!(
            observed,
            vec![
                (0, 0, 1),
                (1, 1, 1),
                (2, 2, 1),
                (3, 2, 2),
                (4, 1, 2),
                (5, 0, 2)
            ]
        );
    }

    #[test]
    fn health_url_uses_localhost_and_selected_port() {
        assert_eq!(health_url(19000), "http://127.0.0.1:19000/health");
    }

    #[test]
    fn final_artifact_paths_are_stable() {
        let paths = FinalArtifactPaths::new(
            Path::new("/tmp/out"),
            Some(PathBuf::from("/tmp/profile.json")),
        );

        assert_eq!(
            paths.calibration.to_string_lossy(),
            "/tmp/out/calibration.json"
        );
        assert_eq!(
            paths.selection_json.to_string_lossy(),
            "/tmp/out/selection.json"
        );
        assert_eq!(
            paths.selection_text.to_string_lossy(),
            "/tmp/out/selection.txt"
        );
        assert_eq!(
            paths
                .runtime_profile
                .as_ref()
                .expect("profile")
                .to_string_lossy(),
            "/tmp/profile.json"
        );
    }

    #[test]
    fn final_artifact_paths_default_runtime_profile_to_output_dir() {
        let paths = FinalArtifactPaths::new(Path::new("/tmp/out"), None);

        assert_eq!(
            paths
                .runtime_profile
                .as_ref()
                .expect("profile")
                .to_string_lossy(),
            "/tmp/out/scheduler-profile.json"
        );
    }

    #[test]
    fn write_run_order_manifest_records_planned_jobs_and_artifacts() {
        let temp_dir = unique_temp_dir("scheduler-autotune-run-order");
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        let mut args = sample_resolved_config();
        args.candidates = vec![
            profile_config_with_chunk_and_cap(1024, 128),
            profile_config_with_chunk_and_cap(2048, 256),
        ];
        args.concurrency = vec![1, 2];
        args.output_dir = temp_dir.clone();
        let plan = build_candidate_benchmark_plan(&args);

        write_run_order_manifest(&args.output_dir, &plan).expect("write manifest");

        let raw = std::fs::read_to_string(temp_dir.join("run-order.json")).expect("read manifest");
        let json: serde_json::Value = serde_json::from_str(&raw).expect("manifest json");

        assert_eq!(json["schema_version"], 1);
        assert_eq!(
            json["strategy"],
            "concurrency-major-mirrored-candidate-order"
        );
        assert_eq!(json["jobs"].as_array().expect("jobs").len(), 4);
        assert_eq!(json["jobs"][0]["ordinal"], 0);
        assert_eq!(json["jobs"][0]["candidate_idx"], 0);
        assert_eq!(json["jobs"][0]["concurrency"], 1);
        assert_eq!(json["jobs"][0]["config"]["prefill_chunk_size"], 1024);
        assert_eq!(json["jobs"][0]["output_json"], "candidate-000-c1.json");
        assert_eq!(json["jobs"][0]["stderr_log"], "candidate-000-c1.stderr.log");
        assert_eq!(json["jobs"][0]["serve_log"], "serve-candidate-000-c1.log");
        assert_eq!(json["jobs"][2]["candidate_idx"], 1);
        assert_eq!(json["jobs"][2]["concurrency"], 2);
        assert_eq!(json["jobs"][3]["candidate_idx"], 0);
        assert_eq!(json["jobs"][3]["concurrency"], 2);

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn resolve_run_config_supplies_full_auto_defaults() {
        let mut args = sample_args();
        args.model = PathBuf::from("/models/GLM-4.7-flash-4bit");
        args.model_name = None;
        args.iron_bench_bin = None;
        args.output_dir = None;
        args.candidates.clear();
        args.prompt_len.clear();
        args.concurrency.clear();
        args.write_profile = None;

        let resolved =
            resolve_run_config(&args, Path::new("/opt/ironmlx/bin/ironmlx")).expect("resolve");

        assert_eq!(resolved.model_name, "GLM-4.7-flash-4bit");
        assert_eq!(
            resolved.iron_bench_bin.to_string_lossy(),
            "/opt/ironmlx/bin/iron-bench"
        );
        assert_eq!(
            resolved.output_dir.to_string_lossy(),
            "reports/scheduler-autotune"
        );
        assert_eq!(
            resolved.write_profile.to_string_lossy(),
            "reports/scheduler-autotune/scheduler-profile.json"
        );
        assert_eq!(resolved.prompt_len, vec![1024, 4096]);
        assert_eq!(resolved.concurrency, vec![1, 2]);
        assert_eq!(
            resolved.selection_profile,
            SchedulerAutotuneSelectionProfile::AgentLongPrompt
        );
        assert_eq!(resolved.candidates.len(), 4);
        assert!(resolved.candidates.iter().any(|config| config.b_max == 1));
        assert!(resolved.candidates.iter().any(|config| config.b_max == 2));
        assert!(resolved
            .candidates
            .iter()
            .any(|config| config.decode_cadence_mid_chunk_cap == 128));
        assert!(resolved
            .candidates
            .iter()
            .any(|config| config.decode_cadence_mid_chunk_cap == 256));
    }

    #[test]
    fn resolve_run_config_keeps_explicit_overrides() {
        let args = sample_args();

        let resolved =
            resolve_run_config(&args, Path::new("/opt/ironmlx/bin/ironmlx")).expect("resolve");

        assert_eq!(resolved.model_name, "GLM-4.7-flash-4bit");
        assert_eq!(resolved.iron_bench_bin.to_string_lossy(), "/tmp/iron-bench");
        assert_eq!(resolved.output_dir.to_string_lossy(), "/tmp/out");
        assert_eq!(
            resolved.write_profile.to_string_lossy(),
            "/tmp/profile.json"
        );
        assert_eq!(resolved.prompt_len, vec![1024, 2048]);
        assert_eq!(resolved.concurrency, vec![1, 2]);
        assert_eq!(
            resolved.selection_profile,
            SchedulerAutotuneSelectionProfile::AgentLongPrompt
        );
        assert_eq!(resolved.candidates, vec![profile_config()]);
    }

    #[test]
    fn write_final_outputs_writes_calibration_selection_and_profile() {
        let temp_dir = unique_temp_dir("scheduler-autotune-calibrate-output");
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let profile_path = temp_dir.join("scheduler-profile.json");
        let paths = FinalArtifactPaths::new(&temp_dir, Some(profile_path.clone()));

        write_final_outputs(
            vec![sample_calibration(profile_config())],
            &paths,
            SchedulerAutotuneSelectionProfile::AgentLongPrompt,
        )
        .expect("write final outputs");

        let calibration =
            std::fs::read_to_string(&paths.calibration).expect("read calibration output");
        let selection =
            std::fs::read_to_string(&paths.selection_json).expect("read selection json");
        let selection_text =
            std::fs::read_to_string(&paths.selection_text).expect("read selection text");
        let profile = std::fs::read_to_string(&profile_path).expect("read runtime profile");

        assert!(calibration.contains("\"measurements\""));
        assert!(selection.contains("\"selected\""));
        assert!(selection_text.contains("scheduler/autotune profile selection"));
        assert!(profile.contains("\"prefill_chunk_size\": 1024"));

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn persist_runtime_profile_from_artifact_writes_profile_store() {
        let temp_dir = unique_temp_dir("scheduler-autotune-profile-store");
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let profile_path = temp_dir.join("scheduler-profile.json");
        let runtime_profile = crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfile {
            schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
            model_name: "GLM-4.7-Flash-4bit".to_string(),
            hardware_label: "m5-max-128gb".to_string(),
            config: profile_config(),
            rules: Vec::new(),
        };
        let output = serde_json::to_string_pretty(&runtime_profile).expect("serialize profile");
        std::fs::write(&profile_path, format!("{output}\n")).expect("write profile artifact");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));

        let stored_path = persist_runtime_profile_from_artifact(&store, &model_dir, &profile_path)
            .expect("persist runtime profile");

        assert_eq!(
            stored_path,
            store.profile_path("GLM-4.7-Flash-4bit", "m5-max-128gb", &model_dir)
        );
        assert!(stored_path.exists());
        assert_eq!(
            store
                .find_profile(&model_dir, "GLM-4.7-Flash-4bit", "m5-max-128gb")
                .expect("find profile")
                .expect("stored profile should match"),
            stored_path
        );

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    fn sample_args() -> SchedulerAutotuneCalibrateArgs {
        SchedulerAutotuneCalibrateArgs {
            model: PathBuf::from("/tmp/model"),
            model_name: Some("GLM-4.7-flash-4bit".to_string()),
            iron_bench_bin: Some(PathBuf::from("/tmp/iron-bench")),
            output_dir: Some(PathBuf::from("/tmp/out")),
            candidates: vec![profile_config()],
            prompt_len: vec![1024, 2048],
            max_tokens: 128,
            concurrency: vec![1, 2],
            selection_profile: SchedulerAutotuneSelectionProfileArg::AgentLongPrompt,
            runs: 5,
            warmup: 1,
            duration: 30,
            warmup_duration: 5,
            port: 18080,
            startup_timeout_sec: 300,
            write_profile: Some(PathBuf::from("/tmp/profile.json")),
        }
    }

    fn sample_resolved_config() -> super::ResolvedRunConfig {
        resolve_run_config(&sample_args(), Path::new("/tmp/ironmlx")).expect("resolve sample")
    }

    fn profile_config() -> SchedulerAutotuneProfileConfig {
        profile_config_with_chunk_and_cap(1024, 256)
    }

    fn profile_config_with_chunk_and_cap(
        prefill_chunk_size: usize,
        decode_cadence_mid_chunk_cap: usize,
    ) -> SchedulerAutotuneProfileConfig {
        SchedulerAutotuneProfileConfig {
            b_max: 2,
            prefill_chunk_size,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap,
        }
    }

    fn sample_calibration(
        config: SchedulerAutotuneProfileConfig,
    ) -> SchedulerAutotuneCalibrationInput {
        SchedulerAutotuneCalibrationInput {
            schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
            model_name: "GLM-4.7-flash-4bit".to_string(),
            hardware_label: "m5-max-128g".to_string(),
            objective: SchedulerAutotuneObjective::agent_default(),
            measurements: vec![SchedulerAutotuneMeasurement {
                config,
                prompt_len: 2048,
                max_new_tokens: 128,
                concurrency: 1,
                ttft_ms_p95: 120.0,
                itl_ms_p95: 12.0,
                e2e_s_p95: 2.5,
                tokens_per_sec: 90.0,
                early_itl_ms_p95: 12.0,
                memory_budget_ok: true,
                cached_tokens_warning: false,
            }],
        }
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        std::env::temp_dir().join(format!("{label}-{}-{nanos}", std::process::id()))
    }
}
