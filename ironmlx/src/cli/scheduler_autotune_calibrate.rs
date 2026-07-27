use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Args;
use serde::Serialize;

use super::scheduler_profile_context::SchedulerProfileRuntimeArgs;
use super::scheduler_profile_store::SchedulerProfileStore;
use crate::core::scheduler_autotune::{
    build_scheduler_autotune_runtime_profile, merge_scheduler_autotune_calibrations,
    select_scheduler_autotune_profile_with_options, SchedulerAutotuneCacheState,
    SchedulerAutotuneCalibrationInput, SchedulerAutotuneMergeOptions,
    SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeContext,
    SchedulerAutotuneSelectionOptions, SchedulerAutotuneSelectionProfile, SchedulerSpeculativeMode,
};
use crate::core::server::chat_format::{render_and_encode, ChatMessage};
use crate::core::Tokenizer;
use crate::Result;

const DEFAULT_PORT: u16 = 18080;
const DEFAULT_STARTUP_TIMEOUT_SEC: u64 = 300;
const DEFAULT_OUTPUT_DIR: &str = "reports/scheduler-autotune";
const DEFAULT_RUNTIME_PROFILE_FILE: &str = "scheduler-profile.json";
const DEFAULT_CONCURRENCY: &[usize] = &[1, 2, 4, 8];
const RUNTIME_CONTEXT_FILE: &str = "runtime-context.json";
const RUN_ORDER_MANIFEST_FILE: &str = "run-order.json";
const RUN_ORDER_STRATEGY: &str = "concurrency-major-mirrored-candidate-order";
const BENCHMARK_PROMPT_TOKEN_SAFETY_MARGIN: usize = 8;
const BENCHMARK_STDERR_SUMMARY_MAX_CHARS: usize = 512;
const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(100);
static SIGNAL_CANCELLATION_REQUESTED: AtomicBool = AtomicBool::new(false);
const BENCHMARK_PROMPT_SAMPLE: &str =
    "Benchmark request 0 \u{2014} The quick brown fox jumps over the lazy dog.";

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

    /// Prompt content token lengths to test. Chat-template capacity is
    /// reserved automatically.
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

    #[command(flatten)]
    pub runtime: SchedulerProfileRuntimeArgs,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedRunConfig {
    model: PathBuf,
    model_name: String,
    iron_bench_bin: PathBuf,
    output_dir: PathBuf,
    candidates: Vec<SchedulerAutotuneProfileConfig>,
    prompt_len: Vec<usize>,
    prompt_token_reserve: usize,
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
    runtime: SchedulerProfileRuntimeArgs,
    runtime_context: SchedulerAutotuneRuntimeContext,
    runtime_context_path: PathBuf,
    cache_states: Vec<SchedulerAutotuneCacheState>,
}

#[derive(Clone, Debug)]
struct CalibrationCancellation {
    requested: Arc<AtomicBool>,
}

impl CalibrationCancellation {
    fn install() -> Result<Self> {
        SIGNAL_CANCELLATION_REQUESTED.store(false, Ordering::Release);
        let requested = Arc::new(AtomicBool::new(false));
        install_cancellation_signal(libc::SIGINT)?;
        install_cancellation_signal(libc::SIGTERM)?;
        Ok(Self { requested })
    }

    #[cfg(test)]
    fn requested() -> Self {
        Self {
            requested: Arc::new(AtomicBool::new(true)),
        }
    }

    fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
            || SIGNAL_CANCELLATION_REQUESTED.load(Ordering::Acquire)
    }

    fn check(&self) -> Result<()> {
        if self.is_requested() {
            return Err(CalibrationCancelled.into());
        }
        Ok(())
    }
}

extern "C" fn request_calibration_cancellation(_signal: libc::c_int) {
    SIGNAL_CANCELLATION_REQUESTED.store(true, Ordering::Release);
}

fn install_cancellation_signal(signal: libc::c_int) -> Result<()> {
    // SAFETY: the handler only performs an atomic store, which is
    // async-signal-safe on the supported Apple Silicon targets.
    let previous = unsafe {
        libc::signal(
            signal,
            request_calibration_cancellation as *const () as libc::sighandler_t,
        )
    };
    if previous == libc::SIG_ERR {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("registering scheduler-autotune signal {signal}"));
    }
    Ok(())
}

#[derive(Debug)]
struct CalibrationCancelled;

impl std::fmt::Display for CalibrationCancelled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("scheduler calibration cancelled")
    }
}

impl std::error::Error for CalibrationCancelled {}

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

fn default_candidate_configs(
    runtime_context: &SchedulerAutotuneRuntimeContext,
) -> Vec<SchedulerAutotuneProfileConfig> {
    let b_max_values: &[usize] = match runtime_context.speculative.mode {
        SchedulerSpeculativeMode::QwenMtp | SchedulerSpeculativeMode::QwenMtpPromptLookup => {
            &[1, 2]
        }
        SchedulerSpeculativeMode::Disabled
        | SchedulerSpeculativeMode::Gemma4Drafter
        | SchedulerSpeculativeMode::Gemma4DrafterPromptLookup
        | SchedulerSpeculativeMode::PromptLookup => &[1, 2, 4],
    };
    let mut candidates = Vec::new();
    for &b_max in b_max_values {
        for prefill_chunk_size in [1024, 2048] {
            for decode_cadence_mid_chunk_cap in [128, 256] {
                candidates.push(SchedulerAutotuneProfileConfig {
                    b_max,
                    prefill_chunk_size,
                    admission_deadline_ms: 5,
                    admission_queue_max: 32,
                    max_cache_cap: runtime_context.logical_kv_cap_tokens,
                    decode_cadence_mid_chunk_cap,
                });
            }
        }
    }
    candidates
}

fn default_prompt_lengths(
    max_cache_cap: usize,
    max_tokens: usize,
    prompt_token_reserve: usize,
) -> Vec<usize> {
    let largest = max_cache_cap
        .saturating_sub(max_tokens)
        .saturating_sub(prompt_token_reserve)
        .clamp(1, 32768);
    let mut lengths = vec![1024.min(largest), 8192.min(largest), largest];
    lengths.sort_unstable();
    lengths.dedup();
    lengths
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
    let runtime_context = args.runtime.context_for_model(&args.model)?;
    let prompt_token_reserve = benchmark_prompt_token_reserve(&args.model)?;
    resolve_run_config_with_context(args, ironmlx_bin, runtime_context, prompt_token_reserve)
}

fn benchmark_prompt_token_reserve(model_dir: &Path) -> Result<usize> {
    let tokenizer = Tokenizer::from_model_dir(model_dir).with_context(|| {
        format!(
            "loading tokenizer for scheduler-autotune prompt capacity from {}",
            model_dir.display()
        )
    })?;
    let content_tokens = tokenizer.encode(BENCHMARK_PROMPT_SAMPLE, false)?.len();
    let messages = [ChatMessage::text("user", BENCHMARK_PROMPT_SAMPLE)];
    let template_kwargs = serde_json::json!({"enable_thinking": false});
    let prompt_tokens = render_and_encode(&tokenizer, &messages, Some(&template_kwargs))?.len();
    Ok(prompt_tokens
        .saturating_sub(content_tokens)
        .saturating_add(BENCHMARK_PROMPT_TOKEN_SAFETY_MARGIN))
}

fn resolve_run_config_with_context(
    args: &SchedulerAutotuneCalibrateArgs,
    ironmlx_bin: &Path,
    runtime_context: SchedulerAutotuneRuntimeContext,
    prompt_token_reserve: usize,
) -> Result<ResolvedRunConfig> {
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_DIR));
    let write_profile = args
        .write_profile
        .clone()
        .unwrap_or_else(|| output_dir.join(DEFAULT_RUNTIME_PROFILE_FILE));
    let cache_states = if runtime_context.prefix_cache.enabled {
        vec![
            SchedulerAutotuneCacheState::Cold,
            SchedulerAutotuneCacheState::Warm,
        ]
    } else {
        vec![SchedulerAutotuneCacheState::Cold]
    };
    let prompt_len = if args.prompt_len.is_empty() {
        default_prompt_lengths(
            runtime_context.logical_kv_cap_tokens,
            args.max_tokens,
            prompt_token_reserve,
        )
    } else {
        args.prompt_len.clone()
    };

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
        output_dir: output_dir.clone(),
        candidates: if args.candidates.is_empty() {
            default_candidate_configs(&runtime_context)
        } else {
            args.candidates.clone()
        },
        prompt_len,
        prompt_token_reserve,
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
        runtime: args.runtime.clone(),
        runtime_context_path: output_dir.join(RUNTIME_CONTEXT_FILE),
        runtime_context,
        cache_states,
    })
}

fn write_runtime_context(args: &ResolvedRunConfig) -> Result<()> {
    let output = serde_json::to_string_pretty(&args.runtime_context)?;
    std::fs::write(&args.runtime_context_path, format!("{output}\n")).with_context(|| {
        format!(
            "writing scheduler runtime context {}",
            args.runtime_context_path.display()
        )
    })
}

pub fn run(args: SchedulerAutotuneCalibrateArgs) -> Result<()> {
    let cancellation = CalibrationCancellation::install()?;
    let result = run_with_cancellation(args, &cancellation);
    if result
        .as_ref()
        .is_err_and(|error| error.downcast_ref::<CalibrationCancelled>().is_some())
    {
        eprintln!("calibration_cancelled: scheduler profile calibration was cancelled");
    }
    result
}

fn run_with_cancellation(
    args: SchedulerAutotuneCalibrateArgs,
    cancellation: &CalibrationCancellation,
) -> Result<()> {
    cancellation.check()?;
    let ironmlx_bin = std::env::current_exe().context("locating current ironmlx executable")?;
    let resolved = resolve_run_config(&args, &ironmlx_bin)?;
    cancellation.check()?;
    validate_matrix(&resolved)?;
    std::fs::create_dir_all(&resolved.output_dir)
        .with_context(|| format!("creating {}", resolved.output_dir.display()))?;
    write_runtime_context(&resolved)?;

    let target_url = format!("http://127.0.0.1:{}", resolved.port);
    let health = health_url(resolved.port);
    let mut candidate_outputs = Vec::new();
    let benchmark_plan = build_candidate_benchmark_plan(&resolved);
    write_run_order_manifest(&resolved.output_dir, &benchmark_plan)?;
    let total_jobs = benchmark_plan.len();

    for job in benchmark_plan {
        cancellation.check()?;
        let job_started = Instant::now();
        eprintln!(
            "[scheduler-autotune] job {}/{} stage=starting candidate={} concurrency={} cache={}",
            job.ordinal + 1,
            total_jobs,
            job.candidate_idx,
            job.concurrency,
            cache_state_label(job.cache_state),
        );
        let serve_log = serve_log_path(
            &resolved.output_dir,
            job.candidate_idx,
            job.concurrency,
            job.cache_state,
        );
        let serve_invocation =
            build_serve_invocation(&ironmlx_bin, &resolved, job.config, resolved.port);
        let _serve = spawn_serve(&serve_invocation, &serve_log)?;

        wait_for_health(
            &health,
            Duration::from_secs(resolved.startup_timeout_sec),
            cancellation,
        )
        .with_context(|| format!("serve log: {}", serve_log.display()))?;
        cancellation.check()?;

        let output_json = candidate_artifact_path(
            &resolved.output_dir,
            job.candidate_idx,
            job.concurrency,
            job.cache_state,
        );
        let stderr_log = candidate_stderr_log_path(
            &resolved.output_dir,
            job.candidate_idx,
            job.concurrency,
            job.cache_state,
        );
        let bench_invocation = build_iron_bench_invocation(
            &resolved,
            job.config,
            job.concurrency,
            job.cache_state,
            &target_url,
        );
        eprintln!(
            "[scheduler-autotune] job {}/{} stage=benchmarking candidate={} concurrency={} cache={}",
            job.ordinal + 1,
            total_jobs,
            job.candidate_idx,
            job.concurrency,
            cache_state_label(job.cache_state),
        );
        run_iron_bench(&bench_invocation, &output_json, &stderr_log, cancellation)?;
        candidate_outputs.push(output_json);
        eprintln!(
            "[scheduler-autotune] job {}/{} stage=completed elapsed_s={:.1}",
            job.ordinal + 1,
            total_jobs,
            job_started.elapsed().as_secs_f64(),
        );
    }

    cancellation.check()?;
    eprintln!("[scheduler-autotune] stage=finalizing completed_jobs={total_jobs}/{total_jobs}");
    let mut inputs = Vec::with_capacity(candidate_outputs.len());
    for path in &candidate_outputs {
        inputs.push(read_calibration(path)?);
    }
    let artifacts = FinalArtifactPaths::new(&resolved.output_dir, Some(resolved.write_profile));
    write_final_outputs(inputs, &artifacts, resolved.selection_profile)?;
    cancellation.check()?;
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
    if args.prompt_len.iter().any(|prompt_len| {
        prompt_len
            .saturating_add(args.max_tokens)
            .saturating_add(args.prompt_token_reserve)
            > args.runtime.max_cache_cap
    }) {
        anyhow::bail!(
            "every prompt length plus --max-tokens and the model chat-template reserve ({}) must fit within --max-cache-cap={}",
            args.prompt_token_reserve,
            args.runtime.max_cache_cap,
        );
    }
    if let Some(candidate) = args
        .candidates
        .iter()
        .find(|candidate| candidate.max_cache_cap != args.runtime.max_cache_cap)
    {
        anyhow::bail!(
            "candidate max_cache_cap={} does not match runtime context --max-cache-cap={}",
            candidate.max_cache_cap,
            args.runtime.max_cache_cap
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CandidateBenchmarkJob {
    ordinal: usize,
    candidate_idx: usize,
    config: SchedulerAutotuneProfileConfig,
    concurrency: usize,
    cache_state: SchedulerAutotuneCacheState,
}

fn build_candidate_benchmark_plan(args: &ResolvedRunConfig) -> Vec<CandidateBenchmarkJob> {
    let mut jobs = Vec::with_capacity(
        args.candidates.len() * args.concurrency.len() * args.cache_states.len(),
    );
    for &cache_state in &args.cache_states {
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
                    cache_state,
                });
            }
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
    cache_state: SchedulerAutotuneCacheState,
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
                cache_state: job.cache_state,
                config: job.config,
                output_json: artifact_file_name(candidate_artifact_path(
                    output_dir,
                    job.candidate_idx,
                    job.concurrency,
                    job.cache_state,
                )),
                stderr_log: artifact_file_name(candidate_stderr_log_path(
                    output_dir,
                    job.candidate_idx,
                    job.concurrency,
                    job.cache_state,
                )),
                serve_log: artifact_file_name(serve_log_path(
                    output_dir,
                    job.candidate_idx,
                    job.concurrency,
                    job.cache_state,
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

fn cache_state_label(cache_state: SchedulerAutotuneCacheState) -> &'static str {
    match cache_state {
        SchedulerAutotuneCacheState::Cold => "cold",
        SchedulerAutotuneCacheState::Warm => "warm",
    }
}

fn candidate_artifact_path(
    output_dir: &Path,
    candidate_idx: usize,
    concurrency: usize,
    cache_state: SchedulerAutotuneCacheState,
) -> PathBuf {
    output_dir.join(format!(
        "candidate-{candidate_idx:03}-c{concurrency}-{}.json",
        cache_state_label(cache_state)
    ))
}

fn candidate_stderr_log_path(
    output_dir: &Path,
    candidate_idx: usize,
    concurrency: usize,
    cache_state: SchedulerAutotuneCacheState,
) -> PathBuf {
    output_dir.join(format!(
        "candidate-{candidate_idx:03}-c{concurrency}-{}.stderr.log",
        cache_state_label(cache_state)
    ))
}

fn serve_log_path(
    output_dir: &Path,
    candidate_idx: usize,
    concurrency: usize,
    cache_state: SchedulerAutotuneCacheState,
) -> PathBuf {
    output_dir.join(format!(
        "serve-candidate-{candidate_idx:03}-c{concurrency}-{}.log",
        cache_state_label(cache_state)
    ))
}

fn build_serve_invocation(
    ironmlx_bin: &Path,
    args: &ResolvedRunConfig,
    config: SchedulerAutotuneProfileConfig,
    port: u16,
) -> ProcessInvocation {
    let mut invocation_args = vec![
        "serve".to_string(),
        "--model".to_string(),
        args.model.to_string_lossy().into_owned(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--prefill-chunk-size".to_string(),
        config.prefill_chunk_size.to_string(),
        "--max-sequences".to_string(),
        config.b_max.to_string(),
        "--admission-deadline-ms".to_string(),
        config.admission_deadline_ms.to_string(),
        "--admission-queue-max".to_string(),
        config.admission_queue_max.to_string(),
        "--max-cache-cap".to_string(),
        config.max_cache_cap.to_string(),
        "--decode-cadence-mid-chunk-cap".to_string(),
        config.decode_cadence_mid_chunk_cap.to_string(),
        "--kv-quant".to_string(),
        args.runtime.kv_quant.cli_value().to_string(),
    ];
    if let Some(path) = &args.runtime.mtp_model_dir {
        invocation_args.extend([
            "--mtp-model-dir".to_string(),
            path.to_string_lossy().into_owned(),
        ]);
    }
    if let Some(draft_tokens) = args.runtime.mtp_draft_tokens {
        invocation_args.extend(["--mtp-draft-tokens".to_string(), draft_tokens.to_string()]);
    }
    if let Some(path) = &args.runtime.paged_prefix_cache_dir {
        invocation_args.extend([
            "--paged-prefix-cache-dir".to_string(),
            path.to_string_lossy().into_owned(),
            "--paged-prefix-cache-block-size".to_string(),
            args.runtime.paged_prefix_cache_block_size.to_string(),
        ]);
    }
    if let Some(max_pages) = args.runtime.paged_prefix_cache_max_pages {
        invocation_args.extend([
            "--paged-prefix-cache-max-pages".to_string(),
            max_pages.to_string(),
        ]);
    }
    if let Some(max_bytes) = args.runtime.prefix_lru_cache_max_bytes {
        invocation_args.extend([
            "--prefix-lru-cache-max-bytes".to_string(),
            max_bytes.to_string(),
        ]);
    }
    if let Some(max_gb) = args.runtime.ssd_prefix_cache_max_gb {
        invocation_args.extend(["--ssd-prefix-cache-max-gb".to_string(), max_gb.to_string()]);
    }
    if args.runtime.active_kv_offload {
        invocation_args.push("--active-kv-offload".to_string());
    }
    if let Some(limit_gb) = args.runtime.memory_limit_total_gb {
        invocation_args.extend(["--memory-limit-total-gb".to_string(), limit_gb.to_string()]);
    }
    if let Some(limit_gb) = args.runtime.memory_limit_model_gb {
        invocation_args.extend(["--memory-limit-model-gb".to_string(), limit_gb.to_string()]);
    }
    ProcessInvocation {
        program: ironmlx_bin.to_path_buf(),
        args: invocation_args,
    }
}

fn build_iron_bench_invocation(
    args: &ResolvedRunConfig,
    config: SchedulerAutotuneProfileConfig,
    concurrency: usize,
    cache_state: SchedulerAutotuneCacheState,
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
        "--autotune-runtime-context".to_string(),
        args.runtime_context_path.to_string_lossy().into_owned(),
        "--autotune-cache-state".to_string(),
        cache_state_label(cache_state).to_string(),
    ];

    if cache_state == SchedulerAutotuneCacheState::Warm {
        invocation_args.push("--prefix-cache-probe".to_string());
    }

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

fn wait_for_health(
    url: &str,
    timeout: Duration,
    cancellation: &CalibrationCancellation,
) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building scheduler-autotune health wait runtime")?;
    runtime.block_on(async move {
        let client = reqwest::Client::new();
        let deadline = Instant::now() + timeout;
        loop {
            cancellation.check()?;
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

struct ManagedChild {
    child: Option<Child>,
}

impl ManagedChild {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        let Some(child) = self.child.as_mut() else {
            return Ok(None);
        };
        let status = child.try_wait()?;
        if status.is_some() {
            self.child = None;
        }
        Ok(status)
    }

    fn terminate(&mut self) {
        if let Some(mut child) = self.child.take() {
            if !matches!(child.try_wait(), Ok(Some(_))) {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn spawn_serve(invocation: &ProcessInvocation, log_path: &Path) -> Result<ManagedChild> {
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
    Ok(ManagedChild::new(child))
}

fn run_iron_bench(
    invocation: &ProcessInvocation,
    output_json: &Path,
    stderr_log: &Path,
    cancellation: &CalibrationCancellation,
) -> Result<()> {
    let partial_output = IncompleteArtifactGuard::new(partial_artifact_path(output_json));
    let stdout = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(partial_output.path())
        .with_context(|| {
            format!(
                "opening benchmark output {}",
                partial_output.path().display()
            )
        })?;
    let stderr = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(stderr_log)
        .with_context(|| format!("opening benchmark stderr {}", stderr_log.display()))?;
    let child = Command::new(&invocation.program)
        .args(&invocation.args)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .with_context(|| format!("spawning {}", invocation.program.display()))?;
    let mut child = ManagedChild::new(child);
    let status = loop {
        if cancellation.is_requested() {
            child.terminate();
            return Err(CalibrationCancelled.into());
        }
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("waiting for {}", invocation.program.display()))?
        {
            break status;
        }
        std::thread::sleep(CHILD_POLL_INTERVAL);
    };

    if !status.success() {
        let stderr = std::fs::read(stderr_log)
            .with_context(|| format!("reading {}", stderr_log.display()))?;
        if let Some(summary) = last_stderr_line(&stderr) {
            anyhow::bail!(
                "iron-bench failed with status {}; cause: {}; stderr log: {}",
                status,
                summary,
                stderr_log.display()
            );
        }
        anyhow::bail!(
            "iron-bench failed with status {}; stderr log: {}",
            status,
            stderr_log.display()
        );
    }
    std::fs::rename(partial_output.path(), output_json).with_context(|| {
        format!(
            "promoting benchmark output {} to {}",
            partial_output.path().display(),
            output_json.display()
        )
    })?;
    Ok(())
}

struct IncompleteArtifactGuard {
    path: PathBuf,
}

impl IncompleteArtifactGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for IncompleteArtifactGuard {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.path) {
            if error.kind() != std::io::ErrorKind::NotFound {
                eprintln!(
                    "warning: failed to remove incomplete benchmark output {}: {error}",
                    self.path.display()
                );
            }
        }
    }
}

fn partial_artifact_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .expect("benchmark output path should have a UTF-8 file name");
    path.with_file_name(format!("{file_name}.partial"))
}

fn last_stderr_line(stderr: &[u8]) -> Option<String> {
    let stderr = String::from_utf8_lossy(stderr);
    stderr
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| {
            line.chars()
                .take(BENCHMARK_STDERR_SUMMARY_MAX_CHARS)
                .collect()
        })
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
        candidate_artifact_path, health_url, last_stderr_line, parse_candidate_config,
        partial_artifact_path, persist_runtime_profile_from_artifact,
        resolve_run_config_with_context, run_iron_bench, validate_matrix, write_final_outputs,
        write_run_order_manifest, CalibrationCancellation, CalibrationCancelled,
        FinalArtifactPaths, ProcessInvocation, SchedulerAutotuneCalibrateArgs,
    };
    use crate::cli::scheduler_autotune::SchedulerAutotuneSelectionProfileArg;
    use crate::cli::scheduler_profile_context::SchedulerProfileRuntimeArgs;
    use crate::cli::scheduler_profile_store::SchedulerProfileStore;
    use crate::core::scheduler_autotune::{
        SchedulerAutotuneCacheState, SchedulerAutotuneCalibrationInput,
        SchedulerAutotuneMeasurement, SchedulerAutotuneObjective, SchedulerAutotuneProfileConfig,
        SchedulerAutotuneRuntimeContext, SchedulerAutotuneRuntimeHealth,
        SchedulerAutotuneSelectionProfile, SchedulerSpeculativeMode,
        SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
    };

    const TEST_PROMPT_TOKEN_RESERVE: usize = 21;

    #[test]
    fn stderr_summary_reports_last_non_empty_line() {
        let stderr = b"running benchmark\n\nError: HTTP 413 Payload Too Large\n";

        assert_eq!(
            last_stderr_line(stderr).as_deref(),
            Some("Error: HTTP 413 Payload Too Large")
        );
    }

    #[test]
    fn cancelled_benchmark_is_killed_without_promoting_partial_output() {
        let temp_dir = unique_temp_dir("scheduler-autotune-cancelled-benchmark");
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let output = temp_dir.join("candidate.json");
        let stderr = temp_dir.join("candidate.stderr.log");
        let invocation = ProcessInvocation {
            program: PathBuf::from("/bin/sleep"),
            args: vec!["30".to_string()],
        };

        let error = run_iron_bench(
            &invocation,
            &output,
            &stderr,
            &CalibrationCancellation::requested(),
        )
        .expect_err("cancelled benchmark should fail");

        assert!(error.downcast_ref::<CalibrationCancelled>().is_some());
        assert!(!output.exists());
        assert!(!partial_artifact_path(&output).exists());
        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

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
        let path = candidate_artifact_path(
            Path::new("/tmp/out"),
            3,
            2,
            SchedulerAutotuneCacheState::Cold,
        );
        assert_eq!(
            path.to_string_lossy(),
            "/tmp/out/candidate-003-c2-cold.json"
        );
    }

    #[test]
    fn build_serve_invocation_includes_scheduler_config() {
        let args = sample_resolved_config();
        let command =
            build_serve_invocation(Path::new("/tmp/ironmlx"), &args, profile_config(), 19000);

        assert_eq!(command.program.to_string_lossy(), "/tmp/ironmlx");
        assert!(command.args.contains(&"serve".to_string()));
        assert!(command.args.contains(&"--max-sequences".to_string()));
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
        let command = build_iron_bench_invocation(
            &args,
            profile_config(),
            1,
            SchedulerAutotuneCacheState::Cold,
            "http://127.0.0.1:18080",
        );

        assert!(!command.args.contains(&"--concurrent".to_string()));
        assert!(command.args.contains(&"--runs".to_string()));
        assert!(command.args.contains(&"5".to_string()));
        assert!(command.args.contains(&"--warmup".to_string()));
        assert!(command.args.contains(&"1".to_string()));
    }

    #[test]
    fn build_iron_bench_invocation_uses_concurrent_mode_for_concurrency_above_one() {
        let args = sample_resolved_config();
        let command = build_iron_bench_invocation(
            &args,
            profile_config(),
            2,
            SchedulerAutotuneCacheState::Cold,
            "http://127.0.0.1:18080",
        );

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
        assert_eq!(json["jobs"][0]["cache_state"], "cold");
        assert_eq!(json["jobs"][0]["output_json"], "candidate-000-c1-cold.json");
        assert_eq!(
            json["jobs"][0]["stderr_log"],
            "candidate-000-c1-cold.stderr.log"
        );
        assert_eq!(
            json["jobs"][0]["serve_log"],
            "serve-candidate-000-c1-cold.log"
        );
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

        let resolved = resolve_run_config_with_context(
            &args,
            Path::new("/opt/ironmlx/bin/ironmlx"),
            sample_runtime_context(),
            TEST_PROMPT_TOKEN_RESERVE,
        )
        .expect("resolve");

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
        assert_eq!(resolved.prompt_len, vec![1024, 8192, 32619]);
        assert_eq!(resolved.prompt_token_reserve, TEST_PROMPT_TOKEN_RESERVE);
        assert_eq!(resolved.concurrency, vec![1, 2, 4, 8]);
        assert_eq!(
            resolved.selection_profile,
            SchedulerAutotuneSelectionProfile::AgentLongPrompt
        );
        assert_eq!(resolved.candidates.len(), 12);
        assert_eq!(
            resolved.candidates.first().copied(),
            Some(profile_config_with_b_max_chunk_and_cap(1, 1024, 128))
        );
        assert_eq!(
            resolved.candidates.last().copied(),
            Some(profile_config_with_b_max_chunk_and_cap(4, 2048, 256))
        );
    }

    #[test]
    fn resolve_run_config_keeps_explicit_overrides() {
        let args = sample_args();

        let resolved = resolve_run_config_with_context(
            &args,
            Path::new("/opt/ironmlx/bin/ironmlx"),
            sample_runtime_context(),
            TEST_PROMPT_TOKEN_RESERVE,
        )
        .expect("resolve");

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
    fn resolve_run_config_limits_qwen_mtp_default_batch_candidates() {
        let mut args = sample_args();
        args.candidates.clear();
        let mut context = sample_runtime_context();
        context.speculative.mode = SchedulerSpeculativeMode::QwenMtp;
        context.speculative.source_fingerprint = Some("mtp-model".to_string());
        context.speculative.draft_tokens = Some(3);

        let resolved = resolve_run_config_with_context(
            &args,
            Path::new("/opt/ironmlx/bin/ironmlx"),
            context,
            TEST_PROMPT_TOKEN_RESERVE,
        )
        .expect("resolve Qwen MTP run");

        assert_eq!(resolved.candidates.len(), 8);
        assert!(resolved
            .candidates
            .iter()
            .all(|candidate| candidate.b_max <= 2));
    }

    #[test]
    fn prefix_cache_context_schedules_cold_and_warm_jobs() {
        let args = sample_args();
        let mut context = sample_runtime_context();
        context.prefix_cache.enabled = true;
        context.prefix_cache.block_size = Some(256);
        context.prefix_cache.max_pages = Some(512);

        let resolved = resolve_run_config_with_context(
            &args,
            Path::new("/opt/ironmlx/bin/ironmlx"),
            context,
            TEST_PROMPT_TOKEN_RESERVE,
        )
        .expect("resolve prefix-cache run");
        let plan = build_candidate_benchmark_plan(&resolved);

        assert_eq!(
            resolved.cache_states,
            vec![
                SchedulerAutotuneCacheState::Cold,
                SchedulerAutotuneCacheState::Warm,
            ]
        );
        assert_eq!(plan.len(), 4);
        assert!(plan
            .iter()
            .any(|job| job.cache_state == SchedulerAutotuneCacheState::Cold));
        assert!(plan
            .iter()
            .any(|job| job.cache_state == SchedulerAutotuneCacheState::Warm));
    }

    #[test]
    fn validate_matrix_reserves_capacity_for_chat_template_tokens() {
        let mut args = sample_args();
        args.prompt_len = vec![32640];
        let resolved = resolve_run_config_with_context(
            &args,
            Path::new("/opt/ironmlx/bin/ironmlx"),
            sample_runtime_context(),
            TEST_PROMPT_TOKEN_RESERVE,
        )
        .expect("resolve boundary prompt run");

        let error = validate_matrix(&resolved).expect_err("boundary prompt must reserve template");

        assert!(format!("{error:#}").contains("chat-template reserve (21)"));
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
            runtime_context: sample_runtime_context(),
            config: profile_config(),
            rules: Vec::new(),
            metadata:
                crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata::synthetic(
                    1811606400000,
                ),
        };
        let output = serde_json::to_string_pretty(&runtime_profile).expect("serialize profile");
        std::fs::write(&profile_path, format!("{output}\n")).expect("write profile artifact");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));

        let stored_path = persist_runtime_profile_from_artifact(&store, &model_dir, &profile_path)
            .expect("persist runtime profile");

        assert_eq!(
            stored_path,
            store.profile_path(
                "GLM-4.7-Flash-4bit",
                "m5-max-128gb",
                runtime_profile.metadata.selection_profile,
                &model_dir,
                &runtime_profile.runtime_context.fingerprint(),
            )
        );
        assert!(stored_path.exists());
        assert_eq!(
            store
                .find_profile(
                    &model_dir,
                    "m5-max-128gb",
                    &runtime_profile.runtime_context.fingerprint(),
                )
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
            runtime: SchedulerProfileRuntimeArgs::default(),
        }
    }

    fn sample_resolved_config() -> super::ResolvedRunConfig {
        resolve_run_config_with_context(
            &sample_args(),
            Path::new("/tmp/ironmlx"),
            sample_runtime_context(),
            TEST_PROMPT_TOKEN_RESERVE,
        )
        .expect("resolve sample")
    }

    fn profile_config() -> SchedulerAutotuneProfileConfig {
        profile_config_with_chunk_and_cap(1024, 256)
    }

    fn profile_config_with_chunk_and_cap(
        prefill_chunk_size: usize,
        decode_cadence_mid_chunk_cap: usize,
    ) -> SchedulerAutotuneProfileConfig {
        profile_config_with_b_max_chunk_and_cap(2, prefill_chunk_size, decode_cadence_mid_chunk_cap)
    }

    fn profile_config_with_b_max_chunk_and_cap(
        b_max: usize,
        prefill_chunk_size: usize,
        decode_cadence_mid_chunk_cap: usize,
    ) -> SchedulerAutotuneProfileConfig {
        SchedulerAutotuneProfileConfig {
            b_max,
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
            runtime_context: sample_runtime_context(),
            objective: SchedulerAutotuneObjective::agent_default(),
            measurements: vec![SchedulerAutotuneMeasurement {
                config,
                prompt_len: 2048,
                max_new_tokens: 128,
                concurrency: 1,
                cache_state: SchedulerAutotuneCacheState::Cold,
                ttft_ms_p95: 120.0,
                itl_ms_p95: 12.0,
                e2e_s_p95: 2.5,
                tokens_per_sec: 90.0,
                early_itl_ms_p95: 12.0,
                memory_budget_ok: true,
                cached_tokens_warning: false,
                runtime_health: sample_runtime_health(),
            }],
        }
    }

    fn sample_runtime_context() -> SchedulerAutotuneRuntimeContext {
        SchedulerAutotuneRuntimeContext::local_default(32768)
    }

    fn sample_runtime_health() -> SchedulerAutotuneRuntimeHealth {
        SchedulerAutotuneRuntimeHealth {
            healthy: true,
            status: "healthy".to_string(),
            request_completion_ok: true,
            admission_queue_full_count_delta: 0,
            memory_budget_exceeded_count_delta: 0,
            active_kv_degraded: false,
            active_kv_swap_error_count_delta: 0,
            logical_kv_cap_tokens: 32768,
            resident_kv_cap_tokens: 32768,
            mtp: None,
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
