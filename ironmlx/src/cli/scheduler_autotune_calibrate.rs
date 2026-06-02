use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Args;

use crate::core::scheduler_autotune::{
    build_scheduler_autotune_runtime_profile, merge_scheduler_autotune_calibrations,
    select_scheduler_autotune_profile, SchedulerAutotuneCalibrationInput,
    SchedulerAutotuneMergeOptions, SchedulerAutotuneProfileConfig,
};
use crate::Result;

const DEFAULT_PORT: u16 = 18080;
const DEFAULT_STARTUP_TIMEOUT_SEC: u64 = 300;

#[derive(Args, Debug)]
pub struct SchedulerAutotuneCalibrateArgs {
    /// Local directory containing config.json + model.safetensors + tokenizer.json.
    #[arg(long)]
    pub model: PathBuf,

    /// Model name to pass to iron-bench request payloads and calibration JSON.
    #[arg(long)]
    pub model_name: String,

    /// Path to the iron-bench binary.
    #[arg(long)]
    pub iron_bench_bin: PathBuf,

    /// Directory for candidate JSON files, logs, and final outputs.
    #[arg(long)]
    pub output_dir: PathBuf,

    /// Scheduler candidate config, repeated once per candidate.
    #[arg(long = "candidate", required = true, value_parser = parse_candidate_config)]
    pub candidates: Vec<SchedulerAutotuneProfileConfig>,

    /// Prompt token lengths to test.
    #[arg(long, value_delimiter = ',', required = true)]
    pub prompt_len: Vec<usize>,

    /// Number of generated tokens per request.
    #[arg(long, default_value_t = 128)]
    pub max_tokens: usize,

    /// Concurrency levels to test. `1` uses sequential iron-bench mode.
    #[arg(long, value_delimiter = ',', required = true)]
    pub concurrency: Vec<usize>,

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

    /// Optional runtime scheduler profile output path.
    #[arg(long)]
    pub write_profile: Option<PathBuf>,
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

pub fn run(args: SchedulerAutotuneCalibrateArgs) -> Result<()> {
    validate_matrix(&args)?;
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating {}", args.output_dir.display()))?;

    let ironmlx_bin = std::env::current_exe().context("locating current ironmlx executable")?;
    let target_url = format!("http://127.0.0.1:{}", args.port);
    let health = health_url(args.port);
    let mut candidate_outputs = Vec::new();

    for (candidate_idx, config) in args.candidates.iter().copied().enumerate() {
        let serve_log = serve_log_path(&args.output_dir, candidate_idx);
        let serve_invocation = build_serve_invocation(&ironmlx_bin, &args, config, args.port);
        let _serve = spawn_serve(&serve_invocation, &serve_log)?;

        wait_for_health(&health, Duration::from_secs(args.startup_timeout_sec))
            .with_context(|| format!("serve log: {}", serve_log.display()))?;

        for &concurrency in &args.concurrency {
            let output_json = candidate_artifact_path(&args.output_dir, candidate_idx, concurrency);
            let stderr_log =
                candidate_stderr_log_path(&args.output_dir, candidate_idx, concurrency);
            let bench_invocation =
                build_iron_bench_invocation(&args, config, concurrency, &target_url);
            run_iron_bench(&bench_invocation, &output_json, &stderr_log)?;
            candidate_outputs.push(output_json);
        }
    }

    let mut inputs = Vec::with_capacity(candidate_outputs.len());
    for path in &candidate_outputs {
        inputs.push(read_calibration(path)?);
    }
    let artifacts = FinalArtifactPaths::new(&args.output_dir, args.write_profile.clone());
    write_final_outputs(inputs, &artifacts)?;

    println!("calibration: {}", artifacts.calibration.display());
    println!("selection_json: {}", artifacts.selection_json.display());
    println!("selection_text: {}", artifacts.selection_text.display());
    if let Some(path) = &artifacts.runtime_profile {
        println!("runtime_profile: {}", path.display());
    }

    Ok(())
}

fn validate_matrix(args: &SchedulerAutotuneCalibrateArgs) -> Result<()> {
    if args.prompt_len.contains(&0) {
        anyhow::bail!("--prompt-len values must be > 0");
    }
    if args.concurrency.contains(&0) {
        anyhow::bail!("--concurrency values must be > 0");
    }
    Ok(())
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

fn serve_log_path(output_dir: &Path, candidate_idx: usize) -> PathBuf {
    output_dir.join(format!("serve-candidate-{candidate_idx:03}.log"))
}

fn build_serve_invocation(
    ironmlx_bin: &Path,
    args: &SchedulerAutotuneCalibrateArgs,
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
    args: &SchedulerAutotuneCalibrateArgs,
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
            runtime_profile,
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

fn write_final_outputs(
    inputs: Vec<SchedulerAutotuneCalibrationInput>,
    artifacts: &FinalArtifactPaths,
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

    let selection = select_scheduler_autotune_profile(merged);
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
        build_iron_bench_invocation, build_serve_invocation, candidate_artifact_path, health_url,
        parse_candidate_config, write_final_outputs, FinalArtifactPaths,
        SchedulerAutotuneCalibrateArgs,
    };
    use crate::core::scheduler_autotune::{
        SchedulerAutotuneCalibrationInput, SchedulerAutotuneMeasurement,
        SchedulerAutotuneObjective, SchedulerAutotuneProfileConfig,
        SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
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
        let args = sample_args();
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
        let args = sample_args();
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
        let args = sample_args();
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
    fn write_final_outputs_writes_calibration_selection_and_profile() {
        let temp_dir = unique_temp_dir("scheduler-autotune-calibrate-output");
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let profile_path = temp_dir.join("scheduler-profile.json");
        let paths = FinalArtifactPaths::new(&temp_dir, Some(profile_path.clone()));

        write_final_outputs(vec![sample_calibration(profile_config())], &paths)
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

    fn sample_args() -> SchedulerAutotuneCalibrateArgs {
        SchedulerAutotuneCalibrateArgs {
            model: PathBuf::from("/tmp/model"),
            model_name: "GLM-4.7-flash-4bit".to_string(),
            iron_bench_bin: PathBuf::from("/tmp/iron-bench"),
            output_dir: PathBuf::from("/tmp/out"),
            candidates: vec![profile_config()],
            prompt_len: vec![1024, 2048],
            max_tokens: 128,
            concurrency: vec![1, 2],
            runs: 5,
            warmup: 1,
            duration: 30,
            warmup_duration: 5,
            port: 18080,
            startup_timeout_sec: 300,
            write_profile: Some(PathBuf::from("/tmp/profile.json")),
        }
    }

    fn profile_config() -> SchedulerAutotuneProfileConfig {
        SchedulerAutotuneProfileConfig {
            b_max: 2,
            prefill_chunk_size: 1024,
            admission_deadline_ms: 5,
            admission_queue_max: 32,
            max_cache_cap: 32768,
            decode_cadence_mid_chunk_cap: 256,
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
