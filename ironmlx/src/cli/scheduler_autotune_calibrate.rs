use std::path::PathBuf;

use clap::Args;

use crate::core::scheduler_autotune::SchedulerAutotuneProfileConfig;
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

    /// Hardware label embedded in calibration JSON.
    #[arg(long)]
    pub hardware_label: String,

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

pub fn run(_args: SchedulerAutotuneCalibrateArgs) -> Result<()> {
    anyhow::bail!("scheduler-autotune calibrate runner is added in the next step")
}

#[cfg(test)]
mod tests {
    use super::parse_candidate_config;

    #[test]
    fn parse_candidate_config_accepts_all_required_fields() {
        let config = parse_candidate_config(
            "b_max=2,prefill_chunk_size=1024,admission_deadline_ms=5,admission_queue_max=32,max_cache_cap=32768",
        )
        .expect("candidate should parse");

        assert_eq!(config.b_max, 2);
        assert_eq!(config.prefill_chunk_size, 1024);
        assert_eq!(config.admission_deadline_ms, 5);
        assert_eq!(config.admission_queue_max, 32);
        assert_eq!(config.max_cache_cap, 32768);
    }

    #[test]
    fn parse_candidate_config_rejects_missing_field() {
        let err = parse_candidate_config(
            "b_max=2,prefill_chunk_size=1024,admission_deadline_ms=5,admission_queue_max=32",
        )
        .expect_err("missing max_cache_cap should fail");

        assert!(err.contains("max_cache_cap"), "unexpected error: {err}");
    }
}
