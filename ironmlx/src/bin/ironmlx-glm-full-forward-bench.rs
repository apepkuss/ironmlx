//! GLM-4.7 full-model decode forward micro-benchmark.
//!
//! This benchmark moves above isolated MLA/MoE/decoder-layer probes and measures
//! the full 47-layer single-token decode forward path with a real checkpoint.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::{build_position_ids, Loader, Sampler};
use ironmlx::models::glm4_moe_lite::config::Glm4MoeLiteConfig;
use ironmlx::models::Glm4MoeLiteModel;
use mlx::{Array, Device, Dtype, StreamOrDevice};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-glm-full-forward-bench",
    about = "Direct GLM-4.7 full-model decode forward benchmark",
    version
)]
struct Args {
    /// Local GLM-4.7-Flash-4bit model directory.
    #[arg(long)]
    model: PathBuf,

    /// Existing cache lengths to prefill before decode. Pass multiple times.
    #[arg(long = "ctx-len")]
    ctx_lens: Vec<i32>,

    /// Timed decode runs per case. Each run advances the cache by one token.
    #[arg(long, default_value_t = 50)]
    runs: usize,

    /// Warmup decode runs per case. Each warmup advances the cache by one token.
    #[arg(long, default_value_t = 10)]
    warmup_runs: usize,

    /// PRNG seed for synthetic token ids.
    #[arg(long, default_value_t = 20260531)]
    seed: u64,

    /// JSON output path.
    #[arg(long)]
    out: PathBuf,

    /// Stream target mode for diagnostics.
    #[arg(long, value_enum, default_value_t = StreamMode::Default)]
    stream_mode: StreamMode,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StreamMode {
    /// Preserve existing behavior: pass no explicit target.
    Default,
    /// Pass the current GPU default stream explicitly.
    ExplicitDefault,
    /// Create a fresh GPU stream and promote it to this thread's default.
    NewDefault,
    /// Create a fresh GPU stream and pass it explicitly.
    NewExplicit,
}

#[derive(Clone, Copy)]
struct BenchTarget {
    label: &'static str,
    target: StreamOrDevice,
}

#[derive(Serialize)]
struct BenchOutput {
    meta: Meta,
    records: Vec<Record>,
}

#[derive(Serialize)]
struct Meta {
    backend: &'static str,
    model_dir: String,
    ctx_lens: Vec<i32>,
    warmup_runs: usize,
    measured_runs: usize,
    stream_mode: &'static str,
    hidden_size: i32,
    vocab_size: i32,
    num_hidden_layers: i32,
    dtype: &'static str,
    cache_prealloc: &'static str,
    token_source: &'static str,
}

#[derive(Serialize)]
struct Record {
    ctx_len: i32,
    case: &'static str,
    output_shapes: Vec<Vec<i32>>,
    summary: Summary,
    warmups_ms: Vec<f64>,
    values_ms: Vec<f64>,
}

#[derive(Serialize)]
struct Summary {
    runs: usize,
    p50_ms: Option<f64>,
    p95_ms: Option<f64>,
    mean_ms: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let bench_target = args.stream_mode.configure()?;
    let ctx_lens = if args.ctx_lens.is_empty() {
        vec![128, 512, 725, 2048]
    } else {
        args.ctx_lens.clone()
    };

    let loader = Loader::open(&args.model).context("Loader::open")?;
    let cfg = Glm4MoeLiteConfig::from_loader(&loader).context("loading GLM config")?;
    let model = Glm4MoeLiteModel::from_loader_with_config(&loader, cfg.clone())
        .context("loading Glm4MoeLiteModel")?;
    let sampler = Sampler::greedy();
    let mut records = Vec::new();

    for &ctx_len in &ctx_lens {
        records.push(run_full_hidden_case(
            &model,
            &cfg,
            ctx_len,
            &args,
            bench_target.target,
        )?);
        records.push(run_full_logits_case(
            &model,
            &cfg,
            ctx_len,
            &args,
            bench_target.target,
        )?);
        records.push(run_full_logits_sample_case(
            &model,
            &cfg,
            &sampler,
            ctx_len,
            &args,
            bench_target.target,
        )?);
        records.push(run_full_logits_repeat_case(
            &model,
            &cfg,
            ctx_len,
            &args,
            bench_target.target,
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-glm-full-forward",
            model_dir: args.model.display().to_string(),
            ctx_lens,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            stream_mode: bench_target.label,
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            dtype: "bfloat16",
            cache_prealloc: "model.make_cache(..., cap >= ctx_len + warmup + runs)",
            token_source: "deterministic synthetic token ids in [256, vocab_size)",
        },
        records,
    };
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&output)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;
    print_summary(&output);
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    validate_ctx_lens(&args.ctx_lens)?;
    if args.runs == 0 {
        return Err(anyhow!("--runs must be positive"));
    }
    Ok(())
}

fn validate_ctx_lens(ctx_lens: &[i32]) -> Result<()> {
    for &ctx_len in ctx_lens {
        if ctx_len <= 0 {
            return Err(anyhow!("--ctx-len values must be positive, got {ctx_len}"));
        }
    }
    Ok(())
}

fn run_full_hidden_case(
    model: &Glm4MoeLiteModel,
    cfg: &Glm4MoeLiteConfig,
    ctx_len: i32,
    args: &Args,
    target: StreamOrDevice,
) -> Result<Record> {
    let mut cache = prepare_cache(
        model,
        cfg,
        ctx_len,
        args,
        target,
        args.seed + ctx_len as u64,
    )?;
    let mut decode_tokens = DecodeTokens::new(
        args.seed + 1_000_000 + ctx_len as u64,
        args.warmup_runs + args.runs,
        cfg.vocab_size,
    )?;
    bench_case(ctx_len, "full-hidden", args.warmup_runs, args.runs, || {
        run_decode_hidden(model, decode_tokens.next()?, &mut cache, target)
    })
}

fn run_full_logits_case(
    model: &Glm4MoeLiteModel,
    cfg: &Glm4MoeLiteConfig,
    ctx_len: i32,
    args: &Args,
    target: StreamOrDevice,
) -> Result<Record> {
    let mut cache = prepare_cache(
        model,
        cfg,
        ctx_len,
        args,
        target,
        args.seed + 2_000_000 + ctx_len as u64,
    )?;
    let mut decode_tokens = DecodeTokens::new(
        args.seed + 3_000_000 + ctx_len as u64,
        args.warmup_runs + args.runs,
        cfg.vocab_size,
    )?;
    bench_case(ctx_len, "full-logits", args.warmup_runs, args.runs, || {
        run_decode_logits(model, decode_tokens.next()?, &mut cache, target)
    })
}

fn run_full_logits_sample_case(
    model: &Glm4MoeLiteModel,
    cfg: &Glm4MoeLiteConfig,
    sampler: &Sampler,
    ctx_len: i32,
    args: &Args,
    target: StreamOrDevice,
) -> Result<Record> {
    let mut cache = prepare_cache(
        model,
        cfg,
        ctx_len,
        args,
        target,
        args.seed + 4_000_000 + ctx_len as u64,
    )?;
    let mut decode_tokens = DecodeTokens::new(
        args.seed + 5_000_000 + ctx_len as u64,
        args.warmup_runs + args.runs,
        cfg.vocab_size,
    )?;
    bench_case(
        ctx_len,
        "full-logits-sample",
        args.warmup_runs,
        args.runs,
        || run_decode_logits_sample(model, sampler, decode_tokens.next()?, &mut cache, target),
    )
}

fn run_full_logits_repeat_case(
    model: &Glm4MoeLiteModel,
    cfg: &Glm4MoeLiteConfig,
    ctx_len: i32,
    args: &Args,
    target: StreamOrDevice,
) -> Result<Record> {
    let mut cache = prepare_cache(
        model,
        cfg,
        ctx_len,
        args,
        target,
        args.seed + 6_000_000 + ctx_len as u64,
    )?;
    let mut decode_tokens = DecodeTokens::new(
        args.seed + 7_000_000 + ctx_len as u64,
        args.warmup_runs + args.runs,
        cfg.vocab_size,
    )?;
    bench_case(
        ctx_len,
        "full-logits-repeat",
        args.warmup_runs,
        args.runs,
        || run_decode_logits(model, decode_tokens.next()?, &mut cache, target),
    )
}

fn prepare_cache(
    model: &Glm4MoeLiteModel,
    cfg: &Glm4MoeLiteConfig,
    ctx_len: i32,
    args: &Args,
    target: StreamOrDevice,
    seed: u64,
) -> Result<Vec<ironmlx::nn::LayerCache>> {
    let extra_steps = i32::try_from(args.warmup_runs.saturating_add(args.runs))
        .context("warmup+runs exceeds i32")?;
    let cap = ctx_len
        .checked_add(extra_steps)
        .and_then(|n| n.checked_add(8))
        .ok_or_else(|| anyhow!("cache cap overflow for ctx_len={ctx_len}"))?;
    let mut cache = model.make_cache(1, cap, Dtype::Bfloat16)?;
    let ids = synthetic_token_ids(seed, ctx_len, cfg.vocab_size)?;
    let input: Array = (&ids[..], &[1_i32, ctx_len][..]).try_into()?;
    let position_ids = build_position_ids(0, ctx_len)?;
    let hidden =
        model.forward_text_hidden(&input, &position_ids, None, None, Some(&mut cache), target)?;
    mlx::transforms::eval(&[&hidden])?;
    mlx::transforms::synchronize()?;
    Ok(cache)
}

fn run_decode_hidden(
    model: &Glm4MoeLiteModel,
    token: u32,
    cache: &mut [ironmlx::nn::LayerCache],
    target: StreamOrDevice,
) -> Result<Vec<Array>> {
    let input = decode_token_array(token)?;
    let position_ids = build_position_ids(0, 1)?;
    let hidden =
        model.forward_text_hidden(&input, &position_ids, None, None, Some(cache), target)?;
    Ok(vec![hidden])
}

fn run_decode_logits(
    model: &Glm4MoeLiteModel,
    token: u32,
    cache: &mut [ironmlx::nn::LayerCache],
    target: StreamOrDevice,
) -> Result<Vec<Array>> {
    let logits = decode_logits(model, token, cache, target)?;
    Ok(vec![logits])
}

fn run_decode_logits_sample(
    model: &Glm4MoeLiteModel,
    sampler: &Sampler,
    token: u32,
    cache: &mut [ironmlx::nn::LayerCache],
    target: StreamOrDevice,
) -> Result<Vec<Array>> {
    let logits = decode_logits(model, token, cache, target)?;
    let vocab = logits.shape().as_slice()[2];
    let flat = logits.reshape((vocab,))?;
    let next = sampler.sample_async_greedy(&flat)?;
    Ok(vec![next])
}

fn decode_logits(
    model: &Glm4MoeLiteModel,
    token: u32,
    cache: &mut [ironmlx::nn::LayerCache],
    target: StreamOrDevice,
) -> Result<Array> {
    let input = decode_token_array(token)?;
    let position_ids = build_position_ids(0, 1)?;
    model.forward_on(&input, &position_ids, None, None, Some(cache), target)
}

fn decode_token_array(token: u32) -> Result<Array> {
    let ids = [token];
    Ok((&ids[..], &[1_i32, 1_i32][..]).try_into()?)
}

struct DecodeTokens {
    ids: Vec<u32>,
    next_idx: usize,
}

impl DecodeTokens {
    fn new(seed: u64, len: usize, vocab_size: i32) -> Result<Self> {
        let len = i32::try_from(len).context("decode token count exceeds i32")?;
        Ok(Self {
            ids: synthetic_token_ids(seed, len, vocab_size)?,
            next_idx: 0,
        })
    }

    fn next(&mut self) -> Result<u32> {
        let token = self
            .ids
            .get(self.next_idx)
            .copied()
            .ok_or_else(|| anyhow!("DecodeTokens exhausted at index {}", self.next_idx))?;
        self.next_idx += 1;
        Ok(token)
    }
}

fn synthetic_token_ids(seed: u64, len: i32, vocab_size: i32) -> Result<Vec<u32>> {
    if len < 0 {
        return Err(anyhow!(
            "synthetic token len must be non-negative, got {len}"
        ));
    }
    if vocab_size <= 256 {
        return Err(anyhow!(
            "vocab_size must be greater than 256 for normal synthetic token range, got {vocab_size}"
        ));
    }
    let span = (vocab_size - 256) as u64;
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut ids = Vec::with_capacity(len as usize);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ids.push(256 + (state % span) as u32);
    }
    Ok(ids)
}

fn bench_case<F>(
    ctx_len: i32,
    case: &'static str,
    warmup_runs: usize,
    runs: usize,
    mut f: F,
) -> Result<Record>
where
    F: FnMut() -> Result<Vec<Array>>,
{
    let mut output_shapes = Vec::new();
    let mut warmups_ms = Vec::with_capacity(warmup_runs);
    for _ in 0..warmup_runs {
        let (elapsed_ms, shapes) = time_once(&mut f)?;
        output_shapes = shapes;
        warmups_ms.push(elapsed_ms);
    }

    let mut values_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let (elapsed_ms, shapes) = time_once(&mut f)?;
        output_shapes = shapes;
        values_ms.push(elapsed_ms);
    }

    Ok(Record {
        ctx_len,
        case,
        output_shapes,
        summary: summarize(&values_ms),
        warmups_ms,
        values_ms,
    })
}

fn time_once<F>(f: &mut F) -> Result<(f64, Vec<Vec<i32>>)>
where
    F: FnMut() -> Result<Vec<Array>>,
{
    let started = Instant::now();
    let outputs = f()?;
    let refs: Vec<&Array> = outputs.iter().collect();
    mlx::transforms::eval(&refs)?;
    mlx::transforms::synchronize()?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let shapes = outputs
        .iter()
        .map(|a| a.shape().as_slice().to_vec())
        .collect();
    Ok((elapsed_ms, shapes))
}

fn summarize(values: &[f64]) -> Summary {
    Summary {
        runs: values.len(),
        p50_ms: percentile(values, 50.0),
        p95_ms: percentile(values, 95.0),
        mean_ms: if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        },
    }
}

fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.len() == 1 {
        return sorted.first().copied();
    }
    let rank = (p / 100.0) * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let weight = rank - lo as f64;
    Some(sorted[lo] * (1.0 - weight) + sorted[hi] * weight)
}

impl StreamMode {
    fn configure(self) -> Result<BenchTarget> {
        let gpu = Device::gpu(0);
        match self {
            StreamMode::Default => Ok(BenchTarget {
                label: "default",
                target: StreamOrDevice::default(),
            }),
            StreamMode::ExplicitDefault => {
                let stream = mlx::default_stream(gpu);
                Ok(BenchTarget {
                    label: "explicit-default",
                    target: stream.into(),
                })
            }
            StreamMode::NewDefault => {
                let stream = mlx::new_stream(gpu).context("creating diagnostic GPU stream")?;
                mlx::set_default_stream(stream);
                Ok(BenchTarget {
                    label: "new-default",
                    target: StreamOrDevice::default(),
                })
            }
            StreamMode::NewExplicit => {
                let stream = mlx::new_stream(gpu).context("creating diagnostic GPU stream")?;
                Ok(BenchTarget {
                    label: "new-explicit",
                    target: stream.into(),
                })
            }
        }
    }
}

fn print_summary(output: &BenchOutput) {
    println!("# ironmlx-glm-full-forward-bench");
    println!(
        "layers={} H={} V={} stream={}",
        output.meta.num_hidden_layers,
        output.meta.hidden_size,
        output.meta.vocab_size,
        output.meta.stream_mode
    );
    for record in &output.records {
        println!(
            "ctx={:<5} case={:<22} p50={:>8.4} ms p95={:>8.4} ms",
            record.ctx_len,
            record.case,
            record.summary.p50_ms.unwrap_or(f64::NAN),
            record.summary.p95_ms.unwrap_or(f64::NAN)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_ctx_lens_rejects_non_positive_values() {
        assert!(validate_ctx_lens(&[128, 0]).is_err());
        assert!(validate_ctx_lens(&[-1]).is_err());
    }

    #[test]
    fn validate_ctx_lens_accepts_default_empty_and_positive_values() {
        validate_ctx_lens(&[]).unwrap();
        validate_ctx_lens(&[128, 512]).unwrap();
    }

    #[test]
    fn synthetic_token_ids_stay_in_normal_vocab_range() {
        let ids = synthetic_token_ids(7, 16, 1024).unwrap();
        assert_eq!(ids.len(), 16);
        assert!(ids.iter().all(|&id| (256..1024).contains(&id)));
        assert_eq!(ids, synthetic_token_ids(7, 16, 1024).unwrap());
        assert_ne!(ids, synthetic_token_ids(8, 16, 1024).unwrap());
    }

    #[test]
    fn synthetic_token_ids_rejects_invalid_inputs() {
        assert!(synthetic_token_ids(7, -1, 1024).is_err());
        assert!(synthetic_token_ids(7, 1, 256).is_err());
    }

    #[test]
    fn decode_tokens_errors_when_exhausted() {
        let mut tokens = DecodeTokens::new(7, 1, 1024).unwrap();
        tokens.next().unwrap();
        assert!(tokens.next().is_err());
    }

    #[test]
    fn percentile_interpolates() {
        assert_eq!(percentile(&[1.0, 3.0], 50.0), Some(2.0));
    }
}
