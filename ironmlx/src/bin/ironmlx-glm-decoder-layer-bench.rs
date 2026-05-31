//! GLM-4.7 decoder layer micro-benchmark.
//!
//! This measures the real checkpoint's full single-layer decode path:
//! input norm -> MLA attention -> residual -> post norm -> MoE -> residual.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::Loader;
use ironmlx::models::glm4_moe_lite::config::Glm4MoeLiteConfig;
use ironmlx::models::glm4_moe_lite::decoder_layer::Glm4DecoderLayer;
use ironmlx::models::glm4_moe_lite::mla_cache::MlaLatentCache;
use mlx::{random, Array, Device, Dtype, StreamOrDevice};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-glm-decoder-layer-bench",
    about = "Direct GLM-4.7 full decoder-layer decode benchmark",
    version
)]
struct Args {
    /// Local GLM-4.7-Flash-4bit model directory.
    #[arg(long)]
    model: PathBuf,

    /// Decoder layer index to load. Layer 1 is the first MoE layer.
    #[arg(long, default_value_t = 1)]
    layer: i32,

    /// Existing cache lengths to prefill before decode. Pass multiple times.
    #[arg(long = "ctx-len")]
    ctx_lens: Vec<i32>,

    /// Timed decode runs per case. Each run advances the cache by one token.
    #[arg(long, default_value_t = 50)]
    runs: usize,

    /// Warmup decode runs per case. Each warmup advances the cache by one token.
    #[arg(long, default_value_t = 10)]
    warmup_runs: usize,

    /// PRNG seed for synthetic hidden states.
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
    layer: i32,
    ctx_lens: Vec<i32>,
    warmup_runs: usize,
    measured_runs: usize,
    stream_mode: &'static str,
    hidden_size: i32,
    kv_lora_rank: i32,
    qk_rope_head_dim: i32,
    dtype: &'static str,
    cache_prealloc: &'static str,
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
    validate_layer(&cfg, args.layer)?;
    let layer = Glm4DecoderLayer::from_loader(&loader, args.layer, &cfg)
        .with_context(|| format!("loading decoder layer {}", args.layer))?;

    let mut records = Vec::new();
    for &ctx_len in &ctx_lens {
        let decode_x = build_hidden(args.seed + 1_000_000 + ctx_len as u64, 1, &cfg)?;
        mlx::transforms::eval(&[&decode_x])?;

        let mut cache = prepare_cache(
            &layer,
            &cfg,
            ctx_len,
            args.seed + ctx_len as u64,
            args.warmup_runs,
            args.runs,
            bench_target.target,
            args.layer,
        )?;
        records.push(bench_case(
            ctx_len,
            "full-decoder-layer",
            args.warmup_runs,
            args.runs,
            || {
                run_decode_step(
                    &layer,
                    &decode_x,
                    &mut cache,
                    bench_target.target,
                    args.layer,
                )
            },
        )?);

        let mut cache = prepare_cache(
            &layer,
            &cfg,
            ctx_len,
            args.seed + 2_000_000 + ctx_len as u64,
            args.warmup_runs,
            args.runs,
            bench_target.target,
            args.layer,
        )?;
        records.push(bench_case(
            ctx_len,
            "full-decoder-layer-repeat",
            args.warmup_runs,
            args.runs,
            || {
                run_decode_step(
                    &layer,
                    &decode_x,
                    &mut cache,
                    bench_target.target,
                    args.layer,
                )
            },
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-glm-decoder-layer",
            model_dir: args.model.display().to_string(),
            layer: args.layer,
            ctx_lens,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            stream_mode: bench_target.label,
            hidden_size: cfg.hidden_size,
            kv_lora_rank: cfg.kv_lora_rank,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
            dtype: "bfloat16",
            cache_prealloc: "MlaLatentCache::with_step(cap)",
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
    if let Some(&ctx_len) = ctx_lens.iter().find(|&&ctx_len| ctx_len <= 0) {
        return Err(anyhow!("--ctx-len values must be positive, got {ctx_len}"));
    }
    Ok(())
}

fn validate_layer(cfg: &Glm4MoeLiteConfig, layer: i32) -> Result<()> {
    if layer < 0 || layer >= cfg.num_hidden_layers {
        return Err(anyhow!(
            "--layer must be in [0, {}), got {}",
            cfg.num_hidden_layers,
            layer
        ));
    }
    Ok(())
}

fn build_hidden(seed: u64, seq_len: i32, cfg: &Glm4MoeLiteConfig) -> Result<Array> {
    let key = random::key(seed).context("random key")?;
    random::normal()
        .shape((1_i32, seq_len, cfg.hidden_size))
        .dtype(Dtype::Bfloat16)
        .key(&key)
        .sample()
        .map_err(anyhow::Error::from)
}

#[allow(clippy::too_many_arguments)]
fn prepare_cache(
    layer: &Glm4DecoderLayer,
    cfg: &Glm4MoeLiteConfig,
    ctx_len: i32,
    seed: u64,
    warmup_runs: usize,
    runs: usize,
    target: StreamOrDevice,
    layer_idx: i32,
) -> Result<MlaLatentCache> {
    let extra = i32::try_from(warmup_runs + runs + 8).context("run count exceeds i32")?;
    let cap = ctx_len + extra;
    let mut cache = MlaLatentCache::new(
        1,
        cfg.kv_lora_rank,
        cfg.qk_rope_head_dim,
        Dtype::Bfloat16,
        cap,
    )
    .with_step(cap);
    let x = build_hidden(seed, ctx_len, cfg)?;
    let offset = offset_array(0)?;
    let out = layer.forward_on(&x, &offset, &mut cache, &[ctx_len], None, target, layer_idx)?;
    mlx::transforms::eval(&[&out])?;
    mlx::transforms::synchronize()?;
    Ok(cache)
}

fn offset_array(offset: i32) -> Result<Array> {
    (&[offset][..], &[1_i32][..])
        .try_into()
        .map_err(|e| anyhow!("build offset array: {e}"))
}

fn run_decode_step(
    layer: &Glm4DecoderLayer,
    x: &Array,
    cache: &mut MlaLatentCache,
    target: StreamOrDevice,
    layer_idx: i32,
) -> Result<Vec<Array>> {
    let offset_value = cache.offsets()[0];
    let offset = offset_array(offset_value)?;
    let out = layer.forward_on(x, &offset, cache, &[1], None, target, layer_idx)?;
    Ok(vec![out])
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
    println!("# ironmlx-glm-decoder-layer-bench");
    println!(
        "layer={} H={} stream={}",
        output.meta.layer, output.meta.hidden_size, output.meta.stream_mode
    );
    for record in &output.records {
        println!(
            "ctx={:<5} case={:<28} p50={:>8.4} ms p95={:>8.4} ms",
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
        assert!(validate_ctx_lens(&[0]).is_err());
    }

    #[test]
    fn validate_ctx_lens_accepts_positive_values() {
        validate_ctx_lens(&[128, 512]).unwrap();
    }

    #[test]
    fn percentile_interpolates() {
        assert_eq!(percentile(&[1.0, 3.0], 50.0), Some(2.0));
    }
}
