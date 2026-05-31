//! GLM-4.7 MoE block micro-benchmark.
//!
//! This binary complements `ironmlx-glm-moe-bench`, which isolates routed
//! experts only. Here we measure the full GLM MoE block and the router/shared
//! paths that participate in single-token decode.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::Loader;
use ironmlx::models::glm4_moe_lite::config::Glm4MoeLiteConfig;
use ironmlx::models::glm4_moe_lite::moe::{noaux_tc_route, Glm4MoeBlock};
use ironmlx::models::qwen3_5_moe::RoutedExperts;
use ironmlx::nn::{Linear, Mlp};
use mlx::ops::shape::reshape_on;
use mlx::{random, Array, Device, Dtype, StreamOrDevice};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-glm-moe-block-bench",
    about = "Direct GLM-4.7 MoE block/router/shared benchmark",
    version
)]
struct Args {
    /// Local GLM-4.7-Flash-4bit model directory.
    #[arg(long)]
    model: PathBuf,

    /// MoE layer index to load. Layer 0 is dense, so default starts at 1.
    #[arg(long, default_value_t = 1)]
    layer: i32,

    /// Batch sizes to measure with sequence length 1. Pass multiple times.
    #[arg(long)]
    bs: Vec<i32>,

    /// Timed runs per case.
    #[arg(long, default_value_t = 50)]
    runs: usize,

    /// Warmup runs per case.
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
    bs_values: Vec<i32>,
    warmup_runs: usize,
    measured_runs: usize,
    stream_mode: &'static str,
    hidden_size: i32,
    num_experts: i32,
    k: i32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
}

#[derive(Serialize)]
struct Record {
    bs: i32,
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
    let bs_values = if args.bs.is_empty() {
        vec![1]
    } else {
        args.bs.clone()
    };

    let loader = Loader::open(&args.model).context("Loader::open")?;
    let cfg = Glm4MoeLiteConfig::from_loader(&loader).context("loading GLM config")?;
    validate_layer(&cfg, args.layer)?;
    let prefix = format!("model.layers.{}.mlp", args.layer);
    let block = Glm4MoeBlock::from_loader(&loader, &prefix, &cfg)
        .with_context(|| format!("loading Glm4MoeBlock from {prefix}"))?;
    let gate =
        Linear::from_loader(&loader, &format!("{prefix}.gate")).context("loading router gate")?;
    let bias = loader
        .tensor(&format!("{prefix}.gate.e_score_correction_bias"))
        .context("loading router correction bias")?
        .clone();
    let experts = RoutedExperts::from_loader(&loader, &format!("{prefix}.switch_mlp"))
        .context("loading routed experts")?;
    let shared = Mlp::from_loader(&loader, &format!("{prefix}.shared_experts"))
        .context("loading shared expert")?;

    let mut records = Vec::new();
    for &bs in &bs_values {
        let x = build_input(args.seed + bs as u64, bs, cfg.hidden_size)?;
        mlx::transforms::eval(&[&x])?;
        let flat = flatten_bs1(&x, bs, cfg.hidden_size, bench_target.target)?;
        mlx::transforms::eval(&[&flat])?;

        records.push(bench_case(
            bs,
            "production-moe-block",
            args.warmup_runs,
            args.runs,
            || {
                block
                    .forward_on(&x, bench_target.target, args.layer)
                    .map(|out| vec![out])
            },
        )?);

        records.push(bench_case(
            bs,
            "reshape-input",
            args.warmup_runs,
            args.runs,
            || flatten_rank3_input(&x, bench_target.target).map(|(out, _)| vec![out]),
        )?);

        records.push(bench_case(
            bs,
            "router-gate",
            args.warmup_runs,
            args.runs,
            || {
                gate.forward_on(&flat, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);

        let logits = gate.forward_on(&flat, bench_target.target)?;
        mlx::transforms::eval(&[&logits])?;
        records.push(bench_case(
            bs,
            "noaux-route",
            args.warmup_runs,
            args.runs,
            || {
                let (inds, weights) = noaux_tc_route(
                    &logits,
                    &bias,
                    cfg.num_experts_per_tok,
                    cfg.norm_topk_prob,
                    cfg.routed_scaling_factor,
                    bench_target.target,
                )?;
                Ok(vec![inds, weights])
            },
        )?);

        records.push(bench_case(
            bs,
            "router-full",
            args.warmup_runs,
            args.runs,
            || {
                let logits = gate.forward_on(&flat, bench_target.target)?;
                let (inds, weights) = noaux_tc_route(
                    &logits,
                    &bias,
                    cfg.num_experts_per_tok,
                    cfg.norm_topk_prob,
                    cfg.routed_scaling_factor,
                    bench_target.target,
                )?;
                Ok(vec![inds, weights])
            },
        )?);

        let (inds, weights) = noaux_tc_route(
            &logits,
            &bias,
            cfg.num_experts_per_tok,
            cfg.norm_topk_prob,
            cfg.routed_scaling_factor,
            bench_target.target,
        )?;
        mlx::transforms::eval(&[&inds, &weights])?;

        records.push(bench_case(
            bs,
            "routed-experts",
            args.warmup_runs,
            args.runs,
            || {
                experts
                    .apply_experts(&flat, &inds, &weights, bench_target.target, args.layer)
                    .map(|out| vec![out])
            },
        )?);

        records.push(bench_case(
            bs,
            "shared-expert",
            args.warmup_runs,
            args.runs,
            || {
                shared
                    .forward_on(&flat, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);

        let routed =
            experts.apply_experts(&flat, &inds, &weights, bench_target.target, args.layer)?;
        let shared_out = shared.forward_on(&flat, bench_target.target)?;
        mlx::transforms::eval(&[&routed, &shared_out])?;
        records.push(bench_case(
            bs,
            "output-sum",
            args.warmup_runs,
            args.runs,
            || {
                let out_flat = routed.try_add_on(&shared_out, bench_target.target)?;
                reshape_on(&out_flat, [bs, 1_i32, cfg.hidden_size], bench_target.target)
                    .map(|out| vec![out])
                    .map_err(anyhow::Error::from)
            },
        )?);

        records.push(bench_case(
            bs,
            "local-full-block",
            args.warmup_runs,
            args.runs,
            || {
                local_full_block(
                    &gate,
                    &bias,
                    &experts,
                    &shared,
                    &flat,
                    [bs, 1_i32, cfg.hidden_size],
                    &cfg,
                    bench_target.target,
                    args.layer,
                )
                .map(|out| vec![out])
            },
        )?);

        records.push(bench_case(
            bs,
            "local-full-block-with-reshape",
            args.warmup_runs,
            args.runs,
            || {
                let (flat, out_shape) = flatten_rank3_input(&x, bench_target.target)?;
                local_full_block(
                    &gate,
                    &bias,
                    &experts,
                    &shared,
                    &flat,
                    out_shape,
                    &cfg,
                    bench_target.target,
                    args.layer,
                )
                .map(|out| vec![out])
            },
        )?);

        records.push(bench_case(
            bs,
            "production-moe-block-repeat",
            args.warmup_runs,
            args.runs,
            || {
                block
                    .forward_on(&x, bench_target.target, args.layer)
                    .map(|out| vec![out])
            },
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-glm-moe-block",
            model_dir: args.model.display().to_string(),
            layer: args.layer,
            bs_values,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            stream_mode: bench_target.label,
            hidden_size: cfg.hidden_size,
            num_experts: cfg.n_routed_experts,
            k: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
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
    if args.layer <= 0 {
        return Err(anyhow!(
            "--layer must be a MoE layer >= 1, got {}",
            args.layer
        ));
    }
    validate_bs_values(&args.bs)?;
    if args.runs == 0 {
        return Err(anyhow!("--runs must be positive"));
    }
    Ok(())
}

fn validate_bs_values(bs_values: &[i32]) -> Result<()> {
    if let Some(&bs) = bs_values.iter().find(|&&bs| bs <= 0) {
        return Err(anyhow!("--bs values must be positive, got {bs}"));
    }
    Ok(())
}

fn validate_layer(cfg: &Glm4MoeLiteConfig, layer: i32) -> Result<()> {
    if layer >= cfg.num_hidden_layers {
        return Err(anyhow!(
            "--layer must be < num_hidden_layers={}, got {}",
            cfg.num_hidden_layers,
            layer
        ));
    }
    if !cfg.is_moe_layer(layer) {
        return Err(anyhow!("--layer must select a MoE layer, got {layer}"));
    }
    Ok(())
}

fn build_input(seed: u64, bs: i32, hidden_size: i32) -> Result<Array> {
    let key = random::key(seed).context("random key")?;
    random::normal()
        .shape((bs, 1_i32, hidden_size))
        .dtype(Dtype::Bfloat16)
        .key(&key)
        .sample()
        .map_err(anyhow::Error::from)
}

fn rank3_to_flat_dims(dims: &[i32]) -> Result<(i32, i32)> {
    if dims.len() != 3 {
        return Err(anyhow!(
            "expected rank-3 [B,S,H] input, got rank {}",
            dims.len()
        ));
    }
    Ok((dims[0] * dims[1], dims[2]))
}

fn flatten_rank3_input(x: &Array, target: StreamOrDevice) -> Result<(Array, [i32; 3])> {
    let shape = x.shape();
    let dims = shape.as_slice();
    let (flat_bs, hidden_size) = rank3_to_flat_dims(dims)?;
    let flat = reshape_on(x, [flat_bs, hidden_size], target)
        .context("flatten_rank3_input: reshape [B,S,H] to [BS,H]")?;
    Ok((flat, [dims[0], dims[1], dims[2]]))
}

fn flatten_bs1(x: &Array, bs: i32, hidden_size: i32, target: StreamOrDevice) -> Result<Array> {
    reshape_on(x, [bs, hidden_size], target).map_err(anyhow::Error::from)
}

#[allow(clippy::too_many_arguments)]
fn local_full_block(
    gate: &Linear,
    bias: &Array,
    experts: &RoutedExperts,
    shared: &Mlp,
    flat: &Array,
    out_shape: [i32; 3],
    cfg: &Glm4MoeLiteConfig,
    target: StreamOrDevice,
    layer: i32,
) -> Result<Array> {
    let bs = out_shape[0] * out_shape[1];
    if flat.shape().as_slice() != [bs, out_shape[2]] {
        return Err(anyhow!(
            "local_full_block: flat shape must be [{bs},{}], got {:?}",
            out_shape[2],
            flat.shape().as_slice()
        ));
    }
    let logits = gate.forward_on(flat, target)?;
    let (inds, weights) = noaux_tc_route(
        &logits,
        bias,
        cfg.num_experts_per_tok,
        cfg.norm_topk_prob,
        cfg.routed_scaling_factor,
        target,
    )?;
    let routed = experts.apply_experts(flat, &inds, &weights, target, layer)?;
    let shared_out = shared.forward_on(flat, target)?;
    let out_flat = routed.try_add_on(&shared_out, target)?;
    if out_shape[2] != cfg.hidden_size {
        return Err(anyhow!(
            "local_full_block: output hidden size {} does not match config hidden size {}",
            out_shape[2],
            cfg.hidden_size
        ));
    }
    reshape_on(&out_flat, out_shape, target).map_err(anyhow::Error::from)
}

fn bench_case<F>(
    bs: i32,
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
        bs,
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
    println!("# ironmlx-glm-moe-block-bench");
    println!(
        "layer={} H={} E={} k={} stream={}",
        output.meta.layer,
        output.meta.hidden_size,
        output.meta.num_experts,
        output.meta.k,
        output.meta.stream_mode
    );
    for record in &output.records {
        println!(
            "bs={:<3} case={:<22} p50={:>8.4} ms p95={:>8.4} ms",
            record.bs,
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
    fn validate_bs_values_rejects_non_positive_values() {
        assert!(validate_bs_values(&[0]).is_err());
    }

    #[test]
    fn validate_bs_values_accepts_positive_values() {
        validate_bs_values(&[1, 16, 64]).unwrap();
    }

    #[test]
    fn percentile_interpolates() {
        assert_eq!(percentile(&[1.0, 3.0], 50.0), Some(2.0));
    }

    #[test]
    fn rank3_to_flat_dims_flattens_batch_sequence() {
        assert_eq!(rank3_to_flat_dims(&[2, 3, 2048]).unwrap(), (6, 2048));
    }

    #[test]
    fn rank3_to_flat_dims_rejects_non_rank3() {
        assert!(rank3_to_flat_dims(&[2, 2048]).is_err());
    }
}
