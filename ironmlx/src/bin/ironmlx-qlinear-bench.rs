//! Focused quantized-linear benchmark for Qwen3.6 GatedDeltaNet projections.
//!
//! This isolates the qkvz and out_proj shapes that the GDN stage breakdown
//! identified as Rust-side hotspots.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::{Loader, QuantMeta};
use ironmlx::models::Qwen35MoeConfig;
use ironmlx::nn::{self_qmm, AttnKind, GatedDeltaNetConfig, Linear, RmsNormGated};
use mlx::{random, Array, Device, Dtype, StreamOrDevice};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-qlinear-bench",
    about = "Direct Qwen3.6 GDN quantized-linear projection benchmark",
    version
)]
struct Args {
    /// Local model directory.
    #[arg(long)]
    model: PathBuf,

    /// Linear-attention layer index to load.
    #[arg(long, default_value_t = 0)]
    layer: i32,

    /// Sequence lengths to measure. Pass multiple times for multiple shapes.
    #[arg(long)]
    seq: Vec<i32>,

    /// Timed runs per shape.
    #[arg(long, default_value_t = 25)]
    runs: usize,

    /// Warmup runs per shape.
    #[arg(long, default_value_t = 5)]
    warmup_runs: usize,

    /// PRNG seed for synthetic hidden states.
    #[arg(long, default_value_t = 20260528)]
    seed: u64,

    /// JSON output path.
    #[arg(long)]
    out: PathBuf,

    /// Include diagnostic direct self_qmm cases, bypassing Linear's production threshold.
    #[arg(long)]
    include_self_qmm: bool,

    /// Include C++-side direct quantized_matmul timing-loop diagnostics.
    #[arg(long)]
    include_cxx_qmm: bool,

    /// Stream target mode for qlinear diagnostics.
    #[arg(long, value_enum, default_value_t = StreamMode::Default)]
    stream_mode: StreamMode,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StreamMode {
    /// Preserve existing behavior: pass no explicit target, so MLX resolves the current default.
    Default,
    /// Pass the current GPU default stream explicitly to targeted ops.
    ExplicitDefault,
    /// Create a fresh GPU stream and promote it to this thread's default.
    NewDefault,
    /// Create a fresh GPU stream and pass it explicitly to targeted ops.
    NewExplicit,
}

#[derive(Clone, Copy)]
struct BenchTarget {
    label: &'static str,
    target: StreamOrDevice,
}

struct QuantProjection {
    linear: Linear,
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
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
    seqs: Vec<i32>,
    warmup_runs: usize,
    measured_runs: usize,
    stream_mode: &'static str,
    include_cxx_qmm: bool,
    hidden_size: i32,
    conv_dim: i32,
    value_dim: i32,
    qkvz_out_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
    group_size: i32,
    bits: i32,
}

#[derive(Serialize)]
struct Record {
    seq: i32,
    case: &'static str,
    output_shapes: Vec<Vec<i32>>,
    summary: Summary,
    warmups: Vec<f64>,
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
    let bench_target = args.stream_mode.configure()?;
    let seqs = if args.seq.is_empty() {
        vec![521, 1]
    } else {
        args.seq.clone()
    };

    let loader = Loader::open(&args.model).context("Loader::open")?;
    let cfg = Qwen35MoeConfig::from_loader(&loader).context("Qwen35MoeConfig::from_loader")?;
    if !matches!(cfg.layer_kind(args.layer), AttnKind::Linear) {
        return Err(anyhow!(
            "layer {} is not a linear-attention GatedDeltaNet layer",
            args.layer
        ));
    }

    let gdn_cfg = GatedDeltaNetConfig {
        hidden_size: cfg.hidden_size,
        num_v_heads: cfg.linear_num_value_heads,
        num_k_heads: cfg.linear_num_key_heads,
        head_k_dim: cfg.linear_key_head_dim,
        head_v_dim: cfg.linear_value_head_dim,
        conv_kernel_size: cfg.linear_conv_kernel_dim,
        rms_norm_eps: cfg.rms_norm_eps,
    };
    let prefix = format!("model.layers.{}.linear_attn", args.layer);
    let qkvz = load_fused_projection(&loader, &prefix, "in_proj_qkv", "in_proj_z")
        .with_context(|| format!("loading fused qkvz projection for {prefix}"))?;
    let out_proj = load_quant_projection(&loader, &format!("{prefix}.out_proj"))
        .with_context(|| format!("loading out_proj for {prefix}"))?;
    let norm = RmsNormGated::from_loader(&loader, &format!("{prefix}.norm"), cfg.rms_norm_eps)?;

    let mut records = Vec::new();
    for &seq in &seqs {
        if seq <= 0 {
            return Err(anyhow!("seq must be positive, got {seq}"));
        }
        let x_hidden = random_bf16(args.seed + seq as u64, (1, seq, gdn_cfg.hidden_size))?;
        let x_value = random_bf16(
            args.seed + 100_000 + seq as u64,
            (1, seq, gdn_cfg.value_dim()),
        )?;
        let y_heads = random_bf16(
            args.seed + 200_000 + seq as u64,
            (1, seq, gdn_cfg.num_v_heads, gdn_cfg.head_v_dim),
        )?;
        let z_heads = random_bf16(
            args.seed + 300_000 + seq as u64,
            (1, seq, gdn_cfg.num_v_heads, gdn_cfg.head_v_dim),
        )?;
        mlx::transforms::eval(&[&x_hidden, &x_value, &y_heads, &z_heads])?;

        records.push(bench_case(
            seq,
            "qkvz-direct-qmm",
            args.warmup_runs,
            args.runs,
            || {
                qkvz.forward_direct_on(&x_hidden, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);
        if args.include_cxx_qmm {
            records.push(bench_cxx_qmm(
                seq,
                "qkvz-cxx-loop-qmm",
                &qkvz,
                &x_hidden,
                args.warmup_runs,
                args.runs,
                bench_target.target,
            )?);
        }
        if args.include_self_qmm {
            records.push(bench_case(
                seq,
                "qkvz-self-qmm",
                args.warmup_runs,
                args.runs,
                || {
                    qkvz.forward_self_qmm_on(&x_hidden, bench_target.target)
                        .map(|out| vec![out])
                },
            )?);
        }
        records.push(bench_case(
            seq,
            "qkvz-linear",
            args.warmup_runs,
            args.runs,
            || {
                qkvz.forward_linear_on(&x_hidden, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);
        records.push(bench_case(
            seq,
            "qkvz-linear-slice",
            args.warmup_runs,
            args.runs,
            || qkvz_linear_slice(&qkvz, &x_hidden, gdn_cfg, seq, bench_target.target),
        )?);
        records.push(bench_case(
            seq,
            "out-direct-qmm",
            args.warmup_runs,
            args.runs,
            || {
                out_proj
                    .forward_direct_on(&x_value, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);
        if args.include_cxx_qmm {
            records.push(bench_cxx_qmm(
                seq,
                "out-cxx-loop-qmm",
                &out_proj,
                &x_value,
                args.warmup_runs,
                args.runs,
                bench_target.target,
            )?);
        }
        if args.include_self_qmm {
            records.push(bench_case(
                seq,
                "out-self-qmm",
                args.warmup_runs,
                args.runs,
                || {
                    out_proj
                        .forward_self_qmm_on(&x_value, bench_target.target)
                        .map(|out| vec![out])
                },
            )?);
        }
        records.push(bench_case(
            seq,
            "out-linear",
            args.warmup_runs,
            args.runs,
            || {
                out_proj
                    .forward_linear_on(&x_value, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);
        records.push(bench_case(
            seq,
            "norm-out-linear",
            args.warmup_runs,
            args.runs,
            || {
                norm_out_linear(
                    &norm,
                    &out_proj,
                    &y_heads,
                    &z_heads,
                    gdn_cfg,
                    seq,
                    bench_target.target,
                )
            },
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-qlinear",
            model_dir: args.model.display().to_string(),
            layer: args.layer,
            seqs,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            stream_mode: bench_target.label,
            include_cxx_qmm: args.include_cxx_qmm,
            hidden_size: gdn_cfg.hidden_size,
            conv_dim: gdn_cfg.conv_dim(),
            value_dim: gdn_cfg.value_dim(),
            qkvz_out_dim: gdn_cfg.conv_dim() + gdn_cfg.value_dim(),
            num_v_heads: gdn_cfg.num_v_heads,
            head_v_dim: gdn_cfg.head_v_dim,
            group_size: qkvz.group_size,
            bits: qkvz.bits,
        },
        records,
    };
    std::fs::write(&args.out, serde_json::to_string_pretty(&output)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;
    Ok(())
}

impl QuantProjection {
    fn forward_direct_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        let mut y = mlx::quantization::quantized_matmul_on(
            x,
            &self.weight,
            &self.scales,
            self.biases.as_ref(),
            true,
            Some(self.group_size),
            Some(self.bits),
            "affine",
            target,
        )?;
        if let Some(bias) = &self.bias {
            y = mlx::ops::binary::add_on(&y, bias, target)?;
        }
        Ok(y)
    }

    fn forward_linear_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        self.linear.forward_on(x, target)
    }

    fn forward_self_qmm_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        let biases = self
            .biases
            .as_ref()
            .ok_or_else(|| anyhow!("self_qmm diagnostic requires affine quant biases"))?;
        let mut y = self_qmm::qmm_t_on(
            x,
            &self.weight,
            &self.scales,
            biases,
            self.bits,
            self.group_size,
            target,
        )?;
        if let Some(bias) = &self.bias {
            y = mlx::ops::binary::add_on(&y, bias, target)?;
        }
        Ok(y)
    }

    fn cxx_qmm_timings_ms(
        &self,
        x: &Array,
        runs: usize,
        target: StreamOrDevice,
    ) -> Result<Vec<f64>> {
        mlx::quantization::quantized_matmul_bench_ms(
            x,
            &self.weight,
            &self.scales,
            self.biases.as_ref(),
            true,
            Some(self.group_size),
            Some(self.bits),
            "affine",
            runs,
            target,
        )
        .map_err(anyhow::Error::from)
    }

    fn output_shape_for(&self, x: &Array) -> Vec<i32> {
        let mut shape = x.shape().as_slice().to_vec();
        if let Some(last) = shape.last_mut() {
            *last = self.weight.shape().as_slice()[0];
        }
        shape
    }
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

fn load_quant_projection(loader: &Loader, prefix: &str) -> Result<QuantProjection> {
    let qmeta = loader
        .quant_meta_for(prefix)
        .ok_or_else(|| anyhow!("{prefix}: missing quantization metadata"))?;
    let weight = loader.tensor(&format!("{prefix}.weight"))?.clone();
    let scales = loader.tensor(&format!("{prefix}.scales"))?.clone();
    let biases = loader.tensor_opt(&format!("{prefix}.biases")).cloned();
    let bias = loader.tensor_opt(&format!("{prefix}.bias")).cloned();
    Ok(make_quant_projection(
        weight, scales, biases, bias, qmeta, prefix,
    ))
}

fn load_fused_projection(
    loader: &Loader,
    prefix: &str,
    left_name: &str,
    right_name: &str,
) -> Result<QuantProjection> {
    let left = format!("{prefix}.{left_name}");
    let right = format!("{prefix}.{right_name}");
    let left_meta = loader
        .quant_meta_for(&left)
        .ok_or_else(|| anyhow!("{left}: missing quantization metadata"))?;
    let right_meta = loader
        .quant_meta_for(&right)
        .ok_or_else(|| anyhow!("{right}: missing quantization metadata"))?;
    if left_meta != right_meta {
        return Err(anyhow!(
            "{left} and {right} quantization metadata differ: {left_meta:?} vs {right_meta:?}"
        ));
    }

    let left_weight = loader.tensor(&format!("{left}.weight"))?.clone();
    let right_weight = loader.tensor(&format!("{right}.weight"))?.clone();
    let weight = mlx::ops::shape::concatenate(&[&left_weight, &right_weight], 0)?;

    let left_scales = loader.tensor(&format!("{left}.scales"))?.clone();
    let right_scales = loader.tensor(&format!("{right}.scales"))?.clone();
    let scales = mlx::ops::shape::concatenate(&[&left_scales, &right_scales], 0)?;

    let biases = concat_optional(
        loader.tensor_opt(&format!("{left}.biases")).cloned(),
        loader.tensor_opt(&format!("{right}.biases")).cloned(),
        "quant biases",
    )?;
    let bias = concat_optional(
        loader.tensor_opt(&format!("{left}.bias")).cloned(),
        loader.tensor_opt(&format!("{right}.bias")).cloned(),
        "linear bias",
    )?;

    let mut to_eval: Vec<&Array> = vec![&weight, &scales];
    if let Some(v) = &biases {
        to_eval.push(v);
    }
    if let Some(v) = &bias {
        to_eval.push(v);
    }
    mlx::transforms::eval(&to_eval)?;

    Ok(make_quant_projection(
        weight,
        scales,
        biases,
        bias,
        left_meta,
        &format!("{left}+{right}"),
    ))
}

fn make_quant_projection(
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    bias: Option<Array>,
    qmeta: QuantMeta,
    _label: &str,
) -> QuantProjection {
    let linear = Linear::new_quant(
        weight.clone(),
        scales.clone(),
        biases.clone(),
        bias.clone(),
        qmeta.group_size,
        qmeta.bits,
    );
    QuantProjection {
        linear,
        weight,
        scales,
        biases,
        bias,
        group_size: qmeta.group_size,
        bits: qmeta.bits,
    }
}

fn concat_optional(
    left: Option<Array>,
    right: Option<Array>,
    label: &str,
) -> Result<Option<Array>> {
    match (left, right) {
        (Some(a), Some(b)) => Ok(Some(mlx::ops::shape::concatenate(&[&a, &b], 0)?)),
        (None, None) => Ok(None),
        _ => Err(anyhow!("fused projection {label} presence mismatch")),
    }
}

fn qkvz_linear_slice(
    qkvz: &QuantProjection,
    x: &Array,
    cfg: GatedDeltaNetConfig,
    seq: i32,
    target: StreamOrDevice,
) -> Result<Vec<Array>> {
    let qkvz_out = qkvz.forward_linear_on(x, target)?;
    let conv_dim = cfg.conv_dim();
    let value_dim = cfg.value_dim();
    let qkv = mlx::ops::indexing::slice_strided_on(
        &qkvz_out,
        &[0_i32, 0, 0][..],
        &[1_i32, seq, conv_dim][..],
        &[1_i32, 1, 1][..],
        target,
    )?;
    let z = mlx::ops::indexing::slice_strided_on(
        &qkvz_out,
        &[0_i32, 0, conv_dim][..],
        &[1_i32, seq, conv_dim + value_dim][..],
        &[1_i32, 1, 1][..],
        target,
    )?
    .reshape_on((1_i32, seq, cfg.num_v_heads, cfg.head_v_dim), target)?;
    Ok(vec![qkv, z])
}

fn norm_out_linear(
    norm: &RmsNormGated,
    out_proj: &QuantProjection,
    y_heads: &Array,
    z_heads: &Array,
    cfg: GatedDeltaNetConfig,
    seq: i32,
    target: StreamOrDevice,
) -> Result<Vec<Array>> {
    let normed = norm.forward_on(y_heads, Some(z_heads), target)?;
    let normed_flat = normed.reshape_on((1_i32, seq, cfg.value_dim()), target)?;
    out_proj
        .forward_linear_on(&normed_flat, target)
        .map(|out| vec![out])
}

fn bench_cxx_qmm(
    seq: i32,
    case: &'static str,
    projection: &QuantProjection,
    x: &Array,
    warmup_runs: usize,
    runs: usize,
    target: StreamOrDevice,
) -> Result<Record> {
    let warmups = projection.cxx_qmm_timings_ms(x, warmup_runs, target)?;
    let values_ms = projection.cxx_qmm_timings_ms(x, runs, target)?;
    Ok(Record {
        seq,
        case,
        output_shapes: vec![projection.output_shape_for(x)],
        summary: summarize(&values_ms),
        warmups,
        values_ms,
    })
}

fn random_bf16<S>(seed: u64, shape: S) -> Result<Array>
where
    S: mlx::IntoShape,
{
    let key = random::key(seed).context("random key")?;
    let x = random::normal()
        .shape(shape)
        .dtype(Dtype::Bfloat16)
        .key(&key)
        .sample()
        .context("sample synthetic tensor")?;
    Ok(x)
}

fn bench_case<F>(
    seq: i32,
    case: &'static str,
    warmup_runs: usize,
    runs: usize,
    mut f: F,
) -> Result<Record>
where
    F: FnMut() -> Result<Vec<Array>>,
{
    let mut output_shapes = Vec::new();
    let mut warmups = Vec::with_capacity(warmup_runs);
    for _ in 0..warmup_runs {
        let (elapsed_ms, shapes) = time_once(&mut f)?;
        output_shapes = shapes;
        warmups.push(elapsed_ms);
    }

    let mut values_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let (elapsed_ms, shapes) = time_once(&mut f)?;
        output_shapes = shapes;
        values_ms.push(elapsed_ms);
    }

    Ok(Record {
        seq,
        case,
        output_shapes,
        summary: summarize(&values_ms),
        warmups,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interpolates_between_points() {
        assert_eq!(percentile(&[1.0, 3.0], 50.0), Some(2.0));
    }

    #[test]
    fn percentile_sorts_before_interpolating() {
        assert_eq!(percentile(&[9.0, 1.0, 5.0], 50.0), Some(5.0));
    }

    #[test]
    fn percentile_returns_none_for_empty_input() {
        assert_eq!(percentile(&[], 95.0), None);
    }

    #[test]
    fn stream_mode_cli_names_are_stable() {
        use clap::ValueEnum;

        let names: Vec<String> = StreamMode::value_variants()
            .iter()
            .filter_map(|mode| {
                mode.to_possible_value()
                    .map(|value| value.get_name().to_string())
            })
            .collect();
        assert_eq!(
            names,
            vec!["default", "explicit-default", "new-default", "new-explicit"]
        );
    }

    #[test]
    fn cxx_quantized_matmul_bench_ms_allows_zero_runs() {
        let x = Array::zeros((1_i32, 1, 4), Dtype::Float32).expect("x");
        let w = Array::zeros((2_i32, 1), Dtype::Uint32).expect("w");
        let scales = Array::zeros((2_i32, 1), Dtype::Float32).expect("scales");

        let timings = mlx::quantization::quantized_matmul_bench_ms(
            &x,
            &w,
            &scales,
            None,
            true,
            Some(32),
            Some(4),
            "affine",
            0,
            StreamOrDevice::default(),
        )
        .expect("zero-run cxx qmm bench");

        assert!(timings.is_empty());
    }
}
