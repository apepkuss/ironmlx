//! Quantized projection calibration for speculative Q>1 verify shapes.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::{Loader, QuantMode};
use ironmlx::nn::Linear;
use mlx::{random, Array, Dtype, StreamOrDevice};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-verify-qmm-bench",
    about = "Calibrate quantized projection morphologies used by exact Q>1 verify"
)]
struct Args {
    /// Local model directory.
    #[arg(long)]
    model: PathBuf,

    /// Quantized linear prefix, without `.weight`/`.scales`.
    #[arg(long)]
    projection: String,

    /// Batch widths to sweep. Repeat the flag or pass a comma-separated list.
    #[arg(long, value_delimiter = ',')]
    batch: Vec<i32>,

    /// Verify widths to sweep. Repeat the flag or pass a comma-separated list.
    #[arg(long, value_delimiter = ',')]
    verify_width: Vec<i32>,

    /// Timed runs per morphology.
    #[arg(long, default_value_t = 30)]
    runs: usize,

    /// Warmup runs per morphology.
    #[arg(long, default_value_t = 5)]
    warmup_runs: usize,

    /// Synthetic input dtype.
    #[arg(long, value_enum, default_value_t = InputDtype::Bf16)]
    dtype: InputDtype,

    /// PRNG seed.
    #[arg(long, default_value_t = 20260727)]
    seed: u64,

    /// JSON output path.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InputDtype {
    Bf16,
    F16,
}

impl InputDtype {
    fn mlx(self) -> Dtype {
        match self {
            Self::Bf16 => Dtype::Bfloat16,
            Self::F16 => Dtype::Float16,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bfloat16",
            Self::F16 => "float16",
        }
    }
}

struct QuantProjection {
    linear: Linear,
    weight: Array,
    scales: Array,
    biases: Array,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
    input_width: i32,
    output_width: i32,
}

#[derive(Serialize)]
struct BenchOutput {
    metadata: Metadata,
    records: Vec<Record>,
}

#[derive(Serialize)]
struct Metadata {
    model: String,
    projection: String,
    architecture: String,
    dtype: &'static str,
    bits: i32,
    group_size: i32,
    input_width: i32,
    output_width: i32,
    batches: Vec<i32>,
    verify_widths: Vec<i32>,
    warmup_runs: usize,
    runs: usize,
}

#[derive(Serialize)]
struct Record {
    batch: i32,
    verify_width: i32,
    morphology: &'static str,
    summary: Summary,
    values_ms: Vec<f64>,
    max_abs_diff_from_sequential_q1: f32,
    argmax_match_ratio: f64,
    argmax_tokens: Vec<u32>,
    output_f32_fingerprint: String,
}

#[derive(Serialize)]
struct Summary {
    p50_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.runs == 0 {
        bail!("--runs must be positive");
    }
    let batches = nonempty_positive(args.batch, 1, "--batch")?;
    let verify_widths = nonempty_positive(args.verify_width, 2, "--verify-width")?;
    let projection = load_projection(&args.model, &args.projection)?;
    let architecture = mlx::metal::architecture().unwrap_or_else(|_| "unknown".to_string());
    let target = StreamOrDevice::default();

    let mut records = Vec::new();
    for &batch in &batches {
        for &verify_width in &verify_widths {
            let input = random_input(
                args.seed
                    .wrapping_add(batch as u64 * 1_003)
                    .wrapping_add(verify_width as u64 * 10_007),
                (batch, verify_width, projection.input_width),
                args.dtype.mlx(),
            )?;
            mlx::transforms::eval(&[&input])?;
            mlx::transforms::synchronize()?;

            let reference = projection.sequential_q1(&input, target, true)?;
            materialize(&reference)?;
            records.push(bench_morphology(
                batch,
                verify_width,
                "sequential-q1-eager",
                &reference,
                args.warmup_runs,
                args.runs,
                || projection.sequential_q1(&input, target, true),
            )?);
            records.push(bench_morphology(
                batch,
                verify_width,
                "sequential-q1-lazy",
                &reference,
                args.warmup_runs,
                args.runs,
                || projection.sequential_q1(&input, target, false),
            )?);
            records.push(bench_morphology(
                batch,
                verify_width,
                "native-batched",
                &reference,
                args.warmup_runs,
                args.runs,
                || projection.native_batched(&input, target),
            )?);
            records.push(bench_morphology(
                batch,
                verify_width,
                "position-isolated",
                &reference,
                args.warmup_runs,
                args.runs,
                || projection.position_isolated(&input, target),
            )?);
            if batch == 1 {
                records.push(bench_morphology(
                    batch,
                    verify_width,
                    "verify-splitk-msg-candidate",
                    &reference,
                    args.warmup_runs,
                    args.runs,
                    || projection.verify_candidate(&input, target),
                )?);
            }
        }
    }

    let output = BenchOutput {
        metadata: Metadata {
            model: args.model.display().to_string(),
            projection: args.projection,
            architecture,
            dtype: args.dtype.label(),
            bits: projection.bits,
            group_size: projection.group_size,
            input_width: projection.input_width,
            output_width: projection.output_width,
            batches,
            verify_widths,
            warmup_runs: args.warmup_runs,
            runs: args.runs,
        },
        records,
    };
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(&args.out, format!("{json}\n"))
        .with_context(|| format!("writing {}", args.out.display()))?;
    println!("{}", args.out.display());
    Ok(())
}

fn nonempty_positive(mut values: Vec<i32>, default: i32, flag: &str) -> Result<Vec<i32>> {
    if values.is_empty() {
        values.push(default);
    }
    if let Some(value) = values.iter().find(|&&value| value <= 0) {
        bail!("{flag} values must be positive, got {value}");
    }
    values.sort_unstable();
    values.dedup();
    Ok(values)
}

fn load_projection(model: &Path, prefix: &str) -> Result<QuantProjection> {
    let loader = Loader::open(model).context("opening model")?;
    let meta = loader
        .quant_meta_for(prefix)
        .ok_or_else(|| anyhow!("{prefix}: missing quantization metadata"))?;
    if meta.mode != QuantMode::Affine || meta.bits != 8 {
        bail!(
            "{prefix}: P1-D calibration requires Affine8, got {:?} {}-bit",
            meta.mode,
            meta.bits
        );
    }
    let weight = loader
        .tensor(&format!("{prefix}.weight"))
        .with_context(|| format!("loading {prefix}.weight"))?
        .clone();
    let scales = loader
        .tensor(&format!("{prefix}.scales"))
        .with_context(|| format!("loading {prefix}.scales"))?
        .clone();
    let biases = loader
        .tensor(&format!("{prefix}.biases"))
        .with_context(|| format!("loading {prefix}.biases"))?
        .clone();
    let bias = loader.tensor_opt(&format!("{prefix}.bias")).cloned();
    let shape = weight.shape();
    let [output_width, packed_input_width] = shape.as_slice() else {
        bail!("{prefix}.weight must be rank 2, got {:?}", shape.as_slice());
    };
    let input_width = packed_input_width
        .checked_mul(32)
        .and_then(|bits| bits.checked_div(meta.bits))
        .ok_or_else(|| anyhow!("{prefix}: invalid packed input width"))?;
    let linear = Linear::new_quant(
        weight.clone(),
        scales.clone(),
        Some(biases.clone()),
        bias.clone(),
        meta.group_size,
        meta.bits,
    );
    let mut arrays = vec![&weight, &scales, &biases];
    if let Some(bias) = &bias {
        arrays.push(bias);
    }
    mlx::transforms::eval(&arrays)?;
    mlx::transforms::synchronize()?;
    Ok(QuantProjection {
        linear,
        weight,
        scales,
        biases,
        bias,
        group_size: meta.group_size,
        bits: meta.bits,
        input_width,
        output_width: *output_width,
    })
}

impl QuantProjection {
    fn native_batched(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        self.quantized_matmul(input, target)
    }

    fn position_isolated(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let isolated = input.transpose_axes_on(&[1_i32, 0, 2][..], target)?;
        let mut output = mlx::quantization::quantized_matmul_batch_isolated_on(
            &isolated,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            Some(self.group_size),
            Some(self.bits),
            "affine",
            target,
        )?;
        if let Some(bias) = &self.bias {
            output = &output + bias;
        }
        Ok(output.transpose_axes_on(&[1_i32, 0, 2][..], target)?)
    }

    fn sequential_q1(&self, input: &Array, target: StreamOrDevice, eager: bool) -> Result<Array> {
        let input_shape = input.shape();
        let [batch, verify_width, input_width] = input_shape.as_slice() else {
            bail!("sequential Q1 input must be rank 3");
        };
        let mut outputs = Vec::with_capacity(*verify_width as usize);
        for depth in 0..*verify_width {
            let position = mlx::ops::indexing::slice_strided_on(
                input,
                &[0_i32, depth, 0][..],
                &[*batch, depth + 1, *input_width][..],
                &[1_i32, 1, 1][..],
                target,
            )?;
            let output = self.quantized_matmul(&position, target)?;
            if eager {
                materialize(&output)?;
            }
            outputs.push(output);
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        Ok(mlx::ops::shape::concatenate_on(&refs, 1, target)?)
    }

    fn verify_candidate(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        self.linear.forward_mtp_verify_on(input, target)
    }

    fn quantized_matmul(&self, input: &Array, target: StreamOrDevice) -> Result<Array> {
        let mut output = mlx::quantization::quantized_matmul_on(
            input,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            Some(self.group_size),
            Some(self.bits),
            "affine",
            target,
        )?;
        if let Some(bias) = &self.bias {
            output = &output + bias;
        }
        Ok(output)
    }
}

fn random_input<S>(seed: u64, shape: S, dtype: Dtype) -> Result<Array>
where
    S: mlx::IntoShape,
{
    let key = random::key(seed)?;
    Ok(random::normal()
        .shape(shape)
        .dtype(dtype)
        .key(&key)
        .sample()?)
}

fn bench_morphology<F>(
    batch: i32,
    verify_width: i32,
    morphology: &'static str,
    reference: &Array,
    warmup_runs: usize,
    runs: usize,
    mut operation: F,
) -> Result<Record>
where
    F: FnMut() -> Result<Array>,
{
    for _ in 0..warmup_runs {
        let output = operation()?;
        materialize(&output)?;
    }
    let mut values_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let output = operation()?;
        materialize(&output)?;
        values_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let output = operation()?;
    materialize(&output)?;
    let (
        max_abs_diff_from_sequential_q1,
        argmax_match_ratio,
        argmax_tokens,
        output_f32_fingerprint,
    ) = compare_outputs(reference, &output)?;
    Ok(Record {
        batch,
        verify_width,
        morphology,
        summary: summarize(&values_ms),
        values_ms,
        max_abs_diff_from_sequential_q1,
        argmax_match_ratio,
        argmax_tokens,
        output_f32_fingerprint,
    })
}

fn materialize(output: &Array) -> Result<()> {
    mlx::transforms::eval(&[output])?;
    mlx::transforms::synchronize()?;
    Ok(())
}

fn compare_outputs(reference: &Array, candidate: &Array) -> Result<(f32, f64, Vec<u32>, String)> {
    if reference.shape() != candidate.shape() {
        bail!(
            "output shape mismatch: reference={:?} candidate={:?}",
            reference.shape().as_slice(),
            candidate.shape().as_slice()
        );
    }
    let reference_f32 = mlx::ops::cast::astype(reference, Dtype::Float32)?.to_vec::<f32>()?;
    let candidate_f32 = mlx::ops::cast::astype(candidate, Dtype::Float32)?.to_vec::<f32>()?;
    let max_abs = reference_f32
        .iter()
        .zip(candidate_f32.iter())
        .map(|(reference, candidate)| (reference - candidate).abs())
        .fold(0.0_f32, f32::max);

    let reference_argmax = mlx::ops::reduction::argmax(reference, -1, false)?.to_vec::<u32>()?;
    let candidate_argmax = mlx::ops::reduction::argmax(candidate, -1, false)?.to_vec::<u32>()?;
    let matches = reference_argmax
        .iter()
        .zip(candidate_argmax.iter())
        .filter(|(reference, candidate)| reference == candidate)
        .count();
    let ratio = if reference_argmax.is_empty() {
        1.0
    } else {
        matches as f64 / reference_argmax.len() as f64
    };
    let fingerprint = candidate_f32
        .iter()
        .fold(0xcbf29ce484222325_u64, |hash, value| {
            (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3)
        });
    Ok((
        max_abs,
        ratio,
        candidate_argmax,
        format!("{fingerprint:016x}"),
    ))
}

fn summarize(values: &[f64]) -> Summary {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Summary {
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        mean_ms: sorted.iter().sum::<f64>() / sorted.len() as f64,
    }
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let rank = percentile * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = rank - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}
