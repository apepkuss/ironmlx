use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use mlx::{random, Array, Device, Dtype, Shape, StreamOrDevice};
use serde::{Deserialize, Serialize};

const DEFAULT_WEIGHT_KEY: &str = "language_model.model.layers.0.mlp.gate_proj.weight";

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-bf16-linear-layout-ab",
    about = "Dense bf16 Linear layout A/B benchmark: lazy transpose vs row-major materialized transpose",
    version
)]
struct Args {
    /// Local MLX model directory containing safetensors files.
    #[arg(long)]
    model: PathBuf,

    /// Exact safetensors tensor key to benchmark. Shape must be [out, in].
    #[arg(long, default_value = DEFAULT_WEIGHT_KEY)]
    weight_key: String,

    /// Benchmark case in `label:batch:seq` format. May be repeated.
    ///
    /// Defaults: decode-c1:1:1, decode-c8:8:1, prefill-2048:1:2048.
    #[arg(long = "case")]
    cases: Vec<String>,

    /// Timed runs per case/path.
    #[arg(long, default_value_t = 30)]
    runs: usize,

    /// Warmup runs per case/path.
    #[arg(long, default_value_t = 8)]
    warmup_runs: usize,

    /// PRNG seed for synthetic hidden states.
    #[arg(long, default_value_t = 20260709)]
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
    /// Preserve production-style behavior: use MLX's current default stream.
    Default,
    /// Pass the current GPU default stream explicitly.
    ExplicitDefault,
    /// Create a fresh GPU stream and set it as this thread's default stream.
    NewDefault,
    /// Create a fresh GPU stream and pass it explicitly.
    NewExplicit,
}

#[derive(Clone, Copy)]
struct BenchTarget {
    label: &'static str,
    target: StreamOrDevice,
}

#[derive(Debug)]
struct BenchCase {
    label: String,
    batch: i32,
    seq: i32,
}

#[derive(Serialize)]
struct BenchOutput {
    meta: Meta,
    materialization: Materialization,
    records: Vec<Record>,
}

#[derive(Serialize)]
struct Meta {
    backend: &'static str,
    model_dir: String,
    weight_key: String,
    weight_shape: Vec<i32>,
    weight_dtype: String,
    warmup_runs: usize,
    measured_runs: usize,
    stream_mode: &'static str,
}

#[derive(Serialize)]
struct Materialization {
    row_major_pretranspose_ms: f64,
    row_major_weight_shape: Vec<i32>,
}

#[derive(Serialize)]
struct Record {
    label: String,
    path: &'static str,
    batch: i32,
    seq: i32,
    logical_m: i32,
    input_shape: Vec<i32>,
    output_shape: Vec<i32>,
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

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.runs == 0 {
        return Err(anyhow!("--runs must be > 0"));
    }
    let cases = parse_cases(&args.cases)?;
    let bench_target = args.stream_mode.configure()?;

    let weight = load_one_safetensors_tensor(&args.model, &args.weight_key)
        .with_context(|| format!("loading {}", args.weight_key))?;
    validate_weight(&weight, &args.weight_key)?;
    mlx::transforms::eval(&[&weight]).context("eval loaded weight")?;

    let (row_major_weight, row_major_pretranspose_ms) =
        time_materialize_row_major_weight(&weight, bench_target.target)
            .context("materializing row-major transposed weight")?;

    let mut records = Vec::with_capacity(cases.len() * 3);
    for (case_idx, case) in cases.iter().enumerate() {
        let x = random_bf16(
            args.seed + case_idx as u64,
            (case.batch, case.seq, input_features(&weight)),
            bench_target.target,
        )
        .with_context(|| format!("sampling input for {}", case.label))?;
        mlx::transforms::eval(&[&x]).context("eval synthetic input")?;

        records.push(bench_case(
            case,
            "lazy-transpose",
            &x,
            args.warmup_runs,
            args.runs,
            || run_lazy_transpose_matmul(&x, &weight, bench_target.target),
        )?);
        records.push(bench_case(
            case,
            "flatten-lazy-transpose",
            &x,
            args.warmup_runs,
            args.runs,
            || run_flatten_lazy_transpose_matmul(&x, &weight, bench_target.target),
        )?);
        records.push(bench_case(
            case,
            "row-major-pretranspose",
            &x,
            args.warmup_runs,
            args.runs,
            || Ok(x.matmul_on(&row_major_weight, bench_target.target)?),
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-bf16-linear-layout-ab",
            model_dir: args.model.display().to_string(),
            weight_key: args.weight_key,
            weight_shape: weight.shape().as_slice().to_vec(),
            weight_dtype: weight.dtype().to_string(),
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            stream_mode: bench_target.label,
        },
        materialization: Materialization {
            row_major_pretranspose_ms,
            row_major_weight_shape: row_major_weight.shape().as_slice().to_vec(),
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

fn run_lazy_transpose_matmul(
    x: &Array,
    weight: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let target = target.into();
    let wt = weight.transpose_on(target)?;
    x.matmul_on(&wt, target).map_err(Into::into)
}

fn run_flatten_lazy_transpose_matmul(
    x: &Array,
    weight: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let target = target.into();
    let x_shape = x.shape();
    let dims = x_shape.as_slice();
    if dims.is_empty() {
        return Err(anyhow!("flatten-lazy-transpose requires non-scalar input"));
    }
    let in_features = *dims.last().expect("non-empty");
    let weight_shape = weight.shape();
    let weight_dims = weight_shape.as_slice();
    if weight_dims.len() != 2 {
        return Err(anyhow!(
            "flatten-lazy-transpose requires rank-2 weight, got {}",
            weight_shape
        ));
    }
    if in_features != weight_dims[1] {
        return Err(anyhow!(
            "flatten-lazy-transpose input features {in_features} != weight input features {}",
            weight_dims[1]
        ));
    }

    let wt = weight.transpose_on(target)?;
    if dims.len() <= 2 {
        return x.matmul_on(&wt, target).map_err(Into::into);
    }

    let logical_m: i32 = dims[..dims.len() - 1].iter().product();
    let flat = x.reshape((logical_m, in_features))?;
    let y_flat = flat.matmul_on(&wt, target)?;
    let mut out_shape = dims.to_vec();
    *out_shape.last_mut().expect("non-empty") = weight_dims[0];
    y_flat.reshape(Shape::from(out_shape)).map_err(Into::into)
}

#[cfg(test)]
fn run_row_major_materialized_matmul(
    x: &Array,
    weight: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let target = target.into();
    let wt = materialize_row_major_weight(weight, target)?;
    x.matmul_on(&wt, target).map_err(Into::into)
}

fn materialize_row_major_weight(weight: &Array, target: StreamOrDevice) -> Result<Array> {
    let wt = weight.transpose_on(target)?.contiguous_on(false, target)?;
    mlx::transforms::eval(&[&wt])?;
    Ok(wt)
}

fn time_materialize_row_major_weight(
    weight: &Array,
    target: StreamOrDevice,
) -> Result<(Array, f64)> {
    let started = Instant::now();
    let wt = materialize_row_major_weight(weight, target)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok((wt, elapsed_ms))
}

fn bench_case<F>(
    case: &BenchCase,
    path: &'static str,
    x: &Array,
    warmup_runs: usize,
    runs: usize,
    mut f: F,
) -> Result<Record>
where
    F: FnMut() -> Result<Array>,
{
    let mut output_shape = Vec::new();
    let mut warmups_ms = Vec::with_capacity(warmup_runs);
    for _ in 0..warmup_runs {
        let (elapsed_ms, shape) = time_once(&mut f)?;
        output_shape = shape;
        warmups_ms.push(elapsed_ms);
    }

    let mut values_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let (elapsed_ms, shape) = time_once(&mut f)?;
        output_shape = shape;
        values_ms.push(elapsed_ms);
    }

    Ok(Record {
        label: case.label.clone(),
        path,
        batch: case.batch,
        seq: case.seq,
        logical_m: case.batch * case.seq,
        input_shape: x.shape().as_slice().to_vec(),
        output_shape,
        summary: summarize(&values_ms),
        warmups_ms,
        values_ms,
    })
}

fn time_once<F>(f: &mut F) -> Result<(f64, Vec<i32>)>
where
    F: FnMut() -> Result<Array>,
{
    let started = Instant::now();
    let out = f()?;
    mlx::transforms::eval(&[&out])?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok((elapsed_ms, out.shape().as_slice().to_vec()))
}

fn parse_cases(raw_cases: &[String]) -> Result<Vec<BenchCase>> {
    let defaults;
    let cases = if raw_cases.is_empty() {
        defaults = vec![
            "decode-c1:1:1".to_string(),
            "decode-c8:8:1".to_string(),
            "prefill-2048:1:2048".to_string(),
        ];
        defaults.as_slice()
    } else {
        raw_cases
    };

    cases
        .iter()
        .map(|raw| {
            let parts: Vec<&str> = raw.split(':').collect();
            if parts.len() != 3 {
                return Err(anyhow!(
                    "invalid --case `{raw}`; expected `label:batch:seq`"
                ));
            }
            let label = parts[0].to_string();
            if label.is_empty() {
                return Err(anyhow!("invalid --case `{raw}`; label must not be empty"));
            }
            let batch = parse_positive_i32(parts[1], raw, "batch")?;
            let seq = parse_positive_i32(parts[2], raw, "seq")?;
            Ok(BenchCase { label, batch, seq })
        })
        .collect()
}

fn parse_positive_i32(value: &str, raw: &str, field: &str) -> Result<i32> {
    let parsed: i32 = value
        .parse()
        .with_context(|| format!("invalid --case `{raw}`; {field} must be an integer"))?;
    if parsed <= 0 {
        return Err(anyhow!("invalid --case `{raw}`; {field} must be positive"));
    }
    Ok(parsed)
}

fn load_one_safetensors_tensor(model_dir: &Path, weight_key: &str) -> Result<Array> {
    let single = model_dir.join("model.safetensors");
    let path = if single.exists() {
        single
    } else {
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_file = std::fs::File::open(&index_path)
            .with_context(|| format!("opening {}", index_path.display()))?;
        let index: SafetensorsIndex = serde_json::from_reader(index_file)
            .with_context(|| format!("parsing {}", index_path.display()))?;
        let shard = index
            .weight_map
            .get(weight_key)
            .ok_or_else(|| anyhow!("safetensors index has no tensor key `{weight_key}`"))?;
        model_dir.join(shard)
    };

    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF8 safetensors path: {}", path.display()))?;
    let (mut tensors, _metadata) = mlx::io::load_safetensors(path_str)
        .with_context(|| format!("load_safetensors {}", path.display()))?;
    tensors.remove(weight_key).ok_or_else(|| {
        anyhow!(
            "{} does not contain tensor key `{weight_key}`",
            path.display()
        )
    })
}

fn validate_weight(weight: &Array, weight_key: &str) -> Result<()> {
    if weight.ndim() != 2 {
        return Err(anyhow!(
            "`{weight_key}` must be rank-2 [out, in], got shape {}",
            weight.shape()
        ));
    }
    if weight.dtype() != Dtype::Bfloat16 {
        return Err(anyhow!(
            "`{weight_key}` must be bf16 for this dense bf16 layout bench, got {}",
            weight.dtype()
        ));
    }
    Ok(())
}

fn input_features(weight: &Array) -> i32 {
    weight.shape().as_slice()[1]
}

fn random_bf16<S>(seed: u64, shape: S, target: StreamOrDevice) -> Result<Array>
where
    S: mlx::IntoShape,
{
    let key = random::key(seed).context("random key")?;
    random::normal()
        .shape(shape)
        .dtype(Dtype::Bfloat16)
        .key(&key)
        .stream(target)
        .sample()
        .context("sample synthetic bf16 input")
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
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted.get(lo).copied();
    }
    let w = rank - lo as f64;
    Some(sorted[lo] * (1.0 - w) + sorted[hi] * w)
}

fn print_summary(output: &BenchOutput) {
    println!(
        "weight={} shape={:?} dtype={} row_major_pretranspose_ms={:.3}",
        output.meta.weight_key,
        output.meta.weight_shape,
        output.meta.weight_dtype,
        output.materialization.row_major_pretranspose_ms
    );
    for record in &output.records {
        let p50 = record
            .summary
            .p50_ms
            .map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}"));
        let p95 = record
            .summary
            .p95_ms
            .map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}"));
        let mean = record
            .summary
            .mean_ms
            .map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}"));
        println!(
            "{} {} input={:?} output={:?} p50_ms={} p95_ms={} mean_ms={}",
            record.label, record.path, record.input_shape, record.output_shape, p50, p95, mean
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_and_row_major_materialized_paths_match_small_bf16() -> Result<()> {
        let x =
            Array::try_from((&[1.0_f32, -2.0, 0.5, 3.0][..], (2, 2)))?.astype(Dtype::Bfloat16)?;
        let weight = Array::try_from((&[0.25_f32, -0.5, 1.5, 2.0, -1.25, 0.75][..], (3, 2)))?
            .astype(Dtype::Bfloat16)?;

        let lazy = run_lazy_transpose_matmul(&x, &weight, ())?;
        let row_major = run_row_major_materialized_matmul(&x, &weight, ())?;
        mlx::transforms::eval(&[&lazy, &row_major])?;

        let lazy = lazy.astype(Dtype::Float32)?.to_vec::<f32>()?;
        let row_major = row_major.astype(Dtype::Float32)?.to_vec::<f32>()?;
        assert_eq!(lazy.len(), row_major.len());
        for (left, right) in lazy.iter().zip(row_major.iter()) {
            assert!((left - right).abs() <= 1e-3, "left={left}, right={right}");
        }
        Ok(())
    }

    #[test]
    fn lazy_and_flatten_paths_match_rank3_bf16() -> Result<()> {
        let x = Array::try_from((
            &[1.0_f32, -2.0, 0.5, 3.0, 0.25, 1.25, -1.0, 2.0][..],
            (2, 2, 2),
        ))?
        .astype(Dtype::Bfloat16)?;
        let weight = Array::try_from((&[0.25_f32, -0.5, 1.5, 2.0, -1.25, 0.75][..], (3, 2)))?
            .astype(Dtype::Bfloat16)?;

        let lazy = run_lazy_transpose_matmul(&x, &weight, ())?;
        let flat = run_flatten_lazy_transpose_matmul(&x, &weight, ())?;
        mlx::transforms::eval(&[&lazy, &flat])?;

        assert_eq!(lazy.shape().as_slice(), &[2, 2, 3]);
        assert_eq!(flat.shape().as_slice(), &[2, 2, 3]);
        let lazy = lazy.astype(Dtype::Float32)?.to_vec::<f32>()?;
        let flat = flat.astype(Dtype::Float32)?.to_vec::<f32>()?;
        assert_eq!(lazy.len(), flat.len());
        for (left, right) in lazy.iter().zip(flat.iter()) {
            assert!((left - right).abs() <= 1e-3, "left={left}, right={right}");
        }
        Ok(())
    }

    #[test]
    fn parses_case_label_batch_seq() -> Result<()> {
        let cases = parse_cases(&["decode-c8:8:1".to_string()])?;
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].label, "decode-c8");
        assert_eq!(cases[0].batch, 8);
        assert_eq!(cases[0].seq, 1);
        Ok(())
    }

    #[test]
    fn rejects_non_positive_case_dims() {
        let err = parse_cases(&["bad:0:1".to_string()]).unwrap_err();
        assert!(err.to_string().contains("batch must be positive"));
    }
}
