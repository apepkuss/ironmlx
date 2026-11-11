//! Narrow affine4 projection micro-benchmark for the M1 Pro/Max GPU plan.
//!
//! Compares, on Qwen3.8-27B projection shapes, the routes that
//! `--m1-profile off|auto` select for 1–16 rows:
//!
//! - `off_batch`: MLX `quantized_matmul` on `[rows, 1, K]` (batched decode).
//! - `off_seq`: MLX `quantized_matmul` on `[1, rows, K]` (multi-token rows).
//! - `m1_nv2`: the M1 Pro recipe. One row uses product-stable QMV; 2–16 rows
//!   use `qmv_fast_wide` with at most 2 rows per threadgroup.
//! - `m1_nv4`: the M1 Max recipe (at most 4 rows per threadgroup), runnable on
//!   any Apple GPU because the kernel is not hardware-gated.
//!
//! Each pass evaluates several distinct weight copies so the working set
//! exceeds the system-level cache, as in real decode. Routes are interleaved
//! within each round and their order rotates between rounds. A sustained
//! warmup keeps the GPU clock up before measuring.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use mlx::{Array, Dtype};

const DEFAULT_SHAPES: &str = "gate_up:17408x5120,down:5120x17408,gdn_qkv:10240x5120,\
gdn_z:6144x5120,out_o:5120x6144,attn_q:12288x5120,attn_kv:1024x5120,lm_head:248320x5120";

#[derive(Parser, Debug)]
#[command(about = "M1 narrow affine4 projection micro-benchmark")]
struct Args {
    /// Comma-separated `name:NxK` projection shapes (N outputs, K inputs).
    #[arg(long, default_value = DEFAULT_SHAPES)]
    shapes: String,

    /// Comma-separated row counts.
    #[arg(long, default_value = "1,2,3,4,5,8,16")]
    rows: String,

    /// Comma-separated routes: off_batch, off_seq, m1_nv2, m1_nv4.
    #[arg(long, default_value = "off_batch,off_seq,m1_nv2,m1_nv4")]
    routes: String,

    /// Minimum weight bytes per pass; copies are added until it is reached.
    #[arg(long, default_value_t = 512)]
    working_set_mib: usize,

    /// Interleaved measurement rounds per shape and row count.
    #[arg(long, default_value_t = 5)]
    rounds: usize,

    /// Timed passes per route in each round (median is used).
    #[arg(long, default_value_t = 7)]
    passes: usize,

    /// Untimed passes before each route's timed passes.
    #[arg(long, default_value_t = 2)]
    warmup_passes: usize,

    /// Sustained GPU warmup before the first measurement.
    #[arg(long, default_value_t = 30)]
    warmup_secs: u64,

    /// Size of the read-bandwidth reference buffer.
    #[arg(long, default_value_t = 1024)]
    peak_mib: usize,

    /// Per-round CSV output.
    #[arg(long)]
    out: PathBuf,

    /// MLX metallib. Defaults to `$MLX_METAL_PATH/mlx.metallib`, then
    /// `$MLX_DIR/lib/mlx.metallib`.
    #[arg(long)]
    mlx_metallib: Option<PathBuf>,
}

fn configure_metallib(explicit: Option<&PathBuf>) -> Result<()> {
    let candidate = explicit.cloned().or_else(|| {
        std::env::var_os("MLX_METAL_PATH")
            .map(|dir| PathBuf::from(dir).join("mlx.metallib"))
            .or_else(|| {
                std::env::var_os("MLX_DIR").map(|dir| PathBuf::from(dir).join("lib/mlx.metallib"))
            })
    });
    let path = candidate.context("pass --mlx-metallib or set MLX_METAL_PATH / MLX_DIR")?;
    ensure!(path.is_file(), "metallib not found: {}", path.display());
    mlx::metal::set_metallib_path(path.to_str().context("metallib path must be UTF-8")?)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    OffBatch,
    OffSeq,
    M1Nv2,
    M1Nv4,
}

impl Route {
    fn parse(value: &str) -> Result<Self> {
        Ok(match value.trim() {
            "off_batch" => Self::OffBatch,
            "off_seq" => Self::OffSeq,
            "m1_nv2" => Self::M1Nv2,
            "m1_nv4" => Self::M1Nv4,
            other => bail!("unknown route {other}"),
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::OffBatch => "off_batch",
            Self::OffSeq => "off_seq",
            Self::M1Nv2 => "m1_nv2",
            Self::M1Nv4 => "m1_nv4",
        }
    }

    /// One row is identical across off routes and across M1 recipes.
    fn applies(self, rows: i32) -> bool {
        rows > 1 || matches!(self, Self::OffBatch | Self::M1Nv2)
    }

    fn input_shape(self, rows: i32, k: i32) -> [i32; 3] {
        match self {
            Self::OffBatch => [rows, 1, k],
            _ => [1, rows, k],
        }
    }

    fn forward(self, x: &Array, w: &Weights, rows: i32) -> Result<Array> {
        let y = match self {
            Self::OffBatch | Self::OffSeq => mlx::quantization::quantized_matmul_on(
                x,
                &w.packed,
                &w.scales,
                Some(&w.biases),
                true,
                Some(64),
                Some(4),
                "affine",
                (),
            )?,
            Self::M1Nv2 | Self::M1Nv4 if rows == 1 => {
                mlx::quantization::quantized_matmul_product_stable_on(
                    x,
                    &w.packed,
                    &w.scales,
                    Some(&w.biases),
                    true,
                    Some(64),
                    Some(4),
                    "affine",
                    (),
                )?
            }
            Self::M1Nv2 => {
                mlx::quantization::qmv_fast_wide_on(x, &w.packed, &w.scales, &w.biases, 2, ())?
            }
            Self::M1Nv4 => {
                mlx::quantization::qmv_fast_wide_on(x, &w.packed, &w.scales, &w.biases, 4, ())?
            }
        };
        Ok(y)
    }
}

struct Shape {
    name: String,
    n: i32,
    k: i32,
}

struct Weights {
    packed: Array,
    scales: Array,
    biases: Array,
}

fn parse_shapes(value: &str) -> Result<Vec<Shape>> {
    value
        .split(',')
        .map(|item| {
            let (name, dims) = item
                .trim()
                .split_once(':')
                .with_context(|| format!("shape {item} must be name:NxK"))?;
            let (n, k) = dims
                .split_once('x')
                .with_context(|| format!("shape {item} must be name:NxK"))?;
            let shape = Shape {
                name: name.to_owned(),
                n: n.parse()?,
                k: k.parse()?,
            };
            ensure!(
                shape.n >= 8 && shape.n % 8 == 0 && shape.k >= 512 && shape.k % 512 == 0,
                "shape {item} is outside the M1 route (N % 8, K % 512)"
            );
            Ok(shape)
        })
        .collect()
}

fn weight_bytes(shape: &Shape) -> usize {
    let (n, k) = (shape.n as usize, shape.k as usize);
    n * k / 2 + 2 * n * (k / 64) * 2
}

fn build_weights(shape: &Shape, copies: usize, seed: u64) -> Result<Vec<Weights>> {
    let mut out = Vec::with_capacity(copies);
    for copy in 0..copies {
        let key = mlx::random::key(seed.wrapping_add(copy as u64))?;
        let dense = mlx::random::normal()
            .shape(&[shape.n, shape.k][..])
            .dtype(Dtype::Bfloat16)
            .scale(0.02)
            .key(&key)
            .sample()?;
        let mut q = mlx::quantization::quantize(&dense, Some(64), Some(4), "affine", None)?;
        ensure!(q.len() == 3, "affine quantize returned {} arrays", q.len());
        let biases = q.pop().expect("checked");
        let scales = q.pop().expect("checked");
        let packed = q.pop().expect("checked");
        mlx::transforms::eval(&[&packed, &scales, &biases])?;
        out.push(Weights {
            packed,
            scales,
            biases,
        });
    }
    Ok(out)
}

fn run_pass(route: Route, x: &Array, weights: &[Weights], rows: i32) -> Result<f64> {
    let started = Instant::now();
    let outputs = weights
        .iter()
        .map(|w| route.forward(x, w, rows))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Array> = outputs.iter().collect();
    mlx::transforms::eval(&refs)?;
    Ok(started.elapsed().as_secs_f64())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn bits(x: &Array) -> Result<Vec<u32>> {
    Ok(mlx::ops::cast::astype(x, Dtype::Float32)?
        .to_vec::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn peak_read_gbps(mib: usize, passes: usize) -> Result<f64> {
    let elements = (mib * 1024 * 1024 / 2) as i32;
    let key = mlx::random::key(7)?;
    let buffer = mlx::random::normal()
        .shape(&[elements][..])
        .dtype(Dtype::Bfloat16)
        .key(&key)
        .sample()?;
    mlx::transforms::eval(&[&buffer])?;
    let mut times = Vec::with_capacity(passes);
    for pass in 0..passes + 2 {
        let started = Instant::now();
        let total = mlx::ops::reduction::sum(&buffer, 0, false)?;
        mlx::transforms::eval(&[&total])?;
        if pass >= 2 {
            times.push(started.elapsed().as_secs_f64());
        }
    }
    Ok((mib * 1024 * 1024) as f64 / median(&mut times) / 1e9)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.rounds > 0 && args.passes > 0,
        "rounds and passes must be positive"
    );
    configure_metallib(args.mlx_metallib.as_ref())?;
    let shapes = parse_shapes(&args.shapes)?;
    let rows: Vec<i32> = args
        .rows
        .split(',')
        .map(|r| r.trim().parse::<i32>().context("rows"))
        .collect::<Result<_>>()?;
    ensure!(
        rows.iter().all(|r| (1..=16).contains(r)),
        "rows must be within 1..=16 (the M1 narrow route)"
    );
    let routes: Vec<Route> = args
        .routes
        .split(',')
        .map(Route::parse)
        .collect::<Result<_>>()?;

    let snapshot = mlx::memory::snapshot();
    let device = snapshot.device_name.unwrap_or_else(|| "unknown".into());
    let architecture = mlx::metal::architecture().unwrap_or_else(|_| "unknown".into());
    println!("# m1-projection-bench");
    println!("device: {device} ({architecture})");
    println!(
        "rounds={} passes={} warmup_passes={} working_set>={} MiB",
        args.rounds, args.passes, args.warmup_passes, args.working_set_mib
    );

    let mut csv = BufWriter::new(File::create(&args.out).context("creating --out")?);
    writeln!(
        csv,
        "device,architecture,shape,n,k,rows,route,copies,round,us_per_projection,gbps"
    )?;

    // Sustained warmup on the first shape so DVFS settles before timing.
    {
        let shape = &shapes[0];
        let copies = (args.working_set_mib * 1024 * 1024).div_ceil(weight_bytes(shape));
        let weights = build_weights(shape, copies, 1)?;
        let x = mlx::random::normal()
            .shape(&[1, 1, shape.k][..])
            .dtype(Dtype::Bfloat16)
            .key(&mlx::random::key(2)?)
            .sample()?;
        let deadline = Instant::now() + Duration::from_secs(args.warmup_secs);
        while Instant::now() < deadline {
            run_pass(Route::OffBatch, &x, &weights, 1)?;
        }
    }

    let peak = peak_read_gbps(args.peak_mib, 9)?;
    println!(
        "reference read bandwidth (sum over {} MiB): {peak:.1} GB/s",
        args.peak_mib
    );
    writeln!(
        csv,
        "{device},{architecture},peak_read,0,0,0,reference,1,0,0,{peak:.3}"
    )?;
    println!();
    println!("| shape | rows | route | µs/projection (median) | GB/s | % of reference | vs off_batch (paired median) |");
    println!("|---|---:|---|---:|---:|---:|---:|");

    for (shape_index, shape) in shapes.iter().enumerate() {
        let bytes = weight_bytes(shape);
        let copies = (args.working_set_mib * 1024 * 1024).div_ceil(bytes);
        let weights = build_weights(shape, copies, 1000 * (shape_index as u64 + 1))?;
        for &row_count in &rows {
            let active: Vec<Route> = routes
                .iter()
                .copied()
                .filter(|r| r.applies(row_count))
                .collect();
            let inputs: Vec<Array> = active
                .iter()
                .map(|route| {
                    let x = mlx::random::normal()
                        .shape(&[1, row_count, shape.k][..])
                        .dtype(Dtype::Bfloat16)
                        .key(&mlx::random::key(3)?)
                        .sample()?;
                    let x = x.reshape(&route.input_shape(row_count, shape.k)[..])?;
                    mlx::transforms::eval(&[&x])?;
                    Ok(x)
                })
                .collect::<Result<_>>()?;

            // Both M1 recipes must agree bit for bit (row-serial arithmetic).
            if let (Some(a), Some(b)) = (
                active.iter().position(|r| *r == Route::M1Nv2),
                active.iter().position(|r| *r == Route::M1Nv4),
            ) {
                let ya = active[a].forward(&inputs[a], &weights[0], row_count)?;
                let yb = active[b].forward(&inputs[b], &weights[0], row_count)?;
                ensure!(
                    bits(&ya)? == bits(&yb)?,
                    "m1_nv2 and m1_nv4 differ for {} rows={row_count}",
                    shape.name
                );
            }

            let mut per_round: Vec<Vec<f64>> = vec![Vec::new(); active.len()];
            for round in 0..args.rounds {
                for offset in 0..active.len() {
                    let index = (round + offset) % active.len();
                    let route = active[index];
                    for _ in 0..args.warmup_passes {
                        run_pass(route, &inputs[index], &weights, row_count)?;
                    }
                    let mut times = (0..args.passes)
                        .map(|_| run_pass(route, &inputs[index], &weights, row_count))
                        .collect::<Result<Vec<_>>>()?;
                    let pass = median(&mut times);
                    let us = pass / copies as f64 * 1e6;
                    let gbps = (bytes * copies) as f64 / pass / 1e9;
                    per_round[index].push(pass);
                    writeln!(
                        csv,
                        "{device},{architecture},{},{},{},{row_count},{},{copies},{round},{us:.3},{gbps:.3}",
                        shape.name,
                        shape.n,
                        shape.k,
                        route.name()
                    )?;
                }
            }
            csv.flush()?;

            let baseline = active.iter().position(|r| *r == Route::OffBatch);
            for (index, route) in active.iter().enumerate() {
                let mut passes = per_round[index].clone();
                let pass = median(&mut passes);
                let us = pass / copies as f64 * 1e6;
                let gbps = (bytes * copies) as f64 / pass / 1e9;
                let ratio = baseline
                    .map(|b| {
                        let mut ratios: Vec<f64> = per_round[b]
                            .iter()
                            .zip(&per_round[index])
                            .map(|(base, this)| base / this)
                            .collect();
                        format!("{:.3}×", median(&mut ratios))
                    })
                    .unwrap_or_else(|| "-".into());
                println!(
                    "| {} | {row_count} | {} | {us:.1} | {gbps:.1} | {:.0}% | {ratio} |",
                    shape.name,
                    route.name(),
                    gbps / peak * 100.0
                );
            }
        }
    }
    println!();
    println!("vs off_batch > 1.000× means faster than the off path for batched decode.");
    println!("Per-round CSV: {}", args.out.display());
    Ok(())
}
