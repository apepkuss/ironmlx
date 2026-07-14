//! Micro-benchmark for MLX affine quantized matmul.
//!
//! This binary keeps a small, direct qmm timing harness in the workspace.

use std::time::Instant;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use mlx::{ops, Array, Dtype};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Layout {
    /// Quantize logical [N, K] and call quantized_matmul(transpose=true).
    Transposed,
    /// Quantize logical [K, N] and call quantized_matmul(transpose=false).
    NonTransposed,
}

#[derive(Parser, Debug)]
#[command(about = "MLX affine quantized matmul micro-benchmark")]
struct Args {
    /// Matmul rows (typically prompt length x batch).
    #[arg(long, default_value_t = 2048)]
    m: i32,

    /// Matmul output cols (typically intermediate_size).
    #[arg(long, default_value_t = 9216)]
    n: i32,

    /// Matmul depth (typically hidden_size).
    #[arg(long, default_value_t = 2560)]
    k: i32,

    /// Quantization bits.
    #[arg(long, default_value_t = 4)]
    bits: i32,

    /// Quantization group size.
    #[arg(long, default_value_t = 64)]
    group_size: i32,

    /// Number of timed runs. Median is reported.
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Number of warmup runs excluded from stats.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Quantized weight layout.
    #[arg(long, value_enum, default_value_t = Layout::Transposed)]
    layout: Layout,
}

/// Build `(x, w_packed, w_scales, w_biases)` test inputs at the given shape.
fn build_inputs(
    m: i32,
    n: i32,
    k: i32,
    group_size: i32,
    bits: i32,
    layout: Layout,
) -> Result<(Array, Array, Array, Array)> {
    let x_count = (m as usize) * (k as usize);
    let x_data: Vec<f32> = (0..x_count).map(|i| ((i as f32) * 0.001) - 0.5).collect();
    let x_f32: Array = (x_data.as_slice(), (m, k)).try_into()?;
    let x = ops::cast::astype(&x_f32, Dtype::Bfloat16)?;

    let w_shape = match layout {
        Layout::Transposed => (n, k),
        Layout::NonTransposed => (k, n),
    };
    let w_count = (w_shape.0 as usize) * (w_shape.1 as usize);
    let w_data: Vec<f32> = (0..w_count).map(|i| ((i as f32) * 0.0005) - 0.3).collect();
    let raw_w_f32: Array = (w_data.as_slice(), w_shape).try_into()?;
    let raw_w_bf16 = ops::cast::astype(&raw_w_f32, Dtype::Bfloat16)?;

    let q_outs =
        mlx::quantization::quantize(&raw_w_bf16, Some(group_size), Some(bits), "affine", None)?;
    anyhow::ensure!(
        q_outs.len() == 3,
        "expected 3 outputs from affine quantize, got {}",
        q_outs.len()
    );
    let mut iter = q_outs.into_iter();
    let w_packed = iter.next().expect("checked output count");
    let w_scales = iter.next().expect("checked output count");
    let w_biases = iter.next().expect("checked output count");

    Ok((x, w_packed, w_scales, w_biases))
}

fn time_mlx_affine(args: &Args, inputs: &(Array, Array, Array, Array)) -> Result<f64> {
    let (x, w, s, b) = inputs;

    for _ in 0..args.warmup {
        let y = mlx::quantization::quantized_matmul_on(
            x,
            w,
            s,
            Some(b),
            args.layout == Layout::Transposed,
            Some(args.group_size),
            Some(args.bits),
            "affine",
            (),
        )?;
        mlx::transforms::eval(&[&y])?;
    }

    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let started = Instant::now();
        let y = mlx::quantization::quantized_matmul_on(
            x,
            w,
            s,
            Some(b),
            args.layout == Layout::Transposed,
            Some(args.group_size),
            Some(args.bits),
            "affine",
            (),
        )?;
        mlx::transforms::eval(&[&y])?;
        times.push(started.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.total_cmp(b));
    Ok(times[times.len() / 2])
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.runs > 0, "--runs must be greater than 0");

    println!("# ironmlx-bench-kernel");
    println!("M={}, N={}, K={}", args.m, args.n, args.k);
    println!("MLX layout: {:?}", args.layout);
    println!("Quant: bits={}, group_size={}", args.bits, args.group_size);
    println!(
        "Runs: {} measured (after {} warmup)",
        args.runs, args.warmup
    );
    println!();

    let inputs = build_inputs(
        args.m,
        args.n,
        args.k,
        args.group_size,
        args.bits,
        args.layout,
    )?;

    let elapsed = time_mlx_affine(&args, &inputs)?;
    let flops = 2.0 * (args.m as f64) * (args.n as f64) * (args.k as f64);
    let gflops = flops / elapsed / 1e9;
    println!(
        "mlx affine: median {:.3} ms, {:.1} GFLOP/s",
        elapsed * 1000.0,
        gflops
    );

    Ok(())
}
