//! Micro-benchmark for ironmlx self-quant matmul kernel.
//!
//! Measures wall-clock time of a single `qmm_t` dispatch with given
//! `(M, N, K, BM, BN, BK, bits, group_size)`. Used by stage 9 task 7 to
//! sweep tile candidates on M1 Pro and pick the best `(BM, BN, BK)`.
//!
//! Reaches into `ironmlx::nn::self_qmm::kernel::dispatch_qmm_t` directly
//! to avoid env-var/lookup overhead in `qmm_t_on` — the bench needs to
//! pin the tile, not let the lookup table choose one.

use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use mlx::{ops, Array, Dtype};

#[derive(Parser, Debug)]
#[command(about = "ironmlx self-quant matmul kernel micro-benchmark")]
struct Args {
    /// Matmul rows (typically prompt length × batch)
    #[arg(long, default_value_t = 2048)]
    m: i32,

    /// Matmul output cols (typically intermediate_size)
    #[arg(long, default_value_t = 9216)]
    n: i32,

    /// Matmul depth (typically hidden_size)
    #[arg(long, default_value_t = 2560)]
    k: i32,

    /// Tile BM
    #[arg(long, default_value_t = 64)]
    bm: i32,

    /// Tile BN
    #[arg(long, default_value_t = 128)]
    bn: i32,

    /// Tile BK
    #[arg(long, default_value_t = 32)]
    bk: i32,

    /// Quantization bits (only 4 supported in stage 9)
    #[arg(long, default_value_t = 4)]
    bits: i32,

    /// Quantization group size (only 64 supported in stage 9)
    #[arg(long, default_value_t = 64)]
    group_size: i32,

    /// Number of timed runs (median reported)
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Number of warmup runs (excluded from stats)
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Run mlx baseline (`quantized_matmul_on` affine) for comparison
    #[arg(long, default_value_t = false)]
    mlx_baseline: bool,
}

/// Build `(x, w_packed, w_scales, w_biases)` test inputs at the given shape.
///
/// Uses deterministic data (no random-seed flakiness across runs) within a
/// small numeric range, keeping matmul output values well within bf16
/// precision so the kernel doesn't NaN under uninitialized memory bugs.
fn build_inputs(
    m: i32,
    n: i32,
    k: i32,
    group_size: i32,
    bits: i32,
) -> Result<(Array, Array, Array, Array)> {
    // x bf16 [M, K] — small uniform range
    let x_count = (m as usize) * (k as usize);
    let x_data: Vec<f32> = (0..x_count).map(|i| ((i as f32) * 0.001) - 0.5).collect();
    let x_f32: Array = (x_data.as_slice(), (m, k)).try_into()?;
    let x = ops::cast::astype(&x_f32, Dtype::Bfloat16)?;

    // raw weights bf16 [N, K] — small uniform range
    let w_count = (n as usize) * (k as usize);
    let w_data: Vec<f32> = (0..w_count).map(|i| ((i as f32) * 0.0005) - 0.3).collect();
    let raw_w_f32: Array = (w_data.as_slice(), (n, k)).try_into()?;
    let raw_w_bf16 = ops::cast::astype(&raw_w_f32, Dtype::Bfloat16)?;

    // Quantize via mlx public API: returns [packed, scales, biases] for "affine"
    let q_outs =
        mlx::quantization::quantize(&raw_w_bf16, Some(group_size), Some(bits), "affine", None)?;
    anyhow::ensure!(
        q_outs.len() == 3,
        "expected 3 outputs from affine quantize, got {}",
        q_outs.len()
    );
    let mut iter = q_outs.into_iter();
    let w_packed = iter.next().unwrap();
    let w_scales = iter.next().unwrap();
    let w_biases = iter.next().unwrap();

    Ok((x, w_packed, w_scales, w_biases))
}

fn time_self_qmm(args: &Args, inputs: &(Array, Array, Array, Array)) -> Result<f64> {
    let (x, w, s, b) = inputs;

    // Warmup
    for _ in 0..args.warmup {
        let y =
            ironmlx::nn::self_qmm::kernel::dispatch_qmm_t(x, w, s, b, args.bm, args.bn, args.bk)?;
        mlx::transforms::eval(&[&y])?;
    }

    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let t0 = Instant::now();
        let y =
            ironmlx::nn::self_qmm::kernel::dispatch_qmm_t(x, w, s, b, args.bm, args.bn, args.bk)?;
        mlx::transforms::eval(&[&y])?;
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(times[times.len() / 2])
}

fn time_mlx_baseline(args: &Args, inputs: &(Array, Array, Array, Array)) -> Result<f64> {
    let (x, w, s, b) = inputs;

    for _ in 0..args.warmup {
        let y = mlx::quantization::quantized_matmul_on(
            x,
            w,
            s,
            Some(b),
            /* transpose = */ true,
            Some(args.group_size),
            Some(args.bits),
            "affine",
            (),
        )?;
        mlx::transforms::eval(&[&y])?;
    }

    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let t0 = Instant::now();
        let y = mlx::quantization::quantized_matmul_on(
            x,
            w,
            s,
            Some(b),
            /* transpose = */ true,
            Some(args.group_size),
            Some(args.bits),
            "affine",
            (),
        )?;
        mlx::transforms::eval(&[&y])?;
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(times[times.len() / 2])
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("# ironmlx-bench-kernel");
    println!("M={}, N={}, K={}", args.m, args.n, args.k);
    println!("Tile: BM={}, BN={}, BK={}", args.bm, args.bn, args.bk);
    println!("Quant: bits={}, group_size={}", args.bits, args.group_size);
    println!(
        "Runs: {} measured (after {} warmup)",
        args.runs, args.warmup
    );
    println!();

    let inputs = build_inputs(args.m, args.n, args.k, args.group_size, args.bits)?;

    let self_t = time_self_qmm(&args, &inputs)?;
    let flops = 2.0 * (args.m as f64) * (args.n as f64) * (args.k as f64);
    let self_gflops = flops / self_t / 1e9;
    println!(
        "self_qmm:    median {:.3} ms, {:.1} GFLOP/s",
        self_t * 1000.0,
        self_gflops
    );

    if args.mlx_baseline {
        let mlx_t = time_mlx_baseline(&args, &inputs)?;
        let mlx_gflops = flops / mlx_t / 1e9;
        let speedup = mlx_t / self_t;
        println!(
            "mlx affine:  median {:.3} ms, {:.1} GFLOP/s",
            mlx_t * 1000.0,
            mlx_gflops
        );
        println!("self_qmm vs mlx: {speedup:.2}x speedup");
    }

    Ok(())
}
