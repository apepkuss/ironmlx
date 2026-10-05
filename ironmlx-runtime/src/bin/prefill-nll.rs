//! Teacher-forced next-token NLL over long texts through the DFlash2 target
//! prefill route (2048-row chunks, M5 lane, experimental flags from the
//! environment). Quality instrument for numeric-changing prefill candidates:
//! compares per-token NLL between process runs with different flags.

use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use ironmlx_lm::core::loader::Loader;
use ironmlx_lm::core::tokenizer::Tokenizer;
use ironmlx_lm::core::Model;
use ironmlx_lm::models::dflash2::{DFlash2Target, DFlash2TargetForwardMode};
use ironmlx_lm::models::Qwen35Model;
use mlx::{Array, Dtype, StreamOrDevice};
use serde_json::json;

#[derive(Parser)]
struct Args {
    /// Target model snapshot directory.
    #[arg(long)]
    target: PathBuf,
    /// UTF-8 text files to score (each truncated to --tokens).
    #[arg(long, required = true)]
    text: Vec<PathBuf>,
    #[arg(long, default_value_t = 32768)]
    tokens: usize,
    #[arg(long, default_value_t = 2048)]
    chunk: usize,
    /// Rows projected per lm_head call when scoring a chunk.
    #[arg(long, default_value_t = 256)]
    rows: i32,
    /// Output JSON (must not exist).
    #[arg(long)]
    output: PathBuf,
}

fn positions(start: i32, len: i32) -> Result<Array> {
    let p = mlx::ops::arange(start as f64, (start + len) as f64, 1.0, Dtype::Int32)?;
    let p = mlx::ops::reshape(&p, &[1, 1, len][..])?;
    Ok(mlx::ops::broadcast_to(&p, &[3, 1, len][..])?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "preserve an existing result");
    let mlx_dir = std::env::var("MLX_DIR").context("MLX_DIR")?;
    mlx::metal::set_metallib_path(&format!("{mlx_dir}/lib/mlx.metallib"))?;
    mlx::set_default_device(mlx::Device::gpu(0));
    let mut loader = Loader::open(&args.target)?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let model = Qwen35Model::from_loader_dflash2(&mut loader)?;
    drop(loader);
    let target = StreamOrDevice::default();
    let mut results = Vec::new();
    for path in &args.text {
        let text = std::fs::read_to_string(path)?;
        let mut ids = tokenizer.encode(&text, false)?;
        ensure!(
            ids.len() >= args.tokens,
            "{} has only {} tokens",
            path.display(),
            ids.len()
        );
        ids.truncate(args.tokens);
        let mut cache = model.make_cache(1, (args.tokens + 64) as i32, model.cache_dtype())?;
        let mut nll = Vec::with_capacity(args.tokens - 1);
        let mut position = 0usize;
        let started = std::time::Instant::now();
        while position < ids.len() {
            let len = args.chunk.min(ids.len() - position);
            let input: Array =
                (&ids[position..position + len], &[1_i32, len as i32][..]).try_into()?;
            let out = model.dflash2_forward_target_on(
                &input,
                &positions(position as i32, len as i32)?,
                Some(&mut cache),
                // Same capture layers as the serving DFlash2 draft config.
                &[5, 19, 33, 47, 61],
                DFlash2TargetForwardMode::Prefill,
                target,
            )?;
            mlx::transforms::eval(&[&out.hidden])?;
            // Score rows whose next token lies inside the text.
            let scored = len.min(ids.len() - 1 - position);
            let mut row = 0usize;
            while row < scored {
                let n = (args.rows as usize).min(scored - row);
                let h = mlx::ops::slice(
                    &out.hidden,
                    &[0, row as i32, 0][..],
                    &[1, (row + n) as i32, out.hidden.shape().as_slice()[2]][..],
                )?;
                let logits = mlx::ops::astype(
                    &model.dflash2_project_hidden_on(&h, target)?,
                    Dtype::Float32,
                )?;
                let lse = mlx::ops::logsumexp(&logits, &[-1][..], true)?;
                let next: Vec<u32> = ids[position + row + 1..position + row + 1 + n].to_vec();
                let next: Array = (&next[..], &[1_i32, n as i32, 1][..]).try_into()?;
                let next = mlx::ops::astype(&next, Dtype::Int32)?;
                let picked = mlx::ops::take_along_axis(&logits, &next, -1)?;
                let values = (&lse - &picked).to_vec::<f32>()?;
                nll.extend(values);
                row += n;
            }
            position += len;
        }
        let mean = nll.iter().map(|v| *v as f64).sum::<f64>() / nll.len() as f64;
        println!(
            "nll {} tokens={} mean={mean:.6} secs={:.1}",
            path.display(),
            nll.len(),
            started.elapsed().as_secs_f64()
        );
        results.push(
            json!({"path": path, "tokens": ids.len(), "scored": nll.len(),
                            "mean_nll": mean, "nll": nll}),
        );
    }
    std::fs::write(
        &args.output,
        serde_json::to_vec(&json!({"results": results,
        "chunk": args.chunk, "tokens": args.tokens}))?,
    )?;
    Ok(())
}
