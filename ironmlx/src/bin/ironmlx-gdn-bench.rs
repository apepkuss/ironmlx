//! Direct GatedDeltaNet benchmark for Qwen3.6/Qwen3.5-MoE linear-attention layers.
//!
//! This bypasses model assembly, HTTP, and scheduler code so Phase 4 can compare
//! ironmlx's Rust GDN steady-state path against the MLX Python microbench.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::cache::GatedDeltaCache;
use ironmlx::core::Loader;
use ironmlx::models::Qwen35MoeConfig;
use ironmlx::nn::{AttnKind, GatedDeltaNet, GatedDeltaNetConfig};
use mlx::{random, Array, Dtype};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-gdn-bench",
    about = "Direct Qwen3.6/Qwen3.5-MoE GatedDeltaNet benchmark",
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

    /// Cache materialization mode.
    #[arg(long, value_enum, default_value_t = CacheMode::All)]
    cache_mode: CacheMode,

    /// PRNG seed for synthetic hidden states.
    #[arg(long, default_value_t = 20260528)]
    seed: u64,

    /// JSON output path.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CacheMode {
    All,
    NoCache,
    CacheOutOnly,
    CacheStateEval,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum BenchShape {
    NoCache,
    CacheOutOnly,
    CacheStateEval,
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
    load_ms: f64,
    hidden_size: i32,
    conv_dim: i32,
    num_k_heads: i32,
    num_v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    conv_kernel_size: i32,
}

#[derive(Serialize)]
struct Record {
    seq: i32,
    shape: BenchShape,
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
    let seqs = if args.seq.is_empty() {
        vec![521, 1]
    } else {
        args.seq.clone()
    };
    let load_started = Instant::now();
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
    let gdn = GatedDeltaNet::from_loader(&loader, &prefix, gdn_cfg)
        .with_context(|| format!("GatedDeltaNet::from_loader prefix={prefix}"))?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let shapes = shapes_for_mode(args.cache_mode);
    let mut records = Vec::new();
    for &seq in &seqs {
        if seq <= 0 {
            return Err(anyhow!("seq must be positive, got {seq}"));
        }
        let key = random::key(args.seed + seq as u64).context("random key")?;
        let x = random::normal()
            .shape((1_i32, seq, gdn_cfg.hidden_size))
            .dtype(Dtype::Bfloat16)
            .key(&key)
            .sample()
            .context("sample synthetic hidden states")?;
        mlx::transforms::eval(&[&x])?;
        for &shape in &shapes {
            records.push(run_shape(
                &gdn,
                gdn_cfg,
                &x,
                seq,
                shape,
                args.warmup_runs,
                args.runs,
            )?);
        }
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-gdn",
            model_dir: args.model.display().to_string(),
            layer: args.layer,
            seqs,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            load_ms,
            hidden_size: gdn_cfg.hidden_size,
            conv_dim: gdn_cfg.conv_dim(),
            num_k_heads: gdn_cfg.num_k_heads,
            num_v_heads: gdn_cfg.num_v_heads,
            head_k_dim: gdn_cfg.head_k_dim,
            head_v_dim: gdn_cfg.head_v_dim,
            conv_kernel_size: gdn_cfg.conv_kernel_size,
        },
        records,
    };
    std::fs::write(&args.out, serde_json::to_string_pretty(&output)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;
    Ok(())
}

fn shapes_for_mode(mode: CacheMode) -> Vec<BenchShape> {
    match mode {
        CacheMode::All => vec![
            BenchShape::NoCache,
            BenchShape::CacheOutOnly,
            BenchShape::CacheStateEval,
        ],
        CacheMode::NoCache => vec![BenchShape::NoCache],
        CacheMode::CacheOutOnly => vec![BenchShape::CacheOutOnly],
        CacheMode::CacheStateEval => vec![BenchShape::CacheStateEval],
    }
}

fn run_shape(
    gdn: &GatedDeltaNet,
    cfg: GatedDeltaNetConfig,
    x: &Array,
    seq: i32,
    shape: BenchShape,
    warmup_runs: usize,
    runs: usize,
) -> Result<Record> {
    let mut warmups = Vec::with_capacity(warmup_runs);
    for _ in 0..warmup_runs {
        warmups.push(run_once(gdn, cfg, x, seq, shape)?);
    }

    let mut values_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        values_ms.push(run_once(gdn, cfg, x, seq, shape)?);
    }

    Ok(Record {
        seq,
        shape,
        summary: summarize(&values_ms),
        warmups,
        values_ms,
    })
}

fn run_once(
    gdn: &GatedDeltaNet,
    cfg: GatedDeltaNetConfig,
    x: &Array,
    seq: i32,
    shape: BenchShape,
) -> Result<f64> {
    let mut cache = match shape {
        BenchShape::NoCache => None,
        BenchShape::CacheOutOnly | BenchShape::CacheStateEval => Some(make_cache(cfg, seq)?),
    };

    let started = Instant::now();
    let out = gdn.forward_on(x, None, None, cache.as_mut(), (), 0)?;
    match (shape, cache.as_ref()) {
        (BenchShape::CacheStateEval, Some(cache)) => {
            mlx::transforms::eval(&[&out, cache.conv_state(), cache.recurrent_state()])?;
        }
        _ => {
            mlx::transforms::eval(&[&out])?;
        }
    }
    mlx::transforms::synchronize()?;
    Ok(started.elapsed().as_secs_f64() * 1000.0)
}

fn make_cache(cfg: GatedDeltaNetConfig, seq: i32) -> Result<GatedDeltaCache> {
    GatedDeltaCache::new_with_cap(
        1,
        cfg.conv_kernel_size,
        cfg.conv_dim(),
        cfg.num_v_heads,
        cfg.head_v_dim,
        cfg.head_k_dim,
        Dtype::Bfloat16,
        seq.max(1) + 1,
    )
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
