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
use ironmlx::models::{ModelArchitecture, Qwen35Config, Qwen35MoeConfig};
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

    /// Simulated cache offset. Values above zero use non-zero steady-state cache tensors.
    #[arg(long, default_value_t = 0)]
    cache_offset: i32,

    /// Execution routes to measure. Pass multiple times; defaults to both.
    #[arg(long, value_enum)]
    route: Vec<GdnRoute>,

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

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum GdnRoute {
    Regular,
    PositionStable,
    SequenceStable,
    ExactVerify,
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
    model_type: &'static str,
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
    route: GdnRoute,
    shape: BenchShape,
    exactness: Exactness,
    summary: Summary,
    warmups: Vec<f64>,
    values_ms: Vec<f64>,
}

#[derive(Serialize)]
struct Exactness {
    output_exact: bool,
    output_max_abs_diff: f32,
    conv_state_exact: Option<bool>,
    conv_state_max_abs_diff: Option<f32>,
    recurrent_state_exact: Option<bool>,
    recurrent_state_max_abs_diff: Option<f32>,
}

#[derive(Serialize)]
struct Summary {
    runs: usize,
    p50_ms: Option<f64>,
    p95_ms: Option<f64>,
    mean_ms: Option<f64>,
}

fn main() -> Result<()> {
    init_tracing();

    let args = Args::parse();
    let seqs = if args.seq.is_empty() {
        vec![521, 1]
    } else {
        args.seq.clone()
    };
    let load_started = Instant::now();
    let loader = Loader::open(&args.model).context("Loader::open")?;
    let arch = ModelArchitecture::from_config_value(loader.config_raw_value())
        .context("ModelArchitecture::from_config_value")?;
    let cfg = load_gdn_bench_config(&loader, arch)?;
    if !matches!(cfg.layer_kind(args.layer), AttnKind::Linear) {
        return Err(anyhow!(
            "layer {} is not a linear-attention GatedDeltaNet layer",
            args.layer
        ));
    }

    let gdn_cfg = cfg.gdn_config();
    let prefix = format!("model.layers.{}.linear_attn", args.layer);
    let gdn = GatedDeltaNet::from_loader(&loader, &prefix, gdn_cfg)
        .with_context(|| format!("GatedDeltaNet::from_loader prefix={prefix}"))?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let shapes = shapes_for_mode(args.cache_mode);
    let routes = if args.route.is_empty() {
        vec![
            GdnRoute::Regular,
            GdnRoute::PositionStable,
            GdnRoute::SequenceStable,
            GdnRoute::ExactVerify,
        ]
    } else {
        args.route.clone()
    };
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
        for &route in &routes {
            for &shape in &shapes {
                records.push(run_shape(
                    &gdn,
                    gdn_cfg,
                    &x,
                    RunShapeConfig {
                        seq,
                        route,
                        shape,
                        cache_offset: args.cache_offset,
                        warmup_runs: args.warmup_runs,
                        runs: args.runs,
                    },
                )?);
            }
        }
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-gdn",
            model_dir: args.model.display().to_string(),
            model_type: arch.model_type(),
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

#[derive(Clone, Copy)]
struct GdnBenchConfig {
    hidden_size: i32,
    rms_norm_eps: f32,
    full_attention_interval: i32,
    linear_num_value_heads: i32,
    linear_num_key_heads: i32,
    linear_key_head_dim: i32,
    linear_value_head_dim: i32,
    linear_conv_kernel_dim: i32,
}

impl GdnBenchConfig {
    fn layer_kind(self, layer_idx: i32) -> AttnKind {
        if (layer_idx + 1) % self.full_attention_interval == 0 {
            AttnKind::Full
        } else {
            AttnKind::Linear
        }
    }

    fn gdn_config(self) -> GatedDeltaNetConfig {
        GatedDeltaNetConfig {
            hidden_size: self.hidden_size,
            num_v_heads: self.linear_num_value_heads,
            num_k_heads: self.linear_num_key_heads,
            head_k_dim: self.linear_key_head_dim,
            head_v_dim: self.linear_value_head_dim,
            conv_kernel_size: self.linear_conv_kernel_dim,
            rms_norm_eps: self.rms_norm_eps,
        }
    }
}

fn load_gdn_bench_config(loader: &Loader, arch: ModelArchitecture) -> Result<GdnBenchConfig> {
    match arch {
        ModelArchitecture::Qwen35Dense => {
            let cfg = Qwen35Config::from_loader(loader).context("Qwen35Config::from_loader")?;
            Ok(GdnBenchConfig {
                hidden_size: cfg.hidden_size,
                rms_norm_eps: cfg.rms_norm_eps,
                full_attention_interval: cfg.full_attention_interval,
                linear_num_value_heads: cfg.linear_num_value_heads,
                linear_num_key_heads: cfg.linear_num_key_heads,
                linear_key_head_dim: cfg.linear_key_head_dim,
                linear_value_head_dim: cfg.linear_value_head_dim,
                linear_conv_kernel_dim: cfg.linear_conv_kernel_dim,
            })
        }
        ModelArchitecture::Qwen35Moe => {
            let cfg =
                Qwen35MoeConfig::from_loader(loader).context("Qwen35MoeConfig::from_loader")?;
            Ok(GdnBenchConfig {
                hidden_size: cfg.hidden_size,
                rms_norm_eps: cfg.rms_norm_eps,
                full_attention_interval: cfg.full_attention_interval,
                linear_num_value_heads: cfg.linear_num_value_heads,
                linear_num_key_heads: cfg.linear_num_key_heads,
                linear_key_head_dim: cfg.linear_key_head_dim,
                linear_value_head_dim: cfg.linear_value_head_dim,
                linear_conv_kernel_dim: cfg.linear_conv_kernel_dim,
            })
        }
        other => Err(anyhow!(
            "ironmlx-gdn-bench supports Qwen3.5 dense/MoE only; got {}",
            other.model_type()
        )),
    }
}

fn init_tracing() {}

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

#[derive(Clone, Copy)]
struct RunShapeConfig {
    seq: i32,
    route: GdnRoute,
    shape: BenchShape,
    cache_offset: i32,
    warmup_runs: usize,
    runs: usize,
}

fn run_shape(
    gdn: &GatedDeltaNet,
    cfg: GatedDeltaNetConfig,
    x: &Array,
    run: RunShapeConfig,
) -> Result<Record> {
    let mut warmups = Vec::with_capacity(run.warmup_runs);
    for _ in 0..run.warmup_runs {
        warmups.push(run_once(
            gdn,
            cfg,
            x,
            run.seq,
            run.route,
            run.shape,
            run.cache_offset,
        )?);
    }

    let mut values_ms = Vec::with_capacity(run.runs);
    for _ in 0..run.runs {
        values_ms.push(run_once(
            gdn,
            cfg,
            x,
            run.seq,
            run.route,
            run.shape,
            run.cache_offset,
        )?);
    }

    Ok(Record {
        seq: run.seq,
        route: run.route,
        shape: run.shape,
        exactness: qualify_route(gdn, cfg, x, run.seq, run.route, run.shape, run.cache_offset)?,
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
    route: GdnRoute,
    shape: BenchShape,
    cache_offset: i32,
) -> Result<f64> {
    let started = Instant::now();
    let (out, cache) = forward_route(gdn, cfg, x, seq, route, shape, cache_offset)?;
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

fn forward_route(
    gdn: &GatedDeltaNet,
    cfg: GatedDeltaNetConfig,
    x: &Array,
    seq: i32,
    route: GdnRoute,
    shape: BenchShape,
    cache_offset: i32,
) -> Result<(Array, Option<GatedDeltaCache>)> {
    let mut cache = match shape {
        BenchShape::NoCache => None,
        BenchShape::CacheOutOnly | BenchShape::CacheStateEval => {
            Some(make_cache(cfg, seq, cache_offset)?)
        }
    };
    let position_stable = matches!(route, GdnRoute::PositionStable | GdnRoute::ExactVerify);
    let sequence_stable = matches!(route, GdnRoute::SequenceStable | GdnRoute::ExactVerify);
    let _position_stable = position_stable.then(ironmlx::nn::position_stable_qmm_scope);
    let _sequence_stable = sequence_stable.then(ironmlx::nn::sequence_stable_gated_delta_scope);
    let out = gdn.forward_on(x, None, None, cache.as_mut(), (), 0)?;
    Ok((out, cache))
}

fn qualify_route(
    gdn: &GatedDeltaNet,
    cfg: GatedDeltaNetConfig,
    x: &Array,
    seq: i32,
    route: GdnRoute,
    shape: BenchShape,
    cache_offset: i32,
) -> Result<Exactness> {
    let (actual_out, actual_cache) = forward_route(gdn, cfg, x, seq, route, shape, cache_offset)?;
    let (expected_out, expected_cache) =
        forward_sequential_q1(gdn, cfg, x, seq, shape, cache_offset)?;

    let actual_out = materialize_f32(&actual_out)?;
    let expected_out = materialize_f32(&expected_out)?;
    let output_max_abs_diff = max_abs_diff(&actual_out, &expected_out)?;

    let (
        conv_state_exact,
        conv_state_max_abs_diff,
        recurrent_state_exact,
        recurrent_state_max_abs_diff,
    ) = match (actual_cache.as_ref(), expected_cache.as_ref()) {
        (Some(actual), Some(expected)) => {
            let actual_conv = materialize_f32(actual.conv_state())?;
            let expected_conv = materialize_f32(expected.conv_state())?;
            let conv_diff = max_abs_diff(&actual_conv, &expected_conv)?;
            let actual_recurrent = materialize_f32(actual.recurrent_state())?;
            let expected_recurrent = materialize_f32(expected.recurrent_state())?;
            let recurrent_diff = max_abs_diff(&actual_recurrent, &expected_recurrent)?;
            (
                Some(actual_conv == expected_conv),
                Some(conv_diff),
                Some(actual_recurrent == expected_recurrent),
                Some(recurrent_diff),
            )
        }
        (None, None) => (None, None, None, None),
        _ => return Err(anyhow!("candidate/reference cache presence mismatch")),
    };

    Ok(Exactness {
        output_exact: actual_out == expected_out,
        output_max_abs_diff,
        conv_state_exact,
        conv_state_max_abs_diff,
        recurrent_state_exact,
        recurrent_state_max_abs_diff,
    })
}

fn forward_sequential_q1(
    gdn: &GatedDeltaNet,
    cfg: GatedDeltaNetConfig,
    x: &Array,
    seq: i32,
    shape: BenchShape,
    cache_offset: i32,
) -> Result<(Array, Option<GatedDeltaCache>)> {
    let mut cache = match shape {
        BenchShape::NoCache => None,
        BenchShape::CacheOutOnly | BenchShape::CacheStateEval => {
            Some(make_cache(cfg, seq, cache_offset)?)
        }
    };
    let hidden = x.shape().as_slice()[2];
    let mut outputs = Vec::with_capacity(seq as usize);
    for position in 0..seq {
        let step = mlx::ops::indexing::slice_strided(
            x,
            &[0_i32, position, 0][..],
            &[1_i32, position + 1, hidden][..],
            &[1_i32, 1, 1][..],
        )?;
        outputs.push(gdn.forward_on(&step, None, None, cache.as_mut(), (), 0)?);
    }
    let refs = outputs.iter().collect::<Vec<_>>();
    let out = mlx::ops::shape::concatenate(&refs, 1)?;
    Ok((out, cache))
}

fn materialize_f32(value: &Array) -> Result<Vec<f32>> {
    let value = mlx::ops::cast::astype(value, Dtype::Float32)?;
    value.to_vec::<f32>().map_err(Into::into)
}

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err(anyhow!(
            "candidate/reference element count mismatch: {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max))
}

fn make_cache(cfg: GatedDeltaNetConfig, seq: i32, cache_offset: i32) -> Result<GatedDeltaCache> {
    if cache_offset < 0 {
        return Err(anyhow!("cache-offset must be non-negative"));
    }
    let mut cache = GatedDeltaCache::new_with_cap(
        1,
        cfg.conv_kernel_size,
        cfg.conv_dim(),
        cfg.num_v_heads,
        cfg.head_v_dim,
        cfg.head_k_dim,
        Dtype::Bfloat16,
        cache_offset + seq.max(1) + 1,
    )?;
    if cache_offset > 0 {
        let conv_key = random::key(7_001 + cache_offset as u64)?;
        let recurrent_key = random::key(7_002 + cache_offset as u64)?;
        cache.update_conv(
            random::normal()
                .shape((1, cfg.conv_kernel_size - 1, cfg.conv_dim()))
                .dtype(Dtype::Bfloat16)
                .key(&conv_key)
                .sample()?,
        );
        cache.update_recurrent(
            random::normal()
                .shape((1, cfg.num_v_heads, cfg.head_v_dim, cfg.head_k_dim))
                .dtype(Dtype::Float32)
                .key(&recurrent_key)
                .sample()?,
        );
        cache.advance(&[cache_offset])?;
    }
    Ok(cache)
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
