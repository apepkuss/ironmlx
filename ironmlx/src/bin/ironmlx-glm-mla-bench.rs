//! GLM-4.7 MLA decode micro-benchmark.
//!
//! This binary isolates the real-checkpoint shapes used by the absorbed MLA
//! decode path. It keeps cache contents synthetic, but uses the real quantized
//! projection weights so the subpath timings can be compared with mlx-lm and
//! decode attribution runs.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::Loader;
use ironmlx::models::glm4_moe_lite::config::Glm4MoeLiteConfig;
use ironmlx::models::glm4_moe_lite::mla_attention::{MlaAttention, PerHeadQuantLinear};
use ironmlx::nn::Linear;
use mlx::ops::indexing::{slice_strided_on, slice_update_on};
use mlx::{random, Array, Device, Dtype, StreamOrDevice};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-glm-mla-bench",
    about = "Direct GLM-4.7 absorbed-MLA decode benchmark",
    version
)]
struct Args {
    /// Local GLM-4.7-Flash-4bit model directory.
    #[arg(long)]
    model: PathBuf,

    /// Decoder layer index to load.
    #[arg(long, default_value_t = 1)]
    layer: i32,

    /// Existing cache lengths to measure. Pass multiple times.
    #[arg(long = "ctx-len")]
    ctx_lens: Vec<i32>,

    /// Decode batch size to benchmark.
    #[arg(long, default_value_t = 1)]
    batch: i32,

    /// Timed runs per case.
    #[arg(long, default_value_t = 50)]
    runs: usize,

    /// Warmup runs per case.
    #[arg(long, default_value_t = 10)]
    warmup_runs: usize,

    /// PRNG seed for synthetic activations/cache contents.
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

struct SdpaMaskCase {
    case: &'static str,
    mask_mode: &'static str,
}

struct DecodeInputs {
    x: Array,
    offset: Array,
    base_c_kv: Array,
    base_k_pe: Array,
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
    ctx_lens: Vec<i32>,
    batch_size: i32,
    warmup_runs: usize,
    measured_runs: usize,
    stream_mode: &'static str,
    hidden_size: i32,
    num_attention_heads: i32,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    kv_lora_rank: i32,
    v_head_dim: i32,
    softmax_scale: f32,
}

#[derive(Serialize)]
struct Record {
    ctx_len: i32,
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
    let ctx_lens = if args.ctx_lens.is_empty() {
        vec![128, 512, 725, 2048]
    } else {
        args.ctx_lens.clone()
    };

    let loader = Loader::open(&args.model).context("Loader::open")?;
    let cfg = Glm4MoeLiteConfig::from_loader(&loader).context("loading GLM config")?;
    validate_layer(&cfg, args.layer)?;
    let prefix = format!("model.layers.{}.self_attn", args.layer);
    let attn = MlaAttention::from_loader(&loader, &prefix, &cfg)
        .with_context(|| format!("loading MlaAttention from {prefix}"))?;
    let embed_q = PerHeadQuantLinear::from_loader(&loader, &format!("{prefix}.embed_q"))?;
    let unembed_out = PerHeadQuantLinear::from_loader(&loader, &format!("{prefix}.unembed_out"))?;
    let o_proj = Linear::from_loader(&loader, &format!("{prefix}.o_proj"))?;

    let mut records = Vec::new();
    for &ctx_len in &ctx_lens {
        let inputs = build_decode_inputs(args.seed + ctx_len as u64, ctx_len, args.batch, &cfg)?;
        mlx::transforms::eval(&[
            &inputs.x,
            &inputs.offset,
            &inputs.base_c_kv,
            &inputs.base_k_pe,
        ])?;

        records.push(bench_case(
            ctx_len,
            "project-qkv",
            args.warmup_runs,
            args.runs,
            || {
                let out = attn.project_qkv(&inputs.x, &inputs.offset, bench_target.target)?;
                Ok(vec![out.0, out.1, out.2, out.3])
            },
        )?);

        let (q_nope, q_pe, c_kv_n, k_pe) =
            attn.project_qkv(&inputs.x, &inputs.offset, bench_target.target)?;
        mlx::transforms::eval(&[&q_nope, &q_pe, &c_kv_n, &k_pe])?;

        records.push(bench_case(
            ctx_len,
            "cache-update-fetch-local",
            args.warmup_runs,
            args.runs,
            || {
                let out = cache_update_fetch_local(
                    &inputs.base_c_kv,
                    &inputs.base_k_pe,
                    &c_kv_n,
                    &k_pe,
                    ctx_len,
                    &cfg,
                    bench_target.target,
                )?;
                Ok(vec![out.0, out.1])
            },
        )?);

        let (kv_latent, k_pe_all) = cache_update_fetch_local(
            &inputs.base_c_kv,
            &inputs.base_k_pe,
            &c_kv_n,
            &k_pe,
            ctx_len,
            &cfg,
            bench_target.target,
        )?;
        mlx::transforms::eval(&[&kv_latent, &k_pe_all])?;

        records.push(bench_case(
            ctx_len,
            "rope-scores",
            args.warmup_runs,
            args.runs,
            || {
                compute_rope_scores(&q_pe, &k_pe_all, &cfg, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);

        let pe_scores = compute_rope_scores(&q_pe, &k_pe_all, &cfg, bench_target.target)?;
        mlx::transforms::eval(&[&pe_scores])?;

        records.push(bench_case(
            ctx_len,
            "embed-q",
            args.warmup_runs,
            args.runs,
            || {
                embed_q
                    .apply(&q_nope, true, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);

        let q_lat = embed_q.apply(&q_nope, true, bench_target.target)?;
        mlx::transforms::eval(&[&q_lat])?;

        for sdpa_case in sdpa_mask_cases() {
            records.push(bench_case(
                ctx_len,
                sdpa_case.case,
                args.warmup_runs,
                args.runs,
                || {
                    run_decode_sdpa(
                        &q_lat,
                        &kv_latent,
                        &pe_scores,
                        &cfg,
                        sdpa_case.mask_mode,
                        bench_target.target,
                    )
                    .map(|out| vec![out])
                },
            )?);
        }

        let sdpa_out = run_decode_sdpa(
            &q_lat,
            &kv_latent,
            &pe_scores,
            &cfg,
            "",
            bench_target.target,
        )?;
        mlx::transforms::eval(&[&sdpa_out])?;

        records.push(bench_case(
            ctx_len,
            "unembed-out",
            args.warmup_runs,
            args.runs,
            || {
                unembed_out
                    .apply(&sdpa_out, true, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);

        let unembedded = unembed_out.apply(&sdpa_out, true, bench_target.target)?;
        mlx::transforms::eval(&[&unembedded])?;
        let merged = merge_heads(&unembedded, &cfg, bench_target.target)?;
        mlx::transforms::eval(&[&merged])?;

        records.push(bench_case(
            ctx_len,
            "attend-regime",
            args.warmup_runs,
            args.runs,
            || {
                attn.attend_regime(
                    &q_nope,
                    &kv_latent,
                    &pe_scores,
                    true,
                    bench_target.target,
                    args.layer,
                )
                .map(|out| vec![out])
            },
        )?);

        records.push(bench_case(
            ctx_len,
            "o-proj",
            args.warmup_runs,
            args.runs,
            || {
                o_proj
                    .forward_on(&merged, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);

        records.push(bench_case(
            ctx_len,
            "local-full-decode",
            args.warmup_runs,
            args.runs,
            || {
                local_full_decode(
                    &attn,
                    &o_proj,
                    &inputs,
                    ctx_len,
                    &cfg,
                    bench_target.target,
                    args.layer,
                )
                .map(|out| vec![out])
            },
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-glm-mla",
            model_dir: args.model.display().to_string(),
            layer: args.layer,
            ctx_lens,
            batch_size: args.batch,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            stream_mode: bench_target.label,
            hidden_size: cfg.hidden_size,
            num_attention_heads: cfg.num_attention_heads,
            qk_nope_head_dim: cfg.qk_nope_head_dim,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
            kv_lora_rank: cfg.kv_lora_rank,
            v_head_dim: cfg.v_head_dim,
            softmax_scale: cfg.softmax_scale(),
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
    if args.layer < 0 {
        return Err(anyhow!("--layer must be non-negative, got {}", args.layer));
    }
    validate_ctx_lens(&args.ctx_lens)?;
    validate_batch(args.batch)?;
    if args.runs == 0 {
        return Err(anyhow!("--runs must be positive"));
    }
    Ok(())
}

fn validate_ctx_lens(ctx_lens: &[i32]) -> Result<()> {
    if let Some(&ctx_len) = ctx_lens.iter().find(|&&ctx_len| ctx_len < 0) {
        return Err(anyhow!(
            "--ctx-len values must be non-negative, got {ctx_len}"
        ));
    }
    Ok(())
}

fn validate_batch(batch: i32) -> Result<()> {
    if batch <= 0 {
        return Err(anyhow!("--batch must be positive, got {batch}"));
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
    Ok(())
}

fn build_decode_inputs(
    seed: u64,
    ctx_len: i32,
    batch: i32,
    cfg: &Glm4MoeLiteConfig,
) -> Result<DecodeInputs> {
    validate_batch(batch)?;
    let x = random_bf16_3d(seed, batch, 1, cfg.hidden_size).context("sampling x")?;
    let cache_len = ctx_len + 1;
    let base_c_kv =
        random_bf16_4d(seed + 1, batch, 1, cache_len, cfg.kv_lora_rank).context("sampling c_kv")?;
    let base_k_pe = random_bf16_4d(seed + 2, batch, 1, cache_len, cfg.qk_rope_head_dim)
        .context("sampling k_pe")?;
    let offsets = vec![ctx_len; batch as usize];
    let offset: Array = (&offsets[..], &[batch][..]).try_into()?;
    Ok(DecodeInputs {
        x,
        offset,
        base_c_kv,
        base_k_pe,
    })
}

fn random_bf16_3d(seed: u64, d0: i32, d1: i32, d2: i32) -> Result<Array> {
    let key = random::key(seed).context("random key")?;
    random::normal()
        .shape((d0, d1, d2))
        .dtype(Dtype::Bfloat16)
        .key(&key)
        .sample()
        .map_err(anyhow::Error::from)
}

fn random_bf16_4d(seed: u64, d0: i32, d1: i32, d2: i32, d3: i32) -> Result<Array> {
    let key = random::key(seed).context("random key")?;
    random::normal()
        .shape((d0, d1, d2, d3))
        .dtype(Dtype::Bfloat16)
        .key(&key)
        .sample()
        .map_err(anyhow::Error::from)
}

fn cache_update_fetch_local(
    base_c_kv: &Array,
    base_k_pe: &Array,
    c_kv_new: &Array,
    k_pe_new: &Array,
    ctx_len: i32,
    cfg: &Glm4MoeLiteConfig,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    let base_shape = base_c_kv.shape();
    let batch = base_shape.as_slice()[0];
    let end = ctx_len + 1;
    let c_kv_updated = slice_update_on(
        base_c_kv,
        c_kv_new,
        [0_i32, 0, ctx_len, 0],
        [batch, 1, end, cfg.kv_lora_rank],
        [1_i32, 1, 1, 1],
        target,
    )?;
    let k_pe_updated = slice_update_on(
        base_k_pe,
        k_pe_new,
        [0_i32, 0, ctx_len, 0],
        [batch, 1, end, cfg.qk_rope_head_dim],
        [1_i32, 1, 1, 1],
        target,
    )?;
    let c_kv_slice = slice_strided_on(
        &c_kv_updated,
        [0_i32, 0, 0, 0],
        [batch, 1, end, cfg.kv_lora_rank],
        [1_i32, 1, 1, 1],
        target,
    )?;
    let k_pe_slice = slice_strided_on(
        &k_pe_updated,
        [0_i32, 0, 0, 0],
        [batch, 1, end, cfg.qk_rope_head_dim],
        [1_i32, 1, 1, 1],
        target,
    )?;
    Ok((c_kv_slice, k_pe_slice))
}

fn compute_rope_scores(
    q_pe: &Array,
    k_pe_all: &Array,
    cfg: &Glm4MoeLiteConfig,
    target: StreamOrDevice,
) -> Result<Array> {
    let scale: Array = (&[cfg.softmax_scale()][..], ()).try_into()?;
    let q_pe_scaled = mlx::ops::binary::multiply_on(q_pe, &scale, target)?;
    let k_pe_t = k_pe_all.transpose_axes_on(&[0, 1, 3, 2][..], target)?;
    q_pe_scaled
        .matmul_on(&k_pe_t, target)
        .map_err(anyhow::Error::from)
}

fn run_decode_sdpa(
    q_lat: &Array,
    kv_latent: &Array,
    pe_scores: &Array,
    cfg: &Glm4MoeLiteConfig,
    mask_mode: &str,
    target: StreamOrDevice,
) -> Result<Array> {
    let mask = if pe_scores.dtype() == kv_latent.dtype() {
        pe_scores.clone()
    } else {
        mlx::ops::cast::astype_on(pe_scores, kv_latent.dtype(), target)?
    };
    Ok(mlx::fast::scaled_dot_product_attention_on(
        q_lat,
        kv_latent,
        kv_latent,
        cfg.softmax_scale(),
        mask_mode,
        Some(&mask),
        None,
        target,
    )?)
}

fn sdpa_mask_cases() -> [SdpaMaskCase; 2] {
    [
        SdpaMaskCase {
            case: "sdpa",
            mask_mode: "",
        },
        SdpaMaskCase {
            case: "sdpa-mask-array",
            mask_mode: "array",
        },
    ]
}

fn merge_heads(out: &Array, cfg: &Glm4MoeLiteConfig, target: StreamOrDevice) -> Result<Array> {
    let shape = out.shape();
    let dims = shape.as_slice();
    let batch = dims[0];
    let seq_len = dims[2];
    out.transpose_axes_on(&[0, 2, 1, 3][..], target)?
        .reshape_on(
            (batch, seq_len, cfg.num_attention_heads * cfg.v_head_dim),
            target,
        )
        .map_err(anyhow::Error::from)
}

fn local_full_decode(
    attn: &MlaAttention,
    o_proj: &Linear,
    inputs: &DecodeInputs,
    ctx_len: i32,
    cfg: &Glm4MoeLiteConfig,
    target: StreamOrDevice,
    layer: i32,
) -> Result<Array> {
    let (q_nope, q_pe, c_kv_n, k_pe) = attn.project_qkv(&inputs.x, &inputs.offset, target)?;
    let (kv_latent, k_pe_all) = cache_update_fetch_local(
        &inputs.base_c_kv,
        &inputs.base_k_pe,
        &c_kv_n,
        &k_pe,
        ctx_len,
        cfg,
        target,
    )?;
    let pe_scores = compute_rope_scores(&q_pe, &k_pe_all, cfg, target)?;
    let out = attn.attend_regime(&q_nope, &kv_latent, &pe_scores, true, target, layer)?;
    let out = merge_heads(&out, cfg, target)?;
    o_proj.forward_on(&out, target)
}

fn bench_case<F>(
    ctx_len: i32,
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
        ctx_len,
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
    println!("# ironmlx-glm-mla-bench");
    println!(
        "layer={} H={} heads={} kv_lora={} rope={} v_head={} B={} stream={}",
        output.meta.layer,
        output.meta.hidden_size,
        output.meta.num_attention_heads,
        output.meta.kv_lora_rank,
        output.meta.qk_rope_head_dim,
        output.meta.v_head_dim,
        output.meta.batch_size,
        output.meta.stream_mode
    );
    for record in &output.records {
        println!(
            "ctx={:<5} case={:<24} p50={:>8.4} ms p95={:>8.4} ms",
            record.ctx_len,
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
    fn validate_ctx_lens_rejects_negative_values() {
        assert!(validate_ctx_lens(&[-1]).is_err());
    }

    #[test]
    fn validate_ctx_lens_accepts_zero_and_positive_values() {
        validate_ctx_lens(&[0, 1, 2048]).unwrap();
    }

    #[test]
    fn validate_batch_rejects_non_positive_values() {
        assert!(validate_batch(0).is_err());
        assert!(validate_batch(-1).is_err());
    }

    #[test]
    fn validate_batch_accepts_positive_values() {
        validate_batch(1).unwrap();
        validate_batch(4).unwrap();
    }

    #[test]
    fn sdpa_mask_cases_cover_production_and_array_diagnostic() {
        let cases = sdpa_mask_cases();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].case, "sdpa");
        assert_eq!(cases[0].mask_mode, "");
        assert_eq!(cases[1].case, "sdpa-mask-array");
        assert_eq!(cases[1].mask_mode, "array");
    }

    #[test]
    fn percentile_interpolates() {
        assert_eq!(percentile(&[1.0, 3.0], 50.0), Some(2.0));
    }
}
