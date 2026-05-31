//! GLM-4.7 full-model decode forward micro-benchmark.
//!
//! This benchmark moves above isolated MLA/MoE/decoder-layer probes and measures
//! the full 47-layer single-token decode forward path with a real checkpoint.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::{Loader, Sampler};
use ironmlx::models::glm4_moe_lite::config::Glm4MoeLiteConfig;
use ironmlx::models::glm4_moe_lite::decoder_layer::{DecoderBlockMode, Glm4DecoderLayer};
use ironmlx::models::glm4_moe_lite::mla_cache::MlaLatentCache;
use ironmlx::nn::{Embedding, LayerCache, Linear, RmsNorm};
use mlx::compile::CompileMode as MlxCompileMode;
use mlx::{Array, Device, Dtype, StreamOrDevice};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-glm-full-forward-bench",
    about = "Direct GLM-4.7 full-model decode forward benchmark",
    version
)]
struct Args {
    /// Local GLM-4.7-Flash-4bit model directory.
    #[arg(long)]
    model: PathBuf,

    /// Existing cache lengths to prefill before decode. Pass multiple times.
    #[arg(long = "ctx-len")]
    ctx_lens: Vec<i32>,

    /// Number of leading decoder layers to execute. Pass multiple times.
    #[arg(long = "depth")]
    depths: Vec<i32>,

    /// Timed decode runs per case. Each run advances the cache by one token.
    #[arg(long, default_value_t = 50)]
    runs: usize,

    /// Warmup decode runs per case. Each warmup advances the cache by one token.
    #[arg(long, default_value_t = 10)]
    warmup_runs: usize,

    /// PRNG seed for synthetic token ids.
    #[arg(long, default_value_t = 20260531)]
    seed: u64,

    /// JSON output path.
    #[arg(long)]
    out: PathBuf,

    /// Stream target mode for diagnostics.
    #[arg(long, value_enum, default_value_t = StreamMode::Default)]
    stream_mode: StreamMode,

    /// Optional materialization after every decoder layer for graph-boundary diagnostics.
    #[arg(long, value_enum, default_value_t = LayerEvalMode::None)]
    layer_eval_mode: LayerEvalMode,

    /// Diagnostic sub-block execution mode.
    #[arg(long, value_enum, default_value_t = BlockMode::Full)]
    block_mode: BlockMode,

    /// Global MLX compile mode for diagnostic attribution.
    #[arg(long, value_enum, default_value_t = CompileMode::Enabled)]
    compile_mode: CompileMode,
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LayerEvalMode {
    /// Preserve the default full-model lazy graph.
    None,
    /// Call mlx::eval on the hidden state after every decoder layer.
    Eval,
    /// Call mlx::eval and synchronize after every decoder layer.
    EvalSync,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BlockMode {
    /// Run the full decoder block.
    Full,
    /// Skip attention and run only post-attention norm plus FFN residual.
    SkipAttn,
    /// Skip attention and run only the routed MoE branch in MoE layers.
    SkipAttnRouted,
    /// Skip attention and run routed experts with fixed synthetic routing.
    SkipAttnRoutedFixed,
    /// Skip attention and run only the shared expert branch in MoE layers.
    SkipAttnShared,
    /// Run attention residual and skip post-attention norm plus FFN.
    SkipFfn,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CompileMode {
    /// Full MLX compile mode.
    Enabled,
    /// Disable MLX compile and execute compiled closures eagerly.
    Disabled,
    /// MLX compile without simplify pass.
    NoSimplify,
    /// MLX compile without fusion pass.
    NoFuse,
}

#[derive(Clone, Copy)]
struct BenchTarget {
    label: &'static str,
    target: StreamOrDevice,
}

#[derive(Clone, Copy)]
struct ForwardOptions {
    target: StreamOrDevice,
    layer_eval_mode: LayerEvalMode,
    block_mode: DecoderBlockMode,
    depth: i32,
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
    ctx_lens: Vec<i32>,
    depths: Vec<i32>,
    warmup_runs: usize,
    measured_runs: usize,
    stream_mode: &'static str,
    layer_eval_mode: &'static str,
    block_mode: &'static str,
    compile_mode: &'static str,
    hidden_size: i32,
    vocab_size: i32,
    num_hidden_layers: i32,
    dtype: &'static str,
    cache_prealloc: &'static str,
    token_source: &'static str,
}

#[derive(Serialize)]
struct Record {
    ctx_len: i32,
    depth: i32,
    case: &'static str,
    output_shapes: Vec<Vec<i32>>,
    summary: Summary,
    build_summary: Summary,
    eval_sync_summary: Summary,
    warmups_ms: Vec<f64>,
    build_warmups_ms: Vec<f64>,
    eval_sync_warmups_ms: Vec<f64>,
    values_ms: Vec<f64>,
    build_values_ms: Vec<f64>,
    eval_sync_values_ms: Vec<f64>,
}

#[derive(Serialize)]
struct Summary {
    runs: usize,
    p50_ms: Option<f64>,
    p95_ms: Option<f64>,
    mean_ms: Option<f64>,
}

struct Timing {
    total_ms: f64,
    build_ms: f64,
    eval_sync_ms: f64,
    output_shapes: Vec<Vec<i32>>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    args.compile_mode.configure();
    let bench_target = args.stream_mode.configure()?;
    let ctx_lens = if args.ctx_lens.is_empty() {
        vec![128, 512, 725, 2048]
    } else {
        args.ctx_lens.clone()
    };

    let loader = Loader::open(&args.model).context("Loader::open")?;
    let cfg = Glm4MoeLiteConfig::from_loader(&loader).context("loading GLM config")?;
    let depths = selected_depths(&args.depths, cfg.num_hidden_layers)?;
    let model = BenchGlmModel::from_loader_with_config(&loader, cfg.clone())
        .context("loading BenchGlmModel")?;
    let sampler = Sampler::greedy();
    let mut records = Vec::new();

    for &ctx_len in &ctx_lens {
        for &depth in &depths {
            records.push(run_full_hidden_case(
                &model,
                &cfg,
                ctx_len,
                depth,
                &args,
                bench_target.target,
            )?);
            records.push(run_full_logits_case(
                &model,
                &cfg,
                ctx_len,
                depth,
                &args,
                bench_target.target,
            )?);
            records.push(run_full_logits_sample_case(
                &model,
                &cfg,
                &sampler,
                ctx_len,
                depth,
                &args,
                bench_target.target,
            )?);
            records.push(run_full_logits_repeat_case(
                &model,
                &cfg,
                ctx_len,
                depth,
                &args,
                bench_target.target,
            )?);
        }
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-glm-full-forward",
            model_dir: args.model.display().to_string(),
            ctx_lens,
            depths,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            stream_mode: bench_target.label,
            layer_eval_mode: args.layer_eval_mode.label(),
            block_mode: args.block_mode.label(),
            compile_mode: args.compile_mode.label(),
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            dtype: "bfloat16",
            cache_prealloc: "BenchGlmModel::make_cache(depth, ..., cap >= ctx_len + warmup + runs)",
            token_source: "deterministic synthetic token ids in [256, vocab_size)",
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
    validate_ctx_lens(&args.ctx_lens)?;
    if args.runs == 0 {
        return Err(anyhow!("--runs must be positive"));
    }
    Ok(())
}

fn validate_ctx_lens(ctx_lens: &[i32]) -> Result<()> {
    for &ctx_len in ctx_lens {
        if ctx_len <= 0 {
            return Err(anyhow!("--ctx-len values must be positive, got {ctx_len}"));
        }
    }
    Ok(())
}

fn validate_depths(depths: &[i32], num_hidden_layers: i32) -> Result<()> {
    if num_hidden_layers <= 0 {
        return Err(anyhow!(
            "num_hidden_layers must be positive, got {num_hidden_layers}"
        ));
    }
    for &depth in depths {
        if depth <= 0 || depth > num_hidden_layers {
            return Err(anyhow!(
                "--depth values must be in [1, {num_hidden_layers}], got {depth}"
            ));
        }
    }
    Ok(())
}

fn selected_depths(depths: &[i32], num_hidden_layers: i32) -> Result<Vec<i32>> {
    validate_depths(depths, num_hidden_layers)?;
    if depths.is_empty() {
        Ok(vec![num_hidden_layers])
    } else {
        Ok(depths.to_vec())
    }
}

struct BenchGlmModel {
    embed_tokens: Embedding,
    layers: Vec<Glm4DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    cfg: Glm4MoeLiteConfig,
}

impl BenchGlmModel {
    fn from_loader_with_config(loader: &Loader, cfg: Glm4MoeLiteConfig) -> Result<Self> {
        if cfg.tie_word_embeddings {
            return Err(anyhow!(
                "BenchGlmModel: tie_word_embeddings expected false (got true)"
            ));
        }
        let embed_tokens = Embedding::from_loader(loader, "model.embed_tokens")
            .context("loading BenchGlmModel embed_tokens")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            layers.push(
                Glm4DecoderLayer::from_loader(loader, i, &cfg)
                    .with_context(|| format!("loading BenchGlmModel layer {i}"))?,
            );
        }
        let norm = RmsNorm::from_loader(loader, "model.norm", cfg.rms_norm_eps)
            .context("loading BenchGlmModel norm")?;
        let lm_head =
            Linear::from_loader(loader, "lm_head").context("loading BenchGlmModel lm_head")?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg,
        })
    }

    fn make_cache(
        &self,
        depth: i32,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<Vec<LayerCache>> {
        validate_depths(&[depth], self.cfg.num_hidden_layers)?;
        Ok((0..depth)
            .map(|_| {
                LayerCache::Mla(
                    MlaLatentCache::new(
                        batch,
                        self.cfg.kv_lora_rank,
                        self.cfg.qk_rope_head_dim,
                        dtype,
                        cap,
                    )
                    .with_step(cap),
                )
            })
            .collect())
    }

    fn forward_hidden_depth(
        &self,
        input_ids: &Array,
        per_row_lens: Option<&[i32]>,
        mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        opts: ForwardOptions,
    ) -> Result<Array> {
        let target = opts.target;
        let depth = opts.depth;
        let depth_usize = usize::try_from(depth).context("depth must be positive")?;
        if depth_usize == 0 || depth_usize > self.layers.len() {
            return Err(anyhow!(
                "BenchGlmModel: depth {} out of [1, {}]",
                depth,
                self.layers.len()
            ));
        }

        let in_dims = input_ids.shape();
        let in_s = in_dims.as_slice();
        let batch = in_s[0];
        let seq_len = in_s[1];

        let caches = cache.ok_or_else(|| anyhow!("BenchGlmModel requires a cache"))?;
        if caches.len() != depth_usize {
            return Err(anyhow!(
                "BenchGlmModel: cache.len()={} != depth={depth}",
                caches.len()
            ));
        }

        let prl: Vec<i32> = per_row_lens
            .map(|s| s.to_vec())
            .unwrap_or_else(|| vec![seq_len; batch as usize]);
        if prl.len() != batch as usize {
            return Err(anyhow!(
                "BenchGlmModel: per_row_lens.len()={} != batch={}",
                prl.len(),
                batch
            ));
        }

        let offsets_vec = match &caches[0] {
            LayerCache::Mla(c) => c.offsets().to_vec(),
            _ => {
                return Err(anyhow!(
                    "BenchGlmModel: expected LayerCache::Mla at layer 0"
                ))
            }
        };
        let scalar_offset = (batch == 1).then_some(offsets_vec[0]);
        let per_row_offset: Option<Array> = if scalar_offset.is_some() {
            None
        } else {
            Some((&offsets_vec[..], &[batch][..]).try_into()?)
        };

        let owned_mask: Option<Array> = match mask {
            Some(_) => None,
            None if seq_len > 1 => {
                let lc = offsets_vec.iter().copied().max().unwrap_or(0) + seq_len;
                Some(build_internal_causal_mask(seq_len, lc, Dtype::Bfloat16)?)
            }
            None => None,
        };
        let effective_mask: Option<&Array> = mask.or(owned_mask.as_ref());

        let mut h = self.embed_tokens.forward_on(input_ids, target)?;
        for (i, layer) in self.layers.iter().take(depth_usize).enumerate() {
            let LayerCache::Mla(c) = &mut caches[i] else {
                return Err(anyhow!(
                    "BenchGlmModel: expected LayerCache::Mla at layer {i}"
                ));
            };
            h = if let Some(offset) = scalar_offset {
                layer.forward_on_scalar_offset_with_block_mode(
                    &h,
                    offset,
                    c,
                    &prl,
                    effective_mask,
                    target,
                    i as i32,
                    opts.block_mode,
                )?
            } else {
                let offset = per_row_offset
                    .as_ref()
                    .expect("per_row_offset must exist for batch > 1");
                layer.forward_on_with_block_mode(
                    &h,
                    offset,
                    c,
                    &prl,
                    effective_mask,
                    target,
                    i as i32,
                    opts.block_mode,
                )?
            };
            opts.layer_eval_mode.apply(&h)?;
        }
        self.norm.forward_on(&h, target)
    }

    fn forward_logits_depth(
        &self,
        input_ids: &Array,
        per_row_lens: Option<&[i32]>,
        mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        opts: ForwardOptions,
    ) -> Result<Array> {
        let target = opts.target;
        let hidden = self.forward_hidden_depth(input_ids, per_row_lens, mask, cache, opts)?;
        let dims_borrow = hidden.shape();
        let dims = dims_borrow.as_slice();
        let (b, s, hsz) = (dims[0], dims[1], dims[2]);
        let last_hidden = if s > 1 {
            mlx::ops::indexing::slice_strided_on(
                &hidden,
                &[0_i32, s - 1, 0][..],
                &[b, s, hsz][..],
                &[1_i32, 1, 1][..],
                target,
            )?
        } else {
            hidden
        };
        self.lm_head.forward_on(&last_hidden, target)
    }
}

fn build_internal_causal_mask(l: i32, lc: i32, dtype: Dtype) -> Result<Array> {
    let chunk_start = lc - l;
    if chunk_start < 0 {
        return Err(anyhow!(
            "build_internal_causal_mask: cache len Lc={lc} < query len L={l}"
        ));
    }
    let q_len = l as usize;
    let kv_len = lc as usize;
    let cs = chunk_start as usize;
    let neg_inf = f32::NEG_INFINITY;
    let mut flat = vec![neg_inf; q_len * kv_len];
    for q in 0..q_len {
        for k in 0..cs {
            flat[q * kv_len + k] = 0.0;
        }
        for k in 0..=q {
            flat[q * kv_len + cs + k] = 0.0;
        }
    }
    let arr_f32: Array = (&flat[..], &[1_i32, 1_i32, l, lc][..]).try_into()?;
    Ok(mlx::ops::cast::astype(&arr_f32, dtype)?)
}

fn run_full_hidden_case(
    model: &BenchGlmModel,
    cfg: &Glm4MoeLiteConfig,
    ctx_len: i32,
    depth: i32,
    args: &Args,
    target: StreamOrDevice,
) -> Result<Record> {
    let opts = ForwardOptions {
        target,
        layer_eval_mode: args.layer_eval_mode,
        block_mode: args.block_mode.to_decoder(),
        depth,
    };
    let mut cache = prepare_cache(
        model,
        cfg,
        ctx_len,
        depth,
        args,
        target,
        args.seed + depth as u64 * 10_000_000 + ctx_len as u64,
    )?;
    let mut decode_tokens = DecodeTokens::new(
        args.seed + depth as u64 * 10_000_000 + 1_000_000 + ctx_len as u64,
        args.warmup_runs + args.runs,
        cfg.vocab_size,
    )?;
    bench_case(
        ctx_len,
        depth,
        "full-hidden",
        args.warmup_runs,
        args.runs,
        || run_decode_hidden(model, decode_tokens.next()?, &mut cache, opts),
    )
}

fn run_full_logits_case(
    model: &BenchGlmModel,
    cfg: &Glm4MoeLiteConfig,
    ctx_len: i32,
    depth: i32,
    args: &Args,
    target: StreamOrDevice,
) -> Result<Record> {
    let opts = ForwardOptions {
        target,
        layer_eval_mode: args.layer_eval_mode,
        block_mode: args.block_mode.to_decoder(),
        depth,
    };
    let mut cache = prepare_cache(
        model,
        cfg,
        ctx_len,
        depth,
        args,
        target,
        args.seed + depth as u64 * 10_000_000 + 2_000_000 + ctx_len as u64,
    )?;
    let mut decode_tokens = DecodeTokens::new(
        args.seed + depth as u64 * 10_000_000 + 3_000_000 + ctx_len as u64,
        args.warmup_runs + args.runs,
        cfg.vocab_size,
    )?;
    bench_case(
        ctx_len,
        depth,
        "full-logits",
        args.warmup_runs,
        args.runs,
        || run_decode_logits(model, decode_tokens.next()?, &mut cache, opts),
    )
}

fn run_full_logits_sample_case(
    model: &BenchGlmModel,
    cfg: &Glm4MoeLiteConfig,
    sampler: &Sampler,
    ctx_len: i32,
    depth: i32,
    args: &Args,
    target: StreamOrDevice,
) -> Result<Record> {
    let opts = ForwardOptions {
        target,
        layer_eval_mode: args.layer_eval_mode,
        block_mode: args.block_mode.to_decoder(),
        depth,
    };
    let mut cache = prepare_cache(
        model,
        cfg,
        ctx_len,
        depth,
        args,
        target,
        args.seed + depth as u64 * 10_000_000 + 4_000_000 + ctx_len as u64,
    )?;
    let mut decode_tokens = DecodeTokens::new(
        args.seed + depth as u64 * 10_000_000 + 5_000_000 + ctx_len as u64,
        args.warmup_runs + args.runs,
        cfg.vocab_size,
    )?;
    bench_case(
        ctx_len,
        depth,
        "full-logits-sample",
        args.warmup_runs,
        args.runs,
        || run_decode_logits_sample(model, sampler, decode_tokens.next()?, &mut cache, opts),
    )
}

fn run_full_logits_repeat_case(
    model: &BenchGlmModel,
    cfg: &Glm4MoeLiteConfig,
    ctx_len: i32,
    depth: i32,
    args: &Args,
    target: StreamOrDevice,
) -> Result<Record> {
    let opts = ForwardOptions {
        target,
        layer_eval_mode: args.layer_eval_mode,
        block_mode: args.block_mode.to_decoder(),
        depth,
    };
    let mut cache = prepare_cache(
        model,
        cfg,
        ctx_len,
        depth,
        args,
        target,
        args.seed + depth as u64 * 10_000_000 + 6_000_000 + ctx_len as u64,
    )?;
    let mut decode_tokens = DecodeTokens::new(
        args.seed + depth as u64 * 10_000_000 + 7_000_000 + ctx_len as u64,
        args.warmup_runs + args.runs,
        cfg.vocab_size,
    )?;
    bench_case(
        ctx_len,
        depth,
        "full-logits-repeat",
        args.warmup_runs,
        args.runs,
        || run_decode_logits(model, decode_tokens.next()?, &mut cache, opts),
    )
}

fn prepare_cache(
    model: &BenchGlmModel,
    cfg: &Glm4MoeLiteConfig,
    ctx_len: i32,
    depth: i32,
    args: &Args,
    target: StreamOrDevice,
    seed: u64,
) -> Result<Vec<LayerCache>> {
    let extra_steps = i32::try_from(args.warmup_runs.saturating_add(args.runs))
        .context("warmup+runs exceeds i32")?;
    let cap = ctx_len
        .checked_add(extra_steps)
        .and_then(|n| n.checked_add(8))
        .ok_or_else(|| anyhow!("cache cap overflow for ctx_len={ctx_len}"))?;
    let mut cache = model.make_cache(depth, 1, cap, Dtype::Bfloat16)?;
    let ids = synthetic_token_ids(seed, ctx_len, cfg.vocab_size)?;
    let input: Array = (&ids[..], &[1_i32, ctx_len][..]).try_into()?;
    let hidden = model.forward_hidden_depth(
        &input,
        None,
        None,
        Some(&mut cache),
        ForwardOptions {
            target,
            layer_eval_mode: args.layer_eval_mode,
            block_mode: args.block_mode.to_decoder(),
            depth,
        },
    )?;
    mlx::transforms::eval(&[&hidden])?;
    mlx::transforms::synchronize()?;
    Ok(cache)
}

fn run_decode_hidden(
    model: &BenchGlmModel,
    token: u32,
    cache: &mut [LayerCache],
    opts: ForwardOptions,
) -> Result<Vec<Array>> {
    let input = decode_token_array(token)?;
    let hidden = model.forward_hidden_depth(&input, None, None, Some(cache), opts)?;
    Ok(vec![hidden])
}

fn run_decode_logits(
    model: &BenchGlmModel,
    token: u32,
    cache: &mut [LayerCache],
    opts: ForwardOptions,
) -> Result<Vec<Array>> {
    let logits = decode_logits(model, token, cache, opts)?;
    Ok(vec![logits])
}

fn run_decode_logits_sample(
    model: &BenchGlmModel,
    sampler: &Sampler,
    token: u32,
    cache: &mut [LayerCache],
    opts: ForwardOptions,
) -> Result<Vec<Array>> {
    let logits = decode_logits(model, token, cache, opts)?;
    let vocab = logits.shape().as_slice()[2];
    let flat = logits.reshape((vocab,))?;
    let next = sampler.sample_async_greedy(&flat)?;
    Ok(vec![next])
}

fn decode_logits(
    model: &BenchGlmModel,
    token: u32,
    cache: &mut [LayerCache],
    opts: ForwardOptions,
) -> Result<Array> {
    let input = decode_token_array(token)?;
    model.forward_logits_depth(&input, None, None, Some(cache), opts)
}

fn decode_token_array(token: u32) -> Result<Array> {
    let ids = [token];
    Ok((&ids[..], &[1_i32, 1_i32][..]).try_into()?)
}

struct DecodeTokens {
    ids: Vec<u32>,
    next_idx: usize,
}

impl DecodeTokens {
    fn new(seed: u64, len: usize, vocab_size: i32) -> Result<Self> {
        let len = i32::try_from(len).context("decode token count exceeds i32")?;
        Ok(Self {
            ids: synthetic_token_ids(seed, len, vocab_size)?,
            next_idx: 0,
        })
    }

    fn next(&mut self) -> Result<u32> {
        let token = self
            .ids
            .get(self.next_idx)
            .copied()
            .ok_or_else(|| anyhow!("DecodeTokens exhausted at index {}", self.next_idx))?;
        self.next_idx += 1;
        Ok(token)
    }
}

fn synthetic_token_ids(seed: u64, len: i32, vocab_size: i32) -> Result<Vec<u32>> {
    if len < 0 {
        return Err(anyhow!(
            "synthetic token len must be non-negative, got {len}"
        ));
    }
    if vocab_size <= 256 {
        return Err(anyhow!(
            "vocab_size must be greater than 256 for normal synthetic token range, got {vocab_size}"
        ));
    }
    let span = (vocab_size - 256) as u64;
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut ids = Vec::with_capacity(len as usize);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ids.push(256 + (state % span) as u32);
    }
    Ok(ids)
}

fn bench_case<F>(
    ctx_len: i32,
    depth: i32,
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
    let mut build_warmups_ms = Vec::with_capacity(warmup_runs);
    let mut eval_sync_warmups_ms = Vec::with_capacity(warmup_runs);
    for _ in 0..warmup_runs {
        let timing = time_once(&mut f)?;
        output_shapes = timing.output_shapes;
        warmups_ms.push(timing.total_ms);
        build_warmups_ms.push(timing.build_ms);
        eval_sync_warmups_ms.push(timing.eval_sync_ms);
    }

    let mut values_ms = Vec::with_capacity(runs);
    let mut build_values_ms = Vec::with_capacity(runs);
    let mut eval_sync_values_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let timing = time_once(&mut f)?;
        output_shapes = timing.output_shapes;
        values_ms.push(timing.total_ms);
        build_values_ms.push(timing.build_ms);
        eval_sync_values_ms.push(timing.eval_sync_ms);
    }

    Ok(Record {
        ctx_len,
        depth,
        case,
        output_shapes,
        summary: summarize(&values_ms),
        build_summary: summarize(&build_values_ms),
        eval_sync_summary: summarize(&eval_sync_values_ms),
        warmups_ms,
        build_warmups_ms,
        eval_sync_warmups_ms,
        values_ms,
        build_values_ms,
        eval_sync_values_ms,
    })
}

fn time_once<F>(f: &mut F) -> Result<Timing>
where
    F: FnMut() -> Result<Vec<Array>>,
{
    let started = Instant::now();
    let outputs = f()?;
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;
    let eval_started = Instant::now();
    let refs: Vec<&Array> = outputs.iter().collect();
    mlx::transforms::eval(&refs)?;
    mlx::transforms::synchronize()?;
    let eval_sync_ms = eval_started.elapsed().as_secs_f64() * 1000.0;
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    let output_shapes = outputs
        .iter()
        .map(|a| a.shape().as_slice().to_vec())
        .collect();
    Ok(Timing {
        total_ms,
        build_ms,
        eval_sync_ms,
        output_shapes,
    })
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

impl LayerEvalMode {
    fn label(self) -> &'static str {
        match self {
            LayerEvalMode::None => "none",
            LayerEvalMode::Eval => "eval",
            LayerEvalMode::EvalSync => "eval-sync",
        }
    }

    fn apply(self, hidden: &Array) -> Result<()> {
        match self {
            LayerEvalMode::None => Ok(()),
            LayerEvalMode::Eval => {
                mlx::transforms::eval(&[hidden])?;
                Ok(())
            }
            LayerEvalMode::EvalSync => {
                mlx::transforms::eval(&[hidden])?;
                mlx::transforms::synchronize()?;
                Ok(())
            }
        }
    }
}

impl BlockMode {
    fn label(self) -> &'static str {
        match self {
            BlockMode::Full => "full",
            BlockMode::SkipAttn => "skip-attn",
            BlockMode::SkipAttnRouted => "skip-attn-routed",
            BlockMode::SkipAttnRoutedFixed => "skip-attn-routed-fixed",
            BlockMode::SkipAttnShared => "skip-attn-shared",
            BlockMode::SkipFfn => "skip-ffn",
        }
    }

    fn to_decoder(self) -> DecoderBlockMode {
        match self {
            BlockMode::Full => DecoderBlockMode::Full,
            BlockMode::SkipAttn => DecoderBlockMode::SkipAttention,
            BlockMode::SkipAttnRouted => DecoderBlockMode::SkipAttentionRoutedOnly,
            BlockMode::SkipAttnRoutedFixed => DecoderBlockMode::SkipAttentionRoutedFixedOnly,
            BlockMode::SkipAttnShared => DecoderBlockMode::SkipAttentionSharedOnly,
            BlockMode::SkipFfn => DecoderBlockMode::SkipFfn,
        }
    }
}

impl CompileMode {
    fn label(self) -> &'static str {
        match self {
            CompileMode::Enabled => "enabled",
            CompileMode::Disabled => "disabled",
            CompileMode::NoSimplify => "no-simplify",
            CompileMode::NoFuse => "no-fuse",
        }
    }

    fn configure(self) {
        let mode = match self {
            CompileMode::Enabled => MlxCompileMode::Enabled,
            CompileMode::Disabled => MlxCompileMode::Disabled,
            CompileMode::NoSimplify => MlxCompileMode::NoSimplify,
            CompileMode::NoFuse => MlxCompileMode::NoFuse,
        };
        mlx::compile::set_compile_mode(mode);
    }
}

fn print_summary(output: &BenchOutput) {
    println!("# ironmlx-glm-full-forward-bench");
    println!(
        "layers={} H={} V={} stream={} layer_eval={} block_mode={} compile_mode={}",
        output.meta.num_hidden_layers,
        output.meta.hidden_size,
        output.meta.vocab_size,
        output.meta.stream_mode,
        output.meta.layer_eval_mode,
        output.meta.block_mode,
        output.meta.compile_mode
    );
    for record in &output.records {
        println!(
            "ctx={:<5} depth={:<2} case={:<22} p50={:>8.4} ms p95={:>8.4} ms build_p50={:>8.4} ms eval_sync_p50={:>8.4} ms",
            record.ctx_len,
            record.depth,
            record.case,
            record.summary.p50_ms.unwrap_or(f64::NAN),
            record.summary.p95_ms.unwrap_or(f64::NAN),
            record.build_summary.p50_ms.unwrap_or(f64::NAN),
            record.eval_sync_summary.p50_ms.unwrap_or(f64::NAN)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_ctx_lens_rejects_non_positive_values() {
        assert!(validate_ctx_lens(&[128, 0]).is_err());
        assert!(validate_ctx_lens(&[-1]).is_err());
    }

    #[test]
    fn validate_ctx_lens_accepts_default_empty_and_positive_values() {
        validate_ctx_lens(&[]).unwrap();
        validate_ctx_lens(&[128, 512]).unwrap();
    }

    #[test]
    fn validate_depths_rejects_out_of_range_values() {
        assert!(validate_depths(&[0], 47).is_err());
        assert!(validate_depths(&[48], 47).is_err());
    }

    #[test]
    fn selected_depths_default_to_full_depth() {
        assert_eq!(selected_depths(&[], 47).unwrap(), vec![47]);
        assert_eq!(selected_depths(&[1, 4, 47], 47).unwrap(), vec![1, 4, 47]);
    }

    #[test]
    fn layer_eval_mode_labels_are_stable() {
        assert_eq!(LayerEvalMode::None.label(), "none");
        assert_eq!(LayerEvalMode::Eval.label(), "eval");
        assert_eq!(LayerEvalMode::EvalSync.label(), "eval-sync");
    }

    #[test]
    fn block_mode_labels_are_stable() {
        assert_eq!(BlockMode::Full.label(), "full");
        assert_eq!(BlockMode::SkipAttn.label(), "skip-attn");
        assert_eq!(BlockMode::SkipAttnRouted.label(), "skip-attn-routed");
        assert_eq!(
            BlockMode::SkipAttnRoutedFixed.label(),
            "skip-attn-routed-fixed"
        );
        assert_eq!(BlockMode::SkipAttnShared.label(), "skip-attn-shared");
        assert_eq!(BlockMode::SkipFfn.label(), "skip-ffn");
    }

    #[test]
    fn compile_mode_labels_are_stable() {
        assert_eq!(CompileMode::Enabled.label(), "enabled");
        assert_eq!(CompileMode::Disabled.label(), "disabled");
        assert_eq!(CompileMode::NoSimplify.label(), "no-simplify");
        assert_eq!(CompileMode::NoFuse.label(), "no-fuse");
    }

    #[test]
    fn synthetic_token_ids_stay_in_normal_vocab_range() {
        let ids = synthetic_token_ids(7, 16, 1024).unwrap();
        assert_eq!(ids.len(), 16);
        assert!(ids.iter().all(|&id| (256..1024).contains(&id)));
        assert_eq!(ids, synthetic_token_ids(7, 16, 1024).unwrap());
        assert_ne!(ids, synthetic_token_ids(8, 16, 1024).unwrap());
    }

    #[test]
    fn synthetic_token_ids_rejects_invalid_inputs() {
        assert!(synthetic_token_ids(7, -1, 1024).is_err());
        assert!(synthetic_token_ids(7, 1, 256).is_err());
    }

    #[test]
    fn decode_tokens_errors_when_exhausted() {
        let mut tokens = DecodeTokens::new(7, 1, 1024).unwrap();
        tokens.next().unwrap();
        assert!(tokens.next().is_err());
    }

    #[test]
    fn percentile_interpolates() {
        assert_eq!(percentile(&[1.0, 3.0], 50.0), Some(2.0));
    }
}
