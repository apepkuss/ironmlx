//! GLM-4.7 MoE routed-experts micro-benchmark.
//!
//! This binary isolates the exact `gather_quantized_matmul_on` shapes used by
//! `glm4_moe_lite` single-token decode routed experts. It intentionally uses
//! the real checkpoint tensors, not synthetic quantized weights, so the numbers
//! can be compared with decode attribution runs.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::Loader;
use ironmlx::models::qwen3_5_moe::RoutedExperts;
use mlx::compile::{compile, CompiledFn, ShapeMode};
use mlx::ops::indexing::{slice_on, take_along_axis_on, take_on};
use mlx::ops::sort::argsort_on;
use mlx::{random, Array, Device, Dtype, StreamOrDevice};
use serde::Serialize;

const SORTED_ROUTING_MIN_BS_K: i32 = 64;
const MAX_EXACT_U32_IN_F32: i32 = 1 << 24;

#[derive(Parser, Debug)]
#[command(
    name = "ironmlx-glm-moe-bench",
    about = "Direct GLM-4.7 routed-experts gather_qmm benchmark",
    version
)]
struct Args {
    /// Local GLM-4.7-Flash-4bit model directory.
    #[arg(long)]
    model: PathBuf,

    /// MoE layer index to load. Layer 0 is dense, so default starts at 1.
    #[arg(long, default_value_t = 1)]
    layer: i32,

    /// Flat token counts (BS) to measure. Pass multiple times.
    #[arg(long)]
    bs: Vec<i32>,

    /// Routed experts per token.
    #[arg(long, default_value_t = 4)]
    k: i32,

    /// Timed runs per case.
    #[arg(long, default_value_t = 50)]
    runs: usize,

    /// Warmup runs per case.
    #[arg(long, default_value_t = 10)]
    warmup_runs: usize,

    /// PRNG seed for synthetic hidden states.
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

struct ExpertWeights {
    fused_gate_up_weight: Array,
    fused_gate_up_scales: Array,
    fused_gate_up_biases: Option<Array>,
    down_weight: Array,
    down_scales: Array,
    down_biases: Option<Array>,
    group_size: i32,
    bits: i32,
    num_experts: i32,
    hidden_size: i32,
    moe_intermediate: i32,
}

struct GateUpResult {
    gate_out: Array,
    up_out: Array,
    rhs_idx_used: Array,
    sorted_flag: bool,
    sort_perm: Option<Array>,
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
    bs_values: Vec<i32>,
    k: i32,
    warmup_runs: usize,
    measured_runs: usize,
    stream_mode: &'static str,
    sorted_routing_min_bs_k: i32,
    hidden_size: i32,
    moe_intermediate: i32,
    num_experts: i32,
    group_size: i32,
    bits: i32,
}

#[derive(Serialize)]
struct Record {
    bs: i32,
    k: i32,
    branch: &'static str,
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
    let bs_values = if args.bs.is_empty() {
        vec![1, 16]
    } else {
        args.bs.clone()
    };

    let loader = Loader::open(&args.model).context("Loader::open")?;
    let prefix = format!("model.layers.{}.mlp.switch_mlp", args.layer);
    let experts = load_expert_weights(&loader, &prefix, bench_target.target)
        .with_context(|| format!("loading routed experts from {prefix}"))?;
    let production_routed = RoutedExperts::from_loader(&loader, &prefix)
        .with_context(|| format!("loading production RoutedExperts from {prefix}"))?;
    let swiglu = build_swiglu()?;

    let mut records = Vec::new();
    for &bs in &bs_values {
        let route = build_route_inputs(
            args.seed + bs as u64,
            bs,
            args.k,
            experts.hidden_size,
            experts.num_experts,
        )?;
        mlx::transforms::eval(&[&route.x, &route.inds, &route.weights])?;

        let branch = routing_branch(bs, args.k);
        records.push(bench_case(
            bs,
            args.k,
            branch,
            "production-apply-experts",
            args.warmup_runs,
            args.runs,
            || {
                production_routed
                    .apply_experts(
                        &route.x,
                        &route.inds,
                        &route.weights,
                        bench_target.target,
                        args.layer,
                    )
                    .map(|out| vec![out])
            },
        )?);

        records.push(bench_case(
            bs,
            args.k,
            branch,
            "local-full-pipeline",
            args.warmup_runs,
            args.runs,
            || {
                local_routed_pipeline(
                    &experts,
                    &swiglu,
                    &route.x,
                    &route.inds,
                    &route.weights,
                    bench_target.target,
                )
                .map(|out| vec![out])
            },
        )?);

        records.push(bench_case(
            bs,
            args.k,
            branch,
            "gate-up-gather-qmm",
            args.warmup_runs,
            args.runs,
            || {
                compute_gate_up(&experts, &route.x, &route.inds, bench_target.target)
                    .map(|out| vec![out.gate_out, out.up_out])
            },
        )?);

        let gate_state = compute_gate_up(&experts, &route.x, &route.inds, bench_target.target)?;
        mlx::transforms::eval(&[&gate_state.gate_out, &gate_state.up_out])?;
        records.push(bench_case(
            bs,
            args.k,
            branch,
            "swiglu",
            args.warmup_runs,
            args.runs,
            || {
                invoke_swiglu(&swiglu, &gate_state.gate_out, &gate_state.up_out)
                    .map(|out| vec![out])
            },
        )?);

        let act = invoke_swiglu(&swiglu, &gate_state.gate_out, &gate_state.up_out)?;
        mlx::transforms::eval(&[&act])?;
        records.push(bench_case(
            bs,
            args.k,
            branch,
            "down-gather-qmm",
            args.warmup_runs,
            args.runs,
            || {
                compute_down(&experts, &act, &gate_state, bs, args.k, bench_target.target)
                    .map(|out| vec![out])
            },
        )?);

        let down_out = compute_down(&experts, &act, &gate_state, bs, args.k, bench_target.target)?;
        mlx::transforms::eval(&[&down_out])?;
        records.push(bench_case(
            bs,
            args.k,
            branch,
            "weighted-reduce",
            args.warmup_runs,
            args.runs,
            || weighted_reduce(&down_out, &route.weights, bench_target.target).map(|out| vec![out]),
        )?);
    }

    let output = BenchOutput {
        meta: Meta {
            backend: "ironmlx-glm-moe",
            model_dir: args.model.display().to_string(),
            layer: args.layer,
            bs_values,
            k: args.k,
            warmup_runs: args.warmup_runs,
            measured_runs: args.runs,
            stream_mode: bench_target.label,
            sorted_routing_min_bs_k: SORTED_ROUTING_MIN_BS_K,
            hidden_size: experts.hidden_size,
            moe_intermediate: experts.moe_intermediate,
            num_experts: experts.num_experts,
            group_size: experts.group_size,
            bits: experts.bits,
        },
        records,
    };
    std::fs::write(&args.out, serde_json::to_string_pretty(&output)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;
    print_summary(&output);
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    if args.layer <= 0 {
        return Err(anyhow!(
            "--layer must be a MoE layer >= 1, got {}",
            args.layer
        ));
    }
    if let Some(&bs) = args.bs.iter().find(|&&bs| bs <= 0) {
        return Err(anyhow!("--bs values must be positive, got {bs}"));
    }
    if args.k <= 0 {
        return Err(anyhow!("--k must be positive, got {}", args.k));
    }
    if args.runs == 0 {
        return Err(anyhow!("--runs must be positive"));
    }
    Ok(())
}

fn load_expert_weights(
    loader: &Loader,
    prefix: &str,
    target: StreamOrDevice,
) -> Result<ExpertWeights> {
    let qmeta = loader
        .quant_meta()
        .ok_or_else(|| anyhow!("missing global quantization metadata"))?;
    let gate_weight = loader
        .tensor(&format!("{prefix}.gate_proj.weight"))?
        .clone();
    let gate_scales = loader
        .tensor(&format!("{prefix}.gate_proj.scales"))?
        .clone();
    let gate_biases = loader
        .tensor_opt(&format!("{prefix}.gate_proj.biases"))
        .cloned();
    let up_weight = loader.tensor(&format!("{prefix}.up_proj.weight"))?.clone();
    let up_scales = loader.tensor(&format!("{prefix}.up_proj.scales"))?.clone();
    let up_biases = loader
        .tensor_opt(&format!("{prefix}.up_proj.biases"))
        .cloned();
    let down_weight = loader
        .tensor(&format!("{prefix}.down_proj.weight"))?
        .clone();
    let down_scales = loader
        .tensor(&format!("{prefix}.down_proj.scales"))?
        .clone();
    let down_biases = loader
        .tensor_opt(&format!("{prefix}.down_proj.biases"))
        .cloned();

    let num_experts = gate_weight.shape().as_slice()[0];
    let moe_intermediate = gate_weight.shape().as_slice()[1];
    let hidden_size = down_weight.shape().as_slice()[1];
    let fused_gate_up_weight =
        mlx::ops::shape::concatenate_on(&[&gate_weight, &up_weight], 1, target)?;
    let fused_gate_up_scales =
        mlx::ops::shape::concatenate_on(&[&gate_scales, &up_scales], 1, target)?;
    let fused_gate_up_biases = match (gate_biases.as_ref(), up_biases.as_ref()) {
        (Some(gb), Some(ub)) => Some(mlx::ops::shape::concatenate_on(&[gb, ub], 1, target)?),
        (None, None) => None,
        _ => return Err(anyhow!("{prefix}: gate/up quant bias presence mismatch")),
    };

    let mut to_eval = vec![
        &fused_gate_up_weight,
        &fused_gate_up_scales,
        &down_weight,
        &down_scales,
    ];
    if let Some(b) = &fused_gate_up_biases {
        to_eval.push(b);
    }
    if let Some(b) = &down_biases {
        to_eval.push(b);
    }
    mlx::transforms::eval(&to_eval)?;

    Ok(ExpertWeights {
        fused_gate_up_weight,
        fused_gate_up_scales,
        fused_gate_up_biases,
        down_weight,
        down_scales,
        down_biases,
        group_size: qmeta.group_size,
        bits: qmeta.bits,
        num_experts,
        hidden_size,
        moe_intermediate,
    })
}

struct RouteInputs {
    x: Array,
    inds: Array,
    weights: Array,
}

fn build_route_inputs(
    seed: u64,
    bs: i32,
    k: i32,
    hidden_size: i32,
    num_experts: i32,
) -> Result<RouteInputs> {
    let key = random::key(seed).context("random key")?;
    let x = random::normal()
        .shape((bs, hidden_size))
        .dtype(Dtype::Bfloat16)
        .key(&key)
        .sample()
        .context("sample hidden states")?;

    let inds_vec = build_cycled_topk_indices(bs, k, num_experts);
    let inds: Array = (inds_vec.as_slice(), [bs, k]).try_into()?;
    let weight = 1.0_f32 / k as f32;
    let weights_vec = vec![weight; (bs * k) as usize];
    let weights: Array = (weights_vec.as_slice(), [bs, k]).try_into()?;
    Ok(RouteInputs { x, inds, weights })
}

fn build_cycled_topk_indices(bs: i32, k: i32, num_experts: i32) -> Vec<u32> {
    let total = usize::try_from(bs.saturating_mul(k)).unwrap_or(0);
    let experts = num_experts.max(1) as u32;
    (0..total).map(|i| (i as u32) % experts).collect()
}

fn routing_branch(bs: i32, k: i32) -> &'static str {
    if uses_sorted_routing(bs, k) {
        "sorted"
    } else {
        "default"
    }
}

fn uses_sorted_routing(bs: i32, k: i32) -> bool {
    bs * k >= SORTED_ROUTING_MIN_BS_K
}

fn compute_gate_up(
    experts: &ExpertWeights,
    x: &Array,
    inds: &Array,
    target: StreamOrDevice,
) -> Result<GateUpResult> {
    let bs = x.shape().as_slice()[0];
    let idims = inds.shape();
    let k = idims.as_slice()[1];
    let bs_k = bs * k;
    if uses_sorted_routing(bs, k) {
        let flat_topk = mlx::ops::shape::reshape(inds, [bs_k])?;
        let sort_perm = argsort_on(&flat_topk, -1_i32, target)?;
        let sorted_topk = take_along_axis_on(&flat_topk, &sort_perm, -1_i32, target)?;
        let sorted_token_idx = sorted_token_indices_from_sort_perm(&sort_perm, k, bs_k, target)?;
        let sorted_x_2d = take_on(x, &sorted_token_idx, 0_i32, target)?;
        let sorted_x_3d = mlx::ops::shape::expand_dims_on(&sorted_x_2d, -2_i32, target)?;
        let gate_up_out = mlx::quantization::gather_quantized_matmul_on(
            &sorted_x_3d,
            &experts.fused_gate_up_weight,
            &experts.fused_gate_up_scales,
            experts.fused_gate_up_biases.as_ref(),
            None,
            Some(&sorted_topk),
            true,
            Some(experts.group_size),
            Some(experts.bits),
            "affine",
            true,
            target,
        )?;
        let i = experts.moe_intermediate;
        let gate_out = slice_on(&gate_up_out, [0_i32, 0, 0], [bs_k, 1, i], target)?;
        let up_out = slice_on(&gate_up_out, [0_i32, 0, i], [bs_k, 1, 2 * i], target)?;
        Ok(GateUpResult {
            gate_out,
            up_out,
            rhs_idx_used: sorted_topk,
            sorted_flag: true,
            sort_perm: Some(sort_perm),
        })
    } else {
        let x_in = mlx::ops::shape::expand_dims_on(x, &[-2_i32, -3_i32][..], target)?;
        let gate_up_out = mlx::quantization::gather_quantized_matmul_on(
            &x_in,
            &experts.fused_gate_up_weight,
            &experts.fused_gate_up_scales,
            experts.fused_gate_up_biases.as_ref(),
            None,
            Some(inds),
            true,
            Some(experts.group_size),
            Some(experts.bits),
            "affine",
            false,
            target,
        )?;
        let i = experts.moe_intermediate;
        let gate_out = slice_on(&gate_up_out, [0_i32, 0, 0, 0], [bs, k, 1, i], target)?;
        let up_out = slice_on(&gate_up_out, [0_i32, 0, 0, i], [bs, k, 1, 2 * i], target)?;
        Ok(GateUpResult {
            gate_out,
            up_out,
            rhs_idx_used: inds.clone(),
            sorted_flag: false,
            sort_perm: None,
        })
    }
}

fn compute_down(
    experts: &ExpertWeights,
    act: &Array,
    gate_state: &GateUpResult,
    bs: i32,
    k: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let bs_k = bs * k;
    let down_out_raw = mlx::quantization::gather_quantized_matmul_on(
        act,
        &experts.down_weight,
        &experts.down_scales,
        experts.down_biases.as_ref(),
        None,
        Some(&gate_state.rhs_idx_used),
        true,
        Some(experts.group_size),
        Some(experts.bits),
        "affine",
        gate_state.sorted_flag,
        target,
    )?;

    if let Some(sort_perm) = &gate_state.sort_perm {
        let inv_perm = argsort_on(sort_perm, -1_i32, target)?;
        let down_out_2d = mlx::ops::shape::reshape(&down_out_raw, [bs_k, experts.hidden_size])?;
        let unsorted_2d = take_on(&down_out_2d, &inv_perm, 0_i32, target)?;
        mlx::ops::shape::reshape(&unsorted_2d, [bs, k, experts.hidden_size])
            .map_err(anyhow::Error::from)
    } else {
        mlx::ops::shape::squeeze_on(&down_out_raw, &[-2_i32][..], target)
            .map_err(anyhow::Error::from)
    }
}

fn weighted_reduce(down_out: &Array, weights: &Array, target: StreamOrDevice) -> Result<Array> {
    let weights_unsq = mlx::ops::shape::expand_dims_on(weights, -1_i32, target)?;
    let weighted = down_out * &weights_unsq;
    mlx::ops::sum_on(&weighted, -2_i32, false, target).map_err(anyhow::Error::from)
}

fn local_routed_pipeline(
    experts: &ExpertWeights,
    swiglu: &CompiledFn,
    x: &Array,
    inds: &Array,
    weights: &Array,
    target: StreamOrDevice,
) -> Result<Array> {
    let bs = x.shape().as_slice()[0];
    let k = inds.shape().as_slice()[1];
    let gate_state = compute_gate_up(experts, x, inds, target)?;
    let act = invoke_swiglu(swiglu, &gate_state.gate_out, &gate_state.up_out)?;
    let down_out = compute_down(experts, &act, &gate_state, bs, k, target)?;
    weighted_reduce(&down_out, weights, target)
}

fn build_swiglu() -> Result<CompiledFn> {
    compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let gate = inputs[0];
            let up = inputs[1];
            let gate_sig = gate.sigmoid()?;
            let gate_silu = gate * &gate_sig;
            Ok(vec![&gate_silu * up])
        },
        ShapeMode::Shapeless,
    )
    .map_err(anyhow::Error::from)
}

fn invoke_swiglu(func: &CompiledFn, gate: &Array, up: &Array) -> Result<Array> {
    let mut outs = func.invoke(&[gate, up]).map_err(anyhow::Error::from)?;
    outs.pop()
        .ok_or_else(|| anyhow!("SwiGLU returned no output"))
}

fn sorted_token_indices_from_sort_perm(
    sort_perm: &Array,
    k: i32,
    bs_k: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    if bs_k <= MAX_EXACT_U32_IN_F32 {
        let sort_perm_f32 = mlx::ops::cast::astype_on(sort_perm, Dtype::Float32, target)?;
        let k_scalar: Array = (&[k as f32][..], ()).try_into()?;
        let div = sort_perm_f32.try_div_on(&k_scalar, target)?;
        let sorted_token_idx_f32 = div.floor_on(target)?;
        return mlx::ops::cast::astype_on(&sorted_token_idx_f32, Dtype::Uint32, target)
            .map_err(anyhow::Error::from);
    }

    let bs_k_usize = usize::try_from(bs_k)?;
    let k_usize = usize::try_from(k)?;
    let token_idx_vec: Vec<u32> = (0..bs_k_usize).map(|i| (i / k_usize) as u32).collect();
    let token_idx: Array = (token_idx_vec.as_slice(), [bs_k]).try_into()?;
    take_along_axis_on(&token_idx, sort_perm, -1_i32, target).map_err(anyhow::Error::from)
}

fn bench_case<F>(
    bs: i32,
    k: i32,
    branch: &'static str,
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
        bs,
        k,
        branch,
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
    println!("# ironmlx-glm-moe-bench");
    println!(
        "layer={} H={} I={} E={} k={} stream={}",
        output.meta.layer,
        output.meta.hidden_size,
        output.meta.moe_intermediate,
        output.meta.num_experts,
        output.meta.k,
        output.meta.stream_mode
    );
    for record in &output.records {
        println!(
            "bs={:<3} branch={:<7} case={:<24} p50={:>8.4} ms p95={:>8.4} ms",
            record.bs,
            record.branch,
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
    fn build_cycled_topk_indices_wraps_expert_ids() {
        assert_eq!(
            build_cycled_topk_indices(3, 4, 5),
            vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
        );
    }

    #[test]
    fn routing_branch_matches_production_threshold() {
        assert_eq!(routing_branch(1, 4), "default");
        assert_eq!(routing_branch(16, 4), "sorted");
    }

    #[test]
    fn percentile_interpolates_between_points() {
        assert_eq!(percentile(&[1.0, 3.0], 50.0), Some(2.0));
    }
}
