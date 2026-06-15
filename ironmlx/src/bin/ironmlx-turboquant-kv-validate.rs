use std::collections::HashSet;
use std::fs::File;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use ironmlx::core::cache::TurboQuantKVBits;
use ironmlx::core::generate::{build_position_ids, slice_logits_row};
use ironmlx::core::{Loader, Tokenizer};
use ironmlx::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF;
use ironmlx::models::{ModelArchitecture, Qwen35Model};
use ironmlx::nn::enable_turboquant_kv_caches;
use mlx::{Array, Dtype};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(about = "Validate TurboQuant KV logits drift against a greedy baseline")]
struct Args {
    /// Model directory containing config.json, tokenizer.json, and safetensors.
    #[arg(long)]
    model: PathBuf,

    /// Plain text prompt. The content is encoded directly without chat templating.
    #[arg(long)]
    prompt_file: PathBuf,

    /// Number of greedy baseline decode steps to compare.
    #[arg(long, default_value_t = 16)]
    max_tokens: usize,

    /// Output JSON path.
    #[arg(long)]
    out: PathBuf,

    /// KV quantization configs to validate.
    #[arg(
        long = "kv-quant",
        value_enum,
        value_delimiter = ',',
        default_value = "none,turbo3,turbo4,k3v4"
    )]
    kv_quant: Vec<KvQuantArg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
enum KvQuantArg {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "turbo3")]
    Turbo3,
    #[serde(rename = "turbo4")]
    Turbo4,
    #[value(name = "k3v4")]
    #[serde(rename = "k3v4")]
    K3V4,
}

impl KvQuantArg {
    fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Turbo3 => "turbo3",
            Self::Turbo4 => "turbo4",
            Self::K3V4 => "k3v4",
        }
    }

    fn turboquant_bits(self) -> Option<TurboQuantKVBits> {
        match self {
            Self::None => None,
            Self::Turbo3 => Some(TurboQuantKVBits::K3V3),
            Self::Turbo4 => Some(TurboQuantKVBits::K4V4),
            Self::K3V4 => Some(TurboQuantKVBits::K3V4),
        }
    }
}

#[derive(Serialize)]
struct ValidationOutput {
    meta: Meta,
    baseline: BaselineOutput,
    configs: Vec<ConfigOutput>,
}

#[derive(Serialize)]
struct Meta {
    backend: &'static str,
    model_dir: String,
    prompt_file: String,
    prompt_tokens: usize,
    max_tokens: usize,
    kv_quant: Vec<KvQuantArg>,
}

#[derive(Serialize)]
struct BaselineOutput {
    generated_token_ids: Vec<u32>,
    generated_text: String,
}

#[derive(Serialize)]
struct ConfigOutput {
    kv_quant: KvQuantArg,
    label: &'static str,
    generated_token_ids: Vec<u32>,
    generated_text: String,
    exact_token_match_count: usize,
    first_token_mismatch_step: Option<usize>,
    steps: Vec<StepMetrics>,
}

#[derive(Serialize)]
struct StepMetrics {
    step: usize,
    baseline_token: u32,
    baseline_argmax: u32,
    candidate_argmax: u32,
    argmax_matches: bool,
    max_abs_diff: f32,
    mean_abs_diff: f32,
    rms_diff: f32,
    cosine_similarity: f32,
    top5_overlap: usize,
}

struct BaselineRun {
    generated_token_ids: Vec<u32>,
    logits_by_step: Vec<Vec<f32>>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

fn run(args: Args) -> Result<()> {
    if args.max_tokens == 0 {
        return Err(anyhow!("--max-tokens must be > 0"));
    }
    if args.kv_quant.is_empty() {
        return Err(anyhow!("--kv-quant must include at least one config"));
    }

    let rendered_prompt = std::fs::read_to_string(&args.prompt_file)
        .with_context(|| format!("reading {}", args.prompt_file.display()))?;
    let loader = Loader::open(&args.model).context("Loader::open")?;
    let arch = ModelArchitecture::from_config_value(loader.config_raw_value())?;
    if arch != ModelArchitecture::Qwen35Dense {
        return Err(anyhow!(
            "ironmlx-turboquant-kv-validate currently validates Qwen3.5 dense text models only; got model_type={}",
            arch.model_type()
        ));
    }

    let model = Qwen35Model::from_loader(&loader).context("Qwen35Model::from_loader")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let prompt_ids = tokenizer.encode(&rendered_prompt, false)?;
    if prompt_ids.is_empty() {
        return Err(anyhow!("prompt_file encoded to zero tokens"));
    }

    let baseline = run_baseline(&model, &prompt_ids, args.max_tokens)?;
    let baseline_text = tokenizer.decode(&baseline.generated_token_ids, true)?;
    let mut configs = Vec::with_capacity(args.kv_quant.len());
    for kv_quant in &args.kv_quant {
        configs.push(run_candidate_replay(
            &model,
            &tokenizer,
            &prompt_ids,
            &baseline,
            *kv_quant,
        )?);
    }

    let output = ValidationOutput {
        meta: Meta {
            backend: "mlx",
            model_dir: args.model.display().to_string(),
            prompt_file: args.prompt_file.display().to_string(),
            prompt_tokens: prompt_ids.len(),
            max_tokens: args.max_tokens,
            kv_quant: args.kv_quant,
        },
        baseline: BaselineOutput {
            generated_token_ids: baseline.generated_token_ids,
            generated_text: baseline_text,
        },
        configs,
    };

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let file =
        File::create(&args.out).with_context(|| format!("creating {}", args.out.display()))?;
    serde_json::to_writer_pretty(file, &output)?;
    Ok(())
}

fn run_baseline(model: &Qwen35Model, prompt_ids: &[u32], max_tokens: usize) -> Result<BaselineRun> {
    let mut cache = make_cache(model, prompt_ids.len(), max_tokens, KvQuantArg::None)?;
    let mut logits = prefill_logits(model, prompt_ids, &mut cache)?;
    let mut generated_token_ids = Vec::with_capacity(max_tokens);
    let mut logits_by_step = Vec::with_capacity(max_tokens);

    for step in 0..max_tokens {
        logits_by_step.push(logits.clone());
        let next_token = argmax(&logits)?;
        generated_token_ids.push(next_token);
        if step + 1 < max_tokens {
            logits = decode_logits(model, prompt_ids.len() + step, next_token, &mut cache)?;
        }
    }

    Ok(BaselineRun {
        generated_token_ids,
        logits_by_step,
    })
}

fn run_candidate_replay(
    model: &Qwen35Model,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    baseline: &BaselineRun,
    kv_quant: KvQuantArg,
) -> Result<ConfigOutput> {
    let mut cache = make_cache(
        model,
        prompt_ids.len(),
        baseline.generated_token_ids.len(),
        kv_quant,
    )?;
    let mut logits = prefill_logits(model, prompt_ids, &mut cache)?;
    let mut generated_token_ids = Vec::with_capacity(baseline.generated_token_ids.len());
    let mut steps = Vec::with_capacity(baseline.generated_token_ids.len());

    for step in 0..baseline.generated_token_ids.len() {
        let baseline_logits = &baseline.logits_by_step[step];
        let candidate_argmax = argmax(&logits)?;
        generated_token_ids.push(candidate_argmax);
        let baseline_argmax = argmax(baseline_logits)?;
        steps.push(compare_logits(
            step,
            baseline.generated_token_ids[step],
            baseline_argmax,
            candidate_argmax,
            baseline_logits,
            &logits,
        )?);

        if step + 1 < baseline.generated_token_ids.len() {
            logits = decode_logits(
                model,
                prompt_ids.len() + step,
                baseline.generated_token_ids[step],
                &mut cache,
            )?;
        }
    }

    let exact_token_match_count = generated_token_ids
        .iter()
        .zip(&baseline.generated_token_ids)
        .take_while(|(candidate, baseline)| candidate == baseline)
        .count();
    let first_token_mismatch_step = (exact_token_match_count < baseline.generated_token_ids.len())
        .then_some(exact_token_match_count);
    let generated_text = tokenizer.decode(&generated_token_ids, true)?;

    Ok(ConfigOutput {
        kv_quant,
        label: kv_quant.label(),
        generated_token_ids,
        generated_text,
        exact_token_match_count,
        first_token_mismatch_step,
        steps,
    })
}

fn make_cache(
    model: &Qwen35Model,
    prompt_len: usize,
    max_tokens: usize,
    kv_quant: KvQuantArg,
) -> Result<Vec<ironmlx::nn::LayerCache>> {
    let cap = prompt_len
        .saturating_add(max_tokens)
        .max(MIN_KV_CACHE_CAP_FOR_GPU_PERF as usize) as i32;
    let mut cache = model.make_cache(1, cap, Dtype::Bfloat16)?;
    if let Some(bits) = kv_quant.turboquant_bits() {
        enable_turboquant_kv_caches(&mut cache, bits)?;
    }
    Ok(cache)
}

fn prefill_logits(
    model: &Qwen35Model,
    prompt_ids: &[u32],
    cache: &mut [ironmlx::nn::LayerCache],
) -> Result<Vec<f32>> {
    let prompt_len = prompt_ids.len() as i32;
    let input_ids = token_array(prompt_ids)?;
    let position_ids = build_position_ids(0, prompt_len)?;
    let logits = model.forward_on(&input_ids, &position_ids, None, None, Some(cache), ())?;
    logits_vec(&logits)
}

fn decode_logits(
    model: &Qwen35Model,
    token_position: usize,
    token: u32,
    cache: &mut [ironmlx::nn::LayerCache],
) -> Result<Vec<f32>> {
    let token_ids = [token];
    let input_ids = token_array(&token_ids)?;
    let position_ids = build_position_ids(token_position as i32, 1)?;
    let logits = model.forward_on(&input_ids, &position_ids, None, None, Some(cache), ())?;
    logits_vec(&logits)
}

fn token_array(ids: &[u32]) -> Result<Array> {
    let seq = i32::try_from(ids.len()).context("token sequence length does not fit i32")?;
    let arr: Array = (ids, &[1_i32, seq][..]).try_into()?;
    Ok(arr)
}

fn logits_vec(logits: &Array) -> Result<Vec<f32>> {
    let row = slice_logits_row(logits, 0)?;
    Ok(row.astype(Dtype::Float32)?.to_vec::<f32>()?)
}

fn compare_logits(
    step: usize,
    baseline_token: u32,
    baseline_argmax: u32,
    candidate_argmax: u32,
    baseline: &[f32],
    candidate: &[f32],
) -> Result<StepMetrics> {
    if baseline.len() != candidate.len() {
        return Err(anyhow!(
            "logits length mismatch at step {step}: baseline={} candidate={}",
            baseline.len(),
            candidate.len()
        ));
    }

    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;
    let mut sum_sq_diff = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut baseline_norm_sq = 0.0_f64;
    let mut candidate_norm_sq = 0.0_f64;
    for (&b, &c) in baseline.iter().zip(candidate) {
        let diff = (b - c).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += f64::from(diff);
        sum_sq_diff += f64::from(diff * diff);
        dot += f64::from(b) * f64::from(c);
        baseline_norm_sq += f64::from(b) * f64::from(b);
        candidate_norm_sq += f64::from(c) * f64::from(c);
    }

    let len = baseline.len() as f64;
    let cosine_similarity = if baseline_norm_sq == 0.0 || candidate_norm_sq == 0.0 {
        0.0
    } else {
        (dot / (baseline_norm_sq.sqrt() * candidate_norm_sq.sqrt())) as f32
    };
    Ok(StepMetrics {
        step,
        baseline_token,
        baseline_argmax,
        candidate_argmax,
        argmax_matches: baseline_argmax == candidate_argmax,
        max_abs_diff,
        mean_abs_diff: (sum_abs_diff / len) as f32,
        rms_diff: (sum_sq_diff / len).sqrt() as f32,
        cosine_similarity,
        top5_overlap: top_k_overlap(baseline, candidate, 5),
    })
}

fn argmax(values: &[f32]) -> Result<u32> {
    let (idx, _) = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .ok_or_else(|| anyhow!("argmax called on empty logits"))?;
    Ok(idx as u32)
}

fn top_k_overlap(a: &[f32], b: &[f32], k: usize) -> usize {
    let a_top: HashSet<usize> = top_k_indices(a, k).into_iter().collect();
    top_k_indices(b, k)
        .into_iter()
        .filter(|idx| a_top.contains(idx))
        .count()
}

fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));
    indexed
        .into_iter()
        .take(k.min(values.len()))
        .map(|(idx, _)| idx)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn default_matrix_covers_all_kv_quant_configs() {
        let args = Args::parse_from([
            "ironmlx-turboquant-kv-validate",
            "--model",
            "/tmp/model",
            "--prompt-file",
            "/tmp/prompt.txt",
            "--out",
            "/tmp/out.json",
        ]);

        assert_eq!(
            args.kv_quant,
            vec![
                KvQuantArg::None,
                KvQuantArg::Turbo3,
                KvQuantArg::Turbo4,
                KvQuantArg::K3V4,
            ]
        );
    }
}
