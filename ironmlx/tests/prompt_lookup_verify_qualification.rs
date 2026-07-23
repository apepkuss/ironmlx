//! Real-model qualification for exact multi-token PromptLookup verification.
//!
//! The safe production path verifies each token with a sequential Q=1
//! forward. These ignored tests compare that reference against one Q>1
//! teacher-forced forward from the same prefix and an independent cache.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ironmlx::core::generate::build_position_ids;
use ironmlx::core::{Loader, Model, Tokenizer};
use ironmlx::models::{Gemma4Model, LlamaModel, Qwen35Model};
use mlx::{Array, Dtype, StreamOrDevice};
use serial_test::serial;

const QUALIFICATION_TEXT: &str = "\
IronMLX verifies copied prompt continuations against the target model. \
This qualification sentence intentionally repeats: exact batched verification \
must match sequential verification token by token. \
IronMLX verifies copied prompt continuations against the target model. \
This qualification sentence intentionally repeats: exact batched verification \
must match sequential verification token by token. \
The final clause supplies enough continuation tokens for every verify width.";

fn snapshot_from_env_or_cache(env_name: &str, cache_repo: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(env_name) {
        let path = PathBuf::from(path);
        return path.is_dir().then_some(path);
    }
    let snapshots = dirs::home_dir()?
        .join(".ironmlx")
        .join("models")
        .join(cache_repo)
        .join("snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
}

fn input_array(tokens: &[u32], batch: usize, sequence: usize) -> Result<Array> {
    anyhow::ensure!(
        tokens.len() == batch.saturating_mul(sequence),
        "token count {} != batch {batch} * sequence {sequence}",
        tokens.len()
    );
    let batch = i32::try_from(batch).context("batch exceeds i32")?;
    let sequence = i32::try_from(sequence).context("sequence length exceeds i32")?;
    (tokens, &[batch, sequence][..])
        .try_into()
        .context("building token input")
}

fn eval(array: &Array) -> Result<()> {
    mlx::transforms::eval(&[array]).context("evaluating MLX graph")
}

fn argmax_tokens(logits: &Array) -> Result<Vec<u32>> {
    mlx::ops::reduction::argmax(logits, -1, false)?
        .to_vec()
        .context("materializing greedy token ids")
}

fn max_abs_diff(a: &Array, b: &Array) -> Result<f32> {
    let a = mlx::ops::cast::astype(a, Dtype::Float32)?
        .to_vec::<f32>()
        .context("materializing sequential logits")?;
    let b = mlx::ops::cast::astype(b, Dtype::Float32)?
        .to_vec::<f32>()
        .context("materializing batched logits")?;
    anyhow::ensure!(a.len() == b.len(), "logit element count mismatch");
    Ok(a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max))
}

fn slice_position(logits: &Array, row: i32, position: i32) -> Result<Array> {
    let shape = logits.shape();
    let shape = shape.as_slice();
    anyhow::ensure!(
        shape.len() == 3 && row >= 0 && row < shape[0],
        "expected logits [B,Q,V] containing row {row}, got {shape:?}"
    );
    mlx::ops::indexing::slice_strided(
        logits,
        &[row, position, 0][..],
        &[row + 1, position + 1, shape[2]][..],
        &[1_i32, 1, 1][..],
    )
    .context("slicing verify position")
}

fn prefill<M: Model>(
    model: &M,
    tokens: &[u32],
    batch: usize,
    prefix_len: usize,
    cache: &mut [ironmlx::nn::LayerCache],
) -> Result<()> {
    const CHUNK_SIZE: usize = 256;
    for chunk_start in (0..prefix_len).step_by(CHUNK_SIZE) {
        let chunk_len = CHUNK_SIZE.min(prefix_len - chunk_start);
        let mut chunk_tokens = Vec::with_capacity(batch.saturating_mul(chunk_len));
        for row in 0..batch {
            let row_start = row.saturating_mul(prefix_len).saturating_add(chunk_start);
            chunk_tokens.extend_from_slice(&tokens[row_start..row_start + chunk_len]);
        }
        let input = input_array(&chunk_tokens, batch, chunk_len)?;
        let chunk_start = i32::try_from(chunk_start).context("chunk start exceeds i32")?;
        let chunk_len = i32::try_from(chunk_len).context("chunk length exceeds i32")?;
        let positions = build_position_ids(chunk_start, chunk_len)?;
        let per_row_lens = vec![chunk_len; batch];
        let hidden = model.forward_text_hidden(
            &input,
            &positions,
            Some(&per_row_lens),
            None,
            Some(cache),
            StreamOrDevice::default(),
        )?;
        eval(&hidden)?;
    }
    Ok(())
}

fn qualify_case<M: Model>(
    model: &M,
    tokens: &[u32],
    batch: usize,
    prefix_len: usize,
    verify_width: usize,
) -> Result<()> {
    let row_stride = prefix_len.saturating_add(verify_width);
    anyhow::ensure!(
        batch.saturating_mul(row_stride) <= tokens.len(),
        "qualification token sequence is too short"
    );
    let mut prefix_tokens = Vec::with_capacity(batch.saturating_mul(prefix_len));
    let mut verify_tokens = Vec::with_capacity(batch.saturating_mul(verify_width));
    for row in 0..batch {
        let row_start = row.saturating_mul(row_stride);
        prefix_tokens.extend_from_slice(&tokens[row_start..row_start + prefix_len]);
        verify_tokens.extend_from_slice(
            &tokens[row_start + prefix_len..row_start + prefix_len + verify_width],
        );
    }
    let cap = i32::try_from(prefix_len + verify_width + 8).context("cache cap exceeds i32")?;
    let batch_i32 = i32::try_from(batch).context("batch exceeds i32")?;
    let mut sequential_cache = model.make_cache(batch_i32, cap, model.cache_dtype())?;
    let mut batched_cache = model.make_cache(batch_i32, cap, model.cache_dtype())?;
    prefill(
        model,
        &prefix_tokens,
        batch,
        prefix_len,
        &mut sequential_cache,
    )?;
    prefill(model, &prefix_tokens, batch, prefix_len, &mut batched_cache)?;

    let mut sequential_logits = Vec::with_capacity(verify_width);
    let mut sequential_tokens = vec![Vec::with_capacity(verify_width); batch];
    for depth in 0..verify_width {
        let step_tokens = (0..batch)
            .map(|row| verify_tokens[row * verify_width + depth])
            .collect::<Vec<_>>();
        let input = input_array(&step_tokens, batch, 1)?;
        let start = i32::try_from(prefix_len + depth).context("position exceeds i32")?;
        let positions = build_position_ids(start, 1)?;
        let per_row_lens = vec![1_i32; batch];
        let logits = model.forward_on(
            &input,
            &positions,
            Some(&per_row_lens),
            None,
            Some(sequential_cache.as_mut_slice()),
            StreamOrDevice::default(),
        )?;
        eval(&logits)?;
        for (row, token) in argmax_tokens(&logits)?.into_iter().enumerate() {
            sequential_tokens[row].push(token);
        }
        sequential_logits.push(logits);
    }

    let verify_input = input_array(&verify_tokens, batch, verify_width)?;
    let verify_width_i32 = i32::try_from(verify_width).context("verify width exceeds i32")?;
    let verify_start = i32::try_from(prefix_len).context("verify start exceeds i32")?;
    let positions = build_position_ids(verify_start, verify_width_i32)?;
    let per_row_lens = vec![verify_width_i32; batch];
    let hidden = model.forward_text_hidden(
        &verify_input,
        &positions,
        Some(&per_row_lens),
        None,
        Some(batched_cache.as_mut_slice()),
        StreamOrDevice::default(),
    )?;
    let batched_logits = model.project_hidden_on(&hidden, StreamOrDevice::default())?;
    eval(&batched_logits)?;
    let batched_tokens = argmax_tokens(&batched_logits)?
        .chunks(verify_width)
        .map(<[u32]>::to_vec)
        .collect::<Vec<_>>();

    let mut max_logit_diff = 0.0_f32;
    for (depth, sequential) in sequential_logits.iter().enumerate() {
        for row in 0..batch {
            let sequential = slice_position(sequential, row as i32, 0)?;
            let batched = slice_position(&batched_logits, row as i32, depth as i32)?;
            max_logit_diff = max_logit_diff.max(max_abs_diff(&sequential, &batched)?);
        }
    }
    eprintln!(
        "exact verify qualification: model={} batch={} prefix_len={} verify_width={} \
         sequential={sequential_tokens:?} batched={batched_tokens:?} max_abs={max_logit_diff:.6}",
        std::any::type_name::<M>(),
        batch,
        prefix_len,
        verify_width
    );
    anyhow::ensure!(
        sequential_tokens == batched_tokens,
        "greedy token mismatch for model={} batch={batch} prefix_len={prefix_len} \
         verify_width={verify_width}: \
         sequential={sequential_tokens:?}, batched={batched_tokens:?}, max_abs={max_logit_diff:.6}",
        std::any::type_name::<M>()
    );
    Ok(())
}

fn qualify_model<M: Model>(model: &M, tokenizer: &Tokenizer) -> Result<()> {
    let mut tokens = tokenizer
        .encode(QUALIFICATION_TEXT, false)
        .context("tokenizing qualification text")?;
    let max_verify_width = std::env::var("PROMPT_LOOKUP_VERIFY_MAX_WIDTH")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("parsing PROMPT_LOOKUP_VERIFY_MAX_WIDTH")?
        .unwrap_or(8);
    while tokens.len() < 8 * (1_024 + 8) {
        let copy = tokens.clone();
        tokens.extend(copy);
    }
    for batch in [1_usize, 2, 4, 8] {
        for prefix_len in [8_usize, 32, 64, 128, 1_024] {
            for verify_width in [2_usize, 4, 5, 8]
                .into_iter()
                .filter(|&width| width <= max_verify_width)
            {
                qualify_case(model, &tokens, batch, prefix_len, verify_width)?;
            }
        }
    }
    Ok(())
}

fn load_tokenizer(loader: &Loader, model_dir: &Path) -> Result<Tokenizer> {
    Tokenizer::from_loader(loader)
        .with_context(|| format!("loading tokenizer from {}", model_dir.display()))
}

#[test]
#[ignore = "requires a real Qwen3.5 checkpoint and Apple Silicon"]
#[serial(mlx_metal)]
fn qwen35_dense_qgt1_matches_sequential_verify() -> Result<()> {
    let Some(model_dir) = snapshot_from_env_or_cache(
        "PROMPT_LOOKUP_VERIFY_QWEN35_MODEL",
        "models--mlx-community--Qwen3.5-2B-4bit",
    ) else {
        eprintln!("skip: no Qwen3.5 qualification checkpoint");
        return Ok(());
    };
    let loader = Loader::open(&model_dir).context("opening Qwen3.5 checkpoint")?;
    let tokenizer = load_tokenizer(&loader, &model_dir)?;
    let model = Qwen35Model::from_loader(&loader).context("loading Qwen3.5 model")?;
    qualify_model(&model, &tokenizer)
}

#[test]
#[ignore = "requires a real Gemma4 checkpoint and Apple Silicon"]
#[serial(mlx_metal)]
fn gemma4_qgt1_matches_sequential_verify() -> Result<()> {
    let Some(model_dir) = snapshot_from_env_or_cache(
        "PROMPT_LOOKUP_VERIFY_GEMMA4_MODEL",
        "models--mlx-community--gemma-4-12B-it-4bit",
    ) else {
        eprintln!("skip: no Gemma4 qualification checkpoint");
        return Ok(());
    };
    let loader = Loader::open(&model_dir).context("opening Gemma4 checkpoint")?;
    let tokenizer = load_tokenizer(&loader, &model_dir)?;
    let model = Gemma4Model::from_loader(&loader).context("loading Gemma4 model")?;
    qualify_model(&model, &tokenizer)
}

#[test]
#[ignore = "requires a real Llama-family checkpoint and Apple Silicon"]
#[serial(mlx_metal)]
fn llama_qgt1_matches_sequential_verify() -> Result<()> {
    let Some(model_dir) = snapshot_from_env_or_cache(
        "PROMPT_LOOKUP_VERIFY_LLAMA_MODEL",
        "models--mlx-community--MiniCPM5-1B-8bit",
    ) else {
        eprintln!("skip: no Llama-family qualification checkpoint");
        return Ok(());
    };
    let loader = Loader::open(&model_dir).context("opening Llama-family checkpoint")?;
    let tokenizer = load_tokenizer(&loader, &model_dir)?;
    let model = LlamaModel::from_loader(&loader).context("loading Llama-family model")?;
    qualify_model(&model, &tokenizer)
}
