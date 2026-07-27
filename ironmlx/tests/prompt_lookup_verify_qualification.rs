//! Real-model qualification for exact multi-token PromptLookup verification.
//!
//! The safe production path verifies each token with a sequential Q=1
//! forward. These ignored tests compare that reference against one Q>1
//! teacher-forced forward from the same prefix and an independent cache.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ironmlx::core::cache::TurboQuantKVBits;
use ironmlx::core::generate::{build_batched_append_attention_mask, build_position_ids};
use ironmlx::core::{Loader, Model, Tokenizer};
use ironmlx::models::{Gemma4Model, LlamaModel, Qwen35Model, Qwen35MoeModel, Qwen36MoeModel};
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

#[derive(Clone, Copy, Debug)]
enum QualificationCache {
    Dense,
    Paged,
    TurboQuant(TurboQuantKVBits),
}

fn configure_cache(
    cache: &mut [ironmlx::nn::LayerCache],
    mode: QualificationCache,
    batch: usize,
    cap: i32,
) -> Result<()> {
    match mode {
        QualificationCache::Dense => Ok(()),
        QualificationCache::Paged => {
            let block_size = 16_i32;
            let pages_per_row = (cap + block_size - 1) / block_size;
            let max_pages = i32::try_from(batch)
                .context("batch exceeds i32")?
                .saturating_mul(pages_per_row)
                .saturating_add(8);
            for layer in cache {
                layer.enable_paged_kv(block_size, max_pages)?;
            }
            Ok(())
        }
        QualificationCache::TurboQuant(bits) => {
            for layer in cache {
                layer.enable_turboquant(bits)?;
            }
            Ok(())
        }
    }
}

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

fn uniform_position_ids(start: i32, len: i32, batch: usize) -> Result<Array> {
    let positions = build_position_ids(start, len)?;
    if batch == 1 {
        return Ok(positions);
    }
    let batch = i32::try_from(batch).context("batch exceeds i32")?;
    mlx::ops::shape::broadcast_to_on(&positions, &[3_i32, batch, len][..], ())
        .context("broadcasting uniform position ids")
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
        let positions = uniform_position_ids(chunk_start, chunk_len, batch)?;
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
    cache_mode: QualificationCache,
) -> Result<()> {
    const TAIL_STEPS: usize = 8;
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
    let cap =
        i32::try_from(prefix_len + verify_width + TAIL_STEPS).context("cache cap exceeds i32")?;
    let batch_i32 = i32::try_from(batch).context("batch exceeds i32")?;
    let mut sequential_cache = model.make_cache(batch_i32, cap, model.cache_dtype())?;
    let mut batched_cache = model.make_cache(batch_i32, cap, model.cache_dtype())?;
    configure_cache(&mut sequential_cache, cache_mode, batch, cap)?;
    configure_cache(&mut batched_cache, cache_mode, batch, cap)?;
    prefill(
        model,
        &prefix_tokens,
        batch,
        prefix_len,
        &mut sequential_cache,
    )?;
    prefill(model, &prefix_tokens, batch, prefix_len, &mut batched_cache)?;

    let mut sequential_hidden = Vec::with_capacity(verify_width);
    let mut sequential_logits = Vec::with_capacity(verify_width);
    let mut sequential_tokens = vec![Vec::with_capacity(verify_width); batch];
    for depth in 0..verify_width {
        let step_tokens = (0..batch)
            .map(|row| verify_tokens[row * verify_width + depth])
            .collect::<Vec<_>>();
        let input = input_array(&step_tokens, batch, 1)?;
        let start = i32::try_from(prefix_len + depth).context("position exceeds i32")?;
        let positions = uniform_position_ids(start, 1, batch)?;
        let per_row_lens = vec![1_i32; batch];
        let hidden = model.forward_text_hidden(
            &input,
            &positions,
            Some(&per_row_lens),
            None,
            Some(sequential_cache.as_mut_slice()),
            StreamOrDevice::default(),
        )?;
        let logits = model.project_hidden_on(&hidden, StreamOrDevice::default())?;
        eval(&logits)?;
        for (row, token) in argmax_tokens(&logits)?.into_iter().enumerate() {
            sequential_tokens[row].push(token);
        }
        sequential_hidden.push(hidden);
        sequential_logits.push(logits);
    }

    let verify_input = input_array(&verify_tokens, batch, verify_width)?;
    let verify_width_i32 = i32::try_from(verify_width).context("verify width exceeds i32")?;
    let verify_start = i32::try_from(prefix_len).context("verify start exceeds i32")?;
    let positions = uniform_position_ids(verify_start, verify_width_i32, batch)?;
    let per_row_lens = vec![verify_width_i32; batch];
    let verify_qmm = ironmlx::nn::verify_qmm_scope();
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
    drop(verify_qmm);
    let batched_tokens = argmax_tokens(&batched_logits)?
        .chunks(verify_width)
        .map(<[u32]>::to_vec)
        .collect::<Vec<_>>();
    let mut isolated_projection_tokens = vec![Vec::with_capacity(verify_width); batch];
    for row in 0..batch {
        for depth in 0..verify_width {
            let position_hidden = mlx::ops::indexing::slice_strided(
                &hidden,
                &[row as i32, depth as i32, 0][..],
                &[
                    row as i32 + 1,
                    depth as i32 + 1,
                    hidden.shape().as_slice()[2],
                ][..],
                &[1_i32, 1, 1][..],
            )?;
            let position_logits =
                model.project_hidden_on(&position_hidden, StreamOrDevice::default())?;
            isolated_projection_tokens[row].push(argmax_tokens(&position_logits)?[0]);
        }
    }

    let mut max_logit_diff = 0.0_f32;
    let mut max_hidden_diff = 0.0_f32;
    for (depth, sequential) in sequential_logits.iter().enumerate() {
        for row in 0..batch {
            let sequential_hidden = slice_position(&sequential_hidden[depth], row as i32, 0)?;
            let batched_hidden = slice_position(&hidden, row as i32, depth as i32)?;
            max_hidden_diff =
                max_hidden_diff.max(max_abs_diff(&sequential_hidden, &batched_hidden)?);
            let sequential = slice_position(sequential, row as i32, 0)?;
            let batched = slice_position(&batched_logits, row as i32, depth as i32)?;
            max_logit_diff = max_logit_diff.max(max_abs_diff(&sequential, &batched)?);
        }
    }
    eprintln!(
        "exact verify qualification: model={} cache={cache_mode:?} batch={} prefix_len={} verify_width={} \
         sequential={sequential_tokens:?} batched={batched_tokens:?} \
         isolated_projection={isolated_projection_tokens:?} \
         max_hidden_abs={max_hidden_diff:.6} max_logit_abs={max_logit_diff:.6}",
        std::any::type_name::<M>(),
        batch,
        prefix_len,
        verify_width
    );
    anyhow::ensure!(
        sequential_tokens == batched_tokens,
        "greedy token mismatch for model={} cache={cache_mode:?} batch={batch} prefix_len={prefix_len} \
         verify_width={verify_width}: \
         sequential={sequential_tokens:?}, batched={batched_tokens:?}, \
         isolated_projection={isolated_projection_tokens:?}, \
         max_hidden_abs={max_hidden_diff:.6}, max_logit_abs={max_logit_diff:.6}",
        std::any::type_name::<M>()
    );

    let mut tail_input = sequential_tokens
        .iter()
        .map(|row| row.last().copied().context("missing last verify token"))
        .collect::<Result<Vec<_>>>()?;
    for tail_depth in 0..TAIL_STEPS {
        let input = input_array(&tail_input, batch, 1)?;
        let start = i32::try_from(prefix_len + verify_width + tail_depth)
            .context("tail position exceeds i32")?;
        let positions = uniform_position_ids(start, 1, batch)?;
        let per_row_lens = vec![1_i32; batch];
        let sequential_hidden = model.forward_text_hidden(
            &input,
            &positions,
            Some(&per_row_lens),
            None,
            Some(sequential_cache.as_mut_slice()),
            StreamOrDevice::default(),
        )?;
        let batched_hidden = model.forward_text_hidden(
            &input,
            &positions,
            Some(&per_row_lens),
            None,
            Some(batched_cache.as_mut_slice()),
            StreamOrDevice::default(),
        )?;
        let sequential_logits =
            model.project_hidden_on(&sequential_hidden, StreamOrDevice::default())?;
        let batched_logits = model.project_hidden_on(&batched_hidden, StreamOrDevice::default())?;
        eval(&sequential_logits)?;
        eval(&batched_logits)?;
        let sequential_tail = argmax_tokens(&sequential_logits)?;
        let batched_tail = argmax_tokens(&batched_logits)?;
        anyhow::ensure!(
            sequential_tail == batched_tail,
            "post-verify tail mismatch for model={} cache={cache_mode:?} B{batch} \
             C{prefix_len} Q{verify_width} tail_depth={tail_depth}: \
             sequential={sequential_tail:?}, batched={batched_tail:?}",
            std::any::type_name::<M>()
        );
        tail_input = sequential_tail;
    }
    Ok(())
}

fn qualify_ragged_case<M: Model>(
    model: &M,
    tokens: &[u32],
    prefix_len: usize,
    verify_lens: &[usize],
) -> Result<()> {
    qualify_ragged_case_with_cache(
        model,
        tokens,
        prefix_len,
        verify_lens,
        QualificationCache::Dense,
    )
}

fn qualify_ragged_case_with_cache<M: Model>(
    model: &M,
    tokens: &[u32],
    prefix_len: usize,
    verify_lens: &[usize],
    cache_mode: QualificationCache,
) -> Result<()> {
    let batch = verify_lens.len();
    let verify_width = verify_lens.iter().copied().max().unwrap_or(0);
    anyhow::ensure!(batch > 0 && verify_width > 1, "invalid ragged verify shape");
    let row_stride = prefix_len.saturating_add(verify_width);
    anyhow::ensure!(
        batch.saturating_mul(row_stride) <= tokens.len(),
        "ragged qualification token sequence is too short"
    );

    let mut prefix_tokens = Vec::with_capacity(batch.saturating_mul(prefix_len));
    let mut verify_tokens = vec![0_u32; batch.saturating_mul(verify_width)];
    for row in 0..batch {
        let row_start = row.saturating_mul(row_stride);
        prefix_tokens.extend_from_slice(&tokens[row_start..row_start + prefix_len]);
        let verify_start = row_start + prefix_len;
        verify_tokens[row * verify_width..row * verify_width + verify_lens[row]]
            .copy_from_slice(&tokens[verify_start..verify_start + verify_lens[row]]);
    }

    let cap = i32::try_from(prefix_len + verify_width + 8).context("cache cap exceeds i32")?;
    let batch_i32 = i32::try_from(batch).context("batch exceeds i32")?;
    let mut sequential_cache = model.make_cache(batch_i32, cap, model.cache_dtype())?;
    let mut batched_cache = model.make_cache(batch_i32, cap, model.cache_dtype())?;
    configure_cache(&mut sequential_cache, cache_mode, batch, cap)?;
    configure_cache(&mut batched_cache, cache_mode, batch, cap)?;
    prefill(
        model,
        &prefix_tokens,
        batch,
        prefix_len,
        &mut sequential_cache,
    )?;
    prefill(model, &prefix_tokens, batch, prefix_len, &mut batched_cache)?;

    let mut offsets = vec![i32::try_from(prefix_len).context("prefix length exceeds i32")?; batch];
    let mut sequential_tokens = vec![Vec::new(); batch];
    for depth in 0..verify_width {
        let step_tokens = (0..batch)
            .map(|row| verify_tokens[row * verify_width + depth])
            .collect::<Vec<_>>();
        let step_lens = verify_lens
            .iter()
            .map(|&len| i32::from(depth < len))
            .collect::<Vec<_>>();
        let input = input_array(&step_tokens, batch, 1)?;
        let positions = uniform_position_ids(
            i32::try_from(prefix_len + depth).context("position exceeds i32")?,
            1,
            batch,
        )?;
        let mask = build_batched_append_attention_mask(&offsets, &step_lens, 1, Dtype::Bfloat16)?;
        let hidden = model.forward_text_hidden(
            &input,
            &positions,
            Some(&step_lens),
            Some(&mask),
            Some(sequential_cache.as_mut_slice()),
            StreamOrDevice::default(),
        )?;
        let logits = model.project_hidden_on(&hidden, StreamOrDevice::default())?;
        eval(&logits)?;
        for (row, token) in argmax_tokens(&logits)?.into_iter().enumerate() {
            if step_lens[row] == 1 {
                sequential_tokens[row].push(token);
            }
            offsets[row] += step_lens[row];
        }
    }

    let input = input_array(&verify_tokens, batch, verify_width)?;
    let positions = uniform_position_ids(
        i32::try_from(prefix_len).context("verify start exceeds i32")?,
        i32::try_from(verify_width).context("verify width exceeds i32")?,
        batch,
    )?;
    let pre_offsets = vec![i32::try_from(prefix_len).context("prefix length exceeds i32")?; batch];
    let verify_lens_i32 = verify_lens
        .iter()
        .map(|&len| i32::try_from(len).context("verify length exceeds i32"))
        .collect::<Result<Vec<_>>>()?;
    let mask = build_batched_append_attention_mask(
        &pre_offsets,
        &verify_lens_i32,
        i32::try_from(verify_width).context("verify width exceeds i32")?,
        Dtype::Bfloat16,
    )?;
    let verify_qmm = ironmlx::nn::verify_qmm_scope();
    let hidden = model.forward_text_hidden(
        &input,
        &positions,
        Some(&verify_lens_i32),
        Some(&mask),
        Some(batched_cache.as_mut_slice()),
        StreamOrDevice::default(),
    )?;
    let logits = model.project_hidden_on(&hidden, StreamOrDevice::default())?;
    eval(&logits)?;
    drop(verify_qmm);
    let flat = argmax_tokens(&logits)?;
    let batched_tokens = (0..batch)
        .map(|row| flat[row * verify_width..row * verify_width + verify_lens[row]].to_vec())
        .collect::<Vec<_>>();
    eprintln!(
        "ragged exact verify qualification: model={} cache={cache_mode:?} prefix_len={} verify_lens={verify_lens:?} \
         sequential={sequential_tokens:?} batched={batched_tokens:?}",
        std::any::type_name::<M>(),
        prefix_len
    );
    anyhow::ensure!(
        sequential_tokens == batched_tokens,
        "ragged greedy token mismatch for model={} cache={cache_mode:?} prefix_len={prefix_len} \
         verify_lens={verify_lens:?}: sequential={sequential_tokens:?}, \
         batched={batched_tokens:?}",
        std::any::type_name::<M>()
    );
    Ok(())
}

fn qualify_model<M: Model>(model: &M, tokenizer: &Tokenizer) -> Result<()> {
    let force_candidate = std::env::var_os("PROMPT_LOOKUP_VERIFY_FORCE_CANDIDATE").is_some();
    let max_verify_width = std::env::var("PROMPT_LOOKUP_VERIFY_MAX_WIDTH")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("parsing PROMPT_LOOKUP_VERIFY_MAX_WIDTH")?
        .unwrap_or(5);
    let parse_axis = |name: &str, defaults: &[usize]| -> Result<Vec<usize>> {
        let Some(value) = std::env::var_os(name) else {
            return Ok(defaults.to_vec());
        };
        value
            .to_string_lossy()
            .split(',')
            .map(|item| {
                item.trim()
                    .parse::<usize>()
                    .with_context(|| format!("parsing {name}"))
            })
            .collect()
    };
    let batches = parse_axis("PROMPT_LOOKUP_VERIFY_BATCHES", &[1, 2, 4, 8])?;
    let prefix_lens = parse_axis("PROMPT_LOOKUP_VERIFY_PREFIX_LENS", &[8, 32, 64, 128, 1_024])?;
    let verify_widths = parse_axis("PROMPT_LOOKUP_VERIFY_WIDTHS", &[2, 4, 5, 8])?;
    let max_batch = batches.iter().copied().max().unwrap_or(0);
    let max_prefix_len = prefix_lens.iter().copied().max().unwrap_or(0);
    let max_requested_width = verify_widths
        .iter()
        .copied()
        .filter(|&width| width <= max_verify_width)
        .max()
        .unwrap_or(0);
    anyhow::ensure!(
        max_batch > 0 && max_prefix_len > 0 && max_requested_width > 1,
        "qualification axes must contain a positive batch/prefix and verify width greater than one"
    );
    let required_tokens =
        max_batch.saturating_mul(max_prefix_len.saturating_add(max_requested_width));
    let mut tokens = tokenizer
        .encode(QUALIFICATION_TEXT, false)
        .context("tokenizing qualification text")?;
    while tokens.len() < required_tokens {
        let copy = tokens.clone();
        tokens.extend(copy);
    }
    let mut qualified_cases = 0_usize;
    let mut candidate_failures = Vec::new();
    for batch in batches {
        for &prefix_len in &prefix_lens {
            for &verify_width in verify_widths
                .iter()
                .filter(|&&width| width <= max_verify_width)
            {
                if !force_candidate
                    && !model.supports_exact_batched_speculative_verify(
                        batch,
                        prefix_len,
                        verify_width,
                    )
                {
                    continue;
                }
                let result = qualify_case(
                    model,
                    &tokens,
                    batch,
                    prefix_len,
                    verify_width,
                    QualificationCache::Dense,
                );
                match result {
                    Ok(()) => qualified_cases += 1,
                    Err(error) if force_candidate => {
                        let failure =
                            format!("B{batch} context={prefix_len} Q{verify_width}: {error:#}");
                        eprintln!("candidate exact verify failure: {failure}");
                        candidate_failures.push(failure);
                    }
                    Err(error) => return Err(error),
                }
            }
        }
    }
    anyhow::ensure!(
        candidate_failures.is_empty(),
        "candidate exact Q>1 qualification failures:\n{}",
        candidate_failures.join("\n")
    );
    anyhow::ensure!(
        qualified_cases > 0,
        "model reported no exact Q>1 qualification cases"
    );
    Ok(())
}

fn qualify_qwen_cache_and_ragged<M: Model>(model: &M, tokenizer: &Tokenizer) -> Result<()> {
    let mut tokens = tokenizer
        .encode(QUALIFICATION_TEXT, false)
        .context("tokenizing Qwen cache qualification text")?;
    while tokens.len() < 8 * (128 + 5) {
        let copy = tokens.clone();
        tokens.extend(copy);
    }
    for cache_mode in [
        QualificationCache::Paged,
        QualificationCache::TurboQuant(TurboQuantKVBits::K3V4),
        QualificationCache::TurboQuant(TurboQuantKVBits::K4V4),
    ] {
        if model.supports_exact_batched_speculative_verify(4, 64, 5) {
            qualify_case(model, &tokens, 4, 64, 5, cache_mode)?;
        } else if model.supports_exact_batched_speculative_verify(8, 64, 2) {
            for &batch in &[1_usize, 4, 8] {
                qualify_case(model, &tokens, batch, 64, 2, cache_mode)?;
            }
        } else {
            for &(batch, verify_width) in &[(1_usize, 5_usize), (2, 4), (4, 2)] {
                anyhow::ensure!(
                    model.supports_exact_batched_speculative_verify(batch, 64, verify_width),
                    "missing expected Affine8 exact qualification for B{batch} Q{verify_width}"
                );
                qualify_case(model, &tokens, batch, 64, verify_width, cache_mode)?;
            }
        }
    }
    if model.supports_exact_batched_speculative_verify(8, 128, 5) {
        qualify_ragged_case(model, &tokens, 64, &[5, 4, 2, 0])?;
        qualify_ragged_case(model, &tokens, 128, &[5, 1, 4, 0, 3, 2, 5, 0])
    } else if model.supports_exact_batched_speculative_verify(8, 128, 2) {
        qualify_ragged_case(model, &tokens, 64, &[2, 1, 2, 0])?;
        qualify_ragged_case(model, &tokens, 128, &[2, 1, 2, 0, 2, 1, 2, 0])
    } else {
        qualify_ragged_case(model, &tokens, 64, &[4, 2])?;
        qualify_ragged_case(model, &tokens, 128, &[2, 1, 2, 0])
    }
}

fn qualify_gemma_cache_and_ragged<M: Model>(
    model: &M,
    tokenizer: &Tokenizer,
    force_turboquant_candidate: bool,
) -> Result<()> {
    let mut tokens = tokenizer
        .encode(QUALIFICATION_TEXT, false)
        .context("tokenizing Gemma4 cache qualification text")?;
    while tokens.len() < 2 * (1_024 + 5) {
        let copy = tokens.clone();
        tokens.extend(copy);
    }
    let batch = if model.supports_exact_batched_speculative_verify(4, 64, 5) {
        4
    } else {
        anyhow::ensure!(
            model.supports_exact_batched_speculative_verify(2, 64, 5),
            "missing expected Gemma4 exact qualification for B2 Q5"
        );
        2
    };
    let cache_modes = if force_turboquant_candidate {
        vec![
            QualificationCache::TurboQuant(TurboQuantKVBits::K3V4),
            QualificationCache::TurboQuant(TurboQuantKVBits::K4V4),
        ]
    } else {
        vec![
            QualificationCache::Paged,
            QualificationCache::TurboQuant(TurboQuantKVBits::K3V4),
            QualificationCache::TurboQuant(TurboQuantKVBits::K4V4),
        ]
    };
    for cache_mode in cache_modes {
        let kv_bits = match cache_mode {
            QualificationCache::TurboQuant(bits) => Some(bits),
            QualificationCache::Dense | QualificationCache::Paged => None,
        };
        if batch == 2 && kv_bits.is_some() {
            for case_batch in [1_usize, 2] {
                for context_tokens in [64_usize, 1_024] {
                    for verify_width in [2_usize, 4, 5] {
                        anyhow::ensure!(
                            force_turboquant_candidate
                                || model.supports_exact_batched_speculative_verify_for_kv_cache(
                                    case_batch,
                                    context_tokens,
                                    verify_width,
                                    kv_bits,
                                ),
                            "missing Gemma4 Affine8 MoE TurboQuant exact qualification for \
                             cache={cache_mode:?} B{case_batch} Q{verify_width} C{context_tokens}"
                        );
                        qualify_case(
                            model,
                            &tokens,
                            case_batch,
                            context_tokens,
                            verify_width,
                            cache_mode,
                        )?;
                    }
                }
            }
            qualify_ragged_case_with_cache(model, &tokens, 64, &[5, 2], cache_mode)?;
            qualify_ragged_case_with_cache(model, &tokens, 128, &[5, 1], cache_mode)?;
        } else if force_turboquant_candidate
            || model.supports_exact_batched_speculative_verify_for_kv_cache(batch, 64, 5, kv_bits)
        {
            qualify_case(model, &tokens, batch, 64, 5, cache_mode)?;
        } else {
            anyhow::ensure!(
                kv_bits.is_some(),
                "unexpected Gemma4 cache qualification rejection for {cache_mode:?}"
            );
            if model.supports_exact_batched_speculative_verify_for_kv_cache(batch, 64, 2, kv_bits) {
                qualify_case(model, &tokens, batch, 64, 2, cache_mode)?;
            }
        }
    }
    if batch == 4 {
        qualify_ragged_case(model, &tokens, 64, &[5, 4, 2, 0])?;
        qualify_ragged_case(model, &tokens, 128, &[5, 1, 4, 0, 3, 2, 5, 0])
    } else if !force_turboquant_candidate {
        qualify_ragged_case(model, &tokens, 64, &[5, 2])?;
        qualify_ragged_case(model, &tokens, 128, &[5, 1])
    } else {
        Ok(())
    }
}

fn qualify_identical_row_decode<M: Model>(
    model: &M,
    tokenizer: &Tokenizer,
    prefix_len: usize,
    decode_steps: usize,
) -> Result<()> {
    const BATCH: usize = 2;

    let prompt_path = std::env::var_os("QWEN35_MOE_ORDINARY_PROMPT");
    let mut source = if let Some(path) = prompt_path.as_ref() {
        let content = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "reading ordinary prompt from {}",
                PathBuf::from(path).display()
            )
        })?;
        let prompt = tokenizer.apply_chat_template(
            &[ironmlx::core::Message {
                role: "user".to_owned(),
                content,
            }],
            true,
            Some(&serde_json::json!({"enable_thinking": false})),
        )?;
        tokenizer
            .encode(&prompt, false)
            .context("tokenizing ordinary chat prompt")?
    } else {
        tokenizer
            .encode(QUALIFICATION_TEXT, false)
            .context("tokenizing identical-row qualification text")?
    };
    let (prefix_len, decode_steps) = if prompt_path.is_some() {
        (source.len(), 0)
    } else {
        while source.len() < prefix_len.saturating_add(decode_steps) {
            let copy = source.clone();
            source.extend(copy);
        }
        (
            prefix_len.min(source.len().saturating_sub(decode_steps)),
            decode_steps,
        )
    };
    let prefix = &source[..prefix_len];
    let mut prefix_tokens = Vec::with_capacity(BATCH.saturating_mul(prefix_len));
    for _ in 0..BATCH {
        prefix_tokens.extend_from_slice(prefix);
    }

    let cap = i32::try_from(prefix_len.saturating_add(decode_steps).saturating_add(8))
        .context("identical-row cache cap exceeds i32")?;
    let mut cache = model.make_cache(BATCH as i32, cap, model.cache_dtype())?;
    let input = input_array(&prefix_tokens, BATCH, prefix_len)?;
    let prefix_len_i32 = i32::try_from(prefix_len).context("prefix length exceeds i32")?;
    let per_row_lens = vec![prefix_len_i32; BATCH];
    let positions =
        ironmlx::core::generate::build_position_ids_batched(&per_row_lens, prefix_len_i32)?;
    let first_logits = model.batched_prefill_causal(
        &input,
        &positions,
        &per_row_lens,
        Some(cache.as_mut_slice()),
        StreamOrDevice::default(),
    )?;
    eval(&first_logits)?;
    let first_row0 = slice_position(&first_logits, 0, 0)?;
    let first_row1 = slice_position(&first_logits, 1, 0)?;
    let first_logit_diff = max_abs_diff(&first_row0, &first_row1)?;
    let first_greedy = argmax_tokens(&first_logits)?;
    eprintln!(
        "identical-row ordinary prefill: model={} prefix_len={prefix_len} \
         logit_abs={first_logit_diff:.6} greedy={first_greedy:?}",
        std::any::type_name::<M>()
    );
    anyhow::ensure!(
        first_logit_diff == 0.0 && first_greedy[0] == first_greedy[1],
        "identical ordinary B2 prefill rows diverged for model={} prefix_len={prefix_len}: \
         logit_abs={first_logit_diff:.6} greedy={first_greedy:?}",
        std::any::type_name::<M>()
    );

    for depth in 0..decode_steps {
        let token = source[prefix_len + depth];
        let input = input_array(&[token, token], BATCH, 1)?;
        let position = i32::try_from(prefix_len + depth).context("decode position exceeds i32")?;
        let positions = uniform_position_ids(position, 1, BATCH)?;
        let per_row_lens = vec![1_i32; BATCH];
        let hidden = model.forward_text_hidden(
            &input,
            &positions,
            Some(&per_row_lens),
            None,
            Some(cache.as_mut_slice()),
            StreamOrDevice::default(),
        )?;
        let logits = model.project_hidden_on(&hidden, StreamOrDevice::default())?;
        eval(&logits)?;

        let row0_hidden = slice_position(&hidden, 0, 0)?;
        let row1_hidden = slice_position(&hidden, 1, 0)?;
        let row0_logits = slice_position(&logits, 0, 0)?;
        let row1_logits = slice_position(&logits, 1, 0)?;
        let hidden_diff = max_abs_diff(&row0_hidden, &row1_hidden)?;
        let logit_diff = max_abs_diff(&row0_logits, &row1_logits)?;
        let greedy = argmax_tokens(&logits)?;
        eprintln!(
            "identical-row ordinary decode: model={} prefix_len={prefix_len} depth={depth} \
             hidden_abs={hidden_diff:.6} logit_abs={logit_diff:.6} greedy={greedy:?}",
            std::any::type_name::<M>()
        );
        anyhow::ensure!(
            hidden_diff == 0.0 && logit_diff == 0.0 && greedy[0] == greedy[1],
            "identical ordinary B2 rows diverged for model={} prefix_len={prefix_len} \
             depth={depth}: hidden_abs={hidden_diff:.6} logit_abs={logit_diff:.6} \
             greedy={greedy:?}",
            std::any::type_name::<M>()
        );
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
    let quant_bits = loader.quant_meta().map(|quant| quant.bits);
    if quant_bits == Some(8) {
        anyhow::ensure!(
            model.supports_exact_batched_speculative_verify(1, 4_096, 5)
                && model.supports_exact_batched_speculative_verify(2, 4_096, 4)
                && model.supports_exact_batched_speculative_verify(4, 4_096, 2),
            "Qwen3.5 Dense Affine8 checkpoint is missing its exact qualification staircase"
        );
        anyhow::ensure!(
            !model.supports_exact_batched_speculative_verify(2, 4_096, 5)
                && !model.supports_exact_batched_speculative_verify(4, 4_096, 3)
                && !model.supports_exact_batched_speculative_verify(8, 4_096, 2),
            "Qwen3.5 Dense Affine8 checkpoint exceeded its qualified exact shape"
        );
    } else if matches!(quant_bits, Some(5 | 6)) {
        anyhow::ensure!(
            model.supports_exact_batched_speculative_verify(1, 4_096, 5)
                && model.supports_exact_batched_speculative_verify(2, 4_096, 4)
                && model.supports_exact_batched_speculative_verify(4, 4_096, 2)
                && !model.supports_exact_batched_speculative_verify(2, 4_096, 5)
                && !model.supports_exact_batched_speculative_verify(4, 4_096, 4)
                && !model.supports_exact_batched_speculative_verify(8, 4_096, 2)
                && !model.supports_exact_batched_speculative_verify(1, 4_097, 5),
            "Qwen3.5 Dense Affine5/6 checkpoint exceeded its exact qualification staircase"
        );
    }
    qualify_model(&model, &tokenizer)?;
    qualify_qwen_cache_and_ragged(&model, &tokenizer)
}

#[test]
#[ignore = "requires a real Qwen3.5 MoE checkpoint and Apple Silicon"]
#[serial(mlx_metal)]
fn qwen35_moe_qgt1_matches_sequential_verify() -> Result<()> {
    let Some(model_dir) = snapshot_from_env_or_cache(
        "PROMPT_LOOKUP_VERIFY_QWEN35_MOE_MODEL",
        "models--mlx-community--Qwen3.5-35B-A3B-4bit",
    ) else {
        eprintln!("skip: no Qwen3.5 MoE qualification checkpoint");
        return Ok(());
    };
    let loader = Loader::open(&model_dir).context("opening Qwen3.5 MoE checkpoint")?;
    let tokenizer = load_tokenizer(&loader, &model_dir)?;
    let model = Qwen35MoeModel::from_loader(&loader).context("loading Qwen3.5 MoE model")?;
    let quant_bits = loader.quant_meta().map(|quant| quant.bits);
    if quant_bits == Some(8) {
        anyhow::ensure!(
            model.supports_exact_batched_speculative_verify(8, 1_024, 2)
                && !model.supports_exact_batched_speculative_verify(8, 1_025, 2)
                && !model.supports_exact_batched_speculative_verify(1, 1_024, 4),
            "Qwen3.5 MoE Affine8 checkpoint exceeded its exact Q2 qualification"
        );
    } else if matches!(quant_bits, Some(5 | 6)) {
        anyhow::ensure!(
            model.supports_exact_batched_speculative_verify(8, 1_024, 5)
                && !model.supports_exact_batched_speculative_verify(8, 1_025, 5)
                && !model.supports_exact_batched_speculative_verify(8, 1_024, 6),
            "Qwen3.5 MoE Affine5/6 checkpoint exceeded its exact qualification"
        );
    } else {
        anyhow::ensure!(
            model.supports_exact_batched_speculative_verify(8, 4_096, 5),
            "Qwen3.5 MoE checkpoint is not exact-verify precision qualified"
        );
    }
    qualify_model(&model, &tokenizer)?;
    qualify_qwen_cache_and_ragged(&model, &tokenizer)
}

#[test]
#[ignore = "requires a real Qwen3.5 MoE checkpoint and Apple Silicon"]
#[serial(mlx_metal)]
fn qwen35_moe_identical_ordinary_b2_rows_match() -> Result<()> {
    let Some(model_dir) = snapshot_from_env_or_cache(
        "PROMPT_LOOKUP_VERIFY_QWEN35_MOE_MODEL",
        "models--mlx-community--Qwen3.5-35B-A3B-4bit",
    ) else {
        eprintln!("skip: no Qwen3.5 MoE qualification checkpoint");
        return Ok(());
    };
    let loader = Loader::open(&model_dir).context("opening Qwen3.5 MoE checkpoint")?;
    let tokenizer = load_tokenizer(&loader, &model_dir)?;
    let model = Qwen35MoeModel::from_loader(&loader).context("loading Qwen3.5 MoE model")?;
    qualify_identical_row_decode(&model, &tokenizer, 2_713, 16)
}

#[test]
#[ignore = "requires a real Qwen3.6 MoE checkpoint and Apple Silicon"]
#[serial(mlx_metal)]
fn qwen36_moe_qgt1_matches_sequential_verify() -> Result<()> {
    let Some(model_dir) = snapshot_from_env_or_cache(
        "PROMPT_LOOKUP_VERIFY_QWEN36_MOE_MODEL",
        "models--mlx-community--Qwen3.6-35B-A3B-4bit",
    ) else {
        eprintln!("skip: no Qwen3.6 MoE qualification checkpoint");
        return Ok(());
    };
    let loader = Loader::open(&model_dir).context("opening Qwen3.6 MoE checkpoint")?;
    let tokenizer = load_tokenizer(&loader, &model_dir)?;
    let model = Qwen36MoeModel::from_loader(&loader).context("loading Qwen3.6 MoE model")?;
    let force_candidate = std::env::var_os("PROMPT_LOOKUP_VERIFY_FORCE_CANDIDATE").is_some();
    let quant_bits = loader.quant_meta().map(|quant| quant.bits);
    if !force_candidate && quant_bits == Some(8) {
        anyhow::ensure!(
            model.supports_exact_batched_speculative_verify(8, 1_024, 2)
                && !model.supports_exact_batched_speculative_verify(8, 1_025, 2)
                && !model.supports_exact_batched_speculative_verify(1, 1_024, 4),
            "Qwen3.6 MoE Affine8 checkpoint exceeded its exact Q2 qualification"
        );
    } else if !force_candidate && matches!(quant_bits, Some(5 | 6)) {
        anyhow::ensure!(
            model.supports_exact_batched_speculative_verify(8, 1_024, 5)
                && !model.supports_exact_batched_speculative_verify(8, 1_025, 5)
                && !model.supports_exact_batched_speculative_verify(8, 1_024, 6),
            "Qwen3.6 MoE Affine5/6 checkpoint exceeded its exact qualification"
        );
    } else if !force_candidate {
        anyhow::ensure!(
            model.supports_exact_batched_speculative_verify(8, 4_096, 5),
            "Qwen3.6 MoE checkpoint is not exact-verify precision qualified"
        );
    }
    qualify_model(&model, &tokenizer)?;
    if force_candidate {
        return Ok(());
    }
    qualify_qwen_cache_and_ragged(&model, &tokenizer)
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
    let force_candidate = std::env::var_os("PROMPT_LOOKUP_VERIFY_FORCE_CANDIDATE").is_some();
    let force_turboquant_candidate =
        std::env::var_os("PROMPT_LOOKUP_VERIFY_FORCE_TURBOQUANT_CANDIDATE").is_some();
    let quant_bits = loader.quant_meta().map(|quant| quant.bits);
    let is_moe = loader
        .config_raw_value()
        .pointer("/text_config/enable_moe_block")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if !force_candidate && quant_bits == Some(8) {
        if is_moe {
            anyhow::ensure!(
                model.supports_exact_batched_speculative_verify(2, 1_024, 5)
                    && !model.supports_exact_batched_speculative_verify(4, 1_024, 2)
                    && !model.supports_exact_batched_speculative_verify(2, 1_025, 5),
                "Gemma4 MoE Affine8 checkpoint exceeded its B1-B2 exact qualification"
            );
        } else {
            anyhow::ensure!(
                model.supports_exact_batched_speculative_verify(8, 1_024, 5)
                    && !model.supports_exact_batched_speculative_verify(8, 1_025, 5)
                    && !model.supports_exact_batched_speculative_verify(8, 1_024, 6),
                "Gemma4 Dense Affine8 checkpoint exceeded its exact qualification"
            );
        }
    } else if !force_candidate && matches!(quant_bits, Some(5 | 6)) {
        if is_moe {
            anyhow::ensure!(
                !model.supports_exact_batched_speculative_verify(1, 1_024, 2),
                "Gemma4 MoE Affine5/6 must remain fail closed without a qualification model"
            );
        } else {
            anyhow::ensure!(
                model.supports_exact_batched_speculative_verify(8, 1_024, 5)
                    && !model.supports_exact_batched_speculative_verify(8, 1_025, 5)
                    && !model.supports_exact_batched_speculative_verify(8, 1_024, 6)
                    && model
                        .supports_exact_batched_speculative_verify_for_kv_cache(8, 1_024, 5, None,)
                    && !model.supports_exact_batched_speculative_verify_for_kv_cache(
                        8,
                        1_024,
                        2,
                        Some(TurboQuantKVBits::K3V4),
                    )
                    && !model.supports_exact_batched_speculative_verify_for_kv_cache(
                        8,
                        1_024,
                        2,
                        Some(TurboQuantKVBits::K4V4),
                    ),
                "Gemma4 Dense Affine5/6 checkpoint exceeded its exact qualification"
            );
        }
    }
    if !force_turboquant_candidate {
        qualify_model(&model, &tokenizer)?;
    }
    if force_candidate {
        return Ok(());
    }
    qualify_gemma_cache_and_ragged(&model, &tokenizer, force_turboquant_candidate)
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
    qualify_model(&model, &tokenizer)?;

    let mut tokens = tokenizer
        .encode(QUALIFICATION_TEXT, false)
        .context("tokenizing ragged qualification text")?;
    while tokens.len() < 8 * (128 + 5) {
        let copy = tokens.clone();
        tokens.extend(copy);
    }
    for cache_mode in [
        QualificationCache::Paged,
        QualificationCache::TurboQuant(TurboQuantKVBits::K3V4),
        QualificationCache::TurboQuant(TurboQuantKVBits::K4V4),
    ] {
        qualify_case(&model, &tokens, 4, 64, 5, cache_mode)?;
    }
    qualify_ragged_case(&model, &tokens, 64, &[5, 4, 2, 0])?;
    qualify_ragged_case(&model, &tokens, 128, &[5, 1, 4, 0, 3, 2, 5, 0])
}
