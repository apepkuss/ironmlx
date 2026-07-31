//! Real-checkpoint qualification gates for MiniCPM-V-4.6 PromptLookup verify.
//!
//! Point `MINICPMV46_PROMPT_LOOKUP_MODEL` at one BF16 or qualified affine
//! checkpoint and run this ignored test once per checkpoint.

use std::path::PathBuf;

use anyhow::{Context, Result};
use ironmlx::core::cache::TurboQuantKVBits;
use ironmlx::core::generate::build_position_ids;
use ironmlx::core::{Loader, Model, Tokenizer};
use ironmlx::models::MiniCpmV46Model;
use mlx::{Array, StreamOrDevice};
use serial_test::serial;

const MODEL_ENV: &str = "MINICPMV46_PROMPT_LOOKUP_MODEL";
const QUALIFICATION_TEXT: &str = "\
IronMLX checks request-local copied continuations one target token at a time. \
The same sentence is repeated so the PromptLookup index has a deterministic \
candidate. IronMLX checks request-local copied continuations one target token \
at a time. The final clause supplies enough teacher-forced verification tokens.";

fn input_array(tokens: &[u32], batch: usize, sequence: usize) -> Result<Array> {
    anyhow::ensure!(
        tokens.len() == batch.saturating_mul(sequence),
        "token count {} != batch {batch} * sequence {sequence}",
        tokens.len()
    );
    (
        tokens,
        &[
            i32::try_from(batch).context("batch exceeds i32")?,
            i32::try_from(sequence).context("sequence exceeds i32")?,
        ][..],
    )
        .try_into()
        .context("building token input")
}

fn uniform_position_ids(start: i32, len: i32, batch: usize) -> Result<Array> {
    let positions = build_position_ids(start, len)?;
    if batch == 1 {
        return Ok(positions);
    }
    mlx::ops::shape::broadcast_to_on(
        &positions,
        &[
            3_i32,
            i32::try_from(batch).context("batch exceeds i32")?,
            len,
        ][..],
        (),
    )
    .context("broadcasting MiniCPM-V-4.6 position ids")
}

fn argmax_tokens(logits: &Array) -> Result<Vec<u32>> {
    mlx::ops::reduction::argmax(logits, -1, false)?
        .to_vec()
        .context("materializing greedy token ids")
}

#[test]
#[ignore = "requires a real MiniCPM-V-4.6 checkpoint and Apple Silicon"]
#[serial(mlx_metal)]
fn minicpmv46_prompt_lookup_profiles_are_strictly_scoped() -> Result<()> {
    let Some(model_dir) = std::env::var_os(MODEL_ENV).map(PathBuf::from) else {
        eprintln!("skip: set {MODEL_ENV} to a MiniCPM-V-4.6 checkpoint");
        return Ok(());
    };
    let loader = Loader::open(&model_dir)
        .with_context(|| format!("opening MiniCPM-V-4.6 checkpoint {}", model_dir.display()))?;
    let quant_bits = loader.quant_meta().map(|quant| quant.bits);
    let model =
        MiniCpmV46Model::from_loader(&loader).context("loading MiniCPM-V-4.6 checkpoint")?;

    if quant_bits == Some(6) {
        anyhow::ensure!(
            !model.supports_sequential_prompt_lookup_verify(1, 128, 2)
                && !model.supports_exact_batched_speculative_verify(1, 128, 2),
            "unqualified MiniCPM-V-4.6 Affine6 checkpoint must fail closed"
        );
        return Ok(());
    }
    let max_context = match quant_bits {
        None | Some(5 | 8) => 4_096,
        Some(4) => 1_024,
        other => {
            anyhow::bail!("checkpoint has unsupported MiniCPM-V-4.6 quantization profile {other:?}")
        }
    };
    for batch in [1_usize, 2, 4, 8] {
        for verify_width in [2_usize, 5] {
            anyhow::ensure!(
                model.supports_sequential_prompt_lookup_verify(batch, max_context, verify_width,),
                "missing Sequential Q1 qualification for bits={quant_bits:?} \
                 B{batch} C{max_context} Q{verify_width}"
            );
            anyhow::ensure!(
                model.supports_exact_batched_speculative_verify(batch, max_context, verify_width,),
                "missing exact Q>1 qualification for bits={quant_bits:?} \
                 B{batch} C{max_context} Q{verify_width}"
            );
        }
    }
    anyhow::ensure!(
        !model.supports_sequential_prompt_lookup_verify(1, max_context + 1, 2)
            && !model.supports_sequential_prompt_lookup_verify(0, 128, 2)
            && !model.supports_sequential_prompt_lookup_verify(9, 128, 2)
            && !model.supports_sequential_prompt_lookup_verify(1, 128, 1)
            && !model.supports_sequential_prompt_lookup_verify(1, 128, 6),
        "MiniCPM-V-4.6 Sequential Q1 qualification exceeded its production envelope"
    );
    anyhow::ensure!(
        !model.supports_exact_batched_speculative_verify(1, max_context + 1, 2)
            && !model.supports_exact_batched_speculative_verify(0, 128, 2)
            && !model.supports_exact_batched_speculative_verify(9, 128, 2)
            && !model.supports_exact_batched_speculative_verify(1, 128, 1)
            && !model.supports_exact_batched_speculative_verify(1, 128, 6)
            && model.supports_exact_batched_speculative_verify_for_kv_cache(
                8,
                max_context,
                5,
                None,
            )
            && !model.supports_exact_batched_speculative_verify_for_kv_cache(
                8,
                max_context,
                5,
                Some(TurboQuantKVBits::K3V4),
            )
            && !model.supports_exact_batched_speculative_verify_for_kv_cache(
                8,
                max_context,
                5,
                Some(TurboQuantKVBits::K4V4),
            ),
        "MiniCPM-V-4.6 exact Q>1 qualification exceeded its production envelope"
    );
    Ok(())
}

#[test]
#[ignore = "requires a real MiniCPM-V-4.6 checkpoint and Apple Silicon"]
#[serial(mlx_metal)]
fn minicpmv46_sequential_q1_batch_shapes_match_b1() -> Result<()> {
    const PREFIX_LEN: usize = 64;
    const VERIFY_WIDTH: usize = 5;

    let Some(model_dir) = std::env::var_os(MODEL_ENV).map(PathBuf::from) else {
        eprintln!("skip: set {MODEL_ENV} to a MiniCPM-V-4.6 checkpoint");
        return Ok(());
    };
    let loader = Loader::open(&model_dir)
        .with_context(|| format!("opening MiniCPM-V-4.6 checkpoint {}", model_dir.display()))?;
    let tokenizer = Tokenizer::from_loader(&loader).context("loading MiniCPM-V-4.6 tokenizer")?;
    let model =
        MiniCpmV46Model::from_loader(&loader).context("loading MiniCPM-V-4.6 checkpoint")?;
    let mut source = tokenizer
        .encode(QUALIFICATION_TEXT, false)
        .context("tokenizing qualification text")?;
    while source.len() < PREFIX_LEN + VERIFY_WIDTH {
        source.extend(source.clone());
    }
    let prefix = &source[..PREFIX_LEN];
    let verify = &source[PREFIX_LEN..PREFIX_LEN + VERIFY_WIDTH];
    let cap = i32::try_from(PREFIX_LEN + VERIFY_WIDTH + 1).context("cache cap exceeds i32")?;

    let mut baseline_cache = model.make_cache(1, cap, model.cache_dtype())?;
    let prefix_input = input_array(prefix, 1, PREFIX_LEN)?;
    let prefix_positions = uniform_position_ids(0, PREFIX_LEN as i32, 1)?;
    let baseline_prefill = model.forward_on(
        &prefix_input,
        &prefix_positions,
        Some(&[PREFIX_LEN as i32]),
        None,
        Some(&mut baseline_cache),
        StreamOrDevice::default(),
    )?;
    mlx::transforms::eval(&[&baseline_prefill])?;
    let mut baseline_tokens = Vec::with_capacity(VERIFY_WIDTH);
    for (depth, &token) in verify.iter().enumerate() {
        let input = input_array(&[token], 1, 1)?;
        let positions = uniform_position_ids((PREFIX_LEN + depth) as i32, 1, 1)?;
        let logits = model.forward_on(
            &input,
            &positions,
            Some(&[1]),
            None,
            Some(&mut baseline_cache),
            StreamOrDevice::default(),
        )?;
        mlx::transforms::eval(&[&logits])?;
        baseline_tokens.push(argmax_tokens(&logits)?[0]);
    }

    for batch in [1_usize, 2, 4, 8] {
        anyhow::ensure!(
            model.supports_sequential_prompt_lookup_verify(batch, PREFIX_LEN, VERIFY_WIDTH,),
            "checkpoint did not expose B{batch} Sequential Q1 qualification"
        );
        let mut cache = model.make_cache(batch as i32, cap, model.cache_dtype())?;
        let mut batch_prefix = Vec::with_capacity(batch * PREFIX_LEN);
        for _ in 0..batch {
            batch_prefix.extend_from_slice(prefix);
        }
        let input = input_array(&batch_prefix, batch, PREFIX_LEN)?;
        let positions = uniform_position_ids(0, PREFIX_LEN as i32, batch)?;
        let prefix_lens = vec![PREFIX_LEN as i32; batch];
        let prefill = model.forward_on(
            &input,
            &positions,
            Some(&prefix_lens),
            None,
            Some(&mut cache),
            StreamOrDevice::default(),
        )?;
        mlx::transforms::eval(&[&prefill])?;
        for (depth, &token) in verify.iter().enumerate() {
            let input = input_array(&vec![token; batch], batch, 1)?;
            let positions = uniform_position_ids((PREFIX_LEN + depth) as i32, 1, batch)?;
            let step_lens = vec![1_i32; batch];
            let logits = model.forward_on(
                &input,
                &positions,
                Some(&step_lens),
                None,
                Some(&mut cache),
                StreamOrDevice::default(),
            )?;
            mlx::transforms::eval(&[&logits])?;
            let actual = argmax_tokens(&logits)?;
            anyhow::ensure!(
                actual == vec![baseline_tokens[depth]; batch],
                "MiniCPM-V-4.6 Sequential Q1 batch-shape divergence at \
                 B{batch} depth={depth}: baseline={} actual={actual:?}",
                baseline_tokens[depth]
            );
        }
    }
    drop(baseline_cache);
    mlx::clear_cache();
    Ok(())
}
