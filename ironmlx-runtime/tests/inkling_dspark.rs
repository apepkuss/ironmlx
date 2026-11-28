//! Native token parity for the published Inkling oQ checkpoint and its DSpark sidecar.
use anyhow::{ensure, Context, Result};
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::{
    core::{vision::DenseVlMethods, Loader, Model, Tokenizer},
    models::{
        dflash2::{DFlash2Target, DFlash2TargetForwardMode},
        DFlash2DraftModel, Inkling2Model,
    },
};
use ironmlx_runtime::core::{
    DFlash2TextGenerationStream, GenerateEvent, GenerateRequest, GenerationStream,
};
use mlx::{Array, Dtype};
use std::{path::PathBuf, time::Instant};

fn decode(
    mut next: impl FnMut() -> Result<Option<GenerateEvent>>,
    started: Instant,
) -> Result<(Vec<u32>, f64, f64)> {
    let mut ids = Vec::new();
    let mut ttft = None;
    while let Some(event) = next()? {
        ttft.get_or_insert_with(|| started.elapsed().as_secs_f64());
        ids.push(event.token);
        if event.finish_reason.is_some() {
            break;
        }
    }
    let first = ttft.context("no tokens generated")?;
    let rate = ids.len().saturating_sub(1) as f64 / (started.elapsed().as_secs_f64() - first);
    Ok((ids, first * 1000.0, rate))
}

fn cosine(a: &Array, b: &Array) -> Result<f64> {
    ensure!(a.size() == b.size(), "reference tensor shape differs");
    let a = a.astype(Dtype::Float32)?.reshape((-1,))?.to_vec::<f32>()?;
    let b = b.astype(Dtype::Float32)?.reshape((-1,))?.to_vec::<f32>()?;
    let (dot, a2, b2) = a
        .iter()
        .zip(b)
        .fold((0.0, 0.0, 0.0), |(dot, a2, b2), (a, b)| {
            let (a, b) = (f64::from(*a), f64::from(b));
            (dot + a * b, a2 + a * a, b2 + b * b)
        });
    Ok(dot / (a2 * b2).sqrt())
}

#[test]
#[ignore = "requires the real Inkling oQ4e checkpoint, its sidecar and INKLING_DS_PARK_TEST_DIR"]
fn inkling_o_q4e_dspark_matches_ordinary_decode() -> Result<()> {
    let root = PathBuf::from(std::env::var("INKLING_DS_PARK_TEST_DIR")?);
    if let Ok(mlx) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{mlx}/lib/mlx.metallib"))?;
    }
    let loader = Loader::open_multimodal(&root)?;
    let model = Inkling2Model::from_loader(&loader)?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let draft = DFlash2DraftModel::from_inkling_sidecar(&loader, &root.join("mtp.safetensors"))?;
    let text = "Write a Rust function to merge two sorted integer iterators. Preserve duplicates.";
    let mut prompt = tokenizer.encode(text, false)?;
    let references = if let Ok(path) = std::env::var("INKLING_OQ_REFERENCE") {
        let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        prompt = serde_json::from_value(reference["prompt"].clone())?;
        Some(reference)
    } else {
        None
    };
    let reference_logits = std::env::var("INKLING_OQ_REFERENCE_TENSORS")
        .ok()
        .map(|path| mlx::io::load_safetensors(&path))
        .transpose()?
        .map(|(tensors, _)| tensors);
    let long_prompt = std::env::var("INKLING_OQ_LONG_PROMPT")
        .ok()
        .map(std::fs::read_to_string)
        .transpose()?
        .map(|p| tokenizer.encode(&p, false))
        .transpose()?;
    let mut cases = vec!["text", "image"];
    if long_prompt.is_some() {
        cases.push("4k");
    }
    for case in cases {
        let image = case == "image";
        let mut ids = if case == "4k" {
            long_prompt.clone().expect("read before adding case")
        } else {
            prompt.clone()
        };
        if image {
            ids.insert(0, 200_054);
        }
        let request = GenerateRequest {
            priority: Default::default(),
            prompt_ids: ids,
            max_new_tokens: 96,
            stop_token_ids: Vec::new(),
            sampler: Sampler::greedy(),
            prefill_chunk_size: 512,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: if image {
                Some(vec![Array::zeros((3, 40, 40), Dtype::Float32)?])
            } else {
                None
            },
            image_grid_thw: image.then_some(vec![(1, 2, 2)]),
            image_spatial_merge_size: 2,
            image_token_id: 200_054,
            constraint: None,
        };
        if let Some(reference) = reference_logits.as_ref().filter(|_| case != "4k") {
            let t = mlx::StreamOrDevice::default();
            let capacity = request.prompt_ids.len() as i32 + 1;
            let mut cache = model.make_cache(1, capacity, model.cache_dtype())?;
            let n = request.prompt_ids.len() as i32;
            let ids: Array = (request.prompt_ids.as_slice(), &[1, n][..]).try_into()?;
            let positions = Array::zeros((1, 1), Dtype::Int32)?;
            let logits = if let Some(pixels) = request.pixel_values.as_ref() {
                let features = model.compute_vision_embeds(pixels, &[(1, 2, 2)], t)?;
                let similarity = cosine(&features, &reference["vision"])?;
                eprintln!("vision cosine={similarity}");
                ensure!(similarity > 0.999, "oQ vision tensor mismatch");
                model.forward_vl_chunk(
                    &ids,
                    &positions,
                    None,
                    None,
                    Some(&mut cache),
                    Some(&features),
                    200_054,
                    t,
                )?
            } else {
                model.forward_on(&ids, &positions, None, None, Some(&mut cache), t)?
            };
            let vocab = model.config().text_config.unpadded_vocab_size;
            let expected = mlx::ops::indexing::slice(&reference[case], &[0][..], &[vocab][..])?;
            let similarity = cosine(&logits, &expected)?;
            eprintln!("{case} prefill logits cosine={similarity}");
            ensure!(similarity > 0.999, "oQ {case} reference tensor mismatch");
        }
        let t = Instant::now();
        let mut ordinary = GenerationStream::new(&model, &tokenizer, request.clone())?;
        let (before, base_ms, base_rate) = decode(|| ordinary.next_token(), t)?;
        drop(ordinary);
        let t = Instant::now();
        let mut stream =
            DFlash2TextGenerationStream::new_text_only(&model, &draft, &tokenizer, request, 8)?;
        let (after, draft_ms, draft_rate) = decode(|| stream.next_token(), t)?;
        let metrics = stream.metrics();
        let ref_first_diff = if case != "text" {
            None
        } else {
            references.as_ref().and_then(|r| {
                r["tokens"]
                    .as_array()?
                    .iter()
                    .zip(&before)
                    .position(|(a, b)| a.as_u64() != Some(u64::from(*b)))
            })
        };
        eprintln!(
            "{}",
            serde_json::json!({
                "case":case, "before_ids":before, "after_ids":after,
                "reference_first_diff":ref_first_diff, "base_ttft_ms":base_ms, "base_decode_tps":base_rate,
                "draft_ttft_ms":draft_ms, "draft_decode_tps":draft_rate, "metrics":metrics,
            })
        );
        ensure!(
            metrics.drafted_tokens > 0 && metrics.accepted_draft_tokens > 0,
            "{case}: no accepted DSpark proposals"
        );
        assert_eq!(
            before, after,
            "{case}: DSpark changed the native target token IDs"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires the real Inkling oQ4e checkpoint and INKLING_DS_PARK_TEST_DIR"]
fn inkling_o_q4e_batched_verify_matches_serial_rows_and_rollback() -> Result<()> {
    let root = PathBuf::from(std::env::var("INKLING_DS_PARK_TEST_DIR")?);
    if let Ok(mlx) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{mlx}/lib/mlx.metallib"))?;
    }
    let loader = Loader::open_multimodal(&root)?;
    let model = Inkling2Model::from_loader(&loader)?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let token = tokenizer.encode("A silent blue river runs past the station.", false)?;
    let prompt: Vec<_> = token.iter().copied().cycle().take(1056).collect();
    let draft = tokenizer.encode(
        "The detective follows the river past an empty train car.",
        false,
    )?;
    let taps = [5, 23, 41, 48];
    let target = mlx::StreamOrDevice::default();
    let positions = Array::zeros((1, 1), Dtype::Int32)?;
    let singleton = |id: u32| -> Result<Array> { Ok((&[id][..], &[1, 1][..]).try_into()?) };
    let snapshot = |cache: &[ironmlx_lm::core::cache::LayerCache]| {
        cache
            .iter()
            .map(|layer| layer.dflash2_transaction_snapshot())
            .collect::<Result<Vec<_>>>()
    };
    let assert_equal = |actual: &Array, expected: &Array, label: &str| -> Result<()> {
        ensure!(
            actual.shape() == expected.shape(),
            "{label}: shape mismatch"
        );
        ensure!(
            actual
                .equal(expected)?
                .all(mlx::ops::All, false)?
                .item::<bool>()?,
            "{label}: different values"
        );
        Ok(())
    };
    for prompt_len in [48, 1056] {
        let input: Array = (&prompt[..prompt_len], &[1, prompt_len as i32][..]).try_into()?;
        for width in 2..=8 {
            let mut serial = model.make_cache(1, 1152, model.cache_dtype())?;
            let mut batched = model.make_cache(1, 1152, model.cache_dtype())?;
            for cache in [&mut serial, &mut batched] {
                let logits =
                    model.forward_on(&input, &positions, None, None, Some(cache), target)?;
                mlx::transforms::eval(&[&logits])?;
            }
            let accepted = width / 2;
            let mut serial_prefix = Vec::new();
            let mut expected_logits = Vec::new();
            let mut expected_taps = Vec::new();
            for (i, &token) in draft[..width].iter().enumerate() {
                let row = model.dflash2_forward_target_on(
                    &singleton(token)?,
                    &positions,
                    Some(&mut serial),
                    &taps,
                    DFlash2TargetForwardMode::OrdinaryDecode,
                    target,
                )?;
                let logits = model.dflash2_project_hidden_on(&row.hidden, target)?;
                mlx::transforms::eval(&[&logits, &row.context_hidden])?;
                expected_logits.push(logits);
                expected_taps.push(row.context_hidden);
                if i + 1 == accepted {
                    serial_prefix = snapshot(&serial)?;
                }
            }
            let before = snapshot(&batched)?;
            for layer in &mut batched {
                layer.begin_speculative_prefix_capture()?;
            }
            let ids: Array = (&draft[..width], &[1, width as i32][..]).try_into()?;
            let block = model.dflash2_forward_target_on(
                &ids,
                &positions,
                Some(&mut batched),
                &taps,
                DFlash2TargetForwardMode::GreedyVerify,
                target,
            )?;
            let expected_logits =
                mlx::ops::shape::concatenate(&expected_logits.iter().collect::<Vec<_>>(), 1)?;
            let expected_taps =
                mlx::ops::shape::concatenate(&expected_taps.iter().collect::<Vec<_>>(), 1)?;
            let logits = model.dflash2_project_hidden_on(&block.hidden, target)?;
            let case = format!("prompt {prompt_len}, width {width}");
            assert_equal(&logits, &expected_logits, &format!("{case}: verify logits"))?;
            assert_equal(
                &block.context_hidden,
                &expected_taps,
                &format!("{case}: verify taps"),
            )?;
            model.dflash2_restore_target_prefix_on(&mut batched, &before, accepted, target)?;
            for (layer, saved) in serial.iter_mut().zip(&serial_prefix) {
                layer.restore(saved)?;
            }
            let input = singleton(draft[width])?;
            let decode = |cache: &mut [_]| {
                model.forward_on(&input, &positions, None, None, Some(cache), target)
            };
            assert_equal(
                &decode(&mut batched)?,
                &decode(&mut serial)?,
                &format!("{case}: after rollback"),
            )?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "fixed-trajectory B1 benchmark; requires real Inkling weights on Metal"]
fn inkling_fixed_trajectory() -> Result<()> {
    let root = PathBuf::from(std::env::var("INKLING_DS_PARK_TEST_DIR")?);
    if let Ok(mlx) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{mlx}/lib/mlx.metallib"))?;
    }
    let loader = Loader::open_multimodal(&root)?;
    let model = Inkling2Model::from_loader(&loader)?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let seed = tokenizer.encode(
        "A blue river turns east below the old railway bridge. ",
        false,
    )?;
    let trajectory = tokenizer.encode("The train left just before sunrise. A passenger in the rear carriage looked up from her book and watched the river disappear behind a bend.",false)?;
    let t = mlx::StreamOrDevice::default();
    let positions = Array::zeros((1, 1), Dtype::Int32)?;
    let sync = |cache: &[ironmlx_lm::core::cache::LayerCache]| -> Result<()> {
        let roots = cache
            .iter()
            .flat_map(|l| l.state_arrays())
            .collect::<Vec<_>>();
        mlx::transforms::eval(&roots)?;
        Ok(())
    };
    let lengths =
        std::env::var("CONTEXT_LENGTHS").unwrap_or_else(|_| "4096,512,1024,2048,8192".to_owned());
    let widths = std::env::var("CONTEXT_WIDTHS").unwrap_or_else(|_| "1,2,3,4".to_owned());
    let repeats: usize = std::env::var("CONTEXT_STEPS")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(20);
    ensure!(repeats > 0, "CONTEXT_STEPS must be positive");
    for length in lengths.split(',').map(str::parse::<usize>) {
        let length = length?;
        ensure!(length > 0, "context length must be positive");
        let prompt: Vec<_> = seed.iter().copied().cycle().take(length).collect();
        for width in widths.split(',').map(str::parse::<usize>) {
            let width = width?;
            ensure!(
                (1..=4).contains(&width),
                "width must be in 1..=4 including the anchor"
            );
            let steps = repeats.checked_add(4).context("too many steps")?;
            // Use the same cache growth and headroom for every draft width.
            let capacity = repeats
                .checked_mul(4)
                .and_then(|n| n.checked_add(1024))
                .and_then(|n| n.checked_add(length))
                .context("cache capacity overflow")?;
            let mut cache = model.make_cache(1, i32::try_from(capacity)?, model.cache_dtype())?;
            for chunk in prompt.chunks(1024) {
                let ids: Array = (chunk, &[1, chunk.len() as i32][..]).try_into()?;
                let logits = model.forward_on(&ids, &positions, None, None, Some(&mut cache), t)?;
                mlx::transforms::eval(&[&logits])?;
            }
            sync(&cache)?;
            let mut cursor = 0;
            let mut readings = Vec::new();
            for step in 0..steps {
                let ids = (0..width)
                    .map(|i| trajectory[(cursor + i) % trajectory.len()])
                    .collect::<Vec<_>>();
                let input: Array = (ids.as_slice(), &[1, width as i32][..]).try_into()?;
                let start = Instant::now();
                let before = if width > 1 {
                    let snapshots = cache
                        .iter()
                        .map(|l| l.dflash2_transaction_snapshot())
                        .collect::<Result<Vec<_>>>()?;
                    for layer in &mut cache {
                        layer.begin_speculative_prefix_capture()?;
                    }
                    snapshots
                } else {
                    vec![]
                };
                let build_us;
                let pred = if width == 1 {
                    let logits =
                        model.forward_on(&input, &positions, None, None, Some(&mut cache), t)?;
                    build_us = start.elapsed().as_micros();
                    logits.argmax(-1, false)?
                } else {
                    let out = model.dflash2_forward_target_on(
                        &input,
                        &positions,
                        Some(&mut cache),
                        &[5, 23, 41, 48],
                        DFlash2TargetForwardMode::GreedyVerify,
                        t,
                    )?;
                    let pred = model
                        .dflash2_project_hidden_on(&out.hidden, t)?
                        .argmax(-1, false)?;
                    build_us = start.elapsed().as_micros();
                    mlx::transforms::eval(&[&pred, &out.context_hidden])?;
                    pred
                };
                let tokens = pred.to_vec::<u32>()?;
                let forward_us = start.elapsed().as_micros();
                let accepted = if width == 1 { 1 } else { 1 + step % width };
                if accepted < width {
                    model.dflash2_restore_target_prefix_on(&mut cache, &before, accepted, t)?;
                } else if width > 1 {
                    for layer in &mut cache {
                        layer.discard_speculative_prefix_capture();
                    }
                }
                sync(&cache)?;
                if step >= 4 {
                    readings.push(serde_json::json!({"forward_us":forward_us,"build_us":build_us,"commit_us":start.elapsed().as_micros()-forward_us,"accepted_inputs":accepted,"predictions":tokens}));
                }
                cursor += accepted;
            }
            eprintln!(
                "FIXED {}",
                serde_json::json!({"context":length,"width":width,"runs":readings})
            );
        }
    }
    Ok(())
}

/// How far 8-bit global K/V moves the next-token distribution from bfloat16
/// K/V: both caches prefill the same prompt, then follow the bfloat16 greedy
/// continuation token by token. Prints only.
#[test]
#[ignore = "8-bit K/V quality; requires real Inkling weights on Metal"]
fn inkling_int8_kv_logits_against_bfloat16() -> Result<()> {
    let root = PathBuf::from(std::env::var("INKLING_DS_PARK_TEST_DIR")?);
    if let Ok(mlx) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{mlx}/lib/mlx.metallib"))?;
    }
    let loader = Loader::open_multimodal(&root)?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let steps: usize = std::env::var("KV_QUALITY_STEPS")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(64);
    let t = mlx::StreamOrDevice::default();
    let positions = Array::zeros((1, 1), Dtype::Int32)?;
    let mut prompts = vec![std::fs::read_to_string(std::env::var(
        "INKLING_OQ_LONG_PROMPT",
    )?)?];
    if let Ok(path) = std::env::var("KV_QUALITY_LONG_PROMPT") {
        prompts.push(std::fs::read_to_string(path)?);
    }
    for text in prompts {
        // A fresh model per prompt: the 8-bit switch only turns off.
        let model = Inkling2Model::from_loader(&loader)?;
        let prompt = tokenizer.encode(&text, false)?;
        let cap = i32::try_from(prompt.len() + steps + 64)?;
        // KV_QUALITY_BASELINE=1 compares two bfloat16 caches prefilled in
        // different chunks instead: the noise floor for the figures below.
        let baseline = std::env::var_os("KV_QUALITY_BASELINE").is_some();
        if baseline {
            model.use_dense_kv();
        }
        let mut int8 = model.make_cache(1, cap, model.cache_dtype())?;
        let mut eight_bit = 0;
        for layer in &int8 {
            if let ironmlx_lm::core::cache::LayerCache::Full(kv) = layer {
                eight_bit += usize::from(kv.int8().is_some());
            }
        }
        model.use_dense_kv();
        let mut dense = model.make_cache(1, cap, model.cache_dtype())?;
        ensure!(
            baseline || eight_bit > 0,
            "no cache stores 8-bit K/V (IRONMLX_INKLING_KV_INT8=0?)"
        );
        let mut last = Vec::new();
        for (index, cache) in [&mut dense, &mut int8].into_iter().enumerate() {
            let mut logits = None;
            let size = if baseline && index == 1 { 1024 } else { 2048 };
            for chunk in prompt.chunks(size) {
                let ids: Array = (chunk, &[1, chunk.len() as i32][..]).try_into()?;
                let out = model.forward_on(&ids, &positions, None, None, Some(cache), t)?;
                mlx::transforms::eval(&[&out])?;
                logits = Some(out);
            }
            last.push(logits.context("empty prompt")?);
        }
        let (mut agree, mut kl_sum, mut kl_max) = (0usize, 0f64, 0f64);
        for step in 0..steps {
            let reference = last[0].astype(Dtype::Float32)?.reshape(&[-1][..])?;
            let candidate = last[1].astype(Dtype::Float32)?.reshape(&[-1][..])?;
            let p = reference.softmax(-1, true)?;
            let log_p = p.log()?;
            let log_q = candidate.softmax(-1, true)?.log()?;
            let kl = p
                .try_mul(&log_p.try_sub(&log_q)?)?
                .sum(mlx::ops::All, false)?
                .item::<f32>()? as f64;
            let next = reference.argmax(-1, false)?.item::<u32>()?;
            agree += usize::from(candidate.argmax(-1, false)?.item::<u32>()? == next);
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            if step + 1 == steps {
                break;
            }
            let ids: Array = (&[next][..], &[1, 1][..]).try_into()?;
            last = vec![
                model.forward_on(&ids, &positions, None, None, Some(&mut dense), t)?,
                model.forward_on(&ids, &positions, None, None, Some(&mut int8), t)?,
            ];
            mlx::transforms::eval(&[&last[0], &last[1]])?;
        }
        println!(
            "KVQUALITY prompt_tokens={} steps={steps} eight_bit_layers={eight_bit} top1_agree={agree}/{steps} mean_kl={:.2e} max_kl={:.2e}",
            prompt.len(),
            kl_sum / steps as f64,
            kl_max
        );
    }
    Ok(())
}
