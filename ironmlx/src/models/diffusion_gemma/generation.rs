use std::collections::HashSet;
use std::sync::OnceLock;

use anyhow::{anyhow, Context};
use mlx::compile::CompiledFn;
use mlx::{random, Array, Device, Dtype, Stream, StreamOrDevice, ThreadLocalStream};

use crate::core::constrained::{
    apply_diffusion_token_masks, apply_token_mask, ConstraintPlan, ConstraintSession,
};
use crate::core::loader::EosTokenId;
use crate::core::Tokenizer;
use crate::Result;
use llguidance::toktrie::SimpleVob;

use super::config::DiffusionGemmaGenerationConfig;
use super::model::{DiffusionGemmaEncoderInputs, DiffusionGemmaModel};
use super::ops::entropy_probs_chain_on;

const DEFAULT_MIN_CANVAS_LENGTH: i32 = 64;
static GENERATION_STREAM: OnceLock<ThreadLocalStream> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct DiffusionGemmaGenerateEvent {
    pub token: u32,
    pub text: String,
    pub finish_reason: Option<&'static str>,
}

pub type DiffusionGemmaEventSink<'a> =
    &'a mut dyn FnMut(DiffusionGemmaGenerateEvent) -> Result<bool>;

struct DiffusionGemmaMultimodalInput<'a> {
    pixel_values: &'a [Array],
    image_grid_thw: &'a [(i32, i32, i32)],
    image_token_id: i32,
}

struct DefaultStreamGuard {
    previous_device: Device,
    previous_stream: Stream,
}

impl Drop for DefaultStreamGuard {
    fn drop(&mut self) {
        mlx::set_default_device(self.previous_device);
        mlx::set_default_stream(self.previous_stream);
    }
}

fn diffusion_generation_stream() -> Result<ThreadLocalStream> {
    if let Some(stream) = GENERATION_STREAM.get() {
        return Ok(*stream);
    }

    let stream = mlx::new_thread_local_stream(mlx::default_device())
        .context("DiffusionGemma: create generation thread-local stream")?;
    if GENERATION_STREAM.set(stream).is_ok() {
        Ok(stream)
    } else {
        Ok(*GENERATION_STREAM
            .get()
            .expect("DiffusionGemma generation stream was set"))
    }
}

fn enter_diffusion_generation_stream() -> Result<(StreamOrDevice, DefaultStreamGuard)> {
    let stream = diffusion_generation_stream()?;
    let previous_device = mlx::default_device();
    let previous_stream = mlx::default_stream(previous_device);
    let concrete_stream = mlx::stream_from_thread_local_stream(stream);

    mlx::set_default_device(concrete_stream.device);
    mlx::set_default_stream(concrete_stream);

    Ok((
        StreamOrDevice::ThreadLocalStream(stream),
        DefaultStreamGuard {
            previous_device,
            previous_stream,
        },
    ))
}

pub fn generate_text(
    model: &DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    generation_config: &DiffusionGemmaGenerationConfig,
    max_new_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
) -> Result<Vec<DiffusionGemmaGenerateEvent>> {
    let mut events = Vec::new();
    generate_text_with_events(
        model,
        tokenizer,
        prompt_ids,
        generation_config,
        max_new_tokens,
        temperature,
        seed,
        &mut |event| {
            events.push(event);
            Ok(true)
        },
    )?;
    Ok(events)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_text_with_events(
    model: &DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    generation_config: &DiffusionGemmaGenerationConfig,
    max_new_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    emit: DiffusionGemmaEventSink<'_>,
) -> Result<()> {
    generate_impl(
        model,
        tokenizer,
        prompt_ids,
        generation_config,
        max_new_tokens,
        temperature,
        seed,
        None,
        None,
        true,
        emit,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn generate_text_with_events_constrained(
    model: &DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    generation_config: &DiffusionGemmaGenerationConfig,
    max_new_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    constraint: &ConstraintPlan,
    skip_special_tokens: bool,
    emit: DiffusionGemmaEventSink<'_>,
) -> Result<()> {
    generate_impl(
        model,
        tokenizer,
        prompt_ids,
        generation_config,
        max_new_tokens,
        temperature,
        seed,
        None,
        Some(constraint),
        skip_special_tokens,
        emit,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn generate_image_text(
    model: &DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    pixel_values: &[Array],
    image_grid_thw: &[(i32, i32, i32)],
    image_token_id: i32,
    generation_config: &DiffusionGemmaGenerationConfig,
    max_new_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
) -> Result<Vec<DiffusionGemmaGenerateEvent>> {
    let mut events = Vec::new();
    generate_image_text_with_events(
        model,
        tokenizer,
        prompt_ids,
        pixel_values,
        image_grid_thw,
        image_token_id,
        generation_config,
        max_new_tokens,
        temperature,
        seed,
        &mut |event| {
            events.push(event);
            Ok(true)
        },
    )?;
    Ok(events)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_image_text_with_events(
    model: &DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    pixel_values: &[Array],
    image_grid_thw: &[(i32, i32, i32)],
    image_token_id: i32,
    generation_config: &DiffusionGemmaGenerationConfig,
    max_new_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    emit: DiffusionGemmaEventSink<'_>,
) -> Result<()> {
    generate_impl(
        model,
        tokenizer,
        prompt_ids,
        generation_config,
        max_new_tokens,
        temperature,
        seed,
        Some(DiffusionGemmaMultimodalInput {
            pixel_values,
            image_grid_thw,
            image_token_id,
        }),
        None,
        true,
        emit,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn generate_image_text_with_events_constrained(
    model: &DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    pixel_values: &[Array],
    image_grid_thw: &[(i32, i32, i32)],
    image_token_id: i32,
    generation_config: &DiffusionGemmaGenerationConfig,
    max_new_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    constraint: &ConstraintPlan,
    skip_special_tokens: bool,
    emit: DiffusionGemmaEventSink<'_>,
) -> Result<()> {
    generate_impl(
        model,
        tokenizer,
        prompt_ids,
        generation_config,
        max_new_tokens,
        temperature,
        seed,
        Some(DiffusionGemmaMultimodalInput {
            pixel_values,
            image_grid_thw,
            image_token_id,
        }),
        Some(constraint),
        skip_special_tokens,
        emit,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_impl(
    model: &DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    generation_config: &DiffusionGemmaGenerationConfig,
    max_new_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    multimodal: Option<DiffusionGemmaMultimodalInput<'_>>,
    constraint: Option<&ConstraintPlan>,
    skip_special_tokens: bool,
    emit: DiffusionGemmaEventSink<'_>,
) -> Result<()> {
    if prompt_ids.is_empty() {
        return Err(anyhow!(
            "DiffusionGemma text generation requires a non-empty prompt"
        ));
    }
    if temperature < 0.0 {
        return Err(anyhow!("DiffusionGemma temperature must be >= 0"));
    }
    let (target, _stream_guard) = enter_diffusion_generation_stream()?;
    if max_new_tokens == 0 {
        return Ok(());
    }
    let max_denoising_steps = generation_config.max_denoising_steps;
    let entropy_bound = generation_config.entropy_bound()?;
    let entropy_bound: Array = (&[entropy_bound][..], ()).try_into()?;
    let confidence_threshold: Array =
        (&[generation_config.confidence_threshold][..], ()).try_into()?;
    let mut stop_ids: HashSet<u32> = tokenizer.eos_token_ids().iter().copied().collect();
    if let Some(ids) = &generation_config.eos_token_id {
        match ids {
            EosTokenId::Single(id) => {
                stop_ids.insert(*id);
            }
            EosTokenId::Multi(ids) => {
                stop_ids.extend(ids.iter().copied());
            }
        }
    }

    let input_ids = prompt_ids_array(prompt_ids)?;
    let mut cache = model.make_cache();
    if let Some(mm) = multimodal {
        let mm_token_type_ids = multimodal_token_type_ids(prompt_ids, mm.image_token_id as u32);
        model.encode_inputs_on(
            &input_ids,
            DiffusionGemmaEncoderInputs {
                pixel_values: Some(mm.pixel_values),
                image_grid_thw: Some(mm.image_grid_thw),
                mm_token_type_ids: Some(&mm_token_type_ids),
                image_token_id: mm.image_token_id,
            },
            &mut cache,
            target,
        )?;
    } else {
        model.encode_tokens_on(&input_ids, &mut cache, target)?;
    }

    let soft_embedding_weight = model
        .soft_embedding_weight_on(target)
        .context("DiffusionGemma: dequantizing embedding table for self-conditioning")?;
    let entropy_probs_chain = model.entropy_probs_chain();
    let entropy_transfer_mask_chain = model.entropy_transfer_mask_chain();
    let canvas_cap = model.config.canvas_length;
    let min_canvas_length = canvas_cap.clamp(1, DEFAULT_MIN_CANVAS_LENGTH);
    let vocab_size = model.config.text_config.vocab_size;
    let mut rng = seed.map(random::key).transpose()?;
    let mut current_canvas: Option<Array> = None;
    let mut generated = 0usize;
    let mut detok = tokenizer.decode_stream(skip_special_tokens);
    let mut constraint_session = constraint.map(ConstraintPlan::start_session).transpose()?;

    while generated < max_new_tokens {
        if let Some(canvas) = current_canvas.as_ref() {
            model.encode_tokens_on(canvas, &mut cache, target)?;
        }

        let remaining = (max_new_tokens - generated) as i32;
        let canvas_len = canvas_cap.min(remaining.max(min_canvas_length));
        let sample_key = next_random_key(&mut rng)?;
        let mut canvas =
            sample_diffusion_canvas_on(vocab_size, canvas_len, sample_key.as_ref(), target)
                .context("DiffusionGemma: initialize canvas")?;
        let mut argmax_canvas = canvas.clone();
        let mut self_conditioning: Option<Array> = None;
        let mut stability_history: Vec<Array> = Vec::new();
        let mut denoising_steps_this_canvas = 0_i32;

        for cur_step in (1..=max_denoising_steps).rev() {
            denoising_steps_this_canvas += 1;
            let mut logits =
                model.decode_logits_on(&canvas, &cache, self_conditioning.as_ref(), target)?;
            let schedule_temperature =
                linear_temperature(cur_step, max_denoising_steps, generation_config);
            logits = super::ops::div_scalar_like_on(&logits, schedule_temperature, target)?;

            let constrained_logits;
            if let Some(session) = constraint_session.as_ref() {
                let projection = constrained_canvas_on(&logits, session, false, &mut rng, target)?;
                constrained_logits = Some(apply_diffusion_token_masks(&logits, &projection.masks)?);
                argmax_canvas = projection.canvas;
            } else {
                constrained_logits = None;
                argmax_canvas = argmax_canvas_on(&logits, target)?;
            }
            if cur_step == 1 {
                break;
            }

            let entropy_logits = constrained_logits.as_ref().unwrap_or(&logits);
            let (entropy, probs) =
                entropy_probs_chain_on(entropy_logits, Some(entropy_probs_chain), target)?;
            if stable_and_confident_on(
                &argmax_canvas,
                &entropy,
                &mut stability_history,
                generation_config,
                Some(&confidence_threshold),
                Some(model.stable_confidence_chain()),
                target,
            )? {
                if seed.is_some() {
                    if temperature > 0.0 {
                        let _ = next_random_key(&mut rng)?;
                    }
                    let _ = next_random_key(&mut rng)?;
                }
                break;
            }

            let denoiser_canvas = if temperature <= 0.0 {
                argmax_canvas.clone()
            } else {
                let sample_logits = super::ops::div_scalar_like_on(&logits, temperature, target)?;
                if let Some(session) = constraint_session.as_ref() {
                    constrained_canvas_on(&sample_logits, session, true, &mut rng, target)?.canvas
                } else {
                    let sample_key = next_random_key(&mut rng)?;
                    let mut sampler = random::categorical(&sample_logits).axis(-1).stream(target);
                    if let Some(key) = sample_key.as_ref() {
                        sampler = sampler.key(key);
                    }
                    let sampled = sampler.sample()?;
                    sampled.astype_on(Dtype::Int32, target)?
                }
            };

            let acceptance_mask = entropy_transfer_mask_on(
                &entropy,
                &entropy_bound,
                Some(entropy_transfer_mask_chain),
                target,
            )?;
            let sample_key = next_random_key(&mut rng)?;
            let random_canvas =
                sample_diffusion_canvas_on(vocab_size, canvas_len, sample_key.as_ref(), target)?;
            canvas = mlx::ops::indexing::where_on(
                &acceptance_mask,
                &denoiser_canvas,
                &random_canvas,
                target,
            )?;
            let next_self_conditioning = soft_embeddings_from_probs_on(
                &probs,
                &soft_embedding_weight,
                model.embed_scale(),
                target,
            )?;
            self_conditioning = Some(next_self_conditioning);
        }

        current_canvas = Some(argmax_canvas.clone());
        mlx::transforms::eval(&[&argmax_canvas]).context("DiffusionGemma: eval canvas")?;
        tracing::debug!(
            canvas_len,
            generated,
            steps = denoising_steps_this_canvas,
            "DiffusionGemma denoised canvas"
        );
        let token_ids = token_ids_from_canvas(&argmax_canvas)?;
        for token in token_ids {
            generated += 1;
            if stop_ids.contains(&token) {
                if let Some(session) = constraint_session.as_mut() {
                    session.commit_token(token)?;
                }
                let keep_going = emit(DiffusionGemmaGenerateEvent {
                    token,
                    text: String::new(),
                    finish_reason: Some("stop"),
                })?;
                let _ = keep_going;
                return Ok(());
            }
            if let Some(session) = constraint_session.as_mut() {
                session.commit_token(token)?;
            }
            let text = detok.step(token)?.unwrap_or_default();
            if !emit(DiffusionGemmaGenerateEvent {
                token,
                text,
                finish_reason: None,
            })? {
                return Ok(());
            }
            if generated >= max_new_tokens {
                break;
            }
        }
    }

    if let Some(session) = constraint_session.as_mut() {
        anyhow::ensure!(
            session.is_accepting()?,
            "DiffusionGemma reached max_new_tokens before constrained output became complete"
        );
    }

    let _ = emit(DiffusionGemmaGenerateEvent {
        token: 0,
        text: String::new(),
        finish_reason: Some("length"),
    })?;
    Ok(())
}

fn next_random_key(rng: &mut Option<Array>) -> Result<Option<Array>> {
    let Some(current) = rng.take() else {
        return Ok(None);
    };
    let (next, key) = random::split(&current)?;
    *rng = Some(next);
    Ok(Some(key))
}

fn prompt_ids_array(prompt_ids: &[u32]) -> Result<Array> {
    let len = i32::try_from(prompt_ids.len()).context("DiffusionGemma: prompt too long")?;
    let mut ids = Vec::with_capacity(prompt_ids.len());
    for &id in prompt_ids {
        ids.push(
            i32::try_from(id)
                .with_context(|| format!("DiffusionGemma: token id {id} does not fit int32"))?,
        );
    }
    Ok((ids.as_slice(), &[1_i32, len][..]).try_into()?)
}

fn sample_diffusion_canvas_on(
    vocab_size: i32,
    canvas_len: i32,
    key: Option<&Array>,
    target: StreamOrDevice,
) -> Result<Array> {
    if vocab_size <= 0 {
        return Err(anyhow!(
            "DiffusionGemma: vocab_size must be positive, got {vocab_size}"
        ));
    }
    if canvas_len <= 0 {
        return Err(anyhow!(
            "DiffusionGemma: canvas_len must be positive, got {canvas_len}"
        ));
    }
    let mut sampler = random::randint()
        .low(0)
        .high(vocab_size as i64)
        .shape((1_i32, canvas_len))
        .dtype(Dtype::Int32)
        .stream(target);
    if let Some(key) = key {
        sampler = sampler.key(key);
    }
    Ok(sampler.sample()?)
}

fn argmax_canvas_on(logits: &Array, target: StreamOrDevice) -> Result<Array> {
    let argmax = mlx::ops::argmax_on(logits, -1_i32, false, target)?;
    Ok(argmax.astype_on(Dtype::Int32, target)?)
}

struct ConstrainedCanvas {
    canvas: Array,
    masks: Vec<SimpleVob>,
}

fn constrained_canvas_on(
    logits: &Array,
    committed: &ConstraintSession,
    sample: bool,
    rng: &mut Option<Array>,
    target: StreamOrDevice,
) -> Result<ConstrainedCanvas> {
    let shape = logits.shape();
    let dims = shape.as_slice();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1,
        "DiffusionGemma constrained logits must be [1,L,V], got {dims:?}"
    );
    let canvas_len = dims[1] as usize;
    let vocab = dims[2] as usize;
    anyhow::ensure!(
        committed.vocab_size() <= vocab,
        "DiffusionGemma constraint vocabulary {} exceeds model vocabulary {vocab}",
        committed.vocab_size()
    );

    let mut session = committed.fork();
    let mut masks = Vec::with_capacity(canvas_len);
    let mut tokens = Vec::with_capacity(canvas_len);
    let mut terminal_eos = None;

    for position in 0..canvas_len {
        if let Some(eos) = terminal_eos {
            let mut mask = SimpleVob::alloc(committed.vocab_size());
            mask.allow_token(eos);
            masks.push(mask);
            tokens.push(eos);
            continue;
        }

        let mask = session.compute_mask()?;
        let row = mlx::ops::indexing::slice_on(
            logits,
            [0_i32, position as i32, 0_i32],
            [1_i32, position as i32 + 1, vocab as i32],
            target,
        )?;
        let row = mlx::ops::shape::squeeze_on(&row, &[0_i32, 1_i32][..], target)?;
        let masked = apply_token_mask(&row, &mask)?;
        let token = if sample {
            let key = next_random_key(rng)?;
            let mut sampler = random::categorical(&masked).axis(-1).stream(target);
            if let Some(key) = key.as_ref() {
                sampler = sampler.key(key);
            }
            sampler.sample()?.item::<u32>()?
        } else {
            mlx::ops::argmax_on(&masked, -1_i32, false, target)?.item::<u32>()?
        };
        anyhow::ensure!(
            mask.is_allowed(token),
            "DiffusionGemma selected token {token} outside its constrained mask"
        );
        session.commit_token(token)?;
        if session.eos_token_ids().contains(&token) {
            terminal_eos = Some(token);
        }
        masks.push(mask);
        tokens.push(token);
    }

    let token_ids = tokens
        .into_iter()
        .map(|token| i32::try_from(token).context("DiffusionGemma token id exceeds int32"))
        .collect::<Result<Vec<_>>>()?;
    let canvas: Array = (&token_ids[..], &[1_i32, canvas_len as i32][..]).try_into()?;
    Ok(ConstrainedCanvas {
        canvas: canvas.astype_on(Dtype::Int32, target)?,
        masks,
    })
}

fn token_ids_from_canvas(canvas: &Array) -> Result<Vec<u32>> {
    let ids: Vec<i32> = canvas.to_vec()?;
    ids.into_iter()
        .map(|id| {
            u32::try_from(id).map_err(|_| {
                anyhow!("DiffusionGemma: generated negative token id {id} cannot detokenize")
            })
        })
        .collect()
}

pub(crate) fn multimodal_token_type_ids(prompt_ids: &[u32], image_token_id: u32) -> Vec<i32> {
    prompt_ids
        .iter()
        .map(|&id| if id == image_token_id { 1 } else { 0 })
        .collect()
}

fn linear_temperature(
    cur_step: i32,
    max_denoising_steps: i32,
    config: &DiffusionGemmaGenerationConfig,
) -> f32 {
    config.t_min + ((config.t_max - config.t_min) * (cur_step as f32 / max_denoising_steps as f32))
}

#[cfg(test)]
fn entropy_and_soft_embeddings_on(
    logits: &Array,
    embedding_weight: &Array,
    embed_scale: f32,
    entropy_probs_chain: Option<&CompiledFn>,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    let (entropy, probs) = entropy_probs_chain_on(logits, entropy_probs_chain, target)?;
    let soft = soft_embeddings_from_probs_on(&probs, embedding_weight, embed_scale, target)?;
    Ok((entropy, soft))
}

fn soft_embeddings_from_probs_on(
    probs: &Array,
    embedding_weight: &Array,
    embed_scale: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    let probs = probs.astype_on(embedding_weight.dtype(), target)?;
    let soft = probs.matmul_on(embedding_weight, target)?;
    let soft = super::ops::mul_scalar_like_on(&soft, embed_scale, target)?;
    Ok(soft)
}

fn entropy_transfer_mask_on(
    entropy: &Array,
    entropy_bound: &Array,
    compiled: Option<&CompiledFn>,
    target: StreamOrDevice,
) -> Result<Array> {
    let shape = entropy.shape();
    let dims = shape.as_slice();
    if dims.len() != 2 || dims[0] < 1 || dims[1] < 1 {
        return Err(anyhow!(
            "DiffusionGemma entropy mask expects [B,L], got {:?}",
            dims
        ));
    }
    if let Some(compiled) = compiled {
        let mut outputs = compiled.invoke(&[entropy, entropy_bound])?;
        if outputs.len() != 1 {
            return Err(anyhow!(
                "DiffusionGemma entropy transfer mask returned {} outputs",
                outputs.len()
            ));
        }
        return Ok(outputs.pop().expect("checked output length"));
    }

    let sorted_indices = mlx::ops::sort::argsort_on(entropy, -1, target)?;
    let sorted_entropy =
        mlx::ops::indexing::take_along_axis_on(entropy, &sorted_indices, -1, target)?;
    let prefix_entropy =
        mlx::ops::cumulative::cumsum_on(&sorted_entropy, -1, false, false, target)?;
    let sorted_mask = prefix_entropy.less_equal_on(entropy_bound, target)?;
    let scattered_mask = mlx::ops::constructors::zeros_like_on(&sorted_mask, target)?;
    Ok(mlx::ops::indexing::put_along_axis_on(
        &scattered_mask,
        &sorted_indices,
        &sorted_mask,
        -1,
        target,
    )?)
}

fn stable_and_confident_on(
    argmax_canvas: &Array,
    token_entropy: &Array,
    history: &mut Vec<Array>,
    config: &DiffusionGemmaGenerationConfig,
    confidence_threshold: Option<&Array>,
    stable_confidence_chain: Option<&CompiledFn>,
    target: StreamOrDevice,
) -> Result<bool> {
    if config.stability_threshold == 1 {
        let previous = history.first().cloned();
        history.clear();
        history.push(argmax_canvas.clone());
        let Some(previous) = previous else {
            return Ok(false);
        };
        let owned_confidence_threshold;
        let confidence_threshold = if let Some(threshold) = confidence_threshold {
            threshold
        } else {
            owned_confidence_threshold = super::ops::scalar_array_like_on(
                config.confidence_threshold,
                token_entropy,
                target,
            )?;
            &owned_confidence_threshold
        };
        let should_stop = stable_confident_threshold_one_on(
            argmax_canvas,
            &previous,
            token_entropy,
            confidence_threshold,
            stable_confidence_chain,
            target,
        )?;
        return Ok(should_stop.item::<bool>()?);
    }

    let stable = if history.len() == config.stability_threshold {
        let mut stable = true;
        for prev in history.iter() {
            let same_tokens = argmax_canvas.equal_on(prev, target)?;
            let same_canvas = mlx::ops::all_on(&same_tokens, mlx::ops::All, false, target)?;
            stable &= same_canvas.item::<bool>()?;
        }
        stable
    } else {
        false
    };

    history.push(argmax_canvas.clone());
    if history.len() > config.stability_threshold {
        history.remove(0);
    }
    if !stable {
        return Ok(false);
    }
    let mean = mlx::ops::mean_on(token_entropy, mlx::ops::All, false, target)?;
    let mean = mean.item::<f32>()?;
    Ok(mean < config.confidence_threshold)
}

fn stable_confident_threshold_one_on(
    current_canvas: &Array,
    previous_canvas: &Array,
    token_entropy: &Array,
    confidence_threshold: &Array,
    compiled: Option<&CompiledFn>,
    target: StreamOrDevice,
) -> Result<Array> {
    if let Some(compiled) = compiled {
        let mut outputs = compiled.invoke(&[
            current_canvas,
            previous_canvas,
            token_entropy,
            confidence_threshold,
        ])?;
        if outputs.len() != 1 {
            return Err(anyhow!(
                "DiffusionGemma stable confidence chain returned {} outputs",
                outputs.len()
            ));
        }
        return Ok(outputs.pop().expect("checked output length"));
    }

    let same_tokens = current_canvas.equal_on(previous_canvas, target)?;
    let stable = mlx::ops::all_on(&same_tokens, mlx::ops::All, false, target)?;
    let mean_entropy = mlx::ops::mean_on(token_entropy, mlx::ops::All, false, target)?;
    let confident = mean_entropy.less_on(confidence_threshold, target)?;
    Ok(mlx::ops::indexing::where_on(
        &stable, &confident, &stable, target,
    )?)
}

#[cfg(test)]
mod tests {
    use super::super::ops::{
        build_entropy_probs_chain, build_entropy_transfer_mask_chain, build_stable_confidence_chain,
    };
    use super::*;
    use crate::core::constrained::{
        ConstraintTokenizer, ToolChoiceConstraint, ToolConstraintOptions,
    };
    use crate::core::tool_calling::ToolDefinition;

    fn gemma_tool_plan_and_tokens() -> (ConstraintPlan, Vec<u32>) {
        let tokenizer = ConstraintTokenizer::byte_level_gemma().unwrap();
        let plan = tokenizer
            .compile_gemma_tools(
                &[ToolDefinition {
                    name: "get_weather".to_owned(),
                    description: None,
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    }),
                    strict: None,
                }],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Required,
                    allow_parallel_calls: false,
                },
            )
            .unwrap();
        let mut expected = vec![256_u32];
        expected.extend(b"call:get_weather{city:".iter().copied().map(u32::from));
        expected.push(258);
        expected.extend(b"Tokyo".iter().copied().map(u32::from));
        expected.push(258);
        expected.push(u32::from(b'}'));
        expected.push(257);
        expected.push(261);
        (plan, expected)
    }

    fn logits_preferring(tokens: &[u32], vocab: usize) -> Array {
        let mut values = vec![-100.0_f32; tokens.len() * vocab];
        for (position, token) in tokens.iter().enumerate() {
            values[position * vocab + *token as usize] = 100.0;
        }
        (&values[..], &[1_i32, tokens.len() as i32, vocab as i32][..])
            .try_into()
            .unwrap()
    }

    #[test]
    fn multimodal_token_type_ids_mark_image_soft_tokens() {
        assert_eq!(
            multimodal_token_type_ids(&[10, 258_880, 258_880, 11], 258_880),
            vec![0, 1, 1, 0]
        );
    }

    #[test]
    fn diffusion_token_arrays_use_signed_int32_like_mlx_vlm() {
        let prompt_ids = [1_u32, 2, 3, 258_880];
        let input_ids = prompt_ids_array(&prompt_ids).unwrap();
        assert_eq!(input_ids.shape().as_slice(), &[1, 4]);
        assert_eq!(input_ids.dtype(), Dtype::Int32);
        assert_eq!(
            token_ids_from_canvas(&input_ids).unwrap(),
            prompt_ids.to_vec()
        );

        let key = random::key(7).unwrap();
        let canvas =
            sample_diffusion_canvas_on(16, 8, Some(&key), StreamOrDevice::default()).unwrap();
        assert_eq!(canvas.shape().as_slice(), &[1, 8]);
        assert_eq!(canvas.dtype(), Dtype::Int32);
        let global_canvas = sample_diffusion_canvas_on(16, 8, None, StreamOrDevice::default())
            .expect("global PRNG canvas");
        assert_eq!(global_canvas.shape().as_slice(), &[1, 8]);
        assert_eq!(global_canvas.dtype(), Dtype::Int32);

        let logits: Array = (
            &[0.1_f32, 0.9, 0.8, 0.2, 2.0, 1.0][..],
            &[1_i32, 3_i32, 2_i32][..],
        )
            .try_into()
            .unwrap();
        let argmax = argmax_canvas_on(&logits, StreamOrDevice::default()).unwrap();
        assert_eq!(argmax.dtype(), Dtype::Int32);
        assert_eq!(token_ids_from_canvas(&argmax).unwrap(), vec![1, 0, 0]);
    }

    #[test]
    fn constrained_canvas_projects_one_prefix_legal_gemma_path_and_eos() {
        let (plan, expected) = gemma_tool_plan_and_tokens();
        let vocab = 262_usize;
        let logits = logits_preferring(&expected, vocab);
        let session = plan.start_session().unwrap();
        let mut rng = None;
        let projected = constrained_canvas_on(
            &logits,
            &session,
            false,
            &mut rng,
            StreamOrDevice::default(),
        )
        .unwrap();
        assert_eq!(token_ids_from_canvas(&projected.canvas).unwrap(), expected);

        let masked = apply_diffusion_token_masks(&logits, &projected.masks).unwrap();
        assert_eq!(
            mlx::ops::reduction::argmax(&masked, -1, false)
                .unwrap()
                .to_vec::<u32>()
                .unwrap(),
            expected
        );
    }

    #[test]
    fn constrained_canvas_carries_committed_grammar_state_across_canvases() {
        let (plan, expected) = gemma_tool_plan_and_tokens();
        let split = 19_usize;
        let mut committed = plan.start_session().unwrap();
        let mut rng = None;

        let first = constrained_canvas_on(
            &logits_preferring(&expected[..split], 262),
            &committed,
            false,
            &mut rng,
            StreamOrDevice::default(),
        )
        .unwrap();
        let first_tokens = token_ids_from_canvas(&first.canvas).unwrap();
        assert_eq!(first_tokens, expected[..split]);
        committed.commit_tokens(&first_tokens).unwrap();
        assert!(!committed.is_accepting().unwrap());

        let second = constrained_canvas_on(
            &logits_preferring(&expected[split..], 262),
            &committed,
            false,
            &mut rng,
            StreamOrDevice::default(),
        )
        .unwrap();
        let second_tokens = token_ids_from_canvas(&second.canvas).unwrap();
        assert_eq!(second_tokens, expected[split..]);
        committed.commit_tokens(&second_tokens).unwrap();
        assert!(committed.is_accepting().unwrap());
    }

    #[test]
    fn constrained_canvas_sampling_stays_on_a_complete_legal_path() {
        let (plan, expected) = gemma_tool_plan_and_tokens();
        let logits = logits_preferring(&expected, 262);
        let session = plan.start_session().unwrap();
        let mut rng = Some(random::key(7).unwrap());

        let projected =
            constrained_canvas_on(&logits, &session, true, &mut rng, StreamOrDevice::default())
                .unwrap();
        let sampled = token_ids_from_canvas(&projected.canvas).unwrap();
        assert_eq!(sampled, expected);
        assert_eq!(
            session
                .validate_tokens(&sampled[..sampled.len() - 1])
                .unwrap(),
            sampled.len() - 1
        );
    }

    #[test]
    fn entropy_transfer_mask_builds_lazy_batched_mlx_graph() {
        let entropy: Array = (
            &[0.01_f32, 0.03, 0.20, 0.04, 0.30, 0.10, 0.11, 0.12][..],
            &[2_i32, 4_i32][..],
        )
            .try_into()
            .unwrap();

        let entropy_bound: Array = (&[0.03_f32][..], ()).try_into().unwrap();

        let mask =
            entropy_transfer_mask_on(&entropy, &entropy_bound, None, StreamOrDevice::default())
                .unwrap();

        assert_eq!(mask.shape().as_slice(), &[2, 4]);
        assert!(
            format!("{mask:?}").contains("evaluated: false"),
            "entropy transfer mask should remain lazy until the generation loop synchronizes"
        );
        assert_eq!(
            mask.to_vec::<bool>().unwrap(),
            vec![true, true, false, false, false, true, false, false]
        );
    }

    #[test]
    fn entropy_transfer_mask_selects_lowest_entropy_token_below_bound() {
        let entropy: Array = (&[0.10_f32, 0.11, 0.12][..], &[1_i32, 3_i32][..])
            .try_into()
            .unwrap();
        let entropy_bound: Array = (&[0.03_f32][..], ()).try_into().unwrap();

        let mask =
            entropy_transfer_mask_on(&entropy, &entropy_bound, None, StreamOrDevice::default())
                .unwrap();

        assert_eq!(mask.to_vec::<bool>().unwrap(), vec![true, false, false]);
    }

    #[test]
    fn compiled_entropy_transfer_mask_matches_eager_mask() {
        let entropy: Array = (
            &[0.02_f32, 0.01, 0.12, 0.07, 0.05, 0.06, 0.20, 0.08][..],
            &[2_i32, 4_i32][..],
        )
            .try_into()
            .unwrap();
        let entropy_bound: Array = (&[0.04_f32][..], ()).try_into().unwrap();

        let compiled = build_entropy_transfer_mask_chain();
        let mask = entropy_transfer_mask_on(
            &entropy,
            &entropy_bound,
            Some(&compiled),
            StreamOrDevice::default(),
        )
        .unwrap();
        let eager =
            entropy_transfer_mask_on(&entropy, &entropy_bound, None, StreamOrDevice::default())
                .unwrap();

        assert_eq!(
            mask.to_vec::<bool>().unwrap(),
            eager.to_vec::<bool>().unwrap()
        );
    }

    #[test]
    fn compiled_entropy_probs_chain_matches_eager_entropy() {
        let logits: Array = (&[0.5_f32, 1.5, -0.5, 0.25][..], &[1_i32, 2_i32, 2_i32][..])
            .try_into()
            .unwrap();

        let compiled = build_entropy_probs_chain();
        let (entropy, probs) =
            entropy_probs_chain_on(&logits, Some(&compiled), StreamOrDevice::default()).unwrap();
        let (eager_entropy, eager_probs) =
            entropy_probs_chain_on(&logits, None, StreamOrDevice::default()).unwrap();

        assert_eq!(entropy.shape().as_slice(), &[1, 2]);
        assert_eq!(probs.shape().as_slice(), &[1, 2, 2]);
        assert_eq!(
            entropy.to_vec::<f32>().unwrap(),
            eager_entropy.to_vec::<f32>().unwrap()
        );
        assert_eq!(
            probs.to_vec::<f32>().unwrap(),
            eager_probs.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn entropy_soft_embeddings_preserve_embedding_dtype_after_scale() {
        let logits: Array = (&[0.5_f32, 1.5, -0.5][..], &[1_i32, 1_i32, 3_i32][..])
            .try_into()
            .unwrap();
        let embedding_weight: Array = (
            &[1.0_f32, 0.0, 0.0, 1.0, 0.5, -0.5][..],
            &[3_i32, 2_i32][..],
        )
            .try_into()
            .unwrap();
        let embedding_weight = embedding_weight
            .astype_on(Dtype::Bfloat16, StreamOrDevice::default())
            .unwrap();

        let (_, soft_embeddings) = entropy_and_soft_embeddings_on(
            &logits,
            &embedding_weight,
            2.0,
            None,
            StreamOrDevice::default(),
        )
        .unwrap();

        assert_eq!(soft_embeddings.dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn stable_and_confident_keeps_canvas_history_on_device() {
        let logits: Array = (
            &[0.5_f32, 1.5, 1.5, 0.5, 0.25, 0.75, 0.75, 0.25][..],
            &[1_i32, 4_i32, 2_i32][..],
        )
            .try_into()
            .unwrap();
        let argmax_canvas =
            mlx::ops::argmax_on(&logits, -1_i32, false, StreamOrDevice::default()).unwrap();
        let config = DiffusionGemmaGenerationConfig {
            max_denoising_steps: 48,
            max_new_tokens: 256,
            t_min: 0.4,
            t_max: 0.8,
            confidence_threshold: 0.005,
            stability_threshold: 1,
            eos_token_id: None,
            sampler_config: None,
        };
        let mut history = Vec::new();

        let stable = stable_and_confident_on(
            &argmax_canvas,
            &logits,
            &mut history,
            &config,
            None,
            None,
            StreamOrDevice::default(),
        )
        .unwrap();

        assert!(!stable);
        assert_eq!(history.len(), 1);
        assert!(
            format!("{argmax_canvas:?}").contains("evaluated: false"),
            "stable check should not force the full argmax canvas back to CPU"
        );
    }

    #[test]
    fn compiled_stable_confidence_chain_matches_threshold_one_stop_condition() {
        let previous: Array = (&[1_i32, 2, 3, 4][..], &[1_i32, 4_i32][..])
            .try_into()
            .unwrap();
        let current_same = previous.clone();
        let current_changed: Array = (&[1_i32, 2, 3, 5][..], &[1_i32, 4_i32][..])
            .try_into()
            .unwrap();
        let low_entropy: Array = (&[0.001_f32, 0.002, 0.003, 0.004][..], &[1_i32, 4_i32][..])
            .try_into()
            .unwrap();
        let high_entropy: Array = (&[0.01_f32, 0.02, 0.03, 0.04][..], &[1_i32, 4_i32][..])
            .try_into()
            .unwrap();
        let threshold =
            super::super::ops::scalar_array_like_on(0.005, &low_entropy, StreamOrDevice::default())
                .unwrap();
        let compiled = build_stable_confidence_chain();

        let should_stop = stable_confident_threshold_one_on(
            &current_same,
            &previous,
            &low_entropy,
            &threshold,
            Some(&compiled),
            StreamOrDevice::default(),
        )
        .unwrap();
        let changed = stable_confident_threshold_one_on(
            &current_changed,
            &previous,
            &low_entropy,
            &threshold,
            Some(&compiled),
            StreamOrDevice::default(),
        )
        .unwrap();
        let uncertain = stable_confident_threshold_one_on(
            &current_same,
            &previous,
            &high_entropy,
            &threshold,
            Some(&compiled),
            StreamOrDevice::default(),
        )
        .unwrap();

        assert!(should_stop.item::<bool>().unwrap());
        assert!(!changed.item::<bool>().unwrap());
        assert!(!uncertain.item::<bool>().unwrap());
    }

    #[test]
    fn generation_stream_guard_restores_default_stream() {
        let original_device = mlx::default_device();
        let original_stream = mlx::default_stream(original_device);

        {
            let (target, _guard) = enter_diffusion_generation_stream().unwrap();
            assert!(matches!(target, StreamOrDevice::ThreadLocalStream(_)));
            assert_eq!(mlx::default_device(), original_device);
            assert_ne!(mlx::default_stream(original_device), original_stream);
        }

        assert_eq!(mlx::default_device(), original_device);
        assert_eq!(mlx::default_stream(original_device), original_stream);
    }
}
