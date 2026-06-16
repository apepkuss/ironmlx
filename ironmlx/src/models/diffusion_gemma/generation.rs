use std::collections::HashSet;

use anyhow::{anyhow, Context};
use mlx::{random, Array, Dtype, StreamOrDevice};

use crate::core::loader::EosTokenId;
use crate::core::Tokenizer;
use crate::Result;

use super::config::DiffusionGemmaGenerationConfig;
use super::model::{DiffusionGemmaEncoderInputs, DiffusionGemmaModel};

const DEFAULT_MIN_CANVAS_LENGTH: i32 = 64;

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

pub fn generate_text(
    model: &DiffusionGemmaModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    generation_config: &DiffusionGemmaGenerationConfig,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
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
    seed: u64,
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
    seed: u64,
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
    seed: u64,
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
    seed: u64,
    multimodal: Option<DiffusionGemmaMultimodalInput<'_>>,
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
    let target = StreamOrDevice::default();
    if max_new_tokens == 0 {
        return Ok(());
    }
    let max_denoising_steps = generation_config.max_denoising_steps;
    let entropy_bound = generation_config.entropy_bound()?;
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

    let input_ids: Array = (prompt_ids, &[1_i32, prompt_ids.len() as i32][..]).try_into()?;
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
    let canvas_cap = model.config.canvas_length;
    let min_canvas_length = canvas_cap.clamp(1, DEFAULT_MIN_CANVAS_LENGTH);
    let vocab_size = model.config.text_config.vocab_size;
    let mut rng = random::key(seed)?;
    let mut current_canvas: Option<Array> = None;
    let mut generated = 0usize;
    let mut detok = tokenizer.decode_stream(true);

    while generated < max_new_tokens {
        if let Some(canvas) = current_canvas.as_ref() {
            model.encode_tokens_on(canvas, &mut cache, target)?;
        }

        let remaining = (max_new_tokens - generated) as i32;
        let canvas_len = canvas_cap.min(remaining.max(min_canvas_length));
        let (next_rng, sample_key) = random::split(&rng)?;
        rng = next_rng;
        let mut canvas = random::randint()
            .low(0)
            .high(vocab_size as i64)
            .shape((1_i32, canvas_len))
            .dtype(Dtype::Uint32)
            .key(&sample_key)
            .stream(target)
            .sample()
            .context("DiffusionGemma: initialize canvas")?;
        let mut argmax_canvas = canvas.clone();
        let mut self_conditioning: Option<Array> = None;
        let mut stability_history: Vec<Vec<u32>> = Vec::new();

        for cur_step in (1..=max_denoising_steps).rev() {
            let mut logits =
                model.decode_logits_on(&canvas, &cache, self_conditioning.as_ref(), target)?;
            let schedule_temperature =
                linear_temperature(cur_step, max_denoising_steps, generation_config);
            let schedule: Array = (&[schedule_temperature][..], ()).try_into()?;
            logits = logits.try_div_on(&schedule, target)?;

            argmax_canvas = mlx::ops::argmax_on(&logits, -1_i32, false, target)?;
            if cur_step == 1 {
                break;
            }

            let denoiser_canvas = if temperature <= 0.0 {
                argmax_canvas.clone()
            } else {
                let temp: Array = (&[temperature][..], ()).try_into()?;
                let sample_logits = logits.try_div_on(&temp, target)?;
                let (next_rng, sample_key) = random::split(&rng)?;
                rng = next_rng;
                random::categorical(&sample_logits)
                    .axis(-1)
                    .key(&sample_key)
                    .stream(target)
                    .sample()?
            };

            let (entropy, next_self_conditioning) = entropy_and_soft_embeddings_on(
                &logits,
                &soft_embedding_weight,
                model.embed_scale(),
                target,
            )?;
            let acceptance_mask = entropy_transfer_mask(&entropy, entropy_bound)?;
            let accepted =
                mlx::ops::indexing::where_on(&acceptance_mask, &denoiser_canvas, &canvas, target)?;
            let (next_rng, sample_key) = random::split(&rng)?;
            rng = next_rng;
            let random_canvas = random::randint()
                .low(0)
                .high(vocab_size as i64)
                .shape((1_i32, canvas_len))
                .dtype(Dtype::Uint32)
                .key(&sample_key)
                .stream(target)
                .sample()?;
            canvas =
                mlx::ops::indexing::where_on(&acceptance_mask, &accepted, &random_canvas, target)?;

            if stable_and_confident(
                &argmax_canvas,
                &logits,
                &mut stability_history,
                generation_config,
            )? {
                break;
            }
            self_conditioning = Some(next_self_conditioning);
        }

        current_canvas = Some(argmax_canvas.clone());
        mlx::transforms::eval(&[&argmax_canvas]).context("DiffusionGemma: eval canvas")?;
        let token_ids: Vec<u32> = argmax_canvas.to_vec()?;
        for token in token_ids {
            generated += 1;
            if stop_ids.contains(&token) {
                let keep_going = emit(DiffusionGemmaGenerateEvent {
                    token,
                    text: String::new(),
                    finish_reason: Some("stop"),
                })?;
                let _ = keep_going;
                return Ok(());
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

    let _ = emit(DiffusionGemmaGenerateEvent {
        token: 0,
        text: String::new(),
        finish_reason: Some("length"),
    })?;
    Ok(())
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

fn entropy_and_soft_embeddings_on(
    logits: &Array,
    embedding_weight: &Array,
    embed_scale: f32,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    let logits = logits.astype_on(Dtype::Float32, target)?;
    let lse = mlx::ops::logsumexp_on(&logits, -1_i32, true, target)?;
    let log_probs = &logits - &lse;
    let probs = log_probs.exp_on(target)?;
    let entropy_terms = &probs * &log_probs;
    let entropy = mlx::ops::sum_on(&entropy_terms, -1_i32, false, target)?;
    let entropy = entropy.try_mul_on(&(&[-1.0_f32][..], ()).try_into()?, target)?;
    let probs = probs.astype_on(embedding_weight.dtype(), target)?;
    let soft = probs.matmul_on(embedding_weight, target)?;
    let soft = &soft * embed_scale;
    Ok((entropy, soft))
}

fn entropy_transfer_mask(entropy: &Array, entropy_bound: f32) -> Result<Array> {
    let shape = entropy.shape();
    let dims = shape.as_slice();
    if dims != [1, dims[1]] {
        return Err(anyhow!(
            "DiffusionGemma entropy mask expects [1,L], got {:?}",
            dims
        ));
    }
    let values: Vec<f32> = entropy.to_vec()?;
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut cumulative = 0.0_f32;
    let mut cumulative_max = f32::NEG_INFINITY;
    let mut mask = vec![false; values.len()];
    for idx in order {
        let v = values[idx];
        cumulative += v;
        cumulative_max = cumulative_max.max(v);
        if cumulative - cumulative_max <= entropy_bound {
            mask[idx] = true;
        }
    }
    Ok((mask.as_slice(), &[1_i32, values.len() as i32][..]).try_into()?)
}

fn stable_and_confident(
    argmax_canvas: &Array,
    logits: &Array,
    history: &mut Vec<Vec<u32>>,
    config: &DiffusionGemmaGenerationConfig,
) -> Result<bool> {
    let current: Vec<u32> = argmax_canvas.to_vec()?;
    let stable =
        history.len() == config.stability_threshold && history.iter().all(|prev| prev == &current);
    history.push(current);
    if history.len() > config.stability_threshold {
        history.remove(0);
    }
    if !stable {
        return Ok(false);
    }
    let logits = logits.astype(Dtype::Float32)?;
    let lse = mlx::ops::logsumexp(&logits, -1_i32, true)?;
    let log_probs = &logits - &lse;
    let probs = log_probs.exp()?;
    let entropy = mlx::ops::sum(&(&probs * &log_probs), -1_i32, false)?;
    let entropy = entropy.try_mul(&(&[-1.0_f32][..], ()).try_into()?)?;
    let values: Vec<f32> = entropy.to_vec()?;
    let mean = values.iter().copied().sum::<f32>() / values.len().max(1) as f32;
    Ok(mean < config.confidence_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_token_type_ids_mark_image_soft_tokens() {
        assert_eq!(
            multimodal_token_type_ids(&[10, 258_880, 258_880, 11], 258_880),
            vec![0, 1, 1, 0]
        );
    }
}
