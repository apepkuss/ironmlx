use std::collections::VecDeque;
use std::time::Instant;

use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::generate::{
    build_position_ids, build_position_ids_vl, count_image_pad, extend_vl_chunk_end_for_image_pad,
    slice_pos_ids_axis2, slice_vision_embeds_rows, GenerateEvent, GenerateRequest,
};
use crate::core::scheduler::DenseVlMethods;
use crate::core::speculative::{
    add_elapsed_us, adjust_mtp_draft_budget, resolve_speculative_tokens, restore_layer_cache,
    sample_logits_positions, slice_hidden_position, verify_input, MtpSpeculativeConfig,
    MtpSpeculativeStats,
};
use crate::core::tokenizer::{DecodeStream, Tokenizer};
use crate::core::{Loader, Model};
use crate::nn::{enable_turboquant_kv_caches, LayerCache, LayerCacheSnapshot, Linear};
use crate::Result;

use super::config::{Gemma4AssistantConfig, Gemma4LayerKind};
use super::model::Gemma4Model;
use super::text_model::{Gemma4SharedKvStates, Gemma4TextModel};

pub struct Gemma4DrafterMasks {
    sliding: Option<Array>,
    full: Option<Array>,
}

impl Gemma4DrafterMasks {
    pub fn get(&self, kind: Gemma4LayerKind) -> Option<&Array> {
        match kind {
            Gemma4LayerKind::Sliding => self.sliding.as_ref(),
            Gemma4LayerKind::Full => self.full.as_ref(),
        }
    }
}

pub struct Gemma4DrafterStepOutput {
    pub hidden_states: Array,
    pub logits: Array,
}

pub struct Gemma4AssistantModel {
    cfg: Gemma4AssistantConfig,
    text: Gemma4TextModel,
    pre_projection: Linear,
    post_projection: Linear,
    masked_embedding: Option<MaskedEmbedder>,
}

impl Gemma4AssistantModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = Gemma4AssistantConfig::from_loader(loader)?;
        let text =
            Gemma4TextModel::from_loader_external_shared_kv(loader, cfg.text_config.clone())?;
        let pre_projection = Linear::from_loader(loader, "pre_projection")?;
        let post_projection = Linear::from_loader(loader, "post_projection")?;
        let masked_embedding = if cfg.use_ordered_embeddings {
            Some(MaskedEmbedder::from_loader(loader, &cfg)?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            text,
            pre_projection,
            post_projection,
            masked_embedding,
        })
    }

    pub fn config(&self) -> &Gemma4AssistantConfig {
        &self.cfg
    }

    pub fn forward_on(
        &self,
        inputs_embeds: &Array,
        shared_kv: &Gemma4SharedKvStates,
        position: i32,
        kv_valid_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Gemma4DrafterStepOutput> {
        let target = target.into();
        let h = self.pre_projection.forward_on(inputs_embeds, target)?;
        let shape = h.shape();
        let dims = shape.as_slice();
        if dims.len() != 3 {
            return Err(anyhow!(
                "Gemma4AssistantModel::forward_on: expected hidden [B,S,H], got {dims:?}"
            ));
        }
        let masks = make_drafter_masks(
            shared_kv,
            dims[1],
            position,
            self.cfg.text_config.sliding_window,
            h.dtype(),
            kv_valid_len,
            target,
        )?;
        let h = self
            .text
            .forward_external_shared_kv_on(&h, shared_kv, &masks, position, target)?;
        let hidden_states = self.post_projection.forward_on(&h, target)?;
        let logits = match self.masked_embedding.as_ref() {
            Some(masked) => {
                let weight = self.text_embedding_dense_weight_on(target)?;
                masked.forward_on(&h, &weight, target)?
            }
            None => self.text.as_output_on(&h, target)?,
        };
        Ok(Gemma4DrafterStepOutput {
            hidden_states,
            logits,
        })
    }

    fn text_embedding_dense_weight_on(&self, target: StreamOrDevice) -> Result<Array> {
        self.text.dense_embedding_weight_on(target)
    }
}

pub struct Gemma4DrafterGenerationStream<'m> {
    model: &'m Gemma4Model,
    drafter: &'m Gemma4AssistantModel,
    cache: Vec<LayerCache>,
    history: Vec<u32>,
    request: GenerateRequest,
    cfg: MtpSpeculativeConfig,
    pending_tokens: VecDeque<u32>,
    detok: DecodeStream<'m>,
    /// Hidden state for the token immediately before the current pending token.
    last_hidden: Array,
    shared_kv: Gemma4SharedKvStates,
    emitted_new_tokens: usize,
    finished: bool,
    dummy_position_ids: Option<Array>,
    prng_state: Array,
    adaptive_draft_tokens: usize,
    stats: MtpSpeculativeStats,
    trace_window_limit: usize,
    trace_windows: Vec<Gemma4DrafterTraceWindow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4DrafterTraceWindow {
    pub history_len: usize,
    pub verify_start_pos: i32,
    pub draft_tokens: Vec<u32>,
    pub verified_tokens: Vec<u32>,
    pub accepted_draft_len: usize,
}

impl<'m> Gemma4DrafterGenerationStream<'m> {
    pub fn new(
        model: &'m Gemma4Model,
        drafter: &'m Gemma4AssistantModel,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Self> {
        if request.prompt_ids.is_empty() {
            return Err(anyhow!(
                "Gemma4DrafterGenerationStream::new: prompt_ids cannot be empty"
            ));
        }
        if cfg.max_draft_tokens == 0 {
            return Err(anyhow!(
                "Gemma4DrafterGenerationStream::new: max_draft_tokens must be > 0"
            ));
        }
        if !request.sampler.is_pipelinable() {
            return Err(anyhow!(
                "Gemma4DrafterGenerationStream::new: Gemma4 drafter decoding currently requires greedy sampling"
            ));
        }
        if request.pixel_values.is_none() && request.image_grid_thw.is_some() {
            return Err(anyhow!(
                "Gemma4DrafterGenerationStream::new: image_grid_thw present but pixel_values is None"
            ));
        }

        let prompt_len = request.prompt_ids.len();
        let cap = ((prompt_len + request.max_new_tokens) as i32)
            .max(crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let dtype = model.cache_dtype();
        let mut cache = model.make_cache(1, cap, dtype)?;
        if let Some(bits) = request.kv_cache_turboquant_bits {
            enable_turboquant_kv_caches(&mut cache, bits)?;
        }
        let dummy_position_ids = if model.requires_position_ids() {
            None
        } else {
            Some(build_position_ids(0, 1)?)
        };

        let mut stats = MtpSpeculativeStats::default();
        let mut pos = 0_i32;
        let prompt_len_i32 = prompt_len as i32;
        let mut image_pad_consumed = 0usize;
        let is_vl = request.pixel_values.is_some();
        let mut vision_embeds_full = None;
        let position_ids_full = if is_vl && dummy_position_ids.is_none() {
            let grids = request.image_grid_thw.as_deref().ok_or_else(|| {
                anyhow!("Gemma4DrafterGenerationStream::new: pixel_values present but image_grid_thw is None")
            })?;
            if model.vl_positions_sequential() {
                Some(build_position_ids(0, prompt_len_i32)?)
            } else {
                let prompt_ids_i32: Vec<i32> =
                    request.prompt_ids.iter().map(|&id| id as i32).collect();
                Some(build_position_ids_vl(
                    &prompt_ids_i32,
                    grids,
                    request.image_token_id,
                    request.image_spatial_merge_size,
                )?)
            }
        } else {
            None
        };
        let mut last_prompt_hidden = None;
        let mut last_shared_kv = None;

        while pos < prompt_len_i32 {
            let remaining = prompt_len_i32 - pos;
            let mut n = if request.prefill_chunk_size == 0 {
                remaining
            } else {
                remaining.min(request.prefill_chunk_size as i32)
            };
            if n <= 0 {
                return Err(anyhow!(
                    "Gemma4DrafterGenerationStream::new: invalid prefill chunk length {n}"
                ));
            }
            if is_vl && request.prefill_chunk_size != 0 {
                let adjusted_end = extend_vl_chunk_end_for_image_pad(
                    &request.prompt_ids,
                    request.image_token_id,
                    pos,
                    pos + n,
                );
                n = adjusted_end - pos;
            }

            let chunk_ids = &request.prompt_ids[pos as usize..(pos as usize + n as usize)];
            let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;
            let chunk_pos_ids = match (dummy_position_ids.as_ref(), position_ids_full.as_ref()) {
                (Some(dummy), _) => dummy.clone(),
                (None, Some(full)) => slice_pos_ids_axis2(full, pos, pos + n)?,
                (None, None) => build_position_ids(pos, n)?,
            };

            let forward_start = Instant::now();
            let out = if is_vl {
                let image_tokens = count_image_pad(chunk_ids, request.image_token_id);
                if image_tokens > 0 && vision_embeds_full.is_none() {
                    let pixel_values = request.pixel_values.as_deref().ok_or_else(|| {
                        anyhow!(
                            "Gemma4DrafterGenerationStream::new: image tokens without pixel_values"
                        )
                    })?;
                    let grid_thw = request.image_grid_thw.as_deref().ok_or_else(|| {
                        anyhow!("Gemma4DrafterGenerationStream::new: image tokens without image_grid_thw")
                    })?;
                    vision_embeds_full =
                        Some(model.compute_vision_embeds(pixel_values, grid_thw, ().into())?);
                }
                let vision_slice = match vision_embeds_full.as_ref() {
                    Some(ve) if image_tokens > 0 => Some(slice_vision_embeds_rows(
                        ve,
                        image_pad_consumed,
                        image_pad_consumed + image_tokens,
                    )?),
                    _ => None,
                };
                image_pad_consumed += image_tokens;
                model.forward_vl_hidden_with_shared_kv_on(
                    &chunk_arr,
                    &chunk_pos_ids,
                    None,
                    None,
                    Some(&mut cache),
                    vision_slice.as_ref(),
                    request.image_token_id,
                    ().into(),
                )?
            } else {
                model.forward_text_hidden_with_shared_kv_on(
                    &chunk_arr,
                    &chunk_pos_ids,
                    None,
                    None,
                    Some(&mut cache),
                    (),
                )?
            };
            add_elapsed_us(&mut stats.verify_forward_us, forward_start);
            let chunk_last_hidden = slice_hidden_position(&out.hidden, n - 1)?;
            if pos + n == prompt_len_i32 {
                last_prompt_hidden = Some(chunk_last_hidden);
                last_shared_kv = Some(out.shared_kv);
            } else {
                mlx::transforms::eval(&[&out.hidden])?;
            }
            pos += n;
        }

        let last_prompt_hidden = last_prompt_hidden
            .ok_or_else(|| anyhow!("Gemma4 drafter prefill produced no prompt hidden"))?;
        let shared_kv = last_shared_kv
            .ok_or_else(|| anyhow!("Gemma4 drafter prefill produced no shared KV"))?;
        let projection_start = Instant::now();
        let first_logits = model.project_hidden_on(&last_prompt_hidden, ())?;
        add_elapsed_us(&mut stats.projection_us, projection_start);
        let mut prng_state = mlx::random::key(request.sampler.seed)?;
        let sampling_start = Instant::now();
        let first_tokens = sample_logits_positions(
            &first_logits,
            request.sampler,
            &request.prompt_ids,
            &mut prng_state,
        )?;
        add_elapsed_us(&mut stats.sampling_us, sampling_start);
        let first_token = *first_tokens
            .first()
            .ok_or_else(|| anyhow!("Gemma4 drafter prefill produced no first token"))?;

        let mut history = request.prompt_ids.clone();
        history.push(first_token);
        let mut pending_tokens = VecDeque::new();
        pending_tokens.push_back(first_token);

        Ok(Self {
            model,
            drafter,
            cache,
            history,
            request,
            cfg,
            pending_tokens,
            detok: tokenizer.decode_stream(true),
            last_hidden: last_prompt_hidden,
            shared_kv,
            emitted_new_tokens: 0,
            finished: false,
            dummy_position_ids,
            prng_state,
            adaptive_draft_tokens: cfg.max_draft_tokens,
            stats,
            trace_window_limit: 0,
            trace_windows: Vec::new(),
        })
    }

    pub fn stats(&self) -> MtpSpeculativeStats {
        self.stats.clone()
    }

    pub fn set_trace_window_limit(&mut self, limit: usize) {
        self.trace_window_limit = limit;
        self.trace_windows.truncate(limit);
    }

    pub fn trace_windows(&self) -> &[Gemma4DrafterTraceWindow] {
        &self.trace_windows
    }

    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }

        let token = self
            .pending_tokens
            .pop_front()
            .ok_or_else(|| anyhow!("Gemma4 drafter stream invariant: pending queue is empty"))?;
        self.emitted_new_tokens += 1;
        let text = self.detok.step(token)?.unwrap_or_default();
        let finish_reason = if self.request.stop_token_ids.contains(&token) {
            Some("stop")
        } else if self.emitted_new_tokens >= self.request.max_new_tokens {
            Some("length")
        } else {
            None
        };

        if finish_reason.is_some() {
            self.finished = true;
            return Ok(Some(GenerateEvent {
                token,
                text,
                finish_reason,
            }));
        }

        if self.pending_tokens.is_empty() {
            self.fill_window(token)?;
        }

        Ok(Some(GenerateEvent {
            token,
            text,
            finish_reason: None,
        }))
    }

    fn fill_window(&mut self, current_token: u32) -> Result<()> {
        let remaining = self
            .request
            .max_new_tokens
            .saturating_sub(self.emitted_new_tokens);
        if remaining == 0 {
            return Ok(());
        }

        let draft_budget = self
            .adaptive_draft_tokens
            .clamp(1, self.cfg.max_draft_tokens)
            .min(remaining);
        let draft_tokens = self.draft_tokens(current_token, draft_budget)?;
        let verify_input = verify_input(current_token, &draft_tokens);
        let verify_start_pos = (self.history.len() - 1) as i32;
        let verify_pos_ids = self.position_ids(verify_start_pos, verify_input.len() as i32)?;
        let verify_arr: Array =
            (&verify_input[..], &[1_i32, verify_input.len() as i32][..]).try_into()?;

        let base_snapshot: Vec<LayerCacheSnapshot> =
            self.cache.iter().map(LayerCache::snapshot).collect();
        let verify_forward_start = Instant::now();
        let verified = self.model.forward_text_hidden_with_shared_kv_on(
            &verify_arr,
            &verify_pos_ids,
            None,
            None,
            Some(&mut self.cache),
            (),
        )?;
        add_elapsed_us(&mut self.stats.verify_forward_us, verify_forward_start);
        let projection_start = Instant::now();
        let verified_logits = self.model.project_hidden_on(&verified.hidden, ())?;
        add_elapsed_us(&mut self.stats.projection_us, projection_start);
        let sampling_start = Instant::now();
        let verified_tokens = sample_logits_positions(
            &verified_logits,
            self.request.sampler,
            &self.history,
            &mut self.prng_state,
        )?;
        add_elapsed_us(&mut self.stats.sampling_us, sampling_start);

        let resolution = resolve_speculative_tokens(&draft_tokens, &verified_tokens)?;
        if self.trace_windows.len() < self.trace_window_limit {
            self.trace_windows.push(Gemma4DrafterTraceWindow {
                history_len: self.history.len(),
                verify_start_pos,
                draft_tokens: draft_tokens.clone(),
                verified_tokens: verified_tokens.clone(),
                accepted_draft_len: resolution.accepted_draft_len,
            });
        }
        self.stats.windows += 1;
        self.stats.drafted_tokens += draft_tokens.len();
        self.stats.accepted_draft_tokens += resolution.accepted_draft_len;
        self.stats
            .record_window_acceptance(draft_tokens.len(), resolution.accepted_draft_len);
        if resolution.needs_rollback {
            self.stats.rollback_count += 1;
        }
        adjust_mtp_draft_budget(
            self.cfg.max_draft_tokens,
            &mut self.adaptive_draft_tokens,
            draft_tokens.len(),
            resolution.accepted_draft_len,
            &mut self.stats,
        );

        let (accepted_last_hidden, accepted_shared_kv) = if resolution.needs_rollback {
            let rollback_start = Instant::now();
            restore_layer_cache(&mut self.cache, &base_snapshot)?;
            add_elapsed_us(&mut self.stats.main_rollback_us, rollback_start);
            let replay_len = resolution.accepted_verify_input_len;
            let replay_input = &verify_input[..replay_len];
            let replay_arr: Array = (replay_input, &[1_i32, replay_len as i32][..]).try_into()?;
            let replay_pos_ids = self.position_ids(verify_start_pos, replay_len as i32)?;
            let replay_forward_start = Instant::now();
            let replay = self.model.forward_text_hidden_with_shared_kv_on(
                &replay_arr,
                &replay_pos_ids,
                None,
                None,
                Some(&mut self.cache),
                (),
            )?;
            add_elapsed_us(&mut self.stats.verify_forward_us, replay_forward_start);
            (
                slice_hidden_position(&replay.hidden, replay_len as i32 - 1)?,
                replay.shared_kv,
            )
        } else {
            (
                slice_hidden_position(
                    &verified.hidden,
                    resolution.accepted_verify_input_len as i32 - 1,
                )?,
                verified.shared_kv,
            )
        };
        self.last_hidden = accepted_last_hidden;
        self.shared_kv = accepted_shared_kv;

        let mut tokens_to_append = resolution.tokens_to_append;
        if let Some(stop_idx) = tokens_to_append
            .iter()
            .position(|token| self.request.stop_token_ids.contains(token))
        {
            tokens_to_append.truncate(stop_idx + 1);
        }
        tokens_to_append.truncate(remaining);
        for token in tokens_to_append {
            self.history.push(token);
            self.pending_tokens.push_back(token);
        }
        Ok(())
    }

    fn draft_tokens(&mut self, current_token: u32, draft_budget: usize) -> Result<Vec<u32>> {
        let mut draft_tokens = Vec::with_capacity(draft_budget);
        let mut draft_history = self.history.clone();
        let mut input_hidden = self.last_hidden.clone();
        let mut input_token = current_token;
        let kv_valid_len = (self.history.len() - 1) as i32;
        let draft_position = draft_position_for_shared_kv(kv_valid_len);

        for _ in 0..draft_budget {
            let token_arr: Array = (&[input_token][..], &[1_i32, 1_i32][..]).try_into()?;
            let token_embed = self.model.embed_on(&token_arr, ())?;
            let inputs_embeds =
                mlx::ops::shape::concatenate_on(&[&token_embed, &input_hidden], 2, ())?;
            let draft_forward_start = Instant::now();
            let output = self.drafter.forward_on(
                &inputs_embeds,
                &self.shared_kv,
                draft_position,
                kv_valid_len,
                (),
            )?;
            add_elapsed_us(&mut self.stats.draft_forward_us, draft_forward_start);
            let sampling_start = Instant::now();
            let sampled = sample_logits_positions(
                &output.logits,
                self.request.sampler,
                &draft_history,
                &mut self.prng_state,
            )?;
            add_elapsed_us(&mut self.stats.sampling_us, sampling_start);
            let next_token = *sampled
                .first()
                .ok_or_else(|| anyhow!("Gemma4 drafter produced no token"))?;
            draft_tokens.push(next_token);
            draft_history.push(next_token);
            input_hidden = output.hidden_states;
            input_token = next_token;
        }

        Ok(draft_tokens)
    }

    fn position_ids(&self, start_pos: i32, len: i32) -> Result<Array> {
        match self.dummy_position_ids.as_ref() {
            Some(dummy) => Ok(dummy.clone()),
            None => build_position_ids(start_pos, len),
        }
    }
}

struct MaskedEmbedder {
    centroids: Linear,
    token_ordering: Array,
    hidden_size: i32,
    vocab_size: i32,
    num_centroids: i32,
    top_k: i32,
    vocab_size_per_centroid: i32,
}

impl MaskedEmbedder {
    fn from_loader(loader: &Loader, cfg: &Gemma4AssistantConfig) -> Result<Self> {
        let num_centroids = cfg
            .num_centroids
            .ok_or_else(|| anyhow!("Gemma4 MaskedEmbedder: num_centroids missing"))?;
        let top_k = cfg
            .centroid_intermediate_top_k
            .ok_or_else(|| anyhow!("Gemma4 MaskedEmbedder: centroid_intermediate_top_k missing"))?;
        let vocab_size = cfg.text_config.vocab_size;
        if vocab_size % num_centroids != 0 {
            return Err(anyhow!(
                "Gemma4 MaskedEmbedder: vocab_size {vocab_size} not divisible by num_centroids {num_centroids}"
            ));
        }
        Ok(Self {
            centroids: Linear::from_loader(loader, "masked_embedding.centroids")?,
            token_ordering: loader.tensor("masked_embedding.token_ordering")?.clone(),
            hidden_size: cfg.text_config.hidden_size,
            vocab_size,
            num_centroids,
            top_k,
            vocab_size_per_centroid: vocab_size / num_centroids,
        })
    }

    fn forward_on(
        &self,
        hidden_states: &Array,
        lm_head_weight: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = hidden_states.shape();
        let dims = shape.as_slice();
        if dims.len() != 3 {
            return Err(anyhow!(
                "Gemma4 MaskedEmbedder: expected hidden [B,S,H], got {dims:?}"
            ));
        }
        let (b, seq, h) = (dims[0], dims[1], dims[2]);
        if h != self.hidden_size {
            return Err(anyhow!(
                "Gemma4 MaskedEmbedder: hidden size {h} != {}",
                self.hidden_size
            ));
        }
        let centroid_logits = self.centroids.forward_on(hidden_states, target)?;
        let partition = mlx::ops::sort::argpartition_on(&centroid_logits, -self.top_k, -1, target)?;
        let c = centroid_logits.shape_at(2);
        let topk_idx = mlx::ops::indexing::slice_strided_on(
            &partition,
            &[0_i32, 0, c - self.top_k][..],
            &[b, seq, c][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        let ordering = self
            .token_ordering
            .reshape_on((self.num_centroids, self.vocab_size_per_centroid), target)?;
        let selected_canonical = ordering.take_on(&topk_idx, 0, target)?;
        let selected = self.top_k * self.vocab_size_per_centroid;
        let flat_idx = selected_canonical.reshape_on((b * seq * selected,), target)?;
        let selected_emb = lm_head_weight
            .take_on(&flat_idx, 0, target)?
            .reshape_on((b, seq, selected, self.hidden_size), target)?;
        let hidden4 = hidden_states.reshape_on((b, seq, 1_i32, self.hidden_size), target)?;
        let selected_t = selected_emb.transpose_axes_on(&[0_i32, 1, 3, 2][..], target)?;
        let selected_logits = hidden4
            .matmul_on(&selected_t, target)?
            .reshape_on((b, seq, selected), target)?;
        let min = mlx::ops::reduction::min_on(&selected_logits, mlx::ops::All, false, target)?;
        let mask_value = &min - 1.0_f32;
        let full = &Array::zeros_on((b, seq, self.vocab_size), hidden_states.dtype(), target)?
            + &mask_value;
        mlx::ops::indexing::put_along_axis_on(
            &full,
            &selected_canonical.reshape_on((b, seq, selected), target)?,
            &selected_logits,
            -1,
            target,
        )
        .map_err(anyhow::Error::from)
    }
}

fn make_drafter_masks(
    shared_kv: &Gemma4SharedKvStates,
    query_len: i32,
    query_offset: i32,
    sliding_window: i32,
    dtype: Dtype,
    kv_valid_len: i32,
    target: StreamOrDevice,
) -> Result<Gemma4DrafterMasks> {
    let sliding = match shared_kv.get(Gemma4LayerKind::Sliding) {
        Some(kv) => {
            let len = kv_len(kv)?;
            bidirectional_swa_mask_on(
                query_len,
                query_offset.min(len),
                len,
                sliding_window,
                Some(kv_valid_len.min(len)),
                0,
                dtype,
                target,
            )?
        }
        None => None,
    };
    let full = match shared_kv.get(Gemma4LayerKind::Full) {
        Some(kv) => {
            let len = kv_len(kv)?;
            let key_offset = (kv_valid_len - len).max(0);
            bidirectional_full_mask_on(
                query_len,
                len,
                Some(kv_valid_len),
                key_offset,
                dtype,
                target,
            )?
        }
        None => None,
    };
    Ok(Gemma4DrafterMasks { sliding, full })
}

fn kv_len(kv: &super::attention::SharedKv) -> Result<i32> {
    let shape = kv.keys.shape();
    let dims = shape.as_slice();
    if dims.len() != 4 {
        return Err(anyhow!("Gemma4 drafter expected K/V rank 4, got {dims:?}"));
    }
    Ok(dims[2])
}

fn draft_position_for_shared_kv(kv_valid_len: i32) -> i32 {
    (kv_valid_len - 1).max(0)
}

#[cfg(test)]
pub(crate) fn build_bidirectional_swa_mask_for_test(
    query_len: i32,
    query_offset: i32,
    kv_len: i32,
    window: i32,
    kv_valid_len: Option<i32>,
    key_offset: i32,
    dtype: Dtype,
) -> Result<Option<Array>> {
    bidirectional_swa_mask_on(
        query_len,
        query_offset,
        kv_len,
        window,
        kv_valid_len,
        key_offset,
        dtype,
        ().into(),
    )
}

#[cfg(test)]
mod tests {
    use super::draft_position_for_shared_kv;

    #[test]
    fn draft_position_uses_previous_target_hidden_position() {
        assert_eq!(draft_position_for_shared_kv(0), 0);
        assert_eq!(draft_position_for_shared_kv(1), 0);
        assert_eq!(draft_position_for_shared_kv(20_400), 20_399);
    }
}

fn bidirectional_full_mask_on(
    query_len: i32,
    kv_len: i32,
    kv_valid_len: Option<i32>,
    key_offset: i32,
    dtype: Dtype,
    target: StreamOrDevice,
) -> Result<Option<Array>> {
    let Some(valid_len) = kv_valid_len else {
        return Ok(None);
    };
    if key_offset + kv_len <= valid_len {
        return Ok(None);
    }
    let mut flat = vec![f32::NEG_INFINITY; query_len as usize * kv_len as usize];
    for q in 0..query_len {
        let base = q as usize * kv_len as usize;
        for k in 0..kv_len {
            if key_offset + k < valid_len {
                flat[base + k as usize] = 0.0;
            }
        }
    }
    let arr: Array = (&flat[..], &[1_i32, 1, query_len, kv_len][..]).try_into()?;
    Ok(Some(mlx::ops::cast::astype_on(&arr, dtype, target)?))
}

#[allow(clippy::too_many_arguments)]
fn bidirectional_swa_mask_on(
    query_len: i32,
    query_offset: i32,
    kv_len: i32,
    window: i32,
    kv_valid_len: Option<i32>,
    key_offset: i32,
    dtype: Dtype,
    target: StreamOrDevice,
) -> Result<Option<Array>> {
    if kv_len <= 0 || query_len <= 0 || window <= 0 {
        return Err(anyhow!(
            "Gemma4 drafter mask: query_len={query_len} kv_len={kv_len} window={window}"
        ));
    }
    if kv_len <= window
        && query_offset - key_offset < window
        && key_offset + kv_len - (query_offset + query_len) < window
        && kv_valid_len.is_none_or(|valid| key_offset + kv_len <= valid)
    {
        return Ok(None);
    }

    let valid_len = kv_valid_len.unwrap_or(i32::MAX);
    let mut flat = vec![f32::NEG_INFINITY; query_len as usize * kv_len as usize];
    for q in 0..query_len {
        let q_abs = query_offset + q;
        let base = q as usize * kv_len as usize;
        for k in 0..kv_len {
            let k_abs = key_offset + k;
            let dist = q_abs - k_abs;
            if dist > -window && dist < window && k_abs < valid_len {
                flat[base + k as usize] = 0.0;
            }
        }
    }
    let arr: Array = (&flat[..], &[1_i32, 1, query_len, kv_len][..]).try_into()?;
    Ok(Some(mlx::ops::cast::astype_on(&arr, dtype, target)?))
}
