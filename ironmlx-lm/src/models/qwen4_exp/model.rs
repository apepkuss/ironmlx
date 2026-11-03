//! Scheduler adapter using independent request rows. Verification starts from
//! an exact Q=1 chain; capability/performance routing is qualified separately.
use super::{
    cache::FlashLayerCache,
    text_model::{FlashCache, FlashTextModel, LayerSnapshot, LayerState},
};
use crate::core::{cache::layer::LayerCache, model::ModelMeta, Model};
use crate::models::dflash2::{DFlash2TargetForwardMode, DFlash2TargetOutput};
use anyhow::{ensure, Result};
use mlx::{Array, Dtype, StreamOrDevice};
use std::sync::Arc;

fn pad_time(array: &Array, length: i32, target: StreamOrDevice) -> Result<Array> {
    let shape = array.shape();
    let dims = shape.as_slice();
    if dims[1] == length {
        return Ok(array.clone());
    }
    let padding = Array::zeros([1, length - dims[1], dims[2]], array.dtype())?;
    Ok(mlx::ops::shape::concatenate_on(
        &[array, &padding],
        1,
        target,
    )?)
}

pub(crate) fn row_lengths(
    batch: usize,
    seq: i32,
    lengths: Option<&[i32]>,
    mask: Option<&Array>,
) -> Result<Vec<i32>> {
    let mut lengths = lengths.map_or_else(|| vec![seq; batch], |lens| lens.to_vec());
    ensure!(
        lengths.len() == batch && lengths.iter().all(|&n| (0..=seq).contains(&n)),
        "invalid Flash row lengths"
    );
    if let Some(mask) = mask {
        // The scheduler's decode/append mask addresses a padded shared KV
        // tensor. Flash owns unpadded KV independently for each request, so
        // causal visibility is already exact from that row's state and lens.
        // Only the separate Boolean prefill mask encodes new-token padding.
        if mask.dtype() != Dtype::Bool {
            let shape = mask.shape();
            let d = shape.as_slice();
            ensure!(
                lengths.len() == batch
                    && d.len() == 4
                    && d[0] == batch as i32
                    && d[1] == 1
                    && d[2] == seq
                    && d[3] > 0
                    && matches!(
                        mask.dtype(),
                        Dtype::Bfloat16 | Dtype::Float16 | Dtype::Float32
                    ),
                "invalid Flash scheduler attention mask"
            );
            return Ok(lengths);
        }
        ensure!(
            mask.size() == batch * seq as usize,
            "Flash padding mask must contain B*S booleans"
        );
        let mask = mask.to_vec::<bool>()?;
        for row in 0..batch {
            let values = &mask[row * seq as usize..(row + 1) * seq as usize];
            let valid = values.iter().take_while(|&&value| value).count();
            ensure!(
                values[valid..].iter().all(|&value| !value),
                "Flash supports right-padded rows only"
            );
            lengths[row] = lengths[row].min(valid as i32);
        }
    }
    Ok(lengths)
}

impl FlashTextModel {
    pub(crate) fn last_logits_on(
        &self,
        hidden: &Array,
        lengths: &[i32],
        target: StreamOrDevice,
    ) -> Result<Array> {
        let mut rows = Vec::new();
        for (row, &length) in lengths.iter().enumerate() {
            if length == 0 {
                rows.push(Array::zeros(
                    [1, 1, self.config.vocab_size as i32],
                    Dtype::Bfloat16,
                )?);
            } else {
                rows.push(self.project_on(
                    &hidden.slice(
                        &[row as i32, length - 1, 0][..],
                        &[row as i32 + 1, length, self.config.hidden_size][..],
                    )?,
                    target,
                )?);
            }
        }
        let refs: Vec<&Array> = rows.iter().collect();
        Ok(mlx::ops::shape::concatenate_on(&refs, 0, target)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn runtime_inputs_on(
        &self,
        ids: &Array,
        position_ids: Option<&Array>,
        embeddings: Option<&Array>,
        lengths: Option<&[i32]>,
        mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        capture_layers: &[usize],
        mode: DFlash2TargetForwardMode,
        target: StreamOrDevice,
    ) -> Result<DFlash2TargetOutput> {
        let shape = ids.shape();
        let dims = shape.as_slice();
        ensure!(
            dims.len() == 2 && dims[0] > 0 && dims[1] > 0,
            "Flash token input must be [B,S]"
        );
        let batch = dims[0] as usize;
        let seq = dims[1];
        let _phase = crate::core::paging_stats::PhaseGuard::enter(if mode.is_verify() {
            crate::core::paging_stats::PagingPhase::Verify
        } else if matches!(mode, DFlash2TargetForwardMode::OrdinaryDecode) {
            crate::core::paging_stats::PagingPhase::Decode
        } else {
            crate::core::paging_stats::PagingPhase::Prefill
        });
        if let Some(p) = position_ids {
            ensure!(
                p.shape().as_slice() == [3, dims[0], seq],
                "Flash positions must be [3,B,S]"
            );
            ensure!(
                matches!(p.dtype(), Dtype::Int32 | Dtype::Int64 | Dtype::Uint32),
                "Flash positions must be integers"
            );
            let coordinates = p.astype_on(Dtype::Int64, target)?.to_vec::<i64>()?;
            ensure!(
                coordinates
                    .iter()
                    .all(|&n| n >= 0 && n < self.config.max_position_embeddings as i64),
                "Flash position exceeds model context"
            );
        }
        if let Some(e) = embeddings {
            ensure!(
                e.shape().as_slice() == [dims[0], seq, self.config.hidden_size],
                "Flash embeddings must be [B,S,H]"
            );
        }
        let tokens = ids.astype_on(Dtype::Uint32, target)?.to_vec::<u32>()?;
        ensure!(
            tokens.iter().all(|&id| id < self.config.vocab_size),
            "invalid Flash token ID"
        );
        let lengths = row_lengths(batch, seq, lengths, mask)?;
        if self.expert_paging_enabled() {
            self.paging_telemetry
                .experts
                .current()
                .request_width
                .fetch_max(
                    lengths.iter().filter(|&&n| n > 0).count() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
        }
        let mut fresh;
        let cache = match cache {
            Some(cache) => cache,
            None => {
                fresh = self.make_cache(
                    batch as i32,
                    self.config.max_position_embeddings as i32,
                    Dtype::Bfloat16,
                )?;
                &mut fresh
            }
        };
        ensure!(
            cache.len() == self.config.num_hidden_layers,
            "Flash cache layer count mismatch"
        );
        let mut positions = None;
        let mut cap = None;
        for (index, layer) in cache.iter().enumerate() {
            let LayerCache::Flash(layer) = layer else {
                anyhow::bail!("Flash received a foreign architecture cache");
            };
            ensure!(
                layer.layer == index
                    && Arc::ptr_eq(&layer.instance, &self.instance)
                    && layer.rows.len() == batch,
                "Flash cache instance/layer/row mismatch"
            );
            if let Some(positions) = &positions {
                ensure!(
                    layer.offsets() == positions,
                    "Flash layer positions disagree"
                );
            } else {
                positions = Some(layer.offsets().to_vec());
            }
            if let Some(cap) = cap {
                ensure!(cap == layer.cap(), "Flash layer capacities disagree");
            } else {
                cap = Some(layer.cap());
            }
        }
        let positions = positions.expect("nonempty model");
        let cap = cap.unwrap();
        ensure!(
            positions
                .iter()
                .zip(&lengths)
                .all(|(&start, &len)| start.checked_add(len).is_some_and(|end| end <= cap)),
            "Flash context capacity exceeded"
        );
        let mut rows: Vec<FlashCache> = positions
            .into_iter()
            .map(|position| FlashCache {
                layers: Vec::with_capacity(cache.len()),
                position,
                cap,
                instance: Arc::clone(&self.instance),
            })
            .collect();
        for layer in cache.iter_mut() {
            let LayerCache::Flash(layer) = layer else {
                unreachable!()
            };
            for (row, state) in std::mem::take(&mut layer.rows).into_iter().enumerate() {
                rows[row].layers.push(state);
            }
        }
        let bases: Vec<_> = rows.iter().map(FlashCache::snapshot).collect();
        let mut prefixes: Vec<Vec<Vec<LayerSnapshot>>> = (0..cache.len())
            .map(|_| (0..batch).map(|_| Vec::new()).collect())
            .collect();
        let result = (|| {
            if self.expert_paging_enabled()
                && self.native_paged_batching
                && batch > 1
                && (!mode.is_verify() || self.layer_major_verify)
            {
                let row_tokens: Vec<_> = (0..batch)
                    .map(|r| {
                        tokens[r * seq as usize..r * seq as usize + lengths[r] as usize].to_vec()
                    })
                    .collect();
                let row_positions = (0..batch)
                    .map(|r| {
                        position_ids
                            .map(|p| {
                                p.slice(&[0, r as i32, 0][..], &[3, r as i32 + 1, lengths[r]][..])
                            })
                            .transpose()
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let row_embeddings = (0..batch)
                    .map(|r| {
                        embeddings
                            .map(|e| {
                                e.slice(
                                    &[r as i32, 0, 0][..],
                                    &[r as i32 + 1, lengths[r], self.config.hidden_size][..],
                                )
                            })
                            .transpose()
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let (outputs, new_prefixes) = self.forward_group_on(
                    &row_tokens,
                    &mut rows,
                    capture_layers,
                    &row_positions,
                    &row_embeddings,
                    mode.is_verify(),
                    target,
                )?;
                prefixes = new_prefixes;
                let mut hidden = Vec::new();
                let mut context = Vec::new();
                for output in outputs {
                    let length = output.hidden.shape().as_slice()[1];
                    let c = if output.captures.is_empty() {
                        Array::zeros(
                            [
                                1,
                                length,
                                (capture_layers.len() as i32) * self.config.hidden_size,
                            ],
                            Dtype::Bfloat16,
                        )?
                    } else {
                        mlx::ops::shape::concatenate_on(
                            &output.captures.iter().collect::<Vec<_>>(),
                            -1,
                            target,
                        )?
                    };
                    hidden.push(pad_time(&output.hidden, seq, target)?);
                    context.push(pad_time(&c, seq, target)?);
                }
                let hidden =
                    mlx::ops::shape::concatenate_on(&hidden.iter().collect::<Vec<_>>(), 0, target)?;
                let context_hidden = mlx::ops::shape::concatenate_on(
                    &context.iter().collect::<Vec<_>>(),
                    0,
                    target,
                )?;
                mlx::transforms::eval(&[&hidden, &context_hidden])?;
                return Ok(DFlash2TargetOutput {
                    hidden,
                    context_hidden,
                });
            }
            if self.expert_paging_enabled() && batch > 1 {
                let count = if self.native_batch_unqualified {
                    &self.paging_telemetry.row_fallback_unqualified
                } else if !self.native_paged_batching {
                    &self.paging_telemetry.row_fallback_disabled
                } else {
                    &self.paging_telemetry.row_fallback_verify_order
                };
                count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let mut hidden = Vec::new();
            let mut context = Vec::new();
            for (row, state) in rows.iter_mut().enumerate() {
                let len = lengths[row];
                if len == 0 {
                    hidden.push(Array::zeros(
                        [1, seq, self.config.hidden_size],
                        Dtype::Bfloat16,
                    )?);
                    context.push(Array::zeros(
                        [
                            1,
                            seq,
                            (capture_layers.len() as i32) * self.config.hidden_size,
                        ],
                        Dtype::Bfloat16,
                    )?);
                    continue;
                }
                let input = &tokens[row * seq as usize..row * seq as usize + len as usize];
                let row_positions = position_ids
                    .map(|p| p.slice(&[0, row as i32, 0][..], &[3, row as i32 + 1, len][..]))
                    .transpose()?;
                let row_embeddings = embeddings
                    .map(|e| {
                        e.slice(
                            &[row as i32, 0, 0][..],
                            &[row as i32 + 1, len, self.config.hidden_size][..],
                        )
                    })
                    .transpose()?;
                let mut row_hidden = Vec::new();
                let mut row_context = Vec::new();
                if mode.is_verify() && self.layer_major_verify {
                    let (output, captured) = self.forward_verify_hidden_tokens_on(
                        input,
                        state,
                        capture_layers,
                        row_positions.as_ref(),
                        row_embeddings.as_ref(),
                        target,
                    )?;
                    row_hidden.push(output.hidden);
                    let context = if output.captures.is_empty() {
                        Array::zeros([1, len, 0], Dtype::Bfloat16)?
                    } else {
                        let refs: Vec<&Array> = output.captures.iter().collect();
                        mlx::ops::shape::concatenate_on(&refs, -1, target)?
                    };
                    row_context.push(context);
                    for (index, captured) in captured.into_iter().enumerate() {
                        prefixes[index][row] = captured;
                    }
                } else {
                    let chunks: Vec<&[u32]> = if mode.is_verify() {
                        input.chunks(1).collect()
                    } else {
                        vec![input]
                    };
                    let mut chunk_start = 0;
                    for chunk in chunks {
                        let chunk_end = chunk_start + chunk.len() as i32;
                        let p = row_positions
                            .as_ref()
                            .map(|p| p.slice(&[0, 0, chunk_start][..], &[3, 1, chunk_end][..]))
                            .transpose()?;
                        let e = row_embeddings
                            .as_ref()
                            .map(|e| {
                                e.slice(
                                    &[0, chunk_start, 0][..],
                                    &[1, chunk_end, self.config.hidden_size][..],
                                )
                            })
                            .transpose()?;
                        let output = self.forward_inputs_on(
                            chunk,
                            state,
                            capture_layers,
                            p.as_ref(),
                            e.as_ref(),
                            target,
                        )?;
                        chunk_start = chunk_end;
                        row_hidden.push(output.hidden);
                        let context = if output.captures.is_empty() {
                            Array::zeros([1, chunk.len() as i32, 0], Dtype::Bfloat16)?
                        } else {
                            let refs: Vec<&Array> = output.captures.iter().collect();
                            mlx::ops::shape::concatenate_on(&refs, -1, target)?
                        };
                        row_context.push(context);
                        if mode.is_verify() {
                            for (index, layer) in state.layers.iter().enumerate() {
                                if matches!(layer, LayerState::Linear { .. }) {
                                    prefixes[index][row].push(layer.snapshot());
                                }
                            }
                        }
                    }
                }
                let refs: Vec<&Array> = row_hidden.iter().collect();
                hidden.push(pad_time(
                    &mlx::ops::shape::concatenate_on(&refs, 1, target)?,
                    seq,
                    target,
                )?);
                let refs: Vec<&Array> = row_context.iter().collect();
                context.push(pad_time(
                    &mlx::ops::shape::concatenate_on(&refs, 1, target)?,
                    seq,
                    target,
                )?);
            }
            let hidden_refs: Vec<&Array> = hidden.iter().collect();
            let context_refs: Vec<&Array> = context.iter().collect();
            let hidden = mlx::ops::shape::concatenate_on(&hidden_refs, 0, target)?;
            let context_hidden = mlx::ops::shape::concatenate_on(&context_refs, 0, target)?;
            mlx::transforms::eval(&[&hidden, &context_hidden])?;
            Ok(DFlash2TargetOutput {
                hidden,
                context_hidden,
            })
        })();
        if result.is_err() {
            for (row, base) in rows.iter_mut().zip(&bases) {
                row.restore(base)?;
            }
        }
        for row in rows {
            for (index, state) in row.layers.into_iter().enumerate() {
                let LayerCache::Flash(layer) = &mut cache[index] else {
                    unreachable!()
                };
                layer.rows.push(state);
            }
        }
        for (index, layer) in cache.iter_mut().enumerate() {
            let LayerCache::Flash(layer) = layer else {
                unreachable!()
            };
            layer.refresh_offsets();
            layer.prefixes = if result.is_ok() && mode.is_verify() {
                std::mem::take(&mut prefixes[index])
            } else {
                Vec::new()
            };
        }
        result
    }
}

impl Model for FlashTextModel {
    fn execution_cache_fingerprint(&self) -> Option<String> {
        Some(<Self as crate::models::dflash2::DFlash2Target>::dflash2_execution_fingerprint(self))
    }
    fn paging_telemetry(&self) -> Option<Arc<crate::core::paging_stats::PagingTelemetry>> {
        Some(Arc::clone(&self.paging_telemetry))
    }
    fn requires_split_batched_prefill_for_token_parity(&self) -> bool {
        !(self.expert_paging_enabled() && self.native_paged_batching)
    }

    fn requires_row_isolated_multimodal_prefill(&self) -> bool {
        true
    }

    fn scheduler_cache_cost() -> Option<(usize, usize)> {
        Some((
            super::load_plan::FLASH_CACHE_BYTES_PER_TOKEN as usize,
            super::load_plan::FLASH_FIXED_BYTES_PER_SEQUENCE as usize,
        ))
    }

    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        ensure!(
            batch > 0 && dtype == Dtype::Bfloat16,
            "Flash cache requires positive batch and BF16 dtype"
        );
        let mut layers: Vec<Vec<LayerState>> = (0..self.config.num_hidden_layers)
            .map(|_| Vec::new())
            .collect();
        for _ in 0..batch {
            for (index, state) in self.new_cache(cap)?.layers.into_iter().enumerate() {
                layers[index].push(state);
            }
        }
        layers
            .into_iter()
            .enumerate()
            .map(|(index, rows)| {
                Ok(LayerCache::Flash(FlashLayerCache::new(
                    rows,
                    Arc::clone(&self.instance),
                    index,
                    cap,
                )?))
            })
            .collect()
    }
    fn forward_on(
        &self,
        ids: &Array,
        positions: &Array,
        lengths: Option<&[i32]>,
        mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = ids.shape();
        let dims = shape.as_slice();
        ensure!(
            dims.len() == 2 && dims[0] > 0 && dims[1] > 0,
            "Flash token input must be [B,S]"
        );
        let valid_lengths = row_lengths(dims[0] as usize, dims[1], lengths, mask)?;
        let output = self.runtime_inputs_on(
            ids,
            Some(positions),
            None,
            lengths,
            mask,
            cache,
            &[],
            if dims[1] == 1
                && !matches!(
                    crate::core::paging_stats::phase_hint(),
                    Some(crate::core::paging_stats::PagingPhase::Prefill)
                )
            {
                DFlash2TargetForwardMode::OrdinaryDecode
            } else {
                DFlash2TargetForwardMode::Prefill
            },
            target,
        )?;
        self.last_logits_on(&output.hidden, &valid_lengths, target)
    }

    fn batched_prefill(
        &self,
        ids: &Array,
        positions: &Array,
        _attention_mask: &Array,
        linear_mask: &Array,
        lengths: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let lengths = row_lengths(
            ids.shape().as_slice()[0] as usize,
            ids.shape().as_slice()[1],
            Some(lengths),
            Some(linear_mask),
        )?;
        let output = self.runtime_inputs_on(
            ids,
            Some(positions),
            None,
            Some(&lengths),
            Some(linear_mask),
            cache,
            &[],
            DFlash2TargetForwardMode::Prefill,
            target,
        )?;
        self.last_logits_on(&output.hidden, &lengths, target)
    }
    fn batched_prefill_causal(
        &self,
        ids: &Array,
        positions: &Array,
        lengths: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let output = self.runtime_inputs_on(
            ids,
            Some(positions),
            None,
            Some(lengths),
            None,
            cache,
            &[],
            DFlash2TargetForwardMode::Prefill,
            target,
        )?;
        self.last_logits_on(&output.hidden, lengths, target)
    }

    fn forward_text_hidden(
        &self,
        ids: &Array,
        positions: &Array,
        lengths: Option<&[i32]>,
        mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        Ok(self
            .runtime_inputs_on(
                ids,
                Some(positions),
                None,
                lengths,
                mask,
                cache,
                &[],
                DFlash2TargetForwardMode::Prefill,
                target,
            )?
            .hidden)
    }
    fn project_hidden_on(&self, hidden: &Array, target: StreamOrDevice) -> Result<Array> {
        let shape = hidden.shape();
        let dims = shape.as_slice();
        ensure!(
            dims.len() == 3 && dims[0] > 0 && dims[2] == self.config.hidden_size,
            "invalid Flash hidden projection"
        );
        let mut rows = Vec::new();
        for row in 0..dims[0] {
            rows.push(self.project_on(
                &hidden.slice(&[row, 0, 0][..], &[row + 1, dims[1], dims[2]][..])?,
                target,
            )?);
        }
        let refs: Vec<&Array> = rows.iter().collect();
        Ok(mlx::ops::shape::concatenate_on(&refs, 0, target)?)
    }
    fn requires_position_ids(&self) -> bool {
        true
    }
    fn model_meta(&self) -> ModelMeta {
        let cfg = &self.config;
        ModelMeta {
            num_hidden_layers: cfg.num_hidden_layers as i32,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_size: cfg.hidden_size,
            head_dim: Some(cfg.head_dim),
            weight_bytes: self.resident_bytes,
            max_position_embeddings: cfg.max_position_embeddings as i32,
            spatial_merge_size: 2,
        }
    }
    fn num_hidden_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}
