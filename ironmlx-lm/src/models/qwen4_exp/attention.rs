//! Correctness-first text QSA. Completed blocks pool raw indexer keys before
//! normalization and RoPE; incomplete tails remain visible causally. State
//! snapshots clone handles and include every auxiliary indexer tensor.
use super::{config::FlashTextConfig, math::grouped_residual_rms_on};
use crate::{
    core::weights::WeightSource,
    nn::{Linear, Mrope},
};
use anyhow::{ensure, Result};
use mlx::ops::{
    binary::{divide_on, maximum_on, multiply_on},
    shape::{concatenate_on, transpose_axes_on},
};
use mlx::{Array, Dtype, StreamOrDevice};

#[derive(Clone)]
pub struct QsaState {
    pub keys: Option<Array>,
    pub values: Option<Array>,
    pub raw_index_keys: Option<Array>,
    pub block_keys: Option<Array>,
    pub positions: Option<Array>,
    pub offset: i32,
    cap: i32,
}

impl QsaState {
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.raw_index_keys = None;
        self.block_keys = None;
        self.positions = None;
        self.offset = 0;
    }
    pub(crate) fn set_cap(&mut self, cap: i32) -> Result<()> {
        ensure!(
            cap > 0 && (0..=cap).contains(&self.offset),
            "QSA snapshot exceeds destination capacity"
        );
        self.cap = cap;
        Ok(())
    }
    pub fn new(cap: i32) -> Result<Self> {
        ensure!(cap > 0, "QSA cache capacity must be positive");
        Ok(Self {
            keys: None,
            values: None,
            raw_index_keys: None,
            block_keys: None,
            positions: None,
            offset: 0,
            cap,
        })
    }
    /// Rollback trims all token histories and invalidates pooled blocks.
    pub fn trim_on(&mut self, offset: i32, target: StreamOrDevice) -> Result<()> {
        ensure!(
            (0..=self.offset).contains(&offset),
            "invalid QSA rollback offset"
        );
        let trim = |array: &Option<Array>, axis: usize| -> Result<Option<Array>> {
            array
                .as_ref()
                .map(|x| {
                    let mut end = x.shape().as_slice().to_vec();
                    end[axis] = offset;
                    let start = vec![0; end.len()];
                    Ok(mlx::ops::indexing::slice_strided_on(
                        x,
                        start.as_slice(),
                        end.as_slice(),
                        vec![1; end.len()].as_slice(),
                        target,
                    )?)
                })
                .transpose()
        };
        let keys = trim(&self.keys, 2)?;
        let values = trim(&self.values, 2)?;
        let index = trim(&self.raw_index_keys, 1)?;
        let positions = trim(&self.positions, 2)?;
        self.keys = keys;
        self.values = values;
        self.raw_index_keys = index;
        self.positions = positions;
        self.block_keys = None;
        self.offset = offset;
        Ok(())
    }
}

pub struct QsaAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    index: Linear,
    q_norm: Array,
    k_norm: Array,
    index_q_norm: Array,
    index_k_norm: Array,
    rope: Mrope,
    index_rope: Mrope,
    cfg: FlashTextConfig,
    /// Dense fallback mask/scores are allowed only inside this reservation.
    scratch_bytes: usize,
}

fn positions(start: i32, length: i32, stride: i32) -> Result<Array> {
    let one: Vec<i32> = (0..length).map(|i| start + i * stride).collect();
    let mut all = Vec::with_capacity(one.len() * 3);
    for _ in 0..3 {
        all.extend_from_slice(&one);
    }
    Ok(Array::try_from((all.as_slice(), [3, 1, length]))?)
}

fn append(old: &Option<Array>, new: &Array, axis: i32, target: StreamOrDevice) -> Result<Array> {
    Ok(match old {
        Some(old) => concatenate_on(&[old, new], axis, target)?,
        None => new.clone(),
    })
}

impl QsaAttention {
    pub fn from_weights(
        source: &(impl WeightSource + ?Sized),
        prefix: &str,
        cfg: &FlashTextConfig,
        scratch_bytes: usize,
    ) -> Result<Self> {
        let linear = |name| Linear::from_loader(source, &format!("{prefix}.{name}"));
        let norm = |name| source.tensor(&format!("{prefix}.{name}.weight")).cloned();
        let rope = &cfg.rope_parameters;
        let rotary = cfg.head_dim as f32 * rope.partial_rotary_factor;
        Ok(Self {
            q: linear("q_proj")?,
            k: linear("k_proj")?,
            v: linear("v_proj")?,
            out: linear("o_proj")?,
            index: linear("indexer.index_qk_proj")?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            index_q_norm: norm("indexer.q_layernorm")?,
            index_k_norm: norm("indexer.k_layernorm")?,
            rope: Mrope::new_reference_power(
                cfg.head_dim,
                rope.rope_theta,
                rope.partial_rotary_factor,
                &rope.mrope_section,
                true,
            )?,
            index_rope: Mrope::new_reference_power(
                cfg.indexer_head_dim,
                rope.rope_theta,
                rotary / cfg.indexer_head_dim as f32,
                &rope.mrope_section,
                true,
            )?,
            cfg: cfg.clone(),
            scratch_bytes,
        })
    }

    fn sparse_mask_on(
        &self,
        projected: &Array,
        state: &mut QsaState,
        seq: i32,
        chunk_positions: &Array,
        target: StreamOrDevice,
    ) -> Result<(Option<Array>, Option<Array>)> {
        let cfg = &self.cfg;
        let hd = cfg.indexer_head_dim;
        let heads = cfg.indexer_n_heads;
        let projected = projected.reshape_on([1, seq, heads + 1, hd], target)?;
        let raw = projected
            .slice(&[0, 0, heads, 0][..], &[1, seq, heads + 1, hd][..])?
            .reshape_on([1, seq, hd], target)?;
        let raw = append(&state.raw_index_keys, &raw, 1, target)?;
        state.raw_index_keys = Some(raw.clone());
        let total = state.offset + seq;
        let ratio = cfg.indexer_compress_ratio;
        let blocks = total / ratio;
        let picks = cfg.indexer_budget / ratio;
        if blocks <= picks {
            if seq == 1 {
                return Ok((None, None));
            }
            let mut selected = Vec::with_capacity(seq as usize * picks as usize);
            for t in 0..seq {
                let complete = (state.offset + t + 1) / ratio;
                selected.extend(std::iter::repeat_n(-1_i32, (picks - complete) as usize));
                selected.extend(0..complete);
            }
            return Ok((
                None,
                Some(Array::try_from((selected.as_slice(), [1, seq, picks]))?),
            ));
        }
        let required = (seq as usize)
            .checked_mul(total as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("QSA scratch overflow"))?;
        ensure!(
            required <= self.scratch_bytes && required / 4 <= i32::MAX as usize,
            "QSA mask exceeds scratch reservation; chunk prefill"
        );
        let query = projected.slice(&[0, 0, 0, 0][..], &[1, seq, heads, hd][..])?;
        let query =
            grouped_residual_rms_on(&query, &self.index_q_norm, hd, cfg.rms_norm_eps, target)?;
        let query = transpose_axes_on(&query, [0, 2, 1, 3], target)?;
        let (cos, sin) = self.index_rope.cos_sin(chunk_positions)?;
        let (query, _) = self.index_rope.apply(&query, &query, &cos, &sin)?;
        let previous = state
            .block_keys
            .as_ref()
            .map_or(0, |x| x.shape().as_slice()[2]);
        ensure!(previous <= blocks, "QSA block cache exceeds raw history");
        let pooled = if previous < blocks {
            let pooled = raw
                .slice(&[0, previous * ratio, 0][..], &[1, blocks * ratio, hd][..])?
                .reshape_on([1, blocks - previous, ratio, hd], target)?
                .astype_on(Dtype::Float32, target)?
                .mean_on(2, false, target)?
                .astype_on(raw.dtype(), target)?;
            let pooled =
                grouped_residual_rms_on(&pooled, &self.index_k_norm, hd, cfg.rms_norm_eps, target)?
                    .reshape_on([1, 1, blocks - previous, hd], target)?;
            let block_positions = mlx::ops::indexing::slice_strided_on(
                state.positions.as_ref().expect("QSA position history"),
                &[0, 0, previous * ratio][..],
                &[3, 1, blocks * ratio][..],
                &[1, 1, ratio][..],
                target,
            )?;
            let (cos, sin) = self.index_rope.cos_sin(&block_positions)?;
            let (pooled, _) = self.index_rope.apply(&pooled, &pooled, &cos, &sin)?;
            append(&state.block_keys, &pooled, 2, target)?
        } else {
            state.block_keys.as_ref().unwrap().clone()
        };
        state.block_keys = Some(pooled.clone());
        let keys_t = transpose_axes_on(
            &pooled.astype_on(Dtype::Float32, target)?,
            [0, 1, 3, 2],
            target,
        )?;
        let scores = mlx::ops::matmul::matmul_on(
            &query.astype_on(Dtype::Float32, target)?,
            &keys_t,
            target,
        )?;
        let zero = Array::try_from((&[0f32][..], ()))?;
        let scores = maximum_on(&scores, &zero, target)?.sum_on(1, false, target)?;
        let scale = Array::try_from((&[(hd as f32).sqrt()][..], ()))?;
        let scores = divide_on(&scores, &scale, target)?;
        let mut valid = Vec::with_capacity((seq * blocks) as usize);
        for t in 0..seq {
            for block in 0..blocks {
                valid.push(block < (state.offset + t + 1) / ratio);
            }
        }
        let valid = Array::try_from((valid.as_slice(), [1, seq, blocks]))?;
        let neginf = Array::try_from((&[f32::NEG_INFINITY][..], ()))?;
        let scores = mlx::ops::indexing::where_on(&valid, &scores, &neginf, target)?;
        let selected = mlx::ops::sort::argpartition_on(&scores, -picks, -1, target)?
            .slice(&[0, 0, blocks - picks][..], &[1, seq, blocks][..])?
            .astype_on(Dtype::Int32, target)?;
        // Sentinels must precede real blocks after sorting, including future
        // blocks invalidated independently for every causal query position.
        let counts: Vec<i32> = (0..seq).map(|t| (state.offset + t + 1) / ratio).collect();
        let counts = Array::try_from((counts.as_slice(), [1, seq, 1]))?;
        let valid = selected.less_on(&counts, target)?;
        let sentinel = Array::try_from((&[-1_i32][..], ()))?;
        let selected = mlx::ops::indexing::where_on(&valid, &selected, &sentinel, target)?;
        if seq > 1 {
            return Ok((None, Some(selected)));
        }
        let selected_host = selected.to_vec::<i32>()?;
        let mut mask = vec![false; (seq * total) as usize];
        for t in 0..seq {
            let end = state.offset + t + 1;
            let complete = end / ratio;
            let row = &mut mask[(t * total) as usize..((t + 1) * total) as usize];
            if complete <= picks {
                row[..end as usize].fill(true);
            } else {
                for &block in &selected_host[(t * picks) as usize..((t + 1) * picks) as usize] {
                    if block >= 0 && block < complete {
                        row[(block * ratio) as usize..((block + 1) * ratio) as usize].fill(true);
                    }
                }
                row[(complete * ratio) as usize..end as usize].fill(true);
            }
        }
        Ok((
            Some(Array::try_from((mask.as_slice(), [1, 1, seq, total]))?),
            Some(selected),
        ))
    }

    pub fn forward_on(
        &self,
        input: &Array,
        cache: &mut QsaState,
        target: StreamOrDevice,
    ) -> Result<Array> {
        self.forward_positions_on(input, cache, None, target)
    }

    pub fn forward_positions_on(
        &self,
        input: &Array,
        cache: &mut QsaState,
        position_ids: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = input.shape();
        let dims = shape.as_slice();
        ensure!(
            dims.len() == 3 && dims[0] == 1 && dims[1] > 0 && dims[2] == self.cfg.hidden_size,
            "initial text QSA qualification path requires B1 [1,S,hidden]"
        );
        let seq = dims[1];
        ensure!(
            cache
                .offset
                .checked_add(seq)
                .is_some_and(|n| n <= cache.cap),
            "QSA context capacity exceeded"
        );
        let chunk_positions = match position_ids {
            Some(ids) => {
                ensure!(
                    ids.shape().as_slice() == [3, 1, seq],
                    "QSA positions must be [3,1,S]"
                );
                ids.clone()
            }
            None => positions(cache.offset, seq, 1)?,
        };
        let mut state = cache.clone();
        state.positions = Some(append(&state.positions, &chunk_positions, 2, target)?);
        let (mask, selected) = self.sparse_mask_on(
            &self.index.forward_on(input, target)?,
            &mut state,
            seq,
            &chunk_positions,
            target,
        )?;
        let cfg = &self.cfg;
        let heads = cfg.num_attention_heads;
        let kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_gate = self
            .q
            .forward_on(input, target)?
            .reshape_on([1, seq, heads, 2 * hd], target)?;
        let q = q_gate.slice(&[0, 0, 0, 0][..], &[1, seq, heads, hd][..])?;
        let gate = q_gate
            .slice(&[0, 0, 0, hd][..], &[1, seq, heads, 2 * hd][..])?
            .reshape_on([1, seq, heads * hd], target)?;
        let q = grouped_residual_rms_on(&q, &self.q_norm, hd, cfg.rms_norm_eps, target)?;
        let k = self
            .k
            .forward_on(input, target)?
            .reshape_on([1, seq, kv, hd], target)?;
        let k = grouped_residual_rms_on(&k, &self.k_norm, hd, cfg.rms_norm_eps, target)?;
        let q = transpose_axes_on(&q, [0, 2, 1, 3], target)?;
        let k = transpose_axes_on(&k, [0, 2, 1, 3], target)?;
        let v = self
            .v
            .forward_on(input, target)?
            .reshape_on([1, seq, kv, hd], target)?;
        let v = transpose_axes_on(&v, [0, 2, 1, 3], target)?;
        let (cos, sin) = self.rope.cos_sin(&chunk_positions)?;
        let (q, k) = self.rope.apply(&q, &k, &cos, &sin)?;
        let k = append(&state.keys, &k, 2, target)?;
        let v = append(&state.values, &v, 2, target)?;
        let mode = if mask.is_some() || seq == 1 {
            ""
        } else {
            "causal"
        };
        let attended = if seq > 1 {
            let ends: Vec<i32> = (0..seq).map(|t| state.offset + t + 1).collect();
            let ends = Array::try_from((ends.as_slice(), [1, seq]))?;
            super::qsa_kernel::prefill_on(
                &q,
                &k,
                &v,
                selected.as_ref().expect("prefill selection"),
                &ends,
                (hd as f32).powf(-0.5),
                cfg.indexer_compress_ratio,
                target,
            )?
        } else {
            mlx::fast::scaled_dot_product_attention_on(
                &q,
                &k,
                &v,
                (hd as f32).powf(-0.5),
                mode,
                mask.as_ref(),
                None,
                target,
            )?
        };
        let attended = transpose_axes_on(&attended, [0, 2, 1, 3], target)?
            .reshape_on([1, seq, heads * hd], target)?;
        let output = self.out.forward_on(
            &multiply_on(&attended, &gate.sigmoid_on(target)?, target)?,
            target,
        )?;
        state.keys = Some(k);
        state.values = Some(v);
        state.offset += seq;
        *cache = state;
        Ok(output)
    }
}
