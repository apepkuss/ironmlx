//! PLE token lookup and dilated short convolution. Both n-gram history and
//! convolution state are request-owned and included in rollback snapshots.
use super::{
    config::FlashTextConfig, math::grouped_residual_rms_on, ple_index::PleHistory,
    ple_rows::PleRows,
};
use crate::{
    core::weights::WeightSource,
    nn::{Conv1d, Conv1dConfig, Linear},
};
use anyhow::{ensure, Context, Result};
use mlx::ops::{
    binary::{add_on, divide_on, maximum_on, multiply_on},
    shape::{concatenate_on, expand_dims_on},
};
use mlx::{Array, StreamOrDevice};
use std::sync::Mutex;

#[derive(Clone)]
pub struct PleState {
    pub history: PleHistory,
    pub convolution: Option<Array>,
    initial_history: PleHistory,
}
impl PleState {
    pub(crate) fn from_history(history: PleHistory) -> Self {
        Self {
            initial_history: history.clone(),
            history,
            convolution: None,
        }
    }
    pub fn reset(&mut self) {
        self.history = self.initial_history.clone();
        self.convolution = None;
    }
}

pub struct PleLayer {
    rows: Mutex<PleRows>,
    key: Linear,
    value: Linear,
    key_norm: Array,
    query_norm: Array,
    conv_norm: Array,
    conv: Conv1d,
    cfg: FlashTextConfig,
    history_rows: i32,
}

impl PleLayer {
    pub fn from_weights(
        source: &(impl WeightSource + ?Sized),
        prefix: &str,
        cfg: &FlashTextConfig,
        rows: PleRows,
    ) -> Result<Self> {
        let channels = cfg
            .hidden_size
            .checked_mul(cfg.hc_count)
            .context("PLE channel overflow")?;
        let dilation = i32::try_from(cfg.ngram_size)?;
        let history_rows = (cfg.ple_conv_kernel_size - 1)
            .checked_mul(dilation)
            .context("PLE history overflow")?;
        Ok(Self {
            rows: Mutex::new(rows),
            key: Linear::from_loader(source, &format!("{prefix}.key_proj"))?,
            value: Linear::from_loader(source, &format!("{prefix}.value_proj"))?,
            key_norm: source.tensor(&format!("{prefix}.norm_key.weight"))?.clone(),
            query_norm: source
                .tensor(&format!("{prefix}.norm_query.weight"))?
                .clone(),
            conv_norm: source
                .tensor(&format!("{prefix}.norm_conv.weight"))?
                .clone(),
            conv: Conv1d::from_loader(
                source,
                &format!("{prefix}.conv1d"),
                Conv1dConfig {
                    in_channels: channels,
                    out_channels: channels,
                    kernel_size: cfg.ple_conv_kernel_size,
                    stride: 1,
                    padding: 0,
                    dilation,
                    groups: channels,
                },
            )?,
            cfg: cfg.clone(),
            history_rows,
        })
    }

    pub fn new_state(&self) -> Result<PleState> {
        let history = self
            .rows
            .lock()
            .map_err(|_| anyhow::anyhow!("PLE cache lock poisoned"))?
            .index()
            .history();
        Ok(PleState::from_history(history))
    }

    pub fn forward_on(
        &self,
        hidden: &Array,
        tokens: &[u32],
        state: &mut PleState,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let seq = i32::try_from(tokens.len())?;
        let width = cfg.hidden_size * cfg.hc_count;
        ensure!(
            seq > 0 && hidden.shape().as_slice() == [1, seq, width],
            "invalid PLE B1 input"
        );
        let mut history = state.history.clone();
        let embeddings = {
            let mut rows = self
                .rows
                .lock()
                .map_err(|_| anyhow::anyhow!("PLE cache lock poisoned"))?;
            let indices: Vec<u64> = rows
                .index()
                .indices(tokens, &mut history)?
                .into_iter()
                .flatten()
                .collect();
            let per_chunk = rows.max_gather_rows() / rows.index().heads() * rows.index().heads();
            let mut chunks = Vec::new();
            for chunk in indices.chunks(per_chunk) {
                chunks.push(rows.gather_on(chunk, target)?);
            }
            let refs: Vec<&Array> = chunks.iter().collect();
            let embeddings = mlx::ops::shape::concatenate_on(&refs, 0, target)?
                .reshape_on([1, seq, cfg.ple_embed_dim], target)?;
            embeddings.eval()?;
            embeddings
        };
        let norm = |x: &Array, gamma: &Array| {
            grouped_residual_rms_on(x, gamma, cfg.hidden_size, cfg.rms_norm_eps, target)
        };
        let keys = norm(&self.key.forward_on(&embeddings, target)?, &self.key_norm)?
            .reshape_on([1, seq, cfg.hc_count, cfg.hidden_size], target)?;
        let queries = norm(hidden, &self.query_norm)?
            .reshape_on([1, seq, cfg.hc_count, cfg.hidden_size], target)?;
        let values = self.value.forward_on(&embeddings, target)?;
        let gate = multiply_on(&keys, &queries, target)?.sum_on(-1, true, target)?;
        let scalar = |value| -> Result<Array> {
            Ok(Array::try_from((&[value][..], ()))?.astype_on(gate.dtype(), target)?)
        };
        let gate = divide_on(&gate, &scalar((cfg.hidden_size as f32).sqrt())?, target)?;
        let magnitude =
            maximum_on(&gate.abs_on(target)?, &scalar(1e-6)?, target)?.sqrt_on(target)?;
        let gate = multiply_on(&gate.sign_on(target)?, &magnitude, target)?.sigmoid_on(target)?;
        let gated = multiply_on(&gate, &expand_dims_on(&values, -2, target)?, target)?
            .reshape_on([1, seq, width], target)?;
        let normed = norm(&gated, &self.conv_norm)?;
        let initial;
        let previous = match &state.convolution {
            Some(previous) => {
                ensure!(
                    previous.shape().as_slice() == [1, self.history_rows, width]
                        && previous.dtype() == hidden.dtype(),
                    "PLE convolution state mismatch"
                );
                previous
            }
            None => {
                initial = Array::zeros([1, self.history_rows, width], hidden.dtype())?;
                &initial
            }
        };
        let conv_input = concatenate_on(&[previous, &normed], 1, target)?;
        let convolved = self.conv.forward_on(&conv_input, target)?;
        let activated = multiply_on(&convolved, &convolved.sigmoid_on(target)?, target)?;
        let output = add_on(&gated, &activated, target)?;
        let convolution =
            conv_input.slice(&[0, seq, 0][..], &[1, seq + self.history_rows, width][..])?;
        state.history = history;
        state.convolution = Some(convolution);
        Ok(output)
    }
}
