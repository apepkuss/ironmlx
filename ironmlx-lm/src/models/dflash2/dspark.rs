//! Inkling 2's bundled DSpark block draft and recurrent Markov logits.
use super::{attention::DFlash2KvCache, model::DFlash2DraftCache, DFlash2Target};
use crate::core::Loader;
use crate::models::inkling2::Inkling2Config;
use crate::nn::{Linear, RmsNorm};
use anyhow::{ensure, Context, Result};
use mlx::ops::{indexing, shape};
use mlx::{Array, Dtype, StreamOrDevice};
use serde::Deserialize;
use std::{collections::HashMap, path::Path};

#[derive(Debug, Deserialize)]
pub(super) struct Config {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub max_position_embeddings: i32,
    pub sliding_window_size: i32,
    pub local_layer_ids: Vec<i32>,
    pub block_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub markov_rank: i32,
    pub target_layer_ids: Vec<usize>,
    pub markov_head_type: String,
    pub q_bias: bool,
    pub o_bias: bool,
}

struct Weights {
    tensors: HashMap<String, Array>,
    quantize_layers: bool,
}
impl Weights {
    fn tensor(&self, key: &str, shape: &[i32]) -> Result<Array> {
        let param = self
            .tensors
            .get(key)
            .with_context(|| format!("missing DSpark tensor {key}"))?;
        ensure!(
            param.shape().as_slice() == shape && param.dtype() == Dtype::Bfloat16,
            "DSpark tensor {key} must have shape {shape:?} and BF16 dtype"
        );
        Ok(param.clone())
    }

    fn linear(&self, key: &str, rows: i32, cols: i32, bias: bool) -> Result<Linear> {
        let weight = self.tensor(&format!("{key}.weight"), &[rows, cols])?;
        let bias = bias
            .then(|| self.tensor(&format!("{key}.bias"), &[rows]))
            .transpose()?;
        if self.quantize_layers && key.starts_with("layers.") && cols % 64 == 0 {
            return affine8(&weight, bias, StreamOrDevice::default());
        }
        Ok(Linear::new_fp(weight, bias))
    }

    fn norm(&self, key: &str, dim: i32, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm::new(
            self.tensor(&format!("{key}.weight"), &[dim])?,
            eps,
        ))
    }
}

fn affine8(weight: &Array, bias: Option<Array>, t: StreamOrDevice) -> Result<Linear> {
    let [weight, scales, biases]: [Array; 3] =
        mlx::quantization::quantize_on(weight, Some(64), Some(8), "affine", None, t)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected three affine DSpark weight tensors"))?;
    mlx::transforms::eval(&[&weight, &scales, &biases])?;
    Ok(Linear::new_quant(weight, scales, Some(biases), bias, 64, 8))
}

enum GateUp {
    Separate(Box<[Linear; 2]>),
    Packed(Linear),
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    context_k: Linear,
    context_v: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    context_k_norm: RmsNorm,
    input_norm: RmsNorm,
    mlp_norm: RmsNorm,
    context_norm: Array,
    gate_up: GateUp,
    down: Linear,
    alpha: Array,
    local: bool,
}

struct MarkovHead {
    embeddings: Array,
    projection: Linear,
    hidden_out: Linear,
    folded: Option<(Linear, Linear)>,
    context_norm: Array,
    alpha: Array,
}

fn fold_markov_head(
    head: Linear,
    latent: &Array,
    bias: &Array,
    vocab: i32,
    t: StreamOrDevice,
) -> Result<(Linear, Linear)> {
    let parts = head
        .quantized_parts()
        .context("DSpark requires a packed output head")?;
    let latent = latent.astype_on(Dtype::Float32, t)?;
    let bias = bias.astype_on(Dtype::Float32, t)?;
    let mut weight_rows = Vec::new();
    let mut bias_rows = Vec::new();
    for start in (0..vocab).step_by(8192) {
        let end = (start + 8192).min(vocab);
        let quant_bias = parts
            .biases
            .map(|b| slice(b, 0, start, end, t))
            .transpose()?;
        let weights = mlx::quantization::dequantize_on(
            &slice(parts.weight, 0, start, end, t)?,
            &slice(parts.scales, 0, start, end, t)?,
            quant_bias.as_ref(),
            Some(parts.group_size),
            Some(parts.bits),
            parts.mode.mlx_backend_mode(),
            None,
            Some(Dtype::Float32),
            t,
        )?;
        let weight = weights
            .matmul_on(&latent, t)?
            .astype_on(Dtype::Bfloat16, t)?;
        let projected_bias = weights.matmul_on(&bias, t)?.astype_on(Dtype::Bfloat16, t)?;
        mlx::transforms::eval(&[&weight, &projected_bias])?;
        weight_rows.push(weight);
        bias_rows.push(projected_bias);
    }
    let weight = shape::concatenate_on(&weight_rows.iter().collect::<Vec<_>>(), 0, t)?;
    let bias = shape::concatenate_on(&bias_rows.iter().collect::<Vec<_>>(), 0, t)?;
    mlx::transforms::eval(&[&weight, &bias])?;
    let correction = if latent.shape_at(1) % 64 == 0 {
        affine8(&weight, Some(bias), t)?
    } else {
        Linear::new_fp(weight, Some(bias))
    };
    Ok((head, correction))
}

pub(super) struct DSparkDraft {
    pub config: Config,
    layers: Vec<Layer>,
    final_norm: RmsNorm,
    mask: Array,
    markov: MarkovHead,
    vocab: i32,
}

fn slice(x: &Array, axis: usize, start: i32, end: i32, t: StreamOrDevice) -> Result<Array> {
    let mut ends = x.shape().as_slice().to_vec();
    let mut starts = vec![0; ends.len()];
    let strides = vec![1; ends.len()];
    starts[axis] = start;
    ends[axis] = end;
    Ok(indexing::slice_strided_on(
        x,
        &starts[..],
        &ends[..],
        &strides[..],
        t,
    )?)
}

fn mix(taps: &Array, alpha: &Array, norm: &Array, eps: f32, t: StreamOrDevice) -> Result<Array> {
    let length = taps.shape_at(1);
    let count = alpha.shape_at(0);
    let h = norm.shape_at(0);
    let rows = taps
        .reshape_on(&[1, length, count, h][..], t)?
        .astype_on(Dtype::Float32, t)?;
    let w = alpha
        .reshape_on((1, 1, count, 1), t)?
        .astype_on(Dtype::Float32, t)?;
    let sum = rows.try_mul_on(&w, t)?.sum_on(-2, false, t)?;
    Ok(mlx::fast::rms_norm_on(&sum, Some(norm), eps, t)?.astype_on(taps.dtype(), t)?)
}

impl DSparkDraft {
    pub fn load(path: &Path, target: &Inkling2Config, loader: &Loader) -> Result<Self> {
        let config: Config =
            serde_json::from_value(loader.config_raw_value()["mtp_config"].clone())
                .context("reading Inkling DSpark mtp_config")?;
        let c = &config;
        let visits = target.text_config.visits().len();
        ensure!(
            c.hidden_size == target.text_config.hidden_size
                && c.hidden_size > 0
                && c.num_hidden_layers > 0
                && c.num_attention_heads > 0
                && c.num_key_value_heads > 0
                && c.num_attention_heads % c.num_key_value_heads == 0
                && c.head_dim > 0
                && c.head_dim % 2 == 0
                && c.intermediate_size > 0
                && c.rms_norm_eps > 0.0
                && c.rope_theta > 0.0
                && c.markov_rank > 0
                && (2..=8).contains(&c.block_size)
                && c.sliding_window_size > c.block_size
                && c.max_position_embeddings == target.text_config.model_max_length
                && c.markov_head_type == "rnn_hidden"
                && !c.q_bias
                && !c.o_bias
                && c.target_layer_ids.last() == Some(&visits)
                && c.target_layer_ids.windows(2).all(|w| w[0] < w[1])
                && c.local_layer_ids
                    .iter()
                    .all(|i| (0..c.num_hidden_layers).contains(i)),
            "bundled DSpark configuration is incompatible with the Inkling target"
        );
        let t = StreamOrDevice::default();
        let w = Weights {
            tensors: mlx::io::load_safetensors(path.to_str().context("invalid DSpark path")?)?.0,
            quantize_layers: mlx::default_device().device_type == mlx::DeviceType::Gpu,
        };
        let h = c.hidden_size;
        let r = c.markov_rank;
        let eps = c.rms_norm_eps;
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let alpha = w
            .tensor(
                "fuse_weight",
                &[c.num_hidden_layers + 1, c.target_layer_ids.len() as i32],
            )?
            .astype_on(Dtype::Float32, t)?
            .softmax_on(-1, true, t)?
            .astype_on(Dtype::Bfloat16, t)?;
        let alpha_at =
            |i| -> Result<Array> { Ok(slice(&alpha, 0, i, i + 1, t)?.squeeze_on(0, t)?) };
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            let p = format!("layers.{i}");
            let linear = |n, rows, cols| w.linear(&format!("{p}.{n}"), rows, cols, false);
            let norm = |n, d| w.norm(&format!("{p}.{n}"), d, eps);
            let gate = linear("mlp.gate_proj", c.intermediate_size, h)?;
            let up = linear("mlp.up_proj", c.intermediate_size, h)?;
            let gate_up = if gate.quantized_parts().is_some() {
                GateUp::Packed(Linear::fuse_quantized_outputs(&[&gate, &up], &p)?)
            } else {
                GateUp::Separate(Box::new([gate, up]))
            };
            layers.push(Layer {
                q: linear("self_attn.q_proj", q, h)?,
                k: linear("self_attn.k_proj", kv, h)?,
                v: linear("self_attn.v_proj", kv, h)?,
                o: linear("self_attn.o_proj", h, q)?,
                context_k: linear("self_attn.k_proj_ctx", kv, h)?,
                context_v: linear("self_attn.v_proj_ctx", kv, h)?,
                q_norm: norm("self_attn.q_norm", c.head_dim)?,
                k_norm: norm("self_attn.k_norm", c.head_dim)?,
                context_k_norm: norm("self_attn.k_norm_ctx", c.head_dim)?,
                input_norm: norm("input_layernorm", h)?,
                mlp_norm: norm("post_attention_layernorm", h)?,
                context_norm: w
                    .tensor(&format!("{p}.dflare_norm.weight"), &[h])?
                    .astype_on(Dtype::Float32, t)?,
                gate_up,
                down: linear("mlp.down_proj", h, c.intermediate_size)?,
                alpha: alpha_at(i)?,
                local: c.local_layer_ids.contains(&i),
            });
        }
        let vocab = target.text_config.unpadded_vocab_size;
        let folded = if w.quantize_layers {
            let head = Linear::from_loader(loader, "lm_head")?;
            if head.quantized_parts().is_some() {
                Some(fold_markov_head(
                    head,
                    &w.tensor("markov_head.hidden_out.weight", &[h, r])?,
                    &w.tensor("markov_head.hidden_out.bias", &[h])?,
                    vocab,
                    t,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let markov = MarkovHead {
            embeddings: w.tensor(
                "markov_head.markov_w1.weight",
                &[target.text_config.vocab_size, r],
            )?,
            projection: w.linear("markov_head.rnn_proj", 3 * r, 2 * r + 2 * h, true)?,
            hidden_out: w.linear("markov_head.hidden_out", h, r, true)?,
            folded,
            context_norm: w
                .tensor("markov_head.dflare_norm.weight", &[h])?
                .astype_on(Dtype::Float32, t)?,
            alpha: alpha_at(c.num_hidden_layers)?,
        };
        let final_norm = w.norm("norm", h, eps)?;
        let mask = w.tensor("mask_embed", &[h])?.reshape_on((1, 1, h), t)?;
        let mut resident: Vec<_> = w.tensors.values().collect();
        for layer in &layers {
            resident.extend([&layer.alpha, &layer.context_norm]);
        }
        resident.extend([&mask, &markov.alpha, &markov.context_norm]);
        mlx::transforms::eval(&resident)?;
        Ok(Self {
            config,
            layers,
            final_norm,
            mask,
            markov,
            vocab,
        })
    }

    /// Proposals for `ids[1..]` after the anchor `ids[0]`, with the
    /// log-probability of each when `with_confidence`.
    pub fn propose<T: DFlash2Target>(
        &self,
        model: &T,
        ids: &Array,
        taps: &Array,
        cache: &mut DFlash2DraftCache,
        t: StreamOrDevice,
        with_confidence: bool,
    ) -> Result<(Array, Option<Array>)> {
        let rows = ids.shape_at(1);
        ensure!(
            ids.shape_at(0) == 1 && (2..=self.config.block_size).contains(&rows),
            "DSpark requires a single sequence and 2–{} block positions",
            self.config.block_size
        );
        let anchor = slice(ids, 1, 0, 1, t)?;
        let mut block = self.begin(model, &anchor, taps, cache, rows as usize - 1, t)?;
        self.extend(model, &mut block, rows as usize - 1, with_confidence, t)?;
        let tokens: Vec<_> = block.tokens.iter().collect();
        let confidence = with_confidence
            .then(|| shape::concatenate_on(&block.confidence.iter().collect::<Vec<_>>(), 0, t))
            .transpose()?;
        Ok((shape::stack_on(&tokens, 1, t)?, confidence))
    }

    /// Run the block layers over the committed target context and prepare up
    /// to `limit` proposals after `anchor` (`[1, 1]`). The proposals
    /// themselves are produced by [`Self::extend`], so a caller can stop
    /// after the leading ones.
    pub fn begin<T: DFlash2Target>(
        &self,
        model: &T,
        anchor: &Array,
        taps: &Array,
        cache: &mut DFlash2DraftCache,
        limit: usize,
        t: StreamOrDevice,
    ) -> Result<DSparkBlock> {
        let c = &self.config;
        ensure!(
            anchor.shape().as_slice() == [1, 1] && (1..c.block_size as usize).contains(&limit),
            "DSpark requires one anchor and 1–{} proposals",
            c.block_size - 1
        );
        ensure!(
            taps.shape_at(0) == 1
                && taps.shape_at(2) == c.hidden_size * c.target_layer_ids.len() as i32,
            "invalid DSpark target context"
        );
        let embedding = model.dflash2_embed_on(anchor, t)?;
        // Every proposed position attends to the whole trained mask block.
        // The limit only bounds how many Markov outputs can be projected.
        let mask = self
            .mask
            .broadcast_to_on((1, c.block_size - 1, c.hidden_size), t)?;
        let mut hidden = shape::concatenate_on(&[&embedding, &mask], 1, t)?;
        let take = taps.shape_at(1);
        let (before, _) = cache.position_signature()?;
        let start = before.max(0);
        for (layer, kv) in self.layers.iter().zip(&mut cache.layers) {
            hidden = layer.forward(&hidden, taps, kv, c, start, t)?;
        }
        hidden = self.final_norm.forward_on(&hidden, t)?;
        if take > 0 {
            cache.last_draft_context = Some(mix(
                &slice(taps, 1, take - 1, take, t)?,
                &self.markov.alpha,
                &self.markov.context_norm,
                c.rms_norm_eps,
                t,
            )?);
        }
        let last = cache
            .last_draft_context
            .as_ref()
            .context("DSpark requires a committed target context")?
            .squeeze_on(1, t)?;
        // One pass over the output head serves every position of the block.
        let base_logits = self
            .markov
            .folded
            .as_ref()
            .map(|(head, _)| -> Result<Array> {
                let values = head.forward_on(&slice(&hidden, 1, 1, limit as i32 + 1, t)?, t)?;
                slice(&values, 2, 0, self.vocab, t)
            })
            .transpose()?;
        Ok(DSparkBlock {
            state: Array::zeros((1, c.markov_rank), hidden.dtype())?,
            previous: anchor.squeeze_on(1, t)?,
            hidden,
            last,
            base_logits,
            limit,
            tokens: Vec::new(),
            confidence: Vec::new(),
        })
    }

    /// Extend the block's proposals to `count` positions (at most its limit).
    pub fn extend<T: DFlash2Target>(
        &self,
        model: &T,
        block: &mut DSparkBlock,
        count: usize,
        with_confidence: bool,
        t: StreamOrDevice,
    ) -> Result<()> {
        let c = &self.config;
        ensure!(
            count <= block.limit,
            "DSpark block holds {} proposals",
            block.limit
        );
        for i in block.tokens.len() as i32 + 1..=count as i32 {
            let h = slice(&block.hidden, 1, i, i + 1, t)?.squeeze_on(1, t)?;
            let prev = indexing::take_on(&self.markov.embeddings, &block.previous, 0, t)?;
            let inputs = shape::concatenate_on(&[&block.state, &prev, &h, &block.last], -1, t)?;
            let proj = self
                .markov
                .projection
                .forward_on(&inputs, t)?
                .astype_on(Dtype::Float32, t)?;
            let r = c.markov_rank;
            let gate = slice(&proj, 1, 0, r, t)?
                .sigmoid_on(t)?
                .astype_on(h.dtype(), t)?;
            let blend = slice(&proj, 1, r, 2 * r, t)?
                .tanh_on(t)?
                .astype_on(h.dtype(), t)?;
            block.state = gate.try_mul_on(&block.state, t)?.try_add_on(
                &gate
                    .try_neg_on(t)?
                    .try_add_on(
                        &Array::try_from((&[1.0_f32][..], ()))?.astype_on(h.dtype(), t)?,
                        t,
                    )?
                    .try_mul_on(&blend, t)?,
                t,
            )?;
            let latent = slice(&proj, 1, 2 * r, 3 * r, t)?
                .tanh_on(t)?
                .astype_on(h.dtype(), t)?;
            let logits = if let (Some((_, folded)), Some(base)) =
                (&self.markov.folded, &block.base_logits)
            {
                let correction = folded
                    .forward_on(&latent, t)?
                    .astype_on(Dtype::Float32, t)?;
                let base = slice(base, 1, i - 1, i, t)?
                    .squeeze_on(1, t)?
                    .astype_on(Dtype::Float32, t)?;
                base.try_add_on(&correction, t)?
            } else {
                let biased = h.try_add_on(&self.markov.hidden_out.forward_on(&latent, t)?, t)?;
                model.dflash2_project_hidden_on(&biased.expand_dims_on(1, t)?, t)?
            };
            block.previous = logits.argmax_on(-1, false, t)?.reshape_on((1,), t)?;
            block.tokens.push(block.previous.clone());
            if with_confidence {
                // Log-probability of the proposed token.
                let logits = logits.astype_on(Dtype::Float32, t)?;
                let top = mlx::ops::reduction::max_on(&logits, &[-1][..], false, t)?;
                let total = mlx::ops::reduction::logsumexp_on(&logits, &[-1][..], false, t)?;
                block
                    .confidence
                    .push(top.try_sub_on(&total, t)?.reshape_on((1,), t)?);
            }
        }
        Ok(())
    }
}

/// A DSpark proposal block whose positions are projected on demand.
pub struct DSparkBlock {
    hidden: Array,
    last: Array,
    base_logits: Option<Array>,
    state: Array,
    previous: Array,
    limit: usize,
    /// Proposed token of each projected position, `[1]` each.
    pub tokens: Vec<Array>,
    /// Log-probability of each proposed token, `[1]` each, when requested.
    pub confidence: Vec<Array>,
}

impl Layer {
    fn forward(
        &self,
        hidden: &Array,
        taps: &Array,
        cache: &mut DFlash2KvCache,
        c: &Config,
        start: i32,
        t: StreamOrDevice,
    ) -> Result<Array> {
        let s = hidden.shape_at(1);
        let n = taps.shape_at(1);
        let normalized = self.input_norm.forward_on(hidden, t)?;
        let heads = |a: Array, len, h| -> Result<Array> {
            Ok(a.reshape_on((1, len, h, c.head_dim), t)?
                .transpose_axes_on([0, 2, 1, 3], t)?)
        };
        let rope = |a: &Array, offset| -> Result<Array> {
            Ok(mlx::fast::rope_on(
                a,
                c.head_dim,
                false,
                Some(c.rope_theta),
                1.0,
                offset,
                None,
                t,
            )?)
        };
        let q = self.q_norm.forward_on(
            &heads(self.q.forward_on(&normalized, t)?, s, c.num_attention_heads)?,
            t,
        )?;
        let k = self.k_norm.forward_on(
            &heads(self.k.forward_on(&normalized, t)?, s, c.num_key_value_heads)?,
            t,
        )?;
        let v = heads(self.v.forward_on(&normalized, t)?, s, c.num_key_value_heads)?;
        let position = start + n;
        let q = rope(&q, position)?;
        let k = rope(&k, position)?;
        let (ctx_k, ctx_v) = if n == 0 {
            (slice(&k, 2, 0, 0, t)?, slice(&v, 2, 0, 0, t)?)
        } else {
            let context = mix(taps, &self.alpha, &self.context_norm, c.rms_norm_eps, t)?;
            let k = heads(
                self.context_k.forward_on(&context, t)?,
                n,
                c.num_key_value_heads,
            )?;
            let k = rope(&self.context_k_norm.forward_on(&k, t)?, start)?;
            (
                k,
                heads(
                    self.context_v.forward_on(&context, t)?,
                    n,
                    c.num_key_value_heads,
                )?,
            )
        };
        let (keys, values) = cache.append_with_proposal_on(&ctx_k, &ctx_v, &k, &v, t)?;
        let mask = (self.local && keys.shape_at(2) > c.sliding_window_size)
            .then(|| {
                super::model::build_block_mask(
                    s,
                    keys.shape_at(2) - s,
                    c.sliding_window_size,
                    hidden.dtype(),
                    t,
                )
            })
            .transpose()?;
        let attended = mlx::fast::scaled_dot_product_attention_on(
            &q,
            &keys,
            &values,
            (c.head_dim as f32).powf(-0.5),
            "",
            mask.as_ref(),
            None,
            t,
        )?
        .transpose_axes_on([0, 2, 1, 3], t)?
        .reshape_on((1, s, c.num_attention_heads * c.head_dim), t)?;
        let residual = hidden.try_add_on(&self.o.forward_on(&attended, t)?, t)?;
        let input = self.mlp_norm.forward_on(&residual, t)?;
        let (gate, up) = match &self.gate_up {
            GateUp::Separate(projections) => (
                projections[0].forward_on(&input, t)?,
                projections[1].forward_on(&input, t)?,
            ),
            GateUp::Packed(projection) => {
                let values = projection.forward_on(&input, t)?;
                let width = c.intermediate_size;
                (
                    slice(&values, 2, 0, width, t)?,
                    slice(&values, 2, width, 2 * width, t)?,
                )
            }
        };
        let mlp = gate
            .try_mul_on(&gate.sigmoid_on(t)?, t)?
            .try_mul_on(&up, t)?;
        Ok(residual.try_add_on(&self.down.forward_on(&mlp, t)?, t)?)
    }
}
