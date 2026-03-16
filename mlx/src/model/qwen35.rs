use std::collections::HashMap;

use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::fast;
use crate::nn::{Conv1d, EmbeddingLayer, GatedDeltaNet, LinearLayer, MLP, RMSNorm, RMSNormGated};
use crate::ops;
use crate::stream::Stream;
use crate::vector::VectorArray;

use super::config::Qwen35Config;

// ── Qwen3.5 Full Attention (with output gate) ──────────────────────────────

/// Qwen3.5-specific attention with output gate embedded in q_proj.
///
/// q_proj outputs `n_heads * head_dim * 2`: first half is Q, second half is gate.
/// Final output: `o_proj(attn_output * sigmoid(gate))`.
pub struct Qwen35Attention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub partial_rotary_factor: f32,
    pub rope_base: f32,
}

impl Qwen35Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cache(
        &self,
        x: &Array,
        cache_keys: Option<&Array>,
        cache_values: Option<&Array>,
        offset: i32,
        stream: &Stream,
    ) -> Result<(Array, Array, Array)> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        // Q projection: output [B, L, n_heads * head_dim * 2]
        let q_proj_out = self.q_proj.forward_with_stream(x, stream)?;
        // Reshape to [B, L, n_heads, head_dim * 2], then split last dim in half
        let q_reshaped = ops::reshape(
            &q_proj_out,
            &[b, l, self.n_heads, self.head_dim * 2],
            stream,
        )?;
        let q_split = ops::split(&q_reshaped, 2, -1, stream)?;
        let queries = q_split.get(0)?; // [B, L, n_heads, head_dim]
        let gate = q_split.get(1)?; // [B, L, n_heads, head_dim]
        let gate = ops::reshape(&gate, &[b, l, self.n_heads * self.head_dim], stream)?;

        // K, V projections
        let k = self.k_proj.forward_with_stream(x, stream)?;
        let v = self.v_proj.forward_with_stream(x, stream)?;

        // QK Norm (applied before transpose)
        let queries = self.q_norm.forward_with_stream(&queries, stream)?;
        let k = ops::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim], stream)?;
        let k = self.k_norm.forward_with_stream(&k, stream)?;
        let v = ops::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim], stream)?;

        // Transpose to [B, n_heads, L, head_dim]
        let queries = ops::transpose_axes(&queries, &[0, 2, 1, 3], stream)?;
        let k = ops::transpose_axes(&k, &[0, 2, 1, 3], stream)?;
        let v = ops::transpose_axes(&v, &[0, 2, 1, 3], stream)?;

        // Partial RoPE
        let rope_dim = (self.head_dim as f32 * self.partial_rotary_factor) as i32;
        let (queries, k) = if rope_dim < self.head_dim {
            // Split along last axis for partial rotation
            let q_shape = queries.shape();
            let k_shape = k.shape();

            let q_rot = ops::slice(
                &queries,
                &[0, 0, 0, 0],
                &[q_shape[0], q_shape[1], q_shape[2], rope_dim],
                &[1, 1, 1, 1],
                stream,
            )?;
            let q_pass = ops::slice(
                &queries,
                &[0, 0, 0, rope_dim],
                &[q_shape[0], q_shape[1], q_shape[2], q_shape[3]],
                &[1, 1, 1, 1],
                stream,
            )?;
            let k_rot = ops::slice(
                &k,
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], k_shape[2], rope_dim],
                &[1, 1, 1, 1],
                stream,
            )?;
            let k_pass = ops::slice(
                &k,
                &[0, 0, 0, rope_dim],
                &[k_shape[0], k_shape[1], k_shape[2], k_shape[3]],
                &[1, 1, 1, 1],
                stream,
            )?;

            let q_rot = fast::rope(
                &q_rot,
                rope_dim,
                false,
                Some(self.rope_base),
                1.0,
                offset,
                None,
                stream,
            )?;
            let k_rot = fast::rope(
                &k_rot,
                rope_dim,
                false,
                Some(self.rope_base),
                1.0,
                offset,
                None,
                stream,
            )?;

            let q_arr = VectorArray::from_arrays(&[&q_rot, &q_pass]);
            let k_arr = VectorArray::from_arrays(&[&k_rot, &k_pass]);
            (
                ops::concatenate(&q_arr, 3, stream)?,
                ops::concatenate(&k_arr, 3, stream)?,
            )
        } else {
            let queries = fast::rope(
                &queries,
                self.head_dim,
                false,
                Some(self.rope_base),
                1.0,
                offset,
                None,
                stream,
            )?;
            let k = fast::rope(
                &k,
                self.head_dim,
                false,
                Some(self.rope_base),
                1.0,
                offset,
                None,
                stream,
            )?;
            (queries, k)
        };

        // KV cache update
        let (k, v) = if let (Some(ck), Some(cv)) = (cache_keys, cache_values) {
            let arr_k = VectorArray::from_arrays(&[ck, &k]);
            let arr_v = VectorArray::from_arrays(&[cv, &v]);
            (
                ops::concatenate(&arr_k, 2, stream)?,
                ops::concatenate(&arr_v, 2, stream)?,
            )
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_out = fast::scaled_dot_product_attention(
            &queries, &k, &v, scale, "causal", None, None, stream,
        )?;

        // Transpose back [B, n_heads, L, head_dim] → [B, L, n_heads * head_dim]
        let attn_out = ops::transpose_axes(&attn_out, &[0, 2, 1, 3], stream)?;
        let attn_out = ops::reshape(&attn_out, &[b, l, self.n_heads * self.head_dim], stream)?;

        // Apply output gate: output = o_proj(attn_out * sigmoid(gate))
        let gate_sig = ops::sigmoid(&gate, stream)?;
        let gated = ops::multiply(&attn_out, &gate_sig, stream)?;
        let output = self.o_proj.forward_with_stream(&gated, stream)?;

        Ok((output, k, v))
    }
}

/// A single decoder layer — either GatedDeltaNet (linear) or FullAttention.
pub enum LayerAttention {
    GatedDelta(GatedDeltaNet),
    Full(Qwen35Attention),
}

pub struct Qwen35DecoderLayer {
    pub attention: LayerAttention,
    pub mlp: MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub is_linear: bool,
}

impl Qwen35DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cache(
        &self,
        x: &Array,
        cache: &mut (Option<Array>, Option<Array>),
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array> {
        let normed = self.input_layernorm.forward_with_stream(x, stream)?;

        let attn_out = match &self.attention {
            LayerAttention::GatedDelta(gdn) => {
                // GatedDeltaNet uses SSM mask (just valid positions, not causal triangle)
                gdn.forward_with_cache(&normed, mask, cache, stream)?
            }
            LayerAttention::Full(attn) => {
                let offset = cache.0.as_ref().map_or(0, |k| k.shape()[2]);
                let (out, new_k, new_v) = attn.forward_with_cache(
                    &normed,
                    cache.0.as_ref(),
                    cache.1.as_ref(),
                    offset,
                    stream,
                )?;
                *cache = (Some(new_k), Some(new_v));
                out
            }
        };

        let h = ops::add(x, &attn_out, stream)?;
        let normed = self
            .post_attention_layernorm
            .forward_with_stream(&h, stream)?;
        let mlp_out = self.mlp.forward_with_stream(&normed, stream)?;
        ops::add(&h, &mlp_out, stream)
    }
}

pub struct Qwen35Model {
    pub embed_tokens: EmbeddingLayer,
    pub layers: Vec<Qwen35DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: LinearLayer,
    pub full_attention_interval: usize,
}

impl Qwen35Model {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass. Returns logits [batch, seq_len, vocab_size].
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        tokens: &Array,
        cache: &mut [(Option<Array>, Option<Array>)],
        _mask_mode: &str,
        _mask: Option<&Array>,
    ) -> Result<Array> {
        let stream = Stream::new(&Device::gpu());
        let mut h = self.embed_tokens.forward_with_stream(tokens, &stream)?;

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_with_cache(&h, &mut cache[i], None, &stream)?;
        }

        h = self.norm.forward_with_stream(&h, &stream)?;
        let logits = self.lm_head.forward_with_stream(&h, &stream)?;
        Ok(logits)
    }
}

/// Build a Qwen3.5 model from config file and weights.
pub fn from_config_file(
    config_path: &str,
    weights: &HashMap<String, Array>,
) -> Result<Qwen35Model> {
    let config = Qwen35Config::from_file(config_path)?;
    let tc = &config.text_config;

    let (group_size, bits) = match &config.quantization {
        Some(qc) => (qc.group_size, qc.bits),
        None => (64, 4),
    };

    // Weight key helper — Qwen3.5 uses "language_model.model." prefix
    let w = |name: &str| -> Result<Array> {
        // Try with prefix first, then without
        let prefixed = format!("language_model.model.{}", name);
        if let Some(arr) = weights.get(&prefixed) {
            return Ok(arr.clone());
        }
        if let Some(arr) = weights.get(name) {
            return Ok(arr.clone());
        }
        Err(crate::error::Error::Mlx(format!(
            "missing weight: {} (tried prefixed: {})",
            name, prefixed
        )))
    };

    let linear = |prefix: &str| -> Result<LinearLayer> {
        // Try with language_model.model. prefix
        let prefixed = format!("language_model.model.{}", prefix);
        if weights.contains_key(&format!("{}.weight", prefixed)) {
            return LinearLayer::from_weights(weights, &prefixed, group_size, bits);
        }
        LinearLayer::from_weights(weights, prefix, group_size, bits)
    };

    // Embedding
    let embed_tokens = if weights.contains_key("language_model.model.embed_tokens.weight") {
        EmbeddingLayer::from_weights(
            weights,
            "language_model.model.embed_tokens",
            group_size,
            bits,
        )?
    } else {
        EmbeddingLayer::from_weights(weights, "model.embed_tokens", group_size, bits)?
    };

    // Build layers
    let n_heads = tc.num_attention_heads as i32;
    let n_kv_heads = tc.n_kv_heads() as i32;
    let head_dim = tc.head_dim() as i32;
    let eps = tc.rms_norm_eps as f32;

    let mut layers = Vec::with_capacity(tc.num_hidden_layers);
    for i in 0..tc.num_hidden_layers {
        let lp = format!("layers.{}", i);
        let is_linear = (i + 1) % tc.full_attention_interval != 0;

        let attention = if is_linear {
            // GatedDeltaNet layer
            let key_dim = (tc.linear_key_head_dim * tc.linear_num_key_heads) as i32;
            let value_dim = (tc.linear_value_head_dim * tc.linear_num_value_heads) as i32;
            let conv_dim = key_dim * 2 + value_dim;

            let in_proj_qkv = linear(&format!("{}.linear_attn.in_proj_qkv", lp))?;
            let in_proj_z = linear(&format!("{}.linear_attn.in_proj_z", lp))?;
            let in_proj_b = linear(&format!("{}.linear_attn.in_proj_b", lp))?;
            let in_proj_a = linear(&format!("{}.linear_attn.in_proj_a", lp))?;

            // Conv1d weight — transpose [C, K, 1] → [C, 1, K]
            let conv_weight_raw = w(&format!("{}.linear_attn.conv1d.weight", lp))?;
            let conv_weight = {
                let shape = conv_weight_raw.shape();
                let gpu_stream = Stream::new(&Device::gpu());
                if shape.len() == 3 && shape[1] != 1 {
                    // [C, K, 1] → [C, 1, K] via moveaxis(2, 1)
                    ops::transpose_axes(&conv_weight_raw, &[0, 2, 1], &gpu_stream)?
                } else {
                    conv_weight_raw
                }
            };

            let conv1d = Conv1d::new(
                conv_weight,
                None,
                tc.linear_conv_kernel_dim,
                conv_dim as usize,
            );

            let a_log = w(&format!("{}.linear_attn.A_log", lp))?;
            let dt_bias = w(&format!("{}.linear_attn.dt_bias", lp))?;

            // Norm weight — may need +1.0 shift
            let norm_weight = w(&format!("{}.linear_attn.norm.weight", lp))?;
            let norm = RMSNormGated::new(norm_weight, eps);

            let out_proj = linear(&format!("{}.linear_attn.out_proj", lp))?;

            let gdn = GatedDeltaNet {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
                conv1d,
                a_log,
                dt_bias,
                norm,
                out_proj,
                num_k_heads: tc.linear_num_key_heads as i32,
                num_v_heads: tc.linear_num_value_heads as i32,
                head_k_dim: tc.linear_key_head_dim as i32,
                head_v_dim: tc.linear_value_head_dim as i32,
                key_dim,
                value_dim,
                conv_kernel_size: tc.linear_conv_kernel_dim,
            };

            LayerAttention::GatedDelta(gdn)
        } else {
            // Full attention layer with output gate (Qwen3.5 specific)
            let attn = Qwen35Attention {
                q_proj: linear(&format!("{}.self_attn.q_proj", lp))?,
                k_proj: linear(&format!("{}.self_attn.k_proj", lp))?,
                v_proj: linear(&format!("{}.self_attn.v_proj", lp))?,
                o_proj: linear(&format!("{}.self_attn.o_proj", lp))?,
                q_norm: RMSNorm::new(w(&format!("{}.self_attn.q_norm.weight", lp))?, eps),
                k_norm: RMSNorm::new(w(&format!("{}.self_attn.k_norm.weight", lp))?, eps),
                n_heads,
                n_kv_heads,
                head_dim,
                partial_rotary_factor: tc.partial_rotary_factor() as f32,
                rope_base: tc.rope_theta() as f32,
            };

            LayerAttention::Full(attn)
        };

        let gate_proj = linear(&format!("{}.mlp.gate_proj", lp))?;
        let up_proj = linear(&format!("{}.mlp.up_proj", lp))?;
        let down_proj = linear(&format!("{}.mlp.down_proj", lp))?;
        let mlp = MLP::new(gate_proj, up_proj, down_proj);

        let input_layernorm = RMSNorm::new(w(&format!("{}.input_layernorm.weight", lp))?, eps);
        let post_attention_layernorm =
            RMSNorm::new(w(&format!("{}.post_attention_layernorm.weight", lp))?, eps);

        layers.push(Qwen35DecoderLayer {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            is_linear,
        });
    }

    // Final norm
    let norm = RMSNorm::new(w("norm.weight")?, eps);

    // LM head
    let lm_head = if config.tie_word_embeddings || tc.tie_word_embeddings {
        if weights.contains_key("language_model.model.embed_tokens.weight") {
            LinearLayer::from_weights(
                weights,
                "language_model.model.embed_tokens",
                group_size,
                bits,
            )?
        } else {
            LinearLayer::from_weights(weights, "model.embed_tokens", group_size, bits)?
        }
    } else {
        linear("lm_head")?
    };

    Ok(Qwen35Model {
        embed_tokens,
        layers,
        norm,
        lm_head,
        full_attention_interval: tc.full_attention_interval,
    })
}
