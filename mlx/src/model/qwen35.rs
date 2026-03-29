use std::collections::HashMap;

use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::fast;
use crate::nn::mrope;
use crate::nn::{
    Conv1d, EmbeddingLayer, Expert, GatedDeltaNet, Linear, LinearLayer, MLP, MoEMLP,
    QuantizedLinear, RMSNorm, RMSNormGated,
};
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
    pub mrope_section: Option<[usize; 3]>,
}

impl Qwen35Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cache(
        &self,
        x: &Array,
        cache_keys: Option<&Array>,
        cache_values: Option<&Array>,
        offset: i32,
        position_ids: Option<&Array>,
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

        // Apply positional encoding: M-RoPE or standard partial RoPE
        let (queries, k) = if let (Some(pos_ids), Some(section)) =
            (position_ids, &self.mrope_section)
        {
            // M-RoPE path: apply multi-resolution rotary embeddings
            let queries = mrope::apply_mrope(&queries, pos_ids, section, self.rope_base, stream)?;
            let k = mrope::apply_mrope(&k, pos_ids, section, self.rope_base, stream)?;
            (queries, k)
        } else {
            // Standard partial RoPE path
            let rope_dim = (self.head_dim as f32 * self.partial_rotary_factor) as i32;
            if rope_dim < self.head_dim {
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
            }
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

/// MLP variant: standard SwiGLU or Mixture-of-Experts.
pub enum Qwen35MLP {
    Standard(MLP),
    MoE(MoEMLP),
}

impl Qwen35MLP {
    pub fn forward_with_stream(&self, x: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Qwen35MLP::Standard(mlp) => mlp.forward_with_stream(x, stream),
            Qwen35MLP::MoE(moe) => moe.forward_with_stream(x, stream),
        }
    }
}

pub struct Qwen35DecoderLayer {
    pub attention: LayerAttention,
    pub mlp: Qwen35MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub is_linear: bool,
}

impl Qwen35DecoderLayer {
    pub fn n_kv_heads(&self) -> usize {
        match &self.attention {
            LayerAttention::Full(attn) => attn.n_kv_heads as usize,
            LayerAttention::GatedDelta(_) => 0,
        }
    }

    pub fn head_dim(&self) -> usize {
        match &self.attention {
            LayerAttention::Full(attn) => attn.head_dim as usize,
            LayerAttention::GatedDelta(_) => 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cache(
        &self,
        x: &Array,
        cache: &mut (Option<Array>, Option<Array>),
        mask: Option<&Array>,
        position_ids: Option<&Array>,
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
                    position_ids,
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
        stream: &Stream,
    ) -> Result<Array> {
        let mut h = self.embed_tokens.forward_with_stream(tokens, stream)?;

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_with_cache(&h, &mut cache[i], None, None, stream)?;
        }

        h = self.norm.forward_with_stream(&h, stream)?;
        let logits = self.lm_head.forward_with_stream(&h, stream)?;
        Ok(logits)
    }

    /// Forward pass with pre-computed embeddings (for VLM).
    /// Skips embed_tokens, uses provided embeddings directly.
    pub fn forward_with_embeddings(
        &self,
        embeddings: &Array,
        cache: &mut [(Option<Array>, Option<Array>)],
        position_ids: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array> {
        let mut h = embeddings.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_with_cache(&h, &mut cache[i], None, position_ids, stream)?;
        }

        h = self.norm.forward_with_stream(&h, stream)?;
        self.lm_head.forward_with_stream(&h, stream)
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
            // Extract mrope_section from config if available
            let mrope_section = tc
                .rope_parameters
                .as_ref()
                .and_then(|rp| rp.mrope_section.as_ref())
                .and_then(|s| {
                    if s.len() == 3 {
                        Some([s[0], s[1], s[2]])
                    } else {
                        None
                    }
                });

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
                mrope_section,
            };

            LayerAttention::Full(attn)
        };

        // Build MLP: standard or MoE depending on config
        let mlp = if let Some(num_experts) = tc.num_experts {
            // MoE layer
            let num_experts_per_tok = tc.num_experts_per_tok.unwrap_or(8);
            let moe_intermediate = tc.moe_intermediate_size.unwrap_or(tc.intermediate_size);
            let _shared_intermediate = tc
                .shared_expert_intermediate_size
                .unwrap_or(moe_intermediate);
            let norm_topk_prob = tc.norm_topk_prob.unwrap_or(true);

            // Router gate: hidden_size -> num_experts
            let gate_weight = w(&format!("{}.mlp.gate.weight", lp))?;
            let gate = Linear::new(gate_weight, None);

            // Build experts from fused gate_up_proj weights
            // Raw HF format: experts.gate_up_proj [num_experts, 2*intermediate, hidden]
            //                 experts.down_proj   [num_experts, hidden, intermediate]
            let experts = build_moe_experts(
                weights,
                &format!("language_model.model.{}", lp),
                num_experts,
                moe_intermediate,
                group_size,
                bits,
            )?;

            // Shared expert (standard SwiGLU MLP)
            let shared_gate_proj = linear(&format!("{}.mlp.shared_expert.gate_proj", lp))?;
            let shared_up_proj = linear(&format!("{}.mlp.shared_expert.up_proj", lp))?;
            let shared_down_proj = linear(&format!("{}.mlp.shared_expert.down_proj", lp))?;
            let shared_expert = MLP::new(shared_gate_proj, shared_up_proj, shared_down_proj);

            // Shared expert gate: hidden_size -> 1
            let shared_expert_gate_weight = w(&format!("{}.mlp.shared_expert_gate.weight", lp))?;
            let shared_expert_gate = Linear::new(shared_expert_gate_weight, None);

            Qwen35MLP::MoE(MoEMLP {
                gate,
                experts,
                shared_expert,
                shared_expert_gate,
                num_experts_per_tok,
                norm_topk_prob,
            })
        } else {
            // Standard MLP
            let gate_proj = linear(&format!("{}.mlp.gate_proj", lp))?;
            let up_proj = linear(&format!("{}.mlp.up_proj", lp))?;
            let down_proj = linear(&format!("{}.mlp.down_proj", lp))?;
            Qwen35MLP::Standard(MLP::new(gate_proj, up_proj, down_proj))
        };

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

/// Build MoE experts from fused HuggingFace weight format.
///
/// Raw HF weights use fused `gate_up_proj` per expert:
/// - `{prefix}.mlp.experts.gate_up_proj`       [num_experts, 2*intermediate, hidden]
/// - `{prefix}.mlp.experts.down_proj`           [num_experts, hidden, intermediate]
///
/// For quantized models, scales/biases follow the same pattern.
///
/// We split `gate_up_proj` along dim 1 into `gate_proj` and `up_proj`,
/// then slice each expert along dim 0.
#[allow(clippy::too_many_arguments)]
fn build_moe_experts(
    weights: &HashMap<String, Array>,
    prefix: &str,
    num_experts: usize,
    _moe_intermediate: usize,
    group_size: i32,
    bits: i32,
) -> Result<Vec<Expert>> {
    let stream = Stream::new(&Device::gpu());

    // Check for quantized format (has scales)
    let gate_up_scales_key = format!("{}.mlp.experts.gate_up_proj.scales", prefix);
    let is_quantized = weights.contains_key(&gate_up_scales_key);

    if is_quantized {
        // Quantized expert weights
        let gate_up_w = weights
            .get(&format!("{}.mlp.experts.gate_up_proj.weight", prefix))
            .ok_or_else(|| {
                crate::error::Error::Mlx(format!(
                    "missing weight: {}.mlp.experts.gate_up_proj.weight",
                    prefix
                ))
            })?;
        let gate_up_s = weights.get(&gate_up_scales_key).ok_or_else(|| {
            crate::error::Error::Mlx(format!("missing weight: {}", gate_up_scales_key))
        })?;
        let gate_up_b = weights
            .get(&format!("{}.mlp.experts.gate_up_proj.biases", prefix))
            .ok_or_else(|| {
                crate::error::Error::Mlx(format!(
                    "missing weight: {}.mlp.experts.gate_up_proj.biases",
                    prefix
                ))
            })?;

        let down_w = weights
            .get(&format!("{}.mlp.experts.down_proj.weight", prefix))
            .ok_or_else(|| {
                crate::error::Error::Mlx(format!(
                    "missing weight: {}.mlp.experts.down_proj.weight",
                    prefix
                ))
            })?;
        let down_s = weights
            .get(&format!("{}.mlp.experts.down_proj.scales", prefix))
            .ok_or_else(|| {
                crate::error::Error::Mlx(format!(
                    "missing weight: {}.mlp.experts.down_proj.scales",
                    prefix
                ))
            })?;
        let down_b = weights
            .get(&format!("{}.mlp.experts.down_proj.biases", prefix))
            .ok_or_else(|| {
                crate::error::Error::Mlx(format!(
                    "missing weight: {}.mlp.experts.down_proj.biases",
                    prefix
                ))
            })?;

        // gate_up_proj shape: [num_experts, 2*intermediate/pack, hidden] (quantized)
        // Split along dim 1 into gate and up halves
        let gate_up_w_shape = gate_up_w.shape();
        let half_dim1_w = gate_up_w_shape[1] / 2;
        let gate_up_s_shape = gate_up_s.shape();
        let half_dim1_s = gate_up_s_shape[1] / 2;

        let mut experts = Vec::with_capacity(num_experts);
        for ei in 0..num_experts {
            let e = ei as i32;
            // Slice expert ei from fused tensors: [1, dim1, dim2] -> squeeze to [dim1, dim2]
            // gate_up weight
            let gu_w = ops::slice(
                gate_up_w,
                &[e, 0, 0],
                &[e + 1, gate_up_w_shape[1], gate_up_w_shape[2]],
                &[1, 1, 1],
                &stream,
            )?;
            let gu_w = ops::squeeze_axis(&gu_w, 0, &stream)?;
            // Split into gate and up
            let gate_w = ops::slice(
                &gu_w,
                &[0, 0],
                &[half_dim1_w, gate_up_w_shape[2]],
                &[1, 1],
                &stream,
            )?;
            let up_w = ops::slice(
                &gu_w,
                &[half_dim1_w, 0],
                &[gate_up_w_shape[1], gate_up_w_shape[2]],
                &[1, 1],
                &stream,
            )?;

            // gate_up scales
            let gu_s = ops::slice(
                gate_up_s,
                &[e, 0, 0],
                &[e + 1, gate_up_s_shape[1], gate_up_s_shape[2]],
                &[1, 1, 1],
                &stream,
            )?;
            let gu_s = ops::squeeze_axis(&gu_s, 0, &stream)?;
            let gate_s = ops::slice(
                &gu_s,
                &[0, 0],
                &[half_dim1_s, gate_up_s_shape[2]],
                &[1, 1],
                &stream,
            )?;
            let up_s = ops::slice(
                &gu_s,
                &[half_dim1_s, 0],
                &[gate_up_s_shape[1], gate_up_s_shape[2]],
                &[1, 1],
                &stream,
            )?;

            // gate_up biases (same shape as scales)
            let gu_b = ops::slice(
                gate_up_b,
                &[e, 0, 0],
                &[e + 1, gate_up_s_shape[1], gate_up_s_shape[2]],
                &[1, 1, 1],
                &stream,
            )?;
            let gu_b = ops::squeeze_axis(&gu_b, 0, &stream)?;
            let gate_b = ops::slice(
                &gu_b,
                &[0, 0],
                &[half_dim1_s, gate_up_s_shape[2]],
                &[1, 1],
                &stream,
            )?;
            let up_b = ops::slice(
                &gu_b,
                &[half_dim1_s, 0],
                &[gate_up_s_shape[1], gate_up_s_shape[2]],
                &[1, 1],
                &stream,
            )?;

            // down_proj: [num_experts, out_dim, in_dim_packed]
            let down_w_shape = down_w.shape();
            let dw = ops::slice(
                down_w,
                &[e, 0, 0],
                &[e + 1, down_w_shape[1], down_w_shape[2]],
                &[1, 1, 1],
                &stream,
            )?;
            let dw = ops::squeeze_axis(&dw, 0, &stream)?;

            let down_s_shape = down_s.shape();
            let ds = ops::slice(
                down_s,
                &[e, 0, 0],
                &[e + 1, down_s_shape[1], down_s_shape[2]],
                &[1, 1, 1],
                &stream,
            )?;
            let ds = ops::squeeze_axis(&ds, 0, &stream)?;

            let db = ops::slice(
                down_b,
                &[e, 0, 0],
                &[e + 1, down_s_shape[1], down_s_shape[2]],
                &[1, 1, 1],
                &stream,
            )?;
            let db = ops::squeeze_axis(&db, 0, &stream)?;

            let gate_proj = LinearLayer::Quantized(QuantizedLinear::new(
                gate_w, gate_s, gate_b, group_size, bits,
            ));
            let up_proj =
                LinearLayer::Quantized(QuantizedLinear::new(up_w, up_s, up_b, group_size, bits));
            let down_proj =
                LinearLayer::Quantized(QuantizedLinear::new(dw, ds, db, group_size, bits));

            experts.push(Expert::new(gate_proj, up_proj, down_proj));
        }

        Ok(experts)
    } else {
        // Non-quantized expert weights
        let gate_up = weights
            .get(&format!("{}.mlp.experts.gate_up_proj.weight", prefix))
            .ok_or_else(|| {
                crate::error::Error::Mlx(format!(
                    "missing weight: {}.mlp.experts.gate_up_proj.weight",
                    prefix
                ))
            })?;
        let down = weights
            .get(&format!("{}.mlp.experts.down_proj.weight", prefix))
            .ok_or_else(|| {
                crate::error::Error::Mlx(format!(
                    "missing weight: {}.mlp.experts.down_proj.weight",
                    prefix
                ))
            })?;

        // gate_up shape: [num_experts, 2*intermediate, hidden]
        let gu_shape = gate_up.shape();
        let half_dim1 = gu_shape[1] / 2;

        let mut experts = Vec::with_capacity(num_experts);
        for ei in 0..num_experts {
            let e = ei as i32;
            // Slice and split gate_up
            let gu = ops::slice(
                gate_up,
                &[e, 0, 0],
                &[e + 1, gu_shape[1], gu_shape[2]],
                &[1, 1, 1],
                &stream,
            )?;
            let gu = ops::squeeze_axis(&gu, 0, &stream)?;

            let gate_w = ops::slice(&gu, &[0, 0], &[half_dim1, gu_shape[2]], &[1, 1], &stream)?;
            let up_w = ops::slice(
                &gu,
                &[half_dim1, 0],
                &[gu_shape[1], gu_shape[2]],
                &[1, 1],
                &stream,
            )?;

            // down_proj: [num_experts, hidden, intermediate]
            let d_shape = down.shape();
            let dw = ops::slice(
                down,
                &[e, 0, 0],
                &[e + 1, d_shape[1], d_shape[2]],
                &[1, 1, 1],
                &stream,
            )?;
            let dw = ops::squeeze_axis(&dw, 0, &stream)?;

            let gate_proj = LinearLayer::Full(Linear::new(gate_w, None));
            let up_proj = LinearLayer::Full(Linear::new(up_w, None));
            let down_proj = LinearLayer::Full(Linear::new(dw, None));

            experts.push(Expert::new(gate_proj, up_proj, down_proj));
        }

        Ok(experts)
    }
}
