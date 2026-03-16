use std::collections::HashMap;

use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::nn::{Attention, EmbeddingLayer, LinearLayer, MLP, Module, RMSNorm};
use crate::ops;
use crate::stream::Stream;

use super::config::ModelConfig;

pub struct TransformerBlock {
    pub attention: Attention,
    pub mlp: MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl TransformerBlock {
    /// Forward pass with KV cache support.
    /// Returns (output, new_keys, new_values).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cache(
        &self,
        x: &Array,
        cache_keys: Option<&Array>,
        cache_values: Option<&Array>,
        offset: i32,
        mask_mode: &str,
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<(Array, Array, Array)> {
        // Pre-norm + attention
        let normed = self.input_layernorm.forward_with_stream(x, stream)?;
        let (attn_out, new_k, new_v) = self.attention.forward_with_cache(
            &normed,
            cache_keys,
            cache_values,
            offset,
            mask_mode,
            mask,
            stream,
        )?;
        let h = ops::add(x, &attn_out, stream)?;

        // Pre-norm + MLP
        let normed = self
            .post_attention_layernorm
            .forward_with_stream(&h, stream)?;
        let mlp_out = self.mlp.forward_with_stream(&normed, stream)?;
        let out = ops::add(&h, &mlp_out, stream)?;

        Ok((out, new_k, new_v))
    }

    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        let p = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{}.{}", prefix, name)
            }
        };
        self.attention.load_weights(weights, &p("self_attn"))?;
        self.mlp.load_weights(weights, &p("mlp"))?;
        self.input_layernorm
            .load_weights(weights, &p("input_layernorm"))?;
        self.post_attention_layernorm
            .load_weights(weights, &p("post_attention_layernorm"))?;
        Ok(())
    }
}

pub struct LlamaModel {
    pub embed_tokens: EmbeddingLayer,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: LinearLayer,
}

impl LlamaModel {
    /// Build a LlamaModel from config and weights.
    pub fn from_config(config: &ModelConfig, weights: &HashMap<String, Array>) -> Result<Self> {
        let n_heads = config.num_attention_heads as i32;
        let n_kv_heads = config.n_kv_heads() as i32;
        let head_dim = config.head_dim() as i32;
        let eps = config.rms_norm_eps as f32;

        // Quantization config
        let (group_size, bits) = match &config.quantization {
            Some(qc) => (qc.group_size, qc.bits),
            None => (64, 4), // defaults, only used if scales are present
        };

        // Helper to get a weight
        let w = |name: &str| -> Result<Array> {
            weights
                .get(name)
                .cloned()
                .ok_or_else(|| crate::error::Error::Mlx(format!("missing weight: {}", name)))
        };

        // Helper to create a LinearLayer (auto-detects quantized vs full)
        let linear = |prefix: &str| -> Result<LinearLayer> {
            LinearLayer::from_weights(weights, prefix, group_size, bits)
        };

        // Embedding (auto-detects quantized)
        let embed_tokens =
            EmbeddingLayer::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let lp = format!("model.layers.{}", i);

            let wq = linear(&format!("{}.self_attn.q_proj", lp))?;
            let wk = linear(&format!("{}.self_attn.k_proj", lp))?;
            let wv = linear(&format!("{}.self_attn.v_proj", lp))?;
            let wo = linear(&format!("{}.self_attn.o_proj", lp))?;

            let attention = Attention::new(
                wq,
                wk,
                wv,
                wo,
                n_heads,
                n_kv_heads,
                head_dim,
                head_dim, // rope_dims = head_dim for Llama
                false,    // traditional = false
                Some(config.rope_theta as f32),
                1.0, // rope_scale
            );

            let gate_proj = linear(&format!("{}.mlp.gate_proj", lp))?;
            let up_proj = linear(&format!("{}.mlp.up_proj", lp))?;
            let down_proj = linear(&format!("{}.mlp.down_proj", lp))?;
            let mlp = MLP::new(gate_proj, up_proj, down_proj);

            let input_layernorm = RMSNorm::new(w(&format!("{}.input_layernorm.weight", lp))?, eps);
            let post_attention_layernorm =
                RMSNorm::new(w(&format!("{}.post_attention_layernorm.weight", lp))?, eps);

            layers.push(TransformerBlock {
                attention,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            });
        }

        // Final norm
        let norm = RMSNorm::new(w("model.norm.weight")?, eps);

        // LM head — when tied, shares embedding weights (may be quantized)
        let lm_head = if config.tie_word_embeddings {
            LinearLayer::from_weights(weights, "model.embed_tokens", group_size, bits)?
        } else {
            LinearLayer::from_weights(weights, "lm_head", group_size, bits)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Forward pass. Returns logits [batch, seq_len, vocab_size].
    /// `cache` is a mutable slice of (Option<keys>, Option<values>) per layer.
    /// On return, each element is updated with the new KV cache state.
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        tokens: &Array,
        cache: &mut [(Option<Array>, Option<Array>)],
        mask_mode: &str,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let stream = Stream::new(&Device::gpu());

        let mut h = self.embed_tokens.forward_with_stream(tokens, &stream)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let (ck, cv) = &cache[i];
            let offset = ck.as_ref().map_or(0, |k| k.shape()[2]);
            let (out, new_k, new_v) = layer.forward_with_cache(
                &h,
                ck.as_ref(),
                cv.as_ref(),
                offset,
                mask_mode,
                mask,
                &stream,
            )?;
            h = out;
            cache[i] = (Some(new_k), Some(new_v));
        }

        // Final norm + lm_head
        h = self.norm.forward_with_stream(&h, &stream)?;
        let logits = self.lm_head.forward_with_stream(&h, &stream)?;
        Ok(logits)
    }
}
