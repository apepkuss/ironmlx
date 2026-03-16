use std::collections::HashMap;

use crate::array::Array;
use crate::error::Result;
use crate::nn::{Attention, EmbeddingLayer, LinearLayer, MLP, RMSNorm};

use super::config::ModelConfig;
use super::llama::{LlamaModel, TransformerBlock};

/// Build a Qwen3 model from config and weights.
///
/// Qwen3 is structurally identical to Llama with the addition of QK Norm
/// (RMSNorm on Q and K before RoPE). Reuses `LlamaModel` for the forward pass.
pub fn from_config(config: &ModelConfig, weights: &HashMap<String, Array>) -> Result<LlamaModel> {
    let n_heads = config.num_attention_heads as i32;
    let n_kv_heads = config.n_kv_heads() as i32;
    let head_dim = config.head_dim() as i32;
    let eps = config.rms_norm_eps as f32;

    let (group_size, bits) = match &config.quantization {
        Some(qc) => (qc.group_size, qc.bits),
        None => (64, 4),
    };

    let w = |name: &str| -> Result<Array> {
        weights
            .get(name)
            .cloned()
            .ok_or_else(|| crate::error::Error::Mlx(format!("missing weight: {}", name)))
    };

    let linear = |prefix: &str| -> Result<LinearLayer> {
        LinearLayer::from_weights(weights, prefix, group_size, bits)
    };

    // Embedding
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

        // QK Norm — Qwen3 specific
        let q_norm = RMSNorm::new(w(&format!("{}.self_attn.q_norm.weight", lp))?, eps);
        let k_norm = RMSNorm::new(w(&format!("{}.self_attn.k_norm.weight", lp))?, eps);

        let attention = Attention::new(
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim,
            false,
            Some(config.rope_theta as f32),
            1.0,
            1.0,
        )
        .with_qk_norm(q_norm, k_norm);

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

    // LM head
    let lm_head = if config.tie_word_embeddings {
        LinearLayer::from_weights(weights, "model.embed_tokens", group_size, bits)?
    } else {
        LinearLayer::from_weights(weights, "lm_head", group_size, bits)?
    };

    Ok(LlamaModel {
        embed_tokens,
        layers,
        norm,
        lm_head,
    })
}
