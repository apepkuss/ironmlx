use std::collections::HashMap;

use serde::Deserialize;

use crate::array::Array;
use crate::error::Result;
use crate::nn::{EmbeddingLayer, LayerNorm, LinearLayer, gelu};
use crate::ops;
use crate::stream::Stream;

use super::config::QuantizationConfig;

// ---------------------------------------------------------------------------
// BertConfig
// ---------------------------------------------------------------------------

fn default_hidden_size() -> usize {
    384
}
fn default_num_hidden_layers() -> usize {
    6
}
fn default_num_attention_heads() -> usize {
    12
}
fn default_intermediate_size() -> usize {
    1536
}
fn default_vocab_size() -> usize {
    30522
}
fn default_max_position_embeddings() -> usize {
    512
}
fn default_type_vocab_size() -> usize {
    2
}
fn default_layer_norm_eps() -> f32 {
    1e-12
}
fn default_hidden_act() -> String {
    "gelu".to_string()
}
fn default_bert_model_type() -> String {
    "bert".to_string()
}

#[derive(Debug, Deserialize)]
pub struct BertConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_bert_model_type")]
    pub model_type: String,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

// ---------------------------------------------------------------------------
// BertEmbeddings
// ---------------------------------------------------------------------------

pub struct BertEmbeddings {
    pub word_embeddings: EmbeddingLayer,
    pub position_embeddings: EmbeddingLayer,
    pub token_type_embeddings: EmbeddingLayer,
    pub layer_norm: LayerNorm,
}

impl BertEmbeddings {
    /// Forward: word_emb + position_emb + token_type_emb -> LayerNorm
    pub fn forward(&self, token_ids: &Array, stream: &Stream) -> Result<Array> {
        let word_emb = self
            .word_embeddings
            .forward_with_stream(token_ids, stream)?;

        // Position IDs: [0, 1, 2, ..., seq_len-1]
        let shape = token_ids.shape();
        let seq_len = shape[shape.len() - 1];
        let position_ids =
            ops::arange(0.0, seq_len as f64, 1.0, crate::dtype::Dtype::Int32, stream)?;
        let pos_emb = self
            .position_embeddings
            .forward_with_stream(&position_ids, stream)?;

        // Token type IDs: all zeros (single-sentence embedding)
        let tt_ids = ops::zeros(&[seq_len], crate::dtype::Dtype::Int32, stream)?;
        let tt_emb = self
            .token_type_embeddings
            .forward_with_stream(&tt_ids, stream)?;

        // Sum embeddings
        let sum1 = ops::add(&word_emb, &pos_emb, stream)?;
        let sum2 = ops::add(&sum1, &tt_emb, stream)?;

        // LayerNorm
        self.layer_norm.forward_with_stream(&sum2, stream)
    }
}

// ---------------------------------------------------------------------------
// BertSelfAttention
// ---------------------------------------------------------------------------

pub struct BertSelfAttention {
    pub query: LinearLayer,
    pub key: LinearLayer,
    pub value: LinearLayer,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl BertSelfAttention {
    /// Bidirectional self-attention (no causal mask).
    /// Input: [batch, seq_len, hidden_size]
    /// Output: [batch, seq_len, hidden_size]
    pub fn forward(&self, x: &Array, stream: &Stream) -> Result<Array> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let num_heads = self.num_heads as i32;
        let head_dim = self.head_dim as i32;

        // Q, K, V projections: [batch, seq_len, hidden_size]
        let q = self.query.forward_with_stream(x, stream)?;
        let k = self.key.forward_with_stream(x, stream)?;
        let v = self.value.forward_with_stream(x, stream)?;

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q = ops::reshape(&q, &[batch, seq_len, num_heads, head_dim], stream)?;
        let k = ops::reshape(&k, &[batch, seq_len, num_heads, head_dim], stream)?;
        let v = ops::reshape(&v, &[batch, seq_len, num_heads, head_dim], stream)?;

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = ops::transpose_axes(&q, &[0, 2, 1, 3], stream)?;
        let k = ops::transpose_axes(&k, &[0, 2, 1, 3], stream)?;
        let v = ops::transpose_axes(&v, &[0, 2, 1, 3], stream)?;

        // Attention scores: Q @ K^T / sqrt(head_dim)
        let k_t = ops::transpose_axes(&k, &[0, 1, 3, 2], stream)?;
        let scores = ops::matmul(&q, &k_t, stream)?;
        let scale = Array::from_float(1.0 / (self.head_dim as f32).sqrt());
        let scores = ops::multiply(&scores, &scale, stream)?;

        // Softmax (no causal mask — bidirectional)
        let attn_weights = ops::softmax(&scores, &[-1], stream)?;

        // Weighted sum: attn_weights @ V -> [batch, num_heads, seq_len, head_dim]
        let attn_out = ops::matmul(&attn_weights, &v, stream)?;

        // Transpose back to [batch, seq_len, num_heads, head_dim]
        let attn_out = ops::transpose_axes(&attn_out, &[0, 2, 1, 3], stream)?;

        // Reshape to [batch, seq_len, hidden_size]
        let hidden_size = num_heads * head_dim;
        ops::reshape(&attn_out, &[batch, seq_len, hidden_size], stream)
    }
}

// ---------------------------------------------------------------------------
// BertAttentionOutput
// ---------------------------------------------------------------------------

pub struct BertAttentionOutput {
    pub dense: LinearLayer,
    pub layer_norm: LayerNorm,
}

impl BertAttentionOutput {
    /// dense(attn_out) + residual -> LayerNorm
    pub fn forward(&self, attn_out: &Array, residual: &Array, stream: &Stream) -> Result<Array> {
        let projected = self.dense.forward_with_stream(attn_out, stream)?;
        let with_residual = ops::add(&projected, residual, stream)?;
        self.layer_norm.forward_with_stream(&with_residual, stream)
    }
}

// ---------------------------------------------------------------------------
// BertIntermediate
// ---------------------------------------------------------------------------

pub struct BertIntermediate {
    pub dense: LinearLayer,
    pub use_gelu: bool,
}

impl BertIntermediate {
    /// dense(x) -> activation
    pub fn forward(&self, x: &Array, stream: &Stream) -> Result<Array> {
        let out = self.dense.forward_with_stream(x, stream)?;
        if self.use_gelu {
            gelu(&out, stream)
        } else {
            // Default to GELU for BERT
            gelu(&out, stream)
        }
    }
}

// ---------------------------------------------------------------------------
// BertOutput
// ---------------------------------------------------------------------------

pub struct BertOutput {
    pub dense: LinearLayer,
    pub layer_norm: LayerNorm,
}

impl BertOutput {
    /// dense(intermediate_out) + residual -> LayerNorm
    pub fn forward(
        &self,
        intermediate_out: &Array,
        residual: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let projected = self.dense.forward_with_stream(intermediate_out, stream)?;
        let with_residual = ops::add(&projected, residual, stream)?;
        self.layer_norm.forward_with_stream(&with_residual, stream)
    }
}

// ---------------------------------------------------------------------------
// BertLayer
// ---------------------------------------------------------------------------

pub struct BertLayer {
    pub attention_self: BertSelfAttention,
    pub attention_output: BertAttentionOutput,
    pub intermediate: BertIntermediate,
    pub output: BertOutput,
}

impl BertLayer {
    /// Forward pass for a single BERT encoder layer.
    pub fn forward(&self, x: &Array, stream: &Stream) -> Result<Array> {
        // Self-attention + residual + LayerNorm
        let attn_out = self.attention_self.forward(x, stream)?;
        let attn_normed = self.attention_output.forward(&attn_out, x, stream)?;

        // Intermediate (FFN up) + activation
        let intermediate_out = self.intermediate.forward(&attn_normed, stream)?;

        // Output (FFN down) + residual + LayerNorm
        self.output.forward(&intermediate_out, &attn_normed, stream)
    }
}

// ---------------------------------------------------------------------------
// BertEncoder
// ---------------------------------------------------------------------------

pub struct BertEncoder {
    pub layers: Vec<BertLayer>,
}

impl BertEncoder {
    /// Forward through all encoder layers.
    pub fn forward(&self, x: &Array, stream: &Stream) -> Result<Array> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, stream)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// BertPooler
// ---------------------------------------------------------------------------

pub struct BertPooler {
    pub dense: LinearLayer,
}

impl BertPooler {
    /// Pool by taking the [CLS] token (first token) and applying dense + tanh.
    pub fn forward(&self, hidden_states: &Array, stream: &Stream) -> Result<Array> {
        let shape = hidden_states.shape();
        let batch = shape[0];
        let hidden_size = shape[2];

        // Take first token: hidden_states[:, 0, :]
        let cls = ops::slice(
            hidden_states,
            &[0, 0, 0],
            &[batch, 1, hidden_size],
            &[1, 1, 1],
            stream,
        )?;
        let cls = ops::squeeze_axis(&cls, 1, stream)?;

        let pooled = self.dense.forward_with_stream(&cls, stream)?;
        ops::tanh(&pooled, stream)
    }
}

// ---------------------------------------------------------------------------
// BertModel
// ---------------------------------------------------------------------------

pub struct BertModel {
    pub embeddings: BertEmbeddings,
    pub encoder: BertEncoder,
    pub pooler: Option<BertPooler>,
    pub config: BertConfig,
}

impl BertModel {
    /// Forward pass. Returns hidden states [batch, seq_len, hidden_size].
    pub fn forward(&self, token_ids: &Array, stream: &Stream) -> Result<Array> {
        let embedded = self.embeddings.forward(token_ids, stream)?;
        self.encoder.forward(&embedded, stream)
    }

    /// Forward pass with pooled output.
    /// Returns (hidden_states, pooled_output) where pooled is [batch, hidden_size].
    pub fn forward_with_pooling(
        &self,
        token_ids: &Array,
        stream: &Stream,
    ) -> Result<(Array, Option<Array>)> {
        let hidden = self.forward(token_ids, stream)?;
        let pooled = match &self.pooler {
            Some(pooler) => Some(pooler.forward(&hidden, stream)?),
            None => None,
        };
        Ok((hidden, pooled))
    }

    /// Number of encoder layers.
    pub fn num_layers(&self) -> usize {
        self.encoder.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Build from config file + weights
// ---------------------------------------------------------------------------

/// Build a BertModel from a config.json path and pre-loaded weights.
pub fn from_config_file(config_path: &str, weights: &HashMap<String, Array>) -> Result<BertModel> {
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| crate::error::Error::Mlx(format!("failed to read config: {}", e)))?;
    let config: BertConfig = serde_json::from_str(&content)
        .map_err(|e| crate::error::Error::Mlx(format!("failed to parse BERT config: {}", e)))?;

    from_config(&config, weights)
}

/// Build a BertModel from a parsed BertConfig and pre-loaded weights.
pub fn from_config(config: &BertConfig, weights: &HashMap<String, Array>) -> Result<BertModel> {
    let (group_size, bits) = match &config.quantization {
        Some(qc) => (qc.group_size, qc.bits),
        None => (64, 4),
    };

    let head_dim = config.hidden_size / config.num_attention_heads;
    let use_gelu = config.hidden_act == "gelu" || config.hidden_act == "gelu_new";

    // Weight prefix: BERT weights may be prefixed with "bert." or not.
    // Detect by checking if "bert.embeddings.word_embeddings.weight" exists.
    let prefix = if weights.contains_key("bert.embeddings.word_embeddings.weight") {
        "bert"
    } else {
        ""
    };

    let pfx = |name: &str| -> String {
        if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", prefix, name)
        }
    };

    // --- Embeddings ---
    let word_embeddings = EmbeddingLayer::from_weights(
        weights,
        &pfx("embeddings.word_embeddings"),
        group_size,
        bits,
    )?;
    let position_embeddings = EmbeddingLayer::from_weights(
        weights,
        &pfx("embeddings.position_embeddings"),
        group_size,
        bits,
    )?;
    let token_type_embeddings = EmbeddingLayer::from_weights(
        weights,
        &pfx("embeddings.token_type_embeddings"),
        group_size,
        bits,
    )?;

    let emb_ln = build_layer_norm(weights, &pfx("embeddings.LayerNorm"), config.layer_norm_eps);

    let embeddings = BertEmbeddings {
        word_embeddings,
        position_embeddings,
        token_type_embeddings,
        layer_norm: emb_ln,
    };

    // --- Encoder layers ---
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let lp = pfx(&format!("encoder.layer.{}", i));

        let attention_self = BertSelfAttention {
            query: LinearLayer::from_weights(
                weights,
                &format!("{}.attention.self.query", lp),
                group_size,
                bits,
            )?,
            key: LinearLayer::from_weights(
                weights,
                &format!("{}.attention.self.key", lp),
                group_size,
                bits,
            )?,
            value: LinearLayer::from_weights(
                weights,
                &format!("{}.attention.self.value", lp),
                group_size,
                bits,
            )?,
            num_heads: config.num_attention_heads,
            head_dim,
        };

        let attention_output = BertAttentionOutput {
            dense: LinearLayer::from_weights(
                weights,
                &format!("{}.attention.output.dense", lp),
                group_size,
                bits,
            )?,
            layer_norm: build_layer_norm(
                weights,
                &format!("{}.attention.output.LayerNorm", lp),
                config.layer_norm_eps,
            ),
        };

        let intermediate = BertIntermediate {
            dense: LinearLayer::from_weights(
                weights,
                &format!("{}.intermediate.dense", lp),
                group_size,
                bits,
            )?,
            use_gelu,
        };

        let output = BertOutput {
            dense: LinearLayer::from_weights(
                weights,
                &format!("{}.output.dense", lp),
                group_size,
                bits,
            )?,
            layer_norm: build_layer_norm(
                weights,
                &format!("{}.output.LayerNorm", lp),
                config.layer_norm_eps,
            ),
        };

        layers.push(BertLayer {
            attention_self,
            attention_output,
            intermediate,
            output,
        });
    }

    let encoder = BertEncoder { layers };

    // --- Pooler (optional) ---
    let pooler_key = pfx("pooler.dense");
    let pooler = if weights.contains_key(&format!("{}.weight", pooler_key))
        || weights.contains_key(&format!("{}.scales", pooler_key))
    {
        Some(BertPooler {
            dense: LinearLayer::from_weights(weights, &pooler_key, group_size, bits)?,
        })
    } else {
        None
    };

    // Clone config fields for ownership
    let owned_config = BertConfig {
        hidden_size: config.hidden_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        max_position_embeddings: config.max_position_embeddings,
        type_vocab_size: config.type_vocab_size,
        layer_norm_eps: config.layer_norm_eps,
        hidden_act: config.hidden_act.clone(),
        model_type: config.model_type.clone(),
        quantization: config.quantization.clone(),
    };

    Ok(BertModel {
        embeddings,
        encoder,
        pooler,
        config: owned_config,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a LayerNorm from weights map. LayerNorm always has weight and bias.
fn build_layer_norm(weights: &HashMap<String, Array>, prefix: &str, eps: f32) -> LayerNorm {
    let w_key = format!("{}.weight", prefix);
    let b_key = format!("{}.bias", prefix);
    let weight = weights.get(&w_key).cloned();
    let bias = weights.get(&b_key).cloned();
    LayerNorm::new(weight, bias, eps)
}
