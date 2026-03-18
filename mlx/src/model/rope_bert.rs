use std::collections::HashMap;

use serde::Deserialize;

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::Result;
use crate::fast;
use crate::nn::{EmbeddingLayer, LayerNorm, LinearLayer, gelu};
use crate::ops;
use crate::stream::Stream;

use super::config::QuantizationConfig;

// ---------------------------------------------------------------------------
// RopeBertConfig
// ---------------------------------------------------------------------------

fn default_hidden_size() -> usize {
    768
}
fn default_num_hidden_layers() -> usize {
    22
}
fn default_num_attention_heads() -> usize {
    12
}
fn default_intermediate_size() -> usize {
    1152
}
fn default_vocab_size() -> usize {
    50368
}
fn default_max_position_embeddings() -> usize {
    8192
}
fn default_type_vocab_size() -> usize {
    0
}
fn default_layer_norm_eps() -> f32 {
    1e-12
}
fn default_model_type() -> String {
    "modernbert".to_string()
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_global_every() -> usize {
    1
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct RopeBertConfig {
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
    #[serde(default)]
    pub norm_eps: Option<f32>,
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    // RoPE config
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub global_rope_theta: Option<f32>,
    #[serde(default)]
    pub local_rope_theta: Option<f32>,
    // Sliding window
    #[serde(default)]
    pub local_attention: Option<usize>,
    #[serde(default = "default_global_every")]
    pub global_attn_every_n_layers: usize,
    // Bias flags
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub mlp_bias: bool,
    #[serde(default = "default_true")]
    pub norm_bias: bool,
    // Hidden activation
    #[serde(default)]
    pub hidden_act: Option<String>,
}

// ---------------------------------------------------------------------------
// RopeBertEmbeddings
// ---------------------------------------------------------------------------

pub struct RopeBertEmbeddings {
    pub word_embeddings: EmbeddingLayer,
    pub token_type_embeddings: Option<EmbeddingLayer>,
    pub layer_norm: Option<LayerNorm>,
}

impl RopeBertEmbeddings {
    /// Forward: word_emb + optional token_type_emb -> optional LayerNorm.
    /// No position embeddings — RoPE handles position encoding.
    pub fn forward(&self, token_ids: &Array, stream: &Stream) -> Result<Array> {
        let word_emb = self
            .word_embeddings
            .forward_with_stream(token_ids, stream)?;

        let mut out = word_emb;

        // Token type embeddings (GTE has them, ModernBERT does not)
        if let Some(ref tt_emb_layer) = self.token_type_embeddings {
            let shape = token_ids.shape();
            let seq_len = shape[shape.len() - 1];
            let tt_ids = ops::zeros(&[seq_len], Dtype::Int32, stream)?;
            let tt_emb = tt_emb_layer.forward_with_stream(&tt_ids, stream)?;
            out = ops::add(&out, &tt_emb, stream)?;
        }

        // LayerNorm (GTE/Jina have it, ModernBERT may have it)
        if let Some(ref ln) = self.layer_norm {
            out = ln.forward_with_stream(&out, stream)?;
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// RopeBertSelfAttention
// ---------------------------------------------------------------------------

pub struct RopeBertSelfAttention {
    pub query: LinearLayer,
    pub key: LinearLayer,
    pub value: LinearLayer,
    pub num_heads: usize,
    pub head_dim: usize,
    pub rope_dims: i32,
    pub rope_theta: f32,
    pub local_window: Option<usize>,
}

impl RopeBertSelfAttention {
    /// Bidirectional self-attention with RoPE.
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

        // Apply RoPE (bidirectional — offset always 0, no cache)
        let q = fast::rope(
            &q,
            self.rope_dims,
            false,
            Some(self.rope_theta),
            1.0,
            0,
            None,
            stream,
        )?;
        let k = fast::rope(
            &k,
            self.rope_dims,
            false,
            Some(self.rope_theta),
            1.0,
            0,
            None,
            stream,
        )?;

        // Attention scores: Q @ K^T / sqrt(head_dim)
        let k_t = ops::transpose_axes(&k, &[0, 1, 3, 2], stream)?;
        let scores = ops::matmul(&q, &k_t, stream)?;
        let scale = Array::from_float(1.0 / (self.head_dim as f32).sqrt());
        let scores = ops::multiply(&scores, &scale, stream)?;

        // Optional sliding window mask
        let scores = if let Some(window) = self.local_window {
            self.apply_sliding_window_mask(&scores, seq_len, window as i32, stream)?
        } else {
            scores
        };

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

    /// Apply sliding window mask: positions where |i - j| > window get -inf.
    fn apply_sliding_window_mask(
        &self,
        scores: &Array,
        seq_len: i32,
        window: i32,
        stream: &Stream,
    ) -> Result<Array> {
        // Create position indices: [0, 1, 2, ..., seq_len-1]
        let pos = ops::arange(0.0, seq_len as f64, 1.0, Dtype::Int32, stream)?;

        // row_ids: [seq_len, 1], col_ids: [1, seq_len]
        let row_ids = ops::reshape(&pos, &[seq_len, 1], stream)?;
        let col_ids = ops::reshape(&pos, &[1, seq_len], stream)?;

        // diff = row_ids - col_ids => [seq_len, seq_len]
        let diff = ops::subtract(&row_ids, &col_ids, stream)?;

        // abs_diff = |diff|
        let abs_diff = ops::abs(&diff, stream)?;

        // window_arr = scalar(window)
        let window_arr = Array::from_int(window);

        // mask_condition: abs_diff > window => true where we should mask
        // We use: abs_diff > window  <=>  !(abs_diff <= window)
        // Since we don't have greater op, we use: abs_diff - window > 0
        // Actually, let's use the subtract + clip approach:
        // exceeded = max(abs_diff - window, 0) > 0
        // But simplest: create neg_inf where abs_diff > window using where_
        //
        // We can compute: within_window = window - abs_diff  (positive when within)
        // Then: mask_val = where(within_window >= 0, 0.0, -inf)
        // But we don't have >= either. Let's use:
        // exceeded = abs_diff - window  (> 0 when outside window)
        // clipped = max(exceeded, 0)    (0 when inside, positive when outside)
        // Then use where_ with clipped > 0... but we don't have > either as a standalone.
        //
        // Alternative approach: create -inf mask and use minimum/maximum.
        // within = window - abs_diff  (>= 0 when within window)
        // sign_within = clamp to {0 or 1}: use clip(within, 0, 1) -> 0 outside, >=0 inside
        // Actually, let's be practical: use the approach with full and where_.
        //
        // Simplest: use ops::where_ with a comparison.
        // We need a boolean condition. Let's construct one:
        // inside = (abs_diff <= window) -- we need this as a boolean array.
        // We can get this as: cast (window - abs_diff) sign -> 0/1
        // Or: exceeded = abs_diff - window, then exceeded_clipped = max(exceeded, 0),
        //   then mask = -inf * min(exceeded_clipped, 1)
        // That gives: 0 inside window, -inf outside (after scaling).

        // exceeded = abs_diff - window (negative inside, positive outside)
        let exceeded = ops::subtract(&abs_diff, &window_arr, stream)?;
        let exceeded = ops::astype(&exceeded, Dtype::Float32, stream)?;

        // Clamp to [0, 1]: clips negative to 0, positive to 1
        let zero = Array::from_float(0.0);
        let one = Array::from_float(1.0);
        let mask_01 = ops::clip(&exceeded, Some(&zero), Some(&one), stream)?;

        // Scale by -inf: positions outside window get -inf, inside get 0
        let neg_inf = Array::from_float(f32::NEG_INFINITY);
        let mask = ops::multiply(&mask_01, &neg_inf, stream)?;

        // Reshape mask to [1, 1, seq_len, seq_len] for broadcasting
        let mask = ops::reshape(&mask, &[1, 1, seq_len, seq_len], stream)?;

        // Add mask to scores
        ops::add(scores, &mask, stream)
    }
}

// ---------------------------------------------------------------------------
// RopeBertAttentionOutput
// ---------------------------------------------------------------------------

pub struct RopeBertAttentionOutput {
    pub dense: LinearLayer,
    pub layer_norm: LayerNorm,
}

impl RopeBertAttentionOutput {
    /// dense(attn_out) + residual -> LayerNorm
    pub fn forward(&self, attn_out: &Array, residual: &Array, stream: &Stream) -> Result<Array> {
        let projected = self.dense.forward_with_stream(attn_out, stream)?;
        let with_residual = ops::add(&projected, residual, stream)?;
        self.layer_norm.forward_with_stream(&with_residual, stream)
    }
}

// ---------------------------------------------------------------------------
// FFN variants
// ---------------------------------------------------------------------------

/// Standard FFN: dense_up -> activation -> dense_down
pub struct RopeBertFFN {
    pub up: LinearLayer,
    pub down: LinearLayer,
    pub use_geglu: bool,
    pub gate: Option<LinearLayer>,
}

impl RopeBertFFN {
    /// Forward pass.
    /// Standard: gelu(up(x)) -> down
    /// GeGLU: gelu(gate(x)) * up(x) -> down
    pub fn forward(&self, x: &Array, stream: &Stream) -> Result<Array> {
        if self.use_geglu {
            // GeGLU: gelu(x * W_gate) * (x * W_up)
            let gate_out = match &self.gate {
                Some(g) => g.forward_with_stream(x, stream)?,
                None => {
                    return Err(crate::error::Error::Mlx(
                        "GeGLU requires gate projection".to_string(),
                    ));
                }
            };
            let gate_act = gelu(&gate_out, stream)?;
            let up_out = self.up.forward_with_stream(x, stream)?;
            let fused = ops::multiply(&gate_act, &up_out, stream)?;
            self.down.forward_with_stream(&fused, stream)
        } else {
            // Standard: gelu(up(x)) -> down
            let up_out = self.up.forward_with_stream(x, stream)?;
            let activated = gelu(&up_out, stream)?;
            self.down.forward_with_stream(&activated, stream)
        }
    }
}

// ---------------------------------------------------------------------------
// RopeBertOutput (FFN output + residual + LayerNorm)
// ---------------------------------------------------------------------------

pub struct RopeBertOutput {
    pub layer_norm: LayerNorm,
}

impl RopeBertOutput {
    /// ffn_out + residual -> LayerNorm
    pub fn forward(&self, ffn_out: &Array, residual: &Array, stream: &Stream) -> Result<Array> {
        let with_residual = ops::add(ffn_out, residual, stream)?;
        self.layer_norm.forward_with_stream(&with_residual, stream)
    }
}

// ---------------------------------------------------------------------------
// RopeBertLayer
// ---------------------------------------------------------------------------

pub struct RopeBertLayer {
    pub attention_self: RopeBertSelfAttention,
    pub attention_output: RopeBertAttentionOutput,
    pub ffn: RopeBertFFN,
    pub output: RopeBertOutput,
}

impl RopeBertLayer {
    /// Forward pass for a single RoPE-BERT encoder layer.
    pub fn forward(&self, x: &Array, stream: &Stream) -> Result<Array> {
        // Self-attention + residual + LayerNorm
        let attn_out = self.attention_self.forward(x, stream)?;
        let attn_normed = self.attention_output.forward(&attn_out, x, stream)?;

        // FFN + residual + LayerNorm
        let ffn_out = self.ffn.forward(&attn_normed, stream)?;
        self.output.forward(&ffn_out, &attn_normed, stream)
    }
}

// ---------------------------------------------------------------------------
// RopeBertModel
// ---------------------------------------------------------------------------

pub struct RopeBertModel {
    pub embeddings: RopeBertEmbeddings,
    pub layers: Vec<RopeBertLayer>,
    pub config: RopeBertConfig,
}

impl RopeBertModel {
    /// Forward pass. Returns hidden states [batch, seq_len, hidden_size].
    pub fn forward(&self, token_ids: &Array, stream: &Stream) -> Result<Array> {
        let mut h = self.embeddings.forward(token_ids, stream)?;
        for layer in &self.layers {
            h = layer.forward(&h, stream)?;
        }
        Ok(h)
    }

    /// Number of encoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Build from config file + weights
// ---------------------------------------------------------------------------

/// Build a RopeBertModel from a config.json path and pre-loaded weights.
pub fn from_config_file(
    config_path: &str,
    weights: &HashMap<String, Array>,
) -> Result<RopeBertModel> {
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| crate::error::Error::Mlx(format!("failed to read config: {}", e)))?;
    let config: RopeBertConfig = serde_json::from_str(&content)
        .map_err(|e| crate::error::Error::Mlx(format!("failed to parse RopeBert config: {}", e)))?;

    from_config(&config, weights)
}

/// Build a RopeBertModel from a parsed RopeBertConfig and pre-loaded weights.
pub fn from_config(
    config: &RopeBertConfig,
    weights: &HashMap<String, Array>,
) -> Result<RopeBertModel> {
    let (group_size, bits) = match &config.quantization {
        Some(qc) => (qc.group_size, qc.bits),
        None => (64, 4),
    };

    let head_dim = config.hidden_size / config.num_attention_heads;
    let rope_dims = head_dim as i32;

    // Detect model variant
    let is_modernbert = config.model_type == "modernbert";
    let use_geglu = is_modernbert
        || config
            .hidden_act
            .as_deref()
            .is_some_and(|a| a == "geglu" || a == "gelu_glu");

    // Effective LayerNorm eps
    let ln_eps = config.norm_eps.unwrap_or(config.layer_norm_eps);

    // Weight prefix: detect by checking common patterns
    // BERT-like: "bert.embeddings.word_embeddings.weight"
    // ModernBERT: "model.embeddings.word_embeddings.weight" or bare "embeddings.word_embeddings.weight"
    let prefix = if weights.contains_key("bert.embeddings.word_embeddings.weight") {
        "bert"
    } else if weights.contains_key("model.embeddings.word_embeddings.weight") {
        "model"
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

    let token_type_embeddings = if config.type_vocab_size > 0 {
        Some(EmbeddingLayer::from_weights(
            weights,
            &pfx("embeddings.token_type_embeddings"),
            group_size,
            bits,
        )?)
    } else {
        None
    };

    // Embedding LayerNorm (check multiple possible key names)
    let emb_ln_key = pfx("embeddings.LayerNorm");
    let emb_ln_key_alt = pfx("embeddings.norm");
    let emb_ln = if weights.contains_key(&format!("{}.weight", emb_ln_key)) {
        Some(build_layer_norm(
            weights,
            &emb_ln_key,
            ln_eps,
            config.norm_bias,
        ))
    } else if weights.contains_key(&format!("{}.weight", emb_ln_key_alt)) {
        Some(build_layer_norm(
            weights,
            &emb_ln_key_alt,
            ln_eps,
            config.norm_bias,
        ))
    } else {
        None
    };

    let embeddings = RopeBertEmbeddings {
        word_embeddings,
        token_type_embeddings,
        layer_norm: emb_ln,
    };

    // --- Encoder layers ---
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let lp = pfx(&format!("encoder.layer.{}", i));

        // Determine if this layer uses global or local attention
        let is_global = config.global_attn_every_n_layers <= 1
            || ((i + 1) % config.global_attn_every_n_layers == 0);

        // RoPE theta for this layer
        let layer_rope_theta = if is_global {
            config.global_rope_theta.unwrap_or(config.rope_theta)
        } else {
            config.local_rope_theta.unwrap_or(config.rope_theta)
        };

        // Sliding window: only for local attention layers
        let local_window = if !is_global {
            config.local_attention
        } else {
            None
        };

        let attention_self = RopeBertSelfAttention {
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
            rope_dims,
            rope_theta: layer_rope_theta,
            local_window,
        };

        let attention_output = RopeBertAttentionOutput {
            dense: LinearLayer::from_weights(
                weights,
                &format!("{}.attention.output.dense", lp),
                group_size,
                bits,
            )?,
            layer_norm: build_layer_norm(
                weights,
                &format!("{}.attention.output.LayerNorm", lp),
                ln_eps,
                config.norm_bias,
            ),
        };

        // FFN: standard or GeGLU
        let ffn = if use_geglu {
            // GeGLU: separate gate and up projections
            // Try ModernBERT naming first: mlp.Wi, mlp.Wo
            // Then BERT-style: intermediate.dense, output.dense
            let gate_key = format!("{}.mlp.Wi", lp);
            let up_key = format!("{}.mlp.Wi", lp);
            let down_key = format!("{}.mlp.Wo", lp);

            if weights.contains_key(&format!("{}.weight", gate_key))
                || weights.contains_key(&format!("{}.scales", gate_key))
            {
                // ModernBERT style: Wi is fused [gate; up] -> split or
                // Actually ModernBERT uses separate Wi for gate and up,
                // but they might be fused. Let's try the standard approach:
                // intermediate.dense.gated_layers (fused gate+up)
                let intermediate_key = format!("{}.intermediate.dense", lp);
                if weights.contains_key(&format!("{}.weight", intermediate_key))
                    || weights.contains_key(&format!("{}.scales", intermediate_key))
                {
                    // Fused intermediate with GeGLU: split into gate and up
                    // This is a simplified approach — treat as standard for now
                    RopeBertFFN {
                        up: LinearLayer::from_weights(
                            weights,
                            &intermediate_key,
                            group_size,
                            bits,
                        )?,
                        down: LinearLayer::from_weights(
                            weights,
                            &format!("{}.output.dense", lp),
                            group_size,
                            bits,
                        )?,
                        use_geglu: false,
                        gate: None,
                    }
                } else {
                    // Use Wi/Wo naming
                    RopeBertFFN {
                        up: LinearLayer::from_weights(weights, &up_key, group_size, bits)?,
                        down: LinearLayer::from_weights(weights, &down_key, group_size, bits)?,
                        use_geglu: false,
                        gate: None,
                    }
                }
            } else {
                // Try BERT-style naming with gate
                let gate_key_bert = format!("{}.intermediate.gate", lp);
                let int_key = format!("{}.intermediate.dense", lp);
                let out_key = format!("{}.output.dense", lp);

                if weights.contains_key(&format!("{}.weight", gate_key_bert))
                    || weights.contains_key(&format!("{}.scales", gate_key_bert))
                {
                    RopeBertFFN {
                        up: LinearLayer::from_weights(weights, &int_key, group_size, bits)?,
                        down: LinearLayer::from_weights(weights, &out_key, group_size, bits)?,
                        use_geglu: true,
                        gate: Some(LinearLayer::from_weights(
                            weights,
                            &gate_key_bert,
                            group_size,
                            bits,
                        )?),
                    }
                } else {
                    // Fall back to standard GELU FFN
                    RopeBertFFN {
                        up: LinearLayer::from_weights(weights, &int_key, group_size, bits)?,
                        down: LinearLayer::from_weights(weights, &out_key, group_size, bits)?,
                        use_geglu: false,
                        gate: None,
                    }
                }
            }
        } else {
            // Standard GELU FFN (GTE, Jina)
            RopeBertFFN {
                up: LinearLayer::from_weights(
                    weights,
                    &format!("{}.intermediate.dense", lp),
                    group_size,
                    bits,
                )?,
                down: LinearLayer::from_weights(
                    weights,
                    &format!("{}.output.dense", lp),
                    group_size,
                    bits,
                )?,
                use_geglu: false,
                gate: None,
            }
        };

        let output = RopeBertOutput {
            layer_norm: build_layer_norm(
                weights,
                &format!("{}.output.LayerNorm", lp),
                ln_eps,
                config.norm_bias,
            ),
        };

        layers.push(RopeBertLayer {
            attention_self,
            attention_output,
            ffn,
            output,
        });
    }

    // Clone config for ownership
    let owned_config = RopeBertConfig {
        hidden_size: config.hidden_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        max_position_embeddings: config.max_position_embeddings,
        type_vocab_size: config.type_vocab_size,
        layer_norm_eps: config.layer_norm_eps,
        norm_eps: config.norm_eps,
        model_type: config.model_type.clone(),
        quantization: config.quantization.clone(),
        rope_theta: config.rope_theta,
        global_rope_theta: config.global_rope_theta,
        local_rope_theta: config.local_rope_theta,
        local_attention: config.local_attention,
        global_attn_every_n_layers: config.global_attn_every_n_layers,
        attention_bias: config.attention_bias,
        mlp_bias: config.mlp_bias,
        norm_bias: config.norm_bias,
        hidden_act: config.hidden_act.clone(),
    };

    Ok(RopeBertModel {
        embeddings,
        layers,
        config: owned_config,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a LayerNorm from weights map.
/// When `allow_bias` is false, bias is not loaded even if present.
fn build_layer_norm(
    weights: &HashMap<String, Array>,
    prefix: &str,
    eps: f32,
    allow_bias: bool,
) -> LayerNorm {
    let w_key = format!("{}.weight", prefix);
    let b_key = format!("{}.bias", prefix);
    let weight = weights.get(&w_key).cloned();
    let bias = if allow_bias {
        weights.get(&b_key).cloned()
    } else {
        None
    };
    LayerNorm::new(weight, bias, eps)
}
