use std::collections::HashMap;

use crate::array::Array;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;

use super::Model;
use super::bert;
use super::rope_bert;

// ---------------------------------------------------------------------------
// PoolingStrategy
// ---------------------------------------------------------------------------

/// Pooling strategy for extracting a fixed-size embedding from encoder output.
#[derive(Debug, Clone, Copy)]
pub enum PoolingStrategy {
    /// Use the [CLS] token output (first token).
    Cls,
    /// Mean of all token hidden states.
    Mean,
    /// Use the last token output (for decoder-based embeddings like E5-Mistral).
    LastToken,
}

// ---------------------------------------------------------------------------
// EmbeddingModel
// ---------------------------------------------------------------------------

/// Wrapper around any model for embedding inference.
/// Supports encoder-only (BERT, XLM-RoBERTa) and decoder-only (E5-Mistral).
pub struct EmbeddingModel {
    pub model: Model,
    pub pooling: PoolingStrategy,
    pub normalize: bool,
    pub hidden_size: usize,
}

impl EmbeddingModel {
    /// Encode token IDs into a fixed-size embedding vector.
    ///
    /// Input: `token_ids` of shape [batch, seq_len]
    /// Output: embeddings of shape [batch, hidden_size]
    pub fn encode(&self, token_ids: &Array, stream: &Stream) -> Result<Array> {
        // 1. Forward through model -> [batch, seq_len, hidden_size]
        let hidden = match &self.model {
            Model::Bert(bert) => bert.forward(token_ids, stream)?,
            Model::RopeBert(rb) => rb.forward(token_ids, stream)?,
            _ => {
                // Decoder models: forward returns logits [batch, seq_len, vocab_size]
                // For decoder-based embeddings (E5-Mistral), we pool the logits.
                // Proper implementation would need a forward_hidden() method.
                let mut cache = vec![];
                self.model
                    .forward(token_ids, &mut cache, "none", None, stream)?
            }
        };

        // 2. Pool
        let pooled = self.pool(&hidden, stream)?;

        // 3. L2 normalize if requested
        if self.normalize {
            l2_normalize(&pooled, stream)
        } else {
            Ok(pooled)
        }
    }

    /// Pool hidden states according to the configured strategy.
    fn pool(&self, hidden: &Array, stream: &Stream) -> Result<Array> {
        let shape = hidden.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];

        match self.pooling {
            PoolingStrategy::Cls => {
                let cls = ops::slice(
                    hidden,
                    &[0, 0, 0],
                    &[batch, 1, hidden_size],
                    &[1, 1, 1],
                    stream,
                )?;
                ops::squeeze_axis(&cls, 1, stream)
            }
            PoolingStrategy::Mean => ops::mean(hidden, &[1], false, stream),
            PoolingStrategy::LastToken => {
                let last = ops::slice(
                    hidden,
                    &[0, seq_len - 1, 0],
                    &[batch, seq_len, hidden_size],
                    &[1, 1, 1],
                    stream,
                )?;
                ops::squeeze_axis(&last, 1, stream)
            }
        }
    }

    /// Build an EmbeddingModel from a BERT/XLM-RoBERTa config.
    /// Defaults to Mean pooling + L2 normalization.
    pub fn from_bert(config_path: &str, weights: &HashMap<String, Array>) -> Result<Self> {
        let bert = bert::from_config_file(config_path, weights)?;
        let hidden_size = bert.config.hidden_size;
        Ok(Self {
            model: Model::Bert(bert),
            pooling: PoolingStrategy::Mean,
            normalize: true,
            hidden_size,
        })
    }

    /// Build an EmbeddingModel from a RoPE-BERT config (ModernBERT, GTE, Jina).
    /// Defaults to Mean pooling + L2 normalization.
    pub fn from_rope_bert(config_path: &str, weights: &HashMap<String, Array>) -> Result<Self> {
        let model = rope_bert::from_config_file(config_path, weights)?;
        let hidden_size = model.config.hidden_size;
        Ok(Self {
            model: Model::RopeBert(model),
            pooling: PoolingStrategy::Mean,
            normalize: true,
            hidden_size,
        })
    }

    /// Build an EmbeddingModel from any Model with explicit options.
    pub fn from_model(
        model: Model,
        hidden_size: usize,
        pooling: PoolingStrategy,
        normalize: bool,
    ) -> Self {
        Self {
            model,
            pooling,
            normalize,
            hidden_size,
        }
    }

    /// Hidden size of the underlying model.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// L2 normalize along the last axis: x / ||x||_2
fn l2_normalize(x: &Array, stream: &Stream) -> Result<Array> {
    // ||x||_2 = sqrt(sum(x^2, axis=-1, keepdims=true))
    let squared = ops::square(x, stream)?;
    let sum_sq = ops::sum(&squared, &[-1], true, stream)?;
    let norm = ops::sqrt(&sum_sq, stream)?;

    // Avoid division by zero: clamp norm to a small epsilon
    let eps = Array::from_float(1e-12);
    let safe_norm = ops::maximum(&norm, &eps, stream)?;

    ops::divide(x, &safe_norm, stream)
}
