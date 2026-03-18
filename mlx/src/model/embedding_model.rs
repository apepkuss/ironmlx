use std::collections::HashMap;

use crate::array::Array;
use crate::device::Device;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;

use super::bert::{self, BertModel};

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
    /// Use the last token output (for decoder-based embeddings).
    LastToken,
}

// ---------------------------------------------------------------------------
// EmbeddingModel
// ---------------------------------------------------------------------------

/// Wrapper around an encoder model for embedding inference.
pub struct EmbeddingModel {
    pub model: BertModel,
    pub pooling: PoolingStrategy,
    pub normalize: bool,
}

impl EmbeddingModel {
    /// Encode token IDs into a fixed-size embedding vector.
    ///
    /// Input: `token_ids` of shape [batch, seq_len]
    /// Output: embeddings of shape [batch, hidden_size]
    pub fn encode(&self, token_ids: &Array, stream: &Stream) -> Result<Array> {
        // 1. Forward through encoder -> [batch, seq_len, hidden_size]
        let hidden = self.model.forward(token_ids, stream)?;

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
                // hidden[:, 0, :] -> [batch, hidden_size]
                let cls = ops::slice(
                    hidden,
                    &[0, 0, 0],
                    &[batch, 1, hidden_size],
                    &[1, 1, 1],
                    stream,
                )?;
                ops::squeeze_axis(&cls, 1, stream)
            }
            PoolingStrategy::Mean => {
                // mean(hidden, axis=1) -> [batch, hidden_size]
                ops::mean(hidden, &[1], false, stream)
            }
            PoolingStrategy::LastToken => {
                // hidden[:, -1, :] -> [batch, hidden_size]
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

    /// Build an EmbeddingModel from a config file path and weights.
    ///
    /// Defaults to Mean pooling + L2 normalization (sentence-transformers convention).
    pub fn from_config_file(config_path: &str, weights: &HashMap<String, Array>) -> Result<Self> {
        let model = bert::from_config_file(config_path, weights)?;

        Ok(Self {
            model,
            pooling: PoolingStrategy::Mean,
            normalize: true,
        })
    }

    /// Build with explicit pooling strategy and normalize flag.
    pub fn from_config_file_with_options(
        config_path: &str,
        weights: &HashMap<String, Array>,
        pooling: PoolingStrategy,
        normalize: bool,
    ) -> Result<Self> {
        let model = bert::from_config_file(config_path, weights)?;

        Ok(Self {
            model,
            pooling,
            normalize,
        })
    }

    /// Hidden size of the underlying model.
    pub fn hidden_size(&self) -> usize {
        self.model.config.hidden_size
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
