use std::collections::HashMap;

use crate::array::Array;
use crate::error::{Error, Result};
use crate::nn::{Linear, LinearLayer, QuantizedLinear};
use crate::ops;
use crate::stream::Stream;

use super::{Model, bert, rope_bert};

/// Reranker model: encoder + classification head.
pub struct RerankerModel {
    pub model: Model,
    pub classifier: LinearLayer,
    pub hidden_size: usize,
}

impl RerankerModel {
    /// Score a batch of (query, document) pairs.
    /// Input: token_ids [batch, seq_len]
    /// Output: scores [batch]
    pub fn score(&self, token_ids: &Array, stream: &Stream) -> Result<Array> {
        let hidden = match &self.model {
            Model::Bert(m) => m.forward(token_ids, stream)?,
            Model::RopeBert(m) => m.forward(token_ids, stream)?,
            _ => {
                let mut cache = vec![];
                self.model
                    .forward(token_ids, &mut cache, "none", None, stream)?
            }
        };

        // [CLS] token → [batch, hidden_size]
        let shape = hidden.shape();
        let batch = shape[0];
        let hidden_size = shape[2];
        let cls = ops::slice(
            &hidden,
            &[0, 0, 0],
            &[batch, 1, hidden_size],
            &[1, 1, 1],
            stream,
        )?;
        let cls = ops::squeeze_axis(&cls, 1, stream)?;

        // Classification head
        let logits = self.classifier.forward_with_stream(&cls, stream)?;

        let ls = logits.shape();
        if ls.len() == 2 && ls[1] == 1 {
            ops::squeeze_axis(&logits, 1, stream)
        } else {
            Ok(logits)
        }
    }

    pub fn from_bert(config_path: &str, weights: &HashMap<String, Array>) -> Result<Self> {
        let bert = bert::from_config_file(config_path, weights)?;
        let hidden_size = bert.config.hidden_size;
        let classifier = load_classifier(weights)?;
        Ok(Self {
            model: Model::Bert(bert),
            classifier,
            hidden_size,
        })
    }

    pub fn from_rope_bert(config_path: &str, weights: &HashMap<String, Array>) -> Result<Self> {
        let model = rope_bert::from_config_file(config_path, weights)?;
        let hidden_size = model.config.hidden_size;
        let classifier = load_classifier(weights)?;
        Ok(Self {
            model: Model::RopeBert(model),
            classifier,
            hidden_size,
        })
    }
}

fn load_classifier(weights: &HashMap<String, Array>) -> Result<LinearLayer> {
    let prefixes = ["classifier", "classifier.dense", "score"];

    for prefix in &prefixes {
        let w_key = format!("{prefix}.weight");
        if let Some(weight) = weights.get(&w_key) {
            let bias = weights.get(&format!("{prefix}.bias")).cloned();
            let s_key = format!("{prefix}.scales");

            if let Some(scales) = weights.get(&s_key) {
                let biases = weights
                    .get(&format!("{prefix}.biases"))
                    .cloned()
                    .ok_or_else(|| Error::Mlx(format!("missing quantized biases for {prefix}")))?;
                return Ok(LinearLayer::Quantized(QuantizedLinear::new(
                    weight.clone(),
                    scales.clone(),
                    biases,
                    64,
                    4,
                )));
            } else {
                return Ok(LinearLayer::Full(Linear::new(weight.clone(), bias)));
            }
        }
    }

    Err(Error::Mlx(
        "no classifier weights found (tried: classifier, classifier.dense, score)".to_string(),
    ))
}
